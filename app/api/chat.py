"""
Chat API — AI chat endpoints with LangGraph-powered multi-turn agent.

Endpoints:
  POST /chat       — Full response (sync)
  POST /chat/stream — SSE streaming response
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import time
from collections import OrderedDict

import redis
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from motor.motor_asyncio import AsyncIOMotorClient

from ..core import agent, llm as llm_provider
from ..models import ChatRequest, ChatResponse, PlanStepResponse
from ..utils import auth, mermaid

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

# ── MongoDB Trace Store (chat-specific) ─────────────────────────────
_mongo_url = os.getenv("MONGODB_URL", "")
if _mongo_url:
    try:
        _mongo_client = AsyncIOMotorClient(_mongo_url)
        _mongo_db = _mongo_client["forge_traces"]
        _traces_collection = _mongo_db["traces"]
        logger.info("Chat: MongoDB tracing ENABLED (database: forge_traces)")
    except Exception as e:
        logger.error("Chat: MongoDB init failed: %s", e)
        _mongo_client = None
        _traces_collection = None
else:
    _mongo_client = None
    _traces_collection = None
    logger.info("Chat: MongoDB tracing disabled (set MONGODB_URL to enable)")


async def _log_trace_to_mongo(
    thread_id: str,
    workspace_id: str,
    user_email: str,
    request_data: dict,
    response_data: dict,
    execution_time_ms: float,
    status: str,
    error: str = None,
):
    """Log a complete trace (request + response) to MongoDB."""
    if _traces_collection is None:
        return

    try:
        trace_doc = {
            "thread_id": thread_id,
            "workspace_id": workspace_id,
            "user_email": user_email,
            "timestamp": datetime.datetime.utcnow(),
            "request": request_data,
            "response": response_data,
            "execution_time_ms": execution_time_ms,
            "status": status,
            "error": error,
        }
        await _traces_collection.insert_one(trace_doc)
        logger.info(
            "[mongo_trace] Logged trace for thread=%s, time=%.1fms, status=%s",
            thread_id, execution_time_ms, status,
        )
    except Exception as e:
        logger.error("[mongo_trace] Failed to log trace: %s", e)


# ── Message serialization (for ConversationStore) ───────────────────

def _serialize_messages(messages: list) -> list[dict]:
    """Serialize LangChain messages to JSON-safe dicts."""
    result = []
    for m in messages:
        if isinstance(m, HumanMessage):
            result.append({"type": "human", "content": m.content})
        elif isinstance(m, AIMessage):
            entry = {"type": "ai", "content": m.content}
            if m.tool_calls:
                entry["tool_calls"] = m.tool_calls
            result.append(entry)
        elif isinstance(m, ToolMessage):
            result.append({
                "type": "tool",
                "content": m.content,
                "tool_call_id": m.tool_call_id,
            })
        elif isinstance(m, SystemMessage):
            result.append({"type": "system", "content": m.content})
    return result


def _deserialize_messages(data: list[dict]) -> list:
    """Deserialize JSON dicts back to LangChain messages."""
    messages = []
    for m in data:
        if m["type"] == "human":
            messages.append(HumanMessage(content=m["content"]))
        elif m["type"] == "ai":
            messages.append(AIMessage(
                content=m["content"],
                tool_calls=m.get("tool_calls", []),
            ))
        elif m["type"] == "tool":
            messages.append(ToolMessage(
                content=m["content"],
                tool_call_id=m["tool_call_id"],
            ))
        elif m["type"] == "system":
            messages.append(SystemMessage(content=m["content"]))
    return messages


# ── ConversationStore (Redis-backed) ─────────────────────────────────

class ConversationStore:
    """Redis-backed conversation state store with in-memory fallback.

    Conversation state survives server restarts when Redis is available.
    Each conversation expires after 24 hours of inactivity.
    """

    CONV_TTL = 86400  # 24 hours
    PREFIX = "conv:"

    def __init__(self, redis_url: str | None = None, max_memory_size: int = 200):
        self._redis: redis.Redis | None = None
        self._memory: OrderedDict[str, dict] = OrderedDict()
        self._max_memory_size = max_memory_size

        if redis_url:
            try:
                self._redis = redis.from_url(redis_url, decode_responses=True)
                self._redis.ping()
                logger.info("ConversationStore: Redis connected at %s", redis_url[:40] + "...")
            except Exception as e:
                logger.warning(
                    "ConversationStore: Redis unavailable (%s), using in-memory fallback", e
                )
                self._redis = None
        else:
            logger.info("ConversationStore: No REDIS_URL set, using in-memory store")

    def get(self, conv_id: str) -> dict | None:
        if self._redis:
            try:
                raw = self._redis.get(f"{self.PREFIX}{conv_id}")
                if raw:
                    data = json.loads(raw)
                    data["messages"] = _deserialize_messages(data.get("messages", []))
                    self._redis.expire(f"{self.PREFIX}{conv_id}", self.CONV_TTL)
                    return data
            except Exception as e:
                logger.warning("Redis get failed for %s: %s", conv_id, e)

        if conv_id in self._memory:
            self._memory.move_to_end(conv_id)
            return self._memory[conv_id]
        return None

    def set(self, conv_id: str, state: dict):
        if conv_id in self._memory:
            self._memory.move_to_end(conv_id)
        self._memory[conv_id] = state
        while len(self._memory) > self._max_memory_size:
            self._memory.popitem(last=False)

        if self._redis:
            try:
                data = dict(state)
                data["messages"] = _serialize_messages(data.get("messages", []))
                self._redis.setex(
                    f"{self.PREFIX}{conv_id}",
                    self.CONV_TTL,
                    json.dumps(data),
                )
            except Exception as e:
                logger.warning("Redis set failed for %s: %s", conv_id, e)

    def delete(self, conv_id: str):
        self._memory.pop(conv_id, None)
        if self._redis:
            try:
                self._redis.delete(f"{self.PREFIX}{conv_id}")
            except Exception:
                pass


_redis_url = os.getenv("REDIS_URL", "")
_conversations = ConversationStore(redis_url=_redis_url if _redis_url else None)


# ── Roo-Code Style: Minimal tool result processing ─────────────────
# We no longer need cross-trace memory or background error analysis.
# The agent's state-based repetition detection handles loops within the task.

def _process_tool_result_for_memory(
    workspace_id: str,
    messages: list,
    res,
) -> None:
    """Placeholder for future task-level persistence (optional)."""
    # In Roo-Code style, we don't track cross-trace failures.
    # All loop detection happens within the current conversation state.
    pass


# ── POST /chat ─────────────────────────────────────────────────────

@router.post("", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest, user: dict = Depends(auth.get_current_user)):
    """
    AI Chat — LangGraph-powered multi-turn coding agent.

    Conversation state (enriched_context, full message history) is tracked
    server-side per conversation_id. The client only needs to send back
    tool_results on continuation turns — no need to echo history.
    """
    t0 = time.monotonic()

    conv_id = req.conversation_id or f"{req.workspace_id}-default"
    has_tool_results = bool(req.tool_results)

    stored = _conversations.get(conv_id)

    plan_steps = []
    current_step = 0
    enriched_question = ""
    enriched_step = 0
    project_profile = ""

    if has_tool_results and stored:
        messages = stored["messages"]
        enriched_context = stored.get("enriched_context", "")
        enriched_question = stored.get("enriched_question", "")
        enriched_step = stored.get("enriched_step", 0)
        project_profile = stored.get("project_profile", "")
        attached_files_dict = stored.get("attached_files", {})
        plan_steps = stored.get("plan_steps", [])
        current_step = stored.get("current_step", 0)

        for res in req.tool_results:
            messages.append(ToolMessage(
                content=res.output,
                tool_call_id=res.call_id,
                status="success" if res.success else "error",
            ))
            _process_tool_result_for_memory(req.workspace_id, messages, res)

        logger.info(
            "[chat] TOOL CONTINUATION conv=%s, restored %d messages, enriched=%d chars, plan_steps=%d, step=%d",
            conv_id, len(messages), len(enriched_context), len(plan_steps), current_step,
        )
    elif stored and req.question:
        messages = stored["messages"]
        enriched_context = stored.get("enriched_context", "")
        enriched_question = stored.get("enriched_question", "")
        enriched_step = stored.get("enriched_step", 0)
        project_profile = stored.get("project_profile", "")
        attached_files_dict = stored.get("attached_files", {})
        plan_steps = stored.get("plan_steps", [])
        current_step = stored.get("current_step", 0)

        messages.append(HumanMessage(content=req.question))

        logger.info(
            "[chat] FOLLOW-UP conv=%s, restored %d messages + new question, enriched=%d chars",
            conv_id, len(messages), len(enriched_context),
        )
    else:
        messages = []
        enriched_context = ""
        attached_files_dict = {}

        if req.history:
            for m in req.history:
                if m["type"] == "human":
                    messages.append(HumanMessage(content=m["content"]))
                elif m["type"] == "ai":
                    tool_calls = m.get("tool_calls", [])
                    messages.append(AIMessage(content=m["content"], tool_calls=tool_calls))
                elif m["type"] == "tool":
                    messages.append(ToolMessage(content=m["content"], tool_call_id=m["tool_call_id"]))

        if req.question:
            messages.append(HumanMessage(content=req.question))

        if req.tool_results:
            for res in req.tool_results:
                messages.append(ToolMessage(
                    content=res.output,
                    tool_call_id=res.call_id,
                    status="success" if res.success else "error",
                ))
                _process_tool_result_for_memory(req.workspace_id, messages, res)

        logger.info("[chat] NEW conv=%s, %d messages", conv_id, len(messages))

    attached_files_dict_new = {}
    if req.attached_files:
        for af in req.attached_files:
            attached_files_dict_new[af.path] = af.content
    if stored and has_tool_results:
        attached_files_dict = {**attached_files_dict, **attached_files_dict_new}
    else:
        attached_files_dict = attached_files_dict_new

    attached_images_list = []
    if req.attached_images:
        attached_images_list = [
            {"filename": img.filename, "data": img.data, "mime_type": img.mime_type}
            for img in req.attached_images
        ]

    try:
        turn_number = 1
        if stored:
            turn_number = sum(1 for m in messages if isinstance(m, HumanMessage))

        user_email = user.get("email", "anonymous") if user else "anonymous"
        user_name = user.get("name", "") if user else ""

        config = {
            "configurable": {"thread_id": conv_id},
            "run_name": f"forge-chat:{req.workspace_id}:turn-{turn_number}",
            "tags": [
                f"workspace:{req.workspace_id}",
                f"user:{user_email}",
                "continuation" if has_tool_results else ("follow-up" if stored else "new-conversation"),
            ],
            "metadata": {
                "conversation_id": conv_id,
                "session_id": conv_id,
                "workspace_id": req.workspace_id,
                "user_email": user_email,
                "user_name": user_name,
                "turn_number": turn_number,
                "is_continuation": has_tool_results,
                "question": (req.question or "")[:200],
                "num_messages": len(messages),
                "num_attached_files": len(attached_files_dict),
                "enriched_context_chars": len(enriched_context),
            },
        }

        logger.info(
            "[chat] Invoking agent for workspace=%s conv=%s with %d messages (tool_cont=%s, stored=%s, enriched=%d chars)",
            req.workspace_id, conv_id, len(messages), has_tool_results, bool(stored), len(enriched_context),
        )

        result = await agent.forge_agent.ainvoke(
            {
                "messages": messages,
                "workspace_id": req.workspace_id,
                "attached_files": attached_files_dict,
                "attached_images": attached_images_list,
                "enriched_context": enriched_context,
                "enriched_question": enriched_question,
                "enriched_step": enriched_step,
                "project_profile": project_profile,
                "plan_steps": plan_steps,
                "current_step": current_step,
            },
            config=config,
        )

        enriched_ctx = result.get("enriched_context", "")
        result_enriched_question = result.get("enriched_question", "")
        result_enriched_step = result.get("enriched_step", 0)
        result_project_profile = result.get("project_profile", "")
        final_messages = result["messages"]
        last_message = final_messages[-1]

        result_plan_steps = result.get("plan_steps", [])
        result_current_step = result.get("current_step", 0)

        logger.info(
            "[chat] Agent returned, enriched_context=%d, messages=%d, plan_steps=%d, step=%d",
            len(enriched_ctx), len(final_messages), len(result_plan_steps), result_current_step,
        )

        status = "done"
        tool_calls = None
        answer = None

        last_ai = None
        for msg in reversed(final_messages):
            if isinstance(msg, AIMessage):
                last_ai = msg
                break

        if last_ai and last_ai.tool_calls:
            responded_ids = {m.tool_call_id for m in final_messages if isinstance(m, ToolMessage)}
            pending_ide_calls = [
                tc for tc in last_ai.tool_calls
                if tc["name"] in agent.IDE_TOOL_NAMES and tc.get("id") not in responded_ids
            ]

            if pending_ide_calls:
                status = "requires_action"
                tool_calls = pending_ide_calls
            else:
                answer = last_ai.content if last_ai.content else last_message.content
        else:
            answer = last_message.content

        _conversations.set(conv_id, {
            "messages": final_messages,
            "enriched_context": enriched_ctx,
            "enriched_question": result_enriched_question,
            "enriched_step": result_enriched_step,
            "project_profile": result_project_profile,
            "attached_files": attached_files_dict,
            "plan_steps": result_plan_steps,
            "current_step": result_current_step,
        })

        serialized_history = []
        for m in final_messages:
            if isinstance(m, HumanMessage):
                m_type = "human"
            elif isinstance(m, AIMessage):
                m_type = "ai"
            elif isinstance(m, ToolMessage):
                m_type = "tool"
            elif isinstance(m, SystemMessage):
                m_type = "system"
            else:
                continue
            entry = {"type": m_type, "content": m.content}
            if m_type == "ai" and m.tool_calls:
                entry["tool_calls"] = m.tool_calls
            if m_type == "tool":
                entry["tool_call_id"] = m.tool_call_id
            serialized_history.append(entry)

        elapsed = (time.monotonic() - t0) * 1000

        if answer:
            answer = await mermaid.render_mermaid_blocks(answer)

        plan_step_responses = None
        if result_plan_steps:
            plan_step_responses = [
                PlanStepResponse(
                    number=s["number"],
                    description=s["description"],
                    status=s["status"],
                )
                for s in result_plan_steps
            ]

        serialized_tool_results = None
        if req.tool_results:
            try:
                serialized_tool_results = [
                    {"call_id": tr.call_id, "output": tr.output[:500], "success": tr.success}
                    for tr in req.tool_results
                ]
            except Exception:
                serialized_tool_results = str(req.tool_results)[:500]

        await _log_trace_to_mongo(
            thread_id=conv_id,
            workspace_id=req.workspace_id,
            user_email=user.get("email", "unknown") if user else "unknown",
            request_data={
                "question": req.question,
                "tool_results": serialized_tool_results,
                "attached_files": len(attached_files_dict),
                "attached_images": len(attached_images_list),
                "is_continuation": has_tool_results,
            },
            response_data={
                "answer": answer,
                "tool_calls": [
                    {"name": tc.get("name"), "args": str(tc.get("args", ""))[:300], "id": tc.get("id")}
                    for tc in (tool_calls or [])
                ] if tool_calls else None,
                "plan_steps": [dict(s) for s in result_plan_steps] if result_plan_steps else None,
                "current_step": result_current_step,
                "message_count": len(final_messages),
                "messages": serialized_history,
            },
            execution_time_ms=elapsed,
            status=status,
        )

        return ChatResponse(
            answer=answer,
            tool_calls=tool_calls,
            history=serialized_history,
            status=status,
            total_time_ms=round(elapsed, 1),
            plan_steps=plan_step_responses,
            current_step=result_current_step if result_plan_steps else None,
        )

    except Exception as e:
        error_msg = str(e)[:500]
        await _log_trace_to_mongo(
            thread_id=conv_id,
            workspace_id=req.workspace_id,
            user_email=user.get("email", "unknown") if user else "unknown",
            request_data={
                "question": req.question,
                "tool_results": req.tool_results,
            },
            response_data={},
            execution_time_ms=(time.monotonic() - t0) * 1000,
            status="error",
            error=error_msg,
        )
        logger.error("Agent execution failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)[:200]}")


# ── POST /chat/stream (SSE) ─────────────────────────────────────────

@router.post("/stream")
async def chat_stream_endpoint(req: ChatRequest, user: dict = Depends(auth.get_current_user)):
    """
    AI Chat with Server-Sent Events (SSE) streaming.

    Provides real-time visibility into agent thinking:
    - thinking: Agent reasoning steps
    - tool_start/tool_end: Server-side tool execution
    - text_delta: Incremental text output
    - requires_action: IDE needs to execute tools
    - done: Completion
    - error: Failure
    """

    async def event_generator():
        t0 = time.monotonic()
        conv_id = req.conversation_id or f"{req.workspace_id}-default"
        try:
            has_tool_results = bool(req.tool_results)

            stored = _conversations.get(conv_id)
            plan_steps = []
            current_step = 0
            enriched_question = ""
            enriched_step = 0
            project_profile = ""

            if has_tool_results and stored:
                messages = stored["messages"]
                enriched_context = stored.get("enriched_context", "")
                enriched_question = stored.get("enriched_question", "")
                enriched_step = stored.get("enriched_step", 0)
                project_profile = stored.get("project_profile", "")
                attached_files_dict = stored.get("attached_files", {})
                plan_steps = stored.get("plan_steps", [])
                current_step = stored.get("current_step", 0)

                for res in req.tool_results:
                    messages.append(ToolMessage(
                        content=res.output,
                        tool_call_id=res.call_id,
                        status="success" if res.success else "error",
                    ))
            elif stored and req.question:
                messages = stored["messages"]
                enriched_context = stored.get("enriched_context", "")
                enriched_question = stored.get("enriched_question", "")
                enriched_step = stored.get("enriched_step", 0)
                project_profile = stored.get("project_profile", "")
                attached_files_dict = stored.get("attached_files", {})
                plan_steps = stored.get("plan_steps", [])
                current_step = stored.get("current_step", 0)
                messages.append(HumanMessage(content=req.question))
            else:
                messages = []
                enriched_context = ""
                attached_files_dict = {}

                if req.history:
                    for m in req.history:
                        if m["type"] == "human":
                            messages.append(HumanMessage(content=m["content"]))
                        elif m["type"] == "ai":
                            tool_calls = m.get("tool_calls", [])
                            messages.append(AIMessage(content=m["content"], tool_calls=tool_calls))
                        elif m["type"] == "tool":
                            messages.append(ToolMessage(content=m["content"], tool_call_id=m["tool_call_id"]))

                if req.question:
                    messages.append(HumanMessage(content=req.question))

                if req.tool_results:
                    for res in req.tool_results:
                        messages.append(ToolMessage(
                            content=res.output,
                            tool_call_id=res.call_id,
                            status="success" if res.success else "error",
                        ))

            attached_files_dict_new = {}
            if req.attached_files:
                for af in req.attached_files:
                    attached_files_dict_new[af.path] = af.content
            if stored and has_tool_results:
                attached_files_dict = {**attached_files_dict, **attached_files_dict_new}
            else:
                attached_files_dict = attached_files_dict_new

            turn_number = 1
            if stored:
                turn_number = sum(1 for m in messages if isinstance(m, HumanMessage))

            user_email = user.get("email", "anonymous") if user else "anonymous"
            user_name = user.get("name", "") if user else ""

            config = {
                "configurable": {"thread_id": conv_id},
                "run_name": f"forge-chat:{req.workspace_id}:turn-{turn_number}",
                "tags": [
                    f"workspace:{req.workspace_id}",
                    f"user:{user_email}",
                    "continuation" if has_tool_results else ("follow-up" if stored else "new-conversation"),
                ],
                "metadata": {
                    "conversation_id": conv_id,
                    "session_id": conv_id,
                    "workspace_id": req.workspace_id,
                    "user_email": user_email,
                    "user_name": user_name,
                    "turn_number": turn_number,
                    "is_continuation": has_tool_results,
                    "question": (req.question or "")[:200],
                    "num_messages": len(messages),
                    "num_attached_files": len(attached_files_dict),
                    "enriched_context_chars": len(enriched_context),
                },
            }

            yield f"event: thinking\ndata: {json.dumps({'step_type': 'start', 'message': 'Processing request...', 'detail': ''})}\n\n"

            result = await agent.forge_agent.ainvoke(
                {
                    "messages": messages,
                    "workspace_id": req.workspace_id,
                    "attached_files": attached_files_dict,
                    "enriched_context": enriched_context,
                    "enriched_question": enriched_question,
                    "enriched_step": enriched_step,
                    "project_profile": project_profile,
                    "plan_steps": plan_steps,
                    "current_step": current_step,
                },
                config=config,
            )

            enriched_ctx = result.get("enriched_context", "")
            result_enriched_question = result.get("enriched_question", "")
            result_enriched_step = result.get("enriched_step", 0)
            result_project_profile = result.get("project_profile", "")
            final_messages = result["messages"]
            last_message = final_messages[-1]

            result_plan_steps = result.get("plan_steps", [])
            result_current_step = result.get("current_step", 0)

            if result_plan_steps:
                yield f"event: plan\ndata: {json.dumps({'steps': result_plan_steps, 'current_step': result_current_step})}\n\n"

            last_ai = None
            for msg in reversed(final_messages):
                if isinstance(msg, AIMessage):
                    last_ai = msg
                    break

            if last_ai and last_ai.tool_calls:
                responded_ids = {m.tool_call_id for m in final_messages if isinstance(m, ToolMessage)}
                pending_ide_calls = [
                    tc for tc in last_ai.tool_calls
                    if tc["name"] in agent.IDE_TOOL_NAMES and tc.get("id") not in responded_ids
                ]

                if pending_ide_calls:
                    tool_calls_data = [
                        {"id": tc["id"], "name": tc["name"], "args": tc["args"]}
                        for tc in pending_ide_calls
                    ]
                    yield f"event: requires_action\ndata: {json.dumps({'tool_calls': tool_calls_data})}\n\n"
                else:
                    text = last_ai.content if last_ai.content else (last_message.content if last_message.content else "")
                    if text:
                        yield f"event: text_delta\ndata: {json.dumps({'text': text})}\n\n"
            else:
                if last_message.content:
                    content = await mermaid.render_mermaid_blocks(last_message.content)
                    yield f"event: text_delta\ndata: {json.dumps({'text': content})}\n\n"

            _conversations.set(conv_id, {
                "messages": final_messages,
                "enriched_context": enriched_ctx,
                "enriched_question": result_enriched_question,
                "enriched_step": result_enriched_step,
                "project_profile": result_project_profile,
                "attached_files": attached_files_dict,
                "plan_steps": result_plan_steps,
                "current_step": result_current_step,
            })

            answer = None
            if isinstance(last_message, AIMessage):
                if not last_message.tool_calls or not any(
                    tc["name"] in agent.IDE_TOOL_NAMES for tc in last_message.tool_calls
                ):
                    answer = last_message.content

            elapsed = (time.monotonic() - t0) * 1000

            await _log_trace_to_mongo(
                thread_id=conv_id,
                workspace_id=req.workspace_id,
                user_email="stream-user",
                request_data={
                    "question": req.question,
                    "tool_results": req.tool_results,
                },
                response_data={
                    "answer": answer,
                    "plan_steps": [dict(s) for s in result_plan_steps] if result_plan_steps else None,
                    "current_step": result_current_step,
                },
                execution_time_ms=elapsed,
                status="done",
            )

            yield f"event: done\ndata: {json.dumps({'answer': answer, 'total_time_ms': round(elapsed, 1)})}\n\n"

        except Exception as e:
            error_msg = str(e)[:500]
            await _log_trace_to_mongo(
                thread_id=conv_id,
                workspace_id=req.workspace_id,
                user_email="stream-user",
                request_data={"question": req.question},
                response_data={},
                execution_time_ms=(time.monotonic() - t0) * 1000,
                status="error",
                error=error_msg,
            )
            logger.error("Streaming agent execution failed: %s", e, exc_info=True)
            yield f"event: error\ndata: {json.dumps({'error': str(e)[:200]})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
