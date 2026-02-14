"""
LangGraph Agent Orchestrator for Forge.

Architecture Goal: Make a small model (Groq Kimi-K2) match Claude Opus performance
by providing it with "perfect context" through pre-enrichment.

The flow:
  1. User sends question + optional attached files
  2. BEFORE calling LLM, we gather:
     - Semantic search results from pgvector
     - Call chain traces for mentioned symbols  
     - Impact analysis if editing
     - Attached file contents from IDE
  3. Build a "perfect prompt" with all this context
  4. LLM reasons + executes with full context and tools
  5. For complex tasks, agent uses create_plan/update_plan tools
     to manage a structured plan (shown live in the IDE)
  6. For IDE tools (file ops), pause and return to IDE for execution
  7. Agent can use lookup_documentation tool for library docs on-demand
"""

from typing import Annotated, Literal
from typing_extensions import TypedDict
import json
import logging
import os
import re
import httpx
from datetime import datetime, timezone

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool

from ..storage import store
from . import embeddings, chat as chat_utils
from . import llm as llm_provider
from ..intelligence.phase1 import workspace_memory as ws_memory

# Intelligence System - Phase 1, 2, 3 (Simplified Roo-Code Style)
ENABLE_PHASE_3 = False
# (Old imports removed)

# Documentation API configuration
CONTEXT7_API_URL = "https://mcp.context7.com/mcp"
DEVDOCS_API_URL = "https://devdocs.io"
CONTEXT7_API_KEY = os.getenv("CONTEXT7_API_KEY", "")

logger = logging.getLogger(__name__)

# ── Configuration: Enable/Disable Intelligence Phases ──────────────
ENABLE_PHASE_1 = os.getenv("ENABLE_PHASE_1", "true").lower() == "true"  # Persistent memory
ENABLE_PHASE_2 = os.getenv("ENABLE_PHASE_2", "true").lower() == "true"  # LLM-powered intelligence

logger.info(
    "[config] Intelligence phases: Phase1=%s, Phase2=%s",
    ENABLE_PHASE_1, ENABLE_PHASE_2
)

# ── System Prompt ──────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert software engineer in Forge IDE. Execute tasks using tools — never just describe what you'd do.

## Rules

1. **Use tools.** Don't say "I would use X" — call X.
2. **Read before editing.** Always `read_file` before `replace_in_file`. The `old_str` must match exactly.
3. **Search smart.** Use `codebase_search` for meaning, `grep` for exact text. Never `execute_command` with grep/find.
4. **Don't loop.** If a command fails twice, use `lookup_documentation` or try a different approach. Never retry 3+ times.
5. **Rename safely.** Prefer `lsp_rename` over manual find/replace — it's atomic and cross-file.
6. **Plan adaptively.** Use `create_plan` for 3+ step tasks. Update progress with `update_plan`. If things aren't working (2+ failures), use `replan` to revise your strategy. Use `add_plan_step`/`remove_plan_step` to adjust scope as you learn more.
7. **Verify your work.** After edits: `diagnostics` on changed files → `find_symbol_references` on changed symbols → build/test command. Not done until checks pass.
8. **Dev servers and long-running processes:** NEVER use `execute_command` for servers or watchers (npm run dev, cargo watch, etc.) — it blocks until timeout. Instead:
   - `execute_background(command, label)` to start the process
   - `wait_for_port(port, timeout, http_check=True)` to wait until it's ready
   - `read_process_output(pid)` to check logs if something goes wrong
   Install-then-run commands should be split: `execute_command("npm install")` first, then `execute_background("npm run dev")`.
9. **Docker auto-start:** If a Docker command fails with "Cannot connect to the Docker daemon":
   - On macOS: `execute_command("open -a Docker")` then wait 10 seconds and retry
   - On Linux: `execute_command("sudo systemctl start docker")` then retry
   After starting Docker, wait for it to be ready before retrying the original command.

## Tool Selection Guide

- **Architecture First**: ALWAYS start with `get_architecture_map()` to understand the system structure before making changes or searching.
- **Navigation**: Use `lsp_go_to_definition` to jump to code and `lsp_hover` to understand types/docs. This is 100% accurate, unlike search.
- **Deep Understanding**: Use `trace_call_chain` to see the flow of data (who calls this, what does this call).
- **Refactoring & Impact**: Use `impact_analysis` BEFORE changing a symbol to see what might break. Use `lsp_rename` for renaming.
- **Search - Architecture Aware**: Use `codebase_search` with `component_focus` to search within specific architectural components:
  - `component_focus="api_endpoints"` - Search only API routes and handlers
  - `component_focus="state_management"` - Search only React Context and global state
  - `component_focus="feature_components"` - Search only business logic components
  - `component_focus="ui_components"` - Search only reusable UI components
- **Search - Targeted**: Use specialized search for focused results:
  - `search_functions` - Find implementation logic, algorithms, business rules
  - `search_classes` - Find data structures, models, type definitions  
  - `search_constants` - Find configuration, limits, API endpoints, defaults
  - `search_files` - Find modules, understand high-level architecture
- **Search - General**: Use `codebase_search` for broad "how does X work" questions and `grep` for exact strings.
- **Diagnostics**: Always run `diagnostics` after editing to catch syntax/type errors immediately.

## Efficiency

**Batch your tool calls.** If you need to read multiple files, perform multiple searches, or run multiple commands, call all of them in a single turn. The system will execute them in parallel where possible to minimize latency. Do not wait for one result before calling the next if they are independent."""


# ── Master Planning Prompt (used when Claude is called for planning) ──
# This prompt extracts maximum value from expensive Claude calls by asking
# for deep analysis, comprehensive planning, and upfront risk identification.

MASTER_PLANNING_PROMPT = """## Architect Mode
You are in planning mode. Your goal is to analyze the task and the pre-gathered context to create a surgical execution strategy.

### Planning Principles:
1. **Context First**: Use the "Pre-gathered Context" above as your primary source. Reference specific file paths and patterns found there.
2. **Atomic Steps**: Create steps that are self-contained and verifiable (e.g., "Run pytest tests/auth.py" instead of "Verify it works").
3. **Risk Analysis**: Identify potential breaking changes, side effects, or conflict risks before listing steps.
4. **Worker-Ready**: Write steps for a less-capable execution model. Be explicit with file paths, function names, and exact logic.

### Your Decision:
- **Simple Task** (typos, single-file reads, simple explanations) → **Execute immediately** using tools. Do NOT call `create_plan`.
- **Complex Task** (refactors, new features, multi-file changes) → **Call `create_plan`** with a structured checklist.

Think deeply, identify risks, then execute or plan."""


# ── Replan Prompt (used when Claude is called to recover from stuck state) ──

REPLAN_PROMPT = """## Replanning Mode
The current approach failed. Analyze the failures and create a revised strategy.

### Analysis:
1. **Identify Root Cause**: Why did it fail? (e.g., wrong file, dependency issue, test failure).
2. **Preserve Progress**: What work is still valid?

### Strategy:
Use `replan(reason, new_steps, keep_completed=True)` to:
- Explain the failure briefly in `reason`.
- Provide surgical `new_steps` that avoid the previous pitfalls.

Diagnose, then replan."""



# ── State Definition ──────────────────────────────────────────────

class PlanStep(TypedDict):
    """A single step in the agent's execution plan."""
    number: int
    description: str
    status: str  # "pending", "in_progress", "done", "failed"


class AgentState(TypedDict):
    """The state of the agent graph."""
    messages: Annotated[list[BaseMessage], add_messages]
    workspace_id: str
    # Pre-enriched context (set by first node)
    enriched_context: str
    # The question we enriched for (used to detect topic shifts)
    enriched_question: str
    # Which plan step we last enriched for (0 = no step-specific enrichment)
    enriched_step: int
    # Project profile: tech stack, versions, warnings (built once per workspace)
    project_profile: str
    # Attached files from IDE (live file contents)
    attached_files: dict[str, str]  # path -> content
    # Attached images from user (paste/screenshot) — list of {filename, data, mime_type}
    attached_images: list[dict]
    # ── Plan state (managed by create_plan/update_plan/discard_plan tools) ──
    plan_steps: list[PlanStep]
    # 1-indexed step the agent is currently executing (0 = no plan)
    current_step: int
    # ── Roo-Code Style Mechanisms ──
    consecutive_mistake_count: int
    last_tool_call: str | None
    repetition_count: int


# ── Tool Definitions (Cloud-side) ────────────────────────────────

@tool
async def codebase_search(query: str, state: Annotated[AgentState, InjectedState], component_focus: str = None) -> str:
    """
    Intelligent semantic code search with architectural awareness.
    
    Args:
        query: What you're looking for (e.g., "authentication logic", "database queries", "error handling")
        component_focus: Optional component to focus on (api_endpoints, state_management, feature_components, etc.)
    
    Use for: "how does authentication work?", "where is the database connection handled?", "find examples of error handling".
    Returns relevant code snippets with architectural context and file paths.
    """
    workspace_id = state['workspace_id']
    logger.info("codebase_search: query=%s, workspace=%s, component_focus=%s", query, workspace_id, component_focus)
    
    # Get architectural context first
    arch_context = ""
    if not component_focus:
        # Get architecture overview to understand what components exist
        arch_map = await store.get_project_map(workspace_id=workspace_id)
        components = [n for n in arch_map.get("nodes", []) if n.get("kind") == "component"]
        if components:
            arch_context = "🏗️ Available Components:\n"
            for comp in components:
                arch_context += f"- {comp['name']}: {comp.get('description', 'No description')}\n"
            arch_context += "\n"
    
    # Perform semantic search
    query_emb = await embeddings.embed_query(query)
    results = await store.vector_search(workspace_id, query_emb, top_k=12)  # Get more results for better context
    logger.info("codebase_search: got %d results", len(results))
    
    if not results:
        return f"{arch_context}No results found for this query. Try a different search term or use get_architecture_map() to explore the codebase structure."
    
    # Filter results by component if specified
    filtered_results = []
    if component_focus:
        # Get files in the focused component
        comp_map = await store.get_project_map(workspace_id=workspace_id, focus_path=component_focus)
        component_files = {n["file_path"] for n in comp_map.get("nodes", []) if n.get("kind") == "file"}
        
        for r in results:
            if r["symbol"].get("file_path") in component_files:
                filtered_results.append(r)
        
        if not filtered_results:
            return f"{arch_context}No results found in component '{component_focus}'. Try a broader search or different component."
    else:
        filtered_results = results
    
    # Group results by architectural component for better organization
    component_groups = {}
    for r in filtered_results:
        file_path = r["symbol"].get("file_path", "")
        
        # Determine which component this file belongs to
        if file_path.startswith("pages/api/"):
            comp = "API Endpoints"
        elif file_path.startswith("pages/"):
            comp = "Pages & Routes"
        elif file_path.startswith("components/ui/"):
            comp = "UI Components"
        elif file_path.startswith("components/"):
            comp = "Feature Components"
        elif file_path.startswith("context/"):
            comp = "State Management"
        elif file_path.startswith(("lib/", "utils/")):
            comp = "Utilities"
        else:
            comp = "Configuration"
        
        if comp not in component_groups:
            component_groups[comp] = []
        component_groups[comp].append(r)
    
    # Build organized context
    content_parts = [arch_context] if arch_context else []
    
    for comp_name, comp_results in component_groups.items():
        content_parts.append(f"📦 {comp_name}:")
        content_parts.append("-" * 30)
        
        flat_results = [r["symbol"] for r in comp_results[:4]]  # Limit per component
        comp_content = chat_utils.build_context_from_results(flat_results)
        content_parts.append(comp_content)
        content_parts.append("")
    
    final_content = "\n".join(content_parts)
    logger.info("codebase_search: built organized context len=%d", len(final_content))
    
    return final_content


@tool
async def search_functions(query: str, state: Annotated[AgentState, InjectedState]) -> str:
    """
    Search only functions and methods for implementation logic.
    Use for: "how is X implemented?", "find the logic for Y", "what functions handle Z?".
    More focused than codebase_search when you specifically need executable code.
    """
    workspace_id = state['workspace_id']
    logger.info("search_functions: query=%s, workspace=%s", query, workspace_id)
    query_emb = await embeddings.embed_query(query)
    results = await store.vector_search_by_type(workspace_id, query_emb, ['function', 'method'], top_k=8)
    logger.info("search_functions: got %d results", len(results))
    
    if results:
        content = f"🔧 **Function Search Results for '{query}':**\n\n"
        for i, result in enumerate(results, 1):
            sym = result["symbol"]
            score = result["score"]
            content += f"**{i}. {sym['name']}** ({sym['kind']}) - Score: {score:.3f}\n"
            content += f"   📁 `{sym['file_path']}:{sym['start_line']}-{sym['end_line']}`\n"
            if sym.get('signature'):
                content += f"   📝 `{sym['signature']}`\n"
            if sym.get('content'):
                preview = sym['content'][:200].replace('\n', ' ')
                content += f"   💡 {preview}{'...' if len(sym['content']) > 200 else ''}\n"
            content += "\n"
        return content
    else:
        return "No functions or methods found for this query. Try a broader search term or use codebase_search for general results."


@tool
async def search_classes(query: str, state: Annotated[AgentState, InjectedState]) -> str:
    """
    Search only classes, structs, and type definitions for data structures.
    Use for: "what data structures exist?", "find the User class", "what types are defined?".
    Perfect for understanding the data model and object hierarchies.
    """
    workspace_id = state['workspace_id']
    logger.info("search_classes: query=%s, workspace=%s", query, workspace_id)
    query_emb = await embeddings.embed_query(query)
    results = await store.vector_search_by_type(workspace_id, query_emb, ['class', 'struct', 'type'], top_k=8)
    logger.info("search_classes: got %d results", len(results))
    
    if results:
        content = f"🏛️ **Class/Type Search Results for '{query}':**\n\n"
        for i, result in enumerate(results, 1):
            sym = result["symbol"]
            score = result["score"]
            content += f"**{i}. {sym['name']}** ({sym['kind']}) - Score: {score:.3f}\n"
            content += f"   📁 `{sym['file_path']}:{sym['start_line']}-{sym['end_line']}`\n"
            if sym.get('signature'):
                content += f"   📝 `{sym['signature']}`\n"
            if sym.get('parent'):
                content += f"   👆 Parent: `{sym['parent']}`\n"
            if sym.get('content'):
                preview = sym['content'][:200].replace('\n', ' ')
                content += f"   💡 {preview}{'...' if len(sym['content']) > 200 else ''}\n"
            content += "\n"
        return content
    else:
        return "No classes, structs, or types found for this query. Try a broader search term or use codebase_search for general results."


@tool
async def search_constants(query: str, state: Annotated[AgentState, InjectedState]) -> str:
    """
    Search only constants and configuration values.
    Use for: "what timeouts are configured?", "find API endpoints", "what limits are set?".
    Great for finding configuration, limits, defaults, and static values.
    """
    workspace_id = state['workspace_id']
    logger.info("search_constants: query=%s, workspace=%s", query, workspace_id)
    query_emb = await embeddings.embed_query(query)
    results = await store.vector_search_by_type(workspace_id, query_emb, ['constant', 'const', 'static'], top_k=8)
    logger.info("search_constants: got %d results", len(results))
    
    if results:
        content = f"⚙️ **Constants Search Results for '{query}':**\n\n"
        for i, result in enumerate(results, 1):
            sym = result["symbol"]
            score = result["score"]
            content += f"**{i}. {sym['name']}** ({sym['kind']}) - Score: {score:.3f}\n"
            content += f"   📁 `{sym['file_path']}:{sym['start_line']}-{sym['end_line']}`\n"
            if sym.get('signature'):
                content += f"   📝 `{sym['signature']}`\n"
            if sym.get('content'):
                preview = sym['content'][:150].replace('\n', ' ')
                content += f"   💡 {preview}{'...' if len(sym['content']) > 150 else ''}\n"
            content += "\n"
        return content
    else:
        return "No constants or configuration values found for this query. Try a broader search term or use codebase_search for general results."


@tool
async def search_files(query: str, state: Annotated[AgentState, InjectedState]) -> str:
    """
    Search only file-level symbols to understand module structure.
    Use for: "what modules exist?", "find the auth module", "what files handle payments?".
    Perfect for understanding the high-level architecture and finding the right modules.
    """
    workspace_id = state['workspace_id']
    logger.info("search_files: query=%s, workspace=%s", query, workspace_id)
    query_emb = await embeddings.embed_query(query)
    results = await store.vector_search_by_type(workspace_id, query_emb, ['file'], top_k=8)
    logger.info("search_files: got %d results", len(results))
    
    if results:
        content = f"📄 **File/Module Search Results for '{query}':**\n\n"
        for i, result in enumerate(results, 1):
            sym = result["symbol"]
            score = result["score"]
            content += f"**{i}. {sym['name']}** ({sym['kind']}) - Score: {score:.3f}\n"
            content += f"   📁 `{sym['file_path']}`\n"
            if sym.get('content'):
                preview = sym['content'][:200].replace('\n', ' ')
                content += f"   💡 {preview}{'...' if len(sym['content']) > 200 else ''}\n"
            content += "\n"
        return content
    else:
        return "No files or modules found for this query. Try a broader search term or use codebase_search for general results."


@tool  
async def trace_call_chain(symbol_name: str, state: Annotated[AgentState, InjectedState], direction: str = "both") -> str:
    """
    Deep call-chain traversal to understand the flow of execution.
    Direction: 'upstream' (who calls this), 'downstream' (what does this call), 'both'.
    Use this to: understand how data flows through the system or find the root cause of a bug by tracing logic.
    Returns the execution flow as a graph of symbol relationships.
    """
    workspace_id = state['workspace_id']
    result = await store.trace_call_chain(workspace_id, symbol_name, direction=direction, max_depth=3)
    return json.dumps(result, indent=2)


@tool
async def impact_analysis(symbol_name: str, state: Annotated[AgentState, InjectedState]) -> str:
    """
    Blast radius analysis - find what code is affected if you change this symbol.
    Use this BEFORE editing a function or class to identify potential breaking changes in other modules.
    Returns all callers and dependent code across the entire workspace.
    """
    workspace_id = state['workspace_id']
    result = await store.impact_analysis(workspace_id, symbol_name, max_depth=3)
    return json.dumps(result, indent=2)


@tool
async def lookup_documentation(library: str, query: str) -> str:
    """
    Look up documentation for a library/framework.
    Uses Context7 and DevDocs.io to find official documentation.
    
    Args:
        library: The library/framework name (e.g., "react", "fastapi", "langchain", "pandas")
        query: What you want to know (e.g., "how to use hooks", "async endpoints", "dataframe groupby")
    
    Returns:
        Relevant documentation snippets from official sources.
    """
    return await _fetch_documentation(library, query)


@tool
async def get_architecture_map(state: Annotated[AgentState, InjectedState], focus_path: str = None, focus_symbol: str = None) -> str:
    """
    Get the hierarchical architecture map of the codebase.
    
    Level 0 (no focus): Shows high-level architectural components (API, UI, State, etc.)
    Level 1 (focus_path=component): Shows files within that component  
    Level 2 (focus_path=file): Shows symbols/methods within that file
    Level 3 (focus_symbol=method): Shows call graph for that method
    
    Use this to understand system architecture before making changes or to navigate large codebases.
    Perfect for: "show me the architecture", "what components exist?", "how is this organized?"
    """
    workspace_id = state['workspace_id']
    logger.info("get_architecture_map: workspace=%s, focus_path=%s, focus_symbol=%s", workspace_id, focus_path, focus_symbol)
    
    result = await store.get_project_map(
        workspace_id=workspace_id,
        focus_path=focus_path, 
        focus_symbol=focus_symbol,
        depth=1
    )
    
    # Format the result for the agent
    nodes = result.get("nodes", [])
    edges = result.get("edges", [])
    
    if not nodes:
        return "No architecture data found. The codebase may need to be indexed first."
    
    # Build a readable summary
    summary = []
    
    if not focus_path and not focus_symbol:
        # Level 0: Architecture overview
        summary.append("🏗️ SYSTEM ARCHITECTURE OVERVIEW")
        summary.append("=" * 40)
        
        components = [n for n in nodes if n.get("kind") == "component"]
        if components:
            for comp in components:
                summary.append(f"📦 {comp['name']} ({comp['id']})")
                summary.append(f"   └─ {comp.get('file_count', 0)} files, {comp.get('symbol_count', 0)} symbols")
                summary.append(f"   └─ {comp.get('description', 'No description')}")
                summary.append("")
        
        if edges:
            summary.append("🔗 COMPONENT DEPENDENCIES:")
            for edge in edges:
                summary.append(f"   {edge['from']} → {edge['to']} ({edge['type']})")
        
        summary.append(f"\n💡 Use focus_path='component_id' to drill down into a specific component")
        
    elif focus_path and not focus_symbol:
        if focus_path in ["api_endpoints", "pages_ui", "feature_components", "ui_components", "state_management", "utilities", "configuration"]:
            # Level 1: Component drill-down
            summary.append(f"📂 COMPONENT: {focus_path}")
            summary.append("=" * 40)
            
            for node in nodes:
                if node.get("kind") == "file":
                    summary.append(f"📄 {node['name']} ({node['file_path']})")
                    summary.append(f"   └─ {node.get('description', 'No description')}")
            
            if edges:
                summary.append("\n🔗 FILE RELATIONSHIPS:")
                for edge in edges:
                    summary.append(f"   {edge['from']} → {edge['to']} ({edge['type']})")
            
            summary.append(f"\n💡 Use focus_path='file_path' to see symbols in a specific file")
        else:
            # Level 2: File drill-down
            summary.append(f"📄 FILE: {focus_path}")
            summary.append("=" * 40)
            
            for node in nodes:
                summary.append(f"⚙️  {node['name']} ({node.get('kind', 'unknown')})")
                if node.get('signature'):
                    summary.append(f"   └─ {node['signature'][:80]}{'...' if len(node['signature']) > 80 else ''}")
                if node.get('description'):
                    summary.append(f"   └─ {node['description']}")
                summary.append("")
            
            if edges:
                summary.append("🔗 SYMBOL RELATIONSHIPS:")
                for edge in edges:
                    summary.append(f"   {edge['from']} → {edge['to']} ({edge['type']})")
            
            summary.append(f"\n💡 Use focus_symbol='symbol_name' to see call graph")
    
    elif focus_symbol:
        # Level 3: Symbol call graph
        summary.append(f"🕸️ CALL GRAPH: {focus_symbol}")
        summary.append("=" * 40)
        
        root_node = next((n for n in nodes if n['name'] == focus_symbol), None)
        if root_node:
            summary.append(f"🎯 ROOT: {root_node['name']} in {root_node['file_path']}")
            summary.append("")
        
        for node in nodes:
            if node['name'] != focus_symbol:
                summary.append(f"🔗 {node['name']} ({node.get('kind', 'unknown')}) in {node['file_path']}")
        
        if edges:
            summary.append("\n🔗 CALL RELATIONSHIPS:")
            for edge in edges:
                summary.append(f"   {edge['from']} → {edge['to']} ({edge['type']})")
    
    return "\n".join(summary)


# ── Plan Management Tools (Cloud-side) ────────────────────────────
# These let the agent create and manage its own execution plan.
# The IDE shows the plan with live status updates (checkmarks, etc.).

@tool
async def create_plan(steps: list[str], state: Annotated[AgentState, InjectedState]) -> str:
    """
    Create an execution plan for a complex task.
    Use this BEFORE starting work on tasks that involve multiple files,
    refactoring, new features, or anything requiring 3+ coordinated steps.
    
    Args:
        steps: List of step descriptions in execution order.
               Each step should be a concrete, actionable item.
    """
    if not steps:
        return "Error: steps list cannot be empty."
    
    new_steps = [
        PlanStep(number=i + 1, description=desc, status="pending")
        for i, desc in enumerate(steps)
    ]
    # Auto-mark step 1 as in_progress
    new_steps[0]["status"] = "in_progress"
    
    # We return a special value that the orchestration node will use to update state
    return json.dumps({
        "status": f"Plan created with {len(new_steps)} steps. Step 1 is now in progress.",
        "plan_steps": new_steps,
        "current_step": 1
    })


@tool
async def update_plan(step_number: int, status: str, state: Annotated[AgentState, InjectedState], new_description: str = "") -> str:
    """
    Update a plan step's status or description.
    Call this as you complete each step so the user can see your progress.
    
    Args:
        step_number: Which step to update (1-indexed).
        status: New status — one of: "done", "in_progress", "failed", "pending".
        new_description: Optional new description (leave empty to keep current).
    """
    plan_steps = list(state.get('plan_steps', []))
    current_step = state.get('current_step', 0)
    
    if not plan_steps:
        return "Error: No active plan. Use create_plan first."
    if step_number < 1 or step_number > len(plan_steps):
        return f"Error: Invalid step number {step_number}. Plan has {len(plan_steps)} steps."
    if status not in ("done", "in_progress", "failed", "pending"):
        return f"Error: Invalid status '{status}'. Use: done, in_progress, failed, pending."
    
    idx = step_number - 1
    old_status = plan_steps[idx]["status"]
    plan_steps[idx]["status"] = status
    if new_description:
        plan_steps[idx]["description"] = new_description
    
    # Auto-advance current_step when marking done
    if status == "done" and step_number == current_step:
        next_step = step_number + 1
        if next_step <= len(plan_steps):
            plan_steps[next_step - 1]["status"] = "in_progress"
            current_step = next_step
        else:
            current_step = 0  # All steps done
    
    msg = f"Step {step_number}: {old_status} → {status}"
    if current_step == 0:
        msg += " | All steps complete!"
    elif status == "done":
        msg += f" | Now on step {current_step}"
        
    return json.dumps({
        "status": msg,
        "plan_steps": plan_steps,
        "current_step": current_step
    })


@tool
async def discard_plan() -> str:
    """
    Discard the current plan entirely.
    Use when the approach needs to change fundamentally, or the task
    turned out to be simpler than expected and doesn't need a plan.
    """
    return json.dumps({
        "status": "Plan discarded.",
        "plan_steps": [],
        "current_step": 0
    })


@tool
async def add_plan_step(after_step: int, description: str, state: Annotated[AgentState, InjectedState]) -> str:
    """
    Insert a new step into the current plan.
    Use when you discover the task needs additional work not in the original plan.
    
    Args:
        after_step: Insert the new step AFTER this step number (0 = insert at beginning).
        description: What the new step should accomplish.
    """
    plan_steps = list(state.get('plan_steps', []))
    current_step = state.get('current_step', 0)
    
    if not plan_steps:
        return "Error: No active plan. Use create_plan first."
    if after_step < 0 or after_step > len(plan_steps):
        return f"Error: Invalid position. after_step must be 0-{len(plan_steps)}."
    if not description:
        return "Error: description cannot be empty."
    
    new_step = PlanStep(
        number=after_step + 1,
        description=description,
        status="pending"
    )
    plan_steps.insert(after_step, new_step)
    
    # Renumber all steps
    for i, step in enumerate(plan_steps):
        step["number"] = i + 1
    
    # Adjust current_step if we inserted before it
    if after_step < current_step:
        current_step += 1
        
    return json.dumps({
        "status": f"Inserted new step {after_step + 1}: '{description[:50]}...' Plan now has {len(plan_steps)} steps.",
        "plan_steps": plan_steps,
        "current_step": current_step
    })


@tool
async def remove_plan_step(step_number: int, state: Annotated[AgentState, InjectedState]) -> str:
    """
    Remove a step from the current plan.
    Use when a step is no longer needed (already handled, or approach changed).
    
    Args:
        step_number: Which step to remove (1-indexed).
    """
    plan_steps = list(state.get('plan_steps', []))
    current_step = state.get('current_step', 0)
    
    if not plan_steps:
        return "Error: No active plan."
    if step_number < 1 or step_number > len(plan_steps):
        return f"Error: Invalid step number {step_number}. Plan has {len(plan_steps)} steps."
    if len(plan_steps) == 1:
        return "Error: Cannot remove the last step. Use discard_plan instead."
    
    removed = plan_steps.pop(step_number - 1)
    
    # Renumber remaining steps
    for i, step in enumerate(plan_steps):
        step["number"] = i + 1
    
    # Adjust current_step
    if step_number < current_step:
        current_step -= 1
    elif step_number == current_step and current_step > len(plan_steps):
        current_step = len(plan_steps) if plan_steps else 0
        if plan_steps and current_step > 0:
            plan_steps[current_step - 1]["status"] = "in_progress"
            
    return json.dumps({
        "status": f"Removed step {step_number}: '{removed['description'][:40]}...' Plan now has {len(plan_steps)} steps.",
        "plan_steps": plan_steps,
        "current_step": current_step
    })


@tool
async def replan(reason: str, new_steps: list[str], state: Annotated[AgentState, InjectedState], keep_completed: bool = True) -> str:
    """
    Replace the current plan with a new one, optionally preserving completed work.
    Use when:
    - The original approach isn't working (2+ step failures)
    - You discovered the task is significantly different than expected
    - Major blockers require a different strategy
    
    Args:
        reason: Brief explanation of why re-planning is needed (shown to user).
        new_steps: The new list of steps to execute.
        keep_completed: If True, completed steps from the old plan are preserved
                       as a "completed" section for context. Default True.
    """
    old_plan_steps = state.get('plan_steps', [])
    
    if not new_steps:
        return "Error: new_steps list cannot be empty."
    
    # Capture completed steps for context
    completed_summary = ""
    if keep_completed and old_plan_steps:
        completed = [s for s in old_plan_steps if s["status"] == "done"]
        if completed:
            completed_summary = "Completed work preserved: " + "; ".join(
                f"✓ {s['description'][:40]}" for s in completed
            )
    
    # Create new plan
    plan_steps = [
        PlanStep(number=i + 1, description=desc, status="pending")
        for i, desc in enumerate(new_steps)
    ]
    plan_steps[0]["status"] = "in_progress"
    current_step = 1
    
    status_msg = f"Re-planned: {reason}\nNew plan has {len(plan_steps)} steps. Step 1 is now in progress."
    if completed_summary:
        status_msg += f"\n{completed_summary}"
        
    return json.dumps({
        "status": status_msg,
        "plan_steps": plan_steps,
        "current_step": current_step
    })


# ── Tool Definitions (IDE-side) ───────────────────────────────────
# These are "virtual" tools. When the agent calls them, the graph 
# pauses and returns to the IDE for execution.

@tool
def read_file(path: str) -> str:
    """Read the contents of a file from the user's machine."""
    return "PENDING_IDE_EXECUTION"


@tool
def write_to_file(path: str, content: str) -> str:
    """Create a new file or completely overwrite an existing one."""
    return "PENDING_IDE_EXECUTION"


@tool
def replace_in_file(path: str, old_str: str, new_str: str) -> str:
    """Replace text in a file. old_str must match exactly including whitespace."""
    return "PENDING_IDE_EXECUTION"


@tool
def execute_command(command: str) -> str:
    """Execute a shell command (git, npm, cargo, tests, etc.)."""
    return "PENDING_IDE_EXECUTION"


@tool
def grep(pattern: str, path: str = ".", glob: str = None, case_insensitive: bool = False) -> str:
    """
    Literal/regex text search using ripgrep. Fast, respects .gitignore.
    Use when you know the exact string to find (function name, error message, import).
    For semantic/conceptual search, use codebase_search instead.
    """
    return "PENDING_IDE_EXECUTION"


@tool
def list_files(path: str, recursive: bool = False) -> str:
    """List files in a directory."""
    return "PENDING_IDE_EXECUTION"


@tool
def delete_file(path: str) -> str:
    """Delete a file from the workspace."""
    return "PENDING_IDE_EXECUTION"


@tool
def apply_patch(patch: str) -> str:
    """Apply a unified diff patch to modify multiple files at once. The patch should be in standard unified diff format."""
    return "PENDING_IDE_EXECUTION"


@tool
def glob(pattern: str, path: str = ".") -> str:
    """Find files matching a glob pattern. E.g. '**/*.py' finds all Python files."""
    return "PENDING_IDE_EXECUTION"


@tool
def diagnostics(path: str, fix: bool = False) -> str:
    """Get linter/compiler errors for a file or directory. If fix=True, attempt auto-fix where supported."""
    return "PENDING_IDE_EXECUTION"


# ── Background Process Tools (IDE-side) ───────────────────────────

@tool
def execute_background(command: str, label: str = "") -> str:
    """Start a long-running command in background (e.g. dev servers). Returns process ID for later management."""
    return "PENDING_IDE_EXECUTION"


@tool
def read_process_output(pid: int, lines: int = 100) -> str:
    """Read recent output from a background process. Useful for checking server logs."""
    return "PENDING_IDE_EXECUTION"


@tool
def check_process_status(pid: int) -> str:
    """Check if a background process is still running."""
    return "PENDING_IDE_EXECUTION"


@tool
def kill_process(pid: int) -> str:
    """Stop a background process by its PID."""
    return "PENDING_IDE_EXECUTION"


@tool
def wait_for_port(port: int, timeout: int = 30, http_check: bool = False, path: str = "/") -> str:
    """Wait for a port to become available. Set http_check=True to also verify HTTP response is healthy (recommended for web servers)."""
    return "PENDING_IDE_EXECUTION"


@tool
def check_port(port: int) -> str:
    """Check if a port is currently in use."""
    return "PENDING_IDE_EXECUTION"


@tool
def kill_port(port: int) -> str:
    """Kill whatever process is using a port."""
    return "PENDING_IDE_EXECUTION"


# ── Code Intelligence Tools (IDE-side) ────────────────────────────

@tool
def list_code_definition_names(path: str) -> str:
    """List all symbol definitions (functions, classes, etc.) in a file. Great for understanding file structure."""
    return "PENDING_IDE_EXECUTION"


@tool
def get_symbol_definition(symbol: str, path: str = "") -> str:
    """Get the definition location and source code for a symbol. Uses tree-sitter for fast local lookup."""
    return "PENDING_IDE_EXECUTION"


@tool
def find_symbol_references(symbol: str, path: str = "") -> str:
    """Find all usages/references to a symbol across the codebase."""
    return "PENDING_IDE_EXECUTION"


# ── LSP Tools (IDE-side, powered by language servers) ───────────

@tool
def lsp_go_to_definition(path: str, line: int, column: int) -> str:
    """
    Get the exact definition location for a symbol at a specific position.
    Uses the IDE's language server (rust-analyzer, pyright, etc.) for 100% accurate results.
    Use this for reliable navigation instead of searching.
    Returns the file path and line number where the symbol is defined.
    """
    return "PENDING_IDE_EXECUTION"


@tool
def lsp_find_references(path: str, line: int, column: int) -> str:
    """
    Find ALL exact references to a symbol at a specific position.
    Uses the IDE's language server for accurate cross-file results.
    Great for understanding usage patterns and safe refactoring.
    Returns list of locations where the symbol is used.
    """
    return "PENDING_IDE_EXECUTION"


@tool
def lsp_hover(path: str, line: int, column: int) -> str:
    """
    Get exact type information and documentation for a symbol at a position.
    Uses the IDE's language server for accurate type info.
    Use this to understand complex types or function signatures without reading the whole file.
    Returns type signature, documentation, and other hover info.
    """
    return "PENDING_IDE_EXECUTION"


@tool
def lsp_rename(path: str, line: int, column: int, new_name: str) -> str:
    """
    Safely rename a symbol across the entire workspace.
    Uses the IDE's language server to find all occurrences and rename them atomically.
    This is the safest way to rename - better than find/replace.
    """
    return "PENDING_IDE_EXECUTION"


# Tool lists
PLAN_TOOLS = [create_plan, update_plan, discard_plan, add_plan_step, remove_plan_step, replan]
SERVER_TOOLS = [codebase_search, search_functions, search_classes, search_constants, search_files, trace_call_chain, impact_analysis, lookup_documentation, get_architecture_map] + PLAN_TOOLS
IDE_FILE_TOOLS = [read_file, write_to_file, replace_in_file, delete_file, apply_patch, list_files, glob]
IDE_EXEC_TOOLS = [execute_command, execute_background, read_process_output, check_process_status, kill_process]
IDE_PORT_TOOLS = [wait_for_port, check_port, kill_port]
IDE_CODE_TOOLS = [list_code_definition_names, get_symbol_definition, find_symbol_references, diagnostics]
LSP_TOOLS = [lsp_go_to_definition, lsp_find_references, lsp_hover, lsp_rename]

IDE_TOOLS = IDE_FILE_TOOLS + IDE_EXEC_TOOLS + IDE_PORT_TOOLS + IDE_CODE_TOOLS + [grep]
ALL_TOOLS = SERVER_TOOLS + IDE_TOOLS + LSP_TOOLS

# Tag tools for routing
for t in SERVER_TOOLS:
    t.metadata = {"type": "server"}
for t in IDE_TOOLS + LSP_TOOLS:
    t.metadata = {"type": "ide"}

IDE_TOOL_NAMES = {t.name for t in IDE_TOOLS + LSP_TOOLS}
SERVER_TOOL_NAMES = {t.name for t in SERVER_TOOLS}
PLAN_TOOL_NAMES = {t.name for t in PLAN_TOOLS}


# ── Query Decomposition ───────────────────────────────────────────

DECOMPOSE_PROMPT = """You are a code search query optimizer. Given a user's question about a codebase, extract:

1. **search_queries**: 2-4 short, focused search queries optimized for finding relevant code in a semantic code index. Each query should target a different aspect (e.g., one for the module, one for the feature, one for the pattern). Think "what would the function signatures and class names look like?" not "what is the user asking?".

2. **symbols**: 0-3 exact code symbol names (function names, class names, variable names) mentioned or implied. Only real identifiers, not English words.

If the user's open files are provided, use the imports, function names, and class names you see there to make your queries and symbols more precise. Prefer real symbol names from the files over guesses.

Respond with ONLY valid JSON:
{"search_queries": ["query1", "query2", ...], "symbols": ["SymbolName", "func_name", ...]}

Examples:
- "refactor auth to use JWT" → {"search_queries": ["authentication login session handler", "JWT token verification", "user auth middleware"], "symbols": []}
- "fix the bug in UserService.get_profile" → {"search_queries": ["UserService get_profile method", "user profile data fetch"], "symbols": ["UserService", "get_profile"]}
- "how does the payment flow work?" → {"search_queries": ["payment processing checkout", "payment gateway integration", "order payment status"], "symbols": []}
- "update the create_order function to validate inventory" → {"search_queries": ["create_order function implementation", "inventory validation stock check"], "symbols": ["create_order"]}"""


def _extract_file_hints(attached_files: dict[str, str] | None) -> str:
    """Extract useful hints from attached files to improve query decomposition.
    
    Pulls out: import statements, function/class definitions, and file paths.
    This tells the decomposer what symbols actually exist in the codebase.
    """
    if not attached_files:
        return ""
    
    hints = []
    for path, content in list(attached_files.items())[:3]:  # Limit to 3 files
        file_hints = [f"File: {path}"]
        
        for line in content.split("\n")[:100]:  # Scan first 100 lines
            stripped = line.strip()
            # Imports (Python, JS/TS, Rust, Go)
            if stripped.startswith(("import ", "from ", "use ", "require(", "const ", "export ")):
                file_hints.append(f"  {stripped[:120]}")
            # Function/class/struct definitions
            elif stripped.startswith(("def ", "class ", "async def ", "fn ", "func ", "function ", "struct ", "interface ", "type ")):
                file_hints.append(f"  {stripped[:120]}")
        
        if len(file_hints) > 1:  # More than just the filename
            hints.append("\n".join(file_hints[:15]))  # Cap per file
    
    return "\n".join(hints)


async def _decompose_query(question: str, attached_files: dict[str, str] | None = None) -> tuple[list[str], list[str]]:
    """Use the tool model to decompose a user question into focused search queries + symbols.
    
    If attached files are provided, extracts imports/definitions from them
    so the LLM knows what symbols actually exist in the codebase.
    
    Returns (search_queries, symbols). Falls back to the raw question on failure.
    """
    model = llm_provider.get_tool_model(temperature=0.0)
    
    # Build the user message with file context if available
    user_content = question
    file_hints = _extract_file_hints(attached_files)
    if file_hints:
        user_content = f"{question}\n\n--- User's open files (for reference) ---\n{file_hints}"
    
    try:
        response = await model.ainvoke([
            SystemMessage(content=DECOMPOSE_PROMPT),
            HumanMessage(content=user_content),
        ])
        
        text = response.content.strip()
        # Handle models that wrap JSON in markdown code blocks
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        
        result = json.loads(text)
        queries = result.get("search_queries", [question])
        symbols = result.get("symbols", [])
        
        # Validate
        queries = [q for q in queries if isinstance(q, str) and q.strip()][:4]
        symbols = [s for s in symbols if isinstance(s, str) and s.strip()][:3]
        
        if not queries:
            queries = [question]
        
        logger.info("[decompose] %d queries, %d symbols from: %s", len(queries), len(symbols), question[:80])
        return queries, symbols
    except Exception as e:
        logger.warning("[decompose] Failed (%s), falling back to raw question", e)
        return [question], []


# ── Project Profile ───────────────────────────────────────────────
#
# Reads key config files and uses the tool model to understand the
# project's tech stack, versions, and detect compatibility issues
# BEFORE the agent writes any code.

# Config files to look for (checked in order, missing files are skipped)
PROJECT_CONFIG_FILES = [
    # Package managers / dependencies
    "package.json", "Cargo.toml", "requirements.txt", "pyproject.toml",
    "go.mod", "Gemfile", "pom.xml", "build.gradle",
    # Framework configs
    "next.config.js", "next.config.mjs", "next.config.ts",
    "vite.config.ts", "vite.config.js",
    "nuxt.config.ts", "angular.json", "svelte.config.js",
    # TypeScript / JavaScript
    "tsconfig.json", "jsconfig.json",
    # Styling
    "tailwind.config.js", "tailwind.config.ts", "postcss.config.js", "postcss.config.mjs",
    # UI libraries
    "components.json",  # Shadcn
    # Linting / formatting
    ".eslintrc.js", ".eslintrc.json", "eslint.config.js",
    ".prettierrc", ".prettierrc.json",
    # Environment / runtime
    ".nvmrc", ".node-version", ".python-version", ".tool-versions",
    "Dockerfile", "docker-compose.yml",
    # Git
    ".gitignore",
]

# Global CSS files to check for styling config
PROJECT_CSS_FILES = [
    "styles/globals.css", "src/styles/globals.css",
    "app/globals.css", "src/app/globals.css",
    "src/index.css", "styles/global.css",
]

PROFILE_ANALYSIS_PROMPT = """Analyze this project's configuration files and produce a concise project profile.

Extract:
1. **Framework** + exact version (e.g., "Next.js 13.4.0 (Pages Router)" or "Vite 5.2 + React 18")
2. **Language** + version (e.g., "TypeScript 5.3, strict mode")
3. **Styling** + version (e.g., "Tailwind CSS 4.1.18 with v4 CSS syntax")
4. **UI Library** (e.g., "Shadcn UI" if components.json present)
5. **Package manager** (npm/yarn/pnpm/bun — infer from lock file or config)
6. **Test framework** (jest, vitest, pytest, etc. or "none detected")
7. **Key conventions** (src/ vs app/ structure, module system ESM/CJS)

Then check for **WARNINGS** — version conflicts, configuration mismatches, or common pitfalls:
- Framework version vs dependency version compatibility
- Config file syntax vs installed version mismatches (e.g., Tailwind v4 CSS syntax but v3 config file format)
- Missing peer dependencies
- Outdated patterns that will cause issues

Format as a concise profile. Keep it under 400 words. Start warnings with "WARNING:" so they stand out.

Respond with ONLY the profile text, no JSON wrapping."""


async def build_project_profile(attached_files: dict[str, str] | None = None) -> str:
    """Build a project profile from config files.
    
    Reads key config files from attached_files (sent by IDE) and uses the
    tool model to analyze tech stack, versions, and detect conflicts.
    
    Returns a profile string to inject into the agent's context.
    """
    if not attached_files:
        return ""
    
    # Step 1: Collect config file contents from attached files
    config_contents = {}
    all_config_names = set(PROJECT_CONFIG_FILES + PROJECT_CSS_FILES)
    
    for path, content in attached_files.items():
        # Match by filename (attached files have full paths)
        filename = path.rsplit("/", 1)[-1] if "/" in path else path
        # Also check path suffixes for nested configs like styles/globals.css
        matches = (
            filename in all_config_names
            or any(path.endswith(f) for f in all_config_names)
        )
        if matches and content.strip():
            # Truncate very large configs (package-lock.json etc.)
            truncated = content[:3000] if len(content) > 3000 else content
            config_contents[path] = truncated
    
    if not config_contents:
        logger.info("[project_profile] No config files found in attached files")
        return ""
    
    logger.info("[project_profile] Found %d config files: %s", 
               len(config_contents), list(config_contents.keys()))
    
    # Step 2: Analyze with tool model
    files_text = "\n\n".join(
        f"### {path}\n```\n{content}\n```"
        for path, content in config_contents.items()
    )
    
    try:
        model = llm_provider.get_tool_model(temperature=0.0)
        response = await model.ainvoke([
            SystemMessage(content=PROFILE_ANALYSIS_PROMPT),
            HumanMessage(content=files_text),
        ])
        
        profile = response.content.strip()
        logger.info("[project_profile] Built profile (%d chars): %s", len(profile), profile[:200])
        return profile
        
    except Exception as e:
        logger.warning("[project_profile] Analysis failed (%s), building basic profile", e)
        # Fallback: just list what we found
        files_list = ", ".join(config_contents.keys())
        return f"Project config files detected: {files_list}"


# ── Pre-Enrichment Logic ──────────────────────────────────────────

async def build_pre_enrichment(
    workspace_id: str,
    question: str,
    attached_files: dict[str, str] | None = None,
) -> str:
    """
    The secret sauce: gather ALL relevant context BEFORE the LLM sees the question.
    
    This is how we make a small model match Claude Opus - by doing the hard work
    of finding the right context upfront.
    
    Uses query decomposition to turn a user instruction into multiple focused
    search queries, getting much better recall from the code index.
    """
    import asyncio
    parts = []
    
    # 1. Attached files from IDE (live file contents)
    if attached_files:
        parts.append("## Live Files (from IDE)\n")
        for path, content in list(attached_files.items())[:5]:  # Limit to 5 files
            truncated = content[:8000] + "\n... (truncated)" if len(content) > 8000 else content
            parts.append(f"### `{path}`\n```\n{truncated}\n```\n")
        parts.append("")
    
    # 2. Decompose question into focused search queries + symbols
    #    Pass attached files so the LLM can reference real symbol names
    search_queries, symbols = await _decompose_query(question, attached_files)
    logger.info("Pre-enrichment: %d queries=%s, %d symbols=%s", 
                len(search_queries), search_queries, len(symbols), symbols)
    
    # 3. Run all search queries in parallel for speed
    async def _search(query: str) -> list[dict]:
        try:
            query_emb = await embeddings.embed_query(query)
            results = await store.vector_search(workspace_id, query_emb, top_k=5)
            return results
        except Exception as e:
            logger.error("Search failed for query '%s': %s", query, e)
            return []
    
    all_search_results = await asyncio.gather(*[_search(q) for q in search_queries])
    
    # 4. Deduplicate results across queries (by symbol name/path)
    seen = set()
    unique_results = []
    for results in all_search_results:
        for r in results:
            sym = r.get("symbol", {})
            # Deduplicate by (name, file_path) to avoid showing the same code twice
            key = (sym.get("name", ""), sym.get("file_path", ""))
            if key not in seen:
                seen.add(key)
                unique_results.append(sym)
    
    logger.info("Pre-enrichment: %d unique results from %d queries", len(unique_results), len(search_queries))
    
    if unique_results:
        # Cap at 12 results to avoid token explosion
        context_text = chat_utils.build_context_from_results(unique_results[:12])
        if context_text:
            parts.append(context_text)
            logger.info("Pre-enrichment: built context len=%d", len(context_text))
    
    # 5. Trace call chains for extracted symbols
    for sym in symbols[:2]:
        try:
            trace = await store.trace_call_chain(workspace_id, sym, direction="both", max_depth=2)
            if trace.get("nodes"):
                parts.append(f"\n## Call Chain: `{sym}`\n")
                for edge in trace.get("edges", [])[:10]:
                    parts.append(f"  `{edge['from']}` → `{edge['to']}`")
                parts.append("")
        except Exception as e:
            logger.debug("Trace failed for %s: %s", sym, e)
    
    return "\n".join(parts).strip()




# ── Plan Display ──────────────────────────────────────────────────

def format_plan_for_prompt(steps: list[PlanStep], current_step: int) -> str:
    """Format the plan as a prompt section for the LLM to reference."""
    if not steps:
        return ""
    
    lines = ["## Your Execution Plan\n"]
    for step in steps:
        status_icon = {
            "pending": "⬜",
            "in_progress": "🔄",
            "done": "✅",
            "failed": "❌",
        }.get(step["status"], "⬜")
        
        marker = " ← YOU ARE HERE" if step["number"] == current_step else ""
        lines.append(f"{status_icon} {step['number']}. {step['description']}{marker}")
    
    lines.append("")
    lines.append(f"**Current step: {current_step} of {len(steps)}**")
    lines.append("Focus on completing the current step. Use tools to execute it, then move to the next.")
    
    return "\n".join(lines)


# ── Documentation Lookup (Context7 + DevDocs.io) ──────────────────

async def _fetch_documentation(library: str, query: str) -> str:
    """
    Fetch documentation from Context7 and DevDocs.io.
    
    Context7: MCP-based, provides curated library docs with semantic search
    DevDocs.io: Free API for browsing official documentation
    """
    results = []
    
    # Try Context7 first (if API key is configured)
    if CONTEXT7_API_KEY:
        try:
            c7_result = await _query_context7(library, query)
            if c7_result:
                results.append(f"## Context7 Documentation\n\n{c7_result}")
        except Exception as e:
            logger.warning("Context7 lookup failed: %s", e)
    
    # Try DevDocs.io as fallback/supplement
    try:
        devdocs_result = await _query_devdocs(library, query)
        if devdocs_result:
            results.append(f"## DevDocs.io Documentation\n\n{devdocs_result}")
    except Exception as e:
        logger.warning("DevDocs lookup failed: %s", e)
    
    if results:
        return "\n\n---\n\n".join(results)
    else:
        return f"No documentation found for '{library}' with query '{query}'. Try a different library name or query."


async def _query_context7(library: str, query: str) -> str:
    """Query Context7 MCP API for library documentation."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        # Step 1: Resolve library ID
        resolve_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "resolve-library-id",
                "arguments": {"libraryName": library}
            }
        }
        
        headers = {"Content-Type": "application/json"}
        if CONTEXT7_API_KEY:
            headers["Authorization"] = f"Bearer {CONTEXT7_API_KEY}"
        
        response = await client.post(CONTEXT7_API_URL, json=resolve_payload, headers=headers)
        response.raise_for_status()
        resolve_result = response.json()
        
        # Extract library ID from response
        if "result" not in resolve_result or "content" not in resolve_result["result"]:
            logger.debug("Context7: No library found for %s", library)
            return ""
        
        content = resolve_result["result"]["content"]
        if not content or not isinstance(content, list):
            return ""
        
        # Parse the text content to find library ID
        text_content = content[0].get("text", "") if content else ""
        if not text_content or "No libraries found" in text_content:
            return ""
        
        # Extract first library ID (format: /library-name/version)
        lines = text_content.strip().split("\n")
        library_id = None
        for line in lines:
            if line.startswith("- "):
                # Extract ID from format: "- library-name: /library/version"
                parts = line.split(": ")
                if len(parts) >= 2:
                    library_id = parts[1].strip()
                    break
        
        if not library_id:
            return ""
        
        logger.info("Context7: Resolved %s -> %s", library, library_id)
        
        # Step 2: Query documentation
        query_payload = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "get-library-docs",
                "arguments": {
                    "context7CompatibleLibraryID": library_id,
                    "topic": query
                }
            }
        }
        
        response = await client.post(CONTEXT7_API_URL, json=query_payload, headers=headers)
        response.raise_for_status()
        docs_result = response.json()
        
        if "result" in docs_result and "content" in docs_result["result"]:
            doc_content = docs_result["result"]["content"]
            if doc_content and isinstance(doc_content, list):
                return doc_content[0].get("text", "")[:8000]  # Limit size
        
        return ""


async def _query_devdocs(library: str, query: str) -> str:
    """Query DevDocs.io API for documentation."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        # Map common library names to DevDocs slugs (verified against docs.json)
        # Note: Some libraries like pydantic, langchain are NOT available on DevDocs
        slug_map = {
            "react": "react",
            "vue": "vue~3",
            "angular": "angular",
            "fastapi": "fastapi",
            "flask": "flask",
            "django": "django~5.2",
            "express": "express",
            "nextjs": "next.js",
            "next": "next.js",
            "typescript": "typescript",
            "python": "python~3.12",
            "rust": "rust",
            "go": "go",
            "nodejs": "node",
            "node": "node",
            "pandas": "pandas~2",
            "numpy": "numpy~2",
            "pytorch": "pytorch",
            "tensorflow": "tensorflow~2",
            "sqlalchemy": "sqlalchemy~2",
            "tailwind": "tailwindcss",
            "tailwindcss": "tailwindcss",
            "docker": "docker",
            "git": "git",
            "redis": "redis",
            "postgresql": "postgresql~16",
            "postgres": "postgresql~16",
            "mongodb": "mongoose",
            "graphql": "graphql",
            "webpack": "webpack~5",
            "vite": "vite",
            "jest": "jest",
            "cypress": "cypress",
            "axios": "axios",
            "lodash": "lodash~4",
            "moment": "moment",
            "d3": "d3~7",
            "sass": "sass",
            "css": "css",
            "html": "html",
            "dom": "dom",
            "javascript": "javascript",
            "js": "javascript",
        }
        
        slug = slug_map.get(library.lower(), library.lower())
        
        # Libraries not available on DevDocs - return early with helpful message
        not_available = {"pydantic", "langchain", "langgraph", "openai", "anthropic", "groq", "huggingface", "transformers"}
        if library.lower() in not_available:
            return f"Note: {library} documentation is not available on DevDocs.io. The agent can still help based on its training knowledge."
        
        try:
            # Get doc index from devdocs.io
            index_url = f"https://devdocs.io/docs/{slug}/index.json"
            response = await client.get(index_url)
            
            if response.status_code == 404:
                logger.debug("DevDocs: No docs found for slug %s", slug)
                return ""
            
            response.raise_for_status()
            index = response.json()
            
            # Search entries for matching items
            entries = index.get("entries", [])
            query_lower = query.lower()
            
            # Find matching entries by name or path
            matches = []
            for entry in entries:
                name = entry.get("name", "").lower()
                path = entry.get("path", "").lower()
                entry_type = entry.get("type", "").lower()
                if query_lower in name or query_lower in path or query_lower in entry_type:
                    matches.append(entry)
                    if len(matches) >= 5:  # Collect more matches
                        break
            
            if not matches:
                # Try partial word match
                query_words = query_lower.split()
                for entry in entries:
                    name = entry.get("name", "").lower()
                    entry_type = entry.get("type", "").lower()
                    if any(word in name or word in entry_type for word in query_words):
                        matches.append(entry)
                        if len(matches) >= 5:
                            break
            
            if not matches:
                return f"Found docs for {library} but no entries matching '{query}'. Available topics include: {', '.join(e['name'] for e in entries[:10])}"
            
            # Fetch content from the db.json (documents CDN)
            db_url = f"https://documents.devdocs.io/{slug}/db.json"
            db_response = await client.get(db_url)
            
            if db_response.status_code != 200:
                logger.debug("DevDocs: Could not fetch db.json for %s", slug)
                return f"Found matching topics for {library}: {', '.join(m['name'] for m in matches[:5])}"
            
            db = db_response.json()
            
            # Get content for matched entries
            results = []
            for match in matches[:3]:  # Limit to 3 full docs
                path = match.get("path", "")
                # db.json uses path without anchor (fragment)
                base_path = path.split('#')[0] if '#' in path else path
                
                if base_path in db:
                    html = db[base_path]
                    # Convert HTML to readable text
                    text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
                    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
                    text = re.sub(r'<pre[^>]*>(.*?)</pre>', r'\n```\n\1\n```\n', text, flags=re.DOTALL)
                    text = re.sub(r'<code[^>]*>(.*?)</code>', r'`\1`', text, flags=re.DOTALL)
                    text = re.sub(r'<h([1-6])[^>]*>(.*?)</h\1>', r'\n\n## \2\n\n', text, flags=re.DOTALL)
                    text = re.sub(r'<li[^>]*>(.*?)</li>', r'- \1\n', text, flags=re.DOTALL)
                    text = re.sub(r'<p[^>]*>(.*?)</p>', r'\1\n\n', text, flags=re.DOTALL)
                    text = re.sub(r'<[^>]+>', '', text)
                    text = re.sub(r'\n{3,}', '\n\n', text)
                    text = re.sub(r' +', ' ', text).strip()
                    
                    if text:
                        results.append(f"### {match['name']}\n\n{text[:4000]}")
            
            return "\n\n---\n\n".join(results) if results else ""
            
        except Exception as e:
            logger.warning("DevDocs query failed: %s", e)
            return ""


# ── Graph Nodes ───────────────────────────────────────────────────

def _get_last_question(state: AgentState) -> str:
    """Extract the last human question from message history."""
    for msg in reversed(state['messages']):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


# Cosine similarity threshold: below this, the topic has shifted enough to re-enrich.
# 0.65 means ~50% topical overlap — tolerant of rephrasing, triggers on real pivots.
TOPIC_SHIFT_THRESHOLD = 0.65


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def _step_enrichment(workspace_id: str, step_description: str) -> str:
    """Lightweight supplementary search based on a plan step description.
    
    Unlike full pre-enrichment (query decomposition + multi-query + call chains),
    this is a single focused search — just enough to give the agent context
    for the current step without the overhead.
    """
    try:
        query_emb = await embeddings.embed_query(step_description)
        results = await store.vector_search(workspace_id, query_emb, top_k=5)
        
        if results:
            flat_results = [r["symbol"] for r in results]
            context_text = chat_utils.build_context_from_results(flat_results)
            if context_text:
                logger.info("[step_enrichment] Found %d results for step: %s", len(flat_results), step_description[:60])
                return f"\n\n## Context for Current Step\n\n{context_text}"
        
        return ""
    except Exception as e:
        logger.warning("[step_enrichment] Failed for step '%s': %s", step_description[:60], e)
        return ""


# ── Intent Analysis ───────────────────────────────────────────────

INTENT_ANALYSIS_PROMPT = """You are a retrieval optimizer for an AI software engineer. 
Analyze the recent conversation history and decide if the AI needs more code context to complete its next step.

If context is needed, generate a concise, focused search query (2-5 words) that targets the specific symbols, logic, or patterns the AI is currently working on.
If the AI already has enough context or is performing a simple task (like explaining a concept or acknowledging), respond with "NONE_NEEDED".

Examples:
- User: "Fix the bug in auth.py" -> "auth.py error handling"
- Assistant: "I see an issue in the JWT validator" -> "JWT validation implementation"
- User: "Thanks!" -> "NONE_NEEDED"

Respond with ONLY the query or "NONE_NEEDED"."""


async def _analyze_retrieval_intent(messages: list[BaseMessage]) -> str:
    """Use the tool model to identify EXACTLY what context is missing."""
    try:
        # Use the tool model as requested for intent analysis
        model = llm_provider.get_tool_model(temperature=0.0)
        
        # Build a compact history
        history_parts = []
        for msg in messages[-3:]:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            history_parts.append(f"{role}: {content[:300]}")
        
        history_text = "\n".join(history_parts)
        
        response = await model.ainvoke([
            SystemMessage(content=INTENT_ANALYSIS_PROMPT),
            HumanMessage(content=f"Conversation history:\n{history_text}")
        ])
        
        intent = response.content.strip().upper()
        if "NONE_NEEDED" in intent:
            return "NONE_NEEDED"
            
        # Clean up any potential markdown or extra text
        intent = intent.replace('"', '').replace("'", "").split('\n')[0]
        logger.info(f"[intent_analysis] Context gap identified using tool model: {intent}")
        return intent
    except Exception as e:
        logger.warning(f"[intent_analysis] Failed: {e}")
        return "NONE_NEEDED"


async def enrich_context(state: AgentState) -> dict:
    """
    MCP-STYLE TARGETED RETRIEVAL: 
    1. Analyzes history to find the 'Context Gap'.
    2. Generates a precise code-search query.
    3. Injects only the highly relevant snippets.
    
    NOTE: This node now explicitly CLEARS stale context if no new gap is identified,
    ensuring the model only sees what it needs for the CURRENT turn.
    """
    workspace_id = state['workspace_id']
    messages = state['messages']
    
    # 1. ANALYZE INTENT (The 'MCP' part)
    # Use a fast model to identify EXACTLY what context is missing.
    context_intent = await _analyze_retrieval_intent(messages)
    
    if context_intent == "NONE_NEEDED":
        logger.info("[enrich_context] No additional context needed - clearing stale context")
        # Explicitly return an empty string to clear previous turn's context from state
        return {"enriched_context": ""}

    # 2. ARCHITECTURE-AWARE TARGETED SEARCH
    # First get architectural context to understand the system structure
    try:
        # Get high-level architecture overview
        arch_map = await store.get_project_map(workspace_id=workspace_id)
        components = [n for n in arch_map.get("nodes", []) if n.get("kind") == "component"]
        
        # Build architecture context
        arch_context = "🏗️ SYSTEM ARCHITECTURE:\n"
        for comp in components:
            arch_context += f"- {comp['name']}: {comp.get('description', 'No description')} ({comp.get('file_count', 0)} files)\n"
        arch_context += "\n"
        
        # Perform semantic search with architectural awareness
        query_emb = await embeddings.embed_query(context_intent)
        results = await store.vector_search(workspace_id, query_emb, top_k=8)  # Get more results for better coverage
        
        if results:
            # Group results by architectural component for better organization
            component_groups = {}
            for r in results:
                file_path = r["symbol"].get("file_path", "")
                
                # Determine which component this file belongs to
                if file_path.startswith("pages/api/"):
                    comp = "API Endpoints"
                elif file_path.startswith("pages/"):
                    comp = "Pages & Routes"
                elif file_path.startswith("components/ui/"):
                    comp = "UI Components"
                elif file_path.startswith("components/"):
                    comp = "Feature Components"
                elif file_path.startswith("context/"):
                    comp = "State Management"
                elif file_path.startswith(("lib/", "utils/")):
                    comp = "Utilities"
                else:
                    comp = "Configuration"
                
                if comp not in component_groups:
                    component_groups[comp] = []
                component_groups[comp].append(r)
            
            # Build organized context with architecture information
            context_parts = [arch_context]
            context_parts.append("🎯 RELEVANT CODE BY COMPONENT:")
            context_parts.append("=" * 40)
            
            for comp_name, comp_results in component_groups.items():
                if comp_results:  # Only show components that have relevant results
                    context_parts.append(f"\n📦 {comp_name}:")
                    context_parts.append("-" * 20)
                    
                    flat_results = [r["symbol"] for r in comp_results[:3]]  # Limit per component
                    comp_context = chat_utils.build_context_from_results(flat_results)
                    context_parts.append(comp_context)
            
            new_context = "\n".join(context_parts)
            logger.info(f"[enrich_context] Injected architecture-aware context for: {context_intent}")
            return {"enriched_context": new_context}
        else:
            # Even if no search results, provide architecture context
            logger.info(f"[enrich_context] No search results, providing architecture context only")
            return {"enriched_context": arch_context}
            
    except Exception as e:
        logger.error(f"[enrich_context] Architecture-aware RAG failed: {e}")
        # Fallback to basic search
        try:
            query_emb = await embeddings.embed_query(context_intent)
            results = await store.vector_search(workspace_id, query_emb, top_k=5)
            
            if results:
                flat_results = [r["symbol"] for r in results]
                new_context = chat_utils.build_context_from_results(flat_results)
                logger.info(f"[enrich_context] Fallback: Injected {len(flat_results)} snippets for: {context_intent}")
                return {"enriched_context": new_context}
        except Exception as fallback_e:
            logger.error(f"[enrich_context] Fallback also failed: {fallback_e}")
        
        # Clear context on error to prevent model from acting on stale/wrong information
        return {"enriched_context": ""}



# ── Context Window Management ─────────────────────────────────────

# Rough char budget: 120K tokens ≈ 480K chars.  Reserve room for
# system prompt (~4K), enriched context (~20K), and model output (~4K).
MAX_HISTORY_CHARS = 200_000  # ~50K tokens for conversation history
MAX_TOOL_MSG_CHARS = 30_000  # Truncate individual tool results

SUMMARIZE_PROMPT = """Summarize this conversation segment concisely. Focus on:
- What files were read or edited (exact paths)
- What changes were made and why
- Key decisions or findings
- Any errors encountered and how they were resolved
- Current state of the task

Be brief (under 500 words). Use bullet points. Include exact file paths and symbol names."""


async def _summarize_dropped_messages(dropped_messages: list[BaseMessage]) -> str:
    """Use the tool model to summarize conversation messages that are being dropped.
    
    This preserves the key decisions, file edits, and findings from the middle
    of a long conversation instead of just saying "[N messages omitted]".
    """
    # Build a compact representation of the dropped messages
    parts = []
    for msg in dropped_messages:
        if isinstance(msg, HumanMessage):
            parts.append(f"[User]: {msg.content[:500]}")
        elif isinstance(msg, AIMessage):
            text = msg.content[:500] if msg.content else ""
            tools = ", ".join(tc["name"] for tc in (msg.tool_calls or []))
            if tools:
                parts.append(f"[Assistant]: {text}\n  Tools called: {tools}")
            elif text:
                parts.append(f"[Assistant]: {text}")
        elif isinstance(msg, ToolMessage):
            # Keep tool results short — just the first/last lines
            content = msg.content
            if len(content) > 300:
                content = content[:150] + "\n...\n" + content[-150:]
            parts.append(f"[Tool result]: {content}")
    
    conversation_text = "\n\n".join(parts)
    
    # Cap the input to avoid blowing up the summarization call itself
    if len(conversation_text) > 30_000:
        conversation_text = conversation_text[:30_000] + "\n\n... (further messages truncated for summarization)"
    
    try:
        model = llm_provider.get_tool_model(temperature=0.0)
        response = await model.ainvoke([
            SystemMessage(content=SUMMARIZE_PROMPT),
            HumanMessage(content=conversation_text),
        ])
        summary = response.content.strip()
        logger.info("[summarize] Summarized %d messages into %d chars", len(dropped_messages), len(summary))
        return summary
    except Exception as e:
        logger.warning("[summarize] Failed (%s), using fallback", e)
        # Fallback: extract just the tool names and file paths
        tool_names = set()
        for msg in dropped_messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_names.add(tc["name"])
                    path = tc.get("args", {}).get("path", "")
                    if path:
                        tool_names.add(f"  {tc['name']}({path})")
        if tool_names:
            return f"[Summary of {len(dropped_messages)} earlier messages — tools used: {', '.join(sorted(tool_names))}]"
        return f"[{len(dropped_messages)} earlier messages omitted to fit context window]"


async def _truncate_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Truncate oversized tool messages and trim old history if needed.
    
    When dropping middle messages, summarizes them using the tool model
    so the agent remembers what it did (files edited, decisions made).
    """
    result = []
    total_chars = 0
    
    for msg in messages:
        if isinstance(msg, ToolMessage) and len(msg.content) > MAX_TOOL_MSG_CHARS:
            # Truncate large tool results (file dumps, command output)
            truncated = msg.content[:MAX_TOOL_MSG_CHARS] + (
                f"\n\n... (truncated from {len(msg.content)} chars)"
            )
            msg = ToolMessage(
                content=truncated,
                tool_call_id=msg.tool_call_id,
            )
        
        total_chars += len(msg.content)
        result.append(msg)
    
    # If total history is still too large, keep first 2 + last N messages
    # and SUMMARIZE the dropped middle instead of discarding it
    if total_chars > MAX_HISTORY_CHARS:
        if len(result) > 6:
            kept_start = result[:2]   # First human msg + first AI response
            kept_end = result[-4:]     # Last 4 messages (most recent context)
            dropped = result[2:-4]     # Middle messages to summarize
            
            logger.warning("[truncate] Summarizing %d middle messages to fit context", len(dropped))
            summary_text = await _summarize_dropped_messages(dropped)
            
            summary = AIMessage(content=f"## Summary of Earlier Work\n\n{summary_text}")
            result = kept_start + [summary] + kept_end
            logger.info("[truncate] Kept %d messages (2 start + summary + 4 end)", len(result))
    
    return result





# ── Multi-Model Routing ───────────────────────────────────────────
#
# Strategy: use the best model for each phase of the agent loop.
# Models are configured via app/llm.py (backed by LiteLLM).
#
#   REASONING model (LLM_REASONING_MODEL):
#     First call with enriched context + user question.
#     Needs strong reasoning to understand the codebase and plan.
#
#   TOOL model (LLM_TOOL_MODEL):
#     Subsequent calls that process tool results and decide next action.
#     Needs fast responses + reliable function calling.
#
# Works with ANY provider: Groq, Gemini, Claude, Fireworks, OpenAI, etc.
# Just set the model string: "groq/kimi-k2", "gemini/gemini-2.0-flash", etc.


async def _pick_model_name_intelligent(state: AgentState) -> str:
    """PHASE 3: Use LLM to intelligently select optimal model."""
    config = llm_provider.get_config()
    
    available_models = {
        "fast": config.tool_model,
        "reasoning": config.reasoning_model,
        "planning": llm_provider.get_planning_model_name() or config.reasoning_model
    }
    
    # Get budget constraint from environment or default to balanced
    budget = os.getenv("MODEL_ROUTING_BUDGET", "balanced")  # fast/balanced/quality
    
    try:
        model = await intelligent_model_router.get_optimal_model_for_turn(
            state,
            available_models,
            budget
        )
        return model
    except Exception as e:
        logger.error("[model_routing] Intelligent routing failed: %s, falling back", e)
        # Fallback to reasoning model
        return config.reasoning_model


def _pick_model_name(messages: list[BaseMessage], plan_steps: list[PlanStep] = None, current_step: int = 0) -> str:
    """Pick model string based on what the agent needs to do RIGHT NOW.
    
    Heuristic-based routing:
    - Planning model (Claude): First call when no plan exists.
    - Reasoning model (Stronger): After a tool failure or first step of a new plan.
    - Tool model (Faster): Straightforward execution and routine steps.
    """
    config = llm_provider.get_config()
    planning_model = llm_provider.get_planning_model_name()
    
    # No tool messages at all → first call
    has_tool_messages = any(isinstance(m, ToolMessage) for m in messages)
    if not has_tool_messages:
        if not plan_steps and planning_model:
            return planning_model
        return config.reasoning_model
    
    # Check for failure signals → reasoning model
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            content_lower = msg.content.lower() if msg.content else ""
            if any(signal in content_lower for signal in ("error", "failed", "exception", "not found")):
                return config.reasoning_model
        elif isinstance(msg, AIMessage):
            break
    
    # First step of new plan → reasoning
    if plan_steps and current_step == 1:
        any_done = any(s["status"] == "done" for s in plan_steps)
        if not any_done:
            return config.reasoning_model
            
    return config.tool_model


async def call_model(state: AgentState) -> dict:
    """The 'Brain' node - LLM reasoning with full context.
    
    Now plan-aware: if there's an active plan, it injects the plan display
    and current step instructions into the system prompt.
    """
    enriched_context = state.get('enriched_context', '')
    plan_steps = list(state.get('plan_steps', []))  # Make mutable copy
    current_step = state.get('current_step', 0)
    workspace_id = state['workspace_id']
    
    # ── PHASE 1: Load workspace memory (cross-trace learning) ──────────────
    memory = await ws_memory.load_workspace_memory(workspace_id)
    
    # ── PHASE 1: Check if we should ask for help ────────────────────────────
    should_ask, help_message = await ws_memory.should_ask_for_help(workspace_id, state)
    if should_ask:
        logger.warning("[call_model] Should ask for help - exhausted approaches detected")
        return {"messages": [AIMessage(content=help_message)]}
    
    logger.info("[call_model] enriched_context=%d chars, plan_steps=%d, current_step=%d, memory_failures=%d",
                len(enriched_context), len(plan_steps), current_step, len(memory.get('failed_commands', {})))
    
    # ── PHASE 3: Check if we should create a learning checkpoint ────────────────
    # (Phase 3 disabled in simplified Roo-Code style)
    if False:
        checkpoints = state.get('checkpoints', [])
    last_checkpoint_time_str = state.get('last_checkpoint_time')
    last_checkpoint_time = None
    if last_checkpoint_time_str:
        from datetime import datetime
        last_checkpoint_time = datetime.fromisoformat(last_checkpoint_time_str)
    
    # if ENABLE_PHASE_3 and await learning_checkpoints.should_create_checkpoint(state, last_checkpoint_time):
    if False:
        logger.info("[call_model] Creating learning checkpoint...")
        
        # Extract recent errors and successes
        recent_errors = []
        recent_successes = []
        for msg in reversed(state['messages'][-20:]):
            if isinstance(msg, ToolMessage) and msg.content:
                content = msg.content[:300]
                if any(sig in content.lower() for sig in ['error', 'failed', 'exception']):
                    recent_errors.append(content)
                elif any(sig in content.lower() for sig in ['success', 'completed', 'done']):
                    recent_successes.append(content)
        
        try:
            # Use fast model for checkpoint creation
            config = llm_provider.get_config()
            fast_model = llm_provider.get_chat_model(config.tool_model, temperature=0)
            
            checkpoint = await learning_checkpoints.create_checkpoint_with_llm(
                state['messages'][-30:],
                recent_errors[:5],
                recent_successes[:5],
                fast_model
            )
            
            checkpoints.append(checkpoint)
            last_checkpoint_time = datetime.now(timezone.utc)
            
            logger.info("[call_model] Checkpoint created: %d facts learned", 
                       len(checkpoint['learned_facts']))
        except Exception as e:
            logger.error("[call_model] Checkpoint creation failed: %s", e)
    

    
    # ── User message during plan execution: handle gracefully ────────────
    # If the user sends a message while we're in the middle of a plan,
    # inject guidance so the agent responds appropriately
    if plan_steps and current_step > 0 and state['messages']:
        last_msg = state['messages'][-1]
        if isinstance(last_msg, HumanMessage):
            user_input = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)
            logger.info("[call_model] User message during plan execution: %s", user_input[:100])
            
            # Check if this is a simple acknowledgment or a real question/command
            simple_acks = {'ok', 'okay', 'yes', 'no', 'sure', 'hello', 'hi', 'hey', 'continue', 'proceed', 'go ahead', 'thanks', 'thank you', 'got it'}
            if user_input.strip().lower() in simple_acks:
                # Inject guidance to continue with the plan
                state['messages'].append(SystemMessage(
                    content=f"The user acknowledged. Continue executing the current plan step {current_step}."
                ))
                logger.info("[call_model] Injected continue guidance for acknowledgment")
    
    # Build messages with system prompt and context
    messages_to_send = []
    
    # System prompt
    messages_to_send.append(SystemMessage(content=SYSTEM_PROMPT))
    
    # ── Project profile (tech stack, versions, warnings) ────
    project_profile = state.get('project_profile', '')
    is_first_call = not any(isinstance(m, ToolMessage) for m in state['messages'])
    
    # ── Determine model early so we know if we should inject planning prompt ──
    planning_model = llm_provider.get_planning_model_name()
    using_planning_model = is_first_call and not plan_steps and planning_model
    
    # ── Master Planning Prompt (when Claude is doing the initial analysis) ──
    if using_planning_model:
        messages_to_send.append(SystemMessage(content=MASTER_PLANNING_PROMPT))
        logger.info("[call_model] Injected MASTER_PLANNING_PROMPT (using planning model)")
        
        # Also inject full attached file contents for planning context
        attached_files = state.get('attached_files', {})
        if attached_files:
            files_context = "## User's Open Files (Full Content)\n\nThe user has these files open in their IDE. Use them to understand context, patterns, and code style:\n\n"
            for path, content in list(attached_files.items())[:8]:  # Up to 8 files for planning
                # Truncate very large files but be generous for planning
                if len(content) > 15000:
                    content = content[:15000] + f"\n\n... (truncated, {len(content)} chars total)"
                files_context += f"### {path}\n```\n{content}\n```\n\n"
            messages_to_send.append(SystemMessage(content=files_context))
            logger.info("[call_model] Injected %d attached files for planning", len(attached_files))
    
    if project_profile:
        messages_to_send.append(SystemMessage(
            content=f"## Project Profile\n\n{project_profile}\n\nIMPORTANT: If there are WARNINGs above, address them BEFORE starting the user's task. Fix version conflicts, configuration mismatches, or compatibility issues first."
        ))
        logger.info("[call_model] Injected project profile (%d chars)", len(project_profile))
    elif is_first_call:
        # No profile yet — tell the agent to read config files first
        messages_to_send.append(SystemMessage(
            content="## Project Profile: Not Yet Available\n\nBefore making any changes, read the project's key config files to understand the tech stack:\n- `read_file(\"package.json\")` or `read_file(\"Cargo.toml\")` or `read_file(\"requirements.txt\")`\n- Check framework config, styling config (tailwind.config.*, globals.css)\n- Check for version conflicts between dependencies\n\nThis prevents writing code that's incompatible with the project's environment."
        ))
        logger.info("[call_model] No profile — injected config-read guidance")
    
    # ── Active plan context (created by create_plan tool) ────
    if plan_steps and current_step > 0:
        plan_display = format_plan_for_prompt(plan_steps, current_step)
        messages_to_send.append(SystemMessage(content=plan_display))
        logger.info("[call_model] Injected active plan context (step %d/%d)", 
                    current_step, len(plan_steps))
    
    # Add enriched context as a system message if we have it
    if enriched_context:
        # Cap enriched context to avoid blowing up on its own
        ctx = enriched_context[:80_000] if len(enriched_context) > 80_000 else enriched_context
        context_msg = f"""## Pre-gathered Context (SEMANTIC SEARCH RESULTS)

The following code was found via AI-powered semantic search on the user's codebase.
USE THIS CODE as the primary reference for understanding patterns, conventions, and structure.

{ctx}

---

IMPORTANT: The semantic search above already queried the codebase for relevant code.
- Study the code snippets above to understand existing patterns before creating your plan
- Reference specific file paths, function names, and patterns from this context
- You can call `codebase_search` for ADDITIONAL context if needed, but start with what's above"""
        messages_to_send.append(SystemMessage(content=context_msg))
        logger.info("[call_model] Added context message, total messages: %d", len(messages_to_send))
    else:
        logger.warning("[call_model] No enriched context to add!")
    
    # ── PHASE 1: Inject failure summary (pre-emptive blocking) ──────────────
    failure_summary = await ws_memory.get_failure_summary(workspace_id)
    if failure_summary:
        messages_to_send.append(SystemMessage(content=failure_summary))
        logger.warning("[call_model] Injected failure summary (%d exhausted approaches)",
                      len(memory.get('exhausted_approaches', [])))
    

    
    # Add conversation history (with truncation)
    history = await _truncate_messages(state['messages'])
    
    # ── Inject attached images into the last HumanMessage ────
    # Vision models expect images as part of the message content array.
    attached_images = state.get('attached_images', [])
    if attached_images and history:
        # Find the last HumanMessage in history and make it multimodal
        for i in range(len(history) - 1, -1, -1):
            if isinstance(history[i], HumanMessage):
                text_content = history[i].content
                # Build multimodal content: text + images
                content_parts = [{"type": "text", "text": text_content}]
                for img in attached_images:
                    data_uri = f"data:{img.get('mime_type', 'image/png')};base64,{img['data']}"
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                    })
                history[i] = HumanMessage(content=content_parts)
                logger.info("[call_model] Attached %d images to HumanMessage", len(attached_images))
                break
    
    messages_to_send.extend(history)
    
    total_chars = sum(len(m.content) if isinstance(m.content, str) else len(str(m.content)) for m in messages_to_send)
    
    # Heuristic-based routing (removed Phase 3 intelligent routing for speed/simplicity)
    model_name = _pick_model_name(state['messages'], plan_steps=plan_steps, current_step=current_step)
    logger.info("[call_model] Using %s | %d messages", model_name, len(messages_to_send))
    
    model = llm_provider.get_chat_model(model_name, temperature=0.1)
    model_with_tools = model.bind_tools(ALL_TOOLS)
    
    # Invoke
    try:
        response = await model_with_tools.ainvoke(messages_to_send)
    except Exception as e:
        config = llm_provider.get_config()
        if model_name != config.reasoning_model:
            fallback_model = llm_provider.get_chat_model(config.reasoning_model, temperature=0.1)
            model_with_tools = fallback_model.bind_tools(ALL_TOOLS)
            response = await model_with_tools.ainvoke(messages_to_send)
        else:
            raise
    
    # ── Roo-Code Style: Repetition Detection ──
    if response.tool_calls:
        for tc in response.tool_calls:
            # Check for identical consecutive tool calls
            current_call = f"{tc['name']}:{json.dumps(tc.get('args', {}), sort_keys=True)}"
            if state.get('last_tool_call') == current_call:
                repetition_count = state.get('repetition_count', 0) + 1
                if repetition_count >= 3:
                    return {"messages": [AIMessage(content=f"I've attempted the same tool call `{tc['name']}` 3 times without success. I'll stop to avoid a loop. Please guide me on how to proceed.")]}
            
    # Empty response fallback
    if not response.content and not response.tool_calls:
        return {"messages": [AIMessage(content="I encountered an issue. Let me try a different approach.")]}
    
    return {"messages": [response]}


async def execute_server_tools(state: AgentState) -> dict:
    """Execute tools that run on the server and update agent state."""
    import asyncio
    import inspect
    last_message = state['messages'][-1]
    
    # Identify all server tool calls in this turn
    server_calls = [tc for tc in last_message.tool_calls if tc['name'] in SERVER_TOOL_NAMES]
    if not server_calls:
        return {}

    # Track state updates
    updates = {}
    tool_outputs = []

    async def _run_tool(tool_call):
        tool_name = tool_call['name']
        tool_func = next((t for t in SERVER_TOOLS if t.name == tool_name), None)
        if not tool_func:
            return ToolMessage(content=f"Error: Tool {tool_name} not found", tool_call_id=tool_call['id'])
            
        try:
            # Extract kwargs and inject state if needed
            # For async tools, use coroutine instead of func
            actual_func = tool_func.coroutine if tool_func.coroutine else tool_func.func
            if actual_func is None:
                raise ValueError(f"Tool {tool_name} has no callable function or coroutine")
                
            sig = inspect.signature(actual_func)
            kwargs = tool_call['args'].copy()
            
            # Find if any parameter expects state
            state_param = next(
                (p.name for p in sig.parameters.values() 
                 if p.annotation == Annotated[AgentState, InjectedState] or "InjectedState" in str(p.annotation)),
                None
            )
            if state_param:
                kwargs[state_param] = state
            
            content = await actual_func(**kwargs)
            
            # Check if the output is a plan update (JSON string with plan_steps)
            # Note: We capture these updates to return them in the final dict
            plan_update = None
            if isinstance(content, str) and content.startswith('{') and '"plan_steps"' in content:
                plan_data = json.loads(content)
                plan_update = {
                    'plan_steps': plan_data['plan_steps'],
                    'current_step': plan_data['current_step']
                }
                content = plan_data['status']
                
            return ToolMessage(content=content, tool_call_id=tool_call['id']), plan_update
        except Exception as e:
            logger.error("Server tool %s failed: %s", tool_name, e)
            return ToolMessage(content=f"Error executing {tool_name}: {e}", tool_call_id=tool_call['id']), None

    # Execute all server tools in parallel
    results = await asyncio.gather(*[_run_tool(tc) for tc in server_calls])
    
    for msg, plan_upd in results:
        tool_outputs.append(msg)
        if plan_upd:
            updates.update(plan_upd)
    
    updates["messages"] = tool_outputs
    return updates


# ── Router Logic ──────────────────────────────────────────────────

def agent_router(state: AgentState) -> Literal["server_tools", "pause", "end"]:
    """After agent: route to server tools, pause for IDE, or finish."""
    last_message = state['messages'][-1]
    
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return "end"
    
    # Check if any tool is an IDE tool
    has_server_tools = any(tc['name'] in SERVER_TOOL_NAMES for tc in last_message.tool_calls)
    has_ide_tools = any(tc['name'] in IDE_TOOL_NAMES for tc in last_message.tool_calls)
    
    if has_server_tools:
        return "server_tools"
    if has_ide_tools:
        return "pause"
    return "end"


def post_server_tools_router(state: AgentState) -> Literal["pause", "agent"]:
    """After server tools: check if the same AI message also had IDE tools."""
    # Find the last AIMessage
    for msg in reversed(state['messages']):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            if any(tc['name'] in IDE_TOOL_NAMES for tc in msg.tool_calls):
                return "pause"
            break
    return "agent"


# ── Graph Construction ────────────────────────────────────────────
#
# Simplified flow with tool-based planning:
#
#   enrich → agent ─┬→ server_tools → post_router ─┬→ agent
#                    │                               └→ pause (IDE tools)
#                    ├→ pause (IDE tools only) → END
#                    └→ end → END
#
# The agent manages its own plan via create_plan/update_plan/discard_plan
# tools. No separate classifier, planning node, or continue_plan needed.

def create_agent():
    """Build the LangGraph agent with tool-based planning.
    
    The agent decides when to plan using create_plan/update_plan/discard_plan
    tools. This replaces the old classifier → planning node flow.
    """
    workflow = StateGraph(AgentState)

    # Nodes
    workflow.add_node("enrich", enrich_context)
    workflow.add_node("agent", call_model)
    workflow.add_node("server_tools", execute_server_tools)

    # ── Edges ─────────────────────────────────────────────────
    
    # Entry: pre-enrichment (semantic search, call chains)
    workflow.set_entry_point("enrich")
    
    # After enrichment, go to agent
    workflow.add_edge("enrich", "agent")
    
    # After agent, route based on tool calls
    workflow.add_conditional_edges(
        "agent",
        agent_router,
        {
            "server_tools": "server_tools",
            "pause": END,           # Return to API/IDE for tool execution
            "end": END,
        }
    )
    
    # After server tools, check if we also need to pause for IDE tools
    workflow.add_conditional_edges(
        "server_tools",
        post_server_tools_router,
        {
            "agent": "agent",       # No IDE tools, continue reasoning
            "pause": END,           # Mixed call — pause for IDE tools
        }
    )

    return workflow.compile()


# Global agent instance
forge_agent = create_agent()
