"""
AI Chat — forge-search context + LLM (any provider via LiteLLM).

User asks a question → forge-search finds relevant code → 
LLM reasons about it → returns precise answer.

Supports: Groq, Gemini, Claude, Fireworks, OpenAI, Mistral, etc.
The user never sees API keys. forge-search handles everything.
"""

from __future__ import annotations

import logging
import time

from . import llm as llm_provider

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a senior software engineer analyzing a codebase.
You have been given relevant code snippets, call chains, and impact analysis from a code intelligence system.

RULES:
1. Reference EXACT function names, file paths, and line numbers from the context
2. Be concise and actionable
3. Only reference what's in the provided context — don't guess
4. When suggesting changes, show the exact current code and what to modify"""


async def chat_with_context(
    question: str,
    code_context: str,
    max_tokens: int = 1024,
    temperature: float = 0.1,
) -> dict:
    """Send question + code context to the configured LLM (any provider)."""
    config = llm_provider.get_config()
    model_name = config.reasoning_model

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"## Code Context\n{code_context}\n\n## Question\n{question}"},
    ]

    t0 = time.time()
    try:
        response = await llm_provider.completion(
            messages=messages,
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as e:
        logger.error("LLM API call failed (%s): %s", model_name, e)
        return {"response": f"LLM error ({model_name}): {e}", "tokens": 0, "time_ms": 0, "model": model_name}

    usage = response.get("usage", {}) or {}
    elapsed_ms = (time.time() - t0) * 1000

    return {
        "response": response["choices"][0]["message"]["content"],
        "tokens": usage.get("completion_tokens", 0),
        "time_ms": round(elapsed_ms, 1),
        "model": model_name,
    }


def build_context_from_results(search_results: list, trace_data: dict = None, impact_data: dict = None) -> str:
    """Build LLM context from forge-search API results."""
    parts = []

    if search_results:
        parts.append("## Relevant Code\n")
        for r in search_results[:8]:
            sym_type = r.get('symbol_type') or r.get('kind', 'symbol')
            parts.append(f"### {sym_type} `{r['name']}` — `{r['file_path']}:{r['start_line']}-{r['end_line']}`")
            if r.get("signature"):
                parts.append(f"Signature: `{r['signature']}`")
            parts.append(f"```\n{r['content']}\n```")
            if r.get("related"):
                rels = [f"`{rel['name']}` ({rel['relationship']})" for rel in r["related"][:5]]
                parts.append(f"Related: {', '.join(rels)}")
            parts.append("")

    if trace_data and trace_data.get("nodes"):
        parts.append(f"## Call Chain: `{trace_data['root']}`\n")
        for e in trace_data.get("edges", [])[:15]:
            if e["type"] == "CALLS":
                parts.append(f"  `{e['from']}` → `{e['to']}`")
        parts.append("")

    if impact_data and impact_data.get("total_affected", 0) > 0:
        parts.append(f"## Impact: {impact_data['total_affected']} symbols affected\n")
        for f in impact_data.get("by_file", [])[:5]:
            syms = ", ".join([f"`{s['name']}`" for s in f["symbols"][:5]])
            parts.append(f"  `{f['file_path']}`: {syms}")
        parts.append("")

    return "\n".join(parts)
