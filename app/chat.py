"""
AI Chat — forge-search context + Groq Kimi-K2.

User asks a question → forge-search finds relevant code → 
Kimi-K2 reasons about it → returns precise answer.

The user never sees API keys. forge-search handles everything.
"""

from __future__ import annotations

import logging
import os
import time

import httpx

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "moonshotai/kimi-k2-instruct-0905")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

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
    """Send question + code context to Groq Kimi-K2."""
    if not GROQ_API_KEY:
        return {"response": "GROQ_API_KEY not configured", "tokens": 0, "time_ms": 0, "model": "none"}

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"## Code Context\n{code_context}\n\n## Question\n{question}"},
    ]

    t0 = time.time()
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )

    if resp.status_code != 200:
        return {"response": f"Groq error: {resp.text[:300]}", "tokens": 0, "time_ms": 0, "model": GROQ_MODEL}

    data = resp.json()
    usage = data.get("usage", {})
    elapsed_ms = (time.time() - t0) * 1000

    return {
        "response": data["choices"][0]["message"]["content"],
        "tokens": usage.get("completion_tokens", 0),
        "time_ms": round(elapsed_ms, 1),
        "model": GROQ_MODEL,
    }


def build_context_from_results(search_results: list, trace_data: dict = None, impact_data: dict = None) -> str:
    """Build LLM context from forge-search API results."""
    parts = []

    if search_results:
        parts.append("## Relevant Code\n")
        for r in search_results[:8]:
            parts.append(f"### {r['symbol_type']} `{r['name']}` — `{r['file_path']}:{r['start_line']}-{r['end_line']}`")
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
