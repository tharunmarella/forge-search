"""
Code embeddings via Jina AI API.

Uses jinaai/jina-embeddings-v2-base-code hosted by Jina.
No local model, no torch, no GPU. Just an API call.

768 dims, trained on 5.5M code-text pairs, 161 programming languages.
Free tier: 1M tokens/month.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Sequence

import httpx

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────

JINA_API_KEY = os.getenv("JINA_API_KEY", "")
JINA_MODEL = os.getenv("JINA_MODEL", "jina-embeddings-v2-base-code")
JINA_URL = "https://api.jina.ai/v1/embeddings"
BATCH_SIZE = 64
MAX_TEXT_LENGTH = 8000
DIMENSIONS = 768


def get_dimensions() -> int:
    return DIMENSIONS


def _truncate(text: str, max_len: int = MAX_TEXT_LENGTH) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\n... [truncated]"


# ── Jina API call ─────────────────────────────────────────────────

async def _call_jina(texts: list[str]) -> list[list[float]]:
    """Call Jina's embedding API."""
    if not JINA_API_KEY:
        raise RuntimeError("JINA_API_KEY not set. Get one at https://jina.ai/embeddings/")

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            JINA_URL,
            headers={
                "Authorization": f"Bearer {JINA_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": JINA_MODEL,
                "input": texts,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        # Sort by index to preserve order
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_data]


# ── Public API ────────────────────────────────────────────────────

async def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    text = _truncate(text)
    results = await _call_jina([text])
    return results[0]


async def embed_batch(texts: Sequence[str]) -> list[list[float]]:
    """Embed a batch of texts. Splits into sub-batches for API limits."""
    if not texts:
        return []

    truncated = [_truncate(t) for t in texts]
    all_embeddings: list[list[float]] = []
    total_batches = (len(truncated) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(total_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(truncated))
        batch = truncated[start:end]

        logger.info("Embedding batch %d/%d (%d texts)", batch_idx + 1, total_batches, len(batch))
        t0 = time.monotonic()

        batch_embeddings = await _call_jina(batch)

        logger.info("Batch %d/%d done in %.2fs", batch_idx + 1, total_batches, time.monotonic() - t0)
        all_embeddings.extend(batch_embeddings)

    return all_embeddings
