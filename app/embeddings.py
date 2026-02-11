"""
Code embeddings via Voyage AI API.

Uses voyage-code-3 - purpose-built for code retrieval.
1024 dims, trained specifically for code search.
Free tier: 200M tokens/month.
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

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")
VOYAGE_MODEL = os.getenv("VOYAGE_MODEL", "voyage-code-3")
VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"
BATCH_SIZE = 128  # Voyage supports up to 128 texts per request
MAX_TEXT_LENGTH = 16000  # voyage-code-3 supports up to 16K tokens
DIMENSIONS = 1024  # voyage-code-3 outputs 1024 dimensions


def get_dimensions() -> int:
    return DIMENSIONS


def _truncate(text: str, max_len: int = MAX_TEXT_LENGTH) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\n... [truncated]"


# ── Voyage API call ─────────────────────────────────────────────────

async def _call_voyage(texts: list[str], input_type: str = "document") -> list[list[float]]:
    """Call Voyage AI's embedding API.
    
    Args:
        texts: List of texts to embed
        input_type: Either "document" (for indexing) or "query" (for search queries)
    """
    if not VOYAGE_API_KEY:
        raise RuntimeError("VOYAGE_API_KEY not set. Get one at https://dash.voyageai.com/")

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            VOYAGE_URL,
            headers={
                "Authorization": f"Bearer {VOYAGE_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": VOYAGE_MODEL,
                "input": texts,
                "input_type": input_type,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        # Sort by index to preserve order
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_data]


# ── Public API ────────────────────────────────────────────────────

async def embed_query(text: str) -> list[float]:
    """Embed a single query string (for search)."""
    text = _truncate(text)
    results = await _call_voyage([text], input_type="query")
    return results[0]


async def embed_batch(texts: Sequence[str]) -> list[list[float]]:
    """Embed a batch of texts (for indexing). Splits into sub-batches for API limits."""
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

        batch_embeddings = await _call_voyage(batch, input_type="document")

        logger.info("Batch %d/%d done in %.2fs", batch_idx + 1, total_batches, time.monotonic() - t0)
        all_embeddings.extend(batch_embeddings)

    return all_embeddings
