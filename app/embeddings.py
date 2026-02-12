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
import re
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


def is_embeddable(text: str) -> bool:
    """Check if text is suitable for embedding.
    
    Filters out content that will cause Voyage API 400 errors:
    - Minified JS/CSS (very long lines, no structure)
    - Binary-looking content
    - Empty or whitespace-only
    """
    if not text or not text.strip():
        return False
    
    # Very short content is not useful
    if len(text.strip()) < 20:
        return False
    
    # Detect minified code: if average line length > 500 chars, it's likely minified
    lines = text.split('\n')
    if lines:
        avg_line_len = sum(len(l) for l in lines) / len(lines)
        if avg_line_len > 500 and len(lines) < 10:
            return False
    
    # Detect minified: very few newlines relative to total length
    if len(text) > 1000 and text.count('\n') < len(text) / 1000:
        return False
    
    # Detect binary/garbage content
    non_printable = sum(1 for c in text[:1000] if ord(c) < 32 and c not in '\n\r\t')
    if non_printable > 50:
        return False
    
    return True


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
        
        if resp.status_code == 400:
            # Log the actual error from Voyage for debugging
            body = resp.text
            logger.warning("Voyage 400 error (batch of %d): %s", len(texts), body[:300])
            raise httpx.HTTPStatusError(
                f"Voyage 400: {body[:200]}",
                request=resp.request,
                response=resp,
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
    """Embed a batch of texts (for indexing).
    
    Resilient: if a batch fails (400 error from Voyage), falls back to
    embedding texts individually so one bad text doesn't kill the whole batch.
    Returns None for texts that fail to embed.
    """
    if not texts:
        return []

    # Pre-filter: skip texts that are clearly not embeddable
    filtered = []
    index_map = []  # Maps filtered index → original index
    for i, t in enumerate(texts):
        if is_embeddable(t):
            filtered.append(_truncate(t))
            index_map.append(i)
        else:
            logger.debug("Skipping non-embeddable text %d (%d chars, %.0f avg line)", 
                        i, len(t), sum(len(l) for l in t.split('\n')) / max(t.count('\n'), 1))
    
    if not filtered:
        logger.warning("All %d texts filtered as non-embeddable", len(texts))
        return [None] * len(texts)  # type: ignore
    
    logger.info("Embedding %d/%d texts (filtered %d non-embeddable)", 
                len(filtered), len(texts), len(texts) - len(filtered))
    
    # Embed in batches with fallback
    filtered_embeddings: list[list[float] | None] = [None] * len(filtered)
    total_batches = (len(filtered) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(total_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(filtered))
        batch = filtered[start:end]

        logger.info("Embedding batch %d/%d (%d texts)", batch_idx + 1, total_batches, len(batch))
        t0 = time.monotonic()

        try:
            batch_embeddings = await _call_voyage(batch, input_type="document")
            for j, emb in enumerate(batch_embeddings):
                filtered_embeddings[start + j] = emb
            logger.info("Batch %d/%d done in %.2fs", batch_idx + 1, total_batches, time.monotonic() - t0)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                # Batch failed — fall back to individual embedding
                logger.warning("Batch %d failed (400), falling back to individual texts (%d items)", 
                             batch_idx + 1, len(batch))
                for j, text in enumerate(batch):
                    try:
                        individual = await _call_voyage([text], input_type="document")
                        filtered_embeddings[start + j] = individual[0]
                    except Exception as inner_e:
                        logger.warning("Individual embed failed for text %d (%d chars): %s", 
                                     start + j, len(text), str(inner_e)[:100])
                        filtered_embeddings[start + j] = None
            else:
                raise  # Re-raise non-400 errors

    # Map back to original indices
    result: list[list[float] | None] = [None] * len(texts)  # type: ignore
    for fi, oi in enumerate(index_map):
        result[oi] = filtered_embeddings[fi]
    
    embedded_count = sum(1 for e in result if e is not None)
    logger.info("Embedded %d/%d texts successfully", embedded_count, len(texts))
    
    return result  # type: ignore
