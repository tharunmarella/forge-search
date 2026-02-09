"""
Gemini text-embedding-004 client with batching.

Handles embedding generation for both indexing (batch) and search (single query).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Sequence

import google.generativeai as genai

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────

MODEL_NAME = "models/text-embedding-004"
DIMENSIONS = 768
BATCH_SIZE = 100  # Max texts per Gemini embedding request
MAX_TEXT_LENGTH = 8000  # Truncate texts to stay within token limits


# ── Client ────────────────────────────────────────────────────────

_configured = False


def _ensure_configured():
    global _configured
    if not _configured:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is required for embeddings"
            )
        genai.configure(api_key=api_key)
        _configured = True


def _truncate(text: str, max_len: int = MAX_TEXT_LENGTH) -> str:
    """Truncate text to max characters, keeping the beginning (signature + body start)."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\n... [truncated]"


# ── Single embed ──────────────────────────────────────────────────

async def embed_query(text: str) -> list[float]:
    """Embed a single query string. Returns a 768-dim vector."""
    _ensure_configured()
    text = _truncate(text)

    # google-generativeai's embed_content is sync, run in executor
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: genai.embed_content(
            model=MODEL_NAME,
            content=text,
            task_type="retrieval_query",
            output_dimensionality=DIMENSIONS,
        ),
    )
    return result["embedding"]


# ── Batch embed ───────────────────────────────────────────────────

async def embed_batch(texts: Sequence[str]) -> list[list[float]]:
    """
    Embed a batch of texts (for indexing).
    Automatically splits into sub-batches of BATCH_SIZE.

    Returns list of embedding vectors in the same order as input.
    """
    _ensure_configured()

    if not texts:
        return []

    # Truncate all texts
    truncated = [_truncate(t) for t in texts]

    all_embeddings: list[list[float]] = []
    total_batches = (len(truncated) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(total_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(truncated))
        batch = truncated[start:end]

        logger.info(
            "Embedding batch %d/%d (%d texts)",
            batch_idx + 1, total_batches, len(batch),
        )
        t0 = time.monotonic()

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda b=batch: genai.embed_content(
                model=MODEL_NAME,
                content=b,
                task_type="retrieval_document",
                output_dimensionality=DIMENSIONS,
            ),
        )

        elapsed = time.monotonic() - t0
        batch_embeddings = result["embedding"]
        all_embeddings.extend(batch_embeddings)

        logger.info(
            "Batch %d/%d done in %.1fs (%d embeddings)",
            batch_idx + 1, total_batches, elapsed, len(batch_embeddings),
        )

        # Small delay between batches to avoid rate limiting
        if batch_idx < total_batches - 1:
            await asyncio.sleep(0.2)

    return all_embeddings


async def embed_texts_to_map(texts: list[str]) -> dict[str, list[float]]:
    """
    Given a list of texts, return a mapping from text -> embedding.
    Deduplicates texts before embedding to save API calls.
    """
    unique_texts = list(set(texts))
    if not unique_texts:
        return {}

    embeddings = await embed_batch(unique_texts)

    return dict(zip(unique_texts, embeddings))
