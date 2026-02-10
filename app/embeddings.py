"""
Code embeddings using jinaai/jina-embeddings-v2-base-code.

Runs locally on CPU via sentence-transformers.
161M params, 768 dims, trained on 5.5M code-text pairs.
No API key, no GPU, no network needed.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Sequence

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "jinaai/jina-embeddings-v2-base-code")
BATCH_SIZE = 64
MAX_TEXT_LENGTH = 8000

# ── Model (lazy-loaded) ──────────────────────────────────────────

_model = None
_dimensions: int | None = None


def _get_model():
    global _model, _dimensions
    if _model is None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", MODEL_NAME)
        t0 = time.monotonic()
        _model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
        _dimensions = _model.get_sentence_embedding_dimension()
        logger.info("Model loaded in %.1fs (dims=%d)", time.monotonic() - t0, _dimensions)
    return _model


def get_dimensions() -> int:
    global _dimensions
    if _dimensions is not None:
        return _dimensions
    _get_model()
    return _dimensions  # type: ignore


def _truncate(text: str, max_len: int = MAX_TEXT_LENGTH) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\n... [truncated]"


# ── Public API ────────────────────────────────────────────────────

async def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    text = _truncate(text)
    model = _get_model()
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: model.encode([text], batch_size=1, show_progress_bar=False, normalize_embeddings=True).tolist()[0],
    )
    return result


async def embed_batch(texts: Sequence[str]) -> list[list[float]]:
    """Embed a batch of texts (for indexing). Splits into sub-batches."""
    if not texts:
        return []

    truncated = [_truncate(t) for t in texts]
    model = _get_model()
    all_embeddings: list[list[float]] = []
    total_batches = (len(truncated) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(total_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(truncated))
        batch = truncated[start:end]

        logger.info("Embedding batch %d/%d (%d texts)", batch_idx + 1, total_batches, len(batch))
        t0 = time.monotonic()

        loop = asyncio.get_event_loop()
        batch_embeddings = await loop.run_in_executor(
            None,
            lambda b=batch: model.encode(b, batch_size=BATCH_SIZE, show_progress_bar=False, normalize_embeddings=True).tolist(),
        )

        logger.info("Batch %d/%d done in %.2fs", batch_idx + 1, total_batches, time.monotonic() - t0)
        all_embeddings.extend(batch_embeddings)

    return all_embeddings
