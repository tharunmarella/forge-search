"""
Mermaid diagram rendering via mermaid.ink.

Renders ```mermaid code blocks to PNG images and replaces them with
inline data URI images. This moves rendering from the IDE (blocking)
to the server (async + cached).

Usage:
    answer = await render_mermaid_blocks(llm_answer)
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import re
from functools import lru_cache
from typing import Optional
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────

MERMAID_INK_URL = "https://mermaid.ink/img"
RENDER_TIMEOUT = 8.0  # seconds per diagram
MAX_CONCURRENT = 4    # max parallel renders
CACHE_SIZE = 128      # LRU cache entries

# Dark theme injection (prepended to all diagrams)
DARK_THEME_INIT = "%%{init: {'theme':'dark', 'themeVariables': {'primaryColor': '#89b4fa', 'primaryTextColor': '#cdd6f4', 'primaryBorderColor': '#585b70', 'lineColor': '#a6adc8', 'secondaryColor': '#313244', 'tertiaryColor': '#1e1e2e', 'background': '#1e1e2e'}}}%%\n"

# Regex to find ```mermaid ... ``` blocks
MERMAID_BLOCK_RE = re.compile(
    r"```mermaid\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE
)


# ── Cache ─────────────────────────────────────────────────────────

# In-memory cache: hash(source) -> base64 PNG
# Using a simple dict + manual LRU logic since functools.lru_cache
# doesn't work well with async functions
_cache: dict[str, str] = {}
_cache_order: list[str] = []


def _cache_get(key: str) -> Optional[str]:
    """Get from cache, update LRU order."""
    if key in _cache:
        _cache_order.remove(key)
        _cache_order.append(key)
        return _cache[key]
    return None


def _cache_set(key: str, value: str):
    """Set in cache, evict oldest if needed."""
    if key in _cache:
        _cache_order.remove(key)
    _cache[key] = value
    _cache_order.append(key)
    
    # Evict oldest entries
    while len(_cache_order) > CACHE_SIZE:
        oldest = _cache_order.pop(0)
        _cache.pop(oldest, None)


def _source_hash(source: str) -> str:
    """Hash mermaid source for cache key."""
    return hashlib.sha256(source.encode()).hexdigest()[:16]


# ── Rendering ─────────────────────────────────────────────────────

async def _render_one(source: str, client: httpx.AsyncClient) -> Optional[str]:
    """
    Render a single mermaid diagram via mermaid.ink.
    
    Returns base64-encoded PNG, or None on failure.
    """
    # Check cache first
    cache_key = _source_hash(source)
    cached = _cache_get(cache_key)
    if cached:
        logger.debug("Mermaid cache hit: %s", cache_key[:8])
        return cached
    
    # Inject dark theme if not already present
    if "%%{init:" not in source:
        source = DARK_THEME_INIT + source
    
    # mermaid.ink expects base64-encoded diagram in URL
    encoded = base64.urlsafe_b64encode(source.encode()).decode()
    url = f"{MERMAID_INK_URL}/{encoded}"
    
    try:
        resp = await client.get(url, timeout=RENDER_TIMEOUT)
        resp.raise_for_status()
        
        # Check content type
        content_type = resp.headers.get("content-type", "")
        if "image" not in content_type:
            logger.warning("Mermaid render returned non-image: %s", content_type)
            return None
        
        # Convert to base64
        png_base64 = base64.b64encode(resp.content).decode()
        
        # Cache it
        _cache_set(cache_key, png_base64)
        logger.info("Mermaid rendered: %d bytes, cached as %s", len(resp.content), cache_key[:8])
        
        return png_base64
        
    except httpx.TimeoutException:
        logger.warning("Mermaid render timeout after %.1fs", RENDER_TIMEOUT)
        return None
    except httpx.HTTPStatusError as e:
        logger.warning("Mermaid render failed: %s", e)
        return None
    except Exception as e:
        logger.error("Mermaid render error: %s", e)
        return None


async def render_mermaid_blocks(text: str) -> str:
    """
    Find all ```mermaid blocks in text, render to PNG, replace with data URI images.
    
    If rendering fails for a block, the original ```mermaid block is preserved
    (IDE will display it as a syntax-highlighted code block).
    
    Args:
        text: Markdown text potentially containing mermaid blocks
        
    Returns:
        Text with mermaid blocks replaced by inline images
    """
    if not text or "```mermaid" not in text.lower():
        return text
    
    # Find all mermaid blocks
    matches = list(MERMAID_BLOCK_RE.finditer(text))
    if not matches:
        return text
    
    logger.info("Found %d mermaid block(s) to render", len(matches))
    
    # Extract sources
    sources = [m.group(1).strip() for m in matches]
    
    # Render all in parallel (with concurrency limit)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    
    async def render_with_limit(source: str, client: httpx.AsyncClient) -> Optional[str]:
        async with semaphore:
            return await _render_one(source, client)
    
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            *[render_with_limit(s, client) for s in sources],
            return_exceptions=True
        )
    
    # Replace blocks with rendered images (in reverse order to preserve positions)
    result_text = text
    for match, png_b64 in zip(reversed(matches), reversed(results)):
        if isinstance(png_b64, str) and png_b64:
            # Success: replace with data URI image
            image_md = f"![diagram](data:image/png;base64,{png_b64})"
            result_text = result_text[:match.start()] + image_md + result_text[match.end():]
        # else: keep original block (render failed)
    
    return result_text


# ── Utility ───────────────────────────────────────────────────────

def clear_cache():
    """Clear the mermaid render cache."""
    _cache.clear()
    _cache_order.clear()
    logger.info("Mermaid cache cleared")


def cache_stats() -> dict:
    """Get cache statistics."""
    return {
        "entries": len(_cache),
        "max_size": CACHE_SIZE,
    }
