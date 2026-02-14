"""
Forge Search API — Code intelligence for any codebase.

Clean, modular FastAPI application with organized endpoints.
All endpoints are in app/api/ modules.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient

from .storage import store

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

# ── MongoDB Trace Store ───────────────────────────────────────────
_mongo_url = os.getenv("MONGODB_URL", "")
if _mongo_url:
    try:
        mongo_client = AsyncIOMotorClient(_mongo_url)
        mongo_db = mongo_client["forge_traces"]
        traces_collection = mongo_db["traces"]
        logger.info("MongoDB tracing ENABLED (database: forge_traces)")
    except Exception as e:
        logger.error(f"MongoDB init failed: {e}")
        mongo_client = None
        traces_collection = None
else:
    mongo_client = None
    traces_collection = None
    logger.info("MongoDB tracing disabled (set MONGODB_URL to enable)")


async def _ensure_mongo_indexes():
    """Create MongoDB indexes for efficient querying."""
    if traces_collection is None:
        return
    try:
        await traces_collection.create_index("thread_id")
        await traces_collection.create_index("workspace_id")
        await traces_collection.create_index("timestamp")
        logger.info("MongoDB indexes ensured")
    except Exception as e:
        logger.warning("Could not create MongoDB indexes: %s", e)


# ── Application Lifespan ──────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    logger.info("Starting up Forge Search API...")
    await _ensure_mongo_indexes()
    logger.info("✓ Ready to serve requests")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    await store.close_driver()


# ── FastAPI Application ───────────────────────────────────────────

app = FastAPI(
    title="Forge Search API",
    description="Code intelligence API — semantic search, call tracing, impact analysis",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Include API Routers ───────────────────────────────────────────

from .api import (
    health,
    memory,
    llm_config,
    debug,
    auth_endpoints,
    watch,
    search,
    analysis,
    chat,
    map,
)

# Health check (no prefix)
app.include_router(health.router, tags=["health"])

# Memory endpoints
app.include_router(memory.router, prefix="/memory", tags=["memory"])

# LLM configuration
app.include_router(llm_config.router, prefix="/models", tags=["models"])

# Debug tools
app.include_router(debug.router, prefix="/debug", tags=["debug"])

# Authentication
app.include_router(auth_endpoints.router, prefix="/auth", tags=["auth"])

# File watching (no prefix to keep /watch, /scan, DELETE /watch/{id} at root)
app.include_router(watch.router, tags=["watch"])

# Search and indexing (no prefix - /index, /search, /reindex at root)
app.include_router(search.router, tags=["search"])

# Code analysis (no prefix - /trace, /impact at root)
app.include_router(analysis.router, tags=["analysis"])

# Chat (already has prefix="/chat" in router)
app.include_router(chat.router, tags=["chat"])

# Map (no prefix - /map at root)
app.include_router(map.router, tags=["map"])

logger.info("✓ All API routers registered")
logger.info("  - /health (health check)")
logger.info("  - /memory/* (workspace memory)")
logger.info("  - /models/* (LLM config)")
logger.info("  - /debug/* (debug tools)")
logger.info("  - /auth/* (authentication)")
logger.info("  - /watch/* (file watching)")
logger.info("  - /index, /search, /reindex (search & indexing)")
logger.info("  - /trace, /impact (code analysis)")
logger.info("  - /map (project visualization)")
logger.info("  - /chat/* (AI chat)")
