"""
Forge Search API — Code intelligence for any codebase.

Auth:
  GET  /auth/github          — Start GitHub OAuth
  GET  /auth/github/callback — GitHub OAuth callback
  GET  /auth/google          — Start Google OAuth
  GET  /auth/google/callback — Google OAuth callback
  GET  /auth/me              — Current user info

Code Intelligence:
  POST /index    — Parse + embed + store code symbols
  POST /search   — Semantic code search
  POST /trace    — Deep call chain traversal
  POST /impact   — Blast radius analysis
  POST /chat     — AI chat (search context + any LLM via LiteLLM)
  POST /watch    — Intelligent file watcher
  POST /scan     — One-shot intelligent scan
  GET  /health   — Healthcheck

LLM Model Management:
  GET  /models       — List providers, models, current config
  POST /models/set   — Switch reasoning/tool model at runtime
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import os
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()  # Load .env file for local dev (no-op if missing)

import json

# ── MongoDB Tracing ──────────────────────────────────────────────
# All traces (requests, responses, execution details) are logged to MongoDB
# for analytics, debugging, and replay.
#   - @traceable on sub-functions (pre-enrichment, doc lookup, server tools)

from fastapi import FastAPI, HTTPException, Depends

from fastapi.responses import RedirectResponse

from .storage import store
from .core import embeddings, chat, agent
from .core import llm as llm_provider
from .utils import watcher, auth, mermaid
from .api import chat as chat_api
from .intelligence.phase1 import workspace_memory as ws_memory
from .models import (
    IndexRequest, IndexResponse,
    SearchRequest, SearchResponse, SearchResult, RelatedSymbol,
    ReindexRequest,
    HealthResponse,
    TraceRequest, TraceResponse, TraceNode, TraceEdge,
    ImpactRequest, ImpactResponse, AffectedFile, AffectedSymbol,
    WatchRequest, WatchResponse,
    ProviderInfo, ActiveModelsResponse, SetModelRequest, SetModelResponse,
)
from .parser import parse_file, SymbolDef, SymbolRef, FileParseResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── MongoDB Trace Store ───────────────────────────────────────────
from motor.motor_asyncio import AsyncIOMotorClient

_mongo_url = os.getenv("MONGODB_URL", "")
if _mongo_url:
    try:
        mongo_client = AsyncIOMotorClient(_mongo_url)
        mongo_db = mongo_client["forge_traces"]
        traces_collection = mongo_db["traces"]
        logger.info(f"MongoDB tracing ENABLED (database: forge_traces)")
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
        await traces_collection.create_index([("thread_id", 1), ("timestamp", 1)])
        logger.info("MongoDB indexes created")
    except Exception as e:
        logger.warning(f"MongoDB index creation failed: {e}")


# ── Pending auth tokens (for polling-based OAuth) ──────────────────
# When IDE initiates OAuth with state=poll-{session_id}, the callback
# stores the token here. IDE polls GET /auth/poll/{session_id}.
_pending_auth: dict[str, dict] = {}  # session_id -> {token, user_info, expires}


# ── Context-enriched embedding text ──────────────────────────────

def _build_enriched_text(
    defn: SymbolDef,
    calls_map: dict[str, list[str]],
    called_by_map: dict[str, list[str]],
    imports: list[str],
) -> str:
    """
    Build a context-enriched text for a symbol to embed.

    Instead of embedding raw source code, we prepend structured context
    so the embedding captures *what this code means in the system*:
      - Where it lives (file path, module)
      - What it is (kind, parent type)
      - What it connects to (calls, called by)
      - Its actual code

    This is how a developer reads code: with surrounding context.
    """
    parts: list[str] = []

    # Location context
    parts.append(f"[File] {defn.file_path}")

    # Module hint from file path (e.g., "database/connection_manager")
    module = defn.file_path.rsplit(".", 1)[0].replace("/src/", "/").replace("\\", "/")
    parts.append(f"[Module] {module}")

    # Symbol identity
    parts.append(f"[{defn.kind.capitalize()}] {defn.name}")

    if defn.parent:
        parts.append(f"[Parent] {defn.parent}")

    # Signature (the most important single line)
    if defn.signature:
        parts.append(f"[Signature] {defn.signature}")

    # Relationship context -- what does this symbol connect to?
    outgoing = calls_map.get(defn.name)
    if outgoing:
        # Deduplicate and limit
        unique_calls = list(dict.fromkeys(outgoing))[:15]
        parts.append(f"[Calls] {', '.join(unique_calls)}")

    incoming = called_by_map.get(defn.name)
    if incoming:
        unique_callers = list(dict.fromkeys(incoming))[:10]
        parts.append(f"[Called by] {', '.join(unique_callers)}")

    # Key imports for module-level context
    if imports:
        # Keep short -- just the first few meaningful imports
        short_imports = [imp.split("::")[-1].split(".")[-1] for imp in imports[:8]]
        parts.append(f"[Imports] {', '.join(short_imports)}")

    # The actual source code
    parts.append("")
    parts.append(defn.content)

    return "\n".join(parts)


def _build_context_maps(
    parse_result: FileParseResult,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """
    From a file's parse result, build:
      - calls_map:     {caller_name: [callee_name, ...]}
      - called_by_map: {callee_name: [caller_name, ...]}
    """
    calls_map: dict[str, list[str]] = {}
    called_by_map: dict[str, list[str]] = {}

    # Set of defined symbol names in this file for filtering
    defined_names = {d.name for d in parse_result.definitions}

    for ref in parse_result.references:
        if not ref.context_name:
            continue
        # Only include if both caller and callee are defined in this file
        # (cross-file relationships are handled by the graph layer)
        caller = ref.context_name
        callee = ref.name

        calls_map.setdefault(caller, []).append(callee)

        if callee in defined_names:
            called_by_map.setdefault(callee, []).append(caller)

    return calls_map, called_by_map


# ── App lifecycle ─────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown hooks."""
    logger.info("Starting Forge Search API...")
    await store.ensure_schema()
    await auth.ensure_user_table()
    await _ensure_mongo_indexes()
    llm_provider.log_provider_status()
    logger.info("Store ready (auth enabled)")
    yield
    logger.info("Shutting down...")
    await store.close_driver()


app = FastAPI(
    title="Forge Search API",
    description="Code intelligence API — semantic search, call tracing, impact analysis",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(chat_api.router)

# ── POST /index ───────────────────────────────────────────────────

@app.post("/index", response_model=IndexResponse)
async def index_files(req: IndexRequest):
    """
    Index source files into the code store.

    Incremental: only re-indexes files whose content hash has changed.
    """
    t0 = time.monotonic()
    workspace_id = req.workspace_id

    total_nodes = 0
    total_rels = 0
    total_embeddings = 0
    files_indexed = 0

    for file_payload in req.files:
        try:
            # Check if file has changed
            existing_hash = await store.get_file_hash(workspace_id, file_payload.path)

            # Parse the file
            parse_result = parse_file(file_payload.path, file_payload.content)

            if existing_hash == parse_result.content_hash:
                logger.debug("Skipping unchanged file: %s", file_payload.path)
                continue

            # Build context maps from references (what calls what)
            calls_map, called_by_map = _build_context_maps(parse_result)

            # Build context-enriched text for each definition
            # This is what gets embedded -- raw code + structured context
            enriched_texts: dict[str, str] = {}  # defn uid -> enriched text
            for defn in parse_result.definitions:
                uid = f"{defn.file_path}:{defn.name}:{defn.start_line}"
                enriched_texts[uid] = _build_enriched_text(
                    defn, calls_map, called_by_map, parse_result.imports,
                )

            # FALLBACK: If tree-sitter found no definitions (e.g. TSX/JSX),
            # embed the file as a whole so it's still searchable
            if not enriched_texts and file_payload.content.strip():
                file_uid = f"{file_payload.path}:__file__:0"
                # Build a minimal enriched text from the raw file
                truncated = file_payload.content[:5000]
                ext = file_payload.path.rsplit(".", 1)[-1] if "." in file_payload.path else "unknown"
                enriched_texts[file_uid] = (
                    f"File: {file_payload.path}\n"
                    f"Type: {ext}\n"
                    f"---\n{truncated}"
                )
                logger.info("Fallback: embedding whole file %s (no symbols extracted)", 
                           file_payload.path)

            # Generate embeddings from enriched text
            if enriched_texts:
                texts_list = list(enriched_texts.values())
                uids_list = list(enriched_texts.keys())
                raw_embeddings = await embeddings.embed_batch(texts_list)
                # Map uid -> embedding for graph storage
                embedding_map = dict(zip(uids_list, raw_embeddings))
                # Only count successful (non-None) embeddings
                total_embeddings += sum(1 for v in embedding_map.values() if v is not None)
            else:
                embedding_map = {}

            # Store in Neo4j (pass enriched texts for storage too)
            stats = await store.index_file_result(
                workspace_id, parse_result, embedding_map, enriched_texts
            )
            total_nodes += stats["nodes_created"]
            total_rels += stats["relationships_created"]

            # Build call graph edges from references
            if parse_result.references:
                await store.build_call_edges(workspace_id, parse_result.references)
                total_rels += len(parse_result.references)

            # Build import edges
            if parse_result.imports:
                await store.build_import_edges(
                    workspace_id, file_payload.path, parse_result.imports
                )

            files_indexed += 1

        except Exception as e:
            logger.error("Failed to index %s: %s", file_payload.path, e, exc_info=True)
            continue

    elapsed = (time.monotonic() - t0) * 1000

    logger.info(
        "Indexed workspace=%s: %d files, %d nodes, %d rels, %d embeddings in %.0fms",
        workspace_id, files_indexed, total_nodes, total_rels, total_embeddings, elapsed,
    )

    return IndexResponse(
        workspace_id=workspace_id,
        files_indexed=files_indexed,
        nodes_created=total_nodes,
        relationships_created=total_rels,
        embeddings_generated=total_embeddings,
        index_time_ms=round(elapsed, 1),
    )


# ── POST /search ──────────────────────────────────────────────────

@app.post("/search", response_model=SearchResponse)
async def search_code(req: SearchRequest):
    """
    Semantic + graph search.

    1. Embed the query via Gemini
    2. Vector similarity search on Neo4j symbol nodes
    3. Expand results by 1 hop (callers, callees, related structs)
    4. Return enriched results
    """
    t0 = time.monotonic()

    try:
        query_embedding = await embeddings.embed_query(req.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    # Vector search
    try:
        raw_results = await store.vector_search(
            req.workspace_id, query_embedding, req.top_k
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector search failed: {e}")

    if not raw_results:
        elapsed = (time.monotonic() - t0) * 1000
        return SearchResponse(
            results=[],
            query=req.query,
            workspace_id=req.workspace_id,
            search_time_ms=round(elapsed, 1),
        )

    # Expand neighbors for graph context
    symbol_uids = [r["symbol"]["uid"] for r in raw_results]
    try:
        neighbors = await store.expand_neighbors(req.workspace_id, symbol_uids)
    except Exception as e:
        logger.warning("Neighbor expansion failed: %s", e)
        neighbors = []

    # Build neighbor lookup
    neighbor_map: dict[str, list[RelatedSymbol]] = {}
    for n in neighbors:
        for uid in symbol_uids:
            key = uid
            if key not in neighbor_map:
                neighbor_map[key] = []
            neighbor_map[key].append(RelatedSymbol(
                name=n["name"],
                file_path=n["file_path"] or "",
                relationship=n["rel_type"],
                signature=n.get("signature", ""),
            ))

    # Assemble response
    results: list[SearchResult] = []
    for r in raw_results:
        sym = r["symbol"]
        score = r["score"]
        uid = sym["uid"]

        related = neighbor_map.get(uid, [])

        # Deduplicate related symbols
        seen = set()
        deduped_related = []
        for rel in related:
            key = (rel.name, rel.relationship)
            if key not in seen:
                seen.add(key)
                deduped_related.append(rel)

        results.append(SearchResult(
            file_path=sym["file_path"] or "",
            name=sym["name"] or "",
            symbol_type=sym["kind"] or "unknown",
            signature=sym["signature"] or "",
            content=sym["content"] or "",
            start_line=sym["start_line"] or 0,
            end_line=sym["end_line"] or 0,
            score=round(score, 4),
            related=deduped_related[:10],
        ))

    # Get workspace stats
    stats = await store.get_workspace_stats(req.workspace_id)
    elapsed = (time.monotonic() - t0) * 1000

    logger.info(
        "Search workspace=%s query=%r -> %d results in %.0fms",
        req.workspace_id, req.query, len(results), elapsed,
    )

    return SearchResponse(
        results=results,
        query=req.query,
        workspace_id=req.workspace_id,
        total_nodes=stats.get("symbols", 0),
        search_time_ms=round(elapsed, 1),
    )


# ── POST /reindex ─────────────────────────────────────────────────

@app.post("/reindex", response_model=IndexResponse)
async def reindex_workspace(req: ReindexRequest):
    """Force a full re-index by clearing the workspace first."""
    await store.clear_workspace(req.workspace_id)
    # Return empty response -- the caller should follow up with /index
    return IndexResponse(
        workspace_id=req.workspace_id,
        files_indexed=0,
        nodes_created=0,
        relationships_created=0,
        embeddings_generated=0,
        index_time_ms=0,
    )


# ── POST /trace ───────────────────────────────────────────────────

@app.post("/trace", response_model=TraceResponse)
async def trace_symbol(req: TraceRequest):
    """
    Deep call-chain traversal.

    Walks the CALLS graph upstream and/or downstream from a symbol
    to build the full execution flow — like a developer tracing through
    the code to understand "how does this work end to end?"

    Returns a subgraph of nodes and edges.
    """
    t0 = time.monotonic()

    result = await store.trace_call_chain(
        req.workspace_id,
        req.symbol_name,
        direction=req.direction,
        max_depth=req.max_depth,
    )

    elapsed = (time.monotonic() - t0) * 1000
    logger.info(
        "Trace workspace=%s symbol=%r direction=%s -> %d nodes, %d edges in %.0fms",
        req.workspace_id, req.symbol_name, req.direction,
        len(result["nodes"]), len(result["edges"]), elapsed,
    )

    return TraceResponse(
        root=result["root"],
        nodes=[TraceNode(**n) for n in result["nodes"]],
        edges=[TraceEdge(**e) for e in result["edges"]],
        depth_reached=result["depth_reached"],
    )


# ── POST /impact ──────────────────────────────────────────────────

@app.post("/impact", response_model=ImpactResponse)
async def impact_analysis(req: ImpactRequest):
    """
    Blast radius analysis.

    "If I change this symbol, what else is affected?"

    Walks upstream through CALLS, BELONGS_TO, and file IMPORTS
    to find every symbol and file that depends on the target.
    Groups results by file for a developer-friendly view.
    """
    t0 = time.monotonic()

    result = await store.impact_analysis(
        req.workspace_id,
        req.symbol_name,
        max_depth=req.max_depth,
    )

    elapsed = (time.monotonic() - t0) * 1000
    logger.info(
        "Impact workspace=%s symbol=%r -> %d affected across %d files in %.0fms",
        req.workspace_id, req.symbol_name,
        result["total_affected"], result["files_affected"], elapsed,
    )

    return ImpactResponse(
        symbol=result["symbol"],
        total_affected=result["total_affected"],
        files_affected=result["files_affected"],
        by_file=[
            AffectedFile(
                file_path=f["file_path"],
                symbols=[AffectedSymbol(**s) for s in f["symbols"]],
            )
            for f in result["by_file"]
        ],
    )


# ── Internal: index files (reusable by /index and /watch) ─────────

async def _index_files_internal(workspace_id: str, files: list[dict]) -> dict:
    """Shared indexing logic used by both /index endpoint and watcher."""
    total_nodes = 0
    total_rels = 0
    total_embeddings = 0
    files_indexed = 0

    for file_payload in files:
        try:
            existing_hash = await store.get_file_hash(workspace_id, file_payload["path"])
            parse_result = parse_file(file_payload["path"], file_payload["content"])

            if existing_hash == parse_result.content_hash:
                continue

            calls_map, called_by_map = _build_context_maps(parse_result)

            enriched_texts: dict[str, str] = {}
            for defn in parse_result.definitions:
                uid = f"{defn.file_path}:{defn.name}:{defn.start_line}"
                enriched_texts[uid] = _build_enriched_text(
                    defn, calls_map, called_by_map, parse_result.imports,
                )

            # FALLBACK: embed whole file if no symbols were extracted
            if not enriched_texts and file_payload["content"].strip():
                file_uid = f"{file_payload['path']}:__file__:0"
                truncated = file_payload["content"][:5000]
                ext = file_payload["path"].rsplit(".", 1)[-1] if "." in file_payload["path"] else "unknown"
                enriched_texts[file_uid] = (
                    f"File: {file_payload['path']}\n"
                    f"Type: {ext}\n"
                    f"---\n{truncated}"
                )
                logger.info("Fallback: embedding whole file %s", file_payload["path"])

            if enriched_texts:
                texts_list = list(enriched_texts.values())
                uids_list = list(enriched_texts.keys())
                raw_embeddings = await embeddings.embed_batch(texts_list)
                embedding_map = dict(zip(uids_list, raw_embeddings))
                # Only count successful (non-None) embeddings
                total_embeddings += sum(1 for v in embedding_map.values() if v is not None)
            else:
                embedding_map = {}

            stats = await store.index_file_result(
                workspace_id, parse_result, embedding_map, enriched_texts
            )
            total_nodes += stats["nodes_created"]
            total_rels += stats["relationships_created"]

            if parse_result.references:
                await store.build_call_edges(workspace_id, parse_result.references)
                total_rels += len(parse_result.references)

            if parse_result.imports:
                await store.build_import_edges(
                    workspace_id, file_payload["path"], parse_result.imports
                )

            files_indexed += 1
        except Exception as e:
            logger.error("Failed to index %s: %s", file_payload["path"], e, exc_info=True)
            continue

    return {
        "files_indexed": files_indexed,
        "nodes_created": total_nodes,
        "relationships_created": total_rels,
        "embeddings_generated": total_embeddings,
    }


# ── POST /watch ───────────────────────────────────────────────────

@app.post("/watch", response_model=WatchResponse)
async def watch_directory(req: WatchRequest):
    """
    Start intelligent watching of a codebase directory.

    - Debounces rapid saves (AI agents edit many files at once)
    - Skips non-code files and gitignored paths
    - Detects structural changes (new/deleted/renamed symbols)
    - Cascade re-embeds callers when a function signature changes
    """
    from pathlib import Path

    root = Path(req.root_path)
    if not root.exists():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {req.root_path}")

    # If already watching, stop the old watcher
    watcher.stop_watching(req.workspace_id)

    # Do an initial scan and index
    stats = await watcher.scan_and_index(
        req.workspace_id, root, store, _index_files_internal,
    )

    # Start background watcher
    task = asyncio.create_task(
        watcher.start_watching(
            req.workspace_id, root, store, _index_files_internal,
        )
    )
    watcher._watchers[req.workspace_id] = task

    return WatchResponse(
        workspace_id=req.workspace_id,
        status="watching",
        files_scanned=stats["files_scanned"],
        files_changed=stats["files_changed"],
        symbols_added=stats["symbols_added"],
        symbols_removed=stats["symbols_removed"],
        symbols_modified=stats["symbols_modified"],
        cascade_reembeds=stats["cascade_reembeds"],
        time_ms=stats["time_ms"],
    )


# ── POST /scan ────────────────────────────────────────────────────

@app.post("/scan", response_model=WatchResponse)
async def scan_directory(req: WatchRequest):
    """
    One-shot intelligent scan — detect and index only what changed.
    Same intelligence as /watch but doesn't start a background watcher.
    """
    from pathlib import Path

    root = Path(req.root_path)
    if not root.exists():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {req.root_path}")

    stats = await watcher.scan_and_index(
        req.workspace_id, root, store, _index_files_internal,
    )

    return WatchResponse(
        workspace_id=req.workspace_id,
        status="scan_complete",
        files_scanned=stats["files_scanned"],
        files_changed=stats["files_changed"],
        symbols_added=stats["symbols_added"],
        symbols_removed=stats["symbols_removed"],
        symbols_modified=stats["symbols_modified"],
        cascade_reembeds=stats["cascade_reembeds"],
        time_ms=stats["time_ms"],
    )


# ── DELETE /watch ─────────────────────────────────────────────────

@app.delete("/watch/{workspace_id}")
async def stop_watch(workspace_id: str):
    """Stop watching a workspace."""
    stopped = watcher.stop_watching(workspace_id)
    return {"workspace_id": workspace_id, "stopped": stopped}


# ── Auth: GitHub OAuth ────────────────────────────────────────────

@app.get("/auth/github")
async def auth_github(state: str = ""):
    """Start GitHub OAuth flow — redirects to GitHub."""
    logger.info("[auth] GitHub OAuth start, state=%r", state)
    return RedirectResponse(auth.github_auth_url(state=state))


@app.get("/auth/github/callback")
async def auth_github_callback(code: str, state: str = ""):
    """GitHub OAuth callback — creates user, returns JWT."""
    logger.info("[auth] GitHub callback received, state=%r", state)
    user_info = await auth.github_exchange_code(code)
    user_id = await auth.upsert_user(user_info)
    token = auth.create_token(user_id, user_info["email"], user_info["name"])
    
    # Polling-based auth: store token for IDE to poll
    if state.startswith("poll-"):
        session_id = state[5:]  # Remove "poll-" prefix
        _pending_auth[session_id] = {
            "token": token,
            "user_info": user_info,
            "expires": time.time() + 300,  # 5 minute expiry
        }
        # Clean up old entries
        now = time.time()
        for sid in list(_pending_auth.keys()):
            if _pending_auth[sid]["expires"] < now:
                del _pending_auth[sid]
        return auth.success_page(user_info, token)
    
    # Legacy: custom URL scheme (may not work on all platforms)
    if state.startswith("forge-ide"):
        return RedirectResponse(f"forge-ide://auth?token={token}")
    
    # For browser: show success page
    return auth.success_page(user_info, token)


# ── Auth: Google OAuth ────────────────────────────────────────────

@app.get("/auth/google")
async def auth_google(state: str = ""):
    """Start Google OAuth flow — redirects to Google."""
    return RedirectResponse(auth.google_auth_url(state=state))


@app.get("/auth/google/callback")
async def auth_google_callback(code: str, state: str = ""):
    """Google OAuth callback — creates user, returns JWT."""
    logger.info("[auth] Google callback received, state=%r", state)
    user_info = await auth.google_exchange_code(code)
    user_id = await auth.upsert_user(user_info)
    token = auth.create_token(user_id, user_info["email"], user_info["name"])
    
    # Polling-based auth: store token for IDE to poll
    if state.startswith("poll-"):
        session_id = state[5:]
        _pending_auth[session_id] = {
            "token": token,
            "user_info": user_info,
            "expires": time.time() + 300,
        }
        now = time.time()
        for sid in list(_pending_auth.keys()):
            if _pending_auth[sid]["expires"] < now:
                del _pending_auth[sid]
        return auth.success_page(user_info, token)
    
    if state.startswith("forge-ide"):
        return RedirectResponse(f"forge-ide://auth?token={token}")
    return auth.success_page(user_info, token)


# ── Auth: Current user ────────────────────────────────────────────

@app.get("/auth/me")
async def auth_me(user: dict = Depends(auth.require_user)):
    """Get current authenticated user."""
    return {"user_id": user["sub"], "email": user["email"], "name": user["name"]}


@app.get("/auth/poll/{session_id}")
async def auth_poll(session_id: str):
    """
    Poll for OAuth token.
    
    IDE opens browser with state=poll-{session_id}, then polls this endpoint
    until the token is available (user completes OAuth in browser).
    
    Returns:
        - {"status": "pending"} if auth not complete
        - {"status": "success", "token": "...", "email": "...", "name": "..."} on success
        - {"status": "expired"} if session expired or not found
    """
    if session_id not in _pending_auth:
        # Only log every ~12th poll (once per minute at 5s interval) to avoid spam
        return {"status": "pending"}
    
    entry = _pending_auth[session_id]
    
    # Check expiry
    if entry["expires"] < time.time():
        del _pending_auth[session_id]
        logger.info("[auth] Poll session %s expired", session_id)
        return {"status": "expired"}
    
    # Token is ready - remove from store and return
    logger.info("[auth] Poll session %s SUCCESS, delivering token for %s", 
                session_id, entry["user_info"].get("email", "?"))
    del _pending_auth[session_id]
    return {
        "status": "success",
        "token": entry["token"],
        "email": entry["user_info"].get("email", ""),
        "name": entry["user_info"].get("name", ""),
    }


# ── POST /debug/pre-enrichment ─────────────────────────────────────

@app.post("/debug/pre-enrichment")
async def debug_pre_enrichment(workspace_id: str, question: str):
    """Debug endpoint to test pre-enrichment directly."""
    try:
        context = await agent.build_pre_enrichment(workspace_id, question, {})
        return {
            "workspace_id": workspace_id,
            "question": question,
            "context_length": len(context),
            "context": context[:5000] if context else "",  # Truncate for readability
            "has_content": len(context) > 50,
        }
    except Exception as e:
        return {
            "error": str(e),
            "workspace_id": workspace_id,
            "question": question,
        }


# ── GET /models ───────────────────────────────────────────────────

@app.get("/models", response_model=ActiveModelsResponse)
async def get_models():
    """
    List available LLM providers, their models, and the current config.
    
    Providers with a configured API key show configured=true.
    Switch models at runtime with POST /models/set.
    """
    active = llm_provider.get_active_models()
    providers_raw = llm_provider.get_available_providers()
    
    providers = [
        ProviderInfo(
            provider=p["provider"],
            name=p["name"],
            env_key=p["env_key"],
            configured=p["configured"],
            models=p["models"],
        )
        for p in providers_raw
    ]
    
    return ActiveModelsResponse(
        reasoning_model=active["reasoning_model"],
        tool_model=active["tool_model"],
        providers=providers,
    )


@app.post("/models/set", response_model=SetModelResponse)
async def set_models(req: SetModelRequest):
    """
    Switch the active reasoning and/or tool model at runtime.
    
    Model string format: "provider/model-name"
    Examples:
        "groq/moonshotai/kimi-k2-instruct-0905"
        "gemini/gemini-2.0-flash"
        "anthropic/claude-sonnet-4-20250514"
        "fireworks_ai/accounts/fireworks/models/llama-v3p1-70b-instruct"
        "openai/gpt-4o"
    """
    if req.reasoning_model:
        llm_provider.set_reasoning_model(req.reasoning_model)
    if req.tool_model:
        llm_provider.set_tool_model(req.tool_model)
    
    config = llm_provider.get_config()
    
    changes = []
    if req.reasoning_model:
        changes.append(f"reasoning → {req.reasoning_model}")
    if req.tool_model:
        changes.append(f"tool → {req.tool_model}")
    
    return SetModelResponse(
        reasoning_model=config.reasoning_model,
        tool_model=config.tool_model,
        message=f"Updated: {', '.join(changes)}" if changes else "No changes",
    )


# ── GET /health ───────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Healthcheck endpoint."""
    store_ok = await store.check_connection()
    status = "healthy" if store_ok else "degraded"
    return HealthResponse(
        status=status,
        store_ok=store_ok,
        version="1.0.0",
    )


# ── GET/DELETE /memory ─────────────────────────────────────────────────
# PHASE 1: Workspace memory management endpoints

@app.get("/memory/{workspace_id}")
async def get_workspace_memory(
    workspace_id: str,
    user: dict = Depends(auth.get_current_user)
):
    """Get workspace memory for debugging (PHASE 1)."""
    memory = await ws_memory.load_workspace_memory(workspace_id)
    return memory


@app.delete("/memory/{workspace_id}")
async def clear_workspace_memory(
    workspace_id: str,
    user: dict = Depends(auth.get_current_user)
):
    """Clear workspace memory (PHASE 1 - useful for testing/reset)."""
    await ws_memory.clear_workspace_memory(workspace_id)
    return {"status": "cleared", "workspace_id": workspace_id}
