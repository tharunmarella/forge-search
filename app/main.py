"""
Forge Search API -- FastAPI application.

Endpoints:
  POST /index    -- Parse + embed + store files in Neo4j
  POST /search   -- Semantic + graph search
  POST /reindex  -- Force full re-index of a workspace
  GET  /health   -- Healthcheck
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from . import graph, embeddings
from .models import (
    IndexRequest, IndexResponse,
    SearchRequest, SearchResponse, SearchResult, RelatedSymbol,
    ReindexRequest,
    HealthResponse,
)
from .parser import parse_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── App lifecycle ─────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown hooks."""
    logger.info("Starting Forge Search API...")
    await graph.ensure_schema()
    logger.info("Neo4j schema ready")
    yield
    logger.info("Shutting down...")
    await graph.close_driver()


app = FastAPI(
    title="Forge Search API",
    description="Cloud-hosted code intelligence with Neo4j graph + Gemini embeddings",
    version="0.1.0",
    lifespan=lifespan,
)


# ── POST /index ───────────────────────────────────────────────────

@app.post("/index", response_model=IndexResponse)
async def index_files(req: IndexRequest):
    """
    Index source files into the code graph.

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
            existing_hash = await graph.get_file_hash(workspace_id, file_payload.path)

            # Parse the file
            parse_result = parse_file(file_payload.path, file_payload.content)

            if existing_hash == parse_result.content_hash:
                logger.debug("Skipping unchanged file: %s", file_payload.path)
                continue

            # Generate embeddings for all definitions
            contents_to_embed = [d.content for d in parse_result.definitions]
            if contents_to_embed:
                embedding_map = await embeddings.embed_texts_to_map(contents_to_embed)
                total_embeddings += len(embedding_map)
            else:
                embedding_map = {}

            # Store in Neo4j
            stats = await graph.index_file_result(
                workspace_id, parse_result, embedding_map
            )
            total_nodes += stats["nodes_created"]
            total_rels += stats["relationships_created"]

            # Build call graph edges from references
            if parse_result.references:
                await graph.build_call_edges(workspace_id, parse_result.references)
                total_rels += len(parse_result.references)

            # Build import edges
            if parse_result.imports:
                await graph.build_import_edges(
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
        raw_results = await graph.vector_search(
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
        neighbors = await graph.expand_neighbors(req.workspace_id, symbol_uids)
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
    stats = await graph.get_workspace_stats(req.workspace_id)
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
    await graph.clear_workspace(req.workspace_id)
    # Return empty response -- the caller should follow up with /index
    return IndexResponse(
        workspace_id=req.workspace_id,
        files_indexed=0,
        nodes_created=0,
        relationships_created=0,
        embeddings_generated=0,
        index_time_ms=0,
    )


# ── GET /health ───────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Healthcheck endpoint."""
    neo4j_ok = await graph.check_connection()
    status = "healthy" if neo4j_ok else "degraded"
    return HealthResponse(
        status=status,
        neo4j_connected=neo4j_ok,
        version="0.1.0",
    )
