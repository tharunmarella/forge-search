"""Search and indexing endpoints."""

import time
import logging
from fastapi import APIRouter, Depends, HTTPException
from ..core import embeddings
from ..core.parser import parse_file
from ..storage import store
from ..utils import auth
from ..models import (
    IndexRequest,
    IndexResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    RelatedSymbol,
    ReindexRequest,
)
from .indexing_helpers import build_context_maps, build_enriched_text

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/index", response_model=IndexResponse)
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
            calls_map, called_by_map = build_context_maps(parse_result)

            # Build context-enriched text for each definition
            enriched_texts: dict[str, str] = {}
            for defn in parse_result.definitions:
                uid = f"{defn.file_path}:{defn.name}:{defn.start_line}"
                enriched_texts[uid] = build_enriched_text(
                    defn, calls_map, called_by_map, parse_result.imports,
                )

            # FALLBACK: If tree-sitter found no definitions, embed whole file
            if not enriched_texts and file_payload.content.strip():
                file_uid = f"{file_payload.path}:__file__:0"
                truncated = file_payload.content[:5000]
                ext = file_payload.path.rsplit(".", 1)[-1] if "." in file_payload.path else "unknown"
                enriched_texts[file_uid] = (
                    f"File: {file_payload.path}\n"
                    f"Type: {ext}\n"
                    f"---\n{truncated}"
                )
                logger.info("Fallback: embedding whole file %s", file_payload.path)

            # Generate embeddings
            if enriched_texts:
                texts_list = list(enriched_texts.values())
                uids_list = list(enriched_texts.keys())
                raw_embeddings = await embeddings.embed_batch(texts_list)
                embedding_map = dict(zip(uids_list, raw_embeddings))
                total_embeddings += sum(1 for v in embedding_map.values() if v is not None)
            else:
                embedding_map = {}

            # Store in database
            stats = await store.index_file_result(
                workspace_id, parse_result, embedding_map, enriched_texts
            )
            total_nodes += stats["nodes_created"]
            total_rels += stats["relationships_created"]

            # Build call graph edges
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
        "Index workspace=%s, %d files → %d nodes, %d rels, %d embeddings in %.0fms",
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


@router.post("/search", response_model=SearchResponse)
async def search_code(req: SearchRequest):
    """
    Semantic code search powered by pgvector.

    Understands intent, not just keywords.
    Examples:
        - "how is authentication handled" → finds login logic
        - "where is database connection" → finds DB config
        - "error handling patterns" → finds try/catch blocks
    """
    t0 = time.monotonic()

    # Generate embedding for query
    query_embedding = await embeddings.embed_query(req.query)

    if query_embedding is None:
        raise HTTPException(status_code=500, detail="Failed to generate query embedding")

    # Search with pgvector cosine similarity
    results = await store.vector_search(
        workspace_id=req.workspace_id,
        query_embedding=query_embedding,
        top_k=req.top_k,
    )

    # For each result, find related symbols via expand_neighbors
    search_results = []
    for item in results:
        sym = item["symbol"]
        uid = sym["uid"]

        neighbors = await store.expand_neighbors(
            req.workspace_id, [uid], hops=1,
        )

        search_results.append(
            SearchResult(
                file_path=sym["file_path"],
                name=sym["name"],
                symbol_type=sym["kind"],
                start_line=sym["start_line"],
                end_line=sym["end_line"],
                signature=sym.get("signature", ""),
                content=sym.get("content", ""),
                score=item["score"],
                related=[
                    RelatedSymbol(
                        name=r["name"],
                        file_path=r["file_path"],
                        relationship=r["rel_type"],
                        signature=r.get("signature", ""),
                    )
                    for r in neighbors
                ],
            )
        )

    elapsed = (time.monotonic() - t0) * 1000
    logger.info(
        "Search workspace=%s query=%r -> %d results in %.0fms",
        req.workspace_id, req.query, len(results), elapsed,
    )

    return SearchResponse(
        results=search_results,
        query=req.query,
        workspace_id=req.workspace_id,
        search_time_ms=round(elapsed, 1),
    )


@router.post("/reindex", response_model=IndexResponse)
async def reindex_workspace(
    req: ReindexRequest,
    user: dict = Depends(auth.get_current_user)
):
    """
    Force re-index of all files in a workspace.
    
    WARNING: This clears existing embeddings and re-generates them.
    Use only if the index is corrupted or embeddings need updating.
    """
    workspace_id = req.workspace_id
    
    logger.info("[reindex] Starting full reindex for workspace=%s", workspace_id)
    
    # Clear existing data
    await store.clear_workspace(workspace_id)
    
    logger.info("[reindex] Cleared workspace %s", workspace_id)
    
    return IndexResponse(
        workspace_id=workspace_id,
        files_indexed=0,
        nodes_created=0,
        relationships_created=0,
        embeddings_generated=0,
        index_time_ms=0,
    )
