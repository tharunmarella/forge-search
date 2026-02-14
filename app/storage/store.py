"""
Postgres + pgvector code store.

Everything in one database:
  - Symbols, files, edges in relational tables
  - 1024-dim embeddings with pgvector (Voyage voyage-code-3)
  - Graph traversal via recursive CTEs
  - No in-memory state needed (stateless API instances)

Fixes applied:
  1. HNSW index instead of IVFFlat (works on any data size)
  2. Async psycopg3 with connection pooling (no blocked requests)
  3. Score threshold on vector search (no junk results)
  4. Proper transaction handling for bulk operations
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from ..core import embeddings

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "")
VECTOR_DIMENSIONS = embeddings.get_dimensions()  # 1024 for voyage-code-3

# Score threshold: filter out results below this cosine similarity
# 0.25 is a reasonable default; adjust based on your embedding model
SCORE_THRESHOLD = float(os.getenv("VECTOR_SCORE_THRESHOLD", "0.25"))

# ── Connection Pool ───────────────────────────────────────────────
# Async connection pool with psycopg3 - no more blocking!

_pool: AsyncConnectionPool | None = None


async def _get_pool() -> AsyncConnectionPool:
    """Get or create the async connection pool."""
    global _pool
    if _pool is None:
        _pool = AsyncConnectionPool(
            conninfo=DATABASE_URL,
            min_size=2,
            max_size=10,
            open=False,  # We'll open it explicitly
        )
        await _pool.open()
        logger.info("Async connection pool opened (min=2, max=10)")
    return _pool


@asynccontextmanager
async def _connection():
    """Get a connection from the pool."""
    pool = await _get_pool()
    async with pool.connection() as conn:
        yield conn


@asynccontextmanager
async def _cursor():
    """Get a dict-cursor from the pool."""
    async with _connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            yield cur


# ── Schema ────────────────────────────────────────────────────────

async def _migrate_embedding_dimensions(conn, cur):
    """Migrate embedding column from 768 dims (Jina) to 1024 dims (Voyage)."""
    # Check current column dimensions
    await cur.execute("""
        SELECT a.atttypmod 
        FROM pg_attribute a
        JOIN pg_class c ON a.attrelid = c.oid
        WHERE c.relname = 'symbols' AND a.attname = 'embedding'
    """)
    row = await cur.fetchone()
    if row:
        current_dims = row['atttypmod'] if isinstance(row, dict) else row[0]
        if current_dims > 0 and current_dims != VECTOR_DIMENSIONS:
            logger.info(f"Migrating embedding column from {current_dims} to {VECTOR_DIMENSIONS} dimensions")
            try:
                # Drop the column and recreate with new dimensions
                await cur.execute("ALTER TABLE symbols DROP COLUMN IF EXISTS embedding")
                await cur.execute(f"ALTER TABLE symbols ADD COLUMN embedding vector({VECTOR_DIMENSIONS})")
                await conn.commit()
                logger.info("Embedding dimension migration complete - reindexing required")
            except Exception as e:
                logger.error(f"Migration failed: {e}")
                await conn.rollback()
                raise


async def _migrate_ivfflat_to_hnsw(conn, cur):
    """Migrate from IVFFlat to HNSW index if needed."""
    # Check if old IVFFlat index exists
    await cur.execute("""
        SELECT indexname, indexdef 
        FROM pg_indexes 
        WHERE tablename = 'symbols' AND indexname = 'idx_symbols_embedding'
    """)
    row = await cur.fetchone()
    
    if row:
        indexdef = row['indexdef'] if isinstance(row, dict) else row[1]
        if 'ivfflat' in indexdef.lower():
            logger.info("Migrating from IVFFlat to HNSW index...")
            await cur.execute("DROP INDEX IF EXISTS idx_symbols_embedding")
            await cur.execute("""
                CREATE INDEX idx_symbols_embedding
                ON symbols USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
            """)
            await conn.commit()
            logger.info("HNSW index migration complete")


async def ensure_schema():
    """Create tables + pgvector HNSW index."""
    async with _connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    uid TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    path TEXT NOT NULL,
                    content_hash TEXT,
                    language TEXT
                )
            """)
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS symbols (
                    uid TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    start_line INTEGER DEFAULT 0,
                    end_line INTEGER DEFAULT 0,
                    signature TEXT DEFAULT '',
                    content TEXT DEFAULT '',
                    enriched_content TEXT DEFAULT '',
                    parent TEXT DEFAULT '',
                    embedding vector({dims})
                )
            """.format(dims=VECTOR_DIMENSIONS))
            
            # Migrate if needed (handles Jina 768 -> Voyage 1024 transition)
            await _migrate_embedding_dimensions(conn, cur)
            
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS edges (
                    id SERIAL PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    from_name TEXT NOT NULL,
                    to_name TEXT NOT NULL,
                    edge_type TEXT NOT NULL
                )
            """)
            
            # Indexes for fast lookups
            await cur.execute("CREATE INDEX IF NOT EXISTS idx_symbols_ws ON symbols(workspace_id)")
            await cur.execute("CREATE INDEX IF NOT EXISTS idx_symbols_ws_name ON symbols(workspace_id, name)")
            await cur.execute("CREATE INDEX IF NOT EXISTS idx_symbols_ws_fp ON symbols(workspace_id, file_path)")
            await cur.execute("CREATE INDEX IF NOT EXISTS idx_files_ws ON files(workspace_id)")
            await cur.execute("CREATE INDEX IF NOT EXISTS idx_files_ws_path ON files(workspace_id, path)")
            await cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_ws ON edges(workspace_id)")
            await cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(workspace_id, from_name, edge_type)")
            await cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_to ON edges(workspace_id, to_name, edge_type)")

            # Check if HNSW index exists, if not create it (or migrate from IVFFlat)
            await cur.execute("""
                SELECT indexname FROM pg_indexes 
                WHERE tablename = 'symbols' AND indexname = 'idx_symbols_embedding'
            """)
            existing_idx = await cur.fetchone()
            
            if existing_idx:
                # Migrate from IVFFlat to HNSW if needed
                await _migrate_ivfflat_to_hnsw(conn, cur)
            else:
                # Create HNSW index (works well from 10 to 10M vectors)
                # m=16: connections per node (higher = better recall, more memory)
                # ef_construction=64: build-time beam width (higher = better quality)
                await cur.execute("""
                    CREATE INDEX idx_symbols_embedding
                    ON symbols USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64)
                """)
                logger.info("Created HNSW vector index")

            await conn.commit()
            logger.info("Postgres schema ready (pgvector + HNSW enabled)")


async def close_driver():
    """Close the connection pool."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        logger.info("Connection pool closed")


async def check_connection() -> bool:
    """Check if the database connection is healthy."""
    try:
        async with _cursor() as cur:
            await cur.execute("SELECT 1")
            await cur.fetchone()
        return True
    except Exception as e:
        logger.error(f"Connection check failed: {e}")
        return False


# ── Indexing ──────────────────────────────────────────────────────

async def clear_workspace(workspace_id: str):
    """Delete all data for a workspace."""
    async with _connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("DELETE FROM symbols WHERE workspace_id = %s", (workspace_id,))
            await cur.execute("DELETE FROM files WHERE workspace_id = %s", (workspace_id,))
            await cur.execute("DELETE FROM edges WHERE workspace_id = %s", (workspace_id,))
            await conn.commit()
    logger.info("Cleared workspace %s", workspace_id)


async def delete_file_data(workspace_id: str, file_path: str):
    """Delete all data associated with a file (symbols and outgoing edges)."""
    async with _connection() as conn:
        try:
            async with conn.cursor() as cur:
                # 1. Delete edges originating from symbols in this file
                await cur.execute("""
                    DELETE FROM edges 
                    WHERE workspace_id = %s 
                    AND from_name IN (
                        SELECT name FROM symbols 
                        WHERE workspace_id = %s AND file_path = %s
                    )
                """, (workspace_id, workspace_id, file_path))

                # 2. Delete IMPORTS edges from this file
                await cur.execute("""
                    DELETE FROM edges 
                    WHERE workspace_id = %s 
                    AND from_name = %s 
                    AND edge_type = 'IMPORTS'
                """, (workspace_id, file_path))

                # 3. Delete symbols in this file
                await cur.execute(
                    "DELETE FROM symbols WHERE workspace_id = %s AND file_path = %s",
                    (workspace_id, file_path)
                )

                # 4. Delete the file record itself
                await cur.execute(
                    "DELETE FROM files WHERE uid = %s",
                    (f"{workspace_id}:{file_path}",)
                )

                await conn.commit()
                logger.info("Deleted all data for file %s in workspace %s", file_path, workspace_id)
        except Exception as e:
            await conn.rollback()
            logger.error("Failed to delete file data for %s: %s", file_path, e)
            raise


async def get_file_hash(workspace_id: str, file_path: str) -> str | None:
    """Get the content hash for a file (for change detection)."""
    async with _cursor() as cur:
        await cur.execute(
            "SELECT content_hash FROM files WHERE uid = %s", 
            (f"{workspace_id}:{file_path}",)
        )
        row = await cur.fetchone()
        return row["content_hash"] if row else None


async def index_file_result(
    workspace_id: str,
    parse_result,
    embeddings_map: dict[str, list[float]],
    enriched_texts: dict[str, str] | None = None,
) -> dict[str, int]:
    """
    Index a parsed file with embeddings.
    
    Uses a single transaction for atomicity - if anything fails,
    the entire operation is rolled back cleanly.
    """
    enriched_texts = enriched_texts or {}
    stats = {"nodes_created": 0, "relationships_created": 0}

    file_uid = f"{workspace_id}:{parse_result.file_path}"
    lang = _detect_language(parse_result.file_path)

    async with _connection() as conn:
        try:
            async with conn.cursor() as cur:
                # Upsert file
                await cur.execute("""
                    INSERT INTO files (uid, workspace_id, path, content_hash, language)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (uid) DO UPDATE SET content_hash = %s, language = %s
                """, (file_uid, workspace_id, parse_result.file_path,
                      parse_result.content_hash, lang, parse_result.content_hash, lang))
                stats["nodes_created"] += 1

                # Remove old symbols for this file
                await cur.execute(
                    "DELETE FROM symbols WHERE workspace_id = %s AND file_path = %s",
                    (workspace_id, parse_result.file_path)
                )

                # Remove old edges originating from this file
                # This ensures that if a call or import is removed, the edge is also removed.
                await cur.execute("""
                    DELETE FROM edges 
                    WHERE workspace_id = %s 
                    AND from_name IN (
                        SELECT name FROM symbols 
                        WHERE workspace_id = %s AND file_path = %s
                    )
                """, (workspace_id, workspace_id, parse_result.file_path))

                # Also remove old IMPORTS edges where the file_path itself is the from_name
                await cur.execute("""
                    DELETE FROM edges 
                    WHERE workspace_id = %s 
                    AND from_name = %s 
                    AND edge_type = 'IMPORTS'
                """, (workspace_id, parse_result.file_path))

                # Insert new symbols
                for defn in parse_result.definitions:
                    uid = f"{workspace_id}:{defn.file_path}:{defn.name}:{defn.start_line}"
                    defn_key = f"{defn.file_path}:{defn.name}:{defn.start_line}"
                    emb = embeddings_map.get(defn_key)
                    enriched = enriched_texts.get(defn_key, "")

                    emb_str = f"[{','.join(str(x) for x in emb)}]" if emb else None

                    await cur.execute("""
                        INSERT INTO symbols (uid, workspace_id, name, kind, file_path, start_line, end_line,
                                             signature, content, enriched_content, parent, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (uid) DO UPDATE SET
                            name=%s, kind=%s, signature=%s, content=%s, enriched_content=%s, parent=%s, embedding=%s
                    """, (uid, workspace_id, defn.name, defn.kind, defn.file_path,
                          defn.start_line, defn.end_line, defn.signature, defn.content,
                          enriched, defn.parent or "", emb_str,
                          defn.name, defn.kind, defn.signature, defn.content, enriched, defn.parent or "", emb_str))

                    stats["nodes_created"] += 1
                    stats["relationships_created"] += 1

                    if defn.parent:
                        await cur.execute("""
                            INSERT INTO edges (workspace_id, from_name, to_name, edge_type)
                            VALUES (%s, %s, %s, 'BELONGS_TO')
                        """, (workspace_id, defn.name, defn.parent))
                        stats["relationships_created"] += 1
                
                # Handle fallback whole-file embeddings when tree-sitter extraction failed
                # These have keys like "path/to/file.tsx:__file__:0"
                if not parse_result.definitions:
                    file_key = f"{parse_result.file_path}:__file__:0"
                    emb = embeddings_map.get(file_key)
                    enriched = enriched_texts.get(file_key, "")
                    
                    if emb:
                        uid = f"{workspace_id}:{file_key}"
                        file_name = parse_result.file_path.rsplit("/", 1)[-1]
                        emb_str = f"[{','.join(str(x) for x in emb)}]"
                        
                        await cur.execute("""
                            INSERT INTO symbols (uid, workspace_id, name, kind, file_path, start_line, end_line,
                                                 signature, content, enriched_content, parent, embedding)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (uid) DO UPDATE SET
                                name=%s, kind=%s, signature=%s, content=%s, enriched_content=%s, parent=%s, embedding=%s
                        """, (uid, workspace_id, file_name, "file", parse_result.file_path,
                              0, 0, f"File: {parse_result.file_path}", "", enriched, "", emb_str,
                              file_name, "file", f"File: {parse_result.file_path}", "", enriched, "", emb_str))
                        
                        stats["nodes_created"] += 1
                        logger.debug("Stored fallback file embedding for %s", parse_result.file_path)

                # Commit the entire transaction
                await conn.commit()
                
        except Exception as e:
            # Rollback on any error - no partial state
            await conn.rollback()
            logger.error(f"index_file_result failed, rolled back: {e}")
            raise

    return stats


async def build_call_edges(workspace_id: str, references):
    """Build CALLS edges from symbol references."""
    async with _connection() as conn:
        async with conn.cursor() as cur:
            for ref in references:
                if not ref.context_name or ref.context_name == ref.name:
                    continue
                await cur.execute("""
                    INSERT INTO edges (workspace_id, from_name, to_name, edge_type)
                    SELECT %s, %s, %s, 'CALLS'
                    WHERE NOT EXISTS (
                        SELECT 1 FROM edges WHERE workspace_id=%s AND from_name=%s AND to_name=%s AND edge_type='CALLS'
                    )
                """, (workspace_id, ref.context_name, ref.name,
                      workspace_id, ref.context_name, ref.name))
            await conn.commit()


async def build_import_edges(workspace_id: str, file_path: str, imports: list[str]):
    """Build IMPORTS edges from import statements."""
    async with _connection() as conn:
        async with conn.cursor() as cur:
            for imp in imports:
                hint = _import_to_path_hint(imp)
                await cur.execute("""
                    INSERT INTO edges (workspace_id, from_name, to_name, edge_type)
                    SELECT %s, %s, f.path, 'IMPORTS'
                    FROM files f
                    WHERE f.workspace_id = %s AND f.path LIKE %s AND f.path != %s
                    AND NOT EXISTS (
                        SELECT 1 FROM edges WHERE workspace_id=%s AND from_name=%s AND to_name=f.path AND edge_type='IMPORTS'
                    )
                """, (workspace_id, file_path, workspace_id, f"%{hint}%", file_path,
                      workspace_id, file_path))
            await conn.commit()


# ── Vector search (pgvector + HNSW) ──────────────────────────────

async def vector_search(
    workspace_id: str,
    query_embedding: list[float],
    top_k: int = 10,
    score_threshold: float | None = None,
) -> list[dict[str, Any]]:
    """
    Native vector search using pgvector cosine distance with HNSW index.
    
    Args:
        workspace_id: Filter to this workspace
        query_embedding: The query vector
        top_k: Maximum results to return
        score_threshold: Minimum cosine similarity (0-1). Defaults to SCORE_THRESHOLD.
                        Set to 0 to disable filtering.
    """
    threshold = score_threshold if score_threshold is not None else SCORE_THRESHOLD
    
    async with _cursor() as cur:
        emb_str = f"[{','.join(str(x) for x in query_embedding)}]"

        # Set HNSW search parameter for quality vs speed tradeoff
        # ef_search: higher = better recall, slower (default 40, we use 64)
        await cur.execute("SET hnsw.ef_search = 64")

        # Query with score threshold to filter irrelevant results
        await cur.execute("""
            SELECT uid, name, kind, file_path, start_line, end_line,
                   signature, content, parent,
                   1 - (embedding <=> %s::vector) AS score
            FROM symbols
            WHERE workspace_id = %s 
              AND embedding IS NOT NULL
              AND 1 - (embedding <=> %s::vector) > %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (emb_str, workspace_id, emb_str, threshold, emb_str, top_k))

        results = []
        async for row in cur:
            results.append({
                "symbol": dict(row),
                "score": float(row["score"]),
            })
        
        return results


# ── Graph traversal ───────────────────────────────────────────────

async def expand_neighbors(workspace_id: str, symbol_uids: list[str], hops: int = 1):
    """Get immediate neighbors of symbols (calls, called_by, belongs_to, has_member)."""
    async with _cursor() as cur:
        names = []
        for uid in symbol_uids:
            await cur.execute("SELECT name FROM symbols WHERE uid = %s", (uid,))
            row = await cur.fetchone()
            if row:
                names.append(row["name"])

        if not names:
            return []

        results = []
        for name in names:
            await cur.execute("""
                SELECT s.name, s.file_path, s.signature, s.kind, 'calls' as rel_type
                FROM edges e JOIN symbols s ON s.workspace_id = e.workspace_id AND s.name = e.to_name
                WHERE e.workspace_id = %s AND e.from_name = %s AND e.edge_type = 'CALLS'
                UNION
                SELECT s.name, s.file_path, s.signature, s.kind, 'called_by' as rel_type
                FROM edges e JOIN symbols s ON s.workspace_id = e.workspace_id AND s.name = e.from_name
                WHERE e.workspace_id = %s AND e.to_name = %s AND e.edge_type = 'CALLS'
                UNION
                SELECT s.name, s.file_path, s.signature, s.kind, 'belongs_to' as rel_type
                FROM edges e JOIN symbols s ON s.workspace_id = e.workspace_id AND s.name = e.to_name
                WHERE e.workspace_id = %s AND e.from_name = %s AND e.edge_type = 'BELONGS_TO'
                UNION
                SELECT s.name, s.file_path, s.signature, s.kind, 'has_member' as rel_type
                FROM edges e JOIN symbols s ON s.workspace_id = e.workspace_id AND s.name = e.from_name
                WHERE e.workspace_id = %s AND e.to_name = %s AND e.edge_type = 'BELONGS_TO'
            """, (workspace_id, name, workspace_id, name,
                  workspace_id, name, workspace_id, name))
            
            async for row in cur:
                results.append(dict(row))
                
        return results


async def trace_call_chain(workspace_id: str, symbol_name: str, direction: str = "both", max_depth: int = 5):
    """
    Deep call chain traversal using recursive CTE — the Postgres superpower.
    
    Returns all symbols in the call chain up/downstream from the given symbol.
    """
    async with _cursor() as cur:
        nodes = {}
        edges_list = []

        # Find root symbol(s)
        await cur.execute("""
            SELECT uid, name, kind, file_path, signature, start_line, end_line
            FROM symbols WHERE workspace_id = %s AND name = %s
        """, (workspace_id, symbol_name))
        
        roots = [dict(row) async for row in cur]
        if not roots:
            return {"nodes": [], "edges": [], "root": symbol_name, "depth_reached": 0}

        for root in roots:
            nodes[root["uid"]] = {**root, "depth": 0, "direction": "root"}

        # Downstream: what does this symbol call?
        if direction in ("downstream", "both"):
            await cur.execute("""
                WITH RECURSIVE chain AS (
                    SELECT e.from_name, e.to_name, 1 as depth
                    FROM edges e
                    WHERE e.workspace_id = %s AND e.from_name = %s AND e.edge_type = 'CALLS'

                    UNION

                    SELECT e.from_name, e.to_name, c.depth + 1
                    FROM edges e
                    JOIN chain c ON c.to_name = e.from_name
                    WHERE e.workspace_id = %s AND e.edge_type = 'CALLS' AND c.depth < %s
                )
                SELECT DISTINCT c.from_name, c.to_name, c.depth,
                       s.uid, s.name, s.kind, s.file_path, s.signature, s.start_line, s.end_line
                FROM chain c
                JOIN symbols s ON s.workspace_id = %s AND s.name = c.to_name
                ORDER BY c.depth
                LIMIT 50
            """, (workspace_id, symbol_name, workspace_id, max_depth, workspace_id))

            async for row in cur:
                r = dict(row)
                if r["uid"] not in nodes:
                    nodes[r["uid"]] = {
                        "uid": r["uid"], "name": r["name"], "kind": r["kind"],
                        "file_path": r["file_path"], "signature": r["signature"],
                        "start_line": r["start_line"], "end_line": r["end_line"],
                        "depth": r["depth"], "direction": "downstream",
                    }
                edges_list.append({"from": r["from_name"], "to": r["to_name"], "type": "CALLS"})

        # Upstream: what calls this symbol?
        if direction in ("upstream", "both"):
            await cur.execute("""
                WITH RECURSIVE chain AS (
                    SELECT e.from_name, e.to_name, 1 as depth
                    FROM edges e
                    WHERE e.workspace_id = %s AND e.to_name = %s AND e.edge_type = 'CALLS'

                    UNION

                    SELECT e.from_name, e.to_name, c.depth + 1
                    FROM edges e
                    JOIN chain c ON c.from_name = e.to_name
                    WHERE e.workspace_id = %s AND e.edge_type = 'CALLS' AND c.depth < %s
                )
                SELECT DISTINCT c.from_name, c.to_name, c.depth,
                       s.uid, s.name, s.kind, s.file_path, s.signature, s.start_line, s.end_line
                FROM chain c
                JOIN symbols s ON s.workspace_id = %s AND s.name = c.from_name
                ORDER BY c.depth
                LIMIT 50
            """, (workspace_id, symbol_name, workspace_id, max_depth, workspace_id))

            async for row in cur:
                r = dict(row)
                if r["uid"] not in nodes:
                    nodes[r["uid"]] = {
                        "uid": r["uid"], "name": r["name"], "kind": r["kind"],
                        "file_path": r["file_path"], "signature": r["signature"],
                        "start_line": r["start_line"], "end_line": r["end_line"],
                        "depth": r["depth"], "direction": "upstream",
                    }
                edges_list.append({"from": r["from_name"], "to": r["to_name"], "type": "CALLS"})

        # Deduplicate edges
        unique_edges = list({(e["from"], e["to"], e["type"]): e for e in edges_list}.values())

        return {
            "root": symbol_name,
            "nodes": list(nodes.values()),
            "edges": unique_edges,
            "depth_reached": max((n.get("depth", 0) for n in nodes.values()), default=0),
        }


async def impact_analysis(workspace_id: str, symbol_name: str, max_depth: int = 4):
    """
    Blast radius analysis using recursive CTE.
    
    Find all upstream dependents that would be affected if this symbol changes.
    """
    async with _cursor() as cur:
        affected = {}

        # Upstream callers (recursive)
        await cur.execute("""
            WITH RECURSIVE chain AS (
                SELECT e.from_name, e.to_name, 1 as depth
                FROM edges e
                WHERE e.workspace_id = %s AND e.to_name = %s AND e.edge_type = 'CALLS'

                UNION

                SELECT e.from_name, e.to_name, c.depth + 1
                FROM edges e
                JOIN chain c ON c.from_name = e.to_name
                WHERE e.workspace_id = %s AND e.edge_type = 'CALLS' AND c.depth < %s
            )
            SELECT DISTINCT s.uid, s.name, s.kind, s.file_path, s.signature,
                   s.start_line, s.end_line, c.depth as distance, 'caller' as impact_type
            FROM chain c
            JOIN symbols s ON s.workspace_id = %s AND s.name = c.from_name
            ORDER BY c.depth
            LIMIT 100
        """, (workspace_id, symbol_name, workspace_id, max_depth, workspace_id))

        async for row in cur:
            r = dict(row)
            if r["uid"] not in affected:
                affected[r["uid"]] = r

        # Sibling members (same parent class/struct)
        await cur.execute("""
            SELECT s.uid, s.name, s.kind, s.file_path, s.signature,
                   s.start_line, s.end_line, 1 as distance, 'sibling_member' as impact_type
            FROM edges e1
            JOIN edges e2 ON e2.to_name = e1.to_name AND e2.workspace_id = e1.workspace_id AND e2.edge_type = 'BELONGS_TO'
            JOIN symbols s ON s.workspace_id = e1.workspace_id AND s.name = e2.from_name
            WHERE e1.workspace_id = %s AND e1.from_name = %s AND e1.edge_type = 'BELONGS_TO'
            AND e2.from_name != %s
        """, (workspace_id, symbol_name, symbol_name))

        async for row in cur:
            r = dict(row)
            if r["uid"] not in affected:
                affected[r["uid"]] = r

    # Group by file
    by_file: dict[str, list] = {}
    for sym in affected.values():
        by_file.setdefault(sym["file_path"], []).append(sym)

    sorted_files = sorted(by_file.items(), key=lambda x: -len(x[1]))

    return {
        "symbol": symbol_name,
        "total_affected": len(affected),
        "files_affected": len(by_file),
        "by_file": [
            {"file_path": fp, "symbols": sorted(syms, key=lambda s: s.get("start_line", 0))}
            for fp, syms in sorted_files
        ],
    }


async def get_project_map(
    workspace_id: str,
    focus_path: str | None = None,
    focus_symbol: str | None = None,
    depth: int = 1
) -> dict[str, Any]:
    """
    Build a hierarchical project map.
    
    Level 1 (High): Project/Module view (file tree + imports)
    Level 2 (Mid): File/Class view (symbols inside a file)
    Level 3 (Low): Method/Call view (call graph for a symbol)
    """
    nodes = []
    edges = []

    async with _cursor() as cur:
        if focus_symbol:
            # Level 3: Low-level call graph for a specific symbol
            # Find the symbol first
            await cur.execute(
                "SELECT uid, name, kind, file_path, signature FROM symbols WHERE workspace_id = %s AND name = %s LIMIT 1",
                (workspace_id, focus_symbol)
            )
            root_sym = await cur.fetchone()
            if not root_sym:
                return {
                    "workspace_id": workspace_id,
                    "nodes": [],
                    "edges": [],
                    "focus_path": focus_path,
                    "focus_symbol": focus_symbol
                }

            # Add root node
            nodes.append({
                "id": root_sym["uid"],
                "name": root_sym["name"],
                "kind": root_sym["kind"],
                "file_path": root_sym["file_path"],
                "signature": root_sym["signature"]
            })

            # Find immediate calls (downstream)
            await cur.execute("""
                SELECT s.uid, s.name, s.kind, s.file_path, s.signature, e.edge_type
                FROM edges e
                JOIN symbols s ON s.workspace_id = e.workspace_id AND s.name = e.to_name
                WHERE e.workspace_id = %s AND e.from_name = %s AND e.edge_type = 'CALLS'
                LIMIT 50
            """, (workspace_id, focus_symbol))
            
            async for row in cur:
                nodes.append({
                    "id": row["uid"],
                    "name": row["name"],
                    "kind": row["kind"],
                    "file_path": row["file_path"],
                    "signature": row["signature"]
                })
                edges.append({"from": root_sym["uid"], "to": row["uid"], "type": "CALLS"})

            # Find immediate callers (upstream)
            await cur.execute("""
                SELECT s.uid, s.name, s.kind, s.file_path, s.signature, e.edge_type
                FROM edges e
                JOIN symbols s ON s.workspace_id = e.workspace_id AND s.name = e.from_name
                WHERE e.workspace_id = %s AND e.to_name = %s AND e.edge_type = 'CALLS'
                LIMIT 50
            """, (workspace_id, focus_symbol))
            
            async for row in cur:
                nodes.append({
                    "id": row["uid"],
                    "name": row["name"],
                    "kind": row["kind"],
                    "file_path": row["file_path"],
                    "signature": row["signature"]
                })
                edges.append({"from": row["uid"], "to": root_sym["uid"], "type": "CALLS"})

        elif focus_path:
            # Level 2: Mid-level view of symbols inside a file
            await cur.execute("""
                SELECT uid, name, kind, file_path, signature, parent
                FROM symbols
                WHERE workspace_id = %s AND file_path = %s
                ORDER BY start_line
            """, (workspace_id, focus_path))
            
            async for row in cur:
                nodes.append({
                    "id": row["uid"],
                    "name": row["name"],
                    "kind": row["kind"],
                    "file_path": row["file_path"],
                    "signature": row["signature"]
                })
                if row["parent"]:
                    # Find parent symbol UID
                    await cur.execute(
                        "SELECT uid FROM symbols WHERE workspace_id = %s AND name = %s AND file_path = %s LIMIT 1",
                        (workspace_id, row["parent"], focus_path)
                    )
                    parent_row = await cur.fetchone()
                    if parent_row:
                        edges.append({"from": row["uid"], "to": parent_row["uid"], "type": "BELONGS_TO"})

        else:
            # Level 1: High-level project view (file tree + imports)
            await cur.execute("SELECT path, language FROM files WHERE workspace_id = %s", (workspace_id,))
            async for row in cur:
                nodes.append({
                    "id": row["path"],
                    "name": row["path"].split("/")[-1],
                    "kind": "file",
                    "file_path": row["path"]
                })

            # Add directory nodes
            dirs = set()
            for node in nodes:
                parts = node["file_path"].split("/")[:-1]
                curr = ""
                for p in parts:
                    parent = curr
                    curr = f"{curr}/{p}" if curr else p
                    if curr not in dirs:
                        dirs.add(curr)
                        nodes.append({
                            "id": curr,
                            "name": p,
                            "kind": "directory",
                            "file_path": curr
                        })
                        if parent:
                            edges.append({"from": curr, "to": parent, "type": "CONTAINS"})

            # Add file-level IMPORTS edges
            await cur.execute("""
                SELECT from_name, to_name
                FROM edges
                WHERE workspace_id = %s AND edge_type = 'IMPORTS'
            """, (workspace_id,))
            async for row in cur:
                edges.append({"from": row["from_name"], "to": row["to_name"], "type": "IMPORTS"})

    # Deduplicate nodes by ID
    unique_nodes = {n["id"]: n for n in nodes}.values()
    
    return {
        "workspace_id": workspace_id,
        "nodes": list(unique_nodes),
        "edges": edges,
        "focus_path": focus_path,
        "focus_symbol": focus_symbol
    }


async def get_workspace_stats(workspace_id: str) -> dict[str, int]:
    """Get symbol and file counts for a workspace."""
    async with _cursor() as cur:
        await cur.execute("SELECT COUNT(*) as c FROM symbols WHERE workspace_id = %s", (workspace_id,))
        row = await cur.fetchone()
        symbols = row["c"] if row else 0
        
        await cur.execute("SELECT COUNT(*) as c FROM files WHERE workspace_id = %s", (workspace_id,))
        row = await cur.fetchone()
        files = row["c"] if row else 0
        
        return {"symbols": symbols, "files": files}


# ── Watcher support ───────────────────────────────────────────────

async def _get_ws(workspace_id: str):
    """Compatibility shim for watcher.py — returns a namespace with called_by + symbols."""
    
    class _WS:
        def __init__(self):
            self.symbols = {}
            self.called_by = {}
            self.files = {}

    ws = _WS()

    async with _cursor() as cur:
        await cur.execute(
            "SELECT uid, name, kind, file_path, start_line, end_line, signature, content, parent "
            "FROM symbols WHERE workspace_id = %s", 
            (workspace_id,)
        )
        async for row in cur:
            r = dict(row)
            ws.symbols[r["uid"]] = r

        await cur.execute(
            "SELECT from_name, to_name FROM edges WHERE workspace_id = %s AND edge_type = 'CALLS'", 
            (workspace_id,)
        )
        async for row in cur:
            ws.called_by.setdefault(row["to_name"], set()).add(row["from_name"])

        await cur.execute(
            "SELECT uid, path, content_hash FROM files WHERE workspace_id = %s", 
            (workspace_id,)
        )
        async for row in cur:
            ws.files[row["uid"]] = dict(row)

    return ws


# Sync wrapper for watcher.py compatibility (if it uses sync calls)
def _get_ws_sync(workspace_id: str):
    """Sync wrapper - run the async version in a new event loop."""
    import asyncio
    try:
        loop = asyncio.get_running_loop()
        # If we're in an async context, we can't use run_until_complete
        # This shouldn't happen in normal usage
        raise RuntimeError("Cannot call sync wrapper from async context")
    except RuntimeError:
        # No running loop, safe to create one
        return asyncio.run(_get_ws(workspace_id))


# ── Utilities ─────────────────────────────────────────────────────

def _detect_language(file_path: str) -> str:
    """Detect language from file extension."""
    ext_map = {
        ".rs": "rust", ".py": "python", ".js": "javascript",
        ".ts": "typescript", ".tsx": "typescript", ".go": "go",
        ".java": "java", ".cpp": "cpp", ".c": "c", ".h": "c",
    }
    from pathlib import Path
    return ext_map.get(Path(file_path).suffix, "unknown")


def _import_to_path_hint(imp: str) -> str:
    """Convert import statement to file path hint for matching."""
    hint = imp.replace("::", "/").replace(".", "/")
    for prefix in ("crate/", "self/", "super/"):
        if hint.startswith(prefix):
            hint = hint[len(prefix):]
    return hint
