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
                    embedding vector({dims}),
                    service_id TEXT DEFAULT '',
                    architectural_role TEXT DEFAULT '',
                    description TEXT DEFAULT ''
                )
            """.format(dims=VECTOR_DIMENSIONS))
            
            # Migration: Add columns if they don't exist
            await cur.execute("ALTER TABLE symbols ADD COLUMN IF NOT EXISTS service_id TEXT DEFAULT ''")
            await cur.execute("ALTER TABLE symbols ADD COLUMN IF NOT EXISTS architectural_role TEXT DEFAULT ''")
            await cur.execute("ALTER TABLE symbols ADD COLUMN IF NOT EXISTS description TEXT DEFAULT ''")
            
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

            # Services table for hierarchical map (Level 0)
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS services (
                    id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    service_type TEXT DEFAULT 'core',
                    files TEXT[] DEFAULT '{}',
                    PRIMARY KEY (id, workspace_id)
                )
            """)
            await cur.execute("CREATE INDEX IF NOT EXISTS idx_services_ws ON services(workspace_id)")
            
            # Indexes for fast lookups
            await cur.execute("CREATE INDEX IF NOT EXISTS idx_symbols_ws ON symbols(workspace_id)")
            await cur.execute("CREATE INDEX IF NOT EXISTS idx_symbols_ws_name ON symbols(workspace_id, name)")
            await cur.execute("CREATE INDEX IF NOT EXISTS idx_symbols_ws_fp ON symbols(workspace_id, file_path)")
            await cur.execute("CREATE INDEX IF NOT EXISTS idx_symbols_service_id ON symbols(workspace_id, service_id)")
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


async def upsert_service(
    workspace_id: str,
    service_id: str,
    name: str,
    description: str = "",
    service_type: str = "core",
    files: list[str] | None = None
):
    """Upsert a service definition."""
    files = files or []
    async with _connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("""
                INSERT INTO services (id, workspace_id, name, description, service_type, files)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id, workspace_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    description = EXCLUDED.description,
                    service_type = EXCLUDED.service_type,
                    files = EXCLUDED.files
            """, (service_id, workspace_id, name, description, service_type, files))
            await conn.commit()


async def get_services(workspace_id: str) -> list[dict[str, Any]]:
    """Get all services for a workspace."""
    async with _cursor() as cur:
        await cur.execute(
            "SELECT id, name, description, service_type, files FROM services WHERE workspace_id = %s",
            (workspace_id,)
        )
        return [dict(row) for row in await cur.fetchall()]


async def get_external_calls(
    workspace_id: str,
    file_path: str,
    defined_names: set[str],
    references: list,
) -> dict[str, dict[str, list[tuple[str, str]]]]:
    """
    Get cross-file callers and callees for symbols in the current file.
    Used to enrich embeddings with architectural context.

    Returns:
        For each symbol_name in defined_names:
        {
            "callers": [(caller_name, caller_file_path), ...],
            "callees": [(callee_name, callee_file_path), ...]
        }
    """
    result: dict[str, dict[str, list[tuple[str, str]]]] = {
        name: {"callers": [], "callees": []} for name in defined_names
    }

    async with _cursor() as cur:
        # External callers: who (in other files) calls each of our symbols?
        for sym_name in defined_names:
            await cur.execute("""
                SELECT DISTINCT s.file_path, e.from_name
                FROM edges e
                JOIN symbols s ON s.workspace_id = e.workspace_id AND s.name = e.from_name
                WHERE e.workspace_id = %s AND e.to_name = %s AND e.edge_type = 'CALLS'
                  AND s.file_path != %s
            """, (workspace_id, sym_name, file_path))
            async for row in cur:
                result[sym_name]["callers"].append((row["from_name"], row["file_path"]))

        # External callees: for each of our symbols as caller, what external symbols do they call?
        # From references: (context_name, name) where name not in defined_names
        callee_refs: dict[str, set[str]] = {}  # context_name -> set of external callee names
        for ref in references:
            if not ref.context_name or ref.name in defined_names:
                continue
            callee_refs.setdefault(ref.context_name, set()).add(ref.name)

        for caller_name, callee_names in callee_refs.items():
            if caller_name not in result:
                continue
            seen: set[tuple[str, str]] = set()
            for callee_name in callee_names:
                await cur.execute("""
                    SELECT DISTINCT name, file_path FROM symbols
                    WHERE workspace_id = %s AND name = %s AND file_path != %s
                """, (workspace_id, callee_name, file_path))
                async for row in cur:
                    key = (row["name"], row["file_path"])
                    if key not in seen:
                        seen.add(key)
                        result[caller_name]["callees"].append((row["name"], row["file_path"]))

    return result


async def index_file_result(
    workspace_id: str,
    parse_result,
    embeddings_map: dict[str, list[float]],
    enriched_texts: dict[str, str] | None = None,
    classifications: dict[str, Any] | None = None,
) -> dict[str, int]:
    """
    Index a parsed file with embeddings and architectural metadata.
    
    classifications: Map of symbol_uid -> {service_id, architectural_role, description}
    """
    enriched_texts = enriched_texts or {}
    classifications = classifications or {}
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
                    
                    # Classification metadata
                    cls = classifications.get(defn_key, {})
                    service_id = cls.get("service_id", "")
                    arch_role = cls.get("architectural_role", "")
                    description = cls.get("description", "")

                    emb_str = f"[{','.join(str(x) for x in emb)}]" if emb else None

                    await cur.execute("""
                        INSERT INTO symbols (uid, workspace_id, name, kind, file_path, start_line, end_line,
                                             signature, content, enriched_content, parent, embedding,
                                             service_id, architectural_role, description)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (uid) DO UPDATE SET
                            name=%s, kind=%s, signature=%s, content=%s, enriched_content=%s, parent=%s, embedding=%s,
                            service_id=%s, architectural_role=%s, description=%s
                    """, (uid, workspace_id, defn.name, defn.kind, defn.file_path,
                          defn.start_line, defn.end_line, defn.signature, defn.content,
                          enriched, defn.parent or "", emb_str, service_id, arch_role, description,
                          defn.name, defn.kind, defn.signature, defn.content, enriched, defn.parent or "", emb_str,
                          service_id, arch_role, description))

                    stats["nodes_created"] += 1
                    stats["relationships_created"] += 1

                    if defn.parent:
                        await cur.execute("""
                            INSERT INTO edges (workspace_id, from_name, to_name, edge_type)
                            VALUES (%s, %s, %s, 'BELONGS_TO')
                        """, (workspace_id, defn.name, defn.parent))
                        stats["relationships_created"] += 1
                
                # Handle fallback whole-file embeddings
                if not parse_result.definitions:
                    file_key = f"{parse_result.file_path}:__file__:0"
                    emb = embeddings_map.get(file_key)
                    enriched = enriched_texts.get(file_key, "")
                    
                    cls = classifications.get(file_key, {})
                    service_id = cls.get("service_id", "")
                    arch_role = cls.get("architectural_role", "")
                    description = cls.get("description", "")
                    
                    if emb:
                        uid = f"{workspace_id}:{file_key}"
                        file_name = parse_result.file_path.rsplit("/", 1)[-1]
                        emb_str = f"[{','.join(str(x) for x in emb)}]"
                        
                        await cur.execute("""
                            INSERT INTO symbols (uid, workspace_id, name, kind, file_path, start_line, end_line,
                                                 signature, content, enriched_content, parent, embedding,
                                                 service_id, architectural_role, description)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (uid) DO UPDATE SET
                                name=%s, kind=%s, signature=%s, content=%s, enriched_content=%s, parent=%s, embedding=%s,
                                service_id=%s, architectural_role=%s, description=%s
                        """, (uid, workspace_id, file_name, "file", parse_result.file_path,
                              0, 0, f"File: {parse_result.file_path}", "", enriched, "", emb_str,
                              service_id, arch_role, description,
                              file_name, "file", f"File: {parse_result.file_path}", "", enriched, "", emb_str,
                              service_id, arch_role, description))
                        
                        stats["nodes_created"] += 1

                await conn.commit()
        except Exception as e:
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


async def vector_search_by_type(
    workspace_id: str,
    query_embedding: list[float],
    symbol_types: list[str],
    top_k: int = 10,
    score_threshold: float | None = None,
) -> list[dict[str, Any]]:
    """
    Type-filtered vector search using pgvector cosine distance.
    
    Args:
        workspace_id: Filter to this workspace
        query_embedding: The query vector
        symbol_types: List of symbol types to filter by (e.g., ['function', 'method'])
        top_k: Maximum results to return
        score_threshold: Minimum cosine similarity (0-1). Defaults to SCORE_THRESHOLD.
    """
    threshold = score_threshold if score_threshold is not None else SCORE_THRESHOLD
    
    async with _cursor() as cur:
        emb_str = f"[{','.join(str(x) for x in query_embedding)}]"
        types_placeholder = ','.join(['%s'] * len(symbol_types))

        # Set HNSW search parameter for quality vs speed tradeoff
        await cur.execute("SET hnsw.ef_search = 64")

        # Query with type filtering and score threshold
        await cur.execute(f"""
            SELECT uid, name, kind, file_path, start_line, end_line,
                   signature, content, parent,
                   1 - (embedding <=> %s::vector) AS score
            FROM symbols
            WHERE workspace_id = %s 
              AND embedding IS NOT NULL
              AND kind = ANY(%s)
              AND 1 - (embedding <=> %s::vector) > %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (emb_str, workspace_id, symbol_types, emb_str, threshold, emb_str, top_k))

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
    
    Level 0: Service architecture view (logical components)
    Level 1: Service drill-down (files within a service)
    Level 2: File drill-down (symbols within a file)
    Level 3: Symbol drill-down (call graph)
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

        elif focus_path and focus_path in ["api_endpoints", "pages_ui", "feature_components", "ui_components", "state_management", "utilities", "configuration"]:
            # Level 1: Component drill-down (files within a component)
            component_id = focus_path
            
            # Define file patterns for each component
            file_patterns = {
                "api_endpoints": ["pages/api/%"],
                "pages_ui": ["pages/%"],
                "feature_components": ["components/%"],
                "ui_components": ["components/ui/%"],
                "state_management": ["context/%"],
                "utilities": ["lib/%", "utils/%"],
                "configuration": ["%"]  # Catch-all for config files
            }
            
            patterns = file_patterns.get(component_id, [])
            
            # Get files that match the component patterns with proper exclusions
            files_to_add = []
            
            if component_id == "feature_components":
                # Special case: feature components excludes UI components
                await cur.execute("""
                    SELECT DISTINCT file_path, COUNT(*) as symbol_count
                    FROM symbols
                    WHERE workspace_id = %s 
                      AND file_path LIKE %s 
                      AND file_path NOT LIKE %s
                      AND kind = 'function'
                    GROUP BY file_path
                    ORDER BY file_path
                """, (workspace_id, 'components/%', 'components/ui/%'))
                files_to_add = await cur.fetchall()
                
            elif component_id == "pages_ui":
                # Special case: pages UI excludes API endpoints
                await cur.execute("""
                    SELECT DISTINCT file_path, COUNT(*) as symbol_count
                    FROM symbols
                    WHERE workspace_id = %s 
                      AND file_path LIKE %s 
                      AND file_path NOT LIKE %s
                      AND kind = 'function'
                    GROUP BY file_path
                    ORDER BY file_path
                """, (workspace_id, 'pages/%', 'pages/api/%'))
                files_to_add = await cur.fetchall()
                
            else:
                # Standard pattern matching for other components
                for pattern in patterns:
                    await cur.execute("""
                        SELECT DISTINCT file_path, COUNT(*) as symbol_count
                        FROM symbols
                        WHERE workspace_id = %s AND file_path LIKE %s AND kind = 'function'
                        GROUP BY file_path
                        ORDER BY file_path
                    """, (workspace_id, pattern))
                    pattern_files = await cur.fetchall()
                    files_to_add.extend(pattern_files)
            
            # Process the results
            for row in files_to_add:
                file_path = row["file_path"]
                
                # Additional filtering for configuration component
                if component_id == "configuration" and any(file_path.startswith(p.rstrip('%')) for p in ["pages/", "components/", "context/", "lib/", "utils/"]):
                    continue  # Skip files that belong to other components
                
                nodes.append({
                    "id": file_path,
                    "name": file_path.split("/")[-1],
                    "kind": "file",
                    "file_path": file_path,
                    "description": f"{row['symbol_count']} functions"
                })
            
            # Add CALLS edges between files in this component
            file_paths = [n["id"] for n in nodes]
            if file_paths:
                await cur.execute("""
                    SELECT DISTINCT s1.file_path as from_file, s2.file_path as to_file
                    FROM edges e
                    JOIN symbols s1 ON s1.workspace_id = e.workspace_id AND s1.name = e.from_name
                    JOIN symbols s2 ON s2.workspace_id = e.workspace_id AND s2.name = e.to_name
                    WHERE e.workspace_id = %s AND e.edge_type = 'CALLS'
                      AND s1.file_path = ANY(%s) AND s2.file_path = ANY(%s)
                """, (workspace_id, file_paths, file_paths))
                
                async for row in cur:
                    edges.append({"from": row["from_file"], "to": row["to_file"], "type": "CALLS"})

        elif focus_path and not focus_path.startswith("service_"):
            # Level 2: Mid-level view of symbols inside a file
            await cur.execute("""
                SELECT uid, name, kind, file_path, signature, parent, architectural_role, description
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
                    "signature": row["signature"],
                    "description": f"[{row['architectural_role']}] {row['description']}" if row['architectural_role'] else row['description']
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

        elif focus_path and focus_path.startswith("service_"):
            # Level 1: Service drill-down (files within a service)
            service_id = focus_path
            await cur.execute("""
                SELECT DISTINCT file_path, architectural_role, description
                FROM symbols
                WHERE workspace_id = %s AND service_id = %s
            """, (workspace_id, service_id))
            
            async for row in cur:
                nodes.append({
                    "id": row["file_path"],
                    "name": row["file_path"].split("/")[-1],
                    "kind": "file",
                    "file_path": row["file_path"],
                    "description": f"[{row['architectural_role']}] {row['description']}" if row['architectural_role'] else row['description']
                })
            
            # Add IMPORTS edges between these files
            file_paths = [n["id"] for n in nodes]
            if file_paths:
                placeholders = ', '.join(['%s'] * len(file_paths))
                await cur.execute(f"""
                    SELECT from_name, to_name
                    FROM edges
                    WHERE workspace_id = %s AND edge_type = 'IMPORTS'
                      AND from_name IN ({placeholders})
                      AND to_name IN ({placeholders})
                """, (workspace_id, *file_paths, *file_paths))
                async for row in cur:
                    edges.append({"from": row["from_name"], "to": row["to_name"], "type": "IMPORTS"})

        else:
            # Level 0: High-level service architecture view
            await cur.execute("""
                SELECT id, name, description, service_type, files
                FROM services
                WHERE workspace_id = %s
            """, (workspace_id,))
            
            service_nodes = []
            async for row in cur:
                s_node = {
                    "id": row["id"],
                    "name": row["name"],
                    "kind": "service",
                    "description": row["description"],
                    "file_count": len(row["files"]) if row["files"] else 0,
                    "files": row["files"] or []
                }
                service_nodes.append(s_node)
                nodes.append(s_node)

            # If no services found in table, try to infer from symbols
            if not nodes:
                await cur.execute("""
                    SELECT DISTINCT service_id
                    FROM symbols
                    WHERE workspace_id = %s AND service_id != ''
                """, (workspace_id,))
                
                s_ids = [row["service_id"] for row in await cur.fetchall()]
                for sid in s_ids:
                    # Get file count and some key symbols for this service
                    await cur.execute("""
                        SELECT COUNT(DISTINCT file_path) as f_count, 
                               COUNT(*) as s_count
                        FROM symbols
                        WHERE workspace_id = %s AND service_id = %s
                    """, (workspace_id, sid))
                    stats = await cur.fetchone()
                    
                    s_node = {
                        "id": sid,
                        "name": sid.replace("service_", "").replace("_", " ").title(),
                        "kind": "service",
                        "file_count": stats["f_count"],
                        "symbol_count": stats["s_count"]
                    }
                    nodes.append(s_node)
                    service_nodes.append(s_node)

            # If still no services found, create component-based architecture view
            if not nodes:
                component_groups = {}
                
                # Get all files and their symbols to analyze patterns
                await cur.execute("""
                    SELECT DISTINCT file_path, COUNT(*) as symbol_count
                    FROM symbols 
                    WHERE workspace_id = %s AND kind = 'function'
                    GROUP BY file_path
                    ORDER BY file_path
                """, (workspace_id,))
                
                files_data = await cur.fetchall()
                
                # Classify files into architectural components
                for file_data in files_data:
                    file_path = file_data["file_path"]
                    symbol_count = file_data["symbol_count"]
                    
                    # Determine component category
                    component_id = None
                    component_name = None
                    component_description = None
                    
                    if file_path.startswith("pages/"):
                        if "api/" in file_path:
                            component_id = "api_endpoints"
                            component_name = "API Endpoints"
                            component_description = "Backend API routes and handlers"
                        else:
                            component_id = "pages_ui"
                            component_name = "Pages & Routes"
                            component_description = "Frontend pages and routing"
                    elif file_path.startswith("components/"):
                        if "ui/" in file_path:
                            component_id = "ui_components"
                            component_name = "UI Components"
                            component_description = "Reusable UI components and design system"
                        else:
                            component_id = "feature_components"
                            component_name = "Feature Components"
                            component_description = "Business logic components"
                    elif file_path.startswith("context/"):
                        component_id = "state_management"
                        component_name = "State Management"
                        component_description = "React Context and global state"
                    elif file_path.startswith("lib/") or file_path.startswith("utils/"):
                        component_id = "utilities"
                        component_name = "Utilities"
                        component_description = "Helper functions and utilities"
                    else:
                        component_id = "configuration"
                        component_name = "Configuration"
                        component_description = "Config files and setup"
                    
                    # Add to component group
                    if component_id not in component_groups:
                        component_groups[component_id] = {
                            "id": component_id,
                            "name": component_name,
                            "kind": "component",
                            "description": component_description,
                            "files": [],
                            "symbol_count": 0,
                            "file_count": 0
                        }
                    
                    component_groups[component_id]["files"].append(file_path)
                    component_groups[component_id]["symbol_count"] += symbol_count
                    component_groups[component_id]["file_count"] += 1
                
                # Add component nodes
                for comp in component_groups.values():
                    nodes.append(comp)
                    service_nodes.append(comp)  # Treat components as services for edge logic

            # Add DEPENDS_ON edges between services/components
            for s1 in service_nodes:
                for s2 in service_nodes:
                    if s1["id"] == s2["id"]:
                        continue
                    
                    # For traditional services, check service_id
                    if s1.get("kind") == "service" and "files" not in s1:
                        await cur.execute("""
                            SELECT 1 FROM edges e
                            JOIN symbols sym1 ON sym1.workspace_id = e.workspace_id AND sym1.file_path = e.from_name
                            JOIN symbols sym2 ON sym2.workspace_id = e.workspace_id AND sym2.file_path = e.to_name
                            WHERE e.workspace_id = %s AND e.edge_type = 'IMPORTS'
                              AND sym1.service_id = %s AND sym2.service_id = %s
                            LIMIT 1
                        """, (workspace_id, s1["id"], s2["id"]))
                        
                        if await cur.fetchone():
                            edges.append({"from": s1["id"], "to": s2["id"], "type": "DEPENDS_ON"})
                    
                    # For component groups, check if any file in s1 calls functions in s2
                    elif s1.get("kind") == "component" and "files" in s1:
                        await cur.execute("""
                            SELECT 1 FROM edges e
                            JOIN symbols s1_sym ON s1_sym.workspace_id = e.workspace_id AND s1_sym.name = e.from_name
                            JOIN symbols s2_sym ON s2_sym.workspace_id = e.workspace_id AND s2_sym.name = e.to_name
                            WHERE e.workspace_id = %s AND e.edge_type = 'CALLS'
                              AND s1_sym.file_path = ANY(%s) AND s2_sym.file_path = ANY(%s)
                            LIMIT 1
                        """, (workspace_id, s1["files"], s2["files"]))
                        
                        if await cur.fetchone():
                            edges.append({"from": s1["id"], "to": s2["id"], "type": "DEPENDS_ON"})

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
