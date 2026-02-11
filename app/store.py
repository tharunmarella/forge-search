"""
Postgres + pgvector code store.

Everything in one database:
  - Symbols, files, edges in relational tables
  - 1024-dim embeddings with pgvector (Voyage voyage-code-3)
  - Graph traversal via recursive CTEs
  - No in-memory state needed (stateless API instances)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import psycopg2
import psycopg2.extras

from . import embeddings

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "")
VECTOR_DIMENSIONS = embeddings.get_dimensions()  # 1024 for voyage-code-3

# ── Connection ────────────────────────────────────────────────────

_conn = None


def _get_conn():
    global _conn
    if _conn is None or _conn.closed:
        _conn = psycopg2.connect(DATABASE_URL)
        _conn.autocommit = False
    return _conn


def _cursor():
    return _get_conn().cursor(cursor_factory=psycopg2.extras.RealDictCursor)


# ── Schema ────────────────────────────────────────────────────────

def _migrate_embedding_dimensions(cur):
    """Migrate embedding column from 768 dims (Jina) to 1024 dims (Voyage)."""
    # Check current column dimensions
    cur.execute("""
        SELECT a.atttypmod 
        FROM pg_attribute a
        JOIN pg_class c ON a.attrelid = c.oid
        WHERE c.relname = 'symbols' AND a.attname = 'embedding'
    """)
    row = cur.fetchone()
    if row:
        current_dims = row[0] if isinstance(row, tuple) else row.get('atttypmod', 0)
        if current_dims > 0 and current_dims != VECTOR_DIMENSIONS:
            logger.info(f"Migrating embedding column from {current_dims} to {VECTOR_DIMENSIONS} dimensions")
            # Clear embeddings and alter column type
            cur.execute("UPDATE symbols SET embedding = NULL")
            cur.execute(f"ALTER TABLE symbols ALTER COLUMN embedding TYPE vector({VECTOR_DIMENSIONS})")
            logger.info("Embedding dimension migration complete - reindexing required")


async def ensure_schema():
    """Create tables + pgvector index."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS files (
            uid TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL,
            path TEXT NOT NULL,
            content_hash TEXT,
            language TEXT
        )
    """)
    cur.execute("""
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
    _migrate_embedding_dimensions(cur)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS edges (
            id SERIAL PRIMARY KEY,
            workspace_id TEXT NOT NULL,
            from_name TEXT NOT NULL,
            to_name TEXT NOT NULL,
            edge_type TEXT NOT NULL
        )
    """)
    # Indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_symbols_ws ON symbols(workspace_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_symbols_ws_name ON symbols(workspace_id, name)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_symbols_ws_fp ON symbols(workspace_id, file_path)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_files_ws ON files(workspace_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_files_ws_path ON files(workspace_id, path)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_ws ON edges(workspace_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(workspace_id, from_name, edge_type)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_to ON edges(workspace_id, to_name, edge_type)")

    # pgvector index for fast similarity search
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_symbols_embedding
        ON symbols USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
    """)

    conn.commit()
    logger.info("Postgres schema ready (pgvector enabled)")


async def close_driver():
    global _conn
    if _conn:
        _conn.close()
        _conn = None


async def check_connection() -> bool:
    try:
        cur = _cursor()
        cur.execute("SELECT 1")
        cur.close()
        return True
    except Exception:
        return False


# ── Indexing ──────────────────────────────────────────────────────

async def clear_workspace(workspace_id: str):
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM symbols WHERE workspace_id = %s", (workspace_id,))
    cur.execute("DELETE FROM files WHERE workspace_id = %s", (workspace_id,))
    cur.execute("DELETE FROM edges WHERE workspace_id = %s", (workspace_id,))
    conn.commit()
    logger.info("Cleared workspace %s", workspace_id)


async def get_file_hash(workspace_id: str, file_path: str) -> str | None:
    cur = _cursor()
    cur.execute("SELECT content_hash FROM files WHERE uid = %s", (f"{workspace_id}:{file_path}",))
    row = cur.fetchone()
    cur.close()
    return row["content_hash"] if row else None


async def index_file_result(
    workspace_id: str,
    parse_result,
    embeddings: dict[str, list[float]],
    enriched_texts: dict[str, str] | None = None,
) -> dict[str, int]:
    conn = _get_conn()
    cur = conn.cursor()
    enriched_texts = enriched_texts or {}
    stats = {"nodes_created": 0, "relationships_created": 0}

    file_uid = f"{workspace_id}:{parse_result.file_path}"
    lang = _detect_language(parse_result.file_path)

    # Upsert file
    cur.execute("""
        INSERT INTO files (uid, workspace_id, path, content_hash, language)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (uid) DO UPDATE SET content_hash = %s, language = %s
    """, (file_uid, workspace_id, parse_result.file_path,
          parse_result.content_hash, lang, parse_result.content_hash, lang))
    stats["nodes_created"] += 1

    # Remove old symbols for this file
    cur.execute("DELETE FROM symbols WHERE workspace_id = %s AND file_path = %s",
                (workspace_id, parse_result.file_path))

    # Insert new symbols
    for defn in parse_result.definitions:
        uid = f"{workspace_id}:{defn.file_path}:{defn.name}:{defn.start_line}"
        defn_key = f"{defn.file_path}:{defn.name}:{defn.start_line}"
        emb = embeddings.get(defn_key)
        enriched = enriched_texts.get(defn_key, "")

        emb_str = f"[{','.join(str(x) for x in emb)}]" if emb else None

        cur.execute("""
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
            cur.execute("""
                INSERT INTO edges (workspace_id, from_name, to_name, edge_type)
                VALUES (%s, %s, %s, 'BELONGS_TO')
            """, (workspace_id, defn.name, defn.parent))
            stats["relationships_created"] += 1

    conn.commit()
    return stats


async def build_call_edges(workspace_id: str, references):
    conn = _get_conn()
    cur = conn.cursor()
    for ref in references:
        if not ref.context_name or ref.context_name == ref.name:
            continue
        cur.execute("""
            INSERT INTO edges (workspace_id, from_name, to_name, edge_type)
            SELECT %s, %s, %s, 'CALLS'
            WHERE NOT EXISTS (
                SELECT 1 FROM edges WHERE workspace_id=%s AND from_name=%s AND to_name=%s AND edge_type='CALLS'
            )
        """, (workspace_id, ref.context_name, ref.name,
              workspace_id, ref.context_name, ref.name))
    conn.commit()


async def build_import_edges(workspace_id: str, file_path: str, imports: list[str]):
    conn = _get_conn()
    cur = conn.cursor()
    for imp in imports:
        hint = _import_to_path_hint(imp)
        cur.execute("""
            INSERT INTO edges (workspace_id, from_name, to_name, edge_type)
            SELECT %s, %s, f.path, 'IMPORTS'
            FROM files f
            WHERE f.workspace_id = %s AND f.path LIKE %s AND f.path != %s
            AND NOT EXISTS (
                SELECT 1 FROM edges WHERE workspace_id=%s AND from_name=%s AND to_name=f.path AND edge_type='IMPORTS'
            )
        """, (workspace_id, file_path, workspace_id, f"%{hint}%", file_path,
              workspace_id, file_path))
    conn.commit()


# ── Vector search (pgvector!) ─────────────────────────────────────

async def vector_search(
    workspace_id: str,
    query_embedding: list[float],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Native vector search using pgvector cosine distance."""
    cur = _cursor()
    emb_str = f"[{','.join(str(x) for x in query_embedding)}]"

    cur.execute("""
        SELECT uid, name, kind, file_path, start_line, end_line,
               signature, content, parent,
               1 - (embedding <=> %s::vector) AS score
        FROM symbols
        WHERE workspace_id = %s AND embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (emb_str, workspace_id, emb_str, top_k))

    results = []
    for row in cur.fetchall():
        results.append({
            "symbol": dict(row),
            "score": float(row["score"]),
        })
    cur.close()
    return results


# ── Graph traversal ───────────────────────────────────────────────

async def expand_neighbors(workspace_id: str, symbol_uids: list[str], hops: int = 1):
    cur = _cursor()
    names = []
    for uid in symbol_uids:
        cur.execute("SELECT name FROM symbols WHERE uid = %s", (uid,))
        row = cur.fetchone()
        if row:
            names.append(row["name"])

    if not names:
        return []

    results = []
    for name in names:
        cur.execute("""
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
        results.extend([dict(r) for r in cur.fetchall()])
    cur.close()
    return results


async def trace_call_chain(workspace_id: str, symbol_name: str, direction: str = "both", max_depth: int = 5):
    """Deep BFS using recursive CTE — the Postgres superpower."""
    cur = _cursor()
    nodes = {}
    edges_list = []

    # Root
    cur.execute("""
        SELECT uid, name, kind, file_path, signature, start_line, end_line
        FROM symbols WHERE workspace_id = %s AND name = %s
    """, (workspace_id, symbol_name))
    roots = [dict(r) for r in cur.fetchall()]
    if not roots:
        cur.close()
        return {"nodes": [], "edges": [], "root": symbol_name, "depth_reached": 0}

    for root in roots:
        nodes[root["uid"]] = {**root, "depth": 0, "direction": "root"}

    # Downstream: recursive CTE
    if direction in ("downstream", "both"):
        cur.execute("""
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

        for row in cur.fetchall():
            r = dict(row)
            if r["uid"] not in nodes:
                nodes[r["uid"]] = {
                    "uid": r["uid"], "name": r["name"], "kind": r["kind"],
                    "file_path": r["file_path"], "signature": r["signature"],
                    "start_line": r["start_line"], "end_line": r["end_line"],
                    "depth": r["depth"], "direction": "downstream",
                }
            edges_list.append({"from": r["from_name"], "to": r["to_name"], "type": "CALLS"})

    # Upstream: recursive CTE
    if direction in ("upstream", "both"):
        cur.execute("""
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

        for row in cur.fetchall():
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

    cur.close()
    return {
        "root": symbol_name,
        "nodes": list(nodes.values()),
        "edges": unique_edges,
        "depth_reached": max((n.get("depth", 0) for n in nodes.values()), default=0),
    }


async def impact_analysis(workspace_id: str, symbol_name: str, max_depth: int = 4):
    """Blast radius using recursive CTE — find all upstream dependents."""
    cur = _cursor()
    affected = {}

    # Upstream callers (recursive)
    cur.execute("""
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

    for row in cur.fetchall():
        r = dict(row)
        if r["uid"] not in affected:
            affected[r["uid"]] = r

    # Sibling members
    cur.execute("""
        SELECT s.uid, s.name, s.kind, s.file_path, s.signature,
               s.start_line, s.end_line, 1 as distance, 'sibling_member' as impact_type
        FROM edges e1
        JOIN edges e2 ON e2.to_name = e1.to_name AND e2.workspace_id = e1.workspace_id AND e2.edge_type = 'BELONGS_TO'
        JOIN symbols s ON s.workspace_id = e1.workspace_id AND s.name = e2.from_name
        WHERE e1.workspace_id = %s AND e1.from_name = %s AND e1.edge_type = 'BELONGS_TO'
        AND e2.from_name != %s
    """, (workspace_id, symbol_name, symbol_name))

    for row in cur.fetchall():
        r = dict(row)
        if r["uid"] not in affected:
            affected[r["uid"]] = r

    cur.close()

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


async def get_workspace_stats(workspace_id: str) -> dict[str, int]:
    cur = _cursor()
    cur.execute("SELECT COUNT(*) as c FROM symbols WHERE workspace_id = %s", (workspace_id,))
    symbols = cur.fetchone()["c"]
    cur.execute("SELECT COUNT(*) as c FROM files WHERE workspace_id = %s", (workspace_id,))
    files = cur.fetchone()["c"]
    cur.close()
    return {"symbols": symbols, "files": files}


# ── Watcher support ───────────────────────────────────────────────

def _get_ws(workspace_id: str):
    """Compatibility shim for watcher.py — returns a namespace with called_by + symbols."""
    cur = _cursor()

    class _WS:
        def __init__(self):
            self.symbols = {}
            self.called_by = {}
            self.files = {}

    ws = _WS()

    cur.execute("SELECT uid, name, kind, file_path, start_line, end_line, signature, content, parent FROM symbols WHERE workspace_id = %s", (workspace_id,))
    for row in cur.fetchall():
        r = dict(row)
        ws.symbols[r["uid"]] = r

    cur.execute("SELECT from_name, to_name FROM edges WHERE workspace_id = %s AND edge_type = 'CALLS'", (workspace_id,))
    for row in cur.fetchall():
        ws.called_by.setdefault(row["to_name"], set()).add(row["from_name"])

    cur.execute("SELECT uid, path, content_hash FROM files WHERE workspace_id = %s", (workspace_id,))
    for row in cur.fetchall():
        ws.files[row["uid"]] = dict(row)

    cur.close()
    return ws


# ── Utilities ─────────────────────────────────────────────────────

def _detect_language(file_path: str) -> str:
    ext_map = {
        ".rs": "rust", ".py": "python", ".js": "javascript",
        ".ts": "typescript", ".tsx": "typescript", ".go": "go",
        ".java": "java", ".cpp": "cpp", ".c": "c", ".h": "c",
    }
    from pathlib import Path
    return ext_map.get(Path(file_path).suffix, "unknown")


def _import_to_path_hint(imp: str) -> str:
    hint = imp.replace("::", "/").replace(".", "/")
    for prefix in ("crate/", "self/", "super/"):
        if hint.startswith(prefix):
            hint = hint[len(prefix):]
    return hint
