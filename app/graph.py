"""
Neo4j graph layer.

Manages the connection, schema (constraints + vector index), and all
Cypher operations for indexing and querying the code graph.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession

from .parser import SymbolDef, SymbolRef, FileParseResult

logger = logging.getLogger(__name__)

# ── Vector index config ──────────────────────────────────────────

VECTOR_DIMENSIONS = 768  # Gemini text-embedding-004 output dimension
VECTOR_INDEX_NAME = "symbol_embedding_index"
SIMILARITY_FUNCTION = "cosine"


# ── Connection ────────────────────────────────────────────────────

_driver: AsyncDriver | None = None


async def get_driver() -> AsyncDriver:
    global _driver
    if _driver is None:
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        _driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        # Verify connectivity
        await _driver.verify_connectivity()
        logger.info("Connected to Neo4j at %s", uri)
    return _driver


async def close_driver():
    global _driver
    if _driver:
        await _driver.close()
        _driver = None


async def check_connection() -> bool:
    try:
        driver = await get_driver()
        await driver.verify_connectivity()
        return True
    except Exception:
        return False


# ── Schema setup ──────────────────────────────────────────────────

async def ensure_schema():
    """Create constraints, indexes, and the vector index if they don't exist."""
    driver = await get_driver()
    async with driver.session() as session:
        # Uniqueness constraints
        await session.run(
            "CREATE CONSTRAINT IF NOT EXISTS FOR (f:File) REQUIRE f.uid IS UNIQUE"
        )
        await session.run(
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Symbol) REQUIRE s.uid IS UNIQUE"
        )
        # Composite index for fast workspace scoping
        await session.run(
            "CREATE INDEX IF NOT EXISTS FOR (s:Symbol) ON (s.workspace_id, s.name)"
        )
        await session.run(
            "CREATE INDEX IF NOT EXISTS FOR (f:File) ON (f.workspace_id, f.path)"
        )

        # Vector index for semantic search
        # Neo4j 5.x vector index syntax
        try:
            await session.run(f"""
                CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS
                FOR (s:Symbol)
                ON (s.embedding)
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: {VECTOR_DIMENSIONS},
                        `vector.similarity_function`: '{SIMILARITY_FUNCTION}'
                    }}
                }}
            """)
            logger.info("Vector index '%s' ensured", VECTOR_INDEX_NAME)
        except Exception as e:
            logger.warning("Vector index creation: %s", e)

    logger.info("Neo4j schema ensured")


# ── Indexing ──────────────────────────────────────────────────────

async def clear_workspace(workspace_id: str):
    """Remove all nodes and relationships for a workspace."""
    driver = await get_driver()
    async with driver.session() as session:
        await session.run(
            "MATCH (n {workspace_id: $ws}) DETACH DELETE n",
            ws=workspace_id,
        )
    logger.info("Cleared workspace %s", workspace_id)


async def get_file_hash(workspace_id: str, file_path: str) -> str | None:
    """Return the stored content hash for a file, or None if not indexed."""
    driver = await get_driver()
    async with driver.session() as session:
        result = await session.run(
            "MATCH (f:File {workspace_id: $ws, path: $path}) RETURN f.content_hash AS hash",
            ws=workspace_id, path=file_path,
        )
        record = await result.single()
        return record["hash"] if record else None


async def index_file_result(
    workspace_id: str,
    parse_result: FileParseResult,
    embeddings: dict[str, list[float]],
) -> dict[str, int]:
    """
    Index a single file's parse result into Neo4j.

    Parameters
    ----------
    workspace_id : str
        Workspace identifier for multi-tenant isolation.
    parse_result : FileParseResult
        Tree-sitter extracted definitions and references.
    embeddings : dict[str, list[float]]
        Mapping from symbol content -> embedding vector.

    Returns
    -------
    dict with counts: nodes_created, relationships_created
    """
    driver = await get_driver()
    stats = {"nodes_created": 0, "relationships_created": 0}

    async with driver.session() as session:
        # 1. Upsert File node
        await session.run("""
            MERGE (f:File {uid: $uid})
            SET f.workspace_id = $ws,
                f.path = $path,
                f.content_hash = $hash,
                f.language = $lang
        """,
            uid=f"{workspace_id}:{parse_result.file_path}",
            ws=workspace_id,
            path=parse_result.file_path,
            hash=parse_result.content_hash,
            lang=_detect_language(parse_result.file_path),
        )
        stats["nodes_created"] += 1

        # 2. Remove old symbols for this file (clean re-index)
        await session.run("""
            MATCH (s:Symbol {workspace_id: $ws, file_path: $path})
            DETACH DELETE s
        """, ws=workspace_id, path=parse_result.file_path)

        # 3. Create Symbol nodes for each definition
        for defn in parse_result.definitions:
            uid = f"{workspace_id}:{defn.file_path}:{defn.name}:{defn.start_line}"
            embedding = embeddings.get(defn.content)

            params: dict[str, Any] = {
                "uid": uid,
                "ws": workspace_id,
                "name": defn.name,
                "kind": defn.kind,
                "file_path": defn.file_path,
                "start_line": defn.start_line,
                "end_line": defn.end_line,
                "signature": defn.signature,
                "content": defn.content,
                "parent": defn.parent,
                "file_uid": f"{workspace_id}:{defn.file_path}",
            }

            if embedding is not None:
                params["embedding"] = embedding
                await session.run("""
                    MERGE (s:Symbol {uid: $uid})
                    SET s.workspace_id = $ws,
                        s.name = $name,
                        s.kind = $kind,
                        s.file_path = $file_path,
                        s.start_line = $start_line,
                        s.end_line = $end_line,
                        s.signature = $signature,
                        s.content = $content,
                        s.parent = $parent,
                        s.embedding = $embedding
                    WITH s
                    MATCH (f:File {uid: $file_uid})
                    MERGE (s)-[:DEFINED_IN]->(f)
                """, **params)
            else:
                await session.run("""
                    MERGE (s:Symbol {uid: $uid})
                    SET s.workspace_id = $ws,
                        s.name = $name,
                        s.kind = $kind,
                        s.file_path = $file_path,
                        s.start_line = $start_line,
                        s.end_line = $end_line,
                        s.signature = $signature,
                        s.content = $content,
                        s.parent = $parent
                    WITH s
                    MATCH (f:File {uid: $file_uid})
                    MERGE (s)-[:DEFINED_IN]->(f)
                """, **params)

            stats["nodes_created"] += 1
            stats["relationships_created"] += 1

            # BELONGS_TO relationship for methods
            if defn.parent:
                await session.run("""
                    MATCH (m:Symbol {uid: $method_uid})
                    MATCH (p:Symbol {workspace_id: $ws, name: $parent_name, kind: $parent_kind})
                    WHERE p.kind IN ['struct', 'class', 'enum', 'trait', 'interface']
                    MERGE (m)-[:BELONGS_TO]->(p)
                """,
                    method_uid=uid,
                    ws=workspace_id,
                    parent_name=defn.parent,
                    parent_kind=defn.kind,
                )
                stats["relationships_created"] += 1

    return stats


async def build_call_edges(workspace_id: str, references: list[SymbolRef]):
    """
    Build CALLS relationships from references to definitions.
    A reference from function A to symbol B creates (A)-[:CALLS]->(B).
    """
    driver = await get_driver()
    async with driver.session() as session:
        for ref in references:
            if not ref.context_name:
                continue
            # Link caller -> callee
            result = await session.run("""
                MATCH (caller:Symbol {workspace_id: $ws, name: $caller_name})
                MATCH (callee:Symbol {workspace_id: $ws, name: $callee_name})
                WHERE caller <> callee
                MERGE (caller)-[r:CALLS]->(callee)
                ON CREATE SET r.count = 1
                ON MATCH SET r.count = r.count + 1
                RETURN count(r) AS created
            """,
                ws=workspace_id,
                caller_name=ref.context_name,
                callee_name=ref.name,
            )
            await result.consume()


async def build_import_edges(workspace_id: str, file_path: str, imports: list[str]):
    """Build IMPORTS edges between File nodes based on import statements."""
    driver = await get_driver()
    async with driver.session() as session:
        for imp in imports:
            # Try to resolve import to a file in the workspace
            await session.run("""
                MATCH (src:File {workspace_id: $ws, path: $src_path})
                MATCH (dst:File {workspace_id: $ws})
                WHERE dst.path CONTAINS $imp_hint
                  AND src <> dst
                MERGE (src)-[:IMPORTS]->(dst)
            """,
                ws=workspace_id,
                src_path=file_path,
                imp_hint=_import_to_path_hint(imp),
            )


# ── Search ────────────────────────────────────────────────────────

async def vector_search(
    workspace_id: str,
    query_embedding: list[float],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """
    Perform vector similarity search on Symbol nodes.
    Returns top_k results with similarity score.
    """
    driver = await get_driver()
    async with driver.session() as session:
        result = await session.run(f"""
            CALL db.index.vector.queryNodes('{VECTOR_INDEX_NAME}', $top_k, $embedding)
            YIELD node, score
            WHERE node.workspace_id = $ws
            RETURN node {{
                .uid, .name, .kind, .file_path, .start_line, .end_line,
                .signature, .content, .parent
            }} AS symbol, score
            ORDER BY score DESC
        """,
            top_k=top_k * 2,  # Fetch extra to account for workspace filtering
            embedding=query_embedding,
            ws=workspace_id,
        )
        records = [record async for record in result]
        return [
            {"symbol": dict(record["symbol"]), "score": record["score"]}
            for record in records[:top_k]
        ]


async def expand_neighbors(
    workspace_id: str,
    symbol_uids: list[str],
    hops: int = 1,
) -> list[dict[str, Any]]:
    """
    Expand from a set of symbols by 1-2 hops to find related symbols.
    Returns callers, callees, and type relationships.
    """
    driver = await get_driver()
    async with driver.session() as session:
        result = await session.run("""
            UNWIND $uids AS uid
            MATCH (s:Symbol {uid: uid})
            CALL {
                WITH s
                MATCH (s)-[:CALLS]->(callee:Symbol)
                RETURN callee AS related, 'calls' AS rel_type
                UNION
                WITH s
                MATCH (caller:Symbol)-[:CALLS]->(s)
                RETURN caller AS related, 'called_by' AS rel_type
                UNION
                WITH s
                MATCH (s)-[:BELONGS_TO]->(parent:Symbol)
                RETURN parent AS related, 'belongs_to' AS rel_type
                UNION
                WITH s
                MATCH (child:Symbol)-[:BELONGS_TO]->(s)
                RETURN child AS related, 'has_member' AS rel_type
            }
            RETURN DISTINCT
                related.uid AS uid,
                related.name AS name,
                related.file_path AS file_path,
                related.signature AS signature,
                related.kind AS kind,
                rel_type
        """, uids=symbol_uids)

        records = [record async for record in result]
        return [dict(r) for r in records]


async def get_workspace_stats(workspace_id: str) -> dict[str, int]:
    """Return counts of nodes and relationships for a workspace."""
    driver = await get_driver()
    async with driver.session() as session:
        result = await session.run("""
            MATCH (s:Symbol {workspace_id: $ws})
            WITH count(s) AS symbols
            MATCH (f:File {workspace_id: $ws})
            WITH symbols, count(f) AS files
            RETURN symbols, files
        """, ws=workspace_id)
        record = await result.single()
        if record:
            return {"symbols": record["symbols"], "files": record["files"]}
        return {"symbols": 0, "files": 0}


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
    """Convert an import statement to a file path hint for matching."""
    # "crate::tools::search" -> "tools/search"
    hint = imp.replace("::", "/").replace(".", "/")
    # Remove common prefixes
    for prefix in ("crate/", "self/", "super/"):
        if hint.startswith(prefix):
            hint = hint[len(prefix):]
    return hint
