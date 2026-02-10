"""
In-memory code store with numpy vector search and SQLite persistence.

Replaces Neo4j with:
  - numpy arrays for cosine similarity search  (~1ms for 10K symbols)
  - Python dicts for graph adjacency lists      (~0ms for BFS traversal)
  - SQLite for durable persistence              (~0ms reads, zero config)

All data stays in-process. No network hops. No external services.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np

from .parser import SymbolDef, SymbolRef, FileParseResult

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────

VECTOR_DIMENSIONS = 768
DB_PATH = os.getenv("STORE_DB_PATH", "data/forge_search.db")


# ── In-memory workspace data ─────────────────────────────────────

class _WorkspaceData:
    """All data for one workspace, held entirely in memory."""

    __slots__ = (
        "symbols", "files",
        "embeddings", "_matrix", "_matrix_uids", "_matrix_dirty",
        "calls", "called_by", "belongs_to", "has_members",
        "file_imports", "_name_index",
    )

    def __init__(self):
        # Symbol storage: uid -> {name, kind, file_path, ...}
        self.symbols: dict[str, dict[str, Any]] = {}
        # File storage: file_uid -> {path, content_hash, language}
        self.files: dict[str, dict[str, Any]] = {}

        # Embeddings: uid -> numpy array
        self.embeddings: dict[str, np.ndarray] = {}
        self._matrix: np.ndarray | None = None
        self._matrix_uids: list[str] = []
        self._matrix_dirty: bool = True

        # Graph adjacency lists (by symbol name, not uid, for cross-file linking)
        self.calls: dict[str, set[str]] = {}        # caller -> {callees}
        self.called_by: dict[str, set[str]] = {}    # callee -> {callers}
        self.belongs_to: dict[str, str] = {}         # child -> parent
        self.has_members: dict[str, set[str]] = {}   # parent -> {children}
        self.file_imports: dict[str, set[str]] = {}  # file_path -> {imported_paths}
        self._name_index: dict[str, str] = {}         # name -> first uid (cache)

    def get_matrix(self) -> tuple[np.ndarray, list[str]]:
        """Return the (N, 768) matrix and aligned uid list. Cached."""
        if self._matrix_dirty or self._matrix is None:
            if self.embeddings:
                uids = list(self.embeddings.keys())
                vecs = [self.embeddings[uid] for uid in uids]
                self._matrix = np.vstack(vecs)
                self._matrix_uids = uids
            else:
                self._matrix = np.empty((0, VECTOR_DIMENSIONS), dtype=np.float32)
                self._matrix_uids = []
            self._matrix_dirty = False
        return self._matrix, self._matrix_uids

    def invalidate_matrix(self):
        self._matrix_dirty = True


# ── Store singleton ───────────────────────────────────────────────

_workspaces: dict[str, _WorkspaceData] = {}
_db: sqlite3.Connection | None = None


def _get_ws(workspace_id: str) -> _WorkspaceData:
    if workspace_id not in _workspaces:
        _workspaces[workspace_id] = _WorkspaceData()
    return _workspaces[workspace_id]


def _get_db() -> sqlite3.Connection:
    global _db
    if _db is None:
        db_path = Path(DB_PATH)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _db = sqlite3.connect(str(db_path))
        _db.execute("PRAGMA journal_mode=WAL")
        _db.execute("PRAGMA synchronous=NORMAL")
        _db.row_factory = sqlite3.Row
    return _db


# ── Schema ────────────────────────────────────────────────────────

async def ensure_schema():
    """Create SQLite tables and load data into memory."""
    db = _get_db()
    db.executescript("""
        CREATE TABLE IF NOT EXISTS files (
            uid TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL,
            path TEXT NOT NULL,
            content_hash TEXT,
            language TEXT
        );
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
            embedding BLOB
        );
        CREATE TABLE IF NOT EXISTS edges (
            workspace_id TEXT NOT NULL,
            from_name TEXT NOT NULL,
            to_name TEXT NOT NULL,
            edge_type TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_symbols_ws ON symbols(workspace_id);
        CREATE INDEX IF NOT EXISTS idx_symbols_ws_name ON symbols(workspace_id, name);
        CREATE INDEX IF NOT EXISTS idx_files_ws ON files(workspace_id);
        CREATE INDEX IF NOT EXISTS idx_files_ws_path ON files(workspace_id, path);
        CREATE INDEX IF NOT EXISTS idx_edges_ws ON edges(workspace_id);
        CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(workspace_id, from_name, edge_type);
        CREATE INDEX IF NOT EXISTS idx_edges_to ON edges(workspace_id, to_name, edge_type);
    """)

    # Load existing data into memory
    _load_all_from_db()
    stats = {ws_id: len(ws.symbols) for ws_id, ws in _workspaces.items()}
    if stats:
        logger.info("Loaded from SQLite: %s", stats)
    logger.info("Store ready (SQLite: %s)", DB_PATH)


def _load_all_from_db():
    """Rebuild in-memory state from SQLite."""
    db = _get_db()

    # Load files
    for row in db.execute("SELECT * FROM files"):
        ws = _get_ws(row["workspace_id"])
        ws.files[row["uid"]] = dict(row)

    # Load symbols + embeddings
    for row in db.execute("SELECT * FROM symbols"):
        ws = _get_ws(row["workspace_id"])
        sym = dict(row)
        emb_blob = sym.pop("embedding", None)
        ws.symbols[row["uid"]] = sym
        if emb_blob:
            ws.embeddings[row["uid"]] = np.frombuffer(emb_blob, dtype=np.float32).copy()
    for ws in _workspaces.values():
        ws.invalidate_matrix()

    # Load edges
    for row in db.execute("SELECT * FROM edges"):
        ws = _get_ws(row["workspace_id"])
        etype = row["edge_type"]
        fn, tn = row["from_name"], row["to_name"]
        if etype == "CALLS":
            ws.calls.setdefault(fn, set()).add(tn)
            ws.called_by.setdefault(tn, set()).add(fn)
        elif etype == "BELONGS_TO":
            ws.belongs_to[fn] = tn
            ws.has_members.setdefault(tn, set()).add(fn)
        elif etype == "IMPORTS":
            ws.file_imports.setdefault(fn, set()).add(tn)


async def close_driver():
    """Close the SQLite connection."""
    global _db
    if _db:
        _db.close()
        _db = None


async def check_connection() -> bool:
    return True  # always available — it's in-process


# ── Indexing ──────────────────────────────────────────────────────

async def clear_workspace(workspace_id: str):
    db = _get_db()
    db.execute("DELETE FROM symbols WHERE workspace_id = ?", (workspace_id,))
    db.execute("DELETE FROM files WHERE workspace_id = ?", (workspace_id,))
    db.execute("DELETE FROM edges WHERE workspace_id = ?", (workspace_id,))
    db.commit()
    _workspaces.pop(workspace_id, None)
    logger.info("Cleared workspace %s", workspace_id)


async def get_file_hash(workspace_id: str, file_path: str) -> str | None:
    ws = _get_ws(workspace_id)
    file_uid = f"{workspace_id}:{file_path}"
    f = ws.files.get(file_uid)
    return f["content_hash"] if f else None


async def index_file_result(
    workspace_id: str,
    parse_result: FileParseResult,
    embeddings: dict[str, list[float]],
    enriched_texts: dict[str, str] | None = None,
) -> dict[str, int]:
    """Index a file's parse result. Same signature as the old graph.py."""
    ws = _get_ws(workspace_id)
    db = _get_db()
    enriched_texts = enriched_texts or {}
    stats = {"nodes_created": 0, "relationships_created": 0}

    file_uid = f"{workspace_id}:{parse_result.file_path}"
    lang = _detect_language(parse_result.file_path)

    # 1. Upsert file
    file_data = {
        "uid": file_uid, "workspace_id": workspace_id,
        "path": parse_result.file_path,
        "content_hash": parse_result.content_hash,
        "language": lang,
    }
    ws.files[file_uid] = file_data
    db.execute(
        "INSERT OR REPLACE INTO files (uid, workspace_id, path, content_hash, language) "
        "VALUES (?, ?, ?, ?, ?)",
        (file_uid, workspace_id, parse_result.file_path,
         parse_result.content_hash, lang),
    )
    stats["nodes_created"] += 1

    # 2. Remove old symbols for this file
    old_uids = [uid for uid, s in ws.symbols.items()
                 if s["file_path"] == parse_result.file_path]
    for uid in old_uids:
        ws.symbols.pop(uid, None)
        ws.embeddings.pop(uid, None)
    if old_uids:
        db.execute(
            f"DELETE FROM symbols WHERE workspace_id = ? AND file_path = ?",
            (workspace_id, parse_result.file_path),
        )

    # 3. Create symbol nodes
    for defn in parse_result.definitions:
        uid = f"{workspace_id}:{defn.file_path}:{defn.name}:{defn.start_line}"
        defn_key = f"{defn.file_path}:{defn.name}:{defn.start_line}"
        emb = embeddings.get(defn_key)
        enriched = enriched_texts.get(defn_key, "")

        sym = {
            "uid": uid, "workspace_id": workspace_id,
            "name": defn.name, "kind": defn.kind,
            "file_path": defn.file_path,
            "start_line": defn.start_line, "end_line": defn.end_line,
            "signature": defn.signature, "content": defn.content,
            "enriched_content": enriched, "parent": defn.parent or "",
        }
        ws.symbols[uid] = sym

        emb_blob = None
        if emb is not None:
            vec = np.array(emb, dtype=np.float32)
            ws.embeddings[uid] = vec
            emb_blob = vec.tobytes()

        db.execute(
            "INSERT OR REPLACE INTO symbols "
            "(uid, workspace_id, name, kind, file_path, start_line, end_line, "
            " signature, content, enriched_content, parent, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (uid, workspace_id, defn.name, defn.kind, defn.file_path,
             defn.start_line, defn.end_line, defn.signature, defn.content,
             enriched, defn.parent or "", emb_blob),
        )

        stats["nodes_created"] += 1
        stats["relationships_created"] += 1  # DEFINED_IN

        # BELONGS_TO
        if defn.parent:
            ws.belongs_to[defn.name] = defn.parent
            ws.has_members.setdefault(defn.parent, set()).add(defn.name)
            db.execute(
                "INSERT INTO edges (workspace_id, from_name, to_name, edge_type) "
                "VALUES (?, ?, ?, 'BELONGS_TO')",
                (workspace_id, defn.name, defn.parent),
            )
            stats["relationships_created"] += 1

    ws.invalidate_matrix()
    db.commit()
    return stats


async def build_call_edges(workspace_id: str, references: list[SymbolRef]):
    ws = _get_ws(workspace_id)
    db = _get_db()
    defined_names = {s["name"] for s in ws.symbols.values()}

    for ref in references:
        if not ref.context_name:
            continue
        caller, callee = ref.context_name, ref.name
        if caller == callee:
            continue
        # Only create edges between known symbols
        if caller in defined_names and callee in defined_names:
            if callee not in ws.calls.get(caller, set()):
                ws.calls.setdefault(caller, set()).add(callee)
                ws.called_by.setdefault(callee, set()).add(caller)
                db.execute(
                    "INSERT INTO edges (workspace_id, from_name, to_name, edge_type) "
                    "VALUES (?, ?, ?, 'CALLS')",
                    (workspace_id, caller, callee),
                )
    db.commit()


async def build_import_edges(workspace_id: str, file_path: str, imports: list[str]):
    ws = _get_ws(workspace_id)
    db = _get_db()
    known_paths = {f["path"] for f in ws.files.values()}

    for imp in imports:
        hint = _import_to_path_hint(imp)
        for kp in known_paths:
            if hint in kp and kp != file_path:
                if kp not in ws.file_imports.get(file_path, set()):
                    ws.file_imports.setdefault(file_path, set()).add(kp)
                    db.execute(
                        "INSERT INTO edges (workspace_id, from_name, to_name, edge_type) "
                        "VALUES (?, ?, ?, 'IMPORTS')",
                        (workspace_id, file_path, kp),
                    )
    db.commit()


# ── Vector search ─────────────────────────────────────────────────

async def vector_search(
    workspace_id: str,
    query_embedding: list[float],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """
    Cosine similarity search using numpy. ~1ms for 10K symbols.
    """
    ws = _get_ws(workspace_id)
    matrix, uids = ws.get_matrix()

    if matrix.shape[0] == 0:
        return []

    t0 = time.monotonic()

    query = np.array(query_embedding, dtype=np.float32)
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return []

    # Batch cosine similarity: (N, 768) @ (768,) -> (N,)
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1  # avoid division by zero
    scores = (matrix @ query) / (norms * query_norm)

    # Top-k
    if len(scores) <= top_k:
        top_indices = np.argsort(scores)[::-1]
    else:
        # argpartition is O(n) vs O(n log n) for full sort
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    elapsed = (time.monotonic() - t0) * 1000
    logger.debug("Vector search: %d vectors in %.1fms", matrix.shape[0], elapsed)

    results = []
    for idx in top_indices:
        uid = uids[idx]
        sym = ws.symbols.get(uid)
        if sym is None:
            continue
        results.append({
            "symbol": {
                "uid": uid, "name": sym["name"], "kind": sym["kind"],
                "file_path": sym["file_path"],
                "start_line": sym["start_line"], "end_line": sym["end_line"],
                "signature": sym["signature"], "content": sym["content"],
                "parent": sym.get("parent", ""),
            },
            "score": float(scores[idx]),
        })

    return results


# ── Graph traversal ───────────────────────────────────────────────

async def expand_neighbors(
    workspace_id: str,
    symbol_uids: list[str],
    hops: int = 1,
) -> list[dict[str, Any]]:
    """1-hop neighbor expansion for /search results."""
    ws = _get_ws(workspace_id)
    results = []
    seen = set()

    for uid in symbol_uids:
        sym = ws.symbols.get(uid)
        if not sym:
            continue
        name = sym["name"]

        # Callees
        for callee in ws.calls.get(name, set()):
            if callee not in seen:
                seen.add(("calls", callee))
                callee_sym = _find_symbol_by_name(ws, callee)
                if callee_sym:
                    results.append({
                        "name": callee, "rel_type": "calls",
                        "file_path": callee_sym["file_path"],
                        "signature": callee_sym.get("signature", ""),
                        "kind": callee_sym["kind"],
                    })

        # Callers
        for caller in ws.called_by.get(name, set()):
            if caller not in seen:
                seen.add(("called_by", caller))
                caller_sym = _find_symbol_by_name(ws, caller)
                if caller_sym:
                    results.append({
                        "name": caller, "rel_type": "called_by",
                        "file_path": caller_sym["file_path"],
                        "signature": caller_sym.get("signature", ""),
                        "kind": caller_sym["kind"],
                    })

        # Parent
        parent = ws.belongs_to.get(name)
        if parent:
            parent_sym = _find_symbol_by_name(ws, parent)
            if parent_sym:
                results.append({
                    "name": parent, "rel_type": "belongs_to",
                    "file_path": parent_sym["file_path"],
                    "signature": parent_sym.get("signature", ""),
                    "kind": parent_sym["kind"],
                })

        # Children
        for child in ws.has_members.get(name, set()):
            child_sym = _find_symbol_by_name(ws, child)
            if child_sym:
                results.append({
                    "name": child, "rel_type": "has_member",
                    "file_path": child_sym["file_path"],
                    "signature": child_sym.get("signature", ""),
                    "kind": child_sym["kind"],
                })

    return results


async def trace_call_chain(
    workspace_id: str,
    symbol_name: str,
    direction: str = "both",
    max_depth: int = 5,
) -> dict[str, Any]:
    """
    Deep BFS on the in-memory call graph. Instant — no DB queries.
    """
    ws = _get_ws(workspace_id)
    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    # Find root symbol(s)
    roots = [s for s in ws.symbols.values() if s["name"] == symbol_name]
    if not roots:
        return {"nodes": [], "edges": [], "root": symbol_name, "depth_reached": 0}

    for root in roots:
        nodes[root["uid"]] = {
            "uid": root["uid"], "name": root["name"], "kind": root["kind"],
            "file_path": root["file_path"], "signature": root.get("signature", ""),
            "start_line": root["start_line"], "end_line": root["end_line"],
            "depth": 0, "direction": "root",
        }

    # Downstream BFS
    if direction in ("downstream", "both"):
        frontier = {symbol_name}
        visited = {symbol_name}
        for depth in range(1, max_depth + 1):
            if not frontier:
                break
            next_frontier: set[str] = set()
            for name in frontier:
                for callee in ws.calls.get(name, set()):
                    edges.append({"from": name, "to": callee, "type": "CALLS"})
                    if callee not in visited:
                        visited.add(callee)
                        next_frontier.add(callee)
                        sym = _find_symbol_by_name(ws, callee)
                        if sym:
                            nodes[sym["uid"]] = {
                                "uid": sym["uid"], "name": sym["name"],
                                "kind": sym["kind"], "file_path": sym["file_path"],
                                "signature": sym.get("signature", ""),
                                "start_line": sym["start_line"],
                                "end_line": sym["end_line"],
                                "depth": depth, "direction": "downstream",
                            }
            frontier = next_frontier

    # Upstream BFS
    if direction in ("upstream", "both"):
        frontier = {symbol_name}
        visited_up = {symbol_name}
        for depth in range(1, max_depth + 1):
            if not frontier:
                break
            next_frontier: set[str] = set()
            for name in frontier:
                for caller in ws.called_by.get(name, set()):
                    edges.append({"from": caller, "to": name, "type": "CALLS"})
                    if caller not in visited_up:
                        visited_up.add(caller)
                        next_frontier.add(caller)
                        sym = _find_symbol_by_name(ws, caller)
                        if sym:
                            nodes[sym["uid"]] = {
                                "uid": sym["uid"], "name": sym["name"],
                                "kind": sym["kind"], "file_path": sym["file_path"],
                                "signature": sym.get("signature", ""),
                                "start_line": sym["start_line"],
                                "end_line": sym["end_line"],
                                "depth": depth, "direction": "upstream",
                            }
            frontier = next_frontier

    # Add BELONGS_TO context
    all_names = {n["name"] for n in nodes.values()}
    for name in list(all_names):
        parent = ws.belongs_to.get(name)
        if parent:
            edges.append({"from": name, "to": parent, "type": "BELONGS_TO"})
        for child in ws.has_members.get(name, set()):
            edges.append({"from": child, "to": name, "type": "BELONGS_TO"})

    # Deduplicate edges
    unique_edges = list({(e["from"], e["to"], e["type"]): e for e in edges}.values())

    return {
        "root": symbol_name,
        "nodes": list(nodes.values()),
        "edges": unique_edges,
        "depth_reached": max((n.get("depth", 0) for n in nodes.values()), default=0),
    }


async def impact_analysis(
    workspace_id: str,
    symbol_name: str,
    max_depth: int = 4,
) -> dict[str, Any]:
    """
    Blast radius: who depends on this symbol? Pure in-memory BFS.
    """
    ws = _get_ws(workspace_id)
    affected: dict[str, dict] = {}

    # 1. Upstream callers (transitive)
    frontier = {symbol_name}
    visited = {symbol_name}
    for depth in range(1, max_depth + 1):
        if not frontier:
            break
        next_frontier: set[str] = set()
        for name in frontier:
            for caller in ws.called_by.get(name, set()):
                if caller not in visited:
                    visited.add(caller)
                    next_frontier.add(caller)
                    sym = _find_symbol_by_name(ws, caller)
                    if sym and sym["uid"] not in affected:
                        affected[sym["uid"]] = {
                            **sym, "distance": depth, "impact_type": "caller",
                        }
        frontier = next_frontier

    # 2. Sibling members (same parent struct/class)
    parent = ws.belongs_to.get(symbol_name)
    if parent:
        for sibling in ws.has_members.get(parent, set()):
            if sibling != symbol_name:
                sym = _find_symbol_by_name(ws, sibling)
                if sym and sym["uid"] not in affected:
                    affected[sym["uid"]] = {
                        **sym, "distance": 1, "impact_type": "sibling_member",
                    }

    # 3. Files that import the file containing this symbol
    target_sym = _find_symbol_by_name(ws, symbol_name)
    if target_sym:
        target_file = target_sym["file_path"]
        # Find all files that import target_file
        for fp, imports in ws.file_imports.items():
            if target_file in imports:
                # All symbols in that importing file
                for sym in ws.symbols.values():
                    if sym["file_path"] == fp and sym["uid"] not in affected:
                        affected[sym["uid"]] = {
                            **sym, "distance": 2, "impact_type": "importing_file",
                        }

    # Group by file
    by_file: dict[str, list[dict]] = {}
    for sym in affected.values():
        by_file.setdefault(sym["file_path"], []).append(sym)

    sorted_files = sorted(by_file.items(), key=lambda x: -len(x[1]))

    return {
        "symbol": symbol_name,
        "total_affected": len(affected),
        "files_affected": len(by_file),
        "by_file": [
            {
                "file_path": fp,
                "symbols": sorted(syms, key=lambda s: s.get("start_line", 0)),
            }
            for fp, syms in sorted_files
        ],
    }


async def get_workspace_stats(workspace_id: str) -> dict[str, int]:
    ws = _get_ws(workspace_id)
    return {
        "symbols": len(ws.symbols),
        "files": len(ws.files),
    }


# ── Utilities ─────────────────────────────────────────────────────

def _find_symbol_by_name(ws: _WorkspaceData, name: str) -> dict | None:
    """Find the first symbol with this name. O(1) amortized via cache."""
    if not ws._name_index:
        for uid, sym in ws.symbols.items():
            ws._name_index.setdefault(sym["name"], uid)
    uid = ws._name_index.get(name)
    if uid:
        return ws.symbols.get(uid)
    # Fallback: linear scan for names added after index was built
    for uid, sym in ws.symbols.items():
        if sym["name"] == name:
            ws._name_index[name] = uid
            return sym
    return None


def _detect_language(file_path: str) -> str:
    ext_map = {
        ".rs": "rust", ".py": "python", ".js": "javascript",
        ".ts": "typescript", ".tsx": "typescript", ".go": "go",
        ".java": "java", ".cpp": "cpp", ".c": "c", ".h": "c",
    }
    return ext_map.get(Path(file_path).suffix, "unknown")


def _import_to_path_hint(imp: str) -> str:
    hint = imp.replace("::", "/").replace(".", "/")
    for prefix in ("crate/", "self/", "super/"):
        if hint.startswith(prefix):
            hint = hint[len(prefix):]
    return hint
