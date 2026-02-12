"""
Intelligent code watcher — re-indexes only what matters.

Thinks like a developer:
  1. Debounces rapid saves (AI agents edit 20 files at once)
  2. Filters non-code files (README, images, configs)
  3. Skips gitignored paths (node_modules, target/, .git)
  4. Detects structural changes (new/deleted/renamed symbols)
  5. Cascade re-embeds affected symbols (callers whose context changed)

Usage:
  POST /watch {"workspace_id": "my-project", "root_path": "/path/to/code"}
  → starts watching, returns immediately
  → re-indexes intelligently on file changes
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from pathlib import Path
from typing import Any

from .parser import parse_file, is_code_file, should_skip, FileParseResult

logger = logging.getLogger(__name__)

# ── Active watchers ───────────────────────────────────────────────

_watchers: dict[str, asyncio.Task] = {}  # workspace_id -> watcher task

DEBOUNCE_SECONDS = 2.0   # Wait for burst of saves to settle
CASCADE_RE_EMBED = True   # Re-embed callers when a symbol changes


# ── Core: detect what changed ─────────────────────────────────────

def detect_changes(
    old_parse: FileParseResult | None,
    new_parse: FileParseResult,
) -> dict[str, Any]:
    """
    Compare old vs new parse of a file. Returns what changed.

    Returns:
      {
        "changed": bool,
        "added_symbols": [names],
        "removed_symbols": [names],
        "modified_symbols": [names],  # signature or content changed
        "needs_reembed": bool,
        "needs_cascade": bool,  # callers should be re-embedded too
      }
    """
    if old_parse is None:
        # New file — everything is new
        return {
            "changed": True,
            "added_symbols": [d.name for d in new_parse.definitions],
            "removed_symbols": [],
            "modified_symbols": [],
            "needs_reembed": True,
            "needs_cascade": False,
        }

    if old_parse.content_hash == new_parse.content_hash:
        return {"changed": False, "added_symbols": [], "removed_symbols": [],
                "modified_symbols": [], "needs_reembed": False, "needs_cascade": False}

    # Build lookup by (name, kind, start_line)
    old_syms = {(d.name, d.kind): d for d in old_parse.definitions}
    new_syms = {(d.name, d.kind): d for d in new_parse.definitions}

    old_keys = set(old_syms.keys())
    new_keys = set(new_syms.keys())

    added = new_keys - old_keys
    removed = old_keys - new_keys
    common = old_keys & new_keys

    # Check which common symbols actually changed
    modified = []
    signature_changed = False
    for key in common:
        old_d, new_d = old_syms[key], new_syms[key]
        if old_d.signature != new_d.signature:
            modified.append(key[0])
            signature_changed = True  # Signature change affects callers
        elif old_d.content != new_d.content:
            modified.append(key[0])

    # Needs cascade if: symbols added/removed/renamed, or signature changed
    needs_cascade = bool(added or removed or signature_changed)

    # Needs re-embed if anything structural changed
    needs_reembed = bool(added or removed or modified)

    return {
        "changed": needs_reembed or needs_cascade,
        "added_symbols": [k[0] for k in added],
        "removed_symbols": [k[0] for k in removed],
        "modified_symbols": modified,
        "needs_reembed": needs_reembed,
        "needs_cascade": needs_cascade,
    }


# ── Scan directory for changes ────────────────────────────────────

async def scan_and_index(
    workspace_id: str,
    root_path: Path,
    store_module,
    index_fn,
) -> dict[str, Any]:
    """
    Scan a directory, detect changes, and intelligently re-index.

    Returns stats about what was done.
    """
    t0 = time.monotonic()
    stats = {
        "files_scanned": 0,
        "files_changed": 0,
        "files_skipped": 0,
        "symbols_added": 0,
        "symbols_removed": 0,
        "symbols_modified": 0,
        "cascade_reembeds": 0,
        "time_ms": 0,
    }

    changed_files = []

    # 1. Scan all code files
    for path in root_path.rglob("*"):
        if not path.is_file():
            continue
        if not is_code_file(path):
            continue

        stats["files_scanned"] += 1
        rel_path = str(path.relative_to(root_path))

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        # Quick hash check — skip unchanged files
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        existing_hash = await store_module.get_file_hash(workspace_id, rel_path)

        if existing_hash == content_hash:
            stats["files_skipped"] += 1
            continue

        # Parse and detect what changed
        new_parse = parse_file(rel_path, content)

        # Get old parse for comparison (rebuild from stored symbols)
        old_parse = await _get_old_parse(store_module, workspace_id, rel_path)

        changes = detect_changes(old_parse, new_parse)

        if not changes["changed"]:
            stats["files_skipped"] += 1
            continue

        stats["files_changed"] += 1
        stats["symbols_added"] += len(changes["added_symbols"])
        stats["symbols_removed"] += len(changes["removed_symbols"])
        stats["symbols_modified"] += len(changes["modified_symbols"])

        changed_files.append({
            "path": rel_path,
            "content": content,
            "changes": changes,
        })

        logger.info(
            "Changed: %s (+%d -%d ~%d symbols%s)",
            rel_path,
            len(changes["added_symbols"]),
            len(changes["removed_symbols"]),
            len(changes["modified_symbols"]),
            " CASCADE" if changes["needs_cascade"] else "",
        )

    # 2. Index changed files via the normal /index pipeline
    if changed_files:
        from .models import IndexRequest, FilePayload
        files_payload = [{"path": f["path"], "content": f["content"]} for f in changed_files]
        await index_fn(workspace_id, files_payload)

    # 3. Cascade re-embed: if a symbol's signature changed,
    #    re-embed all its callers (their enriched context references this symbol)
    if CASCADE_RE_EMBED:
        cascade_symbols = set()
        for f in changed_files:
            changes = f["changes"]
            if changes["needs_cascade"]:
                # All added/removed/modified symbols need cascade
                for name in changes["added_symbols"] + changes["removed_symbols"] + changes["modified_symbols"]:
                    # Find callers of this symbol
                    ws = await store_module._get_ws(workspace_id)
                    callers = ws.called_by.get(name, set())
                    cascade_symbols.update(callers)

        if cascade_symbols:
            stats["cascade_reembeds"] = len(cascade_symbols)
            logger.info("Cascade re-embedding %d caller symbols", len(cascade_symbols))
            # Find files containing these symbols and re-index them
            ws = await store_module._get_ws(workspace_id)
            cascade_files = set()
            for sym_name in cascade_symbols:
                for uid, sym in ws.symbols.items():
                    if sym["name"] == sym_name:
                        cascade_files.add(sym["file_path"])
                        break

            for fp in cascade_files:
                full_path = root_path / fp
                if full_path.exists():
                    content = full_path.read_text(encoding="utf-8", errors="replace")
                    await index_fn(workspace_id, [{"path": fp, "content": content}])

    elapsed = (time.monotonic() - t0) * 1000
    stats["time_ms"] = round(elapsed, 1)

    logger.info(
        "Watch scan: %d files, %d changed, %d skipped, %d cascaded in %.0fms",
        stats["files_scanned"], stats["files_changed"],
        stats["files_skipped"], stats["cascade_reembeds"], elapsed,
    )

    return stats


# ── Background watcher ────────────────────────────────────────────

async def start_watching(
    workspace_id: str,
    root_path: Path,
    store_module,
    index_fn,
    poll_interval: float = 5.0,
) -> None:
    """
    Background task that watches a directory and re-indexes on changes.
    Uses polling with debounce (no OS-level file watching needed).
    """
    logger.info("Watching %s for workspace %s (poll every %.0fs)", root_path, workspace_id, poll_interval)

    # Initial full scan
    await scan_and_index(workspace_id, root_path, store_module, index_fn)

    # Track file mtimes for change detection
    last_mtimes: dict[str, float] = {}
    for path in root_path.rglob("*"):
        if path.is_file() and is_code_file(path):
            rel = str(path.relative_to(root_path))
            last_mtimes[rel] = path.stat().st_mtime

    while True:
        await asyncio.sleep(poll_interval)

        # Check for mtime changes
        changed_paths = []
        current_mtimes: dict[str, float] = {}

        for path in root_path.rglob("*"):
            if not path.is_file() or not is_code_file(path):
                continue
            rel = str(path.relative_to(root_path))
            mtime = path.stat().st_mtime
            current_mtimes[rel] = mtime

            if rel not in last_mtimes or last_mtimes[rel] != mtime:
                changed_paths.append(rel)

        # Check for deleted files
        deleted = set(last_mtimes.keys()) - set(current_mtimes.keys())

        if not changed_paths and not deleted:
            continue

        # Debounce: wait for more changes to settle
        await asyncio.sleep(DEBOUNCE_SECONDS)

        # Re-check after debounce (more files may have changed)
        for path in root_path.rglob("*"):
            if not path.is_file() or not is_code_file(path):
                continue
            rel = str(path.relative_to(root_path))
            mtime = path.stat().st_mtime
            current_mtimes[rel] = mtime
            if rel not in last_mtimes or last_mtimes[rel] != mtime:
                if rel not in changed_paths:
                    changed_paths.append(rel)

        if changed_paths or deleted:
            logger.info("Detected %d changed, %d deleted files", len(changed_paths), len(deleted))
            await scan_and_index(workspace_id, root_path, store_module, index_fn)

        last_mtimes = current_mtimes


def stop_watching(workspace_id: str) -> bool:
    """Stop watching a workspace. Returns True if was watching."""
    task = _watchers.pop(workspace_id, None)
    if task:
        task.cancel()
        logger.info("Stopped watching workspace %s", workspace_id)
        return True
    return False


def is_watching(workspace_id: str) -> bool:
    task = _watchers.get(workspace_id)
    return task is not None and not task.done()


# ── Helper ────────────────────────────────────────────────────────

async def _get_old_parse(store_module, workspace_id: str, file_path: str) -> FileParseResult | None:
    """Reconstruct old parse from stored symbols (for change detection)."""
    ws = await store_module._get_ws(workspace_id)
    old_defs = []
    for uid, sym in ws.symbols.items():
        if sym["file_path"] == file_path:
            from .parser import SymbolDef
            old_defs.append(SymbolDef(
                name=sym["name"],
                kind=sym["kind"],
                file_path=sym["file_path"],
                start_line=sym["start_line"],
                end_line=sym["end_line"],
                signature=sym["signature"],
                content=sym["content"],
                parent=sym.get("parent"),
            ))

    if not old_defs:
        return None

    file_uid = f"{workspace_id}:{file_path}"
    file_data = ws.files.get(file_uid)
    content_hash = file_data["content_hash"] if file_data else ""

    return FileParseResult(
        file_path=file_path,
        content_hash=content_hash,
        definitions=old_defs,
    )
