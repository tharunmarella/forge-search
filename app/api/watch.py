"""File watching endpoints - Intelligent code monitoring."""

import asyncio
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException
from ..utils import watcher
from ..storage import store
from ..models import WatchRequest, WatchResponse
from .indexing_helpers import index_files_batch

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/", response_model=WatchResponse)
async def watch_directory(req: WatchRequest):
    """
    Start intelligent watching of a codebase directory.

    - Debounces rapid saves (AI agents edit many files at once)
    - Skips non-code files and gitignored paths
    - Detects structural changes (new/deleted/renamed symbols)
    - Cascade re-embeds callers when a function signature changes
    """
    root = Path(req.root_path)
    if not root.exists():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {req.root_path}")

    # If already watching, stop the old watcher
    watcher.stop_watching(req.workspace_id)

    # Do an initial scan and index
    stats = await watcher.scan_and_index(
        req.workspace_id, root, store, index_files_batch,
    )

    # Start background watcher
    task = asyncio.create_task(
        watcher.start_watching(
            req.workspace_id, root, store, index_files_batch,
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


@router.post("/scan", response_model=WatchResponse)
async def scan_directory(req: WatchRequest):
    """
    One-shot intelligent scan — detect and index only what changed.
    Same intelligence as /watch but doesn't start a background watcher.
    """
    root = Path(req.root_path)
    if not root.exists():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {req.root_path}")

    stats = await watcher.scan_and_index(
        req.workspace_id, root, store, index_files_batch,
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


@router.delete("/{workspace_id}")
async def stop_watch(workspace_id: str):
    """Stop watching a workspace."""
    stopped = watcher.stop_watching(workspace_id)
    return {"workspace_id": workspace_id, "stopped": stopped}
