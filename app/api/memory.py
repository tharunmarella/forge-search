"""Workspace memory management endpoints (Phase 1)."""

from fastapi import APIRouter, Depends
from ..utils import auth
from ..intelligence.phase1 import workspace_memory as ws_memory

router = APIRouter()


@router.get("/{workspace_id}")
async def get_workspace_memory(
    workspace_id: str,
    user: dict = Depends(auth.get_current_user)
):
    """Get workspace memory for debugging (PHASE 1)."""
    memory = await ws_memory.load_workspace_memory(workspace_id)
    return memory


@router.delete("/{workspace_id}")
async def clear_workspace_memory(
    workspace_id: str,
    user: dict = Depends(auth.get_current_user)
):
    """Clear workspace memory (PHASE 1 - useful for testing/reset)."""
    await ws_memory.clear_workspace_memory(workspace_id)
    return {"status": "cleared", "workspace_id": workspace_id}
