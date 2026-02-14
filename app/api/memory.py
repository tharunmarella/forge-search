"""Workspace memory management endpoints (Simplified Roo-Code Style)."""

from fastapi import APIRouter, Depends
from ..utils import auth

router = APIRouter()


@router.get("/{workspace_id}")
async def get_workspace_memory(
    workspace_id: str,
    user: dict = Depends(auth.get_current_user)
):
    """Get workspace memory (Legacy Phase 1 - now simplified)."""
    # Return empty memory structure for backward compatibility
    return {
        "workspace_id": workspace_id,
        "failed_commands": {},
        "exhausted_approaches": []
    }


@router.delete("/{workspace_id}")
async def clear_workspace_memory(
    workspace_id: str,
    user: dict = Depends(auth.get_current_user)
):
    """Clear workspace memory (Legacy Phase 1 - now simplified)."""
    return {"status": "cleared", "workspace_id": workspace_id}
