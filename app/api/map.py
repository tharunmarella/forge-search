"""Project map endpoints - Hierarchical project visualization."""

import time
import logging
from fastapi import APIRouter
from ..storage import store
from ..models import (
    MapRequest,
    MapResponse,
    MapNode,
    MapEdge
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/map", response_model=MapResponse)
async def get_project_map(req: MapRequest):
    """
    Hierarchical project map.
    
    Provides a high-level, mid-level, or low-level view of the project structure
    and relationships based on the provided focus.
    """
    t0 = time.monotonic()

    result = await store.get_project_map(
        req.workspace_id,
        focus_path=req.focus_path,
        focus_symbol=req.focus_symbol,
        depth=req.depth
    )

    elapsed = (time.monotonic() - t0) * 1000
    logger.info(
        "Map workspace=%s path=%r symbol=%r -> %d nodes, %d edges in %.0fms",
        req.workspace_id, req.focus_path, req.focus_symbol,
        len(result["nodes"]), len(result["edges"]), elapsed,
    )

    return MapResponse(
        workspace_id=result["workspace_id"],
        nodes=[MapNode(**n) for n in result["nodes"]],
        edges=[MapEdge(**e) for e in result["edges"]],
        focus_path=result["focus_path"],
        focus_symbol=result["focus_symbol"]
    )
