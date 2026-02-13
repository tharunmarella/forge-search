"""Code analysis endpoints - Trace and impact analysis."""

import time
import logging
from fastapi import APIRouter
from ..storage import store
from ..models import (
    TraceRequest,
    TraceResponse,
    TraceNode,
    TraceEdge,
    ImpactRequest,
    ImpactResponse,
    AffectedFile,
    AffectedSymbol,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/trace", response_model=TraceResponse)
async def trace_symbol(req: TraceRequest):
    """
    Deep call-chain traversal.

    Walks the CALLS graph upstream and/or downstream from a symbol
    to build the full execution flow — like a developer tracing through
    the code to understand "how does this work end to end?"

    Returns a subgraph of nodes and edges.
    """
    t0 = time.monotonic()

    result = await store.trace_call_chain(
        req.workspace_id,
        req.symbol_name,
        direction=req.direction,
        max_depth=req.max_depth,
    )

    elapsed = (time.monotonic() - t0) * 1000
    logger.info(
        "Trace workspace=%s symbol=%r direction=%s -> %d nodes, %d edges in %.0fms",
        req.workspace_id, req.symbol_name, req.direction,
        len(result["nodes"]), len(result["edges"]), elapsed,
    )

    return TraceResponse(
        root=result["root"],
        nodes=[TraceNode(**n) for n in result["nodes"]],
        edges=[TraceEdge(**e) for e in result["edges"]],
        depth_reached=result["depth_reached"],
    )


@router.post("/impact", response_model=ImpactResponse)
async def impact_analysis(req: ImpactRequest):
    """
    Blast radius analysis.

    "If I change this symbol, what else is affected?"

    Walks upstream through CALLS, BELONGS_TO, and file IMPORTS
    to find every symbol and file that depends on the target.
    Groups results by file for a developer-friendly view.
    """
    t0 = time.monotonic()

    result = await store.impact_analysis(
        req.workspace_id,
        req.symbol_name,
        max_depth=req.max_depth,
    )

    elapsed = (time.monotonic() - t0) * 1000
    logger.info(
        "Impact workspace=%s symbol=%r -> %d affected across %d files in %.0fms",
        req.workspace_id, req.symbol_name,
        result["total_affected"], result["files_affected"], elapsed,
    )

    return ImpactResponse(
        symbol=result["symbol"],
        total_affected=result["total_affected"],
        files_affected=result["files_affected"],
        by_file=[
            AffectedFile(
                file_path=f["file_path"],
                symbols=[AffectedSymbol(**s) for s in f["symbols"]],
            )
            for f in result["by_file"]
        ],
    )
