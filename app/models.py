"""Pydantic request/response models for the search API."""

from pydantic import BaseModel, Field


# ── Request models ────────────────────────────────────────────────


class FilePayload(BaseModel):
    path: str
    content: str


class IndexRequest(BaseModel):
    workspace_id: str
    files: list[FilePayload]


class SearchRequest(BaseModel):
    workspace_id: str
    query: str
    top_k: int = Field(default=10, ge=1, le=50)


class ReindexRequest(BaseModel):
    workspace_id: str


# ── Response models ───────────────────────────────────────────────


class RelatedSymbol(BaseModel):
    name: str
    file_path: str
    relationship: str  # "calls", "called_by", "uses_type", "belongs_to"
    signature: str = ""


class SearchResult(BaseModel):
    file_path: str
    name: str
    symbol_type: str  # "function", "method", "struct", "class", etc.
    signature: str
    content: str
    start_line: int
    end_line: int
    score: float
    related: list[RelatedSymbol] = []


class SearchResponse(BaseModel):
    results: list[SearchResult]
    query: str
    workspace_id: str
    total_nodes: int = 0
    search_time_ms: float = 0


class IndexResponse(BaseModel):
    workspace_id: str
    files_indexed: int
    nodes_created: int
    relationships_created: int
    embeddings_generated: int
    index_time_ms: float


class HealthResponse(BaseModel):
    status: str
    store_ok: bool
    version: str = "0.2.0"


# ── Deep traversal models ────────────────────────────────────────


class TraceRequest(BaseModel):
    workspace_id: str
    symbol_name: str
    direction: str = Field(
        default="both",
        description="'upstream' (who calls me), 'downstream' (what do I call), or 'both'",
    )
    max_depth: int = Field(default=5, ge=1, le=10)


class TraceNode(BaseModel):
    name: str
    kind: str
    file_path: str
    signature: str = ""
    start_line: int = 0
    end_line: int = 0
    depth: int = 0
    direction: str = ""  # "root", "upstream", "downstream"


class TraceEdge(BaseModel):
    from_symbol: str = Field(alias="from")
    to_symbol: str = Field(alias="to")
    type: str  # "CALLS", "BELONGS_TO"

    class Config:
        populate_by_name = True


class TraceResponse(BaseModel):
    root: str
    nodes: list[TraceNode]
    edges: list[TraceEdge]
    depth_reached: int


class ImpactRequest(BaseModel):
    workspace_id: str
    symbol_name: str
    max_depth: int = Field(default=4, ge=1, le=8)


class AffectedSymbol(BaseModel):
    name: str
    kind: str
    signature: str = ""
    start_line: int = 0
    end_line: int = 0
    distance: int = 0
    impact_type: str = ""  # "caller", "sibling_member", "importing_file"


class AffectedFile(BaseModel):
    file_path: str
    symbols: list[AffectedSymbol]


class ImpactResponse(BaseModel):
    symbol: str
    total_affected: int
    files_affected: int
    by_file: list[AffectedFile]


# ── Watch models ─────────────────────────────────────────────────


class WatchRequest(BaseModel):
    workspace_id: str
    root_path: str  # Absolute path to codebase directory


class WatchResponse(BaseModel):
    workspace_id: str
    status: str  # "watching", "stopped", "scan_complete"
    files_scanned: int = 0
    files_changed: int = 0
    symbols_added: int = 0
    symbols_removed: int = 0
    symbols_modified: int = 0
    cascade_reembeds: int = 0
    time_ms: float = 0
