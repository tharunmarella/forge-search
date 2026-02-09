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
    neo4j_connected: bool
    version: str = "0.1.0"
