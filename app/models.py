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
    features: dict[str, bool] = {"streaming": True}


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


# ── Chat models ──────────────────────────────────────────────────


class ToolResultPayload(BaseModel):
    call_id: str
    output: str
    success: bool


class AttachedFile(BaseModel):
    """A file attached by the IDE with its live content."""
    path: str
    content: str


class ChatRequest(BaseModel):
    workspace_id: str
    conversation_id: str | None = None  # Unique ID per conversation (for state tracking)
    question: str | None = None
    # Support for multi-turn tool results from the IDE
    tool_results: list[ToolResultPayload] | None = None
    # Live file contents from the IDE (open files, recently edited, etc.)
    attached_files: list[AttachedFile] | None = None
    # History of messages (serialized LangChain messages)
    history: list[dict] | None = None
    max_tokens: int = Field(default=1024, ge=1, le=4096)
    temperature: float = Field(default=0.1, ge=0, le=1)


class PlanStepResponse(BaseModel):
    """A single step in the agent's execution plan."""
    number: int
    description: str
    status: str  # "pending", "in_progress", "done", "failed"


class ChatResponse(BaseModel):
    answer: str | None = None
    tool_calls: list[dict] | None = None
    # Updated history to send back to the IDE
    history: list[dict] | None = None
    status: str  # "done", "requires_action", "error"
    model: str = ""
    tokens: int = 0
    total_time_ms: float = 0
    # ── Task decomposition ────────────────────────────────────
    task_complexity: str | None = None  # "simple" or "complex"
    plan_steps: list[PlanStepResponse] | None = None
    current_step: int | None = None


# ── LLM Model management ─────────────────────────────────────────


class ProviderInfo(BaseModel):
    """A supported LLM provider."""
    provider: str           # "groq", "gemini", "anthropic", etc.
    name: str               # "Groq", "Google Gemini", "Anthropic (Claude)"
    env_key: str            # "GROQ_API_KEY"
    configured: bool        # True if API key is set
    models: list[str]       # Available model strings


class ActiveModelsResponse(BaseModel):
    """Currently active model configuration."""
    reasoning_model: str
    tool_model: str
    providers: list[ProviderInfo]


class SetModelRequest(BaseModel):
    """Request to change the active model(s)."""
    reasoning_model: str | None = None
    tool_model: str | None = None


class SetModelResponse(BaseModel):
    """Response after changing models."""
    reasoning_model: str
    tool_model: str
    message: str
