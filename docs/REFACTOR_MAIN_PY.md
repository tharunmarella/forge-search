# Refactor main.py - Split into Clean API Modules

## Current State: ❌ BLOATED

```
main.py: 1,844 lines, 22 endpoints
- FastAPI app initialization
- MongoDB setup
- 5 auth endpoints
- 3 search/index endpoints
- 2 code analysis endpoints
- 3 file watching endpoints
- 2 chat endpoints
- 2 memory endpoints
- 2 model config endpoints
- 1 debug endpoint
- 1 health endpoint
- Helper functions
- Conversation store
- Middleware
```

**Problems**:
- ❌ Violates Single Responsibility Principle
- ❌ Hard to navigate (1,844 lines)
- ❌ Tight coupling (everything depends on everything)
- ❌ Cannot test endpoints in isolation
- ❌ Cannot reuse endpoints in microservices
- ❌ Merge conflicts when multiple people work on it

---

## Proposed Structure: ✅ CLEAN

```
app/
├── main.py                    # App init, middleware, startup (150 lines)
│
├── api/                       # API routes (organized by domain)
│   ├── __init__.py
│   ├── auth.py                # Authentication (5 endpoints, ~200 lines)
│   ├── search.py              # Search & indexing (3 endpoints, ~300 lines)
│   ├── analysis.py            # Trace & impact (2 endpoints, ~200 lines)
│   ├── chat.py                # Chat endpoints (2 endpoints, ~400 lines)
│   ├── watch.py               # File watching (3 endpoints, ~250 lines)
│   ├── models.py              # Model config (2 endpoints, ~150 lines)
│   ├── memory.py              # Workspace memory (2 endpoints, ~100 lines)
│   ├── debug.py               # Debug endpoints (1 endpoint, ~100 lines)
│   └── health.py              # Health check (1 endpoint, ~50 lines)
│
├── core/                      # Business logic (already organized)
├── storage/                   # Data layer
├── intelligence/              # Intelligence system
└── utils/                     # Utilities
```

**Benefits**:
- ✅ Each file has single responsibility
- ✅ Easy to find endpoints (auth → auth.py, chat → chat.py)
- ✅ Can test each API module independently
- ✅ Can reuse in microservices
- ✅ Parallel development (no merge conflicts)
- ✅ Clear dependencies

---

## File Breakdown

### 1. `app/main.py` (NEW: ~150 lines)
**Responsibility**: App initialization only

```python
"""
Forge Search API - Main application entry point.
"""

from fastapi import FastAPI
from .api import auth, search, analysis, chat, watch, models, memory, health, debug

app = FastAPI(title="Forge Search")

# Include routers
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(search.router, tags=["search"])
app.include_router(analysis.router, tags=["analysis"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(watch.router, prefix="/watch", tags=["watch"])
app.include_router(models.router, prefix="/models", tags=["models"])
app.include_router(memory.router, prefix="/memory", tags=["memory"])
app.include_router(debug.router, prefix="/debug", tags=["debug"])
app.include_router(health.router, tags=["health"])

# Startup/shutdown
@app.on_event("startup")
async def startup():
    await store.init_pool()
    await _ensure_mongo_indexes()

@app.on_event("shutdown")
async def shutdown():
    await store.close_pool()
```

---

### 2. `app/api/auth.py` (~200 lines)
**Responsibility**: Authentication only

```python
"""Authentication endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from ..utils import auth

router = APIRouter()

@router.get("/github")
async def github_auth():
    """Start GitHub OAuth flow."""
    ...

@router.get("/github/callback")
async def github_callback(code: str):
    """GitHub OAuth callback."""
    ...

@router.get("/google")
async def google_auth():
    """Start Google OAuth flow."""
    ...

@router.get("/google/callback")
async def google_callback(code: str):
    """Google OAuth callback."""
    ...

@router.get("/me")
async def get_current_user(user: dict = Depends(auth.get_current_user)):
    """Get current user info."""
    ...
```

---

### 3. `app/api/search.py` (~300 lines)
**Responsibility**: Search & indexing

```python
"""Search and indexing endpoints."""

from fastapi import APIRouter, Depends
from ..storage import store
from ..core import embeddings
from ..utils import auth

router = APIRouter()

@router.post("/index")
async def index_code(req: IndexRequest, user: dict = Depends(auth.get_current_user)):
    """Parse and index code."""
    ...

@router.post("/search")
async def search_code(req: SearchRequest, user: dict = Depends(auth.get_current_user)):
    """Semantic code search."""
    ...

@router.post("/reindex")
async def reindex_workspace(req: ReindexRequest, user: dict = Depends(auth.get_current_user)):
    """Reindex entire workspace."""
    ...
```

---

### 4. `app/api/analysis.py` (~200 lines)
**Responsibility**: Code analysis (trace, impact)

```python
"""Code analysis endpoints."""

from fastapi import APIRouter, Depends
from ..storage import store
from ..utils import auth, mermaid

router = APIRouter()

@router.post("/trace")
async def trace_call_chain(req: TraceRequest, user: dict = Depends(auth.get_current_user)):
    """Trace call chains."""
    ...

@router.post("/impact")
async def impact_analysis(req: ImpactRequest, user: dict = Depends(auth.get_current_user)):
    """Analyze blast radius."""
    ...
```

---

### 5. `app/api/chat.py` (~400 lines)
**Responsibility**: Chat & conversation management

```python
"""Chat endpoints."""

from fastapi import APIRouter, Depends
from ..core import agent
from ..utils import auth

router = APIRouter()

# Conversation store (module-scoped)
_conversations = {}

@router.post("/")
async def chat_endpoint(req: ChatRequest, user: dict = Depends(auth.get_current_user)):
    """Main chat endpoint."""
    ...

@router.post("/stream")
async def chat_stream(req: ChatRequest, user: dict = Depends(auth.get_current_user)):
    """Streaming chat endpoint."""
    ...
```

---

### 6. `app/api/watch.py` (~250 lines)
**Responsibility**: File watching

```python
"""File watching endpoints."""

from fastapi import APIRouter, Depends
from ..utils import watcher, auth

router = APIRouter()

@router.post("/")
async def start_watch(req: WatchRequest, user: dict = Depends(auth.get_current_user)):
    """Start file watcher."""
    ...

@router.post("/scan")
async def scan_once(req: WatchRequest, user: dict = Depends(auth.get_current_user)):
    """One-shot scan."""
    ...

@router.delete("/{workspace_id}")
async def stop_watch(workspace_id: str, user: dict = Depends(auth.get_current_user)):
    """Stop watcher."""
    ...
```

---

### 7. `app/api/models.py` (~150 lines)
**Responsibility**: Model configuration

```python
"""Model configuration endpoints."""

from fastapi import APIRouter, Depends
from ..core import llm
from ..utils import auth

router = APIRouter()

@router.get("/")
async def get_models(user: dict = Depends(auth.get_current_user)):
    """List available models."""
    ...

@router.post("/set")
async def set_model(req: SetModelRequest, user: dict = Depends(auth.get_current_user)):
    """Switch model at runtime."""
    ...
```

---

### 8. `app/api/memory.py` (~100 lines)
**Responsibility**: Workspace memory management

```python
"""Workspace memory endpoints (Phase 1)."""

from fastapi import APIRouter, Depends
from ..intelligence.phase1 import workspace_memory as ws_memory
from ..utils import auth

router = APIRouter()

@router.get("/{workspace_id}")
async def get_memory(workspace_id: str, user: dict = Depends(auth.get_current_user)):
    """Get workspace memory."""
    ...

@router.delete("/{workspace_id}")
async def clear_memory(workspace_id: str, user: dict = Depends(auth.get_current_user)):
    """Clear workspace memory."""
    ...
```

---

### 9. `app/api/health.py` (~50 lines)
**Responsibility**: Health & status

```python
"""Health and status endpoints."""

from fastapi import APIRouter
from ..storage import store

router = APIRouter()

@router.get("/health")
async def health():
    """Health check."""
    ...
```

---

### 10. `app/api/debug.py` (~100 lines)
**Responsibility**: Debug & development tools

```python
"""Debug endpoints."""

from fastapi import APIRouter, Depends
from ..core import agent
from ..utils import auth

router = APIRouter()

@router.post("/pre-enrichment")
async def debug_pre_enrichment(req: dict, user: dict = Depends(auth.get_current_user)):
    """Debug pre-enrichment."""
    ...
```

---

## Migration Plan

### Step 1: Create API module structure
```bash
mkdir app/api
touch app/api/__init__.py
touch app/api/{auth,search,analysis,chat,watch,models,memory,health,debug}.py
```

### Step 2: Extract each endpoint group
```bash
# Copy endpoints from main.py to respective files
# Update imports
# Test each module independently
```

### Step 3: Update main.py
```bash
# Remove all endpoints
# Keep only app initialization
# Include routers
```

### Step 4: Update tests
```bash
# Update test imports
# Test each API module
```

### Step 5: Verify
```bash
# Run tests
pytest tests/

# Start server
docker-compose up

# Test all endpoints
./test_all_endpoints.sh
```

---

## Expected Results

### File Sizes After Refactor

| File | Lines | Responsibility |
|------|-------|----------------|
| **main.py** | 150 | App init only |
| api/auth.py | 200 | Auth |
| api/search.py | 300 | Search/index |
| api/analysis.py | 200 | Trace/impact |
| api/chat.py | 400 | Chat |
| api/watch.py | 250 | File watching |
| api/models.py | 150 | Model config |
| api/memory.py | 100 | Memory |
| api/health.py | 50 | Health |
| api/debug.py | 100 | Debug |
| **Total** | **1,900** | Same functionality, 10 files |

---

## Benefits

### Before:
```python
# Want to add auth endpoint? 
# Open main.py (1,844 lines)
# Scroll to auth section (where is it?)
# Add code
# Hope you don't break something else
```

### After:
```python
# Want to add auth endpoint?
# Open app/api/auth.py (200 lines)
# All auth code in one place
# Add endpoint
# Test auth module in isolation
# Done!
```

---

## Testing Improvement

### Before:
```python
# Test chat endpoint
# Must mock: auth, search, store, embeddings, agent, watcher, etc.
# Tests are fragile
```

### After:
```python
# Test chat endpoint
from app.api.chat import router

# Only mock: agent and auth
# Clean, focused tests
```

---

## Comparison to Well-Organized Projects

### FastAPI Best Practice:
```
app/
├── main.py              # App initialization
├── api/                 # All endpoints
│   ├── v1/              # API versioning
│   │   ├── auth.py
│   │   ├── users.py
│   │   └── items.py
├── core/                # Business logic
├── models/              # Pydantic models
└── utils/               # Helpers
```

### Our structure matches this! ✅

---

## Action Required

**Option 1: Do it now** (2-3 hours)
- Split main.py into API modules
- Update all imports
- Test everything

**Option 2: Do it gradually** (over next week)
- Extract one endpoint group per day
- No downtime
- Lower risk

**Option 3: Keep as technical debt**
- Fast to deploy current system
- Refactor later when needed
- Risk: Gets worse over time

---

## Recommendation

**Do Option 1 (split now) because:**

1. ✅ **You're already reorganizing** - might as well finish the job
2. ✅ **No production traffic yet** - safe time to refactor
3. ✅ **Future maintenance** - much easier to work with
4. ✅ **Team scaling** - multiple people can work on different APIs
5. ✅ **Professional structure** - matches industry standards

**Takes 2-3 hours but saves weeks of future pain.**

---

## Immediate Next Step

Want me to:
1. **Split main.py now** (recommended)
2. **Keep as TODO** (defer to later)
3. **Create the structure but not move code** (template only)

Choose 1, 2, or 3?
