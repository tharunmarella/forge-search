# Project Structure

Clean, organized codebase with clear separation of concerns.

## Directory Structure

```
forge-search/
├── app/
│   ├── main.py                      # FastAPI application entry point
│   ├── models.py                    # Pydantic models for API
│   ├── sse.py                       # Server-Sent Events
│   ├── stream.py                    # Streaming utilities
│   │
│   ├── core/                        # Core business logic
│   │   ├── __init__.py
│   │   ├── agent.py                 # LangGraph agent orchestrator (2,466 lines)
│   │   ├── llm.py                   # LLM provider (LiteLLM integration)
│   │   ├── embeddings.py            # Text embeddings
│   │   ├── parser.py                # Code parsing (Tree-sitter)
│   │   └── chat.py                  # Chat utilities
│   │
│   ├── storage/                     # Data persistence
│   │   ├── __init__.py
│   │   └── store.py                 # PostgreSQL + pgvector
│   │
│   ├── intelligence/                # Intelligence system (NEW)
│   │   ├── __init__.py              # Unified exports
│   │   │
│   │   ├── phase1/                  # Persistent Memory
│   │   │   ├── __init__.py
│   │   │   └── workspace_memory.py  # Cross-trace failure tracking
│   │   │
│   │   ├── phase2/                  # LLM-Powered Intelligence
│   │   │   ├── __init__.py
│   │   │   ├── error_analyzer.py    # LLM-based error understanding
│   │   │   └── adaptive_config.py   # Learned thresholds
│   │   │
│   │   └── phase3/                  # Long-term Intelligence
│   │       ├── __init__.py
│   │       ├── hierarchical_planner.py    # Tree-based planning
│   │       ├── learning_checkpoints.py    # Knowledge consolidation
│   │       ├── model_router.py            # Intelligent model selection
│   │       └── parallel_executor.py       # Concurrent execution
│   │
│   └── utils/                       # Utilities
│       ├── __init__.py
│       ├── auth.py                  # Authentication
│       ├── watcher.py               # File watching
│       └── mermaid.py               # Diagram generation
│
├── tests/                           # Test suite
│   ├── test_phase1.py               # Phase 1 tests
│   ├── test_phase3.py               # Phase 3 tests
│   ├── ide_simulator/               # IDE tool simulator
│   └── examples/                    # Example tests
│
├── docs/                            # Documentation (NEW - organize docs)
│   ├── PHASE1_README.md
│   ├── PHASE2_BETTER_DESIGN.md
│   ├── PHASE3_README.md
│   ├── ALL_PHASES_SUMMARY.md
│   └── INTEGRATION_COMPLETE.md
│
├── docker-compose.yml               # Docker services
├── requirements.txt                 # Python dependencies
└── README.md                        # Project README
```

---

## Module Organization

### Core (`app/core/`)
**Purpose**: Essential agent functionality

- **agent.py**: Main LangGraph orchestrator with tool execution
- **llm.py**: LLM provider abstraction (LiteLLM)
- **embeddings.py**: Text embedding generation
- **parser.py**: Code parsing with Tree-sitter
- **chat.py**: Chat utilities

**Import pattern**: `from app.core import agent, llm, embeddings`

---

### Storage (`app/storage/`)
**Purpose**: Data persistence layer

- **store.py**: PostgreSQL + pgvector for code indexing

**Import pattern**: `from app.storage import store`

---

### Intelligence (`app/intelligence/`)
**Purpose**: All 3 phases of intelligent agent behavior

#### Phase 1: Persistent Memory
- **workspace_memory.py**: Track failures across conversations
  - Prevents endless loops
  - Pre-emptive blocking
  - Ask for help after 5+ failures

#### Phase 2: LLM-Powered Intelligence
- **error_analyzer.py**: LLM analyzes errors (not regex)
- **adaptive_config.py**: Learned thresholds per workspace

#### Phase 3: Long-term Intelligence
- **hierarchical_planner.py**: Tree-based plans with alternatives
- **learning_checkpoints.py**: Consolidate knowledge
- **model_router.py**: Select optimal model per task
- **parallel_executor.py**: Concurrent tool execution

**Import patterns**:
```python
# Individual modules
from app.intelligence.phase1 import workspace_memory
from app.intelligence.phase2 import error_analyzer
from app.intelligence.phase3 import model_router

# Or unified
from app.intelligence import (
    load_workspace_memory,     # Phase 1
    analyze_error_with_llm,    # Phase 2
    get_optimal_model_for_turn # Phase 3
)
```

---

### Utils (`app/utils/`)
**Purpose**: Helper utilities

- **auth.py**: Authentication & authorization
- **watcher.py**: File system monitoring
- **mermaid.py**: Diagram generation

**Import pattern**: `from app.utils import auth, watcher`

---

## Configuration

### Phase Control
```python
# In app/core/agent.py
ENABLE_PHASE_1 = os.getenv("ENABLE_PHASE_1", "true").lower() == "true"
ENABLE_PHASE_2 = os.getenv("ENABLE_PHASE_2", "true").lower() == "true"
ENABLE_PHASE_3 = os.getenv("ENABLE_PHASE_3", "true").lower() == "true"
```

### Model Routing
```python
# In app/core/agent.py
MODEL_ROUTING_BUDGET = os.getenv("MODEL_ROUTING_BUDGET", "balanced")
# Options: fast, balanced, quality
```

---

## Import Examples

### From main.py (FastAPI app)
```python
# Core modules
from .core import agent, llm, embeddings, chat

# Storage
from .storage import store

# Utils
from .utils import auth, watcher, mermaid

# Intelligence (Phase 1)
from .intelligence.phase1 import workspace_memory
```

### From agent.py (core module)
```python
# Sibling modules in core
from . import embeddings, chat, llm

# Storage (parent level)
from ..storage import store

# Intelligence system
from ..intelligence.phase1 import workspace_memory
from ..intelligence.phase2 import error_analyzer, adaptive_config
from ..intelligence.phase3 import (
    hierarchical_planner,
    learning_checkpoints,
    model_router,
    parallel_executor,
)
```

### From tests
```python
# Test Phase 1
from app.intelligence.phase1 import workspace_memory

# Test Phase 3
from app.intelligence.phase3.hierarchical_planner import HierarchicalPlan
from app.core import llm
```

---

## Benefits of New Structure

### Before (Flat)
```
❌ app/
   ├── agent.py
   ├── llm.py
   ├── workspace_memory.py
   ├── intelligent_error_analyzer.py
   ├── adaptive_config.py
   ├── hierarchical_planner.py
   ├── learning_checkpoints.py
   ├── intelligent_model_router.py
   ├── parallel_executor.py
   ├── store.py
   ├── auth.py
   ├── ... (20 files in one directory)
```

**Problems**:
- Hard to navigate
- No clear module boundaries
- Difficult to understand relationships
- "Junk drawer" pattern

### After (Organized)
```
✅ app/
   ├── core/              # Core agent logic
   ├── storage/           # Data layer
   ├── intelligence/      # Intelligence system (3 phases)
   │   ├── phase1/        # Persistent memory
   │   ├── phase2/        # LLM-powered
   │   └── phase3/        # Long-term intelligence
   └── utils/             # Helpers
```

**Benefits**:
- ✅ Clear module boundaries
- ✅ Easy to navigate
- ✅ Obvious where new code goes
- ✅ Can enable/disable intelligence phases
- ✅ Clean separation of concerns

---

## File Count

| Directory | Files | Purpose |
|-----------|-------|---------|
| `app/core/` | 5 | Core agent logic |
| `app/storage/` | 1 | Data persistence |
| `app/intelligence/phase1/` | 1 | Persistent memory |
| `app/intelligence/phase2/` | 2 | LLM-powered intelligence |
| `app/intelligence/phase3/` | 4 | Long-term intelligence |
| `app/utils/` | 3 | Utilities |
| `app/` (root) | 4 | Entry point, models, streaming |
| **Total** | **20** | Clean, organized |

---

## Testing Organization

Run tests from project root:
```bash
# Test Phase 1
python3 test_phase1.py

# Test Phase 3
python3 test_phase3.py

# Test IDE simulator
python3 -m tests.ide_simulator

# Test specific examples
python3 -m pytest tests/examples/
```

---

## Documentation Organization

Move docs to dedicated directory:
```bash
mkdir docs/
mv PHASE*.md docs/
mv ALL_PHASES_SUMMARY.md docs/
mv INTEGRATION_COMPLETE.md docs/
mv PROJECT_STRUCTURE.md docs/  # This file
```

Result:
```
docs/
├── PROJECT_STRUCTURE.md         # This file
├── PHASE1_README.md
├── PHASE2_BETTER_DESIGN.md
├── PHASE3_README.md
├── ALL_PHASES_SUMMARY.md
└── INTEGRATION_COMPLETE.md
```

---

## Migration Checklist

✅ Created organized directory structure
✅ Moved files to appropriate directories
✅ Updated all imports in moved files
✅ Updated test files
✅ Created __init__.py for each module
✅ Documented new structure

**Status**: Organization complete. Much cleaner!

---

## Future Organization

As the project grows:

### Add API module
```python
app/api/
├── __init__.py
├── chat.py        # /chat endpoint
├── search.py      # /search, /trace, /impact
├── memory.py      # /memory endpoints
└── health.py      # /health, /metrics
```

### Add monitoring module
```python
app/monitoring/
├── __init__.py
├── metrics.py     # Prometheus metrics
└── logging.py     # Structured logging
```

### Add config module
```python
app/config/
├── __init__.py
├── settings.py    # Centralized configuration
└── models.py      # Config models
```

---

## Summary

**Before**: 20 files flat in `app/`, hard to navigate
**After**: Organized into 7 logical modules, clear structure

**Key improvements**:
- 🗂️ Clear module boundaries
- 🎯 Easy to find code
- 🔌 Phases can be enabled/disabled independently
- 📚 Better documentation organization
- ✅ Professional project structure

**Project is now well-organized and scalable.**
