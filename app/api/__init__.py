"""
API Module - All FastAPI endpoints organized by domain.

Each module contains related endpoints and uses FastAPI APIRouter for modularity.
"""

from . import health
from . import memory
from . import llm_config
from . import debug
from . import auth_endpoints
from . import watch
from . import search
from . import analysis
from . import chat
from . import traces

__all__ = [
    "health",
    "memory", 
    "llm_config",
    "debug",
    "auth_endpoints",
    "watch",
    "search",
    "analysis",
    "chat",
    "traces",
]
