"""Health check endpoint."""

from fastapi import APIRouter
from ..storage import store
from ..models import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health():
    """Healthcheck endpoint."""
    store_ok = await store.check_connection()
    status = "healthy" if store_ok else "degraded"
    return HealthResponse(
        status=status,
        store_ok=store_ok,
        version="1.0.0",
    )
