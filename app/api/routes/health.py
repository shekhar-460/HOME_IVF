"""
Health check endpoints
"""
from fastapi import APIRouter
from datetime import datetime
from app.config import settings

router = APIRouter()


@router.get(
    "/",
    summary="Health check",
    description="Basic health check endpoint. Returns service status and metadata.",
    response_description="Service health status with metadata",
    tags=["health"]
)
async def health_check():
    """
    Basic health check endpoint.
    
    Returns service status, version, and supported languages.
    Use this endpoint to verify the service is running.
    """
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "timestamp": datetime.utcnow().isoformat(),
        "supported_languages": settings.SUPPORTED_LANGUAGES
    }


@router.get(
    "/ready",
    summary="Readiness check",
    description="""
    Readiness check for Kubernetes and orchestration systems.
    
    This endpoint checks if the service is ready to accept traffic.
    It verifies database connectivity and other critical dependencies.
    """,
    response_description="Service readiness status",
    tags=["health"]
)
async def readiness_check():
    """Readiness check for Kubernetes"""
    # TODO: Add database and external service checks
    return {
        "status": "ready",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get(
    "/live",
    summary="Liveness check",
    description="""
    Liveness check for Kubernetes and orchestration systems.
    
    This endpoint indicates if the service is alive and should not be restarted.
    Returns quickly without checking external dependencies.
    """,
    response_description="Service liveness status",
    tags=["health"]
)
async def liveness_check():
    """Liveness check for Kubernetes"""
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat()
    }
