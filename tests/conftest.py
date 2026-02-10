"""
Pytest configuration and shared fixtures.
Engagement tests override get_engagement_service to avoid DB and MedGemma.
"""
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.services.engagement_service import EngagementService
from app.api.routes.engagement import get_engagement_service


@pytest.fixture
def engagement_service_no_ai():
    """Engagement service without MedGemma (fast, no model load)."""
    return EngagementService(knowledge_engine=None)


@pytest.fixture(autouse=True)
def _override_engagement_dependency():
    """Override engagement dependency so tests don't need DB or KnowledgeEngine."""
    app.dependency_overrides[get_engagement_service] = lambda: EngagementService(knowledge_engine=None)
    yield
    app.dependency_overrides.pop(get_engagement_service, None)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    """Async HTTP client for testing the FastAPI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
