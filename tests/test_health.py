"""
Tests for health and root endpoints.
"""
import pytest


@pytest.mark.asyncio
async def test_root(client):
    """Root endpoint returns service info."""
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert "service" in data
    assert "version" in data
    assert "docs" in data
    assert "supported_languages" in data


@pytest.mark.asyncio
async def test_health_check(client):
    """Health check returns healthy status."""
    response = await client.get("/health/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "supported_languages" in data


@pytest.mark.asyncio
async def test_health_ready(client):
    """Readiness check returns ready."""
    response = await client.get("/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"


@pytest.mark.asyncio
async def test_health_live(client):
    """Liveness check returns alive."""
    response = await client.get("/health/live")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "alive"


@pytest.mark.asyncio
async def test_favicon_returns_no_content(client):
    """Favicon returns 204 No Content."""
    response = await client.get("/favicon.ico")
    assert response.status_code == 204
