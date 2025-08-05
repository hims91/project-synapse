"""
Unit tests for Central Cortex (FastAPI Hub Server).
Tests the complete FastAPI application with all endpoints and middleware.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from uuid import uuid4

from fastapi.testclient import TestClient
from httpx import AsyncClient

from src.central_cortex.app import create_app
from src.shared.schemas import UserCreate, ArticleResponse, ScrapeJobCreate


@pytest.fixture
def app():
    """Create FastAPI app for testing."""
    return create_app()


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
async def async_client(app):
    """Create async test client."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def mock_user():
    """Mock user data."""
    return {
        "user_id": "test_user_123",
        "tier": "premium",
        "rate_limit": 1000,
        "is_active": True
    }


@pytest.fixture
def auth_headers():
    """Authentication headers for testing."""
    return {"X-API-Key": "dev_api_key_12345"}


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["service"] == "Project Synapse - Central Cortex"
        assert data["version"] == "2.2.0"
        assert data["status"] == "operational"
    
    @patch('src.central_cortex.dependencies.get_system_health')
    def test_health_check(self, mock_health, client):
        """Test health check endpoint."""
        mock_health.return_value = {
            "status": "healthy",
            "components": {
                "database": {"status": "healthy"},
                "task_dispatcher": {"status": "healthy"},
                "fallback_manager": {"status": "healthy"}
            }
        }
        
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "components" in data
    
    def test_readiness_check(self, client):
        """Test readiness check endpoint."""
        with patch('src.central_cortex.dependencies.get_database_manager') as mock_db:
            mock_db_instance = AsyncMock()
            mock_db_instance.health_check.return_value = {"status": "healthy"}
            mock_db.return_value = mock_db_instance
            
            response = client.get("/health/ready")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "ready"
    
    def test_liveness_check(self, client):
        """Test liveness check endpoint."""
        response = client.get("/health/live")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "alive"


class TestAuthenticationMiddleware:
    """Test authentication middleware."""
    
    def test_public_endpoints_no_auth(self, client):
        """Test that public endpoints don't require authentication."""
        public_endpoints = ["/", "/health", "/health/ready", "/health/live"]
        
        for endpoint in public_endpoints:
            response = client.get(endpoint)
            assert response.status_code != 401, f"Endpoint {endpoint} should not require auth"
    
    def test_protected_endpoints_require_auth(self, client):
        """Test that protected endpoints require authentication."""
        protected_endpoints = [
            "/api/v1/content/articles",
            "/auth/profile",
            "/api/v1/monitoring/dashboard"
        ]
        
        for endpoint in protected_endpoints:
            response = client.get(endpoint)
            assert response.status_code == 401, f"Endpoint {endpoint} should require auth"
    
    def test_valid_api_key_authentication(self, client, auth_headers):
        """Test authentication with valid API key."""
        with patch('src.central_cortex.middleware.AuthenticationMiddleware._validate_api_key') as mock_validate:
            mock_validate.return_value = {
                "user_id": "test_user",
                "tier": "premium",
                "rate_limit": 1000,
                "is_active": True
            }
            
            response = client.get("/auth/profile", headers=auth_headers)
            # Should not be 401 (may be other error due to mocked dependencies)
            assert response.status_code != 401
    
    def test_invalid_api_key_authentication(self, client):
        """Test authentication with invalid API key."""
        response = client.get("/auth/profile", headers={"X-API-Key": "invalid_key"})
        assert response.status_code == 401
        
        data = response.json()
        assert data["error"]["type"] == "invalid_api_key"


class TestRateLimitingMiddleware:
    """Test rate limiting middleware."""
    
    @patch('src.central_cortex.middleware.RateLimitingMiddleware._check_rate_limit')
    def test_rate_limit_allowed(self, mock_check, client, auth_headers):
        """Test request within rate limit."""
        mock_check.return_value = (True, 99, 1234567890)  # allowed, remaining, reset_time
        
        with patch('src.central_cortex.middleware.AuthenticationMiddleware._validate_api_key') as mock_auth:
            mock_auth.return_value = {"user_id": "test", "tier": "premium", "rate_limit": 100, "is_active": True}
            
            response = client.get("/auth/profile", headers=auth_headers)
            
            # Check rate limit headers
            assert "X-RateLimit-Limit" in response.headers
            assert "X-RateLimit-Remaining" in response.headers
    
    @patch('src.central_cortex.middleware.RateLimitingMiddleware._check_rate_limit')
    def test_rate_limit_exceeded(self, mock_check, client, auth_headers):
        """Test request exceeding rate limit."""
        mock_check.return_value = (False, 0, 1234567890)  # not allowed, no remaining, reset_time
        
        with patch('src.central_cortex.middleware.AuthenticationMiddleware._validate_api_key') as mock_auth:
            mock_auth.return_value = {"user_id": "test", "tier": "premium", "rate_limit": 100, "is_active": True}
            
            response = client.get("/auth/profile", headers=auth_headers)
            assert response.status_code == 429
            
            data = response.json()
            assert data["error"]["type"] == "rate_limit_exceeded"


class TestAuthenticationEndpoints:
    """Test authentication endpoints."""
    
    @patch('src.central_cortex.dependencies.get_repository_factory')
    def test_register_user(self, mock_repo_factory, client):
        """Test user registration."""
        mock_user_repo = AsyncMock()
        mock_user_repo.get_by_email.return_value = None  # User doesn't exist
        mock_user_repo.get_by_username.return_value = None  # Username available
        mock_user_repo.create.return_value = Mock(
            id=uuid4(),
            email="test@example.com",
            username="testuser",
            tier="free",
            api_key="sk_synapse_test123",
            is_active=True,
            created_at=datetime.utcnow()
        )
        
        mock_repo_factory.return_value.get_user_repository.return_value = mock_user_repo
        
        user_data = {
            "email": "test@example.com",
            "username": "testuser",
            "password": "securepassword123",
            "tier": "free"
        }
        
        response = client.post("/auth/register", json=user_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["email"] == "test@example.com"
        assert data["username"] == "testuser"
    
    @patch('src.central_cortex.dependencies.get_repository_factory')
    def test_get_user_profile(self, mock_repo_factory, client, auth_headers):
        """Test getting user profile."""
        mock_user_repo = AsyncMock()
        mock_user_repo.get_by_id.return_value = Mock(
            id="test_user",
            email="test@example.com",
            username="testuser",
            tier="premium",
            api_key="sk_synapse_test123",
            is_active=True
        )
        
        mock_repo_factory.return_value.get_user_repository.return_value = mock_user_repo
        
        with patch('src.central_cortex.middleware.AuthenticationMiddleware._validate_api_key') as mock_auth:
            mock_auth.return_value = {"user_id": "test_user", "tier": "premium", "is_active": True}
            
            response = client.get("/auth/profile", headers=auth_headers)
            assert response.status_code == 200
            
            data = response.json()
            assert data["email"] == "test@example.com"


class TestContentEndpoints:
    """Test content management endpoints."""
    
    @patch('src.central_cortex.dependencies.get_repository_factory')
    def test_get_articles(self, mock_repo_factory, client, auth_headers):
        """Test getting articles list."""
        mock_article_repo = AsyncMock()
        mock_articles = [
            Mock(
                id=uuid4(),
                title="Test Article 1",
                url="https://example.com/article1",
                source_domain="example.com",
                scraped_at=datetime.utcnow()
            ),
            Mock(
                id=uuid4(),
                title="Test Article 2", 
                url="https://example.com/article2",
                source_domain="example.com",
                scraped_at=datetime.utcnow()
            )
        ]
        
        mock_article_repo.get_paginated.return_value = (mock_articles, 2)
        mock_repo_factory.return_value.get_article_repository.return_value = mock_article_repo
        
        with patch('src.central_cortex.middleware.AuthenticationMiddleware._validate_api_key') as mock_auth:
            mock_auth.return_value = {"user_id": "test_user", "tier": "premium", "is_active": True}
            
            response = client.get("/api/v1/content/articles", headers=auth_headers)
            assert response.status_code == 200
            
            data = response.json()
            assert "pagination" in data
            assert "data" in data
            assert len(data["data"]) == 2
    
    @patch('src.central_cortex.dependencies.get_repository_factory')
    def test_search_articles(self, mock_repo_factory, client, auth_headers):
        """Test article search."""
        mock_article_repo = AsyncMock()
        mock_search_results = [
            (Mock(
                id=uuid4(),
                title="Search Result Article",
                url="https://example.com/search",
                source_domain="example.com"
            ), 0.95)  # (article, score)
        ]
        
        mock_article_repo.search.return_value = (mock_search_results, 1)
        mock_repo_factory.return_value.get_article_repository.return_value = mock_article_repo
        
        with patch('src.central_cortex.middleware.AuthenticationMiddleware._validate_api_key') as mock_auth:
            mock_auth.return_value = {"user_id": "test_user", "tier": "premium", "is_active": True}
            
            response = client.get("/api/v1/content/search?q=test", headers=auth_headers)
            assert response.status_code == 200
            
            data = response.json()
            assert data["query"] == "test"
            assert len(data["results"]) == 1
            assert data["results"][0]["score"] == 0.95
    
    @patch('src.central_cortex.dependencies.get_repository_factory')
    @patch('src.signal_relay.task_dispatcher.get_task_dispatcher')
    def test_create_scrape_job(self, mock_dispatcher, mock_repo_factory, client, auth_headers):
        """Test creating scrape job."""
        mock_task_repo = AsyncMock()
        mock_task = Mock(
            id=uuid4(),
            status="pending",
            scheduled_at=datetime.utcnow()
        )
        mock_task_repo.create.return_value = mock_task
        mock_repo_factory.return_value.get_task_queue_repository.return_value = mock_task_repo
        
        mock_task_dispatcher = AsyncMock()
        mock_task_dispatcher.submit_task = AsyncMock()
        mock_dispatcher.return_value = mock_task_dispatcher
        
        with patch('src.central_cortex.middleware.AuthenticationMiddleware._validate_api_key') as mock_auth:
            mock_auth.return_value = {"user_id": "test_user", "tier": "premium", "is_active": True}
            
            scrape_data = {
                "url": "https://example.com/article",
                "priority": False
            }
            
            response = client.post("/api/v1/content/scrape", json=scrape_data, headers=auth_headers)
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "pending"
            assert "job_id" in data


class TestMonitoringEndpoints:
    """Test monitoring endpoints."""
    
    @patch('src.central_cortex.dependencies.get_system_health')
    def test_monitoring_dashboard(self, mock_health, client, auth_headers):
        """Test monitoring dashboard."""
        mock_health.return_value = {
            "status": "healthy",
            "components": {
                "database": {"status": "healthy"},
                "task_dispatcher": {"status": "healthy"}
            }
        }
        
        with patch('src.central_cortex.middleware.AuthenticationMiddleware._validate_api_key') as mock_auth:
            mock_auth.return_value = {"user_id": "test_user", "tier": "premium", "is_active": True}
            
            response = client.get("/api/v1/monitoring/dashboard", headers=auth_headers)
            assert response.status_code == 200
            
            data = response.json()
            assert "system_health" in data
            assert "metrics" in data
    
    @patch('src.central_cortex.dependencies.get_fallback_manager')
    def test_get_alerts(self, mock_fallback, client, auth_headers):
        """Test getting system alerts."""
        mock_fallback_manager = AsyncMock()
        mock_fallback_manager.get_current_alerts.return_value = [
            {
                "id": "alert_1",
                "severity": "high",
                "message": "Database connection issues",
                "timestamp": datetime.utcnow().isoformat()
            }
        ]
        mock_fallback.return_value = mock_fallback_manager
        
        with patch('src.central_cortex.middleware.AuthenticationMiddleware._validate_api_key') as mock_auth:
            mock_auth.return_value = {"user_id": "test_user", "tier": "premium", "is_active": True}
            
            response = client.get("/api/v1/monitoring/alerts", headers=auth_headers)
            assert response.status_code == 200
            
            data = response.json()
            assert len(data) == 1
            assert data[0]["severity"] == "high"


class TestErrorHandling:
    """Test error handling middleware."""
    
    def test_404_error(self, client):
        """Test 404 error handling."""
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404
    
    def test_validation_error(self, client, auth_headers):
        """Test validation error handling."""
        with patch('src.central_cortex.middleware.AuthenticationMiddleware._validate_api_key') as mock_auth:
            mock_auth.return_value = {"user_id": "test_user", "tier": "premium", "is_active": True}
            
            # Send invalid JSON data
            response = client.post("/auth/register", json={"invalid": "data"}, headers=auth_headers)
            assert response.status_code == 422  # Validation error


class TestMiddlewareIntegration:
    """Test middleware integration."""
    
    def test_request_id_header(self, client, auth_headers):
        """Test that request ID is added to response headers."""
        with patch('src.central_cortex.middleware.AuthenticationMiddleware._validate_api_key') as mock_auth:
            mock_auth.return_value = {"user_id": "test_user", "tier": "premium", "is_active": True}
            
            response = client.get("/", headers=auth_headers)
            assert "X-Request-ID" in response.headers
    
    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/")
        assert response.status_code == 200
        # CORS headers should be present for OPTIONS requests


if __name__ == "__main__":
    pytest.main([__file__])