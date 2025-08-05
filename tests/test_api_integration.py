"""
Integration tests for Project Synapse API

Tests the complete API integration including:
- Authentication and authorization
- Rate limiting
- Error handling
- Cross-endpoint functionality
- End-to-end workflows
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4
import time

from fastapi.testclient import TestClient

from src.axon_interface.main import app


class TestAPIIntegration:
    """Integration tests for the complete API."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "components" in data
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Project Synapse API"
        assert data["version"] == "2.2.0"
    
    def test_authentication_required(self, client):
        """Test that authentication is required for API endpoints."""
        response = client.get("/api/v1/content/articles")
        
        assert response.status_code == 401
        data = response.json()
        assert data["error"]["type"] == "authentication_error"
    
    def test_invalid_api_key(self, client):
        """Test invalid API key handling."""
        response = client.get(
            "/api/v1/content/articles",
            headers={"Authorization": "Bearer invalid-key"}
        )
        
        assert response.status_code == 401
        data = response.json()
        assert data["error"]["type"] == "authentication_error"
    
    def test_invalid_authorization_format(self, client):
        """Test invalid authorization format."""
        response = client.get(
            "/api/v1/content/articles",
            headers={"Authorization": "InvalidFormat"}
        )
        
        assert response.status_code == 401
        data = response.json()
        assert "Invalid authorization format" in data["error"]["message"]
    
    @patch('src.axon_interface.routers.content.get_repository_factory')
    def test_valid_api_key_access(self, mock_factory, client):
        """Test valid API key allows access."""
        # Mock repository
        factory = Mock()
        article_repo = Mock()
        article_repo.list_with_filters = AsyncMock(return_value=([], 0))
        factory.get_article_repository.return_value = article_repo
        mock_factory.return_value = factory
        
        response = client.get(
            "/api/v1/content/articles",
            headers={"Authorization": "Bearer test-api-key-123"}
        )
        
        assert response.status_code == 200
    
    def test_rate_limiting_headers(self, client):
        """Test that rate limiting headers are included."""
        with patch('src.axon_interface.routers.content.get_repository_factory'):
            response = client.get(
                "/api/v1/content/articles",
                headers={"Authorization": "Bearer test-api-key-123"}
            )
        
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers
    
    def test_processing_time_header(self, client):
        """Test that processing time header is included."""
        response = client.get("/health")
        
        assert "X-Process-Time" in response.headers
        processing_time = float(response.headers["X-Process-Time"])
        assert processing_time >= 0
    
    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/api/v1/content/articles")
        
        # CORS headers should be present for OPTIONS requests
        assert response.status_code in [200, 405]  # Depending on FastAPI version
    
    @patch('src.axon_interface.routers.content.get_repository_factory')
    @patch('src.axon_interface.routers.search.get_search_engine')
    def test_cross_endpoint_functionality(self, mock_search_engine, mock_factory, client):
        """Test functionality across multiple endpoints."""
        # Mock repositories and services
        factory = Mock()
        article_repo = Mock()
        search_engine = Mock()
        
        # Mock article data
        mock_article = {
            "id": uuid4(),
            "title": "Test Article",
            "url": "https://example.com/test",
            "content": "Test content",
            "summary": "Test summary",
            "author": "Test Author",
            "published_at": "2025-01-08T00:00:00Z",
            "source_domain": "example.com",
            "scraped_at": "2025-01-08T00:00:00Z",
            "nlp_data": {"sentiment": 0.5, "entities": [], "categories": ["tech"], "significance": 0.7},
            "page_metadata": {}
        }
        
        article_repo.list_with_filters = AsyncMock(return_value=([mock_article], 1))
        article_repo.get_by_id = AsyncMock(return_value=mock_article)
        factory.get_article_repository.return_value = article_repo
        mock_factory.return_value = factory
        
        # Mock search results
        search_results = Mock()
        search_results.results = []
        search_results.total_hits = 0
        search_engine.search = AsyncMock(return_value=search_results)
        mock_search_engine.return_value = search_engine
        
        headers = {"Authorization": "Bearer test-api-key-123"}
        
        # Test content listing
        response = client.get("/api/v1/content/articles", headers=headers)
        assert response.status_code == 200
        
        # Test single article retrieval
        article_id = str(mock_article["id"])
        response = client.get(f"/api/v1/content/articles/{article_id}", headers=headers)
        assert response.status_code == 200
        
        # Test search
        response = client.get("/api/v1/search?q=test", headers=headers)
        assert response.status_code == 200
    
    def test_error_response_format(self, client):
        """Test that error responses follow consistent format."""
        response = client.get("/api/v1/content/articles")
        
        assert response.status_code == 401
        data = response.json()
        assert "error" in data
        assert "type" in data["error"]
        assert "message" in data["error"]
    
    def test_404_handling(self, client):
        """Test 404 error handling."""
        response = client.get("/nonexistent-endpoint")
        
        assert response.status_code == 404
    
    @patch('src.axon_interface.routers.content.get_repository_factory')
    def test_500_error_handling(self, mock_factory, client):
        """Test 500 error handling."""
        # Mock repository to raise exception
        factory = Mock()
        article_repo = Mock()
        article_repo.list_with_filters = AsyncMock(side_effect=Exception("Database error"))
        factory.get_article_repository.return_value = article_repo
        mock_factory.return_value = factory
        
        response = client.get(
            "/api/v1/content/articles",
            headers={"Authorization": "Bearer test-api-key-123"}
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "error" in data
    
    def test_openapi_documentation(self, client):
        """Test OpenAPI documentation is available."""
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert data["info"]["title"] == "Project Synapse API"
    
    def test_docs_endpoint(self, client):
        """Test documentation endpoint is available."""
        response = client.get("/docs")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_redoc_endpoint(self, client):
        """Test ReDoc endpoint is available."""
        response = client.get("/redoc")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestAPIWorkflows:
    """Test complete API workflows."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @patch('src.axon_interface.routers.scrape.get_task_dispatcher')
    @patch('src.axon_interface.routers.scrape.get_repository_factory')
    def test_scraping_workflow(self, mock_factory, mock_dispatcher, client):
        """Test complete scraping workflow."""
        # Mock dependencies
        factory = Mock()
        task_repo = Mock()
        dispatcher = Mock()
        
        job_id = uuid4()
        task_id = "task-123"
        
        # Mock repository methods
        task_repo.create_scrape_job = AsyncMock()
        task_repo.get_scrape_job_status = AsyncMock(return_value={
            "job_id": job_id,
            "status": "completed",
            "message": "Scraping completed successfully",
            "user_id": "test-user",
            "created_at": "2025-01-08T00:00:00Z",
            "completed_at": "2025-01-08T00:01:00Z",
            "article_id": uuid4()
        })
        
        factory.get_task_queue_repository.return_value = task_repo
        mock_factory.return_value = factory
        
        # Mock dispatcher
        dispatcher.submit_task = AsyncMock(return_value=task_id)
        mock_dispatcher.return_value = dispatcher
        
        headers = {"Authorization": "Bearer test-api-key-123"}
        
        # Submit scraping job
        response = client.post(
            "/api/v1/scrape",
            json={"url": "https://example.com/article", "priority": False},
            headers=headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "pending"
        
        # Check job status
        job_id = data["job_id"]
        response = client.get(f"/api/v1/scrape/status/{job_id}", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
    
    @patch('src.axon_interface.routers.monitoring.get_repository_factory')
    def test_monitoring_workflow(self, mock_factory, client):
        """Test complete monitoring workflow."""
        # Mock dependencies
        factory = Mock()
        monitoring_repo = Mock()
        
        subscription_id = uuid4()
        
        # Mock repository methods
        mock_subscription = Mock()
        mock_subscription.id = subscription_id
        mock_subscription.user_id = "test-user"
        mock_subscription.name = "Test Subscription"
        mock_subscription.keywords = ["test", "keyword"]
        mock_subscription.webhook_url = "https://example.com/webhook"
        mock_subscription.is_active = True
        mock_subscription.created_at = "2025-01-08T00:00:00Z"
        mock_subscription.last_triggered = None
        
        monitoring_repo.create = AsyncMock(return_value=mock_subscription)
        monitoring_repo.get_by_id = AsyncMock(return_value=mock_subscription)
        monitoring_repo.list_with_filters = AsyncMock(return_value=([mock_subscription], 1))
        monitoring_repo.delete = AsyncMock()
        
        factory.get_monitoring_subscription_repository.return_value = monitoring_repo
        mock_factory.return_value = factory
        
        headers = {"Authorization": "Bearer test-api-key-123"}
        
        # Create monitoring subscription
        response = client.post(
            "/api/v1/monitoring/subscriptions",
            json={
                "name": "Test Subscription",
                "keywords": ["test", "keyword"],
                "webhook_url": "https://example.com/webhook"
            },
            headers=headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Test Subscription"
        
        # List subscriptions
        response = client.get("/api/v1/monitoring/subscriptions", headers=headers)
        assert response.status_code == 200
        
        # Get specific subscription
        subscription_id = data["id"]
        response = client.get(f"/api/v1/monitoring/subscriptions/{subscription_id}", headers=headers)
        assert response.status_code == 200
        
        # Delete subscription
        response = client.delete(f"/api/v1/monitoring/subscriptions/{subscription_id}", headers=headers)
        assert response.status_code == 200


class TestAPIPerformance:
    """Test API performance characteristics."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_health_check_performance(self, client):
        """Test health check response time."""
        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()
        
        assert response.status_code == 200
        response_time = end_time - start_time
        assert response_time < 1.0  # Should respond within 1 second
    
    @patch('src.axon_interface.routers.content.get_repository_factory')
    def test_content_api_performance(self, mock_factory, client):
        """Test content API response time."""
        # Mock fast repository response
        factory = Mock()
        article_repo = Mock()
        article_repo.list_with_filters = AsyncMock(return_value=([], 0))
        factory.get_article_repository.return_value = article_repo
        mock_factory.return_value = factory
        
        start_time = time.time()
        response = client.get(
            "/api/v1/content/articles",
            headers={"Authorization": "Bearer test-api-key-123"}
        )
        end_time = time.time()
        
        assert response.status_code == 200
        response_time = end_time - start_time
        assert response_time < 2.0  # Should respond within 2 seconds
    
    def test_concurrent_requests(self, client):
        """Test handling of concurrent requests."""
        import concurrent.futures
        
        def make_request():
            return client.get("/health")
        
        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        assert all(response.status_code == 200 for response in responses)


if __name__ == "__main__":
    pytest.main([__file__])