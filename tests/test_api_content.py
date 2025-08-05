"""
Unit tests for Content API Endpoints

Tests the core content retrieval API endpoints:
- GET /content/articles - List articles with pagination and filtering
- GET /content/articles/{id} - Retrieve single article by ID
- GET /content/articles/{id}/related - Get related articles
- GET /content/stats - Get content statistics
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4
from datetime import datetime

from fastapi.testclient import TestClient
from fastapi import FastAPI

from src.axon_interface.routers.content import router
from src.shared.schemas import ArticleResponse, PaginatedResponse


# Create test app
app = FastAPI()
app.include_router(router, prefix="/api/v1")


class TestContentAPI:
    """Test cases for Content API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_article(self):
        """Create mock article data."""
        return {
            "id": uuid4(),
            "url": "https://example.com/article",
            "title": "Test Article",
            "content": "This is test content",
            "summary": "Test summary",
            "author": "Test Author",
            "published_at": datetime.now(),
            "source_domain": "example.com",
            "scraped_at": datetime.now(),
            "nlp_data": {
                "sentiment": 0.5,
                "entities": [],
                "categories": ["tech"],
                "significance": 0.7
            },
            "page_metadata": {}
        }
    
    @pytest.fixture
    def mock_repository_factory(self):
        """Create mock repository factory."""
        factory = Mock()
        article_repo = Mock()
        factory.get_article_repository.return_value = article_repo
        return factory, article_repo
    
    def test_list_articles_success(self, client, mock_article, mock_repository_factory):
        """Test successful article listing."""
        factory, article_repo = mock_repository_factory
        
        # Mock repository response
        article_repo.list_with_filters = AsyncMock(return_value=([mock_article], 1))
        
        with patch('src.axon_interface.routers.content.get_repository_factory', return_value=factory):
            with patch('src.axon_interface.routers.content.Request') as mock_request:
                mock_request.state.user_id = "test-user"
                
                response = client.get(
                    "/api/v1/content/articles",
                    headers={"Authorization": "Bearer test-api-key-123"}
                )
        
        assert response.status_code == 200
        data = response.json()
        assert "pagination" in data
        assert "data" in data
        assert len(data["data"]) == 1
    
    def test_list_articles_with_filters(self, client, mock_article, mock_repository_factory):
        """Test article listing with filters."""
        factory, article_repo = mock_repository_factory
        article_repo.list_with_filters = AsyncMock(return_value=([mock_article], 1))
        
        with patch('src.axon_interface.routers.content.get_repository_factory', return_value=factory):
            with patch('src.axon_interface.routers.content.Request') as mock_request:
                mock_request.state.user_id = "test-user"
                
                response = client.get(
                    "/api/v1/content/articles",
                    params={
                        "source_domain": "example.com",
                        "categories": "tech,news",
                        "min_sentiment": 0.0,
                        "max_sentiment": 1.0,
                        "page": 1,
                        "page_size": 10
                    },
                    headers={"Authorization": "Bearer test-api-key-123"}
                )
        
        assert response.status_code == 200
        # Verify filters were passed to repository
        article_repo.list_with_filters.assert_called_once()
        call_args = article_repo.list_with_filters.call_args
        filters = call_args[1]["filters"]
        assert filters["source_domain"] == "example.com"
        assert filters["categories"] == ["tech", "news"]
    
    def test_get_article_success(self, client, mock_article, mock_repository_factory):
        """Test successful single article retrieval."""
        factory, article_repo = mock_repository_factory
        article_repo.get_by_id = AsyncMock(return_value=mock_article)
        
        article_id = str(mock_article["id"])
        
        with patch('src.axon_interface.routers.content.get_repository_factory', return_value=factory):
            with patch('src.axon_interface.routers.content.Request') as mock_request:
                mock_request.state.user_id = "test-user"
                
                response = client.get(
                    f"/api/v1/content/articles/{article_id}",
                    headers={"Authorization": "Bearer test-api-key-123"}
                )
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == article_id
        assert data["title"] == "Test Article"
    
    def test_get_article_not_found(self, client, mock_repository_factory):
        """Test article not found."""
        factory, article_repo = mock_repository_factory
        article_repo.get_by_id = AsyncMock(return_value=None)
        
        article_id = str(uuid4())
        
        with patch('src.axon_interface.routers.content.get_repository_factory', return_value=factory):
            with patch('src.axon_interface.routers.content.Request') as mock_request:
                mock_request.state.user_id = "test-user"
                
                response = client.get(
                    f"/api/v1/content/articles/{article_id}",
                    headers={"Authorization": "Bearer test-api-key-123"}
                )
        
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()
    
    def test_get_related_articles_success(self, client, mock_article, mock_repository_factory):
        """Test successful related articles retrieval."""
        factory, article_repo = mock_repository_factory
        article_repo.get_by_id = AsyncMock(return_value=mock_article)
        article_repo.find_related_articles = AsyncMock(return_value=[mock_article])
        
        article_id = str(mock_article["id"])
        
        with patch('src.axon_interface.routers.content.get_repository_factory', return_value=factory):
            with patch('src.axon_interface.routers.content.Request') as mock_request:
                mock_request.state.user_id = "test-user"
                
                response = client.get(
                    f"/api/v1/content/articles/{article_id}/related",
                    headers={"Authorization": "Bearer test-api-key-123"}
                )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
    
    def test_get_related_articles_source_not_found(self, client, mock_repository_factory):
        """Test related articles when source article not found."""
        factory, article_repo = mock_repository_factory
        article_repo.get_by_id = AsyncMock(return_value=None)
        
        article_id = str(uuid4())
        
        with patch('src.axon_interface.routers.content.get_repository_factory', return_value=factory):
            with patch('src.axon_interface.routers.content.Request') as mock_request:
                mock_request.state.user_id = "test-user"
                
                response = client.get(
                    f"/api/v1/content/articles/{article_id}/related",
                    headers={"Authorization": "Bearer test-api-key-123"}
                )
        
        assert response.status_code == 404
    
    def test_get_content_stats_success(self, client, mock_repository_factory):
        """Test successful content statistics retrieval."""
        factory, article_repo = mock_repository_factory
        
        mock_stats = {
            "total_articles": 1000,
            "articles_last_24h": 50,
            "articles_last_7d": 300,
            "top_domains": [{"domain": "example.com", "count": 100}],
            "top_categories": [{"category": "tech", "count": 200}],
            "avg_sentiment": 0.2,
            "avg_significance": 0.6,
            "last_updated": "2025-01-08T00:00:00Z"
        }
        
        article_repo.get_content_statistics = AsyncMock(return_value=mock_stats)
        
        with patch('src.axon_interface.routers.content.get_repository_factory', return_value=factory):
            with patch('src.axon_interface.routers.content.Request') as mock_request:
                mock_request.state.user_id = "test-user"
                
                response = client.get(
                    "/api/v1/content/stats",
                    headers={"Authorization": "Bearer test-api-key-123"}
                )
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_articles"] == 1000
        assert data["articles_last_24h"] == 50
        assert len(data["top_domains"]) == 1
    
    def test_list_articles_pagination(self, client, mock_article, mock_repository_factory):
        """Test article listing pagination."""
        factory, article_repo = mock_repository_factory
        
        # Mock multiple articles
        articles = [mock_article] * 5
        article_repo.list_with_filters = AsyncMock(return_value=(articles, 25))
        
        with patch('src.axon_interface.routers.content.get_repository_factory', return_value=factory):
            with patch('src.axon_interface.routers.content.Request') as mock_request:
                mock_request.state.user_id = "test-user"
                
                response = client.get(
                    "/api/v1/content/articles",
                    params={"page": 2, "page_size": 5},
                    headers={"Authorization": "Bearer test-api-key-123"}
                )
        
        assert response.status_code == 200
        data = response.json()
        
        pagination = data["pagination"]
        assert pagination["page"] == 2
        assert pagination["page_size"] == 5
        assert pagination["total_results"] == 25
        assert pagination["total_pages"] == 5
    
    def test_list_articles_sorting(self, client, mock_article, mock_repository_factory):
        """Test article listing with sorting."""
        factory, article_repo = mock_repository_factory
        article_repo.list_with_filters = AsyncMock(return_value=([mock_article], 1))
        
        with patch('src.axon_interface.routers.content.get_repository_factory', return_value=factory):
            with patch('src.axon_interface.routers.content.Request') as mock_request:
                mock_request.state.user_id = "test-user"
                
                response = client.get(
                    "/api/v1/content/articles",
                    params={"sort_by": "significance", "sort_order": "asc"},
                    headers={"Authorization": "Bearer test-api-key-123"}
                )
        
        assert response.status_code == 200
        # Verify sorting parameters were passed
        call_args = article_repo.list_with_filters.call_args
        assert call_args[1]["sort_by"] == "significance"
        assert call_args[1]["sort_order"] == "asc"
    
    def test_list_articles_invalid_page(self, client):
        """Test article listing with invalid page number."""
        response = client.get(
            "/api/v1/content/articles",
            params={"page": 0},
            headers={"Authorization": "Bearer test-api-key-123"}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_list_articles_invalid_page_size(self, client):
        """Test article listing with invalid page size."""
        response = client.get(
            "/api/v1/content/articles",
            params={"page_size": 101},  # Exceeds maximum
            headers={"Authorization": "Bearer test-api-key-123"}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_get_article_invalid_uuid(self, client):
        """Test get article with invalid UUID."""
        response = client.get(
            "/api/v1/content/articles/invalid-uuid",
            headers={"Authorization": "Bearer test-api-key-123"}
        )
        
        assert response.status_code == 422  # Validation error


class TestContentAPIErrorHandling:
    """Test error handling in Content API."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_list_articles_repository_error(self, client):
        """Test handling of repository errors."""
        factory = Mock()
        article_repo = Mock()
        article_repo.list_with_filters = AsyncMock(side_effect=Exception("Database error"))
        factory.get_article_repository.return_value = article_repo
        
        with patch('src.axon_interface.routers.content.get_repository_factory', return_value=factory):
            with patch('src.axon_interface.routers.content.Request') as mock_request:
                mock_request.state.user_id = "test-user"
                
                response = client.get(
                    "/api/v1/content/articles",
                    headers={"Authorization": "Bearer test-api-key-123"}
                )
        
        assert response.status_code == 500
        data = response.json()
        assert "Failed to retrieve articles" in data["detail"]
    
    def test_get_article_repository_error(self, client):
        """Test handling of repository errors in single article retrieval."""
        factory = Mock()
        article_repo = Mock()
        article_repo.get_by_id = AsyncMock(side_effect=Exception("Database error"))
        factory.get_article_repository.return_value = article_repo
        
        article_id = str(uuid4())
        
        with patch('src.axon_interface.routers.content.get_repository_factory', return_value=factory):
            with patch('src.axon_interface.routers.content.Request') as mock_request:
                mock_request.state.user_id = "test-user"
                
                response = client.get(
                    f"/api/v1/content/articles/{article_id}",
                    headers={"Authorization": "Bearer test-api-key-123"}
                )
        
        assert response.status_code == 500
        data = response.json()
        assert "Failed to retrieve article" in data["detail"]


if __name__ == "__main__":
    pytest.main([__file__])