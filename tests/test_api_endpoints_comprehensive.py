"""
Comprehensive automated testing for all Project Synapse API endpoints.

Tests every API endpoint with various scenarios including success cases,
error cases, edge cases, and validation testing.
"""

import pytest
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient

from src.axon_interface.main import app


class TestAPIEndpointsComprehensive:
    """Comprehensive API endpoint testing."""
    
    @pytest.fixture(scope="class")
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture(scope="class")
    def auth_headers(self):
        """Create authentication headers."""
        return {"Authorization": "Bearer test_api_key_comprehensive"}
    
    @pytest.fixture(scope="class")
    def mock_auth(self):
        """Mock authentication for all tests."""
        with patch('src.axon_interface.auth.validate_api_key') as mock:
            mock.return_value = {
                "user_id": "comprehensive_test_user",
                "tier": "pro",
                "rate_limit": 1000
            }
            yield mock

    # Health and System Endpoints
    def test_health_endpoint(self, client):
        """Test /health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "timestamp" in data
        assert "components" in data
        
        # Test component health
        components = data["components"]
        assert "database" in components
        assert "cache" in components
        assert "task_dispatcher" in components
    
    def test_status_endpoint(self, client):
        """Test /status endpoint."""
        response = client.get("/status")
        assert response.status_code == 200
        
        data = response.json()
        assert "uptime" in data
        assert "version" in data
        assert "environment" in data
        assert "components" in data
    
    def test_metrics_endpoint(self, client):
        """Test /metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        
        # Should return Prometheus format
        content = response.text
        assert "# HELP" in content
        assert "# TYPE" in content
    
    def test_openapi_schema(self, client):
        """Test OpenAPI schema endpoint."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert "openapi" in schema
        assert "paths" in schema
        assert "components" in schema

    # Content API Endpoints
    def test_content_articles_endpoint(self, client, auth_headers, mock_auth):
        """Test /content/articles endpoint."""
        with patch('src.shared.database.repositories.ArticleRepository.get_articles') as mock_get:
            mock_get.return_value = {
                "articles": [
                    {
                        "id": "article_1",
                        "title": "Test Article 1",
                        "content": "Content of test article 1",
                        "url": "https://example.com/article1",
                        "created_at": datetime.utcnow().isoformat(),
                        "metadata": {"author": "Test Author"}
                    }
                ],
                "total": 1,
                "page": 1,
                "per_page": 20
            }
            
            # Test basic request
            response = client.get("/content/articles", headers=auth_headers)
            assert response.status_code == 200
            
            data = response.json()
            assert "articles" in data
            assert "total" in data
            assert len(data["articles"]) == 1
            
            # Test with pagination
            response = client.get("/content/articles?page=2&per_page=10", headers=auth_headers)
            assert response.status_code == 200
            
            # Test with filters
            response = client.get("/content/articles?sentiment=positive&topic=technology", 
                                headers=auth_headers)
            assert response.status_code == 200
    
    def test_content_article_by_id(self, client, auth_headers, mock_auth):
        """Test /content/articles/{id} endpoint."""
        article_id = "test_article_123"
        
        with patch('src.shared.database.repositories.ArticleRepository.get_article_by_id') as mock_get:
            mock_get.return_value = {
                "id": article_id,
                "title": "Test Article",
                "content": "Full content of the test article",
                "url": "https://example.com/test-article",
                "analysis": {
                    "sentiment": {"score": 0.75, "label": "positive"},
                    "topics": [{"name": "technology", "confidence": 0.89}]
                }
            }
            
            response = client.get(f"/content/articles/{article_id}", headers=auth_headers)
            assert response.status_code == 200
            
            data = response.json()
            assert data["id"] == article_id
            assert "analysis" in data
            
            # Test non-existent article
            mock_get.return_value = None
            response = client.get("/content/articles/nonexistent", headers=auth_headers)
            assert response.status_code == 404

    # Search API Endpoints
    def test_search_endpoint(self, client, auth_headers, mock_auth):
        """Test /search endpoint."""
        with patch('src.thalamus.semantic_search.SemanticSearchEngine.search') as mock_search:
            mock_search.return_value = {
                "results": [
                    {
                        "id": "result_1",
                        "title": "Search Result 1",
                        "content": "Content matching the search query",
                        "relevance_score": 0.95,
                        "url": "https://example.com/result1"
                    }
                ],
                "total": 1,
                "query_time_ms": 45,
                "query": "artificial intelligence"
            }
            
            # Test basic search
            search_request = {
                "query": "artificial intelligence",
                "limit": 10
            }
            
            response = client.post("/search", json=search_request, headers=auth_headers)
            assert response.status_code == 200
            
            data = response.json()
            assert "results" in data
            assert "total" in data
            assert "query_time_ms" in data
            
            # Test search with filters
            search_request_filtered = {
                "query": "machine learning",
                "limit": 5,
                "filters": {
                    "sentiment": ["positive"],
                    "topics": ["technology"],
                    "date_range": {
                        "start": "2024-01-01",
                        "end": "2024-01-08"
                    }
                }
            }
            
            response = client.post("/search", json=search_request_filtered, headers=auth_headers)
            assert response.status_code == 200
            
            # Test invalid search request
            invalid_request = {"query": ""}  # Empty query
            response = client.post("/search", json=invalid_request, headers=auth_headers)
            assert response.status_code == 400
    
    def test_search_similar_endpoint(self, client, auth_headers, mock_auth):
        """Test /search/similar endpoint."""
        with patch('src.thalamus.semantic_search.SemanticSearchEngine.find_similar') as mock_similar:
            mock_similar.return_value = {
                "similar_content": [
                    {
                        "id": "similar_1",
                        "title": "Similar Article",
                        "similarity_score": 0.87,
                        "url": "https://example.com/similar1"
                    }
                ],
                "reference_id": "ref_article_123"
            }
            
            similar_request = {
                "content_id": "ref_article_123",
                "limit": 5,
                "similarity_threshold": 0.7
            }
            
            response = client.post("/search/similar", json=similar_request, headers=auth_headers)
            assert response.status_code == 200
            
            data = response.json()
            assert "similar_content" in data
            assert "reference_id" in data

    # Scraping API Endpoints
    def test_scrape_endpoint(self, client, auth_headers, mock_auth):
        """Test /scrape endpoint."""
        with patch('src.signal_relay.task_dispatcher.TaskDispatcher.submit_task') as mock_submit:
            mock_submit.return_value = {
                "job_id": "scrape_job_123",
                "status": "queued",
                "estimated_completion": datetime.utcnow() + timedelta(minutes=5)
            }
            
            # Test basic scrape request
            scrape_request = {
                "url": "https://example.com/article-to-scrape",
                "analysis_types": ["sentiment", "topics"],
                "priority": "normal"
            }
            
            response = client.post("/scrape", json=scrape_request, headers=auth_headers)
            assert response.status_code == 202
            
            data = response.json()
            assert "job_id" in data
            assert "status" in data
            assert data["status"] == "queued"
            
            # Test invalid URL
            invalid_request = {"url": "not-a-valid-url"}
            response = client.post("/scrape", json=invalid_request, headers=auth_headers)
            assert response.status_code == 400
            
            # Test missing URL
            missing_url_request = {"analysis_types": ["sentiment"]}
            response = client.post("/scrape", json=missing_url_request, headers=auth_headers)
            assert response.status_code == 422
    
    def test_scrape_status_endpoint(self, client, auth_headers, mock_auth):
        """Test /scrape/status/{job_id} endpoint."""
        job_id = "test_job_123"
        
        with patch('src.signal_relay.task_dispatcher.TaskDispatcher.get_task_status') as mock_status:
            mock_status.return_value = {
                "job_id": job_id,
                "status": "processing",
                "progress": 0.6,
                "created_at": datetime.utcnow().isoformat(),
                "estimated_completion": (datetime.utcnow() + timedelta(minutes=2)).isoformat()
            }
            
            response = client.get(f"/scrape/status/{job_id}", headers=auth_headers)
            assert response.status_code == 200
            
            data = response.json()
            assert data["job_id"] == job_id
            assert "status" in data
            assert "progress" in data
            
            # Test non-existent job
            mock_status.return_value = None
            response = client.get("/scrape/status/nonexistent", headers=auth_headers)
            assert response.status_code == 404
    
    def test_scrape_result_endpoint(self, client, auth_headers, mock_auth):
        """Test /scrape/result/{job_id} endpoint."""
        job_id = "completed_job_123"
        
        with patch('src.signal_relay.task_dispatcher.TaskDispatcher.get_task_result') as mock_result:
            mock_result.return_value = {
                "job_id": job_id,
                "status": "completed",
                "result": {
                    "content": {
                        "title": "Scraped Article",
                        "content": "Full content of the scraped article",
                        "url": "https://example.com/scraped"
                    },
                    "analysis": {
                        "sentiment": {"score": 0.82, "label": "positive"},
                        "topics": [{"name": "technology", "confidence": 0.91}]
                    }
                },
                "completed_at": datetime.utcnow().isoformat()
            }
            
            response = client.get(f"/scrape/result/{job_id}", headers=auth_headers)
            assert response.status_code == 200
            
            data = response.json()
            assert data["job_id"] == job_id
            assert data["status"] == "completed"
            assert "result" in data
            assert "analysis" in data["result"]

    # Analysis API Endpoints
    def test_analysis_content_endpoint(self, client, auth_headers, mock_auth):
        """Test /analysis/content endpoint."""
        with patch('src.thalamus.nlp_pipeline.NLPPipeline.analyze') as mock_analyze:
            mock_analyze.return_value = {
                "sentiment": {"score": 0.78, "label": "positive", "confidence": 0.92},
                "bias": {"overall_score": 0.15, "types": {"political": 0.05}},
                "topics": [{"name": "healthcare", "confidence": 0.91}],
                "entities": [{"text": "artificial intelligence", "type": "TECHNOLOGY"}],
                "summary": {"text": "AI is transforming healthcare.", "length": 32}
            }
            
            analysis_request = {
                "text": "Artificial intelligence is revolutionizing healthcare with new diagnostic tools.",
                "analysis_types": ["sentiment", "bias", "topics", "entities", "summary"]
            }
            
            response = client.post("/analysis/content", json=analysis_request, headers=auth_headers)
            assert response.status_code == 200
            
            data = response.json()
            assert "sentiment" in data
            assert "bias" in data
            assert "topics" in data
            assert "entities" in data
            assert "summary" in data
            
            # Test with URL instead of text
            url_request = {
                "url": "https://example.com/article",
                "analysis_types": ["sentiment"]
            }
            
            response = client.post("/analysis/content", json=url_request, headers=auth_headers)
            assert response.status_code == 200
            
            # Test invalid request (no text or URL)
            invalid_request = {"analysis_types": ["sentiment"]}
            response = client.post("/analysis/content", json=invalid_request, headers=auth_headers)
            assert response.status_code == 400
    
    def test_analysis_bias_endpoint(self, client, auth_headers, mock_auth):
        """Test /analysis/bias endpoint."""
        with patch('src.thalamus.bias_analyzer.BiasAnalyzer.analyze') as mock_bias:
            mock_bias.return_value = {
                "bias": {
                    "overall_score": 0.34,
                    "types": {
                        "political": {"score": 0.28, "direction": "left"},
                        "corporate": {"score": 0.15, "indicators": ["favoritism"]}
                    }
                },
                "narrative": {
                    "framing": "government_criticism",
                    "perspective": "skeptical",
                    "key_phrases": ["clear favoritism", "tech companies"]
                }
            }
            
            bias_request = {
                "text": "The government's new AI policy shows clear favoritism toward tech companies.",
                "analysis_types": ["bias", "narrative", "framing"]
            }
            
            response = client.post("/analysis/bias", json=bias_request, headers=auth_headers)
            assert response.status_code == 200
            
            data = response.json()
            assert "bias" in data
            assert "narrative" in data
            assert data["bias"]["overall_score"] > 0

    # Monitoring API Endpoints
    def test_monitoring_subscriptions_crud(self, client, auth_headers, mock_auth):
        """Test monitoring subscriptions CRUD operations."""
        # Create subscription
        with patch('src.shared.database.repositories.MonitoringRepository.create_subscription') as mock_create:
            mock_create.return_value = {
                "subscription_id": "sub_123",
                "name": "AI Healthcare Monitor",
                "keywords": ["artificial intelligence", "healthcare"],
                "webhook_url": "https://example.com/webhook",
                "frequency": "daily",
                "created_at": datetime.utcnow().isoformat()
            }
            
            create_request = {
                "name": "AI Healthcare Monitor",
                "keywords": ["artificial intelligence", "healthcare"],
                "webhook_url": "https://example.com/webhook",
                "frequency": "daily"
            }
            
            response = client.post("/monitoring/subscriptions", json=create_request, headers=auth_headers)
            assert response.status_code == 201
            
            data = response.json()
            assert "subscription_id" in data
            assert data["name"] == "AI Healthcare Monitor"
        
        # List subscriptions
        with patch('src.shared.database.repositories.MonitoringRepository.get_user_subscriptions') as mock_list:
            mock_list.return_value = {
                "subscriptions": [
                    {
                        "subscription_id": "sub_123",
                        "name": "AI Healthcare Monitor",
                        "keywords": ["artificial intelligence", "healthcare"],
                        "frequency": "daily",
                        "active": True
                    }
                ],
                "total": 1
            }
            
            response = client.get("/monitoring/subscriptions", headers=auth_headers)
            assert response.status_code == 200
            
            data = response.json()
            assert "subscriptions" in data
            assert len(data["subscriptions"]) == 1
        
        # Update subscription
        subscription_id = "sub_123"
        with patch('src.shared.database.repositories.MonitoringRepository.update_subscription') as mock_update:
            mock_update.return_value = True
            
            update_request = {"frequency": "hourly"}
            response = client.patch(f"/monitoring/subscriptions/{subscription_id}", 
                                  json=update_request, headers=auth_headers)
            assert response.status_code == 200
        
        # Delete subscription
        with patch('src.shared.database.repositories.MonitoringRepository.delete_subscription') as mock_delete:
            mock_delete.return_value = True
            
            response = client.delete(f"/monitoring/subscriptions/{subscription_id}", headers=auth_headers)
            assert response.status_code == 204

    # Trends API Endpoints
    def test_trends_topics_endpoint(self, client, auth_headers, mock_auth):
        """Test /trends/topics endpoint."""
        with patch('src.thalamus.trends_analyzer.TrendsAnalyzer.get_trending_topics') as mock_trends:
            mock_trends.return_value = {
                "trending_topics": [
                    {
                        "topic": "artificial_intelligence",
                        "mentions": 1247,
                        "growth_rate": 0.45,
                        "sentiment": {"overall": 0.73},
                        "keywords": ["AI", "machine learning", "neural networks"]
                    }
                ],
                "time_range": "24h",
                "generated_at": datetime.utcnow().isoformat()
            }
            
            response = client.get("/trends/topics?time_range=24h&limit=10", headers=auth_headers)
            assert response.status_code == 200
            
            data = response.json()
            assert "trending_topics" in data
            assert "time_range" in data
            assert len(data["trending_topics"]) > 0
    
    def test_trends_analysis_endpoint(self, client, auth_headers, mock_auth):
        """Test /trends/analyze endpoint."""
        with patch('src.thalamus.trends_analyzer.TrendsAnalyzer.analyze_trends') as mock_analyze:
            mock_analyze.return_value = {
                "analysis_id": "trend_analysis_123",
                "topics": ["artificial intelligence"],
                "results": {
                    "artificial intelligence": {
                        "volume_trend": {"trend_direction": "upward", "growth_rate": 0.23},
                        "sentiment_trend": {"trend_direction": "improving", "average_sentiment": 0.67}
                    }
                }
            }
            
            analysis_request = {
                "topics": ["artificial intelligence"],
                "time_range": {
                    "start": "2024-01-01T00:00:00Z",
                    "end": "2024-01-08T23:59:59Z"
                },
                "metrics": ["volume", "sentiment"]
            }
            
            response = client.post("/trends/analyze", json=analysis_request, headers=auth_headers)
            assert response.status_code == 200
            
            data = response.json()
            assert "analysis_id" in data
            assert "results" in data

    # Financial API Endpoints
    def test_financial_market_endpoint(self, client, auth_headers, mock_auth):
        """Test /financial/market endpoint."""
        with patch('src.thalamus.financial_analyzer.FinancialAnalyzer.get_market_pulse') as mock_market:
            mock_market.return_value = {
                "market_data": [
                    {
                        "ticker": "AAPL",
                        "sentiment": {"score": 0.78, "label": "positive"},
                        "mentions": 234,
                        "trend": "bullish"
                    }
                ],
                "overall_sentiment": 0.72,
                "generated_at": datetime.utcnow().isoformat()
            }
            
            response = client.get("/financial/market?tickers=AAPL,GOOGL", headers=auth_headers)
            assert response.status_code == 200
            
            data = response.json()
            assert "market_data" in data
            assert "overall_sentiment" in data

    # Summarization API Endpoints
    def test_summarize_endpoint(self, client, auth_headers, mock_auth):
        """Test /summarize endpoint."""
        with patch('src.thalamus.summarizer.Summarizer.summarize') as mock_summarize:
            mock_summarize.return_value = {
                "summary": {
                    "text": "AI is transforming healthcare with new diagnostic tools and treatment recommendations.",
                    "length": 89,
                    "compression_ratio": 0.15
                },
                "key_points": [
                    "AI diagnostic tools are improving accuracy",
                    "Treatment recommendations are becoming more personalized"
                ]
            }
            
            summarize_request = {
                "text": "Long article text about AI in healthcare...",
                "summary_type": "extractive",
                "max_length": 100
            }
            
            response = client.post("/summarize", json=summarize_request, headers=auth_headers)
            assert response.status_code == 200
            
            data = response.json()
            assert "summary" in data
            assert "key_points" in data

    # WebSocket Endpoints
    def test_websocket_connection(self, client, auth_headers, mock_auth):
        """Test WebSocket connection."""
        with client.websocket_connect("/ws?token=test_api_key_comprehensive") as websocket:
            # Send subscription message
            websocket.send_json({
                "type": "subscribe",
                "channel": "job_updates",
                "user_id": "comprehensive_test_user"
            })
            
            # Should receive confirmation
            data = websocket.receive_json()
            assert data["type"] == "subscription_confirmed"
            assert data["channel"] == "job_updates"

    # Error Handling Tests
    def test_authentication_errors(self, client):
        """Test authentication error handling."""
        # No auth header
        response = client.get("/content/articles")
        assert response.status_code == 401
        
        # Invalid auth header
        headers = {"Authorization": "Bearer invalid_token"}
        response = client.get("/content/articles", headers=headers)
        assert response.status_code == 401
        
        # Malformed auth header
        headers = {"Authorization": "InvalidFormat"}
        response = client.get("/content/articles", headers=headers)
        assert response.status_code == 401
    
    def test_rate_limiting_errors(self, client, mock_auth):
        """Test rate limiting error handling."""
        with patch('src.shared.security.rate_limiter.RateLimiter.is_allowed', return_value=False):
            headers = {"Authorization": "Bearer test_token"}
            response = client.get("/content/articles", headers=headers)
            assert response.status_code == 429
            
            data = response.json()
            assert "rate_limit" in data["error"]["code"].lower()
    
    def test_validation_errors(self, client, auth_headers, mock_auth):
        """Test request validation errors."""
        # Invalid JSON
        response = client.post("/search", data="invalid json", headers=auth_headers)
        assert response.status_code == 422
        
        # Missing required fields
        response = client.post("/search", json={}, headers=auth_headers)
        assert response.status_code == 422
        
        # Invalid field values
        invalid_request = {"query": "test", "limit": -1}  # Negative limit
        response = client.post("/search", json=invalid_request, headers=auth_headers)
        assert response.status_code == 422
    
    def test_server_error_handling(self, client, auth_headers, mock_auth):
        """Test server error handling."""
        with patch('src.shared.database.repositories.ArticleRepository.get_articles', 
                   side_effect=Exception("Database error")):
            response = client.get("/content/articles", headers=auth_headers)
            assert response.status_code == 500
            
            data = response.json()
            assert "error" in data
            assert "internal_server_error" in data["error"]["code"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])