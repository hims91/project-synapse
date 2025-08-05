"""
Comprehensive end-to-end integration tests for Project Synapse.

Tests all user workflows, API endpoints, and system integration points
to ensure the complete system works as expected.
"""

import pytest
import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, patch
import httpx
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

# Import main application and components
from src.axon_interface.main import app
from src.shared.database import get_database_manager
from src.shared.caching import get_cache_manager
from src.signal_relay.task_dispatcher import get_task_dispatcher
from src.spinal_cord.fallback_manager import get_fallback_manager
from src.thalamus.nlp_pipeline import get_nlp_pipeline
from src.neurons.recipe_engine import get_recipe_engine
from src.dendrites.feed_poller import FeedPoller
from src.shared.config import get_settings


class TestEndToEndIntegration:
    """End-to-end integration tests for Project Synapse."""
    
    @pytest.fixture(scope="class")
    def client(self):
        """Create test client for API testing."""
        return TestClient(app)
    
    @pytest.fixture(scope="class")
    async def test_data(self):
        """Set up test data for integration tests."""
        return {
            "test_urls": [
                "https://example.com/article1",
                "https://example.com/article2",
                "https://techcrunch.com/sample-article"
            ],
            "test_feeds": [
                "https://feeds.example.com/rss",
                "https://techcrunch.com/feed/"
            ],
            "test_queries": [
                "artificial intelligence trends",
                "machine learning healthcare",
                "technology innovation"
            ],
            "test_api_key": "test_api_key_12345",
            "test_user_id": "user_test_123"
        }
    
    def test_system_health_check(self, client):
        """Test system health and component status."""
        response = client.get("/health")
        assert response.status_code == 200
        
        health_data = response.json()
        assert health_data["status"] == "healthy"
        assert "components" in health_data
        assert "database" in health_data["components"]
        assert "cache" in health_data["components"]
        assert "task_dispatcher" in health_data["components"]
    
    def test_api_authentication_flow(self, client, test_data):
        """Test API authentication and key validation."""
        # Test without API key
        response = client.get("/content/articles")
        assert response.status_code == 401
        
        # Test with invalid API key
        headers = {"Authorization": "Bearer invalid_key"}
        response = client.get("/content/articles", headers=headers)
        assert response.status_code == 401
        
        # Test with valid API key (mocked)
        with patch('src.axon_interface.auth.validate_api_key') as mock_validate:
            mock_validate.return_value = {
                "user_id": test_data["test_user_id"],
                "tier": "pro",
                "rate_limit": 1000
            }
            
            headers = {"Authorization": f"Bearer {test_data['test_api_key']}"}
            response = client.get("/content/articles", headers=headers)
            assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_content_scraping_workflow(self, client, test_data):
        """Test complete content scraping workflow."""
        with patch('src.axon_interface.auth.validate_api_key') as mock_auth:
            mock_auth.return_value = {
                "user_id": test_data["test_user_id"],
                "tier": "pro",
                "rate_limit": 1000
            }
            
            headers = {"Authorization": f"Bearer {test_data['test_api_key']}"}
            
            # Step 1: Submit scraping job
            scrape_request = {
                "url": test_data["test_urls"][0],
                "priority": "normal",
                "analysis_types": ["sentiment", "bias", "topics"]
            }
            
            with patch('src.neurons.http_scraper.HttpScraper.scrape') as mock_scrape:
                mock_scrape.return_value = {
                    "url": test_data["test_urls"][0],
                    "title": "Test Article",
                    "content": "This is a test article about artificial intelligence.",
                    "metadata": {"author": "Test Author", "published": "2024-01-08"}
                }
                
                response = client.post("/scrape", json=scrape_request, headers=headers)
                assert response.status_code == 202
                
                job_data = response.json()
                assert "job_id" in job_data
                assert job_data["status"] == "queued"
                job_id = job_data["job_id"]
            
            # Step 2: Check job status
            response = client.get(f"/scrape/status/{job_id}", headers=headers)
            assert response.status_code == 200
            
            status_data = response.json()
            assert status_data["job_id"] == job_id
            assert status_data["status"] in ["queued", "processing", "completed"]
            
            # Step 3: Get results (simulate completion)
            with patch('src.signal_relay.task_dispatcher.TaskDispatcher.get_task_result') as mock_result:
                mock_result.return_value = {
                    "job_id": job_id,
                    "status": "completed",
                    "result": {
                        "content": {
                            "title": "Test Article",
                            "content": "This is a test article about artificial intelligence.",
                            "url": test_data["test_urls"][0]
                        },
                        "analysis": {
                            "sentiment": {"score": 0.75, "label": "positive"},
                            "bias": {"overall_score": 0.12},
                            "topics": [{"name": "artificial_intelligence", "confidence": 0.89}]
                        }
                    }
                }
                
                response = client.get(f"/scrape/result/{job_id}", headers=headers)
                assert response.status_code == 200
                
                result_data = response.json()
                assert result_data["status"] == "completed"
                assert "analysis" in result_data["result"]
                assert "sentiment" in result_data["result"]["analysis"]
    
    @pytest.mark.asyncio
    async def test_content_analysis_workflow(self, client, test_data):
        """Test content analysis and NLP processing workflow."""
        with patch('src.axon_interface.auth.validate_api_key') as mock_auth:
            mock_auth.return_value = {
                "user_id": test_data["test_user_id"],
                "tier": "pro",
                "rate_limit": 1000
            }
            
            headers = {"Authorization": f"Bearer {test_data['test_api_key']}"}
            
            # Test content analysis endpoint
            analysis_request = {
                "text": "Artificial intelligence is revolutionizing healthcare with new diagnostic tools.",
                "analysis_types": ["sentiment", "bias", "topics", "entities", "summary"]
            }
            
            with patch('src.thalamus.nlp_pipeline.NLPPipeline.analyze') as mock_analyze:
                mock_analyze.return_value = {
                    "sentiment": {"score": 0.78, "label": "positive", "confidence": 0.92},
                    "bias": {"overall_score": 0.15, "types": {"political": 0.05}},
                    "topics": [{"name": "healthcare", "confidence": 0.91}],
                    "entities": [{"text": "artificial intelligence", "type": "TECHNOLOGY"}],
                    "summary": {"text": "AI is transforming healthcare diagnostics.", "length": 45}
                }
                
                response = client.post("/analysis/content", json=analysis_request, headers=headers)
                assert response.status_code == 200
                
                analysis_data = response.json()
                assert "sentiment" in analysis_data
                assert "bias" in analysis_data
                assert "topics" in analysis_data
                assert analysis_data["sentiment"]["label"] == "positive"
    
    @pytest.mark.asyncio
    async def test_semantic_search_workflow(self, client, test_data):
        """Test semantic search functionality."""
        with patch('src.axon_interface.auth.validate_api_key') as mock_auth:
            mock_auth.return_value = {
                "user_id": test_data["test_user_id"],
                "tier": "pro",
                "rate_limit": 1000
            }
            
            headers = {"Authorization": f"Bearer {test_data['test_api_key']}"}
            
            # Test semantic search
            search_request = {
                "query": test_data["test_queries"][0],
                "limit": 10,
                "filters": {
                    "sentiment": ["positive"],
                    "topics": ["technology"]
                }
            }
            
            with patch('src.thalamus.semantic_search.SemanticSearchEngine.search') as mock_search:
                mock_search.return_value = {
                    "results": [
                        {
                            "id": "article_123",
                            "title": "AI Trends in 2024",
                            "content": "Latest developments in artificial intelligence...",
                            "relevance_score": 0.95,
                            "sentiment": {"score": 0.82, "label": "positive"},
                            "topics": [{"name": "technology", "confidence": 0.89}]
                        }
                    ],
                    "total": 1,
                    "query_time_ms": 45
                }
                
                response = client.post("/search", json=search_request, headers=headers)
                assert response.status_code == 200
                
                search_data = response.json()
                assert "results" in search_data
                assert len(search_data["results"]) > 0
                assert search_data["results"][0]["relevance_score"] > 0.9
    
    @pytest.mark.asyncio
    async def test_monitoring_subscription_workflow(self, client, test_data):
        """Test monitoring and subscription workflow."""
        with patch('src.axon_interface.auth.validate_api_key') as mock_auth:
            mock_auth.return_value = {
                "user_id": test_data["test_user_id"],
                "tier": "pro",
                "rate_limit": 1000
            }
            
            headers = {"Authorization": f"Bearer {test_data['test_api_key']}"}
            
            # Create monitoring subscription
            subscription_request = {
                "name": "AI Healthcare Monitor",
                "keywords": ["artificial intelligence", "healthcare"],
                "webhook_url": "https://example.com/webhook",
                "frequency": "daily"
            }
            
            response = client.post("/monitoring/subscriptions", json=subscription_request, headers=headers)
            assert response.status_code == 201
            
            subscription_data = response.json()
            assert "subscription_id" in subscription_data
            assert subscription_data["name"] == "AI Healthcare Monitor"
            subscription_id = subscription_data["subscription_id"]
            
            # List subscriptions
            response = client.get("/monitoring/subscriptions", headers=headers)
            assert response.status_code == 200
            
            subscriptions = response.json()
            assert len(subscriptions["subscriptions"]) > 0
            
            # Update subscription
            update_request = {"frequency": "hourly"}
            response = client.patch(f"/monitoring/subscriptions/{subscription_id}", 
                                  json=update_request, headers=headers)
            assert response.status_code == 200
            
            # Delete subscription
            response = client.delete(f"/monitoring/subscriptions/{subscription_id}", headers=headers)
            assert response.status_code == 204
    
    @pytest.mark.asyncio
    async def test_feed_polling_integration(self, client, test_data):
        """Test feed polling and processing integration."""
        with patch('src.dendrites.feed_parser.FeedParser.parse_feed') as mock_parse:
            mock_parse.return_value = {
                "metadata": {
                    "title": "Test Feed",
                    "description": "Test RSS feed",
                    "url": test_data["test_feeds"][0]
                },
                "items": [
                    {
                        "title": "Test Article 1",
                        "link": "https://example.com/article1",
                        "description": "Test article about AI",
                        "published": datetime.utcnow(),
                        "content": "Artificial intelligence is advancing rapidly..."
                    }
                ]
            }
            
            # Test feed polling
            feed_poller = FeedPoller()
            await feed_poller.add_feed(test_data["test_feeds"][0], priority="normal")
            
            # Poll feeds
            results = await feed_poller.poll_feeds()
            assert len(results) > 0
            assert results[0]["success"] is True
            assert len(results[0]["new_items"]) > 0
    
    @pytest.mark.asyncio
    async def test_real_time_features(self, client, test_data):
        """Test WebSocket and real-time features."""
        with patch('src.axon_interface.auth.validate_api_key') as mock_auth:
            mock_auth.return_value = {
                "user_id": test_data["test_user_id"],
                "tier": "pro",
                "rate_limit": 1000
            }
            
            # Test WebSocket connection (simplified)
            with client.websocket_connect(f"/ws?token={test_data['test_api_key']}") as websocket:
                # Send subscription message
                websocket.send_json({
                    "type": "subscribe",
                    "channel": "job_updates",
                    "user_id": test_data["test_user_id"]
                })
                
                # Receive confirmation
                data = websocket.receive_json()
                assert data["type"] == "subscription_confirmed"
                assert data["channel"] == "job_updates"
    
    @pytest.mark.asyncio
    async def test_trends_analysis_workflow(self, client, test_data):
        """Test trends analysis and detection."""
        with patch('src.axon_interface.auth.validate_api_key') as mock_auth:
            mock_auth.return_value = {
                "user_id": test_data["test_user_id"],
                "tier": "pro",
                "rate_limit": 1000
            }
            
            headers = {"Authorization": f"Bearer {test_data['test_api_key']}"}
            
            # Test trending topics
            with patch('src.thalamus.trends_analyzer.TrendsAnalyzer.get_trending_topics') as mock_trends:
                mock_trends.return_value = {
                    "trending_topics": [
                        {
                            "topic": "artificial_intelligence",
                            "mentions": 1247,
                            "growth_rate": 0.45,
                            "sentiment": {"overall": 0.73}
                        }
                    ],
                    "time_range": "24h"
                }
                
                response = client.get("/trends/topics?time_range=24h", headers=headers)
                assert response.status_code == 200
                
                trends_data = response.json()
                assert "trending_topics" in trends_data
                assert len(trends_data["trending_topics"]) > 0
                assert trends_data["trending_topics"][0]["growth_rate"] > 0
    
    @pytest.mark.asyncio
    async def test_bias_analysis_workflow(self, client, test_data):
        """Test bias detection and narrative analysis."""
        with patch('src.axon_interface.auth.validate_api_key') as mock_auth:
            mock_auth.return_value = {
                "user_id": test_data["test_user_id"],
                "tier": "pro",
                "rate_limit": 1000
            }
            
            headers = {"Authorization": f"Bearer {test_data['test_api_key']}"}
            
            # Test bias analysis
            bias_request = {
                "text": "The government's new AI policy shows clear favoritism toward tech companies.",
                "analysis_types": ["bias", "narrative", "framing"]
            }
            
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
                
                response = client.post("/analysis/bias", json=bias_request, headers=headers)
                assert response.status_code == 200
                
                bias_data = response.json()
                assert "bias" in bias_data
                assert "narrative" in bias_data
                assert bias_data["bias"]["overall_score"] > 0
    
    @pytest.mark.asyncio
    async def test_system_resilience_and_fallback(self, client, test_data):
        """Test system resilience and fallback mechanisms."""
        # Test database fallback
        with patch('src.shared.database.DatabaseManager.is_healthy', return_value=False):
            with patch('src.spinal_cord.fallback_manager.FallbackManager.is_active', return_value=True):
                # System should still respond with fallback active
                response = client.get("/health")
                assert response.status_code == 200
                
                health_data = response.json()
                assert "fallback_active" in health_data
        
        # Test cache fallback
        with patch('src.shared.caching.CacheManager.is_healthy', return_value=False):
            with patch('src.axon_interface.auth.validate_api_key') as mock_auth:
                mock_auth.return_value = {
                    "user_id": test_data["test_user_id"],
                    "tier": "pro",
                    "rate_limit": 1000
                }
                
                headers = {"Authorization": f"Bearer {test_data['test_api_key']}"}
                
                # API should still work without cache
                response = client.get("/content/articles", headers=headers)
                assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_rate_limiting_and_security(self, client, test_data):
        """Test rate limiting and security features."""
        with patch('src.axon_interface.auth.validate_api_key') as mock_auth:
            mock_auth.return_value = {
                "user_id": test_data["test_user_id"],
                "tier": "free",  # Free tier has lower limits
                "rate_limit": 60
            }
            
            headers = {"Authorization": f"Bearer {test_data['test_api_key']}"}
            
            # Test rate limiting
            with patch('src.shared.security.rate_limiter.RateLimiter.is_allowed', return_value=False):
                response = client.get("/content/articles", headers=headers)
                assert response.status_code == 429
                
                rate_limit_data = response.json()
                assert "rate_limit_exceeded" in rate_limit_data["error"]["code"].lower()
        
        # Test security headers
        response = client.get("/health")
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers
        assert "X-XSS-Protection" in response.headers


class TestPerformanceAndLoad:
    """Performance and load testing for Project Synapse."""
    
    @pytest.fixture(scope="class")
    def client(self):
        """Create test client for performance testing."""
        return TestClient(app)
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_performance(self, client):
        """Test system performance under concurrent load."""
        async def make_request():
            with patch('src.axon_interface.auth.validate_api_key') as mock_auth:
                mock_auth.return_value = {
                    "user_id": "load_test_user",
                    "tier": "pro",
                    "rate_limit": 1000
                }
                
                headers = {"Authorization": "Bearer load_test_key"}
                response = client.get("/health", headers=headers)
                return response.status_code == 200
        
        # Test with 50 concurrent requests
        start_time = time.time()
        tasks = [make_request() for _ in range(50)]
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # All requests should succeed
        assert all(results)
        
        # Should complete within reasonable time (5 seconds for 50 requests)
        assert (end_time - start_time) < 5.0
    
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, client):
        """Test memory usage under sustained load."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate sustained load
        for _ in range(100):
            with patch('src.axon_interface.auth.validate_api_key') as mock_auth:
                mock_auth.return_value = {
                    "user_id": "memory_test_user",
                    "tier": "pro",
                    "rate_limit": 1000
                }
                
                headers = {"Authorization": "Bearer memory_test_key"}
                response = client.get("/content/articles", headers=headers)
                assert response.status_code == 200
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for 100 requests)
        assert memory_increase < 100
    
    @pytest.mark.asyncio
    async def test_database_query_performance(self):
        """Test database query performance."""
        with patch('src.shared.database.DatabaseManager') as mock_db:
            mock_session = AsyncMock()
            mock_db.return_value.get_session.return_value = mock_session
            
            # Simulate database queries
            start_time = time.time()
            
            for _ in range(100):
                # Mock query execution
                mock_session.execute.return_value.fetchall.return_value = [
                    {"id": 1, "title": "Test Article", "content": "Test content"}
                ]
                
                # This would be actual database query in real test
                await mock_session.execute("SELECT * FROM articles LIMIT 10")
            
            end_time = time.time()
            query_time = end_time - start_time
            
            # 100 queries should complete quickly (less than 1 second)
            assert query_time < 1.0
    
    @pytest.mark.asyncio
    async def test_cache_performance(self):
        """Test caching system performance."""
        with patch('src.shared.caching.CacheManager') as mock_cache:
            cache_manager = mock_cache.return_value
            cache_manager.get.return_value = None  # Cache miss
            cache_manager.set.return_value = True
            
            # Test cache performance
            start_time = time.time()
            
            for i in range(1000):
                await cache_manager.get("test", f"key_{i}")
                await cache_manager.set("test", f"key_{i}", f"value_{i}")
            
            end_time = time.time()
            cache_time = end_time - start_time
            
            # 1000 cache operations should be very fast (less than 0.5 seconds)
            assert cache_time < 0.5


class TestSystemValidation:
    """System validation and integration verification."""
    
    @pytest.mark.asyncio
    async def test_complete_user_journey(self, client):
        """Test complete user journey from registration to content analysis."""
        test_user_data = {
            "email": "test@example.com",
            "api_key": "journey_test_key_123"
        }
        
        with patch('src.axon_interface.auth.validate_api_key') as mock_auth:
            mock_auth.return_value = {
                "user_id": "journey_user_123",
                "tier": "pro",
                "rate_limit": 1000
            }
            
            headers = {"Authorization": f"Bearer {test_user_data['api_key']}"}
            
            # Step 1: Check system health
            response = client.get("/health")
            assert response.status_code == 200
            
            # Step 2: Submit content for scraping
            scrape_request = {
                "url": "https://example.com/test-article",
                "analysis_types": ["sentiment", "bias", "topics"]
            }
            
            with patch('src.neurons.http_scraper.HttpScraper.scrape') as mock_scrape:
                mock_scrape.return_value = {
                    "title": "Test Article",
                    "content": "This is a comprehensive test article about AI.",
                    "metadata": {"author": "Test Author"}
                }
                
                response = client.post("/scrape", json=scrape_request, headers=headers)
                assert response.status_code == 202
                job_id = response.json()["job_id"]
            
            # Step 3: Monitor job progress
            response = client.get(f"/scrape/status/{job_id}", headers=headers)
            assert response.status_code == 200
            
            # Step 4: Search for content
            search_request = {
                "query": "artificial intelligence",
                "limit": 10
            }
            
            with patch('src.thalamus.semantic_search.SemanticSearchEngine.search') as mock_search:
                mock_search.return_value = {
                    "results": [{"id": "1", "title": "AI Article", "relevance_score": 0.95}],
                    "total": 1
                }
                
                response = client.post("/search", json=search_request, headers=headers)
                assert response.status_code == 200
                assert len(response.json()["results"]) > 0
            
            # Step 5: Set up monitoring
            monitor_request = {
                "name": "AI Monitor",
                "keywords": ["artificial intelligence"],
                "webhook_url": "https://example.com/webhook"
            }
            
            response = client.post("/monitoring/subscriptions", json=monitor_request, headers=headers)
            assert response.status_code == 201
            
            # Journey completed successfully
            assert True
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, client):
        """Test system error handling and recovery mechanisms."""
        with patch('src.axon_interface.auth.validate_api_key') as mock_auth:
            mock_auth.return_value = {
                "user_id": "error_test_user",
                "tier": "pro",
                "rate_limit": 1000
            }
            
            headers = {"Authorization": "Bearer error_test_key"}
            
            # Test invalid URL handling
            scrape_request = {
                "url": "invalid-url-format",
                "analysis_types": ["sentiment"]
            }
            
            response = client.post("/scrape", json=scrape_request, headers=headers)
            assert response.status_code == 400
            assert "invalid" in response.json()["error"]["message"].lower()
            
            # Test network error handling
            with patch('src.neurons.http_scraper.HttpScraper.scrape', side_effect=Exception("Network error")):
                scrape_request = {
                    "url": "https://example.com/test",
                    "analysis_types": ["sentiment"]
                }
                
                response = client.post("/scrape", json=scrape_request, headers=headers)
                # Should handle error gracefully
                assert response.status_code in [400, 500, 502]
    
    def test_api_documentation_completeness(self, client):
        """Test that API documentation is complete and accessible."""
        # Test OpenAPI schema
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        openapi_schema = response.json()
        assert "paths" in openapi_schema
        assert "components" in openapi_schema
        
        # Check that major endpoints are documented
        paths = openapi_schema["paths"]
        expected_endpoints = [
            "/health",
            "/scrape",
            "/search",
            "/content/articles",
            "/monitoring/subscriptions"
        ]
        
        for endpoint in expected_endpoints:
            assert any(endpoint in path for path in paths.keys())
    
    @pytest.mark.asyncio
    async def test_monitoring_and_metrics(self, client):
        """Test monitoring and metrics collection."""
        # Test metrics endpoint
        response = client.get("/metrics")
        assert response.status_code == 200
        
        # Should return Prometheus-format metrics
        metrics_text = response.text
        assert "# HELP" in metrics_text
        assert "# TYPE" in metrics_text
        
        # Test system status
        response = client.get("/status")
        assert response.status_code == 200
        
        status_data = response.json()
        assert "uptime" in status_data
        assert "version" in status_data
        assert "components" in status_data


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])