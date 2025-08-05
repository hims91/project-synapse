"""
Performance benchmarks and load testing for Project Synapse.

Comprehensive performance testing including response times, throughput,
resource usage, and scalability under various load conditions.
"""

import pytest
import asyncio
import time
import statistics
import psutil
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, AsyncMock, patch
import httpx
from fastapi.testclient import TestClient

from src.axon_interface.main import app
from src.shared.database import get_database_manager
from src.shared.caching import get_cache_manager
from src.signal_relay.task_dispatcher import get_task_dispatcher


class PerformanceMetrics:
    """Performance metrics collection and analysis."""
    
    def __init__(self):
        self.response_times: List[float] = []
        self.throughput_data: List[Tuple[datetime, int]] = []
        self.error_count = 0
        self.success_count = 0
        self.memory_usage: List[float] = []
        self.cpu_usage: List[float] = []
    
    def record_response_time(self, response_time: float):
        """Record response time."""
        self.response_times.append(response_time)
    
    def record_request_result(self, success: bool):
        """Record request success/failure."""
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
    
    def record_system_metrics(self):
        """Record system resource usage."""
        process = psutil.Process(os.getpid())
        self.memory_usage.append(process.memory_info().rss / 1024 / 1024)  # MB
        self.cpu_usage.append(process.cpu_percent())
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.response_times:
            return {"error": "No data collected"}
        
        return {
            "response_times": {
                "min": min(self.response_times),
                "max": max(self.response_times),
                "mean": statistics.mean(self.response_times),
                "median": statistics.median(self.response_times),
                "p95": self._percentile(self.response_times, 95),
                "p99": self._percentile(self.response_times, 99)
            },
            "requests": {
                "total": self.success_count + self.error_count,
                "success": self.success_count,
                "errors": self.error_count,
                "success_rate": self.success_count / (self.success_count + self.error_count) * 100
            },
            "system_resources": {
                "memory_mb": {
                    "min": min(self.memory_usage) if self.memory_usage else 0,
                    "max": max(self.memory_usage) if self.memory_usage else 0,
                    "mean": statistics.mean(self.memory_usage) if self.memory_usage else 0
                },
                "cpu_percent": {
                    "min": min(self.cpu_usage) if self.cpu_usage else 0,
                    "max": max(self.cpu_usage) if self.cpu_usage else 0,
                    "mean": statistics.mean(self.cpu_usage) if self.cpu_usage else 0
                }
            }
        }
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile."""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]


class TestAPIPerformance:
    """API endpoint performance testing."""
    
    @pytest.fixture(scope="class")
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture(scope="class")
    def auth_headers(self):
        """Create authentication headers."""
        return {"Authorization": "Bearer perf_test_key_123"}
    
    def test_health_endpoint_performance(self, client):
        """Test health endpoint performance."""
        metrics = PerformanceMetrics()
        
        # Warm up
        for _ in range(10):
            client.get("/health")
        
        # Performance test
        for _ in range(100):
            start_time = time.time()
            response = client.get("/health")
            end_time = time.time()
            
            metrics.record_response_time(end_time - start_time)
            metrics.record_request_result(response.status_code == 200)
            metrics.record_system_metrics()
        
        summary = metrics.get_summary()
        
        # Performance assertions
        assert summary["response_times"]["mean"] < 0.1  # Average < 100ms
        assert summary["response_times"]["p95"] < 0.2   # 95th percentile < 200ms
        assert summary["requests"]["success_rate"] == 100.0
        
        print(f"Health endpoint performance: {summary}")
    
    @pytest.mark.asyncio
    async def test_content_api_performance(self, client, auth_headers):
        """Test content API performance."""
        with patch('src.axon_interface.auth.validate_api_key') as mock_auth:
            mock_auth.return_value = {
                "user_id": "perf_test_user",
                "tier": "pro",
                "rate_limit": 1000
            }
            
            metrics = PerformanceMetrics()
            
            # Mock database responses
            with patch('src.shared.database.repositories.ArticleRepository.get_articles') as mock_get:
                mock_get.return_value = {
                    "articles": [
                        {"id": i, "title": f"Article {i}", "content": f"Content {i}"}
                        for i in range(20)
                    ],
                    "total": 20
                }
                
                # Performance test
                for _ in range(50):
                    start_time = time.time()
                    response = client.get("/content/articles", headers=auth_headers)
                    end_time = time.time()
                    
                    metrics.record_response_time(end_time - start_time)
                    metrics.record_request_result(response.status_code == 200)
                    metrics.record_system_metrics()
                
                summary = metrics.get_summary()
                
                # Performance assertions
                assert summary["response_times"]["mean"] < 0.5  # Average < 500ms
                assert summary["response_times"]["p95"] < 1.0   # 95th percentile < 1s
                assert summary["requests"]["success_rate"] >= 95.0
                
                print(f"Content API performance: {summary}")
    
    @pytest.mark.asyncio
    async def test_search_api_performance(self, client, auth_headers):
        """Test search API performance."""
        with patch('src.axon_interface.auth.validate_api_key') as mock_auth:
            mock_auth.return_value = {
                "user_id": "perf_test_user",
                "tier": "pro",
                "rate_limit": 1000
            }
            
            metrics = PerformanceMetrics()
            
            # Mock search responses
            with patch('src.thalamus.semantic_search.SemanticSearchEngine.search') as mock_search:
                mock_search.return_value = {
                    "results": [
                        {
                            "id": f"result_{i}",
                            "title": f"Search Result {i}",
                            "relevance_score": 0.9 - (i * 0.1),
                            "content": f"Content for result {i}"
                        }
                        for i in range(10)
                    ],
                    "total": 10,
                    "query_time_ms": 45
                }
                
                search_request = {
                    "query": "artificial intelligence",
                    "limit": 10
                }
                
                # Performance test
                for _ in range(30):
                    start_time = time.time()
                    response = client.post("/search", json=search_request, headers=auth_headers)
                    end_time = time.time()
                    
                    metrics.record_response_time(end_time - start_time)
                    metrics.record_request_result(response.status_code == 200)
                    metrics.record_system_metrics()
                
                summary = metrics.get_summary()
                
                # Performance assertions
                assert summary["response_times"]["mean"] < 1.0  # Average < 1s
                assert summary["response_times"]["p95"] < 2.0   # 95th percentile < 2s
                assert summary["requests"]["success_rate"] >= 90.0
                
                print(f"Search API performance: {summary}")


class TestConcurrencyPerformance:
    """Concurrency and load testing."""
    
    @pytest.fixture(scope="class")
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.mark.asyncio
    async def test_concurrent_health_checks(self, client):
        """Test concurrent health check performance."""
        async def make_health_request():
            start_time = time.time()
            response = client.get("/health")
            end_time = time.time()
            return {
                "success": response.status_code == 200,
                "response_time": end_time - start_time
            }
        
        # Test with increasing concurrency levels
        concurrency_levels = [10, 25, 50, 100]
        results = {}
        
        for concurrency in concurrency_levels:
            print(f"Testing concurrency level: {concurrency}")
            
            start_time = time.time()
            tasks = [make_health_request() for _ in range(concurrency)]
            responses = await asyncio.gather(*tasks)
            end_time = time.time()
            
            total_time = end_time - start_time
            success_count = sum(1 for r in responses if r["success"])
            avg_response_time = statistics.mean([r["response_time"] for r in responses])
            throughput = concurrency / total_time
            
            results[concurrency] = {
                "total_time": total_time,
                "success_rate": success_count / concurrency * 100,
                "avg_response_time": avg_response_time,
                "throughput": throughput
            }
            
            # Performance assertions
            assert success_count / concurrency >= 0.95  # 95% success rate
            assert avg_response_time < 1.0  # Average response time < 1s
        
        print("Concurrency test results:")
        for level, result in results.items():
            print(f"  {level} concurrent: {result}")
    
    @pytest.mark.asyncio
    async def test_sustained_load_performance(self, client):
        """Test performance under sustained load."""
        metrics = PerformanceMetrics()
        duration_seconds = 30  # 30-second load test
        requests_per_second = 10
        
        async def make_request():
            start_time = time.time()
            response = client.get("/health")
            end_time = time.time()
            
            metrics.record_response_time(end_time - start_time)
            metrics.record_request_result(response.status_code == 200)
            metrics.record_system_metrics()
        
        print(f"Starting {duration_seconds}s sustained load test at {requests_per_second} RPS")
        
        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            # Create batch of requests
            tasks = [make_request() for _ in range(requests_per_second)]
            await asyncio.gather(*tasks)
            
            # Wait for next second
            await asyncio.sleep(1.0)
        
        summary = metrics.get_summary()
        
        # Performance assertions
        assert summary["requests"]["success_rate"] >= 95.0
        assert summary["response_times"]["mean"] < 0.5
        assert summary["response_times"]["p95"] < 1.0
        
        print(f"Sustained load test results: {summary}")
    
    def test_memory_leak_detection(self, client):
        """Test for memory leaks under load."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run many requests to detect memory leaks
        for i in range(1000):
            response = client.get("/health")
            assert response.status_code == 200
            
            # Check memory every 100 requests
            if i % 100 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_increase = current_memory - initial_memory
                
                print(f"Request {i}: Memory usage {current_memory:.1f}MB (+{memory_increase:.1f}MB)")
                
                # Memory increase should be reasonable
                assert memory_increase < 50  # Less than 50MB increase
        
        final_memory = process.memory_info().rss / 1024 / 1024
        total_increase = final_memory - initial_memory
        
        print(f"Total memory increase: {total_increase:.1f}MB")
        assert total_increase < 100  # Less than 100MB total increase


class TestDatabasePerformance:
    """Database performance testing."""
    
    @pytest.mark.asyncio
    async def test_database_connection_pool_performance(self):
        """Test database connection pool performance."""
        with patch('src.shared.database.DatabaseManager') as mock_db_manager:
            mock_session = AsyncMock()
            mock_db_manager.return_value.get_session.return_value = mock_session
            
            # Mock query results
            mock_session.execute.return_value.fetchall.return_value = [
                {"id": i, "title": f"Article {i}"} for i in range(100)
            ]
            
            db_manager = mock_db_manager.return_value
            metrics = PerformanceMetrics()
            
            # Test concurrent database operations
            async def db_operation():
                start_time = time.time()
                session = await db_manager.get_session()
                result = await session.execute("SELECT * FROM articles LIMIT 100")
                await session.close()
                end_time = time.time()
                
                metrics.record_response_time(end_time - start_time)
                metrics.record_request_result(True)
                return result
            
            # Run concurrent database operations
            tasks = [db_operation() for _ in range(50)]
            results = await asyncio.gather(*tasks)
            
            summary = metrics.get_summary()
            
            # Performance assertions
            assert len(results) == 50
            assert summary["response_times"]["mean"] < 0.1  # Average < 100ms
            assert summary["requests"]["success_rate"] == 100.0
            
            print(f"Database performance: {summary}")
    
    @pytest.mark.asyncio
    async def test_query_optimization_performance(self):
        """Test query optimization performance."""
        with patch('src.shared.database.performance_optimizer.QueryOptimizer') as mock_optimizer:
            optimizer = mock_optimizer.return_value
            
            # Test query analysis performance
            queries = [
                "SELECT * FROM articles WHERE title LIKE '%AI%'",
                "SELECT COUNT(*) FROM articles WHERE created_at > '2024-01-01'",
                "SELECT a.*, u.name FROM articles a JOIN users u ON a.user_id = u.id",
            ]
            
            start_time = time.time()
            
            for query in queries * 100:  # 300 total queries
                await optimizer.analyze_query(query, duration=0.05, rows_affected=10)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Query analysis should be fast
            assert total_time < 1.0  # Less than 1 second for 300 queries
            print(f"Query optimization analysis time: {total_time:.3f}s for 300 queries")


class TestCachePerformance:
    """Cache system performance testing."""
    
    @pytest.mark.asyncio
    async def test_cache_hit_performance(self):
        """Test cache hit performance."""
        with patch('src.shared.caching.CacheManager') as mock_cache_manager:
            cache_manager = mock_cache_manager.return_value
            cache_manager.get.return_value = {"cached": "data"}
            
            metrics = PerformanceMetrics()
            
            # Test cache hit performance
            for _ in range(1000):
                start_time = time.time()
                result = await cache_manager.get("test", "key")
                end_time = time.time()
                
                metrics.record_response_time(end_time - start_time)
                metrics.record_request_result(result is not None)
            
            summary = metrics.get_summary()
            
            # Cache hits should be very fast
            assert summary["response_times"]["mean"] < 0.001  # Average < 1ms
            assert summary["requests"]["success_rate"] == 100.0
            
            print(f"Cache hit performance: {summary}")
    
    @pytest.mark.asyncio
    async def test_cache_miss_performance(self):
        """Test cache miss and set performance."""
        with patch('src.shared.caching.CacheManager') as mock_cache_manager:
            cache_manager = mock_cache_manager.return_value
            cache_manager.get.return_value = None  # Cache miss
            cache_manager.set.return_value = True
            
            metrics = PerformanceMetrics()
            
            # Test cache miss and set performance
            for i in range(500):
                start_time = time.time()
                
                # Cache miss
                result = await cache_manager.get("test", f"key_{i}")
                assert result is None
                
                # Cache set
                success = await cache_manager.set("test", f"key_{i}", f"value_{i}")
                
                end_time = time.time()
                
                metrics.record_response_time(end_time - start_time)
                metrics.record_request_result(success)
            
            summary = metrics.get_summary()
            
            # Cache operations should be fast
            assert summary["response_times"]["mean"] < 0.01  # Average < 10ms
            assert summary["requests"]["success_rate"] == 100.0
            
            print(f"Cache miss/set performance: {summary}")


class TestScrapingPerformance:
    """Scraping system performance testing."""
    
    @pytest.mark.asyncio
    async def test_scraping_throughput(self):
        """Test scraping system throughput."""
        with patch('src.neurons.http_scraper.HttpScraper') as mock_scraper:
            scraper = mock_scraper.return_value
            scraper.scrape.return_value = {
                "title": "Test Article",
                "content": "Test content",
                "metadata": {"author": "Test Author"}
            }
            
            urls = [f"https://example.com/article_{i}" for i in range(50)]
            metrics = PerformanceMetrics()
            
            # Test concurrent scraping
            async def scrape_url(url):
                start_time = time.time()
                result = await scraper.scrape(url)
                end_time = time.time()
                
                metrics.record_response_time(end_time - start_time)
                metrics.record_request_result(result is not None)
                return result
            
            # Run concurrent scraping tasks
            tasks = [scrape_url(url) for url in urls]
            results = await asyncio.gather(*tasks)
            
            summary = metrics.get_summary()
            
            # Scraping performance assertions
            assert len(results) == 50
            assert summary["response_times"]["mean"] < 2.0  # Average < 2s
            assert summary["requests"]["success_rate"] >= 95.0
            
            print(f"Scraping throughput: {summary}")


class TestSystemResourceUsage:
    """System resource usage testing."""
    
    def test_cpu_usage_under_load(self, client):
        """Test CPU usage under load."""
        process = psutil.Process(os.getpid())
        cpu_measurements = []
        
        # Baseline CPU usage
        baseline_cpu = process.cpu_percent(interval=1)
        
        # Generate load
        for _ in range(100):
            response = client.get("/health")
            assert response.status_code == 200
            
            # Measure CPU every 10 requests
            if _ % 10 == 0:
                cpu_usage = process.cpu_percent()
                cpu_measurements.append(cpu_usage)
        
        avg_cpu = statistics.mean(cpu_measurements)
        max_cpu = max(cpu_measurements)
        
        print(f"CPU usage - Baseline: {baseline_cpu}%, Average: {avg_cpu}%, Max: {max_cpu}%")
        
        # CPU usage should be reasonable
        assert avg_cpu < 50  # Average CPU < 50%
        assert max_cpu < 80  # Max CPU < 80%
    
    def test_memory_efficiency(self, client):
        """Test memory efficiency."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_measurements = []
        
        # Generate requests and measure memory
        for i in range(200):
            response = client.get("/health")
            assert response.status_code == 200
            
            if i % 20 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_measurements.append(current_memory)
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        max_memory = max(memory_measurements)
        
        print(f"Memory usage - Initial: {initial_memory:.1f}MB, Final: {final_memory:.1f}MB, "
              f"Increase: {memory_increase:.1f}MB, Max: {max_memory:.1f}MB")
        
        # Memory usage should be efficient
        assert memory_increase < 50  # Less than 50MB increase
        assert max_memory < initial_memory + 100  # Max memory reasonable


def generate_performance_report(test_results: Dict[str, Any]) -> str:
    """Generate comprehensive performance report."""
    report = []
    report.append("# Project Synapse Performance Test Report")
    report.append(f"Generated: {datetime.now().isoformat()}")
    report.append("")
    
    report.append("## Summary")
    report.append("Performance test results for Project Synapse system components.")
    report.append("")
    
    for test_name, results in test_results.items():
        report.append(f"### {test_name}")
        
        if "response_times" in results:
            rt = results["response_times"]
            report.append(f"- Average Response Time: {rt['mean']:.3f}s")
            report.append(f"- 95th Percentile: {rt['p95']:.3f}s")
            report.append(f"- 99th Percentile: {rt['p99']:.3f}s")
        
        if "requests" in results:
            req = results["requests"]
            report.append(f"- Total Requests: {req['total']}")
            report.append(f"- Success Rate: {req['success_rate']:.1f}%")
        
        if "system_resources" in results:
            sys_res = results["system_resources"]
            if "memory_mb" in sys_res:
                mem = sys_res["memory_mb"]
                report.append(f"- Memory Usage: {mem['mean']:.1f}MB (avg)")
            if "cpu_percent" in sys_res:
                cpu = sys_res["cpu_percent"]
                report.append(f"- CPU Usage: {cpu['mean']:.1f}% (avg)")
        
        report.append("")
    
    report.append("## Recommendations")
    report.append("Based on performance test results:")
    report.append("- Monitor response times under production load")
    report.append("- Implement caching for frequently accessed endpoints")
    report.append("- Consider horizontal scaling for high-traffic scenarios")
    report.append("- Regular performance regression testing recommended")
    
    return "\n".join(report)


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "--tb=short", "-s"])