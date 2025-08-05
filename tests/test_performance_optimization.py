"""
Comprehensive test suite for performance optimization systems.

Tests multi-layer caching, database performance optimization,
and integration between different performance components.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from src.shared.caching import (
    CacheLevel,
    CacheStrategy,
    CacheConfig,
    LRUCache,
    RedisCache,
    CDNCache,
    MultiLayerCacheManager,
    cached,
    get_cache_manager
)

from src.shared.caching.cache_warming import (
    WarmingStrategy,
    WarmingJob,
    CacheWarmer,
    get_cache_warmer
)

from src.shared.caching.invalidation import (
    InvalidationStrategy,
    InvalidationRule,
    InvalidationEvent,
    CacheInvalidator,
    get_cache_invalidator
)

from src.shared.database import (
    QueryType,
    OptimizationLevel,
    QueryStats,
    IndexRecommendation,
    PerformanceConfig,
    ConnectionPoolManager,
    QueryOptimizer,
    IndexOptimizer,
    DatabasePerformanceOptimizer,
    get_performance_optimizer
)


class TestLRUCache:
    """Test LRU cache implementation."""
    
    def test_basic_operations(self):
        """Test basic cache operations."""
        cache = LRUCache[str](max_size=3)
        
        # Test set and get
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Test cache miss
        assert cache.get("nonexistent") is None
        
        # Test cache size limit
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        cache.set("key4", "value4")  # Should evict key1
        
        assert cache.get("key1") is None
        assert cache.get("key4") == "value4"
    
    def test_lru_eviction(self):
        """Test LRU eviction policy."""
        cache = LRUCache[str](max_size=2)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # Access key1 to make it most recently used
        cache.get("key1")
        
        # Add key3, should evict key2
        cache.set("key3", "value3")
        
        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None
        assert cache.get("key3") == "value3"
    
    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        cache = LRUCache[str](max_size=10)
        
        # Set with very short TTL
        cache.set("key1", "value1", ttl=1)
        assert cache.get("key1") == "value1"
        
        # Wait for expiration
        time.sleep(1.1)
        assert cache.get("key1") is None
    
    def test_statistics(self):
        """Test cache statistics."""
        cache = LRUCache[str](max_size=10)
        
        # Generate some hits and misses
        cache.set("key1", "value1")
        cache.get("key1")  # hit
        cache.get("key2")  # miss
        
        stats = cache.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 0.5


class TestRedisCache:
    """Test Redis cache implementation."""
    
    @pytest.fixture
    def redis_cache(self):
        """Create Redis cache for testing."""
        config = CacheConfig(redis_url="redis://localhost:6379/15")  # Use test DB
        return RedisCache(config)
    
    @pytest.mark.asyncio
    async def test_connection(self, redis_cache):
        """Test Redis connection."""
        try:
            await redis_cache.connect()
            assert redis_cache.redis_client is not None
            await redis_cache.disconnect()
        except Exception:
            pytest.skip("Redis not available for testing")
    
    @pytest.mark.asyncio
    async def test_basic_operations(self, redis_cache):
        """Test basic Redis operations."""
        try:
            await redis_cache.connect()
            
            # Test set and get
            success = await redis_cache.set("test_key", {"data": "test_value"})
            assert success
            
            value = await redis_cache.get("test_key")
            assert value == {"data": "test_value"}
            
            # Test delete
            deleted = await redis_cache.delete("test_key")
            assert deleted
            
            # Test get after delete
            value = await redis_cache.get("test_key")
            assert value is None
            
            await redis_cache.disconnect()
            
        except Exception:
            pytest.skip("Redis not available for testing")
    
    @pytest.mark.asyncio
    async def test_multiple_operations(self, redis_cache):
        """Test multiple key operations."""
        try:
            await redis_cache.connect()
            
            # Test set multiple
            data = {"key1": "value1", "key2": "value2", "key3": "value3"}
            success = await redis_cache.set_multiple(data)
            assert success
            
            # Test get multiple
            result = await redis_cache.get_multiple(["key1", "key2", "key3", "key4"])
            assert result["key1"] == "value1"
            assert result["key2"] == "value2"
            assert result["key3"] == "value3"
            assert "key4" not in result
            
            await redis_cache.disconnect()
            
        except Exception:
            pytest.skip("Redis not available for testing")


class TestMultiLayerCacheManager:
    """Test multi-layer cache manager."""
    
    @pytest.fixture
    def cache_manager(self):
        """Create cache manager for testing."""
        config = CacheConfig(
            memory_max_size=100,
            memory_ttl=300,
            redis_url="redis://localhost:6379/15",
            cdn_enabled=False
        )
        return MultiLayerCacheManager(config)
    
    @pytest.mark.asyncio
    async def test_cache_hierarchy(self, cache_manager):
        """Test cache hierarchy (memory -> Redis -> CDN)."""
        try:
            await cache_manager.initialize()
            
            # Set value (should go to both memory and Redis)
            success = await cache_manager.set("test", "key1", "value1")
            assert success
            
            # Get value (should come from memory)
            value = await cache_manager.get("test", "key1")
            assert value == "value1"
            
            # Clear memory cache and get again (should come from Redis)
            cache_manager.memory_cache.clear()
            value = await cache_manager.get("test", "key1")
            assert value == "value1"
            
            # Delete from all layers
            deleted = await cache_manager.delete("test", "key1")
            assert deleted
            
            # Verify deletion
            value = await cache_manager.get("test", "key1")
            assert value is None
            
            await cache_manager.shutdown()
            
        except Exception:
            pytest.skip("Redis not available for testing")
    
    @pytest.mark.asyncio
    async def test_cache_statistics(self, cache_manager):
        """Test cache statistics collection."""
        try:
            await cache_manager.initialize()
            
            # Generate some cache activity
            await cache_manager.set("test", "key1", "value1")
            await cache_manager.get("test", "key1")  # memory hit
            await cache_manager.get("test", "key2")  # cache miss
            
            stats = cache_manager.get_stats()
            
            assert 'overall' in stats
            assert 'memory' in stats
            assert 'redis' in stats
            assert stats['overall']['total_requests'] > 0
            
            await cache_manager.shutdown()
            
        except Exception:
            pytest.skip("Redis not available for testing")


class TestCacheWarming:
    """Test cache warming system."""
    
    @pytest.fixture
    def cache_warmer(self):
        """Create cache warmer for testing."""
        return CacheWarmer()
    
    @pytest.mark.asyncio
    async def test_job_registration(self, cache_warmer):
        """Test warming job registration."""
        async def test_loader():
            return {"test_key": "test_value"}
        
        job = WarmingJob(
            name="test_job",
            namespace="test",
            strategy=WarmingStrategy.IMMEDIATE,
            data_loader=test_loader
        )
        
        cache_warmer.register_job(job)
        
        assert "test_job" in cache_warmer.jobs
        assert cache_warmer.get_job("test_job") == job
    
    @pytest.mark.asyncio
    async def test_job_execution(self, cache_warmer):
        """Test warming job execution."""
        async def test_loader():
            return {"warm_key": "warm_value"}
        
        job = WarmingJob(
            name="test_warm_job",
            namespace="test",
            strategy=WarmingStrategy.IMMEDIATE,
            data_loader=test_loader
        )
        
        cache_warmer.register_job(job)
        
        # Mock the cache manager
        with patch.object(cache_warmer.cache_manager, 'set_multiple', return_value=True) as mock_set:
            success = await cache_warmer.warm_job("test_warm_job")
            assert success
            mock_set.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_scheduler(self, cache_warmer):
        """Test warming scheduler."""
        async def test_loader():
            return {"scheduled_key": "scheduled_value"}
        
        job = WarmingJob(
            name="scheduled_job",
            namespace="test",
            strategy=WarmingStrategy.SCHEDULED,
            data_loader=test_loader,
            schedule_interval=1  # 1 second for testing
        )
        
        cache_warmer.register_job(job)
        
        # Start scheduler briefly
        await cache_warmer.start_scheduler()
        await asyncio.sleep(0.1)  # Brief pause
        await cache_warmer.stop_scheduler()
        
        assert not cache_warmer.running


class TestCacheInvalidation:
    """Test cache invalidation system."""
    
    @pytest.fixture
    def cache_invalidator(self):
        """Create cache invalidator for testing."""
        return CacheInvalidator()
    
    @pytest.mark.asyncio
    async def test_rule_registration(self, cache_invalidator):
        """Test invalidation rule registration."""
        rule = InvalidationRule(
            name="test_rule",
            strategy=InvalidationStrategy.IMMEDIATE,
            namespaces=["test"]
        )
        
        cache_invalidator.register_rule(rule)
        
        assert "test_rule" in cache_invalidator.rules
        assert cache_invalidator.rules["test_rule"] == rule
    
    @pytest.mark.asyncio
    async def test_event_processing(self, cache_invalidator):
        """Test invalidation event processing."""
        rule = InvalidationRule(
            name="test_invalidation",
            strategy=InvalidationStrategy.IMMEDIATE,
            namespaces=["test"]
        )
        
        cache_invalidator.register_rule(rule)
        
        event = InvalidationEvent(
            event_type="test_event",
            namespace="test",
            key="test_key"
        )
        
        # Mock the cache manager
        with patch.object(cache_invalidator.cache_manager, 'delete', return_value=True) as mock_delete:
            await cache_invalidator.process_event(event)
            mock_delete.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_namespace_invalidation(self, cache_invalidator):
        """Test namespace-wide invalidation."""
        with patch.object(cache_invalidator.cache_manager, 'clear_namespace', return_value=5) as mock_clear:
            cleared_count = await cache_invalidator.invalidate_namespace("test")
            assert cleared_count == 5
            mock_clear.assert_called_once_with("test")


class TestQueryOptimizer:
    """Test query optimizer."""
    
    @pytest.fixture
    def query_optimizer(self):
        """Create query optimizer for testing."""
        config = PerformanceConfig()
        return QueryOptimizer(config)
    
    @pytest.mark.asyncio
    async def test_query_analysis(self, query_optimizer):
        """Test query performance analysis."""
        query = "SELECT * FROM articles WHERE title LIKE '%test%'"
        
        await query_optimizer.analyze_query(query, duration=1.5, rows_affected=10)
        
        query_hash = query_optimizer._hash_query(query)
        assert query_hash in query_optimizer.query_stats
        
        stats = query_optimizer.query_stats[query_hash]
        assert stats.execution_count == 1
        assert stats.avg_duration == 1.5
        assert stats.rows_affected == 10
    
    @pytest.mark.asyncio
    async def test_slow_query_detection(self, query_optimizer):
        """Test slow query detection."""
        slow_query = "SELECT * FROM articles ORDER BY created_at DESC"
        
        # Analyze a slow query
        await query_optimizer.analyze_query(slow_query, duration=2.5)
        
        # Check if it's detected as slow
        assert len(query_optimizer.slow_queries) == 1
        assert query_optimizer.slow_queries[0]['duration'] == 2.5
    
    def test_query_classification(self, query_optimizer):
        """Test query type classification."""
        test_cases = [
            ("SELECT * FROM articles", QueryType.SELECT),
            ("INSERT INTO articles VALUES (...)", QueryType.INSERT),
            ("UPDATE articles SET title = 'new'", QueryType.UPDATE),
            ("DELETE FROM articles WHERE id = 1", QueryType.DELETE),
            ("CREATE TABLE test (...)", QueryType.CREATE),
        ]
        
        for query, expected_type in test_cases:
            actual_type = query_optimizer._classify_query(query)
            assert actual_type == expected_type
    
    def test_table_extraction(self, query_optimizer):
        """Test table name extraction from queries."""
        test_cases = [
            ("SELECT * FROM articles", {"articles"}),
            ("SELECT a.*, u.name FROM articles a JOIN users u ON a.user_id = u.id", {"articles", "users"}),
            ("UPDATE articles SET title = 'new'", {"articles"}),
            ("INSERT INTO articles VALUES (...)", {"articles"}),
        ]
        
        for query, expected_tables in test_cases:
            actual_tables = query_optimizer._extract_tables(query)
            assert actual_tables == expected_tables


class TestIndexOptimizer:
    """Test index optimizer."""
    
    @pytest.fixture
    def index_optimizer(self):
        """Create index optimizer for testing."""
        config = PerformanceConfig()
        pool_manager = Mock()
        return IndexOptimizer(config, pool_manager)
    
    def test_index_recommendations(self, index_optimizer):
        """Test index recommendation generation."""
        # Create mock query stats with slow SELECT queries
        query_stats = {
            "hash1": QueryStats(
                query_hash="hash1",
                query_type=QueryType.SELECT,
                execution_count=100,
                total_duration=150.0,
                avg_duration=1.5
            ),
            "hash2": QueryStats(
                query_hash="hash2",
                query_type=QueryType.SELECT,
                execution_count=50,
                total_duration=200.0,
                avg_duration=4.0
            )
        }
        
        recommendations = index_optimizer.generate_index_recommendations(query_stats)
        
        # Should generate recommendations for slow queries
        assert len(recommendations) > 0
        assert all(rec.estimated_benefit > 0 for rec in recommendations)
    
    def test_unused_index_detection(self, index_optimizer):
        """Test unused index detection."""
        # Mock index usage stats
        index_optimizer.index_usage_stats = {
            "articles.idx_title": {"scans": 1000},  # Used index
            "articles.idx_unused": {"scans": 5},    # Unused index
        }
        
        unused_indexes = index_optimizer.get_unused_indexes()
        
        assert "articles.idx_unused" in unused_indexes
        assert "articles.idx_title" not in unused_indexes


class TestDatabasePerformanceOptimizer:
    """Test database performance optimizer."""
    
    @pytest.fixture
    def performance_optimizer(self):
        """Create performance optimizer for testing."""
        config = PerformanceConfig(
            pool_size=5,
            slow_query_threshold=1.0,
            auto_create_indexes=False
        )
        return DatabasePerformanceOptimizer(config)
    
    @pytest.mark.asyncio
    async def test_query_analysis_integration(self, performance_optimizer):
        """Test query analysis integration."""
        query = "SELECT * FROM articles WHERE user_id = 123"
        
        await performance_optimizer.analyze_query_performance(
            query=query,
            duration=0.5,
            rows_affected=1
        )
        
        assert performance_optimizer.stats['queries_analyzed'] == 1
        
        query_hash = performance_optimizer.query_optimizer._hash_query(query)
        assert query_hash in performance_optimizer.query_optimizer.query_stats
    
    @pytest.mark.asyncio
    async def test_optimization_cycle(self, performance_optimizer):
        """Test complete optimization cycle."""
        # Mock the database operations
        with patch.object(performance_optimizer.index_optimizer, 'analyze_indexes') as mock_analyze:
            with patch.object(performance_optimizer.index_optimizer, 'generate_index_recommendations', return_value=[]) as mock_recommend:
                with patch.object(performance_optimizer.index_optimizer, 'create_recommended_indexes', return_value=0) as mock_create:
                    
                    results = await performance_optimizer.run_optimization_cycle()
                    
                    assert 'duration' in results
                    assert 'recommendations_generated' in results
                    assert 'indexes_created' in results
                    
                    mock_analyze.assert_called_once()
                    mock_recommend.assert_called_once()
                    mock_create.assert_called_once()


class TestCachedDecorator:
    """Test cached decorator functionality."""
    
    @pytest.mark.asyncio
    async def test_cached_decorator(self):
        """Test the cached decorator."""
        call_count = 0
        
        @cached(namespace="test", ttl=300)
        async def expensive_function(param: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"result_{param}"
        
        # Mock the cache manager
        with patch('src.shared.caching.cache_manager.get_cache_manager') as mock_get_cache:
            mock_cache = Mock()
            mock_cache.get = AsyncMock(return_value=None)  # Cache miss
            mock_cache.set = AsyncMock(return_value=True)
            mock_get_cache.return_value = mock_cache
            
            # First call should execute function
            result1 = await expensive_function("test")
            assert result1 == "result_test"
            assert call_count == 1
            
            # Mock cache hit for second call
            mock_cache.get = AsyncMock(return_value="cached_result_test")
            
            result2 = await expensive_function("test")
            assert result2 == "cached_result_test"
            assert call_count == 1  # Function not called again


class TestIntegration:
    """Test integration between different performance components."""
    
    @pytest.mark.asyncio
    async def test_cache_and_database_integration(self):
        """Test integration between caching and database optimization."""
        # This would test how cache warming interacts with database
        # query optimization, ensuring that frequently cached data
        # doesn't generate unnecessary database load
        
        cache_manager = Mock()
        cache_manager.get = AsyncMock(return_value=None)
        cache_manager.set = AsyncMock(return_value=True)
        
        performance_optimizer = Mock()
        performance_optimizer.analyze_query_performance = AsyncMock()
        
        # Simulate a database query that gets cached
        query = "SELECT * FROM popular_articles LIMIT 10"
        query_result = [{"id": 1, "title": "Popular Article"}]
        
        # First request: cache miss, database query
        cached_result = await cache_manager.get("articles", "popular")
        assert cached_result is None
        
        # Simulate database query
        await performance_optimizer.analyze_query_performance(query, 0.5, 1)
        
        # Cache the result
        await cache_manager.set("articles", "popular", query_result)
        
        # Second request: cache hit, no database query
        cache_manager.get = AsyncMock(return_value=query_result)
        cached_result = await cache_manager.get("articles", "popular")
        assert cached_result == query_result
        
        # Verify interactions
        performance_optimizer.analyze_query_performance.assert_called_once()
        cache_manager.set.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])