"""
Multi-layer caching system for Project Synapse.

This module provides a comprehensive caching solution with multiple layers:
- In-memory LRU cache for hot data
- Redis cache for distributed caching
- CDN integration for static content
- Intelligent cache invalidation and warming strategies
"""

from .cache_manager import (
    CacheLevel,
    CacheStrategy,
    CacheConfig,
    CacheEntry,
    LRUCache,
    RedisCache,
    CDNCache,
    MultiLayerCacheManager,
    cached,
    get_cache_manager,
    initialize_cache,
    shutdown_cache
)

from .cache_warming import (
    CacheWarmer,
    WarmingStrategy,
    WarmingJob,
    get_cache_warmer
)

from .invalidation import (
    InvalidationStrategy,
    CacheInvalidator,
    get_cache_invalidator
)

__all__ = [
    # Core classes
    'CacheLevel',
    'CacheStrategy',
    'CacheConfig',
    'CacheEntry',
    'LRUCache',
    'RedisCache',
    'CDNCache',
    'MultiLayerCacheManager',
    
    # Cache warming
    'CacheWarmer',
    'WarmingStrategy',
    'WarmingJob',
    
    # Cache invalidation
    'InvalidationStrategy',
    'CacheInvalidator',
    
    # Decorators and utilities
    'cached',
    
    # Factory functions
    'get_cache_manager',
    'get_cache_warmer',
    'get_cache_invalidator',
    
    # Lifecycle functions
    'initialize_cache',
    'shutdown_cache'
]