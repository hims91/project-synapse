"""
Multi-layer caching system for Project Synapse.

Provides Redis caching, in-memory caching, and CDN integration
for optimal performance across different data access patterns.
"""

import asyncio
import json
import pickle
import time
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
import functools
from collections import OrderedDict
import threading

import redis.asyncio as redis
from redis.asyncio import Redis
import aiohttp

from ..logging_config import get_logger
from ..metrics_collector import get_metrics_collector
from ..config import get_settings

T = TypeVar('T')


class CacheLevel(str, Enum):
    """Cache level enumeration."""
    MEMORY = "memory"
    REDIS = "redis"
    CDN = "cdn"
    DATABASE = "database"


class CacheStrategy(str, Enum):
    """Cache strategy enumeration."""
    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"
    WRITE_AROUND = "write_around"
    READ_THROUGH = "read_through"
    CACHE_ASIDE = "cache_aside"


@dataclass
class CacheConfig:
    """Cache configuration settings."""
    # Memory cache settings
    memory_max_size: int = 1000
    memory_ttl: int = 300  # 5 minutes
    
    # Redis cache settings
    redis_url: str = "redis://localhost:6379"
    redis_db: int = 0
    redis_ttl: int = 3600  # 1 hour
    redis_max_connections: int = 20
    
    # CDN settings
    cdn_enabled: bool = False
    cdn_base_url: str = ""
    cdn_ttl: int = 86400  # 24 hours
    
    # General settings
    default_strategy: CacheStrategy = CacheStrategy.CACHE_ASIDE
    compression_enabled: bool = True
    serialization_format: str = "json"  # json, pickle, msgpack
    
    # Performance settings
    batch_size: int = 100
    pipeline_enabled: bool = True
    connection_pool_size: int = 10


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime]
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    size_bytes: int = 0
    tags: List[str] = field(default_factory=list)
    
    def is_expired(self) -> bool:
        """Check if the cache entry is expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def touch(self):
        """Update access information."""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()


class LRUCache(Generic[T]):
    """Thread-safe LRU cache implementation."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size': 0
        }
    
    def get(self, key: str) -> Optional[T]:
        """Get value from cache."""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                if entry.is_expired():
                    del self.cache[key]
                    self.stats['misses'] += 1
                    return None
                
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                entry.touch()
                self.stats['hits'] += 1
                return entry.value
            
            self.stats['misses'] += 1
            return None
    
    def set(self, key: str, value: T, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        with self.lock:
            now = datetime.utcnow()
            expires_at = now + timedelta(seconds=ttl) if ttl else None
            
            # Calculate approximate size
            size_bytes = len(str(value).encode('utf-8'))
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                expires_at=expires_at,
                size_bytes=size_bytes
            )
            
            if key in self.cache:
                # Update existing entry
                self.cache[key] = entry
                self.cache.move_to_end(key)
            else:
                # Add new entry
                self.cache[key] = entry
                
                # Evict if necessary
                while len(self.cache) > self.max_size:
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                    self.stats['evictions'] += 1
            
            self.stats['size'] = len(self.cache)
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                self.stats['size'] = len(self.cache)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.stats['size'] = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                **self.stats,
                'hit_rate': hit_rate,
                'total_requests': total_requests
            }


class RedisCache:
    """Redis-based cache implementation."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.redis_client: Optional[Redis] = None
        self.logger = get_logger(__name__, 'redis_cache')
        self.metrics = get_metrics_collector()
        
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'errors': 0
        }
    
    async def connect(self) -> None:
        """Connect to Redis."""
        try:
            self.redis_client = redis.from_url(
                self.config.redis_url,
                db=self.config.redis_db,
                max_connections=self.config.redis_max_connections,
                decode_responses=False  # We handle encoding ourselves
            )
            
            # Test connection
            await self.redis_client.ping()
            self.logger.info("Connected to Redis successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            self.stats['errors'] += 1
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None
            self.logger.info("Disconnected from Redis")
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage."""
        try:
            if self.config.serialization_format == "json":
                return json.dumps(value, default=str).encode('utf-8')
            elif self.config.serialization_format == "pickle":
                return pickle.dumps(value)
            else:
                return str(value).encode('utf-8')
        except Exception as e:
            self.logger.error(f"Serialization error: {e}")
            raise
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        try:
            if self.config.serialization_format == "json":
                return json.loads(data.decode('utf-8'))
            elif self.config.serialization_format == "pickle":
                return pickle.loads(data)
            else:
                return data.decode('utf-8')
        except Exception as e:
            self.logger.error(f"Deserialization error: {e}")
            raise
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        if not self.redis_client:
            await self.connect()
        
        try:
            data = await self.redis_client.get(key)
            if data is None:
                self.stats['misses'] += 1
                return None
            
            value = self._deserialize(data)
            self.stats['hits'] += 1
            
            # Record metrics
            counter = self.metrics.get_counter('cache_redis_hits_total')
            counter.increment(1)
            
            return value
            
        except Exception as e:
            self.logger.error(f"Redis get error for key {key}: {e}")
            self.stats['errors'] += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache."""
        if not self.redis_client:
            await self.connect()
        
        try:
            data = self._serialize(value)
            ttl = ttl or self.config.redis_ttl
            
            await self.redis_client.setex(key, ttl, data)
            self.stats['sets'] += 1
            
            # Record metrics
            counter = self.metrics.get_counter('cache_redis_sets_total')
            counter.increment(1)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Redis set error for key {key}: {e}")
            self.stats['errors'] += 1
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from Redis cache."""
        if not self.redis_client:
            await self.connect()
        
        try:
            result = await self.redis_client.delete(key)
            self.stats['deletes'] += 1
            return result > 0
            
        except Exception as e:
            self.logger.error(f"Redis delete error for key {key}: {e}")
            self.stats['errors'] += 1
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache."""
        if not self.redis_client:
            await self.connect()
        
        try:
            return await self.redis_client.exists(key) > 0
        except Exception as e:
            self.logger.error(f"Redis exists error for key {key}: {e}")
            self.stats['errors'] += 1
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration time for key."""
        if not self.redis_client:
            await self.connect()
        
        try:
            return await self.redis_client.expire(key, ttl)
        except Exception as e:
            self.logger.error(f"Redis expire error for key {key}: {e}")
            self.stats['errors'] += 1
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear keys matching pattern."""
        if not self.redis_client:
            await self.connect()
        
        try:
            keys = await self.redis_client.keys(pattern)
            if keys:
                return await self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            self.logger.error(f"Redis clear pattern error for {pattern}: {e}")
            self.stats['errors'] += 1
            return 0
    
    async def get_multiple(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from Redis cache."""
        if not self.redis_client:
            await self.connect()
        
        try:
            values = await self.redis_client.mget(keys)
            result = {}
            
            for key, data in zip(keys, values):
                if data is not None:
                    try:
                        result[key] = self._deserialize(data)
                        self.stats['hits'] += 1
                    except Exception as e:
                        self.logger.error(f"Deserialization error for key {key}: {e}")
                        self.stats['errors'] += 1
                else:
                    self.stats['misses'] += 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"Redis mget error: {e}")
            self.stats['errors'] += 1
            return {}
    
    async def set_multiple(self, data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple values in Redis cache."""
        if not self.redis_client:
            await self.connect()
        
        try:
            pipe = self.redis_client.pipeline()
            ttl = ttl or self.config.redis_ttl
            
            for key, value in data.items():
                serialized_data = self._serialize(value)
                pipe.setex(key, ttl, serialized_data)
            
            await pipe.execute()
            self.stats['sets'] += len(data)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Redis mset error: {e}")
            self.stats['errors'] += 1
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics."""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            **self.stats,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }


class CDNCache:
    """CDN-based cache implementation."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.logger = get_logger(__name__, 'cdn_cache')
        self.session: Optional[aiohttp.ClientSession] = None
        
        self.stats = {
            'hits': 0,
            'misses': 0,
            'purges': 0,
            'errors': 0
        }
    
    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self.session
    
    async def close(self) -> None:
        """Close HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    def _get_cdn_url(self, key: str) -> str:
        """Get CDN URL for key."""
        return f"{self.config.cdn_base_url.rstrip('/')}/{key}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from CDN cache."""
        if not self.config.cdn_enabled:
            return None
        
        try:
            session = await self.get_session()
            url = self._get_cdn_url(key)
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.text()
                    self.stats['hits'] += 1
                    return json.loads(data)
                elif response.status == 404:
                    self.stats['misses'] += 1
                    return None
                else:
                    self.logger.warning(f"CDN get unexpected status {response.status} for key {key}")
                    self.stats['errors'] += 1
                    return None
                    
        except Exception as e:
            self.logger.error(f"CDN get error for key {key}: {e}")
            self.stats['errors'] += 1
            return None
    
    async def purge(self, key: str) -> bool:
        """Purge key from CDN cache."""
        if not self.config.cdn_enabled:
            return False
        
        try:
            session = await self.get_session()
            url = self._get_cdn_url(key)
            
            # This would typically use CDN-specific purge API
            # For now, we'll simulate with a DELETE request
            async with session.delete(url) as response:
                success = response.status in [200, 204, 404]
                if success:
                    self.stats['purges'] += 1
                else:
                    self.stats['errors'] += 1
                return success
                
        except Exception as e:
            self.logger.error(f"CDN purge error for key {key}: {e}")
            self.stats['errors'] += 1
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get CDN cache statistics."""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            **self.stats,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }


class MultiLayerCacheManager:
    """Multi-layer cache manager coordinating all cache levels."""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.logger = get_logger(__name__, 'cache_manager')
        self.metrics = get_metrics_collector()
        
        # Initialize cache layers
        self.memory_cache = LRUCache[Any](self.config.memory_max_size)
        self.redis_cache = RedisCache(self.config)
        self.cdn_cache = CDNCache(self.config)
        
        # Cache statistics
        self.stats = {
            'total_requests': 0,
            'memory_hits': 0,
            'redis_hits': 0,
            'cdn_hits': 0,
            'cache_misses': 0,
            'write_operations': 0,
            'invalidations': 0
        }
    
    async def initialize(self) -> None:
        """Initialize cache manager."""
        try:
            await self.redis_cache.connect()
            self.logger.info("Cache manager initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize cache manager: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown cache manager."""
        try:
            await self.redis_cache.disconnect()
            await self.cdn_cache.close()
            self.logger.info("Cache manager shutdown completed")
        except Exception as e:
            self.logger.error(f"Error during cache manager shutdown: {e}")
    
    def _generate_cache_key(self, namespace: str, key: str, **kwargs) -> str:
        """Generate cache key with namespace and parameters."""
        if kwargs:
            # Sort kwargs for consistent key generation
            params = "&".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
            key_data = f"{namespace}:{key}:{params}"
        else:
            key_data = f"{namespace}:{key}"
        
        # Hash long keys to avoid Redis key length limits
        if len(key_data) > 250:
            key_hash = hashlib.sha256(key_data.encode()).hexdigest()
            return f"{namespace}:hash:{key_hash}"
        
        return key_data
    
    async def get(self, namespace: str, key: str, **kwargs) -> Optional[Any]:
        """Get value from multi-layer cache."""
        cache_key = self._generate_cache_key(namespace, key, **kwargs)
        self.stats['total_requests'] += 1
        
        # Try memory cache first
        value = self.memory_cache.get(cache_key)
        if value is not None:
            self.stats['memory_hits'] += 1
            self.logger.debug(f"Cache hit (memory): {cache_key}")
            return value
        
        # Try Redis cache
        value = await self.redis_cache.get(cache_key)
        if value is not None:
            self.stats['redis_hits'] += 1
            self.logger.debug(f"Cache hit (Redis): {cache_key}")
            
            # Populate memory cache
            self.memory_cache.set(cache_key, value, self.config.memory_ttl)
            return value
        
        # Try CDN cache
        value = await self.cdn_cache.get(cache_key)
        if value is not None:
            self.stats['cdn_hits'] += 1
            self.logger.debug(f"Cache hit (CDN): {cache_key}")
            
            # Populate lower-level caches
            self.memory_cache.set(cache_key, value, self.config.memory_ttl)
            await self.redis_cache.set(cache_key, value, self.config.redis_ttl)
            return value
        
        # Cache miss
        self.stats['cache_misses'] += 1
        self.logger.debug(f"Cache miss: {cache_key}")
        return None
    
    async def set(self, namespace: str, key: str, value: Any, ttl: Optional[int] = None, **kwargs) -> bool:
        """Set value in multi-layer cache."""
        cache_key = self._generate_cache_key(namespace, key, **kwargs)
        self.stats['write_operations'] += 1
        
        success = True
        
        # Set in memory cache
        memory_ttl = min(ttl or self.config.memory_ttl, self.config.memory_ttl)
        self.memory_cache.set(cache_key, value, memory_ttl)
        
        # Set in Redis cache
        redis_ttl = ttl or self.config.redis_ttl
        redis_success = await self.redis_cache.set(cache_key, value, redis_ttl)
        success = success and redis_success
        
        self.logger.debug(f"Cache set: {cache_key}")
        return success
    
    async def delete(self, namespace: str, key: str, **kwargs) -> bool:
        """Delete value from multi-layer cache."""
        cache_key = self._generate_cache_key(namespace, key, **kwargs)
        self.stats['invalidations'] += 1
        
        # Delete from all cache layers
        memory_success = self.memory_cache.delete(cache_key)
        redis_success = await self.redis_cache.delete(cache_key)
        cdn_success = await self.cdn_cache.purge(cache_key)
        
        self.logger.debug(f"Cache delete: {cache_key}")
        return memory_success or redis_success or cdn_success
    
    async def clear_namespace(self, namespace: str) -> int:
        """Clear all keys in a namespace."""
        pattern = f"{namespace}:*"
        
        # Clear from Redis (memory cache will naturally expire)
        cleared_count = await self.redis_cache.clear_pattern(pattern)
        
        self.logger.info(f"Cleared {cleared_count} keys from namespace: {namespace}")
        return cleared_count
    
    async def get_multiple(self, namespace: str, keys: List[str], **kwargs) -> Dict[str, Any]:
        """Get multiple values from cache."""
        cache_keys = [self._generate_cache_key(namespace, key, **kwargs) for key in keys]
        result = {}
        
        # Try memory cache first
        memory_results = {}
        redis_keys_needed = []
        
        for original_key, cache_key in zip(keys, cache_keys):
            value = self.memory_cache.get(cache_key)
            if value is not None:
                memory_results[original_key] = value
                self.stats['memory_hits'] += 1
            else:
                redis_keys_needed.append((original_key, cache_key))
        
        result.update(memory_results)
        
        # Try Redis for remaining keys
        if redis_keys_needed:
            redis_cache_keys = [cache_key for _, cache_key in redis_keys_needed]
            redis_results = await self.redis_cache.get_multiple(redis_cache_keys)
            
            for (original_key, cache_key) in redis_keys_needed:
                if cache_key in redis_results:
                    value = redis_results[cache_key]
                    result[original_key] = value
                    self.stats['redis_hits'] += 1
                    
                    # Populate memory cache
                    self.memory_cache.set(cache_key, value, self.config.memory_ttl)
                else:
                    self.stats['cache_misses'] += 1
        
        self.stats['total_requests'] += len(keys)
        return result
    
    async def set_multiple(self, namespace: str, data: Dict[str, Any], ttl: Optional[int] = None, **kwargs) -> bool:
        """Set multiple values in cache."""
        cache_data = {}
        
        for key, value in data.items():
            cache_key = self._generate_cache_key(namespace, key, **kwargs)
            cache_data[cache_key] = value
            
            # Set in memory cache
            memory_ttl = min(ttl or self.config.memory_ttl, self.config.memory_ttl)
            self.memory_cache.set(cache_key, value, memory_ttl)
        
        # Set in Redis cache
        redis_ttl = ttl or self.config.redis_ttl
        success = await self.redis_cache.set_multiple(cache_data, redis_ttl)
        
        self.stats['write_operations'] += len(data)
        return success
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        total_hits = self.stats['memory_hits'] + self.stats['redis_hits'] + self.stats['cdn_hits']
        total_requests = self.stats['total_requests']
        overall_hit_rate = total_hits / total_requests if total_requests > 0 else 0
        
        return {
            'overall': {
                'total_requests': total_requests,
                'total_hits': total_hits,
                'cache_misses': self.stats['cache_misses'],
                'hit_rate': overall_hit_rate,
                'write_operations': self.stats['write_operations'],
                'invalidations': self.stats['invalidations']
            },
            'memory': self.memory_cache.get_stats(),
            'redis': self.redis_cache.get_stats(),
            'cdn': self.cdn_cache.get_stats()
        }


# Decorator for caching function results
def cached(namespace: str, ttl: Optional[int] = None, key_func: Optional[Callable] = None):
    """Decorator for caching function results."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            cache_manager = get_cache_manager()
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                if args:
                    key_parts.extend(str(arg) for arg in args)
                if kwargs:
                    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = ":".join(key_parts)
            
            # Try to get from cache
            cached_result = await cache_manager.get(namespace, cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache_manager.set(namespace, cache_key, result, ttl)
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For synchronous functions, we can't use async cache
            # This would need to be implemented with a sync cache or thread pool
            return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# Global cache manager instance
_cache_manager: Optional[MultiLayerCacheManager] = None


def get_cache_manager() -> MultiLayerCacheManager:
    """Get the global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        settings = get_settings()
        config = CacheConfig(
            redis_url=settings.redis_url,
            memory_max_size=getattr(settings, 'cache_memory_max_size', 1000),
            memory_ttl=getattr(settings, 'cache_memory_ttl', 300),
            redis_ttl=getattr(settings, 'cache_redis_ttl', 3600),
            cdn_enabled=getattr(settings, 'cache_cdn_enabled', False),
            cdn_base_url=getattr(settings, 'cache_cdn_base_url', ''),
        )
        _cache_manager = MultiLayerCacheManager(config)
    return _cache_manager


async def initialize_cache() -> None:
    """Initialize the global cache manager."""
    cache_manager = get_cache_manager()
    await cache_manager.initialize()


async def shutdown_cache() -> None:
    """Shutdown the global cache manager."""
    global _cache_manager
    if _cache_manager:
        await _cache_manager.shutdown()
        _cache_manager = None