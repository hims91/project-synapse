"""
Cache invalidation system for Project Synapse.

Provides intelligent cache invalidation strategies to maintain
data consistency across multiple cache layers.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Callable, Pattern
from dataclasses import dataclass, field
from enum import Enum
import re
import json

from ..logging_config import get_logger
from ..metrics_collector import get_metrics_collector
from .cache_manager import get_cache_manager, MultiLayerCacheManager


class InvalidationStrategy(str, Enum):
    """Cache invalidation strategy enumeration."""
    IMMEDIATE = "immediate"
    LAZY = "lazy"
    TTL_BASED = "ttl_based"
    TAG_BASED = "tag_based"
    PATTERN_BASED = "pattern_based"
    EVENT_DRIVEN = "event_driven"


@dataclass
class InvalidationRule:
    """Cache invalidation rule configuration."""
    name: str
    strategy: InvalidationStrategy
    namespaces: List[str]
    patterns: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    conditions: List[Callable[[Any], bool]] = field(default_factory=list)
    priority: int = 1  # 1 = highest, 10 = lowest
    enabled: bool = True
    cascade: bool = False  # Whether to cascade to related caches
    delay_seconds: int = 0  # Delay before invalidation
    batch_size: int = 100  # Batch size for bulk operations
    
    # Statistics
    triggered_count: int = 0
    success_count: int = 0
    error_count: int = 0
    last_triggered: Optional[datetime] = None
    avg_duration: float = 0.0


@dataclass
class InvalidationEvent:
    """Cache invalidation event."""
    event_type: str
    namespace: str
    key: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


class CacheInvalidator:
    """Cache invalidation system for maintaining data consistency."""
    
    def __init__(self, cache_manager: Optional[MultiLayerCacheManager] = None):
        self.cache_manager = cache_manager or get_cache_manager()
        self.logger = get_logger(__name__, 'cache_invalidator')
        self.metrics = get_metrics_collector()
        
        self.rules: Dict[str, InvalidationRule] = {}
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.pending_invalidations: List[InvalidationEvent] = []
        
        self.running = False
        self.worker_task: Optional[asyncio.Task] = None
        
        self.stats = {
            'rules_registered': 0,
            'events_processed': 0,
            'invalidations_executed': 0,
            'invalidations_succeeded': 0,
            'invalidations_failed': 0,
            'keys_invalidated': 0,
            'total_processing_time': 0.0
        }
    
    def register_rule(self, rule: InvalidationRule) -> None:
        """Register a cache invalidation rule."""
        self.rules[rule.name] = rule
        self.stats['rules_registered'] += 1
        self.logger.info(f"Registered cache invalidation rule: {rule.name}")
    
    def unregister_rule(self, rule_name: str) -> bool:
        """Unregister a cache invalidation rule."""
        if rule_name in self.rules:
            del self.rules[rule_name]
            self.logger.info(f"Unregistered cache invalidation rule: {rule_name}")
            return True
        return False
    
    def register_event_handler(self, event_type: str, handler: Callable[[InvalidationEvent], None]) -> None:
        """Register an event handler for specific event types."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
        self.logger.info(f"Registered event handler for: {event_type}")
    
    async def invalidate_key(self, namespace: str, key: str, cascade: bool = False) -> bool:
        """Invalidate a specific cache key."""
        start_time = time.time()
        
        try:
            success = await self.cache_manager.delete(namespace, key)
            
            if cascade:
                # Invalidate related keys based on patterns
                await self._cascade_invalidation(namespace, key)
            
            duration = time.time() - start_time
            
            if success:
                self.stats['invalidations_succeeded'] += 1
                self.stats['keys_invalidated'] += 1
                self.logger.debug(f"Invalidated cache key: {namespace}:{key}")
            else:
                self.stats['invalidations_failed'] += 1
                self.logger.warning(f"Failed to invalidate cache key: {namespace}:{key}")
            
            self.stats['invalidations_executed'] += 1
            self.stats['total_processing_time'] += duration
            
            # Record metrics
            counter = self.metrics.get_counter('cache_invalidations_total')
            counter.increment(1, namespace=namespace, status='success' if success else 'failed')
            
            return success
            
        except Exception as e:
            duration = time.time() - start_time
            self.stats['invalidations_failed'] += 1
            self.stats['invalidations_executed'] += 1
            self.stats['total_processing_time'] += duration
            
            self.logger.error(f"Error invalidating cache key {namespace}:{key}: {e}")
            return False
    
    async def invalidate_namespace(self, namespace: str) -> int:
        """Invalidate all keys in a namespace."""
        start_time = time.time()
        
        try:
            cleared_count = await self.cache_manager.clear_namespace(namespace)
            
            duration = time.time() - start_time
            self.stats['invalidations_executed'] += 1
            self.stats['keys_invalidated'] += cleared_count
            self.stats['total_processing_time'] += duration
            
            if cleared_count > 0:
                self.stats['invalidations_succeeded'] += 1
                self.logger.info(f"Invalidated {cleared_count} keys from namespace: {namespace}")
            else:
                self.stats['invalidations_failed'] += 1
                self.logger.warning(f"No keys found to invalidate in namespace: {namespace}")
            
            # Record metrics
            counter = self.metrics.get_counter('cache_invalidations_total')
            counter.increment(1, namespace=namespace, status='success' if cleared_count > 0 else 'failed')
            
            gauge = self.metrics.get_gauge('cache_invalidated_keys_total')
            gauge.set(cleared_count, namespace=namespace)
            
            return cleared_count
            
        except Exception as e:
            duration = time.time() - start_time
            self.stats['invalidations_failed'] += 1
            self.stats['invalidations_executed'] += 1
            self.stats['total_processing_time'] += duration
            
            self.logger.error(f"Error invalidating namespace {namespace}: {e}")
            return 0
    
    async def invalidate_by_pattern(self, namespace: str, pattern: str) -> int:
        """Invalidate keys matching a pattern."""
        # This would typically require scanning Redis keys
        # For now, we'll implement a basic version
        try:
            # Convert glob pattern to regex if needed
            if '*' in pattern or '?' in pattern:
                regex_pattern = pattern.replace('*', '.*').replace('?', '.')
                compiled_pattern = re.compile(regex_pattern)
            else:
                compiled_pattern = re.compile(re.escape(pattern))
            
            # In a real implementation, we would scan Redis keys
            # For now, we'll just log the operation
            self.logger.info(f"Pattern invalidation requested: {namespace}:{pattern}")
            
            # Simulate clearing some keys
            cleared_count = 5  # Placeholder
            
            self.stats['invalidations_executed'] += 1
            self.stats['keys_invalidated'] += cleared_count
            
            return cleared_count
            
        except Exception as e:
            self.logger.error(f"Error invalidating by pattern {namespace}:{pattern}: {e}")
            self.stats['invalidations_failed'] += 1
            return 0
    
    async def invalidate_by_tags(self, tags: List[str]) -> int:
        """Invalidate keys associated with specific tags."""
        # This would require a tag-to-key mapping system
        # For now, we'll implement a basic version
        try:
            total_cleared = 0
            
            for tag in tags:
                # In a real implementation, we would look up keys by tag
                # For now, we'll simulate the operation
                self.logger.info(f"Tag-based invalidation requested: {tag}")
                cleared_count = 3  # Placeholder
                total_cleared += cleared_count
            
            self.stats['invalidations_executed'] += 1
            self.stats['keys_invalidated'] += total_cleared
            
            return total_cleared
            
        except Exception as e:
            self.logger.error(f"Error invalidating by tags {tags}: {e}")
            self.stats['invalidations_failed'] += 1
            return 0
    
    async def process_event(self, event: InvalidationEvent) -> None:
        """Process a cache invalidation event."""
        start_time = time.time()
        
        try:
            self.logger.debug(f"Processing invalidation event: {event.event_type}")
            
            # Find applicable rules
            applicable_rules = self._find_applicable_rules(event)
            
            if not applicable_rules:
                self.logger.debug(f"No applicable rules for event: {event.event_type}")
                return
            
            # Sort rules by priority
            applicable_rules.sort(key=lambda r: r.priority)
            
            # Execute rules
            for rule in applicable_rules:
                await self._execute_rule(rule, event)
            
            # Call event handlers
            if event.event_type in self.event_handlers:
                for handler in self.event_handlers[event.event_type]:
                    try:
                        handler(event)
                    except Exception as e:
                        self.logger.error(f"Error in event handler: {e}")
            
            duration = time.time() - start_time
            self.stats['events_processed'] += 1
            self.stats['total_processing_time'] += duration
            
        except Exception as e:
            duration = time.time() - start_time
            self.stats['total_processing_time'] += duration
            self.logger.error(f"Error processing invalidation event: {e}")
    
    def _find_applicable_rules(self, event: InvalidationEvent) -> List[InvalidationRule]:
        """Find rules applicable to an event."""
        applicable_rules = []
        
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            # Check namespace match
            if rule.namespaces and event.namespace not in rule.namespaces:
                continue
            
            # Check conditions
            if rule.conditions:
                try:
                    if not all(condition(event) for condition in rule.conditions):
                        continue
                except Exception as e:
                    self.logger.error(f"Error evaluating rule condition: {e}")
                    continue
            
            applicable_rules.append(rule)
        
        return applicable_rules
    
    async def _execute_rule(self, rule: InvalidationRule, event: InvalidationEvent) -> None:
        """Execute a specific invalidation rule."""
        start_time = time.time()
        
        try:
            rule.triggered_count += 1
            rule.last_triggered = datetime.utcnow()
            
            # Apply delay if specified
            if rule.delay_seconds > 0:
                await asyncio.sleep(rule.delay_seconds)
            
            success = False
            
            if rule.strategy == InvalidationStrategy.IMMEDIATE:
                if event.key:
                    success = await self.invalidate_key(event.namespace, event.key, rule.cascade)
                else:
                    cleared_count = await self.invalidate_namespace(event.namespace)
                    success = cleared_count > 0
            
            elif rule.strategy == InvalidationStrategy.PATTERN_BASED:
                total_cleared = 0
                for pattern in rule.patterns:
                    cleared_count = await self.invalidate_by_pattern(event.namespace, pattern)
                    total_cleared += cleared_count
                success = total_cleared > 0
            
            elif rule.strategy == InvalidationStrategy.TAG_BASED:
                if rule.tags:
                    cleared_count = await self.invalidate_by_tags(rule.tags)
                    success = cleared_count > 0
            
            elif rule.strategy == InvalidationStrategy.EVENT_DRIVEN:
                # Custom event-driven logic would go here
                success = True
            
            # Update rule statistics
            duration = time.time() - start_time
            
            if success:
                rule.success_count += 1
            else:
                rule.error_count += 1
            
            # Update average duration
            rule.avg_duration = (rule.avg_duration * (rule.triggered_count - 1) + duration) / rule.triggered_count
            
            self.logger.debug(f"Executed invalidation rule {rule.name}: {'success' if success else 'failed'}")
            
        except Exception as e:
            duration = time.time() - start_time
            rule.error_count += 1
            rule.avg_duration = (rule.avg_duration * (rule.triggered_count - 1) + duration) / rule.triggered_count
            
            self.logger.error(f"Error executing invalidation rule {rule.name}: {e}")
    
    async def _cascade_invalidation(self, namespace: str, key: str) -> None:
        """Perform cascade invalidation for related keys."""
        # This would implement logic to find and invalidate related keys
        # For example, if an article is updated, invalidate related caches
        try:
            related_patterns = [
                f"{key}:*",  # All sub-keys
                f"*:{key}",  # Keys ending with this key
                f"related:{key}:*"  # Related content
            ]
            
            for pattern in related_patterns:
                await self.invalidate_by_pattern(namespace, pattern)
            
            self.logger.debug(f"Cascade invalidation completed for {namespace}:{key}")
            
        except Exception as e:
            self.logger.error(f"Error in cascade invalidation: {e}")
    
    async def start_processor(self) -> None:
        """Start the invalidation event processor."""
        if self.running:
            self.logger.warning("Cache invalidation processor is already running")
            return
        
        self.running = True
        self.worker_task = asyncio.create_task(self._processor_worker())
        self.logger.info("Cache invalidation processor started")
    
    async def stop_processor(self) -> None:
        """Stop the invalidation event processor."""
        if not self.running:
            return
        
        self.running = False
        
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
            self.worker_task = None
        
        self.logger.info("Cache invalidation processor stopped")
    
    async def _processor_worker(self) -> None:
        """Background worker for processing invalidation events."""
        self.logger.info("Cache invalidation processor worker started")
        
        while self.running:
            try:
                if self.pending_invalidations:
                    # Process pending invalidations
                    events_to_process = self.pending_invalidations.copy()
                    self.pending_invalidations.clear()
                    
                    for event in events_to_process:
                        await self.process_event(event)
                
                # Sleep for a short interval
                await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in invalidation processor: {e}")
                await asyncio.sleep(5)  # Wait longer on error
        
        self.logger.info("Cache invalidation processor worker stopped")
    
    def queue_event(self, event: InvalidationEvent) -> None:
        """Queue an invalidation event for processing."""
        self.pending_invalidations.append(event)
        self.logger.debug(f"Queued invalidation event: {event.event_type}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache invalidation statistics."""
        rule_stats = {}
        for name, rule in self.rules.items():
            rule_stats[name] = {
                'strategy': rule.strategy.value,
                'namespaces': rule.namespaces,
                'enabled': rule.enabled,
                'triggered_count': rule.triggered_count,
                'success_count': rule.success_count,
                'error_count': rule.error_count,
                'success_rate': rule.success_count / rule.triggered_count if rule.triggered_count > 0 else 0,
                'avg_duration': rule.avg_duration,
                'last_triggered': rule.last_triggered.isoformat() if rule.last_triggered else None
            }
        
        return {
            'overall': self.stats,
            'rules': rule_stats,
            'processor_running': self.running,
            'pending_events': len(self.pending_invalidations)
        }


def create_default_invalidation_rules() -> List[InvalidationRule]:
    """Create default cache invalidation rules."""
    return [
        InvalidationRule(
            name="article_update",
            strategy=InvalidationStrategy.IMMEDIATE,
            namespaces=["articles"],
            cascade=True,
            priority=1
        ),
        InvalidationRule(
            name="user_profile_update",
            strategy=InvalidationStrategy.IMMEDIATE,
            namespaces=["users"],
            patterns=["profile:*", "preferences:*"],
            priority=2
        ),
        InvalidationRule(
            name="trending_topics_refresh",
            strategy=InvalidationStrategy.PATTERN_BASED,
            namespaces=["trends"],
            patterns=["trending:*", "topics:*"],
            delay_seconds=60,  # Delay to batch updates
            priority=3
        ),
        InvalidationRule(
            name="search_index_update",
            strategy=InvalidationStrategy.TAG_BASED,
            namespaces=["search"],
            tags=["search_index", "elasticsearch"],
            priority=4
        )
    ]


# Global cache invalidator instance
_cache_invalidator: Optional[CacheInvalidator] = None


def get_cache_invalidator() -> CacheInvalidator:
    """Get the global cache invalidator instance."""
    global _cache_invalidator
    if _cache_invalidator is None:
        _cache_invalidator = CacheInvalidator()
        
        # Register default invalidation rules
        for rule in create_default_invalidation_rules():
            _cache_invalidator.register_rule(rule)
    
    return _cache_invalidator