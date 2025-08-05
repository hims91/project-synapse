"""
Dendrites - Feed Polling System
Layer 0: Sensory Input

This module implements a priority-based feed polling system with adaptive
frequency based on feed activity and categorization.
"""
import asyncio
import hashlib
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import structlog

from .feed_parser import FeedParser, ParsedFeed, FeedItem
from ..shared.url_resolver import resolve_final_url

logger = structlog.get_logger(__name__)


class FeedPriority(str, Enum):
    """Feed priority levels for polling frequency."""
    CRITICAL = "critical"    # Every 5 minutes
    HIGH = "high"           # Every 15 minutes
    NORMAL = "normal"       # Every 30 minutes
    LOW = "low"            # Every 60 minutes
    INACTIVE = "inactive"   # Every 4 hours


class FeedCategory(str, Enum):
    """Feed categories for automatic priority assignment."""
    BREAKING_NEWS = "breaking_news"
    FINANCIAL = "financial"
    TECHNOLOGY = "technology"
    POLITICS = "politics"
    SPORTS = "sports"
    ENTERTAINMENT = "entertainment"
    GENERAL = "general"


@dataclass
class FeedMetrics:
    """Metrics for feed activity tracking."""
    total_polls: int = 0
    successful_polls: int = 0
    failed_polls: int = 0
    last_poll_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    last_new_items: int = 0
    average_items_per_poll: float = 0.0
    consecutive_failures: int = 0
    consecutive_empty_polls: int = 0
    
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_polls == 0:
            return 0.0
        return (self.successful_polls / self.total_polls) * 100
    
    def is_healthy(self) -> bool:
        """Check if feed is healthy based on metrics."""
        # New feeds (no polls yet) are considered healthy
        if self.total_polls == 0:
            return True
            
        return (
            self.consecutive_failures < 5 and
            self.success_rate() > 70 and
            self.consecutive_empty_polls < 10
        )


@dataclass
class FeedConfig:
    """Configuration for a single feed."""
    feed_id: str
    url: str
    name: str
    category: FeedCategory = FeedCategory.GENERAL
    priority: FeedPriority = FeedPriority.NORMAL
    custom_interval: Optional[int] = None  # Custom interval in minutes
    enabled: bool = True
    user_agent: str = "Project-Synapse-Dendrite/1.0"
    timeout: int = 30
    max_retries: int = 3
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if not self.feed_id:
            # Generate feed ID from URL hash
            self.feed_id = hashlib.sha256(self.url.encode()).hexdigest()[:16]


@dataclass
class PollResult:
    """Result of a feed polling operation."""
    feed_id: str
    success: bool
    poll_time: datetime
    new_items: List[FeedItem] = field(default_factory=list)
    total_items: int = 0
    error: Optional[str] = None
    response_time: float = 0.0
    feed_metadata: Optional[Dict] = None
    
    @property
    def new_item_count(self) -> int:
        """Number of new items found."""
        return len(self.new_items)


class FeedScheduler:
    """Manages feed polling schedules and priorities."""
    
    # Priority-based polling intervals (in minutes)
    PRIORITY_INTERVALS = {
        FeedPriority.CRITICAL: 5,
        FeedPriority.HIGH: 15,
        FeedPriority.NORMAL: 30,
        FeedPriority.LOW: 60,
        FeedPriority.INACTIVE: 240,
    }
    
    def __init__(self):
        self.feeds: Dict[str, FeedConfig] = {}
        self.metrics: Dict[str, FeedMetrics] = {}
        self.last_poll_times: Dict[str, datetime] = {}
        self.seen_items: Dict[str, Set[str]] = {}  # Track seen item GUIDs
        self.logger = logger.bind(component="feed_scheduler")
    
    def add_feed(self, feed_config: FeedConfig) -> None:
        """Add a feed to the polling schedule."""
        self.feeds[feed_config.feed_id] = feed_config
        self.metrics[feed_config.feed_id] = FeedMetrics()
        self.seen_items[feed_config.feed_id] = set()
        
        self.logger.info(
            "Feed added to scheduler",
            feed_id=feed_config.feed_id,
            url=feed_config.url,
            priority=feed_config.priority,
            category=feed_config.category
        )
    
    def remove_feed(self, feed_id: str) -> bool:
        """Remove a feed from the polling schedule."""
        if feed_id in self.feeds:
            del self.feeds[feed_id]
            del self.metrics[feed_id]
            del self.seen_items[feed_id]
            self.last_poll_times.pop(feed_id, None)
            
            self.logger.info("Feed removed from scheduler", feed_id=feed_id)
            return True
        return False
    
    def update_feed_priority(self, feed_id: str, priority: FeedPriority) -> bool:
        """Update feed priority."""
        if feed_id in self.feeds:
            old_priority = self.feeds[feed_id].priority
            self.feeds[feed_id].priority = priority
            
            self.logger.info(
                "Feed priority updated",
                feed_id=feed_id,
                old_priority=old_priority,
                new_priority=priority
            )
            return True
        return False
    
    def get_polling_interval(self, feed_id: str) -> int:
        """Get polling interval for a feed in minutes."""
        if feed_id not in self.feeds:
            return self.PRIORITY_INTERVALS[FeedPriority.NORMAL]
        
        feed = self.feeds[feed_id]
        
        # Use custom interval if specified
        if feed.custom_interval:
            return feed.custom_interval
        
        # Use priority-based interval
        base_interval = self.PRIORITY_INTERVALS[feed.priority]
        
        # Adaptive adjustment based on metrics
        metrics = self.metrics[feed_id]
        
        # Only adjust if we have polling history
        if metrics.total_polls > 0:
            # Increase interval for unhealthy feeds
            if not metrics.is_healthy():
                base_interval *= 2
            
            # Decrease interval for very active feeds
            elif metrics.average_items_per_poll > 5:
                base_interval = max(base_interval // 2, 5)  # Minimum 5 minutes
        
        return base_interval
    
    def should_poll_feed(self, feed_id: str) -> bool:
        """Check if a feed should be polled now."""
        if feed_id not in self.feeds or not self.feeds[feed_id].enabled:
            return False
        
        last_poll = self.last_poll_times.get(feed_id)
        if not last_poll:
            return True  # Never polled before
        
        interval_minutes = self.get_polling_interval(feed_id)
        next_poll_time = last_poll + timedelta(minutes=interval_minutes)
        
        return datetime.now(timezone.utc) >= next_poll_time
    
    def get_feeds_to_poll(self) -> List[FeedConfig]:
        """Get list of feeds that should be polled now."""
        feeds_to_poll = []
        
        for feed_id, feed_config in self.feeds.items():
            if self.should_poll_feed(feed_id):
                feeds_to_poll.append(feed_config)
        
        # Sort by priority (critical first)
        priority_order = {
            FeedPriority.CRITICAL: 0,
            FeedPriority.HIGH: 1,
            FeedPriority.NORMAL: 2,
            FeedPriority.LOW: 3,
            FeedPriority.INACTIVE: 4,
        }
        
        feeds_to_poll.sort(key=lambda f: priority_order[f.priority])
        return feeds_to_poll
    
    def update_metrics(self, result: PollResult) -> None:
        """Update feed metrics based on poll result."""
        feed_id = result.feed_id
        if feed_id not in self.metrics:
            return
        
        metrics = self.metrics[feed_id]
        metrics.total_polls += 1
        metrics.last_poll_time = result.poll_time
        
        if result.success:
            metrics.successful_polls += 1
            metrics.last_success_time = result.poll_time
            metrics.consecutive_failures = 0
            
            # Update item metrics
            metrics.last_new_items = result.new_item_count
            
            if result.new_item_count == 0:
                metrics.consecutive_empty_polls += 1
            else:
                metrics.consecutive_empty_polls = 0
            
            # Update average items per poll
            total_items = (metrics.average_items_per_poll * (metrics.successful_polls - 1) + 
                          result.new_item_count)
            metrics.average_items_per_poll = total_items / metrics.successful_polls
            
        else:
            metrics.failed_polls += 1
            metrics.consecutive_failures += 1
        
        # Auto-adjust priority based on activity
        self._auto_adjust_priority(feed_id)
        
        self.last_poll_times[feed_id] = result.poll_time
    
    def _auto_adjust_priority(self, feed_id: str) -> None:
        """Automatically adjust feed priority based on metrics."""
        if feed_id not in self.feeds or feed_id not in self.metrics:
            return
        
        feed = self.feeds[feed_id]
        metrics = self.metrics[feed_id]
        
        # Don't auto-adjust if custom interval is set
        if feed.custom_interval:
            return
        
        old_priority = feed.priority
        new_priority = old_priority
        
        # Promote to higher priority if very active
        if metrics.average_items_per_poll > 10 and metrics.success_rate() > 90:
            if old_priority == FeedPriority.NORMAL:
                new_priority = FeedPriority.HIGH
            elif old_priority == FeedPriority.LOW:
                new_priority = FeedPriority.NORMAL
        
        # Demote if consistently inactive or unhealthy
        elif metrics.consecutive_empty_polls > 20 or not metrics.is_healthy():
            if old_priority == FeedPriority.HIGH:
                new_priority = FeedPriority.NORMAL
            elif old_priority == FeedPriority.NORMAL:
                new_priority = FeedPriority.LOW
            elif old_priority == FeedPriority.LOW:
                new_priority = FeedPriority.INACTIVE
        
        if new_priority != old_priority:
            feed.priority = new_priority
            self.logger.info(
                "Auto-adjusted feed priority",
                feed_id=feed_id,
                old_priority=old_priority,
                new_priority=new_priority,
                avg_items=metrics.average_items_per_poll,
                success_rate=metrics.success_rate()
            )
    
    def get_scheduler_stats(self) -> Dict:
        """Get overall scheduler statistics."""
        total_feeds = len(self.feeds)
        enabled_feeds = sum(1 for f in self.feeds.values() if f.enabled)
        
        priority_counts = {}
        category_counts = {}
        
        for feed in self.feeds.values():
            priority_counts[feed.priority] = priority_counts.get(feed.priority, 0) + 1
            category_counts[feed.category] = category_counts.get(feed.category, 0) + 1
        
        total_polls = sum(m.total_polls for m in self.metrics.values())
        total_successes = sum(m.successful_polls for m in self.metrics.values())
        
        return {
            "total_feeds": total_feeds,
            "enabled_feeds": enabled_feeds,
            "priority_distribution": priority_counts,
            "category_distribution": category_counts,
            "total_polls": total_polls,
            "overall_success_rate": (total_successes / total_polls * 100) if total_polls > 0 else 0,
            "feeds_due_for_polling": len(self.get_feeds_to_poll())
        }


class FeedPoller:
    """Main feed polling engine."""
    
    def __init__(self, scheduler: Optional[FeedScheduler] = None):
        self.scheduler = scheduler or FeedScheduler()
        self.parser = FeedParser()
        self.logger = logger.bind(component="feed_poller")
        self._running = False
        self._poll_tasks: Set[asyncio.Task] = set()
    
    async def poll_feed(self, feed_config: FeedConfig) -> PollResult:
        """Poll a single feed and return results."""
        start_time = time.time()
        poll_time = datetime.now(timezone.utc)
        
        self.logger.info(
            "Polling feed",
            feed_id=feed_config.feed_id,
            url=feed_config.url,
            priority=feed_config.priority
        )
        
        try:
            # Fetch and parse feed
            import aiohttp
            
            timeout = aiohttp.ClientTimeout(total=feed_config.timeout)
            headers = {"User-Agent": feed_config.user_agent}
            
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                async with session.get(feed_config.url) as response:
                    response.raise_for_status()
                    content = await response.text()
            
            # Parse feed content
            parsed_feed = await self.parser.parse_feed(content, feed_config.url)
            
            # Resolve Google News URLs using the superior HTTP redirect method
            resolved_items = []
            for item in parsed_feed.items:
                if item.link and 'news.google.com' in item.link:
                    try:
                        resolved_url = await resolve_final_url(item.link)
                        if resolved_url != item.link:
                            self.logger.info(
                                "Resolved Google News URL",
                                feed_id=feed_config.feed_id,
                                original=item.link,
                                resolved=resolved_url
                            )
                            # Create a new item with the resolved URL
                            item.link = resolved_url
                    except Exception as e:
                        self.logger.warning(
                            "Failed to resolve Google News URL",
                            feed_id=feed_config.feed_id,
                            url=item.link,
                            error=str(e)
                        )
                resolved_items.append(item)
            
            # Update the parsed feed with resolved items
            parsed_feed.items = resolved_items
            
            # Filter new items
            new_items = self._filter_new_items(feed_config.feed_id, parsed_feed.items)
            
            response_time = time.time() - start_time
            
            result = PollResult(
                feed_id=feed_config.feed_id,
                success=True,
                poll_time=poll_time,
                new_items=new_items,
                total_items=len(parsed_feed.items),
                response_time=response_time,
                feed_metadata=parsed_feed.metadata.to_dict()
            )
            
            self.logger.info(
                "Feed poll successful",
                feed_id=feed_config.feed_id,
                new_items=len(new_items),
                total_items=len(parsed_feed.items),
                response_time=response_time
            )
            
            return result
            
        except Exception as e:
            response_time = time.time() - start_time
            error_msg = str(e)
            
            result = PollResult(
                feed_id=feed_config.feed_id,
                success=False,
                poll_time=poll_time,
                error=error_msg,
                response_time=response_time
            )
            
            self.logger.error(
                "Feed poll failed",
                feed_id=feed_config.feed_id,
                url=feed_config.url,
                error=error_msg,
                response_time=response_time
            )
            
            return result
    
    def _filter_new_items(self, feed_id: str, items: List[FeedItem]) -> List[FeedItem]:
        """Filter out items that have been seen before."""
        if feed_id not in self.scheduler.seen_items:
            self.scheduler.seen_items[feed_id] = set()
        
        seen_guids = self.scheduler.seen_items[feed_id]
        new_items = []
        
        for item in items:
            # Use GUID or URL as unique identifier
            item_id = item.guid or item.link
            if item_id and item_id not in seen_guids:
                new_items.append(item)
                seen_guids.add(item_id)
        
        # Limit seen items to prevent memory growth (keep last 1000)
        if len(seen_guids) > 1000:
            # Keep only the most recent 800 items
            seen_list = list(seen_guids)
            self.scheduler.seen_items[feed_id] = set(seen_list[-800:])
        
        return new_items
    
    async def poll_feeds_batch(self, feed_configs: List[FeedConfig]) -> List[PollResult]:
        """Poll multiple feeds concurrently."""
        if not feed_configs:
            return []
        
        self.logger.info(f"Polling batch of {len(feed_configs)} feeds")
        
        # Create polling tasks
        tasks = [self.poll_feed(config) for config in feed_configs]
        
        # Execute with timeout
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            poll_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # Create error result
                    error_result = PollResult(
                        feed_id=feed_configs[i].feed_id,
                        success=False,
                        poll_time=datetime.now(timezone.utc),
                        error=str(result)
                    )
                    poll_results.append(error_result)
                else:
                    poll_results.append(result)
            
            # Update metrics
            for result in poll_results:
                self.scheduler.update_metrics(result)
            
            return poll_results
            
        except Exception as e:
            self.logger.error("Batch polling failed", error=str(e))
            return []
    
    async def run_polling_cycle(self) -> Dict:
        """Run one complete polling cycle."""
        cycle_start = time.time()
        
        # Get feeds that need polling
        feeds_to_poll = self.scheduler.get_feeds_to_poll()
        
        if not feeds_to_poll:
            return {
                "feeds_polled": 0,
                "new_items": 0,
                "cycle_time": time.time() - cycle_start
            }
        
        self.logger.info(f"Starting polling cycle for {len(feeds_to_poll)} feeds")
        
        # Poll feeds in batches to avoid overwhelming servers
        batch_size = 10
        all_results = []
        
        for i in range(0, len(feeds_to_poll), batch_size):
            batch = feeds_to_poll[i:i + batch_size]
            batch_results = await self.poll_feeds_batch(batch)
            all_results.extend(batch_results)
            
            # Small delay between batches
            if i + batch_size < len(feeds_to_poll):
                await asyncio.sleep(1)
        
        # Calculate summary
        successful_polls = sum(1 for r in all_results if r.success)
        total_new_items = sum(r.new_item_count for r in all_results)
        cycle_time = time.time() - cycle_start
        
        self.logger.info(
            "Polling cycle completed",
            feeds_polled=len(feeds_to_poll),
            successful_polls=successful_polls,
            new_items=total_new_items,
            cycle_time=cycle_time
        )
        
        return {
            "feeds_polled": len(feeds_to_poll),
            "successful_polls": successful_polls,
            "new_items": total_new_items,
            "cycle_time": cycle_time,
            "results": all_results
        }
    
    async def start_continuous_polling(self, cycle_interval: int = 60) -> None:
        """Start continuous polling with specified cycle interval (seconds)."""
        self._running = True
        self.logger.info(f"Starting continuous polling with {cycle_interval}s cycles")
        
        while self._running:
            try:
                await self.run_polling_cycle()
                await asyncio.sleep(cycle_interval)
            except Exception as e:
                self.logger.error("Error in polling cycle", error=str(e))
                await asyncio.sleep(cycle_interval)
    
    def stop_polling(self) -> None:
        """Stop continuous polling."""
        self._running = False
        self.logger.info("Stopping continuous polling")
    
    def get_status(self) -> Dict:
        """Get current poller status."""
        return {
            "running": self._running,
            "active_tasks": len(self._poll_tasks),
            "scheduler_stats": self.scheduler.get_scheduler_stats()
        }


# Convenience functions for easy integration
async def create_feed_poller_with_defaults() -> FeedPoller:
    """Create a feed poller with some default feeds for testing."""
    scheduler = FeedScheduler()
    
    # Add some default feeds
    default_feeds = [
        FeedConfig(
            feed_id="bbc_news",
            url="http://feeds.bbci.co.uk/news/rss.xml",
            name="BBC News",
            category=FeedCategory.BREAKING_NEWS,
            priority=FeedPriority.HIGH
        ),
        FeedConfig(
            feed_id="techcrunch",
            url="https://techcrunch.com/feed/",
            name="TechCrunch",
            category=FeedCategory.TECHNOLOGY,
            priority=FeedPriority.NORMAL
        ),
        FeedConfig(
            feed_id="reuters_business",
            url="https://feeds.reuters.com/reuters/businessNews",
            name="Reuters Business",
            category=FeedCategory.FINANCIAL,
            priority=FeedPriority.HIGH
        )
    ]
    
    for feed_config in default_feeds:
        scheduler.add_feed(feed_config)
    
    return FeedPoller(scheduler)


if __name__ == "__main__":
    async def test_polling():
        """Test the polling system."""
        poller = await create_feed_poller_with_defaults()
        
        print("Running test polling cycle...")
        result = await poller.run_polling_cycle()
        
        print(f"Polling results: {result}")
        print(f"Scheduler stats: {poller.scheduler.get_scheduler_stats()}")
    
    asyncio.run(test_polling())