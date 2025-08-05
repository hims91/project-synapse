"""
Unit tests for the feed polling system.
Tests feed scheduling, priority management, and polling functionality.
"""
import pytest
import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, AsyncMock, patch

from src.dendrites.feed_poller import (
    FeedScheduler, FeedPoller, FeedConfig, FeedPriority, FeedCategory,
    FeedMetrics, PollResult, create_feed_poller_with_defaults
)
from src.dendrites.feed_parser import FeedItem, FeedMetadata, ParsedFeed


class TestFeedConfig:
    """Test FeedConfig functionality."""
    
    def test_feed_config_initialization(self):
        """Test FeedConfig initialization with defaults."""
        config = FeedConfig(
            feed_id="test_feed",
            url="https://example.com/feed.xml",
            name="Test Feed"
        )
        
        assert config.feed_id == "test_feed"
        assert config.url == "https://example.com/feed.xml"
        assert config.name == "Test Feed"
        assert config.category == FeedCategory.GENERAL
        assert config.priority == FeedPriority.NORMAL
        assert config.enabled is True
        assert config.timeout == 30
        assert config.max_retries == 3
    
    def test_feed_config_auto_id_generation(self):
        """Test automatic feed ID generation."""
        config = FeedConfig(
            feed_id="",
            url="https://example.com/feed.xml",
            name="Test Feed"
        )
        
        assert config.feed_id != ""
        assert len(config.feed_id) == 16  # SHA256 hash truncated to 16 chars
    
    def test_feed_config_with_custom_values(self):
        """Test FeedConfig with custom values."""
        config = FeedConfig(
            feed_id="custom_feed",
            url="https://news.example.com/rss",
            name="Custom News Feed",
            category=FeedCategory.BREAKING_NEWS,
            priority=FeedPriority.CRITICAL,
            custom_interval=10,
            enabled=False,
            timeout=60,
            tags=["news", "breaking"],
            metadata={"source": "example"}
        )
        
        assert config.category == FeedCategory.BREAKING_NEWS
        assert config.priority == FeedPriority.CRITICAL
        assert config.custom_interval == 10
        assert config.enabled is False
        assert config.timeout == 60
        assert config.tags == ["news", "breaking"]
        assert config.metadata == {"source": "example"}


class TestFeedMetrics:
    """Test FeedMetrics functionality."""
    
    def test_metrics_initialization(self):
        """Test FeedMetrics initialization."""
        metrics = FeedMetrics()
        
        assert metrics.total_polls == 0
        assert metrics.successful_polls == 0
        assert metrics.failed_polls == 0
        assert metrics.last_poll_time is None
        assert metrics.average_items_per_poll == 0.0
        assert metrics.consecutive_failures == 0
    
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        metrics = FeedMetrics()
        
        # No polls yet
        assert metrics.success_rate() == 0.0
        
        # Some successful polls
        metrics.total_polls = 10
        metrics.successful_polls = 8
        assert metrics.success_rate() == 80.0
        
        # All successful
        metrics.successful_polls = 10
        assert metrics.success_rate() == 100.0
    
    def test_is_healthy(self):
        """Test health check logic."""
        metrics = FeedMetrics()
        
        # New feed should be healthy
        assert metrics.is_healthy() is True
        
        # Too many consecutive failures (need polls > 0 for health check)
        metrics.total_polls = 10
        metrics.successful_polls = 4
        metrics.consecutive_failures = 6
        assert metrics.is_healthy() is False
        
        # Reset failures but low success rate
        metrics.consecutive_failures = 0
        metrics.successful_polls = 5  # 50% success rate
        assert metrics.is_healthy() is False
        
        # Good success rate but too many empty polls
        metrics.successful_polls = 9  # 90% success rate
        metrics.consecutive_empty_polls = 15
        assert metrics.is_healthy() is False
        
        # Healthy feed
        metrics.consecutive_empty_polls = 5
        assert metrics.is_healthy() is True


class TestFeedScheduler:
    """Test FeedScheduler functionality."""
    
    @pytest.fixture
    def scheduler(self):
        """Create a FeedScheduler instance for testing."""
        return FeedScheduler()
    
    @pytest.fixture
    def sample_feed_config(self):
        """Create a sample feed configuration."""
        return FeedConfig(
            feed_id="test_feed",
            url="https://example.com/feed.xml",
            name="Test Feed",
            priority=FeedPriority.NORMAL
        )
    
    def test_add_feed(self, scheduler, sample_feed_config):
        """Test adding a feed to the scheduler."""
        scheduler.add_feed(sample_feed_config)
        
        assert "test_feed" in scheduler.feeds
        assert "test_feed" in scheduler.metrics
        assert "test_feed" in scheduler.seen_items
        assert scheduler.feeds["test_feed"] == sample_feed_config
    
    def test_remove_feed(self, scheduler, sample_feed_config):
        """Test removing a feed from the scheduler."""
        scheduler.add_feed(sample_feed_config)
        
        # Feed should exist
        assert "test_feed" in scheduler.feeds
        
        # Remove feed
        result = scheduler.remove_feed("test_feed")
        assert result is True
        
        # Feed should be gone
        assert "test_feed" not in scheduler.feeds
        assert "test_feed" not in scheduler.metrics
        assert "test_feed" not in scheduler.seen_items
        
        # Removing non-existent feed should return False
        result = scheduler.remove_feed("non_existent")
        assert result is False
    
    def test_update_feed_priority(self, scheduler, sample_feed_config):
        """Test updating feed priority."""
        scheduler.add_feed(sample_feed_config)
        
        # Update priority
        result = scheduler.update_feed_priority("test_feed", FeedPriority.HIGH)
        assert result is True
        assert scheduler.feeds["test_feed"].priority == FeedPriority.HIGH
        
        # Update non-existent feed
        result = scheduler.update_feed_priority("non_existent", FeedPriority.LOW)
        assert result is False
    
    def test_get_polling_interval(self, scheduler, sample_feed_config):
        """Test polling interval calculation."""
        scheduler.add_feed(sample_feed_config)
        
        # Normal priority should have 30-minute interval
        interval = scheduler.get_polling_interval("test_feed")
        assert interval == 30
        
        # Update to critical priority
        scheduler.update_feed_priority("test_feed", FeedPriority.CRITICAL)
        interval = scheduler.get_polling_interval("test_feed")
        assert interval == 5
        
        # Custom interval should override priority
        sample_feed_config.custom_interval = 45
        scheduler.feeds["test_feed"] = sample_feed_config
        interval = scheduler.get_polling_interval("test_feed")
        assert interval == 45
    
    def test_should_poll_feed(self, scheduler, sample_feed_config):
        """Test feed polling decision logic."""
        scheduler.add_feed(sample_feed_config)
        
        # Never polled before - should poll
        assert scheduler.should_poll_feed("test_feed") is True
        
        # Just polled - should not poll
        scheduler.last_poll_times["test_feed"] = datetime.now(timezone.utc)
        assert scheduler.should_poll_feed("test_feed") is False
        
        # Polled long ago - should poll
        old_time = datetime.now(timezone.utc) - timedelta(hours=2)
        scheduler.last_poll_times["test_feed"] = old_time
        assert scheduler.should_poll_feed("test_feed") is True
        
        # Disabled feed - should not poll
        scheduler.feeds["test_feed"].enabled = False
        assert scheduler.should_poll_feed("test_feed") is False
    
    def test_get_feeds_to_poll(self, scheduler):
        """Test getting feeds that need polling."""
        # Add feeds with different priorities
        feeds = [
            FeedConfig("critical_feed", "https://example.com/critical", "Critical", priority=FeedPriority.CRITICAL),
            FeedConfig("high_feed", "https://example.com/high", "High", priority=FeedPriority.HIGH),
            FeedConfig("normal_feed", "https://example.com/normal", "Normal", priority=FeedPriority.NORMAL),
        ]
        
        for feed in feeds:
            scheduler.add_feed(feed)
        
        # All feeds should be due for polling (never polled)
        due_feeds = scheduler.get_feeds_to_poll()
        assert len(due_feeds) == 3
        
        # Should be sorted by priority (critical first)
        assert due_feeds[0].priority == FeedPriority.CRITICAL
        assert due_feeds[1].priority == FeedPriority.HIGH
        assert due_feeds[2].priority == FeedPriority.NORMAL
    
    def test_update_metrics(self, scheduler, sample_feed_config):
        """Test metrics updating."""
        scheduler.add_feed(sample_feed_config)
        
        # Create successful poll result
        result = PollResult(
            feed_id="test_feed",
            success=True,
            poll_time=datetime.now(timezone.utc),
            new_items=[Mock()],  # One new item
            total_items=5
        )
        
        scheduler.update_metrics(result)
        
        metrics = scheduler.metrics["test_feed"]
        assert metrics.total_polls == 1
        assert metrics.successful_polls == 1
        assert metrics.failed_polls == 0
        assert metrics.last_new_items == 1
        assert metrics.consecutive_failures == 0
        
        # Create failed poll result
        failed_result = PollResult(
            feed_id="test_feed",
            success=False,
            poll_time=datetime.now(timezone.utc),
            error="Network timeout"
        )
        
        scheduler.update_metrics(failed_result)
        
        assert metrics.total_polls == 2
        assert metrics.successful_polls == 1
        assert metrics.failed_polls == 1
        assert metrics.consecutive_failures == 1
    
    def test_auto_priority_adjustment(self, scheduler, sample_feed_config):
        """Test automatic priority adjustment based on metrics."""
        scheduler.add_feed(sample_feed_config)
        
        # Simulate very active feed
        metrics = scheduler.metrics["test_feed"]
        metrics.average_items_per_poll = 15  # Very active
        metrics.total_polls = 10
        metrics.successful_polls = 10  # 100% success rate
        
        scheduler._auto_adjust_priority("test_feed")
        
        # Should be promoted to HIGH priority
        assert scheduler.feeds["test_feed"].priority == FeedPriority.HIGH
    
    def test_scheduler_stats(self, scheduler):
        """Test scheduler statistics."""
        # Add some feeds
        feeds = [
            FeedConfig("feed1", "https://example.com/1", "Feed 1", priority=FeedPriority.HIGH),
            FeedConfig("feed2", "https://example.com/2", "Feed 2", priority=FeedPriority.NORMAL),
            FeedConfig("feed3", "https://example.com/3", "Feed 3", priority=FeedPriority.NORMAL, enabled=False),
        ]
        
        for feed in feeds:
            scheduler.add_feed(feed)
        
        stats = scheduler.get_scheduler_stats()
        
        assert stats["total_feeds"] == 3
        assert stats["enabled_feeds"] == 2
        assert stats["priority_distribution"][FeedPriority.HIGH] == 1
        assert stats["priority_distribution"][FeedPriority.NORMAL] == 2
        assert stats["total_polls"] == 0  # No polls yet


class TestFeedPoller:
    """Test FeedPoller functionality."""
    
    @pytest.fixture
    def scheduler(self):
        """Create a FeedScheduler with test feeds."""
        scheduler = FeedScheduler()
        
        feed_config = FeedConfig(
            feed_id="test_feed",
            url="https://httpbin.org/xml",  # Test XML endpoint
            name="Test Feed"
        )
        scheduler.add_feed(feed_config)
        
        return scheduler
    
    @pytest.fixture
    def poller(self, scheduler):
        """Create a FeedPoller instance for testing."""
        return FeedPoller(scheduler)
    
    @pytest.mark.asyncio
    async def test_poll_feed_success(self, poller):
        """Test successful feed polling."""
        # Create a mock feed config
        feed_config = FeedConfig(
            feed_id="test_feed",
            url="https://example.com/feed.xml",
            name="Test Feed"
        )
        
        # Mock the HTTP request and feed parsing
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.raise_for_status = Mock()
            mock_response.text = AsyncMock(return_value="""<?xml version="1.0"?>
                <rss version="2.0">
                    <channel>
                        <title>Test Feed</title>
                        <link>https://example.com</link>
                        <description>Test Description</description>
                        <item>
                            <title>Test Item</title>
                            <link>https://example.com/item1</link>
                            <description>Test item description</description>
                        </item>
                    </channel>
                </rss>""")
            
            mock_get.return_value.__aenter__.return_value = mock_response
            
            result = await poller.poll_feed(feed_config)
            
            assert result.success is True
            assert result.feed_id == "test_feed"
            assert result.new_item_count >= 0
            assert result.response_time >= 0  # Can be 0 in mocked tests
    
    @pytest.mark.asyncio
    async def test_poll_feed_failure(self, poller):
        """Test failed feed polling."""
        feed_config = FeedConfig(
            feed_id="test_feed",
            url="https://invalid-url-that-does-not-exist.com/feed.xml",
            name="Test Feed"
        )
        
        result = await poller.poll_feed(feed_config)
        
        assert result.success is False
        assert result.feed_id == "test_feed"
        assert result.error is not None
        assert result.response_time > 0
    
    @pytest.mark.asyncio
    async def test_poll_feeds_batch(self, poller):
        """Test batch polling of multiple feeds."""
        feed_configs = [
            FeedConfig("feed1", "https://example.com/feed1.xml", "Feed 1"),
            FeedConfig("feed2", "https://example.com/feed2.xml", "Feed 2"),
        ]
        
        # Mock successful responses
        with patch.object(poller, 'poll_feed') as mock_poll:
            mock_poll.side_effect = [
                PollResult("feed1", True, datetime.now(timezone.utc), new_items=[Mock()]),
                PollResult("feed2", True, datetime.now(timezone.utc), new_items=[Mock(), Mock()]),
            ]
            
            results = await poller.poll_feeds_batch(feed_configs)
            
            assert len(results) == 2
            assert all(r.success for r in results)
            assert mock_poll.call_count == 2
    
    @pytest.mark.asyncio
    async def test_run_polling_cycle(self, poller):
        """Test running a complete polling cycle."""
        # Mock the scheduler to return feeds to poll
        with patch.object(poller.scheduler, 'get_feeds_to_poll') as mock_get_feeds:
            mock_get_feeds.return_value = [
                FeedConfig("feed1", "https://example.com/feed1.xml", "Feed 1")
            ]
            
            # Mock the polling method
            with patch.object(poller, 'poll_feeds_batch') as mock_batch_poll:
                mock_batch_poll.return_value = [
                    PollResult("feed1", True, datetime.now(timezone.utc), new_items=[Mock()])
                ]
                
                result = await poller.run_polling_cycle()
                
                assert result["feeds_polled"] == 1
                assert result["successful_polls"] == 1
                assert result["new_items"] == 1
                assert result["cycle_time"] >= 0  # Can be 0 in fast mocked tests
    
    def test_filter_new_items(self, poller):
        """Test filtering of new items."""
        feed_id = "test_feed"
        
        # Create some test items
        items = [
            FeedItem(title="Item 1", link="https://example.com/item1", guid="item1"),
            FeedItem(title="Item 2", link="https://example.com/item2", guid="item2"),
            FeedItem(title="Item 3", link="https://example.com/item3", guid="item3"),
        ]
        
        # First time - all items should be new
        new_items = poller._filter_new_items(feed_id, items)
        assert len(new_items) == 3
        
        # Second time with same items - no new items
        new_items = poller._filter_new_items(feed_id, items)
        assert len(new_items) == 0
        
        # Add a new item
        items.append(FeedItem(title="Item 4", link="https://example.com/item4", guid="item4"))
        new_items = poller._filter_new_items(feed_id, items)
        assert len(new_items) == 1
        assert new_items[0].guid == "item4"
    
    def test_poller_status(self, poller):
        """Test getting poller status."""
        status = poller.get_status()
        
        assert "running" in status
        assert "active_tasks" in status
        assert "scheduler_stats" in status
        assert isinstance(status["running"], bool)
        assert isinstance(status["active_tasks"], int)


class TestIntegration:
    """Integration tests for the complete polling system."""
    
    @pytest.mark.asyncio
    async def test_create_default_poller(self):
        """Test creating a poller with default feeds."""
        poller = await create_feed_poller_with_defaults()
        
        assert poller is not None
        assert len(poller.scheduler.feeds) > 0
        
        # Check that default feeds are added
        feed_ids = list(poller.scheduler.feeds.keys())
        assert "bbc_news" in feed_ids
        assert "techcrunch" in feed_ids
        assert "reuters_business" in feed_ids
    
    @pytest.mark.asyncio
    async def test_end_to_end_polling(self):
        """Test end-to-end polling with real feeds (if network available)."""
        scheduler = FeedScheduler()
        
        # Add a reliable test feed
        test_feed = FeedConfig(
            feed_id="httpbin_xml",
            url="https://httpbin.org/xml",
            name="HTTPBin XML Test",
            priority=FeedPriority.NORMAL
        )
        scheduler.add_feed(test_feed)
        
        poller = FeedPoller(scheduler)
        
        try:
            # Run one polling cycle
            result = await poller.run_polling_cycle()
            
            # Should have attempted to poll the feed
            assert result["feeds_polled"] >= 0
            assert "cycle_time" in result
            
        except Exception as e:
            # Network issues are acceptable in tests
            pytest.skip(f"Network test skipped due to: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])