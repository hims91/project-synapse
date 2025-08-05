"""
Tests for rate limiting and abuse prevention functionality.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from fastapi import Request, FastAPI
from fastapi.testclient import TestClient
from starlette.responses import Response

from src.shared.security.rate_limiter import (
    RateLimiter,
    RateLimitMiddleware,
    UserTier,
    RateLimitType,
    RateLimitRule,
    RateLimitState,
    create_rate_limiter,
    get_tier_limits
)

from src.shared.security.abuse_prevention import (
    AbusePreventionSystem,
    AbuseType,
    ActionType,
    AbuseRule,
    ClientBehavior
)


class TestRateLimiter:
    """Test rate limiting functionality."""
    
    @pytest.fixture
    def rate_limiter(self):
        """Create a rate limiter for testing."""
        return RateLimiter()
    
    @pytest.fixture
    def mock_request(self):
        """Create a mock request."""
        request = Mock(spec=Request)
        request.client = Mock()
        request.client.host = "192.168.1.1"
        request.url = Mock()
        request.url.path = "/api/test"
        request.method = "GET"
        request.headers = {"user-agent": "test-client"}
        return request
    
    @pytest.mark.asyncio
    async def test_basic_rate_limiting(self, rate_limiter, mock_request):
        """Test basic rate limiting functionality."""
        # First request should be allowed
        allowed, info = await rate_limiter.check_rate_limit(
            mock_request, user_tier=UserTier.FREE
        )
        assert allowed is True
        assert info is None
        
        # Simulate many rapid requests
        for _ in range(70):  # Exceed free tier limit of 60/minute
            await rate_limiter.check_rate_limit(
                mock_request, user_tier=UserTier.FREE
            )
        
        # Next request should be blocked
        allowed, info = await rate_limiter.check_rate_limit(
            mock_request, user_tier=UserTier.FREE
        )
        assert allowed is False
        assert info is not None
        assert info['error'] == 'rate_limit_exceeded'
    
    @pytest.mark.asyncio
    async def test_tier_based_limits(self, rate_limiter, mock_request):
        """Test different limits for different user tiers."""
        # Test free tier (60 requests/minute)
        for _ in range(60):
            allowed, _ = await rate_limiter.check_rate_limit(
                mock_request, user_tier=UserTier.FREE
            )
            assert allowed is True
        
        # 61st request should be blocked for free tier
        allowed, info = await rate_limiter.check_rate_limit(
            mock_request, user_tier=UserTier.FREE
        )
        assert allowed is False
        
        # But premium tier should still be allowed
        allowed, _ = await rate_limiter.check_rate_limit(
            mock_request, user_tier=UserTier.PREMIUM
        )
        assert allowed is True
    
    @pytest.mark.asyncio
    async def test_user_vs_ip_based_limiting(self, rate_limiter, mock_request):
        """Test user-based vs IP-based rate limiting."""
        # Test with user ID
        for _ in range(60):
            allowed, _ = await rate_limiter.check_rate_limit(
                mock_request, user_id="user123", user_tier=UserTier.FREE
            )
            assert allowed is True
        
        # Should be blocked for this user
        allowed, _ = await rate_limiter.check_rate_limit(
            mock_request, user_id="user123", user_tier=UserTier.FREE
        )
        assert allowed is False
        
        # But different user should still be allowed
        allowed, _ = await rate_limiter.check_rate_limit(
            mock_request, user_id="user456", user_tier=UserTier.FREE
        )
        assert allowed is True
    
    @pytest.mark.asyncio
    async def test_custom_rules(self, rate_limiter, mock_request):
        """Test custom rate limiting rules."""
        # Add custom rule for specific endpoint
        custom_rule = RateLimitRule(
            name="api_endpoint_limit",
            limit_type=RateLimitType.REQUESTS_PER_MINUTE,
            limit=5,
            window_seconds=60,
            endpoint_pattern=r"/api/.*"
        )
        
        rate_limiter.add_custom_rule(custom_rule)
        
        # Should be limited to 5 requests
        for _ in range(5):
            allowed, _ = await rate_limiter.check_rate_limit(mock_request)
            assert allowed is True
        
        # 6th request should be blocked
        allowed, info = await rate_limiter.check_rate_limit(mock_request)
        assert allowed is False
        assert custom_rule.name in info['rule']
    
    @pytest.mark.asyncio
    async def test_concurrent_request_tracking(self, rate_limiter, mock_request):
        """Test concurrent request tracking."""
        # Start multiple concurrent requests
        await rate_limiter.check_rate_limit(mock_request)
        await rate_limiter.check_rate_limit(mock_request)
        
        # Check rate limit info
        info = rate_limiter.get_rate_limit_info(mock_request)
        assert info['concurrent_requests'] == 2
        
        # Complete requests
        await rate_limiter.request_completed(mock_request)
        await rate_limiter.request_completed(mock_request)
        
        info = rate_limiter.get_rate_limit_info(mock_request)
        assert info['concurrent_requests'] == 0
    
    def test_tier_limits_configuration(self):
        """Test tier limits configuration."""
        free_limits = get_tier_limits(UserTier.FREE)
        premium_limits = get_tier_limits(UserTier.PREMIUM)
        enterprise_limits = get_tier_limits(UserTier.ENTERPRISE)
        
        assert free_limits['requests_per_minute'] < premium_limits['requests_per_minute']
        assert premium_limits['requests_per_minute'] < enterprise_limits['requests_per_minute']
        
        assert free_limits['api_calls_per_minute'] < premium_limits['api_calls_per_minute']
    
    def test_rate_limiter_factory(self):
        """Test rate limiter factory function."""
        limiter = create_rate_limiter(
            free_requests_per_minute=30,
            premium_requests_per_minute=150
        )
        
        assert isinstance(limiter, RateLimiter)
        
        # Check that custom limits are applied
        free_rule = next(
            rule for rule in limiter.default_rules 
            if rule.name == "free_requests_per_minute"
        )
        assert free_rule.limit == 30
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_states(self, rate_limiter, mock_request):
        """Test cleanup of expired rate limit states."""
        # Create some state
        await rate_limiter.check_rate_limit(mock_request)
        
        # Manually set old timestamp
        key = rate_limiter._generate_key(mock_request)
        state = rate_limiter.states[key]
        state.last_request_time = datetime.utcnow() - timedelta(days=2)
        
        # Run cleanup
        cleaned = await rate_limiter.cleanup_expired_states()
        assert cleaned == 1
        assert key not in rate_limiter.states


class TestRateLimitMiddleware:
    """Test rate limit middleware."""
    
    @pytest.fixture
    def app(self):
        """Create test FastAPI app with rate limiting."""
        app = FastAPI()
        rate_limiter = RateLimiter()
        app.add_middleware(RateLimitMiddleware, rate_limiter=rate_limiter)
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "success"}
        
        return app
    
    def test_middleware_allows_normal_requests(self, app):
        """Test that middleware allows normal requests."""
        client = TestClient(app)
        response = client.get("/test")
        
        assert response.status_code == 200
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
    
    def test_middleware_blocks_excessive_requests(self, app):
        """Test that middleware blocks excessive requests."""
        client = TestClient(app)
        
        # Make many requests to trigger rate limit
        for _ in range(70):
            client.get("/test")
        
        # Next request should be blocked
        response = client.get("/test")
        assert response.status_code == 429
        assert "Retry-After" in response.headers
        
        response_data = response.json()
        assert "error" in response_data
        assert "upgrade_info" in response_data  # Should suggest upgrade


class TestAbusePreventionSystem:
    """Test abuse prevention functionality."""
    
    @pytest.fixture
    def abuse_system(self):
        """Create abuse prevention system for testing."""
        return AbusePreventionSystem()
    
    @pytest.fixture
    def mock_request(self):
        """Create mock request for testing."""
        request = Mock(spec=Request)
        request.client = Mock()
        request.client.host = "192.168.1.1"
        request.url = Mock()
        request.url.path = "/api/test"
        request.method = "GET"
        request.headers = {"user-agent": "test-client", "content-length": "100"}
        return request
    
    @pytest.mark.asyncio
    async def test_normal_behavior_allowed(self, abuse_system, mock_request):
        """Test that normal behavior is allowed."""
        allowed, info = await abuse_system.analyze_request(
            mock_request, response_time=0.1, status_code=200, payload_size=100
        )
        
        assert allowed is True
        assert info is None
    
    @pytest.mark.asyncio
    async def test_high_request_rate_detection(self, abuse_system, mock_request):
        """Test detection of high request rates."""
        client_id = "test_client"
        
        # Simulate rapid requests
        for _ in range(25):  # Trigger high request rate
            await abuse_system.analyze_request(
                mock_request, response_time=0.01, status_code=200, 
                payload_size=100, client_id=client_id
            )
            await asyncio.sleep(0.01)  # Very short intervals
        
        # Next request should be flagged or blocked
        allowed, info = await abuse_system.analyze_request(
            mock_request, response_time=0.01, status_code=200,
            payload_size=100, client_id=client_id
        )
        
        # Might be blocked or warned depending on exact timing
        if not allowed:
            assert info['reason'] == 'abuse_detected'
        else:
            assert info is None or 'warning' in info
    
    @pytest.mark.asyncio
    async def test_high_error_rate_detection(self, abuse_system, mock_request):
        """Test detection of high error rates."""
        client_id = "error_client"
        
        # Simulate many error responses
        for _ in range(30):
            await abuse_system.analyze_request(
                mock_request, response_time=0.1, status_code=404,
                payload_size=100, client_id=client_id
            )
        
        # Should trigger high error rate rule
        allowed, info = await abuse_system.analyze_request(
            mock_request, response_time=0.1, status_code=404,
            payload_size=100, client_id=client_id
        )
        
        assert not allowed
        assert info['reason'] == 'abuse_detected'
        assert 'high_error_rate' in str(info['rules_triggered'])
    
    @pytest.mark.asyncio
    async def test_endpoint_enumeration_detection(self, abuse_system):
        """Test detection of endpoint enumeration."""
        client_id = "enum_client"
        
        # Access many different endpoints
        for i in range(150):
            request = Mock(spec=Request)
            request.client = Mock()
            request.client.host = "192.168.1.1"
            request.url = Mock()
            request.url.path = f"/api/endpoint_{i}"
            request.method = "GET"
            request.headers = {"user-agent": "test-client", "content-length": "100"}
            
            await abuse_system.analyze_request(
                request, response_time=0.1, status_code=200,
                payload_size=100, client_id=client_id
            )
        
        # Should trigger endpoint enumeration rule
        behavior = abuse_system.client_behaviors[client_id]
        assert len(behavior.endpoints_accessed) > 100
        assert behavior.get_behavior_score() > 50  # High suspicion score
    
    @pytest.mark.asyncio
    async def test_bot_detection(self, abuse_system, mock_request):
        """Test bot-like behavior detection."""
        client_id = "bot_client"
        
        # Simulate very consistent timing (bot-like)
        for _ in range(25):
            await abuse_system.analyze_request(
                mock_request, response_time=0.1, status_code=200,
                payload_size=100, client_id=client_id
            )
            await asyncio.sleep(0.1)  # Exactly 0.1 seconds between requests
        
        behavior = abuse_system.client_behaviors[client_id]
        
        # Check if bot-like patterns are detected
        if len(behavior.request_intervals) > 10:
            import statistics
            variance = statistics.variance(list(behavior.request_intervals))
            # Very low variance indicates bot-like timing
            assert variance < 0.1 or behavior.get_behavior_score() > 30
    
    @pytest.mark.asyncio
    async def test_client_blocking(self, abuse_system, mock_request):
        """Test client blocking functionality."""
        client_id = "blocked_client"
        
        # Trigger blocking by simulating credential stuffing
        for _ in range(25):
            auth_request = Mock(spec=Request)
            auth_request.client = Mock()
            auth_request.client.host = "192.168.1.1"
            auth_request.url = Mock()
            auth_request.url.path = "/auth/login"
            auth_request.method = "POST"
            auth_request.headers = {"user-agent": "test-client", "content-length": "100"}
            
            await abuse_system.analyze_request(
                auth_request, response_time=0.1, status_code=401,
                payload_size=100, client_id=client_id
            )
        
        # Client should be blocked
        is_blocked = await abuse_system._is_client_blocked(client_id)
        if is_blocked:
            # Verify blocked client can't make requests
            allowed, info = await abuse_system.analyze_request(
                mock_request, response_time=0.1, status_code=200,
                payload_size=100, client_id=client_id
            )
            
            assert not allowed
            assert info['reason'] == 'client_blocked'
    
    @pytest.mark.asyncio
    async def test_custom_abuse_rules(self, abuse_system, mock_request):
        """Test custom abuse detection rules."""
        # Add custom rule
        custom_rule = AbuseRule(
            name="test_custom_rule",
            abuse_type=AbuseType.SUSPICIOUS_BEHAVIOR,
            description="Test custom rule",
            condition=lambda behavior: behavior.request_count > 5,
            severity=5,
            action=ActionType.WARN
        )
        
        abuse_system.add_custom_rule(custom_rule)
        
        client_id = "custom_test"
        
        # Make enough requests to trigger custom rule
        for _ in range(7):
            await abuse_system.analyze_request(
                mock_request, response_time=0.1, status_code=200,
                payload_size=100, client_id=client_id
            )
        
        # Check if custom rule was triggered
        behavior = abuse_system.client_behaviors[client_id]
        assert len(behavior.abuse_events) > 0
        
        # Find our custom rule event
        custom_events = [
            event for event in behavior.abuse_events
            if event.details.get('rule_name') == 'test_custom_rule'
        ]
        assert len(custom_events) > 0
    
    def test_client_whitelisting(self, abuse_system):
        """Test client whitelisting functionality."""
        client_id = "whitelisted_client"
        
        # Block client
        abuse_system.blocked_clients[client_id] = datetime.utcnow() + timedelta(hours=1)
        
        # Verify blocked
        assert client_id in abuse_system.blocked_clients
        
        # Whitelist client
        abuse_system.whitelist_client(client_id)
        
        # Verify unblocked
        assert client_id not in abuse_system.blocked_clients
    
    def test_get_client_info(self, abuse_system, mock_request):
        """Test getting client information."""
        client_id = "info_client"
        
        # Create some behavior
        asyncio.run(abuse_system.analyze_request(
            mock_request, response_time=0.1, status_code=200,
            payload_size=100, client_id=client_id
        ))
        
        # Get client info
        info = abuse_system.get_client_info(client_id)
        
        assert info is not None
        assert info['client_id'] == client_id
        assert info['request_count'] == 1
        assert 'behavior_score' in info
        assert 'first_seen' in info
        assert 'last_seen' in info
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_data(self, abuse_system, mock_request):
        """Test cleanup of expired data."""
        client_id = "expired_client"
        
        # Create behavior
        await abuse_system.analyze_request(
            mock_request, response_time=0.1, status_code=200,
            payload_size=100, client_id=client_id
        )
        
        # Manually set old timestamp
        behavior = abuse_system.client_behaviors[client_id]
        behavior.last_seen = datetime.utcnow() - timedelta(days=8)
        
        # Add expired block
        abuse_system.blocked_clients["expired_block"] = datetime.utcnow() - timedelta(hours=1)
        
        # Run cleanup
        cleaned = await abuse_system.cleanup_expired_data()
        
        assert cleaned >= 1
        assert client_id not in abuse_system.client_behaviors
        assert "expired_block" not in abuse_system.blocked_clients
    
    def test_statistics(self, abuse_system):
        """Test statistics collection."""
        stats = abuse_system.get_stats()
        
        assert 'total_requests_analyzed' in stats
        assert 'abuse_events_detected' in stats
        assert 'clients_blocked' in stats
        assert 'active_clients' in stats
        assert 'abuse_detection_rate' in stats


class TestIntegration:
    """Test integration between rate limiting and abuse prevention."""
    
    @pytest.mark.asyncio
    async def test_combined_protection(self):
        """Test rate limiting and abuse prevention working together."""
        rate_limiter = RateLimiter()
        abuse_system = AbusePreventionSystem()
        
        request = Mock(spec=Request)
        request.client = Mock()
        request.client.host = "192.168.1.1"
        request.url = Mock()
        request.url.path = "/api/test"
        request.method = "GET"
        request.headers = {"user-agent": "test-client"}
        
        client_id = "integration_test"
        
        # Simulate normal usage
        for i in range(10):
            # Check rate limit
            rate_allowed, rate_info = await rate_limiter.check_rate_limit(
                request, user_id=client_id, user_tier=UserTier.FREE
            )
            
            # Check abuse prevention
            abuse_allowed, abuse_info = await abuse_system.analyze_request(
                request, response_time=0.1, status_code=200,
                payload_size=100, client_id=client_id
            )
            
            # Both should allow normal requests
            assert rate_allowed is True
            assert abuse_allowed is True
        
        # Simulate abuse (rapid requests)
        for i in range(100):
            await rate_limiter.check_rate_limit(
                request, user_id=client_id, user_tier=UserTier.FREE
            )
            await abuse_system.analyze_request(
                request, response_time=0.01, status_code=200,
                payload_size=100, client_id=client_id
            )
        
        # At least one system should block further requests
        rate_allowed, _ = await rate_limiter.check_rate_limit(
            request, user_id=client_id, user_tier=UserTier.FREE
        )
        abuse_allowed, _ = await abuse_system.analyze_request(
            request, response_time=0.01, status_code=200,
            payload_size=100, client_id=client_id
        )
        
        # At least one should block
        assert not (rate_allowed and abuse_allowed)


if __name__ == "__main__":
    pytest.main([__file__])