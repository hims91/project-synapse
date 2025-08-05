"""
Rate limiting and abuse prevention system for Project Synapse.

Provides tier-based rate limiting, IP-based limiting, abuse detection,
and graceful rate limit responses with upgrade prompts.
"""

import asyncio
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ..logging_config import get_logger
from ..metrics_collector import get_metrics_collector


class UserTier(str, Enum):
    """User tier levels."""
    FREE = "free"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"
    ADMIN = "admin"


class RateLimitType(str, Enum):
    """Types of rate limits."""
    REQUESTS_PER_MINUTE = "requests_per_minute"
    REQUESTS_PER_HOUR = "requests_per_hour"
    REQUESTS_PER_DAY = "requests_per_day"
    CONCURRENT_REQUESTS = "concurrent_requests"
    BANDWIDTH_PER_MINUTE = "bandwidth_per_minute"
    API_CALLS_PER_MINUTE = "api_calls_per_minute"


@dataclass
class RateLimitRule:
    """Rate limiting rule configuration."""
    name: str
    limit_type: RateLimitType
    limit: int
    window_seconds: int
    tier: Optional[UserTier] = None
    endpoint_pattern: Optional[str] = None
    method: Optional[str] = None
    burst_limit: Optional[int] = None
    
    def matches_request(self, request: Request, user_tier: UserTier) -> bool:
        """Check if this rule applies to the request."""
        # Check tier
        if self.tier and self.tier != user_tier:
            return False
        
        # Check method
        if self.method and self.method.upper() != request.method.upper():
            return False
        
        # Check endpoint pattern
        if self.endpoint_pattern:
            import re
            if not re.match(self.endpoint_pattern, str(request.url.path)):
                return False
        
        return True


@dataclass
class RateLimitState:
    """Current state of rate limiting for a key."""
    requests: deque = field(default_factory=deque)
    bandwidth_used: deque = field(default_factory=deque)
    concurrent_requests: int = 0
    first_request_time: Optional[datetime] = None
    last_request_time: Optional[datetime] = None
    total_requests: int = 0
    violations: int = 0
    blocked_until: Optional[datetime] = None


@dataclass
class AbusePattern:
    """Abuse detection pattern."""
    name: str
    description: str
    threshold: int
    window_seconds: int
    action: str  # "warn", "block", "throttle"
    block_duration_seconds: int = 300  # 5 minutes default


class RateLimiter:
    """Advanced rate limiting system."""
    
    def __init__(self):
        self.logger = get_logger(__name__, 'rate_limiter')
        self.metrics = get_metrics_collector()
        
        # Rate limit states: key -> RateLimitState
        self.states: Dict[str, RateLimitState] = defaultdict(RateLimitState)
        
        # Abuse detection patterns
        self.abuse_patterns = [
            AbusePattern(
                name="rapid_requests",
                description="Too many requests in short time",
                threshold=100,
                window_seconds=60,
                action="throttle",
                block_duration_seconds=300
            ),
            AbusePattern(
                name="repeated_errors",
                description="Too many error responses",
                threshold=50,
                window_seconds=300,
                action="block",
                block_duration_seconds=600
            ),
            AbusePattern(
                name="large_payloads",
                description="Consistently large request payloads",
                threshold=10,
                window_seconds=60,
                action="throttle",
                block_duration_seconds=180
            ),
            AbusePattern(
                name="suspicious_patterns",
                description="Suspicious request patterns",
                threshold=20,
                window_seconds=120,
                action="warn",
                block_duration_seconds=60
            )
        ]
        
        # Default rate limit rules
        self.default_rules = [
            # Free tier limits
            RateLimitRule(
                name="free_requests_per_minute",
                limit_type=RateLimitType.REQUESTS_PER_MINUTE,
                limit=60,
                window_seconds=60,
                tier=UserTier.FREE
            ),
            RateLimitRule(
                name="free_requests_per_hour",
                limit_type=RateLimitType.REQUESTS_PER_HOUR,
                limit=1000,
                window_seconds=3600,
                tier=UserTier.FREE
            ),
            RateLimitRule(
                name="free_api_calls_per_minute",
                limit_type=RateLimitType.API_CALLS_PER_MINUTE,
                limit=30,
                window_seconds=60,
                tier=UserTier.FREE,
                endpoint_pattern=r"/api/.*"
            ),
            
            # Premium tier limits
            RateLimitRule(
                name="premium_requests_per_minute",
                limit_type=RateLimitType.REQUESTS_PER_MINUTE,
                limit=300,
                window_seconds=60,
                tier=UserTier.PREMIUM
            ),
            RateLimitRule(
                name="premium_requests_per_hour",
                limit_type=RateLimitType.REQUESTS_PER_HOUR,
                limit=10000,
                window_seconds=3600,
                tier=UserTier.PREMIUM
            ),
            RateLimitRule(
                name="premium_api_calls_per_minute",
                limit_type=RateLimitType.API_CALLS_PER_MINUTE,
                limit=150,
                window_seconds=60,
                tier=UserTier.PREMIUM,
                endpoint_pattern=r"/api/.*"
            ),
            
            # Enterprise tier limits
            RateLimitRule(
                name="enterprise_requests_per_minute",
                limit_type=RateLimitType.REQUESTS_PER_MINUTE,
                limit=1000,
                window_seconds=60,
                tier=UserTier.ENTERPRISE
            ),
            RateLimitRule(
                name="enterprise_requests_per_hour",
                limit_type=RateLimitType.REQUESTS_PER_HOUR,
                limit=50000,
                window_seconds=3600,
                tier=UserTier.ENTERPRISE
            ),
            
            # IP-based limits (for unauthenticated requests)
            RateLimitRule(
                name="ip_requests_per_minute",
                limit_type=RateLimitType.REQUESTS_PER_MINUTE,
                limit=30,
                window_seconds=60
            ),
            RateLimitRule(
                name="ip_requests_per_hour",
                limit_type=RateLimitType.REQUESTS_PER_HOUR,
                limit=500,
                window_seconds=3600
            )
        ]
        
        # Custom rules
        self.custom_rules: List[RateLimitRule] = []
        
        # Statistics
        self.stats = {
            'requests_processed': 0,
            'requests_blocked': 0,
            'requests_throttled': 0,
            'abuse_detected': 0,
            'unique_clients': 0,
            'rules_triggered': defaultdict(int)
        }
    
    def add_custom_rule(self, rule: RateLimitRule):
        """Add a custom rate limiting rule."""
        self.custom_rules.append(rule)
        self.logger.info(f"Added custom rate limit rule: {rule.name}")
    
    def remove_custom_rule(self, rule_name: str):
        """Remove a custom rate limiting rule."""
        self.custom_rules = [r for r in self.custom_rules if r.name != rule_name]
        self.logger.info(f"Removed custom rate limit rule: {rule_name}")
    
    async def check_rate_limit(self, request: Request, user_id: Optional[str] = None, 
                              user_tier: UserTier = UserTier.FREE) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Check if request should be rate limited."""
        self.stats['requests_processed'] += 1
        
        # Generate rate limit key
        key = self._generate_key(request, user_id)
        
        # Get current state
        state = self.states[key]
        current_time = datetime.utcnow()
        
        # Update state
        state.last_request_time = current_time
        if state.first_request_time is None:
            state.first_request_time = current_time
        state.total_requests += 1
        
        # Check if currently blocked
        if state.blocked_until and current_time < state.blocked_until:
            self.stats['requests_blocked'] += 1
            return False, {
                'error': 'rate_limit_exceeded',
                'message': 'Request blocked due to rate limiting',
                'blocked_until': state.blocked_until.isoformat(),
                'retry_after': int((state.blocked_until - current_time).total_seconds())
            }
        
        # Clear expired block
        if state.blocked_until and current_time >= state.blocked_until:
            state.blocked_until = None
        
        # Get applicable rules
        all_rules = self.default_rules + self.custom_rules
        applicable_rules = [rule for rule in all_rules if rule.matches_request(request, user_tier)]
        
        # Check each applicable rule
        for rule in applicable_rules:
            is_violated, violation_info = await self._check_rule(rule, state, request, current_time)
            
            if is_violated:
                self.stats['rules_triggered'][rule.name] += 1
                state.violations += 1
                
                # Determine action based on rule and violation history
                action = self._determine_action(rule, state, violation_info)
                
                if action['type'] == 'block':
                    state.blocked_until = current_time + timedelta(seconds=action['duration'])
                    self.stats['requests_blocked'] += 1
                    
                    # Record metrics
                    counter = self.metrics.get_counter('rate_limit_blocks_total')
                    counter.increment(1, rule=rule.name, tier=user_tier.value)
                    
                    return False, {
                        'error': 'rate_limit_exceeded',
                        'message': f'Rate limit exceeded for {rule.name}',
                        'rule': rule.name,
                        'limit': rule.limit,
                        'window_seconds': rule.window_seconds,
                        'retry_after': action['duration'],
                        'upgrade_available': user_tier == UserTier.FREE
                    }
                
                elif action['type'] == 'throttle':
                    self.stats['requests_throttled'] += 1
                    
                    # Add delay header for client to throttle
                    return True, {
                        'warning': 'rate_limit_warning',
                        'message': f'Approaching rate limit for {rule.name}',
                        'throttle_delay': action.get('delay', 1),
                        'upgrade_available': user_tier == UserTier.FREE
                    }
        
        # Check for abuse patterns
        abuse_detected = await self._check_abuse_patterns(key, state, request, current_time)
        if abuse_detected:
            self.stats['abuse_detected'] += 1
            return False, {
                'error': 'abuse_detected',
                'message': 'Suspicious activity detected',
                'contact_support': True
            }
        
        # Update concurrent requests
        state.concurrent_requests += 1
        
        # Record successful request
        counter = self.metrics.get_counter('rate_limit_requests_allowed_total')
        counter.increment(1, tier=user_tier.value)
        
        return True, None
    
    async def _check_rule(self, rule: RateLimitRule, state: RateLimitState, 
                         request: Request, current_time: datetime) -> Tuple[bool, Dict[str, Any]]:
        """Check if a specific rule is violated."""
        window_start = current_time - timedelta(seconds=rule.window_seconds)
        
        if rule.limit_type == RateLimitType.REQUESTS_PER_MINUTE:
            # Clean old requests
            while state.requests and state.requests[0] < window_start:
                state.requests.popleft()
            
            # Add current request
            state.requests.append(current_time)
            
            # Check limit
            if len(state.requests) > rule.limit:
                return True, {
                    'current_count': len(state.requests),
                    'limit': rule.limit,
                    'window_seconds': rule.window_seconds
                }
        
        elif rule.limit_type == RateLimitType.REQUESTS_PER_HOUR:
            # Similar logic for hourly limits
            while state.requests and state.requests[0] < window_start:
                state.requests.popleft()
            
            state.requests.append(current_time)
            
            if len(state.requests) > rule.limit:
                return True, {
                    'current_count': len(state.requests),
                    'limit': rule.limit,
                    'window_seconds': rule.window_seconds
                }
        
        elif rule.limit_type == RateLimitType.BANDWIDTH_PER_MINUTE:
            # Check bandwidth usage
            content_length = int(request.headers.get('content-length', 0))
            
            # Clean old bandwidth records
            while state.bandwidth_used and state.bandwidth_used[0][0] < window_start:
                state.bandwidth_used.popleft()
            
            # Add current bandwidth
            state.bandwidth_used.append((current_time, content_length))
            
            # Calculate total bandwidth
            total_bandwidth = sum(size for _, size in state.bandwidth_used)
            
            if total_bandwidth > rule.limit:
                return True, {
                    'current_bandwidth': total_bandwidth,
                    'limit': rule.limit,
                    'window_seconds': rule.window_seconds
                }
        
        elif rule.limit_type == RateLimitType.CONCURRENT_REQUESTS:
            if state.concurrent_requests >= rule.limit:
                return True, {
                    'current_concurrent': state.concurrent_requests,
                    'limit': rule.limit
                }
        
        return False, {}
    
    async def _check_abuse_patterns(self, key: str, state: RateLimitState, 
                                   request: Request, current_time: datetime) -> bool:
        """Check for abuse patterns."""
        for pattern in self.abuse_patterns:
            if await self._check_pattern(pattern, key, state, request, current_time):
                self.logger.warning(
                    f"Abuse pattern detected: {pattern.name}",
                    operation="abuse_detection",
                    key=key,
                    pattern=pattern.name
                )
                
                if pattern.action == "block":
                    state.blocked_until = current_time + timedelta(seconds=pattern.block_duration_seconds)
                    return True
        
        return False
    
    async def _check_pattern(self, pattern: AbusePattern, key: str, state: RateLimitState,
                            request: Request, current_time: datetime) -> bool:
        """Check a specific abuse pattern."""
        window_start = current_time - timedelta(seconds=pattern.window_seconds)
        
        if pattern.name == "rapid_requests":
            # Check for rapid requests
            recent_requests = [req for req in state.requests if req >= window_start]
            return len(recent_requests) > pattern.threshold
        
        elif pattern.name == "repeated_errors":
            # This would need to be tracked separately with response status codes
            # For now, we'll use a simplified check
            return state.violations > pattern.threshold
        
        elif pattern.name == "large_payloads":
            # Check for consistently large payloads
            content_length = int(request.headers.get('content-length', 0))
            if content_length > 1024 * 1024:  # 1MB
                recent_large = [size for _, size in state.bandwidth_used 
                               if _ >= window_start and size > 1024 * 1024]
                return len(recent_large) > pattern.threshold
        
        elif pattern.name == "suspicious_patterns":
            # Check for suspicious request patterns
            # This is a simplified implementation
            return state.violations > pattern.threshold // 2
        
        return False
    
    def _determine_action(self, rule: RateLimitRule, state: RateLimitState, 
                         violation_info: Dict[str, Any]) -> Dict[str, Any]:
        """Determine what action to take for a rate limit violation."""
        # Base action on violation severity and history
        violation_ratio = violation_info.get('current_count', 0) / rule.limit
        
        if violation_ratio > 2.0 or state.violations > 10:
            # Severe violation or repeat offender
            return {
                'type': 'block',
                'duration': min(3600, 60 * (state.violations + 1))  # Max 1 hour
            }
        elif violation_ratio > 1.5 or state.violations > 5:
            # Moderate violation
            return {
                'type': 'block',
                'duration': min(600, 30 * (state.violations + 1))  # Max 10 minutes
            }
        else:
            # Minor violation
            return {
                'type': 'throttle',
                'delay': min(5, state.violations + 1)
            }
    
    def _generate_key(self, request: Request, user_id: Optional[str] = None) -> str:
        """Generate a unique key for rate limiting."""
        if user_id:
            return f"user:{user_id}"
        
        # Use IP address for anonymous users
        client_ip = request.client.host if request.client else 'unknown'
        
        # Hash IP for privacy
        ip_hash = hashlib.sha256(client_ip.encode()).hexdigest()[:16]
        return f"ip:{ip_hash}"
    
    async def request_completed(self, request: Request, user_id: Optional[str] = None):
        """Mark a request as completed (for concurrent request tracking)."""
        key = self._generate_key(request, user_id)
        state = self.states[key]
        
        if state.concurrent_requests > 0:
            state.concurrent_requests -= 1
    
    def get_rate_limit_info(self, request: Request, user_id: Optional[str] = None,
                           user_tier: UserTier = UserTier.FREE) -> Dict[str, Any]:
        """Get current rate limit information for a key."""
        key = self._generate_key(request, user_id)
        state = self.states[key]
        current_time = datetime.utcnow()
        
        # Get applicable rules
        all_rules = self.default_rules + self.custom_rules
        applicable_rules = [rule for rule in all_rules if rule.matches_request(request, user_tier)]
        
        info = {
            'key': key,
            'total_requests': state.total_requests,
            'violations': state.violations,
            'blocked_until': state.blocked_until.isoformat() if state.blocked_until else None,
            'concurrent_requests': state.concurrent_requests,
            'limits': []
        }
        
        for rule in applicable_rules:
            window_start = current_time - timedelta(seconds=rule.window_seconds)
            
            if rule.limit_type in [RateLimitType.REQUESTS_PER_MINUTE, RateLimitType.REQUESTS_PER_HOUR]:
                recent_requests = [req for req in state.requests if req >= window_start]
                remaining = max(0, rule.limit - len(recent_requests))
                
                info['limits'].append({
                    'rule': rule.name,
                    'type': rule.limit_type.value,
                    'limit': rule.limit,
                    'used': len(recent_requests),
                    'remaining': remaining,
                    'reset_time': (window_start + timedelta(seconds=rule.window_seconds)).isoformat()
                })
        
        return info
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        return {
            **self.stats,
            'active_keys': len(self.states),
            'rules_configured': len(self.default_rules) + len(self.custom_rules),
            'abuse_patterns': len(self.abuse_patterns)
        }
    
    async def cleanup_expired_states(self):
        """Clean up expired rate limit states."""
        current_time = datetime.utcnow()
        expired_keys = []
        
        for key, state in self.states.items():
            # Remove states that haven't been used in 24 hours
            if (state.last_request_time and 
                (current_time - state.last_request_time).total_seconds() > 86400):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.states[key]
        
        if expired_keys:
            self.logger.info(f"Cleaned up {len(expired_keys)} expired rate limit states")
        
        return len(expired_keys)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware for FastAPI."""
    
    def __init__(self, app: ASGIApp, rate_limiter: Optional[RateLimiter] = None):
        super().__init__(app)
        self.rate_limiter = rate_limiter or RateLimiter()
        self.logger = get_logger(__name__, 'rate_limit_middleware')
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request through rate limiting middleware."""
        # Extract user information (this would come from authentication middleware)
        user_id = getattr(request.state, 'user_id', None)
        user_tier = UserTier(getattr(request.state, 'user_tier', 'free'))
        
        # Check rate limit
        allowed, limit_info = await self.rate_limiter.check_rate_limit(request, user_id, user_tier)
        
        if not allowed:
            # Request blocked by rate limiter
            self.logger.warning(
                "Request blocked by rate limiter",
                operation="rate_limit_block",
                user_id=user_id,
                client_ip=request.client.host if request.client else 'unknown',
                path=str(request.url.path)
            )
            
            response_data = {
                "error": limit_info,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Add upgrade prompt for free tier users
            if user_tier == UserTier.FREE and limit_info.get('upgrade_available'):
                response_data['upgrade_info'] = {
                    'message': 'Upgrade to Premium for higher rate limits',
                    'premium_limits': {
                        'requests_per_minute': 300,
                        'requests_per_hour': 10000
                    },
                    'upgrade_url': '/upgrade'
                }
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content=response_data,
                headers={
                    'Retry-After': str(limit_info.get('retry_after', 60)),
                    'X-RateLimit-Limit': str(limit_info.get('limit', 'unknown')),
                    'X-RateLimit-Remaining': '0',
                    'X-RateLimit-Reset': str(int(time.time()) + limit_info.get('retry_after', 60))
                }
            )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Add rate limit headers
            rate_limit_info = self.rate_limiter.get_rate_limit_info(request, user_id, user_tier)
            
            if rate_limit_info['limits']:
                primary_limit = rate_limit_info['limits'][0]  # Use first applicable limit
                response.headers['X-RateLimit-Limit'] = str(primary_limit['limit'])
                response.headers['X-RateLimit-Remaining'] = str(primary_limit['remaining'])
                response.headers['X-RateLimit-Reset'] = str(int(time.time()) + 60)  # Simplified
            
            # Add throttling warning if present
            if limit_info and limit_info.get('warning'):
                response.headers['X-RateLimit-Warning'] = limit_info['message']
            
            return response
        
        finally:
            # Mark request as completed
            await self.rate_limiter.request_completed(request, user_id)


# Utility functions
def create_rate_limiter(
    free_requests_per_minute: int = 60,
    premium_requests_per_minute: int = 300,
    enterprise_requests_per_minute: int = 1000
) -> RateLimiter:
    """Create a rate limiter with custom limits."""
    limiter = RateLimiter()
    
    # Update default rules
    for rule in limiter.default_rules:
        if rule.name == "free_requests_per_minute":
            rule.limit = free_requests_per_minute
        elif rule.name == "premium_requests_per_minute":
            rule.limit = premium_requests_per_minute
        elif rule.name == "enterprise_requests_per_minute":
            rule.limit = enterprise_requests_per_minute
    
    return limiter


def get_tier_limits(tier: UserTier) -> Dict[str, int]:
    """Get rate limits for a user tier."""
    limits = {
        UserTier.FREE: {
            'requests_per_minute': 60,
            'requests_per_hour': 1000,
            'api_calls_per_minute': 30
        },
        UserTier.PREMIUM: {
            'requests_per_minute': 300,
            'requests_per_hour': 10000,
            'api_calls_per_minute': 150
        },
        UserTier.ENTERPRISE: {
            'requests_per_minute': 1000,
            'requests_per_hour': 50000,
            'api_calls_per_minute': 500
        },
        UserTier.ADMIN: {
            'requests_per_minute': 10000,
            'requests_per_hour': 100000,
            'api_calls_per_minute': 1000
        }
    }
    
    return limits.get(tier, limits[UserTier.FREE])


async def start_rate_limiter_cleanup_task(rate_limiter: RateLimiter):
    """Start background cleanup task for rate limiter."""
    while True:
        try:
            await asyncio.sleep(3600)  # Run every hour
            cleaned = await rate_limiter.cleanup_expired_states()
            if cleaned > 0:
                logger = get_logger(__name__, 'rate_limiter_cleanup')
                logger.info(f"Cleaned up {cleaned} expired rate limit states")
        except Exception as e:
            logger = get_logger(__name__, 'rate_limiter_cleanup')
            logger.error(f"Error in rate limiter cleanup: {e}")


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter