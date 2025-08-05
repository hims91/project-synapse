"""
Security module for Project Synapse.

Provides comprehensive security features including rate limiting,
abuse prevention, request protection, and security middleware.
"""

from .rate_limiter import (
    RateLimiter,
    RateLimitMiddleware,
    UserTier,
    RateLimitType,
    RateLimitRule,
    create_rate_limiter,
    get_tier_limits,
    get_rate_limiter,
    start_rate_limiter_cleanup_task
)

from .protection_middleware import (
    SecurityConfig,
    ThreatDetector,
    RequestValidator,
    RequestSigner,
    SecurityHeadersMiddleware,
    CORSMiddleware,
    ComprehensiveProtectionMiddleware,
    create_security_config,
    create_protection_middleware,
    get_security_config,
    get_protection_middleware
)

from .abuse_prevention import (
    AbusePreventionSystem,
    AbuseType,
    ActionType,
    AbuseEvent,
    ClientBehavior,
    AbuseRule,
    get_abuse_prevention_system,
    start_abuse_prevention_cleanup_task
)

__all__ = [
    # Rate Limiting
    'RateLimiter',
    'RateLimitMiddleware', 
    'UserTier',
    'RateLimitType',
    'RateLimitRule',
    'create_rate_limiter',
    'get_tier_limits',
    'get_rate_limiter',
    'start_rate_limiter_cleanup_task',
    
    # Protection Middleware
    'SecurityConfig',
    'ThreatDetector',
    'RequestValidator',
    'RequestSigner',
    'SecurityHeadersMiddleware',
    'CORSMiddleware',
    'ComprehensiveProtectionMiddleware',
    'create_security_config',
    'create_protection_middleware',
    'get_security_config',
    'get_protection_middleware',
    
    # Abuse Prevention
    'AbusePreventionSystem',
    'AbuseType',
    'ActionType',
    'AbuseEvent',
    'ClientBehavior',
    'AbuseRule',
    'get_abuse_prevention_system',
    'start_abuse_prevention_cleanup_task'
]