"""
Example of integrating all security features in Project Synapse.

This example shows how to set up comprehensive security including:
- Rate limiting with tier-based limits
- Abuse prevention with behavioral analysis
- Request protection with threat detection
- Security headers and CORS
"""

import asyncio
from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Import security components
from src.shared.security import (
    # Rate limiting
    RateLimiter,
    RateLimitMiddleware,
    UserTier,
    get_rate_limiter,
    start_rate_limiter_cleanup_task,
    
    # Protection middleware
    SecurityConfig,
    ComprehensiveProtectionMiddleware,
    SecurityHeadersMiddleware,
    create_security_config,
    
    # Abuse prevention
    AbusePreventionSystem,
    get_abuse_prevention_system,
    start_abuse_prevention_cleanup_task
)


# Background tasks for cleanup
async def start_background_tasks():
    """Start background cleanup tasks."""
    # Start rate limiter cleanup
    asyncio.create_task(start_rate_limiter_cleanup_task(get_rate_limiter()))
    
    # Start abuse prevention cleanup
    asyncio.create_task(start_abuse_prevention_cleanup_task())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    await start_background_tasks()
    yield
    # Shutdown - cleanup would go here


# Create FastAPI app with lifespan
app = FastAPI(
    title="Project Synapse - Security Example",
    description="Example of comprehensive security integration",
    version="1.0.0",
    lifespan=lifespan
)


# Security configuration
security_config = create_security_config(
    cors_origins=["http://localhost:3000", "https://yourdomain.com"],
    enable_request_signing=False,  # Enable in production
    enable_ip_filtering=False,     # Enable if needed
    blocked_user_agents=[
        r".*bot.*",
        r".*crawler.*",
        r".*scraper.*"
    ]
)


# Add security middleware (order matters!)
# 1. CORS (first)
app.add_middleware(
    CORSMiddleware,
    allow_origins=security_config.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Security headers
app.add_middleware(SecurityHeadersMiddleware, config=security_config)

# 3. Comprehensive protection (threat detection, validation, etc.)
app.add_middleware(ComprehensiveProtectionMiddleware, config=security_config)

# 4. Rate limiting (last, so it can see processed request)
app.add_middleware(RateLimitMiddleware, rate_limiter=get_rate_limiter())


# Authentication dependency (simplified for example)
async def get_current_user(request: Request) -> dict:
    """Get current user from request (simplified)."""
    # In real implementation, this would validate JWT tokens, API keys, etc.
    api_key = request.headers.get("X-API-Key")
    user_id = request.headers.get("X-User-ID")
    
    if api_key == "premium_key":
        return {
            "user_id": user_id or "premium_user",
            "tier": UserTier.PREMIUM,
            "authenticated": True
        }
    elif api_key == "enterprise_key":
        return {
            "user_id": user_id or "enterprise_user", 
            "tier": UserTier.ENTERPRISE,
            "authenticated": True
        }
    else:
        return {
            "user_id": None,
            "tier": UserTier.FREE,
            "authenticated": False
        }


# Middleware to set user context for rate limiting
@app.middleware("http")
async def set_user_context(request: Request, call_next):
    """Set user context for rate limiting and abuse prevention."""
    user = await get_current_user(request)
    
    # Set user context for rate limiting
    request.state.user_id = user["user_id"]
    request.state.user_tier = user["tier"]
    request.state.authenticated = user["authenticated"]
    
    response = await call_next(request)
    return response


# API endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Project Synapse Security Example", "status": "protected"}


@app.get("/api/content")
async def get_content(
    request: Request,
    user: dict = Depends(get_current_user)
):
    """Get content with rate limiting and abuse prevention."""
    # This endpoint is automatically protected by middleware
    
    # Simulate content retrieval
    content = {
        "articles": [
            {"id": 1, "title": "Sample Article 1", "content": "..."},
            {"id": 2, "title": "Sample Article 2", "content": "..."}
        ],
        "user_tier": user["tier"],
        "rate_limit_info": get_rate_limiter().get_rate_limit_info(
            request, user["user_id"], user["tier"]
        )
    }
    
    return content


@app.post("/api/scrape")
async def scrape_url(
    request: Request,
    data: dict,
    user: dict = Depends(get_current_user)
):
    """Scrape URL endpoint with enhanced protection."""
    # Validate input
    if "url" not in data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="URL is required"
        )
    
    url = data["url"]
    
    # Additional validation for scraping requests
    if not url.startswith(("http://", "https://")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid URL format"
        )
    
    # Simulate scraping (would integrate with actual scraping system)
    result = {
        "job_id": "job_123",
        "url": url,
        "status": "queued",
        "user_tier": user["tier"],
        "estimated_completion": "30 seconds"
    }
    
    return result


@app.get("/api/search")
async def search_content(
    request: Request,
    q: str,
    user: dict = Depends(get_current_user)
):
    """Search content with rate limiting."""
    # Simulate search
    results = {
        "query": q,
        "results": [
            {"id": 1, "title": f"Result for {q}", "relevance": 0.95},
            {"id": 2, "title": f"Another result for {q}", "relevance": 0.87}
        ],
        "total": 2,
        "user_tier": user["tier"]
    }
    
    return results


@app.get("/admin/security/stats")
async def get_security_stats(
    request: Request,
    user: dict = Depends(get_current_user)
):
    """Get security statistics (admin only)."""
    # Check admin access
    if user["tier"] != UserTier.ENTERPRISE:  # Simplified admin check
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    # Get statistics from security systems
    rate_limiter = get_rate_limiter()
    abuse_system = get_abuse_prevention_system()
    
    stats = {
        "rate_limiting": rate_limiter.get_stats(),
        "abuse_prevention": abuse_system.get_stats(),
        "timestamp": "2024-01-01T00:00:00Z"
    }
    
    return stats


@app.get("/admin/security/client/{client_id}")
async def get_client_info(
    client_id: str,
    request: Request,
    user: dict = Depends(get_current_user)
):
    """Get information about a specific client."""
    # Check admin access
    if user["tier"] != UserTier.ENTERPRISE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    abuse_system = get_abuse_prevention_system()
    client_info = abuse_system.get_client_info(client_id)
    
    if not client_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Client not found"
        )
    
    return client_info


@app.post("/admin/security/whitelist/{client_id}")
async def whitelist_client(
    client_id: str,
    request: Request,
    user: dict = Depends(get_current_user)
):
    """Whitelist a client (remove from blocks)."""
    # Check admin access
    if user["tier"] != UserTier.ENTERPRISE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    abuse_system = get_abuse_prevention_system()
    abuse_system.whitelist_client(client_id)
    
    return {"message": f"Client {client_id} has been whitelisted"}


# Health check endpoint (bypasses some security for monitoring)
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "security": "enabled",
        "timestamp": "2024-01-01T00:00:00Z"
    }


# Error handlers
@app.exception_handler(429)
async def rate_limit_handler(request: Request, exc: HTTPException):
    """Custom rate limit error handler."""
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "message": "Too many requests. Please slow down or upgrade your plan.",
            "upgrade_info": {
                "current_tier": "free",
                "upgrade_benefits": [
                    "Higher rate limits",
                    "Priority processing",
                    "Advanced features"
                ],
                "upgrade_url": "/upgrade"
            }
        },
        headers={
            "Retry-After": "60",
            "X-RateLimit-Limit": "60",
            "X-RateLimit-Remaining": "0"
        }
    )


@app.exception_handler(403)
async def security_block_handler(request: Request, exc: HTTPException):
    """Custom security block error handler."""
    return JSONResponse(
        status_code=403,
        content={
            "error": "Request blocked",
            "message": "Your request was blocked by our security system.",
            "support": "If you believe this is an error, please contact support.",
            "reference_id": getattr(request.state, 'correlation_id', 'unknown')
        }
    )


# Example of custom abuse rule
def setup_custom_abuse_rules():
    """Set up custom abuse detection rules."""
    from src.shared.security.abuse_prevention import AbuseRule, AbuseType, ActionType
    
    abuse_system = get_abuse_prevention_system()
    
    # Custom rule for API abuse
    api_abuse_rule = AbuseRule(
        name="api_heavy_usage",
        abuse_type=AbuseType.API_ABUSE,
        description="Heavy API usage pattern",
        condition=lambda behavior: (
            len([ep for ep in behavior.endpoints_accessed if '/api/' in ep]) > 20 and
            behavior.get_request_rate() > 5
        ),
        severity=6,
        action=ActionType.THROTTLE,
        cooldown_seconds=300
    )
    
    abuse_system.add_custom_rule(api_abuse_rule)


# Initialize custom rules
setup_custom_abuse_rules()


if __name__ == "__main__":
    import uvicorn
    
    print("Starting Project Synapse Security Example...")
    print("Security features enabled:")
    print("- Rate limiting with tier-based limits")
    print("- Abuse prevention with behavioral analysis") 
    print("- Request protection with threat detection")
    print("- Security headers and CORS")
    print("- Comprehensive logging and monitoring")
    print()
    print("Test endpoints:")
    print("- GET / - Root endpoint")
    print("- GET /api/content - Content API (rate limited)")
    print("- POST /api/scrape - Scraping API (enhanced protection)")
    print("- GET /api/search?q=test - Search API")
    print("- GET /admin/security/stats - Security statistics (admin)")
    print("- GET /health - Health check")
    print()
    print("Test with different API keys:")
    print("- No key: Free tier (60 req/min)")
    print("- X-API-Key: premium_key - Premium tier (300 req/min)")
    print("- X-API-Key: enterprise_key - Enterprise tier (1000 req/min)")
    
    uvicorn.run(
        "security_integration_example:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )