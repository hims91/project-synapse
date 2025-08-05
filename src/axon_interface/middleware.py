"""
Custom middleware for Project Synapse API

Implements authentication, rate limiting, and logging middleware
for the Axon Interface.
"""

import time
import logging
from typing import Dict, Optional
from collections import defaultdict, deque

from fastapi import Request, Response, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from ..shared.config import get_settings


logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Authentication middleware for API key validation."""
    
    def __init__(self, app):
        super().__init__(app)
        self.settings = get_settings()
        # In a real implementation, this would connect to the database
        # For now, we'll use a simple in-memory store
        self.valid_api_keys = {
            "test-api-key-123": {
                "user_id": "test-user",
                "tier": "premium",
                "rate_limit": 1000
            }
        }
    
    async def dispatch(self, request: Request, call_next):
        """Process authentication for each request."""
        # Skip authentication for health check and docs
        if request.url.path in ["/health", "/docs", "/redoc", "/openapi.json", "/"]:
            return await call_next(request)
        
        # Extract API key from Authorization header
        authorization = request.headers.get("Authorization")
        if not authorization:
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "type": "authentication_error",
                        "message": "API key required"
                    }
                }
            )
        
        try:
            scheme, api_key = authorization.split(" ", 1)
            if scheme.lower() != "bearer":
                raise ValueError("Invalid scheme")
        except ValueError:
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "type": "authentication_error",
                        "message": "Invalid authorization format. Use 'Bearer <api_key>'"
                    }
                }
            )
        
        # Validate API key
        user_info = self.valid_api_keys.get(api_key)
        if not user_info:
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "type": "authentication_error",
                        "message": "Invalid API key"
                    }
                }
            )
        
        # Add user info to request state
        request.state.user_id = user_info["user_id"]
        request.state.user_tier = user_info["tier"]
        request.state.rate_limit = user_info["rate_limit"]
        
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware based on user tier."""
    
    def __init__(self, app):
        super().__init__(app)
        self.request_counts: Dict[str, deque] = defaultdict(deque)
        self.window_size = 3600  # 1 hour window
    
    async def dispatch(self, request: Request, call_next):
        """Process rate limiting for each request."""
        # Skip rate limiting for health check and docs
        if request.url.path in ["/health", "/docs", "/redoc", "/openapi.json", "/"]:
            return await call_next(request)
        
        # Get user info from authentication middleware
        user_id = getattr(request.state, "user_id", None)
        rate_limit = getattr(request.state, "rate_limit", 100)  # Default limit
        
        if not user_id:
            return await call_next(request)
        
        current_time = time.time()
        user_requests = self.request_counts[user_id]
        
        # Remove old requests outside the window
        while user_requests and user_requests[0] < current_time - self.window_size:
            user_requests.popleft()
        
        # Check rate limit
        if len(user_requests) >= rate_limit:
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "type": "rate_limit_error",
                        "message": f"Rate limit exceeded. Maximum {rate_limit} requests per hour."
                    }
                },
                headers={
                    "X-RateLimit-Limit": str(rate_limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(current_time + self.window_size))
                }
            )
        
        # Add current request
        user_requests.append(current_time)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(rate_limit)
        response.headers["X-RateLimit-Remaining"] = str(rate_limit - len(user_requests))
        response.headers["X-RateLimit-Reset"] = str(int(current_time + self.window_size))
        
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Logging middleware for request/response tracking."""
    
    async def dispatch(self, request: Request, call_next):
        """Log request and response information."""
        start_time = time.time()
        
        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        
        # Process request
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Log response
        logger.info(
            f"Response: {response.status_code} "
            f"in {process_time:.3f}s"
        )
        
        # Add processing time header
        response.headers["X-Process-Time"] = str(process_time)
        
        return response