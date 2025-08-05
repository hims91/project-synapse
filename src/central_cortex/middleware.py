"""
Central Cortex - Middleware Components
Layer 3: Cerebral Cortex

This module implements custom middleware for the FastAPI application.
Provides authentication, rate limiting, logging, and error handling.
"""
import time
import uuid
import logging
from typing import Callable, Dict, Any, Optional
from datetime import datetime, timedelta

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import redis.asyncio as redis

from ..shared.config import get_settings

logger = logging.getLogger(__name__)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware for API key validation.
    Validates API keys and sets user context for requests.
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.settings = get_settings()
        self.public_paths = {
            "/", "/health", "/health/", "/health/ready", "/health/live",
            "/docs", "/redoc", "/openapi.json"
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process authentication for incoming requests."""
        # Skip authentication for public paths
        if request.url.path in self.public_paths:
            return await call_next(request)
        
        # Skip authentication for OPTIONS requests (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)
        
        # Extract API key from header
        api_key = request.headers.get("X-API-Key") or request.headers.get("Authorization")
        
        if api_key and api_key.startswith("Bearer "):
            api_key = api_key[7:]  # Remove "Bearer " prefix
        
        if not api_key:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": {
                        "type": "authentication_required",
                        "message": "API key is required"
                    }
                }
            )
        
        # Validate API key (simplified for now)
        user_info = await self._validate_api_key(api_key)
        
        if not user_info:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": {
                        "type": "invalid_api_key",
                        "message": "Invalid API key"
                    }
                }
            )
        
        # Set user context
        request.state.user = user_info
        request.state.api_key = api_key
        
        return await call_next(request)
    
    async def _validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """
        Validate API key and return user information.
        
        Args:
            api_key: API key to validate
            
        Returns:
            User information if valid, None otherwise
        """
        try:
            # In a real implementation, this would query the database
            # For now, we'll use a simple validation
            if api_key == "dev_api_key_12345":
                return {
                    "user_id": "dev_user",
                    "tier": "premium",
                    "rate_limit": 1000,
                    "is_active": True
                }
            
            # TODO: Implement proper database lookup
            # from ..synaptic_vesicle.repositories import UserRepository
            # user_repo = UserRepository(session)
            # user = await user_repo.get_by_api_key(api_key)
            # return user if user and user.is_active else None
            
            return None
            
        except Exception as e:
            logger.error(f"Error validating API key: {e}")
            return None


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using Redis for distributed rate limiting.
    Implements sliding window rate limiting per API key.
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.settings = get_settings()
        self.redis_client: Optional[redis.Redis] = None
        self.default_rate_limit = 100  # requests per minute
        self.window_size = 60  # seconds
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process rate limiting for incoming requests."""
        # Skip rate limiting for health checks
        if request.url.path.startswith("/health"):
            return await call_next(request)
        
        # Get user info from authentication middleware
        user_info = getattr(request.state, "user", None)
        api_key = getattr(request.state, "api_key", None)
        
        if not user_info or not api_key:
            # If no user info, apply default rate limiting by IP
            client_ip = self._get_client_ip(request)
            rate_limit_key = f"rate_limit:ip:{client_ip}"
            rate_limit = self.default_rate_limit
        else:
            # Apply user-specific rate limiting
            rate_limit_key = f"rate_limit:user:{user_info['user_id']}"
            rate_limit = user_info.get("rate_limit", self.default_rate_limit)
        
        # Check rate limit
        is_allowed, remaining, reset_time = await self._check_rate_limit(
            rate_limit_key, rate_limit
        )
        
        if not is_allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": {
                        "type": "rate_limit_exceeded",
                        "message": f"Rate limit of {rate_limit} requests per minute exceeded",
                        "retry_after": reset_time
                    }
                },
                headers={
                    "X-RateLimit-Limit": str(rate_limit),
                    "X-RateLimit-Remaining": str(remaining),
                    "X-RateLimit-Reset": str(reset_time),
                    "Retry-After": str(reset_time)
                }
            )
        
        # Add rate limit headers to response
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(rate_limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_time)
        
        return response
    
    async def _check_rate_limit(
        self, 
        key: str, 
        limit: int
    ) -> tuple[bool, int, int]:
        """
        Check rate limit using sliding window algorithm.
        
        Args:
            key: Rate limit key
            limit: Request limit per window
            
        Returns:
            Tuple of (is_allowed, remaining_requests, reset_time)
        """
        try:
            # Initialize Redis client if needed
            if not self.redis_client:
                self.redis_client = redis.from_url(
                    self.settings.redis_url,
                    decode_responses=True
                )
            
            current_time = int(time.time())
            window_start = current_time - self.window_size
            
            # Use Redis pipeline for atomic operations
            pipe = self.redis_client.pipeline()
            
            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count current requests
            pipe.zcard(key)
            
            # Add current request
            pipe.zadd(key, {str(uuid.uuid4()): current_time})
            
            # Set expiration
            pipe.expire(key, self.window_size)
            
            results = await pipe.execute()
            current_count = results[1] + 1  # +1 for the request we just added
            
            is_allowed = current_count <= limit
            remaining = max(0, limit - current_count)
            reset_time = current_time + self.window_size
            
            return is_allowed, remaining, reset_time
            
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            # Allow request on Redis error
            return True, limit, int(time.time()) + self.window_size
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers (reverse proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct connection
        return request.client.host if request.client else "unknown"


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Logging middleware for request/response logging.
    Provides structured logging with request IDs and performance metrics.
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.settings = get_settings()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process logging for incoming requests."""
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Start timing
        start_time = time.time()
        
        # Extract request info
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("User-Agent", "unknown")
        user_info = getattr(request.state, "user", None)
        user_id = user_info.get("user_id") if user_info else None
        
        # Log request
        logger.info(
            f"Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "client_ip": client_ip,
                "user_agent": user_agent,
                "user_id": user_id
            }
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log response
            logger.info(
                f"Request completed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "url": str(request.url),
                    "status_code": response.status_code,
                    "duration_ms": round(duration * 1000, 2),
                    "user_id": user_id
                }
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            # Calculate duration
            duration = time.time() - start_time
            
            # Log error
            logger.error(
                f"Request failed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "url": str(request.url),
                    "duration_ms": round(duration * 1000, 2),
                    "error": str(e),
                    "user_id": user_id
                },
                exc_info=True
            )
            
            raise
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers (reverse proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct connection
        return request.client.host if request.client else "unknown"


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Error handling middleware for consistent error responses.
    Catches and formats exceptions into standardized error responses.
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.settings = get_settings()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process error handling for incoming requests."""
        try:
            return await call_next(request)
            
        except HTTPException as e:
            # FastAPI HTTP exceptions - pass through
            raise
            
        except ValueError as e:
            # Validation errors
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "error": {
                        "type": "validation_error",
                        "message": str(e),
                        "request_id": getattr(request.state, "request_id", None)
                    }
                }
            )
            
        except PermissionError as e:
            # Permission errors
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={
                    "error": {
                        "type": "permission_denied",
                        "message": str(e),
                        "request_id": getattr(request.state, "request_id", None)
                    }
                }
            )
            
        except Exception as e:
            # Unexpected errors
            logger.error(f"Unhandled exception in middleware: {e}", exc_info=True)
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": {
                        "type": "internal_server_error",
                        "message": "An internal server error occurred" if not self.settings.debug else str(e),
                        "request_id": getattr(request.state, "request_id", None)
                    }
                }
            )