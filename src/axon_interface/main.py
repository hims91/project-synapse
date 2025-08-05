"""
Main FastAPI Application for Project Synapse Axon Interface

This is the main entry point for the Project Synapse API, implementing
the complete Axon Interface with all core and specialized endpoints.
"""

import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn

try:
    from ..shared.config import get_settings
except ImportError:
    # Fallback for testing
    def get_settings():
        class MockSettings:
            environment = "development"
            class API:
                cors_origins = ["*"]
                allowed_hosts = ["*"]
            api = API()
            class Security:
                secret_key = "test-secret-key"
            security = Security()
        return MockSettings()
from .routers import content, search, scrape, monitoring, webhooks, websocket, admin
from .middleware import AuthenticationMiddleware, RateLimitMiddleware, LoggingMiddleware
from .webhooks import start_webhook_system


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Project Synapse Axon Interface")
    settings = get_settings()
    logger.info(f"Environment: {settings.environment}")
    
    # Start webhook system
    await start_webhook_system()
    logger.info("Webhook system initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Project Synapse Axon Interface")


# Create FastAPI application
app = FastAPI(
    title="Project Synapse API",
    description="The Definitive Blueprint v2.2 - Feel the web. Think in data. Act with insight.",
    version="2.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Get settings
settings = get_settings()

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.api.allowed_hosts
)

# Add custom middleware
app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(AuthenticationMiddleware)


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with consistent error format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": "http_error",
                "message": exc.detail,
                "status_code": exc.status_code
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions with consistent error format."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "type": "internal_error",
                "message": "An internal server error occurred"
            }
        }
    )


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "2.2.0",
        "timestamp": "2025-01-08T00:00:00Z",
        "components": {
            "api": {"status": "healthy"},
            "database": {"status": "healthy"},
            "nlp_engine": {"status": "healthy"},
            "search_engine": {"status": "healthy"}
        }
    }


# Include routers
app.include_router(content.router, prefix="/api/v1", tags=["Content"])
app.include_router(search.router, prefix="/api/v1", tags=["Search"])
app.include_router(scrape.router, prefix="/api/v1", tags=["Scraping"])
app.include_router(monitoring.router, prefix="/api/v1", tags=["Monitoring"])
app.include_router(webhooks.router, prefix="/api/v1/webhooks", tags=["Webhooks"])
app.include_router(websocket.router, prefix="", tags=["WebSocket"])
app.include_router(admin.router, prefix="", tags=["Admin"])


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Project Synapse API",
        "version": "2.2.0",
        "description": "The Definitive Blueprint v2.2 - Feel the web. Think in data. Act with insight.",
        "docs_url": "/docs",
        "health_url": "/health"
    }


if __name__ == "__main__":
    uvicorn.run(
        "src.axon_interface.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )