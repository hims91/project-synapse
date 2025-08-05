"""
Central Cortex - FastAPI Application
Layer 3: Cerebral Cortex

This module implements the main FastAPI application for Project Synapse.
The Central Cortex serves as the hub server coordinating all system components.
"""
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from ..shared.config import get_settings
from .middleware import (
    AuthenticationMiddleware,
    RateLimitingMiddleware,
    LoggingMiddleware,
    ErrorHandlingMiddleware
)
from .routers import health, auth, content, monitoring
from .dependencies import get_database_manager, get_task_dispatcher, get_fallback_manager

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events for the FastAPI application.
    """
    # Startup
    logger.info("Starting Project Synapse Central Cortex...")
    
    try:
        # Initialize core components
        settings = get_settings()
        
        # Initialize database
        db_manager = await get_database_manager()
        await db_manager.initialize()
        logger.info("Database initialized successfully")
        
        # Initialize task dispatcher
        task_dispatcher = await get_task_dispatcher()
        await task_dispatcher.start()
        logger.info("Task dispatcher started successfully")
        
        # Initialize fallback manager
        fallback_manager = await get_fallback_manager()
        await fallback_manager.start_monitoring()
        logger.info("Fallback manager started successfully")
        
        logger.info("Central Cortex startup completed successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start Central Cortex: {e}")
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down Project Synapse Central Cortex...")
        
        try:
            # Cleanup task dispatcher
            task_dispatcher = await get_task_dispatcher()
            await task_dispatcher.shutdown()
            logger.info("Task dispatcher shutdown completed")
            
            # Cleanup fallback manager
            fallback_manager = await get_fallback_manager()
            await fallback_manager.stop_monitoring()
            logger.info("Fallback manager shutdown completed")
            
            # Cleanup database
            db_manager = await get_database_manager()
            await db_manager.close()
            logger.info("Database connections closed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        
        logger.info("Central Cortex shutdown completed")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application instance
    """
    settings = get_settings()
    
    # Create FastAPI app with lifespan
    app = FastAPI(
        title="Project Synapse - Central Cortex",
        description="The Definitive Blueprint v2.2 - Hub Server",
        version="2.2.0",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        openapi_url="/openapi.json" if settings.debug else None,
        lifespan=lifespan
    )
    
    # Add security middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"] if settings.debug else ["localhost", "127.0.0.1"]
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        max_age=3600
    )
    
    # Add custom middleware (order matters - last added is executed first)
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(RateLimitingMiddleware)
    app.add_middleware(AuthenticationMiddleware)
    
    # Include routers
    app.include_router(
        health.router,
        prefix="/health",
        tags=["Health"]
    )
    
    app.include_router(
        auth.router,
        prefix="/auth",
        tags=["Authentication"]
    )
    
    app.include_router(
        content.router,
        prefix="/api/v1/content",
        tags=["Content"]
    )
    
    app.include_router(
        monitoring.router,
        prefix="/api/v1/monitoring",
        tags=["Monitoring"]
    )
    
    # Root endpoint
    @app.get("/", include_in_schema=False)
    async def root():
        """Root endpoint with basic system information."""
        return {
            "service": "Project Synapse - Central Cortex",
            "version": "2.2.0",
            "status": "operational",
            "motto": "Feel the web. Think in data. Act with insight.",
            "docs": "/docs" if settings.debug else "disabled",
            "health": "/health"
        }
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler for unhandled errors."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "type": "internal_server_error",
                    "message": "An internal server error occurred",
                    "request_id": getattr(request.state, "request_id", None)
                }
            }
        )
    
    return app


# Create the application instance
app = create_app()


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    workers: int = 1
):
    """
    Run the FastAPI server with uvicorn.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload for development
        workers: Number of worker processes
    """
    settings = get_settings()
    
    # Configure logging
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "root": {
            "level": settings.log_level,
            "handlers": ["default"],
        },
    }
    
    # Run server
    uvicorn.run(
        "src.central_cortex.app:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
        log_config=log_config,
        access_log=settings.debug
    )


if __name__ == "__main__":
    # Development server
    settings = get_settings()
    run_server(
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        workers=1
    )