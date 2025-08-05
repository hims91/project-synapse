"""
Simple Central Hub Application for Deployment
Minimal FastAPI app for the Central Hub deployment without complex dependencies.
"""
import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from minimal_config import get_settings

# Initialize settings
settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    print(f"Starting Project Synapse Central Hub - {settings.environment}")
    print(f"Database configured: {'Yes' if settings.database_url else 'No'}")
    print(f"Redis configured: {'Yes' if settings.redis_url else 'No'}")
    yield
    print("Shutting down Central Hub...")

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="The Definitive Web Intelligence Network - Central Hub",
    version=settings.app_version,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint with basic system information."""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "operational",
        "motto": "Feel the web. Think in data. Act with insight.",
        "environment": settings.environment,
        "health": "/health",
        "docs": "/docs" if settings.debug else "disabled"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "central-hub",
        "version": settings.app_version,
        "environment": settings.environment,
        "database": "configured" if settings.database_url else "not_configured",
        "redis": "configured" if settings.redis_url else "not_configured",
        "debug": settings.debug
    }

@app.get("/api/v1/status")
async def api_status():
    """API status endpoint."""
    return {
        "api_version": settings.api_version,
        "service": "central-hub",
        "status": "operational",
        "environment": settings.environment,
        "endpoints": {
            "root": "/",
            "health": "/health",
            "status": "/api/v1/status",
            "docs": "/docs" if settings.debug else "disabled"
        }
    }

@app.post("/api/v1/scrape")
async def scrape_content(request: dict):
    """Placeholder scraping endpoint."""
    return {
        "message": "Central Hub received scraping request",
        "status": "accepted",
        "job_id": request.get("job_id", "unknown"),
        "note": "This is a placeholder - full implementation coming soon"
    }

@app.get("/api/v1/components")
async def get_components():
    """Get status of all distributed components."""
    return {
        "central_hub": {
            "status": "operational",
            "version": settings.app_version,
            "environment": settings.environment,
            "url": "https://synapse-central-hub.onrender.com"
        },
        "dendrites": {
            "status": "operational",
            "note": "Edge cache layer - deployed",
            "url": "https://synapse-dendrites.thehimzack.workers.dev"
        },
        "neurons": {
            "status": "operational", 
            "note": "Scraping workers - deployed",
            "url": "https://synapse-neurons.onrender.com"
        },
        "sensory_neurons": {
            "status": "ready_to_deploy",
            "note": "Learning scrapers - GitHub Actions setup required",
            "platform": "github_actions"
        },
        "spinal_cord": {
            "status": "not_deployed",
            "note": "Fallback system - deploy last",
            "platform": "vercel"
        }
    }

@app.post("/api/v1/callbacks/sensory")
async def sensory_callback(request: dict):
    """Receive callbacks from Sensory Neurons."""
    return {
        "status": "received",
        "message": "Sensory Neurons callback processed",
        "job_id": request.get("job_id"),
        "component": "central_hub",
        "timestamp": request.get("timestamp")
    }

@app.post("/api/v1/trigger/sensory")
async def trigger_sensory_neurons(request: dict):
    """Trigger Sensory Neurons via GitHub Actions (placeholder)."""
    url = request.get("url")
    priority = request.get("priority", "normal")
    
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")
    
    # In a full implementation, this would trigger GitHub Actions
    # For now, return a placeholder response
    return {
        "status": "triggered",
        "message": "Sensory Neurons job queued",
        "url": url,
        "priority": priority,
        "note": "This is a placeholder - full GitHub Actions integration coming soon",
        "estimated_completion": "2-5 minutes"
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 10000)),
        log_level=settings.log_level.lower()
    )