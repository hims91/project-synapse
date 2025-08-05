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

@app.get("/api/v1/test/system")
async def test_system_integration():
    """Comprehensive system integration test."""
    import httpx
    import asyncio
    from datetime import datetime
    
    test_results = {
        "test_id": f"system-test-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "components": {},
        "integration_tests": {},
        "overall_status": "testing"
    }
    
    async with httpx.AsyncClient(timeout=30) as client:
        # Test 1: Dendrites Health
        try:
            response = await client.get("https://synapse-dendrites.thehimzack.workers.dev/health")
            test_results["components"]["dendrites"] = {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "response_time": response.elapsed.total_seconds() if hasattr(response, 'elapsed') else 0,
                "data": response.json() if response.status_code == 200 else None
            }
        except Exception as e:
            test_results["components"]["dendrites"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Test 2: Neurons Health
        try:
            response = await client.get("https://synapse-neurons.onrender.com/health")
            test_results["components"]["neurons"] = {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "response_time": response.elapsed.total_seconds() if hasattr(response, 'elapsed') else 0,
                "data": response.json() if response.status_code == 200 else None
            }
        except Exception as e:
            test_results["components"]["neurons"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Test 3: Dendrites Proxy (Edge Cache)
        try:
            response = await client.get("https://synapse-dendrites.thehimzack.workers.dev/proxy?path=/health")
            test_results["integration_tests"]["dendrites_proxy"] = {
                "status": "working" if response.status_code == 200 else "failed",
                "cache_header": response.headers.get("X-Cache", "unknown"),
                "data": response.json() if response.status_code == 200 else None
            }
        except Exception as e:
            test_results["integration_tests"]["dendrites_proxy"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Test 4: Neurons Scraping Test
        try:
            test_job = {
                "job_id": f"integration-test-{datetime.now().strftime('%H%M%S')}",
                "url": "https://httpbin.org/html",
                "options": {"extract_text": True, "extract_title": True}
            }
            response = await client.post("https://synapse-neurons.onrender.com/scrape", json=test_job)
            test_results["integration_tests"]["neurons_scraping"] = {
                "status": "working" if response.status_code == 200 else "failed",
                "job_accepted": response.status_code == 200,
                "data": response.json() if response.status_code == 200 else None
            }
        except Exception as e:
            test_results["integration_tests"]["neurons_scraping"] = {
                "status": "error",
                "error": str(e)
            }
    
    # Determine overall status
    component_statuses = [comp.get("status") for comp in test_results["components"].values()]
    integration_statuses = [test.get("status") for test in test_results["integration_tests"].values()]
    
    all_healthy = all(status == "healthy" for status in component_statuses)
    all_working = all(status == "working" for status in integration_statuses)
    
    if all_healthy and all_working:
        test_results["overall_status"] = "all_systems_operational"
    elif any(status == "error" for status in component_statuses + integration_statuses):
        test_results["overall_status"] = "errors_detected"
    else:
        test_results["overall_status"] = "partial_functionality"
    
    test_results["summary"] = {
        "components_tested": len(test_results["components"]),
        "components_healthy": sum(1 for comp in test_results["components"].values() if comp.get("status") == "healthy"),
        "integration_tests": len(test_results["integration_tests"]),
        "integration_working": sum(1 for test in test_results["integration_tests"].values() if test.get("status") == "working"),
        "recommendation": "All systems operational" if test_results["overall_status"] == "all_systems_operational" else "Check failed components"
    }
    
    return test_results

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 10000)),
        log_level=settings.log_level.lower()
    )