"""
Simple Neurons Application for Deployment
Lightweight scraping workers for Project Synapse.
"""
import os
import asyncio
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import httpx
from bs4 import BeautifulSoup
import uvicorn
from minimal_config import get_settings

# Initialize settings
settings = get_settings()

# Global job tracking
active_jobs = {}
job_stats = {
    "total_jobs": 0,
    "completed_jobs": 0,
    "failed_jobs": 0,
    "start_time": time.time()
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    print(f"Starting {settings.app_name} - {settings.environment}")
    print(f"Hub URL: {settings.hub_url}")
    print(f"API Key configured: {'Yes' if settings.neurons_api_key else 'No'}")
    yield
    print("Shutting down Neurons...")

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="Lightweight scraping workers for Project Synapse",
    version=settings.app_version,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": settings.app_name,
        "component": "neurons",
        "version": settings.app_version,
        "status": "operational",
        "environment": settings.environment,
        "hub_url": settings.hub_url,
        "endpoints": {
            "health": "/health",
            "scrape": "/scrape",
            "status": "/status",
            "jobs": "/jobs"
        },
        "capabilities": [
            "HTTP scraping",
            "HTML parsing",
            "Background job processing",
            "Hub integration"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "component": "neurons",
        "version": settings.app_version,
        "environment": settings.environment,
        "hub_connected": bool(settings.neurons_api_key),
        "active_jobs": len(active_jobs),
        "uptime": time.time() - job_stats["start_time"]
    }

@app.post("/scrape")
async def scrape_content(
    background_tasks: BackgroundTasks,
    request: Dict[str, Any]
):
    """
    Scrape content from a URL.
    
    Expected request format:
    {
        "job_id": "unique_job_id",
        "url": "https://example.com/article",
        "callback_url": "https://hub.com/callback",
        "options": {
            "extract_text": true,
            "extract_links": false,
            "timeout": 30
        }
    }
    """
    try:
        job_id = request.get("job_id")
        url = request.get("url")
        callback_url = request.get("callback_url")
        options = request.get("options", {})
        
        if not job_id or not url:
            raise HTTPException(status_code=400, detail="job_id and url are required")
        
        # Check if job already exists
        if job_id in active_jobs:
            raise HTTPException(status_code=409, detail="Job ID already exists")
        
        # Add job to tracking
        active_jobs[job_id] = {
            "url": url,
            "status": "queued",
            "created_at": time.time()
        }
        job_stats["total_jobs"] += 1
        
        # Start scraping in background
        background_tasks.add_task(
            perform_scraping,
            job_id=job_id,
            url=url,
            callback_url=callback_url,
            options=options
        )
        
        return {
            "status": "accepted",
            "job_id": job_id,
            "message": "Scraping job queued",
            "estimated_completion": "30-60 seconds"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def perform_scraping(
    job_id: str,
    url: str,
    callback_url: Optional[str] = None,
    options: Dict[str, Any] = None
):
    """Perform the actual scraping work."""
    if options is None:
        options = {}
    
    try:
        # Update job status
        active_jobs[job_id]["status"] = "scraping"
        active_jobs[job_id]["started_at"] = time.time()
        
        print(f"Starting scraping job {job_id} for URL: {url}")
        
        # Perform HTTP request
        timeout = options.get("timeout", settings.request_timeout)
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(
                url,
                headers={
                    "User-Agent": settings.user_agent,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
                },
                follow_redirects=True
            )
            response.raise_for_status()
        
        # Parse content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract data based on options
        result = {
            "url": url,
            "status_code": response.status_code,
            "content_type": response.headers.get("content-type", ""),
            "scraped_at": time.time()
        }
        
        if options.get("extract_text", True):
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            result["text"] = soup.get_text(strip=True)[:10000]  # Limit to 10KB
        
        if options.get("extract_title", True):
            title_tag = soup.find("title")
            result["title"] = title_tag.get_text(strip=True) if title_tag else ""
        
        if options.get("extract_links", False):
            links = []
            for link in soup.find_all("a", href=True)[:50]:  # Limit to 50 links
                links.append({
                    "url": link["href"],
                    "text": link.get_text(strip=True)[:100]
                })
            result["links"] = links
        
        if options.get("extract_images", False):
            images = []
            for img in soup.find_all("img", src=True)[:20]:  # Limit to 20 images
                images.append({
                    "src": img["src"],
                    "alt": img.get("alt", "")[:100]
                })
            result["images"] = images
        
        # Update job status
        active_jobs[job_id]["status"] = "completed"
        active_jobs[job_id]["completed_at"] = time.time()
        job_stats["completed_jobs"] += 1
        
        # Send callback if specified
        if callback_url:
            await send_callback(callback_url, {
                "job_id": job_id,
                "status": "completed",
                "result": result
            })
        
        print(f"Scraping job {job_id} completed successfully")
        
        # Clean up job after 5 minutes
        await asyncio.sleep(300)
        active_jobs.pop(job_id, None)
        
    except Exception as e:
        print(f"Scraping job {job_id} failed: {e}")
        
        # Update job status
        if job_id in active_jobs:
            active_jobs[job_id]["status"] = "failed"
            active_jobs[job_id]["error"] = str(e)
            active_jobs[job_id]["failed_at"] = time.time()
        
        job_stats["failed_jobs"] += 1
        
        # Send error callback if specified
        if callback_url:
            await send_callback(callback_url, {
                "job_id": job_id,
                "status": "failed",
                "error": str(e)
            })

async def send_callback(callback_url: str, data: Dict[str, Any]):
    """Send callback to the specified URL."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                callback_url,
                json=data,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": settings.user_agent
                }
            )
            response.raise_for_status()
            print(f"Callback sent successfully to {callback_url}")
    except Exception as e:
        print(f"Failed to send callback to {callback_url}: {e}")

@app.get("/jobs")
async def get_jobs():
    """Get current job status."""
    return {
        "active_jobs": len(active_jobs),
        "jobs": {
            job_id: {
                "url": job_data["url"],
                "status": job_data["status"],
                "created_at": job_data["created_at"]
            }
            for job_id, job_data in active_jobs.items()
        }
    }

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a specific job."""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "job_id": job_id,
        **active_jobs[job_id]
    }

@app.get("/status")
async def get_status():
    """Get component status and metrics."""
    uptime = time.time() - job_stats["start_time"]
    
    return {
        "component": "neurons",
        "status": "operational",
        "environment": settings.environment,
        "version": settings.app_version,
        "uptime_seconds": uptime,
        "metrics": {
            "total_jobs": job_stats["total_jobs"],
            "completed_jobs": job_stats["completed_jobs"],
            "failed_jobs": job_stats["failed_jobs"],
            "active_jobs": len(active_jobs),
            "success_rate": (
                job_stats["completed_jobs"] / max(job_stats["total_jobs"], 1)
            ) * 100,
            "jobs_per_hour": (
                job_stats["total_jobs"] / max(uptime / 3600, 1)
            )
        },
        "configuration": {
            "concurrent_jobs": settings.concurrent_jobs,
            "request_timeout": settings.request_timeout,
            "max_retries": settings.max_retries,
            "requests_per_minute": settings.requests_per_minute
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 10000)),
        log_level=settings.log_level.lower()
    )