"""
Neurons - Lightweight Scrapers Main Application

This is the entry point for the Neurons component that runs on Railway.
It provides HTTP endpoints for scraping jobs and WebSocket connections to the hub.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import os
from typing import Dict, Any
import logging

from ..shared.logging_config import get_logger
from .http_scraper import HttpScraper
from .recipe_engine import get_recipe_engine

# Configure logging
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Project Synapse - Neurons",
    description="Lightweight scrapers for Project Synapse",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components
scraper = None
recipe_engine = None

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global scraper, recipe_engine
    
    logger.info("Starting Neurons component...")
    
    # Initialize scraper
    scraper = HttpScraper()
    
    # Initialize recipe engine
    recipe_engine = get_recipe_engine()
    
    logger.info("Neurons component started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Neurons component...")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "component": "neurons",
        "version": "1.0.0",
        "environment": os.getenv("ENVIRONMENT", "development")
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
        "recipe_id": "optional_recipe_id",
        "priority": "normal",
        "callback_url": "https://hub.com/callback"
    }
    """
    try:
        job_id = request.get("job_id")
        url = request.get("url")
        recipe_id = request.get("recipe_id")
        callback_url = request.get("callback_url")
        
        if not job_id or not url:
            raise HTTPException(status_code=400, detail="job_id and url are required")
        
        # Start scraping in background
        background_tasks.add_task(
            perform_scraping,
            job_id=job_id,
            url=url,
            recipe_id=recipe_id,
            callback_url=callback_url
        )
        
        return {
            "status": "accepted",
            "job_id": job_id,
            "message": "Scraping job started"
        }
        
    except Exception as e:
        logger.error(f"Error starting scraping job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def perform_scraping(
    job_id: str,
    url: str,
    recipe_id: str = None,
    callback_url: str = None
):
    """Perform the actual scraping work."""
    try:
        logger.info(f"Starting scraping job {job_id} for URL: {url}")
        
        # Get recipe if specified
        recipe = None
        if recipe_id and recipe_engine:
            recipe = await recipe_engine.get_recipe(recipe_id)
        
        # Perform scraping
        result = await scraper.scrape(url, recipe=recipe)
        
        # Send callback if specified
        if callback_url:
            import httpx
            async with httpx.AsyncClient() as client:
                await client.post(
                    callback_url,
                    json={
                        "job_id": job_id,
                        "status": "completed",
                        "result": result
                    },
                    timeout=30
                )
        
        logger.info(f"Scraping job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Scraping job {job_id} failed: {e}")
        
        # Send error callback if specified
        if callback_url:
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    await client.post(
                        callback_url,
                        json={
                            "job_id": job_id,
                            "status": "failed",
                            "error": str(e)
                        },
                        timeout=30
                    )
            except Exception as callback_error:
                logger.error(f"Failed to send error callback: {callback_error}")

@app.get("/status")
async def get_status():
    """Get component status and metrics."""
    return {
        "component": "neurons",
        "status": "running",
        "metrics": {
            "jobs_processed": 0,  # Would track actual metrics
            "success_rate": 0.95,
            "average_response_time": 2.5
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)