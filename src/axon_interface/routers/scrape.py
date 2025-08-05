"""
ScrapeDrop API Endpoints - On-demand Scraping

Implements the ScrapeDrop functionality for on-demand content scraping:
- POST /scrape - Submit URL for scraping with job creation
- GET /scrape/status/{job_id} - Check scraping job status
- GET /scrape/jobs - List user's scraping jobs
- DELETE /scrape/jobs/{job_id} - Cancel scraping job

Provides priority queue management and comprehensive job tracking.
"""

import logging
from typing import List, Optional, Dict, Any
from uuid import UUID
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from ...shared.schemas import (
    ScrapeJobCreate, 
    ScrapeJobResponse, 
    JobStatus,
    PaginatedResponse,
    PaginationInfo
)
from ...signal_relay.task_dispatcher import get_task_dispatcher, TaskDispatcher, TaskType, TaskPriority
from ...synaptic_vesicle.repositories import RepositoryFactory
from ..dependencies import get_repository_factory
from ..webhooks import get_webhook_event_bus, create_job_webhook_event, WebhookEventType


logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/scrape",
    response_model=ScrapeJobResponse,
    summary="Submit URL for on-demand scraping",
    description="""
    Submit a URL for immediate scraping and processing.
    
    Features:
    - Priority queue management (normal vs high priority)
    - Automatic recipe selection and learning
    - Real-time job status tracking
    - Comprehensive error handling and retry logic
    - Integration with NLP processing pipeline
    
    The scraping job will be queued and processed asynchronously.
    Use the returned job_id to track progress via the status endpoint.
    """
)
async def submit_scrape_job(
    request: Request,
    job_request: ScrapeJobCreate,
    task_dispatcher: TaskDispatcher = Depends(get_task_dispatcher),
    repository_factory: RepositoryFactory = Depends(get_repository_factory),
    webhook_event_bus = Depends(get_webhook_event_bus)
) -> ScrapeJobResponse:
    """Submit a URL for scraping."""
    try:
        # Generate job ID
        job_id = uuid.uuid4()
        
        # Determine priority based on user tier and request
        user_tier = getattr(request.state, "user_tier", "free")
        priority = TaskPriority.HIGH if (job_request.priority and user_tier in ["premium", "enterprise"]) else TaskPriority.NORMAL
        
        # Create task payload
        task_payload = {
            "url": str(job_request.url),
            "job_id": str(job_id),
            "user_id": request.state.user_id,
            "priority": job_request.priority,
            "requested_at": "2025-01-08T00:00:00Z"
        }
        
        # Submit task to dispatcher
        task_id = await task_dispatcher.submit_task(
            task_type=TaskType.SCRAPE_URL,
            payload=task_payload,
            priority=priority
        )
        
        # Store job information
        task_repo = repository_factory.get_task_queue_repository()
        await task_repo.create_scrape_job(
            job_id=job_id,
            task_id=task_id,
            url=str(job_request.url),
            user_id=request.state.user_id,
            priority=job_request.priority
        )
        
        # Publish webhook event for job creation
        job_created_event = create_job_webhook_event(
            event_type=WebhookEventType.JOB_CREATED,
            job_id=str(job_id),
            job_type="scrape",
            status="pending",
            user_id=request.state.user_id,
            url=str(job_request.url),
            priority=job_request.priority,
            task_id=str(task_id)
        )
        await webhook_event_bus.publish_event(job_created_event)
        
        logger.info(f"Submitted scrape job {job_id} for URL {job_request.url} by user {request.state.user_id}")
        
        return ScrapeJobResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            message="Scraping job submitted successfully",
            data=None,
            created_at="2025-01-08T00:00:00Z",
            completed_at=None
        )
        
    except Exception as e:
        logger.error(f"Error submitting scrape job: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to submit scraping job"
        )


@router.get(
    "/scrape/status/{job_id}",
    response_model=ScrapeJobResponse,
    summary="Check scraping job status",
    description="""
    Check the status of a scraping job by its ID.
    
    Job statuses:
    - PENDING: Job is queued and waiting to be processed
    - PROCESSING: Job is currently being scraped
    - LEARNING: Advanced scrapers are learning new patterns
    - COMPLETED: Job completed successfully with data
    - FAILED: Job failed with error details
    
    For completed jobs, the response includes the scraped article data.
    """
)
async def get_scrape_job_status(
    job_id: UUID,
    request: Request,
    repository_factory: RepositoryFactory = Depends(get_repository_factory)
) -> ScrapeJobResponse:
    """Get the status of a scraping job."""
    try:
        # Get job information
        task_repo = repository_factory.get_task_queue_repository()
        job_info = await task_repo.get_scrape_job_status(job_id)
        
        if not job_info:
            raise HTTPException(
                status_code=404,
                detail=f"Scraping job {job_id} not found"
            )
        
        # Check if job belongs to user
        if job_info.get("user_id") != request.state.user_id:
            raise HTTPException(
                status_code=403,
                detail="Access denied to this scraping job"
            )
        
        # Get article data if completed
        article_data = None
        if job_info.get("status") == JobStatus.COMPLETED and job_info.get("article_id"):
            article_repo = repository_factory.get_article_repository()
            article = await article_repo.get_by_id(job_info["article_id"])
            if article:
                from ...shared.schemas import ArticleResponse
                article_data = ArticleResponse(
                    id=article.id,
                    url=article.url,
                    title=article.title,
                    content=article.content,
                    summary=article.summary,
                    author=article.author,
                    published_at=article.published_at,
                    source_domain=article.source_domain,
                    scraped_at=article.scraped_at,
                    nlp_data=article.nlp_data,
                    page_metadata=article.page_metadata
                )
        
        logger.info(f"Retrieved status for scrape job {job_id}")
        
        return ScrapeJobResponse(
            job_id=job_id,
            status=JobStatus(job_info["status"]),
            message=job_info.get("message", ""),
            data=article_data,
            created_at=job_info["created_at"],
            completed_at=job_info.get("completed_at")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving job status for {job_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve job status"
        )


@router.get(
    "/scrape/jobs",
    response_model=PaginatedResponse,
    summary="List user's scraping jobs",
    description="""
    Retrieve a paginated list of the user's scraping jobs.
    
    Supports filtering by:
    - Job status (pending, processing, completed, failed)
    - Date range
    - URL domain
    
    Results are sorted by creation date (most recent first) by default.
    """
)
async def list_scrape_jobs(
    request: Request,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=50, description="Items per page"),
    status: Optional[JobStatus] = Query(None, description="Filter by job status"),
    created_after: Optional[str] = Query(None, description="Filter jobs created after this date"),
    created_before: Optional[str] = Query(None, description="Filter jobs created before this date"),
    repository_factory: RepositoryFactory = Depends(get_repository_factory)
) -> PaginatedResponse:
    """List user's scraping jobs."""
    try:
        # Build filters
        filters = {"user_id": request.state.user_id}
        if status:
            filters["status"] = status.value
        if created_after:
            filters["created_after"] = created_after
        if created_before:
            filters["created_before"] = created_before
        
        # Calculate offset
        offset = (page - 1) * page_size
        
        # Get jobs
        task_repo = repository_factory.get_task_queue_repository()
        jobs, total_count = await task_repo.list_scrape_jobs(
            filters=filters,
            limit=page_size,
            offset=offset
        )
        
        # Calculate pagination
        total_pages = (total_count + page_size - 1) // page_size
        pagination = PaginationInfo(
            page=page,
            page_size=page_size,
            total_results=total_count,
            total_pages=total_pages
        )
        
        # Convert to response format
        job_responses = [
            ScrapeJobResponse(
                job_id=job["job_id"],
                status=JobStatus(job["status"]),
                message=job.get("message", ""),
                data=None,  # Don't include full article data in list view
                created_at=job["created_at"],
                completed_at=job.get("completed_at")
            )
            for job in jobs
        ]
        
        logger.info(f"Retrieved {len(jobs)} scrape jobs for user {request.state.user_id}")
        
        return PaginatedResponse(
            pagination=pagination,
            data=job_responses
        )
        
    except Exception as e:
        logger.error(f"Error listing scrape jobs: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve scraping jobs"
        )


@router.delete(
    "/scrape/jobs/{job_id}",
    summary="Cancel scraping job",
    description="""
    Cancel a pending or processing scraping job.
    
    Only jobs in PENDING or PROCESSING status can be cancelled.
    Completed or failed jobs cannot be cancelled.
    
    Returns the updated job status after cancellation.
    """
)
async def cancel_scrape_job(
    job_id: UUID,
    request: Request,
    task_dispatcher: TaskDispatcher = Depends(get_task_dispatcher),
    repository_factory: RepositoryFactory = Depends(get_repository_factory)
) -> Dict[str, str]:
    """Cancel a scraping job."""
    try:
        # Get job information
        task_repo = repository_factory.get_task_queue_repository()
        job_info = await task_repo.get_scrape_job_status(job_id)
        
        if not job_info:
            raise HTTPException(
                status_code=404,
                detail=f"Scraping job {job_id} not found"
            )
        
        # Check if job belongs to user
        if job_info.get("user_id") != request.state.user_id:
            raise HTTPException(
                status_code=403,
                detail="Access denied to this scraping job"
            )
        
        # Check if job can be cancelled
        current_status = job_info.get("status")
        if current_status not in [JobStatus.PENDING.value, JobStatus.PROCESSING.value]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel job with status: {current_status}"
            )
        
        # Cancel the task
        task_id = job_info.get("task_id")
        if task_id:
            await task_dispatcher.cancel_task(task_id)
        
        # Update job status
        await task_repo.update_scrape_job_status(
            job_id=job_id,
            status=JobStatus.CANCELLED,
            message="Job cancelled by user"
        )
        
        logger.info(f"Cancelled scrape job {job_id} for user {request.state.user_id}")
        
        return {
            "status": "success",
            "message": f"Scraping job {job_id} cancelled successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling scrape job {job_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to cancel scraping job"
        )


@router.get(
    "/scrape/stats",
    summary="Get scraping statistics",
    description="""
    Retrieve scraping statistics and metrics including:
    - Total jobs processed
    - Success/failure rates
    - Average processing times
    - Popular domains and sources
    - Queue status and performance metrics
    
    Useful for monitoring scraping performance and usage patterns.
    """
)
async def get_scrape_stats(
    request: Request,
    time_range: str = Query("24h", regex="^(1h|6h|24h|7d|30d)$", description="Time range for statistics"),
    repository_factory: RepositoryFactory = Depends(get_repository_factory)
) -> Dict[str, Any]:
    """Get scraping statistics."""
    try:
        task_repo = repository_factory.get_task_queue_repository()
        
        # Get user-specific stats
        user_stats = await task_repo.get_user_scrape_stats(
            user_id=request.state.user_id,
            time_range=time_range
        )
        
        # Get general system stats (if user has appropriate tier)
        system_stats = {}
        if getattr(request.state, "user_tier", "free") in ["premium", "enterprise"]:
            system_stats = await task_repo.get_system_scrape_stats(time_range)
        
        logger.info(f"Retrieved scrape statistics for user {request.state.user_id}")
        
        return {
            "time_range": time_range,
            "user_stats": user_stats,
            "system_stats": system_stats,
            "generated_at": "2025-01-08T00:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Error retrieving scrape statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve scraping statistics"
        )