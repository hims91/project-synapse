"""
Central Cortex - Content Router
Layer 3: Cerebral Cortex

This module implements content management endpoints.
Provides article retrieval, search, and content operations.
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ..dependencies import (
    get_db_session_dependency, get_repository_factory, get_current_user,
    require_active_user, get_pagination_params, get_search_params,
    PaginationParams, SearchParams
)
from ...shared.schemas import (
    ArticleResponse, ArticleCreate, ArticleUpdate,
    PaginatedResponse, PaginationInfo, SearchResponse, SearchResult,
    ScrapeJobCreate, ScrapeJobResponse
)
from ...synaptic_vesicle.repositories import RepositoryFactory

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/articles", response_model=PaginatedResponse)
async def get_articles(
    pagination: PaginationParams = Depends(get_pagination_params),
    category: Optional[str] = Query(None, description="Filter by category"),
    source_domain: Optional[str] = Query(None, description="Filter by source domain"),
    date_from: Optional[str] = Query(None, description="Filter from date (ISO format)"),
    date_to: Optional[str] = Query(None, description="Filter to date (ISO format)"),
    current_user: Dict[str, Any] = Depends(require_active_user),
    repo_factory: RepositoryFactory = Depends(get_repository_factory)
) -> PaginatedResponse:
    """
    Get paginated list of articles.
    
    Args:
        pagination: Pagination parameters
        category: Category filter
        source_domain: Source domain filter
        date_from: Start date filter
        date_to: End date filter
        current_user: Current authenticated user
        repo_factory: Repository factory
        
    Returns:
        Paginated article list
    """
    try:
        article_repo = repo_factory.get_article_repository()
        
        # Build filters
        filters = {}
        if category:
            filters['category'] = category
        if source_domain:
            filters['source_domain'] = source_domain
        if date_from:
            filters['date_from'] = date_from
        if date_to:
            filters['date_to'] = date_to
        
        # Get articles with pagination
        articles, total_count = await article_repo.get_paginated(
            limit=pagination.limit,
            offset=pagination.offset,
            filters=filters
        )
        
        # Calculate pagination info
        total_pages = (total_count + pagination.page_size - 1) // pagination.page_size
        
        pagination_info = PaginationInfo(
            page=pagination.page,
            page_size=pagination.page_size,
            total_results=total_count,
            total_pages=total_pages
        )
        
        return PaginatedResponse(
            pagination=pagination_info,
            data=articles
        )
        
    except Exception as e:
        logger.error(f"Failed to get articles: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve articles"
        )


@router.get("/articles/{article_id}", response_model=ArticleResponse)
async def get_article(
    article_id: UUID,
    current_user: Dict[str, Any] = Depends(require_active_user),
    repo_factory: RepositoryFactory = Depends(get_repository_factory)
) -> ArticleResponse:
    """
    Get specific article by ID.
    
    Args:
        article_id: Article ID
        current_user: Current authenticated user
        repo_factory: Repository factory
        
    Returns:
        Article details
    """
    try:
        article_repo = repo_factory.get_article_repository()
        article = await article_repo.get_by_id(article_id)
        
        if not article:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Article not found"
            )
        
        return article
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get article {article_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve article"
        )


@router.get("/search", response_model=SearchResponse)
async def search_articles(
    search_params: SearchParams = Depends(get_search_params),
    pagination: PaginationParams = Depends(get_pagination_params),
    current_user: Dict[str, Any] = Depends(require_active_user),
    repo_factory: RepositoryFactory = Depends(get_repository_factory)
) -> SearchResponse:
    """
    Search articles with full-text search.
    
    Args:
        search_params: Search parameters
        pagination: Pagination parameters
        current_user: Current authenticated user
        repo_factory: Repository factory
        
    Returns:
        Search results with relevance scores
    """
    try:
        if not search_params.query:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Search query is required"
            )
        
        article_repo = repo_factory.get_article_repository()
        
        # Perform search
        start_time = datetime.utcnow()
        
        search_results, total_hits = await article_repo.search(
            query=search_params.query,
            limit=pagination.limit,
            offset=pagination.offset,
            filters={
                'category': search_params.category,
                'source_domain': search_params.source_domain,
                'date_from': search_params.date_from,
                'date_to': search_params.date_to
            },
            sort_by=search_params.sort_by,
            sort_order=search_params.sort_order
        )
        
        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Format results
        results = []
        for article, score in search_results:
            results.append(SearchResult(
                score=score,
                article=article
            ))
        
        return SearchResponse(
            query=search_params.query,
            total_hits=total_hits,
            results=results,
            took_ms=int(execution_time)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search failed"
        )


@router.post("/scrape", response_model=ScrapeJobResponse)
async def create_scrape_job(
    scrape_request: ScrapeJobCreate,
    current_user: Dict[str, Any] = Depends(require_active_user),
    repo_factory: RepositoryFactory = Depends(get_repository_factory)
) -> ScrapeJobResponse:
    """
    Create a new scraping job.
    
    Args:
        scrape_request: Scraping job request
        current_user: Current authenticated user
        repo_factory: Repository factory
        
    Returns:
        Scraping job information
    """
    try:
        from ...signal_relay.task_dispatcher import get_task_dispatcher
        from ...shared.schemas import TaskPayload, TaskQueueCreate
        
        # Get task dispatcher
        task_dispatcher = await get_task_dispatcher()
        
        # Create task payload
        task_payload = TaskPayload(
            url=scrape_request.url,
            priority=scrape_request.priority,
            metadata={
                "user_id": current_user["user_id"],
                "requested_at": datetime.utcnow().isoformat()
            }
        )
        
        # Create scraping task
        task_create = TaskQueueCreate(
            task_type="scrape_url",
            payload=task_payload,
            priority=1 if scrape_request.priority else 5
        )
        
        # Submit task
        task_repo = repo_factory.get_task_queue_repository()
        task = await task_repo.create(task_create)
        
        # Dispatch task
        await task_dispatcher.submit_task(task)
        
        logger.info(f"Scrape job created: {task.id} for URL: {scrape_request.url}")
        
        return ScrapeJobResponse(
            job_id=task.id,
            status="pending",
            message="Scraping job created successfully",
            created_at=task.scheduled_at
        )
        
    except Exception as e:
        logger.error(f"Failed to create scrape job: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create scraping job"
        )


@router.get("/scrape/{job_id}", response_model=ScrapeJobResponse)
async def get_scrape_job(
    job_id: UUID,
    current_user: Dict[str, Any] = Depends(require_active_user),
    repo_factory: RepositoryFactory = Depends(get_repository_factory)
) -> ScrapeJobResponse:
    """
    Get scraping job status.
    
    Args:
        job_id: Scraping job ID
        current_user: Current authenticated user
        repo_factory: Repository factory
        
    Returns:
        Scraping job status
    """
    try:
        task_repo = repo_factory.get_task_queue_repository()
        task = await task_repo.get_by_id(job_id)
        
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Scraping job not found"
            )
        
        # Check if user owns this job
        task_user_id = task.payload.get("metadata", {}).get("user_id")
        if task_user_id != current_user["user_id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this scraping job"
            )
        
        # Get article data if completed
        article_data = None
        if task.status == "completed":
            # Try to find the scraped article
            article_repo = repo_factory.get_article_repository()
            articles = await article_repo.get_by_url(str(task.payload.url))
            if articles:
                article_data = articles[0]
        
        return ScrapeJobResponse(
            job_id=task.id,
            status=task.status,
            message=f"Job is {task.status}",
            data=article_data,
            created_at=task.scheduled_at,
            completed_at=task.processed_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get scrape job {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve scraping job"
        )


@router.get("/trending")
async def get_trending_content(
    time_window: str = Query("24h", description="Time window (1h, 6h, 24h)"),
    category: Optional[str] = Query(None, description="Category filter"),
    limit: int = Query(10, description="Number of trending items"),
    current_user: Dict[str, Any] = Depends(require_active_user),
    repo_factory: RepositoryFactory = Depends(get_repository_factory)
) -> Dict[str, Any]:
    """
    Get trending content.
    
    Args:
        time_window: Time window for trending analysis
        category: Category filter
        limit: Number of items to return
        current_user: Current authenticated user
        repo_factory: Repository factory
        
    Returns:
        Trending content data
    """
    try:
        # Validate time window
        valid_windows = ["1h", "6h", "24h"]
        if time_window not in valid_windows:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid time window. Must be one of: {valid_windows}"
            )
        
        # Get trending data (simplified implementation)
        article_repo = repo_factory.get_article_repository()
        
        # Get recent popular articles as trending content
        trending_articles = await article_repo.get_trending(
            time_window=time_window,
            category=category,
            limit=limit
        )
        
        return {
            "time_window": time_window,
            "category": category,
            "trending_articles": trending_articles,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get trending content: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve trending content"
        )


@router.get("/categories")
async def get_content_categories(
    current_user: Dict[str, Any] = Depends(require_active_user),
    repo_factory: RepositoryFactory = Depends(get_repository_factory)
) -> List[Dict[str, Any]]:
    """
    Get available content categories.
    
    Args:
        current_user: Current authenticated user
        repo_factory: Repository factory
        
    Returns:
        List of content categories with counts
    """
    try:
        article_repo = repo_factory.get_article_repository()
        categories = await article_repo.get_categories_with_counts()
        
        return categories
        
    except Exception as e:
        logger.error(f"Failed to get categories: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve categories"
        )


@router.get("/sources")
async def get_content_sources(
    current_user: Dict[str, Any] = Depends(require_active_user),
    repo_factory: RepositoryFactory = Depends(get_repository_factory)
) -> List[Dict[str, Any]]:
    """
    Get available content sources.
    
    Args:
        current_user: Current authenticated user
        repo_factory: Repository factory
        
    Returns:
        List of content sources with counts
    """
    try:
        article_repo = repo_factory.get_article_repository()
        sources = await article_repo.get_sources_with_counts()
        
        return sources
        
    except Exception as e:
        logger.error(f"Failed to get sources: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve sources"
        )