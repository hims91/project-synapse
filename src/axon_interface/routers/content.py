"""
Content API Endpoints

Implements the core content retrieval API endpoints:
- GET /content/articles - List articles with pagination and filtering
- GET /content/articles/{id} - Retrieve single article by ID

These endpoints provide access to the scraped and processed article content
with comprehensive filtering, pagination, and response formatting.
"""

import logging
from typing import List, Optional, Dict, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from ...shared.schemas import (
    ArticleResponse, 
    PaginatedResponse, 
    PaginationInfo,
    ErrorResponse
)
from ...synaptic_vesicle.repositories import RepositoryFactory
from ..dependencies import get_repository_factory


logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/content/articles",
    response_model=PaginatedResponse,
    summary="List articles with pagination and filtering",
    description="""
    Retrieve a paginated list of articles with optional filtering capabilities.
    
    Supports filtering by:
    - Source domain
    - Date range (published_after, published_before)
    - Content categories
    - Sentiment range
    - Significance threshold
    
    Results are paginated and can be sorted by various criteria.
    """
)
async def list_articles(
    request: Request,
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page"),
    source_domain: Optional[str] = Query(None, description="Filter by source domain"),
    published_after: Optional[str] = Query(None, description="Filter articles published after this date (ISO format)"),
    published_before: Optional[str] = Query(None, description="Filter articles published before this date (ISO format)"),
    categories: Optional[str] = Query(None, description="Comma-separated list of categories to filter by"),
    min_sentiment: Optional[float] = Query(None, ge=-1.0, le=1.0, description="Minimum sentiment score"),
    max_sentiment: Optional[float] = Query(None, ge=-1.0, le=1.0, description="Maximum sentiment score"),
    min_significance: Optional[float] = Query(None, ge=0.0, le=10.0, description="Minimum significance score"),
    sort_by: str = Query("published_at", description="Sort field (published_at, scraped_at, significance)"),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="Sort order"),
    repository_factory: RepositoryFactory = Depends(get_repository_factory)
) -> PaginatedResponse:
    """List articles with pagination and filtering."""
    try:
        # Get article repository
        article_repo = repository_factory.get_article_repository()
        
        # Build filters
        filters = {}
        if source_domain:
            filters["source_domain"] = source_domain
        if published_after:
            filters["published_after"] = published_after
        if published_before:
            filters["published_before"] = published_before
        if categories:
            filters["categories"] = [cat.strip() for cat in categories.split(",")]
        if min_sentiment is not None:
            filters["min_sentiment"] = min_sentiment
        if max_sentiment is not None:
            filters["max_sentiment"] = max_sentiment
        if min_significance is not None:
            filters["min_significance"] = min_significance
        
        # Calculate offset
        offset = (page - 1) * page_size
        
        # Get articles with filters
        articles, total_count = await article_repo.list_with_filters(
            filters=filters,
            limit=page_size,
            offset=offset,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        # Calculate pagination info
        total_pages = (total_count + page_size - 1) // page_size
        
        pagination = PaginationInfo(
            page=page,
            page_size=page_size,
            total_results=total_count,
            total_pages=total_pages
        )
        
        # Convert to response format
        article_responses = [
            ArticleResponse(
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
            for article in articles
        ]
        
        logger.info(f"Retrieved {len(articles)} articles for user {request.state.user_id}")
        
        return PaginatedResponse(
            pagination=pagination,
            data=article_responses
        )
        
    except Exception as e:
        logger.error(f"Error listing articles: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve articles"
        )


@router.get(
    "/content/articles/{article_id}",
    response_model=ArticleResponse,
    summary="Retrieve single article by ID",
    description="""
    Retrieve a single article by its unique identifier.
    
    Returns the complete article data including:
    - Basic article information (title, content, author, etc.)
    - NLP analysis results (sentiment, entities, categories, etc.)
    - Page metadata (technical information about the webpage)
    - Scraping metadata (when and how the article was obtained)
    """
)
async def get_article(
    article_id: UUID,
    request: Request,
    repository_factory: RepositoryFactory = Depends(get_repository_factory)
) -> ArticleResponse:
    """Retrieve a single article by ID."""
    try:
        # Get article repository
        article_repo = repository_factory.get_article_repository()
        
        # Get article by ID
        article = await article_repo.get_by_id(article_id)
        
        if not article:
            raise HTTPException(
                status_code=404,
                detail=f"Article with ID {article_id} not found"
            )
        
        logger.info(f"Retrieved article {article_id} for user {request.state.user_id}")
        
        # Convert to response format
        return ArticleResponse(
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
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving article {article_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve article"
        )


@router.get(
    "/content/articles/{article_id}/related",
    response_model=List[ArticleResponse],
    summary="Get related articles",
    description="""
    Retrieve articles related to the specified article based on:
    - Content similarity (using NLP analysis)
    - Shared entities
    - Similar categories
    - Same source domain
    
    Results are ranked by relevance and limited to the most relevant matches.
    """
)
async def get_related_articles(
    article_id: UUID,
    request: Request,
    limit: int = Query(5, ge=1, le=20, description="Maximum number of related articles to return"),
    repository_factory: RepositoryFactory = Depends(get_repository_factory)
) -> List[ArticleResponse]:
    """Get articles related to the specified article."""
    try:
        # Get article repository
        article_repo = repository_factory.get_article_repository()
        
        # First, get the source article
        source_article = await article_repo.get_by_id(article_id)
        if not source_article:
            raise HTTPException(
                status_code=404,
                detail=f"Article with ID {article_id} not found"
            )
        
        # Get related articles
        related_articles = await article_repo.find_related_articles(
            article_id=article_id,
            limit=limit
        )
        
        logger.info(f"Retrieved {len(related_articles)} related articles for {article_id}")
        
        # Convert to response format
        return [
            ArticleResponse(
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
            for article in related_articles
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving related articles for {article_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve related articles"
        )


@router.get(
    "/content/stats",
    summary="Get content statistics",
    description="""
    Retrieve statistics about the content database including:
    - Total article count
    - Articles by source domain
    - Articles by category
    - Recent activity metrics
    - Content quality metrics
    """
)
async def get_content_stats(
    request: Request,
    repository_factory: RepositoryFactory = Depends(get_repository_factory)
) -> Dict[str, Any]:
    """Get content statistics."""
    try:
        # Get article repository
        article_repo = repository_factory.get_article_repository()
        
        # Get various statistics
        stats = await article_repo.get_content_statistics()
        
        logger.info(f"Retrieved content statistics for user {request.state.user_id}")
        
        return {
            "total_articles": stats.get("total_articles", 0),
            "articles_last_24h": stats.get("articles_last_24h", 0),
            "articles_last_7d": stats.get("articles_last_7d", 0),
            "top_domains": stats.get("top_domains", []),
            "top_categories": stats.get("top_categories", []),
            "avg_sentiment": stats.get("avg_sentiment", 0.0),
            "avg_significance": stats.get("avg_significance", 0.0),
            "last_updated": stats.get("last_updated")
        }
        
    except Exception as e:
        logger.error(f"Error retrieving content statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve content statistics"
        )