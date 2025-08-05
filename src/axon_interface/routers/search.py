"""
Semantic Search API Endpoints

Implements intelligent search capabilities:
- GET /search - Natural language query processing with semantic search
- GET /search/suggestions - Query suggestions and autocomplete
- GET /search/trends - Search trends and popular queries

Combines semantic similarity search with traditional full-text search
for optimal relevance and performance.
"""

import logging
from typing import List, Optional, Dict, Any
import time

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from ...shared.schemas import SearchResponse, SearchResult, ArticleResponse
from ...thalamus.semantic_search import get_search_engine, SemanticSearchEngine
from ...synaptic_vesicle.repositories import RepositoryFactory
from ..dependencies import get_repository_factory


logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/search",
    response_model=SearchResponse,
    summary="Semantic search with natural language queries",
    description="""
    Perform intelligent search across all content using natural language queries.
    
    Features:
    - Semantic similarity matching using vector embeddings
    - Traditional full-text search with TF-IDF scoring
    - Hybrid ranking combining semantic and lexical relevance
    - Query expansion and synonym matching
    - Relevance scoring and result ranking
    
    The search engine automatically determines the best approach based on
    the query type and content characteristics.
    """
)
async def search_content(
    request: Request,
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Result offset for pagination"),
    min_score: float = Query(0.0, ge=0.0, le=1.0, description="Minimum relevance score"),
    source_domain: Optional[str] = Query(None, description="Filter by source domain"),
    categories: Optional[str] = Query(None, description="Comma-separated categories to filter by"),
    published_after: Optional[str] = Query(None, description="Filter articles published after this date"),
    published_before: Optional[str] = Query(None, description="Filter articles published before this date"),
    search_engine: SemanticSearchEngine = Depends(get_search_engine),
    repository_factory: RepositoryFactory = Depends(get_repository_factory)
) -> SearchResponse:
    """Perform semantic search across content."""
    start_time = time.time()
    
    try:
        # Build search filters
        filters = {}
        if source_domain:
            filters["source_domain"] = source_domain
        if categories:
            filters["categories"] = [cat.strip() for cat in categories.split(",")]
        if published_after:
            filters["published_after"] = published_after
        if published_before:
            filters["published_before"] = published_before
        
        # Perform search
        search_results = await search_engine.search(
            query=q,
            limit=limit,
            offset=offset,
            min_score=min_score,
            filters=filters
        )
        
        # Convert results to response format
        results = []
        for result in search_results.results:
            article_response = ArticleResponse(
                id=result.article.id,
                url=result.article.url,
                title=result.article.title,
                content=result.article.content,
                summary=result.article.summary,
                author=result.article.author,
                published_at=result.article.published_at,
                source_domain=result.article.source_domain,
                scraped_at=result.article.scraped_at,
                nlp_data=result.article.nlp_data,
                page_metadata=result.article.page_metadata
            )
            
            results.append(SearchResult(
                score=result.score,
                article=article_response
            ))
        
        processing_time = int((time.time() - start_time) * 1000)
        
        logger.info(
            f"Search query '{q}' returned {len(results)} results "
            f"for user {request.state.user_id} in {processing_time}ms"
        )
        
        return SearchResponse(
            query=q,
            total_hits=search_results.total_hits,
            results=results,
            took_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error performing search for query '{q}': {e}")
        raise HTTPException(
            status_code=500,
            detail="Search request failed"
        )


@router.get(
    "/search/suggestions",
    summary="Get query suggestions and autocomplete",
    description="""
    Get query suggestions and autocomplete options based on:
    - Popular search queries
    - Content-based suggestions
    - Entity names and topics
    - Historical search patterns
    
    Useful for implementing search autocomplete and query assistance.
    """
)
async def get_search_suggestions(
    request: Request,
    q: str = Query(..., min_length=1, description="Partial query for suggestions"),
    limit: int = Query(5, ge=1, le=20, description="Maximum number of suggestions"),
    search_engine: SemanticSearchEngine = Depends(get_search_engine)
) -> Dict[str, Any]:
    """Get search suggestions and autocomplete."""
    try:
        suggestions = await search_engine.get_suggestions(
            partial_query=q,
            limit=limit
        )
        
        logger.info(f"Generated {len(suggestions)} suggestions for '{q}'")
        
        return {
            "query": q,
            "suggestions": suggestions,
            "count": len(suggestions)
        }
        
    except Exception as e:
        logger.error(f"Error generating suggestions for '{q}': {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate suggestions"
        )


@router.get(
    "/search/trends",
    summary="Get search trends and popular queries",
    description="""
    Retrieve search trends and analytics including:
    - Most popular search queries
    - Trending topics and entities
    - Search volume patterns
    - Query performance metrics
    
    Useful for understanding user interests and content discovery patterns.
    """
)
async def get_search_trends(
    request: Request,
    time_range: str = Query("24h", regex="^(1h|6h|24h|7d|30d)$", description="Time range for trends"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of trends"),
    search_engine: SemanticSearchEngine = Depends(get_search_engine)
) -> Dict[str, Any]:
    """Get search trends and popular queries."""
    try:
        trends = await search_engine.get_search_trends(
            time_range=time_range,
            limit=limit
        )
        
        logger.info(f"Retrieved search trends for {time_range} range")
        
        return {
            "time_range": time_range,
            "trends": trends,
            "generated_at": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error retrieving search trends: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve search trends"
        )


@router.post(
    "/search/feedback",
    summary="Submit search result feedback",
    description="""
    Submit feedback on search results to improve search quality.
    
    Feedback types:
    - Relevance ratings for search results
    - Click-through data
    - User satisfaction scores
    - Query refinement suggestions
    
    This data is used to continuously improve search algorithms and ranking.
    """
)
async def submit_search_feedback(
    request: Request,
    feedback_data: Dict[str, Any],
    search_engine: SemanticSearchEngine = Depends(get_search_engine)
) -> Dict[str, str]:
    """Submit search result feedback."""
    try:
        # Validate feedback data
        required_fields = ["query", "result_id", "feedback_type", "rating"]
        if not all(field in feedback_data for field in required_fields):
            raise HTTPException(
                status_code=400,
                detail=f"Missing required fields: {required_fields}"
            )
        
        # Process feedback
        await search_engine.process_feedback(
            user_id=request.state.user_id,
            feedback_data=feedback_data
        )
        
        logger.info(f"Processed search feedback from user {request.state.user_id}")
        
        return {
            "status": "success",
            "message": "Feedback submitted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing search feedback: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to process feedback"
        )


@router.get(
    "/search/similar",
    response_model=List[SearchResult],
    summary="Find similar content",
    description="""
    Find content similar to a given article or text snippet.
    
    Uses semantic similarity to identify related content based on:
    - Content embeddings and vector similarity
    - Entity overlap and relationship analysis
    - Topic modeling and thematic similarity
    - Linguistic pattern matching
    
    Useful for content recommendation and discovery.
    """
)
async def find_similar_content(
    request: Request,
    text: Optional[str] = Query(None, description="Text to find similar content for"),
    article_id: Optional[str] = Query(None, description="Article ID to find similar content for"),
    limit: int = Query(5, ge=1, le=20, description="Maximum number of similar items"),
    min_similarity: float = Query(0.3, ge=0.0, le=1.0, description="Minimum similarity score"),
    search_engine: SemanticSearchEngine = Depends(get_search_engine)
) -> List[SearchResult]:
    """Find content similar to given text or article."""
    try:
        if not text and not article_id:
            raise HTTPException(
                status_code=400,
                detail="Either 'text' or 'article_id' parameter is required"
            )
        
        # Find similar content
        similar_results = await search_engine.find_similar(
            text=text,
            article_id=article_id,
            limit=limit,
            min_similarity=min_similarity
        )
        
        # Convert to response format
        results = []
        for result in similar_results:
            article_response = ArticleResponse(
                id=result.article.id,
                url=result.article.url,
                title=result.article.title,
                content=result.article.content,
                summary=result.article.summary,
                author=result.article.author,
                published_at=result.article.published_at,
                source_domain=result.article.source_domain,
                scraped_at=result.article.scraped_at,
                nlp_data=result.article.nlp_data,
                page_metadata=result.article.page_metadata
            )
            
            results.append(SearchResult(
                score=result.similarity_score,
                article=article_response
            ))
        
        logger.info(f"Found {len(results)} similar items for user {request.state.user_id}")
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding similar content: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to find similar content"
        )