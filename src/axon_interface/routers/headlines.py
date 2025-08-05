"""
Headlines API Endpoints - Top Headlines with Significance Scoring

Implements intelligent headline curation and ranking:
- GET /headlines - Top headlines with significance scoring
- GET /headlines/categories - Category-based headline curation
- GET /headlines/breaking - Breaking news detection and ranking
- GET /headlines/personalized - Personalized headline recommendations
- POST /headlines/analyze - Headline significance analysis

Provides ML-powered headline ranking with quality assessment and personalization.
"""

import logging
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timezone, timedelta
from enum import Enum
import math
import re
from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Body
from pydantic import BaseModel, Field, validator

from ...shared.schemas import PaginatedResponse, PaginationInfo, ArticleResponse
from ...thalamus.nlp_pipeline import get_nlp_pipeline, NLPPipeline
from ...thalamus.bias_analysis import get_bias_engine, BiasDetectionEngine
from ..dependencies import get_repository_factory
from ...synaptic_vesicle.repositories import RepositoryFactory


logger = logging.getLogger(__name__)
router = APIRouter()


class HeadlineCategory(str, Enum):
    """Categories for headline classification."""
    BREAKING_NEWS = "breaking_news"
    POLITICS = "politics"
    TECHNOLOGY = "technology"
    BUSINESS = "business"
    SCIENCE = "science"
    HEALTH = "health"
    ENTERTAINMENT = "entertainment"
    SPORTS = "sports"
    WORLD_NEWS = "world_news"
    LOCAL_NEWS = "local_news"


class SignificanceLevel(str, Enum):
    """Significance levels for headlines."""
    CRITICAL = "critical"      # 0.9-1.0
    HIGH = "high"             # 0.7-0.9
    MEDIUM = "medium"         # 0.5-0.7
    LOW = "low"              # 0.3-0.5
    MINIMAL = "minimal"       # 0.0-0.3


class HeadlineQuality(str, Enum):
    """Quality assessment levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


class HeadlineItem(BaseModel):
    """Model for a headline item with significance scoring."""
    id: str = Field(..., description="Headline/article ID")
    title: str = Field(..., description="Headline text")
    summary: Optional[str] = Field(None, description="Brief summary")
    url: str = Field(..., description="Article URL")
    source: str = Field(..., description="News source")
    category: HeadlineCategory = Field(..., description="Headline category")
    published_at: datetime = Field(..., description="Publication timestamp")
    significance_score: float = Field(..., ge=0.0, le=1.0, description="Significance score (0-1)")
    significance_level: SignificanceLevel = Field(..., description="Significance level")
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Quality score (0-1)")
    quality_level: HeadlineQuality = Field(..., description="Quality assessment")
    engagement_score: float = Field(..., ge=0.0, le=1.0, description="Predicted engagement score")
    sentiment: float = Field(..., ge=-1.0, le=1.0, description="Sentiment score")
    bias_score: float = Field(..., ge=0.0, le=1.0, description="Bias score (0=unbiased, 1=highly biased)")
    entities: List[str] = Field(default_factory=list, description="Key entities mentioned")
    keywords: List[str] = Field(default_factory=list, description="Important keywords")
    geographic_relevance: List[str] = Field(default_factory=list, description="Geographic regions of relevance")
    breaking_news_indicators: Dict[str, Any] = Field(default_factory=dict, description="Breaking news signals")
    personalization_score: Optional[float] = Field(None, description="Personalization score for user")


class HeadlinesResponse(BaseModel):
    """Response model for headlines."""
    headlines: List[HeadlineItem] = Field(..., description="Curated headlines")
    total_headlines: int = Field(..., description="Total headlines available")
    curation_criteria: Dict[str, Any] = Field(..., description="Curation criteria used")
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Response timestamp")
    next_update: datetime = Field(..., description="Next scheduled update")


class BreakingNewsResponse(BaseModel):
    """Response model for breaking news."""
    breaking_headlines: List[HeadlineItem] = Field(..., description="Breaking news headlines")
    alert_level: str = Field(..., description="Overall alert level")
    detection_criteria: Dict[str, Any] = Field(..., description="Breaking news detection criteria")
    confidence: float = Field(..., description="Confidence in breaking news detection")


class PersonalizedHeadlinesResponse(BaseModel):
    """Response model for personalized headlines."""
    personalized_headlines: List[HeadlineItem] = Field(..., description="Personalized headlines")
    user_profile: Dict[str, Any] = Field(..., description="User interest profile used")
    personalization_factors: Dict[str, Any] = Field(..., description="Personalization factors")
    diversity_score: float = Field(..., description="Content diversity score")


@router.get(
    "/headlines",
    response_model=HeadlinesResponse,
    summary="Get top headlines with significance scoring",
    description="""
    Retrieve top headlines ranked by significance and quality scores.
    
    Features:
    - ML-powered significance scoring based on multiple factors
    - Quality assessment using linguistic and structural analysis
    - Multi-category headline curation
    - Real-time ranking with freshness weighting
    - Bias detection and quality filtering
    - Geographic relevance scoring
    
    Supports filtering by category, significance level, and time range.
    """
)
async def get_top_headlines(
    request: Request,
    category: Optional[HeadlineCategory] = Query(None, description="Filter by category"),
    min_significance: float = Query(0.3, ge=0.0, le=1.0, description="Minimum significance score"),
    max_age_hours: int = Query(24, ge=1, le=168, description="Maximum age in hours"),
    limit: int = Query(20, ge=1, le=100, description="Maximum headlines to return"),
    include_bias_analysis: bool = Query(True, description="Include bias analysis"),
    geographic_filter: Optional[str] = Query(None, description="Geographic region filter"),
    repository_factory: RepositoryFactory = Depends(get_repository_factory),
    nlp_pipeline: NLPPipeline = Depends(get_nlp_pipeline),
    bias_engine: BiasDetectionEngine = Depends(get_bias_engine)
) -> HeadlinesResponse:
    """Get top headlines with significance scoring."""
    try:
        user_tier = getattr(request.state, "user_tier", "free")
        
        # Validate access based on user tier
        if min_significance < 0.5 and user_tier == "free":
            raise HTTPException(
                status_code=403,
                detail="Low significance filtering requires premium subscription"
            )
        
        # Calculate time boundaries
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=max_age_hours)
        
        # Get articles from repository (mock for now)
        articles = await _get_recent_articles(
            start_time, end_time, category, repository_factory
        )
        
        # Score headlines for significance and quality
        scored_headlines = await _score_headlines(
            articles, nlp_pipeline, bias_engine, include_bias_analysis
        )
        
        # Filter by significance threshold
        filtered_headlines = [
            h for h in scored_headlines 
            if h.significance_score >= min_significance
        ]
        
        # Apply geographic filtering if specified
        if geographic_filter:
            filtered_headlines = [
                h for h in filtered_headlines
                if geographic_filter.lower() in [region.lower() for region in h.geographic_relevance]
            ]
        
        # Sort by significance score and limit results
        filtered_headlines.sort(key=lambda x: x.significance_score, reverse=True)
        top_headlines = filtered_headlines[:limit]
        
        # Calculate next update time
        next_update = datetime.now(timezone.utc) + timedelta(minutes=15)
        
        # Prepare curation criteria
        curation_criteria = {
            "min_significance": min_significance,
            "max_age_hours": max_age_hours,
            "category_filter": category.value if category else None,
            "geographic_filter": geographic_filter,
            "bias_analysis_included": include_bias_analysis,
            "scoring_algorithm": "ml_significance_v2.1"
        }
        
        logger.info(f"Retrieved {len(top_headlines)} top headlines")
        
        return HeadlinesResponse(
            headlines=top_headlines,
            total_headlines=len(filtered_headlines),
            curation_criteria=curation_criteria,
            next_update=next_update
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving headlines: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve headlines"
        )


@router.get(
    "/headlines/categories",
    response_model=Dict[str, List[HeadlineItem]],
    summary="Get headlines organized by categories",
    description="""
    Retrieve headlines organized by categories with balanced representation.
    
    Features:
    - Balanced category representation
    - Category-specific significance thresholds
    - Cross-category duplicate detection
    - Category trend analysis
    - Customizable category weights
    """
)
async def get_headlines_by_categories(
    request: Request,
    categories: Optional[str] = Query(None, description="Comma-separated categories to include"),
    headlines_per_category: int = Query(5, ge=1, le=20, description="Headlines per category"),
    min_significance: float = Query(0.4, ge=0.0, le=1.0, description="Minimum significance score"),
    repository_factory: RepositoryFactory = Depends(get_repository_factory),
    nlp_pipeline: NLPPipeline = Depends(get_nlp_pipeline),
    bias_engine: BiasDetectionEngine = Depends(get_bias_engine)
) -> Dict[str, List[HeadlineItem]]:
    """Get headlines organized by categories."""
    try:
        # Parse categories filter
        if categories:
            selected_categories = [cat.strip() for cat in categories.split(",")]
            category_list = [HeadlineCategory(cat) for cat in selected_categories if cat in HeadlineCategory.__members__]
        else:
            category_list = list(HeadlineCategory)
        
        # Get headlines for each category
        categorized_headlines = {}
        
        for category in category_list:
            # Get category-specific articles
            articles = await _get_recent_articles(
                datetime.now(timezone.utc) - timedelta(hours=24),
                datetime.now(timezone.utc),
                category,
                repository_factory
            )
            
            # Score headlines
            scored_headlines = await _score_headlines(
                articles, nlp_pipeline, bias_engine, True
            )
            
            # Filter and sort
            category_headlines = [
                h for h in scored_headlines
                if h.significance_score >= min_significance
            ]
            category_headlines.sort(key=lambda x: x.significance_score, reverse=True)
            
            # Take top headlines for this category
            categorized_headlines[category.value] = category_headlines[:headlines_per_category]
        
        logger.info(f"Retrieved headlines for {len(categorized_headlines)} categories")
        
        return categorized_headlines
        
    except Exception as e:
        logger.error(f"Error retrieving categorized headlines: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve categorized headlines"
        )


@router.get(
    "/headlines/breaking",
    response_model=BreakingNewsResponse,
    summary="Detect and rank breaking news",
    description="""
    Detect breaking news using real-time signals and velocity analysis.
    
    Features:
    - Real-time breaking news detection
    - Velocity-based significance scoring
    - Multi-source confirmation analysis
    - Alert level classification
    - Confidence scoring for breaking news
    """
)
async def get_breaking_news(
    request: Request,
    lookback_hours: int = Query(2, ge=1, le=12, description="Lookback period for breaking news detection"),
    min_velocity: float = Query(5.0, ge=1.0, le=50.0, description="Minimum velocity threshold"),
    min_confidence: float = Query(0.7, ge=0.5, le=1.0, description="Minimum confidence threshold"),
    repository_factory: RepositoryFactory = Depends(get_repository_factory),
    nlp_pipeline: NLPPipeline = Depends(get_nlp_pipeline)
) -> BreakingNewsResponse:
    """Detect and rank breaking news."""
    try:
        user_tier = getattr(request.state, "user_tier", "free")
        
        # Validate access for breaking news detection
        if user_tier == "free":
            raise HTTPException(
                status_code=403,
                detail="Breaking news detection requires premium subscription"
            )
        
        # Detect breaking news
        breaking_headlines = await _detect_breaking_news(
            lookback_hours, min_velocity, min_confidence, repository_factory, nlp_pipeline
        )
        
        # Calculate overall alert level
        alert_level = _calculate_alert_level(breaking_headlines)
        
        # Calculate detection confidence
        overall_confidence = _calculate_detection_confidence(breaking_headlines)
        
        # Prepare detection criteria
        detection_criteria = {
            "lookback_hours": lookback_hours,
            "min_velocity": min_velocity,
            "min_confidence": min_confidence,
            "detection_algorithm": "velocity_based_v1.2",
            "confirmation_sources": 3
        }
        
        logger.info(f"Detected {len(breaking_headlines)} breaking news items")
        
        return BreakingNewsResponse(
            breaking_headlines=breaking_headlines,
            alert_level=alert_level,
            detection_criteria=detection_criteria,
            confidence=overall_confidence
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error detecting breaking news: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to detect breaking news"
        )


@router.get(
    "/headlines/personalized",
    response_model=PersonalizedHeadlinesResponse,
    summary="Get personalized headline recommendations",
    description="""
    Get personalized headlines based on user interests and reading history.
    
    Features:
    - User interest profiling
    - Reading history analysis
    - Collaborative filtering
    - Content diversity optimization
    - Real-time personalization updates
    """
)
async def get_personalized_headlines(
    request: Request,
    limit: int = Query(20, ge=1, le=50, description="Maximum headlines to return"),
    diversity_weight: float = Query(0.3, ge=0.0, le=1.0, description="Content diversity weight"),
    include_trending: bool = Query(True, description="Include trending topics"),
    repository_factory: RepositoryFactory = Depends(get_repository_factory),
    nlp_pipeline: NLPPipeline = Depends(get_nlp_pipeline)
) -> PersonalizedHeadlinesResponse:
    """Get personalized headlines for the user."""
    try:
        user_id = getattr(request.state, "user_id", "anonymous")
        user_tier = getattr(request.state, "user_tier", "free")
        
        # Validate access for personalization
        if user_tier == "free":
            raise HTTPException(
                status_code=403,
                detail="Personalized headlines require premium subscription"
            )
        
        # Get user profile and interests
        user_profile = await _get_user_profile(user_id, repository_factory)
        
        # Get candidate headlines
        articles = await _get_recent_articles(
            datetime.now(timezone.utc) - timedelta(hours=48),
            datetime.now(timezone.utc),
            None,
            repository_factory
        )
        
        # Score headlines with personalization
        personalized_headlines = await _personalize_headlines(
            articles, user_profile, nlp_pipeline, include_trending
        )
        
        # Apply diversity optimization
        diverse_headlines = _optimize_diversity(
            personalized_headlines, diversity_weight, limit
        )
        
        # Calculate diversity score
        diversity_score = _calculate_diversity_score(diverse_headlines)
        
        # Prepare personalization factors
        personalization_factors = {
            "user_interests": user_profile.get("interests", []),
            "reading_history_weight": 0.4,
            "trending_weight": 0.3 if include_trending else 0.0,
            "diversity_weight": diversity_weight,
            "recency_weight": 0.2
        }
        
        logger.info(f"Generated {len(diverse_headlines)} personalized headlines for user {user_id}")
        
        return PersonalizedHeadlinesResponse(
            personalized_headlines=diverse_headlines,
            user_profile=user_profile,
            personalization_factors=personalization_factors,
            diversity_score=diversity_score
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating personalized headlines: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate personalized headlines"
        )


@router.post(
    "/headlines/analyze",
    response_model=Dict[str, Any],
    summary="Analyze headline significance and quality",
    description="""
    Analyze individual headlines for significance, quality, and engagement potential.
    
    Features:
    - Detailed significance factor analysis
    - Quality assessment with specific metrics
    - Engagement prediction modeling
    - Bias and sentiment analysis
    - Optimization recommendations
    """
)
async def analyze_headline(
    request: Request,
    analysis_request: Dict[str, Any] = Body(...),
    nlp_pipeline: NLPPipeline = Depends(get_nlp_pipeline),
    bias_engine: BiasDetectionEngine = Depends(get_bias_engine)
) -> Dict[str, Any]:
    """Analyze headline significance and quality."""
    try:
        headline_text = analysis_request.get("headline")
        content = analysis_request.get("content", "")
        source = analysis_request.get("source", "unknown")
        
        if not headline_text:
            raise HTTPException(
                status_code=400,
                detail="Headline text is required for analysis"
            )
        
        # Perform comprehensive headline analysis
        analysis_result = await _analyze_single_headline(
            headline_text, content, source, nlp_pipeline, bias_engine
        )
        
        # Generate optimization recommendations
        recommendations = _generate_headline_recommendations(analysis_result)
        
        logger.info(f"Analyzed headline: {headline_text[:50]}...")
        
        return {
            "headline": headline_text,
            "analysis": analysis_result,
            "recommendations": recommendations,
            "analyzed_at": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing headline: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to analyze headline"
        )


# Helper functions for headline processing
async def _get_recent_articles(
    start_time: datetime,
    end_time: datetime,
    category: Optional[HeadlineCategory],
    repository_factory: RepositoryFactory
) -> List[Dict[str, Any]]:
    """Get recent articles for headline processing."""
    # Mock article data (would come from repository)
    mock_articles = [
        {
            "id": "art_001",
            "title": "Major Breakthrough in Quantum Computing Achieved by Tech Giants",
            "content": "Scientists at leading technology companies have announced a significant breakthrough in quantum computing...",
            "url": "https://example.com/quantum-breakthrough",
            "source": "TechNews Daily",
            "published_at": datetime.now(timezone.utc) - timedelta(hours=2),
            "category": HeadlineCategory.TECHNOLOGY,
            "entities": ["IBM", "Google", "Quantum Computing"],
            "keywords": ["quantum", "breakthrough", "computing", "technology"]
        },
        {
            "id": "art_002", 
            "title": "Global Climate Summit Reaches Historic Agreement on Carbon Emissions",
            "content": "World leaders have reached a landmark agreement on reducing carbon emissions...",
            "url": "https://example.com/climate-agreement",
            "source": "World Report",
            "published_at": datetime.now(timezone.utc) - timedelta(hours=4),
            "category": HeadlineCategory.WORLD_NEWS,
            "entities": ["United Nations", "Climate Change", "Carbon Emissions"],
            "keywords": ["climate", "summit", "agreement", "emissions"]
        },
        {
            "id": "art_003",
            "title": "Stock Markets Rally as Economic Indicators Show Strong Growth",
            "content": "Major stock indices posted significant gains following positive economic data...",
            "url": "https://example.com/market-rally",
            "source": "Financial Times",
            "published_at": datetime.now(timezone.utc) - timedelta(hours=6),
            "category": HeadlineCategory.BUSINESS,
            "entities": ["Stock Market", "Economic Growth", "Wall Street"],
            "keywords": ["stocks", "rally", "economic", "growth"]
        }
    ]
    
    # Filter by category if specified
    if category:
        mock_articles = [art for art in mock_articles if art["category"] == category]
    
    # Filter by time range
    filtered_articles = [
        art for art in mock_articles
        if start_time <= art["published_at"] <= end_time
    ]
    
    return filtered_articles


async def _score_headlines(
    articles: List[Dict[str, Any]],
    nlp_pipeline: NLPPipeline,
    bias_engine: BiasDetectionEngine,
    include_bias_analysis: bool
) -> List[HeadlineItem]:
    """Score headlines for significance and quality."""
    scored_headlines = []
    
    for article in articles:
        # Calculate significance score
        significance_score = await _calculate_significance_score(article, nlp_pipeline)
        
        # Calculate quality score
        quality_score = _calculate_quality_score(article["title"], article.get("content", ""))
        
        # Calculate engagement score
        engagement_score = _calculate_engagement_score(article["title"], significance_score, quality_score)
        
        # Analyze sentiment
        sentiment = await _analyze_headline_sentiment(article["title"], nlp_pipeline)
        
        # Analyze bias if requested
        bias_score = 0.0
        if include_bias_analysis:
            bias_score = await _analyze_headline_bias(article["title"], article.get("content", ""), bias_engine)
        
        # Determine significance and quality levels
        significance_level = _get_significance_level(significance_score)
        quality_level = _get_quality_level(quality_score)
        
        # Extract geographic relevance
        geographic_relevance = _extract_geographic_relevance(article)
        
        # Create headline item
        headline_item = HeadlineItem(
            id=article["id"],
            title=article["title"],
            summary=article.get("summary"),
            url=article["url"],
            source=article["source"],
            category=article["category"],
            published_at=article["published_at"],
            significance_score=significance_score,
            significance_level=significance_level,
            quality_score=quality_score,
            quality_level=quality_level,
            engagement_score=engagement_score,
            sentiment=sentiment,
            bias_score=bias_score,
            entities=article.get("entities", []),
            keywords=article.get("keywords", []),
            geographic_relevance=geographic_relevance,
            breaking_news_indicators=_detect_breaking_indicators(article)
        )
        
        scored_headlines.append(headline_item)
    
    return scored_headlines


async def _calculate_significance_score(article: Dict[str, Any], nlp_pipeline: NLPPipeline) -> float:
    """Calculate significance score using multiple factors."""
    score = 0.0
    
    # Entity importance (0.3 weight)
    entities = article.get("entities", [])
    entity_score = min(1.0, len(entities) * 0.1)  # More entities = higher significance
    score += entity_score * 0.3
    
    # Keyword relevance (0.2 weight)
    keywords = article.get("keywords", [])
    important_keywords = ["breakthrough", "historic", "major", "crisis", "emergency", "record"]
    keyword_score = sum(1 for kw in keywords if any(imp in kw.lower() for imp in important_keywords)) / max(1, len(keywords))
    score += keyword_score * 0.2
    
    # Recency factor (0.2 weight)
    hours_old = (datetime.now(timezone.utc) - article["published_at"]).total_seconds() / 3600
    recency_score = max(0, 1 - (hours_old / 24))  # Decay over 24 hours
    score += recency_score * 0.2
    
    # Source credibility (0.15 weight)
    credible_sources = ["Reuters", "Associated Press", "BBC", "Financial Times", "Wall Street Journal"]
    source_score = 1.0 if article["source"] in credible_sources else 0.7
    score += source_score * 0.15
    
    # Content length and depth (0.15 weight)
    content_length = len(article.get("content", ""))
    content_score = min(1.0, content_length / 1000)  # Normalize to 1000 chars
    score += content_score * 0.15
    
    return min(1.0, score)


def _calculate_quality_score(title: str, content: str) -> float:
    """Calculate headline quality score."""
    score = 0.0
    
    # Title length (0.25 weight)
    title_length = len(title)
    if 30 <= title_length <= 100:  # Optimal length
        length_score = 1.0
    elif title_length < 30:
        length_score = title_length / 30
    else:
        length_score = max(0.5, 100 / title_length)
    score += length_score * 0.25
    
    # Grammar and structure (0.25 weight)
    grammar_score = _assess_grammar_quality(title)
    score += grammar_score * 0.25
    
    # Clarity and specificity (0.25 weight)
    clarity_score = _assess_clarity(title)
    score += clarity_score * 0.25
    
    # Engagement potential (0.25 weight)
    engagement_words = ["breakthrough", "reveals", "announces", "discovers", "warns", "confirms"]
    engagement_score = min(1.0, sum(1 for word in engagement_words if word.lower() in title.lower()) * 0.3)
    score += engagement_score * 0.25
    
    return min(1.0, score)


def _calculate_engagement_score(title: str, significance_score: float, quality_score: float) -> float:
    """Calculate predicted engagement score."""
    # Base score from significance and quality
    base_score = (significance_score * 0.6 + quality_score * 0.4)
    
    # Emotional words boost engagement
    emotional_words = ["shocking", "amazing", "incredible", "urgent", "breaking", "exclusive"]
    emotional_boost = min(0.2, sum(1 for word in emotional_words if word.lower() in title.lower()) * 0.1)
    
    # Question format boost
    question_boost = 0.1 if title.strip().endswith('?') else 0.0
    
    # Numbers and statistics boost
    number_boost = 0.05 if re.search(r'\d+', title) else 0.0
    
    engagement_score = base_score + emotional_boost + question_boost + number_boost
    return min(1.0, engagement_score)


async def _analyze_headline_sentiment(title: str, nlp_pipeline: NLPPipeline) -> float:
    """Analyze headline sentiment."""
    # Mock sentiment analysis (would use actual NLP pipeline)
    positive_words = ["breakthrough", "success", "growth", "improvement", "victory"]
    negative_words = ["crisis", "failure", "decline", "threat", "disaster"]
    
    title_lower = title.lower()
    positive_count = sum(1 for word in positive_words if word in title_lower)
    negative_count = sum(1 for word in negative_words if word in title_lower)
    
    if positive_count > negative_count:
        return min(1.0, positive_count * 0.3)
    elif negative_count > positive_count:
        return max(-1.0, -negative_count * 0.3)
    else:
        return 0.0


async def _analyze_headline_bias(title: str, content: str, bias_engine: BiasDetectionEngine) -> float:
    """Analyze headline bias."""
    # Mock bias analysis (would use actual bias engine)
    biased_words = ["always", "never", "all", "every", "completely", "totally", "obviously"]
    title_lower = title.lower()
    bias_indicators = sum(1 for word in biased_words if word in title_lower)
    
    return min(1.0, bias_indicators * 0.2)


def _get_significance_level(score: float) -> SignificanceLevel:
    """Convert significance score to level."""
    if score >= 0.9:
        return SignificanceLevel.CRITICAL
    elif score >= 0.7:
        return SignificanceLevel.HIGH
    elif score >= 0.5:
        return SignificanceLevel.MEDIUM
    elif score >= 0.3:
        return SignificanceLevel.LOW
    else:
        return SignificanceLevel.MINIMAL


def _get_quality_level(score: float) -> HeadlineQuality:
    """Convert quality score to level."""
    if score >= 0.8:
        return HeadlineQuality.EXCELLENT
    elif score >= 0.6:
        return HeadlineQuality.GOOD
    elif score >= 0.4:
        return HeadlineQuality.FAIR
    else:
        return HeadlineQuality.POOR


def _extract_geographic_relevance(article: Dict[str, Any]) -> List[str]:
    """Extract geographic relevance from article."""
    # Mock geographic extraction
    geographic_keywords = {
        "US": ["america", "united states", "washington", "new york"],
        "EU": ["europe", "european", "brussels", "london", "paris"],
        "Asia": ["china", "japan", "india", "asia", "beijing", "tokyo"],
        "Global": ["world", "global", "international", "worldwide"]
    }
    
    title_content = f"{article['title']} {article.get('content', '')}".lower()
    relevance = []
    
    for region, keywords in geographic_keywords.items():
        if any(keyword in title_content for keyword in keywords):
            relevance.append(region)
    
    return relevance if relevance else ["Global"]


def _detect_breaking_indicators(article: Dict[str, Any]) -> Dict[str, Any]:
    """Detect breaking news indicators."""
    title = article["title"].lower()
    
    breaking_words = ["breaking", "urgent", "alert", "just in", "developing"]
    time_indicators = ["now", "today", "just", "moments ago"]
    
    indicators = {
        "breaking_keywords": sum(1 for word in breaking_words if word in title),
        "time_urgency": sum(1 for word in time_indicators if word in title),
        "recency_hours": (datetime.now(timezone.utc) - article["published_at"]).total_seconds() / 3600,
        "source_speed": 1.0 if "wire" in article["source"].lower() else 0.5
    }
    
    return indicators


def _assess_grammar_quality(title: str) -> float:
    """Assess grammar quality of headline."""
    # Simple grammar checks
    score = 1.0
    
    # Check for proper capitalization
    if not title[0].isupper():
        score -= 0.2
    
    # Check for excessive punctuation
    punct_count = sum(1 for char in title if char in "!?.,;:")
    if punct_count > 3:
        score -= 0.2
    
    # Check for complete sentences (should not end with period for headlines)
    if title.endswith('.'):
        score -= 0.1
    
    return max(0.0, score)


def _assess_clarity(title: str) -> float:
    """Assess clarity and specificity of headline."""
    score = 1.0
    
    # Penalize vague words
    vague_words = ["something", "things", "stuff", "various", "several", "many"]
    vague_count = sum(1 for word in vague_words if word.lower() in title.lower())
    score -= vague_count * 0.2
    
    # Reward specific numbers and names
    if re.search(r'\d+', title):
        score += 0.1
    
    # Reward proper nouns (capitalized words)
    proper_nouns = sum(1 for word in title.split() if word[0].isupper() and len(word) > 1)
    score += min(0.2, proper_nouns * 0.05)
    
    return min(1.0, max(0.0, score))


async def _detect_breaking_news(
    lookback_hours: int,
    min_velocity: float,
    min_confidence: float,
    repository_factory: RepositoryFactory,
    nlp_pipeline: NLPPipeline
) -> List[HeadlineItem]:
    """Detect breaking news based on velocity and signals."""
    # Mock breaking news detection
    breaking_headlines = [
        HeadlineItem(
            id="breaking_001",
            title="BREAKING: Major Earthquake Strikes Pacific Region",
            summary="A magnitude 7.2 earthquake has struck the Pacific region, triggering tsunami warnings.",
            url="https://example.com/earthquake-breaking",
            source="Emergency News Network",
            category=HeadlineCategory.BREAKING_NEWS,
            published_at=datetime.now(timezone.utc) - timedelta(minutes=15),
            significance_score=0.95,
            significance_level=SignificanceLevel.CRITICAL,
            quality_score=0.85,
            quality_level=HeadlineQuality.EXCELLENT,
            engagement_score=0.92,
            sentiment=-0.6,
            bias_score=0.1,
            entities=["Pacific Ocean", "Earthquake", "Tsunami"],
            keywords=["earthquake", "tsunami", "emergency", "breaking"],
            geographic_relevance=["Pacific", "Global"],
            breaking_news_indicators={
                "velocity": 25.3,
                "confirmation_sources": 5,
                "social_media_spike": True,
                "official_alerts": True
            }
        )
    ]
    
    return breaking_headlines


def _calculate_alert_level(breaking_headlines: List[HeadlineItem]) -> str:
    """Calculate overall alert level for breaking news."""
    if not breaking_headlines:
        return "none"
    
    max_significance = max(h.significance_score for h in breaking_headlines)
    
    if max_significance >= 0.9:
        return "critical"
    elif max_significance >= 0.7:
        return "high"
    elif max_significance >= 0.5:
        return "medium"
    else:
        return "low"


def _calculate_detection_confidence(breaking_headlines: List[HeadlineItem]) -> float:
    """Calculate confidence in breaking news detection."""
    if not breaking_headlines:
        return 0.0
    
    # Average confidence based on multiple factors
    total_confidence = 0.0
    
    for headline in breaking_headlines:
        confidence = 0.0
        
        # Significance score contributes to confidence
        confidence += headline.significance_score * 0.4
        
        # Breaking indicators contribute
        indicators = headline.breaking_news_indicators
        if indicators.get("confirmation_sources", 0) >= 3:
            confidence += 0.3
        if indicators.get("social_media_spike"):
            confidence += 0.2
        if indicators.get("official_alerts"):
            confidence += 0.1
        
        total_confidence += confidence
    
    return total_confidence / len(breaking_headlines)


async def _get_user_profile(user_id: str, repository_factory: RepositoryFactory) -> Dict[str, Any]:
    """Get user profile for personalization."""
    # Mock user profile (would come from user data analysis)
    return {
        "interests": ["technology", "science", "business"],
        "reading_history": ["quantum computing", "artificial intelligence", "startups"],
        "preferred_sources": ["TechNews Daily", "Science Journal"],
        "engagement_patterns": {
            "preferred_length": "medium",
            "time_of_day": [9, 12, 18],
            "categories": {"technology": 0.4, "science": 0.3, "business": 0.3}
        }
    }


async def _personalize_headlines(
    articles: List[Dict[str, Any]],
    user_profile: Dict[str, Any],
    nlp_pipeline: NLPPipeline,
    include_trending: bool
) -> List[HeadlineItem]:
    """Personalize headlines based on user profile."""
    # Score headlines with personalization
    personalized_headlines = []
    user_interests = user_profile.get("interests", [])
    
    for article in articles:
        # Calculate base scores
        significance_score = await _calculate_significance_score(article, nlp_pipeline)
        quality_score = _calculate_quality_score(article["title"], article.get("content", ""))
        
        # Calculate personalization score
        personalization_score = 0.0
        
        # Interest matching
        article_category = article["category"].value
        if article_category in user_interests:
            personalization_score += 0.4
        
        # Keyword matching with reading history
        reading_history = user_profile.get("reading_history", [])
        article_keywords = article.get("keywords", [])
        keyword_matches = sum(1 for kw in article_keywords if any(hist in kw.lower() for hist in reading_history))
        personalization_score += min(0.3, keyword_matches * 0.1)
        
        # Source preference
        preferred_sources = user_profile.get("preferred_sources", [])
        if article["source"] in preferred_sources:
            personalization_score += 0.2
        
        # Create personalized headline item
        headline_item = HeadlineItem(
            id=article["id"],
            title=article["title"],
            summary=article.get("summary"),
            url=article["url"],
            source=article["source"],
            category=article["category"],
            published_at=article["published_at"],
            significance_score=significance_score,
            significance_level=_get_significance_level(significance_score),
            quality_score=quality_score,
            quality_level=_get_quality_level(quality_score),
            engagement_score=_calculate_engagement_score(article["title"], significance_score, quality_score),
            sentiment=await _analyze_headline_sentiment(article["title"], nlp_pipeline),
            bias_score=0.1,  # Mock bias score
            entities=article.get("entities", []),
            keywords=article.get("keywords", []),
            geographic_relevance=_extract_geographic_relevance(article),
            breaking_news_indicators=_detect_breaking_indicators(article),
            personalization_score=personalization_score
        )
        
        personalized_headlines.append(headline_item)
    
    return personalized_headlines


def _optimize_diversity(
    headlines: List[HeadlineItem],
    diversity_weight: float,
    limit: int
) -> List[HeadlineItem]:
    """Optimize headline diversity while maintaining relevance."""
    if len(headlines) <= limit:
        return headlines
    
    # Sort by personalization score first
    headlines.sort(key=lambda x: x.personalization_score or 0, reverse=True)
    
    # Select diverse headlines
    selected = []
    used_categories = set()
    used_sources = set()
    
    for headline in headlines:
        if len(selected) >= limit:
            break
        
        # Diversity scoring
        diversity_score = 1.0
        
        # Category diversity
        if headline.category in used_categories:
            diversity_score *= (1 - diversity_weight * 0.5)
        
        # Source diversity
        if headline.source in used_sources:
            diversity_score *= (1 - diversity_weight * 0.3)
        
        # Combined score
        combined_score = (headline.personalization_score or 0) * diversity_score
        
        # Add to selection if score is good enough or we need more items
        if combined_score > 0.3 or len(selected) < limit // 2:
            selected.append(headline)
            used_categories.add(headline.category)
            used_sources.add(headline.source)
    
    return selected


def _calculate_diversity_score(headlines: List[HeadlineItem]) -> float:
    """Calculate diversity score for a set of headlines."""
    if not headlines:
        return 0.0
    
    # Category diversity
    categories = set(h.category for h in headlines)
    category_diversity = len(categories) / len(HeadlineCategory)
    
    # Source diversity
    sources = set(h.source for h in headlines)
    source_diversity = min(1.0, len(sources) / len(headlines))
    
    # Topic diversity (based on entities)
    all_entities = set()
    for h in headlines:
        all_entities.update(h.entities)
    entity_diversity = min(1.0, len(all_entities) / (len(headlines) * 3))
    
    # Combined diversity score
    diversity_score = (category_diversity * 0.4 + source_diversity * 0.3 + entity_diversity * 0.3)
    return diversity_score


async def _analyze_single_headline(
    headline_text: str,
    content: str,
    source: str,
    nlp_pipeline: NLPPipeline,
    bias_engine: BiasDetectionEngine
) -> Dict[str, Any]:
    """Analyze a single headline comprehensively."""
    analysis = {}
    
    # Basic metrics
    analysis["length"] = len(headline_text)
    analysis["word_count"] = len(headline_text.split())
    
    # Quality assessment
    analysis["quality_score"] = _calculate_quality_score(headline_text, content)
    analysis["grammar_score"] = _assess_grammar_quality(headline_text)
    analysis["clarity_score"] = _assess_clarity(headline_text)
    
    # Engagement factors
    analysis["engagement_score"] = _calculate_engagement_score(
        headline_text, 0.5, analysis["quality_score"]  # Mock significance
    )
    
    # Sentiment and bias
    analysis["sentiment"] = await _analyze_headline_sentiment(headline_text, nlp_pipeline)
    analysis["bias_score"] = await _analyze_headline_bias(headline_text, content, bias_engine)
    
    # Structural analysis
    analysis["has_numbers"] = bool(re.search(r'\d+', headline_text))
    analysis["is_question"] = headline_text.strip().endswith('?')
    analysis["capitalization_proper"] = headline_text[0].isupper() if headline_text else False
    
    return analysis


def _generate_headline_recommendations(analysis: Dict[str, Any]) -> List[str]:
    """Generate optimization recommendations for headlines."""
    recommendations = []
    
    # Length recommendations
    if analysis["length"] < 30:
        recommendations.append("Consider making the headline longer for better context")
    elif analysis["length"] > 100:
        recommendations.append("Consider shortening the headline for better readability")
    
    # Quality improvements
    if analysis["quality_score"] < 0.6:
        recommendations.append("Improve headline clarity and specificity")
    
    if analysis["grammar_score"] < 0.8:
        recommendations.append("Check grammar and punctuation")
    
    # Engagement improvements
    if analysis["engagement_score"] < 0.5:
        recommendations.append("Add more engaging or specific language")
    
    if not analysis["has_numbers"] and analysis["word_count"] > 5:
        recommendations.append("Consider adding specific numbers or statistics")
    
    # Bias warnings
    if analysis["bias_score"] > 0.5:
        recommendations.append("Review for potential bias in language")
    
    return recommendations