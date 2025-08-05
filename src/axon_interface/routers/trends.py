"""
Trends API Endpoints - Real-time Trend Analysis

Implements comprehensive trend analysis and monitoring:
- GET /trends - Real-time trending topics and entities
- GET /trends/velocity - Trend velocity and momentum analysis
- GET /trends/volume - Content volume and frequency analysis
- GET /trends/emerging - Emerging trend detection
- POST /trends/analyze - Custom trend analysis

Provides real-time trend identification with velocity calculations and background processing.
"""

import logging
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timezone, timedelta
from enum import Enum
import asyncio
from collections import defaultdict, Counter
import math

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Body, BackgroundTasks
from pydantic import BaseModel, Field, validator

from ...shared.schemas import PaginatedResponse, PaginationInfo
from ...thalamus.nlp_pipeline import get_nlp_pipeline, NLPPipeline
from ...thalamus.bias_analysis import get_bias_engine, BiasDetectionEngine
from ..dependencies import get_repository_factory
from ...synaptic_vesicle.repositories import RepositoryFactory


logger = logging.getLogger(__name__)
router = APIRouter()


class TrendTimeframe(str, Enum):
    """Timeframes for trend analysis."""
    HOUR = "1h"
    SIX_HOURS = "6h"
    DAY = "24h"
    WEEK = "7d"
    MONTH = "30d"


class TrendCategory(str, Enum):
    """Categories for trend classification."""
    BREAKING_NEWS = "breaking_news"
    TECHNOLOGY = "technology"
    POLITICS = "politics"
    BUSINESS = "business"
    ENTERTAINMENT = "entertainment"
    SPORTS = "sports"
    SCIENCE = "science"
    HEALTH = "health"
    GENERAL = "general"


class TrendType(str, Enum):
    """Types of trends that can be detected."""
    ENTITY = "entity"
    TOPIC = "topic"
    KEYWORD = "keyword"
    HASHTAG = "hashtag"
    PHRASE = "phrase"


class TrendingItem(BaseModel):
    """Model for a trending item."""
    text: str = Field(..., description="Trending text/entity")
    trend_type: TrendType = Field(..., description="Type of trending item")
    category: TrendCategory = Field(..., description="Trend category")
    score: float = Field(..., description="Trend score (0-1)")
    velocity: float = Field(..., description="Trend velocity (mentions per hour)")
    volume: int = Field(..., description="Total mentions in timeframe")
    sentiment: float = Field(..., description="Average sentiment (-1 to 1)")
    confidence: float = Field(..., description="Confidence in trend detection")
    first_seen: datetime = Field(..., description="First mention timestamp")
    peak_time: datetime = Field(..., description="Peak activity timestamp")
    related_entities: List[str] = Field(default_factory=list, description="Related entities")
    sample_articles: List[str] = Field(default_factory=list, description="Sample article IDs")
    geographic_distribution: Dict[str, int] = Field(default_factory=dict, description="Geographic mention distribution")


class TrendAnalysisResponse(BaseModel):
    """Response model for trend analysis."""
    timeframe: TrendTimeframe = Field(..., description="Analysis timeframe")
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Analysis timestamp")
    trending_items: List[TrendingItem] = Field(..., description="Trending items")
    total_trends: int = Field(..., description="Total number of trends detected")
    analysis_stats: Dict[str, Any] = Field(..., description="Analysis statistics")
    background_processing: bool = Field(default=True, description="Whether background processing is active")


class VelocityAnalysisResponse(BaseModel):
    """Response model for velocity analysis."""
    item: str = Field(..., description="Analyzed item")
    current_velocity: float = Field(..., description="Current velocity (mentions/hour)")
    peak_velocity: float = Field(..., description="Peak velocity in timeframe")
    acceleration: float = Field(..., description="Velocity acceleration")
    momentum_score: float = Field(..., description="Momentum score (0-1)")
    velocity_history: List[Dict[str, Any]] = Field(..., description="Historical velocity data")
    prediction: Dict[str, Any] = Field(..., description="Velocity prediction")


class VolumeAnalysisResponse(BaseModel):
    """Response model for volume analysis."""
    timeframe: TrendTimeframe = Field(..., description="Analysis timeframe")
    total_volume: int = Field(..., description="Total content volume")
    volume_by_category: Dict[str, int] = Field(..., description="Volume by category")
    volume_by_hour: List[Dict[str, Any]] = Field(..., description="Hourly volume breakdown")
    peak_hours: List[int] = Field(..., description="Peak activity hours")
    volume_trends: Dict[str, float] = Field(..., description="Volume trend indicators")


class EmergingTrendResponse(BaseModel):
    """Response model for emerging trend detection."""
    emerging_trends: List[TrendingItem] = Field(..., description="Newly emerging trends")
    growth_rate_threshold: float = Field(..., description="Growth rate threshold used")
    detection_criteria: Dict[str, Any] = Field(..., description="Detection criteria used")
    confidence_threshold: float = Field(..., description="Confidence threshold")


@router.get(
    "/trends",
    response_model=TrendAnalysisResponse,
    summary="Get real-time trending topics and entities",
    description="""
    Retrieve current trending topics, entities, and keywords with real-time analysis.
    
    Features:
    - Real-time trend detection with velocity calculations
    - Multi-category trend analysis (news, tech, politics, etc.)
    - Sentiment analysis for trending items
    - Geographic distribution of trends
    - Related entity identification
    - Background processing for continuous updates
    
    Supports multiple timeframes and filtering by category, trend type, and confidence.
    """
)
async def get_trends(
    request: Request,
    timeframe: TrendTimeframe = Query(TrendTimeframe.DAY, description="Analysis timeframe"),
    category: Optional[TrendCategory] = Query(None, description="Filter by category"),
    trend_type: Optional[TrendType] = Query(None, description="Filter by trend type"),
    min_confidence: float = Query(0.5, ge=0.0, le=1.0, description="Minimum confidence threshold"),
    limit: int = Query(50, ge=1, le=200, description="Maximum trends to return"),
    include_sentiment: bool = Query(True, description="Include sentiment analysis"),
    repository_factory: RepositoryFactory = Depends(get_repository_factory),
    nlp_pipeline: NLPPipeline = Depends(get_nlp_pipeline),
    bias_engine: BiasDetectionEngine = Depends(get_bias_engine)
) -> TrendAnalysisResponse:
    """Get current trending topics and entities."""
    try:
        user_tier = getattr(request.state, "user_tier", "free")
        
        # Validate access based on user tier
        if timeframe in [TrendTimeframe.HOUR, TrendTimeframe.SIX_HOURS] and user_tier == "free":
            raise HTTPException(
                status_code=403,
                detail="Real-time trends require premium subscription"
            )
        
        # Calculate timeframe boundaries
        end_time = datetime.now(timezone.utc)
        timeframe_hours = _get_timeframe_hours(timeframe)
        start_time = end_time - timedelta(hours=timeframe_hours)
        
        # Get trending data from background processing
        trending_data = await _get_trending_data(
            start_time, end_time, category, trend_type, min_confidence, limit
        )
        
        # Enhance with sentiment analysis if requested
        if include_sentiment:
            trending_data = await _enhance_with_sentiment(trending_data, nlp_pipeline, bias_engine)
        
        # Calculate analysis statistics
        analysis_stats = _calculate_analysis_stats(trending_data, timeframe_hours)
        
        logger.info(f"Retrieved {len(trending_data)} trends for timeframe {timeframe}")
        
        return TrendAnalysisResponse(
            timeframe=timeframe,
            trending_items=trending_data,
            total_trends=len(trending_data),
            analysis_stats=analysis_stats,
            background_processing=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving trends: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve trending data"
        )


@router.get(
    "/trends/velocity",
    response_model=VelocityAnalysisResponse,
    summary="Analyze trend velocity and momentum",
    description="""
    Analyze the velocity and momentum of specific trending items.
    
    Features:
    - Real-time velocity calculations (mentions per hour)
    - Acceleration and momentum analysis
    - Historical velocity tracking
    - Predictive velocity modeling
    - Peak detection and timing analysis
    """
)
async def analyze_velocity(
    request: Request,
    item: str = Query(..., description="Item to analyze (entity, topic, keyword)"),
    timeframe: TrendTimeframe = Query(TrendTimeframe.DAY, description="Analysis timeframe"),
    granularity: str = Query("hour", regex="^(minute|hour|day)$", description="Data granularity"),
    repository_factory: RepositoryFactory = Depends(get_repository_factory)
) -> VelocityAnalysisResponse:
    """Analyze velocity and momentum for a specific trending item."""
    try:
        # Calculate timeframe boundaries
        end_time = datetime.now(timezone.utc)
        timeframe_hours = _get_timeframe_hours(timeframe)
        start_time = end_time - timedelta(hours=timeframe_hours)
        
        # Get velocity data
        velocity_data = await _calculate_velocity_data(item, start_time, end_time, granularity)
        
        # Calculate momentum and acceleration
        momentum_analysis = _calculate_momentum_analysis(velocity_data)
        
        # Generate velocity prediction
        prediction = _generate_velocity_prediction(velocity_data)
        
        logger.info(f"Analyzed velocity for item: {item}")
        
        return VelocityAnalysisResponse(
            item=item,
            current_velocity=velocity_data["current_velocity"],
            peak_velocity=velocity_data["peak_velocity"],
            acceleration=momentum_analysis["acceleration"],
            momentum_score=momentum_analysis["momentum_score"],
            velocity_history=velocity_data["history"],
            prediction=prediction
        )
        
    except Exception as e:
        logger.error(f"Error analyzing velocity for {item}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to analyze velocity"
        )


@router.get(
    "/trends/volume",
    response_model=VolumeAnalysisResponse,
    summary="Analyze content volume and frequency",
    description="""
    Analyze content volume patterns and frequency distributions.
    
    Features:
    - Total volume analysis across timeframes
    - Category-based volume breakdown
    - Hourly activity patterns
    - Peak detection and timing analysis
    - Volume trend indicators and growth rates
    """
)
async def analyze_volume(
    request: Request,
    timeframe: TrendTimeframe = Query(TrendTimeframe.DAY, description="Analysis timeframe"),
    category: Optional[TrendCategory] = Query(None, description="Filter by category"),
    repository_factory: RepositoryFactory = Depends(get_repository_factory)
) -> VolumeAnalysisResponse:
    """Analyze content volume and frequency patterns."""
    try:
        # Calculate timeframe boundaries
        end_time = datetime.now(timezone.utc)
        timeframe_hours = _get_timeframe_hours(timeframe)
        start_time = end_time - timedelta(hours=timeframe_hours)
        
        # Get volume data
        volume_data = await _calculate_volume_data(start_time, end_time, category)
        
        # Analyze hourly patterns
        hourly_analysis = _analyze_hourly_patterns(volume_data["hourly_data"])
        
        # Calculate volume trends
        volume_trends = _calculate_volume_trends(volume_data["historical_data"])
        
        logger.info(f"Analyzed volume for timeframe {timeframe}")
        
        return VolumeAnalysisResponse(
            timeframe=timeframe,
            total_volume=volume_data["total_volume"],
            volume_by_category=volume_data["by_category"],
            volume_by_hour=volume_data["hourly_data"],
            peak_hours=hourly_analysis["peak_hours"],
            volume_trends=volume_trends
        )
        
    except Exception as e:
        logger.error(f"Error analyzing volume: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to analyze volume"
        )


@router.get(
    "/trends/emerging",
    response_model=EmergingTrendResponse,
    summary="Detect emerging trends",
    description="""
    Detect newly emerging trends with rapid growth patterns.
    
    Features:
    - Early trend detection algorithms
    - Growth rate analysis and thresholds
    - Confidence scoring for emerging trends
    - Noise filtering and validation
    - Predictive trend emergence modeling
    """
)
async def detect_emerging_trends(
    request: Request,
    growth_threshold: float = Query(2.0, ge=1.1, le=10.0, description="Minimum growth rate multiplier"),
    confidence_threshold: float = Query(0.7, ge=0.5, le=1.0, description="Minimum confidence threshold"),
    min_volume: int = Query(10, ge=5, le=100, description="Minimum mention volume"),
    lookback_hours: int = Query(6, ge=1, le=24, description="Lookback period for comparison"),
    repository_factory: RepositoryFactory = Depends(get_repository_factory),
    nlp_pipeline: NLPPipeline = Depends(get_nlp_pipeline)
) -> EmergingTrendResponse:
    """Detect emerging trends with rapid growth."""
    try:
        user_tier = getattr(request.state, "user_tier", "free")
        
        # Validate access for emerging trend detection
        if user_tier == "free":
            raise HTTPException(
                status_code=403,
                detail="Emerging trend detection requires premium subscription"
            )
        
        # Calculate analysis periods
        current_time = datetime.now(timezone.utc)
        current_period_start = current_time - timedelta(hours=lookback_hours)
        previous_period_start = current_period_start - timedelta(hours=lookback_hours)
        
        # Detect emerging trends
        emerging_trends = await _detect_emerging_trends(
            previous_period_start,
            current_period_start,
            current_time,
            growth_threshold,
            confidence_threshold,
            min_volume,
            nlp_pipeline
        )
        
        # Prepare detection criteria
        detection_criteria = {
            "growth_threshold": growth_threshold,
            "min_volume": min_volume,
            "lookback_hours": lookback_hours,
            "analysis_method": "comparative_growth_rate"
        }
        
        logger.info(f"Detected {len(emerging_trends)} emerging trends")
        
        return EmergingTrendResponse(
            emerging_trends=emerging_trends,
            growth_rate_threshold=growth_threshold,
            detection_criteria=detection_criteria,
            confidence_threshold=confidence_threshold
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error detecting emerging trends: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to detect emerging trends"
        )


@router.post(
    "/trends/analyze",
    response_model=Dict[str, Any],
    summary="Custom trend analysis",
    description="""
    Perform custom trend analysis with user-defined parameters.
    
    Features:
    - Custom entity and keyword tracking
    - Flexible timeframe and granularity settings
    - Advanced filtering and aggregation options
    - Comparative analysis between items
    - Export capabilities for analysis results
    """
)
async def custom_trend_analysis(
    request: Request,
    background_tasks: BackgroundTasks,
    analysis_request: Dict[str, Any] = Body(...),
    repository_factory: RepositoryFactory = Depends(get_repository_factory),
    nlp_pipeline: NLPPipeline = Depends(get_nlp_pipeline)
) -> Dict[str, Any]:
    """Perform custom trend analysis."""
    try:
        user_tier = getattr(request.state, "user_tier", "free")
        
        # Validate request parameters
        items = analysis_request.get("items", [])
        timeframe = analysis_request.get("timeframe", "24h")
        analysis_type = analysis_request.get("analysis_type", "comparative")
        
        if not items:
            raise HTTPException(
                status_code=400,
                detail="At least one item must be specified for analysis"
            )
        
        # Validate item limits based on user tier
        max_items = {"free": 3, "premium": 20, "enterprise": 100}.get(user_tier, 3)
        if len(items) > max_items:
            raise HTTPException(
                status_code=400,
                detail=f"Maximum {max_items} items allowed for {user_tier} tier"
            )
        
        # Perform custom analysis
        analysis_results = await _perform_custom_analysis(
            items, timeframe, analysis_type, analysis_request, nlp_pipeline
        )
        
        # Schedule background processing for detailed analysis
        if analysis_request.get("detailed_analysis", False):
            background_tasks.add_task(
                _schedule_detailed_analysis,
                items,
                analysis_request,
                getattr(request.state, "user_id", "anonymous")
            )
        
        logger.info(f"Completed custom trend analysis for {len(items)} items")
        
        return {
            "analysis_id": f"custom_{int(datetime.now().timestamp())}",
            "items_analyzed": len(items),
            "timeframe": timeframe,
            "analysis_type": analysis_type,
            "results": analysis_results,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "background_processing": analysis_request.get("detailed_analysis", False)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in custom trend analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to perform custom trend analysis"
        )


# Helper functions for trend analysis
def _get_timeframe_hours(timeframe: TrendTimeframe) -> int:
    """Convert timeframe enum to hours."""
    timeframe_mapping = {
        TrendTimeframe.HOUR: 1,
        TrendTimeframe.SIX_HOURS: 6,
        TrendTimeframe.DAY: 24,
        TrendTimeframe.WEEK: 168,
        TrendTimeframe.MONTH: 720
    }
    return timeframe_mapping[timeframe]


async def _get_trending_data(
    start_time: datetime,
    end_time: datetime,
    category: Optional[TrendCategory],
    trend_type: Optional[TrendType],
    min_confidence: float,
    limit: int
) -> List[TrendingItem]:
    """Get trending data from background processing cache."""
    # Mock trending data (would come from background processing system)
    mock_trends = [
        TrendingItem(
            text="Artificial Intelligence",
            trend_type=TrendType.TOPIC,
            category=TrendCategory.TECHNOLOGY,
            score=0.85,
            velocity=45.2,
            volume=1250,
            sentiment=0.3,
            confidence=0.92,
            first_seen=start_time + timedelta(hours=2),
            peak_time=end_time - timedelta(hours=1),
            related_entities=["OpenAI", "Machine Learning", "ChatGPT"],
            sample_articles=["art_123", "art_456", "art_789"],
            geographic_distribution={"US": 450, "EU": 320, "Asia": 280, "Other": 200}
        ),
        TrendingItem(
            text="Climate Change",
            trend_type=TrendType.TOPIC,
            category=TrendCategory.SCIENCE,
            score=0.72,
            velocity=32.1,
            volume=890,
            sentiment=-0.2,
            confidence=0.88,
            first_seen=start_time + timedelta(hours=1),
            peak_time=end_time - timedelta(hours=3),
            related_entities=["Global Warming", "Carbon Emissions", "Renewable Energy"],
            sample_articles=["art_234", "art_567", "art_890"],
            geographic_distribution={"EU": 380, "US": 290, "Asia": 150, "Other": 70}
        ),
        TrendingItem(
            text="Tesla",
            trend_type=TrendType.ENTITY,
            category=TrendCategory.BUSINESS,
            score=0.68,
            velocity=28.7,
            volume=720,
            sentiment=0.1,
            confidence=0.85,
            first_seen=start_time + timedelta(hours=3),
            peak_time=end_time - timedelta(hours=2),
            related_entities=["Elon Musk", "Electric Vehicles", "Stock Market"],
            sample_articles=["art_345", "art_678", "art_901"],
            geographic_distribution={"US": 400, "EU": 180, "Asia": 100, "Other": 40}
        )
    ]
    
    # Apply filters
    filtered_trends = mock_trends
    
    if category:
        filtered_trends = [t for t in filtered_trends if t.category == category]
    
    if trend_type:
        filtered_trends = [t for t in filtered_trends if t.trend_type == trend_type]
    
    if min_confidence:
        filtered_trends = [t for t in filtered_trends if t.confidence >= min_confidence]
    
    # Sort by score and limit
    filtered_trends.sort(key=lambda x: x.score, reverse=True)
    return filtered_trends[:limit]


async def _enhance_with_sentiment(
    trending_data: List[TrendingItem],
    nlp_pipeline: NLPPipeline,
    bias_engine: BiasDetectionEngine
) -> List[TrendingItem]:
    """Enhance trending data with detailed sentiment analysis."""
    # In a real implementation, this would analyze actual content
    # For now, the mock data already includes sentiment
    return trending_data


def _calculate_analysis_stats(trending_data: List[TrendingItem], timeframe_hours: int) -> Dict[str, Any]:
    """Calculate comprehensive analysis statistics."""
    if not trending_data:
        return {}
    
    total_volume = sum(item.volume for item in trending_data)
    avg_velocity = sum(item.velocity for item in trending_data) / len(trending_data)
    avg_sentiment = sum(item.sentiment for item in trending_data) / len(trending_data)
    
    # Category distribution
    category_dist = Counter(item.category.value for item in trending_data)
    
    # Type distribution
    type_dist = Counter(item.trend_type.value for item in trending_data)
    
    return {
        "total_volume": total_volume,
        "average_velocity": avg_velocity,
        "average_sentiment": avg_sentiment,
        "timeframe_hours": timeframe_hours,
        "category_distribution": dict(category_dist),
        "type_distribution": dict(type_dist),
        "confidence_range": {
            "min": min(item.confidence for item in trending_data),
            "max": max(item.confidence for item in trending_data),
            "avg": sum(item.confidence for item in trending_data) / len(trending_data)
        }
    }


async def _calculate_velocity_data(
    item: str,
    start_time: datetime,
    end_time: datetime,
    granularity: str
) -> Dict[str, Any]:
    """Calculate velocity data for a specific item."""
    # Mock velocity data (would come from time-series analysis)
    hours = int((end_time - start_time).total_seconds() / 3600)
    
    # Generate mock velocity history
    history = []
    peak_velocity = 0
    current_velocity = 0
    
    for i in range(hours):
        timestamp = start_time + timedelta(hours=i)
        # Simulate velocity with some randomness and trends
        base_velocity = 20 + 10 * math.sin(i * 0.5) + (i * 0.5)
        velocity = max(0, base_velocity + (i % 3 - 1) * 5)
        
        history.append({
            "timestamp": timestamp.isoformat(),
            "velocity": velocity,
            "mentions": int(velocity * 0.8)
        })
        
        peak_velocity = max(peak_velocity, velocity)
        if i == hours - 1:  # Current velocity is the last data point
            current_velocity = velocity
    
    return {
        "current_velocity": current_velocity,
        "peak_velocity": peak_velocity,
        "history": history
    }


def _calculate_momentum_analysis(velocity_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate momentum and acceleration from velocity data."""
    history = velocity_data["history"]
    
    if len(history) < 2:
        return {"acceleration": 0.0, "momentum_score": 0.0}
    
    # Calculate acceleration (change in velocity)
    recent_velocities = [point["velocity"] for point in history[-3:]]
    if len(recent_velocities) >= 2:
        acceleration = recent_velocities[-1] - recent_velocities[0]
    else:
        acceleration = 0.0
    
    # Calculate momentum score (0-1)
    current_velocity = velocity_data["current_velocity"]
    peak_velocity = velocity_data["peak_velocity"]
    
    if peak_velocity > 0:
        momentum_score = min(1.0, current_velocity / peak_velocity)
    else:
        momentum_score = 0.0
    
    # Adjust momentum based on acceleration
    if acceleration > 0:
        momentum_score = min(1.0, momentum_score * 1.2)
    elif acceleration < 0:
        momentum_score = momentum_score * 0.8
    
    return {
        "acceleration": acceleration,
        "momentum_score": momentum_score
    }


def _generate_velocity_prediction(velocity_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate velocity prediction based on historical data."""
    history = velocity_data["history"]
    
    if len(history) < 3:
        return {"prediction": "insufficient_data", "confidence": 0.0}
    
    # Simple trend analysis
    recent_velocities = [point["velocity"] for point in history[-5:]]
    trend = (recent_velocities[-1] - recent_velocities[0]) / len(recent_velocities)
    
    # Predict next hour velocity
    predicted_velocity = max(0, recent_velocities[-1] + trend)
    
    # Calculate prediction confidence based on trend consistency
    velocity_changes = [recent_velocities[i+1] - recent_velocities[i] for i in range(len(recent_velocities)-1)]
    trend_consistency = 1.0 - (sum(abs(change - trend) for change in velocity_changes) / len(velocity_changes) / max(1, abs(trend)))
    confidence = max(0.0, min(1.0, trend_consistency))
    
    return {
        "predicted_velocity": predicted_velocity,
        "trend_direction": "increasing" if trend > 0 else "decreasing" if trend < 0 else "stable",
        "confidence": confidence,
        "prediction_horizon": "1_hour"
    }


async def _calculate_volume_data(
    start_time: datetime,
    end_time: datetime,
    category: Optional[TrendCategory]
) -> Dict[str, Any]:
    """Calculate volume data for the specified timeframe."""
    hours = int((end_time - start_time).total_seconds() / 3600)
    
    # Mock volume data
    total_volume = 0
    hourly_data = []
    by_category = defaultdict(int)
    
    for i in range(hours):
        timestamp = start_time + timedelta(hours=i)
        # Simulate hourly volume with daily patterns
        hour_of_day = timestamp.hour
        base_volume = 100 + 50 * math.sin((hour_of_day - 6) * math.pi / 12)  # Peak around noon
        volume = max(20, int(base_volume + (i % 5 - 2) * 10))
        
        hourly_data.append({
            "timestamp": timestamp.isoformat(),
            "hour": hour_of_day,
            "volume": volume
        })
        
        total_volume += volume
        
        # Distribute volume across categories
        for cat in TrendCategory:
            if category is None or cat == category:
                cat_volume = int(volume * (0.1 + 0.1 * (hash(cat.value) % 5)))
                by_category[cat.value] += cat_volume
    
    return {
        "total_volume": total_volume,
        "hourly_data": hourly_data,
        "by_category": dict(by_category),
        "historical_data": hourly_data  # For trend calculation
    }


def _analyze_hourly_patterns(hourly_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze hourly activity patterns."""
    if not hourly_data:
        return {"peak_hours": []}
    
    # Group by hour of day
    hour_volumes = defaultdict(list)
    for data_point in hourly_data:
        hour = data_point["hour"]
        volume = data_point["volume"]
        hour_volumes[hour].append(volume)
    
    # Calculate average volume per hour
    hour_averages = {}
    for hour, volumes in hour_volumes.items():
        hour_averages[hour] = sum(volumes) / len(volumes)
    
    # Find peak hours (top 25%)
    sorted_hours = sorted(hour_averages.items(), key=lambda x: x[1], reverse=True)
    peak_count = max(1, len(sorted_hours) // 4)
    peak_hours = [hour for hour, _ in sorted_hours[:peak_count]]
    
    return {
        "peak_hours": sorted(peak_hours),
        "hour_averages": hour_averages
    }


def _calculate_volume_trends(historical_data: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate volume trend indicators."""
    if len(historical_data) < 2:
        return {}
    
    volumes = [point["volume"] for point in historical_data]
    
    # Calculate overall trend
    first_half = volumes[:len(volumes)//2]
    second_half = volumes[len(volumes)//2:]
    
    first_avg = sum(first_half) / len(first_half)
    second_avg = sum(second_half) / len(second_half)
    
    overall_trend = (second_avg - first_avg) / first_avg if first_avg > 0 else 0
    
    # Calculate recent trend (last 25% vs previous 25%)
    quarter_size = max(1, len(volumes) // 4)
    recent_volumes = volumes[-quarter_size:]
    previous_volumes = volumes[-2*quarter_size:-quarter_size] if len(volumes) >= 2*quarter_size else volumes[:-quarter_size]
    
    if previous_volumes:
        recent_avg = sum(recent_volumes) / len(recent_volumes)
        previous_avg = sum(previous_volumes) / len(previous_volumes)
        recent_trend = (recent_avg - previous_avg) / previous_avg if previous_avg > 0 else 0
    else:
        recent_trend = 0
    
    return {
        "overall_trend": overall_trend,
        "recent_trend": recent_trend,
        "volatility": _calculate_volatility(volumes)
    }


def _calculate_volatility(values: List[float]) -> float:
    """Calculate volatility of values."""
    if len(values) < 2:
        return 0.0
    
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance) / mean if mean > 0 else 0.0


async def _detect_emerging_trends(
    previous_start: datetime,
    current_start: datetime,
    current_end: datetime,
    growth_threshold: float,
    confidence_threshold: float,
    min_volume: int,
    nlp_pipeline: NLPPipeline
) -> List[TrendingItem]:
    """Detect emerging trends by comparing periods."""
    # Mock emerging trend detection
    emerging_trends = [
        TrendingItem(
            text="Quantum Computing Breakthrough",
            trend_type=TrendType.TOPIC,
            category=TrendCategory.TECHNOLOGY,
            score=0.78,
            velocity=15.3,
            volume=45,
            sentiment=0.4,
            confidence=0.82,
            first_seen=current_start + timedelta(hours=1),
            peak_time=current_end - timedelta(minutes=30),
            related_entities=["IBM", "Google", "Quantum Supremacy"],
            sample_articles=["art_new_123", "art_new_456"],
            geographic_distribution={"US": 25, "EU": 15, "Asia": 5}
        )
    ]
    
    # Filter by confidence and volume thresholds
    filtered_trends = [
        trend for trend in emerging_trends
        if trend.confidence >= confidence_threshold and trend.volume >= min_volume
    ]
    
    return filtered_trends


async def _perform_custom_analysis(
    items: List[str],
    timeframe: str,
    analysis_type: str,
    analysis_request: Dict[str, Any],
    nlp_pipeline: NLPPipeline
) -> Dict[str, Any]:
    """Perform custom trend analysis."""
    results = {}
    
    for item in items:
        # Mock custom analysis results
        results[item] = {
            "trend_score": 0.65,
            "velocity": 22.1,
            "volume": 340,
            "sentiment": 0.1,
            "related_items": [f"related_to_{item}_1", f"related_to_{item}_2"],
            "peak_times": ["2025-01-15T14:00:00Z", "2025-01-15T18:00:00Z"],
            "geographic_spread": {"primary": "US", "secondary": ["EU", "Asia"]}
        }
    
    # Add comparative analysis if requested
    if analysis_type == "comparative" and len(items) > 1:
        results["comparison"] = {
            "strongest_trend": max(items, key=lambda x: results[x]["trend_score"]),
            "highest_velocity": max(items, key=lambda x: results[x]["velocity"]),
            "most_positive_sentiment": max(items, key=lambda x: results[x]["sentiment"]),
            "correlations": _calculate_item_correlations(items, results)
        }
    
    return results


def _calculate_item_correlations(items: List[str], results: Dict[str, Any]) -> Dict[str, float]:
    """Calculate correlations between analyzed items."""
    correlations = {}
    
    for i, item1 in enumerate(items):
        for item2 in items[i+1:]:
            # Simple correlation based on trend scores and sentiment
            score1 = results[item1]["trend_score"]
            score2 = results[item2]["trend_score"]
            sent1 = results[item1]["sentiment"]
            sent2 = results[item2]["sentiment"]
            
            # Simple correlation calculation
            correlation = (score1 * score2 + sent1 * sent2) / 2
            correlations[f"{item1}_vs_{item2}"] = min(1.0, max(-1.0, correlation))
    
    return correlations


async def _schedule_detailed_analysis(
    items: List[str],
    analysis_request: Dict[str, Any],
    user_id: str
):
    """Schedule detailed background analysis."""
    # In a real implementation, this would queue a background job
    logger.info(f"Scheduled detailed analysis for {len(items)} items for user {user_id}")
    
    # Simulate background processing
    await asyncio.sleep(0.1)
    logger.info(f"Background analysis completed for user {user_id}")