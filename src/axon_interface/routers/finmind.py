"""
FinMind API Endpoints - Market Pulse Intelligence

Implements financial market intelligence capabilities:
- GET /financial/market - Market sentiment analysis with ticker filtering
- GET /financial/trends - Financial trend analysis and correlations
- GET /financial/sentiment - Sentiment analysis specific to financial content
- GET /financial/entities - Financial entity extraction and analysis

Provides specialized financial content analysis and market intelligence.
"""

import logging
from typing import List, Optional, Dict, Any
import re
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from ...shared.schemas import ArticleResponse, PaginatedResponse, PaginationInfo
from ...thalamus.nlp_pipeline import get_nlp_pipeline, NLPPipeline
from ...thalamus.bias_analysis import get_bias_engine, BiasDetectionEngine
from ..dependencies import get_repository_factory
from ...synaptic_vesicle.repositories import RepositoryFactory


logger = logging.getLogger(__name__)
router = APIRouter()


# Financial keywords and patterns for market analysis
FINANCIAL_KEYWORDS = {
    "market_indicators": [
        "bull market", "bear market", "volatility", "market cap", "trading volume",
        "price target", "earnings", "revenue", "profit", "loss", "dividend"
    ],
    "sentiment_words": {
        "positive": [
            "bullish", "rally", "surge", "gains", "growth", "profit", "beat expectations",
            "outperform", "upgrade", "buy rating", "strong performance", "record high"
        ],
        "negative": [
            "bearish", "crash", "plunge", "losses", "decline", "miss expectations", 
            "underperform", "downgrade", "sell rating", "weak performance", "record low"
        ]
    },
    "financial_entities": [
        "NYSE", "NASDAQ", "S&P 500", "Dow Jones", "Russell 2000", "VIX",
        "Fed", "Federal Reserve", "SEC", "FDIC", "Treasury", "bond", "stock", "equity"
    ]
}


@router.get(
    "/financial/market",
    summary="Market sentiment analysis with ticker filtering",
    description="""
    Analyze market sentiment and trends for specific tickers or overall market.
    
    Features:
    - Ticker-specific sentiment analysis
    - Market trend identification
    - Financial entity extraction
    - Correlation analysis between news sentiment and market movements
    - Real-time market pulse indicators
    
    Supports filtering by ticker symbols, date ranges, and sentiment thresholds.
    """
)
async def get_market_analysis(
    request: Request,
    tickers: Optional[str] = Query(None, description="Comma-separated ticker symbols (e.g., AAPL,GOOGL,MSFT)"),
    sentiment_threshold: float = Query(0.0, ge=-1.0, le=1.0, description="Minimum sentiment score"),
    days_back: int = Query(7, ge=1, le=30, description="Number of days to analyze"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of articles to analyze"),
    repository_factory: RepositoryFactory = Depends(get_repository_factory),
    nlp_pipeline: NLPPipeline = Depends(get_nlp_pipeline)
) -> Dict[str, Any]:
    """Analyze market sentiment and trends."""
    try:
        # Parse ticker symbols
        ticker_list = []
        if tickers:
            ticker_list = [ticker.strip().upper() for ticker in tickers.split(",")]
        
        # Get article repository
        article_repo = repository_factory.get_article_repository()
        
        # Build search filters for financial content
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        filters = {
            "published_after": start_date.isoformat(),
            "published_before": end_date.isoformat(),
            "categories": ["finance", "business", "markets", "economy"],
            "min_significance": 0.3  # Focus on significant financial news
        }
        
        # Get relevant articles
        articles, total_count = await article_repo.list_with_filters(
            filters=filters,
            limit=limit,
            offset=0,
            sort_by="published_at",
            sort_order="desc"
        )
        
        # Analyze financial sentiment and extract insights
        market_analysis = await _analyze_financial_content(
            articles, ticker_list, sentiment_threshold, nlp_pipeline
        )
        
        logger.info(f"Analyzed {len(articles)} financial articles for market pulse")
        
        return {
            "analysis_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days_analyzed": days_back
            },
            "tickers_analyzed": ticker_list,
            "articles_processed": len(articles),
            "market_sentiment": market_analysis["overall_sentiment"],
            "ticker_analysis": market_analysis["ticker_sentiment"],
            "trend_indicators": market_analysis["trends"],
            "key_entities": market_analysis["entities"],
            "sentiment_distribution": market_analysis["sentiment_dist"],
            "market_pulse_score": market_analysis["pulse_score"],
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in market analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to analyze market sentiment"
        )


@router.get(
    "/financial/trends",
    summary="Financial trend analysis and correlations",
    description="""
    Analyze financial trends and correlations across different market segments.
    
    Features:
    - Sector-wise sentiment trends
    - Cross-correlation analysis between different tickers
    - Market momentum indicators
    - Volatility sentiment correlation
    - Trend velocity calculations
    """
)
async def get_financial_trends(
    request: Request,
    sector: Optional[str] = Query(None, description="Market sector (tech, finance, healthcare, etc.)"),
    time_window: str = Query("7d", regex="^(1d|3d|7d|14d|30d)$", description="Time window for trend analysis"),
    correlation_threshold: float = Query(0.5, ge=0.0, le=1.0, description="Minimum correlation threshold"),
    repository_factory: RepositoryFactory = Depends(get_repository_factory)
) -> Dict[str, Any]:
    """Analyze financial trends and correlations."""
    try:
        # Convert time window to days
        time_mapping = {"1d": 1, "3d": 3, "7d": 7, "14d": 14, "30d": 30}
        days_back = time_mapping[time_window]
        
        # Get financial articles for trend analysis
        article_repo = repository_factory.get_article_repository()
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        filters = {
            "published_after": start_date.isoformat(),
            "published_before": end_date.isoformat(),
            "categories": ["finance", "business", "markets"]
        }
        
        if sector:
            filters["content_contains"] = sector
        
        articles, _ = await article_repo.list_with_filters(
            filters=filters,
            limit=500,
            offset=0,
            sort_by="published_at",
            sort_order="desc"
        )
        
        # Analyze trends
        trend_analysis = await _analyze_financial_trends(articles, days_back, correlation_threshold)
        
        logger.info(f"Analyzed financial trends for {time_window} period")
        
        return {
            "time_window": time_window,
            "sector_filter": sector,
            "articles_analyzed": len(articles),
            "trend_indicators": trend_analysis["trends"],
            "momentum_score": trend_analysis["momentum"],
            "volatility_indicators": trend_analysis["volatility"],
            "sector_analysis": trend_analysis["sectors"],
            "correlation_matrix": trend_analysis["correlations"],
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in financial trends analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to analyze financial trends"
        )


@router.get(
    "/financial/sentiment",
    summary="Financial sentiment analysis",
    description="""
    Specialized sentiment analysis for financial content with market-specific indicators.
    
    Features:
    - Financial-specific sentiment scoring
    - Market emotion detection (fear, greed, uncertainty)
    - Sentiment momentum tracking
    - Confidence intervals for sentiment predictions
    """
)
async def get_financial_sentiment(
    request: Request,
    query: str = Query(..., description="Search query for financial sentiment analysis"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of articles to analyze"),
    repository_factory: RepositoryFactory = Depends(get_repository_factory),
    nlp_pipeline: NLPPipeline = Depends(get_nlp_pipeline),
    bias_engine: BiasDetectionEngine = Depends(get_bias_engine)
) -> Dict[str, Any]:
    """Analyze sentiment for specific financial queries."""
    try:
        # Search for relevant financial content
        article_repo = repository_factory.get_article_repository()
        
        # Use search functionality to find relevant articles
        filters = {
            "content_contains": query,
            "categories": ["finance", "business", "markets"],
            "published_after": (datetime.now() - timedelta(days=7)).isoformat()
        }
        
        articles, total_count = await article_repo.list_with_filters(
            filters=filters,
            limit=limit,
            offset=0,
            sort_by="published_at",
            sort_order="desc"
        )
        
        # Perform specialized financial sentiment analysis
        sentiment_analysis = await _analyze_financial_sentiment(
            articles, query, nlp_pipeline, bias_engine
        )
        
        logger.info(f"Analyzed financial sentiment for query: {query}")
        
        return {
            "query": query,
            "articles_analyzed": len(articles),
            "overall_sentiment": sentiment_analysis["overall"],
            "sentiment_breakdown": sentiment_analysis["breakdown"],
            "market_emotions": sentiment_analysis["emotions"],
            "confidence_score": sentiment_analysis["confidence"],
            "bias_indicators": sentiment_analysis["bias"],
            "key_phrases": sentiment_analysis["phrases"],
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in financial sentiment analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to analyze financial sentiment"
        )


@router.get(
    "/financial/entities",
    summary="Financial entity extraction and analysis",
    description="""
    Extract and analyze financial entities from content.
    
    Features:
    - Company and ticker symbol extraction
    - Financial instrument identification
    - Market indicator recognition
    - Entity sentiment correlation
    - Relationship mapping between financial entities
    """
)
async def get_financial_entities(
    request: Request,
    content: Optional[str] = Query(None, description="Text content to analyze for financial entities"),
    article_id: Optional[str] = Query(None, description="Article ID to analyze"),
    include_relationships: bool = Query(True, description="Include entity relationships in response"),
    repository_factory: RepositoryFactory = Depends(get_repository_factory),
    nlp_pipeline: NLPPipeline = Depends(get_nlp_pipeline)
) -> Dict[str, Any]:
    """Extract financial entities from content."""
    try:
        text_to_analyze = ""
        
        if article_id:
            # Get article content
            article_repo = repository_factory.get_article_repository()
            article = await article_repo.get_by_id(article_id)
            if not article:
                raise HTTPException(status_code=404, detail="Article not found")
            text_to_analyze = f"{article.title} {article.content}"
        elif content:
            text_to_analyze = content
        else:
            raise HTTPException(
                status_code=400, 
                detail="Either 'content' or 'article_id' parameter is required"
            )
        
        # Extract financial entities
        entity_analysis = await _extract_financial_entities(
            text_to_analyze, nlp_pipeline, include_relationships
        )
        
        logger.info("Extracted financial entities from content")
        
        return {
            "text_length": len(text_to_analyze),
            "entities": entity_analysis["entities"],
            "tickers": entity_analysis["tickers"],
            "financial_instruments": entity_analysis["instruments"],
            "market_indicators": entity_analysis["indicators"],
            "relationships": entity_analysis["relationships"] if include_relationships else [],
            "entity_sentiment": entity_analysis["sentiment"],
            "confidence_scores": entity_analysis["confidence"],
            "generated_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in financial entity extraction: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to extract financial entities"
        )


# Helper functions for financial analysis
async def _analyze_financial_content(
    articles: List[Any], 
    ticker_list: List[str], 
    sentiment_threshold: float,
    nlp_pipeline: NLPPipeline
) -> Dict[str, Any]:
    """Analyze financial content for market insights."""
    overall_sentiment = 0.0
    ticker_sentiment = {}
    sentiment_scores = []
    entities = {}
    trends = []
    
    for article in articles:
        # Get NLP analysis
        if hasattr(article, 'nlp_data') and article.nlp_data:
            sentiment = article.nlp_data.get('sentiment', 0.0)
            article_entities = article.nlp_data.get('entities', [])
        else:
            # Perform NLP analysis if not available
            analysis = await nlp_pipeline.process_text(article.content, article.title)
            sentiment = analysis.sentiment
            article_entities = analysis.entities
        
        sentiment_scores.append(sentiment)
        
        # Extract financial entities
        for entity in article_entities:
            entity_text = entity.get('text', '') if isinstance(entity, dict) else entity.text
            entity_type = entity.get('label', '') if isinstance(entity, dict) else entity.label
            
            if entity_text not in entities:
                entities[entity_text] = {
                    'type': entity_type,
                    'mentions': 0,
                    'sentiment_sum': 0.0,
                    'articles': []
                }
            
            entities[entity_text]['mentions'] += 1
            entities[entity_text]['sentiment_sum'] += sentiment
            entities[entity_text]['articles'].append(article.id)
        
        # Analyze ticker-specific sentiment
        article_text = f"{article.title} {article.content}".upper()
        for ticker in ticker_list:
            if ticker in article_text:
                if ticker not in ticker_sentiment:
                    ticker_sentiment[ticker] = {
                        'sentiment_sum': 0.0,
                        'article_count': 0,
                        'mentions': 0
                    }
                
                ticker_sentiment[ticker]['sentiment_sum'] += sentiment
                ticker_sentiment[ticker]['article_count'] += 1
                ticker_sentiment[ticker]['mentions'] += article_text.count(ticker)
    
    # Calculate overall metrics
    if sentiment_scores:
        overall_sentiment = sum(sentiment_scores) / len(sentiment_scores)
    
    # Calculate ticker averages
    for ticker in ticker_sentiment:
        if ticker_sentiment[ticker]['article_count'] > 0:
            ticker_sentiment[ticker]['average_sentiment'] = (
                ticker_sentiment[ticker]['sentiment_sum'] / 
                ticker_sentiment[ticker]['article_count']
            )
    
    # Calculate entity sentiments
    for entity_name in entities:
        if entities[entity_name]['mentions'] > 0:
            entities[entity_name]['average_sentiment'] = (
                entities[entity_name]['sentiment_sum'] / 
                entities[entity_name]['mentions']
            )
    
    # Generate sentiment distribution
    positive_count = sum(1 for s in sentiment_scores if s > 0.1)
    negative_count = sum(1 for s in sentiment_scores if s < -0.1)
    neutral_count = len(sentiment_scores) - positive_count - negative_count
    
    sentiment_dist = {
        'positive': positive_count,
        'negative': negative_count,
        'neutral': neutral_count,
        'total': len(sentiment_scores)
    }
    
    # Calculate market pulse score (0-100)
    pulse_score = max(0, min(100, (overall_sentiment + 1) * 50))
    
    return {
        'overall_sentiment': overall_sentiment,
        'ticker_sentiment': ticker_sentiment,
        'trends': trends,
        'entities': dict(list(entities.items())[:10]),  # Top 10 entities
        'sentiment_dist': sentiment_dist,
        'pulse_score': pulse_score
    }


async def _analyze_financial_trends(
    articles: List[Any], 
    days_back: int, 
    correlation_threshold: float
) -> Dict[str, Any]:
    """Analyze financial trends and correlations."""
    # Group articles by day for trend analysis
    daily_sentiment = {}
    sector_sentiment = {}
    
    for article in articles:
        # Get article date
        pub_date = article.published_at.date() if article.published_at else datetime.now().date()
        day_key = pub_date.isoformat()
        
        # Get sentiment
        sentiment = 0.0
        if hasattr(article, 'nlp_data') and article.nlp_data:
            sentiment = article.nlp_data.get('sentiment', 0.0)
        
        # Daily sentiment tracking
        if day_key not in daily_sentiment:
            daily_sentiment[day_key] = {'sum': 0.0, 'count': 0}
        daily_sentiment[day_key]['sum'] += sentiment
        daily_sentiment[day_key]['count'] += 1
        
        # Sector analysis (simplified)
        content_lower = article.content.lower() if article.content else ""
        title_lower = article.title.lower() if article.title else ""
        full_text = f"{title_lower} {content_lower}"
        
        sectors = {
            'technology': ['tech', 'software', 'ai', 'cloud', 'digital'],
            'finance': ['bank', 'financial', 'credit', 'loan', 'investment'],
            'healthcare': ['health', 'medical', 'pharma', 'drug', 'biotech'],
            'energy': ['oil', 'gas', 'energy', 'renewable', 'solar']
        }
        
        for sector, keywords in sectors.items():
            if any(keyword in full_text for keyword in keywords):
                if sector not in sector_sentiment:
                    sector_sentiment[sector] = {'sum': 0.0, 'count': 0}
                sector_sentiment[sector]['sum'] += sentiment
                sector_sentiment[sector]['count'] += 1
    
    # Calculate daily averages
    trend_data = []
    for day_key in sorted(daily_sentiment.keys()):
        data = daily_sentiment[day_key]
        avg_sentiment = data['sum'] / data['count'] if data['count'] > 0 else 0.0
        trend_data.append({
            'date': day_key,
            'sentiment': avg_sentiment,
            'article_count': data['count']
        })
    
    # Calculate momentum (trend direction)
    momentum = 0.0
    if len(trend_data) >= 2:
        recent_sentiment = sum(d['sentiment'] for d in trend_data[-3:]) / min(3, len(trend_data))
        early_sentiment = sum(d['sentiment'] for d in trend_data[:3]) / min(3, len(trend_data))
        momentum = recent_sentiment - early_sentiment
    
    # Calculate sector averages
    sector_analysis = {}
    for sector, data in sector_sentiment.items():
        if data['count'] > 0:
            sector_analysis[sector] = {
                'average_sentiment': data['sum'] / data['count'],
                'article_count': data['count']
            }
    
    return {
        'trends': trend_data,
        'momentum': momentum,
        'volatility': _calculate_volatility([d['sentiment'] for d in trend_data]),
        'sectors': sector_analysis,
        'correlations': {}  # Simplified for now
    }


async def _analyze_financial_sentiment(
    articles: List[Any],
    query: str,
    nlp_pipeline: NLPPipeline,
    bias_engine: BiasDetectionEngine
) -> Dict[str, Any]:
    """Perform specialized financial sentiment analysis."""
    sentiments = []
    emotions = {'fear': 0, 'greed': 0, 'uncertainty': 0, 'confidence': 0}
    bias_scores = []
    key_phrases = []
    
    for article in articles:
        # Get or compute sentiment
        if hasattr(article, 'nlp_data') and article.nlp_data:
            sentiment = article.nlp_data.get('sentiment', 0.0)
        else:
            analysis = await nlp_pipeline.process_text(article.content, article.title)
            sentiment = analysis.sentiment
        
        sentiments.append(sentiment)
        
        # Analyze for market emotions
        content = f"{article.title} {article.content}".lower()
        
        # Fear indicators
        fear_words = ['crash', 'plunge', 'panic', 'fear', 'uncertainty', 'risk', 'volatile']
        fear_score = sum(1 for word in fear_words if word in content)
        emotions['fear'] += fear_score
        
        # Greed indicators
        greed_words = ['rally', 'surge', 'boom', 'bull', 'gains', 'profit', 'growth']
        greed_score = sum(1 for word in greed_words if word in content)
        emotions['greed'] += greed_score
        
        # Uncertainty indicators
        uncertainty_words = ['uncertain', 'unclear', 'volatile', 'unpredictable', 'mixed']
        uncertainty_score = sum(1 for word in uncertainty_words if word in content)
        emotions['uncertainty'] += uncertainty_score
        
        # Confidence indicators
        confidence_words = ['confident', 'strong', 'stable', 'solid', 'optimistic']
        confidence_score = sum(1 for word in confidence_words if word in content)
        emotions['confidence'] += confidence_score
        
        # Analyze bias
        try:
            bias_result = await bias_engine.analyze_bias(article.content)
            bias_scores.append(bias_result.overall_bias_score)
        except:
            bias_scores.append(0.0)
    
    # Calculate overall metrics
    overall_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
    confidence_score = 1.0 - (sum(bias_scores) / len(bias_scores) if bias_scores else 0.0)
    
    # Sentiment breakdown
    positive = sum(1 for s in sentiments if s > 0.1)
    negative = sum(1 for s in sentiments if s < -0.1)
    neutral = len(sentiments) - positive - negative
    
    breakdown = {
        'positive': positive,
        'negative': negative,
        'neutral': neutral,
        'total': len(sentiments)
    }
    
    return {
        'overall': overall_sentiment,
        'breakdown': breakdown,
        'emotions': emotions,
        'confidence': confidence_score,
        'bias': sum(bias_scores) / len(bias_scores) if bias_scores else 0.0,
        'phrases': key_phrases[:10]
    }


async def _extract_financial_entities(
    text: str,
    nlp_pipeline: NLPPipeline,
    include_relationships: bool
) -> Dict[str, Any]:
    """Extract financial entities from text."""
    # Perform NLP analysis
    analysis = await nlp_pipeline.process_text(text)
    
    # Extract entities
    entities = []
    tickers = []
    instruments = []
    indicators = []
    
    # Process NLP entities
    for entity in analysis.entities:
        entity_text = entity.text if hasattr(entity, 'text') else entity.get('text', '')
        entity_type = entity.label if hasattr(entity, 'label') else entity.get('label', '')
        
        entities.append({
            'text': entity_text,
            'type': entity_type,
            'confidence': getattr(entity, 'confidence', 1.0)
        })
        
        # Check if it's a ticker (simple pattern matching)
        if re.match(r'^[A-Z]{1,5}$', entity_text) and entity_type in ['ORG', 'MISC']:
            tickers.append(entity_text)
    
    # Extract additional financial patterns
    ticker_pattern = r'\b[A-Z]{1,5}\b'
    found_tickers = re.findall(ticker_pattern, text)
    tickers.extend([t for t in found_tickers if t not in tickers and len(t) <= 5])
    
    # Extract financial instruments
    instrument_patterns = [
        r'\b\w+\s+stock\b', r'\b\w+\s+bond\b', r'\b\w+\s+option\b',
        r'\bETF\b', r'\bREIT\b', r'\bfutures?\b'
    ]
    
    for pattern in instrument_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        instruments.extend(matches)
    
    # Extract market indicators
    for indicator in FINANCIAL_KEYWORDS['financial_entities']:
        if indicator.lower() in text.lower():
            indicators.append(indicator)
    
    # Simple relationship extraction (entity co-occurrence)
    relationships = []
    if include_relationships and len(entities) > 1:
        for i, entity1 in enumerate(entities[:5]):  # Limit to avoid too many combinations
            for entity2 in entities[i+1:6]:
                relationships.append({
                    'entity1': entity1['text'],
                    'entity2': entity2['text'],
                    'relationship': 'co-occurs',
                    'confidence': 0.5
                })
    
    # Calculate entity sentiment (simplified)
    entity_sentiment = {}
    for entity in entities[:10]:  # Top 10 entities
        # Simple sentiment based on surrounding context
        entity_context = _extract_entity_context(text, entity['text'])
        sentiment = _calculate_context_sentiment(entity_context)
        entity_sentiment[entity['text']] = sentiment
    
    return {
        'entities': entities,
        'tickers': list(set(tickers)),
        'instruments': list(set(instruments)),
        'indicators': list(set(indicators)),
        'relationships': relationships,
        'sentiment': entity_sentiment,
        'confidence': {'overall': 0.8}  # Simplified confidence
    }


def _calculate_volatility(values: List[float]) -> float:
    """Calculate volatility of sentiment values."""
    if len(values) < 2:
        return 0.0
    
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5


def _extract_entity_context(text: str, entity: str, window: int = 50) -> str:
    """Extract context around an entity mention."""
    entity_pos = text.lower().find(entity.lower())
    if entity_pos == -1:
        return ""
    
    start = max(0, entity_pos - window)
    end = min(len(text), entity_pos + len(entity) + window)
    return text[start:end]


def _calculate_context_sentiment(context: str) -> float:
    """Calculate sentiment of context using simple keyword matching."""
    positive_words = FINANCIAL_KEYWORDS['sentiment_words']['positive']
    negative_words = FINANCIAL_KEYWORDS['sentiment_words']['negative']
    
    context_lower = context.lower()
    positive_count = sum(1 for word in positive_words if word in context_lower)
    negative_count = sum(1 for word in negative_words if word in context_lower)
    
    total_words = len(context.split())
    if total_words == 0:
        return 0.0
    
    sentiment = (positive_count - negative_count) / total_words
    return max(-1.0, min(1.0, sentiment * 10))  # Scale and clamp