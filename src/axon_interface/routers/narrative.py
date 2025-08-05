"""
Narrative Analysis API Endpoints - Bias & Narrative Detection

Implements comprehensive bias and narrative analysis capabilities:
- POST /analysis/narrative - Comprehensive bias analysis and narrative extraction
- GET /analysis/bias - Bias indicator calculation and reporting
- POST /analysis/framing - Framing detection and analysis
- GET /analysis/sentiment-aggregation - Sentiment aggregation across content
- POST /analysis/batch - Batch narrative analysis processing

Provides advanced bias detection, narrative extraction, and framing analysis.
"""

import logging
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from datetime import datetime, timedelta
import asyncio
import statistics

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Body
from pydantic import BaseModel, Field

from ...shared.schemas import ArticleResponse
from ...thalamus.nlp_pipeline import get_nlp_pipeline, NLPPipeline
from ...thalamus.bias_analysis import get_bias_engine, BiasDetectionEngine
from ..dependencies import get_repository_factory
from ...synaptic_vesicle.repositories import RepositoryFactory


logger = logging.getLogger(__name__)
router = APIRouter()


class BiasType(str, Enum):
    """Types of bias that can be detected."""
    CONFIRMATION = "confirmation"
    SELECTION = "selection"
    FRAMING = "framing"
    LINGUISTIC = "linguistic"
    SOURCE = "source"
    TEMPORAL = "temporal"
    CULTURAL = "cultural"


class NarrativeTheme(str, Enum):
    """Common narrative themes."""
    CONFLICT = "conflict"
    PROGRESS = "progress"
    DECLINE = "decline"
    HEROIC = "heroic"
    VICTIM = "victim"
    CRISIS = "crisis"
    TRIUMPH = "triumph"
    CONSPIRACY = "conspiracy"


class FramingType(str, Enum):
    """Types of framing techniques."""
    EPISODIC = "episodic"
    THEMATIC = "thematic"
    STRATEGIC = "strategic"
    CONFLICT_ORIENTED = "conflict_oriented"
    HUMAN_INTEREST = "human_interest"
    ECONOMIC = "economic"
    MORAL = "moral"


class NarrativeAnalysisRequest(BaseModel):
    """Request model for narrative analysis."""
    text: Optional[str] = Field(None, description="Text content to analyze")
    article_id: Optional[str] = Field(None, description="Article ID to analyze")
    article_ids: Optional[List[str]] = Field(None, description="Multiple article IDs for comparative analysis")
    include_bias_indicators: bool = Field(True, description="Include detailed bias indicators")
    include_framing_analysis: bool = Field(True, description="Include framing detection")
    include_narrative_themes: bool = Field(True, description="Include narrative theme extraction")
    include_sentiment_aggregation: bool = Field(True, description="Include sentiment aggregation")
    include_linguistic_patterns: bool = Field(True, description="Include linguistic pattern analysis")
    confidence_threshold: float = Field(0.3, ge=0.0, le=1.0, description="Minimum confidence threshold")


class BiasIndicator(BaseModel):
    """Individual bias indicator."""
    type: BiasType = Field(..., description="Type of bias detected")
    description: str = Field(..., description="Description of the bias")
    evidence: List[str] = Field(..., description="Evidence supporting the bias detection")
    confidence: float = Field(..., description="Confidence score (0-1)")
    severity: str = Field(..., description="Severity level (low, medium, high)")
    location: Dict[str, Any] = Field(default_factory=dict, description="Location in text")


class FramingPattern(BaseModel):
    """Framing pattern detection."""
    type: FramingType = Field(..., description="Type of framing detected")
    description: str = Field(..., description="Description of the framing")
    keywords: List[str] = Field(..., description="Keywords associated with framing")
    confidence: float = Field(..., description="Confidence score (0-1)")
    examples: List[str] = Field(..., description="Example phrases")


class NarrativeThemeResult(BaseModel):
    """Narrative theme extraction result."""
    theme: NarrativeTheme = Field(..., description="Identified narrative theme")
    description: str = Field(..., description="Description of the theme")
    strength: float = Field(..., description="Strength of the theme (0-1)")
    supporting_elements: List[str] = Field(..., description="Elements supporting this theme")
    character_roles: Dict[str, str] = Field(default_factory=dict, description="Character roles in narrative")


class LinguisticPattern(BaseModel):
    """Linguistic pattern analysis."""
    pattern_type: str = Field(..., description="Type of linguistic pattern")
    description: str = Field(..., description="Description of the pattern")
    frequency: int = Field(..., description="Frequency of occurrence")
    examples: List[str] = Field(..., description="Example instances")
    bias_potential: float = Field(..., description="Potential for bias (0-1)")


class NarrativeAnalysisResponse(BaseModel):
    """Response model for narrative analysis."""
    overall_bias_score: float = Field(..., description="Overall bias score (0-1)")
    bias_indicators: List[BiasIndicator] = Field(..., description="Detected bias indicators")
    framing_patterns: List[FramingPattern] = Field(..., description="Detected framing patterns")
    narrative_themes: List[NarrativeThemeResult] = Field(..., description="Identified narrative themes")
    sentiment_aggregation: Dict[str, Any] = Field(..., description="Aggregated sentiment analysis")
    linguistic_patterns: List[LinguisticPattern] = Field(..., description="Linguistic patterns")
    source_bias_indicators: Dict[str, Any] = Field(..., description="Source-specific bias indicators")
    confidence: float = Field(..., description="Overall analysis confidence")
    processing_time: float = Field(..., description="Processing time in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# Bias detection patterns and indicators
BIAS_PATTERNS = {
    "confirmation": {
        "keywords": [
            "obviously", "clearly", "undoubtedly", "without question", "everyone knows",
            "it's clear that", "there's no doubt", "certainly", "definitely"
        ],
        "patterns": [
            r"(?:obviously|clearly|undoubtedly)\s+\w+",
            r"everyone\s+(?:knows|agrees|understands)",
            r"it's\s+(?:clear|obvious)\s+that",
            r"without\s+(?:question|doubt)"
        ]
    },
    "selection": {
        "keywords": [
            "cherry-picked", "selective", "biased sample", "unrepresentative",
            "handpicked", "carefully chosen", "selected examples"
        ],
        "patterns": [
            r"(?:cherry.picked|selective|handpicked)",
            r"(?:carefully|specifically)\s+(?:chosen|selected)",
            r"unrepresentative\s+sample"
        ]
    },
    "framing": {
        "keywords": [
            "spin", "angle", "perspective", "viewpoint", "slant", "interpretation",
            "portrayal", "characterization", "depiction"
        ],
        "patterns": [
            r"(?:spin|slant|angle)\s+on",
            r"from\s+(?:this|that|the)\s+perspective",
            r"(?:portrays|characterizes|depicts)\s+as"
        ]
    },
    "linguistic": {
        "keywords": [
            "allegedly", "supposedly", "reportedly", "claims", "purports",
            "so-called", "self-proclaimed", "would have us believe"
        ],
        "patterns": [
            r"(?:allegedly|supposedly|reportedly)",
            r"so.called\s+\w+",
            r"(?:claims|purports)\s+to",
            r"would\s+have\s+us\s+believe"
        ]
    }
}

# Framing detection patterns
FRAMING_PATTERNS = {
    "conflict_oriented": {
        "keywords": [
            "battle", "fight", "war", "clash", "confrontation", "dispute",
            "versus", "against", "opposition", "rivalry", "conflict"
        ],
        "patterns": [
            r"\b(?:battle|fight|war|clash)\b",
            r"\b(?:versus|vs\.?|against)\b",
            r"\b(?:confrontation|dispute|conflict)\b"
        ]
    },
    "crisis": {
        "keywords": [
            "crisis", "emergency", "urgent", "critical", "dire", "catastrophic",
            "disaster", "collapse", "breakdown", "failure"
        ],
        "patterns": [
            r"\b(?:crisis|emergency|urgent|critical)\b",
            r"\b(?:catastrophic|disaster|collapse)\b",
            r"\b(?:breakdown|failure)\b"
        ]
    },
    "progress": {
        "keywords": [
            "breakthrough", "advancement", "progress", "improvement", "innovation",
            "development", "growth", "success", "achievement"
        ],
        "patterns": [
            r"\b(?:breakthrough|advancement|progress)\b",
            r"\b(?:improvement|innovation|development)\b",
            r"\b(?:growth|success|achievement)\b"
        ]
    },
    "human_interest": {
        "keywords": [
            "personal", "individual", "family", "community", "local", "human",
            "emotional", "touching", "heartwarming", "inspiring"
        ],
        "patterns": [
            r"\b(?:personal|individual|family)\b",
            r"\b(?:community|local|human)\b",
            r"\b(?:emotional|touching|heartwarming)\b"
        ]
    }
}

# Narrative theme patterns
NARRATIVE_THEMES = {
    "heroic": {
        "keywords": [
            "hero", "champion", "leader", "pioneer", "visionary", "brave",
            "courageous", "determined", "overcame", "triumph"
        ],
        "character_roles": ["hero", "mentor", "ally"],
        "story_elements": ["challenge", "journey", "victory"]
    },
    "victim": {
        "keywords": [
            "victim", "suffered", "oppressed", "disadvantaged", "vulnerable",
            "exploited", "marginalized", "discriminated", "harmed"
        ],
        "character_roles": ["victim", "oppressor", "rescuer"],
        "story_elements": ["injustice", "suffering", "need for help"]
    },
    "conspiracy": {
        "keywords": [
            "conspiracy", "cover-up", "hidden", "secret", "behind closed doors",
            "agenda", "manipulation", "deception", "plot"
        ],
        "character_roles": ["conspirator", "whistleblower", "investigator"],
        "story_elements": ["secrecy", "revelation", "truth"]
    },
    "decline": {
        "keywords": [
            "decline", "deterioration", "worse", "failing", "crisis",
            "collapse", "downfall", "regression", "backward"
        ],
        "character_roles": ["victim of decline", "cause of decline", "observer"],
        "story_elements": ["past glory", "current problems", "uncertain future"]
    }
}


@router.post(
    "/analysis/narrative",
    response_model=NarrativeAnalysisResponse,
    summary="Comprehensive bias analysis and narrative extraction",
    description="""
    Perform comprehensive bias and narrative analysis on text content.
    
    Features:
    - Multi-dimensional bias detection (confirmation, selection, framing, linguistic)
    - Narrative theme extraction and character role identification
    - Framing pattern detection with confidence scoring
    - Sentiment aggregation across different dimensions
    - Linguistic pattern analysis for bias potential
    - Source bias indicators and credibility assessment
    - Comparative analysis across multiple articles
    
    Supports single text, article ID, or multiple articles for comparative analysis.
    """
)
async def analyze_narrative(
    request: Request,
    analysis_request: NarrativeAnalysisRequest,
    repository_factory: RepositoryFactory = Depends(get_repository_factory),
    nlp_pipeline: NLPPipeline = Depends(get_nlp_pipeline),
    bias_engine: BiasDetectionEngine = Depends(get_bias_engine)
) -> NarrativeAnalysisResponse:
    """Perform comprehensive narrative and bias analysis."""
    import time
    start_time = time.time()
    
    try:
        # Get content for analysis
        content_data = await _get_content_for_narrative_analysis(
            analysis_request, repository_factory
        )
        
        texts = content_data["texts"]
        metadata = content_data["metadata"]
        
        if not texts or all(not text.strip() for text in texts):
            raise HTTPException(
                status_code=400,
                detail="No valid content provided for analysis"
            )
        
        # Perform comprehensive analysis
        analysis_results = await _perform_comprehensive_narrative_analysis(
            texts=texts,
            metadata=metadata,
            request=analysis_request,
            nlp_pipeline=nlp_pipeline,
            bias_engine=bias_engine
        )
        
        processing_time = time.time() - start_time
        
        logger.info(f"Completed narrative analysis in {processing_time:.2f}s")
        
        return NarrativeAnalysisResponse(
            overall_bias_score=analysis_results["overall_bias_score"],
            bias_indicators=analysis_results["bias_indicators"],
            framing_patterns=analysis_results["framing_patterns"],
            narrative_themes=analysis_results["narrative_themes"],
            sentiment_aggregation=analysis_results["sentiment_aggregation"],
            linguistic_patterns=analysis_results["linguistic_patterns"],
            source_bias_indicators=analysis_results["source_bias_indicators"],
            confidence=analysis_results["confidence"],
            processing_time=processing_time,
            metadata=analysis_results["metadata"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in narrative analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to perform narrative analysis"
        )


@router.get(
    "/analysis/bias",
    summary="Bias indicator calculation and reporting",
    description="""
    Calculate and report bias indicators for content.
    
    Features:
    - Detailed bias type classification
    - Confidence scoring for each indicator
    - Evidence extraction and location tracking
    - Severity assessment (low, medium, high)
    - Bias pattern recognition and analysis
    """
)
async def analyze_bias(
    request: Request,
    text: Optional[str] = Query(None, description="Text content to analyze"),
    article_id: Optional[str] = Query(None, description="Article ID to analyze"),
    bias_types: Optional[str] = Query(None, description="Comma-separated bias types to focus on"),
    min_confidence: float = Query(0.3, ge=0.0, le=1.0, description="Minimum confidence threshold"),
    repository_factory: RepositoryFactory = Depends(get_repository_factory),
    bias_engine: BiasDetectionEngine = Depends(get_bias_engine)
) -> Dict[str, Any]:
    """Analyze bias indicators in content."""
    try:
        # Get content
        content = await _get_single_content_for_analysis(text, article_id, repository_factory)
        
        # Parse bias types filter
        focus_types = []
        if bias_types:
            focus_types = [t.strip() for t in bias_types.split(",")]
        
        # Perform bias analysis
        bias_analysis = await bias_engine.analyze_bias(content)
        
        # Extract detailed bias indicators
        bias_indicators = await _extract_detailed_bias_indicators(
            content, bias_analysis, focus_types, min_confidence
        )
        
        # Calculate bias distribution
        bias_distribution = _calculate_bias_distribution(bias_indicators)
        
        logger.info(f"Analyzed bias indicators: {len(bias_indicators)} found")
        
        return {
            "overall_bias_score": bias_analysis.overall_bias_score,
            "bias_indicators": bias_indicators,
            "bias_distribution": bias_distribution,
            "confidence": bias_analysis.confidence,
            "analysis_summary": {
                "total_indicators": len(bias_indicators),
                "high_confidence_indicators": len([b for b in bias_indicators if b["confidence"] > 0.7]),
                "bias_types_detected": len(set(b["type"] for b in bias_indicators))
            }
        }
        
    except Exception as e:
        logger.error(f"Error in bias analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to analyze bias indicators"
        )


@router.post(
    "/analysis/framing",
    summary="Framing detection and analysis",
    description="""
    Detect and analyze framing techniques in content.
    
    Features:
    - Multiple framing type detection
    - Keyword and pattern analysis
    - Confidence scoring and evidence extraction
    - Framing strength assessment
    - Comparative framing analysis
    """
)
async def analyze_framing(
    request: Request,
    framing_request: Dict[str, Any] = Body(...),
    repository_factory: RepositoryFactory = Depends(get_repository_factory)
) -> Dict[str, Any]:
    """Analyze framing techniques in content."""
    try:
        # Get content
        content = await _get_single_content_for_analysis(
            framing_request.get("text"),
            framing_request.get("article_id"),
            repository_factory
        )
        
        # Analyze framing patterns
        framing_analysis = await _analyze_framing_patterns(content)
        
        # Calculate framing strength
        framing_strength = _calculate_framing_strength(framing_analysis)
        
        logger.info(f"Analyzed framing patterns: {len(framing_analysis)} found")
        
        return {
            "framing_patterns": framing_analysis,
            "framing_strength": framing_strength,
            "dominant_framing": _identify_dominant_framing(framing_analysis),
            "framing_diversity": len(set(f["type"] for f in framing_analysis)),
            "analysis_metadata": {
                "content_length": len(content),
                "patterns_detected": len(framing_analysis),
                "confidence_average": statistics.mean([f["confidence"] for f in framing_analysis]) if framing_analysis else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error in framing analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to analyze framing patterns"
        )


@router.get(
    "/analysis/sentiment-aggregation",
    summary="Sentiment aggregation across content",
    description="""
    Aggregate sentiment analysis across multiple dimensions.
    
    Features:
    - Multi-dimensional sentiment analysis
    - Temporal sentiment trends
    - Entity-specific sentiment aggregation
    - Sentiment distribution analysis
    - Comparative sentiment metrics
    """
)
async def analyze_sentiment_aggregation(
    request: Request,
    article_ids: str = Query(..., description="Comma-separated article IDs"),
    time_window: Optional[str] = Query("7d", description="Time window for analysis"),
    group_by: Optional[str] = Query("day", description="Grouping for temporal analysis"),
    repository_factory: RepositoryFactory = Depends(get_repository_factory),
    nlp_pipeline: NLPPipeline = Depends(get_nlp_pipeline)
) -> Dict[str, Any]:
    """Aggregate sentiment analysis across multiple articles."""
    try:
        # Parse article IDs
        article_id_list = [aid.strip() for aid in article_ids.split(",")]
        
        if len(article_id_list) > 100:
            raise HTTPException(
                status_code=400,
                detail="Too many articles requested (max 100)"
            )
        
        # Get articles
        article_repo = repository_factory.get_article_repository()
        articles = []
        
        for article_id in article_id_list:
            article = await article_repo.get_by_id(article_id)
            if article:
                articles.append(article)
        
        if not articles:
            raise HTTPException(
                status_code=404,
                detail="No valid articles found"
            )
        
        # Perform sentiment aggregation
        sentiment_aggregation = await _perform_sentiment_aggregation(
            articles, time_window, group_by, nlp_pipeline
        )
        
        logger.info(f"Aggregated sentiment for {len(articles)} articles")
        
        return sentiment_aggregation
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in sentiment aggregation: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to aggregate sentiment analysis"
        )


@router.post(
    "/analysis/batch",
    summary="Batch narrative analysis processing",
    description="""
    Process multiple texts for narrative analysis in batch.
    
    Features:
    - Concurrent processing for efficiency
    - Comparative analysis across items
    - Aggregated insights and patterns
    - Batch statistics and reporting
    """
)
async def batch_narrative_analysis(
    request: Request,
    batch_request: Dict[str, Any] = Body(...),
    repository_factory: RepositoryFactory = Depends(get_repository_factory),
    nlp_pipeline: NLPPipeline = Depends(get_nlp_pipeline),
    bias_engine: BiasDetectionEngine = Depends(get_bias_engine)
) -> Dict[str, Any]:
    """Process batch narrative analysis requests."""
    try:
        user_tier = getattr(request.state, "user_tier", "free")
        
        # Validate batch size based on user tier
        max_batch_size = {"free": 5, "premium": 25, "enterprise": 100}.get(user_tier, 5)
        
        items = batch_request.get("items", [])
        if len(items) > max_batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size exceeds limit for {user_tier} tier: {max_batch_size}"
            )
        
        # Process items concurrently
        batch_results = await _process_batch_narrative_analysis(
            items, repository_factory, nlp_pipeline, bias_engine
        )
        
        # Aggregate insights
        aggregated_insights = _aggregate_narrative_insights(batch_results)
        
        # Calculate batch statistics
        successful = sum(1 for r in batch_results if r["success"])
        
        logger.info(f"Processed batch narrative analysis: {successful}/{len(items)} successful")
        
        return {
            "batch_summary": {
                "total_items": len(items),
                "successful": successful,
                "failed": len(items) - successful
            },
            "results": batch_results,
            "aggregated_insights": aggregated_insights,
            "comparative_analysis": _perform_comparative_analysis(batch_results)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch narrative analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to process batch narrative analysis"
        )


# Helper functions for narrative analysis
async def _get_content_for_narrative_analysis(
    request: NarrativeAnalysisRequest,
    repository_factory: RepositoryFactory
) -> Dict[str, Any]:
    """Get content from various sources for narrative analysis."""
    texts = []
    metadata = []
    
    if request.text:
        texts.append(request.text)
        metadata.append({"source": "direct_text", "length": len(request.text)})
    
    elif request.article_id:
        article_repo = repository_factory.get_article_repository()
        article = await article_repo.get_by_id(request.article_id)
        if not article:
            raise HTTPException(status_code=404, detail="Article not found")
        
        texts.append(article.content)
        metadata.append({
            "source": "article",
            "article_id": article.id,
            "title": article.title,
            "url": article.url,
            "published_at": article.published_at.isoformat() if article.published_at else None,
            "source_domain": article.source_domain
        })
    
    elif request.article_ids:
        article_repo = repository_factory.get_article_repository()
        
        for article_id in request.article_ids:
            article = await article_repo.get_by_id(article_id)
            if article:
                texts.append(article.content)
                metadata.append({
                    "source": "article",
                    "article_id": article.id,
                    "title": article.title,
                    "url": article.url,
                    "published_at": article.published_at.isoformat() if article.published_at else None,
                    "source_domain": article.source_domain
                })
    
    else:
        raise HTTPException(
            status_code=400,
            detail="Must provide either text, article_id, or article_ids"
        )
    
    return {"texts": texts, "metadata": metadata}


async def _get_single_content_for_analysis(
    text: Optional[str],
    article_id: Optional[str],
    repository_factory: RepositoryFactory
) -> str:
    """Get single content for analysis."""
    if text:
        return text
    
    elif article_id:
        article_repo = repository_factory.get_article_repository()
        article = await article_repo.get_by_id(article_id)
        if not article:
            raise HTTPException(status_code=404, detail="Article not found")
        return article.content
    
    else:
        raise HTTPException(
            status_code=400,
            detail="Must provide either text or article_id"
        )


async def _perform_comprehensive_narrative_analysis(
    texts: List[str],
    metadata: List[Dict[str, Any]],
    request: NarrativeAnalysisRequest,
    nlp_pipeline: NLPPipeline,
    bias_engine: BiasDetectionEngine
) -> Dict[str, Any]:
    """Perform comprehensive narrative analysis on texts."""
    # Combine texts for analysis if multiple
    combined_text = " ".join(texts)
    
    # Initialize results
    results = {
        "overall_bias_score": 0.0,
        "bias_indicators": [],
        "framing_patterns": [],
        "narrative_themes": [],
        "sentiment_aggregation": {},
        "linguistic_patterns": [],
        "source_bias_indicators": {},
        "confidence": 0.0,
        "metadata": {}
    }
    
    # Perform bias analysis
    if request.include_bias_indicators:
        bias_analysis = await bias_engine.analyze_bias(combined_text)
        results["overall_bias_score"] = bias_analysis.overall_bias_score
        results["bias_indicators"] = await _extract_detailed_bias_indicators(
            combined_text, bias_analysis, [], request.confidence_threshold
        )
    
    # Perform framing analysis
    if request.include_framing_analysis:
        results["framing_patterns"] = await _analyze_framing_patterns(combined_text)
    
    # Perform narrative theme extraction
    if request.include_narrative_themes:
        results["narrative_themes"] = await _extract_narrative_themes(combined_text)
    
    # Perform sentiment aggregation
    if request.include_sentiment_aggregation:
        results["sentiment_aggregation"] = await _perform_text_sentiment_aggregation(
            texts, nlp_pipeline
        )
    
    # Perform linguistic pattern analysis
    if request.include_linguistic_patterns:
        results["linguistic_patterns"] = await _analyze_linguistic_patterns(combined_text)
    
    # Analyze source bias indicators
    results["source_bias_indicators"] = _analyze_source_bias_indicators(metadata)
    
    # Calculate overall confidence
    confidences = []
    if results["bias_indicators"]:
        confidences.extend([b["confidence"] for b in results["bias_indicators"]])
    if results["framing_patterns"]:
        confidences.extend([f["confidence"] for f in results["framing_patterns"]])
    if results["narrative_themes"]:
        confidences.extend([n["strength"] for n in results["narrative_themes"]])
    
    results["confidence"] = statistics.mean(confidences) if confidences else 0.5
    
    # Add metadata
    results["metadata"] = {
        "texts_analyzed": len(texts),
        "total_length": sum(len(text) for text in texts),
        "analysis_components": {
            "bias_indicators": request.include_bias_indicators,
            "framing_analysis": request.include_framing_analysis,
            "narrative_themes": request.include_narrative_themes,
            "sentiment_aggregation": request.include_sentiment_aggregation,
            "linguistic_patterns": request.include_linguistic_patterns
        }
    }
    
    return results


async def _extract_detailed_bias_indicators(
    text: str,
    bias_analysis: Any,
    focus_types: List[str],
    min_confidence: float
) -> List[Dict[str, Any]]:
    """Extract detailed bias indicators from text."""
    indicators = []
    text_lower = text.lower()
    
    # Check each bias type
    for bias_type, patterns in BIAS_PATTERNS.items():
        if focus_types and bias_type not in focus_types:
            continue
        
        # Check keywords
        for keyword in patterns["keywords"]:
            if keyword.lower() in text_lower:
                confidence = 0.6 + (text_lower.count(keyword.lower()) * 0.1)
                confidence = min(1.0, confidence)
                
                if confidence >= min_confidence:
                    indicators.append({
                        "type": bias_type,
                        "description": f"Detected {bias_type} bias through keyword '{keyword}'",
                        "evidence": [keyword],
                        "confidence": confidence,
                        "severity": _calculate_bias_severity(confidence),
                        "location": {"keyword": keyword, "type": "keyword_match"}
                    })
        
        # Check patterns
        import re
        for pattern in patterns["patterns"]:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                confidence = 0.7
                if confidence >= min_confidence:
                    indicators.append({
                        "type": bias_type,
                        "description": f"Detected {bias_type} bias through pattern matching",
                        "evidence": [match.group(0)],
                        "confidence": confidence,
                        "severity": _calculate_bias_severity(confidence),
                        "location": {
                            "start": match.start(),
                            "end": match.end(),
                            "type": "pattern_match"
                        }
                    })
    
    return indicators


async def _analyze_framing_patterns(text: str) -> List[Dict[str, Any]]:
    """Analyze framing patterns in text."""
    framing_patterns = []
    text_lower = text.lower()
    
    for framing_type, patterns in FRAMING_PATTERNS.items():
        keywords_found = []
        pattern_matches = []
        
        # Check keywords
        for keyword in patterns["keywords"]:
            if keyword.lower() in text_lower:
                keywords_found.append(keyword)
        
        # Check patterns
        import re
        for pattern in patterns["patterns"]:
            matches = re.findall(pattern, text, re.IGNORECASE)
            pattern_matches.extend(matches)
        
        if keywords_found or pattern_matches:
            confidence = min(1.0, (len(keywords_found) + len(pattern_matches)) * 0.2)
            
            framing_patterns.append({
                "type": framing_type,
                "description": f"Detected {framing_type} framing pattern",
                "keywords": keywords_found,
                "confidence": confidence,
                "examples": (keywords_found + pattern_matches)[:5]  # Limit examples
            })
    
    return framing_patterns


async def _extract_narrative_themes(text: str) -> List[Dict[str, Any]]:
    """Extract narrative themes from text."""
    narrative_themes = []
    text_lower = text.lower()
    
    for theme_name, theme_data in NARRATIVE_THEMES.items():
        keywords_found = []
        
        # Check for theme keywords
        for keyword in theme_data["keywords"]:
            if keyword.lower() in text_lower:
                keywords_found.append(keyword)
        
        if keywords_found:
            strength = min(1.0, len(keywords_found) * 0.15)
            
            # Identify character roles
            character_roles = {}
            for role in theme_data["character_roles"]:
                if any(role_word in text_lower for role_word in [role, role + "s"]):
                    character_roles[role] = "present"
            
            narrative_themes.append({
                "theme": theme_name,
                "description": f"Detected {theme_name} narrative theme",
                "strength": strength,
                "supporting_elements": keywords_found,
                "character_roles": character_roles
            })
    
    return narrative_themes


async def _perform_text_sentiment_aggregation(
    texts: List[str],
    nlp_pipeline: NLPPipeline
) -> Dict[str, Any]:
    """Perform sentiment aggregation across texts."""
    sentiments = []
    
    for text in texts:
        try:
            analysis = await nlp_pipeline.process_text(text)
            sentiments.append(analysis.sentiment)
        except:
            sentiments.append(0.0)  # Neutral fallback
    
    if not sentiments:
        return {"error": "No sentiment data available"}
    
    return {
        "overall_sentiment": statistics.mean(sentiments),
        "sentiment_variance": statistics.variance(sentiments) if len(sentiments) > 1 else 0,
        "sentiment_range": {
            "min": min(sentiments),
            "max": max(sentiments)
        },
        "sentiment_distribution": {
            "positive": len([s for s in sentiments if s > 0.1]),
            "negative": len([s for s in sentiments if s < -0.1]),
            "neutral": len([s for s in sentiments if -0.1 <= s <= 0.1])
        },
        "individual_sentiments": sentiments
    }


async def _analyze_linguistic_patterns(text: str) -> List[Dict[str, Any]]:
    """Analyze linguistic patterns that may indicate bias."""
    patterns = []
    
    # Analyze sentence structure
    sentences = text.split('. ')
    avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
    
    if avg_sentence_length > 30:
        patterns.append({
            "pattern_type": "complex_sentences",
            "description": "Unusually complex sentence structure detected",
            "frequency": 1,
            "examples": [s for s in sentences if len(s.split()) > 30][:3],
            "bias_potential": 0.4
        })
    
    # Analyze passive voice usage
    import re
    passive_patterns = [
        r'\b(?:was|were|is|are|been|being)\s+\w+ed\b',
        r'\b(?:was|were|is|are|been|being)\s+\w+en\b'
    ]
    
    passive_matches = 0
    for pattern in passive_patterns:
        passive_matches += len(re.findall(pattern, text, re.IGNORECASE))
    
    if passive_matches > 5:
        patterns.append({
            "pattern_type": "excessive_passive_voice",
            "description": "High usage of passive voice detected",
            "frequency": passive_matches,
            "examples": re.findall(passive_patterns[0], text, re.IGNORECASE)[:3],
            "bias_potential": 0.6
        })
    
    # Analyze hedging language
    hedging_words = ["might", "could", "possibly", "perhaps", "maybe", "allegedly", "reportedly"]
    hedging_count = sum(text.lower().count(word) for word in hedging_words)
    
    if hedging_count > 3:
        patterns.append({
            "pattern_type": "hedging_language",
            "description": "Frequent use of hedging language detected",
            "frequency": hedging_count,
            "examples": [word for word in hedging_words if word in text.lower()][:3],
            "bias_potential": 0.5
        })
    
    return patterns


def _analyze_source_bias_indicators(metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze source-specific bias indicators."""
    source_indicators = {
        "source_diversity": len(set(m.get("source_domain", "") for m in metadata)),
        "temporal_distribution": {},
        "source_credibility": {}
    }
    
    # Analyze temporal distribution
    dates = [m.get("published_at") for m in metadata if m.get("published_at")]
    if dates:
        source_indicators["temporal_distribution"] = {
            "date_range": f"{min(dates)} to {max(dates)}",
            "temporal_clustering": len(set(d[:10] for d in dates))  # Group by date
        }
    
    # Analyze source domains
    domains = [m.get("source_domain") for m in metadata if m.get("source_domain")]
    if domains:
        domain_counts = {}
        for domain in domains:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        source_indicators["source_credibility"] = {
            "domain_distribution": domain_counts,
            "single_source_dominance": max(domain_counts.values()) / len(domains) if domains else 0
        }
    
    return source_indicators


def _calculate_bias_severity(confidence: float) -> str:
    """Calculate bias severity based on confidence."""
    if confidence >= 0.8:
        return "high"
    elif confidence >= 0.6:
        return "medium"
    else:
        return "low"


def _calculate_bias_distribution(bias_indicators: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate distribution of bias types."""
    if not bias_indicators:
        return {}
    
    type_counts = {}
    severity_counts = {"low": 0, "medium": 0, "high": 0}
    
    for indicator in bias_indicators:
        bias_type = indicator["type"]
        severity = indicator["severity"]
        
        type_counts[bias_type] = type_counts.get(bias_type, 0) + 1
        severity_counts[severity] += 1
    
    return {
        "by_type": type_counts,
        "by_severity": severity_counts,
        "total_indicators": len(bias_indicators)
    }


def _calculate_framing_strength(framing_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate overall framing strength."""
    if not framing_patterns:
        return {"overall_strength": 0.0, "dominant_type": None}
    
    # Calculate weighted strength
    total_strength = sum(p["confidence"] for p in framing_patterns)
    avg_strength = total_strength / len(framing_patterns)
    
    # Find dominant framing type
    type_strengths = {}
    for pattern in framing_patterns:
        frame_type = pattern["type"]
        if frame_type not in type_strengths:
            type_strengths[frame_type] = []
        type_strengths[frame_type].append(pattern["confidence"])
    
    dominant_type = max(type_strengths.keys(), 
                       key=lambda k: statistics.mean(type_strengths[k])) if type_strengths else None
    
    return {
        "overall_strength": avg_strength,
        "dominant_type": dominant_type,
        "type_distribution": {k: statistics.mean(v) for k, v in type_strengths.items()}
    }


def _identify_dominant_framing(framing_patterns: List[Dict[str, Any]]) -> Optional[str]:
    """Identify the dominant framing pattern."""
    if not framing_patterns:
        return None
    
    # Find the framing type with highest confidence
    return max(framing_patterns, key=lambda x: x["confidence"])["type"]


async def _perform_sentiment_aggregation(
    articles: List[Any],
    time_window: str,
    group_by: str,
    nlp_pipeline: NLPPipeline
) -> Dict[str, Any]:
    """Perform sentiment aggregation across articles."""
    # Group articles by time
    time_groups = {}
    
    for article in articles:
        if not article.published_at:
            continue
        
        # Group by specified time unit
        if group_by == "day":
            time_key = article.published_at.date().isoformat()
        elif group_by == "hour":
            time_key = article.published_at.strftime("%Y-%m-%d %H:00")
        else:
            time_key = "all"
        
        if time_key not in time_groups:
            time_groups[time_key] = []
        time_groups[time_key].append(article)
    
    # Calculate sentiment for each group
    aggregated_results = {}
    
    for time_key, group_articles in time_groups.items():
        sentiments = []
        
        for article in group_articles:
            # Get sentiment from NLP data or calculate
            if hasattr(article, 'nlp_data') and article.nlp_data:
                sentiment = article.nlp_data.get('sentiment', 0.0)
            else:
                try:
                    analysis = await nlp_pipeline.process_text(article.content, article.title)
                    sentiment = analysis.sentiment
                except:
                    sentiment = 0.0
            
            sentiments.append(sentiment)
        
        if sentiments:
            aggregated_results[time_key] = {
                "average_sentiment": statistics.mean(sentiments),
                "sentiment_variance": statistics.variance(sentiments) if len(sentiments) > 1 else 0,
                "article_count": len(sentiments),
                "sentiment_distribution": {
                    "positive": len([s for s in sentiments if s > 0.1]),
                    "negative": len([s for s in sentiments if s < -0.1]),
                    "neutral": len([s for s in sentiments if -0.1 <= s <= 0.1])
                }
            }
    
    # Calculate overall statistics
    all_sentiments = []
    for group_data in aggregated_results.values():
        all_sentiments.extend([group_data["average_sentiment"]] * group_data["article_count"])
    
    return {
        "temporal_aggregation": aggregated_results,
        "overall_statistics": {
            "total_articles": len(articles),
            "overall_sentiment": statistics.mean(all_sentiments) if all_sentiments else 0,
            "sentiment_trend": _calculate_sentiment_trend(aggregated_results),
            "time_window": time_window,
            "group_by": group_by
        }
    }


def _calculate_sentiment_trend(temporal_data: Dict[str, Any]) -> str:
    """Calculate sentiment trend direction."""
    if len(temporal_data) < 2:
        return "insufficient_data"
    
    # Sort by time and get sentiment values
    sorted_times = sorted(temporal_data.keys())
    sentiments = [temporal_data[time]["average_sentiment"] for time in sorted_times]
    
    # Simple trend calculation
    if len(sentiments) >= 3:
        recent_avg = statistics.mean(sentiments[-3:])
        early_avg = statistics.mean(sentiments[:3])
        
        if recent_avg > early_avg + 0.1:
            return "improving"
        elif recent_avg < early_avg - 0.1:
            return "declining"
        else:
            return "stable"
    
    return "stable"


async def _process_batch_narrative_analysis(
    items: List[Dict[str, Any]],
    repository_factory: RepositoryFactory,
    nlp_pipeline: NLPPipeline,
    bias_engine: BiasDetectionEngine
) -> List[Dict[str, Any]]:
    """Process batch narrative analysis requests."""
    results = []
    
    for i, item in enumerate(items):
        try:
            # Create analysis request from item
            analysis_request = NarrativeAnalysisRequest(**item)
            
            # Get content
            content_data = await _get_content_for_narrative_analysis(
                analysis_request, repository_factory
            )
            
            # Perform analysis
            analysis_results = await _perform_comprehensive_narrative_analysis(
                texts=content_data["texts"],
                metadata=content_data["metadata"],
                request=analysis_request,
                nlp_pipeline=nlp_pipeline,
                bias_engine=bias_engine
            )
            
            results.append({
                "item_index": i,
                "success": True,
                "overall_bias_score": analysis_results["overall_bias_score"],
                "bias_indicators_count": len(analysis_results["bias_indicators"]),
                "framing_patterns_count": len(analysis_results["framing_patterns"]),
                "narrative_themes_count": len(analysis_results["narrative_themes"]),
                "confidence": analysis_results["confidence"],
                "analysis_results": analysis_results
            })
            
        except Exception as e:
            results.append({
                "item_index": i,
                "success": False,
                "error": str(e)
            })
    
    return results


def _aggregate_narrative_insights(batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate insights from batch narrative analysis."""
    successful_results = [r for r in batch_results if r["success"]]
    
    if not successful_results:
        return {}
    
    # Aggregate bias scores
    bias_scores = [r["overall_bias_score"] for r in successful_results]
    avg_bias_score = statistics.mean(bias_scores)
    
    # Aggregate indicator counts
    total_bias_indicators = sum(r["bias_indicators_count"] for r in successful_results)
    total_framing_patterns = sum(r["framing_patterns_count"] for r in successful_results)
    total_narrative_themes = sum(r["narrative_themes_count"] for r in successful_results)
    
    # Aggregate confidence scores
    confidence_scores = [r["confidence"] for r in successful_results]
    avg_confidence = statistics.mean(confidence_scores)
    
    return {
        "bias_analysis": {
            "average_bias_score": avg_bias_score,
            "bias_score_range": {"min": min(bias_scores), "max": max(bias_scores)},
            "total_bias_indicators": total_bias_indicators,
            "high_bias_items": len([r for r in successful_results if r["overall_bias_score"] > 0.7])
        },
        "framing_analysis": {
            "total_framing_patterns": total_framing_patterns,
            "average_patterns_per_item": total_framing_patterns / len(successful_results)
        },
        "narrative_analysis": {
            "total_narrative_themes": total_narrative_themes,
            "average_themes_per_item": total_narrative_themes / len(successful_results)
        },
        "quality_metrics": {
            "average_confidence": avg_confidence,
            "confidence_range": {"min": min(confidence_scores), "max": max(confidence_scores)}
        }
    }


def _perform_comparative_analysis(batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Perform comparative analysis across batch results."""
    successful_results = [r for r in batch_results if r["success"]]
    
    if len(successful_results) < 2:
        return {"error": "Insufficient data for comparative analysis"}
    
    # Compare bias scores
    bias_scores = [r["overall_bias_score"] for r in successful_results]
    bias_variance = statistics.variance(bias_scores) if len(bias_scores) > 1 else 0
    
    # Identify outliers
    mean_bias = statistics.mean(bias_scores)
    std_bias = statistics.stdev(bias_scores) if len(bias_scores) > 1 else 0
    
    outliers = []
    for i, result in enumerate(successful_results):
        if abs(result["overall_bias_score"] - mean_bias) > 2 * std_bias:
            outliers.append({
                "item_index": result["item_index"],
                "bias_score": result["overall_bias_score"],
                "deviation": abs(result["overall_bias_score"] - mean_bias)
            })
    
    return {
        "bias_comparison": {
            "variance": bias_variance,
            "consistency": "high" if bias_variance < 0.1 else "medium" if bias_variance < 0.3 else "low",
            "outliers": outliers
        },
        "pattern_consistency": {
            "similar_framing": len([r for r in successful_results if r["framing_patterns_count"] > 0]),
            "narrative_alignment": len([r for r in successful_results if r["narrative_themes_count"] > 0])
        }
    }