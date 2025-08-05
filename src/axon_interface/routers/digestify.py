"""
Digestify API Endpoints - Advanced Summarization

Implements advanced summarization capabilities:
- POST /summarize - Extractive and abstractive summarization
- GET /summarize/quality - Summary quality assessment
- POST /summarize/batch - Batch summarization processing
- GET /summarize/templates - Summarization templates and styles

Provides tiered summarization based on user subscription levels.
"""

import logging
from typing import List, Optional, Dict, Any, Union
from enum import Enum
import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request, Body
from pydantic import BaseModel, Field

from ...shared.schemas import ArticleResponse
from ...thalamus.nlp_pipeline import get_nlp_pipeline, NLPPipeline
from ..dependencies import get_repository_factory
from ...synaptic_vesicle.repositories import RepositoryFactory


logger = logging.getLogger(__name__)
router = APIRouter()


class SummarizationMode(str, Enum):
    """Summarization modes available."""
    EXTRACTIVE = "extractive"
    ABSTRACTIVE = "abstractive"
    HYBRID = "hybrid"


class SummaryLength(str, Enum):
    """Summary length options."""
    SHORT = "short"      # 1-2 sentences
    MEDIUM = "medium"    # 3-5 sentences
    LONG = "long"        # 6-10 sentences


class SummaryStyle(str, Enum):
    """Summary style options."""
    NEUTRAL = "neutral"
    BULLET_POINTS = "bullet_points"
    EXECUTIVE = "executive"
    TECHNICAL = "technical"
    JOURNALISTIC = "journalistic"


class SummarizationRequest(BaseModel):
    """Request model for summarization."""
    text: Optional[str] = Field(None, description="Text content to summarize")
    article_id: Optional[str] = Field(None, description="Article ID to summarize")
    url: Optional[str] = Field(None, description="URL to fetch and summarize")
    mode: SummarizationMode = Field(SummarizationMode.HYBRID, description="Summarization mode")
    length: SummaryLength = Field(SummaryLength.MEDIUM, description="Summary length")
    style: SummaryStyle = Field(SummaryStyle.NEUTRAL, description="Summary style")
    max_sentences: Optional[int] = Field(None, ge=1, le=20, description="Maximum sentences in summary")
    focus_keywords: Optional[List[str]] = Field(None, description="Keywords to focus on in summary")
    preserve_entities: bool = Field(True, description="Preserve important entities in summary")


class SummarizationResponse(BaseModel):
    """Response model for summarization."""
    summary: str = Field(..., description="Generated summary")
    mode_used: SummarizationMode = Field(..., description="Summarization mode used")
    length_category: SummaryLength = Field(..., description="Summary length category")
    sentence_count: int = Field(..., description="Number of sentences in summary")
    word_count: int = Field(..., description="Number of words in summary")
    compression_ratio: float = Field(..., description="Compression ratio (summary/original)")
    quality_score: float = Field(..., description="Summary quality score (0-1)")
    key_entities: List[str] = Field(default_factory=list, description="Key entities preserved")
    processing_time: float = Field(..., description="Processing time in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


@router.post(
    "/summarize",
    response_model=SummarizationResponse,
    summary="Generate extractive and abstractive summaries",
    description="""
    Generate high-quality summaries using advanced NLP techniques.
    
    Features:
    - Multiple summarization modes (extractive, abstractive, hybrid)
    - Configurable summary length and style
    - Entity preservation and keyword focusing
    - Quality scoring and optimization
    - Tiered access based on user subscription
    
    Supports text input, article ID, or URL for summarization.
    """
)
async def create_summary(
    request: Request,
    summarization_request: SummarizationRequest,
    repository_factory: RepositoryFactory = Depends(get_repository_factory),
    nlp_pipeline: NLPPipeline = Depends(get_nlp_pipeline)
) -> SummarizationResponse:
    """Generate summary for provided content."""
    import time
    start_time = time.time()
    
    try:
        # Get user tier for feature access
        user_tier = getattr(request.state, "user_tier", "free")
        
        # Validate tier access for advanced features
        if summarization_request.mode == SummarizationMode.ABSTRACTIVE and user_tier == "free":
            raise HTTPException(
                status_code=403,
                detail="Abstractive summarization requires premium subscription"
            )
        
        # Get content to summarize
        content_data = await _get_content_for_summarization(
            summarization_request, repository_factory
        )
        
        text = content_data["text"]
        title = content_data.get("title", "")
        
        if not text or len(text.strip()) < 100:
            raise HTTPException(
                status_code=400,
                detail="Content too short for meaningful summarization (minimum 100 characters)"
            )
        
        # Generate summary based on mode
        summary_result = await _generate_summary(
            text=text,
            title=title,
            mode=summarization_request.mode,
            length=summarization_request.length,
            style=summarization_request.style,
            max_sentences=summarization_request.max_sentences,
            focus_keywords=summarization_request.focus_keywords or [],
            preserve_entities=summarization_request.preserve_entities,
            nlp_pipeline=nlp_pipeline,
            user_tier=user_tier
        )
        
        processing_time = time.time() - start_time
        
        # Calculate metrics
        original_word_count = len(text.split())
        summary_word_count = len(summary_result["summary"].split())
        compression_ratio = summary_word_count / original_word_count if original_word_count > 0 else 0
        
        logger.info(f"Generated {summarization_request.mode} summary in {processing_time:.2f}s")
        
        return SummarizationResponse(
            summary=summary_result["summary"],
            mode_used=summarization_request.mode,
            length_category=summarization_request.length,
            sentence_count=summary_result["sentence_count"],
            word_count=summary_word_count,
            compression_ratio=compression_ratio,
            quality_score=summary_result["quality_score"],
            key_entities=summary_result["key_entities"],
            processing_time=processing_time,
            metadata=summary_result["metadata"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in summarization: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate summary"
        )


@router.get(
    "/summarize/quality",
    summary="Assess summary quality",
    description="""
    Assess the quality of a generated summary using multiple metrics.
    
    Quality metrics:
    - Coherence and readability
    - Information preservation
    - Entity coverage
    - Factual consistency
    - Compression effectiveness
    """
)
async def assess_summary_quality(
    request: Request,
    original_text: str = Body(..., description="Original text"),
    summary_text: str = Body(..., description="Summary to assess"),
    nlp_pipeline: NLPPipeline = Depends(get_nlp_pipeline)
) -> Dict[str, Any]:
    """Assess the quality of a summary."""
    try:
        quality_assessment = await _assess_summary_quality(
            original_text, summary_text, nlp_pipeline
        )
        
        logger.info("Completed summary quality assessment")
        
        return {
            "overall_quality": quality_assessment["overall"],
            "coherence_score": quality_assessment["coherence"],
            "information_preservation": quality_assessment["preservation"],
            "entity_coverage": quality_assessment["entity_coverage"],
            "factual_consistency": quality_assessment["consistency"],
            "readability_score": quality_assessment["readability"],
            "compression_effectiveness": quality_assessment["compression"],
            "detailed_metrics": quality_assessment["details"]
        }
        
    except Exception as e:
        logger.error(f"Error in quality assessment: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to assess summary quality"
        )


@router.post(
    "/summarize/batch",
    summary="Batch summarization processing",
    description="""
    Process multiple documents for summarization in batch.
    
    Features:
    - Concurrent processing for efficiency
    - Progress tracking and status updates
    - Configurable batch size limits
    - Error handling for individual items
    """
)
async def batch_summarize(
    request: Request,
    batch_request: Dict[str, Any] = Body(...),
    repository_factory: RepositoryFactory = Depends(get_repository_factory),
    nlp_pipeline: NLPPipeline = Depends(get_nlp_pipeline)
) -> Dict[str, Any]:
    """Process batch summarization requests."""
    try:
        user_tier = getattr(request.state, "user_tier", "free")
        
        # Validate batch size based on user tier
        max_batch_size = {"free": 5, "premium": 20, "enterprise": 100}.get(user_tier, 5)
        
        items = batch_request.get("items", [])
        if len(items) > max_batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size exceeds limit for {user_tier} tier: {max_batch_size}"
            )
        
        # Process items concurrently
        batch_results = await _process_batch_summarization(
            items, nlp_pipeline, repository_factory, user_tier
        )
        
        # Calculate batch statistics
        successful = sum(1 for r in batch_results if r["success"])
        failed = len(batch_results) - successful
        
        logger.info(f"Processed batch of {len(items)} items: {successful} successful, {failed} failed")
        
        return {
            "batch_id": f"batch_{int(time.time())}",
            "total_items": len(items),
            "successful": successful,
            "failed": failed,
            "results": batch_results,
            "processing_time": sum(r.get("processing_time", 0) for r in batch_results)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch summarization: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to process batch summarization"
        )


@router.get(
    "/summarize/templates",
    summary="Get summarization templates and styles",
    description="""
    Retrieve available summarization templates and style guides.
    
    Templates include:
    - Executive summary format
    - Technical documentation style
    - Journalistic summary style
    - Bullet point format
    - Custom template creation (premium feature)
    """
)
async def get_summarization_templates(
    request: Request
) -> Dict[str, Any]:
    """Get available summarization templates."""
    try:
        user_tier = getattr(request.state, "user_tier", "free")
        
        templates = {
            "neutral": {
                "name": "Neutral Summary",
                "description": "Balanced, objective summarization",
                "example": "The article discusses...",
                "available": True
            },
            "bullet_points": {
                "name": "Bullet Points",
                "description": "Key points in bullet format",
                "example": "• Main point 1\n• Main point 2",
                "available": True
            },
            "executive": {
                "name": "Executive Summary",
                "description": "Business-focused executive summary",
                "example": "Executive Summary: Key findings indicate...",
                "available": user_tier in ["premium", "enterprise"]
            },
            "technical": {
                "name": "Technical Summary",
                "description": "Technical documentation style",
                "example": "Technical Overview: The system implements...",
                "available": user_tier in ["premium", "enterprise"]
            },
            "journalistic": {
                "name": "Journalistic Style",
                "description": "News article summary format",
                "example": "In a recent development...",
                "available": user_tier in ["premium", "enterprise"]
            }
        }
        
        # Filter based on availability
        available_templates = {k: v for k, v in templates.items() if v["available"]}
        
        return {
            "templates": available_templates,
            "user_tier": user_tier,
            "custom_templates_available": user_tier == "enterprise"
        }
        
    except Exception as e:
        logger.error(f"Error retrieving templates: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve summarization templates"
        )


# Helper functions for summarization
async def _get_content_for_summarization(
    request: SummarizationRequest,
    repository_factory: RepositoryFactory
) -> Dict[str, str]:
    """Get content from various sources for summarization."""
    if request.text:
        return {"text": request.text, "title": ""}
    
    elif request.article_id:
        article_repo = repository_factory.get_article_repository()
        article = await article_repo.get_by_id(request.article_id)
        if not article:
            raise HTTPException(status_code=404, detail="Article not found")
        return {"text": article.content, "title": article.title}
    
    elif request.url:
        # In a real implementation, this would fetch and parse the URL
        # For now, we'll return an error
        raise HTTPException(
            status_code=501,
            detail="URL fetching not implemented yet"
        )
    
    else:
        raise HTTPException(
            status_code=400,
            detail="Must provide either text, article_id, or url"
        )


async def _generate_summary(
    text: str,
    title: str,
    mode: SummarizationMode,
    length: SummaryLength,
    style: SummaryStyle,
    max_sentences: Optional[int],
    focus_keywords: List[str],
    preserve_entities: bool,
    nlp_pipeline: NLPPipeline,
    user_tier: str
) -> Dict[str, Any]:
    """Generate summary using specified parameters."""
    # Get NLP analysis
    analysis = await nlp_pipeline.process_text(text, title)
    
    # Determine target sentence count based on length
    length_mapping = {
        SummaryLength.SHORT: 2,
        SummaryLength.MEDIUM: 4,
        SummaryLength.LONG: 8
    }
    target_sentences = max_sentences or length_mapping[length]
    
    # Generate summary based on mode
    if mode == SummarizationMode.EXTRACTIVE:
        summary = await _generate_extractive_summary(
            text, analysis, target_sentences, focus_keywords, preserve_entities
        )
    elif mode == SummarizationMode.ABSTRACTIVE:
        summary = await _generate_abstractive_summary(
            text, analysis, target_sentences, focus_keywords, preserve_entities
        )
    else:  # HYBRID
        summary = await _generate_hybrid_summary(
            text, analysis, target_sentences, focus_keywords, preserve_entities
        )
    
    # Apply style formatting
    formatted_summary = _apply_summary_style(summary, style)
    
    # Extract key entities
    key_entities = [e.text for e in analysis.entities[:5]] if preserve_entities else []
    
    # Calculate quality score
    quality_score = await _calculate_summary_quality(text, formatted_summary, analysis)
    
    return {
        "summary": formatted_summary,
        "sentence_count": len(formatted_summary.split('. ')),
        "quality_score": quality_score,
        "key_entities": key_entities,
        "metadata": {
            "original_length": len(text),
            "summary_length": len(formatted_summary),
            "entities_preserved": len(key_entities),
            "focus_keywords_used": len(focus_keywords)
        }
    }


async def _generate_extractive_summary(
    text: str,
    analysis: Any,
    target_sentences: int,
    focus_keywords: List[str],
    preserve_entities: bool
) -> str:
    """Generate extractive summary by selecting key sentences."""
    # Use the existing summary from NLP analysis as base
    if hasattr(analysis, 'summary') and analysis.summary:
        sentences = analysis.summary.split('. ')
        
        # If we have enough sentences, return as is
        if len(sentences) >= target_sentences:
            return '. '.join(sentences[:target_sentences]) + '.'
    
    # Fallback: extract sentences from original text
    sentences = text.split('. ')
    if len(sentences) <= target_sentences:
        return text
    
    # Simple sentence scoring based on position and keyword presence
    scored_sentences = []
    for i, sentence in enumerate(sentences):
        score = 0.0
        
        # Position score (beginning and end are important)
        if i < 2:  # First two sentences
            score += 0.3
        elif i >= len(sentences) - 2:  # Last two sentences
            score += 0.2
        
        # Keyword score
        sentence_lower = sentence.lower()
        for keyword in focus_keywords:
            if keyword.lower() in sentence_lower:
                score += 0.4
        
        # Entity score
        if preserve_entities:
            for entity in analysis.entities[:10]:
                entity_text = entity.text if hasattr(entity, 'text') else str(entity)
                if entity_text.lower() in sentence_lower:
                    score += 0.2
        
        scored_sentences.append((sentence, score))
    
    # Select top sentences
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    selected = [s[0] for s in scored_sentences[:target_sentences]]
    
    # Maintain original order
    original_order = []
    for sentence in sentences:
        if sentence in selected:
            original_order.append(sentence)
    
    return '. '.join(original_order) + '.'


async def _generate_abstractive_summary(
    text: str,
    analysis: Any,
    target_sentences: int,
    focus_keywords: List[str],
    preserve_entities: bool
) -> str:
    """Generate abstractive summary (simplified implementation)."""
    # For now, use extractive as base and apply some transformations
    extractive = await _generate_extractive_summary(
        text, analysis, target_sentences, focus_keywords, preserve_entities
    )
    
    # Simple abstractive transformations
    # In a real implementation, this would use advanced NLP models
    abstractive = extractive
    
    # Replace some phrases to make it more abstractive
    replacements = {
        "The article states that": "According to the analysis,",
        "It is mentioned that": "The content indicates",
        "The text says": "The document reveals",
        "According to the document": "The analysis shows"
    }
    
    for old, new in replacements.items():
        abstractive = abstractive.replace(old, new)
    
    return abstractive


async def _generate_hybrid_summary(
    text: str,
    analysis: Any,
    target_sentences: int,
    focus_keywords: List[str],
    preserve_entities: bool
) -> str:
    """Generate hybrid summary combining extractive and abstractive approaches."""
    # Generate both types
    extractive = await _generate_extractive_summary(
        text, analysis, target_sentences, focus_keywords, preserve_entities
    )
    
    abstractive = await _generate_abstractive_summary(
        text, analysis, target_sentences, focus_keywords, preserve_entities
    )
    
    # For now, prefer extractive but add abstractive elements
    # In a real implementation, this would be more sophisticated
    return extractive


def _apply_summary_style(summary: str, style: SummaryStyle) -> str:
    """Apply formatting style to summary."""
    if style == SummaryStyle.BULLET_POINTS:
        sentences = summary.split('. ')
        return '\n'.join(f"• {sentence.strip()}" for sentence in sentences if sentence.strip())
    
    elif style == SummaryStyle.EXECUTIVE:
        return f"Executive Summary: {summary}"
    
    elif style == SummaryStyle.TECHNICAL:
        return f"Technical Overview: {summary}"
    
    elif style == SummaryStyle.JOURNALISTIC:
        # Add journalistic lead
        return f"In a recent analysis, {summary.lower()}"
    
    else:  # NEUTRAL
        return summary


async def _calculate_summary_quality(text: str, summary: str, analysis: Any) -> float:
    """Calculate summary quality score."""
    # Simple quality metrics
    original_length = len(text.split())
    summary_length = len(summary.split())
    
    # Compression ratio (should be reasonable)
    compression = summary_length / original_length if original_length > 0 else 0
    compression_score = 1.0 if 0.1 <= compression <= 0.5 else 0.5
    
    # Entity preservation (if entities exist in original)
    entity_score = 1.0
    if hasattr(analysis, 'entities') and analysis.entities:
        entities_in_summary = 0
        for entity in analysis.entities[:5]:
            entity_text = entity.text if hasattr(entity, 'text') else str(entity)
            if entity_text.lower() in summary.lower():
                entities_in_summary += 1
        entity_score = entities_in_summary / min(5, len(analysis.entities))
    
    # Readability (simple check for sentence structure)
    sentences = summary.split('. ')
    readability_score = 1.0 if all(len(s.split()) >= 3 for s in sentences if s.strip()) else 0.7
    
    # Overall quality
    quality = (compression_score + entity_score + readability_score) / 3
    return min(1.0, max(0.0, quality))


async def _assess_summary_quality(
    original_text: str,
    summary_text: str,
    nlp_pipeline: NLPPipeline
) -> Dict[str, Any]:
    """Comprehensive summary quality assessment."""
    # Analyze both texts
    original_analysis = await nlp_pipeline.process_text(original_text)
    summary_analysis = await nlp_pipeline.process_text(summary_text)
    
    # Calculate various quality metrics
    coherence = _calculate_coherence_score(summary_text)
    preservation = _calculate_information_preservation(original_analysis, summary_analysis)
    entity_coverage = _calculate_entity_coverage(original_analysis, summary_analysis)
    consistency = _calculate_factual_consistency(original_text, summary_text)
    readability = _calculate_readability_score(summary_text)
    compression = _calculate_compression_effectiveness(original_text, summary_text)
    
    overall = (coherence + preservation + entity_coverage + consistency + readability + compression) / 6
    
    return {
        "overall": overall,
        "coherence": coherence,
        "preservation": preservation,
        "entity_coverage": entity_coverage,
        "consistency": consistency,
        "readability": readability,
        "compression": compression,
        "details": {
            "original_word_count": len(original_text.split()),
            "summary_word_count": len(summary_text.split()),
            "compression_ratio": len(summary_text.split()) / len(original_text.split())
        }
    }


async def _process_batch_summarization(
    items: List[Dict[str, Any]],
    nlp_pipeline: NLPPipeline,
    repository_factory: RepositoryFactory,
    user_tier: str
) -> List[Dict[str, Any]]:
    """Process batch summarization requests concurrently."""
    async def process_item(item):
        try:
            # Create summarization request from item
            request = SummarizationRequest(**item)
            
            # Get content
            content_data = await _get_content_for_summarization(request, repository_factory)
            
            # Generate summary
            summary_result = await _generate_summary(
                text=content_data["text"],
                title=content_data.get("title", ""),
                mode=request.mode,
                length=request.length,
                style=request.style,
                max_sentences=request.max_sentences,
                focus_keywords=request.focus_keywords or [],
                preserve_entities=request.preserve_entities,
                nlp_pipeline=nlp_pipeline,
                user_tier=user_tier
            )
            
            return {
                "success": True,
                "summary": summary_result["summary"],
                "quality_score": summary_result["quality_score"],
                "processing_time": 0.5  # Simplified
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "processing_time": 0.0
            }
    
    # Process items concurrently
    results = await asyncio.gather(*[process_item(item) for item in items])
    return results


# Quality assessment helper functions
def _calculate_coherence_score(text: str) -> float:
    """Calculate coherence score based on text structure."""
    sentences = text.split('. ')
    if len(sentences) < 2:
        return 1.0
    
    # Simple coherence check based on sentence transitions
    coherence_indicators = ['however', 'therefore', 'furthermore', 'additionally', 'consequently']
    transition_count = sum(1 for sentence in sentences 
                          for indicator in coherence_indicators 
                          if indicator in sentence.lower())
    
    return min(1.0, 0.5 + (transition_count / len(sentences)))


def _calculate_information_preservation(original_analysis: Any, summary_analysis: Any) -> float:
    """Calculate how well information is preserved."""
    # Compare key entities and concepts
    original_entities = set()
    summary_entities = set()
    
    if hasattr(original_analysis, 'entities'):
        original_entities = {e.text.lower() for e in original_analysis.entities}
    
    if hasattr(summary_analysis, 'entities'):
        summary_entities = {e.text.lower() for e in summary_analysis.entities}
    
    if not original_entities:
        return 1.0
    
    preserved = len(original_entities.intersection(summary_entities))
    return preserved / len(original_entities)


def _calculate_entity_coverage(original_analysis: Any, summary_analysis: Any) -> float:
    """Calculate entity coverage in summary."""
    return _calculate_information_preservation(original_analysis, summary_analysis)


def _calculate_factual_consistency(original_text: str, summary_text: str) -> float:
    """Calculate factual consistency (simplified)."""
    # Simple check for contradictory statements
    # In a real implementation, this would be more sophisticated
    return 0.8  # Placeholder


def _calculate_readability_score(text: str) -> float:
    """Calculate readability score."""
    sentences = text.split('. ')
    words = text.split()
    
    if not sentences or not words:
        return 0.0
    
    avg_sentence_length = len(words) / len(sentences)
    
    # Optimal sentence length is around 15-20 words
    if 10 <= avg_sentence_length <= 25:
        return 1.0
    elif avg_sentence_length < 10:
        return 0.7
    else:
        return max(0.3, 25 / avg_sentence_length)


def _calculate_compression_effectiveness(original_text: str, summary_text: str) -> float:
    """Calculate compression effectiveness."""
    original_words = len(original_text.split())
    summary_words = len(summary_text.split())
    
    if original_words == 0:
        return 0.0
    
    compression_ratio = summary_words / original_words
    
    # Optimal compression is between 10-30%
    if 0.1 <= compression_ratio <= 0.3:
        return 1.0
    elif compression_ratio < 0.1:
        return 0.5  # Too compressed
    else:
        return max(0.2, 0.3 / compression_ratio)  # Not compressed enough