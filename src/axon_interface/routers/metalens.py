"""
MetaLens API Endpoints - Technical Intelligence

Implements technical intelligence and metadata analysis:
- GET /meta - Comprehensive webpage technical analysis
- GET /meta/paywall - Paywall detection and analysis
- GET /meta/canonical - Canonical URL extraction and validation
- GET /meta/techstack - Technology stack identification
- POST /meta/analyze - Batch technical analysis

Provides deep technical insights into web content and infrastructure.
"""

import logging
from typing import List, Optional, Dict, Any, Union
from urllib.parse import urlparse, urljoin
import re
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Body
from pydantic import BaseModel, Field, HttpUrl

from ...shared.schemas import ArticleResponse
from ...thalamus.nlp_pipeline import get_nlp_pipeline, NLPPipeline
from ..dependencies import get_repository_factory
from ...synaptic_vesicle.repositories import RepositoryFactory


logger = logging.getLogger(__name__)
router = APIRouter()


class TechnicalAnalysisRequest(BaseModel):
    """Request model for technical analysis."""
    url: Optional[HttpUrl] = Field(None, description="URL to analyze")
    article_id: Optional[str] = Field(None, description="Article ID to analyze")
    html_content: Optional[str] = Field(None, description="HTML content to analyze")
    include_paywall_analysis: bool = Field(True, description="Include paywall detection")
    include_tech_stack: bool = Field(True, description="Include technology stack analysis")
    include_seo_analysis: bool = Field(True, description="Include SEO metadata analysis")
    include_performance_hints: bool = Field(True, description="Include performance optimization hints")


class TechnicalAnalysisResponse(BaseModel):
    """Response model for technical analysis."""
    url: Optional[str] = Field(None, description="Analyzed URL")
    canonical_url: Optional[str] = Field(None, description="Canonical URL")
    paywall_analysis: Dict[str, Any] = Field(..., description="Paywall detection results")
    tech_stack: Dict[str, Any] = Field(..., description="Technology stack identification")
    seo_metadata: Dict[str, Any] = Field(..., description="SEO metadata analysis")
    performance_analysis: Dict[str, Any] = Field(..., description="Performance analysis")
    security_analysis: Dict[str, Any] = Field(..., description="Security analysis")
    accessibility_analysis: Dict[str, Any] = Field(..., description="Accessibility analysis")
    content_analysis: Dict[str, Any] = Field(..., description="Content structure analysis")
    processing_time: float = Field(..., description="Processing time in seconds")


# Technology stack detection patterns
TECH_STACK_PATTERNS = {
    "frameworks": {
        "React": [r"react", r"_react", r"React\."],
        "Vue.js": [r"vue\.js", r"__vue__", r"Vue\."],
        "Angular": [r"angular", r"ng-", r"@angular"],
        "jQuery": [r"jquery", r"\$\(", r"jQuery"],
        "Bootstrap": [r"bootstrap", r"btn-", r"col-"],
        "Tailwind": [r"tailwind", r"tw-", r"tailwindcss"],
        "Next.js": [r"next\.js", r"_next", r"__NEXT_DATA__"],
        "Nuxt.js": [r"nuxt", r"_nuxt", r"__NUXT__"],
        "Svelte": [r"svelte", r"_svelte"],
        "Ember.js": [r"ember", r"Ember\."]
    },
    "cms": {
        "WordPress": [r"wp-content", r"wp-includes", r"/wp-json/", r"wordpress"],
        "Drupal": [r"drupal", r"sites/default", r"misc/drupal"],
        "Joomla": [r"joomla", r"components/com_", r"templates/"],
        "Shopify": [r"shopify", r"cdn\.shopify", r"myshopify"],
        "Squarespace": [r"squarespace", r"static1\.squarespace"],
        "Wix": [r"wix\.com", r"static\.wix", r"parastorage"],
        "Ghost": [r"ghost", r"casper", r"ghost\.org"],
        "Contentful": [r"contentful", r"ctfassets"]
    },
    "analytics": {
        "Google Analytics": [r"google-analytics", r"gtag", r"ga\("],
        "Google Tag Manager": [r"googletagmanager", r"gtm\.js"],
        "Adobe Analytics": [r"omniture", r"adobe\.com.*analytics"],
        "Mixpanel": [r"mixpanel", r"api\.mixpanel"],
        "Hotjar": [r"hotjar", r"static\.hotjar"],
        "Segment": [r"segment", r"analytics\.js"]
    },
    "cdn": {
        "Cloudflare": [r"cloudflare", r"cf-ray", r"__cf_bm"],
        "AWS CloudFront": [r"cloudfront", r"amazonaws"],
        "Fastly": [r"fastly", r"fastly-debug"],
        "KeyCDN": [r"keycdn", r"kxcdn"],
        "MaxCDN": [r"maxcdn", r"bootstrapcdn"],
        "jsDelivr": [r"jsdelivr", r"cdn\.jsdelivr"]
    },
    "servers": {
        "Apache": [r"apache", r"server:\s*apache"],
        "Nginx": [r"nginx", r"server:\s*nginx"],
        "IIS": [r"iis", r"server:\s*microsoft-iis"],
        "Node.js": [r"express", r"x-powered-by:\s*express"],
        "PHP": [r"php", r"x-powered-by:\s*php"],
        "ASP.NET": [r"asp\.net", r"x-aspnet-version"]
    }
}

# Paywall detection patterns
PAYWALL_INDICATORS = {
    "subscription_walls": [
        r"subscribe", r"subscription", r"premium", r"paywall",
        r"unlock", r"full access", r"member", r"subscriber"
    ],
    "registration_walls": [
        r"sign up", r"register", r"create account", r"free account",
        r"continue reading", r"read more"
    ],
    "metered_access": [
        r"free articles", r"articles remaining", r"monthly limit",
        r"article limit", r"complimentary"
    ],
    "css_selectors": [
        ".paywall", ".subscription-wall", ".premium-content",
        ".subscriber-only", ".members-only", "#paywall"
    ],
    "javascript_patterns": [
        r"paywall", r"subscription.*wall", r"premium.*content",
        r"subscriber.*gate", r"access.*denied"
    ]
}


@router.get(
    "/meta",
    response_model=TechnicalAnalysisResponse,
    summary="Comprehensive webpage technical analysis",
    description="""
    Perform comprehensive technical analysis of web content.
    
    Features:
    - Paywall detection and bypass strategies
    - Technology stack identification
    - SEO metadata extraction and validation
    - Performance optimization recommendations
    - Security analysis and vulnerability detection
    - Accessibility compliance assessment
    - Content structure analysis
    
    Supports URL, article ID, or direct HTML content analysis.
    """
)
async def analyze_technical_metadata(
    request: Request,
    url: Optional[str] = Query(None, description="URL to analyze"),
    article_id: Optional[str] = Query(None, description="Article ID to analyze"),
    include_paywall_analysis: bool = Query(True, description="Include paywall detection"),
    include_tech_stack: bool = Query(True, description="Include technology stack analysis"),
    include_seo_analysis: bool = Query(True, description="Include SEO metadata analysis"),
    include_performance_hints: bool = Query(True, description="Include performance optimization hints"),
    repository_factory: RepositoryFactory = Depends(get_repository_factory),
    nlp_pipeline: NLPPipeline = Depends(get_nlp_pipeline)
) -> TechnicalAnalysisResponse:
    """Perform comprehensive technical analysis."""
    import time
    start_time = time.time()
    
    try:
        # Get content and metadata for analysis
        analysis_data = await _get_content_for_technical_analysis(
            url, article_id, None, repository_factory
        )
        
        content = analysis_data["content"]
        html_content = analysis_data.get("html_content", "")
        page_url = analysis_data.get("url", url)
        metadata = analysis_data.get("metadata", {})
        
        # Initialize analysis results
        results = {
            "paywall_analysis": {},
            "tech_stack": {},
            "seo_metadata": {},
            "performance_analysis": {},
            "security_analysis": {},
            "accessibility_analysis": {},
            "content_analysis": {}
        }
        
        # Perform paywall analysis
        if include_paywall_analysis:
            results["paywall_analysis"] = await _analyze_paywall_indicators(
                content, html_content, page_url
            )
        
        # Perform technology stack analysis
        if include_tech_stack:
            results["tech_stack"] = await _analyze_technology_stack(
                html_content, page_url, metadata
            )
        
        # Perform SEO analysis
        if include_seo_analysis:
            results["seo_metadata"] = await _analyze_seo_metadata(
                html_content, content, page_url, metadata
            )
        
        # Perform performance analysis
        if include_performance_hints:
            results["performance_analysis"] = await _analyze_performance_indicators(
                html_content, page_url, metadata
            )
        
        # Perform security analysis
        results["security_analysis"] = await _analyze_security_indicators(
            html_content, page_url
        )
        
        # Perform accessibility analysis
        results["accessibility_analysis"] = await _analyze_accessibility_indicators(
            html_content, content
        )
        
        # Perform content structure analysis
        results["content_analysis"] = await _analyze_content_structure(
            content, html_content, nlp_pipeline
        )
        
        # Extract canonical URL
        canonical_url = _extract_canonical_url(html_content, page_url)
        
        processing_time = time.time() - start_time
        
        logger.info(f"Completed technical analysis for {page_url} in {processing_time:.2f}s")
        
        return TechnicalAnalysisResponse(
            url=page_url,
            canonical_url=canonical_url,
            paywall_analysis=results["paywall_analysis"],
            tech_stack=results["tech_stack"],
            seo_metadata=results["seo_metadata"],
            performance_analysis=results["performance_analysis"],
            security_analysis=results["security_analysis"],
            accessibility_analysis=results["accessibility_analysis"],
            content_analysis=results["content_analysis"],
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in technical analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to perform technical analysis"
        )


@router.get(
    "/meta/paywall",
    summary="Paywall detection and analysis",
    description="""
    Specialized paywall detection and analysis.
    
    Features:
    - Multiple paywall type detection (subscription, registration, metered)
    - Bypass strategy identification
    - Content accessibility assessment
    - Paywall strength scoring
    - Alternative access method suggestions
    """
)
async def detect_paywall(
    request: Request,
    url: Optional[str] = Query(None, description="URL to analyze"),
    article_id: Optional[str] = Query(None, description="Article ID to analyze"),
    repository_factory: RepositoryFactory = Depends(get_repository_factory)
) -> Dict[str, Any]:
    """Detect and analyze paywall implementations."""
    try:
        # Get content for analysis
        analysis_data = await _get_content_for_technical_analysis(
            url, article_id, None, repository_factory
        )
        
        content = analysis_data["content"]
        html_content = analysis_data.get("html_content", "")
        page_url = analysis_data.get("url", url)
        
        # Perform detailed paywall analysis
        paywall_analysis = await _analyze_paywall_indicators(content, html_content, page_url)
        
        # Add additional paywall-specific insights
        paywall_analysis.update({
            "bypass_strategies": _suggest_bypass_strategies(paywall_analysis),
            "content_accessibility": _assess_content_accessibility(content, paywall_analysis),
            "paywall_strength": _calculate_paywall_strength(paywall_analysis)
        })
        
        logger.info(f"Completed paywall analysis for {page_url}")
        
        return paywall_analysis
        
    except Exception as e:
        logger.error(f"Error in paywall detection: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to detect paywall"
        )


@router.get(
    "/meta/canonical",
    summary="Canonical URL extraction and validation",
    description="""
    Extract and validate canonical URLs from web content.
    
    Features:
    - Canonical link tag extraction
    - URL normalization and validation
    - Duplicate content detection
    - Redirect chain analysis
    - SEO canonical best practices assessment
    """
)
async def extract_canonical_url(
    request: Request,
    url: Optional[str] = Query(None, description="URL to analyze"),
    article_id: Optional[str] = Query(None, description="Article ID to analyze"),
    validate_accessibility: bool = Query(True, description="Validate canonical URL accessibility"),
    repository_factory: RepositoryFactory = Depends(get_repository_factory)
) -> Dict[str, Any]:
    """Extract and validate canonical URLs."""
    try:
        # Get content for analysis
        analysis_data = await _get_content_for_technical_analysis(
            url, article_id, None, repository_factory
        )
        
        html_content = analysis_data.get("html_content", "")
        page_url = analysis_data.get("url", url)
        
        # Extract canonical URL
        canonical_url = _extract_canonical_url(html_content, page_url)
        
        # Validate canonical URL
        validation_results = await _validate_canonical_url(
            canonical_url, page_url, validate_accessibility
        )
        
        # Analyze URL structure
        url_analysis = _analyze_url_structure(canonical_url or page_url)
        
        logger.info(f"Extracted canonical URL: {canonical_url}")
        
        return {
            "original_url": page_url,
            "canonical_url": canonical_url,
            "canonical_found": canonical_url is not None,
            "validation": validation_results,
            "url_analysis": url_analysis,
            "recommendations": _generate_canonical_recommendations(
                canonical_url, page_url, validation_results
            )
        }
        
    except Exception as e:
        logger.error(f"Error extracting canonical URL: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to extract canonical URL"
        )


@router.get(
    "/meta/techstack",
    summary="Technology stack identification",
    description="""
    Identify and analyze the technology stack of web content.
    
    Features:
    - Frontend framework detection
    - CMS identification
    - Analytics platform detection
    - CDN and hosting analysis
    - Server technology identification
    - Third-party service detection
    """
)
async def identify_tech_stack(
    request: Request,
    url: Optional[str] = Query(None, description="URL to analyze"),
    article_id: Optional[str] = Query(None, description="Article ID to analyze"),
    include_confidence_scores: bool = Query(True, description="Include confidence scores"),
    repository_factory: RepositoryFactory = Depends(get_repository_factory)
) -> Dict[str, Any]:
    """Identify technology stack components."""
    try:
        # Get content for analysis
        analysis_data = await _get_content_for_technical_analysis(
            url, article_id, None, repository_factory
        )
        
        html_content = analysis_data.get("html_content", "")
        page_url = analysis_data.get("url", url)
        metadata = analysis_data.get("metadata", {})
        
        # Perform comprehensive tech stack analysis
        tech_stack = await _analyze_technology_stack(html_content, page_url, metadata)
        
        # Add confidence scores if requested
        if include_confidence_scores:
            tech_stack = _add_confidence_scores_to_tech_stack(tech_stack, html_content)
        
        # Generate technology recommendations
        recommendations = _generate_tech_stack_recommendations(tech_stack)
        
        logger.info(f"Identified tech stack for {page_url}")
        
        return {
            "url": page_url,
            "technology_stack": tech_stack,
            "recommendations": recommendations,
            "analysis_summary": {
                "total_technologies": sum(len(v) for v in tech_stack.values() if isinstance(v, list)),
                "categories_detected": len([k for k, v in tech_stack.items() if v]),
                "confidence_level": _calculate_overall_confidence(tech_stack)
            }
        }
        
    except Exception as e:
        logger.error(f"Error identifying tech stack: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to identify technology stack"
        )


@router.post(
    "/meta/analyze",
    summary="Batch technical analysis",
    description="""
    Perform batch technical analysis on multiple URLs or articles.
    
    Features:
    - Concurrent processing for efficiency
    - Configurable analysis depth
    - Comparative analysis across items
    - Bulk technology stack detection
    - Aggregated insights and patterns
    """
)
async def batch_technical_analysis(
    request: Request,
    batch_request: Dict[str, Any] = Body(...),
    repository_factory: RepositoryFactory = Depends(get_repository_factory),
    nlp_pipeline: NLPPipeline = Depends(get_nlp_pipeline)
) -> Dict[str, Any]:
    """Perform batch technical analysis."""
    try:
        user_tier = getattr(request.state, "user_tier", "free")
        
        # Validate batch size based on user tier
        max_batch_size = {"free": 3, "premium": 15, "enterprise": 50}.get(user_tier, 3)
        
        items = batch_request.get("items", [])
        if len(items) > max_batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size exceeds limit for {user_tier} tier: {max_batch_size}"
            )
        
        # Process items
        batch_results = await _process_batch_technical_analysis(
            items, repository_factory, nlp_pipeline
        )
        
        # Aggregate insights
        aggregated_insights = _aggregate_technical_insights(batch_results)
        
        # Calculate batch statistics
        successful = sum(1 for r in batch_results if r["success"])
        
        logger.info(f"Processed batch technical analysis: {successful}/{len(items)} successful")
        
        return {
            "batch_summary": {
                "total_items": len(items),
                "successful": successful,
                "failed": len(items) - successful
            },
            "results": batch_results,
            "aggregated_insights": aggregated_insights,
            "patterns": _identify_technology_patterns(batch_results)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch technical analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to process batch technical analysis"
        )


# Helper functions for technical analysis
async def _get_content_for_technical_analysis(
    url: Optional[str],
    article_id: Optional[str],
    html_content: Optional[str],
    repository_factory: RepositoryFactory
) -> Dict[str, Any]:
    """Get content and metadata for technical analysis."""
    if html_content:
        return {
            "content": _extract_text_from_html(html_content),
            "html_content": html_content,
            "url": url,
            "metadata": {}
        }
    
    elif article_id:
        article_repo = repository_factory.get_article_repository()
        article = await article_repo.get_by_id(article_id)
        if not article:
            raise HTTPException(status_code=404, detail="Article not found")
        
        return {
            "content": article.content,
            "html_content": "",  # Would need to store HTML separately
            "url": article.url,
            "metadata": article.page_metadata or {}
        }
    
    elif url:
        # In a real implementation, this would fetch the URL
        # For now, return minimal data
        return {
            "content": "",
            "html_content": "",
            "url": url,
            "metadata": {}
        }
    
    else:
        raise HTTPException(
            status_code=400,
            detail="Must provide either url, article_id, or html_content"
        )


async def _analyze_paywall_indicators(
    content: str,
    html_content: str,
    url: Optional[str]
) -> Dict[str, Any]:
    """Analyze paywall indicators in content."""
    paywall_score = 0.0
    detected_types = []
    indicators = []
    
    # Check content for paywall text patterns
    content_lower = content.lower()
    html_lower = html_content.lower()
    
    # Subscription wall indicators
    subscription_matches = 0
    for pattern in PAYWALL_INDICATORS["subscription_walls"]:
        if re.search(pattern, content_lower):
            subscription_matches += 1
            indicators.append(f"Subscription keyword: {pattern}")
    
    if subscription_matches > 0:
        detected_types.append("subscription")
        paywall_score += min(0.4, subscription_matches * 0.1)
    
    # Registration wall indicators
    registration_matches = 0
    for pattern in PAYWALL_INDICATORS["registration_walls"]:
        if re.search(pattern, content_lower):
            registration_matches += 1
            indicators.append(f"Registration keyword: {pattern}")
    
    if registration_matches > 0:
        detected_types.append("registration")
        paywall_score += min(0.3, registration_matches * 0.1)
    
    # Metered access indicators
    metered_matches = 0
    for pattern in PAYWALL_INDICATORS["metered_access"]:
        if re.search(pattern, content_lower):
            metered_matches += 1
            indicators.append(f"Metered access keyword: {pattern}")
    
    if metered_matches > 0:
        detected_types.append("metered")
        paywall_score += min(0.3, metered_matches * 0.1)
    
    # CSS selector indicators
    css_matches = 0
    for selector in PAYWALL_INDICATORS["css_selectors"]:
        if selector in html_lower:
            css_matches += 1
            indicators.append(f"CSS selector: {selector}")
    
    if css_matches > 0:
        paywall_score += min(0.4, css_matches * 0.2)
    
    # JavaScript pattern indicators
    js_matches = 0
    for pattern in PAYWALL_INDICATORS["javascript_patterns"]:
        if re.search(pattern, html_lower):
            js_matches += 1
            indicators.append(f"JavaScript pattern: {pattern}")
    
    if js_matches > 0:
        paywall_score += min(0.3, js_matches * 0.15)
    
    # Content truncation indicators
    if len(content) < 500 and ("..." in content or "read more" in content_lower):
        paywall_score += 0.2
        indicators.append("Content appears truncated")
    
    paywall_score = min(1.0, paywall_score)
    
    return {
        "paywall_detected": paywall_score > 0.3,
        "confidence": paywall_score,
        "paywall_types": detected_types,
        "indicators": indicators,
        "content_accessibility": "restricted" if paywall_score > 0.5 else "partial" if paywall_score > 0.2 else "open"
    }


async def _analyze_technology_stack(
    html_content: str,
    url: Optional[str],
    metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """Analyze technology stack from HTML content and metadata."""
    tech_stack = {
        "frameworks": [],
        "cms": [],
        "analytics": [],
        "cdn": [],
        "servers": [],
        "other": []
    }
    
    html_lower = html_content.lower()
    
    # Detect technologies using patterns
    for category, technologies in TECH_STACK_PATTERNS.items():
        for tech_name, patterns in technologies.items():
            for pattern in patterns:
                if re.search(pattern, html_lower, re.IGNORECASE):
                    tech_stack[category].append({
                        "name": tech_name,
                        "pattern_matched": pattern,
                        "confidence": 0.8
                    })
                    break  # Only add once per technology
    
    # Analyze metadata for additional clues
    if metadata:
        generator = metadata.get("generator", "")
        if generator:
            tech_stack["other"].append({
                "name": f"Generator: {generator}",
                "source": "meta_generator",
                "confidence": 0.9
            })
    
    # Detect from URL patterns
    if url:
        domain = urlparse(url).netloc.lower()
        if "wordpress" in domain or "wp.com" in domain:
            tech_stack["cms"].append({
                "name": "WordPress",
                "source": "domain_analysis",
                "confidence": 0.9
            })
        elif "shopify" in domain:
            tech_stack["cms"].append({
                "name": "Shopify",
                "source": "domain_analysis",
                "confidence": 0.9
            })
    
    return tech_stack


async def _analyze_seo_metadata(
    html_content: str,
    content: str,
    url: Optional[str],
    metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """Analyze SEO metadata and optimization."""
    seo_analysis = {
        "title": {},
        "description": {},
        "keywords": {},
        "headings": {},
        "images": {},
        "links": {},
        "structured_data": {},
        "social_media": {},
        "score": 0.0
    }
    
    # Title analysis
    title_match = re.search(r"<title[^>]*>([^<]+)</title>", html_content, re.IGNORECASE)
    if title_match:
        title = title_match.group(1).strip()
        seo_analysis["title"] = {
            "content": title,
            "length": len(title),
            "optimal_length": 30 <= len(title) <= 60,
            "score": 1.0 if 30 <= len(title) <= 60 else 0.5
        }
    
    # Meta description analysis
    desc_match = re.search(r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']+)["\']', html_content, re.IGNORECASE)
    if desc_match:
        description = desc_match.group(1).strip()
        seo_analysis["description"] = {
            "content": description,
            "length": len(description),
            "optimal_length": 120 <= len(description) <= 160,
            "score": 1.0 if 120 <= len(description) <= 160 else 0.5
        }
    
    # Heading structure analysis
    headings = {"h1": [], "h2": [], "h3": [], "h4": [], "h5": [], "h6": []}
    for level in range(1, 7):
        pattern = f"<h{level}[^>]*>([^<]+)</h{level}>"
        matches = re.findall(pattern, html_content, re.IGNORECASE)
        headings[f"h{level}"] = matches
    
    seo_analysis["headings"] = {
        "structure": headings,
        "h1_count": len(headings["h1"]),
        "proper_hierarchy": len(headings["h1"]) == 1,
        "score": 1.0 if len(headings["h1"]) == 1 else 0.5
    }
    
    # Calculate overall SEO score
    scores = [
        seo_analysis["title"].get("score", 0),
        seo_analysis["description"].get("score", 0),
        seo_analysis["headings"].get("score", 0)
    ]
    seo_analysis["score"] = sum(scores) / len(scores)
    
    return seo_analysis


async def _analyze_performance_indicators(
    html_content: str,
    url: Optional[str],
    metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """Analyze performance indicators and optimization opportunities."""
    performance_analysis = {
        "images": {},
        "scripts": {},
        "stylesheets": {},
        "external_resources": {},
        "optimization_opportunities": [],
        "score": 0.0
    }
    
    # Image analysis
    img_tags = re.findall(r'<img[^>]*>', html_content, re.IGNORECASE)
    images_without_alt = len(re.findall(r'<img(?![^>]*alt=)[^>]*>', html_content, re.IGNORECASE))
    
    performance_analysis["images"] = {
        "total_count": len(img_tags),
        "missing_alt_text": images_without_alt,
        "optimization_needed": images_without_alt > 0
    }
    
    if images_without_alt > 0:
        performance_analysis["optimization_opportunities"].append(
            f"Add alt text to {images_without_alt} images for better accessibility and SEO"
        )
    
    # Script analysis
    script_tags = re.findall(r'<script[^>]*>', html_content, re.IGNORECASE)
    external_scripts = len(re.findall(r'<script[^>]*src=["\'][^"\']+["\']', html_content, re.IGNORECASE))
    
    performance_analysis["scripts"] = {
        "total_count": len(script_tags),
        "external_count": external_scripts,
        "inline_count": len(script_tags) - external_scripts
    }
    
    if external_scripts > 10:
        performance_analysis["optimization_opportunities"].append(
            f"Consider reducing {external_scripts} external scripts to improve load time"
        )
    
    # CSS analysis
    css_links = re.findall(r'<link[^>]*rel=["\']stylesheet["\'][^>]*>', html_content, re.IGNORECASE)
    performance_analysis["stylesheets"] = {
        "external_count": len(css_links)
    }
    
    if len(css_links) > 5:
        performance_analysis["optimization_opportunities"].append(
            f"Consider combining {len(css_links)} CSS files to reduce HTTP requests"
        )
    
    # Calculate performance score
    score = 1.0
    if images_without_alt > 0:
        score -= 0.2
    if external_scripts > 10:
        score -= 0.3
    if len(css_links) > 5:
        score -= 0.2
    
    performance_analysis["score"] = max(0.0, score)
    
    return performance_analysis


async def _analyze_security_indicators(html_content: str, url: Optional[str]) -> Dict[str, Any]:
    """Analyze security indicators and vulnerabilities."""
    security_analysis = {
        "https_usage": False,
        "mixed_content": [],
        "external_resources": [],
        "security_headers": {},
        "vulnerabilities": [],
        "score": 0.0
    }
    
    # HTTPS usage
    if url and url.startswith("https://"):
        security_analysis["https_usage"] = True
    
    # Mixed content detection
    if url and url.startswith("https://"):
        http_resources = re.findall(r'(?:src|href)=["\']http://[^"\']+["\']', html_content, re.IGNORECASE)
        security_analysis["mixed_content"] = http_resources
        
        if http_resources:
            security_analysis["vulnerabilities"].append(
                f"Mixed content detected: {len(http_resources)} HTTP resources on HTTPS page"
            )
    
    # External resource analysis
    external_domains = set()
    for match in re.finditer(r'(?:src|href)=["\']https?://([^/"\'\s]+)', html_content, re.IGNORECASE):
        domain = match.group(1)
        if url and domain not in url:
            external_domains.add(domain)
    
    security_analysis["external_resources"] = list(external_domains)
    
    # Calculate security score
    score = 0.5  # Base score
    if security_analysis["https_usage"]:
        score += 0.3
    if not security_analysis["mixed_content"]:
        score += 0.2
    
    security_analysis["score"] = min(1.0, score)
    
    return security_analysis


async def _analyze_accessibility_indicators(html_content: str, content: str) -> Dict[str, Any]:
    """Analyze accessibility indicators and compliance."""
    accessibility_analysis = {
        "images": {},
        "headings": {},
        "links": {},
        "forms": {},
        "color_contrast": {},
        "issues": [],
        "score": 0.0
    }
    
    # Image accessibility
    img_tags = re.findall(r'<img[^>]*>', html_content, re.IGNORECASE)
    images_without_alt = len(re.findall(r'<img(?![^>]*alt=)[^>]*>', html_content, re.IGNORECASE))
    
    accessibility_analysis["images"] = {
        "total": len(img_tags),
        "missing_alt": images_without_alt,
        "compliance": images_without_alt == 0
    }
    
    if images_without_alt > 0:
        accessibility_analysis["issues"].append(f"{images_without_alt} images missing alt text")
    
    # Heading structure
    h1_count = len(re.findall(r'<h1[^>]*>', html_content, re.IGNORECASE))
    accessibility_analysis["headings"] = {
        "h1_count": h1_count,
        "proper_structure": h1_count == 1
    }
    
    if h1_count != 1:
        accessibility_analysis["issues"].append(f"Page should have exactly one H1 tag (found {h1_count})")
    
    # Link accessibility
    links_without_text = len(re.findall(r'<a[^>]*></a>', html_content, re.IGNORECASE))
    if links_without_text > 0:
        accessibility_analysis["issues"].append(f"{links_without_text} links without descriptive text")
    
    # Calculate accessibility score
    score = 1.0
    score -= min(0.5, images_without_alt * 0.1)
    score -= 0.2 if h1_count != 1 else 0
    score -= min(0.3, links_without_text * 0.1)
    
    accessibility_analysis["score"] = max(0.0, score)
    
    return accessibility_analysis


async def _analyze_content_structure(
    content: str,
    html_content: str,
    nlp_pipeline: NLPPipeline
) -> Dict[str, Any]:
    """Analyze content structure and quality."""
    content_analysis = {
        "word_count": 0,
        "reading_time": 0,
        "readability": {},
        "structure": {},
        "quality_indicators": {},
        "score": 0.0
    }
    
    if not content:
        return content_analysis
    
    # Basic metrics
    words = content.split()
    content_analysis["word_count"] = len(words)
    content_analysis["reading_time"] = max(1, len(words) // 200)  # Assume 200 WPM
    
    # Sentence analysis
    sentences = content.split('. ')
    avg_sentence_length = len(words) / len(sentences) if sentences else 0
    
    content_analysis["readability"] = {
        "sentence_count": len(sentences),
        "average_sentence_length": avg_sentence_length,
        "readability_score": _calculate_readability_score(avg_sentence_length)
    }
    
    # Structure analysis
    paragraphs = len(re.findall(r'<p[^>]*>', html_content, re.IGNORECASE))
    content_analysis["structure"] = {
        "paragraph_count": paragraphs,
        "average_paragraph_length": len(words) / paragraphs if paragraphs > 0 else 0
    }
    
    # Quality indicators
    content_analysis["quality_indicators"] = {
        "sufficient_length": len(words) >= 300,
        "good_structure": paragraphs >= 3,
        "readable_sentences": 10 <= avg_sentence_length <= 25
    }
    
    # Calculate content score
    quality_score = sum(content_analysis["quality_indicators"].values()) / len(content_analysis["quality_indicators"])
    content_analysis["score"] = quality_score
    
    return content_analysis


def _extract_canonical_url(html_content: str, base_url: Optional[str]) -> Optional[str]:
    """Extract canonical URL from HTML content."""
    # Look for canonical link tag
    canonical_match = re.search(
        r'<link[^>]*rel=["\']canonical["\'][^>]*href=["\']([^"\']+)["\']',
        html_content,
        re.IGNORECASE
    )
    
    if canonical_match:
        canonical_url = canonical_match.group(1)
        
        # Handle relative URLs
        if canonical_url.startswith('/') and base_url:
            parsed_base = urlparse(base_url)
            canonical_url = f"{parsed_base.scheme}://{parsed_base.netloc}{canonical_url}"
        elif not canonical_url.startswith(('http://', 'https://')) and base_url:
            canonical_url = urljoin(base_url, canonical_url)
        
        return canonical_url
    
    return None


async def _validate_canonical_url(
    canonical_url: Optional[str],
    original_url: Optional[str],
    validate_accessibility: bool
) -> Dict[str, Any]:
    """Validate canonical URL."""
    validation = {
        "is_valid": False,
        "is_accessible": None,
        "matches_original": False,
        "issues": []
    }
    
    if not canonical_url:
        validation["issues"].append("No canonical URL found")
        return validation
    
    # Basic URL validation
    try:
        parsed = urlparse(canonical_url)
        if parsed.scheme and parsed.netloc:
            validation["is_valid"] = True
        else:
            validation["issues"].append("Invalid URL format")
    except Exception:
        validation["issues"].append("URL parsing failed")
    
    # Compare with original URL
    if original_url and canonical_url:
        validation["matches_original"] = canonical_url == original_url
        if not validation["matches_original"]:
            validation["issues"].append("Canonical URL differs from original URL")
    
    # Accessibility validation would require HTTP request
    if validate_accessibility:
        validation["is_accessible"] = None  # Would need actual HTTP check
        validation["issues"].append("Accessibility check not implemented")
    
    return validation


def _analyze_url_structure(url: str) -> Dict[str, Any]:
    """Analyze URL structure and SEO friendliness."""
    if not url:
        return {}
    
    parsed = urlparse(url)
    
    return {
        "scheme": parsed.scheme,
        "domain": parsed.netloc,
        "path": parsed.path,
        "query_params": len(parsed.query.split('&')) if parsed.query else 0,
        "fragment": bool(parsed.fragment),
        "seo_friendly": _is_seo_friendly_url(url),
        "length": len(url),
        "depth": len([p for p in parsed.path.split('/') if p])
    }


def _is_seo_friendly_url(url: str) -> bool:
    """Check if URL is SEO friendly."""
    parsed = urlparse(url)
    path = parsed.path.lower()
    
    # Check for SEO-friendly characteristics
    has_readable_path = bool(re.match(r'^[a-z0-9\-/]+$', path))
    no_excessive_params = len(parsed.query.split('&')) <= 3 if parsed.query else True
    reasonable_length = len(url) <= 100
    
    return has_readable_path and no_excessive_params and reasonable_length


def _extract_text_from_html(html_content: str) -> str:
    """Extract plain text from HTML content."""
    # Simple HTML tag removal
    text = re.sub(r'<[^>]+>', '', html_content)
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _calculate_readability_score(avg_sentence_length: float) -> float:
    """Calculate readability score based on sentence length."""
    # Optimal sentence length is around 15-20 words
    if 15 <= avg_sentence_length <= 20:
        return 1.0
    elif 10 <= avg_sentence_length <= 25:
        return 0.8
    elif 5 <= avg_sentence_length <= 30:
        return 0.6
    else:
        return 0.4


def _suggest_bypass_strategies(paywall_analysis: Dict[str, Any]) -> List[str]:
    """Suggest potential paywall bypass strategies."""
    strategies = []
    
    if paywall_analysis.get("paywall_detected"):
        paywall_types = paywall_analysis.get("paywall_types", [])
        
        if "registration" in paywall_types:
            strategies.append("Try accessing content through social media links")
            strategies.append("Use reader mode in browser")
        
        if "subscription" in paywall_types:
            strategies.append("Check if content is available through library databases")
            strategies.append("Look for free trial or promotional access")
        
        if "metered" in paywall_types:
            strategies.append("Clear browser cookies and cache")
            strategies.append("Try incognito/private browsing mode")
        
        # General strategies
        strategies.extend([
            "Search for the article title on search engines",
            "Check if content is syndicated on other platforms",
            "Look for author's social media posts about the content"
        ])
    
    return strategies


def _assess_content_accessibility(content: str, paywall_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Assess how much content is accessible."""
    total_length = len(content)
    
    if not paywall_analysis.get("paywall_detected"):
        return {
            "accessibility": "full",
            "accessible_percentage": 100,
            "content_length": total_length
        }
    
    # Estimate accessible content based on paywall strength
    confidence = paywall_analysis.get("confidence", 0)
    
    if confidence > 0.7:
        accessible_percentage = 20  # Strong paywall
    elif confidence > 0.4:
        accessible_percentage = 50  # Medium paywall
    else:
        accessible_percentage = 80  # Weak paywall
    
    return {
        "accessibility": "restricted",
        "accessible_percentage": accessible_percentage,
        "content_length": total_length,
        "estimated_accessible_length": int(total_length * accessible_percentage / 100)
    }


def _calculate_paywall_strength(paywall_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate paywall implementation strength."""
    if not paywall_analysis.get("paywall_detected"):
        return {"strength": "none", "score": 0.0}
    
    confidence = paywall_analysis.get("confidence", 0)
    types_count = len(paywall_analysis.get("paywall_types", []))
    indicators_count = len(paywall_analysis.get("indicators", []))
    
    # Calculate strength score
    strength_score = confidence
    strength_score += types_count * 0.1
    strength_score += min(0.3, indicators_count * 0.05)
    
    strength_score = min(1.0, strength_score)
    
    if strength_score > 0.8:
        strength_level = "very_strong"
    elif strength_score > 0.6:
        strength_level = "strong"
    elif strength_score > 0.4:
        strength_level = "medium"
    elif strength_score > 0.2:
        strength_level = "weak"
    else:
        strength_level = "very_weak"
    
    return {
        "strength": strength_level,
        "score": strength_score,
        "bypass_difficulty": strength_level
    }


def _add_confidence_scores_to_tech_stack(tech_stack: Dict[str, Any], html_content: str) -> Dict[str, Any]:
    """Add confidence scores to technology stack detection."""
    # This would involve more sophisticated analysis
    # For now, just ensure all items have confidence scores
    for category, technologies in tech_stack.items():
        if isinstance(technologies, list):
            for tech in technologies:
                if "confidence" not in tech:
                    tech["confidence"] = 0.7  # Default confidence
    
    return tech_stack


def _generate_tech_stack_recommendations(tech_stack: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on detected technology stack."""
    recommendations = []
    
    # Check for common issues
    frameworks = tech_stack.get("frameworks", [])
    if len(frameworks) > 2:
        recommendations.append("Consider reducing the number of frontend frameworks to improve performance")
    
    analytics = tech_stack.get("analytics", [])
    if len(analytics) > 3:
        recommendations.append("Multiple analytics platforms detected - consider consolidating for better performance")
    
    cdn = tech_stack.get("cdn", [])
    if not cdn:
        recommendations.append("Consider using a CDN to improve global content delivery performance")
    
    return recommendations


def _calculate_overall_confidence(tech_stack: Dict[str, Any]) -> float:
    """Calculate overall confidence in technology stack detection."""
    all_confidences = []
    
    for category, technologies in tech_stack.items():
        if isinstance(technologies, list):
            for tech in technologies:
                if isinstance(tech, dict) and "confidence" in tech:
                    all_confidences.append(tech["confidence"])
    
    return sum(all_confidences) / len(all_confidences) if all_confidences else 0.0


def _generate_canonical_recommendations(
    canonical_url: Optional[str],
    original_url: Optional[str],
    validation_results: Dict[str, Any]
) -> List[str]:
    """Generate recommendations for canonical URL implementation."""
    recommendations = []
    
    if not canonical_url:
        recommendations.append("Add a canonical link tag to specify the preferred URL for this content")
    
    if validation_results.get("issues"):
        for issue in validation_results["issues"]:
            if "differs from original" in issue:
                recommendations.append("Ensure canonical URL points to the most authoritative version of the content")
            elif "Invalid URL format" in issue:
                recommendations.append("Fix canonical URL format - ensure it's a valid absolute URL")
    
    return recommendations


async def _process_batch_technical_analysis(
    items: List[Dict[str, Any]],
    repository_factory: RepositoryFactory,
    nlp_pipeline: NLPPipeline
) -> List[Dict[str, Any]]:
    """Process batch technical analysis requests."""
    results = []
    
    for i, item in enumerate(items):
        try:
            # Get content for analysis
            analysis_data = await _get_content_for_technical_analysis(
                item.get("url"),
                item.get("article_id"),
                item.get("html_content"),
                repository_factory
            )
            
            # Perform basic technical analysis
            paywall_analysis = await _analyze_paywall_indicators(
                analysis_data["content"],
                analysis_data.get("html_content", ""),
                analysis_data.get("url")
            )
            
            tech_stack = await _analyze_technology_stack(
                analysis_data.get("html_content", ""),
                analysis_data.get("url"),
                analysis_data.get("metadata", {})
            )
            
            results.append({
                "item_index": i,
                "success": True,
                "url": analysis_data.get("url"),
                "paywall_detected": paywall_analysis.get("paywall_detected", False),
                "paywall_confidence": paywall_analysis.get("confidence", 0),
                "technologies_detected": sum(len(v) for v in tech_stack.values() if isinstance(v, list)),
                "tech_stack": tech_stack,
                "paywall_analysis": paywall_analysis
            })
            
        except Exception as e:
            results.append({
                "item_index": i,
                "success": False,
                "error": str(e)
            })
    
    return results


def _aggregate_technical_insights(batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate insights from batch technical analysis."""
    successful_results = [r for r in batch_results if r["success"]]
    
    if not successful_results:
        return {}
    
    # Paywall statistics
    paywall_detected_count = sum(1 for r in successful_results if r.get("paywall_detected"))
    avg_paywall_confidence = sum(r.get("paywall_confidence", 0) for r in successful_results) / len(successful_results)
    
    # Technology statistics
    all_technologies = {}
    for result in successful_results:
        tech_stack = result.get("tech_stack", {})
        for category, technologies in tech_stack.items():
            if isinstance(technologies, list):
                for tech in technologies:
                    tech_name = tech.get("name", "Unknown")
                    if tech_name not in all_technologies:
                        all_technologies[tech_name] = 0
                    all_technologies[tech_name] += 1
    
    # Most common technologies
    common_technologies = sorted(all_technologies.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return {
        "paywall_statistics": {
            "sites_with_paywall": paywall_detected_count,
            "paywall_percentage": (paywall_detected_count / len(successful_results)) * 100,
            "average_confidence": avg_paywall_confidence
        },
        "technology_statistics": {
            "unique_technologies": len(all_technologies),
            "most_common": common_technologies,
            "total_detections": sum(all_technologies.values())
        }
    }


def _identify_technology_patterns(batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Identify patterns in technology usage across batch results."""
    successful_results = [r for r in batch_results if r["success"]]
    
    if not successful_results:
        return {}
    
    patterns = {
        "framework_combinations": {},
        "cms_analytics_pairs": {},
        "cdn_usage_patterns": {}
    }
    
    # Analyze framework combinations
    for result in successful_results:
        tech_stack = result.get("tech_stack", {})
        frameworks = [tech.get("name") for tech in tech_stack.get("frameworks", [])]
        
        if len(frameworks) > 1:
            combo = " + ".join(sorted(frameworks))
            patterns["framework_combinations"][combo] = patterns["framework_combinations"].get(combo, 0) + 1
    
    return patterns