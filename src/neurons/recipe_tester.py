"""
Neurons - Recipe Testing Framework
Layer 1: Perception Layer

This module provides testing and validation capabilities for scraping recipes.
It includes recipe testing against live URLs and validation frameworks.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from ..shared.schemas import ScrapingRecipeResponse, ScrapingSelectors, ScrapingAction
from .recipe_engine import ScrapingRecipeEngine, RecipeExecutionError

logger = logging.getLogger(__name__)


@dataclass
class RecipeTestResult:
    """Result of recipe testing."""
    success: bool
    extracted_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time_ms: Optional[int] = None
    selectors_found: Dict[str, bool] = None
    content_quality_score: Optional[float] = None


@dataclass
class RecipeValidationResult:
    """Result of recipe validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]


class RecipeTester:
    """
    Testing framework for scraping recipes.
    
    Provides comprehensive testing capabilities including:
    - Live URL testing
    - Recipe validation
    - Content quality assessment
    - Performance measurement
    """
    
    def __init__(self, recipe_engine: ScrapingRecipeEngine):
        self.recipe_engine = recipe_engine
        self.http_client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        )
    
    async def test_recipe_on_url(
        self, 
        recipe: ScrapingRecipeResponse, 
        test_url: str
    ) -> RecipeTestResult:
        """
        Test a recipe against a live URL.
        
        Args:
            recipe: Recipe to test
            test_url: URL to test against
            
        Returns:
            Test result with success status and extracted data
        """
        start_time = datetime.utcnow()
        
        try:
            # Fetch the webpage
            response = await self.http_client.get(test_url)
            response.raise_for_status()
            
            html_content = response.text
            
            # Test recipe execution
            try:
                article_data = await self.recipe_engine.execute_recipe(
                    recipe, html_content, test_url
                )
                
                execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                
                # Analyze selector effectiveness
                selectors_found = await self._analyze_selectors(
                    recipe.selectors, html_content
                )
                
                # Calculate content quality score
                quality_score = self._calculate_content_quality(article_data)
                
                return RecipeTestResult(
                    success=True,
                    extracted_data={
                        'title': article_data.title,
                        'content': article_data.content,
                        'author': article_data.author,
                        'published_at': article_data.published_at.isoformat() if article_data.published_at else None,
                        'summary': article_data.summary,
                        'source_domain': article_data.source_domain
                    },
                    execution_time_ms=execution_time,
                    selectors_found=selectors_found,
                    content_quality_score=quality_score
                )
                
            except RecipeExecutionError as e:
                execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                
                return RecipeTestResult(
                    success=False,
                    error_message=str(e),
                    execution_time_ms=execution_time,
                    selectors_found=await self._analyze_selectors(
                        recipe.selectors, html_content
                    )
                )
                
        except httpx.RequestError as e:
            return RecipeTestResult(
                success=False,
                error_message=f"HTTP request failed: {e}"
            )
        except Exception as e:
            return RecipeTestResult(
                success=False,
                error_message=f"Unexpected error: {e}"
            )
    
    async def validate_recipe(
        self, 
        selectors: ScrapingSelectors,
        actions: List[ScrapingAction],
        test_urls: Optional[List[str]] = None
    ) -> RecipeValidationResult:
        """
        Comprehensive recipe validation.
        
        Args:
            selectors: CSS selectors to validate
            actions: Scraping actions to validate
            test_urls: Optional URLs to test against
            
        Returns:
            Validation result with errors, warnings, and suggestions
        """
        errors = []
        warnings = []
        suggestions = []
        
        # Basic validation
        is_valid, validation_errors = self.recipe_engine.validate_recipe(selectors, actions)
        errors.extend(validation_errors)
        
        # Advanced selector analysis
        selector_analysis = self._analyze_selector_patterns(selectors)
        warnings.extend(selector_analysis['warnings'])
        suggestions.extend(selector_analysis['suggestions'])
        
        # Action validation
        action_analysis = self._analyze_actions(actions)
        warnings.extend(action_analysis['warnings'])
        suggestions.extend(action_analysis['suggestions'])
        
        # Test against URLs if provided
        if test_urls and is_valid:
            for url in test_urls[:3]:  # Limit to 3 test URLs
                try:
                    # Create a temporary recipe for testing
                    temp_recipe = ScrapingRecipeResponse(
                        id="test",
                        domain=urlparse(url).netloc,
                        selectors=selectors,
                        actions=actions,
                        success_rate=0.0,
                        usage_count=0,
                        last_updated=datetime.utcnow(),
                        created_by="manual"
                    )
                    
                    test_result = await self.test_recipe_on_url(temp_recipe, url)
                    
                    if not test_result.success:
                        warnings.append(f"Recipe failed on test URL {url}: {test_result.error_message}")
                    elif test_result.content_quality_score and test_result.content_quality_score < 0.5:
                        warnings.append(f"Low content quality score ({test_result.content_quality_score:.2f}) on {url}")
                        
                except Exception as e:
                    warnings.append(f"Could not test URL {url}: {e}")
        
        return RecipeValidationResult(
            is_valid=is_valid and len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    async def benchmark_recipe(
        self, 
        recipe: ScrapingRecipeResponse, 
        test_urls: List[str],
        iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Benchmark recipe performance across multiple URLs.
        
        Args:
            recipe: Recipe to benchmark
            test_urls: URLs to test against
            iterations: Number of iterations per URL
            
        Returns:
            Benchmark results with performance metrics
        """
        results = {
            'total_tests': len(test_urls) * iterations,
            'successful_tests': 0,
            'failed_tests': 0,
            'average_execution_time_ms': 0,
            'success_rate': 0.0,
            'url_results': {},
            'error_summary': {}
        }
        
        total_execution_time = 0
        
        for url in test_urls:
            url_results = {
                'success_count': 0,
                'fail_count': 0,
                'execution_times': [],
                'errors': []
            }
            
            for i in range(iterations):
                test_result = await self.test_recipe_on_url(recipe, url)
                
                if test_result.success:
                    results['successful_tests'] += 1
                    url_results['success_count'] += 1
                else:
                    results['failed_tests'] += 1
                    url_results['fail_count'] += 1
                    if test_result.error_message:
                        url_results['errors'].append(test_result.error_message)
                
                if test_result.execution_time_ms:
                    url_results['execution_times'].append(test_result.execution_time_ms)
                    total_execution_time += test_result.execution_time_ms
                
                # Small delay between iterations
                await asyncio.sleep(0.1)
            
            results['url_results'][url] = url_results
        
        # Calculate averages
        if results['total_tests'] > 0:
            results['success_rate'] = results['successful_tests'] / results['total_tests']
            
        if total_execution_time > 0:
            results['average_execution_time_ms'] = total_execution_time / results['total_tests']
        
        # Summarize errors
        error_counts = {}
        for url_result in results['url_results'].values():
            for error in url_result['errors']:
                error_counts[error] = error_counts.get(error, 0) + 1
        
        results['error_summary'] = dict(sorted(error_counts.items(), key=lambda x: x[1], reverse=True))
        
        return results
    
    async def _analyze_selectors(
        self, 
        selectors: ScrapingSelectors, 
        html_content: str
    ) -> Dict[str, bool]:
        """Analyze which selectors find elements in the HTML."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        results = {}
        
        for field_name, selector in [
            ('title', selectors.title),
            ('content', selectors.content),
            ('author', selectors.author),
            ('publish_date', selectors.publish_date),
            ('summary', selectors.summary)
        ]:
            if selector:
                try:
                    elements = soup.select(selector)
                    results[field_name] = len(elements) > 0
                except Exception:
                    results[field_name] = False
            else:
                results[field_name] = None  # Optional selector not provided
        
        return results
    
    def _analyze_selector_patterns(self, selectors: ScrapingSelectors) -> Dict[str, List[str]]:
        """Analyze selector patterns for potential issues."""
        warnings = []
        suggestions = []
        
        # Check for overly specific selectors
        for field_name, selector in [
            ('title', selectors.title),
            ('content', selectors.content),
            ('author', selectors.author),
            ('publish_date', selectors.publish_date),
            ('summary', selectors.summary)
        ]:
            if not selector:
                continue
                
            # Count selector complexity
            if selector.count(' ') > 5:
                warnings.append(f"{field_name} selector is very specific and may be fragile")
                suggestions.append(f"Consider simplifying {field_name} selector")
            
            # Check for ID-based selectors (can be fragile)
            if '#' in selector and selector.count('#') > 1:
                warnings.append(f"{field_name} selector uses multiple IDs")
            
            # Check for index-based selectors
            if ':nth-child(' in selector or ':nth-of-type(' in selector:
                warnings.append(f"{field_name} selector uses position-based selection (fragile)")
                suggestions.append(f"Consider using class or semantic selectors for {field_name}")
        
        return {'warnings': warnings, 'suggestions': suggestions}
    
    def _analyze_actions(self, actions: List[ScrapingAction]) -> Dict[str, List[str]]:
        """Analyze scraping actions for potential issues."""
        warnings = []
        suggestions = []
        
        if len(actions) > 10:
            warnings.append("Recipe has many actions, which may slow down scraping")
            suggestions.append("Consider optimizing the action sequence")
        
        # Check for long waits
        for i, action in enumerate(actions):
            if action.type == 'wait' and action.timeout and action.timeout > 10:
                warnings.append(f"Action {i} has a long wait time ({action.timeout}s)")
                suggestions.append(f"Consider reducing wait time for action {i}")
        
        return {'warnings': warnings, 'suggestions': suggestions}
    
    def _calculate_content_quality(self, article_data) -> float:
        """Calculate content quality score (0.0 to 1.0)."""
        score = 0.0
        
        # Title quality (0.3 weight)
        if article_data.title:
            title_len = len(article_data.title.strip())
            if 10 <= title_len <= 200:
                score += 0.3
            elif title_len > 0:
                score += 0.15
        
        # Content quality (0.5 weight)
        if article_data.content:
            content_len = len(article_data.content.strip())
            if content_len >= 500:
                score += 0.5
            elif content_len >= 100:
                score += 0.3
            elif content_len > 0:
                score += 0.1
        
        # Author presence (0.1 weight)
        if article_data.author and article_data.author.strip():
            score += 0.1
        
        # Published date presence (0.1 weight)
        if article_data.published_at:
            score += 0.1
        
        return min(score, 1.0)
    
    async def close(self):
        """Close HTTP client."""
        await self.http_client.aclose()


# Factory function for creating recipe tester
def create_recipe_tester(recipe_engine: ScrapingRecipeEngine) -> RecipeTester:
    """Create a recipe tester instance."""
    return RecipeTester(recipe_engine)