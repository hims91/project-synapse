"""
Neurons - Scraping Recipe Engine
Layer 1: Perception Layer

This module implements the recipe-based scraping engine for lightweight scrapers.
It provides recipe caching, validation, and success rate tracking.

The recipe engine serves as the core intelligence for content extraction,
using cached CSS selectors and actions to achieve >99% scrape success rates.
"""
import asyncio
import logging
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from urllib.parse import urlparse

from bs4 import BeautifulSoup, Tag
from sqlalchemy.ext.asyncio import AsyncSession

from ..shared.schemas import (
    ScrapingRecipeResponse, ScrapingRecipeCreate, ScrapingRecipeUpdate,
    ScrapingSelectors, ScrapingAction, ArticleCreate, ArticleNLPData, ArticleMetadata
)
from ..synaptic_vesicle.repositories import ScrapingRecipeRepository
from ..synaptic_vesicle.database import get_db_session

logger = logging.getLogger(__name__)


class RecipeValidationError(Exception):
    """Raised when recipe validation fails."""
    pass


class RecipeExecutionError(Exception):
    """Raised when recipe execution fails."""
    pass


class ScrapingRecipeEngine:
    """
    Core recipe engine for content extraction.
    
    Manages recipe caching, validation, and execution with success tracking.
    Provides the intelligence layer for lightweight scrapers.
    """
    
    def __init__(self):
        self.recipe_cache: Dict[str, ScrapingRecipeResponse] = {}
        self.cache_ttl = timedelta(hours=1)  # Cache recipes for 1 hour
        self.last_cache_update: Dict[str, datetime] = {}
        
    async def get_recipe(self, domain: str) -> Optional[ScrapingRecipeResponse]:
        """
        Get scraping recipe for a domain with caching.
        
        Args:
            domain: Target domain (e.g., 'example.com')
            
        Returns:
            ScrapingRecipeResponse if found, None otherwise
        """
        try:
            # Check cache first
            if self._is_cache_valid(domain):
                logger.debug(f"Using cached recipe for domain: {domain}")
                return self.recipe_cache[domain]
            
            # Fetch from database
            async with get_db_session() as session:
                repo = ScrapingRecipeRepository(session)
                recipe = await repo.get_by_domain(domain)
                
                if recipe:
                    # Update cache
                    self.recipe_cache[domain] = recipe
                    self.last_cache_update[domain] = datetime.utcnow()
                    logger.info(f"Loaded recipe for domain: {domain}")
                    return recipe
                    
            logger.debug(f"No recipe found for domain: {domain}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting recipe for domain {domain}: {e}")
            return None
    
    async def create_recipe(
        self, 
        domain: str, 
        selectors: ScrapingSelectors,
        actions: Optional[List[ScrapingAction]] = None,
        created_by: str = "learning"
    ) -> ScrapingRecipeResponse:
        """
        Create a new scraping recipe.
        
        Args:
            domain: Target domain
            selectors: CSS selectors for content extraction
            actions: Optional scraping actions
            created_by: Creation method ('learning' or 'manual')
            
        Returns:
            Created recipe
            
        Raises:
            RecipeValidationError: If recipe validation fails
        """
        try:
            # Validate recipe
            self._validate_recipe(selectors, actions or [])
            
            # Create recipe
            recipe_data = ScrapingRecipeCreate(
                domain=domain,
                selectors=selectors,
                actions=actions or [],
                created_by=created_by
            )
            
            async with get_db_session() as session:
                repo = ScrapingRecipeRepository(session)
                recipe = await repo.create(recipe_data)
                await session.commit()
                
                # Update cache
                self.recipe_cache[domain] = recipe
                self.last_cache_update[domain] = datetime.utcnow()
                
                logger.info(f"Created recipe for domain: {domain}")
                return recipe
                
        except Exception as e:
            logger.error(f"Error creating recipe for domain {domain}: {e}")
            raise RecipeValidationError(f"Failed to create recipe: {e}")
    
    async def update_recipe_success(
        self, 
        domain: str, 
        success: bool
    ) -> None:
        """
        Update recipe success rate based on scraping result.
        
        Args:
            domain: Target domain
            success: Whether the scraping was successful
        """
        try:
            async with get_db_session() as session:
                repo = ScrapingRecipeRepository(session)
                recipe = await repo.get_by_domain(domain)
                
                if not recipe:
                    logger.warning(f"No recipe found to update for domain: {domain}")
                    return
                
                # Calculate new success rate
                total_attempts = recipe.usage_count + 1
                successful_attempts = int(recipe.success_rate * recipe.usage_count)
                
                if success:
                    successful_attempts += 1
                
                new_success_rate = successful_attempts / total_attempts
                
                # Update recipe
                update_data = ScrapingRecipeUpdate(
                    success_rate=new_success_rate,
                    usage_count=total_attempts
                )
                
                updated_recipe = await repo.update(recipe.id, update_data)
                await session.commit()
                
                # Update cache
                if updated_recipe:
                    self.recipe_cache[domain] = updated_recipe
                    self.last_cache_update[domain] = datetime.utcnow()
                
                logger.info(
                    f"Updated recipe for {domain}: "
                    f"success_rate={new_success_rate:.3f}, "
                    f"usage_count={total_attempts}"
                )
                
        except Exception as e:
            logger.error(f"Error updating recipe success for domain {domain}: {e}")
    
    async def execute_recipe(
        self, 
        recipe: ScrapingRecipeResponse, 
        html_content: str,
        url: str
    ) -> ArticleCreate:
        """
        Execute a scraping recipe on HTML content.
        
        Args:
            recipe: Scraping recipe to execute
            html_content: HTML content to parse
            url: Source URL
            
        Returns:
            Extracted article data
            
        Raises:
            RecipeExecutionError: If recipe execution fails
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            domain = urlparse(url).netloc
            
            # Extract content using selectors
            extracted_data = {}
            
            # Title (required)
            title_element = soup.select_one(recipe.selectors.title)
            if not title_element:
                raise RecipeExecutionError(f"Title selector '{recipe.selectors.title}' found no elements")
            extracted_data['title'] = self._clean_text(title_element.get_text())
            
            # Content (required)
            content_element = soup.select_one(recipe.selectors.content)
            if not content_element:
                raise RecipeExecutionError(f"Content selector '{recipe.selectors.content}' found no elements")
            extracted_data['content'] = self._clean_text(content_element.get_text())
            
            # Author (optional)
            if recipe.selectors.author:
                author_element = soup.select_one(recipe.selectors.author)
                if author_element:
                    extracted_data['author'] = self._clean_text(author_element.get_text())
            
            # Publish date (optional)
            if recipe.selectors.publish_date:
                date_element = soup.select_one(recipe.selectors.publish_date)
                if date_element:
                    date_text = self._clean_text(date_element.get_text())
                    extracted_data['published_at'] = self._parse_date(date_text)
            
            # Summary (optional)
            if recipe.selectors.summary:
                summary_element = soup.select_one(recipe.selectors.summary)
                if summary_element:
                    extracted_data['summary'] = self._clean_text(summary_element.get_text())
            
            # Create article data
            article_data = ArticleCreate(
                url=url,
                title=extracted_data['title'],
                content=extracted_data.get('content'),
                summary=extracted_data.get('summary'),
                author=extracted_data.get('author'),
                published_at=extracted_data.get('published_at'),
                source_domain=domain,
                nlp_data=ArticleNLPData(),
                page_metadata=ArticleMetadata()
            )
            
            logger.info(f"Successfully executed recipe for {domain}")
            return article_data
            
        except Exception as e:
            logger.error(f"Error executing recipe for {url}: {e}")
            raise RecipeExecutionError(f"Recipe execution failed: {e}")
    
    def validate_recipe(
        self, 
        selectors: ScrapingSelectors, 
        actions: List[ScrapingAction]
    ) -> Tuple[bool, List[str]]:
        """
        Validate a scraping recipe.
        
        Args:
            selectors: CSS selectors to validate
            actions: Scraping actions to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        try:
            self._validate_recipe(selectors, actions)
            return True, []
        except RecipeValidationError as e:
            return False, [str(e)]
    
    def _is_cache_valid(self, domain: str) -> bool:
        """Check if cached recipe is still valid."""
        if domain not in self.recipe_cache:
            return False
        
        if domain not in self.last_cache_update:
            return False
        
        return datetime.utcnow() - self.last_cache_update[domain] < self.cache_ttl
    
    def _validate_recipe(
        self, 
        selectors: ScrapingSelectors, 
        actions: List[ScrapingAction]
    ) -> None:
        """
        Validate recipe components.
        
        Raises:
            RecipeValidationError: If validation fails
        """
        # Validate required selectors
        if not selectors.title or not selectors.title.strip():
            raise RecipeValidationError("Title selector is required")
        
        if not selectors.content or not selectors.content.strip():
            raise RecipeValidationError("Content selector is required")
        
        # Validate CSS selector syntax (basic check)
        for field_name, selector in [
            ('title', selectors.title),
            ('content', selectors.content),
            ('author', selectors.author),
            ('publish_date', selectors.publish_date),
            ('summary', selectors.summary)
        ]:
            if selector and not self._is_valid_css_selector(selector):
                raise RecipeValidationError(f"Invalid CSS selector for {field_name}: {selector}")
        
        # Validate actions
        for i, action in enumerate(actions):
            if not action.type:
                raise RecipeValidationError(f"Action {i}: type is required")
            
            if action.type not in ['click', 'wait', 'scroll', 'input', 'hover']:
                raise RecipeValidationError(f"Action {i}: invalid type '{action.type}'")
            
            if action.timeout and (action.timeout < 1 or action.timeout > 60):
                raise RecipeValidationError(f"Action {i}: timeout must be between 1 and 60 seconds")
    
    def _is_valid_css_selector(self, selector: str) -> bool:
        """Basic CSS selector validation."""
        try:
            # Try to parse with BeautifulSoup
            soup = BeautifulSoup("<div></div>", 'html.parser')
            soup.select(selector)
            return True
        except Exception:
            return False
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        cleaned = " ".join(text.split())
        return cleaned.strip()
    
    def _parse_date(self, date_text: str) -> Optional[datetime]:
        """Parse date from text (basic implementation)."""
        # This is a simplified implementation
        # In production, you'd want more robust date parsing
        try:
            from dateutil import parser
            return parser.parse(date_text)
        except Exception:
            logger.warning(f"Could not parse date: {date_text}")
            return None


# Global recipe engine instance
recipe_engine = ScrapingRecipeEngine()