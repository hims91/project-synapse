"""
Neurons - Lightweight Scrapers
Layer 1: Perception Layer

This module implements lightweight scrapers that use cached recipes
for efficient content extraction with >99% success rates.

The Neurons layer provides:
- Recipe-based scraping engine
- Recipe validation and testing framework
- Content extraction with CSS selectors
- Success rate tracking and optimization
"""

from .recipe_engine import ScrapingRecipeEngine, recipe_engine
from .recipe_tester import RecipeTester, create_recipe_tester
from .http_scraper import HTTPScraper, http_scraper

__all__ = [
    'ScrapingRecipeEngine',
    'recipe_engine',
    'RecipeTester', 
    'create_recipe_tester',
    'HTTPScraper',
    'http_scraper'
]