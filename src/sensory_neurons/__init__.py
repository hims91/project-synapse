"""
Sensory Neurons - Learning Scrapers
Layer 1: Perception Layer

This module implements advanced learning scrapers with browser automation,
machine learning-based pattern recognition, and proxy network capabilities.

The Sensory Neurons layer provides:
- Browser automation with Playwright for JavaScript-heavy sites
- ML-based recipe learning and pattern recognition
- Proxy rotation and anonymization networks
- Advanced anti-bot detection evasion
"""

from .playwright_scraper import PlaywrightScraper, playwright_scraper
from .recipe_learner import RecipeLearner, recipe_learner
from .chameleon_network import ChameleonNetwork, chameleon_network

__all__ = [
    'PlaywrightScraper',
    'playwright_scraper',
    'RecipeLearner',
    'recipe_learner',
    'ChameleonNetwork',
    'chameleon_network'
]