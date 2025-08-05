"""
Sensory Neurons - Recipe Learning System
Layer 1: Perception Layer

This module implements ML-based pattern recognition for automatic recipe generation
and optimization. It learns from successful scrapes to improve extraction accuracy.
"""
import asyncio
import logging
import json
import pickle
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup, Tag

from ..shared.schemas import (
    ScrapingRecipeCreate, ScrapingSelectors, ScrapingAction,
    ArticleCreate, RecipeCreatedBy
)
from ..neurons.recipe_engine import ScrapingRecipeEngine, recipe_engine
from ..synaptic_vesicle.database import get_db_session
from ..synaptic_vesicle.repositories import ScrapingRecipeRepository

logger = logging.getLogger(__name__)


@dataclass
class ExtractionPattern:
    """Represents a learned extraction pattern."""
    selector: str
    confidence: float
    frequency: int
    domains: List[str]
    success_rate: float
    content_type: str  # 'title', 'content', 'author', etc.


@dataclass
class LearningExample:
    """Training example for recipe learning."""
    url: str
    domain: str
    html_content: str
    extracted_data: Dict[str, str]
    selectors_used: Dict[str, str]
    success: bool
    timestamp: datetime


class RecipeLearner:
    """
    ML-based recipe learning system.
    
    Learns extraction patterns from successful scrapes and generates
    optimized recipes for improved content extraction accuracy.
    """
    
    def __init__(
        self,
        recipe_engine: Optional[ScrapingRecipeEngine] = None,
        model_dir: str = "models/recipe_learning",
        min_examples: int = 5,
        confidence_threshold: float = 0.7
    ):
        """
        Initialize recipe learner.
        
        Args:
            recipe_engine: Recipe engine for recipe management
            model_dir: Directory to save learned models
            min_examples: Minimum examples needed to generate recipe
            confidence_threshold: Minimum confidence for pattern acceptance
        """
        self.recipe_engine = recipe_engine or recipe_engine
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.min_examples = min_examples
        self.confidence_threshold = confidence_threshold
        
        # Learning data storage
        self.examples: Dict[str, List[LearningExample]] = defaultdict(list)
        self.patterns: Dict[str, List[ExtractionPattern]] = defaultdict(list)
        
        # ML models
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.clustering_model = DBSCAN(eps=0.3, min_samples=2)
        
        # Pattern recognition cache
        self.pattern_cache: Dict[str, List[str]] = {}
        self.cache_ttl = timedelta(hours=1)
        self.last_cache_update: Dict[str, datetime] = {}
    
    async def learn_from_scrape(
        self,
        url: str,
        html_content: str,
        extracted_data: Dict[str, str],
        selectors_used: Dict[str, str],
        success: bool
    ):
        """
        Learn from a scraping attempt.
        
        Args:
            url: Scraped URL
            html_content: HTML content
            extracted_data: Successfully extracted data
            selectors_used: Selectors that were used
            success: Whether the scrape was successful
        """
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            
            # Create learning example
            example = LearningExample(
                url=url,
                domain=domain,
                html_content=html_content,
                extracted_data=extracted_data,
                selectors_used=selectors_used,
                success=success,
                timestamp=datetime.utcnow()
            )
            
            # Store example
            self.examples[domain].append(example)
            
            # Analyze patterns if we have enough examples
            if len(self.examples[domain]) >= self.min_examples:
                await self._analyze_domain_patterns(domain)
            
            # Generate recipe if patterns are strong enough
            if len(self.examples[domain]) >= self.min_examples * 2:
                await self._generate_recipe_for_domain(domain)
            
            logger.info(f"Learned from scrape: {url} (success: {success})")
            
        except Exception as e:
            logger.error(f"Error learning from scrape {url}: {e}")
    
    async def generate_recipe(self, domain: str) -> Optional[ScrapingRecipeCreate]:
        """
        Generate a recipe for a domain based on learned patterns.
        
        Args:
            domain: Target domain
            
        Returns:
            Generated recipe or None if insufficient data
        """
        try:
            if domain not in self.examples or len(self.examples[domain]) < self.min_examples:
                logger.info(f"Insufficient examples for {domain}: {len(self.examples.get(domain, []))}")
                return None
            
            # Analyze patterns for the domain
            await self._analyze_domain_patterns(domain)
            
            if domain not in self.patterns:
                logger.info(f"No patterns found for {domain}")
                return None
            
            # Generate selectors from patterns
            selectors = await self._generate_selectors_from_patterns(domain)
            
            if not selectors:
                logger.info(f"Could not generate selectors for {domain}")
                return None
            
            # Generate actions if needed
            actions = await self._generate_actions_from_patterns(domain)
            
            recipe = ScrapingRecipeCreate(
                domain=domain,
                selectors=selectors,
                actions=actions,
                created_by=RecipeCreatedBy.LEARNING
            )
            
            logger.info(f"Generated recipe for {domain}")
            return recipe
            
        except Exception as e:
            logger.error(f"Error generating recipe for {domain}: {e}")
            return None
    
    async def optimize_existing_recipe(
        self,
        domain: str,
        current_success_rate: float
    ) -> Optional[ScrapingRecipeCreate]:
        """
        Optimize an existing recipe based on recent learning data.
        
        Args:
            domain: Target domain
            current_success_rate: Current recipe success rate
            
        Returns:
            Optimized recipe or None if no improvement possible
        """
        try:
            if domain not in self.examples:
                return None
            
            # Filter recent examples
            recent_cutoff = datetime.utcnow() - timedelta(days=7)
            recent_examples = [
                ex for ex in self.examples[domain]
                if ex.timestamp > recent_cutoff
            ]
            
            if len(recent_examples) < self.min_examples:
                return None
            
            # Analyze recent patterns
            await self._analyze_domain_patterns(domain, examples=recent_examples)
            
            # Generate improved selectors
            new_selectors = await self._generate_selectors_from_patterns(domain)
            
            if not new_selectors:
                return None
            
            # Estimate improvement potential
            estimated_success_rate = await self._estimate_recipe_success_rate(
                domain, new_selectors
            )
            
            # Only return if significant improvement expected
            if estimated_success_rate > current_success_rate + 0.05:  # 5% improvement
                actions = await self._generate_actions_from_patterns(domain)
                
                optimized_recipe = ScrapingRecipeCreate(
                    domain=domain,
                    selectors=new_selectors,
                    actions=actions,
                    created_by=RecipeCreatedBy.LEARNING
                )
                
                logger.info(
                    f"Optimized recipe for {domain}: "
                    f"{current_success_rate:.3f} -> {estimated_success_rate:.3f}"
                )
                return optimized_recipe
            
            return None
            
        except Exception as e:
            logger.error(f"Error optimizing recipe for {domain}: {e}")
            return None
    
    async def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning system statistics."""
        try:
            total_examples = sum(len(examples) for examples in self.examples.values())
            total_patterns = sum(len(patterns) for patterns in self.patterns.values())
            
            domain_stats = {}
            for domain, examples in self.examples.items():
                successful = sum(1 for ex in examples if ex.success)
                domain_stats[domain] = {
                    'total_examples': len(examples),
                    'successful_examples': successful,
                    'success_rate': successful / len(examples) if examples else 0,
                    'patterns_learned': len(self.patterns.get(domain, [])),
                    'last_example': max(ex.timestamp for ex in examples).isoformat() if examples else None
                }
            
            return {
                'total_examples': total_examples,
                'total_patterns': total_patterns,
                'domains_learned': len(self.examples),
                'domain_stats': domain_stats,
                'model_info': {
                    'vectorizer_features': getattr(self.vectorizer, 'n_features_in_', 0),
                    'clustering_eps': self.clustering_model.eps,
                    'min_examples': self.min_examples,
                    'confidence_threshold': self.confidence_threshold
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting learning stats: {e}")
            return {}
    
    async def save_models(self):
        """Save learned models to disk."""
        try:
            # Save examples
            examples_file = self.model_dir / "examples.pkl"
            with open(examples_file, 'wb') as f:
                pickle.dump(dict(self.examples), f)
            
            # Save patterns
            patterns_file = self.model_dir / "patterns.pkl"
            with open(patterns_file, 'wb') as f:
                pickle.dump(dict(self.patterns), f)
            
            # Save vectorizer if fitted
            if hasattr(self.vectorizer, 'vocabulary_'):
                vectorizer_file = self.model_dir / "vectorizer.pkl"
                with open(vectorizer_file, 'wb') as f:
                    pickle.dump(self.vectorizer, f)
            
            logger.info(f"Saved learning models to {self.model_dir}")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    async def load_models(self):
        """Load learned models from disk."""
        try:
            # Load examples
            examples_file = self.model_dir / "examples.pkl"
            if examples_file.exists():
                with open(examples_file, 'rb') as f:
                    loaded_examples = pickle.load(f)
                    self.examples = defaultdict(list, loaded_examples)
            
            # Load patterns
            patterns_file = self.model_dir / "patterns.pkl"
            if patterns_file.exists():
                with open(patterns_file, 'rb') as f:
                    loaded_patterns = pickle.load(f)
                    self.patterns = defaultdict(list, loaded_patterns)
            
            # Load vectorizer
            vectorizer_file = self.model_dir / "vectorizer.pkl"
            if vectorizer_file.exists():
                with open(vectorizer_file, 'rb') as f:
                    self.vectorizer = pickle.load(f)
            
            logger.info(f"Loaded learning models from {self.model_dir}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    async def _analyze_domain_patterns(
        self, 
        domain: str, 
        examples: Optional[List[LearningExample]] = None
    ):
        """Analyze patterns for a specific domain."""
        try:
            if examples is None:
                examples = self.examples[domain]
            
            # Filter successful examples
            successful_examples = [ex for ex in examples if ex.success]
            
            if len(successful_examples) < 2:
                return
            
            # Analyze selector patterns for each content type
            content_types = ['title', 'content', 'author', 'publish_date', 'summary']
            
            for content_type in content_types:
                patterns = await self._find_selector_patterns(
                    successful_examples, content_type
                )
                
                if patterns:
                    self.patterns[domain].extend(patterns)
            
            # Remove duplicate patterns
            self.patterns[domain] = self._deduplicate_patterns(self.patterns[domain])
            
            logger.debug(f"Analyzed patterns for {domain}: {len(self.patterns[domain])} patterns")
            
        except Exception as e:
            logger.error(f"Error analyzing patterns for {domain}: {e}")
    
    async def _find_selector_patterns(
        self, 
        examples: List[LearningExample], 
        content_type: str
    ) -> List[ExtractionPattern]:
        """Find selector patterns for a specific content type."""
        try:
            # Collect selectors used for this content type
            selectors = []
            for example in examples:
                if content_type in example.selectors_used:
                    selectors.append(example.selectors_used[content_type])
            
            if not selectors:
                return []
            
            # Count selector frequency
            selector_counts = Counter(selectors)
            
            # Analyze HTML structure patterns
            html_patterns = await self._analyze_html_patterns(examples, content_type)
            
            patterns = []
            for selector, frequency in selector_counts.items():
                confidence = frequency / len(examples)
                
                if confidence >= self.confidence_threshold:
                    pattern = ExtractionPattern(
                        selector=selector,
                        confidence=confidence,
                        frequency=frequency,
                        domains=[examples[0].domain],
                        success_rate=confidence,
                        content_type=content_type
                    )
                    patterns.append(pattern)
            
            # Add HTML structure-based patterns
            patterns.extend(html_patterns)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error finding selector patterns for {content_type}: {e}")
            return []
    
    async def _analyze_html_patterns(
        self, 
        examples: List[LearningExample], 
        content_type: str
    ) -> List[ExtractionPattern]:
        """Analyze HTML structure patterns using ML."""
        try:
            # Extract HTML features for the content type
            features = []
            selectors = []
            
            for example in examples:
                if content_type in example.extracted_data:
                    soup = BeautifulSoup(example.html_content, 'html.parser')
                    
                    # Find elements containing the extracted content
                    target_text = example.extracted_data[content_type]
                    matching_elements = self._find_matching_elements(soup, target_text)
                    
                    for element in matching_elements:
                        # Extract features from the element
                        element_features = self._extract_element_features(element)
                        features.append(element_features)
                        
                        # Generate selector for the element
                        selector = self._generate_element_selector(element)
                        selectors.append(selector)
            
            if len(features) < 2:
                return []
            
            # Vectorize features
            try:
                feature_vectors = self.vectorizer.fit_transform(features)
            except ValueError:
                # Not enough features to vectorize
                return []
            
            # Cluster similar patterns
            clusters = self.clustering_model.fit_predict(feature_vectors)
            
            # Generate patterns from clusters
            patterns = []
            for cluster_id in set(clusters):
                if cluster_id == -1:  # Noise cluster
                    continue
                
                cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
                cluster_selectors = [selectors[i] for i in cluster_indices]
                
                # Find most common selector in cluster
                selector_counts = Counter(cluster_selectors)
                best_selector, frequency = selector_counts.most_common(1)[0]
                
                confidence = frequency / len(cluster_selectors)
                
                if confidence >= self.confidence_threshold:
                    pattern = ExtractionPattern(
                        selector=best_selector,
                        confidence=confidence,
                        frequency=frequency,
                        domains=[examples[0].domain],
                        success_rate=confidence,
                        content_type=content_type
                    )
                    patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing HTML patterns: {e}")
            return []
    
    def _find_matching_elements(self, soup: BeautifulSoup, target_text: str) -> List[Tag]:
        """Find HTML elements containing the target text."""
        matching_elements = []
        
        # Clean target text for comparison
        target_clean = " ".join(target_text.split()).lower()
        
        # Search through all elements
        for element in soup.find_all():
            if element.string:
                element_text = " ".join(element.string.split()).lower()
                
                # Check for exact match or substantial overlap
                if target_clean in element_text or element_text in target_clean:
                    if len(element_text) > 10:  # Avoid very short matches
                        matching_elements.append(element)
        
        return matching_elements[:5]  # Limit to top 5 matches
    
    def _extract_element_features(self, element: Tag) -> str:
        """Extract features from an HTML element for ML analysis."""
        features = []
        
        # Tag name
        features.append(f"tag:{element.name}")
        
        # Classes
        if element.get('class'):
            for cls in element.get('class'):
                features.append(f"class:{cls}")
        
        # ID
        if element.get('id'):
            features.append(f"id:{element.get('id')}")
        
        # Parent tag
        if element.parent:
            features.append(f"parent:{element.parent.name}")
        
        # Position in parent
        if element.parent:
            siblings = [child for child in element.parent.children if hasattr(child, 'name')]
            position = siblings.index(element) if element in siblings else 0
            features.append(f"position:{position}")
        
        # Text length category
        text_length = len(element.get_text().strip())
        if text_length < 50:
            features.append("length:short")
        elif text_length < 200:
            features.append("length:medium")
        else:
            features.append("length:long")
        
        return " ".join(features)
    
    def _generate_element_selector(self, element: Tag) -> str:
        """Generate a CSS selector for an HTML element."""
        selectors = []
        
        # Try ID first (most specific)
        if element.get('id'):
            return f"#{element.get('id')}"
        
        # Try class combinations
        if element.get('class'):
            classes = element.get('class')
            if len(classes) == 1:
                selectors.append(f"{element.name}.{classes[0]}")
            else:
                # Try most specific class combinations
                for cls in classes:
                    selectors.append(f"{element.name}.{cls}")
                selectors.append(f"{element.name}.{'.'.join(classes[:2])}")
        
        # Try tag with parent context
        if element.parent and element.parent.name != 'body':
            parent_selector = ""
            if element.parent.get('class'):
                parent_class = element.parent.get('class')[0]
                parent_selector = f".{parent_class}"
            elif element.parent.get('id'):
                parent_selector = f"#{element.parent.get('id')}"
            else:
                parent_selector = element.parent.name
            
            selectors.append(f"{parent_selector} {element.name}")
        
        # Fallback to just tag name
        if not selectors:
            selectors.append(element.name)
        
        # Return the most specific selector
        return selectors[0] if selectors else element.name
    
    def _deduplicate_patterns(self, patterns: List[ExtractionPattern]) -> List[ExtractionPattern]:
        """Remove duplicate patterns and keep the best ones."""
        # Group by content type and selector
        pattern_groups = defaultdict(list)
        
        for pattern in patterns:
            key = (pattern.content_type, pattern.selector)
            pattern_groups[key].append(pattern)
        
        # Keep the best pattern from each group
        deduplicated = []
        for group in pattern_groups.values():
            # Sort by confidence and frequency
            best_pattern = max(group, key=lambda p: (p.confidence, p.frequency))
            deduplicated.append(best_pattern)
        
        return deduplicated
    
    async def _generate_selectors_from_patterns(self, domain: str) -> Optional[ScrapingSelectors]:
        """Generate selectors from learned patterns."""
        try:
            if domain not in self.patterns:
                return None
            
            patterns_by_type = defaultdict(list)
            for pattern in self.patterns[domain]:
                patterns_by_type[pattern.content_type].append(pattern)
            
            # Find best selector for each content type
            selectors = {}
            
            for content_type in ['title', 'content', 'author', 'publish_date', 'summary']:
                if content_type in patterns_by_type:
                    # Sort by confidence and frequency
                    best_pattern = max(
                        patterns_by_type[content_type],
                        key=lambda p: (p.confidence, p.frequency)
                    )
                    selectors[content_type] = best_pattern.selector
            
            # Ensure required selectors exist
            if 'title' not in selectors or 'content' not in selectors:
                return None
            
            return ScrapingSelectors(
                title=selectors['title'],
                content=selectors['content'],
                author=selectors.get('author'),
                publish_date=selectors.get('publish_date'),
                summary=selectors.get('summary')
            )
            
        except Exception as e:
            logger.error(f"Error generating selectors for {domain}: {e}")
            return None
    
    async def _generate_actions_from_patterns(self, domain: str) -> List[ScrapingAction]:
        """Generate actions from learned patterns."""
        # For now, return empty list
        # In the future, this could analyze common actions needed for the domain
        return []
    
    async def _estimate_recipe_success_rate(
        self, 
        domain: str, 
        selectors: ScrapingSelectors
    ) -> float:
        """Estimate success rate for a recipe based on historical data."""
        try:
            if domain not in self.examples:
                return 0.5  # Default estimate
            
            # Test selectors against historical examples
            successful_tests = 0
            total_tests = 0
            
            for example in self.examples[domain][-10:]:  # Test on last 10 examples
                soup = BeautifulSoup(example.html_content, 'html.parser')
                
                # Test if selectors would work
                title_found = bool(soup.select_one(selectors.title))
                content_found = bool(soup.select_one(selectors.content))
                
                if title_found and content_found:
                    successful_tests += 1
                total_tests += 1
            
            return successful_tests / total_tests if total_tests > 0 else 0.5
            
        except Exception as e:
            logger.error(f"Error estimating success rate: {e}")
            return 0.5
    
    async def _generate_recipe_for_domain(self, domain: str):
        """Generate and save a recipe for a domain if patterns are strong enough."""
        try:
            recipe = await self.generate_recipe(domain)
            
            if recipe:
                # Create recipe in database
                created_recipe = await self.recipe_engine.create_recipe(
                    domain=recipe.domain,
                    selectors=recipe.selectors,
                    actions=recipe.actions,
                    created_by=recipe.created_by
                )
                
                logger.info(f"Auto-generated recipe for {domain}: {created_recipe.id}")
                
        except Exception as e:
            logger.error(f"Error generating recipe for domain {domain}: {e}")


# Global recipe learner instance
recipe_learner = RecipeLearner()