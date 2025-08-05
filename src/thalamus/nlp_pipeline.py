"""
Thalamus NLP Engine - Multi-tier NLP Processing Pipeline

This module implements a comprehensive NLP processing pipeline with multiple tiers:
1. Rule-based processing with spaCy for fast analysis
2. TextRank algorithm for extractive summarization
3. Hybrid TF-IDF + sentence position analysis

The pipeline is designed for scalability and graceful degradation.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import math
from collections import Counter, defaultdict

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.matcher import Matcher
from spacy.tokens import Span
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

from ..shared.schemas import NLPAnalysis, EntityExtraction, SentimentAnalysis


class ProcessingTier(Enum):
    """NLP processing tiers for different complexity levels."""
    RULE_BASED = "rule_based"
    TEXTRANK = "textrank"
    HYBRID_TFIDF = "hybrid_tfidf"


class SentimentPolarity(Enum):
    """Sentiment polarity classifications."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class ProcessingResult:
    """Result from NLP processing with metadata."""
    tier: ProcessingTier
    success: bool
    processing_time: float
    error_message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


@dataclass
class TextRankNode:
    """Node in TextRank graph representing a sentence."""
    sentence: str
    index: int
    score: float = 0.0
    tokens: List[str] = field(default_factory=list)
    
    
@dataclass
class SummaryCandidate:
    """Candidate sentence for summarization with scoring."""
    sentence: str
    index: int
    tfidf_score: float = 0.0
    position_score: float = 0.0
    length_score: float = 0.0
    combined_score: float = 0.0


class NLPPipeline:
    """
    Multi-tier NLP processing pipeline with graceful degradation.
    
    Implements three processing tiers:
    1. Rule-based: Fast spaCy processing for basic analysis
    2. TextRank: Graph-based extractive summarization
    3. Hybrid TF-IDF: Advanced scoring with position analysis
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize the NLP pipeline with spaCy model."""
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.nlp = None
        self.matcher = None
        self.vader_analyzer = None
        self._initialize_models()
        
        # Processing configuration
        self.max_summary_sentences = 3
        self.textrank_iterations = 30
        self.textrank_damping = 0.85
        self.min_sentence_length = 10
        self.max_sentence_length = 500
        
        # TF-IDF configuration
        self.tfidf_max_features = 1000
        self.tfidf_ngram_range = (1, 2)
        
        # Custom entity patterns for business-specific entities
        self.custom_entity_patterns = {
            "CRYPTO_COIN": [
                [{"LOWER": "bitcoin"}, {"LOWER": {"IN": ["btc", "bitcoin"]}}],
                [{"LOWER": "ethereum"}, {"LOWER": {"IN": ["eth", "ethereum"]}}],
                [{"LOWER": {"IN": ["dogecoin", "doge", "shiba", "cardano", "ada", "solana", "sol"]}}]
            ],
            "AI_TOOL": [
                [{"LOWER": {"IN": ["chatgpt", "gpt", "claude", "gemini", "copilot"]}}, {"LOWER": {"IN": ["ai", "assistant", "model"]}, "OP": "?"}],
                [{"LOWER": {"IN": ["tensorflow", "pytorch", "keras", "scikit-learn", "huggingface"]}}],
                [{"LOWER": {"IN": ["openai", "anthropic", "google", "microsoft"]}}, {"LOWER": {"IN": ["ai", "ml", "model"]}}]
            ],
            "TECH_COMPANY": [
                [{"LOWER": {"IN": ["apple", "google", "microsoft", "amazon", "meta", "tesla", "nvidia", "intel", "amd"]}}],
                [{"LOWER": {"IN": ["openai", "anthropic", "deepmind", "huggingface", "stability"]}}, {"LOWER": {"IN": ["ai", "inc", "labs"]}, "OP": "?"}]
            ]
        }
        
    def _initialize_models(self):
        """Initialize spaCy model with error handling."""
        try:
            self.nlp = spacy.load(self.model_name)
            self.logger.info(f"Loaded spaCy model: {self.model_name}")
            
            # Initialize custom entity matcher
            self.matcher = Matcher(self.nlp.vocab)
            self._setup_custom_patterns()
            
        except OSError:
            self.logger.error(f"Failed to load spaCy model: {self.model_name}")
            # Try to load a smaller model as fallback
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.logger.info("Loaded fallback spaCy model: en_core_web_sm")
                self.matcher = Matcher(self.nlp.vocab)
                self._setup_custom_patterns()
            except OSError:
                self.logger.error("No spaCy model available")
                self.nlp = None
                self.matcher = None
        
        # Initialize VADER sentiment analyzer
        if VADER_AVAILABLE:
            self.vader_analyzer = SentimentIntensityAnalyzer()
            self.logger.info("Initialized VADER sentiment analyzer")
        else:
            self.logger.warning("VADER sentiment analyzer not available")
    
    def _setup_custom_patterns(self):
        """Setup custom entity patterns for business-specific recognition."""
        if not self.matcher:
            return
            
        for entity_type, patterns in self.custom_entity_patterns.items():
            for i, pattern in enumerate(patterns):
                pattern_name = f"{entity_type}_{i}"
                self.matcher.add(pattern_name, [pattern])
    
    async def process_text(
        self,
        text: str,
        title: str = "",
        tiers: List[ProcessingTier] = None
    ) -> NLPAnalysis:
        """
        Process text through multiple NLP tiers.
        
        Args:
            text: Input text to process
            title: Optional title for context
            tiers: List of processing tiers to use (default: all)
            
        Returns:
            NLPAnalysis with results from all successful tiers
        """
        if tiers is None:
            tiers = [ProcessingTier.RULE_BASED, ProcessingTier.TEXTRANK, ProcessingTier.HYBRID_TFIDF]
        
        results = {}
        
        # Process through each tier
        for tier in tiers:
            try:
                if tier == ProcessingTier.RULE_BASED:
                    result = await self._process_rule_based(text, title)
                elif tier == ProcessingTier.TEXTRANK:
                    result = await self._process_textrank(text)
                elif tier == ProcessingTier.HYBRID_TFIDF:
                    result = await self._process_hybrid_tfidf(text, title)
                else:
                    continue
                    
                results[tier.value] = result
                
                if not result.success:
                    self.logger.warning(f"Tier {tier.value} failed: {result.error_message}")
                    
            except Exception as e:
                self.logger.error(f"Error in tier {tier.value}: {str(e)}")
                results[tier.value] = ProcessingResult(
                    tier=tier,
                    success=False,
                    processing_time=0.0,
                    error_message=str(e)
                )
        
        # Combine results into final analysis
        return self._combine_results(results, text, title)
    
    async def _process_rule_based(self, text: str, title: str = "") -> ProcessingResult:
        """
        Tier 1: Rule-based processing with spaCy for fast analysis.
        
        Extracts:
        - Named entities (standard + custom business entities)
        - Advanced sentiment analysis (VADER + TextBlob + rule-based)
        - Part-of-speech tags
        - Sentence structure
        - Category classification
        """
        import time
        start_time = time.time()
        
        if not self.nlp:
            return ProcessingResult(
                tier=ProcessingTier.RULE_BASED,
                success=False,
                processing_time=0.0,
                error_message="spaCy model not available"
            )
        
        try:
            # Process text with spaCy
            doc = self.nlp(text)
            
            # Extract standard entities
            entities = []
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": 1.0  # spaCy doesn't provide confidence scores
                })
            
            # Extract custom business entities using pattern matching
            if self.matcher:
                custom_entities = self._extract_custom_entities(doc)
                entities.extend(custom_entities)
            
            # Advanced sentiment analysis
            sentiment_analysis = self._calculate_advanced_sentiment(text, doc)
            
            # Category classification
            categories = self._classify_content_categories(text, entities)
            
            # Extract key statistics
            sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 5]
            tokens = [token.text for token in doc if not token.is_space]
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                tier=ProcessingTier.RULE_BASED,
                success=True,
                processing_time=processing_time,
                data={
                    "entities": entities,
                    "sentiment_analysis": sentiment_analysis,
                    "sentiment_score": sentiment_analysis["compound_score"],
                    "categories": categories,
                    "sentence_count": len(sentences),
                    "token_count": len(tokens),
                    "sentences": sentences[:5]  # First 5 sentences for preview
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return ProcessingResult(
                tier=ProcessingTier.RULE_BASED,
                success=False,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    async def _process_textrank(self, text: str) -> ProcessingResult:
        """
        Tier 2: TextRank algorithm for extractive summarization.
        
        Implements PageRank algorithm on sentence similarity graph
        to identify most important sentences.
        """
        import time
        start_time = time.time()
        
        try:
            # Split text into sentences
            sentences = self._split_sentences(text)
            
            if len(sentences) < 2:
                return ProcessingResult(
                    tier=ProcessingTier.TEXTRANK,
                    success=False,
                    processing_time=time.time() - start_time,
                    error_message="Insufficient sentences for TextRank"
                )
            
            # Create TextRank nodes
            nodes = []
            for i, sentence in enumerate(sentences):
                tokens = self._tokenize_sentence(sentence)
                nodes.append(TextRankNode(
                    sentence=sentence,
                    index=i,
                    tokens=tokens
                ))
            
            # Build similarity matrix
            similarity_matrix = self._build_similarity_matrix(nodes)
            
            # Run TextRank algorithm
            scores = self._run_textrank(similarity_matrix)
            
            # Update node scores
            for i, score in enumerate(scores):
                nodes[i].score = score
            
            # Sort by score and select top sentences
            ranked_nodes = sorted(nodes, key=lambda x: x.score, reverse=True)
            top_sentences = [node.sentence for node in ranked_nodes[:self.max_summary_sentences]]
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                tier=ProcessingTier.TEXTRANK,
                success=True,
                processing_time=processing_time,
                data={
                    "summary_sentences": top_sentences,
                    "sentence_scores": [(node.sentence, node.score) for node in ranked_nodes],
                    "algorithm": "textrank"
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return ProcessingResult(
                tier=ProcessingTier.TEXTRANK,
                success=False,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    async def _process_hybrid_tfidf(self, text: str, title: str = "") -> ProcessingResult:
        """
        Tier 3: Hybrid TF-IDF + sentence position analysis.
        
        Combines TF-IDF scoring with positional importance and
        sentence length normalization for advanced summarization.
        """
        import time
        start_time = time.time()
        
        try:
            sentences = self._split_sentences(text)
            
            if len(sentences) < 2:
                return ProcessingResult(
                    tier=ProcessingTier.HYBRID_TFIDF,
                    success=False,
                    processing_time=time.time() - start_time,
                    error_message="Insufficient sentences for hybrid analysis"
                )
            
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_features=self.tfidf_max_features,
                ngram_range=self.tfidf_ngram_range,
                stop_words='english',
                lowercase=True
            )
            
            # Fit TF-IDF on sentences
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Calculate sentence scores
            candidates = []
            for i, sentence in enumerate(sentences):
                # TF-IDF score (mean of all term scores)
                tfidf_score = np.mean(tfidf_matrix[i].toarray())
                
                # Position score (higher for beginning and end)
                position_score = self._calculate_position_score(i, len(sentences))
                
                # Length score (penalize very short/long sentences)
                length_score = self._calculate_length_score(sentence)
                
                # Combined score
                combined_score = (
                    0.5 * tfidf_score +
                    0.3 * position_score +
                    0.2 * length_score
                )
                
                candidates.append(SummaryCandidate(
                    sentence=sentence,
                    index=i,
                    tfidf_score=tfidf_score,
                    position_score=position_score,
                    length_score=length_score,
                    combined_score=combined_score
                ))
            
            # Sort by combined score and select top sentences
            candidates.sort(key=lambda x: x.combined_score, reverse=True)
            top_sentences = [c.sentence for c in candidates[:self.max_summary_sentences]]
            
            # Calculate additional metrics
            feature_names = vectorizer.get_feature_names_out()
            top_terms = self._extract_top_terms(tfidf_matrix, feature_names)
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                tier=ProcessingTier.HYBRID_TFIDF,
                success=True,
                processing_time=processing_time,
                data={
                    "summary_sentences": top_sentences,
                    "sentence_scores": [(c.sentence, c.combined_score) for c in candidates],
                    "top_terms": top_terms,
                    "algorithm": "hybrid_tfidf",
                    "score_breakdown": {
                        "tfidf_weight": 0.5,
                        "position_weight": 0.3,
                        "length_weight": 0.2
                    }
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return ProcessingResult(
                tier=ProcessingTier.HYBRID_TFIDF,
                success=False,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    def _calculate_rule_based_sentiment(self, doc) -> float:
        """Calculate sentiment using rule-based approach with spaCy."""
        # Simple rule-based sentiment using word polarity
        positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'positive', 'success', 'win', 'best', 'love', 'like', 'happy'
        }
        negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'dislike',
            'negative', 'fail', 'failure', 'lose', 'sad', 'angry', 'disappointed'
        }
        
        positive_count = 0
        negative_count = 0
        total_words = 0
        
        for token in doc:
            if not token.is_alpha or token.is_stop:
                continue
                
            word = token.text.lower()
            total_words += 1
            
            if word in positive_words:
                positive_count += 1
            elif word in negative_words:
                negative_count += 1
        
        if total_words == 0:
            return 0.0
        
        # Calculate sentiment score (-1 to 1)
        sentiment_score = (positive_count - negative_count) / total_words
        return max(-1.0, min(1.0, sentiment_score))
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences with filtering."""
        if not self.nlp:
            # Fallback sentence splitting
            sentences = re.split(r'[.!?]+', text)
        else:
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]
        
        # Filter sentences
        filtered_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if (self.min_sentence_length <= len(sentence) <= self.max_sentence_length and
                not sentence.isdigit() and
                len(sentence.split()) >= 3):
                filtered_sentences.append(sentence)
        
        return filtered_sentences
    
    def _tokenize_sentence(self, sentence: str) -> List[str]:
        """Tokenize sentence and remove stop words."""
        if self.nlp:
            doc = self.nlp(sentence)
            tokens = [token.lemma_.lower() for token in doc 
                     if token.is_alpha and not token.is_stop and len(token.text) > 2]
        else:
            # Fallback tokenization
            words = re.findall(r'\b[a-zA-Z]{3,}\b', sentence.lower())
            tokens = [word for word in words if word not in STOP_WORDS]
        
        return tokens
    
    def _build_similarity_matrix(self, nodes: List[TextRankNode]) -> np.ndarray:
        """Build similarity matrix for TextRank algorithm."""
        n = len(nodes)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Calculate Jaccard similarity
                    tokens_i = set(nodes[i].tokens)
                    tokens_j = set(nodes[j].tokens)
                    
                    if len(tokens_i) == 0 or len(tokens_j) == 0:
                        similarity = 0.0
                    else:
                        intersection = len(tokens_i.intersection(tokens_j))
                        union = len(tokens_i.union(tokens_j))
                        similarity = intersection / union if union > 0 else 0.0
                    
                    similarity_matrix[i][j] = similarity
        
        return similarity_matrix
    
    def _run_textrank(self, similarity_matrix: np.ndarray) -> List[float]:
        """Run TextRank algorithm on similarity matrix."""
        n = similarity_matrix.shape[0]
        scores = np.ones(n) / n
        
        for _ in range(self.textrank_iterations):
            new_scores = np.zeros(n)
            
            for i in range(n):
                for j in range(n):
                    if i != j and similarity_matrix[j][i] > 0:
                        # Sum of similarities from j to all other nodes
                        sum_similarities = np.sum(similarity_matrix[j])
                        if sum_similarities > 0:
                            new_scores[i] += (similarity_matrix[j][i] / sum_similarities) * scores[j]
                
                new_scores[i] = (1 - self.textrank_damping) / n + self.textrank_damping * new_scores[i]
            
            scores = new_scores
        
        return scores.tolist()
    
    def _calculate_position_score(self, position: int, total_sentences: int) -> float:
        """Calculate position-based score (higher for beginning and end)."""
        if total_sentences <= 1:
            return 1.0
        
        # Normalize position (0 to 1)
        normalized_pos = position / (total_sentences - 1)
        
        # U-shaped curve: higher scores for beginning (0) and end (1)
        # We want a curve that gives high scores at 0 and 1, low at 0.5
        # Use: 1 - 4 * (x - 0.5)^2, but this gives 0 at endpoints
        # Instead use: 2 * |x - 0.5| to get linear increase from middle
        position_score = 2 * abs(normalized_pos - 0.5)
        
        return max(0.0, position_score)
    
    def _calculate_length_score(self, sentence: str) -> float:
        """Calculate length-based score (penalize very short/long sentences)."""
        word_count = len(sentence.split())
        
        # Optimal length range: 10-25 words
        if 10 <= word_count <= 25:
            return 1.0
        elif word_count < 10:
            return word_count / 10.0
        else:
            # Penalize very long sentences
            return max(0.1, 25.0 / word_count)
    
    def _extract_top_terms(self, tfidf_matrix: np.ndarray, feature_names: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """Extract top terms from TF-IDF matrix."""
        # Sum TF-IDF scores across all sentences
        if hasattr(tfidf_matrix, 'toarray'):
            term_scores = np.sum(tfidf_matrix.toarray(), axis=0)
        else:
            term_scores = np.sum(tfidf_matrix, axis=0)
        
        # Get top terms
        top_indices = np.argsort(term_scores)[-top_k:][::-1]
        top_terms = [(feature_names[i], term_scores[i]) for i in top_indices]
        
        return top_terms
    
    def _extract_custom_entities(self, doc) -> List[Dict[str, Any]]:
        """Extract custom business entities using pattern matching."""
        custom_entities = []
        
        if not self.matcher:
            return custom_entities
        
        matches = self.matcher(doc)
        
        for match_id, start, end in matches:
            # Get the matched span
            span = doc[start:end]
            
            # Determine entity type from pattern name
            pattern_name = self.nlp.vocab.strings[match_id]
            entity_type = pattern_name.split('_')[0] + '_' + pattern_name.split('_')[1]  # e.g., "CRYPTO_COIN"
            
            custom_entities.append({
                "text": span.text,
                "label": entity_type,
                "start": span.start_char,
                "end": span.end_char,
                "confidence": 0.9  # High confidence for pattern matches
            })
        
        return custom_entities
    
    def _calculate_advanced_sentiment(self, text: str, doc) -> Dict[str, Any]:
        """Calculate sentiment using multiple approaches for higher accuracy."""
        sentiment_results = {
            "compound_score": 0.0,
            "positive": 0.0,
            "negative": 0.0,
            "neutral": 0.0,
            "confidence": 0.0,
            "methods_used": []
        }
        
        scores = []
        methods = []
        
        # 1. Rule-based sentiment (existing method)
        rule_score = self._calculate_rule_based_sentiment(doc)
        scores.append(rule_score)
        methods.append("rule_based")
        
        # 2. VADER sentiment analysis
        if self.vader_analyzer:
            try:
                vader_scores = self.vader_analyzer.polarity_scores(text)
                scores.append(vader_scores['compound'])
                methods.append("vader")
                
                sentiment_results.update({
                    "positive": vader_scores['pos'],
                    "negative": vader_scores['neg'],
                    "neutral": vader_scores['neu']
                })
            except Exception as e:
                self.logger.warning(f"VADER sentiment analysis failed: {e}")
        
        # 3. TextBlob sentiment analysis
        if TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(text)
                textblob_score = blob.sentiment.polarity
                scores.append(textblob_score)
                methods.append("textblob")
            except Exception as e:
                self.logger.warning(f"TextBlob sentiment analysis failed: {e}")
        
        # Calculate weighted average (more methods = higher confidence)
        if scores:
            sentiment_results["compound_score"] = sum(scores) / len(scores)
            sentiment_results["confidence"] = min(1.0, len(scores) / 3.0)  # Max confidence with 3 methods
            sentiment_results["methods_used"] = methods
        
        return sentiment_results
    
    def _classify_content_categories(self, text: str, entities: List[Dict[str, Any]]) -> List[str]:
        """Classify content into categories based on entities and keywords."""
        categories = []
        text_lower = text.lower()
        
        # Technology category
        tech_keywords = {
            'ai', 'artificial intelligence', 'machine learning', 'deep learning', 'neural network',
            'algorithm', 'software', 'hardware', 'computer', 'technology', 'tech', 'digital',
            'programming', 'coding', 'development', 'app', 'application', 'platform'
        }
        
        # Finance category
        finance_keywords = {
            'money', 'finance', 'financial', 'bank', 'banking', 'investment', 'stock', 'market',
            'economy', 'economic', 'business', 'revenue', 'profit', 'loss', 'trading', 'crypto',
            'cryptocurrency', 'bitcoin', 'ethereum', 'blockchain', 'fintech'
        }
        
        # Politics category
        politics_keywords = {
            'government', 'political', 'politics', 'election', 'vote', 'voting', 'president',
            'congress', 'senate', 'policy', 'law', 'legislation', 'democrat', 'republican',
            'campaign', 'candidate', 'minister', 'parliament'
        }
        
        # Health category
        health_keywords = {
            'health', 'medical', 'medicine', 'doctor', 'hospital', 'patient', 'disease',
            'treatment', 'therapy', 'drug', 'pharmaceutical', 'vaccine', 'covid', 'pandemic',
            'healthcare', 'wellness', 'fitness'
        }
        
        # Science category
        science_keywords = {
            'science', 'scientific', 'research', 'study', 'experiment', 'discovery', 'innovation',
            'laboratory', 'scientist', 'physics', 'chemistry', 'biology', 'astronomy', 'climate'
        }
        
        # Sports category
        sports_keywords = {
            'sport', 'sports', 'game', 'team', 'player', 'match', 'tournament', 'championship',
            'football', 'basketball', 'baseball', 'soccer', 'tennis', 'golf', 'olympics'
        }
        
        # Check for category keywords
        category_mapping = {
            'Technology': tech_keywords,
            'Finance': finance_keywords,
            'Politics': politics_keywords,
            'Health': health_keywords,
            'Science': science_keywords,
            'Sports': sports_keywords
        }
        
        for category, keywords in category_mapping.items():
            # Check text content
            if any(keyword in text_lower for keyword in keywords):
                categories.append(category)
                continue
            
            # Check entities
            entity_texts = [ent['text'].lower() for ent in entities]
            if any(any(keyword in entity_text for keyword in keywords) for entity_text in entity_texts):
                categories.append(category)
        
        # Check custom entity types for additional categorization
        entity_labels = [ent['label'] for ent in entities]
        if any(label.startswith('CRYPTO_') for label in entity_labels):
            if 'Finance' not in categories:
                categories.append('Finance')
        
        if any(label.startswith('AI_') or label.startswith('TECH_') for label in entity_labels):
            if 'Technology' not in categories:
                categories.append('Technology')
        
        # Default to "General" if no specific category found
        if not categories:
            categories.append('General')
        
        return categories
    
    def _combine_results(self, results: Dict[str, ProcessingResult], text: str, title: str) -> NLPAnalysis:
        """Combine results from all tiers into final NLP analysis."""
        # Extract data from successful tiers
        entities = []
        sentiment_score = 0.0
        summary_sentences = []
        categories = []
        significance_score = 0.5  # Default significance
        
        # Rule-based results
        if ProcessingTier.RULE_BASED.value in results and results[ProcessingTier.RULE_BASED.value].success:
            rule_data = results[ProcessingTier.RULE_BASED.value].data
            entities = rule_data.get("entities", [])
            sentiment_score = rule_data.get("sentiment_score", 0.0)
            categories = rule_data.get("categories", [])
        
        # TextRank results (preferred for summarization)
        if ProcessingTier.TEXTRANK.value in results and results[ProcessingTier.TEXTRANK.value].success:
            textrank_data = results[ProcessingTier.TEXTRANK.value].data
            summary_sentences = textrank_data.get("summary_sentences", [])
        elif ProcessingTier.HYBRID_TFIDF.value in results and results[ProcessingTier.HYBRID_TFIDF.value].success:
            # Fallback to hybrid TF-IDF
            hybrid_data = results[ProcessingTier.HYBRID_TFIDF.value].data
            summary_sentences = hybrid_data.get("summary_sentences", [])
        
        # Calculate significance based on entity count and content length
        if entities:
            significance_score = min(1.0, len(entities) * 0.1 + len(text) / 10000)
        
        # Determine sentiment polarity
        if sentiment_score > 0.1:
            sentiment_polarity = SentimentPolarity.POSITIVE.value
        elif sentiment_score < -0.1:
            sentiment_polarity = SentimentPolarity.NEGATIVE.value
        else:
            sentiment_polarity = SentimentPolarity.NEUTRAL.value
        
        # Create summary from top sentences
        summary = " ".join(summary_sentences[:3]) if summary_sentences else ""
        
        return NLPAnalysis(
            sentiment=sentiment_score,
            entities=entities,
            categories=categories,
            significance=significance_score,
            summary=summary,
            processing_tiers=[tier for tier in results.keys() if results[tier].success],
            processing_time=sum(r.processing_time for r in results.values()),
            metadata={
                "sentiment_polarity": sentiment_polarity,
                "entity_count": len(entities),
                "summary_sentence_count": len(summary_sentences),
                "text_length": len(text),
                "title_length": len(title)
            }
        )


# Global pipeline instance for dependency injection
_nlp_pipeline: Optional[NLPPipeline] = None


def get_nlp_pipeline() -> NLPPipeline:
    """Get global NLP pipeline instance for dependency injection."""
    global _nlp_pipeline
    if _nlp_pipeline is None:
        _nlp_pipeline = NLPPipeline()
    return _nlp_pipeline


async def process_article_content(text: str, title: str = "") -> NLPAnalysis:
    """
    Convenience function to process article content through NLP pipeline.
    
    Args:
        text: Article content to process
        title: Article title for context
        
    Returns:
        NLPAnalysis with comprehensive results
    """
    pipeline = get_nlp_pipeline()
    return await pipeline.process_text(text, title)