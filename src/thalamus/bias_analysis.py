"""
Thalamus Bias Detection and Narrative Analysis Engine

This module implements advanced bias detection and narrative analysis:
1. Framing detection using dependency parsing and sentiment polarity shifts
2. Source attribution analysis and bias indicators
3. Narrative extraction using topic modeling (LDA/NMF)
4. Linguistic pattern matching for bias detection
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from collections import Counter, defaultdict
import math

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from spacy.matcher import Matcher

try:
    import gensim
    from gensim import corpora, models
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False


@dataclass
class BiasIndicator:
    """Bias indicator with confidence score."""
    type: str
    description: str
    confidence: float
    evidence: List[str]
    severity: str  # low, medium, high


@dataclass
class FramingPattern:
    """Detected framing pattern in text."""
    pattern_type: str
    matched_text: str
    context: str
    bias_direction: str  # positive, negative, neutral
    confidence: float


@dataclass
class NarrativeTheme:
    """Extracted narrative theme."""
    theme_id: int
    keywords: List[str]
    coherence_score: float
    prevalence: float
    description: str


@dataclass
class BiasAnalysisResult:
    """Complete bias analysis result."""
    overall_bias_score: float
    bias_indicators: List[BiasIndicator]
    framing_patterns: List[FramingPattern]
    narrative_themes: List[NarrativeTheme]
    source_bias_indicators: Dict[str, Any]
    linguistic_patterns: Dict[str, Any]
    confidence: float


class BiasDetectionEngine:
    """
    Advanced bias detection and narrative analysis engine.
    
    Features:
    - Framing detection with linguistic pattern matching
    - Source attribution analysis
    - Narrative extraction using topic modeling
    - Bias indicator calculation with confidence scoring
    - Linguistic bias pattern recognition
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize the bias detection engine."""
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.nlp = None
        self.matcher = None
        
        # Initialize models
        self._initialize_models()
        
        # Configuration
        self.num_topics = 5
        self.min_topic_coherence = 0.3
        self.bias_threshold = 0.6
        
        # Bias patterns and indicators
        self._setup_bias_patterns()
        self._setup_framing_patterns()
    
    def _initialize_models(self):
        """Initialize spaCy model and matcher."""
        try:
            self.nlp = spacy.load(self.model_name)
            self.matcher = Matcher(self.nlp.vocab)
            self.logger.info(f"Loaded spaCy model: {self.model_name}")
        except OSError:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.matcher = Matcher(self.nlp.vocab)
                self.logger.info("Loaded fallback spaCy model: en_core_web_sm")
            except OSError:
                self.logger.error("No spaCy model available")
                self.nlp = None
                self.matcher = None
    
    def _setup_bias_patterns(self):
        """Setup bias detection patterns."""
        self.bias_keywords = {
            "loaded_language": {
                "positive_spin": [
                    "revolutionary", "breakthrough", "game-changing", "unprecedented",
                    "remarkable", "extraordinary", "phenomenal", "outstanding"
                ],
                "negative_spin": [
                    "devastating", "catastrophic", "alarming", "shocking",
                    "disturbing", "troubling", "concerning", "worrying"
                ]
            },
            "emotional_appeals": [
                "heartbreaking", "inspiring", "outrageous", "unbelievable",
                "incredible", "amazing", "terrible", "wonderful"
            ],
            "absolute_terms": [
                "always", "never", "all", "none", "every", "completely",
                "totally", "absolutely", "entirely", "perfectly"
            ],
            "weasel_words": [
                "some say", "critics argue", "many believe", "it is said",
                "sources claim", "reportedly", "allegedly", "supposedly"
            ]
        }
        
        # Source credibility indicators
        self.source_indicators = {
            "authority_appeals": [
                "expert", "professor", "doctor", "researcher", "scientist",
                "analyst", "specialist", "authority", "official"
            ],
            "credibility_boosters": [
                "study shows", "research indicates", "data reveals",
                "statistics show", "evidence suggests", "proven"
            ],
            "uncertainty_markers": [
                "might", "could", "possibly", "perhaps", "maybe",
                "seems", "appears", "suggests", "indicates"
            ]
        }
    
    def _setup_framing_patterns(self):
        """Setup framing detection patterns using spaCy matcher."""
        if not self.matcher:
            return
        
        # Victim framing patterns
        victim_patterns = [
            [{"LOWER": {"IN": ["victim", "victims"]}}, {"LOWER": "of"}],
            [{"LOWER": {"IN": ["suffered", "endured", "faced"]}}, {"POS": "NOUN"}],
            [{"LOWER": {"IN": ["targeted", "attacked", "harassed"]}}]
        ]
        
        # Hero framing patterns
        hero_patterns = [
            [{"LOWER": {"IN": ["hero", "champion", "leader"]}}, {"LOWER": {"IN": ["of", "for"]}}],
            [{"LOWER": {"IN": ["fought", "battled", "struggled"]}}, {"LOWER": "for"}],
            [{"LOWER": {"IN": ["defended", "protected", "saved"]}}]
        ]
        
        # Threat framing patterns
        threat_patterns = [
            [{"LOWER": {"IN": ["threat", "danger", "risk"]}}, {"LOWER": "to"}],
            [{"LOWER": {"IN": ["threatens", "endangers", "jeopardizes"]}}],
            [{"LOWER": {"IN": ["crisis", "emergency", "disaster"]}}]
        ]
        
        # Add patterns to matcher
        for i, pattern in enumerate(victim_patterns):
            self.matcher.add(f"VICTIM_FRAME_{i}", [pattern])
        
        for i, pattern in enumerate(hero_patterns):
            self.matcher.add(f"HERO_FRAME_{i}", [pattern])
        
        for i, pattern in enumerate(threat_patterns):
            self.matcher.add(f"THREAT_FRAME_{i}", [pattern])
    
    async def analyze_bias(
        self,
        text: str,
        source_domain: Optional[str] = None,
        title: Optional[str] = None
    ) -> BiasAnalysisResult:
        """
        Perform comprehensive bias analysis on text.
        
        Args:
            text: Text to analyze for bias
            source_domain: Optional source domain for attribution analysis
            title: Optional title for context
            
        Returns:
            BiasAnalysisResult with comprehensive bias analysis
        """
        try:
            # Detect framing patterns
            framing_patterns = await self._detect_framing_patterns(text)
            
            # Analyze linguistic bias patterns
            linguistic_patterns = await self._analyze_linguistic_patterns(text)
            
            # Extract narrative themes
            narrative_themes = await self._extract_narrative_themes(text)
            
            # Analyze source bias indicators
            source_bias = await self._analyze_source_bias(text, source_domain)
            
            # Calculate bias indicators
            bias_indicators = await self._calculate_bias_indicators(
                text, framing_patterns, linguistic_patterns, source_bias
            )
            
            # Calculate overall bias score
            overall_bias_score = self._calculate_overall_bias_score(
                bias_indicators, framing_patterns, linguistic_patterns
            )
            
            # Calculate confidence
            confidence = self._calculate_analysis_confidence(
                bias_indicators, framing_patterns, narrative_themes
            )
            
            return BiasAnalysisResult(
                overall_bias_score=overall_bias_score,
                bias_indicators=bias_indicators,
                framing_patterns=framing_patterns,
                narrative_themes=narrative_themes,
                source_bias_indicators=source_bias,
                linguistic_patterns=linguistic_patterns,
                confidence=confidence
            )
            
        except Exception as e:
            self.logger.error(f"Error in bias analysis: {e}")
            return BiasAnalysisResult(
                overall_bias_score=0.0,
                bias_indicators=[],
                framing_patterns=[],
                narrative_themes=[],
                source_bias_indicators={},
                linguistic_patterns={},
                confidence=0.0
            )
    
    async def _detect_framing_patterns(self, text: str) -> List[FramingPattern]:
        """Detect framing patterns in text using dependency parsing."""
        patterns = []
        
        if not self.nlp or not self.matcher:
            return patterns
        
        try:
            doc = self.nlp(text)
            matches = self.matcher(doc)
            
            for match_id, start, end in matches:
                span = doc[start:end]
                pattern_name = self.nlp.vocab.strings[match_id]
                
                # Determine pattern type and bias direction
                if "VICTIM" in pattern_name:
                    pattern_type = "victim_framing"
                    bias_direction = "negative"
                elif "HERO" in pattern_name:
                    pattern_type = "hero_framing"
                    bias_direction = "positive"
                elif "THREAT" in pattern_name:
                    pattern_type = "threat_framing"
                    bias_direction = "negative"
                else:
                    pattern_type = "unknown"
                    bias_direction = "neutral"
                
                # Get context (surrounding sentences)
                sent = span.sent
                context = sent.text
                
                # Calculate confidence based on pattern strength
                confidence = self._calculate_pattern_confidence(span, pattern_type)
                
                patterns.append(FramingPattern(
                    pattern_type=pattern_type,
                    matched_text=span.text,
                    context=context,
                    bias_direction=bias_direction,
                    confidence=confidence
                ))
        
        except Exception as e:
            self.logger.error(f"Error detecting framing patterns: {e}")
        
        return patterns
    
    async def _analyze_linguistic_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze linguistic patterns that indicate bias."""
        patterns = {
            "loaded_language_score": 0.0,
            "emotional_appeals_count": 0,
            "absolute_terms_count": 0,
            "weasel_words_count": 0,
            "passive_voice_ratio": 0.0,
            "sentence_complexity": 0.0,
            "hedge_words_count": 0
        }
        
        try:
            text_lower = text.lower()
            
            # Count loaded language
            positive_spin_count = sum(1 for word in self.bias_keywords["loaded_language"]["positive_spin"] 
                                    if word in text_lower)
            negative_spin_count = sum(1 for word in self.bias_keywords["loaded_language"]["negative_spin"] 
                                    if word in text_lower)
            
            total_words = len(text.split())
            if total_words > 0:
                patterns["loaded_language_score"] = (positive_spin_count + negative_spin_count) / total_words
            
            # Count emotional appeals
            patterns["emotional_appeals_count"] = sum(1 for word in self.bias_keywords["emotional_appeals"] 
                                                    if word in text_lower)
            
            # Count absolute terms
            patterns["absolute_terms_count"] = sum(1 for word in self.bias_keywords["absolute_terms"] 
                                                 if word in text_lower)
            
            # Count weasel words
            patterns["weasel_words_count"] = sum(1 for phrase in self.bias_keywords["weasel_words"] 
                                               if phrase in text_lower)
            
            # Analyze with spaCy if available
            if self.nlp:
                doc = self.nlp(text)
                
                # Calculate passive voice ratio
                passive_count = 0
                total_sentences = 0
                
                for sent in doc.sents:
                    total_sentences += 1
                    # Simple passive voice detection
                    if any(token.dep_ == "auxpass" for token in sent):
                        passive_count += 1
                
                if total_sentences > 0:
                    patterns["passive_voice_ratio"] = passive_count / total_sentences
                
                # Calculate sentence complexity (average dependency depth)
                total_depth = 0
                sentence_count = 0
                
                for sent in doc.sents:
                    sentence_count += 1
                    max_depth = max((self._get_token_depth(token) for token in sent), default=0)
                    total_depth += max_depth
                
                if sentence_count > 0:
                    patterns["sentence_complexity"] = total_depth / sentence_count
                
                # Count hedge words (uncertainty markers)
                patterns["hedge_words_count"] = sum(1 for phrase in self.source_indicators["uncertainty_markers"] 
                                                  if phrase in text_lower)
        
        except Exception as e:
            self.logger.error(f"Error analyzing linguistic patterns: {e}")
        
        return patterns
    
    async def _extract_narrative_themes(self, text: str) -> List[NarrativeTheme]:
        """Extract narrative themes using topic modeling."""
        themes = []
        
        try:
            # Preprocess text for topic modeling
            sentences = self._preprocess_for_topics(text)
            
            if len(sentences) < 3:  # Need minimum sentences for topic modeling
                return themes
            
            # Use LDA for topic modeling
            if GENSIM_AVAILABLE:
                themes = await self._extract_themes_with_gensim(sentences)
            else:
                themes = await self._extract_themes_with_sklearn(sentences)
        
        except Exception as e:
            self.logger.error(f"Error extracting narrative themes: {e}")
        
        return themes
    
    async def _extract_themes_with_gensim(self, sentences: List[str]) -> List[NarrativeTheme]:
        """Extract themes using Gensim LDA."""
        themes = []
        
        try:
            # Tokenize and create dictionary
            texts = [sentence.split() for sentence in sentences]
            dictionary = corpora.Dictionary(texts)
            corpus = [dictionary.doc2bow(text) for text in texts]
            
            # Train LDA model
            lda_model = models.LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=self.num_topics,
                random_state=42,
                passes=10,
                alpha='auto',
                per_word_topics=True
            )
            
            # Extract themes
            for topic_id in range(self.num_topics):
                topic_words = lda_model.show_topic(topic_id, topn=10)
                keywords = [word for word, _ in topic_words]
                
                # Calculate coherence (simplified)
                coherence_score = sum(prob for _, prob in topic_words) / len(topic_words)
                
                if coherence_score >= self.min_topic_coherence:
                    # Calculate prevalence
                    topic_prevalence = sum(1 for doc_topics in lda_model.get_document_topics(corpus)
                                         if any(tid == topic_id and prob > 0.1 for tid, prob in doc_topics))
                    prevalence = topic_prevalence / len(corpus) if corpus else 0
                    
                    # Generate description
                    description = f"Theme about {', '.join(keywords[:3])}"
                    
                    themes.append(NarrativeTheme(
                        theme_id=topic_id,
                        keywords=keywords,
                        coherence_score=coherence_score,
                        prevalence=prevalence,
                        description=description
                    ))
        
        except Exception as e:
            self.logger.error(f"Error with Gensim LDA: {e}")
        
        return themes
    
    async def _extract_themes_with_sklearn(self, sentences: List[str]) -> List[NarrativeTheme]:
        """Extract themes using scikit-learn LDA/NMF."""
        themes = []
        
        try:
            # Vectorize text
            vectorizer = CountVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8
            )
            
            doc_term_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            
            # Use LDA
            lda = LatentDirichletAllocation(
                n_components=self.num_topics,
                random_state=42,
                max_iter=10
            )
            
            lda.fit(doc_term_matrix)
            
            # Extract themes
            for topic_id, topic in enumerate(lda.components_):
                # Get top words for topic
                top_word_indices = topic.argsort()[-10:][::-1]
                keywords = [feature_names[i] for i in top_word_indices]
                
                # Calculate coherence (simplified)
                coherence_score = np.mean(topic[top_word_indices])
                
                if coherence_score >= self.min_topic_coherence:
                    # Calculate prevalence
                    doc_topic_probs = lda.transform(doc_term_matrix)
                    prevalence = np.mean(doc_topic_probs[:, topic_id] > 0.1)
                    
                    description = f"Theme about {', '.join(keywords[:3])}"
                    
                    themes.append(NarrativeTheme(
                        theme_id=topic_id,
                        keywords=keywords,
                        coherence_score=float(coherence_score),
                        prevalence=float(prevalence),
                        description=description
                    ))
        
        except Exception as e:
            self.logger.error(f"Error with sklearn topic modeling: {e}")
        
        return themes
    
    async def _analyze_source_bias(self, text: str, source_domain: Optional[str]) -> Dict[str, Any]:
        """Analyze source attribution and bias indicators."""
        source_analysis = {
            "authority_appeals_count": 0,
            "credibility_boosters_count": 0,
            "uncertainty_markers_count": 0,
            "source_diversity_score": 0.0,
            "attribution_ratio": 0.0,
            "domain_bias_score": 0.0
        }
        
        try:
            text_lower = text.lower()
            
            # Count authority appeals
            source_analysis["authority_appeals_count"] = sum(
                1 for word in self.source_indicators["authority_appeals"] 
                if word in text_lower
            )
            
            # Count credibility boosters
            source_analysis["credibility_boosters_count"] = sum(
                1 for phrase in self.source_indicators["credibility_boosters"] 
                if phrase in text_lower
            )
            
            # Count uncertainty markers
            source_analysis["uncertainty_markers_count"] = sum(
                1 for phrase in self.source_indicators["uncertainty_markers"] 
                if phrase in text_lower
            )
            
            # Analyze source attribution
            if self.nlp:
                doc = self.nlp(text)
                
                # Count quoted sources
                quoted_sources = len(re.findall(r'"[^"]*"', text))
                total_sentences = len(list(doc.sents))
                
                if total_sentences > 0:
                    source_analysis["attribution_ratio"] = quoted_sources / total_sentences
            
            # Domain bias score (simplified - would need external bias database)
            if source_domain:
                # This would typically use a bias rating database
                # For now, use a simple heuristic
                known_biased_domains = {
                    "example-biased.com": 0.8,
                    "partisan-news.com": 0.9
                }
                source_analysis["domain_bias_score"] = known_biased_domains.get(source_domain, 0.0)
        
        except Exception as e:
            self.logger.error(f"Error analyzing source bias: {e}")
        
        return source_analysis
    
    async def _calculate_bias_indicators(
        self,
        text: str,
        framing_patterns: List[FramingPattern],
        linguistic_patterns: Dict[str, Any],
        source_bias: Dict[str, Any]
    ) -> List[BiasIndicator]:
        """Calculate specific bias indicators with confidence scores."""
        indicators = []
        
        try:
            # Loaded language indicator
            if linguistic_patterns["loaded_language_score"] > 0.05:
                severity = "high" if linguistic_patterns["loaded_language_score"] > 0.1 else "medium"
                indicators.append(BiasIndicator(
                    type="loaded_language",
                    description="Text contains emotionally charged or loaded language",
                    confidence=min(1.0, linguistic_patterns["loaded_language_score"] * 10),
                    evidence=[f"Loaded language score: {linguistic_patterns['loaded_language_score']:.3f}"],
                    severity=severity
                ))
            
            # Framing bias indicator
            if framing_patterns:
                avg_confidence = sum(p.confidence for p in framing_patterns) / len(framing_patterns)
                indicators.append(BiasIndicator(
                    type="framing_bias",
                    description="Text uses biased framing patterns",
                    confidence=avg_confidence,
                    evidence=[f"{p.pattern_type}: {p.matched_text}" for p in framing_patterns[:3]],
                    severity="high" if avg_confidence > 0.7 else "medium"
                ))
            
            # Source credibility indicator
            if source_bias["domain_bias_score"] > 0.5:
                indicators.append(BiasIndicator(
                    type="source_bias",
                    description="Content from potentially biased source",
                    confidence=source_bias["domain_bias_score"],
                    evidence=[f"Domain bias score: {source_bias['domain_bias_score']:.3f}"],
                    severity="high" if source_bias["domain_bias_score"] > 0.8 else "medium"
                ))
            
            # Weasel words indicator
            if linguistic_patterns["weasel_words_count"] > 2:
                confidence = min(1.0, linguistic_patterns["weasel_words_count"] / 10)
                indicators.append(BiasIndicator(
                    type="weasel_words",
                    description="Text contains vague attribution and weasel words",
                    confidence=confidence,
                    evidence=[f"Weasel words count: {linguistic_patterns['weasel_words_count']}"],
                    severity="medium"
                ))
            
            # Absolute terms indicator
            if linguistic_patterns["absolute_terms_count"] > 3:
                confidence = min(1.0, linguistic_patterns["absolute_terms_count"] / 15)
                indicators.append(BiasIndicator(
                    type="absolute_language",
                    description="Text uses absolute terms that may indicate bias",
                    confidence=confidence,
                    evidence=[f"Absolute terms count: {linguistic_patterns['absolute_terms_count']}"],
                    severity="low"
                ))
        
        except Exception as e:
            self.logger.error(f"Error calculating bias indicators: {e}")
        
        return indicators
    
    def _calculate_overall_bias_score(
        self,
        bias_indicators: List[BiasIndicator],
        framing_patterns: List[FramingPattern],
        linguistic_patterns: Dict[str, Any]
    ) -> float:
        """Calculate overall bias score from all indicators."""
        if not bias_indicators:
            return 0.0
        
        # Weight different types of bias
        weights = {
            "loaded_language": 0.3,
            "framing_bias": 0.4,
            "source_bias": 0.2,
            "weasel_words": 0.1,
            "absolute_language": 0.05
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for indicator in bias_indicators:
            weight = weights.get(indicator.type, 0.1)
            weighted_score += indicator.confidence * weight
            total_weight += weight
        
        if total_weight > 0:
            return min(1.0, weighted_score / total_weight)
        
        return 0.0
    
    def _calculate_analysis_confidence(
        self,
        bias_indicators: List[BiasIndicator],
        framing_patterns: List[FramingPattern],
        narrative_themes: List[NarrativeTheme]
    ) -> float:
        """Calculate confidence in the bias analysis."""
        # Base confidence on number and quality of indicators
        indicator_confidence = sum(i.confidence for i in bias_indicators) / max(1, len(bias_indicators))
        pattern_confidence = sum(p.confidence for p in framing_patterns) / max(1, len(framing_patterns))
        theme_confidence = sum(t.coherence_score for t in narrative_themes) / max(1, len(narrative_themes))
        
        # Combine confidences
        overall_confidence = (indicator_confidence + pattern_confidence + theme_confidence) / 3
        
        return min(1.0, overall_confidence)
    
    def _calculate_pattern_confidence(self, span, pattern_type: str) -> float:
        """Calculate confidence for a detected pattern."""
        # Base confidence on pattern type and context
        base_confidence = {
            "victim_framing": 0.8,
            "hero_framing": 0.7,
            "threat_framing": 0.9,
            "unknown": 0.5
        }.get(pattern_type, 0.5)
        
        # Adjust based on span length and context
        length_factor = min(1.0, len(span.text.split()) / 5)
        
        return base_confidence * length_factor
    
    def _get_token_depth(self, token) -> int:
        """Get dependency depth of a token."""
        depth = 0
        current = token
        while current.head != current:
            depth += 1
            current = current.head
            if depth > 10:  # Prevent infinite loops
                break
        return depth
    
    def _preprocess_for_topics(self, text: str) -> List[str]:
        """Preprocess text for topic modeling."""
        # Split into sentences and clean
        sentences = re.split(r'[.!?]+', text)
        
        processed_sentences = []
        for sentence in sentences:
            sentence = sentence.strip().lower()
            if len(sentence) > 20 and len(sentence.split()) > 3:  # Filter short sentences
                # Remove special characters but keep spaces
                sentence = re.sub(r'[^\w\s]', '', sentence)
                processed_sentences.append(sentence)
        
        return processed_sentences


# Global bias detection engine instance for dependency injection
_bias_engine: Optional[BiasDetectionEngine] = None


def get_bias_engine() -> BiasDetectionEngine:
    """Get global bias detection engine instance for dependency injection."""
    global _bias_engine
    if _bias_engine is None:
        _bias_engine = BiasDetectionEngine()
    return _bias_engine


async def analyze_text_bias(
    text: str,
    source_domain: Optional[str] = None,
    title: Optional[str] = None
) -> BiasAnalysisResult:
    """
    Convenience function to analyze text for bias and narrative patterns.
    
    Args:
        text: Text to analyze
        source_domain: Optional source domain
        title: Optional title for context
        
    Returns:
        BiasAnalysisResult with comprehensive analysis
    """
    engine = get_bias_engine()
    return await engine.analyze_bias(text, source_domain, title)