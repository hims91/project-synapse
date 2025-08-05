"""
Unit tests for Thalamus Bias Detection and Narrative Analysis Engine

Tests bias detection capabilities including:
1. Framing detection using dependency parsing
2. Source attribution analysis and bias indicators
3. Narrative extraction using topic modeling
4. Linguistic pattern matching for bias detection
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.thalamus.bias_analysis import (
    BiasDetectionEngine,
    BiasIndicator,
    FramingPattern,
    NarrativeTheme,
    BiasAnalysisResult,
    get_bias_engine,
    analyze_text_bias
)


class TestBiasDetectionEngine:
    """Test cases for Bias Detection Engine functionality."""
    
    @pytest.fixture
    def bias_engine(self):
        """Create bias detection engine instance for testing."""
        return BiasDetectionEngine()
    
    @pytest.fixture
    def biased_text_samples(self):
        """Sample texts with various types of bias."""
        return {
            "loaded_language": "This devastating policy will completely destroy our economy and absolutely ruin everything.",
            "framing_bias": "The victims of this terrible attack suffered unimaginable trauma while the hero fought bravely.",
            "weasel_words": "Some say this policy is bad. Critics argue it will fail. Sources claim it's problematic.",
            "neutral": "The policy was implemented last year. It affects economic indicators. Researchers are studying the results.",
            "emotional_appeals": "This heartbreaking story will inspire you to take action against this outrageous injustice.",
            "source_bias": "According to experts, this study shows that research indicates the data reveals significant problems.",
            "absolute_terms": "This policy always fails and never works. All economists agree it's completely wrong.",
            "complex_bias": "The devastating attack on our democracy by these radical extremists threatens everything we hold dear. Experts unanimously agree this crisis demands immediate action."
        }   
 
    def test_bias_engine_initialization(self, bias_engine):
        """Test bias detection engine initialization."""
        assert bias_engine is not None
        assert bias_engine.num_topics == 5
        assert bias_engine.min_topic_coherence == 0.3
        assert bias_engine.bias_threshold == 0.6
        assert hasattr(bias_engine, 'bias_keywords')
        assert hasattr(bias_engine, 'source_indicators')
    
    @pytest.mark.asyncio
    async def test_analyze_bias_loaded_language(self, bias_engine, biased_text_samples):
        """Test bias analysis for loaded language."""
        result = await bias_engine.analyze_bias(biased_text_samples["loaded_language"])
        
        assert isinstance(result, BiasAnalysisResult)
        assert result.overall_bias_score > 0.0
        assert len(result.bias_indicators) > 0
        
        # Check for loaded language indicator
        loaded_lang_indicators = [i for i in result.bias_indicators if i.type == "loaded_language"]
        assert len(loaded_lang_indicators) > 0
        assert loaded_lang_indicators[0].confidence > 0.0
    
    @pytest.mark.asyncio
    async def test_analyze_bias_neutral_text(self, bias_engine, biased_text_samples):
        """Test bias analysis for neutral text."""
        result = await bias_engine.analyze_bias(biased_text_samples["neutral"])
        
        assert isinstance(result, BiasAnalysisResult)
        assert result.overall_bias_score < 0.3  # Should be low for neutral text
        assert result.confidence >= 0.0
    
    @pytest.mark.asyncio
    async def test_analyze_bias_with_source_domain(self, bias_engine, biased_text_samples):
        """Test bias analysis with source domain."""
        result = await bias_engine.analyze_bias(
            biased_text_samples["complex_bias"],
            source_domain="example-biased.com"
        )
        
        assert isinstance(result, BiasAnalysisResult)
        assert "domain_bias_score" in result.source_bias_indicators
        assert result.source_bias_indicators["domain_bias_score"] > 0.0
    
    @pytest.mark.asyncio
    async def test_detect_framing_patterns(self, bias_engine, biased_text_samples):
        """Test framing pattern detection."""
        patterns = await bias_engine._detect_framing_patterns(biased_text_samples["framing_bias"])
        
        assert isinstance(patterns, list)
        if patterns:  # Only test if patterns are detected
            assert all(isinstance(p, FramingPattern) for p in patterns)
            assert all(hasattr(p, 'pattern_type') for p in patterns)
            assert all(hasattr(p, 'confidence') for p in patterns)
            assert all(p.confidence >= 0.0 and p.confidence <= 1.0 for p in patterns)
    
    @pytest.mark.asyncio
    async def test_analyze_linguistic_patterns(self, bias_engine, biased_text_samples):
        """Test linguistic pattern analysis."""
        patterns = await bias_engine._analyze_linguistic_patterns(biased_text_samples["loaded_language"])
        
        assert isinstance(patterns, dict)
        assert "loaded_language_score" in patterns
        assert "emotional_appeals_count" in patterns
        assert "absolute_terms_count" in patterns
        assert "weasel_words_count" in patterns
        assert "passive_voice_ratio" in patterns
        assert "sentence_complexity" in patterns
        
        # Check that loaded language is detected
        assert patterns["loaded_language_score"] > 0.0
    
    @pytest.mark.asyncio
    async def test_analyze_linguistic_patterns_weasel_words(self, bias_engine, biased_text_samples):
        """Test detection of weasel words."""
        patterns = await bias_engine._analyze_linguistic_patterns(biased_text_samples["weasel_words"])
        
        assert patterns["weasel_words_count"] > 0
    
    @pytest.mark.asyncio
    async def test_analyze_linguistic_patterns_absolute_terms(self, bias_engine, biased_text_samples):
        """Test detection of absolute terms."""
        patterns = await bias_engine._analyze_linguistic_patterns(biased_text_samples["absolute_terms"])
        
        assert patterns["absolute_terms_count"] > 0
    
    @pytest.mark.asyncio
    async def test_extract_narrative_themes(self, bias_engine):
        """Test narrative theme extraction."""
        long_text = """
        Technology is rapidly changing our world. Artificial intelligence and machine learning 
        are transforming industries. Companies are investing heavily in AI research and development.
        The future of work will be shaped by automation and digital transformation.
        Climate change is another major challenge facing humanity. Environmental policies
        and sustainable practices are becoming increasingly important.
        """
        
        themes = await bias_engine._extract_narrative_themes(long_text)
        
        assert isinstance(themes, list)
        if themes:  # Only test if themes are extracted
            assert all(isinstance(t, NarrativeTheme) for t in themes)
            assert all(hasattr(t, 'keywords') for t in themes)
            assert all(hasattr(t, 'coherence_score') for t in themes)
            assert all(t.coherence_score >= 0.0 for t in themes)
    
    @pytest.mark.asyncio
    async def test_analyze_source_bias(self, bias_engine, biased_text_samples):
        """Test source bias analysis."""
        source_analysis = await bias_engine._analyze_source_bias(
            biased_text_samples["source_bias"],
            "test-domain.com"
        )
        
        assert isinstance(source_analysis, dict)
        assert "authority_appeals_count" in source_analysis
        assert "credibility_boosters_count" in source_analysis
        assert "uncertainty_markers_count" in source_analysis
        assert "attribution_ratio" in source_analysis
        
        # Should detect authority appeals and credibility boosters
        assert source_analysis["authority_appeals_count"] > 0
        assert source_analysis["credibility_boosters_count"] > 0
    
    @pytest.mark.asyncio
    async def test_calculate_bias_indicators(self, bias_engine, biased_text_samples):
        """Test bias indicator calculation."""
        # First get the components needed
        framing_patterns = await bias_engine._detect_framing_patterns(biased_text_samples["complex_bias"])
        linguistic_patterns = await bias_engine._analyze_linguistic_patterns(biased_text_samples["complex_bias"])
        source_bias = await bias_engine._analyze_source_bias(biased_text_samples["complex_bias"], None)
        
        indicators = await bias_engine._calculate_bias_indicators(
            biased_text_samples["complex_bias"],
            framing_patterns,
            linguistic_patterns,
            source_bias
        )
        
        assert isinstance(indicators, list)
        if indicators:  # Only test if indicators are found
            assert all(isinstance(i, BiasIndicator) for i in indicators)
            assert all(hasattr(i, 'type') for i in indicators)
            assert all(hasattr(i, 'confidence') for i in indicators)
            assert all(hasattr(i, 'severity') for i in indicators)
            assert all(i.confidence >= 0.0 and i.confidence <= 1.0 for i in indicators)
    
    def test_calculate_overall_bias_score(self, bias_engine):
        """Test overall bias score calculation."""
        # Create mock bias indicators
        indicators = [
            BiasIndicator(
                type="loaded_language",
                description="Test",
                confidence=0.8,
                evidence=["test"],
                severity="high"
            ),
            BiasIndicator(
                type="framing_bias",
                description="Test",
                confidence=0.6,
                evidence=["test"],
                severity="medium"
            )
        ]
        
        score = bias_engine._calculate_overall_bias_score(indicators, [], {})
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score > 0.0  # Should be positive with bias indicators
    
    def test_calculate_analysis_confidence(self, bias_engine):
        """Test analysis confidence calculation."""
        indicators = [
            BiasIndicator("test", "test", 0.8, ["test"], "high"),
            BiasIndicator("test", "test", 0.6, ["test"], "medium")
        ]
        patterns = [
            FramingPattern("test", "test", "test", "positive", 0.7)
        ]
        themes = [
            NarrativeTheme(1, ["test"], 0.5, 0.3, "test")
        ]
        
        confidence = bias_engine._calculate_analysis_confidence(indicators, patterns, themes)
        
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
    
    def test_calculate_pattern_confidence(self, bias_engine):
        """Test pattern confidence calculation."""
        # Mock span object
        mock_span = Mock()
        mock_span.text = "victim of attack"
        
        confidence = bias_engine._calculate_pattern_confidence(mock_span, "victim_framing")
        
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
    
    def test_get_token_depth(self, bias_engine):
        """Test token depth calculation."""
        # Mock token with head relationship
        mock_token = Mock()
        mock_head = Mock()
        mock_head.head = mock_head  # Self-reference to stop recursion
        mock_token.head = mock_head
        
        depth = bias_engine._get_token_depth(mock_token)
        
        assert isinstance(depth, int)
        assert depth >= 0
    
    def test_preprocess_for_topics(self, bias_engine):
        """Test text preprocessing for topic modeling."""
        text = "This is a test sentence. Another sentence here! And a third one?"
        
        processed = bias_engine._preprocess_for_topics(text)
        
        assert isinstance(processed, list)
        assert all(isinstance(s, str) for s in processed)
        assert all(len(s) > 0 for s in processed)
    
    @pytest.mark.asyncio
    async def test_bias_analysis_error_handling(self, bias_engine):
        """Test error handling in bias analysis."""
        # Test with empty text
        result = await bias_engine.analyze_bias("")
        
        assert isinstance(result, BiasAnalysisResult)
        assert result.overall_bias_score >= 0.0
        assert result.confidence >= 0.0
    
    @pytest.mark.asyncio
    async def test_bias_analysis_no_spacy_model(self):
        """Test bias analysis when spaCy model is not available."""
        engine = BiasDetectionEngine()
        engine.nlp = None
        engine.matcher = None
        
        result = await engine.analyze_bias("Test text with some bias indicators.")
        
        assert isinstance(result, BiasAnalysisResult)
        # Should still work with limited functionality
        assert result.confidence >= 0.0


class TestBiasAnalysisIntegration:
    """Integration tests for bias analysis."""
    
    @pytest.mark.asyncio
    async def test_analyze_text_bias_function(self):
        """Test the convenience function for bias analysis."""
        with patch('src.thalamus.bias_analysis.get_bias_engine') as mock_get_engine:
            mock_engine = Mock()
            mock_result = BiasAnalysisResult(
                overall_bias_score=0.7,
                bias_indicators=[],
                framing_patterns=[],
                narrative_themes=[],
                source_bias_indicators={},
                linguistic_patterns={},
                confidence=0.8
            )
            from unittest.mock import AsyncMock
            mock_engine.analyze_bias = AsyncMock(return_value=mock_result)
            mock_get_engine.return_value = mock_engine
            
            result = await analyze_text_bias("Test text", "test-domain.com")
            
            assert isinstance(result, BiasAnalysisResult)
            mock_engine.analyze_bias.assert_called_once()
    
    def test_get_bias_engine_singleton(self):
        """Test that get_bias_engine returns singleton instance."""
        # Reset global instance
        import src.thalamus.bias_analysis as bias_module
        bias_module._bias_engine = None
        
        engine1 = get_bias_engine()
        engine2 = get_bias_engine()
        
        assert engine1 is engine2
        assert isinstance(engine1, BiasDetectionEngine)


class TestBiasAnalysisDataClasses:
    """Test data classes used in bias analysis."""
    
    def test_bias_indicator_creation(self):
        """Test BiasIndicator data class."""
        indicator = BiasIndicator(
            type="loaded_language",
            description="Test description",
            confidence=0.8,
            evidence=["evidence1", "evidence2"],
            severity="high"
        )
        
        assert indicator.type == "loaded_language"
        assert indicator.description == "Test description"
        assert indicator.confidence == 0.8
        assert indicator.evidence == ["evidence1", "evidence2"]
        assert indicator.severity == "high"
    
    def test_framing_pattern_creation(self):
        """Test FramingPattern data class."""
        pattern = FramingPattern(
            pattern_type="victim_framing",
            matched_text="victim of attack",
            context="The victim of attack was hospitalized",
            bias_direction="negative",
            confidence=0.9
        )
        
        assert pattern.pattern_type == "victim_framing"
        assert pattern.matched_text == "victim of attack"
        assert pattern.context == "The victim of attack was hospitalized"
        assert pattern.bias_direction == "negative"
        assert pattern.confidence == 0.9
    
    def test_narrative_theme_creation(self):
        """Test NarrativeTheme data class."""
        theme = NarrativeTheme(
            theme_id=1,
            keywords=["technology", "innovation", "future"],
            coherence_score=0.7,
            prevalence=0.4,
            description="Theme about technology and innovation"
        )
        
        assert theme.theme_id == 1
        assert theme.keywords == ["technology", "innovation", "future"]
        assert theme.coherence_score == 0.7
        assert theme.prevalence == 0.4
        assert theme.description == "Theme about technology and innovation"
    
    def test_bias_analysis_result_creation(self):
        """Test BiasAnalysisResult data class."""
        result = BiasAnalysisResult(
            overall_bias_score=0.6,
            bias_indicators=[],
            framing_patterns=[],
            narrative_themes=[],
            source_bias_indicators={},
            linguistic_patterns={},
            confidence=0.8
        )
        
        assert result.overall_bias_score == 0.6
        assert result.bias_indicators == []
        assert result.framing_patterns == []
        assert result.narrative_themes == []
        assert result.source_bias_indicators == {}
        assert result.linguistic_patterns == {}
        assert result.confidence == 0.8


class TestBiasAnalysisPerformance:
    """Performance tests for bias analysis."""
    
    @pytest.mark.asyncio
    async def test_bias_analysis_performance(self):
        """Test bias analysis performance with large text."""
        import time
        
        # Create a large text sample
        large_text = """
        Technology is rapidly transforming our world in unprecedented ways. 
        Artificial intelligence and machine learning are revolutionizing industries 
        across the globe. Companies are investing billions in AI research and development.
        """ * 50  # Repeat to make it larger
        
        engine = BiasDetectionEngine()
        
        start_time = time.time()
        result = await engine.analyze_bias(large_text)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        assert isinstance(result, BiasAnalysisResult)
        assert processing_time < 10.0  # Should complete within 10 seconds
        assert result.confidence >= 0.0
    
    @pytest.mark.asyncio
    async def test_concurrent_bias_analysis(self):
        """Test concurrent bias analysis."""
        engine = BiasDetectionEngine()
        
        texts = [
            "This devastating policy will destroy everything.",
            "The research shows positive results.",
            "Some say this might be problematic.",
            "Experts unanimously agree this is terrible."
        ]
        
        # Run concurrent analysis
        tasks = [engine.analyze_bias(text) for text in texts]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == len(texts)
        assert all(isinstance(r, BiasAnalysisResult) for r in results)
        assert all(r.confidence >= 0.0 for r in results)


class TestBiasAnalysisEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.asyncio
    async def test_empty_text_analysis(self):
        """Test analysis with empty text."""
        engine = BiasDetectionEngine()
        
        result = await engine.analyze_bias("")
        
        assert isinstance(result, BiasAnalysisResult)
        assert result.overall_bias_score == 0.0
        assert len(result.bias_indicators) == 0
    
    @pytest.mark.asyncio
    async def test_very_short_text_analysis(self):
        """Test analysis with very short text."""
        engine = BiasDetectionEngine()
        
        result = await engine.analyze_bias("Hi.")
        
        assert isinstance(result, BiasAnalysisResult)
        assert result.confidence >= 0.0
    
    @pytest.mark.asyncio
    async def test_special_characters_text(self):
        """Test analysis with special characters."""
        engine = BiasDetectionEngine()
        
        text_with_special_chars = "This is a test!!! @#$%^&*() with special characters... 😀"
        result = await engine.analyze_bias(text_with_special_chars)
        
        assert isinstance(result, BiasAnalysisResult)
        assert result.confidence >= 0.0
    
    @pytest.mark.asyncio
    async def test_non_english_text(self):
        """Test analysis with non-English text."""
        engine = BiasDetectionEngine()
        
        # Spanish text
        spanish_text = "Esta es una prueba en español con algunas palabras."
        result = await engine.analyze_bias(spanish_text)
        
        assert isinstance(result, BiasAnalysisResult)
        # Should handle gracefully even if not optimized for Spanish
        assert result.confidence >= 0.0
    
    @pytest.mark.asyncio
    async def test_extremely_biased_text(self):
        """Test analysis with extremely biased text."""
        engine = BiasDetectionEngine()
        
        extremely_biased = """
        This absolutely devastating and completely catastrophic policy will totally destroy 
        everything we hold dear. The terrible victims of this outrageous attack suffered 
        unimaginable trauma. Some say this crisis threatens our very existence. 
        Critics argue this is the worst disaster in history. Sources claim this will 
        never work and always fails completely.
        """
        
        result = await engine.analyze_bias(extremely_biased)
        
        assert isinstance(result, BiasAnalysisResult)
        assert result.overall_bias_score > 0.3  # Should detect significant bias
        assert len(result.bias_indicators) > 0
        assert result.confidence > 0.0


if __name__ == "__main__":
    pytest.main([__file__])