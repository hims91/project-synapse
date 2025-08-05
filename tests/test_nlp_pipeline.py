"""
Unit tests for Thalamus NLP Pipeline

Tests all three processing tiers:
1. Rule-based processing with spaCy
2. TextRank algorithm for extractive summarization
3. Hybrid TF-IDF + sentence position analysis
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import numpy as np

from src.thalamus.nlp_pipeline import (
    NLPPipeline,
    ProcessingTier,
    SentimentPolarity,
    ProcessingResult,
    TextRankNode,
    SummaryCandidate,
    get_nlp_pipeline,
    process_article_content
)
from src.shared.schemas import NLPAnalysis, EntityExtraction


class TestNLPPipeline:
    """Test cases for NLP Pipeline functionality."""
    
    @pytest.fixture
    def sample_text(self):
        """Sample text for testing."""
        return """
        Artificial intelligence is transforming the world. Machine learning algorithms 
        are becoming more sophisticated every day. Companies like Google and Microsoft 
        are investing heavily in AI research. The future of technology looks very promising.
        However, there are also concerns about job displacement and privacy issues.
        """
    
    @pytest.fixture
    def sample_title(self):
        """Sample title for testing."""
        return "AI Revolution: The Future of Technology"
    
    @pytest.fixture
    def nlp_pipeline(self):
        """Create NLP pipeline instance for testing."""
        return NLPPipeline()
    
    def create_mock_spacy_doc(self):
        """Create mock spaCy document for testing."""
        mock_doc = Mock()
        
        # Mock entities
        mock_entity1 = Mock()
        mock_entity1.text = "Google"
        mock_entity1.label_ = "ORG"
        mock_entity1.start_char = 100
        mock_entity1.end_char = 106
        
        mock_entity2 = Mock()
        mock_entity2.text = "Microsoft"
        mock_entity2.label_ = "ORG"
        mock_entity2.start_char = 111
        mock_entity2.end_char = 120
        
        mock_doc.ents = [mock_entity1, mock_entity2]
        
        # Mock sentences
        mock_sent1 = Mock()
        mock_sent1.text = "Artificial intelligence is transforming the world."
        mock_sent2 = Mock()
        mock_sent2.text = "Machine learning algorithms are becoming more sophisticated."
        mock_doc.sents = [mock_sent1, mock_sent2]
        
        # Mock tokens
        mock_tokens = []
        for word in ["Artificial", "intelligence", "is", "transforming", "the", "world"]:
            token = Mock()
            token.text = word
            token.is_space = False
            token.is_alpha = True
            token.is_stop = word.lower() in ["is", "the"]
            token.lemma_ = word.lower()
            mock_tokens.append(token)
        mock_doc.__iter__ = lambda self: iter(mock_tokens)
        
        return mock_doc    

    def test_nlp_pipeline_initialization(self, nlp_pipeline):
        """Test NLP pipeline initialization."""
        assert nlp_pipeline is not None
        assert nlp_pipeline.max_summary_sentences == 3
        assert nlp_pipeline.textrank_iterations == 30
        assert nlp_pipeline.textrank_damping == 0.85
        assert nlp_pipeline.min_sentence_length == 10
        assert nlp_pipeline.max_sentence_length == 500
    
    @pytest.mark.asyncio
    async def test_process_text_all_tiers(self, nlp_pipeline, sample_text, sample_title):
        """Test processing text through all tiers."""
        with patch.object(nlp_pipeline, 'nlp') as mock_nlp:
            # Mock spaCy processing
            mock_doc = self.create_mock_spacy_doc()
            mock_nlp.return_value = mock_doc
            
            result = await nlp_pipeline.process_text(sample_text, sample_title)
            
            assert isinstance(result, NLPAnalysis)
            assert result.processing_time > 0
            assert len(result.processing_tiers) > 0
            assert result.sentiment is not None
            assert isinstance(result.entities, list)
    
    @pytest.mark.asyncio
    async def test_process_rule_based_tier(self, nlp_pipeline, sample_text):
        """Test rule-based processing tier."""
        with patch.object(nlp_pipeline, 'nlp') as mock_nlp:
            mock_doc = self.create_mock_spacy_doc()
            mock_nlp.return_value = mock_doc
            
            result = await nlp_pipeline._process_rule_based(sample_text)
            
            assert result.tier == ProcessingTier.RULE_BASED
            assert result.success is True
            assert result.processing_time >= 0
            assert result.data is not None
            assert "entities" in result.data
            assert "sentiment_score" in result.data
            assert len(result.data["entities"]) == 2
    
    @pytest.mark.asyncio
    async def test_process_rule_based_no_spacy(self, nlp_pipeline, sample_text):
        """Test rule-based processing when spaCy is not available."""
        nlp_pipeline.nlp = None
        
        result = await nlp_pipeline._process_rule_based(sample_text)
        
        assert result.tier == ProcessingTier.RULE_BASED
        assert result.success is False
        assert result.error_message == "spaCy model not available"
    
    @pytest.mark.asyncio
    async def test_process_textrank_tier(self, nlp_pipeline, sample_text):
        """Test TextRank processing tier."""
        result = await nlp_pipeline._process_textrank(sample_text)
        
        assert result.tier == ProcessingTier.TEXTRANK
        assert result.success is True
        assert result.processing_time > 0
        assert result.data is not None
        assert "summary_sentences" in result.data
        assert "sentence_scores" in result.data
        assert result.data["algorithm"] == "textrank"
    
    @pytest.mark.asyncio
    async def test_process_textrank_insufficient_sentences(self, nlp_pipeline):
        """Test TextRank with insufficient sentences."""
        short_text = "Short text."
        
        result = await nlp_pipeline._process_textrank(short_text)
        
        assert result.tier == ProcessingTier.TEXTRANK
        assert result.success is False
        assert "Insufficient sentences" in result.error_message
    
    @pytest.mark.asyncio
    async def test_process_hybrid_tfidf_tier(self, nlp_pipeline, sample_text, sample_title):
        """Test hybrid TF-IDF processing tier."""
        result = await nlp_pipeline._process_hybrid_tfidf(sample_text, sample_title)
        
        assert result.tier == ProcessingTier.HYBRID_TFIDF
        assert result.success is True
        assert result.processing_time > 0
        assert result.data is not None
        assert "summary_sentences" in result.data
        assert "top_terms" in result.data
        assert result.data["algorithm"] == "hybrid_tfidf"
        assert "score_breakdown" in result.data
    
    @pytest.mark.asyncio
    async def test_process_hybrid_tfidf_insufficient_sentences(self, nlp_pipeline):
        """Test hybrid TF-IDF with insufficient sentences."""
        short_text = "Short."
        
        result = await nlp_pipeline._process_hybrid_tfidf(short_text)
        
        assert result.tier == ProcessingTier.HYBRID_TFIDF
        assert result.success is False
        assert "Insufficient sentences" in result.error_message
    
    def test_calculate_rule_based_sentiment_positive(self, nlp_pipeline):
        """Test rule-based sentiment calculation for positive text."""
        with patch.object(nlp_pipeline, 'nlp') as mock_nlp:
            # Create mock doc with positive words
            mock_doc = Mock()
            positive_tokens = []
            for word in ["great", "excellent", "wonderful", "success"]:
                token = Mock()
                token.text = word
                token.is_alpha = True
                token.is_stop = False
                positive_tokens.append(token)
            mock_doc.__iter__ = lambda self: iter(positive_tokens)
            
            sentiment = nlp_pipeline._calculate_rule_based_sentiment(mock_doc)
            
            assert sentiment > 0
            assert -1.0 <= sentiment <= 1.0
    
    def test_calculate_rule_based_sentiment_negative(self, nlp_pipeline):
        """Test rule-based sentiment calculation for negative text."""
        with patch.object(nlp_pipeline, 'nlp') as mock_nlp:
            # Create mock doc with negative words
            mock_doc = Mock()
            negative_tokens = []
            for word in ["terrible", "awful", "horrible", "failure"]:
                token = Mock()
                token.text = word
                token.is_alpha = True
                token.is_stop = False
                negative_tokens.append(token)
            mock_doc.__iter__ = lambda self: iter(negative_tokens)
            
            sentiment = nlp_pipeline._calculate_rule_based_sentiment(mock_doc)
            
            assert sentiment < 0
            assert -1.0 <= sentiment <= 1.0
    
    def test_split_sentences(self, nlp_pipeline, sample_text):
        """Test sentence splitting functionality."""
        with patch.object(nlp_pipeline, 'nlp') as mock_nlp:
            # Mock spaCy sentence splitting
            mock_doc = Mock()
            sentences = [
                "Artificial intelligence is transforming the world.",
                "Machine learning algorithms are becoming more sophisticated every day.",
                "Companies like Google and Microsoft are investing heavily in AI research."
            ]
            mock_sents = []
            for sent in sentences:
                mock_sent = Mock()
                mock_sent.text = sent
                mock_sents.append(mock_sent)
            mock_doc.sents = mock_sents
            mock_nlp.return_value = mock_doc
            
            result = nlp_pipeline._split_sentences(sample_text)
            
            assert isinstance(result, list)
            assert len(result) > 0
            assert all(isinstance(sent, str) for sent in result)
            assert all(len(sent) >= nlp_pipeline.min_sentence_length for sent in result)
    
    def test_split_sentences_fallback(self, nlp_pipeline, sample_text):
        """Test sentence splitting fallback when spaCy is not available."""
        nlp_pipeline.nlp = None
        
        result = nlp_pipeline._split_sentences(sample_text)
        
        assert isinstance(result, list)
        assert len(result) > 0
    
    def test_tokenize_sentence(self, nlp_pipeline):
        """Test sentence tokenization."""
        sentence = "Artificial intelligence is transforming the world rapidly."
        
        with patch.object(nlp_pipeline, 'nlp') as mock_nlp:
            # Mock spaCy tokenization
            mock_doc = Mock()
            tokens = []
            words = ["artificial", "intelligence", "transforming", "world", "rapidly"]
            for word in words:
                token = Mock()
                token.lemma_ = word
                token.is_alpha = True
                token.is_stop = False
                token.text = word
                tokens.append(token)
            mock_doc.__iter__ = lambda self: iter(tokens)
            mock_nlp.return_value = mock_doc
            
            result = nlp_pipeline._tokenize_sentence(sentence)
            
            assert isinstance(result, list)
            assert len(result) > 0
            assert all(isinstance(token, str) for token in result)
            assert all(len(token) > 2 for token in result)
    
    def test_tokenize_sentence_fallback(self, nlp_pipeline):
        """Test sentence tokenization fallback when spaCy is not available."""
        nlp_pipeline.nlp = None
        sentence = "Artificial intelligence is transforming the world rapidly."
        
        result = nlp_pipeline._tokenize_sentence(sentence)
        
        assert isinstance(result, list)
        assert len(result) > 0
    
    def test_build_similarity_matrix(self, nlp_pipeline):
        """Test similarity matrix building for TextRank."""
        nodes = [
            TextRankNode("First sentence about AI.", 0, tokens=["first", "sentence", "ai"]),
            TextRankNode("Second sentence about machine learning.", 1, tokens=["second", "sentence", "machine", "learning"]),
            TextRankNode("Third sentence about AI and machine learning.", 2, tokens=["third", "sentence", "ai", "machine", "learning"])
        ]
        
        matrix = nlp_pipeline._build_similarity_matrix(nodes)
        
        assert matrix.shape == (3, 3)
        assert np.all(np.diag(matrix) == 0)  # Diagonal should be zero
        assert np.all(matrix >= 0)  # All similarities should be non-negative
        assert np.all(matrix <= 1)  # All similarities should be <= 1
        # Sentences with common words should have higher similarity
        assert matrix[0][2] > 0  # Both have "ai"
        assert matrix[1][2] > 0  # Both have "machine", "learning"
    
    def test_run_textrank_algorithm(self, nlp_pipeline):
        """Test TextRank algorithm execution."""
        # Create a simple similarity matrix
        similarity_matrix = np.array([
            [0.0, 0.3, 0.5],
            [0.3, 0.0, 0.7],
            [0.5, 0.7, 0.0]
        ])
        
        scores = nlp_pipeline._run_textrank(similarity_matrix)
        
        assert len(scores) == 3
        assert all(score > 0 for score in scores)
        assert abs(sum(scores) - 1.0) < 0.1  # Scores should roughly sum to 1
    
    def test_calculate_position_score(self, nlp_pipeline):
        """Test position-based scoring."""
        total_sentences = 10
        
        # First sentence should have high score
        first_score = nlp_pipeline._calculate_position_score(0, total_sentences)
        assert first_score > 0.9  # Should be 1.0 for first position
        
        # Last sentence should have high score
        last_score = nlp_pipeline._calculate_position_score(total_sentences - 1, total_sentences)
        assert last_score > 0.9  # Should be 1.0 for last position
        
        # Middle sentences should have lower scores
        middle_score = nlp_pipeline._calculate_position_score(total_sentences // 2, total_sentences)
        assert middle_score < first_score
        assert middle_score < last_score
        assert middle_score >= 0.0  # Should be 0.0 for exact middle
    
    def test_calculate_length_score(self, nlp_pipeline):
        """Test length-based scoring."""
        # Optimal length sentence (10-25 words)
        optimal_sentence = " ".join(["word"] * 15)
        optimal_score = nlp_pipeline._calculate_length_score(optimal_sentence)
        assert optimal_score == 1.0
        
        # Short sentence
        short_sentence = " ".join(["word"] * 5)
        short_score = nlp_pipeline._calculate_length_score(short_sentence)
        assert short_score < 1.0
        
        # Long sentence
        long_sentence = " ".join(["word"] * 50)
        long_score = nlp_pipeline._calculate_length_score(long_sentence)
        assert long_score < 1.0
    
    def test_extract_top_terms(self, nlp_pipeline):
        """Test top terms extraction from TF-IDF matrix."""
        # Create mock TF-IDF matrix and feature names
        tfidf_matrix = np.array([[0.5, 0.3, 0.8], [0.2, 0.9, 0.1]])
        feature_names = np.array(["artificial", "intelligence", "machine"])
        
        top_terms = nlp_pipeline._extract_top_terms(tfidf_matrix, feature_names, top_k=2)
        
        assert len(top_terms) == 2
        assert all(isinstance(term, tuple) and len(term) == 2 for term in top_terms)
        assert all(isinstance(term[0], str) and isinstance(term[1], (int, float)) for term in top_terms)
        # Should be sorted by score (descending)
        assert top_terms[0][1] >= top_terms[1][1]
    
    def test_combine_results(self, nlp_pipeline, sample_text, sample_title):
        """Test combining results from multiple tiers."""
        # Create mock results
        rule_result = ProcessingResult(
            tier=ProcessingTier.RULE_BASED,
            success=True,
            processing_time=0.1,
            data={
                "entities": [{"text": "Google", "label": "ORG", "start": 0, "end": 6, "confidence": 1.0}],
                "sentiment_score": 0.3,
                "sentence_count": 5,
                "token_count": 50
            }
        )
        
        textrank_result = ProcessingResult(
            tier=ProcessingTier.TEXTRANK,
            success=True,
            processing_time=0.2,
            data={
                "summary_sentences": ["First summary sentence.", "Second summary sentence."],
                "algorithm": "textrank"
            }
        )
        
        results = {
            ProcessingTier.RULE_BASED.value: rule_result,
            ProcessingTier.TEXTRANK.value: textrank_result
        }
        
        analysis = nlp_pipeline._combine_results(results, sample_text, sample_title)
        
        assert isinstance(analysis, NLPAnalysis)
        assert len(analysis.entities) == 1
        assert analysis.sentiment == 0.3
        assert len(analysis.summary.split()) > 0
        assert len(analysis.processing_tiers) == 2
        assert abs(analysis.processing_time - 0.3) < 0.001
        assert "sentiment_polarity" in analysis.metadata


class TestNLPPipelineIntegration:
    """Integration tests for NLP Pipeline."""
    
    @pytest.mark.asyncio
    async def test_process_article_content_function(self):
        """Test the convenience function for processing article content."""
        sample_text = "Test article content"
        sample_title = "Test Title"
        
        with patch('src.thalamus.nlp_pipeline.get_nlp_pipeline') as mock_get_pipeline:
            mock_pipeline = Mock()
            mock_analysis = NLPAnalysis(
                sentiment=0.2,
                entities=[],
                categories=[],
                significance=0.7,
                summary="Test summary",
                processing_tiers=["rule_based"],
                processing_time=0.1,
                metadata={}
            )
            mock_pipeline.process_text = AsyncMock(return_value=mock_analysis)
            mock_get_pipeline.return_value = mock_pipeline
            
            result = await process_article_content(sample_text, sample_title)
            
            assert isinstance(result, NLPAnalysis)
            mock_pipeline.process_text.assert_called_once_with(sample_text, sample_title)
    
    def test_get_nlp_pipeline_singleton(self):
        """Test that get_nlp_pipeline returns singleton instance."""
        # Reset global instance
        import src.thalamus.nlp_pipeline as nlp_module
        nlp_module._nlp_pipeline = None
        
        pipeline1 = get_nlp_pipeline()
        pipeline2 = get_nlp_pipeline()
        
        assert pipeline1 is pipeline2
        assert isinstance(pipeline1, NLPPipeline)


class TestNLPPipelineErrorHandling:
    """Test error handling in NLP Pipeline."""
    
    @pytest.mark.asyncio
    async def test_process_text_with_exceptions(self):
        """Test processing text when tiers raise exceptions."""
        pipeline = NLPPipeline()
        
        with patch.object(pipeline, '_process_rule_based', side_effect=Exception("Rule-based error")):
            with patch.object(pipeline, '_process_textrank', side_effect=Exception("TextRank error")):
                with patch.object(pipeline, '_process_hybrid_tfidf', side_effect=Exception("Hybrid error")):
                    result = await pipeline.process_text("Test text")
                    
                    assert isinstance(result, NLPAnalysis)
                    # Should still return a valid analysis even with all failures
                    assert len(result.processing_tiers) == 0
    
    @pytest.mark.asyncio
    async def test_process_text_partial_failures(self):
        """Test processing text with some tier failures."""
        pipeline = NLPPipeline()
        sample_text = "Test text for processing"
        
        # Mock successful rule-based processing
        with patch.object(pipeline, '_process_rule_based') as mock_rule:
            mock_rule.return_value = ProcessingResult(
                tier=ProcessingTier.RULE_BASED,
                success=True,
                processing_time=0.1,
                data={"entities": [], "sentiment_score": 0.0, "sentence_count": 3}
            )
            
            # Mock failed TextRank processing
            with patch.object(pipeline, '_process_textrank') as mock_textrank:
                mock_textrank.return_value = ProcessingResult(
                    tier=ProcessingTier.TEXTRANK,
                    success=False,
                    processing_time=0.0,
                    error_message="TextRank failed"
                )
                
                result = await pipeline.process_text(sample_text)
                
                assert isinstance(result, NLPAnalysis)
                assert len(result.processing_tiers) == 1
                assert ProcessingTier.RULE_BASED.value in result.processing_tiers


class TestDataClasses:
    """Test data classes used in NLP Pipeline."""
    
    def test_processing_result_creation(self):
        """Test ProcessingResult data class."""
        result = ProcessingResult(
            tier=ProcessingTier.RULE_BASED,
            success=True,
            processing_time=0.5,
            data={"test": "data"}
        )
        
        assert result.tier == ProcessingTier.RULE_BASED
        assert result.success is True
        assert result.processing_time == 0.5
        assert result.data == {"test": "data"}
        assert result.error_message is None
    
    def test_textrank_node_creation(self):
        """Test TextRankNode data class."""
        node = TextRankNode(
            sentence="Test sentence.",
            index=0,
            score=0.8,
            tokens=["test", "sentence"]
        )
        
        assert node.sentence == "Test sentence."
        assert node.index == 0
        assert node.score == 0.8
        assert node.tokens == ["test", "sentence"]
    
    def test_summary_candidate_creation(self):
        """Test SummaryCandidate data class."""
        candidate = SummaryCandidate(
            sentence="Test sentence.",
            index=0,
            tfidf_score=0.5,
            position_score=0.8,
            length_score=0.9,
            combined_score=0.7
        )
        
        assert candidate.sentence == "Test sentence."
        assert candidate.index == 0
        assert candidate.tfidf_score == 0.5
        assert candidate.position_score == 0.8
        assert candidate.length_score == 0.9
        assert candidate.combined_score == 0.7


class TestEnums:
    """Test enums used in NLP Pipeline."""
    
    def test_processing_tier_enum(self):
        """Test ProcessingTier enum."""
        assert ProcessingTier.RULE_BASED.value == "rule_based"
        assert ProcessingTier.TEXTRANK.value == "textrank"
        assert ProcessingTier.HYBRID_TFIDF.value == "hybrid_tfidf"
    
    def test_sentiment_polarity_enum(self):
        """Test SentimentPolarity enum."""
        assert SentimentPolarity.POSITIVE.value == "positive"
        assert SentimentPolarity.NEGATIVE.value == "negative"
        assert SentimentPolarity.NEUTRAL.value == "neutral"


class TestAdvancedNLPFeatures:
    """Test advanced NLP features including custom entities and sentiment analysis."""
    
    @pytest.fixture
    def nlp_pipeline_with_custom_entities(self):
        """Create NLP pipeline with custom entity patterns."""
        return NLPPipeline()
    
    def test_custom_entity_extraction(self, nlp_pipeline_with_custom_entities):
        """Test custom entity pattern matching."""
        text = "Bitcoin and Ethereum are popular cryptocurrencies. ChatGPT is an AI tool from OpenAI."
        
        with patch.object(nlp_pipeline_with_custom_entities, 'nlp') as mock_nlp:
            # Create mock doc
            mock_doc = Mock()
            mock_doc.ents = []  # No standard entities
            
            # Mock matcher results
            with patch.object(nlp_pipeline_with_custom_entities, 'matcher') as mock_matcher:
                # Mock matches for custom entities
                mock_matches = [
                    (123, 0, 1),  # Bitcoin match
                    (456, 2, 3),  # ChatGPT match
                ]
                mock_matcher.return_value = mock_matches
                
                # Mock vocab strings
                mock_nlp.vocab.strings = {123: "CRYPTO_COIN_0", 456: "AI_TOOL_0"}
                
                # Mock spans
                mock_span1 = Mock()
                mock_span1.text = "Bitcoin"
                mock_span1.start_char = 0
                mock_span1.end_char = 7
                
                mock_span2 = Mock()
                mock_span2.text = "ChatGPT"
                mock_span2.start_char = 50
                mock_span2.end_char = 57
                
                mock_doc.__getitem__ = Mock(side_effect=[mock_span1, mock_span2])
                mock_nlp.return_value = mock_doc
                
                custom_entities = nlp_pipeline_with_custom_entities._extract_custom_entities(mock_doc)
                
                assert len(custom_entities) == 2
                assert custom_entities[0]["label"] == "CRYPTO_COIN"
                assert custom_entities[0]["text"] == "Bitcoin"
                assert custom_entities[1]["label"] == "AI_TOOL"
                assert custom_entities[1]["text"] == "ChatGPT"
    
    def test_advanced_sentiment_analysis(self, nlp_pipeline_with_custom_entities):
        """Test advanced sentiment analysis with multiple methods."""
        text = "This is a wonderful and amazing product that I absolutely love!"
        
        with patch.object(nlp_pipeline_with_custom_entities, 'nlp') as mock_nlp:
            mock_doc = Mock()
            mock_tokens = []
            for word in ["wonderful", "amazing", "love"]:
                token = Mock()
                token.text = word
                token.is_alpha = True
                token.is_stop = False
                mock_tokens.append(token)
            mock_doc.__iter__ = lambda self: iter(mock_tokens)
            
            # Mock VADER analyzer
            if hasattr(nlp_pipeline_with_custom_entities, 'vader_analyzer') and nlp_pipeline_with_custom_entities.vader_analyzer:
                with patch.object(nlp_pipeline_with_custom_entities.vader_analyzer, 'polarity_scores') as mock_vader:
                    mock_vader.return_value = {
                        'compound': 0.8,
                        'pos': 0.7,
                        'neg': 0.1,
                        'neu': 0.2
                    }
                    
                    sentiment_result = nlp_pipeline_with_custom_entities._calculate_advanced_sentiment(text, mock_doc)
                    
                    assert sentiment_result["compound_score"] > 0
                    assert "vader" in sentiment_result["methods_used"]
                    assert sentiment_result["confidence"] > 0
    
    def test_content_category_classification(self, nlp_pipeline_with_custom_entities):
        """Test content category classification."""
        # Technology text
        tech_text = "Artificial intelligence and machine learning are transforming software development."
        tech_entities = [{"text": "AI", "label": "AI_TOOL"}]
        
        categories = nlp_pipeline_with_custom_entities._classify_content_categories(tech_text, tech_entities)
        assert "Technology" in categories
        
        # Finance text
        finance_text = "Bitcoin price surged as cryptocurrency markets rallied."
        finance_entities = [{"text": "Bitcoin", "label": "CRYPTO_COIN"}]
        
        categories = nlp_pipeline_with_custom_entities._classify_content_categories(finance_text, finance_entities)
        assert "Finance" in categories
        
        # General text
        general_text = "The weather is nice today."
        general_entities = []
        
        categories = nlp_pipeline_with_custom_entities._classify_content_categories(general_text, general_entities)
        assert "General" in categories
    
    @pytest.mark.asyncio
    async def test_enhanced_rule_based_processing(self, nlp_pipeline_with_custom_entities):
        """Test enhanced rule-based processing with all features."""
        text = "OpenAI's ChatGPT is an amazing AI tool that revolutionizes technology."
        
        with patch.object(nlp_pipeline_with_custom_entities, 'nlp') as mock_nlp:
            # Mock spaCy doc
            mock_doc = Mock()
            
            # Mock standard entities
            mock_entity = Mock()
            mock_entity.text = "OpenAI"
            mock_entity.label_ = "ORG"
            mock_entity.start_char = 0
            mock_entity.end_char = 6
            mock_doc.ents = [mock_entity]
            
            # Mock sentences and tokens
            mock_sent = Mock()
            mock_sent.text = text
            mock_doc.sents = [mock_sent]
            
            mock_tokens = []
            for word in ["amazing", "revolutionizes"]:
                token = Mock()
                token.text = word
                token.is_space = False
                token.is_alpha = True
                token.is_stop = False
                mock_tokens.append(token)
            mock_doc.__iter__ = lambda self: iter(mock_tokens)
            
            mock_nlp.return_value = mock_doc
            
            # Mock matcher to enable custom entity extraction
            nlp_pipeline_with_custom_entities.matcher = Mock()
            
            # Mock custom entity extraction
            with patch.object(nlp_pipeline_with_custom_entities, '_extract_custom_entities') as mock_custom:
                mock_custom.return_value = [{"text": "ChatGPT", "label": "AI_TOOL", "start": 8, "end": 15, "confidence": 0.9}]
                
                # Mock advanced sentiment
                with patch.object(nlp_pipeline_with_custom_entities, '_calculate_advanced_sentiment') as mock_sentiment:
                    mock_sentiment.return_value = {
                        "compound_score": 0.6,
                        "positive": 0.7,
                        "negative": 0.1,
                        "neutral": 0.2,
                        "confidence": 0.8,
                        "methods_used": ["rule_based", "vader"]
                    }
                    
                    # Mock category classification
                    with patch.object(nlp_pipeline_with_custom_entities, '_classify_content_categories') as mock_categories:
                        mock_categories.return_value = ["Technology"]
                        
                        result = await nlp_pipeline_with_custom_entities._process_rule_based(text)
                        
                        assert result.success is True
                        # Check that we have at least the standard entity
                        assert len(result.data["entities"]) >= 1
                        # Check that custom entity extraction was called
                        mock_custom.assert_called_once()
                        assert result.data["sentiment_analysis"]["compound_score"] == 0.6
                        assert "Technology" in result.data["categories"]


class TestPerformanceOptimizations:
    """Test performance optimizations and sub-second processing."""
    
    @pytest.mark.asyncio
    async def test_processing_speed(self):
        """Test that processing completes in sub-second time."""
        pipeline = NLPPipeline()
        text = """
        Artificial intelligence is transforming the technology industry. Companies like Google and Microsoft
        are investing heavily in AI research and development. Machine learning algorithms are becoming more
        sophisticated every day, enabling new applications in healthcare, finance, and autonomous vehicles.
        The future of AI looks very promising with breakthrough innovations expected in the coming years.
        """
        
        import time
        start_time = time.time()
        
        result = await pipeline.process_text(text)
        
        processing_time = time.time() - start_time
        
        # Should complete in under 1 second
        assert processing_time < 1.0
        assert isinstance(result, NLPAnalysis)
        assert result.processing_time < 1.0
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self):
        """Test concurrent processing of multiple texts."""
        pipeline = NLPPipeline()
        texts = [
            "Technology news about artificial intelligence and machine learning.",
            "Financial markets are showing strong performance with cryptocurrency gains.",
            "Political developments in the upcoming election campaign.",
            "Healthcare innovations in medical research and drug development.",
            "Sports news about the championship tournament results."
        ]
        
        import time
        start_time = time.time()
        
        # Process all texts concurrently
        tasks = [pipeline.process_text(text) for text in texts]
        results = await asyncio.gather(*tasks)
        
        processing_time = time.time() - start_time
        
        # Concurrent processing should be faster than sequential
        assert processing_time < 2.0  # Should complete all 5 texts in under 2 seconds
        assert len(results) == 5
        assert all(isinstance(result, NLPAnalysis) for result in results)


if __name__ == "__main__":
    pytest.main([__file__])