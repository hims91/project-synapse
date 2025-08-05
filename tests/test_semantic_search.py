"""
Unit tests for Thalamus Semantic Search Engine

Tests semantic search capabilities including:
1. Vector embeddings with sentence-transformers
2. Full-text search with TF-IDF
3. Hybrid semantic + lexical matching
4. Query optimization and result ranking
"""

import pytest
import asyncio
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from datetime import datetime
import uuid

from src.thalamus.semantic_search import (
    SemanticSearchEngine,
    SearchDocument,
    SearchIndex,
    get_search_engine,
    search_articles
)
from src.shared.schemas import SearchResponse, SearchResult, ArticleResponse


class TestSemanticSearchEngine:
    """Test cases for Semantic Search Engine functionality."""
    
    @pytest.fixture
    def temp_index_path(self):
        """Create temporary directory for search index."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def search_engine(self, temp_index_path):
        """Create search engine instance for testing."""
        return SemanticSearchEngine(index_path=temp_index_path)
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            SearchDocument(
                id="1",
                title="Artificial Intelligence Revolution",
                content="AI and machine learning are transforming technology industry",
                summary="AI transforms tech",
                url="https://example.com/ai-revolution",
                metadata={
                    "scraped_at": datetime.now(),
                    "source_domain": "example.com",
                    "nlp_data": {},
                    "page_metadata": {}
                }
            ),
            SearchDocument(
                id="2",
                title="Cryptocurrency Market Analysis",
                content="Bitcoin and Ethereum prices surge in crypto markets",
                summary="Crypto prices surge",
                url="https://example.com/crypto-analysis",
                metadata={
                    "scraped_at": datetime.now(),
                    "source_domain": "example.com",
                    "nlp_data": {},
                    "page_metadata": {}
                }
            ),
            SearchDocument(
                id="3",
                title="Climate Change Impact",
                content="Global warming affects weather patterns worldwide",
                summary="Climate affects weather",
                url="https://example.com/climate-change",
                metadata={
                    "scraped_at": datetime.now(),
                    "source_domain": "example.com",
                    "nlp_data": {},
                    "page_metadata": {}
                }
            )
        ]
    
    def test_search_engine_initialization(self, search_engine):
        """Test search engine initialization."""
        assert search_engine is not None
        assert search_engine.embedding_model_name == "all-MiniLM-L6-v2"
        assert search_engine.semantic_weight == 0.6
        assert search_engine.lexical_weight == 0.4
        assert search_engine.min_similarity_threshold == 0.1
        assert isinstance(search_engine.search_index, SearchIndex)
    
    @pytest.mark.asyncio
    async def test_add_single_document(self, search_engine, sample_documents):
        """Test adding a single document to the search index."""
        document = sample_documents[0]
        
        # Mock embedding model
        with patch.object(search_engine, 'embedding_model') as mock_model:
            mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
            
            result = await search_engine.add_document(document)
            
            assert result is True
            assert document.id in search_engine.search_index.documents
            assert document.embedding is not None
            mock_model.encode.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_add_multiple_documents(self, search_engine, sample_documents):
        """Test adding multiple documents to the search index."""
        with patch.object(search_engine, 'embedding_model') as mock_model:
            mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
            
            with patch.object(search_engine, '_rebuild_indices') as mock_rebuild:
                added_count = await search_engine.add_documents(sample_documents)
                
                assert added_count == 3
                assert len(search_engine.search_index.documents) == 3
                mock_rebuild.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_semantic_search_basic(self, search_engine, sample_documents):
        """Test basic semantic search functionality."""
        # Add documents
        with patch.object(search_engine, 'embedding_model') as mock_model:
            mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
            await search_engine.add_documents(sample_documents)
        
        # Mock search methods
        with patch.object(search_engine, '_get_semantic_scores') as mock_semantic:
            with patch.object(search_engine, '_get_lexical_scores') as mock_lexical:
                mock_semantic.return_value = {"1": 0.8, "2": 0.3, "3": 0.1}
                mock_lexical.return_value = {"1": 0.6, "2": 0.7, "3": 0.2}
                
                result = await search_engine.search("artificial intelligence", limit=2)
                
                assert isinstance(result, SearchResponse)
                assert result.query == "artificial intelligence"
                assert len(result.results) <= 2
                assert result.total_hits > 0
                assert result.took_ms > 0
    
    @pytest.mark.asyncio
    async def test_semantic_scores_calculation(self, search_engine, sample_documents):
        """Test semantic similarity score calculation."""
        # Add documents with mock embeddings
        for doc in sample_documents:
            doc.embedding = np.random.rand(384)  # Typical sentence-transformer dimension
            search_engine.search_index.documents[doc.id] = doc
        
        with patch.object(search_engine, 'embedding_model') as mock_model:
            mock_model.encode.return_value = np.random.rand(384)
            
            scores = await search_engine._get_semantic_scores("test query")
            
            assert isinstance(scores, dict)
            assert len(scores) == 3
            assert all(isinstance(score, float) for score in scores.values())
            assert all(-1 <= score <= 1 for score in scores.values())
    
    @pytest.mark.asyncio
    async def test_lexical_scores_calculation(self, search_engine, sample_documents):
        """Test lexical similarity score calculation with TF-IDF."""
        # Add documents
        await search_engine.add_documents(sample_documents)
        
        # Mock TF-IDF components
        with patch.object(search_engine.search_index, 'tfidf_vectorizer') as mock_vectorizer:
            with patch.object(search_engine.search_index, 'tfidf_matrix') as mock_matrix:
                # Mock vectorizer transform
                mock_query_vector = np.array([[0.1, 0.2, 0.3]])
                mock_vectorizer.transform.return_value = mock_query_vector
                
                # Mock similarity calculation
                with patch('src.thalamus.semantic_search.cosine_similarity') as mock_cosine:
                    mock_cosine.return_value = np.array([[0.8, 0.5, 0.2]])
                    
                    scores = await search_engine._get_lexical_scores("artificial intelligence")
                    
                    assert isinstance(scores, dict)
                    assert len(scores) == 3
                    mock_vectorizer.transform.assert_called_once_with(["artificial intelligence"])
    
    def test_score_combination(self, search_engine):
        """Test combining semantic and lexical scores."""
        semantic_scores = {"1": 0.8, "2": 0.3, "3": 0.1}
        lexical_scores = {"1": 0.6, "2": 0.7, "3": 0.2}
        
        combined = search_engine._combine_scores(semantic_scores, lexical_scores)
        
        assert isinstance(combined, dict)
        assert len(combined) == 3
        
        # Check weighted combination (0.6 * semantic + 0.4 * lexical)
        expected_score_1 = 0.6 * 0.8 + 0.4 * 0.6  # 0.48 + 0.24 = 0.72
        assert abs(combined["1"] - expected_score_1) < 0.001
    
    def test_query_optimization(self, search_engine):
        """Test query optimization functionality."""
        # Test basic optimization
        optimized = search_engine._optimize_query("  The AI Revolution  ")
        assert optimized == "ai revolution"
        
        # Test stop word removal
        optimized = search_engine._optimize_query("artificial intelligence and machine learning")
        assert "and" not in optimized.split()
        
        # Test preserving important words in short queries
        optimized = search_engine._optimize_query("the AI")
        assert "the" in optimized  # Should preserve in short queries
    
    def test_filter_application(self, search_engine, sample_documents):
        """Test applying filters to search results."""
        # Add documents to index
        for doc in sample_documents:
            search_engine.search_index.documents[doc.id] = doc
        
        scores = {"1": 0.8, "2": 0.6, "3": 0.4}
        filters = {"source_domain": "example.com"}
        
        filtered_scores = search_engine._apply_filters(scores, filters)
        
        assert isinstance(filtered_scores, dict)
        assert len(filtered_scores) == 3  # All documents match filter
        
        # Test with non-matching filter
        filters = {"source_domain": "other.com"}
        filtered_scores = search_engine._apply_filters(scores, filters)
        assert len(filtered_scores) == 0  # No documents match
    
    @pytest.mark.asyncio
    async def test_index_rebuild(self, search_engine, sample_documents):
        """Test search index rebuilding."""
        # Add documents
        for doc in sample_documents:
            doc.embedding = np.random.rand(384)
            search_engine.search_index.documents[doc.id] = doc
        
        # Mock TF-IDF vectorizer
        with patch.object(search_engine.search_index.tfidf_vectorizer, 'fit_transform') as mock_fit:
            mock_fit.return_value = np.random.rand(3, 1000)  # Mock TF-IDF matrix
            
            await search_engine._rebuild_indices()
            
            mock_fit.assert_called_once()
            assert search_engine.search_index.tfidf_matrix is not None
    
    @pytest.mark.asyncio
    async def test_search_with_pagination(self, search_engine, sample_documents):
        """Test search with pagination parameters."""
        # Add documents
        with patch.object(search_engine, 'embedding_model') as mock_model:
            mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
            await search_engine.add_documents(sample_documents)
        
        # Mock search methods
        with patch.object(search_engine, '_get_semantic_scores') as mock_semantic:
            with patch.object(search_engine, '_get_lexical_scores') as mock_lexical:
                mock_semantic.return_value = {"1": 0.8, "2": 0.6, "3": 0.4}
                mock_lexical.return_value = {"1": 0.7, "2": 0.5, "3": 0.3}
                
                # Test first page
                result = await search_engine.search("test", limit=2, offset=0)
                assert len(result.results) == 2
                
                # Test second page
                result = await search_engine.search("test", limit=2, offset=2)
                assert len(result.results) == 1  # Only one document left
    
    @pytest.mark.asyncio
    async def test_search_performance(self, search_engine, sample_documents):
        """Test search performance is under 200ms."""
        # Add documents
        with patch.object(search_engine, 'embedding_model') as mock_model:
            mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
            await search_engine.add_documents(sample_documents)
        
        # Mock search methods for fast execution
        with patch.object(search_engine, '_get_semantic_scores') as mock_semantic:
            with patch.object(search_engine, '_get_lexical_scores') as mock_lexical:
                mock_semantic.return_value = {"1": 0.8, "2": 0.6, "3": 0.4}
                mock_lexical.return_value = {"1": 0.7, "2": 0.5, "3": 0.3}
                
                result = await search_engine.search("test query")
                
                # Should complete in under 200ms
                assert result.took_ms < 200
    
    @pytest.mark.asyncio
    async def test_get_index_stats(self, search_engine, sample_documents):
        """Test getting search index statistics."""
        await search_engine.add_documents(sample_documents)
        
        stats = await search_engine.get_index_stats()
        
        assert isinstance(stats, dict)
        assert "total_documents" in stats
        assert "has_faiss_index" in stats
        assert "has_tfidf_index" in stats
        assert "embedding_model" in stats
        assert stats["total_documents"] == 3
        assert stats["embedding_model"] == "all-MiniLM-L6-v2"


class TestSemanticSearchIntegration:
    """Integration tests for semantic search."""
    
    @pytest.mark.asyncio
    async def test_search_articles_function(self):
        """Test the convenience function for searching articles."""
        with patch('src.thalamus.semantic_search.get_search_engine') as mock_get_engine:
            mock_engine = Mock()
            mock_response = SearchResponse(
                query="test query",
                total_hits=1,
                results=[],
                took_ms=50
            )
            mock_engine.search = Mock(return_value=mock_response)
            mock_get_engine.return_value = mock_engine
            
            result = await search_articles("test query", limit=10, offset=0)
            
            assert isinstance(result, SearchResponse)
            mock_engine.search.assert_called_once_with("test query", 10, 0)
    
    def test_get_search_engine_singleton(self):
        """Test that get_search_engine returns singleton instance."""
        # Reset global instance
        import src.thalamus.semantic_search as search_module
        search_module._search_engine = None
        
        engine1 = get_search_engine()
        engine2 = get_search_engine()
        
        assert engine1 is engine2
        assert isinstance(engine1, SemanticSearchEngine)


class TestSemanticSearchErrorHandling:
    """Test error handling in semantic search."""
    
    @pytest.mark.asyncio
    async def test_search_with_no_documents(self):
        """Test search behavior with empty index."""
        engine = SemanticSearchEngine()
        
        result = await engine.search("test query")
        
        assert isinstance(result, SearchResponse)
        assert result.total_hits == 0
        assert len(result.results) == 0
        assert result.took_ms >= 0
    
    @pytest.mark.asyncio
    async def test_add_document_error_handling(self):
        """Test error handling when adding documents fails."""
        engine = SemanticSearchEngine()
        
        # Create document with invalid data
        document = SearchDocument(
            id="test",
            title="Test",
            content="Test content",
            summary="Test summary",
            url="invalid-url",
            metadata={}
        )
        
        # Mock embedding model to raise exception
        with patch.object(engine, 'embedding_model') as mock_model:
            mock_model.encode.side_effect = Exception("Encoding failed")
            
            result = await engine.add_document(document)
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_search_error_handling(self):
        """Test search error handling."""
        engine = SemanticSearchEngine()
        
        # Mock methods to raise exceptions
        with patch.object(engine, '_get_semantic_scores', side_effect=Exception("Semantic error")):
            result = await engine.search("test query")
            
            assert isinstance(result, SearchResponse)
            assert result.total_hits == 0
            assert len(result.results) == 0


class TestSemanticSearchPerformance:
    """Test performance characteristics of semantic search."""
    
    @pytest.mark.asyncio
    async def test_large_document_set_performance(self):
        """Test performance with larger document set."""
        engine = SemanticSearchEngine()
        
        # Create many documents
        documents = []
        for i in range(50):
            documents.append(SearchDocument(
                id=str(i),
                title=f"Document {i}",
                content=f"Content for document {i} with various keywords",
                summary=f"Summary {i}",
                url=f"https://example.com/doc{i}",
                metadata={"source_domain": "example.com"}
            ))
        
        # Mock embedding model for fast execution
        with patch.object(engine, 'embedding_model') as mock_model:
            mock_model.encode.return_value = np.random.rand(384)
            
            # Add documents
            import time
            start_time = time.time()
            added_count = await engine.add_documents(documents)
            add_time = time.time() - start_time
            
            assert added_count == 50
            assert add_time < 5.0  # Should add 50 documents in under 5 seconds
            
            # Test search performance
            with patch.object(engine, '_get_semantic_scores') as mock_semantic:
                with patch.object(engine, '_get_lexical_scores') as mock_lexical:
                    # Mock realistic scores
                    semantic_scores = {str(i): np.random.rand() for i in range(50)}
                    lexical_scores = {str(i): np.random.rand() for i in range(50)}
                    mock_semantic.return_value = semantic_scores
                    mock_lexical.return_value = lexical_scores
                    
                    start_time = time.time()
                    result = await engine.search("test query", limit=10)
                    search_time = time.time() - start_time
                    
                    assert search_time < 1.0  # Should search in under 1 second
                    assert len(result.results) <= 10
    
    @pytest.mark.asyncio
    async def test_concurrent_searches(self):
        """Test concurrent search performance."""
        engine = SemanticSearchEngine()
        
        # Add some documents
        documents = [
            SearchDocument(
                id=str(i),
                title=f"Document {i}",
                content=f"Content {i}",
                summary=f"Summary {i}",
                url=f"https://example.com/{i}",
                metadata={}
            ) for i in range(10)
        ]
        
        with patch.object(engine, 'embedding_model') as mock_model:
            mock_model.encode.return_value = np.random.rand(384)
            await engine.add_documents(documents)
        
        # Mock search methods
        with patch.object(engine, '_get_semantic_scores') as mock_semantic:
            with patch.object(engine, '_get_lexical_scores') as mock_lexical:
                mock_semantic.return_value = {str(i): np.random.rand() for i in range(10)}
                mock_lexical.return_value = {str(i): np.random.rand() for i in range(10)}
                
                # Run concurrent searches
                queries = ["query 1", "query 2", "query 3", "query 4", "query 5"]
                
                import time
                start_time = time.time()
                
                tasks = [engine.search(query) for query in queries]
                results = await asyncio.gather(*tasks)
                
                concurrent_time = time.time() - start_time
                
                assert len(results) == 5
                assert all(isinstance(result, SearchResponse) for result in results)
                assert concurrent_time < 2.0  # Should complete 5 concurrent searches in under 2 seconds


if __name__ == "__main__":
    pytest.main([__file__])