"""
Thalamus Semantic Search Engine - Vector Embeddings and Full-Text Search

This module implements semantic search capabilities with:
1. Vector embeddings using sentence-transformers
2. Full-text search with TF-IDF
3. Hybrid semantic + lexical matching
4. Query optimization and result ranking
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from ..shared.schemas import ArticleResponse, SearchQuery, SearchResult, SearchResponse


@dataclass
class SearchDocument:
    """Document for semantic search indexing."""
    id: str
    title: str
    content: str
    summary: str
    url: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    tfidf_vector: Optional[np.ndarray] = None


@dataclass
class SearchIndex:
    """Search index containing both semantic and lexical components."""
    documents: Dict[str, SearchDocument]
    faiss_index: Optional[Any] = None
    tfidf_vectorizer: Optional[TfidfVectorizer] = None
    tfidf_matrix: Optional[np.ndarray] = None
    embedding_model: Optional[Any] = None


class SemanticSearchEngine:
    """
    Semantic search engine with vector embeddings and full-text search.
    
    Features:
    - Vector embeddings with sentence-transformers
    - FAISS for efficient similarity search
    - TF-IDF for lexical matching
    - Hybrid scoring combining semantic and lexical relevance
    - Query optimization and result ranking
    """
    
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        index_path: Optional[str] = None
    ):
        """Initialize the semantic search engine."""
        self.logger = logging.getLogger(__name__)
        self.embedding_model_name = embedding_model_name
        self.index_path = Path(index_path) if index_path else Path("data/search_index")
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self.embedding_model = None
        self.search_index = SearchIndex(documents={})
        
        # Configuration
        self.max_results = 100
        self.semantic_weight = 0.6
        self.lexical_weight = 0.4
        self.min_similarity_threshold = 0.1
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize embedding model and search components."""
        # Initialize sentence transformer
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                self.search_index.embedding_model = self.embedding_model
                self.logger.info(f"Loaded sentence transformer: {self.embedding_model_name}")
            except Exception as e:
                self.logger.error(f"Failed to load sentence transformer: {e}")
                self.embedding_model = None
        else:
            self.logger.warning("sentence-transformers not available")
        
        # Initialize TF-IDF vectorizer
        self.search_index.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True,
            min_df=2,
            max_df=0.8
        )
        
        # Load existing index if available
        self._load_index()
    
    async def add_document(self, document: SearchDocument) -> bool:
        """Add a document to the search index."""
        try:
            # Generate embeddings
            if self.embedding_model:
                text_for_embedding = f"{document.title} {document.content}"
                document.embedding = self.embedding_model.encode(text_for_embedding)
            
            # Add to document store
            self.search_index.documents[document.id] = document
            
            # Rebuild indices if we have enough documents
            if len(self.search_index.documents) % 100 == 0:
                await self._rebuild_indices()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding document {document.id}: {e}")
            return False
    
    async def add_documents(self, documents: List[SearchDocument]) -> int:
        """Add multiple documents to the search index."""
        added_count = 0
        
        for document in documents:
            if await self.add_document(document):
                added_count += 1
        
        # Rebuild indices after batch addition
        await self._rebuild_indices()
        
        return added_count
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None
    ) -> SearchResponse:
        """
        Perform semantic search with hybrid scoring.
        
        Args:
            query: Search query
            limit: Maximum results to return
            offset: Result offset for pagination
            filters: Optional filters to apply
            
        Returns:
            SearchResponse with ranked results
        """
        start_time = time.time()
        
        try:
            # Optimize query
            optimized_query = self._optimize_query(query)
            
            # Get semantic similarity scores
            semantic_scores = await self._get_semantic_scores(optimized_query)
            
            # Get lexical similarity scores
            lexical_scores = await self._get_lexical_scores(optimized_query)
            
            # Combine scores
            combined_scores = self._combine_scores(semantic_scores, lexical_scores)
            
            # Apply filters
            if filters:
                combined_scores = self._apply_filters(combined_scores, filters)
            
            # Sort by relevance and apply pagination
            sorted_results = sorted(
                combined_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Apply pagination
            paginated_results = sorted_results[offset:offset + limit]
            
            # Create search results
            search_results = []
            for doc_id, score in paginated_results:
                if score < self.min_similarity_threshold:
                    break
                    
                document = self.search_index.documents[doc_id]
                
                # Convert to ArticleResponse format
                article = ArticleResponse(
                    id=document.id,
                    url=document.url,
                    title=document.title,
                    content=document.content,
                    summary=document.summary,
                    scraped_at=document.metadata.get("scraped_at"),
                    source_domain=document.metadata.get("source_domain", ""),
                    nlp_data=document.metadata.get("nlp_data", {}),
                    page_metadata=document.metadata.get("page_metadata", {})
                )
                
                search_results.append(SearchResult(
                    score=float(score),
                    article=article
                ))
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return SearchResponse(
                query=query,
                total_hits=len([s for s in combined_scores.values() if s >= self.min_similarity_threshold]),
                results=search_results,
                took_ms=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Search error: {e}")
            return SearchResponse(
                query=query,
                total_hits=0,
                results=[],
                took_ms=int((time.time() - start_time) * 1000)
            )
    
    async def _get_semantic_scores(self, query: str) -> Dict[str, float]:
        """Get semantic similarity scores using vector embeddings."""
        scores = {}
        
        if not self.embedding_model or not self.search_index.documents:
            return scores
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query)
            
            # Use FAISS if available and index is built
            if FAISS_AVAILABLE and self.search_index.faiss_index:
                # Search using FAISS
                similarities, indices = self.search_index.faiss_index.search(
                    query_embedding.reshape(1, -1).astype('float32'),
                    min(self.max_results, len(self.search_index.documents))
                )
                
                doc_ids = list(self.search_index.documents.keys())
                for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                    if idx < len(doc_ids):
                        scores[doc_ids[idx]] = float(similarity)
            else:
                # Fallback to manual cosine similarity
                for doc_id, document in self.search_index.documents.items():
                    if document.embedding is not None:
                        similarity = cosine_similarity(
                            query_embedding.reshape(1, -1),
                            document.embedding.reshape(1, -1)
                        )[0][0]
                        scores[doc_id] = float(similarity)
        
        except Exception as e:
            self.logger.error(f"Error calculating semantic scores: {e}")
        
        return scores
    
    async def _get_lexical_scores(self, query: str) -> Dict[str, float]:
        """Get lexical similarity scores using TF-IDF."""
        scores = {}
        
        if not self.search_index.tfidf_vectorizer or self.search_index.tfidf_matrix is None:
            return scores
        
        try:
            # Transform query
            query_vector = self.search_index.tfidf_vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.search_index.tfidf_matrix)[0]
            
            doc_ids = list(self.search_index.documents.keys())
            for i, similarity in enumerate(similarities):
                if i < len(doc_ids):
                    scores[doc_ids[i]] = float(similarity)
        
        except Exception as e:
            self.logger.error(f"Error calculating lexical scores: {e}")
        
        return scores
    
    def _combine_scores(
        self,
        semantic_scores: Dict[str, float],
        lexical_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """Combine semantic and lexical scores with weighted average."""
        combined_scores = {}
        
        all_doc_ids = set(semantic_scores.keys()) | set(lexical_scores.keys())
        
        for doc_id in all_doc_ids:
            semantic_score = semantic_scores.get(doc_id, 0.0)
            lexical_score = lexical_scores.get(doc_id, 0.0)
            
            # Weighted combination
            combined_score = (
                self.semantic_weight * semantic_score +
                self.lexical_weight * lexical_score
            )
            
            combined_scores[doc_id] = combined_score
        
        return combined_scores
    
    def _optimize_query(self, query: str) -> str:
        """Optimize query for better search results."""
        # Basic query optimization
        optimized = query.strip().lower()
        
        # Remove common stop words that don't add semantic value
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = optimized.split()
        optimized_words = [word for word in words if word not in stop_words or len(words) <= 3]
        
        return ' '.join(optimized_words)
    
    def _apply_filters(
        self,
        scores: Dict[str, float],
        filters: Dict[str, Any]
    ) -> Dict[str, float]:
        """Apply filters to search results."""
        filtered_scores = {}
        
        for doc_id, score in scores.items():
            document = self.search_index.documents[doc_id]
            
            # Apply filters based on metadata
            include_document = True
            
            for filter_key, filter_value in filters.items():
                if filter_key in document.metadata:
                    if document.metadata[filter_key] != filter_value:
                        include_document = False
                        break
            
            if include_document:
                filtered_scores[doc_id] = score
        
        return filtered_scores
    
    async def _rebuild_indices(self):
        """Rebuild search indices for optimal performance."""
        try:
            documents = list(self.search_index.documents.values())
            
            if not documents:
                return
            
            # Rebuild TF-IDF index
            texts = [f"{doc.title} {doc.content}" for doc in documents]
            self.search_index.tfidf_matrix = self.search_index.tfidf_vectorizer.fit_transform(texts)
            
            # Rebuild FAISS index if available
            if FAISS_AVAILABLE and self.embedding_model:
                embeddings = []
                for doc in documents:
                    if doc.embedding is not None:
                        embeddings.append(doc.embedding)
                
                if embeddings:
                    embeddings_array = np.array(embeddings).astype('float32')
                    
                    # Create FAISS index
                    dimension = embeddings_array.shape[1]
                    self.search_index.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
                    
                    # Normalize embeddings for cosine similarity
                    faiss.normalize_L2(embeddings_array)
                    self.search_index.faiss_index.add(embeddings_array)
            
            self.logger.info(f"Rebuilt search indices for {len(documents)} documents")
            
        except Exception as e:
            self.logger.error(f"Error rebuilding indices: {e}")
    
    def _save_index(self):
        """Save search index to disk."""
        try:
            # Save documents
            documents_path = self.index_path / "documents.pkl"
            with open(documents_path, 'wb') as f:
                pickle.dump(self.search_index.documents, f)
            
            # Save TF-IDF vectorizer
            if self.search_index.tfidf_vectorizer:
                tfidf_path = self.index_path / "tfidf_vectorizer.pkl"
                with open(tfidf_path, 'wb') as f:
                    pickle.dump(self.search_index.tfidf_vectorizer, f)
            
            # Save FAISS index
            if FAISS_AVAILABLE and self.search_index.faiss_index:
                faiss_path = self.index_path / "faiss_index.bin"
                faiss.write_index(self.search_index.faiss_index, str(faiss_path))
            
            self.logger.info("Search index saved to disk")
            
        except Exception as e:
            self.logger.error(f"Error saving index: {e}")
    
    def _load_index(self):
        """Load search index from disk."""
        try:
            # Load documents
            documents_path = self.index_path / "documents.pkl"
            if documents_path.exists():
                with open(documents_path, 'rb') as f:
                    self.search_index.documents = pickle.load(f)
            
            # Load TF-IDF vectorizer
            tfidf_path = self.index_path / "tfidf_vectorizer.pkl"
            if tfidf_path.exists():
                with open(tfidf_path, 'rb') as f:
                    self.search_index.tfidf_vectorizer = pickle.load(f)
            
            # Load FAISS index
            if FAISS_AVAILABLE:
                faiss_path = self.index_path / "faiss_index.bin"
                if faiss_path.exists():
                    self.search_index.faiss_index = faiss.read_index(str(faiss_path))
            
            if self.search_index.documents:
                self.logger.info(f"Loaded search index with {len(self.search_index.documents)} documents")
            
        except Exception as e:
            self.logger.error(f"Error loading index: {e}")
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """Get search index statistics."""
        return {
            "total_documents": len(self.search_index.documents),
            "has_faiss_index": self.search_index.faiss_index is not None,
            "has_tfidf_index": self.search_index.tfidf_matrix is not None,
            "embedding_model": self.embedding_model_name,
            "semantic_weight": self.semantic_weight,
            "lexical_weight": self.lexical_weight
        }
    
    def __del__(self):
        """Save index when object is destroyed."""
        try:
            self._save_index()
        except:
            pass


# Global search engine instance for dependency injection
_search_engine: Optional[SemanticSearchEngine] = None


def get_search_engine() -> SemanticSearchEngine:
    """Get global search engine instance for dependency injection."""
    global _search_engine
    if _search_engine is None:
        _search_engine = SemanticSearchEngine()
    return _search_engine


async def search_articles(query: str, limit: int = 10, offset: int = 0) -> SearchResponse:
    """
    Convenience function to search articles using semantic search.
    
    Args:
        query: Search query
        limit: Maximum results to return
        offset: Result offset for pagination
        
    Returns:
        SearchResponse with ranked results
    """
    search_engine = get_search_engine()
    return await search_engine.search(query, limit, offset)