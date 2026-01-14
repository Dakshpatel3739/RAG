"""
Retrieval module for finding relevant context
"""
from typing import List, Dict, Optional
from modules.embeddings import EmbeddingGenerator
from modules.vector_store import VectorStore
from utils.logger import log
from config.settings import settings


class Retriever:
    """Retrieve relevant chunks for a given query"""
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_generator: EmbeddingGenerator
    ):
        """
        Initialize retriever
        
        Args:
            vector_store: Vector store instance
            embedding_generator: Embedding generator instance
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.similarity_threshold = settings.SIMILARITY_THRESHOLD
    
    def retrieve(
        self,
        query: str,
        top_k: int = None,
        filter_metadata: Optional[Dict] = None,
        rerank: bool = True
    ) -> List[Dict]:
        """
        Retrieve relevant chunks for a query
        
        Args:
            query: User query
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
            rerank: Whether to apply reranking
            
        Returns:
            List of relevant chunks with scores
        """
        try:
            log.info(f"Retrieving context for query: {query[:100]}...")
            
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_embedding(query)
            
            # Search vector store
            top_k = top_k or settings.TOP_K_RESULTS
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k * 2 if rerank else top_k,  # Get more for reranking
                filter_dict=filter_metadata
            )
            
            # Filter by similarity threshold
            filtered_results = [
                r for r in results 
                if r["score"] >= self.similarity_threshold
            ]
            
            if not filtered_results:
                log.warning(f"No results above similarity threshold {self.similarity_threshold}")
                # Return top results anyway
                filtered_results = results[:top_k]
            
            # Rerank if enabled
            if rerank and len(filtered_results) > top_k:
                filtered_results = self._rerank(query, filtered_results)
            
            # Limit to top_k
            final_results = filtered_results[:top_k]
            
            log.info(f"Retrieved {len(final_results)} relevant chunks")
            return final_results
            
        except Exception as e:
            log.error(f"Error during retrieval: {str(e)}")
            raise
    
    def _rerank(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        Rerank results using cross-encoder or other methods
        
        Args:
            query: Original query
            results: Initial search results
            
        Returns:
            Reranked results
        """
        # Simple reranking based on text length and diversity
        # In production, use a cross-encoder model
        
        try:
            log.info("Reranking results")
            
            # Penalize very short chunks
            for result in results:
                text_length = len(result["text"])
                if text_length < 100:
                    result["score"] *= 0.8
            
            # Sort by score
            reranked = sorted(results, key=lambda x: x["score"], reverse=True)
            
            return reranked
            
        except Exception as e:
            log.warning(f"Reranking failed: {str(e)}, returning original order")
            return results
    
    def retrieve_with_context_window(
        self,
        query: str,
        top_k: int = None,
        context_chunks: int = 1
    ) -> List[Dict]:
        """
        Retrieve chunks with surrounding context
        
        Args:
            query: User query
            top_k: Number of primary results
            context_chunks: Number of surrounding chunks to include
            
        Returns:
            List of chunks with context
        """
        try:
            # Get initial results
            results = self.retrieve(query, top_k)
            
            # TODO: Implement logic to fetch surrounding chunks
            # This would require storing chunk sequence information
            
            return results
            
        except Exception as e:
            log.error(f"Error retrieving with context window: {str(e)}")
            raise
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = None,
        keyword_weight: float = 0.3,
        semantic_weight: float = 0.7
    ) -> List[Dict]:
        """
        Combine keyword and semantic search
        
        Args:
            query: User query
            top_k: Number of results
            keyword_weight: Weight for keyword matching
            semantic_weight: Weight for semantic similarity
            
        Returns:
            Combined search results
        """
        try:
            log.info("Performing hybrid search")
            
            # Semantic search
            semantic_results = self.retrieve(query, top_k, rerank=False)
            
            # Simple keyword matching (in production, use BM25 or similar)
            query_terms = set(query.lower().split())
            
            for result in semantic_results:
                text_terms = set(result["text"].lower().split())
                keyword_overlap = len(query_terms & text_terms) / len(query_terms)
                
                # Combine scores
                original_score = result["score"]
                result["score"] = (
                    semantic_weight * original_score +
                    keyword_weight * keyword_overlap
                )
            
            # Sort by combined score
            hybrid_results = sorted(
                semantic_results,
                key=lambda x: x["score"],
                reverse=True
            )[:top_k or settings.TOP_K_RESULTS]
            
            log.info(f"Hybrid search returned {len(hybrid_results)} results")
            return hybrid_results
            
        except Exception as e:
            log.error(f"Error in hybrid search: {str(e)}")
            raise
