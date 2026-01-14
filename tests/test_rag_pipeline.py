"""
Unit tests for RAG pipeline
"""
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from modules.pdf_extractor import PDFExtractor
from modules.text_chunker import TextChunker
from modules.embeddings import EmbeddingGenerator
from modules.vector_store import VectorStore
from rag_pipeline import RAGPipeline


def _has_embedding_key() -> bool:
    provider = (settings.EMBEDDING_PROVIDER or "openai").lower()
    if provider == "openai":
        return bool(settings.OPENAI_API_KEY)
    if provider == "gemini":
        return bool(settings.GEMINI_API_KEY)
    return False


requires_embeddings = pytest.mark.skipif(
    not _has_embedding_key(),
    reason="Embedding API key not configured (set OPENAI_API_KEY or GEMINI_API_KEY)."
)


class TestPDFExtractor:
    """Test PDF extraction"""
    
    def test_initialization(self):
        extractor = PDFExtractor()
        assert extractor is not None
        assert extractor.max_file_size > 0
    
    def test_extract_nonexistent_file(self):
        extractor = PDFExtractor()
        with pytest.raises(FileNotFoundError):
            extractor.extract_text_from_pdf("nonexistent.pdf")


class TestTextChunker:
    """Test text chunking"""
    
    def test_initialization(self):
        chunker = TextChunker()
        assert chunker.chunk_size > 0
        assert chunker.chunk_overlap >= 0
    
    def test_chunk_text(self):
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        text = "This is a test. " * 50  # 800 characters
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) > 0
        assert all('text' in chunk for chunk in chunks)
        assert all('chunk_index' in chunk for chunk in chunks)
    
    def test_different_strategies(self):
        text = "Test paragraph.\n\nAnother paragraph.\n\nThird paragraph."
        
        chunker_recursive = TextChunker(strategy="recursive")
        chunks_recursive = chunker_recursive.chunk_text(text)
        
        chunker_char = TextChunker(strategy="character")
        chunks_char = chunker_char.chunk_text(text)
        
        assert len(chunks_recursive) > 0
        assert len(chunks_char) > 0


class TestEmbeddingGenerator:
    """Test embedding generation"""
    
    @pytest.mark.integration
    @requires_embeddings
    def test_initialization(self):
        generator = EmbeddingGenerator()
        assert generator is not None
        assert generator.model is not None
    
    @pytest.mark.integration
    @requires_embeddings
    def test_generate_single_embedding(self):
        generator = EmbeddingGenerator()
        text = "This is a test sentence."
        embedding = generator.generate_embedding(text)
        
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)
    
    @pytest.mark.integration
    @requires_embeddings
    def test_generate_batch_embeddings(self):
        generator = EmbeddingGenerator()
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        embeddings = generator.generate_embeddings_batch(texts)
        
        assert len(embeddings) == len(texts)
        assert all(isinstance(emb, list) for emb in embeddings)
    
    @pytest.mark.integration
    @requires_embeddings
    def test_embedding_dimension(self):
        generator = EmbeddingGenerator()
        dim = generator.get_embedding_dimension()
        assert dim > 0
    
    @pytest.mark.integration
    @requires_embeddings
    def test_cosine_similarity(self):
        generator = EmbeddingGenerator()
        emb1 = generator.generate_embedding("machine learning")
        emb2 = generator.generate_embedding("machine learning")
        emb3 = generator.generate_embedding("cooking recipes")
        
        # Similar texts should have high similarity
        sim_same = generator.cosine_similarity(emb1, emb2)
        assert sim_same > 0.9
        
        # Different texts should have lower similarity
        sim_diff = generator.cosine_similarity(emb1, emb3)
        assert sim_diff < sim_same


class TestVectorStore:
    """Test vector store operations"""
    
    @pytest.fixture
    def vector_store(self):
        # Use a test collection
        return VectorStore(
            db_type="chroma",
            collection_name="test_collection"
        )
    
    def test_initialization(self, vector_store):
        assert vector_store is not None
        assert vector_store.collection is not None
    
    @pytest.mark.integration
    @requires_embeddings
    def test_add_and_search(self, vector_store):
        # Create test chunks with embeddings
        generator = EmbeddingGenerator()
        
        chunks = [
            {"id": "test_1", "text": "Machine learning is fascinating.", "metadata": {}},
            {"id": "test_2", "text": "Deep learning uses neural networks.", "metadata": {}},
            {"id": "test_3", "text": "Cooking is an art form.", "metadata": {}}
        ]
        
        # Generate embeddings
        for chunk in chunks:
            chunk["embedding"] = generator.generate_embedding(chunk["text"])
        
        # Add to vector store
        vector_store.add_embeddings(chunks)
        
        # Search
        query = "What is machine learning?"
        query_embedding = generator.generate_embedding(query)
        results = vector_store.search(query_embedding, top_k=2)
        
        assert len(results) <= 2
        assert all('score' in r for r in results)
        # The most relevant result should be about machine learning
        assert "machine learning" in results[0]['text'].lower() or "deep learning" in results[0]['text'].lower()
    
    def test_collection_stats(self, vector_store):
        stats = vector_store.get_collection_stats()
        assert 'name' in stats
        assert 'count' in stats
        assert stats['count'] >= 0


class TestRAGPipeline:
    """Test complete RAG pipeline"""
    
    @pytest.fixture
    def pipeline(self):
        return RAGPipeline(collection_name="test_rag_collection")
    
    @pytest.mark.integration
    @requires_embeddings
    def test_initialization(self, pipeline):
        assert pipeline is not None
        assert pipeline.pdf_extractor is not None
        assert pipeline.text_chunker is not None
        assert pipeline.embedding_generator is not None
        assert pipeline.vector_store is not None
        assert pipeline.retriever is not None
        assert pipeline.llm_generator is None
    
    @pytest.mark.integration
    @requires_embeddings
    def test_get_stats(self, pipeline):
        stats = pipeline.get_stats()
        
        assert 'vector_store' in stats
        assert 'embedding_model' in stats
        assert 'llm_model' in stats
        assert 'chunk_size' in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
