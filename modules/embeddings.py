"""
Embedding generation module (OpenAI or Gemini)
"""
from typing import List, Optional
import numpy as np
from openai import OpenAI
from utils.logger import log
from config.settings import settings


class EmbeddingGenerator:
    """Generate embeddings using OpenAI or Gemini"""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        use_openai: Optional[bool] = None,
        provider: Optional[str] = None
    ):
        """Initialize embedding generator with OpenAI or Gemini"""
        self.provider = (provider or settings.EMBEDDING_PROVIDER or "openai").lower()
        if use_openai is True:
            self.provider = "openai"
        
        if self.provider == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings")
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
            self.model_name = model_name or settings.EMBEDDING_MODEL
            self.model = self.model_name
            self.embedding_dimension = 1536
            log.info("Using OpenAI embeddings")
        elif self.provider == "gemini":
            if not settings.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY is required for Gemini embeddings")
            import google.generativeai as genai
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.genai = genai
            self.model_name = model_name or settings.GEMINI_EMBEDDING_MODEL
            self.model = self.model_name
            self.embedding_dimension = 768
            self.query_task_type = "retrieval_query"
            self.doc_task_type = "retrieval_document"
            log.info("Using Gemini embeddings")
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")
    
    def _gemini_model_name(self) -> str:
        if self.model_name.startswith("models/"):
            return self.model_name
        return f"models/{self.model_name}"
    
    def _extract_gemini_embedding(self, response) -> List[float]:
        if isinstance(response, dict):
            embedding = response.get("embedding")
        else:
            embedding = getattr(response, "embedding", None)
        if not embedding:
            raise ValueError("Gemini embedding response missing embedding data")
        return embedding
    
    def _gemini_embed(self, text: str, task_type: str) -> List[float]:
        response = self.genai.embed_content(
            model=self._gemini_model_name(),
            content=text,
            task_type=task_type
        )
        return self._extract_gemini_embedding(response)
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        try:
            if self.provider == "openai":
                response = self.client.embeddings.create(
                    input=text,
                    model=self.model_name
                )
                return response.data[0].embedding
            if self.provider == "gemini":
                return self._gemini_embed(text, self.query_task_type)
            raise ValueError(f"Unsupported embedding provider: {self.provider}")
        except Exception as e:
            log.error(f"Error generating embedding: {str(e)}")
            raise
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = None) -> List[List[float]]:
        """Generate embeddings for multiple texts in batches"""
        try:
            batch_size = batch_size or settings.BATCH_SIZE
            log.info(f"Generating embeddings for {len(texts)} texts")
            
            embeddings = []
            if self.provider == "openai":
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    response = self.client.embeddings.create(
                        input=batch,
                        model=self.model_name
                    )
                    batch_embeddings = [item.embedding for item in response.data]
                    embeddings.extend(batch_embeddings)
                    log.debug(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            elif self.provider == "gemini":
                for i, text in enumerate(texts, 1):
                    embeddings.append(self._gemini_embed(text, self.doc_task_type))
                    if i % batch_size == 0 or i == len(texts):
                        log.debug(f"Processed {i}/{len(texts)} embeddings")
            else:
                raise ValueError(f"Unsupported embedding provider: {self.provider}")
            
            log.info(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            log.error(f"Error generating batch embeddings: {str(e)}")
            raise
    
    def embed_chunks(self, chunks: List[dict]) -> List[dict]:
        """Add embeddings to chunk objects"""
        try:
            texts = [chunk["text"] for chunk in chunks]
            embeddings = self.generate_embeddings_batch(texts)
            
            for chunk, embedding in zip(chunks, embeddings):
                chunk["embedding"] = embedding
            
            log.info(f"Added embeddings to {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            log.error(f"Error embedding chunks: {str(e)}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        return self.embedding_dimension
    
    def cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        return float(dot_product / (norm1 * norm2))
