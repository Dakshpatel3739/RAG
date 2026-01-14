"""
Configuration settings for RAG system
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    PINECONE_API_KEY: Optional[str] = None
    
    # Model configurations
    EMBEDDING_MODEL: str = "text-embedding-ada-002"
    LLM_MODEL: str = "gpt-3.5-turbo"
    EMBEDDING_PROVIDER: str = "openai"  # Options: openai, gemini
    LLM_PROVIDER: str = "openai"  # Options: openai, gemini
    GEMINI_EMBEDDING_MODEL: str = "text-embedding-004"
    GEMINI_LLM_MODEL: str = "gemini-flash-latest"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 500
    
    # Chunking parameters
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Vector DB settings
    VECTOR_DB_TYPE: str = "chroma"  # Options: chroma, faiss, pinecone
    VECTOR_DB_PATH: str = "./data/vector_db"
    COLLECTION_NAME: str = "documents"
    
    # Retrieval parameters
    TOP_K_RESULTS: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    
    # Processing
    MAX_FILE_SIZE_MB: int = 50
    BATCH_SIZE: int = 10
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/rag_system.log"

    # CORS
    CORS_ALLOW_ORIGINS: str = "*"
    CORS_ALLOW_METHODS: str = "*"
    CORS_ALLOW_HEADERS: str = "*"
    CORS_ALLOW_CREDENTIALS: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
