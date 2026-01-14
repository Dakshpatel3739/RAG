"""
Vector database management module
"""
import os
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import chromadb
from chromadb.config import Settings as ChromaSettings
import numpy as np
from utils.logger import log
from config.settings import settings


class VectorStore:
    """Manage vector database operations"""
    
    def __init__(
        self,
        db_type: str = None,
        persist_directory: str = None,
        collection_name: str = None
    ):
        """
        Initialize vector store
        
        Args:
            db_type: Type of vector database ('chroma', 'faiss', 'pinecone')
            persist_directory: Directory to persist the database
            collection_name: Name of the collection
        """
        self.db_type = db_type or settings.VECTOR_DB_TYPE
        self.persist_directory = persist_directory or settings.VECTOR_DB_PATH
        self.collection_name = collection_name or settings.COLLECTION_NAME
        
        # Create directory if it doesn't exist
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        self.client = None
        self.collection = None
        
        self._initialize_store()
    
    def _initialize_store(self):
        """Initialize the vector store based on type"""
        
        if self.db_type == "chroma":
            self._initialize_chroma()
        elif self.db_type == "faiss":
            self._initialize_faiss()
        elif self.db_type == "pinecone":
            self._initialize_pinecone()
        else:
            raise ValueError(f"Unsupported vector database type: {self.db_type}")
    
    def _initialize_chroma(self):
        """Initialize ChromaDB"""
        try:
            log.info("Initializing ChromaDB")
            
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            log.info(f"ChromaDB initialized. Collection: {self.collection_name}")
            
        except Exception as e:
            log.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise
    
    def _initialize_faiss(self):
        """Initialize FAISS (placeholder for future implementation)"""
        log.warning("FAISS initialization not yet implemented")
        raise NotImplementedError("FAISS support coming soon")
    
    def _initialize_pinecone(self):
        """Initialize Pinecone (placeholder for future implementation)"""
        log.warning("Pinecone initialization not yet implemented")
        raise NotImplementedError("Pinecone support coming soon")
    
    def add_embeddings(
        self,
        chunks: List[Dict],
        embeddings: Optional[List[List[float]]] = None
    ) -> bool:
        """
        Add embeddings to vector store
        
        Args:
            chunks: List of chunk dictionaries
            embeddings: Optional list of embeddings (if not in chunks)
            
        Returns:
            Success status
        """
        try:
            if self.db_type == "chroma":
                return self._add_to_chroma(chunks, embeddings)
            else:
                raise NotImplementedError(f"Add operation not implemented for {self.db_type}")
                
        except Exception as e:
            log.error(f"Error adding embeddings: {str(e)}")
            raise
    
    def _add_to_chroma(
        self,
        chunks: List[Dict],
        embeddings: Optional[List[List[float]]] = None
    ) -> bool:
        """Add chunks to ChromaDB"""
        try:
            log.info(f"Adding {len(chunks)} chunks to ChromaDB")
            
            # Prepare data
            ids = [chunk["id"] for chunk in chunks]
            documents = [chunk["text"] for chunk in chunks]
            metadatas = [chunk.get("metadata", {}) for chunk in chunks]
            
            # Get embeddings
            if embeddings is None:
                embeddings = [chunk.get("embedding") for chunk in chunks]
                if None in embeddings:
                    raise ValueError("Chunks must contain embeddings or embeddings must be provided")
            
            # Add to collection (upsert to avoid duplicate ID errors)
            if hasattr(self.collection, "upsert"):
                self.collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas
                )
            else:
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas
                )
            
            log.info(f"Successfully added {len(chunks)} chunks to ChromaDB")
            return True
            
        except Exception as e:
            log.error(f"Error adding to ChromaDB: {str(e)}")
            raise
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = None,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar vectors
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of similar chunks with scores
        """
        try:
            top_k = top_k or settings.TOP_K_RESULTS
            
            if self.db_type == "chroma":
                return self._search_chroma(query_embedding, top_k, filter_dict)
            else:
                raise NotImplementedError(f"Search not implemented for {self.db_type}")
                
        except Exception as e:
            log.error(f"Error searching vector store: {str(e)}")
            raise
    
    def _search_chroma(
        self,
        query_embedding: List[float],
        top_k: int,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """Search ChromaDB"""
        try:
            log.info(f"Searching ChromaDB for top {top_k} results")
            
            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter_dict
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    "id": results['ids'][0][i],
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "score": 1 - results['distances'][0][i]  # Convert distance to similarity
                }
                formatted_results.append(result)
            
            log.info(f"Found {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            log.error(f"Error searching ChromaDB: {str(e)}")
            raise
    
    def delete_collection(self) -> bool:
        """Delete the current collection"""
        try:
            if self.db_type == "chroma":
                self.client.delete_collection(name=self.collection_name)
                log.info(f"Deleted collection: {self.collection_name}")
                return True
            else:
                raise NotImplementedError(f"Delete not implemented for {self.db_type}")
                
        except Exception as e:
            log.error(f"Error deleting collection: {str(e)}")
            raise
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        try:
            if self.db_type == "chroma":
                count = self.collection.count()
                return {
                    "name": self.collection_name,
                    "count": count,
                    "type": self.db_type
                }
            else:
                raise NotImplementedError(f"Stats not implemented for {self.db_type}")
                
        except Exception as e:
            log.error(f"Error getting collection stats: {str(e)}")
            raise
