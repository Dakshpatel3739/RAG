"""
Complete RAG Pipeline - Main orchestration module
"""
from typing import List, Dict, Optional
from pathlib import Path

from modules.pdf_extractor import PDFExtractor
from modules.text_chunker import TextChunker
from modules.embeddings import EmbeddingGenerator
from modules.vector_store import VectorStore
from modules.retriever import Retriever
from modules.llm_generator import LLMGenerator
from utils.logger import log
from config.settings import settings


class RAGPipeline:
    """End-to-end RAG pipeline orchestrator"""
    
    def __init__(
        self,
        use_openai_embeddings: bool = False,
        vector_db_type: str = None,
        collection_name: str = None
    ):
        """
        Initialize RAG pipeline
        
        Args:
            use_openai_embeddings: Whether to use OpenAI for embeddings
            vector_db_type: Type of vector database
            collection_name: Name of vector collection
        """
        log.info("Initializing RAG Pipeline")
        
        # Initialize components
        self.pdf_extractor = PDFExtractor()
        self.text_chunker = TextChunker()
        self.embedding_generator = EmbeddingGenerator(use_openai=use_openai_embeddings)
        self.vector_store = VectorStore(
            db_type=vector_db_type,
            collection_name=collection_name
        )
        self.retriever = Retriever(self.vector_store, self.embedding_generator)
        self.llm_generator = None
        
        log.info("RAG Pipeline initialized successfully")
    
    def ingest_documents(
        self,
        pdf_paths: List[str],
        chunk_strategy: str = "recursive"
    ) -> Dict[str, any]:
        """
        Ingest PDF documents into the system
        
        Args:
            pdf_paths: List of paths to PDF files
            chunk_strategy: Strategy for text chunking
            
        Returns:
            Ingestion statistics
        """
        try:
            log.info(f"Starting document ingestion for {len(pdf_paths)} files")
            
            # Step 1: Extract text from PDFs
            log.info("Step 1: Extracting text from PDFs")
            extracted_docs = self.pdf_extractor.extract_from_multiple_pdfs(pdf_paths)
            
            if not extracted_docs:
                raise ValueError("No documents were successfully extracted")
            
            # Step 2: Chunk the text
            log.info("Step 2: Chunking text")
            self.text_chunker.set_strategy(chunk_strategy)
            chunks = self.text_chunker.chunk_documents(extracted_docs)
            
            # Step 3: Generate embeddings
            log.info("Step 3: Generating embeddings")
            chunks_with_embeddings = self.embedding_generator.embed_chunks(chunks)
            
            # Step 4: Store in vector database
            log.info("Step 4: Storing in vector database")
            self.vector_store.add_embeddings(chunks_with_embeddings)
            
            stats = {
                "documents_processed": len(extracted_docs),
                "chunks_created": len(chunks),
                "embeddings_generated": len(chunks_with_embeddings),
                "status": "success"
            }
            
            log.info(f"Document ingestion completed: {stats}")
            return stats
            
        except Exception as e:
            log.error(f"Error during document ingestion: {str(e)}")
            raise
    
    def ingest_directory(
        self,
        directory_path: str,
        chunk_strategy: str = "recursive"
    ) -> Dict[str, any]:
        """
        Ingest all PDFs from a directory
        
        Args:
            directory_path: Path to directory
            chunk_strategy: Strategy for chunking
            
        Returns:
            Ingestion statistics
        """
        pdf_files = list(Path(directory_path).glob("**/*.pdf"))
        pdf_paths = [str(f) for f in pdf_files]
        
        return self.ingest_documents(pdf_paths, chunk_strategy)
    
    def query(
        self,
        question: str,
        top_k: int = None,
        include_sources: bool = True,
        use_hybrid_search: bool = False
    ) -> Dict[str, any]:
        """
        Query the RAG system
        
        Args:
            question: User question
            top_k: Number of context chunks to retrieve
            include_sources: Whether to include source citations
            use_hybrid_search: Whether to use hybrid retrieval
            
        Returns:
            Generated answer with metadata
        """
        try:
            log.info(f"Processing query: {question[:100]}...")
            
            # Step 1: Retrieve relevant chunks
            log.info("Step 1: Retrieving relevant context")
            if use_hybrid_search:
                relevant_chunks = self.retriever.hybrid_search(question, top_k)
            else:
                relevant_chunks = self.retriever.retrieve(question, top_k)
            
            if not relevant_chunks:
                return {
                    "answer": "I couldn't find relevant information to answer your question.",
                    "query": question,
                    "sources": [],
                    "context_chunks_used": 0,
                    "model": self._default_llm_model()
                }
            
            # Step 2: Generate answer
            log.info("Step 2: Generating answer with LLM")
            llm = self._get_llm_generator()
            result = llm.generate_answer(
                query=question,
                context_chunks=relevant_chunks,
                include_sources=include_sources
            )
            
            log.info("Query processed successfully")
            return result
            
        except Exception as e:
            log.error(f"Error processing query: {str(e)}")
            raise
    
    def conversational_query(
        self,
        question: str,
        conversation_history: List[Dict],
        top_k: int = None
    ) -> Dict[str, any]:
        """
        Query with conversation history
        
        Args:
            question: Current question
            conversation_history: Previous conversation
            top_k: Number of chunks to retrieve
            
        Returns:
            Generated answer with metadata
        """
        try:
            log.info("Processing conversational query")
            
            # Retrieve context
            relevant_chunks = self.retriever.retrieve(question, top_k)
            
            # Generate with history
            llm = self._get_llm_generator()
            result = llm.generate_with_conversation_history(
                query=question,
                context_chunks=relevant_chunks,
                conversation_history=conversation_history
            )
            
            return result
            
        except Exception as e:
            log.error(f"Error in conversational query: {str(e)}")
            raise
    
    def stream_query(self, question: str, top_k: int = None):
        """
        Stream query response
        
        Args:
            question: User question
            top_k: Number of chunks to retrieve
            
        Yields:
            Chunks of generated text
        """
        try:
            # Retrieve context
            relevant_chunks = self.retriever.retrieve(question, top_k)
            
            # Stream answer
            llm = self._get_llm_generator()
            for chunk in llm.stream_answer(question, relevant_chunks):
                yield chunk
                
        except Exception as e:
            log.error(f"Error in streaming query: {str(e)}")
            raise
    
    def get_stats(self) -> Dict[str, any]:
        """Get statistics about the RAG system"""
        try:
            vector_stats = self.vector_store.get_collection_stats()
            
            return {
                "vector_store": vector_stats,
                "embedding_model": self.embedding_generator.model_name,
                "llm_model": self._default_llm_model(),
                "chunk_size": self.text_chunker.chunk_size,
                "chunk_overlap": self.text_chunker.chunk_overlap
            }
            
        except Exception as e:
            log.error(f"Error getting stats: {str(e)}")
            raise
    
    def reset_database(self) -> bool:
        """Reset the vector database"""
        try:
            log.warning("Resetting vector database")
            self.vector_store.delete_collection()
            self.vector_store._initialize_store()
            log.info("Vector database reset successfully")
            return True
            
        except Exception as e:
            log.error(f"Error resetting database: {str(e)}")
            raise

    def _get_llm_generator(self) -> LLMGenerator:
        if self.llm_generator is None:
            self.llm_generator = LLMGenerator()
        return self.llm_generator

    def _default_llm_model(self) -> str:
        if self.llm_generator is not None:
            return self.llm_generator.model
        if settings.LLM_PROVIDER.lower() == "gemini":
            return settings.GEMINI_LLM_MODEL
        return settings.LLM_MODEL
