"""
Text chunking module with multiple strategies
"""
from typing import List, Dict, Optional
from utils.logger import log
from config.settings import settings


class RecursiveCharacterTextSplitter:
    """Simple recursive text splitter"""
    def __init__(self, chunk_size, chunk_overlap, separators=None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]
    
    def split_text(self, text: str) -> List[str]:
        chunks = []
        current_separator = self.separators[0] if self.separators else "\n\n"
        
        # Split by separator
        splits = text.split(current_separator)
        current_chunk = ""
        
        for split in splits:
            if len(current_chunk) + len(split) <= self.chunk_size:
                current_chunk += split + current_separator
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = split + current_separator
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks


class CharacterTextSplitter:
    """Simple character-based text splitter"""
    def __init__(self, chunk_size, chunk_overlap, separator="\n"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
    
    def split_text(self, text: str) -> List[str]:
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap
        
        return chunks


class TokenTextSplitter:
    """Token-based text splitter"""
    def __init__(self, chunk_size, chunk_overlap):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        # Simple word-based splitting as proxy for tokens
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunks.append(" ".join(chunk_words))
        
        return chunks


class TextChunker:
    """Split text into manageable chunks for embedding"""
    
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        strategy: str = "recursive"
    ):
        """
        Initialize text chunker
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
            strategy: Chunking strategy ('recursive', 'character', 'token')
        """
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.strategy = strategy
        
        self.splitter = self._initialize_splitter()
    
    def _initialize_splitter(self):
        """Initialize the appropriate text splitter based on strategy"""
        
        if self.strategy == "recursive":
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
        elif self.strategy == "character":
            return CharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separator="\n"
            )
        elif self.strategy == "token":
            return TokenTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")

    def set_strategy(self, strategy: str) -> None:
        """Update chunking strategy and reinitialize the splitter"""
        if strategy != self.strategy:
            self.strategy = strategy
            self.splitter = self._initialize_splitter()
    
    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict[str, any]]:
        """
        Split text into chunks
        
        Args:
            text: Input text to chunk
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        try:
            log.info(f"Chunking text of length {len(text)} using {self.strategy} strategy")
            
            # Split text
            chunks = self.splitter.split_text(text)
            
            # Create chunk objects with metadata
            chunk_objects = []
            for i, chunk in enumerate(chunks):
                file_hash = metadata.get("file_hash") if metadata else None
                file_tag = file_hash[:8] if file_hash else (metadata.get("filename", "unknown") if metadata else "unknown")
                chunk_obj = {
                    "id": f"{metadata.get('filename', 'unknown')}_{file_tag}_{i}" if metadata else f"chunk_{i}",
                    "text": chunk,
                    "chunk_index": i,
                    "metadata": metadata or {}
                }
                chunk_objects.append(chunk_obj)
            
            log.info(f"Created {len(chunk_objects)} chunks")
            return chunk_objects
            
        except Exception as e:
            log.error(f"Error chunking text: {str(e)}")
            raise
    
    def chunk_documents(self, documents: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """
        Chunk multiple documents
        
        Args:
            documents: List of document dictionaries with 'text' and 'metadata' keys
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        
        for doc in documents:
            try:
                text = doc.get("text", "")
                metadata = doc.get("metadata", {})
                
                chunks = self.chunk_text(text, metadata)
                all_chunks.extend(chunks)
                
            except Exception as e:
                log.error(f"Failed to chunk document: {str(e)}")
                continue
        
        log.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
    
    def adaptive_chunking(self, text: str, metadata: Optional[Dict] = None) -> List[Dict[str, any]]:
        """
        Adaptive chunking that respects semantic boundaries
        
        Args:
            text: Input text to chunk
            metadata: Optional metadata
            
        Returns:
            List of semantically meaningful chunks
        """
        # Split by paragraphs first
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for para in paragraphs:
            # If adding this paragraph exceeds chunk size, save current chunk
            if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                chunks.append({
                    "id": f"{metadata.get('filename', 'unknown')}_{chunk_index}" if metadata else f"chunk_{chunk_index}",
                    "text": current_chunk.strip(),
                    "chunk_index": chunk_index,
                    "metadata": metadata or {}
                })
                chunk_index += 1
                current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        # Add the last chunk
        if current_chunk:
            chunks.append({
                "id": f"{metadata.get('filename', 'unknown')}_{chunk_index}" if metadata else f"chunk_{chunk_index}",
                "text": current_chunk.strip(),
                "chunk_index": chunk_index,
                "metadata": metadata or {}
            })
        
        log.info(f"Adaptive chunking created {len(chunks)} chunks")
        return chunks
