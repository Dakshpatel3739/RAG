print("Testing imports...")
try:
    from config.settings import settings
    print("✓ Config")
    from utils.logger import log
    print("✓ Logger")
    from modules.pdf_extractor import PDFExtractor
    print("✓ PDF Extractor")
    from modules.text_chunker import TextChunker
    print("✓ Text Chunker")
    from modules.embeddings import EmbeddingGenerator
    print("✓ Embeddings")
    from modules.vector_store import VectorStore
    print("✓ Vector Store")
    from modules.retriever import Retriever
    print("✓ Retriever")
    from modules.llm_generator import LLMGenerator
    print("✓ LLM Generator")
    from rag_pipeline import RAGPipeline
    print("✓ RAG Pipeline")
    print("\n✅ ALL IMPORTS SUCCESSFUL!")
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
