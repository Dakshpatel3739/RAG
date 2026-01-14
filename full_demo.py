from rag_pipeline import RAGPipeline
import glob
import os

SKIP_EMBEDDINGS = os.getenv("SKIP_EMBEDDINGS", "").lower() in {"1", "true", "yes"}

print("="*60)
print("RAG SYSTEM - FULL DEMO")
print("="*60)

# Initialize
print("\n[1/4] Initializing RAG Pipeline...")
if SKIP_EMBEDDINGS:
    print("      ⚠️  SKIP_EMBEDDINGS enabled - extraction+chunking only")
    rag = None
else:
    rag = RAGPipeline()
    print("      ✓ Pipeline ready!")

# Check for PDFs
print("\n[2/4] Looking for PDF files...")
pdf_files = glob.glob("data/pdfs/*.pdf")

if not pdf_files:
    print("      ⚠️  No PDF files found in data/pdfs/")
    print("\n      Let's create a sample text file instead...")
    
    # Create sample document
    os.makedirs("data/pdfs", exist_ok=True)
    sample_text = """
    Artificial Intelligence and Machine Learning
    
    Machine learning is a subset of artificial intelligence that focuses on 
    the development of algorithms and statistical models that enable computer 
    systems to improve their performance on a specific task through experience.
    
    Deep learning is a type of machine learning based on artificial neural networks.
    It has been successfully applied to various fields including computer vision,
    natural language processing, and speech recognition.
    
    Key concepts in machine learning include:
    - Supervised learning
    - Unsupervised learning
    - Reinforcement learning
    - Neural networks
    - Feature engineering
    """
    
    with open("data/pdfs/sample_ml.txt", "w") as f:
        f.write(sample_text)
    
    print("      ✓ Created sample document: sample_ml.txt")
    
    # For text files, we'll process them as plain text
    from modules.text_chunker import TextChunker
    
    print("\n[3/4] Processing sample document...")
    chunker = TextChunker()
    chunks = chunker.chunk_text(sample_text, {"filename": "sample_ml.txt"})
    
    if SKIP_EMBEDDINGS:
        print(f"      ✓ Created {len(chunks)} chunks (embeddings skipped)")
        if chunks:
            preview = " ".join(chunks[0]["text"].split())
            print(f"      ✓ Sample chunk: {preview[:200]}{'...' if len(preview) > 200 else ''}")
    else:
        from modules.embeddings import EmbeddingGenerator
        
        embedder = EmbeddingGenerator()
        chunks_with_embeddings = embedder.embed_chunks(chunks)
        
        rag.vector_store.add_embeddings(chunks_with_embeddings)
        print(f"      ✓ Created {len(chunks)} chunks")
    
else:
    print(f"      ✓ Found {len(pdf_files)} PDF file(s)")
    for pdf in pdf_files:
        print(f"        - {os.path.basename(pdf)}")
    
    # Ingest PDFs
    print("\n[3/4] Ingesting documents...")
    if SKIP_EMBEDDINGS:
        from modules.pdf_extractor import PDFExtractor
        from modules.text_chunker import TextChunker
        
        extractor = PDFExtractor()
        documents = extractor.extract_from_multiple_pdfs(pdf_files)
        chunker = TextChunker()
        chunks = chunker.chunk_documents(documents)
        
        print(f"      ✓ Processed: {len(documents)} document(s)")
        print(f"      ✓ Created: {len(chunks)} chunks")
        if chunks:
            preview = " ".join(chunks[0]["text"].split())
            print(f"      ✓ Sample chunk: {preview[:200]}{'...' if len(preview) > 200 else ''}")
    else:
        try:
            stats = rag.ingest_documents(pdf_files)
            print(f"      ✓ Processed: {stats['documents_processed']} document(s)")
            print(f"      ✓ Created: {stats['chunks_created']} chunks")
        except Exception as e:
            print(f"      ✗ Error: {e}")
            exit(1)

# Query the system
print("\n[4/4] Testing queries...")
print("="*60)

if SKIP_EMBEDDINGS:
    print("⚠️  Skipping queries because embeddings/LLM are disabled.")
else:
    questions = [
        "What is this document about?",
        "What are the main topics discussed?",
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\nQ{i}: {question}")
        print("-" * 60)
        
        try:
            result = rag.query(question, top_k=3)
            print(f"A{i}: {result['answer']}\n")
            
            if result.get('sources'):
                print(f"    Sources: {len(result['sources'])} document(s)")
                
        except Exception as e:
            print(f"    ✗ Error: {e}\n")

print("="*60)
print("✅ Demo Complete!")
print("\nTo add your own PDFs:")
print("  1. Upload PDFs to data/pdfs/")
print("  2. Run: python full_demo.py")
print("\nTo start API server:")
print("  python api/main.py")
print("="*60)
