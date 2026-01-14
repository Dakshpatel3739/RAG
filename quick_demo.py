from rag_pipeline import RAGPipeline

print("="*60)
print("RAG SYSTEM QUICK DEMO")
print("="*60)

print("\n1. Initializing RAG Pipeline...")
rag = RAGPipeline()
print("   ✓ Pipeline ready!")

print("\n2. System Statistics:")
stats = rag.get_stats()
print(f"   - Vector Store: {stats['vector_store']['type']}")
print(f"   - Documents: {stats['vector_store']['count']}")
print(f"   - Embedding Model: {stats['embedding_model']}")
print(f"   - LLM Model: {stats['llm_model']}")

print("\n" + "="*60)
print("✅ Setup complete!")
print("\nNext steps:")
print("  1. Add PDFs to data/pdfs/")
print("  2. Run: python full_demo.py")
print("="*60)
