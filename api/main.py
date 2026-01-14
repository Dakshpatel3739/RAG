"""
FastAPI REST API for RAG system
"""
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import tempfile
import os
import glob

from rag_pipeline import RAGPipeline
from utils.logger import log
from config.settings import settings

# Helpers
def _parse_csv_setting(value: str) -> List[str]:
    if not value:
        return []
    value = value.strip()
    if value == "*":
        return ["*"]
    return [item.strip() for item in value.split(",") if item.strip()]

def _get_bearer_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    parts = authorization.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return None

def require_api_key(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None),
):
    expected = settings.API_AUTH_TOKEN
    if not expected:
        return
    token = x_api_key or _get_bearer_token(authorization)
    if not token or token != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")

# Initialize FastAPI app
app = FastAPI(
    title="RAG System API",
    description="Retrieval-Augmented Generation API for document Q&A",
    version="1.0.0"
)

# Track app start time for diagnostics
APP_STARTED_AT = datetime.now(timezone.utc)

# CORS middleware
cors_origins = _parse_csv_setting(settings.CORS_ALLOW_ORIGINS)
cors_allow_credentials = settings.CORS_ALLOW_CREDENTIALS and cors_origins != ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=cors_allow_credentials,
    allow_methods=_parse_csv_setting(settings.CORS_ALLOW_METHODS),
    allow_headers=_parse_csv_setting(settings.CORS_ALLOW_HEADERS),
)

# Initialize RAG pipeline
rag_pipeline = None

@app.on_event("startup")
async def startup_event():
    """Initialize RAG pipeline on startup"""
    global rag_pipeline
    global APP_STARTED_AT
    APP_STARTED_AT = datetime.now(timezone.utc)
    log.info("Initializing RAG Pipeline...")
    try:
        rag_pipeline = RAGPipeline()
        log.info("RAG Pipeline initialized successfully")
    except Exception as e:
        log.error(f"Failed to initialize RAG Pipeline: {e}")
        raise

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    include_sources: bool = True
    use_hybrid_search: bool = False

class ConversationalQueryRequest(BaseModel):
    question: str
    conversation_history: List[dict] = []
    top_k: Optional[int] = None

class QueryResponse(BaseModel):
    answer: str
    query: str
    sources: List[dict] = []
    context_chunks_used: int
    model: str

class RetrieveRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    use_hybrid_search: bool = False

class RetrievedChunk(BaseModel):
    id: str
    text: str
    metadata: dict
    score: float

class RetrieveResponse(BaseModel):
    query: str
    chunks: List[RetrievedChunk]
    total: int

class IngestionResponse(BaseModel):
    documents_processed: int
    chunks_created: int
    embeddings_generated: int
    status: str

class StatsResponse(BaseModel):
    vector_store: dict
    embedding_model: str
    llm_model: str
    chunk_size: int
    chunk_overlap: int

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RAG System API",
        "version": "1.0.0",
        "status": "operational"
    }

@app.get("/ui", response_class=HTMLResponse)
async def ui():
    """Simple local UI for ingestion and retrieval"""
    try:
        html_path = Path(__file__).parent / "static" / "index.html"
        return HTMLResponse(html_path.read_text())
    except Exception as e:
        log.error(f"Failed to load UI: {str(e)}")
        raise HTTPException(status_code=500, detail="UI not available")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        if rag_pipeline is None:
            raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
        stats = rag_pipeline.get_stats()
        return {
            "status": "healthy",
            "stats": stats
        }
    except Exception as e:
        log.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/diagnostics", dependencies=[Depends(require_api_key)])
async def diagnostics():
    """Basic diagnostics for monitoring (no secrets)"""
    base = {
        "status": "starting" if rag_pipeline is None else "ok",
        "started_at": APP_STARTED_AT.isoformat(),
        "uptime_seconds": int((datetime.now(timezone.utc) - APP_STARTED_AT).total_seconds()),
        "providers": {
            "embedding": settings.EMBEDDING_PROVIDER,
            "llm": settings.LLM_PROVIDER
        },
        "auth_enabled": bool(settings.API_AUTH_TOKEN)
    }
    if rag_pipeline is None:
        return base
    try:
        stats = rag_pipeline.get_stats()
        base.update({
            "vector_store": stats.get("vector_store"),
            "embedding_model": stats.get("embedding_model"),
            "llm_model": stats.get("llm_model"),
            "chunk_size": stats.get("chunk_size"),
            "chunk_overlap": stats.get("chunk_overlap")
        })
        return base
    except Exception as e:
        log.error(f"Diagnostics failed: {str(e)}")
        base["status"] = "error"
        base["error"] = str(e)
        return base

@app.post("/ingest/upload", response_model=IngestionResponse, dependencies=[Depends(require_api_key)])
async def upload_and_ingest(
    files: List[UploadFile] = File(...),
    chunk_strategy: str = "recursive",
    background_tasks: BackgroundTasks = None
):
    """Upload and ingest PDF documents"""
    try:
        log.info(f"Received {len(files)} files for ingestion")
        
        temp_paths = []
        for file in files:
            if not file.filename.endswith('.pdf'):
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} is not a PDF"
                )
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                content = await file.read()
                tmp.write(content)
                temp_paths.append(tmp.name)
        
        result = rag_pipeline.ingest_documents(temp_paths, chunk_strategy)
        
        if background_tasks:
            for path in temp_paths:
                background_tasks.add_task(os.unlink, path)
        
        return IngestionResponse(**result)
        
    except Exception as e:
        log.error(f"Error during ingestion: {str(e)}")
        for path in temp_paths:
            try:
                os.unlink(path)
            except:
                pass
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
async def query(request: QueryRequest):
    """Query the RAG system"""
    try:
        if rag_pipeline is None:
            raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
        result = rag_pipeline.query(
            question=request.question,
            top_k=request.top_k,
            include_sources=request.include_sources,
            use_hybrid_search=request.use_hybrid_search
        )
        return QueryResponse(**result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrieve", response_model=RetrieveResponse, dependencies=[Depends(require_api_key)])
async def retrieve(request: RetrieveRequest):
    """Retrieve relevant chunks without calling an LLM"""
    try:
        if rag_pipeline is None:
            raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
        if request.use_hybrid_search:
            results = rag_pipeline.retriever.hybrid_search(
                request.question,
                top_k=request.top_k
            )
        else:
            results = rag_pipeline.retriever.retrieve(
                request.question,
                top_k=request.top_k
            )
        chunks = [RetrievedChunk(**r) for r in results]
        return RetrieveResponse(
            query=request.question,
            chunks=chunks,
            total=len(chunks)
        )
        
    except Exception as e:
        log.error(f"Error retrieving chunks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", response_model=StatsResponse, dependencies=[Depends(require_api_key)])
async def get_stats():
    """Get system statistics"""
    try:
        stats = rag_pipeline.get_stats()
        return StatsResponse(**stats)
        
    except Exception as e:
        log.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset", dependencies=[Depends(require_api_key)])
async def reset_database():
    """Reset the vector database"""
    try:
        rag_pipeline.reset_database()
        return {"message": "Database reset successfully"}
        
    except Exception as e:
        log.error(f"Error resetting database: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
