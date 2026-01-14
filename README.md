# RAG System

Retrieval-augmented Q&A over PDF documents with a FastAPI service and a local vector database.

## How it works
1. Ingest PDFs: extract text, split into chunks, generate embeddings, and store in Chroma.
2. Query: embed the question, retrieve the most similar chunks, then the LLM answers with citations.
3. Retrieve-only: skip the LLM and just return the best chunks.

## Local setup (Python)
Prereqs: Python 3.11+ and `pip`.

1. Create your env file:
   `cp .env.example .env`
2. Set your provider and API key in `.env`:
   - OpenAI:
     - `EMBEDDING_PROVIDER=openai`
     - `LLM_PROVIDER=openai`
     - `OPENAI_API_KEY=...`
   - Gemini:
     - `EMBEDDING_PROVIDER=gemini`
     - `LLM_PROVIDER=gemini`
     - `GEMINI_API_KEY=...`
3. Create a virtual environment and install dependencies:
   `python3 -m venv .venv`
   `source .venv/bin/activate`
   `pip install -r requirements-prod.txt`
4. Start the API:
   `uvicorn api.main:app --host 0.0.0.0 --port 8000`

## Use the system (local)
You can use the simple UI or call the API directly.

### UI
Open `http://localhost:8000/ui`, upload PDFs, then ask questions.

### API examples
If you set `API_AUTH_TOKEN`, include `-H "X-API-Key: <token>"` in requests.

Health check:
```bash
curl http://localhost:8000/health
```

Diagnostics:
```bash
curl http://localhost:8000/diagnostics
```

Ingest PDFs:
```bash
curl -F "files=@/path/to/file.pdf" http://localhost:8000/ingest/upload
```

Query:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is this document about?","top_k":5,"include_sources":true}'
```

Retrieve-only (no LLM):
```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"question":"What is this document about?","top_k":5}'
```

Reset the vector store:
```bash
curl -X POST http://localhost:8000/reset
```

## Docker
Build and run:
```bash
docker build -t rag-system .
docker run --rm -p 8000:8000 --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  rag-system
```

## Configuration
All settings are managed via environment variables. See `.env.example` for the full list.

Key settings:
- Providers: `EMBEDDING_PROVIDER`, `LLM_PROVIDER`
- Models: `EMBEDDING_MODEL`, `LLM_MODEL`, `GEMINI_EMBEDDING_MODEL`, `GEMINI_LLM_MODEL`
- Storage: `VECTOR_DB_PATH`, `COLLECTION_NAME`
- Retrieval: `TOP_K_RESULTS`, `SIMILARITY_THRESHOLD`
- CORS: `CORS_ALLOW_ORIGINS`, `CORS_ALLOW_METHODS`, `CORS_ALLOW_HEADERS`
- Auth: `API_AUTH_TOKEN` (set this to require `X-API-Key` or `Authorization: Bearer`)

Notes:
- Vector data persists in `./data/vector_db`.
- Logs are written to `./logs/rag_system.log`.
- Do not commit `.env` to source control.
- If you use local Chroma persistence, keep Uvicorn workers at 1 (default in Docker).

## Testing
Run unit tests (no external API calls):
```bash
pytest -m "not integration"
```

Integration tests require API keys:
```bash
pytest -m "integration"
```
