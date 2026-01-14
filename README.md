# RAG System

FastAPI API for ingestion and retrieval-augmented Q&A over PDF documents.

## Quick start (local)
1. Copy env file and set keys: `cp .env.example .env`
2. Create a virtualenv and install deps:
   `python3 -m venv .venv && source .venv/bin/activate`
   `pip install -r requirements-prod.txt`
3. Run the API:
   `uvicorn api.main:app --host 0.0.0.0 --port 8000`

Open `http://localhost:8000/ui` or use the API endpoints:
- `POST /ingest/upload`
- `POST /query`
- `GET /health`

## Docker (production)
Build and run:
- `docker build -t rag-system .`
- `docker run --rm -p 8000:8000 --env-file .env -v $(pwd)/data:/app/data -v $(pwd)/logs:/app/logs rag-system`

## Configuration
All settings are managed via environment variables. See `.env.example` for the full list.

Production notes:
- Set `CORS_ALLOW_ORIGINS` to a comma-separated list of allowed origins (not `*`).
- Persist vector data by mounting `./data` into the container.
- Do not commit `.env` to source control.
