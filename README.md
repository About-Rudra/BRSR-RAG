# Compliance RAG Chatbot

A RAG-based chatbot for SEBI, CPCB, and CCTS regulations.

Documents are stored under `data/raw/<regime>/`, where `<regime>` is one of
`sebi_brsr`, `cpcb_epr`, `ccts`. The same key is stored as the `category` of each chunk in Qdrant.

## Setup
1. Create virtual env: `python -m venv venv`
2. Activate: `source venv/bin/activate` (or Windows equivalent)
3. Install deps: `pip install -r requirements.txt`

## Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

## Download documents
python main.py --scrape

Downloads the key PDFs listed under `direct_pdfs` in `src/config/config.yaml` and runs the
SEBI / CPCB / CCTS scrapers. Government sites are often slow or unreachable; failures are
logged and skipped.

By default (`INGEST_MODE=sync`) each downloaded PDF is ingested straight away.
Set `INGEST_MODE=redis` to push jobs onto a Redis stream instead, then run a worker with
`redis-server` and `python -m src.queue.worker`.

## Ingest
python main.py --ingest

Ingests every PDF under `data/raw/sebi_brsr/`, `data/raw/cpcb_epr/` and `data/raw/ccts/`
synchronously. Safe to re-run: chunk IDs are deterministic, so re-ingesting a file replaces
its chunks instead of duplicating them.

## Install & Start Ollama
ollama serve

## Pull a lightweight model
ollama pull phi3:mini

## Start backend
uvicorn src.api.main:app --reload
