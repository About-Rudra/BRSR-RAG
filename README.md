# Compliance RAG Chatbot

A RAG-based chatbot for SEBI, CPCB, and CCTS regulations.

## Setup
1. Create virtual env: `python -m venv venv`
2. Activate: `source venv/bin/activate` (or Windows equivalent)
3. Install deps: `pip install -r requirements.txt`

## Running Scrapers
python main.py --scrape

## Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# 2. Start Redis if using queues
redis-server

## 3. Then run ingestion:
python main.py --ingest

## 4. Install & Start Ollama
ollama serve

## 5. Pull a lightweight model
ollama pull phi3:mini

# 7. Start backend
uvicorn src.api.main:app --reload
