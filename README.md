# Compliance RAG Chatbot

A RAG-based chatbot for SEBI, CPCB, and CCTS regulations.

Documents are stored under `data/raw/<regime>/`, where `<regime>` is one of
`sebi_brsr`, `cpcb_epr`, `ccts`. The same key is stored as the `category` of each chunk in Qdrant.

## Setup
1. Create virtual env: `python -m venv venv`
2. Activate: `source venv/bin/activate` (or Windows equivalent)
3. Install deps: `pip install -r requirements.txt`

## Configure
Copy `.env.example` to `.env` and fill in your API key.

## Qdrant
By default (`QDRANT_MODE=local`) Qdrant runs embedded and stores its data in `vector_db/`;
no server is needed. To use a server instead, set `QDRANT_MODE=server` and run
`docker run -p 6333:6333 qdrant/qdrant`.

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

## LLM
Set `LLM_PROVIDER` (`gemini` default, `groq`, or `ollama`) and `LLM_MODEL`. The baseline
chatbot and the agent read the same settings. Ollama is optional: `pip install langchain-ollama`,
`ollama serve`, `ollama pull phi3:mini`.

## Research agent
python main.py --ask "What must obligated entities do under CCTS, and what is BRSR Core?"

A LangGraph workflow in `src/agent/`: the **Planner** splits the question into up to 4
sub-questions tagged by regime; the **Researcher** answers each with read-only tools
(`search_regulations`, `list_sources`, `fetch_official_page`, max 6 calls per sub-question);
the **Verifier** checks every claim against its cited passages (first in code, then with the
LLM) and sends rejected work back once; the **Writer** answers from verified claims only, with
document and page citations and a "Could not verify" section. The full trace is saved to
`runs/research_<timestamp>.json`. The single-call baseline (`src/rag/chatbot.py::chat`) is unchanged.

## Regulation watch
python main.py --watch                                   # live check of the SEBI, CPCB and BEE sites
python main.py --watch --simulate path/to/new.pdf        # SIMULATION: treat a local PDF as newly found
python main.py --pending                                 # list summaries waiting for approval
python main.py --approve <id>   /   --reject <id>

New or changed documents (SHA-256 diff against `data/registry.json`) pass a keyword-then-LLM
relevance filter, are ingested, summarised with page citations, checked by the same verifier as
the research agent, and wait in `data/updates_pending.json` until a person approves them into
`data/updates_log.json`.

## Evaluation
See [eval/README.md](eval/README.md): write your own questions in `eval/questions.jsonl`, then
`python -m eval.run_eval` and `python -m eval.summarise <summary.csv>`.

## Web UI
streamlit run streamlit_app.py

For a public link, see [docs/deploy_streamlit.md](docs/deploy_streamlit.md).

## Tests
pytest -m "not llm"   # no API key needed
pytest               # also runs the live-LLM behaviour tests

## Start backend (baseline API)
uvicorn src.api.main:app --reload
