"""Regression tests for the Phase 0 bug fixes and the read-only tools (temporary in-memory Qdrant)."""
import pytest
from qdrant_client import QdrantClient

import src.rag.vector_client as vector_client
from src.rag import ingestion, retrieval
from src.agent import tools

PAGE_TEXT = [
    "Obligated entities shall surrender carbon credit certificates equal to their shortfall. " * 12,
    "Producers of plastic packaging must register on the centralised EPR portal. " * 12,
    "The top 1000 listed entities shall file a BRSR with their annual report. " * 12,
]


class FakePage:
    def __init__(self, text):
        self.text = text

    def extract_text(self):
        return self.text


class FakePDF:
    def __init__(self, texts):
        self.pages = [FakePage(t) for t in texts]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def memory_db(monkeypatch, tmp_path):
    client = QdrantClient(":memory:")
    monkeypatch.setattr(vector_client, "_client", client)
    monkeypatch.setattr(ingestion.pdfplumber, "open", lambda path: FakePDF(PAGE_TEXT))
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    return client, str(pdf)


def test_retrieval_empty_collection_returns_empty_list(memory_db):
    assert retrieval.retrieve("anything") == []


def test_retrieval_returns_multiple(memory_db):
    # Regression for bug 0.1: results.append sat outside the loop, so only one chunk came back.
    client, pdf = memory_db
    ingestion.ingest_single_file(pdf, "ccts")
    results = retrieval.retrieve("carbon credit certificates", top_k=3, score_threshold=None)
    assert len(results) == 3
    assert results[0]["score"] >= results[-1]["score"]
    assert all(r["page"] in (1, 2, 3) and r["category"] == "ccts" for r in results)


def test_retrieval_regime_filter(memory_db):
    client, pdf = memory_db
    ingestion.ingest_single_file(pdf, "ccts")
    assert retrieval.retrieve("plastic", top_k=5, score_threshold=None, category="sebi_brsr") == []


def test_ingest_ids_stable(memory_db):
    # Regression for bug 0.6: hash()-based ids changed every run and duplicated points.
    client, pdf = memory_db
    ingestion.ingest_single_file(pdf, "ccts")
    first = sorted(str(p.id) for p in client.scroll(ingestion.COLLECTION, limit=1000)[0])
    ingestion.ingest_single_file(pdf, "ccts")
    second = sorted(str(p.id) for p in client.scroll(ingestion.COLLECTION, limit=1000)[0])
    assert first == second and len(first) > 0
    assert ingestion.point_id("ccts", "doc.pdf", 0) == ingestion.point_id("ccts", "doc.pdf", 0)


class ReadOnlyClient:
    """Wraps a client and fails the test if anything tries to write."""

    WRITES = {"upsert", "delete", "create_collection", "delete_collection", "recreate_collection",
              "create_payload_index", "set_payload", "overwrite_payload", "update_vectors"}

    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        if name in self.WRITES:
            raise AssertionError(f"tool attempted a write: {name}")
        return getattr(self.inner, name)


def test_tools_are_read_only(memory_db, monkeypatch, tmp_path):
    client, pdf = memory_db
    ingestion.ingest_single_file(pdf, "ccts")
    monkeypatch.setattr(vector_client, "_client", ReadOnlyClient(client))
    before = sorted(p.name for p in tmp_path.iterdir())
    monkeypatch.chdir(tmp_path)

    assert tools.search_regulations("surrender", regime="ccts", top_k=2)
    assert tools.list_sources("any")[0]["chunks"] == client.count(ingestion.COLLECTION).count
    assert tools.fetch_official_page("https://example.com")["ok"] is False

    assert sorted(p.name for p in tmp_path.iterdir()) == before  # no files written
