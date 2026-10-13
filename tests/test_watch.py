"""Deterministic tests for the regulation watch (no API key, no network, temporary data folder)."""
import json

import pytest

import src.rag.ingestion as ingestion
import src.rag.llm as llm
from src.agent.researcher import ClaimModel
from src.agent.verifier import ClaimVerdict, VerdictListModel
from src.watch import approvals, pipeline
from src.watch import registry as reg
from src.watch.relevance import RelevanceModel, keyword_match
from src.watch.run import run_watch
from tests.fake_llm import FakeChatModel

PAGES = [(1, "SEBI circular on Business Responsibility and Sustainability Reporting."),
         (2, "Applies to the top 1000 listed entities from financial year 2022-23.")]


def test_watch_diff():
    registry = {}
    assert reg.classify(registry, "https://x/a.pdf", "h1") == "new"
    reg.record(registry, "https://x/a.pdf", "A", "ccts", "h1", "ingested")
    assert reg.classify(registry, "https://x/a.pdf", "h1") == "unchanged"
    assert reg.classify(registry, "https://x/a.pdf", "h2") == "changed"
    assert reg.classify(registry, "https://x/b.pdf", "h1") == "new"


def test_keyword_match_tags_regime():
    assert keyword_match("Framework for BRSR Core assurance")[0] == "sebi_brsr"
    assert keyword_match("Guidelines on Extended Producer Responsibility")[0] == "cpcb_epr"
    assert keyword_match("Detailed procedure for Offset Mechanism")[0] == "ccts"
    assert keyword_match("Recruitment notice for drivers") == (None, None)
    assert keyword_match("Deprecated API notes")[1] is None  # 'epr' inside a word is not a match


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Run in an empty folder with ingestion and PDF parsing stubbed out."""
    monkeypatch.chdir(tmp_path)
    ingested = []
    monkeypatch.setattr(ingestion, "ingest_single_file", lambda path, regime: ingested.append((path, regime)))
    monkeypatch.setattr(pipeline, "read_pages", lambda data: PAGES)
    pdf = tmp_path / "new_circular.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    return pdf, ingested


def fake_model(relevant=True):
    def structured(schema, prompt):
        if schema is RelevanceModel:
            return RelevanceModel(relevant=relevant, regime="sebi_brsr", reason="about BRSR")
        if schema is pipeline.SummaryModel:
            return pipeline.SummaryModel(
                what_it_is=[ClaimModel(text="A SEBI circular on BRSR.", sids=["S1"])],
                who_is_affected=[ClaimModel(text="Top 1000 listed entities.", sids=["S2"]),
                                 ClaimModel(text="All unlisted companies.", sids=["S9"])])
        if schema is VerdictListModel:
            return VerdictListModel(verdicts=[ClaimVerdict(claim_number=1, supported=True, reason="p1"),
                                              ClaimVerdict(claim_number=2, supported=True, reason="p2")])
    return FakeChatModel(structured=structured)


def test_simulation_reaches_pending_then_log_only_after_approval(sandbox, monkeypatch):
    pdf, ingested = sandbox
    monkeypatch.setattr(llm, "get_chat_model", lambda: fake_model())
    summary = run_watch(simulate_path=str(pdf), title="New BRSR circular")

    assert summary["mode"] == "simulation"
    result = summary["processed"][0]
    assert result["status"] == "pending_approval" and result["simulated"] is True
    assert ingested and ingested[0][1] == "sebi_brsr"

    pending = approvals.load_pending()
    assert len(pending) == 1 and pending[0]["simulated"] is True
    entry = pending[0]
    assert [c["text"] for c in entry["summary"]["who_is_affected"]] == ["Top 1000 listed entities."]
    assert entry["unverified"][0]["text"] == "All unlisted companies."  # cited a page that does not exist
    assert approvals.load_log() == []  # nothing is logged before a human decides

    approvals.approve(entry["id"])
    assert [e["id"] for e in approvals.load_log()] == [entry["id"]]
    assert reg.load_registry()[entry["url"]]["status"] == "approved"

    # The same file again is unchanged: no second pending entry.
    again = run_watch(simulate_path=str(pdf))
    assert again["processed"] == [] and again["skipped"]


def test_irrelevant_document_is_recorded_not_ingested(sandbox, monkeypatch):
    pdf, ingested = sandbox
    monkeypatch.setattr(llm, "get_chat_model", lambda: fake_model(relevant=False))
    summary = run_watch(simulate_path=str(pdf), title="Canteen tender")
    assert summary["processed"][0]["status"] == "irrelevant"
    assert ingested == [] and approvals.load_pending() == []
    assert reg.load_registry()["simulated://new_circular.pdf"]["status"] == "irrelevant"


def test_reject_never_reaches_log(sandbox, monkeypatch):
    pdf, _ = sandbox
    monkeypatch.setattr(llm, "get_chat_model", lambda: fake_model())
    run_watch(simulate_path=str(pdf), title="New BRSR circular")
    entry = approvals.load_pending()[0]
    approvals.reject(entry["id"])
    assert approvals.load_log() == []
    assert approvals.load_pending()[0]["status"] == "rejected"
    with pytest.raises(KeyError):
        approvals.approve(entry["id"])  # a decided entry cannot be approved afterwards
