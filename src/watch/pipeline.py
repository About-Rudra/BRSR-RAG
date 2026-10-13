"""Regulation watch, per document:

    START -> relevance --(irrelevant)--> END
                      \\--(relevant)--> ingest -> summarise -> verify -> queue_for_approval -> END

Scrape-and-diff (registry.py) runs before this graph, in plain code.
"""
import io
import os
from typing import List, TypedDict

import pdfplumber
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

from src.rag import llm as llm_module
from src.agent import prompts
from src.agent.state import event
from src.agent.verifier import verify_claims
from src.agent.researcher import ClaimModel
from src.agent.writer import source_label
from src.scraping.utils import safe_filename
from src.watch import approvals
from src.watch.relevance import check_relevance

SUMMARY_MAX_PAGES = 8
PAGE_CHARS = 2500
SECTIONS = ("what_it_is", "what_changes", "who_is_affected", "key_dates")


class SummaryModel(BaseModel):
    what_it_is: List[ClaimModel] = Field(default_factory=list)
    what_changes: List[ClaimModel] = Field(default_factory=list)
    who_is_affected: List[ClaimModel] = Field(default_factory=list)
    key_dates: List[ClaimModel] = Field(default_factory=list)


class WatchState(TypedDict):
    doc: dict            # url, title, regime, sha256, change ('new'|'changed'), simulated
    pdf_bytes: bytes
    pages: list          # [(page_number, text)]
    relevance: dict
    path: str
    evidence: list
    summary: dict        # section -> [claim]
    verdicts: list
    status: str          # irrelevant | ingested | pending_approval | error
    pending_id: str
    trace: list
    llm_calls: int


def read_pages(pdf_bytes):
    pages = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for n, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((n, text))
    return pages


def relevance_node(state):
    doc = state["doc"]
    pages = read_pages(state["pdf_bytes"])
    first_text = "\n".join(t for _, t in pages)[:2000]
    rel, n = check_relevance(doc["title"], first_text, doc["regime"])
    status = "ingested" if rel["relevant"] else ("error" if rel["method"] == "error" else "irrelevant")
    return {"pages": pages, "relevance": rel, "status": status, "llm_calls": state["llm_calls"] + n,
            "trace": state["trace"] + [event("relevance", "decision", **rel)]}


def route_after_relevance(state):
    return "ingest" if state["relevance"]["relevant"] else END


def ingest_node(state):
    from src.rag.ingestion import ingest_single_file

    doc, regime = state["doc"], state["relevance"]["regime"]
    if regime not in ("sebi_brsr", "cpcb_epr", "ccts"):
        regime = "sebi_brsr" if doc["regime"] == "any" else doc["regime"]
    name = doc.get("filename") or safe_filename(doc["title"]) + ".pdf"
    path = os.path.join("data", "raw", regime, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(state["pdf_bytes"])
    ingest_single_file(path, regime)
    return {"path": path, "doc": {**doc, "regime": regime}, "status": "ingested",
            "trace": state["trace"] + [event("ingest", "ingested", path=path, regime=regime)]}


def summarise_node(state):
    doc = state["doc"]
    source = os.path.basename(state["path"])
    evidence = [{"sid": f"S{i}", "text": text[:PAGE_CHARS], "source": source, "page": page,
                 "regime": doc["regime"], "score": None}
                for i, (page, text) in enumerate(state["pages"][:SUMMARY_MAX_PAGES], start=1)]
    passages = "\n\n".join(f"<<UNTRUSTED_PAGE {p['sid']}>> (page {p['page']})\n{p['text']}\n<<END>>"
                           for p in evidence)
    prompt = prompts.SUMMARISER_PROMPT.format(untrusted_note=prompts.UNTRUSTED_NOTE, title=doc["title"],
                                              regime=doc["regime"], passages=passages)
    trace = list(state["trace"])
    try:
        result = llm_module.get_chat_model().with_structured_output(SummaryModel).invoke(prompt)
        if isinstance(result, dict):
            result = SummaryModel.model_validate(result)
        result = result or SummaryModel()
    except Exception as e:
        trace.append(event("summariser", "error", detail=str(e)[:300]))
        result = SummaryModel()
    summary = {s: [{"text": c.text.strip(), "sids": c.sids} for c in getattr(result, s) if c.text.strip()]
               for s in SECTIONS}
    trace.append(event("summariser", "summary", claims={s: len(v) for s, v in summary.items()},
                       pages_used=len(evidence)))
    return {"evidence": evidence, "summary": summary, "trace": trace, "llm_calls": state["llm_calls"] + 1}


def verify_node(state):
    """The same verifier as the research agent: code sid check, then one LLM call for all claims."""
    flat = [(s, c) for s in SECTIONS for c in state["summary"][s]]
    evidence_by_sid = {p["sid"]: p for p in state["evidence"]}
    verdicts, n, events = verify_claims("summary", [c for _, c in flat], evidence_by_sid)
    for (section, _), v in zip(flat, verdicts):
        v["section"] = section
    return {"verdicts": verdicts, "trace": state["trace"] + events, "llm_calls": state["llm_calls"] + n}


def queue_node(state):
    evidence_by_sid = {p["sid"]: p for p in state["evidence"]}
    doc = state["doc"]
    sections = {s: [] for s in SECTIONS}
    unverified = []
    for v in state["verdicts"]:
        item = {"text": v["claim"], "sources": [source_label(evidence_by_sid[s]) for s in v["sids"]
                                                 if s in evidence_by_sid]}
        (sections[v["section"]] if v["supported"] else unverified).append(
            item if v["supported"] else {**item, "reason": v["reason"]})
    entry = approvals.add_pending({
        "url": doc["url"], "title": doc["title"], "regime": doc["regime"], "change": doc["change"],
        "simulated": doc.get("simulated", False), "path": state["path"],
        "summary": sections, "unverified": unverified,
    })
    return {"status": "pending_approval", "pending_id": entry["id"],
            "trace": state["trace"] + [event("approval", "queued", id=entry["id"],
                                             verified=sum(len(v) for v in sections.values()),
                                             unverified=len(unverified))]}


def build_graph():
    g = StateGraph(WatchState)
    g.add_node("relevance", relevance_node)
    g.add_node("ingest", ingest_node)
    g.add_node("summarise", summarise_node)
    g.add_node("verify", verify_node)
    g.add_node("queue_for_approval", queue_node)
    g.add_edge(START, "relevance")
    g.add_conditional_edges("relevance", route_after_relevance, {"ingest": "ingest", END: END})
    g.add_edge("ingest", "summarise")
    g.add_edge("summarise", "verify")
    g.add_edge("verify", "queue_for_approval")
    g.add_edge("queue_for_approval", END)
    return g.compile()


_graph = None


def process_document(doc, pdf_bytes):
    """Run one new/changed document through the graph. Returns the final state (minus raw bytes)."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    state = _graph.invoke({"doc": doc, "pdf_bytes": pdf_bytes, "pages": [], "relevance": {}, "path": "",
                           "evidence": [], "summary": {}, "verdicts": [], "status": "", "pending_id": "",
                           "trace": [], "llm_calls": 0})
    state.pop("pdf_bytes", None)
    state.pop("pages", None)
    return state
