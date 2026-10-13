"""Typed state shared by every node of the research graph."""
from datetime import datetime, timezone
from typing import List, Literal, Optional, TypedDict

REGIMES = ("sebi_brsr", "cpcb_epr", "ccts")
Regime = Literal["sebi_brsr", "cpcb_epr", "ccts", "any"]

MAX_SUBQUESTIONS = 4
MAX_TOOL_CALLS_PER_SUBQ = 6
MAX_RETRIES = 1


class SubQuestion(TypedDict):
    id: str              # "Q1", "Q2", ...
    text: str
    regime: str          # one of REGIMES or "any"


class Passage(TypedDict):
    sid: str             # stable label "S1", "S2", ... used in citations
    text: str
    source: str          # document file name, or URL for fetched pages
    page: Optional[int]
    regime: Optional[str]
    score: Optional[float]


class Claim(TypedDict):
    text: str
    sids: List[str]


class Finding(TypedDict):
    subq_id: str
    claims: List[Claim]
    note: str            # e.g. "nothing relevant found"


class Verdict(TypedDict):
    subq_id: str
    claim_index: int
    claim: str
    sids: List[str]
    supported: bool
    reason: str
    checked_by: str      # "code" (sid check) or "llm"


class TraceEvent(TypedDict):
    node: str
    type: str
    detail: dict
    ts: str


class ResearchState(TypedDict):
    question: str
    plan: List[SubQuestion]
    evidence: List[Passage]
    findings: List[Finding]
    verdicts: List[Verdict]
    retries: int
    tool_calls: dict             # subq_id -> number of tool calls (cumulative)
    trace: List[TraceEvent]
    final_answer: str
    # bookkeeping (not in the plan's table, needed for routing and stats)
    to_research: List[str]       # sub-question ids the researcher should (re)work
    feedback: dict               # subq_id -> verifier reasons, for the retry
    llm_calls: int


def event(node, type_, **detail):
    """Build one trace event."""
    return {"node": node, "type": type_, "detail": detail,
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds")}


def initial_state(question):
    return {
        "question": question,
        "plan": [],
        "evidence": [],
        "findings": [],
        "verdicts": [],
        "retries": 0,
        "tool_calls": {},
        "trace": [],
        "final_answer": "",
        "to_research": [],
        "feedback": {},
        "llm_calls": 0,
    }
