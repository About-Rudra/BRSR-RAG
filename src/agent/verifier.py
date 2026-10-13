"""Verifier: checks every claim against the passages it cites.

1. Code check (no LLM): every cited sid must exist in the evidence. A claim with no
   citation, or citing an unknown sid, is unsupported and never sent to the LLM.
2. LLM check: claim + full text of its cited passages -> {supported, reason}.
   To stay within free-tier rate limits, all claims of one sub-question go in a single
   call, but each claim is shown only with its own cited passages and judged separately.
Also used by route_after_verify to decide whether to send sub-questions back once.
"""
from typing import List

from pydantic import BaseModel

from src.rag import llm as llm_module
from src.agent import prompts
from src.agent.state import MAX_RETRIES, event


class ClaimVerdict(BaseModel):
    claim_number: int
    supported: bool
    reason: str


class VerdictListModel(BaseModel):
    verdicts: List[ClaimVerdict]


def _format_cited(passages):
    blocks = []
    for p in passages:
        page = f", page {p['page']}" if p.get("page") is not None else ""
        blocks.append(f"<<UNTRUSTED_PASSAGE {p['sid']}>>\nsource={p['source']}{page}\n{p['text']}\n<<END>>")
    return "\n\n".join(blocks)


def code_check(claim, evidence_by_sid):
    """Returns a reason string if the claim fails the deterministic check, else None."""
    if not claim["sids"]:
        return "claim cites no source"
    missing = [s for s in claim["sids"] if s not in evidence_by_sid]
    if missing:
        return f"cites source id(s) that do not exist: {', '.join(missing)}"
    return None


def verify_claims(subq_id, claims, evidence_by_sid, model=None):
    """Verify a list of claims. Returns (verdicts, llm calls used, trace events)."""
    verdicts, llm_calls = {}, 0
    to_judge = []  # (claim number, claim) pairs that passed the code check
    for i, claim in enumerate(claims):
        base = {"subq_id": subq_id, "claim_index": i, "claim": claim["text"], "sids": claim["sids"]}
        problem = code_check(claim, evidence_by_sid)
        if problem:
            verdicts[i] = {**base, "supported": False, "reason": problem, "checked_by": "code"}
        else:
            to_judge.append((i, claim))

    if to_judge:
        blocks = []
        for i, claim in to_judge:
            blocks.append(f"### Claim {i + 1}\n{claim['text']}\n\nCited passages for claim {i + 1}:\n"
                          + _format_cited([evidence_by_sid[s] for s in claim["sids"]]))
        prompt = prompts.VERIFIER_PROMPT.format(untrusted_note=prompts.UNTRUSTED_NOTE,
                                                claims="\n\n".join(blocks))
        llm_calls += 1
        judged = {}
        try:
            judge = (model or llm_module.get_chat_model()).with_structured_output(VerdictListModel)
            result = judge.invoke(prompt)
            if isinstance(result, dict):
                result = VerdictListModel.model_validate(result)
            judged = {v.claim_number - 1: v for v in (result.verdicts if result else [])}
            error = None
        except Exception as e:
            error = f"verifier error: {e}"[:300]
        for i, claim in to_judge:
            base = {"subq_id": subq_id, "claim_index": i, "claim": claim["text"], "sids": claim["sids"]}
            v = judged.get(i)
            if v is None:
                # Fail closed: a claim we could not check is not treated as verified.
                verdicts[i] = {**base, "supported": False, "checked_by": "llm",
                               "reason": error or "verifier gave no verdict for this claim"}
            else:
                verdicts[i] = {**base, "supported": bool(v.supported), "reason": v.reason,
                               "checked_by": "llm"}

    ordered = [verdicts[i] for i in sorted(verdicts)]
    trace = [event("verifier", "verdict", subq=subq_id, claim=v["claim"][:200], sids=v["sids"],
                   supported=v["supported"], reason=v["reason"], checked_by=v["checked_by"])
             for v in ordered]
    return ordered, llm_calls, trace


def verifier_node(state):
    evidence_by_sid = {p["sid"]: p for p in state["evidence"]}
    # Keep earlier verdicts for sub-questions that were not re-researched.
    verdicts = [v for v in state["verdicts"] if v["subq_id"] not in state["to_research"]]
    trace = list(state["trace"])
    llm_calls = state["llm_calls"]

    for finding in state["findings"]:
        if finding["subq_id"] not in state["to_research"]:
            continue
        new, n, events = verify_claims(finding["subq_id"], finding["claims"], evidence_by_sid)
        verdicts.extend(new)
        llm_calls += n
        trace.extend(events)

    rejected = [v for v in verdicts if not v["supported"]]
    feedback, to_research, retries = {}, [], state["retries"]
    if rejected and retries < MAX_RETRIES:
        for v in rejected:
            feedback.setdefault(v["subq_id"], []).append({"claim": v["claim"], "reason": v["reason"]})
        to_research = list(feedback)
        retries += 1
        trace.append(event("verifier", "retry", subqs=to_research, unsupported=len(rejected)))
    else:
        trace.append(event("verifier", "done", unsupported=len(rejected),
                           supported=len(verdicts) - len(rejected)))

    return {"verdicts": verdicts, "feedback": feedback, "to_research": to_research,
            "retries": retries, "trace": trace, "llm_calls": llm_calls}


def route_after_verify(state):
    """Conditional edge: back to the researcher (once) if claims were rejected, else to the writer."""
    return "researcher" if state["to_research"] else "writer"
