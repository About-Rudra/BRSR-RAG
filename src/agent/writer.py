"""Writer node: final markdown answer from supported claims only, plus a 'Could not verify' section."""
from src.rag import llm as llm_module
from src.agent import prompts
from src.agent.state import event

NOT_FOUND = "The information was not found in the indexed sources."


def source_label(passage):
    if passage.get("page") is not None:
        return f"{passage['source']}, p. {passage['page']}"
    return passage["source"]


def cite(sids, evidence_by_sid):
    labels = list(dict.fromkeys(source_label(evidence_by_sid[s]) for s in sids if s in evidence_by_sid))
    return "[" + "; ".join(labels) + "]"


def could_not_verify_section(state, supported_by_subq):
    """Built in code (not by the LLM) so it always lists exactly what failed."""
    lines = []
    for v in state["verdicts"]:
        if not v["supported"]:
            lines.append(f"- Claim not supported by its sources: \"{v['claim']}\" ({v['reason']})")
    findings = {f["subq_id"]: f for f in state["findings"]}
    for subq in state["plan"]:
        if not supported_by_subq.get(subq["id"]):
            note = findings.get(subq["id"], {}).get("note") or "no supported evidence found"
            lines.append(f"- {subq['id']}: {subq['text']} — {note}")
    if not lines:
        return ""
    return "\n\n### Could not verify\n" + "\n".join(lines)


def writer_node(state):
    evidence_by_sid = {p["sid"]: p for p in state["evidence"]}
    supported = [v for v in state["verdicts"] if v["supported"]]
    supported_by_subq = {}
    for v in supported:
        supported_by_subq.setdefault(v["subq_id"], []).append(v)

    llm_calls = state["llm_calls"]
    trace = list(state["trace"])
    if not supported:
        body = NOT_FOUND
    else:
        claims_text = "\n".join(f"- ({v['subq_id']}) {v['claim']} {cite(v['sids'], evidence_by_sid)}"
                                for v in supported)
        prompt = prompts.WRITER_PROMPT.format(question=state["question"], claims=claims_text)
        try:
            body = llm_module.message_text(llm_module.get_chat_model().invoke(prompt)).strip()
            llm_calls += 1
        except Exception as e:
            # Fall back to the verified claims as a plain list rather than failing the run.
            body = "Verified findings:\n" + claims_text
            trace.append(event("writer", "error", detail=str(e)[:300]))

    answer = body + could_not_verify_section(state, supported_by_subq)
    trace.append(event("writer", "answer", supported_claims=len(supported), chars=len(answer)))
    return {"final_answer": answer, "trace": trace, "llm_calls": llm_calls}
