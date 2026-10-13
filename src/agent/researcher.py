"""Researcher node: a tool-calling loop per sub-question that produces cited claims."""
import json
import re
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field, ValidationError

from src.rag import llm as llm_module
from src.agent import prompts, tools
from src.agent.state import MAX_TOOL_CALLS_PER_SUBQ, event

TOOLS = [tools.search_regulations, tools.list_sources, tools.fetch_official_page]
WEAK_SCORE = 0.35          # below this best score, the tool output suggests rewriting the query
MAX_TURNS = MAX_TOOL_CALLS_PER_SUBQ + 2


class ClaimModel(BaseModel):
    text: str
    sids: List[str] = Field(default_factory=list)


class FindingsModel(BaseModel):
    claims: List[ClaimModel] = Field(default_factory=list)
    note: str = ""


class EvidenceStore:
    """Holds every passage the agent has seen and gives each a stable id (S1, S2, ...)."""

    def __init__(self, evidence):
        self.items = list(evidence)
        self._keys = {self._key(p): p["sid"] for p in self.items}

    @staticmethod
    def _key(p):
        return (p["source"], p.get("page"), p["text"][:200])

    def add(self, passage):
        key = self._key(passage)
        if key in self._keys:
            return self._keys[key]
        sid = f"S{len(self.items) + 1}"
        self.items.append({"sid": sid, **passage})
        self._keys[key] = sid
        return sid

    def get(self, sid):
        return next((p for p in self.items if p["sid"] == sid), None)


def _regime_of_url(url):
    if "sebi.gov.in" in url:
        return "sebi_brsr"
    if "cpcb" in url:
        return "cpcb_epr"
    if "beeindia.gov.in" in url:
        return "ccts"
    return None


def run_tool(name, args, store, subq_regime):
    """Execute one tool call. Returns (text for the LLM, sids added, short summary for the trace)."""
    if name == "search_regulations":
        args = {"query": str(args.get("query", "")), "regime": args.get("regime") or subq_regime,
                "top_k": args.get("top_k", 5)}
        hits = tools.search_regulations(**args)
        with_sids = [{**h, "sid": store.add(h)} for h in hits]
        best = max((h["score"] for h in hits), default=0.0)
        hint = prompts.WEAK_RESULTS_HINT.format(best=best) if best < WEAK_SCORE else ""
        summary = f"{len(hits)} passages, best score {best:.2f}"
        return tools.format_passages(with_sids, hint), [p["sid"] for p in with_sids], args, summary

    if name == "list_sources":
        args = {"regime": args.get("regime", "any")}
        sources = tools.list_sources(**args)
        return tools.format_sources(sources), [], args, f"{len(sources)} documents"

    if name == "fetch_official_page":
        args = {"url": str(args.get("url", ""))}
        result = tools.fetch_official_page(**args)
        if not result["ok"]:
            return tools.format_page(result), [], args, result["error"]
        sid = store.add({"text": result["text"], "source": result["url"], "page": None,
                         "regime": _regime_of_url(result["url"]), "score": None})
        return tools.format_page(result, sid), [sid], args, f"{len(result['text'])} chars"

    return f"Error: unknown tool '{name}'.", [], args, "unknown tool"


def parse_findings(text):
    """Pull the JSON findings object out of the model's final message. None if invalid."""
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return None
    try:
        return FindingsModel.model_validate(json.loads(match.group(0)))
    except (json.JSONDecodeError, ValidationError):
        return None


def research_subquestion(subq, store, feedback_items):
    """Research one sub-question. Returns (findings, tool calls used, llm calls, trace events)."""
    model = llm_module.get_chat_model()
    tool_model = model.bind_tools(TOOLS)
    trace, llm_calls, calls, seen_sids = [], 0, 0, []

    feedback = ""
    if feedback_items:
        feedback = prompts.RESEARCHER_FEEDBACK.format(
            items="\n".join(f"- \"{f['claim']}\" -> {f['reason']}" for f in feedback_items))
    system = prompts.RESEARCHER_SYSTEM.format(regime_guide=prompts.REGIME_GUIDE,
                                              max_calls=MAX_TOOL_CALLS_PER_SUBQ,
                                              untrusted_note=prompts.UNTRUSTED_NOTE)
    task = prompts.RESEARCHER_TASK.format(subq_id=subq["id"], regime=subq["regime"],
                                          text=subq["text"], feedback=feedback)
    messages = [SystemMessage(system), HumanMessage(task)]

    response = None
    for _ in range(MAX_TURNS):
        response = tool_model.invoke(messages)
        llm_calls += 1
        messages.append(response)
        if not response.tool_calls:
            break
        refused = 0
        for tc in response.tool_calls:
            # The cap is enforced here in code, not only in the prompt.
            if calls >= MAX_TOOL_CALLS_PER_SUBQ:
                refused += 1
                messages.append(ToolMessage(prompts.RESEARCHER_CAP_REACHED, tool_call_id=tc["id"]))
                trace.append(event("researcher", "tool_refused", subq=subq["id"], tool=tc["name"],
                                   reason="cap reached"))
                continue
            calls += 1
            try:
                content, sids, args, summary = run_tool(tc["name"], tc.get("args") or {}, store,
                                                        subq["regime"])
            except Exception as e:
                content, sids, args, summary = f"Error: {e}", [], tc.get("args"), f"error: {e}"
            seen_sids.extend(sids)
            messages.append(ToolMessage(content, tool_call_id=tc["id"]))
            trace.append(event("researcher", "tool_call", subq=subq["id"], n=calls, tool=tc["name"],
                               input=args, result=summary, sids=sids))
        if refused:
            break  # the model keeps asking for tools after the cap: stop the loop

    findings = None
    if response is not None and not response.tool_calls:
        findings = parse_findings(llm_module.message_text(response))

    if findings is None:
        # The model never gave a usable final JSON (or hit the cap). Ask once more, in a fresh
        # prompt with the passages it gathered, for structured findings.
        passages = [store.get(s) for s in dict.fromkeys(seen_sids)]
        prompt = (system + "\n\n" + task + "\n\nPassages you found:\n" +
                  tools.format_passages([p for p in passages if p]) +
                  "\n\n" + prompts.RESEARCHER_FORMAT_REMINDER)
        try:
            findings = model.with_structured_output(FindingsModel).invoke(prompt)
            llm_calls += 1
        except Exception as e:
            trace.append(event("researcher", "error", subq=subq["id"], detail=f"final answer failed: {e}"))
            findings = None
        if findings is None:
            findings = FindingsModel(claims=[], note="researcher produced no valid answer")

    claims = [{"text": c.text.strip(), "sids": [s.strip() for s in c.sids]}
              for c in findings.claims if c.text.strip()]
    trace.append(event("researcher", "findings", subq=subq["id"], claims=len(claims),
                       tool_calls=calls, note=findings.note))
    return {"subq_id": subq["id"], "claims": claims, "note": findings.note}, calls, llm_calls, trace


def researcher_node(state):
    store = EvidenceStore(state["evidence"])
    findings = {f["subq_id"]: f for f in state["findings"]}
    tool_calls = dict(state["tool_calls"])
    trace = list(state["trace"])
    llm_calls = state["llm_calls"]

    for subq in state["plan"]:
        if subq["id"] not in state["to_research"]:
            continue
        trace.append(event("researcher", "start", subq=subq["id"], text=subq["text"],
                           regime=subq["regime"], retry=bool(state["feedback"].get(subq["id"]))))
        finding, calls, n_llm, events = research_subquestion(
            subq, store, state["feedback"].get(subq["id"], []))
        findings[subq["id"]] = finding
        tool_calls[subq["id"]] = tool_calls.get(subq["id"], 0) + calls
        llm_calls += n_llm
        trace.extend(events)

    ordered = [findings[s["id"]] for s in state["plan"] if s["id"] in findings]
    return {"evidence": store.items, "findings": ordered, "tool_calls": tool_calls,
            "trace": trace, "llm_calls": llm_calls}
