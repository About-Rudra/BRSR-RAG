"""Builds the research graph:

    START -> planner -> researcher -> verifier --(unsupported claims, retries < 1)--> researcher
                                                \\--(otherwise)--> writer -> END
"""
import json
import os
import time
from datetime import datetime

from langgraph.graph import StateGraph, START, END

from src.agent.state import ResearchState, initial_state
from src.agent.planner import planner_node
from src.agent.researcher import researcher_node
from src.agent.verifier import verifier_node, route_after_verify
from src.agent.writer import writer_node

_graph = None


def build_graph():
    g = StateGraph(ResearchState)
    g.add_node("planner", planner_node)
    g.add_node("researcher", researcher_node)
    g.add_node("verifier", verifier_node)
    g.add_node("writer", writer_node)
    g.add_edge(START, "planner")
    g.add_edge("planner", "researcher")
    g.add_edge("researcher", "verifier")
    g.add_conditional_edges("verifier", route_after_verify,
                            {"researcher": "researcher", "writer": "writer"})
    g.add_edge("writer", END)
    return g.compile()


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_research(question):
    """Run the full workflow. Returns the final answer plus everything needed to inspect it."""
    start = time.perf_counter()
    state = get_graph().invoke(initial_state(question))
    latency = time.perf_counter() - start
    stats = {
        "tool_calls": sum(state["tool_calls"].values()),
        "tool_calls_per_subq": state["tool_calls"],
        "llm_calls": state["llm_calls"],
        "latency_s": round(latency, 2),
        "retries": state["retries"],
        "supported_claims": sum(1 for v in state["verdicts"] if v["supported"]),
        "unsupported_claims": sum(1 for v in state["verdicts"] if not v["supported"]),
    }
    return {
        "question": question,
        "final_answer": state["final_answer"],
        "plan": state["plan"],
        "evidence": state["evidence"],
        "findings": state["findings"],
        "verdicts": state["verdicts"],
        "trace": state["trace"],
        "stats": stats,
    }


def save_run(result, folder="runs"):
    """Write the full result (incl. trace) to runs/research_<timestamp>.json. Returns the path."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return path
