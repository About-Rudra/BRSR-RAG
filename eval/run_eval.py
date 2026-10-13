"""Run every question in eval/questions.jsonl through the baseline chat() and the agent run_research(),
with the same LLM provider and model, and save the raw outputs plus a summary CSV.

    python -m eval.run_eval                 # all questions
    python -m eval.run_eval --limit 3       # first 3
    python -m eval.run_eval --llm-judge     # add an automatic LLM-graded column (secondary)

Output: eval/results/<timestamp>/outputs.jsonl and summary.csv. Correctness is judged by hand:
fill the empty correct_baseline / correct_agent columns with y, n or partial, then run eval/summarise.py.
"""
import argparse
import csv
import json
import os
import re
import time
from datetime import datetime

from src.agent.graph import run_research
from src.agent.writer import NOT_FOUND
from src.rag.chatbot import chat
from src.rag.llm import get_settings, get_chat_model, message_text

QUESTIONS_PATH = os.path.join("eval", "questions.jsonl")
# Phrases the baseline uses when it declines to answer.
BASELINE_REFUSAL = re.compile(r"not (found|present|available|mentioned|provided|contained)|"
                              r"does not (contain|mention|provide)|no relevant information|"
                              r"(isn't|is not) in the (provided )?context|cannot (answer|find)", re.I)


def load_questions(path=QUESTIONS_PATH):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["question"].startswith("<"):
                continue  # placeholder row from the template
            rows.append(row)
    return rows


def source_matches(expected, sources):
    expected = (expected or "").lower().replace(".pdf", "")
    return bool(expected) and any(expected in (s or "").lower() for s in sources)


def run_baseline(q):
    start = time.perf_counter()
    out = chat(q["question"])
    latency = time.perf_counter() - start
    sources = out["sources"]
    return {
        "answer": out["answer"], "sources": sources, "latency_s": round(latency, 2),
        "llm_calls": 1 if sources else 0,
        "has_citation": bool(sources),
        "source_match": source_matches(q.get("expected_source"), sources),
        "refused": bool(BASELINE_REFUSAL.search(out["answer"])) or not sources,
    }


def run_agent(q):
    r = run_research(q["question"])
    evidence = {p["sid"]: p for p in r["evidence"]}
    cited = [evidence[s] for v in r["verdicts"] if v["supported"] for s in v["sids"] if s in evidence]
    sources = sorted({p["source"] for p in cited})
    expected_page = q.get("expected_page")
    page_match = any(source_matches(q.get("expected_source"), [p["source"]]) and p.get("page") == expected_page
                     for p in cited) if expected_page else None
    return {
        "answer": r["final_answer"], "sources": sources, "latency_s": r["stats"]["latency_s"],
        "llm_calls": r["stats"]["llm_calls"], "tool_calls": r["stats"]["tool_calls"],
        "retries": r["stats"]["retries"],
        "supported_claims": r["stats"]["supported_claims"],
        "unsupported_claims": r["stats"]["unsupported_claims"],
        "has_citation": bool(cited),
        "source_match": source_matches(q.get("expected_source"), sources),
        "page_match": page_match,
        "refused": r["final_answer"].startswith(NOT_FOUND),
        "plan": r["plan"], "verdicts": r["verdicts"], "trace": r["trace"],
    }


JUDGE_PROMPT = """Compare a system's answer with the expected answer to a compliance question.
Question: {question}
Expected answer: {expected}
System answer: {answer}
Reply with exactly one word: y (correct), partial, or n (wrong or missing)."""


def llm_judge(q, answer):
    reply = message_text(get_chat_model().invoke(JUDGE_PROMPT.format(
        question=q["question"], expected=q.get("expected_answer", ""), answer=answer))).strip().lower()
    for label in ("partial", "y", "n"):
        if reply.startswith(label):
            return label
    return "?"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--llm-judge", action="store_true", help="add an automatic, secondary LLM-graded column")
    parser.add_argument("--questions", default=QUESTIONS_PATH)
    args = parser.parse_args()

    questions = load_questions(args.questions)[:args.limit]
    if not questions:
        raise SystemExit(f"No questions in {args.questions} (placeholder rows starting with '<' are skipped).")

    provider, model, _ = get_settings()
    folder = os.path.join("eval", "results", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(folder, exist_ok=True)
    print(f"Running {len(questions)} questions with {provider}/{model} -> {folder}")

    rows = []
    with open(os.path.join(folder, "outputs.jsonl"), "w", encoding="utf-8") as out:
        for i, q in enumerate(questions, 1):
            print(f"[{i}/{len(questions)}] {q['id']}: {q['question'][:70]}")
            base = run_baseline(q)
            agent = run_agent(q)
            record = {"question": q, "provider": provider, "model": model, "baseline": base, "agent": agent}
            if args.llm_judge:
                record["llm_judge_auto"] = {"baseline": llm_judge(q, base["answer"]),
                                            "agent": llm_judge(q, agent["answer"])}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            row = {
                "id": q["id"], "type": q.get("type"), "question": q["question"],
                "expected_source": q.get("expected_source"), "expected_page": q.get("expected_page"),
                "baseline_has_citation": base["has_citation"], "baseline_source_match": base["source_match"],
                "baseline_refused": base["refused"], "baseline_latency_s": base["latency_s"],
                "agent_has_citation": agent["has_citation"], "agent_source_match": agent["source_match"],
                "agent_page_match": agent["page_match"], "agent_refused": agent["refused"],
                "agent_supported_claims": agent["supported_claims"],
                "agent_unsupported_claims": agent["unsupported_claims"],
                "agent_tool_calls": agent["tool_calls"], "agent_llm_calls": agent["llm_calls"],
                "agent_latency_s": agent["latency_s"],
                "correct_baseline": "", "correct_agent": "",  # filled in by hand: y / n / partial
            }
            if args.llm_judge:
                row["llm_judge_auto_baseline"] = record["llm_judge_auto"]["baseline"]
                row["llm_judge_auto_agent"] = record["llm_judge_auto"]["agent"]
            rows.append(row)

    with open(os.path.join(folder, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Done. Fill correct_baseline / correct_agent in {folder}/summary.csv, "
          f"then run: python -m eval.summarise {folder}/summary.csv")


if __name__ == "__main__":
    main()
