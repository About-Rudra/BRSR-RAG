"""Print the baseline-vs-agent comparison table from a summary.csv you have marked by hand.

    python -m eval.summarise eval/results/<timestamp>/summary.csv
"""
import csv
import sys


def is_true(v):
    return str(v).strip().lower() == "true"


def main(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    nic = [r for r in rows if r["type"] == "not_in_corpus"]
    answerable = [r for r in rows if r["type"] != "not_in_corpus"]
    unmarked = sum(1 for r in rows if not r["correct_baseline"].strip() or not r["correct_agent"].strip())

    def count(col, value):
        return sum(1 for r in rows if r[col].strip().lower() == value)

    def avg(col):
        vals = [float(r[col]) for r in rows if r[col] not in ("", "None")]
        return f"{sum(vals) / len(vals):.1f}" if vals else "-"

    table = [
        ("Metric", "Baseline RAG", "Agent"),
        (f"Answers judged correct (out of {n})", count("correct_baseline", "y"), count("correct_agent", "y")),
        ("Partially correct", count("correct_baseline", "partial"), count("correct_agent", "partial")),
        (f"Cited source matches expected document (out of {len(answerable)})",
         sum(is_true(r["baseline_source_match"]) for r in answerable),
         sum(is_true(r["agent_source_match"]) for r in answerable)),
        (f'"Not in corpus" questions correctly refused (out of {len(nic)})',
         sum(is_true(r["baseline_refused"]) for r in nic), sum(is_true(r["agent_refused"]) for r in nic)),
        ("Claims rejected by verifier", "n/a", sum(int(r["agent_unsupported_claims"] or 0) for r in rows)),
        ("Average tool calls per question", "n/a", avg("agent_tool_calls")),
        ("Average latency (s)", avg("baseline_latency_s"), avg("agent_latency_s")),
    ]
    width = max(len(str(r[0])) for r in table)
    for metric, b, a in table:
        print(f"{str(metric):<{width}}  {str(b):>12}  {str(a):>8}")
    if unmarked:
        print(f"\nWarning: {unmarked} row(s) still have no hand-marked correctness.")


if __name__ == "__main__":
    main(sys.argv[1])
