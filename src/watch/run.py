"""Entry points for the regulation watch: a live check of the three sites, or a labelled simulation."""
import json
import os
import time
from datetime import datetime

from src.watch import registry as reg
from src.watch.pipeline import process_document
from src.watch.relevance import keyword_match

# Each run downloads at most this many documents (BEE pages alone list ~470 PDF links).
MAX_DOCS_PER_RUN = int(os.getenv("WATCH_MAX_DOCS", 8))
ACTIVE_STATUSES = ("ingested", "pending_approval", "approved", "rejected")


def list_all_candidates():
    from src.scraping import sebi_scraper, cpcb_scraper, ccts_scraper
    candidates = []
    for scraper in (sebi_scraper, cpcb_scraper, ccts_scraper):
        for c in scraper.list_candidates():
            c["title"] = " ".join(c["title"].split()) or os.path.basename(c["url"])
            candidates.append(c)
    return candidates


def prioritise(candidates, registry):
    """Order: keyword-matching titles first. Skip URLs already judged irrelevant (their content is
    only re-checked when they look relevant again), so a run does not re-download hundreds of files."""
    keep = []
    for c in candidates:
        entry = registry.get(c["url"])
        looks_relevant = keyword_match(c["title"])[1] is not None
        if entry and entry["status"] == "irrelevant" and not looks_relevant:
            continue
        keep.append((0 if looks_relevant else 1, 0 if entry is None else 1, c))
    keep.sort(key=lambda t: (t[0], t[1]))
    return [c for _, _, c in keep]


def _summarise_result(doc, state):
    return {"url": doc["url"], "title": doc["title"], "change": doc["change"],
            "simulated": doc.get("simulated", False), "status": state["status"],
            "relevance": state.get("relevance"), "pending_id": state.get("pending_id"),
            "llm_calls": state.get("llm_calls", 0), "trace": state.get("trace", [])}


def _handle(doc, pdf_bytes, registry, results):
    """Run one new/changed document through the pipeline and record the outcome."""
    state = process_document(doc, pdf_bytes)
    status = state["status"]
    if status != "error":  # an LLM failure leaves the entry unchanged so the next run retries it
        reg.record(registry, doc["url"], doc["title"], state["doc"]["regime"], doc["sha256"], status)
        reg.save_registry(registry)
    results.append(_summarise_result(doc, state))


def run_watch(simulate_path=None, max_docs=MAX_DOCS_PER_RUN, title=None):
    """Live check (default) or simulation of one local PDF. Returns a run summary with the trace."""
    from src.scraping.utils import fetch_pdf_bytes

    start = time.perf_counter()
    registry = reg.load_registry()
    results, skipped, errors = [], [], []
    mode = "simulation" if simulate_path else "live"

    if simulate_path:
        # SIMULATION: treat a local PDF as if the scrapers had just found it.
        with open(simulate_path, "rb") as f:
            data = f.read()
        name = os.path.basename(simulate_path)
        doc = {"url": f"simulated://{name}", "title": title or os.path.splitext(name)[0].replace("_", " "),
               "regime": "any", "simulated": True, "filename": name, "sha256": reg.sha256(data)}
        doc["change"] = reg.classify(registry, doc["url"], doc["sha256"])
        if doc["change"] == "unchanged":
            skipped.append({"url": doc["url"], "reason": "unchanged (same hash already processed)"})
        else:
            _handle(doc, data, registry, results)
    else:
        candidates = prioritise(list_all_candidates(), registry)
        for c in candidates[:max_docs]:
            try:
                data = fetch_pdf_bytes(c["url"])
            except Exception as e:
                errors.append({"url": c["url"], "error": str(e)[:200]})
                continue
            if data is None:
                skipped.append({"url": c["url"], "reason": "not a PDF"})
                continue
            digest = reg.sha256(data)
            change = reg.classify(registry, c["url"], digest)
            if change == "unchanged":
                skipped.append({"url": c["url"], "reason": "unchanged"})
                continue
            entry = registry.get(c["url"])
            if change == "changed" and entry and entry["status"] in ACTIVE_STATUSES:
                print(f"Changed: {c['title']}")
            _handle({**c, "sha256": digest, "change": change}, data, registry, results)

    return {"mode": mode, "ts": datetime.now().isoformat(timespec="seconds"),
            "processed": results, "skipped": skipped, "errors": errors,
            "latency_s": round(time.perf_counter() - start, 2)}


def save_watch_run(summary, folder="runs"):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"watch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return path
