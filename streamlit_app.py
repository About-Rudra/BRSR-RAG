"""Streamlit UI for the compliance research agent.

Run locally:   streamlit run streamlit_app.py
Hosted:        Streamlit Community Cloud, settings come from the app's Secrets (see docs/deploy_streamlit.md).
"""
import json
import os

import streamlit as st

st.set_page_config(page_title="Compliance Research Agent", page_icon="📑", layout="wide")

# On Streamlit Cloud, settings live in st.secrets. Copy them into the environment
# before importing project modules, which read os.environ.
try:
    for key, value in st.secrets.items():
        if isinstance(value, (str, int, float)):
            os.environ.setdefault(key, str(value))
except Exception:
    pass  # no secrets file: running locally with .env

from src.rag.llm import get_settings  # noqa: E402
from src.rag.vector_client import get_client, COLLECTION  # noqa: E402

REGIME_NAMES = {"sebi_brsr": "SEBI BRSR", "cpcb_epr": "CPCB EPR", "ccts": "BEE CCTS", "any": "Any"}
EXAMPLES = [
    "What is BRSR Core and what does the value chain disclosure require?",
    "Under CCTS, what must an obligated entity do to comply, and how does the offset mechanism work?",
    "What is environmental compensation under the plastic waste rules, and what must listed companies disclose under BRSR Core?",
]


def password_ok():
    """Optional gate so a public link cannot burn through the API quota. Set APP_PASSWORD to enable."""
    expected = os.getenv("APP_PASSWORD")
    if not expected:
        return True
    if st.session_state.get("authed"):
        return True
    pw = st.text_input("Password", type="password")
    if pw and pw == expected:
        st.session_state["authed"] = True
        st.rerun()
    elif pw:
        st.error("Wrong password.")
    return False


@st.cache_data(ttl=600, show_spinner=False)
def indexed_sources():
    from src.agent.tools import list_sources
    return list_sources("any")


def index_size():
    client = get_client()
    if not client.collection_exists(COLLECTION):
        return 0
    return client.count(COLLECTION).count


def show_agent_result(result):
    st.markdown(result["final_answer"])
    s = result["stats"]
    cols = st.columns(6)
    cols[0].metric("Sub-questions", len(result["plan"]))
    cols[1].metric("Tool calls", s["tool_calls"])
    cols[2].metric("LLM calls", s["llm_calls"])
    cols[3].metric("Verified claims", s["supported_claims"])
    cols[4].metric("Rejected claims", s["unsupported_claims"])
    cols[5].metric("Latency", f"{s['latency_s']} s")

    with st.expander("How the agent worked", expanded=False):
        st.markdown("**1. Plan** (planner)")
        st.table([{"id": p["id"], "regime": REGIME_NAMES.get(p["regime"], p["regime"]), "sub-question": p["text"]}
                  for p in result["plan"]])

        st.markdown("**2. Research** (researcher tool calls)")
        for e in result["trace"]:
            d = e["detail"]
            if e["node"] == "researcher" and e["type"] == "start" and d.get("retry"):
                st.markdown(f"↩️ *Retry for {d['subq']} after verifier feedback*")
            elif e["type"] == "tool_call":
                st.markdown(f"- `{d['subq']}` #{d['n']} **{d['tool']}** `{json.dumps(d['input'], ensure_ascii=False)}` "
                            f"→ {d['result']}" + (f" (ids {', '.join(d['sids'])})" if d.get("sids") else ""))
            elif e["type"] == "tool_refused":
                st.markdown(f"- `{d['subq']}` ⛔ {d['tool']} refused: {d['reason']}")

        st.markdown("**3. Verification** (verifier)")
        if result["verdicts"]:
            st.table([{"sub-q": v["subq_id"], "claim": v["claim"], "cites": ", ".join(v["sids"]),
                       "supported": "✅" if v["supported"] else "❌", "checked by": v["checked_by"],
                       "reason": v["reason"]} for v in result["verdicts"]])
        else:
            st.write("No claims were produced.")
        if s["retries"]:
            st.write(f"The verifier sent work back to the researcher {s['retries']} time(s).")

        st.markdown("**4. Evidence** (passages the agent saw)")
        for p in result["evidence"]:
            page = f", p. {p['page']}" if p.get("page") is not None else ""
            st.markdown(f"**[{p['sid']}]** {p['source']}{page} · score {p.get('score')}")
            st.caption(p["text"][:600] + ("…" if len(p["text"]) > 600 else ""))

    st.download_button("Download full trace (JSON)", json.dumps(result, ensure_ascii=False, indent=2),
                       file_name="research_trace.json", mime="application/json")


def ask_tab():
    question = st.text_area("Ask a compliance question", height=90,
                            placeholder="e.g. " + EXAMPLES[2])
    with st.expander("Example questions"):
        for q in EXAMPLES:
            st.code(q, language=None)
    compare = st.checkbox("Also run the baseline (single retrieve-then-generate call) for comparison")

    if st.button("Ask", type="primary", disabled=not question.strip()):
        from src.agent.graph import run_research
        from src.rag.chatbot import chat

        run = {"question": question.strip(), "agent": None, "agent_error": None,
               "baseline": None, "baseline_error": None}
        with st.spinner("Planning, researching and verifying… (this can take a minute or two: "
                        "calls are spaced out to stay under the API rate limit)"):
            try:
                run["agent"] = run_research(run["question"])
            except Exception as e:
                run["agent_error"] = str(e)
            if compare:
                try:
                    run["baseline"] = chat(run["question"])
                except Exception as e:
                    run["baseline_error"] = str(e)
        # Streamlit reruns the script on every click; keep the result on screen.
        st.session_state["last_run"] = run

    run = st.session_state.get("last_run")
    if not run:
        return
    st.markdown(f"**Question:** {run['question']}")
    show_baseline = run["baseline"] is not None or run["baseline_error"]
    agent_col, base_col = st.columns(2) if show_baseline else (st.container(), None)
    with agent_col:
        st.subheader("Research agent")
        if run["agent_error"]:
            st.error(f"Agent failed: {run['agent_error']}")
        else:
            show_agent_result(run["agent"])
    if show_baseline:
        with base_col:
            st.subheader("Baseline (single RAG call)")
            if run["baseline_error"]:
                st.error(f"Baseline failed: {run['baseline_error']}")
            else:
                st.markdown(run["baseline"]["answer"])
                st.caption("Sources: " + (", ".join(run["baseline"]["sources"]) or "none"))


SECTION_TITLES = {"what_it_is": "What it is", "what_changes": "What it changes",
                  "who_is_affected": "Who is affected", "key_dates": "Key dates"}


def show_update(entry):
    for key, title in SECTION_TITLES.items():
        items = entry["summary"].get(key) or []
        if items:
            st.markdown(f"**{title}**")
            for c in items:
                st.markdown(f"- {c['text']} *[{'; '.join(c['sources'])}]*")
    if entry.get("unverified"):
        st.markdown("**Could not verify** (left out of the summary)")
        for c in entry["unverified"]:
            st.markdown(f"- ~~{c['text']}~~ — {c['reason']}")


def show_watch_summary(summary):
    if summary["mode"] == "simulation":
        st.warning("SIMULATION result: a local file was treated as a newly found document. Not a live scrape.")
    for r in summary["processed"]:
        rel = r.get("relevance") or {}
        st.markdown(f"- **{r['title']}** ({r['change']}) → `{r['status']}`"
                    + (f" — relevance by {rel.get('method')}: {rel.get('reason')}" if rel else ""))
    st.caption(f"Processed {len(summary['processed'])}, skipped {len(summary['skipped'])} "
               f"(unchanged or not a PDF), errors {len(summary['errors'])}, {summary['latency_s']} s.")


def watch_tab():
    from src.watch import approvals
    from src.watch.run import run_watch, save_watch_run

    st.markdown("Detects new or changed documents on the SEBI, CPCB and BEE sites, keeps the relevant ones, "
                "indexes them, writes a page-cited summary, verifies it, and waits for **your approval** "
                "before anything enters the updates log.")
    live_col, sim_col = st.columns(2)
    with live_col:
        st.markdown("##### Live check")
        st.caption("Scrapes the three regulator sites now. Government sites can be slow or unreachable.")
        if st.button("Check for updates"):
            with st.spinner("Checking the regulator sites…"):
                summary = run_watch()
                save_watch_run(summary)
            st.session_state["last_watch"] = summary
    with sim_col:
        st.markdown("##### Simulate a new document")
        st.caption("SIMULATION: upload a PDF and run it through the same pipeline as if it had just been "
                   "published. Labelled as a simulation everywhere; it is not a live scrape.")
        upload = st.file_uploader("Regulatory PDF", type=["pdf"])
        title = st.text_input("Document title (optional)")
        if st.button("Run simulation", disabled=upload is None):
            folder = os.path.join("data", "samples")
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, os.path.basename(upload.name))
            with open(path, "wb") as f:
                f.write(upload.getvalue())
            with st.spinner("Relevance → ingest → summarise → verify…"):
                summary = run_watch(simulate_path=path, title=title.strip() or None)
                save_watch_run(summary)
            st.session_state["last_watch"] = summary
    if st.session_state.get("last_watch"):
        show_watch_summary(st.session_state["last_watch"])

    st.divider()
    pending = [e for e in approvals.load_pending() if e["status"] == "pending_approval"]
    st.markdown(f"#### Waiting for approval ({len(pending)})")
    if not pending:
        st.caption("Nothing waiting.")
    for e in pending:
        with st.container(border=True):
            label = "🧪 SIMULATION · " if e.get("simulated") else ""
            st.markdown(f"**{label}{e['title']}**")
            st.caption(f"{REGIME_NAMES.get(e['regime'], e['regime'])} · {e['change']} · "
                       f"found {e['created']} · id {e['id']}")
            show_update(e)
            a, r, _ = st.columns([1, 1, 4])
            if a.button("✅ Approve", key=f"approve_{e['id']}"):
                approvals.approve(e["id"])
                st.rerun()
            if r.button("❌ Reject", key=f"reject_{e['id']}"):
                approvals.reject(e["id"])
                st.rerun()

    log = approvals.load_log()
    st.markdown(f"#### Approved updates log ({len(log)})")
    if not log:
        st.caption("No approved updates yet.")
    for e in reversed(log):
        label = "🧪 SIMULATION · " if e.get("simulated") else ""
        with st.expander(f"{label}{e['title']} — approved {e['decided']}"):
            show_update(e)


def sources_tab():
    sources = indexed_sources()
    if not sources:
        st.info("No documents are indexed yet.")
        return
    for regime in ("sebi_brsr", "cpcb_epr", "ccts"):
        rows = [{"document": s["source"], "chunks": s["chunks"]} for s in sources if s["regime"] == regime]
        st.markdown(f"**{REGIME_NAMES[regime]}** ({len(rows)} documents)")
        st.table(rows)


def main():
    st.title("📑 Compliance Research Agent")
    st.caption("Indian ESG / compliance regulations: SEBI BRSR, CPCB EPR (plastic waste), BEE CCTS. "
               "Answers come only from indexed official documents, and every claim is checked against its source.")

    if not password_ok():
        return

    try:
        n = index_size()
    except Exception as e:
        st.error(f"Cannot reach the vector database: {e}")
        return
    if n == 0:
        mode = os.getenv("QDRANT_MODE", "local")
        st.warning(f"The index is empty (QDRANT_MODE={mode}). Run `python main.py --ingest` (local) or copy the "
                   "index to the cloud collection (see docs/deploy_streamlit.md).")
        if mode != "cloud":
            st.info("On Streamlit Cloud: open the app's Settings → Secrets, paste the secrets "
                    "(including QDRANT_MODE = \"cloud\"), save, then Reboot the app.")
        return

    with st.sidebar:
        provider, model, _ = get_settings()
        st.markdown(f"**LLM:** {provider} / `{model}`")
        st.markdown(f"**Indexed chunks:** {n}")
        st.markdown("**Research:** Planner → Researcher (tools) → Verifier (one retry) → Writer")
        st.markdown("**Watch:** Diff → Relevance → Ingest → Summariser → Verifier → Human approval")

    ask, watch, sources = st.tabs(["Ask", "Regulation watch", "Sources"])
    with ask:
        ask_tab()
    with watch:
        watch_tab()
    with sources:
        sources_tab()


main()
