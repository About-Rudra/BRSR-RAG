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

    if not st.button("Ask", type="primary", disabled=not question.strip()):
        return

    from src.agent.graph import run_research

    agent_col, base_col = st.columns(2) if compare else (st.container(), None)
    with agent_col:
        st.subheader("Research agent")
        try:
            with st.spinner("Planning, researching and verifying… (on a free-tier API key this "
                            "takes a few minutes, because calls are spaced out to stay under the rate limit)"):
                result = run_research(question.strip())
            show_agent_result(result)
        except Exception as e:
            st.error(f"Agent failed: {e}")

    if compare:
        from src.rag.chatbot import chat
        with base_col:
            st.subheader("Baseline")
            try:
                with st.spinner("Running baseline…"):
                    base = chat(question.strip())
                st.markdown(base["answer"])
                st.caption("Sources: " + (", ".join(base["sources"]) or "none"))
            except Exception as e:
                st.error(f"Baseline failed: {e}")


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
        st.warning("The index is empty. Run `python main.py --ingest` (local) or copy the index to the "
                   "cloud collection (see docs/deploy_streamlit.md).")
        return

    with st.sidebar:
        provider, model, _ = get_settings()
        st.markdown(f"**LLM:** {provider} / `{model}`")
        st.markdown(f"**Indexed chunks:** {n}")
        st.markdown("**Agent:** Planner → Researcher (tools) → Verifier (one retry) → Writer")

    ask, sources = st.tabs(["Ask", "Sources"])
    with ask:
        ask_tab()
    with sources:
        sources_tab()


main()
