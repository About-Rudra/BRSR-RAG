"""The agent's three tools. All are read-only: they never write to Qdrant or to files.

Each tool returns plain data. The researcher assigns citation ids (S1, S2, ...) and turns
results into text for the LLM with format_* below, inside <<UNTRUSTED_...>> blocks.
"""
import io
import re
from collections import Counter
from urllib.parse import urljoin, urlparse, parse_qs

import requests
from bs4 import BeautifulSoup

from src.rag.retrieval import retrieve
from src.rag.vector_client import get_client, COLLECTION

# The plan's allowlist, plus cpcb.gov.in: cpcb.nic.in now redirects there.
ALLOWED_DOMAINS = ("sebi.gov.in", "cpcb.nic.in", "cpcb.gov.in", "beeindia.gov.in")
FETCH_TIMEOUT = 20
FETCH_MAX_CHARS = 4000
MAX_REDIRECTS = 3
VALID_REGIMES = ("sebi_brsr", "cpcb_epr", "ccts", "any")


def _regime_filter(regime):
    return None if regime in (None, "", "any") else regime


def search_regulations(query: str, regime: str = "any", top_k: int = 5) -> list:
    """Semantic search over the indexed regulation PDFs.

    Args:
        query: what to look for, in regulatory wording.
        regime: one of sebi_brsr, cpcb_epr, ccts, or any.
        top_k: number of passages to return (1-10).
    """
    if regime not in VALID_REGIMES:
        regime = "any"
    top_k = max(1, min(int(top_k), 10))
    hits = retrieve(query, top_k=top_k, score_threshold=None, category=_regime_filter(regime))
    return [{"text": h["text"], "source": h["source"], "page": h.get("page"),
             "regime": h.get("category"), "score": round(h["score"], 4)} for h in hits]


def list_sources(regime: str = "any") -> list:
    """List the documents that are indexed, with how many chunks each has.

    Args:
        regime: one of sebi_brsr, cpcb_epr, ccts, or any.
    """
    client = get_client()
    if not client.collection_exists(COLLECTION):
        return []
    wanted = _regime_filter(regime)
    counts = Counter()
    offset = None
    while True:
        points, offset = client.scroll(collection_name=COLLECTION, limit=256, offset=offset,
                                       with_payload=["source", "category"], with_vectors=False)
        for p in points:
            payload = p.payload or {}
            if wanted and payload.get("category") != wanted:
                continue
            counts[(payload.get("source"), payload.get("category"))] += 1
        if offset is None:
            break
    return [{"source": s, "regime": r, "chunks": n} for (s, r), n in sorted(counts.items(), key=str)]


def is_allowed_url(url):
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in ("http", "https") or not host:
        return False
    return any(host == d or host.endswith("." + d) for d in ALLOWED_DOMAINS)


def _embedded_pdf_url(html, base_url):
    """SEBI circular pages show the PDF in <iframe src="../web/?file=<pdf url>">."""
    for iframe in BeautifulSoup(html, "html.parser").find_all("iframe", src=True):
        file_param = parse_qs(urlparse(iframe["src"]).query).get("file")
        if file_param:
            return urljoin(base_url, file_param[0])
    return None


def _page_text(response):
    content_type = response.headers.get("Content-Type", "").lower()
    if "pdf" in content_type or response.content[:5] == b"%PDF-":
        import pdfplumber
        with pdfplumber.open(io.BytesIO(response.content)) as pdf:
            return "\n".join((page.extract_text() or "") for page in pdf.pages[:5])
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer"]):
        tag.decompose()
    return soup.get_text(" ")


def fetch_official_page(url: str) -> dict:
    """Fetch the text of an official regulator page (sebi.gov.in, cpcb.nic.in, beeindia.gov.in only).

    Args:
        url: full https URL on an allowed domain.
    """
    if not is_allowed_url(url):
        return {"ok": False, "url": url,
                "error": f"Refused: only these domains are allowed: {', '.join(ALLOWED_DOMAINS)}."}
    try:
        current = url
        # Follow redirects by hand so a redirect can never leave the allowlist.
        for _ in range(MAX_REDIRECTS + 1):
            response = requests.get(current, timeout=FETCH_TIMEOUT, allow_redirects=False,
                                    headers={"User-Agent": "Mozilla/5.0 (compliance-research-agent)"})
            if response.is_redirect:
                nxt = urljoin(current, response.headers.get("Location", ""))
                if not is_allowed_url(nxt):
                    return {"ok": False, "url": url, "error": f"Refused redirect to {nxt}."}
                current = nxt
                continue
            response.raise_for_status()
            is_html = "html" in response.headers.get("Content-Type", "").lower()
            pdf_url = _embedded_pdf_url(response.text, current) if is_html else None
            if pdf_url and is_allowed_url(pdf_url) and pdf_url != current:
                current = pdf_url  # read the embedded PDF instead of the viewer page
                continue
            text = re.sub(r"\s+", " ", _page_text(response).replace("\ufeff", "")).strip()
            return {"ok": True, "url": current, "text": text[:FETCH_MAX_CHARS]}
        return {"ok": False, "url": url, "error": "Too many redirects."}
    except Exception as e:
        return {"ok": False, "url": url, "error": f"Fetch failed: {e}"}


# ---------- formatting tool results for the LLM ----------

def format_passages(passages_with_sids, hint=""):
    if not passages_with_sids:
        return "<<UNTRUSTED_SEARCH_RESULTS>>\n(no passages found)\n<<END>>"
    blocks = []
    for p in passages_with_sids:
        page = f", page {p['page']}" if p.get("page") is not None else ""
        blocks.append(f"[{p['sid']}] source={p['source']}{page}, regime={p.get('regime')}, "
                      f"score={p.get('score')}\n{p['text']}")
    out = "<<UNTRUSTED_SEARCH_RESULTS>>\n" + "\n\n".join(blocks) + "\n<<END>>"
    return out + ("\n" + hint if hint else "")


def format_sources(sources):
    if not sources:
        return "No documents are indexed for this regime."
    lines = [f"- {s['source']} (regime={s['regime']}, {s['chunks']} chunks)" for s in sources]
    return "Indexed documents:\n" + "\n".join(lines)


def format_page(result, sid=None):
    if not result.get("ok"):
        return f"Error: {result.get('error')}"
    return (f"<<UNTRUSTED_PAGE_CONTENT>>\n[{sid}] source={result['url']}\n{result['text']}\n<<END>>")
