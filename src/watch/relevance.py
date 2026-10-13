"""Relevance filter: a cheap keyword check first, then the LLM only for documents keywords cannot place."""
import re
from typing import Literal

from pydantic import BaseModel

from src.rag import llm as llm_module
from src.agent import prompts
from src.scraping.utils import config

# Keyword -> regime, used to tag obvious matches without an LLM call.
REGIME_KEYWORDS = {
    "sebi_brsr": ["brsr", "business responsibility", "sustainability report", "esg", "value chain", "assurance"],
    "cpcb_epr": ["extended producer responsibility", "epr", "plastic waste", "plastic packaging", "battery waste"],
    "ccts": ["carbon credit", "ccts", "carbon market", "offset mechanism", "emission intensity", "greenhouse gas"],
}
RELEVANCE_TEXT_CHARS = 1500


class RelevanceModel(BaseModel):
    relevant: bool
    regime: Literal["sebi_brsr", "cpcb_epr", "ccts", "any"]
    reason: str


def keyword_match(title):
    """Returns (regime, keyword) for an obvious match on the title, else (None, None)."""
    def found(word):
        # whole words only, so 'epr' does not match inside 'deprecated'
        return re.search(rf"\b{re.escape(word.lower())}\b", title.lower()) is not None

    for regime, words in REGIME_KEYWORDS.items():
        for w in words:
            if found(w):
                return regime, w
    for w in config.get("relevance_keywords", []):
        if found(w):
            return None, w
    return None, None


def check_relevance(title, text, default_regime):
    """Returns ({relevant, regime, reason, method}, llm calls used)."""
    regime, keyword = keyword_match(title)
    if keyword:
        return {"relevant": True, "regime": regime or default_regime,
                "reason": f"title matches keyword '{keyword}'", "method": "keyword"}, 0

    prompt = prompts.RELEVANCE_PROMPT.format(regime_guide=prompts.REGIME_GUIDE,
                                             untrusted_note=prompts.UNTRUSTED_NOTE,
                                             title=title or "(no title)",
                                             text=(text or "")[:RELEVANCE_TEXT_CHARS])
    try:
        result = llm_module.get_chat_model().with_structured_output(RelevanceModel).invoke(prompt)
        if isinstance(result, dict):
            result = RelevanceModel.model_validate(result)
        regime = result.regime if result.regime != "any" else default_regime
        return {"relevant": bool(result.relevant), "regime": regime, "reason": result.reason,
                "method": "llm"}, 1
    except Exception as e:
        # If the LLM is unavailable we cannot judge: skip for now; it will be re-checked next run.
        return {"relevant": False, "regime": default_regime, "reason": f"relevance check failed: {e}"[:300],
                "method": "error"}, 1
