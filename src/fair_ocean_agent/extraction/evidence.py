"""Deterministic evidence-quote verification (section 12): an LLM-produced
fact is invalid unless its evidence_quote is found verbatim (after
whitespace normalization) in the source text it was supposedly extracted
from. This is intentionally exact, not fuzzy -- the whole point is to
catch fabricated or paraphrased quotes, and any similarity threshold would
let fabrications through by design.
"""
from __future__ import annotations

import re


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def verify_evidence_quote(quote: str, source_text: str) -> bool:
    """True only if `quote` appears verbatim (whitespace-normalized) inside
    `source_text`. An empty/blank quote is never valid -- section 2:
    "unknown or blank is preferable to guessing", but a fact with no quote
    at all is not "unknown", it's unsupported, and must be rejected rather
    than persisted."""
    if not quote or not quote.strip():
        return False
    return _normalize_whitespace(quote) in _normalize_whitespace(source_text)
