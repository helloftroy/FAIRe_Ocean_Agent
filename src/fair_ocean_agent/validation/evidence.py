"""Structural evidence-consistency checks (section 16) over already-
persisted RawFacts.

Section 16's "evidence quote exists in source text" is only checkable at
extraction time -- extraction/evidence.py's verify_evidence_quote already
does exactly that, before a fact is ever persisted (extraction/text.py
drops anything that fails). Source text itself isn't retained after
extraction (section 1: the source document may remain ephemeral), so a
RawFact already in the database can't be re-checked against its original
text. What *can* be re-checked, as an audit: does every fact's evidence
bookkeeping match what its support_type promises -- an EXPLICIT fact
should carry a non-blank evidence_quote, a STRUCTURED_SOURCE fact should
carry a non-blank source_locator. A failure here means a bug in an
adapter/extractor, not bad input data.
"""
from __future__ import annotations

from dataclasses import dataclass

from fair_ocean_agent.database.enums import SupportType


@dataclass
class EvidenceCheckResult:
    ok: bool
    message: str


def check_raw_fact_evidence_consistency(
    support_type: str, evidence_quote: str | None, source_locator: str | None
) -> EvidenceCheckResult:
    if support_type == SupportType.EXPLICIT.value:
        if not evidence_quote or not evidence_quote.strip():
            return EvidenceCheckResult(False, "support_type=explicit but evidence_quote is missing/blank")
        return EvidenceCheckResult(True, "explicit fact has a non-blank evidence_quote")

    if support_type == SupportType.STRUCTURED_SOURCE.value:
        if not source_locator or not source_locator.strip():
            return EvidenceCheckResult(False, "support_type=structured_source but source_locator is missing/blank")
        return EvidenceCheckResult(True, "structured_source fact has a non-blank source_locator")

    return EvidenceCheckResult(True, f"no evidence requirement defined for support_type={support_type!r}")
