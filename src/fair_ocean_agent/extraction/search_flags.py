"""Deterministic text flags that run before any LLM extraction.

These are not mappings and they do not decide downstream FAIRe
completeness on their own. They are cheap, evidence-bearing signals used
to decide whether a later targeted paper/supplement search is worth asking
about. Structured sources still get first chance to fill exact values.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from fair_ocean_agent.database.enums import EntityLevel, SupportType
from fair_ocean_agent.sources.base import RawFactCandidate


@dataclass(frozen=True)
class TextSearchFlag:
    fact_type_candidate: str
    description: str
    patterns: tuple[re.Pattern[str], ...]


TEXT_SEARCH_FLAGS: tuple[TextSearchFlag, ...] = (
    TextSearchFlag(
        fact_type_candidate="probe_based_qPCR_ddPCR_assay_0_1",
        description="probe-based qPCR/ddPCR assay keyword evidence",
        patterns=(
            re.compile(r"\bTaqMan\b", re.IGNORECASE),
            re.compile(r"\bhydrolysis\s+probe(s)?\b", re.IGNORECASE),
            re.compile(r"\bprobe[-\s]+based\b", re.IGNORECASE),
            re.compile(r"\breporter\s+dye(s)?\b", re.IGNORECASE),
            re.compile(r"\bquencher(s)?\b", re.IGNORECASE),
            re.compile(r"\bFAM\b"),
            re.compile(r"\bHEX\b"),
            re.compile(r"\bVIC\b"),
            re.compile(r"\bBHQ(?:-\d+)?\b", re.IGNORECASE),
            re.compile(r"\bZEN\b"),
            re.compile(r"\bMGB\b"),
        ),
    ),
    TextSearchFlag(
        fact_type_candidate="pcr_0_1",
        description="PCR/amplification keyword evidence",
        patterns=(
            re.compile(r"\bPCR\b", re.IGNORECASE),
            re.compile(r"\bqPCR\b", re.IGNORECASE),
            re.compile(r"\bddPCR\b", re.IGNORECASE),
            re.compile(r"\bamplification\b", re.IGNORECASE),
            re.compile(r"\bpolymerase\s+chain\s+reaction\b", re.IGNORECASE),
        ),
    ),
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _snippets(text: str) -> Iterable[tuple[int, str]]:
    """Yield short candidate evidence snippets with stable local positions."""
    normalized = " ".join(text.split())
    if not normalized:
        return
    for index, sentence in enumerate(_SENTENCE_SPLIT_RE.split(normalized)):
        cleaned = sentence.strip()
        if cleaned:
            yield index, cleaned


def _match_flag(flag: TextSearchFlag, text: str) -> tuple[int, str, tuple[str, ...]] | None:
    for index, snippet in _snippets(text):
        matches = tuple(
            sorted(
                {
                    match.group(0)
                    for pattern in flag.patterns
                    for match in pattern.finditer(snippet)
                },
                key=str.lower,
            )
        )
        if matches:
            return index, snippet, matches
    return None


def detect_text_search_flags(
    texts: Iterable[tuple[str, str]],
    *,
    locator_prefix: str,
) -> list[RawFactCandidate]:
    """Return one deterministic boolean flag per matched flag type.

    `texts` is an iterable of `(title, text)` pairs. The first supporting
    sentence wins so that a paper with many PCR mentions records one clear
    project-level flag instead of dozens of duplicate boolean facts.
    """
    facts: list[RawFactCandidate] = []
    seen: set[str] = set()
    for title, text in texts:
        for flag in TEXT_SEARCH_FLAGS:
            if flag.fact_type_candidate in seen:
                continue
            match = _match_flag(flag, text)
            if match is None:
                continue
            snippet_index, snippet, matches = match
            seen.add(flag.fact_type_candidate)
            facts.append(
                RawFactCandidate(
                    entity_level=EntityLevel.STUDY,
                    fact_type_candidate=flag.fact_type_candidate,
                    raw_field_name=flag.fact_type_candidate,
                    raw_value="true",
                    source_locator=f"{locator_prefix}:{title}:sentence[{snippet_index}]",
                    support_type=SupportType.DETERMINISTICALLY_DERIVED,
                    evidence_quote=snippet,
                    confidence_metadata={
                        "detector": "keyword_text_search_flag",
                        "matched_terms": list(matches),
                        "description": flag.description,
                    },
                )
            )
    return facts
