"""LLM-GENERATED (not extracted) `study_factor`: deliberately the one
field in this whole pipeline where the model is asked to summarize rather
than quote or classify verbatim -- an explicit, narrowly-scoped exception
to the "never generate, only extract/quote" discipline used everywhere
else (extraction/search_flags.py's quote-candidate-then-judge mechanism,
extraction/section_category_extraction.py's Stage 3, ...), per an
explicit user instruction: "I want the LLM to read the abstract, then
generate a sentence about what this study is testing."

Deliberately narrow in scope to keep the generation task well-bounded:
reads ONLY the paper's own abstract (never the full paper, never a
supplement), produces exactly one field, one sentence, one LLM call.
`study_factor`'s own real FAIRe definition ("the variable(s) examined in
the study, including those of direct interest to address study aims and
covariates") is rarely stated as a single quotable sentence in a real
paper, unlike almost everything else this pipeline extracts -- the
concept genuinely has to be synthesized, not found.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from fair_ocean_agent.database.enums import EntityLevel, SupportType
from fair_ocean_agent.llm.base import LLMBackend, LLMBackendError
from fair_ocean_agent.sources.base import RawFactCandidate

# Real gap found live: PDF-to-text extraction (this function's own
# fallback for a paper with no JATS fulltext XML) routinely jams the
# heading and the abstract's own first sentence onto one line with no
# line break between them (e.g. "Abstract Background: high-latitude
# coral habitats..."), especially for two-column PDF layouts -- the
# original regex required "Abstract"/"Summary" to be the ENTIRE line
# (nothing else, `\s*$`), so any such run-together heading silently
# failed to match at all, and generate_study_factor's own `if not
# abstract: return []` meant the whole field was skipped with no error,
# no review flag, nothing. Now only anchors the heading WORD at the
# start of a line, optionally followed by ":"/"." -- the match still
# ends right after the heading itself, so real abstract content on the
# same line is correctly kept, not consumed by the heading match.
_ABSTRACT_HEADING_RE = re.compile(r"(?im)^\s*(?:abstract|summary)\b[.:]?\s*")
_ABSTRACT_END_HEADING_RE = re.compile(
    r"(?im)^\s*(?:keywords?|introduction|background|materials?\s+and\s+methods?|methods?|results?)\b"
)


def _abstract_from_jats(fulltext_xml: str | None) -> str | None:
    if not fulltext_xml:
        return None
    try:
        root = ET.fromstring(fulltext_xml)
    except ET.ParseError:
        return None
    abstract_el = root.find(".//abstract")
    if abstract_el is None:
        return None
    text = " ".join(" ".join(abstract_el.itertext()).split())
    return text or None


def _abstract_from_plain_text(text: str | None, *, max_chars: int = 4000) -> str | None:
    """Best-effort abstract extraction from local PDF text.

    PDF text has no durable article tree, but most journal PDFs expose an
    "Abstract" or "Summary" heading. Keep the extraction bounded to the
    text before the next major heading so abstract-only LLM fields do not
    accidentally read the whole article.
    """
    if not text:
        return None
    match = _ABSTRACT_HEADING_RE.search(text)
    if not match:
        return None
    start = match.end()
    end_match = _ABSTRACT_END_HEADING_RE.search(text, start)
    end = end_match.start() if end_match else min(len(text), start + max_chars)
    abstract = " ".join(text[start:end].split())
    if len(abstract) > max_chars:
        abstract = abstract[:max_chars].rsplit(" ", 1)[0]
    return abstract or None


def _abstract_from_article_text(article_text: str | None) -> str | None:
    return _abstract_from_jats(article_text) or _abstract_from_plain_text(article_text)


def generate_study_factor(
    backend: LLMBackend,
    fulltext_xml: str | None,
    *,
    locator_prefix: str,
    max_output_tokens: int | None = 256,
) -> list[RawFactCandidate]:
    abstract = _abstract_from_article_text(fulltext_xml)
    if not abstract:
        return []

    prompt = f"""Read the paper abstract below and write ONE concise sentence describing the variable(s) this
study examines -- the study's own factor(s) of interest (e.g. treatment, site, time period, habitat type, or
other comparison the study was designed to test), including relevant covariates.

Write your own summary sentence in your own words; do not copy a sentence verbatim from the abstract. Do not
include citations, unrelated background information, or filler phrases like "This study investigates" or "The
authors examined". Return ONLY the summary sentence, nothing else.

Abstract:
{abstract}

Return ONLY a JSON object: {{"study_factor": "<your one-sentence summary>"}}
"""
    parsed, _response = backend.generate_json(
        prompt,
        system="You summarize a paper's own study design factor(s) from its abstract in one sentence.",
        temperature=0,
        max_tokens=max_output_tokens,
    )
    if parsed is None:
        raise LLMBackendError(f"{backend.label}: study_factor generation returned invalid JSON after retries")
    sentence = str(parsed.get("study_factor") or "").strip() if isinstance(parsed, dict) else ""
    if not sentence:
        return []

    return [
        RawFactCandidate(
            entity_level=EntityLevel.STUDY,
            fact_type_candidate="study_factor",
            raw_field_name="study_factor",
            raw_value=sentence,
            source_locator=f"{locator_prefix}:study_factor:llm_generated_from_abstract",
            # INFERRED, not EXPLICIT: this value is synthesized, not a
            # direct quote -- the one place in this pipeline where that
            # distinction is the honest one to make.
            support_type=SupportType.INFERRED,
            evidence_quote=abstract,
            confidence_metadata={"detector": "llm_generated_study_factor"},
        )
    ]


def generate_study_target_taxonomic_scope(
    backend: LLMBackend,
    fulltext_xml: str | None,
    *,
    locator_prefix: str,
    max_output_tokens: int | None = 256,
) -> list[RawFactCandidate]:
    abstract = _abstract_from_article_text(fulltext_xml)
    if not abstract:
        return []

    prompt = f"""Read the paper abstract below.

Extract the organisms or broad biological/taxonomic group the study intends to investigate. The scope may be broad,
such as microorganisms, prokaryotes, bacteria, archaea, fungi, eukaryotes, fish, plankton, or microbial communities.
Do not require a named species, genus, or formal taxonomic rank, though that is best.

Return only the actual studied organism/group names, not a sentence. Use the paper's own terms when possible. If
multiple scopes are supported, join them with " | ". Do not include locations, habitats, environmental variables,
sequencing methods, or generic phrases like "this study".

Examples:
- "AOA's distribution was explored, and we investigated NOB in oxic marine sediments" -> "AOA | NOB"
- "Nitrosediminicola species were investigated in this study" -> "Nitrosediminicola species"
- "prokaryotic diversity including bacteria and archaea" -> "prokaryotic microorganisms | bacteria | archaea"

Abstract:
{abstract}

Return ONLY a JSON object: {{"study_target_taxonomic_scope": "<pipe-delimited scope values>"}}
"""
    parsed, _response = backend.generate_json(
        prompt,
        system="You identify a paper's intended target taxonomic scope from its abstract.",
        temperature=0,
        max_tokens=max_output_tokens,
    )
    if parsed is None:
        raise LLMBackendError(f"{backend.label}: study_target_taxonomic_scope generation returned invalid JSON after retries")
    value = str(parsed.get("study_target_taxonomic_scope") or "").strip() if isinstance(parsed, dict) else ""
    if not value:
        return []

    return [
        RawFactCandidate(
            entity_level=EntityLevel.STUDY,
            fact_type_candidate="study_target_taxonomic_scope",
            raw_field_name="study_target_taxonomic_scope",
            raw_value=value,
            source_locator=f"{locator_prefix}:study_target_taxonomic_scope:llm_generated_from_abstract",
            support_type=SupportType.INFERRED,
            evidence_quote=abstract,
            confidence_metadata={"detector": "llm_generated_study_target_taxonomic_scope"},
        )
    ]
