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

import xml.etree.ElementTree as ET

from fair_ocean_agent.database.enums import EntityLevel, SupportType
from fair_ocean_agent.llm.base import LLMBackend, LLMBackendError
from fair_ocean_agent.sources.base import RawFactCandidate


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


def generate_study_factor(
    backend: LLMBackend,
    fulltext_xml: str | None,
    *,
    locator_prefix: str,
    max_output_tokens: int | None = 256,
) -> list[RawFactCandidate]:
    abstract = _abstract_from_jats(fulltext_xml)
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
    abstract = _abstract_from_jats(fulltext_xml)
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


def generate_information_withheld_llm_guess(
    backend: LLMBackend,
    texts: list[tuple[str, str]],
    *,
    locator_prefix: str,
    max_chars: int = 12000,
    max_output_tokens: int | None = 400,
) -> list[RawFactCandidate]:
    """EXPERIMENTAL, NOT a real FAIRe field: per an explicit user request
    ("An alternative is to have the LLM list what might be missing from
    methods. I'm curious how that would do -- can we try using that as an
    output please"), this asks the model to read the paper's methods text
    and guess what a careful reader might expect to see reported but
    doesn't -- e.g. no code repository link, no accession for a described-
    but-undeposited dataset, no explicit control description. This is
    model speculation about apparent gaps, not a quote of a real reported
    statement, so it is deliberately kept OUT of the real informationWithheld
    field (extraction/search_flags.py's own LLMJudgedSearchField handles
    that, verbatim-only, plus mapping/faire.py's own "Nothing indicated as
    withheld" default) -- this is surfaced only as a separate internal
    diagnostic column (exports/faire.py's
    INTERNAL_INFORMATION_WITHHELD_LLM_GUESS_FIELD) so the two can be
    compared side by side, never merged into one FAIRe value."""
    combined = "\n\n".join(text for _title, text in texts if text).strip()
    if not combined:
        return []
    combined = combined[:max_chars]

    prompt = f"""Read the paper methods text below. List anything a careful reader might expect this paper to
report but does NOT -- for example: no code/analysis repository link, no accession number for a dataset that
is described but not deposited anywhere, no explicit description of negative/positive controls, no replicate
count, no reference database version, or similar gaps. This is your own assessment of likely gaps based on
what's normally expected in this kind of methods section, not a quote from the text.

Return a short pipe-delimited list of the specific gaps you notice (e.g. "no code repository provided | no
replicate count reported"). If you don't notice any notable gaps, return an empty string.

Methods text:
{combined}

Return ONLY a JSON object: {{"information_withheld_llm_guess": "<pipe-delimited list of gaps, or empty>"}}
"""
    parsed, _response = backend.generate_json(
        prompt,
        system="You identify likely reporting gaps in a paper's methods, as your own assessment, not a quote.",
        temperature=0,
        max_tokens=max_output_tokens,
    )
    if parsed is None:
        raise LLMBackendError(
            f"{backend.label}: information_withheld_llm_guess generation returned invalid JSON after retries"
        )
    value = str(parsed.get("information_withheld_llm_guess") or "").strip() if isinstance(parsed, dict) else ""
    if not value:
        return []

    return [
        RawFactCandidate(
            entity_level=EntityLevel.STUDY,
            fact_type_candidate="information_withheld_llm_guess",
            raw_field_name="information_withheld_llm_guess",
            raw_value=value,
            source_locator=f"{locator_prefix}:information_withheld_llm_guess:llm_generated_from_methods",
            support_type=SupportType.INFERRED,
            confidence_metadata={"detector": "llm_generated_information_withheld_guess"},
        )
    ]
