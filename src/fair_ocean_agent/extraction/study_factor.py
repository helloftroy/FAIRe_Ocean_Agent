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
from fair_ocean_agent.logging_setup import get_logger
from fair_ocean_agent.sources.base import RawFactCandidate

logger = get_logger(__name__)

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

# Same set and same reasoning as extraction/sections.py's own
# _INLINE_TRANSPARENT_TAGS (duplicated rather than cross-imported --
# these two modules have never cross-imported, same precedent as
# section_category_extraction.py/search_flags.py's own independent
# duplicate list-marker helper): a plain flat itertext() join can't tell
# "this inline tag sits mid-word, no space belongs here" (e.g. a bolded
# degenerate base inside a primer sequence, or a structured abstract's
# own emphasis) apart from "this is a genuinely separate block of
# content, a space belongs here" (e.g. a structured abstract's separate
# Background/Results/Conclusions <p> elements sitting immediately
# adjacent with zero whitespace between them in compact XML).
_INLINE_TRANSPARENT_TAGS = frozenset(
    {"bold", "italic", "underline", "sup", "sub", "sc", "monospace", "styled-content", "xref"}
)


def _itertext_with_block_boundaries(element: ET.Element):
    if element.text:
        yield element.text
    for child in element:
        inline = child.tag in _INLINE_TRANSPARENT_TAGS
        if not inline:
            yield " "
        yield from _itertext_with_block_boundaries(child)
        if not inline:
            yield " "
        if child.tail:
            yield child.tail


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
    text = " ".join("".join(_itertext_with_block_boundaries(abstract_el)).split())
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


# generate_json's own retry loop (llm/base.py) only ever retries on
# INVALID JSON -- a response like {"study_factor": ""} is syntactically
# valid, so it's accepted on the very first attempt and generate_json
# never gets a chance to ask again. Per an explicit user request ("make
# that llm call its own if needed, if that helps the llm not miss it"):
# rather than a bigger architectural change (a separate task type), this
# gives the model a couple more chances specifically for the "valid JSON,
# but declined to answer" case, symmetric with generate_json's own
# default of 2 retries for invalid JSON.
#
# Real gap found live: two real studies (10.3389/fmicb.2017.01135,
# 10.1186/s40168-020-00877-y) still came back empty even after this retry
# was added -- confirmed live that abstract extraction itself succeeds
# cleanly for both (_abstract_from_article_text finds a real, substantial
# abstract), so the model itself is declining. The bug: every retry
# attempt reused temperature=0, but a temperature=0 call is deterministic
# -- if the model declines once for a given prompt+input, an identical
# temperature=0 retry asks the exact same question the exact same way and
# gets the exact same empty answer every time, making the "retry" pure
# waste rather than a genuine second chance. Only the FIRST attempt stays
# at temperature=0 (consistent with every other call in this pipeline
# preferring determinism); retries now actually vary the sampling
# temperature so they can land on a different, hopefully non-empty,
# response instead of reproducing the same refusal.
_CONTENT_RETRY_TEMPERATURES: tuple[float, ...] = (0.0, 0.4, 0.7)


def _generate_nonempty_field(
    backend: LLMBackend,
    prompt: str,
    *,
    system: str,
    max_output_tokens: int | None,
    field_name: str,
    error_label: str,
) -> str:
    # Real gap found live: even with the temperature-varying retries below,
    # STUDY-012e2a73836d/STUDY-01f941d6d759 (10.3389/fmicb.2017.01135,
    # 10.1186/s40168-020-00877-y -- the exact two studies that motivated
    # this retry mechanism in the first place) still came back empty. Root
    # cause: this used to `raise` immediately the first time
    # generate_json returned invalid JSON (itself already the result of
    # generate_json's OWN 3 internal sub-attempts, all at that SAME
    # temperature), never even reaching temperature=0.4/0.7 -- the exact
    # "stuck at one temperature" bug this retry loop was built to fix for
    # the valid-but-empty case, left unfixed for the invalid-JSON case.
    # Invalid JSON on one temperature now falls through to the next
    # temperature exactly like an empty value does; only exhausting every
    # temperature still raises.
    saw_invalid_json = False
    for attempt, temperature in enumerate(_CONTENT_RETRY_TEMPERATURES):
        parsed, _response = backend.generate_json(
            prompt, system=system, temperature=temperature, max_tokens=max_output_tokens
        )
        if parsed is None:
            saw_invalid_json = True
            logger.warning(
                "%s: %s attempt %d/%d returned invalid JSON (temperature=%s)",
                backend.label, error_label, attempt + 1, len(_CONTENT_RETRY_TEMPERATURES), temperature,
            )
            continue
        value = str(parsed.get(field_name) or "").strip() if isinstance(parsed, dict) else ""
        if value:
            return value
        logger.warning(
            "%s: %s attempt %d/%d returned an empty value (temperature=%s)",
            backend.label, error_label, attempt + 1, len(_CONTENT_RETRY_TEMPERATURES), temperature,
        )
    if saw_invalid_json:
        raise LLMBackendError(f"{backend.label}: {error_label} generation returned invalid JSON after retries")
    return ""


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
    sentence = _generate_nonempty_field(
        backend,
        prompt,
        system="You summarize a paper's own study design factor(s) from its abstract in one sentence.",
        max_output_tokens=max_output_tokens,
        field_name="study_factor",
        error_label="study_factor",
    )
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
    value = _generate_nonempty_field(
        backend,
        prompt,
        system="You identify a paper's intended target taxonomic scope from its abstract.",
        max_output_tokens=max_output_tokens,
        field_name="study_target_taxonomic_scope",
        error_label="study_target_taxonomic_scope",
    )
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
