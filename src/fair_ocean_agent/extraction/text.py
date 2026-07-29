"""LLM-based text fact extraction (section 12), made FAIRe-aware in v2 and
corrected in v3: every candidate fact the model returns must carry an
evidence_quote that extraction.evidence confirms is present verbatim in the
exact section text it was extracted from -- candidates that fail
verification are dropped here, before they ever reach persistence, never
persisted as a fact and never surfaced to a caller. The model only ever
sees bounded, already-selected text (see extraction/sections.py); this
module never lets it browse further or pull in outside knowledge.

**Why "FAIRe-aware" matters (v1 -> v2):** v1's prompt was fully open-
vocabulary -- "extract whatever facts you find, name them however you
like" -- which never missed an explicitly-stated concept but also never
gave the model any reason to report atomic, structured facts (PCR reaction
volume, primer concentration, annealing temperature, standard-curve
slope, assay name, control types, ...) rather than one coarse blob per
concept ("PCR_amplification_conditions"). mapping/rules.py could only ever
map those blobs to FAIRe's "*_method_additional" free-text fallback
fields, flagged for manual review, never the atomic fields FAIRe actually
wants. v2 handed the model an explicit checklist so it would report atomic
facts instead of blobs.

**Why v2 -> v3:** v2's checklist (extraction/faire_fields.py) used FAIRe's
own slot spellings (annealingTemp, r2, otu_db) directly as
fact_type_candidate -- so a raw fact's own identity was FAIRe's vocabulary,
not a source-native description of it. That's the same coupling this
pipeline deliberately avoids everywhere else: a repository adapter's
fact_type_candidate is never phrased in Darwin Core or MIxS's own spelling
either, and raw_facts standardizes onto FAIRe only as a separate,
downstream step (mapping/rules.py), never at extraction time. v3 keeps the
same atomic checklist (still the FAIRe field list in substance -- PCR
volumes, primer concentrations, standard-curve slope/r2, taxonomy outputs,
etc. are all still there) but fact_type_candidate is now always a plain,
standard-agnostic native_name (annealing_temperature, standard_curve_r_squared,
reference_database). The model may additionally return an OPTIONAL
candidate_standard_fields hint per fact (e.g. {"faire": "annealingTemp"}) --
a suggestion about which standard field a fact might correspond to, stored
in RawFact.confidence_metadata, never folded into fact_type_candidate
itself. Dropping or ignoring every candidate_standard_fields hint would
still leave a fully valid, source-native raw fact -- that's the litmus
test for the hint staying truly optional.

`prompt_version` is deliberately explicit and threaded through to the
caller (for RawFact.prompt_version) rather than inferred from the template
text -- so a future prompt change is a visible version bump, not a silent
behavior change under the same version string.

**Structured-first extraction (v3.1):** before ever asking the LLM about a
study, `resolved_faire_fields_for_study` checks which FAIRe fields already
have a real, present value from a prior `MAP_FAIRE` pass over that study's
*structured* facts (NCBI/ENA/PANGAEA/... adapters, never LLM output --
`workflow/handlers.py` computes this once and passes it down as
`exclude_faire_hints`). Those concepts are dropped from the checklist
entirely: a shorter prompt (less of the context-window ceiling found
during model benchmarking spent on questions that don't need asking),
fewer things the model can hallucinate an answer for, and no risk of a
weaker LLM guess *conflicting* with an already-resolved structured value.
This only ever narrows what's asked, never fabricates or skips a
genuinely-needed fact -- if MAP_FAIRE hasn't run yet for a study (e.g. text
extraction run before structured mapping), `exclude_faire_hints` is simply
empty and every concept is still asked about, exactly as before this
existed.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from fair_ocean_agent.database.enums import EntityLevel, MissingnessStatus, SupportType
from fair_ocean_agent.database.models import StandardizedValue
from fair_ocean_agent.extraction.evidence import verify_evidence_quote
from fair_ocean_agent.extraction.faire_fields import render_field_reference
from fair_ocean_agent.llm.base import LLMBackend, LLMResponse
from fair_ocean_agent.mapping.faire import TARGET_SCHEMA
from fair_ocean_agent.sources.base import RawFactCandidate

PROMPT_VERSION = "text-extraction-v3-native-with-hints"


def resolved_faire_fields_for_study(session: Session, study_id: str) -> frozenset[str]:
    """FAIRe target_field names that already have a real (`missingness_status
    == "present"`), non-LLM-derived value for this study -- i.e. a prior
    MAP_FAIRE pass already resolved them from a structured source. Returns
    an empty set (never raises) if MAP_FAIRE hasn't run for this study yet,
    which is exactly the "ask about everything" behavior this pipeline had
    before this function existed."""
    rows = session.execute(
        select(StandardizedValue.target_field)
        .where(
            StandardizedValue.study_id == study_id,
            StandardizedValue.target_schema == TARGET_SCHEMA,
            StandardizedValue.missingness_status == MissingnessStatus.PRESENT.value,
        )
        .distinct()
    )
    return frozenset(row[0] for row in rows)


def build_extraction_instructions(exclude_faire_hints: frozenset[str] = frozenset()) -> str:
    return (
        "Extract facts about marine eDNA/molecular sampling, PCR/assay setup, "
        "sequencing, and bioinformatics methodology that are EXPLICITLY stated in "
        "the text below. Only extract what is explicitly stated -- do not infer "
        "standard laboratory practice, do not fill in expected or typical values, "
        "and do not use any knowledge beyond this text.\n\n"
        "For each fact, set fact_type_candidate to the EXACT concept name from the "
        "checklist below whenever the fact matches one of those concepts -- do "
        "not paraphrase or invent a variant spelling of a listed name. If an "
        "explicitly stated, clearly relevant fact does not match any listed "
        "concept, you may still report it using a short, descriptive "
        "fact_type_candidate of your own choosing rather than skip it.\n\n"
        "Checklist of concepts to look for (grouped by topic; a concept's example, "
        "if given, only illustrates the expected shape of an answer -- never "
        "copy an example itself into raw_value; a bracketed \"[FAIRe hint: ...]\" "
        "is only a suggestion for the OPTIONAL candidate_standard_fields output "
        "below, never something to put in fact_type_candidate):\n\n"
        f"{render_field_reference(exclude_faire_hints)}\n\n"
        "IMPORTANT: the lines above ending in a colon (e.g. \"PCR / assay "
        "setup:\") are topic headings for your own reference only -- NEVER use "
        "one of them as fact_type_candidate. Each bullet's concept name (the "
        "single word/identifier immediately after \"- \" and before its own "
        "colon, e.g. \"annealing_temperature\" in \"- annealing_temperature: PCR "
        "annealing temperature\") is what fact_type_candidate must be set to, one "
        "concept per fact -- never combine a concept name and its value into one "
        "string."
    )


# Backward-compatible default (no exclusions) -- anything importing this
# constant directly still sees the full, unfiltered checklist.
EXTRACTION_INSTRUCTIONS = build_extraction_instructions()

PROMPT_TEMPLATE = """{instructions}

Return ONLY a JSON array of objects, each with these fields:
- fact_type_candidate (string, required -- an exact concept name from the checklist above when one applies)
- raw_value (string, required)
- evidence_quote (string, required -- copied verbatim from the source text below)
- candidate_standard_fields (object, OPTIONAL -- only include this if the checklist gave a "[FAIRe hint: ...]" for the concept you used; set it to {{"faire": "<that exact hint>"}}. Omit this field entirely rather than guess a hint that wasn't given.)

If nothing in the source text supports a fact, return an empty array. Do
not paraphrase evidence_quote -- it must be an exact substring of the
source text.

Section: {section_title}

Source text:
\"\"\"
{section_text}
\"\"\"
"""


def build_prompt(section_title: str, section_text: str, exclude_faire_hints: frozenset[str] = frozenset()) -> str:
    instructions = (
        EXTRACTION_INSTRUCTIONS if not exclude_faire_hints else build_extraction_instructions(exclude_faire_hints)
    )
    return PROMPT_TEMPLATE.format(instructions=instructions, section_title=section_title, section_text=section_text)


def extract_facts_from_section(
    backend: LLMBackend,
    section_title: str,
    section_text: str,
    exclude_faire_hints: frozenset[str] = frozenset(),
) -> tuple[list[RawFactCandidate], LLMResponse | None]:
    """Returns (verified facts, the last LLMResponse -- for latency/token
    bookkeeping by the caller). An empty fact list can mean either "the
    model found nothing" or "the model's output didn't parse/verify" --
    callers that need to distinguish those should inspect the response.

    `exclude_faire_hints` (see resolved_faire_fields_for_study) drops those
    concepts from the checklist entirely -- a caller with nothing resolved
    yet passes the default empty set and gets the exact same behavior as
    before this parameter existed."""
    prompt = build_prompt(section_title, section_text, exclude_faire_hints)
    parsed, response = backend.generate_json(prompt, temperature=0)
    if parsed is None:
        return [], response

    candidates = parsed if isinstance(parsed, list) else parsed.get("facts", [])
    facts: list[RawFactCandidate] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        quote = candidate.get("evidence_quote", "")
        if not verify_evidence_quote(quote, section_text):
            continue
        fact_type = candidate.get("fact_type_candidate")
        raw_value = candidate.get("raw_value")
        if not fact_type or raw_value in (None, ""):
            continue
        hints = candidate.get("candidate_standard_fields")
        confidence_metadata = (
            {"candidate_standard_fields": hints} if isinstance(hints, dict) and hints else None
        )
        facts.append(
            RawFactCandidate(
                entity_level=EntityLevel.STUDY,
                fact_type_candidate=str(fact_type),
                raw_field_name=str(fact_type),
                raw_value=str(raw_value),
                source_locator=f"llm_text_extraction.{section_title}",
                support_type=SupportType.EXPLICIT,
                evidence_quote=quote,
                confidence_metadata=confidence_metadata,
            )
        )
    return facts, response
