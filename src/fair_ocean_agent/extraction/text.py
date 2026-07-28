"""LLM-based text fact extraction (section 12). Every candidate fact the
model returns must carry an evidence_quote that extraction.evidence
confirms is present verbatim in the exact section text it was extracted
from -- candidates that fail verification are dropped here, before they
ever reach persistence, never persisted as a fact and never surfaced to a
caller. The model only ever sees bounded, already-selected text (see
extraction/sections.py); this module never lets it browse further or pull
in outside knowledge.

`prompt_version` is deliberately explicit and threaded through to the
caller (for RawFact.prompt_version) rather than inferred from the template
text -- so a future prompt change is a visible version bump, not a silent
behavior change under the same version string.
"""
from __future__ import annotations

from fair_ocean_agent.database.enums import EntityLevel, SupportType
from fair_ocean_agent.extraction.evidence import verify_evidence_quote
from fair_ocean_agent.llm.base import LLMBackend, LLMResponse
from fair_ocean_agent.sources.base import RawFactCandidate

PROMPT_VERSION = "text-extraction-v1"

EXTRACTION_INSTRUCTIONS = (
    "Extract facts about marine eDNA/molecular sampling and sequencing methodology "
    "that are EXPLICITLY stated in the text below: sample collection dates, "
    "coordinates, depths, environmental context, DNA extraction method, "
    "PCR/amplification conditions and primer sequences, sequencing platform, and "
    "bioinformatics workflow. Only extract what is explicitly stated -- do not infer "
    "standard laboratory practice, do not fill in expected or typical values, and do "
    "not use any knowledge beyond this text."
)

PROMPT_TEMPLATE = """{instructions}

Return ONLY a JSON array of objects, each with exactly these fields:
- fact_type_candidate (string, e.g. "collection_date", "forward_primer_sequence")
- raw_value (string)
- evidence_quote (string, copied verbatim from the source text below)

If nothing in the source text supports a fact, return an empty array. Do
not paraphrase evidence_quote -- it must be an exact substring of the
source text.

Section: {section_title}

Source text:
\"\"\"
{section_text}
\"\"\"
"""


def build_prompt(section_title: str, section_text: str) -> str:
    return PROMPT_TEMPLATE.format(
        instructions=EXTRACTION_INSTRUCTIONS, section_title=section_title, section_text=section_text
    )


def extract_facts_from_section(
    backend: LLMBackend, section_title: str, section_text: str
) -> tuple[list[RawFactCandidate], LLMResponse | None]:
    """Returns (verified facts, the last LLMResponse -- for latency/token
    bookkeeping by the caller). An empty fact list can mean either "the
    model found nothing" or "the model's output didn't parse/verify" --
    callers that need to distinguish those should inspect the response."""
    prompt = build_prompt(section_title, section_text)
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
        facts.append(
            RawFactCandidate(
                entity_level=EntityLevel.STUDY,
                fact_type_candidate=str(fact_type),
                raw_field_name=str(fact_type),
                raw_value=str(raw_value),
                source_locator=f"llm_text_extraction.{section_title}",
                support_type=SupportType.EXPLICIT,
                evidence_quote=quote,
            )
        )
    return facts, response
