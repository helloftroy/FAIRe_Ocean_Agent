"""LLM-based text fact extraction (section 12), made FAIRe-aware in v2 and
corrected in v3/v4/v5. In v4 the model no longer writes evidence text from
memory: Python assigns stable IDs to source segments, the model returns
which segment ID(s) support each fact, and Python copies the authoritative
segment text into `evidence_quote`. Candidates with unknown/missing segment
IDs are dropped before persistence. The model only ever sees bounded,
already-selected text (see extraction/sections.py); this module never lets
it browse further or pull in outside knowledge.

v5 splits each section chunk into focused topic passes (sample/DNA,
primer/PCR, sequencing/library, bioinformatics/taxonomy) and skips any pass
whose source segments do not contain cues for that topic. This gives small
local models a shorter, less ambiguous checklist per call rather than one
broad "find everything" prompt. v6 adds a recall pass after each
first-pass topic extraction: the second prompt lists only the in-topic fact
types not returned by the first pass and asks whether any were missed. v7
adds a deterministic absent-value guard: candidates whose value is blank,
"none", "not specified", "not explicitly stated", etc. are dropped before
persistence and before benchmark scoring. The prompt also tells recall not
to return placeholder absence values.

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

import re
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from fair_ocean_agent.database.enums import EntityLevel, MappingMethod, MissingnessStatus, SupportType
from fair_ocean_agent.database.models import StandardizedValue
from fair_ocean_agent.extraction.faire_fields import field_names_for_reference, render_field_reference
from fair_ocean_agent.llm.base import LLMBackend, LLMBackendError, LLMResponse
from fair_ocean_agent.mapping.faire import TARGET_SCHEMA
from fair_ocean_agent.sources.base import RawFactCandidate

PROMPT_VERSION = "text-extraction-v7-focused-recall-no-absence-placeholders"
DEFAULT_MAX_SECTION_CHARS_PER_CALL = 1600

ABSENT_RAW_VALUE_STRINGS = frozenset(
    {
        "",
        "-",
        "--",
        "n/a",
        "na",
        "n.a.",
        "none",
        "null",
        "nil",
        "not available",
        "not applicable",
        "not detected",
        "not given",
        "not mentioned",
        "not provided",
        "not reported",
        "not specified",
        "not stated",
        "not explicitly mentioned",
        "not explicitly provided",
        "not explicitly reported",
        "not explicitly specified",
        "not explicitly stated",
        "unknown",
        "unspecified",
    }
)


def is_absent_raw_value(value) -> bool:
    """True when a model returned a placeholder for absence rather than a
    real source value. Missingness is represented downstream as standardized
    status, never as an extracted raw fact."""
    if value is None:
        return True
    folded = " ".join(str(value).strip().lower().split())
    if folded in ABSENT_RAW_VALUE_STRINGS:
        return True
    return bool(
        re.fullmatch(
            r"(not\s+)?(explicitly\s+)?(stated|specified|reported|provided|mentioned)(\s+in\s+(the\s+)?text)?",
            folded,
        )
    )


@dataclass(frozen=True)
class SourceSegment:
    segment_id: str
    text: str


@dataclass(frozen=True)
class ExtractionFocus:
    name: str
    description: str
    group_names: frozenset[str]
    keywords: frozenset[str]
    fallback_names: frozenset[str] = frozenset()


EXTRACTION_FOCUSES: tuple[ExtractionFocus, ...] = (
    ExtractionFocus(
        name="sample_collection_environment",
        description="sample collection, sampling depth, sample handling/storage, and environmental context facts",
        group_names=frozenset({"Sample collection / environment"}),
        keywords=frozenset(
            {
                "sample",
                "sampling",
                "collected",
                "collection",
                "filtered",
                "filter",
                "water",
                "sediment",
                "depth",
                "latitude",
                "longitude",
                "coordinate",
                "stored",
                "storage",
                "preserved",
            }
        ),
        fallback_names=frozenset({"storage_conditions", "collection_method", "environmental_context"}),
    ),
    ExtractionFocus(
        name="dna_extraction",
        description="DNA extraction kit, lysis, separation, cleanup, extraction input amount, and DNA concentration facts",
        group_names=frozenset({"DNA extraction"}),
        keywords=frozenset(
            {
                "dna",
                "extracted",
                "extraction",
                "kit",
                "lysis",
                "purif",
                "clean",
                "concentration",
                "qubit",
                "nanodrop",
            }
        ),
        fallback_names=frozenset({"DNA_extraction_method"}),
    ),
    ExtractionFocus(
        name="primer_pcr_assay",
        description="assay, target marker, primer, PCR condition, qPCR, control, and replicate facts",
        group_names=frozenset({"PCR / assay setup", "Controls & replicates", "qPCR / standard curve"}),
        keywords=frozenset(
            {
                "assay",
                "target",
                "marker",
                "primer",
                "pcr",
                "qpcr",
                "anneal",
                "cycle",
                "thermocycler",
                "amplicon",
                "master mix",
                "control",
                "replicate",
                "standard curve",
                "lod",
                "loq",
            }
        ),
        fallback_names=frozenset({"PCR_amplification_conditions"}),
    ),
    ExtractionFocus(
        name="sequencing_library",
        description="library preparation, sequencing platform/instrument, sequencing chemistry, adapter, and facility facts",
        group_names=frozenset({"Sequencing / library prep"}),
        keywords=frozenset(
            {
                "sequenc",
                "illumina",
                "miseq",
                "hiseq",
                "nextseq",
                "novaseq",
                "ion torrent",
                "pacbio",
                "nanopore",
                "minion",
                "library",
                "adapter",
                "paired-end",
                "single-end",
                "phix",
                "facility",
            }
        ),
        fallback_names=frozenset({"sequencing_platform"}),
    ),
    ExtractionFocus(
        name="bioinformatics_taxonomy",
        description="bioinformatics workflow, sequence processing, reference database, and taxonomic assignment facts",
        group_names=frozenset({"Bioinformatics workflow", "Taxonomic assignment output"}),
        keywords=frozenset(
            {
                "bioinformatic",
                "demultiplex",
                "trim",
                "merge",
                "denois",
                "chimera",
                "cluster",
                "otu",
                "asv",
                "qiime",
                "dada2",
                "mothur",
                "blast",
                "silva",
                "greengenes",
                "midori",
                "reference database",
                "taxonom",
                "classifier",
            }
        ),
    ),
)


def resolved_faire_fields_for_study(session: Session, study_id: str) -> frozenset[str]:
    """FAIRe target_field names already resolved by trusted non-LLM paths.

    Only non-review exact/deterministic rows suppress future LLM asks. A
    review-required or suggested-semantic row is useful evidence, but should
    not stop the paper/supplement text extractor from looking for a better
    explicit statement later.
    """
    rows = session.execute(
        select(StandardizedValue.target_field)
        .where(
            StandardizedValue.study_id == study_id,
            StandardizedValue.target_schema == TARGET_SCHEMA,
            StandardizedValue.missingness_status == MissingnessStatus.PRESENT.value,
            StandardizedValue.review_required.is_(False),
            StandardizedValue.mapping_method.in_(
                (
                    MappingMethod.EXACT_IDENTIFIER.value,
                    MappingMethod.EXACT_LABEL.value,
                    MappingMethod.DETERMINISTIC_SYNONYM.value,
                )
            ),
        )
        .distinct()
    )
    return frozenset(row[0] for row in rows)


def present_faire_fields_for_study(session: Session, study_id: str) -> frozenset[str]:
    """FAIRe fields already supported by any inspected stage.

    This is intentionally broader than `resolved_faire_fields_for_study`.
    The paper pass should only trust structured, non-review mappings when
    deciding what not to ask. The later supplement pass should also avoid
    re-asking for fields the paper explicitly supported, even though those
    LLM-derived mappings remain review-required.
    """
    rows = session.execute(
        select(StandardizedValue.target_field)
        .where(
            StandardizedValue.study_id == study_id,
            StandardizedValue.target_schema == TARGET_SCHEMA,
            StandardizedValue.missingness_status == MissingnessStatus.PRESENT.value,
            StandardizedValue.standardized_value.is_not(None),
        )
        .distinct()
    )
    return frozenset(row[0] for row in rows)


def build_extraction_instructions(
    exclude_faire_hints: frozenset[str] = frozenset(),
    focus: ExtractionFocus | None = None,
    include_native_names: frozenset[str] | None = None,
    recall_pass: bool = False,
) -> str:
    field_reference = render_field_reference(
        exclude_faire_hints,
        include_group_names=focus.group_names if focus else None,
        include_fallback_names=focus.fallback_names if focus else None,
        include_native_names=include_native_names,
    )
    focus_sentence = (
        f"This focused pass is only for {focus.description}. "
        "Ignore explicitly-stated facts outside that focus; another pass will handle them.\n\n"
        if focus
        else ""
    )
    recall_sentence = (
        "This is a recall-focused second pass. A first extraction pass already ran over this same text. "
        "Check whether it missed any of the fact types in the checklist below for this same focused topic only. "
        "Return only newly found facts whose fact_type_candidate is one of the listed missing types; do not repeat "
        "facts that were already extracted. If the text does not explicitly state a listed missing type, omit that "
        "type entirely. Never return placeholder absence values such as \"none\", \"not specified\", "
        "\"not explicitly stated\", \"not reported\", \"unknown\", or an empty string.\n\n"
        if recall_pass
        else ""
    )
    return (
        "Extract facts about marine eDNA/molecular sampling, PCR/assay setup, "
        "sequencing, and bioinformatics methodology that are EXPLICITLY stated in "
        "the text below. Only extract what is explicitly stated -- do not infer "
        "standard laboratory practice, do not fill in expected or typical values, "
        "and do not use any knowledge beyond this text.\n\n"
        f"{focus_sentence}"
        f"{recall_sentence}"
        "For each fact, set fact_type_candidate to the EXACT concept name from the "
        "checklist below whenever the fact matches one of those concepts -- do "
        "not paraphrase or invent a variant spelling of a listed name. If an "
        "explicitly stated fact does not match a listed concept, skip it for "
        "this focused pass.\n\n"
        "Checklist of concepts to look for (grouped by topic; a concept's example, "
        "if given, only illustrates the expected shape of an answer -- never "
        "copy an example itself into raw_value; a bracketed \"[FAIRe hint: ...]\" "
        "is only a suggestion for the OPTIONAL candidate_standard_fields output "
        "below, never something to put in fact_type_candidate):\n\n"
        f"{field_reference}\n\n"
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
- evidence_id (string, required -- one of the source segment IDs below that explicitly supports this fact)
- candidate_standard_fields (object, OPTIONAL -- only include this if the checklist gave a "[FAIRe hint: ...]" for the concept you used; set it to {{"faire": "<that exact hint>"}}. Omit this field entirely rather than guess a hint that wasn't given.)

If nothing in the source text supports a fact, return an empty array. Do
not reproduce or paraphrase source text. Your evidence_id must point to a
listed segment that explicitly supports the fact. Never return a fact whose
raw_value is blank or an absence placeholder such as "none", "not specified",
"not explicitly stated", "not reported", "unknown", or "not applicable";
omit that fact instead.

Section: {section_title}

Source segments:
{source_segments}
"""


def _segment_prefix(section_title: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", section_title.upper())
    return "_".join(words[:3]) or "SECTION"


def segment_source_text(section_title: str, section_text: str) -> list[SourceSegment]:
    """Assign stable IDs to sentence-ish source segments for model citation.

    Paragraphs are preserved when they are short; long paragraphs are split
    on sentence boundaries. This makes the model choose an ID rather than
    regenerate evidence text, while keeping Python in charge of the exact
    quote stored downstream.
    """
    text = section_text.strip()
    if not text:
        return []

    raw_segments: list[str] = []
    for paragraph in [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]:
        if len(paragraph) <= 700:
            raw_segments.append(paragraph)
            continue
        sentence_parts = re.split(r"(?<=[.!?])\s+", paragraph)
        current = ""
        for sentence in [part.strip() for part in sentence_parts if part.strip()]:
            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) <= 700:
                current = candidate
            else:
                if current:
                    raw_segments.append(current)
                current = sentence
        if current:
            raw_segments.append(current)

    prefix = _segment_prefix(section_title)
    return [
        SourceSegment(segment_id=f"{prefix}.P{index:03d}", text=segment)
        for index, segment in enumerate(raw_segments, start=1)
    ]


def _render_source_segments(segments: list[SourceSegment]) -> str:
    return "\n".join(f"{segment.segment_id}: {segment.text}" for segment in segments)


def build_prompt(
    section_title: str,
    section_text: str,
    exclude_faire_hints: frozenset[str] = frozenset(),
    segments: list[SourceSegment] | None = None,
    focus: ExtractionFocus | None = None,
    include_native_names: frozenset[str] | None = None,
    recall_pass: bool = False,
) -> str:
    instructions = build_extraction_instructions(exclude_faire_hints, focus, include_native_names, recall_pass)
    source_segments = segments if segments is not None else segment_source_text(section_title, section_text)
    return PROMPT_TEMPLATE.format(
        instructions=instructions,
        section_title=section_title,
        source_segments=_render_source_segments(source_segments),
    )


def split_section_text(section_text: str, max_chars: int = DEFAULT_MAX_SECTION_CHARS_PER_CALL) -> list[str]:
    """Split long selected paper sections into bounded extraction calls.

    The split is character-based so it works without a model-specific
    tokenizer. The default is conservative for qwen3 under Ollama's common
    4096-token context: the FAIRe-aware checklist already consumes most of
    the prompt before source text is added.
    """
    text = section_text.strip()
    if not text:
        return []
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current = ""
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            start = 0
            while start < len(paragraph):
                end = min(len(paragraph), start + max_chars)
                if end < len(paragraph):
                    split_at = max(
                        paragraph.rfind(". ", start, end),
                        paragraph.rfind("; ", start, end),
                        paragraph.rfind(", ", start, end),
                        paragraph.rfind(" ", start, end),
                    )
                    if split_at > start + max_chars // 2:
                        end = split_at + 1
                chunks.append(paragraph[start:end].strip())
                start = end
            continue

        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = paragraph

    if current:
        chunks.append(current)
    return chunks


def split_segments_for_calls(
    segments: list[SourceSegment],
    max_chars: int = DEFAULT_MAX_SECTION_CHARS_PER_CALL,
) -> list[list[SourceSegment]]:
    if not segments:
        return []
    if max_chars <= 0:
        return [segments]

    chunks: list[list[SourceSegment]] = []
    current: list[SourceSegment] = []
    current_chars = 0
    for segment in segments:
        rendered_len = len(segment.segment_id) + len(segment.text) + 2
        if current and current_chars + rendered_len > max_chars:
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(segment)
        current_chars += rendered_len
    if current:
        chunks.append(current)
    return chunks


def _text_matches_focus(text: str, focus: ExtractionFocus) -> bool:
    folded = text.lower()
    return any(keyword in folded for keyword in focus.keywords)


def segments_for_focus(
    section_title: str,
    segments: list[SourceSegment],
    focus: ExtractionFocus,
) -> list[SourceSegment]:
    title_matches = _text_matches_focus(section_title, focus)
    return [
        segment
        for segment in segments
        if title_matches or _text_matches_focus(segment.text, focus)
    ]


def fact_type_names_for_focus(
    focus: ExtractionFocus,
    exclude_faire_hints: frozenset[str] = frozenset(),
) -> frozenset[str]:
    return field_names_for_reference(
        exclude_faire_hints,
        include_group_names=focus.group_names,
        include_fallback_names=focus.fallback_names,
    )


def _source_has_nucleotide_sequence(segments: list[SourceSegment]) -> bool:
    text = " ".join(segment.text for segment in segments)
    return any(
        len(match.group(0)) >= 8
        for match in re.finditer(r"\b[ACGTRYSWKMBDHVNacgtryswkmbdhvn]{8,}\b", text)
    )


def recall_missing_fact_types(
    focus: ExtractionFocus,
    exclude_faire_hints: frozenset[str],
    accepted_fact_types: set[str],
    segments: list[SourceSegment],
) -> frozenset[str]:
    missing_types = set(fact_type_names_for_focus(focus, exclude_faire_hints) - accepted_fact_types)

    # Small models often confuse primer names (e.g. MiFish-U-F) with primer
    # sequences on a recall pass. Only ask for sequence fields when the text
    # contains a nucleotide-like token long enough to plausibly be a primer.
    if not _source_has_nucleotide_sequence(segments):
        missing_types.discard("forward_primer_sequence")
        missing_types.discard("reverse_primer_sequence")

    return frozenset(missing_types)


def extract_facts_from_section(
    backend: LLMBackend,
    section_title: str,
    section_text: str,
    exclude_faire_hints: frozenset[str] = frozenset(),
    max_section_chars_per_call: int = DEFAULT_MAX_SECTION_CHARS_PER_CALL,
    max_output_tokens: int | None = None,
    focuses: tuple[ExtractionFocus, ...] = EXTRACTION_FOCUSES,
    recall_second_pass: bool = True,
) -> tuple[list[RawFactCandidate], LLMResponse | None]:
    """Returns (verified facts, the last LLMResponse -- for latency/token
    bookkeeping by the caller). An empty fact list can mean either "the
    model found nothing" or "the model's output didn't parse/verify" --
    callers that need to distinguish those should inspect the response.

    `exclude_faire_hints` (see resolved_faire_fields_for_study) drops those
    concepts from the checklist entirely -- a caller with nothing resolved
    yet passes the default empty set and gets the exact same behavior as
    before this parameter existed."""
    segments = segment_source_text(section_title, section_text)
    segment_chunks = split_segments_for_calls(segments, max_section_chars_per_call)
    if not segment_chunks:
        return [], None

    facts: list[RawFactCandidate] = []
    seen: set[tuple[str, str, str]] = set()
    last_response: LLMResponse | None = None

    active_focuses = focuses or EXTRACTION_FOCUSES
    for index, chunk_segments in enumerate(segment_chunks):
        chunk_title = section_title if len(segment_chunks) == 1 else f"{section_title} [chunk {index + 1}/{len(segment_chunks)}]"
        for focus in active_focuses:
            focused_segments = segments_for_focus(section_title, chunk_segments, focus)
            if not focused_segments:
                continue
            segment_lookup = {segment.segment_id: segment.text for segment in focused_segments}
            focused_title = f"{chunk_title} [{focus.name}]"
            prompt = build_prompt(
                focused_title,
                "",
                exclude_faire_hints,
                segments=focused_segments,
                focus=focus,
            )
            parsed, response = backend.generate_json(prompt, temperature=0, max_tokens=max_output_tokens)
            last_response = response
            candidates = []
            accepted_facts: list[RawFactCandidate] = []
            if parsed is not None:
                candidates = parsed if isinstance(parsed, list) else (parsed.get("facts", []) if isinstance(parsed, dict) else [])
                accepted_facts = _facts_from_candidates(
                    candidates,
                    segment_lookup,
                    focused_title,
                    seen,
                    allowed_fact_types=fact_type_names_for_focus(focus, exclude_faire_hints),
                )
                facts.extend(accepted_facts)

            if not recall_second_pass:
                continue

            missing_types = recall_missing_fact_types(
                focus,
                exclude_faire_hints,
                {fact.fact_type_candidate for fact in accepted_facts},
                focused_segments,
            )
            if not missing_types:
                continue
            recall_prompt = build_prompt(
                f"{focused_title} [recall]",
                "",
                exclude_faire_hints,
                segments=focused_segments,
                focus=focus,
                include_native_names=missing_types,
                recall_pass=True,
            )
            try:
                parsed, response = backend.generate_json(recall_prompt, temperature=0, max_tokens=max_output_tokens)
            except LLMBackendError:
                continue
            last_response = response
            if parsed is None:
                continue
            candidates = parsed if isinstance(parsed, list) else (parsed.get("facts", []) if isinstance(parsed, dict) else [])
            facts.extend(
                _facts_from_candidates(
                    candidates,
                    segment_lookup,
                    f"{focused_title} [recall]",
                    seen,
                    allowed_fact_types=missing_types,
                )
            )
    return facts, last_response


def _facts_from_candidates(
    candidates,
    segment_lookup: dict[str, str],
    section_title: str,
    seen: set[tuple[str, str, str]],
    allowed_fact_types: frozenset[str] | None = None,
) -> list[RawFactCandidate]:
    facts: list[RawFactCandidate] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        evidence_ids = _candidate_evidence_ids(candidate)
        if not evidence_ids or any(evidence_id not in segment_lookup for evidence_id in evidence_ids):
            continue
        quote = "\n".join(segment_lookup[evidence_id] for evidence_id in evidence_ids)
        fact_type = candidate.get("fact_type_candidate")
        raw_value = candidate.get("raw_value")
        if not fact_type or is_absent_raw_value(raw_value):
            continue
        if allowed_fact_types is not None and str(fact_type) not in allowed_fact_types:
            continue
        dedupe_key = (str(fact_type), str(raw_value), quote)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        hints = candidate.get("candidate_standard_fields")
        confidence_metadata = {"evidence_ids": evidence_ids}
        if isinstance(hints, dict) and hints:
            confidence_metadata["candidate_standard_fields"] = hints
        facts.append(
            RawFactCandidate(
                entity_level=EntityLevel.STUDY,
                fact_type_candidate=str(fact_type),
                raw_field_name=str(fact_type),
                raw_value=str(raw_value),
                source_locator=f"llm_text_extraction.{section_title}.{'|'.join(evidence_ids)}",
                support_type=SupportType.EXPLICIT,
                evidence_quote=quote,
                confidence_metadata=confidence_metadata,
            )
        )
    return facts


def _candidate_evidence_ids(candidate: dict) -> list[str]:
    value = candidate.get("evidence_id", candidate.get("evidence_ids"))
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []
