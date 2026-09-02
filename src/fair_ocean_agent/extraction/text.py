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

**v7 -> v8: the per-topic split from v5 is no longer the default.**
Measured live against a local qwen3:4b: v5's 5 topic passes x v6's
recall-on-any-missing-type retry meant up to 10 sequential LLM calls per
section chunk regardless of chunk length, at ~30-240s per call on
consumer hardware -- minutes per section, tens of minutes per paper. That
split was sized around Ollama's OpenAI-compatible endpoint silently
enforcing a ~4096-token effective context unless a model's own Modelfile
bakes in a larger `num_ctx` (see config.py's LLMConfig docstring) --
keeping each topic's own prompt small enough to fit. Once a real, larger
context is available (a model variant with `num_ctx` actually baked in),
the full checklist (~3,300 tokens) comfortably fits alongside a real
section in one call, so `extract_facts_from_section` now defaults to a
single collapsed pass over every concept per chunk (`focuses=(None,)`);
`EXTRACTION_FOCUSES` remains available to opt back into the old
fine-grained split for a smaller-context model. v8 also narrows the
recall trigger: a pass now only gets a retry when it found ZERO facts (a
real parse failure or the model missing everything), not merely because
some checklist concept went unmentioned -- most real sections never
mention every concept, and retrying on partial coverage doubled call
volume for little benefit.

**v8 -> v9:** low-value optional project fields are retained in the FAIRe
registry, structured-source mappings, and exports but omitted from both
paper and supplement LLM checklists. The same policy also removes the
generic PCR narrative fallback that mapped only to pcr_method_additional;
the model still receives the atomic PCR fields that carry useful values.

**v10 -> v11:** deterministic paper/supplement search flags now run before
LLM extraction (see extraction/search_flags.py). They record cheap,
evidence-bearing booleans such as "PCR is present" and "probe-based qPCR
or ddPCR cues are present" so later targeted searches can be conditionally
activated without asking the model to hunt for every optional branch on
every paper.

**v11 -> v12:** the deterministic pre-LLM pass also runs the controlled
projectMetadata searches from the FAIRe-NOAA controlled-search sheet.
Conditional searches activate only when their configured flag is found in
the same paper/supplement text, and multiple literal source matches are
stored as one pipe-delimited raw value.

**v12 -> v13:** control-use booleans (`neg_cont_0_1`, `pos_cont_0_1`)
join the deterministic pre-LLM pass. They emit `1` from explicit use
evidence, `0` only from explicit "none/not used" evidence, and otherwise
stay absent so downstream missingness remains unresolved/not found rather
than a false zero.

**v13 -> v14:** the same centralized deterministic pass now captures
`sterilise_method` as direct contamination-minimization sentence text,
`biological_rep` as an integer only from collection/sample replicate
phrasing, and `assay_type` as a fixed-vocabulary cue classifier that may
return both targeted and metabarcoding when both are explicitly evidenced.

**v14 -> v15:** library-preparation sequencing fields that need judgement
(`barcoding_pcr_appr`, `lib_screen`, `adapter_forward`,
`adapter_reverse`) use a narrow candidate-quote LLM pass before the broad
paper/supplement extraction. Python supplies only search-term-matched
sentences with stable quote IDs, the model returns field/value/quote_id,
and Python stores the literal supporting quote itself.

**v15 -> v16: the LLM checklist itself is now conditionally gated on the
same deterministic flags the pre-LLM pass already computes.**
`extraction/faire_fields.py`'s taxonomy entries can now carry
`required_any_flags` (mirroring `search_flags.ControlledSearchField`'s own
field, for consistency -- one gating vocabulary shared by both the
deterministic and LLM sides, not two). The entire "PCR / assay setup"
group (primers, target gene, thermal profile, master mix, ...) requires
`pcr_0_1`; the two new probe fields (`probe_sequence`/
`probe_concentration`) additionally accept `probe_based_qPCR_ddPCR_assay_0_1`.
A non-PCR paper's checklist genuinely shrinks now instead of always
showing the full PCR section; a PCR paper's checklist genuinely includes
it, matching a real NOAA FAIRe checklist's own per-field
conditional-requirement column. `target_gene`/`thermocycler`/
`commercial_master_mix`/`assay_type`/`biological_replicate_count` were
also gated (not excluded) here even though
`search_flags.CONTROLLED_SEARCH_FIELDS` already covers the same concepts
deterministically -- checked real gold data first and found that
deterministic path is literal-substring matching against a curated term
list, not free-text extraction, and it demonstrably misses or mangles
real values a careful LLM read would get right; gating keeps the LLM as
the richer complement instead of replacing it. `active_flags` (the new
parameter on `extract_facts_from_section`/`build_prompt`/
`build_extraction_instructions`) governs both what the prompt shows AND
what `allowed_fact_types` accepts back on the main and recall passes, so
a gated field can never sneak through a hallucinated response even when
it was never shown.

**v16 -> v17: topic-focused extraction (EXTRACTION_FOCUSES) is back, in a
finer-grained form, and workflow/handlers.py + workflow/supplement_handlers.py
now pass it explicitly.** v8 collapsed the old 5-focus split into one
single-pass call over the full checklist once a real, larger model context
became available. That single pass is still this function's own default
(`focuses=(None,)`) -- unchanged, so nothing importing it directly without
an explicit `focuses=` sees any behavior change -- but real production
extraction (both call sites) now passes `focuses=EXTRACTION_FOCUSES`
explicitly, per an explicit user request: confirmed live, the old
`primer_pcr_assay` focus bundled three whole FAIRe field groups (PCR /
assay setup=19 + Controls & replicates=1 + qPCR / standard curve=16 = 36
fields) into one call -- barely smaller than the 68-field full checklist
it replaced -- and a real extraction against a real paper's own Methods
text correctly returned ~15 other facts from that call while silently
dropping forward_primer_name/reverse_primer_name, even though the primer
names were stated in plain text right next to facts the model got right.
EXTRACTION_FOCUSES now has 8 focuses (was 5), sized 5-14 fields each (was
up to 36) -- the oversized PCR/qPCR bundle split into four genuinely
tight topics (primer_target, pcr_assay_setup, qpcr_standard_curve,
qpcr_detection_limits) via ExtractionFocus's new `native_names` field,
which restricts a focus to a field subset within its own group_names
(field_names_for_reference only selected whole FAIRe groups before this).
More LLM calls per section, deliberately traded for smaller, less
ambiguous checklists per call -- per the user's own explicit priority
("even if that means it takes longer"). segments_for_focus's existing
keyword-cue skip (a focus with no matching cues in a section's own text
never fires) keeps this from actually costing 8x calls on every section in
practice.

**v9 -> v10: optional per-fact `assay_tag` for multi-assay papers.** A
paper can describe more than one distinct assay run on the same samples
(e.g. a 16S PCR assay and an 18S PCR assay), each with its own primers,
target gene, annealing temperature, etc. Previously every extracted fact
was hardcoded to `entity_level=EntityLevel.STUDY` with no entity at all --
two assays' PCR/qPCR facts would collide onto the same
`(study, target_field)` mapping key downstream and the second assay's data
was lost, flagged only as a "conflict" for manual review. The model may now
return an optional `assay_tag` string per fact; when present, non-empty,
and the fact's `fact_type_candidate` is one this pipeline considers
assay-scoped (`extraction/faire_fields.assay_scoped_field_names()` --
primers, target gene, PCR/qPCR conditions, ...), `_facts_from_candidates`
tags the resulting `RawFactCandidate` with `entity_level=EntityLevel.ASSAY`
and `entity_external_id=assay_tag`, so `workflow/handlers.py` materializes
a real per-assay `Entity` and `mapping/faire.py` produces one
`projectMetadata` row per assay instead of colliding them (matching real
FAIRe's own export layout -- see `schemas/faire/README.md` -- which is one
`projectMetadata` row per `assay_name`, not one global row per study). A
fact with no `assay_tag` (the overwhelmingly common single-assay case)
keeps today's exact `entity_level=STUDY` behavior -- this is a strictly
additive, backward-compatible change. Cross-call/cross-section assay-tag
consistency is a known, unaddressed limitation: the model has no memory
between separate section calls or between chunks within one long section,
so the prompt biases it toward reusing the paper's own given assay name
(most likely to be repeated consistently) rather than an arbitrary
placeholder -- a real fix would need a separate reconciliation pass, out of
scope for this change.

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
from fair_ocean_agent.extraction.faire_fields import (
    assay_scoped_field_names,
    field_names_for_reference,
    render_field_reference,
)
from fair_ocean_agent.llm.base import LLMBackend, LLMResponse
from fair_ocean_agent.mapping.faire import TARGET_SCHEMA
from fair_ocean_agent.sources.base import RawFactCandidate

PROMPT_VERSION = "text-extraction-v21-expedition-id"
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
        "not resolved",
        "not specified",
        "not stated",
        "not explicitly mentioned",
        "not explicitly provided",
        "not explicitly reported",
        "not explicitly specified",
        "not explicitly stated",
        "see above",
        "see below",
        "see text",
        "unknown",
        "unresolved",
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
    folded_without_terminal_punctuation = folded.rstrip(".;:")
    if folded in ABSENT_RAW_VALUE_STRINGS or folded_without_terminal_punctuation in ABSENT_RAW_VALUE_STRINGS:
        return True
    return bool(
        re.fullmatch(
            r"(not\s+)?(explicitly\s+)?(stated|specified|reported|provided|mentioned)(\s+in\s+(the\s+)?text)?",
            folded_without_terminal_punctuation,
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
    # Restricts a focus to a subset of its own group_names' fields --
    # lets one oversized FAIRe field group (e.g. "PCR / assay setup"'s 19
    # fields) be split into several genuinely small, topically-tight
    # passes instead of one still-large pass. Empty (the default) means
    # every field in group_names is included, same as before this existed.
    native_names: frozenset[str] = frozenset()


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
    # The old single "primer_pcr_assay" focus bundled three whole FAIRe
    # field groups (PCR / assay setup=19 + Controls & replicates=1 +
    # qPCR / standard curve=16 = 36 fields) into one call -- barely
    # smaller than the full 68-field checklist it was meant to replace,
    # and still large enough that a small local model reliably dropped
    # specific fields (confirmed live: forward_primer_name/
    # reverse_primer_name went missing from real extractions even though
    # ~15 OTHER facts from the very same call succeeded). Split into four
    # genuinely small, topically-tight focuses instead -- per an explicit
    # user request to keep every pass small, not just this one group.
    # native_names (see ExtractionFocus) restricts each to its own field
    # subset within group_names, since field_names_for_reference only
    # selects whole FAIRe groups otherwise.
    ExtractionFocus(
        name="primer_target",
        description="PCR/amplicon primer identity: target gene/subfragment, primer names and sequences, probe, and amplicon size",
        group_names=frozenset({"PCR / assay setup"}),
        native_names=frozenset(
            {
                "target_gene",
                "target_subfragment",
                "forward_primer_sequence",
                "reverse_primer_sequence",
                "forward_primer_name",
                "reverse_primer_name",
                "amplicon_size",
                "probe_sequence",
                "probe_concentration",
            }
        ),
        keywords=frozenset({"primer", "target", "marker", "amplicon", "probe", "16s", "18s", "its", "coi"}),
    ),
    ExtractionFocus(
        name="pcr_assay_setup",
        description="assay identity, PCR thermal cycling conditions, master mix, and PCR replicate facts",
        group_names=frozenset({"PCR / assay setup", "Controls & replicates"}),
        # Real gap found live: PCR_amplification_conditions/second_pcr_
        # amplification_conditions (pcr_method_additional/pcr2_method_
        # additional's own source) were completely absent from every real
        # production extraction call once EXTRACTION_FOCUSES went back into
        # use (v17) -- this focus's own `native_names` allowlist restricts
        # which "PCR / assay setup" fields even get shown, and neither
        # native_name was ever added to it. The `fallback_names={"PCR_
        # amplification_conditions"}` that used to be here was already
        # dead: it only ever restricted FALLBACK_NARRATIVE_FIELDS, but this
        # field had already been moved OUT of that list and into
        # FIELD_GROUPS["PCR / assay setup"] by an earlier fix, so
        # `include_fallback_names` never had any effect on it (and
        # second_pcr_amplification_conditions was never referenced by
        # either mechanism at all). Both narrative fields belong in
        # native_names now, alongside their atomic siblings.
        native_names=frozenset(
            {
                "assay_name",
                "assay_type",
                "annealing_temperature",
                "pcr_cycle_count",
                "commercial_master_mix",
                "custom_master_mix",
                "second_pcr_annealing_temperature",
                "second_pcr_cycle_count",
                "PCR_amplification_conditions",
                "second_pcr_amplification_conditions",
                "assay_target_taxa",
                "study_target_taxonomic_scope",
                "pcr_replicate_count",
            }
        ),
        keywords=frozenset({"assay", "pcr", "anneal", "cycle", "thermocycler", "master mix", "replicate"}),
    ),
    ExtractionFocus(
        name="qpcr_standard_curve",
        description="qPCR quantification cycle, standard curve, and amplification-efficiency facts",
        group_names=frozenset({"qPCR / standard curve"}),
        native_names=frozenset(
            {
                "quantification_cycle_threshold",
                "quantification_cycle",
                "qpcr_standard_concentration",
                "qpcr_standard_concentration_unit",
                "qpcr_standard_source",
                "standard_curve_slope",
                "standard_curve_intercept",
                "standard_curve_r_squared",
                "qpcr_amplification_efficiency",
            }
        ),
        keywords=frozenset({"qpcr", "standard curve", "efficiency", "slope", "intercept", " ct ", " cq ", "threshold cycle"}),
    ),
    ExtractionFocus(
        name="qpcr_detection_limits",
        description="qPCR estimated copy number and assay limit of detection/quantification facts",
        group_names=frozenset({"qPCR / standard curve"}),
        native_names=frozenset(
            {
                "estimated_copy_number",
                "estimated_copy_number_unit",
                "estimated_copy_number_method",
                "assay_limit_of_detection",
                "assay_limit_of_detection_unit",
                "assay_limit_of_quantification",
                "assay_limit_of_quantification_unit",
            }
        ),
        keywords=frozenset({"copy number", "limit of detection", "limit of quantification", "lod", "loq"}),
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
    active_flags: frozenset[str] = frozenset(),
) -> str:
    field_reference = render_field_reference(
        exclude_faire_hints,
        include_group_names=focus.group_names if focus else None,
        include_fallback_names=focus.fallback_names if focus else None,
        include_native_names=include_native_names or (focus.native_names if focus and focus.native_names else None),
        active_flags=active_flags,
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
        "string.\n\n"
        "If the text describes MORE THAN ONE distinct assay or primer set (for "
        "example, separate PCR protocols targeting different genes run on the "
        "same samples), give each assay-identity/PCR/qPCR fact (assay_name, "
        "assay_type, target_gene, primers, PCR/qPCR conditions and related "
        "fields) an assay_tag string that is the SAME for every fact belonging "
        "to that one assay -- prefer the paper's own name for the assay if it "
        "gives one (e.g. \"16S-V3V4\"), since that is what you are most likely "
        "to use consistently if this assay is described again elsewhere in the "
        "paper; otherwise use a short, descriptive label of your own. If the "
        "text only describes ONE assay, omit assay_tag entirely -- do not "
        "invent a placeholder tag for a single-assay paper, and never add "
        "assay_tag to a fact that isn't about one specific assay's identity or "
        "setup."
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
- assay_tag (string, OPTIONAL -- only for a fact about one specific assay's identity/PCR/qPCR setup, and only when the text describes more than one assay; see the instructions above. Omit entirely otherwise.)

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
    active_flags: frozenset[str] = frozenset(),
) -> str:
    instructions = build_extraction_instructions(
        exclude_faire_hints, focus, include_native_names, recall_pass, active_flags=active_flags
    )
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
    focus: ExtractionFocus | None,
) -> list[SourceSegment]:
    """`focus=None` means "no topic restriction" -- every segment is in
    scope (used by the default, collapsed single-pass extraction strategy;
    see extract_facts_from_section)."""
    if focus is None:
        return segments
    title_matches = _text_matches_focus(section_title, focus)
    return [
        segment
        for segment in segments
        if title_matches or _text_matches_focus(segment.text, focus)
    ]


def fact_type_names_for_focus(
    focus: ExtractionFocus | None,
    exclude_faire_hints: frozenset[str] = frozenset(),
    active_flags: frozenset[str] = frozenset(),
) -> frozenset[str]:
    """`focus=None` returns every checklist name (all groups, all fallback
    names) -- no topic restriction."""
    if focus is None:
        return field_names_for_reference(exclude_faire_hints, active_flags=active_flags)
    return field_names_for_reference(
        exclude_faire_hints,
        include_group_names=focus.group_names,
        include_fallback_names=focus.fallback_names,
        include_native_names=focus.native_names or None,
        active_flags=active_flags,
    )


def _source_has_nucleotide_sequence(segments: list[SourceSegment]) -> bool:
    text = " ".join(segment.text for segment in segments)
    return any(
        len(match.group(0)) >= 8
        for match in re.finditer(r"\b[ACGTRYSWKMBDHVNacgtryswkmbdhvn]{8,}\b", text)
    )


def recall_missing_fact_types(
    focus: ExtractionFocus | None,
    exclude_faire_hints: frozenset[str],
    accepted_fact_types: set[str],
    segments: list[SourceSegment],
    active_flags: frozenset[str] = frozenset(),
) -> frozenset[str]:
    missing_types = set(fact_type_names_for_focus(focus, exclude_faire_hints, active_flags=active_flags) - accepted_fact_types)

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
    focuses: tuple[ExtractionFocus | None, ...] = (None,),
    recall_second_pass: bool = True,
    active_flags: frozenset[str] = frozenset(),
) -> tuple[list[RawFactCandidate], LLMResponse | None]:
    """Returns (verified facts, the last LLMResponse -- for latency/token
    bookkeeping by the caller). An empty fact list can mean either "the
    model found nothing" or "the model's output didn't parse/verify" --
    callers that need to distinguish those should inspect the response.

    `exclude_faire_hints` (see resolved_faire_fields_for_study) drops those
    concepts from the checklist entirely -- a caller with nothing resolved
    yet passes the default empty set and gets the exact same behavior as
    before this parameter existed.

    `focuses` defaults to a single collapsed pass over the FULL checklist
    per chunk (`(None,)`) -- one LLM call per chunk instead of one per
    topic group. This used to default to `EXTRACTION_FOCUSES` (5 topic-
    scoped passes: sample collection, DNA extraction, PCR/assay,
    sequencing, bioinformatics/taxonomy), each with its own recall retry --
    up to 10 sequential calls per section regardless of section length,
    measured live at ~30-240s per call against a local qwen3:4b, i.e.
    minutes per section. That per-topic split was sized for the ~4096-token
    effective context Ollama's OpenAI-compatible endpoint silently enforces
    unless a model's own Modelfile bakes in a larger num_ctx (see
    config.py's LLMConfig docstring) -- once a real, larger context is
    available, the full checklist (~3,300 tokens) comfortably fits
    alongside a real paper section in one call. `EXTRACTION_FOCUSES`
    remains importable and usable here (pass it explicitly as `focuses=`)
    for anyone who wants the old fine-grained-per-topic behavior back, e.g.
    for a smaller-context model.

    `recall_second_pass` only retries a chunk when its first pass found
    ZERO facts (a real parse failure or the model missing everything) --
    it does not retry merely because some checklist concepts went
    unmentioned, since most real sections never mention every concept and
    retrying on partial coverage doubled call volume for little benefit.

    `active_flags` gates conditional taxonomy fields (extraction/faire_fields.py's
    `required_any_flags`, e.g. the whole "PCR / assay setup" group requires
    `pcr_0_1`) -- both what the prompt shows the model AND what
    `allowed_fact_types` accepts back, on both the main and recall passes,
    so a gated field can never sneak through via the recall pass or a
    hallucinated response even though it was never shown. Empty (the
    default) hides every conditional field -- a caller with real
    deterministic-flag facts already computed (see
    extraction/search_flags.py's detect_text_search_flags, already run
    before this in every production call site) must pass them explicitly."""
    segments = segment_source_text(section_title, section_text)
    segment_chunks = split_segments_for_calls(segments, max_section_chars_per_call)
    if not segment_chunks:
        return [], None

    facts: list[RawFactCandidate] = []
    seen: set[tuple[str, str, str, str]] = set()
    last_response: LLMResponse | None = None

    active_focuses = focuses if focuses else (None,)
    for index, chunk_segments in enumerate(segment_chunks):
        chunk_title = section_title if len(segment_chunks) == 1 else f"{section_title} [chunk {index + 1}/{len(segment_chunks)}]"
        for focus in active_focuses:
            focused_segments = segments_for_focus(section_title, chunk_segments, focus)
            if not focused_segments:
                continue
            segment_lookup = {segment.segment_id: segment.text for segment in focused_segments}
            focused_title = chunk_title if focus is None else f"{chunk_title} [{focus.name}]"
            prompt = build_prompt(
                focused_title,
                "",
                exclude_faire_hints,
                segments=focused_segments,
                focus=focus,
                active_flags=active_flags,
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
                    allowed_fact_types=fact_type_names_for_focus(focus, exclude_faire_hints, active_flags=active_flags),
                )
                facts.extend(accepted_facts)

            if not recall_second_pass:
                continue
            if accepted_facts:
                continue  # found at least one fact already -- no automatic retry

            missing_types = recall_missing_fact_types(
                focus,
                exclude_faire_hints,
                {fact.fact_type_candidate for fact in accepted_facts},
                focused_segments,
                active_flags=active_flags,
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
                active_flags=active_flags,
            )
            parsed, response = backend.generate_json(recall_prompt, temperature=0, max_tokens=max_output_tokens)
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


def _candidate_assay_tag(candidate: dict, fact_type: str) -> str | None:
    """Normalizes a model-supplied assay_tag, dropping placeholder/absent
    values via the same is_absent_raw_value check used for raw_value, and
    ignoring the tag entirely for a fact_type this pipeline doesn't
    consider assay-scoped (see assay_scoped_field_names's docstring)."""
    if fact_type not in assay_scoped_field_names():
        return None
    raw_tag = candidate.get("assay_tag")
    if raw_tag is None or is_absent_raw_value(raw_tag):
        return None
    tag = " ".join(str(raw_tag).strip().split())
    return tag or None


_LITERAL_VOLUME_FACT_TYPES = frozenset(
    {
        "pcr_reaction_volume",
        "template_dna_volume",
        "second_pcr_reaction_volume",
        "second_pcr_template_dna_volume",
    }
)


def _normalize_volume_text_for_literal_check(value: str) -> str:
    return " ".join(
        value.replace("\u2009", " ")
        .replace("\u202f", " ")
        .replace("µ", "u")
        .replace("μ", "u")
        .split()
    ).casefold()


_PRIMER_SEQUENCE_FACT_TYPES = frozenset({"forward_primer_sequence", "reverse_primer_sequence"})
# IUPAC nucleotide codes (standard + degenerate bases), the only characters
# a real primer sequence is ever reported in.
_NUCLEOTIDE_SEQUENCE_RE = re.compile(r"^[ACGTURYSWKMBDHVN]{6,}$", re.IGNORECASE)
_PCR_ASSAY_FACT_TYPES = frozenset(
    {
        "target_gene",
        "target_subfragment",
        "forward_primer_sequence",
        "reverse_primer_sequence",
        "forward_primer_name",
        "reverse_primer_name",
        "amplicon_size",
        "annealing_temperature",
        "pcr_cycle_count",
        "commercial_master_mix",
        "custom_master_mix",
    }
)
_NON_SEQUENCING_QC_PCR_CONTEXT_RE = re.compile(
    r"\b(?:"
    r"absence\s+of\s+bands?|lack\s+of\s+(?:bacterial\s+)?growth|confirmation\s+of\s+a\s+sterile\s+state|"
    r"confirm(?:ed|ation)?\s+(?:of\s+)?steril(?:e|ity)|sterility\s+(?:check|test|confirmation)|"
    r"checked\s+for\s+steril(?:e|ity)"
    r")\b",
    re.IGNORECASE,
)


def _looks_like_nucleotide_sequence(value: str) -> bool:
    return bool(_NUCLEOTIDE_SEQUENCE_RE.match(value.strip()))


def _candidate_value_is_supported_by_quote(fact_type: str, raw_value: object, quote: str) -> bool:
    if fact_type in _PCR_ASSAY_FACT_TYPES and _NON_SEQUENCING_QC_PCR_CONTEXT_RE.search(quote):
        return False
    if fact_type in _PRIMER_SEQUENCE_FACT_TYPES:
        # A real bug found live (10.1002/ece3.6071): when a paper only
        # states a primer's NAME in the main text (its actual sequence
        # lives in a supplementary table this pass never sees), the model
        # substituted the name ("1389F", "mlCOIintF") for the sequence
        # field instead of omitting it -- both are literally present in
        # the quote, so a plain verbatim check wouldn't have caught this;
        # the value itself needs to actually look like a sequence.
        if not _looks_like_nucleotide_sequence(str(raw_value)):
            return False
    if fact_type not in _LITERAL_VOLUME_FACT_TYPES:
        return True
    return _normalize_volume_text_for_literal_check(str(raw_value)) in _normalize_volume_text_for_literal_check(quote)


def _facts_from_candidates(
    candidates,
    segment_lookup: dict[str, str],
    section_title: str,
    seen: set[tuple[str, str, str, str]],
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
        if not _candidate_value_is_supported_by_quote(str(fact_type), raw_value, quote):
            continue
        assay_tag = _candidate_assay_tag(candidate, str(fact_type))
        dedupe_key = (str(fact_type), str(raw_value), quote, assay_tag or "")
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        hints = candidate.get("candidate_standard_fields")
        confidence_metadata = {"evidence_ids": evidence_ids}
        if isinstance(hints, dict) and hints:
            confidence_metadata["candidate_standard_fields"] = hints
        entity_level = EntityLevel.ASSAY if assay_tag else EntityLevel.STUDY
        facts.append(
            RawFactCandidate(
                entity_level=entity_level,
                fact_type_candidate=str(fact_type),
                raw_field_name=str(fact_type),
                raw_value=str(raw_value),
                source_locator=f"llm_text_extraction.{section_title}.{'|'.join(evidence_ids)}",
                support_type=SupportType.EXPLICIT,
                evidence_quote=quote,
                confidence_metadata=confidence_metadata,
                entity_external_id=assay_tag,
                entity_label=assay_tag,
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
