"""Applies mapping/rules.py's RULES table against a study's raw_facts to
produce StandardizedValue (+ StandardizedValueEvidence) rows targeting
FAIRe (target_schema="faire").

Two design decisions worth stating explicitly:

1. **entity_id resolution is table-shaped, not fact-shaped.**
   `projectMetadata` is a study-wide singleton table in FAIRe -- it doesn't
   have a per-sample or per-run row -- so every projectMetadata-targeted
   StandardizedValue gets `entity_id=None` regardless of which entity the
   source RawFact was attached to (e.g. `instrument_platform` facts live on
   `sequencing_run` entities in this pipeline, but 500 identical per-run
   facts must collapse to one project-wide `platform` value, not 500 rows).
   Similarly, a study-level LLM-extracted fact (`DNA_extraction_method`,
   `storage_conditions`, ...) mapped to a `sampleMetadata` field has no
   sample-specific entity_id to inherit -- `entity_id=None` there means
   "applies to every sample row as a broadcast default", which
   `exports/faire.py` interprets accordingly.

2. **`sample_accession` -> `materialSampleID` is a targeted redirect, not a
   broadcast.** A run's BioSample accession belongs to exactly one sample,
   so this rule looks up the `sample` Entity whose `external_identifier`
   matches the fact's value and attaches the StandardizedValue there --
   never `entity_id=None` (which would incorrectly broadcast one sample's
   accession onto every sample in the study).

Because multiple raw_facts (up to 500 sequencing-run facts per study) can
target the same study-wide (entity_id=None) field, this module de-
duplicates by (target_table, target_field, entity_id): the first value
wins; if a later fact disagrees, the existing row is flagged
`review_required=True` rather than silently overwritten or duplicated --
see `_upsert_study_wide` below.
"""
from __future__ import annotations

import re

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from fair_ocean_agent.database.enums import (
    EntityLevel,
    EntityRootStatus,
    MappingMethod,
    MissingnessStatus,
    ReviewStatus,
    SupportType,
)
from fair_ocean_agent.database.models import (
    Entity,
    EntityStudy,
    ExternalIdentifier,
    RawFact,
    StandardizedValue,
    StandardizedValueEvidence,
)
from fair_ocean_agent.extraction.experiment_runs import materialize_legacy_experiment_runs
from fair_ocean_agent.extraction.section_categories import SECTION_CATEGORIES
from fair_ocean_agent.extraction.publication_metadata import sync_recorded_by_from_biosample_or_first_author
from fair_ocean_agent.extraction.taxonomic_assay import sync_assay_target_taxa_from_biosample_organisms
from fair_ocean_agent.identity.sample_alias_reconciliation import reconcile_sample_aliases
from fair_ocean_agent.mapping import vocabularies
from fair_ocean_agent.mapping.rules import MappingRule, rules_for

TARGET_SCHEMA = "faire"
TARGET_SCHEMA_VERSION = "1.0.2"


def _clear_existing_faire_mappings(session: Session, study_id: str) -> None:
    existing_ids = list(
        session.scalars(
            select(StandardizedValue.standardized_value_id).where(
                StandardizedValue.study_id == study_id,
                StandardizedValue.target_schema == TARGET_SCHEMA,
            )
        )
    )
    if not existing_ids:
        return
    session.execute(
        delete(StandardizedValueEvidence).where(
            StandardizedValueEvidence.standardized_value_id.in_(existing_ids)
        )
    )
    session.execute(
        delete(StandardizedValue).where(StandardizedValue.standardized_value_id.in_(existing_ids))
    )
    session.flush()


def _find_sample_entity_by_external_id(session: Session, study_id: str, external_id: str) -> Entity | None:
    return session.scalar(
        select(Entity).where(
            Entity.study_id == study_id,
            Entity.entity_level == EntityLevel.SAMPLE.value,
            Entity.external_identifier == external_id,
        )
    )


def _resolve_entity_id(session: Session, study_id: str, fact: RawFact, rule: MappingRule) -> str | None:
    if rule.target_field == "materialSampleID":
        sample = _find_sample_entity_by_external_id(session, study_id, fact.raw_value)
        return sample.entity_id if sample else None
    if rule.target_table == "projectMetadata":
        # A fact tagged with a real assay (extraction/text.py's assay_tag,
        # EntityLevel.ASSAY) gets its own per-assay row instead of
        # broadcasting -- this is the one case where projectMetadata is not
        # a study-wide singleton, matching real FAIRe's own export layout
        # (one projectMetadata row per assay_name, see
        # schemas/faire/README.md). Every other projectMetadata-targeted
        # fact (ENA's instrument_platform/instrument_model at
        # SEQUENCING_RUN, OBIS/GBIF's PROJECT-level facts, an untagged
        # single-assay STUDY-level fact, ...) keeps broadcasting via
        # entity_id=None exactly as before.
        if fact.entity_level == EntityLevel.ASSAY.value:
            return fact.entity_id
        return None
    if fact.entity_level in (
        EntityLevel.SAMPLE.value,
        EntityLevel.EXPERIMENT_RUN.value,
        EntityLevel.SEQUENCING_RUN.value,
    ):
        return fact.entity_id
    return None  # study-wide fact mapped onto a sample-scoped field: broadcast default


# targetTaxonomicAssay/targetTaxonomicScope: never "first wins, flag review,
# drop the rest" -- every distinct value across every contributing fact is
# kept, pipe-joined, per an explicit user request. targetTaxonomicAssay
# additionally prioritizes its structured ("API": extraction/
# taxonomic_assay.py's BioSample-organism aggregation) signal ahead of its
# LLM-judged-search signal without ever discarding the LLM's own values --
# a paper's stated assay target can be more specific than any one
# BioSample's own `organism` field.
# samp_mat_process joins the same way for a different reason: it's a
# free-text, potentially multi-sentence processing narrative (allows_
# multi_sentence=True in extraction/section_categories.py), and a real
# paper's Methods commonly describes distinct processing steps in more
# than one separate paragraph (e.g. a storage/freeze-drying paragraph and
# a separate DNA-extraction paragraph) -- each one, independently
# classified and extracted as its own sample_prep run, should contribute
# its own verbatim quote rather than the first one winning and the rest
# being discarded, per an explicit user request.
# x_env_var_block is the same shape as samp_mat_process for the same
# reason: it bundles ~18 physicochemical variables (allows_multi_sentence=
# True too), and a paper can report different environmental measurements
# in more than one paragraph or in a separate supplement table -- every
# contributing quote should show up in the pipe-joined value, not just
# the first one found.
# spreadsheet_headers unions for a related but simpler reason: a study can
# have more than one structured supplement file (e.g. one sample-metadata
# table, one separate environmental-data table), and each file's own
# header row should show up, not just the first file's.
_PIPE_UNION_TARGET_FIELDS = frozenset(
    {
        "targetTaxonomicAssay", "targetTaxonomicScope", "platform", "instrument", "samp_mat_process", "otu_db",
        "x_env_var_block", "spreadsheet_headers",
    }
)
_PIPE_UNION_REVIEW_ON_MULTIPLE_FIELDS = frozenset({"platform", "instrument"})
_API_SUPPORT_TYPES = frozenset({SupportType.STRUCTURED_SOURCE.value, SupportType.DETERMINISTICALLY_DERIVED.value})
_NOT_FOUND_VALUE = "not found"
_COLLAPSED_SAMPLE_UNIT_FIELDS = {
    "concentration": "concentration_unit",
    "diss_inorg_carb": "diss_inorg_carb_unit",
    "diss_inorg_nitro": "diss_inorg_nitro_unit",
    "diss_org_carb": "diss_org_carb_unit",
    "diss_org_nitro": "diss_org_nitro_unit",
    "diss_oxygen": "diss_oxygen_unit",
    "nitrate": "nitrate_unit",
    "nitrite": "nitrite_unit",
    "org_matter": "org_matter_unit",
    "part_org_carb": "part_org_carb_unit",
    "part_org_nitro": "part_org_nitro_unit",
    "tot_carb": "tot_carb_unit",
    "tot_diss_nitro": "tot_diss_nitro_unit",
    "tot_inorg_nitro": "tot_inorg_nitro_unit",
    "tot_nitro": "tot_nitro_unit",
    "tot_org_carb": "tot_org_carb_unit",
    "tot_part_carb": "tot_part_carb_unit",
}


def _split_pipe_values(value: str) -> list[str]:
    return [part.strip() for part in value.split("|") if part.strip()]


def _pipe_union(*groups: list[str]) -> str:
    """Dedups (case-insensitive) and joins with ' | ', processing `groups`
    in the order given -- callers pass higher-priority values first. A
    literal "not found" placeholder (extraction/search_flags.py's
    not-found fallback for a targeted search field) is dropped as soon as
    any real value is present anywhere in the union, but survives alone
    when nothing else was ever found for this field."""
    seen: set[str] = set()
    merged: list[str] = []
    for group in groups:
        for value in group:
            key = value.casefold()
            if key in seen:
                continue
            seen.add(key)
            merged.append(value)
    if len(merged) > 1:
        merged = [value for value in merged if value.casefold() != _NOT_FOUND_VALUE]
    return " | ".join(merged)


def _collapsed_unit_lookup(facts: list[RawFact]) -> dict[tuple[str, str | None], str]:
    """First accepted raw unit fact by (unit field, entity).

    The unit fields are no longer FAIRe output columns, but structured
    sources may still report them separately from the numeric value. Keep
    that information by appending the unit to the value field during mapping.
    """
    unit_fields = set(_COLLAPSED_SAMPLE_UNIT_FIELDS.values())
    lookup: dict[tuple[str, str | None], str] = {}
    for fact in facts:
        if fact.fact_type_candidate not in unit_fields or fact.raw_value is None:
            continue
        unit = str(fact.raw_value).strip()
        if not unit:
            continue
        lookup.setdefault((fact.fact_type_candidate, fact.entity_id), unit)
    return lookup


def _append_collapsed_unit(
    value: str,
    rule: MappingRule,
    source_fact: RawFact,
    unit_lookup: dict[tuple[str, str | None], str],
) -> str:
    if rule.target_table != "sampleMetadata":
        return value
    unit_field = _COLLAPSED_SAMPLE_UNIT_FIELDS.get(rule.target_field)
    if not unit_field:
        return value
    unit = unit_lookup.get((unit_field, source_fact.entity_id)) or unit_lookup.get((unit_field, None))
    if not unit or unit.casefold() in value.casefold():
        return value
    return f"{value} {unit}"


def _lib_layout_from_fastq_facts(facts: list[RawFact]) -> tuple[str, bool, list[RawFact]]:
    """Derive layout from verified-accessible FASTQ entries when the source
    has run the accessibility check.

    One `fastq_ftp` entry means single-end; two or more semicolon-separated
    entries means paired-end. If no accepted FASTQ file facts exist, report
    "no files" instead of trusting prose or ENA/SRA's declared layout.
    """
    access_status_by_entity = {
        fact.entity_id: str(fact.raw_value).strip().casefold()
        for fact in facts
        if fact.fact_type_candidate == "fastq_access_status"
        and fact.entity_level == EntityLevel.SEQUENCING_RUN.value
        and fact.entity_id is not None
        and fact.raw_value is not None
    }
    require_accessible = bool(access_status_by_entity)
    file_counts: list[int] = []
    evidence: list[RawFact] = []
    for fact in facts:
        if (
            fact.fact_type_candidate != "fastq_ftp"
            or fact.entity_level != EntityLevel.SEQUENCING_RUN.value
            or fact.raw_value is None
        ):
            continue
        count = len(_split_semicolon_nonempty(str(fact.raw_value)))
        if count < 1:
            continue
        if require_accessible and access_status_by_entity.get(fact.entity_id) != "accessible":
            continue
        file_counts.append(count)
        evidence.append(fact)
    if not file_counts:
        return "no files", False, []
    has_paired = any(count >= 2 for count in file_counts)
    has_single = any(count == 1 for count in file_counts)
    return ("paired end" if has_paired else "single end"), bool(has_paired and has_single), evidence


def _split_semicolon_nonempty(value: str) -> list[str]:
    return [part.strip() for part in value.split(";") if part.strip()]


# A real live audit (10.7717/peerj.333, STUDY-9b31d2733994) found a second
# (indexing) PCR whose forward/reverse primers are the study's own PCR1
# primers with a 454-Titanium sequencing adapter fused on -- e.g. the
# study's forward_primer_sequence facts contain BOTH "TCTCAAAGACTAAGCCATGC"
# (the clean PCR1 primer, SP-F-30) and
# "CCTATCCCCTGTGTGCCTTGGCAGTCTCAGTCTCAAAGACTAAGCCATGC" (the PCR2 fusion
# oligo), and the paper's own text confirms the longer one's tail "matches
# SP-F-30 primer". Crucially, the paper never once uses the word "adapter"
# (it says "454-Titanium primers" instead), so search_flags.py's own
# keyword/quote-anchored adapter_forward/adapter_reverse mechanism never
# fires here. Detected by substring instead: whenever the same primer
# field yields two distinct sequences and the shorter is an exact
# substring of the longer, the longer one is a fusion (adapter + primer),
# not a second distinct primer -- the leftover portion is the adapter.
#
# The two sequences can legitimately land in different entity scopes: this
# same real study's PCR1 facts broadcast (entity_id=None) while its PCR2
# facts carry a model-assigned assay_tag ("18S-V3V4" -- the model's own
# mistake, tagging the second PCR of the SAME assay as if it were a
# distinct one), giving them their own ASSAY-entity StandardizedValue row
# per real FAIRe's one-row-per-assay projectMetadata layout (see
# _resolve_entity_id's docstring above). The fusion pair is therefore
# detected globally across every scope, but the derived override is
# applied separately to EVERY entity_id that actually contributed the long
# (fused) value -- broadcast and/or any assay row -- so whichever row(s)
# the export ultimately emits, none of them keep showing the raw fusion
# oligo as if it were just "the primer".
_MIN_CLEAN_PRIMER_LENGTH = 15  # shorter risks spurious substring coincidences
_MIN_ADAPTER_TAIL_LENGTH = 4  # a couple of stray bases left over isn't a real adapter tag
_NUCLEOTIDE_ONLY_RE = re.compile(r"^[ACGTURYSWKMBDHVN]+$", re.IGNORECASE)
_FUSED_PRIMER_ADAPTER_DIRECTIONS = (
    ("forward_primer_sequence", "pcr_primer_forward", "adapter_forward"),
    ("reverse_primer_sequence", "pcr_primer_reverse", "adapter_reverse"),
)


def _normalize_sequence_value(value: str) -> str:
    return re.sub(r"\s+", "", value).upper()


def _projectmetadata_entity_id_for_fact(fact: RawFact) -> str | None:
    """Mirrors _resolve_entity_id's own projectMetadata rule: a fact tagged
    onto a real ASSAY entity gets its own per-assay row; every other
    projectMetadata-targeted fact broadcasts (entity_id=None)."""
    return fact.entity_id if fact.entity_level == EntityLevel.ASSAY.value else None


def _derive_fused_primer_adapter_values(
    facts: list[RawFact],
) -> dict[tuple[str, str | None], tuple[str, bool, list[RawFact]]]:
    results: dict[tuple[str, str | None], tuple[str, bool, list[RawFact]]] = {}
    for source_native_name, primer_target, adapter_target in _FUSED_PRIMER_ADAPTER_DIRECTIONS:
        value_to_facts: dict[str, list[RawFact]] = {}
        for fact in facts:
            if fact.fact_type_candidate != source_native_name or not fact.raw_value:
                continue
            if fact.entity_level not in (EntityLevel.STUDY.value, EntityLevel.ASSAY.value):
                continue
            for part in _split_pipe_values(fact.raw_value):
                normalized = _normalize_sequence_value(part)
                if normalized and _NUCLEOTIDE_ONLY_RE.fullmatch(normalized):
                    value_to_facts.setdefault(normalized, []).append(fact)
        if len(value_to_facts) < 2:
            continue

        sorted_values = sorted(value_to_facts, key=len)
        fusion: tuple[str, str] | None = None
        for i, short in enumerate(sorted_values):
            if len(short) < _MIN_CLEAN_PRIMER_LENGTH:
                continue
            for longer in sorted_values[i + 1 :]:
                if short in longer:
                    fusion = (short, longer)
                    break
            if fusion:
                break
        if fusion is None:
            continue

        short, longer = fusion
        index = longer.index(short)
        adapter_seq = longer[len(short) :] if index == 0 else longer[:index]
        if len(adapter_seq) < _MIN_ADAPTER_TAIL_LENGTH:
            continue

        entity_ids_with_longer = {
            _projectmetadata_entity_id_for_fact(fact) for fact in value_to_facts[longer]
        }
        for entity_id in entity_ids_with_longer:
            results[(primer_target, entity_id)] = (short, False, value_to_facts[short])
            results[(adapter_target, entity_id)] = (adapter_seq, False, value_to_facts[longer])
    return results


_GENE_SHORT_NAME_RE = re.compile(r"(\d+)\s*S\b", re.IGNORECASE)
_DASH_NORMALIZE_RE = re.compile(r"[‐-―]")
_MOJIBAKE_DASH_RE = re.compile(r"(?:\u7ab6\u5929|\u7ab6\u96fb|\u7ab6\u642d|\u7ab6\u7763|\u7ab6\u6d9b)")
_RRNA_REGION_ASSAY_NAME_RE = re.compile(
    r"^(?P<gene>1[68]\s*S)(?:\s*rRNA)?\s*[- ]\s*(?P<region>V\d(?:-V\d)?)$",
    re.IGNORECASE,
)
_BARE_FUNCTIONAL_GENE_ASSAY_NAME_RE = re.compile(
    r"^(?:hzs[ABC]?|hzo(?:F1|R1)?|narG|nir[SK]|amoA|nxr[AB]|nosZ|nifH|dsr[AB]|mcrA)$",
    re.IGNORECASE,
)


def _normalize_assay_name_part(value: str) -> str:
    part = _MOJIBAKE_DASH_RE.sub("-", value.strip())
    part = _DASH_NORMALIZE_RE.sub("-", part)
    part = re.sub(r"\s*-\s*", "-", part)
    part = re.sub(r"\s+", " ", part)
    marker_region = _RRNA_REGION_ASSAY_NAME_RE.fullmatch(part)
    if marker_region:
        gene = re.sub(r"\s+", "", marker_region.group("gene")).upper()
        return f"{gene}-{marker_region.group('region').upper()}"
    return part


def clean_assay_name_value(value: str) -> str | None:
    """Drop primer-pair strings and bare functional-gene targets from assay_name.

    Those values are useful elsewhere (`pcr_primer_*`, `target_gene`, qPCR
    fields) but they make experimentRunMetadata.assay_name unreadable when a
    study-wide assay_name is inherited by sequencing runs.
    """
    cleaned: list[str] = []
    seen: set[str] = set()
    for part in _split_pipe_values(value):
        part = _normalize_assay_name_part(part)
        if "/" in part:
            continue
        if _BARE_FUNCTIONAL_GENE_ASSAY_NAME_RE.fullmatch(part.strip()):
            continue
        key = part.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(part)
    return " | ".join(cleaned) if cleaned else None


def _compose_assay_name(target_gene: str | None, target_subfragment: str | None) -> str | None:
    """A stand-in assay_name ("16S-V4") for when a paper never states a
    published assay name, composed deterministically from target_gene/
    target_subfragment rather than relying on a separate LLM-judged field
    (extraction/search_flags.py's own assay_name field already tries this
    same "generate a stable concise name... such as 16S-V4" via prompt
    instruction, but that mechanism's candidate-quote gate requires
    gene+region to appear adjacently in one real sentence -- a real paper
    audit found a case where the gene and region were reported separately
    enough that no candidate quote ever reached the LLM, leaving
    assay_name blank even though both target_gene and target_subfragment
    were independently resolved)."""
    if not target_gene:
        return None
    gene_match = _GENE_SHORT_NAME_RE.search(target_gene)
    gene_short = f"{gene_match.group(1)}S" if gene_match else target_gene.strip().split()[0]
    if not target_subfragment:
        return gene_short
    region = _DASH_NORMALIZE_RE.sub("-", target_subfragment.strip())
    return f"{gene_short}-{region}"


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_SEDIMENT_KEYWORD_RE = re.compile(r"\b(sediment|soil)s?\b", re.IGNORECASE)
_WATER_KEYWORD_RE = re.compile(r"\bwater\b", re.IGNORECASE)

# Study-level LLM-extracted facts where a single field can genuinely mean
# different things for different sample types in the same paper -- a real
# gap found live (10.3389/fmicb.2024.1295149 states "10 L" water-sample
# volumes AND, separately, "500 mg of dried sediment samples ... used for
# DNA extraction" for its one real sediment BioSample). Blindly broadcasting
# one winner (mapping/faire.py's normal "oldest fact wins" rule) silently
# wrote the water figure onto the sediment sample too. Every real
# extraction/section_categories.py sample_prep-category field is a
# candidate for this -- per an explicit user request, not just
# samp_vol_we_dna_ext -- plus the older broad-checklist mechanism's own
# sample_volume_for_extraction(_unit), kept for direct comparison.
_SAMPLE_TYPE_ROUTED_NATIVE_NAMES = (
    frozenset(
        term.native_name
        for category in SECTION_CATEGORIES
        if category.name == "sample_prep"
        for term in category.terms
    )
    | frozenset({"sample_volume_for_extraction", "sample_volume_for_extraction_unit"})
) - frozenset({"samp_category"})
# samp_category is excluded even though it lives in the sample_prep
# category: its own "type" axis is real-sample-vs-control, not
# water-vs-sediment, so this router's water/sediment detection
# (_detect_sample_type_from_quote) is the wrong tool for it -- see the
# CategoryTerm's own comment in extraction/section_categories.py for why
# it also deliberately has no MappingRule to broadcast through anyway.

# Real BioSample-API-derived attributes checked, in order, for a sample's
# own type -- isolation_source first (the most direct, canonical signal,
# e.g. "water"/"sediment"), falling back to env_medium then
# samp_mat_process (confirmed live: 10.3389/fmicb.2024.1295149's one real
# sediment BioSample says samp_mat_process="DNA extraction from sediment
# samples") when isolation_source itself is missing.
_SAMPLE_TYPE_ATTRIBUTE_FALLBACK_CHAIN = ("isolation_source", "env_medium", "samp_mat_process")


def _detect_sample_type_from_quote(raw_value: str, quote: str | None) -> str | None:
    """Deterministic, sentence-scoped: a multi-sentence evidence_quote can
    legitimately discuss both sample types in different sentences (e.g. one
    paragraph covering both water and sediment collection), so only the
    sentence that actually contains the extracted raw_value is checked --
    not the whole quote -- and only when that sentence unambiguously
    mentions exactly one of the two keywords."""
    if not quote:
        return None
    for sentence in _SENTENCE_SPLIT_RE.split(quote):
        if raw_value.casefold() not in sentence.casefold():
            continue
        is_sediment = bool(_SEDIMENT_KEYWORD_RE.search(sentence))
        is_water = bool(_WATER_KEYWORD_RE.search(sentence))
        if is_sediment and not is_water:
            return "sediment"
        if is_water and not is_sediment:
            return "water"
    return None


def _sample_type_for_entity(session: Session, entity_id: str) -> str | None:
    for fact_type in _SAMPLE_TYPE_ATTRIBUTE_FALLBACK_CHAIN:
        value = session.scalar(
            select(RawFact.raw_value)
            .where(
                RawFact.entity_id == entity_id,
                RawFact.fact_type_candidate == fact_type,
                RawFact.review_status != ReviewStatus.REJECTED.value,
            )
            .order_by(RawFact.created_at)
            .limit(1)
        )
        if not value:
            continue
        if _SEDIMENT_KEYWORD_RE.search(value):
            return "sediment"
        if _WATER_KEYWORD_RE.search(value):
            return "water"
    return None


def _authoritative_sample_entities(session: Session, study_id: str) -> list[Entity]:
    """Every SAMPLE entity genuinely rooted at this study -- not merely
    linked here. Mirrors exports/faire.py's own
    _entity_broadcast_is_authoritative check exactly (root_status ==
    DETERMINED and root_study_id == this study), so a study-level,
    paper-text-derived sample_prep fact never gets broadcast onto a real
    BioSample that this paper only references/reuses from an older, already
    -published study -- confirmed live as a real risk (10.1038/s42003-024-
    06136-2 shares real BioSample accessions with an older, unrelated
    study). Duplicated here rather than imported from exports/faire.py to
    avoid a circular import (exports/faire.py already imports FROM this
    module)."""
    entity_ids = set(session.scalars(select(EntityStudy.entity_id).where(EntityStudy.study_id == study_id)))
    if not entity_ids:
        return []
    entities = session.scalars(
        select(Entity).where(Entity.entity_id.in_(entity_ids), Entity.entity_level == EntityLevel.SAMPLE.value)
    ).all()
    return [
        entity
        for entity in entities
        if entity.root_status == EntityRootStatus.DETERMINED.value and entity.root_study_id == study_id
    ]


def _detect_sample_type_routed_facts(
    session: Session, study_id: str
) -> dict[str, list[tuple[RawFact, str]]]:
    """Only fields where this study's own facts genuinely disagree AND at
    least two distinct sample types were detected across them are
    candidates for routing; otherwise a field is absent from the returned
    dict and falls through to the normal broadcast/oldest-wins path
    unchanged (case 1: only one sample type is actually described)."""
    routed: dict[str, list[tuple[RawFact, str]]] = {}
    for native_name in _SAMPLE_TYPE_ROUTED_NATIVE_NAMES:
        facts = list(
            session.scalars(
                select(RawFact)
                .where(
                    RawFact.study_id == study_id,
                    RawFact.entity_id.is_(None),
                    RawFact.fact_type_candidate == native_name,
                    RawFact.review_status != ReviewStatus.REJECTED.value,
                )
                .order_by(RawFact.created_at)
            )
        )
        tagged = [
            (fact, sample_type)
            for fact in facts
            if fact.raw_value is not None
            for sample_type in [_detect_sample_type_from_quote(fact.raw_value, fact.evidence_quote)]
            if sample_type is not None
        ]
        if len({sample_type for _fact, sample_type in tagged}) >= 2:
            routed[native_name] = tagged
    return routed


def _apply_biological_rep_from_relations(
    session: Session,
    study_id: str,
    seen: dict[tuple[str, str, str | None], StandardizedValue],
) -> int:
    """projectMetadata.biological_rep comes ONLY from this study's own
    sampleMetadata biological_rep_relation facts (each sample's real
    replicate-group membership, itself derived from structured-API/
    supplement data -- never the paper's text) -- per an explicit user
    request, the deterministic-text-regex and LLM-checklist mechanisms
    that used to compete for this field were removed entirely.

    Each distinct group's size is its replicate count; the study-level
    value is that single number, or a "min-max" range across groups when
    sizes vary (e.g. "2-4"), or "0" when no replicate group was
    detected at all -- this function is now the field's sole writer, so
    it always sets a value rather than only conditionally overriding one."""
    sample_entities = _authoritative_sample_entities(session, study_id)
    if not sample_entities:
        return 0
    sample_ids = {entity.entity_id for entity in sample_entities}
    relation_facts = session.scalars(
        select(RawFact)
        .where(
            RawFact.study_id == study_id,
            RawFact.entity_id.in_(sample_ids),
            RawFact.fact_type_candidate == "biological_rep_relation",
            RawFact.review_status != ReviewStatus.REJECTED.value,
        )
        .order_by(RawFact.created_at)
    ).all()

    group_sizes: dict[str, int] = {}
    evidence_fact_id_by_group: dict[str, str] = {}
    for fact in relation_facts:
        if fact.raw_value not in group_sizes:
            group_sizes[fact.raw_value] = len(fact.raw_value.split(" | "))
            evidence_fact_id_by_group[fact.raw_value] = fact.fact_id
    sizes = sorted(group_sizes.values())
    if not sizes:
        value = "0"
        evidence_fact_id = None
    else:
        value = str(sizes[0]) if sizes[0] == sizes[-1] else f"{sizes[0]}-{sizes[-1]}"
        evidence_fact_id = next(iter(evidence_fact_id_by_group.values()))

    key = ("projectMetadata", "biological_rep", None)
    existing = seen.get(key)
    if existing is not None:
        if existing.standardized_value != value:
            existing.standardized_value = value
            existing.mapping_method = MappingMethod.DETERMINISTIC_SYNONYM.value
            existing.review_required = False
            if evidence_fact_id is not None:
                session.add(
                    StandardizedValueEvidence(
                        standardized_value_id=existing.standardized_value_id,
                        fact_id=evidence_fact_id,
                    )
                )
        return 0

    standardized_value = StandardizedValue(
        study_id=study_id,
        entity_id=None,
        target_schema=TARGET_SCHEMA,
        target_schema_version=TARGET_SCHEMA_VERSION,
        target_field="biological_rep",
        standardized_value=value,
        mapping_method=MappingMethod.DETERMINISTIC_SYNONYM.value,
        review_required=False,
        missingness_status=MissingnessStatus.PRESENT.value,
    )
    session.add(standardized_value)
    session.flush()
    if evidence_fact_id is not None:
        session.add(
            StandardizedValueEvidence(
                standardized_value_id=standardized_value.standardized_value_id,
                fact_id=evidence_fact_id,
            )
        )
    seen[key] = standardized_value
    return 1


def _apply_sample_type_routed_facts(
    session: Session,
    study_id: str,
    routed_facts_by_field: dict[str, list[tuple[RawFact, str]]],
    seen: dict[tuple[str, str, str | None], StandardizedValue],
) -> int:
    """Case 2/3 per an explicit user specification:

    - Case 2 (BioSample API distinguishes sample type for at least one real
      sample): route each tagged value only to the SAMPLE entities whose own
      real type (_sample_type_for_entity) matches -- an entity with no
      determinable type, or a type that matches none of the tagged values,
      is left blank rather than guessed.
    - Case 3 (nothing in the BioSample API distinguishes any sample here):
      a "double pass" -- only treat this as a genuine, unresolvable multi-
      sample-type conflict worth flagging for review if at least one OTHER
      sample_prep field independently shows the same kind of conflict too
      (guards against one miscategorized quote in an otherwise single-
      sample-type paper). When corroborated, all distinct values are pipe-
      joined into one study-wide, review_required=True broadcast so a human
      can resolve it manually. When NOT corroborated (this is the only
      field showing any conflict at all), the routing attempt is abandoned
      and this field instead gets the same "oldest fact wins" broadcast it
      would have received had it never been flagged as routable.

    Case 1 (only one sample type described at all) needs no code here: such
    a field is never a key in `routed_facts_by_field` to begin with, and
    the study-wide broadcast it gets via the normal per-fact loop already
    only reaches authoritatively-rooted samples (exports/faire.py's own
    _entity_broadcast_is_authoritative check at export time).
    """
    created = 0
    sample_entities = _authoritative_sample_entities(session, study_id)
    entity_types = {entity.entity_id: _sample_type_for_entity(session, entity.entity_id) for entity in sample_entities}
    known_entity_types = {sample_type for sample_type in entity_types.values() if sample_type is not None}
    corroboration_count = len(routed_facts_by_field)

    for native_name, tagged_facts in routed_facts_by_field.items():
        rules = rules_for(native_name, EntityLevel.STUDY.value)
        if not rules:
            continue
        rule = rules[0]
        value_by_type: dict[str, str] = {}
        fact_by_type: dict[str, RawFact] = {}
        for fact, sample_type in tagged_facts:
            value = rule.transform(fact.raw_value)
            if rule.target_field == "assay_name" and value is not None:
                value = clean_assay_name_value(value)
            if value is None:
                continue
            value_by_type.setdefault(sample_type, value)
            fact_by_type.setdefault(sample_type, fact)
        if len(value_by_type) < 2:
            continue

        if known_entity_types & value_by_type.keys():
            # Case 2 (at least partial): route per matching entity.
            for entity in sample_entities:
                entity_sample_type = entity_types.get(entity.entity_id)
                if entity_sample_type is None or entity_sample_type not in value_by_type:
                    continue
                key = (rule.target_table, rule.target_field, entity.entity_id)
                if key in seen:
                    continue
                value = value_by_type[entity_sample_type]
                fact = fact_by_type[entity_sample_type]
                review_required = rule.review_required
                if rule.enum_name:
                    check = vocabularies.check_value(rule.enum_name, value)
                    if not check.is_valid:
                        review_required = True
                standardized_value = StandardizedValue(
                    study_id=study_id,
                    entity_id=entity.entity_id,
                    target_schema=TARGET_SCHEMA,
                    target_schema_version=TARGET_SCHEMA_VERSION,
                    target_field=rule.target_field,
                    standardized_value=value,
                    mapping_method=rule.mapping_method,
                    review_required=review_required,
                    missingness_status=MissingnessStatus.PRESENT.value,
                )
                session.add(standardized_value)
                session.flush()
                session.add(
                    StandardizedValueEvidence(
                        standardized_value_id=standardized_value.standardized_value_id, fact_id=fact.fact_id
                    )
                )
                seen[key] = standardized_value
                created += 1
            continue

        key = (rule.target_table, rule.target_field, None)
        if key in seen:
            continue
        if corroboration_count >= 2:
            # Case 3, corroborated: flag for review, every distinct value
            # pipe-joined so a human can resolve it manually per sample.
            distinct_values = list(dict.fromkeys(value_by_type.values()))
            standardized_value = StandardizedValue(
                study_id=study_id,
                entity_id=None,
                target_schema=TARGET_SCHEMA,
                target_schema_version=TARGET_SCHEMA_VERSION,
                target_field=rule.target_field,
                standardized_value=" | ".join(distinct_values),
                mapping_method=rule.mapping_method,
                review_required=True,
                missingness_status=MissingnessStatus.PRESENT.value,
            )
            session.add(standardized_value)
            session.flush()
            for fact in fact_by_type.values():
                session.add(
                    StandardizedValueEvidence(
                        standardized_value_id=standardized_value.standardized_value_id, fact_id=fact.fact_id
                    )
                )
            seen[key] = standardized_value
            created += 1
        else:
            # Not corroborated by any other sample_prep field -- insufficient
            # evidence this is a genuine multi-sample-type conflict rather
            # than one miscategorized quote. Fall back to the same "oldest
            # fact wins" broadcast this field would have received had it
            # never been flagged as routable.
            all_facts = list(
                session.scalars(
                    select(RawFact)
                    .where(
                        RawFact.study_id == study_id,
                        RawFact.entity_id.is_(None),
                        RawFact.fact_type_candidate == native_name,
                        RawFact.review_status != ReviewStatus.REJECTED.value,
                    )
                    .order_by(RawFact.created_at)
                )
            )
            if not all_facts:
                continue
            oldest = all_facts[0]
            value = rule.transform(oldest.raw_value)
            if value is None:
                continue
            review_required = rule.review_required
            if rule.enum_name:
                check = vocabularies.check_value(rule.enum_name, value)
                if not check.is_valid:
                    review_required = True
            standardized_value = StandardizedValue(
                study_id=study_id,
                entity_id=None,
                target_schema=TARGET_SCHEMA,
                target_schema_version=TARGET_SCHEMA_VERSION,
                target_field=rule.target_field,
                standardized_value=value,
                mapping_method=rule.mapping_method,
                review_required=review_required,
                missingness_status=MissingnessStatus.PRESENT.value,
            )
            session.add(standardized_value)
            session.flush()
            session.add(
                StandardizedValueEvidence(
                    standardized_value_id=standardized_value.standardized_value_id, fact_id=oldest.fact_id
                )
            )
            seen[key] = standardized_value
            created += 1
    return created


def _sync_checklist_version(session: Session, study_id: str) -> None:
    """`checkls_ver` is `required: true` in the real FAIRe schema (one of
    the handful of unconditionally-mandatory projectMetadata fields --
    see tests/unit/test_faire_completeness.py), yet had zero extraction
    coverage before this: nothing in this pipeline ever emits any fact
    for it, so it always showed up "missing" in completeness validation
    regardless of how complete a study's data actually was. Its real
    value is a pure constant, not something to extract from a source --
    this pipeline exports against exactly one FAIRe checklist version,
    TARGET_SCHEMA_VERSION, so `checkls_ver` is always that version,
    unconditionally, for every study. Idempotent (checks for an existing
    fact before inserting), matching this module's other pre-step sync
    functions."""
    existing = session.scalar(
        select(RawFact).where(
            RawFact.study_id == study_id,
            RawFact.entity_id.is_(None),
            RawFact.fact_type_candidate == "checkls_ver",
        )
    )
    if existing is not None:
        return
    session.add(
        RawFact(
            study_id=study_id,
            entity_id=None,
            source_id=None,
            source_locator="pipeline:target_schema_version",
            raw_field_name="checkls_ver",
            raw_value=TARGET_SCHEMA_VERSION,
            fact_type_candidate="checkls_ver",
            entity_level=EntityLevel.STUDY.value,
            support_type=SupportType.DETERMINISTICALLY_DERIVED.value,
            extraction_method="derived:target_schema_version_constant",
            review_status=ReviewStatus.ACCEPTED.value,
        )
    )


def map_study_to_faire(session: Session, study_id: str) -> int:
    """Idempotent: re-derives every FAIRe StandardizedValue for a study
    from scratch each time it's called (delete-then-recreate), so it's safe
    to call again after new raw_facts arrive or after a rules.py change."""
    materialize_legacy_experiment_runs(session, study_id)
    reconcile_sample_aliases(session, study_id)
    sync_assay_target_taxa_from_biosample_organisms(session, study_id)
    sync_recorded_by_from_biosample_or_first_author(session, study_id)
    _sync_checklist_version(session, study_id)
    # Computed before _clear_existing_faire_mappings purely to decide which
    # fields (if any) get sample-type-routed treatment below instead of the
    # generic broadcast loop -- reads raw_facts only, unaffected by clearing
    # standardized_values.
    routed_facts_by_field = _detect_sample_type_routed_facts(session, study_id)
    _clear_existing_faire_mappings(session, study_id)

    # REJECTED facts (quarantined -- e.g. extracted under a since-fixed
    # bug, or from a superseded model/prompt version) are excluded from
    # mapping entirely, not merely deprioritized: review_status only has
    # real effect here, at this one filter. `created_at` ASC makes "first
    # (oldest surviving, non-rejected) fact wins" explicit and deterministic
    # rather than relying on whatever order the DB happens to return rows
    # in absent an ORDER BY -- this preserves today's de facto behavior for
    # legitimate multi-source ties, it does not introduce a new preference.
    facts = list(
        session.scalars(
            select(RawFact)
            .where(RawFact.study_id == study_id, RawFact.review_status != ReviewStatus.REJECTED.value)
            .order_by(RawFact.created_at)
        )
    )
    collapsed_unit_lookup = _collapsed_unit_lookup(facts)
    created = 0
    seen: dict[tuple[str, str, str | None], StandardizedValue] = {}
    pipe_union_state: dict[tuple[str, str, str | None], dict[str, list[str] | bool]] = {}

    for fact in facts:
        if fact.fact_type_candidate is None or fact.raw_value is None:
            continue
        if fact.fact_type_candidate in routed_facts_by_field:
            continue  # handled by _apply_sample_type_routed_facts below instead
        for rule in rules_for(fact.fact_type_candidate, fact.entity_level):
            if rule.target_field == "materialSampleID":
                entity_id = _resolve_entity_id(session, study_id, fact, rule)
                if entity_id is None:
                    continue  # no matching sample entity -- don't fabricate one
            else:
                entity_id = _resolve_entity_id(session, study_id, fact, rule)

            value = rule.transform(fact.raw_value)
            if rule.target_field == "assay_name" and value is not None:
                value = clean_assay_name_value(value)
            if value is None:
                continue
            value = _append_collapsed_unit(value, rule, fact, collapsed_unit_lookup)

            review_required = rule.review_required
            if rule.enum_name:
                check = vocabularies.check_value(rule.enum_name, value)
                if not check.is_valid:
                    review_required = True

            key = (rule.target_table, rule.target_field, entity_id)

            if rule.target_field in _PIPE_UNION_TARGET_FIELDS:
                bucket = pipe_union_state.setdefault(key, {"api": [], "llm": [], "review_required": False})
                source_list = bucket["api"] if fact.support_type in _API_SUPPORT_TYPES else bucket["llm"]
                assert isinstance(source_list, list)
                for split_value in _split_pipe_values(value):
                    if split_value not in source_list:
                        source_list.append(split_value)
                api_values = bucket["api"]
                llm_values = bucket["llm"]
                assert isinstance(api_values, list)
                assert isinstance(llm_values, list)
                merged_value = _pipe_union(api_values, llm_values)
                merged_parts = _split_pipe_values(merged_value)
                bucket["review_required"] = bool(
                    bucket["review_required"]
                    or review_required
                    or (rule.target_field in _PIPE_UNION_REVIEW_ON_MULTIPLE_FIELDS and len(merged_parts) > 1)
                )
                existing = seen.get(key)
                if existing is not None:
                    if existing.standardized_value != merged_value:
                        existing.standardized_value = merged_value
                        existing.mapping_method = rule.mapping_method
                        session.add(
                            StandardizedValueEvidence(
                                standardized_value_id=existing.standardized_value_id,
                                fact_id=fact.fact_id,
                            )
                        )
                    existing.review_required = bool(bucket["review_required"])
                    continue
                standardized_value = StandardizedValue(
                    study_id=study_id,
                    entity_id=entity_id,
                    target_schema=TARGET_SCHEMA,
                    target_schema_version=TARGET_SCHEMA_VERSION,
                    target_field=rule.target_field,
                    standardized_value=merged_value,
                    mapping_method=rule.mapping_method,
                    review_required=bool(bucket["review_required"]),
                    missingness_status=MissingnessStatus.PRESENT.value,
                )
                session.add(standardized_value)
                session.flush()
                session.add(
                    StandardizedValueEvidence(
                        standardized_value_id=standardized_value.standardized_value_id,
                        fact_id=fact.fact_id,
                    )
                )
                seen[key] = standardized_value
                created += 1
                continue

            existing = seen.get(key)
            if existing is not None:
                if existing.standardized_value != value:
                    existing.review_required = True
                    session.add(
                        StandardizedValueEvidence(
                            standardized_value_id=existing.standardized_value_id,
                            fact_id=fact.fact_id,
                        )
                    )
                continue

            standardized_value = StandardizedValue(
                study_id=study_id,
                entity_id=entity_id,
                target_schema=TARGET_SCHEMA,
                target_schema_version=TARGET_SCHEMA_VERSION,
                target_field=rule.target_field,
                standardized_value=value,
                mapping_method=rule.mapping_method,
                review_required=review_required,
                missingness_status=MissingnessStatus.PRESENT.value,
            )
            session.add(standardized_value)
            session.flush()
            session.add(
                StandardizedValueEvidence(
                    standardized_value_id=standardized_value.standardized_value_id,
                    fact_id=fact.fact_id,
                )
            )
            seen[key] = standardized_value
            created += 1

    created += _apply_biological_rep_from_relations(session, study_id, seen)

    for entity in session.scalars(select(Entity).where(Entity.study_id == study_id)):
        if not entity.external_identifier:
            continue
        if entity.entity_level == EntityLevel.SAMPLE.value:
            targets = ("samp_name", "materialSampleID")
        elif (
            entity.entity_level == EntityLevel.EXPERIMENT_RUN.value
            and not entity.external_identifier.startswith("internal:")
        ):
            targets = ("lib_id",)
        else:
            continue
        for target_field in targets:
            key = ("sampleMetadata" if entity.entity_level == EntityLevel.SAMPLE.value else "experimentRunMetadata",
                   target_field, entity.entity_id)
            if key in seen:
                continue
            standardized_value = StandardizedValue(
                study_id=study_id,
                entity_id=entity.entity_id,
                target_schema=TARGET_SCHEMA,
                target_schema_version=TARGET_SCHEMA_VERSION,
                target_field=target_field,
                standardized_value=entity.external_identifier,
                mapping_method=MappingMethod.EXACT_IDENTIFIER.value,
                review_required=False,
                missingness_status=MissingnessStatus.PRESENT.value,
            )
            session.add(standardized_value)
            session.flush()
            seen[key] = standardized_value
            created += 1

    lib_layout_key = ("projectMetadata", "lib_layout", None)
    if lib_layout_key not in seen:
        lib_layout, review_required, evidence_facts = _lib_layout_from_fastq_facts(facts)
        standardized_value = StandardizedValue(
            study_id=study_id,
            entity_id=None,
            target_schema=TARGET_SCHEMA,
            target_schema_version=TARGET_SCHEMA_VERSION,
            target_field="lib_layout",
            standardized_value=lib_layout,
            mapping_method=MappingMethod.DETERMINISTIC_SYNONYM.value,
            review_required=review_required,
            missingness_status=MissingnessStatus.PRESENT.value,
        )
        session.add(standardized_value)
        session.flush()
        for fact in evidence_facts:
            session.add(
                StandardizedValueEvidence(
                    standardized_value_id=standardized_value.standardized_value_id,
                    fact_id=fact.fact_id,
                )
            )
        seen[lib_layout_key] = standardized_value
        created += 1

    # Overrides (not gap-fills) whatever the generic per-fact loop above
    # already picked for pcr_primer_forward/reverse and adapter_forward/
    # reverse when a genuine fusion pair is detected -- the generic loop's
    # "oldest wins" tie-break has no way to know that a longer sequence is
    # really the shorter one plus an adapter tag, so it can't be trusted
    # to already have picked the clean primer on its own.
    for (target_field, entity_id), (value, review_required, evidence_facts) in _derive_fused_primer_adapter_values(
        facts
    ).items():
        key = ("projectMetadata", target_field, entity_id)
        existing = seen.get(key)
        if existing is not None:
            existing.standardized_value = value
            existing.mapping_method = MappingMethod.DETERMINISTIC_SYNONYM.value
            existing.review_required = review_required
            existing.missingness_status = MissingnessStatus.PRESENT.value
            session.execute(
                delete(StandardizedValueEvidence).where(
                    StandardizedValueEvidence.standardized_value_id == existing.standardized_value_id
                )
            )
            for fact in evidence_facts:
                session.add(
                    StandardizedValueEvidence(
                        standardized_value_id=existing.standardized_value_id,
                        fact_id=fact.fact_id,
                    )
                )
        else:
            standardized_value = StandardizedValue(
                study_id=study_id,
                entity_id=entity_id,
                target_schema=TARGET_SCHEMA,
                target_schema_version=TARGET_SCHEMA_VERSION,
                target_field=target_field,
                standardized_value=value,
                mapping_method=MappingMethod.DETERMINISTIC_SYNONYM.value,
                review_required=review_required,
                missingness_status=MissingnessStatus.PRESENT.value,
            )
            session.add(standardized_value)
            session.flush()
            for fact in evidence_facts:
                session.add(
                    StandardizedValueEvidence(
                        standardized_value_id=standardized_value.standardized_value_id,
                        fact_id=fact.fact_id,
                    )
                )
            seen[key] = standardized_value
            created += 1

    master_mix_pairs = (("commercial_mm", "custom_mm"),)
    for commercial_field, custom_field in master_mix_pairs:
        commercial_key = ("projectMetadata", commercial_field, None)
        custom_key = ("projectMetadata", custom_field, None)
        commercial_present = commercial_key in seen
        custom_present = custom_key in seen
        if commercial_present == custom_present:
            continue
        missing_field, missing_key, other_field = (
            (custom_field, custom_key, commercial_field)
            if commercial_present
            else (commercial_field, commercial_key, custom_field)
        )
        standardized_value = StandardizedValue(
            study_id=study_id,
            entity_id=None,
            target_schema=TARGET_SCHEMA,
            target_schema_version=TARGET_SCHEMA_VERSION,
            target_field=missing_field,
            standardized_value=f"N/A see {other_field}",
            mapping_method=MappingMethod.DETERMINISTIC_SYNONYM.value,
            review_required=False,
            missingness_status=MissingnessStatus.PRESENT.value,
        )
        session.add(standardized_value)
        session.flush()
        seen[missing_key] = standardized_value
        created += 1

    # assay_name composition fallback -- see _compose_assay_name's own
    # docstring for why this is needed alongside the existing LLM-judged
    # assay_name field rather than replacing it: this only fires when
    # that field genuinely found nothing.
    assay_name_key = ("projectMetadata", "assay_name", None)
    if assay_name_key not in seen:
        target_gene_value = seen.get(("projectMetadata", "target_gene", None))
        target_subfragment_value = seen.get(("projectMetadata", "target_subfragment", None))
        composed = _compose_assay_name(
            target_gene_value.standardized_value if target_gene_value else None,
            target_subfragment_value.standardized_value if target_subfragment_value else None,
        )
        if composed:
            standardized_value = StandardizedValue(
                study_id=study_id,
                entity_id=None,
                target_schema=TARGET_SCHEMA,
                target_schema_version=TARGET_SCHEMA_VERSION,
                target_field="assay_name",
                standardized_value=composed,
                mapping_method=MappingMethod.DETERMINISTIC_SYNONYM.value,
                review_required=True,
                missingness_status=MissingnessStatus.PRESENT.value,
            )
            session.add(standardized_value)
            session.flush()
            seen[assay_name_key] = standardized_value
            created += 1

    # informationWithheld default: per an explicit user request, a study
    # whose Statements/Data-availability text never raised a genuine
    # withheld-information candidate (extraction/search_flags.py's own
    # LLMJudgedSearchField handles the real, verbatim case) should read
    # "Nothing indicated as withheld" rather than a blank cell
    # indistinguishable from "never checked". Done here, post-mapping,
    # rather than as a per-call fallback inside detect_llm_judged_search_
    # facts: informationWithheld isn't in _PIPE_UNION_TARGET_FIELDS, so
    # under the generic "oldest fact wins" conflict rule, a default written
    # by the paper-text pass would otherwise permanently block a real
    # value found later by a supplement-text pass. Firing only once every
    # source has already been mapped avoids that ordering trap entirely.
    information_withheld_key = ("projectMetadata", "informationWithheld", None)
    if information_withheld_key not in seen:
        standardized_value = StandardizedValue(
            study_id=study_id,
            entity_id=None,
            target_schema=TARGET_SCHEMA,
            target_schema_version=TARGET_SCHEMA_VERSION,
            target_field="informationWithheld",
            standardized_value="Nothing indicated as withheld",
            mapping_method=MappingMethod.DETERMINISTIC_SYNONYM.value,
            review_required=False,
            missingness_status=MissingnessStatus.PRESENT.value,
        )
        session.add(standardized_value)
        session.flush()
        seen[information_withheld_key] = standardized_value
        created += 1

    created += _apply_sample_type_routed_facts(session, study_id, routed_facts_by_field, seen)

    return created


def resolve_project_id(session: Session, study_id: str) -> str | None:
    """The FAIRe `project_id` join key. Not a mapped raw_fact -- it's read
    directly from ExternalIdentifier (an accession the study was resolved
    against, not something an extractor "found"), so it never gets a
    StandardizedValue/evidence row of its own. Preference order: a
    BioProject accession is the most FAIRe-idiomatic project identifier;
    fall back to other repository-level accessions, then DOI, so a study
    reached only through a publication still gets a usable project_id."""
    from fair_ocean_agent.database.enums import IdentifierType

    for identifier_type in (
        IdentifierType.BIOPROJECT_ACCESSION,
        IdentifierType.ENA_STUDY_ACCESSION,
        IdentifierType.SRA_STUDY_ACCESSION,
        IdentifierType.DOI,
    ):
        value = session.scalar(
            select(ExternalIdentifier.identifier_value).where(
                ExternalIdentifier.study_id == study_id,
                ExternalIdentifier.identifier_type == identifier_type.value,
            )
        )
        if value:
            return value
    return None
