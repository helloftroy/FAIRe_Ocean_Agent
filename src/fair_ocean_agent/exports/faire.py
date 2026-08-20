"""FAIRe checklist export: CSVs for the populated classes
(`projectMetadata`, `sampleMetadata`, `experimentRunMetadata`), matching
the sheet names and column order of the vendored
`FAIRe_checklist_v1.0.2_FULLtemplate.xlsx` where this pipeline has data.

`ampData`, `stdData`, `eLowQuantData`, `taxaRaw`, and `taxaFinal` are
omitted from exports for now: no source adapter or extraction step in this
pipeline currently produces amplification/standard-curve/taxonomic-
assignment records, so writing them would only create empty files.
`experimentRunMetadata` is populated from sample/assay-specific
experiment_run (library) entities. Sequencing runs remain separate linked
entities, so many library rows may correctly share one `seq_run_id`.

Alongside the per-class data files, `export_faire` also writes
`field_reference.csv` -- one row per FAIRe field, every column that
appears in any of the data files, with its requirement level and
`exact_mappings` (real cross-standard URIs, e.g. MIxS, that came for free
with the vendored FAIRe schema -- see standards/faire_registry.py, built
for Milestone 6b). This is schema-level reference data, not per-study
data, so it isn't squeezed into the data CSVs themselves (which must match
FULLtemplate.xlsx's exact column layout) -- it's a companion data
dictionary instead, built from the same `build_faire_registry()` the
standards registry uses rather than a second, possibly-drifting copy of
the same information.
"""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from fair_ocean_agent.config import REPO_ROOT
from fair_ocean_agent.database.enums import (
    SHAREABLE_ENTITY_LEVELS,
    EntityLevel,
    EntityRelationshipType,
    EntityRootStatus,
    IdentifierType,
    ReviewStatus,
)
from fair_ocean_agent.database.models import (
    ApiPaperCorrection,
    Entity,
    EntityRelationship,
    EntityStudy,
    ExternalIdentifier,
    RawFact,
    StandardizedValue,
    Study,
)
from fair_ocean_agent.extraction.section_categories import SECTION_CATEGORIES
from fair_ocean_agent.mapping.faire import TARGET_SCHEMA, clean_assay_name_value, resolve_project_id
from fair_ocean_agent.mapping.primer_library import (
    INHERITED_SEQUENCE_FIELD,
    PRIMER_NAME_TO_SEQUENCE_FIELD,
    corpus_primer_sequence,
    study_primer_name,
)
from fair_ocean_agent.standards.faire_registry import build_faire_registry

FAIRE_SCHEMA_DIR = REPO_ROOT / "schemas" / "faire"

OMITTED_EMPTY_CLASSES = ("ampData", "stdData", "eLowQuantData", "taxaRaw", "taxaFinal")
# Backward-compatible alias for callers/tests that need to know which FAIRe
# classes are intentionally omitted until the pipeline has row-shaped data.
EMPTY_CLASSES = OMITTED_EMPTY_CLASSES

# Pipeline-internal traceability column, not a real FAIRe field --
# deliberately NOT added to schemas/faire/classes.yaml (this must never be
# mistaken for something submittable to NOAA/GBIF/OBIS as part of the real
# checklist) and deliberately excluded from field_reference.csv
# (which documents real FAIRe fields with real requirement levels/
# exact_mappings -- this column has neither). export_faire() merges every
# study in the database into one shared set of output CSVs with no
# per-study filter; this is the only thing that lets an outside observer
# trace which project/sample/experiment rows belong to the same study once
# more than one is being processed at once.
INTERNAL_STUDY_ID_FIELD = "internal_study_id"

# Same non-FAIRe, internal-only precedent as INTERNAL_STUDY_ID_FIELD above --
# not in schemas/faire/classes.yaml, excluded from field_reference.csv.
# Carries a canonical sample's alias entities' own native names (e.g. a
# supplement table's own "GC04_1") once identity/sample_alias_reconciliation.py
# has folded them into the canonical (accessioned) entity's row, so that
# native name isn't lost even though it no longer gets its own row.
INTERNAL_ALIAS_SAMPLE_IDS_FIELD = "internal_alias_sample_ids"

# Unlike INTERNAL_STUDY_ID_FIELD/INTERNAL_ALIAS_SAMPLE_IDS_FIELD above, this
# one DOES carry real extracted data (mapping/rules.py has a genuine
# MappingRule for it, target_table "sampleMetadata") -- it's just not a real
# FAIRe checklist field (not in schemas/faire/classes.yaml, excluded from
# field_reference.csv the same way), so it needs the same manual
# header-inclusion treatment as those internal columns to show up in
# sampleMetadata.csv at all. Bundles ~18 individually low-yield
# physicochemical fields (temp, salinity, ph, nitrate, chlorophyll, ...)
# into one pipe-joined "name/formula: value unit" column, per an explicit
# user request; those 18 fields' own columns are hidden via
# SAMPLE_METADATA_SUPPRESSED_FIELDS below.
CUSTOM_ENV_VAR_BLOCK_FIELD = "x_env_var_block"

# Same custom-non-FAIRe-field treatment as CUSTOM_ENV_VAR_BLOCK_FIELD, and the
# same STUDY-level MappingRule shape -- a dedicated second pass over
# CUSTOM_ENV_VAR_BLOCK_FIELD's own quotes (extraction/section_category_
# extraction.py::extract_pulled_env_var_facts) that keeps only genuine
# "name = value" measured pairs, per an explicit user request to keep
# x_env_var_block exactly as-is and add this as a separate, filtered
# companion column rather than replacing it.
CUSTOM_PULLED_ENV_VAR_FIELD = "x_pulled_env_var"

# Same custom-non-FAIRe-field treatment as CUSTOM_ENV_VAR_BLOCK_FIELD above,
# for projectMetadata this time. Per an explicit user request: every real
# column header found in a structured (CSV/TSV/XLSX/XLS) supplement table
# gets captured verbatim, pipe-joined, in one study-wide field -- a cheap
# diagnostic so a human can see what a supplement table actually contains
# even when most/all of its columns aren't in SUPPLEMENT_COLUMN_ALIASES yet
# (sources/supplement_parsing.py::_facts_from_rows is the producer).
CUSTOM_SPREADSHEET_HEADERS_FIELD = "spreadsheet_headers"

# samp_name/materialSampleID are the row's own identifying columns -- the
# real accession must win outright as the identifier, never get pipe-joined
# against an alias's native name (that native name is preserved separately,
# in INTERNAL_ALIAS_SAMPLE_IDS_FIELD).
_ALIAS_MERGE_EXCLUDED_FIELDS = frozenset({"samp_name", "materialSampleID"})

# For most sampleMetadata fields, an entity's own structured value
# (_entity_values, e.g. a real NCBI BioSample attribute) is deliberately
# left to win outright over a study-wide paper-text broadcast -- a real
# per-sample measurement should never be second-guessed by an LLM's
# broader claim. samp_mat_process is the one deliberate exception: FAIRe
# defines it as a free-text processing narrative (this pipeline's own
# CategoryTerm sets allows_multi_sentence=True for it), and a real
# BioSample's own samp_mat_process attribute is routinely a terse,
# boilerplate submitter note (e.g. "DNA extraction from sediment
# samples") that says far less than what the paper's own Methods
# describes -- confirmed live (10.3389/fmicb.2024.1295149). Rather than
# the structured value silently winning and discarding the richer paper
# text (mapping/faire.py's own pipe-union already merges multiple paper
# paragraphs' worth of samp_mat_process text into one broadcast value),
# both are kept side by side here.
_BROADCAST_ENTITY_PIPE_JOIN_FIELDS = frozenset({"samp_mat_process"})

# extraction/section_categories.py's per-category "<name>_0_1" detection
# facts (e.g. "sample_prep_0_1") -- a diagnostic coverage
# signal for tuning that module's keyword lists against real papers, never
# registered as real FAIRe checklist terms (not in schemas/faire/
# classes.yaml), same internal-only precedent as INTERNAL_STUDY_ID_FIELD.
# Always "0" or "1" (never blank) so the column reads as a real boolean
# even when the underlying RawFact is absent.
INTERNAL_SECTION_DETECTION_FIELDS = tuple(f"{category.name}_0_1" for category in SECTION_CATEGORIES)


def _section_category_detection_values(session: Session, study_id: str) -> dict[str, str]:
    detected = set(
        session.scalars(
            select(RawFact.fact_type_candidate).where(
                RawFact.study_id == study_id,
                RawFact.entity_id.is_(None),
                RawFact.fact_type_candidate.in_(INTERNAL_SECTION_DETECTION_FIELDS),
            )
        )
    )
    return {field: ("1" if field in detected else "0") for field in INTERNAL_SECTION_DETECTION_FIELDS}


# mapping/primer_library.py's corpus-wide primer name -> sequence lookup --
# a primer whose name we found in the text but whose actual sequence
# couldn't be pinned down, either from this paper's own extraction or from
# any other paper in the corpus that names the same primer, per an
# explicit user request to track these as future targeted-reference-crawl
# candidates (extraction/publication_metadata.py::
# extract_primer_reference_citations + workflow/handlers.py::
# handle_discover_primer_reference_study already do this crawl
# automatically when a real citation DOI is found; this flag surfaces the
# cases still left unresolved even after that). Same internal-only,
# always-"0"-or-"1" precedent as INTERNAL_SECTION_DETECTION_FIELDS above,
# not a real FAIRe checklist term. Computed live at export time (not its
# own persisted RawFact) so it can never go stale relative to the corpus.
INTERNAL_PRIMER_TRACEABILITY_FIELDS = ("primer_forward_source_unresolved", "primer_reverse_source_unresolved")
_SEQUENCE_FIELD_TO_TRACEABILITY_FLAG = {
    "pcr_primer_forward": "primer_forward_source_unresolved",
    "pcr_primer_reverse": "primer_reverse_source_unresolved",
}
# Per an explicit user request: this flag means "no sequence AND no
# reference to chase it down either" -- a real dead end, not merely "the
# sequence itself isn't known yet". A study whose paper names a primer
# without its sequence but DOES cite where it came from (real DOI, or a
# fallback title/citation text when the reference has no DOI -- see
# extract_primer_reference_citations) has a genuine lead recorded in
# pcr_primer_reference_forward/reverse, even before that lead is actually
# chased down to a sequence, so it should NOT be flagged unresolved.
_SEQUENCE_FIELD_TO_REFERENCE_FIELD = {
    "pcr_primer_forward": "pcr_primer_reference_forward",
    "pcr_primer_reverse": "pcr_primer_reference_reverse",
}


def _primer_traceability_values(session: Session, study_id: str) -> dict[str, str]:
    flags = {field: "0" for field in INTERNAL_PRIMER_TRACEABILITY_FIELDS}
    for name_field, sequence_field in PRIMER_NAME_TO_SEQUENCE_FIELD.items():
        primer_name = study_primer_name(session, study_id, name_field)
        if not primer_name:
            continue
        has_own_sequence = (
            session.scalars(
                select(RawFact.fact_id).where(
                    RawFact.study_id == study_id,
                    RawFact.fact_type_candidate.in_((sequence_field, INHERITED_SEQUENCE_FIELD[sequence_field])),
                    RawFact.review_status != ReviewStatus.REJECTED.value,
                )
            ).first()
            is not None
        )
        if has_own_sequence or corpus_primer_sequence(session, primer_name, sequence_field) is not None:
            continue
        has_reference = (
            session.scalars(
                select(RawFact.fact_id).where(
                    RawFact.study_id == study_id,
                    RawFact.fact_type_candidate == _SEQUENCE_FIELD_TO_REFERENCE_FIELD[sequence_field],
                    RawFact.review_status != ReviewStatus.REJECTED.value,
                )
            ).first()
            is not None
        )
        if has_reference:
            continue
        flags[_SEQUENCE_FIELD_TO_TRACEABILITY_FLAG[sequence_field]] = "1"
    return flags


@lru_cache(maxsize=1)
def _load_classes() -> dict:
    with (FAIRE_SCHEMA_DIR / "classes.yaml").open() as f:
        data = yaml.safe_load(f)
    return data["classes"]


def class_columns(class_name: str) -> list[str]:
    return list(_load_classes()[class_name]["slots"])


# Real, non-mandatory FAIRe projectMetadata fields the user explicitly
# asked to never populate and never show as a column at all -- an
# export-time suppression, deliberately NOT removed from schemas/faire/
# classes.yaml/schema.yaml: these are real upstream FAIRe terms, and
# editing the vendored schema mirror to drop a real term would corrupt its
# own fidelity to the actual FAIRe v1.0.2 checklist, a line this project
# has been careful never to cross. (Contrast a genuinely non-upstream
# local addition, safe to remove from the schema mirror outright -- e.g.
# expedition_id/ship_crs_expocode, a former NOAA/SEUS-MBON extension
# retracted per a later explicit user request; nothing here reintroduces
# that pattern.) None of these are `required: true` in the real schema,
# so suppressing them doesn't hide a genuinely mandatory field.
PROJECT_METADATA_SUPPRESSED_FIELDS = frozenset(
    {
        # Re-confirmed directly against the real codebase (a later audit
        # pass, per an explicit user request): none of these ten have any
        # extraction or mapping code populating them either -- suppressing
        # the column costs nothing beyond what was already true.
        "institutionID",
        "parent_project_id",
        "project_name",
        "recordedByID",
        "mod_date",
        "dataGeneralizations",
        "sop_bioinformatics",
        "assay_validation",
        "nucl_acid_amp",
        "sequencing_location",
        # The following removed entirely per an explicit user request
        # ("negligible... don't want to waste compute on them or clutter
        # the code with them") -- no extraction/mapping code populates
        # any of these, same suppression idiom as the fields above.
        "pcr_primer_conc_forward",
        "pcr_primer_conc_reverse",
        "amplificationReactionVolume",
        "pcr_dna_vol",
        "pcr2_thermocycler",
        "thermocycler",
        "pcr2_amplificationReactionVolume",
        "pcr2_dna_vol",
        "pcr2_plate_id",
        "lib_screen",
        "pcr_cond",
        "pcr2_custom_mm",
        "pcr2_commercial_mm",
        "trim_method",
        "trim_param",
        "pcr2_cond",
        "demux_tool",
        "demux_max_mismatch",
        "tax_class_collapse",
        "tax_class_id_cutoff",
        "tax_class_other",
        "tax_class_query_cutoff",
        "otu_clust_cutoff",
        "error_rate_cutoff",
        "error_rate_tool",
        "error_rate_type",
        "min_reads_cutoff",
        "min_reads_tool",
        "min_reads_cutoff_unit",
        "merge_tool",
        "merge_min_overlap",
        "min_len_cutoff",
        "min_len_tool",
        "otu_raw_description",
        "otu_final_description",
        "bioinfo_method_additional",
        "chimera_check_method",
        "chimera_check_param",
        "screen_geograph_method",
        "screen_nontarget_method",
        "screen_other",
        "concentration_method",
        "concentration_unit",
        # Removed entirely per an explicit user request ("i forgot to ask
        # for this removal") -- no longer extracted, mapped, or exported
        # anywhere.
        "tax_assign_cat",
        # Removed per explicit user request: these qPCR/PCR bookkeeping
        # fields are low-value for the current workflow and should not
        # consume LLM work or appear in exported tables.
        "pcr2_analysis_software",
        "pcr_analysis_software",
        "automaticBaselineValue",
        "automaticThresholdQuantificationCycle",
        "baselineValue",
        "pcr_assay_lod_LL",
        "pcr_assay_lod_UL",
        "pcr_assay_lod_techreps",
        "pcr_assay_loq_LL",
        "pcr_assay_loq_UL",
        "pcr_assay_loq_techreps",
        "std_seq",
        "std_type",
    }
)


def _exportable_project_columns() -> list[str]:
    return [field for field in class_columns("projectMetadata") if field not in PROJECT_METADATA_SUPPRESSED_FIELDS]


# Real, non-mandatory-in-practice FAIRe sampleMetadata fields the user
# explicitly asked to never populate and never show as a column at all --
# same export-time suppression idiom as PROJECT_METADATA_SUPPRESSED_FIELDS
# above, deliberately NOT removed from schemas/faire/classes.yaml/
# schema.yaml (a real upstream FAIRe term list, not something this project
# edits). Unlike that project-level set, three of these
# (neg_cont_type/pos_cont_type/detected_notDetected) ARE marked
# `required: true` in the vendored schema mirror with no recorded
# conditional-requirement text -- none of them ever had any real
# extraction code populating them regardless (confirmed directly against
# the real codebase before removal), so suppressing the column doesn't
# lose any data that was ever actually being captured, but it does mean
# the exported CSV no longer carries a column FAIRe's own checklist marks
# mandatory. Per an explicit, repeated user request. (A later audit pass
# found prepped_samp_store_temp/dur/sol still carried a dangling
# MappingRule with no extraction source ever feeding it -- dead code that
# could never actually fire; removed from mapping/rules.py.)
SAMPLE_METADATA_SUPPRESSED_FIELDS = frozenset(
    {
        "nucl_acid_ext",
        "nucl_acid_ext_modify",
        "date_ext",
        "ratioOfAbsorbance260_280",
        "prepped_samp_store_temp",
        "prepped_samp_store_dur",
        "prepped_samp_store_sol",
        "dna_store_loc",
        "size_frac_low",
        "neg_cont_type",
        "pos_cont_type",
        "rel_cont_id",
        "detected_notDetected",
        "nitro",
        "org_carb",
        "org_nitro",
        "tot_org_c_meth",
        "tot_nitro_cont_meth",
        "tot_nitro_content",
        "nitro_unit",
        "tot_inorg_nitro",
        "diss_oxygen",
        "concentration_method",
        "concentration_unit",
        # The following removed entirely per an explicit user request
        # ("negligible... don't want to waste compute on them or clutter
        # the code with them") -- no extraction/mapping code populates
        # any of these, same suppression idiom as the fields above.
        "samp_weather",
        "humidity",
        "alt",
        "solar_irradiance",
        "precip_chem_prep",
        "precip_force_prep",
        "precip_temp_prep",
        "precip_time_prep",
        "light_intensity",
        # The following 18 physicochemical fields are suppressed per an
        # explicit user request: their combined coverage now lives in one
        # column instead, CUSTOM_ENV_VAR_BLOCK_FIELD below (extraction/
        # section_categories.py's x_env_var_block CategoryTerm, sample_prep
        # category). Legacy standalone diss_oxygen is now suppressed too:
        # these variables are extracted through the bundled env-var route.
        "diss_inorg_carb",
        "diss_inorg_nitro",
        "diss_org_carb",
        "diss_org_nitro",
        "nitrate",
        "nitrite",
        "org_matter",
        "part_org_carb",
        "part_org_nitro",
        "ph",
        "suspend_part_matter",
        "tot_carb",
        "tot_diss_nitro",
        "tot_nitro",
        "tot_org_carb",
        "tot_part_carb",
        "chlorophyll",
        "temp",
        "salinity",
    }
)


def _exportable_sample_columns() -> list[str]:
    return [field for field in class_columns("sampleMetadata") if field not in SAMPLE_METADATA_SUPPRESSED_FIELDS]


# Same export-time suppression idiom as PROJECT_METADATA_SUPPRESSED_FIELDS/
# SAMPLE_METADATA_SUPPRESSED_FIELDS above, for experimentRunMetadata.
# otu_num_tax_assigned/output_otu_num/output_read_count are dropped
# entirely per an explicit user judgment call: these are computed
# differently study to study (no single extraction/derivation approach
# would be correct across papers), so guessing at one now isn't worth it --
# revisit once this pipeline computes them itself from its own
# bioinformatics outputs, rather than trying to extract them from a
# paper's prose. mid_forward/mid_reverse and lib_conc/lib_conc_meth/
# lib_conc_unit are dropped as a low-priority scope cut, both per an
# explicit, repeated user request.
EXPERIMENT_RUN_METADATA_SUPPRESSED_FIELDS = frozenset(
    {
        "otu_num_tax_assigned",
        "output_otu_num",
        "output_read_count",
        "mid_forward",
        "mid_reverse",
        "lib_conc",
        "lib_conc_meth",
        "lib_conc_unit",
    }
)


def _exportable_experiment_columns() -> list[str]:
    return [
        field
        for field in class_columns("experimentRunMetadata")
        if field not in EXPERIMENT_RUN_METADATA_SUPPRESSED_FIELDS
    ]


def _study_wide_values(session: Session, study_id: str) -> dict[str, str]:
    """entity_id IS NULL standardized_values for a study: projectMetadata
    fields proper, plus sample-scoped LLM facts broadcast as a default
    (see mapping/faire.py's docstring)."""
    rows = session.execute(
        select(StandardizedValue.target_field, StandardizedValue.standardized_value).where(
            StandardizedValue.study_id == study_id,
            StandardizedValue.target_schema == TARGET_SCHEMA,
            StandardizedValue.entity_id.is_(None),
        )
    ).all()
    return {field: value for field, value in rows if value is not None}


def _linked_study_ids(session: Session, entity_id: str) -> list[str]:
    """Every Study this entity is linked to, home or shared (see EntityStudy's
    docstring in database/models.py) -- for SHAREABLE_ENTITY_LEVELS
    (sample/experiment_run/sequencing_run), this can be more than one when a
    second paper reuses the same real BioSample/run another study already
    created. Sorted for a deterministic pipe-joined internal_study_id
    column."""
    return sorted(session.scalars(select(EntityStudy.study_id).where(EntityStudy.entity_id == entity_id)))


def _entity_broadcast_is_authoritative(entity: Entity, study: Study) -> bool:
    """A shared entity's broadcast-style (study-wide LLM/text) facts only
    fill this entity's blanks from the study identity/root_determination.py
    determined to be its root -- never from a non-root linked study, and
    never while root determination is still pending/ambiguous ("unknown is
    preferable to guessing", same stance as identity/consistency.py).
    Subsumes the pre-root-determination "len(linked_study_ids) == 1" gate
    exactly for the non-shared case: root_status is eagerly DETERMINED/self
    at entity creation (identity/entity_linking.py::create_entity), so an
    entity that was never shared always satisfies this."""
    return entity.root_status == EntityRootStatus.DETERMINED.value and entity.root_study_id == study.study_id


def _entity_values(session: Session, entity_id: str) -> dict[str, str]:
    rows = session.execute(
        select(StandardizedValue.target_field, StandardizedValue.standardized_value).where(
            StandardizedValue.entity_id == entity_id,
            StandardizedValue.target_schema == TARGET_SCHEMA,
        )
    ).all()
    return {field: value for field, value in rows if value is not None}


def _linked_entity(
    session: Session,
    from_entity_id: str,
    relationship_type: EntityRelationshipType,
) -> Entity | None:
    entities = list(
        session.scalars(
            select(Entity)
            .join(EntityRelationship, EntityRelationship.to_entity_id == Entity.entity_id)
            .where(
                EntityRelationship.from_entity_id == from_entity_id,
                EntityRelationship.relationship_type == relationship_type.value,
            )
            .order_by(Entity.external_identifier, Entity.entity_id)
            .limit(2)
        )
    )
    if len(entities) > 1:
        raise ValueError(
            f"experiment entity {from_entity_id} has multiple {relationship_type.value} links; "
            "cannot emit an unambiguous FAIRe experimentRunMetadata row"
        )
    return entities[0] if entities else None


def _alias_entities(session: Session, canonical_entity_id: str) -> list[Entity]:
    """Reverse of _linked_entity's direction: every alias entity that
    identity/sample_alias_reconciliation.py has folded INTO this canonical
    entity. Deliberately doesn't reuse _linked_entity's "raise if >1"
    guard -- more than one supplement-derived alias legitimately folding
    into one real accession is the expected case here, unlike
    DERIVED_FROM_SAMPLE/USES_ASSAY/SEQUENCED_IN_RUN's 1:1 semantics."""
    return list(
        session.scalars(
            select(Entity)
            .join(EntityRelationship, EntityRelationship.from_entity_id == Entity.entity_id)
            .where(
                EntityRelationship.to_entity_id == canonical_entity_id,
                EntityRelationship.relationship_type == EntityRelationshipType.SAME_PHYSICAL_SAMPLE_AS.value,
            )
            .order_by(Entity.external_identifier, Entity.entity_id)
        )
    )


def _merge_field_values(
    primary: dict[str, str], secondary: dict[str, str], *, exclude: frozenset[str] = frozenset()
) -> dict[str, str]:
    """Folds `secondary` into `primary`: a field present on only one side
    passes through unchanged; identical values on both sides collapse to
    one; genuinely differing non-null values get pipe-joined (primary's
    value first) per the user's own explicit request ("if there are
    conflicts we should list both with a pipe"). Fields in `exclude` are
    copied from `primary` only, never touched by `secondary`."""
    merged = dict(primary)
    for field, value in secondary.items():
        if field in exclude:
            continue
        if field not in merged or not merged[field]:
            merged[field] = value
        elif value and value != merged[field]:
            existing_values = merged[field].split("|")
            if value not in existing_values:
                merged[field] = "|".join([*existing_values, value])
    return merged


def _pipe_join_unique(values: list[str]) -> str:
    seen: set[str] = set()
    merged: list[str] = []
    for value in values:
        for part in str(value).split("|"):
            part = part.strip()
            key = part.casefold()
            if not part or key in seen:
                continue
            seen.add(key)
            merged.append(part)
    return "|".join(merged)


def _linked_study_values(session: Session, study_ids: list[str], field: str) -> str:
    value = _pipe_join_unique([_study_wide_values(session, study_id).get(field, "") for study_id in study_ids])
    if field == "assay_name":
        return clean_assay_name_value(value) or ""
    return value


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})
    return len(rows)


def _write_field_reference(output_dir: Path, columns_by_class: dict[str, list[str]]) -> int:
    field_to_classes: dict[str, list[str]] = {}
    for class_name, columns in columns_by_class.items():
        for field in columns:
            field_to_classes.setdefault(field, []).append(class_name)

    terms_by_field = {t["upstream_field_name"]: t for t in build_faire_registry()}
    rows = []
    for field, classes in sorted(field_to_classes.items()):
        term = terms_by_field.get(field, {})
        rows.append(
            {
                "faire_field": field,
                "faire_classes": "|".join(classes),
                "requirement_level_code": term.get("requirement_level_code", ""),
                "requirement_level_condition": term.get("requirement_level_condition") or "",
                "range": term.get("range", ""),
                "exact_mappings": "|".join(term.get("exact_mappings") or []),
            }
        )
    return _write_csv(
        output_dir / "field_reference.csv",
        ["faire_field", "faire_classes", "requirement_level_code", "requirement_level_condition", "range", "exact_mappings"],
        rows,
    )


def _paper_reference(session: Session, study_id: str) -> str:
    doi = session.scalar(
        select(ExternalIdentifier.identifier_value).where(
            ExternalIdentifier.study_id == study_id,
            ExternalIdentifier.identifier_type == IdentifierType.DOI.value,
        )
    )
    return doi or study_id


def _write_api_paper_corrections(session: Session, output_dir: Path) -> int:
    """The durable, code-populated "fixes" spreadsheet an explicit user
    request asked for -- every row here was written by a verification
    mechanism (e.g. extraction/api_verification.py) that found a
    structured API value contradicted by the paper's own text, never
    hand-curated. One row per correction across every study, so this file
    accumulates over time as more papers get processed."""
    corrections = list(session.scalars(select(ApiPaperCorrection).order_by(ApiPaperCorrection.created_at)))
    rows = [
        {
            "paper_reference": _paper_reference(session, correction.study_id),
            "internal_study_id": correction.study_id,
            "api_faire_term": correction.api_faire_term,
            "api_value": correction.api_value,
            "corrected_faire_term": correction.corrected_faire_term,
            "corrected_value": correction.corrected_value,
            "supporting_quote": correction.supporting_quote,
            "detector": correction.detector,
        }
        for correction in corrections
    ]
    return _write_csv(
        output_dir / "api_paper_corrections.csv",
        [
            "paper_reference",
            "internal_study_id",
            "api_faire_term",
            "api_value",
            "corrected_faire_term",
            "corrected_value",
            "supporting_quote",
            "detector",
        ],
        rows,
    )


def export_faire(session: Session, output_dir: str | Path) -> dict[str, int]:
    output_dir = Path(output_dir)
    counts: dict[str, int] = {}
    for class_name in OMITTED_EMPTY_CLASSES:
        stale_path = output_dir / f"{class_name}.csv"
        if stale_path.exists():
            stale_path.unlink()

    studies = list(session.scalars(select(Study)))

    project_columns = _exportable_project_columns()
    project_rows = []
    for study in studies:
        # Analysis-only paper: links to shared samples/runs (EntityStudy)
        # but is home to none of them -- did no original data collection of
        # its own, just reanalysis. Still exists in the network (Study row,
        # EntityStudy links, sampleMetadata/experimentRunMetadata rows for
        # anything it DOES home -- none, by this same condition), but never
        # gets a projectMetadata row. Evaluated unconditionally (not gated
        # on entity_component_status): this is a structural fact about
        # entity ownership, not a cross-study priority judgment like root
        # determination, so it's correct to check at any time.
        homed_entity_exists = (
            session.query(Entity.entity_id)
            .filter(
                Entity.study_id == study.study_id,
                Entity.entity_level.in_([level.value for level in SHAREABLE_ENTITY_LEVELS]),
            )
            .first()
            is not None
        )
        linked_entity_exists = (
            session.query(EntityStudy.entity_study_id).filter(EntityStudy.study_id == study.study_id).first()
            is not None
        )
        if linked_entity_exists and not homed_entity_exists:
            continue

        broadcast = _study_wide_values(session, study.study_id)
        project_id = resolve_project_id(session, study.study_id)
        # Real FAIRe's own projectMetadata export layout is one row per
        # assay_name, not one global row per study (see
        # schemas/faire/README.md) -- a paper describing more than one
        # assay tags each assay's facts with a real ASSAY Entity
        # (extraction/text.py's assay_tag), and mapping/faire.py gives each
        # its own StandardizedValue rows (entity_id set, not broadcast).
        #
        # Gated on assay entities that have at least one direct
        # StandardizedValue, not merely on the entity existing:
        # extraction/experiment_runs.py's materialize_legacy_experiment_runs
        # already creates ASSAY entities today purely as USES_ASSAY link
        # targets for structured ENA/BioProject assay_name facts -- those
        # facts land on the EXPERIMENT_RUN entity's row, never the assay
        # entity itself, so such an assay entity has zero StandardizedValue
        # rows of its own. Gating on mere existence would make every such
        # structured-only multi-assay study suddenly emit one
        # near-duplicate broadcast row per assay where it previously
        # emitted exactly one.
        assay_entities = list(
            session.scalars(
                select(Entity).where(
                    Entity.study_id == study.study_id, Entity.entity_level == EntityLevel.ASSAY.value
                )
            )
        )
        assays_with_values = [assay for assay in assay_entities if _entity_values(session, assay.entity_id)]
        section_detection = _section_category_detection_values(session, study.study_id)
        primer_traceability = _primer_traceability_values(session, study.study_id)
        if not assays_with_values:
            if project_id is None and not broadcast:
                continue  # nothing at all mapped for this study -- don't emit an all-blank row
            row = dict(broadcast)
            row["project_id"] = project_id or ""
            row[INTERNAL_STUDY_ID_FIELD] = study.study_id
            row.update(section_detection)
            row.update(primer_traceability)
            project_rows.append(row)
            continue
        for assay in assays_with_values:
            row = dict(broadcast)
            row.update(_entity_values(session, assay.entity_id))
            row.setdefault("assay_name", assay.external_identifier or assay.label or assay.entity_id)
            row["project_id"] = project_id or ""
            row[INTERNAL_STUDY_ID_FIELD] = study.study_id
            row.update(section_detection)
            row.update(primer_traceability)
            project_rows.append(row)
    counts["projectMetadata"] = _write_csv(
        output_dir / "projectMetadata.csv",
        [
            INTERNAL_STUDY_ID_FIELD,
            *INTERNAL_SECTION_DETECTION_FIELDS,
            *INTERNAL_PRIMER_TRACEABILITY_FIELDS,
            *project_columns,
            CUSTOM_SPREADSHEET_HEADERS_FIELD,
        ],
        project_rows,
    )

    sample_columns = _exportable_sample_columns()
    sample_rows = []
    for study in studies:
        # entities are keyed by home study_id (Entity.study_id, unchanged),
        # so a shared entity is still visited exactly once here, under its
        # home study -- no double-counting despite an entity now possibly
        # being linked to more than one Study.
        sample_entities = session.scalars(
            select(Entity).where(Entity.study_id == study.study_id, Entity.entity_level == EntityLevel.SAMPLE.value)
        )
        for entity in sample_entities:
            # A supplement-derived alias entity (identity/
            # sample_alias_reconciliation.py) never emits its own row --
            # its values are folded into its canonical (accessioned)
            # entity's row below instead.
            if _linked_entity(session, entity.entity_id, EntityRelationshipType.SAME_PHYSICAL_SAMPLE_AS) is not None:
                continue

            linked_study_ids = set(_linked_study_ids(session, entity.entity_id))
            # A shared sample's row must not carry any non-root study's
            # paper-specific interpretive broadcast defaults -- only the
            # root study's (identity/root_determination.py), or its own
            # entity-level facts, which are unconditionally safe regardless
            # of how many studies link to it.
            broadcast = _study_wide_values(session, study.study_id) if _entity_broadcast_is_authoritative(entity, study) else {}
            entity_values = _entity_values(session, entity.entity_id)
            row = dict(broadcast)
            row.update(entity_values)
            for field in _BROADCAST_ENTITY_PIPE_JOIN_FIELDS:
                broadcast_value = broadcast.get(field)
                entity_value = entity_values.get(field)
                if broadcast_value and entity_value and broadcast_value != entity_value:
                    row[field] = _pipe_join_unique([entity_value, broadcast_value])
            row["samp_name"] = entity.external_identifier or entity.entity_id

            alias_sample_ids = []
            for alias in _alias_entities(session, entity.entity_id):
                alias_broadcast = (
                    _study_wide_values(session, alias.study_id)
                    if _entity_broadcast_is_authoritative(alias, alias.study)
                    else {}
                )
                alias_row = dict(alias_broadcast)
                alias_row.update(_entity_values(session, alias.entity_id))
                row = _merge_field_values(row, alias_row, exclude=_ALIAS_MERGE_EXCLUDED_FIELDS)
                alias_sample_ids.append(alias.external_identifier or alias.entity_id)
                linked_study_ids.update(_linked_study_ids(session, alias.entity_id))
            if alias_sample_ids:
                row[INTERNAL_ALIAS_SAMPLE_IDS_FIELD] = "|".join(alias_sample_ids)

            row[INTERNAL_STUDY_ID_FIELD] = "|".join(sorted(linked_study_ids))
            sample_rows.append(row)
    counts["sampleMetadata"] = _write_csv(
        output_dir / "sampleMetadata.csv",
        [
            INTERNAL_STUDY_ID_FIELD,
            INTERNAL_ALIAS_SAMPLE_IDS_FIELD,
            *sample_columns,
            CUSTOM_ENV_VAR_BLOCK_FIELD,
            CUSTOM_PULLED_ENV_VAR_FIELD,
        ],
        sample_rows,
    )

    experiment_columns = _exportable_experiment_columns()
    experiment_rows = []
    for study in studies:
        experiment_entities = session.scalars(
            select(Entity).where(
                Entity.study_id == study.study_id,
                Entity.entity_level == EntityLevel.EXPERIMENT_RUN.value,
            )
        )
        for entity in experiment_entities:
            linked_study_ids = _linked_study_ids(session, entity.entity_id)
            broadcast = _study_wide_values(session, study.study_id) if _entity_broadcast_is_authoritative(entity, study) else {}
            row = dict(broadcast)
            row.update(_entity_values(session, entity.entity_id))
            sample = _linked_entity(
                session,
                entity.entity_id,
                EntityRelationshipType.DERIVED_FROM_SAMPLE,
            )
            assay = _linked_entity(
                session,
                entity.entity_id,
                EntityRelationshipType.USES_ASSAY,
            )
            sequencing_run = _linked_entity(
                session,
                entity.entity_id,
                EntityRelationshipType.SEQUENCED_IN_RUN,
            )
            if sample is not None:
                # Defends against a run's DERIVED_FROM_SAMPLE link pointing
                # at a supplement-derived alias entity (identity/
                # sample_alias_reconciliation.py) that no longer emits its
                # own sampleMetadata row -- prefer the real accessioned
                # entity so samp_name always resolves to a row that exists.
                canonical_sample = (
                    _linked_entity(session, sample.entity_id, EntityRelationshipType.SAME_PHYSICAL_SAMPLE_AS)
                    or sample
                )
                row.setdefault("samp_name", canonical_sample.external_identifier or canonical_sample.entity_id)
            if assay is not None:
                row.setdefault("assay_name", assay.external_identifier or assay.label or assay.entity_id)
            if not row.get("assay_name"):
                linked_assay_name = _linked_study_values(session, linked_study_ids, "assay_name")
                if linked_assay_name:
                    row["assay_name"] = linked_assay_name
            if sequencing_run is not None:
                row.setdefault("seq_run_id", sequencing_run.external_identifier or sequencing_run.entity_id)
            if entity.external_identifier and not entity.external_identifier.startswith("internal:"):
                row.setdefault("lib_id", entity.external_identifier)
            row[INTERNAL_STUDY_ID_FIELD] = "|".join(linked_study_ids)
            experiment_rows.append(row)
    counts["experimentRunMetadata"] = _write_csv(
        output_dir / "experimentRunMetadata.csv",
        [INTERNAL_STUDY_ID_FIELD, *experiment_columns],
        experiment_rows,
    )

    columns_by_class = {
        "projectMetadata": project_columns,
        "sampleMetadata": sample_columns,
        "experimentRunMetadata": experiment_columns,
    }
    counts["field_reference"] = _write_field_reference(output_dir, columns_by_class)
    counts["api_paper_corrections"] = _write_api_paper_corrections(session, output_dir)

    return counts
