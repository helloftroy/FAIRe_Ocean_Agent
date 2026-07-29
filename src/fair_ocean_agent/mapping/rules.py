"""The raw_fact -> FAIRe field rules table.

Each `MappingRule` says: a RawFact with this `fact_type_candidate` (and,
where relevant, this `entity_level`) becomes a StandardizedValue at this
FAIRe `target_table`/`target_field`. `mapping/faire.py` applies these rules
against a study's raw_facts; it does not contain per-field logic itself --
adding a new mapped field should only ever mean adding a rule here.

## What has full rule coverage (grounded in this pipeline's real,
## observed raw_facts vocabulary -- see the Milestone 6 section of
## README.md for the query that produced this list)

- Sample-level structured facts from NCBI BioSample / ENA -- all of these
  arrive through the exact same mechanism (`sources/ncbi.py`'s generic
  `Attributes/Attribute` passthrough: `fact_type_candidate` is literally
  whatever MIxS/INSDC attribute name a real BioSample XML record carries):
  `collection_date`, `depth`, `env_broad_scale`, `env_local_scale`,
  `env_medium`, `geo_loc_name`, `lat_lon`, `collection_method`,
  `elev`, `samp_collect_device`, `samp_size`, `samp_size_unit`, `temp`,
  `salinity`, `ph`, `diss_oxygen`. The last 8 only produce a
  StandardizedValue for a given sample if that sample's real BioSample
  record actually reported that attribute -- a rule existing here does not
  mean every sample gets a value, only that one is captured when the
  source has it.
- Sequencing-run-level structured facts from ENA: `instrument_platform`,
  `instrument_model`, `sample_accession`, `read_count` (-> FAIRe's
  `input_read_count`), `fastq_ftp`/`fastq_md5` (-> FAIRe's
  `filename`/`filename2`/`checksum_filename`/`checksum_filename2`, split on
  ENA's `;`-joined forward/reverse pairing), and (whenever `fastq_md5` is
  present) a deterministic `checksum_method` = "MD5" -- ENA's read_run
  report only ever gives MD5 checksums, never states the algorithm
  explicitly, so this is inferred rather than read verbatim, but no less
  deterministic for it. `library_layout` (added to `sources/ena.py`'s
  `RUN_FIELDS` alongside this) -> FAIRe's `lib_layout`.
- A few unambiguous project-level facts from ENA/BioProject: `study_title`,
  `center_name`.
- `citation` (OBIS/GBIF/PANGAEA, all three emit this exact field name at
  the project level) -> FAIRe's `bibliographicCitation`.

## What's deliberately mapped conservatively

- **LLM-extracted v3 native facts** from `extraction/faire_fields.py` map
  through that taxonomy's optional FAIRe hints. These mappings are
  intentionally marked `review_required=True`: the raw fact has passed
  evidence verification, but it still came from a model reading prose, not
  from a structured repository field. This gives downstream exports and
  completeness checks real coverage without pretending the mapping is a
  human-final curation decision.
- `environmental_context` (LLM-extracted): genuinely ambiguous which of
  `env_broad_scale` / `env_local_scale` / `env_medium` it corresponds to
  without per-case judgement -- mapping it to any single one would risk
  silently miscategorizing it. Left unmapped.
- `library_strategy` / `library_source` / `library_selection` (ENA): no
  FAIRe field asks for these; `assay_type_enum` (targeted | metabarcoding |
  other) is adjacent but not a deterministic function of these ENA
  values -- would need per-case judgement, not a rule.
- `base_count`, `fastq_bytes`: no FAIRe field asks for either (checked
  directly against the vendored schema's field list) -- these two, unlike
  `read_count`/`fastq_ftp`/`fastq_md5` above, genuinely have nowhere to
  map. Still flow into `DataAsset` via `assets/inventory.py` (Milestone 5)
  regardless.
- Publication bibliographic facts (`title`, `authorships`, `license`,
  `publisher`, ...): FAIRe is a molecular/sample metadata checklist, not a
  publication metadata standard -- these were never in scope for FAIRe
  mapping.
- **Not yet possible -- no adapter produces the source fact at all**, so no
  rule exists for these even though a real FAIRe field wants one:
  `associatedSequences` (no adapter surfaces genetic-sequence identifiers
  as their own fact); OBIS/GBIF's `target_gene`/primer/PCR/annealing-temp/
  amplicon-size fields (checked directly against `sources/obis.py`/
  `sources/gbif.py`: neither adapter's current API calls fetch anything
  but basic dataset/occurrence metadata -- OBIS's DNA-derived-data/MIxS
  extension, which can carry these, isn't queried by the current adapter);
  ENA's `LIBRARY_CONSTRUCTION_PROTOCOL` (not in `sources/ena.py`'s
  `RUN_FIELDS` at all). Each needs real adapter-side work before a mapping
  rule here would do anything.

`project_id`, `samp_name`, and `seq_id` (FAIRe join keys linking rows
across tables) are NOT produced from this rules table at all -- see
`mapping/faire.py` for why they're derived from `ExternalIdentifier`/
`Entity` instead of treated as ordinary mapped facts. `assay_name` can now
be mapped from a v3 text-extracted native fact when the paper explicitly
names the assay, but no assay Entity/table-row model exists yet.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

from fair_ocean_agent.database.enums import EntityLevel, MappingMethod
from fair_ocean_agent.extraction.faire_fields import native_name_to_faire_hint
from fair_ocean_agent.mapping.units import to_iso_event_date, to_meters
from fair_ocean_agent.standards.faire_registry import build_faire_registry


def _lat_only(value: str) -> str | None:
    from fair_ocean_agent.mapping.units import to_decimal_lat_lon

    pair = to_decimal_lat_lon(value)
    return pair[0] if pair else None


def _lon_only(value: str) -> str | None:
    from fair_ocean_agent.mapping.units import to_decimal_lat_lon

    pair = to_decimal_lat_lon(value)
    return pair[1] if pair else None


def _identity(value: str) -> str | None:
    return value


def _semicolon_parts(value: str) -> list[str]:
    """ENA's read_run report joins per-read-direction values (fastq_ftp,
    fastq_md5) with ';' -- one entry for single-end, two for paired-end
    (forward;reverse, in that order). Never more than two in practice
    (this pipeline only ever asks for standard short-read Illumina/ONT/
    PacBio single or paired layouts)."""
    return [part for part in value.split(";") if part]


def _fastq_filename_forward(value: str) -> str | None:
    parts = _semicolon_parts(value)
    return parts[0].rsplit("/", 1)[-1] if parts else None


def _fastq_filename_reverse(value: str) -> str | None:
    parts = _semicolon_parts(value)
    return parts[1].rsplit("/", 1)[-1] if len(parts) >= 2 else None


def _fastq_checksum_forward(value: str) -> str | None:
    parts = _semicolon_parts(value)
    return parts[0] if parts else None


def _fastq_checksum_reverse(value: str) -> str | None:
    parts = _semicolon_parts(value)
    return parts[1] if len(parts) >= 2 else None


def _constant_md5(_value: str) -> str:
    """ENA's read_run report only ever reports an MD5 digest in
    fastq_md5 -- it never states the algorithm, but there's nothing to
    disambiguate: whenever fastq_md5 is present at all, the algorithm was
    MD5. A deterministic constant, not a read-through of the source
    value."""
    return "MD5"


def _normalize_lib_layout(value: str) -> str:
    """ENA's read_run report gives PAIRED/SINGLE; FAIRe's lib_layout_enum
    wants "paired end"/"single end". Falls back to the raw value
    unrecognized -- mapping/faire.py's enum check then flags it for review
    rather than silently guessing."""
    normalized = value.strip().upper()
    if normalized == "PAIRED":
        return "paired end"
    if normalized == "SINGLE":
        return "single end"
    return value


@dataclass(frozen=True)
class MappingRule:
    source_fact_type: str
    source_entity_level: str | None  # None means "any entity level"
    target_table: str
    target_field: str
    mapping_method: str
    transform: Callable[[str], str | None] = _identity
    enum_name: str | None = None
    review_required: bool = False


_EXPLICIT_RULES: tuple[MappingRule, ...] = (
    # --- Sample-level structured facts (NCBI BioSample / ENA) ---
    MappingRule("collection_date", EntityLevel.SAMPLE.value, "sampleMetadata", "eventDate",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=to_iso_event_date),
    MappingRule("collection_date", EntityLevel.SAMPLE.value, "sampleMetadata", "verbatimEventDate",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=_identity),
    MappingRule("depth", EntityLevel.SAMPLE.value, "sampleMetadata", "minimumDepthInMeters",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=to_meters),
    MappingRule("depth", EntityLevel.SAMPLE.value, "sampleMetadata", "maximumDepthInMeters",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=to_meters),
    MappingRule("env_broad_scale", EntityLevel.SAMPLE.value, "sampleMetadata", "env_broad_scale",
                MappingMethod.EXACT_LABEL.value, enum_name="env_broad_scale_enum"),
    MappingRule("env_local_scale", EntityLevel.SAMPLE.value, "sampleMetadata", "env_local_scale",
                MappingMethod.EXACT_LABEL.value, enum_name="env_local_scale_enum"),
    MappingRule("env_medium", EntityLevel.SAMPLE.value, "sampleMetadata", "env_medium",
                MappingMethod.EXACT_LABEL.value, enum_name="env_medium_enum"),
    MappingRule("geo_loc_name", EntityLevel.SAMPLE.value, "sampleMetadata", "geo_loc_name",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("lat_lon", EntityLevel.SAMPLE.value, "sampleMetadata", "decimalLatitude",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=_lat_only),
    MappingRule("lat_lon", EntityLevel.SAMPLE.value, "sampleMetadata", "decimalLongitude",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=_lon_only),
    MappingRule("collection_method", EntityLevel.SAMPLE.value, "sampleMetadata", "samp_collect_method",
                MappingMethod.DETERMINISTIC_SYNONYM.value),
    MappingRule("elev", EntityLevel.SAMPLE.value, "sampleMetadata", "elev",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("samp_collect_device", EntityLevel.SAMPLE.value, "sampleMetadata", "samp_collect_device",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("samp_size", EntityLevel.SAMPLE.value, "sampleMetadata", "samp_size",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("samp_size_unit", EntityLevel.SAMPLE.value, "sampleMetadata", "samp_size_unit",
                MappingMethod.EXACT_LABEL.value, enum_name="samp_size_unit_enum"),
    MappingRule("temp", EntityLevel.SAMPLE.value, "sampleMetadata", "temp",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("salinity", EntityLevel.SAMPLE.value, "sampleMetadata", "salinity",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("ph", EntityLevel.SAMPLE.value, "sampleMetadata", "ph",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("diss_oxygen", EntityLevel.SAMPLE.value, "sampleMetadata", "diss_oxygen",
                MappingMethod.EXACT_LABEL.value),

    # --- Sequencing-run-level structured facts (ENA) ---
    MappingRule("instrument_platform", EntityLevel.SEQUENCING_RUN.value, "projectMetadata", "platform",
                MappingMethod.DETERMINISTIC_SYNONYM.value, enum_name="platform_enum"),
    MappingRule("instrument_model", EntityLevel.SEQUENCING_RUN.value, "projectMetadata", "instrument",
                MappingMethod.DETERMINISTIC_SYNONYM.value, enum_name="instrument_enum", review_required=True),
    MappingRule("sample_accession", EntityLevel.SEQUENCING_RUN.value, "sampleMetadata", "materialSampleID",
                MappingMethod.DETERMINISTIC_SYNONYM.value),
    MappingRule("read_count", EntityLevel.SEQUENCING_RUN.value, "experimentRunMetadata", "input_read_count",
                MappingMethod.DETERMINISTIC_SYNONYM.value),
    MappingRule("fastq_ftp", EntityLevel.SEQUENCING_RUN.value, "experimentRunMetadata", "filename",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=_fastq_filename_forward),
    MappingRule("fastq_ftp", EntityLevel.SEQUENCING_RUN.value, "experimentRunMetadata", "filename2",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=_fastq_filename_reverse),
    MappingRule("fastq_md5", EntityLevel.SEQUENCING_RUN.value, "experimentRunMetadata", "checksum_filename",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=_fastq_checksum_forward),
    MappingRule("fastq_md5", EntityLevel.SEQUENCING_RUN.value, "experimentRunMetadata", "checksum_filename2",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=_fastq_checksum_reverse),
    MappingRule("fastq_md5", EntityLevel.SEQUENCING_RUN.value, "projectMetadata", "checksum_method",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=_constant_md5, enum_name="checksum_method_enum"),
    MappingRule("library_layout", EntityLevel.SEQUENCING_RUN.value, "projectMetadata", "lib_layout",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=_normalize_lib_layout, enum_name="lib_layout_enum"),

    # --- Project-level facts (ENA/BioProject) ---
    MappingRule("study_title", EntityLevel.PROJECT.value, "projectMetadata", "project_name",
                MappingMethod.DETERMINISTIC_SYNONYM.value),
    MappingRule("center_name", EntityLevel.PROJECT.value, "projectMetadata", "institution",
                MappingMethod.DETERMINISTIC_SYNONYM.value),
    MappingRule("citation", EntityLevel.PROJECT.value, "projectMetadata", "bibliographicCitation",
                MappingMethod.DETERMINISTIC_SYNONYM.value),

    # --- LLM-extracted free text (study-level): best-effort, always flagged for review ---
    MappingRule("DNA_extraction_method", EntityLevel.STUDY.value, "sampleMetadata",
                "nucl_acid_ext_method_additional", MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("PCR_amplification_conditions", EntityLevel.STUDY.value, "projectMetadata",
                "pcr_method_additional", MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("storage_conditions", EntityLevel.STUDY.value, "sampleMetadata",
                "samp_store_method_additional", MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("sequencing_platform", EntityLevel.STUDY.value, "projectMetadata", "platform",
                MappingMethod.SUGGESTED_SEMANTIC.value, enum_name="platform_enum", review_required=True),
    MappingRule("collection_method", EntityLevel.STUDY.value, "sampleMetadata", "samp_collect_method",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),

    # --- LLM-extracted study-level sampling facts: broadcast defaults ---
    MappingRule("collection_date", EntityLevel.STUDY.value, "sampleMetadata", "eventDate",
                MappingMethod.SUGGESTED_SEMANTIC.value, transform=to_iso_event_date, review_required=True),
    MappingRule("collection_date", EntityLevel.STUDY.value, "sampleMetadata", "verbatimEventDate",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("depth", EntityLevel.STUDY.value, "sampleMetadata", "minimumDepthInMeters",
                MappingMethod.SUGGESTED_SEMANTIC.value, transform=to_meters, review_required=True),
    MappingRule("depth", EntityLevel.STUDY.value, "sampleMetadata", "maximumDepthInMeters",
                MappingMethod.SUGGESTED_SEMANTIC.value, transform=to_meters, review_required=True),
    MappingRule("depths", EntityLevel.STUDY.value, "sampleMetadata", "minimumDepthInMeters",
                MappingMethod.SUGGESTED_SEMANTIC.value, transform=to_meters, review_required=True),
    MappingRule("depths", EntityLevel.STUDY.value, "sampleMetadata", "maximumDepthInMeters",
                MappingMethod.SUGGESTED_SEMANTIC.value, transform=to_meters, review_required=True),
    MappingRule("sediment_sampling_depth", EntityLevel.STUDY.value, "sampleMetadata", "minimumDepthInMeters",
                MappingMethod.SUGGESTED_SEMANTIC.value, transform=to_meters, review_required=True),
    MappingRule("sediment_sampling_depth", EntityLevel.STUDY.value, "sampleMetadata", "maximumDepthInMeters",
                MappingMethod.SUGGESTED_SEMANTIC.value, transform=to_meters, review_required=True),
    MappingRule("coordinates", EntityLevel.STUDY.value, "sampleMetadata", "decimalLatitude",
                MappingMethod.SUGGESTED_SEMANTIC.value, transform=_lat_only, review_required=True),
    MappingRule("coordinates", EntityLevel.STUDY.value, "sampleMetadata", "decimalLongitude",
                MappingMethod.SUGGESTED_SEMANTIC.value, transform=_lon_only, review_required=True),
    MappingRule("sample_collection_method", EntityLevel.STUDY.value, "sampleMetadata", "samp_collect_method",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("sediment_sampling_method", EntityLevel.STUDY.value, "sampleMetadata", "samp_collect_method",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("sample_storage_conditions", EntityLevel.STUDY.value, "sampleMetadata",
                "samp_store_method_additional", MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),

    # --- LLM-extracted study-level assay/protocol facts ---
    MappingRule("primer_sequences", EntityLevel.STUDY.value, "projectMetadata", "pcr_method_additional",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("PCR_forward_primer_sequence", EntityLevel.STUDY.value, "projectMetadata", "pcr_primer_forward",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("PCR_reverse_primer_sequence", EntityLevel.STUDY.value, "projectMetadata", "pcr_primer_reverse",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("PCR_amplification_conditions_forward_primer_sequence", EntityLevel.STUDY.value,
                "projectMetadata", "pcr_primer_forward", MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("PCR_amplification_conditions_reverse_primer_sequence", EntityLevel.STUDY.value,
                "projectMetadata", "pcr_primer_reverse", MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("PCR_conditions", EntityLevel.STUDY.value, "projectMetadata", "pcr_cond",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("PCR_amplification_conditions_temp_cycles", EntityLevel.STUDY.value, "projectMetadata",
                "pcr_cycles", MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("PCR_amplification_conditions_temp_initial", EntityLevel.STUDY.value, "projectMetadata",
                "pcr_cond", MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("PCR_amplification_conditions_temp_final", EntityLevel.STUDY.value, "projectMetadata",
                "pcr_cond", MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("PCR_amplification_conditions_thermocycler", EntityLevel.STUDY.value, "projectMetadata",
                "thermocycler", MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("PCR_amplification_conditions_PCR_adaptor", EntityLevel.STUDY.value, "projectMetadata",
                "pcr_method_additional", MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("PCR_amplification_conditions_spacer", EntityLevel.STUDY.value, "projectMetadata",
                "pcr_method_additional", MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("PCR_amplification_conditions_template", EntityLevel.STUDY.value, "projectMetadata",
                "pcr_method_additional", MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("metabarcoding_region", EntityLevel.STUDY.value, "projectMetadata", "target_gene",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("bioinformatics_workflow", EntityLevel.STUDY.value, "projectMetadata",
                "bioinfo_method_additional", MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("sequencing_methodology", EntityLevel.STUDY.value, "projectMetadata",
                "seq_method_additional", MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
)


# FAIRe fields can appear in multiple checklist classes. A model-extracted
# study-level fact has no assay/taxon/run entity model yet, so choose the
# class that preserves the most useful provenance today; exports for the
# non-project/sample classes are still header-only until those entity
# models exist.
_TARGET_TABLE_OVERRIDES = {
    "assay_name": "projectMetadata",
    "quantificationCycle": "ampData",
    "estimatedNumberOfCopies": "ampData",
    "estimatedNumberOfCopies_unit": "ampData",
    "scientificName": "taxaRaw",
    "taxonRank": "taxaRaw",
    "kingdom": "taxaRaw",
    "phylum": "taxaRaw",
    "class": "taxaRaw",
    "order": "taxaRaw",
    "family": "taxaRaw",
    "genus": "taxaRaw",
    "specificEpithet": "taxaRaw",
    "percent_match": "taxaRaw",
    "percent_query_cover": "taxaRaw",
    "confidence_score": "taxaRaw",
}

_PREFERRED_TABLES = (
    "projectMetadata",
    "sampleMetadata",
    "ampData",
    "stdData",
    "experimentRunMetadata",
    "eLowQuantData",
    "taxaRaw",
    "taxaFinal",
)


@lru_cache(maxsize=1)
def _faire_field_classes() -> dict[str, tuple[str, ...]]:
    return {
        term["upstream_field_name"]: tuple(term.get("data_type") or ())
        for term in build_faire_registry()
    }


def _target_table_for_faire_field(faire_field: str) -> str:
    override = _TARGET_TABLE_OVERRIDES.get(faire_field)
    if override:
        return override
    classes = _faire_field_classes().get(faire_field, ())
    for class_name in _PREFERRED_TABLES:
        if class_name in classes:
            return class_name
    # Defensive fallback for a future taxonomy hint added before the schema
    # resolver is updated. The tests below guard current hints.
    return "projectMetadata"


def _generated_v3_llm_rules() -> tuple[MappingRule, ...]:
    """Every generated rule here is scoped to EntityLevel.STUDY -- so only
    an _EXPLICIT_RULES entry that also applies at STUDY level (its own
    source_entity_level is STUDY, or None for "any level") should suppress
    one. A structured-source rule scoped to a *different* entity level
    (e.g. ENA's "library_layout" at SEQUENCING_RUN) must never suppress
    this taxonomy's own "library_layout" native_name (a genuinely different
    fact, coincidentally sharing a name with an ENA field) at STUDY level --
    a real regression this exact check caught: adding the ENA rule
    silently stopped every LLM-extracted STUDY-level "library_layout" fact
    from mapping at all, since the old check only compared fact_type
    strings, ignoring entity_level entirely."""
    explicit_at_study_level = {
        rule.source_fact_type
        for rule in _EXPLICIT_RULES
        if rule.source_entity_level in (None, EntityLevel.STUDY.value)
    }
    generated: list[MappingRule] = []
    for native_name, faire_field in sorted(native_name_to_faire_hint().items()):
        if native_name in explicit_at_study_level:
            continue
        generated.append(
            MappingRule(
                native_name,
                EntityLevel.STUDY.value,
                _target_table_for_faire_field(faire_field),
                faire_field,
                MappingMethod.SUGGESTED_SEMANTIC.value,
                review_required=True,
            )
        )
    return tuple(generated)


RULES: tuple[MappingRule, ...] = _EXPLICIT_RULES + _generated_v3_llm_rules()


def rules_for(fact_type: str, entity_level: str | None) -> list[MappingRule]:
    return [
        rule for rule in RULES
        if rule.source_fact_type == fact_type
        and (rule.source_entity_level is None or rule.source_entity_level == entity_level)
    ]
