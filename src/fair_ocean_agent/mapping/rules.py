"""The raw_fact -> FAIRe field rules table.

Each `MappingRule` says: a RawFact with this `fact_type_candidate` (and,
where relevant, this `entity_level`) becomes a StandardizedValue at this
FAIRe `target_table`/`target_field`. `mapping/faire.py` applies these rules
against a study's raw_facts; it does not contain per-field logic itself --
adding a new mapped field should only ever mean adding a rule here.

## What has full rule coverage (grounded in this pipeline's real,
## observed raw_facts vocabulary -- see the Milestone 6 section of
## README.md for the query that produced this list)

- Sample-level structured facts from NCBI BioSample / ENA: `collection_date`,
  `depth`, `env_broad_scale`, `env_local_scale`, `env_medium`,
  `geo_loc_name`, `lat_lon`, `collection_method`.
- Sequencing-run-level structured facts from ENA: `instrument_platform`,
  `instrument_model`, `sample_accession`.
- A few unambiguous project-level facts from ENA/BioProject: `study_title`,
  `center_name`.

## What's deliberately NOT mapped here, and why

- **LLM-extracted free-text facts** (`DNA_extraction_method`,
  `PCR_amplification_conditions`, `storage_conditions`,
  `sequencing_platform`) ARE mapped, but only to FAIRe's own
  `*_method_additional` free-text fallback fields (or, for
  `sequencing_platform`, to the enum field with `review_required=True`
  flagged) -- never forced into the atomic fields FAIRe actually wants
  (`pcr_primer_forward`, `pcr_cond`, `pcr_cycles`, `annealingTemp`,
  `nucl_acid_ext_kit`, ...). The current extraction prompt (Milestone 4)
  produces one coarse blob per concept, not FAIRe's atomic granularity, so
  a rule that split it up would be inventing structure the source text
  doesn't actually give deterministic evidence for. Getting full atomic
  coverage here needs a FAIRe-aware extraction prompt. Where the current
  LLM already emitted an atomic-looking fact (for example
  `PCR_forward_primer_sequence`), the value is mapped to the matching FAIRe
  field but still flagged `review_required=True` because those facts came
  from an open-vocabulary prompt rather than a FAIRe-constrained extractor.
- `environmental_context` (LLM-extracted): genuinely ambiguous which of
  `env_broad_scale` / `env_local_scale` / `env_medium` it corresponds to
  without per-case judgement -- mapping it to any single one would risk
  silently miscategorizing it. Left unmapped.
- `library_strategy` / `library_source` / `library_selection` (ENA): no
  FAIRe field asks for these; `assay_type_enum` (targeted | metabarcoding |
  other) is adjacent but not a deterministic function of these ENA
  values -- would need per-case judgement, not a rule.
- `base_count`, `read_count`, `fastq_ftp`, `fastq_bytes`, `fastq_md5`: these
  already flow into `DataAsset` via `assets/inventory.py` (Milestone 5);
  duplicating them as `standardized_values` would be redundant.
- Publication bibliographic facts (`title`, `authorships`, `license`,
  `publisher`, ...): FAIRe is a molecular/sample metadata checklist, not a
  publication metadata standard -- these were never in scope for FAIRe
  mapping.

`project_id`, `samp_name`, `assay_name`, and `seq_id` (the FAIRe join keys
linking rows across tables) are NOT produced from this rules table at all
-- see `mapping/faire.py` for why they're derived from `ExternalIdentifier`/
`Entity` instead of treated as ordinary mapped facts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from fair_ocean_agent.database.enums import EntityLevel, MappingMethod
from fair_ocean_agent.mapping.units import to_iso_event_date, to_meters


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


RULES: tuple[MappingRule, ...] = (
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

    # --- Sequencing-run-level structured facts (ENA) ---
    MappingRule("instrument_platform", EntityLevel.SEQUENCING_RUN.value, "projectMetadata", "platform",
                MappingMethod.DETERMINISTIC_SYNONYM.value, enum_name="platform_enum"),
    MappingRule("instrument_model", EntityLevel.SEQUENCING_RUN.value, "projectMetadata", "instrument",
                MappingMethod.DETERMINISTIC_SYNONYM.value, enum_name="instrument_enum", review_required=True),
    MappingRule("sample_accession", EntityLevel.SEQUENCING_RUN.value, "sampleMetadata", "materialSampleID",
                MappingMethod.DETERMINISTIC_SYNONYM.value),

    # --- Project-level facts (ENA/BioProject) ---
    MappingRule("study_title", EntityLevel.PROJECT.value, "projectMetadata", "project_name",
                MappingMethod.DETERMINISTIC_SYNONYM.value),
    MappingRule("center_name", EntityLevel.PROJECT.value, "projectMetadata", "institution",
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


def rules_for(fact_type: str, entity_level: str | None) -> list[MappingRule]:
    return [
        rule for rule in RULES
        if rule.source_fact_type == fact_type
        and (rule.source_entity_level is None or rule.source_entity_level == entity_level)
    ]
