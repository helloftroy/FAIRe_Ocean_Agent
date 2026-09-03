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
  `salinity`, `ph`, plus selected FAIRe fields tagged
  `in_subset: Environment` in the vendored schema
  (`_ADDITIONAL_ENVIRONMENTAL_SAMPLE_ATTRIBUTES`: `chlorophyll`,
  `nitrate`/`nitrite`, `host_species`/`host_length`/`host_tot_mass`/...).
  Separate `_unit` companions for chemistry measurements are intentionally
  not emitted as FAIRe columns; `mapping/faire.py` folds them into the value
  fields when the raw source provides both parts.
  None of these -- not even the original 8 -- produce a StandardizedValue
  for a given sample unless that sample's real BioSample record actually
  reported that attribute; a rule existing here only means one is captured
  when the source has it, not that every sample gets a value.
- Sequencing-run-level structured facts from ENA: `instrument_platform`,
  `instrument_model`, `sample_accession`, `read_count` (-> FAIRe's
  `input_read_count`), `fastq_ftp`/`fastq_md5` (-> FAIRe's
  `filename`/`filename2`/`checksum_filename`/`checksum_filename2`, split on
  ENA's `;`-joined forward/reverse pairing), and (whenever `fastq_md5` is
  present) a deterministic `checksum_method` = "MD5" -- ENA's read_run
  report only ever gives MD5 checksums, never states the algorithm
  explicitly, so this is inferred rather than read verbatim, but no less
  deterministic for it. FAIRe's `lib_layout` is derived in mapping/faire.py
  from the number of files in `fastq_ftp`, not from ENA's declared
  `library_layout` value or paper prose.
- A few unambiguous project-level facts from ENA/BioProject: `study_title`,
  `center_name`.
- `citation` (OBIS/GBIF/PANGAEA, all three emit this exact field name at
  the project level) -> FAIRe's `bibliographicCitation`.
- `associated_resource` from publication JATS method-section citations ->
  FAIRe's `associated_resource`.
- Repository DNA/assay facts from bounded OBIS/GBIF occurrence previews:
  exact Darwin Core/GBIF DNA-derived-data terms such as
  `associatedSequences`, `target_gene`, `pcr_primer_forward`,
  `pcr_primer_reverse`, primer names/references, `annealingTemp`,
  `ampliconSize`, `assay_type`, and `assay_name`.
- ENA run-level identifiers/protocol text: `run_accession` becomes
  FAIRe `associatedSequences`, and `library_construction_protocol` is
  retained as PCR/library-method narrative text for review.

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
- Still not mapped automatically: normalized/similar labels from repository
  extension blobs. OBIS/GBIF fields only map when the adapter sees exact
  FAIRe/GBIF DNA-derived-data names or explicit camelCase aliases; similar
  labels remain review candidates, not deterministic facts.

`project_id`, `samp_name`, and `seq_id` (FAIRe join keys linking rows
across tables) are NOT produced from this rules table at all -- see
`mapping/faire.py` for why they're derived from `ExternalIdentifier`/
`Entity` instead of treated as ordinary mapped facts. `assay_name` can now
be mapped from a v3 text-extracted native fact when the paper explicitly
names the assay, but no assay Entity/table-row model exists yet.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

from fair_ocean_agent.database.enums import EntityLevel, MappingMethod
from fair_ocean_agent.extraction.faire_fields import assay_scoped_field_names, native_name_to_faire_hint
from fair_ocean_agent.mapping.envo import expand_envo_terms
from fair_ocean_agent.mapping.units import (
    to_decimal_latitude,
    to_decimal_longitude,
    to_iso_event_date,
    to_max_meters,
    to_min_meters,
)
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


_PLACEHOLDER_REFERENCE_VALUES = {
    "as described above",
    "as described below",
    "see above",
    "see below",
    "see text",
}


def _non_placeholder_text(value: str) -> str | None:
    stripped = value.strip()
    if not stripped:
        return None
    normalized = " ".join(stripped.casefold().split()).strip(" .;:")
    if normalized in _PLACEHOLDER_REFERENCE_VALUES:
        return None
    return stripped


_ABSENT_LOCATION_RE = re.compile(
    r"^(?:not\s+collected|not\s+provided|not\s+available|unknown|missing|n/?a|na)$",
    re.IGNORECASE,
)


def _non_absent_geo_loc_name(value: str) -> str | None:
    stripped = value.strip()
    if not stripped or _ABSENT_LOCATION_RE.match(stripped):
        return None
    return stripped


def _license_url(value: str) -> str | None:
    stripped = value.strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = stripped

    def first_url(item) -> str | None:
        if isinstance(item, str):
            return item.strip() or None
        if isinstance(item, dict):
            url = item.get("URL") or item.get("url")
            return str(url).strip() if url else None
        if isinstance(item, list):
            for child in item:
                url = first_url(child)
                if url:
                    return url
        return None

    return first_url(parsed)


def _open_access_from_license(value: str) -> str | None:
    url = _license_url(value)
    if not url:
        return None
    normalized = url.casefold()
    if "creativecommons.org/licenses/" in normalized or "creativecommons.org/publicdomain/" in normalized:
        return "open access"
    return None


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


def _percent_without_unit(value: str) -> str | None:
    normalized = value.strip()
    if not normalized:
        return None
    if normalized.endswith("%"):
        normalized = normalized[:-1].strip()
    return normalized


_VOLUME_VALUE_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:uL|µL|μL|ul|mL|ml|L|lit(?:er|re)s?)\b|"
    r"\b\d+(?:\.\d+)?\s*(?:ng|µg|ug)\s*(?:/|\s+)?(?:uL|µL|μL|ul)(?:\s*(?:-1|−1))?\b",
    re.IGNORECASE,
)


def _volume_value(value: str) -> str | None:
    normalized = value.strip()
    if not normalized:
        return None
    return normalized if _VOLUME_VALUE_RE.search(normalized) else None


def _error_rate_tool_value(value: str) -> str | None:
    normalized = value.strip()
    if not normalized:
        return None
    if re.search(r"\bFastQC\b", normalized, re.IGNORECASE):
        return None
    return normalized


def _control_flag_value(value: str, control_terms: tuple[str, ...]) -> str | None:
    normalized = " ".join(value.strip().lower().replace("-", " ").split())
    if normalized in {"1", "true", "yes", "y"}:
        return "1"
    if normalized in {"0", "false", "no", "n"}:
        return "0"
    if not normalized:
        return None
    explicit_none_markers = ("none", "not used", "not included", "absent", "omitted", "without")
    if any(marker in normalized for marker in explicit_none_markers):
        return "0"
    return "1" if any(term in normalized for term in control_terms) else None


_NEGATIVE_CONTROL_TERMS = (
    "negative control",
    "negative controls",
    "pcr negative",
    "extraction negative",
    "field blank",
    "field blanks",
    "equipment blank",
    "equipment blanks",
    "filtration blank",
    "filtration blanks",
    "extraction blank",
    "extraction blanks",
    "reagent blank",
    "reagent blanks",
    "pcr blank",
    "pcr blanks",
    "no template control",
    "no template controls",
    "ntc",
    "blank sample",
    "blank samples",
)


_POSITIVE_CONTROL_TERMS = (
    "positive control",
    "positive controls",
    "pcr positive",
    "extraction positive",
    "mock community",
    "mock communities",
    "reference dna",
    "known dna",
    "synthetic dna",
    "gblock",
    "gblocks",
    "plasmid control",
    "plasmid controls",
    "reference tissue",
    "positive template",
    "positive amplification control",
    "positive amplification controls",
)


def _pcr_flag(value: str) -> str | None:
    return _control_flag_value(value, ("pcr", "qpcr", "ddpcr", "amplification", "polymerase chain reaction"))


def _negative_control_flag(value: str) -> str | None:
    return _control_flag_value(value, _NEGATIVE_CONTROL_TERMS)


def _positive_control_flag(value: str) -> str | None:
    return _control_flag_value(value, _POSITIVE_CONTROL_TERMS)


def _control_sample_category(value: str) -> str | None:
    normalized = value.strip().lower()
    if "negative" in normalized or "blank" in normalized or "no template" in normalized or normalized == "ntc":
        return "negative control"
    if "positive" in normalized:
        return "positive control"
    if "pcr standard" in normalized or "standard curve" in normalized:
        return "PCR standard"
    return None


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
    MappingRule("eventDate_submitted", EntityLevel.SAMPLE.value, "sampleMetadata", "eventDate_submitted",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=to_iso_event_date),
    MappingRule("depth", EntityLevel.SAMPLE.value, "sampleMetadata", "minimumDepthInMeters",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=to_min_meters),
    MappingRule("depth", EntityLevel.SAMPLE.value, "sampleMetadata", "maximumDepthInMeters",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=to_max_meters),
    MappingRule("Depth", EntityLevel.SAMPLE.value, "sampleMetadata", "minimumDepthInMeters",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=to_min_meters),
    MappingRule("Depth", EntityLevel.SAMPLE.value, "sampleMetadata", "maximumDepthInMeters",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=to_max_meters),
    MappingRule("depth_(m)", EntityLevel.SAMPLE.value, "sampleMetadata", "minimumDepthInMeters",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=to_min_meters),
    MappingRule("depth_(m)", EntityLevel.SAMPLE.value, "sampleMetadata", "maximumDepthInMeters",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=to_max_meters),
    MappingRule("samp_category", EntityLevel.SAMPLE.value, "sampleMetadata", "samp_category",
                MappingMethod.EXACT_LABEL.value, enum_name="samp_category_enum"),
    MappingRule("sample_type", EntityLevel.SAMPLE.value, "sampleMetadata", "samp_category",
                MappingMethod.SUGGESTED_SEMANTIC.value, transform=_control_sample_category,
                enum_name="samp_category_enum", review_required=True),
    MappingRule("sample_type", EntityLevel.SAMPLE.value, "projectMetadata", "neg_cont_0_1",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=_negative_control_flag,
                enum_name="neg_cont_0_1_enum"),
    MappingRule("sample_type", EntityLevel.SAMPLE.value, "projectMetadata", "pos_cont_0_1",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=_positive_control_flag,
                enum_name="pos_cont_0_1_enum"),
    MappingRule("samp_category", EntityLevel.SAMPLE.value, "projectMetadata", "neg_cont_0_1",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=_negative_control_flag,
                enum_name="neg_cont_0_1_enum"),
    MappingRule("samp_category", EntityLevel.SAMPLE.value, "projectMetadata", "pos_cont_0_1",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=_positive_control_flag,
                enum_name="pos_cont_0_1_enum"),
    MappingRule("neg_cont_type", EntityLevel.SAMPLE.value, "projectMetadata", "neg_cont_0_1",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=_negative_control_flag,
                enum_name="neg_cont_0_1_enum"),
    MappingRule("pos_cont_type", EntityLevel.SAMPLE.value, "projectMetadata", "pos_cont_0_1",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=_positive_control_flag,
                enum_name="pos_cont_0_1_enum"),
    MappingRule("env_broad_scale", EntityLevel.SAMPLE.value, "sampleMetadata", "env_broad_scale",
                MappingMethod.EXACT_LABEL.value, transform=expand_envo_terms, enum_name="env_broad_scale_enum"),
    MappingRule("env_local_scale", EntityLevel.SAMPLE.value, "sampleMetadata", "env_local_scale",
                MappingMethod.EXACT_LABEL.value, transform=expand_envo_terms, enum_name="env_local_scale_enum"),
    MappingRule("env_medium", EntityLevel.SAMPLE.value, "sampleMetadata", "env_medium",
                MappingMethod.EXACT_LABEL.value, transform=expand_envo_terms, enum_name="env_medium_enum"),
    MappingRule("isolation_source", EntityLevel.SAMPLE.value, "sampleMetadata", "env_medium",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=expand_envo_terms, enum_name="env_medium_enum"),
    # env_biome/env_feature/env_material: legacy pre-MIxS-5.0 attribute
    # names for what are now env_broad_scale/env_local_scale/env_medium
    # (see sources/ncbi.py's own comment on this exact synonym set, next
    # to its harmonized_name-preference logic). NCBI provides a
    # harmonized_name for some submissions (already handled there), but
    # not all -- real gap found live (STUDY-012a00dba8bc): this
    # submission's own env_biome/env_feature/env_material attributes carry
    # no harmonized_name at all, so they landed in source_unmapped with
    # no MappingRule instead of reaching env_broad_scale/env_local_scale/
    # env_medium. Same DETERMINISTIC_SYNONYM pattern as isolation_source
    # above, onto the same three targets.
    MappingRule("env_biome", EntityLevel.SAMPLE.value, "sampleMetadata", "env_broad_scale",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=expand_envo_terms, enum_name="env_broad_scale_enum"),
    MappingRule("env_feature", EntityLevel.SAMPLE.value, "sampleMetadata", "env_local_scale",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=expand_envo_terms, enum_name="env_local_scale_enum"),
    MappingRule("env_material", EntityLevel.SAMPLE.value, "sampleMetadata", "env_medium",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=expand_envo_terms, enum_name="env_medium_enum"),
    # Sampling_point: a real, custom (non-MIxS) BioSample attribute name
    # found live (STUDY-012a00dba8bc) holding "168h after disturbance" --
    # time elapsed relative to a reference event is exactly
    # eventDurationValue's own concept, per an explicit user request.
    # SUGGESTED_SEMANTIC (not an exact/deterministic synonym): this is a
    # judgment call about a free-form, submitter-chosen attribute name,
    # not a recognized standard synonym the way isolation_source/env_biome
    # above are.
    MappingRule("Sampling_point", EntityLevel.SAMPLE.value, "sampleMetadata", "eventDurationValue",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("geo_loc_name", EntityLevel.SAMPLE.value, "sampleMetadata", "geo_loc_name",
                MappingMethod.EXACT_LABEL.value, transform=_non_absent_geo_loc_name),
    MappingRule("cruise", EntityLevel.SAMPLE.value, "sampleMetadata", "internal_expedition_id",
                MappingMethod.DETERMINISTIC_SYNONYM.value),
    MappingRule("cruise_id", EntityLevel.SAMPLE.value, "sampleMetadata", "internal_expedition_id",
                MappingMethod.DETERMINISTIC_SYNONYM.value),
    MappingRule("expedition", EntityLevel.SAMPLE.value, "sampleMetadata", "internal_expedition_id",
                MappingMethod.DETERMINISTIC_SYNONYM.value),
    MappingRule("expedition_id", EntityLevel.SAMPLE.value, "sampleMetadata", "internal_expedition_id",
                MappingMethod.DETERMINISTIC_SYNONYM.value),
    MappingRule("campaign", EntityLevel.SAMPLE.value, "sampleMetadata", "internal_expedition_id",
                MappingMethod.DETERMINISTIC_SYNONYM.value),
    MappingRule("campaign_id", EntityLevel.SAMPLE.value, "sampleMetadata", "internal_expedition_id",
                MappingMethod.DETERMINISTIC_SYNONYM.value),
    MappingRule("voyage", EntityLevel.SAMPLE.value, "sampleMetadata", "internal_expedition_id",
                MappingMethod.DETERMINISTIC_SYNONYM.value),
    MappingRule("voyage_id", EntityLevel.SAMPLE.value, "sampleMetadata", "internal_expedition_id",
                MappingMethod.DETERMINISTIC_SYNONYM.value),
    MappingRule("station", EntityLevel.SAMPLE.value, "sampleMetadata", "internal_expedition_id",
                MappingMethod.DETERMINISTIC_SYNONYM.value),
    MappingRule("station_id", EntityLevel.SAMPLE.value, "sampleMetadata", "internal_expedition_id",
                MappingMethod.DETERMINISTIC_SYNONYM.value),
    MappingRule("lat_lon", EntityLevel.SAMPLE.value, "sampleMetadata", "decimalLatitude",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=_lat_only),
    MappingRule("lat_lon", EntityLevel.SAMPLE.value, "sampleMetadata", "decimalLongitude",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=_lon_only),
    # Separate latitude/longitude columns (common in supplementary tables,
    # unlike MIxS's combined lat_lon convention) -- "latitude"/"longitude"
    # are this pipeline's own standard-agnostic native names (see
    # sources/supplement_parsing.py), never FAIRe's own field spelling.
    MappingRule("latitude", EntityLevel.SAMPLE.value, "sampleMetadata", "decimalLatitude",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=to_decimal_latitude),
    MappingRule("longitude", EntityLevel.SAMPLE.value, "sampleMetadata", "decimalLongitude",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=to_decimal_longitude),
    MappingRule("collection_method", EntityLevel.SAMPLE.value, "sampleMetadata", "samp_collect_method",
                MappingMethod.DETERMINISTIC_SYNONYM.value),
    MappingRule("elev", EntityLevel.SAMPLE.value, "sampleMetadata", "elev",
                MappingMethod.EXACT_LABEL.value),
    # size_frac/filter_* are emitted by sources/ncbi.py's
    # _derive_filter_facts, parsed out of the samp_mat_process attribute's
    # free-text processing narrative -- never a literal NCBI attribute name
    # itself, so review_required=True.
    MappingRule("size_frac", EntityLevel.SAMPLE.value, "sampleMetadata", "size_frac",
                MappingMethod.EXACT_LABEL.value, review_required=True),
    MappingRule("filter_diameter", EntityLevel.SAMPLE.value, "sampleMetadata", "filter_diameter",
                MappingMethod.EXACT_LABEL.value, review_required=True),
    MappingRule("filter_material", EntityLevel.SAMPLE.value, "sampleMetadata", "filter_material",
                MappingMethod.EXACT_LABEL.value, enum_name="filter_material_enum", review_required=True),
    MappingRule("filter_name", EntityLevel.SAMPLE.value, "sampleMetadata", "filter_name",
                MappingMethod.EXACT_LABEL.value, transform=_non_placeholder_text, review_required=True),
    MappingRule("filter_name", EntityLevel.STUDY.value, "sampleMetadata", "filter_name",
                MappingMethod.SUGGESTED_SEMANTIC.value, transform=_non_placeholder_text, review_required=True),
    MappingRule("filter_passive_active_0_1", EntityLevel.SAMPLE.value, "sampleMetadata", "filter_passive_active_0_1",
                MappingMethod.EXACT_LABEL.value, enum_name="filter_passive_active_0_1_enum", review_required=True),
    MappingRule("samp_collect_device", EntityLevel.SAMPLE.value, "sampleMetadata", "samp_collect_device",
                MappingMethod.EXACT_LABEL.value),
    # sample_derived_from: emitted by sources/ncbi.py's own accession
    # extraction out of a MAG/MIMAG record's "derived-from" attribute
    # (real value is a full sentence, e.g. "This BioSample is a
    # metagenomic assembly obtained from the marine sediment metagenome
    # BioSample: SAMN11268106" -- the accession is pulled out of it), not
    # a literal NCBI attribute name itself, so review_required=True.
    MappingRule("sample_derived_from", EntityLevel.SAMPLE.value, "sampleMetadata", "sample_derived_from",
                MappingMethod.DETERMINISTIC_SYNONYM.value, review_required=True),
    # samp_mat_process: a real, literal NCBI BioSample attribute name when
    # present (confirmed live, 10.3389/fmicb.2024.1295149's one real
    # sediment sample: samp_mat_process="DNA extraction from sediment
    # samples") -- had a real gap of its own, same root cause as the block
    # below: passed through by sources/ncbi.py's generic attribute loop but
    # never had a MappingRule to actually reach the export.
    MappingRule("samp_mat_process", EntityLevel.SAMPLE.value, "sampleMetadata", "samp_mat_process",
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
    # chlorophyll/diss_oxygen: real FAIRe sampleMetadata fields
    # (faire:chlorophyll mixs:0000177, faire:diss_oxygen) that had no
    # MappingRule at any level before -- the LLM-extraction path only ever
    # bundled them into x_env_var_block's own catch-all STUDY-level field,
    # never gave them a real per-column mapping. Added alongside
    # scripts/apply_gold_physicochemical_enrichment.py, whose structured,
    # per-BioSample GOLD data needs a real target column, not a bundled
    # free-text one.
    MappingRule("chlorophyll", EntityLevel.SAMPLE.value, "sampleMetadata", "chlorophyll",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("diss_oxygen", EntityLevel.SAMPLE.value, "sampleMetadata", "diss_oxygen",
                MappingMethod.EXACT_LABEL.value),
    # in_situ_temp/in_situ_salinity: a real paper's own
    # methods text describing conditions measured at the time/site of
    # sample collection (search_flags.py's own LLMJudgedSearchField
    # mechanism, distinct native names from the BioSample-sourced
    # temp/salinity rules above so the two sources are never
    # confused) -- STUDY-level since a single collection event's in-situ
    # reading is typically reported once for the whole site, not per
    # sample; broadcasts into every sample's row exactly like other
    # STUDY-level sampleMetadata facts (see exports/faire.py's own
    # broadcast-as-default docstring).
    MappingRule("in_situ_temp", EntityLevel.STUDY.value, "sampleMetadata", "temp",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("in_situ_salinity", EntityLevel.STUDY.value, "sampleMetadata", "salinity",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    # biological_rep_relation: emitted by sources/replicate_grouping.py's
    # sample-name-suffix detector (via supplement_parsing.py and ncbi.py),
    # never a literal source column -- review_required=True since this is a
    # per-sample identity claim (which samples are replicates of which)
    # worth a human sanity check regardless of which detection signal fired.
    MappingRule("biological_rep_relation", EntityLevel.SAMPLE.value, "sampleMetadata", "biological_rep_relation",
                MappingMethod.EXACT_LABEL.value, review_required=True),
    # biological_rep_presence (TRUE/FALSE) and a raw "biological_rep" pass-
    # through rule used to live here -- both removed per an explicit user
    # request. biological_rep is now derived purely from the
    # biological_rep_relation facts above (mapping/faire.py::
    # _apply_biological_rep_from_relations), never a RawFact/MappingRule
    # of its own.

    # --- Sequencing-run-level structured facts (ENA) ---
    MappingRule("instrument_platform", EntityLevel.SEQUENCING_RUN.value, "projectMetadata", "platform",
                MappingMethod.DETERMINISTIC_SYNONYM.value, enum_name="platform_enum"),
    MappingRule("instrument_model", EntityLevel.SEQUENCING_RUN.value, "projectMetadata", "instrument",
                MappingMethod.DETERMINISTIC_SYNONYM.value, enum_name="instrument_enum", review_required=True),
    MappingRule("sample_accession", EntityLevel.SEQUENCING_RUN.value, "sampleMetadata", "materialSampleID",
                MappingMethod.DETERMINISTIC_SYNONYM.value),
    MappingRule("fastq_md5", EntityLevel.SEQUENCING_RUN.value, "projectMetadata", "checksum_method",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=_constant_md5, enum_name="checksum_method_enum"),
    MappingRule("library_construction_protocol", EntityLevel.SEQUENCING_RUN.value, "projectMetadata",
                "pcr_method_additional", MappingMethod.DETERMINISTIC_SYNONYM.value, review_required=True),

    # --- FAIRe experiment/library instances ---
    # One EXPERIMENT_RUN is one sample+assay-specific library row. It may
    # link to a sequencing run shared by many other libraries.
    MappingRule("samp_name", EntityLevel.EXPERIMENT_RUN.value, "experimentRunMetadata", "samp_name",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("sample_accession", EntityLevel.EXPERIMENT_RUN.value, "experimentRunMetadata", "samp_name",
                MappingMethod.DETERMINISTIC_SYNONYM.value),
    MappingRule("assay_name", EntityLevel.EXPERIMENT_RUN.value, "experimentRunMetadata", "assay_name",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("pcr_plate_id", EntityLevel.EXPERIMENT_RUN.value, "experimentRunMetadata", "pcr_plate_id",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("lib_id", EntityLevel.EXPERIMENT_RUN.value, "experimentRunMetadata", "lib_id",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("seq_run_id", EntityLevel.EXPERIMENT_RUN.value, "experimentRunMetadata", "seq_run_id",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("run_accession", EntityLevel.EXPERIMENT_RUN.value, "experimentRunMetadata", "seq_run_id",
                MappingMethod.DETERMINISTIC_SYNONYM.value),
    MappingRule("run_accession", EntityLevel.EXPERIMENT_RUN.value, "experimentRunMetadata", "associatedSequences",
                MappingMethod.DETERMINISTIC_SYNONYM.value),
    MappingRule("associatedSequences", EntityLevel.EXPERIMENT_RUN.value, "experimentRunMetadata", "associatedSequences",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("phix_perc", EntityLevel.EXPERIMENT_RUN.value, "experimentRunMetadata", "phix_perc",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("filename", EntityLevel.EXPERIMENT_RUN.value, "experimentRunMetadata", "filename",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("filename2", EntityLevel.EXPERIMENT_RUN.value, "experimentRunMetadata", "filename2",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("checksum_filename", EntityLevel.EXPERIMENT_RUN.value, "experimentRunMetadata", "checksum_filename",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("checksum_filename2", EntityLevel.EXPERIMENT_RUN.value, "experimentRunMetadata", "checksum_filename2",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("input_read_count", EntityLevel.EXPERIMENT_RUN.value, "experimentRunMetadata", "input_read_count",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("read_count", EntityLevel.EXPERIMENT_RUN.value, "experimentRunMetadata", "input_read_count",
                MappingMethod.DETERMINISTIC_SYNONYM.value),
    MappingRule("fastq_ftp", EntityLevel.EXPERIMENT_RUN.value, "experimentRunMetadata", "filename",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=_fastq_filename_forward),
    MappingRule("fastq_ftp", EntityLevel.EXPERIMENT_RUN.value, "experimentRunMetadata", "filename2",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=_fastq_filename_reverse),
    MappingRule("fastq_access_status", EntityLevel.EXPERIMENT_RUN.value, "experimentRunMetadata", "fastq_access_status",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("fastq_md5", EntityLevel.EXPERIMENT_RUN.value, "experimentRunMetadata", "checksum_filename",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=_fastq_checksum_forward),
    MappingRule("fastq_md5", EntityLevel.EXPERIMENT_RUN.value, "experimentRunMetadata", "checksum_filename2",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=_fastq_checksum_reverse),
    MappingRule("instrument_platform", EntityLevel.EXPERIMENT_RUN.value, "projectMetadata", "platform",
                MappingMethod.DETERMINISTIC_SYNONYM.value, enum_name="platform_enum"),
    MappingRule("instrument_model", EntityLevel.EXPERIMENT_RUN.value, "projectMetadata", "instrument",
                MappingMethod.DETERMINISTIC_SYNONYM.value, enum_name="instrument_enum", review_required=True),
    MappingRule("fastq_md5", EntityLevel.EXPERIMENT_RUN.value, "projectMetadata", "checksum_method",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=_constant_md5, enum_name="checksum_method_enum"),
    MappingRule("library_construction_protocol", EntityLevel.EXPERIMENT_RUN.value, "projectMetadata",
                "pcr_method_additional", MappingMethod.DETERMINISTIC_SYNONYM.value, review_required=True),

    # --- Project-level facts (ENA/BioProject) ---
    # study_title -> project_name deliberately removed: an explicit user
    # instruction to never populate project_name at all (exports/faire.py's
    # PROJECT_METADATA_SUPPRESSED_FIELDS also drops its column entirely).
    MappingRule("center_name", EntityLevel.PROJECT.value, "projectMetadata", "institution",
                MappingMethod.DETERMINISTIC_SYNONYM.value),
    MappingRule("citation", EntityLevel.PROJECT.value, "projectMetadata", "bibliographicCitation",
                MappingMethod.DETERMINISTIC_SYNONYM.value),

    # --- Publication-metadata facts (extraction/publication_metadata.py) ---
    # Literal FAIRe field names as fact_type_candidate (this module's own
    # project-metadata convention -- see extraction/publication_metadata.py's
    # docstring), EntityLevel.STUDY since every fact here is a plain
    # one-per-study value.
    MappingRule("license", EntityLevel.STUDY.value, "projectMetadata", "license",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=_license_url),
    MappingRule("license", EntityLevel.STUDY.value, "projectMetadata", "accessRights",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=_open_access_from_license),
    MappingRule("rightsHolder", EntityLevel.STUDY.value, "projectMetadata", "rightsHolder",
                MappingMethod.EXACT_LABEL.value, review_required=True),
    MappingRule("accessRights", EntityLevel.STUDY.value, "projectMetadata", "accessRights",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("bibliographicCitation", EntityLevel.STUDY.value, "projectMetadata", "bibliographicCitation",
                MappingMethod.DETERMINISTIC_SYNONYM.value),
    MappingRule("associated_resource", EntityLevel.STUDY.value, "projectMetadata", "associated_resource",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("code_repo", EntityLevel.STUDY.value, "projectMetadata", "code_repo",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("funding_source", EntityLevel.STUDY.value, "projectMetadata", "funding_source",
                MappingMethod.EXACT_LABEL.value, review_required=True),
    MappingRule("recordedBy", EntityLevel.STUDY.value, "projectMetadata", "recordedBy",
                MappingMethod.EXACT_LABEL.value),
    # recordedByID deliberately removed: an explicit user instruction to
    # never populate it at all (exports/faire.py's
    # PROJECT_METADATA_SUPPRESSED_FIELDS also drops its column entirely).
    MappingRule("project_contact", EntityLevel.STUDY.value, "projectMetadata", "project_contact",
                MappingMethod.EXACT_LABEL.value),
    # checkls_ver is `required: true` in the real schema, a pure constant
    # (this pipeline's own TARGET_SCHEMA_VERSION), never extracted from a
    # source -- see mapping/faire.py::_sync_checklist_version.
    MappingRule("checkls_ver", EntityLevel.STUDY.value, "projectMetadata", "checkls_ver",
                MappingMethod.EXACT_LABEL.value),
    # study_factor is LLM-GENERATED, not extracted (extraction/study_factor.py) --
    # review_required=True unconditionally, the same as every other
    # LLM-derived field, since a synthesized sentence deserves at least as
    # much scrutiny as a quoted one.
    MappingRule("study_factor", EntityLevel.STUDY.value, "projectMetadata", "study_factor",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),

    # --- OBIS/GBIF DNA-derived-data / Darwin Core occurrence terms ---
    # (sources/dna_extension.py's own DNA_DERIVED_FACT_ALIASES is the
    # authoritative list of native names this structured, no-LLM
    # mechanism can produce -- every one of them needs its own rule here
    # regardless of whether section_categories.py's now-removed
    # CategoryTerm taxonomy also used to share the same native name;
    # 9 of these were wrongly deleted alongside that CategoryTerm cleanup
    # and restored here, caught live by
    # test_maps_repository_dna_derived_fields_to_faire_without_llm_review_flag.)
    MappingRule("associatedSequences", EntityLevel.PROJECT.value, "experimentRunMetadata", "associatedSequences",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("target_gene", EntityLevel.PROJECT.value, "projectMetadata", "target_gene",
                MappingMethod.EXACT_LABEL.value, enum_name="target_gene_enum"),
    MappingRule("target_subfragment", EntityLevel.PROJECT.value, "projectMetadata", "target_subfragment",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("assay_type", EntityLevel.PROJECT.value, "projectMetadata", "assay_type",
                MappingMethod.EXACT_LABEL.value, enum_name="assay_type_enum"),
    MappingRule("assay_name", EntityLevel.PROJECT.value, "projectMetadata", "assay_name",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("pcr_primer_forward", EntityLevel.PROJECT.value, "projectMetadata", "pcr_primer_forward",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("pcr_primer_reverse", EntityLevel.PROJECT.value, "projectMetadata", "pcr_primer_reverse",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("pcr_primer_name_forward", EntityLevel.PROJECT.value, "projectMetadata", "pcr_primer_name_forward",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("pcr_primer_name_reverse", EntityLevel.PROJECT.value, "projectMetadata", "pcr_primer_name_reverse",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("pcr_primer_reference_forward", EntityLevel.PROJECT.value, "projectMetadata", "pcr_primer_reference_forward",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("pcr_primer_reference_reverse", EntityLevel.PROJECT.value, "projectMetadata", "pcr_primer_reference_reverse",
                MappingMethod.EXACT_LABEL.value),
    # STUDY-level companion to the PROJECT-level OBIS/GBIF rules above --
    # extraction/publication_metadata.py::extract_primer_reference_citations
    # resolves a primer's own bibliographic citation (real JATS <ref-list>
    # DOI linkage, not a guessed parenthetical-shape regex) when the paper
    # names a primer without giving its sequence, per an explicit user
    # request to chase that reference for the actual sequence.
    MappingRule("pcr_primer_reference_forward", EntityLevel.STUDY.value, "projectMetadata", "pcr_primer_reference_forward",
                MappingMethod.DETERMINISTIC_SYNONYM.value),
    MappingRule("pcr_primer_reference_reverse", EntityLevel.STUDY.value, "projectMetadata", "pcr_primer_reference_reverse",
                MappingMethod.DETERMINISTIC_SYNONYM.value),
    # mapping/primer_library.py::resolve_primer_sequences_from_corpus -- a
    # primer sequence found in ANY OTHER paper that named the exact same
    # primer, backfilled here since this paper's own text never gave it.
    # Always review_required=True: inherited from a different paper's own
    # extraction, never this paper's own explicit statement -- deliberately
    # a separate fact type (not "pcr_primer_forward" itself) so it can't
    # inherit whichever review_required setting that field's own-extraction
    # rules happen to have.
    MappingRule("pcr_primer_forward_inherited", EntityLevel.STUDY.value, "projectMetadata", "pcr_primer_forward",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("pcr_primer_reverse_inherited", EntityLevel.STUDY.value, "projectMetadata", "pcr_primer_reverse",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("annealingTemp", EntityLevel.PROJECT.value, "projectMetadata", "annealingTemp",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("ampliconSize", EntityLevel.PROJECT.value, "projectMetadata", "ampliconSize",
                MappingMethod.EXACT_LABEL.value),

    # --- LLM-extracted free text (study-level): best-effort, always flagged for review ---
    MappingRule("DNA_extraction_method", EntityLevel.STUDY.value, "sampleMetadata",
                "nucl_acid_ext_method_additional", MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("PCR_amplification_conditions", EntityLevel.STUDY.value, "projectMetadata",
                "pcr_method_additional", MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    # Explicit ASSAY-level counterpart, not auto-generated: the auto-
    # generation loop below skips any native_name that already has an
    # explicit STUDY-level rule (this one, right above) -- same reason
    # assay_type/target_gene/... each carry their own manually-written
    # EntityLevel.ASSAY rule alongside their STUDY-level one rather than
    # relying on that loop.
    MappingRule("PCR_amplification_conditions", EntityLevel.ASSAY.value, "projectMetadata",
                "pcr_method_additional", MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("storage_conditions", EntityLevel.STUDY.value, "sampleMetadata",
                "samp_store_method_additional", MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("sequencing_platform", EntityLevel.STUDY.value, "projectMetadata", "platform",
                MappingMethod.SUGGESTED_SEMANTIC.value, enum_name="platform_enum", review_required=True),
    MappingRule("seq_kit", EntityLevel.STUDY.value, "projectMetadata", "seq_kit",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("sterilise_method", EntityLevel.STUDY.value, "projectMetadata", "sterilise_method",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    # biological_rep's own raw-fact pass-through rule removed per an
    # explicit user request -- see the biological_rep_relation comment
    # above this MappingRules tuple.
    MappingRule("assay_type", EntityLevel.STUDY.value, "projectMetadata", "assay_type",
                MappingMethod.SUGGESTED_SEMANTIC.value, enum_name="assay_type_enum", review_required=True),
    MappingRule("assay_type", EntityLevel.ASSAY.value, "projectMetadata", "assay_type",
                MappingMethod.SUGGESTED_SEMANTIC.value, enum_name="assay_type_enum", review_required=True),
    MappingRule("barcoding_pcr_appr", EntityLevel.STUDY.value, "projectMetadata", "barcoding_pcr_appr",
                MappingMethod.SUGGESTED_SEMANTIC.value, enum_name="barcoding_pcr_appr_enum", review_required=True),
    MappingRule("informationWithheld", EntityLevel.STUDY.value, "projectMetadata", "informationWithheld",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("adapter_forward", EntityLevel.STUDY.value, "projectMetadata", "adapter_forward",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("adapter_reverse", EntityLevel.STUDY.value, "projectMetadata", "adapter_reverse",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("amp_vis_method", EntityLevel.STUDY.value, "projectMetadata", "amp_vis_method",
                MappingMethod.SUGGESTED_SEMANTIC.value, enum_name="amp_vis_method_enum", review_required=True),
    MappingRule("block_ref", EntityLevel.STUDY.value, "projectMetadata", "block_ref",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("block_seq", EntityLevel.STUDY.value, "projectMetadata", "block_seq",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("block_taxa", EntityLevel.STUDY.value, "projectMetadata", "block_taxa",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("detection_criteria", EntityLevel.STUDY.value, "projectMetadata", "detection_criteria",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("checksum_method", EntityLevel.STUDY.value, "projectMetadata", "checksum_method",
                MappingMethod.DETERMINISTIC_SYNONYM.value, enum_name="checksum_method_enum"),
    MappingRule("inhibition_check_0_1", EntityLevel.STUDY.value, "projectMetadata", "inhibition_check_0_1",
                MappingMethod.SUGGESTED_SEMANTIC.value, enum_name="inhibition_check_0_1_enum", review_required=True),
    MappingRule("inhibition_check", EntityLevel.STUDY.value, "projectMetadata", "inhibition_check",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("lod_method", EntityLevel.STUDY.value, "projectMetadata", "lod_method",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("loq_method", EntityLevel.STUDY.value, "projectMetadata", "loq_method",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("pcr_assay_lod", EntityLevel.STUDY.value, "projectMetadata", "pcr_assay_lod",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("pcr_assay_lod_unit", EntityLevel.STUDY.value, "projectMetadata", "pcr_assay_lod_unit",
                MappingMethod.SUGGESTED_SEMANTIC.value, enum_name="pcr_assay_lod_unit_enum", review_required=True),
    MappingRule("pcr_assay_loq", EntityLevel.STUDY.value, "projectMetadata", "pcr_assay_loq",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("pcr_assay_loq_unit", EntityLevel.STUDY.value, "projectMetadata", "pcr_assay_loq_unit",
                MappingMethod.SUGGESTED_SEMANTIC.value, enum_name="pcr_assay_loq_unit_enum", review_required=True),
    MappingRule("probe_conc", EntityLevel.STUDY.value, "projectMetadata", "probe_conc",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("probe_ref", EntityLevel.STUDY.value, "projectMetadata", "probe_ref",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("probe_seq", EntityLevel.STUDY.value, "projectMetadata", "probe_seq",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("std_source", EntityLevel.STUDY.value, "projectMetadata", "std_source",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("targeted_detection_method_additional", EntityLevel.STUDY.value, "projectMetadata",
                "targeted_detection_method_additional", MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("thresholdQuantificationCycle", EntityLevel.STUDY.value, "projectMetadata",
                "thresholdQuantificationCycle", MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("otu_clust_tool", EntityLevel.STUDY.value, "projectMetadata", "otu_clust_tool",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("otu_db", EntityLevel.STUDY.value, "projectMetadata", "otu_db",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    # otu_seq_comp_appr: a real gap found live -- this (and further
    # section_categories.py CategoryTerm fields) had a RawFact extraction
    # path but NO MappingRule at all, so a real extracted value could
    # never reach the FAIRe export regardless of extraction quality.
    # Fixed here for the field an explicit live audit specifically
    # confirmed. tax_assign_cat's own sibling rule here was removed
    # entirely per an explicit user request.
    MappingRule("otu_seq_comp_appr", EntityLevel.STUDY.value, "projectMetadata", "otu_seq_comp_appr",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("internal_downstream_analysis_techniques", EntityLevel.STUDY.value, "projectMetadata",
                "internal_downstream_analysis_techniques", MappingMethod.DETERMINISTIC_SYNONYM.value),
    MappingRule("pcr_0_1", EntityLevel.STUDY.value, "projectMetadata", "pcr_0_1",
                MappingMethod.SUGGESTED_SEMANTIC.value, transform=_pcr_flag, enum_name="pcr_0_1_enum",
                review_required=True),
    MappingRule("neg_cont_0_1", EntityLevel.STUDY.value, "projectMetadata", "neg_cont_0_1",
                MappingMethod.SUGGESTED_SEMANTIC.value, enum_name="neg_cont_0_1_enum", review_required=True),
    MappingRule("pos_cont_0_1", EntityLevel.STUDY.value, "projectMetadata", "pos_cont_0_1",
                MappingMethod.SUGGESTED_SEMANTIC.value, enum_name="pos_cont_0_1_enum", review_required=True),
    MappingRule("pcr_0_1", EntityLevel.PROJECT.value, "projectMetadata", "pcr_0_1",
                MappingMethod.EXACT_LABEL.value, transform=_pcr_flag, enum_name="pcr_0_1_enum"),
    MappingRule("neg_cont_0_1", EntityLevel.PROJECT.value, "projectMetadata", "neg_cont_0_1",
                MappingMethod.EXACT_LABEL.value, enum_name="neg_cont_0_1_enum"),
    MappingRule("pos_cont_0_1", EntityLevel.PROJECT.value, "projectMetadata", "pos_cont_0_1",
                MappingMethod.EXACT_LABEL.value, enum_name="pos_cont_0_1_enum"),
    MappingRule("collection_method", EntityLevel.STUDY.value, "sampleMetadata", "samp_collect_method",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),

    # --- LLM-extracted study-level sampling facts: broadcast defaults ---
    MappingRule("collection_date", EntityLevel.STUDY.value, "sampleMetadata", "eventDate",
                MappingMethod.SUGGESTED_SEMANTIC.value, transform=to_iso_event_date, review_required=True),
    MappingRule("depth", EntityLevel.STUDY.value, "sampleMetadata", "minimumDepthInMeters",
                MappingMethod.SUGGESTED_SEMANTIC.value, transform=to_min_meters, review_required=True),
    MappingRule("depth", EntityLevel.STUDY.value, "sampleMetadata", "maximumDepthInMeters",
                MappingMethod.SUGGESTED_SEMANTIC.value, transform=to_max_meters, review_required=True),
    MappingRule("depths", EntityLevel.STUDY.value, "sampleMetadata", "minimumDepthInMeters",
                MappingMethod.SUGGESTED_SEMANTIC.value, transform=to_min_meters, review_required=True),
    MappingRule("depths", EntityLevel.STUDY.value, "sampleMetadata", "maximumDepthInMeters",
                MappingMethod.SUGGESTED_SEMANTIC.value, transform=to_max_meters, review_required=True),
    MappingRule("sediment_sampling_depth", EntityLevel.STUDY.value, "sampleMetadata", "minimumDepthInMeters",
                MappingMethod.SUGGESTED_SEMANTIC.value, transform=to_min_meters, review_required=True),
    MappingRule("sediment_sampling_depth", EntityLevel.STUDY.value, "sampleMetadata", "maximumDepthInMeters",
                MappingMethod.SUGGESTED_SEMANTIC.value, transform=to_max_meters, review_required=True),
    MappingRule("coordinates", EntityLevel.STUDY.value, "sampleMetadata", "decimalLatitude",
                MappingMethod.SUGGESTED_SEMANTIC.value, transform=_lat_only, review_required=True),
    MappingRule("coordinates", EntityLevel.STUDY.value, "sampleMetadata", "decimalLongitude",
                MappingMethod.SUGGESTED_SEMANTIC.value, transform=_lon_only, review_required=True),
    # Same shape as coordinates above, for when a paper names a real
    # collection site without ever giving numeric coordinates (real gap
    # found live, PMC10988111: "Yantai Haichang Whale Shark Ocean Park
    # (Shandong, China)") -- broadcasts to every sample lacking its own
    # structured geo_loc_name (from the real API-derived SAMPLE-level rule
    # above), same review_required=True safety net as every other
    # LLM-inferred study-wide broadcast in this section.
    MappingRule("geo_loc_name", EntityLevel.STUDY.value, "sampleMetadata", "geo_loc_name",
                MappingMethod.SUGGESTED_SEMANTIC.value, transform=_non_absent_geo_loc_name, review_required=True),
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
    MappingRule("PCR_amplification_conditions_temp_cycles", EntityLevel.STUDY.value, "projectMetadata",
                "pcr_cycles", MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("PCR_amplification_conditions_PCR_adaptor", EntityLevel.STUDY.value, "projectMetadata",
                "pcr_method_additional", MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("PCR_amplification_conditions_spacer", EntityLevel.STUDY.value, "projectMetadata",
                "pcr_method_additional", MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("PCR_amplification_conditions_template", EntityLevel.STUDY.value, "projectMetadata",
                "pcr_method_additional", MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    # Real gap found live: pcr2_method_additional never had any extraction
    # path at all (its own CategoryTerm/MappingRule were retired alongside
    # the whole PCR2 section-category, per an earlier explicit user
    # request). Per a later explicit user request, this narrative text is
    # independently useful (shows which paragraph/PCR a given atomic
    # pcr2_* value came from when a paper describes more than one assay) --
    # see extraction/faire_fields.py's own "second_pcr_amplification_
    # conditions" comment.
    MappingRule("second_pcr_amplification_conditions", EntityLevel.STUDY.value, "projectMetadata",
                "pcr2_method_additional", MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    # Explicit ASSAY-level counterpart -- same reason as PCR_amplification_
    # conditions' own ASSAY-level rule above (the auto-generation loop
    # skips any native_name with an explicit STUDY-level rule already).
    MappingRule("second_pcr_amplification_conditions", EntityLevel.ASSAY.value, "projectMetadata",
                "pcr2_method_additional", MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("commercial_mm", EntityLevel.STUDY.value, "projectMetadata", "commercial_mm",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("custom_mm", EntityLevel.STUDY.value, "projectMetadata", "custom_mm",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("probeReporter", EntityLevel.STUDY.value, "projectMetadata", "probeReporter",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("probeQuencher", EntityLevel.STUDY.value, "projectMetadata", "probeQuencher",
                MappingMethod.SUGGESTED_SEMANTIC.value, enum_name="probeQuencher_enum", review_required=True),
    MappingRule("metabarcoding_region", EntityLevel.STUDY.value, "projectMetadata", "target_gene",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("sequencing_methodology", EntityLevel.STUDY.value, "projectMetadata",
                "seq_method_additional", MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),

    # PCR1/targeted-qPCR-ddPCR/PCR2/library-prep/raw-read-preprocessing
    # CategoryTerm rules (including their own fallback_only "_method_
    # additional" catch-alls, targeted_detection_method_additional and
    # pcr2_method_additional) removed entirely alongside those whole
    # categories in extraction/section_categories.py, per an explicit
    # user request -- see that module's own docstring.

    # OTU/ASV generation + filtering
    MappingRule("screen_contam_method", EntityLevel.STUDY.value, "projectMetadata", "screen_contam_method",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    # screen_geograph_method/screen_nontarget_method/screen_other removed
    # entirely per an explicit user request -- no longer extracted,
    # mapped, or exported anywhere.

    # --- sample_prep category (extraction/section_categories.py) --
    # STUDY-level, broadcast from paper text, targeting sampleMetadata.
    # Per an explicit user request, this is step 1 only (build the
    # category + mappings); a real gap in this exact field
    # (samp_vol_we_dna_ext) motivated the whole category -- see
    # mapping/faire.py's _SAMPLE_TYPE_ROUTED_NATIVE_NAMES for the one
    # existing per-sample-type router, not yet extended to this new
    # source (a deliberately separate, deferred "step 2").
    #
    # dna_cleanup_method already had a STUDY-level rule (auto-generated
    # from extraction/faire_fields.py's own broad-checklist taxonomy,
    # which happens to use this field's real FAIRe spelling as its
    # native name too) -- not duplicated here.
    MappingRule("samp_mat_process", EntityLevel.STUDY.value, "sampleMetadata", "samp_mat_process",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("samp_mat_process_normalized", EntityLevel.STUDY.value, "sampleMetadata", "samp_mat_process",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("prep_method_additional", EntityLevel.STUDY.value, "sampleMetadata", "prep_method_additional",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("prep_method_additional_normalized", EntityLevel.STUDY.value, "sampleMetadata",
                "prep_method_additional", MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("filter_surface_area", EntityLevel.STUDY.value, "sampleMetadata", "filter_surface_area",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    # filter_material/filter_diameter/size_frac already have a SAMPLE-level
    # rule (sources/ncbi.py's _derive_filter_facts, parsed out of a real
    # BioSample's own samp_mat_process attribute) -- these are the
    # STUDY-level counterparts for this new CategoryTerm's own
    # paper-text-derived facts, same entity-level gap as the rest of this
    # block (a rule at one entity level never matches a fact at another).
    MappingRule("filter_material", EntityLevel.STUDY.value, "sampleMetadata", "filter_material",
                MappingMethod.SUGGESTED_SEMANTIC.value, enum_name="filter_material_enum", review_required=True),
    MappingRule("filter_diameter", EntityLevel.STUDY.value, "sampleMetadata", "filter_diameter",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("size_frac", EntityLevel.STUDY.value, "sampleMetadata", "size_frac",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("prefilter_material", EntityLevel.STUDY.value, "sampleMetadata", "prefilter_material",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("filter_passive_active_0_1", EntityLevel.STUDY.value, "sampleMetadata", "filter_passive_active_0_1",
                MappingMethod.SUGGESTED_SEMANTIC.value, enum_name="filter_passive_active_0_1_enum",
                review_required=True),
    MappingRule("pump_flow_rate", EntityLevel.STUDY.value, "sampleMetadata", "pump_flow_rate",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("pump_flow_rate_unit", EntityLevel.STUDY.value, "sampleMetadata", "pump_flow_rate_unit",
                MappingMethod.SUGGESTED_SEMANTIC.value, enum_name="pump_flow_rate_unit_enum", review_required=True),
    MappingRule("samp_store_temp", EntityLevel.STUDY.value, "sampleMetadata", "samp_store_temp",
                MappingMethod.SUGGESTED_SEMANTIC.value, enum_name="samp_store_temp_enum", review_required=True),
    MappingRule("samp_store_dur", EntityLevel.STUDY.value, "sampleMetadata", "samp_store_dur",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("samp_store_loc", EntityLevel.STUDY.value, "sampleMetadata", "samp_store_loc",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("samp_store_sol", EntityLevel.STUDY.value, "sampleMetadata", "samp_store_sol",
                MappingMethod.SUGGESTED_SEMANTIC.value, enum_name="samp_store_sol_enum", review_required=True),
    MappingRule("samp_store_method_additional", EntityLevel.STUDY.value, "sampleMetadata",
                "samp_store_method_additional", MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("nucl_acid_ext_kit", EntityLevel.STUDY.value, "sampleMetadata", "nucl_acid_ext_kit",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("nucl_acid_ext_method_additional", EntityLevel.STUDY.value, "sampleMetadata",
                "nucl_acid_ext_method_additional", MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("nucl_acid_ext_method_additional_normalized", EntityLevel.STUDY.value, "sampleMetadata",
                "nucl_acid_ext_method_additional", MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("dna_cleanup_0_1", EntityLevel.STUDY.value, "sampleMetadata", "dna_cleanup_0_1",
                MappingMethod.SUGGESTED_SEMANTIC.value, enum_name="dna_cleanup_0_1_enum", review_required=True),
    MappingRule("pool_dna_num", EntityLevel.STUDY.value, "sampleMetadata", "pool_dna_num",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("samp_vol_we_dna_ext", EntityLevel.STUDY.value, "sampleMetadata", "samp_vol_we_dna_ext",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("samp_vol_we_dna_ext_unit", EntityLevel.STUDY.value, "sampleMetadata", "samp_vol_we_dna_ext_unit",
                MappingMethod.SUGGESTED_SEMANTIC.value, enum_name="samp_vol_we_dna_ext_unit_enum",
                review_required=True),
    MappingRule("nucl_acid_ext_lysis", EntityLevel.STUDY.value, "sampleMetadata", "nucl_acid_ext_lysis",
                MappingMethod.SUGGESTED_SEMANTIC.value, enum_name="nucl_acid_ext_lysis_enum", review_required=True),
    MappingRule("nucl_acid_ext_lysis_normalized", EntityLevel.STUDY.value, "sampleMetadata", "nucl_acid_ext_lysis",
                MappingMethod.SUGGESTED_SEMANTIC.value, enum_name="nucl_acid_ext_lysis_enum", review_required=True),
    MappingRule("nucl_acid_ext_sep", EntityLevel.STUDY.value, "sampleMetadata", "nucl_acid_ext_sep",
                MappingMethod.SUGGESTED_SEMANTIC.value, enum_name="nucl_acid_ext_sep_enum", review_required=True),
    MappingRule("nucl_acid_ext_sep_normalized", EntityLevel.STUDY.value, "sampleMetadata", "nucl_acid_ext_sep",
                MappingMethod.SUGGESTED_SEMANTIC.value, enum_name="nucl_acid_ext_sep_enum", review_required=True),

    # sample_prep category, second batch: DNA quantification/purity.
    MappingRule("concentration", EntityLevel.STUDY.value, "sampleMetadata", "concentration",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),

    # sample_prep category, sample-collection terms: samp_collect_device/
    # samp_size/samp_size_unit already have SAMPLE-level rules (real,
    # literal BioSample attribute passthrough) -- these are the STUDY-level
    # (paper-text) counterparts. samp_collect_method's existing STUDY-level
    # rules use the "collection_method"/"sample_collection_method"/
    # "sediment_sampling_method" native names from a different extraction
    # path; this is the new sample_prep CategoryTerm's own native name.
    # sample_composed_of/sample_derived_from have no prior rule at any
    # level. samp_category is deliberately NOT given a rule here -- see the
    # CategoryTerm's own comment in extraction/section_categories.py.
    MappingRule("samp_collect_device", EntityLevel.STUDY.value, "sampleMetadata", "samp_collect_device",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("samp_collect_device_normalized", EntityLevel.STUDY.value, "sampleMetadata", "samp_collect_device",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("samp_collect_method", EntityLevel.STUDY.value, "sampleMetadata", "samp_collect_method",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("samp_collect_method_normalized", EntityLevel.STUDY.value, "sampleMetadata", "samp_collect_method",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("samp_size", EntityLevel.STUDY.value, "sampleMetadata", "samp_size",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("samp_size_unit", EntityLevel.STUDY.value, "sampleMetadata", "samp_size_unit",
                MappingMethod.SUGGESTED_SEMANTIC.value, enum_name="samp_size_unit_enum", review_required=True),
    MappingRule("sample_composed_of", EntityLevel.STUDY.value, "sampleMetadata", "sample_composed_of",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("sample_derived_from", EntityLevel.STUDY.value, "sampleMetadata", "sample_derived_from",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("internal_expedition_id", EntityLevel.STUDY.value, "sampleMetadata", "internal_expedition_id",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    # x_env_var_block: bundles ~18 individually low-yield physicochemical
    # sampleMetadata fields into one pipe-joined free-text broadcast, per an
    # explicit user request -- not a real FAIRe field (exports/faire.py's
    # CUSTOM_ENV_VAR_BLOCK_FIELD carries it as a custom, non-schema column;
    # SAMPLE_METADATA_SUPPRESSED_FIELDS hides the 18 individual fields it
    # replaces). STUDY-level like internal_expedition_id above: this
    # CategoryTerm only ever produces a paper/supplement-text broadcast,
    # never a real per-sample structured value.
    MappingRule("x_env_var_block", EntityLevel.STUDY.value, "sampleMetadata", "x_env_var_block",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    # x_pulled_env_var: a dedicated second pass over x_env_var_block's own
    # quotes (extraction/section_category_extraction.py::
    # extract_pulled_env_var_facts), filtering it down to only genuine
    # "name = value" measured pairs and dropping bare-mention/experimental-
    # manipulation quotes -- per an explicit user request to keep
    # x_env_var_block as-is and add this as a separate, cleaner companion
    # column rather than replacing it. Same STUDY-level broadcast shape.
    MappingRule("x_pulled_env_var", EntityLevel.STUDY.value, "sampleMetadata", "x_pulled_env_var",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    # spreadsheet_headers: a real column header row's own verbatim text
    # (sources/supplement_parsing.py::_facts_from_rows), not an LLM
    # judgment call -- a clean deterministic transcription, so no review
    # needed. Not a real FAIRe field (exports/faire.py's
    # CUSTOM_SPREADSHEET_HEADERS_FIELD carries it as a custom, non-schema
    # projectMetadata column), per an explicit user request.
    MappingRule("spreadsheet_headers", EntityLevel.STUDY.value, "projectMetadata", "spreadsheet_headers",
                MappingMethod.DETERMINISTIC_SYNONYM.value),
)


# Selected FAIRe Environment-subset fields that we still keep in sample
# output. Fields the user explicitly removed from sampleMetadata are absent,
# and chemistry `_unit` companions are folded into their value fields by
# mapping/faire.py instead of being exported as separate columns.
_ADDITIONAL_ENVIRONMENTAL_SAMPLE_ATTRIBUTES: tuple[tuple[str, str | None], ...] = (
    ("chlorophyll", None),
    ("diss_inorg_carb", None),
    ("diss_inorg_nitro", None),
    ("diss_org_carb", None),
    ("diss_org_nitro", None),
    ("host_height", None),
    ("host_height_unit", "host_height_unit_enum"),
    ("host_length", None),
    ("host_length_unit", "host_length_unit_enum"),
    ("host_life_stage", "host_life_stage_enum"),
    ("host_species", None),
    ("host_tot_mass", None),
    ("host_tot_mass_unit", "host_tot_mass_unit_enum"),
    ("nitrate", None),
    ("nitrite", None),
    ("org_matter", None),
    ("part_org_carb", None),
    ("part_org_nitro", None),
    ("suspend_part_matter", None),
    ("tot_carb", None),
    ("tot_depth_water_col", None),
    ("tot_diss_nitro", None),
    ("tot_nitro", None),
    ("tot_org_carb", None),
    ("tot_part_carb", None),
)


def _generated_environmental_sample_rules() -> tuple[MappingRule, ...]:
    return tuple(
        MappingRule(field, EntityLevel.SAMPLE.value, "sampleMetadata", field, MappingMethod.EXACT_LABEL.value, enum_name=enum_name)
        for field, enum_name in _ADDITIONAL_ENVIRONMENTAL_SAMPLE_ATTRIBUTES
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

_V3_NATIVE_TRANSFORMS: dict[str, Callable[[str], str | None]] = {
    "pcr_reaction_volume": _volume_value,
    "template_dna_volume": _volume_value,
    "second_pcr_reaction_volume": _volume_value,
    "second_pcr_template_dna_volume": _volume_value,
}


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
    one. A structured-source rule scoped to a *different* entity level must
    never suppress a taxonomy native_name at STUDY level -- this check
    compares entity_level as well as fact_type so generated LLM rules are
    not accidentally dropped by unrelated API rules.

    Assay-scoped native names (extraction/faire_fields.assay_scoped_field_names --
    primers, target gene, PCR/qPCR conditions, ...) additionally get a
    second, parallel rule at EntityLevel.ASSAY: a paper describing more than
    one distinct assay tags each assay's facts with entity_level=ASSAY
    (extraction/text.py's assay_tag), and without a matching ASSAY-level
    rule those facts would have nowhere to map at all, since rules_for
    requires an exact entity_level match. The existing STUDY-level rule is
    left completely untouched (not widened to source_entity_level=None/"any
    level") deliberately: _EXPLICIT_RULES already has literal-FAIRe-spelling
    rules for "target_gene"/"assay_type"/"assay_name" at EntityLevel.PROJECT
    (repository/OBIS/GBIF-sourced) and EntityLevel.EXPERIMENT_RUN, sharing
    the exact fact_type_candidate string with these native names -- widening
    to None would make the generated rule also match those structured
    facts. EntityLevel.ASSAY is unused as a source_entity_level anywhere in
    _EXPLICIT_RULES, so the new rule is collision-free by construction."""
    explicit_at_study_level = {
        rule.source_fact_type
        for rule in _EXPLICIT_RULES
        if rule.source_entity_level in (None, EntityLevel.STUDY.value)
    }
    assay_scoped = assay_scoped_field_names()
    generated: list[MappingRule] = []
    for native_name, faire_field in sorted(native_name_to_faire_hint().items()):
        if native_name in explicit_at_study_level:
            continue
        target_table = _target_table_for_faire_field(faire_field)
        transform = _V3_NATIVE_TRANSFORMS.get(native_name, _identity)
        generated.append(
            MappingRule(
                native_name,
                EntityLevel.STUDY.value,
                target_table,
                faire_field,
                MappingMethod.SUGGESTED_SEMANTIC.value,
                transform=transform,
                review_required=True,
            )
        )
        if native_name in assay_scoped:
            generated.append(
                MappingRule(
                    native_name,
                    EntityLevel.ASSAY.value,
                    target_table,
                    faire_field,
                    MappingMethod.SUGGESTED_SEMANTIC.value,
                    transform=transform,
                    review_required=True,
                )
            )
    return tuple(generated)


RULES: tuple[MappingRule, ...] = (
    _EXPLICIT_RULES + _generated_environmental_sample_rules() + _generated_v3_llm_rules()
)


def rules_for(fact_type: str, entity_level: str | None) -> list[MappingRule]:
    return [
        rule for rule in RULES
        if rule.source_fact_type == fact_type
        and (rule.source_entity_level is None or rule.source_entity_level == entity_level)
    ]
