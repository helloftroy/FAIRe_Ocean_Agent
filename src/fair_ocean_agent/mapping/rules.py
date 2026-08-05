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
  `salinity`, `ph`, `diss_oxygen`, plus every other FAIRe field tagged
  `in_subset: Environment` in the vendored schema (63 more --
  `_ADDITIONAL_ENVIRONMENTAL_SAMPLE_ATTRIBUTES`: `chlorophyll`, `turbidity`,
  `nitrate`/`nitrite`, `host_species`/`host_length`/`host_tot_mass`/...,
  `tidal_stage`, `wind_speed`, and all their `_unit`/method companions).
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
  deterministic for it. `library_layout` (added to `sources/ena.py`'s
  `RUN_FIELDS` alongside this) -> FAIRe's `lib_layout`.
- A few unambiguous project-level facts from ENA/BioProject: `study_title`,
  `center_name`.
- `citation` (OBIS/GBIF/PANGAEA, all three emit this exact field name at
  the project level) -> FAIRe's `bibliographicCitation`.
- Repository DNA/assay facts from bounded OBIS/GBIF occurrence previews:
  exact Darwin Core/GBIF DNA-derived-data terms such as
  `associatedSequences`, `target_gene`, `pcr_primer_forward`,
  `pcr_primer_reverse`, primer names/references, `annealingTemp`,
  `ampliconSize`, `pcr_cond`, `assay_type`, and `assay_name`.
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
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

from fair_ocean_agent.database.enums import EntityLevel, MappingMethod
from fair_ocean_agent.extraction.faire_fields import assay_scoped_field_names, native_name_to_faire_hint
from fair_ocean_agent.mapping.units import to_decimal_latitude, to_decimal_longitude, to_iso_event_date, to_meters
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
    MappingRule("collection_date", EntityLevel.SAMPLE.value, "sampleMetadata", "verbatimEventDate",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=_identity),
    MappingRule("depth", EntityLevel.SAMPLE.value, "sampleMetadata", "minimumDepthInMeters",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=to_meters),
    MappingRule("depth", EntityLevel.SAMPLE.value, "sampleMetadata", "maximumDepthInMeters",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=to_meters),
    MappingRule("Depth", EntityLevel.SAMPLE.value, "sampleMetadata", "minimumDepthInMeters",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=to_meters),
    MappingRule("Depth", EntityLevel.SAMPLE.value, "sampleMetadata", "maximumDepthInMeters",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=to_meters),
    MappingRule("depth_(m)", EntityLevel.SAMPLE.value, "sampleMetadata", "minimumDepthInMeters",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=to_meters),
    MappingRule("depth_(m)", EntityLevel.SAMPLE.value, "sampleMetadata", "maximumDepthInMeters",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=to_meters),
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
    MappingRule("neg_cont_type", EntityLevel.SAMPLE.value, "sampleMetadata", "neg_cont_type",
                MappingMethod.EXACT_LABEL.value, enum_name="neg_cont_type_enum"),
    MappingRule("neg_cont_type", EntityLevel.SAMPLE.value, "projectMetadata", "neg_cont_0_1",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=_negative_control_flag,
                enum_name="neg_cont_0_1_enum"),
    MappingRule("pos_cont_type", EntityLevel.SAMPLE.value, "sampleMetadata", "pos_cont_type",
                MappingMethod.EXACT_LABEL.value, enum_name="pos_cont_type_enum"),
    MappingRule("pos_cont_type", EntityLevel.SAMPLE.value, "projectMetadata", "pos_cont_0_1",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=_positive_control_flag,
                enum_name="pos_cont_0_1_enum"),
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
    # biological_rep_relation: emitted by sources/replicate_grouping.py's
    # sample-name-suffix detector (via supplement_parsing.py and ncbi.py),
    # never a literal source column -- review_required=True since this is a
    # per-sample identity claim (which samples are replicates of which)
    # worth a human sanity check regardless of which detection signal fired.
    MappingRule("biological_rep_relation", EntityLevel.SAMPLE.value, "sampleMetadata", "biological_rep_relation",
                MappingMethod.EXACT_LABEL.value, review_required=True),

    # --- Sequencing-run-level structured facts (ENA) ---
    MappingRule("instrument_platform", EntityLevel.SEQUENCING_RUN.value, "projectMetadata", "platform",
                MappingMethod.DETERMINISTIC_SYNONYM.value, enum_name="platform_enum"),
    MappingRule("instrument_model", EntityLevel.SEQUENCING_RUN.value, "projectMetadata", "instrument",
                MappingMethod.DETERMINISTIC_SYNONYM.value, enum_name="instrument_enum", review_required=True),
    MappingRule("sample_accession", EntityLevel.SEQUENCING_RUN.value, "sampleMetadata", "materialSampleID",
                MappingMethod.DETERMINISTIC_SYNONYM.value),
    MappingRule("fastq_md5", EntityLevel.SEQUENCING_RUN.value, "projectMetadata", "checksum_method",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=_constant_md5, enum_name="checksum_method_enum"),
    MappingRule("library_layout", EntityLevel.SEQUENCING_RUN.value, "projectMetadata", "lib_layout",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=_normalize_lib_layout, enum_name="lib_layout_enum"),
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
    MappingRule("lib_conc", EntityLevel.EXPERIMENT_RUN.value, "experimentRunMetadata", "lib_conc",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("lib_conc_unit", EntityLevel.EXPERIMENT_RUN.value, "experimentRunMetadata", "lib_conc_unit",
                MappingMethod.EXACT_LABEL.value, enum_name="lib_conc_unit_enum"),
    MappingRule("lib_conc_meth", EntityLevel.EXPERIMENT_RUN.value, "experimentRunMetadata", "lib_conc_meth",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("phix_perc", EntityLevel.EXPERIMENT_RUN.value, "experimentRunMetadata", "phix_perc",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("mid_forward", EntityLevel.EXPERIMENT_RUN.value, "experimentRunMetadata", "mid_forward",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("mid_reverse", EntityLevel.EXPERIMENT_RUN.value, "experimentRunMetadata", "mid_reverse",
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
    MappingRule("fastq_md5", EntityLevel.EXPERIMENT_RUN.value, "experimentRunMetadata", "checksum_filename",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=_fastq_checksum_forward),
    MappingRule("fastq_md5", EntityLevel.EXPERIMENT_RUN.value, "experimentRunMetadata", "checksum_filename2",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=_fastq_checksum_reverse),
    MappingRule("instrument_platform", EntityLevel.EXPERIMENT_RUN.value, "projectMetadata", "platform",
                MappingMethod.DETERMINISTIC_SYNONYM.value, enum_name="platform_enum"),
    MappingRule("instrument_model", EntityLevel.EXPERIMENT_RUN.value, "projectMetadata", "instrument",
                MappingMethod.DETERMINISTIC_SYNONYM.value, enum_name="instrument_enum", review_required=True),
    MappingRule("library_layout", EntityLevel.EXPERIMENT_RUN.value, "projectMetadata", "lib_layout",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=_normalize_lib_layout, enum_name="lib_layout_enum"),
    MappingRule("fastq_md5", EntityLevel.EXPERIMENT_RUN.value, "projectMetadata", "checksum_method",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=_constant_md5, enum_name="checksum_method_enum"),
    MappingRule("library_construction_protocol", EntityLevel.EXPERIMENT_RUN.value, "projectMetadata",
                "pcr_method_additional", MappingMethod.DETERMINISTIC_SYNONYM.value, review_required=True),

    # --- Project-level facts (ENA/BioProject) ---
    MappingRule("study_title", EntityLevel.PROJECT.value, "projectMetadata", "project_name",
                MappingMethod.DETERMINISTIC_SYNONYM.value),
    MappingRule("center_name", EntityLevel.PROJECT.value, "projectMetadata", "institution",
                MappingMethod.DETERMINISTIC_SYNONYM.value),
    MappingRule("citation", EntityLevel.PROJECT.value, "projectMetadata", "bibliographicCitation",
                MappingMethod.DETERMINISTIC_SYNONYM.value),

    # --- Deterministic publication-metadata facts (extraction/publication_metadata.py) ---
    # Literal FAIRe field names as fact_type_candidate (this module's own
    # structured-adapter convention -- see extraction/publication_metadata.py's
    # docstring), EntityLevel.STUDY since every fact here is a plain
    # one-per-study value, no LLM/model-vocabulary coupling to guard
    # against. Every one of these was marked "No LLM" in an explicit user
    # review of a NOAA/SEUS-MBON FAIRe checklist.
    MappingRule("license", EntityLevel.STUDY.value, "projectMetadata", "license",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=_license_url),
    MappingRule("license", EntityLevel.STUDY.value, "projectMetadata", "accessRights",
                MappingMethod.DETERMINISTIC_SYNONYM.value, transform=_open_access_from_license),
    MappingRule("rightsHolder", EntityLevel.STUDY.value, "projectMetadata", "rightsHolder",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("accessRights", EntityLevel.STUDY.value, "projectMetadata", "accessRights",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("bibliographicCitation", EntityLevel.STUDY.value, "projectMetadata", "bibliographicCitation",
                MappingMethod.DETERMINISTIC_SYNONYM.value),
    MappingRule("code_repo", EntityLevel.STUDY.value, "projectMetadata", "code_repo",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("recordedBy", EntityLevel.STUDY.value, "projectMetadata", "recordedBy",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("recordedByID", EntityLevel.STUDY.value, "projectMetadata", "recordedByID",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("project_contact", EntityLevel.STUDY.value, "projectMetadata", "project_contact",
                MappingMethod.EXACT_LABEL.value),

    # --- OBIS/GBIF DNA-derived-data / Darwin Core occurrence terms ---
    MappingRule("associatedSequences", EntityLevel.PROJECT.value, "experimentRunMetadata", "associatedSequences",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("target_gene", EntityLevel.PROJECT.value, "projectMetadata", "target_gene",
                MappingMethod.EXACT_LABEL.value, enum_name="target_gene_enum"),
    MappingRule("target_subfragment", EntityLevel.PROJECT.value, "projectMetadata", "target_subfragment",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("pcr_primer_forward", EntityLevel.PROJECT.value, "projectMetadata", "pcr_primer_forward",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("pcr_primer_reverse", EntityLevel.PROJECT.value, "projectMetadata", "pcr_primer_reverse",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("pcr_primer_name_forward", EntityLevel.PROJECT.value, "projectMetadata", "pcr_primer_name_forward",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("pcr_primer_name_reverse", EntityLevel.PROJECT.value, "projectMetadata", "pcr_primer_name_reverse",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("pcr_primer_reference_forward", EntityLevel.PROJECT.value, "projectMetadata",
                "pcr_primer_reference_forward", MappingMethod.EXACT_LABEL.value),
    MappingRule("pcr_primer_reference_reverse", EntityLevel.PROJECT.value, "projectMetadata",
                "pcr_primer_reference_reverse", MappingMethod.EXACT_LABEL.value),
    MappingRule("annealingTemp", EntityLevel.PROJECT.value, "projectMetadata", "annealingTemp",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("ampliconSize", EntityLevel.PROJECT.value, "projectMetadata", "ampliconSize",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("pcr_cond", EntityLevel.PROJECT.value, "projectMetadata", "pcr_cond",
                MappingMethod.EXACT_LABEL.value),
    MappingRule("assay_type", EntityLevel.PROJECT.value, "projectMetadata", "assay_type",
                MappingMethod.EXACT_LABEL.value, enum_name="assay_type_enum"),
    MappingRule("assay_name", EntityLevel.PROJECT.value, "projectMetadata", "assay_name",
                MappingMethod.EXACT_LABEL.value),

    # --- LLM-extracted free text (study-level): best-effort, always flagged for review ---
    MappingRule("DNA_extraction_method", EntityLevel.STUDY.value, "sampleMetadata",
                "nucl_acid_ext_method_additional", MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("PCR_amplification_conditions", EntityLevel.STUDY.value, "projectMetadata",
                "pcr_method_additional", MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("storage_conditions", EntityLevel.STUDY.value, "sampleMetadata",
                "samp_store_method_additional", MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("sequencing_platform", EntityLevel.STUDY.value, "projectMetadata", "platform",
                MappingMethod.SUGGESTED_SEMANTIC.value, enum_name="platform_enum", review_required=True),
    MappingRule("seq_kit", EntityLevel.STUDY.value, "projectMetadata", "seq_kit",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("sequencing_location", EntityLevel.STUDY.value, "projectMetadata", "sequencing_location",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("sterilise_method", EntityLevel.STUDY.value, "projectMetadata", "sterilise_method",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("biological_rep", EntityLevel.STUDY.value, "projectMetadata", "biological_rep",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("assay_type", EntityLevel.STUDY.value, "projectMetadata", "assay_type",
                MappingMethod.SUGGESTED_SEMANTIC.value, enum_name="assay_type_enum", review_required=True),
    MappingRule("assay_type", EntityLevel.ASSAY.value, "projectMetadata", "assay_type",
                MappingMethod.SUGGESTED_SEMANTIC.value, enum_name="assay_type_enum", review_required=True),
    MappingRule("barcoding_pcr_appr", EntityLevel.STUDY.value, "projectMetadata", "barcoding_pcr_appr",
                MappingMethod.SUGGESTED_SEMANTIC.value, enum_name="barcoding_pcr_appr_enum", review_required=True),
    MappingRule("lib_screen", EntityLevel.STUDY.value, "projectMetadata", "lib_screen",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("adapter_forward", EntityLevel.STUDY.value, "projectMetadata", "adapter_forward",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("adapter_reverse", EntityLevel.STUDY.value, "projectMetadata", "adapter_reverse",
                MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
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
    MappingRule("bioinformatics_workflow", EntityLevel.STUDY.value, "projectMetadata",
                "bioinfo_method_additional", MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
    MappingRule("sequencing_methodology", EntityLevel.STUDY.value, "projectMetadata",
                "seq_method_additional", MappingMethod.SUGGESTED_SEMANTIC.value, review_required=True),
)


# Every FAIRe field tagged in_subset: Environment in the vendored schema,
# minus the ones already given an explicit rule above (elev, temp,
# salinity, ph, diss_oxygen, minimumDepthInMeters/maximumDepthInMeters).
# All 63 arrive through the exact same NCBI BioSample Attributes/Attribute
# passthrough as everything else in this table -- a real BioSample MIxS
# submission's attribute name is expected to equal the FAIRe field name
# exactly (same assumption geo_loc_name/env_broad_scale/elev/etc. already
# make). None of these produce a real StandardizedValue today (checked
# directly against the real database: no BioSample record in this corpus
# has reported any of them yet) -- the rules are correct and inert until a
# real source has the attribute, same as elev/temp/salinity/ph/diss_oxygen
# were before this. A tuple of (field, enum_name_or_None) rather than a
# flat name list since about half of these have a closed-vocabulary
# `_unit`/`_enum` companion; `enum_name` is safe to pass even if a name
# here turns out wrong (mapping/vocabularies.check_value treats an unknown
# enum as "not checked", never a crash or false failure).
_ADDITIONAL_ENVIRONMENTAL_SAMPLE_ATTRIBUTES: tuple[tuple[str, str | None], ...] = (
    ("alt", None),
    ("chlorophyll", None),
    ("diss_inorg_carb", None),
    ("diss_inorg_carb_unit", "diss_inorg_carb_unit_enum"),
    ("diss_inorg_nitro", None),
    ("diss_inorg_nitro_unit", "diss_inorg_nitro_unit_enum"),
    ("diss_org_carb", None),
    ("diss_org_carb_unit", "diss_org_carb_unit_enum"),
    ("diss_org_nitro", None),
    ("diss_org_nitro_unit", "diss_org_nitro_unit_enum"),
    ("diss_oxygen_unit", "diss_oxygen_unit_enum"),
    ("host_height", None),
    ("host_height_unit", "host_height_unit_enum"),
    ("host_length", None),
    ("host_length_unit", "host_length_unit_enum"),
    ("host_life_stage", "host_life_stage_enum"),
    ("host_species", None),
    ("host_tot_mass", None),
    ("host_tot_mass_unit", "host_tot_mass_unit_enum"),
    ("humidity", None),
    ("light_intensity", None),
    ("nitrate", None),
    ("nitrate_unit", "nitrate_unit_enum"),
    ("nitrite", None),
    ("nitrite_unit", "nitrite_unit_enum"),
    ("nitro", None),
    ("nitro_unit", "nitro_unit_enum"),
    ("org_carb", None),
    ("org_carb_unit", "org_carb_unit_enum"),
    ("org_matter", None),
    ("org_matter_unit", "org_matter_unit_enum"),
    ("org_nitro", None),
    ("org_nitro_unit", "org_nitro_unit_enum"),
    ("part_org_carb", None),
    ("part_org_carb_unit", "part_org_carb_unit_enum"),
    ("part_org_nitro", None),
    ("part_org_nitro_unit", "part_org_nitro_unit_enum"),
    ("ph_meth", None),
    ("samp_weather", None),
    ("solar_irradiance", None),
    ("suspend_part_matter", None),
    ("tidal_stage", None),
    ("tot_carb", None),
    ("tot_carb_unit", "tot_carb_unit_enum"),
    ("tot_depth_water_col", None),
    ("tot_diss_nitro", None),
    ("tot_diss_nitro_unit", "tot_diss_nitro_unit_enum"),
    ("tot_inorg_nitro", None),
    ("tot_inorg_nitro_unit", "tot_inorg_nitro_unit_enum"),
    ("tot_nitro", None),
    ("tot_nitro_cont_meth", None),
    ("tot_nitro_content", None),
    ("tot_nitro_content_unit", "tot_nitro_content_unit_enum"),
    ("tot_nitro_unit", "tot_nitro_unit_enum"),
    ("tot_org_c_meth", None),
    ("tot_org_carb", None),
    ("tot_org_carb_unit", "tot_org_carb_unit_enum"),
    ("tot_part_carb", None),
    ("tot_part_carb_unit", "tot_part_carb_unit_enum"),
    ("turbidity", None),
    ("water_current", None),
    ("wind_direction", None),
    ("wind_speed", None),
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
    strings, ignoring entity_level entirely.

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
        generated.append(
            MappingRule(
                native_name,
                EntityLevel.STUDY.value,
                target_table,
                faire_field,
                MappingMethod.SUGGESTED_SEMANTIC.value,
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
