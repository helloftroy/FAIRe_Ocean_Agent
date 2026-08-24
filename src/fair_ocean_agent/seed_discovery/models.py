from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ResolutionStatus(str, Enum):
    RESOLVED = "resolved"
    RESOLVED_MULTIPLE = "resolved_multiple_publications"
    RESOLVED_AMBIGUOUS = "resolved_ambiguous_primary"
    LOW_CONFIDENCE = "publication_candidates_low_confidence"
    NO_PUBLICATION = "no_publication_link_found"
    OPENALEX_REPROCESS = "openalex_no_resolve_reprocess"
    API_ERROR = "api_error"
    NOT_YET_PROCESSED = "not_yet_processed"
    LIKELY_REANALYSIS_ONLY = "likely_reanalysis_only"


class MatchConfidence(str, Enum):
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class MgnifyStudy:
    mgnify_accession: str
    bioproject_accession: str | None = None
    secondary_study_accession: str | None = None
    study_name: str | None = None
    study_abstract: str | None = None
    centre_name: str | None = None
    public_release_date: str | None = None
    sample_count: int | None = None
    biome: str | None = None
    experiment_types: str | None = None
    mgnify_last_updated: str | None = None
    raw_json: str | None = None


@dataclass(frozen=True)
class EnaRun:
    run_accession: str
    experiment_accession: str | None = None
    sample_accession: str | None = None
    secondary_sample_accession: str | None = None
    study_accession: str | None = None
    secondary_study_accession: str | None = None
    bioproject_accession: str | None = None
    submission_accession: str | None = None
    study_title: str | None = None
    project_name: str | None = None
    centre_name: str | None = None
    first_public: str | None = None
    fastq_ftp: str | None = None
    fastq_md5: str | None = None
    fastq_bytes: str | None = None
    submitted_ftp: str | None = None
    submitted_md5: str | None = None
    submitted_bytes: str | None = None
    submitted_format: str | None = None
    sra_ftp: str | None = None
    sra_md5: str | None = None
    sra_bytes: str | None = None
    library_strategy: str | None = None
    library_source: str | None = None
    library_selection: str | None = None
    library_layout: str | None = None
    instrument_platform: str | None = None
    instrument_model: str | None = None
    target_gene: str | None = None
    collection_date: str | None = None
    lat: str | None = None
    lon: str | None = None
    depth: str | None = None
    country: str | None = None
    marine_region: str | None = None
    environment_biome: str | None = None
    environment_feature: str | None = None
    environment_material: str | None = None
    sample_collection: str | None = None
    extraction_protocol: str | None = None
    library_construction_protocol: str | None = None
    marine_tag: str | None = None
    marine_confidence: str | None = None
    marine_match_methods: str | None = None
    sequence_accessibility_status: str = "no_downloadable_reads"
    raw_json: str | None = None


@dataclass(frozen=True)
class EnaStudy:
    canonical_dataset_id: str
    ena_study_accession: str | None = None
    secondary_study_accession: str | None = None
    bioproject_accession: str | None = None
    bioproject_resolution_method: str | None = None
    bioproject_status: str = "unresolved"
    ncbi_bioproject_verified: bool = False
    study_title: str | None = None
    project_name: str | None = None
    centre_name: str | None = None
    first_public: str | None = None
    marine_confidence: str = "low"
    marine_match_methods: str | None = None
    marine_tags: str | None = None
    sample_count: int = 0
    run_count: int = 0
    downloadable_run_count: int = 0
    fastq_run_count: int = 0
    fastq_bytes_total: int = 0
    sequence_accessibility_status: str = "no_downloadable_reads"
    metadata_completeness_json: str | None = None
    metadata_usefulness_score: int = 0
    publication_resolution_status: str = ResolutionStatus.NOT_YET_PROCESSED.value
    raw_json: str | None = None


@dataclass(frozen=True)
class PublicationCandidate:
    doi: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    openalex_id: str | None = None
    title: str | None = None
    publication_date: str | None = None
    publication_year: int | None = None
    publication_type: str | None = None
    match_method: str = ""
    matched_identifier: str | None = None
    match_confidence: MatchConfidence = MatchConfidence.LOW
    match_score: float = 0.0
    raw_json: str | None = None
