"""Controlled string-enum vocabularies used across the schema.

Plain str Enums (not native DB enum types) so SQLite and PostgreSQL behave
identically and new values can be added via a normal migration instead of an
`ALTER TYPE`.
"""
from __future__ import annotations

import enum


class ReviewStatus(str, enum.Enum):
    UNREVIEWED = "unreviewed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"


class CanonicalStatus(str, enum.Enum):
    CANONICAL = "canonical"
    MERGED = "merged"
    CANDIDATE = "candidate"
    REJECTED = "rejected"


class RelevanceStatus(str, enum.Enum):
    RELEVANT = "relevant"
    NOT_RELEVANT = "not_relevant"
    UNKNOWN = "unknown"


class IdentifierType(str, enum.Enum):
    DOI = "doi"
    PMID = "pmid"
    PMCID = "pmcid"
    OPENALEX_ID = "openalex_id"
    BIOPROJECT_ACCESSION = "bioproject_accession"
    BIOSAMPLE_ACCESSION = "biosample_accession"
    SRA_STUDY_ACCESSION = "sra_study_accession"
    ENA_STUDY_ACCESSION = "ena_study_accession"
    DATASET_DOI = "dataset_doi"
    OBIS_DATASET_UUID = "obis_dataset_uuid"
    GBIF_DATASET_KEY = "gbif_dataset_key"
    BCODMO_DATASET_ID = "bcodmo_dataset_id"
    PANGAEA_ID = "pangaea_id"
    NCEI_ACCESSION = "ncei_accession"
    CRUISE_ID = "cruise_id"
    URL = "url"
    OTHER = "other"


class RelationshipType(str, enum.Enum):
    IS_PUBLICATION_OF = "is_publication_of"
    IS_DATASET_FOR = "is_dataset_for"
    IS_SUPPLEMENT_TO = "is_supplement_to"
    IS_PROTOCOL_FOR = "is_protocol_for"
    IS_MIRROR_OF = "is_mirror_of"
    CITES = "cites"
    IS_CITED_BY = "is_cited_by"
    CONTAINS_SAMPLES_FROM = "contains_samples_from"
    RELATED_TO = "related_to"
    IS_HOME_OF = "is_home_of"
    SHARES_ACCESSION_WITH = "shares_accession_with"


class SourceType(str, enum.Enum):
    PUBLICATION_API = "publication_api"
    REPOSITORY_API = "repository_api"
    ARTICLE_FULLTEXT = "article_fulltext"
    ARTICLE_ABSTRACT = "article_abstract"
    SUPPLEMENT = "supplement"
    REPOSITORY_PAGE = "repository_page"
    PROTOCOL = "protocol"
    DATA_MANIFEST = "data_manifest"
    README = "readme"
    OTHER = "other"


class AccessStatus(str, enum.Enum):
    OPEN = "open"
    RESTRICTED = "restricted"
    EMBARGOED = "embargoed"
    UNKNOWN = "unknown"
    NOT_ACCESSIBLE = "not_accessible"


class InspectionStatus(str, enum.Enum):
    NOT_INSPECTED = "not_inspected"
    INSPECTED = "inspected"
    INSPECTION_FAILED = "inspection_failed"
    SKIPPED = "skipped"


class InspectionLevel(str, enum.Enum):
    NONE = "none"
    METADATA_ONLY = "metadata_only"
    LIGHTWEIGHT = "lightweight"
    FULL = "full"


class EntityLevel(str, enum.Enum):
    STUDY = "study"
    PROJECT = "project"
    SAMPLING_EVENT = "sampling_event"
    SAMPLE = "sample"
    ASSAY = "assay"
    EXPERIMENT_RUN = "experiment_run"
    SEQUENCING_RUN = "sequencing_run"
    PROTOCOL = "protocol"
    BIOINFORMATICS_WORKFLOW = "bioinformatics_workflow"
    DATA_ASSET = "data_asset"


class EntityRelationshipType(str, enum.Enum):
    DERIVED_FROM_SAMPLE = "derived_from_sample"
    USES_ASSAY = "uses_assay"
    SEQUENCED_IN_RUN = "sequenced_in_run"


class SupportType(str, enum.Enum):
    EXPLICIT = "explicit"
    STRUCTURED_SOURCE = "structured_source"
    DETERMINISTICALLY_DERIVED = "deterministically_derived"
    NORMALIZED = "normalized"
    INFERRED = "inferred"
    CONFLICTING = "conflicting"
    NOT_FOUND = "not_found"


class MissingnessStatus(str, enum.Enum):
    PRESENT = "present"
    NOT_FOUND_IN_INSPECTED_SOURCES = "not_found_in_inspected_sources"
    SOURCE_NOT_ACCESSIBLE = "source_not_accessible"
    RELEVANT_SOURCE_NOT_INSPECTED = "relevant_source_not_inspected"
    NOT_APPLICABLE = "not_applicable"
    CONFLICTING = "conflicting"
    MAPPING_UNRESOLVED = "mapping_unresolved"
    REPORTED_AMBIGUOUSLY = "reported_ambiguously"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"


class MappingMethod(str, enum.Enum):
    EXACT_IDENTIFIER = "exact_identifier"
    EXACT_LABEL = "exact_label"
    DETERMINISTIC_SYNONYM = "deterministic_synonym"
    SUGGESTED_SEMANTIC = "suggested_semantic"
    UNRESOLVED = "unresolved"


class ValidationStatus(str, enum.Enum):
    CONFIRMED = "confirmed"
    SUPPORTED = "supported"
    DERIVED_VALIDATED = "derived_validated"
    CONFLICTING = "conflicting"
    UNSUPPORTED = "unsupported"
    MISSING = "missing"
    NOT_ASSESSED = "not_assessed"


class ValidationSeverity(str, enum.Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AssetType(str, enum.Enum):
    RAW_SEQUENCE_FILES = "raw_sequence_files"
    SAMPLE_METADATA_TABLE = "sample_metadata_table"
    ENVIRONMENTAL_METADATA_TABLE = "environmental_metadata_table"
    ASV_TABLE = "asv_table"
    OTU_TABLE = "otu_table"
    FEATURE_TABLE = "feature_table"
    TAXONOMY_TABLE = "taxonomy_table"
    SEQUENCE_TABLE = "sequence_table"
    OCCURRENCE_TABLE = "occurrence_table"
    QPCR_DDPCR_RESULT_TABLE = "qpcr_ddpcr_result_table"
    BIOINFORMATICS_OUTPUT = "bioinformatics_output"
    CODE_REPOSITORY = "code_repository"
    PROTOCOL_FILE = "protocol_file"
    SUPPLEMENTARY_SPREADSHEET = "supplementary_spreadsheet"
    OTHER = "other"


class RawOrProcessed(str, enum.Enum):
    RAW = "raw"
    PROCESSED = "processed"
    UNKNOWN = "unknown"


class TaskType(str, enum.Enum):
    INGEST_SEED = "INGEST_SEED"
    DISCOVER_IDENTIFIERS = "DISCOVER_IDENTIFIERS"
    FETCH_SOURCE = "FETCH_SOURCE"
    BUILD_SOURCE_GRAPH = "BUILD_SOURCE_GRAPH"
    RESOLVE_STUDY_IDENTITY = "RESOLVE_STUDY_IDENTITY"
    EXTRACT_STRUCTURED_FACTS = "EXTRACT_STRUCTURED_FACTS"
    RETRIEVE_OPEN_TEXT = "RETRIEVE_OPEN_TEXT"
    SELECT_TEXT_SECTIONS = "SELECT_TEXT_SECTIONS"
    EXTRACT_TEXT_FACTS = "EXTRACT_TEXT_FACTS"
    DISCOVER_SUPPLEMENTS = "DISCOVER_SUPPLEMENTS"
    RETRIEVE_SUPPLEMENTS = "RETRIEVE_SUPPLEMENTS"
    EXTRACT_SUPPLEMENT_TEXT_FACTS = "EXTRACT_SUPPLEMENT_TEXT_FACTS"
    INVENTORY_DATA_ASSETS = "INVENTORY_DATA_ASSETS"
    MAP_FAIRE = "MAP_FAIRE"
    MAP_BEBOP = "MAP_BEBOP"
    VALIDATE_EVIDENCE = "VALIDATE_EVIDENCE"
    VALIDATE_LOGIC = "VALIDATE_LOGIC"
    VALIDATE_CROSS_SOURCE = "VALIDATE_CROSS_SOURCE"
    VALIDATE_FAIRE_COMPLETENESS = "VALIDATE_FAIRE_COMPLETENESS"
    REFRESH_STUDY_SOURCES = "REFRESH_STUDY_SOURCES"
    GENERATE_EXPORTS = "GENERATE_EXPORTS"
    GENERATE_RUN_REPORT = "GENERATE_RUN_REPORT"


class TaskStatus(str, enum.Enum):
    PENDING = "pending"
    CLAIMED = "claimed"
    RUNNING = "running"
    COMPLETED = "completed"
    RETRY_PENDING = "retry_pending"
    FAILED = "failed"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"
    CANCELLED = "cancelled"


class WorkflowRunStatus(str, enum.Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowRunType(str, enum.Enum):
    SEED_BACKFILL = "seed_backfill"
    HISTORICAL_BACKFILL = "historical_backfill"
    WEEKLY_UPDATE = "weekly_update"
    MANUAL = "manual"


class CandidateMatchMethod(str, enum.Enum):
    TITLE_SIMILARITY = "title_similarity"
    AUTHOR_OVERLAP = "author_overlap"
    COMPOSITE_SCORE = "composite_score"
    EXPLICIT_RELATIONSHIP = "explicit_relationship"


class CandidateMatchReviewStatus(str, enum.Enum):
    PENDING = "pending"
    CONFIRMED_MERGE = "confirmed_merge"
    REJECTED = "rejected"
