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


class DataAvailabilityStatus(str, enum.Enum):
    """Whether this study's staged repository search (discovery/
    text_identifiers.py's Pass 1/2/3, see _has_accessible_sequence_data_signal
    in workflow/handlers.py) ever found real, accessible sequence data.
    NOT_ACCESSIBLE is the "give up, stop re-searching" signal -- per an
    explicit user request, checked by enqueue_seed_backfill/
    enqueue_full_rediscovery/enqueue_citation_rediscovery_backfill so a
    study that's already been searched with nothing found doesn't keep
    getting re-queued for the same search. Not set at all for a study whose
    discovery_trigger is "primer_reference_citation" -- that study's only
    goal was a primer sequence or the next reference in the chain, not full
    sample data, so the accessible-data question doesn't apply to it."""
    UNKNOWN = "unknown"
    ACCESSIBLE = "accessible"
    NOT_ACCESSIBLE = "not_accessible"


class IdentifierType(str, enum.Enum):
    DOI = "doi"
    PMID = "pmid"
    PMCID = "pmcid"
    OPENALEX_ID = "openalex_id"
    BIOPROJECT_ACCESSION = "bioproject_accession"
    BIOSAMPLE_ACCESSION = "biosample_accession"
    SRA_SUBMISSION_ACCESSION = "sra_submission_accession"
    SRA_STUDY_ACCESSION = "sra_study_accession"
    SRA_SAMPLE_ACCESSION = "sra_sample_accession"
    SRA_EXPERIMENT_ACCESSION = "sra_experiment_accession"
    SRA_RUN_ACCESSION = "sra_run_accession"
    SRA_ANALYSIS_ACCESSION = "sra_analysis_accession"
    ENA_STUDY_ACCESSION = "ena_study_accession"
    ASSEMBLY_ACCESSION = "assembly_accession"
    CNCB_PROJECT_ACCESSION = "cncb_project_accession"
    CNCB_BIOSAMPLE_ACCESSION = "cncb_biosample_accession"
    CNCB_STUDY_ACCESSION = "cncb_study_accession"
    CNCB_EXPERIMENT_ACCESSION = "cncb_experiment_accession"
    CNCB_RUN_ACCESSION = "cncb_run_accession"
    DATASET_DOI = "dataset_doi"
    OBIS_DATASET_UUID = "obis_dataset_uuid"
    GBIF_DATASET_KEY = "gbif_dataset_key"
    BCODMO_DATASET_ID = "bcodmo_dataset_id"
    PANGAEA_ID = "pangaea_id"
    QIITA_STUDY_ID = "qiita_study_id"
    MGRAST_PROJECT_ID = "mgrast_project_id"
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


# Entity levels that can be linked to more than one Study (via the
# `EntityStudy` join table, database/models.py) because they describe a
# physically invariant real-world object -- a BioSample, an experiment
# library, a sequencing run -- that a second paper reusing the same
# deposited data would cite unchanged. PROJECT stays single-study
# deliberately: a BioProject accession claimed by two studies is already
# handled by identity/resolution.py's merge/sibling-split machinery, and a
# paper's own bioinformatics-pipeline/assay-interpretation facts (PROJECT-
# level) are paper-specific, not physical -- making PROJECT shareable would
# create a second, competing mechanism for the same situation. ASSAY stays
# single-study too (interpretive, not physical). The remaining levels are
# typically unaccessioned (Entity.external_identifier null) so there's no
# stable matching key to share on regardless.
#
# The exact same three values are baked into
# database/models.py::Entity's partial unique index
# ("uq_entity_shareable_level_external_id") -- built FROM this constant,
# not duplicated as separate string literals, so the two can never drift.
SHAREABLE_ENTITY_LEVELS = frozenset({EntityLevel.SAMPLE, EntityLevel.EXPERIMENT_RUN, EntityLevel.SEQUENCING_RUN})


class EntityRootStatus(str, enum.Enum):
    """Which linked Study is the authoritative ("root") source for a
    shared Entity's broadcast-style (study-wide LLM/text) facts --
    identity/root_determination.py. Distinct from Entity.study_id ("home"),
    which is just whichever study's discovery happened to create the row
    first; root is a deliberate, evidence-based answer, decided only once
    every Study sharing this Entity has finished discovering (see
    ComponentStatus below)."""

    NOT_SHARED = "not_shared"  # exactly one linked study -- root by definition, no algorithm needed
    PENDING = "pending"  # shared, component not yet settled
    DETERMINED = "determined"  # root_study_id is authoritative
    AMBIGUOUS = "ambiguous"  # settled, but no signal could pick a root -- flagged for review, never guessed


class ComponentStatus(str, enum.Enum):
    """Study.entity_component_status: whether the connected component of
    Studies this Study belongs to (via shared entities and/or citation-
    discovery lineage -- identity/component.py) has stopped growing.
    MAP_FAIRE is deferred until SETTLED for any study with shareable
    entities (workflow/mapping_handlers.py), so root determination always
    runs against the full, final component membership, never a partial
    one."""

    NOT_APPLICABLE = "not_applicable"  # zero shareable-level entities -- never enters this machinery
    PENDING = "pending"  # still discovering / component may still grow
    SETTLED = "settled"  # no in-flight discovery work anywhere in the component
    STALLED = "stalled"  # exceeded max poll generations -- flagged rather than polled forever


class EntityRelationshipType(str, enum.Enum):
    DERIVED_FROM_SAMPLE = "derived_from_sample"
    USES_ASSAY = "uses_assay"
    SEQUENCED_IN_RUN = "sequenced_in_run"
    # identity/sample_alias_reconciliation.py: a within-study SAMPLE entity
    # created from a paper's own native naming (e.g. supplement-table
    # parsing, sources/supplement_parsing.py) that's been confidently
    # matched, via a real BioSample's own source_material_id cross-
    # reference, to the entity carrying the real accession for the same
    # physical sample. from_entity_id = the alias (native-named) entity,
    # to_entity_id = the canonical (accessioned) entity -- never the
    # reverse, so a canonical entity can be the target of more than one
    # alias without violating entity_relationships' own uq_entity_relationship
    # constraint.
    SAME_PHYSICAL_SAMPLE_AS = "same_physical_sample_as"


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
    DISCOVER_CITING_STUDIES = "DISCOVER_CITING_STUDIES"
    DISCOVER_PRIMER_REFERENCE_STUDIES = "DISCOVER_PRIMER_REFERENCE_STUDIES"
    CHECK_COMPONENT_SETTLED = "CHECK_COMPONENT_SETTLED"
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
