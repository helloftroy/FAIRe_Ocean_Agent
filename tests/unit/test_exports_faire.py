import csv

import pytest

from fair_ocean_agent.database.enums import EntityLevel, EntityRelationshipType, EntityRootStatus, IdentifierType, RelationshipType, SupportType
from fair_ocean_agent.database.models import (
    Entity,
    EntityRelationship,
    EntityStudy,
    ExternalIdentifier,
    RawFact,
    StandardizedValue,
    Study,
)
from fair_ocean_agent.exports.faire import (
    EMPTY_CLASSES,
    INTERNAL_ALIAS_SAMPLE_IDS_FIELD,
    INTERNAL_PRIMER_TRACEABILITY_FIELDS,
    INTERNAL_SECTION_DETECTION_FIELDS,
    INTERNAL_STUDY_ID_FIELD,
    PROJECT_METADATA_COLUMN_ORDER,
    SAMPLE_METADATA_COLUMN_ORDER,
    class_columns,
    export_faire,
)
from fair_ocean_agent.mapping.faire import TARGET_SCHEMA, TARGET_SCHEMA_VERSION, map_study_to_faire


def _home_entity_study(entity: Entity) -> EntityStudy:
    """Every production Entity gets a home entity_studies row AND its own
    root_status/root_study_id set eagerly at creation
    (identity/entity_linking.py::create_entity) -- direct Entity(...)
    construction in these fixtures bypasses both, so tests that need
    exports/faire.py's per-entity internal_study_id/broadcast-gating logic
    to see a real home link (and a real, determined root) must add one
    explicitly. Mutates `entity` in place (root fields) and returns the
    corresponding EntityStudy row for the caller to add separately."""
    entity.root_status = EntityRootStatus.DETERMINED.value
    entity.root_study_id = entity.study_id
    return EntityStudy(
        entity_id=entity.entity_id,
        study_id=entity.study_id,
        relationship_type=RelationshipType.IS_HOME_OF.value,
        confidence=SupportType.STRUCTURED_SOURCE.value,
    )


def _assert_preferred_header_order(header: list[str], preferred_order: tuple[str, ...]) -> None:
    preferred_present = [field for field in preferred_order if field in header]
    assert header[: len(preferred_present)] == preferred_present


def test_export_faire_writes_expected_files_and_rows(db_session, tmp_path):
    study = Study(title="Export test")
    db_session.add(study)
    db_session.flush()
    db_session.add(ExternalIdentifier(study_id=study.study_id, identifier_type=IdentifierType.BIOPROJECT_ACCESSION.value, identifier_value="PRJNA1"))
    sample = Entity(study_id=study.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMN1")
    run = Entity(study_id=study.study_id, entity_level=EntityLevel.SEQUENCING_RUN.value, external_identifier="SRR1")
    db_session.add_all([sample, run])
    db_session.flush()
    db_session.add(
        RawFact(
            study_id=study.study_id, entity_id=sample.entity_id, raw_field_name="geo_loc_name",
            raw_value="USA: California", fact_type_candidate="geo_loc_name", entity_level="sample",
            support_type=SupportType.STRUCTURED_SOURCE.value,
        )
    )
    db_session.add(
        RawFact(
            study_id=study.study_id, entity_id=run.entity_id, raw_field_name="read_count",
            raw_value="1000", fact_type_candidate="read_count", entity_level="sequencing_run",
            support_type=SupportType.STRUCTURED_SOURCE.value,
        )
    )
    db_session.add(
        RawFact(
            study_id=study.study_id, entity_id=run.entity_id, raw_field_name="run_accession",
            raw_value="SRR1", fact_type_candidate="run_accession", entity_level="sequencing_run",
            support_type=SupportType.STRUCTURED_SOURCE.value,
        )
    )
    db_session.add(
        RawFact(
            study_id=study.study_id, entity_id=run.entity_id, raw_field_name="sample_accession",
            raw_value="SAMN1", fact_type_candidate="sample_accession", entity_level="sequencing_run",
            support_type=SupportType.STRUCTURED_SOURCE.value,
        )
    )
    db_session.commit()
    map_study_to_faire(db_session, study.study_id)
    db_session.commit()
    for class_name in EMPTY_CLASSES:
        (tmp_path / f"{class_name}.csv").write_text("stale header-only file\n")

    counts = export_faire(db_session, tmp_path)

    assert counts["projectMetadata"] == 1
    assert counts["sampleMetadata"] == 1
    assert counts["experimentRunMetadata"] == 1
    for class_name in EMPTY_CLASSES:
        assert class_name not in counts
        assert not (tmp_path / f"{class_name}.csv").exists()

    with (tmp_path / "projectMetadata.csv").open() as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames is not None
        _assert_preferred_header_order(reader.fieldnames, PROJECT_METADATA_COLUMN_ORDER)
        rows = list(reader)
    assert rows[0]["project_id"] == "PRJNA1"

    with (tmp_path / "sampleMetadata.csv").open() as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames is not None
        _assert_preferred_header_order(reader.fieldnames, SAMPLE_METADATA_COLUMN_ORDER)
        rows = list(reader)
    assert rows[0]["samp_name"] == "SAMN1"
    assert rows[0]["geo_loc_name"] == "USA: California"

    with (tmp_path / "experimentRunMetadata.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["seq_run_id"] == "SRR1"
    assert rows[0]["samp_name"] == "SAMN1"
    assert rows[0]["associatedSequences"] == "SRR1"
    assert rows[0]["input_read_count"] == "1000"


def _minimal_mapped_study(session, *, bioproject: str, biosample: str, run_accession: str) -> Study:
    study = Study(title=f"Study {bioproject}")
    session.add(study)
    session.flush()
    session.add(ExternalIdentifier(study_id=study.study_id, identifier_type=IdentifierType.BIOPROJECT_ACCESSION.value, identifier_value=bioproject))
    sample = Entity(study_id=study.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier=biosample)
    run = Entity(study_id=study.study_id, entity_level=EntityLevel.SEQUENCING_RUN.value, external_identifier=run_accession)
    session.add_all([sample, run])
    session.flush()
    session.add(
        RawFact(
            study_id=study.study_id, entity_id=run.entity_id, raw_field_name="run_accession",
            raw_value=run_accession, fact_type_candidate="run_accession", entity_level="sequencing_run",
            support_type=SupportType.STRUCTURED_SOURCE.value,
        )
    )
    session.add(
        RawFact(
            study_id=study.study_id, entity_id=run.entity_id, raw_field_name="sample_accession",
            raw_value=biosample, fact_type_candidate="sample_accession", entity_level="sequencing_run",
            support_type=SupportType.STRUCTURED_SOURCE.value,
        )
    )
    session.commit()
    map_study_to_faire(session, study.study_id)
    session.commit()
    return study


def test_export_faire_study_ids_filter_scopes_to_just_those_studies(db_session, tmp_path):
    """A scoped export (e.g. a small LLM troubleshooting batch's own
    manifest) should only ever include the requested studies, even though
    the database has others."""
    kept = _minimal_mapped_study(db_session, bioproject="PRJNA100", biosample="SAMN100", run_accession="SRR100")
    _minimal_mapped_study(db_session, bioproject="PRJNA200", biosample="SAMN200", run_accession="SRR200")

    counts = export_faire(db_session, tmp_path, study_ids=[kept.study_id])

    assert counts["projectMetadata"] == 1
    with (tmp_path / "projectMetadata.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [row["project_id"] for row in rows] == ["PRJNA100"]

    with (tmp_path / "experimentRunMetadata.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [row["seq_run_id"] for row in rows] == ["SRR100"]


def test_export_faire_internal_study_id_traces_rows_across_two_studies(db_session, tmp_path):
    """export_faire() merges every study in the database into one shared
    set of output CSVs with no per-study filter -- internal_study_id is the
    only column that lets an outside observer trace which project/sample/
    experiment rows belong to the same study once more than one paper is
    being processed at once."""
    study_a = Study(title="Paper A")
    study_b = Study(title="Paper B")
    db_session.add_all([study_a, study_b])
    db_session.flush()
    db_session.add_all(
        [
            ExternalIdentifier(study_id=study_a.study_id, identifier_type=IdentifierType.BIOPROJECT_ACCESSION.value, identifier_value="PRJNA_A"),
            ExternalIdentifier(study_id=study_b.study_id, identifier_type=IdentifierType.BIOPROJECT_ACCESSION.value, identifier_value="PRJNA_B"),
        ]
    )
    sample_a = Entity(study_id=study_a.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMN_A")
    sample_b = Entity(study_id=study_b.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMN_B")
    run_a = Entity(study_id=study_a.study_id, entity_level=EntityLevel.SEQUENCING_RUN.value, external_identifier="SRR_A")
    run_b = Entity(study_id=study_b.study_id, entity_level=EntityLevel.SEQUENCING_RUN.value, external_identifier="SRR_B")
    db_session.add_all([sample_a, sample_b, run_a, run_b])
    db_session.flush()
    db_session.add_all(
        [_home_entity_study(e) for e in (sample_a, sample_b, run_a, run_b)]
    )
    db_session.flush()
    for study, sample, run in ((study_a, sample_a, run_a), (study_b, sample_b, run_b)):
        db_session.add(
            RawFact(
                study_id=study.study_id, entity_id=run.entity_id, raw_field_name="run_accession",
                raw_value=run.external_identifier, fact_type_candidate="run_accession", entity_level="sequencing_run",
                support_type=SupportType.STRUCTURED_SOURCE.value,
            )
        )
        db_session.add(
            RawFact(
                study_id=study.study_id, entity_id=run.entity_id, raw_field_name="sample_accession",
                raw_value=sample.external_identifier, fact_type_candidate="sample_accession", entity_level="sequencing_run",
                support_type=SupportType.STRUCTURED_SOURCE.value,
            )
        )
    db_session.commit()
    map_study_to_faire(db_session, study_a.study_id)
    map_study_to_faire(db_session, study_b.study_id)
    db_session.commit()

    export_faire(db_session, tmp_path)

    with (tmp_path / "projectMetadata.csv").open() as f:
        project_rows = {row["project_id"]: row for row in csv.DictReader(f)}
    assert project_rows["PRJNA_A"][INTERNAL_STUDY_ID_FIELD] == study_a.study_id
    assert project_rows["PRJNA_B"][INTERNAL_STUDY_ID_FIELD] == study_b.study_id

    with (tmp_path / "sampleMetadata.csv").open() as f:
        sample_rows = {row["samp_name"]: row for row in csv.DictReader(f)}
    assert sample_rows["SAMN_A"][INTERNAL_STUDY_ID_FIELD] == study_a.study_id
    assert sample_rows["SAMN_B"][INTERNAL_STUDY_ID_FIELD] == study_b.study_id

    with (tmp_path / "experimentRunMetadata.csv").open() as f:
        experiment_rows = {row["seq_run_id"]: row for row in csv.DictReader(f)}
    assert experiment_rows["SRR_A"][INTERNAL_STUDY_ID_FIELD] == study_a.study_id
    assert experiment_rows["SRR_B"][INTERNAL_STUDY_ID_FIELD] == study_b.study_id

    with (tmp_path / "field_reference.csv").open() as f:
        field_names = {row["faire_field"] for row in csv.DictReader(f)}
    assert INTERNAL_STUDY_ID_FIELD not in field_names
    assert INTERNAL_ALIAS_SAMPLE_IDS_FIELD not in field_names


def test_shared_sample_gets_pipe_joined_study_ids_and_no_broadcast_while_root_pending(db_session, tmp_path):
    """The actual point of the whole multi-study entity-sharing mechanism:
    a real BioSample two different papers both cite must appear exactly
    once in sampleMetadata.csv, with internal_study_id listing BOTH
    studies (pipe-joined) -- and while root determination is still
    PENDING (settle hasn't run yet), must NOT carry either study's own
    paper-specific broadcast default, since guessing which one is
    authoritative before it's actually decided would misrepresent it. Its
    own entity-level fact stays unconditional regardless."""
    study_a = Study(title="Original paper")
    study_b = Study(title="Reanalysis paper reusing the same data")
    db_session.add_all([study_a, study_b])
    db_session.flush()

    shared_sample = Entity(
        study_id=study_a.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMN_SHARED",
        root_status=EntityRootStatus.PENDING.value,
    )
    unshared_sample = Entity(
        study_id=study_a.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMN_UNSHARED",
        root_status=EntityRootStatus.DETERMINED.value, root_study_id=study_a.study_id,
    )
    db_session.add_all([shared_sample, unshared_sample])
    db_session.flush()
    db_session.add_all(
        [
            EntityStudy(
                entity_id=shared_sample.entity_id, study_id=study_a.study_id,
                relationship_type=RelationshipType.IS_HOME_OF.value, confidence=SupportType.STRUCTURED_SOURCE.value,
            ),
            EntityStudy(
                entity_id=shared_sample.entity_id, study_id=study_b.study_id,
                relationship_type=RelationshipType.SHARES_ACCESSION_WITH.value, confidence=SupportType.STRUCTURED_SOURCE.value,
            ),
            _home_entity_study(unshared_sample),
        ]
    )
    # study_a's own paper-specific broadcast default (e.g. an
    # interpretive env_broad_scale guess) -- must not leak onto the shared
    # sample's row, but must still appear on the unshared sample's row.
    db_session.add(
        StandardizedValue(
            study_id=study_a.study_id, entity_id=None, target_schema=TARGET_SCHEMA,
            target_schema_version=TARGET_SCHEMA_VERSION, target_field="env_broad_scale",
            standardized_value="marine biome", mapping_method="deterministic_synonym",
        )
    )
    # The shared sample's own entity-level fact -- always safe to show
    # regardless of how many studies link to it.
    db_session.add(
        StandardizedValue(
            study_id=study_a.study_id, entity_id=shared_sample.entity_id, target_schema=TARGET_SCHEMA,
            target_schema_version=TARGET_SCHEMA_VERSION, target_field="geo_loc_name",
            standardized_value="USA: California", mapping_method="exact_identifier",
        )
    )
    db_session.commit()

    export_faire(db_session, tmp_path)

    with (tmp_path / "sampleMetadata.csv").open() as f:
        rows = {row["samp_name"]: row for row in csv.DictReader(f)}

    assert set(rows) == {"SAMN_SHARED", "SAMN_UNSHARED"}, "shared sample must appear exactly once"

    shared_row = rows["SAMN_SHARED"]
    assert shared_row[INTERNAL_STUDY_ID_FIELD] == "|".join(sorted([study_a.study_id, study_b.study_id]))
    assert shared_row["env_broad_scale"] == "", "study_a's broadcast default must not leak onto a shared row"
    assert shared_row["geo_loc_name"] == "USA: California", "the entity's own fact is always shown"

    unshared_row = rows["SAMN_UNSHARED"]
    assert unshared_row[INTERNAL_STUDY_ID_FIELD] == study_a.study_id
    assert unshared_row["env_broad_scale"] == "marine biome", "single-study sample still gets its broadcast default"


def test_shared_sample_gets_roots_broadcast_once_determined(db_session, tmp_path):
    """Once identity/root_determination.py has settled on study_a as root,
    its broadcast default fills the shared sample's blanks -- study_b's own
    broadcast (even though it also links to the same sample) never does."""
    study_a = Study(title="Original paper, root")
    study_b = Study(title="Reanalysis paper reusing the same data")
    db_session.add_all([study_a, study_b])
    db_session.flush()

    shared_sample = Entity(
        study_id=study_a.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMN_ROOTED",
        root_status=EntityRootStatus.DETERMINED.value, root_study_id=study_a.study_id,
    )
    db_session.add(shared_sample)
    db_session.flush()
    db_session.add_all(
        [
            EntityStudy(
                entity_id=shared_sample.entity_id, study_id=study_a.study_id,
                relationship_type=RelationshipType.IS_HOME_OF.value, confidence=SupportType.STRUCTURED_SOURCE.value,
            ),
            EntityStudy(
                entity_id=shared_sample.entity_id, study_id=study_b.study_id,
                relationship_type=RelationshipType.SHARES_ACCESSION_WITH.value, confidence=SupportType.STRUCTURED_SOURCE.value,
            ),
        ]
    )
    db_session.add(
        StandardizedValue(
            study_id=study_a.study_id, entity_id=None, target_schema=TARGET_SCHEMA,
            target_schema_version=TARGET_SCHEMA_VERSION, target_field="env_broad_scale",
            standardized_value="marine biome (root study)", mapping_method="deterministic_synonym",
        )
    )
    db_session.add(
        StandardizedValue(
            study_id=study_b.study_id, entity_id=None, target_schema=TARGET_SCHEMA,
            target_schema_version=TARGET_SCHEMA_VERSION, target_field="env_broad_scale",
            standardized_value="marine biome (non-root study, must not show)", mapping_method="deterministic_synonym",
        )
    )
    db_session.commit()

    export_faire(db_session, tmp_path)

    with (tmp_path / "sampleMetadata.csv").open() as f:
        rows = {row["samp_name"]: row for row in csv.DictReader(f)}

    assert rows["SAMN_ROOTED"]["env_broad_scale"] == "marine biome (root study)"


def test_samp_mat_process_pipe_joins_entity_structured_value_with_study_broadcast(db_session, tmp_path):
    """Grounded in a real gap (10.3389/fmicb.2024.1295149): every real
    sample already has its own terse structured samp_mat_process value
    straight from NCBI ("DNA extraction from sediment samples"), which
    used to silently win outright and discard the study's richer
    paper-text broadcast entirely -- unlike every other sampleMetadata
    field (a genuinely different, unrelated value), this field is a
    free-text narrative where both sources are worth keeping side by
    side."""
    study = Study(title="MAG-adjacent sample prep detail")
    db_session.add(study)
    db_session.flush()
    sample = Entity(
        study_id=study.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMN_PREP",
        root_status=EntityRootStatus.DETERMINED.value, root_study_id=study.study_id,
    )
    db_session.add(sample)
    db_session.flush()
    db_session.add(
        EntityStudy(
            entity_id=sample.entity_id, study_id=study.study_id,
            relationship_type=RelationshipType.IS_HOME_OF.value, confidence=SupportType.STRUCTURED_SOURCE.value,
        )
    )
    db_session.add(
        StandardizedValue(
            study_id=study.study_id, entity_id=None, target_schema=TARGET_SCHEMA,
            target_schema_version=TARGET_SCHEMA_VERSION, target_field="samp_mat_process",
            standardized_value="the sub-sectioned sediment samples were freeze-dried in a freeze-dryer",
            mapping_method="suggested_semantic",
        )
    )
    db_session.add(
        StandardizedValue(
            study_id=study.study_id, entity_id=sample.entity_id, target_schema=TARGET_SCHEMA,
            target_schema_version=TARGET_SCHEMA_VERSION, target_field="samp_mat_process",
            standardized_value="DNA extraction from sediment samples", mapping_method="exact_label",
        )
    )
    db_session.commit()

    export_faire(db_session, tmp_path)

    with (tmp_path / "sampleMetadata.csv").open() as f:
        rows = {row["samp_name"]: row for row in csv.DictReader(f)}

    assert rows["SAMN_PREP"]["samp_mat_process"] == (
        "DNA extraction from sediment samples|"
        "the sub-sectioned sediment samples were freeze-dried in a freeze-dryer"
    )


def test_analysis_only_study_excluded_from_project_metadata(db_session, tmp_path):
    """A study that links to shared samples/runs but is home to none of
    them did no original data collection -- must not get a projectMetadata
    row, even though it has its own project_id and its own broadcast
    facts. Its Study/EntityStudy rows still exist (network membership is
    untouched); sampleMetadata/experimentRunMetadata already correctly
    never emit rows for entities it doesn't home."""
    root_study = Study(title="Original paper")
    analysis_only_study = Study(title="Pure reanalysis, no original data")
    db_session.add_all([root_study, analysis_only_study])
    db_session.flush()
    db_session.add(
        ExternalIdentifier(
            study_id=analysis_only_study.study_id,
            identifier_type=IdentifierType.BIOPROJECT_ACCESSION.value,
            identifier_value="PRJNA_ANALYSIS_ONLY",
        )
    )

    shared_sample = Entity(
        study_id=root_study.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMN_X",
        root_status=EntityRootStatus.DETERMINED.value, root_study_id=root_study.study_id,
    )
    db_session.add(shared_sample)
    db_session.flush()
    db_session.add_all(
        [
            EntityStudy(
                entity_id=shared_sample.entity_id, study_id=root_study.study_id,
                relationship_type=RelationshipType.IS_HOME_OF.value, confidence=SupportType.STRUCTURED_SOURCE.value,
            ),
            EntityStudy(
                entity_id=shared_sample.entity_id, study_id=analysis_only_study.study_id,
                relationship_type=RelationshipType.SHARES_ACCESSION_WITH.value, confidence=SupportType.STRUCTURED_SOURCE.value,
            ),
        ]
    )
    # analysis_only_study's own broadcast fact -- still must not produce a
    # projectMetadata row for it, despite having real data mapped.
    db_session.add(
        StandardizedValue(
            study_id=analysis_only_study.study_id, entity_id=None, target_schema=TARGET_SCHEMA,
            target_schema_version=TARGET_SCHEMA_VERSION, target_field="env_broad_scale",
            standardized_value="marine biome", mapping_method="deterministic_synonym",
        )
    )
    db_session.commit()

    export_faire(db_session, tmp_path)

    with (tmp_path / "projectMetadata.csv").open() as f:
        rows = list(csv.DictReader(f))
    internal_study_ids = {row[INTERNAL_STUDY_ID_FIELD] for row in rows}
    assert analysis_only_study.study_id not in internal_study_ids
    # The root study, which DOES home the shared sample, must still be unaffected.
    assert db_session.query(Study).filter_by(study_id=analysis_only_study.study_id).count() == 1
    assert db_session.query(EntityStudy).filter_by(study_id=analysis_only_study.study_id).count() == 1


def test_export_still_emits_one_project_row_when_assay_entity_has_no_direct_values(db_session, tmp_path):
    """Regression guard: extraction/experiment_runs.py's
    materialize_legacy_experiment_runs already creates ASSAY entities today
    purely as USES_ASSAY link targets for structured ENA/BioProject
    assay_name facts -- those facts land on the EXPERIMENT_RUN entity's
    row, never the assay entity itself, so such an assay entity has zero
    StandardizedValue rows of its own. This must NOT trigger the new
    one-row-per-assay export path; the study should still get exactly one
    broadcast projectMetadata row, same as before this change."""
    study = Study(title="Legacy assay-linkage only")
    db_session.add(study)
    db_session.flush()
    assay = Entity(study_id=study.study_id, entity_level=EntityLevel.ASSAY.value, external_identifier="16S-V4")
    run = Entity(study_id=study.study_id, entity_level=EntityLevel.SEQUENCING_RUN.value, external_identifier="SRR1")
    db_session.add_all([assay, run])
    db_session.flush()
    db_session.add(
        RawFact(
            study_id=study.study_id, entity_id=run.entity_id, raw_field_name="instrument_platform",
            raw_value="ILLUMINA", fact_type_candidate="instrument_platform", entity_level="sequencing_run",
            support_type=SupportType.STRUCTURED_SOURCE.value,
        )
    )
    db_session.commit()
    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    counts = export_faire(db_session, tmp_path)

    assert counts["projectMetadata"] == 1
    with (tmp_path / "projectMetadata.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["platform"] == "ILLUMINA"


def test_export_emits_one_project_row_per_assay_with_distinct_values(db_session, tmp_path):
    """Two distinct assays, each with their own annealing_temperature fact
    (extraction/text.py's assay_tag), must produce two distinct
    projectMetadata rows -- matching real FAIRe's own export layout of one
    row per assay_name (see schemas/faire/README.md)."""
    study = Study(title="Two assays")
    db_session.add(study)
    db_session.flush()
    assay_16s = Entity(study_id=study.study_id, entity_level=EntityLevel.ASSAY.value, external_identifier="16S-V3V4")
    assay_18s = Entity(study_id=study.study_id, entity_level=EntityLevel.ASSAY.value, external_identifier="18S-V9")
    db_session.add_all([assay_16s, assay_18s])
    db_session.flush()
    db_session.add(
        RawFact(
            study_id=study.study_id, entity_id=assay_16s.entity_id, raw_field_name="annealing_temperature",
            raw_value="55C", fact_type_candidate="annealing_temperature", entity_level="assay",
            support_type=SupportType.EXPLICIT.value,
        )
    )
    db_session.add(
        RawFact(
            study_id=study.study_id, entity_id=assay_18s.entity_id, raw_field_name="annealing_temperature",
            raw_value="60C", fact_type_candidate="annealing_temperature", entity_level="assay",
            support_type=SupportType.EXPLICIT.value,
        )
    )
    db_session.commit()
    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    counts = export_faire(db_session, tmp_path)

    assert counts["projectMetadata"] == 2
    with (tmp_path / "projectMetadata.csv").open() as f:
        rows = list(csv.DictReader(f))
    by_assay = {row["assay_name"]: row for row in rows}
    assert set(by_assay) == {"16S-V3V4", "18S-V9"}
    assert by_assay["16S-V3V4"]["annealingTemp"] == "55C"
    assert by_assay["18S-V9"]["annealingTemp"] == "60C"


def test_export_emits_one_library_row_each_when_libraries_share_a_sequencing_run(db_session, tmp_path):
    study = Study(title="Multiplexed libraries")
    db_session.add(study)
    db_session.flush()
    db_session.add(
        ExternalIdentifier(
            study_id=study.study_id,
            identifier_type=IdentifierType.BIOPROJECT_ACCESSION.value,
            identifier_value="PRJNA2",
        )
    )
    sample = Entity(
        study_id=study.study_id,
        entity_level=EntityLevel.SAMPLE.value,
        external_identifier="SAMN1",
    )
    assay = Entity(
        study_id=study.study_id,
        entity_level=EntityLevel.ASSAY.value,
        external_identifier="MiFish-12S",
    )
    run = Entity(
        study_id=study.study_id,
        entity_level=EntityLevel.SEQUENCING_RUN.value,
        external_identifier="RUN-2026-07-31-A",
    )
    libraries = [
        Entity(
            study_id=study.study_id,
            entity_level=EntityLevel.EXPERIMENT_RUN.value,
            external_identifier=lib_id,
        )
        for lib_id in ("LIB-A01", "LIB-A02")
    ]
    db_session.add_all([sample, assay, run, *libraries])
    db_session.flush()
    for library in libraries:
        library.parent_entity_id = sample.entity_id
        db_session.add_all(
            [
                EntityRelationship(
                    study_id=study.study_id,
                    from_entity_id=library.entity_id,
                    to_entity_id=sample.entity_id,
                    relationship_type=EntityRelationshipType.DERIVED_FROM_SAMPLE.value,
                ),
                EntityRelationship(
                    study_id=study.study_id,
                    from_entity_id=library.entity_id,
                    to_entity_id=assay.entity_id,
                    relationship_type=EntityRelationshipType.USES_ASSAY.value,
                ),
                EntityRelationship(
                    study_id=study.study_id,
                    from_entity_id=library.entity_id,
                    to_entity_id=run.entity_id,
                    relationship_type=EntityRelationshipType.SEQUENCED_IN_RUN.value,
                ),
            ]
        )
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()
    export_faire(db_session, tmp_path)

    with (tmp_path / "experimentRunMetadata.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert {row["lib_id"] for row in rows} == {"LIB-A01", "LIB-A02"}
    assert {row["samp_name"] for row in rows} == {"SAMN1"}
    assert {row["assay_name"] for row in rows} == {"MiFish-12S"}
    assert {row["seq_run_id"] for row in rows} == {"RUN-2026-07-31-A"}


def test_export_refuses_ambiguous_library_to_run_relationship(db_session, tmp_path):
    study = Study(title="Ambiguous library")
    db_session.add(study)
    db_session.flush()
    library = Entity(
        study_id=study.study_id,
        entity_level=EntityLevel.EXPERIMENT_RUN.value,
        external_identifier="LIB1",
    )
    run_a = Entity(
        study_id=study.study_id,
        entity_level=EntityLevel.SEQUENCING_RUN.value,
        external_identifier="RUN-A",
    )
    run_b = Entity(
        study_id=study.study_id,
        entity_level=EntityLevel.SEQUENCING_RUN.value,
        external_identifier="RUN-B",
    )
    db_session.add_all([library, run_a, run_b])
    db_session.flush()
    for run in (run_a, run_b):
        db_session.add(
            EntityRelationship(
                study_id=study.study_id,
                from_entity_id=library.entity_id,
                to_entity_id=run.entity_id,
                relationship_type=EntityRelationshipType.SEQUENCED_IN_RUN.value,
            )
        )
    db_session.commit()
    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    with pytest.raises(ValueError, match="multiple sequenced_in_run links"):
        export_faire(db_session, tmp_path)


def test_export_faire_sample_column_order_uses_preferred_human_readable_layout(db_session, tmp_path):
    from fair_ocean_agent.exports.faire import (
        CUSTOM_ENV_VAR_BLOCK_FIELD,
        CUSTOM_PULLED_ENV_VAR_FIELD,
        CUSTOM_SOURCE_UNMAPPED_FIELD,
        SAMPLE_METADATA_SUPPRESSED_FIELDS,
    )

    export_faire(db_session, tmp_path)
    with (tmp_path / "sampleMetadata.csv").open() as f:
        header = next(csv.reader(f))
    # Same exportable fields as before -- real class columns with suppressed
    # fields removed, plus pipeline-local/internal columns -- but arranged in
    # the human-readable order used for audit spreadsheets.
    expected_columns = [
        field for field in class_columns("sampleMetadata") if field not in SAMPLE_METADATA_SUPPRESSED_FIELDS
    ]
    exportable_columns = [
        INTERNAL_STUDY_ID_FIELD,
        INTERNAL_ALIAS_SAMPLE_IDS_FIELD,
        *expected_columns,
        CUSTOM_ENV_VAR_BLOCK_FIELD,
        CUSTOM_PULLED_ENV_VAR_FIELD,
        CUSTOM_SOURCE_UNMAPPED_FIELD,
    ]
    preferred_present = [field for field in SAMPLE_METADATA_COLUMN_ORDER if field in set(exportable_columns)]
    preferred_set = set(preferred_present)
    assert header == preferred_present + [field for field in exportable_columns if field not in preferred_set]
    assert not {
        "verbatimCoordinateSystem",
        "verbatimEventDate",
        "verbatimEventTime",
        "verbatimLatitude",
        "verbatimLongitude",
        "verbatimSRS",
        "habitat_natural_artificial_0_1",
        "ph_meth",
        "stationed_sample_dur",
        "tidal_stage",
        "turbidity",
        "water_current",
        "wind_direction",
        "wind_speed",
        "tot_carb_unit",
        "tot_diss_nitro_unit",
        "tot_inorg_nitro_unit",
        "tot_nitro_content_unit",
        "tot_nitro_unit",
        "tot_org_carb_unit",
        "tot_part_carb_unit",
        "org_carb_unit",
        "org_matter_unit",
        "org_nitro_unit",
        "part_org_carb_unit",
        "part_org_nitro_unit",
        "nitrate_unit",
        "nitrite_unit",
        "diss_inorg_carb_unit",
        "diss_inorg_nitro_unit",
        "diss_org_carb_unit",
        "diss_org_nitro_unit",
        "diss_oxygen_unit",
    } & set(header)
    assert "internal_expedition_id" in header


def test_export_faire_writes_field_reference_with_exact_mappings(db_session, tmp_path):
    export_faire(db_session, tmp_path)

    with (tmp_path / "field_reference.csv").open() as f:
        rows = {row["faire_field"]: row for row in csv.DictReader(f)}

    env = rows["env_broad_scale"]
    assert env["requirement_level_code"] == "M"
    assert "samp_category" in env["requirement_level_condition"]
    assert "mixs:0000012" in env["exact_mappings"]
    assert "sampleMetadata" in env["faire_classes"]

    # a field present in more than one class lists all of them
    assay_name = rows["assay_name"]
    assert "|" in assay_name["faire_classes"]


def test_export_faire_skips_studies_with_nothing_mapped(db_session, tmp_path):
    study = Study(title="Nothing mapped")
    db_session.add(study)
    db_session.commit()

    counts = export_faire(db_session, tmp_path)

    assert counts["projectMetadata"] == 0


def test_alias_and_canonical_sample_merge_with_pipe_joined_conflict(db_session, tmp_path):
    """A paper's own supplement-derived sample entity ("GC04_1") and the
    real BioSample accession for the same physical sample ("SAMN0007")
    fold into one sampleMetadata row -- identity/sample_alias_
    reconciliation.py is invoked automatically by map_study_to_faire. A
    genuine conflict on a shared field pipe-joins (canonical value first,
    per the user's own explicit request); a field present on only the
    alias passes through unchanged; the alias never emits its own row."""
    study = Study(title="Alias reconciliation export test")
    db_session.add(study)
    db_session.flush()
    canonical = Entity(study_id=study.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMN0007")
    alias = Entity(study_id=study.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="GC04_1")
    db_session.add_all([canonical, alias])
    db_session.flush()
    db_session.add_all([_home_entity_study(canonical), _home_entity_study(alias)])
    db_session.add_all(
        [
            # Marks `canonical` as a real, structurally-resolved BioSample --
            # the only signal reconcile_sample_aliases uses to distinguish
            # a canonical entity from an alias candidate.
            RawFact(
                study_id=study.study_id, entity_id=canonical.entity_id, raw_field_name="biosample_accession",
                raw_value="SAMN0007", fact_type_candidate="biosample_accession", entity_level="sample",
                support_type=SupportType.STRUCTURED_SOURCE.value, extraction_method="adapter:ncbi_biosample",
            ),
            RawFact(
                study_id=study.study_id, entity_id=canonical.entity_id, raw_field_name="source-material-id",
                raw_value="GS14-GC04-1", fact_type_candidate="source-material-id", entity_level="sample",
                support_type=SupportType.STRUCTURED_SOURCE.value, extraction_method="adapter:ncbi_biosample",
            ),
            RawFact(
                study_id=study.study_id, entity_id=canonical.entity_id, raw_field_name="geo_loc_name",
                raw_value="USA: California", fact_type_candidate="geo_loc_name", entity_level="sample",
                support_type=SupportType.STRUCTURED_SOURCE.value,
            ),
            RawFact(
                study_id=study.study_id, entity_id=alias.entity_id, raw_field_name="geo_loc_name",
                raw_value="USA: California: Monterey Bay", fact_type_candidate="geo_loc_name", entity_level="sample",
                support_type=SupportType.STRUCTURED_SOURCE.value,
            ),
            RawFact(
                study_id=study.study_id, entity_id=alias.entity_id, raw_field_name="env_broad_scale",
                raw_value="marine biome", fact_type_candidate="env_broad_scale", entity_level="sample",
                support_type=SupportType.STRUCTURED_SOURCE.value,
            ),
        ]
    )
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()
    export_faire(db_session, tmp_path)

    with (tmp_path / "sampleMetadata.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    assert row["samp_name"] == "SAMN0007"
    assert row[INTERNAL_ALIAS_SAMPLE_IDS_FIELD] == "GC04_1"
    assert row["geo_loc_name"] == "USA: California|USA: California: Monterey Bay"
    assert row["env_broad_scale"] == "marine biome"


def test_experiment_run_samp_name_resolves_through_canonical_alias_target(db_session, tmp_path):
    """A run whose DERIVED_FROM_SAMPLE link points at a supplement-derived
    alias entity (which no longer emits its own sampleMetadata row once
    reconciled) must still resolve samp_name to the real accession that
    row exists under."""
    study = Study(title="Run resolves through alias")
    db_session.add(study)
    db_session.flush()
    canonical = Entity(study_id=study.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMN0008")
    alias = Entity(study_id=study.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="GC05_1")
    run = Entity(study_id=study.study_id, entity_level=EntityLevel.EXPERIMENT_RUN.value, external_identifier="LIB-B01")
    db_session.add_all([canonical, alias, run])
    db_session.flush()
    db_session.add_all([_home_entity_study(canonical), _home_entity_study(alias)])
    db_session.add_all(
        [
            RawFact(
                study_id=study.study_id, entity_id=canonical.entity_id, raw_field_name="biosample_accession",
                raw_value="SAMN0008", fact_type_candidate="biosample_accession", entity_level="sample",
                support_type=SupportType.STRUCTURED_SOURCE.value, extraction_method="adapter:ncbi_biosample",
            ),
            RawFact(
                study_id=study.study_id, entity_id=canonical.entity_id, raw_field_name="source_material_id",
                raw_value="GS14-GC05-1", fact_type_candidate="source_material_id", entity_level="sample",
                support_type=SupportType.STRUCTURED_SOURCE.value, extraction_method="adapter:ncbi_biosample",
            ),
            EntityRelationship(
                study_id=study.study_id,
                from_entity_id=run.entity_id,
                to_entity_id=alias.entity_id,
                relationship_type=EntityRelationshipType.DERIVED_FROM_SAMPLE.value,
            ),
        ]
    )
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()
    export_faire(db_session, tmp_path)

    with (tmp_path / "experimentRunMetadata.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["samp_name"] == "SAMN0008"


def test_shared_experiment_row_inherits_assay_name_from_linked_study(db_session, tmp_path):
    home_study = Study(title="Original data paper")
    linked_study = Study(title="Amplicon reanalysis paper")
    db_session.add_all([home_study, linked_study])
    db_session.flush()

    experiment = Entity(
        study_id=home_study.study_id,
        entity_level=EntityLevel.EXPERIMENT_RUN.value,
        external_identifier="SRX_SHARED",
    )
    db_session.add(experiment)
    db_session.flush()
    db_session.add_all(
        [
            _home_entity_study(experiment),
            EntityStudy(
                entity_id=experiment.entity_id,
                study_id=linked_study.study_id,
                relationship_type=RelationshipType.SHARES_ACCESSION_WITH.value,
                confidence=SupportType.STRUCTURED_SOURCE.value,
            ),
            StandardizedValue(
                study_id=linked_study.study_id,
                entity_id=None,
                target_schema=TARGET_SCHEMA,
                target_schema_version=TARGET_SCHEMA_VERSION,
                target_field="assay_name",
                standardized_value="16S | hzsA | 515F/806R | hzsA_1597A/hzsA_1857R",
                mapping_method="deterministic_synonym",
            ),
        ]
    )
    db_session.commit()

    export_faire(db_session, tmp_path)

    with (tmp_path / "experimentRunMetadata.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert set(rows[0][INTERNAL_STUDY_ID_FIELD].split("|")) == {home_study.study_id, linked_study.study_id}
    assert rows[0]["assay_name"] == "16S"


def test_project_metadata_section_detection_columns_default_zero_and_flip_to_one(db_session, tmp_path):
    study = Study(title="Section detection export test")
    db_session.add(study)
    db_session.flush()
    db_session.add(ExternalIdentifier(study_id=study.study_id, identifier_type=IdentifierType.BIOPROJECT_ACCESSION.value, identifier_value="PRJNA9"))
    db_session.add(
        RawFact(
            study_id=study.study_id, entity_id=None, raw_field_name="sample_prep_0_1",
            raw_value="1", fact_type_candidate="sample_prep_0_1", entity_level="study",
            support_type=SupportType.DETERMINISTICALLY_DERIVED.value,
        )
    )
    db_session.commit()

    export_faire(db_session, tmp_path)

    with (tmp_path / "projectMetadata.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    assert row["sample_prep_0_1"] == "1"
    for field in INTERNAL_SECTION_DETECTION_FIELDS:
        if field != "sample_prep_0_1":
            assert row[field] == "0"

    with (tmp_path / "field_reference.csv").open() as f:
        field_names = {r["faire_field"] for r in csv.DictReader(f)}
    for field in INTERNAL_SECTION_DETECTION_FIELDS:
        assert field not in field_names


def test_primer_traceability_flags_when_name_known_but_sequence_unknown_anywhere(db_session, tmp_path):
    """A primer whose name is known but whose sequence isn't known either
    from this paper's own extraction OR from any other paper in the corpus
    (mapping/primer_library.py) is flagged unresolved -- per an explicit
    user request to track these as future reference-crawl candidates."""
    study = Study(title="Primer traceability, still unresolved")
    db_session.add(study)
    db_session.flush()
    db_session.add(ExternalIdentifier(study_id=study.study_id, identifier_type=IdentifierType.BIOPROJECT_ACCESSION.value, identifier_value="PRJNA_PRIMER1"))
    db_session.add(
        RawFact(
            study_id=study.study_id, entity_id=None, raw_field_name="pcr_primer_name_forward",
            raw_value="515F", fact_type_candidate="pcr_primer_name_forward", entity_level="study",
            support_type=SupportType.EXPLICIT.value,
        )
    )
    db_session.commit()

    export_faire(db_session, tmp_path)

    with (tmp_path / "projectMetadata.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["primer_forward_source_unresolved"] == "1"
    assert rows[0]["primer_reverse_source_unresolved"] == "0"


def test_primer_traceability_clears_when_a_reference_citation_is_found(db_session, tmp_path):
    """Per an explicit user request: unresolved means "no sequence AND no
    reference to chase either" -- a real dead end. A study that names a
    primer without its sequence but DOES cite where it came from (a real
    pcr_primer_reference_forward/reverse fact -- DOI, or a fallback title
    when the reference has no DOI, see extract_primer_reference_citations)
    has a genuine lead recorded, so it should NOT be flagged unresolved
    even before that lead is actually chased down to a real sequence."""
    study = Study(title="Primer traceability, resolved via reference citation")
    db_session.add(study)
    db_session.flush()
    db_session.add(ExternalIdentifier(study_id=study.study_id, identifier_type=IdentifierType.BIOPROJECT_ACCESSION.value, identifier_value="PRJNA_PRIMER_REF1"))
    db_session.add(
        RawFact(
            study_id=study.study_id, entity_id=None, raw_field_name="pcr_primer_name_forward",
            raw_value="515F", fact_type_candidate="pcr_primer_name_forward", entity_level="study",
            support_type=SupportType.EXPLICIT.value,
        )
    )
    db_session.add(
        RawFact(
            study_id=study.study_id, entity_id=None, raw_field_name="pcr_primer_reference_forward",
            raw_value="doi: 10.1234/some-other-paper", fact_type_candidate="pcr_primer_reference_forward",
            entity_level="study", support_type=SupportType.DETERMINISTICALLY_DERIVED.value,
        )
    )
    db_session.commit()

    export_faire(db_session, tmp_path)

    with (tmp_path / "projectMetadata.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["primer_forward_source_unresolved"] == "0"
    assert rows[0]["primer_reverse_source_unresolved"] == "0"


def test_primer_traceability_clears_once_this_paper_has_its_own_sequence(db_session, tmp_path):
    study = Study(title="Primer traceability, resolved by this paper")
    db_session.add(study)
    db_session.flush()
    db_session.add(ExternalIdentifier(study_id=study.study_id, identifier_type=IdentifierType.BIOPROJECT_ACCESSION.value, identifier_value="PRJNA_PRIMER2"))
    db_session.add(
        RawFact(
            study_id=study.study_id, entity_id=None, raw_field_name="pcr_primer_name_forward",
            raw_value="515F", fact_type_candidate="pcr_primer_name_forward", entity_level="study",
            support_type=SupportType.EXPLICIT.value,
        )
    )
    db_session.add(
        RawFact(
            study_id=study.study_id, entity_id=None, raw_field_name="pcr_primer_forward",
            raw_value="GTGYCAGCMGCCGCGGTAA", fact_type_candidate="pcr_primer_forward", entity_level="study",
            support_type=SupportType.EXPLICIT.value,
        )
    )
    db_session.commit()

    export_faire(db_session, tmp_path)

    with (tmp_path / "projectMetadata.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["primer_forward_source_unresolved"] == "0"

    with (tmp_path / "field_reference.csv").open() as f:
        field_names = {r["faire_field"] for r in csv.DictReader(f)}
    for field in INTERNAL_PRIMER_TRACEABILITY_FIELDS:
        assert field not in field_names


def test_primer_traceability_clears_via_corpus_lookup_from_a_different_study(db_session, tmp_path):
    """mapping/faire.py's map_study_to_faire calls resolve_primer_
    sequences_from_corpus as a pre-mapping step, so by export time a study
    that only names a primer should already have inherited the sequence
    from a different study that also names it, if one exists."""
    source_study = Study(title="Paper that gives the real sequence")
    db_session.add(source_study)
    db_session.flush()
    db_session.add(ExternalIdentifier(study_id=source_study.study_id, identifier_type=IdentifierType.BIOPROJECT_ACCESSION.value, identifier_value="PRJNA_PRIMER3"))
    db_session.add(
        RawFact(
            study_id=source_study.study_id, entity_id=None, raw_field_name="pcr_primer_name_forward",
            raw_value="515F", fact_type_candidate="pcr_primer_name_forward", entity_level="study",
            support_type=SupportType.EXPLICIT.value,
        )
    )
    db_session.add(
        RawFact(
            study_id=source_study.study_id, entity_id=None, raw_field_name="pcr_primer_forward",
            raw_value="GTGYCAGCMGCCGCGGTAA", fact_type_candidate="pcr_primer_forward", entity_level="study",
            support_type=SupportType.EXPLICIT.value,
        )
    )

    needs_it_study = Study(title="Paper that only names the primer")
    db_session.add(needs_it_study)
    db_session.flush()
    db_session.add(ExternalIdentifier(study_id=needs_it_study.study_id, identifier_type=IdentifierType.BIOPROJECT_ACCESSION.value, identifier_value="PRJNA_PRIMER4"))
    db_session.add(
        RawFact(
            study_id=needs_it_study.study_id, entity_id=None, raw_field_name="pcr_primer_name_forward",
            raw_value="515F", fact_type_candidate="pcr_primer_name_forward", entity_level="study",
            support_type=SupportType.EXPLICIT.value,
        )
    )
    db_session.commit()
    map_study_to_faire(db_session, source_study.study_id)
    map_study_to_faire(db_session, needs_it_study.study_id)
    db_session.commit()

    export_faire(db_session, tmp_path)

    with (tmp_path / "projectMetadata.csv").open() as f:
        rows = {r[INTERNAL_STUDY_ID_FIELD]: r for r in csv.DictReader(f)}
    needs_it_row = rows[needs_it_study.study_id]
    assert needs_it_row["primer_forward_source_unresolved"] == "0"
    assert needs_it_row["pcr_primer_forward"] == "GTGYCAGCMGCCGCGGTAA"


def test_project_metadata_does_not_export_information_withheld_llm_guess_column(db_session, tmp_path):
    """Legacy speculative information-withheld guesses are ignored even if
    an old raw_fact remains in the database."""
    study = Study(title="Removed information withheld guess export test")
    db_session.add(study)
    db_session.flush()
    db_session.add(ExternalIdentifier(study_id=study.study_id, identifier_type=IdentifierType.BIOPROJECT_ACCESSION.value, identifier_value="PRJNA10"))
    db_session.add(
        RawFact(
            study_id=study.study_id, entity_id=None, raw_field_name="information_withheld_llm_guess",
            raw_value="no code repository provided | no replicate count reported",
            fact_type_candidate="information_withheld_llm_guess", entity_level="study",
            support_type=SupportType.INFERRED.value,
        )
    )
    db_session.commit()
    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    export_faire(db_session, tmp_path)

    with (tmp_path / "projectMetadata.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert "internal_information_withheld_llm_guess" not in rows[0]
    # The real FAIRe field is untouched by the guess -- still just the
    # deterministic default since no real withheld-information fact exists.
    assert rows[0]["informationWithheld"] == "Nothing indicated as withheld"

    with (tmp_path / "field_reference.csv").open() as f:
        field_names = {r["faire_field"] for r in csv.DictReader(f)}
    assert "internal_information_withheld_llm_guess" not in field_names


def test_project_metadata_suppressed_fields_never_appear_as_columns(db_session, tmp_path):
    """institutionID/parent_project_id/project_name/recordedByID/mod_date/
    dataGeneralizations are real FAIRe fields (never removed from the
    vendored schema.yaml/classes.yaml mirror -- an explicit user
    instruction not to fabricate/corrupt real schema fidelity) but must
    never appear as columns in the exported CSV or field_reference.csv,
    per an explicit user instruction to never populate them at all."""
    from fair_ocean_agent.exports.faire import PROJECT_METADATA_SUPPRESSED_FIELDS

    study = Study(title="Suppressed field export test")
    db_session.add(study)
    db_session.flush()
    db_session.add(ExternalIdentifier(study_id=study.study_id, identifier_type=IdentifierType.BIOPROJECT_ACCESSION.value, identifier_value="PRJNA_SUPPRESS"))
    db_session.commit()

    export_faire(db_session, tmp_path)

    with (tmp_path / "projectMetadata.csv").open() as f:
        header = next(csv.reader(f))
    for field in PROJECT_METADATA_SUPPRESSED_FIELDS:
        assert field not in header
    # a real, still-populated field must still appear, confirming the
    # suppression is scoped to exactly these fields, not the whole class.
    assert "project_id" in header

    with (tmp_path / "field_reference.csv").open() as f:
        field_names = {r["faire_field"] for r in csv.DictReader(f)}
    for field in PROJECT_METADATA_SUPPRESSED_FIELDS:
        assert field not in field_names


def test_sample_metadata_suppressed_fields_never_appear_as_columns(db_session, tmp_path):
    """nucl_acid_ext/nucl_acid_ext_modify/date_ext/ratioOfAbsorbance260_280/
    prepped_samp_store_temp/prepped_samp_store_dur/prepped_samp_store_sol/
    dna_store_loc/size_frac_low/neg_cont_type/pos_cont_type/rel_cont_id/
    detected_notDetected/nitro/org_carb/org_nitro/tot_org_c_meth/
    tot_nitro_cont_meth/tot_nitro_content are real FAIRe fields (never
    removed from the vendored schema.yaml/classes.yaml mirror) but must
    never appear as columns in the exported CSV or field_reference.csv,
    per an explicit, repeated user instruction. (tot_depth_water_col was
    on this list too, but was un-suppressed -- see
    extraction/api_verification.py.)"""
    from fair_ocean_agent.exports.faire import SAMPLE_METADATA_SUPPRESSED_FIELDS

    study = Study(title="Sample-level suppressed field export test")
    db_session.add(study)
    db_session.flush()
    sample = Entity(study_id=study.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMN_SUPPRESS")
    db_session.add(sample)
    db_session.flush()
    db_session.add(
        RawFact(
            study_id=study.study_id, entity_id=sample.entity_id, raw_field_name="geo_loc_name",
            raw_value="USA: California", fact_type_candidate="geo_loc_name", entity_level="sample",
            support_type=SupportType.STRUCTURED_SOURCE.value,
        )
    )
    db_session.commit()
    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    export_faire(db_session, tmp_path)

    with (tmp_path / "sampleMetadata.csv").open() as f:
        header = next(csv.reader(f))
    for field in SAMPLE_METADATA_SUPPRESSED_FIELDS:
        assert field not in header
    # a real, still-populated field must still appear, confirming the
    # suppression is scoped to exactly these fields, not the whole class.
    assert "geo_loc_name" in header

    with (tmp_path / "field_reference.csv").open() as f:
        field_names = {r["faire_field"] for r in csv.DictReader(f)}
    for field in SAMPLE_METADATA_SUPPRESSED_FIELDS:
        assert field not in field_names


def test_experiment_run_metadata_suppressed_fields_never_appear_as_columns(db_session, tmp_path):
    """otu_num_tax_assigned/output_otu_num/output_read_count/mid_forward/
    mid_reverse/lib_conc/lib_conc_meth/lib_conc_unit are real FAIRe fields
    (never removed from the vendored schema.yaml/classes.yaml mirror) but
    must never appear as columns in the exported CSV or field_reference.csv,
    per an explicit, repeated user instruction. otu_num_tax_assigned/
    output_otu_num/output_read_count in particular are dropped rather than
    guessed at because they're computed differently study to study -- no
    single extraction approach would be correct across papers."""
    from fair_ocean_agent.exports.faire import EXPERIMENT_RUN_METADATA_SUPPRESSED_FIELDS

    study = Study(title="Experiment-run suppressed field export test")
    db_session.add(study)
    db_session.flush()
    run = Entity(study_id=study.study_id, entity_level=EntityLevel.SEQUENCING_RUN.value, external_identifier="SRR_SUPPRESS")
    db_session.add(run)
    db_session.flush()
    db_session.add(
        RawFact(
            study_id=study.study_id, entity_id=run.entity_id, raw_field_name="run_accession",
            raw_value="SRR_SUPPRESS", fact_type_candidate="run_accession", entity_level="sequencing_run",
            support_type=SupportType.STRUCTURED_SOURCE.value,
        )
    )
    db_session.commit()
    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    export_faire(db_session, tmp_path)

    with (tmp_path / "experimentRunMetadata.csv").open() as f:
        header = next(csv.reader(f))
    for field in EXPERIMENT_RUN_METADATA_SUPPRESSED_FIELDS:
        assert field not in header
    # a real, still-populated field must still appear, confirming the
    # suppression is scoped to exactly these fields, not the whole class.
    assert "seq_run_id" in header

    with (tmp_path / "field_reference.csv").open() as f:
        field_names = {r["faire_field"] for r in csv.DictReader(f)}
    for field in EXPERIMENT_RUN_METADATA_SUPPRESSED_FIELDS:
        assert field not in field_names


def test_expedition_id_and_ship_crs_expocode_no_longer_in_real_schema():
    """A former NOAA/SEUS-MBON extension, retracted per an explicit later
    user request -- unlike PROJECT_METADATA_SUPPRESSED_FIELDS's real
    upstream terms, these were never part of the public FAIRe v1.0.2
    checklist, so removing them from the vendored schema mirror entirely
    (not just suppressing their export column) doesn't corrupt schema
    fidelity."""
    assert "expedition_id" not in class_columns("projectMetadata")
    assert "ship_crs_expocode" not in class_columns("projectMetadata")
    assert "internal_expedition_id" in class_columns("sampleMetadata")


def test_checkls_ver_always_synced_as_the_pipelines_own_schema_version(db_session, tmp_path):
    study = Study(title="checkls_ver sync test")
    db_session.add(study)
    db_session.flush()
    db_session.add(ExternalIdentifier(study_id=study.study_id, identifier_type=IdentifierType.BIOPROJECT_ACCESSION.value, identifier_value="PRJNA_CKV"))
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()
    export_faire(db_session, tmp_path)

    with (tmp_path / "projectMetadata.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["checkls_ver"] == "1.0.2"


def test_in_situ_temp_salinity_export_suppresses_removed_env_columns(db_session, tmp_path):
    """Real audit (10.1093/ismejo/wrae013, STUDY-295abf4a8f43): a single
    STUDY-level in-situ reading (one collection event/site) must appear on
    every sample's own sampleMetadata row, exactly like other STUDY-level
    broadcast defaults. These legacy standalone environmental columns are
    now suppressed from export entirely, replaced by x_env_var_block, per
    explicit user requests."""
    study = Study(title="In-situ broadcast test")
    db_session.add(study)
    db_session.flush()
    sample1 = Entity(study_id=study.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMN1")
    sample2 = Entity(study_id=study.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMN2")
    db_session.add_all([sample1, sample2])
    db_session.flush()
    db_session.add_all([_home_entity_study(sample1), _home_entity_study(sample2)])
    db_session.add(
        RawFact(
            study_id=study.study_id, entity_id=None, raw_field_name="in_situ_temp", raw_value="6.5C",
            fact_type_candidate="in_situ_temp", entity_level="study", support_type=SupportType.EXPLICIT.value,
        )
    )
    db_session.add(
        RawFact(
            study_id=study.study_id, entity_id=None, raw_field_name="in_situ_salinity", raw_value="6.4 PSU",
            fact_type_candidate="in_situ_salinity", entity_level="study", support_type=SupportType.EXPLICIT.value,
        )
    )
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()
    export_faire(db_session, tmp_path)

    with (tmp_path / "sampleMetadata.csv").open() as f:
        rows = {row["samp_name"]: row for row in csv.DictReader(f)}
    assert "temp" not in rows["SAMN1"]
    assert "salinity" not in rows["SAMN1"]
    assert "diss_oxygen" not in rows["SAMN1"]
    assert "nitro_unit" not in rows["SAMN1"]
    assert "tot_inorg_nitro" not in rows["SAMN1"]


def test_x_pulled_env_var_broadcasts_into_every_sample_row_alongside_x_env_var_block(db_session, tmp_path):
    """x_pulled_env_var is a dedicated second pass over x_env_var_block's
    own quotes (extraction/section_category_extraction.py::
    extract_pulled_env_var_facts) -- per an explicit user request, it must
    land in its own separate column, broadcast to every sample row the
    same STUDY-level way x_env_var_block already does, without replacing
    or altering x_env_var_block itself."""
    study = Study(title="x_pulled_env_var export test")
    db_session.add(study)
    db_session.flush()
    sample = Entity(study_id=study.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMN_ENV")
    db_session.add(sample)
    db_session.flush()
    db_session.add(_home_entity_study(sample))
    db_session.add(
        RawFact(
            study_id=study.study_id, entity_id=None, raw_field_name="x_env_var_block", raw_value="raw broadcast text",
            fact_type_candidate="x_env_var_block", entity_level="study", support_type=SupportType.EXPLICIT.value,
        )
    )
    db_session.add(
        RawFact(
            study_id=study.study_id, entity_id=None, raw_field_name="x_pulled_env_var",
            raw_value="temperature = 6.5°C | salinity = 6.4 PSU",
            fact_type_candidate="x_pulled_env_var", entity_level="study", support_type=SupportType.EXPLICIT.value,
        )
    )
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()
    export_faire(db_session, tmp_path)

    with (tmp_path / "sampleMetadata.csv").open() as f:
        rows = {row["samp_name"]: row for row in csv.DictReader(f)}
    assert rows["SAMN_ENV"]["x_env_var_block"] == "raw broadcast text"
    assert rows["SAMN_ENV"]["x_pulled_env_var"] == "temperature = 6.5°C | salinity = 6.4 PSU"


def test_api_paper_corrections_csv_includes_paper_reference_and_correction_details(db_session, tmp_path):
    """The durable "fixes" spreadsheet an explicit user request asked for:
    every api_paper_corrections row must surface with its paper's own DOI
    (not the internal study_id) as paper_reference, alongside the API
    term/value, the paper-corrected term/value, and the supporting quote."""
    from fair_ocean_agent.database.models import ApiPaperCorrection

    study = Study(title="Corrections export test")
    db_session.add(study)
    db_session.flush()
    db_session.add(ExternalIdentifier(study_id=study.study_id, identifier_type=IdentifierType.DOI.value, identifier_value="10.1093/ismejo/wrae013"))
    db_session.add(
        ApiPaperCorrection(
            study_id=study.study_id,
            entity_id=None,
            api_faire_term="elev",
            api_value="34 m",
            corrected_faire_term="tot_depth_water_col",
            corrected_value="34",
            supporting_quote="at a site with 34 m water depth",
            detector="elev_depth_mislabel_check",
        )
    )
    db_session.commit()

    export_faire(db_session, tmp_path)

    with (tmp_path / "api_paper_corrections.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    assert row["paper_reference"] == "10.1093/ismejo/wrae013"
    assert row["api_faire_term"] == "elev"
    assert row["api_value"] == "34 m"
    assert row["corrected_faire_term"] == "tot_depth_water_col"
    assert row["corrected_value"] == "34"
    assert row["supporting_quote"] == "at a site with 34 m water depth"
    assert row["detector"] == "elev_depth_mislabel_check"


def test_api_paper_corrections_csv_falls_back_to_study_id_without_a_doi(db_session, tmp_path):
    from fair_ocean_agent.database.models import ApiPaperCorrection

    study = Study(title="Corrections export, no DOI")
    db_session.add(study)
    db_session.flush()
    db_session.add(
        ApiPaperCorrection(
            study_id=study.study_id,
            entity_id=None,
            api_faire_term="elev",
            api_value="10 m",
            corrected_faire_term="tot_depth_water_col",
            corrected_value="10",
            supporting_quote="some quote",
            detector="elev_depth_mislabel_check",
        )
    )
    db_session.commit()

    export_faire(db_session, tmp_path)

    with (tmp_path / "api_paper_corrections.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["paper_reference"] == study.study_id
