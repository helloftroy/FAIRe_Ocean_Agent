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
from fair_ocean_agent.exports.faire import EMPTY_CLASSES, INTERNAL_STUDY_ID_FIELD, class_columns, export_faire
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
        rows = list(csv.DictReader(f))
    assert rows[0]["project_id"] == "PRJNA1"

    with (tmp_path / "sampleMetadata.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["samp_name"] == "SAMN1"
    assert rows[0]["geo_loc_name"] == "USA: California"

    with (tmp_path / "experimentRunMetadata.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["seq_run_id"] == "SRR1"
    assert rows[0]["samp_name"] == "SAMN1"
    assert rows[0]["associatedSequences"] == "SRR1"
    assert rows[0]["input_read_count"] == "1000"


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


def test_export_faire_column_order_matches_classes_yaml(db_session, tmp_path):
    export_faire(db_session, tmp_path)
    with (tmp_path / "sampleMetadata.csv").open() as f:
        header = next(csv.reader(f))
    # internal_study_id is a pipeline-internal traceability column, prepended
    # ahead of the real FAIRe columns -- not itself part of classes.yaml.
    assert header == [INTERNAL_STUDY_ID_FIELD, *class_columns("sampleMetadata")]


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
