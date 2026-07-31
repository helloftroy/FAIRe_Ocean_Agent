import csv

import pytest

from fair_ocean_agent.database.enums import EntityLevel, EntityRelationshipType, IdentifierType, SupportType
from fair_ocean_agent.database.models import Entity, EntityRelationship, ExternalIdentifier, RawFact, Study
from fair_ocean_agent.exports.faire import EMPTY_CLASSES, class_columns, export_faire
from fair_ocean_agent.mapping.faire import map_study_to_faire


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

    counts = export_faire(db_session, tmp_path)

    assert counts["projectMetadata"] == 1
    assert counts["sampleMetadata"] == 1
    assert counts["experimentRunMetadata"] == 1
    for class_name in EMPTY_CLASSES:
        assert counts[class_name] == 0
        assert (tmp_path / f"{class_name}.csv").exists()

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
    assert header == class_columns("sampleMetadata")


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
