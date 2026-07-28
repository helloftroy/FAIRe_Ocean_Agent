import csv

from fair_ocean_agent.database.enums import EntityLevel, IdentifierType, SupportType
from fair_ocean_agent.database.models import Entity, ExternalIdentifier, RawFact, Study
from fair_ocean_agent.exports.faire import EMPTY_CLASSES, class_columns, export_faire
from fair_ocean_agent.mapping.faire import map_study_to_faire


def test_export_faire_writes_expected_files_and_rows(db_session, tmp_path):
    study = Study(title="Export test")
    db_session.add(study)
    db_session.flush()
    db_session.add(ExternalIdentifier(study_id=study.study_id, identifier_type=IdentifierType.BIOPROJECT_ACCESSION.value, identifier_value="PRJNA1"))
    sample = Entity(study_id=study.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMN1")
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

    counts = export_faire(db_session, tmp_path)

    assert counts["projectMetadata"] == 1
    assert counts["sampleMetadata"] == 1
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
