from fair_ocean_agent.database.enums import EntityLevel, IdentifierType, MissingnessStatus, SourceType, SupportType, TaskType
from fair_ocean_agent.database.models import DataAsset, Entity, ExternalIdentifier, RawFact, Source, StandardizedValue, Study, Task
from fair_ocean_agent.mapping.faire import map_study_to_faire
from fair_ocean_agent.workflow.task_queue import enqueue_task
from fair_ocean_agent.workflow.validation_handlers import (
    enqueue_faire_completeness_backfill,
    handle_validate_faire_completeness,
    populate_faire_missingness_for_study,
)


def test_populate_faire_missingness_flags_known_real_gaps(db_session):
    """Regression guard matching this pipeline's real, documented gap:
    assay_name and samp_category have no data source anywhere in this
    pipeline, so they must always come back NOT_FOUND_IN_INSPECTED_SOURCES,
    never silently PRESENT."""
    study = Study(title="Completeness check")
    db_session.add(study)
    db_session.flush()
    db_session.add(Source(study_id=study.study_id, source_type=SourceType.REPOSITORY_API.value, source_name="ncbi_bioproject"))
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

    populate_faire_missingness_for_study(db_session, study.study_id)
    db_session.commit()

    rows = {
        (m.target_field, m.entity_id): m.missingness_status
        for m in db_session.query(StandardizedValue).filter_by(study_id=study.study_id, target_schema="faire").all()
    }
    # project-level: project_id resolves via ExternalIdentifier -> present
    assert rows[("project_id", None)] == MissingnessStatus.PRESENT.value
    # project_contact has no data source anywhere -- genuinely missing
    assert rows[("project_contact", None)] == MissingnessStatus.NOT_FOUND_IN_INSPECTED_SOURCES.value
    # sample-level: geo_loc_name was mapped -> present; samp_name from Entity -> present
    assert rows[("geo_loc_name", sample.entity_id)] == MissingnessStatus.PRESENT.value
    assert rows[("samp_name", sample.entity_id)] == MissingnessStatus.PRESENT.value
    # assay_name/samp_category have no rule anywhere in mapping/rules.py
    assert rows[("assay_name", sample.entity_id)] == MissingnessStatus.NOT_FOUND_IN_INSPECTED_SOURCES.value
    assert rows[("samp_category", sample.entity_id)] == MissingnessStatus.NOT_FOUND_IN_INSPECTED_SOURCES.value


def test_populate_faire_missingness_reports_not_inspected_when_no_source_exists(db_session):
    """Real bug found during a 100-study audit: a study whose only relevant
    source was never actually inspected (e.g. a closed-access paper Europe
    PMC 404s on, or a study that simply hasn't reached discovery yet) used
    to get NOT_FOUND_IN_INSPECTED_SOURCES for every missing field -- wrongly
    implying sources were checked and the field wasn't there. A study with
    zero Source rows at all must get RELEVANT_SOURCE_NOT_INSPECTED instead,
    the same distinction populate_missingness_for_study already makes for
    core_sampling_metadata."""
    study = Study(title="No sources inspected yet")
    db_session.add(study)
    db_session.commit()

    populate_faire_missingness_for_study(db_session, study.study_id)
    db_session.commit()

    rows = {
        m.target_field: m.missingness_status
        for m in db_session.query(StandardizedValue).filter_by(study_id=study.study_id, target_schema="faire", entity_id=None).all()
    }
    assert rows["project_contact"] == MissingnessStatus.RELEVANT_SOURCE_NOT_INSPECTED.value


def test_populate_faire_missingness_prefers_supplement_status_over_not_found(db_session):
    """Same real fix as populate_missingness_for_study's own version: a
    study with an unrelated Source (e.g. a publication-metadata source)
    plus a referenced-but-not-yet-parsed supplementary file must not report
    NOT_FOUND_IN_INSPECTED_SOURCES for a missing field -- the supplement
    might still resolve it once RETRIEVE_SUPPLEMENTS actually runs."""
    study = Study(title="Supplement referenced, not yet parsed")
    db_session.add(study)
    db_session.flush()
    db_session.add(Source(study_id=study.study_id, source_type=SourceType.PUBLICATION_API.value, source_name="crossref"))
    supplement_source = Source(
        study_id=study.study_id,
        source_type=SourceType.SUPPLEMENT.value,
        source_name="europe_pmc_supplement",
        external_identifier="Table_5.xlsx",
        inspection_status="inspected",
        inspection_level="metadata_only",
    )
    db_session.add(supplement_source)
    db_session.flush()
    db_session.add(
        DataAsset(
            study_id=study.study_id,
            asset_type="supplementary_spreadsheet",
            file_name="Table_5.xlsx",
            access_status="unknown",
            inspection_level="metadata_only",
            source_id=supplement_source.source_id,
        )
    )
    db_session.commit()

    populate_faire_missingness_for_study(db_session, study.study_id)
    db_session.commit()

    rows = {
        m.target_field: m.missingness_status
        for m in db_session.query(StandardizedValue).filter_by(study_id=study.study_id, target_schema="faire", entity_id=None).all()
    }
    assert rows["project_contact"] == MissingnessStatus.RELEVANT_SOURCE_NOT_INSPECTED.value


def test_populate_faire_missingness_is_idempotent(db_session):
    study = Study(title="Idempotency check")
    db_session.add(study)
    db_session.commit()

    first = populate_faire_missingness_for_study(db_session, study.study_id)
    db_session.commit()
    second = populate_faire_missingness_for_study(db_session, study.study_id)
    db_session.commit()

    assert first > 0
    assert second == 0


def test_handle_validate_faire_completeness_wraps_populate(db_session):
    study = Study(title="Handler test")
    db_session.add(study)
    db_session.commit()
    task = enqueue_task(db_session, TaskType.VALIDATE_FAIRE_COMPLETENESS, study_id=study.study_id)
    db_session.commit()

    handle_validate_faire_completeness(db_session, task)
    db_session.commit()

    assert db_session.query(StandardizedValue).filter_by(study_id=study.study_id, target_schema="faire").count() > 0


def test_enqueue_faire_completeness_backfill_queues_studies_with_raw_facts(db_session):
    mapped_study = Study(title="Has FAIRe values")
    unmapped_study = Study(title="Raw facts but no FAIRe values yet")
    untouched_study = Study(title="No extracted facts yet")
    db_session.add_all([mapped_study, unmapped_study, untouched_study])
    db_session.flush()
    for study in (mapped_study, unmapped_study):
        db_session.add(
            RawFact(
                study_id=study.study_id,
                raw_field_name="title",
                raw_value=study.title,
                fact_type_candidate="title",
                entity_level=EntityLevel.STUDY.value,
                support_type=SupportType.STRUCTURED_SOURCE.value,
            )
        )
    db_session.add(
        StandardizedValue(
            study_id=mapped_study.study_id, target_schema="faire", target_schema_version="1.0.2",
            target_field="project_id", standardized_value="PRJNA1",
        )
    )
    db_session.commit()

    count = enqueue_faire_completeness_backfill(db_session)
    db_session.commit()

    assert count == 2
    task_study_ids = {t.study_id for t in db_session.query(Task).all()}
    assert task_study_ids == {mapped_study.study_id, unmapped_study.study_id}
