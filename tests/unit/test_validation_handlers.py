"""Integration tests for Milestone 5's task handlers. A live run of these
handlers against all 101 real studies in the seed database (before these
tests were written) surfaced a real bug -- see
test_populate_missingness_does_not_crash_on_real_shaped_data below -- so
several of these tests are deliberately shaped like that real data rather
than a minimal synthetic case, to make sure the same class of bug can't
come back unnoticed.
"""
from fair_ocean_agent.database.enums import (
    IdentifierType,
    MissingnessStatus,
    SupportType,
    TaskType,
    ValidationStatus,
)
from fair_ocean_agent.database.models import (
    DataAsset,
    Entity,
    ExternalIdentifier,
    RawFact,
    Source,
    StandardizedValue,
    Study,
    ValidationResult,
)
from fair_ocean_agent.llm.mock import MockLLMBackend
from fair_ocean_agent.workflow.task_queue import enqueue_task
import fair_ocean_agent.workflow.validation_handlers as validation_handlers_module
from fair_ocean_agent.workflow.validation_handlers import (
    CORE_SAMPLING_FIELDS,
    MISSINGNESS_TARGET_SCHEMA,
    enqueue_data_asset_and_validation_backfill,
    handle_inventory_data_assets,
    handle_validate_cross_source,
    handle_validate_evidence,
    handle_validate_logic_and_missingness,
    populate_missingness_for_study,
)


def _missingness_query(session, study_id, **filters):
    return session.query(StandardizedValue).filter_by(
        study_id=study_id, target_schema=MISSINGNESS_TARGET_SCHEMA, **filters
    )


def _study(session, **kwargs) -> Study:
    study = Study(**kwargs)
    session.add(study)
    session.flush()
    return study


def _task_for(session, study):
    task = enqueue_task(session, TaskType.VALIDATE_LOGIC, study_id=study.study_id)
    session.commit()
    return task


def test_populate_missingness_does_not_crash_on_real_shaped_data(db_session):
    """Regression test: a single-column select() (select(RawFact.fact_type_candidate))
    returns plain strings via session.scalars(), not RawFact objects --
    the original code did `f.fact_type_candidate for f in scalars(...)`,
    which raised AttributeError on every one of the 101 real studies that
    had a matching fact. This is the exact same data shape (a study with a
    real collection_date raw_fact) that a synthetic empty-study test would
    not have caught."""
    study = _study(db_session, title="Real-shaped study")
    source = Source(study_id=study.study_id, source_type="repository_api", source_name="ncbi_biosample")
    db_session.add(source)
    db_session.flush()
    db_session.add(
        RawFact(
            study_id=study.study_id,
            source_id=source.source_id,
            raw_field_name="collection_date",
            raw_value="2023-12-06",
            fact_type_candidate="collection_date",
            entity_level="sample",
            support_type=SupportType.STRUCTURED_SOURCE.value,
        )
    )
    db_session.commit()

    created = populate_missingness_for_study(db_session, study.study_id)  # must not raise
    db_session.commit()

    assert created == len(CORE_SAMPLING_FIELDS)
    collection_date_row = _missingness_query(db_session, study.study_id, target_field="collection_date").one()
    assert collection_date_row.missingness_status == MissingnessStatus.PRESENT.value
    assert collection_date_row.standardized_value is None


def test_populate_missingness_reports_not_found_when_sources_inspected_but_field_absent(db_session):
    study = _study(db_session, title="Inspected but nothing found")
    db_session.add(Source(study_id=study.study_id, source_type="publication_api", source_name="crossref"))
    db_session.commit()

    populate_missingness_for_study(db_session, study.study_id)
    db_session.commit()

    row = _missingness_query(db_session, study.study_id, target_field="depth").one()
    assert row.missingness_status == MissingnessStatus.NOT_FOUND_IN_INSPECTED_SOURCES.value


def test_populate_missingness_prefers_relevant_source_not_inspected_over_supplement(db_session):
    """Real fix for a real gap: before this, a study with *any* Source row
    at all (even an unrelated one, like a publication-metadata source) got
    NOT_FOUND_IN_INSPECTED_SOURCES for a missing field -- even if the one
    source that could plausibly have reported it (a discovered-but-not-yet-
    retrieved supplementary file) was never actually inspected. A
    supplement Source/DataAsset pair below "full" inspection must win over
    that fallback."""
    study = _study(db_session, title="Supplement referenced but not retrieved")
    db_session.add(Source(study_id=study.study_id, source_type="publication_api", source_name="crossref"))
    supplement_source = Source(
        study_id=study.study_id,
        source_type="supplement",
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
            inspection_level="metadata_only",  # referenced/available, not yet retrieved
            source_id=supplement_source.source_id,
        )
    )
    db_session.commit()

    populate_missingness_for_study(db_session, study.study_id)
    db_session.commit()

    row = _missingness_query(db_session, study.study_id, target_field="depth").one()
    assert row.missingness_status == MissingnessStatus.RELEVANT_SOURCE_NOT_INSPECTED.value


def test_populate_missingness_reports_source_not_accessible_for_inaccessible_supplement(db_session):
    study = _study(db_session, title="Supplement inaccessible")
    supplement_source = Source(
        study_id=study.study_id,
        source_type="supplement",
        source_name="europe_pmc_supplement",
        external_identifier="Data_Sheet_1.zip",
        inspection_status="inspected",
        inspection_level="metadata_only",
    )
    db_session.add(supplement_source)
    db_session.flush()
    db_session.add(
        DataAsset(
            study_id=study.study_id,
            asset_type="other",
            file_name="Data_Sheet_1.zip",
            access_status="not_accessible",
            inspection_level="metadata_only",
            source_id=supplement_source.source_id,
        )
    )
    db_session.commit()

    populate_missingness_for_study(db_session, study.study_id)
    db_session.commit()

    row = _missingness_query(db_session, study.study_id, target_field="depth").one()
    assert row.missingness_status == MissingnessStatus.SOURCE_NOT_ACCESSIBLE.value


def test_populate_missingness_ignores_fully_parsed_supplements(db_session):
    """A supplement that reached inspection_level="full" must not suppress
    the normal not-found/not-inspected logic -- its content was genuinely
    already looked at."""
    study = _study(db_session, title="Supplement fully parsed")
    db_session.add(Source(study_id=study.study_id, source_type="publication_api", source_name="crossref"))
    supplement_source = Source(
        study_id=study.study_id,
        source_type="supplement",
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
            access_status="open",
            inspection_level="full",
            source_id=supplement_source.source_id,
        )
    )
    db_session.commit()

    populate_missingness_for_study(db_session, study.study_id)
    db_session.commit()

    row = _missingness_query(db_session, study.study_id, target_field="depth").one()
    assert row.missingness_status == MissingnessStatus.NOT_FOUND_IN_INSPECTED_SOURCES.value


def test_populate_missingness_reports_source_not_inspected_when_no_sources_at_all(db_session):
    study = _study(db_session, title="Nothing attempted yet")
    db_session.commit()

    populate_missingness_for_study(db_session, study.study_id)
    db_session.commit()

    row = _missingness_query(db_session, study.study_id, target_field="depth").one()
    assert row.missingness_status == MissingnessStatus.RELEVANT_SOURCE_NOT_INSPECTED.value


def test_populate_missingness_is_idempotent(db_session):
    study = _study(db_session, title="Idempotency check")
    db_session.commit()

    first = populate_missingness_for_study(db_session, study.study_id)
    db_session.commit()
    second = populate_missingness_for_study(db_session, study.study_id)
    db_session.commit()

    assert first == len(CORE_SAMPLING_FIELDS)
    assert second == 0
    assert _missingness_query(db_session, study.study_id).count() == len(CORE_SAMPLING_FIELDS)


def test_handle_validate_logic_persists_results_for_matching_facts(db_session):
    study = _study(db_session, title="Has coordinates")
    source = Source(study_id=study.study_id, source_type="repository_api", source_name="ncbi_biosample")
    db_session.add(source)
    db_session.flush()
    db_session.add(
        RawFact(
            study_id=study.study_id,
            source_id=source.source_id,
            raw_field_name="lat_lon",
            raw_value="38.03 N 122.151667 W",
            fact_type_candidate="lat_lon",
            entity_level="sample",
            support_type=SupportType.STRUCTURED_SOURCE.value,
        )
    )
    db_session.commit()
    task = _task_for(db_session, study)

    handle_validate_logic_and_missingness(db_session, task)
    db_session.commit()

    results = db_session.query(ValidationResult).filter_by(study_id=study.study_id, validator_name="logical_validator").all()
    assert len(results) == 1
    assert results[0].status == ValidationStatus.CONFIRMED.value
    # missingness also gets populated in the same pass
    assert _missingness_query(db_session, study.study_id).count() == len(CORE_SAMPLING_FIELDS)


def test_handle_validate_logic_is_idempotent_on_retry(db_session):
    study = _study(db_session, title="Retry check")
    source = Source(study_id=study.study_id, source_type="repository_api", source_name="ncbi_biosample")
    db_session.add(source)
    db_session.flush()
    db_session.add(
        RawFact(
            study_id=study.study_id,
            source_id=source.source_id,
            raw_field_name="depth",
            raw_value="5 meters",
            fact_type_candidate="depth",
            entity_level="sample",
            support_type=SupportType.STRUCTURED_SOURCE.value,
        )
    )
    db_session.commit()
    task = _task_for(db_session, study)

    handle_validate_logic_and_missingness(db_session, task)
    db_session.commit()
    handle_validate_logic_and_missingness(db_session, task)  # simulated retry
    db_session.commit()

    assert db_session.query(ValidationResult).filter_by(study_id=study.study_id).count() == 1
    assert _missingness_query(db_session, study.study_id).count() == len(CORE_SAMPLING_FIELDS)


def test_handle_validate_logic_checks_accession_formats(db_session):
    study = _study(db_session, title="Has identifiers")
    db_session.add(
        ExternalIdentifier(
            study_id=study.study_id, identifier_type=IdentifierType.BIOPROJECT_ACCESSION.value, identifier_value="PRJNA123456"
        )
    )
    db_session.commit()
    task = _task_for(db_session, study)

    handle_validate_logic_and_missingness(db_session, task)
    db_session.commit()

    result = db_session.query(ValidationResult).filter_by(study_id=study.study_id, validator_name="accession_format").one()
    assert result.status == ValidationStatus.CONFIRMED.value

    handle_validate_logic_and_missingness(db_session, task)  # retry must not duplicate
    db_session.commit()
    assert db_session.query(ValidationResult).filter_by(validator_name="accession_format").count() == 1


def test_handle_validate_evidence_only_flags_failures(db_session):
    study = _study(db_session, title="Mixed evidence quality")
    source = Source(study_id=study.study_id, source_type="article_fulltext", source_name="europe_pmc_fulltext")
    db_session.add(source)
    db_session.flush()
    db_session.add(
        RawFact(
            study_id=study.study_id, source_id=source.source_id, raw_field_name="x", raw_value="y",
            fact_type_candidate="x", entity_level="study", support_type=SupportType.EXPLICIT.value,
            evidence_quote="a real quote",
        )
    )
    db_session.add(
        RawFact(
            study_id=study.study_id, source_id=source.source_id, raw_field_name="z", raw_value="w",
            fact_type_candidate="z", entity_level="study", support_type=SupportType.EXPLICIT.value,
            evidence_quote=None,
        )
    )
    db_session.commit()
    task = _task_for(db_session, study)

    handle_validate_evidence(db_session, task)
    db_session.commit()

    results = db_session.query(ValidationResult).filter_by(study_id=study.study_id, validator_name="evidence_consistency").all()
    assert len(results) == 1  # only the missing-evidence-quote fact gets flagged


def test_handle_validate_evidence_can_run_independent_llm_support_verifier(db_session, monkeypatch):
    study = _study(db_session, title="LLM verifier")
    source = Source(study_id=study.study_id, source_type="article_fulltext", source_name="europe_pmc_fulltext")
    db_session.add(source)
    db_session.flush()
    db_session.add_all(
        [
            RawFact(
                study_id=study.study_id,
                source_id=source.source_id,
                raw_field_name="annealing_temperature",
                raw_value="54 C",
                fact_type_candidate="annealing_temperature",
                entity_level="study",
                support_type=SupportType.EXPLICIT.value,
                extraction_method="llm_text_extraction",
                evidence_quote="Reactions were annealed at 54 C for 35 cycles.",
            ),
            RawFact(
                study_id=study.study_id,
                source_id=source.source_id,
                raw_field_name="annealing_temperature",
                raw_value="62 C",
                fact_type_candidate="annealing_temperature",
                entity_level="study",
                support_type=SupportType.EXPLICIT.value,
                extraction_method="llm_text_extraction",
                evidence_quote="Reactions were annealed at 54 C for 35 cycles.",
            ),
        ]
    )
    db_session.commit()
    task = _task_for(db_session, study)

    class Config:
        class Verifier:
            enabled = True
            max_output_tokens = 512

        llm_verifier = Verifier()

    verifier = MockLLMBackend(
        label="granite3.3:8b",
        responses=[
            '{"supported": true, "reason": "Quote states 54 C."}',
            '{"supported": false, "reason": "Quote states 54 C, not 62 C."}',
        ],
    )
    monkeypatch.setattr(validation_handlers_module, "load_config", lambda: Config())
    validation_handlers_module.reset_llm_verifier_backend_cache()
    validation_handlers_module._llm_verifier_backend_cache = verifier

    handle_validate_evidence(db_session, task)
    db_session.commit()

    rows = (
        db_session.query(ValidationResult)
        .filter_by(study_id=study.study_id, validator_name="llm_evidence_support:granite3.3:8b")
        .order_by(ValidationResult.status)
        .all()
    )
    assert {row.status for row in rows} == {ValidationStatus.SUPPORTED.value, ValidationStatus.UNSUPPORTED.value}
    assert len(verifier.calls) == 2

    handle_validate_evidence(db_session, task)
    db_session.commit()
    assert len(verifier.calls) == 2


def test_handle_validate_cross_source_persists_comparison(db_session):
    study = _study(db_session, title="Cross-source check")
    for source_name, title in (("crossref", "Same Title"), ("europe_pmc", "Same Title.")):
        source = Source(study_id=study.study_id, source_type="publication_api", source_name=source_name)
        db_session.add(source)
        db_session.flush()
        db_session.add(
            RawFact(
                study_id=study.study_id, source_id=source.source_id, raw_field_name="title", raw_value=title,
                fact_type_candidate="title", entity_level="study", support_type=SupportType.STRUCTURED_SOURCE.value,
            )
        )
    db_session.commit()
    task = _task_for(db_session, study)

    handle_validate_cross_source(db_session, task)
    db_session.commit()

    result = db_session.query(ValidationResult).filter_by(study_id=study.study_id, validator_name="cross_source_agreement").one()
    assert result.status == ValidationStatus.CONFIRMED.value

    handle_validate_cross_source(db_session, task)  # retry must not duplicate
    db_session.commit()
    assert db_session.query(ValidationResult).filter_by(validator_name="cross_source_agreement").count() == 1


def test_handle_inventory_data_assets_wraps_the_inventory_function(db_session):
    study = _study(db_session, title="Asset check")
    source = Source(study_id=study.study_id, source_type="repository_api", source_name="ena")
    db_session.add(source)
    db_session.flush()
    entity = Entity(study_id=study.study_id, entity_level="sequencing_run", external_identifier="SRR1")
    db_session.add(entity)
    db_session.flush()
    db_session.add(
        RawFact(
            study_id=study.study_id, entity_id=entity.entity_id, source_id=source.source_id,
            raw_field_name="fastq_ftp", raw_value="ftp.sra.ebi.ac.uk/vol1/fastq/SRR1/SRR1.fastq.gz",
            fact_type_candidate="fastq_ftp", entity_level="sequencing_run", support_type=SupportType.STRUCTURED_SOURCE.value,
        )
    )
    db_session.commit()
    task = _task_for(db_session, study)

    handle_inventory_data_assets(db_session, task)
    db_session.commit()

    assert db_session.query(DataAsset).filter_by(study_id=study.study_id).count() == 1


def test_enqueue_data_asset_and_validation_backfill_queues_one_of_each_per_study(db_session):
    study = _study(db_session, title="Backfill check")
    db_session.add(
        RawFact(
            study_id=study.study_id, raw_field_name="x", raw_value="y", fact_type_candidate="x",
            entity_level="study", support_type=SupportType.STRUCTURED_SOURCE.value,
        )
    )
    db_session.commit()

    counts = enqueue_data_asset_and_validation_backfill(db_session)
    db_session.commit()

    assert counts == {"inventory": 1, "logic": 1, "evidence": 1, "cross_source": 1}

    # idempotent -- re-running doesn't queue duplicate tasks
    from fair_ocean_agent.database.models import Task
    enqueue_data_asset_and_validation_backfill(db_session)
    db_session.commit()
    assert db_session.query(Task).count() == 4
