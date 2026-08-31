from __future__ import annotations

import sys
from pathlib import Path

from fair_ocean_agent.database.enums import (
    CanonicalStatus,
    EntityLevel,
    IdentifierType,
    SourceType,
    SupportType,
    TaskStatus,
    TaskType,
)
from fair_ocean_agent.database.models import (
    Entity,
    ExternalIdentifier,
    RawFact,
    Source,
    StandardizedValue,
    StandardizedValueEvidence,
    Study,
    Task,
)
from fair_ocean_agent.workflow.task_queue import enqueue_task

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from llm_troubleshooting_batch import report_study, reset_study, select_candidates  # noqa: E402


def _study(session, study_id: str, *, title: str = "t") -> Study:
    study = Study(study_id=study_id, title=title, canonical_status=CanonicalStatus.CANDIDATE.value)
    session.add(study)
    session.flush()
    return study


def _fully_qualified_study(session, study_id: str) -> None:
    _study(session, study_id)
    session.add(ExternalIdentifier(study_id=study_id, identifier_type=IdentifierType.DOI.value, identifier_value=f"10.1/{study_id}"))
    session.add(ExternalIdentifier(study_id=study_id, identifier_type=IdentifierType.PMCID.value, identifier_value=f"PMC{study_id}"))
    session.add(Entity(study_id=study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier=f"SAMN{study_id}"))
    session.flush()


def test_select_candidates_requires_doi_pmcid_and_sample_entity(db_session):
    _fully_qualified_study(db_session, "STUDY-1")

    _study(db_session, "STUDY-2")  # no DOI/PMCID/sample at all

    _study(db_session, "STUDY-3")
    db_session.add(ExternalIdentifier(study_id="STUDY-3", identifier_type=IdentifierType.DOI.value, identifier_value="10.1/3"))
    # no PMCID, no sample entity

    db_session.flush()

    result = select_candidates(db_session, count=10)

    assert result == ["STUDY-1"]


def test_select_candidates_excludes_already_queued_studies(db_session):
    _fully_qualified_study(db_session, "STUDY-4")
    enqueue_task(db_session, TaskType.EXTRACT_TEXT_FACTS, study_id="STUDY-4")

    result = select_candidates(db_session, count=10)

    assert result == []


def test_select_candidates_respects_count(db_session):
    for i in range(5):
        _fully_qualified_study(db_session, f"STUDY-{i}")

    result = select_candidates(db_session, count=3)

    assert len(result) == 3


def test_reset_study_clears_fulltext_output_but_keeps_structured_facts(db_session):
    _fully_qualified_study(db_session, "STUDY-5")

    fulltext_source = Source(study_id="STUDY-5", source_type=SourceType.ARTICLE_FULLTEXT.value, source_name="europe_pmc")
    structured_source = Source(study_id="STUDY-5", source_type=SourceType.REPOSITORY_API.value, source_name="ena")
    db_session.add_all([fulltext_source, structured_source])
    db_session.flush()

    llm_fact = RawFact(
        study_id="STUDY-5", source_id=fulltext_source.source_id, fact_type_candidate="temp", raw_value="4.2",
        extraction_method="llm_text_extraction", support_type=SupportType.INFERRED.value,
    )
    structured_fact = RawFact(
        study_id="STUDY-5", source_id=structured_source.source_id, fact_type_candidate="latitude", raw_value="10.5",
        extraction_method="ena_api", support_type=SupportType.STRUCTURED_SOURCE.value,
    )
    db_session.add_all([llm_fact, structured_fact])
    db_session.flush()

    standardized_value = StandardizedValue(study_id="STUDY-5", target_schema="faire", target_schema_version="1.0", target_field="temp")
    db_session.add(standardized_value)
    db_session.flush()
    db_session.add(StandardizedValueEvidence(standardized_value_id=standardized_value.standardized_value_id, fact_id=llm_fact.fact_id))

    task = enqueue_task(db_session, TaskType.EXTRACT_TEXT_FACTS, study_id="STUDY-5")
    task.status = TaskStatus.COMPLETED.value
    db_session.flush()

    reset_study(db_session, "STUDY-5")
    db_session.commit()

    assert db_session.get(Source, fulltext_source.source_id) is None
    assert db_session.get(RawFact, llm_fact.fact_id) is None
    assert db_session.get(StandardizedValue, standardized_value.standardized_value_id) is None
    assert db_session.query(Task).filter_by(task_type=TaskType.EXTRACT_TEXT_FACTS.value, study_id="STUDY-5").count() == 0

    # Structured-source data (from BioProject/BioSample resolution etc.) is untouched.
    assert db_session.get(Source, structured_source.source_id) is not None
    assert db_session.get(RawFact, structured_fact.fact_id) is not None


def test_reset_then_reenqueue_creates_a_fresh_pending_task(db_session):
    _fully_qualified_study(db_session, "STUDY-6")

    first_task = enqueue_task(db_session, TaskType.EXTRACT_TEXT_FACTS, study_id="STUDY-6")
    first_task.status = TaskStatus.COMPLETED.value
    db_session.flush()

    reset_study(db_session, "STUDY-6")
    db_session.flush()

    second_task = enqueue_task(db_session, TaskType.EXTRACT_TEXT_FACTS, study_id="STUDY-6")

    assert second_task.task_id != first_task.task_id
    assert second_task.status == TaskStatus.PENDING.value


def test_report_study_shows_structured_facts_separately_from_llm_facts(db_session, capsys):
    """Regression test for a real gap found live: the original report only
    ever looked at facts from the article_fulltext Source, so a user
    troubleshooting "why didn't a BioSample-derived field show up" had no
    way to see whether the structured (API-derived) RawFact even existed
    in the first place."""
    _study(db_session, "STUDY-7")
    structured_source = Source(study_id="STUDY-7", source_type=SourceType.REPOSITORY_API.value, source_name="ncbi_biosample")
    fulltext_source = Source(study_id="STUDY-7", source_type=SourceType.ARTICLE_FULLTEXT.value, source_name="europe_pmc")
    db_session.add_all([structured_source, fulltext_source])
    db_session.flush()
    db_session.add(RawFact(
        study_id="STUDY-7", source_id=structured_source.source_id, fact_type_candidate="lat_lon", raw_value="21.9 N 114 E",
        extraction_method="ncbi_biosample_api", support_type=SupportType.STRUCTURED_SOURCE.value,
    ))
    db_session.add(RawFact(
        study_id="STUDY-7", source_id=fulltext_source.source_id, fact_type_candidate="temp", raw_value="26 C",
        extraction_method="llm_text_extraction", support_type=SupportType.INFERRED.value,
    ))
    db_session.commit()

    report_study(db_session, "STUDY-7")

    output = capsys.readouterr().out
    assert "Structured (API-derived) RawFacts (1):" in output
    assert "lat_lon = '21.9 N 114 E'" in output
    assert "RawFacts from EXTRACT_TEXT_FACTS (1):" in output
    assert "temp = '26 C'" in output
