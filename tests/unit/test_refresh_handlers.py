"""Tests for Milestone 7's REFRESH_STUDY_SOURCES handler -- fake, in-
process adapters (same FakeAdapter pattern as test_handlers.py), no
network. Focuses on what's genuinely new here versus DISCOVER_IDENTIFIERS:
content_hash-based change detection, parent_source_id linking, and
watermark bookkeeping.
"""
from datetime import datetime, timezone

from fair_ocean_agent.database.enums import IdentifierType, SourceType, TaskType
from fair_ocean_agent.database.models import ExternalIdentifier, RawFact, Source, SourceWatermark, Study
from fair_ocean_agent.identity.identifiers import normalize_doi
from fair_ocean_agent.sources.base import RawFactCandidate, SourceRecord
from fair_ocean_agent.workflow import handlers, refresh_handlers
from fair_ocean_agent.workflow.refresh_handlers import (
    enqueue_refresh_backfill,
    handle_refresh_study_sources,
    refresh_study,
)
from fair_ocean_agent.workflow.task_queue import enqueue_task


class FakeAdapter:
    def __init__(self, name, record, facts=None, related=None, publication_fields=None):
        self.name = name
        self._record = record
        self._facts = facts or []
        self._related = related or []
        self._publication_fields = publication_fields or {}

    def fetch_record(self, identifier):
        return self._record

    def extract_structured_facts(self, record):
        return self._facts

    def parse_publication_fields(self, record):
        return self._publication_fields

    def find_related(self, record):
        return self._related


def _record(source_name: str, content_hash: str) -> SourceRecord:
    return SourceRecord(
        source_name=source_name,
        external_identifier="10.1234/x",
        url=f"https://example.org/{source_name}",
        raw={"stub": True},
        retrieved_at=datetime.now(timezone.utc),
        content_hash=content_hash,
    )


def _study_with_doi(session, doi="10.1234/x") -> Study:
    study = Study(title=None)
    session.add(study)
    session.flush()
    session.add(
        ExternalIdentifier(study_id=study.study_id, identifier_type=IdentifierType.DOI.value, identifier_value=normalize_doi(doi))
    )
    session.flush()
    return study


def test_refresh_creates_new_source_and_facts_when_none_existed(db_session, monkeypatch):
    study = _study_with_doi(db_session)
    fake = FakeAdapter("crossref", _record("crossref", "hash-1"), facts=[
        RawFactCandidate(entity_level="study", fact_type_candidate="title", raw_field_name="title", raw_value="v1", support_type="structured_source", source_locator="")
    ])
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {"crossref": fake})

    outcome = refresh_study(db_session, study, run_id="RUN-1")
    db_session.commit()

    assert outcome.checked == 1
    assert outcome.changed == 1
    sources = db_session.query(Source).filter_by(study_id=study.study_id).all()
    assert len(sources) == 1
    assert sources[0].parent_source_id is None
    assert db_session.query(RawFact).filter_by(study_id=study.study_id).count() == 1

    watermark = db_session.query(SourceWatermark).filter_by(source_name="crossref", query_identifier="10.1234/x").one()
    assert watermark.last_status == "changed"
    assert watermark.last_run_id == "RUN-1"


def test_refresh_with_unchanged_content_hash_creates_nothing_new(db_session, monkeypatch):
    study = _study_with_doi(db_session)
    fake = FakeAdapter("crossref", _record("crossref", "hash-1"), facts=[
        RawFactCandidate(entity_level="study", fact_type_candidate="title", raw_field_name="title", raw_value="v1", support_type="structured_source", source_locator="")
    ])
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {"crossref": fake})

    refresh_study(db_session, study, run_id="RUN-1")
    db_session.commit()
    outcome = refresh_study(db_session, study, run_id="RUN-2")
    db_session.commit()

    assert outcome.checked == 1
    assert outcome.changed == 0
    assert db_session.query(Source).filter_by(study_id=study.study_id).count() == 1
    assert db_session.query(RawFact).filter_by(study_id=study.study_id).count() == 1

    watermark = db_session.query(SourceWatermark).filter_by(source_name="crossref", query_identifier="10.1234/x").one()
    assert watermark.last_status == "unchanged"
    assert watermark.last_run_id == "RUN-2"


def test_refresh_with_changed_content_hash_adds_new_source_linked_to_prior(db_session, monkeypatch):
    study = _study_with_doi(db_session)
    fake = FakeAdapter("crossref", _record("crossref", "hash-1"), facts=[
        RawFactCandidate(entity_level="study", fact_type_candidate="title", raw_field_name="title", raw_value="v1", support_type="structured_source", source_locator="")
    ])
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {"crossref": fake})
    refresh_study(db_session, study, run_id="RUN-1")
    db_session.commit()
    first_source = db_session.query(Source).filter_by(study_id=study.study_id).one()

    fake._record = _record("crossref", "hash-2")
    fake._facts = [
        RawFactCandidate(entity_level="study", fact_type_candidate="title", raw_field_name="title", raw_value="v2", support_type="structured_source", source_locator="")
    ]
    outcome = refresh_study(db_session, study, run_id="RUN-2")
    db_session.commit()

    assert outcome.changed == 1
    sources = db_session.query(Source).filter_by(study_id=study.study_id).order_by(Source.created_at).all()
    assert len(sources) == 2
    assert sources[1].parent_source_id == first_source.source_id
    assert sources[1].content_hash == "hash-2"
    # both snapshots' facts are preserved -- append-only evidence log
    values = {f.raw_value for f in db_session.query(RawFact).filter_by(study_id=study.study_id).all()}
    assert values == {"v1", "v2"}


def test_handle_refresh_study_sources_reads_run_id_from_payload(db_session, monkeypatch):
    study = _study_with_doi(db_session)
    fake = FakeAdapter("crossref", _record("crossref", "hash-1"))
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {"crossref": fake})
    task = enqueue_task(db_session, TaskType.REFRESH_STUDY_SOURCES, study_id=study.study_id, payload={"run_id": "RUN-42"})
    db_session.commit()

    handle_refresh_study_sources(db_session, task)
    db_session.commit()

    watermark = db_session.query(SourceWatermark).filter_by(source_name="crossref", query_identifier="10.1234/x").one()
    assert watermark.last_run_id == "RUN-42"


def test_enqueue_refresh_backfill_targets_studies_with_refreshable_identifiers_only(db_session):
    with_doi = _study_with_doi(db_session, doi="10.1234/a")
    without_identifiers = Study(title="No identifiers")
    db_session.add(without_identifiers)
    db_session.commit()

    count = enqueue_refresh_backfill(db_session)
    db_session.commit()

    assert count == 1
    task_study_ids = {
        t.study_id
        for t in db_session.query(refresh_handlers.Task).filter_by(task_type=TaskType.REFRESH_STUDY_SOURCES.value).all()
    }
    assert task_study_ids == {with_doi.study_id}
