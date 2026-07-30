"""Integration test for handle_discover_identifiers: fake, in-process
adapters (no network) standing in for Crossref/Europe PMC/OpenAlex, so the
orchestration logic (Source/RawFact creation with bibliographic fields,
related-identifier merging, Stage 2 dedup) is tested independent of any
live API.
"""
from datetime import datetime, timezone

import pytest

from fair_ocean_agent.database.enums import (
    CanonicalStatus,
    EntityLevel,
    IdentifierType,
    RelationshipType,
    SupportType,
)
from fair_ocean_agent.database.models import ExternalIdentifier, RawFact, Source, Study
from fair_ocean_agent.identity.identifiers import normalize_doi
from fair_ocean_agent.sources.base import RawFactCandidate, RelatedIdentifier, SourceRecord, SourceRecordNotFoundError
from fair_ocean_agent.sources.europe_pmc import EuropePmcAdapter
from fair_ocean_agent.workflow import handlers
from fair_ocean_agent.workflow.task_queue import enqueue_task
from fair_ocean_agent.database.enums import TaskType


class FakeAdapter:
    def __init__(self, name, record=None, facts=None, publication_fields=None, related=None, not_found=False):
        self.name = name
        self._record = record
        self._facts = facts or []
        self._publication_fields = publication_fields or {}
        self._related = related or []
        self._not_found = not_found
        self.closed = False

    def fetch_record(self, identifier):
        if self._not_found:
            raise SourceRecordNotFoundError(f"no {self.name} record")
        return self._record

    def extract_structured_facts(self, record):
        return self._facts

    def parse_publication_fields(self, record):
        return self._publication_fields

    def find_related(self, record):
        return self._related

    def close(self):
        self.closed = True


class FakeEuropePmcFullTextAdapter(EuropePmcAdapter):
    name = "europe_pmc"

    def __init__(self, fulltext_xml: str):
        self._fulltext_xml = fulltext_xml
        self._record = _make_record("europe_pmc")

    def fetch_record(self, identifier):
        return self._record

    def fetch_fulltext_xml(self, pmcid):
        return self._fulltext_xml

    def extract_structured_facts(self, record):
        return []

    def parse_publication_fields(self, record):
        return {"title": "DOI paper with repository IDs", "fulltext_available": True}

    def find_related(self, record):
        return [
            RelatedIdentifier(
                identifier_type=IdentifierType.PMCID,
                value="PMC5000000",
                relationship_type=RelationshipType.IS_PUBLICATION_OF,
                source="europe_pmc",
            )
        ]

    def close(self):
        pass


def _make_record(source_name: str) -> SourceRecord:
    return SourceRecord(
        source_name=source_name,
        external_identifier="10.1234/x",
        url=f"https://example.org/{source_name}",
        raw={"stub": True},
        retrieved_at=datetime.now(timezone.utc),
        content_hash="deadbeef",
    )


def _seeded_study_with_doi(session, doi="10.1234/x") -> Study:
    study = Study(title=None)
    session.add(study)
    session.flush()
    session.add(
        ExternalIdentifier(
            study_id=study.study_id, identifier_type=IdentifierType.DOI.value, identifier_value=normalize_doi(doi)
        )
    )
    session.flush()
    return study


def _task_for(session, study) -> "handlers.Task":
    task = enqueue_task(session, TaskType.DISCOVER_IDENTIFIERS, study_id=study.study_id)
    session.commit()
    return task


def test_handler_creates_source_facts_and_publication(db_session, monkeypatch):
    study = _seeded_study_with_doi(db_session)
    task = _task_for(db_session, study)

    record = _make_record("crossref")
    fake = FakeAdapter(
        "crossref",
        record=record,
        facts=[
            RawFactCandidate(
                entity_level=EntityLevel.STUDY,
                fact_type_candidate="title",
                raw_field_name="title",
                raw_value="A Fake Study Title",
                source_locator="crossref.message.title",
                support_type=SupportType.STRUCTURED_SOURCE,
            )
        ],
        publication_fields={"doi": "10.1234/x", "title": "A Fake Study Title", "publication_year": 2020},
    )
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {"crossref": fake})

    handlers.handle_discover_identifiers(db_session, task)
    db_session.commit()

    sources = db_session.query(Source).filter_by(study_id=study.study_id).all()
    assert len(sources) == 1 and sources[0].source_name == "crossref"
    assert sources[0].publication_year == 2020

    facts = db_session.query(RawFact).filter_by(study_id=study.study_id).all()
    assert len(facts) == 1 and facts[0].raw_value == "A Fake Study Title"

    refreshed = db_session.get(Study, study.study_id)
    assert refreshed.title == "A Fake Study Title"


def test_handler_continues_when_one_adapter_has_no_record(db_session, monkeypatch):
    study = _seeded_study_with_doi(db_session)
    task = _task_for(db_session, study)

    missing = FakeAdapter("europe_pmc", not_found=True)
    present = FakeAdapter(
        "openalex",
        record=_make_record("openalex"),
        publication_fields={"openalex_id": "W1", "title": "Still Works"},
    )
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {"europe_pmc": missing, "openalex": present})

    handlers.handle_discover_identifiers(db_session, task)
    db_session.commit()

    sources = db_session.query(Source).filter_by(study_id=study.study_id).all()
    assert [s.source_name for s in sources] == ["openalex"]


def test_handler_merges_study_on_related_identifier_collision(db_session, monkeypatch):
    study = _seeded_study_with_doi(db_session, doi="10.1234/new")
    other_study = Study(title="Pre-existing study found by PMID")
    db_session.add(other_study)
    db_session.flush()
    db_session.add(
        ExternalIdentifier(study_id=other_study.study_id, identifier_type=IdentifierType.PMID.value, identifier_value="999999")
    )
    db_session.commit()

    task = _task_for(db_session, study)
    fake = FakeAdapter(
        "europe_pmc",
        record=_make_record("europe_pmc"),
        related=[
            RelatedIdentifier(
                identifier_type=IdentifierType.PMID,
                value="999999",
                relationship_type=RelationshipType.IS_PUBLICATION_OF,
                source="europe_pmc",
            )
        ],
    )
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {"europe_pmc": fake})

    handlers.handle_discover_identifiers(db_session, task)
    db_session.commit()

    refreshed_other = db_session.get(Study, other_study.study_id)
    assert refreshed_other.canonical_status == CanonicalStatus.MERGED.value

    # the DOI-bearing study absorbed the PMID that used to belong to other_study
    merged_ids = {
        (ei.identifier_type, ei.identifier_value)
        for ei in db_session.query(ExternalIdentifier).filter_by(study_id=study.study_id).all()
    }
    assert (IdentifierType.PMID.value, "999999") in merged_ids


def test_handler_raises_not_implemented_without_doi(db_session):
    study = Study(title="No DOI here")
    db_session.add(study)
    db_session.flush()
    task = _task_for(db_session, study)

    with pytest.raises(NotImplementedError):
        handlers.handle_discover_identifiers(db_session, task)


def test_handler_is_idempotent_on_retry(db_session, monkeypatch):
    """Simulates a retried task: calling the handler twice for the same
    study/adapter must not duplicate Source/RawFact rows, and the Source's
    bibliographic fields must still end up fully populated."""
    study = _seeded_study_with_doi(db_session)
    task = _task_for(db_session, study)

    fake = FakeAdapter(
        "crossref",
        record=_make_record("crossref"),
        facts=[
            RawFactCandidate(
                entity_level=EntityLevel.STUDY,
                fact_type_candidate="title",
                raw_field_name="title",
                raw_value="Retried Study Title",
                source_locator="crossref.message.title",
                support_type=SupportType.STRUCTURED_SOURCE,
            )
        ],
        publication_fields={"doi": "10.1234/x", "title": "Retried Study Title", "publication_year": 2021},
    )
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {"crossref": fake})

    handlers.handle_discover_identifiers(db_session, task)
    db_session.commit()
    handlers.handle_discover_identifiers(db_session, task)  # simulated retry
    db_session.commit()

    assert db_session.query(Source).filter_by(study_id=study.study_id).count() == 1
    assert db_session.query(RawFact).filter_by(study_id=study.study_id).count() == 1
    source = db_session.query(Source).filter_by(study_id=study.study_id).one()
    assert source.publication_year == 2021


def test_handler_mines_fulltext_identifiers_and_resolves_repository_sources(db_session, monkeypatch):
    study = _seeded_study_with_doi(db_session)
    task = _task_for(db_session, study)

    europe_pmc_adapter = FakeEuropePmcFullTextAdapter(
        "<article><sec><title>Data Availability</title>"
        "<p>Raw reads are available under BioProject PRJNA515494.</p>"
        "</sec></article>"
    )
    biosample_adapter = FakeAdapter(
        "ncbi_biosample",
        record=_make_record("ncbi_biosample"),
        facts=[
            RawFactCandidate(
                entity_level=EntityLevel.SAMPLE,
                fact_type_candidate="collection_date",
                raw_field_name="collection_date",
                raw_value="2020-01",
                source_locator="ncbi_biosample.SAMN1.collection_date",
                entity_external_id="SAMN1",
                entity_label="SAMN1",
            )
        ],
    )
    monkeypatch.setattr(
        handlers,
        "_build_enabled_adapters",
        lambda: {"europe_pmc": europe_pmc_adapter, "ncbi_biosample": biosample_adapter},
    )

    handlers.handle_discover_identifiers(db_session, task)
    db_session.commit()

    identifiers = {
        (ei.identifier_type, ei.identifier_value, ei.source)
        for ei in db_session.query(ExternalIdentifier).filter_by(study_id=study.study_id).all()
    }
    assert (
        IdentifierType.BIOPROJECT_ACCESSION.value,
        "PRJNA515494",
        "europe_pmc_fulltext_identifier_scan",
    ) in identifiers
    assert db_session.query(Source).filter_by(study_id=study.study_id, source_name="ncbi_biosample").count() == 1
    assert db_session.query(Source).filter_by(study_id=study.study_id, source_name="europe_pmc_fulltext").count() == 0


def test_handler_raises_runtime_error_when_no_adapters_enabled(db_session, monkeypatch):
    study = _seeded_study_with_doi(db_session)
    task = _task_for(db_session, study)
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {})

    with pytest.raises(RuntimeError):
        handlers.handle_discover_identifiers(db_session, task)
