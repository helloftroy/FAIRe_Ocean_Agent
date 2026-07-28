"""Integration tests for the repository-source half of
handle_discover_identifiers (BioProject/ENA-accession-keyed resolution via
NCBI/ENA), using the same FakeAdapter pattern as test_handlers.py -- no
network, no real NCBI/ENA calls."""
from datetime import datetime, timezone

import pytest

from fair_ocean_agent.database.enums import (
    CanonicalStatus,
    EntityLevel,
    IdentifierType,
    RelationshipType,
    SupportType,
)
from fair_ocean_agent.database.models import Entity, ExternalIdentifier, RawFact, Source, Study
from fair_ocean_agent.database.enums import TaskType
from fair_ocean_agent.identity.identifiers import normalize_identifier
from fair_ocean_agent.sources.base import RawFactCandidate, RelatedIdentifier, SourceRecord, SourceRecordNotFoundError
from fair_ocean_agent.workflow import handlers
from fair_ocean_agent.workflow.task_queue import enqueue_task


class FakeAdapter:
    def __init__(self, name, record=None, facts=None, related=None, not_found=False):
        self.name = name
        self._record = record
        self._facts = facts or []
        self._related = related or []
        self._not_found = not_found

    def fetch_record(self, identifier):
        if self._not_found:
            raise SourceRecordNotFoundError(f"no {self.name} record")
        return self._record

    def extract_structured_facts(self, record):
        return self._facts

    def parse_publication_fields(self, record):
        return {}

    def find_related(self, record):
        return self._related

    def close(self):
        pass


def _make_record(source_name: str, external_identifier: str = "PRJNA1425045", raw: dict | None = None) -> SourceRecord:
    return SourceRecord(
        source_name=source_name,
        external_identifier=external_identifier,
        url=None,
        raw=raw if raw is not None else {"stub": True},
        retrieved_at=datetime.now(timezone.utc),
        content_hash="deadbeef",
    )


def _seeded_study(session, bioproject_accession=None, ena_accession=None) -> Study:
    study = Study(title=None)
    session.add(study)
    session.flush()
    if bioproject_accession:
        session.add(
            ExternalIdentifier(
                study_id=study.study_id,
                identifier_type=IdentifierType.BIOPROJECT_ACCESSION.value,
                identifier_value=normalize_identifier(IdentifierType.BIOPROJECT_ACCESSION, bioproject_accession),
            )
        )
    if ena_accession:
        session.add(
            ExternalIdentifier(
                study_id=study.study_id,
                identifier_type=IdentifierType.ENA_STUDY_ACCESSION.value,
                identifier_value=normalize_identifier(IdentifierType.ENA_STUDY_ACCESSION, ena_accession),
            )
        )
    session.flush()
    return study


def _seeded_dataset_study(session, identifier_type: IdentifierType, value: str) -> Study:
    study = Study(title=None)
    session.add(study)
    session.flush()
    session.add(
        ExternalIdentifier(
            study_id=study.study_id,
            identifier_type=identifier_type.value,
            identifier_value=normalize_identifier(identifier_type, value),
        )
    )
    session.flush()
    return study


def _task_for(session, study):
    task = enqueue_task(session, TaskType.DISCOVER_IDENTIFIERS, study_id=study.study_id)
    session.commit()
    return task


def _sample_facts(accession: str, label: str, attrs: dict) -> list[RawFactCandidate]:
    return [
        RawFactCandidate(
            entity_level=EntityLevel.SAMPLE,
            fact_type_candidate=k,
            raw_field_name=k,
            raw_value=str(v),
            source_locator=f"ncbi_biosample.{accession}.{k}",
            support_type=SupportType.STRUCTURED_SOURCE,
            entity_external_id=accession,
            entity_label=label,
        )
        for k, v in attrs.items()
    ]


def test_biosample_facts_create_per_sample_entities(db_session, monkeypatch):
    study = _seeded_study(db_session, bioproject_accession="PRJNA1425045")
    task = _task_for(db_session, study)

    facts = _sample_facts("SAMN1", "Sample one", {"collection_date": "2023-12-06", "depth": "1"}) + _sample_facts(
        "SAMN2", "Sample two", {"collection_date": "2023-12-07"}
    )
    biosample_adapter = FakeAdapter("ncbi_biosample", record=_make_record("ncbi_biosample"), facts=facts)
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {"ncbi_biosample": biosample_adapter})

    handlers.handle_discover_identifiers(db_session, task)
    db_session.commit()

    entities = db_session.query(Entity).filter_by(study_id=study.study_id, entity_level=EntityLevel.SAMPLE.value).all()
    assert {e.external_identifier for e in entities} == {"SAMN1", "SAMN2"}
    sample_one_entity = next(e for e in entities if e.external_identifier == "SAMN1")
    assert sample_one_entity.label == "Sample one"

    sample_one_facts = db_session.query(RawFact).filter_by(entity_id=sample_one_entity.entity_id).all()
    assert {f.raw_field_name for f in sample_one_facts} == {"collection_date", "depth"}


def test_bioproject_title_propagates_to_study_when_no_doi(db_session, monkeypatch):
    """Repository-only studies (no DOI) never go through
    _resolve_publication_sources, which is the only place that normally
    sets study.title -- the BioProject's own title must still end up on the
    study, not just as a buried raw_fact."""
    study = _seeded_study(db_session, bioproject_accession="PRJNA1425045")
    task = _task_for(db_session, study)

    bioproject_adapter = FakeAdapter(
        "ncbi_bioproject",
        record=_make_record("ncbi_bioproject", raw={"title": "SF Bay 18S Metabarcoding Monitoring"}),
    )
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {"ncbi_bioproject": bioproject_adapter})

    handlers.handle_discover_identifiers(db_session, task)
    db_session.commit()

    refreshed = db_session.get(Study, study.study_id)
    assert refreshed.title == "SF Bay 18S Metabarcoding Monitoring"


def test_ena_title_propagates_to_study_when_no_bioproject_or_doi(db_session, monkeypatch):
    study = _seeded_study(db_session, ena_accession="ERP123456")
    task = _task_for(db_session, study)

    ena_adapter = FakeAdapter(
        "ena",
        record=_make_record(
            "ena", external_identifier="ERP123456", raw={"study": {"study_title": "An ENA-only study"}}
        ),
    )
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {"ena": ena_adapter})

    handlers.handle_discover_identifiers(db_session, task)
    db_session.commit()

    refreshed = db_session.get(Study, study.study_id)
    assert refreshed.title == "An ENA-only study"


def test_repository_resolution_is_idempotent_on_retry(db_session, monkeypatch):
    study = _seeded_study(db_session, bioproject_accession="PRJNA1425045")
    task = _task_for(db_session, study)

    facts = _sample_facts("SAMN1", "Sample one", {"depth": "1"})
    biosample_adapter = FakeAdapter("ncbi_biosample", record=_make_record("ncbi_biosample"), facts=facts)
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {"ncbi_biosample": biosample_adapter})

    handlers.handle_discover_identifiers(db_session, task)
    db_session.commit()
    handlers.handle_discover_identifiers(db_session, task)  # simulated retry
    db_session.commit()

    assert db_session.query(Source).filter_by(study_id=study.study_id).count() == 1
    assert db_session.query(Entity).filter_by(study_id=study.study_id).count() == 1
    assert db_session.query(RawFact).filter_by(study_id=study.study_id).count() == 1


def test_ena_only_study_resolves_via_ena_study_accession(db_session, monkeypatch):
    study = _seeded_study(db_session, ena_accession="ERP123456")
    task = _task_for(db_session, study)

    ena_adapter = FakeAdapter("ena", record=_make_record("ena", external_identifier="ERP123456"))
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {"ena": ena_adapter})

    handlers.handle_discover_identifiers(db_session, task)
    db_session.commit()

    source = db_session.query(Source).filter_by(study_id=study.study_id).one()
    assert source.external_identifier == "ERP123456"
    assert source.source_name == "ena"


def test_two_adapters_discovering_the_same_related_identifier_does_not_raise(db_session, monkeypatch):
    """Regression test: ncbi_biosample and ena commonly surface the exact
    same BioSample accessions for one study (both mirror the same INSDC
    records). Before the fix, _apply_related_identifiers checked a
    `study.external_identifiers` ORM collection that goes stale mid-task --
    rows added by the first adapter's pass through this same function don't
    retroactively appear in an already-loaded collection, so the second
    adapter surfacing the same identifier hit the table's unique constraint
    (a real IntegrityError seen in live validation, not a hypothetical)."""
    study = _seeded_study(db_session, bioproject_accession="PRJNA1425045")
    task = _task_for(db_session, study)

    shared_related = [
        RelatedIdentifier(
            identifier_type=IdentifierType.BIOSAMPLE_ACCESSION,
            value="SAMN1",
            relationship_type=RelationshipType.CONTAINS_SAMPLES_FROM,
            source="shared",
        )
    ]
    biosample_adapter = FakeAdapter("ncbi_biosample", record=_make_record("ncbi_biosample"), related=shared_related)
    ena_adapter = FakeAdapter("ena", record=_make_record("ena"), related=shared_related)
    monkeypatch.setattr(
        handlers, "_build_enabled_adapters", lambda: {"ncbi_biosample": biosample_adapter, "ena": ena_adapter}
    )

    handlers.handle_discover_identifiers(db_session, task)  # must not raise IntegrityError
    db_session.commit()

    matches = db_session.query(ExternalIdentifier).filter_by(
        study_id=study.study_id, identifier_type=IdentifierType.BIOSAMPLE_ACCESSION.value, identifier_value="SAMN1"
    ).all()
    assert len(matches) == 1  # not duplicated despite two adapters reporting it


def test_study_with_both_doi_and_bioproject_resolves_both(db_session, monkeypatch):
    study = _seeded_study(db_session, bioproject_accession="PRJNA1425045")
    db_session.add(
        ExternalIdentifier(
            study_id=study.study_id, identifier_type=IdentifierType.DOI.value, identifier_value="10.1234/x"
        )
    )
    db_session.commit()
    task = _task_for(db_session, study)

    crossref_adapter = FakeAdapter("crossref", record=_make_record("crossref", external_identifier="10.1234/x"))
    biosample_adapter = FakeAdapter(
        "ncbi_biosample",
        record=_make_record("ncbi_biosample"),
        facts=_sample_facts("SAMN1", "Sample one", {"depth": "1"}),
    )
    monkeypatch.setattr(
        handlers, "_build_enabled_adapters", lambda: {"crossref": crossref_adapter, "ncbi_biosample": biosample_adapter}
    )

    handlers.handle_discover_identifiers(db_session, task)
    db_session.commit()

    sources = {s.source_name for s in db_session.query(Source).filter_by(study_id=study.study_id).all()}
    assert sources == {"crossref", "ncbi_biosample"}


def test_dataset_doi_resolves_via_datacite(db_session, monkeypatch):
    study = _seeded_dataset_study(db_session, IdentifierType.DATASET_DOI, "10.1594/PANGAEA.923577")
    task = _task_for(db_session, study)

    datacite_adapter = FakeAdapter(
        "datacite",
        record=_make_record(
            "datacite",
            external_identifier="10.1594/pangaea.923577",
            raw={"data": {"attributes": {"titles": [{"title": "A dataset title"}]}}},
        ),
        facts=[
            RawFactCandidate(
                entity_level=EntityLevel.PROJECT,
                fact_type_candidate="titles",
                raw_field_name="titles",
                raw_value='[{"title": "A dataset title"}]',
                source_locator="datacite.data.attributes.titles",
            )
        ],
    )
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {"datacite": datacite_adapter})

    handlers.handle_discover_identifiers(db_session, task)
    db_session.commit()

    source = db_session.query(Source).filter_by(study_id=study.study_id).one()
    assert source.source_name == "datacite"
    assert source.source_type == "repository_api"
    assert db_session.query(RawFact).filter_by(study_id=study.study_id, fact_type_candidate="titles").count() == 1


def test_obis_only_study_resolves_via_obis_dataset_uuid(db_session, monkeypatch):
    dataset_uuid = "11111111-2222-3333-4444-555555555555"
    study = _seeded_dataset_study(db_session, IdentifierType.OBIS_DATASET_UUID, dataset_uuid)
    task = _task_for(db_session, study)

    obis_adapter = FakeAdapter(
        "obis",
        record=_make_record("obis", external_identifier=dataset_uuid, raw={"title": "OBIS dataset"}),
        facts=[
            RawFactCandidate(
                entity_level=EntityLevel.PROJECT,
                fact_type_candidate="title",
                raw_field_name="title",
                raw_value="OBIS dataset",
                source_locator="obis.dataset.title",
            )
        ],
    )
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {"obis": obis_adapter})

    handlers.handle_discover_identifiers(db_session, task)
    db_session.commit()

    source = db_session.query(Source).filter_by(study_id=study.study_id).one()
    assert source.source_name == "obis"
    assert source.external_identifier == dataset_uuid
    assert db_session.get(Study, study.study_id).title == "OBIS dataset"
    assert db_session.query(Entity).filter_by(study_id=study.study_id).count() == 0


def test_repository_related_identifier_merges_into_existing_study(db_session, monkeypatch):
    study = _seeded_study(db_session, bioproject_accession="PRJNA1425045")
    other_study = Study(title="Found earlier by BioSample accession")
    db_session.add(other_study)
    db_session.flush()
    db_session.add(
        ExternalIdentifier(
            study_id=other_study.study_id,
            identifier_type=IdentifierType.BIOSAMPLE_ACCESSION.value,
            identifier_value="SAMN1",
        )
    )
    db_session.commit()

    task = _task_for(db_session, study)
    biosample_adapter = FakeAdapter(
        "ncbi_biosample",
        record=_make_record("ncbi_biosample"),
        related=[
            RelatedIdentifier(
                identifier_type=IdentifierType.BIOSAMPLE_ACCESSION,
                value="SAMN1",
                relationship_type=RelationshipType.CONTAINS_SAMPLES_FROM,
                source="ncbi_biosample",
            )
        ],
    )
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {"ncbi_biosample": biosample_adapter})

    handlers.handle_discover_identifiers(db_session, task)
    db_session.commit()

    refreshed_other = db_session.get(Study, other_study.study_id)
    assert refreshed_other.canonical_status == CanonicalStatus.MERGED.value


def test_bioproject_only_study_without_doi_does_not_raise(db_session, monkeypatch):
    study = _seeded_study(db_session, bioproject_accession="PRJNA1425045")
    task = _task_for(db_session, study)
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {})

    with pytest.raises(RuntimeError):
        handlers.handle_discover_identifiers(db_session, task)


def test_bioproject_accession_present_but_no_repository_adapter_enabled_logs_not_raises(db_session, monkeypatch):
    study = _seeded_study(db_session, bioproject_accession="PRJNA1425045")
    task = _task_for(db_session, study)
    # Only a publication adapter is enabled -- no repository adapter -- but
    # the study has no DOI, so the publication branch never runs and there's
    # genuinely nothing else the handler can enable-check against.
    crossref_adapter = FakeAdapter("crossref", record=_make_record("crossref"))
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {"crossref": crossref_adapter})

    handlers.handle_discover_identifiers(db_session, task)  # must not raise
    db_session.commit()

    assert db_session.query(Source).filter_by(study_id=study.study_id).count() == 0
