"""Tests for the backward reference-chasing discovery pass:
handle_discover_primer_reference_study (workflow/handlers.py). Unlike
handle_discover_citing_studies (test_citation_discovery.py), this handler
never makes a network call -- the DOI is already resolved (from a real
JATS <ref-list> DOI, see extraction/publication_metadata.py::
extract_primer_reference_citations) by the time the task is enqueued."""
from fair_ocean_agent.config import AppConfig, DiscoveryConfig
from fair_ocean_agent.database.enums import EntityLevel, IdentifierType, ReviewStatus, SupportType, TaskType
from fair_ocean_agent.database.models import ExternalIdentifier, RawFact, Study, Task
from fair_ocean_agent.identity.identifiers import normalize_identifier
from fair_ocean_agent.workflow import handlers
from fair_ocean_agent.workflow.task_queue import enqueue_task


def _seed_parent_study(session, *, discovery_depth=0):
    study = Study(canonical_status="candidate", discovery_depth=discovery_depth)
    session.add(study)
    session.flush()
    return study


def _fact(session, study, *, field, value, review_status=ReviewStatus.ACCEPTED):
    fact = RawFact(
        study_id=study.study_id,
        entity_id=None,
        raw_field_name=field,
        raw_value=value,
        fact_type_candidate=field,
        entity_level=EntityLevel.STUDY.value,
        support_type=SupportType.EXPLICIT.value,
        review_status=review_status.value,
    )
    session.add(fact)
    session.flush()
    return fact


def _reference_task(session, parent, *, doi="10.1038/ismej.2012.8", primer_name="515F", sequence_field="pcr_primer_forward"):
    task = enqueue_task(
        session,
        TaskType.DISCOVER_PRIMER_REFERENCE_STUDIES,
        study_id=parent.study_id,
        payload={"doi": normalize_identifier(IdentifierType.DOI, doi), "primer_name": primer_name, "sequence_field": sequence_field},
        idempotency_key=f"test:{parent.study_id}:{doi}",
    )
    session.commit()
    return task


def test_creates_new_study_for_the_referenced_paper(db_session):
    parent = _seed_parent_study(db_session)
    task = _reference_task(db_session, parent)

    handlers.handle_discover_primer_reference_study(db_session, task)
    db_session.commit()

    referenced = db_session.query(Study).filter(Study.study_id != parent.study_id).one()
    assert referenced.discovery_depth == 1
    assert referenced.discovery_parent_study_id == parent.study_id
    assert referenced.discovery_root_study_id == parent.study_id
    assert referenced.discovery_trigger == "primer_reference_citation"
    assert referenced.canonical_status == "candidate"

    doi_ident = (
        db_session.query(ExternalIdentifier)
        .filter_by(study_id=referenced.study_id, identifier_type=IdentifierType.DOI.value)
        .one()
    )
    assert doi_ident.identifier_value == normalize_identifier(IdentifierType.DOI, "10.1038/ismej.2012.8")
    assert doi_ident.verified is True

    new_task = (
        db_session.query(Task)
        .filter_by(study_id=referenced.study_id, task_type=TaskType.DISCOVER_IDENTIFIERS.value)
        .one()
    )
    assert new_task is not None


def test_already_known_doi_is_not_duplicated(db_session):
    parent = _seed_parent_study(db_session)
    existing = Study(canonical_status="candidate")
    db_session.add(existing)
    db_session.flush()
    db_session.add(
        ExternalIdentifier(
            study_id=existing.study_id,
            identifier_type=IdentifierType.DOI.value,
            identifier_value=normalize_identifier(IdentifierType.DOI, "10.1038/ismej.2012.8"),
        )
    )
    db_session.commit()
    task = _reference_task(db_session, parent)

    handlers.handle_discover_primer_reference_study(db_session, task)
    db_session.commit()

    new_studies = db_session.query(Study).filter(Study.study_id.notin_([parent.study_id, existing.study_id])).all()
    assert new_studies == []


def test_depth_cap_flags_review_instead_of_expanding(db_session, monkeypatch):
    parent = _seed_parent_study(db_session, discovery_depth=1)
    task = _reference_task(db_session, parent)
    monkeypatch.setattr(
        handlers, "load_config", lambda: AppConfig(discovery=DiscoveryConfig(primer_reference_expansion_max_depth=1))
    )

    handlers.handle_discover_primer_reference_study(db_session, task)
    db_session.commit()

    new_studies = db_session.query(Study).filter(Study.study_id != parent.study_id).all()
    assert new_studies == []
    capped = (
        db_session.query(RawFact)
        .filter_by(study_id=parent.study_id, fact_type_candidate="primer_reference_not_expanded")
        .one()
    )
    assert capped.review_status == ReviewStatus.NEEDS_REVIEW.value


def test_no_op_when_corpus_already_has_the_sequence(db_session):
    parent = _seed_parent_study(db_session)
    other_study = _seed_parent_study(db_session)
    _fact(db_session, other_study, field="pcr_primer_name_forward", value="515F")
    _fact(db_session, other_study, field="pcr_primer_forward", value="GTGYCAGCMGCCGCGGTAA")
    task = _reference_task(db_session, parent)

    handlers.handle_discover_primer_reference_study(db_session, task)
    db_session.commit()

    new_studies = db_session.query(Study).filter(Study.study_id.notin_([parent.study_id, other_study.study_id])).all()
    assert new_studies == []


def test_idempotent_rerun_does_not_create_a_second_study(db_session):
    parent = _seed_parent_study(db_session)
    task = _reference_task(db_session, parent)

    handlers.handle_discover_primer_reference_study(db_session, task)
    db_session.commit()
    handlers.handle_discover_primer_reference_study(db_session, task)
    db_session.commit()

    referenced_studies = db_session.query(Study).filter(Study.study_id != parent.study_id).all()
    assert len(referenced_studies) == 1
