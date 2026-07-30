from fair_ocean_agent.database.enums import CanonicalStatus, EntityLevel, IdentifierType, RelationshipType, SupportType
from fair_ocean_agent.database.models import (
    ExternalIdentifier,
    RawFact,
    Source,
    Study,
    StudySource,
    ValidationResult,
)
from fair_ocean_agent.identity.deduplication import merge_study_into


def _study(session, **kwargs) -> Study:
    study = Study(**kwargs)
    session.add(study)
    session.flush()
    return study


def test_merge_study_into_reassigns_child_rows(db_session):
    absorb = _study(db_session, title="Absorbed study")
    into = _study(db_session, title="Canonical study")

    db_session.add(
        ExternalIdentifier(
            study_id=absorb.study_id, identifier_type=IdentifierType.PMID.value, identifier_value="12345"
        )
    )
    db_session.add(
        RawFact(
            study_id=absorb.study_id,
            entity_level=EntityLevel.STUDY.value,
            support_type=SupportType.STRUCTURED_SOURCE.value,
            raw_field_name="title",
            raw_value="Absorbed study",
        )
    )
    db_session.add(Source(study_id=absorb.study_id, source_type="publication_api", source_name="crossref"))
    db_session.commit()

    result = merge_study_into(db_session, absorb=absorb, into=into)
    db_session.commit()

    assert result.study_id == into.study_id
    assert absorb.canonical_status == CanonicalStatus.MERGED.value
    assert db_session.query(ExternalIdentifier).filter_by(study_id=into.study_id).count() == 1
    assert db_session.query(RawFact).filter_by(study_id=into.study_id).count() == 1
    assert db_session.query(Source).filter_by(study_id=into.study_id).count() == 1
    assert db_session.query(ExternalIdentifier).filter_by(study_id=absorb.study_id).count() == 0


def test_merge_study_into_drops_colliding_identifier_instead_of_erroring(db_session):
    absorb = _study(db_session, title="Absorbed")
    into = _study(db_session, title="Canonical")

    db_session.add(
        ExternalIdentifier(
            study_id=absorb.study_id, identifier_type=IdentifierType.DOI.value, identifier_value="10.1/x"
        )
    )
    db_session.add(
        ExternalIdentifier(
            study_id=into.study_id, identifier_type=IdentifierType.DOI.value, identifier_value="10.1/x"
        )
    )
    db_session.commit()

    merge_study_into(db_session, absorb=absorb, into=into)
    db_session.commit()

    # both rows described the same DOI; the duplicate on `absorb` is dropped,
    # not reassigned into a unique-constraint violation
    assert db_session.query(ExternalIdentifier).filter_by(
        study_id=into.study_id, identifier_value="10.1/x"
    ).count() == 1


def test_merge_study_into_is_noop_for_same_study(db_session):
    study = _study(db_session, title="Solo")
    result = merge_study_into(db_session, absorb=study, into=study)
    assert result.study_id == study.study_id
    assert study.canonical_status != CanonicalStatus.MERGED.value


def test_merge_study_into_reassigns_validation_result_rows(db_session):
    absorb = _study(db_session, title="Absorbed")
    into = _study(db_session, title="Canonical")
    db_session.add(ValidationResult(study_id=absorb.study_id, validator_name="cross_source_agreement"))
    db_session.commit()

    merge_study_into(db_session, absorb=absorb, into=into)
    db_session.commit()

    assert db_session.query(ValidationResult).filter_by(study_id=into.study_id).count() == 1
    assert db_session.query(ValidationResult).filter_by(study_id=absorb.study_id).count() == 0


def test_merge_study_into_reassigns_study_source_rows(db_session):
    absorb = _study(db_session, title="Absorbed")
    into = _study(db_session, title="Canonical")
    source = Source(study_id=absorb.study_id, source_type="repository_api", source_name="ena")
    db_session.add(source)
    db_session.flush()
    db_session.add(
        StudySource(
            study_id=absorb.study_id, source_id=source.source_id,
            relationship_type=RelationshipType.IS_HOME_OF.value, confidence=SupportType.STRUCTURED_SOURCE.value,
        )
    )
    db_session.commit()

    merge_study_into(db_session, absorb=absorb, into=into)
    db_session.commit()

    assert db_session.query(StudySource).filter_by(study_id=into.study_id, source_id=source.source_id).count() == 1
    assert db_session.query(StudySource).filter_by(study_id=absorb.study_id).count() == 0


def test_merge_study_into_drops_colliding_study_source_instead_of_erroring(db_session):
    absorb = _study(db_session, title="Absorbed")
    into = _study(db_session, title="Canonical")
    source = Source(study_id=into.study_id, source_type="repository_api", source_name="ena")
    db_session.add(source)
    db_session.flush()
    # Both absorb and into already link to the *same* source_id (e.g. two
    # independent tasks discovered the same accession before either merge
    # ran) -- the collision must be dropped, not raise a unique-constraint error.
    db_session.add(
        StudySource(
            study_id=absorb.study_id, source_id=source.source_id,
            relationship_type=RelationshipType.IS_HOME_OF.value, confidence=SupportType.STRUCTURED_SOURCE.value,
        )
    )
    db_session.add(
        StudySource(
            study_id=into.study_id, source_id=source.source_id,
            relationship_type=RelationshipType.IS_HOME_OF.value, confidence=SupportType.STRUCTURED_SOURCE.value,
        )
    )
    db_session.commit()

    merge_study_into(db_session, absorb=absorb, into=into)
    db_session.commit()

    assert db_session.query(StudySource).filter_by(study_id=into.study_id, source_id=source.source_id).count() == 1
    assert db_session.query(StudySource).filter_by(study_id=absorb.study_id).count() == 0


def test_find_all_existing_studies_by_identifier_returns_every_match(db_session):
    from fair_ocean_agent.identity.deduplication import find_all_existing_studies_by_identifier

    study_a = _study(db_session, title="A")
    study_b = _study(db_session, title="B")
    db_session.add(ExternalIdentifier(study_id=study_a.study_id, identifier_type=IdentifierType.BIOPROJECT_ACCESSION.value, identifier_value="PRJNA1"))
    db_session.add(ExternalIdentifier(study_id=study_b.study_id, identifier_type=IdentifierType.BIOPROJECT_ACCESSION.value, identifier_value="PRJNA1"))
    db_session.commit()

    matches = find_all_existing_studies_by_identifier(db_session, IdentifierType.BIOPROJECT_ACCESSION, "PRJNA1")
    assert {m.study_id for m in matches} == {study_a.study_id, study_b.study_id}


def test_find_all_existing_studies_by_identifier_returns_empty_for_no_match(db_session):
    from fair_ocean_agent.identity.deduplication import find_all_existing_studies_by_identifier

    matches = find_all_existing_studies_by_identifier(db_session, IdentifierType.BIOPROJECT_ACCESSION, "PRJNA999")
    assert matches == []
