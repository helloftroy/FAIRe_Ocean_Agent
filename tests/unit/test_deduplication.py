from fair_ocean_agent.database.enums import CanonicalStatus, EntityLevel, IdentifierType, SupportType
from fair_ocean_agent.database.models import (
    ExternalIdentifier,
    RawFact,
    Source,
    Study,
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
