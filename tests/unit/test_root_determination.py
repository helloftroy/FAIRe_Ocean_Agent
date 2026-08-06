"""Tests for identity/root_determination.py::determine_entity_root."""
from fair_ocean_agent.database.enums import (
    EntityLevel,
    EntityRootStatus,
    RelationshipType,
    ReviewStatus,
    SourceType,
    SupportType,
)
from fair_ocean_agent.database.models import Entity, EntityStudy, RawFact, Source, Study
from fair_ocean_agent.identity.root_determination import determine_entity_root


def _study(session, **kwargs) -> Study:
    study = Study(**kwargs)
    session.add(study)
    session.flush()
    return study


def _publication_source(session, study: Study, *, publication_year: int) -> None:
    session.add(
        Source(
            study_id=study.study_id, source_type=SourceType.PUBLICATION_API.value,
            source_name="crossref", publication_year=publication_year,
        )
    )


def _shared_entity(session, home: Study, others: list[Study]) -> Entity:
    entity = Entity(study_id=home.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMN1")
    session.add(entity)
    session.flush()
    session.add(
        EntityStudy(
            entity_id=entity.entity_id, study_id=home.study_id,
            relationship_type=RelationshipType.IS_HOME_OF.value, confidence=SupportType.STRUCTURED_SOURCE.value,
        )
    )
    for other in others:
        session.add(
            EntityStudy(
                entity_id=entity.entity_id, study_id=other.study_id,
                relationship_type=RelationshipType.SHARES_ACCESSION_WITH.value,
                confidence=SupportType.STRUCTURED_SOURCE.value,
            )
        )
    session.flush()
    return entity


def test_earliest_publication_year_wins(db_session):
    older = _study(db_session, title="Original, 2019")
    newer = _study(db_session, title="Reanalysis, 2024")
    entity = _shared_entity(db_session, older, [newer])
    _publication_source(db_session, older, publication_year=2019)
    _publication_source(db_session, newer, publication_year=2024)
    db_session.commit()

    determine_entity_root(db_session, entity)
    db_session.commit()

    assert entity.root_status == EntityRootStatus.DETERMINED.value
    assert entity.root_study_id == older.study_id
    assert entity.root_determined_at is not None


def test_same_year_tie_is_flagged_ambiguous_not_guessed(db_session):
    study_a = _study(db_session, title="Paper A, 2022")
    study_b = _study(db_session, title="Paper B, 2022")
    entity = _shared_entity(db_session, study_a, [study_b])
    _publication_source(db_session, study_a, publication_year=2022)
    _publication_source(db_session, study_b, publication_year=2022)
    db_session.commit()

    determine_entity_root(db_session, entity)
    db_session.commit()

    assert entity.root_status == EntityRootStatus.AMBIGUOUS.value
    assert entity.root_study_id is None
    flag = db_session.query(RawFact).filter_by(
        entity_id=entity.entity_id, fact_type_candidate="entity_root_ambiguous"
    ).one()
    assert flag.review_status == ReviewStatus.NEEDS_REVIEW.value
    assert set(flag.confidence_metadata["candidate_study_ids"]) == {study_a.study_id, study_b.study_id}


def test_no_publication_date_signal_at_all_is_ambiguous(db_session):
    study_a = _study(db_session, title="No DOI, repository-only A")
    study_b = _study(db_session, title="No DOI, repository-only B")
    entity = _shared_entity(db_session, study_a, [study_b])
    db_session.commit()

    determine_entity_root(db_session, entity)
    db_session.commit()

    assert entity.root_status == EntityRootStatus.AMBIGUOUS.value


def test_ambiguous_flag_is_not_duplicated_on_repeat_calls(db_session):
    study_a = _study(db_session, title="Paper A, 2022")
    study_b = _study(db_session, title="Paper B, 2022")
    entity = _shared_entity(db_session, study_a, [study_b])
    _publication_source(db_session, study_a, publication_year=2022)
    _publication_source(db_session, study_b, publication_year=2022)
    db_session.commit()

    determine_entity_root(db_session, entity)
    db_session.commit()
    determine_entity_root(db_session, entity)
    db_session.commit()

    flags = db_session.query(RawFact).filter_by(
        entity_id=entity.entity_id, fact_type_candidate="entity_root_ambiguous"
    ).all()
    assert len(flags) == 1


def test_non_shared_entity_is_left_unchanged_if_called_directly(db_session):
    """Not the normal call path (create_entity already sets this eagerly),
    but determine_entity_root must stay correct if invoked against a
    not-actually-shared entity."""
    study = _study(db_session, title="Solo paper")
    entity = Entity(
        study_id=study.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMN2",
        root_status=EntityRootStatus.PENDING.value,
    )
    db_session.add(entity)
    db_session.flush()
    db_session.add(
        EntityStudy(
            entity_id=entity.entity_id, study_id=study.study_id,
            relationship_type=RelationshipType.IS_HOME_OF.value, confidence=SupportType.STRUCTURED_SOURCE.value,
        )
    )
    db_session.commit()

    determine_entity_root(db_session, entity)
    db_session.commit()

    assert entity.root_status == EntityRootStatus.DETERMINED.value
    assert entity.root_study_id == study.study_id
