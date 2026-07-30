from fair_ocean_agent.database.enums import RelationshipType, SourceType, SupportType
from fair_ocean_agent.database.models import Source, Study, StudySource
from fair_ocean_agent.identity.source_linking import create_source, link_source_to_study


def _study(session, **kwargs) -> Study:
    study = Study(**kwargs)
    session.add(study)
    session.flush()
    return study


def test_create_source_writes_source_and_home_study_source_row(db_session):
    study = _study(db_session, title="Some study")
    source = create_source(
        db_session,
        Source(study_id=study.study_id, source_type=SourceType.REPOSITORY_API.value, source_name="ena"),
    )
    db_session.commit()

    assert db_session.query(Source).filter_by(source_id=source.source_id).count() == 1
    link = db_session.query(StudySource).filter_by(source_id=source.source_id).one()
    assert link.study_id == study.study_id
    assert link.relationship_type == RelationshipType.IS_HOME_OF.value
    assert link.confidence == SupportType.STRUCTURED_SOURCE.value


def test_create_source_accepts_explicit_confidence(db_session):
    study = _study(db_session, title="Some study")
    source = create_source(
        db_session,
        Source(study_id=study.study_id, source_type=SourceType.SUPPLEMENT.value, source_name="europe_pmc_supplement"),
        confidence=SupportType.DETERMINISTICALLY_DERIVED,
    )
    db_session.commit()

    link = db_session.query(StudySource).filter_by(source_id=source.source_id).one()
    assert link.confidence == SupportType.DETERMINISTICALLY_DERIVED.value


def test_link_source_to_study_adds_an_additional_row_without_touching_source_study_id(db_session):
    home_study = _study(db_session, title="Home")
    sibling_study = _study(db_session, title="Sibling")
    source = create_source(
        db_session,
        Source(study_id=home_study.study_id, source_type=SourceType.REPOSITORY_API.value, source_name="ena"),
    )
    db_session.commit()

    link_source_to_study(
        db_session, source, sibling_study.study_id,
        relationship_type=RelationshipType.SHARES_ACCESSION_WITH, confidence=SupportType.DETERMINISTICALLY_DERIVED,
    )
    db_session.commit()

    assert source.study_id == home_study.study_id  # untouched
    links = {
        (link.study_id, link.relationship_type)
        for link in db_session.query(StudySource).filter_by(source_id=source.source_id).all()
    }
    assert links == {
        (home_study.study_id, RelationshipType.IS_HOME_OF.value),
        (sibling_study.study_id, RelationshipType.SHARES_ACCESSION_WITH.value),
    }
