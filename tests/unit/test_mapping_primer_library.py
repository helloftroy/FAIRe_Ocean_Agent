"""Tests for mapping/primer_library.py -- the corpus-wide primer name ->
sequence lookup (identity/deduplication.py's find_existing_study_by_
identifier idiom, applied to primer names instead of external
identifiers)."""
from fair_ocean_agent.database.enums import EntityLevel, ReviewStatus, SupportType
from fair_ocean_agent.database.models import RawFact, Study
from fair_ocean_agent.mapping.primer_library import (
    corpus_primer_sequence,
    resolve_primer_sequences_from_corpus,
    study_primer_name,
)


def _study(session, **kwargs) -> Study:
    study = Study(**kwargs)
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


def test_corpus_primer_sequence_finds_match_in_a_different_study(db_session):
    source_study = _study(db_session, title="Paper that gives the sequence")
    _fact(db_session, source_study, field="pcr_primer_name_forward", value="515F")
    _fact(db_session, source_study, field="pcr_primer_forward", value="GTGYCAGCMGCCGCGGTAA")
    db_session.commit()

    sequence = corpus_primer_sequence(db_session, "515F", "pcr_primer_forward")

    assert sequence == "GTGYCAGCMGCCGCGGTAA"


def test_corpus_primer_sequence_is_case_insensitive_but_not_fuzzy(db_session):
    source_study = _study(db_session, title="Paper that gives the sequence")
    _fact(db_session, source_study, field="pcr_primer_name_forward", value="515f")
    _fact(db_session, source_study, field="pcr_primer_forward", value="GTGYCAGCMGCCGCGGTAA")
    db_session.commit()

    assert corpus_primer_sequence(db_session, "515F", "pcr_primer_forward") == "GTGYCAGCMGCCGCGGTAA"
    assert corpus_primer_sequence(db_session, "515F-Y", "pcr_primer_forward") is None


def test_corpus_primer_sequence_ignores_rejected_facts(db_session):
    source_study = _study(db_session, title="Paper with a rejected sequence")
    _fact(db_session, source_study, field="pcr_primer_name_forward", value="515F")
    _fact(db_session, source_study, field="pcr_primer_forward", value="BOGUS", review_status=ReviewStatus.REJECTED)
    db_session.commit()

    assert corpus_primer_sequence(db_session, "515F", "pcr_primer_forward") is None


def test_corpus_primer_sequence_none_when_no_study_has_it(db_session):
    assert corpus_primer_sequence(db_session, "515F", "pcr_primer_forward") is None


def test_resolve_primer_sequences_from_corpus_backfills_missing_sequence(db_session):
    source_study = _study(db_session, title="Paper with the real sequence")
    _fact(db_session, source_study, field="pcr_primer_name_forward", value="515F")
    _fact(db_session, source_study, field="pcr_primer_forward", value="GTGYCAGCMGCCGCGGTAA")

    needs_it_study = _study(db_session, title="Paper that only names the primer")
    _fact(db_session, needs_it_study, field="pcr_primer_name_forward", value="515F")
    db_session.commit()

    resolve_primer_sequences_from_corpus(db_session, needs_it_study.study_id)
    db_session.commit()

    inherited = (
        db_session.query(RawFact)
        .filter_by(study_id=needs_it_study.study_id, fact_type_candidate="pcr_primer_forward_inherited")
        .one()
    )
    assert inherited.raw_value == "GTGYCAGCMGCCGCGGTAA"
    assert inherited.review_status == ReviewStatus.NEEDS_REVIEW.value
    assert inherited.support_type == SupportType.INFERRED.value


def test_resolve_primer_sequences_from_corpus_is_idempotent(db_session):
    source_study = _study(db_session, title="Paper with the real sequence")
    _fact(db_session, source_study, field="pcr_primer_name_forward", value="515F")
    _fact(db_session, source_study, field="pcr_primer_forward", value="GTGYCAGCMGCCGCGGTAA")

    needs_it_study = _study(db_session, title="Paper that only names the primer")
    _fact(db_session, needs_it_study, field="pcr_primer_name_forward", value="515F")
    db_session.commit()

    resolve_primer_sequences_from_corpus(db_session, needs_it_study.study_id)
    db_session.commit()
    resolve_primer_sequences_from_corpus(db_session, needs_it_study.study_id)
    db_session.commit()

    count = (
        db_session.query(RawFact)
        .filter_by(study_id=needs_it_study.study_id, fact_type_candidate="pcr_primer_forward_inherited")
        .count()
    )
    assert count == 1


def test_resolve_primer_sequences_from_corpus_no_op_when_study_already_has_its_own_sequence(db_session):
    study = _study(db_session, title="Paper with its own real sequence")
    _fact(db_session, study, field="pcr_primer_name_forward", value="515F")
    _fact(db_session, study, field="pcr_primer_forward", value="OWN-SEQUENCE")

    other_study = _study(db_session, title="Different paper, different sequence text")
    _fact(db_session, other_study, field="pcr_primer_name_forward", value="515F")
    _fact(db_session, other_study, field="pcr_primer_forward", value="OTHER-SEQUENCE")
    db_session.commit()

    resolve_primer_sequences_from_corpus(db_session, study.study_id)
    db_session.commit()

    count = (
        db_session.query(RawFact)
        .filter_by(study_id=study.study_id, fact_type_candidate="pcr_primer_forward_inherited")
        .count()
    )
    assert count == 0


def test_resolve_primer_sequences_from_corpus_no_op_when_no_primer_name_known(db_session):
    study = _study(db_session, title="Paper with no primer info at all")
    db_session.commit()

    resolve_primer_sequences_from_corpus(db_session, study.study_id)
    db_session.commit()

    count = db_session.query(RawFact).filter_by(study_id=study.study_id).count()
    assert count == 0


def test_study_primer_name_returns_oldest_surviving_value(db_session):
    study = _study(db_session, title="Paper with a primer name")
    _fact(db_session, study, field="pcr_primer_name_forward", value="515F")
    db_session.commit()

    assert study_primer_name(db_session, study.study_id, "pcr_primer_name_forward") == "515F"
    assert study_primer_name(db_session, study.study_id, "pcr_primer_name_reverse") is None
