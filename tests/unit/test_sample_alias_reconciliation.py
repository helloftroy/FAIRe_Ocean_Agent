"""Tests for identity/sample_alias_reconciliation.py::reconcile_sample_aliases."""
from fair_ocean_agent.database.enums import (
    EntityLevel,
    EntityRelationshipType,
    RelationshipType,
    ReviewStatus,
    SupportType,
)
from fair_ocean_agent.database.models import Entity, EntityRelationship, EntityStudy, RawFact, Study
from fair_ocean_agent.identity.sample_alias_reconciliation import reconcile_sample_aliases


def _study(session, **kwargs) -> Study:
    study = Study(**kwargs)
    session.add(study)
    session.flush()
    return study


def _sample_entity(session, study: Study, external_identifier: str) -> Entity:
    entity = Entity(
        study_id=study.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier=external_identifier
    )
    session.add(entity)
    session.flush()
    session.add(
        EntityStudy(
            entity_id=entity.entity_id,
            study_id=study.study_id,
            relationship_type=RelationshipType.IS_HOME_OF.value,
            confidence=SupportType.STRUCTURED_SOURCE.value,
        )
    )
    session.flush()
    return entity


def _accession_fact(session, entity: Entity, study: Study) -> None:
    """Marks `entity` as a real, structurally-resolved BioSample --
    reconcile_sample_aliases treats any SAMPLE entity with a RawFact from
    this extraction_method as a canonical candidate, never an alias."""
    session.add(
        RawFact(
            study_id=study.study_id,
            entity_id=entity.entity_id,
            fact_type_candidate="biosample_accession",
            raw_value=entity.external_identifier,
            entity_level=EntityLevel.SAMPLE.value,
            support_type=SupportType.STRUCTURED_SOURCE.value,
            extraction_method="adapter:ncbi_biosample",
            review_status=ReviewStatus.ACCEPTED.value,
        )
    )


def _source_material_id_fact(session, entity: Entity, study: Study, value: str, *, fact_type: str) -> None:
    session.add(
        RawFact(
            study_id=study.study_id,
            entity_id=entity.entity_id,
            fact_type_candidate=fact_type,
            raw_value=value,
            entity_level=EntityLevel.SAMPLE.value,
            support_type=SupportType.STRUCTURED_SOURCE.value,
            extraction_method="adapter:ncbi_biosample",
            review_status=ReviewStatus.ACCEPTED.value,
        )
    )


def test_unambiguous_match_via_hyphenated_fact_type(db_session):
    study = _study(db_session, title="Paper A")
    alias = _sample_entity(db_session, study, "GC04_1")
    canonical = _sample_entity(db_session, study, "SAMN11268098")
    _accession_fact(db_session, canonical, study)
    _source_material_id_fact(db_session, canonical, study, "GS14-GC04-1", fact_type="source-material-id")
    db_session.commit()

    result = reconcile_sample_aliases(db_session, study.study_id)
    db_session.commit()

    assert result.matched == 1
    assert result.ambiguous == 0
    relationship = db_session.query(EntityRelationship).filter_by(
        from_entity_id=alias.entity_id,
        to_entity_id=canonical.entity_id,
        relationship_type=EntityRelationshipType.SAME_PHYSICAL_SAMPLE_AS.value,
    ).one()
    assert relationship is not None
    match_fact = db_session.query(RawFact).filter_by(
        entity_id=alias.entity_id, fact_type_candidate="sample_alias_match"
    ).one()
    assert match_fact.raw_value == canonical.entity_id
    assert match_fact.review_status == ReviewStatus.ACCEPTED.value


def test_ambiguous_match_flags_needs_review_and_creates_no_relationship(db_session):
    study = _study(db_session, title="Paper B")
    alias = _sample_entity(db_session, study, "GC04_1")
    canonical_1 = _sample_entity(db_session, study, "SAMN0001")
    canonical_2 = _sample_entity(db_session, study, "SAMN0002")
    _accession_fact(db_session, canonical_1, study)
    _accession_fact(db_session, canonical_2, study)
    _source_material_id_fact(db_session, canonical_1, study, "GS14-GC04-1", fact_type="source_material_id")
    _source_material_id_fact(db_session, canonical_2, study, "GS99-GC04-1", fact_type="source_material_id")
    db_session.commit()

    result = reconcile_sample_aliases(db_session, study.study_id)
    db_session.commit()

    assert result.matched == 0
    assert result.ambiguous == 1
    assert db_session.query(EntityRelationship).filter_by(from_entity_id=alias.entity_id).count() == 0
    flag = db_session.query(RawFact).filter_by(
        entity_id=alias.entity_id, fact_type_candidate="sample_alias_ambiguous"
    ).one()
    assert flag.review_status == ReviewStatus.NEEDS_REVIEW.value
    assert set(flag.confidence_metadata["candidate_entity_ids"]) == {canonical_1.entity_id, canonical_2.entity_id}


def test_zero_matches_is_silent_no_op(db_session):
    study = _study(db_session, title="Paper C")
    alias = _sample_entity(db_session, study, "GC04_1")
    canonical = _sample_entity(db_session, study, "SAMN0003")
    _accession_fact(db_session, canonical, study)
    _source_material_id_fact(db_session, canonical, study, "totally-unrelated-name", fact_type="source_material_id")
    db_session.commit()

    result = reconcile_sample_aliases(db_session, study.study_id)
    db_session.commit()

    assert result.matched == 0
    assert result.ambiguous == 0
    assert db_session.query(EntityRelationship).filter_by(from_entity_id=alias.entity_id).count() == 0
    assert db_session.query(RawFact).filter_by(entity_id=alias.entity_id).count() == 0


def test_idempotent_rerun_does_not_duplicate_relationship_or_audit_fact(db_session):
    study = _study(db_session, title="Paper D")
    _sample_entity(db_session, study, "GC04_1")
    canonical = _sample_entity(db_session, study, "SAMN0004")
    _accession_fact(db_session, canonical, study)
    _source_material_id_fact(db_session, canonical, study, "GS14-GC04-1", fact_type="source_material_id")
    db_session.commit()

    first = reconcile_sample_aliases(db_session, study.study_id)
    db_session.commit()
    second = reconcile_sample_aliases(db_session, study.study_id)
    db_session.commit()

    assert first.matched == 1
    assert second.matched == 0  # already-aliased entity is excluded from later runs
    assert db_session.query(EntityRelationship).filter_by(to_entity_id=canonical.entity_id).count() == 1
    assert db_session.query(RawFact).filter_by(fact_type_candidate="sample_alias_match").count() == 1


def test_pure_depth_style_value_produces_no_false_match(db_session):
    """A different study's own source_material_id convention embeds only
    depth text (e.g. "3500 m V3-V4", the real convention seen elsewhere in
    this codebase's sources/ncbi.py) -- must not spuriously trailing-match
    an unrelated alias's native sample name."""
    study = _study(db_session, title="Paper E")
    alias = _sample_entity(db_session, study, "GC04_1")
    canonical = _sample_entity(db_session, study, "SAMN0005")
    _accession_fact(db_session, canonical, study)
    _source_material_id_fact(db_session, canonical, study, "3500 m V3-V4", fact_type="source_material_id")
    db_session.commit()

    result = reconcile_sample_aliases(db_session, study.study_id)
    db_session.commit()

    assert result.matched == 0
    assert result.ambiguous == 0
    assert db_session.query(EntityRelationship).filter_by(from_entity_id=alias.entity_id).count() == 0


def test_dedup_by_distinct_entity_id_for_shared_entity_with_duplicate_facts(db_session):
    """A shared canonical entity (linked to two studies) carries one
    source_material_id RawFact per linked study's own independent
    resolution pass -- must dedup to one distinct entity_id match, not
    read as two candidates and flag false ambiguity."""
    home_study = _study(db_session, title="Paper F, home")
    other_study = _study(db_session, title="Paper F, other citing paper")
    alias = _sample_entity(db_session, home_study, "GC04_1")
    canonical = _sample_entity(db_session, home_study, "SAMN0006")
    _accession_fact(db_session, canonical, home_study)
    session_link = EntityStudy(
        entity_id=canonical.entity_id,
        study_id=other_study.study_id,
        relationship_type=RelationshipType.SHARES_ACCESSION_WITH.value,
        confidence=SupportType.STRUCTURED_SOURCE.value,
    )
    db_session.add(session_link)
    # Duplicate source_material_id RawFact rows -- one per linked study's
    # own resolution pass, same identical value.
    _source_material_id_fact(db_session, canonical, home_study, "GS14-GC04-1", fact_type="source_material_id")
    _source_material_id_fact(db_session, canonical, other_study, "GS14-GC04-1", fact_type="source_material_id")
    db_session.commit()

    result = reconcile_sample_aliases(db_session, home_study.study_id)
    db_session.commit()

    assert result.matched == 1
    assert result.ambiguous == 0
    assert db_session.query(EntityRelationship).filter_by(from_entity_id=alias.entity_id).count() == 1
