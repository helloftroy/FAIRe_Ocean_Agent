"""Tests for extraction/taxonomic_assay.py::sync_assay_target_taxa_from_biosample_organisms."""
from fair_ocean_agent.database.enums import EntityLevel, RelationshipType, ReviewStatus, SupportType
from fair_ocean_agent.database.models import Entity, EntityStudy, RawFact, Study
from fair_ocean_agent.extraction.taxonomic_assay import sync_assay_target_taxa_from_biosample_organisms


def _study(session, **kwargs) -> Study:
    study = Study(**kwargs)
    session.add(study)
    session.flush()
    return study


def _linked_sample_with_organism(session, study: Study, external_identifier: str, organism: str) -> Entity:
    entity = Entity(study_id=study.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier=external_identifier)
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
    session.add(
        RawFact(
            study_id=study.study_id,
            entity_id=entity.entity_id,
            raw_field_name="organism",
            raw_value=organism,
            fact_type_candidate="organism",
            entity_level=EntityLevel.SAMPLE.value,
            support_type=SupportType.STRUCTURED_SOURCE.value,
        )
    )
    session.flush()
    return entity


def test_aggregates_distinct_organism_values_across_samples(db_session):
    study = _study(db_session, title="Multi-sample organism aggregation")
    _linked_sample_with_organism(db_session, study, "SAMN1", "seawater metagenome")
    _linked_sample_with_organism(db_session, study, "SAMN2", "seawater metagenome")
    _linked_sample_with_organism(db_session, study, "SAMN3", "marine sediment metagenome")
    db_session.commit()

    sync_assay_target_taxa_from_biosample_organisms(db_session, study.study_id)
    db_session.commit()

    fact = db_session.query(RawFact).filter_by(
        study_id=study.study_id, fact_type_candidate="assay_target_taxa"
    ).one()
    assert fact.raw_value == "seawater metagenome | marine sediment metagenome"
    assert fact.support_type == SupportType.STRUCTURED_SOURCE.value
    assert fact.entity_id is None


def test_no_organism_facts_is_a_silent_no_op(db_session):
    study = _study(db_session, title="No BioSample data")
    db_session.commit()

    sync_assay_target_taxa_from_biosample_organisms(db_session, study.study_id)
    db_session.commit()

    assert db_session.query(RawFact).filter_by(study_id=study.study_id, fact_type_candidate="assay_target_taxa").count() == 0


def test_idempotent_rerun_updates_in_place_rather_than_duplicating(db_session):
    study = _study(db_session, title="Rerun-safe aggregation")
    _linked_sample_with_organism(db_session, study, "SAMN1", "seawater metagenome")
    db_session.commit()

    sync_assay_target_taxa_from_biosample_organisms(db_session, study.study_id)
    db_session.commit()

    _linked_sample_with_organism(db_session, study, "SAMN2", "marine sediment metagenome")
    db_session.commit()

    sync_assay_target_taxa_from_biosample_organisms(db_session, study.study_id)
    db_session.commit()

    facts = db_session.query(RawFact).filter_by(study_id=study.study_id, fact_type_candidate="assay_target_taxa").all()
    assert len(facts) == 1
    assert facts[0].raw_value == "seawater metagenome | marine sediment metagenome"


def test_self_heals_by_rejecting_the_fact_when_organisms_disappear(db_session):
    study = _study(db_session, title="Organism fact later rejected")
    sample = _linked_sample_with_organism(db_session, study, "SAMN1", "seawater metagenome")
    db_session.commit()

    sync_assay_target_taxa_from_biosample_organisms(db_session, study.study_id)
    db_session.commit()

    organism_fact = db_session.query(RawFact).filter_by(entity_id=sample.entity_id, fact_type_candidate="organism").one()
    organism_fact.review_status = ReviewStatus.REJECTED.value
    db_session.commit()

    sync_assay_target_taxa_from_biosample_organisms(db_session, study.study_id)
    db_session.commit()

    aggregated = db_session.query(RawFact).filter_by(study_id=study.study_id, fact_type_candidate="assay_target_taxa").one()
    assert aggregated.review_status == ReviewStatus.REJECTED.value
