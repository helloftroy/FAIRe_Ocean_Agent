"""Tests for identity/component.py::compute_study_component -- confirms
BFS walks BOTH edge types (shared-entity + discovery-lineage), since
neither alone is sufficient (see that module's own docstring)."""
from fair_ocean_agent.database.enums import EntityLevel, RelationshipType, SupportType
from fair_ocean_agent.database.models import Entity, EntityStudy, Study
from fair_ocean_agent.identity.component import compute_study_component


def _study(session, **kwargs) -> Study:
    study = Study(**kwargs)
    session.add(study)
    session.flush()
    return study


def _entity_study(entity: Entity, study: Study, *, home: bool) -> EntityStudy:
    return EntityStudy(
        entity_id=entity.entity_id,
        study_id=study.study_id,
        relationship_type=(RelationshipType.IS_HOME_OF if home else RelationshipType.SHARES_ACCESSION_WITH).value,
        confidence=SupportType.STRUCTURED_SOURCE.value,
    )


def test_two_independently_seeded_studies_sharing_an_entity_are_one_component(db_session):
    """The case discovery_root_study_id alone would miss: neither study has
    a lineage relationship to the other at all."""
    study_a = _study(db_session, title="Original paper", discovery_depth=0)
    study_b = _study(db_session, title="Independently seeded reanalysis", discovery_depth=0)
    entity = Entity(study_id=study_a.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMN1")
    db_session.add(entity)
    db_session.flush()
    db_session.add_all([_entity_study(entity, study_a, home=True), _entity_study(entity, study_b, home=False)])
    db_session.commit()

    component = compute_study_component(db_session, study_a.study_id)

    assert component == {study_a.study_id, study_b.study_id}
    # Symmetric: starting from either study finds the same component.
    assert compute_study_component(db_session, study_b.study_id) == component


def test_freshly_created_citing_study_with_zero_entity_studies_is_still_in_component(db_session):
    """A study handle_discover_citing_studies just created has NO
    EntityStudy rows yet (its own DISCOVER_IDENTIFIERS hasn't run) --
    shared-entity edges alone would miss it; lineage edges must catch it."""
    parent = _study(db_session, title="Original paper", discovery_depth=0)
    citing = _study(
        db_session, title="Citing paper", discovery_depth=1,
        discovery_parent_study_id=parent.study_id, discovery_root_study_id=parent.study_id,
        discovery_trigger="bioproject_pubmed_citation",
    )
    db_session.commit()

    component = compute_study_component(db_session, parent.study_id)

    assert component == {parent.study_id, citing.study_id}
    assert compute_study_component(db_session, citing.study_id) == component


def test_unrelated_studies_are_disjoint_components(db_session):
    study_a = _study(db_session, title="Paper A")
    study_b = _study(db_session, title="Unrelated paper B")
    db_session.commit()

    assert compute_study_component(db_session, study_a.study_id) == {study_a.study_id}
    assert compute_study_component(db_session, study_b.study_id) == {study_b.study_id}


def test_three_way_component_via_mixed_edge_types(db_session):
    """A (seed) --lineage--> B (citing A's accession), and B --shared
    entity--> C (an independently-seeded study sharing an entity with B,
    not A) -- all three must land in one component via BFS transitivity."""
    study_a = _study(db_session, title="Root seed", discovery_depth=0)
    study_b = _study(
        db_session, title="Citing paper", discovery_depth=1,
        discovery_parent_study_id=study_a.study_id, discovery_root_study_id=study_a.study_id,
    )
    study_c = _study(db_session, title="Independently seeded, shares data with B", discovery_depth=0)
    entity = Entity(study_id=study_b.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMN2")
    db_session.add(entity)
    db_session.flush()
    db_session.add_all([_entity_study(entity, study_b, home=True), _entity_study(entity, study_c, home=False)])
    db_session.commit()

    component = compute_study_component(db_session, study_a.study_id)

    assert component == {study_a.study_id, study_b.study_id, study_c.study_id}
