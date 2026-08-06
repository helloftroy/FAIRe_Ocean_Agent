"""Tests for identity/resolution.py's resolve_or_create_study() -- the
single merge-decision function both DOI-seeded and accession-seeded
discovery share. Exercises the function directly (not through the task
queue/handlers), constructing Study/Source/RawFact/ExternalIdentifier rows
by hand to set up each confidence-tier and consistency-check scenario."""
from fair_ocean_agent.database.enums import (
    CandidateMatchReviewStatus,
    CanonicalStatus,
    EntityLevel,
    IdentifierType,
    RelationshipType,
    SourceType,
    SupportType,
)
from fair_ocean_agent.database.models import CandidateMatch, ExternalIdentifier, RawFact, Source, Study
from fair_ocean_agent.identity.resolution import resolve_or_create_study
from fair_ocean_agent.identity.source_linking import create_source
from fair_ocean_agent.sources.base import RelatedIdentifier


def _study(session, **kwargs) -> Study:
    study = Study(**kwargs)
    session.add(study)
    session.flush()
    return study


def _rel(identifier_type, value, *, confidence, source="ena") -> RelatedIdentifier:
    return RelatedIdentifier(
        identifier_type=identifier_type, value=value,
        relationship_type=RelationshipType.RELATED_TO, source=source, confidence=confidence,
    )


def test_no_match_records_identifier_on_current_study(db_session):
    study = _study(db_session, title="Current")
    rel = _rel(IdentifierType.PMID, "99999", confidence=SupportType.STRUCTURED_SOURCE)

    result = resolve_or_create_study(db_session, study, [rel], source=None)
    db_session.commit()

    assert result.study_id == study.study_id
    ei = db_session.query(ExternalIdentifier).filter_by(study_id=study.study_id, identifier_value="99999").one()
    assert ei.identifier_type == IdentifierType.PMID.value
    assert db_session.query(CandidateMatch).count() == 0
    assert db_session.query(Study).count() == 1


def test_tier1_auto_links_without_consistency_check(db_session):
    other_study = _study(db_session, title="Other")
    db_session.add(
        ExternalIdentifier(study_id=other_study.study_id, identifier_type=IdentifierType.PMID.value, identifier_value="12345")
    )
    db_session.commit()

    study = _study(db_session, title="Current")
    rel = _rel(IdentifierType.PMID, "12345", confidence=SupportType.STRUCTURED_SOURCE, source="europe_pmc")

    result = resolve_or_create_study(db_session, study, [rel], source=None)
    db_session.commit()

    assert result.study_id == study.study_id
    db_session.refresh(other_study)
    assert other_study.canonical_status == CanonicalStatus.MERGED.value
    assert db_session.query(CandidateMatch).count() == 0


def test_tier2_consistent_dates_attaches_via_merge(db_session):
    other_study = _study(db_session, title="Other")
    db_session.add(
        ExternalIdentifier(study_id=other_study.study_id, identifier_type=IdentifierType.BIOPROJECT_ACCESSION.value, identifier_value="PRJNA1")
    )
    db_session.add(
        RawFact(
            study_id=other_study.study_id, entity_level=EntityLevel.SAMPLE.value,
            fact_type_candidate="collection_date", raw_value="2020-06-15",
            support_type=SupportType.STRUCTURED_SOURCE.value,
        )
    )
    db_session.commit()

    study = _study(db_session, title="Current")
    source = create_source(db_session, Source(study_id=study.study_id, source_type=SourceType.REPOSITORY_API.value, source_name="ena"))
    db_session.add(
        RawFact(
            study_id=study.study_id, source_id=source.source_id, entity_level=EntityLevel.SAMPLE.value,
            fact_type_candidate="collection_date", raw_value="2020-07-01",
            support_type=SupportType.STRUCTURED_SOURCE.value,
        )
    )
    db_session.commit()

    rel = _rel(IdentifierType.BIOPROJECT_ACCESSION, "PRJNA1", confidence=SupportType.DETERMINISTICALLY_DERIVED)
    result = resolve_or_create_study(db_session, study, [rel], source=source)
    db_session.commit()

    assert result.study_id == study.study_id
    db_session.refresh(other_study)
    assert other_study.canonical_status == CanonicalStatus.MERGED.value
    assert db_session.query(CandidateMatch).count() == 0


def test_tier2_inconsistent_dates_creates_sibling_and_flags(db_session):
    other_study = _study(db_session, title="Other")
    db_session.add(
        ExternalIdentifier(study_id=other_study.study_id, identifier_type=IdentifierType.BIOPROJECT_ACCESSION.value, identifier_value="PRJNA1")
    )
    other_fact = RawFact(
        study_id=other_study.study_id, entity_level=EntityLevel.SAMPLE.value,
        fact_type_candidate="collection_date", raw_value="2010-01-01",
        support_type=SupportType.STRUCTURED_SOURCE.value,
    )
    db_session.add(other_fact)
    db_session.commit()

    study = _study(db_session, title="Current")
    source = create_source(db_session, Source(study_id=study.study_id, source_type=SourceType.REPOSITORY_API.value, source_name="ena"))
    new_fact = RawFact(
        study_id=study.study_id, source_id=source.source_id, entity_level=EntityLevel.SAMPLE.value,
        fact_type_candidate="collection_date", raw_value="2022-01-01",
        support_type=SupportType.STRUCTURED_SOURCE.value,
    )
    db_session.add(new_fact)
    db_session.commit()

    original_study_count = db_session.query(Study).count()
    rel = _rel(IdentifierType.BIOPROJECT_ACCESSION, "PRJNA1", confidence=SupportType.DETERMINISTICALLY_DERIVED)
    result = resolve_or_create_study(db_session, study, [rel], source=source)
    db_session.commit()

    # `study` itself is returned unchanged -- the split evidence moved to a new sibling.
    assert result.study_id == study.study_id
    assert db_session.query(Study).count() == original_study_count + 1

    db_session.refresh(source)
    sibling_id = source.study_id
    assert sibling_id not in (study.study_id, other_study.study_id)

    db_session.refresh(new_fact)
    assert new_fact.study_id == sibling_id

    # other_study's own pre-existing evidence is provably untouched.
    db_session.refresh(other_fact)
    assert other_fact.study_id == other_study.study_id
    db_session.refresh(other_study)
    assert other_study.canonical_status == CanonicalStatus.CANDIDATE.value

    candidate_match = db_session.query(CandidateMatch).one()
    assert candidate_match.review_status == CandidateMatchReviewStatus.PENDING.value
    assert candidate_match.study_a_id == sibling_id
    assert candidate_match.study_b_id == other_study.study_id

    sibling_identifier = db_session.query(ExternalIdentifier).filter_by(study_id=sibling_id).one()
    assert sibling_identifier.relationship_type == RelationshipType.SHARES_ACCESSION_WITH.value


def test_shared_biosample_accession_never_merges_or_flags(db_session):
    """BIOSAMPLE_ACCESSION is excluded from Study-identity resolution
    entirely, at ANY confidence tier -- unlike every other identifier type
    (see resolve_or_create_study's own carve-out). Regression test for a
    real finding from a live end-to-end run: two genuinely different papers
    sharing a real BioSample (confirmed live:
    10.1038/s42003-024-06136-2 / 10.1073/pnas.2005917117, both citing
    PRJNA529480) got silently merged via this exact code path, since
    NcbiBioSampleAdapter/EnaAdapter's find_related() reports every sample
    accession as a default-STRUCTURED_SOURCE RelatedIdentifier."""
    other_study = _study(db_session, title="Other paper, real different study")
    db_session.add(
        ExternalIdentifier(study_id=other_study.study_id, identifier_type=IdentifierType.BIOSAMPLE_ACCESSION.value, identifier_value="SAMN1")
    )
    db_session.commit()

    study = _study(db_session, title="Current paper, also cites SAMN1")
    rel = _rel(IdentifierType.BIOSAMPLE_ACCESSION, "SAMN1", confidence=SupportType.STRUCTURED_SOURCE, source="ncbi_biosample")

    original_study_count = db_session.query(Study).count()
    result = resolve_or_create_study(db_session, study, [rel], source=None)
    db_session.commit()

    assert result.study_id == study.study_id
    assert db_session.query(Study).count() == original_study_count  # no merge, no sibling
    db_session.refresh(other_study)
    assert other_study.canonical_status == CanonicalStatus.CANDIDATE.value  # not merged away
    assert db_session.query(CandidateMatch).count() == 0  # not flagged as ambiguous either

    recorded = db_session.query(ExternalIdentifier).filter_by(
        study_id=study.study_id, identifier_type=IdentifierType.BIOSAMPLE_ACCESSION.value, identifier_value="SAMN1"
    ).one()
    assert recorded.relationship_type == RelationshipType.SHARES_ACCESSION_WITH.value


def test_tier2_inconsistent_dates_but_discovery_lineage_skips_sibling(db_session):
    """Regression test for a real finding from a live end-to-end run: a
    citation-discovered Study's own full-text scan re-mentions the exact
    BioProject accession that discovery already used to link it to its
    parent -- this must NOT be treated as an identity conflict (no sibling,
    no CandidateMatch), since the two studies are already known, by
    construction, to be a citing-paper/cited-paper pair that must stay
    separate regardless. Confirmed live: 10.1038/s42003-024-06136-2 /
    10.1073/pnas.2005917117 both resolving PRJNA529480 hit exactly this."""
    parent_study = _study(db_session, title="Original paper")
    db_session.add(
        ExternalIdentifier(study_id=parent_study.study_id, identifier_type=IdentifierType.BIOPROJECT_ACCESSION.value, identifier_value="PRJNA1")
    )
    db_session.add(
        RawFact(
            study_id=parent_study.study_id, entity_level=EntityLevel.SAMPLE.value,
            fact_type_candidate="collection_date", raw_value="2010-01-01",
            support_type=SupportType.STRUCTURED_SOURCE.value,
        )
    )
    db_session.commit()

    # citing_study was created by handle_discover_citing_studies -- its own
    # discovery_parent_study_id already establishes the relationship.
    citing_study = _study(
        db_session, title="Citing paper", discovery_depth=1,
        discovery_parent_study_id=parent_study.study_id, discovery_root_study_id=parent_study.study_id,
        discovery_trigger="bioproject_pubmed_citation",
    )
    # No date evidence on citing_study at all -- if the lineage check didn't
    # fire, check_study_consistency would have nothing to compare and this
    # would degrade to a sibling+flag exactly like the no-single-Source case.

    original_study_count = db_session.query(Study).count()
    rel = _rel(IdentifierType.BIOPROJECT_ACCESSION, "PRJNA1", confidence=SupportType.DETERMINISTICALLY_DERIVED)
    result = resolve_or_create_study(db_session, citing_study, [rel], source=None)
    db_session.commit()

    assert result.study_id == citing_study.study_id
    assert db_session.query(Study).count() == original_study_count  # no sibling created
    assert db_session.query(CandidateMatch).count() == 0  # no review-queue noise

    recorded = db_session.query(ExternalIdentifier).filter_by(
        study_id=citing_study.study_id, identifier_type=IdentifierType.BIOPROJECT_ACCESSION.value
    ).one()
    assert recorded.identifier_value == "PRJNA1"
    assert recorded.relationship_type == RelationshipType.SHARES_ACCESSION_WITH.value

    # parent_study is provably untouched -- not merged, not modified.
    db_session.refresh(parent_study)
    assert parent_study.canonical_status == CanonicalStatus.CANDIDATE.value


def test_tier3_never_merges_alone_even_when_consistent(db_session):
    other_study = _study(db_session, title="Other")
    db_session.add(
        ExternalIdentifier(study_id=other_study.study_id, identifier_type=IdentifierType.BIOPROJECT_ACCESSION.value, identifier_value="PRJNA1")
    )
    db_session.add(
        RawFact(
            study_id=other_study.study_id, entity_level=EntityLevel.SAMPLE.value,
            fact_type_candidate="collection_date", raw_value="2020-06-01",
            support_type=SupportType.STRUCTURED_SOURCE.value,
        )
    )
    db_session.commit()

    study = _study(db_session, title="Current")
    source = create_source(db_session, Source(study_id=study.study_id, source_type=SourceType.REPOSITORY_API.value, source_name="llm"))
    db_session.add(
        RawFact(
            study_id=study.study_id, source_id=source.source_id, entity_level=EntityLevel.SAMPLE.value,
            fact_type_candidate="collection_date", raw_value="2020-07-01",
            support_type=SupportType.STRUCTURED_SOURCE.value,
        )
    )
    db_session.commit()

    rel = _rel(IdentifierType.BIOPROJECT_ACCESSION, "PRJNA1", confidence=SupportType.INFERRED, source="llm")
    resolve_or_create_study(db_session, study, [rel], source=source)
    db_session.commit()

    db_session.refresh(other_study)
    assert other_study.canonical_status == CanonicalStatus.CANDIDATE.value  # never merged, even though dates agree
    assert db_session.query(CandidateMatch).count() == 1


def test_multiple_matches_picks_the_consistent_candidate(db_session):
    consistent_study = _study(db_session, title="Consistent")
    inconsistent_study = _study(db_session, title="Inconsistent")
    for other_study, date_value in ((consistent_study, "2020-06-01"), (inconsistent_study, "1999-01-01")):
        db_session.add(
            ExternalIdentifier(study_id=other_study.study_id, identifier_type=IdentifierType.BIOPROJECT_ACCESSION.value, identifier_value="PRJNA1")
        )
        db_session.add(
            RawFact(
                study_id=other_study.study_id, entity_level=EntityLevel.SAMPLE.value,
                fact_type_candidate="collection_date", raw_value=date_value,
                support_type=SupportType.STRUCTURED_SOURCE.value,
            )
        )
    db_session.commit()

    study = _study(db_session, title="Current")
    source = create_source(db_session, Source(study_id=study.study_id, source_type=SourceType.REPOSITORY_API.value, source_name="ena"))
    db_session.add(
        RawFact(
            study_id=study.study_id, source_id=source.source_id, entity_level=EntityLevel.SAMPLE.value,
            fact_type_candidate="collection_date", raw_value="2020-07-01",
            support_type=SupportType.STRUCTURED_SOURCE.value,
        )
    )
    db_session.commit()

    rel = _rel(IdentifierType.BIOPROJECT_ACCESSION, "PRJNA1", confidence=SupportType.DETERMINISTICALLY_DERIVED)
    result = resolve_or_create_study(db_session, study, [rel], source=source)
    db_session.commit()

    assert result.study_id == study.study_id
    db_session.refresh(consistent_study)
    db_session.refresh(inconsistent_study)
    assert consistent_study.canonical_status == CanonicalStatus.MERGED.value
    assert inconsistent_study.canonical_status == CanonicalStatus.CANDIDATE.value
    assert db_session.query(CandidateMatch).count() == 0


def test_multiple_matches_none_consistent_creates_sibling_flagged_against_all(db_session):
    study_a = _study(db_session, title="A")
    study_b = _study(db_session, title="B")
    for other_study, date_value in ((study_a, "1990-01-01"), (study_b, "1999-01-01")):
        db_session.add(
            ExternalIdentifier(study_id=other_study.study_id, identifier_type=IdentifierType.BIOPROJECT_ACCESSION.value, identifier_value="PRJNA1")
        )
        db_session.add(
            RawFact(
                study_id=other_study.study_id, entity_level=EntityLevel.SAMPLE.value,
                fact_type_candidate="collection_date", raw_value=date_value,
                support_type=SupportType.STRUCTURED_SOURCE.value,
            )
        )
    db_session.commit()

    study = _study(db_session, title="Current")
    source = create_source(db_session, Source(study_id=study.study_id, source_type=SourceType.REPOSITORY_API.value, source_name="ena"))
    db_session.add(
        RawFact(
            study_id=study.study_id, source_id=source.source_id, entity_level=EntityLevel.SAMPLE.value,
            fact_type_candidate="collection_date", raw_value="2020-07-01",
            support_type=SupportType.STRUCTURED_SOURCE.value,
        )
    )
    db_session.commit()

    original_study_count = db_session.query(Study).count()
    rel = _rel(IdentifierType.BIOPROJECT_ACCESSION, "PRJNA1", confidence=SupportType.DETERMINISTICALLY_DERIVED)
    resolve_or_create_study(db_session, study, [rel], source=source)
    db_session.commit()

    assert db_session.query(Study).count() == original_study_count + 1
    candidate_match = db_session.query(CandidateMatch).one()
    assert candidate_match.study_b_id is None
    assert set(candidate_match.candidate_record_ref.split(",")) == {study_a.study_id, study_b.study_id}


def test_inconsistent_with_no_single_source_degrades_to_flag_only(db_session):
    """Fulltext-mined identifiers (and supplement-discovery ext-links) have
    no single triggering Source -- resolve_or_create_study must still flag
    the ambiguity without attempting to reassign any rows."""
    other_study = _study(db_session, title="Other")
    db_session.add(
        ExternalIdentifier(study_id=other_study.study_id, identifier_type=IdentifierType.BIOPROJECT_ACCESSION.value, identifier_value="PRJNA1")
    )
    db_session.commit()

    study = _study(db_session, title="Current")
    rel = _rel(
        IdentifierType.BIOPROJECT_ACCESSION, "PRJNA1",
        confidence=SupportType.DETERMINISTICALLY_DERIVED, source="europe_pmc_fulltext_identifier_scan",
    )

    result = resolve_or_create_study(db_session, study, [rel], source=None)
    db_session.commit()

    assert result.study_id == study.study_id
    candidate_match = db_session.query(CandidateMatch).one()
    assert candidate_match.review_status == CandidateMatchReviewStatus.PENDING.value
    db_session.refresh(other_study)
    assert other_study.canonical_status == CanonicalStatus.CANDIDATE.value
