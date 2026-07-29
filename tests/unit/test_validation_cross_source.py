from fair_ocean_agent.database.enums import ValidationStatus
from fair_ocean_agent.database.models import RawFact, Source, Study
from fair_ocean_agent.validation.cross_source import compare_field_across_sources


def _study_with_title_facts(session, titles_by_source: dict[str, str]) -> Study:
    study = Study(title="A study")
    session.add(study)
    session.flush()
    for source_name, title in titles_by_source.items():
        source = Source(study_id=study.study_id, source_type="publication_api", source_name=source_name)
        session.add(source)
        session.flush()
        session.add(
            RawFact(
                study_id=study.study_id,
                source_id=source.source_id,
                raw_field_name="title",
                raw_value=title,
                fact_type_candidate="title",
                entity_level="study",
                support_type="structured_source",
            )
        )
    session.flush()
    return study


def test_agreeing_sources_are_confirmed(db_session):
    study = _study_with_title_facts(db_session, {"crossref": "Same Title", "europe_pmc": "Same Title"})
    result = compare_field_across_sources(db_session, study.study_id, "title")
    assert result.status == ValidationStatus.CONFIRMED.value


def test_trailing_period_difference_is_not_a_conflict(db_session):
    """Regression test: a live run across 101 real studies found 61/98
    "conflicting" title comparisons, of which 47 were purely because Europe
    PMC's title field always ends with a period and Crossref/OpenAlex's
    don't -- not a real disagreement."""
    study = _study_with_title_facts(
        db_session, {"crossref": "Same Title", "europe_pmc": "Same Title."}
    )
    result = compare_field_across_sources(db_session, study.study_id, "title")
    assert result.status == ValidationStatus.CONFIRMED.value


def test_case_and_whitespace_differences_are_not_a_conflict(db_session):
    study = _study_with_title_facts(db_session, {"crossref": "Some Title", "europe_pmc": "  some   title  "})
    result = compare_field_across_sources(db_session, study.study_id, "title")
    assert result.status == ValidationStatus.CONFIRMED.value


def test_inline_markup_tags_are_not_a_conflict(db_session):
    """Regression test: a 50-study live audit found 6/7 "conflicting" title
    comparisons were Crossref/OpenAlex titles carrying inline markup
    (<i>species name</i>, <scp>DNA</scp>) that Europe PMC's plain-text
    title never has -- not a real disagreement."""
    study = _study_with_title_facts(
        db_session,
        {
            "crossref": "Environmental\n                    <scp>DNA</scp>\n                    persistence and fish detection in captive sponges",
            "europe_pmc": "Environmental DNA persistence and fish detection in captive sponges.",
        },
    )
    result = compare_field_across_sources(db_session, study.study_id, "title")
    assert result.status == ValidationStatus.CONFIRMED.value


def test_unicode_dash_variants_are_not_a_conflict(db_session):
    """Regression test: same live audit found Crossref/OpenAlex using a
    Unicode en-dash/non-breaking hyphen ("mass–driven") where Europe PMC's
    title uses a plain ASCII hyphen ("mass-driven") for the same word."""
    study = _study_with_title_facts(
        db_session,
        {
            "crossref": "Capacity of deep‐sea corals to obtain nutrition from cold seeps",
            "europe_pmc": "Capacity of deep-sea corals to obtain nutrition from cold seeps.",
        },
    )
    result = compare_field_across_sources(db_session, study.study_id, "title")
    assert result.status == ValidationStatus.CONFIRMED.value


def test_genuinely_different_titles_are_conflicting(db_session):
    """Regression test: a live run found OpenAlex's own API returning the
    wrong title for a real DOI (a genuine third-party data-quality issue,
    confirmed directly against OpenAlex's API, not an adapter bug) -- this
    must still be caught, not smoothed over by normalization."""
    study = _study_with_title_facts(
        db_session,
        {
            "crossref": "Symbiosis modulates gene expression of symbionts",
            "openalex": "Discovery Association Rules in Time Series Data",
        },
    )
    result = compare_field_across_sources(db_session, study.study_id, "title")
    assert result.status == ValidationStatus.CONFLICTING.value


def test_single_source_is_not_assessed(db_session):
    study = _study_with_title_facts(db_session, {"crossref": "Only One Source"})
    result = compare_field_across_sources(db_session, study.study_id, "title")
    assert result.status == ValidationStatus.NOT_ASSESSED.value


def test_no_sources_is_not_assessed(db_session):
    study = Study(title="No facts at all")
    db_session.add(study)
    db_session.flush()
    result = compare_field_across_sources(db_session, study.study_id, "title")
    assert result.status == ValidationStatus.NOT_ASSESSED.value
