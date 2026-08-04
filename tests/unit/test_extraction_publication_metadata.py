"""Tests against the real cached JATS full text for DOI 10.7717/peerj.333
(PMC3994630) -- the user's own validation paper for this deterministic
project-metadata extraction work. `fetch_fulltext_xml` hits the on-disk
retrieval cache, no live network call.
"""
import pytest

from fair_ocean_agent.database.enums import SupportType
from fair_ocean_agent.extraction.publication_metadata import (
    extract_code_repo_from_text,
    extract_from_jats_authors,
    extract_from_jats_permissions,
    extract_publication_metadata_facts,
    format_bibliographic_citation,
)
from fair_ocean_agent.workflow.handlers import _build_enabled_adapters

PMCID = "PMC3994630"
DOI = "10.7717/peerj.333"


@pytest.fixture(scope="module")
def real_fulltext_xml():
    adapters = _build_enabled_adapters()
    europe_pmc = adapters.get("europe_pmc")
    if europe_pmc is None:
        pytest.skip("europe_pmc adapter not enabled")
    return europe_pmc.fetch_fulltext_xml(PMCID)


@pytest.fixture(scope="module")
def real_crossref_raw():
    adapters = _build_enabled_adapters()
    crossref = adapters.get("crossref")
    if crossref is None:
        pytest.skip("crossref adapter not enabled")
    return crossref.fetch_record(DOI).raw


def test_permissions_gives_license_accessrights_rightsholder(real_fulltext_xml):
    facts = {f.fact_type_candidate: f for f in extract_from_jats_permissions(real_fulltext_xml, locator_prefix="t")}
    assert facts["license"].raw_value == "http://creativecommons.org/licenses/by/3.0/"
    assert facts["license"].support_type == SupportType.STRUCTURED_SOURCE
    assert facts["accessRights"].raw_value == "open access"
    assert facts["rightsHolder"].raw_value == "Davies et al."


def test_authors_includes_paper_authors_excludes_editor(real_fulltext_xml):
    facts = {f.fact_type_candidate: f for f in extract_from_jats_authors(real_fulltext_xml, locator_prefix="t")}
    recorded_by = facts["recordedBy"].raw_value
    for author in ("Sarah W. Davies", "Eli Meyer", "Sarah M. Guermond", "Mikhail V. Matz"):
        assert author in recorded_by
    assert "Qian" not in recorded_by  # the editor, not an author -- must never be conflated
    assert facts["project_contact"].raw_value == "daviessw@gmail.com"
    # This real article's JATS XML carries no ORCID for any author --
    # recordedByID must correctly produce nothing, not fabricate a value.
    assert "recordedByID" not in facts


def test_code_repo_correctly_absent_for_this_paper(real_fulltext_xml):
    """This 2014 paper predates common in-text GitHub citation and has no
    code-repository link anywhere in its full text (confirmed) -- a real
    no-false-positive test, not just a happy-path one."""
    assert extract_code_repo_from_text(real_fulltext_xml, locator_prefix="t") == []


def test_code_repo_detects_a_real_github_url():
    xml = "<article><body><p>Code available at https://github.com/someorg/somerepo.</p></body></article>"
    facts = extract_code_repo_from_text(xml, locator_prefix="t")
    assert len(facts) == 1
    assert facts[0].raw_value == "https://github.com/someorg/somerepo"
    assert facts[0].support_type == SupportType.DETERMINISTICALLY_DERIVED


def test_bibliographic_citation_composed_from_crossref(real_crossref_raw):
    facts = format_bibliographic_citation(real_crossref_raw, locator_prefix="t")
    assert len(facts) == 1
    citation = facts[0].raw_value
    assert "Davies SW" in citation
    assert "2014" in citation
    assert "A cross-ocean comparison of responses to settlement cues in reef-building corals" in citation
    assert "PeerJ" in citation
    assert DOI in citation


def test_bibliographic_citation_absent_without_crossref_data():
    assert format_bibliographic_citation(None, locator_prefix="t") == []
    assert format_bibliographic_citation({}, locator_prefix="t") == []


def test_extract_publication_metadata_facts_merges_everything(real_fulltext_xml, real_crossref_raw):
    facts = extract_publication_metadata_facts(real_fulltext_xml, real_crossref_raw, locator_prefix="t")
    fact_types = {f.fact_type_candidate for f in facts}
    assert fact_types == {
        "license",
        "accessRights",
        "rightsHolder",
        "recordedBy",
        "project_contact",
        "bibliographicCitation",
    }


def test_malformed_xml_returns_empty_not_raises():
    assert extract_from_jats_permissions("<not valid xml", locator_prefix="t") == []
    assert extract_from_jats_authors("<not valid xml", locator_prefix="t") == []
