"""Offline tests: construct a SourceRecord directly from a saved fixture and
check extract_structured_facts/parse_publication_fields/find_related -- no
network, no live API dependency. HTTP/caching/retry behavior is covered
separately in test_source_adapters_http.py.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from fair_ocean_agent.database.enums import AccessStatus, IdentifierType
from fair_ocean_agent.sources.base import SourceConfig, SourceRecord, hash_payload
from fair_ocean_agent.sources.crossref import CrossrefAdapter
from fair_ocean_agent.sources.europe_pmc import EuropePmcAdapter
from fair_ocean_agent.sources.openalex import OpenAlexAdapter

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text())


def _record(source_name: str, raw: dict, external_identifier: str = "10.1002/2015jg003300") -> SourceRecord:
    return SourceRecord(
        source_name=source_name,
        external_identifier=external_identifier,
        raw=raw,
        retrieved_at=datetime.now(timezone.utc),
        content_hash=hash_payload(raw),
    )


@pytest.fixture
def crossref_adapter(retrieval_config):
    adapter = CrossrefAdapter(
        SourceConfig(name="crossref", enabled=True, base_url="https://api.crossref.org"), retrieval_config
    )
    yield adapter
    adapter.close()


@pytest.fixture
def europe_pmc_adapter(retrieval_config):
    adapter = EuropePmcAdapter(
        SourceConfig(name="europe_pmc", enabled=True, base_url="https://www.ebi.ac.uk/europepmc/webservices/rest"),
        retrieval_config,
    )
    yield adapter
    adapter.close()


@pytest.fixture
def openalex_adapter(retrieval_config):
    adapter = OpenAlexAdapter(
        SourceConfig(name="openalex", enabled=True, base_url="https://api.openalex.org"), retrieval_config
    )
    yield adapter
    adapter.close()


def test_crossref_parse_publication_fields(crossref_adapter):
    record = _record("crossref", load_fixture("crossref_work.json"))
    fields = crossref_adapter.parse_publication_fields(record)

    assert fields["doi"] == "10.1002/2015jg003300"
    assert fields["title"].startswith("Shifts in the community structure")
    assert fields["publication_year"] == 2016
    assert fields["journal"] == "Journal of Geophysical Research: Biogeosciences"
    assert fields["authors"] == [
        {"given": "Jane", "family": "Doe"},
        {"given": "John", "family": "Smith"},
    ]


def test_crossref_extract_structured_facts_have_evidence_locators(crossref_adapter):
    record = _record("crossref", load_fixture("crossref_work.json"))
    facts = crossref_adapter.extract_structured_facts(record)

    assert len(facts) > 0
    assert all(f.source_locator.startswith("crossref.message.") for f in facts)
    title_fact = next(f for f in facts if f.fact_type_candidate == "title")
    assert "Shifts in the community structure" in title_fact.raw_value


def test_crossref_extract_structured_facts_skips_absent_fields(crossref_adapter):
    record = _record("crossref", {"DOI": "10.1/x", "title": []})
    facts = crossref_adapter.extract_structured_facts(record)
    assert all(f.fact_type_candidate != "title" for f in facts)


def test_crossref_title_fact_is_a_plain_string_not_a_json_encoded_list(crossref_adapter):
    """Regression test: a real live-validation cross-source comparison
    (validation/cross_source.py) found crossref's "title" and
    "container-title" facts stored as '["Some Title"]' -- Crossref's API
    returns these as lists-of-one-string, and the raw fact was storing the
    JSON-encoded list instead of unwrapping it like
    parse_publication_fields already did. This made every crossref title
    compare as "different" from Europe PMC/OpenAlex's plain-string titles,
    even when they were the same title -- a false cross-source conflict."""
    record = _record("crossref", load_fixture("crossref_work.json"))
    facts = crossref_adapter.extract_structured_facts(record)

    title_fact = next(f for f in facts if f.fact_type_candidate == "title")
    container_fact = next(f for f in facts if f.fact_type_candidate == "container-title")

    assert not title_fact.raw_value.startswith("[")
    assert title_fact.raw_value == "Shifts in the community structure of anaerobic ammonium oxidation bacteria"
    assert not container_fact.raw_value.startswith("[")
    assert container_fact.raw_value == "Journal of Geophysical Research: Biogeosciences"


def test_europe_pmc_parse_publication_fields_open_access(europe_pmc_adapter):
    record = _record("europe_pmc", load_fixture("europe_pmc_result.json"))
    fields = europe_pmc_adapter.parse_publication_fields(record)

    assert fields["pmid"] == "27000000"
    assert fields["pmcid"] == "PMC5000000"
    assert fields["open_access_status"] == AccessStatus.OPEN.value
    assert fields["fulltext_available"] is True


def test_europe_pmc_parse_publication_fields_not_open_access(europe_pmc_adapter):
    record = _record("europe_pmc", {"pmid": "1", "isOpenAccess": "N", "inEPMC": "N"})
    fields = europe_pmc_adapter.parse_publication_fields(record)

    assert fields["open_access_status"] == AccessStatus.UNKNOWN.value
    assert fields["fulltext_available"] is False


def test_europe_pmc_find_related_returns_pmid_and_pmcid(europe_pmc_adapter):
    record = _record("europe_pmc", load_fixture("europe_pmc_result.json"))
    related = europe_pmc_adapter.find_related(record)

    types = {r.identifier_type for r in related}
    assert types == {IdentifierType.PMID, IdentifierType.PMCID}


def test_openalex_parse_publication_fields(openalex_adapter):
    record = _record("openalex", load_fixture("openalex_work.json"))
    fields = openalex_adapter.parse_publication_fields(record)

    assert fields["openalex_id"] == "W2413577766"
    assert fields["publication_year"] == 2016
    assert fields["open_access_status"] == AccessStatus.OPEN.value


def test_openalex_find_related_strips_full_url(openalex_adapter):
    record = _record("openalex", load_fixture("openalex_work.json"))
    related = openalex_adapter.find_related(record)

    assert len(related) == 1
    assert related[0].identifier_type == IdentifierType.OPENALEX_ID
    assert related[0].value == "W2413577766"
