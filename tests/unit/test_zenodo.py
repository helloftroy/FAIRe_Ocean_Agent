"""Integration tests for ZenodoAdapter's fetch_record via httpx.MockTransport
-- no live network, but exercises the real parsing/redirect-handling code
path, fixtures shaped from the real live response for 10.5281/zenodo.10381280
(10.1038/s41598-024-60762-8's real Zenodo dataset)."""
import httpx
import pytest

from fair_ocean_agent.sources.base import SourceConfig, SourceRecordNotFoundError
from fair_ocean_agent.sources.sequence_file_heuristics import SequenceDataStatus
from fair_ocean_agent.sources.zenodo import ZenodoAdapter

RECORD_JSON = {
    "id": 10381281,
    "conceptrecid": "10381280",
    "doi": "10.5281/zenodo.10381281",
    "conceptdoi": "10.5281/zenodo.10381280",
    "doi_url": "https://doi.org/10.5281/zenodo.10381281",
    "metadata": {
        "title": "Harnessing eDNA metabarcoding to investigate fish community composition",
        "description": "eDNA metabarcoding using MiFish and Elas02 primer sets.",
        "license": {"id": "cc-by-4.0"},
    },
    "links": {
        "self": "https://zenodo.org/api/records/10381281",
        "self_html": "https://zenodo.org/records/10381281",
        "files": "https://zenodo.org/api/records/10381281/files",
    },
}
FILES_JSON = {
    "entries": [
        {"key": "Elas02.rar", "size": 141252940},
        {"key": "MiFish.rar", "size": 259064594},
    ]
}


def _transport(record_json=RECORD_JSON, files_json=FILES_JSON, redirect_from=None):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if redirect_from and path.endswith(f"/records/{redirect_from}"):
            return httpx.Response(302, headers={"location": record_json["links"]["self"]})
        if path.endswith("/files"):
            return httpx.Response(200, json=files_json)
        if "/records/" in path:
            return httpx.Response(200, json=record_json)
        raise AssertionError(f"unexpected request: {request.url}")

    return httpx.MockTransport(handler)


def _adapter(retrieval_config, transport):
    return ZenodoAdapter(
        SourceConfig(name="zenodo", enabled=True, base_url="https://zenodo.org/api", rate_limit_per_second=1000),
        retrieval_config,
        transport=transport,
    )


def test_fetch_record_follows_concept_doi_redirect_to_real_version(retrieval_config):
    """Real gap, confirmed live: a paper's own cited DOI
    (10.5281/zenodo.10381280) is a "concept" DOI representing all versions
    -- Zenodo 302s it to the actual current version (10381281)."""
    adapter = _adapter(retrieval_config, _transport(redirect_from="10381280"))
    record = adapter.fetch_record("10.5281/zenodo.10381280")
    assert record.external_identifier == "10.5281/zenodo.10381281"
    adapter.close()


def test_fetch_record_wires_sequence_data_status_from_file_listing(retrieval_config):
    adapter = _adapter(retrieval_config, _transport())
    record = adapter.fetch_record("10.5281/zenodo.10381281")
    assert record.raw["sequence_data_status"] == SequenceDataStatus.LIKELY.value
    assert record.raw["listed_file_names"] == ["Elas02.rar", "MiFish.rar"]
    adapter.close()


def test_extract_structured_facts_includes_sequence_data_status_and_placeholder_rows(retrieval_config):
    adapter = _adapter(retrieval_config, _transport())
    record = adapter.fetch_record("10.5281/zenodo.10381281")
    facts = adapter.extract_structured_facts(record)
    by_type = {f.fact_type_candidate: f for f in facts}
    assert by_type["sequence_data_status"].raw_value == "likely"
    assert by_type["title"].raw_value.startswith("Harnessing eDNA")
    # LIKELY still synthesizes the one placeholder sample+run pair.
    assert "samp_category" in by_type
    assert by_type["samp_category"].entity_external_id == "internal:zenodo:10.5281/zenodo.10381281:sample"
    adapter.close()


def test_fetch_record_raises_not_found_for_empty_payload(retrieval_config):
    adapter = _adapter(retrieval_config, httpx.MockTransport(lambda request: httpx.Response(200, json={})))
    with pytest.raises(SourceRecordNotFoundError):
        adapter.fetch_record("10.5281/zenodo.99999999")
    adapter.close()


def test_find_related_emits_the_real_versioned_doi(retrieval_config):
    adapter = _adapter(retrieval_config, _transport())
    record = adapter.fetch_record("10.5281/zenodo.10381281")
    related = adapter.find_related(record)
    assert len(related) == 1
    assert related[0].value == "10.5281/zenodo.10381281"
    adapter.close()
