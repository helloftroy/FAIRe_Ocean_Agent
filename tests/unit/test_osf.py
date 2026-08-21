"""Integration tests for OsfAdapter's fetch_record via httpx.MockTransport
-- fixtures shaped from a real live OSF node response."""
import httpx
import pytest

from fair_ocean_agent.sources.base import SourceConfig, SourceRecordNotFoundError
from fair_ocean_agent.sources.osf import OsfAdapter
from fair_ocean_agent.sources.sequence_file_heuristics import SequenceDataStatus

NODE_JSON = {
    "data": {
        "id": "ezcuj",
        "attributes": {"title": "eDNA metabarcoding survey", "description": "Raw amplicon sequencing data."},
    }
}
FILES_JSON = {
    "data": [
        {"attributes": {"name": "reads.fastq.gz", "size": 300_000_000, "kind": "file"}},
        {"attributes": {"name": "figures", "kind": "folder"}},
    ]
}


def _transport(node_json=NODE_JSON, files_json=FILES_JSON):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/nodes/EZCUJ/"):
            return httpx.Response(200, json=node_json)
        if path.endswith("/nodes/EZCUJ/files/osfstorage/"):
            return httpx.Response(200, json=files_json)
        raise AssertionError(f"unexpected request: {request.url}")

    return httpx.MockTransport(handler)


def _adapter(retrieval_config, transport):
    return OsfAdapter(
        SourceConfig(name="osf", enabled=True, base_url="https://api.osf.io/v2", rate_limit_per_second=1000),
        retrieval_config,
        transport=transport,
    )


def test_fetch_record_extracts_node_id_from_doi(retrieval_config):
    adapter = _adapter(retrieval_config, _transport())
    record = adapter.fetch_record("10.17605/OSF.IO/EZCUJ")
    assert record.external_identifier == "10.17605/OSF.IO/EZCUJ"
    adapter.close()


def test_fetch_record_classifies_fastq_file_ignoring_non_file_entries(retrieval_config):
    """A folder entry ("kind": "folder") must not be treated as a listed
    file -- only real files count toward the sequence-data check."""
    adapter = _adapter(retrieval_config, _transport())
    record = adapter.fetch_record("10.17605/OSF.IO/EZCUJ")
    assert record.raw["sequence_data_status"] == SequenceDataStatus.CONFIRMED.value
    assert record.raw["listed_file_names"] == ["reads.fastq.gz"]
    adapter.close()


def test_extract_structured_facts_includes_placeholder_rows(retrieval_config):
    adapter = _adapter(retrieval_config, _transport())
    record = adapter.fetch_record("10.17605/OSF.IO/EZCUJ")
    facts = adapter.extract_structured_facts(record)
    by_type = {f.fact_type_candidate: f for f in facts}
    assert by_type["title"].raw_value == "eDNA metabarcoding survey"
    assert "samp_category" in by_type
    adapter.close()


def test_fetch_record_raises_not_found_when_node_missing(retrieval_config):
    adapter = _adapter(retrieval_config, httpx.MockTransport(lambda request: httpx.Response(200, json={"data": {}})))
    with pytest.raises(SourceRecordNotFoundError):
        adapter.fetch_record("10.17605/OSF.IO/NOTREAL")
    adapter.close()
