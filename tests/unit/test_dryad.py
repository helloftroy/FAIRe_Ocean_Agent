"""Integration tests for DryadAdapter's fetch_record via httpx.MockTransport
-- fixtures shaped from the real live response for
10.5061/dryad.xksn02vdx (10.1002/edn3.184's real Dryad dataset)."""
import httpx
import pytest

from fair_ocean_agent.sources.base import SourceConfig, SourceRecordNotFoundError
from fair_ocean_agent.sources.dryad import DryadAdapter
from fair_ocean_agent.sources.sequence_file_heuristics import SequenceDataStatus

DOI = "10.5061/dryad.xksn02vdx"
# httpx.Request.url.path is percent-DEcoded -- the request is still sent
# URL-encoded on the wire (confirmed live against the real Dryad API), this
# is just how the mock handler below has to compare it.
DATASET_PATH = f"/api/v2/datasets/doi:{DOI}"
DATASET_JSON = {
    "identifier": f"doi:{DOI}",
    "title": "The applicability of eDNA metabarcoding approaches for sessile benthic surveying",
    "abstract": "The application of environmental DNA technologies is a promising new approach.",
    "_links": {"stash:version": {"href": "/api/v2/versions/91146"}},
}
FILES_JSON = {
    "_embedded": {
        "stash:files": [
            {"path": "CoralITS2_acro_ASV_matrix.csv", "size": 459615, "mimeType": "text/csv"},
            {"path": "CoralITS2_acro_taxa_matrix.csv", "size": 47797, "mimeType": "text/csv"},
            {"path": "Unfiltered_demultiplexed_fastq_files.zip", "size": 4594947904, "mimeType": "application/zip"},
        ]
    }
}


def _transport(dataset_json=DATASET_JSON, files_json=FILES_JSON):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == DATASET_PATH:
            return httpx.Response(200, json=dataset_json)
        if request.url.path == "/api/v2/versions/91146/files":
            return httpx.Response(200, json=files_json)
        raise AssertionError(f"unexpected request: {request.url}")

    return httpx.MockTransport(handler)


def _adapter(retrieval_config, transport):
    return DryadAdapter(
        SourceConfig(name="dryad", enabled=True, base_url="https://datadryad.org/api/v2", rate_limit_per_second=1000),
        retrieval_config,
        transport=transport,
    )


def test_fetch_record_url_encodes_doi_with_doi_prefix(retrieval_config):
    """Confirmed live: Dryad's dataset endpoint requires the DOI prefixed
    with "doi:" and URL-encoded as a single path segment, not a bare DOI
    path -- a naive /datasets/{doi} 404s."""
    adapter = _adapter(retrieval_config, _transport())
    record = adapter.fetch_record(DOI)
    assert record.external_identifier == DOI
    adapter.close()


def test_fetch_record_classifies_the_real_fastq_zip_as_confirmed(retrieval_config):
    """Real motivating case: 10.1002/edn3.184's dataset has
    Unfiltered_demultiplexed_fastq_files.zip alongside processed ASV/taxa
    CSV matrices -- the zip's own name should win."""
    adapter = _adapter(retrieval_config, _transport())
    record = adapter.fetch_record(DOI)
    assert record.raw["sequence_data_status"] == SequenceDataStatus.CONFIRMED.value
    assert "Unfiltered_demultiplexed_fastq_files.zip" in record.raw["listed_file_names"]
    adapter.close()


def test_extract_structured_facts_includes_placeholder_sample_and_run_rows(retrieval_config):
    adapter = _adapter(retrieval_config, _transport())
    record = adapter.fetch_record(DOI)
    facts = adapter.extract_structured_facts(record)
    by_type = {f.fact_type_candidate: f for f in facts}
    assert by_type["sequence_data_status"].raw_value == "confirmed"
    sample_id = f"internal:dryad:{DOI}:sample"
    assert by_type["samp_category"].entity_external_id == sample_id
    assert by_type["sample_accession"].raw_value == sample_id  # materialSampleID redirect
    assert by_type["samp_name"].entity_external_id == f"internal:dryad:{DOI}:run"
    assert by_type["associatedSequences"].raw_value == f"https://doi.org/{DOI}"
    adapter.close()


def test_fetch_record_raises_not_found_for_empty_payload(retrieval_config):
    adapter = _adapter(retrieval_config, httpx.MockTransport(lambda request: httpx.Response(200, json={})))
    with pytest.raises(SourceRecordNotFoundError):
        adapter.fetch_record("10.5061/dryad.notreal")
    adapter.close()
