"""Integration tests for FigshareAdapter's fetch_record via
httpx.MockTransport -- fixture shaped from a real live Figshare article
response (article files are embedded directly, no separate endpoint)."""
import httpx
import pytest

from fair_ocean_agent.sources.base import SourceConfig, SourceRecordNotFoundError
from fair_ocean_agent.sources.figshare import FigshareAdapter
from fair_ocean_agent.sources.sequence_file_heuristics import SequenceDataStatus

ARTICLE_JSON = {
    "id": 21653471,
    "doi": "10.6084/m9.figshare.21653471",
    "title": "18S amplicon raw reads",
    "description": "Raw sequencing data from eDNA metabarcoding.",
    "url_public_html": "https://figshare.com/articles/dataset/x/21653471",
    "license": {"name": "CC BY 4.0"},
    "files": [
        {"id": 1, "name": "sample_R1.fastq.gz", "size": 500_000_000},
        {"id": 2, "name": "sample_R2.fastq.gz", "size": 500_000_000},
    ],
}


def _transport(article_json=ARTICLE_JSON):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/articles/21653471"):
            return httpx.Response(200, json=article_json)
        raise AssertionError(f"unexpected request: {request.url}")

    return httpx.MockTransport(handler)


def _adapter(retrieval_config, transport):
    return FigshareAdapter(
        SourceConfig(name="figshare", enabled=True, base_url="https://api.figshare.com/v2", rate_limit_per_second=1000),
        retrieval_config,
        transport=transport,
    )


def test_fetch_record_extracts_article_id_from_doi_suffix(retrieval_config):
    adapter = _adapter(retrieval_config, _transport())
    record = adapter.fetch_record("10.6084/m9.figshare.21653471")
    assert record.external_identifier == "10.6084/m9.figshare.21653471"
    adapter.close()


def test_fetch_record_extracts_article_id_from_versioned_doi(retrieval_config):
    """Figshare DOIs occasionally carry a trailing version segment."""
    adapter = _adapter(retrieval_config, _transport())
    record = adapter.fetch_record("10.6084/m9.figshare.21653471.v2")
    assert record.external_identifier == "10.6084/m9.figshare.21653471"
    adapter.close()


def test_fetch_record_classifies_fastq_files_directly(retrieval_config):
    """Files are embedded in the article response itself -- confirmed
    live -- no separate files endpoint to fetch, unlike Zenodo/Dryad."""
    adapter = _adapter(retrieval_config, _transport())
    record = adapter.fetch_record("10.6084/m9.figshare.21653471")
    assert record.raw["sequence_data_status"] == SequenceDataStatus.CONFIRMED.value
    adapter.close()


def test_extract_structured_facts_includes_placeholder_rows(retrieval_config):
    adapter = _adapter(retrieval_config, _transport())
    record = adapter.fetch_record("10.6084/m9.figshare.21653471")
    facts = adapter.extract_structured_facts(record)
    by_type = {f.fact_type_candidate: f for f in facts}
    assert by_type["title"].raw_value == "18S amplicon raw reads"
    assert "samp_category" in by_type
    adapter.close()


def test_fetch_record_raises_not_found_for_empty_payload(retrieval_config):
    adapter = _adapter(retrieval_config, httpx.MockTransport(lambda request: httpx.Response(200, json={})))
    with pytest.raises(SourceRecordNotFoundError):
        adapter.fetch_record("10.6084/m9.figshare.00000000")
    adapter.close()
