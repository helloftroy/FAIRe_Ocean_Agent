"""Integration test for QiitaAdapter via httpx.MockTransport, mirroring
test_datacite.py's pattern. Covers the lightweight scope this adapter was
built for: confirm real per-sample structure (one real SAMPLE +
EXPERIMENT_RUN per actual sample row, not a single synthesized
placeholder like Zenodo/Dryad/Figshare/OSF), the "data is there and
downloadable" marker, and reusing a real BioSample accession as the
sample's own external_identifier when the row provides one."""
from __future__ import annotations

import io
from zipfile import ZIP_DEFLATED, ZipFile

import httpx
import pytest

from fair_ocean_agent.database.enums import EntityLevel
from fair_ocean_agent.sources.base import SourceConfig, SourceRecordNotFoundError
from fair_ocean_agent.sources.qiita import QiitaAdapter

STUDY_ID = "12345"


def _sample_zip(tsv_text: str) -> bytes:
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        archive.writestr("sample_information.tsv", tsv_text)
    return buffer.getvalue()


def _adapter(retrieval_config, transport):
    return QiitaAdapter(
        SourceConfig(name="qiita", enabled=True, base_url="https://qiita.ucsd.edu", rate_limit_per_second=1000),
        retrieval_config,
        transport=transport,
    )


def test_fetch_record_and_facts_for_two_real_samples(retrieval_config):
    tsv = "sample_name\ttemperature\tsalinity\n" "sample.001\t14.2\t35.0\n" "sample.002\t14.5\t34.8\n"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params.get("data") == "sample_information"
        assert request.url.params.get("study_id") == STUDY_ID
        return httpx.Response(200, content=_sample_zip(tsv))

    adapter = _adapter(retrieval_config, httpx.MockTransport(handler))
    record = adapter.fetch_record(STUDY_ID)
    facts = adapter.extract_structured_facts(record)

    sample_names = {f.raw_value for f in facts if f.fact_type_candidate == "samp_name"}
    assert sample_names == {"sample.001", "sample.002"}

    temp_facts = {f.entity_label: f.raw_value for f in facts if f.fact_type_candidate == "temp"}
    assert temp_facts == {"sample.001": "14.2", "sample.002": "14.5"}

    run_status_facts = [f for f in facts if f.fact_type_candidate == "sequence_data_status"]
    assert len(run_status_facts) == 2
    assert all(f.entity_level == EntityLevel.EXPERIMENT_RUN.value for f in run_status_facts)
    assert all(f.raw_value == "likely" for f in run_status_facts)

    url_facts = [f for f in facts if f.fact_type_candidate == "associatedSequences"]
    assert all(f.raw_value == record.url for f in url_facts)


def test_real_biosample_accession_used_as_external_identifier(retrieval_config):
    tsv = "sample_name\tbiosample\n" "sample.001\tSAMN00622972\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sample_zip(tsv))

    adapter = _adapter(retrieval_config, httpx.MockTransport(handler))
    record = adapter.fetch_record(STUDY_ID)
    facts = adapter.extract_structured_facts(record)

    sample_fact = next(f for f in facts if f.fact_type_candidate == "samp_name")
    assert sample_fact.entity_external_id == "SAMN00622972"


def test_no_valid_biosample_falls_back_to_internal_id(retrieval_config):
    tsv = "sample_name\n" "sample.001\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sample_zip(tsv))

    adapter = _adapter(retrieval_config, httpx.MockTransport(handler))
    record = adapter.fetch_record(STUDY_ID)
    facts = adapter.extract_structured_facts(record)

    sample_fact = next(f for f in facts if f.fact_type_candidate == "samp_name")
    assert sample_fact.entity_external_id == f"internal:qiita:{STUDY_ID}:sample.001"


def test_missing_study_raises_not_found(retrieval_config):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    adapter = _adapter(retrieval_config, httpx.MockTransport(handler))
    with pytest.raises(SourceRecordNotFoundError):
        adapter.fetch_record(STUDY_ID)


def test_non_zip_response_treated_as_not_found_not_a_crash(retrieval_config):
    """A private/nonexistent study can 200 with an HTML page instead of a
    real zip (e.g. a login redirect) -- must not crash the whole task."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>not a zip</html>")

    adapter = _adapter(retrieval_config, httpx.MockTransport(handler))
    with pytest.raises(SourceRecordNotFoundError):
        adapter.fetch_record(STUDY_ID)
