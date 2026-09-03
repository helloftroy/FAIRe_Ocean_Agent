"""Tests for UnpaywallAdapter and its oa_pdf_urls/best_oa_pdf_url helpers.
Real gap this adapter exists to fix, confirmed live: a real ScienceDirect
paper (10.1016/j.jhazmat.2024.133878) is genuinely CC-BY open access but
was never deposited in PMC and had no usable OpenAlex best_oa_location --
Unpaywall correctly reports it as open when queried directly."""
from datetime import datetime, timezone

import httpx
import pytest

from fair_ocean_agent.sources.base import SourceConfig, SourceRecord, SourceRecordNotFoundError
from fair_ocean_agent.sources.unpaywall import UnpaywallAdapter, best_oa_pdf_url, oa_pdf_urls


def _adapter(retrieval_config, transport):
    return UnpaywallAdapter(
        SourceConfig(name="unpaywall", enabled=True, base_url="https://api.unpaywall.org", rate_limit_per_second=1000),
        retrieval_config,
        transport=transport,
    )


def _record(raw: dict) -> SourceRecord:
    return SourceRecord(
        source_name="unpaywall", external_identifier=raw.get("doi", "10.x/y"), url=None,
        raw=raw, retrieved_at=datetime.now(timezone.utc), content_hash="x",
    )


def test_fetch_record_sends_the_required_email_query_param(retrieval_config, monkeypatch):
    monkeypatch.setenv("FAIR_OCEAN_CONTACT_EMAIL", "test@example.org")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params.get("email") == "test@example.org"
        return httpx.Response(200, json={"doi": "10.1016/j.jhazmat.2024.133878", "is_oa": True})

    adapter = _adapter(retrieval_config, httpx.MockTransport(handler))
    record = adapter.fetch_record("10.1016/j.jhazmat.2024.133878")
    assert record.raw["is_oa"] is True
    adapter.close()


def test_fetch_record_raises_not_found_on_a_plain_404(retrieval_config):
    """A dataset DOI registered with DataCite rather than Crossref (real
    example: 10.15468/bvcp7p, a GBIF/MGnify record) isn't a recognized
    article DOI to Unpaywall at all -- confirmed live, Unpaywall returns a
    plain 404 for it."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    adapter = _adapter(retrieval_config, httpx.MockTransport(handler))
    with pytest.raises(SourceRecordNotFoundError):
        adapter.fetch_record("10.15468/bvcp7p")
    adapter.close()


def test_parse_publication_fields_reflects_open_access_status():
    adapter = UnpaywallAdapter.__new__(UnpaywallAdapter)  # parse_publication_fields needs no HTTP state
    open_fields = adapter.parse_publication_fields(_record({"is_oa": True, "title": "An open paper"}))
    closed_fields = adapter.parse_publication_fields(_record({"is_oa": False, "title": "A closed paper"}))
    assert open_fields == {"fulltext_available": True, "open_access_status": "open", "title": "An open paper"}
    assert closed_fields == {"fulltext_available": False, "open_access_status": "restricted", "title": "A closed paper"}


def test_oa_pdf_urls_prefers_repository_hosted_copies_over_the_publisher():
    """Real gap: the publisher's own OA copy is exactly the copy most
    likely to sit behind a bot-detection block (the ScienceDirect case
    this adapter exists for) -- a repository-hosted copy (institutional
    repository, a PMC-adjacent mirror, a preprint server) is the same
    legally open content on a plain, unprotected file host, so per an
    explicit user question it's tried first even when Unpaywall's own
    best_oa_location ranks the publisher's copy as "best"."""
    record = _record(
        {
            "is_oa": True,
            "best_oa_location": {"host_type": "publisher", "url": "https://publisher.example/paper"},
            "oa_locations": [
                {"host_type": "publisher", "url_for_pdf": None, "url": "https://publisher.example/paper"},
                {
                    "host_type": "repository",
                    "url_for_pdf": "https://university.edu/repo/paper.pdf",
                    "url": "https://university.edu/repo/paper",
                },
            ],
        }
    )
    urls = oa_pdf_urls(record)
    assert urls == ["https://university.edu/repo/paper.pdf", "https://publisher.example/paper"]
    assert best_oa_pdf_url(record) == "https://university.edu/repo/paper.pdf"


def test_oa_pdf_urls_falls_back_to_best_oa_location_when_oa_locations_is_absent():
    record = _record({"is_oa": True, "best_oa_location": {"url_for_pdf": "https://x/paper.pdf"}})
    assert oa_pdf_urls(record) == ["https://x/paper.pdf"]


def test_oa_pdf_urls_returns_empty_list_with_no_locations_at_all():
    assert oa_pdf_urls(_record({"is_oa": False})) == []
