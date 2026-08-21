"""Integration test for DataCiteAdapter.find_datasets_citing (Pass 3 of
discovery/text_identifiers.py's staged repository search) via
httpx.MockTransport. Confirmed live against the real API before writing
this: the query field needs the DOI double-quoted (unquoted, a bare
"10.xxxx/yyyy" is tokenized by slashes and matches nothing)."""
import httpx

from fair_ocean_agent.sources.base import SourceConfig
from fair_ocean_agent.sources.datacite import DataCiteAdapter

ARTICLE_DOI = "10.1038/s41598-024-60762-8"


def _adapter(retrieval_config, transport):
    return DataCiteAdapter(
        SourceConfig(name="datacite", enabled=True, base_url="https://api.datacite.org", rate_limit_per_second=1000),
        retrieval_config,
        transport=transport,
    )


def test_find_datasets_citing_requires_quoted_doi_in_query(retrieval_config):
    def handler(request: httpx.Request) -> httpx.Response:
        query = request.url.params.get("query")
        assert query == f'relatedIdentifiers.relatedIdentifier:"{ARTICLE_DOI}"'
        return httpx.Response(
            200,
            json={"data": [{"id": "10.5281/zenodo.10381281", "attributes": {"doi": "10.5281/zenodo.10381281"}}]},
        )

    adapter = _adapter(retrieval_config, httpx.MockTransport(handler))
    dois = adapter.find_datasets_citing(ARTICLE_DOI)
    assert dois == ["10.5281/zenodo.10381281"]
    adapter.close()


def test_find_datasets_citing_excludes_the_article_doi_itself(retrieval_config):
    """A record shouldn't cite itself back into the result list -- guards
    against a degenerate query match."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": ARTICLE_DOI, "attributes": {"doi": ARTICLE_DOI}}]})

    adapter = _adapter(retrieval_config, httpx.MockTransport(handler))
    assert adapter.find_datasets_citing(ARTICLE_DOI) == []
    adapter.close()


def test_find_datasets_citing_returns_empty_for_no_matches(retrieval_config):
    """Real case: this exact article's own real Zenodo dataset record
    doesn't declare a relatedIdentifiers link back to it (confirmed live)
    -- Pass 3 correctly finds nothing here, it just isn't this paper's
    discovery channel."""
    adapter = _adapter(retrieval_config, httpx.MockTransport(lambda request: httpx.Response(200, json={"data": []})))
    assert adapter.find_datasets_citing(ARTICLE_DOI) == []
    adapter.close()
