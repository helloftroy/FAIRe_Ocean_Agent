"""HTTP-layer behavior (rate limiting, retry-on-5xx, 404 handling, on-disk
caching) tested against httpx.MockTransport -- no real network calls, fully
deterministic."""
import httpx
import pytest

from fair_ocean_agent.sources.base import SourceConfig
from fair_ocean_agent.sources.base import RateLimitedClient, SourceRecordNotFoundError
from fair_ocean_agent.sources.bcodmo import BcoDmoAdapter
from fair_ocean_agent.sources.datacite import DataCiteAdapter
from fair_ocean_agent.sources.pangaea import PangaeaAdapter


def _client(retrieval_config, handler, rate_limit_per_second=1000, tmp_path=None) -> RateLimitedClient:
    if tmp_path is not None:
        retrieval_config = retrieval_config.model_copy(update={"cache_enabled": True, "cache_dir": str(tmp_path)})
    return RateLimitedClient(
        "test-source", retrieval_config, rate_limit_per_second, transport=httpx.MockTransport(handler)
    )


def test_get_json_returns_payload(retrieval_config):
    def handler(request):
        return httpx.Response(200, json={"hello": "world"})

    client = _client(retrieval_config, handler)
    payload, from_cache = client.get_json("https://example.org/thing")
    assert payload == {"hello": "world"}
    assert from_cache is False
    client.close()


def test_404_raises_source_record_not_found(retrieval_config):
    def handler(request):
        return httpx.Response(404)

    client = _client(retrieval_config, handler)
    with pytest.raises(SourceRecordNotFoundError):
        client.get_json("https://example.org/missing")
    client.close()


def test_5xx_retries_then_succeeds(retrieval_config, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda seconds: None)  # skip real backoff delay in tests
    calls = {"count": 0}

    def handler(request):
        calls["count"] += 1
        if calls["count"] < 3:
            return httpx.Response(500)
        return httpx.Response(200, json={"ok": True})

    client = _client(retrieval_config, handler)
    payload, _ = client.get_json("https://example.org/flaky")
    assert payload == {"ok": True}
    assert calls["count"] == 3
    client.close()


def test_5xx_exhausts_retries_and_raises(retrieval_config, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    def handler(request):
        return httpx.Response(503)

    client = _client(retrieval_config, handler)
    with pytest.raises(httpx.HTTPStatusError):
        client.get_json("https://example.org/always-down")
    client.close()


def test_4xx_other_than_404_does_not_retry(retrieval_config, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    calls = {"count": 0}

    def handler(request):
        calls["count"] += 1
        return httpx.Response(400)

    client = _client(retrieval_config, handler)
    with pytest.raises(httpx.HTTPStatusError):
        client.get_json("https://example.org/bad-request")
    assert calls["count"] == 1  # no retry attempted for a permanent client error
    client.close()


def test_response_cache_avoids_second_network_call(retrieval_config, tmp_path):
    calls = {"count": 0}

    def handler(request):
        calls["count"] += 1
        return httpx.Response(200, json={"n": calls["count"]})

    client = _client(retrieval_config, handler, tmp_path=tmp_path)
    first, first_from_cache = client.get_json("https://example.org/cacheable")
    second, second_from_cache = client.get_json("https://example.org/cacheable")

    assert calls["count"] == 1
    assert first == second == {"n": 1}
    assert first_from_cache is False
    assert second_from_cache is True
    client.close()


def test_clear_cache_forces_a_fresh_network_call(retrieval_config, tmp_path):
    calls = {"count": 0}

    def handler(request):
        calls["count"] += 1
        return httpx.Response(200, json={"n": calls["count"]})

    client = _client(retrieval_config, handler, tmp_path=tmp_path)
    first, _ = client.get_json("https://example.org/cacheable")
    removed = client.clear_cache()
    second, second_from_cache = client.get_json("https://example.org/cacheable")

    assert removed == 1
    assert calls["count"] == 2
    assert first == {"n": 1}
    assert second == {"n": 2}
    assert second_from_cache is False
    client.close()


def test_clear_cache_is_a_no_op_when_cache_disabled(retrieval_config):
    def handler(request):
        return httpx.Response(200, json={})

    client = RateLimitedClient("test-source", retrieval_config, 1000, transport=httpx.MockTransport(handler))
    assert client.clear_cache() == 0
    client.close()


def test_rate_limiting_enforces_minimum_interval(retrieval_config, monkeypatch):
    sleep_calls = []
    monkeypatch.setattr("time.sleep", lambda seconds: sleep_calls.append(seconds))
    # monotonic() only advances via our own bookkeeping in this test, so force
    # elapsed time to look like ~0s between requests -- the throttle should
    # then always ask for a sleep close to the configured interval.
    monkeypatch.setattr("time.monotonic", lambda: 1000.0)

    def handler(request):
        return httpx.Response(200, json={})

    client = _client(retrieval_config, handler, rate_limit_per_second=2)  # min interval 0.5s
    client.get_json("https://example.org/a")
    client.get_json("https://example.org/b")

    assert any(s == pytest.approx(0.5, abs=1e-6) for s in sleep_calls)
    client.close()


def test_datacite_fetch_record_uses_doi_endpoint(retrieval_config):
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        return httpx.Response(200, json={"data": {"id": "10.1594/pangaea.923577", "attributes": {"doi": "10.1594/pangaea.923577"}}})

    adapter = DataCiteAdapter(
        SourceConfig(name="datacite", enabled=True, base_url="https://api.datacite.org"),
        retrieval_config,
        transport=httpx.MockTransport(handler),
    )
    record = adapter.fetch_record("10.1594/pangaea.923577")

    assert seen["path"] == "/dois/10.1594/pangaea.923577"
    assert record.external_identifier == "10.1594/pangaea.923577"
    adapter.close()


def test_pangaea_fetch_record_requests_metadata_jsonld(retrieval_config):
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["query"] = str(request.url.query)
        return httpx.Response(200, json={"@id": "https://doi.org/10.1594/PANGAEA.923577", "name": "Dataset"})

    adapter = PangaeaAdapter(
        SourceConfig(name="pangaea", enabled=True, base_url="https://doi.pangaea.de"),
        retrieval_config,
        transport=httpx.MockTransport(handler),
    )
    record = adapter.fetch_record("923577")

    assert seen["path"] == "/10.1594/PANGAEA.923577"
    assert "format=metadata_jsonld" in seen["query"]
    assert record.external_identifier == "10.1594/PANGAEA.923577"
    adapter.close()


def test_bcodmo_fetch_record_extracts_dataset_id_from_doi(retrieval_config):
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        return httpx.Response(200, json={"id": "765432", "title": "Dataset"})

    adapter = BcoDmoAdapter(
        SourceConfig(name="bcodmo", enabled=True, base_url="https://www.bco-dmo.org/api"),
        retrieval_config,
        transport=httpx.MockTransport(handler),
    )
    record = adapter.fetch_record("10.26008/1912/bco-dmo.765432.1")

    assert seen["path"] == "/api/dataset/765432"
    assert record.external_identifier == "765432"
    adapter.close()
