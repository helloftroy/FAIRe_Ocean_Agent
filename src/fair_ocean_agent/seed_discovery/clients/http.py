from __future__ import annotations

import logging
from datetime import datetime
from email.utils import parsedate_to_datetime
import random
import time
from urllib.parse import urlencode

import httpx

from fair_ocean_agent.seed_discovery.config import SeedDiscoveryConfig
from fair_ocean_agent.seed_discovery.db import SeedDiscoveryDB

logger = logging.getLogger(__name__)

_RETRY_STATUS = {429, 500, 502, 503, 504}


class CachedHttpClient:
    def __init__(self, config: SeedDiscoveryConfig, db: SeedDiscoveryDB):
        self.config = config
        self.db = db
        self._last_request_by_source: dict[str, float] = {}
        self._client = httpx.Client(
            timeout=config.request_timeout_seconds,
            headers={"User-Agent": config.user_agent, "Accept": "application/json"},
            follow_redirects=True,
        )

    def close(self) -> None:
        self._client.close()

    def _request_key(self, source: str, url: str, params: dict | None) -> str:
        if not params:
            return f"{source}:GET:{url}"
        return f"{source}:GET:{url}?{urlencode(sorted(params.items()), doseq=True)}"

    def get_json(self, source: str, url: str, *, params: dict | None = None, use_cache: bool = True) -> dict | list:
        request_key = self._request_key(source, url, params)
        if use_cache:
            cached = self.db.get_cache(request_key)
            if cached is not None:
                return cached

        payload, status_code = self._get_json_uncached(source, url, params=params)
        self.db.set_cache(request_key, source, payload, status_code)
        self.db.update_crawl_state(source, request=request_key, status="running")
        return payload

    def _get_json_uncached(self, source: str, url: str, *, params: dict | None = None) -> tuple[dict | list, int]:
        last_exc: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            wait = self._request_interval_wait_seconds(source)
            if wait > 0:
                if source == "openalex":
                    logger.info("waiting %.2fs before next OpenAlex request", wait)
                time.sleep(wait)
            self._last_request_by_source[source] = time.monotonic()
            self.db.update_crawl_state(source, status="running")
            try:
                response = self._client.get(url, params=params)
                if response.status_code not in _RETRY_STATUS:
                    response.raise_for_status()
                    return response.json(), response.status_code
                if source == "openalex" and response.status_code == 429:
                    raise httpx.HTTPStatusError(
                        "OpenAlex returned 429 Too Many Requests; skipping this study for later reprocessing",
                        request=response.request,
                        response=response,
                    )
                retry_after = response.headers.get("Retry-After")
                sleep_seconds = _retry_after_seconds(retry_after)
                last_exc = httpx.HTTPStatusError(
                    f"retryable status {response.status_code}", request=response.request, response=response
                )
            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
                if isinstance(exc, httpx.HTTPStatusError):
                    status_code = exc.response.status_code
                    if source == "openalex" and status_code == 429:
                        raise
                    if status_code not in _RETRY_STATUS:
                        raise
                last_exc = exc
                sleep_seconds = None
            if attempt >= self.config.max_retries:
                break
            if sleep_seconds is None:
                sleep_seconds = self.config.retry_base_seconds * (2 ** attempt) + random.uniform(0, 0.5)
            logger.warning("retrying %s after %.2fs (%s)", url, sleep_seconds, last_exc)
            time.sleep(sleep_seconds)
        assert last_exc is not None
        raise last_exc

    def _request_interval_wait_seconds(self, source: str) -> float:
        interval = self.config.request_interval_for_source(source)
        elapsed = time.monotonic() - self._last_request_by_source.get(source, 0.0)
        persistent_timestamp = self.db.crawl_updated_at(source)
        if persistent_timestamp:
            try:
                persistent_elapsed = time.time() - datetime.fromisoformat(persistent_timestamp).timestamp()
            except ValueError:
                persistent_elapsed = None
            if persistent_elapsed is not None:
                elapsed = min(elapsed, persistent_elapsed)
        return max(0.0, interval - elapsed)


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        retry_at = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, retry_at.timestamp() - time.time())
