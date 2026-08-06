"""Shared source-adapter interface (section 6). Every concrete adapter
implements search()/fetch_record() and returns the shared Pydantic models
below -- never a source-specific dict -- so no source-specific parsing ever
needs to live in the orchestrator (workflow/handlers.py).
"""
from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict, Field
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from fair_ocean_agent.config import REPO_ROOT, RetrievalConfig
from fair_ocean_agent.database.enums import (
    EntityLevel,
    EntityRelationshipType,
    IdentifierType,
    RelationshipType,
    SupportType,
)
from fair_ocean_agent.logging_setup import get_logger

logger = get_logger(__name__)


class SourceRecordNotFoundError(Exception):
    """Raised when an identifier is well-formed but the source has no record
    for it (e.g. a 404, or an empty search result for a search-only API like
    Europe PMC). Distinct from transport/5xx errors, which retry instead."""


class SourceConfig(BaseModel):
    name: str
    enabled: bool = False
    base_url: str
    rate_limit_per_second: float = 3.0
    priority: int = 100


class SearchQuery(BaseModel):
    query: str
    filters: dict = {}
    cursor: str | None = None
    limit: int = 20


class SourceRecord(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_name: str
    external_identifier: str
    url: str | None = None
    raw: dict
    retrieved_at: datetime
    content_hash: str
    from_cache: bool = False


class SearchPage(BaseModel):
    records: list[SourceRecord] = []
    next_cursor: str | None = None
    total_count: int | None = None


class RelatedIdentifier(BaseModel):
    identifier_type: IdentifierType
    value: str
    relationship_type: RelationshipType
    source: str
    # Evidence-tier gating identity/resolution.py's resolve_or_create_study()
    # uses to decide whether a match against a pre-existing Study is safe to
    # auto-link or needs a consistency check first. Reuses SupportType's
    # values as a shared confidence vocabulary (not a second parallel one):
    # STRUCTURED_SOURCE (default) = tier 1, this adapter's own structured API
    # relation field -- safe to auto-link. DETERMINISTICALLY_DERIVED = tier 2,
    # a regex-matched accession, only trustworthy once confirmed against that
    # identifier type's own source API (discovery/text_identifiers.py's
    # verify_deterministic_identifier). INFERRED = tier 3, an LLM prose claim
    # -- always consistency-checked, never sufficient alone even if
    # consistent. Deliberately NOT SupportType.EXPLICIT, which already means
    # something different in this codebase (an LLM-extracted RawFact with a
    # verbatim evidence quote, see extraction/text.py) -- reusing it here
    # would silently collide with that established meaning.
    confidence: SupportType = SupportType.STRUCTURED_SOURCE


class EntityLinkCandidate(BaseModel):
    entity_level: EntityLevel
    external_identifier: str
    relationship_type: EntityRelationshipType
    label: str | None = None


class RawFactCandidate(BaseModel):
    entity_level: EntityLevel
    fact_type_candidate: str
    raw_field_name: str
    raw_value: str
    source_locator: str
    support_type: SupportType = SupportType.STRUCTURED_SOURCE
    # Set when this fact describes a specific sub-entity (a BioSample, an
    # ENA run, ...) rather than the study as a whole -- e.g. a BioProject's
    # accession fetches a *batch* of samples in one Source row, but each
    # attribute fact still needs to land on its own Entity row. The handler
    # get-or-creates an Entity keyed on (study_id, entity_level,
    # external_identifier) when this is set; entity_id stays None otherwise
    # (study-wide fact, same as every Milestone 2 adapter).
    entity_external_id: str | None = None
    entity_label: str | None = None
    entity_links: list[EntityLinkCandidate] = Field(default_factory=list)
    # Verbatim quote from the source text, required for LLM-derived facts.
    # extraction/text.py now obtains this by asking the model for source
    # segment ID(s), then copying the segment text itself before persistence;
    # left None for structured API/XML facts, where source_locator (a
    # JSON/XML path) already pins down the evidence.
    evidence_quote: str | None = None
    # Overrides the default review_status a structured-adapter fact is
    # persisted with (ACCEPTED, see workflow/handlers.py's
    # _persist_source_and_facts) -- None keeps that default unchanged.
    # Set to ReviewStatus.NEEDS_REVIEW.value by sources/ncbi.py when a
    # BioProject/BioSample UID resolution was ambiguous (esearch returned
    # more than one UID and none of their own accessions matched what was
    # searched for) or the biosample->bioproject reverse-elink signal
    # disagreed with the UID esearch picked -- flagging the fact rather than
    # silently trusting a guess, see that module's _esearch_verified_uid.
    review_status: str | None = None
    # Optional, non-authoritative metadata about this fact -- currently used
    # by extraction/text.py to carry a candidate_standard_fields hint (e.g.
    # {"candidate_standard_fields": {"faire": "annealingTemp"}}) suggesting
    # which standard field this fact might map to. A hint, never part of
    # this fact's own identity: fact_type_candidate must stay meaningful and
    # standard-agnostic with this field absent entirely. Persisted as-is
    # into RawFact.confidence_metadata (workflow/handlers.py); standardizing
    # a fact onto FAIRe or any other vocabulary remains mapping/rules.py's
    # job, never decided here.
    confidence_metadata: dict | None = None


def hash_payload(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500 or exc.response.status_code == 429
    return isinstance(exc, (httpx.TransportError, httpx.TimeoutException))


class RateLimitedClient:
    """Synchronous httpx wrapper: minimum-interval rate limiting, retry with
    exponential backoff on 5xx/429/timeouts (never on 4xx, which are
    permanent), and a permanent on-disk response cache keyed by request URL
    + params -- repeated dev/test runs against the same identifiers never
    re-hit the live API."""

    def __init__(
        self,
        source_name: str,
        retrieval_config: RetrievalConfig,
        rate_limit_per_second: float,
        transport: httpx.BaseTransport | None = None,
    ):
        self._client = httpx.Client(
            timeout=retrieval_config.request_timeout_seconds,
            headers={"User-Agent": retrieval_config.user_agent},
            transport=transport,
        )
        self._min_interval = 1.0 / rate_limit_per_second if rate_limit_per_second > 0 else 0.0
        self._last_request_at = 0.0
        self._cache_enabled = retrieval_config.cache_enabled
        self._cache_dir = REPO_ROOT / retrieval_config.cache_dir / source_name
        if self._cache_enabled:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, url: str, params: dict | None, suffix: str) -> Path:
        basis = json.dumps({"url": url, "params": params or {}}, sort_keys=True)
        digest = hashlib.sha256(basis.encode()).hexdigest()
        return self._cache_dir / f"{digest}.{suffix}"

    def _throttle(self) -> None:
        if self._min_interval <= 0:
            return
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    def _get_bytes(self, url: str, params: dict | None) -> bytes:
        """Uncached, unretried-on-404 raw fetch. Raises
        SourceRecordNotFoundError on 404 rather than retrying (a missing
        record won't appear on retry)."""
        self._throttle()
        response = self._client.get(url, params=params)
        self._last_request_at = time.monotonic()
        if response.status_code == 404:
            raise SourceRecordNotFoundError(f"404 Not Found: {url}")
        response.raise_for_status()
        return response.content

    def get_json(self, url: str, params: dict | None = None) -> tuple[dict, bool]:
        """Returns (payload, from_cache)."""
        cache_path = self._cache_path(url, params, "json") if self._cache_enabled else None
        if cache_path is not None and cache_path.exists():
            return json.loads(cache_path.read_text()), True

        content = self._get_bytes(url, params)
        payload = json.loads(content)
        if cache_path is not None:
            cache_path.write_text(json.dumps(payload))
        return payload, False

    def get_text(self, url: str, params: dict | None = None) -> tuple[str, bool]:
        """Same contract as get_json but for XML/plain-text APIs (e.g. NCBI
        E-utilities' efetch/esearch, which return XML, not JSON)."""
        cache_path = self._cache_path(url, params, "xml") if self._cache_enabled else None
        if cache_path is not None and cache_path.exists():
            return cache_path.read_text(), True

        content = self._get_bytes(url, params)
        text = content.decode("utf-8")
        if cache_path is not None:
            cache_path.write_text(text)
        return text, False

    def get_binary(self, url: str, params: dict | None = None) -> tuple[bytes, bool]:
        """Same contract as get_json/get_text but for genuinely binary
        payloads (e.g. Europe PMC's {pmcid}/supplementaryFiles zip bundle) --
        never decoded as text, cached on disk as raw bytes."""
        cache_path = self._cache_path(url, params, "bin") if self._cache_enabled else None
        if cache_path is not None and cache_path.exists():
            return cache_path.read_bytes(), True

        content = self._get_bytes(url, params)
        if cache_path is not None:
            cache_path.write_bytes(content)
        return content, False

    def clear_cache(self) -> int:
        """Deletes every cached response file for this adapter instance.
        The on-disk cache (see get_json/get_text above) never expires on
        its own -- correct for the normal discovery/retry path, where a
        cached response being "stale" is never the point (a DOI's Crossref
        record isn't expected to change between retries of the same task).
        Milestone 7's refresh path calls this once per adapter before a
        refresh pass specifically because it *does* want a genuinely fresh
        network response, and the alternative (getting each adapter's
        fetch_record to accept a per-call cache-bypass flag) would touch
        six adapters' method signatures and internal helper functions for
        the same effect. Returns the number of files removed."""
        if not self._cache_enabled or not self._cache_dir.exists():
            return 0
        removed = 0
        for path in self._cache_dir.iterdir():
            if path.is_file():
                path.unlink()
                removed += 1
        return removed

    def close(self) -> None:
        self._client.close()


class SourceAdapter(ABC):
    name: str

    def __init__(
        self,
        config: SourceConfig,
        retrieval_config: RetrievalConfig,
        transport: httpx.BaseTransport | None = None,
        http: RateLimitedClient | None = None,
    ):
        """`http`, when given, is used as-is instead of building a new
        RateLimitedClient -- for adapters that share one physical API/rate
        limit across multiple "sources" (e.g. NCBI BioProject and BioSample
        both hit eutils.ncbi.nlm.nih.gov and share its single per-IP rate
        limit; two independent throttles would each allow bursts that
        together exceed it). See workflow/handlers.py's NCBI wiring."""
        self.config = config
        self.http = http or RateLimitedClient(
            config.name, retrieval_config, config.rate_limit_per_second, transport=transport
        )

    @abstractmethod
    def search(self, query: SearchQuery) -> SearchPage:
        ...

    @abstractmethod
    def fetch_record(self, identifier: str) -> SourceRecord:
        ...

    def find_related(self, record: SourceRecord) -> list[RelatedIdentifier]:
        return []

    def extract_structured_facts(self, record: SourceRecord) -> list[RawFactCandidate]:
        return []

    def parse_publication_fields(self, record: SourceRecord) -> dict:
        """Adapter-specific mapping from its raw record to the
        bibliographic fields (authors/journal/publication_year/
        fulltext_available/open_access_status/title) that
        workflow/handlers.py applies onto this adapter's own Source row
        (source_type="publication_api") and, for title, onto Study.title.
        Not part of the section-6 interface (which is source-shaped) but
        kept on the adapter rather than the handler so parsing logic stays
        out of the orchestrator, per section 6."""
        return {}

    def get_watermark(self):
        return None

    def close(self) -> None:
        self.http.close()
