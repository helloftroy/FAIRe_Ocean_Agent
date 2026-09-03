"""Unpaywall adapter -- a free, no-API-key REST API
(https://unpaywall.org/products/api) aggregating legal open-access
location data across essentially every publisher, not just PMC. Confirmed
live (10.1016/j.jhazmat.2024.133878, a real ScienceDirect paper blocked
by Cloudflare's bot-detection challenge when fetched directly): Unpaywall
correctly reports this paper as genuinely open access (CC-BY, published
version) even though it was never deposited in PMC and Europe PMC has no
route to it at all.

Used two ways in this pipeline:
1. workflow/handlers.py's `_auto_fetch_open_access_pdf` tries Unpaywall's
   own best_oa_location as an additional candidate PDF URL alongside
   OpenAlex's (a study with no PMCID and no usable OpenAlex OA link may
   still have one via Unpaywall, or vice versa) -- same honest,
   never-spoofed fetch attempt, same graceful "blocked or not really a
   PDF" skip if the publisher's own site refuses it.
2. scripts/check_fulltext_access.py's own diagnostic report: a study with
   no confirmed full text is meaningfully different depending on whether
   Unpaywall says it's genuinely closed (nothing further to do) or
   genuinely open (worth a human's one-click manual download, since no
   subscription is actually needed) -- this pipeline never attempts to
   defeat a publisher's own bot-detection/CAPTCHA to fetch it
   automatically; that's a deliberate access control, not a bug to route
   around (see _auto_fetch_open_access_pdf's own docstring for the
   identical stance already taken for OpenAlex).

Unpaywall's own required `email` query parameter identifies the caller to
Unpaywall (their own "polite pool" contact convention, the same one this
project's other API integrations already use for OpenAlex/NCBI) -- never
sent to the publisher itself, never used for anything beyond this one
query parameter.
"""
from __future__ import annotations

import os

from fair_ocean_agent.clock import utcnow
from fair_ocean_agent.database.enums import AccessStatus
from fair_ocean_agent.sources.base import (
    SearchPage,
    SearchQuery,
    SourceAdapter,
    SourceRecord,
    SourceRecordNotFoundError,
    hash_payload,
)

_DEFAULT_CONTACT_EMAIL = "REPLACE_WITH_CONTACT_EMAIL@example.org"


def _contact_email() -> str:
    return os.environ.get("FAIR_OCEAN_CONTACT_EMAIL", _DEFAULT_CONTACT_EMAIL)


def best_oa_pdf_url(record: SourceRecord) -> str | None:
    """The single most useful candidate PDF URL from this record, if any
    -- prefers a direct `url_for_pdf` (an actual PDF link) but falls back
    to the location's own bare `url` (sometimes a landing page that still
    happens to serve the PDF directly, confirmed live for several
    repository-hosted copies), same fallback OpenAlexAdapter's own
    best_oa_location.pdf_url convention doesn't need since OpenAlex only
    ever populates pdf_url when it found one."""
    best = record.raw.get("best_oa_location") or {}
    return best.get("url_for_pdf") or best.get("url")


class UnpaywallAdapter(SourceAdapter):
    name = "unpaywall"

    def fetch_record(self, identifier: str) -> SourceRecord:
        url = f"{self.config.base_url}/v2/{identifier}"
        payload, from_cache = self.http.get_json(url, params={"email": _contact_email()})
        if not payload.get("doi"):
            raise SourceRecordNotFoundError(f"No Unpaywall record for {identifier}")
        return SourceRecord(
            source_name=self.name,
            external_identifier=payload.get("doi") or identifier,
            url=payload.get("doi_url"),
            raw=payload,
            retrieved_at=utcnow(),
            content_hash=hash_payload(payload),
            from_cache=from_cache,
        )

    def search(self, query: SearchQuery) -> SearchPage:
        # Unpaywall has no free-text search endpoint of its own use here --
        # it's always resolved via a known DOI (fetch_record), same as
        # DataCiteAdapter's own find_datasets_citing-only usage pattern.
        return SearchPage(records=[])

    def parse_publication_fields(self, record: SourceRecord) -> dict:
        payload = record.raw
        return {
            "fulltext_available": bool(payload.get("is_oa")),
            "open_access_status": (AccessStatus.OPEN.value if payload.get("is_oa") else AccessStatus.RESTRICTED.value),
            "title": payload.get("title"),
        }
