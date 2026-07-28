"""Crossref adapter: DOI -> title/authors/year/journal/license metadata.
https://api.crossref.org/works/{doi}
"""
from __future__ import annotations

import json

from fair_ocean_agent.clock import utcnow
from fair_ocean_agent.database.enums import EntityLevel
from fair_ocean_agent.sources.base import (
    RawFactCandidate,
    SearchPage,
    SearchQuery,
    SourceAdapter,
    SourceRecord,
    hash_payload,
)


def _stringify(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, default=str)


class CrossrefAdapter(SourceAdapter):
    name = "crossref"

    def fetch_record(self, identifier: str) -> SourceRecord:
        url = f"{self.config.base_url}/works/{identifier}"
        payload, from_cache = self.http.get_json(url)
        message = payload.get("message", {})
        return SourceRecord(
            source_name=self.name,
            external_identifier=identifier,
            url=message.get("URL"),
            raw=message,
            retrieved_at=utcnow(),
            content_hash=hash_payload(message),
            from_cache=from_cache,
        )

    def search(self, query: SearchQuery) -> SearchPage:
        url = f"{self.config.base_url}/works"
        payload, _ = self.http.get_json(url, params={"query": query.query, "rows": query.limit})
        items = payload.get("message", {}).get("items", [])
        records = [
            SourceRecord(
                source_name=self.name,
                external_identifier=item.get("DOI", ""),
                url=item.get("URL"),
                raw=item,
                retrieved_at=utcnow(),
                content_hash=hash_payload(item),
            )
            for item in items
        ]
        return SearchPage(records=records, total_count=payload.get("message", {}).get("total-results"))

    def extract_structured_facts(self, record: SourceRecord) -> list[RawFactCandidate]:
        m = record.raw
        facts: list[RawFactCandidate] = []

        def add(field: str, value) -> None:
            if value in (None, "", [], {}):
                return
            facts.append(
                RawFactCandidate(
                    entity_level=EntityLevel.STUDY,
                    fact_type_candidate=field,
                    raw_field_name=field,
                    raw_value=_stringify(value),
                    source_locator=f"crossref.message.{field}",
                )
            )

        # Crossref's "title" and "container-title" are lists-of-one-string
        # by convention (unlike "author"/"license", which are genuinely
        # multi-item) -- storing the raw list would JSON-encode it as
        # '["Some Title"]', which then fails cross-source title comparison
        # (validation/cross_source.py) against Europe PMC/OpenAlex, whose
        # "title" facts are plain strings. Store the unwrapped string, same
        # as parse_publication_fields already does for the "title" field it
        # returns (used for Study.title, see workflow/handlers.py).
        titles = m.get("title") or []
        add("title", titles[0] if titles else None)
        add("author", m.get("author"))
        container_titles = m.get("container-title") or []
        add("container-title", container_titles[0] if container_titles else None)
        add("publisher", m.get("publisher"))
        add("type", m.get("type"))
        add("license", m.get("license"))
        add("published", m.get("published") or m.get("published-print") or m.get("published-online"))
        add("is-referenced-by-count", m.get("is-referenced-by-count"))
        return facts

    def parse_publication_fields(self, record: SourceRecord) -> dict:
        m = record.raw
        titles = m.get("title") or []
        authors = [
            {"given": a.get("given"), "family": a.get("family")} for a in (m.get("author") or [])
        ]
        date_parts = (
            (m.get("published") or {}).get("date-parts")
            or (m.get("published-print") or {}).get("date-parts")
            or (m.get("published-online") or {}).get("date-parts")
        )
        year = date_parts[0][0] if date_parts and date_parts[0] else None
        journals = m.get("container-title") or []

        return {
            "doi": m.get("DOI"),
            "title": titles[0] if titles else None,
            "authors": authors or None,
            "publication_year": year,
            "journal": journals[0] if journals else None,
        }
