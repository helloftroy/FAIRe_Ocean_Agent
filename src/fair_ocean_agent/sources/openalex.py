"""OpenAlex adapter: DOI -> OpenAlex work ID + OA status + year/title.
https://api.openalex.org/works/https://doi.org/{doi}
"""
from __future__ import annotations

import json

from fair_ocean_agent.clock import utcnow
from fair_ocean_agent.database.enums import AccessStatus, EntityLevel, IdentifierType, RelationshipType
from fair_ocean_agent.sources.base import (
    RawFactCandidate,
    RelatedIdentifier,
    SearchPage,
    SearchQuery,
    SourceAdapter,
    SourceRecord,
    hash_payload,
)


def _stringify(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, default=str)


def _strip_openalex_id(value: str | None) -> str | None:
    if not value:
        return None
    return value.rsplit("/", 1)[-1]


class OpenAlexAdapter(SourceAdapter):
    name = "openalex"

    def fetch_record(self, identifier: str) -> SourceRecord:
        url = f"{self.config.base_url}/works/https://doi.org/{identifier}"
        payload, from_cache = self.http.get_json(url)
        return SourceRecord(
            source_name=self.name,
            external_identifier=identifier,
            url=payload.get("id"),
            raw=payload,
            retrieved_at=utcnow(),
            content_hash=hash_payload(payload),
            from_cache=from_cache,
        )

    def search(self, query: SearchQuery) -> SearchPage:
        url = f"{self.config.base_url}/works"
        payload, _ = self.http.get_json(url, params={"search": query.query, "per_page": query.limit})
        results = payload.get("results", [])
        records = [
            SourceRecord(
                source_name=self.name,
                external_identifier=(r.get("doi") or "").replace("https://doi.org/", ""),
                url=r.get("id"),
                raw=r,
                retrieved_at=utcnow(),
                content_hash=hash_payload(r),
            )
            for r in results
        ]
        return SearchPage(records=records, total_count=(payload.get("meta") or {}).get("count"))

    def extract_structured_facts(self, record: SourceRecord) -> list[RawFactCandidate]:
        p = record.raw
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
                    source_locator=f"openalex.work.{field}",
                )
            )

        add("title", p.get("title"))
        add("publication_year", p.get("publication_year"))
        add("open_access", p.get("open_access"))
        add("primary_location", p.get("primary_location"))
        add("authorships", p.get("authorships"))
        add("cited_by_count", p.get("cited_by_count"))
        return facts

    def parse_publication_fields(self, record: SourceRecord) -> dict:
        p = record.raw
        oa = p.get("open_access") or {}
        return {
            "openalex_id": _strip_openalex_id(p.get("id")),
            "title": p.get("title") or p.get("display_name"),
            "publication_year": p.get("publication_year"),
            "open_access_status": AccessStatus.OPEN.value if oa.get("is_oa") else AccessStatus.UNKNOWN.value,
        }

    def find_related(self, record: SourceRecord) -> list[RelatedIdentifier]:
        openalex_id = _strip_openalex_id(record.raw.get("id"))
        if not openalex_id:
            return []
        return [
            RelatedIdentifier(
                identifier_type=IdentifierType.OPENALEX_ID,
                value=openalex_id,
                relationship_type=RelationshipType.IS_PUBLICATION_OF,
                source=self.name,
            )
        ]
