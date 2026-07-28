"""OBIS dataset metadata adapter.

Fetches dataset-level metadata from OBIS and, when available, stores
summary counts/coverage only. It does not download occurrence records.
"""
from __future__ import annotations

import json

from fair_ocean_agent.clock import utcnow
from fair_ocean_agent.database.enums import EntityLevel, IdentifierType, RelationshipType
from fair_ocean_agent.sources.base import (
    RawFactCandidate,
    RelatedIdentifier,
    SearchPage,
    SearchQuery,
    SourceAdapter,
    SourceRecord,
    SourceRecordNotFoundError,
    hash_payload,
)


def _stringify(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, default=str)


class ObisAdapter(SourceAdapter):
    name = "obis"

    def fetch_record(self, identifier: str) -> SourceRecord:
        payload, from_cache = self.http.get_json(f"{self.config.base_url}/dataset/{identifier}")
        if not payload:
            raise SourceRecordNotFoundError(f"No OBIS dataset record for {identifier}")
        return SourceRecord(
            source_name=self.name,
            external_identifier=identifier,
            url=payload.get("url") or f"https://obis.org/dataset/{identifier}",
            raw=payload,
            retrieved_at=utcnow(),
            content_hash=hash_payload(payload),
            from_cache=from_cache,
        )

    def search(self, query: SearchQuery) -> SearchPage:
        payload, _ = self.http.get_json(f"{self.config.base_url}/dataset", params={"search": query.query})
        items = payload.get("results") if isinstance(payload, dict) else payload
        items = items or []
        records = []
        for item in items[: query.limit]:
            dataset_id = item.get("id") or item.get("uuid") or item.get("dataset_id")
            if not dataset_id:
                continue
            raw = dict(item)
            records.append(
                SourceRecord(
                    source_name=self.name,
                    external_identifier=dataset_id,
                    url=item.get("url") or f"https://obis.org/dataset/{dataset_id}",
                    raw=raw,
                    retrieved_at=utcnow(),
                    content_hash=hash_payload(raw),
                )
            )
        total = payload.get("total") if isinstance(payload, dict) else len(records)
        return SearchPage(records=records, total_count=total)

    def extract_structured_facts(self, record: SourceRecord) -> list[RawFactCandidate]:
        raw = record.raw
        facts: list[RawFactCandidate] = []

        def add(field: str, value) -> None:
            if value in (None, "", [], {}):
                return
            facts.append(
                RawFactCandidate(
                    entity_level=EntityLevel.PROJECT,
                    fact_type_candidate=field,
                    raw_field_name=field,
                    raw_value=_stringify(value),
                    source_locator=f"obis.dataset.{field}",
                )
            )

        for field in (
            "id", "title", "abstract", "citation", "license", "rights",
            "created", "modified", "published", "contacts", "organization",
            "records", "recordCount", "taxa", "geometry", "geographicCoverage",
            "temporalCoverage", "eml",
        ):
            add(field, raw.get(field))
        return facts

    def find_related(self, record: SourceRecord) -> list[RelatedIdentifier]:
        related = [
            RelatedIdentifier(
                identifier_type=IdentifierType.OBIS_DATASET_UUID,
                value=record.external_identifier,
                relationship_type=RelationshipType.IS_DATASET_FOR,
                source=self.name,
            )
        ]
        doi = record.raw.get("doi") or record.raw.get("DOI")
        if doi:
            related.append(
                RelatedIdentifier(
                    identifier_type=IdentifierType.DATASET_DOI,
                    value=doi,
                    relationship_type=RelationshipType.IS_DATASET_FOR,
                    source=self.name,
                )
            )
        return related
