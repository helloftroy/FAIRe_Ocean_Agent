"""GBIF dataset metadata adapter.

Uses GBIF's registry dataset endpoints for dataset-level metadata and a
small bounded occurrence preview for Darwin Core/DNA-derived-data extension
terms. It does not page through or download full occurrence datasets.
"""
from __future__ import annotations

import json
from typing import Any

from fair_ocean_agent.clock import utcnow
from fair_ocean_agent.database.enums import EntityLevel, IdentifierType, RelationshipType
from fair_ocean_agent.logging_setup import get_logger
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
from fair_ocean_agent.sources.dna_extension import extract_dna_derived_facts

logger = get_logger(__name__)

OCCURRENCE_PREVIEW_LIMIT = 50


def _stringify(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, default=str)


class GbifAdapter(SourceAdapter):
    name = "gbif"

    def _fetch_occurrence_preview(self, identifier: str) -> Any:
        try:
            payload, _ = self.http.get_json(
                f"{self.config.base_url}/occurrence/search",
                params={"datasetKey": identifier, "limit": OCCURRENCE_PREVIEW_LIMIT},
            )
            return payload
        except Exception as exc:  # pragma: no cover - live API best effort
            logger.warning("Could not fetch GBIF occurrence preview for %s: %s", identifier, exc)
            return {}

    def fetch_record(self, identifier: str) -> SourceRecord:
        payload, from_cache = self.http.get_json(f"{self.config.base_url}/dataset/{identifier}")
        if not payload:
            raise SourceRecordNotFoundError(f"No GBIF dataset record for {identifier}")
        raw = dict(payload)
        raw["_occurrence_preview"] = self._fetch_occurrence_preview(identifier)
        return SourceRecord(
            source_name=self.name,
            external_identifier=identifier,
            url=payload.get("homepage") or f"https://www.gbif.org/dataset/{identifier}",
            raw=raw,
            retrieved_at=utcnow(),
            content_hash=hash_payload(raw),
            from_cache=from_cache,
        )

    def search(self, query: SearchQuery) -> SearchPage:
        payload, _ = self.http.get_json(
            f"{self.config.base_url}/dataset/search",
            params={"q": query.query, "limit": query.limit, "offset": int(query.cursor or 0)},
        )
        results = payload.get("results") or []
        records = []
        for item in results:
            key = item.get("key")
            if not key:
                continue
            raw = dict(item)
            records.append(
                SourceRecord(
                    source_name=self.name,
                    external_identifier=key,
                    url=item.get("homepage") or f"https://www.gbif.org/dataset/{key}",
                    raw=raw,
                    retrieved_at=utcnow(),
                    content_hash=hash_payload(raw),
                )
            )
        offset = int(payload.get("offset") or 0)
        count = int(payload.get("count") or 0)
        end = offset + len(records)
        next_cursor = str(end) if end < count else None
        return SearchPage(records=records, next_cursor=next_cursor, total_count=count)

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
                    source_locator=f"gbif.dataset.{field}",
                )
            )

        for field in (
            "key", "doi", "title", "description", "type", "subtype",
            "publishingOrganizationKey", "hostingOrganizationKey",
            "license", "citation", "created", "modified", "pubDate",
            "contacts", "keywords", "geographicCoverage", "temporalCoverages",
            "taxonomicCoverages", "recordCount",
        ):
            add(field, raw.get(field))
        facts.extend(extract_dna_derived_facts(self.name, raw.get("_occurrence_preview")))
        return facts

    def find_related(self, record: SourceRecord) -> list[RelatedIdentifier]:
        related = [
            RelatedIdentifier(
                identifier_type=IdentifierType.GBIF_DATASET_KEY,
                value=record.external_identifier,
                relationship_type=RelationshipType.IS_DATASET_FOR,
                source=self.name,
            )
        ]
        doi = record.raw.get("doi")
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
