"""PANGAEA metadata adapter.

PANGAEA dataset landing pages expose machine-readable metadata via DOI
content negotiation and equivalent `format=metadata_jsonld` URL parameters.
This adapter fetches that metadata only; it does not retrieve tabular data
files.
"""
from __future__ import annotations

import json
import re

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

PANGAEA_ID_RE = re.compile(r"PANGAEA[./](\d+)", re.IGNORECASE)


def _stringify(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, default=str)


def _doi(identifier: str) -> str:
    value = identifier.strip()
    if value.lower().startswith("https://doi.org/"):
        return value.rsplit("doi.org/", 1)[-1]
    if value.lower().startswith("https://doi.pangaea.de/"):
        return value.rsplit("doi.pangaea.de/", 1)[-1]
    match = PANGAEA_ID_RE.search(value)
    if match:
        return f"10.1594/PANGAEA.{match.group(1)}"
    if value.isdigit():
        return f"10.1594/PANGAEA.{value}"
    return value


def _as_list(value) -> list:
    if value in (None, ""):
        return []
    return value if isinstance(value, list) else [value]


class PangaeaAdapter(SourceAdapter):
    name = "pangaea"

    def fetch_record(self, identifier: str) -> SourceRecord:
        doi = _doi(identifier)
        url = f"{self.config.base_url}/{doi}"
        payload, from_cache = self.http.get_json(url, params={"format": "metadata_jsonld"})
        if not payload:
            raise SourceRecordNotFoundError(f"No PANGAEA metadata record for {identifier}")
        return SourceRecord(
            source_name=self.name,
            external_identifier=doi,
            url=payload.get("@id") or f"https://doi.org/{doi}",
            raw=payload,
            retrieved_at=utcnow(),
            content_hash=hash_payload(payload),
            from_cache=from_cache,
        )

    def search(self, query: SearchQuery) -> SearchPage:
        # PANGAEA's broad discovery UI is separate from this DOI metadata
        # adapter. DataCite is the stable JSON search surface for PANGAEA
        # DOIs in this pipeline; keep search() explicit and bounded here.
        return SearchPage(records=[], total_count=0)

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
                    source_locator=f"pangaea.jsonld.{field}",
                )
            )

        for field in (
            "@id", "identifier", "name", "headline", "description", "creator",
            "publisher", "datePublished", "dateModified", "license",
            "spatialCoverage", "temporalCoverage", "variableMeasured", "keywords",
            "distribution", "citation",
        ):
            add(field, raw.get(field))
        return facts

    def find_related(self, record: SourceRecord) -> list[RelatedIdentifier]:
        related = [
            RelatedIdentifier(
                identifier_type=IdentifierType.PANGAEA_ID,
                value=record.external_identifier,
                relationship_type=RelationshipType.IS_DATASET_FOR,
                source=self.name,
            ),
            RelatedIdentifier(
                identifier_type=IdentifierType.DATASET_DOI,
                value=record.external_identifier,
                relationship_type=RelationshipType.IS_DATASET_FOR,
                source=self.name,
            ),
        ]

        for value in _as_list(record.raw.get("citation")):
            if isinstance(value, dict):
                identifier = value.get("identifier") or value.get("@id")
            else:
                identifier = str(value)
            if identifier and "10." in identifier:
                related.append(
                    RelatedIdentifier(
                        identifier_type=IdentifierType.DOI,
                        value=identifier,
                        relationship_type=RelationshipType.CITES,
                        source=self.name,
                    )
                )
        return related
