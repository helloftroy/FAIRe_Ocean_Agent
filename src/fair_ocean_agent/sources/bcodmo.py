"""BCO-DMO metadata adapter.

BCO-DMO exposes dataset metadata through its own dataset API and dataset
files through ERDDAP. This adapter reads only the metadata endpoint
(`/dataset/{id}`) and records file/link metadata when present; it does not
download dataset data files.
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

BCODMO_DOI_ID_RE = re.compile(r"bco-?dmo[./](\d+)", re.IGNORECASE)
BCODMO_NUMERIC_ID_RE = re.compile(r"^\d+$")


def _stringify(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, default=str)


def _dataset_id(identifier: str) -> str:
    value = identifier.strip()
    match = BCODMO_DOI_ID_RE.search(value)
    if match:
        return match.group(1)
    return value if BCODMO_NUMERIC_ID_RE.match(value) else value


def _first_present(raw: dict, *fields: str):
    for field in fields:
        value = raw.get(field)
        if value not in (None, "", [], {}):
            return value
    return None


class BcoDmoAdapter(SourceAdapter):
    name = "bcodmo"

    def fetch_record(self, identifier: str) -> SourceRecord:
        dataset_id = _dataset_id(identifier)
        payload, from_cache = self.http.get_json(f"{self.config.base_url}/dataset/{dataset_id}")
        if not payload:
            raise SourceRecordNotFoundError(f"No BCO-DMO dataset record for {identifier}")
        return SourceRecord(
            source_name=self.name,
            external_identifier=dataset_id,
            url=_first_present(payload, "url", "infoUrl") or f"https://www.bco-dmo.org/dataset/{dataset_id}",
            raw=payload,
            retrieved_at=utcnow(),
            content_hash=hash_payload(payload),
            from_cache=from_cache,
        )

    def search(self, query: SearchQuery) -> SearchPage:
        # BCO-DMO's public, stable search surface is ERDDAP's JSON search.
        erddap_base = self.config.base_url.replace("www.bco-dmo.org/api", "erddap.bco-dmo.org/erddap")
        payload, _ = self.http.get_json(
            f"{erddap_base}/search/index.json",
            params={"page": 1, "itemsPerPage": query.limit, "searchFor": query.query},
        )
        rows = ((payload.get("table") or {}).get("rows") or [])
        columns = ((payload.get("table") or {}).get("columnNames") or [])
        records = []
        for row in rows:
            item = dict(zip(columns, row))
            dataset_id = _dataset_id(item.get("Dataset ID", "") or item.get("datasetID", ""))
            if not dataset_id:
                continue
            raw = {"erddap_search": item}
            records.append(
                SourceRecord(
                    source_name=self.name,
                    external_identifier=dataset_id,
                    url=item.get("tabledap") or item.get("griddap") or item.get("Info"),
                    raw=raw,
                    retrieved_at=utcnow(),
                    content_hash=hash_payload(raw),
                )
            )
        return SearchPage(records=records, total_count=len(records))

    def extract_structured_facts(self, record: SourceRecord) -> list[RawFactCandidate]:
        raw = record.raw
        facts: list[RawFactCandidate] = []

        def add(field: str, value, locator: str | None = None) -> None:
            if value in (None, "", [], {}):
                return
            facts.append(
                RawFactCandidate(
                    entity_level=EntityLevel.PROJECT,
                    fact_type_candidate=field,
                    raw_field_name=field,
                    raw_value=_stringify(value),
                    source_locator=locator or f"bcodmo.dataset.{field}",
                )
            )

        field_aliases = {
            "dataset_id": ("id", "dataset_id", "datasetId"),
            "title": ("title", "name"),
            "doi": ("doi", "DOI"),
            "version": ("version", "datasetVersion"),
            "license": ("license", "rights"),
            "abstract": ("abstract", "description", "summary"),
            "temporal_extent": ("temporal_extent", "temporalExtent"),
            "spatial_extent": ("spatial_extent", "spatialExtent", "bounds"),
            "people": ("people", "contributors", "creators"),
            "projects": ("projects", "project"),
            "parameters": ("parameters",),
            "files": ("files", "datasetFiles"),
        }
        for fact_type, aliases in field_aliases.items():
            add(fact_type, _first_present(raw, *aliases))
        return facts

    def find_related(self, record: SourceRecord) -> list[RelatedIdentifier]:
        related = [
            RelatedIdentifier(
                identifier_type=IdentifierType.BCODMO_DATASET_ID,
                value=record.external_identifier,
                relationship_type=RelationshipType.IS_DATASET_FOR,
                source=self.name,
            )
        ]
        doi = _first_present(record.raw, "doi", "DOI")
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
