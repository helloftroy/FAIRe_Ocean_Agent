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

DCTERMS_TITLE = "http://purl.org/dc/terms/title"
DCTERMS_DESCRIPTION = "http://purl.org/dc/terms/description"
DCTERMS_IDENTIFIER = "http://purl.org/dc/terms/identifier"
DCTERMS_RIGHTS = "http://purl.org/dc/terms/rights"
DCTERMS_BIBLIOGRAPHIC_CITATION = "http://purl.org/dc/terms/bibliographicCitation"
BIBO_DOI = "http://purl.org/ontology/bibo/doi"
FOAF_HOMEPAGE = "http://xmlns.com/foaf/0.1/homepage"
ODO_DATASET_TITLE = "http://ocean-data.org/schema/datasetTitle"
ODO_ABSTRACT = "http://ocean-data.org/schema/abstract"
ODO_HAS_AWARD = "http://ocean-data.org/schema/hasAward"
ODO_FROM_INSTRUMENT = "http://ocean-data.org/schema/fromInstrument"
ODO_STORES_VALUES_FOR = "http://ocean-data.org/schema/storesValuesFor"
ODO_HAS_AGENT_WITH_ROLE = "http://ocean-data.org/schema/hasAgentWithRole"


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


def _jsonld_dataset_node(raw: dict) -> dict:
    graph = raw.get("@graph")
    if not isinstance(graph, list):
        return {}
    for item in graph:
        if not isinstance(item, dict):
            continue
        for node in item.values():
            if not isinstance(node, dict):
                continue
            node_types = node.get("@type") or []
            if isinstance(node_types, str):
                node_types = [node_types]
            if (
                "http://www.w3.org/ns/dcat#Dataset" in node_types
                or "http://ocean-data.org/schema/Dataset" in node_types
            ):
                return node
    return {}


def _jsonld_values(node: dict, field: str):
    values = node.get(field)
    if values in (None, "", [], {}):
        return None
    if not isinstance(values, list):
        values = [values]
    flattened = []
    for value in values:
        if isinstance(value, dict):
            flattened.append(value.get("@value") or value.get("@id") or value)
        else:
            flattened.append(value)
    flattened = [value for value in flattened if value not in (None, "", [], {})]
    if not flattened:
        return None
    return flattened[0] if len(flattened) == 1 else flattened


class BcoDmoAdapter(SourceAdapter):
    name = "bcodmo"

    def fetch_record(self, identifier: str) -> SourceRecord:
        dataset_id = _dataset_id(identifier)
        payload, from_cache = self.http.get_json(f"{self.config.base_url}/dataset/{dataset_id}")
        if not payload:
            raise SourceRecordNotFoundError(f"No BCO-DMO dataset record for {identifier}")
        jsonld_node = _jsonld_dataset_node(payload)
        return SourceRecord(
            source_name=self.name,
            external_identifier=dataset_id,
            url=(
                _first_present(payload, "url", "infoUrl")
                or _jsonld_values(jsonld_node, FOAF_HOMEPAGE)
                or f"https://www.bco-dmo.org/dataset/{dataset_id}"
            ),
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

        jsonld_node = _jsonld_dataset_node(raw)
        jsonld_fields = {
            "dataset_id": DCTERMS_IDENTIFIER,
            "title": ODO_DATASET_TITLE,
            "short_title": DCTERMS_TITLE,
            "doi": BIBO_DOI,
            "license": DCTERMS_RIGHTS,
            "abstract": ODO_ABSTRACT,
            "description": DCTERMS_DESCRIPTION,
            "citation": DCTERMS_BIBLIOGRAPHIC_CITATION,
            "projects": ODO_HAS_AWARD,
            "instruments": ODO_FROM_INSTRUMENT,
            "parameters": ODO_STORES_VALUES_FOR,
            "people": ODO_HAS_AGENT_WITH_ROLE,
        }
        for fact_type, jsonld_field in jsonld_fields.items():
            add(fact_type, _jsonld_values(jsonld_node, jsonld_field), locator=f"bcodmo.jsonld.{jsonld_field}")
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
        doi = _first_present(record.raw, "doi", "DOI") or _jsonld_values(_jsonld_dataset_node(record.raw), BIBO_DOI)
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
