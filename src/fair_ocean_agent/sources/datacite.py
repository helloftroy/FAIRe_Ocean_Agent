"""DataCite REST API adapter.

Uses the public JSON:API endpoint at /dois/{doi} for dataset DOI metadata
and /dois?query=... for discovery. DataCite is treated as a structured
metadata source: it may expose repository landing pages and related
identifiers, but this adapter never downloads the described data files.
"""
from __future__ import annotations

import json

from fair_ocean_agent.clock import utcnow
from fair_ocean_agent.database.enums import EntityLevel, IdentifierType, RelationshipType
from fair_ocean_agent.identity.identifiers import guess_identifier_type
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


def _attrs(record: SourceRecord) -> dict:
    return (record.raw.get("data") or {}).get("attributes") or {}


def _doi_from_record(record: SourceRecord) -> str | None:
    return _attrs(record).get("doi") or (record.raw.get("data") or {}).get("id")


def _relationship_type(value: str | None) -> RelationshipType:
    relation = (value or "").lower()
    if relation in {"isreferencedby", "iscitedby"}:
        return RelationshipType.IS_CITED_BY
    if relation in {"references", "cites"}:
        return RelationshipType.CITES
    if relation in {"ispartof", "haspart", "isderivedfrom", "isversionof", "hasversion"}:
        return RelationshipType.RELATED_TO
    return RelationshipType.RELATED_TO


class DataCiteAdapter(SourceAdapter):
    name = "datacite"

    def fetch_record(self, identifier: str) -> SourceRecord:
        url = f"{self.config.base_url}/dois/{identifier}"
        payload, from_cache = self.http.get_json(url, params={"affiliation": "true", "publisher": "true"})
        data = payload.get("data")
        if not data:
            raise SourceRecordNotFoundError(f"No DataCite DOI record for {identifier}")
        attributes = data.get("attributes") or {}
        return SourceRecord(
            source_name=self.name,
            external_identifier=attributes.get("doi") or data.get("id") or identifier,
            url=attributes.get("url") or f"https://doi.org/{identifier}",
            raw=payload,
            retrieved_at=utcnow(),
            content_hash=hash_payload(payload),
            from_cache=from_cache,
        )

    def find_datasets_citing(self, article_doi: str) -> list[str]:
        """Pass 3 (discovery/text_identifiers.py's staged repository search):
        datasets whose OWN DataCite record declares a relatedIdentifiers
        link back to `article_doi` -- a structured-API discovery channel,
        distinct from mining the article's own text, that can surface a
        dataset the paper's Data Availability section never names at all.
        Confirmed live (both that the query mechanism itself works, via a
        record with a known relation, and that it correctly returns nothing
        for a record that has none): DataCite's query field needs the DOI
        quoted (unquoted, a bare "10.xxxx/yyyy" is tokenized by slashes and
        matches nothing)."""
        payload, _ = self.http.get_json(
            f"{self.config.base_url}/dois",
            params={"query": f'relatedIdentifiers.relatedIdentifier:"{article_doi}"', "page[size]": 25},
        )
        dois: list[str] = []
        for item in payload.get("data") or []:
            doi = (item.get("attributes") or {}).get("doi") or item.get("id")
            if doi and doi.lower() != article_doi.lower():
                dois.append(doi)
        return dois

    def search(self, query: SearchQuery) -> SearchPage:
        payload, _ = self.http.get_json(
            f"{self.config.base_url}/dois",
            params={"query": query.query, "page[size]": query.limit, "page[cursor]": query.cursor or "1"},
        )
        data = payload.get("data") or []
        records = []
        for item in data:
            attributes = item.get("attributes") or {}
            raw = {"data": item}
            records.append(
                SourceRecord(
                    source_name=self.name,
                    external_identifier=attributes.get("doi") or item.get("id") or "",
                    url=attributes.get("url"),
                    raw=raw,
                    retrieved_at=utcnow(),
                    content_hash=hash_payload(raw),
                )
            )
        meta = payload.get("meta") or {}
        return SearchPage(records=records, next_cursor=meta.get("nextCursor"), total_count=meta.get("total"))

    def extract_structured_facts(self, record: SourceRecord) -> list[RawFactCandidate]:
        attrs = _attrs(record)
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
                    source_locator=f"datacite.data.attributes.{field}",
                )
            )

        for field in (
            "doi", "titles", "creators", "publisher", "publicationYear", "types",
            "subjects", "dates", "descriptions", "geoLocations", "rightsList",
            "relatedIdentifiers", "url", "version", "schemaVersion",
        ):
            add(field, attrs.get(field))
        return facts

    def find_related(self, record: SourceRecord) -> list[RelatedIdentifier]:
        related: list[RelatedIdentifier] = []
        doi = _doi_from_record(record)
        if doi:
            related.append(
                RelatedIdentifier(
                    identifier_type=IdentifierType.DATASET_DOI,
                    value=doi,
                    relationship_type=RelationshipType.IS_DATASET_FOR,
                    source=self.name,
                )
            )

        for item in _attrs(record).get("relatedIdentifiers") or []:
            value = item.get("relatedIdentifier")
            if not value:
                continue
            related_type = (item.get("relatedIdentifierType") or "").lower()
            identifier_type = IdentifierType.DOI if related_type == "doi" else guess_identifier_type(value)
            if identifier_type is None:
                identifier_type = IdentifierType.URL if related_type == "url" else IdentifierType.OTHER
            related.append(
                RelatedIdentifier(
                    identifier_type=identifier_type,
                    value=value,
                    relationship_type=_relationship_type(item.get("relationType")),
                    source=self.name,
                )
            )
        return related
