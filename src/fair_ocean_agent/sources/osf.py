"""OSF (Open Science Framework) metadata adapter.

OSF node ids are case-insensitive (confirmed live: /v2/nodes/ezcuj/ and
/v2/nodes/EZCUJ/ both resolve), so the DOI's own uppercase suffix
(10.17605/OSF.IO/XXXXX) can be used directly without lowercasing.
"""
from __future__ import annotations

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
from fair_ocean_agent.sources.sequence_file_heuristics import (
    ListedFile,
    SequenceDataStatus,
    classify_file_listing,
    synthesize_placeholder_sample_and_run_facts,
)

_OSF_DOI_ID_RE = re.compile(r"osf\.io/([A-Z0-9]+)", re.IGNORECASE)


def _node_id(identifier: str) -> str:
    value = identifier.strip()
    match = _OSF_DOI_ID_RE.search(value)
    if match:
        return match.group(1)
    return value


class OsfAdapter(SourceAdapter):
    name = "osf"

    def fetch_record(self, identifier: str) -> SourceRecord:
        node_id = _node_id(identifier)
        try:
            payload, from_cache = self.http.get_json(f"{self.config.base_url}/nodes/{node_id}/")
        except SourceRecordNotFoundError:
            raise
        node = payload.get("data") or {}
        if not node:
            raise SourceRecordNotFoundError(f"No OSF node for {identifier}")

        files_payload, _ = self.http.get_json(f"{self.config.base_url}/nodes/{node_id}/files/osfstorage/")
        listed_files = [
            ListedFile(name=entry["attributes"]["name"], size_bytes=entry["attributes"].get("size"))
            for entry in files_payload.get("data") or []
            if entry.get("attributes", {}).get("kind") == "file"
        ]
        node["sequence_data_status"] = classify_file_listing(listed_files).value
        node["listed_file_names"] = [f.name for f in listed_files]

        attributes = node.get("attributes") or {}
        doi = (attributes.get("preprint_doi") or f"10.17605/OSF.IO/{node_id.upper()}")
        return SourceRecord(
            source_name=self.name,
            external_identifier=doi,
            url=f"https://osf.io/{node_id}",
            raw=node,
            retrieved_at=utcnow(),
            content_hash=hash_payload(node),
            from_cache=from_cache,
        )

    def search(self, query: SearchQuery) -> SearchPage:
        return SearchPage(records=[], total_count=0)

    def extract_structured_facts(self, record: SourceRecord) -> list[RawFactCandidate]:
        raw = record.raw
        attributes = raw.get("attributes") or {}
        facts: list[RawFactCandidate] = []

        def add(field: str, value, *, source_locator: str) -> None:
            if value in (None, "", [], {}):
                return
            facts.append(
                RawFactCandidate(
                    entity_level=EntityLevel.PROJECT,
                    fact_type_candidate=field,
                    raw_field_name=field,
                    raw_value=value if isinstance(value, str) else str(value),
                    source_locator=source_locator,
                )
            )

        add("title", attributes.get("title"), source_locator="osf.attributes.title")
        add("description", attributes.get("description"), source_locator="osf.attributes.description")
        add("associated_resource", f"https://doi.org/{record.external_identifier}", source_locator="osf.doi")
        add("sequence_data_status", raw.get("sequence_data_status"), source_locator="osf.sequence_data_status")
        facts.extend(
            synthesize_placeholder_sample_and_run_facts(
                repo="osf",
                doi=record.external_identifier,
                status=SequenceDataStatus(raw.get("sequence_data_status", SequenceDataStatus.ABSENT.value)),
            )
        )
        return facts

    def find_related(self, record: SourceRecord) -> list[RelatedIdentifier]:
        return [
            RelatedIdentifier(
                identifier_type=IdentifierType.DATASET_DOI,
                value=record.external_identifier,
                relationship_type=RelationshipType.IS_DATASET_FOR,
                source=self.name,
            )
        ]
