"""Zenodo metadata adapter.

Zenodo DOIs cited in a paper's Data Availability section are very often the
"concept" DOI (representing every version of a record together), not the
specific version DOI -- confirmed live: 10.5281/zenodo.10381280 (cited in a
real paper) 302-redirects record id 10381280 to the actual current version,
10381281. sources/base.py's RateLimitedClient now follows redirects for
exactly this reason, so fetch_record here can just request the cited id
directly and trust it lands on the real record.
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

_ZENODO_DOI_ID_RE = re.compile(r"zenodo\.(\d+)", re.IGNORECASE)


def _record_id(identifier: str) -> str:
    value = identifier.strip()
    match = _ZENODO_DOI_ID_RE.search(value)
    if match:
        return match.group(1)
    if value.isdigit():
        return value
    return value


class ZenodoAdapter(SourceAdapter):
    name = "zenodo"

    def fetch_record(self, identifier: str) -> SourceRecord:
        record_id = _record_id(identifier)
        payload, from_cache = self.http.get_json(f"{self.config.base_url}/records/{record_id}")
        if not payload:
            raise SourceRecordNotFoundError(f"No Zenodo record for {identifier}")

        files_url = (payload.get("links") or {}).get("files")
        listed_files: list[ListedFile] = []
        if files_url:
            files_payload, _ = self.http.get_json(files_url)
            for entry in files_payload.get("entries") or []:
                key = entry.get("key")
                if key:
                    listed_files.append(ListedFile(name=key, size_bytes=entry.get("size")))
        payload["sequence_data_status"] = classify_file_listing(listed_files).value
        payload["listed_file_names"] = [f.name for f in listed_files]

        real_doi = payload.get("doi") or identifier
        return SourceRecord(
            source_name=self.name,
            external_identifier=real_doi,
            url=(payload.get("links") or {}).get("self_html") or f"https://doi.org/{real_doi}",
            raw=payload,
            retrieved_at=utcnow(),
            content_hash=hash_payload(payload),
            from_cache=from_cache,
        )

    def search(self, query: SearchQuery) -> SearchPage:
        # DataCite is the stable JSON search surface for dataset DOIs in
        # this pipeline (same rationale as pangaea.py/bcodmo.py); keep
        # search() explicit and bounded here.
        return SearchPage(records=[], total_count=0)

    def extract_structured_facts(self, record: SourceRecord) -> list[RawFactCandidate]:
        raw = record.raw
        metadata = raw.get("metadata") or {}
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

        add("title", metadata.get("title"), source_locator="zenodo.metadata.title")
        add("description", metadata.get("description"), source_locator="zenodo.metadata.description")
        add("license", (metadata.get("license") or {}).get("id"), source_locator="zenodo.metadata.license.id")
        add(
            "associated_resource",
            raw.get("doi_url") or (raw.get("links") or {}).get("self_html"),
            source_locator="zenodo.doi_url",
        )
        add(
            "sequence_data_status",
            raw.get("sequence_data_status"),
            source_locator="zenodo.sequence_data_status",
        )
        facts.extend(
            synthesize_placeholder_sample_and_run_facts(
                repo="zenodo",
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
