"""Figshare metadata adapter.

Unlike Zenodo/Dryad, an article's file listing is embedded directly in its
own metadata response (confirmed live: GET /v2/articles/{id} returns a top
-level "files" array) -- no separate files endpoint to fetch.
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

# Figshare DOIs occasionally carry a trailing version segment
# (10.6084/m9.figshare.NNNNNNN.v2) -- the article id is always the bare
# numeric segment right after "figshare.".
_FIGSHARE_DOI_ID_RE = re.compile(r"figshare\.(\d+)", re.IGNORECASE)


def _article_id(identifier: str) -> str:
    value = identifier.strip()
    match = _FIGSHARE_DOI_ID_RE.search(value)
    if match:
        return match.group(1)
    if value.isdigit():
        return value
    return value


class FigshareAdapter(SourceAdapter):
    name = "figshare"

    def fetch_record(self, identifier: str) -> SourceRecord:
        article_id = _article_id(identifier)
        payload, from_cache = self.http.get_json(f"{self.config.base_url}/articles/{article_id}")
        if not payload:
            raise SourceRecordNotFoundError(f"No Figshare article for {identifier}")

        listed_files = [
            ListedFile(name=entry["name"], size_bytes=entry.get("size"))
            for entry in payload.get("files") or []
            if entry.get("name")
        ]
        payload["sequence_data_status"] = classify_file_listing(listed_files).value
        payload["listed_file_names"] = [f.name for f in listed_files]

        real_doi = payload.get("doi") or identifier
        return SourceRecord(
            source_name=self.name,
            external_identifier=real_doi,
            url=payload.get("url_public_html") or f"https://doi.org/{real_doi}",
            raw=payload,
            retrieved_at=utcnow(),
            content_hash=hash_payload(payload),
            from_cache=from_cache,
        )

    def search(self, query: SearchQuery) -> SearchPage:
        return SearchPage(records=[], total_count=0)

    def extract_structured_facts(self, record: SourceRecord) -> list[RawFactCandidate]:
        raw = record.raw
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

        add("title", raw.get("title"), source_locator="figshare.title")
        add("description", raw.get("description"), source_locator="figshare.description")
        add("license", (raw.get("license") or {}).get("name"), source_locator="figshare.license.name")
        add(
            "associated_resource",
            raw.get("url_public_html") or f"https://doi.org/{record.external_identifier}",
            source_locator="figshare.url_public_html",
        )
        add("sequence_data_status", raw.get("sequence_data_status"), source_locator="figshare.sequence_data_status")
        facts.extend(
            synthesize_placeholder_sample_and_run_facts(
                repo="figshare",
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
