"""Dryad metadata adapter.

Dryad's own dataset-lookup endpoint requires the DOI prefixed with "doi:"
and URL-encoded as a single path segment (confirmed live:
/api/v2/datasets/doi%3A10.5061%2Fdryad.xksn02vdx, not a bare DOI path). File
listings live under the dataset's *version*, a separate resource reached
via the dataset response's own _links.
"""
from __future__ import annotations

import re
from urllib.parse import quote

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

_DRYAD_DOI_RE = re.compile(r"10\.5061/dryad\.[a-z0-9]+", re.IGNORECASE)


def _doi(identifier: str) -> str:
    value = identifier.strip()
    if value.lower().startswith("https://doi.org/"):
        value = value.rsplit("doi.org/", 1)[-1]
    match = _DRYAD_DOI_RE.search(value)
    return match.group(0) if match else value


class DryadAdapter(SourceAdapter):
    name = "dryad"

    def fetch_record(self, identifier: str) -> SourceRecord:
        doi = _doi(identifier)
        dataset_url = f"{self.config.base_url}/datasets/{quote(f'doi:{doi}', safe='')}"
        try:
            payload, from_cache = self.http.get_json(dataset_url)
        except SourceRecordNotFoundError:
            raise
        if not payload:
            raise SourceRecordNotFoundError(f"No Dryad dataset for {identifier}")

        version_href = ((payload.get("_links") or {}).get("stash:version") or {}).get("href")
        listed_files: list[ListedFile] = []
        if version_href:
            files_payload, _ = self.http.get_json(f"https://datadryad.org{version_href}/files")
            for entry in (files_payload.get("_embedded") or {}).get("stash:files") or []:
                path = entry.get("path")
                if path:
                    listed_files.append(ListedFile(name=path, size_bytes=entry.get("size")))
        payload["sequence_data_status"] = classify_file_listing(listed_files).value
        payload["listed_file_names"] = [f.name for f in listed_files]

        return SourceRecord(
            source_name=self.name,
            external_identifier=doi,
            url=f"https://doi.org/{doi}",
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

        add("title", raw.get("title"), source_locator="dryad.title")
        add("description", raw.get("abstract"), source_locator="dryad.abstract")
        add("associated_resource", f"https://doi.org/{record.external_identifier}", source_locator="dryad.doi")
        add("sequence_data_status", raw.get("sequence_data_status"), source_locator="dryad.sequence_data_status")
        facts.extend(
            synthesize_placeholder_sample_and_run_facts(
                repo="dryad",
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
