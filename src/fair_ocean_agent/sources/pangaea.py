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


def _defined_terms(value) -> list[dict]:
    """Return schema.org DefinedTerm-like dicts from PANGAEA's nested
    variableMeasured.subjectOf.hasDefinedTerm shape. PANGAEA emits either a
    single object or a list, and occasionally uses loose string-ish terms;
    only dict terms have identifiers/names worth decomposing."""
    if not isinstance(value, dict):
        return []
    subject = value.get("subjectOf")
    if not isinstance(subject, dict):
        return []
    return [term for term in _as_list(subject.get("hasDefinedTerm")) if isinstance(term, dict)]


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

        def add(field: str, value, *, source_locator: str | None = None, confidence_metadata: dict | None = None) -> None:
            if value in (None, "", [], {}):
                return
            facts.append(
                RawFactCandidate(
                    entity_level=EntityLevel.PROJECT,
                    fact_type_candidate=field,
                    raw_field_name=field,
                    raw_value=_stringify(value),
                    source_locator=source_locator or f"pangaea.jsonld.{field}",
                    confidence_metadata=confidence_metadata,
                )
            )

        for field in (
            "@id", "identifier", "name", "headline", "description", "creator",
            "publisher", "datePublished", "dateModified", "license",
            "spatialCoverage", "temporalCoverage", "variableMeasured", "keywords",
            "distribution", "citation",
        ):
            add(field, raw.get(field))

        # `variableMeasured` is often where PANGAEA exposes dataset
        # parameter/column metadata (name, unit, measurement technique,
        # ontology identifiers). Keep the original blob above, but also emit
        # one structured fact per parameter property so downstream mapping
        # and review can use the details without parsing a giant JSON value.
        for index, variable in enumerate(_as_list(raw.get("variableMeasured"))):
            if not isinstance(variable, dict):
                continue
            prefix = f"pangaea.jsonld.variableMeasured[{index}]"
            variable_name = variable.get("name")
            metadata = {"variable_index": index}
            if variable_name:
                metadata["variable_name"] = variable_name
                add(
                    "pangaea_variable_name",
                    variable_name,
                    source_locator=f"{prefix}.name",
                    confidence_metadata=metadata,
                )
            for field, fact_type in (
                ("unitText", "pangaea_variable_unit"),
                ("measurementTechnique", "pangaea_variable_measurement_technique"),
                ("description", "pangaea_variable_description"),
                ("url", "pangaea_variable_url"),
            ):
                add(
                    fact_type,
                    variable.get(field),
                    source_locator=f"{prefix}.{field}",
                    confidence_metadata=metadata,
                )
            for term_index, term in enumerate(_defined_terms(variable)):
                term_metadata = {**metadata, "defined_term_index": term_index}
                add(
                    "pangaea_variable_defined_term_name",
                    term.get("name"),
                    source_locator=f"{prefix}.subjectOf.hasDefinedTerm[{term_index}].name",
                    confidence_metadata=term_metadata,
                )
                add(
                    "pangaea_variable_defined_term_identifier",
                    term.get("identifier") or term.get("@id"),
                    source_locator=f"{prefix}.subjectOf.hasDefinedTerm[{term_index}].identifier",
                    confidence_metadata=term_metadata,
                )
                add(
                    "pangaea_variable_defined_term_url",
                    term.get("url"),
                    source_locator=f"{prefix}.subjectOf.hasDefinedTerm[{term_index}].url",
                    confidence_metadata=term_metadata,
                )
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
