"""Europe PMC adapter: DOI -> PMID/PMCID + open-access status + fulltext
availability. https://www.ebi.ac.uk/europepmc/webservices/rest/search

Europe PMC has no per-identifier REST fetch, so fetch_record() is
implemented as a search for an exact DOI and takes the first hit.
"""
from __future__ import annotations

import json

from fair_ocean_agent.clock import utcnow
from fair_ocean_agent.database.enums import AccessStatus, EntityLevel, IdentifierType, RelationshipType
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


class EuropePmcAdapter(SourceAdapter):
    name = "europe_pmc"

    def fetch_record(self, identifier: str) -> SourceRecord:
        page = self.search(SearchQuery(query=f'DOI:"{identifier}"', limit=1))
        if not page.records:
            raise SourceRecordNotFoundError(f"No Europe PMC record for DOI {identifier}")
        return page.records[0]

    def fetch_fulltext_xml(self, pmcid: str) -> str:
        """Raw JATS full-text XML for an open-access article
        (section 10/13: open-access retrieval only, never a paywalled or
        access-controlled source). Raises SourceRecordNotFoundError if the
        PMCID doesn't exist or has no full text in Europe PMC (a genuine
        404, not a heuristic -- Europe PMC returns 404/empty body for
        exactly this case)."""
        url = f"{self.config.base_url}/{pmcid}/fullTextXML"
        text, _ = self.http.get_text(url)
        return text

    def search(self, query: SearchQuery) -> SearchPage:
        url = f"{self.config.base_url}/search"
        params = {
            "query": query.query,
            "format": "json",
            "resultType": "core",
            "pageSize": query.limit,
        }
        payload, _ = self.http.get_json(url, params=params)
        results = payload.get("resultList", {}).get("result", [])
        records = [
            SourceRecord(
                source_name=self.name,
                external_identifier=r.get("doi") or r.get("pmid") or "",
                url=None,
                raw=r,
                retrieved_at=utcnow(),
                content_hash=hash_payload(r),
            )
            for r in results
        ]
        return SearchPage(records=records, total_count=payload.get("hitCount"))

    def extract_structured_facts(self, record: SourceRecord) -> list[RawFactCandidate]:
        r = record.raw
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
                    source_locator=f"europe_pmc.result.{field}",
                )
            )

        add("title", r.get("title"))
        add("authorString", r.get("authorString"))
        add("journalTitle", r.get("journalTitle"))
        add("pubYear", r.get("pubYear"))
        add("isOpenAccess", r.get("isOpenAccess"))
        add("inEPMC", r.get("inEPMC"))
        add("fullTextUrlList", r.get("fullTextUrlList"))
        return facts

    def parse_publication_fields(self, record: SourceRecord) -> dict:
        r = record.raw
        is_open = r.get("isOpenAccess") == "Y"
        return {
            "pmid": r.get("pmid"),
            "pmcid": r.get("pmcid"),
            "title": r.get("title"),
            "journal": r.get("journalTitle"),
            "publication_year": int(r["pubYear"]) if r.get("pubYear") else None,
            "open_access_status": AccessStatus.OPEN.value if is_open else AccessStatus.UNKNOWN.value,
            "fulltext_available": r.get("inEPMC") == "Y" or r.get("inPMC") == "Y",
        }

    def find_related(self, record: SourceRecord) -> list[RelatedIdentifier]:
        r = record.raw
        related = []
        if r.get("pmid"):
            related.append(
                RelatedIdentifier(
                    identifier_type=IdentifierType.PMID,
                    value=r["pmid"],
                    relationship_type=RelationshipType.IS_PUBLICATION_OF,
                    source=self.name,
                )
            )
        if r.get("pmcid"):
            related.append(
                RelatedIdentifier(
                    identifier_type=IdentifierType.PMCID,
                    value=r["pmcid"],
                    relationship_type=RelationshipType.IS_PUBLICATION_OF,
                    source=self.name,
                )
            )
        return related
