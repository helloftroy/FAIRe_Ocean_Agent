"""Europe PMC adapter: DOI -> PMID/PMCID + open-access status + fulltext
availability. https://www.ebi.ac.uk/europepmc/webservices/rest/search

Europe PMC has no per-identifier REST fetch, so fetch_record() is
implemented as a search for an exact DOI and takes the first hit.

Supplementary-material discovery/retrieval (supplement-material and
structured-table retrieval layer): the main JATS full text already contains
<supplementary-material><media xlink:href="..." mimetype="..."><?size N?>
elements -- filename, mimetype, and exact byte size, for zero extra network
cost (confirmed live). Europe PMC's own {pmcid}/supplementaryFiles endpoint
has no per-file fetch -- it returns ONE zip bundling every supplementary
file for the article together (confirmed live: a real article's bundle was
33MB for 4 small XLSX tables plus one unrelated 32MB file) -- so the
bundle-vs-per-file size decision has to happen *before* ever calling
fetch_supplementary_bundle, using the sizes already known from the XML.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from fair_ocean_agent.clock import utcnow
from fair_ocean_agent.database.enums import AccessStatus, EntityLevel, IdentifierType, RelationshipType
from fair_ocean_agent.identity.identifiers import IdentifierError, normalize_doi
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


@dataclass(frozen=True)
class SupplementReference:
    supplement_id: str
    file_name: str
    mimetype: str | None
    size_bytes: int | None


def discover_supplementary_materials(fulltext_xml: str) -> list[SupplementReference]:
    """Parses <supplementary-material><media> elements out of an already-
    fetched main article XML -- no network call. Real articles repeat the
    same <supplementary-material> list verbatim under more than one parent
    section (confirmed live); this dedupes by file_name (xlink:href), first
    occurrence wins. Returns [] on unparseable XML or no matches, same
    convention as extraction/sections.py's select_relevant_sections.

    Europe PMC's per-file byte size lives in a <?size N?> processing
    instruction *inside* <media> -- ElementTree's default parser silently
    drops PIs from the tree entirely (confirmed live), so this parses with
    `TreeBuilder(insert_pis=True)` (stdlib, Python 3.8+) specifically to see
    them, rather than falling back to ad hoc regex text-matching."""
    try:
        target = ET.TreeBuilder(insert_pis=True)
        parser = ET.XMLParser(target=target)
        parser.feed(fulltext_xml)
        root = parser.close()
    except ET.ParseError:
        return []

    xlink_href = "{http://www.w3.org/1999/xlink}href"
    seen_file_names: set[str] = set()
    references: list[SupplementReference] = []
    for supp in root.iter("supplementary-material"):
        media = supp.find("media")
        if media is None:
            continue
        file_name = media.get(xlink_href)
        if not file_name or file_name in seen_file_names:
            continue
        seen_file_names.add(file_name)

        mimetype = media.get("mimetype")
        mime_subtype = media.get("mime-subtype")
        full_mimetype = f"{mimetype}/{mime_subtype}" if mimetype and mime_subtype else mimetype

        size_bytes = None
        for child in media:
            if callable(child.tag) and isinstance(child.text, str) and child.text.startswith("size "):
                size_text = child.text[len("size "):].strip()
                if size_text.isdigit():
                    size_bytes = int(size_text)
                break

        references.append(
            SupplementReference(
                supplement_id=supp.get("id") or file_name,
                file_name=file_name,
                mimetype=full_mimetype,
                size_bytes=size_bytes,
            )
        )
    return references


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

    def fetch_supplementary_bundle(self, pmcid: str) -> bytes:
        """Raw bytes of Europe PMC's single supplementary-files zip for this
        article. There is no per-file fetch here -- this endpoint always
        returns one zip bundling every supplementary file together
        (confirmed live), so a caller must already have decided (from
        discover_supplementary_materials's known sizes) that the whole
        bundle is worth downloading before calling this. Raises
        SourceRecordNotFoundError if the article has no supplementary
        files."""
        url = f"{self.config.base_url}/{pmcid}/supplementaryFiles"
        content, _ = self.http.get_binary(url)
        return content

    def find_related_from_fulltext(self, fulltext_xml: str) -> list[RelatedIdentifier]:
        """Externally-hosted repository supplements: a <supplementary-material>
        element sometimes wraps an <ext-link xlink:href="https://doi.org/..."> instead
        of a <media> (no size/mimetype available -- it isn't Europe-PMC-hosted at
        all). Surfaced as RelatedIdentifiers via the same normalize_doi this
        pipeline already uses everywhere else for DOI identification, so the
        *existing* DISCOVER_IDENTIFIERS -> structured-adapter pipeline picks
        up a PANGAEA/BCO-DMO/DataCite-resolvable DOI automatically -- no new
        fetch/parse logic needed for this case."""
        try:
            root = ET.fromstring(fulltext_xml)
        except ET.ParseError:
            return []

        xlink_href = "{http://www.w3.org/1999/xlink}href"
        seen: set[str] = set()
        related: list[RelatedIdentifier] = []
        for supp in root.iter("supplementary-material"):
            if supp.find("media") is not None:
                continue  # Europe-PMC-hosted; handled by discover_supplementary_materials
            for ext_link in supp.iter("ext-link"):
                href = ext_link.get(xlink_href)
                if not href:
                    continue
                try:
                    doi = normalize_doi(href)
                except IdentifierError:
                    continue
                if doi in seen:
                    continue
                seen.add(doi)
                related.append(
                    RelatedIdentifier(
                        identifier_type=IdentifierType.DATASET_DOI,
                        value=doi,
                        relationship_type=RelationshipType.IS_SUPPLEMENT_TO,
                        source=self.name,
                    )
                )
        return related

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
