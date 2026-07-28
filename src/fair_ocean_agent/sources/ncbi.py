"""NCBI BioProject + BioSample adapters, via E-utilities (esearch/elink/efetch).

Unlike Crossref/Europe PMC/OpenAlex (Milestone 2, all Publication-oriented),
these are data-repository adapters: they describe the project and its
samples, not a paper, so parse_publication_fields() is unused (default {})
and extract_structured_facts() emits per-sample facts via
RawFactCandidate.entity_external_id for the handler to materialize as
`sample` Entity rows.

NCBI SRA is deliberately not implemented as a separate adapter: run-level
data (library strategy, platform, file accessions) is available from the
same underlying INSDC-shared records via the ENA adapter's read_run query,
which returns clean JSON where NCBI's SRA XML (EXPERIMENT_PACKAGE_SET) is
considerably messier to parse for the same facts. See docs/architecture.md.

esearch/efetch return JSON and XML respectively (E-utilities doesn't offer
XML-free responses for full BioProject/BioSample records), so this module
does its own XML parsing rather than reusing the JSON-only hash_payload
convention from sources/base.py -- content_hash is computed over the raw
XML/JSON text instead.
"""
from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET

from fair_ocean_agent.clock import utcnow
from fair_ocean_agent.database.enums import EntityLevel, IdentifierType, RelationshipType
from fair_ocean_agent.logging_setup import get_logger
from fair_ocean_agent.sources.base import (
    RawFactCandidate,
    RelatedIdentifier,
    SearchPage,
    SearchQuery,
    SourceAdapter,
    SourceRecord,
    SourceRecordNotFoundError,
)

logger = get_logger(__name__)

# NCBI recommends batching efetch calls rather than one request per ID; this
# also bounds worst-case work for a single DISCOVER_IDENTIFIERS task against
# a project with an unusually large number of deposited samples (some eDNA
# time series run into the hundreds). Truncation is logged, never silent.
MAX_SAMPLES_PER_PROJECT = 300
EFETCH_BATCH_SIZE = 100


def _esearch_first_uid(http, base_url: str, db: str, term: str) -> str | None:
    payload, _ = http.get_json(f"{base_url}/esearch.fcgi", params={"db": db, "term": term, "retmode": "json"})
    ids = payload.get("esearchresult", {}).get("idlist", [])
    return ids[0] if ids else None


def _elink_ids(http, base_url: str, dbfrom: str, db: str, uid: str) -> list[str]:
    payload, _ = http.get_json(
        f"{base_url}/elink.fcgi", params={"dbfrom": dbfrom, "db": db, "id": uid, "retmode": "json"}
    )
    linksets = payload.get("linksets", [])
    if not linksets or not linksets[0].get("linksetdbs"):
        return []
    return list(linksets[0]["linksetdbs"][0].get("links", []))


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class NcbiBioProjectAdapter(SourceAdapter):
    name = "ncbi_bioproject"

    def fetch_record(self, identifier: str) -> SourceRecord:
        uid = _esearch_first_uid(self.http, self.config.base_url, "bioproject", identifier)
        if uid is None:
            raise SourceRecordNotFoundError(f"No BioProject UID found for {identifier}")

        xml_text, from_cache = self.http.get_text(
            f"{self.config.base_url}/efetch.fcgi", params={"db": "bioproject", "id": uid, "retmode": "xml"}
        )
        root = ET.fromstring(xml_text)
        summary = root.find("DocumentSummary")
        if summary is None:
            raise SourceRecordNotFoundError(f"Empty BioProject record for {identifier} (uid {uid})")

        archive_id = summary.find("Project/ProjectID/ArchiveID")
        accession = archive_id.get("accession") if archive_id is not None else identifier

        raw = {
            "uid": uid,
            "accession": accession,
            "name": summary.findtext("Project/ProjectDescr/Name"),
            "title": summary.findtext("Project/ProjectDescr/Title"),
            "description": summary.findtext("Project/ProjectDescr/Description"),
            "organism": summary.findtext(".//ProjectType//Organism/OrganismName"),
            "submitted": (summary.find("Submission").get("submitted") if summary.find("Submission") is not None else None),
        }

        return SourceRecord(
            source_name=self.name,
            external_identifier=identifier,
            url=f"https://www.ncbi.nlm.nih.gov/bioproject/{uid}",
            raw=raw,
            retrieved_at=utcnow(),
            content_hash=_hash_text(xml_text),
            from_cache=from_cache,
        )

    def search(self, query: SearchQuery) -> SearchPage:
        payload, _ = self.http.get_json(
            f"{self.config.base_url}/esearch.fcgi",
            params={"db": "bioproject", "term": query.query, "retmode": "json", "retmax": query.limit},
        )
        ids = payload.get("esearchresult", {}).get("idlist", [])
        records = []
        for uid in ids:
            try:
                records.append(self.fetch_record(uid))
            except SourceRecordNotFoundError:
                continue
        return SearchPage(records=records, total_count=int(payload.get("esearchresult", {}).get("count", 0)))

    def extract_structured_facts(self, record: SourceRecord) -> list[RawFactCandidate]:
        r = record.raw
        facts: list[RawFactCandidate] = []

        def add(field: str, value) -> None:
            if value in (None, "", [], {}):
                return
            facts.append(
                RawFactCandidate(
                    entity_level=EntityLevel.PROJECT,
                    fact_type_candidate=field,
                    raw_field_name=field,
                    raw_value=str(value),
                    source_locator=f"ncbi_bioproject.DocumentSummary.{field}",
                )
            )

        add("name", r.get("name"))
        add("title", r.get("title"))
        add("description", r.get("description"))
        add("organism", r.get("organism"))
        add("submitted", r.get("submitted"))
        return facts


class NcbiBioSampleAdapter(SourceAdapter):
    """Given a BioProject accession, discovers and fetches every linked
    BioSample's full attribute record. One fetch_record() call = one
    project's worth of samples (see module docstring on Source-row
    granularity in workflow/handlers.py)."""

    name = "ncbi_biosample"

    def fetch_record(self, identifier: str) -> SourceRecord:
        project_uid = _esearch_first_uid(self.http, self.config.base_url, "bioproject", identifier)
        if project_uid is None:
            raise SourceRecordNotFoundError(f"No BioProject UID found for {identifier}")

        sample_uids = _elink_ids(self.http, self.config.base_url, "bioproject", "biosample", project_uid)
        if not sample_uids:
            raise SourceRecordNotFoundError(f"No linked BioSamples for BioProject {identifier}")

        total_linked_samples = len(sample_uids)
        truncated = total_linked_samples > MAX_SAMPLES_PER_PROJECT
        if truncated:
            logger.warning(
                "BioProject %s has %d linked BioSamples; processing only the first %d "
                "(MAX_SAMPLES_PER_PROJECT) -- not a silent drop, see raw_facts for the count.",
                identifier, total_linked_samples, MAX_SAMPLES_PER_PROJECT,
            )
        sample_uids = sample_uids[:MAX_SAMPLES_PER_PROJECT]

        samples: list[dict] = []
        combined_xml_for_hash: list[str] = []
        for i in range(0, len(sample_uids), EFETCH_BATCH_SIZE):
            batch = sample_uids[i : i + EFETCH_BATCH_SIZE]
            xml_text, _ = self.http.get_text(
                f"{self.config.base_url}/efetch.fcgi",
                params={"db": "biosample", "id": ",".join(batch), "rettype": "full", "retmode": "xml"},
            )
            combined_xml_for_hash.append(xml_text)
            root = ET.fromstring(xml_text)
            for bs in root.findall("BioSample"):
                accession = bs.get("accession")
                if not accession:
                    continue
                samples.append(
                    {
                        "accession": accession,
                        "title": bs.findtext("Description/Title"),
                        "attributes": {
                            attr.get("attribute_name"): attr.text
                            for attr in bs.findall("Attributes/Attribute")
                            if attr.get("attribute_name")
                        },
                    }
                )

        raw = {
            "bioproject_accession": identifier,
            "total_linked_samples": total_linked_samples,
            "truncated": truncated,
            "samples": samples,
        }

        return SourceRecord(
            source_name=self.name,
            external_identifier=identifier,
            url=None,
            raw=raw,
            retrieved_at=utcnow(),
            content_hash=_hash_text("".join(combined_xml_for_hash)),
        )

    def search(self, query: SearchQuery) -> SearchPage:
        # BioSample has no free-text "search" use case in this pipeline --
        # it's always resolved via a BioProject accession (fetch_record).
        return SearchPage(records=[])

    def extract_structured_facts(self, record: SourceRecord) -> list[RawFactCandidate]:
        r = record.raw
        facts: list[RawFactCandidate] = []

        if r.get("truncated"):
            facts.append(
                RawFactCandidate(
                    entity_level=EntityLevel.PROJECT,
                    fact_type_candidate="biosample_coverage_note",
                    raw_field_name="biosample_coverage_note",
                    raw_value=(
                        f"Only the first {len(r['samples'])} of {r['total_linked_samples']} "
                        f"linked BioSamples were processed (MAX_SAMPLES_PER_PROJECT cap)."
                    ),
                    source_locator="ncbi_biosample.truncation_note",
                )
            )

        for sample in r.get("samples", []):
            accession = sample["accession"]
            for attr_name, attr_value in sample.get("attributes", {}).items():
                if attr_value in (None, ""):
                    continue
                facts.append(
                    RawFactCandidate(
                        entity_level=EntityLevel.SAMPLE,
                        fact_type_candidate=attr_name,
                        raw_field_name=attr_name,
                        raw_value=str(attr_value),
                        source_locator=f"ncbi_biosample.{accession}.Attributes.{attr_name}",
                        entity_external_id=accession,
                        entity_label=sample.get("title"),
                    )
                )
        return facts

    def find_related(self, record: SourceRecord) -> list[RelatedIdentifier]:
        return [
            RelatedIdentifier(
                identifier_type=IdentifierType.BIOSAMPLE_ACCESSION,
                value=sample["accession"],
                relationship_type=RelationshipType.CONTAINS_SAMPLES_FROM,
                source=self.name,
            )
            for sample in record.raw.get("samples", [])
        ]
