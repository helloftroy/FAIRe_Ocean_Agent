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
import re
from collections import defaultdict
import xml.etree.ElementTree as ET

from fair_ocean_agent.clock import utcnow
from fair_ocean_agent.database.enums import EntityLevel, IdentifierType, RelationshipType, SupportType
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
from fair_ocean_agent.sources.replicate_grouping import detect_replicate_groups

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


def _clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(value.split())
    return cleaned or None


# A BioSample submitted as a metagenome-assembled genome (MAG) describes a
# reconstructed genome BIN, not the physical environmental sample it was
# assembled from -- confirmed live (a real efetch against a real MAG
# accession, ISME J/marine-sediment audit) that such a record's "sample
# derived from"/title only ever reference the real sample, they never
# carry the real sample's own attributes (depth, lat/lon, collection
# method) themselves, and instead carry assembly-specific attributes
# (assembly software, completeness/contamination score, binning software)
# a raw environmental sample never has. A BioProject can legitimately link
# to BOTH the raw samples and their downstream MAG BioSamples together
# (confirmed live: one real BioProject returned a mix of ~80 real
# environmental BioSamples and a handful of MIMAG-packaged ones cross-
# linked from an entirely different downstream assembly project) --
# without this exclusion, MAG records pollute sampleMetadata with
# assembly-bin-level facts under the same SAMPLE entity_level as real
# samples. MIMAG ("Minimum Information about a MAG") is the real,
# standard INSDC checklist used for exactly this kind of submission --
# `package`/`Models/Model` carrying it is the authoritative structural
# signal; the title text is a secondary fallback for a record that
# doesn't carry the package/model fields for some reason.
_MAG_TITLE_RE = re.compile(r"\bmetagenome-assembled genome\b", re.IGNORECASE)


def _is_mag_biosample(sample: dict) -> bool:
    package = (sample.get("package") or "").upper()
    model = (sample.get("model") or "").upper()
    if package.startswith("MIMAG") or model == "MIMAG":
        return True
    return bool(_MAG_TITLE_RE.search(sample.get("title") or ""))


# samp_mat_process is schema-documented to hold a full free-text
# processing narrative (e.g. "0.22 um cartridge filtration followed by DNA
# extraction") -- these patterns pull the specific sub-facts FAIRe asks
# for out of that narrative without altering samp_mat_process's own raw
# fact, which stays the verbatim sentence. Confirmed via real evidence
# (PeerJ/marine-microbiome audits) NOT to conflate pore size (µm, this
# module's size_frac) with filter_diameter (mm, "Diameter of a filter if
# circular" -- the physical disc size, a different concept and unit a
# naive parser could easily get wrong).
_PORE_SIZE_RE = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*(?:µm|um|micrometers?|microns?)\b",
    re.IGNORECASE,
)
_PORE_CONTEXT_RE = re.compile(r"\bpore\b|\bfilt\w*\b", re.IGNORECASE)
_FILTER_DIAMETER_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*mm\b", re.IGNORECASE)
_FILTER_DIAMETER_CONTEXT_RE = re.compile(r"\bfilter\b|\bdisc\b|\bdisk\b|\bmembrane\b", re.IGNORECASE)
_FILTER_MATERIAL_TERMS = (
    "cellulose ester",
    "cellulose",
    "glass fiber",
    "nylon",
    "polyethersulfone",
    "thermoplastic membrane",
    "track etched polycarbonate",
)
_FILTER_NAME_TERMS = ("Sterivex", "Millipore", "Whatman", "Nalgene", "Supor", "Durapore")
_FILTER_ACTIVE_RE = re.compile(r"\bcartridge\b|\bpump(?:ed|ing)?\b|\bperistaltic\b", re.IGNORECASE)
_FILTER_PASSIVE_RE = re.compile(r"\bpassive\b|\bsubmerged\b|\bgravity\b", re.IGNORECASE)
_CONTEXT_WINDOW = 40


def _derive_filter_facts(value: str) -> dict[str, str]:
    """Parses samp_mat_process's free text for size_frac/filter_* sub-facts
    -- see this module's own comment above for why pore size and filter
    diameter are deliberately never conflated. Only ever returns a field
    when its own specific pattern (and, for size_frac/filter_diameter,
    nearby context) genuinely matched -- never guesses."""
    derived: dict[str, str] = {}
    pore_match = _PORE_SIZE_RE.search(value)
    if pore_match:
        window = value[max(0, pore_match.start() - _CONTEXT_WINDOW) : pore_match.end() + _CONTEXT_WINDOW]
        if _PORE_CONTEXT_RE.search(window):
            derived["size_frac"] = pore_match.group(0)
    diameter_match = _FILTER_DIAMETER_RE.search(value)
    if diameter_match:
        window = value[max(0, diameter_match.start() - _CONTEXT_WINDOW) : diameter_match.end() + _CONTEXT_WINDOW]
        if _FILTER_DIAMETER_CONTEXT_RE.search(window):
            derived["filter_diameter"] = diameter_match.group("value")
    for term in _FILTER_MATERIAL_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", value, re.IGNORECASE):
            derived["filter_material"] = term
            break
    for term in _FILTER_NAME_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", value, re.IGNORECASE):
            derived["filter_name"] = term
            break
    if _FILTER_ACTIVE_RE.search(value):
        derived["filter_passive_active_0_1"] = "1"
    elif _FILTER_PASSIVE_RE.search(value):
        derived["filter_passive_active_0_1"] = "0"
    return derived


# source_material_id is a real BioSample/MIxS attribute name generically
# meaning "an identifier for the source material" -- NOT inherently about
# depth. Confirmed via real evidence (a real study's own submission
# convention, "3500 m V3-V4"/"Overlaying water V3-V4") that one submitter
# repurposed it to embed per-sample depth alongside an unrelated marker-gene
# suffix. Deliberately narrow: only matches a value that IS just a leading
# number (optionally "m"), never a value embedded deeper inside a longer
# alphanumeric id, to avoid misreading a genuine catalog/voucher id (e.g.
# "MSC2019-047") as if it were a depth in meters.
_SOURCE_MATERIAL_ID_DEPTH_RE = re.compile(r"^(?P<value>\d+(?:\.\d+)?)\s*m?\b")


def _derive_depth_from_source_material_id(value: str) -> str | None:
    match = _SOURCE_MATERIAL_ID_DEPTH_RE.match(value.strip())
    return match.group(0).strip() if match else None


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
                organism_el = bs.find("Description/Organism")
                organism = dict(organism_el.attrib) if organism_el is not None else {}
                owner_name_el = bs.find("Owner/Name")
                owner_name = _clean_text(owner_name_el.text if owner_name_el is not None else None)
                submission_name = _clean_text(bs.findtext("Submission/Name") or bs.findtext("Submission"))
                samples.append(
                    {
                        "accession": accession,
                        "title": bs.findtext("Description/Title"),
                        "package": bs.get("package"),
                        "model": bs.findtext("Models/Model"),
                        "organism": {key: value for key, value in organism.items() if value},
                        "attributes": {
                            attr.get("attribute_name"): attr.text
                            for attr in bs.findall("Attributes/Attribute")
                            if attr.get("attribute_name")
                        },
                        "owner": {
                            key: value
                            for key, value in {
                                "name": owner_name,
                                "abbreviation": (
                                    owner_name_el.get("abbreviation") if owner_name_el is not None else None
                                ),
                            }.items()
                            if value
                        },
                        "submission": {"name": submission_name} if submission_name else {},
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
            if _is_mag_biosample(sample):
                continue
            accession = sample["accession"]
            normalized_attrs: dict[str, str] = {}
            for attr_name, attr_value in sample.get("attributes", {}).items():
                if attr_value in (None, ""):
                    continue
                normalized_attrs[attr_name] = str(attr_value)
                if attr_name.casefold() == "geographic location":
                    normalized_attrs["geo_loc_name"] = str(attr_value)
                if attr_name.casefold() == "cultivar":
                    normalized_attrs["host_species"] = str(attr_value)
            organism = sample.get("organism") or {}
            organism_name = organism.get("taxonomy_name") or organism.get("taxonomy_id")
            if organism_name:
                normalized_attrs.setdefault("organism", str(organism_name))
                normalized_attrs.setdefault("host_species", str(organism_name))
            for attr_name, attr_value in normalized_attrs.items():
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
            samp_mat_process = sample.get("attributes", {}).get("samp_mat_process")
            if samp_mat_process:
                for field, value in _derive_filter_facts(str(samp_mat_process)).items():
                    facts.append(
                        RawFactCandidate(
                            entity_level=EntityLevel.SAMPLE,
                            fact_type_candidate=field,
                            raw_field_name="samp_mat_process",
                            raw_value=value,
                            source_locator=f"ncbi_biosample.{accession}.Attributes.samp_mat_process.{field}",
                            entity_external_id=accession,
                            entity_label=sample.get("title"),
                            support_type=SupportType.DETERMINISTICALLY_DERIVED,
                        )
                    )
            source_material_id = sample.get("attributes", {}).get("source_material_id")
            if source_material_id:
                depth_value = _derive_depth_from_source_material_id(str(source_material_id))
                if depth_value:
                    facts.append(
                        RawFactCandidate(
                            entity_level=EntityLevel.SAMPLE,
                            fact_type_candidate="depth",
                            raw_field_name="source_material_id",
                            raw_value=depth_value,
                            source_locator=f"ncbi_biosample.{accession}.Attributes.source_material_id.depth",
                            entity_external_id=accession,
                            entity_label=sample.get("title"),
                            support_type=SupportType.DETERMINISTICALLY_DERIVED,
                        )
                    )
        facts.extend(self._recorded_by_facts(r.get("samples", [])))
        explicit_rep_facts = self._biological_rep_relation_facts_from_replicate_attribute(r.get("samples", []))
        facts.extend(explicit_rep_facts)
        if explicit_rep_facts:
            facts.append(
                RawFactCandidate(
                    entity_level=EntityLevel.STUDY,
                    fact_type_candidate="biological_rep_presence",
                    raw_field_name="replicate",
                    raw_value="TRUE",
                    source_locator="ncbi_biosample.Attributes.replicate",
                    support_type=SupportType.DETERMINISTICALLY_DERIVED,
                    confidence_metadata={
                        "replicate_detection_signal": "explicit_biosample_replicate_attribute",
                        "replicate_group_count": len({fact.raw_value for fact in explicit_rep_facts}),
                    },
                )
            )
        facts.extend(
            self._biological_rep_relation_facts(
                r.get("samples", []),
                excluded_accessions={fact.entity_external_id for fact in explicit_rep_facts if fact.entity_external_id},
            )
        )
        return facts

    @staticmethod
    def _recorded_by_facts(samples: list[dict]) -> list[RawFactCandidate]:
        recorded_by_values: list[str] = []
        locators: list[str] = []
        seen: set[str] = set()
        for sample in samples:
            accession = sample.get("accession")
            candidates = (
                ("Owner.Name", sample.get("owner", {}).get("name")),
                ("Submission.Name", sample.get("submission", {}).get("name")),
            )
            for field_name, value in candidates:
                value = _clean_text(value)
                if not accession or not value or value.casefold() in seen:
                    continue
                seen.add(value.casefold())
                recorded_by_values.append(value)
                locators.append(f"ncbi_biosample.{accession}.{field_name}")

        if not recorded_by_values:
            return []
        return [
            RawFactCandidate(
                entity_level=EntityLevel.STUDY,
                fact_type_candidate="recordedBy",
                raw_field_name="recordedBy",
                raw_value=" | ".join(recorded_by_values),
                source_locator=" | ".join(locators),
                confidence_metadata={"biosample_submitter_sample_count": len(locators)},
            )
        ]

    @staticmethod
    def _biological_rep_relation_facts_from_replicate_attribute(samples: list[dict]) -> list[RawFactCandidate]:
        accessions_by_replicate: dict[str, list[str]] = defaultdict(list)
        titles_by_accession: dict[str, str | None] = {}
        replicate_value_by_accession: dict[str, str] = {}
        for sample in samples:
            accession = sample.get("accession")
            if not accession:
                continue
            titles_by_accession[accession] = sample.get("title")
            replicate_value = _clean_text(sample.get("attributes", {}).get("replicate"))
            if not replicate_value:
                continue
            normalized_value = replicate_value.casefold()
            accessions_by_replicate[normalized_value].append(accession)
            replicate_value_by_accession[accession] = replicate_value

        facts: list[RawFactCandidate] = []
        for accessions in accessions_by_replicate.values():
            if len(accessions) < 2:
                continue
            group_members = sorted(accessions)
            relation = " | ".join(group_members)
            for accession in group_members:
                facts.append(
                    RawFactCandidate(
                        entity_level=EntityLevel.SAMPLE,
                        fact_type_candidate="biological_rep_relation",
                        raw_field_name="replicate",
                        raw_value=relation,
                        source_locator=f"ncbi_biosample.{accession}.Attributes.replicate",
                        entity_external_id=accession,
                        entity_label=titles_by_accession.get(accession),
                        support_type=SupportType.DETERMINISTICALLY_DERIVED,
                        confidence_metadata={
                            "replicate_detection_signal": "explicit_biosample_replicate_attribute",
                            "replicate_group_size": len(group_members),
                            "replicate_value": replicate_value_by_accession[accession],
                        },
                    )
                )
        return facts

    @staticmethod
    def _biological_rep_relation_facts(
        samples: list[dict], excluded_accessions: set[str] | None = None
    ) -> list[RawFactCandidate]:
        """Detects replicate groupings from each BioSample's sample_name
        attribute (falling back to its title when no sample_name attribute
        was submitted) via sources/replicate_grouping.py's shared,
        source-agnostic suffix-pattern detector, and emits one
        biological_rep_relation fact per grouped sample. The BioSample
        accession -- never the free-text name/title used only to detect the
        pattern -- is what ends up in raw_value, since exports/faire.py uses
        the accession as the exported samp_name for NCBI-sourced samples;
        pipe-joining the name/title text instead would reference values that
        never appear in that exported column."""
        excluded_accessions = excluded_accessions or set()
        name_and_field_by_accession: dict[str, tuple[str, str]] = {}
        titles_by_accession: dict[str, str | None] = {}
        for sample in samples:
            accession = sample.get("accession")
            if not accession or accession in excluded_accessions:
                continue
            titles_by_accession[accession] = sample.get("title")
            sample_name_attr = sample.get("attributes", {}).get("sample_name")
            if sample_name_attr:
                name_and_field_by_accession[accession] = (sample_name_attr, "sample_name")
            elif sample.get("title"):
                name_and_field_by_accession[accession] = (sample["title"], "title")

        replicate_group_by_accession = {
            member: group
            for group in detect_replicate_groups(
                {accession: name for accession, (name, _field) in name_and_field_by_accession.items()}
            )
            for member in group.members
        }

        facts: list[RawFactCandidate] = []
        for accession, group in replicate_group_by_accession.items():
            _, raw_field_name = name_and_field_by_accession[accession]
            facts.append(
                RawFactCandidate(
                    entity_level=EntityLevel.SAMPLE,
                    fact_type_candidate="biological_rep_relation",
                    raw_field_name=raw_field_name,
                    raw_value=" | ".join(group.members),
                    source_locator=f"ncbi_biosample.{accession}.biological_rep_relation",
                    entity_external_id=accession,
                    entity_label=titles_by_accession.get(accession),
                    support_type=SupportType.DETERMINISTICALLY_DERIVED,
                    confidence_metadata={
                        "replicate_detection_signal": group.signal.value,
                        "replicate_group_size": len(group.members),
                    },
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
