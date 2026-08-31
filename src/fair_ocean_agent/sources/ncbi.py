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
from typing import NamedTuple
import xml.etree.ElementTree as ET

from fair_ocean_agent.clock import utcnow
from fair_ocean_agent.database.enums import EntityLevel, IdentifierType, RelationshipType, ReviewStatus, SupportType
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


class UidResolution(NamedTuple):
    """Result of _esearch_verified_uid. `ambiguous=True` means esearch
    returned more than one UID and none of their own accessions (per
    esummary) matched what was searched for -- `uid` is then esearch's own
    first-ranked result used as a last-resort fallback, and
    `candidate_accessions` (uid -> that uid's own accession) is carried
    along so the caller can attach it to a review-flagged RawFact instead of
    trusting the fallback silently."""

    uid: str
    ambiguous: bool
    candidate_accessions: dict[str, str] | None


# esummary's own accession-bearing field differs per db -- confirmed live
# against real PRJNA529480/PRJEB73262/SAMN11268098 records.
_ESUMMARY_ACCESSION_FIELD = {"bioproject": "project_acc", "biosample": "accession"}


def _esummary_records(http, base_url: str, db: str, uids: list[str]) -> dict[str, dict]:
    if not uids:
        return {}
    payload, _ = http.get_json(
        f"{base_url}/esummary.fcgi", params={"db": db, "id": ",".join(uids), "retmode": "json"}
    )
    result = payload.get("result", {})
    return {uid: result[uid] for uid in result.get("uids", []) if uid in result}


def _esearch_verified_uid(http, base_url: str, db: str, term: str, expected_accession: str) -> UidResolution | None:
    """Replaces the old _esearch_first_uid, which blindly trusted esearch's
    first UID -- confirmed live to pick the WRONG BioProject for a real
    paper (esearch for "PRJNA529480" returns two UIDs; the first, 1356142,
    is actually PRJEB73262, a downstream MAG-only reanalysis project that
    just mentions the real accession in its own title; the correct UID,
    529480, is the second one). A single UID needs no extra call -- esearch
    already disambiguated for us. Multiple UIDs get resolved by fetching
    each candidate's own accession via esummary and matching it against
    what was actually searched for, rather than trusting esearch's
    relevance ranking."""
    payload, _ = http.get_json(f"{base_url}/esearch.fcgi", params={"db": db, "term": term, "retmode": "json"})
    ids = payload.get("esearchresult", {}).get("idlist", [])
    if not ids:
        return None
    if len(ids) == 1:
        return UidResolution(uid=ids[0], ambiguous=False, candidate_accessions=None)

    accession_field = _ESUMMARY_ACCESSION_FIELD[db]
    records = _esummary_records(http, base_url, db, ids)
    candidate_accessions = {uid: (records.get(uid, {}).get(accession_field) or "") for uid in ids}
    normalized_expected = expected_accession.strip().upper()
    for uid, accession in candidate_accessions.items():
        if accession.strip().upper() == normalized_expected:
            return UidResolution(uid=uid, ambiguous=False, candidate_accessions=None)

    logger.warning(
        "esearch for %s=%s returned %d UIDs and none of their own accessions "
        "(%s) matched -- falling back to the first UID (%s), flagged for review",
        db, term, len(ids), candidate_accessions, ids[0],
    )
    return UidResolution(uid=ids[0], ambiguous=True, candidate_accessions=candidate_accessions)


def _elink_ids(http, base_url: str, dbfrom: str, db: str, uid: str) -> list[str]:
    payload, _ = http.get_json(
        f"{base_url}/elink.fcgi", params={"dbfrom": dbfrom, "db": db, "id": uid, "retmode": "json"}
    )
    ids: list[str] = []
    seen: set[str] = set()
    for linkset in payload.get("linksets", []):
        for linksetdb in linkset.get("linksetdbs") or []:
            for linked_id in linksetdb.get("links") or []:
                if linked_id in seen:
                    continue
                seen.add(linked_id)
                ids.append(linked_id)
    return ids


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


# See extract_structured_facts' own MAG-safe-attribute loop for why this
# exists: a curated allowlist of attribute names that describe the real
# physical environment a MAG was assembled from, not the assembled genome
# bin itself.
_MAG_SAFE_ENVIRONMENTAL_ATTRIBUTES = frozenset(
    {
        "geo_loc_name",
        "lat_lon",
        "collection_date",
        "depth",
        "elev",
        "env_broad_scale",
        "env_local_scale",
        "env_medium",
        "isolation_source",
    }
)

# A real MIMAG.sediment record (SAMN12415826) carries its own "derived-
# from" attribute pointing back at the real environmental BioSample it was
# assembled from -- not an environmental attribute itself (so it's never
# added to _MAG_SAFE_ENVIRONMENTAL_ATTRIBUTES above), but FAIRe's own
# sample_derived_from field wants exactly this: "the samp_name of the
# original (or parent) sample from which the current sample was derived".
# The real attribute value is a full sentence ("This BioSample is a
# metagenomic assembly obtained from the marine sediment metagenome
# BioSample: SAMN11268106"), not a bare accession, so the accession is
# extracted out of it rather than storing the whole sentence as the
# "parent sample name" FAIRe wants.
_DERIVED_FROM_ACCESSION_RE = re.compile(r"\bSAM[NED]\d+\b")


def _derive_sample_derived_from(value: str) -> str | None:
    match = _DERIVED_FROM_ACCESSION_RE.search(value)
    if match:
        return match.group(0)
    stripped = value.strip()
    return stripped or None


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
_FILTER_ACTIVE_RE = re.compile(
    r"\b(?:Sterivex|active(?:ly)?\s+filter|pumped?|pumping|pump|peristaltic|vacuum|"
    r"suction|pressure|overpressure|pressuri[sz]ed|compressed\s+air|forced\s+through|"
    r"flow\s*rate|flowrate)\b",
    re.IGNORECASE,
)
_FILTER_PASSIVE_RE = re.compile(r"\bpassive\b|\bsubmerged\b|\bgravity\b", re.IGNORECASE)
_CONTEXT_WINDOW = 40
_GENERIC_BIOSAMPLE_TITLE_RE = re.compile(
    r"^(?:"
    r"MIMARKS|MIMS|MIGS|MIMAG|MIUVIG|MISAG|"
    r"Environmental|Metagenome|Metagenomic"
    r")\b.*\bsample\b.*$|^(?:Bio)?Sample\s+\d+$",
    re.IGNORECASE,
)
_SOURCE_MATERIAL_SAMPLE_NAME_RE = re.compile(
    r"^[A-Za-z0-9]+[-_][A-Za-z]+\d+[-_]\d+$"
)


def _source_material_id_sample_name(value: str | None) -> str | None:
    cleaned = _clean_text(value)
    if not cleaned:
        return None
    normalized = cleaned.replace("-", "_")
    return normalized if _SOURCE_MATERIAL_SAMPLE_NAME_RE.match(cleaned) else None


def _normalize_sample_name(value: str | None) -> str | None:
    """Normalizes a BioSample's own "Sample name" (the <Ids> element, see
    NcbiBioSampleAdapter.fetch_record's own comment on where this comes
    from) the same way _source_material_id_sample_name already normalizes
    source_material_id -- all "-" become "_" -- so two BioSamples from the
    same physical replicate series never fail to group together purely
    because one submitter's convention used a hyphen and detect_replicate_
    groups' own suffix-pattern matching is separator-sensitive. Unlike
    source_material_id, sample_name is a dedicated, reliably-populated
    BioSample element (not a loosely-typed Attribute), so no shape
    validation is applied here -- confirmed live, a real sample_name
    ("GS16_GC05_55cm") doesn't match _SOURCE_MATERIAL_SAMPLE_NAME_RE's own
    stricter source_material_id-specific shape at all."""
    cleaned = _clean_text(value)
    if not cleaned:
        return None
    return cleaned.replace("-", "_")


def _sample_category_from_title_or_name(sample: dict) -> tuple[str, str] | None:
    """Return a useful BioSample display name for samp_category.

    source_material_id (e.g. GS16-GC05-20 -> GS16_GC05_20) is preferred
    when present -- an explicit user instruction ("source material id...
    should stay the default"). A submitted sample_name (from the
    BioSample's own <Ids> element, e.g. "GS16_GC05_55cm") is the fallback
    when a record has no source_material_id attribute at all -- a real
    live gap (SAMN29179945 carries a Sample name but no source_material_id
    attribute whatsoever). BioSample titles sometimes carry real labels
    ("LM 7"), but many only say boilerplate like "MIMS Environmental
    sample"; those are not useful as a category/name log, so title is
    always the last resort.
    """
    attributes = sample.get("attributes", {})
    for field_name, value in (
        ("source_material_id", _source_material_id_sample_name(_get_attribute(attributes, "source_material_id"))),
        ("sample_name", _normalize_sample_name(_get_attribute(attributes, "sample_name"))),
        ("title", sample.get("title")),
    ):
        cleaned = _clean_text(str(value) if value is not None else None)
        if not cleaned or _GENERIC_BIOSAMPLE_TITLE_RE.match(cleaned):
            continue
        return field_name, cleaned
    return None


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
    elif any(field in derived for field in ("size_frac", "filter_diameter", "filter_material", "filter_name", "prefilter_material")):
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


def _normalize_attribute_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _get_attribute(attributes: dict, name: str) -> str | None:
    """BioSample `attribute_name` spelling is submitter-controlled and
    varies (hyphens vs underscores vs spaces) -- confirmed live that the
    exact same MIxS attribute appears as both "source_material_id" and
    "source-material-id" across real submissions. Looks up by exact key
    first (cheap, the common case), falling back to a normalized
    (case/separator-insensitive) match across all attribute keys rather
    than silently missing real data stored under a differently-spelled
    key -- this is what caused depth derivation below to never fire at
    all for a real dataset that happened to use the hyphenated spelling."""
    if name in attributes:
        return attributes[name]
    target = _normalize_attribute_name(name)
    for key, value in attributes.items():
        if _normalize_attribute_name(key) == target:
            return value
    return None


def _canonical_biosample_attribute_name(name: str) -> str:
    normalized = _normalize_attribute_name(name)
    if normalized == "geographiclocation":
        return "geo_loc_name"
    if normalized == "isolationsource":
        return "isolation_source"
    return name


def _uid_verification_fact(
    *,
    bioproject_accession: str,
    uid_resolution_ambiguous: bool,
    uid_resolution_candidates: dict[str, str] | None,
    reverse_elink_verified: bool | None = None,
    source_locator: str,
) -> RawFactCandidate | None:
    """Emits a PROJECT-level, review-flagged RawFact when this BioProject's
    UID resolution wasn't fully trustworthy -- either the esearch/esummary
    cross-check (_esearch_verified_uid) had to fall back to an unconfirmed
    UID, or (BioSample adapter only) the independent biosample->bioproject
    reverse-elink signal disagreed. Both signals are recorded together in
    confidence_metadata even when only one fired, so a reviewer sees the
    full picture rather than just whichever check happened to trip.
    Returns None when both signals are clean -- most BioProjects only have
    one esearch UID and never need this at all."""
    if not uid_resolution_ambiguous and reverse_elink_verified is not False:
        return None
    return RawFactCandidate(
        entity_level=EntityLevel.PROJECT,
        fact_type_candidate="ambiguous_uid_resolution",
        raw_field_name="bioproject_uid_verification",
        raw_value=f"bioproject_accession={bioproject_accession}",
        source_locator=source_locator,
        review_status=ReviewStatus.NEEDS_REVIEW.value,
        confidence_metadata={
            "esearch_uid_ambiguous": uid_resolution_ambiguous,
            "esearch_uid_candidates": uid_resolution_candidates,
            "reverse_elink_verified": reverse_elink_verified,
        },
    )


class NcbiBioProjectAdapter(SourceAdapter):
    name = "ncbi_bioproject"

    def fetch_record(self, identifier: str) -> SourceRecord:
        resolution = _esearch_verified_uid(
            self.http, self.config.base_url, "bioproject", identifier, expected_accession=identifier
        )
        if resolution is None:
            raise SourceRecordNotFoundError(f"No BioProject UID found for {identifier}")
        uid = resolution.uid

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
            "uid_resolution_ambiguous": resolution.ambiguous,
            "uid_resolution_candidates": resolution.candidate_accessions,
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

        verification_fact = _uid_verification_fact(
            bioproject_accession=r.get("accession") or "",
            uid_resolution_ambiguous=bool(r.get("uid_resolution_ambiguous")),
            uid_resolution_candidates=r.get("uid_resolution_candidates"),
            source_locator="ncbi_bioproject.esearch.idlist",
        )
        if verification_fact is not None:
            facts.append(verification_fact)
        return facts


class NcbiBioSampleAdapter(SourceAdapter):
    """Given a BioProject accession, discovers and fetches every linked
    BioSample's full attribute record. One fetch_record() call = one
    project's worth of samples (see module docstring on Source-row
    granularity in workflow/handlers.py)."""

    name = "ncbi_biosample"

    def fetch_record(self, identifier: str) -> SourceRecord:
        resolution = _esearch_verified_uid(
            self.http, self.config.base_url, "bioproject", identifier, expected_accession=identifier
        )
        if resolution is None:
            raise SourceRecordNotFoundError(f"No BioProject UID found for {identifier}")
        project_uid = resolution.uid

        sample_uids = _elink_ids(self.http, self.config.base_url, "bioproject", "biosample", project_uid)
        if not sample_uids:
            raise SourceRecordNotFoundError(f"No linked BioSamples for BioProject {identifier}")

        # Second, independent signal: this is the check that would have
        # caught PRJEB73262 directly. One extra call per project (using a
        # single already-fetched sample's UID), not per sample -- confirming
        # that BioSample actually links back to the same BioProject UID we
        # used to fetch it, rather than trusting esearch/esummary alone.
        reverse_bioproject_uids = _elink_ids(
            self.http, self.config.base_url, "biosample", "bioproject", sample_uids[0]
        )
        reverse_elink_verified = project_uid in reverse_bioproject_uids

        total_linked_samples = len(sample_uids)
        truncated = total_linked_samples > MAX_SAMPLES_PER_PROJECT
        if truncated:
            logger.warning(
                "BioProject %s has %d linked BioSamples; processing only the first %d "
                "(MAX_SAMPLES_PER_PROJECT) -- not a silent drop, see raw_facts for the count.",
                identifier, total_linked_samples, MAX_SAMPLES_PER_PROJECT,
            )
        sample_uids = sample_uids[:MAX_SAMPLES_PER_PROJECT]

        samples, combined_xml_for_hash = self._fetch_and_parse_biosamples(sample_uids)

        raw = {
            "bioproject_accession": identifier,
            "total_linked_samples": total_linked_samples,
            "truncated": truncated,
            "samples": samples,
            "uid_resolution_ambiguous": resolution.ambiguous,
            "uid_resolution_candidates": resolution.candidate_accessions,
            "reverse_elink_verified": reverse_elink_verified,
        }

        return SourceRecord(
            source_name=self.name,
            external_identifier=identifier,
            url=None,
            raw=raw,
            retrieved_at=utcnow(),
            content_hash=_hash_text("".join(combined_xml_for_hash)),
        )

    def fetch_record_by_accessions(self, bioproject_accession: str, accessions: list[str]) -> SourceRecord:
        """Fallback for when fetch_record's own bioproject<->biosample
        elink cross-reference comes back empty even though real, already-
        known BioSample accessions exist for this study (typically
        discovered independently via ENA's own sample_accession
        cross-reference, not NCBI's). Confirmed live (2026-08-31, a real
        study -- PRJNA762627): NCBI's elink index has no bioproject<->
        biosample link at all for this project, yet its real BioSamples
        (e.g. SAMN21399411) are perfectly fetchable directly by accession
        -- efetch accepts a real accession string exactly like a numeric
        UID, no separate esearch/UID-resolution step needed. Skips the
        reverse-elink cross-check fetch_record does (nothing to reverse-
        check against when we never resolved a project UID in the first
        place) -- already knowing these specific accessions belong to
        this study via an independent source is itself the trust signal.
        Raises SourceRecordNotFoundError if none of the given accessions
        are real, live BioSample records."""
        truncated = len(accessions) > MAX_SAMPLES_PER_PROJECT
        if truncated:
            logger.warning(
                "%d known BioSample accessions for BioProject %s; processing only the first %d "
                "(MAX_SAMPLES_PER_PROJECT)",
                len(accessions), bioproject_accession, MAX_SAMPLES_PER_PROJECT,
            )
        accessions = accessions[:MAX_SAMPLES_PER_PROJECT]

        samples, combined_xml_for_hash = self._fetch_and_parse_biosamples(accessions)
        if not samples:
            raise SourceRecordNotFoundError(
                f"None of {len(accessions)} known BioSample accession(s) resolved for BioProject {bioproject_accession}"
            )

        raw = {
            "bioproject_accession": bioproject_accession,
            "total_linked_samples": len(accessions),
            "truncated": truncated,
            "samples": samples,
            "fetched_via": "known_biosample_accessions_fallback",
        }

        return SourceRecord(
            source_name=self.name,
            external_identifier=bioproject_accession,
            url=None,
            raw=raw,
            retrieved_at=utcnow(),
            content_hash=_hash_text("".join(combined_xml_for_hash)),
        )

    def _fetch_and_parse_biosamples(self, ids: list[str]) -> tuple[list[dict], list[str]]:
        """Shared by fetch_record (numeric UIDs from elink) and
        fetch_record_by_accessions (real accession strings) -- efetch's
        own id= param accepts either interchangeably, confirmed live."""
        samples: list[dict] = []
        combined_xml_for_hash: list[str] = []
        for i in range(0, len(ids), EFETCH_BATCH_SIZE):
            batch = ids[i : i + EFETCH_BATCH_SIZE]
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
                submitted = _clean_text(
                    bs.get("submission_date")
                    or bs.get("submitted")
                    or bs.get("submission")
                )
                organism_el = bs.find("Description/Organism")
                organism = dict(organism_el.attrib) if organism_el is not None else {}
                owner_name_el = bs.find("Owner/Name")
                owner_name = _clean_text(owner_name_el.text if owner_name_el is not None else None)
                # Owner/Name is the submitting INSTITUTION's name (e.g.
                # "San Francisco Estuary Institute"), confirmed against
                # real cached BioSample XML -- never a person. The real
                # per-sample submitter PERSON lives one level deeper, at
                # Owner/Contacts/Contact/Name/First+Last (also confirmed
                # against the same real record). Only the first <Contact>
                # is read, matching this whole block's existing
                # single-element convention (a BioSample practically
                # always carries exactly one).
                contact_name_el = bs.find("Owner/Contacts/Contact/Name")
                contact_first = _clean_text(contact_name_el.findtext("First") if contact_name_el is not None else None)
                contact_last = _clean_text(contact_name_el.findtext("Last") if contact_name_el is not None else None)
                contact_name = " ".join(part for part in (contact_first, contact_last) if part)
                attributes = {
                    attr.get("attribute_name"): attr.text
                    for attr in bs.findall("Attributes/Attribute")
                    if attr.get("attribute_name")
                }
                # <Ids><Id db_label="Sample name">...</Id></Ids> is a
                # standard BioSample element every record carries (never an
                # Attribute), previously never parsed at all -- confirmed
                # live (SAMN29179945): its "GS16_GC05_55cm" Sample name
                # exists only here, not as any Attributes/Attribute, while
                # this same record has no source_material_id attribute at
                # all. Only fills the gap: a submitter that redundantly
                # also declares a real "sample_name" Attribute keeps that
                # value untouched.
                if "sample_name" not in attributes:
                    sample_name_id_el = bs.find('Ids/Id[@db_label="Sample name"]')
                    if sample_name_id_el is not None and sample_name_id_el.text:
                        attributes["sample_name"] = sample_name_id_el.text
                samples.append(
                    {
                        "accession": accession,
                        "title": bs.findtext("Description/Title"),
                        "submitted": submitted,
                        "package": bs.get("package"),
                        "model": bs.findtext("Models/Model"),
                        "organism": {key: value for key, value in organism.items() if value},
                        "attributes": attributes,
                        "owner": {
                            key: value
                            for key, value in {
                                "name": owner_name,
                                "abbreviation": (
                                    owner_name_el.get("abbreviation") if owner_name_el is not None else None
                                ),
                                "contact_name": contact_name,
                            }.items()
                            if value
                        },
                    }
                )
        return samples, combined_xml_for_hash

    def search(self, query: SearchQuery) -> SearchPage:
        # BioSample has no free-text "search" use case in this pipeline --
        # it's always resolved via a BioProject accession (fetch_record).
        return SearchPage(records=[])

    def extract_structured_facts(self, record: SourceRecord) -> list[RawFactCandidate]:
        r = record.raw
        facts: list[RawFactCandidate] = []

        verification_fact = _uid_verification_fact(
            bioproject_accession=r.get("bioproject_accession") or "",
            uid_resolution_ambiguous=bool(r.get("uid_resolution_ambiguous")),
            uid_resolution_candidates=r.get("uid_resolution_candidates"),
            reverse_elink_verified=r.get("reverse_elink_verified"),
            source_locator="ncbi_biosample.uid_verification",
        )
        if verification_fact is not None:
            facts.append(verification_fact)

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

        # MAG (metagenome-assembled-genome) BioSamples are excluded from
        # every per-sample-derived signal below, not just the attribute
        # loop immediately following: a real live audit found the
        # replicate-relation tiers further down used to read straight from
        # r.get("samples", []) (the UNFILTERED list), so MAG BioSamples --
        # which never get their own lat_lon/collection_date/depth facts
        # persisted, since the loop below skips them -- could still get
        # silently grouped as "biological replicates" of each other by the
        # metadata-match tier purely because they share the one real
        # sediment sample's coordinates/date/depth they were assembled
        # from. A MAG is a computational construct derived from one
        # sample's sequencing data, not a biological replicate.
        non_mag_samples = [sample for sample in r.get("samples", []) if not _is_mag_biosample(sample)]

        for sample in non_mag_samples:
            accession = sample["accession"]
            normalized_attrs: dict[str, str] = {}
            sample_category = _sample_category_from_title_or_name(sample)
            if sample_category:
                raw_field_name, raw_value = sample_category
                facts.append(
                    RawFactCandidate(
                        entity_level=EntityLevel.SAMPLE,
                        fact_type_candidate="samp_category",
                        raw_field_name=raw_field_name,
                        raw_value=raw_value,
                        source_locator=f"ncbi_biosample.{accession}.{raw_field_name}",
                        entity_external_id=accession,
                        entity_label=sample.get("title"),
                    )
                )
            submitted = _clean_text(sample.get("submitted"))
            if submitted:
                facts.append(
                    RawFactCandidate(
                        entity_level=EntityLevel.SAMPLE,
                        fact_type_candidate="eventDate_submitted",
                        raw_field_name="submission_date",
                        raw_value=submitted,
                        source_locator=f"ncbi_biosample.{accession}.submission_date",
                        entity_external_id=accession,
                        entity_label=sample.get("title"),
                    )
                )
            for attr_name, attr_value in sample.get("attributes", {}).items():
                if attr_value in (None, ""):
                    continue
                normalized_attrs[attr_name] = str(attr_value)
                canonical_attr_name = _canonical_biosample_attribute_name(attr_name)
                if canonical_attr_name != attr_name:
                    normalized_attrs[canonical_attr_name] = str(attr_value)
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
            source_material_id = _get_attribute(sample.get("attributes", {}), "source_material_id")
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
        # A MAG record is still excluded from every per-sample-derived
        # signal above and below (entity_label/replicate grouping/generic
        # attribute passthrough all stay scoped to non_mag_samples, see its
        # own comment) -- but the premise that a MAG record "never carries
        # the real sample's own attributes themselves" doesn't hold for
        # every submitter's convention: confirmed live, a real
        # MIMAG.sediment-packaged record (SAMN42764696) directly carries
        # geo_loc_name/lat_lon/collection_date/depth/elev/env_*/
        # isolation_source, identical in spirit to the original sample it
        # was assembled from. Deliberately a narrow allowlist, not the
        # generic attribute passthrough: excludes the MAG's own
        # assembly-specific attributes (assembly software, completeness/
        # contamination score, binning software, the assembled genome's
        # own organism/taxonomy, ...), which describe the bin, not the
        # environment it came from. Writing these here (not into
        # non_mag_samples) can't reintroduce the replicate-grouping bug
        # the comment above describes, since all three replicate tiers
        # only ever read non_mag_samples, never these facts.
        for sample in r.get("samples", []):
            if not _is_mag_biosample(sample):
                continue
            accession = sample["accession"]
            for attr_name, attr_value in sample.get("attributes", {}).items():
                if attr_value in (None, ""):
                    continue
                attr_name = _canonical_biosample_attribute_name(attr_name)
                if attr_name not in _MAG_SAFE_ENVIRONMENTAL_ATTRIBUTES:
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
            # _get_attribute already matches "derived-from"/"derived_from"
            # interchangeably (case/separator-insensitive fallback).
            derived_from = _get_attribute(sample.get("attributes", {}), "derived_from")
            if derived_from:
                parent_sample = _derive_sample_derived_from(str(derived_from))
                if parent_sample:
                    facts.append(
                        RawFactCandidate(
                            entity_level=EntityLevel.SAMPLE,
                            fact_type_candidate="sample_derived_from",
                            raw_field_name="derived_from",
                            raw_value=parent_sample,
                            source_locator=f"ncbi_biosample.{accession}.Attributes.derived_from",
                            entity_external_id=accession,
                            entity_label=sample.get("title"),
                            support_type=SupportType.DETERMINISTICALLY_DERIVED,
                        )
                    )

        facts.extend(self._recorded_by_facts(non_mag_samples))
        # Three replicate-grouping signals, most confident first, each
        # scoped to only the samples the earlier signal(s) left ungrouped:
        # (1) an explicit BioSample `replicate` attribute, (2) a sample-
        # name/title suffix pattern (sources/replicate_grouping.py, including
        # exact-prefix title shapes like "LM 6"/"LM_7"), (3) an
        # explicit user request: samples sharing the same coordinates,
        # collection date, and depth (and the same assay when one is
        # actually reported per-sample, otherwise assumed the same) are
        # grouped as biological replicates. All three read non_mag_samples
        # (not the raw list) -- see non_mag_samples's own comment above.
        explicit_rep_facts = self._biological_rep_relation_facts_from_replicate_attribute(non_mag_samples)
        facts.extend(explicit_rep_facts)
        name_pattern_excluded = {fact.entity_external_id for fact in explicit_rep_facts if fact.entity_external_id}
        name_pattern_rep_facts = self._biological_rep_relation_facts(
            non_mag_samples, excluded_accessions=name_pattern_excluded
        )
        facts.extend(name_pattern_rep_facts)
        metadata_excluded = name_pattern_excluded | {
            fact.entity_external_id for fact in name_pattern_rep_facts if fact.entity_external_id
        }
        metadata_match_rep_facts = self._biological_rep_relation_facts_from_metadata_match(
            non_mag_samples, excluded_accessions=metadata_excluded
        )
        facts.extend(metadata_match_rep_facts)
        # The study-level projectMetadata.biological_rep value itself (a
        # replicate count/range, e.g. "2-4", or "0") is no longer computed
        # here -- per an explicit user request, it's derived once at
        # map-time from ALL of a study's biological_rep_relation facts
        # (mapping/faire.py::_apply_biological_rep_from_relations), across
        # every source (this adapter, supplement_parsing.py, ...), not just
        # this one adapter's own local view of the study's samples.
        return facts

    @staticmethod
    def _recorded_by_facts(samples: list[dict]) -> list[RawFactCandidate]:
        """The real per-sample submitter PERSON, from each BioSample's own
        Owner/Contacts/Contact/Name -- NOT Owner/Name, which real BioSample
        XML confirms is the submitting institution's own name (e.g. "San
        Francisco Estuary Institute"), not a person, and was previously
        (incorrectly) piped into this same field alongside real names."""
        recorded_by_values: list[str] = []
        locators: list[str] = []
        seen: set[str] = set()
        for sample in samples:
            accession = sample.get("accession")
            value = _clean_text(sample.get("owner", {}).get("contact_name"))
            if not accession or not value or value.casefold() in seen:
                continue
            seen.add(value.casefold())
            recorded_by_values.append(value)
            locators.append(f"ncbi_biosample.{accession}.Owner.Contacts.Contact.Name")

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
        """Detects replicate groupings from each BioSample's own name
        signal via sources/replicate_grouping.py's shared, source-agnostic
        suffix-pattern detector, and emits one biological_rep_relation
        fact per grouped sample. source_material_id is the preferred
        signal when present (an explicit user instruction: "source
        material id... should stay the default"), normalized the same way
        as sample_name below (all "-" become "_") so two replicates never
        fail to group together purely because of a hyphen-vs-underscore
        naming difference. Falls back to the BioSample's own submitted
        sample_name when source_material_id is absent (a real live gap:
        SAMN29179945 has a Sample name but no source_material_id
        attribute at all), then to its title as the last resort. The
        BioSample accession -- never the free-text name/title used only
        to detect the pattern -- is what ends up in raw_value, since
        exports/faire.py uses the accession as the exported samp_name for
        NCBI-sourced samples; pipe-joining the name/title text instead
        would reference values that never appear in that exported
        column."""
        excluded_accessions = excluded_accessions or set()
        name_and_field_by_accession: dict[str, tuple[str, str]] = {}
        titles_by_accession: dict[str, str | None] = {}
        for sample in samples:
            accession = sample.get("accession")
            if not accession or accession in excluded_accessions:
                continue
            titles_by_accession[accession] = sample.get("title")
            attributes = sample.get("attributes", {})
            source_material_name = _source_material_id_sample_name(_get_attribute(attributes, "source_material_id"))
            if source_material_name:
                name_and_field_by_accession[accession] = (source_material_name, "source_material_id")
            else:
                sample_name_attr = _normalize_sample_name(_get_attribute(attributes, "sample_name"))
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

    @staticmethod
    def _effective_depth(sample: dict) -> str | None:
        attributes = sample.get("attributes", {})
        depth = _clean_text(attributes.get("depth") or "")
        if depth:
            return depth
        source_material_id = _get_attribute(attributes, "source_material_id")
        if source_material_id:
            return _derive_depth_from_source_material_id(str(source_material_id))
        return None

    @staticmethod
    def _biological_rep_relation_facts_from_metadata_match(
        samples: list[dict], excluded_accessions: set[str] | None = None
    ) -> list[RawFactCandidate]:
        """Third, lowest-priority replicate-grouping signal (after an
        explicit BioSample `replicate` attribute, then a sample-name
        suffix pattern) -- an explicit user request: BioSamples sharing
        the same coordinates, collection date, and depth are grouped as
        biological replicates. The assay/marker is also required to match
        when at least one of the two samples actually reports one (a real
        BioSample attribute, rare but possible); when NEITHER reports one
        -- true for essentially all real BioSamples, which have no
        per-sample assay attribute at all -- it's assumed the same, per
        the user's own explicit instruction, since one paper/BioProject
        overwhelmingly uses one consistent assay across all its samples.
        Never groups a sample missing lat_lon/collection_date/depth: an
        unknown value is never treated as "the same" as another unknown
        or known value (this pipeline's standing "never guess" discipline
        applied here too)."""
        excluded_accessions = excluded_accessions or set()
        titles_by_accession: dict[str, str | None] = {}
        key_by_accession: dict[str, tuple[str, str, str, str | None]] = {}
        for sample in samples:
            accession = sample.get("accession")
            if not accession or accession in excluded_accessions:
                continue
            attributes = sample.get("attributes", {})
            lat_lon = _clean_text(_get_attribute(attributes, "lat_lon") or "")
            collection_date = _clean_text(attributes.get("collection_date") or "")
            depth = NcbiBioSampleAdapter._effective_depth(sample) or ""
            if not lat_lon or not collection_date or not depth:
                continue
            assay = _clean_text(_get_attribute(attributes, "assay") or _get_attribute(attributes, "assay_type") or "")
            titles_by_accession[accession] = sample.get("title")
            key_by_accession[accession] = (
                lat_lon.casefold(),
                collection_date.casefold(),
                depth.casefold(),
                assay.casefold() if assay else None,
            )

        groups: dict[tuple, list[str]] = defaultdict(list)
        for accession, key in key_by_accession.items():
            groups[key].append(accession)

        facts: list[RawFactCandidate] = []
        for accessions in groups.values():
            if len(accessions) < 2:
                continue
            group_members = sorted(accessions)
            relation = " | ".join(group_members)
            for accession in group_members:
                facts.append(
                    RawFactCandidate(
                        entity_level=EntityLevel.SAMPLE,
                        fact_type_candidate="biological_rep_relation",
                        raw_field_name="lat_lon+collection_date+depth",
                        raw_value=relation,
                        source_locator=f"ncbi_biosample.{accession}.biological_rep_relation.metadata_match",
                        entity_external_id=accession,
                        entity_label=titles_by_accession.get(accession),
                        support_type=SupportType.DETERMINISTICALLY_DERIVED,
                        confidence_metadata={
                            "replicate_detection_signal": "biosample_metadata_match",
                            "replicate_group_size": len(group_members),
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
