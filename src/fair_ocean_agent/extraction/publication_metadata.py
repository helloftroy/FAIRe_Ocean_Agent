"""Deterministic (no-LLM) extraction of project-level metadata directly
from a publication's own structured sources -- never the LLM checklist.

Per an explicit user review of a NOAA/SEUS-MBON FAIRe checklist, every
field this module produces was marked "No LLM": structured sources first,
falling back to deterministic parsing of the paper's own text, since these
are all more reliably sourced this way than by asking a model to read
prose. Two distinct techniques, kept in separate functions rather than one
grab-bag extractor:

- **JATS full-text XML tree structure** (`extract_from_jats_permissions`,
  `extract_from_jats_authors`) -- parsed with `xml.etree.ElementTree`
  directly, not flattened-then-regexed, because `<permissions>`/
  `<contrib-group>` are real structured elements, not prose needing
  pattern-matching. Confirmed live against a real article
  (DOI 10.7717/peerj.333, PMC3994630) that Crossref's own record can be
  missing `license` even though the journal's JATS XML has the answer
  structurally -- Crossref alone is not sufficient for these fields.
  `<contrib-group>` filtering is deliberately strict: a real article can
  have a separate `contrib-type="editor"` group alongside the author one
  (confirmed live in the same test article), which must never be
  conflated with `recordedBy`.
- **Flat-text regex fallback** (`extract_code_repo_from_text`) for
  `code_repo` only, reusing `discovery/text_identifiers.xml_to_text()`
  rather than reimplementing XML flattening.

`format_bibliographic_citation` is a pure formatter, not extraction: every
piece of a citation (title/authors/year/journal/DOI) already exists as a
structured Crossref fact (`sources/crossref.py`) by the time this runs.

Every fact_type_candidate here is the literal FAIRe field spelling
(`license`, `rightsHolder`, ...), matching this codebase's structured-
adapter convention ("fact_type_candidate is literally whatever attribute
name a real record carries", per `mapping/rules.py`'s own docstring) --
not the LLM taxonomy's standard-agnostic native-name indirection
(`extraction/faire_fields.py`), since none of this is LLM output that
needs protecting from vocabulary coupling.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from fair_ocean_agent.database.enums import EntityLevel, SupportType
from fair_ocean_agent.discovery.text_identifiers import xml_to_text
from fair_ocean_agent.sources.base import RawFactCandidate

_XLINK_HREF = "{http://www.w3.org/1999/xlink}href"

_CODE_REPO_PATTERN = re.compile(
    r"https?://(?:www\.)?(?:github|gitlab|bitbucket)\.(?:com|org)/[^\s<>()\[\]{}\"']+",
    re.IGNORECASE,
)


def _candidate(
    fact_type: str,
    raw_value: str,
    locator: str,
    support_type: SupportType = SupportType.STRUCTURED_SOURCE,
) -> RawFactCandidate:
    return RawFactCandidate(
        entity_level=EntityLevel.STUDY,
        fact_type_candidate=fact_type,
        raw_field_name=fact_type,
        raw_value=raw_value,
        source_locator=locator,
        support_type=support_type,
    )


def extract_from_jats_permissions(xml: str, *, locator_prefix: str) -> list[RawFactCandidate]:
    """`license`/`accessRights`/`rightsHolder` from JATS `<permissions>`."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return []
    permissions = root.find(".//permissions")
    if permissions is None:
        return []

    facts: list[RawFactCandidate] = []
    license_el = permissions.find("license")
    if license_el is not None:
        href = license_el.get(_XLINK_HREF)
        if href:
            facts.append(_candidate("license", href, f"{locator_prefix}:permissions/license/@xlink:href"))
        license_type = license_el.get("license-type")
        if license_type:
            access = "open access" if license_type == "open-access" else license_type
            facts.append(_candidate("accessRights", access, f"{locator_prefix}:permissions/license/@license-type"))

    holder_el = permissions.find("copyright-holder")
    holder = (holder_el.text or "").strip() if holder_el is not None else ""
    if holder:
        facts.append(_candidate("rightsHolder", holder, f"{locator_prefix}:permissions/copyright-holder"))
    return facts


def extract_from_jats_authors(xml: str, *, locator_prefix: str) -> list[RawFactCandidate]:
    """`recordedBy`/`recordedByID`/`project_contact` from JATS
    `<contrib-group>`, filtered to `contrib-type="author"` only -- never
    editors or other contributor roles that share the same element shape."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return []

    names: list[str] = []
    orcids: list[str] = []
    contact_email: str | None = None
    for contrib_group in root.iter("contrib-group"):
        for contrib in contrib_group.findall("contrib"):
            if contrib.get("contrib-type") != "author":
                continue
            name_el = contrib.find("name")
            if name_el is not None:
                given = (name_el.findtext("given-names") or "").strip()
                surname = (name_el.findtext("surname") or "").strip()
                full_name = " ".join(part for part in (given, surname) if part)
                if full_name:
                    names.append(full_name)
            for contrib_id in contrib.findall("contrib-id"):
                if contrib_id.get("contrib-id-type") == "orcid":
                    orcid = (contrib_id.text or "").strip()
                    if orcid:
                        orcids.append(orcid)
            if contact_email is None and contrib.get("corresp") == "yes":
                email_el = contrib.find("email")
                email = (email_el.text or "").strip() if email_el is not None else ""
                if email:
                    contact_email = email

    facts: list[RawFactCandidate] = []
    if names:
        facts.append(
            _candidate(
                "recordedBy",
                " | ".join(names),
                f"{locator_prefix}:contrib-group/contrib[@contrib-type='author']",
            )
        )
    if orcids:
        facts.append(
            _candidate(
                "recordedByID",
                " | ".join(orcids),
                f"{locator_prefix}:contrib-group/contrib/contrib-id[@contrib-id-type='orcid']",
            )
        )
    if contact_email:
        facts.append(
            _candidate(
                "project_contact",
                contact_email,
                f"{locator_prefix}:contrib-group/contrib[@corresp='yes']/email",
            )
        )
    return facts


def extract_code_repo_from_text(xml: str, *, locator_prefix: str) -> list[RawFactCandidate]:
    """`code_repo` via a flat-text URL regex -- the one field in this
    module without a reliable structured JATS element to read instead."""
    text = xml_to_text(xml)
    match = _CODE_REPO_PATTERN.search(text)
    if not match:
        return []
    url = match.group(0).rstrip(".,;:)")
    return [
        _candidate(
            "code_repo",
            url,
            f"{locator_prefix}:fulltext_regex",
            support_type=SupportType.DETERMINISTICALLY_DERIVED,
        )
    ]


def _crossref_authors_short(crossref_raw: dict) -> str:
    parts = []
    for author in crossref_raw.get("author") or []:
        given = (author.get("given") or "").strip()
        family = (author.get("family") or "").strip()
        if not family:
            continue
        initials = "".join(part[0] for part in given.split() if part)
        parts.append(f"{family} {initials}".strip() if initials else family)
    return ", ".join(parts)


def format_bibliographic_citation(crossref_raw: dict | None, *, locator_prefix: str) -> list[RawFactCandidate]:
    """Composes `bibliographicCitation` from Crossref's own bibliographic
    fields (title/authors/year/journal/DOI) -- a formatting step over
    already-structured facts, not extraction."""
    if not crossref_raw:
        return []
    titles = crossref_raw.get("title") or []
    title = titles[0] if titles else None
    if not title:
        return []

    authors = _crossref_authors_short(crossref_raw)
    date_parts = (
        (crossref_raw.get("published") or {}).get("date-parts")
        or (crossref_raw.get("published-print") or {}).get("date-parts")
        or (crossref_raw.get("published-online") or {}).get("date-parts")
    )
    year = date_parts[0][0] if date_parts and date_parts[0] else None
    containers = crossref_raw.get("container-title") or []
    journal = containers[0] if containers else None
    volume = crossref_raw.get("volume")
    page = crossref_raw.get("page")
    doi = crossref_raw.get("DOI")

    parts: list[str] = []
    if authors:
        parts.append(f"{authors}.")
    if year:
        parts.append(f"({year}).")
    parts.append(f"{title}.")
    if journal:
        journal_part = journal
        if volume:
            journal_part += f" {volume}"
        if page:
            journal_part += f":{page}"
        parts.append(f"{journal_part}.")
    if doi:
        parts.append(f"https://doi.org/{doi}")
    citation = " ".join(parts).strip()
    if not citation:
        return []
    return [
        _candidate(
            "bibliographicCitation",
            citation,
            f"{locator_prefix}:crossref_composed",
            support_type=SupportType.DETERMINISTICALLY_DERIVED,
        )
    ]


def extract_publication_metadata_facts(
    fulltext_xml: str | None,
    crossref_raw: dict | None,
    *,
    locator_prefix: str,
) -> list[RawFactCandidate]:
    """Runs every extractor in this module and merges the results -- the
    single entry point `workflow/handlers.py` calls."""
    facts: list[RawFactCandidate] = []
    if fulltext_xml:
        facts.extend(extract_from_jats_permissions(fulltext_xml, locator_prefix=locator_prefix))
        facts.extend(extract_from_jats_authors(fulltext_xml, locator_prefix=locator_prefix))
        facts.extend(extract_code_repo_from_text(fulltext_xml, locator_prefix=locator_prefix))
    facts.extend(format_bibliographic_citation(crossref_raw, locator_prefix=locator_prefix))
    return facts
