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
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from fair_ocean_agent.database.enums import EntityLevel, ReviewStatus, SupportType
from fair_ocean_agent.database.models import RawFact
from fair_ocean_agent.discovery.text_identifiers import xml_to_text
from fair_ocean_agent.extraction.sections import (
    RELEVANT_SECTION_TITLE_PATTERNS,
    RESULT_DISCUSSION_SECTION_TITLE_PATTERNS,
)
from fair_ocean_agent.identity.identifiers import IdentifierError, normalize_doi
from fair_ocean_agent.sources.base import RawFactCandidate

_XLINK_HREF = "{http://www.w3.org/1999/xlink}href"

_CODE_REPO_PATTERN = re.compile(
    r"https?://(?:www\.)?(?:github|gitlab|bitbucket)\.(?:com|org)/[^\s<>()\[\]{}\"']+",
    re.IGNORECASE,
)
_URL_PATTERN = re.compile(r"https?://[^\s<>()\[\]{}\"']+", re.IGNORECASE)

# Fallback for a paper whose analysis code isn't in a public repository at
# all (schemas/faire/schema.yaml's own code_repo description is "Link to
# public repository where analysis code is archived") but the authors still
# say where it lives -- e.g. "available in Supplemental Information 1",
# "available upon request", a non-GitHub/GitLab/Bitbucket URL. Confirmed via
# a real paper (PeerJ 10.7717/peerj.333) whose only code mention is exactly
# this shape ("Perl script for rarefaction analysis (cca_rarefaction.pl)
# ... are available in Supplemental Information 1.") -- leaving code_repo
# blank for a paper like this silently drops a real, useful pointer to
# where the code can be found, even though it isn't a public-repository
# link. Captures the whole sentence as the value/evidence so a reviewer can
# see exactly what the paper said, not just a bare "supplement" flag.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CODE_AVAILABILITY_KEYWORDS_RE = re.compile(
    r"\b(?:source\s+code|analysis\s+code|custom\s+scripts?|supplement(?:ary|al)\s+code|"
    r"supplement(?:ary|al)\s+software|scripts?|code|software)s?\b.*"
    r"\b(?:available|deposited|provided|included|archived|accessible|hosted|can\s+be\s+found)\b"
    r"|"
    r"\b(?:available|deposited|provided|included|archived|accessible|hosted|can\s+be\s+found)\b.*"
    r"\b(?:source\s+code|analysis\s+code|custom\s+scripts?|supplement(?:ary|al)\s+code|"
    r"supplement(?:ary|al)\s+software|scripts?|code|software)s?\b",
    re.IGNORECASE,
)
_CITATION_MARKER_RE = re.compile(r"\[__CITE:([^_\]]+)__\]")

# Cleans a JATS <copyright-statement>'s free text down to (an approximation
# of) just the holder's own name -- see extract_from_jats_permissions's
# rightsHolder fallback. Confirmed against ~40 real cached articles that
# real copyright-statement text comes in two structurally different
# shapes, both needing the year token treated as the boundary, but on
# opposite sides of the holder name:
#   "(c) 2013 Jessen et al"                       (year BEFORE the name)
#   "(c) The Author(s) 2024"                        (year AFTER the name)
#   "(c) The Author(s) 2020. Published by Oxford..." (year, then an
#                                                    unrelated trailing
#                                                    publisher-attribution
#                                                    clause)
_COPYRIGHT_SYMBOL_RE = re.compile(r"^(?:copyright\s*)?©?\s*", re.IGNORECASE)
_LEADING_YEAR_RE = re.compile(r"^(?:19|20)\d{2}\b[.,]?\s*")
_YEAR_ONWARD_RE = re.compile(r"\s*\b(?:19|20)\d{2}\b.*$")

# A stripped copyright-statement can bottom out at a generic, non-
# identifying phrase ("The Author(s)") rather than a real named holder --
# confirmed against a real paper (10.1038/s42003-024-06136-2) whose only
# <copyright-statement> is literally "(c) The Author(s) 2024", unlike a
# sibling paper with a real <copyright-holder> naming its authors. Per an
# explicit user complaint that "The Author(s)" isn't a satisfactory
# rightsHolder value, this falls back further to the paper's own author
# surnames when the copyright statement itself names no one.
_GENERIC_RIGHTS_HOLDER_RE = re.compile(r"^\s*the\s+author\(?s\)?\.?\s*$", re.IGNORECASE)


def _author_names_full(root: ET.Element) -> str:
    names: list[str] = []
    for contrib_group in root.iter("contrib-group"):
        for contrib in contrib_group.findall("contrib"):
            if contrib.get("contrib-type") != "author":
                continue
            name_el = contrib.find("name")
            if name_el is None:
                continue
            given = (name_el.findtext("given-names") or "").strip()
            surname = (name_el.findtext("surname") or "").strip()
            full_name = " ".join(part for part in (given, surname) if part)
            if full_name:
                names.append(full_name)
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def _strip_copyright_statement_noise(text: str) -> str:
    text = _COPYRIGHT_SYMBOL_RE.sub("", text, count=1)
    without_leading_year = _LEADING_YEAR_RE.sub("", text, count=1)
    if without_leading_year != text:
        return without_leading_year.strip()
    return _YEAR_ONWARD_RE.sub("", text, count=1).strip()


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _clean_text(text: str) -> str:
    return " ".join(text.split())


def _first_child_text(element: ET.Element, child_name: str) -> str:
    for child in list(element):
        if _local_name(child.tag) == child_name:
            return _clean_text("".join(child.itertext()))
    return ""


def _section_title(section: ET.Element) -> str:
    return _first_child_text(section, "title")


def _iter_leaf_sections(root: ET.Element) -> list[tuple[ET.Element, list[str]]]:
    leaves: list[tuple[ET.Element, list[str]]] = []

    def walk(section: ET.Element, parent_titles: list[str]) -> None:
        title = _section_title(section)
        titles = [*parent_titles, title] if title else parent_titles
        children = [child for child in list(section) if _local_name(child.tag) == "sec"]
        if children:
            for child in children:
                walk(child, titles)
        else:
            leaves.append((section, titles))

    for body in root.iter():
        if _local_name(body.tag) != "body":
            continue
        for section in list(body):
            if _local_name(section.tag) == "sec":
                walk(section, [])
    return leaves


def _matches_any(title: str, patterns: Sequence[re.Pattern[str]]) -> bool:
    return any(pattern.search(title) for pattern in patterns)


def _is_method_leaf(titles: list[str]) -> bool:
    if any(_matches_any(title, RESULT_DISCUSSION_SECTION_TITLE_PATTERNS) for title in titles):
        return False
    return any(_matches_any(title, RELEVANT_SECTION_TITLE_PATTERNS) for title in titles)


def _ref_doi_url(ref: ET.Element) -> str | None:
    for node in ref.iter():
        if _local_name(node.tag) != "pub-id":
            continue
        if (node.get("pub-id-type") or "").casefold() != "doi":
            continue
        raw_doi = _clean_text("".join(node.itertext()))
        if not raw_doi:
            continue
        try:
            return f"https://doi.org/{normalize_doi(raw_doi)}"
        except IdentifierError:
            continue
    return None


def _ref_fallback_text(ref: ET.Element) -> str:
    title = ""
    for node in ref.iter():
        if _local_name(node.tag) == "article-title":
            title = _clean_text("".join(node.itertext()))
            break
    if title:
        return title[:300]
    text = _clean_text("".join(ref.itertext()))
    return f"{text[:297].rstrip()}..." if len(text) > 300 else text


def _bibliography_resources(root: ET.Element) -> dict[str, dict[str, str]]:
    resources: dict[str, dict[str, str]] = {}
    for ref in root.iter():
        if _local_name(ref.tag) != "ref":
            continue
        ref_id = ref.get("id")
        if not ref_id:
            continue
        doi_url = _ref_doi_url(ref)
        fallback = _ref_fallback_text(ref)
        resource = doi_url or fallback
        if resource:
            resources[ref_id] = {"resource": resource, "citation_text": fallback}
    return resources


def _text_with_citation_markers(element: ET.Element) -> str:
    parts: list[str] = []

    def walk(node: ET.Element) -> None:
        if _local_name(node.tag) == "xref" and (node.get("ref-type") or "").casefold() == "bibr":
            rid = node.get("rid")
            if rid:
                for ref_id in rid.split():
                    parts.append(f" [__CITE:{ref_id}__] ")
        if node.text:
            parts.append(node.text)
        for child in list(node):
            walk(child)
            if child.tail:
                parts.append(child.tail)

    walk(element)
    return _clean_text("".join(parts))


def _paragraph_like_nodes(section: ET.Element) -> list[ET.Element]:
    return [node for node in section.iter() if _local_name(node.tag) in {"p", "li"}]


def _citation_section_heading(titles: list[str]) -> str:
    return titles[-1] if titles else "Methods"


def _find_code_availability_sentence(text: str) -> str | None:
    """Scans real sentence boundaries (splitting only on punctuation
    followed by whitespace) rather than a single regex spanning the whole
    sentence -- a naive "no periods until the sentence ends" regex breaks on
    this exact real paper, whose sentence contains two embedded periods in
    filenames (cca_rarefaction.pl, rarefaction_figs.R) before ever reaching
    "available"."""
    normalized = " ".join(text.split())
    for sentence in _SENTENCE_SPLIT_RE.split(normalized):
        sentence = sentence.strip()
        if _CODE_AVAILABILITY_KEYWORDS_RE.search(sentence):
            return sentence
    return None


def _candidate(
    fact_type: str,
    raw_value: str,
    locator: str,
    support_type: SupportType = SupportType.STRUCTURED_SOURCE,
    evidence_quote: str | None = None,
) -> RawFactCandidate:
    return RawFactCandidate(
        entity_level=EntityLevel.STUDY,
        fact_type_candidate=fact_type,
        raw_field_name=fact_type,
        raw_value=raw_value,
        source_locator=locator,
        support_type=support_type,
        evidence_quote=evidence_quote,
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
    holder = _first_child_text(permissions, "copyright-holder") if holder_el is not None else ""
    holder_locator = f"{locator_prefix}:permissions/copyright-holder"
    if not holder:
        # <copyright-holder> is real but inconsistently populated across
        # publishers -- confirmed against this repo's own cached fulltext
        # (Nature/Macmillan, MDPI, and a third 2013 article all carry a
        # real <copyright-statement> with no separate <copyright-holder>
        # element at all). <copyright-statement> is free prose ("(c) 2016,
        # Macmillan Publishers Limited", "(c) 2023 by the authors."), so
        # the leading copyright-symbol/year noise is stripped
        # deterministically, leaving the holder's own wording intact
        # rather than dropping the field entirely for the majority of
        # real articles that only carry this tag.
        statement_el = permissions.find("copyright-statement")
        if statement_el is not None:
            statement_text = _clean_text("".join(statement_el.itertext()))
            holder = _strip_copyright_statement_noise(statement_text)
            holder_locator = f"{locator_prefix}:permissions/copyright-statement"
    if holder and _GENERIC_RIGHTS_HOLDER_RE.match(holder):
        author_names = _author_names_full(root)
        if author_names:
            holder = author_names
            holder_locator = (
                f"{locator_prefix}:contrib-group/contrib[@contrib-type='author']:"
                "author_name_fallback_for_generic_rights_holder"
            )
    if holder:
        facts.append(_candidate("rightsHolder", holder, holder_locator))
    return facts


def extract_from_jats_authors(xml: str, *, locator_prefix: str) -> list[RawFactCandidate]:
    """`paper_authors_list`/`project_contact` from JATS `<contrib-group>`,
    filtered to `contrib-type="author"` only -- never editors or other
    contributor roles that share the same element shape.

    `paper_authors_list` is deliberately NOT `fact_type_candidate=
    "recordedBy"` (a pipe-joined list of every author) -- an explicit user
    instruction: `recordedBy` should preferentially come from the real
    per-study submitter contact on the study's own NCBI BioSample records
    (sources/ncbi.py::_recorded_by_facts), falling back to just the FIRST
    paper author, never the full list, only when no BioSample submitter
    data exists at all. `paper_authors_list` is that fallback's own raw
    material -- see
    identity/sample_alias_reconciliation.py-adjacent sync_recorded_by_
    from_biosample_or_first_author (mapping/faire.py's pre-step chain),
    which reads this fact and decides the real `recordedBy` value.
    recordedByID is no longer extracted at all here -- an explicit user
    instruction to never populate it (exports/faire.py's
    PROJECT_METADATA_SUPPRESSED_FIELDS also drops its column entirely)."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return []

    names: list[str] = []
    contact_value: str | None = None
    for contrib_group in root.iter("contrib-group"):
        for contrib in contrib_group.findall("contrib"):
            if contrib.get("contrib-type") != "author":
                continue
            full_name = ""
            name_el = contrib.find("name")
            if name_el is not None:
                given = (name_el.findtext("given-names") or "").strip()
                surname = (name_el.findtext("surname") or "").strip()
                full_name = " ".join(part for part in (given, surname) if part)
                if full_name:
                    names.append(full_name)
            if contact_value is None and contrib.get("corresp") == "yes":
                email_el = contrib.find("email")
                email = (email_el.text or "").strip() if email_el is not None else ""
                if full_name and email:
                    contact_value = f"{full_name} <{email}>"
                elif full_name:
                    contact_value = full_name
                elif email:
                    contact_value = email

    facts: list[RawFactCandidate] = []
    if names:
        facts.append(
            _candidate(
                "paper_authors_list",
                " | ".join(names),
                f"{locator_prefix}:contrib-group/contrib[@contrib-type='author']",
            )
        )
    if contact_value:
        facts.append(
            _candidate(
                "project_contact",
                contact_value,
                f"{locator_prefix}:contrib-group/contrib[@corresp='yes']",
            )
        )
    return facts


def extract_code_repo_from_text(xml: str, *, locator_prefix: str) -> list[RawFactCandidate]:
    """`code_repo` via a flat-text regex -- the one field in this module
    without a reliable structured JATS element to read instead. Prefers a
    real public-repository URL (GitHub/GitLab/Bitbucket); falls back to a
    code-availability sentence, including non-GitHub URLs or supplementary
    code, and finally emits the explicit FAIRe value requested when no code
    source is published."""
    text = xml_to_text(xml)
    url_match = _CODE_REPO_PATTERN.search(text)
    if url_match:
        url = url_match.group(0).rstrip(".,;:)")
        return [
            _candidate(
                "code_repo",
                url,
                f"{locator_prefix}:fulltext_regex",
                support_type=SupportType.DETERMINISTICALLY_DERIVED,
            )
        ]
    sentence = _find_code_availability_sentence(text)
    if not sentence:
        return [
            _candidate(
                "code_repo",
                "no code published",
                f"{locator_prefix}:fulltext_regex:no_code_published",
                support_type=SupportType.DETERMINISTICALLY_DERIVED,
            )
        ]
    sentence_url = _URL_PATTERN.search(sentence)
    if sentence_url:
        url = sentence_url.group(0).rstrip(".,;:)")
        return [
            _candidate(
                "code_repo",
                url,
                f"{locator_prefix}:fulltext_regex_availability_statement_url",
                support_type=SupportType.DETERMINISTICALLY_DERIVED,
                evidence_quote=sentence,
            )
        ]
    return [
        _candidate(
            "code_repo",
            sentence,
            f"{locator_prefix}:fulltext_regex_availability_statement",
            support_type=SupportType.DETERMINISTICALLY_DERIVED,
            evidence_quote=sentence,
        )
    ]


def extract_method_section_citations(xml: str, *, locator_prefix: str) -> list[RawFactCandidate]:
    """`associated_resource` from every bibliography citation inside
    Methods-like sections, grouped by the leaf subsection heading. DOI links
    are preferred from the JATS `<ref-list>`; if a cited reference has no
    DOI, the compact reference title/text is retained as a fallback."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return []

    bibliography = _bibliography_resources(root)
    if not bibliography:
        return []

    section_values: list[str] = []
    section_details: list[dict[str, object]] = []
    evidence_snippets: list[str] = []
    for section, titles in _iter_leaf_sections(root):
        if not _is_method_leaf(titles):
            continue
        heading = _citation_section_heading(titles)
        section_title = " > ".join(titles) if titles else heading
        resources: list[str] = []
        citation_details: list[dict[str, str]] = []
        seen_in_section: set[str] = set()
        for node in _paragraph_like_nodes(section):
            marked_text = _text_with_citation_markers(node)
            ref_ids = _CITATION_MARKER_RE.findall(marked_text)
            if not ref_ids:
                continue
            clean_snippet = _CITATION_MARKER_RE.sub("", marked_text)
            clean_snippet = _clean_text(clean_snippet)
            if clean_snippet and clean_snippet not in evidence_snippets:
                evidence_snippets.append(clean_snippet)
            for ref_id in ref_ids:
                resource = bibliography.get(ref_id)
                if resource is None or resource["resource"] in seen_in_section:
                    continue
                seen_in_section.add(resource["resource"])
                resources.append(resource["resource"])
                citation_details.append(
                    {
                        "ref_id": ref_id,
                        "resource": resource["resource"],
                        "citation_text": resource["citation_text"],
                    }
                )
        if resources:
            section_values.append(f"**{heading}**: {'; '.join(resources)}")
            section_details.append(
                {
                    "section_title": section_title,
                    "heading": heading,
                    "citations": citation_details,
                }
            )

    if not section_values:
        return []
    return [
        RawFactCandidate(
            entity_level=EntityLevel.STUDY,
            fact_type_candidate="associated_resource",
            raw_field_name="associated_resource",
            raw_value=" | ".join(section_values),
            source_locator=f"{locator_prefix}:method_section_citations",
            support_type=SupportType.DETERMINISTICALLY_DERIVED,
            evidence_quote=" | ".join(evidence_snippets),
            confidence_metadata={"method_section_citations": section_details},
        )
    ]


def extract_method_protocol_citations(xml: str, *, locator_prefix: str) -> list[RawFactCandidate]:
    return extract_method_section_citations(xml, locator_prefix=locator_prefix)


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
        facts.extend(extract_method_section_citations(fulltext_xml, locator_prefix=locator_prefix))
    facts.extend(format_bibliographic_citation(crossref_raw, locator_prefix=locator_prefix))
    return facts


# recordedBy's own extraction_method marker, set on the synced fallback
# fact this function writes -- lets a rerun find and re-evaluate its own
# prior output rather than accumulating duplicates.
_RECORDED_BY_FALLBACK_EXTRACTION_METHOD = "derived:recorded_by_first_author_fallback"


def sync_recorded_by_from_biosample_or_first_author(session: Session, study_id: str) -> None:
    """Idempotent, re-run-safe (same pattern as identity/sample_alias_
    reconciliation.py::reconcile_sample_aliases and extraction/
    taxonomic_assay.py::sync_assay_target_taxa_from_biosample_organisms):
    called from mapping/faire.py::map_study_to_faire's pre-step chain, so
    it always reflects the study's current raw_facts regardless of which
    order discovery/extraction stages ran in.

    Per an explicit user instruction: `recordedBy` should come from the
    real per-BioSample submitter contact (sources/ncbi.py::
    _recorded_by_facts, now sourced from BioSample's own
    Owner/Contacts/Contact/Name -- a real person -- not Owner/Name, which
    real data confirmed is the submitting INSTITUTION's name, not a
    person's). Only when the study has no such BioSample submitter data
    at all does this fall back to the paper's own FIRST author (never the
    full author list) -- extract_from_jats_authors's own
    `paper_authors_list` fact is the raw material for that fallback,
    itself deliberately not `fact_type_candidate="recordedBy"` (see that
    function's docstring)."""
    biosample_recorded_by_exists = (
        session.query(RawFact.fact_id)
        .filter(
            RawFact.study_id == study_id,
            RawFact.fact_type_candidate == "recordedBy",
            RawFact.source_locator.like("ncbi_biosample.%"),
            RawFact.review_status != ReviewStatus.REJECTED.value,
        )
        .first()
        is not None
    )
    existing_fallback = session.scalar(
        select(RawFact).where(
            RawFact.study_id == study_id,
            RawFact.entity_id.is_(None),
            RawFact.fact_type_candidate == "recordedBy",
            RawFact.extraction_method == _RECORDED_BY_FALLBACK_EXTRACTION_METHOD,
        )
    )
    if biosample_recorded_by_exists:
        # Self-heals if a later discovery pass adds real BioSample
        # submitter data after an earlier run already wrote the fallback.
        if existing_fallback is not None:
            existing_fallback.review_status = ReviewStatus.REJECTED.value
        return

    authors_fact = session.scalar(
        select(RawFact).where(
            RawFact.study_id == study_id,
            RawFact.fact_type_candidate == "paper_authors_list",
            RawFact.review_status != ReviewStatus.REJECTED.value,
        )
    )
    first_author = authors_fact.raw_value.split(" | ")[0].strip() if authors_fact and authors_fact.raw_value else ""
    if not first_author:
        if existing_fallback is not None:
            existing_fallback.review_status = ReviewStatus.REJECTED.value
        return

    if existing_fallback is not None:
        if existing_fallback.raw_value != first_author:
            existing_fallback.raw_value = first_author
        existing_fallback.review_status = ReviewStatus.ACCEPTED.value
        return

    session.add(
        RawFact(
            study_id=study_id,
            entity_id=None,
            source_id=None,
            source_locator=authors_fact.source_locator,
            raw_field_name="recordedBy",
            raw_value=first_author,
            fact_type_candidate="recordedBy",
            entity_level=EntityLevel.STUDY.value,
            support_type=SupportType.DETERMINISTICALLY_DERIVED.value,
            extraction_method=_RECORDED_BY_FALLBACK_EXTRACTION_METHOD,
            review_status=ReviewStatus.ACCEPTED.value,
            confidence_metadata={"detector": "recorded_by_first_author_fallback"},
        )
    )
