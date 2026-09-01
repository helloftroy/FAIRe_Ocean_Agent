"""Publication-level project metadata directly from a paper's own
structured sources and tightly scoped paper sections.

Most fields in this module remain deterministic structured-source
extractions. Narrow exceptions (`rightsHolder`, `funding_source`) use the
LLM only after JATS has already isolated the relevant rights/funding text;
the model's job there is selection/cleanup from a short paragraph, not a
paper-wide search.

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
(`license`, `rightsHolder`, ...), matching this codebase's project-
metadata convention for facts that are already scoped to one FAIRe field.
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
from fair_ocean_agent.llm.base import LLMBackend, LLMBackendError
from fair_ocean_agent.sources.base import RawFactCandidate

_XLINK_HREF = "{http://www.w3.org/1999/xlink}href"
_OPEN_ACCESS_LICENSE_RE = re.compile(
    r"creativecommons\.org/(?:licenses|publicdomain)/",
    re.IGNORECASE,
)
_CREATIVE_COMMONS_LICENSE_URL_RE = re.compile(
    r"https?://creativecommons\.org/(?:licenses|publicdomain)/[^\s<>()\[\]{}\"']+/?",
    re.IGNORECASE,
)

_CODE_REPO_PATTERN = re.compile(
    r"https?://(?:www\.)?(?:github|gitlab|bitbucket)\.(?:com|org)/[^\s<>()\[\]{}\"']+",
    re.IGNORECASE,
)
_URL_PATTERN = re.compile(r"https?://[^\s<>()\[\]{}\"']+", re.IGNORECASE)
_DOI_IN_TEXT_RE = re.compile(
    r"(?i)\b(?:doi\s*:\s*|https?://(?:dx\.)?doi\.org/)?(10\.\d{4,9}/[^\s<>()\[\]{}\"']+)"
)

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

_FUNDING_TITLE_RE = re.compile(r"\b(?:funding|funding information|financial disclosure|grant support)\b", re.IGNORECASE)
_FUNDING_TEXT_RE = re.compile(
    r"\b(?:funded|funding|financial support|grant(?:s)?|award(?:s)?|"
    r"fellowship|scholarship|start-?up\s+grant)\b"
    r"|"
    r"\bsupported by\b.{0,140}\b(?:foundation|fund|grant|award|council|ministry|agency|"
    r"fellowship|scholarship|commission|program(?:me)?)\b",
    re.IGNORECASE,
)
_FUNDING_AUTHOR_CONTRIBUTION_RE = re.compile(r"\bfunding\s+acquisition\s*:", re.IGNORECASE)
_FUNDED_BY_SEGMENT_RE = re.compile(
    r"\b(?:funded|supported)\s+by\s+(.+?)(?:\.\s*The\s+funders\b|\.?\s*$)",
    re.IGNORECASE,
)
_FUNDING_PARENTHESES_RE = re.compile(
    r"\s*\((?:[^)]*\b(?:grant|grants|award|awards|fellowship|scholarship|GRK)\b[^)]*|[A-Z]{2,}[-\s]?\d+[^)]*)\)",
    re.IGNORECASE,
)
_RIGHTS_TITLE_RE = re.compile(
    r"\b(?:rights|rights and permissions|permissions|copyright|license|open access)\b",
    re.IGNORECASE,
)
_ABSENT_FUNDING_RE = re.compile(
    r"^\s*(?:none|not found|no funding(?: source)?|no external funding|"
    r"no specific funding|not applicable|n/a)\s*\.?\s*$",
    re.IGNORECASE,
)
_FUNDING_INSTITUTIONAL_UNIT_RE = re.compile(
    r"\b(?:university|department|section|faculty|school|institute|centre|center|"
    r"laborator(?:y|ies)|facility|facilities|infrastructure)\b",
    re.IGNORECASE,
)
_FUNDING_KEEP_UNIT_RE = re.compile(
    r"\b(?:foundation|fund|council|ministry|agency|commission|trust|society|association|"
    r"fellowship|scholarship|award|grant|program(?:me)?)\b",
    re.IGNORECASE,
)
_PLAIN_HEADING_RE = re.compile(r"(?m)^\s*([A-Z][A-Za-z0-9 /&,\-()]{2,90})\s*$")
_PLAIN_SECTION_END_TITLE_RE = re.compile(
    r"\b(?:abstract|introduction|background|materials?\s+and\s+methods?|methods?|results?|"
    r"discussion|conclusions?|references|bibliography|author contributions?|competing interests?|"
    r"conflicts? of interest|data availability|supplementary material)\b",
    re.IGNORECASE,
)
_PLAIN_METADATA_HEADING_TITLE_RE = re.compile(
    r"^(?:abstract|introduction|background|materials?\s+and\s+methods?|methods?|results?|"
    r"discussion|conclusions?|references|bibliography|funding|funding information|"
    r"financial disclosure|grant support|rights|rights and permissions|permissions|"
    r"copyright|license|open access|author contributions?|competing interests?|"
    r"conflicts? of interest|data availability|supplementary material)$",
    re.IGNORECASE,
)
_RIGHTS_SENTENCE_RE = re.compile(
    r"\b(?:copyright|©|\(c\)|all rights reserved|creative commons|open access|"
    r"under exclusive licence|distributed under|licensed under)\b",
    re.IGNORECASE,
)


def _is_author_contrib(contrib: ET.Element, contrib_group: ET.Element) -> bool:
    """A JATS <contrib> is an author either via its OWN contrib-type="author"
    attribute, or -- confirmed live, a real gap (10.1038/s42003-024-06136-2's
    Europe PMC fullTextXML) -- via its PARENT <contrib-group
    content-type="author">, when the contrib itself carries no contrib-type
    at all. Only the group-level signal is trusted as a fallback (never
    overrides an explicit non-author contrib-type like "editor")."""
    contrib_type = contrib.get("contrib-type")
    if contrib_type is not None:
        return contrib_type == "author"
    return contrib_group.get("content-type") == "author"


def _corresponding_author_fn_ids(root: ET.Element) -> set[str]:
    """<author-notes><fn id="X"><p>Corresponding author.</p></fn></author-notes>
    -- the real convention when a document marks its corresponding author
    via a footnote reference rather than a per-contrib corresp="yes"
    attribute (confirmed live: 10.1038/s42003-024-06136-2's Europe PMC
    fullTextXML uses this shape, with no corresp attribute or <email>
    element on the <contrib> itself anywhere)."""
    ids: set[str] = set()
    for fn in root.iter("fn"):
        text = _clean_text("".join(fn.itertext())).casefold()
        fn_id = fn.get("id")
        if fn_id and "correspond" in text:
            ids.add(fn_id)
    return ids


def _is_corresponding_contrib(contrib: ET.Element, corresponding_fn_ids: set[str]) -> bool:
    if contrib.get("corresp") == "yes":
        return True
    return any(
        _local_name(xref.tag) == "xref" and xref.get("ref-type") == "author-notes" and xref.get("rid") in corresponding_fn_ids
        for xref in contrib.iter("xref")
    )


def _author_names_full(root: ET.Element) -> str:
    names: list[str] = []
    for contrib_group in root.iter("contrib-group"):
        for contrib in contrib_group.findall("contrib"):
            if not _is_author_contrib(contrib, contrib_group):
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


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _clean_text(text: str) -> str:
    return " ".join(text.split())


def _element_text(element: ET.Element) -> str:
    return _clean_text("".join(element.itertext()))


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


def _doi_resource(raw_doi: str) -> str | None:
    raw_doi = raw_doi.strip().rstrip(".,;:")
    try:
        return f"doi: {normalize_doi(raw_doi)}"
    except IdentifierError:
        return None


def _ref_doi_url(ref: ET.Element) -> str | None:
    for node in ref.iter():
        if _local_name(node.tag) != "pub-id":
            continue
        if (node.get("pub-id-type") or "").casefold() != "doi":
            continue
        raw_doi = _clean_text("".join(node.itertext()))
        if not raw_doi:
            continue
        resource = _doi_resource(raw_doi)
        if resource:
            return resource
    text = _clean_text("".join(ref.itertext()))
    for match in _DOI_IN_TEXT_RE.finditer(text):
        resource = _doi_resource(match.group(1))
        if resource:
            return resource
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


def _iter_elements(root: ET.Element, local_name: str) -> list[ET.Element]:
    return [element for element in root.iter() if _local_name(element.tag) == local_name]


def _paragraphs_under(element: ET.Element) -> list[str]:
    paragraphs = [_element_text(child) for child in element.iter() if _local_name(child.tag) == "p"]
    paragraphs = [paragraph for paragraph in paragraphs if paragraph]
    return paragraphs or ([_element_text(element)] if _element_text(element) else [])


def _direct_paragraphs_under(element: ET.Element) -> list[str]:
    paragraphs = [_element_text(child) for child in list(element) if _local_name(child.tag) == "p"]
    paragraphs = [paragraph for paragraph in paragraphs if paragraph]
    return paragraphs or ([_element_text(element)] if _element_text(element) else [])


def _funding_sentences(text: str) -> list[str]:
    return [
        sentence
        for sentence in _sentences(text)
        if _FUNDING_TEXT_RE.search(sentence) and not _FUNDING_AUTHOR_CONTRIBUTION_RE.search(sentence)
    ]


def _funding_paragraphs_from_jats(fulltext_xml: str | None) -> list[str]:
    if not fulltext_xml:
        return []
    try:
        root = ET.fromstring(fulltext_xml)
    except ET.ParseError:
        return []

    paragraphs: list[str] = []

    for funding_group in _iter_elements(root, "funding-group"):
        paragraphs.extend(_paragraphs_under(funding_group))

    for sec in _iter_elements(root, "sec"):
        title = _section_title(sec)
        sec_type = sec.get("sec-type", "")
        if _FUNDING_TITLE_RE.search(title) or _FUNDING_TITLE_RE.search(sec_type):
            paragraphs.extend(_direct_paragraphs_under(sec))

    for fn in _iter_elements(root, "fn"):
        fn_type = fn.get("fn-type", "")
        text = _element_text(fn)
        if _FUNDING_TITLE_RE.search(fn_type) or _FUNDING_TITLE_RE.search(text[:160]):
            paragraphs.extend(_direct_paragraphs_under(fn))

    for ack in _iter_elements(root, "ack"):
        for paragraph in _direct_paragraphs_under(ack):
            paragraphs.extend(_funding_sentences(paragraph))

    unique: list[str] = []
    seen: set[str] = set()
    for paragraph in paragraphs:
        cleaned = _clean_text(paragraph)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        unique.append(cleaned)
    return unique


def _plain_text_sections(text: str | None) -> list[tuple[str, str]]:
    if not text:
        return []
    matches = list(_PLAIN_HEADING_RE.finditer(text))
    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        title = _clean_text(match.group(1))
        if not _PLAIN_METADATA_HEADING_TITLE_RE.match(title):
            continue
        start = match.end()
        end = len(text)
        for next_match in matches[index + 1 :]:
            next_title = _clean_text(next_match.group(1))
            if _PLAIN_METADATA_HEADING_TITLE_RE.match(next_title):
                end = next_match.start()
                break
        body = _clean_text(text[start:end])
        if title and body:
            sections.append((title, body))
    return sections


def _sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in _SENTENCE_SPLIT_RE.split(_clean_text(text)) if sentence.strip()]


def _funding_paragraphs_from_plain_text(text: str | None) -> list[str]:
    if not text:
        return []
    paragraphs: list[str] = []
    for title, body in _plain_text_sections(text):
        if _FUNDING_TITLE_RE.search(title):
            paragraphs.append(body)
            continue
        if _PLAIN_SECTION_END_TITLE_RE.search(title) and not _FUNDING_TITLE_RE.search(title):
            continue
        paragraphs.extend(_funding_sentences(body))

    if not paragraphs:
        paragraphs = _funding_sentences(text)

    unique: list[str] = []
    seen: set[str] = set()
    for paragraph in paragraphs:
        cleaned = _clean_text(paragraph)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        unique.append(cleaned)
    return unique


def _normalize_funding_source_value(value: object) -> str:
    return _filter_funding_source_value(value)


def _filter_funding_source_value(value: object) -> str:
    if isinstance(value, list):
        pieces = [str(piece).strip() for piece in value]
    else:
        pieces = [piece.strip() for piece in str(value or "").split("|")]

    normalized: list[str] = []
    seen: set[str] = set()
    for piece in pieces:
        piece = piece.strip(" ;,.")
        piece = re.sub(r"\bReversibi\s+lity\b", "Reversibility", piece)
        if not piece or _ABSENT_FUNDING_RE.match(piece):
            continue
        if len(piece) <= 2 or not re.search(r"[A-Za-z]", piece):
            continue
        if _FUNDING_INSTITUTIONAL_UNIT_RE.search(piece) and not _FUNDING_KEEP_UNIT_RE.search(piece):
            continue
        key = piece.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(piece)
    return " | ".join(normalized)


def _fallback_funding_sources_from_text(funding_text: str) -> str:
    """Conservative backup for explicit "funded/supported by X" prose.

    This is intentionally narrower than the LLM: it only handles direct
    funded-by clauses and lets the existing post-filter drop plain host
    institutions after grant-number parentheticals are removed. That keeps
    PLOS-style lines such as "funded by DFG Research Training Group R3 ...
    (GRK 2272) and by the University of Konstanz (AFF grants ...)" from
    going blank or losing the DFG program, without reintroducing random
    affiliation/institution noise.
    """
    candidates: list[str] = []
    for match in _FUNDED_BY_SEGMENT_RE.finditer(funding_text):
        segment = _clean_text(match.group(1))
        segment = re.sub(r"\bThe\s+funders\b.*$", "", segment, flags=re.IGNORECASE).strip()
        pieces = re.split(r"\s+(?:and|,)\s+by\s+(?:the\s+)?|\s*;\s*", segment, flags=re.IGNORECASE)
        for piece in pieces:
            piece = _FUNDING_PARENTHESES_RE.sub("", piece)
            piece = re.sub(r"^\s*(?:the\s+)?", "", piece, flags=re.IGNORECASE)
            piece = piece.strip(" ;,.")
            if piece:
                candidates.append(piece)
    return _filter_funding_source_value(candidates)


def _rights_paragraphs_from_jats(fulltext_xml: str | None) -> list[str]:
    if not fulltext_xml:
        return []
    try:
        root = ET.fromstring(fulltext_xml)
    except ET.ParseError:
        return []

    paragraphs: list[str] = []
    for permissions in _iter_elements(root, "permissions"):
        text = _element_text(permissions)
        if text:
            paragraphs.append(text)

    for sec in _iter_elements(root, "sec"):
        title = _section_title(sec)
        sec_type = sec.get("sec-type", "")
        if _RIGHTS_TITLE_RE.search(title) or _RIGHTS_TITLE_RE.search(sec_type):
            paragraphs.extend(_paragraphs_under(sec))

    for fn in _iter_elements(root, "fn"):
        fn_type = fn.get("fn-type", "")
        text = _element_text(fn)
        if _RIGHTS_TITLE_RE.search(fn_type) or _RIGHTS_TITLE_RE.search(text[:200]):
            paragraphs.extend(_paragraphs_under(fn))

    unique: list[str] = []
    seen: set[str] = set()
    for paragraph in paragraphs:
        cleaned = _clean_text(paragraph)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        unique.append(cleaned)
    return unique


def _rights_paragraphs_from_plain_text(text: str | None) -> list[str]:
    if not text:
        return []
    paragraphs: list[str] = []
    for title, body in _plain_text_sections(text):
        if _RIGHTS_TITLE_RE.search(title):
            paragraphs.append(body)
            continue
        if _PLAIN_SECTION_END_TITLE_RE.search(title) and not _RIGHTS_TITLE_RE.search(title):
            continue
        paragraphs.extend(sentence for sentence in _sentences(body) if _RIGHTS_SENTENCE_RE.search(sentence))

    if not paragraphs:
        paragraphs = [sentence for sentence in _sentences(text) if _RIGHTS_SENTENCE_RE.search(sentence)]

    unique: list[str] = []
    seen: set[str] = set()
    for paragraph in paragraphs:
        cleaned = _clean_text(paragraph)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        unique.append(cleaned)
    return unique


def _normalize_rights_holder_value(value: object) -> str:
    holder = str(value or "").strip()
    holder = re.sub(r"^\s*(?:copyright\s*)?(?:©|\(c\))\s*", "", holder, flags=re.IGNORECASE).strip()
    holder = holder.strip(" ;,.")
    if holder.casefold() in {"", "none", "not found", "unknown", "n/a", "not applicable"}:
        return ""
    return holder


def generate_rights_holder(
    backend: LLMBackend,
    fulltext_xml: str | None,
    *,
    locator_prefix: str,
    max_output_tokens: int | None = 256,
    max_input_chars: int = 8000,
) -> list[RawFactCandidate]:
    """Extract `rightsHolder` from explicit rights/permissions text.

    Real articles vary: the holder may be named authors, "The Author(s)",
    a publisher, the journal, a society, or another organization. The year
    is often part of the rights statement and should be preserved rather
    than stripped or replaced with the paper's author list.
    """
    paragraphs = _rights_paragraphs_from_jats(fulltext_xml)
    if not paragraphs:
        paragraphs = _rights_paragraphs_from_plain_text(fulltext_xml)
    if not paragraphs:
        return []

    rights_text = "\n\n".join(paragraphs)
    if len(rights_text) > max_input_chars:
        rights_text = rights_text[:max_input_chars].rsplit(" ", 1)[0]

    prompt = f"""Read the rights, copyright, license, or permissions text below.

Extract the rights holder for the paper. The rights holder may be named authors, "The Author(s)", the journal,
publisher, society, or another organization. Preserve the year when the rights statement includes it as part of
the holder expression, for example "2024 The Author(s)" or "2013 Davies et al." Do not replace "The Author(s)"
with the actual author list. Do not return the license URL, license name, usage permissions, or open-access
boilerplate unless no rights holder is stated. Remove only leading copyright symbols like © or (c).

Rights text:
{rights_text}

Return ONLY a JSON object: {{"rightsHolder": "<rights holder>"}}
"""
    parsed, _response = backend.generate_json(
        prompt,
        system="You extract the rights holder from a paper rights or permissions section.",
        temperature=0,
        max_tokens=max_output_tokens,
    )
    if parsed is None:
        raise LLMBackendError(f"{backend.label}: rightsHolder generation returned invalid JSON after retries")
    value = _normalize_rights_holder_value(parsed.get("rightsHolder") if isinstance(parsed, dict) else "")
    if not value:
        return []

    return [
        RawFactCandidate(
            entity_level=EntityLevel.STUDY,
            fact_type_candidate="rightsHolder",
            raw_field_name="rightsHolder",
            raw_value=value,
            source_locator=f"{locator_prefix}:rightsHolder:llm_extracted_from_rights_text",
            support_type=SupportType.EXPLICIT,
            evidence_quote=rights_text,
            confidence_metadata={"detector": "llm_generated_rights_holder", "rights_paragraph_count": len(paragraphs)},
        )
    ]


def generate_funding_source(
    backend: LLMBackend,
    fulltext_xml: str | None,
    *,
    locator_prefix: str,
    max_output_tokens: int | None = 256,
    max_input_chars: int = 8000,
) -> list[RawFactCandidate]:
    """Extract project-level `funding_source` from explicit funding text.

    JATS already identifies funding/financial-disclosure paragraphs much
    more reliably than a broad paper-wide search. The LLM's job is only to
    reduce those paragraphs to funder/source names and drop grant numbers,
    author initials, and "funder had no role" boilerplate.
    """
    paragraphs = _funding_paragraphs_from_jats(fulltext_xml)
    if not paragraphs:
        paragraphs = _funding_paragraphs_from_plain_text(fulltext_xml)
    if not paragraphs:
        return []

    funding_text = "\n\n".join(paragraphs)
    if len(funding_text) > max_input_chars:
        funding_text = funding_text[:max_input_chars].rsplit(" ", 1)[0]

    prompt = f"""Read the funding or financial-disclosure text below.

Extract only funding sources that financially supported the study: funding agencies, foundations, councils,
ministries, named grant programs, named fellowships, named scholarships, or named awards. Do not include grant
numbers, award numbers, author initials, ordinary conflict-of-interest statements, or "the funders had no role"
boilerplate. Do not include universities, departments, sections, institutes, laboratories, facilities, centers,
field stations, or consortia unless the text explicitly names them as a grant/fellowship/award/scholarship
program. For named programs that include a funding agency acronym, such as "DFG Research Training Group R3",
preserve the full agency + program phrase. Do not include acknowledgements, collaborators, sequencing facilities,
host institutions, affiliations, or partial/truncated fragments. Use the names as written when possible. If there
is more than one funding source, join the names with " | ". If no funding source name is present, return an empty
string.

Funding text:
{funding_text}

Return ONLY a JSON object: {{"funding_source": "<pipe-delimited funder names>"}}
"""
    parsed, _response = backend.generate_json(
        prompt,
        system="You extract funder names from a paper funding paragraph.",
        temperature=0,
        max_tokens=max_output_tokens,
    )
    if parsed is None:
        raise LLMBackendError(f"{backend.label}: funding_source generation returned invalid JSON after retries")
    value = _normalize_funding_source_value(parsed.get("funding_source") if isinstance(parsed, dict) else "")
    if not value:
        value = _fallback_funding_sources_from_text(funding_text)
    if not value:
        return []

    return [
        RawFactCandidate(
            entity_level=EntityLevel.STUDY,
            fact_type_candidate="funding_source",
            raw_field_name="funding_source",
            raw_value=value,
            source_locator=f"{locator_prefix}:funding_source:llm_extracted_from_funding_paragraph",
            support_type=SupportType.EXPLICIT,
            evidence_quote=funding_text,
            confidence_metadata={"detector": "llm_generated_funding_source", "funding_paragraph_count": len(paragraphs)},
        )
    ]


def extract_from_jats_permissions(xml: str, *, locator_prefix: str) -> list[RawFactCandidate]:
    """`license`/`accessRights` from JATS `<permissions>`.

    `rightsHolder` is extracted by generate_rights_holder(), because the
    paper-facing rights wording can name authors, "The Author(s)", a
    publisher, journal, or society, and deterministic cleanup was too
    eager to rewrite those real holder statements.
    """
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return extract_license_from_text(xml, locator_prefix=locator_prefix)
    permissions = _iter_elements(root, "permissions")
    if not permissions:
        return extract_license_from_text(xml, locator_prefix=locator_prefix)

    facts: list[RawFactCandidate] = []
    seen: set[tuple[str, str]] = set()
    for permissions_el in permissions:
        license_els = [el for el in permissions_el.iter() if _local_name(el.tag) == "license"]
        for license_el in license_els:
            href = license_el.get(_XLINK_HREF) or license_el.get("href")
            if href and ("license", href) not in seen:
                seen.add(("license", href))
                facts.append(_candidate("license", href, f"{locator_prefix}:permissions/license/@xlink:href"))
            license_type = license_el.get("license-type")
            if license_type:
                access = "open access" if license_type == "open-access" else license_type
                if ("accessRights", access) not in seen:
                    seen.add(("accessRights", access))
                    facts.append(_candidate("accessRights", access, f"{locator_prefix}:permissions/license/@license-type"))
            elif href and _OPEN_ACCESS_LICENSE_RE.search(href) and ("accessRights", "open access") not in seen:
                seen.add(("accessRights", "open access"))
                facts.append(_candidate("accessRights", "open access", f"{locator_prefix}:permissions/license/@xlink:href"))

    if facts:
        return facts
    return extract_license_from_text(xml, locator_prefix=locator_prefix)


def extract_license_from_text(text: str, *, locator_prefix: str) -> list[RawFactCandidate]:
    """Fallback for HTML/PDF-ish text where the rights block is readable
    but not represented as a JATS `<license>` element."""
    facts: list[RawFactCandidate] = []
    plain = xml_to_text(text)
    for match in _CREATIVE_COMMONS_LICENSE_URL_RE.finditer(text):
        url = match.group(0).rstrip(".,;")
        facts.append(_candidate("license", url, f"{locator_prefix}:license_text:creativecommons_url"))
        facts.append(_candidate("accessRights", "open access", f"{locator_prefix}:license_text:creativecommons_url"))
        return facts
    if re.search(r"\bCreative Commons Attribution License\b|\bCC BY\b", plain, re.IGNORECASE):
        facts.append(_candidate("license", "https://creativecommons.org/licenses/by/4.0/", f"{locator_prefix}:license_text:cc_by"))
        facts.append(_candidate("accessRights", "open access", f"{locator_prefix}:license_text:cc_by"))
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
    corresponding_fn_ids = _corresponding_author_fn_ids(root)
    for contrib_group in root.iter("contrib-group"):
        for contrib in contrib_group.findall("contrib"):
            if not _is_author_contrib(contrib, contrib_group):
                continue
            full_name = ""
            name_el = contrib.find("name")
            if name_el is not None:
                given = (name_el.findtext("given-names") or "").strip()
                surname = (name_el.findtext("surname") or "").strip()
                full_name = " ".join(part for part in (given, surname) if part)
                if full_name:
                    names.append(full_name)
            if contact_value is None and _is_corresponding_contrib(contrib, corresponding_fn_ids):
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


_XLINK_HREF_ATTR = "{http://www.w3.org/1999/xlink}href"
# A supplementary file's own <caption>/<title> is a real, structured JATS
# element that commonly names its content directly ("Code for all analysis
# carried out") without ever using prose "availability" language (available/
# deposited/provided/...) that _CODE_AVAILABILITY_KEYWORDS_RE requires --
# confirmed missed live (10.7717/peerj.17091, PMC11067900): a "Code for all
# analysis carried out" caption on an attached .r file was dropped entirely,
# even though the paper clearly named and attached its own analysis code --
# it just wasn't on GitHub/GitLab/Bitbucket and never phrased it as prose.
_SUPPLEMENTARY_CODE_CAPTION_RE = re.compile(
    r"\b(?:code|scripts?|software|analysis\s+pipeline|R\s+script|python\s+script)\b",
    re.IGNORECASE,
)


def _code_repo_from_supplementary_caption(xml: str) -> tuple[str, str] | None:
    """Returns (value, evidence_quote) for the first <supplementary-material>
    element whose own <caption> mentions code/script/software, or None.
    value includes the attached file's own name (from <media xlink:href>)
    when present, since that's the concrete, useful pointer a reviewer
    wants -- not just "yes, there is code somewhere in the supplement"."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return None
    for supp in root.iter():
        if _local_name(supp.tag) != "supplementary-material":
            continue
        caption = next((child for child in list(supp) if _local_name(child.tag) == "caption"), None)
        if caption is None:
            continue
        caption_text = _element_text(caption)
        if not caption_text or not _SUPPLEMENTARY_CODE_CAPTION_RE.search(caption_text):
            continue
        filename = next(
            (
                media.get(_XLINK_HREF_ATTR)
                for media in supp.iter()
                if _local_name(media.tag) == "media" and media.get(_XLINK_HREF_ATTR)
            ),
            None,
        )
        value = f"{caption_text} ({filename})" if filename else caption_text
        return value, caption_text
    return None


def extract_code_repo_from_text(xml: str, *, locator_prefix: str) -> list[RawFactCandidate]:
    """`code_repo` via a flat-text regex -- the one field in this module
    without a reliable structured JATS element to read instead. Prefers a
    real public-repository URL (GitHub/GitLab/Bitbucket), then a prose
    code-availability sentence in the main text (including non-GitHub
    URLs) when one exists -- a real, already-composed sentence a human
    wrote to point readers at the code is preferable to a raw
    supplementary-material caption when both describe the same resource.
    Falls back to that structured supplementary-material caption only when
    the main text never explicitly says the code is available anywhere
    (real gap found live, 10.7717/peerj.17091/PMC11067900: a "Code for all
    analysis carried out" caption on an attached .r file, with no prose
    availability sentence anywhere in the main text at all, was dropped
    entirely). Finally emits the explicit FAIRe value requested when no
    code source is published anywhere."""
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
        supplementary_code = _code_repo_from_supplementary_caption(xml)
        if supplementary_code:
            value, evidence_quote = supplementary_code
            return [
                _candidate(
                    "code_repo",
                    value,
                    f"{locator_prefix}:supplementary_material_caption",
                    support_type=SupportType.DETERMINISTICALLY_DERIVED,
                    evidence_quote=evidence_quote,
                )
            ]
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


# pcr_primer_name_forward/reverse -> pcr_primer_reference_forward/reverse:
# when a primer's own sequence isn't given but the paper names it (e.g.
# "515F") and cites where it came from, that citation is the paper's own
# pointer to whoever DOES have the sequence -- per an explicit user
# request to chase that reference (and, when the referenced paper itself
# only cites further back, chase that too) so the sequence eventually
# becomes known for every paper that reuses this same primer name, not
# just this one. Reuses the exact same real citation-linking machinery as
# extract_method_section_citations above (_text_with_citation_markers's
# JATS <xref ref-type="bibr"> markers + _bibliography_resources' <ref-list>
# DOI lookup) rather than a guessed parenthetical-citation-shape regex --
# an exact, structured link to which reference the paper itself cites.
_PRIMER_NAME_TO_REFERENCE_FIELD = {
    "pcr_primer_name_forward": "pcr_primer_reference_forward",
    "pcr_primer_name_reverse": "pcr_primer_reference_reverse",
}
_PRIMER_REFERENCE_CONTEXT_RE = re.compile(
    r"\b(?:"
    r"primers?|primer\s+(?:pair|pairs|set|sets)|forward\s+primer|reverse\s+primer|"
    r"oligonucleotides?|amplicon\s+primers?|PCR\s+primers?|universal\s+primers?"
    r")\b",
    re.IGNORECASE,
)
_FORWARD_PRIMER_CONTEXT_RE = re.compile(r"\bforward\s+primer\b|\bforward\b|\bF\b", re.IGNORECASE)
_REVERSE_PRIMER_CONTEXT_RE = re.compile(r"\breverse\s+primer\b|\breverse\b|\bR\b", re.IGNORECASE)
_BOTH_PRIMER_CONTEXT_RE = re.compile(
    r"\b(?:primers|primer\s+(?:pair|pairs|set|sets)|forward\s+and\s+reverse|"
    r"forward/reverse|F/R|oligonucleotides?)\b|[A-Za-z0-9_-]+F\s*/\s*[A-Za-z0-9_-]+R",
    re.IGNORECASE,
)
_REFERENCES_HEADING_RE = re.compile(r"(?im)^\s*(?:references|literature\s+cited)\s*$")
_TEXT_REFERENCE_ENTRY_START_RE = re.compile(
    r"(?m)^\s*(?:(?:\[\s*)?(?P<number>\d{1,3})(?:\s*\])?[\.\)]?\s+)?(?P<entry>.+)"
)
_TEXT_NUMERIC_CITATION_RE = re.compile(r"\[(?P<number>\d{1,3})\]")
_TEXT_AUTHOR_YEAR_CITATION_RE = re.compile(
    r"\b(?P<author>[A-Z][A-Za-z'`-]+)(?:\s+et\s+al\.)?\s*\((?P<year>(?:19|20)\d{2}[a-z]?)\)"
)
_TEXT_PAREN_AUTHOR_YEAR_RE = re.compile(
    r"\((?P<citation>[^()]*?\b(?:19|20)\d{2}[a-z]?[^()]*?)\)"
)


def _reference_resource_from_text(entry: str) -> str:
    for match in _DOI_IN_TEXT_RE.finditer(entry):
        resource = _doi_resource(match.group(1))
        if resource:
            return resource
    cleaned = _clean_text(entry)
    return f"{cleaned[:297].rstrip()}..." if cleaned and len(cleaned) > 300 else (cleaned or entry)


def _split_text_references(full_text: str) -> list[dict[str, str]]:
    heading = None
    for match in _REFERENCES_HEADING_RE.finditer(full_text):
        heading = match
    if heading is None:
        return []
    references_text = full_text[heading.end():]
    entries: list[dict[str, str]] = []
    current_number: str | None = None
    current_lines: list[str] = []
    for raw_line in references_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _TEXT_REFERENCE_ENTRY_START_RE.match(line)
        starts_numbered_entry = bool(match and match.group("number"))
        starts_author_entry = bool(
            not starts_numbered_entry
            and re.match(r"^[A-Z][A-Za-z'`-]+,\s+(?:[A-Z]\.|[A-Z][a-z]+)", line)
            and re.search(r"\b(?:19|20)\d{2}[a-z]?\b", line[:160])
        )
        if current_lines and (starts_numbered_entry or starts_author_entry):
            entry = _clean_text(" ".join(current_lines))
            if entry:
                entries.append({"number": current_number or "", "text": entry, "resource": _reference_resource_from_text(entry)})
            current_lines = []
        if starts_numbered_entry:
            current_number = match.group("number")
            current_lines.append(match.group("entry"))
        else:
            if starts_author_entry:
                current_number = None
            current_lines.append(line)
    if current_lines:
        entry = _clean_text(" ".join(current_lines))
        if entry:
            entries.append({"number": current_number or "", "text": entry, "resource": _reference_resource_from_text(entry)})
    return entries


def _text_reference_for_numeric_marker(references: list[dict[str, str]], number: str) -> dict[str, str] | None:
    return next((entry for entry in references if entry.get("number") == number), None)


def _text_reference_for_author_year(references: list[dict[str, str]], author: str, year: str) -> dict[str, str] | None:
    year_base = year[:4]
    author_folded = author.casefold()
    for entry in references:
        text = entry["text"]
        if author_folded in text[:180].casefold() and re.search(rf"\b{re.escape(year_base)}[a-z]?\b", text[:240]):
            return entry
    return None


def _nearest_text_reference(sentence: str, references: list[dict[str, str]], anchor_pos: int) -> tuple[str, dict[str, str]] | None:
    candidates: list[tuple[int, str, dict[str, str]]] = []
    for match in _TEXT_NUMERIC_CITATION_RE.finditer(sentence):
        reference = _text_reference_for_numeric_marker(references, match.group("number"))
        if reference is not None:
            candidates.append((abs(match.start() - anchor_pos), f"ref{match.group('number')}", reference))
    for match in _TEXT_AUTHOR_YEAR_CITATION_RE.finditer(sentence):
        reference = _text_reference_for_author_year(references, match.group("author"), match.group("year"))
        if reference is not None:
            candidates.append((abs(match.start() - anchor_pos), f"{match.group('author')} {match.group('year')}", reference))
    for paren_match in _TEXT_PAREN_AUTHOR_YEAR_RE.finditer(sentence):
        citation_text = paren_match.group("citation")
        for match in re.finditer(r"(?P<author>[A-Z][A-Za-z'`-]+)(?:\s+et\s+al\.)?,?\s+(?P<year>(?:19|20)\d{2}[a-z]?)", citation_text):
            reference = _text_reference_for_author_year(references, match.group("author"), match.group("year"))
            if reference is not None:
                candidates.append((abs(paren_match.start() - anchor_pos), f"{match.group('author')} {match.group('year')}", reference))
    if not candidates:
        return None
    _, label, reference = sorted(candidates, key=lambda item: item[0])[0]
    return label, reference


def _text_citation_windows(methods_text: str) -> list[str]:
    paragraphs = [
        _clean_text(paragraph)
        for paragraph in re.split(r"\n\s*\n+", methods_text)
        if _clean_text(paragraph)
    ]
    if paragraphs:
        return paragraphs
    return [_clean_text(methods_text)] if _clean_text(methods_text) else []


def _first_primer_reference_fact(
    root: ET.Element,
    bibliography: dict[str, dict[str, str]],
    primer_name: str,
    reference_field: str,
    locator_prefix: str,
) -> RawFactCandidate | None:
    name_pattern = re.compile(r"\b" + re.escape(primer_name) + r"\b", re.IGNORECASE)
    for section, titles in _iter_leaf_sections(root):
        if not _is_method_leaf(titles):
            continue
        for node in _paragraph_like_nodes(section):
            marked_text = _text_with_citation_markers(node)
            name_match = name_pattern.search(marked_text)
            if name_match is None:
                continue
            # A real Methods paragraph routinely cites more than one thing
            # (e.g. "Similar to Zhao et al. [28] ... primers of Uni519F/806r,
            # as described in Zhao et al. [38]" -- confirmed live,
            # 10.1038/s42003-024-06136-2/PMC11009272): the FIRST citation
            # marker in the whole paragraph is not necessarily the one that
            # actually attributes the primer. The nearest marker to the
            # primer name's own position is the real one.
            markers = list(_CITATION_MARKER_RE.finditer(marked_text))
            if not markers:
                continue
            name_pos = name_match.start()
            markers.sort(key=lambda m: abs(m.start() - name_pos))
            clean_snippet = _clean_text(_CITATION_MARKER_RE.sub("", marked_text))
            for marker in markers:
                ref_id = marker.group(1)
                resource = bibliography.get(ref_id)
                if resource is None:
                    continue
                return RawFactCandidate(
                    entity_level=EntityLevel.STUDY,
                    fact_type_candidate=reference_field,
                    raw_field_name=reference_field,
                    raw_value=resource["resource"],
                    source_locator=f"{locator_prefix}:primer_reference_citation:{reference_field}",
                    support_type=SupportType.DETERMINISTICALLY_DERIVED,
                    evidence_quote=clean_snippet,
                    confidence_metadata={
                        "detector": "primer_reference_citation",
                        "primer_name": primer_name,
                        "ref_id": ref_id,
                        "citation_text": resource["citation_text"],
                    },
                )
    return None


def _nearest_bibliography_resource(
    bibliography: dict[str, dict[str, str]],
    marked_text: str,
    anchor_pos: int,
) -> tuple[str, dict[str, str]] | None:
    markers = list(_CITATION_MARKER_RE.finditer(marked_text))
    if not markers:
        return None
    markers.sort(key=lambda marker: abs(marker.start() - anchor_pos))
    for marker in markers:
        ref_id = marker.group(1)
        resource = bibliography.get(ref_id)
        if resource is not None:
            return ref_id, resource
    return None


def _primer_reference_direction_fields(marked_text: str, primer_names: dict[str, str]) -> set[str]:
    clean_text = _CITATION_MARKER_RE.sub("", marked_text)
    fields: set[str] = set()
    forward_name = (primer_names.get("pcr_primer_name_forward") or "").strip()
    reverse_name = (primer_names.get("pcr_primer_name_reverse") or "").strip()
    if forward_name and re.search(r"\b" + re.escape(forward_name) + r"\b", clean_text, re.IGNORECASE):
        fields.add("pcr_primer_reference_forward")
    if reverse_name and re.search(r"\b" + re.escape(reverse_name) + r"\b", clean_text, re.IGNORECASE):
        fields.add("pcr_primer_reference_reverse")
    if _BOTH_PRIMER_CONTEXT_RE.search(clean_text):
        fields.update(("pcr_primer_reference_forward", "pcr_primer_reference_reverse"))
    elif _FORWARD_PRIMER_CONTEXT_RE.search(clean_text):
        fields.add("pcr_primer_reference_forward")
    elif _REVERSE_PRIMER_CONTEXT_RE.search(clean_text):
        fields.add("pcr_primer_reference_reverse")
    return fields


def _primer_reference_fallback_facts(
    root: ET.Element,
    bibliography: dict[str, dict[str, str]],
    primer_names: dict[str, str],
    locator_prefix: str,
    existing_fields: set[str],
) -> list[RawFactCandidate]:
    """Fallback for primer-set citation sentences where exact extracted
    primer names are absent or not adjacent to the citation marker.

    This intentionally reuses the same JATS citation graph as
    associated_resource, but requires strong primer language in a
    Methods-like paragraph so ordinary methods citations do not all become
    primer references.
    """
    facts: list[RawFactCandidate] = []
    for section, titles in _iter_leaf_sections(root):
        if not _is_method_leaf(titles):
            continue
        for node in _paragraph_like_nodes(section):
            marked_text = _text_with_citation_markers(node)
            context_match = _PRIMER_REFERENCE_CONTEXT_RE.search(marked_text)
            if context_match is None or not _CITATION_MARKER_RE.search(marked_text):
                continue
            nearest = _nearest_bibliography_resource(bibliography, marked_text, context_match.end())
            if nearest is None:
                continue
            ref_id, resource = nearest
            clean_snippet = _clean_text(_CITATION_MARKER_RE.sub("", marked_text))
            for reference_field in sorted(_primer_reference_direction_fields(marked_text, primer_names)):
                if reference_field in existing_fields:
                    continue
                name_field = (
                    "pcr_primer_name_forward"
                    if reference_field == "pcr_primer_reference_forward"
                    else "pcr_primer_name_reverse"
                )
                if (primer_names.get(name_field) or "").strip():
                    continue
                facts.append(
                    RawFactCandidate(
                        entity_level=EntityLevel.STUDY,
                        fact_type_candidate=reference_field,
                        raw_field_name=reference_field,
                        raw_value=resource["resource"],
                        source_locator=f"{locator_prefix}:primer_reference_citation:{reference_field}:primer_context_fallback",
                        support_type=SupportType.DETERMINISTICALLY_DERIVED,
                        evidence_quote=clean_snippet,
                        confidence_metadata={
                            "detector": "primer_reference_citation_context_fallback",
                            "primer_name": primer_names.get("pcr_primer_name_forward")
                            if reference_field.endswith("_forward")
                            else primer_names.get("pcr_primer_name_reverse"),
                            "ref_id": ref_id,
                            "citation_text": resource["citation_text"],
                            "section_title": " > ".join(titles),
                        },
                    )
                )
                existing_fields.add(reference_field)
            if {"pcr_primer_reference_forward", "pcr_primer_reference_reverse"} <= existing_fields:
                return facts
    return facts


def extract_primer_reference_citations(
    xml: str, primer_names: dict[str, str], *, locator_prefix: str
) -> list[RawFactCandidate]:
    """`primer_names` maps {"pcr_primer_name_forward": <name found for this
    paper, if any>, "pcr_primer_name_reverse": <...>}. Exact primer-name
    matches are tried first. If name extraction missed a paper, a second
    pass still accepts strong primer-pair/forward/reverse language linked
    to a real JATS bibliography citation in Methods. Returns at most one
    pcr_primer_reference_forward and one pcr_primer_reference_reverse fact."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return []

    bibliography = _bibliography_resources(root)
    if not bibliography:
        return []

    facts: list[RawFactCandidate] = []
    existing_fields: set[str] = set()
    for name_field, reference_field in _PRIMER_NAME_TO_REFERENCE_FIELD.items():
        primer_name = (primer_names.get(name_field) or "").strip()
        if not primer_name:
            continue
        fact = _first_primer_reference_fact(root, bibliography, primer_name, reference_field, locator_prefix)
        if fact is not None:
            facts.append(fact)
            existing_fields.add(reference_field)
    facts.extend(_primer_reference_fallback_facts(root, bibliography, primer_names, locator_prefix, existing_fields))
    return facts


def extract_primer_reference_citations_from_text(
    text: str, primer_names: dict[str, str], *, locator_prefix: str
) -> list[RawFactCandidate]:
    """Best-effort primer-reference extraction for local PDFs/plain text.

    The JATS path above is preferred whenever XML is available because it
    has exact citation graph links. PDFs do not preserve those links, so
    this fallback reconstructs enough: primer-context sentences in the
    methods text + numeric or author/year markers + a parsed References
    section. DOI output still uses the same `doi: ...` normalization; if
    no DOI is present, the matched reference text is kept as the lead.
    """
    references = _split_text_references(text)
    if not references:
        return []
    references_heading = _REFERENCES_HEADING_RE.search(text)
    methods_text = text[: references_heading.start()] if references_heading else text
    citation_windows = _text_citation_windows(methods_text)

    facts: list[RawFactCandidate] = []
    existing_fields: set[str] = set()

    for name_field, reference_field in _PRIMER_NAME_TO_REFERENCE_FIELD.items():
        primer_name = (primer_names.get(name_field) or "").strip()
        if not primer_name:
            continue
        name_pattern = re.compile(r"\b" + re.escape(primer_name) + r"\b", re.IGNORECASE)
        for sentence in citation_windows:
            name_match = name_pattern.search(sentence)
            if name_match is None:
                continue
            nearest = _nearest_text_reference(sentence, references, name_match.start())
            if nearest is None:
                continue
            ref_id, reference = nearest
            facts.append(
                RawFactCandidate(
                    entity_level=EntityLevel.STUDY,
                    fact_type_candidate=reference_field,
                    raw_field_name=reference_field,
                    raw_value=reference["resource"],
                    source_locator=f"{locator_prefix}:primer_reference_citation_text:{reference_field}",
                    support_type=SupportType.DETERMINISTICALLY_DERIVED,
                    evidence_quote=sentence,
                    confidence_metadata={
                        "detector": "primer_reference_citation_text",
                        "primer_name": primer_name,
                        "ref_id": ref_id,
                        "citation_text": reference["text"],
                    },
                )
            )
            existing_fields.add(reference_field)
            break

    for sentence in citation_windows:
        if {"pcr_primer_reference_forward", "pcr_primer_reference_reverse"} <= existing_fields:
            break
        context_match = _PRIMER_REFERENCE_CONTEXT_RE.search(sentence)
        if context_match is None:
            continue
        nearest = _nearest_text_reference(sentence, references, context_match.end())
        if nearest is None:
            continue
        ref_id, reference = nearest
        for reference_field in sorted(_primer_reference_direction_fields(sentence, primer_names)):
            if reference_field in existing_fields:
                continue
            name_field = (
                "pcr_primer_name_forward"
                if reference_field == "pcr_primer_reference_forward"
                else "pcr_primer_name_reverse"
            )
            if (primer_names.get(name_field) or "").strip():
                continue
            facts.append(
                RawFactCandidate(
                    entity_level=EntityLevel.STUDY,
                    fact_type_candidate=reference_field,
                    raw_field_name=reference_field,
                    raw_value=reference["resource"],
                    source_locator=f"{locator_prefix}:primer_reference_citation_text:{reference_field}:primer_context_fallback",
                    support_type=SupportType.DETERMINISTICALLY_DERIVED,
                    evidence_quote=sentence,
                    confidence_metadata={
                        "detector": "primer_reference_citation_text_context_fallback",
                        "primer_name": primer_names.get("pcr_primer_name_forward")
                        if reference_field.endswith("_forward")
                        else primer_names.get("pcr_primer_name_reverse"),
                        "ref_id": ref_id,
                        "citation_text": reference["text"],
                    },
                )
            )
            existing_fields.add(reference_field)
    return facts


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
