"""Deterministic section selection (section 10): pick out just the
sections relevant to metadata extraction (Methods, Data Availability,
etc.) from a JATS full-text XML document, so the LLM only ever sees
bounded, relevant text instead of an entire paper. This is a fixed-pattern
pass, not an LLM call -- section 10 allows an "LLM fallback only when
necessary", but Europe PMC's fullTextXML is consistently JATS-structured
with a <title> per <sec>, so a deterministic pass covers it; a fallback
selector is not implemented (out of scope for now, add one if deterministic
coverage turns out to be insufficient on other section-heading styles).
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Iterable

RELEVANT_SECTION_TITLE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"method",
        r"material",
        r"procedure",  # some journals (notably Cell Press) title their Methods
                       # section "Experimental Procedures" instead of "Methods" --
                       # found via live validation against a real paper (PMC7820986)
                       # that this pattern list previously missed entirely.
        r"\bdna\b",  # method subsections such as "DNA degradation experiment" and
                     # "DNA quantification and quality assessment" carry sample
                     # storage, extraction-input, concentration, and purity facts
                     # even when the heading does not say "extraction".
        r"sampl",
        r"study (area|site)",  # common location/environment methods headings
                               # containing station ranges, depth ranges, cruise
                               # context, and collection platform details.
        r"environmental",
        r"extraction",
        r"fixation",
        r"sorting",
        r"pcr",
        r"qpcr",
        r"quantitative pcr",
        r"amplif",
        r"assay",
        r"primer",
        r"librar",  # "Library preparation" -- distinct subsection from "sequencing"
                    # in many papers, and this taxonomy's atomic lib_conc/adapter
                    # fields specifically live there, not under "sequencing".
        r"sequenc",
        r"data analys",  # e.g. "16S data analysis", "Microbiome data analysis",
                         # "Transcriptomic data analyses": where trimming,
                         # ASV/OTU inference, read mapping, and databases are
                         # often reported without "bioinformatics" in the title.
        r"bioinformatic",
        r"taxonom",  # "Taxonomic assignment"/"Taxonomy" -- where otu_db/
                     # scientificName-level facts are reported, distinct from the
                     # broader "bioinformatic" pattern (e.g. "Taxonomy" alone, with no
                     # "bioinformatic" in the title at all).
        r"standard curve",
        r"data availab",
        r"supplementary method",
        r"quality control",
    )
]
RESULT_DISCUSSION_SECTION_TITLE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bresults?\b",
        r"\bdiscussion\b",
        r"\bconclusions?\b",
    )
]

# A truncated fragment below this length is rarely worth its own LLM call --
# not enough context for the model to report much, and it still costs a
# full extraction call. Skip it (leave the remaining budget for whatever
# section comes next in document order) rather than send a near-empty
# section through the pipeline.
MIN_FRAGMENT_CHARS = 500


# A citation/reference-number marker (e.g. a superscript "71" linking to
# the bibliography) -- confirmed live (10.1038/s42003-024-06136-2's real
# JATS XML: `SILVA 138.1 Release<sup><xref rid="CR71"
# ref-type="bibr">71</xref></sup>`) that Europe PMC's fullTextXML embeds
# these as plain sibling text with no visual distinction from real
# sentence content once flattened via a naive itertext() call, corrupting
# extracted values like otu_db's "SILVA 138.1 Release" into "SILVA 138.1
# Release 71" and similarly contaminating whatever text immediately
# follows a tool/database name cited this way. Other real ref-types found
# in this same document (fig/table/aff/author-notes/supplementary-material)
# are left alone -- their inline text (e.g. "Fig. 2") is meaningful
# sentence content, unlike a bare citation number.
_CITATION_XREF_REF_TYPES = frozenset({"bibr"})
# When a citation number sits inside literal parentheses in the source
# text (the parens are the paragraph's own text, not part of the stripped
# xref), stripping the number alone leaves a dangling empty "( )" behind --
# confirmed live (10.1002/ece3.6071's own sterilise_method text). Purely
# cosmetic (an LLM has no reason to ever quote "( )" as a real value), but
# cleaned up anyway rather than leaving visible noise in extracted text.
_EMPTY_PARENTHETICAL_RE = re.compile(r"\(\s*\)")
# Same gap, square brackets -- confirmed live (10.3390/microorganisms10030558's
# own methods text: "amplified with primers 338F and 806R [ ]" and
# "primers cbbL_K2f and cbbL_V2r [ ]", both real citation markers stripped
# down to an empty bracket pair immediately after the primer names).
_EMPTY_BRACKET_RE = re.compile(r"\[\s*\]")


def _itertext_excluding_citations(element: ET.Element) -> Iterable[str]:
    """Same traversal as Element.itertext(), except the inner text of a
    <xref ref-type="bibr"> is skipped -- only the citation number itself
    is dropped, its own tail text (whatever immediately follows it, e.g.
    a period or space) is still yielded in place."""
    if element.text:
        yield element.text
    for child in element:
        if not (child.tag == "xref" and child.get("ref-type") in _CITATION_XREF_REF_TYPES):
            yield from _itertext_excluding_citations(child)
        if child.tail:
            yield child.tail


def _title_for(sec: ET.Element) -> str:
    title_el = sec.find("title")
    if title_el is None:
        return ""
    return " ".join(t.strip() for t in _itertext_excluding_citations(title_el) if t.strip())


def _is_relevant_title(title: str) -> bool:
    return bool(title and any(pattern.search(title) for pattern in RELEVANT_SECTION_TITLE_PATTERNS))


def _is_result_or_discussion_title(title: str) -> bool:
    return bool(title and any(pattern.search(title) for pattern in RESULT_DISCUSSION_SECTION_TITLE_PATTERNS))


def _iter_leaf_sections(element: ET.Element, ancestor_titles: tuple[str, ...] = ()) -> Iterable[tuple[ET.Element, tuple[str, ...]]]:
    # A real live audit (10.7717/peerj.333, PeerJ) caught this: journals
    # that put "Additional Information and Declarations" content --
    # competing interests, author contributions, ethical/permit
    # approvals, and (the one that actually matters here) a "DNA
    # Deposition" section stating the paper's own accession numbers --
    # encode each as a separate, titled <fn-group> back-matter element,
    # not a <sec>. The generic "recurse into every non-<sec> element"
    # branch below used to walk straight through <fn-group> without ever
    # yielding it, so this content was structurally invisible to every
    # consumer of select_relevant_sections regardless of title relevance.
    # Treated as its own leaf, gated by the exact same _is_relevant_title
    # check as everything else -- "DNA Deposition" matches (via the
    # existing \bdna\b pattern) and gets included, "Competing Interests"/
    # "Author Contributions"/"Field Study Permissions" don't and stay
    # excluded, with no new heuristic needed.
    if element.tag == "fn-group":
        yield element, ancestor_titles
        return
    if element.tag != "sec":
        for child in list(element):
            yield from _iter_leaf_sections(child, ancestor_titles)
        return

    title = _title_for(element)
    path = (*ancestor_titles, title) if title else ancestor_titles
    # A real live audit (10.7717/peerj.333): the four declaration
    # fn-groups above are themselves wrapped in one <sec> ("Additional
    # Information and Declarations") with no <sec> children of its own --
    # child_sections used to come back empty, so this wrapper was treated
    # as an atomic leaf and tested against its OWN (irrelevant) title,
    # never recursing far enough to reach "DNA Deposition" at all. A <sec>
    # with fn-group children is recursed into exactly like one with <sec>
    # children, rather than being treated as a leaf.
    child_sections = [child for child in list(element) if child.tag in ("sec", "fn-group")]
    if not child_sections:
        yield element, ancestor_titles
        return
    for child in child_sections:
        yield from _iter_leaf_sections(child, path)


def select_relevant_sections(fulltext_xml: str, max_chars: int = 40000) -> list[dict]:
    """Returns [{"title": str, "text": str}, ...] for <sec> elements whose
    <title> matches a relevant-section pattern, in document order, with
    combined text truncated to max_chars total across all selected
    sections -- bounding what gets sent to the LLM (section 10: "Avoids
    sending an entire long paper"). Returns [] on unparseable XML rather
    than raising, since a malformed document simply has nothing to
    extract from.

    max_chars defaults to 40000 (raised from an original 20000): the
    FAIRe-aware taxonomy (extraction/faire_fields.py) targets fields spread
    across more, often separately-titled leaf sections than the original
    prompt did (Sampling, DNA extraction, PCR, Library prep, Sequencing,
    Bioinformatics, Taxonomic assignment can each be their own <sec> in a
    real paper) -- the original budget risked truncating away exactly the
    later sections (bioinformatics/taxonomy) this expansion is meant to
    reach, since sections are accepted in document order until the budget
    runs out."""
    try:
        root = ET.fromstring(fulltext_xml)
    except ET.ParseError:
        return []

    sections: list[dict] = []
    total_chars = 0
    for sec, ancestor_titles in _iter_leaf_sections(root):
        if total_chars >= max_chars:
            break

        # JATS nests subsections (<sec><title>Materials and Methods</title>
        # <sec><title>Sampling</title>...</sec><sec><title>DNA
        # extraction...</title></sec></sec>) -- a parent sec's itertext()
        # already includes all its children's text, so processing both
        # would duplicate every subsection's content. Only leaf sections
        # are considered. A leaf can inherit relevance from a Methods-like
        # parent: real papers often use child headings like "Caribbean spawn
        # I" that are only recognizable as methods from the parent section.
        title = _title_for(sec)
        relevant_by_title = _is_relevant_title(title)
        relevant_by_parent = any(_is_relevant_title(parent_title) for parent_title in ancestor_titles)
        under_results_or_discussion = any(_is_result_or_discussion_title(parent_title) for parent_title in ancestor_titles)
        if under_results_or_discussion:
            continue
        if not title or not (relevant_by_title or relevant_by_parent):
            continue

        text = " ".join(t.strip() for t in _itertext_excluding_citations(sec) if t.strip())
        text = " ".join(_EMPTY_PARENTHETICAL_RE.sub(" ", text).split())
        text = " ".join(_EMPTY_BRACKET_RE.sub(" ", text).split())
        if not text:
            continue

        remaining = max_chars - total_chars
        if remaining < MIN_FRAGMENT_CHARS:
            continue
        truncated_text = text[:remaining]
        sections.append({"title": title, "text": truncated_text})
        total_chars += len(truncated_text)

    return sections
