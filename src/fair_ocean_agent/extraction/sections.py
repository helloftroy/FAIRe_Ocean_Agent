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
        r"taxonom",  # "Taxonomic assignment"/"Taxonomy" -- where otu_db/tax_assign_cat/
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


def _title_for(sec: ET.Element) -> str:
    title_el = sec.find("title")
    if title_el is None:
        return ""
    return " ".join(t.strip() for t in title_el.itertext() if t.strip())


def _is_relevant_title(title: str) -> bool:
    return bool(title and any(pattern.search(title) for pattern in RELEVANT_SECTION_TITLE_PATTERNS))


def _is_result_or_discussion_title(title: str) -> bool:
    return bool(title and any(pattern.search(title) for pattern in RESULT_DISCUSSION_SECTION_TITLE_PATTERNS))


def _iter_leaf_sections(element: ET.Element, ancestor_titles: tuple[str, ...] = ()) -> Iterable[tuple[ET.Element, tuple[str, ...]]]:
    if element.tag != "sec":
        for child in list(element):
            yield from _iter_leaf_sections(child, ancestor_titles)
        return

    title = _title_for(element)
    path = (*ancestor_titles, title) if title else ancestor_titles
    child_sections = [child for child in list(element) if child.tag == "sec"]
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

        text = " ".join(t.strip() for t in sec.itertext() if t.strip())
        if not text:
            continue

        remaining = max_chars - total_chars
        if remaining < MIN_FRAGMENT_CHARS:
            continue
        truncated_text = text[:remaining]
        sections.append({"title": title, "text": truncated_text})
        total_chars += len(truncated_text)

    return sections
