"""Conservative extraction of assay-target taxa from publication metadata.

`targetTaxonomicAssay` is easy to overfill from a whole paper: ecological
scope, background taxa, result taxa, and reference taxa all look tempting.
This module therefore reads only title/abstract/keywords from publication
metadata and emits review-required candidate facts for the assay target
taxa, preserving the source phrase and evidence text.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from fair_ocean_agent.database.enums import EntityLevel, SupportType
from fair_ocean_agent.sources.base import RawFactCandidate

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_TAG_RE = re.compile(r"<[^>]+>")
_ASSAY_CONTEXT_RE = re.compile(
    r"\b(?:assays?|primers?|probes?|target(?:ed|ing)?|amplif(?:y|ied|ies|ication)|"
    r"metabarcod(?:e|ed|ing)|amplicon|pcr|qpcr|ddpcr|detect(?:ed|ion|ing)?)\b",
    re.IGNORECASE,
)
_TARGET_PHRASE_RE = re.compile(
    r"\b(?:target(?:ed|ing)?|designed\s+to\s+(?:amplify|detect)|"
    r"amplif(?:y|ied|ies)|detect(?:ed|ing)?)\s+"
    r"(?:the\s+)?(?P<phrase>[^.;:()]{0,140})",
    re.IGNORECASE,
)
_SCIENTIFIC_NAME_RE = re.compile(
    r"\b(?P<value>[A-Z][a-z]{2,}(?:\s+(?:[a-z][a-z-]{2,}|sp\.|spp\.)){1,2})\b"
)
_SCIENTIFIC_NAME_FALSE_POSITIVES = frozenset(
    {
        "Background coral",
        "Background coral reefs",
        "National Science",
        "Great Barrier",
        "Florida Keys",
        "Flower Garden",
        "Supplemental Information",
    }
)
_TAXON_PHRASES: tuple[re.Pattern[str], ...] = tuple(
    re.compile(rf"\b{phrase}\b", re.IGNORECASE)
    for phrase in (
        "crustose coralline algae",
        "red algae",
        "green algae",
        "brown algae",
        "diatoms",
        "dinoflagellates",
        "cyanobacteria",
        "bacteria",
        "archaea",
        "fungi",
        "eukaryotes",
        "eukaryotic",
        "Eukaryota",
        "Metazoa",
        "Animalia",
        "Plantae",
        "Viridiplantae",
        "Chordata",
        "Chondrichthyes",
        "Actinopterygii",
        "Mammalia",
        "Aves",
        "Amphibia",
        "Reptilia",
    )
)
_NON_TAXON_KEYWORD_RE = re.compile(
    r"\b(?:metabarcoding|barcoding|environmental\s+dna|edna|otu|asv|"
    r"recruitment|settlement|monitoring|biodiversity|sequencing|amplicon)\b",
    re.IGNORECASE,
)


def _element_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return " ".join(part.strip() for part in element.itertext() if part.strip())


def _strip_markup(value: str) -> str:
    return " ".join(_TAG_RE.sub(" ", value).split())


def _jats_metadata(fulltext_xml: str | None) -> list[tuple[str, str, str]]:
    if not fulltext_xml:
        return []
    try:
        root = ET.fromstring(fulltext_xml)
    except ET.ParseError:
        return []

    rows: list[tuple[str, str, str]] = []
    title = _element_text(root.find(".//article-title"))
    if title:
        rows.append(("title", "jats:article-title", title))
    for index, abstract in enumerate(root.findall(".//abstract"), start=1):
        text = _element_text(abstract)
        if text:
            rows.append(("abstract", f"jats:abstract[{index}]", text))
    for index, keyword in enumerate(root.findall(".//kwd"), start=1):
        text = _element_text(keyword)
        if text:
            rows.append(("keyword", f"jats:kwd[{index}]", text))
    return rows


def _crossref_metadata(crossref_raw: dict | None) -> list[tuple[str, str, str]]:
    if not crossref_raw:
        return []

    rows: list[tuple[str, str, str]] = []
    for index, title in enumerate(crossref_raw.get("title") or [], start=1):
        if title:
            rows.append(("title", f"crossref:title[{index}]", _strip_markup(str(title))))
    abstract = crossref_raw.get("abstract")
    if abstract:
        rows.append(("abstract", "crossref:abstract", _strip_markup(str(abstract))))
    for index, subject in enumerate(crossref_raw.get("subject") or [], start=1):
        if subject:
            rows.append(("keyword", f"crossref:subject[{index}]", _strip_markup(str(subject))))
    return rows


def _taxon_mentions(text: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for pattern in _TAXON_PHRASES:
        for match in pattern.finditer(text):
            value = match.group(0).strip(" .;,")
            key = value.casefold()
            if key and key not in seen:
                seen.add(key)
                values.append(value)
    for match in _SCIENTIFIC_NAME_RE.finditer(text):
        value = match.group("value").strip(" .;,")
        if value in _SCIENTIFIC_NAME_FALSE_POSITIVES:
            continue
        key = value.casefold()
        if key and key not in seen:
            seen.add(key)
            values.append(value)
    return values


def _target_phrase_taxa(sentence: str) -> list[str]:
    values: list[str] = []
    for match in _TARGET_PHRASE_RE.finditer(sentence):
        phrase = re.split(
            r"\b(?:for|using|with|via|based|to|from|in)\b",
            match.group("phrase"),
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        values.extend(_taxon_mentions(phrase))
    return values


def _taxa_from_metadata_item(kind: str, text: str) -> tuple[list[str], str | None]:
    normalized = " ".join(text.split())
    if kind == "keyword":
        if _NON_TAXON_KEYWORD_RE.search(normalized):
            return [], None
        values = _taxon_mentions(normalized)
        return values, normalized if values else None

    values: list[str] = []
    evidence: str | None = None
    for sentence in _SENTENCE_SPLIT_RE.split(normalized):
        sentence = sentence.strip()
        if "without assay context" in sentence.casefold():
            continue
        if not sentence or not _ASSAY_CONTEXT_RE.search(sentence):
            continue
        # Deliberately NOT a `_target_phrase_taxa(sentence) or
        # _taxon_mentions(sentence)` fallback: scanning a whole sentence for
        # "looks like a binomial name" (_SCIENTIFIC_NAME_RE's Capitalized-
        # word + lowercase-word pattern) matches ordinary English far too
        # often -- confirmed on a real paper (PLOS ONE 10.1371/
        # journal.pone.0303937), whose abstract mentions "PCR" and
        # "amplicon" in sentences that have nothing to do with an assay's
        # target taxon at all, false-positiving on "Diversity studies" (the
        # sentence's own opening two words) and "The device was" (a
        # different sentence's opening three words). Only the narrower
        # target-phrase capture (an explicit "targeting X"/"designed to
        # amplify X"/etc. phrase) is trusted for sentence-level extraction;
        # a sentence with assay-context but no such phrase contributes
        # nothing, rather than guessing from its raw text.
        sentence_values = _target_phrase_taxa(sentence)
        if not sentence_values:
            continue
        values.extend(sentence_values)
        evidence = sentence
    return values, evidence


def extract_assay_target_taxa_from_publication_metadata(
    fulltext_xml: str | None,
    crossref_raw: dict | None,
    *,
    locator_prefix: str,
) -> list[RawFactCandidate]:
    values: list[str] = []
    evidence_quotes: list[str] = []
    matches: list[dict] = []
    seen: set[str] = set()

    for kind, locator, text in [*_jats_metadata(fulltext_xml), *_crossref_metadata(crossref_raw)]:
        item_values, evidence = _taxa_from_metadata_item(kind, text)
        if not item_values or not evidence:
            continue
        for value in item_values:
            key = value.casefold()
            if key in seen:
                continue
            seen.add(key)
            values.append(value)
            if evidence not in evidence_quotes:
                evidence_quotes.append(evidence)
            matches.append(
                {
                    "matched_value": value,
                    "metadata_kind": kind,
                    "source_locator": f"{locator_prefix}:{locator}",
                }
            )

    if not values:
        return []
    return [
        RawFactCandidate(
            entity_level=EntityLevel.STUDY,
            fact_type_candidate="assay_target_taxa",
            raw_field_name="assay_target_taxa",
            raw_value=" | ".join(values),
            source_locator=f"{locator_prefix}:target_taxonomic_assay:title_abstract_keywords",
            support_type=SupportType.DETERMINISTICALLY_DERIVED,
            evidence_quote=" | ".join(evidence_quotes),
            confidence_metadata={
                "detector": "publication_metadata_assay_taxa",
                "scope": "title_abstract_keywords_only",
                "matches": matches,
            },
        )
    ]
