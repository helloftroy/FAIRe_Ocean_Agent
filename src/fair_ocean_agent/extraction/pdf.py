"""Text extraction and section selection for local main-paper PDFs.

The online path prefers JATS XML because it carries real structure. Local
PDFs do not, so this module reconstructs just enough structure for the
existing extraction pipeline: clean page text, remove repeated page
furniture, split likely section headings, and return relevant sections in
the same ``{"title": ..., "text": ...}`` shape as
extraction.sections.select_relevant_sections().
"""
from __future__ import annotations

import io
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from fair_ocean_agent.extraction.sections import (
    MIN_FRAGMENT_CHARS,
    RELEVANT_SECTION_TITLE_PATTERNS,
    RESULT_DISCUSSION_SECTION_TITLE_PATTERNS,
)

_PAGE_BREAK = "\n\n\f\n\n"
_LIGATURES = str.maketrans(
    {
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬀ": "ff",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "\u00a0": " ",
    }
)
_HYPHENATED_LINEBREAK_RE = re.compile(r"(?<=[A-Za-z])-\n(?=[a-z])")
_PDF_HEADER_FOOTER_RE = re.compile(
    r"^(?:"
    r"\d+\s*$|"
    r".*\bdoi\b.*|"
    r"https?://.*|"
    r".*\bfrontiersin\.org\b.*|"
    r".*\bPLOS\s+ONE\b.*|"
    r".*\bCommunications\s+Biology\b.*|"
    r".*\bPeerJ\b.*DOI.*|"
    r".*\|\s*\d+\s*$"
    r")",
    re.IGNORECASE,
)
_RESULTS_OR_END_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*\s+)?(?:results?|discussion|conclusions?|references|bibliography)\b",
    re.IGNORECASE,
)
_METHODS_ROOT_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*\s+)?(?:materials?\s+and\s+methods?|methods?|methodology|experimental\s+procedures?)\b",
    re.IGNORECASE,
)
_KNOWN_HEADING_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*\s+)?(?:"
    r"abstract|introduction|materials?\s+and\s+methods?|methods?|methodology|experimental\s+procedures?|"
    r"study\s+(?:area|site)s?(?:\s+and\s+sampling\s+campaign)?|sampling(?:\s+\w+){0,5}|"
    r"sample\s+collection(?:\s+\w+){0,4}|sediment\s+sampling|water\s+sampling|"
    r"dna\s+(?:extraction|isolation|amplification|quantification)(?:\s+\w+){0,5}|"
    r"rna\s+(?:extraction|isolation|sequencing)(?:\s+\w+){0,5}|"
    r"pcr(?:\s+\w+){0,8}|qpcr(?:\s+\w+){0,6}|primer(?:\s+\w+){0,6}|"
    r"library(?:\s+\w+){0,8}|sequenc(?:ing|e)(?:\s+\w+){0,8}|"
    r"bioinformatic(?:s)?(?:\s+\w+){0,8}|data\s+analys(?:is|es)(?:\s+\w+){0,8}|"
    r"taxonom(?:y|ic)(?:\s+\w+){0,8}|quality\s+control(?:\s+\w+){0,5}|"
    r"data\s+availability|funding|acknowledg(?:e)?ments?|rights(?:\s+and\s+permissions)?|"
    r"results?|discussion|conclusions?|references|bibliography"
    r")$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PdfPageText:
    page_number: int
    text: str


@dataclass(frozen=True)
class PdfSection:
    title: str
    text: str
    start_page: int | None = None
    end_page: int | None = None


def extract_pdf_pages(content: bytes | str | Path | BinaryIO) -> list[PdfPageText]:
    """Extract text page-by-page from a PDF using pypdf.

    The plain extractor is the primary source because it usually gives
    better sentence order for two-column journal PDFs than pypdf's layout
    mode. Layout mode is still tried per page and selected only when it
    appears meaningfully cleaner.
    """
    from pypdf import PdfReader

    reader = PdfReader(_as_binary_stream(content))
    raw_pages: list[str] = []
    for page in reader.pages:
        plain = page.extract_text() or ""
        try:
            layout = page.extract_text(extraction_mode="layout") or ""
        except TypeError:
            layout = ""
        raw_pages.append(_choose_better_page_text(plain, layout))

    repeated = _repeated_page_furniture(raw_pages)
    return [
        PdfPageText(page_number=index + 1, text=_clean_page_text(raw, repeated))
        for index, raw in enumerate(raw_pages)
    ]


def extract_pdf_text(content: bytes | str | Path | BinaryIO) -> str:
    pages = extract_pdf_pages(content)
    return _PAGE_BREAK.join(page.text for page in pages if page.text).strip()


def extract_pdf_sections(content: bytes | str | Path | BinaryIO, max_chars: int = 40000) -> list[dict]:
    """Return relevant PDF sections in the same shape as JATS extraction.

    Includes subsections under a Methods-like heading until a Results,
    Discussion, Conclusions, or References heading is reached.
    """
    sections = split_pdf_sections(extract_pdf_pages(content))
    selected: list[dict] = []
    total_chars = 0
    under_methods = False
    for section in sections:
        if total_chars >= max_chars:
            break
        title = section.title
        if _is_result_or_discussion_title(title) or _RESULTS_OR_END_RE.search(title):
            under_methods = False
            continue
        if _METHODS_ROOT_RE.search(title):
            under_methods = True
        relevant = under_methods or _is_relevant_title(title)
        if not relevant or not section.text:
            continue
        remaining = max_chars - total_chars
        if remaining < MIN_FRAGMENT_CHARS:
            continue
        text = section.text[:remaining]
        selected.append({"title": title, "text": text, "page_start": section.start_page, "page_end": section.end_page})
        total_chars += len(text)
    return selected


def split_pdf_sections(pages: list[PdfPageText]) -> list[PdfSection]:
    sections: list[PdfSection] = []
    current_title = "Front Matter"
    current_lines: list[str] = []
    start_page: int | None = pages[0].page_number if pages else None
    end_page: int | None = start_page
    under_methods = False

    def flush() -> None:
        nonlocal current_lines
        text = _paragraph_text(current_lines)
        if text or _METHODS_ROOT_RE.search(current_title) or _RESULTS_OR_END_RE.search(current_title):
            sections.append(PdfSection(current_title, text, start_page, end_page))
        current_lines = []

    for page in pages:
        for raw_line in page.text.splitlines():
            line = _clean_line(raw_line)
            if not line:
                continue
            if _is_heading_line(line, under_methods=under_methods):
                flush()
                current_title = _normalize_heading(line)
                start_page = page.page_number
                end_page = page.page_number
                if _RESULTS_OR_END_RE.search(current_title):
                    under_methods = False
                elif _METHODS_ROOT_RE.search(current_title):
                    under_methods = True
                continue
            current_lines.append(line)
            end_page = page.page_number
    flush()
    return sections


def _as_binary_stream(content: bytes | str | Path | BinaryIO) -> BinaryIO:
    if isinstance(content, bytes):
        return io.BytesIO(content)
    if isinstance(content, (str, Path)):
        return Path(content).open("rb")
    return content


def _choose_better_page_text(plain: str, layout: str) -> str:
    if not layout.strip():
        return plain
    plain_score = _text_quality_score(plain)
    layout_score = _text_quality_score(layout)
    return layout if layout_score > plain_score + 0.15 else plain


def _text_quality_score(text: str) -> float:
    cleaned = _normalize_unicode(text)
    if not cleaned.strip():
        return 0.0
    words = re.findall(r"[A-Za-z][A-Za-z-]{2,}", cleaned)
    spaces = cleaned.count(" ")
    lines = [line for line in cleaned.splitlines() if line.strip()]
    very_wide_spaces = sum(1 for line in lines if re.search(r"\S\s{8,}\S", line))
    joined_words = len(re.findall(r"[a-z]{18,}", cleaned))
    return len(words) / max(len(cleaned), 1) + spaces / max(len(cleaned), 1) * 0.1 - very_wide_spaces * 0.02 - joined_words * 0.01


def _normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text.translate(_LIGATURES))


def _repeated_page_furniture(raw_pages: list[str]) -> set[str]:
    counts: Counter[str] = Counter()
    for raw in raw_pages:
        seen_on_page: set[str] = set()
        for line in raw.splitlines():
            cleaned = _clean_line(line).casefold()
            if cleaned:
                seen_on_page.add(cleaned)
        counts.update(seen_on_page)
    threshold = max(3, len(raw_pages) // 3)
    return {line for line, count in counts.items() if count >= threshold and len(line) <= 120}


def _clean_page_text(raw: str, repeated_lines: set[str]) -> str:
    text = _normalize_unicode(raw)
    text = _HYPHENATED_LINEBREAK_RE.sub("", text)
    lines = []
    for line in text.splitlines():
        cleaned = _clean_line(line)
        if not cleaned:
            continue
        if cleaned.casefold() in repeated_lines:
            continue
        if _PDF_HEADER_FOOTER_RE.match(cleaned):
            continue
        lines.append(cleaned)
    return "\n".join(lines)


def _clean_line(line: str) -> str:
    line = _normalize_unicode(line)
    line = re.sub(r"\s+", " ", line).strip()
    line = re.sub(r"(?<=\w)\s+([,.;:])", r"\1", line)
    line = re.sub(r"([([])\s+", r"\1", line)
    line = re.sub(r"\s+([])])", r"\1", line)
    return line


def _is_heading_line(line: str, *, under_methods: bool = False) -> bool:
    clean = _normalize_heading(line)
    if not clean or len(clean) > 140:
        return False
    if clean.endswith((".", ",", ";", ":")) and not clean.isupper():
        return False
    if _KNOWN_HEADING_RE.match(clean):
        return True
    if under_methods and _is_generic_methods_subheading(clean):
        return True
    return False


def _is_generic_methods_subheading(line: str) -> bool:
    words = line.split()
    if not (2 <= len(words) <= 10):
        return False
    if any(char in line for char in ".,;:()=#"):
        return False
    if re.search(r"\d|[<>≤≥~±×]|[α-ωΑ-Ωµ]", line):
        return False
    if not line[:1].isupper():
        return False
    letters = re.findall(r"[A-Za-z]", line)
    if not letters:
        return False
    titleish_words = sum(1 for word in words if word[:1].isupper() or word.isupper())
    return titleish_words >= max(1, len(words) // 2)


def _normalize_heading(line: str) -> str:
    line = _clean_line(line).strip()
    line = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", line)
    return line.strip()


def _paragraph_text(lines: list[str]) -> str:
    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        if _is_heading_line(line, under_methods=True):
            if current:
                paragraphs.append(" ".join(current))
                current = []
            paragraphs.append(line)
            continue
        current.append(line)
    if current:
        paragraphs.append(" ".join(current))
    text = "\n\n".join(paragraphs)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def _is_relevant_title(title: str) -> bool:
    return bool(title and any(pattern.search(title) for pattern in RELEVANT_SECTION_TITLE_PATTERNS))


def _is_result_or_discussion_title(title: str) -> bool:
    return bool(title and any(pattern.search(title) for pattern in RESULT_DISCUSSION_SECTION_TITLE_PATTERNS))
