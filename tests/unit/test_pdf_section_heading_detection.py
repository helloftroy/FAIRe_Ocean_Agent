"""Unit tests for local-PDF heading detection using synthetic page text --
deliberately separate from test_pdf_extraction.py, whose whole module is
skipped unless real, gitignored publisher PDFs are present locally. These
tests need no PDF fixture at all, so they run everywhere (CI, a fresh
clone, the cluster)."""
from fair_ocean_agent.extraction.pdf import (
    PdfPageText,
    _is_generic_methods_subheading,
    split_pdf_sections,
)


def test_split_pdf_sections_recognizes_a_list_conjunction_subheading_with_commas():
    """Real gap found live (10.3390/microorganisms10030558, STUDY-0049c7972ece):
    "2.2. DNA Extraction, Amplification, and Sequence Analysis" was never
    recognized as a heading -- a bare comma anywhere unconditionally
    rejected _is_generic_methods_subheading's candidate check, even though
    "X, Y, and Z" is an ordinary way to title a methods subsection covering
    several steps. Left the whole Methods section (site description,
    seawater temperature, DNA extraction, PCR, sequencing, bioinformatics)
    merged into one 40+-sentence undifferentiated paragraph instead of
    splitting at this real subsection boundary."""
    page_text = (
        "2. Materials and Methods\n"
        "2.1. Sample Collection\n"
        "Seawater temperature was 28.1 C. Seawater samples were collected at the surface.\n"
        "2.2. DNA Extraction, Amplification, and Sequence Analysis\n"
        "Filters were cut into small pieces, and DNA was extracted using a commercial kit.\n"
    )
    pages = [PdfPageText(page_number=1, text=page_text)]

    sections = split_pdf_sections(pages)

    titles = [section.title for section in sections]
    assert "DNA Extraction, Amplification, and Sequence Analysis" in titles
    by_title = {section.title: section.text for section in sections}
    assert "Seawater temperature" in by_title["Sample Collection"]
    assert "DNA was extracted" not in by_title["Sample Collection"]


def test_is_generic_methods_subheading_still_rejects_a_non_list_comma_fragment():
    """The list-conjunction allowance must stay narrow: a comma-containing
    fragment that isn't a genuine "X, Y, and Z" list (no trailing "and ..."
    clause) must still be rejected, even when it would otherwise pass every
    other check (title-case, no digits, short enough)."""
    assert _is_generic_methods_subheading("Sample Processing, Storage Conditions") is False
