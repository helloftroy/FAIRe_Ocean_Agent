from pathlib import Path

import pytest

from fair_ocean_agent.extraction.pdf import extract_pdf_sections, extract_pdf_text


PDF_DIR = Path("data/PDFs")


@pytest.mark.parametrize(
    ("file_name", "expected_terms"),
    [
        ("fmicb-15-1295149.pdf", ["Sampling", "cruise #SSD062", "Niskin", "Trimmomatic"]),
        ("journal.pone.0303937.pdf", ["Study sites and sampling campaign", "Bioinformatic pipeline", "Trimmomatic"]),
        ("peerj-333.pdf", ["Settlement cue collections", "Reads that became shorter than 250"]),
        ("s42003-024-06136-2.pdf", ["Sampling collection and characterization", "Trimmomatic"]),
        ("wrae013.pdf", ["Sediment sampling", "R/V Augusta", "Bioinformatics of the RNA-seq data", "Trimmomatic"]),
    ],
)
def test_extract_pdf_sections_recovers_methods_terms_from_local_papers(file_name, expected_terms):
    path = PDF_DIR / file_name

    sections = extract_pdf_sections(path)
    joined = "\n".join(f"{section['title']}\n{section['text']}" for section in sections)

    assert sections
    assert len(joined) > 500
    for term in expected_terms:
        assert term.lower() in joined.lower()


def test_extract_pdf_text_preserves_page_boundaries_for_main_paper_pdf():
    text = extract_pdf_text(PDF_DIR / "fmicb-15-1295149.pdf")

    assert "\f" in text
    assert "Sampling" in text
    assert "DNA extraction and amplicon sequencing" in text
