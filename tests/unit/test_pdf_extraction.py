from pathlib import Path

import pytest

from fair_ocean_agent.extraction.pdf import extract_pdf_sections, extract_pdf_text


PDF_DIR = Path("data/PDFs")

# These are real, manually-downloaded publisher PDFs -- copyrighted content
# that can never be committed to the repo (data/PDFs/ is gitignored), so
# this whole file only has fixtures to run against on a machine where
# someone has actually downloaded them locally. Skips cleanly everywhere
# else (a fresh clone, CI, the cluster) instead of hard-failing on a
# missing-file error that looks like a real regression.
pytestmark = pytest.mark.skipif(not PDF_DIR.is_dir(), reason=f"{PDF_DIR} not present locally (gitignored real PDFs)")


@pytest.mark.parametrize(
    ("file_name", "expected_terms"),
    [
        ("10.3389_fmicb.2024.1295149.pdf", ["Sampling", "cruise #SSD062", "Niskin", "Trimmomatic"]),
        ("10.1371_journal.pone.0303937.pdf", ["Study sites and sampling campaign", "Bioinformatic pipeline", "Trimmomatic"]),
        ("10.7717_peerj.333.pdf", ["Settlement cue collections", "Reads that became shorter than 250"]),
        ("10.1038_s42003-024-06136-2.pdf", ["Sampling collection and characterization", "Trimmomatic"]),
        ("10.1093_ismejo_wrae013.pdf", ["Sediment sampling", "R/V Augusta", "Bioinformatics of the RNA-seq data", "Trimmomatic"]),
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
    text = extract_pdf_text(PDF_DIR / "10.3389_fmicb.2024.1295149.pdf")

    assert "\f" in text
    assert "Sampling" in text
    assert "DNA extraction and amplicon sequencing" in text


def test_extract_pdf_sections_excludes_results_subsection_sharing_methods_vocabulary():
    """Real regression from 10.3389/fmicb.2024.1295149: the Results section
    has its own subsection titled "Taxonomic abundance of the prokaryotic
    ...", which was wrongly re-included (~14,000 chars of Results/
    Discussion prose, 3x the correct total) because its title shares the
    "taxonom" keyword with a legitimate Methods-subheading pattern. The
    literal "Results" heading itself was already correctly excluded --
    the bug was the very next heading slipping back in via the broad
    keyword fallback, independent of the Results/Methods boundary state."""
    sections = extract_pdf_sections(PDF_DIR / "10.3389_fmicb.2024.1295149.pdf")

    titles = [section["title"] for section in sections]
    assert "Taxonomic abundance of the prokaryotic" not in titles
    joined = "\n".join(section["text"] for section in sections)
    assert "884,780 ASVs categorized as Bacteria" not in joined
    total_chars = sum(len(section["text"]) for section in sections)
    assert total_chars < 10000
