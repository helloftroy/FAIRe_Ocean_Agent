from fair_ocean_agent.extraction.sections import MIN_FRAGMENT_CHARS, select_relevant_sections

NESTED_JATS_XML = """<article>
  <body>
    <sec>
      <title>Introduction</title>
      <p>Not relevant background text.</p>
    </sec>
    <sec>
      <title>Materials and Methods</title>
      <sec>
        <title>Sampling</title>
        <p>Water samples were collected at each station.</p>
      </sec>
      <sec>
        <title>DNA extraction and sequencing</title>
        <p>DNA was extracted using a standard kit.</p>
      </sec>
    </sec>
    <sec>
      <title>Results</title>
      <p>Not relevant results text.</p>
    </sec>
    <sec>
      <title>Data Availability Statement</title>
      <p>Data are available at the named repository.</p>
    </sec>
  </body>
</article>
"""


def test_selects_only_relevant_leaf_sections():
    sections = select_relevant_sections(NESTED_JATS_XML)
    titles = {s["title"] for s in sections}
    assert titles == {"Sampling", "DNA extraction and sequencing", "Data Availability Statement"}
    assert "Introduction" not in titles
    assert "Results" not in titles


def test_does_not_duplicate_parent_and_child_text():
    sections = select_relevant_sections(NESTED_JATS_XML)
    # "Materials and Methods" itself (the parent wrapper) must not appear --
    # only its leaf children, otherwise their text would be counted twice.
    assert "Materials and Methods" not in {s["title"] for s in sections}


def test_matches_experimental_procedures_heading():
    """Regression test: live validation against a real paper (PMC7820986,
    Environmental Microbiology 2020) found its entire Methods section
    titled "Experimental procedures" -- a heading style neither "method"
    nor "material" catches, used notably by Cell Press-style journals. This
    was a real 0-of-38 miss in a 41-PMCID pressure test before the
    "procedure" pattern was added."""
    xml = "<article><body><sec><title>Experimental procedures</title><p>Samples were processed as follows.</p></sec></body></article>"
    sections = select_relevant_sections(xml)
    assert [s["title"] for s in sections] == ["Experimental procedures"]


def test_truncates_to_max_chars():
    sections = select_relevant_sections(NESTED_JATS_XML, max_chars=10)
    total_chars = sum(len(s["text"]) for s in sections)
    assert total_chars <= 10


def test_unparseable_xml_returns_empty_list():
    assert select_relevant_sections("not xml at all <<<") == []


def test_no_relevant_sections_returns_empty_list():
    xml = "<article><body><sec><title>Introduction</title><p>text</p></sec></body></article>"
    assert select_relevant_sections(xml) == []


# --- FAIRe-aware taxonomy additions (Milestone 8) ---------------------------
# The expanded taxonomy (extraction/faire_fields.py) targets fields spread
# across more, often separately-titled leaf sections than before (PCR,
# assay, primer, library, sequencing, bioinformatics, taxonomy can each be
# their own <sec>) -- these check the new title patterns and the
# raised default budget that exists specifically so later sections in that
# list aren't truncated away.


def test_matches_new_topic_specific_section_titles():
    for title in (
        "Assay design and validation",
        "Primer selection",
        "Library preparation",
        "qPCR conditions",
        "Taxonomic assignment",
        "Standard curve generation",
    ):
        xml = f"<article><body><sec><title>{title}</title><p>Relevant text.</p></sec></body></article>"
        sections = select_relevant_sections(xml)
        assert [s["title"] for s in sections] == [title], f"{title!r} was not recognized as relevant"


def test_matches_real_audit_section_titles_that_were_previously_dropped():
    """Regression guard from a hand audit of cached Europe PMC papers:
    several FAIRe-relevant method sections used headings outside the
    original selector vocabulary. These headings contain sample/site
    context, DNA handling, storage/fixation, concentration/purity methods,
    and bioinformatics workflows."""
    for title in (
        "Study sites",
        "Study Area",
        "DNA degradation experiment",
        "DNA quantification and quality assessment",
        "Nematode Sorting and Fixation",
        "16S data analysis",
        "Microbiome data analysis",
        "Transcriptomic data analyses",
    ):
        xml = f"<article><body><sec><title>{title}</title><p>Relevant text.</p></sec></body></article>"
        sections = select_relevant_sections(xml)
        assert [s["title"] for s in sections] == [title], f"{title!r} was not recognized as relevant"


def test_default_max_chars_is_raised_for_the_expanded_taxonomy():
    """Regression guard: the original 20000-char default risked truncating
    away exactly the later sections (bioinformatics/taxonomy) this
    taxonomy expansion is meant to reach, since sections are accepted in
    document order until the budget runs out."""
    import inspect

    default_max_chars = inspect.signature(select_relevant_sections).parameters["max_chars"].default
    assert default_max_chars > 20000


def test_skips_a_fragment_too_small_to_be_worth_an_extraction_call():
    xml = """<article><body>
      <sec><title>Sampling</title><p>{first}</p></sec>
      <sec><title>PCR</title><p>{second}</p></sec>
    </body></article>""".format(first="A" * 100, second="B" * 100)
    # Budget covers the first section fully, leaving less than
    # MIN_FRAGMENT_CHARS for the second -- the second should be dropped
    # entirely rather than sent through as a near-empty fragment.
    sections = select_relevant_sections(xml, max_chars=100 + MIN_FRAGMENT_CHARS - 1)
    assert [s["title"] for s in sections] == ["Sampling"]
