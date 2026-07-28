from fair_ocean_agent.extraction.sections import select_relevant_sections

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
