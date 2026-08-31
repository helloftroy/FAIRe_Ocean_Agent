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


CITATION_XREF_JATS_XML = """<article>
  <body>
    <sec>
      <title>Taxonomic assignment</title>
      <p>The taxonomic classification of OTUs was performed using the lowest common ancestor
algorithm implemented in the Python version of CREST<sup><xref rid="CR70" ref-type="bibr">70</xref></sup>
against the SILVA 138.1 Release<sup><xref rid="CR71" ref-type="bibr">71</xref></sup>.
See <xref rid="F2" ref-type="fig">Fig. 2</xref> for the phylogenetic tree.</p>
    </sec>
  </body>
</article>
"""


EMPTY_PARENTHETICAL_XML = """<article>
  <body>
    <sec>
      <title>Methods</title>
      <p>Sediment traps (<xref rid="CR5" ref-type="bibr">5</xref>) based on decantation and
homogenization phases, and retrieved benthic macroinvertebrates were stored in ethanol.</p>
    </sec>
  </body>
</article>
"""


def test_strips_dangling_empty_parenthetical_left_by_a_removed_citation_number():
    """Regression guard for a real gap found live (10.1002/ece3.6071):
    when a citation number sits inside literal parentheses that are the
    paragraph's own text (not part of the stripped xref), removing just
    the number leaves a dangling "( )" behind."""
    sections = select_relevant_sections(EMPTY_PARENTHETICAL_XML)
    text = sections[0]["text"]
    assert "( )" not in text
    assert "Sediment traps based on decantation" in text


EMPTY_BRACKET_XML = """<article>
  <body>
    <sec>
      <title>Methods</title>
      <p>They were amplified with primers cbbL_K2f and cbbL_V2r [<xref rid="CR25" ref-type="bibr">25</xref>]
following the PCR progress described below.</p>
    </sec>
  </body>
</article>
"""


def test_strips_dangling_empty_bracket_left_by_a_removed_citation_number():
    """Regression guard for a real gap found live
    (10.3390/microorganisms10030558): same shape as the empty-parenthetical
    gap above, but with a citation number inside literal square brackets
    (a real, common in-text citation style) instead of parentheses --
    confirmed live this paper's own real text leaves "primers cbbL_K2f and
    cbbL_V2r [ ]" once the citation number itself is stripped."""
    sections = select_relevant_sections(EMPTY_BRACKET_XML)
    text = sections[0]["text"]
    assert "[ ]" not in text
    assert "cbbL_K2f and cbbL_V2r" in text


def test_strips_bibliography_citation_reference_numbers_from_section_text():
    """Regression guard for a real bug found live (10.1038/s42003-024-06136-2's
    real JATS XML): a naive itertext() call flattens a superscript
    bibliography xref's own number into plain sibling text with no visual
    distinction from real content, corrupting "SILVA 138.1 Release" into
    "SILVA 138.1 Release 71" (71 being a citation/reference number, not
    part of the database name). Other xref ref-types (e.g. fig) are left
    alone since their inline text is real sentence content."""
    sections = select_relevant_sections(CITATION_XREF_JATS_XML)
    text = sections[0]["text"]
    assert "SILVA 138.1 Release ." in text or "SILVA 138.1 Release." in text
    assert "71" not in text
    assert "70" not in text
    assert "Fig. 2" in text


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


def test_method_parent_context_selects_leaf_children_with_non_method_titles():
    xml = """<article><body>
      <sec>
        <title>Materials and Methods</title>
        <sec><title>Caribbean spawn I</title><p>Larvae were collected from replicate colonies.</p></sec>
        <sec><title>Pacific spawn I</title><p>Settlement tiles were conditioned before the assay.</p></sec>
      </sec>
    </body></article>"""

    sections = select_relevant_sections(xml)

    assert [s["title"] for s in sections] == ["Caribbean spawn I", "Pacific spawn I"]
    assert all("Materials and Methods" not in s["title"] for s in sections)


def test_does_not_select_result_leaf_only_because_title_mentions_samples():
    xml = """<article><body>
      <sec>
        <title>Results</title>
        <sec><title>Metabarcoding of cue samples</title><p>Read counts differed between samples.</p></sec>
      </sec>
      <sec>
        <title>Materials and Methods</title>
        <sec><title>Metabarcoding of cue communities</title><p>DNA libraries were sequenced.</p></sec>
      </sec>
    </body></article>"""

    sections = select_relevant_sections(xml)

    assert [s["title"] for s in sections] == ["Metabarcoding of cue communities"]


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


# --- fn-group back-matter declarations (real live audit) --------------------
# PeerJ's real JATS XML (10.7717/peerj.333) wraps "Additional Information and
# Declarations" content in a <sec> whose own children are <fn-group>
# elements, not further <sec>s -- Competing Interests, Author Contributions,
# Field Study Permissions, and (the one that matters for extraction) DNA
# Deposition, which states the paper's own GenBank/SRA accession numbers.
# Confirmed live: this content was structurally invisible to every consumer
# of select_relevant_sections before the fn-group-aware fix below, causing
# real false "no accession number" claims elsewhere in the pipeline even
# though the paper states one.
PEERJ_DECLARATIONS_JATS_XML = """<article>
  <body>
    <sec>
      <title>Materials and Methods</title>
      <sec><title>Sampling</title><p>Water samples were collected at each station.</p></sec>
    </sec>
  </body>
  <back>
    <sec>
      <title>Additional Information and Declarations</title>
      <fn-group content-type="competing-interests">
        <title>Competing Interests</title>
        <fn><p>The authors declare no competing interests.</p></fn>
      </fn-group>
      <fn-group content-type="author-contributions">
        <title>Author Contributions</title>
        <fn><p>All authors contributed equally.</p></fn>
      </fn-group>
      <fn-group content-type="other">
        <title>Field Study Permissions</title>
        <fn><p>Flower Garden Banks National Marine Sanctuary: FGBNMS-2009-005-A2.</p></fn>
      </fn-group>
      <fn-group content-type="other">
        <title>DNA Deposition</title>
        <fn><p>Accession numbers KJ609521-KJ609532 have been deposited on GenBank.</p></fn>
      </fn-group>
    </sec>
  </back>
</article>
"""


def test_includes_relevantly_titled_fn_group_declarations():
    sections = select_relevant_sections(PEERJ_DECLARATIONS_JATS_XML)
    titles = {s["title"] for s in sections}
    assert "DNA Deposition" in titles
    dna_deposition = next(s for s in sections if s["title"] == "DNA Deposition")
    assert "KJ609521-KJ609532" in dna_deposition["text"]


def test_excludes_irrelevantly_titled_fn_group_declarations():
    sections = select_relevant_sections(PEERJ_DECLARATIONS_JATS_XML)
    titles = {s["title"] for s in sections}
    assert "Competing Interests" not in titles
    assert "Author Contributions" not in titles
    assert "Field Study Permissions" not in titles


def test_does_not_yield_the_declarations_wrapper_sec_itself():
    sections = select_relevant_sections(PEERJ_DECLARATIONS_JATS_XML)
    titles = {s["title"] for s in sections}
    assert "Additional Information and Declarations" not in titles


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
