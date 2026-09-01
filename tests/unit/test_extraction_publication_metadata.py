"""Tests against the real cached JATS full text for DOI 10.7717/peerj.333
(PMC3994630) -- the user's own validation paper for this deterministic
project-metadata extraction work. `fetch_fulltext_xml` hits the on-disk
retrieval cache, no live network call.
"""
import json

import pytest

from fair_ocean_agent.database.enums import EntityLevel, ReviewStatus, SupportType
from fair_ocean_agent.database.models import RawFact, Study
from fair_ocean_agent.extraction.publication_metadata import (
    _funding_paragraphs_from_jats,
    extract_code_repo_from_text,
    extract_from_jats_authors,
    extract_from_jats_permissions,
    extract_method_section_citations,
    extract_primer_reference_citations,
    extract_primer_reference_citations_from_text,
    extract_publication_metadata_facts,
    format_bibliographic_citation,
    generate_funding_source,
    generate_rights_holder,
    sync_recorded_by_from_biosample_or_first_author,
)
from fair_ocean_agent.llm.mock import MockLLMBackend
from fair_ocean_agent.workflow.handlers import _build_enabled_adapters

PMCID = "PMC3994630"
DOI = "10.7717/peerj.333"


@pytest.fixture(scope="module")
def real_fulltext_xml():
    adapters = _build_enabled_adapters()
    europe_pmc = adapters.get("europe_pmc")
    if europe_pmc is None:
        pytest.skip("europe_pmc adapter not enabled")
    return europe_pmc.fetch_fulltext_xml(PMCID)


@pytest.fixture(scope="module")
def real_crossref_raw():
    adapters = _build_enabled_adapters()
    crossref = adapters.get("crossref")
    if crossref is None:
        pytest.skip("crossref adapter not enabled")
    return crossref.fetch_record(DOI).raw


def test_permissions_gives_license_accessrights_rightsholder(real_fulltext_xml):
    facts = {f.fact_type_candidate: f for f in extract_from_jats_permissions(real_fulltext_xml, locator_prefix="t")}
    assert facts["license"].raw_value == "http://creativecommons.org/licenses/by/3.0/"
    assert facts["license"].support_type == SupportType.STRUCTURED_SOURCE
    assert facts["accessRights"].raw_value == "open access"
    assert "rightsHolder" not in facts


def test_authors_includes_paper_authors_excludes_editor(real_fulltext_xml):
    """paper_authors_list (not recordedBy directly -- see
    extract_from_jats_authors's own docstring) still carries every real
    author, still excludes the editor. recordedByID is no longer
    extracted at all (an explicit user instruction)."""
    facts = {f.fact_type_candidate: f for f in extract_from_jats_authors(real_fulltext_xml, locator_prefix="t")}
    authors_list = facts["paper_authors_list"].raw_value
    for author in ("Sarah W. Davies", "Eli Meyer", "Sarah M. Guermond", "Mikhail V. Matz"):
        assert author in authors_list
    assert "Qian" not in authors_list  # the editor, not an author -- must never be conflated
    assert facts["project_contact"].raw_value == "Sarah W. Davies <daviessw@gmail.com>"
    assert "recordedByID" not in facts
    assert "recordedBy" not in facts


def test_code_repo_captures_availability_statement_for_this_paper(real_fulltext_xml):
    """This 2014 paper predates common in-text GitHub citation and has no
    public-repository link anywhere in its full text (confirmed) -- but it
    does say where its custom scripts live ("... are available in
    Supplemental Information 1"). A prior round of this test asserted
    code_repo should stay blank for this paper, which was schema-correct
    (code_repo's real definition is a public-repository link) but silently
    dropped this real, useful pointer -- fixed to capture the whole
    sentence instead of requiring a URL."""
    facts = extract_code_repo_from_text(real_fulltext_xml, locator_prefix="t")
    assert len(facts) == 1
    assert facts[0].raw_value == facts[0].evidence_quote
    assert "cca_rarefaction.pl" in facts[0].raw_value
    assert "available in Supplemental Information 1" in facts[0].raw_value
    assert facts[0].support_type == SupportType.DETERMINISTICALLY_DERIVED


def test_code_repo_detects_a_real_github_url():
    xml = "<article><body><p>Code available at https://github.com/someorg/somerepo.</p></body></article>"
    facts = extract_code_repo_from_text(xml, locator_prefix="t")
    assert len(facts) == 1
    assert facts[0].raw_value == "https://github.com/someorg/somerepo"
    assert facts[0].support_type == SupportType.DETERMINISTICALLY_DERIVED


def test_code_repo_prefers_a_real_url_over_an_availability_statement():
    xml = (
        "<article><body><p>Custom scripts are available in Supplemental Information 1. "
        "The full pipeline is also archived at https://github.com/someorg/somerepo.</p></body></article>"
    )
    facts = extract_code_repo_from_text(xml, locator_prefix="t")
    assert len(facts) == 1
    assert facts[0].raw_value == "https://github.com/someorg/somerepo"


def test_code_repo_availability_statement_survives_periods_inside_filenames():
    """Regression guard: a naive "no periods until the sentence ends"
    approach breaks on this exact real sentence shape, which has two
    embedded periods in filenames (cca_rarefaction.pl, rarefaction_figs.R)
    before ever reaching the word "available"."""
    xml = (
        "<article><body><p>Perl script for rarefaction analysis (cca_rarefaction.pl) and R script for "
        "plotting rarefaction curves (rarefaction_figs.R) are available in Supplemental Information 1. "
        "A separate unrelated sentence follows here.</p></body></article>"
    )
    facts = extract_code_repo_from_text(xml, locator_prefix="t")
    assert len(facts) == 1
    assert "unrelated sentence" not in facts[0].raw_value
    assert facts[0].raw_value.endswith("Supplemental Information 1.")


def test_code_repo_does_not_false_positive_on_unrelated_availability_mentions():
    xml = (
        "<article><body><p>Samples were available at each of the seven sites surveyed. "
        "Software packages used in this study include R and Python.</p></body></article>"
    )
    facts = extract_code_repo_from_text(xml, locator_prefix="t")
    assert len(facts) == 1
    assert facts[0].raw_value == "no code published"
    assert facts[0].evidence_quote is None


def test_code_repo_captures_non_github_code_url_from_availability_sentence():
    xml = (
        "<article><body><p>Analysis code is available at "
        "https://example.org/projects/pipeline-code.</p></body></article>"
    )
    facts = extract_code_repo_from_text(xml, locator_prefix="t")
    assert len(facts) == 1
    assert facts[0].raw_value == "https://example.org/projects/pipeline-code"
    assert "Analysis code is available" in facts[0].evidence_quote


def test_method_section_citations_prefer_reference_doi_and_group_by_heading():
    xml = """
    <article>
      <body>
        <sec>
          <title>Materials and methods</title>
          <sec>
            <title>Sediment sampling</title>
            <p>Sampling followed a regional design
            <xref ref-type="bibr" rid="B1">Smith et al.</xref>.</p>
          </sec>
        </sec>
      </body>
      <back>
        <ref-list>
          <ref id="B1">
            <element-citation>
              <article-title>A careful extraction protocol</article-title>
              <pub-id pub-id-type="doi">https://doi.org/10.1234/Protocol.1</pub-id>
            </element-citation>
          </ref>
        </ref-list>
      </back>
    </article>
    """
    facts = extract_method_section_citations(xml, locator_prefix="t")
    assert len(facts) == 1
    assert facts[0].fact_type_candidate == "associated_resource"
    assert facts[0].raw_value == "**Sediment sampling**: doi: 10.1234/protocol.1"
    assert facts[0].support_type == SupportType.DETERMINISTICALLY_DERIVED
    assert "Sampling followed a regional design" in facts[0].evidence_quote
    section = facts[0].confidence_metadata["method_section_citations"][0]
    assert section["heading"] == "Sediment sampling"
    assert section["citations"][0]["ref_id"] == "B1"


def test_method_section_citations_fall_back_to_reference_title_without_doi():
    xml = """
    <article>
      <body>
        <sec>
          <title>Methods</title>
          <p>Amplification followed a previously described method
          <xref ref-type="bibr" rid="B2">22</xref>.</p>
        </sec>
      </body>
      <back>
        <ref-list>
          <ref id="B2">
            <element-citation>
              <article-title>Standard operating procedure for marine DNA barcoding</article-title>
            </element-citation>
          </ref>
        </ref-list>
      </back>
    </article>
    """
    facts = extract_method_section_citations(xml, locator_prefix="t")
    assert len(facts) == 1
    assert facts[0].raw_value == "**Methods**: Standard operating procedure for marine DNA barcoding"


def test_method_section_citations_extract_doi_from_reference_text_tail():
    xml = """
    <article>
      <body>
        <sec>
          <title>Materials and methods</title>
          <sec>
            <title>Bioinformatic pipeline</title>
            <p>Reads were filtered as described previously
            <xref ref-type="bibr" rid="B3">36</xref>.</p>
          </sec>
        </sec>
      </body>
      <back>
        <ref-list>
          <ref id="B3">
            <mixed-citation>
              Bolger AM, Lohse M, Usadel B. Trimmomatic: a flexible trimmer
              for Illumina sequence data. Bioinformatics. 2014;30:2114-2120.
              doi: 10.1093/bioinformatics/btu170.
            </mixed-citation>
          </ref>
        </ref-list>
      </back>
    </article>
    """
    facts = extract_method_section_citations(xml, locator_prefix="t")
    assert len(facts) == 1
    assert facts[0].raw_value == "**Bioinformatic pipeline**: doi: 10.1093/bioinformatics/btu170"
    citation = facts[0].confidence_metadata["method_section_citations"][0]["citations"][0]
    assert citation["resource"] == "doi: 10.1093/bioinformatics/btu170"
    assert "Trimmomatic" in citation["citation_text"]


def test_method_section_citations_include_all_method_citations_but_ignore_results_sections():
    xml = """
    <article>
      <body>
        <sec>
          <title>Methods</title>
          <p>Reads were processed with DADA2 <xref ref-type="bibr" rid="B1">1</xref>.</p>
        </sec>
        <sec>
          <title>Results</title>
          <p>Patterns matched a previously described method <xref ref-type="bibr" rid="B2">2</xref>.</p>
        </sec>
      </body>
      <back>
        <ref-list>
          <ref id="B1"><element-citation><pub-id pub-id-type="doi">10.1000/dada2</pub-id></element-citation></ref>
          <ref id="B2"><element-citation><pub-id pub-id-type="doi">10.1000/results</pub-id></element-citation></ref>
        </ref-list>
      </back>
    </article>
    """
    facts = extract_method_section_citations(xml, locator_prefix="t")
    assert len(facts) == 1
    assert facts[0].raw_value == "**Methods**: doi: 10.1000/dada2"
    assert "10.1000/results" not in facts[0].raw_value


def test_method_section_citations_preserve_multiple_subsection_headings():
    xml = """
    <article>
      <body>
        <sec>
          <title>Materials and methods</title>
          <sec>
            <title>Sediment sampling</title>
            <p>Sampling design cited <xref ref-type="bibr" rid="B1">1</xref>.</p>
          </sec>
          <sec>
            <title>Meiofauna extractions and experimental setup</title>
            <p>Extraction setup cited <xref ref-type="bibr" rid="B2">2</xref>.</p>
          </sec>
        </sec>
      </body>
      <back>
        <ref-list>
          <ref id="B1"><element-citation><pub-id pub-id-type="doi">10.1000/sediment</pub-id></element-citation></ref>
          <ref id="B2"><element-citation><pub-id pub-id-type="doi">10.1000/meiofauna</pub-id></element-citation></ref>
        </ref-list>
      </back>
    </article>
    """
    facts = extract_method_section_citations(xml, locator_prefix="t")
    assert len(facts) == 1
    assert facts[0].raw_value == (
        "**Sediment sampling**: doi: 10.1000/sediment | "
        "**Meiofauna extractions and experimental setup**: doi: 10.1000/meiofauna"
    )


_PRIMER_XML = """
<article>
  <body>
    <sec>
      <title>Methods</title>
      <sec>
        <title>DNA extraction and PCR amplification</title>
        <p>We amplified the V4 region using the forward primer 515F
        <xref ref-type="bibr" rid="ref1">1</xref> without further modification.</p>
      </sec>
    </sec>
  </body>
  <back>
    <ref-list>
      <ref id="ref1">
        <element-citation>
          <article-title>Global patterns of 16S rRNA diversity</article-title>
          <pub-id pub-id-type="doi">10.1038/ismej.2012.8</pub-id>
        </element-citation>
      </ref>
    </ref-list>
  </back>
</article>
"""


def test_primer_reference_citations_resolves_real_doi_next_to_primer_name():
    facts = extract_primer_reference_citations(
        _PRIMER_XML, {"pcr_primer_name_forward": "515F", "pcr_primer_name_reverse": ""}, locator_prefix="t"
    )
    assert len(facts) == 1
    assert facts[0].fact_type_candidate == "pcr_primer_reference_forward"
    assert facts[0].raw_value == "doi: 10.1038/ismej.2012.8"
    assert facts[0].support_type == SupportType.DETERMINISTICALLY_DERIVED
    assert "515F" in facts[0].evidence_quote
    assert facts[0].confidence_metadata["primer_name"] == "515F"
    assert facts[0].confidence_metadata["ref_id"] == "ref1"


def test_primer_reference_citations_prefers_the_citation_nearest_the_primer_name():
    """Real bug caught live against 10.1038/s42003-024-06136-2 (PMC11009272):
    a real Methods paragraph cites something UNRELATED earlier
    ("Similar to Zhao et al. [28] where AOA's distribution was explored...")
    before actually attributing the primers later in the same paragraph
    ("...primers of Uni519F/806r, as described in Zhao et al. [38]").
    Taking the first citation marker in the whole paragraph resolved to the
    wrong reference (28, about AOA distribution) instead of the real one
    (38, the actual primer source) -- the nearest marker to the primer
    name's own position must win instead."""
    xml = """
    <article>
      <body>
        <sec>
          <title>Methods</title>
          <p>Similar to Zhao et al. <xref ref-type="bibr" rid="ref28">28</xref> where AOA's distribution was
          explored, we investigated NOB. Amplicon of the 16S rRNA gene was prepared using the primers of
          Uni519F/806r, as described in Zhao et al. <xref ref-type="bibr" rid="ref38">38</xref>.</p>
        </sec>
      </body>
      <back>
        <ref-list>
          <ref id="ref28"><element-citation><pub-id pub-id-type="doi">10.1000/wrong-reference</pub-id></element-citation></ref>
          <ref id="ref38"><element-citation><pub-id pub-id-type="doi">10.1000/right-reference</pub-id></element-citation></ref>
        </ref-list>
      </back>
    </article>
    """
    facts = extract_primer_reference_citations(
        xml, {"pcr_primer_name_forward": "Uni519F", "pcr_primer_name_reverse": ""}, locator_prefix="t"
    )
    assert {fact.fact_type_candidate for fact in facts} == {
        "pcr_primer_reference_forward",
        "pcr_primer_reference_reverse",
    }
    assert {fact.raw_value for fact in facts} == {"doi: 10.1000/right-reference"}
    assert {fact.confidence_metadata["ref_id"] for fact in facts} == {"ref38"}


def test_primer_reference_citations_falls_back_to_reference_title_without_doi():
    xml = """
    <article>
      <body>
        <sec>
          <title>Methods</title>
          <p>We used the reverse primer 806R <xref ref-type="bibr" rid="ref2">2</xref> as previously described.</p>
        </sec>
      </body>
      <back>
        <ref-list>
          <ref id="ref2">
            <element-citation>
              <article-title>Improved reverse primer design for prokaryotic diversity</article-title>
            </element-citation>
          </ref>
        </ref-list>
      </back>
    </article>
    """
    facts = extract_primer_reference_citations(
        xml, {"pcr_primer_name_forward": "", "pcr_primer_name_reverse": "806R"}, locator_prefix="t"
    )
    assert len(facts) == 1
    assert facts[0].fact_type_candidate == "pcr_primer_reference_reverse"
    assert facts[0].raw_value == "Improved reverse primer design for prokaryotic diversity"
    assert not facts[0].raw_value.startswith("doi: ")


def test_primer_reference_citations_fallback_extracts_generic_primer_pair_reference():
    xml = """
    <article>
      <body>
        <sec>
          <title>Methods</title>
          <p>The V4 region was amplified using a universal primer pair as previously described
          <xref ref-type="bibr" rid="ref3">3</xref>.</p>
        </sec>
      </body>
      <back>
        <ref-list>
          <ref id="ref3">
            <element-citation><pub-id pub-id-type="doi">10.1000/primer-pair</pub-id></element-citation>
          </ref>
        </ref-list>
      </back>
    </article>
    """

    facts = extract_primer_reference_citations(xml, {}, locator_prefix="t")
    by_type = {fact.fact_type_candidate: fact for fact in facts}

    assert by_type["pcr_primer_reference_forward"].raw_value == "doi: 10.1000/primer-pair"
    assert by_type["pcr_primer_reference_reverse"].raw_value == "doi: 10.1000/primer-pair"
    assert by_type["pcr_primer_reference_forward"].confidence_metadata["detector"] == "primer_reference_citation_context_fallback"


def test_primer_reference_citations_fallback_extracts_slash_primer_pair_reference():
    xml = """
    <article>
      <body>
        <sec>
          <title>Methods</title>
          <p>Amplicons were generated with 515F/806R primers
          <xref ref-type="bibr" rid="ref4">4</xref>.</p>
        </sec>
      </body>
      <back>
        <ref-list>
          <ref id="ref4">
            <element-citation><article-title>Primer pair source paper</article-title></element-citation>
          </ref>
        </ref-list>
      </back>
    </article>
    """

    facts = extract_primer_reference_citations(xml, {}, locator_prefix="t")
    assert {fact.fact_type_candidate for fact in facts} == {
        "pcr_primer_reference_forward",
        "pcr_primer_reference_reverse",
    }
    assert {fact.raw_value for fact in facts} == {"Primer pair source paper"}


def test_primer_reference_citations_fallback_respects_single_forward_direction():
    xml = """
    <article>
      <body>
        <sec>
          <title>Methods</title>
          <p>The forward primer was used as described by Smith et al.
          <xref ref-type="bibr" rid="ref5">5</xref>.</p>
        </sec>
      </body>
      <back>
        <ref-list>
          <ref id="ref5">
            <element-citation><pub-id pub-id-type="doi">10.1000/forward-only</pub-id></element-citation>
          </ref>
        </ref-list>
      </back>
    </article>
    """

    facts = extract_primer_reference_citations(xml, {}, locator_prefix="t")
    assert [fact.fact_type_candidate for fact in facts] == ["pcr_primer_reference_forward"]
    assert facts[0].raw_value == "doi: 10.1000/forward-only"


def test_primer_reference_citations_no_match_when_primer_name_not_mentioned():
    facts = extract_primer_reference_citations(
        _PRIMER_XML, {"pcr_primer_name_forward": "806R", "pcr_primer_name_reverse": ""}, locator_prefix="t"
    )
    assert facts == []


def test_primer_reference_citations_empty_primer_names_still_use_directional_context():
    facts = extract_primer_reference_citations(_PRIMER_XML, {}, locator_prefix="t")
    assert [fact.fact_type_candidate for fact in facts] == ["pcr_primer_reference_forward"]
    assert facts[0].raw_value == "doi: 10.1038/ismej.2012.8"


def test_primer_reference_citations_ignores_a_sentence_with_no_citation_marker():
    xml = """
    <article>
      <body>
        <sec>
          <title>Methods</title>
          <p>We amplified the V4 region using the forward primer 515F with no citation at all.</p>
        </sec>
      </body>
      <back>
        <ref-list>
          <ref id="ref1">
            <element-citation><pub-id pub-id-type="doi">10.1038/ismej.2012.8</pub-id></element-citation>
          </ref>
        </ref-list>
      </back>
    </article>
    """
    facts = extract_primer_reference_citations(
        xml, {"pcr_primer_name_forward": "515F", "pcr_primer_name_reverse": ""}, locator_prefix="t"
    )
    assert facts == []


def test_primer_reference_citations_from_text_resolves_numeric_pdf_reference_doi():
    text = """
    Methods
    Amplicons were generated with 515F/806R primers [4].

    References
    [4] Caporaso, J. G. et al. Global patterns of 16S rRNA diversity.
    ISME Journal. doi: 10.1038/ismej.2012.8.
    """

    facts = extract_primer_reference_citations_from_text(text, {}, locator_prefix="pdf")
    by_type = {fact.fact_type_candidate: fact for fact in facts}

    assert by_type["pcr_primer_reference_forward"].raw_value == "doi: 10.1038/ismej.2012.8"
    assert by_type["pcr_primer_reference_reverse"].raw_value == "doi: 10.1038/ismej.2012.8"
    assert by_type["pcr_primer_reference_forward"].confidence_metadata["detector"] == (
        "primer_reference_citation_text_context_fallback"
    )


def test_primer_reference_citations_from_text_resolves_author_year_pdf_reference_doi_near_primer_name():
    text = """
    Methods
    Similar to Zhao et al. (2020) where AOA's distribution was explored, we investigated NOB.
    Amplicons were prepared using the primers of Uni519F/806r, as described in Smith et al. (2018).

    References
    Zhao, R. 2020. AOA distributions in marine sediments. Journal. doi: 10.1000/wrong-reference.
    Smith, K. 2018. Primer pair source paper. Journal. doi: 10.1000/right-reference.
    """

    facts = extract_primer_reference_citations_from_text(
        text, {"pcr_primer_name_forward": "Uni519F", "pcr_primer_name_reverse": ""}, locator_prefix="pdf"
    )

    assert {fact.fact_type_candidate for fact in facts} == {
        "pcr_primer_reference_forward",
        "pcr_primer_reference_reverse",
    }
    assert {fact.raw_value for fact in facts} == {"doi: 10.1000/right-reference"}


def test_generate_funding_source_from_jats_funding_paragraph():
    xml = """
    <article>
      <back>
        <sec sec-type="funding">
          <title>Funding</title>
          <p>This work was supported by the National Science Foundation (OCE-123)
          and the Gordon and Betty Moore Foundation. The funders had no role
          in study design.</p>
        </sec>
      </back>
    </article>
    """
    backend = MockLLMBackend(
        responses=[
            json.dumps(
                {
                    "funding_source": (
                        "National Science Foundation | Gordon and Betty Moore Foundation"
                    )
                }
            )
        ]
    )

    facts = generate_funding_source(backend, xml, locator_prefix="t")

    assert len(facts) == 1
    assert facts[0].fact_type_candidate == "funding_source"
    assert facts[0].raw_value == "National Science Foundation | Gordon and Betty Moore Foundation"
    assert facts[0].support_type == SupportType.EXPLICIT
    assert "National Science Foundation" in facts[0].evidence_quote
    assert "The funders had no role" in backend.calls[0]["prompt"]


def test_generate_funding_source_from_pdf_plain_text_funding_section():
    backend = MockLLMBackend(
        responses=[
            json.dumps(
                {
                    "funding_source": (
                        "National Science Foundation | Gordon and Betty Moore Foundation"
                    )
                }
            )
        ]
    )
    pdf_text = """Results
These are results.

Funding
This work was supported by the National Science Foundation (OCE-123)
and the Gordon and Betty Moore Foundation. The funders had no role in study design.

References
Reference text.
"""

    facts = generate_funding_source(backend, pdf_text, locator_prefix="pdf")

    assert len(facts) == 1
    assert facts[0].raw_value == "National Science Foundation | Gordon and Betty Moore Foundation"
    assert facts[0].support_type == SupportType.EXPLICIT
    assert "National Science Foundation" in facts[0].evidence_quote
    assert "The funders had no role" in backend.calls[0]["prompt"]


def test_generate_funding_source_falls_back_for_plos_dfg_funding_line():
    """Regression guard for 10.1371/journal.pone.0303937: the funding
    paragraph is present in PDF/plain text, but the model can over-trim it
    to blank or a fragment. The deterministic backup should preserve the
    explicit DFG agency + named program phrase."""
    backend = MockLLMBackend(responses=[json.dumps({"funding_source": ""})])
    text = """Funding
This research was funded by DFG
Research Training Group R3 - Responses to biotic
and abiotic Changes, Resilience and Reversibi lity of
Lake Ecosystems (GRK 2272) and by the
University of Konstanz (AFF grants 2019-2021 to
DS). The funders had no role in study design.

Author contributions
Funding acquisition: David Schleheck.
"""

    facts = generate_funding_source(backend, text, locator_prefix="pdf")

    assert len(facts) == 1
    assert facts[0].raw_value == (
        "DFG Research Training Group R3 - Responses to biotic and abiotic "
        "Changes, Resilience and Reversibility of Lake Ecosystems"
    )
    assert "Funding acquisition: David Schleheck" not in facts[0].evidence_quote


def test_funding_sentences_ignore_author_contribution_funding_acquisition():
    backend = MockLLMBackend(responses=[json.dumps({"funding_source": "David Schleheck"})])

    facts = generate_funding_source(
        backend,
        "Author contributions\nFunding acquisition: David Schleheck.\n",
        locator_prefix="pdf",
    )

    assert facts == []
    assert backend.calls == []


def test_funding_source_filters_institutional_units_and_fragments():
    xml = """
    <article>
      <back>
        <sec sec-type="funding">
          <title>Funding</title>
          <p>Research was funded by the National Science Foundation (DEB-1054766),
          a departmental start-up grant from Section of Integrative Biology at the
          University of Texas at Austin, and a PADI Foundation Award.</p>
        </sec>
      </back>
    </article>
    """
    backend = MockLLMBackend(
        responses=[
            json.dumps(
                {
                    "funding_source": (
                        "National Science Foundation | Section of Integrative Biology at the "
                        "University of Texas at Austin | PADI Foundation | W"
                    )
                }
            )
        ]
    )

    facts = generate_funding_source(backend, xml, locator_prefix="t")

    assert len(facts) == 1
    assert facts[0].raw_value == "National Science Foundation | PADI Foundation"


def test_funding_source_drops_host_university_when_not_named_program():
    xml = """
    <article>
      <back>
        <sec sec-type="funding">
          <title>Funding</title>
          <p>This work was funded by Simons Foundation and the National Science
          Foundation. R.Z. was supported by MIT Molina Postdoctoral Fellowship.
          R.Z. was supported by Trond Mohn Foundation and University of Bergen
          through Centre for Deep Sea Research.</p>
        </sec>
      </back>
    </article>
    """
    backend = MockLLMBackend(
        responses=[
            json.dumps(
                {
                    "funding_source": (
                        "Simons Foundation | National Science Foundation | "
                        "MIT Molina Postdoctoral Fellowship | Trond Mohn Foundation | University of Bergen"
                    )
                }
            )
        ]
    )

    facts = generate_funding_source(backend, xml, locator_prefix="t")

    assert len(facts) == 1
    assert facts[0].raw_value == (
        "Simons Foundation | National Science Foundation | "
        "MIT Molina Postdoctoral Fellowship | Trond Mohn Foundation"
    )


def test_funding_jats_section_uses_direct_paragraphs_only():
    xml = """
    <article>
      <back>
        <sec sec-type="funding">
          <title>Funding</title>
          <p>This study was funded by the Research Council of Norway and the Austrian Science Fund.</p>
          <sec>
            <title>Sequencing method note</title>
            <p>The utilization of these two compounds has been suggested and was supported by single-cell
            genome sequencing in a previous experiment.</p>
          </sec>
        </sec>
      </back>
    </article>
    """

    paragraphs = _funding_paragraphs_from_jats(xml)

    assert paragraphs == ["This study was funded by the Research Council of Norway and the Austrian Science Fund."]


def test_generate_funding_source_no_funding_text_makes_no_llm_call():
    backend = MockLLMBackend(responses=[json.dumps({"funding_source": "National Science Foundation"})])

    facts = generate_funding_source(backend, "<article><body><p>No methods here.</p></body></article>", locator_prefix="t")

    assert facts == []
    assert backend.calls == []


def test_bibliographic_citation_composed_from_crossref(real_crossref_raw):
    facts = format_bibliographic_citation(real_crossref_raw, locator_prefix="t")
    assert len(facts) == 1
    citation = facts[0].raw_value
    assert "Davies SW" in citation
    assert "2014" in citation
    assert "A cross-ocean comparison of responses to settlement cues in reef-building corals" in citation
    assert "PeerJ" in citation
    assert DOI in citation


def test_bibliographic_citation_absent_without_crossref_data():
    assert format_bibliographic_citation(None, locator_prefix="t") == []
    assert format_bibliographic_citation({}, locator_prefix="t") == []


def test_extract_publication_metadata_facts_merges_everything(real_fulltext_xml, real_crossref_raw):
    facts = extract_publication_metadata_facts(real_fulltext_xml, real_crossref_raw, locator_prefix="t")
    fact_types = {f.fact_type_candidate for f in facts}
    assert fact_types == {
        "license",
        "accessRights",
        "paper_authors_list",
        "project_contact",
        "bibliographicCitation",
        "code_repo",
        "associated_resource",
    }


def test_malformed_xml_returns_empty_not_raises():
    assert extract_from_jats_permissions("<not valid xml", locator_prefix="t") == []
    assert extract_from_jats_authors("<not valid xml", locator_prefix="t") == []


def test_generate_rights_holder_keeps_generic_author_holder_and_year():
    xml = """<article><front><article-meta><contrib-group>
    <contrib contrib-type="author"><name><given-names>A.</given-names><surname>Alpha</surname></name></contrib>
    <contrib contrib-type="author"><name><given-names>Beatrice</given-names><surname>Beta</surname></name></contrib>
    </contrib-group><permissions>
    <copyright-statement>&#169; The Author(s) 2020. Published by Oxford University Press.</copyright-statement>
    </permissions></article-meta></front></article>"""
    backend = MockLLMBackend(responses=[json.dumps({"rightsHolder": "2020 The Author(s)"})])

    facts = generate_rights_holder(backend, xml, locator_prefix="t")

    assert len(facts) == 1
    assert facts[0].raw_value == "2020 The Author(s)"
    assert facts[0].support_type == SupportType.EXPLICIT
    assert "The Author(s) 2020" in facts[0].evidence_quote
    assert "Do not replace \"The Author(s)\"" in backend.calls[0]["prompt"]


def test_generate_rights_holder_can_extract_journal_or_publisher_holder():
    xml = """<article><body><sec><permissions>
    <copyright-statement>&#169; 2024 International Society for Microbial Ecology</copyright-statement>
    </permissions></sec></body></article>"""
    backend = MockLLMBackend(responses=[json.dumps({"rightsHolder": "2024 International Society for Microbial Ecology"})])

    facts = generate_rights_holder(backend, xml, locator_prefix="t")

    assert len(facts) == 1
    assert facts[0].raw_value == "2024 International Society for Microbial Ecology"


def test_rightsholder_absent_when_neither_tag_present():
    xml = "<article><body><sec><permissions><license/></permissions></sec></body></article>"
    backend = MockLLMBackend(responses=[json.dumps({"rightsHolder": "2024 The Author(s)"})])

    assert generate_rights_holder(backend, xml, locator_prefix="t") == []
    assert backend.calls == []


def test_generate_rights_holder_uses_rights_section_when_no_permissions_element():
    xml = """<article><front><article-meta><contrib-group content-type="author">
    <contrib><name><given-names>Rui</given-names><surname>Zhao</surname></name></contrib>
    <contrib><name><given-names>Andrew R</given-names><surname>Babbin</surname></name></contrib>
    </contrib-group></article-meta></front>
    <body><sec sec-type="rights-and-permissions"><title>Rights and permissions</title>
    <p>Copyright 2024 The Author(s), under exclusive licence to Springer Nature Limited.</p>
    </sec></body></article>"""
    backend = MockLLMBackend(responses=[json.dumps({"rightsHolder": "2024 The Author(s)"})])

    facts = generate_rights_holder(backend, xml, locator_prefix="t")

    assert len(facts) == 1
    assert facts[0].raw_value == "2024 The Author(s)"
    assert "exclusive licence to Springer Nature Limited" in facts[0].evidence_quote


def test_generate_rights_holder_from_pdf_plain_text_rights_section():
    pdf_text = """Abstract
Study abstract.

Rights and permissions
Copyright 2024 The Author(s), under exclusive licence to Springer Nature Limited.

References
Reference text.
"""
    backend = MockLLMBackend(responses=[json.dumps({"rightsHolder": "2024 The Author(s)"})])

    facts = generate_rights_holder(backend, pdf_text, locator_prefix="pdf")

    assert len(facts) == 1
    assert facts[0].raw_value == "2024 The Author(s)"
    assert "exclusive licence to Springer Nature Limited" in facts[0].evidence_quote


def test_authors_and_contact_from_group_level_content_type_and_author_notes_corresp():
    """Regression guard for the same real gap, plus a second one found
    alongside it: this same real document marks its corresponding author
    via an <xref ref-type="author-notes"> pointing at an <fn> saying
    "Corresponding author." -- not a corresp="yes" attribute or an
    <email> element anywhere -- so project_contact used to come back
    entirely empty too."""
    xml = """<article><front><article-meta><contrib-group content-type="author">
    <contrib><name><given-names>Rui</given-names><surname>Zhao</surname></name>
    <xref ref-type="author-notes" rid="fn1">&#10038;</xref></contrib>
    <contrib><name><given-names>Andrew R</given-names><surname>Babbin</surname></name></contrib>
    </contrib-group>
    <author-notes><fn id="fn1"><label>&#10038;</label><p>Corresponding author.</p></fn></author-notes>
    </article-meta></front></article>"""
    facts = {f.fact_type_candidate: f for f in extract_from_jats_authors(xml, locator_prefix="t")}
    assert facts["paper_authors_list"].raw_value == "Rui Zhao | Andrew R Babbin"
    assert facts["project_contact"].raw_value == "Rui Zhao"


def test_authors_group_level_content_type_never_overrides_explicit_non_author_contrib_type():
    """The group-level content-type="author" fallback only fills in for a
    contrib with NO contrib-type of its own -- it must never override an
    explicit non-author role (e.g. an editor correctly excluded per this
    module's own docstring) that happens to sit inside such a group."""
    xml = """<article><front><article-meta><contrib-group content-type="author">
    <contrib><name><given-names>Rui</given-names><surname>Zhao</surname></name></contrib>
    <contrib contrib-type="editor"><name><given-names>Some</given-names><surname>Editor</surname></name></contrib>
    </contrib-group></article-meta></front></article>"""
    facts = {f.fact_type_candidate: f for f in extract_from_jats_authors(xml, locator_prefix="t")}
    assert facts["paper_authors_list"].raw_value == "Rui Zhao"


def _study(session, **kwargs) -> Study:
    study = Study(**kwargs)
    session.add(study)
    session.flush()
    return study


def _biosample_recorded_by_fact(session, study: Study, value: str) -> RawFact:
    fact = RawFact(
        study_id=study.study_id,
        entity_id=None,
        raw_field_name="recordedBy",
        raw_value=value,
        fact_type_candidate="recordedBy",
        entity_level=EntityLevel.STUDY.value,
        support_type=SupportType.STRUCTURED_SOURCE.value,
        source_locator="ncbi_biosample.SAMN1.Owner.Contacts.Contact.Name",
    )
    session.add(fact)
    session.flush()
    return fact


def _paper_authors_list_fact(session, study: Study, value: str) -> RawFact:
    fact = RawFact(
        study_id=study.study_id,
        entity_id=None,
        raw_field_name="paper_authors_list",
        raw_value=value,
        fact_type_candidate="paper_authors_list",
        entity_level=EntityLevel.STUDY.value,
        support_type=SupportType.STRUCTURED_SOURCE.value,
        source_locator="publication_metadata:10.1/x:contrib-group",
    )
    session.add(fact)
    session.flush()
    return fact


def test_sync_recorded_by_prefers_real_biosample_contact_over_first_author(db_session):
    study = _study(db_session, title="BioSample contact present")
    _biosample_recorded_by_fact(db_session, study, "Daniel Killam")
    _paper_authors_list_fact(db_session, study, "Sarah W. Davies | Eli Meyer")
    db_session.commit()

    sync_recorded_by_from_biosample_or_first_author(db_session, study.study_id)
    db_session.commit()

    facts = db_session.query(RawFact).filter_by(study_id=study.study_id, fact_type_candidate="recordedBy").all()
    assert len(facts) == 1
    assert facts[0].raw_value == "Daniel Killam"


def test_sync_recorded_by_falls_back_to_first_author_only_when_no_biosample_contact(db_session):
    study = _study(db_session, title="No BioSample contact")
    _paper_authors_list_fact(db_session, study, "Sarah W. Davies | Eli Meyer | Sarah M. Guermond")
    db_session.commit()

    sync_recorded_by_from_biosample_or_first_author(db_session, study.study_id)
    db_session.commit()

    fact = db_session.query(RawFact).filter_by(study_id=study.study_id, fact_type_candidate="recordedBy").one()
    assert fact.raw_value == "Sarah W. Davies"
    assert fact.extraction_method == "derived:recorded_by_first_author_fallback"


def test_sync_recorded_by_no_op_when_neither_source_exists(db_session):
    study = _study(db_session, title="No data at all")
    db_session.commit()

    sync_recorded_by_from_biosample_or_first_author(db_session, study.study_id)
    db_session.commit()

    assert db_session.query(RawFact).filter_by(study_id=study.study_id, fact_type_candidate="recordedBy").count() == 0


def test_sync_recorded_by_self_heals_when_biosample_contact_appears_later(db_session):
    """A study synced with the first-author fallback on an earlier run
    must switch over to the real BioSample contact once one becomes
    available on a later run, not keep the stale fallback forever."""
    study = _study(db_session, title="Self-healing fallback")
    _paper_authors_list_fact(db_session, study, "Sarah W. Davies | Eli Meyer")
    db_session.commit()
    sync_recorded_by_from_biosample_or_first_author(db_session, study.study_id)
    db_session.commit()
    assert (
        db_session.query(RawFact)
        .filter_by(study_id=study.study_id, fact_type_candidate="recordedBy", review_status=ReviewStatus.ACCEPTED.value)
        .one()
        .raw_value
        == "Sarah W. Davies"
    )

    _biosample_recorded_by_fact(db_session, study, "Daniel Killam")
    db_session.commit()
    sync_recorded_by_from_biosample_or_first_author(db_session, study.study_id)
    db_session.commit()

    active_facts = [
        fact
        for fact in db_session.query(RawFact).filter_by(study_id=study.study_id, fact_type_candidate="recordedBy")
        if fact.review_status != ReviewStatus.REJECTED.value
    ]
    assert len(active_facts) == 1
    assert active_facts[0].raw_value == "Daniel Killam"


def test_sync_recorded_by_is_idempotent_on_rerun(db_session):
    study = _study(db_session, title="Idempotent rerun")
    _paper_authors_list_fact(db_session, study, "Sarah W. Davies | Eli Meyer")
    db_session.commit()

    sync_recorded_by_from_biosample_or_first_author(db_session, study.study_id)
    db_session.commit()
    sync_recorded_by_from_biosample_or_first_author(db_session, study.study_id)
    db_session.commit()

    assert db_session.query(RawFact).filter_by(study_id=study.study_id, fact_type_candidate="recordedBy").count() == 1
