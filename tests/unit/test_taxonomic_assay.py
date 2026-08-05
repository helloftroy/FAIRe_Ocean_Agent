from fair_ocean_agent.database.enums import SupportType
from fair_ocean_agent.extraction.taxonomic_assay import (
    extract_assay_target_taxa_from_publication_metadata,
)


def test_extract_assay_target_taxa_from_jats_keywords_and_assay_context():
    xml = """
    <article>
      <front>
        <article-meta>
          <title-group>
            <article-title>A reef-building coral settlement study</article-title>
          </title-group>
          <abstract>
            <p>We used a metabarcoding assay targeting Chordata for elasmobranch monitoring.</p>
            <p>Background coral reefs were discussed without assay context.</p>
          </abstract>
          <kwd-group>
            <kwd>Crustose coralline algae</kwd>
            <kwd>Coral recruitment</kwd>
            <kwd>Settlement cues</kwd>
            <kwd>Metabarcoding</kwd>
          </kwd-group>
        </article-meta>
      </front>
    </article>
    """

    facts = extract_assay_target_taxa_from_publication_metadata(
        xml,
        None,
        locator_prefix="t",
    )

    assert len(facts) == 1
    assert facts[0].fact_type_candidate == "assay_target_taxa"
    assert facts[0].raw_value == "Chordata | Crustose coralline algae"
    assert facts[0].support_type == SupportType.DETERMINISTICALLY_DERIVED
    assert "targeting Chordata" in facts[0].evidence_quote
    assert "Crustose coralline algae" in facts[0].evidence_quote


def test_extract_assay_target_taxa_prefers_target_phrase_over_scope_phrase():
    xml = """
    <article><front><article-meta><abstract>
      <p>The metabarcoding assay targeted Chordata for sharks and rays.</p>
    </abstract></article-meta></front></article>
    """

    facts = extract_assay_target_taxa_from_publication_metadata(
        xml,
        None,
        locator_prefix="t",
    )

    assert len(facts) == 1
    assert facts[0].raw_value == "Chordata"


def test_extract_assay_target_taxa_from_crossref_abstract_when_jats_absent():
    crossref_raw = {
        "abstract": "<jats:p>Primers were designed to amplify Acropora millepora DNA.</jats:p>",
        "subject": ["Environmental DNA"],
    }

    facts = extract_assay_target_taxa_from_publication_metadata(
        None,
        crossref_raw,
        locator_prefix="t",
    )

    assert len(facts) == 1
    assert facts[0].raw_value == "Acropora millepora"
    assert facts[0].confidence_metadata["scope"] == "title_abstract_keywords_only"


def test_extract_assay_target_taxa_does_not_treat_sentence_openers_as_species_names():
    """Regression guard for a real false positive (PLOS ONE
    10.1371/journal.pone.0303937): its abstract mentions "PCR" and
    "amplicon" in sentences that pass the loose assay-context gate but
    describe a filtration device, not an assay's target taxon. The old
    `_target_phrase_taxa(sentence) or _taxon_mentions(sentence)` fallback
    scanned the whole sentence for anything matching the "looks like a
    binomial name" regex and produced "Diversity studies" (the abstract's
    own opening two words) and "The device was" (a different sentence's
    opening three words) as if they were taxa."""
    crossref_raw = {
        "abstract": (
            "<jats:p>Diversity studies of aquatic picoplankton communities using size-class "
            "filtration, DNA extraction, PCR and sequencing of phylogenetic markers, require a "
            "robust methodological pipeline. The device was tested using freshwater plankton of "
            "Lake Constance, and total DNA was extracted and an 16S rDNA amplicon was "
            "sequenced.</jats:p>"
        )
    }

    facts = extract_assay_target_taxa_from_publication_metadata(None, crossref_raw, locator_prefix="t")

    assert facts == []


def test_extract_assay_target_taxa_returns_empty_without_taxa_or_context():
    xml = """
    <article><front><article-meta>
      <abstract><p>Coral reefs were monitored across seasons.</p></abstract>
      <kwd-group><kwd>Metabarcoding</kwd><kwd>Environmental DNA</kwd></kwd-group>
    </article-meta></front></article>
    """

    assert extract_assay_target_taxa_from_publication_metadata(xml, None, locator_prefix="t") == []
