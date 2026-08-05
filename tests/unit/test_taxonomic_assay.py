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


def test_extract_assay_target_taxa_returns_empty_without_taxa_or_context():
    xml = """
    <article><front><article-meta>
      <abstract><p>Coral reefs were monitored across seasons.</p></abstract>
      <kwd-group><kwd>Metabarcoding</kwd><kwd>Environmental DNA</kwd></kwd-group>
    </article-meta></front></article>
    """

    assert extract_assay_target_taxa_from_publication_metadata(xml, None, locator_prefix="t") == []
