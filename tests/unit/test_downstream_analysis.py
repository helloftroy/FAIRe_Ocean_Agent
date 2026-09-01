from fair_ocean_agent.extraction.downstream_analysis import (
    FIELD_NAME,
    detect_downstream_analysis_techniques,
)


def test_detect_downstream_analysis_techniques_from_methods_sequence_context():
    facts = detect_downstream_analysis_techniques(
        [
            (
                "Statistical Analysis",
                "ASV abundances were converted to relative abundance and Hellinger transformed. "
                "NMDS based on Bray-Curtis dissimilarity was used to visualize community structure. "
                "Differences in the OTU table were tested using PERMANOVA, and CatBoost models were "
                "trained on taxonomic abundances.",
            )
        ],
        locator_prefix="test",
    )

    assert len(facts) == 1
    assert facts[0].fact_type_candidate == FIELD_NAME
    assert facts[0].raw_value == "relative abundance | Hellinger | NMDS | Bray-Curtis | PERMANOVA | CatBoost"
    assert "ASV abundances were converted" in (facts[0].evidence_quote or "")


def test_detect_downstream_analysis_techniques_from_multivariate_methods_paragraph():
    facts = detect_downstream_analysis_techniques(
        [
            (
                "Statistical analyses",
                "Multivariate statistical approaches including Analysis of Similarity "
                "(ANOSIM, 'vegan package'), Permutation Multivariate Analysis of Variance "
                "(PERMANOVA, 'vegan package'), Multivariate Homogeneity of Group "
                "Dispersion/variance ('vegan package') and Non-metric Multidimensional "
                "Scaling (NMDS, 'phyloseq package') were based on Bray Curtis dissimilarities.",
            )
        ],
        locator_prefix="test",
    )

    assert len(facts) == 1
    assert facts[0].raw_value == "ANOSIM | PERMANOVA | PERMDISP | NMDS | Bray-Curtis"
    assert "Multivariate statistical approaches" in (facts[0].evidence_quote or "")


def test_detect_downstream_analysis_techniques_ignores_non_methods_sections():
    facts = detect_downstream_analysis_techniques(
        [
            ("Abstract", "We used Bray-Curtis and NMDS to analyze ASV community composition."),
            ("Results", "PERMANOVA showed significant differences in OTU abundances."),
        ],
        locator_prefix="test",
    )

    assert facts == []


def test_detect_downstream_analysis_techniques_requires_sequence_data_context():
    facts = detect_downstream_analysis_techniques(
        [
            (
                "Data Analysis",
                "PCA was performed on environmental chemistry measurements. "
                "Temperature, salinity, and oxygen were included as environmental variables.",
            )
        ],
        locator_prefix="test",
    )

    assert facts == []


def test_detect_downstream_analysis_techniques_uses_strict_ambiguous_acronyms():
    facts = detect_downstream_analysis_techniques(
        [
            (
                "Community Analysis",
                "Canonical correspondence analysis was applied to the ASV abundance table. "
                "ANCOM-BC was used for differential abundance analysis of taxa.",
            )
        ],
        locator_prefix="test",
    )

    assert len(facts) == 1
    assert facts[0].raw_value == "canonical correspondence analysis | ANCOM-BC"
