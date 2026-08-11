import json

from fair_ocean_agent.extraction.search_flags import (
    detect_controlled_search_facts,
    detect_llm_judged_search_facts,
    detect_text_search_flags,
    quote_candidates_for_llm_judged_search,
)
from fair_ocean_agent.extraction.section_categories import (
    derive_pcr_0_1_from_category_detection,
    detect_section_categories_present,
)
from fair_ocean_agent.llm.mock import MockLLMBackend


def test_detect_text_search_flags_records_probe_flag_only_pcr_0_1_moved_to_category_detection():
    """pcr_0_1 is no longer detect_text_search_flags's own concern -- see
    derive_pcr_0_1_from_category_detection, which now derives it from
    extraction/section_categories.py's pcr1_primary_amplification_0_1 /
    targeted_qpcr_ddpcr_detection_0_1 detection instead (an explicit user
    instruction to stop computing it via an independent regex scan)."""
    facts = detect_text_search_flags(
        (
            (
                "Methods",
                "The assay used a TaqMan hydrolysis probe with a FAM reporter dye. "
                "PCR amplification was performed in triplicate.",
            ),
            ("Repeated PCR", "PCR was mentioned again later."),
        ),
        locator_prefix="paper:PMC1",
    )

    by_type = {fact.fact_type_candidate: fact for fact in facts}
    assert set(by_type) == {"probe_based_qPCR_ddPCR_assay_0_1"}
    assert by_type["probe_based_qPCR_ddPCR_assay_0_1"].raw_value == "true"
    assert by_type["probe_based_qPCR_ddPCR_assay_0_1"].evidence_quote.startswith("The assay used a TaqMan")
    assert by_type["probe_based_qPCR_ddPCR_assay_0_1"].confidence_metadata["matched_terms"] == [
        "FAM",
        "hydrolysis probe",
        "reporter dye",
        "TaqMan",
    ]


def test_pcr_0_1_derivation_matches_amplified_verb_forms_not_just_amplification():
    """Regression guard for a real gold case (data/benchmark/gold/example-001.json)
    that describes explicit PCR content ("...was amplified using primers...
    in a 25 uL reaction volume with an annealing temperature of 57C for 35
    cycles...") but never uses the word "PCR" or the noun "amplification" --
    only the verb "amplified". pcr_0_1 must still activate (now via
    category detection, not its own regex), or every downstream
    flag-gated PCR checklist field silently becomes unreachable for a
    real paper phrased this way."""
    texts = (("Methods", "The target region was amplified using primers X and Y."),)
    section_category_facts = detect_section_categories_present(list(texts), locator_prefix="paper:PMC1")
    assert "pcr1_primary_amplification_0_1" in {f.fact_type_candidate for f in section_category_facts}

    pcr_0_1_fact = derive_pcr_0_1_from_category_detection(section_category_facts)
    assert pcr_0_1_fact is not None
    assert pcr_0_1_fact.raw_value == "1"


def test_detect_text_search_flags_does_not_invent_absent_flags():
    facts = detect_text_search_flags(
        (("Methods", "Water samples were collected at 5 m depth."),),
        locator_prefix="paper:PMC1",
    )

    assert facts == []


def test_detect_llm_judged_search_facts_records_control_booleans_from_context():
    """neg_cont_0_1/pos_cont_0_1 moved from a regex-only TextSearchFlag
    pair to LLM-judged keyword+context classification, per an explicit
    user instruction: "use keyword 'control','blank' then context in
    sentence to determine if positive or negative control or not a
    control." """
    text = (
        "Negative controls included field blanks and NTCs. "
        "A positive control used synthetic DNA from a mock community."
    )
    response = json.dumps(
        [
            {"field": "neg_cont_0_1", "raw_value": "1", "quote_id": "Q001"},
            {"field": "pos_cont_0_1", "raw_value": "1", "quote_id": "Q002"},
        ]
    )
    facts = detect_llm_judged_search_facts(
        MockLLMBackend(responses=[response]),
        (("Controls", text),),
        locator_prefix="paper:PMC1",
    )
    by_type = {fact.fact_type_candidate: fact for fact in facts}
    assert by_type["neg_cont_0_1"].raw_value == "1"
    assert by_type["pos_cont_0_1"].raw_value == "1"


def test_detect_llm_judged_search_facts_sets_control_zero_from_explicit_absence():
    text = "No negative controls were used. Positive controls were not included."
    response = json.dumps(
        [
            {"field": "neg_cont_0_1", "raw_value": "0", "quote_id": "Q001"},
            {"field": "pos_cont_0_1", "raw_value": "0", "quote_id": "Q001"},
        ]
    )
    facts = detect_llm_judged_search_facts(
        MockLLMBackend(responses=[response]),
        (("Controls", text),),
        locator_prefix="paper:PMC1",
    )
    by_type = {fact.fact_type_candidate: fact for fact in facts}
    assert by_type["neg_cont_0_1"].raw_value == "0"
    assert by_type["pos_cont_0_1"].raw_value == "0"


def test_detect_llm_judged_search_facts_mirrors_not_a_control_to_sibling_field():
    """Regression guard for a real live-paper finding: the model reliably
    emitted only ONE of neg_cont_0_1/pos_cont_0_1 for a "not a real
    control" quote (e.g. "quality control" in a data-processing sense)
    instead of both, leaving the other blank. A quote judged "0" (not a
    control at all) is equally not a positive control and not a negative
    control, so it's mirrored to the sibling field deterministically."""
    text = "The raw sequencing reads underwent quality control before assembly."
    response = json.dumps([{"field": "neg_cont_0_1", "raw_value": "0", "quote_id": "Q001"}])
    facts = detect_llm_judged_search_facts(
        MockLLMBackend(responses=[response]),
        (("Controls", text),),
        locator_prefix="paper:PMC1",
    )
    by_type = {fact.fact_type_candidate: fact for fact in facts}
    assert by_type["neg_cont_0_1"].raw_value == "0"
    assert by_type["pos_cont_0_1"].raw_value == "0"
    assert by_type["pos_cont_0_1"].source_locator == by_type["neg_cont_0_1"].source_locator


def test_detect_llm_judged_search_facts_never_mirrors_a_positive_control_result():
    """A real "1" for one side says nothing definitive about the other --
    a paper can report its positive control separately from its negative
    control -- so a "1" is never auto-mirrored to the sibling field. The
    sibling still ends up "0" here, but via the separate not-found
    default (no quote resolved it), never via mirroring the "1"."""
    text = "A positive control using synthetic DNA from a mock community was included."
    response = json.dumps([{"field": "pos_cont_0_1", "raw_value": "1", "quote_id": "Q001"}])
    facts = detect_llm_judged_search_facts(
        MockLLMBackend(responses=[response]),
        (("Controls", text),),
        locator_prefix="paper:PMC1",
    )
    by_type = {fact.fact_type_candidate: fact for fact in facts}
    assert by_type["pos_cont_0_1"].raw_value == "1"
    assert by_type["neg_cont_0_1"].raw_value == "0"
    assert by_type["neg_cont_0_1"].confidence_metadata["detector"] == "control_not_found_default"


def test_detect_llm_judged_search_facts_defaults_both_controls_to_zero_when_never_mentioned():
    """Per an explicit user request ("i also see no mention of +/-
    controls ... I think they should both be 0"): when a paper's text
    never even raises a "control"/"blank" candidate at all, the confident
    default is "0" for both fields, not a blank indistinguishable from
    "never checked"."""
    facts = detect_llm_judged_search_facts(
        MockLLMBackend(responses=["[]"]),
        (("Methods", "Water samples were filtered and DNA was extracted using a standard kit."),),
        locator_prefix="paper:PMC1",
    )
    by_type = {fact.fact_type_candidate: fact for fact in facts}
    assert by_type["neg_cont_0_1"].raw_value == "0"
    assert by_type["pos_cont_0_1"].raw_value == "0"
    assert by_type["neg_cont_0_1"].confidence_metadata["detector"] == "control_not_found_default"


def test_detect_llm_judged_search_facts_control_prefers_positive_evidence_over_negative_across_quotes():
    """If the model judges "1" for one candidate quote and "0" for
    another (e.g. a confusing paper, or one candidate genuinely stronger
    than another), real evidence of a control being used must win over a
    quote that merely didn't support one -- matches the old regex
    mechanism's own "explicit positive evidence wins" precedent."""
    text = "Field blanks were included at each site. Quality control was performed on the raw reads."
    response = json.dumps(
        [
            {"field": "neg_cont_0_1", "raw_value": "1", "quote_id": "Q001"},
            {"field": "neg_cont_0_1", "raw_value": "0", "quote_id": "Q002"},
        ]
    )
    facts = detect_llm_judged_search_facts(
        MockLLMBackend(responses=[response]),
        (("Controls", text),),
        locator_prefix="paper:PMC1",
    )
    by_type = {fact.fact_type_candidate: fact for fact in facts}
    assert by_type["neg_cont_0_1"].raw_value == "1"


def test_detect_llm_judged_search_facts_control_never_confuses_unrelated_control_usage():
    """A bare "control" mention with no contamination/assay-validation
    purpose (e.g. "quality control" on sequencing reads) must not be
    reported as a real positive/negative sample control -- the exact
    false-positive risk a bare keyword regex couldn't avoid, and the
    reason this moved to LLM context judgment. The word "control" here
    still passes the keyword gate (unlike "controlled", which the word-
    boundary search_terms match rejects outright) -- this specifically
    exercises the LLM's own judgment, not just the keyword filter. Per an
    explicit user request, an unresolved neg/pos_cont_0_1 now confidently
    defaults to "0" rather than staying blank, so both fields ARE present
    here, just correctly "0" rather than any misreported "1"."""
    text = "Quality control was performed on the raw sequencing reads to check for errors."
    facts = detect_llm_judged_search_facts(
        MockLLMBackend(responses=["[]"]),
        (("Bioinformatics", text),),
        locator_prefix="paper:PMC1",
        exclude_field_names=frozenset({"chimera_check_method", "trim_method", "trim_param", "tax_assign_cat", "tax_class_other"}),
    )
    by_type = {fact.fact_type_candidate: fact for fact in facts}
    assert by_type["neg_cont_0_1"].raw_value == "0"
    assert by_type["pos_cont_0_1"].raw_value == "0"


def test_detect_controlled_search_facts_uses_active_flags_and_pipe_delimited_matches():
    text = (
        "PCR targeted 12S rRNA and COI with the MiFish-U assay. "
        "The TaqMan probe used FAM reporter dye and BHQ-1 quencher. "
        "Libraries used the MiSeq Reagent Kit v3 chemistry for water and sediment samples."
    )
    flag_facts = detect_text_search_flags((("Methods", text),), locator_prefix="paper:PMC1")
    controlled = detect_controlled_search_facts(
        (("Methods", text),),
        locator_prefix="paper:PMC1",
        # pcr_0_1 is no longer detected by detect_text_search_flags itself
        # (derived from section-category detection instead, see
        # extraction/section_categories.py::derive_pcr_0_1_from_category_detection) --
        # added explicitly here since these tests exercise pcr_0_1-gated
        # controlled-search behavior directly, not that derivation.
        active_flags=frozenset(fact.fact_type_candidate for fact in flag_facts) | {"pcr_0_1"},
    )

    by_type = {fact.fact_type_candidate: fact for fact in controlled}
    assert by_type["target_gene"].raw_value == "12S rRNA | COI"
    assert by_type["probeReporter"].raw_value == "FAM | reporter"
    assert by_type["probeQuencher"].raw_value == "BHQ-1 | quencher"
    assert by_type["seq_kit"].raw_value == "MiSeq Reagent Kit v3"
    assert by_type["sample_type"].raw_value == "water | sediment"
    assert by_type["target_gene"].entity_level.value == "study"
    assert by_type["target_gene"].confidence_metadata["activated_by_flags"] == ["pcr_0_1"]
    assert "assay_name" not in by_type


def test_detect_controlled_search_facts_skips_conditional_fields_without_flag():
    controlled = detect_controlled_search_facts(
        (("Methods", "The assay name MiFish-U appears in background only."),),
        locator_prefix="paper:PMC1",
        active_flags=frozenset(),
    )

    assert {fact.fact_type_candidate for fact in controlled} == set()


def test_detect_controlled_search_facts_extracts_sterilise_method_as_direct_quotes():
    text = (
        "Sampling bottles were rinsed with 10% bleach and DI water. "
        "Single-use equipment was used during filtration. "
        "Sterile technique was used by all staff."
    )
    controlled = detect_controlled_search_facts(
        (("Methods", text),),
        locator_prefix="paper:PMC1",
        active_flags=frozenset(),
    )

    by_type = {fact.fact_type_candidate: fact for fact in controlled}
    assert by_type["sterilise_method"].raw_value == (
        "Sampling bottles were rinsed with 10% bleach and DI water. | "
        "Single-use equipment was used during filtration."
    )
    assert "Sterile technique was used" not in by_type["sterilise_method"].raw_value
    assert by_type["sterilise_method"].evidence_quote == by_type["sterilise_method"].raw_value


def test_detect_controlled_search_facts_extracts_sterile_containers_and_tools():
    text = (
        "Each sub-section was put into sterile bags and stored onboard at -20C. "
        "The dried samples were transferred to sterile tubes until further use. "
        "Subsampling of microbiology samples used sterile 10 mL cutoff syringes."
    )
    controlled = detect_controlled_search_facts(
        (("Sampling", text),),
        locator_prefix="paper:PMC1",
        active_flags=frozenset(),
    )

    by_type = {fact.fact_type_candidate: fact for fact in controlled}
    assert by_type["sterilise_method"].raw_value == (
        "Each sub-section was put into sterile bags and stored onboard at -20C. | "
        "The dried samples were transferred to sterile tubes until further use. | "
        "Subsampling of microbiology samples used sterile 10 mL cutoff syringes."
    )


def test_detect_controlled_search_facts_keeps_full_method_phrases():
    text = (
        "PCR was conducted in a DNA Engine Tetrad2 Thermal Cycler. "
        "Amplicons were pyrosequenced using 454-FLX with Titanium chemistry at the "
        "Genome Sequencing and Analysis Facility (GSAF) at the University of Texas at Austin. "
        "Control wells received no settlement cue."
    )
    flag_facts = detect_text_search_flags((("Methods", text),), locator_prefix="paper:PMC1")
    controlled = detect_controlled_search_facts(
        (("Methods", text),),
        locator_prefix="paper:PMC1",
        # pcr_0_1 is no longer detected by detect_text_search_flags itself
        # (derived from section-category detection instead, see
        # extraction/section_categories.py::derive_pcr_0_1_from_category_detection) --
        # added explicitly here since these tests exercise pcr_0_1-gated
        # controlled-search behavior directly, not that derivation.
        active_flags=frozenset(fact.fact_type_candidate for fact in flag_facts) | {"pcr_0_1"},
    )

    by_type = {fact.fact_type_candidate: fact for fact in [*flag_facts, *controlled]}
    assert by_type["thermocycler"].raw_value == "DNA Engine Tetrad2 Thermal Cycler"
    assert by_type["seq_kit"].raw_value == "Titanium chemistry"
    assert "sequencing_location" not in by_type


def test_detect_controlled_search_facts_extracts_library_index_kit_not_extraction_kit():
    text = (
        "DNA was extracted using PowerLyze DNA extraction kits. "
        "The amplicon library was prepared using Nextera XT Index Kit (Illumina Inc.) "
        "according to the 16S Metagenomic Sequencing Library preparation protocol."
    )

    controlled = detect_controlled_search_facts(
        (("DNA extraction and amplicon sequencing", text),),
        locator_prefix="paper:PMC1",
        active_flags=frozenset(),
    )

    by_type = {fact.fact_type_candidate: fact for fact in controlled}
    assert by_type["seq_kit"].raw_value == "Nextera XT Index Kit (Illumina Inc.)"
    assert "PowerLyze" not in by_type["seq_kit"].raw_value


def test_detect_controlled_search_facts_extracts_directional_primer_names_and_sequences():
    text = (
        "Each PCR contained 0.1 uM of the universal Btn-SPR-F forward primer "
        "(5′ CCTATCCCCTGTGTGCCTTGGCAGTCTCAGTCTCAAAGACTAAGCCATGC 3′, "
        "underlined stretch matches SP-F-30 primer) and 0.1 uM of unique "
        "reverse primer containing a 4-bp barcode "
        "(5′ CCATCTCATCCCTGCGTGTCTCCGACTCAG**TACT**TTACAGAGCTGGAATTACCG 3′, "
        "underlined stretch matches SP-R-540 primer, bold indicates 4 bp barcode)."
    )
    flag_facts = detect_text_search_flags((("Methods", text),), locator_prefix="paper:PMC1")
    controlled = detect_controlled_search_facts(
        (("Methods", text),),
        locator_prefix="paper:PMC1",
        # pcr_0_1 is no longer detected by detect_text_search_flags itself
        # (derived from section-category detection instead, see
        # extraction/section_categories.py::derive_pcr_0_1_from_category_detection) --
        # added explicitly here since these tests exercise pcr_0_1-gated
        # controlled-search behavior directly, not that derivation.
        active_flags=frozenset(fact.fact_type_candidate for fact in flag_facts) | {"pcr_0_1"},
    )

    by_type = {fact.fact_type_candidate: fact for fact in controlled}
    assert by_type["forward_primer_name"].raw_value == "Btn-SPR-F | SP-F-30"
    assert by_type["reverse_primer_name"].raw_value == "SP-R-540"
    assert by_type["forward_primer_sequence"].raw_value == (
        "CCTATCCCCTGTGTGCCTTGGCAGTCTCAGTCTCAAAGACTAAGCCATGC"
    )
    assert by_type["reverse_primer_sequence"].raw_value == (
        "CCATCTCATCCCTGCGTGTCTCCGACTCAGTACTTTACAGAGCTGGAATTACCG"
    )
    assert by_type["forward_primer_sequence"].evidence_quote == text
    assert by_type["reverse_primer_name"].evidence_quote == text


def test_detect_controlled_search_facts_extracts_rrna_f_r_primer_names():
    text = (
        "The V3-V4 region of the 16S rRNA gene was amplified using universal primers "
        "16S rRNA F and 16S rRNA R."
    )

    controlled = detect_controlled_search_facts(
        (("Methods", text),),
        locator_prefix="paper:PMC1",
        active_flags=frozenset({"pcr_0_1"}),
    )

    by_type = {fact.fact_type_candidate: fact for fact in controlled}
    assert by_type["forward_primer_name"].raw_value == "16S rRNA F"
    assert by_type["reverse_primer_name"].raw_value == "16S rRNA R"
    assert by_type["forward_primer_name"].evidence_quote == text
    assert by_type["reverse_primer_name"].evidence_quote == text


def test_detect_controlled_search_facts_does_not_match_bare_its_as_target_gene():
    text = "PCR amplified 18S rRNA; its sequence reads were clustered after filtering."
    flag_facts = detect_text_search_flags((("Methods", text),), locator_prefix="paper:PMC1")
    controlled = detect_controlled_search_facts(
        (("Methods", text),),
        locator_prefix="paper:PMC1",
        # pcr_0_1 is no longer detected by detect_text_search_flags itself
        # (derived from section-category detection instead, see
        # extraction/section_categories.py::derive_pcr_0_1_from_category_detection) --
        # added explicitly here since these tests exercise pcr_0_1-gated
        # controlled-search behavior directly, not that derivation.
        active_flags=frozenset(fact.fact_type_candidate for fact in flag_facts) | {"pcr_0_1"},
    )

    by_type = {fact.fact_type_candidate: fact for fact in controlled}
    assert by_type["target_gene"].raw_value == "18S rRNA"


def test_detect_controlled_search_facts_prioritizes_specific_ssu_target_gene_names():
    text = (
        "PCR amplified 16S rRNA, 16S, and 18S markers. "
        "The final assay targeted 16S SSU rRNA and 18S rRNA SSU regions."
    )
    flag_facts = detect_text_search_flags((("Methods", text),), locator_prefix="paper:PMC1")
    controlled = detect_controlled_search_facts(
        (("Methods", text),),
        locator_prefix="paper:PMC1",
        # pcr_0_1 is no longer detected by detect_text_search_flags itself
        # (derived from section-category detection instead, see
        # extraction/section_categories.py::derive_pcr_0_1_from_category_detection) --
        # added explicitly here since these tests exercise pcr_0_1-gated
        # controlled-search behavior directly, not that derivation.
        active_flags=frozenset(fact.fact_type_candidate for fact in flag_facts) | {"pcr_0_1"},
    )

    by_type = {fact.fact_type_candidate: fact for fact in controlled}
    assert by_type["target_gene"].raw_value == "16S rRNA SSU | 18S rRNA SSU"


def test_detect_controlled_search_facts_expands_coordinated_ssu_target_gene_names():
    text = "Taxonomic classification extracted small subunit rRNA (16S and 18S SSU rRNA) reads."
    flag_facts = detect_text_search_flags((("Methods", f"PCR was performed. {text}"),), locator_prefix="paper:PMC1")
    controlled = detect_controlled_search_facts(
        (("Methods", text),),
        locator_prefix="paper:PMC1",
        # pcr_0_1 is no longer detected by detect_text_search_flags itself
        # (derived from section-category detection instead, see
        # extraction/section_categories.py::derive_pcr_0_1_from_category_detection) --
        # added explicitly here since these tests exercise pcr_0_1-gated
        # controlled-search behavior directly, not that derivation.
        active_flags=frozenset(fact.fact_type_candidate for fact in flag_facts) | {"pcr_0_1"},
    )

    by_type = {fact.fact_type_candidate: fact for fact in controlled}
    assert by_type["target_gene"].raw_value == "16S rRNA SSU | 18S rRNA SSU"


def test_detect_controlled_search_facts_keeps_rrna_when_no_ssu_target_gene_name():
    text = "The PCR assay amplified 16S rRNA and later refers to the 16S amplicons."
    flag_facts = detect_text_search_flags((("Methods", text),), locator_prefix="paper:PMC1")
    controlled = detect_controlled_search_facts(
        (("Methods", text),),
        locator_prefix="paper:PMC1",
        # pcr_0_1 is no longer detected by detect_text_search_flags itself
        # (derived from section-category detection instead, see
        # extraction/section_categories.py::derive_pcr_0_1_from_category_detection) --
        # added explicitly here since these tests exercise pcr_0_1-gated
        # controlled-search behavior directly, not that derivation.
        active_flags=frozenset(fact.fact_type_candidate for fact in flag_facts) | {"pcr_0_1"},
    )

    by_type = {fact.fact_type_candidate: fact for fact in controlled}
    assert by_type["target_gene"].raw_value == "16S rRNA"


def test_detect_controlled_search_facts_extracts_biological_rep_integer_not_pcr_reps():
    text = (
        "At each station, three independent samples were collected. "
        "PCR reactions were performed in triplicate."
    )
    controlled = detect_controlled_search_facts(
        (("Methods", text),),
        locator_prefix="paper:PMC1",
        active_flags=frozenset(),
    )

    by_type = {fact.fact_type_candidate: fact for fact in controlled}
    assert by_type["biological_rep"].raw_value == "3"
    assert by_type["biological_rep"].evidence_quote == (
        "At each station, three independent samples were collected."
    )


def test_detect_controlled_search_facts_does_not_treat_ecological_species_specific_as_targeted_assay():
    """Regression guard for a real gold-data false positive
    (PeerJ 10.7717/peerj.333): "species-specific"/"taxon-specific" alone
    are too generic to imply a targeted PCR/qPCR assay -- the paper's own
    Discussion says "coral species-specific cue preferences", an
    ecological statement with zero PCR/probe/qPCR content nearby, which
    previously still classified as assay_type "targeted"."""
    text = (
        "To visualize coral species-specific cue preferences, both principal "
        "components analysis (PCA) and non-metric multidimensional scaling "
        "(NMDS) ordination were used."
    )
    controlled = detect_controlled_search_facts(
        (("Discussion", text),),
        locator_prefix="paper:PMC1",
        active_flags=frozenset({"pcr_0_1"}),
    )

    assert not any(fact.fact_type_candidate == "assay_type" for fact in controlled)


def test_detect_controlled_search_facts_classifies_custom_pcr_mixture_not_commercial_master_mix():
    """Regression guard for a real gold-data false positive
    (PeerJ 10.7717/peerj.333): the paper describes a custom-assembled PCR
    mixture (ExTaq buffer/polymerase, Pfu polymerase) amplified on a
    "DNA Engine Tetrad2 Thermal Cycler (Bio-Rad, Hercules, CA, USA)" --
    commercial_mm previously matched the bare "Bio-Rad" thermocycler-brand
    mention as if it were a commercial master-mix product. Now the whole
    PCR-mixture sentence is captured, classified as custom_mm (no
    master-mix product/brand named) rather than commercial_mm."""
    text = (
        "Each 30 ul polymerase chain reaction (PCR) mixture contained 10 ng of DNA template, 0.1 uM "
        "forward primer, 0.2 mM dNTP, 3 ul 10X ExTaq buffer, 0.025 U ExTaq "
        "Polymerase (Takara Biotechnology) and 0.0125 U Pfu Polymerase "
        "(Agilent Technologies), and was amplified using a DNA Engine "
        "Tetrad2 Thermal Cycler (Bio-Rad, Hercules, CA, USA)."
    )
    controlled = detect_controlled_search_facts(
        (("Methods", text),),
        locator_prefix="paper:PMC1",
        active_flags=frozenset({"pcr_0_1"}),
    )

    by_type = {fact.fact_type_candidate: fact for fact in controlled}
    assert "commercial_mm" not in by_type
    assert by_type["custom_mm"].raw_value == text
    assert by_type["custom_mm"].evidence_quote == text


def test_detect_controlled_search_facts_classifies_reagent_listing_without_the_word_mixture_as_custom():
    """Regression guard for a real miss (PLOS ONE 10.1371/
    journal.pone.0303937): the paper lists PCR reagents (a named
    polymerase, buffer, dNTPs) without ever using the word "mixture"/"mix"
    at all -- the old _PCR_MIXTURE_MARKERS_RE-only trigger required one of
    those words and missed this sentence entirely."""
    text = (
        "16S rRNA gene was performed with 0.02 ul of Phusion High Fidelity DNA polymerase, "
        "1X Phusion HF Buffer and 200 uM of dNTPs (New England Biolabs, USA)."
    )
    controlled = detect_controlled_search_facts(
        (("Methods", text),),
        locator_prefix="paper:PMC1",
        active_flags=frozenset({"pcr_0_1"}),
    )

    by_type = {fact.fact_type_candidate: fact for fact in controlled}
    assert "commercial_mm" not in by_type
    assert by_type["custom_mm"].raw_value == text


def test_detect_controlled_search_facts_classifies_named_master_mix_product_as_commercial():
    text = "PCR was performed using PowerUp SYBR Green Master Mix according to the manufacturer's instructions."
    controlled = detect_controlled_search_facts(
        (("Methods", text),),
        locator_prefix="paper:PMC1",
        active_flags=frozenset({"pcr_0_1"}),
    )

    by_type = {fact.fact_type_candidate: fact for fact in controlled}
    assert "custom_mm" not in by_type
    assert by_type["commercial_mm"].raw_value == text


def test_detect_controlled_search_facts_pcr_mixture_phrase_requires_pcr_0_1_flag():
    controlled = detect_controlled_search_facts(
        (("Methods", "The PCR mixture contained 10 ng of DNA template and 0.2 mM dNTP."),),
        locator_prefix="paper:PMC1",
        active_flags=frozenset(),
    )

    assert {fact.fact_type_candidate for fact in controlled} == set()


def test_detect_controlled_search_facts_does_not_match_well_or_cycle_counts_as_biological_rep():
    """Regression guard for a real gold-data false positive
    (PeerJ 10.7717/peerj.333): a bare "n = <number>"/"N = <number>" pattern
    previously matched both "(n = 4 well replicates per cue...)" (a
    technical well count) and "...with N = 17-24 depending on the sample"
    (a PCR cycle count), producing a nonsensical joined biological_rep
    value ("4 | 17") from two numbers that have nothing to do with
    biological replication."""
    text = (
        "Cue samples were finely ground with a mortar and pestle shortly "
        "before the settlement trials and a single drop of the resulting "
        "uniform slurry was added to each well (n = 4 well replicates per "
        "cue, randomly assigning cues to wells). The PCR mixture was "
        "amplified with a cycling profile of 94 C 5 min-(94 C 40 s-55 C 2 "
        "min-72 C 60 s) x N-72 C 10 min, with N = 17-24 depending on the "
        "sample."
    )
    controlled = detect_controlled_search_facts(
        (("Methods", text),),
        locator_prefix="paper:PMC1",
        active_flags=frozenset(),
    )

    assert not any(fact.fact_type_candidate == "biological_rep" for fact in controlled)


def test_detect_controlled_search_facts_extracts_explicit_biological_rep_negative():
    text = "All sediment depths were analyzed without replicates because sample material was limited."
    controlled = detect_controlled_search_facts(
        (("Methods", text),),
        locator_prefix="paper:PMC1",
        active_flags=frozenset(),
    )

    by_type = {fact.fact_type_candidate: fact for fact in controlled}
    assert by_type["biological_rep_presence"].raw_value == "FALSE"
    assert by_type["biological_rep_presence"].evidence_quote == text


def test_detect_controlled_search_facts_does_not_treat_technical_rep_negative_as_biological_rep():
    text = "PCR was performed without technical replicates."
    controlled = detect_controlled_search_facts(
        (("Methods", text),),
        locator_prefix="paper:PMC1",
        active_flags=frozenset(),
    )

    assert "biological_rep_presence" not in {fact.fact_type_candidate for fact in controlled}


def test_detect_controlled_search_facts_classifies_assay_type_and_keeps_evidence():
    text = (
        "We used qPCR with a hydrolysis probe for species-specific detection. "
        "A separate metabarcoding workflow used universal primers for community profiling."
    )
    controlled = detect_controlled_search_facts(
        (("Methods", text),),
        locator_prefix="paper:PMC1",
        active_flags=frozenset(),
    )

    by_type = {fact.fact_type_candidate: fact for fact in controlled}
    assert by_type["assay_type"].raw_value == "targeted | metabarcoding"
    assert by_type["assay_type"].evidence_quote == (
        "We used qPCR with a hydrolysis probe for species-specific detection. | "
        "A separate metabarcoding workflow used universal primers for community profiling."
    )


def test_detect_controlled_search_facts_extracts_trimmomatic_minlen_parameter():
    text = (
        "Raw paired-end reads were processed with Trimmomatic using the parameters "
        "ILLUMINACLIP:adapters.fa:2:30:10 SLIDINGWINDOW:4:20 MINLEN:80 before downstream analysis."
    )
    controlled = detect_controlled_search_facts(
        (("Bioinformatics", text),),
        locator_prefix="paper:PMC1",
        active_flags=frozenset(),
    )

    by_type = {fact.fact_type_candidate: fact for fact in controlled}
    assert by_type["adapter_trimming_method"].raw_value == "Trimmomatic"
    assert by_type["length_filtering_tool"].raw_value == "Trimmomatic"
    assert by_type["minimum_read_length"].raw_value == "80 bp"
    assert by_type["minimum_read_length"].evidence_quote == text


def test_detect_controlled_search_facts_extracts_paired_end_library_layout():
    text = "The raw paired-end reads were subjected to quality check using FastQC."
    controlled = detect_controlled_search_facts(
        (("Bioinformatics", text),),
        locator_prefix="paper:PMC1",
        active_flags=frozenset(),
    )

    by_type = {fact.fact_type_candidate: fact for fact in controlled}
    assert by_type["library_layout"].raw_value == "paired end"
    assert by_type["library_layout"].evidence_quote == text


def test_detect_controlled_search_facts_extracts_two_by_300_as_paired_end_library_layout():
    text = "Sequencing was performed on an Illumina MiSeq platform using 2 x 300 bp chemistry."
    controlled = detect_controlled_search_facts(
        (("Sequencing", text),),
        locator_prefix="paper:PMC1",
        active_flags=frozenset(),
    )

    by_type = {fact.fact_type_candidate: fact for fact in controlled}
    assert by_type["library_layout"].raw_value == "paired end"
    assert by_type["library_layout"].confidence_metadata["matches"][0]["matched_terms"] == ["2 x 300"]


def test_detect_controlled_search_facts_extracts_checksum_methods_directly():
    text = "FASTQ files were verified using MD5 checksums, with legacy SHA-1 hashes also reported."
    controlled = detect_controlled_search_facts(
        (("Data availability", text),),
        locator_prefix="paper:PMC1",
        active_flags=frozenset(),
    )

    by_type = {fact.fact_type_candidate: fact for fact in controlled}
    assert by_type["checksum_method"].raw_value == "MD5 | other:"
    assert by_type["checksum_method"].evidence_quote == text


def test_detect_controlled_search_facts_extracts_trimmomatic_reads_below_phrase():
    """This text never mentions "adapter" at all -- Trimmomatic is
    described purely as a length/quality filter, so adapter_trimming_method
    must correctly stay unpopulated (see the real-paper regression test
    below for the case where the same tool genuinely does both, in
    separately-worded clauses)."""
    text = (
        "The sequence libraries were trimmed using trimmomatic, removing all reads below 500 bp, "
        "with a phred quality below 3 for the start and the end of the reads."
    )
    controlled = detect_controlled_search_facts(
        (("Bioinformatics", text),),
        locator_prefix="paper:PMC1",
        active_flags=frozenset(),
    )

    by_type = {fact.fact_type_candidate: fact for fact in controlled}
    assert "adapter_trimming_method" not in by_type
    assert by_type["length_filtering_tool"].raw_value == "trimmomatic"
    assert by_type["minimum_read_length"].raw_value == "500 bp"


def test_detect_controlled_search_facts_extracts_usearch_trimmed_to_length_phrase():
    text = (
        "The raw sequencing reads were quality filtered and trimmed to 220 bp "
        "using the USEARCH v11.0.667 pipeline."
    )
    controlled = detect_controlled_search_facts(
        (("Bioinformatics", text),),
        locator_prefix="paper:PMC1",
        active_flags=frozenset(),
    )

    by_type = {fact.fact_type_candidate: fact for fact in controlled}
    assert by_type["length_filtering_tool"].raw_value == "USEARCH v11.0.667"
    assert by_type["minimum_read_length"].raw_value == "220 bp"


def test_detect_controlled_search_facts_distinguishes_seqprep_adapter_removal_from_trimmomatic_length_filtering():
    """Regression guard for a real paper (ISME J 10.1093/ismejo/wrae013):
    describes adapter removal via SeqPrep and quality/length trimming via
    Trimmomatic as separate numbered clauses inside one long compound
    sentence. Confirmed this exact shape previously made the (context-blind)
    Trimmomatic detector wrongly stamp "Trimmomatic" onto
    adapter_trimming_method too, ~250 characters away from any adapter
    mention -- proximity-gated context checking (not just "somewhere in the
    same sentence") is required to tell the two clauses apart."""
    text = (
        "Quality trimming was conducted by: (i) removing Illumina adapters using SeqPrep 1.2 with "
        "default settings targeting the adapter sequences; (ii) remove any leftover PhiX control "
        "sequences by mapping the reads to the PhiX genome using bowtie2 2.3.5.1, and (iii) remove "
        "low quality and short reads using Trimmomatic 0.39 with settings: LEADING:20, TRAILING:20, "
        "and MINLEN:80."
    )
    controlled = detect_controlled_search_facts(
        (("Bioinformatics", text),),
        locator_prefix="paper:PMC1",
        active_flags=frozenset(),
    )

    by_type = {fact.fact_type_candidate: fact for fact in controlled}
    assert by_type["adapter_trimming_method"].raw_value == "SeqPrep 1.2"
    assert by_type["length_filtering_tool"].raw_value == "Trimmomatic 0.39"
    assert by_type["minimum_read_length"].raw_value == "80 bp"


def test_detect_controlled_search_facts_attributes_trimmomatic_to_both_fields_when_both_are_nearby():
    """When Trimmomatic genuinely does both jobs in one invocation
    (adapter clipping AND quality/length trimming, both named close
    together), both fields should correctly get it -- the fix gates on
    nearby context, not on "never allow the same tool in both fields"."""
    text = "Reads were processed with Trimmomatic to remove adapters and trim low quality bases."
    controlled = detect_controlled_search_facts(
        (("Bioinformatics", text),),
        locator_prefix="paper:PMC1",
        active_flags=frozenset(),
    )

    by_type = {fact.fact_type_candidate: fact for fact in controlled}
    assert by_type["adapter_trimming_method"].raw_value == "Trimmomatic"
    assert by_type["length_filtering_tool"].raw_value == "Trimmomatic"


def test_quote_candidates_for_llm_judged_library_prep_search_are_narrow():
    candidates = quote_candidates_for_llm_judged_search(
        (
            (
                "Library construction",
                "A two-step PCR was used for library construction. "
                "Libraries were cleaned with AMPure beads and quantified with Qubit. "
                "Water samples were filtered on deck.",
            ),
        )
    )

    assert [candidate.quote_id for candidate in candidates] == ["Q001", "Q002"]
    assert candidates[0].field_names == ("barcoding_pcr_appr",)
    assert candidates[0].text == "A two-step PCR was used for library construction."
    assert candidates[1].field_names == ("lib_screen",)
    assert candidates[1].text == "Libraries were cleaned with AMPure beads and quantified with Qubit."


def test_quote_candidates_for_llm_judged_barcoding_maps_two_round_pcr():
    candidates = quote_candidates_for_llm_judged_search(
        (
            (
                "Amplicon sequencing",
                "Amplicon of the 16S rRNA gene was prepared using the two-round PCR amplification strategy.",
            ),
        )
    )

    assert len(candidates) == 1
    assert candidates[0].field_names == ("barcoding_pcr_appr",)


def test_detect_llm_judged_lib_screen_keeps_best_priority_value():
    backend = MockLLMBackend(
        responses=[
            json.dumps(
                [
                    {"field": "lib_screen", "raw_value": "Qubit quantification", "quote_id": "Q001"},
                    {
                        "field": "lib_screen",
                        "raw_value": "libraries were size-selected using BluePippin and quantified with Qubit",
                        "quote_id": "Q001",
                    },
                ]
            )
        ]
    )

    facts = detect_llm_judged_search_facts(
        backend,
        (
            (
                "Library preparation",
                "The libraries were size-selected using BluePippin and quantified with Qubit before pooling.",
            ),
        ),
        locator_prefix="paper:PMC1",
    )

    by_type = {fact.fact_type_candidate: fact for fact in facts}
    assert by_type["lib_screen"].raw_value == (
        "libraries were size-selected using BluePippin and quantified with Qubit"
    )


def test_quote_candidates_for_llm_judged_error_rate_searches_are_targeted():
    candidates = quote_candidates_for_llm_judged_search(
        (
            (
                "Bioinformatics",
                "Reads were denoised using DADA2 filterAndTrim with maxEE=2. "
                "Unrelated quality language without a configured threshold follows.",
            ),
        )
    )

    assert len(candidates) == 1
    assert candidates[0].field_names == ("error_rate_tool", "error_rate_type", "otu_clust_tool")
    assert candidates[0].text == "Reads were denoised using DADA2 filterAndTrim with maxEE=2."


def test_quote_candidates_for_llm_judged_chimera_search_is_targeted():
    candidates = quote_candidates_for_llm_judged_search(
        (("Bioinformatics", "Chimeras were removed with VSEARCH --uchime_denovo in de novo mode."),)
    )

    assert len(candidates) == 1
    assert candidates[0].field_names == ("demux_tool", "error_rate_tool", "chimera_check_method")


def test_quote_candidates_for_llm_judged_demux_search_is_targeted():
    candidates = quote_candidates_for_llm_judged_search(
        (("Bioinformatics", "Reads were demultiplexed using QIIME 2 demux emp-paired with no barcode mismatches."),)
    )

    assert len(candidates) == 1
    assert candidates[0].field_names == ("demux_tool", "error_rate_tool")


def test_quote_candidates_for_llm_judged_inhibition_search_is_targeted():
    candidates = quote_candidates_for_llm_judged_search(
        (
            (
                "qPCR",
                "PCR inhibition was tested with an internal positive control, and inhibited extracts were diluted 1:10.",
            ),
        )
    )

    assert len(candidates) == 1
    # Also a candidate for neg_cont_0_1/pos_cont_0_1 now: "positive
    # control" contains the bare word "control", one of their own search
    # terms -- correctly so, since the LLM (not this deterministic gate)
    # is what decides whether the quote actually supports either.
    assert candidates[0].field_names == ("inhibition_check_0_1", "inhibition_check", "neg_cont_0_1", "pos_cont_0_1")


def test_quote_candidates_for_llm_judged_chimera_requires_explicit_context():
    candidates = quote_candidates_for_llm_judged_search(
        (
            (
                "Bioinformatics",
                "Reads were denoised using DADA2 filterAndTrim with maxEE=2. "
                "Hybrid amplicons were removed with VSEARCH --uchime_ref.",
            ),
        )
    )

    assert len(candidates) == 2
    assert "chimera_check_method" not in candidates[0].field_names
    assert "chimera_check_method" in candidates[1].field_names


def test_quote_candidates_for_llm_judged_trim_search_requires_specific_context():
    candidates = quote_candidates_for_llm_judged_search(
        (
            (
                "Bioinformatics",
                "Reads were filtered with Trimmomatic SLIDINGWINDOW:4:20. "
                "PCR primer sequences were trimmed using Cutadapt v4.2 with -e 0.1 and -O 5.",
            ),
        )
    )

    assert len(candidates) == 2
    assert "trim_method" not in candidates[0].field_names
    assert "trim_param" not in candidates[0].field_names
    assert "trim_method" in candidates[1].field_names
    assert "trim_param" in candidates[1].field_names


def test_quote_candidates_for_llm_judged_min_reads_search_is_targeted():
    candidates = quote_candidates_for_llm_judged_search(
        (("Bioinformatics", "Low abundance ASVs with fewer than 10 reads were removed using phyloseq prune_taxa."),)
    )

    assert len(candidates) == 1
    assert candidates[0].field_names == ("min_reads_tool",)


def test_quote_candidates_for_llm_judged_otu_clustering_search_is_targeted():
    candidates = quote_candidates_for_llm_judged_search(
        (
            (
                "Bioinformatics",
                "OTUs were clustered at 97% similarity using VSEARCH --cluster_fast.",
            ),
        )
    )

    assert len(candidates) == 1
    assert "otu_clust_tool" in candidates[0].field_names
    assert "otu_clust_cutoff" in candidates[0].field_names


def test_quote_candidates_for_llm_judged_otu_db_search_is_targeted():
    candidates = quote_candidates_for_llm_judged_search(
        (
            (
                "Bioinformatics",
                "Taxonomy was assigned using a naive Bayes classifier trained on SILVA release 138.",
            ),
            ("Software", "The analysis used version 1.2 of the workflow."),
        )
    )

    assert len(candidates) == 1
    assert candidates[0].field_names == ("otu_db", "tax_assign_cat")


def test_quote_candidates_for_llm_judged_tax_assignment_requires_context():
    candidates = quote_candidates_for_llm_judged_search(
        (
            (
                "Software",
                "The analysis used BLASTn and MegaBLAST in separate utility scripts.",
            ),
            (
                "Bioinformatics",
                "ASVs were classified taxonomically using BLASTn against GenBank with a 97% sequence identity threshold.",
            ),
        )
    )

    assert len(candidates) == 1
    assert "tax_assign_cat" in candidates[0].field_names


def test_quote_candidates_for_llm_judged_assay_name_search_is_targeted():
    candidates = quote_candidates_for_llm_judged_search(
        (("PCR", "The MiFish-U assay amplified the 12S marker using the MiFish primer set."),)
    )

    assert len(candidates) == 1
    assert candidates[0].field_names == ("assay_name",)


def test_quote_candidates_for_llm_judged_assay_name_ignores_software_versions():
    candidates = quote_candidates_for_llm_judged_search(
        (
            (
                "Genome annotation",
                "Genome annotation was conducted using Prokka v1.13, GTDB-tk v2.3.0, GeneSpy v1.2, and IQ-TREE v1.5.5.",
            ),
        )
    )

    assert all("assay_name" not in candidate.field_names for candidate in candidates)


def test_quote_candidates_for_llm_judged_assay_name_keeps_marker_region_context():
    candidates = quote_candidates_for_llm_judged_search(
        (
            (
                "PCR",
                "Amplicons of the 16S V4 region were prepared using the Uni519F/806R primer pair.",
            ),
        )
    )

    assert len(candidates) == 1
    assert candidates[0].field_names == ("assay_name",)


def test_quote_candidates_for_llm_judged_search_respects_excluded_fields():
    candidates = quote_candidates_for_llm_judged_search(
        (("PCR", "The MiFish-U assay amplified the 12S marker."),),
        exclude_field_names=frozenset({"assay_name"}),
    )

    assert candidates == ()


def test_quote_candidates_for_adapter_fields_accept_overhang_and_tailed_primer_terms():
    candidates = quote_candidates_for_llm_judged_search(
        (
            (
                "Library construction",
                "Forward overhang sequences were added with adapter-tailed primers, "
                "and the reverse overhang contained an Illumina overhang sequence.",
            ),
        )
    )

    assert len(candidates) == 1
    assert "adapter_forward" in candidates[0].field_names
    assert "adapter_reverse" in candidates[0].field_names


def test_detect_llm_judged_search_facts_accepts_quote_id_and_stores_literal_quote():
    def respond(prompt: str) -> str:
        assert "Q001 [barcoding_pcr_appr]" in prompt
        assert "Q002 [lib_screen]" in prompt
        assert "Q003 [adapter_forward" in prompt
        return json.dumps(
            [
                {"field": "barcoding_pcr_appr", "raw_value": "two-step PCR", "quote_id": "Q001"},
                {"field": "lib_screen", "raw_value": "cleaned with AMPure beads", "quote_id": "Q002"},
                {
                    "field": "adapter_forward",
                    "raw_value": "AATGATACGGCGACCACCGAGATCTACACGCT",
                    "quote_id": "Q003",
                },
            ]
        )

    text = (
        "A two-step PCR was used for library construction. "
        "Libraries were cleaned with AMPure beads before sequencing. "
        "The forward adapter sequence was AATGATACGGCGACCACCGAGATCTACACGCT."
    )
    backend = MockLLMBackend(label="judge", responses=respond)
    facts = detect_llm_judged_search_facts(
        backend,
        (("Library construction", text),),
        locator_prefix="paper:PMC1",
    )

    by_type = {fact.fact_type_candidate: fact for fact in facts}
    assert by_type["barcoding_pcr_appr"].raw_value == "two-step PCR"
    assert by_type["barcoding_pcr_appr"].evidence_quote == (
        "A two-step PCR was used for library construction."
    )
    assert by_type["lib_screen"].raw_value == "cleaned with AMPure beads"
    assert by_type["lib_screen"].evidence_quote == "Libraries were cleaned with AMPure beads before sequencing."
    assert by_type["adapter_forward"].raw_value == "AATGATACGGCGACCACCGAGATCTACACGCT"
    assert by_type["adapter_forward"].evidence_quote == (
        "The forward adapter sequence was AATGATACGGCGACCACCGAGATCTACACGCT."
    )
    assert by_type["lib_screen"].confidence_metadata["matches"][0]["quote_id"] == "Q002"
    assert backend.calls[0]["max_tokens"] == 512


def test_detect_llm_judged_search_facts_extracts_assay_name_from_quote():
    backend = MockLLMBackend(
        responses=[
            json.dumps(
                [
                    {"field": "assay_name", "raw_value": "MiFish-U", "quote_id": "Q001"},
                ]
            )
        ]
    )

    facts = detect_llm_judged_search_facts(
        backend,
        (("PCR", "The MiFish-U assay amplified the 12S marker."),),
        locator_prefix="paper:PMC1",
    )

    by_type = {fact.fact_type_candidate: fact for fact in facts}
    assert by_type["assay_name"].raw_value == "MiFish-U"
    assert by_type["chimera_check_method"].raw_value == "no chimeric recorded."


def test_detect_llm_judged_search_facts_extracts_multi_value_assay_target_taxa_pipe_joined():
    backend = MockLLMBackend(
        responses=[
            json.dumps(
                [
                    {"field": "assay_target_taxa", "raw_value": "vertebrates", "quote_id": "Q001"},
                    {"field": "assay_target_taxa", "raw_value": "Atlantic salmon (Salmo salar)", "quote_id": "Q002"},
                ]
            )
        ]
    )

    facts = detect_llm_judged_search_facts(
        backend,
        (
            (
                "PCR",
                "Universal primers targeting vertebrates were used. Species-specific primers for "
                "Atlantic salmon (Salmo salar) were also included. The study focused on teleost fishes "
                "in the survey area.",
            ),
        ),
        locator_prefix="paper:PMC1",
    )

    by_type = {fact.fact_type_candidate: fact for fact in facts}
    assert by_type["assay_target_taxa"].raw_value == "vertebrates | Atlantic salmon (Salmo salar)"
    assert by_type["assay_target_taxa"].support_type.value == "explicit"
    assert "study_target_taxonomic_scope" not in by_type


def test_detect_llm_judged_search_facts_defaults_target_taxa_fields_to_not_found_without_context():
    """A bare taxon mention with no assay/study-scope context word must not
    even reach the LLM as a candidate -- this is the exact over-fill
    failure mode the retired title/abstract/keyword-only extractor hit on
    a real paper (ecological background taxa, results taxa)."""
    facts = detect_llm_judged_search_facts(
        MockLLMBackend(responses=[]),
        (("Results", "Sharks and rays were the most abundant vertebrates observed at the reef site."),),
        locator_prefix="paper:PMC1",
        exclude_field_names=frozenset({"chimera_check_method", "trim_method", "trim_param", "tax_assign_cat", "tax_class_other"}),
    )

    by_type = {fact.fact_type_candidate: fact for fact in facts}
    assert by_type["assay_target_taxa"].raw_value == "not found"
    assert by_type["assay_target_taxa"].support_type.value == "deterministically_derived"
    assert "study_target_taxonomic_scope" not in by_type


def test_detect_llm_judged_search_facts_extracts_information_withheld_verbatim():
    text = "Exact coordinates were withheld to protect an endangered species."
    response = json.dumps([{"field": "informationWithheld", "raw_value": text, "quote_id": "Q001"}])
    facts = detect_llm_judged_search_facts(
        MockLLMBackend(responses=[response]),
        (("Data Availability", text),),
        locator_prefix="paper:PMC1",
    )
    by_type = {fact.fact_type_candidate: fact for fact in facts}
    assert by_type["informationWithheld"].raw_value == text
    assert by_type["informationWithheld"].support_type.value == "explicit"


def test_detect_llm_judged_search_facts_information_withheld_absent_for_unrelated_text():
    """This function itself has no 'not found' fallback for this field
    (unlike the PCR-family terms above) -- unresolved here just means no
    candidate quote existed. mapping/faire.py's own post-mapping step
    supplies the actual "Nothing indicated as withheld" default at a
    later stage, deliberately not here: doing it here would risk a
    paper-text default permanently blocking a real value found later by a
    supplement-text pass, under the generic "oldest fact wins" rule."""
    facts = detect_llm_judged_search_facts(
        MockLLMBackend(responses=[]),
        (("Methods", "Water samples were collected at 5 m depth and filtered on site."),),
        locator_prefix="paper:PMC1",
    )
    assert "informationWithheld" not in {fact.fact_type_candidate for fact in facts}


def test_detect_llm_judged_search_facts_information_withheld_catches_reasonable_request_boilerplate():
    """Regression guard for a real live-paper miss (10.1038/s42003-024-06136-2):
    the Data Availability Statement said "All other data are available
    from the corresponding authors on reasonable request." -- a common
    boilerplate pattern that none of the older search terms (which assumed
    stronger phrasing like "not publicly available" or "confidential")
    ever matched, so this field was silently left blank for a real paper
    that DOES restrict access to some of its data."""
    text = (
        "Microbial abundance and relevant geochemical data are available with this paper. "
        "All other data are available from the corresponding authors on reasonable request."
    )
    response = json.dumps(
        [
            {
                "field": "informationWithheld",
                "raw_value": "All other data are available from the corresponding authors on reasonable request.",
                "quote_id": "Q001",
            }
        ]
    )
    facts = detect_llm_judged_search_facts(
        MockLLMBackend(responses=[response]),
        (("Data Availability Statement", text),),
        locator_prefix="paper:PMC1",
    )
    by_type = {fact.fact_type_candidate: fact for fact in facts}
    assert by_type["informationWithheld"].raw_value == (
        "All other data are available from the corresponding authors on reasonable request."
    )


def test_detect_llm_judged_search_facts_defaults_barcoding_to_one_step_when_pcr_true():
    backend = MockLLMBackend(responses=[])

    facts = detect_llm_judged_search_facts(
        backend,
        (("PCR", "PCR amplification was performed with primers 515F and 806R."),),
        locator_prefix="paper:PMC1",
        active_flags=frozenset({"pcr_0_1"}),
    )

    by_type = {fact.fact_type_candidate: fact for fact in facts}
    assert by_type["barcoding_pcr_appr"].raw_value == "one-step PCR"
    assert by_type["barcoding_pcr_appr"].support_type.value == "deterministically_derived"
    assert by_type["chimera_check_method"].raw_value == "no chimeric recorded."


def test_detect_llm_judged_search_facts_does_not_default_barcoding_when_resolved():
    facts = detect_llm_judged_search_facts(
        MockLLMBackend(responses=[]),
        (("PCR", "PCR amplification was performed with primers 515F and 806R."),),
        locator_prefix="paper:PMC1",
        active_flags=frozenset({"pcr_0_1"}),
        exclude_field_names=frozenset(
            {
                "barcoding_pcr_appr", "trim_method", "trim_param", "tax_assign_cat", "tax_class_other",
                "assay_target_taxa", "study_target_taxonomic_scope", "neg_cont_0_1", "pos_cont_0_1",
            }
        ),
    )

    assert [fact.fact_type_candidate for fact in facts] == ["chimera_check_method"]


def test_detect_llm_judged_search_facts_defaults_chimera_to_not_recorded_without_context():
    facts = detect_llm_judged_search_facts(
        MockLLMBackend(responses=[]),
        (("Bioinformatics", "Reads were processed using DADA2 filterAndTrim with maxEE=2."),),
        locator_prefix="paper:PMC1",
    )

    by_type = {fact.fact_type_candidate: fact for fact in facts}
    assert by_type["chimera_check_method"].raw_value == "no chimeric recorded."
    assert by_type["chimera_check_method"].support_type.value == "deterministically_derived"


def test_detect_llm_judged_search_facts_defaults_trim_fields_to_not_found_without_context():
    facts = detect_llm_judged_search_facts(
        MockLLMBackend(responses=[]),
        (("Bioinformatics", "Reads were denoised using DADA2 filterAndTrim with maxEE=2."),),
        locator_prefix="paper:PMC1",
        exclude_field_names=frozenset({"chimera_check_method"}),
    )

    by_type = {fact.fact_type_candidate: fact for fact in facts}
    assert by_type["trim_method"].raw_value == "not found"
    assert by_type["trim_param"].raw_value == "not found"
    assert by_type["trim_method"].support_type.value == "deterministically_derived"
    assert by_type["tax_assign_cat"].raw_value == "not found"
    assert "tax_class_other" not in by_type


def test_detect_llm_judged_trim_fields_extract_method_and_pipe_parameters():
    def respond(prompt: str) -> str:
        assert "trim_method" in prompt
        assert "trim_param" in prompt
        return json.dumps(
            [
                {"field": "trim_method", "raw_value": "Cutadapt v4.2", "quote_id": "Q001"},
                {"field": "trim_param", "raw_value": "-e 0.1", "quote_id": "Q001"},
                {"field": "trim_param", "raw_value": "-O 5", "quote_id": "Q001"},
                {"field": "trim_method", "raw_value": "primer sequences were trimmed", "quote_id": "Q001"},
            ]
        )

    facts = detect_llm_judged_search_facts(
        MockLLMBackend(label="judge", responses=respond),
        (
            (
                "Bioinformatics",
                "PCR primer sequences were trimmed using Cutadapt v4.2 with -e 0.1 and -O 5.",
            ),
        ),
        locator_prefix="paper:PMC1",
        exclude_field_names=frozenset({"chimera_check_method"}),
    )

    by_type = {fact.fact_type_candidate: fact for fact in facts}
    assert by_type["trim_method"].raw_value == "Cutadapt v4.2"
    assert by_type["trim_param"].raw_value == "-e 0.1 | -O 5"


def test_detect_llm_judged_error_rate_facts_order_values_by_configured_terms():
    def respond(prompt: str) -> str:
        assert "Q001 [error_rate_tool, error_rate_type]" in prompt
        return json.dumps(
            [
                {"field": "error_rate_tool", "raw_value": "Trimmomatic", "quote_id": "Q001"},
                {"field": "error_rate_tool", "raw_value": "DADA2", "quote_id": "Q001"},
                {"field": "error_rate_type", "raw_value": "Phred score", "quote_id": "Q001"},
                {"field": "error_rate_type", "raw_value": "expected error rate", "quote_id": "Q001"},
            ]
        )

    backend = MockLLMBackend(label="judge", responses=respond)
    facts = detect_llm_judged_search_facts(
        backend,
        (
            (
                "Bioinformatics",
                "Reads were filtered with DADA2 filterAndTrim maxEE=2 and then trimmed with Trimmomatic SLIDINGWINDOW:4:20.",
            ),
        ),
        locator_prefix="paper:PMC1",
    )

    by_type = {fact.fact_type_candidate: fact for fact in facts}
    assert by_type["error_rate_tool"].raw_value == "DADA2"
    assert by_type["error_rate_type"].raw_value == "expected error rate | Phred score"


def test_detect_llm_judged_error_rate_type_accepts_generic_quality_filtered_phrase():
    backend = MockLLMBackend(
        responses=[
            json.dumps(
                [
                    {"field": "error_rate_tool", "raw_value": "USEARCH v11.0.667", "quote_id": "Q001"},
                    {"field": "error_rate_type", "raw_value": "quality filtered", "quote_id": "Q001"},
                ]
            )
        ]
    )

    facts = detect_llm_judged_search_facts(
        backend,
        (
            (
                "Bioinformatics",
                "The raw sequencing reads were quality filtered and trimmed to 220 bp using the USEARCH v11.0.667 pipeline.",
            ),
        ),
        locator_prefix="paper:PMC1",
        exclude_field_names=frozenset(
            {"chimera_check_method", "trim_method", "trim_param", "tax_assign_cat", "tax_class_other"}
        ),
    )

    by_type = {fact.fact_type_candidate: fact for fact in facts}
    assert by_type["error_rate_tool"].raw_value == "USEARCH v11.0.667"
    assert by_type["error_rate_type"].raw_value == "quality filtered"


def test_detect_llm_judged_demux_tool_keeps_best_priority_value():
    def respond(prompt: str) -> str:
        assert "Q001 [demux_tool, error_rate_tool]" in prompt
        return json.dumps(
            [
                {"field": "demux_tool", "raw_value": "bcl2fastq", "quote_id": "Q001"},
                {"field": "demux_tool", "raw_value": "QIIME 2 demux emp-paired with no barcode mismatches", "quote_id": "Q001"},
            ]
        )

    backend = MockLLMBackend(label="judge", responses=respond)
    facts = detect_llm_judged_search_facts(
        backend,
        (
            (
                "Bioinformatics",
                "Reads were demultiplexed using QIIME 2 demux emp-paired with no barcode mismatches after bcl2fastq conversion.",
            ),
        ),
        locator_prefix="paper:PMC1",
    )

    by_type = {fact.fact_type_candidate: fact for fact in facts}
    assert by_type["demux_tool"].raw_value == "QIIME 2 demux emp-paired with no barcode mismatches"


def test_detect_llm_judged_inhibition_fields_extract_status_and_method():
    backend = MockLLMBackend(
        responses=[
            json.dumps(
                [
                    {"field": "inhibition_check_0_1", "raw_value": "1", "quote_id": "Q001"},
                    {
                        "field": "inhibition_check",
                        "raw_value": "PCR inhibition was tested with an internal positive control; inhibited extracts were diluted 1:10",
                        "quote_id": "Q001",
                    },
                ]
            )
        ]
    )

    facts = detect_llm_judged_search_facts(
        backend,
        (
            (
                "qPCR",
                "PCR inhibition was tested with an internal positive control, and inhibited extracts were diluted 1:10.",
            ),
        ),
        locator_prefix="paper:PMC1",
    )

    by_type = {fact.fact_type_candidate: fact for fact in facts}
    assert by_type["inhibition_check_0_1"].raw_value == "1"
    assert by_type["inhibition_check"].raw_value.startswith("PCR inhibition was tested")


def test_detect_llm_judged_inhibition_zero_requires_explicit_quote():
    backend = MockLLMBackend(
        responses=[json.dumps([{"field": "inhibition_check_0_1", "raw_value": "0", "quote_id": "Q001"}])]
    )

    facts = detect_llm_judged_search_facts(
        backend,
        (("qPCR", "PCR inhibition was not tested in this study."),),
        locator_prefix="paper:PMC1",
        exclude_field_names=frozenset(
            {"chimera_check_method", "trim_method", "trim_param", "tax_assign_cat", "tax_class_other", "assay_target_taxa", "study_target_taxonomic_scope", "neg_cont_0_1", "pos_cont_0_1"}
        ),
    )

    assert len(facts) == 1
    assert facts[0].raw_value == "0"


def test_detect_llm_judged_inhibition_rejects_invalid_status_value():
    backend = MockLLMBackend(
        responses=[json.dumps([{"field": "inhibition_check_0_1", "raw_value": "yes", "quote_id": "Q001"}])]
    )

    facts = detect_llm_judged_search_facts(
        backend,
        (("qPCR", "PCR inhibition was tested with an internal positive control."),),
        locator_prefix="paper:PMC1",
        exclude_field_names=frozenset(
            {"chimera_check_method", "trim_method", "trim_param", "tax_assign_cat", "tax_class_other", "assay_target_taxa", "study_target_taxonomic_scope", "neg_cont_0_1", "pos_cont_0_1"}
        ),
    )

    assert facts == []


def test_detect_llm_judged_inhibition_bsa_only_can_be_rejected_by_llm():
    backend = MockLLMBackend(responses=[json.dumps([])])

    facts = detect_llm_judged_search_facts(
        backend,
        (("PCR", "BSA was added to each PCR reaction."),),
        locator_prefix="paper:PMC1",
        exclude_field_names=frozenset(
            {"chimera_check_method", "trim_method", "trim_param", "tax_assign_cat", "tax_class_other", "assay_target_taxa", "study_target_taxonomic_scope", "neg_cont_0_1", "pos_cont_0_1"}
        ),
    )

    assert facts == []


def test_detect_llm_judged_min_reads_tool_pipes_multiple_values():
    backend = MockLLMBackend(
        responses=[
            json.dumps(
                [
                    {"field": "min_reads_tool", "raw_value": "phyloseq prune_taxa", "quote_id": "Q001"},
                    {"field": "min_reads_tool", "raw_value": "decontam isContaminant", "quote_id": "Q002"},
                ]
            )
        ]
    )

    facts = detect_llm_judged_search_facts(
        backend,
        (
            (
                "Bioinformatics",
                "Low abundance ASVs with fewer than 10 reads were removed using phyloseq prune_taxa. "
                "Contaminants were removed with decontam isContaminant using blank thresholds.",
            ),
        ),
        locator_prefix="paper:PMC1",
    )

    by_type = {fact.fact_type_candidate: fact for fact in facts}
    assert by_type["min_reads_tool"].raw_value == "phyloseq prune_taxa | decontam isContaminant"


def test_detect_llm_judged_otu_clustering_fields_keep_best_values():
    def respond(prompt: str) -> str:
        assert "otu_clust_tool" in prompt
        assert "otu_clust_cutoff" in prompt
        return json.dumps(
            [
                {"field": "otu_clust_tool", "raw_value": "VSEARCH --cluster_fast", "quote_id": "Q001"},
                {"field": "otu_clust_tool", "raw_value": "mothur cluster", "quote_id": "Q001"},
                {"field": "otu_clust_cutoff", "raw_value": "97", "quote_id": "Q001"},
                {"field": "otu_clust_cutoff", "raw_value": "99", "quote_id": "Q001"},
            ]
        )

    facts = detect_llm_judged_search_facts(
        MockLLMBackend(label="judge", responses=respond),
        (("Bioinformatics", "OTUs were clustered at 97% similarity using VSEARCH --cluster_fast."),),
        locator_prefix="paper:PMC1",
    )

    by_type = {fact.fact_type_candidate: fact for fact in facts}
    assert by_type["otu_clust_tool"].raw_value == "VSEARCH --cluster_fast"
    assert by_type["otu_clust_cutoff"].raw_value == "97"


def test_detect_llm_judged_otu_db_keeps_best_database_value():
    backend = MockLLMBackend(
        responses=[
            json.dumps(
                [
                    {"field": "otu_db", "raw_value": "naive Bayes classifier", "quote_id": "Q001"},
                    {"field": "otu_db", "raw_value": "SILVA release 138", "quote_id": "Q001"},
                ]
            )
        ]
    )

    facts = detect_llm_judged_search_facts(
        backend,
        (
            (
                "Bioinformatics",
                "Taxonomy was assigned using a naive Bayes classifier trained on SILVA release 138.",
            ),
        ),
        locator_prefix="paper:PMC1",
    )

    by_type = {fact.fact_type_candidate: fact for fact in facts}
    assert by_type["otu_db"].raw_value == "SILVA release 138"


def test_detect_llm_judged_tax_assignment_fields_keep_best_values():
    """tax_assign_cat is a controlled-enum classification (allowed_values):
    a hallucinated full-sentence "value" must be rejected by
    _valid_llm_judged_value even though a genuine quote does support it,
    while the real enum classification for the very same quote survives."""
    def respond(prompt: str) -> str:
        assert "tax_assign_cat" in prompt
        return json.dumps(
            [
                {
                    "field": "tax_assign_cat",
                    "raw_value": "ASVs were classified taxonomically using BLASTn against GenBank",
                    "quote_id": "Q001",
                },
                {
                    "field": "tax_assign_cat",
                    "raw_value": "sequence similarity",
                    "quote_id": "Q001",
                },
            ]
        )

    facts = detect_llm_judged_search_facts(
        MockLLMBackend(label="judge", responses=respond),
        (
            (
                "Bioinformatics",
                "ASVs were classified taxonomically using BLASTn against GenBank with a 97% sequence identity threshold.",
            ),
        ),
        locator_prefix="paper:PMC1",
        exclude_field_names=frozenset(
            {"chimera_check_method", "trim_method", "trim_param", "assay_target_taxa", "study_target_taxonomic_scope", "neg_cont_0_1", "pos_cont_0_1"}
        ),
    )

    by_type = {fact.fact_type_candidate: fact for fact in facts}
    assert by_type["tax_assign_cat"].raw_value == "sequence similarity"
    assert by_type["tax_assign_cat"].support_type.value == "inferred"


def test_detect_llm_judged_error_rate_type_accepts_other_prefix():
    backend = MockLLMBackend(
        responses=[
            json.dumps(
                [
                    {
                        "field": "error_rate_type",
                        "raw_value": "other: nanopore read quality model",
                        "quote_id": "Q001",
                    }
                ]
            )
        ]
    )

    facts = detect_llm_judged_search_facts(
        backend,
        (("Bioinformatics", "Reads were filtered with Dorado using a nanopore read quality model."),),
        locator_prefix="paper:PMC1",
        exclude_field_names=frozenset(
            {"chimera_check_method", "trim_method", "trim_param", "tax_assign_cat", "tax_class_other", "assay_target_taxa", "study_target_taxonomic_scope", "neg_cont_0_1", "pos_cont_0_1"}
        ),
    )

    assert len(facts) == 1
    assert facts[0].raw_value == "other: nanopore read quality model"


def test_detect_llm_judged_chimera_method_extracts_supported_phrase():
    backend = MockLLMBackend(
        responses=[
            json.dumps(
                [
                    {
                        "field": "chimera_check_method",
                        "raw_value": "VSEARCH --uchime_denovo de novo",
                        "quote_id": "Q001",
                    }
                ]
            )
        ]
    )

    facts = detect_llm_judged_search_facts(
        backend,
        (("Bioinformatics", "Chimeras were removed with VSEARCH --uchime_denovo in de novo mode."),),
        locator_prefix="paper:PMC1",
        exclude_field_names=frozenset(
            {"trim_method", "trim_param", "tax_assign_cat", "tax_class_other", "assay_target_taxa", "study_target_taxonomic_scope", "neg_cont_0_1", "pos_cont_0_1"}
        ),
    )

    assert len(facts) == 1
    assert facts[0].fact_type_candidate == "chimera_check_method"
    assert facts[0].raw_value == "VSEARCH --uchime_denovo de novo"


def test_detect_llm_judged_search_facts_rejects_bad_quote_ids_and_vocab_values():
    backend = MockLLMBackend(
        responses=[
            json.dumps(
                [
                    {"field": "barcoding_pcr_appr", "raw_value": "banana", "quote_id": "Q001"},
                    {"field": "lib_screen", "raw_value": "AMPure cleanup", "quote_id": "Q999"},
                    {"field": "unknown_field", "raw_value": "two-step PCR", "quote_id": "Q001"},
                ]
            )
        ]
    )

    facts = detect_llm_judged_search_facts(
        backend,
        (("Library construction", "A two-step PCR was used for library construction."),),
        locator_prefix="paper:PMC1",
        exclude_field_names=frozenset(
            {"chimera_check_method", "trim_method", "trim_param", "tax_assign_cat", "tax_class_other", "assay_target_taxa", "study_target_taxonomic_scope", "neg_cont_0_1", "pos_cont_0_1"}
        ),
    )

    assert facts == []
