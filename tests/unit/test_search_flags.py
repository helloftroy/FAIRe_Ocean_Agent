import json

from fair_ocean_agent.config import MIN_LLM_MAX_OUTPUT_TOKENS
from fair_ocean_agent.extraction.search_flags import (
    _barcoding_pcr_appr_keyword_match,
    confirm_value_described_as_depth,
    detect_controlled_search_facts,
    detect_llm_judged_search_facts,
    detect_phix_percentage_facts,
    detect_text_search_flags,
    quote_candidates_for_llm_judged_search,
)
from fair_ocean_agent.extraction.section_categories import derive_pcr_0_1_from_category_detection
from fair_ocean_agent.llm.mock import MockLLMBackend


def test_detect_phix_percentage_facts_matches_common_real_phrasings():
    """A percentage number co-occurring with "PhiX" in the same sentence
    is unambiguous enough for a plain deterministic regex -- per an
    explicit user request to replace the old LLM-askable phix_percentage
    taxonomy field with "a quick search ... for PhiX or its variations"."""
    for text in (
        "A 15% PhiX spike-in was added to offset low sequence diversity.",
        "PhiX control was included at 1% to improve base calling.",
        "The library was spiked with PhiX (10%) prior to sequencing.",
    ):
        facts = detect_phix_percentage_facts((("Methods", text),), locator_prefix="paper:PMC1")
        assert len(facts) == 1
        assert facts[0].fact_type_candidate == "phix_perc"
        assert facts[0].support_type.value == "deterministically_derived"


def test_detect_phix_percentage_facts_requires_both_signals_in_one_sentence():
    """A percentage elsewhere in the paper with no PhiX mention nearby
    (or vice versa) must not be misattributed as the PhiX percentage."""
    facts = detect_phix_percentage_facts(
        (
            (
                "Methods",
                "Approximately 15% of reads failed quality control. "
                "PhiX was used as a sequencing control.",
            ),
        ),
        locator_prefix="paper:PMC1",
    )
    assert facts == []


def test_detect_phix_percentage_facts_first_match_wins_across_texts():
    facts = detect_phix_percentage_facts(
        (
            ("Methods", "A 15% PhiX spike-in was used."),
            ("Supplement", "PhiX was added at 20%."),
        ),
        locator_prefix="paper:PMC1",
    )
    assert len(facts) == 1
    assert facts[0].raw_value == "15"


def test_confirm_value_described_as_depth_finds_real_water_depth_sentence():
    """Real audit (10.1093/ismejo/wrae013, STUDY-295abf4a8f43): BioSample's
    own elev=34m for a sediment sample is confirmed by the paper's own
    text as the site's water depth, not elevation."""
    backend = MockLLMBackend(responses=[json.dumps({"confirmed": True, "quote_id": "Q001"})])
    text = (
        "Cores were subsampled from a box corer at a site with 34 m water depth "
        "(Lat 59.8559, Long: 23.26695) previously shown to have high methanotrophic activity."
    )
    quote = confirm_value_described_as_depth(backend, "34", [("Methods", text)])
    assert quote == (
        "Cores were subsampled from a box corer at a site with 34 m water depth "
        "(Lat 59.8559, Long: 23.26695) previously shown to have high methanotrophic activity."
    )


def test_confirm_value_described_as_depth_returns_none_when_llm_does_not_confirm():
    backend = MockLLMBackend(responses=[json.dumps({"confirmed": False, "quote_id": ""})])
    text = "Samples were incubated at 34 degrees C with a depth of 5 mm agar."
    quote = confirm_value_described_as_depth(backend, "34", [("Methods", text)])
    assert quote is None


def test_confirm_value_described_as_depth_never_calls_llm_without_a_candidate_sentence():
    def fail_if_called(prompt: str) -> str:
        raise AssertionError("LLM should never be called with no candidate sentences")

    backend = MockLLMBackend(responses=fail_if_called)
    text = "The site was visited on 34 separate occasions during the survey period."
    quote = confirm_value_described_as_depth(backend, "34", [("Methods", text)])
    assert quote is None


def test_detect_text_search_flags_records_probe_flag_only_pcr_0_1_moved_to_category_detection():
    """pcr_0_1 is not detect_text_search_flags's own concern -- see
    extraction/section_categories.py::derive_pcr_0_1_from_category_detection,
    its own independent PCR/qPCR/ddPCR-mention regex scan."""
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
    only the verb "amplified". pcr_0_1 must still activate, or every
    downstream flag-gated PCR checklist field silently becomes
    unreachable for a real paper phrased this way."""
    texts = (("Methods", "The target region was amplified using primers X and Y."),)
    pcr_0_1_fact = derive_pcr_0_1_from_category_detection(list(texts))
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


def test_detect_controlled_search_facts_extracts_nextflex_rapid_dna_seq_kit():
    text = (
        "Libraries were prepared with the NEXTflex™ Rapid DNA-Seq kit "
        "according to the manufacturer's protocol."
    )

    controlled = detect_controlled_search_facts(
        (("Methods", text),),
        locator_prefix="paper:PMC1",
        active_flags=frozenset(),
    )

    by_type = {fact.fact_type_candidate: fact for fact in controlled}
    assert by_type["seq_kit"].raw_value == "NEXTflex™ Rapid DNA-Seq kit"


def test_detect_controlled_search_facts_extracts_plain_nextflex_dna_seq_kit():
    text = "Sequencing libraries were generated using a NEXTflex Rapid DNA-Seq kit."

    controlled = detect_controlled_search_facts(
        (("Methods", text),),
        locator_prefix="paper:PMC1",
        active_flags=frozenset(),
    )

    by_type = {fact.fact_type_candidate: fact for fact in controlled}
    assert by_type["seq_kit"].raw_value == "NEXTflex Rapid DNA-Seq kit"


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


def test_detect_controlled_search_facts_extracts_frontiers_primer_pair_from_table():
    text = (
        "TABLE 2. Primer | Sequence (5'-3') | Target | Use | Reference\n"
        "U515F | TGYCAGCMGCCGCCGTAA | Prokaryote | S | Hoshino and Inagaki, 2017\n"
        "U806R | GGACTACHVGGGTWTCTAAT | Prokaryote | S | Walters et al., 2011\n"
        "B27F | AGRGTTYGATYMTGGCTCAG | Bacteria | D | Lane, 1991\n"
        "B357R | CTGCWGCCNCCCGTAGG | Bacteria | D | Herlemann et al., 2011\n"
        "A806F | ATTAGATACCCSBGTAGTCC | Archaea | D | Raskin et al., 1994\n"
        "A958R | YCCGGCGTTGAMTCCAATT | Archaea | D | DeLong, 1992\n"
        "Digital PCR was performed with domain-specific primers B27F-B357R and A806F-A958R. "
        "The V3-V4 hyper-variable region of the 16S rRNA gene was amplified by PCR "
        "using universal primers U515F/U806R (Table 2)."
    )

    controlled = detect_controlled_search_facts(
        (("Methods", text),),
        locator_prefix="paper:PMC5476839",
        active_flags=frozenset({"pcr_0_1"}),
    )

    by_type = {fact.fact_type_candidate: fact for fact in controlled}
    assert by_type["forward_primer_name"].raw_value == "B27F | A806F | U515F"
    assert by_type["reverse_primer_name"].raw_value == "B357R | A958R | U806R"
    assert by_type["forward_primer_sequence"].raw_value == (
        "AGRGTTYGATYMTGGCTCAG | ATTAGATACCCSBGTAGTCC | TGYCAGCMGCCGCCGTAA"
    )
    assert by_type["reverse_primer_sequence"].raw_value == (
        "CTGCWGCCNCCCGTAGG | YCCGGCGTTGAMTCCAATT | GGACTACHVGGGTWTCTAAT"
    )


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
    mention as if it were a commercial master-mix product. The mixture
    sentence is classified as custom_mm (no master-mix product/brand named),
    but the stored value stops before thermocycler/cycling-program details."""
    text = (
        "Each 30 ul polymerase chain reaction (PCR) mixture contained 10 ng of DNA template, 0.1 uM "
        "forward primer, 0.2 mM dNTP, 3 ul 10X ExTaq buffer, 0.025 U ExTaq "
        "Polymerase (Takara Biotechnology) and 0.0125 U Pfu Polymerase "
        "(Agilent Technologies), and was amplified using a DNA Engine "
        "Tetrad2 Thermal Cycler (Bio-Rad, Hercules, CA, USA)."
    )
    expected = (
        "Each 30 ul polymerase chain reaction (PCR) mixture contained 10 ng of DNA template, 0.1 uM "
        "forward primer, 0.2 mM dNTP, 3 ul 10X ExTaq buffer, 0.025 U ExTaq "
        "Polymerase (Takara Biotechnology) and 0.0125 U Pfu Polymerase "
        "(Agilent Technologies)"
    )
    controlled = detect_controlled_search_facts(
        (("Methods", text),),
        locator_prefix="paper:PMC1",
        active_flags=frozenset({"pcr_0_1"}),
    )

    by_type = {fact.fact_type_candidate: fact for fact in controlled}
    assert "commercial_mm" not in by_type
    assert by_type["custom_mm"].raw_value == expected
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


def test_detect_controlled_search_facts_classifies_shotgun_metagenomics_alongside_a_false_metabarcoding_cue():
    """Real gap found live (PMC10988111 / ISME Communications
    10.1093/ismeco/ycae036, "Metagenomic insights into jellyfish-associated
    microbiome dynamics"): a pure shotgun-metagenomics paper's own
    Introduction separately contrasts its method against "16S rRNA
    amplicon sequencing" used by *other* studies -- that sentence still
    (correctly, given this classifier can't attribute "we did X" vs
    "others did X") trips the metabarcoding bucket. Per explicit user
    direction, list both rather than have the false metabarcoding cue
    crowd out the real metagenomics signal."""
    text = (
        "We used shotgun metagenomic sequencing to provide a detailed characterization of microbiome "
        "succession over strobilation. However, the typical marker gene (e.g. 16S rRNA) amplicon "
        "sequencing applied in existing studies cannot offer phylogenetic resolution."
    )
    controlled = detect_controlled_search_facts(
        (("Introduction", text),),
        locator_prefix="paper:PMC1",
        active_flags=frozenset(),
    )

    by_type = {fact.fact_type_candidate: fact for fact in controlled}
    assert by_type["assay_type"].raw_value == "other:metagenomics | metabarcoding"


def test_detect_controlled_search_facts_classifies_metabarcoding_from_otu_and_gene_amplicon_phrasing():
    """Real gap found live (10.3390/microorganisms10030558): this paper's
    own explicit "amplicon sequencing of 16S rRNA and cbbL gene" framing
    sentence lives in the Introduction, which select_relevant_sections
    excludes entirely (Methods-only scoping) -- so the classifier never
    saw it in the real pipeline. The *Methods* section alone still
    describes unmistakably metabarcoding-shaped methodology for both its
    16S marker and its cbbL functional-gene marker -- OTU clustering and
    "gene amplicons" sequencing -- without ever using the words "amplicon
    sequencing" together."""
    text = (
        "The resulting sequences were clustered as operational taxonomic units (OTUs) at 97% sequence "
        "identity. Sequenced using an Illumina HiSeq platform with 2 x 250 bp paired-end reads for 16S "
        "rRNA gene amplicons, and Illumina MiSeq platform with 2 x 300 bp paired-end reads for cbbL "
        "gene amplicons."
    )
    controlled = detect_controlled_search_facts(
        (("Methods", text),),
        locator_prefix="paper:PMC1",
        active_flags=frozenset(),
    )

    by_type = {fact.fact_type_candidate: fact for fact in controlled}
    assert by_type["assay_type"].raw_value == "metabarcoding"


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


def test_detect_controlled_search_facts_extracts_reads_became_shorter_than_discarded_phrase():
    text = "Reads that became shorter than 250 bp after this trimming step were discarded."
    controlled = detect_controlled_search_facts(
        (("Bioinformatics", text),),
        locator_prefix="paper:PMC1",
        active_flags=frozenset(),
    )

    by_type = {fact.fact_type_candidate: fact for fact in controlled}
    assert by_type["minimum_read_length"].raw_value == "250 bp"


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

    assert [candidate.quote_id for candidate in candidates] == ["Q001"]
    assert candidates[0].field_names == ("barcoding_pcr_appr",)
    assert candidates[0].text == "A two-step PCR was used for library construction."


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


def test_quote_candidates_for_llm_judged_barcoding_maps_second_round_pcr():
    """Real gap found live (10.3389/fmicb.2017.01135): "index and adapter
    were added to the purified product during the eight cycles of
    second-round PCR" never matched any barcoding_pcr_appr search term at
    all -- the inserted "-round" broke the fixed "second PCR" phrase,
    silently defaulting the whole study to one-step PCR and leaving the
    entire pcr2_* field family blank."""
    candidates = quote_candidates_for_llm_judged_search(
        (
            (
                "Methods",
                "Index and adapter were added to the purified product during the eight cycles of "
                "second-round PCR using KAPA HiFi HotStart Ready mix.",
            ),
        )
    )
    assert any("barcoding_pcr_appr" in c.field_names for c in candidates)


def test_barcoding_pcr_appr_keyword_fallback_matches_second_round_pcr():
    """Same real gap, the deterministic safety-net path: this fallback
    exists specifically because a small local model can silently drop a
    two-step-PCR candidate quote, so it must not share the same
    "second-round" blind spot as the LLM-judged search_terms."""
    text = (
        "Index and adapter were added to the purified product during the eight cycles of "
        "second-round PCR using KAPA HiFi HotStart Ready mix."
    )
    result = _barcoding_pcr_appr_keyword_match((("Methods", text),))
    assert result is not None
    assert result[0] == "two-step PCR"


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
    assert candidates[0].field_names == (
        "inhibition_check_0_1",
        "inhibition_check",
        "targeted_detection_method_additional",
        "neg_cont_0_1",
        "pos_cont_0_1",
    )


def test_quote_candidates_split_semicolon_joined_enumerated_steps_into_separate_quotes():
    """Real bug found live (10.1093/ismejo/wrae013): a methods sentence
    bundling two DIFFERENT tools for two DIFFERENT purposes into one
    semicolon-joined enumerated list previously had only ONE terminal
    period, so the whole blob became a single candidate quote tagged for
    every field named in it at once, regardless of which clause actually
    supported which field."""
    text = (
        "Bioinformatic processing was performed as follows: (i) OTUs were clustered using VSEARCH "
        "--cluster_fast [12]; (ii) taxonomy was assigned using BLASTn against GenBank with a 97 "
        "percent identity threshold [13]."
    )
    candidates = quote_candidates_for_llm_judged_search([("Bioinformatics", text)])
    assert len(candidates) == 2
    assert "VSEARCH" in candidates[0].text
    assert "BLASTn" in candidates[1].text
    # otu_clust_tool must only be offered for the clustering clause, not
    # the taxonomic-assignment one.
    assert "otu_clust_tool" in candidates[0].field_names
    assert "otu_clust_tool" not in candidates[1].field_names


def test_quote_candidates_keeps_et_al_citation_intact_in_one_snippet():
    """'et al.' is mistaken for a sentence boundary just like 'vol.' was --
    real bug found live while building primer-reference extraction, where
    a tool/primer name got permanently separated from its own citation."""
    text = (
        "Taxonomy was assigned using the SILVA reference database, as described by Lanzen et al. (2012). "
        "Chimeras were removed using UCHIME."
    )
    candidates = quote_candidates_for_llm_judged_search([("Bioinformatics", text)])
    assert any("Lanzen et al. (2012)" in c.text and "SILVA" in c.text for c in candidates)


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
    assert candidates[0].field_names == ("otu_db",)


def test_quote_candidates_for_llm_judged_otu_db_search_includes_freshtrain():
    candidates = quote_candidates_for_llm_judged_search(
        (
            (
                "Bioinformatics",
                "Taxonomy was assigned against the SILVA_132 and FreshTrain reference databases.",
            ),
        )
    )

    assert len(candidates) == 1
    assert "otu_db" in candidates[0].field_names


def test_quote_candidates_for_llm_judged_otu_db_search_includes_parallel_meta3():
    candidates = quote_candidates_for_llm_judged_search(
        (
            (
                "Bioinformatics",
                "Taxonomy was assigned with Parallel-META3 using its bundled prokaryotic database.",
            ),
        )
    )

    assert len(candidates) == 1
    assert "otu_db" in candidates[0].field_names


def test_quote_candidates_for_llm_judged_otu_db_search_includes_ncbi_nr_database():
    candidates = quote_candidates_for_llm_judged_search(
        (
            (
                "Bioinformatics",
                "All OTUs accounting for >= 1% mapped reads were assigned to their most likely "
                "taxonomic order based on BLAST matches against nonredundant (nr) NCBI database.",
            ),
        )
    )

    assert len(candidates) == 1
    assert "otu_db" in candidates[0].field_names


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


def test_clean_fused_sequence_part_strips_prime_markers_in_either_digit_quote_order():
    """A real paper's own fusion-primer notation used "5´–" (digit,
    prime mark, dash) for the leading boundary but "–3´" (dash, digit,
    prime mark) for the trailing one, in the same sentence -- both orders
    must be stripped so the remaining nucleotide sequence is clean."""
    from fair_ocean_agent.extraction.search_flags import _clean_fused_sequence_part

    assert _clean_fused_sequence_part("5´–TCGTCGGCAGCGTCAGATGTGTAT AAGAGACAG") == (
        "TCGTCGGCAGCGTCAGATGTGTATAAGAGACAG"
    )
    assert _clean_fused_sequence_part("CTCCTACGGGAGGCAGCAG–3´") == "CTCCTACGGGAGGCAGCAG"


def test_split_fused_adapter_primer_facts_splits_real_forward_and_reverse_sequences():
    """Real, live-verified fusion-primer sequences from a PLOS ONE paper
    (10.1371/journal.pone.0303937): Illumina Nextera overhang adapters
    fused via a plain ASCII hyphen to the 341F forward primer and a
    degenerate-base reverse primer, with en-dash 5'/3' boundary markers
    on the outside. The two concepts must be split into separate facts,
    not left duplicated/concatenated in one field."""
    from fair_ocean_agent.extraction.search_flags import RawFactCandidate, _split_fused_adapter_primer_facts
    from fair_ocean_agent.database.enums import EntityLevel, SupportType

    forward = RawFactCandidate(
        fact_type_candidate="adapter_forward",
        raw_field_name="adapter_forward",
        raw_value="5´–TCGTCGGCAGCGTCAGATGTGTAT AAGAGACAG-CTCCTACGGGAGGCAGCAG–3´",
        evidence_quote="quote",
        source_locator="paper:PMC1#p1",
        support_type=SupportType.EXPLICIT,
        entity_level=EntityLevel.STUDY,
    )
    reverse = RawFactCandidate(
        fact_type_candidate="adapter_reverse",
        raw_field_name="adapter_reverse",
        raw_value="5´–GTCTCGTGGGCTC GGAGATGTGTATAAGAGACAG-CCGYCAATTYMTTTRAGTTT–3´",
        evidence_quote="quote",
        source_locator="paper:PMC1#p2",
        support_type=SupportType.EXPLICIT,
        entity_level=EntityLevel.STUDY,
    )

    facts = _split_fused_adapter_primer_facts([forward, reverse])

    by_type: dict[str, list[str]] = {}
    for fact in facts:
        by_type.setdefault(fact.fact_type_candidate, []).append(fact.raw_value)

    assert by_type["adapter_forward"] == ["TCGTCGGCAGCGTCAGATGTGTATAAGAGACAG"]
    assert by_type["pcr_primer_forward"] == ["CTCCTACGGGAGGCAGCAG"]
    assert by_type["adapter_reverse"] == ["GTCTCGTGGGCTCGGAGATGTGTATAAGAGACAG"]
    assert by_type["pcr_primer_reverse"] == ["CCGYCAATTYMTTTRAGTTT"]


def test_split_fused_adapter_primer_facts_leaves_non_fused_values_untouched():
    """A field with no hyphen (nothing to split) or a value that isn't a
    clean adapter-primer fusion (e.g. free text, not two nucleotide runs)
    must be passed through unchanged rather than mangled."""
    from fair_ocean_agent.extraction.search_flags import RawFactCandidate, _split_fused_adapter_primer_facts
    from fair_ocean_agent.database.enums import EntityLevel, SupportType

    plain = RawFactCandidate(
        fact_type_candidate="adapter_forward",
        raw_field_name="adapter_forward",
        raw_value="AATGATACGGCGACCACCGAGATCTACACGCT",
        evidence_quote="quote",
        source_locator="paper:PMC1#p1",
        support_type=SupportType.EXPLICIT,
        entity_level=EntityLevel.STUDY,
    )
    prose = RawFactCandidate(
        fact_type_candidate="adapter_reverse",
        raw_field_name="adapter_reverse",
        raw_value="a custom Illumina-compatible overhang adapter",
        evidence_quote="quote",
        source_locator="paper:PMC1#p2",
        support_type=SupportType.EXPLICIT,
        entity_level=EntityLevel.STUDY,
    )
    other_field = RawFactCandidate(
        fact_type_candidate="assay_name",
        raw_field_name="assay_name",
        raw_value="MiFish-U",
        evidence_quote="quote",
        source_locator="paper:PMC1#p3",
        support_type=SupportType.EXPLICIT,
        entity_level=EntityLevel.STUDY,
    )

    facts = _split_fused_adapter_primer_facts([plain, prose, other_field])

    assert [fact.raw_value for fact in facts] == [
        "AATGATACGGCGACCACCGAGATCTACACGCT",
        "a custom Illumina-compatible overhang adapter",
        "MiFish-U",
    ]
    assert [fact.fact_type_candidate for fact in facts] == ["adapter_forward", "adapter_reverse", "assay_name"]


def test_detect_llm_judged_search_facts_accepts_quote_id_and_stores_literal_quote():
    def respond(prompt: str) -> str:
        assert "Q001 [barcoding_pcr_appr]" in prompt
        assert "Q002 [assay_name]" in prompt
        return json.dumps(
            [
                {"field": "barcoding_pcr_appr", "raw_value": "two-step PCR", "quote_id": "Q001"},
                {"field": "assay_name", "raw_value": "MiFish-U", "quote_id": "Q002"},
            ]
        )

    text = (
        "A two-step PCR was used for library construction. "
        "The MiFish-U assay amplified the 12S marker before sequencing."
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
    assert by_type["assay_name"].raw_value == "MiFish-U"
    assert by_type["assay_name"].evidence_quote == (
        "The MiFish-U assay amplified the 12S marker before sequencing."
    )
    assert by_type["assay_name"].confidence_metadata["matches"][0]["quote_id"] == "Q002"
    assert backend.calls[0]["max_tokens"] == MIN_LLM_MAX_OUTPUT_TOKENS


def test_detect_llm_judged_search_facts_restores_sequencing_methodology():
    text = (
        "The indexed PCR products were purified twice by AMPure XP. "
        "The PCR product was sequenced by the MiSeq platform with MiSeq Reagent Kit v3, "
        "600 cycles (Illumina). "
        "OTUs were clustered using UPARSE."
    )

    def respond(prompt: str) -> str:
        assert "sequencing_methodology" in prompt
        assert "Q001 [" in prompt
        assert "The PCR product was sequenced by the MiSeq platform" in prompt
        return json.dumps(
            [
                {
                    "field": "sequencing_methodology",
                    "raw_value": (
                        "The PCR product was sequenced by the MiSeq platform with MiSeq Reagent Kit v3, "
                        "600 cycles (Illumina)."
                    ),
                    "quote_id": "Q001",
                }
            ]
        )

    backend = MockLLMBackend(label="judge", responses=respond)
    facts = detect_llm_judged_search_facts(
        backend,
        (("Sequencing", text),),
        locator_prefix="paper:PMC1",
    )

    by_type = {fact.fact_type_candidate: fact for fact in facts}
    assert by_type["sequencing_methodology"].raw_value == (
        "The PCR product was sequenced by the MiSeq platform with MiSeq Reagent Kit v3, "
        "600 cycles (Illumina)."
    )
    assert by_type["sequencing_methodology"].confidence_metadata["section"] == "Library preparation sequencing"


def test_detect_llm_judged_search_facts_extracts_targeted_detection_bundle():
    text = (
        "PCR products were checked by agarose gel electrophoresis. "
        "CARD-FISH was performed with probe Atri578 (5'-ACTTTTAAGACCGCCTACGA-3') "
        "at 0.5 uM, designed in this study to target Atribacteria. "
        "A host-blocking primer 5'-ACGTACGTACGT-3' was used to suppress fish DNA amplification. "
        "Samples with Cq < 40 in two of three replicates were considered positive. "
        "The LOD was determined by dilution series and was 3 copies/reaction; "
        "the LOQ was determined from the lowest standard meeting precision criteria and was 30 copies/reaction. "
        "Standard curves used a plasmid containing the target sequence and the fluorescence threshold was set to 0.02."
    )

    def respond(prompt: str) -> str:
        assert "amp_vis_method" in prompt
        assert "probe_seq" in prompt
        assert "block_seq" in prompt
        assert "targeted_detection_method_additional" in prompt
        return json.dumps(
            [
                {"field": "amp_vis_method", "raw_value": "agarose gel electrophoresis", "quote_id": "Q001"},
                {"field": "probe_seq", "raw_value": "ACTTTTAAGACCGCCTACGA", "quote_id": "Q002"},
                {"field": "probe_conc", "raw_value": "0.5 uM", "quote_id": "Q002"},
                {"field": "probe_ref", "raw_value": "designed in this study", "quote_id": "Q002"},
                {"field": "block_seq", "raw_value": "ACGTACGTACGT", "quote_id": "Q003"},
                {"field": "block_taxa", "raw_value": "fish DNA", "quote_id": "Q003"},
                {"field": "detection_criteria", "raw_value": "Cq < 40 in two of three replicates", "quote_id": "Q004"},
                {"field": "lod_method", "raw_value": "dilution series", "quote_id": "Q005"},
                {"field": "pcr_assay_lod", "raw_value": "3", "quote_id": "Q005"},
                {"field": "pcr_assay_lod_unit", "raw_value": "copies/reaction", "quote_id": "Q005"},
                {
                    "field": "loq_method",
                    "raw_value": "lowest standard meeting precision criteria",
                    "quote_id": "Q006",
                },
                {"field": "pcr_assay_loq", "raw_value": "30", "quote_id": "Q006"},
                {"field": "pcr_assay_loq_unit", "raw_value": "copies/reaction", "quote_id": "Q006"},
                {"field": "std_source", "raw_value": "plasmid containing the target sequence", "quote_id": "Q007"},
                {"field": "thresholdQuantificationCycle", "raw_value": "0.02", "quote_id": "Q007"},
                {
                    "field": "targeted_detection_method_additional",
                    "raw_value": "CARD-FISH was performed with probe Atri578 at 0.5 uM.",
                    "quote_id": "Q002",
                },
            ]
        )

    backend = MockLLMBackend(label="judge", responses=respond)
    facts = detect_llm_judged_search_facts(backend, (("Targeted detection", text),), locator_prefix="paper:PMC1")

    by_type = {fact.fact_type_candidate: fact.raw_value for fact in facts}
    assert by_type["amp_vis_method"] == "agarose gel electrophoresis"
    assert by_type["probe_seq"] == "ACTTTTAAGACCGCCTACGA"
    assert by_type["probe_conc"] == "0.5 uM"
    assert by_type["probe_ref"] == "designed in this study"
    assert by_type["block_seq"] == "ACGTACGTACGT"
    assert by_type["block_taxa"] == "fish DNA"
    assert by_type["detection_criteria"] == "Cq < 40 in two of three replicates"
    assert by_type["lod_method"] == "dilution series"
    assert by_type["pcr_assay_lod"] == "3"
    assert by_type["pcr_assay_lod_unit"] == "copies/reaction"
    assert by_type["loq_method"] == "lowest standard meeting precision criteria"
    assert by_type["pcr_assay_loq"] == "30"
    assert by_type["pcr_assay_loq_unit"] == "copies/reaction"
    assert by_type["std_source"] == "plasmid containing the target sequence"
    assert by_type["thresholdQuantificationCycle"] == "0.02"
    assert by_type["targeted_detection_method_additional"] == "CARD-FISH was performed with probe Atri578 at 0.5 uM."


def test_detect_llm_judged_search_facts_handles_frontiers_atri578_probe_and_gel():
    text = (
        "TABLE 2. Oligonucleotide primers and Atribacteria-specific probe used in this study. "
        "Primer | Sequence (5'-3') | Target | Use | Reference "
        "Atri578 | ACTTTTAAGACCGCCTACGA | Atribacteria | C | This study. "
        "After purification of the desired PCR products by agarose gel electrophoresis, index and adapter "
        "were added to the purified product during the eight cycles of second-round PCR using KAPA HiFi "
        "HotStart Ready mix. "
        "The horseradish peroxidase (HRP)-labeled Atri578 probe was purchased from Biomers GmbH "
        "(Ulm, Germany). "
        "Briefly, hybridization was performed in hybridization buffer containing 10% formamide and 0.5 μM "
        "of the HRP-labeled Atri578 probe at 35°C for 2 h."
    )

    def respond(prompt: str) -> str:
        assert "probe_seq" in prompt
        assert "amp_vis_method" in prompt
        assert "targeted_detection_method_additional" in prompt
        assert "Atri578 | ACTTTTAAGACCGCCTACGA" in prompt
        assert "agarose gel electrophoresis" in prompt
        assert "HRP-labeled Atri578 probe" in prompt
        return json.dumps(
            [
                {"field": "probe_seq", "raw_value": "ACTTTTAAGACCGCCTACGA", "quote_id": "Q001"},
                {"field": "probe_ref", "raw_value": "This study", "quote_id": "Q001"},
                {"field": "amp_vis_method", "raw_value": "agarose gel electrophoresis", "quote_id": "Q003"},
                {"field": "probe_conc", "raw_value": "0.5 μM", "quote_id": "Q005"},
                {
                    "field": "targeted_detection_method_additional",
                    "raw_value": (
                        "horseradish peroxidase (HRP)-labeled Atri578 probe; hybridization buffer "
                        "containing 10% formamide and 0.5 μM probe at 35°C for 2 h"
                    ),
                    "quote_id": "Q004",
                },
            ]
        )

    backend = MockLLMBackend(label="judge", responses=respond)
    facts = detect_llm_judged_search_facts(backend, (("Methods", text),), locator_prefix="paper:PMC5476839")

    by_type = {fact.fact_type_candidate: fact.raw_value for fact in facts}
    assert by_type["probe_seq"] == "ACTTTTAAGACCGCCTACGA"
    assert by_type["probe_ref"] == "This study"
    assert by_type["amp_vis_method"] == "agarose gel electrophoresis"
    assert by_type["probe_conc"] == "0.5 μM"
    assert "(HRP)-labeled Atri578 probe" in by_type["targeted_detection_method_additional"]


def test_detect_llm_judged_search_facts_accepts_multiple_bracketed_fields_from_one_quote():
    """Same class of fix as section_category_extraction.py's own
    multi-field prompt clarification: "OTUs were clustered using UPARSE
    and taxonomically assigned using the SILVA database" genuinely
    supports both otu_clust_tool AND otu_db from the same sentence.
    Confirms the code itself never enforced a one-field-per-quote cap
    here either (nothing rejects a second field for the same quote_id)."""
    text = "OTUs were clustered using UPARSE and taxonomically assigned using the SILVA database."
    response = json.dumps(
        [
            {"field": "otu_clust_tool", "raw_value": "UPARSE", "quote_id": "Q001"},
            {"field": "otu_db", "raw_value": "SILVA", "quote_id": "Q001"},
        ]
    )
    backend = MockLLMBackend(responses=[response])
    facts = detect_llm_judged_search_facts(backend, (("Bioinformatics", text),), locator_prefix="paper:PMC1")
    by_type = {fact.fact_type_candidate: fact.raw_value for fact in facts}
    assert by_type["otu_clust_tool"] == "UPARSE"
    assert by_type["otu_db"] == "SILVA"


def test_detect_llm_judged_search_facts_rejects_value_for_field_the_quote_was_never_offered_for():
    """Mirrors the same real bug fixed in section_category_extraction.py's
    Stage 3 guard: the verbatim check alone doesn't stop the model from
    attaching a value to a field a quote was never candidate-tagged for,
    as long as the text also happens to appear in that quote."""

    def respond(prompt: str) -> str:
        assert "Q001 [barcoding_pcr_appr]" in prompt
        return json.dumps(
            [{"field": "lib_screen", "raw_value": "two-step PCR", "quote_id": "Q001"}]
        )

    backend = MockLLMBackend(label="judge", responses=respond)
    facts = detect_llm_judged_search_facts(
        backend,
        (("Library construction", "A two-step PCR was used for library construction."),),
        locator_prefix="paper:PMC1",
    )

    by_type = {fact.fact_type_candidate: fact for fact in facts}
    assert "lib_screen" not in by_type


def test_detect_llm_judged_search_facts_accepts_exact_verbatim_value_for_a_verbatim_required_field():
    backend = MockLLMBackend(
        responses=[json.dumps([{"field": "in_situ_temp", "raw_value": "14.2 C", "quote_id": "Q001"}])]
    )

    facts = detect_llm_judged_search_facts(
        backend,
        (("Methods", "In situ water temperature at the time of sampling was 14.2 C."),),
        locator_prefix="paper:PMC1",
    )

    by_type = {fact.fact_type_candidate: fact for fact in facts}
    assert by_type["in_situ_temp"].raw_value == "14.2 C"


def test_detect_llm_judged_search_facts_does_not_require_verbatim_for_a_composed_field():
    """assay_name is deliberately a COMPOSED/normalized field (e.g. mojibake
    dash cleanup), not verbatim-required -- confirmed by its own existing
    test (test_detect_llm_judged_search_facts_normalizes_assay_name_marker_
    region) needing a non-verbatim value to pass. This just double-checks
    assay_name was never added to _VERBATIM_REQUIRED_FIELDS."""
    from fair_ocean_agent.extraction.search_flags import _VERBATIM_REQUIRED_FIELDS

    assert "assay_name" not in _VERBATIM_REQUIRED_FIELDS


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


def test_detect_llm_judged_search_facts_filters_primer_pairs_from_assay_name():
    facts = detect_llm_judged_search_facts(
        MockLLMBackend(responses=[]),
        (
            (
                "PCR",
                "Amplicon libraries of the 16S rRNA gene were prepared using primers 515F/806r, "
                "and qPCR was performed for the hzsA gene using primer set hzsA_1597A/hzsA_1857R.",
            ),
        ),
        locator_prefix="paper:PMC1",
        exclude_field_names=frozenset({"chimera_check_method"}),
    )

    by_type = {fact.fact_type_candidate: fact for fact in facts}
    assert "assay_name" not in by_type


def test_detect_llm_judged_search_facts_normalizes_assay_name_marker_region():
    backend = MockLLMBackend(
        responses=[
            json.dumps(
                [
                    {"field": "assay_name", "raw_value": "16S rRNA-V3\u7ab6\u5929V4", "quote_id": "Q001"},
                ]
            )
        ]
    )

    facts = detect_llm_judged_search_facts(
        backend,
        (("PCR", "Amplicon sequencing targeted the V3-V4 region of 16S rRNA."),),
        locator_prefix="paper:PMC1",
        exclude_field_names=frozenset({"chimera_check_method"}),
    )

    by_type = {fact.fact_type_candidate: fact for fact in facts}
    assert by_type["assay_name"].raw_value == "16S-V3-V4"


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


def test_detect_llm_judged_search_facts_rejects_bare_marker_gene_as_target_taxon():
    """A real audit (STUDY-0481bc457aa6, 10.1371/journal.pone.0303937) found
    the model answering assay_target_taxa with "16S rRNA gene" for the
    quote "Amplification of the V3-V5 hypervariable regions ... of the 16S
    rRNA gene was performed with ..." -- the quote only names the
    amplified gene, never an organism/taxonomic group, so the gene name is
    not a valid taxon and must be rejected rather than accepted as-is. A
    candidate sentence was still offered to the model here, so this is a rejected-answer
    case (silently no fact), not the separate "no qualifying candidate at all" -> "not found"
    fallback path."""
    backend = MockLLMBackend(
        responses=[
            json.dumps(
                [{"field": "assay_target_taxa", "raw_value": "16S rRNA gene", "quote_id": "Q001"}]
            )
        ]
    )

    facts = detect_llm_judged_search_facts(
        backend,
        (
            (
                "PCR",
                "Amplification of the V3-V5 hypervariable regions of the 16S rRNA gene was performed "
                "with 0.02 U/μl of Phusion High Fidelity DNA polymerase.",
            ),
        ),
        locator_prefix="paper:PMC1",
    )

    by_type = {fact.fact_type_candidate: fact for fact in facts}
    assert "assay_target_taxa" not in by_type


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


def test_detect_llm_judged_search_facts_does_not_default_barcoding_when_resolved():
    facts = detect_llm_judged_search_facts(
        MockLLMBackend(responses=[]),
        (("PCR", "PCR amplification was performed with primers 515F and 806R."),),
        locator_prefix="paper:PMC1",
        active_flags=frozenset({"pcr_0_1"}),
        exclude_field_names=frozenset({"barcoding_pcr_appr"}),
    )

    assert "barcoding_pcr_appr" not in {fact.fact_type_candidate for fact in facts}


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


def test_detect_llm_judged_otu_clustering_fields_keep_best_values():
    def respond(prompt: str) -> str:
        assert "otu_clust_tool" in prompt
        return json.dumps(
            [
                {"field": "otu_clust_tool", "raw_value": "VSEARCH --cluster_fast", "quote_id": "Q001"},
                {"field": "otu_clust_tool", "raw_value": "mothur cluster", "quote_id": "Q001"},
            ]
        )

    facts = detect_llm_judged_search_facts(
        MockLLMBackend(label="judge", responses=respond),
        (("Bioinformatics", "OTUs were clustered at 97% similarity using VSEARCH --cluster_fast."),),
        locator_prefix="paper:PMC1",
    )

    by_type = {fact.fact_type_candidate: fact for fact in facts}
    assert by_type["otu_clust_tool"].raw_value == "VSEARCH --cluster_fast"


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


def test_detect_llm_judged_otu_db_keeps_multiple_databases():
    backend = MockLLMBackend(
        responses=[
            json.dumps(
                [
                    {"field": "otu_db", "raw_value": "SILVA_132", "quote_id": "Q001"},
                    {"field": "otu_db", "raw_value": "FreshTrain", "quote_id": "Q001"},
                ]
            )
        ]
    )

    facts = detect_llm_judged_search_facts(
        backend,
        (
            (
                "Bioinformatics",
                "Taxonomy was assigned against the SILVA_132 and FreshTrain reference databases.",
            ),
        ),
        locator_prefix="paper:PMC1",
    )

    by_type = {fact.fact_type_candidate: fact for fact in facts}
    assert by_type["otu_db"].raw_value == "SILVA_132 | FreshTrain"


def test_detect_llm_judged_otu_db_keeps_parallel_meta3_with_named_database():
    backend = MockLLMBackend(
        responses=[
            json.dumps(
                [
                    {"field": "otu_db", "raw_value": "Parallel-META3", "quote_id": "Q001"},
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
                "Taxonomic classification was performed with Parallel-META3 and SILVA release 138.",
            ),
        ),
        locator_prefix="paper:PMC1",
    )

    by_type = {fact.fact_type_candidate: fact for fact in facts}
    assert by_type["otu_db"].raw_value == "Parallel-META3 | SILVA release 138"


def test_detect_llm_judged_otu_db_keeps_ncbi_nr_database_value():
    backend = MockLLMBackend(
        responses=[
            json.dumps(
                [
                    {"field": "otu_db", "raw_value": "nonredundant (nr) NCBI database", "quote_id": "Q001"},
                ]
            )
        ]
    )

    facts = detect_llm_judged_search_facts(
        backend,
        (
            (
                "Bioinformatics",
                "All OTUs accounting for >= 1% mapped reads were assigned to their most likely "
                "taxonomic order based on BLAST matches against nonredundant (nr) NCBI database.",
            ),
        ),
        locator_prefix="paper:PMC1",
    )

    by_type = {fact.fact_type_candidate: fact for fact in facts}
    assert by_type["otu_db"].raw_value == "nonredundant (nr) NCBI database"


def test_detect_llm_judged_otu_seq_comp_appr_lists_tools_before_quotes():
    """otu_seq_comp_appr remains reviewable, but the useful machine-readable
    tool names must come first instead of burying BLAST/VSEARCH inside a
    long methods quote."""
    quote = (
        "ASVs were assigned taxonomy using BLASTn against GenBank and "
        "classify-consensus vsearch in QIIME 2."
    )
    backend = MockLLMBackend(
        responses=[
            json.dumps(
                [
                    {"field": "otu_seq_comp_appr", "raw_value": quote, "quote_id": "Q001"},
                ]
            )
        ]
    )

    facts = detect_llm_judged_search_facts(
        backend,
        (("Bioinformatics", quote),),
        locator_prefix="paper:PMC1",
    )

    by_type = {fact.fact_type_candidate: fact for fact in facts}
    assert by_type["otu_seq_comp_appr"].raw_value == (
        "classify-consensus vsearch | BLASTn | ASVs were assigned taxonomy using BLASTn "
        "against GenBank and classify-consensus vsearch in QIIME 2."
    )




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


def test_detect_llm_judged_search_facts_extracts_in_situ_temp_salinity_verbatim():
    """Real audit (10.1093/ismejo/wrae013, STUDY-295abf4a8f43): "In situ
    bottom water temperature (6.5C) and salinity (6.4 PSU) were measured
    with a ProODO probe..." -- a text-based mechanism for these fields
    (previously structured-source-only)."""
    backend = MockLLMBackend(
        responses=[
            json.dumps(
                [
                    {"field": "in_situ_temp", "raw_value": "6.5C", "quote_id": "Q001"},
                    {"field": "in_situ_salinity", "raw_value": "6.4 PSU", "quote_id": "Q001"},
                ]
            )
        ]
    )

    facts = detect_llm_judged_search_facts(
        backend,
        (
            (
                "Methods",
                "In situ bottom water temperature (6.5C), dissolved O2 (11.9 mg L-1), and salinity "
                "(6.4 PSU) were measured with a ProODO probe (YSI, USA) and CTD CastAway CTD (SonTek, "
                "USA).",
            ),
        ),
        locator_prefix="paper:PMC1",
    )

    by_type = {fact.fact_type_candidate: fact for fact in facts}
    assert by_type["in_situ_temp"].raw_value == "6.5C"
    assert by_type["in_situ_salinity"].raw_value == "6.4 PSU"


def test_quote_candidates_for_in_situ_measurements_exclude_later_incubation_readings():
    """Same real paper separately reports post-collection INCUBATION-
    chamber daily-average temperature/oxygen/salinity -- a genuinely
    different concept (not "at the time of sampling", per temp's own
    FAIRe definition) that must never become a candidate for these
    fields, even though it uses the exact same measurement words."""
    candidates = quote_candidates_for_llm_judged_search(
        (
            (
                "Methods",
                "Throughout the acclimation phase oxygen, temperature, and salinity were measured "
                "daily inside the two incubation chambers and had an average temperature of 5.9C.",
            ),
        )
    )
    assert not any(
        field in candidate.field_names
        for candidate in candidates
        for field in ("in_situ_temp", "in_situ_salinity")
    )


def test_quote_candidates_for_in_situ_temp_are_targeted_to_collection_time_context():
    candidates = quote_candidates_for_llm_judged_search(
        (
            (
                "Methods",
                "In situ bottom water temperature (6.5C) was measured at the time of collection.",
            ),
        )
    )
    assert any("in_situ_temp" in c.field_names for c in candidates)


def test_quote_candidates_for_in_situ_temp_reach_llm_when_collection_sentence_is_the_next_one_over():
    """Real gap found live (10.3390/microorganisms10030558, STUDY-
    0049c7972ece): the paper states site temperature as its own sentence
    immediately before the collection sentence, with neither "in situ"
    nor "at the time of collection" anywhere -- "Seawater temperature was
    28.1 C. Seawater samples were collected at the surface layer...".
    _SAMPLING_TIME_CONTEXT_RE alone can't see this (no qualifying phrase
    exists at all), so this also needs the plain "samples were
    collected" phrase treated as sufficient context, checked across the
    +/-1-sentence window rather than the bare temperature sentence alone."""
    candidates = quote_candidates_for_llm_judged_search(
        (
            (
                "Methods",
                "Water depth of each sample site was measured using a depth sounder. "
                "Seawater temperature was 28.1 ± 0.2 °C. Seawater samples were collected "
                "at the surface layer (0.5 m depth) and the bottom layer.",
            ),
        )
    )
    matching = [c for c in candidates if "in_situ_temp" in c.field_names]
    assert len(matching) == 1
    assert "28.1" in matching[0].text
    assert "collected" in matching[0].text


def test_quote_candidates_for_screen_contam_method_excludes_checkm_bin_qc_mention():
    """Real bad extraction found live: a paper's CheckM/CANU description
    ("CheckM (version 1.07) was used to assess genome completeness and
    contamination... CANU (version 1.8) was used for assembly...") got
    merged into a bogus screen_contam_method value even though this is
    metagenome-assembled-genome bin QC, not sequence/OTU/ASV contaminant
    screening -- a bare "contamination" mention was previously sufficient
    on its own to become a candidate."""
    candidates = quote_candidates_for_llm_judged_search(
        (
            (
                "Methods",
                "CheckM (version 1.07) was used to assess genome completeness and contamination using "
                "the default set of bacterial marker genes, and CANU (version 1.8) was used for "
                "assembly with default parameters.",
            ),
        )
    )
    assert not any("screen_contam_method" in c.field_names for c in candidates)


def test_quote_candidates_for_screen_contam_method_excludes_field_equipment_sterilization():
    """Real bad extraction found live: "The water sampler was sanitised
    and rinsed in the water body between samples to avoid contamination
    from previous sites" describes field EQUIPMENT sterilization
    (sterilise_method's own concept), not sequence-level decontamination,
    but got captured as screen_contam_method instead."""
    candidates = quote_candidates_for_llm_judged_search(
        (
            (
                "Methods",
                "The water sampler was sanitised and rinsed in the water body between samples to "
                "avoid contamination from previous sites.",
            ),
        )
    )
    assert not any("screen_contam_method" in c.field_names for c in candidates)


def test_quote_candidates_for_screen_contam_method_still_matches_real_decontam_language():
    candidates = quote_candidates_for_llm_judged_search(
        (
            (
                "Bioinformatics",
                "ASVs present in the extraction blanks and PCR negative controls were flagged as "
                "contaminants and removed using the decontam R package with default settings.",
            ),
        )
    )
    assert any("screen_contam_method" in c.field_names for c in candidates)


def test_quote_candidates_for_screen_contam_method_matches_plural_contaminants_and_negative_controls():
    """Real gap found live (STUDY-012e2a73836d): "...potential
    contaminants..." and "...in the negative controls..." never matched
    the cues "contaminant"/"negative control" at all -- _term_pattern's
    strict right-boundary check correctly rejects a different word
    sharing a prefix, but was equally strict about a simple plural of
    the exact same word. This sentence deliberately uses ONLY plural
    forms (no bare "decontam"/"blank" etc. to accidentally pass via a
    different cue) so it isolates the pluralization gap specifically."""
    text = (
        "To exclude confounding results due to potential contaminants from the extraction process "
        "and chemicals used for PCR and sequencing, the top 10 families in the negative controls, "
        "and previously reported core-human microbiomes were arbitrarily removed."
    )
    candidates = quote_candidates_for_llm_judged_search((("Methods", text),))
    assert any("screen_contam_method" in c.field_names for c in candidates)
