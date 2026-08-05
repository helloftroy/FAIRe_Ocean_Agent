import json

from fair_ocean_agent.extraction.search_flags import (
    detect_controlled_search_facts,
    detect_llm_judged_search_facts,
    detect_text_search_flags,
    quote_candidates_for_llm_judged_search,
)
from fair_ocean_agent.llm.mock import MockLLMBackend


def test_detect_text_search_flags_records_pcr_and_probe_flags_once():
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
    assert set(by_type) == {"probe_based_qPCR_ddPCR_assay_0_1", "pcr_0_1"}
    assert by_type["probe_based_qPCR_ddPCR_assay_0_1"].raw_value == "true"
    assert by_type["pcr_0_1"].raw_value == "1"
    assert by_type["probe_based_qPCR_ddPCR_assay_0_1"].evidence_quote.startswith("The assay used a TaqMan")
    assert by_type["probe_based_qPCR_ddPCR_assay_0_1"].confidence_metadata["matched_terms"] == [
        "FAM",
        "hydrolysis probe",
        "reporter dye",
        "TaqMan",
    ]
    assert by_type["pcr_0_1"].evidence_quote == "PCR amplification was performed in triplicate."


def test_detect_text_search_flags_matches_amplified_verb_forms_not_just_amplification():
    """Regression guard for a real gold case (data/benchmark/gold/example-001.json)
    that describes explicit PCR content ("...was amplified using primers...
    in a 25 uL reaction volume with an annealing temperature of 57C for 35
    cycles...") but never uses the word "PCR" or the noun "amplification" --
    only the verb "amplified". pcr_0_1 must still activate, or every
    downstream flag-gated PCR checklist field silently becomes unreachable
    for a real paper phrased this way."""
    facts = detect_text_search_flags(
        (("Methods", "The target region was amplified using primers X and Y."),),
        locator_prefix="paper:PMC1",
    )
    assert {fact.fact_type_candidate for fact in facts} == {"pcr_0_1"}
    assert facts[0].raw_value == "1"


def test_detect_text_search_flags_does_not_invent_absent_flags():
    facts = detect_text_search_flags(
        (("Methods", "Water samples were collected at 5 m depth."),),
        locator_prefix="paper:PMC1",
    )

    assert facts == []


def test_detect_text_search_flags_records_control_booleans_from_explicit_evidence():
    facts = detect_text_search_flags(
        (
            (
                "Controls",
                "Negative controls included field blanks and NTCs. "
                "A positive control used synthetic DNA from a mock community.",
            ),
        ),
        locator_prefix="paper:PMC1",
    )

    by_type = {fact.fact_type_candidate: fact for fact in facts}
    assert by_type["neg_cont_0_1"].raw_value == "1"
    assert by_type["neg_cont_0_1"].evidence_quote == "Negative controls included field blanks and NTCs."
    assert by_type["neg_cont_0_1"].confidence_metadata["matched_terms"] == [
        "field blanks",
        "Negative controls",
        "NTCs",
    ]
    assert by_type["pos_cont_0_1"].raw_value == "1"
    assert by_type["pos_cont_0_1"].evidence_quote == (
        "A positive control used synthetic DNA from a mock community."
    )


def test_detect_text_search_flags_sets_control_zero_only_from_explicit_none():
    facts = detect_text_search_flags(
        (
            (
                "Controls",
                "No negative controls were used. Positive controls were not included.",
            ),
        ),
        locator_prefix="paper:PMC1",
    )

    by_type = {fact.fact_type_candidate: fact for fact in facts}
    assert by_type["neg_cont_0_1"].raw_value == "0"
    assert by_type["neg_cont_0_1"].evidence_quote == "No negative controls were used."
    assert by_type["pos_cont_0_1"].raw_value == "0"
    assert by_type["pos_cont_0_1"].evidence_quote == "Positive controls were not included."


def test_detect_text_search_flags_treats_control_treatments_as_negative_controls():
    facts = detect_text_search_flags(
        (("Methods", "Four FSW control treatments were also included."),),
        locator_prefix="paper:PMC1",
    )

    by_type = {fact.fact_type_candidate: fact for fact in facts}
    assert by_type["neg_cont_0_1"].raw_value == "1"
    assert by_type["neg_cont_0_1"].evidence_quote == "Four FSW control treatments were also included."


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
        active_flags=frozenset(fact.fact_type_candidate for fact in flag_facts),
    )

    by_type = {fact.fact_type_candidate: fact for fact in controlled}
    assert by_type["target_gene"].raw_value == "12S rRNA | COI"
    assert by_type["assay_name"].raw_value == "MiFish-U"
    assert by_type["probeReporter"].raw_value == "FAM | reporter"
    assert by_type["probeQuencher"].raw_value == "BHQ-1 | quencher"
    assert by_type["seq_kit"].raw_value == "MiSeq Reagent Kit v3"
    assert by_type["sample_type"].raw_value == "water | sediment"
    assert by_type["target_gene"].entity_level.value == "study"
    assert by_type["target_gene"].confidence_metadata["activated_by_flags"] == ["pcr_0_1"]


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
        active_flags=frozenset(fact.fact_type_candidate for fact in flag_facts),
    )

    by_type = {fact.fact_type_candidate: fact for fact in [*flag_facts, *controlled]}
    assert by_type["neg_cont_0_1"].raw_value == "1"
    assert by_type["thermocycler"].raw_value == "DNA Engine Tetrad2 Thermal Cycler"
    assert by_type["seq_kit"].raw_value == "Titanium chemistry"
    assert by_type["sequencing_location"].raw_value == (
        "Genome Sequencing and Analysis Facility (GSAF) at the University of Texas at Austin"
    )


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
        active_flags=frozenset(fact.fact_type_candidate for fact in flag_facts),
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


def test_detect_controlled_search_facts_does_not_match_bare_its_as_target_gene():
    text = "PCR amplified 18S rRNA; its sequence reads were clustered after filtering."
    flag_facts = detect_text_search_flags((("Methods", text),), locator_prefix="paper:PMC1")
    controlled = detect_controlled_search_facts(
        (("Methods", text),),
        locator_prefix="paper:PMC1",
        active_flags=frozenset(fact.fact_type_candidate for fact in flag_facts),
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
        active_flags=frozenset(fact.fact_type_candidate for fact in flag_facts),
    )

    by_type = {fact.fact_type_candidate: fact for fact in controlled}
    assert by_type["target_gene"].raw_value == "16S rRNA SSU | 18S rRNA SSU"


def test_detect_controlled_search_facts_expands_coordinated_ssu_target_gene_names():
    text = "Taxonomic classification extracted small subunit rRNA (16S and 18S SSU rRNA) reads."
    flag_facts = detect_text_search_flags((("Methods", f"PCR was performed. {text}"),), locator_prefix="paper:PMC1")
    controlled = detect_controlled_search_facts(
        (("Methods", text),),
        locator_prefix="paper:PMC1",
        active_flags=frozenset(fact.fact_type_candidate for fact in flag_facts),
    )

    by_type = {fact.fact_type_candidate: fact for fact in controlled}
    assert by_type["target_gene"].raw_value == "16S rRNA SSU | 18S rRNA SSU"


def test_detect_controlled_search_facts_keeps_rrna_when_no_ssu_target_gene_name():
    text = "The PCR assay amplified 16S rRNA and later refers to the 16S amplicons."
    flag_facts = detect_text_search_flags((("Methods", text),), locator_prefix="paper:PMC1")
    controlled = detect_controlled_search_facts(
        (("Methods", text),),
        locator_prefix="paper:PMC1",
        active_flags=frozenset(fact.fact_type_candidate for fact in flag_facts),
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


def test_detect_controlled_search_facts_extracts_trimmomatic_reads_below_phrase():
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
    assert by_type["adapter_trimming_method"].raw_value == "trimmomatic"
    assert by_type["length_filtering_tool"].raw_value == "trimmomatic"
    assert by_type["minimum_read_length"].raw_value == "500 bp"


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
    )

    assert facts == []
