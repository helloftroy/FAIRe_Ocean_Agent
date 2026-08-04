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
    assert all(fact.raw_value == "true" for fact in facts)
    assert by_type["probe_based_qPCR_ddPCR_assay_0_1"].evidence_quote.startswith("The assay used a TaqMan")
    assert by_type["probe_based_qPCR_ddPCR_assay_0_1"].confidence_metadata["matched_terms"] == [
        "FAM",
        "hydrolysis probe",
        "reporter dye",
        "TaqMan",
    ]
    assert by_type["pcr_0_1"].evidence_quote == "PCR amplification was performed in triplicate."


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
    assert by_type["seq_kit"].raw_value == "MiSeq Reagent Kit v3 | chemistry"
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
