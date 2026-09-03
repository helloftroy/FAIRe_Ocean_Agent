import json

import pytest

from fair_ocean_agent.database.enums import MappingMethod, MissingnessStatus, SupportType
from fair_ocean_agent.database.models import StandardizedValue, Study
from fair_ocean_agent.extraction.faire_fields import field_names_for_reference
from fair_ocean_agent.extraction.text import (
    EXTRACTION_INSTRUCTIONS,
    EXTRACTION_FOCUSES,
    PROMPT_VERSION,
    build_prompt,
    extract_facts_from_section,
    is_absent_raw_value,
    present_faire_fields_for_study,
    resolved_faire_fields_for_study,
    recall_missing_fact_types,
    segment_source_text,
    segments_for_focus,
    split_section_text,
)
from fair_ocean_agent.llm.mock import MockLLMBackend
from fair_ocean_agent.mapping.faire import TARGET_SCHEMA, TARGET_SCHEMA_VERSION

SECTION_TEXT = "Samples were collected on 4 January 2022 at a depth of 5 meters near the reef."


def test_segment_id_fact_is_kept_with_python_copied_evidence_quote():
    response = json.dumps(
        [{"fact_type_candidate": "collection_date", "raw_value": "2022-01-04", "evidence_id": "METHODS.P001"}]
    )
    backend = MockLLMBackend(responses=[response])
    facts, _ = extract_facts_from_section(backend, "Methods", SECTION_TEXT)

    assert len(facts) == 1
    assert facts[0].fact_type_candidate == "collection_date"
    assert facts[0].evidence_quote == SECTION_TEXT
    assert facts[0].source_locator.endswith("METHODS.P001")
    assert facts[0].confidence_metadata == {"evidence_ids": ["METHODS.P001"]}
    assert facts[0].support_type == SupportType.EXPLICIT


def test_missing_or_unknown_evidence_id_is_dropped():
    response = json.dumps(
        [{"fact_type_candidate": "fake", "raw_value": "x", "evidence_id": "METHODS.P999"}]
    )
    backend = MockLLMBackend(responses=[response])
    facts, _ = extract_facts_from_section(backend, "Methods", SECTION_TEXT)
    assert facts == []


def test_hallucinated_pcr_volume_not_present_in_quote_is_dropped():
    text = (
        "Amplicon libraries were sequenced on an Ion Torrent Personal Genome Machine. "
        "The raw sequencing reads were quality filtered and trimmed to 220 bp using USEARCH."
    )
    response = json.dumps(
        [{"fact_type_candidate": "pcr_reaction_volume", "raw_value": "25 uL", "evidence_id": "METHODS.P001"}]
    )
    backend = MockLLMBackend(responses=[response])

    facts, _ = extract_facts_from_section(backend, "Methods", text, active_flags=frozenset({"pcr_0_1"}))

    assert facts == []


def test_primer_name_substituted_for_sequence_is_dropped():
    """Regression guard for a real bug found live (10.1002/ece3.6071):
    when a paper only states a primer's NAME in the main text (its real
    sequence lives in a supplementary table this pass never sees), the
    model substituted the name for the sequence field instead of omitting
    it -- both "1389F" and "mlCOIintF" are literally present in real
    quotes, so a plain verbatim check alone wouldn't catch this; the value
    itself must actually look like a nucleotide sequence."""
    text = "The 18S rRNA gene was amplified using the universal primer 1389F."
    response = json.dumps(
        [{"fact_type_candidate": "forward_primer_sequence", "raw_value": "1389F", "evidence_id": "METHODS.P001"}]
    )
    backend = MockLLMBackend(responses=[response])

    facts, _ = extract_facts_from_section(backend, "Methods", text, active_flags=frozenset({"pcr_0_1"}))

    assert facts == []


def test_real_primer_sequence_survives_the_nucleotide_shape_check():
    text = "The forward primer sequence used was GGWACWGGWTGAACWGTWTAYCCYCC."
    response = json.dumps(
        [
            {
                "fact_type_candidate": "forward_primer_sequence",
                "raw_value": "GGWACWGGWTGAACWGTWTAYCCYCC",
                "evidence_id": "METHODS.P001",
            }
        ]
    )
    backend = MockLLMBackend(responses=[response])

    facts, _ = extract_facts_from_section(backend, "Methods", text, active_flags=frozenset({"pcr_0_1"}))

    assert len(facts) == 1
    assert facts[0].raw_value == "GGWACWGGWTGAACWGTWTAYCCYCC"


def test_primer_sequence_with_leading_prime_marker_decoration_is_cleaned_not_dropped():
    """Real gap found live (10.1111/1462-2920.14870): "The Fluidgim V4
    primer set 515F-Y: 5'-GTGYCAGCMGCCGCGGTAA ... and 806RB:
    5'-GGACTACNVGGGTWTCTAAT" -- the model copied the sequence verbatim
    from the quote per its own instructions, decorative 5'/3' boundary
    markers included, which used to fail the strict nucleotide-only shape
    check outright and get silently dropped entirely (the primer NAME
    survived since it has no such check, only the sequence vanished).
    Cleaned instead of discarded, matching search_flags.py's own
    fused-adapter-primer cleaning for the identical decoration."""
    text = (
        "The Fluidgim V4 primer set 515F-Y: 5′‐GTGYCAGCMGCCGCGGTAA (Parada et al., 2016) and "
        "806RB: 5′‐GGACTACNVGGGTWTCTAAT (Apprill et al., 2015), accompanied with Illumina "
        "adapters, index, pad, and linker sequences, were used for amplification."
    )
    response = json.dumps(
        [
            {
                "fact_type_candidate": "forward_primer_sequence",
                "raw_value": "5′‐GTGYCAGCMGCCGCGGTAA",
                "evidence_id": "METHODS.P001",
            },
            {
                "fact_type_candidate": "reverse_primer_sequence",
                "raw_value": "5′‐GGACTACNVGGGTWTCTAAT",
                "evidence_id": "METHODS.P001",
            },
        ]
    )
    backend = MockLLMBackend(responses=[response])

    facts, _ = extract_facts_from_section(backend, "Methods", text, active_flags=frozenset({"pcr_0_1"}))

    by_type = {fact.fact_type_candidate: fact.raw_value for fact in facts}
    assert by_type["forward_primer_sequence"] == "GTGYCAGCMGCCGCGGTAA"
    assert by_type["reverse_primer_sequence"] == "GGACTACNVGGGTWTCTAAT"


def test_filter_name_can_be_extracted_from_sampling_text():
    text = "Samples were filtered directly using a 0.22 μm cartridge filter (Sterivex filter)."
    response = json.dumps(
        [{"fact_type_candidate": "filter_name", "raw_value": "Sterivex filter", "evidence_id": "METHODS.P001"}]
    )
    backend = MockLLMBackend(responses=[response])

    facts, _ = extract_facts_from_section(backend, "Methods", text)

    assert len(facts) == 1
    assert facts[0].fact_type_candidate == "filter_name"
    assert facts[0].raw_value == "Sterivex filter"


def test_assay_tag_on_assay_scoped_fact_becomes_assay_entity():
    from fair_ocean_agent.database.enums import EntityLevel

    text = (
        "For 16S, primers X/Y were used with an annealing temperature of "
        "55C. For 18S, primers Z/W were used with an annealing temperature "
        "of 60C."
    )
    response = json.dumps(
        [
            {
                "fact_type_candidate": "annealing_temperature",
                "raw_value": "55C",
                "evidence_id": "METHODS.P001",
                "assay_tag": "16S-V3V4",
            },
            {
                "fact_type_candidate": "annealing_temperature",
                "raw_value": "60C",
                "evidence_id": "METHODS.P001",
                "assay_tag": "18S-V9",
            },
        ]
    )
    backend = MockLLMBackend(responses=[response])
    facts, _ = extract_facts_from_section(backend, "Methods", text, active_flags=frozenset({"pcr_0_1"}))

    assert len(facts) == 2
    by_value = {fact.raw_value: fact for fact in facts}
    assert by_value["55C"].entity_level == EntityLevel.ASSAY
    assert by_value["55C"].entity_external_id == "16S-V3V4"
    assert by_value["60C"].entity_level == EntityLevel.ASSAY
    assert by_value["60C"].entity_external_id == "18S-V9"


def test_assay_tag_ignored_for_non_assay_scoped_fact_type():
    from fair_ocean_agent.database.enums import EntityLevel

    response = json.dumps(
        [
            {
                "fact_type_candidate": "dna_extraction_kit",
                "raw_value": "DNeasy PowerWater Kit",
                "evidence_id": "METHODS.P001",
                "assay_tag": "16S-V3V4",
            }
        ]
    )
    backend = MockLLMBackend(responses=[response])
    facts, _ = extract_facts_from_section(backend, "Methods", SECTION_TEXT)

    assert len(facts) == 1
    assert facts[0].entity_level == EntityLevel.STUDY
    assert facts[0].entity_external_id is None


def test_placeholder_assay_tag_is_dropped():
    from fair_ocean_agent.database.enums import EntityLevel

    response = json.dumps(
        [
            {
                "fact_type_candidate": "annealing_temperature",
                "raw_value": "55C",
                "evidence_id": "METHODS.P001",
                "assay_tag": "N/A",
            }
        ]
    )
    backend = MockLLMBackend(responses=[response])
    facts, _ = extract_facts_from_section(backend, "Methods", SECTION_TEXT, active_flags=frozenset({"pcr_0_1"}))

    assert len(facts) == 1
    assert facts[0].entity_level == EntityLevel.STUDY
    assert facts[0].entity_external_id is None


def test_identical_value_and_quote_with_different_assay_tags_both_survive_dedup():
    response = json.dumps(
        [
            {
                "fact_type_candidate": "pcr_cycle_count",
                "raw_value": "35",
                "evidence_id": "METHODS.P001",
                "assay_tag": "16S-V3V4",
            },
            {
                "fact_type_candidate": "pcr_cycle_count",
                "raw_value": "35",
                "evidence_id": "METHODS.P001",
                "assay_tag": "18S-V9",
            },
        ]
    )
    backend = MockLLMBackend(responses=[response])
    facts, _ = extract_facts_from_section(backend, "Methods", SECTION_TEXT, active_flags=frozenset({"pcr_0_1"}))

    assert len(facts) == 2
    assert {fact.entity_external_id for fact in facts} == {"16S-V3V4", "18S-V9"}


def test_missing_fields_are_skipped_not_crashed_on():
    response = json.dumps([{"evidence_id": "METHODS.P001"}])  # no fact_type/raw_value
    backend = MockLLMBackend(responses=[response])
    facts, _ = extract_facts_from_section(backend, "Methods", SECTION_TEXT)
    assert facts == []


def test_absent_placeholder_values_are_dropped():
    response = json.dumps(
        [
            {"fact_type_candidate": "collection_method", "raw_value": "not specified", "evidence_id": "METHODS.P001"},
            {"fact_type_candidate": "storage_conditions", "raw_value": " not explicitly stated in the text ", "evidence_id": "METHODS.P001"},
            {"fact_type_candidate": "sampling_depth", "raw_value": "", "evidence_id": "METHODS.P001"},
            {"fact_type_candidate": "collection_date", "raw_value": "2022-01-04", "evidence_id": "METHODS.P001"},
        ]
    )
    backend = MockLLMBackend(responses=[response])
    facts, _ = extract_facts_from_section(backend, "Methods", SECTION_TEXT)

    assert [(fact.fact_type_candidate, fact.raw_value) for fact in facts] == [("collection_date", "2022-01-04")]


def test_absent_raw_value_predicate_covers_common_model_placeholders():
    for value in (
        None,
        "",
        "   ",
        "none",
        "N/A",
        "not specified",
        "not explicitly stated",
        "not reported in the text",
        "not resolved",
        "not resolved.",
        "unresolved",
        "see below",
        "see above.",
    ):
        assert is_absent_raw_value(value)
    assert not is_absent_raw_value("none detected in the negative control")


def test_invalid_json_response_yields_no_facts():
    backend = MockLLMBackend(responses=["not json"] * 5)
    facts, response = extract_facts_from_section(backend, "Methods", SECTION_TEXT)
    assert facts == []
    assert response is not None


def test_valid_json_scalar_response_yields_no_facts_not_crash():
    backend = MockLLMBackend(responses=["42"])
    facts, response = extract_facts_from_section(backend, "Methods", SECTION_TEXT)
    assert facts == []
    assert response is not None


def test_segment_source_text_assigns_stable_ids():
    segments = segment_source_text("PCR", "PCR reactions used primers.\n\nReactions were annealed at 54 C.")
    assert [segment.segment_id for segment in segments] == ["PCR.P001", "PCR.P002"]
    assert segments[1].text == "Reactions were annealed at 54 C."


def test_split_section_text_bounds_long_sections():
    text = "First sentence about sampling. " * 20
    chunks = split_section_text(text, max_chars=120)

    assert len(chunks) > 1
    assert all(len(chunk) <= 120 for chunk in chunks)
    assert "First sentence about sampling" in chunks[0]


def test_extract_facts_from_section_chunks_long_text_and_merges_facts():
    first = "Water samples were collected on 4 January 2022."
    second = "PCR used primers 515F and 806R."
    section_text = f"{first}\n\n{second}"
    responses = [
        json.dumps([{"fact_type_candidate": "collection_date", "raw_value": "2022-01-04", "evidence_id": "METHODS.P001"}]),
        json.dumps([{"fact_type_candidate": "forward_primer_name", "raw_value": "515F", "evidence_id": "METHODS.P002"}]),
    ]
    backend = MockLLMBackend(responses=responses)

    facts, _ = extract_facts_from_section(
        backend,
        "Methods",
        section_text,
        max_section_chars_per_call=len(first) + 1,
        max_output_tokens=2048,
        active_flags=frozenset({"pcr_0_1"}),
    )

    assert [fact.fact_type_candidate for fact in facts] == ["collection_date", "forward_primer_name"]
    # One call per chunk (no per-topic-focus fan-out, and no recall retry
    # since each chunk's single pass already found a fact) -- 2 chunks, 2 calls.
    assert len(backend.calls) == 2
    assert all(call["max_tokens"] == 2048 for call in backend.calls)
    assert "METHODS.P001:" in backend.calls[0]["prompt"]
    assert "METHODS.P002:" in backend.calls[1]["prompt"]
    assert facts[0].evidence_quote == first
    assert facts[1].evidence_quote == second


def test_prompt_version_is_stable_constant():
    assert PROMPT_VERSION == "text-extraction-v21-expedition-id"


def test_recall_second_pass_does_not_fire_when_first_pass_finds_any_facts():
    """A partial main-pass result (found forward_primer_name, missed
    reverse_primer_name) is accepted as-is -- no automatic retry just
    because the checklist wasn't fully satisfied. Recall now only exists as
    a safety net for a pass that found literally nothing (see the sibling
    test below), not a completeness guarantee for every concept."""
    section_text = "PCR reactions used MiFish-U-F and MiFish-U-R primers at 54 C."
    response = json.dumps(
        [{"fact_type_candidate": "forward_primer_name", "raw_value": "MiFish-U-F", "evidence_id": "PCR.P001"}]
    )
    backend = MockLLMBackend(responses=[response])

    facts, _ = extract_facts_from_section(backend, "PCR", section_text, active_flags=frozenset({"pcr_0_1"}))

    assert [fact.fact_type_candidate for fact in facts] == ["forward_primer_name"]
    assert len(backend.calls) == 1


def test_recall_second_pass_fires_only_when_first_pass_finds_nothing():
    section_text = "PCR reactions used MiFish-U-F and MiFish-U-R primers at 54 C."

    def respond(prompt):
        if "[recall]" in prompt:
            return json.dumps(
                [
                    {"fact_type_candidate": "forward_primer_name", "raw_value": "MiFish-U-F", "evidence_id": "PCR.P001"},
                    {"fact_type_candidate": "reverse_primer_name", "raw_value": "MiFish-U-R", "evidence_id": "PCR.P001"},
                ]
            )
        return "[]"

    backend = MockLLMBackend(responses=respond)
    facts, _ = extract_facts_from_section(backend, "PCR", section_text, active_flags=frozenset({"pcr_0_1"}))

    assert {fact.fact_type_candidate for fact in facts} == {"forward_primer_name", "reverse_primer_name"}
    assert len(backend.calls) == 2
    assert "This is a recall-focused second pass" in backend.calls[1]["prompt"]
    assert "Never return placeholder absence values" in backend.calls[1]["prompt"]


def test_extraction_filters_model_invented_fact_type_names():
    section_text = "Surface sediment from the upper few millimeters was collected with a van Veen grab."
    response = json.dumps(
        [
            {"fact_type_candidate": "depth", "raw_value": "upper few millimeters", "evidence_id": "METHODS.P001"},
            {"fact_type_candidate": "sample_depth", "raw_value": "upper few millimeters", "evidence_id": "METHODS.P001"},
        ]
    )
    backend = MockLLMBackend(responses=[response])

    facts, _ = extract_facts_from_section(backend, "Methods", section_text)

    assert [fact.fact_type_candidate for fact in facts] == ["depth"]


def test_recall_second_pass_dedupes_repeated_first_pass_facts():
    section_text = "PCR reactions used MiFish-U-F primers."
    response = json.dumps(
        [{"fact_type_candidate": "forward_primer_name", "raw_value": "MiFish-U-F", "evidence_id": "PCR.P001"}]
    )
    backend = MockLLMBackend(responses=[response])

    facts, _ = extract_facts_from_section(backend, "PCR", section_text, active_flags=frozenset({"pcr_0_1"}))

    assert [fact.fact_type_candidate for fact in facts] == ["forward_primer_name"]


def test_recall_second_pass_failure_fails_the_extraction():
    from fair_ocean_agent.llm.base import LLMBackendError

    class RecallFailsBackend(MockLLMBackend):
        def generate(self, *args, **kwargs):
            if self.calls:
                raise LLMBackendError("recall failed")
            return super().generate(*args, **kwargs)

    backend = RecallFailsBackend(responses=["[]"])

    with pytest.raises(LLMBackendError, match="recall failed"):
        extract_facts_from_section(backend, "PCR", "PCR reactions used MiFish-U-F primers.")


def test_recall_missing_fact_types_skips_primer_sequences_without_nucleotide_text():
    primer_focus = next(focus for focus in EXTRACTION_FOCUSES if focus.name == "primer_target")
    name_only_segments = segment_source_text("PCR", "PCR used MiFish-U-F and MiFish-U-R primers.")
    sequence_segments = segment_source_text("PCR", "PCR used primers GTGYCAGCMGCCGCGGTAA and GGACTACNVGGGTWTCTAAT.")

    active_flags = frozenset({"pcr_0_1"})
    name_only_missing = recall_missing_fact_types(primer_focus, frozenset(), set(), name_only_segments, active_flags=active_flags)
    sequence_missing = recall_missing_fact_types(primer_focus, frozenset(), set(), sequence_segments, active_flags=active_flags)

    assert "forward_primer_sequence" not in name_only_missing
    assert "reverse_primer_sequence" not in name_only_missing
    assert "forward_primer_sequence" in sequence_missing
    assert "reverse_primer_sequence" in sequence_missing


def test_rejects_pcr_assay_facts_from_sterility_check_sentence():
    response = json.dumps(
        [
            {
                "fact_type_candidate": "amplicon_size",
                "raw_value": "441",
                "evidence_id": "PCR.P001",
            },
            {
                "fact_type_candidate": "forward_primer_name",
                "raw_value": "27F",
                "evidence_id": "PCR.P001",
            },
        ]
    )
    backend = MockLLMBackend(responses=[response, response])
    text = (
        "A lack of bacterial growth on 2216E plates after 3 days of incubation in 19C "
        "and the absence of bands under 16S rRNA gene PCR amplification with primers "
        "27F (5'-AGAGTTTGATCMTGGCTCAG-3') and 1492R "
        "(5'-GGTTACCTTGTTACGACTT-3') were considered confirmation of a sterile state."
    )

    facts, _ = extract_facts_from_section(backend, "PCR", text, active_flags=frozenset({"pcr_0_1"}))

    assert facts == []


# --- FAIRe-aware taxonomy regression tests (Milestone 8, corrected in v3) ---
# Before extraction/faire_fields.py existed, this prompt was fully open-
# vocabulary -- nothing forced the model's fact_type_candidate choices to
# line up with any structured taxonomy, so mapping/rules.py could only ever
# route what it extracted to FAIRe's free-text "*_method_additional"
# fallback fields. v2 fixed the coverage gap but coupled fact_type_candidate
# to FAIRe's own field spellings; v3 keeps the same atomic coverage but
# fact_type_candidate is always a standard-agnostic native_name, with a
# FAIRe field name only ever appearing as an optional candidate_standard_fields
# hint. These tests check the prompt carries the native-name checklist (the
# fact identity) and, separately, the FAIRe hints (never the fact identity).


def test_prompt_embeds_the_native_name_checklist():
    prompt = build_prompt("PCR", SECTION_TEXT, active_flags=frozenset({"pcr_0_1"}))
    # Spot-check one native_name from each group actually appears in the
    # prompt the model sees, not just in extraction/faire_fields.py.
    # "Sequencing / library prep" has no representative here at all
    # anymore: every one of its fields (platform/instrument/seq_kit/
    # lib_layout/adapter_forward/adapter_reverse) was already excluded via
    # LLM_EXCLUDED_OPTIONAL_FAIRE_FIELDS, and phix_percentage -- the one
    # remaining askable field in that group -- was removed from the
    # taxonomy entirely per an explicit user request (phix_perc is now
    # handled by a deterministic regex pass instead, search_flags.py's
    # detect_phix_percentage_facts), so this whole group is now 100%
    # excluded from LLM-askability. "Controls & replicates"'s own
    # representative used to be negative_control_type, then biological_
    # replicate_count; negative_control_type (and its positive_control_type
    # sibling) were removed from the taxonomy entirely per an explicit,
    # repeated user request, and biological_replicate_count was removed
    # per a later explicit user request (a live audit of a real 5-paper
    # run found it never actually fired), leaving pcr_replicate_count as
    # this group's only remaining concept.
    # "Bioinformatics workflow"'s own representative used to be read_
    # merge_minimum_overlap; that field's own FAIRe target
    # (merge_min_overlap) was removed entirely per an explicit, repeated
    # user request, and every remaining field in this group (clustering_
    # tool/reference_database/taxonomic_assignment_method) was already
    # excluded via LLM_EXCLUDED_OPTIONAL_FAIRE_FIELDS, so this whole group
    # is now 100% excluded from LLM-askability too.
    for native_name in (
        "annealing_temperature",
        "pcr_replicate_count",
        "standard_curve_slope",
        "scientific_name",
    ):
        assert native_name in prompt


def test_prompt_embeds_faire_hints_as_hints_not_identity():
    prompt = build_prompt("PCR", SECTION_TEXT, active_flags=frozenset({"pcr_0_1"}))
    assert "[FAIRe hint: annealingTemp]" in prompt
    assert "candidate_standard_fields" in prompt
    assert "evidence_id" in prompt
    assert "evidence_quote" not in prompt


def test_focused_prompt_only_embeds_requested_topic_checklist():
    pcr_focus = next(focus for focus in EXTRACTION_FOCUSES if focus.name == "pcr_assay_setup")
    prompt = build_prompt("PCR", SECTION_TEXT, focus=pcr_focus)

    assert "This focused pass is only for assay identity, PCR thermal cycling conditions" in prompt
    assert "annealing_temperature" in prompt
    assert "reference_database" not in prompt
    assert "sequencing_instrument" not in prompt
    # native_names further restricts this focus below its own group_names
    # (PCR / assay setup, Controls & replicates) -- primer_target's own
    # fields must NOT leak into this sibling focus's checklist.
    assert "forward_primer_sequence" not in prompt


def test_pcr_assay_setup_focus_still_asks_for_both_pcr_narrative_fields():
    """Real gap found live: once real production extraction started
    passing focuses=EXTRACTION_FOCUSES (v17), pcr_method_additional/
    pcr2_method_additional (PCR_amplification_conditions/second_pcr_
    amplification_conditions) came back completely blank -- the
    pcr_assay_setup focus's own native_names allowlist never included
    either one, and its fallback_names={"PCR_amplification_conditions"}
    was already dead (that field had already been moved out of
    FALLBACK_NARRATIVE_FIELDS into FIELD_GROUPS by an earlier fix, and
    include_fallback_names only ever restricts the FALLBACK_NARRATIVE_
    FIELDS loop, never FIELD_GROUPS)."""
    pcr_focus = next(focus for focus in EXTRACTION_FOCUSES if focus.name == "pcr_assay_setup")
    prompt = build_prompt("PCR", SECTION_TEXT, focus=pcr_focus, active_flags=frozenset({"pcr_0_1"}))

    assert "PCR_amplification_conditions" in prompt
    assert "second_pcr_amplification_conditions" in prompt


def test_segments_for_focus_skips_unrelated_topic_prompts():
    segments = segment_source_text(
        "Methods",
        "PCR reactions used MiFish-U-F and MiFish-U-R primers.\n\nLibraries were sequenced on an Illumina MiSeq.",
    )
    primer_focus = next(focus for focus in EXTRACTION_FOCUSES if focus.name == "primer_target")
    sequencing_focus = next(focus for focus in EXTRACTION_FOCUSES if focus.name == "sequencing_library")

    assert [segment.segment_id for segment in segments_for_focus("Methods", segments, primer_focus)] == ["METHODS.P001"]
    assert [segment.segment_id for segment in segments_for_focus("Methods", segments, sequencing_focus)] == ["METHODS.P002"]


def test_every_llm_enabled_native_field_name_appears_in_instructions():
    for field_name in field_names_for_reference():
        assert field_name in EXTRACTION_INSTRUCTIONS, f"{field_name!r} missing from the extraction prompt"


def test_extracting_an_atomic_native_field_by_exact_name():
    """A model following the prompt should report the concept's native
    name, not FAIRe's own spelling -- confirms extract_facts_from_section
    doesn't rewrite or reject a native name like "annealing_temperature"."""
    section_text = "PCR was performed with an annealing temperature of 57C for 35 cycles."
    response = json.dumps(
        [
            {
                "fact_type_candidate": "annealing_temperature",
                "raw_value": "57C",
                "evidence_id": "PCR.P001",
                "candidate_standard_fields": {"faire": "annealingTemp"},
            },
            {"fact_type_candidate": "pcr_cycle_count", "raw_value": "35", "evidence_id": "PCR.P001"},
        ]
    )
    backend = MockLLMBackend(responses=[response])
    facts, _ = extract_facts_from_section(backend, "PCR", section_text, active_flags=frozenset({"pcr_0_1"}))

    fact_types = {f.fact_type_candidate for f in facts}
    assert fact_types == {"annealing_temperature", "pcr_cycle_count"}


def test_candidate_standard_fields_hint_is_stored_in_confidence_metadata():
    """The hint must land in confidence_metadata, completely separate from
    fact_type_candidate -- dropping this field entirely must still leave a
    fully valid, standard-agnostic raw fact."""
    section_text = "PCR was performed with an annealing temperature of 57C."
    response = json.dumps(
        [
            {
                "fact_type_candidate": "annealing_temperature",
                "raw_value": "57C",
                "evidence_id": "PCR.P001",
                "candidate_standard_fields": {"faire": "annealingTemp"},
            }
        ]
    )
    backend = MockLLMBackend(responses=[response])
    facts, _ = extract_facts_from_section(backend, "PCR", section_text, active_flags=frozenset({"pcr_0_1"}))

    assert len(facts) == 1
    assert facts[0].fact_type_candidate == "annealing_temperature"
    assert facts[0].confidence_metadata == {
        "evidence_ids": ["PCR.P001"],
        "candidate_standard_fields": {"faire": "annealingTemp"},
    }


def test_omitted_candidate_standard_fields_hint_leaves_a_valid_fact():
    section_text = "PCR was performed with an annealing temperature of 57C."
    response = json.dumps(
        [{"fact_type_candidate": "annealing_temperature", "raw_value": "57C", "evidence_id": "PCR.P001"}]
    )
    backend = MockLLMBackend(responses=[response])
    facts, _ = extract_facts_from_section(backend, "PCR", section_text, active_flags=frozenset({"pcr_0_1"}))

    assert len(facts) == 1
    assert facts[0].fact_type_candidate == "annealing_temperature"
    assert facts[0].confidence_metadata == {"evidence_ids": ["PCR.P001"]}


# --- Structured-first extraction: skip asking about already-resolved
# FAIRe fields (see extraction/text.py's module docstring and
# resolved_faire_fields_for_study) -----------------------------------------


def _standardized_value(
    session,
    study,
    *,
    target_field,
    missingness_status,
    value=None,
    mapping_method=MappingMethod.EXACT_LABEL.value,
    review_required=False,
):
    sv = StandardizedValue(
        study_id=study.study_id,
        target_schema=TARGET_SCHEMA,
        target_schema_version=TARGET_SCHEMA_VERSION,
        target_field=target_field,
        standardized_value=value,
        missingness_status=missingness_status,
        mapping_method=mapping_method,
        review_required=review_required,
    )
    session.add(sv)
    session.flush()
    return sv


def test_resolved_faire_fields_for_study_returns_only_present_values(db_session):
    study = Study(title="structured-first test")
    db_session.add(study)
    db_session.flush()

    _standardized_value(db_session, study, target_field="annealingTemp", missingness_status=MissingnessStatus.PRESENT.value, value="55C")
    _standardized_value(db_session, study, target_field="pcr_cycles", missingness_status=MissingnessStatus.NOT_FOUND_IN_INSPECTED_SOURCES.value)
    _standardized_value(db_session, study, target_field="otu_db", missingness_status=MissingnessStatus.MAPPING_UNRESOLVED.value)

    resolved = resolved_faire_fields_for_study(db_session, study.study_id)
    assert resolved == frozenset({"annealingTemp"})


def test_resolved_faire_fields_for_study_does_not_suppress_review_or_llm_rows(db_session):
    study = Study(title="review rows are not trusted enough to suppress LLM")
    db_session.add(study)
    db_session.flush()

    _standardized_value(
        db_session,
        study,
        target_field="annealingTemp",
        missingness_status=MissingnessStatus.PRESENT.value,
        value="55C",
        review_required=True,
    )
    _standardized_value(
        db_session,
        study,
        target_field="otu_db",
        missingness_status=MissingnessStatus.PRESENT.value,
        value="SILVA 138",
        mapping_method=MappingMethod.SUGGESTED_SEMANTIC.value,
    )

    assert resolved_faire_fields_for_study(db_session, study.study_id) == frozenset()


def test_present_faire_fields_for_supplement_includes_paper_review_rows(db_session):
    study = Study(title="paper facts should suppress duplicate supplement asks")
    db_session.add(study)
    db_session.flush()
    _standardized_value(
        db_session,
        study,
        target_field="annealingTemp",
        missingness_status=MissingnessStatus.PRESENT.value,
        value="55C",
        review_required=True,
        mapping_method=MappingMethod.SUGGESTED_SEMANTIC.value,
    )

    assert present_faire_fields_for_study(db_session, study.study_id) == frozenset(
        {"annealingTemp"}
    )


def test_resolved_faire_fields_for_study_is_empty_when_map_faire_never_ran(db_session):
    """No StandardizedValue rows at all for a study (MAP_FAIRE hasn't run
    yet) must return an empty set, not raise -- callers get the exact
    "ask about everything" behavior this pipeline had before this existed."""
    study = Study(title="never mapped")
    db_session.add(study)
    db_session.flush()

    assert resolved_faire_fields_for_study(db_session, study.study_id) == frozenset()


def test_resolved_faire_fields_for_study_ignores_other_studies_and_schemas(db_session):
    study_a = Study(title="a")
    study_b = Study(title="b")
    db_session.add_all([study_a, study_b])
    db_session.flush()

    _standardized_value(db_session, study_a, target_field="annealingTemp", missingness_status=MissingnessStatus.PRESENT.value, value="55C")
    _standardized_value(
        db_session,
        study_b,
        target_field="tax_assign_cat",
        missingness_status=MissingnessStatus.PRESENT.value,
        value="naive Bayes classifier",
    )
    # A non-FAIRe schema row for study_a must never leak into the result.
    other_schema = StandardizedValue(
        study_id=study_a.study_id,
        target_schema="bebop",
        target_schema_version="1.0",
        target_field="annealingTemp",
        standardized_value="55C",
        missingness_status=MissingnessStatus.PRESENT.value,
    )
    db_session.add(other_schema)
    db_session.flush()

    assert resolved_faire_fields_for_study(db_session, study_a.study_id) == frozenset({"annealingTemp"})
    assert resolved_faire_fields_for_study(db_session, study_b.study_id) == frozenset({"tax_assign_cat"})


def test_build_prompt_excludes_resolved_faire_hint_fields():
    # "scientific_name" (unlike "annealing_temperature") never appears in
    # the instructions' own static illustrative text, so its absence here
    # unambiguously means the checklist entry was actually filtered out.
    prompt_full = build_prompt("PCR", SECTION_TEXT)
    prompt_filtered = build_prompt("PCR", SECTION_TEXT, exclude_faire_hints=frozenset({"scientificName"}))

    assert "scientific_name" in prompt_full
    assert "scientific_name" not in prompt_filtered
    assert len(prompt_filtered) < len(prompt_full)


def test_extract_facts_from_section_passes_exclusions_into_the_prompt():
    """Integration-level check that exclude_faire_hints actually reaches
    the prompt the backend receives, not just build_prompt in isolation."""
    backend = MockLLMBackend(responses=["[]"])
    extract_facts_from_section(backend, "PCR", SECTION_TEXT, exclude_faire_hints=frozenset({"sop_bioinformatics"}))
    assert "bioinformatics_sop_reference" not in backend.calls[-1]["prompt"]


# --- Flag-gated checklist (extraction/faire_fields.py's required_any_flags,
# v15 -> v16): a paper with no detected PCR evidence should never even be
# asked about the "PCR / assay setup" group; a probe-based qPCR/ddPCR paper
# should additionally see probe_sequence/probe_concentration. ------------


def test_build_prompt_hides_pcr_checklist_with_no_active_flags():
    # pcr_cycle_count (unlike annealing_temperature/target_gene) never
    # appears in the static illustrative instruction text, so its absence
    # here unambiguously means the checklist entry was actually filtered.
    prompt = build_prompt("PCR", SECTION_TEXT)
    assert "pcr_cycle_count" not in prompt
    assert "probe_sequence" not in prompt
    # An ungated concept from another group must still be present.
    assert "scientific_name" in prompt


def test_build_prompt_shows_pcr_checklist_when_pcr_0_1_active():
    prompt = build_prompt("PCR", SECTION_TEXT, active_flags=frozenset({"pcr_0_1"}))
    assert "pcr_cycle_count" in prompt
    # probe_sequence requires pcr_0_1 OR probe_based -- pcr_0_1 alone is
    # sufficient (required_any_flags is an OR, matching the source CSV's
    # own "if pcr_0_1 TRUE | if probe_based... TRUE" conditional column).
    assert "probe_sequence" in prompt


def test_build_prompt_hides_probe_fields_without_either_probe_flag():
    """probe_sequence/probe_concentration require pcr_0_1 OR
    probe_based_qPCR_ddPCR_assay_0_1 -- covered by the previous test.
    This one confirms neither flag active means neither shows, even
    though other PCR fields are also absent for the same reason."""
    prompt = build_prompt("PCR", SECTION_TEXT)
    assert "probe_concentration" not in prompt


def test_extract_facts_from_section_rejects_gated_fact_when_flag_inactive():
    """A model that reports a gated fact_type_candidate anyway (e.g.
    hallucinated from general knowledge, never actually shown in the
    prompt) must still be rejected by allowed_fact_types -- gating must
    hold on both the input (what's shown) and output (what's accepted)
    sides, not just the rendered prompt text."""
    section_text = "PCR was performed with an annealing temperature of 57C."
    response = json.dumps(
        [{"fact_type_candidate": "annealing_temperature", "raw_value": "57C", "evidence_id": "PCR.P001"}]
    )
    backend = MockLLMBackend(responses=[response])
    facts, _ = extract_facts_from_section(backend, "PCR", section_text)  # no active_flags
    assert facts == []


def test_extract_facts_from_section_accepts_gated_fact_when_flag_active():
    section_text = "PCR was performed with an annealing temperature of 57C."
    response = json.dumps(
        [{"fact_type_candidate": "annealing_temperature", "raw_value": "57C", "evidence_id": "PCR.P001"}]
    )
    backend = MockLLMBackend(responses=[response])
    facts, _ = extract_facts_from_section(backend, "PCR", section_text, active_flags=frozenset({"pcr_0_1"}))
    assert len(facts) == 1
    assert facts[0].fact_type_candidate == "annealing_temperature"
