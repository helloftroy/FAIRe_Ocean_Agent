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
    facts, _ = extract_facts_from_section(backend, "Methods", text)

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
    facts, _ = extract_facts_from_section(backend, "Methods", SECTION_TEXT)

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
    facts, _ = extract_facts_from_section(backend, "Methods", SECTION_TEXT)

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
    assert PROMPT_VERSION == "text-extraction-v13-control-flags"


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

    facts, _ = extract_facts_from_section(backend, "PCR", section_text)

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
    facts, _ = extract_facts_from_section(backend, "PCR", section_text)

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

    facts, _ = extract_facts_from_section(backend, "PCR", section_text)

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
    primer_focus = next(focus for focus in EXTRACTION_FOCUSES if focus.name == "primer_pcr_assay")
    name_only_segments = segment_source_text("PCR", "PCR used MiFish-U-F and MiFish-U-R primers.")
    sequence_segments = segment_source_text("PCR", "PCR used primers GTGYCAGCMGCCGCGGTAA and GGACTACNVGGGTWTCTAAT.")

    name_only_missing = recall_missing_fact_types(primer_focus, frozenset(), set(), name_only_segments)
    sequence_missing = recall_missing_fact_types(primer_focus, frozenset(), set(), sequence_segments)

    assert "forward_primer_sequence" not in name_only_missing
    assert "reverse_primer_sequence" not in name_only_missing
    assert "forward_primer_sequence" in sequence_missing
    assert "reverse_primer_sequence" in sequence_missing


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
    prompt = build_prompt("PCR", SECTION_TEXT)
    # Spot-check one native_name from each group actually appears in the
    # prompt the model sees, not just in extraction/faire_fields.py.
    for native_name in (
        "annealing_temperature",
        "negative_control_type",
        "standard_curve_slope",
        "sequencing_kit",
        "reference_database",
        "scientific_name",
    ):
        assert native_name in prompt


def test_prompt_embeds_faire_hints_as_hints_not_identity():
    prompt = build_prompt("PCR", SECTION_TEXT)
    assert "[FAIRe hint: annealingTemp]" in prompt
    assert "candidate_standard_fields" in prompt
    assert "evidence_id" in prompt
    assert "evidence_quote" not in prompt


def test_focused_prompt_only_embeds_requested_topic_checklist():
    primer_focus = next(focus for focus in EXTRACTION_FOCUSES if focus.name == "primer_pcr_assay")
    prompt = build_prompt("PCR", SECTION_TEXT, focus=primer_focus)

    assert "This focused pass is only for assay, target marker, primer" in prompt
    assert "annealing_temperature" in prompt
    assert "reference_database" not in prompt
    assert "sequencing_instrument" not in prompt


def test_segments_for_focus_skips_unrelated_topic_prompts():
    segments = segment_source_text(
        "Methods",
        "PCR reactions used MiFish-U-F and MiFish-U-R primers.\n\nLibraries were sequenced on an Illumina MiSeq.",
    )
    primer_focus = next(focus for focus in EXTRACTION_FOCUSES if focus.name == "primer_pcr_assay")
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
    facts, _ = extract_facts_from_section(backend, "PCR", section_text)

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
    facts, _ = extract_facts_from_section(backend, "PCR", section_text)

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
    facts, _ = extract_facts_from_section(backend, "PCR", section_text)

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
    _standardized_value(db_session, study_b, target_field="otu_db", missingness_status=MissingnessStatus.PRESENT.value, value="SILVA 138")
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
    assert resolved_faire_fields_for_study(db_session, study_b.study_id) == frozenset({"otu_db"})


def test_build_prompt_excludes_resolved_faire_hint_fields():
    # "reference_database" (unlike "annealing_temperature") never appears in
    # the instructions' own static illustrative text, so its absence here
    # unambiguously means the checklist entry was actually filtered out.
    prompt_full = build_prompt("PCR", SECTION_TEXT)
    prompt_filtered = build_prompt("PCR", SECTION_TEXT, exclude_faire_hints=frozenset({"otu_db"}))

    assert "reference_database" in prompt_full
    assert "reference_database" not in prompt_filtered
    assert len(prompt_filtered) < len(prompt_full)


def test_extract_facts_from_section_passes_exclusions_into_the_prompt():
    """Integration-level check that exclude_faire_hints actually reaches
    the prompt the backend receives, not just build_prompt in isolation."""
    backend = MockLLMBackend(responses=["[]"])
    extract_facts_from_section(backend, "PCR", SECTION_TEXT, exclude_faire_hints=frozenset({"otu_db"}))
    assert "reference_database" not in backend.calls[-1]["prompt"]
