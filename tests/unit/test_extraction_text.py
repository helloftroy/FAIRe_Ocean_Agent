import json

from fair_ocean_agent.database.enums import MissingnessStatus, SupportType
from fair_ocean_agent.database.models import StandardizedValue, Study
from fair_ocean_agent.extraction.faire_fields import all_field_names
from fair_ocean_agent.extraction.text import (
    EXTRACTION_INSTRUCTIONS,
    PROMPT_VERSION,
    build_prompt,
    extract_facts_from_section,
    resolved_faire_fields_for_study,
)
from fair_ocean_agent.llm.mock import MockLLMBackend
from fair_ocean_agent.mapping.faire import TARGET_SCHEMA, TARGET_SCHEMA_VERSION

SECTION_TEXT = "Samples were collected on 4 January 2022 at a depth of 5 meters near the reef."


def test_verified_fact_is_kept_with_evidence_quote():
    response = json.dumps(
        [{"fact_type_candidate": "collection_date", "raw_value": "2022-01-04", "evidence_quote": "Samples were collected on 4 January 2022"}]
    )
    backend = MockLLMBackend(responses=[response])
    facts, _ = extract_facts_from_section(backend, "Methods", SECTION_TEXT)

    assert len(facts) == 1
    assert facts[0].fact_type_candidate == "collection_date"
    assert facts[0].evidence_quote == "Samples were collected on 4 January 2022"
    assert facts[0].support_type == SupportType.EXPLICIT


def test_fabricated_quote_is_dropped():
    response = json.dumps(
        [{"fact_type_candidate": "fake", "raw_value": "x", "evidence_quote": "this sentence is not in the source"}]
    )
    backend = MockLLMBackend(responses=[response])
    facts, _ = extract_facts_from_section(backend, "Methods", SECTION_TEXT)
    assert facts == []


def test_missing_fields_are_skipped_not_crashed_on():
    response = json.dumps([{"evidence_quote": "Samples were collected on 4 January 2022"}])  # no fact_type/raw_value
    backend = MockLLMBackend(responses=[response])
    facts, _ = extract_facts_from_section(backend, "Methods", SECTION_TEXT)
    assert facts == []


def test_invalid_json_response_yields_no_facts():
    backend = MockLLMBackend(responses=["not json"] * 5)
    facts, response = extract_facts_from_section(backend, "Methods", SECTION_TEXT)
    assert facts == []
    assert response is not None


def test_prompt_version_is_stable_constant():
    assert PROMPT_VERSION == "text-extraction-v3-native-with-hints"


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


def test_every_native_field_name_appears_in_instructions():
    for field_name in all_field_names():
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
                "evidence_quote": "an annealing temperature of 57C",
                "candidate_standard_fields": {"faire": "annealingTemp"},
            },
            {"fact_type_candidate": "pcr_cycle_count", "raw_value": "35", "evidence_quote": "for 35 cycles"},
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
                "evidence_quote": "an annealing temperature of 57C",
                "candidate_standard_fields": {"faire": "annealingTemp"},
            }
        ]
    )
    backend = MockLLMBackend(responses=[response])
    facts, _ = extract_facts_from_section(backend, "PCR", section_text)

    assert len(facts) == 1
    assert facts[0].fact_type_candidate == "annealing_temperature"
    assert facts[0].confidence_metadata == {"candidate_standard_fields": {"faire": "annealingTemp"}}


def test_omitted_candidate_standard_fields_hint_leaves_a_valid_fact():
    section_text = "PCR was performed with an annealing temperature of 57C."
    response = json.dumps(
        [{"fact_type_candidate": "annealing_temperature", "raw_value": "57C", "evidence_quote": "an annealing temperature of 57C"}]
    )
    backend = MockLLMBackend(responses=[response])
    facts, _ = extract_facts_from_section(backend, "PCR", section_text)

    assert len(facts) == 1
    assert facts[0].fact_type_candidate == "annealing_temperature"
    assert facts[0].confidence_metadata is None


# --- Structured-first extraction: skip asking about already-resolved
# FAIRe fields (see extraction/text.py's module docstring and
# resolved_faire_fields_for_study) -----------------------------------------


def _standardized_value(session, study, *, target_field, missingness_status, value=None):
    sv = StandardizedValue(
        study_id=study.study_id,
        target_schema=TARGET_SCHEMA,
        target_schema_version=TARGET_SCHEMA_VERSION,
        target_field=target_field,
        standardized_value=value,
        missingness_status=missingness_status,
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
