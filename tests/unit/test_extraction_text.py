import json

from fair_ocean_agent.database.enums import SupportType
from fair_ocean_agent.extraction.text import PROMPT_VERSION, extract_facts_from_section
from fair_ocean_agent.llm.mock import MockLLMBackend

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
    assert PROMPT_VERSION == "text-extraction-v1"
