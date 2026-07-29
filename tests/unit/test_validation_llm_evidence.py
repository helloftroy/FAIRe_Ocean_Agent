import json

from fair_ocean_agent.database.enums import ValidationStatus
from fair_ocean_agent.llm.mock import MockLLMBackend
from fair_ocean_agent.validation.llm_evidence import build_support_check_prompt, verify_fact_support_with_llm


def test_build_support_check_prompt_asks_for_boolean_json():
    prompt = build_support_check_prompt(
        "annealing_temperature",
        "54 C",
        "Reactions were annealed at 54 C for 35 cycles.",
    )

    assert "supported (boolean)" in prompt
    assert "annealing_temperature" in prompt
    assert "54 C" in prompt
    assert "Do not use outside knowledge" in prompt


def test_verify_fact_support_with_llm_returns_supported_outcome():
    backend = MockLLMBackend(responses=[json.dumps({"supported": True, "reason": "Quote states 54 C."})])

    result = verify_fact_support_with_llm(
        backend,
        fact_type_candidate="annealing_temperature",
        raw_value="54 C",
        evidence_quote="Reactions were annealed at 54 C for 35 cycles.",
    )

    assert result.outcome.status == ValidationStatus.SUPPORTED.value
    assert result.outcome.compared_values["supported"] is True


def test_verify_fact_support_with_llm_returns_unsupported_outcome():
    backend = MockLLMBackend(responses=[json.dumps({"supported": False, "reason": "Quote gives a date, not temperature."})])

    result = verify_fact_support_with_llm(
        backend,
        fact_type_candidate="annealing_temperature",
        raw_value="54 C",
        evidence_quote="Samples were collected on 4 January 2022.",
    )

    assert result.outcome.status == ValidationStatus.UNSUPPORTED.value
    assert result.outcome.compared_values["supported"] is False


def test_verify_fact_support_with_llm_invalid_json_is_not_assessed():
    backend = MockLLMBackend(responses=["[]"])

    result = verify_fact_support_with_llm(
        backend,
        fact_type_candidate="annealing_temperature",
        raw_value="54 C",
        evidence_quote="Reactions were annealed at 54 C for 35 cycles.",
    )

    assert result.outcome.status == ValidationStatus.NOT_ASSESSED.value
