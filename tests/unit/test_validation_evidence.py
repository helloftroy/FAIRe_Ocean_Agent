from fair_ocean_agent.database.enums import SupportType
from fair_ocean_agent.validation.evidence import check_raw_fact_evidence_consistency


def test_explicit_fact_with_evidence_quote_is_ok():
    result = check_raw_fact_evidence_consistency(SupportType.EXPLICIT.value, "the quote", None)
    assert result.ok


def test_explicit_fact_without_evidence_quote_is_flagged():
    result = check_raw_fact_evidence_consistency(SupportType.EXPLICIT.value, None, None)
    assert not result.ok
    assert "evidence_quote" in result.message


def test_explicit_fact_with_blank_evidence_quote_is_flagged():
    result = check_raw_fact_evidence_consistency(SupportType.EXPLICIT.value, "   ", None)
    assert not result.ok


def test_structured_source_fact_with_locator_is_ok():
    result = check_raw_fact_evidence_consistency(SupportType.STRUCTURED_SOURCE.value, None, "crossref.message.title")
    assert result.ok


def test_structured_source_fact_without_locator_is_flagged():
    result = check_raw_fact_evidence_consistency(SupportType.STRUCTURED_SOURCE.value, None, None)
    assert not result.ok
    assert "source_locator" in result.message


def test_other_support_types_have_no_requirement():
    result = check_raw_fact_evidence_consistency(SupportType.INFERRED.value, None, None)
    assert result.ok
