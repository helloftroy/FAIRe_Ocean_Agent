import json

import pytest

from fair_ocean_agent.llm.benchmark import (
    GoldCase,
    _normalize_label,
    _score_case,
    _values_equivalent,
    export_report,
    load_gold_cases,
    run_benchmark,
)
from fair_ocean_agent.llm.mock import MockLLMBackend

CASE = GoldCase(
    case_id="case-1",
    source_text="Samples were collected on 4 January 2022 at a depth of 5 meters.",
    instructions="Extract collection_date and sampling_depth.",
    expected_facts=[
        {"fact_type_candidate": "collection_date", "raw_value": "2022-01-04", "evidence_quote": "collected on 4 January 2022"},
        {"fact_type_candidate": "sampling_depth", "raw_value": "5 meters", "evidence_quote": "depth of 5 meters"},
    ],
    curated_by="claude",
)


def test_load_gold_cases_from_directory(tmp_path):
    (tmp_path / "a.json").write_text(CASE.model_dump_json())
    cases = load_gold_cases(tmp_path)
    assert len(cases) == 1
    assert cases[0].case_id == "case-1"


def test_example_gold_case_includes_primer_names_requested_by_instructions():
    cases = load_gold_cases("data/benchmark/gold")
    example = next(case for case in cases if case.case_id == "example-001")

    requested_primer_names = "primer names" in example.instructions.lower()
    expected_labels = {_normalize_label(f.fact_type_candidate) for f in example.expected_facts}

    assert requested_primer_names
    assert "primer names" in expected_labels


def test_run_benchmark_perfect_model_gets_perfect_scores():
    response = json.dumps(
        [
            {"fact_type_candidate": "collection_date", "raw_value": "2022-01-04", "evidence_quote": "collected on 4 January 2022"},
            {"fact_type_candidate": "sampling_depth", "raw_value": "5 meters", "evidence_quote": "depth of 5 meters"},
        ]
    )
    backend = MockLLMBackend(label="perfect", responses=[response])
    report = run_benchmark([backend], [CASE])

    metrics, results = report["perfect"]
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.json_validity_rate == 1.0
    assert metrics.evidence_verification_rate == 1.0
    assert len(results) == 1


def test_run_benchmark_hallucinated_fact_is_caught_by_evidence_verification():
    response = json.dumps([{"fact_type_candidate": "made_up", "raw_value": "x", "evidence_quote": "not present anywhere"}])
    backend = MockLLMBackend(label="hallucinator", responses=[response])
    report = run_benchmark([backend], [CASE])

    metrics, _ = report["hallucinator"]
    assert metrics.evidence_verification_rate == 0.0
    assert metrics.recall == 0.0  # nothing verified, so nothing counted toward gold


def test_run_benchmark_invalid_json_model_scores_zero_validity():
    backend = MockLLMBackend(label="broken", responses=["not json"] * 5)
    report = run_benchmark([backend], [CASE])

    metrics, _ = report["broken"]
    assert metrics.json_validity_rate == 0.0
    assert metrics.errors == 0  # invalid JSON isn't a backend error, just an unparseable response


def test_run_benchmark_raises_with_no_gold_cases():
    with pytest.raises(ValueError):
        run_benchmark([MockLLMBackend()], [])


def test_export_report_writes_csv_and_json(tmp_path):
    backend = MockLLMBackend(label="model-a", responses=["[]"])
    report = run_benchmark([backend], [CASE])
    export_report(report, tmp_path)

    csv_content = (tmp_path / "summary.csv").read_text()
    assert "model-a" in csv_content
    assert "precision" in csv_content

    detail = json.loads((tmp_path / "detail.json").read_text())
    assert "model-a" in detail
    assert detail["model-a"]["cases"][0]["case_id"] == "case-1"


def test_summarize_multiple_cases_averages_latency():
    backend = MockLLMBackend(label="two-case", responses=["[]", "[]"], simulated_latency_seconds=0.01)
    case_2 = CASE.model_copy(update={"case_id": "case-2"})
    report = run_benchmark([backend], [CASE, case_2])

    metrics, results = report["two-case"]
    assert metrics.n_cases == 2
    assert metrics.mean_latency_seconds >= 0.01


# --- Regression tests for a real finding: qwen2.5:3b scored 0.00 precision/
# recall on a case it answered entirely correctly, because it wrote
# "collection date" / "14 March 2021" where gold wrote "collection_date" /
# "2021-03-14". Label spacing and date formatting aren't disagreements about
# the underlying fact -- see llm/benchmark.py's _normalize_label and
# _values_equivalent docstrings.


def test_normalize_label_treats_underscores_hyphens_and_spaces_as_equivalent():
    assert _normalize_label("collection_date") == _normalize_label("collection date")
    assert _normalize_label("DNA-extraction-method") == _normalize_label("dna_extraction_method")


def test_values_equivalent_treats_reformatted_dates_as_equal():
    assert _values_equivalent("14 March 2021", "2021-03-14")
    assert _values_equivalent("2021-03-14", "March 14, 2021")


def test_values_equivalent_does_not_conflate_different_dates():
    assert not _values_equivalent("14 March 2021", "15 March 2021")


def test_values_equivalent_does_not_treat_differently_precise_dates_as_equal():
    # "March 2021" (no day) must not spuriously match a specific day just
    # because of a parser default -- see _DATE_PARSE_ANCHOR's docstring.
    assert not _values_equivalent("March 2021", "14 March 2021")


def test_values_equivalent_still_requires_exact_match_for_non_dates():
    assert not _values_equivalent("DNeasy PowerWater kit", "Qiagen DNeasy kit")
    assert _values_equivalent("DNeasy PowerWater kit", "  DNeasy   PowerWater KIT  ")


def test_score_case_credits_reformatted_label_and_date_as_a_match():
    case = GoldCase(
        case_id="c",
        source_text="x",
        instructions="x",
        expected_facts=[{"fact_type_candidate": "collection_date", "raw_value": "2021-03-14", "evidence_quote": "x"}],
    )
    verified = [{"fact_type_candidate": "collection date", "raw_value": "14 March 2021", "evidence_quote": "x"}]
    tp, fp, fn = _score_case(case, verified)
    assert (tp, fp, fn) == (1, 0, 0)


def test_qwen_real_world_case_now_scores_correctly():
    """The exact real result from a live qwen2.5:3b run that originally
    scored precision=recall=0.0 despite being fully correct."""
    case = GoldCase(
        case_id="real-qwen-result",
        source_text="irrelevant for this check",
        instructions="irrelevant",
        expected_facts=[
            {"fact_type_candidate": "collection_date", "raw_value": "2021-03-14", "evidence_quote": "x"},
            {"fact_type_candidate": "sampling_depth", "raw_value": "5 meters", "evidence_quote": "x"},
            {"fact_type_candidate": "dna_extraction_method", "raw_value": "DNeasy PowerWater kit", "evidence_quote": "x"},
            {"fact_type_candidate": "sequencing_platform", "raw_value": "Illumina MiSeq", "evidence_quote": "x"},
        ],
    )
    verified = [
        {"fact_type_candidate": "collection date", "raw_value": "14 March 2021", "evidence_quote": "x"},
        {"fact_type_candidate": "sampling depth", "raw_value": "5 meters", "evidence_quote": "x"},
        {"fact_type_candidate": "DNA extraction method", "raw_value": "DNeasy PowerWater kit", "evidence_quote": "x"},
        {"fact_type_candidate": "sequencing platform", "raw_value": "Illumina MiSeq", "evidence_quote": "x"},
    ]
    tp, fp, fn = _score_case(case, verified)
    assert (tp, fp, fn) == (4, 0, 0)
