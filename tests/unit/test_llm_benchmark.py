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
from fair_ocean_agent.llm.base import LLMBackendError
from fair_ocean_agent.llm.mock import MockLLMBackend

CASE = GoldCase(
    case_id="case-1",
    source_text="Samples were collected on 4 January 2022 at a depth of 5 meters.",
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


def test_run_case_uses_the_real_production_prompt(monkeypatch):
    """Regression guard: gold cases used to carry their own free-form
    `instructions` field, rendered by a benchmark-local prompt template --
    a second copy of "the prompt" that could silently drift from what
    extraction/text.py actually sends in production. run_case must build
    its prompt via extraction.text.build_prompt, the same function
    handle_extract_text_facts calls."""
    import fair_ocean_agent.llm.benchmark as benchmark_module

    captured = {}
    real_build_prompt = benchmark_module.build_prompt

    def spy(section_title, section_text, **kwargs):
        captured["section_title"] = section_title
        captured["section_text"] = section_text
        return real_build_prompt(section_title, section_text, **kwargs)

    monkeypatch.setattr(benchmark_module, "build_prompt", spy)
    backend = MockLLMBackend(responses=["[]"])
    benchmark_module.run_case(backend, CASE)

    assert captured["section_title"].startswith(CASE.section_title)
    assert captured["section_text"] == CASE.source_text
    assert len(backend.calls) == 2


def test_gold_cases_use_native_taxonomy_names_or_a_documented_fallback():
    """Every real gold case's expected fact_type_candidate should either be
    an exact native_name from the taxonomy (extraction/faire_fields.py --
    standard-agnostic, never a FAIRe field spelling) or one of the two
    deliberate fallback-demo exceptions in example-001 (collection_date/
    sampling_depth -- normally repository-sourced, kept here only to
    demonstrate the prompt's open-fallback path still works). Catches gold
    data quietly drifting from the taxonomy it's meant to validate."""
    from fair_ocean_agent.extraction.faire_fields import all_field_names

    known_names = all_field_names()
    documented_fallback_exceptions = {
        "adapter_trimming_method",
        "chimera_detection_method",
        "clustering_similarity_threshold",
        "collection_date",
        "demux_tool",
        "denoising_tool",
        "dna_concentration_method",
        "forward_primer_concentration",
        "forward_primer_volume",
        "length_filtering_tool",
        "library_concentration",
        "library_concentration_method",
        "library_layout",
        "minimum_read_length",
        "pcr_conditions",
        "pcr_reaction_volume",
        "phix_percentage",
        "positive_control_type",
        "read_merge_minimum_overlap",
        "read_merging_tool",
        "reverse_primer_concentration",
        "reverse_primer_volume",
        "sampling_depth",
        "sequencing_location",
        "template_dna_volume",
        "thermocycler",
    }

    for case in load_gold_cases("data/benchmark/gold"):
        for fact in case.expected_facts:
            assert (
                fact.fact_type_candidate in known_names
                or fact.fact_type_candidate in documented_fallback_exceptions
            ), f"{case.case_id}: {fact.fact_type_candidate!r} is not in the native-name taxonomy or documented fallbacks"


def test_run_benchmark_perfect_model_gets_perfect_scores():
    response = json.dumps(
        [
            {"fact_type_candidate": "collection_date", "raw_value": "2022-01-04", "evidence_id": "METHODS.P001"},
            {"fact_type_candidate": "sampling_depth", "raw_value": "5 meters", "evidence_id": "METHODS.P001"},
        ]
    )
    backend = MockLLMBackend(
        label="perfect",
        responses=lambda prompt: response if "[recall]" not in prompt else "[]",
    )
    report = run_benchmark([backend], [CASE])

    metrics, results = report["perfect"]
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.json_validity_rate == 1.0
    assert metrics.evidence_verification_rate == 1.0
    assert len(results) == 1


def test_run_benchmark_unknown_evidence_id_is_caught_by_evidence_verification():
    response = json.dumps([{"fact_type_candidate": "made_up", "raw_value": "x", "evidence_id": "METHODS.P999"}])
    backend = MockLLMBackend(label="hallucinator", responses=[response])
    report = run_benchmark([backend], [CASE])

    metrics, _ = report["hallucinator"]
    assert metrics.evidence_verification_rate == 0.0
    assert metrics.recall == 0.0  # nothing verified, so nothing counted toward gold


def test_run_benchmark_filters_absence_placeholder_values_before_scoring():
    response = json.dumps(
        [
            {"fact_type_candidate": "collection_date", "raw_value": "not specified", "evidence_id": "METHODS.P001"},
            {"fact_type_candidate": "sampling_depth", "raw_value": "none", "evidence_id": "METHODS.P001"},
        ]
    )
    backend = MockLLMBackend(label="placeholder-model", responses=lambda prompt: response if "[recall]" not in prompt else "[]")
    report = run_benchmark([backend], [CASE])

    metrics, results = report["placeholder-model"]
    assert results[0].returned_facts == []
    assert results[0].verified_facts == []
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0


def test_run_benchmark_invalid_json_model_scores_zero_validity():
    backend = MockLLMBackend(label="broken", responses=["not json"] * 5)
    report = run_benchmark([backend], [CASE])

    metrics, results = report["broken"]
    assert metrics.json_validity_rate == 0.0
    assert metrics.errors == 0  # invalid JSON isn't a backend error, just an unparseable response
    assert results[0].false_negatives == len(CASE.expected_facts)
    assert metrics.recall == 0.0


def test_run_benchmark_backend_error_counts_gold_facts_as_missed():
    class BrokenBackend(MockLLMBackend):
        def generate(self, *args, **kwargs):
            raise LLMBackendError("simulated timeout")

    report = run_benchmark([BrokenBackend(label="timeout-model")], [CASE])

    metrics, results = report["timeout-model"]
    assert metrics.json_validity_rate == 0.0
    assert metrics.errors == 1
    assert results[0].false_negatives == len(CASE.expected_facts)
    assert metrics.recall == 0.0


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
