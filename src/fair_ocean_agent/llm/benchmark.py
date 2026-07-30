"""Benchmark harness: run every configured candidate model against a fixed
set of gold-standard extraction cases and report comparable metrics --
JSON-validity rate, evidence-ID verification rate, precision/recall/F1
against gold, and latency -- suitable for reporting in a paper.

This module has no opinion about which model is "best"; it only measures.
Gold cases are meant to be curated by Claude now (see GoldCase.curated_by)
and, later, validated/replaced by manual review -- the format doesn't
change either way, only who authored the expected_facts.

Every case is run through the exact same `build_prompt` production code
uses (`extraction/text.py`), not a second, locally-defined prompt template
-- gold cases used to each carry their own free-form `instructions` field,
which meant the benchmark could silently drift from what the real pipeline
actually sends a model. There's now one prompt-construction path; a
change to extraction/text.py's instructions is automatically what the next
benchmark run tests, with no gold file to update in lockstep.
"""
from __future__ import annotations

import csv
import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pydantic import BaseModel

from fair_ocean_agent.dates import try_parse_date
from fair_ocean_agent.extraction.text import (
    EXTRACTION_FOCUSES,
    build_prompt,
    is_absent_raw_value,
    recall_missing_fact_types,
    segment_source_text,
    segments_for_focus,
)
from fair_ocean_agent.llm.base import LLMBackend, LLMBackendError


class GoldFact(BaseModel):
    fact_type_candidate: str
    raw_value: str
    evidence_quote: str


class GoldCase(BaseModel):
    case_id: str
    source_text: str
    section_title: str = "Methods"
    expected_facts: list[GoldFact]
    curated_by: str = "unspecified"
    notes: str | None = None


def load_gold_cases(directory: str | Path) -> list[GoldCase]:
    directory = Path(directory)
    return [GoldCase.model_validate(json.loads(p.read_text())) for p in sorted(directory.glob("*.json"))]


def _normalize(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _normalize_label(value: str) -> str:
    """fact_type_candidate is a category name the model/gold-curator picked
    freely -- "collection_date" and "collection date" are the same label,
    not a disagreement about the fact. Underscores/hyphens are treated as
    interchangeable with spaces before comparing."""
    return _normalize(value.replace("_", " ").replace("-", " "))


def _values_equivalent(a: str, b: str) -> bool:
    """Exact match after normalization, OR both parse as the same calendar
    date (fair_ocean_agent.dates.try_parse_date -- fixed anchor, not
    today's date, so this is deterministic). Section-6-style date
    reformatting (unifying to ISO, etc.) is a later pipeline stage
    (mapping/standardization) -- at the raw-extraction
    benchmark stage, "14 March 2021" and "2021-03-14" report the same fact
    and must not be scored as a disagreement just because of formatting.
    Non-date values (kit names, primer sequences, ...) still require exact
    normalized-string equality -- dateutil's strict mode reliably rejects
    those as unparseable rather than guessing, so this never loosens
    matching for facts that aren't dates."""
    if _normalize(a) == _normalize(b):
        return True
    date_a, date_b = try_parse_date(a), try_parse_date(b)
    return date_a is not None and date_a == date_b


def _score_case(case: GoldCase, verified_facts: list[dict]) -> tuple[int, int, int]:
    """Matches each returned fact against an unmatched gold fact whose
    label is equivalent (underscore/space/case-insensitive) and whose value
    is equivalent (exact-after-normalization, or same calendar date). A
    looser match (e.g. substring/fuzzy on raw_value) would inflate agreement
    scores and defeat the point of comparing models against gold -- this
    only forgives formatting, never content."""
    remaining_expected = list(case.expected_facts)
    true_positives = 0
    for fact in verified_facts:
        got_label = fact.get("fact_type_candidate", "")
        got_value = fact.get("raw_value", "")
        match_index = next(
            (
                i
                for i, exp in enumerate(remaining_expected)
                if _normalize_label(got_label) == _normalize_label(exp.fact_type_candidate)
                and _values_equivalent(got_value, exp.raw_value)
            ),
            None,
        )
        if match_index is not None:
            remaining_expected.pop(match_index)
            true_positives += 1
    false_positives = len(verified_facts) - true_positives
    false_negatives = len(remaining_expected)
    return true_positives, false_positives, false_negatives


def _candidate_evidence_ids(candidate: dict) -> list[str]:
    value = candidate.get("evidence_id", candidate.get("evidence_ids"))
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _facts_with_verified_segment_ids(returned_facts: list[dict], segment_lookup: dict[str, str]) -> list[dict]:
    verified = []
    seen: set[tuple[str, str, str]] = set()
    for fact in returned_facts:
        if is_absent_raw_value(fact.get("raw_value")):
            continue
        evidence_ids = _candidate_evidence_ids(fact)
        if not evidence_ids or any(evidence_id not in segment_lookup for evidence_id in evidence_ids):
            continue
        quote = "\n".join(segment_lookup[evidence_id] for evidence_id in evidence_ids)
        dedupe_key = (
            str(fact.get("fact_type_candidate", "")),
            str(fact.get("raw_value", "")),
            quote,
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        normalized = dict(fact)
        normalized["evidence_quote"] = quote
        verified.append(normalized)
    return verified


def _parsed_facts(parsed) -> list[dict]:
    facts = parsed if isinstance(parsed, list) else (parsed.get("facts", []) if isinstance(parsed, dict) else [])
    return [fact for fact in facts if isinstance(fact, dict) and not is_absent_raw_value(fact.get("raw_value"))]


@dataclass
class CaseResult:
    case_id: str
    candidate_label: str
    json_valid: bool
    latency_seconds: float
    returned_facts: list[dict] = field(default_factory=list)
    verified_facts: list[dict] = field(default_factory=list)
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    error: str | None = None


def run_case(backend: LLMBackend, case: GoldCase) -> CaseResult:
    segments = segment_source_text(case.section_title, case.source_text)
    segment_lookup = {segment.segment_id: segment.text for segment in segments}
    returned_facts: list[dict] = []
    latency_seconds = 0.0
    json_valid = True

    for focus in EXTRACTION_FOCUSES:
        focused_segments = segments_for_focus(case.section_title, segments, focus)
        if not focused_segments:
            continue
        prompt = build_prompt(
            f"{case.section_title} [{focus.name}]",
            case.source_text,
            segments=focused_segments,
            focus=focus,
        )
        try:
            parsed, response = backend.generate_json(prompt, temperature=0)
        except LLMBackendError as exc:
            return CaseResult(
                case_id=case.case_id,
                candidate_label=backend.label,
                json_valid=False,
                latency_seconds=latency_seconds,
                false_negatives=len(case.expected_facts),
                error=str(exc),
            )
        if response is not None:
            latency_seconds += response.latency_seconds
        if parsed is None:
            json_valid = False
            parsed_facts = []
        else:
            parsed_facts = _parsed_facts(parsed)
            returned_facts.extend(parsed_facts)

        verified_first_pass = _facts_with_verified_segment_ids(parsed_facts, segment_lookup)
        accepted_fact_types = {
            str(fact.get("fact_type_candidate"))
            for fact in verified_first_pass
            if fact.get("fact_type_candidate")
        }
        missing_types = recall_missing_fact_types(focus, frozenset(), accepted_fact_types, focused_segments)
        if not missing_types:
            continue
        recall_prompt = build_prompt(
            f"{case.section_title} [{focus.name}] [recall]",
            case.source_text,
            segments=focused_segments,
            focus=focus,
            include_native_names=missing_types,
            recall_pass=True,
        )
        try:
            parsed, response = backend.generate_json(recall_prompt, temperature=0)
        except LLMBackendError:
            continue
        if response is not None:
            latency_seconds += response.latency_seconds
        if parsed is None:
            json_valid = False
            continue
        returned_facts.extend(_parsed_facts(parsed))

    if not json_valid and not returned_facts:
        return CaseResult(
            case_id=case.case_id,
            candidate_label=backend.label,
            json_valid=False,
            latency_seconds=latency_seconds,
            false_negatives=len(case.expected_facts),
        )

    verified_facts = _facts_with_verified_segment_ids(returned_facts, segment_lookup)

    tp, fp, fn = _score_case(case, verified_facts)
    return CaseResult(
        case_id=case.case_id,
        candidate_label=backend.label,
        json_valid=json_valid,
        latency_seconds=latency_seconds,
        returned_facts=returned_facts,
        verified_facts=verified_facts,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
    )


@dataclass
class CandidateMetrics:
    candidate_label: str
    n_cases: int
    json_validity_rate: float
    evidence_verification_rate: float  # verified_facts / returned_facts, across all cases
    precision: float
    recall: float
    f1: float
    mean_latency_seconds: float
    median_latency_seconds: float
    p95_latency_seconds: float
    errors: int


def summarize(results: list[CaseResult]) -> CandidateMetrics:
    n = len(results)
    json_valid = [r for r in results if r.json_valid]
    latencies = [r.latency_seconds for r in json_valid] or [0.0]
    total_returned = sum(len(r.returned_facts) for r in results)
    total_verified = sum(len(r.verified_facts) for r in results)
    tp = sum(r.true_positives for r in results)
    fp = sum(r.false_positives for r in results)
    fn = sum(r.false_negatives for r in results)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    sorted_latencies = sorted(latencies)
    p95_index = max(0, int(len(sorted_latencies) * 0.95) - 1)

    return CandidateMetrics(
        candidate_label=results[0].candidate_label,
        n_cases=n,
        json_validity_rate=len(json_valid) / n if n else 0.0,
        evidence_verification_rate=total_verified / total_returned if total_returned else 0.0,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_latency_seconds=statistics.mean(latencies),
        median_latency_seconds=statistics.median(latencies),
        p95_latency_seconds=sorted_latencies[p95_index],
        errors=sum(1 for r in results if r.error),
    )


def run_benchmark(
    backends: list[LLMBackend], gold_cases: list[GoldCase]
) -> dict[str, tuple[CandidateMetrics, list[CaseResult]]]:
    if not gold_cases:
        raise ValueError("No gold cases provided -- nothing to benchmark against.")
    report: dict[str, tuple[CandidateMetrics, list[CaseResult]]] = {}
    for backend in backends:
        results = [run_case(backend, case) for case in gold_cases]
        report[backend.label] = (summarize(results), results)
    return report


SUMMARY_CSV_FIELDS = (
    "candidate_label",
    "n_cases",
    "json_validity_rate",
    "evidence_verification_rate",
    "precision",
    "recall",
    "f1",
    "mean_latency_seconds",
    "median_latency_seconds",
    "p95_latency_seconds",
    "errors",
)


def export_report(report: dict[str, tuple[CandidateMetrics, list[CaseResult]]], output_dir: str | Path) -> None:
    """Writes summary.csv (one row per candidate -- paste straight into a
    paper's results table) and detail.json (per-case results, for
    debugging disagreements or spot-checking evidence verification)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_CSV_FIELDS)
        writer.writeheader()
        for metrics, _results in report.values():
            writer.writerow({field_name: getattr(metrics, field_name) for field_name in SUMMARY_CSV_FIELDS})

    detail = {
        label: {"metrics": asdict(metrics), "cases": [asdict(r) for r in results]}
        for label, (metrics, results) in report.items()
    }
    (output_dir / "detail.json").write_text(json.dumps(detail, indent=2, default=str))
