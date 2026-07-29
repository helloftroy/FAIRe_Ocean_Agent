#!/usr/bin/env python3
"""Benchmark configured local LLMs on real Europe PMC full-text papers.

This is not a gold-standard precision/recall benchmark. It measures the
operational signals that matter before launching a large extraction run:
latency, JSON validity, evidence verification, section timeout/error rate,
fact yield, and whether extracted native fact names are mappable to FAIRe.
For LLM-derived facts, evidence verification means the model cited a source
segment ID that Python provided in the prompt; Python owns the literal quote.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from sqlalchemy import create_engine, text

from fair_ocean_agent.config import load_benchmark_candidates, load_config, load_sources_config
from fair_ocean_agent.extraction.sections import select_relevant_sections
from fair_ocean_agent.extraction.faire_fields import all_faire_hints
from fair_ocean_agent.extraction.text import (
    DEFAULT_MAX_SECTION_CHARS_PER_CALL,
    EXTRACTION_FOCUSES,
    build_prompt,
    recall_missing_fact_types,
    segment_source_text,
    segments_for_focus,
    split_segments_for_calls,
)
from fair_ocean_agent.llm.base import LLMBackendError
from fair_ocean_agent.llm.factory import build_benchmark_backend
from fair_ocean_agent.mapping.rules import rules_for
from fair_ocean_agent.sources.base import SourceConfig, SourceRecordNotFoundError
from fair_ocean_agent.sources.europe_pmc import EuropePmcAdapter


@dataclass
class Paper:
    doi: str
    pmcid: str


@dataclass
class SectionResult:
    doi: str
    pmcid: str
    candidate_label: str
    section_title: str
    section_chars: int
    json_valid: bool
    latency_seconds: float
    returned_facts: int
    verified_facts: int
    mappable_facts: int
    unique_target_fields: list[str] = field(default_factory=list)
    invalid_faire_hints: int = 0
    empty_faire_hints: int = 0
    error: str | None = None


def _load_papers(db_url: str, limit: int) -> list[Paper]:
    engine = create_engine(db_url)
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                select doi.identifier_value as doi, pmc.identifier_value as pmcid
                from external_identifiers doi
                join external_identifiers pmc
                  on pmc.study_id = doi.study_id
                 and pmc.identifier_type = 'pmcid'
                where doi.identifier_type = 'doi'
                order by doi.identifier_value
                limit :limit
                """
            ),
            {"limit": limit},
        ).all()
    return [Paper(doi=row.doi, pmcid=row.pmcid) for row in rows]


def _build_europe_pmc_adapter() -> EuropePmcAdapter:
    retrieval_config = load_config().retrieval
    sources_config = load_sources_config()
    entry = sources_config["europe_pmc"]
    return EuropePmcAdapter(
        SourceConfig(
            name="europe_pmc",
            enabled=entry.enabled,
            base_url=entry.base_url,
            rate_limit_per_second=entry.rate_limit_per_second,
            priority=entry.priority,
        ),
        retrieval_config,
    )


def _extract_sections(adapter: EuropePmcAdapter, papers: list[Paper], max_chars: int) -> dict[str, list[dict]]:
    by_pmcid: dict[str, list[dict]] = {}
    for paper in papers:
        try:
            xml = adapter.fetch_fulltext_xml(paper.pmcid)
        except SourceRecordNotFoundError:
            by_pmcid[paper.pmcid] = []
            continue
        by_pmcid[paper.pmcid] = select_relevant_sections(xml, max_chars=max_chars)
    return by_pmcid


def _try_parse_json(text_value: str):
    text_value = text_value.strip()
    if text_value.startswith("```"):
        lines = text_value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text_value = "\n".join(lines).strip()
    try:
        return json.loads(text_value)
    except (json.JSONDecodeError, ValueError):
        return None


def _candidate_evidence_ids(candidate: dict) -> list[str]:
    value = candidate.get("evidence_id", candidate.get("evidence_ids"))
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _score_facts(parsed, segment_lookup: dict[str, str]) -> tuple[int, int, int, list[str], int, int]:
    returned = parsed if isinstance(parsed, list) else (parsed.get("facts", []) if isinstance(parsed, dict) else [])
    returned = [fact for fact in returned if isinstance(fact, dict)]
    valid_hints = all_faire_hints()
    verified = []
    seen: set[tuple[str, str, str]] = set()
    invalid_hints = 0
    empty_hints = 0
    target_fields: set[str] = set()
    mappable_facts = 0

    for fact in returned:
        hints = fact.get("candidate_standard_fields")
        if hints == {}:
            empty_hints += 1
        if isinstance(hints, dict) and hints.get("faire") and hints["faire"] not in valid_hints:
            invalid_hints += 1

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
        verified.append(fact)
        rules = rules_for(str(fact.get("fact_type_candidate", "")), "study")
        if rules:
            mappable_facts += 1
            target_fields.update(f"{rule.target_table}.{rule.target_field}" for rule in rules)

    return len(returned), len(verified), mappable_facts, sorted(target_fields), invalid_hints, empty_hints


def _parsed_facts(parsed) -> list[dict]:
    returned = parsed if isinstance(parsed, list) else (parsed.get("facts", []) if isinstance(parsed, dict) else [])
    return [fact for fact in returned if isinstance(fact, dict)]


def _result_key(result: SectionResult) -> tuple[str, str, str]:
    return (result.candidate_label, result.pmcid, result.section_title)


def _csv_row(result: SectionResult) -> dict:
    row = asdict(result)
    row["unique_target_fields"] = "|".join(result.unique_target_fields)
    return row


def _read_existing_results(path: Path) -> list[SectionResult]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    results = []
    for row in rows:
        results.append(
            SectionResult(
                doi=row["doi"],
                pmcid=row["pmcid"],
                candidate_label=row["candidate_label"],
                section_title=row["section_title"],
                section_chars=int(row["section_chars"]),
                json_valid=row["json_valid"] == "True",
                latency_seconds=float(row["latency_seconds"]),
                returned_facts=int(row["returned_facts"]),
                verified_facts=int(row["verified_facts"]),
                mappable_facts=int(row["mappable_facts"]),
                unique_target_fields=[v for v in row["unique_target_fields"].split("|") if v],
                invalid_faire_hints=int(row["invalid_faire_hints"]),
                empty_faire_hints=int(row["empty_faire_hints"]),
                error=row["error"] or None,
            )
        )
    return results


def _append_result(output_dir: Path, result: SectionResult) -> None:
    csv_path = output_dir / "section_results.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(SectionResult.__dataclass_fields__))
        if write_header:
            writer.writeheader()
        writer.writerow(_csv_row(result))

    with (output_dir / "detail.jsonl").open("a") as f:
        f.write(json.dumps(asdict(result), default=str) + "\n")


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    papers = _load_papers(args.db_url, args.papers)
    adapter = _build_europe_pmc_adapter()
    try:
        sections_by_pmcid = _extract_sections(adapter, papers, args.max_chars)
    finally:
        adapter.close()

    candidates = load_benchmark_candidates()
    if args.models:
        wanted = set(args.models)
        candidates = [candidate for candidate in candidates if candidate.model in wanted or candidate.label in wanted]

    _write_papers(output_dir, papers, sections_by_pmcid)
    if not args.resume:
        for path in (output_dir / "section_results.csv", output_dir / "detail.jsonl", output_dir / "detail.json", output_dir / "summary.csv"):
            if path.exists():
                path.unlink()

    section_results: list[SectionResult] = _read_existing_results(output_dir / "section_results.csv") if args.resume else []
    completed = {_result_key(result) for result in section_results}
    if section_results:
        _write_summary(output_dir, section_results)

    backends = []
    try:
        for candidate in candidates:
            backend = build_benchmark_backend(candidate)
            backends.append(backend)
            for paper in papers:
                sections = sections_by_pmcid.get(paper.pmcid, [])[: args.sections_per_paper]
                if not sections:
                    result = SectionResult(
                        doi=paper.doi,
                        pmcid=paper.pmcid,
                        candidate_label=backend.label,
                        section_title="<no fulltext sections>",
                        section_chars=0,
                        json_valid=False,
                        latency_seconds=0.0,
                        returned_facts=0,
                        verified_facts=0,
                        mappable_facts=0,
                        error="no open-access full text sections selected",
                    )
                    if _result_key(result) not in completed:
                        section_results.append(result)
                        completed.add(_result_key(result))
                        _append_result(output_dir, result)
                        _write_summary(output_dir, section_results)
                    continue
                for section in sections:
                    key = (backend.label, paper.pmcid, section["title"])
                    if key in completed:
                        continue
                    start = time.monotonic()
                    parsed_chunks = []
                    chunk_error = None
                    try:
                        segments = segment_source_text(section["title"], section["text"])
                        segment_lookup = {segment.segment_id: segment.text for segment in segments}
                        chunks = split_segments_for_calls(segments, args.extraction_max_chars_per_call)
                        response = None
                        for index, chunk_segments in enumerate(chunks):
                            chunk_title = section["title"] if len(chunks) == 1 else f"{section['title']} [chunk {index + 1}/{len(chunks)}]"
                            for focus in EXTRACTION_FOCUSES:
                                focused_segments = segments_for_focus(section["title"], chunk_segments, focus)
                                if not focused_segments:
                                    continue
                                prompt = build_prompt(
                                    f"{chunk_title} [{focus.name}]",
                                    "",
                                    segments=focused_segments,
                                    focus=focus,
                                )
                                response = backend.generate(prompt, temperature=0)
                                parsed = _try_parse_json(response.text)
                                if parsed is None:
                                    chunk_error = f"invalid JSON in chunk {index + 1}/{len(chunks)} focus {focus.name}"
                                    break
                                returned = _parsed_facts(parsed)
                                parsed_chunks.extend(returned)
                                verified_first_pass = [
                                    fact
                                    for fact in returned
                                    if _candidate_evidence_ids(fact)
                                    and all(evidence_id in segment_lookup for evidence_id in _candidate_evidence_ids(fact))
                                ]
                                accepted_fact_types = {
                                    str(fact.get("fact_type_candidate"))
                                    for fact in verified_first_pass
                                    if fact.get("fact_type_candidate")
                                }
                                missing_types = recall_missing_fact_types(
                                    focus,
                                    frozenset(),
                                    accepted_fact_types,
                                    focused_segments,
                                )
                                if not missing_types:
                                    continue
                                recall_prompt = build_prompt(
                                    f"{chunk_title} [{focus.name}] [recall]",
                                    "",
                                    segments=focused_segments,
                                    focus=focus,
                                    include_native_names=missing_types,
                                    recall_pass=True,
                                )
                                response = backend.generate(recall_prompt, temperature=0)
                                parsed = _try_parse_json(response.text)
                                if parsed is None:
                                    chunk_error = f"invalid JSON in chunk {index + 1}/{len(chunks)} focus {focus.name} recall"
                                    break
                                parsed_chunks.extend(_parsed_facts(parsed))
                            if chunk_error is not None:
                                break
                    except LLMBackendError as exc:
                        result = SectionResult(
                            doi=paper.doi,
                            pmcid=paper.pmcid,
                            candidate_label=backend.label,
                            section_title=section["title"],
                            section_chars=len(section["text"]),
                            json_valid=False,
                            latency_seconds=time.monotonic() - start,
                            returned_facts=0,
                            verified_facts=0,
                            mappable_facts=0,
                            error=str(exc),
                        )
                        section_results.append(result)
                        completed.add(_result_key(result))
                        _append_result(output_dir, result)
                        _write_summary(output_dir, section_results)
                        continue

                    if chunk_error is not None:
                        result = SectionResult(
                            doi=paper.doi,
                            pmcid=paper.pmcid,
                            candidate_label=backend.label,
                            section_title=section["title"],
                            section_chars=len(section["text"]),
                            json_valid=False,
                            latency_seconds=time.monotonic() - start,
                            returned_facts=0,
                            verified_facts=0,
                            mappable_facts=0,
                            error=chunk_error,
                        )
                        section_results.append(result)
                        completed.add(_result_key(result))
                        _append_result(output_dir, result)
                        _write_summary(output_dir, section_results)
                        continue

                    returned, verified, mappable, targets, invalid_hints, empty_hints = _score_facts(
                        parsed_chunks, segment_lookup
                    )
                    result = SectionResult(
                        doi=paper.doi,
                        pmcid=paper.pmcid,
                        candidate_label=backend.label,
                        section_title=section["title"],
                        section_chars=len(section["text"]),
                        json_valid=True,
                        latency_seconds=time.monotonic() - start,
                        returned_facts=returned,
                        verified_facts=verified,
                        mappable_facts=mappable,
                        unique_target_fields=targets,
                        invalid_faire_hints=invalid_hints,
                        empty_faire_hints=empty_hints,
                    )
                    section_results.append(result)
                    completed.add(_result_key(result))
                    _append_result(output_dir, result)
                    _write_summary(output_dir, section_results)
    finally:
        for backend in backends:
            backend.close()

    _write_outputs(output_dir, papers, sections_by_pmcid, section_results)


def _write_papers(output_dir: Path, papers: list[Paper], sections_by_pmcid: dict[str, list[dict]]) -> None:
    (output_dir / "papers.json").write_text(
        json.dumps(
            [
                {
                    "doi": paper.doi,
                    "pmcid": paper.pmcid,
                    "selected_sections": [
                        {"title": section["title"], "chars": len(section["text"])}
                        for section in sections_by_pmcid.get(paper.pmcid, [])
                    ],
                }
                for paper in papers
            ],
            indent=2,
        )
    )


def _write_outputs(
    output_dir: Path,
    papers: list[Paper],
    sections_by_pmcid: dict[str, list[dict]],
    section_results: list[SectionResult],
) -> None:
    _write_papers(output_dir, papers, sections_by_pmcid)
    (output_dir / "detail.json").write_text(json.dumps([asdict(result) for result in section_results], indent=2))

    with (output_dir / "section_results.csv").open("w", newline="") as f:
        fieldnames = list(SectionResult.__dataclass_fields__)
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in section_results:
            writer.writerow(_csv_row(result))

    _write_summary(output_dir, section_results)


def _write_summary(output_dir: Path, section_results: list[SectionResult]) -> None:
    by_candidate: dict[str, list[SectionResult]] = {}
    for result in section_results:
        by_candidate.setdefault(result.candidate_label, []).append(result)

    summary_fields = (
        "candidate_label",
        "papers",
        "sections",
        "json_validity_rate",
        "evidence_verification_rate",
        "returned_facts",
        "verified_facts",
        "mappable_facts",
        "unique_target_fields",
        "invalid_faire_hints",
        "empty_faire_hints",
        "errors",
        "mean_latency_seconds",
        "median_latency_seconds",
        "p95_latency_seconds",
        "facts_per_minute",
    )
    with (output_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for candidate, results in by_candidate.items():
            latencies = [result.latency_seconds for result in results if result.json_valid]
            sorted_latencies = sorted(latencies)
            p95_index = max(0, int(len(sorted_latencies) * 0.95) - 1)
            returned = sum(result.returned_facts for result in results)
            verified = sum(result.verified_facts for result in results)
            latency_total = sum(result.latency_seconds for result in results)
            target_fields = {
                field
                for result in results
                for field in result.unique_target_fields
            }
            writer.writerow(
                {
                    "candidate_label": candidate,
                    "papers": len({result.pmcid for result in results}),
                    "sections": len(results),
                    "json_validity_rate": sum(1 for result in results if result.json_valid) / len(results),
                    "evidence_verification_rate": verified / returned if returned else 0.0,
                    "returned_facts": returned,
                    "verified_facts": verified,
                    "mappable_facts": sum(result.mappable_facts for result in results),
                    "unique_target_fields": len(target_fields),
                    "invalid_faire_hints": sum(result.invalid_faire_hints for result in results),
                    "empty_faire_hints": sum(result.empty_faire_hints for result in results),
                    "errors": sum(1 for result in results if result.error),
                    "mean_latency_seconds": sum(latencies) / len(latencies) if latencies else 0.0,
                    "median_latency_seconds": sorted_latencies[len(sorted_latencies) // 2] if sorted_latencies else 0.0,
                    "p95_latency_seconds": sorted_latencies[p95_index] if sorted_latencies else 0.0,
                    "facts_per_minute": verified / (latency_total / 60) if latency_total else 0.0,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-url", default=os.environ.get("FAIR_OCEAN_DATABASE_URL", "sqlite:///data/fair_ocean.db"))
    parser.add_argument("--papers", type=int, default=10)
    parser.add_argument("--sections-per-paper", type=int, default=3)
    parser.add_argument("--max-chars", type=int, default=20000)
    parser.add_argument("--extraction-max-chars-per-call", type=int, default=DEFAULT_MAX_SECTION_CHARS_PER_CALL)
    parser.add_argument("--output", default="data/exports/benchmark/real_papers_10")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--resume", action="store_true", help="Skip section results already present in section_results.csv")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
