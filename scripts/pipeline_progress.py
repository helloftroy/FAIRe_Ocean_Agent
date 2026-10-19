#!/usr/bin/env python3
"""One-screen answer to "is discovery done, and how much is ready for the
LLM?" -- read-only, safe to run while a discovery or extraction job is
running.

Discovery section: tasks still waiting per discovery stage, and a single
verdict line. DISCOVER_CITING_STUDIES is reported apart from the rest
and left out of the verdict while discovery.citation_expansion_enabled
is off (FAIR_OCEAN_CITATION_EXPANSION_ENABLED=0) -- those tasks are
paused on purpose, not unfinished work.

LLM section: candidate studies with readable full text (a PMCID, or a
PDF in FAIR_OCEAN_LOCAL_PDF_DIR the extraction handler will find --
see workflow/handlers.py's studies_with_local_pdf), split by where their
EXTRACT_TEXT_FACTS task stands. Set FAIR_OCEAN_LOCAL_PDF_DIR exactly as
the extraction jobs will, or PDF-only papers are not counted.

Usage:
    FAIR_OCEAN_LOCAL_PDF_DIR=$(pwd)/data/local_pdfs python scripts/pipeline_progress.py
"""
from __future__ import annotations

from collections import Counter

from sqlalchemy import select

from fair_ocean_agent.config import load_config
from fair_ocean_agent.database.enums import CanonicalStatus, IdentifierType, TaskStatus, TaskType
from fair_ocean_agent.database.models import ExternalIdentifier, Study, Task
from fair_ocean_agent.database.session import session_scope
from fair_ocean_agent.workflow.handlers import _pdf_lookup_dir, studies_with_local_pdf

WAITING = {TaskStatus.PENDING.value, TaskStatus.RETRY_PENDING.value, TaskStatus.CLAIMED.value, TaskStatus.RUNNING.value}
NEEDS_REVIEW = {TaskStatus.MANUAL_REVIEW_REQUIRED.value, TaskStatus.FAILED.value}

DISCOVERY_STAGES = (
    (TaskType.DISCOVER_IDENTIFIERS, "identifier discovery"),
    (TaskType.CHECK_COMPONENT_SETTLED, "settle checks"),
    (TaskType.DISCOVER_SUPPLEMENTS, "supplement discovery"),
    (TaskType.RETRIEVE_SUPPLEMENTS, "supplement retrieval"),
    (TaskType.DISCOVER_PRIMER_REFERENCE_STUDIES, "primer-reference papers"),
)


def _bucket(status: str) -> str:
    if status in WAITING:
        return "waiting"
    if status in NEEDS_REVIEW:
        return "needs review"
    return "done"


def main() -> None:
    citation_expansion_enabled = load_config().discovery.citation_expansion_enabled
    with session_scope() as session:
        candidate_ids = set(
            session.scalars(select(Study.study_id).where(Study.canonical_status == CanonicalStatus.CANDIDATE.value))
        )

        counts: dict[str, Counter] = {}
        for task_type, study_id, status in session.execute(select(Task.task_type, Task.study_id, Task.status)):
            if study_id is not None and study_id not in candidate_ids:
                continue
            counts.setdefault(task_type, Counter())[_bucket(status)] += 1

        resolved_ids = set(
            session.scalars(
                select(Task.study_id).where(
                    Task.task_type == TaskType.DISCOVER_IDENTIFIERS.value,
                    Task.status == TaskStatus.COMPLETED.value,
                )
            )
        )

        pmcid_ids = set(
            session.scalars(
                select(ExternalIdentifier.study_id).where(ExternalIdentifier.identifier_type == IdentifierType.PMCID.value)
            )
        ) & candidate_ids
        pdf_ids = studies_with_local_pdf(session) & candidate_ids

        extraction_status: dict[str, str] = {}
        rank = {"done": 3, "waiting": 2, "needs review": 1}
        for study_id, status in session.execute(
            select(Task.study_id, Task.status).where(Task.task_type == TaskType.EXTRACT_TEXT_FACTS.value)
        ):
            bucket = "done" if status == TaskStatus.COMPLETED.value else _bucket(status)
            if bucket == "done" and status != TaskStatus.COMPLETED.value:
                continue  # cancelled: not extracted
            current = extraction_status.get(study_id)
            if current is None or rank[bucket] > rank[current]:
                extraction_status[study_id] = bucket

    print("== Discovery ==")
    print(f"candidate studies:                     {len(candidate_ids)}")
    print(f"  never through identifier discovery:  {len(candidate_ids - resolved_ids)}")
    print(f"\n{'stage':28s} {'waiting':>9s} {'needs review':>13s} {'done':>9s}")
    discovery_waiting = 0
    for task_type, label in DISCOVERY_STAGES:
        c = counts.get(task_type.value, Counter())
        discovery_waiting += c["waiting"]
        print(f"{label:28s} {c['waiting']:9d} {c['needs review']:13d} {c['done']:9d}")
    citing = counts.get(TaskType.DISCOVER_CITING_STUDIES.value, Counter())
    citing_label = "citation lookups" + ("" if citation_expansion_enabled else " (PAUSED)")
    print(f"{citing_label:28s} {citing['waiting']:9d} {citing['needs review']:13d} {citing['done']:9d}")
    if citation_expansion_enabled:
        discovery_waiting += citing["waiting"]

    if discovery_waiting == 0:
        print("\nDiscovery is DONE" + ("" if citation_expansion_enabled else " (citation lookups paused)") + ".")
    else:
        print(f"\nDiscovery is NOT done: {discovery_waiting} task(s) still waiting.")
    if not citation_expansion_enabled:
        print("(Citation lookups counted as paused because FAIR_OCEAN_CITATION_EXPANSION_ENABLED is off here.)")
    elif citing["waiting"]:
        print(
            "(Citation lookups count toward this. Run with FAIR_OCEAN_CITATION_EXPANSION_ENABLED=0 "
            "if the discovery job is running with them paused.)"
        )

    fulltext_ids = pmcid_ids | pdf_ids
    by_status = Counter(extraction_status.get(study_id, "not queued") for study_id in fulltext_ids)
    print("\n== LLM extraction ==")
    print(f"PDF directory checked: {_pdf_lookup_dir()}")
    print(f"studies with readable full text:       {len(fulltext_ids)}")
    print(f"  via PMCID:                           {len(pmcid_ids)}")
    print(f"  via local PDF:                       {len(pdf_ids)} ({len(pdf_ids - pmcid_ids)} have no PMCID)")
    print(f"studies with no full text yet:         {len(candidate_ids - fulltext_ids)}")
    print()
    print(f"  extracted (LLM done):                {by_status['done']}")
    print(f"  queued, waiting for the LLM:         {by_status['waiting']}")
    print(f"  needs review (out of retries):       {by_status['needs review']}")
    print(f"  ready but not queued yet:            {by_status['not queued']}")
    ready = by_status["waiting"] + by_status["not queued"]
    print(f"\nReady for the next LLM run: {ready}")
    if by_status["not queued"]:
        print(
            "shard_extraction_prep.py queues the 'not queued' ones itself -- run it with this same "
            "FAIR_OCEAN_LOCAL_PDF_DIR."
        )


if __name__ == "__main__":
    main()
