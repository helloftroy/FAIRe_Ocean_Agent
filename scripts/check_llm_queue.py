#!/usr/bin/env python3
"""Reports how much of the pipeline's task queue is still waiting on an
LLM call, broken down by task type/status, plus how many distinct studies
and papers (DOIs) that backlog actually represents. Read-only against
whatever FAIR_OCEAN_DATABASE_URL points at -- safe to run any time,
including while a worker job is actively running.

Only three task types ever call an LLM -- EXTRACT_TEXT_FACTS,
EXTRACT_SUPPLEMENT_TEXT_FACTS, VALIDATE_EVIDENCE (see cluster/
run_discovery.sbatch's own header: "No LLM calls at all"). That script's
own `worker` calls all pass --no-llm and never enqueue any of the three,
so if IT is the job you're currently running, the "waiting on the LLM"
section below will correctly read zero the whole time -- that's not stuck,
it just hasn't reached the LLM stage yet by design (only
cluster/run_extraction.sbatch enqueues those three task types). The
"every task type" table is what actually tells you whether
run_discovery.sbatch itself is stuck vs. still working through a long,
rate-limited discovery/retrieval queue -- check that table first if a
discovery run has been going for a long time.

"Papers" below means distinct DOI values (a paper can only be counted once
even if, in a not-yet-reconciled sibling-split edge case, two Study rows
briefly point at the same DOI); "studies" means distinct Study rows --
these numbers can differ, and the gap itself is informative.

Usage:
    python scripts/check_llm_queue.py
    python scripts/check_llm_queue.py --json
    FAIR_OCEAN_DATABASE_URL=sqlite:////path/to/other.db python scripts/check_llm_queue.py
"""
from __future__ import annotations

import argparse
import json
from typing import Iterable

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from fair_ocean_agent.database.enums import CanonicalStatus, IdentifierType, TaskStatus, TaskType
from fair_ocean_agent.database.models import ExternalIdentifier, Study, Task
from fair_ocean_agent.database.session import session_scope

LLM_TASK_TYPES = (TaskType.EXTRACT_TEXT_FACTS, TaskType.EXTRACT_SUPPLEMENT_TEXT_FACTS, TaskType.VALIDATE_EVIDENCE)
# "Waiting" = still queued to run or actively retrying, not yet a final
# outcome. FAILED/MANUAL_REVIEW_REQUIRED are reported separately: those
# need a human to look, not more worker time, so folding them into
# "waiting" would overstate how much a worker restart alone will resolve.
WAITING_STATUSES = (TaskStatus.PENDING, TaskStatus.CLAIMED, TaskStatus.RUNNING, TaskStatus.RETRY_PENDING)
STUCK_STATUSES = (TaskStatus.FAILED, TaskStatus.MANUAL_REVIEW_REQUIRED)
DONE_STATUSES = (TaskStatus.COMPLETED, TaskStatus.CANCELLED)


def _counts_by_type_and_status(session: Session) -> dict[str, dict[str, int]]:
    rows = session.execute(
        select(Task.task_type, Task.status, func.count(Task.task_id)).group_by(Task.task_type, Task.status)
    ).all()
    counts: dict[str, dict[str, int]] = {t.value: {s.value: 0 for s in TaskStatus} for t in TaskType}
    for task_type, status, count in rows:
        counts.setdefault(task_type, {s.value: 0 for s in TaskStatus})[status] = count
    return counts


def _study_ids_for(
    session: Session, task_types: Iterable[TaskType], statuses: Iterable[TaskStatus] | None = None
) -> set[str]:
    stmt = select(Task.study_id).where(
        Task.task_type.in_([t.value for t in task_types]),
        Task.study_id.is_not(None),
    )
    if statuses is not None:
        stmt = stmt.where(Task.status.in_([s.value for s in statuses]))
    return {row[0] for row in session.execute(stmt.distinct()).all()}


def _studies_with_doi(session: Session, study_ids: set[str]) -> set[str]:
    if not study_ids:
        return set()
    rows = session.execute(
        select(ExternalIdentifier.study_id)
        .where(
            ExternalIdentifier.study_id.in_(study_ids),
            ExternalIdentifier.identifier_type == IdentifierType.DOI.value,
        )
        .distinct()
    ).all()
    return {row[0] for row in rows}


def _dois_for(session: Session, study_ids: set[str]) -> set[str]:
    if not study_ids:
        return set()
    rows = session.execute(
        select(ExternalIdentifier.identifier_value)
        .where(
            ExternalIdentifier.study_id.in_(study_ids),
            ExternalIdentifier.identifier_type == IdentifierType.DOI.value,
        )
        .distinct()
    ).all()
    return {row[0] for row in rows}


def build_report(session: Session) -> dict:
    total_candidate_studies = session.scalar(
        select(func.count(Study.study_id)).where(Study.canonical_status == CanonicalStatus.CANDIDATE.value)
    ) or 0

    by_type_status = _counts_by_type_and_status(session)

    llm_touched = _study_ids_for(session, LLM_TASK_TYPES)
    llm_waiting = _study_ids_for(session, LLM_TASK_TYPES, WAITING_STATUSES)
    llm_stuck = _study_ids_for(session, LLM_TASK_TYPES, STUCK_STATUSES)
    # "Done" here means every LLM task this study has is in a terminal,
    # non-stuck state -- computed as a residual (touched minus waiting minus
    # stuck) rather than its own positive query, since a study can have
    # e.g. one COMPLETED EXTRACT_TEXT_FACTS row and one still-PENDING
    # VALIDATE_EVIDENCE row at the same time; only the residual definition
    # correctly excludes that study from "done".
    llm_done = llm_touched - llm_waiting - llm_stuck
    llm_not_reached = total_candidate_studies - len(llm_touched)

    waiting_papers = _dois_for(session, llm_waiting)
    waiting_studies_with_doi = _studies_with_doi(session, llm_waiting)
    waiting_studies_without_doi = llm_waiting - waiting_studies_with_doi

    return {
        "total_candidate_studies": total_candidate_studies,
        "by_task_type_and_status": by_type_status,
        "llm_stage": {
            "studies_waiting": len(llm_waiting),
            "papers_waiting": len(waiting_papers),
            "studies_waiting_without_a_doi": len(waiting_studies_without_doi),
            "studies_stuck_needs_review": len(llm_stuck),
            "studies_done": len(llm_done),
            "studies_not_yet_reached_llm_stage": llm_not_reached,
        },
    }


def render_text(report: dict) -> str:
    lines: list[str] = []
    lines.append(f"Total candidate studies: {report['total_candidate_studies']}")
    lines.append("")
    lines.append("Task queue, by type and status (non-zero only):")
    for task_type, statuses in report["by_task_type_and_status"].items():
        non_zero = {status: count for status, count in statuses.items() if count}
        if not non_zero:
            continue
        marker = " *LLM*" if task_type in {t.value for t in LLM_TASK_TYPES} else ""
        parts = ", ".join(f"{status}={count}" for status, count in non_zero.items())
        lines.append(f"  {task_type}{marker}: {parts}")
    lines.append("")
    stage = report["llm_stage"]
    lines.append("LLM stage summary (EXTRACT_TEXT_FACTS / EXTRACT_SUPPLEMENT_TEXT_FACTS / VALIDATE_EVIDENCE):")
    lines.append(f"  Studies waiting (pending/claimed/running/retrying): {stage['studies_waiting']}")
    lines.append(f"  Papers waiting (distinct DOIs among those studies): {stage['papers_waiting']}")
    lines.append(f"  Waiting studies with no DOI at all (repository-only): {stage['studies_waiting_without_a_doi']}")
    lines.append(f"  Studies stuck (failed/needs manual review): {stage['studies_stuck_needs_review']}")
    lines.append(f"  Studies fully done with every LLM task: {stage['studies_done']}")
    lines.append(f"  Studies not yet reached the LLM stage at all: {stage['studies_not_yet_reached_llm_stage']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON instead of a text report")
    args = parser.parse_args()

    with session_scope() as session:
        report = build_report(session)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_text(report))


if __name__ == "__main__":
    main()
