#!/usr/bin/env python3
"""Re-map every study to FAIRe after curation (gap 2, 2026-11-11).

Why: MAP_FAIRE for a study with shared samples waits for
CHECK_COMPONENT_SETTLED, and those checks were never claimed, so 4,029
studies were never mapped (data_integrity_audit.md, Finding 4). New mapping
rules (gap-2 spellings) also need every study re-mapped. Run only when
discovery is stopped: with nothing in flight, every component is settled
by definition.

Steps (one transaction per 200 components, so a re-run resumes):
  1. every study component: mark settled + determine roots of shared
     entities (settle_handlers._mark_settled_and_determine_roots);
  2. cancel pending CHECK_COMPONENT_SETTLED tasks (they would re-poll
     forever on leftover discovery tasks and later mark settled studies
     stalled);
  3. queue MAP_FAIRE for every non-merged study with raw facts, with this
     run's label in the payload (a new idempotency key, so studies mapped
     before are mapped again), unless one is already pending.
Then run: python -m fair_ocean_agent.cli worker --task-type MAP_FAIRE --until-empty
(re-mapping deletes and rebuilds a study's values, mapping/faire.py).

Usage: python scripts/remap_all.py [--label LABEL] [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqlalchemy import select, update  # noqa: E402

from fair_ocean_agent.database.enums import TaskStatus, TaskType  # noqa: E402
from fair_ocean_agent.database.models import RawFact, Study, Task  # noqa: E402
from fair_ocean_agent.database.session import session_scope  # noqa: E402
from fair_ocean_agent.identity.component import compute_study_component  # noqa: E402
from fair_ocean_agent.workflow.mapping_handlers import MAPPING_VERSION  # noqa: E402
from fair_ocean_agent.workflow.settle_handlers import _mark_settled_and_determine_roots  # noqa: E402
from fair_ocean_agent.workflow.task_queue import enqueue_task  # noqa: E402


def remap_all(session, label: str, dry_run: bool = False, commit_every: int = 200) -> dict:
    with_facts = set(session.scalars(select(RawFact.study_id).distinct()))
    studies = sorted(
        s for s in session.scalars(select(Study.study_id).where(Study.canonical_status != "merged")) if s in with_facts
    )
    seen: set[str] = set()
    components = 0
    t0 = time.time()
    for sid in studies:
        if sid in seen:
            continue
        component = compute_study_component(session, sid)
        seen |= component
        components += 1
        if not dry_run:
            _mark_settled_and_determine_roots(session, component)
            if components % commit_every == 0:
                session.commit()
                print(f"  settled {components:,} components ({len(seen):,} studies) {time.time() - t0:.0f}s", flush=True)
    cancelled = 0
    if not dry_run:
        session.commit()
        cancelled = session.execute(
            update(Task)
            .where(Task.task_type == TaskType.CHECK_COMPONENT_SETTLED.value, Task.status == TaskStatus.PENDING.value)
            .values(status=TaskStatus.CANCELLED.value)
        ).rowcount
        session.commit()
    pending = set(session.scalars(
        select(Task.study_id).where(Task.task_type == TaskType.MAP_FAIRE.value,
                                    Task.status.in_([TaskStatus.PENDING.value, TaskStatus.RETRY_PENDING.value]))
    ))
    queued = 0
    for sid in studies:
        if sid in pending:
            continue
        if not dry_run:
            enqueue_task(session, TaskType.MAP_FAIRE, study_id=sid,
                         payload={"mapping_version": MAPPING_VERSION, "remap": label})
        queued += 1
    if not dry_run:
        session.commit()
    return {"studies": len(studies), "components": components, "cancelled_settle_checks": cancelled,
            "already_pending_map": len(pending & set(studies)), "queued_map": queued}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default=f"remap-{date.today():%Y-%m-%d}")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    with session_scope() as session:
        print(remap_all(session, args.label, args.dry_run))


if __name__ == "__main__":
    main()
