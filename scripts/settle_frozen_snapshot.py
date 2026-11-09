#!/usr/bin/env python3
"""Settle every study in a FROZEN COPY of the database, so all of them get
FAIRe-mapped.

A study with sample/run entities is only mapped once its connected
component settles: discovery has stopped adding links to it, and root
determination has decided which study owns each shared entity
(workflow/settle_handlers.py). In a snapshot copied off the cluster,
discovery will never run again, so the settle checks still waiting there
would wait forever and those studies would never be mapped. This runs
the settle step (_mark_settled_and_determine_roots) directly for every
component that still has an unmapped candidate study, which also queues
MAP_FAIRE for its studies. Then run:

    python -m fair_ocean_agent.cli worker --task-type MAP_FAIRE --until-empty

Never run this against the live database: it declares components
settled while discovery may still be adding to them. It refuses to run
without --frozen-copy.

Usage:
    FAIR_OCEAN_DATABASE_URL=sqlite:///$(pwd)/data/snapshots/<copy>.db \\
        python scripts/settle_frozen_snapshot.py --frozen-copy
"""
from __future__ import annotations

import argparse
import sys
import time

from sqlalchemy import exists, select

from fair_ocean_agent.config import load_config
from fair_ocean_agent.database.enums import CanonicalStatus
from fair_ocean_agent.database.models import RawFact, StandardizedValue, Study
from fair_ocean_agent.database.session import session_scope
from fair_ocean_agent.identity.component import compute_study_component
from fair_ocean_agent.mapping.faire import TARGET_SCHEMA
from fair_ocean_agent.workflow.settle_handlers import _mark_settled_and_determine_roots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frozen-copy", action="store_true", help="Confirm the database is a frozen copy, not live")
    parser.add_argument("--limit", type=int, default=None, help="Settle at most this many components (for a test)")
    args = parser.parse_args()
    if not args.frozen_copy:
        sys.exit("Refusing to run without --frozen-copy (see this script's docstring).")
    print(f"Database: {load_config().database.url}")

    with session_scope() as session:
        unmapped = list(
            session.scalars(
                select(Study.study_id).where(
                    Study.canonical_status == CanonicalStatus.CANDIDATE.value,
                    exists().where(RawFact.study_id == Study.study_id),
                    ~exists().where(
                        StandardizedValue.study_id == Study.study_id,
                        StandardizedValue.target_schema == TARGET_SCHEMA,
                    ),
                )
            )
        )
    print(f"{len(unmapped)} candidate studies have facts but no FAIRe values.")

    covered: set[str] = set()
    components = 0
    start = time.monotonic()
    for study_id in unmapped:
        if study_id in covered:
            continue
        if args.limit is not None and components >= args.limit:
            break
        component_start = time.monotonic()
        with session_scope() as session:
            component = compute_study_component(session, study_id)
            _mark_settled_and_determine_roots(session, component)
        covered |= component
        components += 1
        print(
            f"  component {components}: {len(component)} stud(y/ies) settled "
            f"({time.monotonic() - component_start:.1f}s; {len(covered & set(unmapped))}/{len(unmapped)} "
            f"unmapped studies covered, {(time.monotonic() - start) / 60:.1f}m total)",
            flush=True,
        )
    print(f"Done: settled {components} component(s). Now run the MAP_FAIRE worker.")


if __name__ == "__main__":
    main()
