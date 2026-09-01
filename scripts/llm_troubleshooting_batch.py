#!/usr/bin/env python3
"""Runs a small, explicitly-scoped LLM extraction test batch against an
ISOLATED copy of the main database -- deliberately separate from
enqueue-text-extraction-backfill/enqueue-mapping-backfill (both "queue
this for every eligible study" commands) and from `worker --until-empty`
(which would drain whatever else is sitting in the SAME queue). The whole
point of this script is the opposite: pick a small, fixed set of real
candidate studies, run only their EXTRACT_TEXT_FACTS + MAP_FAIRE, and let
you inspect the result before touching the real pipeline's queue at all.

IMPORTANT: point FAIR_OCEAN_DATABASE_URL at a COPY of the main database
before running this, not the live one -- e.g.:
    sqlite3 data/fair_ocean.db ".backup data/fair_ocean_llm_test.db"
    export FAIR_OCEAN_DATABASE_URL="sqlite:///$(pwd)/data/fair_ocean_llm_test.db"
Plain `cp` is NOT safe against a live, actively-written database (this
pipeline's own established precedent -- SQLite's own .backup command is
the only safe way to copy a file another process may be writing to).

Phases, run in order (see cluster/run_llm_troubleshoot.sbatch for the
full sequence including starting the LLM server):

  select  Picks up to --count real candidate studies (has a DOI, has a
          PMCID -- required for EXTRACT_TEXT_FACTS's Europe PMC fetch --
          and has at least one real SAMPLE entity already resolved, i.e.
          discovery actually found real sequence data to extract facts
          about) that have never had an EXTRACT_TEXT_FACTS task queued
          before. Writes the chosen study_ids to --manifest.

  rediscover  OPTIONAL, and separate from the reset/enqueue/worker/remap/
          report/export loop below: forces DISCOVER_IDENTIFIERS to run
          again for exactly the manifest's study_ids, bypassing
          enqueue_task's own idempotency (a study already discovered once
          is otherwise never revisited). Real gap found live: a fix to
          sources/ncbi.py's own BioProject<->BioSample resolution (or any
          other discovery-side code) is NEVER exercised by reset/enqueue/
          worker/remap alone -- those four only ever touch
          EXTRACT_TEXT_FACTS/MAP_FAIRE, a completely different task type
          from DISCOVER_IDENTIFIERS, so lat_lon/collection_date/depth
          (structured BioSample-attribute facts, not LLM-extracted) stay
          exactly as stale as whatever the study's LAST real discovery
          run found, no matter how many times the LLM-focused loop below
          gets re-run. Follow this with:
              python -m fair_ocean_agent.cli worker --task-type DISCOVER_IDENTIFIERS --until-empty
          (real, live network calls to NCBI/ENA/etc -- no GPU/LLM server
          needed, unlike the worker step below) before `remap`/`report`/
          `export` so the freshly re-discovered structured facts actually
          get mapped and show up.

  reset   Clears out a study's PREVIOUS EXTRACT_TEXT_FACTS output (its
          article_fulltext Source row, the RawFacts it produced, their
          StandardizedValueEvidence links, and every StandardizedValue for
          the study) plus any existing EXTRACT_TEXT_FACTS Task row, so a
          second `enqueue` on the same manifest actually re-runs the LLM
          call instead of enqueue_task's own idempotency key just handing
          back the already-completed task from last time. Only touches
          rows the manifest's own studies own -- their earlier
          structured-source facts (BioProject/BioSample resolution etc.)
          are never touched. Run this before `enqueue` on every iteration
          AFTER the first.

  enqueue Queues EXTRACT_TEXT_FACTS for exactly the manifest's study_ids
          (via workflow.task_queue.enqueue_task directly -- not the
          blanket enqueue-text-extraction-backfill command). Follow this
          with:
              python -m fair_ocean_agent.cli worker --task-type EXTRACT_TEXT_FACTS --until-empty
          which is safe to run --until-empty here specifically because
          this is an isolated copy where nothing else has ever queued an
          EXTRACT_TEXT_FACTS task -- there is nothing else in the queue
          for it to drain.

  remap   Calls mapping.faire.map_study_to_faire directly for exactly the
          manifest's study_ids (same direct-call pattern
          run_extraction.sbatch's own "belt-and-suspenders re-mapping
          sweep" already uses, just scoped to this batch instead of every
          study in the database) -- MAP_FAIRE is never auto-retriggered by
          EXTRACT_TEXT_FACTS finishing (a known, documented pipeline gap),
          so this step is what actually turns the freshly-extracted facts
          into FAIRe StandardizedValues.

  report  Prints, per study in the manifest: its DOI/PMCID/title, the
          RawFacts its most recent EXTRACT_TEXT_FACTS run produced, and
          the resulting StandardizedValues -- the actual thing worth
          reading to judge whether the extraction is working.

  export  Writes the real FAIRe CSVs (exports.faire.export_faire, the
          same format `export-faire` produces for the whole database,
          one CSV per FAIRe class matching the FULLtemplate.xlsx sheet
          layout) scoped to just the manifest's study_ids, to
          --output-dir -- meant to be scp'd back and opened directly,
          rather than read off the .out log.

Usage:
    python scripts/llm_troubleshooting_batch.py select --count 10
    # only needed after a discovery-side (not extraction-side) code change:
    python scripts/llm_troubleshooting_batch.py rediscover
    # ... run: python -m fair_ocean_agent.cli worker --task-type DISCOVER_IDENTIFIERS --until-empty
    python scripts/llm_troubleshooting_batch.py enqueue
    # ... run the worker ...
    python scripts/llm_troubleshooting_batch.py remap
    python scripts/llm_troubleshooting_batch.py report
    python scripts/llm_troubleshooting_batch.py export --output-dir data/exports/llm_troubleshooting_batch
    # tweak extraction code, then re-run without re-selecting:
    python scripts/llm_troubleshooting_batch.py reset
    python scripts/llm_troubleshooting_batch.py enqueue
    # ... run the worker again ...
    python scripts/llm_troubleshooting_batch.py remap
    python scripts/llm_troubleshooting_batch.py report
    python scripts/llm_troubleshooting_batch.py export --output-dir data/exports/llm_troubleshooting_batch
"""
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from sqlalchemy import delete, select

from fair_ocean_agent.database.enums import EntityLevel, IdentifierType, SourceType, TaskType
from fair_ocean_agent.database.models import (
    Entity,
    ExternalIdentifier,
    RawFact,
    Source,
    StandardizedValue,
    StandardizedValueEvidence,
    Study,
    Task,
)
from fair_ocean_agent.database.session import session_scope
from fair_ocean_agent.exports.faire import export_faire
from fair_ocean_agent.mapping.faire import map_study_to_faire
from fair_ocean_agent.workflow.task_queue import enqueue_task

DEFAULT_MANIFEST = Path("data/llm_troubleshooting_manifest.json")
DEFAULT_EXPORT_DIR = Path("data/exports/llm_troubleshooting_batch")


def _load_manifest(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"{path} does not exist -- run the 'select' phase first.")
    return json.loads(path.read_text(encoding="utf-8"))["study_ids"]


def _save_manifest(path: Path, study_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"study_ids": study_ids}, indent=2), encoding="utf-8")


def select_candidates(session, count: int) -> list[str]:
    has_doi = select(ExternalIdentifier.study_id).where(ExternalIdentifier.identifier_type == IdentifierType.DOI.value)
    has_pmcid = select(ExternalIdentifier.study_id).where(ExternalIdentifier.identifier_type == IdentifierType.PMCID.value)
    has_sample = select(Entity.study_id).where(Entity.entity_level == EntityLevel.SAMPLE.value)
    already_queued = select(Task.study_id).where(Task.task_type == TaskType.EXTRACT_TEXT_FACTS.value)

    return list(
        session.scalars(
            select(Study.study_id)
            .where(Study.study_id.in_(has_doi))
            .where(Study.study_id.in_(has_pmcid))
            .where(Study.study_id.in_(has_sample))
            .where(Study.study_id.notin_(already_queued))
            .distinct()
            .limit(count)
        ).all()
    )


def rediscover_study(session, study_id: str) -> None:
    """Forces a real DISCOVER_IDENTIFIERS re-run, bypassing enqueue_task's
    own idempotency (its default key would just hand back the already-
    completed task from the study's original discovery). Mirrors
    scheduling/rediscovery.py::enqueue_full_rediscovery's own
    idempotency-bypass pattern (a fresh, run-scoped key), scoped to one
    study instead of every candidate in the database."""
    enqueue_task(
        session,
        TaskType.DISCOVER_IDENTIFIERS,
        study_id=study_id,
        idempotency_key=f"llm_troubleshoot_rediscover:{study_id}:{uuid.uuid4()}",
    )


def reset_study(session, study_id: str) -> None:
    fulltext_source_ids = list(
        session.scalars(
            select(Source.source_id).where(Source.study_id == study_id, Source.source_type == SourceType.ARTICLE_FULLTEXT.value)
        ).all()
    )
    if fulltext_source_ids:
        fact_ids = list(session.scalars(select(RawFact.fact_id).where(RawFact.source_id.in_(fulltext_source_ids))).all())
        if fact_ids:
            session.execute(delete(StandardizedValueEvidence).where(StandardizedValueEvidence.fact_id.in_(fact_ids)))
            session.execute(delete(RawFact).where(RawFact.fact_id.in_(fact_ids)))
        session.execute(delete(Source).where(Source.source_id.in_(fulltext_source_ids)))
    session.execute(delete(StandardizedValue).where(StandardizedValue.study_id == study_id))
    session.execute(delete(Task).where(Task.task_type == TaskType.EXTRACT_TEXT_FACTS.value, Task.study_id == study_id))


def _identifier(session, study_id: str, identifier_type: IdentifierType) -> str | None:
    return session.scalars(
        select(ExternalIdentifier.identifier_value)
        .where(ExternalIdentifier.study_id == study_id, ExternalIdentifier.identifier_type == identifier_type.value)
    ).first()


def report_study(session, study_id: str) -> None:
    study = session.get(Study, study_id)
    print(f"\n{'=' * 70}\n{study_id}: {study.title if study else '(study not found)'}")
    print(f"  DOI:   {_identifier(session, study_id, IdentifierType.DOI)}")
    print(f"  PMCID: {_identifier(session, study_id, IdentifierType.PMCID)}")

    fulltext_source_ids = set(
        session.scalars(
            select(Source.source_id).where(Source.study_id == study_id, Source.source_type == SourceType.ARTICLE_FULLTEXT.value)
        ).all()
    )
    all_facts = session.scalars(select(RawFact).where(RawFact.study_id == study_id)).all()
    llm_facts = [f for f in all_facts if f.source_id in fulltext_source_ids]
    structured_facts = [f for f in all_facts if f.source_id not in fulltext_source_ids]

    # Printed separately and BEFORE the LLM facts: "why didn't X show up"
    # is often actually "was the structured API data (BioProject/
    # BioSample resolution, run before EXTRACT_TEXT_FACTS and completely
    # unaffected by it) even fetched in the first place" -- a question the
    # old version of this report couldn't answer at all, since it only
    # ever looked at facts from the article_fulltext source.
    print(f"\n  Structured (API-derived) RawFacts ({len(structured_facts)}):")
    for fact in structured_facts:
        print(f"    [{fact.entity_level}] {fact.fact_type_candidate} = {fact.raw_value!r}  (source_locator={fact.source_locator})")

    print(f"\n  RawFacts from EXTRACT_TEXT_FACTS ({len(llm_facts)}):")
    for fact in llm_facts:
        print(f"    [{fact.entity_level}] {fact.fact_type_candidate} = {fact.raw_value!r}  (via {fact.extraction_method})")

    values = session.scalars(select(StandardizedValue).where(StandardizedValue.study_id == study_id)).all()
    print(f"\n  StandardizedValues ({len(values)}):")
    for value in values:
        print(f"    [{value.target_field}] {value.standardized_value!r}  (missingness={value.missingness_status})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=("select", "rediscover", "reset", "enqueue", "remap", "report", "export"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--count", type=int, default=10, help="select phase only: how many studies to pick")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXPORT_DIR, help="export phase only: where to write the FAIRe CSVs")
    args = parser.parse_args()

    with session_scope() as session:
        if args.phase == "select":
            study_ids = select_candidates(session, args.count)
            _save_manifest(args.manifest, study_ids)
            print(f"Selected {len(study_ids)} candidate studies, wrote {args.manifest}:")
            for study_id in study_ids:
                print(f"  {study_id}")
            if len(study_ids) < args.count:
                print(
                    f"\nOnly found {len(study_ids)} studies with a DOI + PMCID + at least one real "
                    "SAMPLE entity that haven't already been queued for extraction -- fewer "
                    "candidates are currently ready than requested."
                )
            return

        study_ids = _load_manifest(args.manifest)

        if args.phase == "rediscover":
            for study_id in study_ids:
                rediscover_study(session, study_id)
            print(
                f"Queued DISCOVER_IDENTIFIERS for {len(study_ids)} studies. Now run:\n"
                "  python -m fair_ocean_agent.cli worker --task-type DISCOVER_IDENTIFIERS --until-empty"
            )
        elif args.phase == "reset":
            for study_id in study_ids:
                reset_study(session, study_id)
            print(f"Reset EXTRACT_TEXT_FACTS output for {len(study_ids)} studies from {args.manifest}.")
        elif args.phase == "enqueue":
            for study_id in study_ids:
                enqueue_task(session, TaskType.EXTRACT_TEXT_FACTS, study_id=study_id)
            print(
                f"Queued EXTRACT_TEXT_FACTS for {len(study_ids)} studies. Now run:\n"
                "  python -m fair_ocean_agent.cli worker --task-type EXTRACT_TEXT_FACTS --until-empty"
            )
        elif args.phase == "remap":
            for study_id in study_ids:
                created = map_study_to_faire(session, study_id)
                print(f"  {study_id}: {created} standardized value(s)")
        elif args.phase == "report":
            for study_id in study_ids:
                report_study(session, study_id)
        elif args.phase == "export":
            counts = export_faire(session, args.output_dir, study_ids=study_ids)
            print(f"Wrote FAIRe CSVs for {len(study_ids)} studies to {args.output_dir}:")
            for class_name, count in counts.items():
                print(f"  {class_name}.csv: {count} row(s)")


if __name__ == "__main__":
    main()
