#!/usr/bin/env python3
"""Fill BioSamples and runs past the discovery caps, for kept projects only
(2026-11-11). Run AFTER curation (scripts/apply_curation.py --apply) and
BEFORE the re-map (scripts/remap_all.py). Needs the internet (service).

Why: discovery keeps the first MAX_SAMPLES_PER_PROJECT (300) BioSamples per
BioProject (sources/ncbi.py) and the first MAX_RUNS_PER_STUDY (500) runs per
ENA study (sources/ena.py), leaving a coverage note. The user wants the rest
for the projects that survive curation (data_integrity_audit.md).

Targets (from the database, after curation):
  NCBI  (study, BioProject) with a biosample_coverage_note on its
        ncbi_biosample source
  ENA   (study, ENA project) with a run_coverage_note on its ena source
  minus projects curation dropped for that study (curation_exclusions),
  studies flagged not_relevant, merged studies, and projects no longer
  attached to the study.

NCBI: resolve the BioProject UID (skipped if ambiguous), link to every
BioSample UID, efetch by UID (never by accession string: the SAMEA/SAMD
bug), keep only accessions not in the database, not deleted by curation
(tombstones), and not non-marine (marine_audit.classify_sample). Stored like
discovery: a new ncbi_biosample Source per chunk + facts and entities.
ENA: list every run of the study (no limit), keep runs not in the database
whose sample is in the database (existing or just filled; never re-creates a
deleted or skipped sample), store them with fastq_access_status not_checked;
project-level facts are not repeated.

Resumable: re-running skips what is already in the database. Commits per
chunk. Dry run (--dry-run) lists targets and counts only.

Usage: python scripts/fill_capped_projects.py [--dry-run] [--only ncbi|ena] [--max-projects N] [--study STUDY]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sqlalchemy import text  # noqa: E402

from curation_lib import KEPT_KINDS, project_aliases  # noqa: E402
from fair_ocean_agent.clock import utcnow  # noqa: E402
from fair_ocean_agent.database.enums import (  # noqa: E402
    AccessStatus,
    EntityLevel,
    InspectionLevel,
    InspectionStatus,
    ReviewStatus,
    SourceType,
)
from fair_ocean_agent.database.models import RawFact, Source  # noqa: E402
from fair_ocean_agent.database.session import session_scope  # noqa: E402
from fair_ocean_agent.identity.source_linking import create_source  # noqa: E402
from fair_ocean_agent.sources.base import SourceRecord  # noqa: E402
from fair_ocean_agent.sources.ena import RUN_FIELDS, STUDY_FIELDS  # noqa: E402
from fair_ocean_agent.sources.ncbi import _elink_ids, _esearch_verified_uid  # noqa: E402
from fair_ocean_agent.workflow.handlers import _build_enabled_adapters, _materialize_candidate_entity  # noqa: E402
from marine_audit import classify_sample  # noqa: E402

CHUNK = 500


def _targets(session, note: str, source_name: str, aliases: dict[str, str], study: str | None):
    """[(study, project key, original source_id)] for kept capped projects."""
    rows = session.execute(text(
        "select f.study_id, src.external_identifier, src.source_id, f.raw_value from raw_facts f "
        "indexed by ix_raw_facts_fact_type_candidate join sources src on src.source_id = f.source_id "
        "where f.fact_type_candidate = :note and src.source_name = :src"
    ), {"note": note, "src": source_name}).all()
    bad_studies = {s for (s,) in session.execute(text(
        "select study_id from studies where canonical_status = 'merged' or marine_relevance_status = 'not_relevant'"))}
    dropped = set()
    has_table = session.execute(text(
        "select 1 from sqlite_master where type='table' and name='curation_exclusions'")).first()
    if has_table:
        dropped = {(s, aliases.get(a, a)) for s, a in session.execute(text(
            "select study_id, accession from curation_exclusions where level = 'bioproject'"))}
    attached = collections.defaultdict(set)
    for s, v in session.execute(text(
        "select study_id, identifier_value from external_identifiers where identifier_type in "
        "('bioproject_accession','sra_study_accession','ena_study_accession')"
    )):
        attached[s].add(aliases.get(v, v))
    out, seen = [], set()
    for s, ident, source_id, note_text in rows:
        key = aliases.get(ident, ident)
        if (s, ident) in seen or s in bad_studies or (s, key) in dropped or key not in attached[s]:
            continue
        if study and s != study:
            continue
        seen.add((s, ident))
        m = re.search(r"of (\d+)", note_text or "")
        out.append((s, ident, source_id, int(m.group(1)) if m else 0))
    return out


def _present(session, level: str) -> set[str]:
    return {x for (x,) in session.execute(text(
        "select external_identifier from entities where entity_level = :lvl and external_identifier is not null"),
        {"lvl": level})}


def _tombstones(session) -> set[str]:
    if not session.execute(text("select 1 from sqlite_master where type='table' and name='curation_exclusions'")).first():
        return set()
    return {a for (a,) in session.execute(text(
        "select accession from curation_exclusions where action = 'deleted' and accession is not null"))}


def _persist(session, adapter, study_id: str, ident: str, parent_id: str, raw: dict, skip_project_facts: bool) -> int:
    record = SourceRecord(source_name=adapter.name, external_identifier=ident, url=None, raw=raw,
                          retrieved_at=utcnow(),
                          content_hash=hashlib.sha256(json.dumps(raw, sort_keys=True, default=str).encode()).hexdigest())
    source = create_source(session, Source(
        study_id=study_id, source_type=SourceType.REPOSITORY_API.value, source_name=adapter.name,
        external_identifier=ident, url=None, access_status=AccessStatus.UNKNOWN.value,
        retrieved_at=record.retrieved_at, content_hash=record.content_hash,
        inspection_status=InspectionStatus.INSPECTED.value, inspection_level=InspectionLevel.METADATA_ONLY.value,
        parent_source_id=parent_id,
    ))
    n = 0
    for fact in adapter.extract_structured_facts(record):
        if skip_project_facts and fact.entity_level == EntityLevel.PROJECT:
            continue
        entity = _materialize_candidate_entity(session, study_id, fact)
        session.add(RawFact(
            study_id=study_id, entity_id=entity.entity_id if entity is not None else None, source_id=source.source_id,
            source_locator=fact.source_locator, raw_field_name=fact.raw_field_name, raw_value=fact.raw_value,
            fact_type_candidate=fact.fact_type_candidate, entity_level=fact.entity_level.value,
            support_type=fact.support_type.value, extraction_method=f"adapter:{adapter.name}",
            review_status=fact.review_status or ReviewStatus.ACCEPTED.value, confidence_metadata=fact.confidence_metadata,
        ))
        n += 1
    return n


def _sample_fields(sample: dict) -> dict[str, str]:
    attrs = sample.get("attributes") or {}
    organism = sample.get("organism") or {}
    return {
        "organism": organism.get("taxonomy_name") or "",
        "host": attrs.get("host") or attrs.get("host_scientific_name") or "",
        "isolation_source": attrs.get("isolation_source") or "",
        "env_broad_scale": attrs.get("env_broad_scale") or attrs.get("env_biome") or "",
    }


def fill_ncbi(args, aliases) -> None:
    adapter = _build_enabled_adapters()["ncbi_biosample"]
    adapter.http._cache_enabled = False
    with session_scope() as session:
        targets = _targets(session, "biosample_coverage_note", "ncbi_biosample", aliases, args.study)
        present, dead = _present(session, "sample"), _tombstones(session)
    print(f"NCBI: {len(targets):,} kept capped (study, BioProject) targets; up to "
          f"{sum(max(t - 300, 0) for *_, t in targets):,} BioSamples past the cap (before the marine filter "
          f"and before removing ones already present)", flush=True)
    if args.dry_run:
        return
    totals = collections.Counter()
    t0 = time.time()
    for i, (study_id, bp, parent, _total) in enumerate(targets[: args.max_projects or None], 1):
        res = _esearch_verified_uid(adapter.http, adapter.config.base_url, "bioproject", bp, expected_accession=bp)
        if res is None or res.ambiguous:
            totals["skipped_ambiguous_uid"] += 1
            print(f"  [{i}] {bp}: no unambiguous BioProject UID, skipped", flush=True)
            continue
        uids = _elink_ids(adapter.http, adapter.config.base_url, "bioproject", "biosample", res.uid)
        added = 0
        for j in range(0, len(uids), CHUNK):
            try:
                samples, _ = adapter._fetch_and_parse_biosamples(uids[j : j + CHUNK])
                keep = []
                for smp in samples:
                    acc = smp.get("accession")
                    if not acc or acc in present or acc in dead:
                        totals["already_present_or_deleted"] += 1
                        continue
                    cat = classify_sample(_sample_fields(smp))
                    if cat is not None and cat not in KEPT_KINDS:
                        totals[f"skipped_non_marine:{cat}"] += 1
                        continue
                    keep.append(smp)
                if keep:
                    with session_scope() as session:
                        raw = {"bioproject_accession": bp, "total_linked_samples": len(uids), "truncated": False,
                               "samples": keep, "fetched_via": "fill_capped_projects"}
                        totals["facts"] += _persist(session, adapter, study_id, bp, parent, raw, skip_project_facts=True)
                    present.update(s["accession"] for s in keep)
                    added += len(keep)
            except Exception as exc:  # one bad chunk must not stop the run; a re-run retries it
                totals["errors"] += 1
                print(f"  ERROR {bp} chunk {j}: {type(exc).__name__}: {exc}", flush=True)
        totals["samples_added"] += added
        print(f"  [{i}/{len(targets)}] {study_id} {bp}: {len(uids):,} linked, {added:,} added  "
              f"({time.time() - t0:.0f}s, total added {totals['samples_added']:,})", flush=True)
    print(f"NCBI done: {dict(totals)}", flush=True)


def fill_ena(args, aliases) -> None:
    adapter = _build_enabled_adapters()["ena"]
    adapter.http._cache_enabled = False
    with session_scope() as session:
        targets = _targets(session, "run_coverage_note", "ena", aliases, args.study)
    print(f"ENA: {len(targets):,} kept capped (study, ENA project) targets (each has more than 500 runs; "
          f"the true count is only known once listed)", flush=True)
    if args.dry_run:
        return
    totals = collections.Counter()
    t0 = time.time()
    for i, (study_id, ident, parent, _total) in enumerate(targets[: args.max_projects or None], 1):
        try:
            with session_scope() as session:
                samples, runs_present = _present(session, "sample"), _present(session, "experiment_run") | _present(session, "sequencing_run")
            study_rows, _ = adapter.http.get_json(f"{adapter.config.base_url}/search", params={
                "result": "study", "query": f'study_accession="{ident}" OR secondary_study_accession="{ident}"',
                "fields": STUDY_FIELDS, "format": "json"})
            if not study_rows:
                totals["no_study_record"] += 1
                continue
            study = study_rows[0]
            run_rows, _ = adapter.http.get_json(f"{adapter.config.base_url}/search", params={
                "result": "read_run", "query": f'study_accession="{study["study_accession"]}"',
                "fields": RUN_FIELDS, "format": "json", "limit": 0})
            new = []
            for run in run_rows:
                if run.get("run_accession") in runs_present:
                    continue
                if run.get("sample_accession") not in samples:
                    totals["run_sample_not_kept"] += 1
                    continue
                run["fastq_access_status"] = "not_checked"
                new.append(run)
            for j in range(0, len(new), CHUNK):
                with session_scope() as session:
                    raw = {"study": study, "runs": new[j : j + CHUNK], "truncated": False, "total_runs_seen": len(run_rows)}
                    totals["facts"] += _persist(session, adapter, study_id, ident, parent, raw, skip_project_facts=True)
            totals["runs_added"] += len(new)
            print(f"  [{i}/{len(targets)}] {study_id} {ident}: {len(run_rows):,} runs, {len(new):,} added  "
                  f"({time.time() - t0:.0f}s)", flush=True)
        except Exception as exc:
            totals["errors"] += 1
            print(f"  ERROR {ident}: {type(exc).__name__}: {exc}", flush=True)
    print(f"ENA done: {dict(totals)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", choices=("ncbi", "ena"))
    ap.add_argument("--max-projects", type=int, default=0)
    ap.add_argument("--study")
    args = ap.parse_args()
    with session_scope() as session:
        aliases = project_aliases(session.connection().connection.driver_connection)
    if args.only in (None, "ncbi"):
        fill_ncbi(args, aliases)
    if args.only in (None, "ena"):
        fill_ena(args, aliases)  # after NCBI, so runs of just-filled samples are kept


if __name__ == "__main__":
    main()
