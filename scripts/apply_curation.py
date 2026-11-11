#!/usr/bin/env python3
"""Remove non-marine and foreign material from the database (2026-11-11).

User decisions (data_integrity_audit.md): drop non-marine material per
sample and per BioProject; delete what is obviously unwanted (human,
COVID, clinical, lab animals, soil, plants, engineered); freshwater stays;
flag (do not delete) studies left with no marine/freshwater samples and a
non-marine title, so the export skips them.

Decisions
  sample deleted    its own evidence (organism > host > isolation_source >
                    env_broad_scale, scripts/marine_audit.categorize) is
                    medical / terrestrial / engineered / other_host
                    -> rule sample_non_marine:<category>
  BioProject drop   bioproject_audit.csv verdict foreign_bug_chain or
                    non_marine, or "drop" in the manual review CSV
                    -> rule bp_<verdict> / bp_manual_review
  sample unlinked   placed in a dropped BioProject of study S but also
                    linked to a study that keeps it: removed from S only
                    (its S-scoped facts and values, its S link; re-homed
                    if S was its home study)
  sample deleted    placed in dropped BioProjects of every study it is
                    linked to -> the BioProject rule
  tombstone         a sample whose accession an earlier run deleted
  study flagged     no remaining marine/freshwater sample and a non-marine
                    title/description -> marine_relevance_status =
                    not_relevant (the export skips it); a study with a
                    marine/freshwater sample or title -> relevant
Children of a deleted sample are deleted with it: alias samples
(same_physical_sample_as), experiment runs derived only from deleted
samples, sequencing runs used only by deleted experiment runs. Every
decision is written to curation_exclusions (kept after deletion).

Default is a dry run: nothing is written to the database; the plan goes to
OUT_DIR as CSVs. --apply needs a database the user has backed up.

Usage:
  python scripts/apply_curation.py --db DB --audit AUDIT_DIR --review REVIEW_CSV --out OUT_DIR [--apply]
"""
from __future__ import annotations

import argparse
import collections
import csv
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent))
from curation_lib import KEPT_KINDS, NON_MARINE, PROJECT_IDENTIFIER_TYPES, placements, project_aliases, sample_categories  # noqa: E402
from marine_audit import text_category  # noqa: E402

DROP_VERDICTS = ("foreign_bug_chain", "foreign_hub", "non_marine")
CHUNK = 20_000


@dataclass
class Plan:
    delete: dict[str, tuple[str, str]] = field(default_factory=dict)            # entity -> (rule, reason)
    unlink: dict[tuple[str, str], tuple[str, str]] = field(default_factory=dict)  # (entity, study) -> (rule, reason)
    drop_bps: dict[tuple[str, str], tuple[str, str]] = field(default_factory=dict)  # (study, bp) -> (rule, reason)
    flag: dict[str, tuple[str, str]] = field(default_factory=dict)              # study -> (status, reason)
    accession: dict[str, str] = field(default_factory=dict)
    level: dict[str, str] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)                       # SRP/ERP/DRP -> PRJ


def _chunks(seq, n=900):
    seq = list(seq)
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _links(conn) -> dict[str, set[str]]:
    links: dict[str, set[str]] = collections.defaultdict(set)
    for e, s in conn.execute("select entity_id, study_id from entities where entity_level='sample'"):
        links[e].add(s)
    for e, s in conn.execute(
        "select es.entity_id, es.study_id from entity_studies es join entities en on en.entity_id=es.entity_id "
        "where en.entity_level='sample'"
    ):
        links[e].add(s)
    return links


def _children(conn, samples: set[str]) -> tuple[set[str], set[str], set[str]]:
    """(alias samples, experiment runs, sequencing runs) owned only by `samples`."""
    aliases, runs_from = set(), collections.defaultdict(set)
    for ch in _chunks(samples):
        q = ",".join("?" * len(ch))
        for f, t, typ in conn.execute(
            f"select from_entity_id, to_entity_id, relationship_type from entity_relationships where to_entity_id in ({q})", ch
        ):
            if typ == "same_physical_sample_as":
                aliases.add(f)
            elif typ == "derived_from_sample":
                runs_from[f].add(t)
    # an experiment run is deleted only if every sample it derives from is
    all_samples_of_run = collections.defaultdict(set)
    for ch in _chunks(runs_from):
        q = ",".join("?" * len(ch))
        for f, t in conn.execute(
            f"select from_entity_id, to_entity_id from entity_relationships where relationship_type='derived_from_sample' "
            f"and from_entity_id in ({q})", ch
        ):
            all_samples_of_run[f].add(t)
    gone = samples | aliases
    exp_runs = {r for r, ss in all_samples_of_run.items() if ss <= gone}
    seq_users = collections.defaultdict(set)
    for ch in _chunks(exp_runs):
        q = ",".join("?" * len(ch))
        for f, t in conn.execute(
            f"select from_entity_id, to_entity_id from entity_relationships where relationship_type='sequenced_in_run' "
            f"and from_entity_id in ({q})", ch
        ):
            seq_users[t].add(f)
    all_users = collections.defaultdict(set)
    for ch in _chunks(seq_users):
        q = ",".join("?" * len(ch))
        for f, t in conn.execute(
            f"select from_entity_id, to_entity_id from entity_relationships where relationship_type='sequenced_in_run' "
            f"and to_entity_id in ({q})", ch
        ):
            all_users[t].add(f)
    seq_runs = {t for t, users in all_users.items() if users <= exp_runs}
    return aliases - samples, exp_runs, seq_runs


def make_plan(conn, audit_rows: list[dict], review_rows: list[dict]) -> Plan:
    plan = Plan()
    for r in audit_rows:
        if r["verdict"] in DROP_VERDICTS:
            plan.drop_bps[(r["study_id"], r["bioproject"])] = (f"bp_{r['verdict']}", r.get("bioproject_title", "")[:200])
    for r in review_rows:
        if r["decision"] == "drop":
            plan.drop_bps[(r["study_id"], r["bioproject"])] = ("bp_manual_review", r["reason"])

    cats = sample_categories(conn)
    links = _links(conn)
    for e, cat in cats.items():
        if cat in NON_MARINE and e in links:
            plan.delete[e] = (f"sample_non_marine:{cat}", "")

    # tombstones from earlier runs
    has_table = conn.execute("select 1 from sqlite_master where type='table' and name='curation_exclusions'").fetchone()
    if has_table:
        dead = {a for (a,) in conn.execute(
            "select accession from curation_exclusions where level='sample' and action='deleted' and accession is not null")}
        for ch in _chunks(dead):
            q = ",".join("?" * len(ch))
            for e, in conn.execute(f"select entity_id from entities where entity_level='sample' and external_identifier in ({q})", ch):
                plan.delete.setdefault(e, ("tombstone", "deleted by an earlier curation run"))

    dropped_links: dict[str, dict[str, tuple[str, str]]] = collections.defaultdict(dict)
    plan.aliases = project_aliases(conn)
    if plan.drop_bps:
        studies = {s for s, _ in plan.drop_bps}
        for (s, e), bp in placements(conn, studies, plan.aliases).items():
            if (s, bp) in plan.drop_bps:
                rule, _ = plan.drop_bps[(s, bp)]
                dropped_links[e][s] = (rule, f"{bp} dropped from {s}")
    for e, by_study in dropped_links.items():
        if e in plan.delete:
            continue
        if links.get(e, set()) <= set(by_study):
            plan.delete[e] = next(iter(by_study.values()))
        else:
            for s, why in by_study.items():
                plan.unlink[(e, s)] = why

    # children of unlinked samples leave the same study with them
    by_study = collections.defaultdict(set)
    for e, s in plan.unlink:
        by_study[s].add(e)
    for s, samples in by_study.items():
        aliases, exp_runs, seq_runs = _children(conn, samples)
        for child in aliases | exp_runs | seq_runs:
            plan.unlink.setdefault((child, s), ("child_of_unlinked_sample", f"follows its sample out of {s}"))

    # children of deleted samples
    aliases, exp_runs, seq_runs = _children(conn, set(plan.delete))
    for a in aliases:
        plan.delete.setdefault(a, ("child_of_deleted_sample", "alias sample"))
    for r in exp_runs:
        plan.delete.setdefault(r, ("child_of_deleted_sample", "experiment run"))
    for r in seq_runs:
        plan.delete.setdefault(r, ("child_of_deleted_sample", "sequencing run"))

    for ch in _chunks(set(plan.delete) | {e for e, _ in plan.unlink}):
        q = ",".join("?" * len(ch))
        for e, x, lvl in conn.execute(f"select entity_id, external_identifier, entity_level from entities where entity_id in ({q})", ch):
            plan.accession[e] = x or ""
            plan.level[e] = lvl

    # study flags, on the state after deletion / unlinking
    kept_kind = collections.Counter()
    for e, studies in links.items():
        if e in plan.delete or cats.get(e) not in KEPT_KINDS:
            continue
        for s in studies:
            if (e, s) not in plan.unlink:
                kept_kind[s] += 1
    for s, title, desc in conn.execute("select study_id, title, description from studies where canonical_status != 'merged'"):
        tcat, _ = text_category(f"{title or ''} {desc or ''}")
        if kept_kind[s] or tcat in KEPT_KINDS:
            plan.flag[s] = ("relevant", f"{kept_kind[s]} marine/freshwater samples; title: {tcat}")
        elif tcat is not None:
            plan.flag[s] = ("not_relevant", f"no marine/freshwater samples; title: {tcat}")
    return plan


def write_plan(plan: Plan, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "delete_entities.csv").open("w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["entity_id", "level", "accession", "rule", "reason"])
        for e, (rule, why) in sorted(plan.delete.items()):
            w.writerow([e, plan.level.get(e, ""), plan.accession.get(e, ""), rule, why])
    with (out / "unlink.csv").open("w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["entity_id", "accession", "study_id", "rule", "reason"])
        for (e, s), (rule, why) in sorted(plan.unlink.items()):
            w.writerow([e, plan.accession.get(e, ""), s, rule, why])
    with (out / "drop_bioprojects.csv").open("w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["study_id", "bioproject", "rule", "reason"])
        for (s, bp), (rule, why) in sorted(plan.drop_bps.items()):
            w.writerow([s, bp, rule, why])
    with (out / "study_flags.csv").open("w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["study_id", "marine_relevance_status", "reason"])
        for s, (st, why) in sorted(plan.flag.items()):
            w.writerow([s, st, why])


def summarize(plan: Plan) -> str:
    lines = []
    by_level = collections.Counter(plan.level.get(e, "?") for e in plan.delete)
    by_rule = collections.Counter(r.split(":")[0] if r.startswith("sample_non") else r for r, _ in plan.delete.values())
    lines.append(f"delete entities: {len(plan.delete):,}  by level {dict(by_level)}")
    for r, n in by_rule.most_common():
        lines.append(f"    {r:28s} {n:>9,}")
    cats = collections.Counter(r.split(":")[1] for r, _ in plan.delete.values() if r.startswith("sample_non_marine"))
    lines.append(f"    non-marine samples by category: {dict(cats)}")
    lines.append(f"unlink (sample, study): {len(plan.unlink):,}")
    lines.append(f"drop (study, BioProject): {len(plan.drop_bps):,}  {dict(collections.Counter(r for r, _ in plan.drop_bps.values()))}")
    lines.append(f"study flags: {dict(collections.Counter(st for st, _ in plan.flag.values()))}")
    return "\n".join(lines)


# ---------------------------------------------------------------- apply

_ENTITY_DELETES = (
    "delete from standardized_value_evidence where standardized_value_id in "
    "(select standardized_value_id from standardized_values where entity_id in (select entity_id from cur_x))",
    "delete from standardized_value_evidence where fact_id in (select fact_id from raw_facts where entity_id in (select entity_id from cur_x))",
    "delete from validation_results where entity_id in (select entity_id from cur_x)",
    "delete from validation_results where fact_id in (select fact_id from raw_facts where entity_id in (select entity_id from cur_x))",
    "delete from validation_results where standardized_value_id in "
    "(select standardized_value_id from standardized_values where entity_id in (select entity_id from cur_x))",
    "delete from standardized_values where entity_id in (select entity_id from cur_x)",
    "delete from raw_facts where entity_id in (select entity_id from cur_x)",
    "delete from entity_relationships where from_entity_id in (select entity_id from cur_x)",
    "delete from entity_relationships where to_entity_id in (select entity_id from cur_x)",
    "delete from entity_studies where entity_id in (select entity_id from cur_x)",
    "delete from data_assets where entity_id in (select entity_id from cur_x)",
    "delete from api_paper_corrections where entity_id in (select entity_id from cur_x)",
    "update entities set parent_entity_id = null where parent_entity_id in (select entity_id from cur_x)",
    "delete from external_identifiers where identifier_value in (select accession from cur_x where accession != '')",
    "delete from entities where entity_id in (select entity_id from cur_x)",
)
_UNLINK_DELETES = (
    "delete from standardized_value_evidence where standardized_value_id in (select standardized_value_id from "
    "standardized_values where (entity_id, study_id) in (select entity_id, study_id from cur_u))",
    "delete from standardized_values where (entity_id, study_id) in (select entity_id, study_id from cur_u)",
    "delete from standardized_value_evidence where fact_id in (select fact_id from raw_facts where "
    "(entity_id, study_id) in (select entity_id, study_id from cur_u))",
    "delete from raw_facts where (entity_id, study_id) in (select entity_id, study_id from cur_u)",
    "delete from entity_relationships where (from_entity_id, study_id) in (select entity_id, study_id from cur_u)",
    "delete from entity_relationships where (to_entity_id, study_id) in (select entity_id, study_id from cur_u)",
    "delete from entity_studies where (entity_id, study_id) in (select entity_id, study_id from cur_u)",
    "delete from external_identifiers where (identifier_value, study_id) in "
    "(select accession, study_id from cur_u where accession != '')",
    # re-home an entity whose home study it was unlinked from
    "update entities set study_id = (select min(es.study_id) from entity_studies es where es.entity_id = entities.entity_id) "
    "where (entity_id, study_id) in (select entity_id, study_id from cur_u) "
    "and exists (select 1 from entity_studies es where es.entity_id = entities.entity_id)",
)


def _record(conn, rows, run_label):
    now = datetime.now(timezone.utc).isoformat(sep=" ")
    conn.executemany(
        "insert into curation_exclusions (curation_id, level, accession, entity_id, study_id, action, rule, reason, run_label, created_at) "
        "values (?,?,?,?,?,?,?,?,?,?)",
        [(f"CURAT-{uuid4().hex[:12]}", lvl, acc, e, s, action, rule, why, run_label, now)
         for lvl, acc, e, s, action, rule, why in rows],
    )


def apply_plan(conn, plan: Plan, run_label: str) -> None:
    if not conn.execute("select 1 from sqlite_master where type='table' and name='curation_exclusions'").fetchone():
        sys.exit("curation_exclusions table missing: run `alembic upgrade head` first")
    conn.execute("create temp table if not exists cur_x (entity_id text primary key, accession text)")
    conn.execute("create temp table if not exists cur_u (entity_id text, study_id text, accession text, primary key (entity_id, study_id))")
    home = {}
    t0 = time.time()

    # unlink first (needs entity_studies of the kept study intact for re-homing)
    items = sorted(plan.unlink.items())
    for i in range(0, len(items), CHUNK):
        part = items[i : i + CHUNK]
        conn.execute("delete from cur_u")
        conn.executemany("insert into cur_u values (?,?,?)", [(e, s, plan.accession.get(e, "")) for (e, s), _ in part])
        for sql in _UNLINK_DELETES:
            conn.execute(sql)
        _record(conn, [("sample", plan.accession.get(e, ""), e, s, "unlinked", rule, why) for (e, s), (rule, why) in part], run_label)
        conn.commit()
        print(f"  unlinked {min(i + CHUNK, len(items)):,}/{len(items):,}  {time.time() - t0:.0f}s", flush=True)

    ents = sorted(plan.delete.items())
    for ch in _chunks([e for e, _ in ents]):
        q = ",".join("?" * len(ch))
        home.update(dict(conn.execute(f"select entity_id, study_id from entities where entity_id in ({q})", ch)))
    for i in range(0, len(ents), CHUNK):
        part = ents[i : i + CHUNK]
        conn.execute("delete from cur_x")
        conn.executemany("insert into cur_x values (?,?)", [(e, plan.accession.get(e, "")) for e, _ in part])
        for sql in _ENTITY_DELETES:
            conn.execute(sql)
        _record(conn, [(plan.level.get(e, "sample"), plan.accession.get(e, ""), e, home.get(e), "deleted", rule, why)
                       for e, (rule, why) in part], run_label)
        conn.commit()
        print(f"  deleted {min(i + CHUNK, len(ents)):,}/{len(ents):,} entities  {time.time() - t0:.0f}s", flush=True)

    # dropped BioProjects: identifier rows; repository sources left without facts
    names_of = collections.defaultdict(set)
    for alias, prj in plan.aliases.items():
        names_of[prj].add(alias)
    types = ",".join(f"'{t}'" for t in PROJECT_IDENTIFIER_TYPES)
    for (s, bp), (rule, why) in sorted(plan.drop_bps.items()):
        for name in {bp} | names_of.get(bp, set()):
            conn.execute(
                f"delete from external_identifiers where study_id=? and identifier_type in ({types}) and identifier_value=?",
                (s, name),
            )
    _record(conn, [("bioproject", bp, None, s, "excluded", rule, why) for (s, bp), (rule, why) in plan.drop_bps.items()], run_label)
    studies = {s for s, _ in plan.drop_bps} | {s for _, s in plan.unlink} | {s for s in home.values() if s}
    empty = []
    for ch in _chunks(studies):
        q = ",".join("?" * len(ch))
        empty += [x for (x,) in conn.execute(
            f"select src.source_id from sources src where src.study_id in ({q}) "
            "and src.source_name in ('ncbi_biosample','ncbi_bioproject','ena') "
            "and not exists (select 1 from raw_facts f where f.source_id = src.source_id)", ch)]
    for ch in _chunks(empty):
        q = ",".join("?" * len(ch))
        conn.execute(f"delete from study_sources where source_id in ({q})", ch)
        conn.execute(f"delete from source_relationships where from_source_id in ({q}) or to_source_id in ({q})", ch + ch)
        conn.execute(f"update sources set parent_source_id = null where parent_source_id in ({q})", ch)
        conn.execute(f"update tasks set source_id = null where source_id in ({q})", ch)
        conn.execute(f"delete from data_assets where source_id in ({q})", ch)
        conn.execute(f"delete from prepared_source_texts where source_id in ({q})", ch)
        conn.execute(f"delete from sources where source_id in ({q})", ch)
    conn.commit()
    print(f"  dropped {len(plan.drop_bps):,} BioProject links, removed {len(empty):,} empty repository sources", flush=True)

    for s, (status, why) in plan.flag.items():
        conn.execute("update studies set marine_relevance_status=? where study_id=?", (status, s))
    _record(conn, [("study", None, None, s, "excluded", "study_not_marine", why)
                   for s, (st, why) in plan.flag.items() if st == "not_relevant"], run_label)
    conn.commit()
    print(f"  flagged {sum(1 for st, _ in plan.flag.values() if st == 'not_relevant'):,} studies not_relevant", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True)
    ap.add_argument("--audit", required=True, help="folder with bioproject_audit.csv (cluster/run_audits.sbatch)")
    ap.add_argument("--review", required=True, help="manual review CSV (curation/bioproject_review_*.csv)")
    ap.add_argument("--out", required=True, help="folder for the plan CSVs")
    ap.add_argument("--apply", action="store_true", help="write to the database (back it up first)")
    ap.add_argument("--run-label", default=f"curation-{datetime.now():%Y-%m-%d}")
    args = ap.parse_args()

    audit_rows = list(csv.DictReader(open(Path(args.audit) / "bioproject_audit.csv")))
    review_rows = list(csv.DictReader(open(args.review)))
    uri = f"file:{args.db}" + ("" if args.apply else "?mode=ro")
    conn = sqlite3.connect(uri, uri=True, timeout=600)
    t0 = time.time()
    plan = make_plan(conn, audit_rows, review_rows)
    write_plan(plan, Path(args.out))
    print(summarize(plan))
    print(f"plan written to {args.out} in {time.time() - t0:.0f}s")
    if args.apply:
        apply_plan(conn, plan, args.run_label)
        print("applied.")
    else:
        print("dry run: database not changed.")


if __name__ == "__main__":
    main()
