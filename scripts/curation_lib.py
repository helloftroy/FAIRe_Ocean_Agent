"""Shared, read-only helpers for the data-integrity audit and curation
(scripts/bioproject_audit.py, scripts/apply_curation.py). See
data_integrity_audit.md. All functions take a plain sqlite3 connection.
"""
from __future__ import annotations

import collections
import re
import sqlite3
from collections.abc import Iterable

from marine_audit import ORDER, categorize, classify_sample

EVIDENCE_FIELDS = ("organism", "host", "isolation_source", "env_broad_scale")
NON_MARINE = ("medical", "terrestrial", "engineered", "other_host")
KEPT_KINDS = ("marine", "freshwater")


def _in(values: Iterable[str]) -> str:
    return ",".join("'" + v.replace("'", "''") + "'" for v in values)


def sample_categories(conn: sqlite3.Connection) -> dict[str, str | None]:
    """sample entity -> category from its own evidence (organism, else host,
    else isolation_source, else env_broad_scale; any source), or None.
    marine_audit.classify_sample: a non-marine call is overruled by any
    marine/freshwater evidence unless the organism is unambiguous."""
    evidence: dict[str, dict[str, str]] = collections.defaultdict(dict)
    q = (
        "select entity_id, fact_type_candidate, raw_value from raw_facts "
        "indexed by ix_raw_facts_fact_type_candidate "
        f"where fact_type_candidate in ({_in(EVIDENCE_FIELDS)}) and entity_level='sample' and entity_id is not null"
    )
    for eid, field, value in conn.execute(q):
        evidence[eid].setdefault(field, value or "")
    return {eid: classify_sample(f, EVIDENCE_FIELDS) for eid, f in evidence.items()}


PROJECT_IDENTIFIER_TYPES = ("bioproject_accession", "sra_study_accession", "ena_study_accession")


def project_aliases(conn: sqlite3.Connection) -> dict[str, str]:
    """SRA/ENA study accession (SRP/ERP/DRP) -> its BioProject (PRJ...), from
    the secondary_study_accession facts of ENA sources keyed by the PRJ. One
    project is then judged once, under one name."""
    out: dict[str, str] = {}
    for prj, secondary in conn.execute(
        "select src.external_identifier, f.raw_value from raw_facts f "
        "indexed by ix_raw_facts_fact_type_candidate join sources src on src.source_id=f.source_id "
        "where f.fact_type_candidate='secondary_study_accession' and src.source_name='ena' "
        "and src.external_identifier like 'PRJ%'"
    ):
        for v in (secondary or "").replace(";", ",").split(","):
            v = v.strip()
            if v and not v.startswith("PRJ"):
                out.setdefault(v, prj)
    return out


def bioproject_identifiers(conn: sqlite3.Connection, aliases: dict[str, str] | None = None
                           ) -> dict[str, list[tuple[str, str, str]]]:
    """study -> [(project, identifier source, created_at)]. Every BioProject /
    SRA study / ENA study identifier, primary or shares_accession_with (a
    project first attached to another study is recorded as shared, but its
    samples were still fetched into this one), SRP/ERP/DRP translated to the
    BioProject through `aliases`."""
    aliases = aliases if aliases is not None else project_aliases(conn)
    out: dict[str, list[tuple[str, str, str]]] = collections.defaultdict(list)
    for s, v, src, created in conn.execute(
        "select study_id, identifier_value, coalesce(source,''), created_at from external_identifiers "
        f"where identifier_type in ({_in(PROJECT_IDENTIFIER_TYPES)})"
    ):
        out[s].append((aliases.get(v, v), src, str(created)))
    return out


def placements(conn: sqlite3.Connection, studies: Iterable[str], aliases: dict[str, str] | None = None
               ) -> dict[tuple[str, str], str]:
    """(study, sample entity) -> project, from NCBI BioSample facts (the
    fetch's BioProject) or ENA run facts (samp_name accession, the ENA
    source's project, SRP/ERP/DRP translated through `aliases`). NCBI wins
    when both exist."""
    aliases = aliases if aliases is not None else project_aliases(conn)
    sq = _in(studies)
    out: dict[tuple[str, str], str] = {}
    for s, e, bp in conn.execute(
        "select distinct f.study_id, f.entity_id, src.external_identifier from raw_facts f "
        "join sources src on src.source_id=f.source_id "
        f"where f.study_id in ({sq}) and f.entity_level='sample' and f.extraction_method='adapter:ncbi_biosample' "
        "and f.fact_type_candidate='organism'"
    ):
        out.setdefault((s, e), bp)
    acc_to_entity = {
        x: e for e, x in conn.execute(
            f"select entity_id, external_identifier from entities where entity_level='sample' and study_id in ({sq})"
        ) if x
    }
    for e, x in conn.execute(
        f"select es.entity_id, en.external_identifier from entity_studies es join entities en on en.entity_id=es.entity_id "
        f"where es.study_id in ({sq}) and en.entity_level='sample'"
    ):
        if x:
            acc_to_entity.setdefault(x, e)
    for s, acc, bp in conn.execute(
        "select f.study_id, f.raw_value, src.external_identifier from raw_facts f join sources src on src.source_id=f.source_id "
        f"where f.study_id in ({sq}) and f.fact_type_candidate='samp_name' and f.entity_level='experiment_run' "
        "and src.source_name='ena'"
    ):
        e = acc_to_entity.get(acc or "")
        if e:
            out.setdefault((s, e), aliases.get(bp, bp))
    return out


def digit_collisions(conn: sqlite3.Connection, studies: Iterable[str]) -> set[str]:
    """SAMN sample entities whose digits equal a SAMEA/SAMD accession the same
    study already knew: the efetch bug (data_integrity_audit.md Finding 1)."""
    sq = _in(studies)
    known: dict[str, set[int]] = collections.defaultdict(set)
    for s, v in conn.execute(
        f"select study_id, identifier_value from external_identifiers where identifier_type='biosample_accession' "
        f"and study_id in ({sq}) and (identifier_value like 'SAMEA%' or identifier_value like 'SAMD%')"
    ):
        known[s].add(int(re.sub(r"\D", "", v)))
    out = set()
    for s, e, x in conn.execute(
        "select distinct f.study_id, f.entity_id, en.external_identifier from raw_facts f join entities en on en.entity_id=f.entity_id "
        f"where f.study_id in ({sq}) and f.entity_level='sample' and f.extraction_method='adapter:ncbi_biosample' "
        "and f.fact_type_candidate='organism' and en.external_identifier like 'SAMN%'"
    ):
        if x[4:].isdigit() and int(x[4:]) in known[s]:
            out.add(e)
    return out


__all__ = [
    "ORDER", "NON_MARINE", "KEPT_KINDS", "EVIDENCE_FIELDS", "categorize",
    "sample_categories", "bioproject_identifiers", "placements", "digit_collisions",
    "project_aliases", "PROJECT_IDENTIFIER_TYPES",
]
