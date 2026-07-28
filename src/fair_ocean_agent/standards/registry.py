"""Builds the compiled standards registry: parses the vendored FAIRe
(schemas/faire/), MIOP (schemas/miop/), and BeBOP protocol template
(schemas/bebop/) schemas, resolves every BeBOP template field against
FAIRe/MIOP via crosswalk.py, and writes five output files under
standards/compiled/:

- faire_registry.json -- every FAIRe term, enriched with which BeBOP
  protocol template sections reference it.
- bebop_miop_registry.json -- MIOP terms (same enrichment) plus any
  bebop:-namespaced terms found with no upstream MIOP/FAIRe definition.
- term_crosswalk.csv -- one row per raw BeBOP template field instance,
  showing what it resolved to and at what confidence/priority.
- template_field_usage.csv -- the same information collapsed to one row
  per (canonical term, template, section), for "which protocols use this
  field" queries.
- standards_validation_report.json -- the result of every invariant check
  this registry is expected to satisfy (see the checks below), so a
  violation is a visible artifact, not just a possible test failure
  someone has to remember to run.

The upstream repositories are never edited or physically merged -- only
this compiled, clearly-derived registry combines them. Re-run
`fair-ocean build-standards-registry` after updating any vendored schema
file; nothing here is hand-maintained.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from fair_ocean_agent.standards.bebop_templates import parse_all_templates
from fair_ocean_agent.standards.crosswalk import bebop_specific_terms, build_crosswalk
from fair_ocean_agent.standards.faire_registry import (
    FAIRE_COMMIT,
    build_faire_registry,
)
from fair_ocean_agent.standards.miop_registry import MIOP_COMMIT, build_miop_registry


def _attach_template_usage(terms: list[dict], crosswalk) -> None:
    by_canonical_id: dict[str, dict] = {term["canonical_id"]: term for term in terms}
    for row in crosswalk:
        term = by_canonical_id.get(row.canonical_id)
        if term is None:
            continue
        term["bebop_template_usage"].append(
            {"template_file": row.template_file, "protocol_section": row.protocol_section}
        )


def _build_template_field_usage(crosswalk) -> list[dict]:
    seen: dict[tuple, dict] = {}
    for row in crosswalk:
        if row.canonical_id is None:
            continue
        key = (row.canonical_id, row.template_file, row.protocol_section)
        if key not in seen:
            seen[key] = {
                "canonical_id": row.canonical_id,
                "standard": row.canonical_id.split(":", 1)[0],
                "template_file": row.template_file,
                "protocol_section": row.protocol_section,
            }
    return sorted(seen.values(), key=lambda r: (r["template_file"], r["protocol_section"], r["canonical_id"]))


def _check_no_duplicate_canonical_ids(all_terms: list[dict]) -> dict:
    ids = [t["canonical_id"] for t in all_terms]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    return {"passed": not duplicates, "duplicates": duplicates}


def _check_every_bebop_field_resolves_or_marked(crosswalk) -> dict:
    from fair_ocean_agent.standards.crosswalk import (
        RESOLUTION_BEBOP_SPECIFIC,
        RESOLUTION_FAIRE,
        RESOLUTION_MIOP,
        RESOLUTION_REVIEW,
        RESOLUTION_UNRESOLVED,
    )

    known_resolutions = {RESOLUTION_MIOP, RESOLUTION_FAIRE, RESOLUTION_BEBOP_SPECIFIC, RESOLUTION_UNRESOLVED, RESOLUTION_REVIEW}
    unmarked = [
        f"{row.template_file}:{row.protocol_section}:{row.field_name}"
        for row in crosswalk
        if row.resolution not in known_resolutions
    ]
    return {"passed": not unmarked, "unmarked_fields": unmarked}


def _check_every_miop_range_reference_exists(miop_terms: list[dict], miop_enums: dict) -> dict:
    scalar_ranges = {"string", "integer", "double", "boolean", None}
    missing = []
    for term in miop_terms:
        range_name = term.get("range")
        if range_name in scalar_ranges:
            continue
        if range_name not in miop_enums:
            missing.append({"term": term["canonical_id"], "range": range_name})
    return {"passed": not missing, "missing": missing}


def _check_every_faire_class_slot_resolves(faire_terms: list[dict], classes: dict) -> dict:
    faire_field_names = {t["upstream_field_name"] for t in faire_terms}
    missing = []
    for class_name, class_def in classes.items():
        for slot_name in class_def.get("slots", []):
            if slot_name not in faire_field_names:
                missing.append({"class": class_name, "slot": slot_name})
    return {"passed": not missing, "missing": missing}


def _check_provenance_retained(all_terms: list[dict]) -> dict:
    required_fields = ("upstream_repository", "source_file", "upstream_field_name", "git_commit")
    incomplete = [
        t["canonical_id"]
        for t in all_terms
        if any(not t.get(field) for field in required_fields)
    ]
    return {"passed": not incomplete, "incomplete_provenance": incomplete}


def _check_no_reused_field_duplicated_as_bebop(
    bebop_terms: list[dict], faire_terms: list[dict], miop_terms: list[dict]
) -> dict:
    from fair_ocean_agent.standards.miop_registry import normalize_field_name

    known_normalized_names = {normalize_field_name(t["upstream_field_name"]) for t in faire_terms}
    known_normalized_names |= {normalize_field_name(t["upstream_field_name"]) for t in miop_terms}
    known_normalized_names |= {normalize_field_name(t["title"]) for t in miop_terms if t.get("title")}

    offenders = [
        t["canonical_id"]
        for t in bebop_terms
        if normalize_field_name(t["upstream_field_name"]) in known_normalized_names
    ]
    return {"passed": not offenders, "offenders": offenders}


def build_registry() -> dict:
    import yaml

    from fair_ocean_agent.standards.faire_registry import FAIRE_SCHEMA_DIR
    from fair_ocean_agent.standards.miop_registry import MIOP_SCHEMA_DIR

    faire_terms = build_faire_registry()
    miop_terms = build_miop_registry()
    template_fields = parse_all_templates()
    crosswalk = build_crosswalk(template_fields, faire_terms, miop_terms)
    bebop_terms = bebop_specific_terms(crosswalk)

    _attach_template_usage(faire_terms, crosswalk)
    _attach_template_usage(miop_terms, crosswalk)

    template_field_usage = _build_template_field_usage(crosswalk)

    with (FAIRE_SCHEMA_DIR / "classes.yaml").open() as f:
        classes = yaml.safe_load(f)["classes"]
    with (MIOP_SCHEMA_DIR / "terms.yaml").open() as f:
        miop_enums = (yaml.safe_load(f).get("enums") or {})

    all_terms = faire_terms + miop_terms + bebop_terms
    validation_report = {
        "generated_from": {"faire_commit": FAIRE_COMMIT, "miop_commit": MIOP_COMMIT},
        "counts": {
            "faire_terms": len(faire_terms),
            "miop_terms": len(miop_terms),
            "bebop_specific_terms": len(bebop_terms),
            "crosswalk_rows": len(crosswalk),
        },
        "checks": {
            "no_duplicate_canonical_identifiers": _check_no_duplicate_canonical_ids(all_terms),
            "every_bebop_field_resolves_or_marked_unresolved": _check_every_bebop_field_resolves_or_marked(crosswalk),
            "every_miop_range_reference_exists": _check_every_miop_range_reference_exists(miop_terms, miop_enums),
            "every_faire_class_slot_resolves": _check_every_faire_class_slot_resolves(faire_terms, classes),
            "provenance_retained": _check_provenance_retained(faire_terms + miop_terms),
            "no_reused_field_duplicated_as_bebop": _check_no_reused_field_duplicated_as_bebop(
                bebop_terms, faire_terms, miop_terms
            ),
        },
    }

    return {
        "faire_terms": faire_terms,
        "miop_terms": miop_terms,
        "bebop_terms": bebop_terms,
        "crosswalk": crosswalk,
        "template_field_usage": template_field_usage,
        "validation_report": validation_report,
    }


def write_registry(output_dir: str | Path) -> dict[str, int]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = build_registry()

    with (output_dir / "faire_registry.json").open("w") as f:
        json.dump(result["faire_terms"], f, indent=2)

    with (output_dir / "bebop_miop_registry.json").open("w") as f:
        json.dump(
            {"standard": "bebop_miop", "miop_terms": result["miop_terms"], "bebop_specific_terms": result["bebop_terms"]},
            f,
            indent=2,
        )

    with (output_dir / "term_crosswalk.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["template_file", "protocol_section", "field_name", "resolution", "canonical_id", "match_priority", "value"])
        for row in result["crosswalk"]:
            value = row.value if isinstance(row.value, (str, int, float, type(None))) else json.dumps(row.value)
            writer.writerow([row.template_file, row.protocol_section, row.field_name, row.resolution, row.canonical_id, row.match_priority, value])

    with (output_dir / "template_field_usage.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["canonical_id", "standard", "template_file", "protocol_section"])
        writer.writeheader()
        writer.writerows(result["template_field_usage"])

    with (output_dir / "standards_validation_report.json").open("w") as f:
        json.dump(result["validation_report"], f, indent=2)

    return {
        "faire_terms": len(result["faire_terms"]),
        "miop_terms": len(result["miop_terms"]),
        "bebop_specific_terms": len(result["bebop_terms"]),
        "crosswalk_rows": len(result["crosswalk"]),
        "template_field_usage_rows": len(result["template_field_usage"]),
    }
