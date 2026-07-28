"""Resolves each raw BeBOP template field (from bebop_templates.py) against
the FAIRe and MIOP registries, using this fixed dedup priority:

1. Exact ontology URI or explicit term identifier
2. Exact MIOP term identifier (the template field name IS the MIOP slot's
   own name, e.g. `meth_cat`)
3. Exact FAIRe term identifier (the template field name IS a FAIRe slot
   name -- every `# FAIRe terms` field is checked this way)
4. Explicit reference from a BeBOP template to an existing MIOP or FAIRe
   term via a known alias/title (e.g. `methodology_category` for MIOP's
   `meth_cat`, whose title is "methodology category")
5. Normalized field name only as a review candidate, never an automatic
   merge -- flagged `Possible duplicate requiring review`, never silently
   resolved

A field that matches nothing at any priority becomes a `bebop:`-namespaced
term (`BeBOP-specific`) rather than being dropped -- every field either
resolves to a priority level or is explicitly marked, per the "no silent
loss" principle used throughout this pipeline (see validation/*.py).
"""
from __future__ import annotations

from dataclasses import dataclass

from fair_ocean_agent.standards.bebop_templates import BEBOP_COMMIT, BEBOP_REPOSITORY, TemplateField
from fair_ocean_agent.standards.faire_registry import canonical_id as faire_canonical_id
from fair_ocean_agent.standards.miop_registry import MiopNameLookup, build_miop_name_lookup, normalize_field_name

RESOLUTION_MIOP = "MIOP"
RESOLUTION_FAIRE = "FAIRe"
RESOLUTION_BEBOP_SPECIFIC = "BeBOP-specific"
RESOLUTION_UNRESOLVED = "Unresolved"
RESOLUTION_REVIEW = "Possible duplicate requiring review"


@dataclass(frozen=True)
class CrosswalkRow:
    template_file: str
    protocol_section: str
    field_name: str
    resolution: str
    canonical_id: str | None
    match_priority: int | None
    value: object


def _looks_like_identifier(value: str) -> bool:
    return ":" in value and not value.strip().startswith("#")


def resolve_field(
    field: TemplateField,
    faire_field_names: set[str],
    miop_name_lookup: MiopNameLookup,
) -> CrosswalkRow:
    if not field.field_name or not field.field_name.strip():
        # Malformed input, not a genuine novel field -- don't confidently
        # mint a bebop: term for it.
        return CrosswalkRow(field.template_file, field.protocol_section, field.field_name, RESOLUTION_UNRESOLVED, None, None, field.value)

    normalized_field_name = normalize_field_name(field.field_name)

    # Priority 1: the field name itself is already an explicit identifier.
    if _looks_like_identifier(field.field_name):
        return CrosswalkRow(field.template_file, field.protocol_section, field.field_name, RESOLUTION_REVIEW, None, 1, field.value)

    if field.protocol_section == "FAIRe terms":
        # Priority 3: exact FAIRe slot name match.
        if field.field_name in faire_field_names:
            return CrosswalkRow(
                field.template_file, field.protocol_section, field.field_name,
                RESOLUTION_FAIRE, faire_canonical_id(field.field_name), 3, field.value,
            )
    else:  # "MIOP terms"
        # Priority 2: the field name IS the MIOP slot's own structural name.
        exact_id = miop_name_lookup.by_slot_name.get(normalized_field_name)
        if exact_id is not None:
            return CrosswalkRow(
                field.template_file, field.protocol_section, field.field_name,
                RESOLUTION_MIOP, exact_id, 2, field.value,
            )
        # Priority 4: matches only via the slot's human-readable title --
        # covers real spelling variants like "methodology_category" for
        # MIOP's `meth_cat` (title "methodology category").
        alias_id = miop_name_lookup.by_title.get(normalized_field_name)
        if alias_id is not None:
            return CrosswalkRow(
                field.template_file, field.protocol_section, field.field_name,
                RESOLUTION_MIOP, alias_id, 4, field.value,
            )

    # Priority 5: normalized-name-only overlap against the *other*
    # registry than the one its section claims -- a review candidate,
    # never an automatic merge.
    cross_registry_match = miop_name_lookup.get(normalized_field_name) or next(
        (faire_canonical_id(name) for name in faire_field_names if normalized_field_name == normalize_field_name(name)),
        None,
    )
    if cross_registry_match is not None:
        return CrosswalkRow(
            field.template_file, field.protocol_section, field.field_name,
            RESOLUTION_REVIEW, cross_registry_match, 5, field.value,
        )

    return CrosswalkRow(field.template_file, field.protocol_section, field.field_name, RESOLUTION_BEBOP_SPECIFIC, None, None, field.value)


def build_crosswalk(
    fields: list[TemplateField],
    faire_terms: list[dict],
    miop_terms: list[dict],
) -> list[CrosswalkRow]:
    faire_field_names = {term["upstream_field_name"] for term in faire_terms}
    miop_name_lookup = build_miop_name_lookup(miop_terms)
    return [resolve_field(field, faire_field_names, miop_name_lookup) for field in fields]


def bebop_specific_terms(crosswalk: list[CrosswalkRow]) -> list[dict]:
    """One canonical term per distinct BeBOP-specific field name (deduped
    across templates -- the same unresolved field can appear in more than
    one template)."""
    seen: dict[str, dict] = {}
    for row in crosswalk:
        if row.resolution != RESOLUTION_BEBOP_SPECIFIC:
            continue
        key = normalize_field_name(row.field_name)
        if key not in seen:
            seen[key] = {
                "canonical_id": f"bebop:{row.field_name}",
                "upstream_repository": BEBOP_REPOSITORY,
                "source_file": row.template_file,
                "upstream_field_name": row.field_name,
                "git_commit": BEBOP_COMMIT,
                "standard_version": None,
                "definition": None,
                "bebop_template_usage": [],
            }
        seen[key]["bebop_template_usage"].append(
            {"template_file": row.template_file, "protocol_section": row.protocol_section}
        )
    return list(seen.values())
