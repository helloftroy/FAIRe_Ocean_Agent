"""Compiles schemas/faire/schema.yaml into the canonical FAIRe term
registry. schema.yaml, classes.yaml, and enums.yaml are different
representations of the same standard (schema.yaml already embeds each
slot's class/table membership and enum values) -- this module reads from
schema.yaml only, so a term is never compiled twice from two
representations of the same data.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from fair_ocean_agent.config import REPO_ROOT

FAIRE_SCHEMA_DIR = REPO_ROOT / "schemas" / "faire"
FAIRE_REPOSITORY = "FAIRe_checklist"
FAIRE_COMMIT = "042ced519c9a4e3808086e6078c12883cb884cd0"
FAIRE_VERSION = "1.0.2"


def canonical_id(slot_name: str) -> str:
    return f"faire:{slot_name}"


def build_faire_registry() -> list[dict]:
    with (FAIRE_SCHEMA_DIR / "schema.yaml").open() as f:
        schema = yaml.safe_load(f)
    slots = schema.get("slots", {})

    terms = []
    for slot_name, slot in slots.items():
        annotations = slot.get("annotations") or {}
        mappings = slot.get("mappings") or {}
        terms.append(
            {
                "canonical_id": canonical_id(slot_name),
                "upstream_repository": FAIRE_REPOSITORY,
                "source_file": "schema.yaml",
                "upstream_field_name": slot_name,
                "git_commit": FAIRE_COMMIT,
                "standard_version": FAIRE_VERSION,
                "definition": slot.get("description"),
                "range": slot.get("range"),
                "identifier": slot.get("slot_uri"),
                "requirement_level": annotations.get("requirement_level"),
                "requirement_level_code": annotations.get("requirement_level_code"),
                "requirement_level_condition": annotations.get("requirement_level_condition"),
                "data_type": annotations.get("data_type") or [],
                "exact_mappings": mappings.get("exact_mappings") or [],
                "bebop_template_usage": [],
            }
        )
    return terms
