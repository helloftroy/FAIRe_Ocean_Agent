"""Compiles schemas/miop/terms.yaml into the canonical MIOP term registry.
Abstract slots (`core field`, `society field`) are placeholders upstream
uses to group real terms via `is_a` -- they don't describe an actual
field, so they're excluded here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import yaml

from fair_ocean_agent.config import REPO_ROOT

MIOP_SCHEMA_DIR = REPO_ROOT / "schemas" / "miop"
MIOP_REPOSITORY = "miop"
MIOP_COMMIT = "be576bc7c9bce260905a1f6eec5c226104ad3ac4"
# No version file/tag exists upstream (checked README.md, LICENSE.txt) --
# the commit pin above is the only real version marker available; recording
# a fabricated version string here would be worse than admitting there
# isn't one.
MIOP_VERSION = None


def canonical_id(slot_name: str) -> str:
    return f"miop:{slot_name}"


def normalize_field_name(value: str) -> str:
    """Collapses hyphen/underscore/space/case variation so
    "broad-scale_environmental_context", "broad_scale_environmental_context",
    and "Broad Scale Environmental Context" all match one MIOP slot -- the
    real BeBOP protocol templates use all of these spellings interchangeably
    (see schemas/bebop/README.md)."""
    return re.sub(r"[-_\s]+", "", value.lower().strip())


def build_miop_registry() -> list[dict]:
    with (MIOP_SCHEMA_DIR / "terms.yaml").open() as f:
        data = yaml.safe_load(f)
    slots = data.get("slots", {})

    terms = []
    for slot_name, slot in slots.items():
        if slot.get("abstract"):
            continue
        terms.append(
            {
                "canonical_id": canonical_id(slot_name),
                "upstream_repository": MIOP_REPOSITORY,
                "source_file": "model/schema/terms.yaml",
                "upstream_field_name": slot_name,
                "git_commit": MIOP_COMMIT,
                "standard_version": MIOP_VERSION,
                "title": slot.get("title"),
                "definition": slot.get("description"),
                "range": slot.get("range"),
                "identifier": slot.get("slot_uri"),
                "bebop_template_usage": [],
            }
        )
    return terms


@dataclass
class MiopNameLookup:
    """Two separate lookups, not one merged dict, so a caller can tell
    whether a match came from the slot's own structural name (a stronger,
    priority-2 signal) or only from its human-readable title (a weaker,
    priority-4 alias match) -- both normalized the same way
    (normalize_field_name) so `meth_cat`, `methodology_category`, and
    `methodology category` all resolve to the same term regardless of
    which lookup matches."""

    by_slot_name: dict[str, str]
    by_title: dict[str, str]

    def get(self, normalized_field_name: str) -> str | None:
        return self.by_slot_name.get(normalized_field_name) or self.by_title.get(normalized_field_name)


def build_miop_name_lookup(terms: list[dict]) -> MiopNameLookup:
    by_slot_name = {normalize_field_name(term["upstream_field_name"]): term["canonical_id"] for term in terms}
    by_title = {
        normalize_field_name(term["title"]): term["canonical_id"] for term in terms if term.get("title")
    }
    return MiopNameLookup(by_slot_name, by_title)
