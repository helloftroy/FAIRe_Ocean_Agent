"""Parses schemas/bebop/protocol_template_*.md YAML frontmatter into raw
field-usage records, before any MIOP/FAIRe resolution happens (that's
crosswalk.py's job). Each template's frontmatter has two YAML-comment-
delimited sections -- `# MIOP terms` and `# FAIRe terms` -- so parsing
splits on those markers and parses each section as its own YAML document
rather than parsing the whole frontmatter as one document and losing the
section boundary (the comment markers aren't real YAML keys).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from fair_ocean_agent.config import REPO_ROOT

BEBOP_TEMPLATES_DIR = REPO_ROOT / "schemas" / "bebop"
BEBOP_REPOSITORY = "0_protocol_collection_template"
BEBOP_COMMIT = "5a17dabb192e65d3d9ea39613492f8330c9e86fc"

_FRONTMATTER_PATTERN = re.compile(r"^---\n(.*?)\n---\n", re.S)
_SECTION_SPLIT_PATTERN = re.compile(r"^#\s*(MIOP terms|FAIRe terms)\s*$", re.M)

TEMPLATE_FILES = (
    "protocol_template_sampling.md",
    "protocol_template_DNA_extraction.md",
    "protocol_template_PCR.md",
    "protocol_template_sequencing.md",
    "protocol_template_bioinformatics.md",
)


@dataclass(frozen=True)
class TemplateField:
    template_file: str
    protocol_section: str  # "MIOP terms" or "FAIRe terms"
    field_name: str  # raw key as written in the template frontmatter
    value: object  # whatever YAML parsed -- may be a string, None, or a dict


def _parse_template_frontmatter(text: str) -> tuple[dict, dict]:
    match = _FRONTMATTER_PATTERN.match(text)
    if match is None:
        raise ValueError("No YAML frontmatter block found")
    parts = _SECTION_SPLIT_PATTERN.split(match.group(1))
    if len(parts) != 5:
        raise ValueError(
            f"Expected exactly one '# MIOP terms' and one '# FAIRe terms' "
            f"section marker, found {len(parts) - 1} marker(s)"
        )
    _, _, miop_block, _, faire_block = parts
    miop_yaml = yaml.safe_load(miop_block) or {}
    faire_yaml = yaml.safe_load(faire_block) or {}
    return miop_yaml, faire_yaml


def parse_template(template_file: str) -> list[TemplateField]:
    text = (BEBOP_TEMPLATES_DIR / template_file).read_text()
    miop_yaml, faire_yaml = _parse_template_frontmatter(text)
    fields = []
    for field_name, value in miop_yaml.items():
        fields.append(TemplateField(template_file, "MIOP terms", field_name, value))
    for field_name, value in faire_yaml.items():
        fields.append(TemplateField(template_file, "FAIRe terms", field_name, value))
    return fields


def parse_all_templates() -> list[TemplateField]:
    fields = []
    for template_file in TEMPLATE_FILES:
        fields.extend(parse_template(template_file))
    return fields
