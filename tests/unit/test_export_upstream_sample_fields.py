"""Gap-2 rule targets missing from the schemas/faire mirror still get export columns (2026-11-11)."""
from pathlib import Path

import yaml

from fair_ocean_agent.exports import faire
from fair_ocean_agent.mapping import rules

REPO = Path(__file__).resolve().parents[2]


def test_extra_fields_are_real_upstream_sample_terms():
    classes = yaml.safe_load(open(REPO / "reference/faire/FAIRe_checklist/classes.yaml"))["classes"]
    upstream = set(classes["sampleMetadata"]["slots"])
    assert set(faire.UPSTREAM_SAMPLE_FIELDS_NOT_IN_MIRROR) <= upstream


def test_every_sample_rule_target_is_exported_or_deliberately_suppressed():
    columns = set(faire._exportable_sample_columns())
    lost = {
        r.target_field
        for r in rules.RULES
        if r.target_table == "sampleMetadata"
        and r.target_field not in columns
        and r.target_field not in faire.SAMPLE_METADATA_SUPPRESSED_FIELDS
        and not r.target_field.startswith("x_")
    }
    assert not lost, lost
