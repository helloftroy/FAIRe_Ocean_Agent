"""Synthetic-fixture tests for crosswalk.py's resolution paths that the
real vendored schemas don't currently exercise (every real field resolves
cleanly at priority 2 or 3 -- see test_standards_registry.py). These
construct fake terms/fields directly so the less-common paths (BeBOP-
specific, review-candidate, malformed input) are still covered.
"""
from fair_ocean_agent.standards.bebop_templates import TemplateField
from fair_ocean_agent.standards.crosswalk import (
    RESOLUTION_BEBOP_SPECIFIC,
    RESOLUTION_FAIRE,
    RESOLUTION_MIOP,
    RESOLUTION_REVIEW,
    RESOLUTION_UNRESOLVED,
    build_crosswalk,
)

FAKE_FAIRE_TERMS = [{"upstream_field_name": "env_broad_scale"}, {"upstream_field_name": "samp_name"}]
FAKE_MIOP_TERMS = [
    {"upstream_field_name": "meth_cat", "canonical_id": "miop:meth_cat", "title": "methodology category"},
    {"upstream_field_name": "project", "canonical_id": "miop:project", "title": "project"},
]


def test_exact_faire_field_resolves_at_priority_3():
    fields = [TemplateField("t.md", "FAIRe terms", "env_broad_scale", "some value")]
    rows = build_crosswalk(fields, FAKE_FAIRE_TERMS, FAKE_MIOP_TERMS)
    assert rows[0].resolution == RESOLUTION_FAIRE
    assert rows[0].match_priority == 3
    assert rows[0].canonical_id == "faire:env_broad_scale"


def test_exact_miop_slot_name_resolves_at_priority_2():
    fields = [TemplateField("t.md", "MIOP terms", "meth_cat", "sample collection")]
    rows = build_crosswalk(fields, FAKE_FAIRE_TERMS, FAKE_MIOP_TERMS)
    assert rows[0].resolution == RESOLUTION_MIOP
    assert rows[0].match_priority == 2
    assert rows[0].canonical_id == "miop:meth_cat"


def test_miop_title_alias_resolves_at_priority_4():
    fields = [TemplateField("t.md", "MIOP terms", "methodology_category", "sample collection")]
    rows = build_crosswalk(fields, FAKE_FAIRE_TERMS, FAKE_MIOP_TERMS)
    assert rows[0].resolution == RESOLUTION_MIOP
    assert rows[0].match_priority == 4
    assert rows[0].canonical_id == "miop:meth_cat"


def test_field_with_no_match_anywhere_becomes_bebop_specific():
    fields = [TemplateField("t.md", "FAIRe terms", "totally_novel_field", "some value")]
    rows = build_crosswalk(fields, FAKE_FAIRE_TERMS, FAKE_MIOP_TERMS)
    assert rows[0].resolution == RESOLUTION_BEBOP_SPECIFIC
    assert rows[0].canonical_id is None


def test_faire_section_field_that_only_normalized_matches_miop_is_flagged_for_review_not_auto_merged():
    """A field placed under '# FAIRe terms' that doesn't exactly match any
    FAIRe slot, but does normalized-match a MIOP term, must never be
    silently merged into that MIOP term -- per the dedup priority order,
    normalized-name-only overlap is a review candidate, not an automatic
    merge."""
    fields = [TemplateField("t.md", "FAIRe terms", "meth-cat", "value")]
    rows = build_crosswalk(fields, FAKE_FAIRE_TERMS, FAKE_MIOP_TERMS)
    assert rows[0].resolution == RESOLUTION_REVIEW
    assert rows[0].match_priority == 5
    assert rows[0].canonical_id == "miop:meth_cat"  # surfaced, but flagged for review


def test_empty_field_name_is_unresolved_not_bebop_specific():
    fields = [TemplateField("t.md", "FAIRe terms", "", "value")]
    rows = build_crosswalk(fields, FAKE_FAIRE_TERMS, FAKE_MIOP_TERMS)
    assert rows[0].resolution == RESOLUTION_UNRESOLVED


def test_identifier_shaped_field_name_flagged_at_priority_1():
    fields = [TemplateField("t.md", "FAIRe terms", "ENVO:00000447", "value")]
    rows = build_crosswalk(fields, FAKE_FAIRE_TERMS, FAKE_MIOP_TERMS)
    assert rows[0].match_priority == 1
    assert rows[0].resolution == RESOLUTION_REVIEW
