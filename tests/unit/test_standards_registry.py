"""Tests for the standards registry (mapping/schemas -> compiled registry
via src/fair_ocean_agent/standards/). Runs against the real vendored
schemas/faire, schemas/miop, schemas/bebop -- there's no synthetic fixture
here because the whole point of this registry is the real cross-standard
correspondence between real upstream schemas, which a synthetic fixture
can't stand in for. See registry.py's docstring for the design.
"""
from fair_ocean_agent.standards.bebop_templates import TEMPLATE_FILES, parse_all_templates, parse_template
from fair_ocean_agent.standards.crosswalk import RESOLUTION_BEBOP_SPECIFIC, RESOLUTION_FAIRE, RESOLUTION_MIOP, build_crosswalk
from fair_ocean_agent.standards.faire_registry import build_faire_registry
from fair_ocean_agent.standards.miop_registry import build_miop_registry, normalize_field_name
from fair_ocean_agent.standards.registry import build_registry


def test_build_faire_registry_has_full_provenance_and_known_field():
    terms = build_faire_registry()
    assert len(terms) > 300
    by_id = {t["canonical_id"]: t for t in terms}
    env = by_id["faire:env_broad_scale"]
    assert env["upstream_repository"] == "FAIRe_checklist"
    assert env["git_commit"]
    assert env["identifier"] == "mixs:0000012"
    assert env["requirement_level_condition"]  # the conditional-mandatory note


def test_build_miop_registry_excludes_abstract_slots():
    terms = build_miop_registry()
    names = {t["upstream_field_name"] for t in terms}
    assert "core field" not in names
    assert "society field" not in names
    assert "meth_cat" in names
    assert len(terms) == 21


def test_normalize_field_name_collapses_separator_and_case_variants():
    """normalize_field_name only collapses -/_/space/case -- it doesn't
    know "meth_cat" and "methodology_category" name the same MIOP term
    (that equivalence comes from matching one against the slot name and
    the other against the slot's title -- see build_miop_name_lookup and
    the priority-2-vs-4 tests in test_standards_crosswalk_synthetic.py)."""
    assert normalize_field_name("methodology category") == normalize_field_name("methodology_category")
    assert normalize_field_name("broad-scale_environmental_context") == normalize_field_name(
        "broad_scale_environmental_context"
    )
    assert normalize_field_name("meth_cat") != normalize_field_name("methodology_category")


def test_parse_template_splits_miop_and_faire_sections():
    fields = parse_template("protocol_template_sampling.md")
    sections = {f.protocol_section for f in fields}
    assert sections == {"MIOP terms", "FAIRe terms"}
    faire_fields = {f.field_name for f in fields if f.protocol_section == "FAIRe terms"}
    assert "env_broad_scale" in faire_fields
    miop_fields = {f.field_name for f in fields if f.protocol_section == "MIOP terms"}
    assert "methodology_category" in miop_fields


def test_parse_all_templates_covers_all_five_files():
    fields = parse_all_templates()
    seen_files = {f.template_file for f in fields}
    assert seen_files == set(TEMPLATE_FILES)


def test_crosswalk_resolves_every_real_field_with_no_review_or_unresolved():
    """Regression guard for real data: as of the vendored commits, every
    field in every template resolves cleanly (checked manually during
    Milestone 6 development) -- if a future schema/template update
    introduces a genuinely new or ambiguous field, this test should start
    failing loudly rather than the registry silently absorbing it."""
    faire_terms = build_faire_registry()
    miop_terms = build_miop_registry()
    fields = parse_all_templates()
    crosswalk = build_crosswalk(fields, faire_terms, miop_terms)

    resolutions = {row.resolution for row in crosswalk}
    assert resolutions == {RESOLUTION_MIOP, RESOLUTION_FAIRE}
    assert all(row.canonical_id is not None for row in crosswalk)


def test_crosswalk_faire_section_fields_resolve_to_exact_faire_slot():
    faire_terms = build_faire_registry()
    miop_terms = build_miop_registry()
    fields = parse_all_templates()
    crosswalk = build_crosswalk(fields, faire_terms, miop_terms)

    for row in crosswalk:
        if row.protocol_section == "FAIRe terms":
            assert row.resolution == RESOLUTION_FAIRE
            assert row.canonical_id == f"faire:{row.field_name}"


def test_registry_validation_report_all_checks_pass_on_real_schemas():
    result = build_registry()
    report = result["validation_report"]
    failed = {name: r for name, r in report["checks"].items() if not r["passed"]}
    assert not failed, failed


def test_registry_no_duplicate_canonical_ids_across_standards():
    result = build_registry()
    all_ids = (
        [t["canonical_id"] for t in result["faire_terms"]]
        + [t["canonical_id"] for t in result["miop_terms"]]
        + [t["canonical_id"] for t in result["bebop_terms"]]
    )
    assert len(all_ids) == len(set(all_ids))


def test_registry_bebop_template_usage_attached_to_faire_and_miop_terms():
    result = build_registry()
    faire_by_id = {t["canonical_id"]: t for t in result["faire_terms"]}
    env = faire_by_id["faire:env_broad_scale"]
    assert {"template_file": "protocol_template_sampling.md", "protocol_section": "FAIRe terms"} in env["bebop_template_usage"]

    miop_by_id = {t["canonical_id"]: t for t in result["miop_terms"]}
    project = miop_by_id["miop:project"]
    # "project" is a MIOP field used across all 5 templates.
    assert len(project["bebop_template_usage"]) == 5
