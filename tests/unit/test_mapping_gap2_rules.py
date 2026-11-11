"""Gap-2 sample-level spelling rules (mapping/rules.py, 2026-11-10 audit)."""
from fair_ocean_agent.exports.faire import SAMPLE_METADATA_SUPPRESSED_FIELDS
from fair_ocean_agent.mapping.rules import (
    _BASE_RULES,
    _GAP2_COPY_RULES_OF,
    _GAP2_DIRECT_RULES,
    RULES,
    rules_for,
)


def test_aliases_do_not_shadow_existing_rules():
    base = {r.source_fact_type for r in _BASE_RULES if r.source_entity_level in (None, "sample")}
    aliases = list(_GAP2_COPY_RULES_OF) + [alias for alias, _, _ in _GAP2_DIRECT_RULES]
    assert not [a for a in aliases if a in base]
    assert len(aliases) == len(set(aliases))


def test_copied_aliases_map_exactly_like_their_original():
    for alias, original in _GAP2_COPY_RULES_OF.items():
        got = {(r.target_table, r.target_field, r.transform, r.enum_name) for r in rules_for(alias, "sample")}
        want = {(r.target_table, r.target_field, r.transform, r.enum_name) for r in rules_for(original, "sample")}
        assert got == want, alias


def test_no_gap2_rule_targets_a_suppressed_export_field():
    new = RULES[len(_BASE_RULES):]
    assert not {r.target_field for r in new} & SAMPLE_METADATA_SUPPRESSED_FIELDS


def test_ena_spellings_parse():
    (lat,) = rules_for("geographic location (latitude)", "sample")
    assert lat.target_field == "decimalLatitude" and lat.transform("-18.29661667") == "-18.296617"
    (date,) = rules_for("collection_timestamp", "sample")
    assert date.target_field == "eventDate" and date.transform("2013-08-09 14:35") == "2013-08-09"
    fields = {r.target_field: r.transform("19.5") for r in rules_for("Depth (m)", "sample")}
    assert fields == {"minimumDepthInMeters": "19.5", "maximumDepthInMeters": "19.5"}


def test_gap2_rules_are_sample_level_only():
    for r in RULES[len(_BASE_RULES):]:
        assert r.source_entity_level == "sample"
