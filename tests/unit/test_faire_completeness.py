from fair_ocean_agent.validation.faire_completeness import CHECKABLE_CLASSES, unconditionally_mandatory_faire_fields


def test_only_returns_checkable_classes():
    result = unconditionally_mandatory_faire_fields()
    assert set(result.keys()) == set(CHECKABLE_CLASSES)


def test_excludes_conditionally_mandatory_fields():
    """env_broad_scale is Mandatory but conditioned on samp_category --
    must never appear here."""
    result = unconditionally_mandatory_faire_fields()
    sample_field_names = {t["upstream_field_name"] for t in result["sampleMetadata"]}
    assert "env_broad_scale" not in sample_field_names
    assert "decimalLatitude" not in sample_field_names  # also conditional


def test_includes_known_unconditionally_mandatory_fields():
    result = unconditionally_mandatory_faire_fields()
    sample_field_names = {t["upstream_field_name"] for t in result["sampleMetadata"]}
    assert {"eventDate", "geo_loc_name", "samp_name", "samp_category", "assay_name"} <= sample_field_names

    project_field_names = {t["upstream_field_name"] for t in result["projectMetadata"]}
    assert {"project_id", "project_contact", "recordedBy", "assay_type", "checkls_ver"} <= project_field_names


def test_every_returned_term_has_no_condition_and_is_mandatory():
    result = unconditionally_mandatory_faire_fields()
    for terms in result.values():
        for term in terms:
            assert term["requirement_level_code"] == "M"
            assert not term.get("requirement_level_condition")
