from fair_ocean_agent.mapping import vocabularies


def test_platform_enum_is_closed_and_validates_membership():
    assert vocabularies.is_closed_vocab("platform_enum") is True
    result = vocabularies.check_value("platform_enum", "ILLUMINA")
    assert result.is_valid
    assert result.is_closed_vocab

    result = vocabularies.check_value("platform_enum", "NOT_A_REAL_PLATFORM")
    assert not result.is_valid


def test_env_broad_scale_enum_is_open_and_checks_shape_only():
    """env_broad_scale_enum lists exactly one illustrative ENVO purl
    upstream, not an exhaustive set -- any real ENVO-shaped term should
    pass, not just that one example."""
    assert vocabularies.is_closed_vocab("env_broad_scale_enum") is False

    result = vocabularies.check_value("env_broad_scale_enum", "http://purl.obolibrary.org/obo/ENVO_00000447")
    assert result.is_valid
    assert not result.is_closed_vocab

    result = vocabularies.check_value("env_broad_scale_enum", "some free text, not a term")
    assert not result.is_valid


def test_unknown_enum_name_does_not_crash():
    result = vocabularies.check_value("not_a_real_enum", "anything")
    assert result.is_valid  # not checked, not rejected
