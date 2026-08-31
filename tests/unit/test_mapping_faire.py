"""Tests for mapping/faire.py, shaped after this pipeline's real raw_facts
data (see mapping/rules.py's docstring for the query that grounded these
fields): NCBI BioSample sample-level facts, ENA sequencing_run-level
facts repeated identically across many runs, and LLM-extracted study-level
free text.
"""
from fair_ocean_agent.database.enums import EntityLevel, EntityRootStatus, IdentifierType, RelationshipType, SupportType
from fair_ocean_agent.database.models import (
    Entity,
    EntityStudy,
    ExternalIdentifier,
    RawFact,
    StandardizedValue,
    StandardizedValueEvidence,
    Study,
)
from fair_ocean_agent.extraction.faire_fields import assay_scoped_field_names, native_name_to_faire_hint
from fair_ocean_agent.mapping.envo import expand_envo_terms
from fair_ocean_agent.mapping.faire import map_study_to_faire, resolve_project_id
from fair_ocean_agent.mapping.rules import _ADDITIONAL_ENVIRONMENTAL_SAMPLE_ATTRIBUTES, RULES, rules_for


def _home_entity_study(entity: Entity) -> EntityStudy:
    """Every production Entity gets a home entity_studies row AND its own
    root_status/root_study_id set eagerly at creation
    (identity/entity_linking.py::create_entity) -- direct Entity(...)
    construction in these fixtures bypasses both, so tests that need
    mapping/faire.py's own _authoritative_sample_entities (sample-type
    routing) to see a real home link (and a real, determined root) must add
    one explicitly. Mirrors test_exports_faire.py's identical helper.
    Mutates `entity` in place (root fields) and returns the corresponding
    EntityStudy row for the caller to add separately."""
    entity.root_status = EntityRootStatus.DETERMINED.value
    entity.root_study_id = entity.study_id
    return EntityStudy(
        entity_id=entity.entity_id,
        study_id=entity.study_id,
        relationship_type=RelationshipType.IS_HOME_OF.value,
        confidence=SupportType.STRUCTURED_SOURCE.value,
    )


def _study(session, **kwargs) -> Study:
    study = Study(**kwargs)
    session.add(study)
    session.flush()
    return study


def _fact(session, study, *, entity=None, field, value, entity_level, support=SupportType.STRUCTURED_SOURCE):
    fact = RawFact(
        study_id=study.study_id,
        entity_id=entity.entity_id if entity else None,
        raw_field_name=field,
        raw_value=value,
        fact_type_candidate=field,
        entity_level=entity_level,
        support_type=support.value,
    )
    session.add(fact)
    session.flush()
    return fact


def test_expands_bare_envo_ids_to_label_pipe_accession():
    assert expand_envo_terms("ENVO:00000428") == "biome | ENVO:00000428"
    assert expand_envo_terms("http://purl.obolibrary.org/obo/ENVO_00000486") == "shoreline | ENVO:00000486"
    assert expand_envo_terms("ENVO:00010483") == "environmental material | ENVO:00010483"


def test_expands_labelled_envo_values_to_pipe_format():
    assert expand_envo_terms("marine sediment [ENVO:00002164]") == "marine sediment | ENVO:00002164"


def test_filter_passive_active_defaults_from_broad_checklist_evidence(db_session):
    """Real gap caught live (10.1093/ismejo/wrae013): filter_name resolved
    via the generic broad-checklist (llm_text_extraction), not the
    category pipeline or the BioSample samp_mat_process derivation --
    neither of which carries its own fallback for evidence gathered
    outside its own mechanism. filter_passive_active_0_1 must still
    default to passive/0 since no active-filtration language is present."""
    study = _study(db_session, title="Filter passive/active broad-checklist gap")
    _fact(db_session, study, field="filter_name", value="Henke-Ject", entity_level="study")
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    fact = (
        db_session.query(RawFact)
        .filter_by(study_id=study.study_id, fact_type_candidate="filter_passive_active_0_1")
        .one()
    )
    assert fact.raw_value == "0"


def test_filter_passive_active_defaults_to_active_when_pump_language_present(db_session):
    study = _study(db_session, title="Filter passive/active with pump evidence")
    fact = _fact(db_session, study, field="filter_name", value="Sterivex cartridge", entity_level="study")
    fact.evidence_quote = "Water was pumped through a Sterivex cartridge filter using a peristaltic pump."
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    result = (
        db_session.query(RawFact)
        .filter_by(study_id=study.study_id, fact_type_candidate="filter_passive_active_0_1")
        .one()
    )
    assert result.raw_value == "1"


def test_filter_passive_active_defaults_to_active_for_sterivex_without_pump_language(db_session):
    study = _study(db_session, title="Filter passive/active with Sterivex evidence")
    _fact(db_session, study, field="filter_name", value="Sterivex filter", entity_level="study")
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    result = (
        db_session.query(RawFact)
        .filter_by(study_id=study.study_id, fact_type_candidate="filter_passive_active_0_1")
        .one()
    )
    assert result.raw_value == "1"


def test_filter_passive_active_backfill_is_idempotent_and_skips_when_already_resolved(db_session):
    study = _study(db_session, title="Filter passive/active already resolved")
    _fact(db_session, study, field="filter_name", value="Sterivex cartridge", entity_level="study")
    _fact(db_session, study, field="filter_passive_active_0_1", value="1", entity_level="study")
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    count = (
        db_session.query(RawFact)
        .filter_by(study_id=study.study_id, fact_type_candidate="filter_passive_active_0_1")
        .count()
    )
    assert count == 1


def test_maps_sample_level_structured_facts(db_session):
    study = _study(db_session, title="Sample-level mapping")
    sample = Entity(study_id=study.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMN1")
    db_session.add(sample)
    db_session.flush()
    _fact(db_session, study, entity=sample, field="collection_date", value="2023-12-06", entity_level="sample")
    _fact(db_session, study, entity=sample, field="eventDate_submitted", value="2024-02-03", entity_level="sample")
    _fact(db_session, study, entity=sample, field="depth", value="5 meters", entity_level="sample")
    _fact(db_session, study, entity=sample, field="env_broad_scale", value="http://purl.obolibrary.org/obo/ENVO_00000447", entity_level="sample")
    _fact(db_session, study, entity=sample, field="env_local_scale", value="ENVO:00000486", entity_level="sample")
    _fact(db_session, study, entity=sample, field="env_medium", value="ENVO:00010483", entity_level="sample")
    _fact(db_session, study, entity=sample, field="geo_loc_name", value="USA: California", entity_level="sample")
    _fact(db_session, study, entity=sample, field="lat_lon", value="38.03 N 122.151667 W", entity_level="sample")
    db_session.commit()

    created = map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    values = {
        sv.target_field: sv
        for sv in db_session.query(StandardizedValue).filter_by(study_id=study.study_id, entity_id=sample.entity_id)
    }
    assert values["eventDate"].standardized_value == "2023-12-06"
    assert values["eventDate_submitted"].standardized_value == "2024-02-03"
    assert "verbatimEventDate" not in values
    assert values["minimumDepthInMeters"].standardized_value == "5"
    assert values["maximumDepthInMeters"].standardized_value == "5"
    assert values["env_broad_scale"].standardized_value == "marine biome | ENVO:00000447"
    assert values["env_local_scale"].standardized_value == "shoreline | ENVO:00000486"
    assert values["env_medium"].standardized_value == "environmental material | ENVO:00010483"
    assert not values["env_broad_scale"].review_required
    assert values["geo_loc_name"].standardized_value == "USA: California"
    assert values["geo_loc_name"].missingness_status == "present"
    assert values["decimalLatitude"].standardized_value == "38.030000"
    assert values["decimalLongitude"].standardized_value == "-122.151667"
    assert values["samp_name"].standardized_value == "SAMN1"
    assert values["samp_name"].mapping_method == "exact_identifier"
    assert values["samp_name"].missingness_status == "present"
    # +3 for checkls_ver, informationWithheld, and lib_layout (entity_id=None, so
    # outside this sample-scoped `values` dict) -- both always synced as
    # constants/defaults regardless of a study's own facts (mapping/
    # faire.py::_sync_checklist_version, and the informationWithheld
    # "Nothing indicated as withheld" default).
    assert created == len(values) + 3


def test_normalized_sample_prep_fact_wins_over_sentence_fact(db_session):
    study = _study(db_session, title="Normalized sample-prep mapping")
    _fact(
        db_session,
        study,
        field="nucl_acid_ext_lysis",
        value="treated in an ultrasonic water bath",
        entity_level=EntityLevel.STUDY.value,
        support=SupportType.EXPLICIT,
    )
    _fact(
        db_session,
        study,
        field="nucl_acid_ext_lysis_normalized",
        value="sonication | treated in an ultrasonic water bath",
        entity_level=EntityLevel.STUDY.value,
        support=SupportType.EXPLICIT,
    )
    _fact(
        db_session,
        study,
        field="nucl_acid_ext_sep",
        value="DNA was purified by phenol-chloroform extraction",
        entity_level=EntityLevel.STUDY.value,
        support=SupportType.EXPLICIT,
    )
    _fact(
        db_session,
        study,
        field="nucl_acid_ext_sep_normalized",
        value="phenol-chloroform | DNA was purified by phenol-chloroform extraction",
        entity_level=EntityLevel.STUDY.value,
        support=SupportType.EXPLICIT,
    )
    _fact(
        db_session,
        study,
        field="samp_collect_method",
        value="Integrated water samples were collected from the upper 50 m",
        entity_level=EntityLevel.STUDY.value,
        support=SupportType.EXPLICIT,
    )
    _fact(
        db_session,
        study,
        field="samp_collect_method_normalized",
        value="integrated-depth sampling | Integrated water samples were collected from the upper 50 m",
        entity_level=EntityLevel.STUDY.value,
        support=SupportType.EXPLICIT,
    )
    _fact(
        db_session,
        study,
        field="samp_mat_process",
        value="Samples were filtered, freeze-dried, and ground before DNA extraction",
        entity_level=EntityLevel.STUDY.value,
        support=SupportType.EXPLICIT,
    )
    _fact(
        db_session,
        study,
        field="samp_mat_process_normalized",
        value="filtration | freeze-drying | grinding | Samples were filtered, freeze-dried, and ground before DNA extraction",
        entity_level=EntityLevel.STUDY.value,
        support=SupportType.EXPLICIT,
    )
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    values = {
        row.target_field: row
        for row in db_session.query(StandardizedValue).filter_by(study_id=study.study_id)
    }
    assert values["nucl_acid_ext_lysis"].standardized_value == "sonication | treated in an ultrasonic water bath"
    assert values["nucl_acid_ext_sep"].standardized_value == (
        "phenol-chloroform | DNA was purified by phenol-chloroform extraction"
    )
    assert values["samp_collect_method"].standardized_value == (
        "integrated-depth sampling | Integrated water samples were collected from the upper 50 m"
    )
    assert values["samp_mat_process"].standardized_value == (
        "filtration | freeze-drying | grinding | Samples were filtered, freeze-dried, and ground before DNA extraction"
    )
    assert values["nucl_acid_ext_lysis"].review_required is True
    assert values["nucl_acid_ext_sep"].review_required is True
    assert values["samp_collect_method"].review_required is True
    assert values["samp_mat_process"].review_required is True


def test_maps_concentration_with_legacy_unit_fact_collapsed_into_value(db_session):
    study = _study(db_session, title="Concentration unit collapse")
    _fact(
        db_session,
        study,
        field="concentration",
        value="12.4",
        entity_level=EntityLevel.STUDY.value,
        support=SupportType.EXPLICIT,
    )
    # Legacy/raw source compatibility: concentration_unit is no longer mapped
    # or exported as its own FAIRe column, but if a raw fact exists, preserve
    # it by appending it to concentration.
    _fact(
        db_session,
        study,
        field="concentration_unit",
        value="ng/uL",
        entity_level=EntityLevel.STUDY.value,
        support=SupportType.EXPLICIT,
    )
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    values = {
        sv.target_field: sv.standardized_value
        for sv in db_session.query(StandardizedValue).filter_by(study_id=study.study_id)
    }
    assert values["concentration"] == "12.4 ng/uL"
    assert "concentration_unit" not in values
    assert "concentration_method" not in values


def test_maps_biological_rep_relation_sample_level_fact(db_session):
    study = _study(db_session, title="Replicate mapping")
    sample = Entity(study_id=study.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="S01_1")
    db_session.add(sample)
    db_session.flush()
    _fact(
        db_session,
        study,
        entity=sample,
        field="biological_rep_relation",
        value="S01_1 | S01_2 | S01_3",
        entity_level="sample",
        support=SupportType.DETERMINISTICALLY_DERIVED,
    )
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    values = {
        sv.target_field: sv
        for sv in db_session.query(StandardizedValue).filter_by(study_id=study.study_id, entity_id=sample.entity_id)
    }
    assert values["biological_rep_relation"].standardized_value == "S01_1 | S01_2 | S01_3"
    assert values["biological_rep_relation"].review_required is True


def test_biological_rep_derives_a_single_count_from_one_relation_group(db_session):
    """projectMetadata.biological_rep is now derived purely from this
    study's own sampleMetadata biological_rep_relation facts (mapping/
    faire.py::_apply_biological_rep_from_relations) -- per an explicit
    user request, the paper's text is never queried for this field
    anymore. One group of 3 members -> "3", the group's own size."""
    study = _study(db_session, title="Single replicate group")
    samples = [
        Entity(study_id=study.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier=f"S{i}")
        for i in range(5)
    ]
    db_session.add_all(samples)
    db_session.flush()
    for sample in samples:
        db_session.add(_home_entity_study(sample))
    for sample in samples[:3]:
        _fact(
            db_session,
            study,
            entity=sample,
            field="biological_rep_relation",
            value="S0 | S1 | S2",
            entity_level="sample",
            support=SupportType.DETERMINISTICALLY_DERIVED,
        )
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    value = db_session.query(StandardizedValue).filter_by(
        study_id=study.study_id,
        entity_id=None,
        target_field="biological_rep",
    ).one()
    assert value.standardized_value == "3"
    assert value.mapping_method == "deterministic_synonym"
    assert value.review_required is False


def test_biological_rep_reports_a_range_across_differently_sized_groups(db_session):
    """Some samples might have 2 replicates, others 4 -- the study-level
    value is the "min-max" range across the study's own distinct
    replicate groups, per an explicit user request."""
    study = _study(db_session, title="Mixed replicate group sizes")
    samples = [
        Entity(study_id=study.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier=f"S{i}")
        for i in range(6)
    ]
    db_session.add_all(samples)
    db_session.flush()
    for sample in samples:
        db_session.add(_home_entity_study(sample))
    # Group A: S0/S1 (size 2). Group B: S2/S3/S4/S5 (size 4).
    for sample in samples[:2]:
        _fact(
            db_session, study, entity=sample, field="biological_rep_relation",
            value="S0 | S1", entity_level="sample", support=SupportType.DETERMINISTICALLY_DERIVED,
        )
    for sample in samples[2:]:
        _fact(
            db_session, study, entity=sample, field="biological_rep_relation",
            value="S2 | S3 | S4 | S5", entity_level="sample", support=SupportType.DETERMINISTICALLY_DERIVED,
        )
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    value = db_session.query(StandardizedValue).filter_by(
        study_id=study.study_id,
        entity_id=None,
        target_field="biological_rep",
    ).one()
    assert value.standardized_value == "2-4"


def test_biological_rep_is_zero_when_no_replicate_group_is_found(db_session):
    """No biological_rep_relation evidence anywhere for the study -> "0",
    per an explicit user request, rather than leaving the field blank."""
    study = _study(db_session, title="No replicate evidence")
    sample = Entity(study_id=study.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="S0")
    db_session.add(sample)
    db_session.flush()
    db_session.add(_home_entity_study(sample))
    _fact(db_session, study, entity=sample, field="geo_loc_name", value="USA: California", entity_level="sample")
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    value = db_session.query(StandardizedValue).filter_by(
        study_id=study.study_id,
        entity_id=None,
        target_field="biological_rep",
    ).one()
    assert value.standardized_value == "0"
    assert value.review_required is False


def test_maps_depth_aliases_and_control_sample_category(db_session):
    study = _study(db_session, title="Depth aliases and controls")
    sample = Entity(study_id=study.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMN_CONTROL")
    db_session.add(sample)
    db_session.flush()
    _fact(db_session, study, entity=sample, field="Depth", value="12 m", entity_level="sample")
    _fact(db_session, study, entity=sample, field="sample_type", value="extraction blank negative control", entity_level="sample")
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    values = {
        sv.target_field: sv
        for sv in db_session.query(StandardizedValue).filter_by(study_id=study.study_id, entity_id=sample.entity_id)
    }
    assert values["minimumDepthInMeters"].standardized_value == "12"
    assert values["maximumDepthInMeters"].standardized_value == "12"
    assert values["samp_category"].standardized_value == "negative control"
    assert values["samp_category"].review_required

    project_values = {
        sv.target_field: sv
        for sv in db_session.query(StandardizedValue).filter_by(study_id=study.study_id, entity_id=None)
    }
    assert project_values["neg_cont_0_1"].standardized_value == "1"
    assert project_values["neg_cont_0_1"].mapping_method == "deterministic_synonym"
    assert project_values["neg_cont_0_1"].review_required is False


def test_maps_structured_control_type_facts_to_project_control_flags(db_session):
    study = _study(db_session, title="Structured control flags")
    sample = Entity(study_id=study.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMN_CONTROL")
    db_session.add(sample)
    db_session.flush()
    _fact(db_session, study, entity=sample, field="neg_cont_type", value="PCR negative", entity_level="sample")
    _fact(db_session, study, entity=sample, field="pos_cont_type", value="mock community", entity_level="sample")
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    project_values = {
        sv.target_field: sv
        for sv in db_session.query(StandardizedValue).filter_by(study_id=study.study_id, entity_id=None)
    }
    assert project_values["neg_cont_0_1"].standardized_value == "1"
    assert project_values["pos_cont_0_1"].standardized_value == "1"
    assert project_values["neg_cont_0_1"].review_required is False
    assert project_values["pos_cont_0_1"].review_required is False


def test_maps_sample_level_biosample_attributes_not_previously_covered(db_session):
    """elev/samp_collect_device/samp_size/samp_size_unit/temp/salinity/ph
    all arrive through the exact same NCBI BioSample
    Attributes/Attribute passthrough as collection_date/depth/geo_loc_name
    above (sources/ncbi.py) -- these rules just hadn't been added yet."""
    study = _study(db_session, title="More BioSample attributes")
    sample = Entity(study_id=study.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMN1")
    db_session.add(sample)
    db_session.flush()
    _fact(db_session, study, entity=sample, field="elev", value="1200", entity_level="sample")
    _fact(db_session, study, entity=sample, field="samp_collect_device", value="Niskin bottle", entity_level="sample")
    _fact(db_session, study, entity=sample, field="samp_size", value="500", entity_level="sample")
    _fact(db_session, study, entity=sample, field="samp_size_unit", value="mL", entity_level="sample")
    _fact(db_session, study, entity=sample, field="temp", value="18.5", entity_level="sample")
    _fact(db_session, study, entity=sample, field="salinity", value="35", entity_level="sample")
    _fact(db_session, study, entity=sample, field="ph", value="8.1", entity_level="sample")
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    values = {
        sv.target_field: sv.standardized_value
        for sv in db_session.query(StandardizedValue).filter_by(study_id=study.study_id, entity_id=sample.entity_id)
    }
    assert values["elev"] == "1200"
    assert values["samp_collect_device"] == "Niskin bottle"
    assert values["samp_size"] == "500"
    assert values["samp_size_unit"] == "mL"
    assert values["temp"] == "18.5"
    assert values["salinity"] == "35"
    assert values["ph"] == "8.1"


def test_maps_in_situ_temp_salinity_to_sample_metadata_fields_at_study_level(db_session):
    """Real audit (10.1093/ismejo/wrae013, STUDY-295abf4a8f43): the paper's
    own text reports temperature/salinity measured at the time of sample
    collection -- the first TEXT-based source for these fields (previously
    structured-BioSample-only). STUDY-level since one collection event's
    in-situ reading is typically reported once for the whole site, not per
    sample."""
    study = _study(db_session, title="In-situ measurements from text")
    _fact(db_session, study, field="in_situ_temp", value="6.5C", entity_level="study", support=SupportType.EXPLICIT)
    _fact(db_session, study, field="in_situ_salinity", value="6.4 PSU", entity_level="study", support=SupportType.EXPLICIT)
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    values = {
        sv.target_field: sv.standardized_value
        for sv in db_session.query(StandardizedValue).filter_by(study_id=study.study_id, entity_id=None)
    }
    assert values["temp"] == "6.5C"
    assert values["salinity"] == "6.4 PSU"


# Real FAIRe Environment-subset fields deliberately dropped entirely per
# an explicit, repeated user request -- removed from
# _ADDITIONAL_ENVIRONMENTAL_SAMPLE_ATTRIBUTES (mapping/rules.py) and
# suppressed from export (exports/faire.py's SAMPLE_METADATA_SUPPRESSED_
# FIELDS), so no SAMPLE-level rule is expected for them anymore.
# tot_depth_water_col was on this list too, but was restored (real audit,
# STUDY-295abf4a8f43): it's the correct FAIRe home for a sediment/soil
# sample's site water depth, a genuinely different concept from
# minimumDepthInMeters/maximumDepthInMeters's own within-sediment depth --
# see extraction/api_verification.py.
_DELIBERATELY_UNMAPPED_ENVIRONMENT_FIELDS = frozenset(
    {
        "diss_oxygen", "nitro", "nitro_unit", "org_carb", "org_nitro",
        "tot_inorg_nitro", "tot_nitro_cont_meth", "tot_nitro_content", "tot_org_c_meth",
        # Removed entirely per an explicit user request ("negligible...
        # don't want to waste compute on them or clutter the code with
        # them").
        "alt", "humidity", "light_intensity", "samp_weather", "solar_irradiance",
    }
)


def test_every_faire_environment_field_has_a_sample_level_rule():
    """Systematic guard: every FAIRe field tagged in_subset: Environment in
    the vendored schema (70 total) must be reachable *as a target_field* by
    some SAMPLE-level rule -- either an explicit rule above
    (minimumDepthInMeters/maximumDepthInMeters via "depth", elev/temp/
    salinity/ph via their own name) or the generated
    _ADDITIONAL_ENVIRONMENTAL_SAMPLE_ATTRIBUTES batch, except the handful
    deliberately dropped entirely (_DELIBERATELY_UNMAPPED_ENVIRONMENT_
    FIELDS). Catches a future schema update silently adding a new
    environmental field this table never learns about."""
    import yaml

    from fair_ocean_agent.database.enums import EntityLevel as _EntityLevel

    with open("schemas/faire/schema.yaml") as f:
        schema = yaml.safe_load(f)
    env_fields = {
        name for name, defn in schema.get("slots", {}).items() if defn and "Environment" in (defn.get("in_subset") or [])
    }
    sample_level_target_fields = {
        rule.target_field for rule in RULES
        if rule.source_entity_level == _EntityLevel.SAMPLE.value
    }
    missing = env_fields - sample_level_target_fields - _DELIBERATELY_UNMAPPED_ENVIRONMENT_FIELDS
    assert not missing, f"FAIRe Environment fields with no SAMPLE-level rule: {sorted(missing)}"


def test_maps_a_sample_of_the_generated_environmental_attributes(db_session):
    study = _study(db_session, title="Additional environmental attributes")
    sample = Entity(study_id=study.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMN1")
    db_session.add(sample)
    db_session.flush()
    _fact(db_session, study, entity=sample, field="chlorophyll", value="0.8 mg/m3", entity_level="sample")
    _fact(db_session, study, entity=sample, field="host_species", value="Thunnus albacares", entity_level="sample")
    _fact(db_session, study, entity=sample, field="tot_nitro", value="2.3", entity_level="sample")
    _fact(db_session, study, entity=sample, field="tot_nitro_unit", value="mg/L", entity_level="sample")
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    values = {
        sv.target_field: sv.standardized_value
        for sv in db_session.query(StandardizedValue).filter_by(study_id=study.study_id, entity_id=sample.entity_id)
    }
    assert values["chlorophyll"] == "0.8 mg/m3"
    assert values["host_species"] == "Thunnus albacares"
    assert values["tot_nitro"] == "2.3 mg/L"
    assert "tot_nitro_unit" not in values


def test_generated_environmental_attribute_names_have_no_duplicates():
    names = [field for field, _ in _ADDITIONAL_ENVIRONMENTAL_SAMPLE_ATTRIBUTES]
    assert len(names) == len(set(names))


def test_splits_fused_primer_sequence_into_clean_primer_and_adapter(db_session):
    """Real audit (10.7717/peerj.333, STUDY-9b31d2733994): a second
    (indexing) PCR's forward primer is the study's own PCR1 primer with a
    454-Titanium sequencing adapter fused on -- forward_primer_sequence
    facts contain BOTH the clean 20nt PCR1 primer and the 49nt PCR2 fusion
    oligo that contains it as a substring. The paper's own text confirms
    this ("underlined stretch matches SP-F-30 primer") but never once uses
    the word "adapter", so the keyword-based adapter_forward mechanism
    never fires -- this is detected by substring instead."""
    study = _study(db_session, title="Fused primer study")
    _fact(
        db_session, study, field="forward_primer_sequence", value="TCTCAAAGACTAAGCCATGC",
        entity_level="study", support=SupportType.EXPLICIT,
    )
    _fact(
        db_session, study, field="forward_primer_sequence",
        value="CCTATCCCCTGTGTGCCTTGGCAGTCTCAG TCTCAAAGACTAAGCCATGC",
        entity_level="study", support=SupportType.EXPLICIT,
    )
    _fact(
        db_session, study, field="reverse_primer_sequence", value="TTACAGAGCTGGAATTACCG",
        entity_level="study", support=SupportType.EXPLICIT,
    )
    _fact(
        db_session, study, field="reverse_primer_sequence",
        value="CCATCTCATCCCTGCGTGTCTCCGACTCAG TACT TTACAGAGCTGGAATTACCG",
        entity_level="study", support=SupportType.EXPLICIT,
    )
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    values = {
        row.target_field: row.standardized_value
        for row in db_session.query(StandardizedValue).filter_by(study_id=study.study_id, entity_id=None)
        if row.target_field in ("pcr_primer_forward", "pcr_primer_reverse", "adapter_forward", "adapter_reverse")
    }
    assert values["pcr_primer_forward"] == "TCTCAAAGACTAAGCCATGC"
    assert values["adapter_forward"] == "CCTATCCCCTGTGTGCCTTGGCAGTCTCAG"
    assert values["pcr_primer_reverse"] == "TTACAGAGCTGGAATTACCG"
    assert values["adapter_reverse"] == "CCATCTCATCCCTGCGTGTCTCCGACTCAGTACT"


def test_splits_fused_primer_sequence_across_a_mistagged_assay_row_too(db_session):
    """Same real study as above, but faithfully reproducing the exact
    entity split found live: PCR1's clean primer broadcasts (entity_id
    None) while PCR2's fused primer landed on its own ASSAY-tagged row
    (the extraction model mistakenly gave the second PCR of the SAME
    assay its own assay_tag). The fused value must be split wherever it
    actually ends up, not just on the broadcast row -- otherwise the
    per-assay projectMetadata row (the one real FAIRe's own export layout
    would actually keep) still shows the raw, unsplit fusion oligo."""
    study = _study(db_session, title="Fused primer study, assay-tagged PCR2")
    assay = Entity(study_id=study.study_id, entity_level=EntityLevel.ASSAY.value, external_identifier="18S-V3V4")
    db_session.add(assay)
    db_session.flush()
    _fact(
        db_session, study, field="forward_primer_sequence", value="TCTCAAAGACTAAGCCATGC",
        entity_level="study", support=SupportType.EXPLICIT,
    )
    _fact(
        db_session, study, entity=assay, field="forward_primer_sequence",
        value="CCTATCCCCTGTGTGCCTTGGCAGTCTCAG TCTCAAAGACTAAGCCATGC",
        entity_level="assay", support=SupportType.EXPLICIT,
    )
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    rows = {
        row.entity_id: row.standardized_value
        for row in db_session.query(StandardizedValue).filter_by(
            study_id=study.study_id, target_field="pcr_primer_forward"
        )
    }
    assert rows[None] == "TCTCAAAGACTAAGCCATGC"
    assert rows[assay.entity_id] == "TCTCAAAGACTAAGCCATGC"

    # Only the assay row ever saw the fused value, so only it gets an
    # adapter_forward -- the broadcast row's own primer fact was already
    # clean and is left untouched (nothing to split there).
    adapter_rows = {
        row.entity_id: row.standardized_value
        for row in db_session.query(StandardizedValue).filter_by(
            study_id=study.study_id, target_field="adapter_forward"
        )
    }
    assert adapter_rows == {assay.entity_id: "CCTATCCCCTGTGTGCCTTGGCAGTCTCAG"}


def test_does_not_split_two_genuinely_different_primers_with_no_substring_relationship(db_session):
    """Two real, independently-designed primers of ordinary length must
    never be treated as a fusion pair just because both exist -- only an
    actual substring relationship (one sequence embedded in the other)
    triggers the split."""
    study = _study(db_session, title="Two distinct primers, no fusion")
    _fact(
        db_session, study, field="forward_primer_sequence", value="GTGYCAGCMGCCGCGGTAA",
        entity_level="study", support=SupportType.EXPLICIT,
    )
    _fact(
        db_session, study, field="forward_primer_sequence", value="CCTACGGGNGGCWGCAG",
        entity_level="study", support=SupportType.EXPLICIT,
    )
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    assert (
        db_session.query(StandardizedValue)
        .filter_by(study_id=study.study_id, target_field="adapter_forward")
        .first()
        is None
    )


def test_maps_run_level_fastq_and_checksum_and_lib_layout(db_session):
    """ENA's fastq_ftp/fastq_md5 (';'-joined forward;reverse pairs) split
    into FAIRe's filename/filename2/checksum_filename/checksum_filename2;
    checksum_method is inferred as a constant ("MD5" -- ENA never reports
    another algorithm); lib_layout is derived from the FASTQ file count."""
    study = _study(db_session, title="Run-level file facts")
    run = Entity(study_id=study.study_id, entity_level=EntityLevel.SEQUENCING_RUN.value, external_identifier="SRR1")
    db_session.add(run)
    db_session.flush()
    _fact(
        db_session, study, entity=run, entity_level="sequencing_run", field="fastq_ftp",
        value="ftp.sra.ebi.ac.uk/vol1/fastq/SRR001/SRR1_1.fastq.gz;ftp.sra.ebi.ac.uk/vol1/fastq/SRR001/SRR1_2.fastq.gz",
    )
    _fact(db_session, study, entity=run, entity_level="sequencing_run", field="fastq_access_status", value="accessible")
    _fact(
        db_session, study, entity=run, entity_level="sequencing_run", field="fastq_md5",
        value="aaa111;bbb222",
    )
    _fact(db_session, study, entity=run, entity_level="sequencing_run", field="read_count", value="1000000")
    _fact(db_session, study, entity=run, entity_level="sequencing_run", field="library_layout", value="SINGLE")
    _fact(db_session, study, entity=run, entity_level="sequencing_run", field="run_accession", value="SRR1")
    _fact(db_session, study, entity=run, entity_level="sequencing_run", field="sample_accession", value="SAMN1")
    _fact(db_session, study, entity=run, entity_level="sequencing_run", field="instrument_platform", value="ILLUMINA")
    _fact(db_session, study, entity=run, entity_level="sequencing_run", field="assay_name", value="16S metabarcoding")
    _fact(db_session, study, entity=run, entity_level="sequencing_run", field="pcr_plate_id", value="plate-1")
    _fact(db_session, study, entity=run, entity_level="sequencing_run", field="lib_id", value="lib-1")
    _fact(db_session, study, entity=run, entity_level="sequencing_run", field="phix_perc", value="15")
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    values = {sv.target_field: sv for sv in db_session.query(StandardizedValue).filter_by(study_id=study.study_id)}
    assert values["filename"].standardized_value == "SRR1_1.fastq.gz"
    assert values["filename2"].standardized_value == "SRR1_2.fastq.gz"
    assert values["fastq_access_status"].standardized_value == "accessible"
    assert values["checksum_filename"].standardized_value == "aaa111"
    assert values["checksum_filename2"].standardized_value == "bbb222"
    assert values["checksum_method"].standardized_value == "MD5"
    assert not values["checksum_method"].review_required
    assert values["input_read_count"].standardized_value == "1000000"
    assert values["lib_layout"].standardized_value == "paired end"
    assert not values["lib_layout"].review_required
    assert values["seq_run_id"].standardized_value == "SRR1"
    assert values["samp_name"].standardized_value == "SAMN1"
    assert values["platform"].standardized_value == "ILLUMINA"
    assert values["assay_name"].standardized_value == "16S metabarcoding"
    assert values["pcr_plate_id"].standardized_value == "plate-1"
    assert values["lib_id"].standardized_value == "lib-1"
    assert values["phix_perc"].standardized_value == "15"
    assert "lib_conc" not in values


def test_inaccessible_fastq_files_do_not_prove_library_layout(db_session):
    study = _study(db_session, title="Inaccessible files")
    run = Entity(study_id=study.study_id, entity_level=EntityLevel.SEQUENCING_RUN.value, external_identifier="SRR1")
    db_session.add(run)
    db_session.flush()
    _fact(
        db_session,
        study,
        entity=run,
        entity_level="sequencing_run",
        field="fastq_ftp",
        value="ftp.sra.ebi.ac.uk/vol1/fastq/SRR001/SRR1_1.fastq.gz;ftp.sra.ebi.ac.uk/vol1/fastq/SRR001/SRR1_2.fastq.gz",
    )
    _fact(db_session, study, entity=run, entity_level="sequencing_run", field="fastq_access_status", value="not_accessible")
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    values = {
        sv.target_field: sv
        for sv in db_session.query(StandardizedValue).filter_by(study_id=study.study_id)
    }
    assert values["filename"].standardized_value == "SRR1_1.fastq.gz"
    assert values["filename2"].standardized_value == "SRR1_2.fastq.gz"
    assert values["fastq_access_status"].standardized_value == "not_accessible"
    assert values["lib_layout"].standardized_value == "no files"
    assert "lib_conc_unit" not in values
    assert "lib_conc_meth" not in values
    assert "mid_forward" not in values
    assert "mid_reverse" not in values


def test_single_end_run_gets_no_filename2_or_checksum_filename2(db_session):
    """A single-end run's fastq_ftp/fastq_md5 has only one ';'-free entry --
    filename2/checksum_filename2 must not be fabricated from nothing."""
    study = _study(db_session, title="Single-end run")
    run = Entity(study_id=study.study_id, entity_level=EntityLevel.SEQUENCING_RUN.value, external_identifier="SRR1")
    db_session.add(run)
    db_session.flush()
    _fact(db_session, study, entity=run, entity_level="sequencing_run", field="fastq_ftp",
          value="ftp.sra.ebi.ac.uk/vol1/fastq/SRR001/SRR1.fastq.gz")
    _fact(db_session, study, entity=run, entity_level="sequencing_run", field="fastq_md5", value="ccc333")
    _fact(db_session, study, entity=run, entity_level="sequencing_run", field="library_layout", value="PAIRED")
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    values = {sv.target_field: sv.standardized_value for sv in db_session.query(StandardizedValue).filter_by(study_id=study.study_id)}
    assert values["filename"] == "SRR1.fastq.gz"
    assert "filename2" not in values
    assert values["checksum_filename"] == "ccc333"
    assert "checksum_filename2" not in values
    assert values["lib_layout"] == "single end"


def test_run_level_file_facts_get_one_row_per_run_not_collapsed(db_session):
    """Regression guard for a real bug this exact change surfaced:
    _resolve_entity_id only ever resolved a real entity_id for SAMPLE-level
    facts -- every SEQUENCING_RUN-level fact fell through to the study-wide
    broadcast default (entity_id=None), which is correct for
    instrument_platform/instrument_model (expected to agree across a
    study's runs, mapped onto projectMetadata) but was silently discarding
    500 of 500 real runs' distinct filenames down to one row the moment a
    genuinely per-run field (filename/checksum_filename/input_read_count,
    mapped onto experimentRunMetadata) got its first mapping rule. Checked
    directly against the real 101-study database before this fix: only 1
    filename/checksum_filename/input_read_count row existed total, despite
    500 real per-run fastq_ftp/fastq_md5/read_count facts."""
    study = _study(db_session, title="Two distinct runs")
    run_a = Entity(study_id=study.study_id, entity_level=EntityLevel.SEQUENCING_RUN.value, external_identifier="SRR_A")
    run_b = Entity(study_id=study.study_id, entity_level=EntityLevel.SEQUENCING_RUN.value, external_identifier="SRR_B")
    db_session.add_all([run_a, run_b])
    db_session.flush()
    _fact(db_session, study, entity=run_a, entity_level="sequencing_run", field="fastq_ftp", value="host/SRR_A.fastq.gz")
    _fact(db_session, study, entity=run_a, entity_level="sequencing_run", field="read_count", value="1000")
    _fact(db_session, study, entity=run_b, entity_level="sequencing_run", field="fastq_ftp", value="host/SRR_B.fastq.gz")
    _fact(db_session, study, entity=run_b, entity_level="sequencing_run", field="read_count", value="2000")
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    filenames = {
        sv.entity_id: sv.standardized_value
        for sv in db_session.query(StandardizedValue).filter_by(study_id=study.study_id, target_field="filename")
    }
    read_counts = {
        sv.entity_id: sv.standardized_value
        for sv in db_session.query(StandardizedValue).filter_by(study_id=study.study_id, target_field="input_read_count")
    }
    experiment_ids = {
        entity.entity_id
        for entity in db_session.query(Entity).filter_by(
            study_id=study.study_id,
            entity_level=EntityLevel.EXPERIMENT_RUN.value,
        )
    }
    assert set(filenames) == experiment_ids
    assert set(read_counts) == experiment_ids
    assert set(filenames.values()) == {"SRR_A.fastq.gz", "SRR_B.fastq.gz"}
    assert set(read_counts.values()) == {"1000", "2000"}
    assert not ({run_a.entity_id, run_b.entity_id} & set(filenames))


def test_run_level_checksum_method_and_fastq_derived_lib_layout_still_collapse_project_wide(db_session):
    """Unlike filename/checksum_filename/input_read_count above,
    checksum_method and lib_layout map onto projectMetadata (not
    experimentRunMetadata). lib_layout is derived from fastq_ftp file
    counts, not ENA's declared library_layout field."""
    study = _study(db_session, title="Two runs, same layout")
    run_a = Entity(study_id=study.study_id, entity_level=EntityLevel.SEQUENCING_RUN.value, external_identifier="SRR_A")
    run_b = Entity(study_id=study.study_id, entity_level=EntityLevel.SEQUENCING_RUN.value, external_identifier="SRR_B")
    db_session.add_all([run_a, run_b])
    db_session.flush()
    _fact(db_session, study, entity=run_a, entity_level="sequencing_run", field="fastq_md5", value="aaa")
    _fact(db_session, study, entity=run_a, entity_level="sequencing_run", field="fastq_ftp", value="host/SRR_A_1.fastq.gz;host/SRR_A_2.fastq.gz")
    _fact(db_session, study, entity=run_a, entity_level="sequencing_run", field="library_layout", value="SINGLE")
    _fact(db_session, study, entity=run_b, entity_level="sequencing_run", field="fastq_md5", value="bbb")
    _fact(db_session, study, entity=run_b, entity_level="sequencing_run", field="fastq_ftp", value="host/SRR_B_1.fastq.gz;host/SRR_B_2.fastq.gz")
    _fact(db_session, study, entity=run_b, entity_level="sequencing_run", field="library_layout", value="SINGLE")
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    checksum_method_rows = db_session.query(StandardizedValue).filter_by(study_id=study.study_id, target_field="checksum_method").all()
    lib_layout_rows = db_session.query(StandardizedValue).filter_by(study_id=study.study_id, target_field="lib_layout").all()
    assert len(checksum_method_rows) == 1
    assert checksum_method_rows[0].entity_id is None
    assert len(lib_layout_rows) == 1
    assert lib_layout_rows[0].entity_id is None
    assert lib_layout_rows[0].standardized_value == "paired end"
    assert lib_layout_rows[0].review_required is False


def test_mixed_fastq_derived_lib_layout_prefers_paired_end_and_flags_review(db_session):
    study = _study(db_session, title="Mixed FASTQ file layout")
    run_single = Entity(
        study_id=study.study_id,
        entity_level=EntityLevel.SEQUENCING_RUN.value,
        external_identifier="SRR_SINGLE",
    )
    run_paired = Entity(
        study_id=study.study_id,
        entity_level=EntityLevel.SEQUENCING_RUN.value,
        external_identifier="SRR_PAIRED",
    )
    db_session.add_all([run_single, run_paired])
    db_session.flush()
    single_fact = _fact(
        db_session,
        study,
        entity=run_single,
        entity_level="sequencing_run",
        field="fastq_ftp",
        value="host/SRR_SINGLE.fastq.gz",
    )
    paired_fact = _fact(
        db_session,
        study,
        entity=run_paired,
        entity_level="sequencing_run",
        field="fastq_ftp",
        value="host/SRR_PAIRED_1.fastq.gz;host/SRR_PAIRED_2.fastq.gz",
    )
    _fact(db_session, study, entity=run_single, entity_level="sequencing_run", field="library_layout", value="PAIRED")
    _fact(db_session, study, entity=run_paired, entity_level="sequencing_run", field="library_layout", value="SINGLE")
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    lib_layout = db_session.query(StandardizedValue).filter_by(
        study_id=study.study_id,
        target_field="lib_layout",
    ).one()
    evidence_fact_ids = {
        evidence.fact_id
        for evidence in db_session.query(StandardizedValueEvidence).filter_by(
            standardized_value_id=lib_layout.standardized_value_id
        )
    }
    assert lib_layout.standardized_value == "paired end"
    assert lib_layout.review_required is True
    assert evidence_fact_ids == {single_fact.fact_id, paired_fact.fact_id}


def test_lib_layout_no_files_when_no_fastq_facts(db_session):
    study = _study(db_session, title="No downloadable files")
    run = Entity(study_id=study.study_id, entity_level=EntityLevel.SEQUENCING_RUN.value, external_identifier="SRR1")
    db_session.add(run)
    db_session.flush()
    _fact(db_session, study, entity=run, entity_level="sequencing_run", field="library_layout", value="PAIRED")
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    lib_layout = db_session.query(StandardizedValue).filter_by(
        study_id=study.study_id,
        target_field="lib_layout",
    ).one()
    assert lib_layout.standardized_value == "no files"
    assert lib_layout.review_required is False


def test_maps_citation_from_any_repository_adapter_to_bibliographic_citation(db_session):
    """OBIS, GBIF, and PANGAEA all emit this exact field name ("citation")
    at project level -- one rule covers whichever adapter a study
    actually resolved through."""
    study = _study(db_session, title="Repository citation")
    _fact(db_session, study, field="citation", value="Smith et al. 2024. Dataset X. OBIS.", entity_level="project")
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    row = db_session.query(StandardizedValue).filter_by(study_id=study.study_id, target_field="bibliographicCitation").one()
    assert row.standardized_value == "Smith et al. 2024. Dataset X. OBIS."


def test_maps_publication_metadata_facts_to_faire(db_session):
    """extraction/publication_metadata.py's project-level facts use
    literal FAIRe field names at EntityLevel.STUDY and all map through.
    recordedByID is deliberately absent: no longer extracted or mapped at
    all (an explicit user instruction)."""
    study = _study(db_session, title="Publication metadata")
    values = {
        "license": "http://creativecommons.org/licenses/by/3.0/",
        "rightsHolder": "Davies et al.",
        "accessRights": "open access",
        "bibliographicCitation": "Davies SW, et al. (2014). A cross-ocean comparison. PeerJ 2:e333.",
        "associated_resource": "**Methods**: doi: 10.1234/protocol.1",
        "code_repo": "https://github.com/someorg/somerepo",
        "funding_source": "National Science Foundation | Gordon and Betty Moore Foundation",
        "recordedBy": "Sarah W. Davies | Eli Meyer",
        "project_contact": "daviessw@gmail.com",
    }
    for field, value in values.items():
        _fact(db_session, study, field=field, value=value, entity_level="study", support=SupportType.STRUCTURED_SOURCE)
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    rows = {
        sv.target_field: sv.standardized_value
        for sv in db_session.query(StandardizedValue).filter_by(study_id=study.study_id, entity_id=None)
    }
    for field, value in values.items():
        assert rows[field] == value, f"{field}: expected {value!r}, got {rows.get(field)!r}"
    assert (
        db_session.query(StandardizedValue)
        .filter_by(study_id=study.study_id, target_field="associated_resource")
        .one()
        .review_required
        is False
    )
    assert (
        db_session.query(StandardizedValue)
        .filter_by(study_id=study.study_id, target_field="rightsHolder")
        .one()
        .review_required
        is True
    )
    assert (
        db_session.query(StandardizedValue)
        .filter_by(study_id=study.study_id, target_field="funding_source")
        .one()
        .review_required
        is True
    )


def test_maps_crossref_license_json_to_url_and_open_access(db_session):
    study = _study(db_session, title="Crossref license cleanup")
    _fact(
        db_session,
        study,
        field="license",
        value=(
            '[{"start": {"date-parts": [[2024, 1, 31]]}, '
            '"content-version": "vor", "URL": "https://creativecommons.org/licenses/by/4.0/"}]'
        ),
        entity_level="study",
        support=SupportType.STRUCTURED_SOURCE,
    )
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    rows = {
        sv.target_field: sv.standardized_value
        for sv in db_session.query(StandardizedValue).filter_by(study_id=study.study_id, entity_id=None)
    }
    assert rows["license"] == "https://creativecommons.org/licenses/by/4.0/"
    assert rows["accessRights"] == "open access"


def test_maps_repository_dna_derived_fields_to_faire_without_llm_review_flag(db_session):
    study = _study(db_session, title="DNA-derived repository facts")
    for field, value in {
        "associatedSequences": "ENA:ERR123",
        "target_gene": "16S rRNA",
        "pcr_primer_forward": "GTGYCAGCMGCCGCGGTAA",
        "pcr_primer_reverse": "GGACTACNVGGGTWTCTAAT",
        "pcr_primer_name_forward": "515F",
        "pcr_primer_name_reverse": "806R",
        "annealingTemp": "55 C",
        "ampliconSize": "291 bp",
        "assay_name": "16S V4 metabarcoding",
    }.items():
        _fact(db_session, study, field=field, value=value, entity_level="project")
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    values = {
        sv.target_field: sv
        for sv in db_session.query(StandardizedValue).filter_by(study_id=study.study_id)
    }
    assert values["associatedSequences"].target_field == "associatedSequences"
    assert values["target_gene"].standardized_value == "16S rRNA"
    assert values["pcr_primer_forward"].standardized_value == "GTGYCAGCMGCCGCGGTAA"
    assert values["pcr_primer_reverse"].standardized_value == "GGACTACNVGGGTWTCTAAT"
    assert values["pcr_primer_name_forward"].standardized_value == "515F"
    assert values["pcr_primer_name_reverse"].standardized_value == "806R"
    assert values["annealingTemp"].standardized_value == "55 C"
    assert values["ampliconSize"].standardized_value == "291 bp"
    assert values["assay_name"].standardized_value == "16S V4 metabarcoding"
    assert not values["pcr_primer_forward"].review_required


def test_maps_ena_run_accession_and_library_construction_protocol(db_session):
    study = _study(db_session, title="ENA protocol facts")
    run = Entity(study_id=study.study_id, entity_level=EntityLevel.SEQUENCING_RUN.value, external_identifier="ERR123")
    db_session.add(run)
    db_session.flush()
    _fact(db_session, study, entity=run, entity_level="sequencing_run", field="run_accession", value="ERR123")
    _fact(
        db_session,
        study,
        entity=run,
        entity_level="sequencing_run",
        field="library_construction_protocol",
        value="PCR amplified the V4 region with 515F/806R primers.",
    )
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    values = {
        sv.target_field: sv
        for sv in db_session.query(StandardizedValue).filter_by(study_id=study.study_id)
    }
    experiment = db_session.query(Entity).filter_by(
        study_id=study.study_id,
        entity_level=EntityLevel.EXPERIMENT_RUN.value,
    ).one()
    assert values["associatedSequences"].entity_id == experiment.entity_id
    assert values["associatedSequences"].standardized_value == "ERR123"
    assert values["pcr_method_additional"].entity_id is None
    assert values["pcr_method_additional"].review_required is True
    assert values["pcr_method_additional"].standardized_value.startswith("PCR amplified")


def test_dedups_identical_project_wide_facts_across_many_runs(db_session):
    """Real data has this exact shape: 500 sequencing_run raw_facts all
    reporting the same instrument_platform for one study. Must collapse to
    a single project-wide row, not 500 duplicates."""
    study = _study(db_session, title="Many identical runs")
    for i in range(5):
        run = Entity(study_id=study.study_id, entity_level=EntityLevel.SEQUENCING_RUN.value, external_identifier=f"SRR{i}")
        db_session.add(run)
        db_session.flush()
        _fact(db_session, study, entity=run, field="instrument_platform", value="ILLUMINA", entity_level="sequencing_run")
    db_session.commit()

    created = map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    project_rows = db_session.query(StandardizedValue).filter_by(
        study_id=study.study_id, target_field="platform", entity_id=None
    ).all()
    run_rows = (
        db_session.query(StandardizedValue)
        .filter(StandardizedValue.study_id == study.study_id)
        .filter(StandardizedValue.target_field == "platform")
        .filter(StandardizedValue.entity_id.is_not(None))
        .all()
    )
    assert len(project_rows) == 1
    assert len(run_rows) == 0
    assert project_rows[0].standardized_value == "ILLUMINA"
    assert not project_rows[0].review_required
    # +1 each for checkls_ver, informationWithheld, and lib_layout, all
    # synced/defaulted regardless of a study's own facts (mapping/
    # faire.py::_sync_checklist_version, informationWithheld default).
    assert created == 4


def test_pipe_joins_samp_mat_process_from_two_separate_paper_paragraphs(db_session):
    """Grounded in a real gap (10.3389/fmicb.2024.1295149): a storage/
    freeze-drying paragraph and a separate DNA-extraction paragraph each
    independently get classified and extracted as their own sample_prep
    run, producing two distinct STUDY-level samp_mat_process facts. Per an
    explicit user request, both should be kept (pipe-joined), not "first
    wins, second discarded" like a typical broadcast conflict."""
    study = _study(db_session, title="Two processing paragraphs")
    _fact(
        db_session,
        study,
        field="samp_mat_process",
        value="the sub-sectioned sediment samples were freeze-dried in a freeze-dryer",
        entity_level="study",
        support=SupportType.EXPLICIT,
    )
    _fact(
        db_session,
        study,
        field="samp_mat_process",
        value="the Sterivex filter cartridges were cracked open and the filter paper was chipped into small pieces",
        entity_level="study",
        support=SupportType.EXPLICIT,
    )
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    row = db_session.query(StandardizedValue).filter_by(
        study_id=study.study_id, target_field="samp_mat_process", entity_id=None
    ).one()
    assert row.standardized_value == (
        "the sub-sectioned sediment samples were freeze-dried in a freeze-dryer | "
        "the Sterivex filter cartridges were cracked open and the filter paper was chipped into small pieces"
    )


def test_pipe_joins_conflicting_project_wide_platforms_with_review(db_session):
    study = _study(db_session, title="Disagreeing runs")
    run_a = Entity(study_id=study.study_id, entity_level=EntityLevel.SEQUENCING_RUN.value, external_identifier="SRR_A")
    run_b = Entity(study_id=study.study_id, entity_level=EntityLevel.SEQUENCING_RUN.value, external_identifier="SRR_B")
    db_session.add_all([run_a, run_b])
    db_session.flush()
    _fact(db_session, study, entity=run_a, field="instrument_platform", value="ILLUMINA", entity_level="sequencing_run")
    _fact(db_session, study, entity=run_b, field="instrument_platform", value="PACBIO_SMRT", entity_level="sequencing_run")
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    project_rows = db_session.query(StandardizedValue).filter_by(
        study_id=study.study_id, target_field="platform", entity_id=None
    ).all()
    run_rows = (
        db_session.query(StandardizedValue)
        .filter(StandardizedValue.study_id == study.study_id)
        .filter(StandardizedValue.target_field == "platform")
        .filter(StandardizedValue.entity_id.is_not(None))
        .all()
    )
    assert len(project_rows) == 1
    assert project_rows[0].standardized_value == "ILLUMINA | PACBIO_SMRT"
    assert project_rows[0].review_required is True  # but the disagreement isn't silently dropped
    assert len(run_rows) == 0  # platform is project metadata, never a library/run-row field


def test_pipe_joins_two_assays_target_gene_and_primers_instead_of_dropping_the_second(db_session):
    """Real gap found live via the LLM troubleshooting batch: a real study
    (10.3390/microorganisms10030558) ran two genuinely separate assays --
    a 16S-V3V4 amplicon assay and a separate cbbL-gene assay, each with its
    own primers -- but before target_gene/target_subfragment/primer_forward/
    primer_reverse/assay_name/assay_type were added to
    _PIPE_UNION_TARGET_FIELDS, whichever assay's facts were processed
    first simply won the broadcast target_field outright, silently
    dropping the second assay's target gene and primers entirely. Per an
    explicit user request, both should show up pipe-joined instead."""
    study = _study(db_session, title="Two assays, one study")
    _fact(db_session, study, field="target_gene", value="16S rRNA", entity_level="project")
    _fact(db_session, study, field="target_gene", value="cbbL", entity_level="project")
    _fact(db_session, study, field="pcr_primer_name_forward", value="338F", entity_level="project")
    _fact(db_session, study, field="pcr_primer_name_forward", value="cbbL_K2f", entity_level="project")
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    target_gene = db_session.query(StandardizedValue).filter_by(study_id=study.study_id, target_field="target_gene").one()
    primer_name_forward = db_session.query(StandardizedValue).filter_by(study_id=study.study_id, target_field="pcr_primer_name_forward").one()
    assert target_gene.standardized_value == "16S rRNA | cbbL"
    assert primer_name_forward.standardized_value == "338F | cbbL_K2f"


def test_pipe_joins_conflicting_project_wide_instruments_with_review(db_session):
    study = _study(db_session, title="Two sequencing instruments")
    run_a = Entity(study_id=study.study_id, entity_level=EntityLevel.SEQUENCING_RUN.value, external_identifier="SRR_A")
    run_b = Entity(study_id=study.study_id, entity_level=EntityLevel.SEQUENCING_RUN.value, external_identifier="SRR_B")
    db_session.add_all([run_a, run_b])
    db_session.flush()
    _fact(
        db_session,
        study,
        entity=run_a,
        field="instrument_model",
        value="Ion Torrent PGM",
        entity_level="sequencing_run",
    )
    _fact(
        db_session,
        study,
        entity=run_b,
        field="instrument_model",
        value="Illumina HiSeq 2500",
        entity_level="sequencing_run",
    )
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    project_row = db_session.query(StandardizedValue).filter_by(
        study_id=study.study_id, target_field="instrument", entity_id=None
    ).one()
    assert project_row.standardized_value == "Ion Torrent PGM | Illumina HiSeq 2500"
    assert project_row.review_required is True


def test_assay_tagged_facts_get_separate_projectmetadata_rows_not_a_conflict(db_session):
    """Two distinct assays' annealing_temperature facts (extraction/text.py's
    assay_tag) must not collide into one projectMetadata row the way two
    untagged conflicting facts do (see
    test_pipe_joins_conflicting_project_wide_platforms_with_review above) --
    each assay gets its own row instead, since a paper can genuinely
    describe more than one assay run on the same samples."""
    study = _study(db_session, title="Two assays")
    assay_16s = Entity(study_id=study.study_id, entity_level=EntityLevel.ASSAY.value, external_identifier="16S-V3V4")
    assay_18s = Entity(study_id=study.study_id, entity_level=EntityLevel.ASSAY.value, external_identifier="18S-V9")
    db_session.add_all([assay_16s, assay_18s])
    db_session.flush()
    _fact(db_session, study, entity=assay_16s, field="annealing_temperature", value="55C", entity_level="assay", support=SupportType.EXPLICIT)
    _fact(db_session, study, entity=assay_18s, field="annealing_temperature", value="60C", entity_level="assay", support=SupportType.EXPLICIT)
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    rows = db_session.query(StandardizedValue).filter_by(study_id=study.study_id, target_field="annealingTemp").all()
    # Two separate rows, not one collapsed/conflicting row -- this is the
    # real fix: without per-assay entity_ids these would collide onto a
    # single (target_table, target_field, None) key and the second value
    # would be lost, only flagged review_required as a "conflict" (see
    # test_pipe_joins_conflicting_project_wide_platforms_with_review above).
    assert len(rows) == 2
    assert {row.entity_id for row in rows} == {assay_16s.entity_id, assay_18s.entity_id}
    values = {row.entity_id: row.standardized_value for row in rows}
    assert values[assay_16s.entity_id] == "55C"
    assert values[assay_18s.entity_id] == "60C"


def test_project_level_structured_assay_name_still_broadcasts_unaffected(db_session):
    """Regression guard: the new ASSAY-level entity resolution only applies
    to fact.entity_level == EntityLevel.ASSAY -- a PROJECT-level structured
    fact (e.g. OBIS/GBIF-sourced assay_name) must keep broadcasting via
    entity_id=None exactly as before this change."""
    study = _study(db_session, title="Structured assay_name")
    _fact(db_session, study, field="assay_name", value="16S-V4", entity_level="project")
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    rows = db_session.query(StandardizedValue).filter_by(study_id=study.study_id, target_field="assay_name").all()
    assert len(rows) == 1
    assert rows[0].entity_id is None


def test_assay_name_mapping_drops_primer_pairs_and_bare_functional_genes(db_session):
    study = _study(db_session, title="Assay name cleanup")
    _fact(
        db_session,
        study,
        field="assay_name",
        value="16S | hzsA | 515F/806R | hzsA_1597A/hzsA_1857R",
        entity_level="study",
        support=SupportType.EXPLICIT,
    )
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    value = db_session.query(StandardizedValue).filter_by(
        study_id=study.study_id, target_field="assay_name"
    ).one()
    assert value.standardized_value == "16S"


def test_assay_name_mapping_normalizes_rRNA_region_dash_mojibake(db_session):
    study = _study(db_session, title="Assay name dash cleanup")
    _fact(
        db_session,
        study,
        field="assay_name",
        value="16S rRNA-V3\u7ab6\u5929V4",
        entity_level="study",
        support=SupportType.EXPLICIT,
    )
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    value = db_session.query(StandardizedValue).filter_by(
        study_id=study.study_id, target_field="assay_name"
    ).one()
    assert value.standardized_value == "16S-V3-V4"


def test_rejected_facts_are_excluded_from_mapping_entirely(db_session):
    """review_status=REJECTED (the quarantine mechanism for facts extracted
    under a since-fixed bug, or from a superseded model/prompt version)
    must actually be excluded from mapping, not merely deprioritized --
    otherwise quarantining a fact has no real effect on exported data."""
    study = _study(db_session, title="Quarantined fact")
    run = Entity(study_id=study.study_id, entity_level=EntityLevel.SEQUENCING_RUN.value, external_identifier="SRR1")
    db_session.add(run)
    db_session.flush()
    rejected = _fact(db_session, study, entity=run, field="instrument_platform", value="ILLUMINA", entity_level="sequencing_run")
    rejected.review_status = "rejected"
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    assert db_session.query(StandardizedValue).filter_by(
        study_id=study.study_id, target_field="platform"
    ).count() == 0


def test_non_rejected_fact_wins_over_a_quarantined_conflicting_one(db_session):
    study = _study(db_session, title="One quarantined, one good")
    run_a = Entity(study_id=study.study_id, entity_level=EntityLevel.SEQUENCING_RUN.value, external_identifier="SRR_A")
    run_b = Entity(study_id=study.study_id, entity_level=EntityLevel.SEQUENCING_RUN.value, external_identifier="SRR_B")
    db_session.add_all([run_a, run_b])
    db_session.flush()
    stale = _fact(db_session, study, entity=run_a, field="instrument_platform", value="PACBIO_SMRT", entity_level="sequencing_run")
    stale.review_status = "rejected"
    _fact(db_session, study, entity=run_b, field="instrument_platform", value="ILLUMINA", entity_level="sequencing_run")
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    project_rows = db_session.query(StandardizedValue).filter_by(study_id=study.study_id, target_field="platform").all()
    assert len(project_rows) == 1
    assert project_rows[0].standardized_value == "ILLUMINA"
    assert project_rows[0].review_required is False  # the quarantined fact never entered the comparison


def test_flags_review_required_when_value_fails_closed_vocab_check(db_session):
    study = _study(db_session, title="Bad platform value")
    run = Entity(study_id=study.study_id, entity_level=EntityLevel.SEQUENCING_RUN.value, external_identifier="SRR1")
    db_session.add(run)
    db_session.flush()
    _fact(db_session, study, entity=run, field="instrument_platform", value="SOME_UNKNOWN_PLATFORM", entity_level="sequencing_run")
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    project_row = db_session.query(StandardizedValue).filter_by(
        study_id=study.study_id, target_field="platform", entity_id=None
    ).one()
    assert project_row.review_required is True


def test_sample_accession_redirects_to_matching_sample_entity_not_the_run(db_session):
    study = _study(db_session, title="materialSampleID redirect")
    sample = Entity(study_id=study.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMN999")
    run = Entity(study_id=study.study_id, entity_level=EntityLevel.SEQUENCING_RUN.value, external_identifier="SRR999")
    db_session.add_all([sample, run])
    db_session.flush()
    _fact(db_session, study, entity=run, field="sample_accession", value="SAMN999", entity_level="sequencing_run")
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    row = db_session.query(StandardizedValue).filter_by(study_id=study.study_id, target_field="materialSampleID").one()
    assert row.entity_id == sample.entity_id
    assert row.entity_id != run.entity_id


def test_sample_identifier_defaults_to_material_sample_id_without_run_fact(db_session):
    study = _study(db_session, title="BioSample-only materialSampleID")
    sample = Entity(study_id=study.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMN12415826")
    db_session.add(sample)
    db_session.flush()
    _fact(db_session, study, entity=sample, field="collection_date", value="2014-07-22", entity_level="sample")
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    values = {
        row.target_field: row
        for row in db_session.query(StandardizedValue).filter_by(study_id=study.study_id, entity_id=sample.entity_id)
    }
    assert values["samp_name"].standardized_value == "SAMN12415826"
    assert values["materialSampleID"].standardized_value == "SAMN12415826"
    assert values["materialSampleID"].mapping_method == "exact_identifier"


def test_sample_accession_materializes_referenced_sample_identity(db_session):
    study = _study(db_session, title="No matching sample")
    run = Entity(study_id=study.study_id, entity_level=EntityLevel.SEQUENCING_RUN.value, external_identifier="SRR1")
    db_session.add(run)
    db_session.flush()
    _fact(db_session, study, entity=run, field="sample_accession", value="SAMN_NONEXISTENT", entity_level="sequencing_run")
    db_session.commit()

    created = map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    sample = db_session.query(Entity).filter_by(
        study_id=study.study_id,
        entity_level=EntityLevel.SAMPLE.value,
        external_identifier="SAMN_NONEXISTENT",
    ).one()
    material = db_session.query(StandardizedValue).filter_by(
        study_id=study.study_id,
        target_field="materialSampleID",
    ).one()
    assert material.entity_id == sample.entity_id
    experiment = db_session.query(Entity).filter_by(
        study_id=study.study_id,
        entity_level=EntityLevel.EXPERIMENT_RUN.value,
    ).one()
    assert experiment.parent_entity_id == sample.entity_id
    # +1 each for checkls_ver, informationWithheld, and lib_layout, all
    # synced/defaulted regardless of a study's own facts (mapping/
    # faire.py::_sync_checklist_version, informationWithheld default), plus
    # +1 for biological_rep="0" (mapping/faire.py::
    # _apply_biological_rep_from_relations always writes a value, "0" when
    # no biological_rep_relation evidence exists for the study).
    assert created == 7


def test_llm_blob_fact_is_broadcast_and_flagged_for_review(db_session):
    study = _study(db_session, title="LLM blob fact")
    _fact(
        db_session, study, field="storage_conditions", value="Samples stored at -80C in RNAlater",
        entity_level="study", support=SupportType.EXPLICIT,
    )
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    row = db_session.query(StandardizedValue).filter_by(
        study_id=study.study_id, target_field="samp_store_method_additional"
    ).one()
    assert row.entity_id is None
    assert row.review_required is True
    assert row.standardized_value == "Samples stored at -80C in RNAlater"


def test_llm_study_level_sampling_facts_broadcast_to_sample_metadata(db_session):
    study = _study(db_session, title="LLM sampling facts")
    _fact(
        db_session, study, field="coordinates", value="38.03 N 122.151667 W",
        entity_level="study", support=SupportType.EXPLICIT,
    )
    _fact(
        db_session, study, field="depths", value="5 meters",
        entity_level="study", support=SupportType.EXPLICIT,
    )
    _fact(
        db_session, study, field="internal_expedition_id", value="Malaspina 2010",
        entity_level="study", support=SupportType.EXPLICIT,
    )
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    values = {
        sv.target_field: sv
        for sv in db_session.query(StandardizedValue).filter_by(study_id=study.study_id, entity_id=None)
    }
    assert values["decimalLatitude"].standardized_value == "38.030000"
    assert values["decimalLongitude"].standardized_value == "-122.151667"
    assert values["minimumDepthInMeters"].standardized_value == "5"
    assert values["maximumDepthInMeters"].standardized_value == "5"
    assert values["internal_expedition_id"].standardized_value == "Malaspina 2010"
    # checkls_ver and informationWithheld are always synced/defaulted as
    # confident constants (mapping/faire.py::_sync_checklist_version, the
    # informationWithheld "Nothing indicated as withheld" default), never
    # review_required -- excluded here since this test is about the
    # LLM-derived fields' own review flagging.
    assert all(
        row.review_required is True
        for field, row in values.items()
        if field not in ("checkls_ver", "informationWithheld", "lib_layout")
    )


def test_llm_study_level_geo_loc_name_broadcasts_when_only_a_named_site_is_given(db_session):
    """Real gap found live (PMC10988111): a paper can name a real,
    specific collection site ("Yantai Haichang Whale Shark Ocean Park
    (Shandong, China)") without ever giving numeric coordinates anywhere
    -- the existing coordinates broadcast has nothing to extract in that
    case. Same shape, same review_required=True safety net."""
    study = _study(db_session, title="Named site, no coordinates")
    _fact(
        db_session, study, field="geo_loc_name", value="China: Yantai (Yantai Haichang Whale Shark Ocean Park)",
        entity_level="study", support=SupportType.EXPLICIT,
    )
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    row = db_session.query(StandardizedValue).filter_by(study_id=study.study_id, target_field="geo_loc_name").one()
    assert row.entity_id is None
    assert row.review_required is True
    assert row.standardized_value == "China: Yantai (Yantai Haichang Whale Shark Ocean Park)"


def test_study_level_filter_name_maps_as_sample_broadcast_default(db_session):
    study = _study(db_session, title="Filter name")
    _fact(
        db_session,
        study,
        field="filter_name",
        value="Sterivex filter",
        entity_level="study",
        support=SupportType.EXPLICIT,
    )
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    value = db_session.query(StandardizedValue).filter_by(
        study_id=study.study_id,
        target_field="filter_name",
        entity_id=None,
    ).one()
    assert value.standardized_value == "Sterivex filter"
    assert value.review_required is True


def test_sample_level_isolation_source_maps_to_env_medium(db_session):
    study = _study(db_session, title="Isolation source")
    sample = Entity(study_id=study.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMN1")
    db_session.add(sample)
    db_session.flush()
    _fact(
        db_session,
        study,
        entity=sample,
        field="isolation_source",
        value="coral cue material",
        entity_level="sample",
    )
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    value = db_session.query(StandardizedValue).filter_by(
        study_id=study.study_id,
        entity_id=sample.entity_id,
        target_field="env_medium",
    ).one()
    assert value.standardized_value == "coral cue material"
    assert value.mapping_method == "deterministic_synonym"


def test_sample_level_cruise_or_station_attribute_maps_to_internal_expedition_id(db_session):
    study = _study(db_session, title="Cruise attribute")
    sample = Entity(study_id=study.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMN1")
    db_session.add(sample)
    db_session.flush()
    _fact(
        db_session,
        study,
        entity=sample,
        field="cruise",
        value="Tara Oceans",
        entity_level="sample",
    )
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    value = db_session.query(StandardizedValue).filter_by(
        study_id=study.study_id,
        entity_id=sample.entity_id,
        target_field="internal_expedition_id",
    ).one()
    assert value.standardized_value == "Tara Oceans"
    assert value.mapping_method == "deterministic_synonym"


def test_filter_name_placeholder_is_ignored_when_real_filter_name_exists(db_session):
    study = _study(db_session, title="Filter placeholder")
    _fact(
        db_session,
        study,
        field="filter_name",
        value="see below",
        entity_level="study",
        support=SupportType.EXPLICIT,
    )
    _fact(
        db_session,
        study,
        field="filter_name",
        value="Swinnex 47 mm filter holder",
        entity_level="study",
        support=SupportType.EXPLICIT,
    )
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    value = db_session.query(StandardizedValue).filter_by(
        study_id=study.study_id,
        target_field="filter_name",
        entity_id=None,
    ).one()
    assert value.standardized_value == "Swinnex 47 mm filter holder"


def test_study_level_size_frac_values_are_pipe_union_preserved(db_session):
    """Regression guard for the PLOS plankton-filtration paper: the study
    used a filter cascade (180-um, 5.0-um, 0.2-um), but size_frac was not a
    pipe-union target, so only the first standardized value survived."""
    study = _study(db_session, title="Filter cascade")
    _fact(db_session, study, field="size_frac", value="180-μm", entity_level="study", support=SupportType.EXPLICIT)
    _fact(db_session, study, field="size_frac", value="5.0-μm", entity_level="study", support=SupportType.EXPLICIT)
    _fact(db_session, study, field="size_frac", value="0.2-μm", entity_level="study", support=SupportType.EXPLICIT)
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    value = db_session.query(StandardizedValue).filter_by(
        study_id=study.study_id,
        target_field="size_frac",
        entity_id=None,
    ).one()
    assert value.standardized_value == "180-μm | 5.0-μm | 0.2-μm"


def test_llm_atomic_assay_facts_map_to_faire_protocol_fields_with_review(db_session):
    study = _study(db_session, title="LLM protocol facts")
    _fact(
        db_session, study, field="PCR_forward_primer_sequence", value="1055f",
        entity_level="study", support=SupportType.EXPLICIT,
    )
    _fact(
        db_session, study, field="PCR_reverse_primer_sequence", value="1406r",
        entity_level="study", support=SupportType.EXPLICIT,
    )
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    values = {
        sv.target_field: sv
        for sv in db_session.query(StandardizedValue).filter_by(study_id=study.study_id, entity_id=None)
    }
    assert values["pcr_primer_forward"].standardized_value == "1055f"
    assert values["pcr_primer_reverse"].standardized_value == "1406r"
    assert all(
        row.review_required is True
        for field, row in values.items()
        if field not in ("checkls_ver", "informationWithheld", "lib_layout")
    )


def test_controlled_text_search_project_facts_map_to_faire_with_review(db_session):
    study = _study(db_session, title="Controlled project searches")
    _fact(db_session, study, field="seq_kit", value="MiSeq Reagent Kit v3", entity_level="study")
    _fact(db_session, study, field="probeReporter", value="FAM", entity_level="study")
    _fact(db_session, study, field="probeQuencher", value="BHQ-1 | quencher", entity_level="study")
    _fact(db_session, study, field="commercial_mm", value="TaqMan", entity_level="study")
    _fact(db_session, study, field="sterilise_method", value="Bottles were rinsed with bleach.", entity_level="study")
    _fact(db_session, study, field="assay_type", value="targeted | metabarcoding", entity_level="study")
    _fact(db_session, study, field="barcoding_pcr_appr", value="two-step PCR", entity_level="study")
    _fact(db_session, study, field="adapter_forward", value="AATGATACGGCGACCACCGAGATCTACACGCT", entity_level="study")
    _fact(db_session, study, field="adapter_reverse", value="CAAGCAGAAGACGGCATACGAGAT", entity_level="study")
    _fact(db_session, study, field="checksum_method", value="SHA-256", entity_level="study")
    _fact(db_session, study, field="inhibition_check_0_1", value="1", entity_level="study")
    _fact(db_session, study, field="inhibition_check", value="IPC spike-in with 1:10 dilution", entity_level="study")
    _fact(db_session, study, field="otu_clust_tool", value="VSEARCH --cluster_fast", entity_level="study")
    _fact(db_session, study, field="otu_db", value="SILVA release 138", entity_level="study")
    _fact(
        db_session,
        study,
        field="pcr_0_1",
        value="true",
        entity_level="study",
        support=SupportType.DETERMINISTICALLY_DERIVED,
    )
    _fact(db_session, study, field="neg_cont_0_1", value="1", entity_level="study", support=SupportType.DETERMINISTICALLY_DERIVED)
    _fact(db_session, study, field="pos_cont_0_1", value="0", entity_level="study", support=SupportType.DETERMINISTICALLY_DERIVED)
    _fact(db_session, study, field="sample_type", value="water", entity_level="study")
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    values = {
        sv.target_field: sv
        for sv in db_session.query(StandardizedValue).filter_by(study_id=study.study_id, entity_id=None)
    }
    assert values["seq_kit"].standardized_value == "MiSeq Reagent Kit v3"
    assert "sequencing_location" not in values
    assert values["probeReporter"].standardized_value == "FAM"
    assert values["probeQuencher"].standardized_value == "BHQ-1 | quencher"
    assert values["commercial_mm"].standardized_value == "TaqMan"
    assert values["custom_mm"].standardized_value == "N/A see commercial_mm"
    assert values["sterilise_method"].standardized_value == "Bottles were rinsed with bleach."
    assert values["assay_type"].standardized_value == "targeted | metabarcoding"
    assert values["barcoding_pcr_appr"].standardized_value == "two-step PCR"
    assert values["adapter_forward"].standardized_value == "AATGATACGGCGACCACCGAGATCTACACGCT"
    assert values["adapter_reverse"].standardized_value == "CAAGCAGAAGACGGCATACGAGAT"
    assert values["checksum_method"].standardized_value == "SHA-256"
    assert values["inhibition_check_0_1"].standardized_value == "1"
    assert values["inhibition_check"].standardized_value == "IPC spike-in with 1:10 dilution"
    assert values["otu_clust_tool"].standardized_value == "VSEARCH --cluster_fast"
    assert values["otu_db"].standardized_value == "SILVA release 138"
    assert values["pcr_0_1"].standardized_value == "1"
    assert values["neg_cont_0_1"].standardized_value == "1"
    assert values["pos_cont_0_1"].standardized_value == "0"
    assert "sample_type" not in values
    assert "otu_db_custom" not in values
    assert values["custom_mm"].review_required is False
    assert values["checksum_method"].review_required is False
    non_review_fields = {
        field
        for field, row in values.items()
        if row.review_required is False
        and field
        not in (
            "checksum_method",
            "checkls_ver",
            "informationWithheld",
            "lib_layout",
            "custom_mm",
            "pcr_0_1",
            "neg_cont_0_1",
            "pos_cont_0_1",
        )
    }
    assert non_review_fields == set()


def test_all_v3_extraction_hints_have_mapping_rules():
    rule_names = {rule.source_fact_type for rule in RULES}
    missing = set(native_name_to_faire_hint()) - rule_names
    assert not missing


def test_custom_master_mix_cross_fills_commercial_master_mix(db_session):
    study = _study(db_session, title="Custom mix only")
    _fact(
        db_session,
        study,
        field="custom_mm",
        value="0.02 U/ul polymerase, 1X buffer, and 200 uM dNTPs",
        entity_level="study",
    )
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    values = {
        sv.target_field: sv
        for sv in db_session.query(StandardizedValue).filter_by(study_id=study.study_id, entity_id=None)
    }
    assert values["custom_mm"].standardized_value == "0.02 U/ul polymerase, 1X buffer, and 200 uM dNTPs"
    assert values["commercial_mm"].standardized_value == "N/A see custom_mm"
    assert values["commercial_mm"].review_required is False


def test_otu_db_pipe_joins_public_and_custom_database_values(db_session):
    study = _study(db_session, title="Multiple taxonomy databases")
    _fact(db_session, study, field="otu_db", value="SILVA_132", entity_level="study")
    _fact(db_session, study, field="otu_db", value="FreshTrain", entity_level="study")
    _fact(db_session, study, field="otu_db", value="custom database curated for lake taxa", entity_level="study")
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    values = {
        sv.target_field: sv
        for sv in db_session.query(StandardizedValue).filter_by(study_id=study.study_id, entity_id=None)
    }
    assert values["otu_db"].standardized_value == "SILVA_132 | FreshTrain | custom database curated for lake taxa"
    assert "otu_db_custom" not in values


def test_all_v3_extraction_hints_have_a_study_level_rule_specifically():
    """Regression guard for a real bug: a native_name string existing
    *somewhere* in RULES (checked above) is not the same as it being
    reachable at EntityLevel.STUDY, which is the only level LLM-extracted
    v3 facts are ever persisted at. Adding an _EXPLICIT_RULES entry for a
    structured-source fact that happens to share a name with a v3
    native_name used to silently delete the STUDY-level rule for every
    such name, because _generated_v3_llm_rules only checked fact_type
    strings, never entity_level, before excluding one from generation."""
    for native_name in native_name_to_faire_hint():
        assert rules_for(native_name, EntityLevel.STUDY.value), (
            f"{native_name!r} has no rule reachable at EntityLevel.STUDY -- "
            "an LLM-extracted fact with this name would never map"
        )


def test_all_assay_scoped_extraction_hints_have_an_assay_level_rule_specifically():
    """Mirrors test_all_v3_extraction_hints_have_a_study_level_rule_specifically
    for the parallel EntityLevel.ASSAY rule: a paper describing more than
    one assay tags each assay's facts with entity_level=ASSAY
    (extraction/text.py's assay_tag), and without a matching ASSAY-level
    rule those facts would have nowhere to map."""
    for native_name in assay_scoped_field_names():
        assert rules_for(native_name, EntityLevel.ASSAY.value), (
            f"{native_name!r} has no rule reachable at EntityLevel.ASSAY -- "
            "an assay-tagged LLM-extracted fact with this name would never map"
        )


def test_v3_native_atomic_facts_map_through_faire_hints_with_review(db_session):
    study = _study(db_session, title="v3 atomic facts")
    for field, value in {
        "dna_extraction_kit": "DNeasy PowerWater Kit",
        "assay_name": "16S-V4",
        "target_gene": "16S rRNA",
        "forward_primer_name": "515F",
        "reverse_primer_name": "806R",
        "forward_primer_sequence": "GTGYCAGCMGCCGCGGTAA",
        "reverse_primer_sequence": "GGACTACNVGGGTWTCTAAT",
        "assay_target_taxa": "Chordata | Crustose coralline algae",
        "reference_database": "SILVA 138",
        "standard_curve_r_squared": "0.997",
        "scientific_name": "Acropora cervicornis",
    }.items():
        _fact(db_session, study, field=field, value=value, entity_level="study", support=SupportType.EXPLICIT)
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    values = {
        sv.target_field: sv
        for sv in db_session.query(StandardizedValue).filter_by(study_id=study.study_id)
    }
    assert values["nucl_acid_ext_kit"].standardized_value == "DNeasy PowerWater Kit"
    assert values["assay_name"].standardized_value == "16S-V4"
    assert values["target_gene"].standardized_value == "16S rRNA"
    assert values["pcr_primer_name_forward"].standardized_value == "515F"
    assert values["pcr_primer_name_reverse"].standardized_value == "806R"
    assert values["pcr_primer_forward"].standardized_value == "GTGYCAGCMGCCGCGGTAA"
    assert values["pcr_primer_reverse"].standardized_value == "GGACTACNVGGGTWTCTAAT"
    assert values["targetTaxonomicAssay"].standardized_value == "Chordata | Crustose coralline algae"
    assert values["otu_db"].standardized_value == "SILVA 138"
    assert values["r2"].standardized_value == "0.997"
    assert values["scientificName"].standardized_value == "Acropora cervicornis"
    assert "otu_db_custom" not in values
    assert all(
        row.review_required is True
        for field, row in values.items()
        if field not in ("checkls_ver", "informationWithheld", "lib_layout")
    )


def test_map_study_to_faire_is_idempotent(db_session):
    study = _study(db_session, title="Idempotency check")
    sample = Entity(study_id=study.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMN1")
    db_session.add(sample)
    db_session.flush()
    _fact(db_session, study, entity=sample, field="geo_loc_name", value="USA: California", entity_level="sample")
    db_session.commit()

    first = map_study_to_faire(db_session, study.study_id)
    db_session.commit()
    second = map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    assert first == second
    assert db_session.query(StandardizedValue).filter_by(study_id=study.study_id).count() == first


def test_resolve_project_id_prefers_bioproject_over_doi(db_session):
    study = _study(db_session, title="Project id preference")
    db_session.add(ExternalIdentifier(study_id=study.study_id, identifier_type=IdentifierType.DOI.value, identifier_value="10.1/abc"))
    db_session.add(ExternalIdentifier(study_id=study.study_id, identifier_type=IdentifierType.BIOPROJECT_ACCESSION.value, identifier_value="PRJNA1"))
    db_session.commit()

    assert resolve_project_id(db_session, study.study_id) == "PRJNA1"


def test_resolve_project_id_falls_back_to_doi_when_no_repository_accession(db_session):
    study = _study(db_session, title="DOI-only study")
    db_session.add(ExternalIdentifier(study_id=study.study_id, identifier_type=IdentifierType.DOI.value, identifier_value="10.1/xyz"))
    db_session.commit()

    assert resolve_project_id(db_session, study.study_id) == "10.1/xyz"


def test_resolve_project_id_returns_none_when_no_identifiers(db_session):
    study = _study(db_session, title="No identifiers")
    db_session.commit()

    assert resolve_project_id(db_session, study.study_id) is None


def test_target_taxonomic_assay_prioritizes_api_over_llm_without_dropping_llm_values(db_session):
    """Per an explicit user spec: targetTaxonomicAssay's structured
    ("API": extraction/taxonomic_assay.py's BioSample-organism
    aggregation, support_type=STRUCTURED_SOURCE) signal is listed first
    on conflict, but the LLM's own value (support_type=EXPLICIT, from
    search_flags.py's targeted quote-judged search) is never discarded --
    a paper's stated assay target can be more specific than any one
    BioSample's own organism field."""
    study = _study(db_session, title="API+LLM taxon union")
    _fact(db_session, study, field="assay_target_taxa", value="Fish", entity_level="study", support=SupportType.STRUCTURED_SOURCE)
    _fact(
        db_session, study, field="assay_target_taxa", value="Atlantic salmon (Salmo salar)",
        entity_level="study", support=SupportType.EXPLICIT,
    )
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    value = db_session.query(StandardizedValue).filter_by(study_id=study.study_id, target_field="targetTaxonomicAssay").one()
    assert value.standardized_value == "Fish | Atlantic salmon (Salmo salar)"


def test_target_taxonomic_assay_prioritizes_api_regardless_of_fact_arrival_order(db_session):
    """The API-first ordering must not depend on which fact happened to be
    created first -- created_at ordering alone would make this fragile
    across separate discovery/extraction pipeline runs."""
    study = _study(db_session, title="API+LLM taxon union, LLM created first")
    _fact(
        db_session, study, field="assay_target_taxa", value="Atlantic salmon (Salmo salar)",
        entity_level="study", support=SupportType.EXPLICIT,
    )
    _fact(db_session, study, field="assay_target_taxa", value="Fish", entity_level="study", support=SupportType.STRUCTURED_SOURCE)
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    value = db_session.query(StandardizedValue).filter_by(study_id=study.study_id, target_field="targetTaxonomicAssay").one()
    assert value.standardized_value == "Fish | Atlantic salmon (Salmo salar)"


def test_target_taxonomic_scope_unions_multiple_distinct_llm_values(db_session):
    """targetTaxonomicScope has no API source ('just LLM' per spec) --
    multiple distinct LLM-derived values (e.g. from separate paper and
    supplement extraction passes) must union/pipe-join, never
    'first wins, flag review, drop the rest'."""
    study = _study(db_session, title="Multiple taxonomic scope values")
    _fact(db_session, study, field="study_target_taxonomic_scope", value="teleost fishes", entity_level="study", support=SupportType.EXPLICIT)
    _fact(db_session, study, field="study_target_taxonomic_scope", value="sharks and rays", entity_level="study", support=SupportType.EXPLICIT)
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    value = db_session.query(StandardizedValue).filter_by(study_id=study.study_id, target_field="targetTaxonomicScope").one()
    assert value.standardized_value == "teleost fishes | sharks and rays"


def test_target_taxonomic_not_found_placeholder_dropped_once_a_real_value_arrives(db_session):
    study = _study(db_session, title="not found then real value")
    _fact(
        db_session, study, field="study_target_taxonomic_scope", value="not found",
        entity_level="study", support=SupportType.DETERMINISTICALLY_DERIVED,
    )
    _fact(db_session, study, field="study_target_taxonomic_scope", value="marine vertebrates", entity_level="study", support=SupportType.EXPLICIT)
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    value = db_session.query(StandardizedValue).filter_by(study_id=study.study_id, target_field="targetTaxonomicScope").one()
    assert value.standardized_value == "marine vertebrates"


def test_target_taxonomic_not_found_alone_survives(db_session):
    study = _study(db_session, title="not found, nothing else")
    _fact(
        db_session, study, field="study_target_taxonomic_scope", value="not found",
        entity_level="study", support=SupportType.DETERMINISTICALLY_DERIVED,
    )
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    value = db_session.query(StandardizedValue).filter_by(study_id=study.study_id, target_field="targetTaxonomicScope").one()
    assert value.standardized_value == "not found"


def test_information_withheld_defaults_when_never_resolved(db_session):
    study = _study(db_session, title="No withheld-information statement")
    _fact(db_session, study, field="geo_loc_name", value="USA: California", entity_level="sample")
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    value = db_session.query(StandardizedValue).filter_by(
        study_id=study.study_id, target_field="informationWithheld"
    ).one()
    assert value.standardized_value == "Nothing indicated as withheld"
    assert value.review_required is False


def test_information_withheld_real_value_wins_over_default(db_session):
    """A real, quote-anchored value must never be overwritten by the
    "Nothing indicated as withheld" default, regardless of which one the
    mapping loop encounters first (RawFact.created_at order)."""
    study = _study(db_session, title="Real withheld-information statement")
    _fact(
        db_session,
        study,
        field="informationWithheld",
        value="All other data are available from the corresponding authors on reasonable request.",
        entity_level="study",
        support=SupportType.EXPLICIT,
    )
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    value = db_session.query(StandardizedValue).filter_by(
        study_id=study.study_id, target_field="informationWithheld"
    ).one()
    assert value.standardized_value == (
        "All other data are available from the corresponding authors on reasonable request."
    )


def test_information_withheld_default_is_idempotent_on_remap(db_session):
    study = _study(db_session, title="Remapped with no withheld statement")
    _fact(db_session, study, field="geo_loc_name", value="USA: California", entity_level="sample")
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()
    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    values = db_session.query(StandardizedValue).filter_by(
        study_id=study.study_id, target_field="informationWithheld"
    ).all()
    assert len(values) == 1
    assert values[0].standardized_value == "Nothing indicated as withheld"


def test_sample_type_routed_field_sends_each_value_to_the_matching_sample(db_session):
    """Regression guard for a real live-paper finding (10.3389/fmicb.2024.1295149):
    a study-level extracted fact ("sample_volume_for_extraction") had two
    genuinely different real values -- "10 L" for water samples, "500 mg"
    for the one real sediment sample -- and the generic "oldest fact wins"
    broadcast rule silently wrote "10 L" onto the sediment sample too.
    Each real SAMPLE entity's own isolation_source now routes the correct
    value to the correct sample instead."""
    study = _study(db_session, title="Mixed sample types")
    water = Entity(study_id=study.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMN_WATER")
    sediment = Entity(
        study_id=study.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMN_SEDIMENT"
    )
    db_session.add_all([water, sediment])
    db_session.flush()
    db_session.add_all([_home_entity_study(water), _home_entity_study(sediment)])
    _fact(db_session, study, entity=water, field="isolation_source", value="water", entity_level="sample")
    _fact(db_session, study, entity=sediment, field="isolation_source", value="sediment", entity_level="sample")

    water_fact = _fact(
        db_session, study, field="sample_volume_for_extraction", value="10 L",
        entity_level="study", support=SupportType.EXPLICIT,
    )
    water_fact.evidence_quote = (
        "Water samples were collected with Niskin bottles attached to the CTD rosette, each with 10 L capacity."
    )
    sediment_fact = _fact(
        db_session, study, field="sample_volume_for_extraction", value="500 mg",
        entity_level="study", support=SupportType.EXPLICIT,
    )
    sediment_fact.evidence_quote = (
        "DNA extraction was performed using the Soil DNA kit. "
        "For sediment samples, 500 mg of dried sediment samples were used for DNA extraction."
    )
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    values = {
        sv.entity_id: sv
        for sv in db_session.query(StandardizedValue).filter_by(
            study_id=study.study_id, target_field="samp_vol_we_dna_ext"
        )
    }
    assert values[water.entity_id].standardized_value == "10 L"
    assert values[sediment.entity_id].standardized_value == "500 mg"
    assert values[water.entity_id].review_required is True
    # No study-wide broadcast row (entity_id=None) -- both real samples got
    # their own routed value instead.
    assert None not in values


def test_sample_type_routed_field_falls_through_when_only_one_type_detected(db_session):
    """No real per-sample-type conflict -- both candidate facts describe
    the same sample type -- so the field falls back to the normal
    broadcast/oldest-wins path unchanged, same as any other field."""
    study = _study(db_session, title="Single sample type, two mentions")
    water = Entity(study_id=study.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMN_WATER")
    db_session.add(water)
    db_session.flush()
    _fact(db_session, study, entity=water, field="isolation_source", value="water", entity_level="sample")

    first = _fact(
        db_session, study, field="sample_volume_for_extraction", value="10 L",
        entity_level="study", support=SupportType.EXPLICIT,
    )
    first.evidence_quote = "Water samples of 10 L were collected using Niskin bottles."
    second = _fact(
        db_session, study, field="sample_volume_for_extraction", value="5 L",
        entity_level="study", support=SupportType.EXPLICIT,
    )
    second.evidence_quote = "A subset of water samples of 5 L were filtered onboard."
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    values = list(
        db_session.query(StandardizedValue).filter_by(study_id=study.study_id, target_field="samp_vol_we_dna_ext")
    )
    assert len(values) == 1
    assert values[0].entity_id is None
    assert values[0].standardized_value == "10 L"


def test_sample_type_routed_field_skips_sample_with_unknown_type(db_session):
    """A real sample whose own isolation_source doesn't clearly say water
    or sediment never gets a guessed value -- consistent with this
    pipeline's standing "never guess absent data" discipline."""
    study = _study(db_session, title="Ambiguous sample type")
    water = Entity(study_id=study.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMN_WATER")
    unknown = Entity(study_id=study.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMN_UNKNOWN")
    db_session.add_all([water, unknown])
    db_session.flush()
    db_session.add_all([_home_entity_study(water), _home_entity_study(unknown)])
    _fact(db_session, study, entity=water, field="isolation_source", value="water", entity_level="sample")
    _fact(db_session, study, entity=unknown, field="isolation_source", value="marine metagenome", entity_level="sample")

    water_fact = _fact(
        db_session, study, field="sample_volume_for_extraction", value="10 L",
        entity_level="study", support=SupportType.EXPLICIT,
    )
    water_fact.evidence_quote = "Water samples of 10 L were collected using Niskin bottles."
    sediment_fact = _fact(
        db_session, study, field="sample_volume_for_extraction", value="500 mg",
        entity_level="study", support=SupportType.EXPLICIT,
    )
    sediment_fact.evidence_quote = "For sediment samples, 500 mg of dried sediment was used for DNA extraction."
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    values = {
        sv.entity_id: sv
        for sv in db_session.query(StandardizedValue).filter_by(
            study_id=study.study_id, target_field="samp_vol_we_dna_ext"
        )
    }
    assert values[water.entity_id].standardized_value == "10 L"
    assert unknown.entity_id not in values


def test_sample_type_routed_field_is_idempotent_on_remap(db_session):
    study = _study(db_session, title="Remapped mixed sample types")
    water = Entity(study_id=study.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMN_WATER")
    sediment = Entity(
        study_id=study.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMN_SEDIMENT"
    )
    db_session.add_all([water, sediment])
    db_session.flush()
    db_session.add_all([_home_entity_study(water), _home_entity_study(sediment)])
    _fact(db_session, study, entity=water, field="isolation_source", value="water", entity_level="sample")
    _fact(db_session, study, entity=sediment, field="isolation_source", value="sediment", entity_level="sample")
    water_fact = _fact(
        db_session, study, field="sample_volume_for_extraction", value="10 L",
        entity_level="study", support=SupportType.EXPLICIT,
    )
    water_fact.evidence_quote = "Water samples of 10 L were collected using Niskin bottles."
    sediment_fact = _fact(
        db_session, study, field="sample_volume_for_extraction", value="500 mg",
        entity_level="study", support=SupportType.EXPLICIT,
    )
    sediment_fact.evidence_quote = "For sediment samples, 500 mg of dried sediment was used for DNA extraction."
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()
    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    values = list(
        db_session.query(StandardizedValue).filter_by(study_id=study.study_id, target_field="samp_vol_we_dna_ext")
    )
    assert len(values) == 2


def test_detect_sample_type_from_quote_ignores_ambiguous_sentence():
    """A real sentence can mention both keywords (e.g. "water overlaying
    sediment") -- never guessed, since a naive keyword match can't tell
    which sample type the value actually describes."""
    from fair_ocean_agent.mapping.faire import _detect_sample_type_from_quote

    quote = "Water overlaying sediment from the cores was collected (approx. 10 L each) from all three stations."
    assert _detect_sample_type_from_quote("10 L", quote) is None


def test_detect_sample_type_from_quote_only_checks_the_sentence_with_the_value():
    """A multi-sentence quote can discuss one sample type in an earlier
    sentence and the actual extracted value in a later, unrelated one --
    only the sentence containing the raw_value itself is checked."""
    from fair_ocean_agent.mapping.faire import _detect_sample_type_from_quote

    quote = (
        "Water samples were collected using Niskin bottles. "
        "Sediment cores were sectioned into 2 cm intervals for downstream analysis."
    )
    assert _detect_sample_type_from_quote("2 cm", quote) == "sediment"
    assert _detect_sample_type_from_quote("Niskin", quote) == "water"


def test_sample_type_routed_field_never_broadcasts_onto_a_sample_from_an_older_linked_study(db_session):
    """Regression guard for a real live-paper concern (10.1038/s42003-024-
    06136-2 shares real BioSample accessions with an older, unrelated
    study): a sample entity whose real root is a DIFFERENT study must
    never receive THIS study's own sample-type-routed value, even though
    it's linked here (e.g. it cites/reuses the old accession)."""
    old_study = _study(db_session, title="Older, unrelated study")
    new_study = _study(db_session, title="This paper")
    old_sample = Entity(
        study_id=old_study.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMN_OLD_SEDIMENT"
    )
    new_water = Entity(
        study_id=new_study.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMN_NEW_WATER"
    )
    db_session.add_all([old_sample, new_water])
    db_session.flush()
    # old_sample is rooted at old_study but ALSO linked to new_study (e.g.
    # new_study's paper cites/reuses this accession without it being a
    # sample this paper itself collected).
    old_sample.root_status = EntityRootStatus.DETERMINED.value
    old_sample.root_study_id = old_study.study_id
    db_session.add_all(
        [
            EntityStudy(
                entity_id=old_sample.entity_id, study_id=old_study.study_id,
                relationship_type=RelationshipType.IS_HOME_OF.value, confidence=SupportType.STRUCTURED_SOURCE.value,
            ),
            EntityStudy(
                entity_id=old_sample.entity_id, study_id=new_study.study_id,
                relationship_type=RelationshipType.SHARES_ACCESSION_WITH.value,
                confidence=SupportType.STRUCTURED_SOURCE.value,
            ),
            _home_entity_study(new_water),
        ]
    )
    _fact(db_session, old_study, entity=old_sample, field="isolation_source", value="sediment", entity_level="sample")
    _fact(db_session, new_study, entity=new_water, field="isolation_source", value="water", entity_level="sample")

    water_fact = _fact(
        db_session, new_study, field="sample_volume_for_extraction", value="10 L",
        entity_level="study", support=SupportType.EXPLICIT,
    )
    water_fact.evidence_quote = "Water samples of 10 L were collected using Niskin bottles."
    sediment_fact = _fact(
        db_session, new_study, field="sample_volume_for_extraction", value="500 mg",
        entity_level="study", support=SupportType.EXPLICIT,
    )
    sediment_fact.evidence_quote = "For sediment samples, 500 mg of dried sediment was used for DNA extraction."
    db_session.commit()

    map_study_to_faire(db_session, new_study.study_id)
    db_session.commit()

    values = {
        sv.entity_id: sv
        for sv in db_session.query(StandardizedValue).filter_by(
            study_id=new_study.study_id, target_field="samp_vol_we_dna_ext"
        )
    }
    assert values[new_water.entity_id].standardized_value == "10 L"
    # The old, merely-linked sediment sample never got new_study's "500 mg"
    # -- it isn't rooted here, so it's outside this study's authoritative
    # broadcast surface entirely.
    assert old_sample.entity_id not in values


def test_sample_type_routed_field_case3_corroborated_flags_for_review(db_session):
    """Case 3, per an explicit user specification: more than one sample
    type is genuinely described in the text, but nothing in the real
    BioSample data distinguishes which sample is which. A "double pass"
    (corroboration from a second, independently-conflicting sample_prep
    field) confirms this is a real multi-sample-type paper rather than one
    miscategorized quote, so both fields get a pipe-joined, flagged-for-
    review study-wide value instead of silently guessing or dropping data."""
    study = _study(db_session, title="Mixed types, no BioSample signal")
    sample = Entity(study_id=study.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMN1")
    db_session.add(sample)
    db_session.flush()
    db_session.add(_home_entity_study(sample))
    # No isolation_source/env_medium/samp_mat_process fact at all for this
    # sample -- nothing in the BioSample API distinguishes its type.

    vol_water = _fact(
        db_session, study, field="sample_volume_for_extraction", value="10 L",
        entity_level="study", support=SupportType.EXPLICIT,
    )
    vol_water.evidence_quote = "Water samples of 10 L were collected using Niskin bottles."
    vol_sediment = _fact(
        db_session, study, field="sample_volume_for_extraction", value="500 mg",
        entity_level="study", support=SupportType.EXPLICIT,
    )
    vol_sediment.evidence_quote = "For sediment samples, 500 mg of dried sediment was used for DNA extraction."

    kit_water = _fact(
        db_session, study, field="nucl_acid_ext_kit", value="DNeasy PowerWater Kit",
        entity_level="study", support=SupportType.EXPLICIT,
    )
    kit_water.evidence_quote = "Water samples were extracted using the DNeasy PowerWater Kit."
    kit_sediment = _fact(
        db_session, study, field="nucl_acid_ext_kit", value="DNeasy PowerSoil Kit",
        entity_level="study", support=SupportType.EXPLICIT,
    )
    kit_sediment.evidence_quote = "Sediment samples were extracted using the DNeasy PowerSoil Kit."
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    vol_value = db_session.query(StandardizedValue).filter_by(
        study_id=study.study_id, target_field="samp_vol_we_dna_ext"
    ).one()
    kit_value = db_session.query(StandardizedValue).filter_by(
        study_id=study.study_id, target_field="nucl_acid_ext_kit"
    ).one()
    assert vol_value.entity_id is None
    assert vol_value.review_required is True
    assert set(vol_value.standardized_value.split(" | ")) == {"10 L", "500 mg"}
    assert kit_value.review_required is True
    assert set(kit_value.standardized_value.split(" | ")) == {"DNeasy PowerWater Kit", "DNeasy PowerSoil Kit"}


def test_sample_type_routed_field_case3_uncorroborated_falls_back_to_oldest_wins(db_session):
    """The mirror image of the corroborated case: only ONE sample_prep
    field shows a text-level conflict, nothing else corroborates it, and
    no real sample has a determinable type -- insufficient evidence to
    confidently flag a multi-sample-type conflict, so this field quietly
    falls back to the same "oldest fact wins" behavior it would have
    gotten if this routing mechanism didn't exist at all."""
    study = _study(db_session, title="Single ambiguous field, no corroboration")
    sample = Entity(study_id=study.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMN1")
    db_session.add(sample)
    db_session.flush()
    db_session.add(_home_entity_study(sample))

    vol_water = _fact(
        db_session, study, field="sample_volume_for_extraction", value="10 L",
        entity_level="study", support=SupportType.EXPLICIT,
    )
    vol_water.evidence_quote = "Water samples of 10 L were collected using Niskin bottles."
    vol_sediment = _fact(
        db_session, study, field="sample_volume_for_extraction", value="500 mg",
        entity_level="study", support=SupportType.EXPLICIT,
    )
    vol_sediment.evidence_quote = "For sediment samples, 500 mg of dried sediment was used for DNA extraction."
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    value = db_session.query(StandardizedValue).filter_by(
        study_id=study.study_id, target_field="samp_vol_we_dna_ext"
    ).one()
    assert value.entity_id is None
    assert value.standardized_value == "10 L"


def test_sample_type_for_entity_falls_back_to_env_medium_then_samp_mat_process(db_session):
    from fair_ocean_agent.mapping.faire import _sample_type_for_entity

    study = _study(db_session, title="Fallback chain check")
    sample = Entity(study_id=study.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMN1")
    db_session.add(sample)
    db_session.flush()
    _fact(db_session, study, entity=sample, field="samp_mat_process", value="DNA extraction from sediment samples", entity_level="sample")
    db_session.commit()

    assert _sample_type_for_entity(db_session, sample.entity_id) == "sediment"

    _fact(db_session, study, entity=sample, field="env_medium", value="marine water", entity_level="sample")
    db_session.commit()
    # env_medium (checked before samp_mat_process) now wins.
    assert _sample_type_for_entity(db_session, sample.entity_id) == "water"

    _fact(db_session, study, entity=sample, field="isolation_source", value="sediment", entity_level="sample")
    db_session.commit()
    # isolation_source (checked first) now wins over both fallbacks.
    assert _sample_type_for_entity(db_session, sample.entity_id) == "sediment"


def test_sample_collection_terms_have_study_level_rules_except_samp_category():
    """samp_collect_device/samp_collect_method/samp_size/samp_size_unit/
    sample_composed_of/sample_derived_from/internal_expedition_id are
    broadcast-safe (the same physical collection process applies uniformly
    to a study's samples, matching every other sample_prep field), so each needs a reachable
    STUDY-level rule. samp_category deliberately has none -- see its own
    CategoryTerm comment in extraction/section_categories.py: broadcasting
    one extracted "negative control" quote onto every sample would
    mislabel the real environmental samples too."""
    for native_name in (
        "samp_collect_device",
        "samp_collect_method",
        "samp_size",
        "samp_size_unit",
        "sample_composed_of",
        "sample_derived_from",
        "internal_expedition_id",
    ):
        assert rules_for(native_name, EntityLevel.STUDY.value), (
            f"{native_name!r} has no STUDY-level rule -- its sample_prep CategoryTerm fact would never map"
        )
    assert rules_for("samp_category", EntityLevel.STUDY.value) == []


def test_samp_category_excluded_from_sample_type_routing():
    """samp_category lives in the sample_prep category (per an explicit
    user instruction not to give it its own classifier) but its own "type"
    axis is real-sample-vs-control, not water-vs-sediment -- the wrong fit
    for this router's water/sediment detection."""
    from fair_ocean_agent.mapping.faire import _SAMPLE_TYPE_ROUTED_NATIVE_NAMES

    assert "samp_category" not in _SAMPLE_TYPE_ROUTED_NATIVE_NAMES
    for native_name in (
        "samp_collect_device",
        "samp_collect_method",
        "samp_size",
        "samp_size_unit",
        "sample_composed_of",
        "sample_derived_from",
    ):
        assert native_name in _SAMPLE_TYPE_ROUTED_NATIVE_NAMES
