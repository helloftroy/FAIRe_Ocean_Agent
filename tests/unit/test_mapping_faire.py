"""Tests for mapping/faire.py, shaped after this pipeline's real raw_facts
data (see mapping/rules.py's docstring for the query that grounded these
fields): NCBI BioSample sample-level facts, ENA sequencing_run-level
facts repeated identically across many runs, and LLM-extracted study-level
free text.
"""
from fair_ocean_agent.database.enums import EntityLevel, IdentifierType, SupportType
from fair_ocean_agent.database.models import Entity, ExternalIdentifier, RawFact, StandardizedValue, Study
from fair_ocean_agent.mapping.faire import map_study_to_faire, resolve_project_id


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


def test_maps_sample_level_structured_facts(db_session):
    study = _study(db_session, title="Sample-level mapping")
    sample = Entity(study_id=study.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMN1")
    db_session.add(sample)
    db_session.flush()
    _fact(db_session, study, entity=sample, field="collection_date", value="2023-12-06", entity_level="sample")
    _fact(db_session, study, entity=sample, field="depth", value="5 meters", entity_level="sample")
    _fact(db_session, study, entity=sample, field="env_broad_scale", value="http://purl.obolibrary.org/obo/ENVO_00000447", entity_level="sample")
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
    assert values["verbatimEventDate"].standardized_value == "2023-12-06"
    assert values["minimumDepthInMeters"].standardized_value == "5"
    assert values["maximumDepthInMeters"].standardized_value == "5"
    assert values["env_broad_scale"].standardized_value == "http://purl.obolibrary.org/obo/ENVO_00000447"
    assert not values["env_broad_scale"].review_required
    assert values["geo_loc_name"].standardized_value == "USA: California"
    assert values["decimalLatitude"].standardized_value == "38.030000"
    assert values["decimalLongitude"].standardized_value == "-122.151667"
    assert created == len(values)


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

    rows = db_session.query(StandardizedValue).filter_by(study_id=study.study_id, target_field="platform").all()
    assert len(rows) == 1
    assert rows[0].entity_id is None
    assert rows[0].standardized_value == "ILLUMINA"
    assert not rows[0].review_required
    assert created == 1


def test_flags_review_required_on_conflicting_project_wide_facts(db_session):
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

    rows = db_session.query(StandardizedValue).filter_by(study_id=study.study_id, target_field="platform").all()
    assert len(rows) == 1  # first value wins, no duplicate row
    assert rows[0].review_required is True  # but the disagreement isn't silently dropped


def test_flags_review_required_when_value_fails_closed_vocab_check(db_session):
    study = _study(db_session, title="Bad platform value")
    run = Entity(study_id=study.study_id, entity_level=EntityLevel.SEQUENCING_RUN.value, external_identifier="SRR1")
    db_session.add(run)
    db_session.flush()
    _fact(db_session, study, entity=run, field="instrument_platform", value="SOME_UNKNOWN_PLATFORM", entity_level="sequencing_run")
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    row = db_session.query(StandardizedValue).filter_by(study_id=study.study_id, target_field="platform").one()
    assert row.review_required is True


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


def test_sample_accession_with_no_matching_sample_entity_is_skipped_not_fabricated(db_session):
    study = _study(db_session, title="No matching sample")
    run = Entity(study_id=study.study_id, entity_level=EntityLevel.SEQUENCING_RUN.value, external_identifier="SRR1")
    db_session.add(run)
    db_session.flush()
    _fact(db_session, study, entity=run, field="sample_accession", value="SAMN_NONEXISTENT", entity_level="sequencing_run")
    db_session.commit()

    created = map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    assert created == 0
    assert db_session.query(StandardizedValue).filter_by(study_id=study.study_id).count() == 0


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
    assert all(row.review_required is True for row in values.values())


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
    _fact(
        db_session, study, field="PCR_amplification_conditions_thermocycler",
        value="Veriti Thermal Cycler (Applied Biosystems)",
        entity_level="study", support=SupportType.EXPLICIT,
    )
    _fact(
        db_session, study, field="bioinformatics_workflow", value="DADA2 (v1.16) in R",
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
    assert values["thermocycler"].standardized_value == "Veriti Thermal Cycler (Applied Biosystems)"
    assert values["bioinfo_method_additional"].standardized_value == "DADA2 (v1.16) in R"
    assert all(row.review_required is True for row in values.values())


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
