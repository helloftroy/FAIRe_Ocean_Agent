"""Tests for mapping/faire.py, shaped after this pipeline's real raw_facts
data (see mapping/rules.py's docstring for the query that grounded these
fields): NCBI BioSample sample-level facts, ENA sequencing_run-level
facts repeated identically across many runs, and LLM-extracted study-level
free text.
"""
from fair_ocean_agent.database.enums import EntityLevel, IdentifierType, SupportType
from fair_ocean_agent.database.models import Entity, ExternalIdentifier, RawFact, StandardizedValue, Study
from fair_ocean_agent.extraction.faire_fields import assay_scoped_field_names, native_name_to_faire_hint
from fair_ocean_agent.mapping.faire import map_study_to_faire, resolve_project_id
from fair_ocean_agent.mapping.rules import _ADDITIONAL_ENVIRONMENTAL_SAMPLE_ATTRIBUTES, RULES, rules_for


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
    assert values["geo_loc_name"].missingness_status == "present"
    assert values["decimalLatitude"].standardized_value == "38.030000"
    assert values["decimalLongitude"].standardized_value == "-122.151667"
    assert values["samp_name"].standardized_value == "SAMN1"
    assert values["samp_name"].mapping_method == "exact_identifier"
    assert values["samp_name"].missingness_status == "present"
    assert created == len(values)


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
    """elev/samp_collect_device/samp_size/samp_size_unit/temp/salinity/ph/
    diss_oxygen all arrive through the exact same NCBI BioSample
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
    _fact(db_session, study, entity=sample, field="diss_oxygen", value="6.2", entity_level="sample")
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
    assert values["diss_oxygen"] == "6.2"


def test_every_faire_environment_field_has_a_sample_level_rule():
    """Systematic guard: every FAIRe field tagged in_subset: Environment in
    the vendored schema (70 total) must be reachable *as a target_field* by
    some SAMPLE-level rule -- either an explicit rule above
    (minimumDepthInMeters/maximumDepthInMeters via "depth", elev/temp/
    salinity/ph/diss_oxygen via their own name) or the generated
    _ADDITIONAL_ENVIRONMENTAL_SAMPLE_ATTRIBUTES batch. Catches a future
    schema update silently adding a new environmental field this table
    never learns about."""
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
    missing = env_fields - sample_level_target_fields
    assert not missing, f"FAIRe Environment fields with no SAMPLE-level rule: {sorted(missing)}"


def test_maps_a_sample_of_the_generated_environmental_attributes(db_session):
    study = _study(db_session, title="Additional environmental attributes")
    sample = Entity(study_id=study.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMN1")
    db_session.add(sample)
    db_session.flush()
    _fact(db_session, study, entity=sample, field="turbidity", value="4.2 NTU", entity_level="sample")
    _fact(db_session, study, entity=sample, field="chlorophyll", value="0.8 mg/m3", entity_level="sample")
    _fact(db_session, study, entity=sample, field="host_species", value="Thunnus albacares", entity_level="sample")
    _fact(db_session, study, entity=sample, field="tot_nitro_unit", value="mg/L", entity_level="sample")
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    values = {
        sv.target_field: sv.standardized_value
        for sv in db_session.query(StandardizedValue).filter_by(study_id=study.study_id, entity_id=sample.entity_id)
    }
    assert values["turbidity"] == "4.2 NTU"
    assert values["chlorophyll"] == "0.8 mg/m3"
    assert values["host_species"] == "Thunnus albacares"
    assert values["tot_nitro_unit"] == "mg/L"


def test_generated_environmental_attribute_names_have_no_duplicates():
    names = [field for field, _ in _ADDITIONAL_ENVIRONMENTAL_SAMPLE_ATTRIBUTES]
    assert len(names) == len(set(names))


def test_maps_run_level_fastq_and_checksum_and_lib_layout(db_session):
    """ENA's fastq_ftp/fastq_md5 (';'-joined forward;reverse pairs) split
    into FAIRe's filename/filename2/checksum_filename/checksum_filename2;
    checksum_method is inferred as a constant ("MD5" -- ENA never reports
    another algorithm); library_layout normalizes ENA's PAIRED/SINGLE into
    FAIRe's lib_layout_enum spelling."""
    study = _study(db_session, title="Run-level file facts")
    run = Entity(study_id=study.study_id, entity_level=EntityLevel.SEQUENCING_RUN.value, external_identifier="SRR1")
    db_session.add(run)
    db_session.flush()
    _fact(
        db_session, study, entity=run, entity_level="sequencing_run", field="fastq_ftp",
        value="ftp.sra.ebi.ac.uk/vol1/fastq/SRR001/SRR1_1.fastq.gz;ftp.sra.ebi.ac.uk/vol1/fastq/SRR001/SRR1_2.fastq.gz",
    )
    _fact(
        db_session, study, entity=run, entity_level="sequencing_run", field="fastq_md5",
        value="aaa111;bbb222",
    )
    _fact(db_session, study, entity=run, entity_level="sequencing_run", field="read_count", value="1000000")
    _fact(db_session, study, entity=run, entity_level="sequencing_run", field="library_layout", value="PAIRED")
    _fact(db_session, study, entity=run, entity_level="sequencing_run", field="run_accession", value="SRR1")
    _fact(db_session, study, entity=run, entity_level="sequencing_run", field="sample_accession", value="SAMN1")
    _fact(db_session, study, entity=run, entity_level="sequencing_run", field="instrument_platform", value="ILLUMINA")
    _fact(db_session, study, entity=run, entity_level="sequencing_run", field="assay_name", value="16S metabarcoding")
    _fact(db_session, study, entity=run, entity_level="sequencing_run", field="pcr_plate_id", value="plate-1")
    _fact(db_session, study, entity=run, entity_level="sequencing_run", field="lib_id", value="lib-1")
    _fact(db_session, study, entity=run, entity_level="sequencing_run", field="lib_conc", value="4.2")
    _fact(db_session, study, entity=run, entity_level="sequencing_run", field="lib_conc_unit", value="ng/μL")
    _fact(db_session, study, entity=run, entity_level="sequencing_run", field="lib_conc_meth", value="Qubit")
    _fact(db_session, study, entity=run, entity_level="sequencing_run", field="phix_perc", value="15")
    _fact(db_session, study, entity=run, entity_level="sequencing_run", field="mid_forward", value="ACGT")
    _fact(db_session, study, entity=run, entity_level="sequencing_run", field="mid_reverse", value="TGCA")
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    values = {sv.target_field: sv for sv in db_session.query(StandardizedValue).filter_by(study_id=study.study_id)}
    assert values["filename"].standardized_value == "SRR1_1.fastq.gz"
    assert values["filename2"].standardized_value == "SRR1_2.fastq.gz"
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
    assert values["lib_conc"].standardized_value == "4.2"
    assert values["lib_conc_unit"].standardized_value == "ng/μL"
    assert values["lib_conc_meth"].standardized_value == "Qubit"
    assert values["phix_perc"].standardized_value == "15"
    assert values["mid_forward"].standardized_value == "ACGT"
    assert values["mid_reverse"].standardized_value == "TGCA"


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
    _fact(db_session, study, entity=run, entity_level="sequencing_run", field="library_layout", value="SINGLE")
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


def test_run_level_checksum_method_and_lib_layout_still_collapse_project_wide(db_session):
    """Unlike filename/checksum_filename/input_read_count above,
    checksum_method and lib_layout map onto projectMetadata (not
    experimentRunMetadata) -- these are expected to agree across a study's
    runs (ENA always reports MD5; a study's runs are consistently
    paired-end or single-end), so they should still collapse to one
    project-wide row, same as instrument_platform always has."""
    study = _study(db_session, title="Two runs, same layout")
    run_a = Entity(study_id=study.study_id, entity_level=EntityLevel.SEQUENCING_RUN.value, external_identifier="SRR_A")
    run_b = Entity(study_id=study.study_id, entity_level=EntityLevel.SEQUENCING_RUN.value, external_identifier="SRR_B")
    db_session.add_all([run_a, run_b])
    db_session.flush()
    _fact(db_session, study, entity=run_a, entity_level="sequencing_run", field="fastq_md5", value="aaa")
    _fact(db_session, study, entity=run_a, entity_level="sequencing_run", field="library_layout", value="PAIRED")
    _fact(db_session, study, entity=run_b, entity_level="sequencing_run", field="fastq_md5", value="bbb")
    _fact(db_session, study, entity=run_b, entity_level="sequencing_run", field="library_layout", value="PAIRED")
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    checksum_method_rows = db_session.query(StandardizedValue).filter_by(study_id=study.study_id, target_field="checksum_method").all()
    lib_layout_rows = db_session.query(StandardizedValue).filter_by(study_id=study.study_id, target_field="lib_layout").all()
    assert len(checksum_method_rows) == 1
    assert checksum_method_rows[0].entity_id is None
    assert len(lib_layout_rows) == 1
    assert lib_layout_rows[0].entity_id is None


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


def test_maps_deterministic_publication_metadata_facts_to_faire(db_session):
    """extraction/publication_metadata.py's deterministic (no-LLM) facts --
    literal FAIRe field names at EntityLevel.STUDY -- all map through."""
    study = _study(db_session, title="Deterministic publication metadata")
    values = {
        "license": "http://creativecommons.org/licenses/by/3.0/",
        "rightsHolder": "Davies et al.",
        "accessRights": "open access",
        "bibliographicCitation": "Davies SW, et al. (2014). A cross-ocean comparison. PeerJ 2:e333.",
        "code_repo": "https://github.com/someorg/somerepo",
        "recordedBy": "Sarah W. Davies | Eli Meyer",
        "recordedByID": "https://orcid.org/0000-0000-0000-0001",
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
    assert len(project_rows) == 1  # first project-wide value wins, no duplicate row
    assert project_rows[0].review_required is True  # but the disagreement isn't silently dropped
    assert len(run_rows) == 0  # platform is project metadata, never a library/run-row field


def test_assay_tagged_facts_get_separate_projectmetadata_rows_not_a_conflict(db_session):
    """Two distinct assays' annealing_temperature facts (extraction/text.py's
    assay_tag) must not collide into one projectMetadata row the way two
    untagged conflicting facts do (see
    test_flags_review_required_on_conflicting_project_wide_facts above) --
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
    # test_flags_review_required_on_conflicting_project_wide_facts above).
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
    assert created == 3


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


def test_controlled_text_search_project_facts_map_to_faire_with_review(db_session):
    study = _study(db_session, title="Controlled project searches")
    _fact(db_session, study, field="seq_kit", value="MiSeq Reagent Kit v3", entity_level="study")
    _fact(
        db_session,
        study,
        field="sequencing_location",
        value="Genome Sequencing and Analysis Facility (GSAF) at the University of Texas at Austin",
        entity_level="study",
    )
    _fact(db_session, study, field="probeReporter", value="FAM", entity_level="study")
    _fact(db_session, study, field="probeQuencher", value="BHQ-1 | quencher", entity_level="study")
    _fact(db_session, study, field="commercial_mm", value="TaqMan", entity_level="study")
    _fact(db_session, study, field="sterilise_method", value="Bottles were rinsed with bleach.", entity_level="study")
    _fact(db_session, study, field="biological_rep", value="3", entity_level="study")
    _fact(db_session, study, field="assay_type", value="targeted | metabarcoding", entity_level="study")
    _fact(db_session, study, field="barcoding_pcr_appr", value="two-step PCR", entity_level="study")
    _fact(db_session, study, field="lib_screen", value="cleaned with AMPure beads", entity_level="study")
    _fact(db_session, study, field="adapter_forward", value="AATGATACGGCGACCACCGAGATCTACACGCT", entity_level="study")
    _fact(db_session, study, field="adapter_reverse", value="CAAGCAGAAGACGGCATACGAGAT", entity_level="study")
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
    assert values["sequencing_location"].standardized_value == (
        "Genome Sequencing and Analysis Facility (GSAF) at the University of Texas at Austin"
    )
    assert values["probeReporter"].standardized_value == "FAM"
    assert values["probeQuencher"].standardized_value == "BHQ-1 | quencher"
    assert values["commercial_mm"].standardized_value == "TaqMan"
    assert values["sterilise_method"].standardized_value == "Bottles were rinsed with bleach."
    assert values["biological_rep"].standardized_value == "3"
    assert values["assay_type"].standardized_value == "targeted | metabarcoding"
    assert values["barcoding_pcr_appr"].standardized_value == "two-step PCR"
    assert values["lib_screen"].standardized_value == "cleaned with AMPure beads"
    assert values["adapter_forward"].standardized_value == "AATGATACGGCGACCACCGAGATCTACACGCT"
    assert values["adapter_reverse"].standardized_value == "CAAGCAGAAGACGGCATACGAGAT"
    assert values["pcr_0_1"].standardized_value == "1"
    assert values["neg_cont_0_1"].standardized_value == "1"
    assert values["pos_cont_0_1"].standardized_value == "0"
    assert "sample_type" not in values
    assert all(row.review_required is True for row in values.values())


def test_all_v3_extraction_hints_have_mapping_rules():
    rule_names = {rule.source_fact_type for rule in RULES}
    missing = set(native_name_to_faire_hint()) - rule_names
    assert not missing


def test_all_v3_extraction_hints_have_a_study_level_rule_specifically():
    """Regression guard for a real bug: a native_name string existing
    *somewhere* in RULES (checked above) is not the same as it being
    reachable at EntityLevel.STUDY, which is the only level LLM-extracted
    v3 facts are ever persisted at. Adding an _EXPLICIT_RULES entry for a
    structured-source fact that happens to share a name with a v3
    native_name (e.g. ENA's "library_layout" at SEQUENCING_RUN vs this
    taxonomy's own "library_layout" native_name) used to silently delete
    the STUDY-level rule for every such name, because
    _generated_v3_llm_rules only checked fact_type strings, never
    entity_level, before excluding one from generation."""
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
        "denoising_tool": "DADA2 v1.16",
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
    assert values["error_rate_tool"].standardized_value == "DADA2 v1.16"
    assert values["otu_db"].standardized_value == "SILVA 138"
    assert values["r2"].standardized_value == "0.997"
    assert values["scientificName"].standardized_value == "Acropora cervicornis"
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
