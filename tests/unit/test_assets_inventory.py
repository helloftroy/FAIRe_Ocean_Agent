from fair_ocean_agent.assets.inventory import inventory_ena_run_assets
from fair_ocean_agent.database.enums import AssetType, EntityLevel, RawOrProcessed, SupportType
from fair_ocean_agent.database.models import DataAsset, Entity, RawFact, Source, Study


def _seeded_study_with_ena_run(session, run_accession="SRR123", fastq_bytes="1000") -> tuple[Study, Source, Entity]:
    study = Study(title="A study")
    session.add(study)
    session.flush()

    source = Source(study_id=study.study_id, source_type="repository_api", source_name="ena")
    session.add(source)
    session.flush()

    entity = Entity(
        study_id=study.study_id, entity_level=EntityLevel.SEQUENCING_RUN.value, external_identifier=run_accession
    )
    session.add(entity)
    session.flush()

    facts = {
        "fastq_ftp": f"ftp.sra.ebi.ac.uk/vol1/fastq/{run_accession}/{run_accession}.fastq.gz",
        "fastq_bytes": fastq_bytes,
        "library_strategy": "AMPLICON",
        "library_source": "METAGENOMIC",
        "instrument_platform": "ILLUMINA",
    }
    for field, value in facts.items():
        session.add(
            RawFact(
                study_id=study.study_id,
                entity_id=entity.entity_id,
                source_id=source.source_id,
                raw_field_name=field,
                raw_value=value,
                fact_type_candidate=field,
                entity_level=EntityLevel.SEQUENCING_RUN.value,
                support_type=SupportType.STRUCTURED_SOURCE.value,
            )
        )
    session.flush()
    return study, source, entity


def test_inventory_creates_raw_sequence_files_asset(db_session):
    study, source, entity = _seeded_study_with_ena_run(db_session)

    created = inventory_ena_run_assets(db_session, study.study_id)
    db_session.commit()

    assert created == 1
    asset = db_session.query(DataAsset).filter_by(study_id=study.study_id).one()
    assert asset.asset_type == AssetType.RAW_SEQUENCE_FILES.value
    assert asset.repository == "ENA"
    assert asset.identifier == "SRR123"
    assert asset.file_name == "SRR123.fastq.gz"
    assert asset.format == "fastq.gz"
    assert asset.size_bytes == 1000
    assert asset.raw_or_processed == RawOrProcessed.RAW.value
    assert "AMPLICON" in asset.description


def test_inventory_is_idempotent_on_retry(db_session):
    study, _, _ = _seeded_study_with_ena_run(db_session)

    inventory_ena_run_assets(db_session, study.study_id)
    db_session.commit()
    created_second_pass = inventory_ena_run_assets(db_session, study.study_id)
    db_session.commit()

    assert created_second_pass == 0
    assert db_session.query(DataAsset).filter_by(study_id=study.study_id).count() == 1


def test_inventory_returns_zero_when_no_ena_source(db_session):
    study = Study(title="No ENA data here")
    db_session.add(study)
    db_session.flush()

    created = inventory_ena_run_assets(db_session, study.study_id)
    assert created == 0


def test_inventory_skips_runs_without_fastq_ftp(db_session):
    study = Study(title="Run metadata but no listed file")
    db_session.add(study)
    db_session.flush()
    source = Source(study_id=study.study_id, source_type="repository_api", source_name="ena")
    db_session.add(source)
    db_session.flush()
    entity = Entity(study_id=study.study_id, entity_level=EntityLevel.SEQUENCING_RUN.value, external_identifier="SRR999")
    db_session.add(entity)
    db_session.flush()
    db_session.add(
        RawFact(
            study_id=study.study_id,
            entity_id=entity.entity_id,
            source_id=source.source_id,
            raw_field_name="library_strategy",
            raw_value="AMPLICON",
            fact_type_candidate="library_strategy",
            entity_level=EntityLevel.SEQUENCING_RUN.value,
            support_type=SupportType.STRUCTURED_SOURCE.value,
        )
    )
    db_session.flush()

    created = inventory_ena_run_assets(db_session, study.study_id)
    assert created == 0
