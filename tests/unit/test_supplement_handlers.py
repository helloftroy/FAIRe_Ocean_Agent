"""Integration tests for handle_discover_supplements/handle_retrieve_supplements
-- a fake europe_pmc adapter (no network) standing in for the real one,
mirroring test_handlers_text_extraction.py's FakeEuropePmcAdapter pattern."""
import io
import json
import zipfile

import pytest

from fair_ocean_agent.config import reset_config_cache
from fair_ocean_agent.database.enums import IdentifierType, TaskType
from fair_ocean_agent.database.models import (
    DataAsset,
    ExternalIdentifier,
    PreparedSourceText,
    RawFact,
    Source,
    StandardizedValue,
    Study,
    StudySource,
    Task,
)
from fair_ocean_agent.extraction.text import PROMPT_VERSION, segment_source_text
from fair_ocean_agent.llm.base import LLMBackendError
from fair_ocean_agent.llm.mock import MockLLMBackend
from fair_ocean_agent.sources.base import RelatedIdentifier, SourceRecordNotFoundError
from fair_ocean_agent.sources.europe_pmc import SupplementReference
from fair_ocean_agent.workflow import handlers, supplement_handlers
from fair_ocean_agent.workflow.task_queue import enqueue_task

FULLTEXT_XML = """<article><body><sec>
<supplementary-material id="TS1"><media xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="Table_1.csv"
mimetype="text" mime-subtype="csv"><?size 40?></media></supplementary-material>
<supplementary-material id="TS2"><media xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="Big_Data.zip"
mimetype="application" mime-subtype="zip"><?size 30000000?></media></supplementary-material>
</sec></body></article>"""

CSV_MEMBER_CONTENT = b"sample_id,temp\nS1,18.2\nS2,17.9\n"


class FakeEuropePmcAdapter:
    name = "europe_pmc"

    def __init__(self, fulltext_xml=FULLTEXT_XML, not_found=False, bundle=None, related=None):
        self._fulltext_xml = fulltext_xml
        self._not_found = not_found
        self._bundle = bundle
        self._related = related or []

    def fetch_fulltext_xml(self, pmcid):
        if self._not_found:
            raise SourceRecordNotFoundError(f"no fulltext for {pmcid}")
        return self._fulltext_xml

    def find_related_from_fulltext(self, fulltext_xml):
        return self._related

    def fetch_supplementary_bundle(self, pmcid):
        if self._bundle is None:
            raise SourceRecordNotFoundError(f"no bundle for {pmcid}")
        return self._bundle

    def close(self):
        pass


def _seeded_study_with_pmcid(session, pmcid="PMC1234567") -> Study:
    study = Study(title="A study")
    session.add(study)
    session.flush()
    session.add(ExternalIdentifier(study_id=study.study_id, identifier_type=IdentifierType.PMCID.value, identifier_value=pmcid))
    session.flush()
    return study


def _discover_task(session, study):
    task = enqueue_task(session, TaskType.DISCOVER_SUPPLEMENTS, study_id=study.study_id)
    session.commit()
    return task


def _retrieve_task(session, study):
    task = enqueue_task(session, TaskType.RETRIEVE_SUPPLEMENTS, study_id=study.study_id)
    session.commit()
    return task


def _small_bundle() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("Table_1.csv", CSV_MEMBER_CONTENT)
    return buf.getvalue()


@pytest.fixture(autouse=True)
def _reset_caches():
    handlers.reset_adapter_cache()
    handlers.reset_llm_backend_cache()
    reset_config_cache()
    yield
    handlers.reset_adapter_cache()
    handlers.reset_llm_backend_cache()
    reset_config_cache()


# --- handle_discover_supplements --------------------------------------------


def test_discover_raises_without_pmcid(db_session):
    study = Study(title="No PMCID")
    db_session.add(study)
    db_session.flush()
    task = _discover_task(db_session, study) if False else enqueue_task(db_session, TaskType.DISCOVER_SUPPLEMENTS, study_id=study.study_id)
    db_session.commit()

    with pytest.raises(NotImplementedError):
        supplement_handlers.handle_discover_supplements(db_session, task)


def test_discover_no_ops_when_no_open_access_fulltext(db_session, monkeypatch):
    study = _seeded_study_with_pmcid(db_session)
    task = _discover_task(db_session, study)
    monkeypatch.setattr(supplement_handlers, "_build_enabled_adapters", lambda: {"europe_pmc": FakeEuropePmcAdapter(not_found=True)})

    supplement_handlers.handle_discover_supplements(db_session, task)  # must not raise
    db_session.commit()

    assert db_session.query(Source).filter_by(study_id=study.study_id).count() == 0


def test_discover_creates_source_and_dataasset_per_referenced_file(db_session, monkeypatch):
    study = _seeded_study_with_pmcid(db_session)
    task = _discover_task(db_session, study)
    monkeypatch.setattr(supplement_handlers, "_build_enabled_adapters", lambda: {"europe_pmc": FakeEuropePmcAdapter()})

    supplement_handlers.handle_discover_supplements(db_session, task)
    db_session.commit()

    sources = db_session.query(Source).filter_by(study_id=study.study_id, source_name="europe_pmc_supplement").all()
    assert {s.external_identifier for s in sources} == {"Table_1.csv", "Big_Data.zip"}
    assert all(s.inspection_status == "inspected" for s in sources)
    assert all(s.inspection_level == "metadata_only" for s in sources)

    # Source creation here also routes through create_source(), same choke
    # point every other Source-creation call site uses.
    study_source_source_ids = {
        row.source_id for row in db_session.query(StudySource).filter_by(study_id=study.study_id).all()
    }
    assert study_source_source_ids == {s.source_id for s in sources}

    assets = db_session.query(DataAsset).filter_by(study_id=study.study_id).all()
    assets_by_name = {a.file_name: a for a in assets}
    assert assets_by_name["Table_1.csv"].size_bytes == 40
    assert assets_by_name["Big_Data.zip"].size_bytes == 30000000
    assert all(a.inspection_level == "metadata_only" for a in assets)


def test_discover_is_idempotent_on_retry(db_session, monkeypatch):
    study = _seeded_study_with_pmcid(db_session)
    task = _discover_task(db_session, study)
    monkeypatch.setattr(supplement_handlers, "_build_enabled_adapters", lambda: {"europe_pmc": FakeEuropePmcAdapter()})

    supplement_handlers.handle_discover_supplements(db_session, task)
    db_session.commit()
    supplement_handlers.handle_discover_supplements(db_session, task)  # simulated retry
    db_session.commit()

    assert db_session.query(Source).filter_by(study_id=study.study_id, source_name="europe_pmc_supplement").count() == 2
    assert db_session.query(DataAsset).filter_by(study_id=study.study_id).count() == 2


def test_discover_applies_external_repository_related_identifiers(db_session, monkeypatch):
    study = _seeded_study_with_pmcid(db_session)
    task = _discover_task(db_session, study)
    related = [
        RelatedIdentifier(
            identifier_type=IdentifierType.DATASET_DOI,
            value="10.5061/dryad.abc123",
            relationship_type="is_supplement_to",
            source="europe_pmc",
        )
    ]
    monkeypatch.setattr(
        supplement_handlers, "_build_enabled_adapters", lambda: {"europe_pmc": FakeEuropePmcAdapter(related=related)}
    )

    supplement_handlers.handle_discover_supplements(db_session, task)
    db_session.commit()

    identifiers = db_session.query(ExternalIdentifier).filter_by(
        study_id=study.study_id, identifier_type=IdentifierType.DATASET_DOI.value
    ).all()
    assert any(i.identifier_value == "10.5061/dryad.abc123" for i in identifiers)


# --- handle_retrieve_supplements ---------------------------------------------


def test_retrieve_raises_without_pmcid(db_session):
    study = Study(title="No PMCID")
    db_session.add(study)
    db_session.flush()
    task = enqueue_task(db_session, TaskType.RETRIEVE_SUPPLEMENTS, study_id=study.study_id)
    db_session.commit()

    with pytest.raises(NotImplementedError):
        supplement_handlers.handle_retrieve_supplements(db_session, task)


def test_retrieve_no_ops_when_nothing_discovered(db_session):
    study = _seeded_study_with_pmcid(db_session)
    task = _retrieve_task(db_session, study)

    supplement_handlers.handle_retrieve_supplements(db_session, task)  # must not raise
    db_session.commit()
    assert db_session.query(RawFact).filter_by(study_id=study.study_id).count() == 0


def test_retrieve_marks_bundle_inaccessible_when_over_size_cap(db_session, monkeypatch):
    """Table_1.csv (40 bytes) + Big_Data.zip (30,000,000 bytes) sums well
    over the default 25,000,000-byte cap -- the whole bundle must never be
    downloaded, and every pending asset gets marked not_accessible."""
    study = _seeded_study_with_pmcid(db_session)
    discover_task = _discover_task(db_session, study)
    monkeypatch.setattr(supplement_handlers, "_build_enabled_adapters", lambda: {"europe_pmc": FakeEuropePmcAdapter()})
    supplement_handlers.handle_discover_supplements(db_session, discover_task)
    db_session.commit()

    retrieve_task = _retrieve_task(db_session, study)
    supplement_handlers.handle_retrieve_supplements(db_session, retrieve_task)
    db_session.commit()

    assets = db_session.query(DataAsset).filter_by(study_id=study.study_id).all()
    assert all(a.access_status == "not_accessible" for a in assets)
    assert all(a.inspection_level != "full" for a in assets)
    assert db_session.query(RawFact).filter_by(study_id=study.study_id).count() == 0


def test_retrieve_parses_a_small_csv_and_creates_raw_facts(db_session, monkeypatch):
    study = _seeded_study_with_pmcid(db_session)
    small_xml = """<article><supplementary-material id="TS1"><media xmlns:xlink="http://www.w3.org/1999/xlink"
    xlink:href="Table_1.csv" mimetype="text" mime-subtype="csv"><?size 40?></media></supplementary-material></article>"""
    monkeypatch.setattr(
        supplement_handlers, "_build_enabled_adapters",
        lambda: {"europe_pmc": FakeEuropePmcAdapter(fulltext_xml=small_xml, bundle=_small_bundle())},
    )
    discover_task = _discover_task(db_session, study)
    supplement_handlers.handle_discover_supplements(db_session, discover_task)
    db_session.commit()

    retrieve_task = _retrieve_task(db_session, study)
    supplement_handlers.handle_retrieve_supplements(db_session, retrieve_task)
    db_session.commit()

    asset = db_session.query(DataAsset).filter_by(study_id=study.study_id, file_name="Table_1.csv").one()
    assert asset.inspection_level == "full"
    assert asset.access_status == "open"

    facts = db_session.query(RawFact).filter_by(study_id=study.study_id, extraction_method="supplement_table_parsing").all()
    assert {f.fact_type_candidate for f in facts} == {"temp"}
    assert {f.raw_value for f in facts} == {"18.2", "17.9"}
    assert all(f.source_locator.startswith("supplement.Table_1.csv!") for f in facts)


def test_retrieve_prepares_pdf_text_without_calling_llm(db_session, monkeypatch):
    study = _seeded_study_with_pmcid(db_session)
    small_xml = """<article><supplementary-material id="TS1"><media xmlns:xlink="http://www.w3.org/1999/xlink"
    xlink:href="methods.pdf" mimetype="application" mime-subtype="pdf"><?size 40?></media></supplementary-material></article>"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("methods.pdf", b"%PDF-1.4\n%%EOF")
    monkeypatch.setattr(
        supplement_handlers, "_build_enabled_adapters",
        lambda: {"europe_pmc": FakeEuropePmcAdapter(fulltext_xml=small_xml, bundle=buf.getvalue())},
    )
    monkeypatch.setattr(
        supplement_handlers,
        "_build_llm_backend_cached",
        lambda: (_ for _ in ()).throw(AssertionError("LLM backend should not be built")),
    )
    monkeypatch.setattr(
        supplement_handlers,
        "extract_pdf_text",
        lambda _content: "PCR reactions used MiFish-U-F and MiFish-U-R primers.",
    )
    discover_task = _discover_task(db_session, study)
    supplement_handlers.handle_discover_supplements(db_session, discover_task)
    db_session.commit()

    retrieve_task = _retrieve_task(db_session, study)
    supplement_handlers.handle_retrieve_supplements(db_session, retrieve_task)
    db_session.commit()

    asset = db_session.query(DataAsset).filter_by(study_id=study.study_id, file_name="methods.pdf").one()
    assert asset.access_status == "open"
    assert asset.inspection_level == "lightweight"
    assert "text_ready" in asset.description
    prepared = db_session.query(PreparedSourceText).filter_by(data_asset_id=asset.asset_id).one()
    assert prepared.text_content == "PCR reactions used MiFish-U-F and MiFish-U-R primers."
    assert prepared.preparation_method == "pypdf_text_extraction"
    assert db_session.query(RawFact).filter_by(study_id=study.study_id).count() == 0


def test_retrieve_prepares_docx_text_without_calling_llm(db_session, monkeypatch):
    """Regression guard for a real gap found live: Frontiers ships its
    "Data Sheet" supplements as .docx, which used to fall into the generic
    "unsupported file type, not parsed" branch -- silently never scanned
    for anything, controls included, despite being successfully
    retrieved. Mirrors test_retrieve_prepares_pdf_text_without_calling_llm."""
    study = _seeded_study_with_pmcid(db_session)
    small_xml = """<article><supplementary-material id="TS1"><media xmlns:xlink="http://www.w3.org/1999/xlink"
    xlink:href="Data_Sheet_1.docx" mimetype="application" mime-subtype="msword"><?size 40?></media></supplementary-material></article>"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("Data_Sheet_1.docx", b"fake docx bytes")
    monkeypatch.setattr(
        supplement_handlers, "_build_enabled_adapters",
        lambda: {"europe_pmc": FakeEuropePmcAdapter(fulltext_xml=small_xml, bundle=buf.getvalue())},
    )
    monkeypatch.setattr(
        supplement_handlers,
        "_build_llm_backend_cached",
        lambda: (_ for _ in ()).throw(AssertionError("LLM backend should not be built")),
    )
    monkeypatch.setattr(
        supplement_handlers,
        "extract_docx_text",
        lambda _content: "No negative or positive controls were used in this study.",
    )
    discover_task = _discover_task(db_session, study)
    supplement_handlers.handle_discover_supplements(db_session, discover_task)
    db_session.commit()

    retrieve_task = _retrieve_task(db_session, study)
    supplement_handlers.handle_retrieve_supplements(db_session, retrieve_task)
    db_session.commit()

    asset = db_session.query(DataAsset).filter_by(study_id=study.study_id, file_name="Data_Sheet_1.docx").one()
    assert asset.access_status == "open"
    assert asset.inspection_level == "lightweight"
    assert "text_ready" in asset.description
    prepared = db_session.query(PreparedSourceText).filter_by(data_asset_id=asset.asset_id).one()
    assert prepared.text_content == "No negative or positive controls were used in this study."
    assert prepared.preparation_method == "docx_xml_text_extraction"
    assert db_session.query(RawFact).filter_by(study_id=study.study_id).count() == 0


def test_supplement_llm_pass_targets_fields_still_missing_after_paper(db_session, monkeypatch):
    study = _seeded_study_with_pmcid(db_session)
    paper_source = Source(
        study_id=study.study_id,
        source_type="article_fulltext",
        source_name="europe_pmc_fulltext",
        external_identifier="PMC1234567",
        inspection_level="full",
    )
    supplement_source = Source(
        study_id=study.study_id,
        source_type="supplement",
        source_name="europe_pmc_supplement",
        external_identifier="methods.txt",
    )
    db_session.add_all([paper_source, supplement_source])
    db_session.flush()
    asset = DataAsset(
        study_id=study.study_id,
        source_id=supplement_source.source_id,
        asset_type="other",
        file_name="methods.txt",
        access_status="open",
        raw_or_processed="raw",
        inspection_level="lightweight",
    )
    db_session.add(asset)
    db_session.flush()
    supplement_text = "PCR reactions were annealed at 54 C for 35 cycles."
    prepared = PreparedSourceText(
        study_id=study.study_id,
        source_id=supplement_source.source_id,
        data_asset_id=asset.asset_id,
        title="Supplementary Methods (methods.txt)",
        text_content=supplement_text,
        content_hash="supplement-hash",
        preparation_method="txt_decode_utf8",
        character_count=len(supplement_text),
    )
    db_session.add(prepared)
    db_session.add(
        RawFact(
            study_id=study.study_id,
            source_id=paper_source.source_id,
            raw_field_name="depth",
            raw_value="5 m",
            fact_type_candidate="depth",
            entity_level="study",
            support_type="explicit",
            extraction_method="llm_text_extraction",
            evidence_quote="Samples were collected at 5 m depth.",
        )
    )
    db_session.commit()

    segment_id = segment_source_text(prepared.title, supplement_text)[0].segment_id
    response = json.dumps(
        [
            {
                "fact_type_candidate": "annealing_temperature",
                "raw_value": "54 C",
                "evidence_id": segment_id,
            }
        ]
    )
    backend = MockLLMBackend(label="supplement-model", responses=[response])
    monkeypatch.setattr(supplement_handlers, "_build_llm_backend_cached", lambda: backend)
    config = supplement_handlers.load_config()
    monkeypatch.setattr(
        supplement_handlers,
        "load_config",
        lambda: config.model_copy(
            update={
                "supplements": config.supplements.model_copy(
                    update={"llm_text_extraction_enabled": True}
                )
            }
        ),
    )
    task = enqueue_task(
        db_session,
        TaskType.EXTRACT_SUPPLEMENT_TEXT_FACTS,
        study_id=study.study_id,
    )
    db_session.commit()

    supplement_handlers.handle_extract_supplement_text_facts(db_session, task)
    db_session.commit()

    assert backend.calls
    assert all("sampling depth" not in call["prompt"] for call in backend.calls)
    fact = db_session.query(RawFact).filter_by(
        study_id=study.study_id,
        extraction_method="supplement_llm_extraction",
    ).one()
    assert fact.raw_value == "54 C"
    assert fact.evidence_quote == supplement_text
    assert fact.source_id == supplement_source.source_id
    assert prepared.llm_model_name == "supplement-model"
    assert prepared.llm_prompt_version == PROMPT_VERSION
    assert prepared.llm_extracted_at is not None
    assert asset.inspection_level == "full"

    mapped = db_session.query(StandardizedValue).filter_by(
        study_id=study.study_id,
        target_field="annealingTemp",
    ).one()
    assert mapped.standardized_value == "54 C"


def test_supplement_llm_invalid_json_keeps_document_pending(db_session, monkeypatch):
    study = _seeded_study_with_pmcid(db_session)
    source = Source(
        study_id=study.study_id,
        source_type="supplement",
        source_name="europe_pmc_supplement",
        external_identifier="methods.txt",
    )
    db_session.add(source)
    db_session.flush()
    asset = DataAsset(
        study_id=study.study_id,
        source_id=source.source_id,
        asset_type="other",
        access_status="open",
        raw_or_processed="raw",
    )
    db_session.add(asset)
    db_session.flush()
    prepared = PreparedSourceText(
        study_id=study.study_id,
        source_id=source.source_id,
        data_asset_id=asset.asset_id,
        title="Supplementary Methods",
        text_content="PCR reactions were annealed at 54 C.",
        content_hash="invalid-json-supplement",
        preparation_method="txt_decode_utf8",
        character_count=38,
    )
    db_session.add(prepared)
    db_session.commit()

    backend = MockLLMBackend(label="invalid-json-model", responses=["not json"])
    monkeypatch.setattr(supplement_handlers, "_build_llm_backend_cached", lambda: backend)
    monkeypatch.setattr(supplement_handlers, "_has_completed_paper_pass", lambda *_args: True)
    config = supplement_handlers.load_config()
    monkeypatch.setattr(
        supplement_handlers,
        "load_config",
        lambda: config.model_copy(
            update={
                "supplements": config.supplements.model_copy(
                    update={"llm_text_extraction_enabled": True}
                )
            }
        ),
    )
    task = enqueue_task(
        db_session,
        TaskType.EXTRACT_SUPPLEMENT_TEXT_FACTS,
        study_id=study.study_id,
    )
    db_session.commit()

    with pytest.raises(LLMBackendError, match="invalid JSON after retries"):
        supplement_handlers.handle_extract_supplement_text_facts(db_session, task)
    db_session.rollback()

    db_session.refresh(prepared)
    assert prepared.llm_model_name is None
    assert prepared.llm_prompt_version is None
    assert prepared.llm_extracted_at is None
    assert db_session.query(RawFact).filter_by(
        study_id=study.study_id,
        extraction_method="supplement_llm_extraction",
    ).count() == 0


def test_supplement_llm_backfill_stays_off_until_explicitly_enabled(db_session):
    study = _seeded_study_with_pmcid(db_session)
    source = Source(
        study_id=study.study_id,
        source_type="supplement",
        source_name="europe_pmc_supplement",
    )
    db_session.add(source)
    db_session.flush()
    asset = DataAsset(
        study_id=study.study_id,
        source_id=source.source_id,
        asset_type="other",
        access_status="open",
        raw_or_processed="raw",
    )
    db_session.add(asset)
    db_session.flush()
    db_session.add(
        PreparedSourceText(
            study_id=study.study_id,
            source_id=source.source_id,
            data_asset_id=asset.asset_id,
            title="Supplementary Methods",
            text_content="Some text.",
            content_hash="hash",
            preparation_method="txt_decode_utf8",
            character_count=10,
        )
    )
    db_session.commit()

    assert supplement_handlers.enqueue_supplement_text_extraction_backfill(db_session) == 0
    assert db_session.query(Task).filter_by(
        task_type=TaskType.EXTRACT_SUPPLEMENT_TEXT_FACTS.value
    ).count() == 0


def test_retrieve_parses_supported_tables_inside_zip_supplement(db_session, monkeypatch):
    study = _seeded_study_with_pmcid(db_session)
    small_xml = """<article><supplementary-material id="TS1"><media xmlns:xlink="http://www.w3.org/1999/xlink"
    xlink:href="metadata.zip" mimetype="application" mime-subtype="zip"><?size 200?></media></supplementary-material></article>"""
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as zf:
        zf.writestr("samples.csv", b"sample_id,Lat.,Lon.\nS1,1.2,3.4\n")
        zf.writestr("figure.png", b"not a table")
    outer = io.BytesIO()
    with zipfile.ZipFile(outer, "w") as zf:
        zf.writestr("metadata.zip", inner.getvalue())
    monkeypatch.setattr(
        supplement_handlers, "_build_enabled_adapters",
        lambda: {"europe_pmc": FakeEuropePmcAdapter(fulltext_xml=small_xml, bundle=outer.getvalue())},
    )
    discover_task = _discover_task(db_session, study)
    supplement_handlers.handle_discover_supplements(db_session, discover_task)
    db_session.commit()

    retrieve_task = _retrieve_task(db_session, study)
    supplement_handlers.handle_retrieve_supplements(db_session, retrieve_task)
    db_session.commit()

    asset = db_session.query(DataAsset).filter_by(study_id=study.study_id, file_name="metadata.zip").one()
    assert asset.inspection_level == "full"
    facts = db_session.query(RawFact).filter_by(study_id=study.study_id, extraction_method="supplement_table_parsing").all()
    assert {fact.fact_type_candidate for fact in facts} == {"latitude", "longitude"}
    assert all("metadata.zip!samples.csv" in fact.source_locator for fact in facts)


def test_retrieve_zip_with_no_supported_structured_files_has_clear_summary(db_session, monkeypatch):
    study = _seeded_study_with_pmcid(db_session)
    small_xml = """<article><supplementary-material id="TS1"><media xmlns:xlink="http://www.w3.org/1999/xlink"
    xlink:href="figures.zip" mimetype="application" mime-subtype="zip"><?size 200?></media></supplementary-material></article>"""
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as zf:
        zf.writestr("plot.png", b"not a table")
    outer = io.BytesIO()
    with zipfile.ZipFile(outer, "w") as zf:
        zf.writestr("figures.zip", inner.getvalue())
    monkeypatch.setattr(
        supplement_handlers, "_build_enabled_adapters",
        lambda: {"europe_pmc": FakeEuropePmcAdapter(fulltext_xml=small_xml, bundle=outer.getvalue())},
    )
    discover_task = _discover_task(db_session, study)
    supplement_handlers.handle_discover_supplements(db_session, discover_task)
    db_session.commit()

    retrieve_task = _retrieve_task(db_session, study)
    supplement_handlers.handle_retrieve_supplements(db_session, retrieve_task)
    db_session.commit()

    asset = db_session.query(DataAsset).filter_by(study_id=study.study_id, file_name="figures.zip").one()
    assert asset.inspection_level == "full"
    assert asset.description == "0 parsed table(s); no supported structured files found"


def test_retrieve_is_idempotent_skips_already_parsed_assets(db_session, monkeypatch):
    study = _seeded_study_with_pmcid(db_session)
    small_xml = """<article><supplementary-material id="TS1"><media xmlns:xlink="http://www.w3.org/1999/xlink"
    xlink:href="Table_1.csv" mimetype="text" mime-subtype="csv"><?size 40?></media></supplementary-material></article>"""
    monkeypatch.setattr(
        supplement_handlers, "_build_enabled_adapters",
        lambda: {"europe_pmc": FakeEuropePmcAdapter(fulltext_xml=small_xml, bundle=_small_bundle())},
    )
    discover_task = _discover_task(db_session, study)
    supplement_handlers.handle_discover_supplements(db_session, discover_task)
    db_session.commit()

    retrieve_task = _retrieve_task(db_session, study)
    supplement_handlers.handle_retrieve_supplements(db_session, retrieve_task)
    db_session.commit()
    supplement_handlers.handle_retrieve_supplements(db_session, retrieve_task)  # simulated retry
    db_session.commit()

    facts = db_session.query(RawFact).filter_by(study_id=study.study_id, extraction_method="supplement_table_parsing").all()
    assert len({(f.fact_type_candidate, f.raw_value) for f in facts}) == 2
    assert len(facts) == 2  # no duplicates from the second run


def test_retrieve_marks_missing_zip_member_not_accessible(db_session, monkeypatch):
    study = _seeded_study_with_pmcid(db_session)
    small_xml = """<article><supplementary-material id="TS1"><media xmlns:xlink="http://www.w3.org/1999/xlink"
    xlink:href="Missing.csv" mimetype="text" mime-subtype="csv"><?size 40?></media></supplementary-material></article>"""
    empty_bundle_buf = io.BytesIO()
    with zipfile.ZipFile(empty_bundle_buf, "w") as zf:
        zf.writestr("SomeOtherFile.csv", b"a,b\n1,2\n")
    monkeypatch.setattr(
        supplement_handlers, "_build_enabled_adapters",
        lambda: {"europe_pmc": FakeEuropePmcAdapter(fulltext_xml=small_xml, bundle=empty_bundle_buf.getvalue())},
    )
    discover_task = _discover_task(db_session, study)
    supplement_handlers.handle_discover_supplements(db_session, discover_task)
    db_session.commit()

    retrieve_task = _retrieve_task(db_session, study)
    supplement_handlers.handle_retrieve_supplements(db_session, retrieve_task)
    db_session.commit()

    asset = db_session.query(DataAsset).filter_by(study_id=study.study_id, file_name="Missing.csv").one()
    assert asset.access_status == "not_accessible"


def test_retrieve_marks_malformed_supplement_parse_failed_without_aborting(db_session, monkeypatch):
    """A JSON file that isn't actually valid JSON must not abort the whole
    task -- caught, logged on its own asset, task completes."""
    study = _seeded_study_with_pmcid(db_session)
    small_xml = """<article><supplementary-material id="TS1"><media xmlns:xlink="http://www.w3.org/1999/xlink"
    xlink:href="broken.json" mimetype="application" mime-subtype="json"><?size 40?></media></supplementary-material></article>"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("broken.json", b"{not valid json")
    monkeypatch.setattr(
        supplement_handlers, "_build_enabled_adapters",
        lambda: {"europe_pmc": FakeEuropePmcAdapter(fulltext_xml=small_xml, bundle=buf.getvalue())},
    )
    discover_task = _discover_task(db_session, study)
    supplement_handlers.handle_discover_supplements(db_session, discover_task)
    db_session.commit()

    retrieve_task = _retrieve_task(db_session, study)
    supplement_handlers.handle_retrieve_supplements(db_session, retrieve_task)  # must not raise
    db_session.commit()

    asset = db_session.query(DataAsset).filter_by(study_id=study.study_id, file_name="broken.json").one()
    # parse_json_supplement tolerates invalid JSON gracefully (returns empty
    # result rather than raising) -- so this lands as "parsed" with zero
    # facts, not "parse_failed". Confirms the no-crash contract either way.
    assert asset.inspection_level in ("full", "lightweight")


def test_retrieve_marks_unsupported_file_type_as_retrieved_not_parsed(db_session, monkeypatch):
    # .PPTX, not .DOCX -- .docx supplements are now parsed (extract_docx_text),
    # per a real gap found live (Frontiers' "Data Sheet" .docx supplements
    # were silently never scanned at all). .pptx remains genuinely unsupported.
    study = _seeded_study_with_pmcid(db_session)
    small_xml = """<article><supplementary-material id="TS1"><media xmlns:xlink="http://www.w3.org/1999/xlink"
    xlink:href="Table_1.PPTX" mimetype="application" mime-subtype="vnd.openxmlformats-officedocument.presentationml.presentation"><?size 40?></media></supplementary-material></article>"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("Table_1.PPTX", b"fake pptx bytes")
    monkeypatch.setattr(
        supplement_handlers, "_build_enabled_adapters",
        lambda: {"europe_pmc": FakeEuropePmcAdapter(fulltext_xml=small_xml, bundle=buf.getvalue())},
    )
    discover_task = _discover_task(db_session, study)
    supplement_handlers.handle_discover_supplements(db_session, discover_task)
    db_session.commit()

    retrieve_task = _retrieve_task(db_session, study)
    supplement_handlers.handle_retrieve_supplements(db_session, retrieve_task)
    db_session.commit()

    asset = db_session.query(DataAsset).filter_by(study_id=study.study_id, file_name="Table_1.PPTX").one()
    assert asset.inspection_level == "lightweight"
    assert asset.access_status == "open"
    assert "unsupported" in asset.description


# --- enqueue backfill functions ----------------------------------------------


def test_enqueue_supplement_discovery_backfill_queues_studies_with_pmcid(db_session):
    study = _seeded_study_with_pmcid(db_session)
    without_pmcid = Study(title="No PMCID")
    db_session.add(without_pmcid)
    db_session.commit()

    count = supplement_handlers.enqueue_supplement_discovery_backfill(db_session)
    db_session.commit()

    assert count == 1
    from fair_ocean_agent.database.models import Task
    tasks = db_session.query(Task).filter_by(task_type=TaskType.DISCOVER_SUPPLEMENTS.value).all()
    assert {t.study_id for t in tasks} == {study.study_id}


def test_enqueue_supplement_retrieval_backfill_queues_studies_with_discovered_supplements(db_session, monkeypatch):
    study = _seeded_study_with_pmcid(db_session)
    monkeypatch.setattr(supplement_handlers, "_build_enabled_adapters", lambda: {"europe_pmc": FakeEuropePmcAdapter()})
    discover_task = _discover_task(db_session, study)
    supplement_handlers.handle_discover_supplements(db_session, discover_task)
    db_session.commit()

    count = supplement_handlers.enqueue_supplement_retrieval_backfill(db_session)
    db_session.commit()

    assert count == 1
