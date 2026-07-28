from fair_ocean_agent.database.models import SourceWatermark
from fair_ocean_agent.scheduling.watermarks import get_or_create_watermark, record_check


def test_get_or_create_watermark_creates_once(db_session):
    first = get_or_create_watermark(db_session, "crossref", "10.1/abc")
    db_session.commit()
    second = get_or_create_watermark(db_session, "crossref", "10.1/abc")

    assert first.watermark_id == second.watermark_id
    assert db_session.query(SourceWatermark).count() == 1


def test_get_or_create_watermark_is_per_source_and_identifier(db_session):
    get_or_create_watermark(db_session, "crossref", "10.1/abc")
    get_or_create_watermark(db_session, "openalex", "10.1/abc")
    get_or_create_watermark(db_session, "crossref", "10.1/xyz")
    db_session.commit()

    assert db_session.query(SourceWatermark).count() == 3


def test_record_check_updates_status_and_run_id(db_session):
    record_check(db_session, "ena", "PRJNA1", status="changed", run_id="RUN-1")
    db_session.commit()

    watermark = db_session.query(SourceWatermark).filter_by(source_name="ena", query_identifier="PRJNA1").one()
    assert watermark.last_status == "changed"
    assert watermark.last_run_id == "RUN-1"
    assert watermark.last_success_at is not None


def test_record_check_is_idempotent_key_wise_but_updates_in_place(db_session):
    record_check(db_session, "ena", "PRJNA1", status="unchanged")
    db_session.commit()
    first_success_at = db_session.query(SourceWatermark).filter_by(source_name="ena", query_identifier="PRJNA1").one().last_success_at

    record_check(db_session, "ena", "PRJNA1", status="changed")
    db_session.commit()

    assert db_session.query(SourceWatermark).filter_by(source_name="ena", query_identifier="PRJNA1").count() == 1
    watermark = db_session.query(SourceWatermark).filter_by(source_name="ena", query_identifier="PRJNA1").one()
    assert watermark.last_status == "changed"
    assert watermark.last_success_at >= first_success_at
