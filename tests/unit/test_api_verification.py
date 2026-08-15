"""Tests for extraction/api_verification.py -- the narrowly-scoped
elev-vs-tot_depth_water_col mislabeling check (real audit:
10.1093/ismejo/wrae013, STUDY-295abf4a8f43)."""
import json

from fair_ocean_agent.database.enums import EntityLevel, ReviewStatus, SupportType
from fair_ocean_agent.database.models import ApiPaperCorrection, Entity, RawFact, StandardizedValue, Study
from fair_ocean_agent.extraction.api_verification import detect_and_correct_elev_depth_mislabeling
from fair_ocean_agent.llm.mock import MockLLMBackend
from fair_ocean_agent.mapping.faire import map_study_to_faire

_REAL_WATER_DEPTH_TEXT = (
    "Cores with intact sediment and bottom water were subsampled from a box corer at a site "
    "with 34 m water depth (Lat 59.8559, Long: 23.26695) previously shown to have high "
    "methanotrophic activity."
)


def _study(session, **kwargs) -> Study:
    study = Study(**kwargs)
    session.add(study)
    session.flush()
    return study


def _sample(session, study, external_identifier: str) -> Entity:
    entity = Entity(study_id=study.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier=external_identifier)
    session.add(entity)
    session.flush()
    return entity


def _fact(session, study, entity, field, value, support=SupportType.STRUCTURED_SOURCE):
    fact = RawFact(
        study_id=study.study_id, entity_id=entity.entity_id, raw_field_name=field, raw_value=value,
        fact_type_candidate=field, entity_level=EntityLevel.SAMPLE.value, support_type=support.value,
    )
    session.add(fact)
    session.flush()
    return fact


def _confirm_response(quote_id="Q001"):
    return json.dumps({"confirmed": True, "quote_id": quote_id})


def _no_confirm_response():
    return json.dumps({"confirmed": False, "quote_id": ""})


def test_corrects_elev_to_tot_depth_water_col_on_sediment_sample_with_paper_confirmation(db_session):
    study = _study(db_session, title="Sediment elev mislabel")
    sample = _sample(db_session, study, "SAMN1")
    _fact(db_session, study, sample, "elev", "34 m")
    _fact(db_session, study, sample, "env_medium", "sediment")
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    backend = MockLLMBackend(responses=[_confirm_response()])
    detect_and_correct_elev_depth_mislabeling(
        backend, db_session, study.study_id, [("Methods", _REAL_WATER_DEPTH_TEXT)],
        source_id=None, locator_prefix="test",
    )
    db_session.commit()
    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    values = {
        sv.target_field: sv.standardized_value
        for sv in db_session.query(StandardizedValue).filter_by(study_id=study.study_id, entity_id=sample.entity_id)
    }
    assert "elev" not in values
    assert values["tot_depth_water_col"] == "34 m"

    corrections = db_session.query(ApiPaperCorrection).filter_by(study_id=study.study_id).all()
    assert len(corrections) == 1
    assert corrections[0].api_faire_term == "elev"
    assert corrections[0].api_value == "34 m"
    assert corrections[0].corrected_faire_term == "tot_depth_water_col"
    assert corrections[0].corrected_value == "34"
    assert "34 m water depth" in corrections[0].supporting_quote


def test_does_not_correct_when_llm_does_not_confirm(db_session):
    study = _study(db_session, title="Sediment elev, no confirmation")
    sample = _sample(db_session, study, "SAMN1")
    _fact(db_session, study, sample, "elev", "34 m")
    _fact(db_session, study, sample, "env_medium", "sediment")
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    backend = MockLLMBackend(responses=[_no_confirm_response()])
    detect_and_correct_elev_depth_mislabeling(
        backend, db_session, study.study_id,
        [("Methods", "Unrelated text with the number 34 in it but no depth context at all.")],
        source_id=None, locator_prefix="test",
    )
    db_session.commit()

    assert db_session.query(ApiPaperCorrection).filter_by(study_id=study.study_id).count() == 0
    still_elev = db_session.query(RawFact).filter_by(
        study_id=study.study_id, entity_id=sample.entity_id, fact_type_candidate="elev"
    ).one()
    assert still_elev.review_status != ReviewStatus.REJECTED.value


def test_skips_water_column_sample_with_no_llm_call(db_session):
    """A water-column sample's own elev is a legitimate, real concept
    (per FAIRe's own definition: 0m for a seawater sample) -- the check
    must never even reach the LLM for a non-sediment/soil sample."""
    study = _study(db_session, title="Water column sample")
    sample = _sample(db_session, study, "SAMN1")
    _fact(db_session, study, sample, "elev", "0")
    _fact(db_session, study, sample, "env_medium", "sea water")
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    def fail_if_called(prompt: str) -> str:
        raise AssertionError("LLM should never be called for a water-column sample")

    backend = MockLLMBackend(responses=fail_if_called)
    detect_and_correct_elev_depth_mislabeling(
        backend, db_session, study.study_id, [("Methods", _REAL_WATER_DEPTH_TEXT)],
        source_id=None, locator_prefix="test",
    )
    assert db_session.query(ApiPaperCorrection).filter_by(study_id=study.study_id).count() == 0


def test_skips_sediment_sample_whose_tot_depth_water_col_is_already_populated(db_session):
    """If tot_depth_water_col already has a real value (from wherever), the
    check has nothing to fix and must never reach the LLM."""
    study = _study(db_session, title="Already-populated tot_depth_water_col")
    sample = _sample(db_session, study, "SAMN1")
    _fact(db_session, study, sample, "elev", "34 m")
    _fact(db_session, study, sample, "env_medium", "sediment")
    _fact(db_session, study, sample, "tot_depth_water_col", "34")
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    def fail_if_called(prompt: str) -> str:
        raise AssertionError("LLM should never be called when tot_depth_water_col is already populated")

    backend = MockLLMBackend(responses=fail_if_called)
    detect_and_correct_elev_depth_mislabeling(
        backend, db_session, study.study_id, [("Methods", _REAL_WATER_DEPTH_TEXT)],
        source_id=None, locator_prefix="test",
    )
    assert db_session.query(ApiPaperCorrection).filter_by(study_id=study.study_id).count() == 0


def test_one_llm_call_covers_every_sample_sharing_the_same_elev_value(db_session):
    study = _study(db_session, title="Many samples, one site elev")
    samples = [_sample(db_session, study, f"SAMN{i}") for i in range(1, 4)]
    for sample in samples:
        _fact(db_session, study, sample, "elev", "34 m")
        _fact(db_session, study, sample, "env_medium", "sediment")
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    backend = MockLLMBackend(responses=[_confirm_response()])
    detect_and_correct_elev_depth_mislabeling(
        backend, db_session, study.study_id, [("Methods", _REAL_WATER_DEPTH_TEXT)],
        source_id=None, locator_prefix="test",
    )
    db_session.commit()

    assert len(backend.calls) == 1
    assert db_session.query(ApiPaperCorrection).filter_by(study_id=study.study_id).count() == 3


def test_idempotent_rerun_does_not_duplicate_corrections(db_session):
    study = _study(db_session, title="Idempotent rerun")
    sample = _sample(db_session, study, "SAMN1")
    _fact(db_session, study, sample, "elev", "34 m")
    _fact(db_session, study, sample, "env_medium", "sediment")
    db_session.commit()

    map_study_to_faire(db_session, study.study_id)
    db_session.commit()

    backend = MockLLMBackend(responses=[_confirm_response(), _confirm_response()])
    for _ in range(2):
        detect_and_correct_elev_depth_mislabeling(
            backend, db_session, study.study_id, [("Methods", _REAL_WATER_DEPTH_TEXT)],
            source_id=None, locator_prefix="test",
        )
        db_session.commit()

    assert db_session.query(ApiPaperCorrection).filter_by(study_id=study.study_id).count() == 1
