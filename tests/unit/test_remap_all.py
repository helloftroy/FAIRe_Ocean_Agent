"""scripts/remap_all.py (2026-11-11)."""
import sys
from pathlib import Path

from sqlalchemy import select

from fair_ocean_agent.database.models import Entity, EntityStudy, RawFact, Study, Task
from fair_ocean_agent.workflow.task_queue import enqueue_task
from fair_ocean_agent.database.enums import TaskType

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import remap_all  # noqa: E402


def test_remap_all_settles_and_queues_every_study(db_session):
    s = db_session
    s.add_all([Study(study_id="S1", title="a", entity_component_status="pending"),
               Study(study_id="S2", title="b", entity_component_status="pending"),
               Study(study_id="S3", title="merged", canonical_status="merged")])
    s.add(Entity(entity_id="E1", study_id="S1", entity_level="sample", external_identifier="SAMN1", root_status="pending"))
    s.add_all([EntityStudy(entity_id="E1", study_id="S1", relationship_type="home", confidence="structured_source"),
               EntityStudy(entity_id="E1", study_id="S2", relationship_type="shares_accession_with", confidence="structured_source")])
    for sid in ("S1", "S2", "S3"):
        s.add(RawFact(study_id=sid, entity_id="E1", fact_type_candidate="organism", raw_value="marine metagenome",
                      entity_level="sample", support_type="structured_source", review_status="accepted"))
    s.commit()
    # S1 was mapped before (completed task); a stale settle check is pending
    done = enqueue_task(s, TaskType.MAP_FAIRE, study_id="S1")
    done.status = "completed"
    enqueue_task(s, TaskType.CHECK_COMPONENT_SETTLED, study_id="S2", idempotency_key="CHECK:S2:0")
    s.commit()

    result = remap_all.remap_all(s, "remap-test")

    assert result["studies"] == 2 and result["components"] == 1
    assert result["cancelled_settle_checks"] == 1
    statuses = dict(s.execute(select(Study.study_id, Study.entity_component_status)).all())
    assert statuses["S1"] == statuses["S2"] == "settled"
    pending = s.scalars(select(Task.study_id).where(Task.task_type == "MAP_FAIRE", Task.status == "pending")).all()
    assert sorted(pending) == ["S1", "S2"]          # one each: S1 re-mapped despite its completed task
    assert s.get(Entity, "E1").root_status != "pending"

    # running it again does not queue duplicates
    again = remap_all.remap_all(s, "remap-test")
    assert again["queued_map"] == 0


def test_remap_only_listed_studies_and_their_component(db_session):
    s = db_session
    s.add_all([Study(study_id=x, title=x) for x in ("S1", "S2", "S4")])
    s.add(Entity(entity_id="E1", study_id="S1", entity_level="sample", external_identifier="SAMN1", root_status="pending"))
    s.add_all([EntityStudy(entity_id="E1", study_id="S1", relationship_type="home", confidence="structured_source"),
               EntityStudy(entity_id="E1", study_id="S2", relationship_type="shares_accession_with", confidence="structured_source")])
    for sid in ("S1", "S2", "S4"):
        s.add(RawFact(study_id=sid, entity_id="E1" if sid != "S4" else None, fact_type_candidate="organism",
                      raw_value="marine metagenome", entity_level="sample", support_type="structured_source",
                      review_status="accepted"))
    s.commit()
    result = remap_all.remap_all(s, "newpdf-test", only={"S1"})
    pending = s.scalars(select(Task.study_id).where(Task.task_type == "MAP_FAIRE", Task.status == "pending")).all()
    assert sorted(pending) == ["S1", "S2"]          # S2 shares a sample with S1; S4 untouched
    assert result["studies"] == 2 and result["components"] == 1   # S1 plus linked S2
