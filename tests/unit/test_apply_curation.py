"""scripts/apply_curation.py on a small synthetic database (2026-11-11)."""
import sqlite3
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from fair_ocean_agent.database.models import (
    Base,
    Entity,
    EntityRelationship,
    EntityStudy,
    ExternalIdentifier,
    RawFact,
    Source,
    StandardizedValue,
    Study,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import apply_curation  # noqa: E402

A, B, C = "STUDY-A", "STUDY-B", "STUDY-C"


def _build(path: Path) -> None:
    engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    s.add_all([
        Study(study_id=A, title="Habitat specificity in sponge microbiomes from Antarctica"),
        Study(study_id=B, title="Coastal seawater bacterioplankton"),
        Study(study_id=C, title="Gut microbiota of children with inflammatory bowel disease"),
    ])
    srcs = {
        (A, "PRJEB1"): Source(source_id="SRC-A1", study_id=A, source_type="repository_api", source_name="ncbi_biosample",
                               external_identifier="PRJEB1", access_status="unknown", is_mirror=False,
                               inspection_status="inspected", inspection_level="metadata_only"),
        (A, "PRJNA2"): Source(source_id="SRC-A2", study_id=A, source_type="repository_api", source_name="ncbi_biosample",
                               external_identifier="PRJNA2", access_status="unknown", is_mirror=False,
                               inspection_status="inspected", inspection_level="metadata_only"),
        (B, "PRJNA9"): Source(source_id="SRC-B9", study_id=B, source_type="repository_api", source_name="ncbi_biosample",
                               external_identifier="PRJNA9", access_status="unknown", is_mirror=False,
                               inspection_status="inspected", inspection_level="metadata_only"),
        (C, "PRJNA7"): Source(source_id="SRC-C7", study_id=C, source_type="repository_api", source_name="ncbi_biosample",
                               external_identifier="PRJNA7", access_status="unknown", is_mirror=False,
                               inspection_status="inspected", inspection_level="metadata_only"),
    }
    s.add_all(srcs.values())
    samples = {  # entity: (home, accession, organism, [(study, bp)])
        "E1": (A, "SAMEA1", "sponge metagenome", [(A, "PRJEB1")]),
        "E2": (A, "SAMN2", "Mus musculus", [(A, "PRJNA2")]),
        "E3": (A, "SAMN3", "metagenome", [(A, "PRJNA2")]),
        "E4": (A, "SAMN4", "metagenome", [(A, "PRJNA2"), (B, "PRJNA9")]),
        "E6": (B, "SAMN6", "seawater metagenome", [(B, "PRJNA9")]),
        "E5": (C, "SAMN5", "Homo sapiens", [(C, "PRJNA7")]),
    }
    for e, (home, acc, organism, placed) in samples.items():
        s.add(Entity(entity_id=e, study_id=home, entity_level="sample", external_identifier=acc))
        for study, bp in placed:
            s.add(EntityStudy(entity_id=e, study_id=study, relationship_type="home" if study == home else "shares",
                              confidence="structured_source"))
            s.add(RawFact(study_id=study, entity_id=e, source_id=srcs[(study, bp)].source_id, fact_type_candidate="organism",
                          raw_value=organism, entity_level="sample", support_type="structured_source",
                          extraction_method="adapter:ncbi_biosample", review_status="accepted"))
            s.add(StandardizedValue(study_id=study, entity_id=e, target_schema="faire", target_schema_version="1.0.2",
                                    target_field="geo_loc_name", standardized_value="x", mapping_method="exact_label",
                                    validation_status="not_assessed"))
            s.add(ExternalIdentifier(study_id=study, identifier_type="biosample_accession", identifier_value=acc))
    for study, bp in srcs:
        s.add(ExternalIdentifier(study_id=study, identifier_type="bioproject_accession", identifier_value=bp))
    # runs: R3 from E3 (deleted with it), R4 from E4 (leaves A with it)
    for run, sample, home in (("R3", "E3", A), ("R4", "E4", A)):
        s.add(Entity(entity_id=run, study_id=home, entity_level="experiment_run", external_identifier=f"SRX{run}"))
        s.add(EntityStudy(entity_id=run, study_id=home, relationship_type="home", confidence="structured_source"))
        s.add(EntityRelationship(study_id=home, from_entity_id=run, to_entity_id=sample, relationship_type="derived_from_sample"))
    s.add(EntityStudy(entity_id="R4", study_id=B, relationship_type="shares", confidence="structured_source"))
    s.commit()
    s.close()
    engine.dispose()


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "t.db"
    _build(path)
    conn = sqlite3.connect(path)  # curation_exclusions comes from the model (create_all)
    yield conn
    conn.close()


AUDIT = [{"study_id": A, "bioproject": "PRJNA2", "verdict": "foreign_bug_chain", "bioproject_title": "mouse"}]


def test_plan(db):
    plan = apply_curation.make_plan(db, AUDIT, [])
    assert plan.delete["E2"][0] == "sample_non_marine:medical"
    assert plan.delete["E5"][0] == "sample_non_marine:medical"
    assert plan.delete["E3"][0] == "bp_foreign_bug_chain"        # only linked to A
    assert "R3" in plan.delete                                     # child run goes with it
    assert ("E4", A) in plan.unlink and "E4" not in plan.delete    # still wanted by B
    assert ("R4", A) in plan.unlink
    assert "E1" not in plan.delete and "E6" not in plan.delete
    assert plan.flag[C][0] == "not_relevant"
    assert plan.flag[A][0] == "relevant" and plan.flag[B][0] == "relevant"


def test_apply(db):
    plan = apply_curation.make_plan(db, AUDIT, [])
    apply_curation.apply_plan(db, plan, "test")
    q = lambda sql, *a: db.execute(sql, a).fetchall()  # noqa: E731
    assert {e for (e,) in q("select entity_id from entities")} == {"E1", "E4", "E6", "R4"}
    assert q("select study_id from entities where entity_id='E4'") == [(B,)]          # re-homed
    assert q("select study_id from entity_studies where entity_id='E4'") == [(B,)]
    assert q("select study_id from entity_studies where entity_id='R4'") == [(B,)]
    assert q("select count(*) from raw_facts where entity_id in ('E2','E3','E5')") == [(0,)]
    assert q("select study_id from raw_facts where entity_id='E4'") == [(B,)]
    assert q("select study_id from standardized_values where entity_id='E4'") == [(B,)]
    assert q("select count(*) from external_identifiers where identifier_value in ('SAMN2','SAMN3','SAMN5')") == [(0,)]
    assert q("select count(*) from external_identifiers where study_id=? and identifier_value='SAMN4'", A) == [(0,)]
    assert q("select count(*) from external_identifiers where study_id=? and identifier_value='PRJNA2'", A) == [(0,)]
    assert q("select count(*) from sources where source_id='SRC-A2'") == [(0,)]       # empty repository source removed
    assert q("select count(*) from sources where source_id='SRC-C7'") == [(0,)]
    assert q("select marine_relevance_status from studies where study_id=?", C) == [("not_relevant",)]
    actions = dict(q("select entity_id, action from curation_exclusions where level in ('sample','experiment_run') "
                     "and entity_id in ('E2','E4')"))
    assert actions == {"E2": "deleted", "E4": "unlinked"}


def test_tombstone_rerun_deletes_reappeared_sample(db):
    apply_curation.apply_plan(db, apply_curation.make_plan(db, AUDIT, []), "first")
    # discovery re-creates SAMN3 later, with no evidence at all
    db.execute("insert into entities (entity_id, study_id, entity_level, external_identifier, root_status, created_at, updated_at) "
               "values ('E3b', ?, 'sample', 'SAMN3', 'not_shared', '2026-11-11', '2026-11-11')", (A,))
    db.commit()
    plan = apply_curation.make_plan(db, [], [])
    assert plan.delete["E3b"][0] == "tombstone"


def test_shared_identifiers_and_srp_aliases_are_one_project(tmp_path):
    import curation_lib

    path = tmp_path / "a.db"
    engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    s.add(Study(study_id="S", title="Sponge microbiomes"))
    s.add_all([
        Source(source_id="ENA-PRJ", study_id="S", source_type="repository_api", source_name="ena",
               external_identifier="PRJNA1", access_status="unknown", is_mirror=False,
               inspection_status="inspected", inspection_level="metadata_only"),
        Source(source_id="ENA-SRP", study_id="S", source_type="repository_api", source_name="ena",
               external_identifier="SRP1", access_status="unknown", is_mirror=False,
               inspection_status="inspected", inspection_level="metadata_only"),
    ])
    s.add(RawFact(study_id="S", source_id="ENA-PRJ", fact_type_candidate="secondary_study_accession", raw_value="SRP1",
                  entity_level="project", support_type="structured_source", review_status="accepted"))
    s.add(Entity(entity_id="E", study_id="S", entity_level="sample", external_identifier="SAMN9"))
    s.add(RawFact(study_id="S", source_id="ENA-SRP", fact_type_candidate="samp_name", raw_value="SAMN9",
                  entity_level="experiment_run", support_type="structured_source", review_status="accepted"))
    s.add_all([
        ExternalIdentifier(study_id="S", identifier_type="bioproject_accession", identifier_value="PRJNA1"),
        ExternalIdentifier(study_id="S", identifier_type="sra_study_accession", identifier_value="SRP1"),
        ExternalIdentifier(study_id="S", identifier_type="bioproject_accession", identifier_value="PRJNA612578",
                           relationship_type="shares_accession_with", source="external_identifier_sequence_accession_resolution"),
    ])
    s.commit()
    s.close()
    conn = sqlite3.connect(path)
    aliases = curation_lib.project_aliases(conn)
    assert aliases == {"SRP1": "PRJNA1"}
    projects = {p for p, _, _ in curation_lib.bioproject_identifiers(conn, aliases)["S"]}
    assert projects == {"PRJNA1", "PRJNA612578"}            # SRP1 folded in; shared one seen
    assert curation_lib.placements(conn, ["S"], aliases) == {("S", "E"): "PRJNA1"}


def _clone(db, table, key, value, **changes):
    cols = [r[1] for r in db.execute(f"pragma table_info({table})")]
    row = dict(zip(cols, db.execute(f"select * from {table} where {key}=? limit 1", (value,)).fetchone()))
    row.update(changes)
    db.execute(f"insert into {table} ({','.join(cols)}) values ({','.join('?' * len(cols))})", [row[c] for c in cols])


def test_study_emptied_by_curation_with_no_habitat_title_is_not_relevant(db):
    _clone(db, "studies", "study_id", A, study_id="STUDY-D", title="Ants propleura microbial community", description=None)
    _clone(db, "entities", "entity_id", "E5", entity_id="E7", study_id="STUDY-D", external_identifier="SAMN7")
    es = db.execute("select entity_study_id from entity_studies where entity_id='E5'").fetchone()[0]
    _clone(db, "entity_studies", "entity_study_id", es, entity_study_id="ES-7", entity_id="E7", study_id="STUDY-D")
    fact = db.execute("select fact_id from raw_facts where entity_id='E5'").fetchone()[0]
    _clone(db, "raw_facts", "fact_id", fact, fact_id="F-7", entity_id="E7", study_id="STUDY-D")
    db.commit()
    plan = apply_curation.make_plan(db, AUDIT, [])
    assert plan.flag["STUDY-D"][0] == "not_relevant"
    assert plan.flag[A][0] == "relevant"      # still has its sponge sample
