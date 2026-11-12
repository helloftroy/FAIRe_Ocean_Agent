"""scripts/match_new_pdfs.py on a small synthetic database (2026-11-12)."""
import csv
import sqlite3
import subprocess
import sys
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from fair_ocean_agent.database.models import Base, ExternalIdentifier, Study

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "match_new_pdfs.py"


def _db(path: Path) -> None:
    engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    s.add_all([
        Study(study_id="S-DOI", title="Anything"),
        Study(study_id="S-TITLE", title="Coral microbiomes of the Red Sea"),
        Study(study_id="S-OTHER", title="Seagrass fungi: a survey"),
        Study(study_id="S-DUP1", title="Same title"),
        Study(study_id="S-DUP2", title="Same title"),
    ])
    s.add(ExternalIdentifier(study_id="S-DOI", identifier_type="doi", identifier_value="10.1/known"))
    s.add(ExternalIdentifier(study_id="S-OTHER", identifier_type="doi", identifier_value="10.1/preprint"))
    s.commit()
    s.close()
    engine.dispose()


def test_match_and_apply(tmp_path):
    db = tmp_path / "t.db"
    _db(db)
    manifest = tmp_path / "manifest.csv"
    with open(manifest, "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["doi", "pdf_filename", "source", "title", "study_ids", "already_had_pdf"])
        w.writeheader()
        for doi, title in [("10.1/KNOWN", "x"), ("10.1/new", "Coral microbiomes of the Red Sea."),
                           ("10.1/journal", "Seagrass fungi - a survey"), ("10.1/dup", "Same title"),
                           ("10.1/none", "Nothing like it")]:
            w.writerow({"doi": doi, "pdf_filename": "", "source": "", "title": title, "study_ids": "", "already_had_pdf": "no"})
    out = tmp_path / "out"
    run = lambda *x: subprocess.run([sys.executable, str(SCRIPT), str(db), str(manifest), str(out), *x],  # noqa: E731
                                    check=True, capture_output=True, text=True)
    run()
    match = {r["doi"]: r["match"] for r in csv.DictReader(open(out / "pdf_matches.csv"))}
    assert match == {"10.1/KNOWN": "doi", "10.1/new": "title", "10.1/journal": "title_conflict",
                     "10.1/dup": "ambiguous", "10.1/none": "no_study"}
    assert (out / "studies.txt").read_text().split() == ["S-DOI", "S-TITLE"]
    conn = sqlite3.connect(db)
    assert conn.execute("select count(*) from external_identifiers where identifier_value='10.1/new'").fetchone() == (0,)
    run("--apply")
    assert conn.execute("select study_id from external_identifiers where identifier_value='10.1/new'").fetchall() == [("S-TITLE",)]
