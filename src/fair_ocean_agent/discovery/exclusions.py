"""A small, version-controlled "don't read this again" list: DOIs a human
has already manually confirmed have no sequence data worth chasing, so
future seed files (however they were generated) don't keep re-costing
discovery/extraction effort on the same dead paper. Checked at seed
ingestion time (discovery/seed_loader.py's ingest_seed_row), not at
discovery time -- the whole point is to skip the study before it's even
created, not to filter it out after.

Deliberately a flat CSV committed to the repo (cluster/excluded_dois.csv),
not a database table: it needs to survive `reset-database` (see
database/session.py's reset_database docstring) and sync to every
environment via a plain `git pull`, same as the seed CSVs it complements.
"""
from __future__ import annotations

import csv
from pathlib import Path

from fair_ocean_agent.config import REPO_ROOT
from fair_ocean_agent.database.enums import IdentifierType
from fair_ocean_agent.identity.identifiers import IdentifierError, normalize_identifier

EXCLUDED_DOIS_PATH = REPO_ROOT / "cluster" / "excluded_dois.csv"
_FIELDNAMES = ["doi", "reason", "date_added"]


def _normalize_doi(raw: str) -> str:
    try:
        return normalize_identifier(IdentifierType.DOI, raw)
    except IdentifierError:
        return raw.strip().lower()


def load_excluded_dois(path: Path = EXCLUDED_DOIS_PATH) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return {_normalize_doi(row["doi"]) for row in reader if row.get("doi")}


def append_excluded_doi(doi: str, reason: str, path: Path = EXCLUDED_DOIS_PATH) -> bool:
    """Appends one DOI with today's date if it isn't already present.
    Returns True if a row was actually added."""
    from datetime import date

    normalized = _normalize_doi(doi)
    if normalized in load_excluded_dois(path):
        return False

    is_new_file = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        if is_new_file:
            writer.writeheader()
        writer.writerow({"doi": normalized, "reason": reason, "date_added": date.today().isoformat()})
    return True
