#!/usr/bin/env python3
"""Fixes up a directory of locally-supplied PDFs so
workflow/handlers.py's own lookup (_local_pdf_path_for_study /
_doi_pdf_filename: `doi.lower().replace("/", "_") + ".pdf"`) can actually
find them. Handles three real, confirmed-live problems in one pass:

- Colon-separated names (e.g. "10.1002:aff2.87.pdf") -- some download tool
  used ":" where the pipeline expects "_". The DOI is already right there
  in the name; this just fixes the separator.
- Missing/corrupted extension (e.g. "10.1002:lno.10160", no ".pdf" at
  all) -- the DOI's own numeric suffix segment got mistaken for a file
  extension by whatever tool produced the name. Same fix, plus appending
  ".pdf".
- Non-DOI names (e.g. Paperpile's own "Author et al. YYYY - Title.pdf"
  export convention, individually or bundled in a "paperpile-files.zip")
  -- these need their real DOI recovered somehow, then renamed to match.
  Two sources, tried in this order:
  1. A Paperpile "Export as CSV" reference list (--csv), if you have one --
     its own "Attachments" column names the exact same file its "DOI"
     column belongs to, Paperpile's own authoritative metadata, so this
     is tried first, requires no PDF parsing at all, and needs no
     database-tracking check (unlike the fallback below, there's no
     ambiguity to guard against -- the CSV row already names this exact
     file). Filenames are normalized (accents/diacritics stripped, case-
     insensitive) before matching, since zip member names and a CSV
     export don't always survive round-tripping through the same Unicode
     form.
  2. Falling back to the PDF's own first-page text (most journal PDFs
     print their DOI in a running header/footer or title block),
     verified against a study this pipeline already tracks (so a cited
     reference's DOI on the same page is never mistaken for the paper's
     own) -- used only for files the CSV doesn't cover.

Dry-run by default -- prints exactly what would happen to every file
without touching anything. Pass --apply to actually rename/move files.
Never overwrites an existing correctly-named file; such a case is
reported and skipped so you can look at both copies yourself.

Usage:
    python scripts/rename_local_pdfs_by_doi.py
    python scripts/rename_local_pdfs_by_doi.py --dir data/auto_fetched_pdfs --apply
    python scripts/rename_local_pdfs_by_doi.py --csv "data/auto_fetched_pdfs/Paperpile - References.csv" --apply
    FAIR_OCEAN_DATABASE_URL=sqlite:////path/to/other.db python scripts/rename_local_pdfs_by_doi.py
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path

from fair_ocean_agent.config import REPO_ROOT
from fair_ocean_agent.database.enums import IdentifierType
from fair_ocean_agent.database.session import session_scope
from fair_ocean_agent.extraction.pdf import extract_pdf_pages
from fair_ocean_agent.identity.deduplication import find_existing_study_by_identifier
from fair_ocean_agent.identity.identifiers import IdentifierError, normalize_doi
from fair_ocean_agent.workflow.handlers import _doi_pdf_filename

DEFAULT_DIR = REPO_ROOT / "data" / "auto_fetched_pdfs"

# Matches a DOI-shaped substring anywhere in free text -- deliberately
# looser than identity/identifiers.py's own DOI_PATTERN (which requires
# the WHOLE string to already be just the DOI); candidates found here are
# always passed through normalize_doi afterward for real validation.
_DOI_IN_TEXT_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>()\[\]]+")
# A filename already in the pipeline's own exact expected shape
# (_doi_pdf_filename's output) -- nothing to do for these.
_ALREADY_CORRECT_RE = re.compile(r"^10\.\d{4,9}_[^:/]+\.pdf$", re.IGNORECASE)
_PAGES_TO_SCAN = 2
_NON_ALNUM_SPACE_RE = re.compile(r"[^a-z0-9 ]+")


def _normalize_filename_for_matching(name: str) -> str:
    """Strips extension, decomposes accented characters and drops the
    combining marks (e.g. "Galià" -> "Galia"), lowercases, and collapses
    everything else down to bare alphanumerics-and-spaces. Real gap found
    live: a zip's own member names and a Paperpile CSV export don't
    always survive round-tripping through the same Unicode normalization
    form (e.g. "Galià-Camps" vs a decomposed "Galià-Camps"), so an exact
    string match would silently miss real, correct matches."""
    stem = Path(name).stem
    decomposed = unicodedata.normalize("NFKD", stem)
    without_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _NON_ALNUM_SPACE_RE.sub(" ", without_marks.lower()).strip()


def load_csv_doi_lookup(csv_path: Path) -> dict[str, str]:
    """Paperpile's own "Export as CSV" reference list: the "Attachments"
    column names the exact file(s) that row's "DOI" belongs to (Paperpile's
    own authoritative metadata, not a guess) -- semicolon-joined when a
    reference has more than one attachment. Keyed by normalized basename
    (see _normalize_filename_for_matching) so a Unicode-form or case
    mismatch between the CSV and an actual zip member doesn't silently
    miss a real match. A row with no DOI or no attachment is skipped."""
    lookup: dict[str, str] = {}
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            doi = (row.get("DOI") or "").strip()
            attachments = (row.get("Attachments") or "").strip()
            if not doi or not attachments:
                continue
            try:
                normalized_doi = normalize_doi(doi)
            except IdentifierError:
                continue
            for attachment in attachments.split(";"):
                basename = Path(attachment.strip()).name
                if not basename:
                    continue
                key = _normalize_filename_for_matching(basename)
                if key:
                    lookup.setdefault(key, normalized_doi)
    return lookup


@dataclass
class Plan:
    source: Path
    # None means "couldn't figure out a target" -- source.name explains why
    # in the report; action stays "needs_manual_naming"/"doi_not_tracked".
    target: Path | None
    action: str
    detail: str = ""
    # (zip_filename, member_path_within_zip), set only for a plan whose
    # source lives inside a .zip. Real bug found live: encoding this into
    # `source` as a single "zip.zip:member/path.pdf" Path string and
    # re-splitting `source.name` in apply_plan silently corrupted it --
    # Path.name only ever returns the LAST path component, so a member
    # path containing its own "/" (e.g. "Paperpile files/Foo.pdf") threw
    # away the "zip.zip:" prefix entirely before the split ever ran.
    # Carrying the pair explicitly sidesteps that class of bug outright.
    zip_member: tuple[str, str] | None = None


def _colon_doi_candidate(name: str) -> str | None:
    """Fast path: a colon-separated name already IS the DOI, just with the
    wrong separator (and sometimes a missing/corrupted extension)."""
    stem = name
    for suffix in (".pdf", ".PDF"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    if ":" not in stem:
        return None
    return stem.replace(":", "/")


def _doi_candidates_from_pdf_bytes(content: bytes) -> list[str]:
    try:
        pages = extract_pdf_pages(content)
    except Exception:  # noqa: BLE001 -- a malformed/encrypted PDF must not abort the whole batch
        return []
    text = "\n".join(page.text for page in pages[:_PAGES_TO_SCAN])
    seen: list[str] = []
    for match in _DOI_IN_TEXT_RE.finditer(text):
        # Real gap found live: a hyperlink icon glyph immediately after a
        # DOI (e.g. a trailing "⟩" from a PDF's own clickable-link
        # rendering) survived the ASCII-only rstrip below and made an
        # otherwise-real DOI never match its own tracked study --
        # DOIs themselves always end alphanumeric, so strip anything
        # trailing that isn't, ASCII or not.
        candidate = re.sub(r"[^0-9A-Za-z]+$", "", match.group(0).rstrip(".,;:)]"))
        if candidate and candidate not in seen:
            seen.append(candidate)
    return seen


def _first_tracked_doi(session, candidates: list[str]) -> str | None:
    """Prefers a candidate this pipeline already tracks (the paper's own
    DOI, if this run found it via any other route) over just the first
    one to appear in scan order -- a cited reference's DOI can easily sit
    earlier on the page than the paper's own."""
    for raw in candidates:
        try:
            normalized = normalize_doi(raw)
        except IdentifierError:
            continue
        if find_existing_study_by_identifier(session, IdentifierType.DOI, normalized) is not None:
            return normalized
    return None


def _plan_via_csv(
    original_filename: str, source_label: Path, csv_map: dict[str, str], target_dir: Path, action: str
) -> Plan | None:
    """None means "not in the CSV, fall back to a content-based scan" --
    every other outcome is a real Plan; no database-tracking check needed
    here, unlike _plan_for_pdf_bytes's own fallback -- the CSV row already
    names this exact file, no ambiguity to guard against."""
    doi = csv_map.get(_normalize_filename_for_matching(original_filename))
    if doi is None:
        return None
    target = target_dir / _doi_pdf_filename(doi)
    if target.exists() and target.resolve() != source_label.resolve():
        return Plan(source_label, target, "target_exists", f"{target.name} already present")
    return Plan(source_label, target, action)


def _plan_for_pdf_bytes(
    session, source_label: Path, content: bytes, target_dir: Path
) -> Plan:
    candidates = _doi_candidates_from_pdf_bytes(content)
    if not candidates:
        return Plan(source_label, None, "no_doi_found", "no DOI-shaped text on the first pages")
    doi = _first_tracked_doi(session, candidates)
    if doi is None:
        return Plan(
            source_label, None, "doi_not_tracked",
            f"found {candidates[0]!r} (and {len(candidates) - 1} more) but none match a tracked study",
        )
    target = target_dir / _doi_pdf_filename(doi)
    if target.exists() and target.resolve() != source_label.resolve():
        return Plan(source_label, target, "target_exists", f"{target.name} already present")
    return Plan(source_label, target, "rename")


def build_plan(session, directory: Path, csv_map: dict[str, str] | None = None) -> list[Plan]:
    csv_map = csv_map or {}
    plans: list[Plan] = []
    for entry in sorted(directory.iterdir()):
        if entry.is_dir():
            continue
        if entry.suffix.lower() == ".zip":
            with zipfile.ZipFile(entry) as zf:
                for info in zf.infolist():
                    if info.is_dir() or not info.filename.lower().endswith(".pdf"):
                        continue
                    member = (entry.name, info.filename)
                    member_basename = Path(info.filename).name
                    # Label used only for the human-readable report --
                    # apply_plan never re-derives the zip/member pair from
                    # this, it uses `member` (Plan.zip_member) directly.
                    label = Path(f"{entry.name}:{member_basename}")
                    colon_doi = _colon_doi_candidate(member_basename)
                    if colon_doi:
                        try:
                            normalized = normalize_doi(colon_doi)
                        except IdentifierError:
                            pass
                        else:
                            target = directory / _doi_pdf_filename(normalized)
                            plans.append(
                                Plan(label, target, "extract_and_rename", zip_member=member)
                                if not target.exists()
                                else Plan(label, target, "target_exists", f"{target.name} already present", zip_member=member)
                            )
                            continue
                    # CSV lookup is tried next, before ever reading the
                    # member's bytes at all -- real, valuable speedup for a
                    # batch this size (487 PDFs across two zips): no PDF
                    # parsing needed for anything the CSV already covers.
                    csv_plan = _plan_via_csv(member_basename, label, csv_map, directory, "extract_and_rename")
                    if csv_plan is not None:
                        csv_plan.zip_member = member
                        plans.append(csv_plan)
                        continue
                    plan = _plan_for_pdf_bytes(session, label, zf.read(info), directory)
                    plan.action = "extract_and_rename" if plan.action == "rename" else plan.action
                    plan.zip_member = member
                    plans.append(plan)
            continue
        if entry.suffix.lower() != ".pdf":
            plans.append(Plan(entry, None, "skipped_non_pdf", "not a .pdf or .zip"))
            continue
        if _ALREADY_CORRECT_RE.match(entry.name):
            plans.append(Plan(entry, entry, "already_correct"))
            continue
        colon_doi = _colon_doi_candidate(entry.name)
        if colon_doi:
            try:
                normalized = normalize_doi(colon_doi)
            except IdentifierError:
                plans.append(Plan(entry, None, "unparseable_colon_name", colon_doi))
                continue
            target = directory / _doi_pdf_filename(normalized)
            if target.exists() and target.resolve() != entry.resolve():
                plans.append(Plan(entry, target, "target_exists", f"{target.name} already present"))
            else:
                plans.append(Plan(entry, target, "rename"))
            continue
        csv_plan = _plan_via_csv(entry.name, entry, csv_map, directory, "rename")
        if csv_plan is not None:
            plans.append(csv_plan)
            continue
        plans.append(_plan_for_pdf_bytes(session, entry, entry.read_bytes(), directory))
    return plans


def apply_plan(plans: list[Plan], directory: Path) -> None:
    zip_bytes_cache: dict[str, zipfile.ZipFile] = {}
    try:
        for plan in plans:
            if plan.action not in ("rename", "extract_and_rename"):
                continue
            assert plan.target is not None
            if plan.action == "rename":
                plan.source.rename(plan.target)
            else:
                assert plan.zip_member is not None
                zip_name, member_name = plan.zip_member
                zf = zip_bytes_cache.setdefault(zip_name, zipfile.ZipFile(directory / zip_name))
                plan.target.write_bytes(zf.read(member_name))
    finally:
        for zf in zip_bytes_cache.values():
            zf.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument(
        "--csv", type=Path, default=None,
        help="a Paperpile 'Export as CSV' reference list -- matched by filename (via its own "
        "Attachments column) before falling back to scanning each PDF's own first-page text",
    )
    parser.add_argument("--apply", action="store_true", help="actually rename/extract files (default: dry-run report only)")
    args = parser.parse_args()

    if not args.dir.is_dir():
        raise SystemExit(f"not a directory: {args.dir}")

    csv_map: dict[str, str] = {}
    if args.csv:
        if not args.csv.is_file():
            raise SystemExit(f"not a file: {args.csv}")
        csv_map = load_csv_doi_lookup(args.csv)
        print(f"Loaded {len(csv_map)} filename -> DOI entries from {args.csv}\n")

    with session_scope() as session:
        plans = build_plan(session, args.dir, csv_map)

    by_action: dict[str, list[Plan]] = {}
    for plan in plans:
        by_action.setdefault(plan.action, []).append(plan)

    print(f"Scanned {args.dir} ({len(plans)} entries)\n")
    for action, label in (
        ("already_correct", "Already correctly named"),
        ("rename", "Colon/extension fix -> rename"),
        ("extract_and_rename", "Inside a .zip -> extract + rename"),
        ("target_exists", "SKIPPED: target already exists (review both copies yourself)"),
        ("doi_not_tracked", "SKIPPED: found a DOI, but not one this pipeline tracks"),
        ("no_doi_found", "SKIPPED: no DOI found in the first pages (needs manual naming)"),
        ("unparseable_colon_name", "SKIPPED: colon-separated name doesn't look like a real DOI"),
        ("skipped_non_pdf", "SKIPPED: not a .pdf or .zip"),
    ):
        items = by_action.get(action, [])
        if not items:
            continue
        print(f"{label}: {len(items)}")
        for plan in items[:10]:
            arrow = f" -> {plan.target.name}" if plan.target else ""
            detail = f" ({plan.detail})" if plan.detail else ""
            print(f"  {plan.source.name}{arrow}{detail}")
        if len(items) > 10:
            print(f"  ... and {len(items) - 10} more")
        print()

    if args.apply:
        apply_plan(plans, args.dir)
        renamed = len(by_action.get("rename", [])) + len(by_action.get("extract_and_rename", []))
        print(f"Applied: renamed/extracted {renamed} file(s).")
    else:
        actionable = len(by_action.get("rename", [])) + len(by_action.get("extract_and_rename", []))
        print(f"Dry run only -- {actionable} file(s) would be renamed/extracted. Re-run with --apply to do it.")


if __name__ == "__main__":
    main()
