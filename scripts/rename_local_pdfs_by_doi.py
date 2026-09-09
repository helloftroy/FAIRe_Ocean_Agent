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
  2. The PDF's own metadata/links -- a "/doi" document-metadata field, a
     "doi:..." mention in the "/Subject" field, or a doi.org URL inside a
     page-1 hyperlink annotation. Real gap found live: some publisher
     templates (confirmed for Elsevier) render the DOI as a header/footer
     GRAPHIC with zero extractable page text at all, yet still embed the
     same DOI in one of these three places -- the publisher's own
     embedded value, so (like the CSV) no database-tracking check needed.
  3. Falling back to the PDF's own first-page text (most other journal
     PDFs print their DOI in a running header/footer or title block),
     verified against a study this pipeline already tracks (so a cited
     reference's DOI on the same page is never mistaken for the paper's
     own) -- used only when neither of the above found anything.
  4. A title-keyed CSV (--title-csv) -- e.g. this project's own paper
     classification export, which has no per-file "Attachments" column
     the way Paperpile's export does, only a "title"/"doi" pair -- matched
     against the title parsed out of Paperpile's "Author et al. YYYY -
     Title.pdf" naming convention (its own filename truncation, marked
     with " ... ", is handled by a prefix/suffix match). Used only when
     nothing above found anything; an ambiguous truncated match (more
     than one CSV title fits the same prefix/suffix) is skipped rather
     than guessed.

Dry-run by default -- prints exactly what would happen to every file
without touching anything. Pass --apply to actually rename/move files.
Never overwrites an existing correctly-named file; such a case is
reported and skipped ("target already exists") so you can look at both
copies yourself -- or pass --delete-duplicates (with --apply) to delete
the old-named loose copy once its DOI-named counterpart is confirmed
present. Only ever deletes the old-named side, never a zip member (that
would mean rewriting the whole archive) and never the DOI-named file.

Usage:
    python scripts/rename_local_pdfs_by_doi.py
    python scripts/rename_local_pdfs_by_doi.py --dir data/auto_fetched_pdfs --apply
    python scripts/rename_local_pdfs_by_doi.py --csv "data/auto_fetched_pdfs/Paperpile - References.csv" --apply
    python scripts/rename_local_pdfs_by_doi.py --apply --delete-duplicates
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

from pypdf import PdfReader

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
# Paperpile's own "Author et al. YYYY - Title.pdf" naming convention --
# strips the leading author/year segment off to recover just the title.
_TITLE_FROM_FILENAME_RE = re.compile(r"^.*?\d{4}[a-z]?\s*-\s*(.*)$")
# Paperpile's own filename-truncation marker for an overlong title.
_TRUNCATION_MARKER = " ... "
# How much of the (normalized) prefix/suffix around a truncation marker
# to require for a match -- long enough that an unrelated paper sharing a
# few words at the start/end of its title can't collide, short enough to
# survive Paperpile's own truncation point moving by a few characters.
_TITLE_PREFIX_MATCH_LEN = 40
_TITLE_SUFFIX_MATCH_LEN = 30


def _title_from_filename(name: str) -> str:
    stem = Path(name).stem
    match = _TITLE_FROM_FILENAME_RE.match(stem)
    return match.group(1) if match else stem


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


def _normalize_title_for_matching(title: str) -> str:
    """Same normalization as _normalize_filename_for_matching, minus the
    filename-specific Path(...).stem step -- a title has no extension to
    strip, and stripping after its last "." would wrongly truncate a
    title that happens to end in an abbreviation."""
    decomposed = unicodedata.normalize("NFKD", title)
    without_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _NON_ALNUM_SPACE_RE.sub(" ", without_marks.lower()).strip()


def load_title_doi_lookup(csv_path: Path) -> dict[str, str]:
    """A CSV keyed by "title"/"doi" columns (this project's own paper
    classification exports have this shape, not Paperpile's per-file
    "Attachments" column) -- keyed by normalized title. A title that maps
    to two different DOIs across rows is dropped entirely rather than
    guessed at (real, if rare, possibility with a large classification
    export); a row with no title or no valid DOI is skipped."""
    lookup: dict[str, str] = {}
    ambiguous_keys: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            title = (row.get("title") or "").strip()
            doi = (row.get("doi") or "").strip()
            if not title or not doi:
                continue
            try:
                normalized_doi = normalize_doi(doi)
            except IdentifierError:
                continue
            key = _normalize_title_for_matching(title)
            if not key:
                continue
            existing = lookup.get(key)
            if existing is not None and existing != normalized_doi:
                ambiguous_keys.add(key)
                continue
            lookup[key] = normalized_doi
    for key in ambiguous_keys:
        lookup.pop(key, None)
    return lookup


def _doi_from_title_lookup(filename: str, title_map: dict[str, str]) -> str | None:
    """Matches the title parsed out of a Paperpile-style filename against
    a title-keyed CSV lookup. Paperpile itself truncates an overlong
    title in the filename (marked with _TRUNCATION_MARKER), so a
    truncated title is matched by requiring both a normalized prefix and
    suffix to line up -- and, if more than one CSV title fits that same
    prefix/suffix, treated as ambiguous and skipped rather than guessed."""
    if not title_map:
        return None
    title = _title_from_filename(filename)
    if _TRUNCATION_MARKER in title:
        prefix, _, suffix = title.partition(_TRUNCATION_MARKER)
        prefix_key = _normalize_title_for_matching(prefix)[:_TITLE_PREFIX_MATCH_LEN]
        suffix_key = _normalize_title_for_matching(suffix)[-_TITLE_SUFFIX_MATCH_LEN:]
        if not prefix_key or not suffix_key:
            return None
        candidates = {
            doi for key, doi in title_map.items()
            if key.startswith(prefix_key) and key.endswith(suffix_key)
        }
        return candidates.pop() if len(candidates) == 1 else None
    return title_map.get(_normalize_title_for_matching(title))


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


def _clean_doi_match(raw: str) -> str:
    # Real gap found live: a hyperlink icon glyph immediately after a DOI
    # (e.g. a trailing "⟩" from a PDF's own clickable-link rendering)
    # survived an ASCII-only rstrip and made an otherwise-real DOI never
    # match its own tracked study -- DOIs themselves always end
    # alphanumeric, so strip anything trailing that isn't, ASCII or not.
    return re.sub(r"[^0-9A-Za-z]+$", "", raw.rstrip(".,;:)]"))


def _doi_candidates_from_text(text: str) -> list[str]:
    seen: list[str] = []
    for match in _DOI_IN_TEXT_RE.finditer(text):
        candidate = _clean_doi_match(match.group(0))
        if candidate and candidate not in seen:
            seen.append(candidate)
    return seen


# Real gap found live: a real, downloaded Elsevier PDF ("Adebayo et al.
# 2024 - 1-s2.0-...-main.pdf") had ZERO "doi" mentions in its own
# extracted page text at all (confirmed live, 10 pages, all empty) --
# some publisher templates render the DOI as a header/footer GRAPHIC, not
# extractable text -- yet the exact same DOI was sitting in three other
# places pypdf can read directly, with no page-text extraction at all: a
# `/doi` key in the PDF's own document metadata, a doi.org URL inside a
# clickable-link annotation on page 1, and a "doi:..." mention buried in
# the `/Subject` metadata field. All three are the PUBLISHER's own
# embedded value (not scraped from a rendered page), so a match here
# needs no database-tracking cross-check the way a free-text page scan
# does -- there's no cited-reference-DOI ambiguity to guard against.
def _high_confidence_doi_from_pdf_bytes(content: bytes) -> str | None:
    try:
        reader = PdfReader(io.BytesIO(content))
    except Exception:  # noqa: BLE001 -- a malformed/encrypted PDF must not abort the whole batch
        return None

    candidates: list[str] = []
    try:
        metadata = reader.metadata
    except Exception:  # noqa: BLE001
        metadata = None
    if metadata:
        doi_field = metadata.get("/doi")
        if doi_field:
            candidates.append(_clean_doi_match(str(doi_field)))
        subject = metadata.get("/Subject")
        if subject:
            candidates.extend(_doi_candidates_from_text(str(subject)))

    try:
        pages = reader.pages
    except Exception:  # noqa: BLE001
        pages = []
    for page in pages[:_PAGES_TO_SCAN]:
        try:
            annots = page.get("/Annots")
        except Exception:  # noqa: BLE001
            annots = None
        if not annots:
            continue
        for annot in annots:
            try:
                action = annot.get_object().get("/A")
                uri = action.get("/URI") if action else None
            except Exception:  # noqa: BLE001
                uri = None
            if uri and "doi.org/" in str(uri):
                candidates.extend(_doi_candidates_from_text(str(uri)))

    for raw in candidates:
        try:
            return normalize_doi(raw)
        except IdentifierError:
            continue
    return None


def _doi_candidates_from_pdf_bytes(content: bytes) -> list[str]:
    try:
        pages = extract_pdf_pages(content)
    except Exception:  # noqa: BLE001 -- a malformed/encrypted PDF must not abort the whole batch
        return []
    text = "\n".join(page.text for page in pages[:_PAGES_TO_SCAN])
    return _doi_candidates_from_text(text)


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
    session, source_label: Path, content: bytes, target_dir: Path, original_filename: str,
    title_map: dict[str, str] | None = None,
) -> Plan:
    doi = _high_confidence_doi_from_pdf_bytes(content)
    candidates: list[str] = []
    if doi is None:
        candidates = _doi_candidates_from_pdf_bytes(content)
        if candidates:
            doi = _first_tracked_doi(session, candidates)
    if doi is None:
        doi = _doi_from_title_lookup(original_filename, title_map or {})
    if doi is None:
        if candidates:
            return Plan(
                source_label, None, "doi_not_tracked",
                f"found {candidates[0]!r} (and {len(candidates) - 1} more) but none match a tracked study",
            )
        return Plan(source_label, None, "no_doi_found", "no DOI found in metadata, links, page text, or title match")
    target = target_dir / _doi_pdf_filename(doi)
    if target.exists() and target.resolve() != source_label.resolve():
        return Plan(source_label, target, "target_exists", f"{target.name} already present")
    return Plan(source_label, target, "rename")


def build_plan(
    session, directory: Path, csv_map: dict[str, str] | None = None, title_map: dict[str, str] | None = None
) -> list[Plan]:
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
                    plan = _plan_for_pdf_bytes(session, label, zf.read(info), directory, member_basename, title_map)
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
        plans.append(_plan_for_pdf_bytes(session, entry, entry.read_bytes(), directory, entry.name, title_map))
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


def delete_duplicate_plans(plans: list[Plan]) -> tuple[int, int]:
    """Deletes the *source* side of every "target_exists" plan whose
    source is a real loose file on disk (never a zip member -- deleting
    one entry out of a zip would mean rewriting the whole archive, not
    worth it for a handful of already-redundant zips), and only after
    re-confirming the DOI-named target is still actually there. Never
    touches plan.target itself. Returns (files deleted, bytes freed)."""
    deleted = 0
    freed_bytes = 0
    for plan in plans:
        if plan.action != "target_exists" or plan.zip_member is not None:
            continue
        assert plan.target is not None
        if not plan.target.is_file() or not plan.source.is_file():
            continue
        freed_bytes += plan.source.stat().st_size
        plan.source.unlink()
        deleted += 1
    return deleted, freed_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument(
        "--csv", type=Path, default=None,
        help="a Paperpile 'Export as CSV' reference list -- matched by filename (via its own "
        "Attachments column) before falling back to scanning each PDF's own first-page text",
    )
    parser.add_argument(
        "--title-csv", type=Path, default=None,
        help="a title/doi-keyed CSV (e.g. this project's own paper classification export) -- "
        "tried last, after --csv, PDF metadata/links, and page-text scanning all come up empty, "
        "by matching the title parsed out of a Paperpile-style filename",
    )
    parser.add_argument("--apply", action="store_true", help="actually rename/extract files (default: dry-run report only)")
    parser.add_argument(
        "--delete-duplicates", action="store_true",
        help="also delete the old-named loose file for every 'target already exists' case, once its "
        "DOI-named counterpart is confirmed present -- never deletes a zip member or a DOI-named file "
        "itself; requires --apply",
    )
    args = parser.parse_args()

    if args.delete_duplicates and not args.apply:
        raise SystemExit("--delete-duplicates requires --apply")

    if not args.dir.is_dir():
        raise SystemExit(f"not a directory: {args.dir}")

    csv_map: dict[str, str] = {}
    if args.csv:
        if not args.csv.is_file():
            raise SystemExit(f"not a file: {args.csv}")
        csv_map = load_csv_doi_lookup(args.csv)
        print(f"Loaded {len(csv_map)} filename -> DOI entries from {args.csv}\n")

    title_map: dict[str, str] = {}
    if args.title_csv:
        if not args.title_csv.is_file():
            raise SystemExit(f"not a file: {args.title_csv}")
        title_map = load_title_doi_lookup(args.title_csv)
        print(f"Loaded {len(title_map)} title -> DOI entries from {args.title_csv}\n")

    with session_scope() as session:
        plans = build_plan(session, args.dir, csv_map, title_map)

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
        if args.delete_duplicates:
            deleted, freed_bytes = delete_duplicate_plans(plans)
            print(f"Deleted {deleted} old-named duplicate(s), freeing {freed_bytes / 1e9:.2f} GB.")
    else:
        actionable = len(by_action.get("rename", [])) + len(by_action.get("extract_and_rename", []))
        print(f"Dry run only -- {actionable} file(s) would be renamed/extracted. Re-run with --apply to do it.")
        if args.delete_duplicates:
            dup_candidates = [p for p in by_action.get("target_exists", []) if p.zip_member is None]
            print(f"Would also delete {len(dup_candidates)} old-named duplicate(s) (--apply not set, nothing done).")


if __name__ == "__main__":
    main()
