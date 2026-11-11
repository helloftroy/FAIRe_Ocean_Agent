#!/usr/bin/env python3
"""Gap 1: fetch NCBI BioSample attributes for sample entities that never got them.

Why: sources/ncbi.py keeps only the first MAX_SAMPLES_PER_PROJECT (300)
BioSamples per BioProject. Samples past the cap still became entities (from
ENA / SRA run lists) but carry no BioSample attributes, so lat/lon, date,
depth, geo_loc_name, env_* ... are empty for them in the export.

What it does: every SAMPLE entity whose accession looks like a BioSample
(SAMN / SAMEA / SAMD) and that has no `adapter:ncbi_biosample` raw fact yet
is fetched from NCBI efetch (100 per request, the adapter's own parser),
and its facts are written exactly like discovery writes them: one new
`ncbi_biosample` Source per chunk (parent_source_id = the study's existing
BioSample source, when there is one), facts through the same
extract_structured_facts + entity materialization as the discovery path.

What it does NOT do:
  - create new samples (only accessions already in the database), new
    studies, or follow related identifiers;
  - re-map to FAIRe. Run the mapping backfill afterwards (gap 2).

Idempotent and resumable: an entity that already has BioSample facts is
skipped, so re-running after a timeout continues where it stopped.
Commits once per chunk, so it can run next to the discovery job (same
SQLite busy_timeout as the workers).

Usage (repo root, cluster env active):
  python scripts/fetch_missing_biosamples.py --dry-run
  python scripts/fetch_missing_biosamples.py [--chunk 500] [--max-entities N] [--study STUDY-...]
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqlalchemy import text  # noqa: E402

from fair_ocean_agent.clock import utcnow  # noqa: E402
from fair_ocean_agent.database.enums import (  # noqa: E402
    AccessStatus,
    InspectionLevel,
    InspectionStatus,
    ReviewStatus,
    SourceType,
)
from fair_ocean_agent.database.models import RawFact, Source  # noqa: E402
from fair_ocean_agent.database.session import session_scope  # noqa: E402
from fair_ocean_agent.identity.source_linking import create_source  # noqa: E402
from fair_ocean_agent.sources.base import SourceRecord  # noqa: E402
from fair_ocean_agent.workflow.handlers import (  # noqa: E402
    _build_enabled_adapters,
    _materialize_candidate_entity,
)

BIOSAMPLE_RE = re.compile(r"^SAM(N|EA|D)\d+$")
ADAPTER = "ncbi_biosample"

FIND_SQL = """
select e.entity_id, e.external_identifier, e.study_id
from entities e
where e.entity_level = 'sample'
  and e.external_identifier like 'SAM%'
  and not exists (
      select 1 from raw_facts f
      where f.entity_id = e.entity_id and f.extraction_method = 'adapter:ncbi_biosample'
  )
"""


def find_targets(session, study: str | None) -> dict[str, list[str]]:
    """home study_id -> BioSample accessions still missing attributes."""
    sql = FIND_SQL + (" and e.study_id = :study" if study else "")
    by_study: dict[str, list[str]] = defaultdict(list)
    for _entity_id, accession, study_id in session.execute(text(sql), {"study": study} if study else {}):
        if BIOSAMPLE_RE.match(accession):
            by_study[study_id].append(accession)
    return by_study


def study_bioproject_and_parent(session, study_id: str) -> tuple[str, str | None]:
    row = session.execute(
        text(
            "select external_identifier, source_id from sources "
            "where study_id = :s and source_name = :a order by created_at limit 1"
        ),
        {"s": study_id, "a": ADAPTER},
    ).first()
    if row:
        return row[0], row[1]
    row = session.execute(
        text(
            "select identifier_value from external_identifiers "
            "where study_id = :s and identifier_type = 'bioproject_accession' order by is_primary desc limit 1"
        ),
        {"s": study_id},
    ).first()
    return (row[0] if row else "missing_biosample_fill"), None


ESEARCH_BATCH = 100


def resolve_uids(adapter, accessions: list[str]) -> list[str]:
    """BioSample accession -> NCBI UID via esearch "<acc>[accn]".

    Never hand efetch a SAMEA/SAMD accession directly: checked live
    (2026-11-10), efetch strips the prefix and reads the digits as a UID,
    so SAMEA2620084 returns the unrelated SAMN02620084. The returned
    records are also checked against the requested accessions in
    fetch_verified()."""
    uids: list[str] = []
    for i in range(0, len(accessions), ESEARCH_BATCH):
        batch = accessions[i : i + ESEARCH_BATCH]
        payload, _ = adapter.http.get_json(
            f"{adapter.config.base_url}/esearch.fcgi",
            params={
                "db": "biosample",
                "term": " OR ".join(f"{a}[accn]" for a in batch),
                "retmode": "json",
                "retmax": len(batch) * 2,
            },
        )
        uids.extend(payload.get("esearchresult", {}).get("idlist", []))
    return uids


def fetch_verified(adapter, accessions: list[str]) -> tuple[list[dict], list[str], int]:
    """(samples whose accession was requested, xml parts, records dropped)."""
    wanted = set(accessions)
    uids = resolve_uids(adapter, accessions)
    if not uids:
        return [], [], 0
    samples, xml_parts = adapter._fetch_and_parse_biosamples(uids)
    kept = [s for s in samples if s.get("accession") in wanted]
    return kept, xml_parts, len(samples) - len(kept)


def persist_chunk(session, adapter, study_id: str, bioproject: str, parent_id: str | None, accessions: list[str]) -> tuple[int, int]:
    samples, xml_parts, dropped = fetch_verified(adapter, accessions)
    if dropped:
        print(f"  dropped {dropped} returned record(s) not in the requested accessions", flush=True)
    if not samples:
        return 0, 0
    record = SourceRecord(
        source_name=ADAPTER,
        external_identifier=bioproject,
        url=None,
        raw={
            "bioproject_accession": bioproject,
            "total_linked_samples": len(samples),
            "truncated": False,
            "samples": samples,
            "fetched_via": "fetch_missing_biosamples",
        },
        retrieved_at=utcnow(),
        content_hash=__import__("hashlib").sha256("".join(xml_parts).encode()).hexdigest(),
    )
    source = create_source(
        session,
        Source(
            study_id=study_id,
            source_type=SourceType.REPOSITORY_API.value,
            source_name=ADAPTER,
            external_identifier=bioproject,
            url=None,
            access_status=AccessStatus.UNKNOWN.value,
            retrieved_at=record.retrieved_at,
            content_hash=record.content_hash,
            inspection_status=InspectionStatus.INSPECTED.value,
            inspection_level=InspectionLevel.METADATA_ONLY.value,
            parent_source_id=parent_id,
        ),
    )
    n = 0
    for fact in adapter.extract_structured_facts(record):
        entity = _materialize_candidate_entity(session, study_id, fact)
        session.add(
            RawFact(
                study_id=study_id,
                entity_id=entity.entity_id if entity is not None else None,
                source_id=source.source_id,
                source_locator=fact.source_locator,
                raw_field_name=fact.raw_field_name,
                raw_value=fact.raw_value,
                fact_type_candidate=fact.fact_type_candidate,
                entity_level=fact.entity_level.value,
                support_type=fact.support_type.value,
                extraction_method=f"adapter:{ADAPTER}",
                review_status=fact.review_status or ReviewStatus.ACCEPTED.value,
                confidence_metadata=fact.confidence_metadata,
            )
        )
        n += 1
    return len(samples), n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunk", type=int, default=500, help="accessions per Source / commit (efetch still batches 100)")
    ap.add_argument("--max-entities", type=int, default=0, help="stop after this many accessions (0 = all)")
    ap.add_argument("--study", help="only this home study (for a test run)")
    ap.add_argument("--dry-run", action="store_true", help="count targets per study and exit")
    ap.add_argument("--keep-cache", action="store_true", help="also write responses to data/cache (default off: GBs)")
    args = ap.parse_args()

    with session_scope() as session:
        targets = find_targets(session, args.study)
    total = sum(len(v) for v in targets.values())
    print(f"{total:,} BioSample accessions without attributes, in {len(targets):,} home studies", flush=True)
    for study_id, accs in sorted(targets.items(), key=lambda kv: -len(kv[1]))[:15]:
        print(f"  {study_id}  {len(accs):,}", flush=True)
    if args.dry_run:
        return

    adapter = _build_enabled_adapters().get(ADAPTER)
    if adapter is None:
        sys.exit("ncbi_biosample adapter is disabled in config")
    if not args.keep_cache:
        adapter.http._cache_enabled = False

    done = facts = fetched = 0
    t0 = time.time()
    for study_id, accs in sorted(targets.items(), key=lambda kv: -len(kv[1])):
        for i in range(0, len(accs), args.chunk):
            chunk = accs[i : i + args.chunk]
            try:
                with session_scope() as session:
                    bioproject, parent = study_bioproject_and_parent(session, study_id)
                    got, n = persist_chunk(session, adapter, study_id, bioproject, parent, chunk)
            except Exception as exc:  # one bad chunk must not stop the run; re-run retries it
                print(f"  ERROR {study_id} chunk {i}: {type(exc).__name__}: {exc}", flush=True)
                continue
            done += len(chunk)
            fetched += got
            facts += n
            rate = done / max(time.time() - t0, 1)
            print(
                f"{done:,}/{total:,} asked, {fetched:,} returned, {facts:,} facts "
                f"[{study_id} {bioproject}] {rate:.0f}/s eta {(total - done) / max(rate, 0.01) / 3600:.1f} h",
                flush=True,
            )
            if args.max_entities and done >= args.max_entities:
                print("reached --max-entities", flush=True)
                return
    print(f"finished: {fetched:,} BioSamples, {facts:,} facts", flush=True)


if __name__ == "__main__":
    main()
