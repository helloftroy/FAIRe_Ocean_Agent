#!/usr/bin/env python3
"""Runs just the open-access PDF auto-fetch step (workflow/handlers.py's
_auto_fetch_open_access_pdf) across every candidate study, without running
the rest of DISCOVER_IDENTIFIERS. This is deliberately narrow: no LLM, no
GPU, one OpenAlex lookup plus (at most) one PDF download per study, rate-
limited to 1/sec -- purely network I/O, so it's cheap enough to run
locally even at a few thousand studies (a few thousand seconds worst
case), not something that needs cluster compute.

_auto_fetch_open_access_pdf already skips (cheaply, before any network
call) any study with a PMCID or an existing local/auto-fetched PDF, so
it's safe to call unconditionally for every candidate study -- there's no
need to pre-filter to "just the ambiguous ones" here.

This only ever adds PDF coverage; it never re-mines a paper's text into
facts. Follow up with the normal rediscovery step (enqueue-full-
rediscovery-backfill + worker) to actually turn a newly-available PDF
into real sample/run entities, then re-run classify_papers.py to see the
result.

Usage:
    python scripts/auto_fetch_missing_pdfs.py
    FAIR_OCEAN_LOCAL_PDF_DIR=data/PDFs python scripts/auto_fetch_missing_pdfs.py
"""
from __future__ import annotations

import logging

from sqlalchemy import select

from fair_ocean_agent.database.enums import CanonicalStatus, IdentifierType
from fair_ocean_agent.database.models import Study
from fair_ocean_agent.database.session import session_scope
from fair_ocean_agent.workflow.handlers import _auto_fetch_open_access_pdf, _build_enabled_adapters, _local_pdf_path_for_study, _identifier_value

logger = logging.getLogger(__name__)


def main() -> None:
    adapters = _build_enabled_adapters()
    fetched = 0
    already_covered = 0
    checked = 0
    errored = 0

    with session_scope() as session:
        studies = session.scalars(
            select(Study).where(Study.canonical_status == CanonicalStatus.CANDIDATE.value)
        ).all()
        for study in studies:
            has_pmcid = _identifier_value(session, study.study_id, IdentifierType.PMCID) is not None
            had_pdf_before = _local_pdf_path_for_study(session, study) is not None
            if has_pmcid or had_pdf_before:
                already_covered += 1
                continue
            checked += 1
            try:
                _auto_fetch_open_access_pdf(session, study, adapters)
            except Exception as exc:
                # A sustained 429/5xx that outlasts RateLimitedClient's own
                # retry budget (e.g. from running this alongside another
                # script also hitting OpenAlex -- confirmed live, two
                # independent 5/sec limiters can combine past what the
                # unauthenticated pool tolerates) must not take down the
                # rest of a multi-thousand-study run. Skip this one study
                # and keep going; it'll just get picked up again next run.
                errored += 1
                logger.warning("auto-fetch failed for %s (%s): %s", study.study_id, study.title, exc)
                continue
            if _local_pdf_path_for_study(session, study) is not None:
                fetched += 1
                print(f"fetched: {study.title or study.study_id}")

    print()
    print(f"Already covered (PMCID or existing PDF): {already_covered}")
    print(f"Checked (no PMCID, no PDF):               {checked}")
    print(f"Newly auto-fetched:                        {fetched}")
    print(f"Errored (network/rate-limit, safe to re-run): {errored}")
    print(f"Still need a manual PDF:                    {checked - fetched - errored}")


if __name__ == "__main__":
    main()
