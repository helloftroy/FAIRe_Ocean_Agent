#!/usr/bin/env python3
"""Download Europe PMC full text for PMCID papers the GPU jobs cannot
fetch themselves, so the next extraction round can use it.

GPU nodes have no internet, so extraction there only uses full text
already in the disk cache (FAIR_OCEAN_EXTRACTION_CACHE_ONLY) and defers
any PMCID paper that is not cached. Run this first, on a node WITH
internet (login or service node), so those papers are cached. It only
reads the database; the full text goes to the same on-disk cache the
extraction handler reads (EuropePmcAdapter.fetch_fulltext_xml).

Europe PMC answers some PMCIDs, open-access ones included, with a 500
error instead of the article (found live). For those, the article is
fetched from NCBI PMC (efetch db=pmc, same JATS XML) and stored in the
cache under the Europe PMC address, so extraction reads it exactly as if
Europe PMC had sent it. NCBI_API_KEY is used if set. An article whose
publisher does not allow downloading has no body there and is counted
as having no full text.

Targets candidate studies with a PMCID, no local PDF (a PDF is used
first when there is one), and no cached full text yet. Every PMCID of a
study is fetched, since the handler uses whichever one it finds first.
Safe to stop and rerun: anything already cached is skipped.

Usage:
    FAIR_OCEAN_LOCAL_PDF_DIR=$(pwd)/data/local_pdfs python scripts/prefetch_fulltext.py
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import httpx
from sqlalchemy import select

from fair_ocean_agent.config import load_config
from fair_ocean_agent.database.enums import CanonicalStatus, IdentifierType
from fair_ocean_agent.database.models import ExternalIdentifier, Study
from fair_ocean_agent.database.session import session_scope
from fair_ocean_agent.sources.base import RateLimitedClient, SourceRecordNotFoundError
from fair_ocean_agent.sources.europe_pmc import EuropePmcAdapter
from fair_ocean_agent.workflow.handlers import _build_enabled_adapters, studies_with_local_pdf

NCBI_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_ARTICLE_RE = re.compile(r"<article[\s>].*</article>", re.DOTALL)


def _fetch_from_ncbi_pmc(ncbi: RateLimitedClient, pmcid: str) -> str | None:
    """The article's JATS XML from NCBI PMC, unwrapped from NCBI's
    <pmc-articleset> to a bare <article> like Europe PMC sends, or None
    when NCBI has no body for it (not open access)."""
    text, _ = ncbi.get_text(NCBI_EFETCH_URL, params={"db": "pmc", "id": pmcid.removeprefix("PMC")})
    match = _ARTICLE_RE.search(text)
    if match is None or "<body" not in match.group(0):
        return None
    return match.group(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="Fetch at most this many PMCIDs (for a quick test)")
    args = parser.parse_args()

    europe_pmc = _build_enabled_adapters().get("europe_pmc")
    if not isinstance(europe_pmc, EuropePmcAdapter):
        sys.exit("The europe_pmc source is not enabled in config/local.yaml -- nothing to fetch with.")

    with session_scope() as session:
        candidate_ids = set(
            session.scalars(select(Study.study_id).where(Study.canonical_status == CanonicalStatus.CANDIDATE.value))
        )
        with_pdf = studies_with_local_pdf(session)
        pairs = [
            (study_id, pmcid)
            for study_id, pmcid in session.execute(
                select(ExternalIdentifier.study_id, ExternalIdentifier.identifier_value).where(
                    ExternalIdentifier.identifier_type == IdentifierType.PMCID.value
                )
            )
            if study_id in candidate_ids and study_id not in with_pdf
        ]

    api_key = os.environ.get("NCBI_API_KEY")
    ncbi = RateLimitedClient(
        "ncbi_pmc_fulltext",
        load_config().retrieval,
        rate_limit_per_second=9 if api_key else 2.5,
        default_params={"api_key": api_key} if api_key else None,
    )

    todo = sorted({pmcid for _, pmcid in pairs if not europe_pmc.has_cached_fulltext(pmcid)})
    print(f"{len({s for s, _ in pairs})} PMCID studies have no local PDF; {len(todo)} PMCID(s) not cached yet.")
    if args.limit is not None:
        todo = todo[: args.limit]

    fetched = from_ncbi = not_found = errors = 0
    for i, pmcid in enumerate(todo, start=1):
        try:
            europe_pmc.fetch_fulltext_xml(pmcid)
            fetched += 1
        except SourceRecordNotFoundError:
            not_found += 1
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code < 500:
                errors += 1
                print(f"  {pmcid}: {exc}")
                continue
            try:
                article = _fetch_from_ncbi_pmc(ncbi, pmcid)
            except Exception as ncbi_exc:  # noqa: BLE001 - keep going; a rerun retries anything not cached
                errors += 1
                print(f"  {pmcid}: Europe PMC {exc.response.status_code}, NCBI PMC failed too: {ncbi_exc}")
                continue
            if article is None:
                not_found += 1
            else:
                europe_pmc.http.store_text(europe_pmc._fulltext_url(pmcid), article)
                fetched += 1
                from_ncbi += 1
        except Exception as exc:  # noqa: BLE001 - keep going; a rerun retries anything not cached
            errors += 1
            print(f"  {pmcid}: {exc}")
        if i % 50 == 0 or i == len(todo):
            print(f"  [{i}/{len(todo)}] cached {fetched} ({from_ncbi} via NCBI PMC), no full text {not_found}, errors {errors}")

    print(
        f"Done: cached {fetched} ({from_ncbi} via NCBI PMC); {not_found} have no open-access full text "
        f"(those need a PDF); {errors} errors (rerun to retry them)."
    )


if __name__ == "__main__":
    main()
