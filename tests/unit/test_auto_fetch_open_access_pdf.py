"""Tests for _auto_fetch_open_access_pdf -- per an explicit user request:
many papers with no PMCID at all are still genuinely open-access, and
OpenAlex's own best_oa_location.pdf_url (already fetched during ordinary
discovery) often points straight at a real, freely-downloadable copy."""
import time
from datetime import datetime, timezone

import httpx
import pytest

from fair_ocean_agent.database.enums import IdentifierType
from fair_ocean_agent.database.models import ExternalIdentifier, Study
from fair_ocean_agent.identity.identifiers import normalize_doi
from fair_ocean_agent.sources.base import SourceRecord, SourceRecordNotFoundError
from fair_ocean_agent.sources.openalex import OpenAlexAdapter
from fair_ocean_agent.sources.unpaywall import UnpaywallAdapter
from fair_ocean_agent.workflow import handlers

REAL_PDF_BYTES = b"%PDF-1.4\n%%EOF"


def _study_with_doi(session, doi, pmcid=None) -> Study:
    study = Study(title="A study")
    session.add(study)
    session.flush()
    session.add(ExternalIdentifier(study_id=study.study_id, identifier_type=IdentifierType.DOI.value, identifier_value=normalize_doi(doi)))
    if pmcid:
        session.add(ExternalIdentifier(study_id=study.study_id, identifier_type=IdentifierType.PMCID.value, identifier_value=pmcid))
    session.flush()
    return study


class FakeOpenAlexAdapter(OpenAlexAdapter):
    """Real class name matters -- _auto_fetch_open_access_pdf isinstance-checks
    against the real OpenAlexAdapter, so this subclasses it rather than
    duck-typing, same convention as FakeEuropePmcFullTextAdapter elsewhere."""

    def __init__(self, best_oa_location=None, not_found=False):
        self._best_oa_location = best_oa_location or {}
        self._not_found = not_found

    def fetch_record(self, identifier):
        if self._not_found:
            raise SourceRecordNotFoundError(f"no openalex record for {identifier}")
        raw = {"best_oa_location": self._best_oa_location}
        return SourceRecord(
            source_name="openalex", external_identifier=identifier, url=None, raw=raw,
            retrieved_at=datetime.now(timezone.utc), content_hash="deadbeef",
        )


class FakeUnpaywallAdapter(UnpaywallAdapter):
    """Real class name matters -- _open_access_pdf_candidate_urls
    isinstance-checks against the real UnpaywallAdapter."""

    def __init__(self, is_oa=False, url_for_pdf=None, url=None, not_found=False):
        self._is_oa = is_oa
        self._url_for_pdf = url_for_pdf
        self._url = url
        self._not_found = not_found

    def fetch_record(self, identifier):
        if self._not_found:
            raise SourceRecordNotFoundError(f"no unpaywall record for {identifier}")
        raw = {"is_oa": self._is_oa, "best_oa_location": {"url_for_pdf": self._url_for_pdf, "url": self._url}}
        return SourceRecord(
            source_name="unpaywall", external_identifier=identifier, url=None, raw=raw,
            retrieved_at=datetime.now(timezone.utc), content_hash="deadbeef",
        )


class FakeFetchClient:
    def __init__(self, content=None, exc=None):
        self._content = content
        self._exc = exc
        self.closed = False

    def get_binary(self, url):
        if self._exc:
            raise self._exc
        return self._content, False

    def close(self):
        self.closed = True


def _patch_fetch_client(monkeypatch, *, content=None, exc=None):
    monkeypatch.setattr(handlers, "RateLimitedClient", lambda *a, **k: FakeFetchClient(content=content, exc=exc))


def test_skips_when_study_already_has_pmcid(db_session, monkeypatch):
    study = _study_with_doi(db_session, "10.1234/x", pmcid="PMC1")
    adapters = {"openalex": FakeOpenAlexAdapter(best_oa_location={"is_oa": True, "pdf_url": "http://x/y.pdf"})}
    handlers._auto_fetch_open_access_pdf(db_session, study, adapters)
    assert not handlers._pdf_lookup_dir().joinpath("10.1234_x.pdf").exists()


def test_skips_when_openalex_reports_not_open_access(db_session, monkeypatch, tmp_path):
    study = _study_with_doi(db_session, "10.1234/closed")
    monkeypatch.setenv(handlers.LOCAL_PDF_DIR_ENV, str(tmp_path))
    adapters = {"openalex": FakeOpenAlexAdapter(best_oa_location={"is_oa": False, "pdf_url": "http://x/y.pdf"})}
    handlers._auto_fetch_open_access_pdf(db_session, study, adapters)
    assert list(tmp_path.iterdir()) == []


def test_skips_when_no_pdf_url(db_session, monkeypatch, tmp_path):
    study = _study_with_doi(db_session, "10.1234/no-url")
    monkeypatch.setenv(handlers.LOCAL_PDF_DIR_ENV, str(tmp_path))
    adapters = {"openalex": FakeOpenAlexAdapter(best_oa_location={"is_oa": True, "pdf_url": None})}
    handlers._auto_fetch_open_access_pdf(db_session, study, adapters)
    assert list(tmp_path.iterdir()) == []


def test_downloads_and_saves_a_real_pdf(db_session, monkeypatch, tmp_path):
    study = _study_with_doi(db_session, "10.3389/fmars.2019.00373")
    monkeypatch.setenv(handlers.LOCAL_PDF_DIR_ENV, str(tmp_path))
    adapters = {"openalex": FakeOpenAlexAdapter(best_oa_location={"is_oa": True, "pdf_url": "http://frontiers/x.pdf"})}
    _patch_fetch_client(monkeypatch, content=REAL_PDF_BYTES)

    handlers._auto_fetch_open_access_pdf(db_session, study, adapters)

    saved = tmp_path / "10.3389_fmars.2019.00373.pdf"
    assert saved.exists()
    assert saved.read_bytes() == REAL_PDF_BYTES


def test_skips_when_fetch_is_blocked_confirmed_live_wiley_403(db_session, monkeypatch, tmp_path):
    """Real case, confirmed live: Wiley 403s a plain request even to a
    paper OpenAlex itself marks fully open-access -- a publisher's own
    bot-detection block, not routed around, just skipped."""
    study = _study_with_doi(db_session, "10.1111/jeu.12975")
    monkeypatch.setenv(handlers.LOCAL_PDF_DIR_ENV, str(tmp_path))
    adapters = {"openalex": FakeOpenAlexAdapter(best_oa_location={"is_oa": True, "pdf_url": "http://wiley/blocked.pdf"})}
    blocked = httpx.HTTPStatusError("403", request=httpx.Request("GET", "http://wiley/blocked.pdf"), response=httpx.Response(403))
    _patch_fetch_client(monkeypatch, exc=blocked)

    handlers._auto_fetch_open_access_pdf(db_session, study, adapters)

    assert list(tmp_path.iterdir()) == []


def test_skips_when_response_is_not_real_pdf_content(db_session, monkeypatch, tmp_path):
    """A 200 access-denied HTML page (rather than a clean 4xx) must not be
    saved as if it were the real PDF."""
    study = _study_with_doi(db_session, "10.1234/html-not-pdf")
    monkeypatch.setenv(handlers.LOCAL_PDF_DIR_ENV, str(tmp_path))
    adapters = {"openalex": FakeOpenAlexAdapter(best_oa_location={"is_oa": True, "pdf_url": "http://x/access-denied"})}
    _patch_fetch_client(monkeypatch, content=b"<html><body>Access Denied</body></html>")

    handlers._auto_fetch_open_access_pdf(db_session, study, adapters)

    assert list(tmp_path.iterdir()) == []


def test_skips_when_a_local_pdf_already_exists(db_session, monkeypatch, tmp_path):
    study = _study_with_doi(db_session, "10.1234/already-have-one")
    monkeypatch.setenv(handlers.LOCAL_PDF_DIR_ENV, str(tmp_path))
    (tmp_path / "10.1234_already-have-one.pdf").write_bytes(REAL_PDF_BYTES)

    class _FailIfCalledOpenAlex(FakeOpenAlexAdapter):
        def fetch_record(self, identifier):
            raise AssertionError("should never fetch OpenAlex when a local PDF already exists")

    handlers._auto_fetch_open_access_pdf(db_session, study, {"openalex": _FailIfCalledOpenAlex()})
    # Doesn't raise -> fetch_record was correctly never called.


def test_skips_without_a_doi(db_session, monkeypatch, tmp_path):
    study = Study(title="No DOI")
    db_session.add(study)
    db_session.flush()
    monkeypatch.setenv(handlers.LOCAL_PDF_DIR_ENV, str(tmp_path))

    class _FailIfCalledOpenAlex(FakeOpenAlexAdapter):
        def fetch_record(self, identifier):
            raise AssertionError("should never fetch OpenAlex without a DOI")

    handlers._auto_fetch_open_access_pdf(db_session, study, {"openalex": _FailIfCalledOpenAlex()})


def test_falls_back_to_unpaywall_when_openalex_has_no_oa_location(db_session, monkeypatch, tmp_path):
    """Real gap found live (10.1016/j.jhazmat.2024.133878, a real
    ScienceDirect paper): genuinely CC-BY open access per Unpaywall, but
    with no usable OpenAlex best_oa_location.pdf_url -- Unpaywall must be
    tried as an independent second source, not just as a no-op alongside
    OpenAlex."""
    study = _study_with_doi(db_session, "10.1016/j.jhazmat.2024.133878")
    monkeypatch.setenv(handlers.LOCAL_PDF_DIR_ENV, str(tmp_path))
    adapters = {
        "openalex": FakeOpenAlexAdapter(best_oa_location={"is_oa": False}),
        "unpaywall": FakeUnpaywallAdapter(is_oa=True, url_for_pdf="http://example/real.pdf"),
    }
    _patch_fetch_client(monkeypatch, content=REAL_PDF_BYTES)

    handlers._auto_fetch_open_access_pdf(db_session, study, adapters)

    saved = tmp_path / "10.1016_j.jhazmat.2024.133878.pdf"
    assert saved.exists()
    assert saved.read_bytes() == REAL_PDF_BYTES


def test_unpaywall_falls_back_to_bare_url_when_no_pdf_specific_url(db_session, monkeypatch, tmp_path):
    study = _study_with_doi(db_session, "10.1234/bare-url")
    monkeypatch.setenv(handlers.LOCAL_PDF_DIR_ENV, str(tmp_path))
    adapters = {
        "openalex": FakeOpenAlexAdapter(not_found=True),
        "unpaywall": FakeUnpaywallAdapter(is_oa=True, url_for_pdf=None, url="http://example/landing-that-is-a-pdf"),
    }
    _patch_fetch_client(monkeypatch, content=REAL_PDF_BYTES)

    handlers._auto_fetch_open_access_pdf(db_session, study, adapters)

    assert (tmp_path / "10.1234_bare-url.pdf").exists()


def test_skips_when_unpaywall_also_reports_closed_access(db_session, monkeypatch, tmp_path):
    """Real case, confirmed live (10.1126/science.1243768): Unpaywall
    correctly reports genuinely closed-access papers as closed -- nothing
    to fetch, no candidate URL at all."""
    study = _study_with_doi(db_session, "10.1126/science.1243768")
    monkeypatch.setenv(handlers.LOCAL_PDF_DIR_ENV, str(tmp_path))
    adapters = {
        "openalex": FakeOpenAlexAdapter(best_oa_location={"is_oa": False}),
        "unpaywall": FakeUnpaywallAdapter(is_oa=False),
    }

    handlers._auto_fetch_open_access_pdf(db_session, study, adapters)

    assert list(tmp_path.iterdir()) == []


def test_call_with_hard_timeout_returns_the_real_result_when_fast_enough():
    def quick(x):
        return x * 2

    assert handlers._call_with_hard_timeout(quick, 21, timeout_seconds=5) == 42


def test_call_with_hard_timeout_raises_and_returns_promptly_on_a_genuine_hang():
    """Real gap found live: a real ~9746-study batch hung indefinitely
    partway through, well past every RateLimitedClient's own configured
    60s httpx timeout -- httpx's per-phase timeouts don't bound every
    failure mode (a firewall silently dropping packets, a stuck DNS
    resolution, a deliberately slow "tarpit" response). This is the core
    mechanism that now bounds any single external call no matter what's
    actually happening underneath it -- confirmed here that the call
    returns promptly (not blocked for the full duration of the hang) and
    that the abandoned background thread doesn't keep the process alive
    (see this test file's own successful exit as proof: a non-daemon
    thread left running would hang pytest's own process teardown)."""
    def hangs_forever(x):
        time.sleep(999)
        return x

    start = time.monotonic()
    with pytest.raises(TimeoutError, match="did not respond within"):
        handlers._call_with_hard_timeout(hangs_forever, "x", timeout_seconds=0.2)
    assert time.monotonic() - start < 2.0  # returned promptly, not after the full 999s


def test_call_with_hard_timeout_reraises_the_real_exception_when_fast_enough():
    def quick_failure(x):
        raise ValueError(f"bad input: {x}")

    with pytest.raises(ValueError, match="bad input: x"):
        handlers._call_with_hard_timeout(quick_failure, "x", timeout_seconds=5)


def test_open_access_pdf_candidate_urls_skips_a_hanging_unpaywall_lookup(monkeypatch):
    """A hanging Unpaywall lookup must not prevent OpenAlex's own
    (working) candidate from still being tried."""
    def fake_hard_timeout(func, *args, timeout_seconds=90, **kwargs):
        if getattr(func, "__self__", None).__class__.__name__ == "FakeUnpaywallAdapter":
            raise TimeoutError("unpaywall.fetch_record did not respond within 90s")
        return func(*args, **kwargs)

    monkeypatch.setattr(handlers, "_call_with_hard_timeout", fake_hard_timeout)
    adapters = {
        "openalex": FakeOpenAlexAdapter(best_oa_location={"is_oa": True, "pdf_url": "http://openalex-works/x.pdf"}),
        "unpaywall": FakeUnpaywallAdapter(is_oa=True, url_for_pdf="http://never-returns/x.pdf"),
    }
    assert handlers._open_access_pdf_candidate_urls("10.1234/x", adapters) == ["http://openalex-works/x.pdf"]


def test_try_fetch_and_save_oa_pdf_treats_a_hanging_download_as_a_skip(monkeypatch):
    def fake_hard_timeout(func, *args, **kwargs):
        raise TimeoutError("get_binary did not respond within 90s")

    monkeypatch.setattr(handlers, "_call_with_hard_timeout", fake_hard_timeout)
    assert handlers._try_fetch_and_save_oa_pdf("10.1234/x", "http://never-returns/x.pdf") is False
