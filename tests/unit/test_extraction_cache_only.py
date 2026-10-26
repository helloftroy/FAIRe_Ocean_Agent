"""On an offline GPU node, EXTRACT_TEXT_FACTS must never try to fetch full
text over the network. Real gap found live: one study with no local PDF
and uncached Europe PMC text froze a whole GPU shard for two days. With
FAIR_OCEAN_EXTRACTION_CACHE_ONLY set, such a study is deferred: back to
pending, attempt not counted, and not re-claimed in the same run."""
from fair_ocean_agent.database.enums import IdentifierType, TaskStatus, TaskType
from fair_ocean_agent.database.models import ExternalIdentifier, Study, Task
from fair_ocean_agent.sources.europe_pmc import EuropePmcAdapter
from fair_ocean_agent.workflow import handlers
from fair_ocean_agent.workflow.task_queue import enqueue_task
from fair_ocean_agent.workflow.worker import run_worker


class _UncachedEuropePmc(EuropePmcAdapter):
    def __init__(self, cached: bool):
        self.cached = cached
        self.fetches = 0

    def has_cached_fulltext(self, pmcid):
        return self.cached

    def fetch_fulltext_xml(self, pmcid):
        self.fetches += 1
        raise AssertionError("must not fetch over the network")


def _pmcid_study_task(session):
    study = Study(title=None)
    session.add(study)
    session.flush()
    session.add(ExternalIdentifier(study_id=study.study_id, identifier_type=IdentifierType.PMCID.value, identifier_value="PMC999"))
    task = enqueue_task(session, TaskType.EXTRACT_TEXT_FACTS, study_id=study.study_id)
    session.commit()
    return task


def test_uncached_pdfless_paper_is_deferred_not_fetched(db_session, monkeypatch, tmp_path):
    monkeypatch.setenv("FAIR_OCEAN_EXTRACTION_CACHE_ONLY", "1")
    monkeypatch.setenv("FAIR_OCEAN_LOCAL_PDF_DIR", str(tmp_path))
    adapter = _UncachedEuropePmc(cached=False)
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {"europe_pmc": adapter})
    task = _pmcid_study_task(db_session)

    summary = run_worker(db_session, worker_id="w", until_empty=True, task_types=[TaskType.EXTRACT_TEXT_FACTS])

    db_session.refresh(task)
    assert summary["deferred"] == 1 and summary["failed"] == 0
    assert summary["processed"] == 1  # not re-claimed in a loop
    assert adapter.fetches == 0
    assert task.status == TaskStatus.PENDING.value
    assert task.attempt_count == 0  # deferral does not use up an attempt
    assert task.last_error.startswith("deferred:")


def test_without_cache_only_mode_the_handler_still_fetches(db_session, monkeypatch, tmp_path):
    """Default (online machines): unchanged behaviour -- it goes to fetch."""
    monkeypatch.delenv("FAIR_OCEAN_EXTRACTION_CACHE_ONLY", raising=False)
    monkeypatch.setenv("FAIR_OCEAN_LOCAL_PDF_DIR", str(tmp_path))
    adapter = _UncachedEuropePmc(cached=False)
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {"europe_pmc": adapter})
    task = _pmcid_study_task(db_session)

    run_worker(db_session, worker_id="w", until_empty=True, task_types=[TaskType.EXTRACT_TEXT_FACTS])

    assert adapter.fetches == 1


def test_is_cached_reflects_the_disk_cache(tmp_path, monkeypatch):
    from fair_ocean_agent.config import load_config
    from fair_ocean_agent.sources.base import RateLimitedClient

    retrieval = load_config().retrieval.model_copy(update={"cache_dir": str(tmp_path), "cache_enabled": True})
    client = RateLimitedClient("europe_pmc_test", retrieval, rate_limit_per_second=1)
    url = "https://example.org/PMC1/fullTextXML"
    assert client.is_cached(url) is False
    client._cache_path(url, None, "xml").write_text("<article/>")
    assert client.is_cached(url) is True
