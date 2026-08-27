import csv
import json
import sqlite3

import httpx
import pytest

from fair_ocean_agent.seed_discovery.cncb_gsa_discovery import (
    CncbClient,
    CncbConfig,
    CncbDB,
    CncbGsaDiscoveryRunner,
    extract_search_items,
    item_to_project_seed,
    parse_biosample_html,
    parse_gsa_html,
    project_row_from_gsa,
    project_row_from_seed,
    sample_row_to_db,
)
import fair_ocean_agent.seed_discovery.cncb_gsa_discovery as cncb_gsa_discovery
from scripts.cncb_gsa_seeds_to_csv import convert


def _fast_config(tmp_path, *, max_pages_per_query: int = 1) -> CncbConfig:
    return CncbConfig(
        db_path=tmp_path / "cncb.sqlite",
        data_dir=tmp_path / "cncb_data",
        mgnify_db_path=tmp_path / "mgnify.sqlite",
        qiita_db_path=tmp_path / "qiita.sqlite",
        gold_db_path=tmp_path / "gold.sqlite",
        min_request_interval_seconds=0.0,
        max_retries=2,
        retry_base_seconds=0.001,
        page_size=50,
        max_pages_per_query=max_pages_per_query,
    )


def test_cncb_search_parsing_keeps_only_native_bioproject_records():
    """Regression test for a real live bug (confirmed 2026-08-26 by
    directly querying the CNCB API): searching db=gsa returns an index
    where >99% of hits for any generic environmental term are INSDC-
    mirrored Run/Experiment records, not native type=="GSA" project
    records -- a live test found ZERO native hits within 3000 results for
    "amplicon" (18M total hits), "seawater", and "coral". db=bioproject is
    CNCB's own dedicated, much smaller BioProject index; a native
    submission there is identified by attrs.Center == "GSA" (a mirrored
    one is typically "SRA") and already carries its own CrasAcc/SamplesAcc
    cross-references, confirmed live against a real record
    (PRJCA070101/CRA047138)."""
    payload = {
        "code": "200",
        "result": {
            "data": {
                "recordsTotal": 2,
                "data": [
                    {
                        "id": "PRJCA070101",
                        "type": "BioProject",
                        "title": "Marine sediment amplicon reads",
                        "attrs": {"Accession": "PRJCA070101", "Center": "GSA", "CrasAcc": ["CRA047138"]},
                    },
                    {
                        "id": "PRJNA1",
                        "type": "BioProject",
                        "title": "Imported project",
                        "attrs": {"Accession": "PRJNA1", "Center": "SRA"},
                    },
                ],
            }
        },
    }

    total, items = extract_search_items(payload)
    seeds = [item_to_project_seed(item) for item in items]

    assert total == 2
    assert seeds[0]["cncb_bioproject"] == "PRJCA070101"
    assert seeds[0]["cra_accessions"] == ["CRA047138"]
    assert seeds[1] is None


def test_cncb_gsa_html_parser_extracts_hierarchy_and_files():
    raw_html = """
    <div><b>标题:</b> Raw sequencing reads of marine sediment samples (16S V4 amplicon sequencing)</div>
    <div><b>项目编号:</b><a href="/bioproject/browse/PRJCA070101"> PRJCA070101 </a></div>
    <span>HTTPS：<a href="https://download.cncb.ac.cn/gsa5/CRA047138">https://download.cncb.ac.cn/gsa5/CRA047138</a></span>
    <tr class="experiment">
      <td class="experiments"><a href="browse/CRA047138/CRX3152623">CRX3152623</a></td>
      <td>MJG_43.338F_806R_V4</td>
      <td>marine sediment metagenome</td>
      <td>Illumina MiSeq</td>
      <td><a href="/biosample/browse/SAMC8161050">SAMC8161050</a></td>
    </tr>
    <tr class="runTr">
      <td class="runs"><a href="browse/CRA047138/CRR3346370">CRR3346370</a></td>
      <td colspan="2">SAMC8161050</td>
      <td colspan="2"><strong>File: </strong>CRR3346370_r1.fastq.gz<br/><strong>File: </strong>CRR3346370_r2.fastq.gz</td>
    </tr>
    """

    parsed = parse_gsa_html("CRA047138", raw_html)
    experiment = parsed["experiments"][0]

    assert parsed["cncb_bioproject"] == "PRJCA070101"
    assert parsed["sample_accessions"] == ["SAMC8161050"]
    assert experiment["crx_accession"] == "CRX3152623"
    assert json.loads(experiment["crr_accessions_json"]) == ["CRR3346370"]
    assert experiment["layout"] == "paired end"
    assert experiment["target_gene"] == "16S rRNA"
    assert "https://download.cncb.ac.cn/gsa5/CRA047138/CRR3346370_r1.fastq.gz" in json.loads(experiment["download_urls_json"])


def test_cncb_biosample_parser_promotes_common_environment_fields():
    raw_html = """
    <table>
      <tr><th>Sample name</th><td>Station A</td></tr>
      <tr><th>collection date</th><td>2024-01-02</td></tr>
      <tr><th>latitude</th><td>10.1</td></tr>
      <tr><th>longitude</th><td>120.2</td></tr>
      <tr><th>isolation source</th><td>seawater</td></tr>
      <tr><th>salinity</th><td>34 PSU</td></tr>
      <tr><th>nitrate</th><td>1.2 umol/L</td></tr>
    </table>
    """

    parsed = parse_biosample_html("SAMC1", raw_html)
    row = sample_row_to_db("PRJCA1", "CRA1", parsed)

    assert row["sample_name"] == "Station A"
    assert row["collection_date"] == "2024-01-02"
    assert row["latitude"] == "10.1"
    assert row["longitude"] == "120.2"
    assert row["env_medium"] == "seawater"
    assert row["salinity"] == "34 PSU"
    assert row["nitrate"] == "1.2 umol/L"
    assert json.loads(row["other_environmental_measurements_json"])["nitrate"] == "1.2 umol/L"


def test_cncb_db_uses_three_physical_tables_and_views(tmp_path):
    db = CncbDB(tmp_path / "cncb.sqlite")
    try:
        db.initialize()
        tables = {
            row["name"]
            for row in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'cncb_%'"
            )
        }
        views = {
            row["name"]
            for row in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'view' AND name IN "
                "('paper_seeds', 'cncb_faire_sample_enrichment', 'cncb_faire_experiment_enrichment')"
            )
        }

        assert tables == {"cncb_projects", "cncb_samples", "cncb_experiments"}
        assert views == {"paper_seeds", "cncb_faire_sample_enrichment", "cncb_faire_experiment_enrichment"}
    finally:
        db.close()


def test_cncb_csv_export_uses_standard_seed_shape(tmp_path):
    db = CncbDB(tmp_path / "cncb.sqlite")
    try:
        db.initialize()
        db.upsert_project(
            project_row_from_gsa(
                {
                    "cra_accession": "CRA047138",
                    "cncb_bioproject": "PRJCA070101",
                    "cra_accessions": ["CRA047138"],
                    "title": "Marine sediment amplicon reads",
                    "description": None,
                    "download_roots": ["https://download.cncb.ac.cn/gsa5/CRA047138"],
                    "publication_dois": ["10.1234/example"],
                    "pmids": ["123456"],
                    "sample_accessions": ["SAMC1"],
                    "experiments": [],
                    "sequencing_strategy": ["16S rRNA"],
                    "insdc_bioprojects": [],
                    "source": {},
                }
            )
        )
        db.commit()
    finally:
        db.close()

    out = tmp_path / "seeds.csv"
    written, no_doi = convert(tmp_path / "cncb.sqlite", out)
    rows = list(csv.DictReader(out.open()))

    assert written == 1
    assert no_doi == 0
    assert rows[0]["seed_id"] == "cncb_gsa-CRA047138"
    assert rows[0]["doi"] == "10.1234/example"
    assert rows[0]["dataset_id"] == "CRA047138"
    assert rows[0]["repository"] == "cncb_gsa"


def test_cncb_client_retries_transient_timeout_then_succeeds(tmp_path):
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] < 2:
            raise httpx.ReadTimeout("simulated timeout", request=request)
        return httpx.Response(200, text="ok")

    client = CncbClient(_fast_config(tmp_path), transport=httpx.MockTransport(handler))
    response = client.get("https://ngdc.cncb.ac.cn/gsa/browse/CRA1")
    assert response.text == "ok"
    assert calls["count"] == 2


def test_cncb_client_raises_after_exhausting_retries(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated persistent timeout", request=request)

    client = CncbClient(_fast_config(tmp_path), transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.ReadTimeout):
        client.get("https://ngdc.cncb.ac.cn/gsa/browse/CRA1")


def _search_payload(items: list[dict]) -> dict:
    return {"code": "200", "result": {"data": {"recordsTotal": len(items), "recordsFiltered": len(items), "data": items}}}


def test_one_discovery_querys_persistent_failure_does_not_crash_the_whole_discovery_phase(tmp_path, monkeypatch):
    """Regression test mirroring the real live Qiita ReadTimeout crash
    (see test_qiita_discovery.py's own version of this test): before this
    fix, CncbGsaDiscoveryRunner._discover had no try/except around its
    search_gsa call, so one query term persistently failing (after
    CncbClient's own retries were exhausted) would propagate straight out
    of run() and kill the entire job, even though other query terms would
    have found real projects in the same run."""
    monkeypatch.setattr(cncb_gsa_discovery, "DISCOVERY_QUERIES", ("good1", "bad", "good2"))

    def handler(request: httpx.Request) -> httpx.Response:
        query = request.url.params.get("q")
        if query == "bad":
            raise httpx.ReadTimeout("simulated persistent timeout for query=bad", request=request)
        bioproject = "PRJCA0001" if query == "good1" else "PRJCA0002"
        cra = "CRA0001" if query == "good1" else "CRA0002"
        item = {"id": bioproject, "type": "BioProject", "title": "t", "attrs": {"Accession": bioproject, "Center": "GSA", "CrasAcc": [cra]}}
        return httpx.Response(200, json=_search_payload([item]))

    config = _fast_config(tmp_path)
    runner = CncbGsaDiscoveryRunner(config, transport=httpx.MockTransport(handler))
    try:
        result = runner.run(phase="discovery")
    finally:
        runner.close()

    assert result["counts"]["discover_query_errors"] == 1
    assert result["counts"]["native_projects_seen"] == 2

    conn = sqlite3.connect(config.db_path)
    stored = {row[0] for row in conn.execute("SELECT cncb_bioproject FROM cncb_projects")}
    conn.close()
    assert stored == {"PRJCA0001", "PRJCA0002"}


def test_stale_search_cache_from_before_the_db_bioproject_fix_is_not_reused(tmp_path, monkeypatch):
    """Regression test for a real live bug reported after the db=gsa ->
    db=bioproject fix shipped: re-running against an existing data-dir
    (the normal, intentional resume/caching behavior) kept finding only 9
    projects again, identical to before the fix. Root cause: the search
    response cache key was only query+start, with no marker for which
    index was actually searched, so a raw_dir populated by the OLD (buggy)
    db=gsa code kept getting replayed verbatim -- the live API was never
    actually re-queried post-fix. Cache path is now namespaced under
    raw_dir/_search/bioproject/, so a stale file left at the old flat
    raw_dir/_search/ path (simulated here) must never be read."""
    monkeypatch.setattr(cncb_gsa_discovery, "DISCOVERY_QUERIES", ("marine",))
    config = _fast_config(tmp_path)
    raw_dir = config.data_dir / "raw"
    stale_cache_path = raw_dir / "_search" / "marine_0.json"
    stale_cache_path.parent.mkdir(parents=True, exist_ok=True)
    stale_cache_path.write_text(json.dumps(_search_payload([])), encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        item = {"id": "PRJCA9999", "type": "BioProject", "title": "t", "attrs": {"Accession": "PRJCA9999", "Center": "GSA", "CrasAcc": ["CRA9999"]}}
        return httpx.Response(200, json=_search_payload([item]))

    runner = CncbGsaDiscoveryRunner(config, transport=httpx.MockTransport(handler))
    try:
        result = runner.run(phase="discovery")
    finally:
        runner.close()

    assert result["counts"]["native_projects_seen"] == 1
    conn = sqlite3.connect(config.db_path)
    stored = {row[0] for row in conn.execute("SELECT cncb_bioproject FROM cncb_projects")}
    conn.close()
    assert stored == {"PRJCA9999"}
    assert (raw_dir / "_search" / "bioproject" / "marine_0.json").exists()


def test_one_projects_gsa_page_persistent_failure_does_not_crash_the_whole_metadata_phase(tmp_path):
    """Same failure shape as the discovery-phase regression test above, but
    for _metadata's per-project GSA HTML page fetch -- previously
    unprotected even though the neighboring per-sample BioSample HTML fetch
    already had this exact try/except."""
    good_html = """
    <div><b>Title:</b> Marine sediment amplicon reads</div>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "CRA0001" in url:
            raise httpx.ReadTimeout("simulated persistent timeout for CRA0001", request=request)
        if "CRA0002" in url:
            return httpx.Response(200, text=good_html)
        raise AssertionError(f"unexpected request: {url}")

    config = _fast_config(tmp_path)
    db = CncbDB(config.db_path)
    db.initialize()
    db.upsert_project(project_row_from_seed({"cncb_bioproject": "PRJCA0001", "cra_accessions": ["CRA0001"], "title": "p1", "description": None}))
    db.upsert_project(project_row_from_seed({"cncb_bioproject": "PRJCA0002", "cra_accessions": ["CRA0002"], "title": "p2", "description": None}))
    db.commit()
    db.close()

    runner = CncbGsaDiscoveryRunner(config, transport=httpx.MockTransport(handler))
    try:
        result = runner.run(phase="metadata")
    finally:
        runner.close()

    assert result["counts"]["metadata_cra_errors"] == 1

    conn = sqlite3.connect(config.db_path)
    conn.row_factory = sqlite3.Row
    rows = {row["cncb_bioproject"]: row["sequence_accessibility_status"] for row in conn.execute("SELECT cncb_bioproject, sequence_accessibility_status FROM cncb_projects")}
    conn.close()
    assert rows["PRJCA0001"] == "not_yet_checked"  # never merged -- the fetch failed before parsing
    assert rows["PRJCA0002"] != "not_yet_checked"  # successfully merged


def test_discovery_paginates_past_page_zero_even_though_the_api_ignores_size(tmp_path, monkeypatch):
    """Regression test for the real root cause behind a live-reported
    22-native-projects result that should have been much higher: the CNCB
    search API silently ignores the requested `size` param and always
    returns exactly 10 items per page (confirmed live 2026-08-26 --
    size=5/20/50/100/200 all came back with len(items)==10). The old
    stopping condition (`len(items) < page_size`) and `start` increment
    (`+= page_size`) both assumed the request controlled the actual page
    length, so every query term's loop broke after page 0 regardless of
    how many total records existed -- exactly 29 live search requests were
    made for 28 query terms in the real run that produced this bug report.
    This response always returns a fixed 10-item page (mirroring the real
    API) for a term with recordsTotal=25, spread across 3 pages."""
    monkeypatch.setattr(cncb_gsa_discovery, "DISCOVERY_QUERIES", ("multipage",))
    total = 25

    def handler(request: httpx.Request) -> httpx.Response:
        start = int(request.url.params.get("start", "0"))
        page_items = [
            {
                "id": f"PRJCA{i:04d}",
                "type": "BioProject",
                "title": "t",
                "attrs": {"Accession": f"PRJCA{i:04d}", "Center": "GSA", "CrasAcc": []},
            }
            for i in range(start, min(start + 10, total))
        ]
        return httpx.Response(200, json={"code": "200", "result": {"data": {"recordsTotal": total, "recordsFiltered": total, "data": page_items}}})

    config = _fast_config(tmp_path, max_pages_per_query=10)
    runner = CncbGsaDiscoveryRunner(config, transport=httpx.MockTransport(handler))
    try:
        result = runner.run(phase="discovery")
    finally:
        runner.close()

    assert result["counts"]["search_pages"] == 3  # start=0, 10, 20
    assert result["counts"]["native_projects_seen"] == 25

    conn = sqlite3.connect(config.db_path)
    stored = {row[0] for row in conn.execute("SELECT cncb_bioproject FROM cncb_projects")}
    conn.close()
    assert stored == {f"PRJCA{i:04d}" for i in range(total)}


def test_sample_row_to_db_never_produces_a_null_json_column(tmp_path):
    """Regression test for a real live crash: a BioSample fetch that fails
    (503/429 exhausting retries) falls back to the bare
    {"samc_accession": samc} dict. sample_row_to_db's own for-loop
    included other_environmental_measurements_json/source_metadata_json in
    its blind sample.get(field) pass, setting both to None on that bare
    dict -- the row.setdefault(...) calls that used to follow were then
    no-ops (setdefault only fills in an absent key, and the loop had
    already set both present-but-None), so a None got bound straight into
    a NOT NULL DEFAULT '{}' column, raising sqlite3.IntegrityError on
    upsert. Confirmed via temporary revert that this crashes without the
    fix and passes with it."""
    row = sample_row_to_db("PRJCA1", "CRA1", {"samc_accession": "SAMC1"})
    assert row["other_environmental_measurements_json"] == "{}"
    assert row["source_metadata_json"] is not None
    assert json.loads(row["source_metadata_json"]) == {"samc_accession": "SAMC1"}

    db = CncbDB(tmp_path / "cncb.sqlite")
    try:
        db.initialize()
        db.upsert_sample(row)  # must not raise sqlite3.IntegrityError
        db.commit()
    finally:
        db.close()


def test_failed_biosample_fetch_is_retried_on_the_next_run_not_cached_as_empty(tmp_path):
    """Regression test for a real live resumability bug found alongside
    the crash above: the old code wrote sample_html_path unconditionally,
    even when the fetch failed and sample_html was "". On a resubmit,
    sample_html_path.exists() looks identical to a real cache hit, so a
    purely transient rate-limit failure would be silently treated as
    "permanently no data" forever, rather than retried -- the exact
    opposite of what a resumable, "re-submit after walltime" job is
    supposed to do."""
    gsa_html = """
    <div><b>Title:</b> t</div>
    <tr class="experiment">
      <td class="experiments"><a href="browse/CRA1/CRX1">CRX1</a></td>
      <td>t</td><td>t</td><td>t</td>
      <td><a href="/biosample/browse/SAMC1">SAMC1</a></td>
    </tr>
    """
    biosample_html = "<table><tr><th>Sample name</th><td>Station A</td></tr></table>"

    state = {"biosample_should_fail": True}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/gsa/browse/" in url:
            return httpx.Response(200, text=gsa_html)
        if "/biosample/browse/" in url:
            if state["biosample_should_fail"]:
                return httpx.Response(503, text="unavailable")
            return httpx.Response(200, text=biosample_html)
        raise AssertionError(f"unexpected request: {url}")

    config = _fast_config(tmp_path, max_pages_per_query=1)
    db = CncbDB(config.db_path)
    db.initialize()
    db.upsert_project(project_row_from_seed({"cncb_bioproject": "PRJCA1", "cra_accessions": ["CRA1"], "title": "t", "description": None}))
    db.commit()
    db.close()

    # First run: BioSample fetch persistently fails.
    runner = CncbGsaDiscoveryRunner(config, transport=httpx.MockTransport(handler))
    try:
        runner.run(phase="metadata")
    finally:
        runner.close()

    project_dir = config.data_dir / "raw" / "PRJCA1"
    assert not (project_dir / "SAMC1.html").exists()  # not cached as if it had succeeded
    conn = sqlite3.connect(config.db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT sample_name, other_environmental_measurements_json FROM cncb_samples WHERE samc_accession = 'SAMC1'").fetchone()
    conn.close()
    assert row["sample_name"] is None
    assert row["other_environmental_measurements_json"] == "{}"

    # Second run (simulating a resubmit): BioSample fetch now succeeds --
    # must actually be retried, not skipped as "already cached."
    state["biosample_should_fail"] = False
    runner = CncbGsaDiscoveryRunner(config, transport=httpx.MockTransport(handler))
    try:
        runner.run(phase="metadata")
    finally:
        runner.close()

    assert (project_dir / "SAMC1.html").exists()
    conn = sqlite3.connect(config.db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT sample_name FROM cncb_samples WHERE samc_accession = 'SAMC1'").fetchone()
    conn.close()
    assert row["sample_name"] == "Station A"
