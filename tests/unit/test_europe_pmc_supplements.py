"""Tests for the supplementary-material discovery/retrieval additions to
sources/europe_pmc.py: discover_supplementary_materials (pure XML parsing,
no network), find_related_from_fulltext (externally-hosted repository
supplement DOIs), and fetch_supplementary_bundle (mocked HTTP, matching the
existing test_source_adapters_http.py pattern)."""
import httpx
import pytest

from fair_ocean_agent.sources.base import SourceConfig, SourceRecordNotFoundError
from fair_ocean_agent.sources.europe_pmc import (
    EuropePmcAdapter,
    SupplementReference,
    discover_supplementary_materials,
)

REAL_SHAPED_XML = """<article><body><sec>
<supplementary-material id="TS1" position="float"><media xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="Table_1.DOCX" mimetype="application" mime-subtype="vnd.openxmlformats-officedocument.wordprocessingml.document"><?cloudpmc-path 9202/1/a/Table_1.DOCX?><?cloudpmc-bucket app?><?size 16141?><caption><p>Click here for additional data file.</p></caption></media></supplementary-material>
<supplementary-material id="TS5" position="float"><media xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="Table_5.xlsx" mimetype="application" mime-subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet"><?cloudpmc-path 9202/1/b/Table_5.xlsx?><?cloudpmc-bucket app?><?size 308946?><caption><p>Click here for additional data file.</p></caption></media></supplementary-material>
</sec>
<sec id="dup-group">
<supplementary-material id="db_ds_1_reqid_" position="float"><media xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="Table_1.DOCX" mimetype="application" mime-subtype="vnd.openxmlformats-officedocument.wordprocessingml.document"><?cloudpmc-path 9202/1/a/Table_1.DOCX?><?cloudpmc-bucket app?><?size 16141?><caption><p>Click here for additional data file.</p></caption></media></supplementary-material>
</sec>
</body></article>"""


def test_discover_supplementary_materials_reads_filename_mimetype_and_size():
    refs = discover_supplementary_materials(REAL_SHAPED_XML)
    assert refs == [
        SupplementReference(
            supplement_id="TS1",
            file_name="Table_1.DOCX",
            mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            size_bytes=16141,
        ),
        SupplementReference(
            supplement_id="TS5",
            file_name="Table_5.xlsx",
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            size_bytes=308946,
        ),
    ]


def test_discover_supplementary_materials_dedupes_by_file_name():
    """Real articles repeat the same <supplementary-material> list verbatim
    under a second parent section (confirmed live) -- Table_1.DOCX appears
    twice in REAL_SHAPED_XML but must only be reported once."""
    refs = discover_supplementary_materials(REAL_SHAPED_XML)
    file_names = [r.file_name for r in refs]
    assert file_names.count("Table_1.DOCX") == 1


def test_discover_supplementary_materials_returns_empty_for_malformed_xml():
    assert discover_supplementary_materials("<not><valid") == []


def test_discover_supplementary_materials_returns_empty_when_none_referenced():
    assert discover_supplementary_materials("<article><body><sec><title>Methods</title></sec></body></article>") == []


def test_discover_supplementary_materials_handles_missing_size_pi():
    xml = """<article><supplementary-material id="X1"><media xmlns:xlink="http://www.w3.org/1999/xlink"
    xlink:href="no_size.csv" mimetype="text" mime-subtype="csv"></media></supplementary-material></article>"""
    refs = discover_supplementary_materials(xml)
    assert len(refs) == 1
    assert refs[0].size_bytes is None


def _adapter(handler=None) -> EuropePmcAdapter:
    from fair_ocean_agent.config import RetrievalConfig

    retrieval_config = RetrievalConfig(cache_enabled=False)
    transport = httpx.MockTransport(handler) if handler else None
    return EuropePmcAdapter(
        SourceConfig(name="europe_pmc", enabled=True, base_url="https://www.ebi.ac.uk/europepmc/webservices/rest"),
        retrieval_config,
        transport=transport,
    )


def test_find_related_from_fulltext_extracts_external_repository_doi():
    xml = """<article><body><sec>
    <supplementary-material id="ext1"><ext-link xmlns:xlink="http://www.w3.org/1999/xlink"
    xlink:href="https://doi.org/10.5061/DRYAD.abc123"/></supplementary-material>
    </sec></body></article>"""
    adapter = _adapter()
    related = adapter.find_related_from_fulltext(xml)
    assert len(related) == 1
    assert related[0].value == "10.5061/dryad.abc123"
    assert related[0].identifier_type.value == "dataset_doi"
    adapter.close()


def test_find_related_from_fulltext_skips_media_based_supplements():
    """A <supplementary-material> with a <media> child is Europe-PMC-hosted
    -- handled entirely by discover_supplementary_materials, never surfaced
    as a RelatedIdentifier here."""
    adapter = _adapter()
    assert adapter.find_related_from_fulltext(REAL_SHAPED_XML) == []
    adapter.close()


def test_find_related_from_fulltext_ignores_non_doi_ext_links():
    xml = """<article><supplementary-material id="ext1"><ext-link xmlns:xlink="http://www.w3.org/1999/xlink"
    xlink:href="https://example.org/not-a-doi"/></supplementary-material></article>"""
    adapter = _adapter()
    assert adapter.find_related_from_fulltext(xml) == []
    adapter.close()


def test_find_related_from_fulltext_dedupes_repeated_dois():
    xml = """<article>
    <supplementary-material id="a"><ext-link xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="https://doi.org/10.5061/dryad.abc123"/></supplementary-material>
    <supplementary-material id="b"><ext-link xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="https://doi.org/10.5061/dryad.abc123"/></supplementary-material>
    </article>"""
    adapter = _adapter()
    related = adapter.find_related_from_fulltext(xml)
    assert len(related) == 1
    adapter.close()


def test_fetch_supplementary_bundle_returns_raw_bytes():
    def handler(request):
        assert request.url.path.endswith("/PMC1/supplementaryFiles")
        return httpx.Response(200, content=b"PK\x03\x04fakezipcontent")

    adapter = _adapter(handler)
    content = adapter.fetch_supplementary_bundle("PMC1")
    assert content == b"PK\x03\x04fakezipcontent"
    adapter.close()


def test_fetch_supplementary_bundle_raises_on_404():
    def handler(request):
        return httpx.Response(404)

    adapter = _adapter(handler)
    with pytest.raises(SourceRecordNotFoundError):
        adapter.fetch_supplementary_bundle("PMC1")
    adapter.close()
