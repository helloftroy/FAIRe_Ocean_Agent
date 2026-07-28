from fair_ocean_agent.extraction.evidence import verify_evidence_quote


def test_exact_quote_verifies():
    assert verify_evidence_quote("samples were collected", "text: samples were collected here")


def test_whitespace_differences_are_normalized():
    quote = "samples   were\ncollected"
    source = "text: samples were collected here"
    assert verify_evidence_quote(quote, source)


def test_quote_not_in_source_fails():
    assert not verify_evidence_quote("this text does not exist", "some other text entirely")


def test_empty_quote_always_fails():
    assert not verify_evidence_quote("", "any source text")
    assert not verify_evidence_quote("   ", "any source text")


def test_paraphrase_does_not_verify():
    # deliberately exact-match only -- a paraphrase must fail, not fuzzy-pass
    assert not verify_evidence_quote("we gathered water samples", "we collected water samples")
