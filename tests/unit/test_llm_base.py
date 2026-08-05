from fair_ocean_agent.clock import utcnow
from fair_ocean_agent.llm.base import LLMBackend, LLMResponse, try_parse_json


class _ScriptedBackend(LLMBackend):
    """Returns a scripted sequence of raw texts, one per generate() call --
    for testing generate_json's retry-on-invalid-JSON loop precisely."""

    def __init__(self, texts: list[str], label: str = "scripted"):
        self.label = label
        self._texts = texts
        self._index = 0
        self.calls: list[str] = []

    def generate(self, prompt, *, system=None, temperature=0, max_tokens=None) -> LLMResponse:
        self.calls.append(prompt)
        text = self._texts[min(self._index, len(self._texts) - 1)]
        self._index += 1
        return LLMResponse(
            text=text, raw={}, model="scripted-model", backend_label=self.label,
            latency_seconds=0.0, requested_at=utcnow(),
        )


def test_generate_json_parses_valid_json_first_try():
    backend = _ScriptedBackend(['[{"a": 1}]'])
    parsed, response = backend.generate_json("prompt")
    assert parsed == [{"a": 1}]
    assert len(backend.calls) == 1


def test_generate_json_strips_markdown_fences():
    backend = _ScriptedBackend(['```json\n{"a": 1}\n```'])
    parsed, _ = backend.generate_json("prompt")
    assert parsed == {"a": 1}


def test_generate_json_retries_on_invalid_json_then_succeeds():
    backend = _ScriptedBackend(["not json at all", '{"a": 1}'])
    parsed, _ = backend.generate_json("prompt", max_retries=2)
    assert parsed == {"a": 1}
    assert len(backend.calls) == 2
    assert "not valid JSON" in backend.calls[1]  # corrective follow-up prompt


def test_generate_json_gives_up_after_max_retries():
    backend = _ScriptedBackend(["still not json", "still not json", "still not json"])
    parsed, last_response = backend.generate_json("prompt", max_retries=2)
    assert parsed is None
    assert len(backend.calls) == 3  # initial + 2 retries
    assert last_response is not None


def test_try_parse_json_recovers_complete_objects_from_a_truncated_array():
    """Regression guard for a real production failure mode (confirmed live
    against a real dense PCR-methods paragraph, PeerJ 10.7717/peerj.333): a
    max_tokens budget cut the model's response off mid-object after it had
    already correctly produced several complete facts. The old behavior
    (a single json.loads() call, nothing on failure) discarded every one
    of those correct facts along with the one that didn't finish."""
    truncated = (
        '[{"fact_type_candidate": "assay_type", "raw_value": "metabarcoding"},'
        '{"fact_type_candidate": "target_gene", "raw_value": "18S SSU"},'
        '{"fact_type_candidate": "forward_primer_sequence", "raw_value": "TCTC'
    )
    parsed = try_parse_json(truncated)
    assert parsed == [
        {"fact_type_candidate": "assay_type", "raw_value": "metabarcoding"},
        {"fact_type_candidate": "target_gene", "raw_value": "18S SSU"},
    ]


def test_try_parse_json_returns_none_for_a_truncated_non_array():
    """Recovery only ever applies to a bare top-level array (the only
    shape generate_json's callers ever produce) -- a truncated single
    object must still fail closed, not attempt a best-effort partial
    parse of something that was never a list to begin with."""
    assert try_parse_json('{"fact_type_candidate": "assay_type", "raw_valu') is None


def test_try_parse_json_returns_none_when_zero_objects_completed():
    """Cut off before even the first object finished -- nothing to
    recover, must still behave exactly like the old all-or-nothing parse
    (None, triggering generate_json's retry)."""
    assert try_parse_json('[{"fact_type_candidate": "assay_type", "raw_valu') is None


def test_generate_json_accepts_a_partially_truncated_array_without_retrying():
    truncated = (
        '[{"fact_type_candidate": "assay_type", "raw_value": "metabarcoding"},'
        '{"fact_type_candidate": "target_gene", "raw_value": "18S SSU"},'
        '{"fact_type_candidate": "forward_primer_sequence", "raw_value": "TCTC'
    )
    backend = _ScriptedBackend([truncated])
    parsed, _ = backend.generate_json("prompt", max_retries=2)
    assert parsed == [
        {"fact_type_candidate": "assay_type", "raw_value": "metabarcoding"},
        {"fact_type_candidate": "target_gene", "raw_value": "18S SSU"},
    ]
    assert len(backend.calls) == 1  # accepted the partial recovery, no retry wasted
