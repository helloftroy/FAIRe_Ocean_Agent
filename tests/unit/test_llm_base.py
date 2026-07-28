from fair_ocean_agent.clock import utcnow
from fair_ocean_agent.llm.base import LLMBackend, LLMResponse


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
