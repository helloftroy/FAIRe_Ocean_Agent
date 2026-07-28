"""Mock LLMBackend: for tests, and for exercising the benchmark harness and
extraction pipeline without a live inference server. Never makes a network
call."""
from __future__ import annotations

import time
from collections.abc import Callable

from fair_ocean_agent.clock import utcnow
from fair_ocean_agent.llm.base import LLMBackend, LLMResponse


class MockLLMBackend(LLMBackend):
    def __init__(
        self,
        label: str = "mock",
        responses: list[str] | Callable[[str], str] | None = None,
        simulated_latency_seconds: float = 0.0,
    ):
        """`responses`: a fixed string returned for every call (if a plain
        list of one), a list cycled through call-by-call, or a callable
        `(prompt) -> response_text` for prompt-dependent behavior. Defaults
        to `"[]"` (an empty JSON array) if not given."""
        self.label = label
        self._responses = responses
        self._call_index = 0
        self._simulated_latency = simulated_latency_seconds
        self.calls: list[dict] = []  # test/benchmark introspection

    def generate(
        self, prompt: str, *, system: str | None = None, temperature: float = 0, max_tokens: int | None = None
    ) -> LLMResponse:
        self.calls.append({"prompt": prompt, "system": system, "temperature": temperature})
        if self._simulated_latency:
            time.sleep(self._simulated_latency)

        if callable(self._responses):
            text = self._responses(prompt)
        elif self._responses:
            text = self._responses[self._call_index % len(self._responses)]
            self._call_index += 1
        else:
            text = "[]"

        return LLMResponse(
            text=text,
            raw={"mock": True},
            model="mock-model",
            backend_label=self.label,
            latency_seconds=self._simulated_latency,
            requested_at=utcnow(),
        )
