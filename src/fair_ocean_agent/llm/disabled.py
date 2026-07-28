"""LLMBackend that raises clearly on any call -- the default whenever
llm.enabled is false, so a code path that accidentally tries to use the LLM
fails loudly and immediately instead of silently returning empty or
fabricated data."""
from __future__ import annotations

from fair_ocean_agent.llm.base import LLMBackend, LLMBackendError, LLMResponse


class DisabledLLMBackend(LLMBackend):
    label = "disabled"

    def generate(
        self, prompt: str, *, system: str | None = None, temperature: float = 0, max_tokens: int | None = None
    ) -> LLMResponse:
        raise LLMBackendError(
            "The LLM backend is disabled (llm.enabled: false in config). Set "
            "llm.enabled: true and llm.provider: openai_compatible with a real "
            "base_url/model (a currently-running Ollama/vLLM/institutional "
            "endpoint) to use text extraction or the benchmark harness."
        )
