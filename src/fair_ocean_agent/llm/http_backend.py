"""HTTP LLMBackend implementing the OpenAI-compatible chat-completions wire
protocol: POST {base_url}/chat/completions with
{"model", "messages", "temperature", ...}, expecting back
{"choices": [{"message": {"content": ...}}], "usage": {...}}.

"OpenAI-compatible" here describes the request/response SHAPE ONLY. Ollama,
vLLM, TGI's OpenAI-compatible router, LocalAI, and most institutional
inference gateways all speak this same shape. This class:

- never contacts any OpenAI-operated host -- base_url fully determines the
  destination, and is required to be set explicitly (no default that
  resolves to a real endpoint);
- never requires or reads OpenAI credentials -- api_key, if given, is sent
  as a bearer token to whatever `base_url` is configured, exactly as any
  self-hosted or institutional gateway would expect;
- sends data only to that configured base_url, nowhere else.

No model name is chosen or defaulted anywhere in this file.
"""
from __future__ import annotations

import threading
import time

import httpx

from fair_ocean_agent.clock import utcnow
from fair_ocean_agent.llm.base import LLMBackend, LLMBackendError, LLMResponse


class OpenAICompatibleHTTPBackend(LLMBackend):
    def __init__(
        self,
        label: str,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 180,
        max_concurrency: int = 1,
        transport: httpx.BaseTransport | None = None,
        num_ctx: int | None = None,
    ):
        self.label = label
        self.model = model
        self._num_ctx = num_ctx
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
            headers=headers,
            transport=transport,
        )
        # Bounds concurrent requests to this specific endpoint -- relevant
        # when the benchmark harness or a future concurrent worker calls
        # the same backend from multiple threads; a local/institutional
        # inference server is typically latency- not rate-limited, so this
        # is a concurrency cap, not a requests-per-second throttle like
        # sources/base.py's RateLimitedClient.
        self._semaphore = threading.BoundedSemaphore(max_concurrency)

    def generate(
        self, prompt: str, *, system: str | None = None, temperature: float = 0, max_tokens: int | None = None
    ) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {"model": self.model, "messages": messages, "temperature": temperature}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if self._num_ctx is not None:
            # Not honored by Ollama's own OpenAI-compatible endpoint (verified
            # live) -- see LLMConfig.num_ctx's docstring for the real fix on
            # Ollama. Harmless no-op there; may work on other OpenAI-compatible
            # servers that do respect an extra options field.
            payload["options"] = {"num_ctx": self._num_ctx}

        requested_at = utcnow()
        start = time.monotonic()
        with self._semaphore:
            try:
                response = self._client.post("/chat/completions", json=payload)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise LLMBackendError(
                    f"{self.label}: request to {self._client.base_url}/chat/completions failed: {exc}"
                ) from exc
        latency = time.monotonic() - start

        data = response.json()
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMBackendError(f"{self.label}: unexpected response shape: {data}") from exc

        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            raw=data,
            model=self.model,
            backend_label=self.label,
            latency_seconds=latency,
            requested_at=requested_at,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )

    def close(self) -> None:
        self._client.close()
