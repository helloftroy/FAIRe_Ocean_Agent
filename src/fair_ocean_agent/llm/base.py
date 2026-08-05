"""Provider-independent LLM backend abstraction (section 11).

No backend implementation in this package assumes or requires a specific
vendor. The "openai_compatible" backend (llm/http_backend.py) refers only
to the HTTP request/response *wire protocol* -- the same shape Ollama,
vLLM, TGI, LocalAI, and most institutional inference gateways all speak.
It does not mean OpenAI-hosted models, does not require OpenAI
credentials, and never contacts an OpenAI-operated endpoint. The
configured base_url is the only thing that determines where a request
goes.

This project never selects or defaults to a specific model -- see
llm/factory.py and config/benchmark_models.yaml. Candidate models are
compared, not chosen, via llm/benchmark.py.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime

from pydantic import BaseModel


class LLMBackendError(Exception):
    """Raised for backend-level failures: connection refused, timeout,
    non-2xx response, an unexpected response shape, or a disabled backend
    being called."""


class LLMResponse(BaseModel):
    text: str
    raw: dict
    model: str
    backend_label: str
    latency_seconds: float
    requested_at: datetime
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class LLMBackend(ABC):
    """`label` identifies this backend instance in benchmark reports and in
    raw_facts.model_name -- it's whatever the caller/config names it (e.g.
    a model name plus quantization), never inferred or hard-coded here."""

    label: str

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        ...

    def generate_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0,
        max_tokens: int | None = None,
        max_retries: int = 2,
    ) -> tuple[dict | list | None, LLMResponse | None]:
        """Generate and parse the response as JSON, retrying with a
        corrective follow-up prompt on invalid JSON (section 11: "Retry on
        invalid JSON"). Returns (None, last_response) if every attempt
        fails -- callers must treat that as "no fact produced", never fill
        in a default, per section 2 ("unknown or blank is preferable to
        guessing"). No automatic acceptance of model output happens here;
        this only parses JSON, it does not validate the facts within it."""
        attempt_prompt = prompt
        last_response: LLMResponse | None = None
        for _ in range(max_retries + 1):
            response = self.generate(attempt_prompt, system=system, temperature=temperature, max_tokens=max_tokens)
            last_response = response
            parsed = try_parse_json(response.text)
            if parsed is not None:
                return parsed, response
            attempt_prompt = (
                f"{prompt}\n\nYour previous response was not valid JSON:\n{response.text}\n\n"
                "Respond again with ONLY valid JSON -- no prose, no markdown code fences."
            )
        return None, last_response

    def close(self) -> None:
        pass


def try_parse_json(text: str):
    """Parse the JSON response forms accepted by `generate_json`.

    Public so orchestration code can distinguish a valid empty result from
    exhausted JSON-repair attempts. Those have identical fact lists but
    must produce different task outcomes.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return _try_parse_truncated_json_array(text)


def _try_parse_truncated_json_array(text: str) -> list | None:
    """Recovers as many complete objects as possible from a top-level JSON
    array cut off mid-object by a max_tokens budget -- confirmed live
    against a real dense PCR-methods paragraph (PeerJ 10.7717/peerj.333):
    the model correctly extracted 8+ facts in order before being cut off
    mid-object at the configured token limit, and the single invalid-JSON
    check in try_parse_json discarded the entire response, losing every
    fact the model got right along with the one it didn't finish. Only
    ever attempted for a bare top-level array (the only shape
    generate_json's callers ever produce) -- never for any other malformed
    JSON shape, where a partial/best-effort parse would be more likely to
    fabricate structure the model never actually produced."""
    stripped = text.strip()
    if not stripped.startswith("["):
        return None
    decoder = json.JSONDecoder()
    index = 1
    length = len(stripped)
    items: list = []
    while index < length:
        while index < length and stripped[index] in " \t\n\r,":
            index += 1
        if index >= length or stripped[index] == "]":
            break
        try:
            item, index = decoder.raw_decode(stripped, index)
        except json.JSONDecodeError:
            break  # the rest is a truncated fragment -- stop, keep what parsed
        items.append(item)
    return items or None
