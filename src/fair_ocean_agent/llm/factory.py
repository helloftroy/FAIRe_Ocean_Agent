"""Builds the configured LLMBackend. Never picks or defaults to a specific
model -- llm.model must be set explicitly (config/default.yaml or the
LOCAL_LLM_MODEL env var) whenever provider is "openai_compatible"."""
from __future__ import annotations

import os

from fair_ocean_agent.config import BenchmarkCandidateConfig, LLMConfig
from fair_ocean_agent.llm.base import LLMBackend, LLMBackendError
from fair_ocean_agent.llm.disabled import DisabledLLMBackend
from fair_ocean_agent.llm.http_backend import OpenAICompatibleHTTPBackend
from fair_ocean_agent.llm.mock import MockLLMBackend

_PLACEHOLDER_MODEL_NAMES = {"", "REPLACE_WITH_MODEL_NAME"}


def build_llm_backend(config: LLMConfig) -> LLMBackend:
    if not config.enabled or config.provider == "disabled":
        return DisabledLLMBackend()

    if config.provider == "mock":
        return MockLLMBackend()

    if config.provider == "openai_compatible":
        if config.model in _PLACEHOLDER_MODEL_NAMES:
            raise LLMBackendError(
                "llm.model is not set. This project never assumes a default model -- "
                "set LOCAL_LLM_MODEL (or llm.model in config/default.yaml) to the exact "
                "model name your inference server expects."
            )
        api_key = os.environ.get(config.api_key_env) if config.api_key_env else None
        return OpenAICompatibleHTTPBackend(
            label=config.model,
            base_url=config.base_url,
            model=config.model,
            api_key=api_key,
            timeout_seconds=config.timeout_seconds,
            max_concurrency=config.max_concurrency,
            num_ctx=config.num_ctx,
            default_max_tokens=config.max_output_tokens,
        )

    raise LLMBackendError(f"Unknown llm.provider: {config.provider!r}")


def build_benchmark_backend(candidate: BenchmarkCandidateConfig) -> LLMBackend:
    """Builds one candidate model's backend for llm/benchmark.py. Unlike
    build_llm_backend, there's no "disabled"/"mock" branch here -- every
    row in config/benchmark_models.yaml is assumed to be a real endpoint
    the caller wants compared; a placeholder label/model left unfilled
    raises immediately rather than silently benchmarking nothing."""
    if candidate.model in _PLACEHOLDER_MODEL_NAMES or candidate.label.startswith("REPLACE_WITH"):
        raise LLMBackendError(
            f"Benchmark candidate {candidate.label!r} still has placeholder values -- "
            "fill in config/benchmark_models.yaml with a real label/base_url/model "
            "pointing at a currently-running inference endpoint."
        )
    api_key = os.environ.get(candidate.api_key_env) if candidate.api_key_env else None
    return OpenAICompatibleHTTPBackend(
        label=candidate.label,
        base_url=candidate.base_url,
        model=candidate.model,
        api_key=api_key,
        timeout_seconds=candidate.timeout_seconds,
        max_concurrency=candidate.max_concurrency,
        num_ctx=candidate.num_ctx,
        default_max_tokens=candidate.max_output_tokens,
    )
