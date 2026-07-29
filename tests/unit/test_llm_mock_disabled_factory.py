import pytest

from fair_ocean_agent.config import BenchmarkCandidateConfig, LLMConfig, LLMVerifierConfig
from fair_ocean_agent.llm.base import LLMBackendError
from fair_ocean_agent.llm.disabled import DisabledLLMBackend
from fair_ocean_agent.llm.factory import build_benchmark_backend, build_llm_backend, build_llm_verifier_backend
from fair_ocean_agent.llm.http_backend import OpenAICompatibleHTTPBackend
from fair_ocean_agent.llm.mock import MockLLMBackend


def test_mock_backend_cycles_through_responses():
    backend = MockLLMBackend(responses=["a", "b"])
    assert backend.generate("p").text == "a"
    assert backend.generate("p").text == "b"
    assert backend.generate("p").text == "a"  # cycles


def test_mock_backend_supports_callable_response():
    backend = MockLLMBackend(responses=lambda prompt: f"echo:{prompt}")
    assert backend.generate("hello").text == "echo:hello"


def test_mock_backend_defaults_to_empty_json_array():
    backend = MockLLMBackend()
    assert backend.generate("p").text == "[]"


def test_disabled_backend_always_raises():
    backend = DisabledLLMBackend()
    with pytest.raises(LLMBackendError):
        backend.generate("anything")


def test_factory_returns_disabled_when_llm_not_enabled():
    config = LLMConfig(enabled=False, provider="openai_compatible", model="some-model")
    assert isinstance(build_llm_backend(config), DisabledLLMBackend)


def test_factory_returns_mock_for_mock_provider():
    config = LLMConfig(enabled=True, provider="mock")
    assert isinstance(build_llm_backend(config), MockLLMBackend)


def test_factory_rejects_placeholder_model_name():
    config = LLMConfig(enabled=True, provider="openai_compatible", model="REPLACE_WITH_MODEL_NAME")
    with pytest.raises(LLMBackendError):
        build_llm_backend(config)


def test_factory_builds_http_backend_with_real_model_name():
    config = LLMConfig(enabled=True, provider="openai_compatible", model="llama-3.1-8b-instruct", base_url="http://localhost:11434/v1")
    backend = build_llm_backend(config)
    assert isinstance(backend, OpenAICompatibleHTTPBackend)
    assert backend.model == "llama-3.1-8b-instruct"
    backend.close()


def test_verifier_factory_returns_disabled_when_not_enabled():
    config = LLMVerifierConfig(enabled=False, model="granite3.3:8b")
    assert isinstance(build_llm_verifier_backend(config), DisabledLLMBackend)


def test_verifier_factory_rejects_placeholder_model_name():
    config = LLMVerifierConfig(enabled=True, provider="openai_compatible")
    with pytest.raises(LLMBackendError, match="llm_verifier.model"):
        build_llm_verifier_backend(config)


def test_verifier_factory_builds_http_backend_with_real_model_name():
    config = LLMVerifierConfig(
        enabled=True,
        provider="openai_compatible",
        model="granite3.3:8b",
        base_url="http://localhost:11434/v1",
    )
    backend = build_llm_verifier_backend(config)
    assert isinstance(backend, OpenAICompatibleHTTPBackend)
    assert backend.label == "granite3.3:8b"
    backend.close()


def test_factory_rejects_unknown_provider():
    config = LLMConfig(enabled=True, provider="something-else")
    with pytest.raises(LLMBackendError):
        build_llm_backend(config)


def test_build_benchmark_backend_rejects_placeholder_candidate():
    candidate = BenchmarkCandidateConfig(label="REPLACE_WITH_LABEL_A", base_url="http://x", model="REPLACE_WITH_MODEL_NAME_A")
    with pytest.raises(LLMBackendError):
        build_benchmark_backend(candidate)


def test_build_benchmark_backend_builds_real_candidate():
    candidate = BenchmarkCandidateConfig(label="my-local-model", base_url="http://localhost:11434/v1", model="my-model")
    backend = build_benchmark_backend(candidate)
    assert isinstance(backend, OpenAICompatibleHTTPBackend)
    assert backend.label == "my-local-model"
    backend.close()
