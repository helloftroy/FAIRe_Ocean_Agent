"""OpenAICompatibleHTTPBackend tested via httpx.MockTransport -- no live
network, no real inference server. Confirms the request/response shape
this backend speaks (POST {base_url}/chat/completions) and that it never
reaches outside the configured base_url."""
import httpx
import pytest

from fair_ocean_agent.llm.base import LLMBackendError
from fair_ocean_agent.llm.http_backend import OpenAICompatibleHTTPBackend


def _backend(handler, **kwargs) -> OpenAICompatibleHTTPBackend:
    return OpenAICompatibleHTTPBackend(
        label=kwargs.pop("label", "test-model"),
        base_url=kwargs.pop("base_url", "http://localhost:11434/v1"),
        model=kwargs.pop("model", "test-model"),
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def test_generate_sends_openai_shaped_request_and_parses_response():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = request.read()
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hello world"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            },
        )

    backend = _backend(handler)
    response = backend.generate("say hi", system="be terse", temperature=0.5)

    assert response.text == "hello world"
    assert response.prompt_tokens == 10
    assert response.completion_tokens == 2
    assert response.backend_label == "test-model"
    assert captured["url"].endswith("/chat/completions")

    import json as json_module
    body = json_module.loads(captured["body"])
    assert body["model"] == "test-model"
    assert body["temperature"] == 0.5
    assert body["messages"] == [{"role": "system", "content": "be terse"}, {"role": "user", "content": "say hi"}]


def test_generate_never_sends_openai_api_credentials_by_default():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "Authorization" not in request.headers
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    backend = _backend(handler)
    backend.generate("prompt")


def test_generate_sends_configured_api_key_as_bearer_token():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer local-secret"
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    backend = _backend(handler, api_key="local-secret")
    backend.generate("prompt")


def test_generate_raises_llm_backend_error_on_http_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    backend = _backend(handler)
    with pytest.raises(LLMBackendError):
        backend.generate("prompt")


def test_generate_raises_on_unexpected_response_shape():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    backend = _backend(handler)
    with pytest.raises(LLMBackendError):
        backend.generate("prompt")


def test_num_ctx_is_omitted_by_default():
    """Regression guard: omitting num_ctx must leave the request body exactly
    as before this option existed -- no behavior change for anyone not using
    it (vLLM/TGI/other OpenAI-compatible servers that don't understand an
    "options" field)."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json as json_module

        captured["body"] = json_module.loads(request.read())
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    backend = _backend(handler)
    backend.generate("prompt")
    assert "options" not in captured["body"]


def test_num_ctx_is_sent_as_an_options_field_when_configured():
    """Found necessary during a live 100-study audit: real paper sections
    plus the FAIRe-aware v3 prompt's ~70-concept checklist routinely exceed
    a small Ollama-served model's default 4096-token context, something
    gold-case benchmarking's short synthetic snippets never approached.
    This backend sends the option regardless -- confirmed live that
    Ollama's *own* OpenAI-compatible endpoint silently ignores it (its
    native /api/chat endpoint honors the identical option; ollama's
    OpenAI-compat shim does not), so this only helps on OpenAI-compatible
    servers that do respect an extra options field. See
    LLMConfig.num_ctx's docstring for the real Ollama-side fix."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json as json_module

        captured["body"] = json_module.loads(request.read())
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    backend = _backend(handler, num_ctx=8192)
    backend.generate("prompt")
    assert captured["body"]["options"] == {"num_ctx": 8192}
