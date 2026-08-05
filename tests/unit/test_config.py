import os

import fair_ocean_agent.config as config_module
from fair_ocean_agent.config import load_config, reset_config_cache


def test_load_config_defaults(tmp_path, monkeypatch):
    # Isolate from a real config/local.yaml -- e.g. a developer's own
    # machine-specific overrides (a real Ollama endpoint, llm.enabled: true,
    # ...) for local testing. Without this, this test's outcome depends on
    # whatever the person running it happens to have configured locally,
    # which is exactly what local.yaml is *for* but must never leak into
    # what "defaults" means here.
    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)
    (tmp_path / "default.yaml").write_text("")

    reset_config_cache()
    monkeypatch.delenv("FAIR_OCEAN_DATABASE_URL", raising=False)
    monkeypatch.delenv("LOCAL_LLM_MODEL", raising=False)
    config = load_config(env_file=tmp_path / "does-not-exist.env")

    assert config.database.url == "sqlite:///data/fair_ocean.db"
    assert config.llm.enabled is False
    assert config.workflow.max_attempts == 3
    reset_config_cache()


def test_env_override_wins_over_yaml(monkeypatch, tmp_path):
    reset_config_cache()
    monkeypatch.setenv("FAIR_OCEAN_DATABASE_URL", "sqlite:///override.db")
    config = load_config(env_file=tmp_path / "does-not-exist.env")

    assert config.database.url == "sqlite:///override.db"
    reset_config_cache()
    monkeypatch.delenv("FAIR_OCEAN_DATABASE_URL", raising=False)


def test_llm_size_env_overrides(monkeypatch, tmp_path):
    reset_config_cache()
    monkeypatch.setenv("LOCAL_LLM_TIMEOUT_SECONDS", "360")
    monkeypatch.setenv("LOCAL_LLM_MAX_OUTPUT_TOKENS", "4096")
    monkeypatch.setenv("LOCAL_LLM_EXTRACTION_MAX_CHARS_PER_CALL", "20000")
    monkeypatch.setenv("LOCAL_LLM_NUM_CTX", "8192")

    config = load_config(env_file=tmp_path / "does-not-exist.env")

    assert config.llm.timeout_seconds == 360
    assert config.llm.max_output_tokens == 4096
    assert config.llm.extraction_max_chars_per_call == 20000
    assert config.llm.num_ctx == 8192
    reset_config_cache()
    monkeypatch.delenv("LOCAL_LLM_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("LOCAL_LLM_MAX_OUTPUT_TOKENS", raising=False)
    monkeypatch.delenv("LOCAL_LLM_EXTRACTION_MAX_CHARS_PER_CALL", raising=False)
    monkeypatch.delenv("LOCAL_LLM_NUM_CTX", raising=False)


def test_llm_size_env_overrides_cannot_shrink_extraction_budget(monkeypatch, tmp_path):
    reset_config_cache()
    monkeypatch.setenv("LOCAL_LLM_MAX_OUTPUT_TOKENS", "1024")
    monkeypatch.setenv("LOCAL_LLM_EXTRACTION_MAX_CHARS_PER_CALL", "2500")

    config = load_config(env_file=tmp_path / "does-not-exist.env")

    assert config.llm.max_output_tokens == 2048
    assert config.llm.extraction_max_chars_per_call == 16000
    reset_config_cache()
    monkeypatch.delenv("LOCAL_LLM_MAX_OUTPUT_TOKENS", raising=False)
    monkeypatch.delenv("LOCAL_LLM_EXTRACTION_MAX_CHARS_PER_CALL", raising=False)


def test_llm_verifier_env_overrides(monkeypatch, tmp_path):
    reset_config_cache()
    monkeypatch.setenv("LOCAL_LLM_VERIFIER_ENABLED", "true")
    monkeypatch.setenv("LOCAL_LLM_VERIFIER_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("LOCAL_LLM_VERIFIER_MODEL", "granite3.3:8b")
    monkeypatch.setenv("LOCAL_LLM_VERIFIER_TIMEOUT_SECONDS", "300")
    monkeypatch.setenv("LOCAL_LLM_VERIFIER_MAX_OUTPUT_TOKENS", "256")
    monkeypatch.setenv("LOCAL_LLM_VERIFIER_NUM_CTX", "4096")
    config = load_config(env_file=tmp_path / ".env")

    assert config.llm_verifier.enabled is True
    assert config.llm_verifier.base_url == "http://localhost:11434/v1"
    assert config.llm_verifier.model == "granite3.3:8b"
    assert config.llm_verifier.timeout_seconds == 300
    assert config.llm_verifier.max_output_tokens == 256
    assert config.llm_verifier.num_ctx == 4096

    reset_config_cache()
    monkeypatch.delenv("LOCAL_LLM_VERIFIER_ENABLED", raising=False)
    monkeypatch.delenv("LOCAL_LLM_VERIFIER_BASE_URL", raising=False)
    monkeypatch.delenv("LOCAL_LLM_VERIFIER_MODEL", raising=False)
    monkeypatch.delenv("LOCAL_LLM_VERIFIER_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("LOCAL_LLM_VERIFIER_MAX_OUTPUT_TOKENS", raising=False)
    monkeypatch.delenv("LOCAL_LLM_VERIFIER_NUM_CTX", raising=False)


def test_supplement_llm_env_override_is_explicit(monkeypatch, tmp_path):
    reset_config_cache()
    monkeypatch.setenv("FAIR_OCEAN_SUPPLEMENT_LLM_ENABLED", "true")

    config = load_config(env_file=tmp_path / "does-not-exist.env")

    assert config.supplements.llm_text_extraction_enabled is True
    reset_config_cache()
    monkeypatch.delenv("FAIR_OCEAN_SUPPLEMENT_LLM_ENABLED", raising=False)
