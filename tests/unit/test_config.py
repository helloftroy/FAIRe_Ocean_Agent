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
