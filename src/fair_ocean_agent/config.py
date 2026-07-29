"""Configuration loading: config/default.yaml + config/local.yaml (optional,
gitignored) + environment variable overrides, in that order of precedence.

Environment variables are read via python-dotenv (.env) and override YAML
values for the specific keys below. Secrets (API keys, tokens) must only ever
come from the environment, never from YAML.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


class DatabaseConfig(BaseModel):
    url: str = "sqlite:///data/fair_ocean.db"


class WorkflowConfig(BaseModel):
    worker_id: str = "local-worker"
    max_attempts: int = 3
    task_timeout_seconds: int = 600
    stale_claim_minutes: int = 30
    default_batch_size: int = 25


class LLMConfig(BaseModel):
    """provider is one of "disabled", "mock", or "openai_compatible" (see
    llm/factory.py). "openai_compatible" names the HTTP request/response
    *wire protocol* that Ollama, vLLM, TGI, and most institutional
    inference gateways all speak -- it does not mean OpenAI-hosted models,
    OpenAI credentials, or any external data transmission. base_url is the
    only thing that determines where requests go, and model is never
    defaulted to a real value -- see llm/factory.py's build_llm_backend."""

    enabled: bool = False
    provider: str = "mock"
    base_url: str = "http://localhost:11434/v1"
    model: str = "REPLACE_WITH_MODEL_NAME"
    api_key_env: str = "LOCAL_LLM_API_KEY"
    timeout_seconds: int = 180
    max_retries: int = 3
    temperature: float = 0
    max_output_tokens: int | None = None
    extraction_max_chars_per_call: int = 1600
    max_concurrency: int = 1
    # Sent as {"options": {"num_ctx": ...}} in the request body -- an extra
    # field most OpenAI-compatible servers simply ignore, so this stays
    # opt-in and harmless (None matches prior behavior exactly). Found
    # necessary during a live 100-study audit: real paper sections plus the
    # FAIRe-aware v3 prompt's ~70-concept checklist routinely exceed 4096
    # tokens, well past what gold-case benchmarking's short synthetic
    # snippets ever approached. IMPORTANT, verified live against a real
    # Ollama server: Ollama's own OpenAI-compatible endpoint
    # (/v1/chat/completions) silently DROPS this field -- confirmed by
    # sending an identical request straight to Ollama's *native* /api/chat
    # endpoint with the same options, which DOES honor it (succeeds where
    # the OpenAI-compat route still 400s with the same "exceeds context
    # size" error). This field is still worth setting for any
    # OpenAI-compatible server that does respect an extra options field, but
    # for Ollama specifically the only working fix is server-level: restart
    # Ollama with OLLAMA_CONTEXT_LENGTH set, or `ollama create` a model
    # variant whose Modelfile sets `PARAMETER num_ctx`. Neither of those is
    # something this project should do for you automatically (they're
    # machine-wide Ollama configuration, not per-request).
    num_ctx: int | None = None


class RetrievalConfig(BaseModel):
    request_timeout_seconds: int = 60
    cache_enabled: bool = True
    cache_dir: str = "data/cache"
    user_agent: str = "fair-ocean-agent/0.1 (contact: REPLACE_WITH_CONTACT_EMAIL)"


class DiscoveryConfig(BaseModel):
    citation_expansion_max_depth: int = 1
    keyword_search_enabled: bool = False


class SchedulingConfig(BaseModel):
    weekly_overlap_days: int = 14
    retry_failed_after_hours: int = 24
    monthly_unresolved_retry: bool = True
    quarterly_full_rediscovery: bool = True


class LoggingConfig(BaseModel):
    level: str = "INFO"
    json_output: bool = Field(default=False, alias="json")

    model_config = {"populate_by_name": True}


class SourceEntryConfig(BaseModel):
    enabled: bool = False
    base_url: str
    rate_limit_per_second: float = 3.0
    priority: int = 100


def load_sources_config() -> dict[str, SourceEntryConfig]:
    """Loads config/sources.yaml separately from the main AppConfig -- it's
    keyed by adapter name (open-ended, one entry per source.py module) so
    doesn't fit AppConfig's fixed-field model."""
    raw = _load_yaml(CONFIG_DIR / "sources.yaml")
    return {
        name: SourceEntryConfig.model_validate(entry)
        for name, entry in (raw.get("sources") or {}).items()
    }


class BenchmarkCandidateConfig(BaseModel):
    label: str
    base_url: str
    model: str
    api_key_env: str | None = None
    timeout_seconds: int = 180
    max_concurrency: int = 1
    num_ctx: int | None = None
    max_output_tokens: int | None = None


def load_benchmark_candidates() -> list[BenchmarkCandidateConfig]:
    """Loads config/benchmark_models.yaml -- an open-ended list of candidate
    open-weight models to compare via the benchmark harness
    (llm/benchmark.py). This project never chooses or defaults to a
    specific model: every entry here must be filled in with a real,
    currently-running inference endpoint (Ollama, vLLM, an institutional
    gateway, ...) before it's usable, and the list can hold any number of
    candidates for side-by-side comparison."""
    raw = _load_yaml(CONFIG_DIR / "benchmark_models.yaml")
    return [BenchmarkCandidateConfig.model_validate(entry) for entry in (raw.get("candidates") or [])]


class StandardConfig(BaseModel):
    version: str | None = None  # e.g. miop has no version file/tag upstream at all
    schema_source: str | None = None
    schema_source_commit: str | None = None
    note: str | None = None


def load_standards_config() -> dict[str, StandardConfig]:
    """Loads config/models.yaml's `standards` section -- the pinned
    schema/version each mapping module (mapping/faire.py, mapping/bebop.py)
    targets. Kept out of AppConfig since it's keyed by standard name, not a
    fixed set of fields."""
    raw = _load_yaml(CONFIG_DIR / "models.yaml")
    return {
        name: StandardConfig.model_validate(entry)
        for name, entry in (raw.get("standards") or {}).items()
    }


class AppConfig(BaseModel):
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    scheduling: SchedulingConfig = Field(default_factory=SchedulingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    raw = dict(raw)
    db_url = os.environ.get("FAIR_OCEAN_DATABASE_URL")
    if db_url:
        raw.setdefault("database", {})["url"] = db_url

    llm_base_url = os.environ.get("LOCAL_LLM_BASE_URL")
    if llm_base_url:
        raw.setdefault("llm", {})["base_url"] = llm_base_url

    llm_model = os.environ.get("LOCAL_LLM_MODEL")
    if llm_model:
        raw.setdefault("llm", {})["model"] = llm_model

    llm_max_output_tokens = os.environ.get("LOCAL_LLM_MAX_OUTPUT_TOKENS")
    if llm_max_output_tokens:
        raw.setdefault("llm", {})["max_output_tokens"] = int(llm_max_output_tokens)

    llm_extraction_max_chars = os.environ.get("LOCAL_LLM_EXTRACTION_MAX_CHARS_PER_CALL")
    if llm_extraction_max_chars:
        raw.setdefault("llm", {})["extraction_max_chars_per_call"] = int(llm_extraction_max_chars)

    llm_num_ctx = os.environ.get("LOCAL_LLM_NUM_CTX")
    if llm_num_ctx:
        raw.setdefault("llm", {})["num_ctx"] = int(llm_num_ctx)

    contact_email = os.environ.get("FAIR_OCEAN_CONTACT_EMAIL")
    if contact_email:
        raw.setdefault("retrieval", {})["user_agent"] = (
            f"fair-ocean-agent/0.1 (contact: {contact_email})"
        )

    return raw


@lru_cache(maxsize=1)
def load_config(env_file: str | Path | None = None) -> AppConfig:
    """Load and cache the merged application configuration.

    Precedence (lowest to highest): config/default.yaml -> config/local.yaml
    -> environment variables (from .env if present, then real env).
    """
    load_dotenv(dotenv_path=env_file or (REPO_ROOT / ".env"), override=False)

    raw = _load_yaml(CONFIG_DIR / "default.yaml")
    raw = _deep_merge(raw, _load_yaml(CONFIG_DIR / "local.yaml"))
    raw = _apply_env_overrides(raw)

    return AppConfig.model_validate(raw)


def reset_config_cache() -> None:
    """Clear the cached config; mainly useful in tests."""
    load_config.cache_clear()
