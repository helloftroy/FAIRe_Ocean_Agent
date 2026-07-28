"""Structured logging setup (section 22). Loads config/logging.yaml via
dictConfig, and adds a contextvars-backed filter so any log call inside a
`with log_context(run_id=..., task_id=..., study_id=...):` block gets those
fields attached to the record -- without threading them through every
function signature.
"""
from __future__ import annotations

import logging
import logging.config
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

import yaml

from fair_ocean_agent.config import CONFIG_DIR

_context: ContextVar[dict] = ContextVar("fair_ocean_log_context", default={})

_CONTEXT_FIELDS = ("run_id", "task_id", "study_id", "source_id", "adapter", "attempt")


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        ctx = _context.get()
        for field_name in _CONTEXT_FIELDS:
            setattr(record, field_name, ctx.get(field_name, "-"))
        return True


@contextmanager
def log_context(**fields: object) -> Iterator[None]:
    current = dict(_context.get())
    current.update({k: v for k, v in fields.items() if k in _CONTEXT_FIELDS})
    token = _context.set(current)
    try:
        yield
    finally:
        _context.reset(token)


def setup_logging() -> None:
    config_path = CONFIG_DIR / "logging.yaml"
    if config_path.exists():
        with config_path.open() as f:
            config = yaml.safe_load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO)

    context_filter = ContextFilter()
    for handler in logging.getLogger().handlers:
        handler.addFilter(context_filter)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
