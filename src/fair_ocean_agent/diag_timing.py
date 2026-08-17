"""TEMPORARY diagnostic timing instrumentation -- not meant to run by
default and not meant to be a permanent feature. Safe to delete this file
and its two call sites (llm/http_backend.py::generate,
sources/base.py::RateLimitedClient._get_bytes) once the diagnostic run
that prompted this is done.

Gated entirely behind the FAIR_OCEAN_TIMING_LOG env var. Unset (the
default for every normal run, benchmark, and the whole test suite), this
module is a no-op with negligible overhead: one os.environ.get() per
call, no timing, no I/O. Set it to a file path before a run to append one
JSON line per LLM/HTTP call: {"ts", "category", "label", "seconds"}.
"category" is "llm" or "http"; "label" identifies where the time went --
for LLM calls, the calling stage's own module.function (e.g.
"section_category_extraction.categorize_paragraphs"), found by walking
the call stack past this package's own llm/ frames so every call site
gets attributed without having to thread a "purpose" parameter through
generate_json's many real callers; for HTTP calls, the source adapter
name (e.g. "ncbi", "europe_pmc") already available on RateLimitedClient.

Aggregate the log afterward with a one-off script (group by
category+label, sum seconds, sort descending) -- no summarizer is shipped
here since this is meant to be deleted, not maintained.
"""
from __future__ import annotations

import inspect
import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

_ENV_VAR = "FAIR_OCEAN_TIMING_LOG"
_lock = threading.Lock()


@contextmanager
def record(category: str, label: str):
    path = os.environ.get(_ENV_VAR)
    if not path:
        yield
        return
    start = time.monotonic()
    try:
        yield
    finally:
        elapsed = time.monotonic() - start
        line = json.dumps({"ts": time.time(), "category": category, "label": label, "seconds": elapsed})
        with _lock:
            with Path(path).open("a") as f:
                f.write(line + "\n")


def caller_label() -> str:
    """Walks the stack past this package's own llm/ frames (llm/base.py's
    generate_json, llm/http_backend.py's generate) to find whichever real
    extraction-stage function actually asked for a completion -- so every
    call site gets attributed with zero changes to any of them."""
    if not os.environ.get(_ENV_VAR):
        return ""
    for frame_info in inspect.stack()[1:10]:
        module = inspect.getmodule(frame_info.frame)
        modname = module.__name__ if module else ""
        if modname.startswith("fair_ocean_agent.llm") or modname == __name__:
            continue
        short_module = modname.rsplit(".", 1)[-1] if modname else "?"
        return f"{short_module}.{frame_info.function}"
    return "unknown"
