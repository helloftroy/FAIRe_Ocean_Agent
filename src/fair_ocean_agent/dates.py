"""Shared date-parsing helper. Used by both llm/benchmark.py (matching a
model's date-formatted fact against gold) and validation/logical.py
(checking date ordering) -- factored out rather than duplicated once a
second caller needed the exact same "parse loosely, but deterministically"
behavior.
"""
from __future__ import annotations

from datetime import date, datetime

from dateutil import parser as date_parser

# A fixed, non-current-date anchor for partial-date parsing (e.g. "March
# 2021" has no day). dateutil's actual default is today's date, which would
# make parsing depend on what day the code happens to run -- non-
# deterministic and wrong for both matching (llm/benchmark.py) and
# validation (validation/logical.py). With this anchor, "March 2021"
# always resolves to year-month-01, deterministically.
ANCHOR = datetime(1, 1, 1)


def try_parse_date(value: str) -> date | None:
    """Strict parsing (fuzzy=False): rejects non-date strings (kit names,
    primer sequences, instrument names) rather than guessing. Returns None,
    never raises."""
    try:
        return date_parser.parse(value, fuzzy=False, default=ANCHOR).date()
    except (ValueError, OverflowError, TypeError):
        return None
