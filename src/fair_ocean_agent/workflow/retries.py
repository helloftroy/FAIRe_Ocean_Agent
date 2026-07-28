"""Backoff calculation for retry_pending tasks."""
from __future__ import annotations


def compute_backoff_seconds(attempt_count: int, base_seconds: int = 30, max_seconds: int = 3600) -> int:
    """Exponential backoff: base * 2^(attempt_count - 1), capped at
    max_seconds. attempt_count is 1 on the first failure."""
    if attempt_count < 1:
        attempt_count = 1
    return min(base_seconds * (2 ** (attempt_count - 1)), max_seconds)
