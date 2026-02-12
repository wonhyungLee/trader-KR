from __future__ import annotations

import random
import time
from typing import Optional


def compute_backoff(attempt: int, base: float, cap: float, jitter: float) -> float:
    if attempt < 1:
        attempt = 1
    delay = min(base * (2 ** (attempt - 1)), cap)
    if jitter > 0:
        delay += random.uniform(0, jitter)
    return delay


def sleep_backoff(attempt: int, base: float, cap: float, jitter: float) -> float:
    delay = compute_backoff(attempt, base, cap, jitter)
    time.sleep(delay)
    return delay


def is_retryable_status(status: Optional[int]) -> bool:
    if status is None:
        return False
    return status in {429, 500, 502, 503, 504}
