from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True)
class RateLimit:
    requests: int
    window_seconds: int


class InMemoryRateLimiter:
    """Simple fixed-window limiter for low-scale single-process deployments."""

    def __init__(self) -> None:
        self._events: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def check(self, scope: str, identifier: str, limit: RateLimit) -> int | None:
        now = time.time()
        key = (scope, identifier)
        window_start = now - limit.window_seconds

        with self._lock:
            events = self._events[key]
            while events and events[0] <= window_start:
                events.popleft()

            if len(events) >= limit.requests:
                retry_after = max(1, int(events[0] + limit.window_seconds - now))
                if not events:
                    self._events.pop(key, None)
                return retry_after

            events.append(now)
            return None

    def reset(self) -> None:
        with self._lock:
            self._events.clear()


rate_limiter = InMemoryRateLimiter()
