from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True)
class RateLimit:
    requests: int
    window_seconds: int


# Maximum number of distinct (scope, identifier) keys to track.
# Prevents unbounded memory growth from a large number of unique clients.
_MAX_TRACKED_KEYS = 10_000

# How often (in seconds) to run full eviction of stale keys.
_EVICTION_INTERVAL_SECONDS = 300


class InMemoryRateLimiter:
    """Simple fixed-window limiter for low-scale single-process deployments.

    Includes TTL-based key eviction to prevent unbounded memory growth
    from a large number of unique client identifiers.
    """

    def __init__(self) -> None:
        self._events: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._lock = Lock()
        self._last_eviction: float = time.time()

    def check(self, scope: str, identifier: str, limit: RateLimit) -> int | None:
        now = time.time()
        key = (scope, identifier)
        window_start = now - limit.window_seconds

        with self._lock:
            # Periodic eviction of stale keys to prevent unbounded memory growth
            if now - self._last_eviction > _EVICTION_INTERVAL_SECONDS:
                self._evict_stale_keys(now, limit.window_seconds)

            events = self._events[key]
            while events and events[0] <= window_start:
                events.popleft()

            if len(events) >= limit.requests:
                retry_after = max(1, int(events[0] + limit.window_seconds - now))
                if not events:
                    self._events.pop(key, None)
                return retry_after

            events.append(now)

            # Hard cap: if too many keys are tracked, evict the oldest
            if len(self._events) > _MAX_TRACKED_KEYS:
                self._evict_stale_keys(now, limit.window_seconds)

            return None

    def _evict_stale_keys(self, now: float, default_window: int) -> None:
        """Remove keys whose most recent event is older than the window."""
        stale_keys = [
            key for key, events in self._events.items()
            if not events or events[-1] < now - default_window
        ]
        for key in stale_keys:
            del self._events[key]
        self._last_eviction = now

    def reset(self) -> None:
        with self._lock:
            self._events.clear()
            self._last_eviction = time.time()


rate_limiter = InMemoryRateLimiter()
