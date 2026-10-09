from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, Optional


@dataclass
class _RateLimitBucket:
    max_concurrency: int
    concurrency: int
    active: int = 0
    blocked_until: float = 0.0
    consecutive_limits: int = 0
    successful_calls: int = 0


class RateLimitGuard:
    """Adapt concurrency for an independently keyed agent/scope bucket."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        min_concurrency: int = 1,
        cooldown_seconds: float = 30.0,
        max_cooldown_seconds: float = 300.0,
        recovery_successes: int = 20,
        logger: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.enabled = enabled
        self.min_concurrency = max(1, min_concurrency)
        self.cooldown_seconds = max(0.0, cooldown_seconds)
        self.max_cooldown_seconds = max(self.cooldown_seconds, max_cooldown_seconds)
        self.recovery_successes = max(1, recovery_successes)
        self._logger = logger or (lambda message: print(message, flush=True))
        self._condition = threading.Condition()
        self._buckets: Dict[str, _RateLimitBucket] = {}

    @classmethod
    def from_env(cls) -> "RateLimitGuard":
        return cls(
            enabled=_bool_env("V2_RATE_LIMIT_GUARD_ENABLED", True),
            min_concurrency=_int_env("V2_RATE_LIMIT_MIN_CONCURRENCY", 1),
            cooldown_seconds=_float_env("V2_RATE_LIMIT_COOLDOWN_SECONDS", 30.0),
            max_cooldown_seconds=_float_env("V2_RATE_LIMIT_MAX_COOLDOWN_SECONDS", 300.0),
            recovery_successes=_int_env("V2_RATE_LIMIT_RECOVERY_SUCCESSES", 20),
        )

    @contextmanager
    def reserve(self, label: str, max_concurrency: int) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        self.acquire(label, max_concurrency)
        try:
            yield
        finally:
            self.release(label)

    def acquire(self, label: str, max_concurrency: int) -> None:
        with self._condition:
            bucket = self._bucket(label, max_concurrency)
            while True:
                now = time.monotonic()
                cooldown_left = max(0.0, bucket.blocked_until - now)
                if cooldown_left <= 0 and bucket.active < bucket.concurrency:
                    bucket.active += 1
                    return
                wait_seconds = cooldown_left if cooldown_left > 0 else 0.5
                self._condition.wait(timeout=max(0.05, min(wait_seconds, 1.0)))

    def release(self, label: str) -> None:
        if not self.enabled:
            return
        with self._condition:
            bucket = self._buckets.get(label)
            if bucket is not None:
                bucket.active = max(0, bucket.active - 1)
            self._condition.notify_all()

    def rate_limited(self, label: str, max_concurrency: int) -> None:
        if not self.enabled:
            return
        with self._condition:
            bucket = self._bucket(label, max_concurrency)
            previous = bucket.concurrency
            bucket.concurrency = max(self.min_concurrency, bucket.concurrency // 2)
            bucket.consecutive_limits += 1
            bucket.successful_calls = 0
            multiplier = 2 ** min(bucket.consecutive_limits - 1, 8)
            cooldown = min(self.cooldown_seconds * multiplier, self.max_cooldown_seconds)
            bucket.blocked_until = max(bucket.blocked_until, time.monotonic() + cooldown)
            self._logger(
                "rate_limit_guard_throttle: "
                f"agent={label} concurrency={previous}->{bucket.concurrency} "
                f"cooldown={cooldown:.1f}s"
            )
            self._condition.notify_all()

    def succeeded(self, label: str, max_concurrency: int) -> None:
        if not self.enabled:
            return
        with self._condition:
            bucket = self._bucket(label, max_concurrency)
            if bucket.concurrency >= bucket.max_concurrency:
                bucket.consecutive_limits = 0
                bucket.successful_calls = 0
                return
            bucket.successful_calls += 1
            if bucket.successful_calls < self.recovery_successes:
                return
            previous = bucket.concurrency
            bucket.concurrency = min(bucket.max_concurrency, bucket.concurrency + 1)
            bucket.consecutive_limits = max(0, bucket.consecutive_limits - 1)
            bucket.successful_calls = 0
            self._logger(
                "rate_limit_guard_recover: "
                f"agent={label} concurrency={previous}->{bucket.concurrency}"
            )
            self._condition.notify_all()

    def current_limit(self, label: str, max_concurrency: int) -> int:
        with self._condition:
            return self._bucket(label, max_concurrency).concurrency

    def _bucket(self, label: str, max_concurrency: int) -> _RateLimitBucket:
        maximum = max(self.min_concurrency, max(1, max_concurrency))
        bucket = self._buckets.get(label)
        if bucket is None:
            bucket = _RateLimitBucket(max_concurrency=maximum, concurrency=maximum)
            self._buckets[label] = bucket
        return bucket


def _bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default
