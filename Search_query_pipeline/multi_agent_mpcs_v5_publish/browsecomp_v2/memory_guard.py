from __future__ import annotations

import errno
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional


MIB = 1024 * 1024


@dataclass(frozen=True)
class MemorySnapshot:
    current_bytes: int
    limit_bytes: int
    reclaimable_bytes: int = 0
    source: str = "unknown"
    cgroup_path: Optional[Path] = None

    @property
    def working_bytes(self) -> int:
        return max(0, self.current_bytes - min(self.current_bytes, self.reclaimable_bytes))


class MemoryPressureTimeout(RuntimeError):
    pass


class MemoryGuard:
    """Serialize memory admission decisions for resource-heavy agent calls.

    Configured workflow concurrency remains unchanged. The guard only delays a
    new call when its projected memory footprint would leave too little cgroup
    headroom. Active reservations close the race where many threads observe the
    same low usage and launch simultaneously.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_usage_ratio: float = 0.85,
        hard_usage_ratio: float = 0.97,
        min_available_bytes: int = 6 * 1024 * MIB,
        reservation_bytes: int = 1024 * MIB,
        poll_seconds: float = 5.0,
        start_interval_seconds: float = 1.0,
        log_interval_seconds: float = 30.0,
        max_wait_seconds: float = 300.0,
        reclaim_enabled: bool = True,
        reclaim_target_ratio: float = 0.75,
        reclaim_min_bytes: int = 8 * 1024 * MIB,
        reclaim_max_bytes: int = 24 * 1024 * MIB,
        reclaim_cooldown_seconds: float = 300.0,
        reclaim_swappiness: int = 0,
        snapshot_reader: Optional[Callable[[], Optional[MemorySnapshot]]] = None,
        reclaim_writer: Optional[Callable[[Path, int, int], None]] = None,
        logger: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.enabled = enabled
        self.max_usage_ratio = min(0.99, max(0.01, max_usage_ratio))
        self.hard_usage_ratio = min(1.0, max(self.max_usage_ratio, hard_usage_ratio))
        self.min_available_bytes = max(0, min_available_bytes)
        self.reservation_bytes = max(0, reservation_bytes)
        self.poll_seconds = max(0.05, poll_seconds)
        self.start_interval_seconds = max(0.0, start_interval_seconds)
        self.log_interval_seconds = max(0.1, log_interval_seconds)
        self.max_wait_seconds = max(0.0, max_wait_seconds)
        self.reclaim_enabled = reclaim_enabled
        self.reclaim_target_ratio = min(
            max(0.01, self.hard_usage_ratio - 0.01),
            max(0.10, reclaim_target_ratio),
        )
        self.reclaim_min_bytes = max(0, reclaim_min_bytes)
        self.reclaim_max_bytes = max(self.reclaim_min_bytes, reclaim_max_bytes)
        self.reclaim_cooldown_seconds = max(0.0, reclaim_cooldown_seconds)
        self.reclaim_swappiness = min(200, max(0, reclaim_swappiness))
        self._snapshot_reader = snapshot_reader or read_memory_snapshot
        self._reclaim_writer = reclaim_writer or _write_cgroup_reclaim
        self._logger = logger or (lambda message: print(message, flush=True))
        self._condition = threading.Condition()
        self._active = 0
        self._reserved_bytes = 0
        self._last_admission = 0.0
        self._last_reclaim = float("-inf")
        self._reclaim_unavailable = False

    @classmethod
    def from_env(cls) -> "MemoryGuard":
        return cls(
            enabled=_bool_env("V2_MEMORY_GUARD_ENABLED", True),
            max_usage_ratio=_float_env("V2_MEMORY_GUARD_MAX_USAGE_RATIO", 0.85),
            hard_usage_ratio=_float_env("V2_MEMORY_GUARD_HARD_USAGE_RATIO", 0.97),
            min_available_bytes=_int_env("V2_MEMORY_GUARD_MIN_AVAILABLE_MB", 6144) * MIB,
            reservation_bytes=_int_env("V2_MEMORY_GUARD_RESERVATION_MB", 1024) * MIB,
            poll_seconds=_float_env("V2_MEMORY_GUARD_POLL_SECONDS", 5.0),
            start_interval_seconds=_float_env("V2_MEMORY_GUARD_START_INTERVAL_SECONDS", 1.0),
            log_interval_seconds=_float_env("V2_MEMORY_GUARD_LOG_INTERVAL_SECONDS", 30.0),
            max_wait_seconds=_float_env("V2_MEMORY_GUARD_MAX_WAIT_SECONDS", 300.0),
            reclaim_enabled=_bool_env("V2_MEMORY_RECLAIM_ENABLED", True),
            reclaim_target_ratio=_float_env("V2_MEMORY_RECLAIM_TARGET_RATIO", 0.75),
            reclaim_min_bytes=_int_env("V2_MEMORY_RECLAIM_MIN_MB", 8192) * MIB,
            reclaim_max_bytes=_int_env("V2_MEMORY_RECLAIM_MAX_MB", 24576) * MIB,
            reclaim_cooldown_seconds=_float_env("V2_MEMORY_RECLAIM_COOLDOWN_SECONDS", 300.0),
            reclaim_swappiness=_int_env("V2_MEMORY_RECLAIM_SWAPPINESS", 0),
        )

    @property
    def active(self) -> int:
        with self._condition:
            return self._active

    @contextmanager
    def reserve(self, label: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        reservation_bytes = self._reservation_bytes(label)
        self.acquire(label, reservation_bytes=reservation_bytes)
        try:
            yield
        finally:
            self.release(reservation_bytes=reservation_bytes)

    def acquire(
        self,
        label: str,
        *,
        reservation_bytes: Optional[int] = None,
    ) -> None:
        wait_started = time.monotonic()
        last_log = 0.0
        pressure_waited = False
        last_snapshot: Optional[MemorySnapshot] = None
        last_reason = ""
        reservation_bytes = (
            self._reservation_bytes(label)
            if reservation_bytes is None
            else max(0, reservation_bytes)
        )

        with self._condition:
            while True:
                now = time.monotonic()
                snapshot = self._snapshot_reader()
                reason = self._risk_reason(
                    snapshot,
                    reservation_bytes=reservation_bytes,
                )
                if reason == "hard_usage_ratio" and snapshot is not None:
                    requested = self._try_reclaim(
                        label,
                        snapshot,
                        now,
                        reservation_bytes=reservation_bytes,
                    )
                    if requested:
                        before = snapshot
                        refreshed = self._snapshot_reader()
                        if refreshed is not None:
                            snapshot = refreshed
                        now = time.monotonic()
                        reason = self._risk_reason(
                            snapshot,
                            reservation_bytes=reservation_bytes,
                        )
                        self._logger(
                            "memory_guard_reclaim_done: "
                            f"agent={label} requested_mb={requested // MIB} "
                            f"released_mb={max(0, before.current_bytes - snapshot.current_bytes) // MIB} "
                            f"{self._format_snapshot(snapshot)}"
                        )
                    if reason == "hard_usage_ratio":
                        reason = self._working_risk_reason(
                            snapshot,
                            reservation_bytes=reservation_bytes,
                        )
                        if reason is None:
                            self._logger(
                                "memory_guard_cache_admit: "
                                f"agent={label} active={self._active} "
                                f"reclaim_requested_mb={requested // MIB} "
                                f"{self._format_snapshot(snapshot)}"
                            )
                interval_left = max(
                    0.0,
                    self.start_interval_seconds - (now - self._last_admission),
                )
                if reason is None and interval_left <= 0:
                    self._active += 1
                    self._reserved_bytes += reservation_bytes
                    self._last_admission = now
                    if pressure_waited:
                        self._logger(
                            "memory_guard_resume: "
                            f"agent={label} waited={now - wait_started:.1f}s "
                            f"active={self._active} {self._format_snapshot(snapshot)}"
                        )
                    return

                last_snapshot = snapshot
                last_reason = reason or "start_interval"
                pressure_waited = pressure_waited or reason is not None
                elapsed = now - wait_started
                if self.max_wait_seconds and elapsed >= self.max_wait_seconds:
                    raise MemoryPressureTimeout(
                        "memory guard wait timed out: "
                        f"agent={label} reason={last_reason} waited={elapsed:.1f}s "
                        f"{self._format_snapshot(last_snapshot)}"
                    )
                if reason is not None and (
                    last_log == 0.0 or now - last_log >= self.log_interval_seconds
                ):
                    self._logger(
                        "memory_guard_wait: "
                        f"agent={label} reason={last_reason} waited={elapsed:.1f}s "
                        f"active={self._active} {self._format_snapshot(snapshot)}"
                    )
                    last_log = now

                wait_for = self.poll_seconds
                if reason is None and interval_left > 0:
                    wait_for = min(wait_for, interval_left)
                if self.max_wait_seconds:
                    wait_for = min(wait_for, max(0.05, self.max_wait_seconds - elapsed))
                self._condition.wait(timeout=max(0.05, wait_for))

    def release(self, *, reservation_bytes: Optional[int] = None) -> None:
        if not self.enabled:
            return
        with self._condition:
            self._active = max(0, self._active - 1)
            reservation = (
                self.reservation_bytes
                if reservation_bytes is None
                else max(0, reservation_bytes)
            )
            self._reserved_bytes = max(0, self._reserved_bytes - reservation)
            self._condition.notify_all()

    def _reservation_bytes(self, label: str) -> int:
        normalized = "".join(
            char if char.isalnum() else "_" for char in str(label).upper()
        ).strip("_")
        raw = os.environ.get(
            f"V2_MEMORY_GUARD_RESERVATION_MB_{normalized}",
            "",
        ).strip()
        if not raw:
            return self.reservation_bytes
        try:
            return max(0, int(raw)) * MIB
        except ValueError:
            return self.reservation_bytes

    def _risk_reason(
        self,
        snapshot: Optional[MemorySnapshot],
        *,
        reservation_bytes: Optional[int] = None,
    ) -> Optional[str]:
        if snapshot is None or snapshot.limit_bytes <= 0:
            return None

        limit = snapshot.limit_bytes
        reservation = (
            self.reservation_bytes
            if reservation_bytes is None
            else max(0, reservation_bytes)
        )
        # current_bytes already includes memory used by active calls. Add all
        # reservations only to the reclaimable-adjusted working set so a burst
        # cannot outrun cgroup accounting. The raw hard check triggers
        # proactive cache reclaim. After that attempt, the working-set checks
        # decide whether cache-dominated usage is safe to admit.
        projected_raw = snapshot.current_bytes + reservation

        if projected_raw > limit * self.hard_usage_ratio:
            return "hard_usage_ratio"
        return self._working_risk_reason(
            snapshot,
            reservation_bytes=reservation,
        )

    def _working_risk_reason(
        self,
        snapshot: MemorySnapshot,
        *,
        reservation_bytes: Optional[int] = None,
    ) -> Optional[str]:
        limit = snapshot.limit_bytes
        reservation = (
            self.reservation_bytes
            if reservation_bytes is None
            else max(0, reservation_bytes)
        )
        projected_working = snapshot.working_bytes + self._reserved_bytes + reservation
        available = limit - projected_working

        if available < self.min_available_bytes:
            return "minimum_headroom"
        if projected_working > limit * self.max_usage_ratio:
            return "working_set_ratio"
        return None

    def _try_reclaim(
        self,
        label: str,
        snapshot: MemorySnapshot,
        now: float,
        *,
        reservation_bytes: Optional[int] = None,
    ) -> int:
        if (
            not self.reclaim_enabled
            or self._reclaim_unavailable
            or self._active != 0
            or snapshot.source != "cgroup_v2"
            or snapshot.cgroup_path is None
            or now - self._last_reclaim < self.reclaim_cooldown_seconds
        ):
            return 0

        reservation = (
            self.reservation_bytes
            if reservation_bytes is None
            else max(0, reservation_bytes)
        )
        projected_working = snapshot.working_bytes + reservation
        if (
            projected_working > snapshot.limit_bytes * self.max_usage_ratio
            or snapshot.limit_bytes - projected_working < self.min_available_bytes
            or snapshot.reclaimable_bytes < self.reclaim_min_bytes
        ):
            return 0

        target_current = int(snapshot.limit_bytes * self.reclaim_target_ratio)
        projected_raw = snapshot.current_bytes + reservation
        requested = max(self.reclaim_min_bytes, projected_raw - target_current)
        requested = min(
            requested,
            self.reclaim_max_bytes,
            snapshot.reclaimable_bytes,
        )
        if requested <= 0:
            return 0

        self._last_reclaim = now
        self._logger(
            "memory_guard_reclaim_start: "
            f"agent={label} requested_mb={requested // MIB} active={self._active} "
            f"target_ratio={self.reclaim_target_ratio:.2f} "
            f"{self._format_snapshot(snapshot)}"
        )
        try:
            self._reclaim_writer(
                snapshot.cgroup_path / "memory.reclaim",
                requested,
                self.reclaim_swappiness,
            )
        except OSError as exc:
            if exc.errno == errno.EAGAIN:
                self._logger(
                    "memory_guard_reclaim_partial: "
                    f"agent={label} requested_mb={requested // MIB} error={exc}"
                )
                return requested
            if exc.errno in {errno.EACCES, errno.EPERM, errno.EROFS, errno.ENOENT}:
                self._reclaim_unavailable = True
            self._logger(
                "memory_guard_reclaim_failed: "
                f"agent={label} requested_mb={requested // MIB} error={exc}"
            )
            return 0
        return requested

    @staticmethod
    def _format_snapshot(snapshot: Optional[MemorySnapshot]) -> str:
        if snapshot is None:
            return "memory=unavailable"
        return (
            f"current_mb={snapshot.current_bytes // MIB} "
            f"working_mb={snapshot.working_bytes // MIB} "
            f"reclaimable_mb={snapshot.reclaimable_bytes // MIB} "
            f"limit_mb={snapshot.limit_bytes // MIB} source={snapshot.source}"
            + (f" cgroup={snapshot.cgroup_path}" if snapshot.cgroup_path else "")
        )


def read_memory_snapshot() -> Optional[MemorySnapshot]:
    cgroup = _read_cgroup_v2_snapshot()
    if cgroup is not None:
        return cgroup
    return _read_proc_meminfo_snapshot()


def _read_cgroup_v2_snapshot() -> Optional[MemorySnapshot]:
    root = _cgroup_v2_root()
    if root is None:
        return None
    current = _read_int(root / "memory.current")
    maximum = _read_limit(root / "memory.max")
    high = _read_limit(root / "memory.high")
    if current is None or maximum is None:
        return None
    if high is not None:
        maximum = min(maximum, high)
    stat = _read_key_values(root / "memory.stat")
    reclaimable = max(0, stat.get("inactive_file", 0)) + max(
        0, stat.get("slab_reclaimable", 0)
    )
    return MemorySnapshot(
        current_bytes=current,
        limit_bytes=maximum,
        reclaimable_bytes=min(current, reclaimable),
        source="cgroup_v2",
        cgroup_path=root,
    )


def _cgroup_v2_root() -> Optional[Path]:
    override = os.environ.get("V2_MEMORY_GUARD_CGROUP_PATH", "").strip()
    if override:
        candidate = Path(override)
        return candidate if (candidate / "memory.current").exists() else None

    mount = Path("/sys/fs/cgroup")
    relative = ""
    try:
        for line in Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines():
            if line.startswith("0::"):
                relative = line.split("::", 1)[1].lstrip("/")
                break
    except OSError:
        pass
    candidate = mount / relative if relative else mount
    if (candidate / "memory.current").exists():
        return candidate
    return mount if (mount / "memory.current").exists() else None


def _read_proc_meminfo_snapshot() -> Optional[MemorySnapshot]:
    values = _read_key_values(Path("/proc/meminfo"), multiplier=1024)
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if not total or available is None:
        return None
    return MemorySnapshot(
        current_bytes=max(0, total - available),
        limit_bytes=total,
        source="proc_meminfo",
    )


def _read_int(path: Path) -> Optional[int]:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _read_limit(path: Path) -> Optional[int]:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if raw == "max":
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    # cgroup v1 and some runtimes expose an effectively unlimited huge value.
    return value if 0 < value < (1 << 60) else None


def _read_key_values(path: Path, *, multiplier: int = 1) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            values[parts[0].rstrip(":")] = int(parts[1]) * multiplier
        except ValueError:
            continue
    return values


def _write_cgroup_reclaim(path: Path, amount_bytes: int, swappiness: int) -> None:
    try:
        path.write_text(
            f"{amount_bytes} swappiness={swappiness}\n",
            encoding="ascii",
        )
    except OSError as exc:
        # The swappiness key was added after memory.reclaim itself. Older
        # cgroup v2 kernels can still reclaim file cache with the byte-only
        # form, which is the behavior needed here.
        if exc.errno != errno.EINVAL:
            raise
        path.write_text(f"{amount_bytes}\n", encoding="ascii")


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.lower() in {"1", "true", "yes", "y", "on"}


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
