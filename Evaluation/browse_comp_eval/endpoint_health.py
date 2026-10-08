"""Model endpoint health monitoring for BrowseComp evaluation runs."""

from __future__ import annotations

import concurrent.futures
import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


TEMPORARY_ENDPOINT_FAILURE_EXIT_CODE = 75


@dataclass(frozen=True)
class EndpointWatchdogConfig:
    enabled: bool = False
    interval_seconds: float = 30.0
    request_timeout_seconds: float = 5.0
    failure_threshold: int = 3
    recovery_success_threshold: int = 2
    all_down_timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        for name in (
            "interval_seconds",
            "request_timeout_seconds",
            "all_down_timeout_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{name} must be a positive number")
        for name in ("failure_threshold", "recovery_success_threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class EndpointRoute:
    index: int
    base_url: str
    model: str
    api_key: str = field(repr=False)


@dataclass(frozen=True)
class ProbeResult:
    ok: bool
    reason: str = ""
    failure_scope: str = "route"


@dataclass
class _RouteState:
    state: str = "unknown"
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    ever_healthy: bool = False
    generation: int = 0
    last_reason: str = ""
    retired: bool = False


class EndpointWatchdogAbort(RuntimeError):
    """Raised when every configured model endpoint stayed down too long."""


class EndpointHealthManager:
    """Probe routes, quarantine failures, and expose scheduling state."""

    def __init__(
        self,
        routes: list[EndpointRoute],
        config: EndpointWatchdogConfig,
        *,
        event_path: Path | None = None,
        probe: Callable[[EndpointRoute, float], ProbeResult] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        on_change: Callable[[], None] | None = None,
        on_unhealthy: Callable[[list[int], str], None] | None = None,
        on_abort: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.event_path = Path(event_path) if event_path is not None else None
        self._probe = probe or probe_openai_models
        self._monotonic = monotonic
        self._on_change = on_change
        self._on_unhealthy = on_unhealthy
        self._on_abort = on_abort
        self._lock = threading.RLock()
        self._event_lock = threading.Lock()
        self._routes: dict[int, EndpointRoute] = {}
        self._states: dict[int, _RouteState] = {}
        for route in routes:
            if route.index in self._routes:
                raise ValueError(f"duplicate endpoint route index: {route.index}")
            self._routes[route.index] = route
            self._states[route.index] = _RouteState()
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._abort_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._all_down_since: float | None = None
        self._abort_reason = ""

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def routes(self) -> tuple[EndpointRoute, ...]:
        with self._lock:
            return tuple(self._routes[index] for index in sorted(self._routes))

    @property
    def aborted(self) -> bool:
        return self._abort_event.is_set()

    @property
    def abort_reason(self) -> str:
        with self._lock:
            return self._abort_reason

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        # Routes may have been hot-added before the watchdog thread starts.
        # The synchronous startup round already probes them, so discard only
        # that stale wakeup.  A route added during the round sets it again.
        self._wake_event.clear()
        self._run_probe_round()
        if self.aborted:
            return
        self._thread = threading.Thread(
            target=self._run_loop,
            name="browsecomp-endpoint-watchdog",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._wake_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, self.config.request_timeout_seconds + 1.0))
        self._thread = None

    def is_available(self, route_index: int) -> bool:
        if not self.enabled:
            with self._lock:
                state = self._states.get(route_index)
                return state is not None and not state.retired
        with self._lock:
            state = self._states.get(route_index)
            return bool(
                state is not None
                and not state.retired
                and state.ever_healthy
                and state.state in {"healthy", "suspect"}
            )

    def generation(self, route_index: int) -> int:
        with self._lock:
            return self._states[route_index].generation

    def add_routes(self, routes: list[EndpointRoute]) -> None:
        if not routes:
            return
        changed = False
        with self._lock:
            for route in routes:
                if route.index in self._routes:
                    raise ValueError(f"duplicate endpoint route index: {route.index}")
                self._routes[route.index] = route
                self._states[route.index] = _RouteState()
                self._write_route_event_locked("endpoint_added", route, "")
                changed = True
        if changed:
            self._wake_event.set()
            if self._on_change is not None:
                self._on_change()

    def retire_routes(self, route_indexes: list[int], reason: str) -> None:
        retired: list[int] = []
        with self._lock:
            for route_index in route_indexes:
                state = self._states.get(route_index)
                route = self._routes.get(route_index)
                if state is None or route is None or state.retired:
                    continue
                previous = state.state
                state.retired = True
                state.state = "retired"
                state.last_reason = reason
                state.generation += 1
                retired.append(route_index)
                self._write_event_locked(route, previous, state)
        if not retired:
            return
        self._wake_event.set()
        if self._on_change is not None:
            self._on_change()
        if self._on_unhealthy is not None:
            self._on_unhealthy(retired, reason)

    def route(self, route_index: int) -> EndpointRoute | None:
        with self._lock:
            return self._routes.get(route_index)

    def record_event(self, event: str, **fields: object) -> None:
        self._append_event(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "event": event,
                **fields,
            }
        )

    def snapshot(self) -> list[dict[str, object]]:
        with self._lock:
            return [
                {
                    "route_index": route.index,
                    "base_url": route.base_url,
                    "model": route.model,
                    "state": self._states[route.index].state,
                    "consecutive_failures": self._states[route.index].consecutive_failures,
                    "consecutive_successes": self._states[route.index].consecutive_successes,
                    "generation": self._states[route.index].generation,
                    "last_reason": self._states[route.index].last_reason,
                    "retired": self._states[route.index].retired,
                }
                for route in self.routes
            ]

    def ensure_not_aborted(self) -> None:
        if self.aborted:
            raise EndpointWatchdogAbort(self.abort_reason or "all model endpoints unavailable")

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            self._wake_event.wait(self.config.interval_seconds)
            self._wake_event.clear()
            if self._stop_event.is_set():
                return
            self._run_probe_round()
            if self.aborted:
                return

    def _run_probe_round(self) -> None:
        if self._stop_event.is_set() or self.aborted:
            return
        with self._lock:
            routes = [
                route
                for route in self._routes.values()
                if not self._states[route.index].retired
            ]
        results: dict[int, ProbeResult] = {}
        if routes:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, len(routes)),
                thread_name_prefix="endpoint-probe",
            ) as executor:
                futures = {
                    executor.submit(
                        self._probe,
                        route,
                        self.config.request_timeout_seconds,
                    ): route
                    for route in routes
                }
                for future, route in ((future, futures[future]) for future in futures):
                    try:
                        results[route.index] = future.result()
                    except Exception as exc:
                        results[route.index] = ProbeResult(
                            False,
                            f"probe raised {exc.__class__.__name__}",
                            "network",
                        )

        # A transport failure belongs to the URL, not to one API-key/model
        # lane.  Fan it out even if another concurrent probe happened to race
        # with a transient recovery.  Authentication and model-catalog errors
        # retain route scope.
        network_failures = {
            route.base_url: results[route.index]
            for route in routes
            if not results[route.index].ok
            and results[route.index].failure_scope == "network"
        }
        for route in routes:
            if route.base_url in network_failures:
                results[route.index] = network_failures[route.base_url]

        newly_unhealthy: list[int] = []
        state_changed = False
        now = self._monotonic()
        with self._lock:
            all_failed = not any(
                result.ok and not self._states[index].retired
                for index, result in results.items()
            )
            for route in routes:
                if self._states[route.index].retired:
                    continue
                result = results[route.index]
                state = self._states[route.index]
                previous = state.state
                if result.ok:
                    state.consecutive_failures = 0
                    state.last_reason = ""
                    if state.state in {"unhealthy", "recovering"}:
                        state.consecutive_successes += 1
                        if state.consecutive_successes >= self.config.recovery_success_threshold:
                            state.state = "healthy"
                            state.ever_healthy = True
                            state.generation += 1
                            state.consecutive_successes = 0
                        else:
                            state.state = "recovering"
                    else:
                        state.state = "healthy"
                        state.ever_healthy = True
                        state.consecutive_successes = 0
                else:
                    state.consecutive_successes = 0
                    state.consecutive_failures += 1
                    state.last_reason = result.reason
                    if previous in {"unhealthy", "recovering"}:
                        state.state = "unhealthy"
                    elif state.consecutive_failures >= self.config.failure_threshold:
                        state.state = "unhealthy"
                        if previous != "unhealthy":
                            newly_unhealthy.append(route.index)
                    else:
                        state.state = "suspect"
                if state.state != previous:
                    state_changed = True
                    self._write_event_locked(route, previous, state)

            if all_failed:
                if self._all_down_since is None:
                    self._all_down_since = now
                    self._write_global_event_locked("all_endpoints_failing", "")
                elapsed = now - self._all_down_since
                if elapsed >= self.config.all_down_timeout_seconds:
                    self._abort_reason = (
                        "all model endpoints failed health checks continuously for "
                        f"{self.config.all_down_timeout_seconds:g}s"
                    )
                    self._abort_event.set()
                    self._write_global_event_locked("all_endpoints_timeout", self._abort_reason)
            elif self._all_down_since is not None:
                self._all_down_since = None
                self._write_global_event_locked("all_endpoints_recovered", "")

        if state_changed and self._on_change is not None:
            self._on_change()
        if newly_unhealthy and self._on_unhealthy is not None:
            reason = "; ".join(
                f"{self._routes[index].base_url}: {results[index].reason}"
                for index in newly_unhealthy
            )
            self._on_unhealthy(newly_unhealthy, reason)
        if self.aborted:
            if self._on_change is not None:
                self._on_change()
            if self._on_abort is not None:
                self._on_abort(self.abort_reason)

    def _write_event_locked(
        self, route: EndpointRoute, previous: str, state: _RouteState
    ) -> None:
        self._append_event(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "event": "endpoint_state_change",
                "route_index": route.index,
                "base_url": route.base_url,
                "model": route.model,
                "previous_state": previous,
                "state": state.state,
                "consecutive_failures": state.consecutive_failures,
                "consecutive_successes": state.consecutive_successes,
                "generation": state.generation,
                "reason": state.last_reason,
            }
        )

    def _write_global_event_locked(self, event: str, reason: str) -> None:
        self._append_event(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "event": event,
                "reason": reason,
            }
        )

    def _write_route_event_locked(
        self, event: str, route: EndpointRoute, reason: str
    ) -> None:
        self._append_event(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "event": event,
                "route_index": route.index,
                "base_url": route.base_url,
                "model": route.model,
                "reason": reason,
            }
        )

    def _append_event(self, payload: dict[str, object]) -> None:
        path = self.event_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._event_lock:
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def probe_openai_models(route: EndpointRoute, timeout: float) -> ProbeResult:
    """Require an OpenAI-compatible models response containing the route model."""

    root = route.base_url.rstrip("/")
    url = root + ("/models" if root.endswith("/v1") else "/v1/models")
    headers = {"Accept": "application/json"}
    if route.api_key:
        headers["Authorization"] = f"Bearer {route.api_key}"
    request = urllib.request.Request(url, headers=headers, method="GET")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            status = int(getattr(response, "status", 200))
            if not 200 <= status < 300:
                return ProbeResult(False, f"GET /models returned HTTP {status}")
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        scope = "network" if exc.code >= 500 else "route"
        return ProbeResult(False, f"GET /models returned HTTP {exc.code}", scope)
    except (OSError, TimeoutError, urllib.error.URLError) as exc:
        return ProbeResult(False, f"GET /models failed: {exc.__class__.__name__}", "network")
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return ProbeResult(False, f"GET /models returned invalid JSON: {exc.__class__.__name__}")

    rows = payload.get("data") if isinstance(payload, dict) else None
    model_ids = {
        row.get("id")
        for row in rows or []
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }
    if route.model not in model_ids:
        return ProbeResult(False, f"configured model {route.model!r} is absent from /models")
    return ProbeResult(True)
