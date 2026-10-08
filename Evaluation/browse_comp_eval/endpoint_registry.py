"""Shared endpoint registry primitives for Hermes BrowseComp evaluations."""

from __future__ import annotations

import concurrent.futures
import fcntl
import json
import os
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator
from urllib.parse import urlsplit

from .endpoint_health import EndpointRoute, ProbeResult, probe_openai_models


REGISTRY_VERSION = 1
REGISTRY_STATUSES = {"unknown", "healthy", "unhealthy"}
HEALTH_DEFAULTS: dict[str, object] = {
    "status": "unknown",
    "last_checked_at": None,
    "last_healthy_at": None,
    "unreachable_since": None,
    "last_error": None,
}


class EndpointRegistryError(ValueError):
    """Raised when a registry cannot be read or does not match registry v1."""


@dataclass(frozen=True)
class RegistryCheckStats:
    checked: int
    healthy: int
    unhealthy: int
    evicted: int


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def format_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def parse_timestamp(value: object, *, field: str) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise EndpointRegistryError(f"{field} must be an ISO-8601 string or null")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EndpointRegistryError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise EndpointRegistryError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def normalize_base_url(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EndpointRegistryError("endpoint base_url must be a non-empty string")
    normalized = value.strip().rstrip("/")
    parsed = urlsplit(normalized)
    try:
        parsed_port = parsed.port
    except ValueError as exc:
        raise EndpointRegistryError(
            "invalid OpenAI-compatible endpoint base_url"
        ) from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or any(character.isspace() for character in normalized)
        or parsed.username is not None
        or parsed.password is not None
        or (parsed_port is not None and not 1 <= parsed_port <= 65535)
        or parsed.query
        or parsed.fragment
    ):
        raise EndpointRegistryError("invalid OpenAI-compatible endpoint base_url")
    return normalized


def empty_registry(*, now: datetime | None = None) -> dict[str, object]:
    return {
        "version": REGISTRY_VERSION,
        "updated_at": format_timestamp(now or utc_now()),
        "models": {},
    }


def normalize_registry(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise EndpointRegistryError("registry root must be a JSON object")
    if payload.get("version") != REGISTRY_VERSION:
        raise EndpointRegistryError(
            f"registry version must be {REGISTRY_VERSION}"
        )
    models = payload.get("models")
    if not isinstance(models, dict):
        raise EndpointRegistryError("registry models must be a JSON object")

    normalized_models: dict[str, list[dict[str, object]]] = {}
    for model, raw_entries in models.items():
        if not isinstance(model, str) or not model:
            raise EndpointRegistryError("registry model names must be non-empty strings")
        if not isinstance(raw_entries, list):
            raise EndpointRegistryError(f"models[{model!r}] must be a JSON array")
        entries: list[dict[str, object]] = []
        seen_urls: set[str] = set()
        for position, raw_entry in enumerate(raw_entries):
            if not isinstance(raw_entry, dict):
                raise EndpointRegistryError(
                    f"models[{model!r}][{position}] must be a JSON object"
                )
            entry = dict(raw_entry)
            base_url = normalize_base_url(entry.get("base_url"))
            if base_url in seen_urls:
                raise EndpointRegistryError(
                    f"duplicate base_url for model {model!r}: {base_url}"
                )
            seen_urls.add(base_url)
            entry["base_url"] = base_url
            for field, default in HEALTH_DEFAULTS.items():
                entry.setdefault(field, default)
            status = entry["status"]
            if status not in REGISTRY_STATUSES:
                raise EndpointRegistryError(
                    f"models[{model!r}][{position}].status must be one of "
                    + ", ".join(sorted(REGISTRY_STATUSES))
                )
            for field in ("last_checked_at", "last_healthy_at", "unreachable_since"):
                parse_timestamp(
                    entry[field],
                    field=f"models[{model!r}][{position}].{field}",
                )
            if entry["last_error"] is not None and not isinstance(
                entry["last_error"], str
            ):
                raise EndpointRegistryError(
                    f"models[{model!r}][{position}].last_error must be a string or null"
                )
            entries.append(entry)
        normalized_models[model] = entries

    updated_at = payload.get("updated_at")
    if updated_at is not None:
        parse_timestamp(updated_at, field="updated_at")
    return {
        **payload,
        "version": REGISTRY_VERSION,
        "updated_at": updated_at,
        "models": normalized_models,
    }


def load_registry(path: Path | str) -> dict[str, object]:
    registry_path = Path(path)
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EndpointRegistryError(f"registry not found: {registry_path}") from exc
    except OSError as exc:
        raise EndpointRegistryError(
            f"could not read registry {registry_path}: {exc.__class__.__name__}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise EndpointRegistryError(
            f"registry is not valid JSON: {registry_path}: line {exc.lineno}"
        ) from exc
    return normalize_registry(payload)


def model_entries(registry: dict[str, object], model: str) -> list[dict[str, object]]:
    models = registry["models"]
    assert isinstance(models, dict)
    entries = models.get(model, [])
    assert isinstance(entries, list)
    return [dict(entry) for entry in entries]


def model_base_urls(registry: dict[str, object], model: str) -> list[str]:
    return [str(entry["base_url"]) for entry in model_entries(registry, model)]


@contextmanager
def registry_lock(path: Path | str) -> Iterator[None]:
    registry_path = Path(path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = registry_path.with_name(registry_path.name + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)


def atomic_write_registry(path: Path | str, registry: dict[str, object]) -> None:
    registry_path = Path(path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_registry(registry)
    try:
        target_mode = stat.S_IMODE(registry_path.stat().st_mode)
    except FileNotFoundError:
        target_mode = 0o644
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{registry_path.name}.",
        suffix=".tmp",
        dir=str(registry_path.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(fd, target_mode)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(normalized, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, registry_path)
        try:
            directory_fd = os.open(registry_path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = -1
        if directory_fd >= 0:
            try:
                try:
                    os.fsync(directory_fd)
                except OSError:
                    # Some network filesystems do not support fsync on a
                    # directory descriptor.  The file itself was fsynced
                    # before os.replace, so keep the update usable there.
                    pass
            finally:
                os.close(directory_fd)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def add_endpoint(
    path: Path | str,
    model: str,
    base_url: str,
    *,
    now: Callable[[], datetime] = utc_now,
) -> bool:
    if not isinstance(model, str) or not model:
        raise EndpointRegistryError("model must be a non-empty string")
    normalized_url = normalize_base_url(base_url)
    registry_path = Path(path)
    with registry_lock(registry_path):
        if registry_path.exists():
            registry = load_registry(registry_path)
        else:
            registry = empty_registry(now=now())
        models = registry["models"]
        assert isinstance(models, dict)
        entries = models.setdefault(model, [])
        assert isinstance(entries, list)
        if any(entry.get("base_url") == normalized_url for entry in entries):
            return False
        entries.append({"base_url": normalized_url, **HEALTH_DEFAULTS})
        registry["updated_at"] = format_timestamp(now())
        atomic_write_registry(registry_path, registry)
    return True


def registry_events_path(path: Path | str) -> Path:
    registry_path = Path(path)
    return registry_path.with_name(f"{registry_path.stem}.events.jsonl")


def _append_event(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def check_registry_once(
    path: Path | str,
    *,
    api_key: str,
    timeout_seconds: float = 5.0,
    evict_after_seconds: float = 1800.0,
    probe: Callable[[EndpointRoute, float], ProbeResult] = probe_openai_models,
    now: Callable[[], datetime] = utc_now,
) -> RegistryCheckStats:
    if timeout_seconds <= 0:
        raise EndpointRegistryError("timeout_seconds must be positive")
    if evict_after_seconds <= 0:
        raise EndpointRegistryError("evict_after_seconds must be positive")
    registry_path = Path(path)

    # Probes happen without the file lock.  A concurrent add therefore never
    # waits on a slow endpoint and is merged into the latest snapshot below.
    with registry_lock(registry_path):
        snapshot = load_registry(registry_path)
    routes: list[EndpointRoute] = []
    route_keys: dict[int, tuple[str, str]] = {}
    models = snapshot["models"]
    assert isinstance(models, dict)
    for model, entries in models.items():
        assert isinstance(model, str) and isinstance(entries, list)
        for entry in entries:
            assert isinstance(entry, dict)
            index = len(routes)
            routes.append(
                EndpointRoute(
                    index=index,
                    base_url=str(entry["base_url"]),
                    model=model,
                    api_key=api_key,
                )
            )
            route_keys[index] = (model, str(entry["base_url"]))

    results: dict[tuple[str, str], ProbeResult] = {}
    if routes:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(32, len(routes)),
            thread_name_prefix="registry-probe",
        ) as executor:
            futures = {
                executor.submit(probe, route, timeout_seconds): route for route in routes
            }
            for future, route in ((future, futures[future]) for future in futures):
                try:
                    result = future.result()
                except Exception as exc:
                    result = ProbeResult(
                        False,
                        f"probe raised {exc.__class__.__name__}",
                        "network",
                    )
                results[route_keys[route.index]] = result

    checked_at = now().astimezone(timezone.utc)
    checked_at_text = format_timestamp(checked_at)
    healthy = 0
    unhealthy = 0
    evicted = 0
    with registry_lock(registry_path):
        registry = load_registry(registry_path)
        current_models = registry["models"]
        assert isinstance(current_models, dict)
        for model in list(current_models):
            current_entries = current_models[model]
            assert isinstance(current_entries, list)
            retained: list[dict[str, object]] = []
            for entry in current_entries:
                assert isinstance(entry, dict)
                key = (model, str(entry["base_url"]))
                result = results.get(key)
                if result is None:
                    # Added after the probe snapshot: leave it unknown for the
                    # next check instead of fabricating a failed observation.
                    retained.append(entry)
                    continue
                entry["last_checked_at"] = checked_at_text
                if result.ok:
                    healthy += 1
                    entry["status"] = "healthy"
                    entry["last_healthy_at"] = checked_at_text
                    entry["unreachable_since"] = None
                    entry["last_error"] = None
                    retained.append(entry)
                    continue

                unhealthy += 1
                entry["status"] = "unhealthy"
                if entry.get("unreachable_since") is None:
                    entry["unreachable_since"] = checked_at_text
                entry["last_error"] = result.reason or "health probe failed"
                since = parse_timestamp(
                    entry.get("unreachable_since"), field="unreachable_since"
                )
                assert since is not None
                if (checked_at - since).total_seconds() >= evict_after_seconds:
                    evicted += 1
                    _append_event(
                        registry_events_path(registry_path),
                        {
                            "timestamp": checked_at_text,
                            "event": "endpoint_evicted",
                            "model": model,
                            "base_url": entry["base_url"],
                            "unreachable_since": entry["unreachable_since"],
                            "last_error": entry["last_error"],
                            "evict_after_seconds": evict_after_seconds,
                        },
                    )
                else:
                    retained.append(entry)
            current_models[model] = retained
        registry["updated_at"] = checked_at_text
        atomic_write_registry(registry_path, registry)

    return RegistryCheckStats(
        checked=len(results),
        healthy=healthy,
        unhealthy=unhealthy,
        evicted=evicted,
    )
