"""Synchronous E2B/Tencent AGS backend for BrowseComp.

The E2B dependency is optional.  Importing this module keeps the cloud SDK
out of the normal BrowseComp install; the SDK is resolved only when a backend
without an injected sandbox class creates a session.
"""

from __future__ import annotations

import math
import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable

from .ags_worker import (
    AGSBackend,
    AGSCommandResult,
    AGSCreateSpec,
    AGSSandboxSession,
)


_HOSTNAME_LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
_HOSTNAME_PATTERN = re.compile(rf"{_HOSTNAME_LABEL}(?:\.{_HOSTNAME_LABEL})*")
_MISSING_E2B_MESSAGE = (
    "E2B AGS support requires the optional e2b dependency; install it with "
    "pip install 'browse-comp-eval[ags]'"
)


class E2BAGSOperationError(RuntimeError):
    """An AGS SDK operation failed without exposing the backend credential."""


@dataclass(frozen=True)
class E2BAGSBackendConfig:
    """Credentials, endpoint, and request timeouts for one AGS backend."""

    api_key: str = field(repr=False)
    domain: str
    create_request_timeout: int | float = 120
    operation_request_timeout: int | float = 60
    kill_request_timeout: int | float = 30

    def __post_init__(self) -> None:
        if not isinstance(self.api_key, str):
            raise TypeError("api_key must be a string")
        if not self.api_key or not self.api_key.strip():
            raise ValueError("api_key must be a non-empty string")

        if not isinstance(self.domain, str):
            raise TypeError("domain must be a string")
        if (
            not self.domain
            or self.domain != self.domain.strip()
            or "://" in self.domain
            or not _HOSTNAME_PATTERN.fullmatch(self.domain)
        ):
            raise ValueError("domain must be a non-empty bare hostname")

        _require_positive_finite_number(
            "create_request_timeout", self.create_request_timeout
        )
        _require_positive_finite_number(
            "operation_request_timeout", self.operation_request_timeout
        )
        _require_positive_finite_number(
            "kill_request_timeout", self.kill_request_timeout
        )


class E2BAGSBackend(AGSBackend):
    """Create synchronous AGS sessions through the E2B-compatible SDK."""

    def __init__(self, config: E2BAGSBackendConfig, sandbox_class: Any = None) -> None:
        self.config = config
        self._sandbox_class = sandbox_class
        # Session creation performs a network request, so it cannot be made
        # atomic with the registry update by simply holding a lock around the
        # SDK call.  Keep a condition for the small create/close_all handoff:
        # close_all marks the backend as closing, waits for in-flight creates,
        # and then snapshots every registered session.
        self._session_condition = threading.Condition(threading.RLock())
        self._active_sessions: set[E2BAGSSession] = set()
        self._creating_sessions = 0
        self._close_all_started = False

    @property
    def active_session_count(self) -> int:
        with self._session_condition:
            return len(self._active_sessions)

    @property
    def active_session_ids(self) -> tuple[str, ...]:
        """Return a stable snapshot of active sandbox IDs without sessions."""

        with self._session_condition:
            sessions = tuple(self._active_sessions)
        return tuple(sorted(session.sandbox_id for session in sessions))

    def create(self, create_spec: AGSCreateSpec) -> "E2BAGSSession":
        create_started = False
        try:
            with self._session_condition:
                if self._close_all_started:
                    raise E2BAGSOperationError(
                        "E2B AGS create rejected: backend cleanup has started"
                    )
                self._creating_sessions += 1
                create_started = True

            sandbox_class = (
                self._sandbox_class
                if self._sandbox_class is not None
                else _load_sandbox_class()
            )
            metadata = {key: str(value) for key, value in create_spec.metadata.items()}
            try:
                raw_sandbox = sandbox_class.create(
                    template=create_spec.template,
                    timeout=math.ceil(create_spec.lifetime_timeout),
                    metadata=metadata,
                    api_key=self.config.api_key,
                    domain=self.config.domain,
                    request_timeout=self.config.create_request_timeout,
                )
            except Exception as exc:
                raise _wrap_sdk_error("create", exc, self.config.api_key) from exc
            session = E2BAGSSession(
                raw_sandbox,
                operation_request_timeout=self.config.operation_request_timeout,
                kill_request_timeout=self.config.kill_request_timeout,
                api_key=self.config.api_key,
                on_closed=self._discard_session,
            )
            with self._session_condition:
                self._active_sessions.add(session)
            return session
        finally:
            with self._session_condition:
                if create_started:
                    self._creating_sessions -= 1
                    self._session_condition.notify_all()

    def close_all(self) -> list[str]:
        """Best-effort close every active session and return redacted errors.

        A failed kill intentionally stays in ``_active_sessions``.  A later
        call can therefore retry cleanup, while successful sessions remove
        themselves through the callback supplied above.
        """

        with self._session_condition:
            self._close_all_started = True
            while self._creating_sessions:
                self._session_condition.wait()
            sessions = tuple(self._active_sessions)

        errors: list[str] = []
        for session in sessions:
            try:
                session.close()
            except Exception as exc:
                errors.append(_redact_error(str(exc), self.config.api_key))
        return errors

    def _discard_session(self, session: "E2BAGSSession") -> None:
        with self._session_condition:
            self._active_sessions.discard(session)


class E2BAGSSession(AGSSandboxSession):
    """Adapter around one synchronous E2B ``Sandbox`` instance."""

    def __init__(
        self,
        raw_sandbox: Any,
        operation_request_timeout: int | float,
        kill_request_timeout: int | float,
        api_key: str = "",
        on_closed: Callable[["E2BAGSSession"], None] | None = None,
    ) -> None:
        self._raw_sandbox = raw_sandbox
        self._operation_request_timeout = operation_request_timeout
        self._kill_request_timeout = kill_request_timeout
        self._api_key = api_key
        self._on_closed = on_closed
        self._close_lock = threading.Lock()
        self._closed = False

    @property
    def sandbox_id(self) -> str:
        value = self._raw_sandbox.sandbox_id
        if not isinstance(value, str):
            raise TypeError("E2B sandbox_id must be a string")
        if not value:
            raise ValueError("E2B sandbox_id must be a non-empty string")
        return value

    def write_file(self, path: str, content: bytes | str) -> None:
        try:
            self._raw_sandbox.files.write(
                path,
                content,
                request_timeout=self._operation_request_timeout,
            )
        except Exception as exc:
            raise _wrap_sdk_error("write", exc, self._api_key) from exc

    def read_file(self, path: str) -> bytes:
        try:
            content = self._raw_sandbox.files.read(
                path,
                format="bytes",
                request_timeout=self._operation_request_timeout,
            )
        except Exception as exc:
            raise _wrap_sdk_error("read", exc, self._api_key) from exc
        if not isinstance(content, (bytes, bytearray, memoryview)):
            raise TypeError("E2B files.read(format='bytes') must return bytes-like data")
        return bytes(content)

    def run(
        self,
        command: str,
        timeout: float | int,
        envs: Mapping[str, str],
    ) -> AGSCommandResult:
        try:
            result = self._raw_sandbox.commands.run(
                command,
                timeout=timeout,
                envs=dict(envs),
                request_timeout=self._operation_request_timeout,
            )
        except Exception as exc:
            result = _command_exit_result(exc)
            if result is not None:
                return result
            raise _wrap_sdk_error("run", exc, self._api_key) from exc

        exit_code = result.exit_code
        if not isinstance(exit_code, int) or isinstance(exit_code, bool):
            raise TypeError("E2B command exit_code must be an integer")

        stdout = _normalise_output("stdout", result.stdout)
        stderr = _normalise_output("stderr", result.stderr)
        if not stderr:
            error = getattr(result, "error", None)
            if isinstance(error, str) and error:
                stderr = error
        return AGSCommandResult(exit_code, stdout, stderr)

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            try:
                result = self._raw_sandbox.kill(
                    request_timeout=self._kill_request_timeout
                )
            except Exception as exc:
                # Leave the session open so callers and close_all can retry a
                # transient kill failure.
                raise _wrap_sdk_error("close", exc, self._api_key) from exc
            # Some SDK versions use False to signal that the kill did not
            # complete.  Treat that as a retryable operation failure, unlike
            # a successful call returning None (the common SDK behaviour).
            if result is False:
                raise E2BAGSOperationError(
                    "E2B AGS close failed: sandbox kill returned false"
                )
            self._closed = True

        if self._on_closed is not None:
            self._on_closed(self)


def _load_sandbox_class() -> Any:
    try:
        from e2b import Sandbox
    except ImportError as exc:
        raise RuntimeError(_MISSING_E2B_MESSAGE) from exc
    return Sandbox


def _normalise_output(name: str, value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError(f"E2B command {name} must be a string or None")
    return value


def _command_exit_result(exc: Exception) -> AGSCommandResult | None:
    """Normalize SDK versions that raise for an ordinary non-zero exit."""

    if exc.__class__.__name__ != "CommandExitException":
        return None
    exit_code = getattr(exc, "exit_code", None)
    if not isinstance(exit_code, int) or isinstance(exit_code, bool):
        return None
    stdout = _normalise_output("stdout", getattr(exc, "stdout", None))
    stderr = _normalise_output("stderr", getattr(exc, "stderr", None))
    if not stderr:
        error = getattr(exc, "error", None)
        if isinstance(error, str) and error:
            stderr = error
    return AGSCommandResult(exit_code, stdout, stderr)


def _wrap_sdk_error(operation: str, exc: Exception, api_key: str) -> E2BAGSOperationError:
    message = _redact_error(str(exc), api_key)
    return E2BAGSOperationError(f"E2B AGS {operation} failed: {message}")


def _redact_error(message: str, api_key: str) -> str:
    if api_key:
        return message.replace(api_key, "[REDACTED]")
    return message


def _require_positive_finite_number(name: str, value: Any) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number")
    try:
        finite = math.isfinite(float(value))
    except (OverflowError, ValueError):
        finite = False
    if not finite or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
