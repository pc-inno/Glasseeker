"""Synchronous AGS sandbox adapter for BrowseComp attempts.

The adapter deliberately keeps the cloud-backend boundary small.  A backend
only has to create a session; the worker uploads the existing secret-free
request protocol, runs the already packaged sandbox worker, and downloads the
worker's explicitly declared artifacts.
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import shlex
import stat
import threading
import time
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from urllib.parse import urlsplit

from .dataset import RunSpec
from .hermes_client import HermesClient, HermesConfig
from .sandbox_protocol import PROTOCOL_VERSION, SandboxRunRequest, SandboxRunResult


# These are intentionally explicit.  In particular, model credential names
# are not included in the pass-through list; the worker receives only the
# dedicated HERMES_API_KEY value below.
DEFAULT_PASS_ENV: tuple[str, ...] = (
    "BROWSER_CDP_URL",
    "BROWSER_PROXY_URL",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
    "SERPER_API_KEY",
    "JINA_API_KEY",
    "SEARCH_SERVER_ENDPOINT",
    "SEARCH_SERVER_BASE_URL",
    "SEARCH_SERVER_URL",
    "SEARCH_SERVER_PORT",
    "SEARCH_SERVER_SCHEME",
    "SEARCH_SERVER_API_KEY",
    "FETCH_SERVER_BASE_URL",
    "FETCH_SERVER_URL",
    "FETCH_SERVER_JINA_API_KEY",
    "FETCH_SERVER_LLM_BASE_URL",
    "FETCH_SERVER_LLM_MODEL",
    "FETCH_SERVER_LLM_API_KEY",
    "LLM_API_KEY",
    "HERMES_CONTEXT_COMPRESSOR",
    "WEB_EXTRACT_CONTENT_MODE",
    "SUBAGENT_MAX_ITERATIONS",
    "SUBAGENT_TIMEOUT_SECONDS",
)

_TAIL_LIMIT = 4000
_DEFAULT_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
_WORKER_BUNDLE_MEMBERS: tuple[str, ...] = (
    "__init__.py",
    "dataset.py",
    "sandbox_protocol.py",
    "endpoint_health.py",
    "endpoint_registry.py",
    "session_reader.py",
    "hermes_client.py",
    "mock_search_server.py",
    "sandbox_worker.py",
)
_GUARD_PLUGIN_MEMBERS: tuple[str, ...] = ("__init__.py", "plugin.yaml")
_SEARCH_SERVER_PLUGIN_MEMBERS: tuple[str, ...] = (
    "__init__.py",
    "provider.py",
    "plugin.yaml",
)
_HERMES_SOURCE_DIRS: tuple[str, ...] = (
    "agent",
    "tools",
    "hermes_cli",
    "gateway",
    "tui_gateway",
    "cron",
    "acp_adapter",
    "plugins",
    "providers",
    "locales",
    "optional-mcps",
)
_HERMES_SOURCE_EXCLUDED_PARTS = frozenset(
    {
        ".git",
        ".venv",
        "browse_comp_eval",
        "tests",
        "docs",
        "output",
        "data",
        "dataset",
        "datasets",
        "__pycache__",
    }
)
_HERMES_SOURCE_SECRET_SUFFIXES = (
    ".pem",
    ".key",
    ".crt",
    ".cer",
    ".der",
    ".p12",
    ".pfx",
    ".jks",
    ".keystore",
)
_HERMES_SOURCE_SECRET_NAMES = frozenset(
    {
        "id_rsa",
        "id_ed25519",
        "id_ecdsa",
        "id_dsa",
        "private_key",
        "privatekey",
    }
)
_HERMES_SOURCE_ARCHIVE_ENV = "BROWSE_COMP_HERMES_SOURCE_ARCHIVE"
_HERMES_SOURCE_ROOT_ENV = "BROWSE_COMP_HERMES_SOURCE_ROOT"
_FETCH_SERVER_SOURCE_ENV = "BROWSE_COMP_FETCH_SERVER_SOURCE_DIR"
_FETCH_SERVER_PORT_ENV = "BROWSE_COMP_FETCH_SERVER_PORT"
_FETCH_SERVER_INSTALL_ENV = "BROWSE_COMP_FETCH_SERVER_INSTALL_DEPENDENCIES"
_FETCH_SERVER_STARTUP_TIMEOUT_ENV = "BROWSE_COMP_FETCH_SERVER_STARTUP_TIMEOUT"
_FETCH_SERVER_INSTALL_TIMEOUT_ENV = "BROWSE_COMP_FETCH_SERVER_INSTALL_TIMEOUT"
_FETCH_SERVER_PROBE_URL_ENV = "BROWSE_COMP_FETCH_SERVER_PROBE_URL"
_FETCH_SERVER_ARCHIVE_PREFIX = "fetch_server"
_FETCH_SERVER_DEPENDENCY_NAMES = ("requirements.lock.txt", "requirements.txt")
_FETCH_SERVER_OPTIONAL_NAMES = ("config.yaml",)
_FETCH_SERVER_CONFIG_SECRET = re.compile(
    rb"(?im)^\s*(?:api[_-]?key|token|secret|password|authorization)\s*:\s*\S+"
)
_FETCH_SERVER_URL_CREDENTIALS = re.compile(rb"://[^/\s:@]+:[^@\s/]+@")
_CREDENTIAL_ENV_NAME = re.compile(
    r"(?:^|_)(?:KEY|TOKEN|SECRET|PASSWORD)(?:$|_)", re.IGNORECASE
)


@dataclass(frozen=True)
class AGSCommandResult:
    """The synchronous result returned by an AGS session command."""

    exit_code: int
    stdout: str
    stderr: str

    def __post_init__(self) -> None:
        if not isinstance(self.exit_code, int) or isinstance(self.exit_code, bool):
            raise TypeError("exit_code must be an integer")
        if not isinstance(self.stdout, str):
            raise TypeError("stdout must be a string")
        if not isinstance(self.stderr, str):
            raise TypeError("stderr must be a string")


@dataclass(frozen=True)
class AGSCreateSpec:
    """The non-secret parameters used to create one isolated AGS session."""

    template: str
    lifetime_timeout: float | int
    metadata: Mapping[str, str | int]

    def __post_init__(self) -> None:
        if not isinstance(self.template, str) or not self.template:
            raise ValueError("template must be a non-empty string")
        _require_positive_number("lifetime_timeout", self.lifetime_timeout)
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        for key, value in self.metadata.items():
            if not isinstance(key, str) or not key:
                raise ValueError("metadata keys must be non-empty strings")
            if not isinstance(value, (str, int)) or isinstance(value, bool):
                raise TypeError("metadata values must be strings or integers")


class AGSSandboxSession(Protocol):
    @property
    def sandbox_id(self) -> str:
        ...

    def write_file(self, path: str, content: bytes | str) -> None:
        ...

    def read_file(self, path: str) -> bytes:
        ...

    def run(
        self,
        command: str,
        timeout: float | int,
        envs: Mapping[str, str],
    ) -> AGSCommandResult:
        ...

    def close(self) -> None:
        ...


class AGSBackend(Protocol):
    def create(self, create_spec: AGSCreateSpec) -> AGSSandboxSession:
        ...


@dataclass(frozen=True)
class AGSWorkerConfig:
    """Configuration for one synchronous AGS worker invocation."""

    template: str = "browse-comp"
    lifetime_timeout: float | int = 3600
    command_timeout: float | int = 1800
    container_python: str = "python"
    container_hermes_bin: str = "hermes"
    worker_pythonpath: str = ""
    upload_worker_bundle: bool = False
    bootstrap_bundle_path: str = "/tmp/browse_comp_eval/bootstrap_bundle.zip"
    guard_plugin_path: str = "/tmp/browse_comp_eval/plugins/browse_comp_guard"
    search_server_plugin_path: str = "/tmp/browse_comp_eval/plugins/search_server"
    request_path: str = "/tmp/browse_comp_eval/request.json"
    result_path: str = "/tmp/browse_comp_eval/result.json"
    cancel_path: str = "/tmp/browse_comp_eval/cancel.json"
    cancel_grace_seconds: float | int = 10
    workspace_path: str = "/workspace"
    pass_env: tuple[str, ...] = DEFAULT_PASS_ENV
    max_artifact_bytes: int = _DEFAULT_MAX_ARTIFACT_BYTES
    upload_hermes_source: bool = False
    hermes_source_root_path: str = "/tmp/browse_comp_eval/hermes_source"
    fetch_server_source_dir: str | Path | None = None
    fetch_server_container_source_dir: str | None = None
    fetch_server_install_dependencies: bool = False
    fetch_server_port: int = 18081
    fetch_server_startup_timeout: float | int = 30
    fetch_server_install_timeout: float | int = 300
    fetch_server_probe_url: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.template, str) or not self.template:
            raise ValueError("template must be a non-empty string")
        _require_positive_number("lifetime_timeout", self.lifetime_timeout)
        _require_positive_number("command_timeout", self.command_timeout)
        _require_positive_number("cancel_grace_seconds", self.cancel_grace_seconds)
        for name in ("container_python", "container_hermes_bin"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.worker_pythonpath, str):
            raise TypeError("worker_pythonpath must be a string")
        if not isinstance(self.upload_worker_bundle, bool):
            raise TypeError("upload_worker_bundle must be a boolean")
        if not isinstance(self.upload_hermes_source, bool):
            raise TypeError("upload_hermes_source must be a boolean")
        if self.upload_hermes_source and not self.upload_worker_bundle:
            raise ValueError(
                "upload_hermes_source requires upload_worker_bundle"
            )
        if self.fetch_server_source_dir is not None:
            if not self.upload_hermes_source:
                raise ValueError(
                    "fetch_server_source_dir requires upload_hermes_source"
                )
            _validate_fetch_server_source_dir(self.fetch_server_source_dir)
        expected_fetch_source_dir = str(
            PurePosixPath(self.hermes_source_root_path)
            / _FETCH_SERVER_ARCHIVE_PREFIX
        )
        if self.fetch_server_container_source_dir is None:
            object.__setattr__(
                self,
                "fetch_server_container_source_dir",
                expected_fetch_source_dir,
            )
        elif not isinstance(self.fetch_server_container_source_dir, str):
            raise TypeError("fetch_server_container_source_dir must be a string")
        elif self.fetch_server_container_source_dir != expected_fetch_source_dir:
            raise ValueError(
                "fetch_server_container_source_dir must match the deterministic "
                "bootstrap archive location"
            )
        if not isinstance(self.fetch_server_install_dependencies, bool):
            raise TypeError("fetch_server_install_dependencies must be a boolean")
        if not isinstance(self.fetch_server_port, int) or isinstance(
            self.fetch_server_port, bool
        ):
            raise TypeError("fetch_server_port must be an integer")
        if not 1 <= self.fetch_server_port <= 65535:
            raise ValueError("fetch_server_port must be between 1 and 65535")
        _require_positive_number(
            "fetch_server_startup_timeout", self.fetch_server_startup_timeout
        )
        _require_positive_number(
            "fetch_server_install_timeout", self.fetch_server_install_timeout
        )
        _validate_fetch_server_probe_url(self.fetch_server_probe_url)
        for name in (
            "bootstrap_bundle_path",
            "guard_plugin_path",
            "search_server_plugin_path",
            "request_path",
            "result_path",
            "cancel_path",
            "workspace_path",
            "hermes_source_root_path",
            "fetch_server_container_source_dir",
        ):
            _validate_container_path(name, getattr(self, name))
        if not isinstance(self.pass_env, tuple) or not all(
            isinstance(name, str) and name for name in self.pass_env
        ):
            raise TypeError("pass_env must be a tuple of non-empty strings")
        if not isinstance(self.max_artifact_bytes, int) or isinstance(
            self.max_artifact_bytes, bool
        ):
            raise TypeError("max_artifact_bytes must be an integer")
        if self.max_artifact_bytes <= 0:
            raise ValueError("max_artifact_bytes must be positive")


class AGSWorker(HermesClient):
    """Run one BrowseComp attempt inside an isolated AGS session."""

    def __init__(
        self,
        config: HermesConfig,
        worker_config: AGSWorkerConfig,
        backend: AGSBackend,
    ) -> None:
        super().__init__(config)
        self.worker_config = worker_config
        self.backend = backend
        self._active_session_condition = threading.Condition(threading.RLock())
        self._active_sessions: dict[int, AGSSandboxSession] = {}
        self._session_cancel_reasons: dict[int, str] = {}
        self._session_cancel_forced: dict[int, bool] = {}

    def cancel_active(self, reason: str) -> None:
        """Ask active sandbox workers to persist and stop, then kill stragglers."""

        with self._active_process_lock:
            self._cancellation_generation += 1
            self._latest_cancel_reason = reason
        with self._active_session_condition:
            sessions = list(self._active_sessions.items())
            for token, _session in sessions:
                self._session_cancel_reasons[token] = reason
                self._session_cancel_forced[token] = False
        marker = json.dumps(
            {
                "failure_type": "model_endpoint_unavailable",
                "reason": reason,
                "attempt_consumed": False,
            },
            ensure_ascii=False,
        )
        marker_failures: set[int] = set()
        for token, session in sessions:
            try:
                session.write_file(self.worker_config.cancel_path, marker)
            except Exception:
                marker_failures.add(token)
                with self._active_session_condition:
                    self._session_cancel_forced[token] = True

        for token, session in sessions:
            if token not in marker_failures:
                continue
            try:
                session.close()
            except Exception:
                pass

        graceful_tokens = {
            token for token, _session in sessions if token not in marker_failures
        }
        deadline = time.monotonic() + float(self.worker_config.cancel_grace_seconds)
        with self._active_session_condition:
            while any(token in self._active_sessions for token in graceful_tokens):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._active_session_condition.wait(timeout=min(0.2, remaining))
            survivors = [
                (token, session)
                for token, session in sessions
                if token in graceful_tokens and token in self._active_sessions
            ]
            for token, _session in survivors:
                self._session_cancel_forced[token] = True
        for _token, session in survivors:
            try:
                session.close()
            except Exception:
                pass

    def run(self, spec: RunSpec, workspace_dir: Path, attempt: int) -> dict[str, Any]:
        """Execute an attempt and return the runner-compatible dictionary.

        The only reference-answer access in this method is the final host-side
        assignment, after the sandbox result and artifacts have been handled.
        All backend and protocol failures become retryable infrastructure
        failures instead of escaping to :class:`BrowseCompRunner`.
        """

        host_workspace = Path(workspace_dir)
        request: SandboxRunRequest | None = None
        session: AGSSandboxSession | None = None
        result_dict: dict[str, Any] | None = None
        command_result: AGSCommandResult | None = None
        sandbox_id: str | None = None
        session_token: int | None = None
        execution_generation = self._begin_execution()

        try:
            try:
                host_workspace.mkdir(parents=True, exist_ok=True)
                request = replace(
                    self.build_request(spec, attempt),
                    hermes_bin=self.worker_config.container_hermes_bin,
                )
            except Exception as exc:
                result_dict = self._failure_without_request(
                    spec,
                    attempt,
                    host_workspace,
                    self._error_text(exc),
                )

            if result_dict is None:
                assert request is not None
                try:
                    create_spec = AGSCreateSpec(
                        template=self.worker_config.template,
                        lifetime_timeout=self.worker_config.lifetime_timeout,
                        metadata={
                            "run_id": request.run_id,
                            "question_id": request.question_id,
                            "attempt": request.attempt,
                        },
                    )
                    session = self.backend.create(create_spec)
                    session_token = id(session)
                    with self._active_session_condition:
                        self._active_sessions[session_token] = session
                    sandbox_id = self._sandbox_id(session)

                    sticky_cancel_reason = self._execution_cancel_reason(
                        execution_generation
                    )
                    if sticky_cancel_reason:
                        raise RuntimeError(sticky_cancel_reason)

                    self._upload_bootstrap(session, request)
                    session.write_file(
                        self.worker_config.request_path,
                        request.to_json(),
                    )
                    command = shlex.join(
                        [
                            self.worker_config.container_python,
                            "-m",
                            "browse_comp_eval.sandbox_worker",
                            "--request",
                            self.worker_config.request_path,
                            "--result",
                            self.worker_config.result_path,
                            "--workspace",
                            self.worker_config.workspace_path,
                            "--cancel-file",
                            self.worker_config.cancel_path,
                        ]
                    )
                    envs = self._build_env(request)
                    command_result = session.run(
                        command,
                        timeout=self.worker_config.command_timeout,
                        envs=envs,
                    )
                    stdout_tail = self._tail(command_result.stdout)
                    stderr_tail = self._tail(command_result.stderr)

                    # A non-zero worker exit is not enough to classify the
                    # attempt as infrastructure failure: sandbox_worker writes
                    # structured failed results and intentionally exits non-zero
                    # for those results.
                    raw_result = session.read_file(self.worker_config.result_path)
                    if not isinstance(raw_result, bytes):
                        raise TypeError("AGS read_file must return bytes")
                    parsed_result = SandboxRunResult.from_json(
                        raw_result.decode("utf-8")
                    )
                    self._validate_result_identity(parsed_result, request)
                    self._download_artifacts(
                        parsed_result.output_files,
                        session,
                        host_workspace,
                    )

                    result_dict = parsed_result.to_dict()
                    result_dict = self._redact_result_text(result_dict)
                    result_dict["workspace_dir"] = str(host_workspace)
                    result_dict["sandbox_id"] = self._redact(sandbox_id)
                    result_dict["sandbox_command_stdout_tail"] = stdout_tail
                    result_dict["sandbox_command_stderr_tail"] = stderr_tail
                    result_dict["sandbox_command_exit_code"] = command_result.exit_code
                except Exception as exc:
                    cancel_reason = self._cancel_reason(
                        session_token
                    ) or self._execution_cancel_reason(execution_generation)
                    result_dict = self._failure_for_request(
                        request,
                        host_workspace,
                        cancel_reason or self._error_text(exc),
                        sandbox_id=sandbox_id,
                        command_result=command_result,
                        endpoint_cancelled=bool(cancel_reason),
                    )
        finally:
            if session is not None:
                try:
                    session.close()
                except Exception as exc:
                    cleanup_error = self._error_text(exc)
                    if result_dict is None:
                        if request is not None:
                            result_dict = self._failure_for_request(
                                request,
                                host_workspace,
                                "sandbox execution did not produce a result",
                                sandbox_id=sandbox_id,
                                command_result=command_result,
                            )
                        else:
                            result_dict = self._failure_without_request(
                                spec,
                                attempt,
                                host_workspace,
                                "sandbox execution did not produce a result",
                            )
                    result_dict["sandbox_cleanup_error"] = cleanup_error
            if session_token is not None:
                with self._active_session_condition:
                    self._active_sessions.pop(session_token, None)
                    self._active_session_condition.notify_all()

        if result_dict is None:
            # This is only reachable if an unexpected BaseException escaped the
            # ordinary-error handling above.  Keep the public client contract
            # total for normal Exception-based backend failures.
            result_dict = self._failure_without_request(
                spec,
                attempt,
                host_workspace,
                "sandbox execution did not produce a result",
            )
        cancel_reason = self._cancel_reason(
            session_token
        ) or self._execution_cancel_reason(execution_generation)
        if cancel_reason:
            result_dict["status"] = "failed"
            result_dict["error"] = cancel_reason
            result_dict["failure_type"] = "model_endpoint_unavailable"
            result_dict["attempt_consumed"] = False
            result_dict["trace_complete"] = bool(result_dict.get("history")) and not self._cancel_forced(
                session_token
            )
        if session_token is not None:
            with self._active_session_condition:
                self._session_cancel_reasons.pop(session_token, None)
                self._session_cancel_forced.pop(session_token, None)
        result_dict["answer"] = spec.item.answer
        return result_dict

    def _cancel_reason(self, session_token: int | None) -> str:
        if session_token is None:
            return ""
        with self._active_session_condition:
            return self._session_cancel_reasons.get(session_token, "")

    def _cancel_forced(self, session_token: int | None) -> bool:
        if session_token is None:
            return False
        with self._active_session_condition:
            return self._session_cancel_forced.get(session_token, False)

    def _build_env(self, request: SandboxRunRequest | None = None) -> dict[str, str]:
        """Build the intentionally narrow environment passed to the session."""

        python_paths = []
        if self.worker_config.upload_hermes_source:
            python_paths.append(self.worker_config.hermes_source_root_path)
        if self.worker_config.upload_worker_bundle:
            python_paths.append(self.worker_config.bootstrap_bundle_path)
        if self.worker_config.worker_pythonpath:
            python_paths.append(self.worker_config.worker_pythonpath)
        envs = {
            "HERMES_API_KEY": self.config.api_key,
            "PYTHONPATH": ":".join(python_paths),
        }
        fetch_server_enabled = self.worker_config.fetch_server_source_dir is not None
        if self.worker_config.upload_hermes_source:
            envs[_HERMES_SOURCE_ARCHIVE_ENV] = self.worker_config.bootstrap_bundle_path
            envs[_HERMES_SOURCE_ROOT_ENV] = self.worker_config.hermes_source_root_path
            envs["HERMES_BUNDLED_PLUGINS"] = str(
                PurePosixPath(self.worker_config.hermes_source_root_path) / "plugins"
            )
        elif self.worker_config.upload_worker_bundle:
            envs["BROWSE_COMP_GUARD_PLUGIN_SOURCE"] = (
                self.worker_config.guard_plugin_path
            )
            if request is None or request.search_mode != "disabled":
                envs["BROWSE_COMP_SEARCH_SERVER_PLUGIN_SOURCE"] = (
                    self.worker_config.search_server_plugin_path
                )
        for name in self.worker_config.pass_env:
            if name in envs:
                continue
            if fetch_server_enabled and name in {
                "FETCH_SERVER_BASE_URL",
                "FETCH_SERVER_URL",
            }:
                # The sidecar is always loopback inside the sandbox.  Never let
                # a host URL win over this explicit runtime overlay.
                continue
            if request is not None and request.search_mode == "disabled" and name in {
                "SEARCH_SERVER_ENDPOINT",
                "SEARCH_SERVER_BASE_URL",
                "SEARCH_SERVER_URL",
                "SEARCH_SERVER_PORT",
                "SEARCH_SERVER_SCHEME",
                "SEARCH_SERVER_API_KEY",
            }:
                continue
            if name in os.environ:
                envs[name] = os.environ[name]
        if fetch_server_enabled:
            fetch_base_url = (
                f"http://127.0.0.1:{self.worker_config.fetch_server_port}"
            )
            envs["FETCH_SERVER_BASE_URL"] = fetch_base_url
            envs["FETCH_SERVER_URL"] = fetch_base_url
            envs[_FETCH_SERVER_SOURCE_ENV] = str(
                self.worker_config.fetch_server_container_source_dir
            )
            envs[_FETCH_SERVER_PORT_ENV] = str(self.worker_config.fetch_server_port)
            envs[_FETCH_SERVER_INSTALL_ENV] = (
                "1" if self.worker_config.fetch_server_install_dependencies else "0"
            )
            envs[_FETCH_SERVER_STARTUP_TIMEOUT_ENV] = str(
                self.worker_config.fetch_server_startup_timeout
            )
            envs[_FETCH_SERVER_INSTALL_TIMEOUT_ENV] = str(
                self.worker_config.fetch_server_install_timeout
            )
            if self.worker_config.fetch_server_probe_url:
                envs[_FETCH_SERVER_PROBE_URL_ENV] = (
                    self.worker_config.fetch_server_probe_url
                )
        return envs

    def _upload_bootstrap(
        self,
        session: AGSSandboxSession,
        request: SandboxRunRequest | None = None,
    ) -> None:
        """Upload the image-independent worker bootstrap and legacy plugins."""

        if not self.worker_config.upload_worker_bundle:
            return

        if self.worker_config.upload_hermes_source:
            if self.worker_config.fetch_server_source_dir is None:
                bundle = _build_bootstrap_bundle()
            else:
                bundle = _build_bootstrap_bundle(
                    self.worker_config.fetch_server_source_dir,
                    self._fetch_server_archive_prefix(),
                )
            session.write_file(
                self.worker_config.bootstrap_bundle_path,
                bundle,
            )
            return

        session.write_file(
            self.worker_config.bootstrap_bundle_path,
            _build_worker_bundle(),
        )
        plugin_source = _repository_root() / "plugins" / "browse_comp_guard"
        plugin_destination = PurePosixPath(self.worker_config.guard_plugin_path)
        for member in _GUARD_PLUGIN_MEMBERS:
            session.write_file(
                str(plugin_destination / member),
                (plugin_source / member).read_bytes(),
            )

        # Search modes that can issue searches use the same provider in the
        # sandbox, regardless of whether its endpoint is external or the
        # worker's loopback mock server.  Disabled runs intentionally omit the
        # plugin so the backend cannot be selected accidentally.
        if request is not None and request.search_mode == "disabled":
            return
        search_source = _repository_root() / "plugins" / "web" / "search_server"
        search_destination = PurePosixPath(
            self.worker_config.search_server_plugin_path
        )
        for member in _SEARCH_SERVER_PLUGIN_MEMBERS:
            session.write_file(
                str(search_destination / member),
                (search_source / member).read_bytes(),
            )

    def _fetch_server_archive_prefix(self) -> str:
        container_path = PurePosixPath(
            self.worker_config.fetch_server_container_source_dir
        )
        source_root = PurePosixPath(self.worker_config.hermes_source_root_path)
        try:
            return container_path.relative_to(source_root).as_posix()
        except ValueError as exc:
            raise ValueError(
                "fetch_server_container_source_dir must be inside "
                "hermes_source_root_path"
            ) from exc

    def _sandbox_id(self, session: AGSSandboxSession) -> str:
        value = session.sandbox_id
        if not isinstance(value, str):
            raise TypeError("AGS sandbox_id must be a string")
        return self._redact(value)

    def _failure_for_request(
        self,
        request: SandboxRunRequest,
        host_workspace: Path,
        error: str,
        *,
        sandbox_id: str | None = None,
        command_result: AGSCommandResult | None = None,
        endpoint_cancelled: bool = False,
    ) -> dict[str, Any]:
        result = SandboxRunResult.failed(
            request,
            host_workspace,
            self._redact(error),
        ).to_dict()
        result["workspace_dir"] = str(host_workspace)
        result["failure_type"] = (
            "model_endpoint_unavailable"
            if endpoint_cancelled
            else "sandbox_infrastructure"
        )
        result["attempt_consumed"] = not endpoint_cancelled
        result["trace_complete"] = False
        if sandbox_id is not None:
            result["sandbox_id"] = self._redact(sandbox_id)
        if command_result is not None:
            result["stdout_tail"] = self._tail(command_result.stdout)
            result["stderr_tail"] = self._tail(command_result.stderr)
            result["sandbox_command_stdout_tail"] = self._tail(command_result.stdout)
            result["sandbox_command_stderr_tail"] = self._tail(command_result.stderr)
            result["sandbox_command_exit_code"] = command_result.exit_code
        return result

    def _failure_without_request(
        self,
        spec: RunSpec,
        attempt: int,
        host_workspace: Path,
        error: str,
    ) -> dict[str, Any]:
        """Build a runner-shaped failure if request construction itself failed."""

        return {
            "question_id": spec.question_id,
            "repeat_index": spec.repeat_index,
            "run_id": spec.run_id,
            "attempt": attempt,
            "question": spec.item.question,
            "type": spec.item.type,
            "model": self.config.model,
            "provider": self.config.provider,
            "model_base_url": self.config.base_url,
            "save_name": self.config.save_name,
            "status": "failed",
            "error": self._redact(error),
            "attempt_consumed": True,
            "failure_type": "sandbox_infrastructure",
            "trace_complete": False,
            "returncode": None,
            "timed_out": False,
            "model_response": "",
            "final_assistant_valid": False,
            "final_assistant_error": "",
            "session_id": None,
            "profile": "",
            "browser_cdp_url": "",
            "workspace_dir": str(host_workspace),
            "output_files": [],
            "tool_calls": {},
            "rounds": 0,
            "total_tokens_estimate": 0,
            "duration_seconds": 0.0,
            "wall_seconds": 0.0,
            "stdout_tail": "",
            "stderr_tail": "",
            "agent_log_tail": "",
            "mcp_registration_error": "",
            "history": [],
            "protocol_version": PROTOCOL_VERSION,
        }

    def _download_artifacts(
        self,
        manifest: list[dict[str, Any]] | list[Any],
        session: AGSSandboxSession,
        host_workspace: Path,
    ) -> None:
        if not isinstance(manifest, list):
            raise TypeError("result output_files must be a list")

        entries: list[tuple[PurePosixPath, int]] = []
        declared_total = 0
        for index, entry in enumerate(manifest):
            if not isinstance(entry, Mapping):
                raise ValueError(f"artifact manifest entry {index} must be an object")
            relative = self._artifact_path(entry.get("path"))
            size = entry.get("size")
            if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                raise ValueError(f"artifact {relative.as_posix()} has invalid size")
            declared_total += size
            entries.append((relative, size))

        if declared_total > self.worker_config.max_artifact_bytes:
            raise ValueError("declared artifact size exceeds max_artifact_bytes")

        root = Path(host_workspace)
        root_resolved = root.resolve()
        actual_total = 0
        for relative, _declared_size in entries:
            remote_path = str(
                PurePosixPath(self.worker_config.workspace_path).joinpath(
                    *relative.parts
                )
            )
            data = session.read_file(remote_path)
            if not isinstance(data, bytes):
                raise TypeError("AGS read_file must return bytes")
            actual_total += len(data)
            if actual_total > self.worker_config.max_artifact_bytes:
                raise ValueError("actual artifact size exceeds max_artifact_bytes")

            destination = root.joinpath(*relative.parts)
            resolved_destination = destination.resolve()
            try:
                resolved_destination.relative_to(root_resolved)
            except ValueError as exc:
                raise ValueError("artifact path escapes host workspace") from exc
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)

    @staticmethod
    def _artifact_path(value: Any) -> PurePosixPath:
        if not isinstance(value, str):
            raise ValueError("artifact path must be a string")
        relative = PurePosixPath(value)
        if not value or relative == PurePosixPath("."):
            raise ValueError("artifact path must not be empty or '.'")
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("artifact path must be relative and must not contain '..'")
        return relative

    def _validate_result_identity(
        self,
        result: SandboxRunResult,
        request: SandboxRunRequest,
    ) -> None:
        for field_name in (
            "question_id",
            "run_id",
            "repeat_index",
            "attempt",
            "question",
            "type",
            "model",
            "provider",
            "save_name",
        ):
            if getattr(result, field_name) != getattr(request, field_name):
                raise ValueError(f"sandbox result {field_name} does not match request")

    def _error_text(self, error: BaseException) -> str:
        text = str(error) or error.__class__.__name__
        return self._redact(text)

    def _redact(self, text: str) -> str:
        value = str(text)
        for secret in self._redaction_secrets():
            value = value.replace(secret, "[REDACTED]")
        return value

    def _redaction_secrets(self) -> tuple[str, ...]:
        secrets = {self.config.api_key} if self.config.api_key else set()
        for name in self.worker_config.pass_env:
            if not _CREDENTIAL_ENV_NAME.search(name):
                continue
            value = os.environ.get(name, "")
            if value:
                secrets.add(value)
        return tuple(sorted(secrets, key=len, reverse=True))

    def _tail(self, text: str) -> str:
        value = self._redact(text)
        return value[-_TAIL_LIMIT:]

    def _redact_result_text(self, result: dict[str, Any]) -> dict[str, Any]:
        redacted = self._redact_nested(result)
        assert isinstance(redacted, dict)
        return redacted

    def _redact_nested(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._redact(value)
        if isinstance(value, dict):
            return {
                self._redact(key) if isinstance(key, str) else key: self._redact_nested(
                    nested_value
                )
                for key, nested_value in value.items()
            }
        if isinstance(value, list):
            return [self._redact_nested(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._redact_nested(item) for item in value)
        return value


def _require_positive_number(name: str, value: Any) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number")
    if not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_container_path(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{name} must be a non-empty path")
    parsed = PurePosixPath(value)
    if not parsed.is_absolute() or ".." in parsed.parts:
        raise ValueError(f"{name} must be an absolute POSIX path without '..'")


def _validate_fetch_server_source_dir(value: str | Path) -> Path:
    """Validate the allowlisted host source overlay without following symlinks."""

    if not isinstance(value, (str, Path)):
        raise TypeError("fetch_server_source_dir must be a path or None")
    source_dir = Path(value)
    if source_dir.is_symlink() or not source_dir.is_dir():
        raise ValueError(
            "fetch_server_source_dir must be a regular, non-symlink directory"
        )
    _fetch_server_bundle_members(source_dir)
    return source_dir


def _regular_fetch_server_member(
    source_dir: Path,
    filename: str,
    *,
    required: bool,
) -> Path | None:
    source = source_dir / filename
    if not source.exists() and not source.is_symlink():
        if required:
            raise ValueError(
                f"fetch_server_source_dir/{filename} must be a regular, non-symlink file"
            )
        return None
    if source.is_symlink():
        raise ValueError(
            f"fetch_server_source_dir/{filename} must be a regular, non-symlink file"
        )
    try:
        mode = source.lstat().st_mode
    except OSError as exc:
        raise ValueError(
            f"fetch_server_source_dir/{filename} must be a regular, non-symlink file"
        ) from exc
    if not stat.S_ISREG(mode):
        raise ValueError(
            f"fetch_server_source_dir/{filename} must be a regular, non-symlink file"
        )
    return source


def _fetch_server_bundle_members(source_dir: Path) -> tuple[tuple[str, Path], ...]:
    server = _regular_fetch_server_member(source_dir, "server.py", required=True)
    assert server is not None
    dependency_sources: list[Path] = []
    for filename in _FETCH_SERVER_DEPENDENCY_NAMES:
        source = _regular_fetch_server_member(source_dir, filename, required=False)
        if source is not None:
            dependency_sources.append(source)
    if not dependency_sources:
        raise ValueError(
            "fetch_server_source_dir must contain a regular, non-symlink "
            "requirements.lock.txt or requirements.txt"
        )

    # Prefer an explicitly locked file.  The sandbox worker always consumes the
    # normalized archive name, so legacy Fetch Servers with requirements.txt do
    # not need to be modified in place.
    requirements = dependency_sources[0]
    members: list[tuple[str, Path]] = [
        ("requirements.lock.txt", requirements),
        ("server.py", server),
    ]
    for filename in _FETCH_SERVER_OPTIONAL_NAMES:
        source = _regular_fetch_server_member(source_dir, filename, required=False)
        if source is not None:
            content = source.read_bytes()
            if (
                _looks_like_private_material(content)
                or _FETCH_SERVER_CONFIG_SECRET.search(content)
                or _FETCH_SERVER_URL_CREDENTIALS.search(content)
            ):
                raise ValueError(
                    f"fetch_server_source_dir/{filename} must not contain "
                    "credential-like material"
                )
            members.append((filename, source))
    return tuple(members)


def _validate_fetch_server_probe_url(value: str) -> None:
    if not isinstance(value, str):
        raise TypeError("fetch_server_probe_url must be a string")
    if not value:
        return
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError(
            "fetch_server_probe_url must be an absolute HTTP(S) URL without credentials"
        )


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2] / "Evaluation_backend"


def _build_worker_bundle() -> bytes:
    """Build a deterministic, secret-free zip importable through PYTHONPATH."""

    package_source = Path(__file__).resolve().parent
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member in _WORKER_BUNDLE_MEMBERS:
            info = zipfile.ZipInfo(f"browse_comp_eval/{member}")
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, (package_source / member).read_bytes())
    return output.getvalue()


def _write_deterministic_bundle_member(
    archive: zipfile.ZipFile,
    member: str,
    content: bytes,
) -> None:
    info = zipfile.ZipInfo(member)
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.create_system = 3
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, content)


@lru_cache(maxsize=1)
def _build_bootstrap_bundle(
    fetch_server_source_dir: str | Path | None = None,
    fetch_server_archive_prefix: str = _FETCH_SERVER_ARCHIVE_PREFIX,
) -> bytes:
    """Build the combined worker/Hermes archive and optional fetch overlay.

    The no-fetch call retains the original cached byte-for-byte archive.  The
    optional source directory is part of the cache key, while every archive
    member remains deterministic.
    """

    if fetch_server_source_dir is None:
        return _build_bootstrap_bundle_without_fetch()
    return _build_bootstrap_bundle_with_fetch(
        fetch_server_source_dir,
        fetch_server_archive_prefix,
    )


def _build_bootstrap_bundle_without_fetch() -> bytes:
    """Build the combined worker and Hermes source bootstrap archive."""

    return _build_bootstrap_bundle_impl(None)


def _build_bootstrap_bundle_with_fetch(
    fetch_server_source_dir: str | Path,
    fetch_server_archive_prefix: str,
) -> bytes:
    source_dir = _validate_fetch_server_source_dir(fetch_server_source_dir)
    prefix_path = PurePosixPath(fetch_server_archive_prefix)
    if (
        not fetch_server_archive_prefix
        or prefix_path.is_absolute()
        or ".." in prefix_path.parts
        or "\\" in fetch_server_archive_prefix
    ):
        raise ValueError(
            "fetch_server_archive_prefix must be a non-empty relative path"
        )
    return _build_bootstrap_bundle_impl(source_dir, fetch_server_archive_prefix)


def _build_bootstrap_bundle_impl(
    fetch_server_source_dir: Path | None,
    fetch_server_archive_prefix: str = _FETCH_SERVER_ARCHIVE_PREFIX,
) -> bytes:
    """Build one deterministic bootstrap archive."""

    package_source = Path(__file__).resolve().parent
    output = io.BytesIO()
    written_members: set[str] = set()
    with zipfile.ZipFile(
        output,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for member in _WORKER_BUNDLE_MEMBERS:
            archive_member = f"browse_comp_eval/{member}"
            if archive_member in written_members:
                continue
            _write_deterministic_bundle_member(
                archive,
                archive_member,
                (package_source / member).read_bytes(),
            )
            written_members.add(archive_member)

        for member, source in _iter_hermes_source_files():
            content = source.read_bytes()
            if _looks_like_private_material(content) or member in written_members:
                continue
            _write_deterministic_bundle_member(archive, member, content)
            written_members.add(member)

        if fetch_server_source_dir is not None:
            for filename, source in _fetch_server_bundle_members(
                fetch_server_source_dir
            ):
                archive_member = f"{fetch_server_archive_prefix}/{filename}"
                _write_deterministic_bundle_member(
                    archive,
                    archive_member,
                    source.read_bytes(),
                )
                written_members.add(archive_member)
    return output.getvalue()


def _iter_hermes_source_files() -> list[tuple[str, Path]]:
    """Return the regular files allowed in the Hermes source overlay."""

    repository_root = _repository_root()
    candidates: list[tuple[str, Path]] = []

    for directory_name in _HERMES_SOURCE_DIRS:
        directory = repository_root / directory_name
        if directory.is_symlink() or not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if _path_has_symlink(path, repository_root) or not path.is_file():
                continue
            relative = path.relative_to(repository_root)
            if not _allowed_hermes_source_path(relative):
                continue
            candidates.append((relative.as_posix(), path))

    for path in repository_root.glob("*.py"):
        if _path_has_symlink(path, repository_root) or not path.is_file():
            continue
        relative = path.relative_to(repository_root)
        if _allowed_hermes_source_path(relative):
            candidates.append((relative.as_posix(), path))

    return sorted(candidates, key=lambda item: item[0])


def _path_has_symlink(path: Path, root: Path) -> bool:
    current = path
    while current != root:
        if current.is_symlink():
            return True
        current = current.parent
    return root.is_symlink()


def _allowed_hermes_source_path(relative: Path) -> bool:
    parts = relative.parts
    if not parts or any(
        part.lower() in _HERMES_SOURCE_EXCLUDED_PARTS for part in parts
    ):
        return False
    if any(part.lower().startswith(".env") for part in parts):
        return False

    filename = relative.name.lower()
    if filename.endswith(_HERMES_SOURCE_SECRET_SUFFIXES):
        return False
    if filename in _HERMES_SOURCE_SECRET_NAMES:
        return False
    if filename.endswith(".pyc"):
        return False
    return True


def _looks_like_private_material(content: bytes) -> bool:
    begin = re.compile(
        rb"(?m)^-----BEGIN (?:[A-Z0-9]+ )*(?:PRIVATE KEY|CERTIFICATE)-----\s*$"
    )
    end = re.compile(
        rb"(?m)^-----END (?:[A-Z0-9]+ )*(?:PRIVATE KEY|CERTIFICATE)-----\s*$"
    )
    return bool(begin.search(content) and end.search(content))
