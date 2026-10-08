"""CLI entry point for one secret-free BrowseComp Hermes attempt."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import math
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path
from pathlib import PurePosixPath, PureWindowsPath

from .hermes_client import HermesClient, HermesConfig, list_workspace_files
from .sandbox_protocol import (
    SandboxProtocolError,
    SandboxRunRequest,
    SandboxRunResult,
    load_request,
)


_UPLOADED_PLUGIN_SPECS = {
    "browse_comp_guard": (
        "BROWSE_COMP_GUARD_PLUGIN_SOURCE",
        ("plugin.yaml", "__init__.py"),
    ),
    "search_server": (
        "BROWSE_COMP_SEARCH_SERVER_PLUGIN_SOURCE",
        ("plugin.yaml", "__init__.py", "provider.py"),
    ),
}
_HERMES_SOURCE_ARCHIVE_ENV = "BROWSE_COMP_HERMES_SOURCE_ARCHIVE"
_HERMES_SOURCE_ROOT_ENV = "BROWSE_COMP_HERMES_SOURCE_ROOT"
_FETCH_SERVER_SOURCE_ENV = "BROWSE_COMP_FETCH_SERVER_SOURCE_DIR"
_FETCH_SERVER_PORT_ENV = "BROWSE_COMP_FETCH_SERVER_PORT"
_FETCH_SERVER_INSTALL_ENV = "BROWSE_COMP_FETCH_SERVER_INSTALL_DEPENDENCIES"
_FETCH_SERVER_STARTUP_TIMEOUT_ENV = "BROWSE_COMP_FETCH_SERVER_STARTUP_TIMEOUT"
_FETCH_SERVER_INSTALL_TIMEOUT_ENV = "BROWSE_COMP_FETCH_SERVER_INSTALL_TIMEOUT"
_FETCH_SERVER_PROBE_URL_ENV = "BROWSE_COMP_FETCH_SERVER_PROBE_URL"
_FETCH_SERVER_LOG_NAME = "fetch_server.log"
_FETCH_SERVER_SHUTDOWN_TIMEOUT = 5.0
_FETCH_SERVER_HEALTH_INTERVAL = 5.0
_SECRET_ENV_NAME = re.compile(
    r"(?:^|_)(?:KEY|TOKEN|SECRET|PASSWORD)(?:$|_)", re.IGNORECASE
)
_PROXY_ENV_NAMES = frozenset(
    {
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    }
)


def _install_uploaded_plugin(name: str) -> None:
    """Install one explicitly allowed uploaded plugin.

    The worker receives paths from the sandbox environment, but never treats
    those paths as a list of files to copy.  Plugin names and member files are
    fixed above so an untrusted bundle cannot make the worker copy arbitrary
    host/container files.
    """

    try:
        env_name, members = _UPLOADED_PLUGIN_SPECS[name]
    except KeyError as exc:
        raise ValueError(f"unsupported uploaded plugin: {name}") from exc
    raw_source = os.environ.get(env_name, "").strip()
    if not raw_source:
        return
    source = Path(raw_source)
    if not source.is_dir() or not all((source / member).is_file() for member in members):
        raise RuntimeError(f"uploaded {name} plugin is incomplete")

    from hermes_cli.plugins import get_bundled_plugins_dir

    destination = get_bundled_plugins_dir() / name
    if all((destination / member).is_file() for member in members):
        return
    destination.mkdir(parents=True, exist_ok=True)
    for member in members:
        shutil.copy2(source / member, destination / member)


def _install_guard_plugin() -> None:
    """Backward-compatible wrapper for the explicitly allowed guard plugin."""

    _install_uploaded_plugin("browse_comp_guard")


def _install_uploaded_plugins() -> None:
    for name in _UPLOADED_PLUGIN_SPECS:
        _install_uploaded_plugin(name)


def _source_container_path(env_name: str, raw_path: str) -> Path:
    if not raw_path or "\x00" in raw_path:
        raise RuntimeError(f"{env_name} must be a non-empty path")
    parsed = PurePosixPath(raw_path)
    if not parsed.is_absolute() or ".." in parsed.parts:
        raise RuntimeError(f"{env_name} must be an absolute path without '..'")
    return Path(raw_path)


def _validated_source_member(name: str) -> PurePosixPath:
    if not name or "\x00" in name or "\\" in name:
        raise RuntimeError("Hermes source archive contains an invalid member")
    relative = PurePosixPath(name)
    if (
        relative == PurePosixPath(".")
        or relative.is_absolute()
        or PureWindowsPath(name).is_absolute()
        or ".." in relative.parts
    ):
        raise RuntimeError(
            "Hermes source archive member must be relative and must not contain '..'"
        )
    return relative


def _extract_hermes_source(archive_path: Path, source_root: Path) -> None:
    """Extract a validated source archive into a fresh or empty directory."""

    if source_root.is_symlink():
        raise RuntimeError("Hermes source extraction target must not be a symlink")
    if source_root.exists():
        if not source_root.is_dir() or any(source_root.iterdir()):
            raise RuntimeError(
                "Hermes source extraction target must be new or empty"
            )
    else:
        source_root.mkdir(parents=True)

    if archive_path.is_symlink() or not archive_path.is_file():
        raise RuntimeError("Hermes source archive is not a regular file")

    members: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
    seen: set[PurePosixPath] = set()
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            relative = _validated_source_member(info.filename)
            if relative in seen:
                raise RuntimeError(
                    "Hermes source archive contains duplicate members"
                )
            seen.add(relative)
            mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(mode):
                raise RuntimeError("Hermes source archive must not contain symlinks")
            members.append((info, relative))

        file_members = {
            relative for info, relative in members if not info.is_dir()
        }
        for _info, relative in members:
            if any(
                PurePosixPath(*relative.parts[:index]) in file_members
                for index in range(1, len(relative.parts))
            ):
                raise RuntimeError(
                    "Hermes source archive contains a file/directory conflict"
                )

        for info, relative in members:
            destination = source_root.joinpath(*relative.parts)
            if info.is_dir():
                if destination.exists() and not destination.is_dir():
                    raise RuntimeError(
                        "Hermes source archive destination is not a directory"
                    )
                destination.mkdir(parents=True, exist_ok=True)
                continue

            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() or destination.is_symlink():
                raise RuntimeError("Hermes source extraction would overwrite a file")
            with archive.open(info) as source, destination.open("xb") as target:
                shutil.copyfileobj(source, target)


def _prepare_hermes_source() -> None:
    """Install the uploaded source overlay before importing ``hermes_cli``."""

    raw_archive = os.environ.get(_HERMES_SOURCE_ARCHIVE_ENV, "").strip()
    raw_root = os.environ.get(_HERMES_SOURCE_ROOT_ENV, "").strip()
    if not raw_archive and not raw_root:
        return
    if not raw_archive or not raw_root:
        raise RuntimeError(
            "Hermes source archive and root environment variables are required"
        )

    archive_path = _source_container_path(_HERMES_SOURCE_ARCHIVE_ENV, raw_archive)
    source_root = _source_container_path(_HERMES_SOURCE_ROOT_ENV, raw_root)
    _extract_hermes_source(archive_path, source_root)

    source_root_string = str(source_root)
    while source_root_string in sys.path:
        sys.path.remove(source_root_string)
    sys.path.insert(0, source_root_string)
    # PYTHONPATH already contains this path when the interpreter starts, but
    # the directory does not exist until the archive is extracted above.
    # PathFinder may therefore have cached a failed lookup for it.
    sys.path_importer_cache.pop(source_root_string, None)
    importlib.invalidate_caches()

    hermes_cli = importlib.import_module("hermes_cli")
    module_file = getattr(hermes_cli, "__file__", None)
    if not isinstance(module_file, str):
        raise RuntimeError("Hermes source overlay imported hermes_cli without __file__")
    try:
        Path(module_file).resolve().relative_to(source_root.resolve())
    except ValueError as exc:
        raise RuntimeError(
            "Hermes source overlay did not provide the imported hermes_cli"
        ) from exc


@dataclass(frozen=True)
class _FetchServerSidecarConfig:
    source_dir: Path
    port: int
    install_dependencies: bool
    startup_timeout: float
    install_timeout: float
    probe_url: str = ""


def _parse_positive_timeout(env_name: str, raw_value: str) -> float:
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{env_name} must be a positive number") from exc
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"{env_name} must be a positive number")
    return value


def _parse_sidecar_bool(env_name: str, raw_value: str) -> bool:
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{env_name} must be a boolean")


def _validate_fetch_server_runtime_source(source_dir: Path) -> None:
    if not source_dir.is_absolute() or ".." in PurePosixPath(str(source_dir)).parts:
        raise RuntimeError(
            "fetch server source directory must be an absolute path without '..'"
        )
    if source_dir.is_symlink() or not source_dir.is_dir():
        raise RuntimeError("fetch server source directory is not a regular directory")
    for filename in ("server.py", "requirements.lock.txt"):
        source = source_dir / filename
        if source.is_symlink() or not source.exists():
            raise RuntimeError(f"fetch server source is missing {filename}")
        try:
            mode = source.lstat().st_mode
        except OSError as exc:
            raise RuntimeError(f"fetch server source is missing {filename}") from exc
        if not stat.S_ISREG(mode):
            raise RuntimeError(f"fetch server source {filename} is not a regular file")


def _fetch_server_sidecar_config() -> _FetchServerSidecarConfig | None:
    raw_source = os.environ.get(_FETCH_SERVER_SOURCE_ENV, "").strip()
    if not raw_source:
        return None
    if "\x00" in raw_source:
        raise RuntimeError("fetch server source directory contains NUL")

    source_dir = Path(raw_source)
    _validate_fetch_server_runtime_source(source_dir)
    raw_port = os.environ.get(_FETCH_SERVER_PORT_ENV, "18081")
    try:
        port = int(raw_port)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{_FETCH_SERVER_PORT_ENV} must be an integer") from exc
    if not 1 <= port <= 65535:
        raise RuntimeError(f"{_FETCH_SERVER_PORT_ENV} must be between 1 and 65535")
    return _FetchServerSidecarConfig(
        source_dir=source_dir,
        port=port,
        install_dependencies=_parse_sidecar_bool(
            _FETCH_SERVER_INSTALL_ENV,
            os.environ.get(_FETCH_SERVER_INSTALL_ENV, "0"),
        ),
        startup_timeout=_parse_positive_timeout(
            _FETCH_SERVER_STARTUP_TIMEOUT_ENV,
            os.environ.get(_FETCH_SERVER_STARTUP_TIMEOUT_ENV, "30"),
        ),
        install_timeout=_parse_positive_timeout(
            _FETCH_SERVER_INSTALL_TIMEOUT_ENV,
            os.environ.get(_FETCH_SERVER_INSTALL_TIMEOUT_ENV, "300"),
        ),
        probe_url=os.environ.get(_FETCH_SERVER_PROBE_URL_ENV, "").strip(),
    )


def _secret_values_for_log() -> tuple[bytes, ...]:
    values = {
        value.encode("utf-8", errors="ignore")
        for name, value in os.environ.items()
        if _SECRET_ENV_NAME.search(name) and value
    }
    return tuple(sorted((value for value in values if value), key=len, reverse=True))


def _fetch_server_child_env() -> dict[str, str]:
    """Give pip/server only non-credential runtime settings."""

    env = os.environ.copy()
    for name in tuple(env):
        if _SECRET_ENV_NAME.search(name) or name in _PROXY_ENV_NAMES:
            env.pop(name, None)
    return env


def _redact_sidecar_log(data: bytes, secrets: tuple[bytes, ...]) -> bytes:
    redacted = data
    for secret in secrets:
        redacted = redacted.replace(secret, b"[REDACTED]")
    return redacted


class _FetchServerSidecar:
    """Own one fetch-server process for exactly one sandbox worker attempt."""

    def __init__(self, config: _FetchServerSidecarConfig, workspace: Path):
        self.config = config
        self.workspace = Path(workspace)
        self.process: subprocess.Popen[bytes] | None = None
        self._log_file = None
        self._log_thread: threading.Thread | None = None
        self._log_secrets: tuple[bytes, ...] = ()
        self._monitor_stop = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        self._failure: str | None = None
        self._dependency_dir: Path | None = None
        self.log_path = self.workspace / _FETCH_SERVER_LOG_NAME

    def start(self) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._log_secrets = _secret_values_for_log()
        self._log_file = self.log_path.open("wb")
        try:
            if self.config.install_dependencies:
                self._install_dependencies()
            child_env = _fetch_server_child_env()
            if self._dependency_dir is not None:
                existing_pythonpath = child_env.get("PYTHONPATH", "")
                child_env["PYTHONPATH"] = os.pathsep.join(
                    part
                    for part in (str(self._dependency_dir), existing_pythonpath)
                    if part
                )
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    "server.py",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(self.config.port),
                ],
                cwd=str(self.config.source_dir),
                env=child_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            if self.process.stdout is not None:
                self._log_thread = threading.Thread(
                    target=self._drain_process_log,
                    args=(self.process.stdout,),
                    name="fetch-server-log",
                    daemon=True,
                )
                self._log_thread.start()
            self._wait_until_healthy()
            if self.config.probe_url:
                self._run_functional_probe()
            self._monitor_thread = threading.Thread(
                target=self._monitor_liveness,
                name="fetch-server-health",
                daemon=True,
            )
            self._monitor_thread.start()
        except Exception:
            self.stop()
            raise

    def _install_dependencies(self) -> None:
        self._dependency_dir = Path(
            tempfile.mkdtemp(prefix="browse_comp_fetch_deps.")
        )
        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--target",
            str(self._dependency_dir),
            "-r",
            "requirements.lock.txt",
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=str(self.config.source_dir),
                env=_fetch_server_child_env(),
                capture_output=True,
                timeout=self.config.install_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            self._append_log(b"dependency installation timed out\n")
            if exc.stdout:
                self._append_log(_as_bytes(exc.stdout))
            if exc.stderr:
                self._append_log(_as_bytes(exc.stderr))
            raise RuntimeError(
                f"fetch server dependency installation timed out after "
                f"{self.config.install_timeout:g}s; see {self.log_path}"
            ) from None
        except OSError as exc:
            self._append_log(f"dependency installation failed: {exc.__class__.__name__}\n".encode())
            raise RuntimeError(
                f"fetch server dependency installation failed; see {self.log_path}"
            ) from None

        self._append_log(_as_bytes(completed.stdout))
        self._append_log(_as_bytes(completed.stderr))
        if completed.returncode != 0:
            raise RuntimeError(
                f"fetch server dependency installation failed with exit code "
                f"{completed.returncode}; see {self.log_path}"
            )

    def _append_log(self, data: bytes) -> None:
        if not data or self._log_file is None:
            return
        self._log_file.write(_redact_sidecar_log(data, self._log_secrets))
        self._log_file.flush()

    def _drain_process_log(self, stream) -> None:
        try:
            for chunk in iter(stream.readline, b""):
                self._append_log(_as_bytes(chunk))
        finally:
            try:
                stream.close()
            except OSError:
                pass

    def _wait_until_healthy(self) -> None:
        assert self.process is not None
        endpoint = f"http://127.0.0.1:{self.config.port}/health"
        deadline = time.monotonic() + self.config.startup_timeout
        while True:
            if self.process.poll() is not None:
                raise RuntimeError(
                    "fetch server exited before health check succeeded; "
                    f"see {self.log_path}"
                )
            try:
                if _fetch_server_health_ok(endpoint):
                    return
            except (OSError, ValueError, json.JSONDecodeError, urllib.error.URLError):
                pass
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"fetch server health check timed out after "
                    f"{self.config.startup_timeout:g}s; see {self.log_path}"
                )
            time.sleep(min(0.1, remaining))

    def _run_functional_probe(self) -> None:
        endpoint = f"http://127.0.0.1:{self.config.port}/fetch"
        payload = json.dumps(
            {"url": self.config.probe_url, "extractMode": "text"}
        ).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=payload,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            method="POST",
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(request, timeout=self.config.startup_timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(
                "fetch server functional probe request failed; "
                f"see {self.log_path}"
            ) from exc
        if (
            not isinstance(body, dict)
            or body.get("success") is not True
            or not str(body.get("data", "")).strip()
        ):
            raise RuntimeError(
                "fetch server functional probe returned no content; "
                f"see {self.log_path}"
            )

    def _monitor_liveness(self) -> None:
        """Observe process/health during Hermes without cancelling Hermes."""

        assert self.process is not None
        endpoint = f"http://127.0.0.1:{self.config.port}/health"
        consecutive_health_failures = 0
        while not self._monitor_stop.wait(_FETCH_SERVER_HEALTH_INTERVAL):
            if self.process.poll() is not None:
                self._failure = (
                    "fetch server exited during Hermes execution; "
                    f"see {self.log_path}"
                )
                return
            try:
                healthy = _fetch_server_health_ok(endpoint)
            except (OSError, ValueError, json.JSONDecodeError, urllib.error.URLError):
                healthy = False
            if healthy:
                consecutive_health_failures = 0
                continue
            consecutive_health_failures += 1
            if consecutive_health_failures >= 3:
                self._failure = (
                    "fetch server health check failed three consecutive times "
                    "during Hermes execution; "
                    f"see {self.log_path}"
                )
                return

    @property
    def failure(self) -> str | None:
        return self._failure

    def stop(self) -> None:
        self._monitor_stop.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=_FETCH_SERVER_SHUTDOWN_TIMEOUT)
            self._monitor_thread = None
        process = self.process
        self.process = None
        if process is not None:
            try:
                try:
                    # Signal the group even when the leader has already
                    # exited: a server may have left children behind.
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                if process.poll() is None:
                    try:
                        process.wait(timeout=_FETCH_SERVER_SHUTDOWN_TIMEOUT)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait(timeout=_FETCH_SERVER_SHUTDOWN_TIMEOUT)
                elif self._failure is None:
                    self._failure = (
                        "fetch server exited during Hermes execution; "
                        f"see {self.log_path}"
                    )
            finally:
                if self._log_thread is not None:
                    self._log_thread.join(timeout=_FETCH_SERVER_SHUTDOWN_TIMEOUT)
                    self._log_thread = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        if self._dependency_dir is not None:
            shutil.rmtree(self._dependency_dir, ignore_errors=True)
            self._dependency_dir = None


def _as_bytes(value: bytes | str | None) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    return str(value).encode("utf-8", errors="replace")


def _fetch_server_health_ok(endpoint: str) -> bool:
    request = urllib.request.Request(endpoint, headers={"Accept": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=1.0) as response:
        if getattr(response, "status", 200) != 200:
            return False
        payload = json.loads(response.read().decode("utf-8"))
    return isinstance(payload, dict) and str(payload.get("status", "")).lower() == "ok"


@contextlib.contextmanager
def fetch_server_sidecar(workspace: Path):
    """Start the optional fetch sidecar and always reap its process group."""

    config = _fetch_server_sidecar_config()
    if config is None:
        yield None
        return

    sidecar = _FetchServerSidecar(config, Path(workspace))
    sidecar.start()
    try:
        yield sidecar
    finally:
        sidecar.stop()
        if sidecar.failure is not None:
            raise RuntimeError(sidecar.failure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one BrowseComp Hermes sandbox attempt")
    parser.add_argument("--request", required=True, type=Path, help="Secret-free request JSON path")
    parser.add_argument("--result", required=True, type=Path, help="Result JSON path")
    parser.add_argument("--workspace", required=True, type=Path, help="Attempt workspace directory")
    parser.add_argument(
        "--cancel-file",
        type=Path,
        default=Path("/tmp/browse_comp_eval/cancel.json"),
        help="Host-written cancellation marker for endpoint failover",
    )
    parser.add_argument(
        "--api-key-env",
        default="HERMES_API_KEY",
        help="Environment variable containing the Hermes API key",
    )
    return parser


class _CancellationWatcher:
    def __init__(self, path: Path, client: HermesClient):
        self.path = Path(path)
        self.client = client
        self.reason = ""
        self.triggered = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_CancellationWatcher":
        self._thread = threading.Thread(
            target=self._run,
            name="endpoint-cancellation-watcher",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop.wait(0.2):
            if not self.triggered.is_set():
                try:
                    payload = json.loads(self.path.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    continue
                reason = payload.get("reason") if isinstance(payload, dict) else None
                self.reason = (
                    str(reason)
                    if isinstance(reason, str) and reason
                    else "model endpoint became unavailable"
                )
                self.triggered.set()
            self.client.cancel_active(self.reason)


def _write_result_atomic(path: Path, result: SandboxRunResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(result.to_json())
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _client_for_request(request: SandboxRunRequest, api_key: str) -> HermesClient:
    return HermesClient(
        HermesConfig(
            hermes_bin=request.hermes_bin,
            api_key=api_key,
            base_url=request.base_url,
            model=request.model,
            provider=request.provider,
            save_name=request.save_name,
            dataset=request.dataset,
            max_rounds=request.max_rounds,
            timeout_seconds=request.timeout_seconds,
            toolsets=list(request.toolsets),
            skills=list(request.skills),
            custom_system_prompt=request.custom_system_prompt,
            subagent_custom_system_prompt=request.subagent_custom_system_prompt,
            api_mode=request.api_mode,
            context_length=request.context_length,
            context_compression=request.context_compression,
            compression_threshold=request.compression_threshold,
            reasoning_effort=request.reasoning_effort,
            quiet=request.quiet,
            accept_hooks=request.accept_hooks,
            ignore_rules=request.ignore_rules,
            question_match_mode=request.question_match_mode,
            antihack_enabled=request.antihack_enabled,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            search_mode=request.search_mode,
        )
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        request = load_request(args.request)
    except (OSError, SandboxProtocolError) as exc:
        print(f"invalid sandbox request: {exc}", file=sys.stderr)
        return 2

    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        print(f"missing API key: set {args.api_key_env}", file=sys.stderr)
        return 2

    watcher: _CancellationWatcher | None = None
    try:
        _prepare_hermes_source()
        _install_uploaded_plugins()
        client = _client_for_request(request, api_key)
        with fetch_server_sidecar(args.workspace):
            with _CancellationWatcher(args.cancel_file, client) as watcher:
                protocol_result = client.run_request(request, args.workspace)
        if watcher.triggered.is_set():
            protocol_result = replace(
                protocol_result,
                status="failed",
                error=watcher.reason,
                attempt_consumed=False,
                failure_type="model_endpoint_unavailable",
                trace_complete=bool(protocol_result.history),
            )
        _write_result_atomic(args.result, protocol_result)
    except Exception as exc:
        failed = SandboxRunResult.failed(request, args.workspace, str(exc))
        if watcher is not None and watcher.triggered.is_set():
            failed = replace(
                failed,
                error=watcher.reason,
                attempt_consumed=False,
                failure_type="model_endpoint_unavailable",
                trace_complete=False,
            )
        else:
            failed = replace(
                failed,
                failure_type="sandbox_infrastructure",
                trace_complete=False,
            )
        # Infrastructure failures are precisely when sidecar/worker logs are
        # most useful.  Keep the manifest secret-free and let the trusted host
        # download the same workspace artifacts as it does for normal runs.
        failed = replace(
            failed,
            output_files=list_workspace_files(args.workspace),
        )
        try:
            _write_result_atomic(args.result, failed)
        except OSError as write_exc:
            print(
                f"sandbox execution failed: {exc}; could not write result: {write_exc}",
                file=sys.stderr,
            )
            return 1
        print(f"sandbox execution failed: {exc}", file=sys.stderr)
        return 1

    if protocol_result.status != "success":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
