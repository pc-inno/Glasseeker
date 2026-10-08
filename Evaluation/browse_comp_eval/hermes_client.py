from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from .dataset import DatasetItem, RunSpec
from .endpoint_health import (
    EndpointHealthManager,
    EndpointRoute,
    EndpointWatchdogAbort,
    EndpointWatchdogConfig,
)
from .endpoint_registry import (
    EndpointRegistryError,
    load_registry,
    model_entries,
)
from .mock_search_server import MockSearchServer
from .sandbox_protocol import SandboxRunRequest, SandboxRunResult
from .session_reader import (
    extract_final_assistant_message,
    extract_trace,
    load_session,
    session_snapshot_path,
)


CUSTOM_SYSTEM_PROMPT_FILENAME = "browse_comp_custom_system_prompt.txt"
SUBAGENT_CUSTOM_SYSTEM_PROMPT_FILENAME = (
    "browse_comp_subagent_custom_system_prompt.txt"
)

RESEARCH_GUIDANCE = """Research guidance:
① Before searching, decompose the question into independent clues/constraints. Watch for nested descriptions: several phrases may describe the same intermediate entity; resolve that entity first, then use it to find the final target.
② For each major clue, try 2-3 short query variants. Do not paste the full question or combine answer-field terms (e.g. \"cover credit\", \"matrix number\") with the whole description; start from the most concrete factual clue.
③ Search iteratively: broad clue -> add one constraint -> verify the remaining candidate against all clues -> only then look up the requested answer field.
④ Treat task-specific entity names as unverified until they appear in the question or a tool result. Start with clue terms, relationships, dates, locations, titles, and generic synonyms; avoid candidate-name searches until that exact name appeared externally.
⑤ Synthesis: discard memory-only candidates. Cross-check the candidate with 2 independent sources when available; if any condition mismatches, reject it and search again. Keep a compact evidence trail: hard constraints -> fact sheet -> candidate hypothesis -> verified/rejected clues.
Before returning the final answer, self-check every clue against the evidence."""

RESEARCH_TASK_PREAMBLE = """You are answering one hard multi-constraint research question.
Use only the tools made available in this session.
Decompose the question into independent clues, verify each clue against authoritative
web sources, and only then converge on the single entity that satisfies all clues."""

RESEARCH_ANSWER_REQUIREMENTS = """When you produce your answer, you must verify every clue individually, and for each verification you must provide a web-page citation as evidence.

Return your final answer clearly on the last line as: FINAL ANSWER: <answer>"""


def hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))).expanduser()


@dataclass(frozen=True)
class HermesConfig:
    hermes_bin: str
    api_key: str
    base_url: str
    model: str
    provider: str
    save_name: str
    dataset: str
    max_rounds: int
    timeout_seconds: int | None
    toolsets: list[str]
    skills: list[str]
    api_mode: str | None = None
    context_length: int | None = None
    context_compression: bool = True
    compression_threshold: float = 0.5
    reasoning_effort: str | None = None
    quiet: bool = True
    accept_hooks: bool = False
    ignore_rules: bool = False
    question_match_mode: str = "evaluation"
    antihack_enabled: bool = True
    max_tokens: int | None = None
    temperature: float | None = None
    search_mode: str = "external"
    custom_system_prompt: str | None = None
    subagent_custom_system_prompt: str | None = None

    def __post_init__(self) -> None:
        if self.custom_system_prompt is not None and not isinstance(
            self.custom_system_prompt, str
        ):
            raise TypeError("custom_system_prompt must be a string or None")
        if self.subagent_custom_system_prompt is not None and not isinstance(
            self.subagent_custom_system_prompt, str
        ):
            raise TypeError(
                "subagent_custom_system_prompt must be a string or None"
            )
        if self.max_tokens is not None and (
            not isinstance(self.max_tokens, int)
            or isinstance(self.max_tokens, bool)
            or self.max_tokens <= 0
        ):
            raise ValueError("max_tokens must be a positive integer or None")
        if self.temperature is not None and (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(float(self.temperature))
            or self.temperature < 0
        ):
            raise ValueError("temperature must be a finite non-negative number or None")
        if (
            isinstance(self.compression_threshold, bool)
            or not isinstance(self.compression_threshold, (int, float))
            or not math.isfinite(float(self.compression_threshold))
            or not 0 < self.compression_threshold <= 1
        ):
            raise ValueError("compression_threshold must be greater than 0 and at most 1")
        if self.search_mode not in {"external", "mock", "disabled"}:
            raise ValueError("search_mode must be one of: disabled, external, mock")


class HermesClient:
    def __init__(self, config: HermesConfig):
        self.config = config
        self._active_process_lock = threading.RLock()
        self._active_processes: dict[int, subprocess.Popen[str]] = {}
        self._process_cancel_reasons: dict[int, str] = {}
        self._process_cancel_forced: dict[int, bool] = {}
        self._cancellation_generation = 0
        self._latest_cancel_reason = ""

    @contextmanager
    def bind(self, excluded_generations: dict[int, int] | None = None):
        """Bind one complete run, including retries, to this client."""
        yield self

    def run(self, spec: RunSpec, workspace_dir: Path, attempt: int) -> dict[str, Any]:
        """Run one host-side attempt while retaining the historical result schema."""

        request = self.build_request(spec, attempt)
        result = self.run_request(request, workspace_dir).to_dict()
        # The reference answer is host-only and is appended after the protocol
        # boundary so it can never be uploaded to a sandbox.
        result["answer"] = spec.item.answer
        return result

    def cancel_active(self, reason: str) -> None:
        """Gracefully stop every active Hermes process owned by this route."""

        with self._active_process_lock:
            self._cancellation_generation += 1
            self._latest_cancel_reason = reason
            processes = list(self._active_processes.items())
            for token, _process in processes:
                self._process_cancel_reasons[token] = reason
                self._process_cancel_forced[token] = False
        if not processes:
            return

        for _token, process in processes:
            _signal_process_group(process, signal.SIGTERM)
        deadline = time.monotonic() + 10.0
        survivors = [process for _token, process in processes]
        while survivors and time.monotonic() < deadline:
            survivors = [process for process in survivors if process.poll() is None]
            if survivors:
                time.sleep(0.1)
        survivor_ids = {id(process) for process in survivors}
        with self._active_process_lock:
            for token in survivor_ids:
                self._process_cancel_forced[token] = True
        for process in survivors:
            _signal_process_group(process, signal.SIGKILL)

    def build_request(self, spec: RunSpec, attempt: int) -> SandboxRunRequest:
        return SandboxRunRequest.from_run_spec(spec, self.config, attempt)

    def _begin_execution(self) -> int:
        with self._active_process_lock:
            return self._cancellation_generation

    def _execution_cancel_reason(self, generation: int) -> str:
        with self._active_process_lock:
            if generation == self._cancellation_generation:
                return ""
            return self._latest_cancel_reason or "model endpoint became unavailable"

    def run_request(
        self, request: SandboxRunRequest, workspace_dir: Path
    ) -> SandboxRunResult:
        """Execute one secret-free request locally or inside a sandbox worker."""

        spec = RunSpec(
            question_id=request.question_id,
            run_id=request.run_id,
            repeat_index=request.repeat_index,
            item=DatasetItem(
                question_id=request.question_id,
                question=request.question,
                answer="",
                type=request.type,
                source_line=request.source_line,
            ),
        )
        if request.search_mode == "mock":
            with MockSearchServer() as mock_search_server:
                raw = self._run_spec(
                    spec,
                    workspace_dir,
                    request.attempt,
                    mock_search_server=mock_search_server,
                )
        else:
            raw = self._run_spec(spec, workspace_dir, request.attempt)
        raw.pop("answer", None)
        return SandboxRunResult(**raw)

    def _run_spec(
        self,
        spec: RunSpec,
        workspace_dir: Path,
        attempt: int,
        *,
        mock_search_server: MockSearchServer | None = None,
    ) -> dict[str, Any]:
        execution_generation = self._begin_execution()
        workspace_dir.mkdir(parents=True, exist_ok=True)
        profile = profile_name(self.config.save_name, spec.run_id, attempt)
        browser_cdp_url = self._browser_cdp_url_for_spec(spec, attempt=attempt)
        self._setup_profile(profile, browser_cdp_url=browser_cdp_url)
        agent_log_offset = profile_agent_log_size(profile)

        query = build_query(spec)
        cmd = self._build_chat_command(profile, query)
        env = self._build_env(
            spec,
            browser_cdp_url=browser_cdp_url,
            mock_search_server=mock_search_server,
        )

        started = time.time()
        stdout = ""
        stderr = ""
        returncode: int | None = None
        timed_out = False
        error = ""
        endpoint_cancel_reason = ""
        endpoint_cancel_forced = False
        process_token: int | None = None

        try:
            endpoint_cancel_reason = self._execution_cancel_reason(
                execution_generation
            )
            if endpoint_cancel_reason:
                raise EndpointWatchdogAbort(endpoint_cancel_reason)
            popen_kwargs: dict[str, Any] = {}
            if os.name == "posix":
                popen_kwargs["start_new_session"] = True
            proc = subprocess.Popen(
                cmd,
                cwd=str(workspace_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                **popen_kwargs,
            )
            process_token = id(proc)
            with self._active_process_lock:
                self._active_processes[process_token] = proc
                stale_reason = self._execution_cancel_reason(execution_generation)
                if stale_reason:
                    self._process_cancel_reasons[process_token] = stale_reason
                    self._process_cancel_forced[process_token] = False
            if stale_reason:
                _signal_process_group(proc, signal.SIGTERM)
            try:
                stdout, stderr = proc.communicate(timeout=self.config.timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                stdout = _subprocess_text(exc.stdout)
                stderr = _subprocess_text(exc.stderr)
                error = f"hermes timed out after {self.config.timeout_seconds}s"
                _signal_process_group(proc, signal.SIGTERM)
                try:
                    tail_stdout, tail_stderr = proc.communicate(timeout=2.0)
                except subprocess.TimeoutExpired:
                    _signal_process_group(proc, signal.SIGKILL)
                    tail_stdout, tail_stderr = proc.communicate()
                stdout = stdout or tail_stdout or ""
                stderr = stderr or tail_stderr or ""
            returncode = proc.returncode
        except Exception as exc:
            error = str(exc)
            if isinstance(exc, EndpointWatchdogAbort):
                endpoint_cancel_reason = str(exc)
        finally:
            if process_token is not None:
                with self._active_process_lock:
                    self._active_processes.pop(process_token, None)
                    endpoint_cancel_reason = self._process_cancel_reasons.pop(
                        process_token, ""
                    )
                    endpoint_cancel_forced = self._process_cancel_forced.pop(
                        process_token, False
                    )

        wall_seconds = round(time.time() - started, 2)
        session_id = extract_session_id(stdout, stderr) or find_session_id_from_storage(
            self.config.save_name,
            self.config.dataset,
            spec.question_id,
            started,
        )
        snapshot_path = None
        if session_id:
            snapshot_path = session_snapshot_path(
                env.get(
                    "HERMES_SESSION_STORAGE_ROOT",
                    str(hermes_home() / "sessions"),
                ),
                self.config.save_name,
                self.config.dataset,
                spec.question_id,
                session_id,
            )
        session = load_session(
            session_id,
            profile,
            snapshot_path=snapshot_path,
        )
        trace = extract_trace(session)
        model_response, final_assistant_error = extract_final_assistant_message(session)
        agent_log_tail = read_profile_agent_log_tail(profile)
        attempt_agent_log = read_profile_agent_log_since(profile, agent_log_offset)

        status = "success"
        failure_type = ""
        attempt_consumed = True
        trace_complete = True
        if endpoint_cancel_reason:
            status = "failed"
            failure_type = "model_endpoint_unavailable"
            attempt_consumed = False
            error = endpoint_cancel_reason
            trace_complete = bool(session) and not endpoint_cancel_forced
        elif timed_out or error or (returncode not in (0, None)):
            status = "failed"
            if not error:
                error = f"hermes exited with code {returncode}"
        if not model_response and status == "success":
            status = "failed"
            error = final_assistant_error or "empty final assistant response"
        mcp_registration_error = ""
        if "browsecomp-plus-bm25" in self.config.toolsets:
            mcp_registration_error = browsecomp_plus_mcp_registration_error(
                attempt_agent_log
            )
            if mcp_registration_error:
                status = "failed"
                error = "; ".join(
                    part for part in (error, mcp_registration_error) if part
                )

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
            "status": status,
            "error": error,
            "failure_type": failure_type,
            "attempt_consumed": attempt_consumed,
            "trace_complete": trace_complete,
            "returncode": returncode,
            "timed_out": timed_out,
            "model_response": model_response,
            "final_assistant_valid": bool(model_response),
            "final_assistant_error": final_assistant_error,
            "session_id": session_id,
            "profile": profile,
            "browser_cdp_url": browser_cdp_url,
            "workspace_dir": str(workspace_dir),
            "output_files": list_workspace_files(workspace_dir),
            "tool_calls": trace["tool_calls"],
            "rounds": trace["rounds"],
            "total_tokens_estimate": trace["total_tokens_estimate"],
            "duration_seconds": (
                wall_seconds if timed_out else (trace["duration_seconds"] or wall_seconds)
            ),
            "wall_seconds": wall_seconds,
            "stdout_tail": tail(stdout),
            "stderr_tail": tail(stderr),
            "agent_log_tail": agent_log_tail,
            "mcp_registration_error": mcp_registration_error,
            "history": [session] if session else [],
        }

    def _setup_profile(self, profile: str, browser_cdp_url: str = "") -> None:
        self._run_hermes(["profile", "create", profile, "--no-alias"], check=False)
        self._copy_default_profile_config(profile, browser_cdp_url=browser_cdp_url)
        if self.config.custom_system_prompt is not None:
            self._custom_system_prompt_path(profile).write_bytes(
                self.config.custom_system_prompt.encode("utf-8")
            )
        if self.config.subagent_custom_system_prompt is not None:
            self._subagent_custom_system_prompt_path(profile).write_bytes(
                self.config.subagent_custom_system_prompt.encode("utf-8")
            )

        effective_url = self.config.base_url
        provider_name = self.config.provider.strip().lower()
        if provider_name != "custom" and not provider_name.startswith("custom:") and effective_url:
            effective_url = effective_url.rstrip("/")
            if effective_url.endswith("/v1"):
                effective_url = effective_url[:-3]

        coding_context = os.environ.get(
            "BROWSECOMP_CODING_CONTEXT", "auto"
        ).strip().lower()
        if coding_context not in {"auto", "focus", "on", "off"}:
            raise ValueError(
                "BROWSECOMP_CODING_CONTEXT must be one of: auto, focus, on, off"
            )

        configs = [
            ("model.default", self.config.model),
            ("model.provider", self.config.provider),
            ("model.api_key", self.config.api_key),
            ("agent.coding_context", coding_context),
            ("terminal.cwd", "."),
            ("sessions.write_json_snapshots", "true"),
            ("code_execution.mode", "strict"),
            ("code_execution.filesystem_access", "none"),
            ("code_execution.network_access", "none"),
            ("code_execution.tool_access", "none"),
            ("delegation.child_timeout_seconds", "900"),
        ]
        if "browsecomp-plus-bm25" in self.config.toolsets:
            # Qwen models are excluded from Hermes' default "auto" tool-use
            # guidance. Enable it only for BrowseComp-Plus BM25 profiles so a
            # model cannot end the turn after merely describing a search.
            configs.append(("agent.tool_use_enforcement", "true"))
            # BM25 initialization competes with active searches when many
            # evaluation workers start together. Keep both the per-server
            # connection timeout and the first-tool-snapshot wait above the
            # observed 20-30 second saturation latency so tools cannot arrive
            # after the model has already started its turn.
            configs.append(("mcp_discovery_timeout", "120"))
            configs.append(
                ("mcp_servers.browsecomp-plus-bm25.connect_timeout", "120")
            )
            # FastMCP exposes the generic resources/prompts capability families
            # even though this server has no benchmark resources or prompts.
            # Hermes would otherwise register list/read/get utility schemas and
            # the model could call tools that the evaluation allowlist rejects.
            configs.append(
                ("mcp_servers.browsecomp-plus-bm25.tools.resources", "false")
            )
            configs.append(
                ("mcp_servers.browsecomp-plus-bm25.tools.prompts", "false")
            )
        if self.config.api_mode:
            configs.append(("model.api_mode", self.config.api_mode))
        if self.config.timeout_seconds is not None:
            configs.append(("terminal.timeout", str(max(120, self.config.timeout_seconds))))
        if self.config.context_length:
            configs.append(("model.context_length", str(self.config.context_length)))
        if self.config.max_tokens is not None:
            configs.append(("model.max_tokens", str(self.config.max_tokens)))
        configs.append(
            (
                "compression.enabled",
                "true" if self.config.context_compression else "false",
            )
        )
        configs.append(("compression.threshold", str(self.config.compression_threshold)))
        if self.config.reasoning_effort:
            # Hermes uses xhigh internally; DeepSeek V4 TokenHub maps it to
            # the provider's API value "max".
            profile_effort = (
                "xhigh" if self.config.reasoning_effort == "max"
                else self.config.reasoning_effort
            )
            configs.append(("agent.reasoning_effort", profile_effort))
        if effective_url:
            configs.append(("model.base_url", effective_url))
        if browser_cdp_url:
            configs.append(("browser.cdp_url", browser_cdp_url))
        if self.config.search_mode in {"external", "mock"}:
            configs.append(("web.search_backend", "search_server"))
        else:
            configs.append(("web.search_backend", "disabled"))
        fetch_server_url = (
            os.environ.get("FETCH_SERVER_BASE_URL", "").strip()
            or os.environ.get("FETCH_SERVER_URL", "").strip()
        )
        if fetch_server_url:
            configs.append(("web.extract_backend", "fetch_server"))

        web_extract_content_mode = (
            os.environ.get("WEB_EXTRACT_CONTENT_MODE", "")
            .strip()
            .lower()
            .replace("-", "_")
        )
        if web_extract_content_mode:
            valid_web_extract_content_modes = {"legacy", "prefix_overflow"}
            if web_extract_content_mode not in valid_web_extract_content_modes:
                choices = ", ".join(sorted(valid_web_extract_content_modes))
                raise ValueError(
                    "WEB_EXTRACT_CONTENT_MODE must be one of: " + choices
                )
            configs.append(
                ("web.extract_content_mode", web_extract_content_mode)
            )

        for env_name, config_key in (
            ("SUBAGENT_BASE_URL", "delegation.base_url"),
            ("SUBAGENT_API_KEY", "delegation.api_key"),
            ("SUBAGENT_MODEL", "delegation.model"),
            ("SUBAGENT_MAX_ITERATIONS", "delegation.max_iterations"),
            ("SUBAGENT_TIMEOUT_SECONDS", "delegation.child_timeout_seconds"),
        ):
            value = os.environ.get(env_name, "").strip()
            if value:
                configs.append((config_key, value))

        for key, value in configs:
            self._run_hermes(["--profile", profile, "config", "set", key, value])

        if self.config.temperature is not None and effective_url:
            self._add_temperature_provider_override(
                profile,
                base_url=effective_url,
                model=self.config.model,
                temperature=self.config.temperature,
            )

        # The guard is bundled with Hermes but deliberately opt-in. Enable it
        # only in the per-run BrowseComp profile; the plugin itself also checks
        # BROWSE_COMP_RUN_ID before changing any tool behavior. Experiments can
        # explicitly disable it through the evaluator's --no-antihack option.
        if self.config.antihack_enabled:
            self._run_hermes(
                ["--profile", profile, "plugins", "enable", "browse_comp_guard"]
            )

    def _add_temperature_provider_override(
        self,
        profile: str,
        *,
        base_url: str,
        model: str,
        temperature: float,
    ) -> None:
        profile_config = hermes_home() / "profiles" / profile / "config.yaml"
        if not profile_config.exists():
            raise RuntimeError(f"Hermes profile config missing: {profile_config}")
        provider = {
            "name": "browse-comp-eval-temperature",
            "base_url": base_url.rstrip("/"),
            "model": model,
            "api_mode": self.config.api_mode or "chat_completions",
            "extra_body": {"temperature": float(temperature)},
        }
        overlay = (
            "\ncustom_providers: "
            + json.dumps([provider], ensure_ascii=False)
            + "\nmcp_servers: {}\n"
        )
        with profile_config.open("a", encoding="utf-8") as stream:
            stream.write(overlay)

    def _copy_default_profile_config(self, profile: str, browser_cdp_url: str = "") -> None:
        default_home = hermes_home()
        profile_home = default_home / "profiles" / profile
        profile_home.mkdir(parents=True, exist_ok=True)
        for filename in ("config.yaml", ".env"):
            src = default_home / filename
            if src.exists():
                shutil.copy2(src, profile_home / filename)
        self._merge_profile_env(profile_home / ".env", browser_cdp_url=browser_cdp_url)

    def _run_hermes(self, args: list[str], check: bool = True) -> subprocess.CompletedProcess:
        proc = subprocess.run(
            [self.config.hermes_bin] + args,
            capture_output=True,
            text=True,
        )
        if check and proc.returncode != 0:
            raise RuntimeError(
                f"hermes command failed: {' '.join(args)}\n{tail(proc.stderr or proc.stdout)}"
            )
        return proc

    def _build_chat_command(self, profile: str, query: str) -> list[str]:
        cmd = [
            self.config.hermes_bin,
            "--profile",
            profile,
            "chat",
            "--query",
            query,
            "--yolo",
            "--max-turns",
            str(self.config.max_rounds),
            "--model",
            self.config.model,
            "--provider",
            self.config.provider,
            "--source",
            "browse_comp_eval",
        ]
        if self.config.quiet:
            cmd.append("--quiet")
        if self.config.toolsets:
            cmd.extend(["--toolsets", ",".join(self.config.toolsets)])
        else:
            cmd.extend(["--toolsets", "none"])
        if self.config.skills:
            cmd.extend(["--skills", ",".join(self.config.skills)])
        if self.config.accept_hooks:
            cmd.append("--accept-hooks")
        if self.config.ignore_rules:
            cmd.append("--ignore-rules")
        if self.config.custom_system_prompt is not None:
            cmd.extend(
                [
                    "--custom-system-prompt-file",
                    str(self._custom_system_prompt_path(profile)),
                ]
            )
        if self.config.subagent_custom_system_prompt is not None:
            cmd.extend(
                [
                    "--subagent-custom-system-prompt-file",
                    str(self._subagent_custom_system_prompt_path(profile)),
                ]
            )
        return cmd

    @staticmethod
    def _custom_system_prompt_path(profile: str) -> Path:
        return hermes_home() / "profiles" / profile / CUSTOM_SYSTEM_PROMPT_FILENAME

    @staticmethod
    def _subagent_custom_system_prompt_path(profile: str) -> Path:
        return (
            hermes_home()
            / "profiles"
            / profile
            / SUBAGENT_CUSTOM_SYSTEM_PROMPT_FILENAME
        )

    def _build_env(
        self,
        spec: RunSpec,
        browser_cdp_url: str = "",
        mock_search_server: MockSearchServer | None = None,
    ) -> dict[str, str]:
        env = os.environ.copy()
        env["ANTHROPIC_API_KEY"] = self.config.api_key
        env["OPENAI_API_KEY"] = self.config.api_key
        if self.config.base_url:
            env["ANTHROPIC_BASE_URL"] = self.config.base_url
            env["OPENAI_BASE_URL"] = self.config.base_url
        no_proxy = merge_no_proxy(env.get("NO_PROXY") or env.get("no_proxy") or "", self._no_proxy_additions())
        if no_proxy:
            env["NO_PROXY"] = no_proxy
            env["no_proxy"] = no_proxy
        if browser_cdp_url:
            env["BROWSER_CDP_URL"] = browser_cdp_url
        env["HERMES_SESSION_MODEL"] = self.config.save_name
        env["HERMES_SESSION_DATASET"] = self.config.dataset
        env.setdefault("HERMES_SESSION_STORAGE_ROOT", str(Path.home() / ".hermes" / "sessions"))
        env["HERMES_SESSION_TASK_TYPE"] = spec.item.type
        env["HERMES_SESSION_TASK_ID"] = spec.question_id
        env["HERMES_SESSION_TASK_NAME"] = spec.question_id
        env.pop("HERMES_TASK_ID", None)
        if "browsecomp-plus-bm25" in self.config.toolsets:
            # Keep the optional MCP SDK isolated from Hermes' shared venv and
            # expose it only to BrowseComp-Plus BM25 subprocesses.
            mcp_client = (
                Path(__file__).resolve().parents[1]
                / "data/browsecomp_plus/runtime/mcp_client_py311_126"
            )
            inherited_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = str(mcp_client) + (
                os.pathsep + inherited_pythonpath if inherited_pythonpath else ""
            )
        env["BROWSE_COMP_RUN_ID"] = spec.run_id
        env["BROWSE_COMP_QUESTION_MATCH_MODE"] = self.config.question_match_mode
        if self.config.question_match_mode == "evaluation":
            env["BROWSE_COMP_QUESTION"] = spec.item.question
        else:
            env.pop("BROWSE_COMP_QUESTION", None)
        if self.config.search_mode == "disabled":
            for name in (
                "SEARCH_SERVER_ENDPOINT",
                "SEARCH_SERVER_BASE_URL",
                "SEARCH_SERVER_URL",
                "SEARCH_SERVER_PORT",
                "SEARCH_SERVER_SCHEME",
                "SEARCH_SERVER_API_KEY",
            ):
                env.pop(name, None)
        if mock_search_server is not None:
            env["SEARCH_SERVER_ENDPOINT"] = mock_search_server.endpoint
            env["SEARCH_SERVER_API_KEY"] = mock_search_server.api_key
        return env

    def _merge_profile_env(self, env_path: Path, browser_cdp_url: str = "") -> None:
        additions = self._no_proxy_additions()
        if not env_path.exists():
            lines: list[str] = []
        else:
            lines = env_path.read_text(encoding="utf-8").splitlines()

        if not additions:
            return

        browser_cdp_url = browser_cdp_url or os.environ.get("BROWSER_CDP_URL", "").strip()
        seen_keys: set[str] = set()
        updated: list[str] = []
        for line in lines:
            key, sep, value = line.partition("=")
            if sep and key in {"NO_PROXY", "no_proxy"}:
                updated.append(f"{key}={merge_no_proxy(value, additions)}")
                seen_keys.add(key)
            elif sep and key == "BROWSER_CDP_URL":
                if browser_cdp_url:
                    updated.append(f"BROWSER_CDP_URL={browser_cdp_url}")
                    seen_keys.add(key)
            else:
                updated.append(line)

        merged = merge_no_proxy(os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or "", additions)
        if "NO_PROXY" not in seen_keys:
            updated.append(f"NO_PROXY={merged}")
        if "no_proxy" not in seen_keys:
            updated.append(f"no_proxy={merged}")
        if browser_cdp_url and "BROWSER_CDP_URL" not in seen_keys:
            updated.append(f"BROWSER_CDP_URL={browser_cdp_url}")

        env_path.write_text("\n".join(updated) + "\n", encoding="utf-8")

    def _browser_cdp_url_for_spec(self, spec: RunSpec, attempt: int = 1) -> str:
        raw = os.environ.get("BROWSER_CDP_URLS", "").strip()
        if raw:
            urls = [part.strip() for part in raw.split(",") if part.strip()]
            if urls:
                repeat_offset = max(0, int(spec.repeat_index or 1) - 1)
                attempt_offset = max(0, int(attempt or 1) - 1)
                model_offset = stable_offset(f"{self.config.save_name}:{self.config.model}", len(urls))
                index = spec.item.source_line - 1 + repeat_offset + attempt_offset + model_offset
                return urls[index % len(urls)]
        return os.environ.get("BROWSER_CDP_URL", "").strip()

    def _no_proxy_additions(self) -> list[str]:
        additions = ["localhost", "127.0.0.1", "::1"]
        host = endpoint_host(self.config.base_url)
        if host:
            additions.append(host)
        return additions


class _BoundEndpointClient:
    """One leased route with immutable health-generation metadata."""

    def __init__(
        self,
        client: HermesClient,
        route_index: int,
        generation: int,
        health_manager: EndpointHealthManager,
    ):
        self._client = client
        self.route_index = route_index
        self.generation = generation
        self._health_manager = health_manager
        self.config = client.config

    def run(self, spec: RunSpec, workspace_dir: Path, attempt: int) -> dict[str, Any]:
        if not self._lease_is_current():
            return self._interrupted_result(spec, workspace_dir, attempt)
        result = self._client.run(spec, workspace_dir, attempt)
        result["endpoint_route_index"] = self.route_index
        result["endpoint_generation"] = self.generation
        if not self._lease_is_current():
            result.update(
                {
                    "status": "failed",
                    "error": "model endpoint route was quarantined during the attempt",
                    "failure_type": "model_endpoint_unavailable",
                    "attempt_consumed": False,
                    "trace_complete": bool(result.get("history"))
                    and bool(result.get("trace_complete", True)),
                    "endpoint_health_evidence": self._route_health_evidence(),
                }
            )
        return result

    def _lease_is_current(self) -> bool:
        return (
            self._health_manager.is_available(self.route_index)
            and self._health_manager.generation(self.route_index) == self.generation
        )

    def _route_health_evidence(self) -> dict[str, object]:
        for row in self._health_manager.snapshot():
            if row.get("route_index") == self.route_index:
                return row
        return {"route_index": self.route_index}

    def _interrupted_result(
        self, spec: RunSpec, workspace_dir: Path, attempt: int
    ) -> dict[str, Any]:
        return {
            "question_id": spec.question_id,
            "repeat_index": spec.repeat_index,
            "run_id": spec.run_id,
            "attempt": attempt,
            "question": spec.item.question,
            "answer": spec.item.answer,
            "type": spec.item.type,
            "model": self.config.model,
            "provider": self.config.provider,
            "model_base_url": self.config.base_url,
            "save_name": self.config.save_name,
            "status": "failed",
            "error": "model endpoint route was quarantined before the attempt started",
            "failure_type": "model_endpoint_unavailable",
            "attempt_consumed": False,
            "trace_complete": True,
            "workspace_dir": str(workspace_dir),
            "tool_calls": {},
            "history": [],
            "endpoint_route_index": self.route_index,
            "endpoint_generation": self.generation,
            "endpoint_health_evidence": self._route_health_evidence(),
        }


class MultiEndpointHermesClient:
    """Least-loaded pool that leases one endpoint/key lane for a complete run."""

    def __init__(
        self,
        config: HermesConfig,
        base_urls: list[str],
        workers_per_endpoint: int | list[int],
        api_keys: list[str] | None = None,
        models: list[str] | None = None,
        key_base_urls: list[str] | None = None,
        api_modes: list[str | None] | None = None,
        client_factory: Callable[[HermesConfig], HermesClient] = HermesClient,
        watchdog_config: EndpointWatchdogConfig | None = None,
        watchdog_event_path: Path | None = None,
        endpoint_registry: Path | None = None,
        endpoint_registry_refresh_interval: float = 30.0,
    ):
        if not base_urls and endpoint_registry is None:
            raise ValueError("base_urls must not be empty")

        effective_watchdog_config = watchdog_config or EndpointWatchdogConfig(
            enabled=False
        )
        if endpoint_registry is not None and not effective_watchdog_config.enabled:
            raise ValueError("endpoint registry mode requires the endpoint watchdog")
        if endpoint_registry_refresh_interval <= 0:
            raise ValueError("endpoint_registry_refresh_interval must be positive")

        keys = api_keys or [config.api_key]
        if not keys or any(not key for key in keys):
            raise ValueError("api_keys must not contain empty values")
        lane_models = models or [config.model] * len(keys)
        if len(lane_models) != len(keys):
            raise ValueError("models and api_keys must have the same length")
        if any(not model for model in lane_models):
            raise ValueError("models must not contain empty values")
        lane_api_modes = api_modes or [config.api_mode] * len(keys)
        if len(lane_api_modes) != len(keys):
            raise ValueError("api_modes and api_keys must have the same length")
        if isinstance(workers_per_endpoint, int):
            key_worker_limits = [workers_per_endpoint] * len(keys)
        else:
            key_worker_limits = list(workers_per_endpoint)
        if len(key_worker_limits) != len(keys):
            raise ValueError("worker limits and api_keys must have the same length")
        if any(limit < 1 for limit in key_worker_limits):
            raise ValueError("worker limits must be >= 1")
        if endpoint_registry is not None:
            if base_urls or key_base_urls is not None:
                raise ValueError("endpoint registry cannot be combined with static base URLs")
            if len(keys) != 1 or len(lane_models) != 1 or len(lane_api_modes) != 1:
                raise ValueError("endpoint registry requires one model and one API key")
            if len(key_worker_limits) != 1:
                raise ValueError("endpoint registry requires one workers-per-endpoint value")
            route_definitions: list[tuple[str, str, str, str | None]] = []
            self.worker_limits: list[int] = []
        elif key_base_urls is not None:
            if len(key_base_urls) != len(keys):
                raise ValueError("key_base_urls and api_keys must have the same length")
            route_definitions = list(
                zip(key_base_urls, keys, lane_models, lane_api_modes)
            )
            self.worker_limits = key_worker_limits
        else:
            route_definitions = [
                (base_url, api_key, model, api_mode)
                for base_url in base_urls
                for api_key, model, api_mode in zip(keys, lane_models, lane_api_modes)
            ]
            self.worker_limits = key_worker_limits * len(base_urls)
        self._base_config = config
        self._client_factory = client_factory
        self._registry_path = (
            Path(endpoint_registry).resolve() if endpoint_registry is not None else None
        )
        self._registry_refresh_interval = endpoint_registry_refresh_interval
        self._registry_worker_limit = key_worker_limits[0]
        self._registry_stop_event = threading.Event()
        self._registry_thread: threading.Thread | None = None
        self._registry_route_indexes: dict[str, int] = {}
        self._registry_last_successful_read: str | None = None
        self._registry_last_valid_snapshot: list[dict[str, object]] = []
        self._registry_last_error = ""
        self.clients = [
            client_factory(
                HermesConfig(
                    **{
                        **config.__dict__,
                        "base_url": base_url,
                        "api_key": api_key,
                        "model": model,
                        "api_mode": api_mode,
                    }
                )
            )
            for base_url, api_key, model, api_mode in route_definitions
        ]
        self._active = [0] * len(self.clients)
        self._next_index = 0
        self._condition = threading.Condition()
        routes_for_health = [
            EndpointRoute(
                index=index,
                base_url=client.config.base_url,
                model=client.config.model,
                api_key=client.config.api_key,
            )
            for index, client in enumerate(self.clients)
        ]
        self.health_manager = EndpointHealthManager(
            routes_for_health,
            effective_watchdog_config,
            event_path=watchdog_event_path,
            on_change=self._notify_waiters,
            on_unhealthy=self._cancel_unhealthy_routes,
            on_abort=self._cancel_all_routes,
        )

    @contextmanager
    def bind(self, excluded_generations: dict[int, int] | None = None):
        excluded = excluded_generations or {}
        with self._condition:
            while True:
                self.health_manager.ensure_not_aborted()
                available = [
                    index
                    for index, active in enumerate(self._active)
                    if active < self.worker_limits[index]
                    and self.health_manager.is_available(index)
                    and self.health_manager.generation(index)
                    > excluded.get(index, -1)
                ]
                if available:
                    break
                self._condition.wait()
            minimum = min(self._active[index] for index in available)
            candidates = {index for index in available if self._active[index] == minimum}
            index = next(
                candidate
                for offset in range(len(self.clients))
                if (candidate := (self._next_index + offset) % len(self.clients)) in candidates
            )
            self._active[index] += 1
            self._next_index = (index + 1) % len(self.clients)

        try:
            yield _BoundEndpointClient(
                self.clients[index],
                index,
                self.health_manager.generation(index),
                self.health_manager,
            )
        finally:
            with self._condition:
                self._active[index] -= 1
                self._condition.notify_all()

    def start_watchdog(self) -> None:
        if self._registry_path is not None:
            self._refresh_endpoint_registry()
        self.health_manager.start()
        if self._registry_path is not None and self._registry_thread is None:
            self._registry_stop_event.clear()
            self._registry_thread = threading.Thread(
                target=self._registry_loop,
                name="browsecomp-endpoint-registry",
                daemon=True,
            )
            self._registry_thread.start()

    def stop_watchdog(self) -> None:
        self._registry_stop_event.set()
        registry_thread = self._registry_thread
        if registry_thread is not None and registry_thread is not threading.current_thread():
            registry_thread.join(timeout=2.0)
        self._registry_thread = None
        self.health_manager.stop()

    @property
    def watchdog_aborted(self) -> bool:
        return self.health_manager.aborted

    @property
    def watchdog_abort_reason(self) -> str:
        return self.health_manager.abort_reason

    def endpoint_health_snapshot(self) -> list[dict[str, object]]:
        return self.health_manager.snapshot()

    def endpoint_registry_state(self) -> dict[str, object]:
        with self._condition:
            return {
                "path": str(self._registry_path) if self._registry_path else None,
                "last_successful_read": self._registry_last_successful_read,
                "last_valid_endpoint_snapshot": [
                    dict(entry) for entry in self._registry_last_valid_snapshot
                ],
                "last_error": self._registry_last_error,
            }

    def scheduling_capacity(self) -> int:
        with self._condition:
            return sum(
                self.worker_limits[index]
                for index in range(len(self.clients))
                if self.health_manager.is_available(index)
            )

    def wait_for_capacity_change(self, timeout: float = 0.5) -> None:
        with self._condition:
            self.health_manager.ensure_not_aborted()
            self._condition.wait(timeout=max(0.0, timeout))
            self.health_manager.ensure_not_aborted()

    def _notify_waiters(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def _cancel_unhealthy_routes(self, route_indexes: list[int], reason: str) -> None:
        self._cancel_routes_async(route_indexes, reason)

    def _cancel_all_routes(self, reason: str) -> None:
        self._cancel_routes_async(list(range(len(self.clients))), reason)

    def _registry_loop(self) -> None:
        while not self._registry_stop_event.wait(
            self._registry_refresh_interval
        ):
            self._refresh_endpoint_registry()
            if self.health_manager.aborted:
                return

    def _refresh_endpoint_registry(self) -> bool:
        registry_path = self._registry_path
        if registry_path is None:
            return False
        try:
            registry = load_registry(registry_path)
            entries = model_entries(registry, self._base_config.model)
        except EndpointRegistryError as exc:
            message = str(exc)
            with self._condition:
                changed = message != self._registry_last_error
                self._registry_last_error = message
            if changed:
                self.health_manager.record_event(
                    "endpoint_registry_read_failed",
                    registry_path=str(registry_path),
                    reason=message,
                )
            return False

        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        desired_urls = [str(entry["base_url"]) for entry in entries]
        desired_set = set(desired_urls)
        added_routes: list[EndpointRoute] = []
        retired_indexes: list[int] = []
        with self._condition:
            current_urls = set(self._registry_route_indexes)
            for base_url in desired_urls:
                if base_url in current_urls:
                    continue
                route_index = len(self.clients)
                route_config = HermesConfig(
                    **{
                        **self._base_config.__dict__,
                        "base_url": base_url,
                    }
                )
                client = self._client_factory(route_config)
                self.clients.append(client)
                self.worker_limits.append(self._registry_worker_limit)
                self._active.append(0)
                self._registry_route_indexes[base_url] = route_index
                added_routes.append(
                    EndpointRoute(
                        index=route_index,
                        base_url=base_url,
                        model=route_config.model,
                        api_key=route_config.api_key,
                    )
                )
            for base_url in current_urls - desired_set:
                retired_indexes.append(self._registry_route_indexes.pop(base_url))
            self._registry_last_successful_read = timestamp
            snapshot_fields = (
                "base_url",
                "status",
                "last_checked_at",
                "last_healthy_at",
                "unreachable_since",
                "last_error",
            )
            self._registry_last_valid_snapshot = [
                {field: entry.get(field) for field in snapshot_fields}
                for entry in entries
            ]
            self._registry_last_error = ""
            if added_routes:
                self.health_manager.add_routes(added_routes)
            if retired_indexes:
                self.health_manager.retire_routes(
                    retired_indexes,
                    "endpoint removed from registry",
                )
            self._condition.notify_all()
        if added_routes or retired_indexes:
            self.health_manager.record_event(
                "endpoint_registry_refreshed",
                registry_path=str(registry_path),
                model=self._base_config.model,
                added_route_indexes=[route.index for route in added_routes],
                retired_route_indexes=retired_indexes,
                endpoint_count=len(desired_urls),
            )
        return True

    def _cancel_routes_async(self, route_indexes: list[int], reason: str) -> None:
        # Cancellation may spend up to ten seconds waiting for a graceful
        # Hermes/session flush.  Keep it off the probe thread so the all-down
        # deadline and subsequent recovery probes remain wall-clock accurate.
        for index in route_indexes:
            cancel = getattr(self.clients[index], "cancel_active", None)
            if not callable(cancel):
                continue

            def cancel_if_still_unavailable(
                route_index: int = index,
                route_cancel: Callable[[str], None] = cancel,
            ) -> None:
                if (
                    not self.health_manager.aborted
                    and self.health_manager.is_available(route_index)
                ):
                    return
                route_cancel(reason)

            threading.Thread(
                target=cancel_if_still_unavailable,
                name=f"endpoint-cancel-{index}",
                daemon=True,
            ).start()


def build_query(spec: RunSpec) -> str:
    if spec.item.type.startswith("gaia_validation"):
        task_label = "GAIA text-only evaluation task"
    elif spec.item.type.startswith("seal_0"):
        task_label = "SEAL-0 search-augmented factual evaluation task"
    elif spec.item.type.startswith("xbench_deepsearch"):
        task_label = "XBench DeepSearch evaluation task"
    elif spec.item.type.startswith("widesearch"):
        task_label = "WideSearch broad information-seeking evaluation task"
    else:
        task_label = "browse-comparison evaluation task"
    return (
        f"{RESEARCH_TASK_PREAMBLE}\n\n"
        f"{RESEARCH_GUIDANCE}\n\n"
        f"Question:\n{spec.item.question}\n\n"
        f"{RESEARCH_ANSWER_REQUIREMENTS}"
    )


def profile_name(save_name: str, run_id: str, attempt: int) -> str:
    raw = f"{save_name}_{run_id}_a{attempt}".lower()
    cleaned = re.sub(r"[^a-z0-9_-]", "-", raw).strip("-")
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    base = cleaned[:55].strip("-") or "browse-comp"
    return f"{base}-{digest}"[:64]


def endpoint_host(base_url: str) -> str:
    if not base_url:
        return ""
    try:
        return urlparse(base_url).hostname or ""
    except Exception:
        return ""


def merge_no_proxy(existing: str, additions: list[str]) -> str:
    values: list[str] = []
    seen: set[str] = set()
    for raw in [existing, ",".join(additions)]:
        for part in raw.split(","):
            value = part.strip()
            if not value:
                continue
            if value == "*":
                return "*"
            if value not in seen:
                seen.add(value)
                values.append(value)
    return ",".join(values)




def profile_agent_log_size(profile: str) -> int:
    log_path = hermes_home() / "profiles" / profile / "logs" / "agent.log"
    try:
        return log_path.stat().st_size
    except OSError:
        return 0


def read_profile_agent_log_since(profile: str, offset: int) -> str:
    """Read only log bytes written by the current attempt.

    Profiles can be reused when an evaluation is resumed, so inspecting the
    whole log can incorrectly attribute an older MCP failure to a new attempt.
    """
    log_path = hermes_home() / "profiles" / profile / "logs" / "agent.log"
    try:
        size = log_path.stat().st_size
        start = offset if 0 <= offset <= size else 0
        with log_path.open("rb") as handle:
            handle.seek(start)
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


MCP_REGISTRATION_SUMMARY_RE = re.compile(
    r"MCP: registered\s+(\d+)\s+tool\(s\)\s+from\s+(\d+)\s+server\(s\)"
    r"(?:\s+\((\d+)\s+failed\))?"
)


def browsecomp_plus_mcp_registration_error(agent_log: str) -> str:
    """Return an infrastructure error when the two-tool BM25 MCP did not load."""
    if not agent_log:
        return ""

    if re.search(
        r"Failed to connect to MCP server ['\"]browsecomp-plus-bm25['\"]",
        agent_log,
    ):
        return "BrowseComp-Plus BM25 MCP registration failed"

    summaries = MCP_REGISTRATION_SUMMARY_RE.findall(agent_log)
    if not summaries:
        return ""

    tool_count_text, server_count_text, failed_count_text = summaries[-1]
    tool_count = int(tool_count_text)
    server_count = int(server_count_text)
    failed_count = int(failed_count_text or 0)
    if failed_count or tool_count != 2 or server_count != 1:
        return (
            "BrowseComp-Plus BM25 MCP registration incomplete: expected "
            f"2 tools from 1 server, got {tool_count} tools from "
            f"{server_count} servers ({failed_count} failed)"
        )
    return ""


def stable_offset(value: str, modulo: int) -> int:
    if modulo <= 1:
        return 0
    digest = hashlib.sha1(str(value or "").encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def read_profile_agent_log_tail(profile: str, limit: int = 8000) -> str:
    log_path = hermes_home() / "profiles" / profile / "logs" / "agent.log"
    if not log_path.exists():
        return ""
    try:
        return tail(log_path.read_text(encoding="utf-8", errors="replace"), limit)
    except Exception:
        return ""


def find_session_id_from_storage(save_name: str, dataset: str, question_id: str, started: float) -> str | None:
    root = Path(os.environ.get("HERMES_SESSION_STORAGE_ROOT", str(hermes_home() / "sessions")))
    if root.name != "sessions":
        root = root / "sessions"
    session_dir = root / save_name / dataset / question_id / "main"
    if not session_dir.exists():
        return None

    newest: Path | None = None
    newest_mtime = 0.0
    for path in session_dir.glob("session_*.json"):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime < started - 5:
            continue
        if mtime > newest_mtime:
            newest = path
            newest_mtime = mtime

    if newest is None:
        return None
    name = newest.name
    if not name.startswith("session_") or not name.endswith(".json"):
        return None
    return name[len("session_") : -len(".json")]


def extract_session_id(stdout: str, stderr: str) -> str | None:
    combined = f"{stderr}\n{stdout}"
    for pattern in (r"session_id:\s*(\S+)", r"\bsession=([0-9A-Za-z_-]+)\b"):
        match = re.search(pattern, combined)
        if match:
            return match.group(1)
    return None


def tail(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _subprocess_text(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _signal_process_group(process: subprocess.Popen[str], signum: int) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signum)
        else:
            process.send_signal(signum)
    except (OSError, ProcessLookupError):
        pass


def list_workspace_files(workspace_dir: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    if not workspace_dir.exists():
        return files
    for path in sorted(workspace_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(workspace_dir).as_posix()
        files.append({"path": rel, "size": path.stat().st_size})
    return files
