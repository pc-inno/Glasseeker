"""The stdlib-only protocol exchanged with a single BrowseComp worker.

The request deliberately contains the task question but no reference answer or
credential.  A worker receives one request and produces one result; batching,
retries, and offline judging stay outside this boundary.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar


# Version 7 merges mm_dev's optional main-agent/subagent custom system prompt
# overrides on the request with qzx's attempt-tracking fields
# (attempt_consumed, failure_type, trace_complete) on the result. Keep the
# version check strict so a worker never silently interprets a request from a
# different evaluator build.
PROTOCOL_VERSION = 7


class SandboxProtocolError(ValueError):
    """Raised when a sandbox request or result is not valid protocol JSON."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_string(name: str, value: Any, *, allow_none: bool = False) -> None:
    if allow_none and value is None:
        return
    if not isinstance(value, str):
        raise SandboxProtocolError(f"{name} must be a string")


def _require_int(name: str, value: Any, *, allow_none: bool = False) -> None:
    if allow_none and value is None:
        return
    if not _is_int(value):
        raise SandboxProtocolError(f"{name} must be an integer")


def _require_bool(name: str, value: Any) -> None:
    if not isinstance(value, bool):
        raise SandboxProtocolError(f"{name} must be a boolean")


_SEARCH_MODES = frozenset(("external", "mock", "disabled"))


def _require_search_mode(name: str, value: Any) -> None:
    if not isinstance(value, str) or value not in _SEARCH_MODES:
        choices = ", ".join(sorted(_SEARCH_MODES))
        raise SandboxProtocolError(f"{name} must be one of: {choices}")


def _require_positive_int(name: str, value: Any, *, allow_none: bool = False) -> None:
    if allow_none and value is None:
        return
    if not _is_int(value) or value <= 0:
        raise SandboxProtocolError(f"{name} must be a positive integer")


def _require_temperature(name: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SandboxProtocolError(f"{name} must be a finite non-negative number or null")
    if not math.isfinite(float(value)) or value < 0:
        raise SandboxProtocolError(f"{name} must be a finite non-negative number or null")


def _require_ratio(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SandboxProtocolError(f"{name} must be greater than 0 and at most 1")
    if not math.isfinite(float(value)) or not 0 < value <= 1:
        raise SandboxProtocolError(f"{name} must be greater than 0 and at most 1")


def _field_names(cls: type[Any]) -> tuple[str, ...]:
    return tuple(field.name for field in fields(cls))


def _parse_payload(
    cls: type[Any], payload: Mapping[str, Any], required: set[str]
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise SandboxProtocolError("protocol payload must be a JSON object")

    allowed = set(_field_names(cls))
    unknown = set(payload) - allowed
    if unknown:
        names = ", ".join(sorted(str(name) for name in unknown))
        raise SandboxProtocolError(f"unknown protocol field(s): {names}")

    missing = required - set(payload)
    if missing:
        names = ", ".join(sorted(missing))
        raise SandboxProtocolError(f"missing protocol field(s): {names}")

    version = payload.get("protocol_version", PROTOCOL_VERSION)
    if not _is_int(version) or version != PROTOCOL_VERSION:
        raise SandboxProtocolError(
            f"unsupported protocol_version: {version!r}; expected {PROTOCOL_VERSION}"
        )
    return dict(payload)


@dataclass(frozen=True)
class SandboxRunRequest:
    """All non-secret inputs required for one Hermes attempt."""

    question_id: str
    run_id: str
    repeat_index: int
    question: str
    type: str
    source_line: int
    attempt: int
    hermes_bin: str
    base_url: str
    model: str
    provider: str
    save_name: str
    dataset: str
    max_rounds: int
    timeout_seconds: int | None
    toolsets: list[str]
    skills: list[str]
    custom_system_prompt: str | None
    subagent_custom_system_prompt: str | None
    context_length: int | None
    api_mode: str | None
    context_compression: bool
    compression_threshold: float
    reasoning_effort: str | None
    quiet: bool
    accept_hooks: bool
    ignore_rules: bool
    question_match_mode: str
    antihack_enabled: bool
    max_tokens: int | None = None
    temperature: float | None = None
    search_mode: str = "external"
    protocol_version: int = PROTOCOL_VERSION

    _REQUIRED_FIELDS: ClassVar[set[str]] = {
        "question_id",
        "run_id",
        "repeat_index",
        "question",
        "type",
        "source_line",
        "attempt",
        "hermes_bin",
        "base_url",
        "model",
        "provider",
        "save_name",
        "dataset",
        "max_rounds",
        "timeout_seconds",
        "toolsets",
        "skills",
        "custom_system_prompt",
        "subagent_custom_system_prompt",
        "context_length",
        "api_mode",
        "context_compression",
        "compression_threshold",
        "reasoning_effort",
        "quiet",
        "accept_hooks",
        "ignore_rules",
        "question_match_mode",
        "antihack_enabled",
        "max_tokens",
        "temperature",
        "search_mode",
        "protocol_version",
    }

    def __post_init__(self) -> None:
        for name in (
            "question_id",
            "run_id",
            "question",
            "type",
            "hermes_bin",
            "base_url",
            "model",
            "provider",
            "save_name",
            "dataset",
        ):
            _require_string(name, getattr(self, name))
        for name in ("repeat_index", "source_line", "attempt", "max_rounds"):
            _require_int(name, getattr(self, name))
        _require_int("timeout_seconds", self.timeout_seconds, allow_none=True)
        _require_int("context_length", self.context_length, allow_none=True)
        _require_string("api_mode", self.api_mode, allow_none=True)
        _require_string(
            "custom_system_prompt", self.custom_system_prompt, allow_none=True
        )
        _require_string(
            "subagent_custom_system_prompt",
            self.subagent_custom_system_prompt,
            allow_none=True,
        )
        if self.api_mode not in (None, "chat_completions", "codex_responses"):
            raise SandboxProtocolError(
                "api_mode must be chat_completions, codex_responses, or null"
            )
        _require_string("reasoning_effort", self.reasoning_effort, allow_none=True)
        if self.reasoning_effort not in (None, "low", "medium", "high", "max"):
            raise SandboxProtocolError(
                "reasoning_effort must be low, medium, high, max, or null"
            )
        _require_string("question_match_mode", self.question_match_mode)
        if self.question_match_mode not in ("evaluation", "off"):
            raise SandboxProtocolError(
                "question_match_mode must be evaluation or off"
            )
        _require_positive_int("max_tokens", self.max_tokens, allow_none=True)
        _require_temperature("temperature", self.temperature)
        _require_ratio("compression_threshold", self.compression_threshold)
        _require_search_mode("search_mode", self.search_mode)
        for name in ("toolsets", "skills"):
            value = getattr(self, name)
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise SandboxProtocolError(f"{name} must be a list of strings")
        for name in (
            "context_compression",
            "quiet",
            "accept_hooks",
            "ignore_rules",
            "antihack_enabled",
        ):
            _require_bool(name, getattr(self, name))
        if not _is_int(self.protocol_version) or self.protocol_version != PROTOCOL_VERSION:
            raise SandboxProtocolError(
                f"unsupported protocol_version: {self.protocol_version!r}; expected {PROTOCOL_VERSION}"
            )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SandboxRunRequest":
        values = _parse_payload(cls, payload, cls._REQUIRED_FIELDS)
        return cls(**values)

    @classmethod
    def from_json(cls, text: str) -> "SandboxRunRequest":
        try:
            payload = json.loads(text)
        except (TypeError, json.JSONDecodeError) as exc:
            raise SandboxProtocolError(f"invalid request JSON: {exc}") from exc
        return cls.from_dict(payload)

    @classmethod
    def from_run_spec(cls, spec: Any, config: Any, attempt: int) -> "SandboxRunRequest":
        """Build a request while intentionally ignoring ``spec.item.answer``."""

        item = spec.item
        return cls(
            question_id=spec.question_id,
            run_id=spec.run_id,
            repeat_index=spec.repeat_index,
            question=item.question,
            type=item.type,
            source_line=item.source_line,
            attempt=attempt,
            hermes_bin=config.hermes_bin,
            base_url=config.base_url,
            model=config.model,
            provider=config.provider,
            save_name=config.save_name,
            dataset=config.dataset,
            max_rounds=config.max_rounds,
            timeout_seconds=config.timeout_seconds,
            toolsets=list(config.toolsets),
            skills=list(config.skills),
            custom_system_prompt=config.custom_system_prompt,
            subagent_custom_system_prompt=config.subagent_custom_system_prompt,
            context_length=config.context_length,
            api_mode=config.api_mode,
            context_compression=config.context_compression,
            compression_threshold=config.compression_threshold,
            reasoning_effort=config.reasoning_effort,
            quiet=config.quiet,
            accept_hooks=config.accept_hooks,
            ignore_rules=config.ignore_rules,
            question_match_mode=config.question_match_mode,
            antihack_enabled=config.antihack_enabled,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            search_mode=config.search_mode,
        )

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in _field_names(type(self))}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True)
class SandboxRunResult:
    """The result of one attempt, with no reference answer field."""

    question_id: str
    repeat_index: int
    run_id: str
    attempt: int
    question: str
    type: str
    model: str
    provider: str
    model_base_url: str
    save_name: str
    status: str
    attempt_consumed: bool = True
    failure_type: str = ""
    trace_complete: bool = True
    error: str = ""
    returncode: int | None = None
    timed_out: bool = False
    model_response: str = ""
    final_assistant_valid: bool = False
    final_assistant_error: str = ""
    session_id: str | None = None
    profile: str = ""
    browser_cdp_url: str = ""
    workspace_dir: str = ""
    output_files: list[dict[str, Any]] | list[Any] = field(default_factory=list)
    tool_calls: dict[str, Any] = field(default_factory=dict)
    rounds: int = 0
    total_tokens_estimate: int = 0
    duration_seconds: float = 0.0
    wall_seconds: float = 0.0
    stdout_tail: str = ""
    stderr_tail: str = ""
    agent_log_tail: str = ""
    mcp_registration_error: str = ""
    history: list[Any] = field(default_factory=list)
    protocol_version: int = PROTOCOL_VERSION

    _REQUIRED_FIELDS: ClassVar[set[str]] = {
        "question_id",
        "repeat_index",
        "run_id",
        "attempt",
        "question",
        "type",
        "model",
        "provider",
        "model_base_url",
        "save_name",
        "status",
        "attempt_consumed",
        "failure_type",
        "trace_complete",
        "protocol_version",
    }

    def __post_init__(self) -> None:
        for name in (
            "question_id",
            "run_id",
            "question",
            "type",
            "model",
            "provider",
            "model_base_url",
            "save_name",
            "status",
        ):
            _require_string(name, getattr(self, name))
        _require_int("repeat_index", self.repeat_index)
        _require_int("attempt", self.attempt)
        _require_string("error", self.error)
        _require_bool("attempt_consumed", self.attempt_consumed)
        _require_string("failure_type", self.failure_type)
        _require_bool("trace_complete", self.trace_complete)
        _require_int("returncode", self.returncode, allow_none=True)
        _require_bool("timed_out", self.timed_out)
        _require_string("model_response", self.model_response)
        _require_bool("final_assistant_valid", self.final_assistant_valid)
        _require_string("final_assistant_error", self.final_assistant_error)
        _require_string("session_id", self.session_id, allow_none=True)
        for name in (
            "profile",
            "browser_cdp_url",
            "workspace_dir",
            "stdout_tail",
            "stderr_tail",
            "agent_log_tail",
            "mcp_registration_error",
        ):
            _require_string(name, getattr(self, name))
        if not isinstance(self.output_files, list):
            raise SandboxProtocolError("output_files must be a JSON array")
        if not isinstance(self.tool_calls, dict):
            raise SandboxProtocolError("tool_calls must be a JSON object")
        if not isinstance(self.history, list):
            raise SandboxProtocolError("history must be a JSON array")
        _require_int("rounds", self.rounds)
        _require_int("total_tokens_estimate", self.total_tokens_estimate)
        if not isinstance(self.duration_seconds, (int, float)) or isinstance(
            self.duration_seconds, bool
        ):
            raise SandboxProtocolError("duration_seconds must be a number")
        if not isinstance(self.wall_seconds, (int, float)) or isinstance(
            self.wall_seconds, bool
        ):
            raise SandboxProtocolError("wall_seconds must be a number")
        if not _is_int(self.protocol_version) or self.protocol_version != PROTOCOL_VERSION:
            raise SandboxProtocolError(
                f"unsupported protocol_version: {self.protocol_version!r}; expected {PROTOCOL_VERSION}"
            )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SandboxRunResult":
        values = _parse_payload(cls, payload, cls._REQUIRED_FIELDS)
        return cls(**values)

    @classmethod
    def from_json(cls, text: str) -> "SandboxRunResult":
        try:
            payload = json.loads(text)
        except (TypeError, json.JSONDecodeError) as exc:
            raise SandboxProtocolError(f"invalid result JSON: {exc}") from exc
        return cls.from_dict(payload)

    @classmethod
    def failed(
        cls, request: SandboxRunRequest, workspace_dir: Path | str, error: str
    ) -> "SandboxRunResult":
        return cls(
            question_id=request.question_id,
            repeat_index=request.repeat_index,
            run_id=request.run_id,
            attempt=request.attempt,
            question=request.question,
            type=request.type,
            model=request.model,
            provider=request.provider,
            model_base_url=request.base_url,
            save_name=request.save_name,
            status="failed",
            error=str(error),
            workspace_dir=str(workspace_dir),
        )

    def to_dict(self) -> dict[str, Any]:
        # Enumerating dataclass fields means an accidentally added ``answer``
        # attribute cannot silently cross this boundary without changing the
        # protocol schema and its tests.
        return {name: getattr(self, name) for name in _field_names(type(self))}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


def load_request(path: str | Path) -> SandboxRunRequest:
    return SandboxRunRequest.from_json(Path(path).read_text(encoding="utf-8"))
