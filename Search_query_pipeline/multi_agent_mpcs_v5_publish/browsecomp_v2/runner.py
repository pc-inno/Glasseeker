from __future__ import annotations

import json
import os
import re
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .config import AgentConfig
from .memory_guard import MemoryGuard
from .rate_limit_guard import RateLimitGuard


_HERMES_RUNTIME_HOME_LOCK = threading.Lock()
_HERMES_INPROCESS_WORKSPACE_LOCK = threading.Lock()
_HERMES_SHARED_ENTRIES = (
    "SOUL.md",
    "hooks",
    "optional-mcps",
    "optional-skills",
    "packages",
    "plugins",
    "skills",
    "sysroot",
)
_HERMES_SHARED_NODE_ENTRIES = ("bin", "browsers", "lib", "runtime-debs")
_HERMES_LOCAL_CREDENTIAL_FILES = (".env", ".anthropic_oauth.json", "auth.json")
_HERMES_LOCAL_STATE_DIRS = (
    "audio_cache",
    "cache",
    "cron",
    "image_cache",
    "logs",
    "memories",
    "pairing",
    "sessions",
    "workspace",
)
_HERMES_BROWSER_ENV_OVERRIDES = (
    ("BROWSER_CDP_URL", "V2_HERMES_BROWSER_CDP_URL"),
    ("AGENT_BROWSER_AUTO_CONNECT", "V2_HERMES_AGENT_BROWSER_AUTO_CONNECT"),
)
_HERMES_JSON_MESSAGE_COLUMNS = {
    "tool_calls",
    "reasoning_details",
    "codex_reasoning_items",
    "codex_message_items",
}
_HERMES_JSON_SESSION_COLUMNS = {"model_config", "handoff_state"}
_HERMES_CONTENT_JSON_PREFIX = "\x00json:"
_P7_SESSION_TRACE_KEY = "_p7_session_trace"


def _decode_hermes_json(value: Any) -> Any:
    if not isinstance(value, str) or not value:
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def _hermes_timestamp(value: Any) -> Any:
    if value in (None, ""):
        return value
    try:
        return datetime.fromtimestamp(float(value), timezone.utc).isoformat()
    except (TypeError, ValueError, OverflowError, OSError):
        return value


def _load_hermes_session_tree(db_path: Path, session_id: str) -> list[Dict[str, Any]]:
    """Read one complete Hermes lineage, including delegated descendants."""
    if not session_id or not db_path.is_file():
        return []
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    try:
        session_columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(sessions)")
        }
        if "parent_session_id" in session_columns:
            root_id = session_id
            seen: set[str] = set()
            while root_id and root_id not in seen:
                seen.add(root_id)
                row = connection.execute(
                    "SELECT parent_session_id FROM sessions WHERE id = ?", (root_id,)
                ).fetchone()
                parent_id = str(row[0] or "") if row else ""
                if not parent_id:
                    break
                root_id = parent_id
            session_rows = connection.execute(
                """
                WITH RECURSIVE session_tree(id, trace_depth) AS (
                    SELECT id, 0 FROM sessions WHERE id = ?
                    UNION ALL
                    SELECT child.id, session_tree.trace_depth + 1
                    FROM sessions AS child
                    JOIN session_tree ON child.parent_session_id = session_tree.id
                )
                SELECT sessions.*, session_tree.trace_depth
                FROM session_tree
                JOIN sessions ON sessions.id = session_tree.id
                ORDER BY session_tree.trace_depth, sessions.started_at, sessions.id
                """,
                (root_id,),
            ).fetchall()
        else:
            session_rows = connection.execute(
                "SELECT sessions.*, 0 AS trace_depth FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchall()

        sessions: list[Dict[str, Any]] = []
        for raw_session in session_rows:
            session = dict(raw_session)
            current_id = str(session.get("id") or "")
            for column in _HERMES_JSON_SESSION_COLUMNS:
                if column in session:
                    session[column] = _decode_hermes_json(session[column])
            for column in ("started_at", "ended_at"):
                if column in session:
                    session[column] = _hermes_timestamp(session[column])

            messages: list[Dict[str, Any]] = []
            reasoning_summaries: list[Dict[str, Any]] = []
            for raw_message in connection.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp, id",
                (current_id,),
            ).fetchall():
                message = dict(raw_message)
                content = message.get("content")
                if isinstance(content, str) and content.startswith(
                    _HERMES_CONTENT_JSON_PREFIX
                ):
                    message["content"] = _decode_hermes_json(
                        content[len(_HERMES_CONTENT_JSON_PREFIX) :]
                    )
                for column in _HERMES_JSON_MESSAGE_COLUMNS:
                    if column in message:
                        message[column] = _decode_hermes_json(message[column])
                if "timestamp" in message:
                    message["timestamp"] = _hermes_timestamp(message["timestamp"])

                reasoning = message.get("reasoning")
                if isinstance(reasoning, str) and reasoning.strip():
                    message["reasoning_summary"] = reasoning.strip()
                    reasoning_summaries.append(
                        {
                            "message_id": message.get("id"),
                            "timestamp": message.get("timestamp"),
                            "text": reasoning.strip(),
                        }
                    )
                if message.get("codex_reasoning_items") not in (None, "", []):
                    message["responses_reasoning_items"] = message[
                        "codex_reasoning_items"
                    ]
                if message.get("codex_message_items") not in (None, "", []):
                    message["responses_message_items"] = message["codex_message_items"]
                messages.append(message)

            session["session_id"] = current_id
            session["messages"] = messages
            session["reasoning_summaries"] = reasoning_summaries
            sessions.append(session)
        return sessions
    finally:
        connection.close()


def _session_roles(sessions: list[Dict[str, Any]]) -> Dict[str, str]:
    roles: Dict[str, str] = {}
    for session in sessions:
        current_id = str(session.get("session_id") or "")
        model_config = session.get("model_config")
        source = str(session.get("source") or "").strip().lower()
        delegated = bool(
            isinstance(model_config, dict) and model_config.get("_delegate_from")
        ) or source in {"subagent", "delegate", "delegation"}
        if delegated:
            roles[current_id] = "subagent"
        elif not session.get("parent_session_id"):
            roles[current_id] = "main"

    for _ in range(len(sessions) + 1):
        changed = False
        for session in sessions:
            current_id = str(session.get("session_id") or "")
            if current_id in roles:
                continue
            parent_role = roles.get(str(session.get("parent_session_id") or ""))
            if parent_role:
                base = "subagent" if parent_role.startswith("subagent") else "main"
                roles[current_id] = f"{base}_continuation"
                changed = True
        if not changed:
            break
    return roles


def _build_session_trace(db_path: Path, session_id: str) -> Dict[str, Any]:
    sessions = _load_hermes_session_tree(db_path, session_id)
    if not sessions:
        raise RuntimeError(f"Hermes solver session was not persisted: {session_id}")
    roles = _session_roles(sessions)
    summaries = [
        {"session_id": session.get("session_id"), **summary}
        for session in sessions
        for summary in session.get("reasoning_summaries", [])
    ]
    subagent_sessions = [
        session
        for session in sessions
        if roles.get(str(session.get("session_id") or ""), "").startswith("subagent")
    ]
    subagent_roots = [
        session
        for session in subagent_sessions
        if roles.get(str(session.get("session_id") or "")) == "subagent"
    ]
    warnings: list[str] = []
    for session in sessions:
        current_id = str(session.get("session_id") or "unknown")
        messages = session.get("messages")
        messages = messages if isinstance(messages, list) else []
        expected_count = session.get("message_count")
        if isinstance(expected_count, int) and expected_count != len(messages):
            warnings.append(
                f"session {current_id} persisted {len(messages)}/{expected_count} messages"
            )
        role = roles.get(current_id, "")
        if role.startswith(("main", "subagent")) and not any(
            isinstance(message, dict) and message.get("role") == "assistant"
            for message in messages
        ):
            warnings.append(f"session {current_id} has no persisted assistant turn")
    complete = not warnings
    return {
        "schema_version": 1,
        "root_session_id": str(sessions[0].get("session_id") or session_id),
        "primary_session_id": session_id,
        "sessions": sessions,
        "session_roles": roles,
        "reasoning_summaries": summaries,
        "delegation": {
            "subagent_session_count": len(subagent_sessions),
            "subagent_root_session_count": len(subagent_roots),
            "complete": complete,
            "warnings": list(warnings),
        },
        "complete": complete,
        "warnings": warnings,
    }


def _attach_hermes_session_trace(raw_response: str, db_path: Path) -> str:
    envelope: Optional[Dict[str, Any]] = None
    for candidate in _json_object_candidates(raw_response):
        try:
            parsed = json.loads(candidate)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(parsed, dict) and parsed.get("session_id"):
            envelope = parsed
            break
    if envelope is None:
        raise RuntimeError("Hermes solver response did not include a session_id")
    envelope[_P7_SESSION_TRACE_KEY] = _build_session_trace(
        db_path, str(envelope["session_id"])
    )
    return json.dumps(envelope, ensure_ascii=False)


def _capture_failed_hermes_session_trace(
    raw_response: str, db_path: Path, error: str
) -> Optional[str]:
    """Preserve partial solver sessions when Hermes exits before a response."""
    if not db_path.is_file():
        return None
    try:
        if raw_response.strip():
            return _attach_hermes_session_trace(raw_response, db_path)
    except Exception:
        pass

    connection = None
    try:
        connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
        row = connection.execute(
            "SELECT id FROM sessions WHERE parent_session_id IS NULL "
            "ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
    except sqlite3.Error:
        row = None
    finally:
        if connection is not None:
            connection.close()
    if not row or not row[0]:
        return None
    envelope = {
        "session_id": str(row[0]),
        "final_response": "",
        "messages": [],
        "completed": False,
        "failed": True,
        "error": error,
        "backend_response_excerpt": str(raw_response or "")[:2000],
    }
    try:
        return _attach_hermes_session_trace(
            json.dumps(envelope, ensure_ascii=False), db_path
        )
    except Exception:
        return None


def _extract_hermes_session_trace(
    raw_response: Optional[str],
) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    if not raw_response:
        return raw_response, None
    for candidate in _json_object_candidates(raw_response):
        try:
            envelope = json.loads(candidate)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(envelope, dict):
            continue
        trace = envelope.pop(_P7_SESSION_TRACE_KEY, None)
        if isinstance(trace, dict):
            return json.dumps(envelope, ensure_ascii=False), trace
    return raw_response, None


def _sft_tool_call(call: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    function = call.get("function")
    function = function if isinstance(function, dict) else {}
    name = str(function.get("name") or "").strip()
    if not name:
        return None
    arguments = function.get("arguments", "{}")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    return {
        "id": str(
            call.get("id")
            or call.get("call_id")
            or call.get("response_item_id")
            or ""
        ),
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _sft_message(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    role = str(message.get("role") or "")
    if role == "user":
        return {"role": "user", "content": message.get("content") or ""}
    if role == "tool":
        item = {
            "role": "tool",
            "tool_call_id": str(message.get("tool_call_id") or ""),
            "content": message.get("content") or "",
        }
        if message.get("tool_name"):
            item["name"] = str(message["tool_name"])
        return item
    if role != "assistant":
        return None
    item = {"role": "assistant", "content": message.get("content") or ""}
    calls = [
        converted
        for call in message.get("tool_calls") or []
        if isinstance(call, dict)
        and (converted := _sft_tool_call(call)) is not None
    ]
    if calls:
        item["tool_calls"] = calls
    reasoning = message.get("reasoning_content")
    if not isinstance(reasoning, str):
        reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        item["reasoning_content"] = reasoning
    return item


def _solver_delegate_specs(conversation: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    specs: Dict[str, Dict[str, Any]] = {}
    for parent in conversation.get("trace_sessions") or []:
        if not isinstance(parent, dict):
            continue
        calls = [
            call
            for message in parent.get("messages") or []
            if isinstance(message, dict)
            for call in message.get("tool_calls") or []
            if isinstance(call, dict)
            and str((call.get("function") or {}).get("name") or "") == "delegate_task"
        ]
        if not calls:
            continue
        arguments: list[Dict[str, Any]] = []
        for call in calls:
            raw = (call.get("function") or {}).get("arguments", "{}")
            try:
                parsed = json.loads(raw) if isinstance(raw, str) else raw
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(parsed, dict):
                continue
            tasks = parsed.get("tasks")
            if isinstance(tasks, list):
                arguments.extend(item for item in tasks if isinstance(item, dict))
            else:
                arguments.append(parsed)
        for spec in arguments:
            goal = str(spec.get("goal") or spec.get("task") or "").strip()
            if goal:
                specs["goal:" + goal] = spec
        for message in parent.get("messages") or []:
            if not isinstance(message, dict) or message.get("role") != "tool":
                continue
            if str(message.get("tool_name") or "") != "delegate_task":
                continue
            try:
                result = json.loads(str(message.get("content") or ""))
            except json.JSONDecodeError:
                continue
            for item in result.get("results") or []:
                if not isinstance(item, dict):
                    continue
                session_id = str(item.get("session_id") or "")
                index = item.get("task_index")
                if session_id and isinstance(index, int) and index < len(arguments):
                    specs[session_id] = arguments[index]
    return specs


def _solver_child_system_addition(spec: Dict[str, Any]) -> str:
    goal = str(spec.get("goal") or spec.get("task") or "")
    context = str(spec.get("context") or "").strip()
    parts = [
        "You are a focused subagent working on a specific delegated task.",
        "",
        f"YOUR TASK:\n{goal}",
    ]
    if context:
        parts.append(f"\nCONTEXT:\n{context}")
    parts.append(
        "\nComplete this task using the tools available to you. When finished, "
        "provide a clear, concise summary of what you did, what you found, any "
        "files created, and any issues encountered."
    )
    return "\n".join(parts)


def _solver_sft_session(
    session: Dict[str, Any],
    conversation: Dict[str, Any],
    *,
    is_main: bool,
) -> Dict[str, Any]:
    system_prompt = str(session.get("system_prompt") or "")
    system_source = "hermes_persisted_base"
    if not is_main:
        specs = _solver_delegate_specs(conversation)
        session_id = str(session.get("session_id") or session.get("id") or "")
        spec = specs.get(session_id)
        if spec is None:
            first_user = next(
                (
                    str(message.get("content") or "")
                    for message in session.get("messages") or []
                    if isinstance(message, dict) and message.get("role") == "user"
                ),
                "",
            )
            spec = specs.get("goal:" + first_user)
        if spec is not None:
            system_prompt = (
                system_prompt + "\n\n" + _solver_child_system_addition(spec)
            ).strip()
            system_source += " + reconstructed_delegate_ephemeral_system"
        else:
            system_source += " (delegate_ephemeral_system_unavailable)"
    messages: list[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for raw in session.get("messages") or []:
        if isinstance(raw, dict):
            converted = _sft_message(raw)
            if converted is not None:
                messages.append(converted)
    tool_names = sorted(
        {
            str((call.get("function") or {}).get("name") or "")
            for raw in session.get("messages") or []
            if isinstance(raw, dict)
            for call in raw.get("tool_calls") or []
            if isinstance(call, dict) and (call.get("function") or {}).get("name")
        }
    )
    return {
        "session_id": session.get("session_id") or session.get("id"),
        "parent_session_id": session.get("parent_session_id"),
        "source": session.get("source"),
        "model": session.get("model"),
        "messages": messages,
        "tool_names_used": tool_names,
        "api_call_count": session.get("api_call_count"),
        "tool_call_count": session.get("tool_call_count"),
        "system_prompt_source": system_source,
    }


def _solver_sft_payload(
    conversation: Dict[str, Any], trace_context: Dict[str, Any]
) -> Dict[str, Any]:
    sessions = [
        item
        for item in conversation.get("trace_sessions") or []
        if isinstance(item, dict)
    ]
    primary_id = str(conversation.get("session_id") or "")
    main = next(
        (
            session
            for session in sessions
            if str(session.get("session_id") or session.get("id") or "") == primary_id
        ),
        next((session for session in sessions if not session.get("parent_session_id")), None),
    )
    safe_context = {
        key: trace_context.get(key)
        for key in (
            "seed_index",
            "question_version",
            "rollout_id",
            "agent_attempt",
        )
        if key in trace_context
    }
    if main is None:
        return {
            "format": "openai_chat_completions_sessions_v1",
            "messages": conversation.get("messages") or [],
            "metadata": {
                **safe_context,
                "trace_complete": False,
                "training_warning": "Hermes session trace was unavailable",
            },
        }
    root = _solver_sft_session(main, conversation, is_main=True)
    children = [
        _solver_sft_session(session, conversation, is_main=False)
        for session in sessions
        if session is not main
    ]
    trace_complete = bool(
        (conversation.get("trace_capture") or {}).get("complete")
    )
    metadata = {
        **safe_context,
        "trajectory_type": "solver",
        "model": conversation.get("agent", {}).get("model"),
        "provider": conversation.get("agent", {}).get("provider"),
        "enabled_toolsets": conversation.get("agent", {}).get(
            "enabled_toolsets"
        ),
        "root_session": {
            key: value for key, value in root.items() if key != "messages"
        },
        "session_trajectories": children,
        "trace_complete": trace_complete,
        "training_notes": [
            "messages is the root Hermes session in OpenAI chat shape.",
            "Sub-agent sessions are separate conversations in session_trajectories.",
            "reasoning_content, tool calls, and tool results are preserved.",
            "Reference answers and judge labels are intentionally excluded.",
        ],
    }
    if not trace_complete:
        metadata["training_warning"] = (
            "Incomplete Hermes trace; retain for audit only and exclude from SFT."
        )
    return {
        "format": "openai_chat_completions_sessions_v1",
        "messages": root["messages"],
        "metadata": metadata,
    }


class ProviderRateLimitError(RuntimeError):
    pass


class HermesBackendError(RuntimeError):
    def __init__(self, message: str, raw_response: str):
        super().__init__(message)
        self.raw_response = raw_response


class ProviderUnavailableError(HermesBackendError):
    """Provider already exhausted its own transient retry budget."""


class HermesTimeoutError(HermesBackendError):
    """One Hermes attempt consumed its complete configured wall-time budget."""


class AgentRunner:
    def __init__(self, hermes_path: Optional[Path], hermes_command: Optional[str]):
        self.hermes_path = hermes_path
        self.hermes_command = hermes_command
        self._aiagent_cls = None
        self.memory_guard = MemoryGuard.from_env()
        self.rate_limit_guard = RateLimitGuard.from_env()

    def run_json(
        self,
        config: AgentConfig,
        *,
        system_prompt: str,
        user_payload: Dict[str, Any],
        dry_run_response: Optional[Dict[str, Any]] = None,
        rate_limit_scope: Optional[str] = None,
        response_validator: Optional[Callable[[Dict[str, Any]], None]] = None,
        trace_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if dry_run_response is not None:
            return dry_run_response
        if not config.configured:
            raise RuntimeError(f"agent {config.name} is missing api_key/base_url/model")
        backend = (config.backend or "hermes").lower()
        retries = _agent_retries(config.name)
        rate_limit_label = _agent_rate_limit_label(config.name, rate_limit_scope)
        rate_limit_concurrency = _agent_rate_limit_concurrency(config.name)
        for attempt in range(retries + 1):
            started_at = time.time()
            response: Optional[str] = None
            parsed: Optional[Dict[str, Any]] = None
            error: Optional[str] = None
            try:
                with self.rate_limit_guard.reserve(rate_limit_label, rate_limit_concurrency):
                    response = self._run_backend(config, backend, system_prompt, user_payload)
                if config.name == "solver":
                    model_response = _extract_solver_model_response(response)
                    parsed = {
                        "model_response": model_response,
                        "final_answer": model_response,
                        "confidence": "",
                        "evidence": [],
                        "reasoning_summary": "",
                    }
                else:
                    parsed = extract_json(response)
                rate_limit_message = _provider_rate_limit_message(parsed)
                if rate_limit_message:
                    raise ProviderRateLimitError(rate_limit_message)
                if response_validator is not None:
                    response_validator(parsed)
                if config.name == "solver":
                    parsed["_execution"] = _solver_execution_metadata(backend, response)
                self.rate_limit_guard.succeeded(
                    rate_limit_label, rate_limit_concurrency
                )
                return parsed
            except Exception as exc:
                if response is None and isinstance(exc, HermesBackendError):
                    response = exc.raw_response
                error = repr(exc)
                if isinstance(exc, ProviderUnavailableError) or (
                    isinstance(exc, HermesTimeoutError) and config.name == "solver"
                ):
                    raise
                rate_limited = _is_rate_limit_error(exc)
                if rate_limited:
                    self.rate_limit_guard.rate_limited(
                        rate_limit_label, rate_limit_concurrency
                    )
                self._write_debug(config, system_prompt, user_payload, response or "", parsed, error)
                if attempt >= retries:
                    if rate_limited and not isinstance(exc, ProviderRateLimitError):
                        raise ProviderRateLimitError(str(exc)) from exc
                    raise
                time.sleep(_agent_retry_sleep(attempt))
            finally:
                conversation_files = self._write_conversation(
                    config,
                    system_prompt,
                    user_payload,
                    raw_response=response,
                    parsed_response=parsed,
                    error=error,
                    duration_seconds=time.time() - started_at,
                    trace_context={
                        **(trace_context or {}),
                        "agent_attempt": attempt,
                    },
                )
                if (
                    conversation_files
                    and isinstance(parsed, dict)
                    and config.name in {"solver", "solver_verifier"}
                ):
                    parsed["_trace_files"] = conversation_files
                if error is None:
                    self._write_debug(config, system_prompt, user_payload, response or "", parsed, None)
        raise RuntimeError(f"agent {config.name} failed after retries")

    def run_text(
        self,
        config: AgentConfig,
        *,
        system_prompt: str,
        user_payload: Dict[str, Any],
        dry_run_response: Optional[str] = None,
        rate_limit_scope: Optional[str] = None,
    ) -> str:
        if dry_run_response is not None:
            return dry_run_response
        if not config.configured:
            raise RuntimeError(f"agent {config.name} is missing api_key/base_url/model")
        backend = (config.backend or "hermes").lower()
        retries = _agent_retries(config.name)
        rate_limit_label = _agent_rate_limit_label(config.name, rate_limit_scope)
        rate_limit_concurrency = _agent_rate_limit_concurrency(config.name)
        for attempt in range(retries + 1):
            started_at = time.time()
            response: Optional[str] = None
            error: Optional[str] = None
            try:
                with self.rate_limit_guard.reserve(rate_limit_label, rate_limit_concurrency):
                    response = self._run_backend(config, backend, system_prompt, user_payload)
                rate_limit_message = _provider_rate_limit_message_from_text(response)
                if rate_limit_message:
                    raise ProviderRateLimitError(rate_limit_message)
                self.rate_limit_guard.succeeded(
                    rate_limit_label, rate_limit_concurrency
                )
                return response
            except Exception as exc:
                if response is None and isinstance(exc, HermesBackendError):
                    response = exc.raw_response
                error = repr(exc)
                rate_limited = _is_rate_limit_error(exc)
                if rate_limited:
                    self.rate_limit_guard.rate_limited(
                        rate_limit_label, rate_limit_concurrency
                    )
                self._write_debug(config, system_prompt, user_payload, response or "", None, error)
                if attempt >= retries:
                    if rate_limited and not isinstance(exc, ProviderRateLimitError):
                        raise ProviderRateLimitError(str(exc)) from exc
                    raise
                time.sleep(_agent_retry_sleep(attempt))
            finally:
                self._write_conversation(
                    config,
                    system_prompt,
                    user_payload,
                    raw_response=response,
                    parsed_response=None,
                    error=error,
                    duration_seconds=time.time() - started_at,
                )
                if error is None:
                    self._write_debug(config, system_prompt, user_payload, response or "", None, None)
        raise RuntimeError(f"agent {config.name} failed after retries")

    def _run_backend(
        self,
        config: AgentConfig,
        backend: str,
        system_prompt: str,
        user_payload: Dict[str, Any],
    ) -> str:
        if backend == "direct":
            return self._run_direct(config, system_prompt, user_payload)
        with self.memory_guard.reserve(config.name):
            if not self.hermes_path:
                return self._run_hermes_command(config, system_prompt, user_payload)
            return self._run_aiagent(config, system_prompt, user_payload)

    def _run_direct(self, config: AgentConfig, system_prompt: str, user_payload: Dict[str, Any]) -> str:
        endpoint = config.base_url.rstrip("/") + "/chat/completions"
        user_content = (
            str(user_payload["__raw_prompt"])
            if "__raw_prompt" in user_payload
            else "Process this JSON payload and return JSON only.\n\n"
            + json.dumps(user_payload, ensure_ascii=False, indent=2)
        )
        user_message_content: Any = user_content
        if config.name == "solver_verifier":
            user_message_content = [{"type": "text", "text": user_content}]
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message_content})
        request_payload = {
            "model": config.model,
            "messages": messages,
        }
        if config.name == "solver_verifier":
            token_param = os.environ.get(
                "SOLVER_VERIFIER_TOKEN_PARAM", "max_completion_tokens"
            ).strip()
            if token_param not in {"max_completion_tokens", "max_tokens"}:
                raise ValueError(
                    "SOLVER_VERIFIER_TOKEN_PARAM must be max_completion_tokens or max_tokens"
                )
            request_payload.update(
                {
                    token_param: int(
                        os.environ.get(
                            "SOLVER_VERIFIER_JUDGE_MAX_TOKENS", "2048"
                        )
                    ),
                    "temperature": float(
                        os.environ.get("SOLVER_VERIFIER_TEMPERATURE", "1.0")
                    ),
                }
            )
            # Some OpenAI-compatible verifier endpoints reject sampling
            # parameters they do not support. An empty value explicitly
            # omits top_p; a non-empty value remains available when required.
            top_p_raw = os.environ.get("SOLVER_VERIFIER_TOP_P", "").strip()
            if top_p_raw:
                request_payload["top_p"] = float(top_p_raw)
            request_payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "judge_result",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "reason": {"type": "string"},
                            "score": {"type": "integer", "enum": [0, 1]},
                        },
                        "required": ["reason", "score"],
                        "additionalProperties": False,
                    },
                },
            }
        else:
            request_payload.update(
                {
                    "temperature": float(
                        os.environ.get("V2_DIRECT_TEMPERATURE", "0.2")
                    ),
                    "max_tokens": config.max_tokens,
                }
            )
        if config.name != "solver_verifier" and "__raw_prompt" not in user_payload:
            if config.name == "question_verifier":
                request_payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "question_verdict",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "reason": {"type": "string"},
                                "score": {"type": "integer", "enum": [0, 1]},
                            },
                            "required": ["reason", "score"],
                            "additionalProperties": False,
                        },
                    },
                }
            elif _json_object_mode_enabled():
                request_payload["response_format"] = {"type": "json_object"}
        body = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")
        if config.name == "solver_verifier":
            timeout_seconds = float(
                os.environ.get("SOLVER_VERIFIER_TIMEOUT_SECONDS", "300")
            )
            max_attempts = max(
                1,
                int(
                    os.environ.get(
                        "SOLVER_VERIFIER_DIRECT_MAX_ATTEMPTS", "3"
                    )
                ),
            )
            retries = max_attempts - 1
        else:
            timeout_seconds = int(os.environ.get("V2_AGENT_TIMEOUT_SECONDS", "900"))
            retries = int(os.environ.get("V2_DIRECT_RETRIES", "2"))
        retry_statuses = {408, 409, 425, 429, 500, 502, 503, 504}
        last_error = ""
        for attempt in range(retries + 1):
            request = urllib.request.Request(
                endpoint,
                data=body,
                method="POST",
                headers={
                    "Authorization": f"Bearer {config.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                    data = json.loads(response.read().decode("utf-8"))
                rate_limit_message = _provider_rate_limit_message(data)
                if rate_limit_message:
                    raise ProviderRateLimitError(rate_limit_message)
                content = data["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    content = "".join(
                        str(item.get("text", item.get("content", "")))
                        if isinstance(item, dict)
                        else str(item)
                        for item in content
                    )
                return str(content)
            except urllib.error.HTTPError as exc:
                error_text = exc.read().decode("utf-8", errors="replace")
                last_error = f"status={exc.code} body={error_text}"
                if exc.code == 429:
                    raise ProviderRateLimitError(last_error) from exc
                if exc.code not in retry_statuses or attempt >= retries:
                    raise RuntimeError(f"direct API failed for {config.name}: {last_error}") from exc
            except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
                last_error = str(exc)
                if attempt >= retries:
                    raise RuntimeError(f"direct API failed for {config.name}: {last_error}") from exc
            time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"direct API failed for {config.name}: {last_error}")

    def _run_aiagent(
        self,
        config: AgentConfig,
        system_prompt: str,
        user_payload: Dict[str, Any],
    ) -> str:
        workspace = _create_hermes_tool_workspace(config.name)
        # The legacy in-process backend reads TERMINAL_CWD from process-global
        # state. Serialize only this backend so concurrent threads cannot point
        # one another's tools at the wrong workspace. The production command
        # backend keeps full concurrency because every call has its own process.
        with _HERMES_INPROCESS_WORKSPACE_LOCK:
            previous_workspace = os.environ.get("V2_HERMES_TOOL_WORKSPACE")
            previous_terminal_cwd = os.environ.get("TERMINAL_CWD")
            os.environ["V2_HERMES_TOOL_WORKSPACE"] = str(workspace)
            os.environ["TERMINAL_CWD"] = str(workspace)
            try:
                return self._run_aiagent_in_workspace(
                    config,
                    system_prompt,
                    user_payload,
                )
            finally:
                _restore_env_value("V2_HERMES_TOOL_WORKSPACE", previous_workspace)
                _restore_env_value("TERMINAL_CWD", previous_terminal_cwd)
                _remove_empty_hermes_tool_workspace(workspace)

    def _run_aiagent_in_workspace(
        self,
        config: AgentConfig,
        system_prompt: str,
        user_payload: Dict[str, Any],
    ) -> str:
        AIAgent = self._load_aiagent()
        _install_hermes_env_override_guard()
        capture_sessions = config.name == "solver"
        trace_directory = (
            tempfile.TemporaryDirectory(prefix="p7-hermes-solver-")
            if capture_sessions
            else None
        )
        session_db = None
        agent_kwargs = dict(
            base_url=config.base_url,
            api_key=config.api_key,
            provider=config.provider,
            model=config.model,
            max_iterations=config.max_iterations,
            enabled_toolsets=config.enabled_toolsets,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        if system_prompt:
            agent_kwargs["ephemeral_system_prompt"] = system_prompt
        if config.api_mode:
            agent_kwargs["api_mode"] = config.api_mode
        try:
            trace_db_path = None
            if trace_directory is not None:
                from hermes_state import SessionDB  # type: ignore

                trace_db_path = Path(trace_directory.name) / "state.db"
                session_db = SessionDB(db_path=trace_db_path)
                agent_kwargs["session_db"] = session_db
            agent = AIAgent(**agent_kwargs)
            prompt = (
                str(user_payload["__raw_prompt"])
                if "__raw_prompt" in user_payload
                else "Process this JSON payload and return JSON only.\n\n"
                + json.dumps(user_payload, ensure_ascii=False, indent=2)
            )
            result = agent.run_conversation(prompt)
            response = (
                json.dumps(result, ensure_ascii=False)
                if isinstance(result, dict)
                else str(result)
            )
            if trace_db_path is not None:
                response = _attach_hermes_session_trace(response, trace_db_path)
            return response
        except Exception as exc:
            captured = (
                _capture_failed_hermes_session_trace("", trace_db_path, str(exc))
                if trace_db_path is not None
                else None
            )
            if captured is not None:
                raise HermesBackendError(str(exc), captured) from exc
            raise
        finally:
            if session_db is not None:
                try:
                    session_db.close()
                except Exception:
                    pass
            if trace_directory is not None:
                trace_directory.cleanup()

    def _run_hermes_command(
        self,
        config: AgentConfig,
        system_prompt: str,
        user_payload: Dict[str, Any],
    ) -> str:
        python_bin, hermes_root = _resolve_hermes_command(self.hermes_command)
        if not python_bin or not hermes_root:
            raise RuntimeError("could not resolve Hermes command; set V2_HERMES_COMMAND or V2_HERMES_PATH")
        workspace = _create_hermes_tool_workspace(config.name)
        request = {
            "agent_config": {
                "api_key": config.api_key,
                "base_url": config.base_url,
                "model": config.model,
                "provider": config.provider,
                "api_mode": config.api_mode,
                "max_iterations": config.max_iterations,
                "max_tokens": config.max_tokens,
                "enabled_toolsets": config.enabled_toolsets,
            },
            "system_prompt": system_prompt,
            "user_payload": user_payload,
        }
        trace_directory = (
            tempfile.TemporaryDirectory(prefix="p7-hermes-solver-")
            if config.name == "solver"
            else None
        )
        trace_db_path = None
        if trace_directory is not None:
            trace_db_path = Path(trace_directory.name) / "state.db"
            request["session_db_path"] = str(trace_db_path)
        # The installed Hermes AIAgent does not accept ``response_format`` in
        # its constructor. JSON-mode enforcement remains available on the
        # direct API path; Hermes responses are parsed and retried below.
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as f:
            json.dump(request, f, ensure_ascii=False)
            request_path = f.name
        timeout_seconds = _agent_timeout_seconds(config)
        try:
            try:
                subprocess_env = _hermes_subprocess_env(hermes_root)
                subprocess_env["V2_HERMES_TOOL_WORKSPACE"] = str(workspace)
                subprocess_env["TERMINAL_CWD"] = str(workspace)
                proc = subprocess.Popen(
                    [str(python_bin), "-c", _SUBPROCESS_RUNNER_CODE, request_path],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=subprocess_env,
                    cwd=str(workspace),
                    start_new_session=True,
                )
                stdout, stderr = proc.communicate(timeout=timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                _terminate_process_group(proc)
                stdout, stderr = proc.communicate()
                stderr = (stderr or "").strip()
                raise HermesTimeoutError(
                    f"Hermes command timed out for {config.name} "
                    f"model={config.model} timeout={timeout_seconds}s"
                    + (f": {stderr}" if stderr else ""),
                    "",
                ) from exc
            except BaseException:
                if "proc" in locals():
                    _terminate_process_group(proc)
                raise
            if proc.returncode != 0:
                raise RuntimeError(
                    f"Hermes command failed for {config.name} exit={proc.returncode}:\n{stderr.strip()}"
                )
            response = stdout.strip()
            if trace_db_path is not None:
                unavailable_message = _provider_unavailable_message_from_text(response)
                if unavailable_message:
                    raise ProviderUnavailableError(
                        unavailable_message,
                        response,
                    )
                rate_limit_message = _provider_rate_limit_message_from_text(response)
                if rate_limit_message:
                    raise ProviderRateLimitError(rate_limit_message)
                try:
                    response = _attach_hermes_session_trace(response, trace_db_path)
                except RuntimeError as exc:
                    excerpt = response[:1000] if response else "<empty stdout>"
                    raise RuntimeError(
                        f"{exc}; backend_response={excerpt}"
                    ) from exc
            return response
        except Exception as exc:
            captured = (
                _capture_failed_hermes_session_trace(
                    locals().get("stdout", "") or "", trace_db_path, str(exc)
                )
                if trace_db_path is not None
                else None
            )
            if captured is not None:
                if isinstance(exc, ProviderUnavailableError):
                    raise ProviderUnavailableError(str(exc), captured) from exc
                if isinstance(exc, HermesTimeoutError):
                    raise HermesTimeoutError(str(exc), captured) from exc
                raise HermesBackendError(str(exc), captured) from exc
            raise
        finally:
            Path(request_path).unlink(missing_ok=True)
            if trace_directory is not None:
                trace_directory.cleanup()
            _remove_empty_hermes_tool_workspace(workspace)

    def _load_aiagent(self):
        if self._aiagent_cls is not None:
            return self._aiagent_cls
        if not self.hermes_path:
            from run_agent import AIAgent  # type: ignore
        else:
            if not (self.hermes_path / "run_agent.py").exists():
                raise RuntimeError(f"V2_HERMES_PATH must contain run_agent.py: {self.hermes_path}")
            sys.path.insert(0, str(self.hermes_path))
            from run_agent import AIAgent  # type: ignore
        self._aiagent_cls = AIAgent
        return AIAgent

    def _write_debug(
        self,
        config: AgentConfig,
        system_prompt: str,
        user_payload: Dict[str, Any],
        raw_response: str,
        parsed_response: Optional[Dict[str, Any]],
        error: Optional[str],
    ) -> None:
        if os.environ.get("V2_HERMES_DEBUG", "").lower() not in {"1", "true", "yes", "on"}:
            return
        out_dir = Path(os.environ.get("V2_OUTPUT_DIR", "data/runs"))
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "hermes_debug.jsonl").open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "agent": config.name,
                        "model": config.model,
                        "backend": config.backend,
                        "toolsets": config.enabled_toolsets,
                        "system_prompt": system_prompt,
                        "user_payload": user_payload,
                        "raw_response": raw_response,
                        "parsed_response": parsed_response,
                        "error": error,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    def _write_conversation(
        self,
        config: AgentConfig,
        system_prompt: str,
        user_payload: Dict[str, Any],
        *,
        raw_response: Optional[str],
        parsed_response: Optional[Dict[str, Any]],
        error: Optional[str],
        duration_seconds: float,
        trace_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, str]]:
        if os.environ.get("V2_CONVERSATION_LOG", "1").lower() not in {"1", "true", "yes", "on"}:
            return None
        scoped = trace_context if isinstance(trace_context, dict) else {}
        seed_dir = str(scoped.get("seed_dir") or "").strip()
        question_version = int(scoped.get("question_version") or 0)
        rollout_id = int(scoped.get("rollout_id") or 0)
        agent_attempt = int(scoped.get("agent_attempt") or 0)
        if seed_dir and config.name in {"solver", "solver_verifier"}:
            out_dir = (
                Path(seed_dir)
                / "solver"
                / f"question_{question_version:03d}"
                / f"rollout_{rollout_id:03d}"
            )
        else:
            out_dir = Path(
                os.environ.get(
                    "V2_CONVERSATION_LOG_DIR",
                    str(Path(os.environ.get("V2_OUTPUT_DIR", "data/runs")) / "conversations"),
                )
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        user_content = (
            str(user_payload["__raw_prompt"])
            if "__raw_prompt" in user_payload
            else "Process this JSON payload and return JSON only.\n\n"
            + json.dumps(user_payload, ensure_ascii=False, indent=2)
        )
        logged_raw_response, session_trace = _extract_hermes_session_trace(raw_response)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})
        if logged_raw_response is not None:
            messages.append({"role": "assistant", "content": logged_raw_response})
        payload = {
            "schema_version": 2,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(duration_seconds, 3),
            "agent": {
                "name": config.name,
                "backend": config.backend,
                "provider": config.provider,
                "model": config.model,
                "api_mode": config.api_mode,
                "max_iterations": config.max_iterations,
                "enabled_toolsets": config.enabled_toolsets,
            },
            "messages": messages,
            "user_content": user_content,
            "user_payload": user_payload,
            "raw_response": logged_raw_response,
            "parsed_response": parsed_response,
            "error": error,
            "trace_context": scoped,
        }
        if session_trace is not None:
            sessions = session_trace.get("sessions")
            if not isinstance(sessions, list):
                sessions = []
            roles = session_trace.get("session_roles")
            if not isinstance(roles, dict):
                roles = {}
            subagent_sessions = [
                session
                for session in sessions
                if isinstance(session, dict)
                and str(roles.get(str(session.get("session_id") or ""), "")).startswith(
                    "subagent"
                )
            ]
            main_sessions = [
                session
                for session in sessions
                if isinstance(session, dict) and session not in subagent_sessions
            ]
            payload.update(
                {
                    "session_id": session_trace.get("primary_session_id"),
                    "history": main_sessions,
                    "trace_sessions": sessions,
                    "subagent_sessions": subagent_sessions,
                    "trace_reasoning_summaries": session_trace.get(
                        "reasoning_summaries", []
                    ),
                    "delegation": session_trace.get("delegation", {}),
                    "trace_capture": {
                        "complete": bool(session_trace.get("complete")),
                        "session_count": len(sessions),
                        "main_session_count": len(main_sessions),
                        "subagent_session_count": len(subagent_sessions),
                        "warnings": list(session_trace.get("warnings") or []),
                    },
                }
            )
        elif config.name == "solver" and (config.backend or "hermes").lower() == "hermes":
            payload["trace_capture"] = {
                "complete": False,
                "session_count": 0,
                "main_session_count": 0,
                "subagent_session_count": 0,
                "error": error or "Hermes solver session trace was not returned",
            }
        if seed_dir and config.name == "solver":
            filename = f"conversation_attempt_{agent_attempt:02d}.json"
        elif seed_dir and config.name == "solver_verifier":
            filename = f"judge_attempt_{agent_attempt:02d}.json"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{timestamp}_{_safe_filename(config.name)}_{uuid.uuid4().hex[:8]}.json"
        conversation_path = out_dir / filename
        conversation_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        files = {"conversation": str(conversation_path)}
        if seed_dir and config.name == "solver":
            sft_path = out_dir / f"sft_attempt_{agent_attempt:02d}.json"
            sft_path.write_text(
                json.dumps(
                    _solver_sft_payload(payload, scoped),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            files["sft"] = str(sft_path)
        return files


def extract_json(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    errors: list[str] = []
    for candidate in _json_object_candidates(stripped):
        try:
            return _unwrap_hermes_response(json.loads(candidate))
        except json.JSONDecodeError as exc:
            errors.append(str(exc))
            # Try JSON repair strategies before giving up on this candidate
            repaired = _repair_json(candidate)
            if repaired is not None:
                try:
                    return _unwrap_hermes_response(json.loads(repaired))
                except json.JSONDecodeError:
                    continue
            continue
    raise json.JSONDecodeError(
        f"Could not extract a valid JSON object (tried {len(errors)} candidates); "
        f"first error: {errors[0] if errors else 'unknown'}",
        stripped,
        0,
    )


def _extract_solver_model_response(text: str) -> str:
    """Extract Hermes' natural-language final response without JSON coercion."""
    stripped = str(text or "").strip()
    for candidate in _json_object_candidates(stripped):
        try:
            parsed = json.loads(candidate)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(parsed, dict) or "final_response" not in parsed:
            continue
        final_response = parsed.get("final_response")
        if parsed.get("failed"):
            raise RuntimeError(
                f"Hermes agent failed: {parsed.get('error') or final_response}"
            )
        if isinstance(final_response, str) and final_response.strip():
            return final_response.strip()
        if final_response not in (None, ""):
            return json.dumps(final_response, ensure_ascii=False)
        messages = parsed.get("messages")
        if isinstance(messages, list):
            for message in reversed(messages):
                if not isinstance(message, dict) or message.get("role") != "assistant":
                    continue
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
        raise ValueError("Hermes solver returned an empty final response")
    if not stripped:
        raise ValueError("solver returned an empty response")
    return stripped


def _json_object_candidates(text: str) -> list[str]:
    candidates = [text.strip()] if text.strip() else []
    for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE):
        block = match.group(1).strip()
        if block:
            candidates.append(block)
    candidates.extend(_balanced_json_objects(text))
    unique = []
    seen = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def _balanced_json_objects(text: str) -> list[str]:
    objects: list[str] = []
    start = -1
    depth = 0
    in_string = False
    escaped = False
    for idx, char in enumerate(text):
        if start < 0:
            if char == "{":
                start = idx
                depth = 1
                in_string = False
                escaped = False
            continue
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                objects.append(text[start : idx + 1])
                start = -1
    return objects


def _unwrap_hermes_response(parsed: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(parsed, dict) or "final_response" not in parsed:
        return parsed
    final_response = parsed.get("final_response")
    if parsed.get("failed"):
        raise RuntimeError(f"Hermes agent failed: {parsed.get('error') or final_response}")
    if isinstance(final_response, dict):
        return final_response
    if isinstance(final_response, str) and final_response.strip():
        try:
            return extract_json(final_response)
        except json.JSONDecodeError as exc:
            final_response_error = exc
    else:
        final_response_error = json.JSONDecodeError(
            "Hermes final_response is empty or is not valid JSON",
            str(final_response or ""),
            0,
        )
    fallback = _extract_json_from_messages(parsed)
    if fallback is not None:
        return fallback
    raise json.JSONDecodeError(
        "Hermes final_response did not contain a valid JSON object: "
        f"{final_response_error.msg}",
        final_response_error.doc,
        final_response_error.pos,
    )


def _agent_retries(agent_name: str = "") -> int:
    per_agent = f"{agent_name.upper()}_RETRIES" if agent_name else ""
    raw = os.environ.get(per_agent) if per_agent else None
    return max(0, int(raw if raw not in {None, ""} else os.environ.get("V2_AGENT_RETRIES", "2")))


def _agent_retry_sleep(attempt: int) -> float:
    base_sleep = max(0.0, float(os.environ.get("V2_AGENT_RETRY_SLEEP_SECONDS", "20")))
    max_sleep = max(base_sleep, float(os.environ.get("V2_AGENT_RETRY_MAX_SLEEP_SECONDS", "120")))
    return min(base_sleep * (attempt + 1), max_sleep)


def _agent_rate_limit_concurrency(agent_name: str) -> int:
    override = int(os.environ.get("V2_RATE_LIMIT_MAX_INFLIGHT", "0"))
    if override > 0:
        return override
    if agent_name == "local_constraint":
        return max(1, int(os.environ.get("V2_LOCAL_CONCURRENCY", "6")))
    if agent_name in {"solver", "solver_verifier"}:
        return max(1, int(os.environ.get("V2_SOLVER_CONCURRENCY", "6")))
    return max(1, int(os.environ.get("V2_SEED_CONCURRENCY", "4")))


def _agent_rate_limit_label(agent_name: str, rate_limit_scope: Optional[str]) -> str:
    scope = str(rate_limit_scope or "").strip()
    return f"{agent_name}:{scope}" if scope else agent_name


def _provider_rate_limit_message(parsed: Any) -> str:
    if not isinstance(parsed, dict):
        return ""
    error = parsed.get("error")
    nested = error if isinstance(error, dict) else {}
    fields = [
        parsed.get("type"),
        parsed.get("code"),
        parsed.get("message"),
        error if isinstance(error, str) else None,
        nested.get("type"),
        nested.get("code"),
        nested.get("message"),
        parsed.get("failure_reason"),
    ]
    message = " ".join(str(value) for value in fields if value is not None)
    normalized = message.lower().replace("-", "_")
    markers = (
        "insufficient_quota",
        "rate_limit",
        "rate limit",
        "too many requests",
        "exceeded your current quota",
        "http 429",
        "status=429",
    )
    return message if any(marker in normalized for marker in markers) else ""


def _provider_rate_limit_message_from_text(text: str) -> str:
    for candidate in _json_object_candidates(str(text or "")):
        try:
            parsed = json.loads(candidate)
        except (TypeError, json.JSONDecodeError):
            continue
        message = _provider_rate_limit_message(parsed)
        if message:
            return message
    return ""


def _provider_unavailable_message_from_text(text: str) -> str:
    value = str(text or "")
    normalized = value.casefold()
    markers = (
        "no available channel",
        "no channel available",
        "model is currently unavailable",
    )
    if any(marker in normalized for marker in markers):
        return value[-2000:]
    return ""


def _is_rate_limit_error(exc: Exception) -> bool:
    if isinstance(exc, ProviderRateLimitError):
        return True
    value = str(exc).lower()
    return any(
        marker in value
        for marker in (
            "insufficient_quota",
            "rate limit",
            "too many requests",
            "exceeded your current quota",
            "http 429",
            "status=429",
        )
    )


def _json_object_mode_enabled() -> bool:
    """True when 'response_format: json_object' should be requested from the LLM.

    Set ``V2_DIRECT_JSON_MODE=0`` to disable (e.g. when the provider doesn't
    support it).  Default is on.
    """
    raw = os.environ.get("V2_DIRECT_JSON_MODE", "1")
    return raw.lower() in {"1", "true", "yes", "y", "on"}


def _solver_execution_metadata(backend: str, response: Optional[str]) -> Dict[str, Any]:
    """Summarize main/sub-agent solver turns and tools for difficulty feedback."""
    tool_names: list[str] = []
    successful_web_tools = 0
    search_queries: list[str] = []
    reasoning_summaries: list[str] = []
    api_call_count = 0
    main_api_call_count = 0
    subagent_api_call_count = 0
    session_count = 0
    subagent_session_count = 0

    if backend == "hermes" and response:
        envelope: Dict[str, Any] = {}
        for candidate in _json_object_candidates(response):
            try:
                parsed_candidate = json.loads(candidate)
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(parsed_candidate, dict) and isinstance(
                parsed_candidate.get("messages"), list
            ):
                envelope = parsed_candidate
                break
        trace = envelope.get(_P7_SESSION_TRACE_KEY)
        trace = trace if isinstance(trace, dict) else {}
        raw_sessions = trace.get("sessions")
        roles = trace.get("session_roles")
        roles = roles if isinstance(roles, dict) else {}
        if isinstance(raw_sessions, list) and raw_sessions:
            sessions = [item for item in raw_sessions if isinstance(item, dict)]
        else:
            sessions = [{"session_id": "main", "messages": envelope.get("messages", [])}]
            roles = {"main": "main"}

        session_count = len(sessions)
        for session in sessions:
            session_id = str(session.get("session_id") or "main")
            role = str(roles.get(session_id) or "main")
            is_subagent = role.startswith("subagent")
            if is_subagent:
                subagent_session_count += 1
            pending_tool_names: list[str] = []
            for message in session.get("messages", []):
                if not isinstance(message, dict):
                    continue
                if message.get("role") == "assistant":
                    api_call_count += 1
                    if is_subagent:
                        subagent_api_call_count += 1
                    else:
                        main_api_call_count += 1
                    summary = str(
                        message.get("reasoning_summary")
                        or message.get("reasoning")
                        or ""
                    ).strip()
                    if summary and summary not in reasoning_summaries:
                        reasoning_summaries.append(summary[:1000])
                for tool_call in message.get("tool_calls") or []:
                    function = tool_call.get("function") or {}
                    name = str(function.get("name") or "")
                    if not name:
                        continue
                    tool_names.append(name)
                    pending_tool_names.append(name)
                    arguments = function.get("arguments")
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except (TypeError, json.JSONDecodeError):
                            arguments = {"query": arguments}
                    if isinstance(arguments, dict):
                        for key in ("query", "q", "url", "goal", "task"):
                            value = str(arguments.get(key) or "").strip()
                            if value and value not in search_queries:
                                search_queries.append(value[:500])
                                break
                if message.get("role") != "tool" or not pending_tool_names:
                    continue
                pending_name = pending_tool_names.pop(0)
                name = str(message.get("tool_name") or pending_name)
                if name not in {"web_search", "web_extract"} and not name.startswith("browser_"):
                    continue
                content = message.get("content")
                result = content
                if isinstance(content, str):
                    for candidate in _json_object_candidates(content):
                        try:
                            parsed_candidate = json.loads(candidate)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(parsed_candidate, dict):
                            result = parsed_candidate
                            break
                if isinstance(result, dict) and result.get("success") is False:
                    continue
                if isinstance(result, dict) and result.get("error"):
                    continue
                if isinstance(result, dict) and isinstance(result.get("results"), list):
                    usable = any(
                        isinstance(item, dict)
                        and not item.get("error")
                        and (item.get("content") or item.get("title") or item.get("url"))
                        for item in result["results"]
                    )
                    if not usable:
                        continue
                successful_web_tools += 1

    web_tool_names = [
        name
        for name in tool_names
        if name in {"web_search", "web_extract"} or name.startswith("browser_")
    ]
    return {
        "backend": backend,
        "tool_call_count": len(tool_names),
        "tool_names": sorted(set(tool_names)),
        "web_tool_call_count": len(web_tool_names),
        "successful_web_tool_count": successful_web_tools,
        "api_call_count": api_call_count,
        "main_api_call_count": main_api_call_count,
        "subagent_api_call_count": subagent_api_call_count,
        "session_count": session_count,
        "subagent_session_count": subagent_session_count,
        "search_queries": search_queries[:24],
        "reasoning_summaries": reasoning_summaries[:16],
    }


def _repair_json(text: str) -> Optional[str]:
    """Try to fix common JSON formatting errors emitted by LLMs.

    Returns the repaired text, or None if no repair could be applied.
    """
    if not text:
        return None
    repaired = text

    # 1. Preserve all valid fields when only terminal delimiters are missing.
    #    This must run before trailing-text cleanup because strings may contain
    #    brace characters and valid fields may follow the last nested object.
    completed = _complete_json_containers(repaired)
    if completed is not None:
        repaired = completed
    else:
        without_trailing_text = _strip_after_complete_json_container(repaired)
        if without_trailing_text is not None:
            repaired = without_trailing_text

    # 2. Remove trailing commas before closing braces/brackets
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)

    # 3. A trailing comma may have prevented a valid completion on the first pass.
    completed = _complete_json_containers(repaired)
    if completed is not None:
        repaired = re.sub(r",\s*([}\]])", r"\1", completed)

    # 4. Fix single-quoted strings (common when LLM hallucinates Python dicts)
    #    Only do this if the text looks more like Python than JSON
    if repaired.count("'") > repaired.count('"') * 0.8 and repaired.startswith("{"):
        # Heuristic: not worth trying for deeply nested structures
        if repaired.count("{") <= 20:
            repaired = _try_fix_python_dict(repaired)

    return repaired if repaired != text else None


def _complete_json_containers(text: str) -> Optional[str]:
    """Append only missing terminal JSON container delimiters when it is safe."""
    stripped = text.strip()
    if not stripped or stripped[0] not in "{[":
        return None

    expected_closers: list[str] = []
    in_string = False
    escaped = False
    pairs = {"{": "}", "[": "]"}
    for char in stripped:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in pairs:
            expected_closers.append(pairs[char])
        elif char in "}]":
            if not expected_closers or expected_closers[-1] != char:
                return None
            expected_closers.pop()

    if in_string or escaped or not expected_closers:
        return None
    return stripped + "".join(reversed(expected_closers))


def _strip_after_complete_json_container(text: str) -> Optional[str]:
    """Remove text after the first fully closed root JSON object or array."""
    stripped = text.strip()
    if not stripped or stripped[0] not in "{[":
        return None

    expected_closers: list[str] = []
    in_string = False
    escaped = False
    pairs = {"{": "}", "[": "]"}
    for index, char in enumerate(stripped):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in pairs:
            expected_closers.append(pairs[char])
        elif char in "}]":
            if not expected_closers or expected_closers[-1] != char:
                return None
            expected_closers.pop()
            if not expected_closers:
                candidate = stripped[: index + 1]
                return candidate if candidate != stripped else None
    return None


def _try_fix_python_dict(text: str) -> str:
    """Convert a Python dict literal to valid JSON."""
    import ast
    try:
        obj = ast.literal_eval(text)
        return json.dumps(obj, ensure_ascii=False)
    except (ValueError, SyntaxError, MemoryError):
        return text


def _extract_json_from_messages(parsed: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for key in ("codex_message_items", "messages"):
        for item in reversed(parsed.get(key, []) or []):
            if item.get("role") != "assistant":
                continue
            content = item.get("content")
            parts = content if isinstance(content, list) else [{"text": content}]
            for part in parts:
                text = part.get("text") if isinstance(part, dict) else str(part)
                if not text:
                    continue
                try:
                    return extract_json(str(text))
                except json.JSONDecodeError:
                    continue
    return None


def _safe_filename(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._")
    return slug or "agent"


def _resolve_hermes_command(command: Optional[str]) -> tuple[Optional[Path], Optional[Path]]:
    real_hermes_bin = os.environ.get("REAL_HERMES_BIN", "").strip()
    command = real_hermes_bin or command
    if not command:
        return None, None
    command_path = Path(command).expanduser()
    if not command_path.is_absolute():
        resolved = shutil.which(str(command_path))
        if not resolved:
            return None, None
        command_path = Path(resolved)
    command_path = command_path.resolve()
    text = ""
    try:
        text = command_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        pass
    match = re.search(r'exec\s+"([^"]*/bin/hermes)"', text)
    hermes_bin = Path(match.group(1)).resolve() if match else command_path
    hermes_root = _hermes_root_from_bin(hermes_bin)
    python_bin = os.environ.get("PYTHON_BIN", "").strip()
    if python_bin:
        # Match the reference BrowseComp launcher: an explicit interpreter is
        # authoritative and takes precedence over HERMES_VENV.
        candidate = _resolve_executable(python_bin)
        return candidate, hermes_root

    hermes_venv = os.environ.get("HERMES_VENV", "").strip()
    if hermes_venv:
        venv_bin = Path(hermes_venv).expanduser() / "bin"
        for candidate in (venv_bin / "python3", venv_bin / "python"):
            resolved = _resolve_executable(str(candidate))
            if resolved:
                return resolved, hermes_root
        return None, hermes_root

    for candidate in (
        hermes_bin.with_name("python"),
        hermes_bin.parent.parent / "bin" / "python",
    ):
        resolved = _resolve_executable(str(candidate))
        if resolved:
            return resolved, hermes_root
    return None, hermes_root


def _resolve_executable(value: str) -> Optional[Path]:
    expanded = Path(value).expanduser()
    if expanded.is_absolute() or os.sep in value:
        # Keep a virtualenv interpreter's symlink path intact. CPython uses
        # that invocation path to find the adjacent pyvenv.cfg; resolving it
        # to the base interpreter would silently leave the virtualenv.
        candidate = Path(os.path.abspath(str(expanded)))
    else:
        found = shutil.which(value)
        if not found:
            return None
        candidate = Path(os.path.abspath(found))
    if candidate.exists() and os.access(candidate, os.X_OK):
        return candidate
    return None


def _hermes_subprocess_env(hermes_root: Path) -> Dict[str, str]:
    env = os.environ.copy()
    if env.get("V2_HERMES_SHARED_HOME", "").strip():
        env["HERMES_HOME"] = str(_prepare_hermes_runtime_home(env))
    inherited_pythonpath = env.get("PYTHONPATH", "").strip()
    pythonpath = [str(hermes_root)]
    if inherited_pythonpath:
        pythonpath.append(inherited_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)

    path_entries = []
    hermes_home = env.get("HERMES_HOME", "").strip()
    if hermes_home:
        path_entries.append(str(Path(hermes_home).expanduser() / "node" / "bin"))
    hermes_venv = env.get("HERMES_VENV", "").strip()
    if hermes_venv:
        hermes_venv_path = Path(hermes_venv).expanduser()
        installed_node_bin = hermes_venv_path.parent / "node_modules" / ".bin"
        if installed_node_bin.is_dir():
            path_entries.append(str(installed_node_bin))
        path_entries.append(str(hermes_venv_path / "bin"))
    inherited_path = env.get("PATH", "").strip()
    if inherited_path:
        path_entries.append(inherited_path)
    if path_entries:
        env["PATH"] = os.pathsep.join(path_entries)
    return env


def _create_hermes_tool_workspace(agent_name: str) -> Path:
    if str(agent_name or "").strip().lower() == "solver":
        root_raw = os.environ.get(
            "V2_HERMES_SOLVER_WORKSPACE_ROOT",
            str(Path(tempfile.gettempdir()) / "multi_agent_mpcs_solver_workspaces"),
        ).strip()
    else:
        root_raw = os.environ.get(
            "V2_HERMES_TOOL_WORKSPACE_ROOT",
            "data/tool_workspaces",
        ).strip()
    root = Path(root_raw or "data/tool_workspaces").expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    agent_dir = root / _safe_filename(agent_name or "agent")
    workspace = agent_dir / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        + "_"
        + uuid.uuid4().hex[:8]
    )
    workspace.mkdir(parents=True, exist_ok=False)
    return workspace.absolute()


def _remove_empty_hermes_tool_workspace(workspace: Path) -> None:
    """Remove empty per-call directories while preserving downloaded files."""
    try:
        workspace.rmdir()
    except OSError:
        return
    for parent in (workspace.parent, workspace.parent.parent):
        try:
            parent.rmdir()
        except OSError:
            break


def _restore_env_value(name: str, previous: Optional[str]) -> None:
    if previous is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous


def _apply_hermes_tool_workspace_env_override(env: Dict[str, str]) -> None:
    """Keep Hermes tool cwd isolated after its dotenv reloads."""
    workspace = env.get("V2_HERMES_TOOL_WORKSPACE", "").strip()
    if workspace:
        env["TERMINAL_CWD"] = workspace


def _apply_hermes_browser_env_overrides(env: Dict[str, str]) -> None:
    """Restore p7 browser choices after Hermes loads HERMES_HOME/.env."""
    for target, source in _HERMES_BROWSER_ENV_OVERRIDES:
        env[target] = env.get(source, "").strip()


def _install_hermes_env_override_guard() -> None:
    """Keep lazy Hermes dotenv reloads from replacing p7 browser policy."""
    from hermes_cli import env_loader

    current_loader = env_loader.load_hermes_dotenv
    if getattr(current_loader, "_p7_browser_guard", False):
        _apply_hermes_browser_env_overrides(os.environ)
        _apply_hermes_tool_workspace_env_override(os.environ)
        return

    def guarded_loader(*args: Any, **kwargs: Any) -> Any:
        loaded = current_loader(*args, **kwargs)
        _apply_hermes_browser_env_overrides(os.environ)
        _apply_hermes_tool_workspace_env_override(os.environ)
        return loaded

    guarded_loader._p7_browser_guard = True  # type: ignore[attr-defined]
    env_loader.load_hermes_dotenv = guarded_loader
    _apply_hermes_browser_env_overrides(os.environ)
    _apply_hermes_tool_workspace_env_override(os.environ)


def _prepare_hermes_runtime_home(env: Dict[str, str]) -> Path:
    """Build a CCI-local Hermes home backed by shared read-only assets."""
    shared_home = Path(env["V2_HERMES_SHARED_HOME"].strip()).expanduser().absolute()
    if not shared_home.is_dir():
        raise RuntimeError(f"V2_HERMES_SHARED_HOME does not exist: {shared_home}")

    native_home = Path(env.get("HOME", str(Path.home()))).expanduser() / ".hermes"
    config_raw = env.get("V2_HERMES_LOCAL_CONFIG", "").strip()
    local_config = (
        Path(config_raw).expanduser().absolute()
        if config_raw
        else native_home / "config.yaml"
    )
    if not local_config.is_file():
        raise RuntimeError(
            "CCI-local Hermes config does not exist: "
            f"{local_config}; set V2_HERMES_LOCAL_CONFIG explicitly if needed"
        )

    runtime_raw = env.get("V2_HERMES_RUNTIME_HOME", "").strip()
    runtime_home = (
        Path(runtime_raw).expanduser().absolute()
        if runtime_raw
        else native_home / "browsecomp_v2_runtime"
    )
    if runtime_home == shared_home or _is_relative_to(runtime_home, shared_home):
        raise RuntimeError(
            "V2_HERMES_RUNTIME_HOME must be CCI-local and outside V2_HERMES_SHARED_HOME"
        )

    with _HERMES_RUNTIME_HOME_LOCK:
        runtime_home.mkdir(parents=True, exist_ok=True)
        _ensure_runtime_symlink(runtime_home / "config.yaml", local_config)

        local_config_home = local_config.parent
        for name in _HERMES_LOCAL_CREDENTIAL_FILES:
            source = local_config_home / name
            if source.exists():
                _ensure_runtime_symlink(runtime_home / name, source)

        for name in _HERMES_SHARED_ENTRIES:
            source = shared_home / name
            if source.exists():
                _ensure_runtime_symlink(runtime_home / name, source)

        node_home = runtime_home / "node"
        if node_home.is_symlink():
            raise RuntimeError(f"Hermes runtime node directory must be local: {node_home}")
        node_home.mkdir(exist_ok=True)
        shared_node_home = shared_home / "node"
        for name in _HERMES_SHARED_NODE_ENTRIES:
            source = shared_node_home / name
            if source.exists():
                _ensure_runtime_symlink(node_home / name, source)
        (node_home / "agent-browser-state").mkdir(exist_ok=True)

        for name in _HERMES_LOCAL_STATE_DIRS:
            state_dir = runtime_home / name
            if state_dir.is_symlink():
                raise RuntimeError(f"Hermes mutable state directory must be CCI-local: {state_dir}")
            state_dir.mkdir(exist_ok=True)

    return runtime_home


def _ensure_runtime_symlink(link: Path, target: Path) -> None:
    target = target.absolute()
    if link.is_symlink():
        if link.resolve(strict=False) == target.resolve(strict=False):
            return
        raise RuntimeError(f"Hermes runtime link points to an unexpected target: {link}")
    if link.exists():
        raise RuntimeError(f"Refusing to overwrite existing Hermes runtime path: {link}")
    link.symlink_to(target, target_is_directory=target.is_dir())


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
        return True
    except ValueError:
        return False


def _hermes_root_from_bin(hermes_bin: Path) -> Optional[Path]:
    for parent in [hermes_bin.parent, *hermes_bin.parents]:
        if (parent / "run_agent.py").exists():
            return parent
    return None


def _agent_timeout_seconds(config: AgentConfig) -> int:
    raw = os.environ.get(f"{config.name.upper()}_TIMEOUT_SECONDS") or os.environ.get(
        "V2_AGENT_TIMEOUT_SECONDS", "900"
    )
    try:
        return max(1, int(raw))
    except ValueError:
        return 900


def _terminate_process_group(proc: subprocess.Popen[str]) -> None:
    """Stop a timed-out Hermes process and any browser/tool descendants."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


_SUBPROCESS_RUNNER_CODE = r'''
import os

# Prefer losing one recoverable Hermes call over the long-lived workflow when
# the kernel must select an OOM victim despite admission control.
try:
    with open("/proc/self/oom_score_adj", "w", encoding="ascii") as oom_file:
        oom_file.write(os.environ.get("V2_HERMES_OOM_SCORE_ADJ", "500"))
except (OSError, ValueError):
    pass

import json
import sys

from run_agent import AIAgent

# run_agent loads HERMES_HOME/.env with override=True during import. Restore
# p7's explicit browser policy afterwards. Hermes lazily imports some modules
# during the first turn, and those modules load the same dotenv again, so guard
# every future load too. Empty values select local mode.
from hermes_cli import env_loader as _p7_env_loader

_p7_original_load_hermes_dotenv = _p7_env_loader.load_hermes_dotenv

def _p7_apply_browser_policy():
    os.environ["BROWSER_CDP_URL"] = os.environ.get(
        "V2_HERMES_BROWSER_CDP_URL", ""
    ).strip()
    os.environ["AGENT_BROWSER_AUTO_CONNECT"] = os.environ.get(
        "V2_HERMES_AGENT_BROWSER_AUTO_CONNECT", ""
    ).strip()

def _p7_apply_workspace_policy():
    workspace = os.environ.get("V2_HERMES_TOOL_WORKSPACE", "").strip()
    if workspace:
        os.environ["TERMINAL_CWD"] = workspace

def _p7_guarded_load_hermes_dotenv(*args, **kwargs):
    loaded = _p7_original_load_hermes_dotenv(*args, **kwargs)
    _p7_apply_browser_policy()
    _p7_apply_workspace_policy()
    return loaded

_p7_env_loader.load_hermes_dotenv = _p7_guarded_load_hermes_dotenv
_p7_apply_browser_policy()
_p7_apply_workspace_policy()

request = json.load(open(sys.argv[1], encoding="utf-8"))
cfg = request["agent_config"]

# Some OpenAI-compatible DeepSeek routes return encrypted reasoning items without
# the list-valued ``summary`` that the same route requires when those items are
# replayed. Normalize only that model's replay state at the runtime boundary; keep
# reasoning text and encrypted content unchanged.
if cfg.get("model") == "deepseek-v4-pro-0813":
    def _p7_fix_reasoning_summaries(items):
        if not isinstance(items, list):
            return items
        for item in items:
            if (
                isinstance(item, dict)
                and item.get("type") == "reasoning"
                and not isinstance(item.get("summary"), list)
            ):
                item["summary"] = []
        return items

    try:
        from agent import chat_completion_helpers as _p7_chat_helpers

        _p7_original_build_assistant_message = (
            _p7_chat_helpers.build_assistant_message
        )

        def _p7_build_assistant_message(*args, **kwargs):
            message = _p7_original_build_assistant_message(*args, **kwargs)
            if isinstance(message, dict):
                _p7_fix_reasoning_summaries(message.get("reasoning_details"))
            return message

        _p7_chat_helpers.build_assistant_message = _p7_build_assistant_message
    except Exception:
        pass

    try:
        from agent import codex_responses_adapter as _p7_codex_adapter

        _p7_original_normalize_codex_response = (
            _p7_codex_adapter._normalize_codex_response
        )

        def _p7_normalize_codex_response(*args, **kwargs):
            message, finish_reason = _p7_original_normalize_codex_response(
                *args, **kwargs
            )
            _p7_fix_reasoning_summaries(
                getattr(message, "codex_reasoning_items", None)
            )
            return message, finish_reason

        _p7_codex_adapter._normalize_codex_response = (
            _p7_normalize_codex_response
        )
    except Exception:
        pass

agent_kwargs = {
    "base_url": cfg["base_url"],
    "api_key": cfg["api_key"],
    "provider": cfg["provider"],
    "model": cfg["model"],
    "max_iterations": cfg["max_iterations"],
    "enabled_toolsets": cfg.get("enabled_toolsets"),
    "quiet_mode": True,
    "skip_context_files": True,
    "skip_memory": True,
}
if request.get("system_prompt"):
    agent_kwargs["ephemeral_system_prompt"] = request["system_prompt"]
if cfg.get("api_mode"):
    agent_kwargs["api_mode"] = cfg["api_mode"]
if cfg.get("max_tokens"):
    agent_kwargs["max_tokens"] = cfg["max_tokens"]
if request.get("session_db_path"):
    from pathlib import Path
    from hermes_state import SessionDB

    agent_kwargs["session_db"] = SessionDB(
        db_path=Path(request["session_db_path"])
    )
agent = AIAgent(**agent_kwargs)
_p7_apply_browser_policy()
_p7_apply_workspace_policy()
prompt = "Process this JSON payload and return JSON only.\n\n" + json.dumps(
    request["user_payload"], ensure_ascii=False, indent=2
)
if "__raw_prompt" in request["user_payload"]:
    prompt = str(request["user_payload"]["__raw_prompt"])
result = agent.run_conversation(prompt)
if isinstance(result, dict):
    print(json.dumps(result, ensure_ascii=False))
else:
    print(str(result))
# Hermes can leave non-daemon HTTP/event-loop threads alive after returning the
# final response.  The session DB is written synchronously by this point; flush
# the response and terminate the short-lived worker so the parent communicate()
# call cannot wait indefinitely on those background threads.
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
'''
