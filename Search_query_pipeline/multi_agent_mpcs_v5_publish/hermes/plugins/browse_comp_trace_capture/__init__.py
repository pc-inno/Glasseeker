"""Read-only capture of the first effective model request per BrowseComp session."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_ACTIVE_ENV = "BROWSE_COMP_RUN_ID"
_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_write_lock = threading.Lock()


def _active() -> bool:
    return bool(os.environ.get(_ACTIVE_ENV, "").strip())


def _capture_root() -> Path:
    configured = os.environ.get("BROWSE_COMP_TRACE_CAPTURE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    home = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))).expanduser()
    return home / "browsecomp_trace_capture"


def _safe_id(value: Any) -> str:
    cleaned = _SAFE_ID_RE.sub("_", str(value or "").strip()).strip("._-")
    return cleaned or "unknown"


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def _system_prompt(body: dict[str, Any]) -> str:
    instructions = body.get("instructions")
    if isinstance(instructions, str):
        return instructions
    messages = body.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "system":
                return _as_text(message.get("content"))
    return ""


def _tool_name(tool: Any) -> str:
    if not isinstance(tool, dict):
        return ""
    function = tool.get("function")
    if isinstance(function, dict):
        return str(function.get("name") or "").strip()
    return str(tool.get("name") or "").strip()


def _atomic_write_once(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _write_lock:
        if path.exists():
            return
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            temporary.unlink(missing_ok=True)
            return
        os.replace(temporary, path)


def _on_pre_api_request(**kwargs: Any) -> None:
    if not _active():
        return
    session_id = str(kwargs.get("session_id") or "").strip()
    if not session_id:
        return
    request = kwargs.get("request")
    body = request.get("body") if isinstance(request, dict) else {}
    if not isinstance(body, dict):
        body = {}
    request_payload_truncated = bool(
        (isinstance(request, dict) and request.get("_truncated"))
        or body.get("_truncated")
    )
    tools = body.get("tools")
    if not isinstance(tools, list):
        tools = []
    effective_system_prompt = _system_prompt(body)
    first_user_content = _as_text(kwargs.get("user_message"))
    payload = {
        "schema_version": "browsecomp_effective_request_v1",
        "run_id": os.environ.get(_ACTIVE_ENV, ""),
        "session_id": session_id,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "platform": kwargs.get("platform", ""),
        "model": kwargs.get("model", ""),
        "provider": kwargs.get("provider", ""),
        "api_mode": kwargs.get("api_mode", ""),
        "api_request_id": kwargs.get("api_request_id", ""),
        "api_call_count": kwargs.get("api_call_count"),
        "capture_complete": not request_payload_truncated,
        "request_payload_truncated": request_payload_truncated,
        "first_user_content": first_user_content,
        "effective_system_prompt": effective_system_prompt,
        "effective_system_prompt_sha256": hashlib.sha256(
            effective_system_prompt.encode("utf-8")
        ).hexdigest(),
        "request_tool_schemas": tools,
        "tool_names": [name for name in (_tool_name(tool) for tool in tools) if name],
        "request_reasoning": body.get("reasoning"),
        "request_include": body.get("include"),
        "request_max_output_tokens": body.get("max_output_tokens", body.get("max_tokens")),
        "request_store": body.get("store"),
        "request_parallel_tool_calls": body.get("parallel_tool_calls"),
        "request_fields": sorted(str(key) for key in body),
        "request_input_format": (
            "responses_input" if isinstance(body.get("input"), list) else "chat_messages"
        ),
    }
    _atomic_write_once(
        _capture_root() / f"session_{_safe_id(session_id)}.json",
        payload,
    )


def register(ctx) -> None:
    ctx.register_hook("pre_api_request", _on_pre_api_request)
