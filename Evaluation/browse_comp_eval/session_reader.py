from __future__ import annotations

import json
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


FINAL_FINISH_REASONS = {"stop", "end_turn", "done", "complete", "completed"}
TOOL_FINISH_REASONS = {"tool_calls", "function_call"}
RAW_TOOL_CALL_RE = re.compile(
    r"^\s*(?:<\s*tool_call\b|<\s*function\s*=|<\s*parameter\s*=)",
    re.IGNORECASE,
)
EMPTY_SENTINEL_RE = re.compile(r"^\(\s*empty\s*\)$", re.IGNORECASE)


def hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))).expanduser()


def session_snapshot_path(
    storage_root: str | Path,
    save_name: str,
    dataset: str,
    question_id: str,
    session_id: str,
    *,
    namespace: str = "main",
) -> Path:
    """Return the grouped Hermes JSON snapshot path for one session."""

    root = Path(storage_root).expanduser()
    if root.name != "sessions":
        root = root / "sessions"
    return (
        root
        / save_name
        / dataset
        / question_id
        / namespace
        / f"session_{session_id}.json"
    )


def load_session(
    session_id: str | None,
    profile: str | None = None,
    *,
    snapshot_path: str | Path | None = None,
) -> dict[str, Any] | None:
    if not session_id:
        return None

    home = hermes_home()
    snapshot = _load_json_session(Path(snapshot_path)) if snapshot_path else None
    candidates = []
    if profile:
        candidates.append(home / "profiles" / profile / "state.db")
    candidates.append(home / "state.db")

    for db_path in candidates:
        if not db_path.exists():
            continue
        session = _load_sqlite_session(db_path, session_id)
        if session:
            session["tools"] = _session_tools(snapshot)
            return session

    legacy_candidates = []
    if profile:
        legacy_candidates.append(home / "profiles" / profile / "sessions" / f"session_{session_id}.json")
    legacy_candidates.append(home / "sessions" / f"session_{session_id}.json")

    if snapshot:
        snapshot["tools"] = _session_tools(snapshot)
        return snapshot

    for path in legacy_candidates:
        legacy = _load_json_session(path)
        if legacy:
            legacy["tools"] = _session_tools(legacy)
            return legacy
    return None


def _load_json_session(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _session_tools(session: dict[str, Any] | None) -> list[Any]:
    if not session:
        return []
    tools = session.get("tools")
    return tools if isinstance(tools, list) else []


def _load_sqlite_session(db_path: Path, session_id: str) -> dict[str, Any] | None:
    conn = None
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute(
            "SELECT id, model, started_at, ended_at, system_prompt, message_count "
            "FROM sessions WHERE id=?",
            (session_id,),
        )
        row = cur.fetchone()
        if not row:
            return None

        session = {
            "session_id": row[0],
            "model": row[1],
            "session_start": _ts_to_iso(row[2]),
            "last_updated": _ts_to_iso(row[3]),
            "system_prompt": row[4] or "",
            "message_count": row[5] or 0,
            "messages": [],
        }

        if not session["last_updated"]:
            cur.execute("SELECT MAX(timestamp) FROM messages WHERE session_id=?", (session_id,))
            max_ts = cur.fetchone()
            if max_ts and max_ts[0]:
                session["last_updated"] = _ts_to_iso(max_ts[0])

        cur.execute("PRAGMA table_info(messages)")
        available = {row[1] for row in cur.fetchall()}
        wanted = [
            "role",
            "content",
            "tool_calls",
            "tool_name",
            "finish_reason",
            "tool_call_id",
            "inherited_from_previous_session",
        ]
        select_cols = [col if col in available else f"NULL AS {col}" for col in wanted]
        cur.execute(
            f"SELECT {', '.join(select_cols)} FROM messages WHERE session_id=? ORDER BY id",
            (session_id,),
        )
        for msg_row in cur.fetchall():
            (
                role,
                content,
                tool_calls_json,
                tool_name,
                finish_reason,
                tool_call_id,
                inherited,
            ) = msg_row
            msg = {
                "role": role,
                "content": content or "",
                "finish_reason": finish_reason,
            }
            if tool_calls_json:
                try:
                    msg["tool_calls"] = json.loads(tool_calls_json)
                except Exception:
                    msg["tool_calls_raw"] = tool_calls_json
            if tool_name:
                msg["tool_name"] = tool_name
            if tool_call_id:
                msg["tool_call_id"] = tool_call_id
            if inherited:
                msg["_inherited_from_previous_session"] = True
            session["messages"].append(msg)
        return session
    except Exception:
        return None
    finally:
        if conn is not None:
            conn.close()


def _ts_to_iso(value: Any) -> str:
    if not value:
        return ""
    try:
        return datetime.fromtimestamp(float(value)).isoformat()
    except Exception:
        return ""


def extract_last_assistant_message(session: dict[str, Any] | None) -> str:
    if not session:
        return ""
    for msg in reversed(session.get("messages", [])):
        if msg.get("role") == "assistant" and str(msg.get("content", "")).strip():
            return str(msg["content"]).strip()
    return ""


def text_from_content(content: Any) -> str:
    parts: list[str] = []

    def collect(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            text = value.strip()
            if text:
                parts.append(text)
            return
        if isinstance(value, list):
            for item in value:
                collect(item)
            return
        if isinstance(value, dict):
            for key in ("text", "content", "value"):
                if key in value:
                    collect(value.get(key))

    collect(content)
    return "\n".join(parts).strip()


def extract_final_assistant_message(session: dict[str, Any] | None) -> tuple[str, str]:
    if not session:
        return "", "missing session"

    messages = session.get("messages", [])
    if not messages:
        return "", "empty session messages"

    last = messages[-1]
    if not isinstance(last, dict):
        return "", "last message is not an object"

    role = last.get("role")
    if role != "assistant":
        return "", f"last message role is {role!r}, not assistant"

    if last.get("tool_calls") or last.get("tool_calls_raw"):
        return "", "final assistant still has tool_calls"

    finish_reason_raw = last.get("finish_reason")
    finish_reason = str(finish_reason_raw).strip().lower() if finish_reason_raw is not None else ""
    if finish_reason:
        if finish_reason in TOOL_FINISH_REASONS:
            return "", f"final assistant finish_reason is {finish_reason}"
        if finish_reason not in FINAL_FINISH_REASONS:
            return "", f"unexpected final assistant finish_reason: {finish_reason_raw}"

    text = text_from_content(last.get("content"))
    if not text:
        return "", "empty final assistant content"
    if EMPTY_SENTINEL_RE.fullmatch(text):
        return "", "final assistant content is Hermes empty sentinel"
    if _looks_like_raw_tool_call(text):
        return "", "final assistant content looks like a raw tool call"
    return text, ""


def _looks_like_raw_tool_call(text: str) -> bool:
    stripped = text.strip()
    if RAW_TOOL_CALL_RE.search(stripped):
        return True
    lowered = stripped.lower()
    return "<tool_call" in lowered and "</tool_call>" in lowered


def extract_trace(session: dict[str, Any] | None) -> dict[str, Any]:
    if not session:
        return {
            "rounds": 0,
            "total_tokens_estimate": 0,
            "duration_seconds": 0.0,
            "tool_calls": {},
        }

    messages = session.get("messages", [])
    total_tokens_estimate = sum(len(str(m.get("content", ""))) for m in messages) // 4
    duration = _duration_seconds(session.get("session_start", ""), session.get("last_updated", ""))
    tool_calls = extract_tool_calls(messages)
    return {
        "rounds": len(messages),
        "total_tokens_estimate": total_tokens_estimate,
        "duration_seconds": duration,
        "tool_calls": tool_calls,
    }


def extract_tool_calls(messages: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for msg in messages:
        if msg.get("role") == "tool" and msg.get("tool_name"):
            name = str(msg["tool_name"])
            counts[name] = counts.get(name, 0) + 1

    if counts:
        return counts

    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for call in msg.get("tool_calls", []) or []:
            fn = call.get("function", {}) if isinstance(call, dict) else {}
            name = (fn.get("name") or call.get("name")) if isinstance(call, dict) else None
            if name:
                counts[str(name)] = counts.get(str(name), 0) + 1
    return counts


def _duration_seconds(start: str, end: str) -> float:
    if not start or not end:
        return 0.0
    try:
        return round((datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds(), 2)
    except Exception:
        return 0.0
