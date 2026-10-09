"""BrowseComp benchmark-leak guard.

The plugin is opt-in and remains inert unless the process is running a
BrowseComp evaluation (``BROWSE_COMP_RUN_ID`` is set). It blocks benchmark-
aware requests before execution and removes benchmark resources that appear in
web results or browser pages. A block is an ordinary tool failure: the agent
loop continues and the inference record is not marked as a policy violation.
"""

from __future__ import annotations

import html
import json
import os
import re
import threading
from typing import Any
from urllib.parse import unquote_plus


_ACTIVE_ENV = "BROWSE_COMP_RUN_ID"
_BLOCK_MESSAGE = (
    "Blocked by the BrowseComp anti-cheat guard. Continue using ordinary "
    "first-party or independent web sources."
)
_SYNC_DELEGATION_MESSAGE = (
    "Background delegation is disabled for this evaluation run. Call "
    "delegate_task again with background=false and wait for the child result."
)

# Match canonical names plus the variants observed in real cheating traces.
# Decode before matching so percent-encoding and nested URL parameters do not
# provide an escape hatch.
_BENCHMARK_RE = re.compile(
    r"(?:"
    r"\bbrowse\s*[._/-]?\s*comp(?:arison)?(?:\s*[._/-]?\s*plus)?(?![a-z0-9])"
    r"|\bbrowsecomp(?:[._/-]?plus)?(?![a-z0-9])"
    r"|\bbc[._/-]?plus\b"
    r"|\bbc_eval_\d"
    r")",
    re.IGNORECASE,
)

_SEARCH_TOOLS = frozenset({"web_search", "search_server"})
_EXTRACT_TOOLS = frozenset({"web_extract", "fetch_server"})
_BROWSER_RECOVERY_TOOLS = frozenset({"browser_back", "browser_navigate"})
_BROWSER_URL_CHECK_TOOLS = frozenset({
    "browser_navigate",
    "browser_click",
    "browser_type",
    "browser_press",
    "browser_back",
    "browser_console",
    "browser_cdp",
})
_quarantined_browser_sessions: set[tuple[str, str]] = set()
_quarantine_lock = threading.Lock()


def _active() -> bool:
    return bool(os.environ.get(_ACTIVE_ENV, "").strip())


def _normalize_text(value: Any) -> str:
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            text = str(value)

    # Repeated decoding handles URLs embedded inside encoded query parameters.
    for _ in range(3):
        decoded = html.unescape(unquote_plus(text))
        if decoded == text:
            break
        text = decoded
    return " ".join(text.replace("\\", "/").split())


def _contains_benchmark_material(value: Any) -> bool:
    normalized = _normalize_text(value)
    if _BENCHMARK_RE.search(normalized):
        return True
    # Catch simple runtime-string construction such as
    # ``'Browse' + 'Comp'`` and ``['browse', 'comp'].join('')``.
    collapsed = re.sub(r"[\s'\"`+,\[\](){}]", "", normalized)
    return bool(_BENCHMARK_RE.search(collapsed))


def _browser_request_has_network_escape(tool_name: str, args: Any) -> bool:
    args = args if isinstance(args, dict) else {}
    if tool_name == "browser_console" and args.get("expression") is not None:
        expression = _normalize_text(args.get("expression")).lower()
        # Console evaluation remains available for DOM inspection and local
        # computation, but it may not become an unfiltered HTTP client.
        return any(
            marker in expression
            for marker in (
                "fetch",
                "xmlhttprequest",
                "websocket",
                "eventsource",
                "sendbeacon",
                "window.open",
                "createelement",
                "location=",
                "location.href=",
            )
        )
    if tool_name == "browser_cdp":
        method = str(args.get("method") or "").strip().lower()
        return (
            method in {
                "page.navigate",
                "page.getresourcecontent",
                "page.searchinresource",
                "target.createtarget",
                "runtime.evaluate",
                "runtime.callfunctionon",
                "debugger.evaluateoncallframe",
            }
            or method.startswith(("fetch.", "network."))
        )
    return False


def _failure_result() -> str:
    return json.dumps(
        {"success": False, "error": _BLOCK_MESSAGE},
        ensure_ascii=False,
    )


def _browser_key(session_id: str = "", task_id: str = "") -> tuple[str, str]:
    run_id = os.environ.get(_ACTIVE_ENV, "").strip()
    return (str(session_id or run_id), str(task_id or run_id))


def _is_quarantined(key: tuple[str, str]) -> bool:
    with _quarantine_lock:
        return key in _quarantined_browser_sessions


def _set_quarantined(key: tuple[str, str], value: bool) -> None:
    with _quarantine_lock:
        if value:
            _quarantined_browser_sessions.add(key)
        else:
            _quarantined_browser_sessions.discard(key)


def _request_material(tool_name: str, args: Any) -> Any:
    args = args if isinstance(args, dict) else {}
    if tool_name in _SEARCH_TOOLS:
        return {"query": args.get("query"), "queries": args.get("queries")}
    if tool_name in _EXTRACT_TOOLS:
        return args.get("urls") or args.get("url")
    if tool_name == "browser_navigate":
        return args.get("url")
    if tool_name == "browser_type":
        return args.get("text") or args.get("value")
    if tool_name in {"browser_console", "browser_cdp"}:
        return args
    if tool_name == "vision_analyze":
        return args
    return None


def _on_pre_tool_call(
    tool_name: str = "",
    args: Any = None,
    session_id: str = "",
    task_id: str = "",
    **_: Any,
) -> dict[str, str] | None:
    if not _active():
        return None

    if (
        tool_name == "delegate_task"
        and isinstance(args, dict)
        and args.get("background") is True
    ):
        return {"action": "block", "message": _SYNC_DELEGATION_MESSAGE}

    key = _browser_key(session_id, task_id)
    if (
        tool_name.startswith("browser_")
        and _is_quarantined(key)
        and tool_name not in _BROWSER_RECOVERY_TOOLS
    ):
        return {"action": "block", "message": _BLOCK_MESSAGE}

    if _browser_request_has_network_escape(tool_name, args):
        return {"action": "block", "message": _BLOCK_MESSAGE}

    material = _request_material(tool_name, args)
    if material is not None and _contains_benchmark_material(material):
        return {"action": "block", "message": _BLOCK_MESSAGE}
    return None


def _filter_candidate_lists(value: Any) -> tuple[Any, int, int]:
    """Remove blocked web result/doc entries, returning value/removed/kept."""
    if not isinstance(value, dict):
        return value, 0, 0

    removed = 0
    kept = 0
    output: dict[str, Any] = {}
    for key, child in value.items():
        if key in {"web", "results", "documents"} and isinstance(child, list):
            filtered = []
            for item in child:
                if _contains_benchmark_material(item):
                    removed += 1
                else:
                    filtered.append(item)
                    kept += 1
            output[key] = filtered
            continue
        if isinstance(child, dict):
            nested, nested_removed, nested_kept = _filter_candidate_lists(child)
            output[key] = nested
            removed += nested_removed
            kept += nested_kept
        else:
            output[key] = child
    return output, removed, kept


def _filter_web_result(result: str) -> str | None:
    if not _contains_benchmark_material(result):
        return None
    try:
        parsed = json.loads(result)
    except (TypeError, ValueError):
        # Wrapped/non-JSON results cannot be selectively sanitized safely.
        return _failure_result()

    filtered, removed, kept = _filter_candidate_lists(parsed)
    if not removed:
        return _failure_result()
    if kept == 0:
        return _failure_result()
    if _contains_benchmark_material(filtered):
        # A marker outside the result/document lists (for example a benchmark
        # query echoed in metadata) must not be sent back to the model.
        return _failure_result()
    return json.dumps(filtered, ensure_ascii=False)


def _result_succeeded(result: str) -> bool:
    try:
        parsed = json.loads(result)
    except (TypeError, ValueError):
        return bool(re.search(r'"success"\s*:\s*true', result, re.IGNORECASE))
    return isinstance(parsed, dict) and parsed.get("success") is True


def _current_browser_url(task_id: str) -> str:
    """Read the post-action URL without re-entering the model tool hooks."""
    try:
        from tools.browser_tool import _last_session_key, _run_browser_command

        session_key = _last_session_key(task_id or "default")
        result = _run_browser_command(
            session_key,
            "eval",
            ["window.location.href"],
            timeout=10,
        )
        if not isinstance(result, dict) or not result.get("success"):
            return ""
        value = (result.get("data") or {}).get("result", "")
        if not isinstance(value, str):
            return ""
        return value.strip().strip('"').strip("'")
    except Exception:
        # The result-content check remains active if a backend cannot report
        # its current URL (for example, a cloud backend without agent-browser).
        return ""


def _on_transform_tool_result(
    tool_name: str = "",
    result: Any = None,
    session_id: str = "",
    task_id: str = "",
    **_: Any,
) -> str | None:
    if not _active() or not isinstance(result, str):
        return None

    if tool_name in _SEARCH_TOOLS or tool_name in _EXTRACT_TOOLS:
        return _filter_web_result(result)

    key = _browser_key(session_id, task_id)
    if tool_name.startswith("browser_"):
        current_url = ""
        if tool_name in _BROWSER_URL_CHECK_TOOLS and _result_succeeded(result):
            current_url = _current_browser_url(task_id)
        if _contains_benchmark_material(result) or (
            current_url and _contains_benchmark_material(current_url)
        ):
            _set_quarantined(key, True)
            return _failure_result()
        if tool_name in _BROWSER_RECOVERY_TOOLS:
            _set_quarantined(key, False)
        return None

    if tool_name == "vision_analyze":
        if _contains_benchmark_material(result):
            return _failure_result()
    return None


def _on_session_end(session_id: str = "", task_id: str = "", **_: Any) -> None:
    if not _active():
        return
    _set_quarantined(_browser_key(session_id, task_id), False)


def register(ctx) -> None:
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_hook("transform_tool_result", _on_transform_tool_result)
    ctx.register_hook("on_session_end", _on_session_end)
