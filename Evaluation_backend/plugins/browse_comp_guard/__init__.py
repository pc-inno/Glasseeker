"""BrowseComp benchmark-leak guard.

The plugin is opt-in and remains inert unless the process is running a
BrowseComp evaluation (``BROWSE_COMP_RUN_ID`` is set). It blocks benchmark-
aware requests before execution and removes benchmark resources or suspicious
question copies that appear in web results or browser pages. Question matching
is evaluation-only and can be disabled explicitly. A block is an ordinary tool
failure: the agent loop continues and the inference record is not marked as a
policy violation.
"""

from __future__ import annotations

import html
import json
import os
import re
import threading
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote_plus


_ACTIVE_ENV = "BROWSE_COMP_RUN_ID"
_QUESTION_ENV = "BROWSE_COMP_QUESTION"
_QUESTION_MATCH_MODE_ENV = "BROWSE_COMP_QUESTION_MATCH_MODE"
_BLOCK_MESSAGE = (
    "Blocked by the BrowseComp anti-cheat guard. Continue using ordinary "
    "first-party or independent web sources."
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
    r"|\bx[._/-]?\s*bench(?:\s*[._/-]?\s*deep\s*[._/-]?\s*search)?(?![a-z0-9])"
    r"|\bxds(?:\s*[._/-]?\s*(?:deep\s*[._/-]?\s*search|benchmark|bench"
    r"|dataset|eval|answers?|leaderboard|ground\s*[._/-]?\s*truth))?(?![a-z0-9])"
    r"|\bseal\s*[._/-]?\s*(?:0|qa)(?![a-z0-9])"
    r"|\bseal(?:0|qa)(?![a-z0-9])"
    r"|\bvtllms\s*/\s*sealqa\b"
    r"|\bsealqa:[a-f0-9]{8}:"
    r"|\bgaia\s*[._/-]?\s*(?:benchmark|bench|dataset|eval|validation|test|answers?|leaderboard)(?![a-z0-9])"
    r"|\b(?:benchmark|bench|dataset|eval|validation|test|answers?|leaderboard)\s*[._/-]?\s*gaia(?![a-z0-9])"
    r")",
    re.IGNORECASE,
)
_HOSTING_RE = re.compile(
    r"(?:"
    r"\bhugging\s*face\b"
    r"|\bhuggingface(?:\.co)?\b"
    r"|\bhf\.co\b"
    r"|cdn-lfs\.huggingface\.co\b"
    r"|huggingfaceusercontent\.com\b"
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
_EXEC_CODE_FILE_ACCESS_RE = re.compile(
    r"(?:"
    r"\bopen\s*\("
    r"|\b(?:Path|pathlib\.Path)\s*\("
    r"|\b(?:os|shutil|glob)\s*\."
    r"|\bsubprocess\s*\."
    r"|\bfrom\s+hermes_tools\s+import\s+(?:read_file|search_files|terminal)\b"
    r"|\bimport\s+hermes_tools\b"
    r"|\bread_file\s*\("
    r"|\bsearch_files\s*\("
    r"|\bterminal\s*\("
    r")",
    re.IGNORECASE,
)

# Only these model-visible result fields participate in question matching.
# URLs, request arguments, echoed queries, and diagnostic metadata are excluded.
_VISIBLE_RESULT_FIELDS = frozenset(
    {
        "content",
        "description",
        "excerpt",
        "html",
        "markdown",
        "page_content",
        "snippet",
        "snapshot",
        "text",
        "title",
    }
)
_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_QUESTION_STOP_WORDS = frozenset(
    """
    a an the and or of to in on at by for from with as is was were be been being
    that this these those it its their his her they them who whose whom which
    what when where how can could would should do does did have has had into
    about after before between during than then also one same named name tell
    identify find give according based among within inclusive please
    """.split()
)
_STRONG_LEAK_MARKER_RE = re.compile(
    r"(?:ground\s*truth|reference\s*answer|gold(?:en)?\s*answer|"
    r"expected\s*answer|benchmark\s*answer|answer\s*key|labelled?\s*answer)",
    re.IGNORECASE,
)
_AGENT_EVALUATION_CONTEXT_RE = re.compile(
    r"(?:web\s*agent|search\s*agent|agentic\s*search|deep\s*research\s*agent|"
    r"benchmark\s*trajectory|reasoning\s*trajectory|evaluation\s*trace|"
    r"context\s*folding|agentfold|metaagent|fineverify|\bkarl\b|\bresum\b)",
    re.IGNORECASE,
)
_WEAK_ANSWER_MARKER_RE = re.compile(
    r"(?:\bfinal\s*answer\b|\banswer\s*:|\bquestion\s*:|\btask\s*:|\bsolution\s*:)",
    re.IGNORECASE,
)
_quarantined_browser_sessions: set[tuple[str, str]] = set()
_quarantine_lock = threading.Lock()


@dataclass(frozen=True)
class _QuestionOverlap:
    question_token_count: int
    longest_contiguous_match: int
    question_token_recall: float
    matched_fivegram_count: int
    fivegram_recall: float
    exact_normalized_question_substring: bool


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
    if _BENCHMARK_RE.search(normalized) or _HOSTING_RE.search(normalized):
        return True
    # Catch simple runtime-string construction such as
    # ``'Browse' + 'Comp'`` and ``['browse', 'comp'].join('')``.
    collapsed = re.sub(r"[\s'\"`+,\[\](){}]", "", normalized)
    return bool(_BENCHMARK_RE.search(collapsed) or _HOSTING_RE.search(collapsed))


def _question_match_enabled() -> bool:
    """Question-overlap checks are evaluation-only and default to enabled."""
    mode = os.environ.get(_QUESTION_MATCH_MODE_ENV, "evaluation").strip().lower()
    return _active() and mode == "evaluation" and bool(
        os.environ.get(_QUESTION_ENV, "").strip()
    )


def _word_tokens(value: str) -> list[str]:
    return _WORD_RE.findall(html.unescape(value).lower())


def _longest_contiguous_match(question: list[str], result: list[str]) -> int:
    """Return the longest common contiguous token span using O(len(question)) memory."""
    if not question or not result:
        return 0
    previous = [0] * (len(question) + 1)
    longest = 0
    for result_token in result:
        current = [0] * (len(question) + 1)
        for index, question_token in enumerate(question, start=1):
            if question_token == result_token:
                current[index] = previous[index - 1] + 1
                longest = max(longest, current[index])
        previous = current
    return longest


def _question_overlap(question: str, result_text: str) -> _QuestionOverlap:
    question_words = _word_tokens(question)
    result_words = _word_tokens(result_text)
    question_tokens = [
        token for token in question_words if token not in _QUESTION_STOP_WORDS
    ]
    result_tokens = [
        token for token in result_words if token not in _QUESTION_STOP_WORDS
    ]

    question_set = set(question_tokens)
    result_set = set(result_tokens)
    token_recall = (
        len(question_set & result_set) / len(question_set) if question_set else 0.0
    )
    question_fivegrams = {
        tuple(question_tokens[index : index + 5])
        for index in range(max(0, len(question_tokens) - 4))
    }
    result_fivegrams = {
        tuple(result_tokens[index : index + 5])
        for index in range(max(0, len(result_tokens) - 4))
    }
    matched_fivegrams = len(question_fivegrams & result_fivegrams)
    fivegram_recall = (
        matched_fivegrams / len(question_fivegrams) if question_fivegrams else 0.0
    )
    normalized_question = " ".join(question_words)
    normalized_result = " ".join(result_words)
    return _QuestionOverlap(
        question_token_count=len(question_tokens),
        longest_contiguous_match=_longest_contiguous_match(
            question_tokens, result_tokens
        ),
        question_token_recall=token_recall,
        matched_fivegram_count=matched_fivegrams,
        fivegram_recall=fivegram_recall,
        exact_normalized_question_substring=bool(
            normalized_question and normalized_question in normalized_result
        ),
    )


def _visible_result_text(value: Any, visible: bool = False) -> str:
    """Extract visible fields while deliberately ignoring URLs and metadata."""
    if isinstance(value, str):
        return value if visible else ""
    if isinstance(value, list):
        return " ".join(_visible_result_text(item, visible) for item in value)
    if not isinstance(value, dict):
        return ""
    parts = []
    for key, child in value.items():
        child_visible = visible or str(key).lower() in _VISIBLE_RESULT_FIELDS
        text = _visible_result_text(child, child_visible)
        if text:
            parts.append(text)
    return " ".join(parts)


def _contains_question_leak(value: Any, question: str) -> bool:
    result_text = _visible_result_text(value)
    if not result_text:
        return False
    overlap = _question_overlap(question, result_text)

    # A: the full normalized question, or an almost-complete paraphrase.
    criterion_a = overlap.exact_normalized_question_substring or (
        overlap.fivegram_recall >= 0.80
        and overlap.question_token_recall >= 0.85
    )
    # B: a long verbatim span; 20 effective words is independently sufficient.
    criterion_b = (
        overlap.longest_contiguous_match >= 16
        and overlap.question_token_recall >= 0.40
    ) or overlap.longest_contiguous_match >= 20
    # C: distributed phrase reuse, guarded by minimum question and match sizes.
    criterion_c = (
        overlap.question_token_count >= 15
        and overlap.matched_fivegram_count >= 4
        and overlap.fivegram_recall >= 0.40
        and overlap.question_token_recall >= 0.50
    )

    strong_leak_marker = bool(_STRONG_LEAK_MARKER_RE.search(result_text))
    agent_evaluation_context = bool(
        _AGENT_EVALUATION_CONTEXT_RE.search(result_text)
    )
    weak_answer_marker = bool(_WEAK_ANSWER_MARKER_RE.search(result_text))
    # D-strong: explicit gold/reference-answer language permits lower overlap.
    criterion_d_strong = strong_leak_marker and (
        (
            overlap.longest_contiguous_match >= 6
            and overlap.question_token_recall >= 0.20
        )
        or overlap.fivegram_recall >= 0.10
    )
    # D-agent: evaluation/agent traces are suspicious at moderate overlap.
    criterion_d_agent = agent_evaluation_context and (
        (
            overlap.longest_contiguous_match >= 8
            and overlap.question_token_recall >= 0.20
        )
        or overlap.fivegram_recall >= 0.15
    )
    # D-weak: generic answer labels need stronger overlap to avoid false positives.
    criterion_d_weak = weak_answer_marker and (
        (
            overlap.longest_contiguous_match >= 10
            and overlap.question_token_recall >= 0.30
        )
        or overlap.fivegram_recall >= 0.25
    )
    return (
        criterion_a
        or criterion_b
        or criterion_c
        or criterion_d_strong
        or criterion_d_agent
        or criterion_d_weak
    )


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


def _execute_code_has_file_access(tool_name: str, args: Any) -> bool:
    if tool_name != "execute_code":
        return False
    args = args if isinstance(args, dict) else {}
    code = _normalize_text(args.get("code") or "")
    return bool(_EXEC_CODE_FILE_ACCESS_RE.search(code))


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

    key = _browser_key(session_id, task_id)
    if (
        tool_name.startswith("browser_")
        and _is_quarantined(key)
        and tool_name not in _BROWSER_RECOVERY_TOOLS
    ):
        return {"action": "block", "message": _BLOCK_MESSAGE}

    if _browser_request_has_network_escape(tool_name, args):
        return {"action": "block", "message": _BLOCK_MESSAGE}

    if _execute_code_has_file_access(tool_name, args):
        return {"action": "block", "message": _BLOCK_MESSAGE}

    material = _request_material(tool_name, args)
    if material is not None and _contains_benchmark_material(material):
        return {"action": "block", "message": _BLOCK_MESSAGE}
    return None


def _filter_candidate_lists(value: Any, question: str = "") -> tuple[Any, int, int]:
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
                if _contains_benchmark_material(item) or (
                    question and _contains_question_leak(item, question)
                ):
                    removed += 1
                else:
                    filtered.append(item)
                    kept += 1
            output[key] = filtered
            continue
        if isinstance(child, dict):
            nested, nested_removed, nested_kept = _filter_candidate_lists(
                child, question
            )
            output[key] = nested
            removed += nested_removed
            kept += nested_kept
        else:
            output[key] = child
    return output, removed, kept


def _filter_web_result(result: str) -> str | None:
    benchmark_material = _contains_benchmark_material(result)
    question = (
        os.environ.get(_QUESTION_ENV, "") if _question_match_enabled() else ""
    )
    if not benchmark_material and not question:
        return None
    try:
        parsed = json.loads(result)
    except (TypeError, ValueError):
        # Benchmark markers remain fail-closed. Question matching is applied
        # only to structured result items so a wrapper/query echo cannot block.
        return _failure_result() if benchmark_material else None

    filtered, removed, kept = _filter_candidate_lists(parsed, question)
    if not removed:
        return _failure_result() if benchmark_material else None
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
        question_leak = False
        if _question_match_enabled():
            try:
                browser_result = json.loads(result)
            except (TypeError, ValueError):
                browser_result = {}
            question_leak = _contains_question_leak(
                browser_result, os.environ.get(_QUESTION_ENV, "")
            )
        if _contains_benchmark_material(result) or question_leak or (
            current_url and _contains_benchmark_material(current_url)
        ):
            _set_quarantined(key, True)
            return _failure_result()
        if tool_name in _BROWSER_RECOVERY_TOOLS:
            _set_quarantined(key, False)
        return None

    if tool_name == "vision_analyze":
        question_leak = _question_match_enabled() and _contains_question_leak(
            {"content": result}, os.environ.get(_QUESTION_ENV, "")
        )
        if _contains_benchmark_material(result) or question_leak:
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
