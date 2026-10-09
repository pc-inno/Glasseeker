#!/usr/bin/env python3
"""Export unique v49 solver and successful sub-agent trajectories."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "data" / "runs_v49"
CONVERSATION_ROOT = RUN_ROOT / "conversations"
OUTPUT_ROOT = RUN_ROOT / "result"
HERMES_ROOT = PROJECT_ROOT.parent / "hermes-agent-gpt55-trace"
THRESHOLD = 40

SOLVER_FILES = {
    "hard": "hard_correct_solver_traces.jsonl",
    "all_wrong": "all_wrong_correct_solver_traces.jsonl",
    "easy_over40": "easy_over40_correct_solver_traces.jsonl",
    "easy_under40": "easy_under40_correct_solver_traces.jsonl",
}
SUBAGENT_FILES = {
    "hard": "hard_correct_subagent_traces.jsonl",
    "all_wrong": "all_wrong_correct_subagent_traces.jsonl",
    "easy_over40": "easy_over40_correct_subagent_traces.jsonl",
    "easy_under40": "easy_under40_correct_subagent_traces.jsonl",
}


def _signature(report: dict[str, Any]) -> str:
    payload = {
        key: report.get(key)
        for key in ("final_answer", "confidence", "evidence", "reasoning_summary", "_execution")
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _literal_assignment(path: Path, name: str) -> Any:
    """Read a literal schema without importing Hermes or its optional deps."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return None
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        if not any(isinstance(target, ast.Name) and target.id == name for target in targets):
            continue
        try:
            return ast.literal_eval(node.value)
        except (ValueError, TypeError, SyntaxError):
            return _schema_literal(node.value)
    return None


def _schema_literal(node: ast.AST) -> Any:
    """Evaluate schema literals while rendering harmless dynamic f-strings."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return f"<{node.id}>"
    if isinstance(node, ast.JoinedStr):
        pieces = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                pieces.append(str(value.value))
            elif isinstance(value, ast.FormattedValue):
                if isinstance(value.value, ast.Constant):
                    pieces.append(str(value.value.value))
                elif isinstance(value.value, ast.Name):
                    pieces.append(f"<{value.value.id}>")
                else:
                    pieces.append("<dynamic>")
        return "".join(pieces)
    if isinstance(node, ast.Dict):
        return {
            _schema_literal(key): _schema_literal(value)
            for key, value in zip(node.keys, node.values)
            if key is not None
        }
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        values = [_schema_literal(item) for item in node.elts]
        return tuple(values) if isinstance(node, ast.Tuple) else values
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _schema_literal(node.operand)
        return -value if isinstance(node.op, ast.USub) and isinstance(value, (int, float)) else value
    return None


def _tool_schemas() -> dict[str, dict[str, Any]]:
    """Load static schemas without requiring the remote Hermes environment.

    A trace only records the tools that were called, not the request's full
    schema array. The source checkout contains literal schemas for the web,
    browser, terminal, delegation, and vision tools. Existing v49 exports are
    used as a fallback for dynamic schemas such as ``execute_code``.
    """
    schemas: dict[str, dict[str, Any]] = {}
    previous_files = list(OUTPUT_ROOT.glob("*_solver_traces.jsonl")) + list(
        OUTPUT_ROOT.glob("*_subagent_traces.jsonl")
    )
    for path in previous_files:
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                for name, schema in (row.get("tool_schemas") or {}).items():
                    if isinstance(schema, dict):
                        schemas.setdefault(name, schema)
        except (OSError, json.JSONDecodeError):
            continue

    source_constants = {
        "web_search": (HERMES_ROOT / "tools" / "web_tools.py", "WEB_SEARCH_SCHEMA"),
        "web_extract": (HERMES_ROOT / "tools" / "web_tools.py", "WEB_EXTRACT_SCHEMA"),
        "terminal": (HERMES_ROOT / "tools" / "terminal_tool.py", "TERMINAL_SCHEMA"),
        "delegate_task": (HERMES_ROOT / "tools" / "delegate_tool.py", "DELEGATE_TASK_SCHEMA"),
        "vision_analyze": (HERMES_ROOT / "tools" / "vision_tools.py", "VISION_ANALYZE_SCHEMA"),
    }
    for name, (path, constant) in source_constants.items():
        value = _literal_assignment(path, constant)
        if isinstance(value, dict):
            schemas[name] = value

    browser_list = _literal_assignment(
        HERMES_ROOT / "tools" / "browser_tool.py", "BROWSER_TOOL_SCHEMAS"
    )
    if isinstance(browser_list, list):
        for schema in browser_list:
            if isinstance(schema, dict) and schema.get("name"):
                schemas[str(schema["name"])] = schema
    # ``execute_code`` is generated by a function because its description
    # changes with sandbox capabilities. Keep the stable OpenAI parameter
    # shape when the dynamic runtime module cannot be imported here.
    schemas.setdefault(
        "execute_code",
        {
            "name": "execute_code",
            "description": "Run a Python script for in-memory computation and optional Hermes tool calls.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute."}
                },
                "required": ["code"],
            },
        },
    )
    return schemas


def _api_counts(conversation: dict[str, Any]) -> dict[str, Any]:
    sessions = conversation.get("trace_sessions") or []
    session_calls = [
        session.get("api_call_count")
        for session in sessions
        if isinstance(session, dict)
    ]
    complete = bool(sessions) and all(isinstance(value, int) for value in session_calls)
    primary_id = str(conversation.get("session_id") or "")
    main = None
    if complete:
        for session in sessions:
            session_id = str(session.get("session_id") or session.get("id") or "")
            if session_id == primary_id:
                main = session.get("api_call_count")
                break
        if main is None:
            main = session_calls[0]
    if main is None:
        main = _raw_api_calls(conversation.get("raw_response"))
    total = sum(session_calls) if complete else main
    return {
        "main_session_api_calls": main,
        "all_session_api_calls": total,
        "scope": "all_sessions" if complete else "main_only_lower_bound",
        "session_count": len(sessions) if complete else None,
        "subagent_session_count": len(conversation.get("subagent_sessions") or [])
        if complete
        else None,
    }


def _raw_api_calls(raw_response: Any) -> int | None:
    if not isinstance(raw_response, str):
        return None
    decoder = json.JSONDecoder()
    for index, char in enumerate(raw_response):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw_response[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "messages" in value and isinstance(value.get("api_calls"), int):
            return value["api_calls"]
    return None


def _tool_names(trace_sessions: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for session in trace_sessions:
        for message in session.get("messages") or []:
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                if function.get("name"):
                    names.add(str(function["name"]))
    return names


def _delegate_specs(conversation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Map child session ids to the exact delegate_task goal/context inputs."""
    specs: dict[str, dict[str, Any]] = {}
    sessions = conversation.get("trace_sessions") or []
    for parent in sessions:
        for message in parent.get("messages") or []:
            calls = message.get("tool_calls") or []
            delegate_calls = [
                call
                for call in calls
                if (call.get("function") or {}).get("name") == "delegate_task"
            ]
            if not delegate_calls:
                continue
            delegate_args: list[dict[str, Any]] = []
            for call in delegate_calls:
                try:
                    args = json.loads((call.get("function") or {}).get("arguments", "{}"))
                except (TypeError, json.JSONDecodeError):
                    continue
                if isinstance(args.get("tasks"), list):
                    delegate_args.extend(
                        item for item in args["tasks"] if isinstance(item, dict)
                    )
                elif isinstance(args, dict):
                    delegate_args.append(args)
            result_message = next(
                (
                    candidate
                    for candidate in parent.get("messages") or []
                    if candidate.get("role") == "tool"
                    and candidate.get("tool_name") == "delegate_task"
                    and candidate.get("tool_call_id")
                    == delegate_calls[0].get("id")
                ),
                None,
            )
            if result_message is None:
                continue
            try:
                result_payload = json.loads(str(result_message.get("content") or ""))
            except json.JSONDecodeError:
                continue
            results = result_payload.get("results") or []
            for result in results:
                if not isinstance(result, dict):
                    continue
                session_id = str(result.get("session_id") or "")
                index = result.get("task_index")
                if isinstance(index, int) and index < len(delegate_args):
                    spec = delegate_args[index]
                    if session_id:
                        specs[session_id] = spec
                    # Hermes v49's delegate result did not serialize the
                    # child session id. Keep a goal index so the child trace
                    # can still be matched by its first user message.
                    goal = str(spec.get("goal") or "")
                    if goal:
                        specs["__goal__" + goal] = spec
    return specs


def _child_system_addition(spec: dict[str, Any]) -> str:
    """Mirror Hermes ``_build_child_system_prompt`` for leaf children.

    The v49 solver delegates only leaf tasks. The ephemeral child prompt is
    not persisted by Hermes, so it must be reconstructed from the recorded
    delegate_task arguments.
    """
    goal = str(spec.get("goal") or "")
    context = spec.get("context")
    parts = [
        "You are a focused subagent working on a specific delegated task.",
        "",
        f"YOUR TASK:\n{goal}",
    ]
    if context and str(context).strip():
        parts.append(f"\nCONTEXT:\n{context}")
    parts.append(
        "\nComplete this task using the tools available to you. "
        "When finished, provide a clear, concise summary of:\n"
        "- What you did\n"
        "- What you found or accomplished\n"
        "- Any files you created or modified\n"
        "- Any issues encountered\n\n"
        "Important workspace rule: Never assume a repository lives at /workspace/... or any other container-style path unless the task/context explicitly gives that path. "
        "If no exact local path is provided, discover it first before issuing git/workdir-specific commands.\n\n"
        "Be thorough but concise -- your response is returned to the "
        "parent agent as a summary."
    )
    return "\n".join(parts)


def _effective_system_prompt(
    session: dict[str, Any],
    conversation: dict[str, Any],
    *,
    is_main: bool,
) -> tuple[str, str]:
    """Return the actual runtime system message and how it was reconstructed."""
    base = str(session.get("system_prompt") or "")
    if is_main:
        addition = ""
        source = "hermes_persisted_base (research guidance is in user content)"
    else:
        specs = _delegate_specs(conversation)
        session_id = str(session.get("session_id") or session.get("id") or "")
        spec = specs.get(session_id)
        if not spec:
            first_user = next(
                (
                    message.get("content")
                    for message in session.get("messages") or []
                    if message.get("role") == "user"
                ),
                "",
            )
            spec = specs.get("__goal__" + str(first_user or ""))
        if not spec:
            # This should not occur for v49, but retaining the base prompt is
            # preferable to fabricating a child task prompt.
            addition = ""
            source = "hermes_persisted_base (child ephemeral prompt unavailable)"
        else:
            addition = _child_system_addition(spec)
            source = "hermes_persisted_base + Hermes delegate child prompt(ephemeral_system_prompt)"
    effective = (base + "\n\n" + addition).strip() if addition else base
    return effective, source


def _openai_tool_call(call: dict[str, Any]) -> dict[str, Any] | None:
    function = call.get("function") or {}
    name = function.get("name")
    if not name:
        return None
    call_id = call.get("id") or call.get("call_id") or call.get("response_item_id")
    arguments = function.get("arguments", "{}")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    return {
        "id": str(call_id or ""),
        "type": "function",
        "function": {"name": str(name), "arguments": arguments},
    }


def _openai_message(message: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a Hermes DB message to an OpenAI Chat Completions message."""
    role = message.get("role")
    if role == "user":
        return {"role": "user", "content": message.get("content") or ""}
    if role == "tool":
        result = {
            "role": "tool",
            "tool_call_id": str(message.get("tool_call_id") or ""),
            "content": message.get("content") or "",
        }
        return result
    if role != "assistant":
        return None
    calls = [
        converted
        for call in message.get("tool_calls") or []
        if (converted := _openai_tool_call(call)) is not None
    ]
    result: dict[str, Any] = {
        "role": "assistant",
        "content": message.get("content") or "",
    }
    if calls:
        result["tool_calls"] = calls
    # DeepSeek V4 Pro requires this provider-facing field to be echoed on
    # subsequent calls. It is kept in the OpenAI-shaped message for replay.
    reasoning_content = message.get("reasoning_content")
    if isinstance(reasoning_content, str):
        result["reasoning_content"] = reasoning_content
    return result


def _message_metadata(message: dict[str, Any]) -> dict[str, Any]:
    """Keep observability fields that are not part of OpenAI messages."""
    return {
        "id": message.get("id"),
        "timestamp": message.get("timestamp"),
        "token_count": message.get("token_count"),
        "finish_reason": message.get("finish_reason"),
        "tool_name": message.get("tool_name"),
        "platform_message_id": message.get("platform_message_id"),
    }


def _openai_tools(names: set[str]) -> tuple[list[dict[str, Any]], list[str]]:
    tools: list[dict[str, Any]] = []
    missing: list[str] = []
    for name in sorted(names):
        schema = TOOL_SCHEMAS.get(name)
        if not isinstance(schema, dict):
            missing.append(name)
            continue
        function = {
            "name": schema.get("name", name),
            "description": schema.get("description", ""),
            "parameters": schema.get("parameters", {"type": "object", "properties": {}}),
        }
        tools.append({"type": "function", "function": function})
    return tools, missing


_TOOLSET_NAMES = {
    "browser": {
        "browser_navigate", "browser_snapshot", "browser_click", "browser_type",
        "browser_scroll", "browser_back", "browser_press", "browser_get_images",
        "browser_vision", "browser_console",
    },
    "web": {"web_search", "web_extract"},
    "code_execution": {"execute_code"},
    "delegation": {"delegate_task"},
    "vision": {"vision_analyze"},
    "terminal": {"terminal"},
}


def _available_tool_names(
    session: dict[str, Any], conversation: dict[str, Any], *, is_main: bool
) -> set[str]:
    """Approximate the full tool schema array sent to this Hermes session."""
    if is_main:
        toolsets = conversation.get("agent", {}).get("enabled_toolsets") or []
    else:
        specs = _delegate_specs(conversation)
        sid = str(session.get("session_id") or session.get("id") or "")
        spec = specs.get(sid)
        if not spec:
            first_user = next(
                (
                    message.get("content")
                    for message in session.get("messages") or []
                    if message.get("role") == "user"
                ),
                "",
            )
            spec = specs.get("__goal__" + str(first_user or ""))
        toolsets = (spec or {}).get("toolsets") or conversation.get("agent", {}).get("enabled_toolsets") or []
        # Leaf delegation strips the parent's code_execution/delegation
        # toolsets. Explicit v49 child toolsets are web/terminal, so this is
        # only relevant for older records with omitted toolsets.
        if not (spec or {}).get("toolsets"):
            toolsets = [name for name in toolsets if name not in {"code_execution", "delegation"}]
    names = set(_tool_names([session]))
    for toolset in toolsets:
        names.update(_TOOLSET_NAMES.get(str(toolset), set()))
    return names


def _session_openai_record(
    session: dict[str, Any], conversation: dict[str, Any], *, is_main: bool
) -> dict[str, Any]:
    effective_system, prompt_source = _effective_system_prompt(
        session, conversation, is_main=is_main
    )
    raw_messages = session.get("messages") or []
    messages = [{"role": "system", "content": effective_system}]
    message_metadata = []
    for raw_message in raw_messages:
        converted = _openai_message(raw_message)
        if converted is not None:
            messages.append(converted)
            message_metadata.append(_message_metadata(raw_message))
    used_names = _tool_names([session])
    names = _available_tool_names(session, conversation, is_main=is_main)
    tools, missing = _openai_tools(names)
    return {
        "session_id": session.get("session_id") or session.get("id"),
        "source": session.get("source"),
        "parent_session_id": session.get("parent_session_id"),
        "model": session.get("model"),
        "billing_provider": session.get("billing_provider"),
        "billing_base_url": session.get("billing_base_url"),
        "end_reason": session.get("end_reason"),
        "messages": messages,
        "tools": tools,
        "tool_names_used": sorted(used_names),
        "available_tool_names": sorted(names),
        "missing_tool_schemas": missing,
        "effective_system_prompt": effective_system,
        "effective_system_prompt_sha256": hashlib.sha256(
            effective_system.encode("utf-8")
        ).hexdigest(),
        "system_prompt_source": prompt_source,
        "message_metadata": message_metadata,
        "api_call_count": session.get("api_call_count"),
        "tool_call_count": session.get("tool_call_count"),
        "input_tokens": session.get("input_tokens"),
        "output_tokens": session.get("output_tokens"),
        "reasoning_tokens": session.get("reasoning_tokens"),
        "complete": bool(session.get("ended_at")) or session.get("end_reason") in {"completed", "stop", "success"},
    }


def _successful_subagent_sessions(conversation: dict[str, Any]) -> list[dict[str, Any]]:
    sessions = conversation.get("trace_sessions") or []
    children = [
        session
        for session in sessions
        if str(session.get("source") or "").lower() in {"subagent", "delegate", "delegation"}
        or session.get("parent_session_id")
    ]
    if not children:
        return []

    statuses: list[str] = []
    for session in sessions:
        for message in session.get("messages") or []:
            if message.get("role") != "tool" or message.get("tool_name") != "delegate_task":
                continue
            try:
                payload = json.loads(str(message.get("content") or ""))
            except json.JSONDecodeError:
                continue
            statuses.extend(
                str(result.get("status") or "")
                for result in payload.get("results") or []
                if isinstance(result, dict)
            )
    # A child is considered returned successfully only when the parent
    # delegate call reported completed. All v49 child results satisfy this.
    if statuses and any(status != "completed" for status in statuses):
        return []
    return children


def _load_conversations() -> dict[str, list[tuple[float, dict[str, Any]]]]:
    index: dict[str, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    paths = sorted(
        path
        for path in CONVERSATION_ROOT.glob("*_solver_*.json")
        if "_solver_verifier_" not in path.name
    )
    for path in paths:
        try:
            conversation = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        parsed = conversation.get("parsed_response")
        if not isinstance(parsed, dict):
            continue
        conversation["_export_path"] = path.relative_to(PROJECT_ROOT).as_posix()
        index[_signature(parsed)].append((path.stat().st_mtime, conversation))
    return index


def _load_artifacts() -> list[dict[str, Any]]:
    records = []
    for path in sorted(RUN_ROOT.glob("seed_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        summary = record.get("solver_summary") or {}
        difficulty = {
            "accepted:hard": "hard",
            "review:all_wrong": "all_wrong",
            "rejected:too_easy": "easy",
        }.get(summary.get("status"))
        if difficulty and record.get("uniqueness_key") == "unique":
            records.append(
                {
                    "path": path.relative_to(PROJECT_ROOT).as_posix(),
                    "mtime": path.stat().st_mtime,
                    "record": record,
                    "difficulty": difficulty,
                }
            )
    return records


def _match_report(
    artifact: dict[str, Any],
    report: dict[str, Any],
    conversation_index: dict[str, list[tuple[float, dict[str, Any]]]],
) -> dict[str, Any] | None:
    candidates = conversation_index.get(_signature(report), [])
    if not candidates:
        return None
    eligible = [item for item in candidates if item[0] <= artifact["mtime"] + 30]
    pool = eligible or candidates
    return min(pool, key=lambda item: abs(item[0] - artifact["mtime"]))[1]


def _row_base(artifact: dict[str, Any], report: dict[str, Any], conversation: dict[str, Any], counts: dict[str, Any]) -> dict[str, Any]:
    record = artifact["record"]
    target = record.get("target") or {}
    trace_sessions = conversation.get("trace_sessions") or []
    main_session = next(
        (
            session
            for session in trace_sessions
            if not session.get("parent_session_id")
            and str(session.get("source") or "").lower()
            not in {"subagent", "delegate", "delegation"}
        ),
        trace_sessions[0] if trace_sessions else {},
    )
    main_openai = _session_openai_record(
        main_session, conversation, is_main=True
    )
    child_openai = [
        _session_openai_record(session, conversation, is_main=False)
        for session in trace_sessions
        if session is not main_session
    ]
    all_names = set(main_openai.get("available_tool_names", []))
    for child in child_openai:
        all_names.update(child.get("available_tool_names", []))
    all_tools, missing_tools = _openai_tools(all_names)
    metadata = {
        "format": "openai_chat_completions_v1",
        "trajectory_type": "solver",
        "model": conversation.get("agent", {}).get("model"),
        "provider": conversation.get("agent", {}).get("provider"),
        "api_mode": conversation.get("agent", {}).get("api_mode"),
        "enabled_toolsets": conversation.get("agent", {}).get("enabled_toolsets"),
        "max_iterations": conversation.get("agent", {}).get("max_iterations"),
        "source_artifact": artifact["path"],
        "difficulty": artifact["difficulty"],
        "uniqueness_key": record.get("uniqueness_key"),
        "entity_id": target.get("entity_id"),
        "entity_type": target.get("entity_type"),
        "answer_field": target.get("answer_field"),
        "reference_answer_for_evaluation": target.get("answer"),
        "question": record.get("question"),
        "rollout_id": report.get("rollout_id"),
        "verifier_is_correct": report.get("verifier_is_correct"),
        "solver_report": report,
        "api_counts": counts,
        "conversation": conversation.get("_export_path"),
        "session_count": len(trace_sessions),
        "subagent_session_count": len(child_openai),
        "root_session_id": main_openai.get("session_id"),
        "effective_system_prompt": main_openai.get("effective_system_prompt"),
        "effective_system_prompt_sha256": main_openai.get(
            "effective_system_prompt_sha256"
        ),
        "system_prompt_source": main_openai.get("system_prompt_source"),
        "root_tool_names_used": main_openai.get("tool_names_used", []),
        "all_session_tool_names": sorted(all_names),
        "all_session_tools": all_tools,
        "trace_capture": conversation.get("trace_capture"),
        "delegation": conversation.get("delegation"),
        "missing_tool_schemas": missing_tools,
        "tool_schema_source": "Hermes source literals plus previous v49 static schema export; dynamic runtime schema may differ",
        "reasoning_field": "reasoning_content (DeepSeek provider extension; original reasoning is preserved)",
        "response_format_sent_to_model": None,
        "session_trajectories": child_openai,
        "training_notes": [
            "The top-level messages are the root Hermes session in OpenAI Chat Completions shape.",
            "Sub-agent sessions are separate conversations and are listed under metadata.session_trajectories; they must not be concatenated into the root messages array.",
            "Tool results retain Hermes untrusted-tool wrappers exactly as returned to the model.",
            "The raw HTTP request id, headers, retry/fallback decisions, and dynamically rebuilt tool descriptions are not fully persisted in v49; session ids, counts, timestamps, finish reasons, and tool arguments/results are retained.",
            "reference_answer_for_evaluation and solver_report are labels for QC, not model inputs; remove metadata before pure SFT ingestion if labels must be hidden.",
        ],
    }
    return {
        "messages": main_openai["messages"],
        "tools": main_openai["tools"],
        "metadata": metadata,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _call_stats(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = sorted(
        int((row.get("api_counts") or row.get("metadata", {}).get("api_counts") or {}).get(field))
        for row in rows
        if isinstance(
            (row.get("api_counts") or row.get("metadata", {}).get("api_counts") or {}).get(field),
            int,
        )
    )
    if not values:
        return {"n": 0}
    p90_index = min(len(values) - 1, math.ceil(len(values) * 0.9) - 1)
    return {
        "n": len(values),
        "min": values[0],
        "median": statistics.median(values),
        "mean": round(statistics.mean(values), 1),
        "p90": values[p90_index],
        "max": values[-1],
    }


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    conversation_index = _load_conversations()
    artifacts = _load_artifacts()
    solver_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    subagent_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unmapped = []

    for artifact in artifacts:
        summary = artifact["record"].get("solver_summary") or {}
        for report in artifact["record"].get("solver_reports") or []:
            if report.get("verifier_is_correct") is not True:
                continue
            conversation = _match_report(artifact, report, conversation_index)
            if conversation is None:
                unmapped.append({"artifact": artifact["path"], "rollout_id": report.get("rollout_id")})
                continue
            if conversation.get("agent", {}).get("model") != "tencent/deepseek-v4-pro":
                continue
            counts = _api_counts(conversation)
            calls = counts.get("all_session_api_calls")
            if not isinstance(calls, int):
                continue
            if artifact["difficulty"] == "hard":
                category = "hard"
            elif artifact["difficulty"] == "all_wrong":
                # No correct report can exist for all_wrong; kept for explicit empty files.
                continue
            elif calls > THRESHOLD:
                category = "easy_over40"
            else:
                category = "easy_under40"
            row = _row_base(artifact, report, conversation, counts)
            solver_rows[category].append(row)
            for session in _successful_subagent_sessions(conversation):
                sub_counts = {
                    "api_calls": session.get("api_call_count"),
                    "tool_calls": session.get("tool_call_count"),
                    "session_id": session.get("session_id") or session.get("id"),
                }
                child_openai = _session_openai_record(
                    session, conversation, is_main=False
                )
                child_names = _tool_names([session])
                child_tools = child_openai["tools"]
                child_missing = child_openai["missing_tool_schemas"]
                subagent_rows[category].append(
                    {
                        "messages": child_openai["messages"],
                        "tools": child_tools,
                        "metadata": {
                            "format": "openai_chat_completions_v1",
                            "trajectory_type": "subagent",
                            "model": conversation.get("agent", {}).get("model"),
                            "provider": conversation.get("agent", {}).get("provider"),
                            "api_mode": conversation.get("agent", {}).get("api_mode"),
                            "source_artifact": artifact["path"],
                            "difficulty": artifact["difficulty"],
                            "uniqueness_key": artifact["record"].get("uniqueness_key"),
                            "entity_id": (artifact["record"].get("target") or {}).get("entity_id"),
                            "answer_field": (artifact["record"].get("target") or {}).get("answer_field"),
                            "reference_answer_for_evaluation": (artifact["record"].get("target") or {}).get("answer"),
                            "rollout_id": report.get("rollout_id"),
                            "solver_conversation": conversation.get("_export_path"),
                            "subagent_session_id": session.get("session_id") or session.get("id"),
                            "parent_session_id": session.get("parent_session_id"),
                            "effective_system_prompt": child_openai.get("effective_system_prompt"),
                            "effective_system_prompt_sha256": child_openai.get(
                                "effective_system_prompt_sha256"
                            ),
                            "system_prompt_source": child_openai.get("system_prompt_source"),
                            "session_model": child_openai.get("model"),
                            "api_counts": sub_counts,
                            "tool_names_used": sorted(child_names),
                            "available_tool_names": child_openai.get("available_tool_names", []),
                            "missing_tool_schemas": child_missing,
                            "tool_schema_source": "Hermes source literals plus previous v49 static schema export; dynamic runtime schema may differ",
                            "reasoning_field": "reasoning_content (DeepSeek provider extension; original reasoning is preserved)",
                            "successful_delegate_return": True,
                            "training_notes": [
                                "This is a separate child conversation; keep its system and user messages intact.",
                                "The parent delegate_task call and result remain in the solver root trajectory, not this child trajectory.",
                                "The raw HTTP request id, headers, retry/fallback decisions, and dynamic tool-description rebuild are not fully persisted.",
                                "Remove metadata.reference_answer_for_evaluation-like labels before pure SFT ingestion; this row intentionally carries only QC identifiers.",
                            ],
                        },
                    }
                )

    # all_wrong has no correct solver rollout by definition, but the files are
    # still created so every requested category has the same two-file shape.
    for category in SOLVER_FILES:
        _write_jsonl(OUTPUT_ROOT / SOLVER_FILES[category], solver_rows.get(category, []))
        _write_jsonl(OUTPUT_ROOT / SUBAGENT_FILES[category], subagent_rows.get(category, []))

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_version": "v49",
        "uniqueness_filter": "unique only",
        "threshold_definition": ">40 uses all-session api_calls; <=40 is the complementary bucket",
        "correct_solver_rollouts": {category: len(solver_rows.get(category, [])) for category in SOLVER_FILES},
        "successful_subagent_sessions": {category: len(subagent_rows.get(category, [])) for category in SOLVER_FILES},
        "source_question_counts": {
            category: len(
                {
                    row.get("metadata", {}).get("source_artifact")
                    for row in solver_rows.get(category, [])
                }
            )
            for category in SOLVER_FILES
        },
        "solver_api_call_stats": {
            category: _call_stats(solver_rows.get(category, []), "all_session_api_calls")
            for category in SOLVER_FILES
        },
        "subagent_api_call_stats": {
            category: _call_stats(subagent_rows.get(category, []), "api_calls")
            for category in SUBAGENT_FILES
        },
        "all_wrong_note": "all_wrong has no correct solver rollout, so both requested correct-trajectory files are intentionally empty",
        "unique_question_counts": Counter(artifact["difficulty"] for artifact in artifacts),
        "unmapped_correct_rollouts": unmapped,
        "format": "openai_chat_completions_v1",
        "tool_schema_note": "Each row has an OpenAI-format tools array inferred from the session enabled toolsets, including tools not called in that rollout. Literal schemas come from the trace Hermes checkout; dynamic schemas are recovered from the previous v49 static export when available. Missing schemas are listed in metadata.missing_tool_schemas.",
        "system_prompt_note": "messages[0] is the actual runtime system message: the persisted Hermes prompt for the solver root, or the persisted Hermes prompt plus the reconstructed delegate child prompt for a sub-agent. BrowseComp research guidance is embedded in the solver root user content.",
        "reasoning_note": "Assistant messages preserve DeepSeek reasoning_content, which is required for provider-faithful replay but is a provider extension beyond the minimal OpenAI message schema.",
        "training_data_warning": "metadata contains evaluation labels and reference_answer_for_evaluation; strip metadata or at least those fields for pure supervised fine-tuning.",
        "solver_files": SOLVER_FILES,
        "subagent_files": SUBAGENT_FILES,
    }
    (OUTPUT_ROOT / "trajectory_export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=dict) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=dict))
    return 0 if not unmapped else 2


TOOL_SCHEMAS = _tool_schemas()


if __name__ == "__main__":
    raise SystemExit(main())
