from __future__ import annotations

import re
from typing import Iterable, Set


TOOLSETS: dict[str, dict[str, object]] = {
    "web": {"tools": ["web_search", "web_extract"], "includes": []},
    "search": {"tools": ["web_search"], "includes": []},
    "browser": {
        "tools": [
            "browser_navigate",
            "browser_snapshot",
            "browser_click",
            "browser_type",
            "browser_scroll",
            "browser_back",
            "browser_press",
            "browser_get_images",
            "browser_vision",
            "browser_console",
            "browser_cdp",
            "browser_dialog",
            "web_search",
        ],
        "includes": [],
    },
    "terminal": {"tools": ["terminal", "process"], "includes": []},
    "file": {"tools": ["read_file", "write_file", "patch", "search_files"], "includes": []},
    "code_execution": {"tools": ["execute_code"], "includes": []},
    "skills": {"tools": ["skills_list", "skill_view", "skill_manage"], "includes": []},
    "vision": {"tools": ["vision_analyze"], "includes": []},
    "image_gen": {"tools": ["image_generate"], "includes": []},
    "video": {"tools": ["video_analyze"], "includes": []},
    "video_gen": {"tools": ["video_generate"], "includes": []},
    "todo": {"tools": ["todo"], "includes": []},
    "memory": {"tools": ["memory"], "includes": []},
    "session_search": {"tools": ["session_search"], "includes": []},
    "clarify": {"tools": ["clarify"], "includes": []},
    "delegation": {"tools": ["delegate_task"], "includes": []},
    "browsecomp-plus-bm25": {
        "tools": ["search", "get_document"],
        "includes": [],
        "mcp_server": "browsecomp-plus-bm25",
    },
    "safe": {"tools": [], "includes": ["web", "vision", "image_gen"]},
    "debugging": {"tools": ["terminal", "process"], "includes": ["web", "file"]},
}

DEFAULT_TOOL_WHITELIST = "web,code_execution,terminal,vision,file,delegation"


def parse_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    stripped = value.strip()
    if not stripped or stripped.lower() in {"none", "null", "false"}:
        return []
    return [part.strip() for part in stripped.split(",") if part.strip()]


def resolve_toolsets(toolsets: Iterable[str]) -> Set[str]:
    allowed: set[str] = set()
    visited: set[str] = set()
    visiting: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in visiting:
            raise ValueError(f"cyclic toolset include detected at {name}")
        if ":" in name:
            # MCP entries use server:tool notation. Hermes validates them at runtime;
            # for post-run checks, accept both the full name and short tool name.
            allowed.add(name)
            allowed.add(name.split(":", 1)[1])
            visited.add(name)
            return
        if name not in TOOLSETS:
            raise ValueError(f"unknown toolset in whitelist: {name}")
        visiting.add(name)
        toolset = TOOLSETS[name]
        for included in toolset["includes"]:
            visit(included)
        visiting.remove(name)
        tools = toolset["tools"]
        allowed.update(tools)
        mcp_server = toolset.get("mcp_server")
        if mcp_server:
            # Hermes exposes MCP tools as mcp_<normalized server>_<tool>.
            # Keep the short names too because older session records may use them.
            server_prefix = re.sub(r"[^a-zA-Z0-9]+", "_", str(mcp_server)).strip("_")
            allowed.update(f"mcp_{server_prefix}_{tool}" for tool in tools)
        visited.add(name)

    for toolset in toolsets:
        visit(toolset)
    return allowed


def validate_tool_whitelist(toolsets: Iterable[str]) -> list[str]:
    normalized = [name.strip() for name in toolsets if name.strip()]
    resolve_toolsets(normalized)
    return normalized


def normalize_skill_whitelist(skills: Iterable[str]) -> list[str]:
    return [skill.strip() for skill in skills if skill.strip()]


def find_policy_violations(tool_calls: dict[str, int], allowed_tools: set[str]) -> list[str]:
    if not allowed_tools:
        return sorted(tool_calls)
    return sorted(name for name in tool_calls if name not in allowed_tools)
