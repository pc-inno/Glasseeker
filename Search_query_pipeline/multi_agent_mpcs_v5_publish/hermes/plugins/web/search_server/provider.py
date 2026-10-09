"""Web-search provider backed by a SearchTool-compatible HTTP service."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List
from urllib.parse import urlparse

import httpx

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)


def _env_value(name: str) -> str:
    try:
        from hermes_cli.config import get_env_value

        value = get_env_value(name)
    except Exception:
        value = None
    if value is None:
        import os

        value = os.getenv(name, "")
    return str(value or "").strip()


def _load_search_server_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
    except Exception:
        cfg = {}
    web_cfg = cfg.get("web", {}) or {}
    top_level = cfg.get("search_server", {}) or {}
    legacy_web = web_cfg.get("search_server", {}) or {}
    if any(str(top_level.get(key, "") or "").strip() for key in ("endpoint", "base_url", "url", "api_key")):
        return top_level
    if any(str(legacy_web.get(key, "") or "").strip() for key in ("endpoint", "base_url", "url", "api_key")):
        return legacy_web
    return legacy_web or top_level


def _cfg_value(root_cfg: Dict[str, Any], key: str, default: Any) -> Any:
    value = root_cfg.get(key)
    if value not in {None, ""}:
        return value
    return default


def _read_search_server_cfg() -> Dict[str, Any]:
    """Read search_server tool config from top-level config/env."""
    root_cfg = _load_search_server_config()
    endpoint = (
        _env_value("SEARCH_SERVER_ENDPOINT")
        or str(_cfg_value(root_cfg, "endpoint", "") or "").strip()
    )
    base_url = (
        _env_value("SEARCH_SERVER_BASE_URL")
        or str(_cfg_value(root_cfg, "base_url", "") or "").strip()
    )
    url = (
        _env_value("SEARCH_SERVER_URL")
        or str(_cfg_value(root_cfg, "url", "") or "").strip()
    )
    raw_port = _env_value("SEARCH_SERVER_PORT") or _cfg_value(root_cfg, "port", 80)
    try:
        port = int(raw_port or 80)
    except Exception:
        port = 80
    scheme = (
        _env_value("SEARCH_SERVER_SCHEME")
        or str(_cfg_value(root_cfg, "scheme", "") or "").strip().lower()
    )

    return {
        "endpoint": _build_endpoint(endpoint, base_url, url, port, scheme),
        "base_url": base_url,
        "url": url,
        "port": port,
        "scheme": scheme,
        "api_key": (
            _env_value("SEARCH_SERVER_API_KEY")
            or _env_value("SERPER_API_KEY")
            or str(_cfg_value(root_cfg, "api_key", "") or "").strip()
        ),
        "timeout": float(_cfg_value(root_cfg, "timeout", 30.0) or 30.0),
        "max_retries": max(1, int(_cfg_value(root_cfg, "max_retries", 3) or 3)),
        "retry_delay": float(_cfg_value(root_cfg, "retry_delay", 2.0) or 2.0),
    }


def _build_endpoint(
    endpoint: str, base_url: str, url: str, port: int, scheme: str = ""
) -> str:
    if endpoint:
        return endpoint.rstrip("/")
    if base_url:
        return _with_search_path(base_url)
    if not url:
        return ""

    parsed = urlparse(url)
    if parsed.scheme:
        # url already carries a scheme (http/https) - respect it verbatim so
        # "https://google.serper.dev/search" is never rewritten to http.
        return _with_search_path(url)

    # Bare hostname + port fallback. Decide the scheme in this order:
    #   1. explicit scheme argument (SEARCH_SERVER_SCHEME env / config)
    #   2. https for the canonical TLS ports (443, 8443)
    #   3. http otherwise (unchanged legacy behaviour for private endpoints)
    chosen = (scheme or "").strip().lower()
    if chosen not in {"http", "https"}:
        chosen = "https" if port in {443, 8443} else "http"
    if (chosen == "https" and port == 443) or (chosen == "http" and port == 80):
        return f"{chosen}://{url}/search"
    return f"{chosen}://{url}:{port}/search"


def _with_search_path(value: str) -> str:
    value = value.rstrip("/")
    parsed = urlparse(value)
    if parsed.path.rstrip("/").endswith("/search"):
        return value
    return f"{value}/search"


def _payload(query: str) -> Dict[str, str]:
    if any("\u4E00" <= char <= "\u9FFF" for char in query):
        return {"q": query, "location": "China", "gl": "cn", "hl": "zh-cn"}
    return {"q": query, "location": "United States", "gl": "us", "hl": "en"}


def _parse_search_text(text: str, limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None
    snippet_lines: List[str] = []
    in_snippet = False

    def flush() -> None:
        nonlocal current, snippet_lines, in_snippet
        if current is None:
            return
        current["description"] = "\n".join(snippet_lines).strip()
        if current.get("url") or current.get("title") or current.get("description"):
            current["position"] = len(rows) + 1
            rows.append(current)
        current = None
        snippet_lines = []
        in_snippet = False

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if line.startswith("[ID]:"):
            flush()
            current = {"title": "", "url": "", "description": ""}
            continue
        if current is None:
            continue
        if line.startswith("[Title]:"):
            current["title"] = line.split(":", 1)[1].strip()
            in_snippet = False
            continue
        if line.startswith("[URL]:"):
            current["url"] = line.split(":", 1)[1].strip()
            in_snippet = False
            continue
        if line.startswith("[Snippet]:"):
            in_snippet = True
            after = line.split(":", 1)[1].strip()
            if after:
                snippet_lines.append(after)
            continue
        if in_snippet:
            snippet_lines.append(line)

    flush()
    return rows[: max(1, int(limit))]


def _normalize_search_data(data: Any, limit: int) -> List[Dict[str, Any]]:
    safe_limit = max(1, int(limit))
    if isinstance(data, str):
        return _parse_search_text(data, safe_limit)

    if isinstance(data, dict):
        if isinstance(data.get("web"), list):
            return data["web"][:safe_limit]
        if isinstance(data.get("organic"), list):
            source = data["organic"]
        elif isinstance(data.get("results"), list):
            source = data["results"]
        elif isinstance(data.get("text"), list):
            source = data["text"]
        else:
            result = data.get("result")
            if isinstance(result, str):
                return _parse_search_text(result, safe_limit)
            source = []
    elif isinstance(data, list):
        source = data
    else:
        source = []

    rows = []
    for index, item in enumerate(source[:safe_limit]):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "title": str(item.get("title", "")),
                "url": str(item.get("url") or item.get("link") or ""),
                "description": str(item.get("description") or item.get("snippet") or ""),
                "position": int(item.get("position") or index + 1),
            }
        )
    return rows


def check_search_server_requirements() -> bool:
    cfg = _read_search_server_cfg()
    return bool(cfg["endpoint"] and cfg["api_key"])


def _search_server_response(query: str, limit: int = 5) -> Dict[str, Any]:
    query = str(query or "").strip()
    if not query:
        return {"success": False, "error": "query is required for web_search"}

    try:
        limit = max(1, min(100, int(limit or 5)))
    except Exception:
        limit = 5

    cfg = _read_search_server_cfg()
    if not cfg["endpoint"]:
        return {
            "success": False,
            "error": (
                "search_server backend endpoint is not configured; set "
                "search_server.endpoint, search_server.base_url, "
                "web.search_server.url, or SEARCH_SERVER_ENDPOINT"
            ),
        }
    if not cfg["api_key"]:
        return {
            "success": False,
            "error": (
                "Search API key is not configured; set search_server.api_key, "
                "web.search_server.api_key, or SEARCH_SERVER_API_KEY"
            ),
        }

    headers = {
        "Content-Type": "application/json",
        "X-API-KEY": cfg["api_key"],
    }

    last_error = ""
    for attempt in range(1, cfg["max_retries"] + 1):
        try:
            response = httpx.post(
                cfg["endpoint"],
                headers=headers,
                json=_payload(query),
                timeout=cfg["timeout"],
                trust_env=False,
            )
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and data.get("success") is False:
                return {
                    "success": False,
                    "error": str(data.get("error") or "search_server returned success=false"),
                }
            normalized = data.get("data", data) if isinstance(data, dict) else data
            web = _normalize_search_data(normalized, limit)
            return {"success": True, "data": {"web": web}}
        except httpx.TimeoutException as exc:
            last_error = f"timeout: {exc}"
            logger.warning("search_server timeout on attempt %d: %s", attempt, exc)
        except httpx.HTTPStatusError as exc:
            last_error = f"HTTP {exc.response.status_code}: {exc.response.text}"
            logger.warning("search_server HTTP error on attempt %d: %s", attempt, last_error)
        except httpx.RequestError as exc:
            last_error = f"request error: {exc}"
            logger.warning("search_server request error on attempt %d: %s", attempt, exc)
        except json.JSONDecodeError as exc:
            last_error = f"bad JSON: {exc}"
            logger.warning("search_server JSON error on attempt %d: %s", attempt, exc)

        if attempt < cfg["max_retries"]:
            time.sleep(cfg["retry_delay"])

    return {
        "success": False,
        "error": f"search_server failed after {cfg['max_retries']} attempts: {last_error}",
    }


class SearchServerProvider(WebSearchProvider):
    """Route ``web_search`` through the configured internal search server."""

    @property
    def name(self) -> str:
        return "search_server"

    @property
    def display_name(self) -> str:
        return "SearchTool HTTP Server"

    def is_available(self) -> bool:
        return check_search_server_requirements()

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        return _search_server_response(query, limit)

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "SearchTool HTTP Server",
            "badge": "internal · search only",
            "tag": (
                "SearchTool-compatible POST /search backend for web_search. "
                "Configure it under search_server in config.yaml."
            ),
            "env_vars": [
                {
                    "key": "SEARCH_SERVER_ENDPOINT",
                    "prompt": "Search server endpoint (including /search)",
                    "url": None,
                },
                {
                    "key": "SEARCH_SERVER_API_KEY",
                    "prompt": "Search server API key",
                    "url": None,
                },
            ],
        }


# Backward-compatible Python helper.  It is intentionally not registered as a
# model tool; agent-visible calls go through web_search -> SearchServerProvider.
def search_server_tool(query: str, limit: int = 5) -> str:
    result = _search_server_response(query, limit)
    if result.get("success"):
        result = {**result, "tool": "search_server", "query": str(query or "").strip()}
    return json.dumps(result, ensure_ascii=False)


def handle_search_server(args, **kw):
    return search_server_tool(args.get("query", ""), args.get("limit", 5))
