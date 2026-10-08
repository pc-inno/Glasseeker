"""Web-extract provider backed by a Reader HTTP Server."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List

import httpx

from agent.web_search_provider import WebSearchProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)


_DEFAULT_FETCH_CFG = {
    "base_url": "",
    "extract_mode": "text",
    "timeout": 180.0,
    "max_concurrent": 5,
    "max_retries": 3,
    "retry_delay": 1.0,
    "jina_api_key": "",
    "llm_base_url": "",
    "llm_model": "",
    "llm_api_key": "",
}


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


def _load_fetch_server_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
    except Exception:
        cfg = {}
    web_cfg = cfg.get("web", {}) or {}
    top_level = cfg.get("fetch_server", {}) or {}
    legacy_web = web_cfg.get("fetch_server", {}) or {}

    merged: Dict[str, Any] = {}
    for key, value in legacy_web.items():
        if value not in {None, ""}:
            merged[key] = value

    top_level_differs_from_default = any(
        top_level.get(key) not in (None, "", default)
        for key, default in _DEFAULT_FETCH_CFG.items()
    )
    if top_level_differs_from_default:
        for key, value in top_level.items():
            if value not in {None, ""}:
                merged[key] = value
    return merged or top_level


def _cfg_value(root_cfg: Dict[str, Any], key: str) -> Any:
    default = _DEFAULT_FETCH_CFG[key]
    value = root_cfg.get(key)
    if value not in {None, ""}:
        return value
    return default


def _read_fetch_server_cfg() -> Dict[str, Any]:
    """Read fetch_server tool config from top-level config/env."""
    root_cfg = _load_fetch_server_config()
    base_url = (
        _env_value("FETCH_SERVER_BASE_URL")
        or _env_value("FETCH_SERVER_URL")
        or str(_cfg_value(root_cfg, "base_url") or "").strip()
    )
    extract_mode = str(_cfg_value(root_cfg, "extract_mode") or "text").strip().lower()
    if extract_mode not in {"text", "markdown"}:
        extract_mode = "text"

    return {
        "base_url": base_url.rstrip("/"),
        "extract_mode": extract_mode,
        "timeout": float(_cfg_value(root_cfg, "timeout") or 180.0),
        "max_concurrent": max(1, int(_cfg_value(root_cfg, "max_concurrent") or 5)),
        "max_retries": max(1, int(_cfg_value(root_cfg, "max_retries") or 3)),
        "retry_delay": max(0.0, float(_cfg_value(root_cfg, "retry_delay") or 1.0)),
        "jina_api_key": (
            _env_value("FETCH_SERVER_JINA_API_KEY")
            or _env_value("JINA_API_KEY")
            or str(_cfg_value(root_cfg, "jina_api_key") or "").strip()
        ),
        "llm_base_url": (
            _env_value("FETCH_SERVER_LLM_BASE_URL")
            or str(_cfg_value(root_cfg, "llm_base_url") or "").strip()
        ),
        "llm_model": (
            _env_value("FETCH_SERVER_LLM_MODEL")
            or str(_cfg_value(root_cfg, "llm_model") or "").strip()
        ),
        "llm_api_key": (
            _env_value("FETCH_SERVER_LLM_API_KEY")
            or str(_cfg_value(root_cfg, "llm_api_key") or "").strip()
        ),
    }


def _payload_for_url(url: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "url": url,
        "extractMode": cfg["extract_mode"],
    }
    if cfg["jina_api_key"]:
        payload["jina_api_key"] = cfg["jina_api_key"]
    if cfg["llm_base_url"]:
        payload["llm_base_url"] = cfg["llm_base_url"]
    if cfg["llm_model"]:
        payload["llm_model"] = cfg["llm_model"]
    if cfg["llm_api_key"]:
        payload["llm_api_key"] = cfg["llm_api_key"]
    return payload


async def _fetch_one(
    client: httpx.AsyncClient,
    endpoint: str,
    url: str,
    cfg: Dict[str, Any],
    sem: asyncio.Semaphore,
) -> Dict[str, Any]:
    async with sem:
        max_retries = max(1, int(cfg.get("max_retries") or 1))
        retry_delay = max(0.0, float(cfg.get("retry_delay") or 0.0))
        data = None
        last_error = ""
        for attempt in range(1, max_retries + 1):
            try:
                response = await client.post(endpoint, json=_payload_for_url(url, cfg))
                response.raise_for_status()
                data = response.json()
                break
            except httpx.HTTPStatusError as exc:
                last_error = f"HTTP {exc.response.status_code}: {exc.response.text}"
                logger.warning("fetch_server HTTP error for %s on attempt %d: %s", url, attempt, last_error)
                break
            except (httpx.TimeoutException, httpx.RequestError, json.JSONDecodeError) as exc:
                last_error = str(exc)
                logger.warning("fetch_server request failed for %s on attempt %d: %s", url, attempt, exc)
            except Exception as exc:
                last_error = str(exc)
                logger.warning("fetch_server request failed for %s on attempt %d: %s", url, attempt, exc)

            if attempt < max_retries and retry_delay:
                await asyncio.sleep(retry_delay)

        if data is None:
            return {
                "url": url,
                "title": "",
                "content": "",
                "raw_content": "",
                "error": f"fetch_server request failed: {last_error}",
            }

        if not data.get("success"):
            return {
                "url": url,
                "title": "",
                "content": "",
                "raw_content": "",
                "error": str(data.get("error") or "fetch_server returned success=false"),
            }

        content = str(data.get("data") or "")
        if not content.strip():
            return {
                "url": url,
                "title": "",
                "content": "",
                "raw_content": "",
                "error": "fetch_server returned empty content",
            }

        return {
            "url": url,
            "title": "",
            "content": content,
            "raw_content": content,
            "metadata": {
                "source": "fetch_server",
                "extract_mode": cfg["extract_mode"],
            },
        }


def check_fetch_server_requirements() -> bool:
    return bool(_read_fetch_server_cfg()["base_url"])


def _normalize_urls(urls: List[str] | str) -> List[str]:
    if isinstance(urls, str):
        url_list = [urls]
    else:
        url_list = [str(url or "").strip() for url in list(urls or [])]
    return [url for url in url_list if url][:5]


async def _fetch_documents(
    urls: List[str] | str,
    extract_mode: str | None = None,
) -> List[Dict[str, Any]]:
    url_list = _normalize_urls(urls)
    if not url_list:
        return []

    cfg = _read_fetch_server_cfg()
    if extract_mode:
        mode = str(extract_mode).strip().lower()
        if mode in {"text", "markdown"}:
            cfg["extract_mode"] = mode
    if not cfg["base_url"]:
        return [
            {
                "url": url,
                "title": "",
                "content": "",
                "raw_content": "",
                "error": "fetch_server backend base_url is not configured",
            }
            for url in url_list
        ]

    endpoint = f"{cfg['base_url']}/fetch"
    sem = asyncio.Semaphore(cfg["max_concurrent"])
    async with httpx.AsyncClient(timeout=cfg["timeout"], trust_env=False) as client:
        return await asyncio.gather(
            *(_fetch_one(client, endpoint, url, cfg, sem) for url in url_list)
        )


class FetchServerProvider(WebSearchProvider):
    """Route ``web_extract`` through the configured Reader HTTP Server."""

    @property
    def name(self) -> str:
        return "fetch_server"

    @property
    def display_name(self) -> str:
        return "Reader HTTP Server"

    def is_available(self) -> bool:
        return check_fetch_server_requirements()

    def supports_search(self) -> bool:
        return False

    def supports_extract(self) -> bool:
        return True

    async def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        # ``web_extract_tool`` currently passes format="markdown" to every
        # provider.  Reader Server has its own extract_mode setting (and its
        # markdown path may invoke a server-side LLM), so preserve the explicit
        # backend config instead of silently overriding it here.
        return await _fetch_documents(urls)

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Reader HTTP Server",
            "badge": "internal · extract only",
            "tag": (
                "POST /fetch backend for web_extract. Configure its base URL "
                "under fetch_server in config.yaml."
            ),
            "env_vars": [
                {
                    "key": "FETCH_SERVER_BASE_URL",
                    "prompt": "Reader HTTP Server base URL",
                    "url": None,
                },
            ],
        }


# Backward-compatible Python helper.  It is intentionally not registered as a
# model tool; agent-visible calls go through web_extract -> FetchServerProvider.
async def fetch_server_tool(urls: List[str] | str, extract_mode: str | None = None) -> str:
    url_list = _normalize_urls(urls)
    if not url_list:
        return tool_error("urls is required for fetch_server")
    documents = await _fetch_documents(url_list, extract_mode=extract_mode)

    return json.dumps(
        {
            "success": True,
            "tool": "fetch_server",
            "documents": documents,
        },
        ensure_ascii=False,
    )


def handle_fetch_server(args, **kw):
    return fetch_server_tool(args.get("urls", []), args.get("extract_mode"))
