"""Search Server backend compatible with both Hermes provider APIs."""

from __future__ import annotations

import json
import logging

from tools.web_tools import WEB_SEARCH_SCHEMA

from .provider import SearchServerProvider, check_search_server_requirements

logger = logging.getLogger(__name__)
_SEARCH = SearchServerProvider()


def _search_check() -> bool:
    try:
        return bool(check_search_server_requirements())
    except Exception:
        return False


def handle_search_server(args: dict, **_) -> str:
    from tools.interrupt import is_interrupted

    args = args if isinstance(args, dict) else {}
    query = str(args.get("query", "") or "")
    try:
        limit = max(1, int(args.get("limit", 5)))
    except (TypeError, ValueError):
        limit = 5
    if is_interrupted():
        return json.dumps({"success": False, "error": "Interrupted"})
    try:
        return json.dumps(_SEARCH.search(query, limit=limit), ensure_ascii=False)
    except Exception:
        logger.error("Search Server web_search handler failed")
        return json.dumps({"success": False, "error": "web_search failed"})


_handle_search = handle_search_server


def register(ctx) -> None:
    ctx.register_web_search_provider(_SEARCH)
    ctx.register_tool(
        name="web_search",
        toolset="web",
        schema=WEB_SEARCH_SCHEMA,
        handler=handle_search_server,
        check_fn=_search_check,
        requires_env=[],
        emoji="🔍",
        override=True,
    )
