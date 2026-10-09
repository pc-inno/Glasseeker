"""SearchTool HTTP backend for Hermes ``web_search``."""

from __future__ import annotations

from .provider import SearchServerProvider


def register(ctx) -> None:
    ctx.register_web_search_provider(SearchServerProvider())
