"""Reader HTTP backend for Hermes ``web_extract``."""

from __future__ import annotations

from .provider import FetchServerProvider


def register(ctx) -> None:
    ctx.register_web_search_provider(FetchServerProvider())
