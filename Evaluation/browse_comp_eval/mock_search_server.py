"""A deterministic, loopback-only SearchTool-compatible HTTP server.

The sandbox worker uses this server for offline evaluator smoke tests.  It is
deliberately implemented with the standard library so the worker bundle does
not gain a dependency on the host's HTTP stack.  The response is a pure
function of the submitted query; no dataset or reference-answer data is
loaded by this module.
"""

from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


MOCK_SEARCH_API_KEY = "browse-comp-mock-search-key"


def _results_for_query(query: str, limit: int) -> list[dict[str, Any]]:
    """Return stable SearchTool rows derived only from ``query``."""

    query = str(query).strip()
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
    safe_limit = max(1, min(100, int(limit)))
    return [
        {
            "title": f"Mock search result {position} for {query}",
            "url": f"https://mock.search.invalid/{digest[:16]}/{position}",
            "description": f"Deterministic mock result {position} for query: {query}",
            "position": position,
        }
        for position in range(1, safe_limit + 1)
    ]


class _MockSearchHandler(BaseHTTPRequestHandler):
    """Handle only the minimal POST /search contract used by SearchServerProvider."""

    server: "_MockSearchHTTPServer"

    # The benchmark runner should never emit one line per model search.
    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        if self.path != "/search":
            self._send_json(404, {"success": False, "error": "not found"})
            return

        if self.headers.get("X-API-KEY", "") != self.server.api_key:
            self._send_json(401, {"success": False, "error": "invalid API key"})
            return

        raw_length = self.headers.get("Content-Length", "")
        try:
            length = int(raw_length)
        except (TypeError, ValueError):
            self._send_json(400, {"success": False, "error": "invalid content length"})
            return
        if length < 0 or length > 1_048_576:
            self._send_json(400, {"success": False, "error": "invalid request size"})
            return

        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"success": False, "error": "invalid JSON"})
            return
        if not isinstance(payload, dict):
            self._send_json(400, {"success": False, "error": "request must be an object"})
            return

        query = payload.get("q", payload.get("query", ""))
        if not isinstance(query, str) or not query.strip():
            self._send_json(400, {"success": False, "error": "query is required"})
            return
        raw_limit = payload.get("limit", 5)
        try:
            limit = int(raw_limit)
        except (TypeError, ValueError):
            limit = 5
        self._send_json(
            200,
            {"success": True, "data": {"web": _results_for_query(query, limit)}},
        )


class _MockSearchHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, api_key: str):
        super().__init__(("127.0.0.1", 0), _MockSearchHandler)
        self.api_key = api_key


class MockSearchServer:
    """Context-managed deterministic SearchTool server on an ephemeral port."""

    def __init__(self, api_key: str = MOCK_SEARCH_API_KEY) -> None:
        if not isinstance(api_key, str) or not api_key:
            raise ValueError("api_key must be a non-empty string")
        self.api_key = api_key
        self._server = _MockSearchHTTPServer(api_key)
        self._thread: threading.Thread | None = None
        self._closed = False

    @property
    def endpoint(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/search"

    @property
    def url(self) -> str:
        """Alias used by callers that name HTTP endpoints ``url``."""

        return self.endpoint

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._server.server_address[:2]
        return host, port

    def start(self) -> "MockSearchServer":
        if self._closed:
            raise RuntimeError("mock search server is closed")
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._server.serve_forever,
                name="browse-comp-mock-search",
                daemon=True,
            )
            self._thread.start()
        return self

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._thread is not None:
            self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> "MockSearchServer":
        return self.start()

    def __exit__(self, _exc_type: Any, _exc_value: Any, _traceback: Any) -> None:
        self.close()
