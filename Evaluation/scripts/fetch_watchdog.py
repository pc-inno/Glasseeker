#!/usr/bin/env python3
"""Shell-watchdog helper for BrowseComp fetch-server failures."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        import yaml

        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def resolve_fetch_url() -> str:
    for name in ("FETCH_SERVER_BASE_URL", "FETCH_SERVER_URL"):
        value = os.environ.get(name, "").strip()
        if value:
            return value.rstrip("/")

    hermes_home = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))).expanduser()
    config = _read_yaml(hermes_home / "config.yaml")
    web = config.get("web") or {}
    fetch = config.get("fetch_server") or web.get("fetch_server") or {}
    if not isinstance(fetch, dict):
        return ""
    return str(fetch.get("base_url") or fetch.get("url") or "").strip().rstrip("/")


def _json_request(request: urllib.request.Request, timeout: float) -> dict[str, Any]:
    # Match the Hermes provider: backend traffic must not inherit HTTP proxies.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}")
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("response is not a JSON object")
    return payload


def _probe_target(base_url: str, timeout: float, target_url: str) -> None:
    body = json.dumps({"url": target_url, "extractMode": "text"}).encode("utf-8")
    fetch = _json_request(
        urllib.request.Request(
            f"{base_url.rstrip('/')}/fetch",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        ),
        timeout,
    )
    if not fetch.get("success") or not str(fetch.get("data") or "").strip():
        raise RuntimeError(str(fetch.get("error") or "empty content"))


def probe(base_url: str, timeout: float, target_urls: list[str]) -> None:
    health = _json_request(
        urllib.request.Request(f"{base_url.rstrip('/')}/health", method="GET"),
        timeout,
    )
    if str(health.get("status", "")).lower() != "ok":
        raise RuntimeError(f"unhealthy response: {health!r}")

    if not target_urls:
        return

    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(target_urls)) as executor:
        future_to_url = {
            executor.submit(_probe_target, base_url, timeout, target_url): target_url
            for target_url in target_urls
        }
        for future in concurrent.futures.as_completed(future_to_url):
            target_url = future_to_url[future]
            try:
                future.result()
            except Exception as exc:
                failures.append(f"{target_url}: {exc}")
    if failures:
        raise RuntimeError("functional fetch failed: " + "; ".join(sorted(failures)))


def _target_urls(values: list[str]) -> list[str]:
    return list(
        dict.fromkeys(
            target.strip()
            for value in values
            for target in value.split(",")
            if target.strip()
        )
    )


def _process_tree(root_pid: int) -> list[int]:
    pending = [root_pid]
    seen: set[int] = set()
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            task_dirs = list(Path(f"/proc/{pid}/task").iterdir())
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        # A subprocess created by ThreadPoolExecutor belongs to the worker
        # thread's children list, not necessarily the process leader's list.
        for task_dir in task_dirs:
            try:
                raw_children = (task_dir / "children").read_text(encoding="utf-8")
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            pending.extend(int(value) for value in raw_children.split() if value.isdigit())
    return sorted(seen)


def active_run_ids(process_group: int) -> list[str]:
    run_ids: set[str] = set()
    # setsid makes the evaluator both the process-group leader and tree root.
    # Walking /proc/.../children avoids scanning every process on a busy host.
    for pid in _process_tree(process_group):
        proc_dir = Path(f"/proc/{pid}")
        try:
            entries = (proc_dir / "environ").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        for entry in entries:
            if entry.startswith(b"BROWSE_COMP_RUN_ID="):
                run_id = entry.partition(b"=")[2].decode("utf-8", errors="replace").strip()
                if run_id:
                    run_ids.add(run_id)
    return sorted(run_ids)


def _safe_id(value: str) -> str:
    import re

    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("._-")
    return cleaned or "unknown"


def _dataset_metadata(path: Path, project_root: Path) -> dict[str, dict[str, Any]]:
    # Import the project loader so JSON, JSONL, Miro GAIA, and safe IDs stay aligned.
    sys.path.insert(0, str(project_root))
    from browse_comp_eval.dataset import load_dataset

    return {
        item.question_id: {
            "question": item.question,
            "answer": item.answer,
            "type": item.type,
            "source_line": item.source_line,
        }
        for item in load_dataset(path)
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".watchdog.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def mark_failed(args: argparse.Namespace) -> int:
    detected_at = datetime.now(timezone.utc).astimezone().isoformat()
    run_ids = sorted(
        {
            line.strip()
            for line in Path(args.run_ids_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    )
    metadata = _dataset_metadata(Path(args.data_path), Path(args.project_root))
    save_name = args.save_name or _safe_id(args.model)
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = Path(args.project_root) / output_root
    output_base = output_root / save_name / args.dataset
    conv_dir = output_base / "conv"
    summary_path = output_base / "summary.jsonl"

    marked: list[dict[str, str]] = []
    for run_id in run_ids:
        question_id, separator, raw_repeat = run_id.rpartition("__r")
        if not separator or not raw_repeat.isdigit() or question_id not in metadata:
            print(f"watchdog warning: cannot map active run ID {run_id!r}", file=sys.stderr)
            continue
        repeat_index = int(raw_repeat)
        item = metadata[question_id]
        if args.numbered_output:
            output_id = f"q{item['source_line']:06d}"
            result_path = conv_dir / output_id / f"{output_id}__r{repeat_index}.json"
        else:
            result_path = conv_dir / question_id / f"{run_id}.json"

        existing: dict[str, Any] = {}
        try:
            loaded = json.loads(result_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass
        previous_status = existing.get("status")
        failure = {
            "component": "fetch_server",
            "detected_at": detected_at,
            "fetch_server_url": args.fetch_server_url,
            "reason": args.reason,
        }
        result = {
            **existing,
            "question_id": question_id,
            "repeat_index": repeat_index,
            "run_id": run_id,
            "question": item["question"],
            "answer": item["answer"],
            "type": item["type"],
            "status": "failed",
            "completed": False,
            "error": "fetch server watchdog aborted this active evaluation run",
            "failure_type": "infrastructure",
            "infrastructure_failure": failure,
            "watchdog_previous_status": previous_status,
        }
        _atomic_write_json(result_path, result)
        summary = {
            "run_id": run_id,
            "question_id": question_id,
            "repeat_index": repeat_index,
            "type": item["type"],
            "status": "failed",
            "error": result["error"],
            "failure_type": "infrastructure",
            "infrastructure_failure": failure,
            "result_path": str(result_path),
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(summary, ensure_ascii=False) + "\n")
        marked.append({"run_id": run_id, "result_path": str(result_path)})

    manifest = {
        "component": "fetch_server",
        "detected_at": detected_at,
        "fetch_server_url": args.fetch_server_url,
        "reason": args.reason,
        "active_run_ids": run_ids,
        "marked_failed": marked,
        "resume_hint": "rerun with --rerun-failed or --force",
    }
    manifest_path = output_base / f"fetch_watchdog_abort_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    _atomic_write_json(manifest_path, manifest)
    print(f"Fetch watchdog manifest: {manifest_path}")
    print(f"Fetch watchdog marked failed: {len(marked)} active run(s)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("resolve-url")

    probe_parser = subparsers.add_parser("probe")
    probe_parser.add_argument("--base-url", required=True)
    probe_parser.add_argument("--timeout", type=float, required=True)
    probe_parser.add_argument("--target-url", action="append", default=[])

    active_parser = subparsers.add_parser("active-run-ids")
    active_parser.add_argument("--process-group", type=int, required=True)

    mark_parser = subparsers.add_parser("mark-failed")
    mark_parser.add_argument("--run-ids-file", required=True)
    mark_parser.add_argument("--data-path", required=True)
    mark_parser.add_argument("--output-root", required=True)
    mark_parser.add_argument("--project-root", required=True)
    mark_parser.add_argument("--save-name", default="")
    mark_parser.add_argument("--model", required=True)
    mark_parser.add_argument("--dataset", required=True)
    mark_parser.add_argument("--fetch-server-url", required=True)
    mark_parser.add_argument("--reason", required=True)
    mark_parser.add_argument("--numbered-output", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "resolve-url":
        print(resolve_fetch_url())
        return 0
    if args.command == "probe":
        try:
            probe(args.base_url, args.timeout, _target_urls(args.target_url))
        except (OSError, ValueError, RuntimeError, urllib.error.URLError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        return 0
    if args.command == "active-run-ids":
        print("\n".join(active_run_ids(args.process_group)))
        return 0
    if args.command == "mark-failed":
        return mark_failed(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
