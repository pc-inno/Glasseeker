#!/usr/bin/env python3
"""Audit v49 JSONL trajectories and export rows without repeat candidates."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "runs_v49" / "result"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "runs_v49_search_repeat_audit"
def load_repeat_audit_module(script_path: Path):
    script_path = script_path.expanduser().resolve()
    if not script_path.is_file():
        raise FileNotFoundError(f"repeat audit module not found: {script_path}")
    spec = importlib.util.spec_from_file_location("repeat_audit_v3", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repeat audit logic: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_write_jsonl(path: Path, rows: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row)
            if not row.endswith("\n"):
                handle.write("\n")
    os.replace(temporary, path)


def percentile(values: list[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 2)


def row_role(source_name: str) -> str:
    return "subagent" if "_subagent_traces" in source_name else "main"


def is_below(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def audit_row(
    repeat_audit: Any,
    row: dict[str, Any],
    source_name: str,
    line_number: int,
    agent_role: str,
) -> dict[str, Any]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("messages is not a list")
    metadata = row.get("metadata") or {}
    session_id = str(metadata.get("session_id") or "")
    agent_id = f"{source_name}:{line_number}"
    report, _ = repeat_audit.audit_agent_messages(
        messages=messages,
        agent_id=agent_id,
        agent_role=agent_role,
        session_id=session_id,
        parent_session_id=None,
        trace_depth=0,
        source_file=f"{source_name}#line={line_number}",
        prompt_capture_complete=True,
        uid_start=0,
    )
    groups = report["duplicate_groups"]
    return {
        "source_file": source_name,
        "source_line": line_number,
        "agent_role": agent_role,
        "question_id": metadata.get("question_id"),
        "run_id": metadata.get("run_id"),
        "trajectory_type": metadata.get("trajectory_type"),
        "search_count": report["search_count"],
        "message_count": len(messages),
        "has_repeated_search": report["has_repeated_search"],
        "has_redundant_search": report["has_redundant_search"],
        "duplicate_group_count": len(groups),
        "duplicate_kinds": dict(Counter(group["kind"] for group in groups)),
        "assessments": dict(Counter(group["assessment"] for group in groups)),
        "duplicate_groups": groups,
    }


def empty_file_stats(source_name: str) -> dict[str, Any]:
    return {
        "source_file": source_name,
        "agent_role": row_role(source_name),
        "input_rows": 0,
        "audited_rows": 0,
        "invalid_rows": 0,
        "message_count": 0,
        "search_count": 0,
        "with_any_repeated_search": 0,
        "without_any_repeated_search": 0,
        "with_redundant_search": 0,
        "exported_non_repeated_rows": 0,
        "duplicate_groups": {},
        "assessments": {},
        "errors": [],
        "_search_counts": [],
    }


def process_file(
    repeat_audit: Any,
    source_path: Path,
    audit_path: Path,
    export_path: Path,
) -> dict[str, Any]:
    source_name = source_path.name
    stats = empty_file_stats(source_name)
    audit_lines: list[str] = []
    export_lines: list[str] = []
    role = row_role(source_name)
    with source_path.open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            stats["input_rows"] += 1
            try:
                row = json.loads(raw_line)
                if not isinstance(row, dict):
                    raise ValueError("JSONL row is not an object")
                report = audit_row(repeat_audit, row, source_name, line_number, role)
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                stats["invalid_rows"] += 1
                stats["errors"].append({"line": line_number, "error": str(exc)})
                continue

            stats["audited_rows"] += 1
            stats["message_count"] += report["message_count"]
            stats["search_count"] += report["search_count"]
            stats["_search_counts"].append(report["search_count"])
            if report["has_repeated_search"]:
                stats["with_any_repeated_search"] += 1
            else:
                stats["without_any_repeated_search"] += 1
                stats["exported_non_repeated_rows"] += 1
                export_lines.append(raw_line)
            stats["with_redundant_search"] += report["has_redundant_search"]
            for kind, count in report["duplicate_kinds"].items():
                stats["duplicate_groups"][kind] = stats["duplicate_groups"].get(kind, 0) + count
            for assessment, count in report["assessments"].items():
                stats["assessments"][assessment] = stats["assessments"].get(assessment, 0) + count
            audit_lines.append(json.dumps(report, ensure_ascii=False))

    atomic_write_jsonl(audit_path, audit_lines)
    atomic_write_jsonl(export_path, export_lines)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repeat-audit-script", type=Path, required=True,
                        help="Optional external audit implementation; not required by the pipeline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"input directory does not exist: {input_dir}")
    if output_dir == input_dir or is_below(output_dir, input_dir):
        raise SystemExit("refusing to write output inside the source result directory")

    repeat_audit = load_repeat_audit_module(args.repeat_audit_script)
    audit_dir = output_dir / "audit_results"
    export_dir = output_dir / "non_repeated"
    file_stats: dict[str, dict[str, Any]] = {}
    for source_path in sorted(input_dir.glob("*.jsonl")):
        file_stats[source_path.name] = process_file(
            repeat_audit,
            source_path,
            audit_dir / source_path.name,
            export_dir / source_path.name,
        )

    totals = Counter()
    search_counts: list[int] = []
    for stats in file_stats.values():
        for key in (
            "input_rows", "audited_rows", "invalid_rows", "message_count", "search_count",
            "with_any_repeated_search", "without_any_repeated_search",
            "with_redundant_search", "exported_non_repeated_rows",
        ):
            totals[key] += stats[key]
        search_counts.extend(stats.get("_search_counts", []))

    summary = {
        "schema_version": "v49_repeat_audit_export_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "audit_method": getattr(repeat_audit, "METHOD_VERSION", "unknown"),
        "scope": {
            "row_is_independent_sample": True,
            "within_agent_only": True,
            "cross_agent_overlap": "not applicable because solver/subagent files are separate",
            "success_filter": "source JSONL already contains v49 correct solver/successful subagent exports",
        },
        "counts": dict(totals),
        "search_count_distribution": {
            "min": min(search_counts) if search_counts else None,
            "median": percentile(search_counts, 0.5),
            "mean": round(sum(search_counts) / len(search_counts), 2) if search_counts else None,
            "p90": percentile(search_counts, 0.9),
            "max": max(search_counts) if search_counts else None,
        },
        "files": {
            name: {key: value for key, value in stats.items() if key != "_search_counts"}
            for name, stats in file_stats.items()
        },
        "outputs": {
            "audit_results": "audit_results/*.jsonl",
            "non_repeated_original_rows": "non_repeated/*.jsonl",
        },
        "training_data_note": (
            "non_repeated JSONL rows are preserved verbatim. They retain metadata, including "
            "evaluation labels and reference_answer_for_evaluation; remove metadata before pure SFT."
        ),
    }
    atomic_write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
