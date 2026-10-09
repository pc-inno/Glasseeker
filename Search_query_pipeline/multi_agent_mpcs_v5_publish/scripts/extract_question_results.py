#!/usr/bin/env python3
"""Extract questions and solver outcomes from runs_v24 through runs_v30.

The script resolves the project root from its own location, so it can be
invoked from any working directory after the project is moved. Input and
output paths written to JSONL are always relative to that project root.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VERSIONS = tuple(range(24, 31))
HARD_STATUS = "accepted:hard"
OUTPUT_FILENAMES = {
    "accepted_hard_unique": "accepted_hard_unique.jsonl",
    "accepted_hard_uncertain": "accepted_hard_uncertain.jsonl",
    "accepted_hard_other_uniqueness": "accepted_hard_other_uniqueness.jsonl",
    "review_all_wrong_unique": "review_all_wrong_unique.jsonl",
    "review_all_wrong_uncertain": "review_all_wrong_uncertain.jsonl",
    "review_all_wrong_other_uniqueness": "review_all_wrong_other_uniqueness.jsonl",
    "rejected_too_easy": "rejected_too_easy.jsonl",
}
KNOWN_SOLVER_STATUSES = {
    "accepted:hard",
    "rejected:too_easy",
    "rejected:too_few_solver_rollouts",
    "review:all_wrong",
}


def _relative_path(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def _seed_files(versions: Iterable[int]) -> Iterable[tuple[str, Path]]:
    for version in versions:
        run_dir = PROJECT_ROOT / "data" / f"runs_v{version}"
        if not run_dir.is_dir():
            continue
        for path in sorted(run_dir.glob("seed_*.json")):
            yield f"v{version}", path


def _solver_status(record: dict[str, Any]) -> str:
    """Return a solver outcome, without treating generic statuses as ratings."""
    summary = record.get("solver_summary")
    if isinstance(summary, dict):
        status = summary.get("status")
        if isinstance(status, str) and status.strip():
            return status.strip()

    status = record.get("status")
    if isinstance(status, str) and status in KNOWN_SOLVER_STATUSES:
        return status
    return ""


def _has_question(record: dict[str, Any]) -> bool:
    question = record.get("question")
    return isinstance(question, str) and bool(question.strip())


def _group_key(solver_status: str, uniqueness_key: str) -> str:
    if solver_status == "rejected:too_easy":
        return "rejected_too_easy"
    solver_prefix = {
        "accepted:hard": "accepted_hard",
        "review:all_wrong": "review_all_wrong",
    }.get(solver_status)
    if not solver_prefix:
        return ""
    if uniqueness_key in {"unique", "uncertain"}:
        return f"{solver_prefix}_{uniqueness_key}"
    return f"{solver_prefix}_other_uniqueness"


def _result_row(
    *,
    version: str,
    source_path: Path,
    record: dict[str, Any],
    solver_status: str,
    output_path: str,
) -> dict[str, Any]:
    target = record.get("target")
    target = target if isinstance(target, dict) else {}
    summary = record.get("solver_summary")
    summary = summary if isinstance(summary, dict) else {}

    return {
        "source": _relative_path(source_path),
        "output": output_path,
        "run_version": version,
        "seed_file": source_path.name,
        "reason": str(summary.get("reason") or ""),
        "solver_total": summary.get("total"),
        "solver_correct": summary.get("correct"),
        "solver_incorrect": summary.get("incorrect"),
        "status": solver_status,
        "solver_status": solver_status,
        "artifact_status": str(record.get("status") or ""),
        "uniqueness_key": str(record.get("uniqueness_key") or ""),
        "question": str(record.get("question") or ""),
        "answer": str(target.get("answer") or ""),
        "answer_field": str(target.get("answer_field") or ""),
        "entity_id": str(target.get("entity_id") or ""),
        "entity_type": str(target.get("entity_type") or ""),
        "domain_family": str(target.get("domain_family") or ""),
    }


def _write_jsonl(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
    handle.write("\n")


def parse_versions(raw: str) -> tuple[int, ...]:
    versions = []
    for item in raw.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item.startswith("v"):
            item = item[1:]
        version = int(item)
        if version not in versions:
            versions.append(version)
    return tuple(versions)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--versions",
        default=",".join(str(item) for item in DEFAULT_VERSIONS),
        help="Comma-separated run versions (default: 24,25,26,27,28,29,30)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/result",
        help="Output directory relative to the project root (default: data/result)",
    )
    args = parser.parse_args()

    try:
        versions = parse_versions(args.versions)
    except ValueError as exc:
        parser.error(f"invalid --versions: {exc}")

    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.relative_to(PROJECT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "extraction_summary.json"

    counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    uniqueness_counts: Counter[str] = Counter()
    status_by_uniqueness: dict[str, Counter[str]] = {}
    errors: list[dict[str, str]] = []
    scanned = 0
    valid = 0

    output_paths = {key: output_dir / filename for key, filename in OUTPUT_FILENAMES.items()}
    output_files = {
        key: _relative_path(path) for key, path in output_paths.items()
    }

    with ExitStack() as stack:
        output_handles = {
            key: stack.enter_context(path.open("w", encoding="utf-8"))
            for key, path in output_paths.items()
        }
        for version, source_path in _seed_files(versions):
            scanned += 1
            source = _relative_path(source_path)
            try:
                with source_path.open("r", encoding="utf-8") as handle:
                    record = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                counts["invalid_json"] += 1
                errors.append({"source": source, "error": f"{type(exc).__name__}: {exc}"})
                continue
            if not isinstance(record, dict):
                counts["invalid_record"] += 1
                errors.append({"source": source, "error": "top-level JSON value is not an object"})
                continue

            valid += 1
            has_question = _has_question(record)
            solver_status = _solver_status(record)
            counts["with_question" if has_question else "without_question"] += 1
            if solver_status:
                counts["with_solver_difficulty"] += 1
                status_counts[solver_status] += 1
            else:
                counts["without_solver_difficulty"] += 1

            if not has_question or not solver_status:
                counts["incomplete_skipped"] += 1
                continue

            uniqueness_key = str(record.get("uniqueness_key") or "missing")
            uniqueness_counts[uniqueness_key] += 1
            status_by_uniqueness.setdefault(solver_status, Counter())[uniqueness_key] += 1

            group_key = _group_key(solver_status, uniqueness_key)
            output_path = output_files.get(group_key)
            if not output_path:
                counts["unsupported_solver_status_skipped"] += 1
                continue

            row = _result_row(
                version=version,
                source_path=source_path,
                record=record,
                solver_status=solver_status,
                output_path=output_path,
            )
            counts[group_key] += 1
            _write_jsonl(output_handles[group_key], row)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_versions": [f"v{version}" for version in versions],
        "input_pattern": "data/runs_v*/seed_*.json",
        "output_files": output_files,
        "files_scanned": scanned,
        "valid_seed_files": valid,
        "counts": dict(sorted(counts.items())),
        "solver_status_counts": dict(sorted(status_counts.items())),
        "uniqueness_counts": dict(sorted(uniqueness_counts.items())),
        "solver_status_by_uniqueness": {
            status: dict(sorted(values.items()))
            for status, values in sorted(status_by_uniqueness.items())
        },
        "errors": errors,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    sys.exit(main())
