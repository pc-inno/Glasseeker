#!/usr/bin/env python3
"""Incrementally collect hard, unique questions from selected run directories."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FILTER_DIR = DATA_DIR / "filter_data"
MANIFEST = FILTER_DIR / "uniqueness_manifest.jsonl"
OUTPUT_DIR = FILTER_DIR / "uniqueness"
LOCK_PATH = FILTER_DIR / ".collect_uniqueness_incremental.lock"


def _question_key(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def _load_manifest() -> tuple[set[str], set[str]]:
    sources: set[str] = set()
    questions: set[str] = set()
    if not MANIFEST.exists():
        return sources, questions
    for line_number, raw_line in enumerate(
        MANIFEST.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid manifest JSON at line {line_number}: {exc}") from exc
        source = str(record.get("source") or "")
        question = _question_key(record.get("question"))
        if source:
            sources.add(source)
        if question:
            questions.add(question)
    return sources, questions


def _atomic_write_bytes(destination: Path, payload: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, destination)


def _collect(*, dry_run: bool, run_dirs: list[Path]) -> Counter[str]:
    stats: Counter[str] = Counter()
    existing_sources, existing_questions = _load_manifest()
    new_records: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        if not run_dir.exists():
            stats["missing_run_dirs"] += 1
            continue
        for path in sorted(run_dir.glob("*.json")):
            stats["scanned"] += 1
            try:
                payload = path.read_bytes()
                data = json.loads(payload)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                # A producer may still be writing this file. The next hourly run retries it.
                stats["unreadable_or_incomplete"] += 1
                continue

            uniqueness = data.get("uniqueness_report") or {}
            if not (uniqueness.get("enabled") and uniqueness.get("unique")):
                stats["not_unique"] += 1
                continue

            solver_reports = data.get("solver_reports") or []
            incorrect = sum(
                isinstance(report, dict) and report.get("verifier_is_correct") is False
                for report in solver_reports
            )
            if incorrect < 2:
                stats["solver_fewer_than_2_incorrect"] += 1
                continue

            incorrect_with_web = sum(
                isinstance(report, dict)
                and report.get("verifier_is_correct") is False
                and isinstance(report.get("_execution"), dict)
                and report["_execution"].get("backend") == "hermes"
                and int(report["_execution"].get("successful_web_tool_count") or 0) > 0
                for report in solver_reports
            )
            if incorrect_with_web < 2:
                stats["solver_fewer_than_2_incorrect_with_web"] += 1
                continue

            question = str(data.get("question") or "").strip()
            answer = str((data.get("target") or {}).get("answer") or "").strip()
            if not question or not answer:
                stats["missing_question_or_answer"] += 1
                continue

            relative = path.resolve().relative_to(DATA_DIR.resolve())
            source = str(relative)
            destination = OUTPUT_DIR / relative
            if source in existing_sources:
                stats["already_in_manifest"] += 1
                if not dry_run:
                    _atomic_write_bytes(destination, payload)
                continue

            question_key = _question_key(question)
            if question_key in existing_questions:
                stats["duplicate_question"] += 1
                continue

            record = {
                "source": source,
                "output": str(destination.relative_to(FILTER_DIR)),
                "reason": "solver_at_least_2_incorrect",
                "solver_total": len(solver_reports),
                "solver_incorrect": incorrect,
                "solver_incorrect_with_web": incorrect_with_web,
                "status": data.get("status"),
                "uniqueness_key": uniqueness.get("key"),
                "question": question,
                "answer": answer,
            }
            new_records.append(record)
            existing_sources.add(source)
            existing_questions.add(question_key)
            stats["collected"] += 1
            if not dry_run:
                _atomic_write_bytes(destination, payload)

    if new_records and not dry_run:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        with MANIFEST.open("a", encoding="utf-8") as manifest:
            for record in new_records:
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            manifest.flush()
            os.fsync(manifest.fileno())
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help="Artifact directory to scan; may be provided more than once.",
    )
    args = parser.parse_args()

    raw_run_dirs = args.run_dir or ["data/runs_v20"]
    run_dirs = [
        path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
        for path in map(Path, raw_run_dirs)
    ]
    data_root = DATA_DIR.resolve()
    for run_dir in run_dirs:
        try:
            run_dir.relative_to(data_root)
        except ValueError as exc:
            raise SystemExit(f"--run-dir must be inside {DATA_DIR}: {run_dir}") from exc

    FILTER_DIR.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        stats = _collect(dry_run=args.dry_run, run_dirs=run_dirs)

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        **dict(sorted(stats.items())),
        "manifest": str(MANIFEST),
        "run_dirs": [str(path) for path in run_dirs],
    }
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
