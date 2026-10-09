#!/usr/bin/env python3
"""Export too-easy questions with a correct solver rollout over an API-call limit."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import multiprocessing
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILENAME = "rejected_too_easy_correct_over_50_iterations.jsonl"
SUMMARY_FILENAME = "rejected_too_easy_correct_over_50_iterations_summary.json"
TOO_EASY_STATUS = "rejected:too_easy"


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def _parse_versions(raw: str) -> tuple[int, ...]:
    if not raw.strip():
        versions = []
        for path in (PROJECT_ROOT / "data").glob("runs_v*"):
            match = re.fullmatch(r"runs_v(\d+)", path.name)
            if match and path.is_dir():
                versions.append(int(match.group(1)))
        return tuple(sorted(set(versions)))

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


def _question_hash(question: str) -> str:
    return hashlib.sha256(question.strip().encode("utf-8")).hexdigest()


def _report_signature(report: dict[str, Any]) -> str:
    payload = {
        key: report.get(key)
        for key in ("final_answer", "confidence", "evidence", "reasoning_summary")
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_named_value(
    path: Path,
    key: str,
    *,
    initial_bytes: int,
    max_bytes: int,
    from_end: bool = False,
) -> Any:
    marker = f'\n  "{key}": '.encode("utf-8")
    size = path.stat().st_size
    amounts = [min(size, initial_bytes)]
    if size > amounts[0]:
        amounts.append(min(size, max_bytes))

    for amount in amounts:
        with path.open("rb") as handle:
            if from_end:
                handle.seek(max(0, size - amount))
            data = handle.read(amount)
        offset = data.rfind(marker) if from_end else data.find(marker)
        if offset < 0:
            continue
        try:
            value, _ = json.JSONDecoder().raw_decode(
                data[offset + len(marker) :].decode("utf-8")
            )
            return value
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    return None


def _read_user_payload(path: Path) -> dict[str, Any] | None:
    value = _read_named_value(
        path,
        "user_payload",
        initial_bytes=700_000,
        max_bytes=10_000_000,
    )
    return value if isinstance(value, dict) else None


def _read_parsed_response(path: Path) -> dict[str, Any] | None:
    value = _read_named_value(
        path,
        "parsed_response",
        initial_bytes=500_000,
        max_bytes=2_000_000,
        from_end=True,
    )
    return value if isinstance(value, dict) else None


def _decode_hermes_response(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, str):
        return None
    decoder = json.JSONDecoder()
    for index, char in enumerate(raw):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "messages" in value and "api_calls" in value:
            return value
    return None


def _solver_status(record: dict[str, Any]) -> str:
    summary = record.get("solver_summary")
    if isinstance(summary, dict):
        status = summary.get("status")
        if isinstance(status, str) and status.strip():
            return status.strip()
    status = record.get("status")
    return status if isinstance(status, str) else ""


def _conversation_files(version: int) -> list[Path]:
    conversation_dir = PROJECT_ROOT / "data" / f"runs_v{version}" / "conversations"
    return sorted(conversation_dir.glob("*_solver_[0-9a-f]*.json"))


def _artifact_files(version: int) -> Iterable[Path]:
    return sorted((PROJECT_ROOT / "data" / f"runs_v{version}").glob("seed_*.json"))


def _process_version(args: tuple[int, int]) -> dict[str, Any]:
    version, threshold = args
    artifacts: dict[tuple[str, str], dict[str, Any]] = {}
    sources_by_question: dict[str, list[str]] = defaultdict(list)
    invalid_artifacts = []

    for path in _artifact_files(version):
        try:
            with path.open("r", encoding="utf-8") as handle:
                record = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            invalid_artifacts.append(
                {"source": _relative(path), "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        question = record.get("question")
        if _solver_status(record) != TOO_EASY_STATUS:
            continue
        if not isinstance(question, str) or not question.strip():
            continue

        source = _relative(path)
        question_key = _question_hash(question)
        reports = {
            report.get("rollout_id"): report
            for report in (record.get("solver_reports") or [])
            if isinstance(report, dict)
        }
        artifacts[(source, question_key)] = {
            "path": path,
            "mtime": path.stat().st_mtime,
            "record": record,
            "reports": reports,
        }
        sources_by_question[question_key].append(source)

    candidates: dict[tuple[str, Any], list[tuple[float, Path]]] = defaultdict(list)
    conversation_files = _conversation_files(version)
    unreadable_user_payloads = 0

    for index, path in enumerate(conversation_files, 1):
        payload = _read_user_payload(path)
        if payload is None:
            unreadable_user_payloads += 1
            continue
        question = payload.get("question")
        if not isinstance(question, str):
            continue
        question_key = _question_hash(question)
        rollout_id = payload.get("rollout_id")
        mtime = path.stat().st_mtime
        for source in sources_by_question.get(question_key, []):
            artifact = artifacts[(source, question_key)]
            if mtime <= artifact["mtime"] + 30:
                candidates[(source, rollout_id)].append((mtime, path))
        if index % 200 == 0:
            gc.collect()

    selected = {
        key: max(values, key=lambda item: item[0])[1]
        for key, values in candidates.items()
    }

    correct_reports: dict[tuple[str, Any], dict[str, Any]] = {}
    for (source, _), artifact in artifacts.items():
        for rollout_id, report in artifact["reports"].items():
            if report.get("verifier_is_correct") is True:
                correct_reports[(source, rollout_id)] = report

    missing = {
        key: report
        for key, report in correct_reports.items()
        if key not in selected
    }
    if missing:
        missing_signatures: dict[str, list[tuple[str, Any]]] = defaultdict(list)
        for key, report in missing.items():
            missing_signatures[_report_signature(report)].append(key)

        signature_candidates: dict[str, list[tuple[float, Path]]] = defaultdict(list)
        for path in conversation_files:
            response = _read_parsed_response(path)
            if response is None:
                continue
            signature = _report_signature(response)
            if signature in missing_signatures:
                signature_candidates[signature].append((path.stat().st_mtime, path))

        for signature, keys in missing_signatures.items():
            paths = signature_candidates.get(signature, [])
            for key in keys:
                source, _ = key
                artifact_mtime = next(
                    artifact["mtime"]
                    for (artifact_source, _), artifact in artifacts.items()
                    if artifact_source == source
                )
                eligible = [item for item in paths if item[0] <= artifact_mtime + 30]
                if eligible:
                    selected[key] = max(eligible, key=lambda item: item[0])[1]

    rows_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    mapped_correct = 0
    errors = list(invalid_artifacts)

    for key, report in correct_reports.items():
        source, rollout_id = key
        path = selected.get(key)
        if path is None:
            errors.append(
                {
                    "source": source,
                    "error": f"solver conversation not found for rollout {rollout_id}",
                }
            )
            continue
        mapped_correct += 1
        try:
            with path.open("r", encoding="utf-8") as handle:
                conversation = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(
                {
                    "source": _relative(path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        hermes = _decode_hermes_response(conversation.get("raw_response"))
        api_calls = hermes.get("api_calls") if hermes else None
        if not isinstance(api_calls, int) or api_calls <= threshold:
            continue
        agent = conversation.get("agent")
        agent = agent if isinstance(agent, dict) else {}
        rows_by_source[source].append(
            {
                "rollout_id": rollout_id,
                "api_calls": api_calls,
                "model": str(agent.get("model") or ""),
                "backend": str(agent.get("backend") or ""),
                "conversation": _relative(path),
                "final_answer": str(report.get("final_answer") or ""),
                "verifier_is_correct": True,
            }
        )

    output_rows = []
    output_path = f"data/result/{OUTPUT_FILENAME}"
    for (source, _), artifact in sorted(artifacts.items()):
        qualifying = rows_by_source.get(source)
        if not qualifying:
            continue
        record = artifact["record"]
        target = record.get("target")
        target = target if isinstance(target, dict) else {}
        summary = record.get("solver_summary")
        summary = summary if isinstance(summary, dict) else {}
        qualifying.sort(key=lambda item: (item["rollout_id"], item["api_calls"]))
        output_rows.append(
            {
                "source": source,
                "output": output_path,
                "run_version": f"v{version}",
                "seed_file": Path(source).name,
                "reason": str(summary.get("reason") or ""),
                "solver_total": summary.get("total"),
                "solver_correct": summary.get("correct"),
                "solver_incorrect": summary.get("incorrect"),
                "status": TOO_EASY_STATUS,
                "solver_status": TOO_EASY_STATUS,
                "artifact_status": str(record.get("status") or ""),
                "uniqueness_key": str(record.get("uniqueness_key") or ""),
                "question": str(record.get("question") or ""),
                "answer": str(target.get("answer") or ""),
                "answer_field": str(target.get("answer_field") or ""),
                "entity_id": str(target.get("entity_id") or ""),
                "entity_type": str(target.get("entity_type") or ""),
                "domain_family": str(target.get("domain_family") or ""),
                "classification": "correct_solver_rollout_over_iteration_limit",
                "solver_api_calls_threshold": threshold,
                "qualifying_rollout_count": len(qualifying),
                "max_solver_api_calls": max(item["api_calls"] for item in qualifying),
                "qualifying_solver_rollouts": qualifying,
            }
        )

    iteration_bins: Counter[str] = Counter()
    for row in output_rows:
        for rollout in row["qualifying_solver_rollouts"]:
            calls = rollout["api_calls"]
            if calls <= 60:
                iteration_bins["51-60"] += 1
            elif calls <= 100:
                iteration_bins["61-100"] += 1
            elif calls <= 150:
                iteration_bins["101-150"] += 1
            else:
                iteration_bins["151+"] += 1

    return {
        "version": f"v{version}",
        "too_easy_questions": len(artifacts),
        "correct_rollouts": len(correct_reports),
        "mapped_correct_rollouts": mapped_correct,
        "qualifying_questions": len(output_rows),
        "qualifying_rollouts": sum(
            row["qualifying_rollout_count"] for row in output_rows
        ),
        "iteration_bins": dict(sorted(iteration_bins.items())),
        "unreadable_user_payloads": unreadable_user_payloads,
        "rows": output_rows,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--versions",
        default="",
        help="Comma-separated run versions; default discovers data/runs_v* directories",
    )
    parser.add_argument(
        "--api-call-threshold",
        type=int,
        default=50,
        help="Require a correct solver rollout to exceed this Hermes api_calls value",
    )
    parser.add_argument(
        "--output-dir",
        default="data/result",
        help="Output directory relative to the project root",
    )
    args = parser.parse_args()

    try:
        versions = _parse_versions(args.versions)
    except ValueError as exc:
        parser.error(f"invalid --versions: {exc}")
    if not versions:
        parser.error("no run versions found")
    if args.api_call_threshold < 0:
        parser.error("--api-call-threshold must be non-negative")

    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.relative_to(PROJECT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_FILENAME
    summary_path = output_dir / SUMMARY_FILENAME

    context = multiprocessing.get_context("fork")
    results = []
    with context.Pool(1, maxtasksperchild=1) as pool:
        for result in pool.imap(
            _process_version,
            [(version, args.api_call_threshold) for version in versions],
        ):
            results.append(result)

    rows = [row for result in results for row in result.pop("rows")]
    rows.sort(key=lambda row: (row["run_version"], row["source"]))
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")

    total_errors = [error for result in results for error in result["errors"]]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_versions": [f"v{version}" for version in versions],
        "classification": "correct_solver_rollout_over_iteration_limit",
        "solver_api_calls_threshold": args.api_call_threshold,
        "output_file": _relative(output_path),
        "qualifying_questions": len(rows),
        "qualifying_rollouts": sum(
            row["qualifying_rollout_count"] for row in rows
        ),
        "versions": results,
        "errors": total_errors,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not total_errors else 2


if __name__ == "__main__":
    sys.exit(main())
