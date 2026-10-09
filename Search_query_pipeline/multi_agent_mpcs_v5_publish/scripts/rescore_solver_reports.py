#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from browsecomp_v2.config import load_config
from browsecomp_v2.workflow import BrowseCompV2Workflow, SolverSummary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=".env", help="Path to env file.")
    parser.add_argument("--run-dir", default="data/runs", help="Directory with seed_*.json artifacts.")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing artifacts.")
    args = parser.parse_args()

    config = load_config(args.env)
    workflow = BrowseCompV2Workflow(config)
    run_dir = Path(args.run_dir)
    paths = sorted(run_dir.glob("seed_*.json"))
    print(
        "rescore_start: "
        f"artifacts={len(paths)} solver_model={config.agents['solver'].model} "
        f"solver_verifier_model={config.agents['solver_verifier'].model}",
        flush=True,
    )

    changed_files = 0
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        expected = str(data.get("target", {}).get("answer", ""))
        question = str(data.get("question", ""))
        reports = data.get("solver_reports") or []
        if not reports:
            print(f"skip: {path.name} reason=no_solver_reports", flush=True)
            continue

        old_correct = sum(1 for item in reports if item.get("verifier_is_correct"))
        for report in reports:
            old_value = bool(report.get("verifier_is_correct"))
            old_reason = str(report.get("verifier_reason", ""))
            final_answer = report.get("final_answer", "")
            if report.get("solver_error") or not str(final_answer).strip():
                verifier = {
                    "is_correct": None,
                    "status": "verification_failed",
                    "reason": report.get("verifier_reason")
                    or "solver execution failed; correctness was not evaluated",
                }
            else:
                verifier = workflow._verify_solver_answer(report, expected, question=question)
            report["previous_verifier_is_correct"] = old_value
            report["previous_verifier_reason"] = old_reason
            report["verifier_is_correct"] = verifier.get("is_correct")
            report["verifier_status"] = str(verifier.get("status", "verified"))
            report["verifier_reason"] = str(verifier.get("reason", ""))

        summary = workflow._summarize_solvers(reports, expected)
        old_summary = data.get("solver_summary") or {}
        old_status = data.get("status", "")
        data["solver_summary"] = _summary_dict(summary)
        data["status"] = summary.status
        for item in reversed(data.get("iterations", [])):
            if item.get("stage") == "solver":
                item["status"] = summary.status
                item["summary"] = _summary_dict(summary)
                break

        new_correct = summary.correct
        changed = (
            old_correct != new_correct
            or old_summary.get("status") != summary.status
            or old_status != summary.status
        )
        if changed:
            changed_files += 1
        print(
            "rescored: "
            f"{path.name} correct={old_correct}->{new_correct}/{summary.total} "
            f"status={old_summary.get('status')}->{summary.status}",
            flush=True,
        )
        if not args.dry_run:
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"rescore_done: changed_files={changed_files} dry_run={args.dry_run}", flush=True)


def _summary_dict(summary: SolverSummary) -> dict[str, object]:
    return {
        "total": summary.total,
        "correct": summary.correct,
        "incorrect": summary.incorrect,
        "accepted": summary.accepted,
        "status": summary.status,
        "reason": summary.reason,
        "verification_failed": summary.verification_failed,
    }


if __name__ == "__main__":
    main()
