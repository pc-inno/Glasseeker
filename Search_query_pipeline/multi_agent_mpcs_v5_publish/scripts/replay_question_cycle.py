#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from browsecomp_v2.config import load_config
from browsecomp_v2.schema import (
    Artifact,
    ConstraintPath,
    SolverSummary,
    Target,
    VerifierReport,
)
from browsecomp_v2.workflow import (
    BrowseCompV2Workflow,
    _solver_summary_score,
    save_artifact,
)


def _artifact_from_tree(path: Path, seed_dir: Path) -> Artifact:
    raw = json.loads(path.read_text(encoding="utf-8"))
    verifier_raw = raw.get("verifier") or {}
    verifier = VerifierReport(
        accepted=bool(verifier_raw.get("accepted")),
        reason=str(verifier_raw.get("reason") or ""),
        core_path_ids=list(verifier_raw.get("core_path_ids") or []),
        target_key=str(verifier_raw.get("target_key") or ""),
        single_path_candidates={
            str(key): list(value)
            for key, value in (verifier_raw.get("single_path_candidates") or {}).items()
        },
        all_core_candidates=list(verifier_raw.get("all_core_candidates") or []),
        distractor_report=dict(verifier_raw.get("distractor_report") or {}),
    )
    artifact_path = seed_dir / "artifact.json"
    return Artifact(
        target=Target(**raw["target"]),
        constraints=[ConstraintPath.from_dict(item) for item in raw.get("constraints", [])],
        verifier=verifier,
        root_ambiguity_report=dict(raw.get("root_ambiguity_report") or {}),
        iterations=[
            deepcopy(item)
            for item in raw.get("iterations", [])
            if isinstance(item, dict)
            and item.get("stage") in {"constraint", "program_verifier", "local_constraint"}
        ],
        run_context={
            "seed_index": 1,
            "seed_id": str(raw["target"].get("entity_id") or "replay"),
            "seed_dir": str(seed_dir.resolve()),
            "artifact_path": str(artifact_path.resolve()),
            "source_artifact": str(path.resolve()),
            "question_cycle_replay": True,
        },
        notes=[f"Question-cycle replay from {path.resolve()}"],
    )


def _state_from_response(response: dict) -> dict:
    keys = (
        "question",
        "answer",
        "used_core_path_ids",
        "used_distractor_path_ids",
        "chain_node_usage",
        "fuzzified_child_path_ids",
        "note",
        "required_core_root_ids",
        "generalized_required_root_ids",
    )
    return {key: deepcopy(response.get(key)) for key in keys}


def _finalize_status(summary: SolverSummary) -> str:
    if summary.accepted:
        return summary.status
    if summary.status == "needs_repair:too_hard":
        return "review:all_wrong"
    if summary.status == "needs_repair:too_easy":
        return "rejected:too_easy"
    if summary.status == "needs_repair:ambiguous":
        return "rejected:ambiguous"
    return summary.status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_artifact")
    parser.add_argument("--env", default=".env_v2")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--solver-concurrency", type=int, default=3)
    parser.add_argument(
        "--solver-rollouts",
        type=int,
        default=None,
        help="diagnostic override; production keeps the environment rollout count",
    )
    parser.add_argument(
        "--question-only",
        action="store_true",
        help="stop after Question validation and uniqueness; do not launch Solver",
    )
    args = parser.parse_args()

    source = Path(args.source_artifact).resolve()
    out_dir = Path(args.out_dir).resolve()
    seed_slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", source.parent.name)
    seed_dir = out_dir / f"seed_001_{seed_slug}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    os.environ["V2_OUTPUT_DIR"] = str(out_dir)
    os.environ["V2_CONVERSATION_LOG_DIR"] = str(out_dir / "conversations")

    config = load_config(args.env)
    config_overrides = {
        "output_dir": out_dir,
        "solver_concurrency": max(1, args.solver_concurrency),
    }
    if args.solver_rollouts is not None:
        config_overrides["solver_rollouts"] = max(1, args.solver_rollouts)
    config = replace(config, **config_overrides)
    # load_config may layer test output defaults; the explicit replay target wins.
    os.environ["V2_OUTPUT_DIR"] = str(out_dir)
    os.environ["V2_CONVERSATION_LOG_DIR"] = str(out_dir / "conversations")
    workflow = BrowseCompV2Workflow(config)
    artifact = _artifact_from_tree(source, seed_dir)
    artifact_path = seed_dir / "artifact.json"

    def checkpoint() -> None:
        save_artifact(artifact, artifact_path)

    artifact.iterations.append(
        {"stage": "question_cycle_replay", "status": "started", "source": str(source)}
    )
    checkpoint()

    response = workflow._write_question(artifact, revision=0)
    artifact.question = str(response.get("question") or "")
    artifact.question_state = _state_from_response(response)
    artifact.iterations.append(
        {
            "stage": "question",
            "status": "completed" if artifact.question else "failed",
            "response": response,
        }
    )
    if not artifact.question:
        artifact.status = "rejected:no_question"
        checkpoint()
        return
    workflow._record_question_version(
        artifact, source="question", revision=0, response=response
    )
    checkpoint()

    if not workflow._repair_question_until_unique(artifact):
        checkpoint()
        return

    if args.question_only:
        artifact.status = "accepted:question_only"
        artifact.iterations.append(
            {"stage": "question_cycle_replay", "status": artifact.status}
        )
        checkpoint()
        print(artifact_path, flush=True)
        return

    previous_questions = [artifact.question]
    best_accepted_snapshot: dict | None = None
    for revision in range(config.question_revisions + 1):
        question_version = workflow._current_question_version(artifact)
        workflow._write_solver_question_snapshot(
            artifact, question_version=question_version
        )
        reports = workflow._run_solvers(
            artifact, question_version=question_version
        )
        summary = workflow._summarize_solvers(reports, artifact.target.answer)
        artifact.solver_reports = reports
        artifact.solver_summary = summary
        feedback = workflow._summarize_solver_trajectories(
            artifact, reports, summary
        )
        if feedback.get("diagnosis") == "invalid_or_ambiguous":
            summary.accepted = False
            summary.status = "needs_repair:ambiguous"
            summary.reason = (
                "A strong adjudicator verified an alternate answer satisfying the public wording. "
                + str(feedback.get("repair_guidance") or "")
            ).strip()

        attempt = {
            "attempt": len(artifact.solver_attempts),
            "question_revision": revision,
            "question_version": question_version,
            "question": artifact.question,
            "question_state": deepcopy(artifact.question_state),
            "uniqueness_key": artifact.uniqueness_key,
            "uniqueness_report": deepcopy(artifact.uniqueness_report),
            "solver_reports": deepcopy(reports),
            "solver_summary": asdict(summary),
            "trajectory_feedback": deepcopy(feedback),
        }
        artifact.solver_attempts.append(attempt)
        artifact.iterations.extend(
            [
                {
                    "stage": "solver",
                    "status": summary.status,
                    "question_revision": revision,
                    "question": artifact.question,
                    "summary": asdict(summary),
                },
                {
                    "stage": "solver_trajectory_summary",
                    "status": str(feedback.get("diagnosis") or "unknown"),
                    "question_revision": revision,
                    "report": feedback,
                },
            ]
        )
        workflow._write_solver_attempt_snapshot(artifact, attempt)
        checkpoint()

        if summary.accepted:
            candidate_snapshot = {
                "question": artifact.question,
                "question_state": deepcopy(artifact.question_state),
                "solver_reports": deepcopy(reports),
                "solver_summary": deepcopy(summary),
                "uniqueness_key": artifact.uniqueness_key,
                "uniqueness_report": deepcopy(artifact.uniqueness_report),
            }
            if best_accepted_snapshot is None or _solver_summary_score(
                summary
            ) > _solver_summary_score(best_accepted_snapshot["solver_summary"]):
                best_accepted_snapshot = candidate_snapshot
        if summary.accepted and not (
            feedback.get("shortcut_root_path_ids")
            or feedback.get("shortcut_child_path_ids")
            or feedback.get("shortcut_queries")
            or feedback.get("recoverable_intermediate_entities")
        ):
            break
        if summary.status == "review:all_wrong":
            break
        if (
            not summary.accepted
            and summary.status
            not in {
                "needs_repair:too_easy",
                "needs_repair:too_hard",
                "needs_repair:ambiguous",
            }
        ):
            break
        if revision >= config.question_revisions:
            break
        failure = workflow._solver_repair_failure_report(
            reports,
            summary,
            previous_questions,
            feedback,
        )
        repaired = workflow._write_question(
            artifact,
            revision=revision + 1,
            previous_questions=previous_questions,
            failure_report=failure,
            force_fallback=revision > 0,
        )
        new_question = str(repaired.get("question") or "").strip()
        artifact.iterations.append(
            {
                "stage": "question_revision",
                "status": "completed" if new_question else "failed",
                "revision": revision + 1,
                "failure_report": failure,
                "response": repaired,
            }
        )
        if not new_question or new_question in previous_questions:
            break
        artifact.question = new_question
        artifact.question_state = _state_from_response(repaired)
        previous_questions.append(new_question)
        workflow._record_question_version(
            artifact,
            source="solver_repair",
            revision=revision + 1,
            response=repaired,
        )
        checkpoint()
        if not workflow._repair_question_until_unique(artifact):
            checkpoint()
            return

    if best_accepted_snapshot is not None and (
        artifact.solver_summary is None
        or not artifact.solver_summary.accepted
        or _solver_summary_score(artifact.solver_summary)
        < _solver_summary_score(best_accepted_snapshot["solver_summary"])
        or artifact.question != best_accepted_snapshot["question"]
        or artifact.uniqueness_key != best_accepted_snapshot["uniqueness_key"]
    ):
        artifact.question = best_accepted_snapshot["question"]
        artifact.question_state = best_accepted_snapshot["question_state"]
        artifact.solver_reports = best_accepted_snapshot["solver_reports"]
        artifact.solver_summary = best_accepted_snapshot["solver_summary"]
        artifact.uniqueness_key = best_accepted_snapshot["uniqueness_key"]
        artifact.uniqueness_report = best_accepted_snapshot["uniqueness_report"]
    if artifact.solver_summary is None:
        artifact.status = "rejected:no_solver_result"
    else:
        artifact.status = _finalize_status(artifact.solver_summary)
        artifact.solver_summary.status = artifact.status
    artifact.iterations.append(
        {"stage": "question_cycle_replay", "status": artifact.status}
    )
    checkpoint()
    print(artifact_path, flush=True)


if __name__ == "__main__":
    main()
