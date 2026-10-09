#!/usr/bin/env python3
"""Run only the existing root/local expansion stages for a seed batch."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from browsecomp_v2.config import load_config  # noqa: E402
from browsecomp_v2.schema import SeedRecord  # noqa: E402
from browsecomp_v2.workflow import BrowseCompV2Workflow  # noqa: E402
from stage_common import (  # noqa: E402
    artifact_from_expansion,
    configure_stage_environment,
    configure_stage_logging,
    load_seed_files,
    persist_stage_failure,
    read_json,
    save_seed_record,
    seed_dir_name,
    write_json,
)


def _expansion_succeeded(payload: dict) -> bool:
    if not bool((payload.get("verifier") or {}).get("accepted")):
        return False
    local_iterations = [
        item
        for item in payload.get("iterations", [])
        if isinstance(item, dict) and item.get("stage") == "local_constraint"
    ]
    return bool(local_iterations) and local_iterations[-1].get("status") == "completed"


def _artifact_path(output_batch_dir: Path, index: int, seed: SeedRecord) -> Path:
    return output_batch_dir / seed_dir_name(index, seed.target.entity_id) / "artifact.json"


def _persist_expansion_result(
    *,
    seed: SeedRecord,
    output_path: Path,
    input_seed_path: Path,
) -> dict:
    """Add stage metadata and normalize the terminal expansion status."""
    payload = read_json(output_path)
    payload["stage"] = "expand"
    payload["input_seed_path"] = str(input_seed_path.resolve())
    payload["status"] = (
        "expanded" if _expansion_succeeded(payload) else "rejected:expansion"
    )
    write_json(output_path, payload)
    seed_folder = output_path.parent
    save_seed_record(seed, seed_folder / "seed.json")
    write_json(
        seed_folder / "stage_metadata.json",
        {
            "stage": "expand",
            "input_seed_path": str(input_seed_path.resolve()),
            "artifact_path": str(output_path.resolve()),
            "status": payload["status"],
        },
    )
    return payload


def _resume_local_from_checkpoint(
    *,
    seed: SeedRecord,
    seed_path: Path,
    output_path: Path,
    config,
    dry_run: bool,
) -> bool:
    """Continue a checkpoint that finished Root but stopped before Local.

    The existing workflow checkpoints before and after each major stage, but it
    does not serialize every Local node.  Resuming at the Local boundary avoids
    repeating Root calls while preserving the existing Local implementation.
    """
    if dry_run:
        # Dry-run intentionally rebuilds the synthetic artifact; do not call
        # Local models while trying to resume a smoke-test checkpoint.
        return False
    try:
        payload = read_json(output_path)
        verifier = payload.get("verifier")
        constraints = payload.get("constraints")
        if not isinstance(verifier, dict) or not verifier.get("accepted"):
            return False
        if not isinstance(constraints, list) or not constraints:
            return False
        local_iterations = [
            item
            for item in payload.get("iterations", [])
            if isinstance(item, dict) and item.get("stage") == "local_constraint"
        ]
        # A completed or explicitly failed Local stage needs the normal full
        # workflow revision policy; only an interrupted pre-Local checkpoint is
        # safe to continue here.
        if local_iterations and local_iterations[-1].get("status") in {
            "completed",
            "failed",
        }:
            return False

        run_context = {
            "seed_index": int(payload.get("run_context", {}).get("seed_index") or 0),
            "seed_id": seed.target.entity_id,
            "seed_dir": str(output_path.parent.resolve()),
            "artifact_path": str(output_path.resolve()),
        }
        artifact = artifact_from_expansion(payload, run_context=run_context)
        workflow = BrowseCompV2Workflow(config)
        updated_constraints, local_report = workflow._local_fuzzify(artifact)
        artifact.constraints = updated_constraints
        revision = max(
            (
                int(item.get("revision") or 0)
                for item in artifact.iterations
                if isinstance(item, dict) and item.get("stage") == "constraint"
            ),
            default=0,
        )
        artifact.iterations.append(
            {
                "stage": "local_constraint",
                "status": "completed" if local_report.get("accepted", True) else "failed",
                "revision": revision,
                "constraint_count": len(artifact.constraints),
                "report": local_report,
                "resumed": True,
            }
        )
        workflow._checkpoint_artifact(artifact)
        _persist_expansion_result(
            seed=seed,
            output_path=output_path,
            input_seed_path=seed_path,
        )
        return True
    except Exception as exc:
        print(
            f"expand_resume_local_failed: target={seed.target.entity_id} "
            f"error={type(exc).__name__}: {str(exc)[:220]}",
            flush=True,
        )
        return False


def run_one(
    *,
    index: int,
    seed_path: Path,
    output_batch_dir: Path,
    config,
    include_unverified: bool,
    dry_run: bool,
    resume: bool,
) -> Path:
    seed = SeedRecord.from_dict(read_json(seed_path))
    output_path = _artifact_path(output_batch_dir, index, seed)
    if resume and output_path.exists():
        try:
            existing = read_json(output_path)
        except Exception:
            existing = {}
        if _expansion_succeeded(existing):
            _persist_expansion_result(
                seed=seed,
                output_path=output_path,
                input_seed_path=seed_path,
            )
            print(
                f"expand_seed_skipped: idx={index} target={seed.target.entity_id} "
                "reason=already_expanded",
                flush=True,
            )
            return output_path
        if _resume_local_from_checkpoint(
            seed=seed,
            seed_path=seed_path,
            output_path=output_path,
            config=config,
            dry_run=dry_run,
        ):
            print(
                f"expand_seed_resumed: idx={index} target={seed.target.entity_id} "
                "from=local_boundary",
                flush=True,
            )
            return output_path

    input_status = "accepted"
    status_path = seed_path.parent / "status.json"
    if status_path.exists():
        input_status = str(read_json(status_path).get("status") or "accepted")
    if input_status != "accepted" and not include_unverified:
        return persist_stage_failure(
            seed=seed,
            output_dir=output_batch_dir,
            index=index,
            status="rejected:seed_not_verified",
            reason=(
                f"input seed status is {input_status}; use --include-unverified "
                "to override"
            ),
            extra={"input_seed_path": str(seed_path.resolve())},
        )

    workflow = BrowseCompV2Workflow(config)

    # Reuse run_seed's complete root/local implementation and stop exactly at
    # its question boundary.  No expansion prompt, verifier, or retry logic is
    # replaced; this is only a stage adapter.
    def stop_before_question(*_args, **_kwargs):
        return {
            "question": "",
            "answer": seed.target.answer,
            "failure_reason": "expand_stage_only",
        }

    workflow._write_question = stop_before_question
    output_path = workflow.run_seed_and_save(
        seed,
        output_dir=output_batch_dir,
        index=index,
        dry_run=dry_run,
    )
    _persist_expansion_result(
        seed=seed,
        output_path=output_path,
        input_seed_path=seed_path,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand a persisted seed batch.")
    parser.add_argument("--env", default=".env_v5")
    parser.add_argument("--batch-dir", required=True, type=Path)
    parser.add_argument("--output-root", default="data/expand_results")
    parser.add_argument("--batch-id", default="")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--include-unverified", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip completed expansion artifacts and continue checkpoints that "
            "finished Root before Local."
        ),
    )
    args = parser.parse_args()
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be positive")
    input_batch = args.batch_dir.resolve()
    if not input_batch.is_dir():
        raise SystemExit(f"seed batch does not exist: {input_batch}")
    batch_id = args.batch_id.strip() or input_batch.name
    output_batch = (PROJECT_ROOT / args.output_root / batch_id).resolve()
    output_batch.mkdir(parents=True, exist_ok=True)
    configure_stage_logging(output_batch, "expand")
    config = load_config(args.env)
    config = replace(config, seed_verifier_enabled=False)
    if args.dry_run:
        config = replace(
            config,
            root_ambiguity_verifier_enabled=False,
            uniqueness_enabled=False,
            solver_enabled=False,
        )
    configure_stage_environment(output_batch)
    records = load_seed_files(input_batch)
    if not records:
        raise SystemExit(f"no seed subfolders found under {input_batch}")
    paths: list[Path] = []
    skipped_count = 0
    pending_records = []
    if args.resume:
        for record in records:
            index, seed_path, seed = record
            existing_path = _artifact_path(output_batch, index, seed)
            if existing_path.exists():
                try:
                    existing = read_json(existing_path)
                except Exception:
                    existing = {}
                if _expansion_succeeded(existing):
                    _persist_expansion_result(
                        seed=seed,
                        output_path=existing_path,
                        input_seed_path=seed_path,
                    )
                    paths.append(existing_path)
                    skipped_count += 1
                    print(
                        f"expand_seed_skipped: idx={index} target={seed.target.entity_id} "
                        "reason=already_expanded",
                        flush=True,
                    )
                    continue
            pending_records.append(record)
    else:
        pending_records = records

    def persist_manifest() -> None:
        existing_paths = sorted(
            {str(path.resolve()) for path in paths if path.exists()}
        )
        payloads = []
        for raw_path in existing_paths:
            try:
                payloads.append(read_json(Path(raw_path)))
            except Exception:
                continue
        statuses = {str(item.get("status")) for item in payloads}
        write_json(
            output_batch / "manifest.json",
            {
                "batch_id": batch_id,
                "input_batch": str(input_batch),
                "seed_count": len(records),
                "artifact_count": len(existing_paths),
                "expanded_count": sum(
                    item.get("status") == "expanded" for item in payloads
                ),
                "skipped_count": skipped_count,
                "resume": bool(args.resume),
                "pending_count": max(0, len(records) - len(existing_paths)),
                "statuses": sorted(statuses),
                "artifacts": existing_paths,
            },
        )

    persist_manifest()
    print(
        f"expand_stage_start: batch={batch_id} inputs={len(records)} "
        f"pending={len(pending_records)} skipped={skipped_count} "
        f"resume={args.resume} concurrency={args.concurrency} out={output_batch}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=min(args.concurrency, len(records))) as pool:
        futures = {
            pool.submit(
                run_one,
                index=index,
                seed_path=seed_path,
                output_batch_dir=output_batch,
                config=config,
                include_unverified=args.include_unverified,
                dry_run=args.dry_run,
                resume=args.resume,
            ): (index, seed_path, seed)
            for index, seed_path, seed in pending_records
        }
        for future in as_completed(futures):
            index, seed_path, seed = futures[future]
            try:
                path = future.result()
            except Exception as exc:
                path = persist_stage_failure(
                    seed=seed,
                    output_dir=output_batch,
                    index=index,
                    status="rejected:workflow_error",
                    reason=f"expand worker failed: {type(exc).__name__}: {exc}",
                    extra={"input_seed_path": str(seed_path.resolve())},
                )
                print(
                    f"expand_seed_error: idx={index} target={seed.target.entity_id} "
                    f"error={type(exc).__name__}: {str(exc)[:220]}",
                    flush=True,
                )
            paths.append(path)
            print(f"expand_seed_done: idx={index} artifact={path}", flush=True)
            persist_manifest()
    persist_manifest()
    expanded_count = sum(
        read_json(path).get("status") == "expanded"
        for path in paths
        if path.exists()
    )
    print(
        f"expand_stage_done: batch={batch_id} artifacts={len(paths)} "
        f"expanded={expanded_count} skipped={skipped_count}",
        flush=True,
    )


if __name__ == "__main__":
    main()
