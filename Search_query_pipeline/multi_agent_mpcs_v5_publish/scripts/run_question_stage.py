#!/usr/bin/env python3
"""Run the existing question/repair/solver loop from expansion artifacts."""

from __future__ import annotations

import argparse
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from browsecomp_v2.config import load_config  # noqa: E402
from browsecomp_v2.schema import SeedRecord, Target  # noqa: E402
from browsecomp_v2.workflow import BrowseCompV2Workflow  # noqa: E402
from stage_common import (  # noqa: E402
    artifact_from_expansion,
    configure_stage_environment,
    configure_stage_logging,
    copy_json,
    persist_stage_failure,
    read_json,
    save_seed_record,
    write_json,
)


def run_one(
    *,
    index: int,
    artifact_path: Path,
    output_batch_dir: Path,
    config,
    include_failed: bool,
    dry_run: bool,
) -> Path:
    expansion_data = read_json(artifact_path)
    target_data = expansion_data.get("target")
    if not isinstance(target_data, dict):
        raise ValueError(f"expansion artifact has no target: {artifact_path}")
    seed = SeedRecord(
        target=Target(**target_data),
        seed_note=(
            str(expansion_data.get("notes", [""])[0])
            if expansion_data.get("notes")
            else ""
        ),
    )
    expansion_status = str(expansion_data.get("status") or "")
    if expansion_status != "expanded" and not include_failed:
        return persist_stage_failure(
            seed=seed,
            output_dir=output_batch_dir,
            index=index,
            status="rejected:expansion_input",
            reason=(
                f"input expansion status is {expansion_status or 'missing'}; use "
                "--include-failed to process it"
            ),
            extra={"input_expansion_artifact": str(artifact_path.resolve())},
        )

    expansion_artifact = artifact_from_expansion(
        expansion_data,
        run_context={},
    )
    if expansion_artifact.verifier is None or not expansion_artifact.constraints:
        return persist_stage_failure(
            seed=seed,
            output_dir=output_batch_dir,
            index=index,
            status="rejected:expansion_input",
            reason="expansion artifact lacks verifier or constraints",
            extra={"input_expansion_artifact": str(artifact_path.resolve())},
        )

    workflow = BrowseCompV2Workflow(config)
    constraints = expansion_artifact.constraints
    verifier = expansion_artifact.verifier
    local_report = next(
        (
            item.get("report")
            for item in reversed(expansion_data.get("iterations", []))
            if isinstance(item, dict)
            and item.get("stage") == "local_constraint"
            and isinstance(item.get("report"), dict)
        ),
        {"accepted": True, "source": "expansion_artifact"},
    )

    # Adapt only the input boundary.  The existing run_seed question,
    # uniqueness, repair, solver, and trajectory code remains the sole owner of
    # the downstream behavior.
    def reuse_constraints(_target, *, previous_constraints, failure_report, revision):
        return constraints

    def reuse_verifier(_artifact):
        return verifier

    def reuse_local(_artifact):
        return constraints, local_report

    workflow._make_constraints = reuse_constraints
    workflow._verify_program_constraints = reuse_verifier
    workflow._local_fuzzify = reuse_local
    output_path = workflow.run_seed_and_save(
        seed,
        output_dir=output_batch_dir,
        index=index,
        dry_run=dry_run,
    )
    payload = read_json(output_path)
    payload["stage"] = "question"
    payload["input_expansion_artifact"] = str(artifact_path.resolve())
    payload["expansion_status"] = expansion_status
    payload["expansion_iterations"] = expansion_data.get("iterations", [])
    write_json(output_path, payload)
    seed_folder = output_path.parent
    save_seed_record(seed, seed_folder / "seed.json")
    copy_json(artifact_path, seed_folder / "input_expansion_artifact.json")
    write_json(
        seed_folder / "stage_metadata.json",
        {
            "stage": "question",
            "input_expansion_artifact": str(artifact_path.resolve()),
            "artifact_path": str(output_path.resolve()),
            "status": payload.get("status"),
        },
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construct questions and run the existing repair/solver loop."
    )
    parser.add_argument("--env", default=".env_v5")
    parser.add_argument("--batch-dir", required=True, type=Path)
    parser.add_argument("--output-root", default="data/questions")
    parser.add_argument("--batch-id", default="")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--include-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be positive")
    input_batch = args.batch_dir.resolve()
    if not input_batch.is_dir():
        raise SystemExit(f"expansion batch does not exist: {input_batch}")
    batch_id = args.batch_id.strip() or input_batch.name
    output_batch = (PROJECT_ROOT / args.output_root / batch_id).resolve()
    output_batch.mkdir(parents=True, exist_ok=True)
    configure_stage_logging(output_batch, "question")
    config = replace(load_config(args.env), seed_verifier_enabled=False)
    if args.dry_run:
        config = replace(
            config,
            root_ambiguity_verifier_enabled=False,
            uniqueness_enabled=False,
            solver_enabled=False,
        )
    configure_stage_environment(output_batch)
    artifacts = sorted(input_batch.glob("seed_*/artifact.json"))
    if not artifacts:
        raise SystemExit(f"no expansion artifacts found under {input_batch}")
    print(
        f"question_stage_start: batch={batch_id} inputs={len(artifacts)} "
        f"concurrency={args.concurrency} out={output_batch}",
        flush=True,
    )
    paths: list[Path] = []
    with ThreadPoolExecutor(max_workers=min(args.concurrency, len(artifacts))) as pool:
        futures = {
            pool.submit(
                run_one,
                index=(
                    int(match.group(1))
                    if (match := re.match(r"seed_(\d+)_", artifact_path.parent.name))
                    else index
                ),
                artifact_path=artifact_path,
                output_batch_dir=output_batch,
                config=config,
                include_failed=args.include_failed,
                dry_run=args.dry_run,
            ): index
            for index, artifact_path in enumerate(artifacts, start=1)
        }
        for future in as_completed(futures):
            path = future.result()
            paths.append(path)
            print(f"question_seed_done: idx={futures[future]} artifact={path}", flush=True)
    statuses = {str(read_json(path).get("status")) for path in paths if path.exists()}
    write_json(
        output_batch / "manifest.json",
        {
            "batch_id": batch_id,
            "input_batch": str(input_batch),
            "artifact_count": len(paths),
            "statuses": sorted(statuses),
            "artifacts": [str(path.resolve()) for path in sorted(paths)],
        },
    )
    print(
        f"question_stage_done: batch={batch_id} artifacts={len(paths)} out={output_batch}",
        flush=True,
    )


if __name__ == "__main__":
    main()
