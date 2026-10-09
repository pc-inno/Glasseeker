#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from concurrent.futures import ALL_COMPLETED, FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field, replace
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from browsecomp_v2.config import load_config
from browsecomp_v2.diversity import DOMAIN_FAMILIES, choose_domain_slots, target_domain
from browsecomp_v2.schema import SeedRecord
from browsecomp_v2.workflow import (
    BrowseCompV2Workflow,
    filter_diverse_seeds,
    load_seed,
    save_artifact,
    save_seed,
)


@dataclass
class SeedInventory:
    entity_ids: set[str] = field(default_factory=set)
    names: set[str] = field(default_factory=set)
    answers: set[str] = field(default_factory=set)
    domain_counts: Counter[str] = field(default_factory=Counter)


def _identity_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"[\W_]+", " ", normalized, flags=re.UNICODE).strip()


def _register_seed(inventory: SeedInventory, seed: SeedRecord) -> bool:
    entity_id = _identity_key(seed.target.entity_id)
    name = _identity_key(seed.target.name)
    answer = _identity_key(seed.target.answer)
    if (
        (entity_id and entity_id in inventory.entity_ids)
        or (name and name in inventory.names)
        or (answer and answer in inventory.answers)
    ):
        return False
    if entity_id:
        inventory.entity_ids.add(entity_id)
    if name:
        inventory.names.add(name)
    if answer:
        inventory.answers.add(answer)
    domain = target_domain(seed.target)
    if domain != "unknown":
        inventory.domain_counts[domain] += 1
    return True


def _load_manifest_answers(path: Path) -> tuple[set[str], list[str]]:
    keys: set[str] = set()
    values: list[str] = []
    if not path.exists():
        return keys, values
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            value = str(json.loads(raw).get("answer") or "").strip()
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid manifest JSON at line {line_number}: {exc}") from exc
        key = _identity_key(value)
        if key and key not in keys:
            keys.add(key)
            values.append(value)
    return keys, values


def _next_seed_index(seed_dir: Path) -> int:
    indexes = []
    for path in seed_dir.glob("seed_*.json"):
        match = re.match(r"seed_(\d+)_", path.name)
        if match:
            indexes.append(int(match.group(1)))
    return max(indexes, default=0) + 1


def _seed_index_from_path(path: Path, fallback: int) -> int:
    match = re.match(r"seed_(\d+)_", path.name)
    return int(match.group(1)) if match else fallback


def _artifact_is_effective(artifact: dict) -> bool:
    """User's quality standard: solver (deepseek-v4-pro) gets <= 1 of 3 correct."""
    summary = artifact.get("solver_summary")
    if (
        isinstance(summary, dict)
        and summary.get("status") in {"accepted:hard", "review:all_wrong"}
        and summary.get("correct") is not None
    ):
        try:
            return int(summary["correct"]) <= 1
        except (TypeError, ValueError):
            return False
    return False


def _scan_completed_artifacts(out_dir: Path) -> dict[str, dict]:
    """Resume mode (policy B): scan out_dir for finished artifacts.

    Returns {entity_id_key: {"status", "effective", "path"}} for every artifact
    whose workflow completed with a usable evaluation. Artifacts with a
    *workflow_error* or *verification_failed* status are treated as NOT
    completed so transient execution/judging failures get retried on resume.
    """
    completed: dict[str, dict] = {}
    if not out_dir.exists():
        return completed
    paths = [*out_dir.glob("seed_*/artifact.json"), *out_dir.glob("seed_*.json")]
    for path in sorted(paths):
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        status = str(artifact.get("status") or "")
        if "workflow_error" in status or status == "verification_failed":
            # Crashed or incompletely judged run — leave it for retry.
            continue
        if not status.startswith(("accepted:", "rejected:", "review:")):
            # Checkpoints such as status=created are resumable partial runs.
            continue
        target = artifact.get("target") or {}
        entity_id = target.get("entity_id") if isinstance(target, dict) else None
        key = _identity_key(entity_id or "")
        if not key:
            continue
        completed[key] = {
            "status": status,
            "effective": _artifact_is_effective(artifact),
            "path": str(path),
        }
    return completed


def _print_diversity(inventory: SeedInventory, *, target: int) -> None:
    covered = len(inventory.domain_counts)
    missing = [domain for domain in DOMAIN_FAMILIES if inventory.domain_counts.get(domain, 0) == 0]
    print(
        "domain_coverage: "
        f"covered={covered}/{target} seeds={len(inventory.entity_ids)} "
        f"missing={missing[:max(0, target - covered)]}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=".env", help="Path to env file.")
    parser.add_argument(
        "--run-id",
        default="",
        help="One version label used to derive runs, seeds, and conversation directories.",
    )
    parser.add_argument("--seed", default="", help="Run one seed JSON.")
    parser.add_argument("--seed-dir", default="", help="Run all seed JSON files in a directory.")
    parser.add_argument("--auto-seed", action="store_true", help="Call Seed Agent to generate seeds.")
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--domain", default="auto")
    parser.add_argument("--target-type", default="auto")
    parser.add_argument("--answer-field", default="auto")
    parser.add_argument("--generated-seed-dir", default="")
    parser.add_argument("--out-dir", default="", help="Artifact output directory.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug-hermes", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip seeds already completed in --out-dir "
            "(retry workflow_error and verification_failed artifacts)."
        ),
    )
    args = parser.parse_args()

    config = load_config(args.env)
    run_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", args.run_id.strip()).strip("_")
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif run_id:
        out_dir = Path("data") / f"runs_{run_id}"
    else:
        out_dir = config.output_dir
    if args.generated_seed_dir:
        generated_seed_dir = Path(args.generated_seed_dir)
    elif run_id:
        generated_seed_dir = Path("data/seeds") / f"generated_{run_id}"
    else:
        generated_seed_dir = Path("data/seeds/generated")
    os.environ["V2_OUTPUT_DIR"] = str(out_dir)
    os.environ["V2_CONVERSATION_LOG_DIR"] = str(out_dir / "conversations")
    config = replace(config, output_dir=out_dir)
    workflow = BrowseCompV2Workflow(config)
    if args.debug_hermes:
        os.environ["V2_HERMES_DEBUG"] = "1"

    if args.auto_seed:
        seed_dir = generated_seed_dir
        seed_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = PROJECT_ROOT / "data/filter_data/uniqueness_manifest.jsonl"
        manifest_answer_keys, manifest_answer_values = _load_manifest_answers(manifest_path)
        inventory = SeedInventory()
        for existing_seed_path in sorted(seed_dir.glob("seed_*.json")):
            try:
                _register_seed(inventory, load_seed(existing_seed_path))
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
        seen_entity_ids, seen_names = inventory.entity_ids, inventory.names
        seen_answers = inventory.answers
        min_domain_coverage = max(1, int(os.environ.get("V2_MIN_DOMAIN_COVERAGE", "30")))

        verbose = bool(config.verbose_progress)
        use_bar = (tqdm is not None) and not verbose

        def emit(message: str) -> None:
            """Detail log line: printed only in verbose mode. Otherwise the
            progress bar carries the state instead of flooding stdout."""
            if verbose:
                print(message, flush=True)

        # ---- Resume (policy B): skip already-completed artifacts, retry crashes ----
        resume_done = 0
        resume_effective = 0
        if args.resume:
            completed = _scan_completed_artifacts(out_dir)
            for key, info in completed.items():
                seen_entity_ids.add(key)  # block re-generation of this target
            resume_done = len(completed)
            resume_effective = sum(1 for info in completed.values() if info["effective"])
            print(
                "resume_scan: "
                f"out_dir={out_dir} completed={resume_done} "
                f"effective={resume_effective} "
                "(retrying workflow_error and verification_failed artifacts)",
                flush=True,
            )

        _print_diversity(inventory, target=min_domain_coverage)
        print(
            "manifest_answers_loaded: "
            f"answers={len(manifest_answer_keys)} manifest={manifest_path} "
            f"seed_dir={seed_dir} out_dir={out_dir}",
            flush=True,
        )
        all_paths: list[Path] = []
        inflight: dict[Future[Path], tuple[int, SeedRecord]] = {}
        queued_total = 0
        seed_workers = max(1, config.seed_concurrency)
        seed_batch_size = max(1, int(os.environ.get("V2_SEED_BATCH_SIZE", "1")))
        seed_prefetch = max(
            seed_workers,
            int(os.environ.get("V2_SEED_PREFETCH", str(seed_workers * 2))),
        )
        global_idx = _next_seed_index(seed_dir) - 1

        # Progress bar counters. Target = seeds still needed this run.
        counters = {"done": 0, "effective": 0, "err": 0}
        bar = None
        if use_bar and tqdm is not None:
            bar = tqdm(
                total=args.num_seeds,
                desc="seeds",
                unit="seed",
                dynamic_ncols=True,
            )
            bar.set_postfix(effective=0, done=0, err=0)

        def _refresh_postfix() -> None:
            if bar is not None:
                bar.set_postfix(
                    effective=counters["effective"],
                    done=counters["done"],
                    err=counters["err"],
                )

        def collect_completed(*, final: bool = False) -> None:
            if not inflight:
                return
            done, _ = wait(
                tuple(inflight),
                return_when=ALL_COMPLETED if final else FIRST_COMPLETED,
            )
            for future in done:
                index, seed = inflight.pop(future)
                try:
                    out_path = future.result()
                    all_paths.append(out_path)
                    counters["done"] += 1
                    # Read back the artifact to classify effectiveness for the bar.
                    try:
                        artifact = json.loads(Path(out_path).read_text(encoding="utf-8"))
                        status = str(artifact.get("status") or "")
                        if "workflow_error" in status or status == "verification_failed":
                            counters["err"] += 1
                        elif _artifact_is_effective(artifact):
                            counters["effective"] += 1
                    except (json.JSONDecodeError, OSError):
                        pass
                    if bar is not None:
                        bar.update(1)
                        _refresh_postfix()
                except Exception as exc:
                    counters["err"] += 1
                    if bar is not None:
                        bar.update(1)
                        _refresh_postfix()
                    else:
                        print(
                            "seed_pipeline_worker_error: "
                            f"idx={index} target={seed.target.entity_id} "
                            f"error={type(exc).__name__}: {exc}",
                            flush=True,
                        )

        emit(
            "seed_pipeline_start: "
            f"workers={seed_workers} prefetch={seed_prefetch} "
            f"batch_size={seed_batch_size}"
        )
        with ThreadPoolExecutor(max_workers=seed_workers) as executor:
            for round_id in range(1, config.max_rounds + 1):
                remaining_total = args.num_seeds - queued_total
                if remaining_total <= 0:
                    break
                emit(f"seed_round_start: round={round_id}/{config.max_rounds}")
                seeds: list[SeedRecord] = []
                batches_needed = (
                    remaining_total + seed_batch_size - 1
                ) // seed_batch_size
                max_attempts = max(4, batches_needed * 4)
                seed_attempt = 0
                while len(seeds) < remaining_total and seed_attempt < max_attempts:
                    while len(inflight) >= seed_prefetch:
                        collect_completed()
                    seed_attempt += 1
                    requested = min(seed_batch_size, remaining_total - len(seeds))
                    domain_slots = choose_domain_slots(inventory.domain_counts, requested)
                    emit(
                        "seed_attempt_start: "
                        f"round={round_id} attempt={seed_attempt}/{max_attempts} "
                        f"selected={len(seeds)}/{args.num_seeds} "
                        f"inflight={len(inflight)}/{seed_prefetch}"
                    )
                    try:
                        batch = workflow.produce_seeds(
                            domain=args.domain,
                            target_type=args.target_type,
                            answer_field=args.answer_field,
                            num_seeds=requested,
                            avoid_answers=manifest_answer_values,
                            domain_slots=domain_slots,
                            existing_domain_counts=dict(inventory.domain_counts),
                            overused_source_domains=(),
                        )
                    except Exception as exc:
                        emit(
                            "seed_attempt_error: "
                            f"round={round_id} attempt={seed_attempt}/{max_attempts} "
                            f"error={type(exc).__name__}: {exc}"
                        )
                        continue
                    candidates = []
                    batch_entity_ids: set[str] = set()
                    batch_names: set[str] = set()
                    batch_answers: set[str] = set()
                    for seed in batch:
                        entity_id_key = _identity_key(seed.target.entity_id)
                        name_key = _identity_key(seed.target.name)
                        answer_key = _identity_key(seed.target.answer)
                        if (
                            (answer_key and answer_key in manifest_answer_keys)
                            or (answer_key and answer_key in seen_answers)
                            or (entity_id_key and entity_id_key in seen_entity_ids)
                            or (name_key and name_key in seen_names)
                            or (entity_id_key and entity_id_key in batch_entity_ids)
                            or (name_key and name_key in batch_names)
                            or (answer_key and answer_key in batch_answers)
                        ):
                            continue
                        candidates.append(seed)
                        if entity_id_key:
                            batch_entity_ids.add(entity_id_key)
                        if name_key:
                            batch_names.add(name_key)
                        if answer_key:
                            batch_answers.add(answer_key)
                    duplicate_count = len(batch) - len(candidates)
                    candidates = filter_diverse_seeds(
                        candidates,
                        requested=requested,
                        existing_domain_counts=dict(inventory.domain_counts),
                        preferred_domains=domain_slots,
                        overused_source_domains=(),
                    )
                    accepted = candidates[:requested]
                    for seed in accepted:
                        if not _register_seed(inventory, seed):
                            continue
                        global_idx += 1
                        seeds.append(seed)
                        queued_total += 1
                        seed_path = seed_dir / f"seed_{global_idx:03d}_{seed.target.entity_id}.json"
                        save_seed(seed, seed_path)
                        future = executor.submit(
                            workflow.run_seed_and_save,
                            seed,
                            output_dir=out_dir,
                            index=global_idx,
                            dry_run=args.dry_run,
                        )
                        inflight[future] = (global_idx, seed)
                        emit(
                            f"seed_queued: idx={global_idx} round={round_id} "
                            f"inflight={len(inflight)} seed={seed_path}"
                        )
                    emit(
                        "seed_topup: "
                        f"round={round_id} attempt={seed_attempt} "
                        f"received={len(batch)} duplicates={duplicate_count} "
                        f"accepted={len(accepted)} "
                        f"selected_this_round={len(seeds)}/{remaining_total} "
                        f"queued_total={queued_total}/{args.num_seeds}"
                    )
                if not seeds:
                    emit(f"seed_round_stop: round={round_id} reason=no_seeds")
                    break
                _print_diversity(inventory, target=min_domain_coverage)
                emit(
                    f"seed_round_queued: round={round_id} queued={len(seeds)} "
                    f"queued_total={queued_total}/{args.num_seeds} "
                    f"inflight={len(inflight)} completed={len(all_paths)}"
                )
            collect_completed(final=True)
        if bar is not None:
            bar.close()
        total_effective = counters["effective"] + resume_effective
        total_done = counters["done"] + resume_done
        print(
            f"workflow_done: artifacts={len(all_paths)} out_dir={out_dir} "
            f"done={total_done} effective={total_effective} err={counters['err']}"
            + (f" (resumed {resume_done})" if args.resume else ""),
            flush=True,
        )
        return
    seed_indexes: list[int] = []
    if args.seed_dir:
        seed_paths = sorted(Path(args.seed_dir).glob("*.json"))
        seeds = [load_seed(path) for path in seed_paths]
        seed_indexes = [
            _seed_index_from_path(path, fallback)
            for fallback, path in enumerate(seed_paths, start=1)
        ]
    elif args.seed:
        seed_path = Path(args.seed)
        seeds = [load_seed(seed_path)]
        seed_indexes = [_seed_index_from_path(seed_path, 1)]
    else:
        fallback = PROJECT_ROOT / "data/seeds/sample_seed.json"
        if not fallback.exists():
            raise SystemExit("provide --seed, --seed-dir, or --auto-seed")
        seeds = [load_seed(fallback)]

    if len(seeds) == 1:
        out_path = workflow.run_seed_and_save(
            seeds[0],
            output_dir=out_dir,
            index=seed_indexes[0] if seed_indexes else 1,
            dry_run=args.dry_run,
        )
        artifact = json.loads(out_path.read_text(encoding="utf-8"))
        print(
            f"workflow_done: status={artifact.get('status')} out={out_path}",
            flush=True,
        )
        return

    paths = workflow.run_seeds_parallel(
        seeds,
        output_dir=out_dir,
        dry_run=args.dry_run,
        indexes=seed_indexes or None,
    )
    print(f"workflow_done: artifacts={len(paths)} out_dir={out_dir}", flush=True)


if __name__ == "__main__":
    main()
