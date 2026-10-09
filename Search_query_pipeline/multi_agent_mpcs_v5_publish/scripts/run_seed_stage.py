#!/usr/bin/env python3
"""Generate and validate seeds without starting tree expansion.

The script is an interface/persistence wrapper around the existing Seed Agent
and seed verifier.  It does not change any workflow prompt or validation rule.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from browsecomp_v2.config import load_config  # noqa: E402
from browsecomp_v2.diversity import choose_domain_slots  # noqa: E402
from browsecomp_v2.schema import SeedRecord  # noqa: E402
from browsecomp_v2.workflow import (  # noqa: E402
    BrowseCompV2Workflow,
    _target_with_verified_seed_sources,
)
from stage_common import (  # noqa: E402
    configure_stage_environment,
    configure_stage_logging,
    read_json,
    save_seed_record,
    seed_dir_name,
    write_json,
)


def identity_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def load_existing_state(batch_dir: Path) -> tuple[int, set[str], set[str], set[str], Counter[str]]:
    next_index = 1
    entity_ids: set[str] = set()
    names: set[str] = set()
    answers: set[str] = set()
    domains: Counter[str] = Counter()
    for fallback, seed_path in enumerate(sorted(batch_dir.glob("seed_*/seed.json")), start=1):
        match = re.match(r"seed_(\d+)_", seed_path.parent.name)
        next_index = max(next_index, (int(match.group(1)) + 1) if match else fallback + 1)
        try:
            seed = SeedRecord.from_dict(read_json(seed_path))
        except Exception:
            continue
        entity_ids.add(identity_key(seed.target.entity_id))
        names.add(identity_key(seed.target.name))
        answers.add(identity_key(seed.target.answer))
        status_path = seed_path.parent / "status.json"
        status = (
            str(read_json(status_path).get("status") or "")
            if status_path.exists()
            else ""
        )
        if status == "accepted":
            domain = str(seed.target.domain_family or "").strip()
            if domain:
                domains[domain] += 1
    return next_index, entity_ids, names, answers, domains


def diversity_request(
    domain_counts: Counter[str], requested: int, enabled: bool
) -> tuple[list[str], dict[str, int], bool]:
    if not enabled:
        return [], {}, False
    return (
        choose_domain_slots(domain_counts, requested),
        dict(domain_counts),
        True,
    )


def next_empty_streak(current: int, candidate_count: int) -> int:
    return current + 1 if candidate_count == 0 else 0


def verify_one(workflow: BrowseCompV2Workflow, seed: SeedRecord) -> tuple[SeedRecord, dict]:
    try:
        report = dict(workflow._verify_seed_fact(seed.target))
    except Exception as exc:
        report = {
            "accepted": False,
            "enabled": True,
            "error_type": type(exc).__name__,
            "reason": f"seed verification failed: {type(exc).__name__}: {exc}",
        }
    accepted = bool(report.get("accepted"))
    if accepted:
        enriched_target = _target_with_verified_seed_sources(seed.target, report)
        seed = SeedRecord(target=enriched_target, seed_note=seed.seed_note)
    return seed, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate/verify seeds into a reusable batch directory."
    )
    parser.add_argument("--env", default=".env_v5", help="Workflow env file.")
    parser.add_argument("--batch-id", required=True, help="Batch directory name.")
    parser.add_argument("--seed-root", default="data/seed_store")
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, choices=(2, 3), default=3)
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument(
        "--max-consecutive-empty-batches",
        type=int,
        default=3,
        help="Stop after this many consecutive post-filter empty batches.",
    )
    parser.add_argument(
        "--diversity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Direct requests with choose_domain_slots and apply the diversity "
            "filter. Use --no-diversity to disable both."
        ),
    )
    parser.add_argument("--verify-concurrency", type=int, default=3)
    parser.add_argument("--domain", default="auto")
    parser.add_argument("--target-type", default="auto")
    parser.add_argument("--answer-field", default="auto")
    parser.add_argument("--skip-verification", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (
        args.num_seeds < 1
        or args.max_batches < 1
        or args.max_consecutive_empty_batches < 1
        or args.verify_concurrency < 1
    ):
        raise SystemExit(
            "num-seeds, max-batches, max-consecutive-empty-batches and "
            "verify-concurrency must be positive"
        )

    batch_dir = (PROJECT_ROOT / args.seed_root / args.batch_id).resolve()
    if batch_dir.exists() and any(batch_dir.iterdir()) and not args.resume:
        raise SystemExit(f"batch directory is non-empty; use --resume: {batch_dir}")
    batch_dir.mkdir(parents=True, exist_ok=True)

    configure_stage_logging(batch_dir, "seed")
    config = load_config(args.env)
    configure_stage_environment(batch_dir)
    workflow = BrowseCompV2Workflow(config)
    next_index, seen_entities, seen_names, seen_answers, domain_counts = load_existing_state(batch_dir)
    accepted_count = sum(
        1
        for status_path in batch_dir.glob("seed_*/status.json")
        if status_path.exists() and read_json(status_path).get("status") == "accepted"
    )
    existing_manifest = (
        read_json(batch_dir / "manifest.json")
        if args.resume and (batch_dir / "manifest.json").exists()
        else {}
    )
    raw_count = int(existing_manifest.get("raw_generated") or 0)
    verified_count = int(existing_manifest.get("verified") or 0)
    batch_reports = [
        dict(item)
        for item in existing_manifest.get("batches", [])
        if isinstance(item, dict)
    ]
    previous_batch_id = max(
        (int(item.get("batch") or 0) for item in batch_reports),
        default=0,
    )
    consecutive_empty_batches = 0

    def persist_manifest() -> None:
        write_json(
            batch_dir / "manifest.json",
            {
                "batch_id": args.batch_id,
                "requested_seeds": args.num_seeds,
                "batch_size": args.batch_size,
                "max_batches_this_run": args.max_batches,
                "max_consecutive_empty_batches": args.max_consecutive_empty_batches,
                "diversity_enabled": args.diversity,
                "raw_generated": raw_count,
                "accepted": accepted_count,
                "verified": verified_count,
                "domain_counts": dict(domain_counts),
                "consecutive_empty_batches": consecutive_empty_batches,
                "batches": batch_reports,
            },
        )

    print(
        f"seed_stage_start: batch={args.batch_id} target={args.num_seeds} "
        f"batch_size={args.batch_size} seed_dir={batch_dir}",
        flush=True,
    )
    for batch_attempt in range(1, args.max_batches + 1):
        batch_id = previous_batch_id + batch_attempt
        remaining = max(0, args.num_seeds - accepted_count)
        if remaining == 0:
            break
        request_size = args.batch_size if remaining >= 2 else 2
        domain_slots, request_domain_counts, apply_diversity_filter = diversity_request(
            domain_counts,
            request_size,
            args.diversity,
        )
        print(
            f"seed_batch_start: batch={batch_id} attempt={batch_attempt}/{args.max_batches} "
            f"request={request_size} accepted={accepted_count}/{args.num_seeds} "
            f"diversity={args.diversity} domain_slots={domain_slots}",
            flush=True,
        )
        seeds = workflow.produce_seeds(
            domain=args.domain,
            target_type=args.target_type,
            answer_field=args.answer_field,
            num_seeds=request_size,
            avoid_entities=sorted(seen_entities | seen_names),
            avoid_answers=sorted(seen_answers),
            domain_slots=domain_slots,
            existing_domain_counts=request_domain_counts,
            apply_diversity_filter=apply_diversity_filter,
        )
        raw_count += len(seeds)
        unique: list[SeedRecord] = []
        batch_keys: set[tuple[str, str, str]] = set()
        for seed in seeds:
            key = (
                identity_key(seed.target.entity_id),
                identity_key(seed.target.name),
                identity_key(seed.target.answer),
            )
            if key in batch_keys or key[0] in seen_entities or key[1] in seen_names:
                continue
            batch_keys.add(key)
            unique.append(seed)
            seen_entities.add(key[0])
            seen_names.add(key[1])
            seen_answers.add(key[2])
        consecutive_empty_batches = next_empty_streak(
            consecutive_empty_batches,
            len(unique),
        )
        if not unique:
            batch_reports.append(
                {
                    "batch": batch_id,
                    "requested": request_size,
                    "received_after_source_and_diversity": len(seeds),
                    "unique": 0,
                    "accepted": 0,
                    "rejected": 0,
                    "diversity_enabled": args.diversity,
                    "domain_slots": domain_slots,
                    "empty_streak": consecutive_empty_batches,
                    "status": "empty_after_filter_or_dedup",
                }
            )
            persist_manifest()
            if consecutive_empty_batches >= args.max_consecutive_empty_batches:
                print(
                    f"seed_batch_stop: batch={batch_id} "
                    f"reason=consecutive_empty_batches "
                    f"streak={consecutive_empty_batches}/"
                    f"{args.max_consecutive_empty_batches}",
                    flush=True,
                )
                break
            print(
                f"seed_batch_continue: batch={batch_id} reason=empty_batch "
                f"streak={consecutive_empty_batches}/"
                f"{args.max_consecutive_empty_batches}",
                flush=True,
            )
            continue

        checked: list[tuple[SeedRecord, dict]] = []
        if args.skip_verification:
            checked = [(seed, {"accepted": None, "enabled": False, "reason": "verification skipped"}) for seed in unique]
        else:
            with ThreadPoolExecutor(max_workers=min(args.verify_concurrency, len(unique))) as pool:
                futures = [pool.submit(verify_one, workflow, seed) for seed in unique]
                for future in as_completed(futures):
                    checked.append(future.result())
        checked.sort(key=lambda item: item[0].target.entity_id)

        batch_accepted = 0
        batch_rejected = 0
        batch_paths: list[str] = []
        for seed, report in checked:
            status = "accepted" if report.get("accepted") is True else (
                "unverified" if report.get("accepted") is None else "rejected"
            )
            if status == "accepted":
                batch_accepted += 1
                accepted_count += 1
                domain = str(seed.target.domain_family or "").strip()
                if domain:
                    domain_counts[domain] += 1
            else:
                batch_rejected += 1
            folder = batch_dir / seed_dir_name(next_index, seed.target.entity_id)
            folder.mkdir(parents=True, exist_ok=True)
            save_seed_record(seed, folder / "seed.json")
            write_json(folder / "seed_verification.json", report)
            write_json(
                folder / "status.json",
                {
                    "status": status,
                    "batch_request": batch_id,
                    "seed_index": next_index,
                    "entity_id": seed.target.entity_id,
                    "source_urls": list(seed.target.source_urls),
                },
            )
            batch_paths.append(str(folder))
            next_index += 1
            verified_count += int(not args.skip_verification)
            print(
                f"seed_written: status={status} entity={seed.target.entity_id} path={folder}",
                flush=True,
            )
        batch_reports.append(
            {
                "batch": batch_id,
                "requested": request_size,
                "received_after_source_and_diversity": len(seeds),
                "unique": len(unique),
                "accepted": batch_accepted,
                "rejected": batch_rejected,
                "diversity_enabled": args.diversity,
                "domain_slots": domain_slots,
                "empty_streak": consecutive_empty_batches,
                "status": "completed",
                "paths": batch_paths,
            }
        )
        persist_manifest()
        if batch_accepted == 0 and not args.skip_verification:
            print(f"seed_batch_continue: batch={batch_id} reason=no_verified_seed", flush=True)

    print(
        f"seed_stage_done: batch={args.batch_id} raw={raw_count} "
        f"accepted={accepted_count} verified_calls={verified_count} "
        f"empty_streak={consecutive_empty_batches} out={batch_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
