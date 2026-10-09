#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from browsecomp_v2.config import load_config
from browsecomp_v2.workflow import (
    BrowseCompV2Workflow,
    filter_diverse_seeds,
    load_seed,
    save_seed,
)


def existing_seed_state(seed_dir: Path) -> tuple[int, set[str], set[str]]:
    max_index = 0
    entity_ids: set[str] = set()
    names: set[str] = set()
    for path in sorted(seed_dir.glob("seed_*.json")):
        match = re.match(r"seed_(\d+)_", path.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
        try:
            seed = load_seed(path)
        except Exception as exc:
            print(f"skip_bad_seed: path={path} error={exc}", flush=True)
            continue
        entity_ids.add(seed.target.entity_id)
        names.add(seed.target.name)
    return max_index, entity_ids, names


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-").lower()
    return slug or "seed"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate BrowseComp V2 seeds and optionally run workflow.")
    parser.add_argument("--env", default=".env", help="Path to env file.")
    parser.add_argument("--num-seeds", type=int, default=8, help="Seeds to keep per round.")
    parser.add_argument("--rounds", type=int, default=1, help="Auto-seed rounds.")
    parser.add_argument("--attempts", type=int, default=4, help="Top-up attempts per round.")
    parser.add_argument("--domain", default="auto")
    parser.add_argument("--target-type", default="auto")
    parser.add_argument("--answer-field", default="auto")
    parser.add_argument("--generated-seed-dir", default="data/seeds/generated")
    parser.add_argument("--out-dir", default="data/runs_auto")
    parser.add_argument("--seed-only", action="store_true", help="Only write generated seed JSON files.")
    parser.add_argument("--dry-run", action="store_true", help="Dry-run workflow after seeds are generated.")
    args = parser.parse_args()

    if args.num_seeds < 1:
        raise SystemExit("--num-seeds must be >= 1")
    if args.rounds < 1:
        raise SystemExit("--rounds must be >= 1")
    if args.attempts < 1:
        raise SystemExit("--attempts must be >= 1")

    os.environ["V2_MAX_ROUNDS"] = str(args.rounds)
    config = load_config(args.env)
    workflow = BrowseCompV2Workflow(config)

    seed_dir = (PROJECT_ROOT / args.generated_seed_dir).resolve()
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    seed_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    next_index, _, _ = existing_seed_state(seed_dir)
    _, seen_entity_ids, avoid_names = existing_seed_state(out_dir)
    all_artifacts: list[Path] = []

    print(
        "auto_seed_start: "
        f"rounds={args.rounds} num_seeds={args.num_seeds} "
        f"seed_dir={seed_dir} out_dir={out_dir} seed_only={args.seed_only} "
        f"existing_targets={len(seen_entity_ids)}",
        flush=True,
    )

    for round_id in range(1, args.rounds + 1):
        print(f"seed_round_start: round={round_id}/{args.rounds}", flush=True)
        selected = []
        for attempt in range(1, args.attempts + 1):
            if len(selected) >= args.num_seeds:
                break
            need = args.num_seeds - len(selected)
            avoid = sorted(avoid_names | {seed.target.name for seed in selected})
            print(
                "seed_attempt_start: "
                f"round={round_id} attempt={attempt}/{args.attempts} "
                f"selected={len(selected)}/{args.num_seeds} need={need} avoid={len(avoid)}",
                flush=True,
            )
            batch = workflow.produce_seeds(
                domain=args.domain,
                target_type=args.target_type,
                answer_field=args.answer_field,
                num_seeds=max(args.num_seeds, need),
                avoid_entities=avoid,
            )
            for seed in batch:
                if seed.target.entity_id in seen_entity_ids:
                    continue
                if seed.target.entity_id in {item.target.entity_id for item in selected}:
                    continue
                selected.append(seed)
            selected = filter_diverse_seeds(selected, requested=args.num_seeds)
            print(
                "seed_attempt_done: "
                f"round={round_id} attempt={attempt} received={len(batch)} "
                f"selected={len(selected)}/{args.num_seeds}",
                flush=True,
            )

        selected = selected[: args.num_seeds]
        if not selected:
            print(f"seed_round_stop: round={round_id} reason=no_new_seeds", flush=True)
            break

        for seed in selected:
            next_index += 1
            seed_path = seed_dir / f"seed_{next_index:03d}_{safe_slug(seed.target.entity_id)}.json"
            save_seed(seed, seed_path)
            seen_entity_ids.add(seed.target.entity_id)
            avoid_names.add(seed.target.name)
            print(f"seed_written: {seed_path}", flush=True)

        if args.seed_only:
            continue

        paths = workflow.run_seeds_parallel(selected, output_dir=out_dir, dry_run=args.dry_run)
        all_artifacts.extend(paths)
        print(
            f"seed_round_done: round={round_id} seeds={len(selected)} artifacts={len(paths)}",
            flush=True,
        )

    print(
        "auto_seed_done: "
        f"seed_dir={seed_dir} out_dir={out_dir} artifacts={len(all_artifacts)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
