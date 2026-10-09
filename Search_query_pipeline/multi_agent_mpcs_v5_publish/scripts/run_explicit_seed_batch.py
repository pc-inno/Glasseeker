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
from browsecomp_v2.workflow import BrowseCompV2Workflow, load_seed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run existing generated seed files with their persisted indexes."
    )
    parser.add_argument("--env", default=".env_v2")
    parser.add_argument("--seed-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--indexes", required=True, nargs="+", type=int)
    args = parser.parse_args()

    workflow = BrowseCompV2Workflow(load_config(args.env))
    for index in args.indexes:
        matches = sorted(args.seed_dir.glob(f"seed_{index:03d}_*.json"))
        if len(matches) != 1:
            raise SystemExit(
                f"expected one seed file for index {index}, found {len(matches)}"
            )
        output = workflow.run_seed_and_save(
            load_seed(matches[0]),
            output_dir=args.output_dir,
            index=index,
        )
        artifact = json.loads(output.read_text(encoding="utf-8"))
        print(
            f"explicit_seed_done: idx={index} status={artifact.get('status')} "
            f"out={output}",
            flush=True,
        )


if __name__ == "__main__":
    main()
