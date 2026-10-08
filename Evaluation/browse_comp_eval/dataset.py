from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class DatasetItem:
    question_id: str
    question: str
    answer: str
    type: str
    source_line: int


@dataclass(frozen=True)
class RunSpec:
    question_id: str
    run_id: str
    repeat_index: int
    item: DatasetItem


def safe_id(value: str) -> str:
    cleaned = _SAFE_ID_RE.sub("_", str(value).strip()).strip("._-")
    return cleaned or "unknown"


def _coerce_item(raw: dict, source_line: int) -> DatasetItem:
    is_miro_gaia = "task_question" in raw or "ground_truth" in raw or "task_id" in raw
    if is_miro_gaia:
        missing = [
            key
            for key in ("task_id", "task_question", "ground_truth")
            if key not in raw
        ]
        if missing:
            raise ValueError(
                f"line {source_line}: incomplete Miro GAIA row; missing field(s): "
                f"{', '.join(missing)}"
            )
        question_value = raw["task_question"]
        answer_value = raw["ground_truth"]
        id_value = raw["task_id"]
        metadata = raw.get("metadata")
        level = metadata.get("level") if isinstance(metadata, dict) else None
        type_value = f"gaia_validation_level_{level}" if level is not None else "gaia_validation"
    else:
        missing = [key for key in ("question", "answer", "type") if key not in raw]
        if missing:
            raise ValueError(f"line {source_line}: missing required field(s): {', '.join(missing)}")
        question_value = raw["question"]
        answer_value = raw["answer"]
        id_value = raw.get("id")
        type_value = raw["type"]

    question = str(question_value).strip()
    if not question:
        raise ValueError(f"line {source_line}: question must not be empty")

    if id_value is not None and str(id_value).strip():
        question_id = safe_id(str(id_value))
    else:
        question_id = f"q{source_line:06d}"

    return DatasetItem(
        question_id=question_id,
        question=question,
        answer=str(answer_value),
        type=str(type_value),
        source_line=source_line,
    )


def load_dataset(path: str | Path) -> List[DatasetItem]:
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(f"data file not found: {data_path}")

    if data_path.suffix.lower() == ".json":
        payload = json.loads(data_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"{data_path} must contain a JSON array")
        return [_coerce_item(row, idx) for idx, row in enumerate(payload, start=1)]

    items: list[DatasetItem] = []
    with data_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"line {idx}: each JSONL row must be an object")
            items.append(_coerce_item(raw, idx))
    return items


def expand_repeats(items: Iterable[DatasetItem], repeats: int) -> list[RunSpec]:
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    runs: list[RunSpec] = []
    for item in items:
        for repeat_index in range(1, repeats + 1):
            run_id = f"{item.question_id}__r{repeat_index}"
            runs.append(
                RunSpec(
                    question_id=item.question_id,
                    run_id=run_id,
                    repeat_index=repeat_index,
                    item=item,
                )
            )
    return runs


def expand_repeat(items: Iterable[DatasetItem], repeat_index: int) -> list[RunSpec]:
    if repeat_index < 1:
        raise ValueError("repeat_index must be >= 1")

    runs: list[RunSpec] = []
    for item in items:
        run_id = f"{item.question_id}__r{repeat_index}"
        runs.append(
            RunSpec(
                question_id=item.question_id,
                run_id=run_id,
                repeat_index=repeat_index,
                item=item,
            )
        )
    return runs
