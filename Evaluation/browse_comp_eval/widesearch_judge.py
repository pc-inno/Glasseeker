"""WideSearch runner using ByteDance-Seed's structural evaluation protocol."""

from __future__ import annotations

import concurrent.futures
import json
import threading
from dataclasses import asdict
from statistics import mean
from typing import Any

from .widesearch_scorer import configure_completion, evaluate_from_eval_spec

METRICS = (
    "score", "precision_by_row", "recall_by_row", "f1_by_row",
    "precision_by_item", "recall_by_item", "f1_by_item",
)


class WideSearchJudge:
    def __init__(self, client: Any | None, config: Any):
        self.client = client
        self.config = config
        self._summary_lock = threading.Lock()
        configure_completion(client.complete if client else None)

    def run(self) -> dict[str, Any]:
        from .judge import load_judge_results, load_prediction_records

        for path in (
            self.config.output_dir,
            self.config.summary_path.parent,
            self.config.metrics_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)
        records = load_prediction_records(self.config.conv_dir)
        pending = [record for record in records if self._should_run(record)]
        print(f"loaded predictions: {len(records)}")
        print(f"pending judge records: {len(pending)}")
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.num_workers
        ) as executor:
            futures = {
                executor.submit(self._judge_one, record): record
                for record in pending
            }
            for index, future in enumerate(
                concurrent.futures.as_completed(futures), start=1
            ):
                record = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = self._failed(record, exc)
                    self._write(record, result)
                print(
                    f"[{index}/{len(pending)}] {record.run_id}: "
                    f"{result.get('status')} sr={result.get('score')} "
                    f"row_f1={result.get('f1_by_row')} "
                    f"item_f1={result.get('f1_by_item')}"
                )
        metrics = compute_metrics(load_judge_results(self.config.output_dir))
        self.config.metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return metrics

    def _should_run(self, record: Any) -> bool:
        return self.config.force or not self._path(record).exists()

    def _judge_one(self, record: Any) -> dict[str, Any]:
        spec = parse_eval_spec(record.answer, record.question_id)
        result = self._base(record)
        if record.pred_status == "policy_violation":
            result.update(zero("scored_policy_violation", "policy violation"))
        elif record.pred_status != "success" and not self.config.include_failed:
            result.update(zero("scored_infer_failed", record.pred_status))
        elif not record.pred.strip():
            result.update(zero("scored_empty_prediction", "empty prediction"))
        else:
            if not self.client:
                raise RuntimeError("WideSearch judge client is not configured")
            scored = evaluate_from_eval_spec(record.pred, spec)
            result.update(asdict(scored))
            result["status"] = "judged"
            result["judge_reason"] = scored.msg
        self._write(record, result)
        return result

    def _base(self, record: Any) -> dict[str, Any]:
        return {
            "question_id": record.question_id,
            "run_id": record.run_id,
            "repeat_index": record.repeat_index,
            "type": record.type,
            "pred_status": record.pred_status,
            "question": record.question,
            "prediction": record.pred,
            "pred_path": str(record.path),
            "judge_template": "widesearch",
        }

    def _failed(self, record: Any, exc: Exception) -> dict[str, Any]:
        result = self._base(record)
        result.update({"status": "judge_failed", "error": str(exc)})
        result.update({metric: None for metric in METRICS})
        return result

    def _write(self, record: Any, result: dict[str, Any]) -> None:
        path = self._path(record)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        summary = {
            key: result.get(key)
            for key in (
                "question_id", "run_id", "repeat_index", "type", "status",
                "pred_status", *METRICS, "judge_reason", "error",
            )
        }
        summary["pred_path"] = result.get("pred_path")
        summary["result_path"] = str(path)
        with self._summary_lock:
            with self.config.summary_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(summary, ensure_ascii=False) + "\n")

    def _path(self, record: Any):
        return self.config.output_dir / record.question_id / f"{record.run_id}.json"


def parse_eval_spec(answer: Any, question_id: str) -> dict[str, Any]:
    try:
        spec = json.loads(answer) if isinstance(answer, str) else answer
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"{question_id}: invalid WideSearch eval spec") from exc
    required = {"required", "unique_columns", "eval_pipeline", "gold_table"}
    if not isinstance(spec, dict) or not required.issubset(spec):
        raise ValueError(f"{question_id}: incomplete WideSearch eval spec")
    if str(spec.get("_instance_id") or question_id) != question_id:
        raise ValueError(f"{question_id}: eval spec instance_id mismatch")
    return spec


def zero(status: str, reason: str) -> dict[str, Any]:
    return {
        "status": status,
        "judge_reason": f"{reason}; counted as zero",
        **{metric: 0.0 for metric in METRICS},
    }


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    repeats: set[int] = set()
    statuses: dict[str, int] = {}
    for result in results:
        groups.setdefault(str(result.get("question_id") or "unknown"), []).append(result)
        repeats.add(repeat_index(result))
        status = str(result.get("status") or "unknown")
        statuses[status] = statuses.get(status, 0) + 1

    expected = len(repeats)
    aggregate: dict[str, dict[str, float | None]] = {}
    incomplete: set[str] = set()
    for metric in METRICS:
        averages, maxima, minima = [], [], []
        for question_id, group in groups.items():
            values = numeric_values(group, metric)
            if len(values) != expected:
                incomplete.add(question_id)
                continue
            averages.append(mean(values))
            maxima.append(max(values))
            minima.append(min(values))
        aggregate[metric] = {
            "avg_n": rounded_mean(averages),
            "max_n": rounded_mean(maxima),
            "min_n": rounded_mean(minima),
        }

    per_repeat = []
    for index in sorted(repeats):
        group = [result for result in results if repeat_index(result) == index]
        per_repeat.append({
            "repeat_index": index,
            "total": len(group),
            **{metric: rounded_mean(numeric_values(group, metric)) for metric in METRICS},
        })

    by_type = {}
    for item_type in sorted({str(item.get("type") or "unknown") for item in results}):
        group = [item for item in results if str(item.get("type") or "unknown") == item_type]
        by_type[item_type] = {
            "total": len(group),
            "score": rounded_mean(numeric_values(group, "score")),
            "f1_by_row": rounded_mean(numeric_values(group, "f1_by_row")),
            "f1_by_item": rounded_mean(numeric_values(group, "f1_by_item")),
        }

    failed = statuses.get("judge_failed", 0)
    return {
        "prediction_total": len(results),
        "question_total": len(groups),
        "repeat_count": expected,
        "complete": not incomplete and failed == 0,
        "incomplete_question_count": len(incomplete),
        "incomplete_questions": sorted(incomplete),
        "judge_failed": failed,
        "status_counts": statuses,
        **aggregate,
        "sr_avg_n": aggregate["score"]["avg_n"],
        "sr_pass_n": aggregate["score"]["max_n"],
        "row_f1_avg_n": aggregate["f1_by_row"]["avg_n"],
        "row_f1_max_n": aggregate["f1_by_row"]["max_n"],
        "item_f1_avg_n": aggregate["f1_by_item"]["avg_n"],
        "item_f1_max_n": aggregate["f1_by_item"]["max_n"],
        "per_repeat": per_repeat,
        "by_type": by_type,
    }


def repeat_index(result: dict[str, Any]) -> int:
    try:
        value = int(result.get("repeat_index"))
    except (TypeError, ValueError):
        value = 1
    return max(1, value)


def numeric_values(results: list[dict[str, Any]], metric: str) -> list[float]:
    return [
        float(result[metric])
        for result in results
        if isinstance(result.get(metric), (int, float))
    ]


def rounded_mean(values: list[float]) -> float | None:
    return round(mean(values), 6) if values else None
