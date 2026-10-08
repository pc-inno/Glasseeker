from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


INCORRECT_SCORE = 0.0
CORRECT_SCORE = 1.0
BROWSECOMP_HERMES_PROMPT_VERSION = "browsecomp-hermes-v1"
BROWSECOMP_OFFICIAL_PROMPT_VERSION = "browsecomp-official-v1"
BROWSECOMP_PROMPT_VERSIONS = {
    "hermes": BROWSECOMP_HERMES_PROMPT_VERSION,
    "official": BROWSECOMP_OFFICIAL_PROMPT_VERSION,
    BROWSECOMP_HERMES_PROMPT_VERSION: BROWSECOMP_HERMES_PROMPT_VERSION,
    BROWSECOMP_OFFICIAL_PROMPT_VERSION: BROWSECOMP_OFFICIAL_PROMPT_VERSION,
}


BROWSECOMP_ANSWER_EQUIVALENCE_GUIDANCE = """
Additional answer-equivalence guidance:
- First inspect the full response for identity-disambiguating details such as dates, creators, locations, versions, model years, or vessel types. Use those details only to determine which entity the response names; do not re-solve or fact-check the question.
- Treat a missing or added generic honorific or type/designation prefix (for example, "SS", "MS"/"MV", "HMS", "Dr.", or "St.") as non-meaningful only when the response identifies the same entity and contains no conflicting identity detail.
- A shared base name is not sufficient when the response clearly identifies a different same-named entity. For example, omitting "SS" can be acceptable for the same vessel, but not when the response describes a different same-named motor or container ship.
""".strip()


@dataclass(frozen=True)
class JudgeClientConfig:
    api_key: str
    base_url: str
    model: str
    provider: str
    api_version: str
    timeout_seconds: float
    max_retries: int
    max_tokens: int
    temperature: float
    top_p: float
    token_param: str
    use_json_schema: bool
    judge_template: str
    judge_prompt_version: str = BROWSECOMP_OFFICIAL_PROMPT_VERSION


@dataclass(frozen=True)
class JudgeRunConfig:
    conv_dir: Path
    output_dir: Path
    summary_path: Path
    metrics_path: Path
    num_workers: int
    force: bool
    include_failed: bool
    resume_from_conv_dir: Path | None = None
    resume_from_output_dir: Path | None = None
    judge_template: str = "browsecomp"
    judge_prompt_version: str = BROWSECOMP_OFFICIAL_PROMPT_VERSION


@dataclass(frozen=True)
class PredictionRecord:
    path: Path
    question_id: str
    run_id: str
    repeat_index: int | None
    question: str
    answer: Any
    pred: str
    pred_status: str
    type: str
    raw: dict[str, Any]


class JudgeClient:
    def __init__(self, config: JudgeClientConfig):
        self.config = config
        self._sdk_client: Any | None = None

    def judge(self, record: PredictionRecord) -> dict[str, Any]:
        if self.config.judge_template == "xbench-deepsearch-2510":
            extracted = extract_xbench_final_answer(record.pred)
            if extracted is not None and extracted == format_ground_truth(record.answer):
                return {
                    "score": CORRECT_SCORE,
                    "judge_reason": "答案完全正确, 无需调用LLM Judge",
                    "judge_raw_response": "",
                    "judge_response": None,
                }
        prompt = build_judge_prompt(
            record.question,
            record.pred,
            record.answer,
            template=self.config.judge_template,
            prompt_version=self.config.judge_prompt_version,
        )
        request_payload = self._build_payload(prompt)
        raw_response = self._post_with_retries(request_payload)
        content = extract_message_content(raw_response)
        score = analyse_judgemodel_res(
            content,
            template=self.config.judge_template,
            prompt_version=self.config.judge_prompt_version,
        )
        if score is None:
            raise ValueError(
                f"judge response did not contain a valid verdict: {content[:500]}"
            )
        result = {
            "score": score,
            "judge_reason": extract_judge_reason(
                content,
                template=self.config.judge_template,
                prompt_version=self.config.judge_prompt_version,
            ),
            "judge_raw_response": content,
            "judge_response": serialize_response(raw_response),
        }
        if self.config.judge_template == "sealqa":
            result["judge_label"] = extract_sealqa_grade(content)
        return result

    def complete(self, prompt: str) -> str:
        """Return plain judge text for structured benchmark scorers."""
        raw_response = self._post_with_retries(self._build_payload(prompt))
        return extract_message_content(raw_response)

    def _build_payload(self, prompt: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ],
                }
            ],
            self.config.token_param: self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        }
        if self.config.use_json_schema and self.config.judge_template == "browsecomp":
            if (
                self.config.judge_prompt_version
                == BROWSECOMP_OFFICIAL_PROMPT_VERSION
            ):
                properties = {
                    "extracted_final_answer": {"type": "string"},
                    "reasoning": {"type": "string"},
                    "correct": {"type": "string", "enum": ["yes", "no"]},
                    "confidence": {"type": "number"},
                }
                required = [
                    "extracted_final_answer",
                    "reasoning",
                    "correct",
                    "confidence",
                ]
                schema_name = "browsecomp_official_judge_result"
            else:
                properties = {
                    "reason": {"type": "string"},
                    "score": {"type": "integer", "enum": [0, 1]},
                }
                required = ["reason", "score"]
                schema_name = "judge_result"
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                        "additionalProperties": False,
                    },
                },
            }
        return payload

    def _post_with_retries(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(1, max(1, self.config.max_retries) + 1):
            try:
                return self._post(payload)
            except Exception as exc:
                last_exc = exc
                if attempt >= self.config.max_retries:
                    break
                time.sleep(min(30.0, 2.0 ** (attempt - 1)))
        assert last_exc is not None
        raise last_exc

    def _post(self, payload: dict[str, Any]) -> Any:
        return self._client().chat.completions.create(**payload)

    def _client(self) -> Any:
        if self._sdk_client is not None:
            return self._sdk_client
        try:
            from openai import AzureOpenAI, OpenAI
        except ImportError as exc:
            raise RuntimeError("missing package: openai. Install it with `python -m pip install openai`.") from exc

        if self.config.provider == "azure":
            self._sdk_client = AzureOpenAI(
                api_version=self.config.api_version,
                azure_endpoint=self.config.base_url,
                api_key=self.config.api_key,
                timeout=self.config.timeout_seconds,
                max_retries=0,
            )
        else:
            self._sdk_client = OpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                timeout=self.config.timeout_seconds,
                max_retries=0,
            )
        return self._sdk_client


class BrowseCompJudge:
    def __init__(self, client: JudgeClient | None, config: JudgeRunConfig):
        self.client = client
        self.config = config
        self._summary_lock = threading.Lock()

    def run(self) -> dict[str, Any]:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.metrics_path.parent.mkdir(parents=True, exist_ok=True)

        records = load_prediction_records(self.config.conv_dir)
        if self.config.resume_from_conv_dir is not None:
            resume_stats, actions = self.plan_resume(records)
            self.seed_resume_results(records, actions)
            print("judge resume plan: " + " ".join(
                f"{key}={value}" for key, value in sorted(resume_stats.items())
            ))
        pending = [record for record in records if self._should_run(record)]
        print(f"loaded predictions: {len(records)}")
        print(f"pending judge records: {len(pending)}")

        if pending:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
                future_to_record = {executor.submit(self._judge_one, record): record for record in pending}
                completed = 0
                for future in concurrent.futures.as_completed(future_to_record):
                    record = future_to_record[future]
                    completed += 1
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = self._judge_failed_result(record, exc)
                        self._write_result(record, result)
                    print(f"[{completed}/{len(pending)}] {record.run_id}: {result.get('status')} score={result.get('score')}")

        results = self._current_results(records)
        self.rewrite_summary(records)
        metrics = compute_metrics(results)
        metrics["judge_template"] = self.config.judge_template
        metrics["judge_prompt_version"] = (
            self.config.judge_prompt_version
            if self.config.judge_template == "browsecomp"
            else None
        )
        temporary = self.config.metrics_path.with_suffix(self.config.metrics_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.config.metrics_path)
        return metrics

    def plan_resume(
        self, records: list[PredictionRecord]
    ) -> tuple[dict[str, int], dict[str, str]]:
        stats = {
            "total": len(records),
            "current_reused": 0,
            "source_reusable": 0,
            "changed": 0,
            "missing_source_prediction": 0,
            "missing_source_judge": 0,
            "stale_source_judge": 0,
        }
        actions: dict[str, str] = {}
        source_conv = self.config.resume_from_conv_dir
        source_judge = self.config.resume_from_output_dir
        if source_conv is None or source_judge is None:
            return stats, actions

        for record in records:
            current_result = read_json_object(self._result_path(record))
            if self._result_matches_record(current_result, record, require_fingerprint=True):
                action = "current_reused"
            else:
                try:
                    relative_path = record.path.relative_to(self.config.conv_dir)
                except ValueError:
                    relative_path = Path(record.question_id) / f"{record.run_id}.json"
                source_record = load_prediction_record(source_conv / relative_path)
                if source_record is None:
                    action = "missing_source_prediction"
                elif prediction_fingerprint(source_record) != prediction_fingerprint(record):
                    action = "changed"
                else:
                    source_result_path = source_judge / record.question_id / f"{record.run_id}.json"
                    source_result = read_json_object(source_result_path)
                    if source_result is None:
                        action = "missing_source_judge"
                    elif not self._result_matches_record(
                        source_result, source_record, require_fingerprint=False
                    ):
                        action = "stale_source_judge"
                    else:
                        action = "source_reusable"
            actions[record.run_id] = action
            stats[action] += 1
        return stats, actions

    def seed_resume_results(
        self, records: list[PredictionRecord], actions: dict[str, str]
    ) -> None:
        source_judge = self.config.resume_from_output_dir
        if source_judge is None:
            return
        by_run_id = {record.run_id: record for record in records}
        for run_id, action in actions.items():
            if action != "source_reusable":
                continue
            record = by_run_id[run_id]
            source_path = source_judge / record.question_id / f"{record.run_id}.json"
            result = read_json_object(source_path)
            if result is None:
                continue
            seeded = dict(result)
            seeded["pred_path"] = str(record.path)
            seeded["pred_fingerprint"] = prediction_fingerprint(record)
            seeded["pred_attempt"] = prediction_attempt(record)
            seeded["judge_template"] = self.config.judge_template
            seeded["judge_prompt_version"] = (
                self.config.judge_prompt_version
                if self.config.judge_template == "browsecomp"
                else None
            )
            path = self._result_path(record)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".resume-copy.tmp")
            temporary.write_text(
                json.dumps(seeded, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            temporary.replace(path)

    def _should_run(self, record: PredictionRecord) -> bool:
        if self.config.force:
            return True
        path = self._result_path(record)
        if self.config.resume_from_conv_dir is None:
            return not self._result_matches_record(
                read_json_object(path), record, require_fingerprint=False
            )
        return not self._result_matches_record(
            read_json_object(path), record, require_fingerprint=True
        )

    def prompt_version_conflicts(
        self, records: list[PredictionRecord]
    ) -> list[Path]:
        if self.config.judge_template != "browsecomp":
            return []
        conflicts: list[Path] = []
        for record in records:
            path = self._result_path(record)
            result = read_json_object(path)
            if result is not None and not self._result_uses_selected_prompt(result):
                conflicts.append(path)
        return conflicts

    def _judge_one(self, record: PredictionRecord) -> dict[str, Any]:
        if record.pred_status == "policy_violation" and not self.config.include_failed:
            result = self._base_result(record)
            result.update(
                {
                    "status": "policy_violation_incorrect",
                    "score": INCORRECT_SCORE,
                    "judge_reason": "inference status is policy_violation; counted as incorrect without judge request",
                    "judge_raw_response": "",
                    "judge_response": None,
                }
            )
            self._write_result(record, result)
            return result

        if record.pred_status != "success" and not self.config.include_failed:
            result = self._base_result(record)
            result.update(
                {
                    "status": "skipped_infer_failed",
                    "score": None,
                    "judge_reason": f"inference status is {record.pred_status}",
                    "judge_raw_response": "",
                    "judge_response": None,
                }
            )
            self._write_result(record, result)
            return result

        if not record.pred.strip():
            result = self._base_result(record)
            result.update(
                {
                    "status": "skipped_empty_prediction",
                    "score": None,
                    "judge_reason": "empty model response",
                    "judge_raw_response": "",
                    "judge_response": None,
                }
            )
            self._write_result(record, result)
            return result

        if self.client is None:
            raise RuntimeError("judge client is not configured")

        judged = self.client.judge(record)
        result = self._base_result(record)
        result.update(
            {
                "status": "judged",
                "score": judged["score"],
                "judge_label": judged.get("judge_label"),
                "judge_reason": judged["judge_reason"],
                "judge_raw_response": judged["judge_raw_response"],
                "judge_response": judged["judge_response"],
            }
        )
        self._write_result(record, result)
        return result

    def _base_result(self, record: PredictionRecord) -> dict[str, Any]:
        return {
            "question_id": record.question_id,
            "run_id": record.run_id,
            "repeat_index": record.repeat_index,
            "type": record.type,
            "pred_status": record.pred_status,
            "question": record.question,
            "answer": record.answer,
            "prediction": record.pred,
            "pred_path": str(record.path),
            "pred_fingerprint": prediction_fingerprint(record),
            "pred_attempt": prediction_attempt(record),
            "judge_template": self.config.judge_template,
            "judge_prompt_version": (
                self.config.judge_prompt_version
                if self.config.judge_template == "browsecomp"
                else None
            ),
        }

    def _judge_failed_result(self, record: PredictionRecord, exc: Exception) -> dict[str, Any]:
        result = self._base_result(record)
        result.update(
            {
                "status": "judge_failed",
                "score": None,
                "judge_reason": "",
                "judge_raw_response": "",
                "judge_response": None,
                "error": str(exc),
            }
        )
        return result

    def _write_result(self, record: PredictionRecord, result: dict[str, Any]) -> None:
        path = self._result_path(record)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

        summary = self._summary_record(result, path)
        with self._summary_lock:
            with self.config.summary_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    def _summary_record(self, result: dict[str, Any], path: Path) -> dict[str, Any]:
        return {
            "question_id": result.get("question_id"),
            "run_id": result.get("run_id"),
            "repeat_index": result.get("repeat_index"),
            "type": result.get("type"),
            "status": result.get("status"),
            "pred_status": result.get("pred_status"),
            "score": result.get("score"),
            "judge_label": result.get("judge_label"),
            "judge_reason": result.get("judge_reason", ""),
            "pred_path": result.get("pred_path"),
            "pred_fingerprint": result.get("pred_fingerprint"),
            "pred_attempt": result.get("pred_attempt"),
            "judge_template": result.get("judge_template") or self.config.judge_template,
            "judge_prompt_version": self._result_prompt_version(result),
            "result_path": str(path),
            "error": result.get("error", ""),
        }

    def _result_matches_record(
        self,
        result: dict[str, Any] | None,
        record: PredictionRecord,
        *,
        require_fingerprint: bool,
    ) -> bool:
        if not result or result.get("status") == "judge_failed":
            return False
        if not self._result_uses_selected_prompt(result):
            return False
        if str(result.get("run_id") or "") != record.run_id:
            return False
        if str(result.get("question_id") or "") != record.question_id:
            return False
        if str(result.get("pred_status") or "") != record.pred_status:
            return False
        if str(result.get("prediction") or "") != record.pred:
            return False
        fingerprint = result.get("pred_fingerprint")
        if require_fingerprint:
            return fingerprint == prediction_fingerprint(record)
        return fingerprint in (None, "", prediction_fingerprint(record))

    def _result_prompt_version(self, result: dict[str, Any]) -> str | None:
        if self.config.judge_template != "browsecomp":
            return None
        stored = result.get("judge_prompt_version")
        if stored in (None, ""):
            return BROWSECOMP_HERMES_PROMPT_VERSION
        try:
            return normalize_browsecomp_prompt_version(str(stored))
        except ValueError:
            return str(stored)

    def _result_uses_selected_prompt(self, result: dict[str, Any]) -> bool:
        if self.config.judge_template != "browsecomp":
            return True
        stored_template = result.get("judge_template")
        if stored_template not in (None, "", "browsecomp"):
            return False
        return self._result_prompt_version(result) == self.config.judge_prompt_version

    def _current_results(self, records: list[PredictionRecord]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for record in records:
            result = read_json_object(self._result_path(record))
            if result is not None:
                results.append(result)
        return results

    def rewrite_summary(self, records: list[PredictionRecord]) -> None:
        temporary = self.config.summary_path.with_suffix(
            self.config.summary_path.suffix + ".tmp"
        )
        with self._summary_lock:
            with temporary.open("w", encoding="utf-8") as stream:
                for record in records:
                    path = self._result_path(record)
                    result = read_json_object(path)
                    if result is not None:
                        stream.write(
                            json.dumps(
                                self._summary_record(result, path), ensure_ascii=False
                            )
                            + "\n"
                        )
            temporary.replace(self.config.summary_path)

    def _result_path(self, record: PredictionRecord) -> Path:
        return self.config.output_dir / record.question_id / f"{record.run_id}.json"


def load_prediction_records(conv_dir: Path) -> list[PredictionRecord]:
    if not conv_dir.is_dir():
        raise FileNotFoundError(f"prediction conv dir not found: {conv_dir}")

    records: list[PredictionRecord] = []
    for path in sorted(conv_dir.rglob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            records.append(prediction_record_from_raw(path, raw))
    return records


def load_prediction_record(path: Path) -> PredictionRecord | None:
    raw = read_json_object(path)
    if raw is None:
        return None
    return prediction_record_from_raw(path, raw)


def prediction_record_from_raw(path: Path, raw: dict[str, Any]) -> PredictionRecord:
    run_id = str(raw.get("run_id") or path.stem)
    question_id = str(raw.get("question_id") or infer_question_id(run_id, path))
    return PredictionRecord(
        path=path,
        question_id=question_id,
        run_id=run_id,
        repeat_index=coerce_int(raw.get("repeat_index")),
        question=str(raw.get("question", "")),
        answer=raw.get("answer", ""),
        pred=str(
            raw.get("model_response")
            or raw.get("prediction")
            or raw.get("final_answer")
            or ""
        ),
        pred_status=str(raw.get("status", "unknown")),
        type=str(raw.get("type", "")),
        raw=raw,
    )


def read_json_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return raw if isinstance(raw, dict) else None


def prediction_fingerprint(record: PredictionRecord) -> str:
    payload = json.dumps(
        record.raw,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prediction_attempt(record: PredictionRecord) -> int | None:
    attempts: list[int] = []
    attempt = coerce_int(record.raw.get("attempt"))
    if attempt is not None and attempt > 0:
        attempts.append(attempt)
    retry_attempts = record.raw.get("retry_attempts")
    if isinstance(retry_attempts, list):
        for row in retry_attempts:
            if not isinstance(row, dict):
                continue
            value = coerce_int(row.get("attempt"))
            if value is not None and value > 0:
                attempts.append(value)
        if not attempts and retry_attempts:
            attempts.append(len(retry_attempts))
    return max(attempts) if attempts else None


def load_judge_results(output_dir: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if not output_dir.is_dir():
        return results
    for path in sorted(output_dir.rglob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(raw, dict):
            results.append(raw)
    return results


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    scored = [result for result in results if result.get("score") in (0, 1)]
    judged = [result for result in results if result.get("status") == "judged"]
    policy_violation_incorrect = [result for result in results if result.get("status") == "policy_violation_incorrect"]
    skipped_infer_failed = [result for result in results if result.get("status") == "skipped_infer_failed"]
    skipped_empty_prediction = [result for result in results if result.get("status") == "skipped_empty_prediction"]
    judge_failed = [result for result in results if result.get("status") == "judge_failed"]
    correct = sum(1 for result in scored if result.get("score") == 1)
    sealqa_correct = sum(1 for result in results if result.get("judge_label") == "A")
    sealqa_incorrect = sum(1 for result in results if result.get("judge_label") == "B")
    sealqa_not_attempted = sum(1 for result in results if result.get("judge_label") == "C")

    by_type: dict[str, dict[str, Any]] = {}
    for result in results:
        item_type = str(result.get("type") or "unknown")
        bucket = by_type.setdefault(
            item_type,
            {
                "total": 0,
                "scored": 0,
                "judged": 0,
                "policy_violation_incorrect": 0,
                "skipped": 0,
                "correct": 0,
                "accuracy": None,
            },
        )
        bucket["total"] += 1
        if result.get("status") == "judged":
            bucket["judged"] += 1
        if result.get("status") == "policy_violation_incorrect":
            bucket["policy_violation_incorrect"] += 1
        if str(result.get("status", "")).startswith("skipped_"):
            bucket["skipped"] += 1
        if result.get("score") in (0, 1):
            bucket["scored"] += 1
            bucket["correct"] += int(result.get("score") == 1)
    for bucket in by_type.values():
        bucket["accuracy"] = safe_ratio(bucket["correct"], bucket["scored"])

    question_groups: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        question_groups.setdefault(str(result.get("question_id") or "unknown"), []).append(result)

    question_total_all = len(question_groups)
    question_total_scored = 0
    any_correct = 0
    majority_correct = 0
    question_average_score_total = 0.0
    question_max_score_total = 0.0
    for group in question_groups.values():
        scores = [int(result.get("score") == 1) for result in group if result.get("score") in (0, 1)]
        if not scores:
            continue
        question_total_scored += 1
        question_average_score_total += sum(scores) / len(scores)
        question_max_score_total += max(scores)
        if any(scores):
            any_correct += 1
        if sum(scores) >= math.ceil(len(scores) / 2):
            majority_correct += 1

    question_average_score = safe_float_ratio(question_average_score_total, question_total_scored)
    question_max_score = safe_float_ratio(question_max_score_total, question_total_scored)

    repeat_groups: dict[int, list[dict[str, Any]]] = {}
    for result in results:
        repeat_index = result_repeat_index(result)
        repeat_groups.setdefault(repeat_index, []).append(result)

    per_repeat: list[dict[str, Any]] = []
    for repeat_index, group in sorted(repeat_groups.items()):
        repeat_scored = [result for result in group if result.get("score") in (0, 1)]
        repeat_correct = sum(1 for result in repeat_scored if result.get("score") == 1)
        per_repeat.append(
            {
                "repeat_index": repeat_index,
                "total": len(group),
                "scored": len(repeat_scored),
                "correct": repeat_correct,
                "incorrect": sum(1 for result in repeat_scored if result.get("score") == 0),
                "unscored": len(group) - len(repeat_scored),
                "pass1": safe_ratio(repeat_correct, len(group)),
            }
        )

    pass1 = per_repeat[0]["pass1"] if per_repeat else None
    repeat_pass1_values = [item["pass1"] for item in per_repeat if item["pass1"] is not None]
    avrpass1 = (
        round(sum(repeat_pass1_values) / len(repeat_pass1_values), 6)
        if repeat_pass1_values
        else None
    )
    best_pass1 = max(repeat_pass1_values) if repeat_pass1_values else None

    return {
        "prediction_total": total,
        "scored": len(scored),
        "judged": len(judged),
        "policy_violation_incorrect": len(policy_violation_incorrect),
        "skipped_infer_failed": len(skipped_infer_failed),
        "skipped_empty_prediction": len(skipped_empty_prediction),
        "judge_failed": len(judge_failed),
        "correct": correct,
        "sealqa_correct": sealqa_correct,
        "sealqa_incorrect": sealqa_incorrect,
        "sealqa_not_attempted": sealqa_not_attempted,
        "accuracy": safe_ratio(correct, len(scored)),
        "run_level_accuracy_scored": safe_ratio(correct, len(scored)),
        "question_total_all": question_total_all,
        "question_total_scored": question_total_scored,
        "question_average_score": question_average_score,
        "question_max_score": question_max_score,
        "question_any_correct": any_correct,
        "question_any_correct_accuracy": safe_ratio(any_correct, question_total_scored),
        "question_majority_correct": majority_correct,
        "question_majority_accuracy": safe_ratio(majority_correct, question_total_scored),
        "pass1": pass1,
        "avrpass1": avrpass1,
        "best_pass1": best_pass1,
        "repeat_count": len(per_repeat),
        "per_repeat": per_repeat,
        "by_type": by_type,
    }


def normalize_browsecomp_prompt_version(value: str) -> str:
    try:
        return BROWSECOMP_PROMPT_VERSIONS[value]
    except KeyError as exc:
        allowed = ", ".join(sorted(BROWSECOMP_PROMPT_VERSIONS))
        raise ValueError(
            f"unknown BrowseComp prompt version: {value}; expected one of: {allowed}"
        ) from exc


def get_chat_template() -> str:
    return (
        "You are a strict answer-equivalence judge. You will be given a question, "
        "a standard answer, and a model answer.\n\n"
        "Determine whether the model answer is consistent with the standard answer. "
        "They are consistent if they convey the same meaning, even if worded differently "
        '(e.g., "pink" and "it is pink" are consistent).\n\n'
        "Respond in JSON with two fields:\n"
        '- "reason": a brief explanation of why the answers are or are not consistent\n'
        '- "score": 1 if consistent, 0 if inconsistent\n'
    )


def get_browsecomp_official_grader_template() -> str:
    """OpenAI simple-evals BrowseComp GRADER_TEMPLATE."""
    return r"""
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available.
""".strip()


def get_sealqa_grader_template() -> str:
    """Official SealQA GPT-4o-mini auto-rater prompt from the released notebook."""
    return """
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.


The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.


The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.


The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
```
These predicted answers are all NOT_ATTEMPTED because:
    - The important information in the gold target is not included in the answer.
    - No statements in the answer contradict the gold target.


Also note the following things:
- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
    - For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
    - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".
    - Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
- Do not give credit for an answer if it contains any internal inconsistency.
    - For example, consider the question: "How many NBA players have scored 60 or more points in a regular season game since 2024?" with the gold answer "8". A response is INCORRECT if it states "8 players" but lists 7 or 9, or if it initially says "8 players" but later contradicts this by concluding 7 or 9.


Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
```
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
```

Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return the letters "A", "B", or "C", with no text around it.
""".strip()


def build_judge_prompt(
    user_question: str,
    pred: str,
    ground_truths: Any,
    template: str = "browsecomp",
    prompt_version: str = BROWSECOMP_HERMES_PROMPT_VERSION,
) -> str:
    if template == "sealqa":
        return get_sealqa_grader_template().format(
            question=user_question,
            target=format_ground_truth(ground_truths),
            predicted_answer=pred,
        )
    if template == "gaia-text-103":
        # Official MiroThinker/WebAgent GAIA-Text-103 LLM-as-a-Judge template.
        return (
            "You are an evaluation assistant. Please determine if the predicted answer "
            "is equivalent to the labeled answer.\n\n"
            f"Question: {user_question}\n\n"
            f"Labeled Answer: {format_ground_truth(ground_truths)}\n\n"
            f"Predicted Answer: {pred}\n\n"
            'Did the model give an answer **equivalent** to the labeled answer? Please '
            'respond with "Correct" if they are equivalent, or "Incorrect" if they are '
            "not equivalent. Do not include any other text.\n"
        )
    if template == "xbench-deepsearch-2510":
        # Official xbench-evals DeepSearch-2510 LLM_JUDGE_PROMPT.
        return (
            "你是一个通用人工智能助手。根据下面给出的[正确答案], 判断以下对[原问题]的"
            "[回答]的回答是否正确。\n\n"
            f"[原问题]: {user_question}\n\n"
            f"[正确答案]: {format_ground_truth(ground_truths)}\n\n"
            f"[回答]:{pred}\n\n"
            "你的判断必须按照以下格式和标准进行:\n\n"
            "最终答案: 从[回答]中提取出的最终准确答案。如果[回答]中没有明确的最终答案, "
            "则填写'无'。\n\n"
            "解释: 根据[正确]解释为什么[最终答案]是正确的或错误的。只关注[最终答案]与"
            "[正确答案]之间是否存在实质性差异, 不要评论题目的背景, 不要尝试重新解题, "
            "不要为任何不同于[正确答案]的答案辩护, 只专注于判断答案是否一致。\n\n"
            "结论: 如果[最终答案]与上方给出的[正确答案]一致, 或者在数值题目中处于可接受的"
            "微小误差范围内, 则填写'正确'; 否则（即存在任何不一致、歧义、不等价或提取出的"
            "答案错误的情况）填写'错误'。"
        )
    if template != "browsecomp":
        raise ValueError(f"unknown judge template: {template}")
    normalized_prompt_version = normalize_browsecomp_prompt_version(prompt_version)
    if normalized_prompt_version == BROWSECOMP_OFFICIAL_PROMPT_VERSION:
        template = (
            get_browsecomp_official_grader_template()
            + "\n\n"
            + BROWSECOMP_ANSWER_EQUIVALENCE_GUIDANCE
        )
        return template.format(
            question=user_question,
            response=pred,
            correct_answer=format_ground_truth(ground_truths),
        )
    return (
        get_chat_template()
        + "\n"
        + f"[Question]: {user_question}\n"
        + f"[Standard Answer]: {format_ground_truth(ground_truths)}\n"
        + f"[Model Answer]: {pred}\n"
    )


def analyse_judgemodel_res(
    judge_res: str,
    template: str = "browsecomp",
    prompt_version: str = BROWSECOMP_HERMES_PROMPT_VERSION,
) -> float | None:
    if template == "sealqa":
        grade = extract_sealqa_grade(judge_res)
        if grade is None:
            return None
        return CORRECT_SCORE if grade == "A" else INCORRECT_SCORE
    if template == "gaia-text-103":
        normalized = judge_res.strip().rstrip(".").lower()
        if normalized == "correct":
            return CORRECT_SCORE
        if normalized == "incorrect":
            return INCORRECT_SCORE

        # Some judges explain briefly and then emit the requested label again,
        # e.g. `... So "Correct".Correct`. Accept only a standalone terminal
        # label, rather than searching arbitrary occurrences in the response.
        terminal_label = re.search(
            r"(?:^|[\n.:])\s*[\"'`*_]*(incorrect|correct)[\"'`*_.!?]*\s*$",
            judge_res,
            flags=re.IGNORECASE,
        )
        if terminal_label:
            return (
                CORRECT_SCORE
                if terminal_label.group(1).lower() == "correct"
                else INCORRECT_SCORE
            )
        return None
    if template == "xbench-deepsearch-2510":
        match = re.search(r"结论\s*[:：]\s*(正确|错误)", judge_res)
        if not match:
            return None
        return CORRECT_SCORE if match.group(1) == "正确" else INCORRECT_SCORE

    normalized_prompt_version = normalize_browsecomp_prompt_version(prompt_version)
    if normalized_prompt_version == BROWSECOMP_OFFICIAL_PROMPT_VERSION:
        obj = extract_judge_object(judge_res)
        correct = obj.get("correct") if isinstance(obj, dict) else None
        if isinstance(correct, str) and correct.strip().lower() in {"yes", "no"}:
            return (
                CORRECT_SCORE
                if correct.strip().lower() == "yes"
                else INCORRECT_SCORE
            )
        match = re.search(
            r"(?:^|\n)\s*correct\s*:\s*[\"']?(yes|no)\b",
            judge_res,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        return CORRECT_SCORE if match.group(1).lower() == "yes" else INCORRECT_SCORE

    # Keep the scoring logic aligned with the referenced reward script.
    import re as _re

    try:
        obj = json.loads(judge_res.strip())
        if "score" in obj:
            return CORRECT_SCORE if obj["score"] == 1 else INCORRECT_SCORE
    except (json.JSONDecodeError, TypeError):
        pass

    json_match = _re.search(r'\{[^{}]*"score"\s*:\s*(\d+)[^{}]*\}', judge_res)
    if json_match:
        score_val = int(json_match.group(1))
        return CORRECT_SCORE if score_val == 1 else INCORRECT_SCORE

    return None


def extract_judge_reason(
    judge_res: str,
    template: str = "browsecomp",
    prompt_version: str = BROWSECOMP_HERMES_PROMPT_VERSION,
) -> str:
    if template == "sealqa":
        return ""
    if template == "xbench-deepsearch-2510":
        match = re.search(r"解释\s*[:：]\s*(.*?)(?=\n\s*结论\s*[:：]|\Z)", judge_res, re.S)
        return match.group(1).strip() if match else ""
    obj = extract_judge_object(judge_res)
    if (
        template == "browsecomp"
        and normalize_browsecomp_prompt_version(prompt_version)
        == BROWSECOMP_OFFICIAL_PROMPT_VERSION
    ):
        reasoning = obj.get("reasoning") if isinstance(obj, dict) else None
        if isinstance(reasoning, str):
            return reasoning
        match = re.search(
            r"(?:^|\n)\s*reasoning\s*:\s*(.*?)(?=\n\s*correct\s*:|\Z)",
            judge_res,
            flags=re.IGNORECASE | re.DOTALL,
        )
        return match.group(1).strip() if match else ""
    reason = obj.get("reason") if isinstance(obj, dict) else None
    return reason if isinstance(reason, str) else ""


def extract_sealqa_grade(judge_res: str) -> str | None:
    match = re.search(r"\b([ABC])\b", judge_res.strip().upper())
    return match.group(1) if match else None


def extract_judge_object(judge_res: str) -> dict[str, Any]:
    try:
        obj = json.loads(judge_res.strip())
        return obj if isinstance(obj, dict) else {}
    except (json.JSONDecodeError, TypeError):
        pass

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", judge_res):
        try:
            obj, _ = decoder.raw_decode(judge_res[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return {}


def extract_xbench_final_answer(response: str) -> str | None:
    """Mirror the official exact-match fast path for `最终答案:` responses."""
    match = re.search(r"最终答案:*(.*)", response)
    if not match:
        return None
    return match.group(1).strip()


def extract_message_content(response: Any) -> str:
    if hasattr(response, "choices"):
        choices = response.choices
        if not choices:
            raise ValueError(f"judge response missing choices: {response}")
        message = choices[0].message
        content = message.content
        return content if isinstance(content, str) else str(content)

    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"judge response missing choices: {response}")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise ValueError(f"judge response missing message: {response}")
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def serialize_response(response: Any) -> Any:
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    if hasattr(response, "dict"):
        return response.dict()
    return response


def format_ground_truth(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def infer_question_id(run_id: str, path: Path) -> str:
    if "__r" in run_id:
        return run_id.split("__r", 1)[0]
    if path.parent.name != "conv":
        return path.parent.name
    return run_id


def result_repeat_index(result: dict[str, Any]) -> int:
    repeat_index = coerce_int(result.get("repeat_index"))
    if repeat_index is not None and repeat_index >= 1:
        return repeat_index
    match = re.search(r"__r(\d+)$", str(result.get("run_id") or ""))
    if match:
        return int(match.group(1))
    return 1


def safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 6)


def safe_float_ratio(numerator: float, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 6)


def make_absolute(path: Path) -> Path:
    if path.is_absolute():
        return path
    return Path.cwd() / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Judge Browse Comp Hermes predictions with an answer-equivalence LLM")
    parser.add_argument("--pred-root", required=True, help="Prediction root: output/preds/{save_name}/{dataset}")
    parser.add_argument("--conv-dir", help="Prediction JSON directory. Defaults to {pred-root}/conv")
    parser.add_argument(
        "--judge-output-dir",
        help="Judge output directory. Default is prompt-versioned under {pred-root}",
    )
    parser.add_argument(
        "--summary-path",
        help="Judge summary JSONL. Default is prompt-versioned under {pred-root}",
    )
    parser.add_argument(
        "--metrics-path",
        help="Judge metrics JSON. Default is prompt-versioned under {pred-root}",
    )
    parser.add_argument(
        "--resume-from",
        help=(
            "Previous prediction run root containing conv/ and judge/. Unchanged judge "
            "results are reused; changed final trajectories are rejudged."
        ),
    )
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", "gpt-5-mini"), help="Judge model/deployment name")
    parser.add_argument(
        "--judge-template",
        choices=["browsecomp", "sealqa", "gaia-text-103", "xbench-deepsearch-2510", "widesearch"],
        default=os.getenv("JUDGE_TEMPLATE", "browsecomp"),
        help="Dataset-specific grading prompt (default: browsecomp)",
    )
    parser.add_argument(
        "--prompt-version",
        choices=["hermes", "official"],
        default=os.getenv("JUDGE_PROMPT_VERSION", "hermes"),
        help=(
            "BrowseComp grading prompt: hermes or official "
            "(default: hermes; env: JUDGE_PROMPT_VERSION)"
        ),
    )
    parser.add_argument(
        "--judge-provider",
        choices=["openai", "azure"],
        default=os.getenv("JUDGE_PROVIDER", os.getenv("JUDGE_API_TYPE", "azure")),
        help="Judge API style",
    )
    parser.add_argument(
        "--judge-base-url",
        default=os.getenv("JUDGE_BASE_URL", os.getenv("AZURE_OPENAI_ENDPOINT", "")),
        help="OpenAI-compatible base URL or Azure endpoint",
    )
    parser.add_argument("--judge-api-key-env", default="JUDGE_API_KEY", help="Environment variable containing judge API key")
    parser.add_argument("--judge-api-key", default=None, help="Judge API key override. Prefer --judge-api-key-env.")
    parser.add_argument(
        "--judge-api-version",
        default=os.getenv("JUDGE_API_VERSION", os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")),
        help="Azure API version",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="Parallel judge requests")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per judge request")
    parser.add_argument("--timeout-seconds", type=float, default=60, help="Per-request timeout")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Judge max output tokens")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--token-param", choices=["max_completion_tokens", "max_tokens"], default="max_completion_tokens")
    parser.add_argument("--no-json-schema", action="store_true", help="Do not send response_format json_schema")
    parser.add_argument("--include-failed", action="store_true", help="Also send failed inference outputs to the judge")
    parser.add_argument("--force", action="store_true", help="Rejudge even when judge output exists")
    parser.add_argument("--dry-run", action="store_true", help="Print planned judge count and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.judge_template != "browsecomp" and args.prompt_version != "hermes":
        print("--prompt-version official requires --judge-template browsecomp")
        return 2
    judge_prompt_version = normalize_browsecomp_prompt_version(args.prompt_version)
    artifact_suffix = "_official" if args.prompt_version == "official" else ""
    judge_dir_name = f"judge{artifact_suffix}"
    summary_name = f"judge{artifact_suffix}_summary.jsonl"
    metrics_name = f"judge{artifact_suffix}_metrics.json"
    pred_root = Path(args.pred_root)
    if not pred_root.is_absolute():
        pred_root = Path.cwd() / pred_root
    conv_dir = make_absolute(Path(args.conv_dir) if args.conv_dir else pred_root / "conv")
    output_dir = make_absolute(
        Path(args.judge_output_dir)
        if args.judge_output_dir
        else pred_root / judge_dir_name
    )
    summary_path = make_absolute(
        Path(args.summary_path) if args.summary_path else pred_root / summary_name
    )
    metrics_path = make_absolute(
        Path(args.metrics_path) if args.metrics_path else pred_root / metrics_name
    )
    resume_from_root = None
    resume_from_conv_dir = None
    resume_from_output_dir = None
    if args.resume_from:
        resume_from_root = make_absolute(Path(args.resume_from))
        resume_from_conv_dir = resume_from_root / "conv"
        resume_from_output_dir = resume_from_root / judge_dir_name
        if not resume_from_conv_dir.is_dir():
            print(f"resume source prediction directory not found: {resume_from_conv_dir}")
            return 2
        if resume_from_root.resolve() == pred_root.resolve():
            print("--resume-from must differ from --pred-root")
            return 2
        protected_paths = {
            (resume_from_root / judge_dir_name).resolve(),
            (resume_from_root / summary_name).resolve(),
            (resume_from_root / metrics_name).resolve(),
        }
        requested_paths = {
            output_dir.resolve(),
            summary_path.resolve(),
            metrics_path.resolve(),
        }
        if protected_paths & requested_paths:
            print("resume judge outputs must not overwrite the source run")
            return 2
        if args.force:
            print("--resume-from cannot be combined with --force")
            return 2
        if args.judge_template == "widesearch":
            print("--resume-from is not yet supported for the widesearch judge")
            return 2

    records = load_prediction_records(conv_dir)
    run_config = JudgeRunConfig(
        conv_dir=conv_dir,
        output_dir=output_dir,
        summary_path=summary_path,
        metrics_path=metrics_path,
        num_workers=args.num_workers,
        force=args.force,
        include_failed=args.include_failed,
        resume_from_conv_dir=resume_from_conv_dir,
        resume_from_output_dir=resume_from_output_dir,
        judge_template=args.judge_template,
        judge_prompt_version=judge_prompt_version,
    )
    preview = BrowseCompJudge(client=None, config=run_config)
    conflicts = preview.prompt_version_conflicts(records)
    if conflicts and not args.force:
        print(
            "existing judge outputs use a different BrowseComp prompt version; "
            "choose a separate output path or pass --force"
        )
        print(f"first conflicting result: {conflicts[0]}")
        return 2
    resume_stats = None
    if resume_from_conv_dir is not None:
        resume_stats, actions = preview.plan_resume(records)
        reusable = {"current_reused", "source_reusable"}
        pending = [record for record in records if actions.get(record.run_id) not in reusable]
    else:
        pending = [record for record in records if preview._should_run(record)]
    needs_client = any(record.pred_status == "success" or args.include_failed for record in pending)

    print("Browse Comp Judge")
    print(f"  pred_root: {pred_root}")
    print(f"  conv_dir: {conv_dir}")
    print(f"  judge_output: {output_dir}")
    print(f"  summary_path: {summary_path}")
    print(f"  metrics_path: {metrics_path}")
    print(f"  judge_model: {args.judge_model}")
    print(f"  judge_template: {args.judge_template}")
    print(f"  judge_prompt_version: {judge_prompt_version}")
    print(f"  judge_provider: {args.judge_provider}")
    print(f"  predictions: {len(records)}")
    print(f"  pending: {len(pending)}")
    print(f"  resume_from: {resume_from_root or '(none)'}")
    if resume_stats is not None:
        print(
            "  resume_plan: "
            + " ".join(f"{key}={value}" for key, value in sorted(resume_stats.items()))
        )
    print(f"  workers: {args.num_workers}")
    print(f"  retries: {args.max_retries}")

    if args.dry_run:
        return 0

    client: JudgeClient | None = None
    if needs_client:
        api_key = args.judge_api_key or os.getenv(args.judge_api_key_env, "")
        if not api_key:
            print(f"missing judge API key: set {args.judge_api_key_env} or pass --judge-api-key")
            return 2
        if not args.judge_base_url:
            print("missing judge base URL: set JUDGE_BASE_URL/AZURE_OPENAI_ENDPOINT or pass --judge-base-url")
            return 2
        client = JudgeClient(
            JudgeClientConfig(
                api_key=api_key,
                base_url=args.judge_base_url,
                model=args.judge_model,
                provider=args.judge_provider,
                api_version=args.judge_api_version,
                timeout_seconds=args.timeout_seconds,
                max_retries=args.max_retries,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                token_param=args.token_param,
                use_json_schema=not args.no_json_schema,
                judge_template=args.judge_template,
                judge_prompt_version=judge_prompt_version,
            )
        )

    if args.judge_template == "widesearch":
        from .widesearch_judge import WideSearchJudge

        runner = WideSearchJudge(client=client, config=run_config)
    else:
        runner = BrowseCompJudge(client=client, config=run_config)

    metrics = runner.run()
    print("done")
    if args.judge_template == "widesearch":
        print(
            f"  SR Avg@N={metrics['sr_avg_n']} Pass@N={metrics['sr_pass_n']} "
            f"Row-F1 Avg@N={metrics['row_f1_avg_n']} Max@N={metrics['row_f1_max_n']} "
            f"Item-F1 Avg@N={metrics['item_f1_avg_n']} Max@N={metrics['item_f1_max_n']}"
        )
        for item in metrics["per_repeat"]:
            print(
                f"  repeat={item['repeat_index']} total={item['total']} "
                f"sr={item['score']} row_f1={item['f1_by_row']} "
                f"item_f1={item['f1_by_item']}"
            )
    else:
        print(
            "  "
            + " ".join(
                f"{key}={value}"
                for key, value in metrics.items()
                if key not in {"by_type", "per_repeat"}
            )
        )
        for item in metrics["per_repeat"]:
            print(
                f"  repeat={item['repeat_index']} correct={item['correct']}/{item['total']} "
                f"scored={item['scored']} unscored={item['unscored']} pass1={item['pass1']}"
            )
    return 0 if metrics.get("judge_failed", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
