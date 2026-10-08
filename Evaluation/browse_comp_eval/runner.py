from __future__ import annotations

import concurrent.futures
import collections
import json
import shutil
import threading
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .dataset import RunSpec
from .endpoint_health import EndpointWatchdogAbort
from .hermes_client import HermesClient
from .whitelist import find_policy_violations


@dataclass(frozen=True)
class RunnerConfig:
    output_dir: Path
    workspace_dir: Path
    summary_path: Path
    num_workers: int
    max_retries: int
    force: bool
    skip_statuses: set[str]
    allowed_tools: set[str]
    workspace_namespace: str = ""
    numbered_output: bool = False
    rerun_failed: bool = False
    resume_from_dir: Path | None = None


class BrowseCompRunner:
    def __init__(self, client: HermesClient, config: RunnerConfig):
        self.client = client
        self.config = config
        self._summary_lock = threading.Lock()

    def run(self, specs: Iterable[RunSpec]) -> dict[str, int]:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.config.summary_path.parent.mkdir(parents=True, exist_ok=True)

        all_specs = list(specs)
        pending = [spec for spec in all_specs if self._should_run(spec)]
        print(f"loaded runs: {len(all_specs)}")
        print(f"pending runs: {len(pending)}")

        stats = {"total": len(all_specs), "pending": len(pending), "success": 0, "failed": 0, "policy_violation": 0, "interrupted": 0, "skipped": len(all_specs) - len(pending)}
        if not pending:
            return stats

        # Submit only up to the pool's current healthy capacity.  The executor
        # itself may grow when a registry hot-adds endpoints, but Python creates
        # worker threads lazily so an empty or small registry stays lightweight.
        max_workers = max(1, len(pending))
        unscheduled = collections.deque(pending)
        future_to_spec: dict[concurrent.futures.Future, RunSpec] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            completed = 0
            aborted = False
            while unscheduled or future_to_spec:
                capacity = self._scheduling_capacity()
                while unscheduled and len(future_to_spec) < capacity:
                    spec = unscheduled.popleft()
                    future_to_spec[executor.submit(self._run_one, spec)] = spec

                if not future_to_spec:
                    try:
                        self._wait_for_capacity_change()
                    except EndpointWatchdogAbort:
                        aborted = True
                        break
                    continue

                done, _not_done = concurrent.futures.wait(
                    tuple(future_to_spec),
                    timeout=0.5,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                if not done:
                    continue
                for future in done:
                    spec = future_to_spec.pop(future)
                    completed += 1
                    try:
                        result = future.result()
                    except EndpointWatchdogAbort:
                        stats["interrupted"] += 1
                        aborted = True
                        continue
                    except concurrent.futures.CancelledError:
                        stats["interrupted"] += 1
                        continue
                    except Exception as exc:
                        result = self._exception_result(spec, exc)
                        self._write_result(spec, result)
                    status = result.get("status", "failed")
                    stats[status] = stats.get(status, 0) + 1
                    print(f"[{completed}/{len(pending)}] {spec.run_id}: {status}")
                if aborted:
                    break

            if aborted:
                for future in future_to_spec:
                    future.cancel()
        return stats

    def _scheduling_capacity(self) -> int:
        capacity = getattr(self.client, "scheduling_capacity", None)
        if not callable(capacity):
            return self.config.num_workers
        value = capacity()
        return max(0, int(value))

    def _wait_for_capacity_change(self) -> None:
        wait = getattr(self.client, "wait_for_capacity_change", None)
        if callable(wait):
            wait(timeout=0.5)
        else:
            time.sleep(0.1)

    def seed_resume_results(self, specs: Iterable[RunSpec]) -> dict[str, int]:
        """Seed a new output directory from the canonical results of an older run."""
        all_specs = list(specs)
        stats = {
            "total": len(all_specs),
            "copied": 0,
            "reused": 0,
            "missing": 0,
            "malformed": 0,
        }
        source_dir = self.config.resume_from_dir
        if source_dir is None:
            stats["missing"] = len(all_specs)
            return stats

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        for spec in all_specs:
            destination = self._result_path(spec)
            if self._read_result(destination) is not None:
                stats["reused"] += 1
                continue

            source = self._result_path_in(source_dir, spec)
            if not source.is_file():
                stats["missing"] += 1
                continue
            if self._read_result(source) is None:
                stats["malformed"] += 1
                continue

            source_result = self._read_result(source)
            assert source_result is not None
            self._copy_interruption_sidecars(
                source_result,
                source_dir.parent,
                self.config.output_dir.parent,
            )

            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".resume-copy.tmp")
            shutil.copy2(source, temporary)
            temporary.replace(destination)
            stats["copied"] += 1
        return stats

    def rewrite_summary(self, specs: Iterable[RunSpec]) -> dict[str, int]:
        """Atomically rebuild a deduplicated overall summary from canonical results."""
        all_specs = list(specs)
        records: list[dict] = []
        metrics = {
            "total": len(all_specs),
            "observed": 0,
            "missing": 0,
            "malformed": 0,
            "success": 0,
            "failed": 0,
            "policy_violation": 0,
            "interrupted": 0,
            "other_status": 0,
        }
        for spec in all_specs:
            path = self._result_path(spec)
            if not path.is_file():
                metrics["missing"] += 1
                continue
            result = self._read_result(path)
            if result is None:
                metrics["malformed"] += 1
                continue
            metrics["observed"] += 1
            status = result.get("status")
            if status == "interrupted":
                metrics["interrupted"] += 1
                continue
            records.append(self._summary_record(result, path))
            if status in {"success", "failed", "policy_violation"}:
                metrics[status] += 1
            else:
                metrics["other_status"] += 1

        self.config.summary_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.config.summary_path.with_suffix(self.config.summary_path.suffix + ".tmp")
        with self._summary_lock:
            with temporary.open("w", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            temporary.replace(self.config.summary_path)
        return metrics

    def _should_run(self, spec: RunSpec) -> bool:
        if self.config.force:
            return True
        existing = self._load_existing_result(spec)
        if existing is None:
            return True
        status = existing.get("status")
        if status == "failed":
            return (
                self.config.rerun_failed
                and self._completed_attempt_count(existing) < max(1, self.config.max_retries)
            )
        return status not in self.config.skip_statuses

    def _run_one(self, spec: RunSpec) -> dict:
        existing = None if self.config.force else self._load_existing_result(spec)
        attempts = self._existing_attempts(existing)
        interruptions = self._existing_interruptions(existing)
        final_result: dict | None = None
        max_retries = max(1, self.config.max_retries)
        attempt = self._completed_attempt_count(existing) + 1
        excluded_generations: dict[int, int] = {}

        while attempt <= max_retries:
            bind = getattr(self.client, "bind", None)
            client_context = (
                bind(excluded_generations=excluded_generations)
                if callable(bind)
                else nullcontext(self.client)
            )
            rebind = False
            with client_context as client:
                while attempt <= max_retries:
                    if self.config.workspace_namespace:
                        attempt_workspace = (
                            self.config.workspace_dir
                            / self.config.workspace_namespace
                            / spec.run_id
                            / f"attempt_{attempt}"
                        )
                    else:
                        attempt_workspace = (
                            self.config.workspace_dir / spec.run_id / f"attempt_{attempt}"
                        )
                    result = client.run(spec, attempt_workspace, attempt=attempt)
                    if result.get("attempt_consumed", True) is False:
                        reference = self._write_interruption(spec, result, attempt)
                        interruptions.append(reference)
                        route_index = result.get("endpoint_route_index")
                        generation = result.get("endpoint_generation")
                        if isinstance(route_index, int) and isinstance(generation, int):
                            excluded_generations[route_index] = max(
                                generation,
                                excluded_generations.get(route_index, -1),
                            )
                        checkpoint = dict(result)
                        checkpoint.update(
                            {
                                "status": "interrupted",
                                "completed": False,
                                "attempt": attempt,
                                "attempt_consumed": False,
                                "retry_attempts": attempts,
                                "endpoint_interruptions": interruptions,
                                "max_retries": max_retries,
                            }
                        )
                        self._write_result(spec, checkpoint, append_summary=False)
                        rebind = True
                        break

                    result = self._apply_policy(result)
                    result["attempt_consumed"] = True
                    attempts.append(
                        {
                            "attempt": attempt,
                            "attempt_consumed": True,
                            "status": result.get("status"),
                            "error": result.get("error", ""),
                            "session_id": result.get("session_id"),
                            "profile": result.get("profile"),
                            "model_base_url": result.get("model_base_url", ""),
                            "duration_seconds": result.get("duration_seconds", 0),
                        }
                    )
                    final_result = result
                    final_result["retry_attempts"] = attempts
                    final_result["endpoint_interruptions"] = interruptions
                    final_result["max_retries"] = max_retries
                    final_result["completed"] = result.get("status") != "failed"
                    self._write_result(spec, final_result, append_summary=False)
                    if result.get("status") != "failed":
                        break
                    attempt += 1
            if final_result is not None and final_result.get("status") != "failed":
                break
            if rebind:
                continue

        assert final_result is not None
        final_result["completed"] = final_result.get("status") != "failed"
        self._write_result(spec, final_result)
        return final_result

    def _apply_policy(self, result: dict) -> dict:
        tool_calls = result.get("tool_calls") or {}
        violations = find_policy_violations(tool_calls, self.config.allowed_tools)
        if violations:
            result["status"] = "policy_violation"
            result["policy_violations"] = violations
            result["error"] = "tool call outside whitelist: " + ", ".join(violations)
        else:
            result["policy_violations"] = []
        return result

    def _write_result(self, spec: RunSpec, result: dict, *, append_summary: bool = True) -> None:
        path = self._result_path(spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

        if not append_summary:
            return

        summary = self._summary_record(result, path)
        with self._summary_lock:
            with self.config.summary_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    def _write_interruption(self, spec: RunSpec, result: dict, attempt: int) -> dict:
        event_id = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{uuid.uuid4().hex[:10]}"
        relative = (
            Path("endpoint_interruptions")
            / spec.run_id
            / f"attempt_{attempt}"
            / f"{event_id}.json"
        )
        destination = self.config.output_dir.parent / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "event_id": event_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_id": spec.run_id,
            "question_id": spec.question_id,
            "repeat_index": spec.repeat_index,
            "attempt": attempt,
            "status": "failed",
            "attempt_consumed": False,
            "failure_type": "model_endpoint_unavailable",
            "model_base_url": result.get("model_base_url", ""),
            "trace_complete": bool(result.get("trace_complete", False)),
            "endpoint_route_index": result.get("endpoint_route_index"),
            "endpoint_generation": result.get("endpoint_generation"),
            "probe_evidence": result.get("endpoint_health_evidence")
            or {
                "failure_type": result.get("failure_type", ""),
                "reason": result.get("error", ""),
            },
            "result": result,
        }
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(destination)
        return {
            "event_id": event_id,
            "attempt": attempt,
            "status": "failed",
            "attempt_consumed": False,
            "failure_type": "model_endpoint_unavailable",
            "model_base_url": result.get("model_base_url", ""),
            "trace_complete": payload["trace_complete"],
            "result_path": relative.as_posix(),
        }

    @staticmethod
    def _copy_interruption_sidecars(
        result: dict, source_base: Path, destination_base: Path
    ) -> None:
        rows = result.get("endpoint_interruptions")
        if not isinstance(rows, list):
            return
        for row in rows:
            if not isinstance(row, dict):
                continue
            raw_path = row.get("result_path")
            if not isinstance(raw_path, str) or not raw_path:
                continue
            relative = Path(raw_path)
            if relative.is_absolute() or ".." in relative.parts:
                continue
            source = source_base / relative
            destination = destination_base / relative
            if not source.is_file() or destination.is_file():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".resume-copy.tmp")
            shutil.copy2(source, temporary)
            temporary.replace(destination)

    @staticmethod
    def _summary_record(result: dict, path: Path) -> dict:
        return {
            "run_id": result.get("run_id"),
            "question_id": result.get("question_id"),
            "repeat_index": result.get("repeat_index"),
            "type": result.get("type"),
            "status": result.get("status"),
            "attempt": result.get("attempt"),
            "attempt_consumed": result.get("attempt_consumed", True),
            "error": result.get("error", ""),
            "session_id": result.get("session_id"),
            "profile": result.get("profile"),
            "model_base_url": result.get("model_base_url", ""),
            "browser_cdp_url": result.get("browser_cdp_url", ""),
            "duration_seconds": result.get("duration_seconds", 0),
            "tool_calls": result.get("tool_calls", {}),
            "result_path": str(path),
            "workspace_dir": result.get("workspace_dir"),
        }

    def _result_path(self, spec: RunSpec) -> Path:
        return self._result_path_in(self.config.output_dir, spec)

    def _result_path_in(self, output_dir: Path, spec: RunSpec) -> Path:
        if self.config.numbered_output:
            output_id = f"q{spec.item.source_line:06d}"
            return output_dir / output_id / f"{output_id}__r{spec.repeat_index}.json"
        return output_dir / spec.question_id / f"{spec.run_id}.json"

    def _load_existing_result(self, spec: RunSpec) -> dict | None:
        paths = [self._result_path(spec)]
        if self.config.resume_from_dir is not None:
            source_path = self._result_path_in(self.config.resume_from_dir, spec)
            if source_path not in paths:
                paths.append(source_path)
        for path in paths:
            if not path.exists():
                continue
            result = self._read_result(path)
            if result is not None:
                return result
        return None

    @staticmethod
    def _read_result(path: Path) -> dict | None:
        if not path.is_file():
            return None
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return result if isinstance(result, dict) else None

    @staticmethod
    def _completed_attempt_count(existing: dict | None) -> int:
        if not existing:
            return 0
        attempt_numbers: list[int] = []
        attempt = existing.get("attempt")
        if (
            existing.get("attempt_consumed", True) is not False
            and isinstance(attempt, int)
            and attempt > 0
        ):
            attempt_numbers.append(attempt)
        retry_attempts = existing.get("retry_attempts")
        if isinstance(retry_attempts, list):
            consuming_rows = 0
            for row in retry_attempts:
                if not isinstance(row, dict):
                    continue
                if row.get("attempt_consumed", True) is False:
                    continue
                consuming_rows += 1
                value = row.get("attempt")
                if isinstance(value, int) and value > 0:
                    attempt_numbers.append(value)
            if not attempt_numbers and consuming_rows:
                attempt_numbers.append(consuming_rows)
        return max(attempt_numbers, default=0)

    @staticmethod
    def _existing_attempts(existing: dict | None) -> list[dict]:
        if not existing:
            return []
        retry_attempts = existing.get("retry_attempts")
        if not isinstance(retry_attempts, list):
            return []
        return [dict(row) for row in retry_attempts if isinstance(row, dict)]

    @staticmethod
    def _existing_interruptions(existing: dict | None) -> list[dict]:
        if not existing:
            return []
        rows = existing.get("endpoint_interruptions")
        if not isinstance(rows, list):
            return []
        return [dict(row) for row in rows if isinstance(row, dict)]

    def _exception_result(self, spec: RunSpec, exc: Exception) -> dict:
        return {
            "question_id": spec.question_id,
            "repeat_index": spec.repeat_index,
            "run_id": spec.run_id,
            "question": spec.item.question,
            "answer": spec.item.answer,
            "type": spec.item.type,
            "status": "failed",
            "error": str(exc),
            "tool_calls": {},
            "policy_violations": [],
            "history": [],
        }
