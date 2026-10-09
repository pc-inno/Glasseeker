"""Shared filesystem and artifact helpers for split pipeline stages.

These helpers only adapt persistence and stage boundaries.  All generation,
verification, repair, and solver behavior remains in ``browsecomp_v2.workflow``.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import shutil
import sys
import threading
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, TextIO, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from browsecomp_v2.schema import (  # noqa: E402
    Artifact,
    ConstraintPath,
    SeedRecord,
    Target,
    VerifierReport,
)
from browsecomp_v2.workflow import save_artifact, save_seed  # noqa: E402


class _TeeTextIO:
    """Mirror a text stream to a shared, line-buffered stage log."""

    def __init__(self, primary: TextIO, log_file: TextIO, lock: threading.RLock):
        self._primary = primary
        self._log_file = log_file
        self._lock = lock

    def write(self, value: str) -> int:
        with self._lock:
            self._primary.write(value)
            self._log_file.write(value)
        return len(value)

    def flush(self) -> None:
        with self._lock:
            self._primary.flush()
            self._log_file.flush()

    def isatty(self) -> bool:
        return self._primary.isatty()

    def fileno(self) -> int:
        return self._primary.fileno()

    @property
    def encoding(self) -> str | None:
        return getattr(self._primary, "encoding", None)


class StageLogSession:
    """Own the process-wide stream tee installed for one stage invocation."""

    def __init__(self, *, log_path: Path, log_file: TextIO):
        self.log_path = log_path
        self._log_file = log_file
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._lock = threading.RLock()
        self._stdout_tee = _TeeTextIO(
            self._original_stdout, self._log_file, self._lock
        )
        self._stderr_tee = _TeeTextIO(
            self._original_stderr, self._log_file, self._lock
        )
        self._closed = False

    def install(self) -> None:
        sys.stdout = self._stdout_tee
        sys.stderr = self._stderr_tee

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._stdout_tee.flush()
            self._stderr_tee.flush()
            if sys.stdout is self._stdout_tee:
                sys.stdout = self._original_stdout
            if sys.stderr is self._stderr_tee:
                sys.stderr = self._original_stderr
            self._log_file.close()
            self._closed = True


def configure_stage_logging(batch_dir: Path, stage: str) -> StageLogSession:
    """Mirror this stage's console output into its output batch directory."""

    batch_dir = batch_dir.resolve()
    batch_dir.mkdir(parents=True, exist_ok=True)
    normalized_stage = safe_slug(stage).replace("-", "_")
    log_path = batch_dir / f"{normalized_stage}_stage.log"
    log_file = log_path.open("a", encoding="utf-8", buffering=1)
    session = StageLogSession(log_path=log_path, log_file=log_file)
    session.install()
    atexit.register(session.close)
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    print(
        f"stage_log_start: stage={normalized_stage} pid={os.getpid()} "
        f"started_at={started_at} path={log_path}",
        flush=True,
    )
    return session


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "").strip())
    slug = slug.strip("-").lower()
    return slug or "seed"


def seed_dir_name(index: int, entity_id: str) -> str:
    return f"seed_{int(index):03d}_{safe_slug(entity_id)}"


def configure_stage_environment(batch_dir: Path) -> None:
    """Route stage conversations and downloaded tool files into its batch."""
    import os

    batch_dir = batch_dir.resolve()
    batch_dir.mkdir(parents=True, exist_ok=True)
    os.environ["V2_OUTPUT_DIR"] = str(batch_dir)
    os.environ["V2_CONVERSATION_LOG_DIR"] = str(batch_dir / "conversations")
    os.environ["V2_HERMES_TOOL_WORKSPACE_ROOT"] = str(batch_dir / "tool_workspaces")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def copy_json(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def load_seed_files(batch_dir: Path) -> list[Tuple[int, Path, SeedRecord]]:
    records: list[Tuple[int, Path, SeedRecord]] = []
    for fallback, seed_path in enumerate(
        sorted(batch_dir.glob("seed_*/seed.json")), start=1
    ):
        match = re.match(r"seed_(\d+)_", seed_path.parent.name)
        index = int(match.group(1)) if match else fallback
        records.append((index, seed_path, SeedRecord.from_dict(read_json(seed_path))))
    return records


def artifact_from_expansion(data: Dict[str, Any], *, run_context: Dict[str, Any]) -> Artifact:
    """Rehydrate only the fields needed by the existing question/solver stage."""
    raw_target = data.get("target")
    if not isinstance(raw_target, dict):
        raise ValueError("expansion artifact has no target object")
    target = Target(**raw_target)
    constraints = [
        ConstraintPath.from_dict(item)
        for item in data.get("constraints", [])
        if isinstance(item, dict)
    ]
    raw_verifier = data.get("verifier")
    verifier = None
    if isinstance(raw_verifier, dict):
        raw_candidates = raw_verifier.get("single_path_candidates")
        if not isinstance(raw_candidates, dict):
            raw_candidates = {}
        verifier = VerifierReport(
            accepted=bool(raw_verifier.get("accepted")),
            reason=str(raw_verifier.get("reason") or ""),
            core_path_ids=[str(item) for item in raw_verifier.get("core_path_ids", [])],
            target_key=str(raw_verifier.get("target_key") or ""),
            single_path_candidates={
                str(key): [str(item) for item in value]
                for key, value in raw_candidates.items()
                if isinstance(value, list)
            },
            all_core_candidates=[
                str(item) for item in raw_verifier.get("all_core_candidates", [])
            ],
            distractor_report=(
                dict(raw_verifier.get("distractor_report"))
                if isinstance(raw_verifier.get("distractor_report"), dict)
                else {}
            ),
        )
    return Artifact(
        target=target,
        constraints=constraints,
        verifier=verifier,
        root_ambiguity_report=(
            dict(data.get("root_ambiguity_report"))
            if isinstance(data.get("root_ambiguity_report"), dict)
            else {}
        ),
        notes=[str(item) for item in data.get("notes", [])],
        iterations=[
            dict(item) for item in data.get("iterations", []) if isinstance(item, dict)
        ],
        run_context=dict(run_context),
    )


def persist_stage_failure(
    *,
    seed: SeedRecord,
    output_dir: Path,
    index: int,
    status: str,
    reason: str,
    extra: Dict[str, Any] | None = None,
) -> Path:
    seed_dir = output_dir / seed_dir_name(index, seed.target.entity_id)
    artifact_path = seed_dir / "artifact.json"
    payload = {
        "target": asdict(seed.target),
        "constraints": [],
        "question": "",
        "status": status,
        "notes": [seed.seed_note] if seed.seed_note else [],
        "iterations": [
            {"stage": "seed", "status": "loaded", "target": asdict(seed.target)},
            {"stage": "stage_input", "status": "failed", "reason": reason},
        ],
        "run_context": {
            "seed_index": index,
            "seed_id": seed.target.entity_id,
            "seed_dir": str(seed_dir.resolve()),
            "artifact_path": str(artifact_path.resolve()),
        },
    }
    if extra:
        payload.update(extra)
    write_json(artifact_path, payload)
    save_seed(seed, seed_dir / "seed.json")
    return artifact_path


def save_seed_record(seed: SeedRecord, path: Path) -> None:
    save_seed(seed, path)


def save_artifact_json(artifact: Artifact, path: Path) -> None:
    save_artifact(artifact, path)
