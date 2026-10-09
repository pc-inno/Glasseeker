from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from hermes_constants import get_hermes_home

SESSION_KIND_PRIMARY = "primary"
SESSION_KIND_SUBAGENT = "subagent"
SESSION_KIND_COMPRESSION = "compression"
SESSION_KIND_BRANCH = "branch"
SESSION_KIND_CHILD_UNKNOWN = "child_unknown"

KNOWN_SESSION_KINDS = {
    SESSION_KIND_PRIMARY,
    SESSION_KIND_SUBAGENT,
    SESSION_KIND_COMPRESSION,
    SESSION_KIND_BRANCH,
    SESSION_KIND_CHILD_UNKNOWN,
}

SUBAGENT_SESSION_KINDS = {
    SESSION_KIND_SUBAGENT,
}

_SESSION_PATH_SEGMENT_RE = re.compile(r"[^A-Za-z0-9._=-]+")


def _sanitize_session_path_segment(value: Optional[str]) -> Optional[str]:
    """Return a filesystem-safe session grouping segment."""
    raw = str(value or "").strip()
    if not raw:
        return None
    raw = raw.replace("\\", "_").replace("/", "_")
    segment = _SESSION_PATH_SEGMENT_RE.sub("_", raw).strip("._-")
    if not segment or segment in {".", ".."}:
        return None
    return segment[:160]


def _session_group_segments(
    *,
    model_name: Optional[str] = None,
    task_name: Optional[str] = None,
) -> list[str]:
    """Return JSON snapshot grouping subdirectories requested by env/caller.

    Evaluation scripts can set, for example:

        HERMES_SESSION_MODEL=claude-opus-4-8-thinking
        HERMES_SESSION_DATASET=simple_search_2
        HERMES_SESSION_TASK_ID=task_001

    JSON snapshots are then routed under
    ``sessions/<model>/<dataset>/<task>/``. ``HERMES_SESSION_TASK_TYPE``
    remains supported as the dataset segment for older scripts.
    """
    model = _sanitize_session_path_segment(
        os.environ.get("HERMES_SESSION_MODEL")
        or os.environ.get("HERMES_SESSION_SAVE_NAME")
        or model_name
    )
    dataset = _sanitize_session_path_segment(
        os.environ.get("HERMES_SESSION_DATASET")
        or os.environ.get("HERMES_SESSION_DATASET_NAME")
        or os.environ.get("HERMES_SESSION_TASK_TYPE")
        or os.environ.get("HERMES_SESSION_TASK")
    )
    task = _sanitize_session_path_segment(
        task_name
        or os.environ.get("HERMES_SESSION_TASK_NAME")
        or os.environ.get("HERMES_SESSION_TASK_ID")
        or os.environ.get("HERMES_TASK_ID")
    )
    return [segment for segment in (model, dataset, task) if segment]


def get_repo_runtime_root(*, repo_root: Optional[Path] = None) -> Path:
    """Return the runtime directory for session artifacts."""
    override = os.environ.get("HERMES_SESSION_STORAGE_ROOT", "").strip()
    if override:
        override_path = Path(override)
        if override_path.name == "sessions":
            return override_path.parent
        return override_path
    if os.environ.get("PYTEST_CURRENT_TEST"):
        hermes_home = os.environ.get("HERMES_HOME", "").strip()
        if hermes_home:
            return Path(hermes_home)
    if repo_root is not None:
        return Path(repo_root) / ".hermes"
    return get_hermes_home()


def _get_sessions_base(*, runtime_root: Optional[Path] = None) -> Path:
    """Return the ungrouped sessions directory."""
    if runtime_root is not None:
        return Path(runtime_root) / "sessions"

    override = os.environ.get("HERMES_SESSION_STORAGE_ROOT", "").strip()
    if override:
        override_path = Path(override)
        if override_path.name == "sessions":
            return override_path
        return override_path / "sessions"

    return get_repo_runtime_root() / "sessions"


def get_state_db_path(*, runtime_root: Optional[Path] = None) -> Path:
    root = Path(runtime_root) if runtime_root is not None else get_repo_runtime_root()
    return root / "state.db"


def normalize_session_kind(
    session_kind: Optional[str],
    *,
    parent_session_id: Optional[str] = None,
) -> str:
    kind = str(session_kind or "").strip().lower()
    if kind in KNOWN_SESSION_KINDS:
        return kind
    if parent_session_id:
        return SESSION_KIND_CHILD_UNKNOWN
    return SESSION_KIND_PRIMARY


def get_sessions_root(
    *,
    runtime_root: Optional[Path] = None,
    model_name: Optional[str] = None,
    task_name: Optional[str] = None,
) -> Path:
    root = _get_sessions_base(runtime_root=runtime_root)
    for segment in _session_group_segments(
        model_name=model_name,
        task_name=task_name,
    ):
        root = root / segment
    return root


def normalize_sessions_root(path: Optional[Path]) -> Path:
    if path is None:
        return get_sessions_root()
    candidate = Path(path)
    if (
        candidate.name in {"main", "subagents", "saved"}
        and "sessions" in candidate.parts
    ):
        return candidate.parent
    return candidate


def get_session_namespace_dir(
    session_kind: Optional[str],
    *,
    sessions_root: Optional[Path] = None,
    runtime_root: Optional[Path] = None,
) -> Path:
    root = normalize_sessions_root(sessions_root) if sessions_root else get_sessions_root(runtime_root=runtime_root)
    kind = normalize_session_kind(session_kind)
    namespace = "subagents" if kind in SUBAGENT_SESSION_KINDS else "main"
    return root / namespace


def get_session_log_path(
    session_id: str,
    session_kind: Optional[str],
    *,
    sessions_root: Optional[Path] = None,
    runtime_root: Optional[Path] = None,
) -> Path:
    log_dir = get_session_namespace_dir(
        session_kind,
        sessions_root=sessions_root,
        runtime_root=runtime_root,
    )
    return log_dir / f"session_{session_id}.json"


def iter_session_storage_dirs(sessions_root: Optional[Path]) -> list[Path]:
    if sessions_root is None:
        return []
    root = normalize_sessions_root(sessions_root)
    seen: list[Path] = []
    for candidate in (root, root / "main", root / "subagents"):
        if candidate not in seen:
            seen.append(candidate)
    return seen


def sync_agent_session_storage(agent) -> None:
    sessions_root = normalize_sessions_root(getattr(agent, "sessions_root_dir", None))
    log_dir = get_session_namespace_dir(
        getattr(agent, "session_kind", None),
        sessions_root=sessions_root,
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    agent.sessions_root_dir = sessions_root
    agent.logs_dir = log_dir
    agent.session_log_file = log_dir / f"session_{agent.session_id}.json"
