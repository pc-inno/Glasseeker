from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


def load_dotenv(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


@dataclass(frozen=True)
class AgentConfig:
    name: str
    api_key: str
    base_url: str
    model: str
    provider: str = "custom"
    backend: str = "hermes"
    api_mode: Optional[str] = None
    max_iterations: int = 12
    max_tokens: int = 16384
    enabled_toolsets: Optional[List[str]] = None

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)


@dataclass(frozen=True)
class WorkflowConfig:
    hermes_path: Optional[Path]
    hermes_command: Optional[str]
    output_dir: Path
    seed_concurrency: int
    seed_verifier_enabled: bool
    seed_repair_enabled: bool
    seed_repair_retries: int
    seed_verifier_rollouts: int
    seed_cardinality_rollouts: int
    root_ambiguity_verifier_enabled: bool
    root_require_source_diversity: bool
    uniqueness_enabled: bool
    uniqueness_revisions: int
    uniqueness_rollouts: int
    uniqueness_require_consensus: bool
    uniqueness_allow_uncertain_after_repair: bool
    blind_uniqueness_enabled: bool
    answer_cardinality_enabled: bool
    solver_enabled: bool
    solver_rollouts: int
    solver_concurrency: int
    solver_min_success_api_calls: int
    solver_min_success_tool_calls: int
    solver_relaxed_min_success_api_calls: int
    solver_relaxed_min_success_tool_calls: int
    solver_too_easy_api_calls: int
    solver_accept_api_calls: int
    solver_continue_pruning: bool
    solver_pruning_revisions: int
    solver_trace_summary_enabled: bool
    min_core_paths: int
    min_distractor_paths: int
    max_distractor_paths: int
    min_single_candidates: int
    max_single_candidates: int
    min_pairwise_core_intersection: int
    min_distractor_core_overlap: int
    min_distractors_per_core: int
    local_min_core_paths: int
    local_min_distractor_paths: int
    local_min_relation_core_paths: int
    local_min_attribute_core_paths: int
    local_max_depth: int
    local_core_max_depth: int
    local_distractor_max_depth: int
    local_expand_min_paths: int
    local_expand_max_paths: int
    local_expand_core_min_paths: int
    local_expand_core_max_paths: int
    local_expand_distractor_min_paths: int
    local_expand_distractor_max_paths: int
    local_expand_distractor_prob: float
    local_expand_roles: List[str]
    local_max_deep_paths: int
    local_max_deep_paths_distractor: int
    local_deep_path_roles: List[str]
    local_concurrency: int
    local_verification_retries: int
    local_quality_verifier_enabled: bool
    local_require_distinct_parent_child_sources: bool
    local_expand_all_root_cores: bool
    local_min_non_shortcut_root_cores: int
    require_local_verification: bool
    require_local_expansion: bool
    question_min_leaf_depth: int
    question_min_leaf_core_paths: int
    question_min_distractor_paths: int
    question_min_root_core_children: int
    question_min_root_non_shortcut_children: int
    question_require_all_core_leaf: bool
    question_forbid_core_root_fallback: bool
    question_allow_overcomplete_roots: bool
    question_repair_enabled: bool
    question_verifier_enabled: bool
    question_validation_repair_attempts: int
    question_max_words: int
    verbose_progress: bool
    max_revisions: int
    question_revisions: int
    max_rounds: int
    agents: Dict[str, AgentConfig]
    # In v5, pre-solver uniqueness is an audit signal; Solver/adjudication owns
    # the public ambiguity decision.
    uniqueness_defer_to_solver: bool = False
    # Root uniqueness may be established by the full core set while relation
    # anchors outside a minimal subset remain available for Local expansion.
    root_require_expandable_unique_bundle: bool = False
    # Optional strict Root construction gate.  Keep disabled for the default
    # v5 flow; when enabled, every core pair must retain a real alternative.
    root_pairwise_gate_enabled: bool = False
    # Optional Question gate: keep structural/tree checks, and defer semantic
    # wording and uniqueness decisions to Solver plus ambiguity adjudication.
    question_structural_only: bool = False


def load_config(env_path: str | Path = ".env") -> WorkflowConfig:
    load_dotenv(env_path)
    project_root = Path(__file__).resolve().parents[1]
    bundled_hermes = project_root / "hermes"
    bundled_command = bundled_hermes / "hermes"
    hermes_raw = os.environ.get("V2_HERMES_PATH", os.environ.get("HERMES_PATH", "")).strip()
    hermes_command = (
        os.environ.get("V2_HERMES_COMMAND", os.environ.get("HERMES_COMMAND", "")).strip()
        or (str(bundled_command) if bundled_command.is_file() else "")
        or shutil.which("hermes")
    )
    if hermes_command == str(bundled_command):
        os.environ.setdefault("HERMES_VENV", str(bundled_hermes / ".venv"))
        os.environ.setdefault("HERMES_HOME", str(project_root / ".hermes"))
    hermes_path = None
    if hermes_raw:
        candidate = Path(hermes_raw).expanduser().resolve()
        if candidate.is_dir() and (candidate / "run_agent.py").exists():
            hermes_path = candidate
        elif candidate.exists():
            hermes_command = str(candidate)
        else:
            hermes_path = candidate
    local_max_depth = _int_env("V2_LOCAL_MAX_DEPTH", 5)
    return WorkflowConfig(
        hermes_path=hermes_path,
        hermes_command=hermes_command,
        output_dir=Path(os.environ.get("V2_OUTPUT_DIR", "data/runs")),
        seed_concurrency=_int_env("V2_SEED_CONCURRENCY", 4),
        seed_verifier_enabled=_bool_env("V2_SEED_VERIFIER_ENABLED", True),
        seed_repair_enabled=_bool_env("V2_SEED_REPAIR_ENABLED", True),
        seed_repair_retries=max(0, _int_env("V2_SEED_REPAIR_RETRIES", 1)),
        seed_verifier_rollouts=max(1, _int_env("V2_SEED_VERIFIER_ROLLOUTS", 2)),
        seed_cardinality_rollouts=max(
            1, _int_env("V2_SEED_CARDINALITY_ROLLOUTS", 1)
        ),
        root_ambiguity_verifier_enabled=_bool_env(
            "V2_ROOT_AMBIGUITY_VERIFIER_ENABLED", True
        ),
        root_require_source_diversity=_bool_env(
            "V2_ROOT_REQUIRE_SOURCE_DIVERSITY", False
        ),
        uniqueness_enabled=_bool_env("V2_UNIQUENESS_ENABLED", True),
        uniqueness_revisions=max(0, _int_env("V2_UNIQUENESS_REVISIONS", 2)),
        uniqueness_rollouts=max(1, _int_env("V2_UNIQUENESS_ROLLOUTS", 1)),
        uniqueness_require_consensus=_bool_env(
            "V2_UNIQUENESS_REQUIRE_CONSENSUS", True
        ),
        uniqueness_allow_uncertain_after_repair=_bool_env(
            "V2_UNIQUENESS_ALLOW_UNCERTAIN_AFTER_REPAIR", True
        ),
        blind_uniqueness_enabled=_bool_env("V2_BLIND_UNIQUENESS_ENABLED", True),
        answer_cardinality_enabled=_bool_env(
            "V2_ANSWER_CARDINALITY_ENABLED", True
        ),
        solver_enabled=_bool_env("V2_SOLVER_ENABLED", True),
        solver_rollouts=max(3, _int_env("V2_SOLVER_ROLLOUTS", 3)),
        solver_concurrency=_int_env("V2_SOLVER_CONCURRENCY", 3),
        solver_min_success_api_calls=max(
            1, _int_env("V2_SOLVER_MIN_SUCCESS_API_CALLS", 20)
        ),
        solver_min_success_tool_calls=max(
            1, _int_env("V2_SOLVER_MIN_SUCCESS_TOOL_CALLS", 40)
        ),
        solver_relaxed_min_success_api_calls=max(
            1, _int_env("V2_SOLVER_RELAXED_MIN_SUCCESS_API_CALLS", 18)
        ),
        solver_relaxed_min_success_tool_calls=max(
            1, _int_env("V2_SOLVER_RELAXED_MIN_SUCCESS_TOOL_CALLS", 36)
        ),
        solver_too_easy_api_calls=max(
            0, _int_env("V2_SOLVER_TOO_EASY_API_CALLS", 5)
        ),
        # A positive value enables the legacy API-only acceptance shortcut.
        # Keep it opt-in so API and tool effort are evaluated independently by
        # the 20/40 full and 18/36 relaxed thresholds.
        solver_accept_api_calls=max(
            0, _int_env("V2_SOLVER_ACCEPT_API_CALLS", 0)
        ),
        solver_continue_pruning=_bool_env(
            "V2_SOLVER_CONTINUE_PRUNING", True
        ),
        solver_pruning_revisions=max(
            0, _int_env("V2_SOLVER_PRUNING_REVISIONS", 1)
        ),
        solver_trace_summary_enabled=_bool_env(
            "V2_SOLVER_TRACE_SUMMARY_ENABLED", True
        ),
        min_core_paths=_int_env("V2_MIN_CORE_PATHS", 3),
        min_distractor_paths=_int_env("V2_MIN_DISTRACTOR_PATHS", 1),
        max_distractor_paths=_int_env("V2_MAX_DISTRACTOR_PATHS", 3),
        min_single_candidates=_int_env("V2_MIN_SINGLE_CANDIDATES", 2),
        max_single_candidates=_int_env("V2_MAX_SINGLE_CANDIDATES", 500),
        min_pairwise_core_intersection=_int_env("V2_MIN_PAIRWISE_CORE_INTERSECTION", 0),
        min_distractor_core_overlap=_int_env("V2_MIN_DISTRACTOR_CORE_OVERLAP", 0),
        min_distractors_per_core=_int_env("V2_MIN_DISTRACTORS_PER_CORE", 0),
        local_min_core_paths=max(2, _int_env("V2_LOCAL_MIN_CORE_PATHS", 3)),
        local_min_distractor_paths=max(0, _int_env("V2_LOCAL_MIN_DISTRACTOR_PATHS", 1)),
        local_min_relation_core_paths=max(
            1, _int_env("V2_LOCAL_MIN_RELATION_CORE_PATHS", 2)
        ),
        local_min_attribute_core_paths=max(
            1, _int_env("V2_LOCAL_MIN_ATTRIBUTE_CORE_PATHS", 1)
        ),
        local_max_depth=local_max_depth,
        local_core_max_depth=_int_env("V2_LOCAL_CORE_MAX_DEPTH", local_max_depth),
        local_distractor_max_depth=_int_env("V2_LOCAL_DISTRACTOR_MAX_DEPTH", local_max_depth),
        local_expand_min_paths=_int_env("V2_LOCAL_EXPAND_MIN_PATHS", 0),
        local_expand_max_paths=_int_env("V2_LOCAL_EXPAND_MAX_PATHS", 0),
        local_expand_core_min_paths=_int_env(
            "V2_LOCAL_EXPAND_CORE_MIN_PATHS",
            _int_env("V2_LOCAL_EXPAND_MIN_PATHS", 0),
        ),
        local_expand_core_max_paths=_int_env(
            "V2_LOCAL_EXPAND_CORE_MAX_PATHS",
            _int_env("V2_LOCAL_EXPAND_MAX_PATHS", 0),
        ),
        local_expand_distractor_min_paths=_int_env("V2_LOCAL_EXPAND_DISTRACTOR_MIN_PATHS", 0),
        local_expand_distractor_max_paths=_int_env("V2_LOCAL_EXPAND_DISTRACTOR_MAX_PATHS", 0),
        local_expand_distractor_prob=_float_env("V2_LOCAL_EXPAND_DISTRACTOR_PROB", 1.0),
        local_expand_roles=_roles_env("V2_LOCAL_EXPAND_ROLES", ["core", "distractor"]),
        local_max_deep_paths=_int_env("V2_LOCAL_MAX_DEEP_PATHS", 1),
        local_max_deep_paths_distractor=_int_env("V2_LOCAL_MAX_DEEP_PATHS_DISTRACTOR", _int_env("V2_LOCAL_MAX_DEEP_PATHS", 1)),
        local_deep_path_roles=_roles_env("V2_LOCAL_DEEP_PATH_ROLES", ["core"]),
        local_concurrency=_int_env("V2_LOCAL_CONCURRENCY", 6),
        local_verification_retries=max(
            0,
            _int_env("V2_LOCAL_VERIFICATION_RETRIES", 2),
        ),
        local_quality_verifier_enabled=_bool_env(
            "V2_LOCAL_QUALITY_VERIFIER_ENABLED",
            _bool_env("V2_LOCAL_SEMANTIC_VERIFIER_ENABLED", True),
        ),
        local_require_distinct_parent_child_sources=_bool_env(
            "V2_LOCAL_REQUIRE_DISTINCT_PARENT_CHILD_SOURCES", False
        ),
        local_expand_all_root_cores=_bool_env(
            "V2_LOCAL_EXPAND_ALL_ROOT_CORES", False
        ),
        local_min_non_shortcut_root_cores=max(
            0, _int_env("V2_LOCAL_MIN_NON_SHORTCUT_ROOT_CORES", 0)
        ),
        require_local_verification=_bool_env("V2_REQUIRE_LOCAL_VERIFICATION", True),
        require_local_expansion=_bool_env("V2_REQUIRE_LOCAL_EXPANSION", False),
        question_min_leaf_depth=_int_env("V2_QUESTION_MIN_LEAF_DEPTH", 2),
        question_min_leaf_core_paths=_int_env("V2_QUESTION_MIN_LEAF_CORE_PATHS", 1),
        question_min_distractor_paths=max(0, _int_env("V2_QUESTION_MIN_DISTRACTOR_PATHS", 0)),
        question_min_root_core_children=max(
            1, _int_env("V2_QUESTION_MIN_ROOT_CORE_CHILDREN", 1)
        ),
        question_min_root_non_shortcut_children=max(
            0, _int_env("V2_QUESTION_MIN_ROOT_NON_SHORTCUT_CHILDREN", 0)
        ),
        question_require_all_core_leaf=_bool_env("V2_QUESTION_REQUIRE_ALL_CORE_LEAF", False),
        question_forbid_core_root_fallback=_bool_env("V2_QUESTION_FORBID_CORE_ROOT_FALLBACK", True),
        question_allow_overcomplete_roots=_bool_env(
            "V2_QUESTION_ALLOW_OVERCOMPLETE_ROOTS", False
        ),
        question_repair_enabled=_bool_env("V2_QUESTION_REPAIR_ENABLED", True),
        question_verifier_enabled=_bool_env("V2_QUESTION_VERIFIER_ENABLED", True),
        question_validation_repair_attempts=max(
            0, _int_env("V2_QUESTION_VALIDATION_REPAIR_ATTEMPTS", 2)
        ),
        question_max_words=_int_env("V2_QUESTION_MAX_WORDS", 220),
        verbose_progress=_bool_env("V2_VERBOSE_PROGRESS", True),
        max_revisions=_int_env("V2_MAX_REVISIONS", 2),
        question_revisions=_int_env("V2_QUESTION_REVISIONS", 3),
        max_rounds=_int_env("V2_MAX_ROUNDS", 1),
        agents={
            "seed": _agent("SEED"),
            "seed_verifier": _agent(
                "SEED_VERIFIER", fallback="SEED", default_iterations=20
            ),
            "seed_repair": _agent(
                "SEED_REPAIR", fallback="SEED_VERIFIER", default_iterations=12
            ),
            "constraint": _agent("CONSTRAINT"),
            "local_constraint": _agent("LOCAL_CONSTRAINT", fallback="CONSTRAINT"),
            "question": _agent("QUESTION"),
            "question_repair": _agent("QUESTION_REPAIR", fallback="QUESTION"),
            "question_fallback": _agent("QUESTION_FALLBACK", fallback="QUESTION"),
            "question_verifier": _agent("QUESTION_VERIFIER", fallback="SOLVER_VERIFIER"),
            "trajectory_summary": _agent(
                "TRAJECTORY_SUMMARY", fallback="QUESTION_REPAIR"
            ),
            "solver_ambiguity": _agent(
                "SOLVER_AMBIGUITY", fallback="QUESTION_FALLBACK", default_iterations=18
            ),
            "uniqueness": _agent("UNIQUENESS", fallback="SOLVER_VERIFIER"),
            "answer_cardinality": _agent(
                "ANSWER_CARDINALITY",
                fallback="SOLVER_AMBIGUITY",
                default_iterations=18,
            ),
            "solver": _agent("SOLVER", default_iterations=24),
            "solver_verifier": _agent("SOLVER_VERIFIER"),
        },
        uniqueness_defer_to_solver=_bool_env(
            "V2_UNIQUENESS_DEFER_TO_SOLVER", False
        ),
        root_require_expandable_unique_bundle=_bool_env(
            "V2_ROOT_REQUIRE_EXPANDABLE_UNIQUE_BUNDLE", True
        ),
        root_pairwise_gate_enabled=_bool_env(
            "V2_ROOT_PAIRWISE_GATE_ENABLED", False
        ),
        question_structural_only=_bool_env(
            "V2_QUESTION_STRUCTURAL_ONLY", False
        ),
    )


def _agent(prefix: str, *, fallback: str = "DEFAULT", default_iterations: int = 12) -> AgentConfig:
    upper = prefix.upper()
    fallback_upper = fallback.upper()
    if f"{upper}_TOOLSETS" in os.environ:
        toolsets_raw = os.environ.get(f"{upper}_TOOLSETS", "")
    elif f"{fallback_upper}_TOOLSETS" in os.environ:
        toolsets_raw = os.environ.get(f"{fallback_upper}_TOOLSETS", "")
    else:
        toolsets_raw = os.environ.get("DEFAULT_TOOLSETS", "")
    return AgentConfig(
        name=prefix.lower(),
        api_key=(
            os.environ.get(f"{upper}_API_KEY")
            or os.environ.get(f"{fallback_upper}_API_KEY")
            or os.environ.get("DEFAULT_API_KEY", "")
        ),
        base_url=(
            os.environ.get(f"{upper}_BASE_URL")
            or os.environ.get(f"{fallback_upper}_BASE_URL")
            or os.environ.get("DEFAULT_BASE_URL", "")
        ),
        model=(
            os.environ.get(f"{upper}_MODEL")
            or os.environ.get(f"{fallback_upper}_MODEL")
            or os.environ.get("DEFAULT_MODEL", "")
        ),
        provider=(
            os.environ.get(f"{upper}_PROVIDER")
            or os.environ.get(f"{fallback_upper}_PROVIDER")
            or os.environ.get("DEFAULT_PROVIDER", "custom")
        ),
        backend=(
            os.environ.get(f"{upper}_BACKEND")
            or os.environ.get(f"{fallback_upper}_BACKEND")
            or os.environ.get("DEFAULT_BACKEND", "hermes")
        ).lower(),
        api_mode=_api_mode_env(
            upper,
            fallback_upper,
        ),
        max_iterations=_int_env(f"{upper}_MAX_ITERATIONS", default_iterations),
        max_tokens=_int_env(
            f"{upper}_MAX_TOKENS",
            _int_env(
                f"{fallback_upper}_MAX_TOKENS",
                _int_env("DEFAULT_MAX_TOKENS", 16384),
            ),
        ),
        enabled_toolsets=[item.strip() for item in toolsets_raw.split(",") if item.strip()] or None,
    )


def _api_mode_env(prefix: str, fallback: str) -> Optional[str]:
    value = (
        os.environ.get(f"{prefix}_API_MODE")
        or os.environ.get(f"{fallback}_API_MODE")
        or os.environ.get("DEFAULT_API_MODE", "")
    ).strip().lower()
    if not value:
        return None
    allowed = {
        "chat_completions",
        "codex_responses",
        "anthropic_messages",
        "bedrock_converse",
        "codex_app_server",
    }
    if value not in allowed:
        raise ValueError(
            f"invalid API mode {value!r} for {prefix}; expected one of {sorted(allowed)}"
        )
    return value


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw else default


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.lower() in {"1", "true", "yes", "y", "on"}


def _roles_env(name: str, default: List[str]) -> List[str]:
    raw = os.environ.get(name)
    if not raw:
        return default
    roles = []
    for item in raw.split(","):
        role = item.strip().lower()
        if role in {"core", "distractor"} and role not in roles:
            roles.append(role)
    return roles or default
