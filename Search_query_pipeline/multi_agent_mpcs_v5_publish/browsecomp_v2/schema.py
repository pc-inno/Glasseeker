from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import re
from typing import Any, Dict, List, Optional


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _fallback_path_id(data: Dict[str, Any], role: str) -> str:
    base = str(
        data.get("id")
        or data.get("name")
        or data.get("branch")
        or data.get("root_branch")
        or data.get("clue")
        or role
    )
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", base.lower()).strip("_")[:80] or role
    digest_src = "|".join(
        str(data.get(key, ""))
        for key in ("role", "branch", "root_branch", "clue", "fuzzy_text")
    )
    digest = hashlib.sha1(digest_src.encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"{role}_{slug}_{digest}"


@dataclass
class Target:
    entity_id: str
    name: str
    entity_type: str
    answer_field: str
    answer: str
    description: str = ""
    source_urls: List[str] = field(default_factory=list)
    domain_family: str = ""
    domain_subtype: str = ""


@dataclass
class Evidence:
    url: str
    text: str = ""
    supports: str = ""
    source: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Evidence":
        return cls(
            url=str(data.get("url", data.get("source_url", ""))),
            text=str(data.get("text", data.get("snippet", data.get("claim", "")))),
            supports=str(data.get("supports", data.get("claim", ""))),
            source=str(data.get("source", "")),
        )


@dataclass
class ConstraintPath:
    path_id: str
    role: str
    clue: str
    candidates: List[str]
    evidence: List[Evidence] = field(default_factory=list)
    branch: str = ""
    hop_count: int = 1
    estimated_candidate_count: int = 0
    terminal_type: str = ""
    notes: str = ""
    local_target_id: str = ""
    local_target_name: str = ""
    local_target_canonical_name: str = ""
    local_target_type: str = ""
    local_constraints: List["ConstraintPath"] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConstraintPath":
        if not isinstance(data, dict):
            data = {"role": "core", "clue": str(data)}
        role = str(data.get("role", "core")).lower()
        if role not in {"core", "distractor"}:
            role = "core"
        clue = str(data.get("clue", data.get("fuzzy_text", "")))
        path_id = str(data.get("path_id", data.get("id", ""))).strip()
        if not path_id:
            path_id = _fallback_path_id(data, role)
        local_raw = data.get("local_constraints", data.get("sub_constraints", []))
        return cls(
            path_id=path_id,
            role=role,
            clue=clue,
            candidates=[str(item) for item in _as_list(data.get("candidates", []))],
            evidence=[Evidence.from_dict(item) for item in _as_list(data.get("evidence", [])) if isinstance(item, dict)],
            branch=str(data.get("branch", data.get("root_branch", ""))),
            hop_count=int(data.get("hop_count", 1)),
            estimated_candidate_count=int(data.get("estimated_candidate_count", 0) or 0),
            terminal_type=str(data.get("terminal_type", "")),
            notes=str(data.get("notes", "")),
            local_target_id=str(data.get("local_target_id", data.get("target_entity_id", ""))),
            local_target_name=str(data.get("local_target_name", data.get("target_entity_name", ""))),
            local_target_canonical_name=str(
                data.get("local_target_canonical_name", "")
            ),
            local_target_type=str(data.get("local_target_type", data.get("target_entity_type", ""))),
            local_constraints=[cls.from_dict(item) for item in _as_list(local_raw)],
        )


@dataclass
class SeedRecord:
    target: Target
    seed_note: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SeedRecord":
        target = Target(**data["target"])
        return cls(target=target, seed_note=str(data.get("seed_note", data.get("rationale", ""))))


@dataclass
class VerifierReport:
    accepted: bool
    reason: str
    core_path_ids: List[str]
    target_key: str
    single_path_candidates: Dict[str, List[str]]
    all_core_candidates: List[str]
    distractor_report: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SolverSummary:
    total: int
    correct: int
    incorrect: int
    accepted: bool
    status: str
    reason: str
    verification_failed: int = 0
    successful_api_calls: List[int] = field(default_factory=list)
    successful_tool_calls: List[int] = field(default_factory=list)
    min_success_api_calls: int = 0
    median_success_api_calls: float = 0.0
    max_success_api_calls: int = 0
    mean_success_api_calls: float = 0.0
    min_success_tool_calls: int = 0
    median_success_tool_calls: float = 0.0
    max_success_tool_calls: int = 0
    mean_success_tool_calls: float = 0.0
    below_target_api_successes: int = 0
    below_target_tool_successes: int = 0
    effort_relaxation_used: bool = False


@dataclass
class Artifact:
    target: Target
    constraints: List[ConstraintPath] = field(default_factory=list)
    verifier: Optional[VerifierReport] = None
    root_ambiguity_report: Dict[str, Any] = field(default_factory=dict)
    question: str = ""
    question_state: Dict[str, Any] = field(default_factory=dict)
    question_history: List[Dict[str, Any]] = field(default_factory=list)
    # Every Question/Repair candidate, including candidates rejected by the
    # program verifier, for audit and later training-quality review.
    question_attempts: List[Dict[str, Any]] = field(default_factory=list)
    question_repair_history: List[Dict[str, Any]] = field(default_factory=list)
    uniqueness_key: str = ""
    uniqueness_report: Dict[str, Any] = field(default_factory=dict)
    solver_reports: List[Dict[str, Any]] = field(default_factory=list)
    solver_summary: Optional[SolverSummary] = None
    solver_attempts: List[Dict[str, Any]] = field(default_factory=list)
    run_context: Dict[str, Any] = field(default_factory=dict)
    status: str = "created"
    notes: List[str] = field(default_factory=list)
    iterations: List[Dict[str, Any]] = field(default_factory=list)
