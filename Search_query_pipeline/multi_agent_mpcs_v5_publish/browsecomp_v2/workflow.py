from __future__ import annotations

import hashlib
import json
import os
import re
import time
import traceback
import unicodedata
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from datetime import date
from itertools import combinations
from pathlib import Path
from statistics import median
from threading import Lock, local
from typing import Any, Dict, Iterable, List, Sequence
from urllib.parse import urlparse

from .config import WorkflowConfig
from .diversity import DOMAIN_FAMILIES, normalize_domain, target_domain
from .prompts import (
    ANSWER_CARDINALITY_PROMPT,
    BLIND_QUESTION_RESOLUTION_PROMPT,
    CONSTRAINT_PROMPT,
    LOCAL_CONSTRAINT_PROMPT,
    LOCAL_QUALITY_PROMPT,
    QUESTION_PROMPT,
    QUESTION_REPAIR_PROMPT,
    QUESTION_VERIFIER_PROMPT,
    ROOT_PAIRWISE_AMBIGUITY_PROMPT,
    ROOT_PAIRWISE_CONSTRAINT_PROMPT,
    ROOT_AMBIGUITY_PROMPT,
    SEED_FACT_VERIFIER_PROMPT,
    SEED_REPAIR_PROMPT,
    SEED_PROMPT,
    SOLVER_RESEARCH_GUIDANCE,
    SOLVER_ALTERNATIVE_ADJUDICATION_PROMPT,
    SOLVER_TRAJECTORY_SUMMARY_PROMPT,
    SOLVER_VERIFIER_PROMPT,
    UNIQUENESS_PROMPT,
)
from .runner import AgentRunner, ProviderRateLimitError
from .schema import Artifact, ConstraintPath, SeedRecord, SolverSummary, Target, VerifierReport
from .verifier import verify_core_paths, verify_pairwise_core_paths


@dataclass
class LocalExpansionState:
    deep_paths: int = 0
    deep_paths_per_root: Dict[str, int] = field(default_factory=dict)
    root_roles: Dict[str, str] = field(default_factory=dict)
    lock: Lock = field(default_factory=Lock)


def _parse_solver_verifier_correct(response: Any) -> bool:
    """Apply Browse_comp_eval's judge score semantics."""
    parsed = response
    if isinstance(response, str):
        parsed = json.loads(response)
    if not isinstance(parsed, dict):
        raise ValueError("solver verifier response must be a JSON object")
    if "score" not in parsed:
        raise ValueError("solver verifier response did not contain score")
    return parsed["score"] == 1


def _validate_seed_fact_response(response: Any) -> None:
    if not isinstance(response, dict):
        raise ValueError("seed fact verifier response must be a JSON object")
    for key in (
        "accepted",
        "target_found",
        "answer_field_supported",
        "answer_matches",
    ):
        if not isinstance(response.get(key), bool):
            raise ValueError(f"seed fact verifier field {key!r} must be boolean")
    if not str(response.get("reason") or "").strip():
        raise ValueError("seed fact verifier must provide a reason")
    source_urls = response.get("source_urls")
    if not isinstance(source_urls, list) or not any(str(url).strip() for url in source_urls):
        raise ValueError("seed fact verifier must cite at least one source URL")


def _seed_answer_field_problem(target: Target) -> str:
    """Reject lookup keys disguised as answer-field schema labels."""
    field_text = str(target.answer_field or "").strip()
    if not field_text:
        return "answer_field is empty"
    words = re.findall(r"[\w'-]+", field_text, flags=re.UNICODE)
    if len(words) > 10:
        return "answer_field must be a concise semantic label of at most 10 words"
    if re.search(
        r"\b(?:1[0-9]{3}|20[0-9]{2}|january|february|march|april|may|june|"
        r"july|august|september|october|november|december)\b",
        field_text,
        flags=re.IGNORECASE,
    ):
        return "answer_field contains an exact date/year lookup key"
    normalized_field = _entity_text_key(field_text)
    protected_values = [target.name, target.entity_id]
    protected_values.extend(_answer_text_aliases(target))
    for value in protected_values:
        normalized_value = _entity_text_key(value)
        if len(normalized_value) >= 4 and normalized_value in normalized_field:
            return "answer_field contains target- or answer-specific identifying text"
    return ""


def _validate_local_constraint_response(response: Any) -> None:
    """Validate the minimal Local Expansion response contract."""
    if not isinstance(response, dict):
        raise ValueError("local expansion response must be a JSON object")
    action = str(response.get("action") or "").strip().lower()
    if action not in {"expand", "stop"}:
        raise ValueError("local expansion action must be 'expand' or 'stop'")
    if action == "stop":
        return
    target = response.get("local_target")
    if not isinstance(target, dict):
        raise ValueError("expanded local response must contain local_target")
    if not str(target.get("surface_text") or "").strip():
        raise ValueError("local_target.surface_text is required")
    if not str(target.get("entity_id") or "").strip():
        raise ValueError("local_target.entity_id is required")
    constraints = response.get("local_constraints")
    if not isinstance(constraints, list) or not constraints:
        raise ValueError("expanded local response must contain local_constraints")
    if any(not isinstance(item, dict) for item in constraints):
        raise ValueError("local_constraints must contain only JSON objects")


def _validate_root_constraint_response(response: Any) -> None:
    """Validate the Root Constraint response without accepting Local-owned fields."""
    if not isinstance(response, dict):
        raise ValueError("root constraint response must be a JSON object")
    constraints = response.get("constraints")
    if not isinstance(constraints, list) or not constraints:
        raise ValueError("root constraint response must contain constraints")
    if any(not isinstance(item, dict) for item in constraints):
        raise ValueError("root constraints must contain only JSON objects")
    for item in constraints:
        if not str(item.get("path_id") or "").strip():
            raise ValueError("every root constraint requires path_id")
        if not str(item.get("clue") or "").strip():
            raise ValueError("every root constraint requires clue")


def _validate_root_ambiguity_response(response: Any) -> None:
    if not isinstance(response, dict):
        raise ValueError("root ambiguity response must be a JSON object")
    results = response.get("path_results")
    if not isinstance(results, list) or not results:
        raise ValueError("root ambiguity response must contain path_results")
    if any(not isinstance(item, dict) for item in results):
        raise ValueError("root ambiguity path_results must contain objects")
    if not isinstance(response.get("joint_result"), dict):
        raise ValueError("root ambiguity response must contain joint_result")


def _validate_root_pairwise_ambiguity_response(response: Any) -> None:
    """Validate the stricter optional Root pairwise response contract."""
    _validate_root_ambiguity_response(response)
    pair_results = response.get("pair_results")
    if not isinstance(pair_results, list) or any(
        not isinstance(item, dict) for item in pair_results
    ):
        raise ValueError("pairwise root ambiguity response must contain pair_results")
    for item in pair_results:
        path_ids = item.get("path_ids")
        if not isinstance(path_ids, list) or len(path_ids) != 2 or any(
            not str(path_id).strip() for path_id in path_ids
        ):
            raise ValueError("each pairwise result requires two path_ids")


def _validate_trajectory_summary_response(response: Any) -> None:
    if not isinstance(response, dict):
        raise ValueError("trajectory summary must be a JSON object")
    if str(response.get("diagnosis") or "") not in {
        "target_band",
        "shortcut",
        "too_hard",
        "invalid_or_ambiguous",
        "tool_failure",
    }:
        raise ValueError("trajectory summary diagnosis is invalid")
    if str(response.get("diagnosis") or "") == "invalid_or_ambiguous":
        alternatives = response.get("verified_alternatives")
        if not isinstance(alternatives, list) or not alternatives:
            raise ValueError(
                "ambiguous trajectory diagnosis requires a verified alternative"
            )
        for item in alternatives:
            if (
                not isinstance(item, dict)
                or not str(item.get("name") or "").strip()
                or not str(item.get("answer") or "").strip()
                or item.get("matches_all_clues") is not True
                or not isinstance(item.get("clue_checks"), list)
                or not item.get("clue_checks")
                or not isinstance(item.get("source_urls"), list)
                or not any(str(url).strip() for url in item.get("source_urls", []))
            ):
                raise ValueError(
                    "every verified alternative must match all clues and cite sources"
                )


def _public_question_clauses(question: str) -> List[Dict[str, str]]:
    parts = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", str(question or "").strip())
        if part.strip()
    ]
    declarative = [part.rstrip(" .") for part in parts if "?" not in part]
    if not declarative and str(question or "").strip():
        declarative = [str(question).rsplit("?", 1)[0].strip().rstrip(" .")]
    return [
        {"clause_id": f"c{index}", "text": text}
        for index, text in enumerate(declarative, start=1)
        if text
    ]


def _clue_checks_cover_clauses(
    clue_checks: Any,
    required_clause_ids: Sequence[str],
) -> bool:
    required = {str(item) for item in required_clause_ids if str(item)}
    if not required:
        return isinstance(clue_checks, list) and bool(clue_checks)
    if not isinstance(clue_checks, list):
        return False
    covered: set[str] = set()
    for check in clue_checks:
        if isinstance(check, dict):
            clause_id = str(check.get("clause_id") or "").strip().lower()
            if clause_id:
                covered.add(clause_id)
            continue
        match = re.match(r"\s*\[(c\d+)\]", str(check), flags=re.IGNORECASE)
        if match:
            covered.add(match.group(1).lower())
    return required <= covered


def _validate_solver_ambiguity_response(
    response: Any,
    required_clause_ids: Sequence[str] = (),
) -> None:
    if not isinstance(response, dict):
        raise ValueError("solver ambiguity response must be a JSON object")
    if str(response.get("verdict") or "") not in {
        "verified_ambiguity",
        "no_verified_alternative",
        "uncertain",
    }:
        raise ValueError("solver ambiguity verdict is invalid")
    alternatives = response.get("verified_alternatives")
    if not isinstance(alternatives, list):
        raise ValueError("solver ambiguity verified_alternatives must be a list")
    for item in alternatives:
        if (
            not isinstance(item, dict)
            or not str(item.get("name") or "").strip()
            or item.get("matches_all_clues") is not True
            or not isinstance(item.get("clue_checks"), list)
            or not item.get("clue_checks")
            or not _clue_checks_cover_clauses(
                item.get("clue_checks"), required_clause_ids
            )
            or not isinstance(item.get("source_urls"), list)
            or not any(str(url).strip() for url in item.get("source_urls", []))
        ):
            raise ValueError(
                "verified solver alternative must match all clues and cite checks"
            )


def _solver_user_content(artifact: Artifact) -> str:
    """Build the natural-language solver request used by the previous pipeline."""
    task_type = artifact.target.entity_type or artifact.target.domain_family or "browse_comp"
    return (
        "You are running one browse-comparison evaluation task.\n"
        "Use only the tools made available in this Hermes session.\n"
        "Do not use or mention the reference answer. It is reserved for offline evaluation.\n"
        "If the task asks you to create a file, save it in the current working directory.\n\n"
        f"{SOLVER_RESEARCH_GUIDANCE}\n\n"
        f"Task type: {task_type}\n"
        f"Question:\n{artifact.question}\n\n"
        "After receiving tool results, carefully reflect on their quality and determine "
        "optimal next steps before proceeding. Use your thinking to plan and iterate based "
        "on this new information, and then take the best next action.\n\n"
        "Return the final answer clearly."
    )


def _solver_judge_user_content(
    question: str,
    model_response: Any,
    reference_answer: str,
) -> str:
    """Build Browse_comp_eval's answer-equivalence judge request."""
    return (
        SOLVER_VERIFIER_PROMPT
        + "\n"
        + f"[Question]: {question}\n"
        + f"[Standard Answer]: {reference_answer}\n"
        + f"[Model Answer]: {model_response}\n"
    )


class BrowseCompV2Workflow:
    def __init__(self, config: WorkflowConfig):
        self.config = config
        self.runner = AgentRunner(config.hermes_path, config.hermes_command)
        self._local_parallel_state = local()

    def produce_seeds(
        self,
        *,
        domain: str = "auto",
        target_type: str = "auto",
        answer_field: str = "auto",
        num_seeds: int = 5,
        avoid_entities: Sequence[str] = (),
        avoid_answers: Sequence[str] = (),
        domain_slots: Sequence[str] = (),
        existing_domain_counts: Dict[str, int] | None = None,
        overused_source_domains: Sequence[str] = (),
        apply_diversity_filter: bool = True,
    ) -> List[SeedRecord]:
        self._log(
            "seed_agent_start: "
            f"domain={domain} target_type={target_type} answer_field={answer_field} "
            f"requested={num_seeds} avoid_entities={len(avoid_entities)} "
            f"avoid_answers={len(avoid_answers)}"
        )
        response = self.runner.run_json(
            self.config.agents["seed"],
            system_prompt=SEED_PROMPT,
            user_payload={
                "domain": domain,
                "target_type": target_type,
                "answer_field": answer_field,
                "num_seeds": num_seeds,
                "avoid_entities": list(avoid_entities),
                "avoid_answers": list(avoid_answers),
                "required_domain_slots": list(domain_slots),
                "allowed_domain_families": list(DOMAIN_FAMILIES),
                "existing_domain_counts": dict(existing_domain_counts or {}),
                "overused_source_domains": list(overused_source_domains),
            },
        )
        if "seeds" not in response:
            self._log(
                "seed_agent_no_seeds: "
                f"keys={sorted(response.keys())} "
                f"error={str(response.get('error') or response.get('reason') or response.get('note') or '')[:300]}"
            )
        seeds = [SeedRecord.from_dict(item) for item in response.get("seeds", [])]
        sourced = [seed for seed in seeds if has_required_seed_sources(seed)]
        dropped_missing_sources = len(seeds) - len(sourced)
        if dropped_missing_sources:
            self._log(f"seed_agent_dropped_missing_sources: count={dropped_missing_sources}")
        filtered = (
            filter_diverse_seeds(
                sourced,
                requested=num_seeds,
                existing_domain_counts=existing_domain_counts,
                preferred_domains=domain_slots,
                overused_source_domains=overused_source_domains,
            )
            if apply_diversity_filter
            else sourced
        )
        self._log(
            "seed_agent_done: "
            f"raw={len(seeds)} sourced={len(sourced)} filtered={len(filtered)} "
            f"diversity={apply_diversity_filter} "
            f"entities={[seed.target.entity_id for seed in filtered]}"
        )
        return filtered

    def run_seed(
        self,
        seed: SeedRecord,
        *,
        dry_run: bool = False,
        run_context: Dict[str, Any] | None = None,
    ) -> Artifact:
        artifact = Artifact(
            target=seed.target,
            notes=[seed.seed_note] if seed.seed_note else [],
            run_context=dict(run_context or {}),
        )
        artifact.iterations.append({"stage": "seed", "status": "loaded", "target": asdict(seed.target)})
        self._checkpoint_artifact(artifact)
        self._log(f"seed_workflow_start: target={seed.target.entity_id} dry_run={dry_run}")

        if not dry_run and self.config.seed_verifier_enabled:
            self._log(f"seed_verify_start: target={seed.target.entity_id}")
            try:
                seed_verification = self._verify_seed_fact(seed.target)
            except Exception as exc:
                seed_verification = {
                    "enabled": True,
                    "accepted": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                artifact.iterations.append(
                    {
                        "stage": "seed_verification",
                        "status": "error",
                        "report": seed_verification,
                    }
                )
                artifact.status = "verification_failed"
                self._checkpoint_artifact(artifact)
                self._log(
                    "seed_verify_error: "
                    f"target={seed.target.entity_id} error={type(exc).__name__}: {exc}"
                )
                return artifact
            artifact.iterations.append(
                {
                    "stage": "seed_verification",
                    "status": (
                        "accepted" if seed_verification.get("accepted") else "rejected"
                    ),
                    "report": seed_verification,
                }
            )
            self._checkpoint_artifact(artifact)
            self._log(
                "seed_verify_done: "
                f"target={seed.target.entity_id} "
                f"accepted={seed_verification.get('accepted')} "
                f"reason={seed_verification.get('reason', '')}"
            )
            if seed_verification.get("accepted"):
                enriched_target = _target_with_verified_seed_sources(
                    seed.target,
                    seed_verification,
                )
                if enriched_target.source_urls != seed.target.source_urls:
                    seed = SeedRecord(
                        target=enriched_target,
                        seed_note=seed.seed_note,
                    )
                    artifact.target = enriched_target
                    artifact.iterations.append(
                        {
                            "stage": "seed_source_enrichment",
                            "status": "completed",
                            "source_urls": list(enriched_target.source_urls),
                        }
                    )
                    self._checkpoint_artifact(artifact)
            if not seed_verification.get("accepted"):
                repaired = False
                if self.config.seed_repair_enabled:
                    for repair_attempt in range(self.config.seed_repair_retries):
                        self._log(
                            "seed_repair_start: "
                            f"target={seed.target.entity_id} "
                            f"attempt={repair_attempt + 1}/{self.config.seed_repair_retries}"
                        )
                        try:
                            repaired_target, repair_report = self._repair_seed_fact(
                                seed.target,
                                seed_verification,
                                attempt=repair_attempt,
                            )
                        except Exception as exc:
                            repaired_target = None
                            repair_report = {
                                "action": "reject",
                                "reason": f"seed repair failed: {type(exc).__name__}: {exc}",
                            }
                        artifact.iterations.append(
                            {
                                "stage": "seed_repair",
                                "status": "candidate" if repaired_target else "rejected",
                                "attempt": repair_attempt,
                                "report": repair_report,
                            }
                        )
                        self._checkpoint_artifact(artifact)
                        if repaired_target is None:
                            self._log(
                                "seed_repair_done: "
                                f"target={seed.target.entity_id} accepted=False "
                                f"reason={str(repair_report.get('reason') or '')[:240]}"
                            )
                            continue
                        self._log(
                            "seed_repair_verify_start: "
                            f"target={repaired_target.entity_id} attempt={repair_attempt + 1}"
                        )
                        try:
                            repaired_verification = self._verify_seed_fact(repaired_target)
                        except Exception as exc:
                            repaired_verification = {
                                "enabled": True,
                                "accepted": False,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                        artifact.iterations.append(
                            {
                                "stage": "seed_repair_verification",
                                "status": (
                                    "accepted"
                                    if repaired_verification.get("accepted")
                                    else "rejected"
                                ),
                                "attempt": repair_attempt,
                                "target": asdict(repaired_target),
                                "report": repaired_verification,
                            }
                        )
                        self._checkpoint_artifact(artifact)
                        if repaired_verification.get("accepted"):
                            repaired_target = _target_with_verified_seed_sources(
                                repaired_target,
                                repaired_verification,
                            )
                            seed = SeedRecord(
                                target=repaired_target,
                                seed_note=seed.seed_note,
                            )
                            artifact.target = repaired_target
                            seed_verification = repaired_verification
                            artifact.iterations.append(
                                {
                                    "stage": "seed_source_enrichment",
                                    "status": "completed",
                                    "source_urls": list(repaired_target.source_urls),
                                }
                            )
                            self._checkpoint_artifact(artifact)
                            repaired = True
                            self._log(
                                "seed_repair_done: "
                                f"target={seed.target.entity_id} accepted=True"
                            )
                            break
                        self._log(
                            "seed_repair_verify_done: "
                            f"target={repaired_target.entity_id} accepted=False "
                            f"reason={str(repaired_verification.get('reason') or '')[:240]}"
                        )
                if not repaired:
                    artifact.status = "rejected:seed_verification"
                    self._checkpoint_artifact(artifact)
                    return artifact

        previous_failure: Dict[str, Any] | None = None
        for revision in range(self.config.max_revisions + 1):
            artifact.iterations.append({"stage": "revision_start", "status": "running", "revision": revision})
            constraint_error: Dict[str, str] | None = None
            if dry_run and revision == 0:
                artifact.constraints = _dry_constraints(seed.target)
            elif dry_run:
                break
            else:
                self._log(f"constraint_start: target={seed.target.entity_id} revision={revision}")
                try:
                    artifact.constraints = self._make_constraints(
                        seed.target,
                        previous_constraints=artifact.constraints,
                        failure_report=previous_failure,
                        revision=revision,
                    )
                except Exception as exc:
                    artifact.constraints = []
                    constraint_error = {
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    previous_failure = {
                        "stage": "constraint",
                        "reason": (
                            f"constraint agent failed: "
                            f"{constraint_error['error_type']}: {constraint_error['error']}"
                        ),
                    }
                    self._log(
                        "constraint_error: "
                        f"target={seed.target.entity_id} revision={revision} "
                        f"error={type(exc).__name__}: {exc}"
                    )
                self._log(
                    f"constraint_done: target={seed.target.entity_id} revision={revision} "
                    f"paths={len(artifact.constraints)} "
                    f"core={sum(1 for path in artifact.constraints if path.role == 'core')} "
                    f"distractor={sum(1 for path in artifact.constraints if path.role == 'distractor')}"
                )
            artifact.iterations.append(
                {
                    "stage": "constraint",
                    "status": "completed" if artifact.constraints else "failed",
                    "revision": revision,
                    "constraint_count": len(artifact.constraints),
                    **(constraint_error or {}),
                }
            )
            self._checkpoint_artifact(artifact)
            if not artifact.constraints:
                if previous_failure is None or previous_failure.get("stage") != "constraint":
                    previous_failure = {"stage": "constraint", "reason": "no constraints returned"}
                if revision >= self.config.max_revisions:
                    artifact.status = "rejected:no_constraints"
                    return artifact
                continue

            artifact.verifier = self._verify_program_constraints(artifact)
            self._log(
                "program_verifier_done: "
                f"target={seed.target.entity_id} revision={revision} "
                f"accepted={artifact.verifier.accepted} reason={artifact.verifier.reason}"
            )
            artifact.iterations.append(
                {
                    "stage": "program_verifier",
                    "status": "accepted" if artifact.verifier.accepted else "failed",
                    "revision": revision,
                    "report": asdict(artifact.verifier),
                }
            )
            self._checkpoint_artifact(artifact)
            if not artifact.verifier.accepted:
                previous_failure = {
                    "stage": "program_verifier",
                    "reason": artifact.verifier.reason,
                    "report": asdict(artifact.verifier),
                }
                if revision >= self.config.max_revisions:
                    artifact.status = "rejected:program_verifier"
                    return artifact
                self._log(
                    f"constraint_revision_needed: target={seed.target.entity_id} "
                    f"revision={revision} failed_stage=program_verifier"
                )
                continue

            local_report: Dict[str, Any] = {"enabled": self.config.local_max_depth > 0}
            if not dry_run and self.config.local_max_depth > 0:
                self._log(
                    f"local_start: target={seed.target.entity_id} revision={revision} "
                    f"max_depth={self.config.local_max_depth} concurrency={self.config.local_concurrency}"
                )
                artifact.constraints, local_report = self._local_fuzzify(artifact)
                self._log(
                    "local_done: "
                    f"target={seed.target.entity_id} revision={revision} "
                    f"accepted={local_report.get('accepted')} "
                    f"checked={local_report.get('checked_nodes')} failures={len(local_report.get('failures', []))}"
                )
            artifact.iterations.append(
                {
                    "stage": "local_constraint",
                    "status": "completed" if local_report.get("accepted", True) else "failed",
                    "revision": revision,
                    "constraint_count": len(artifact.constraints),
                    "report": local_report,
                }
            )
            self._checkpoint_artifact(artifact)
            if not local_report.get("accepted", True):
                previous_failure = {
                    "stage": "local_constraint",
                    "reason": _local_failure_reason(local_report),
                    "report": _compact_local_failure_report(local_report),
                }
                if revision >= self.config.max_revisions:
                    artifact.status = "rejected:local_verification"
                    return artifact
                self._log(
                    f"constraint_revision_needed: target={seed.target.entity_id} "
                    f"revision={revision} failed_stage=local_constraint"
                )
                continue
            break
        else:
            artifact.status = "rejected:max_revisions"
            return artifact

        question_response: Dict[str, Any] = {}
        if dry_run:
            artifact.question = _dry_question(artifact)
            self._record_question_version(
                artifact,
                source="dry_run",
                revision=0,
                response={"question": artifact.question, "answer": artifact.target.answer},
            )
        else:
            self._log(f"question_start: target={seed.target.entity_id}")
            question_response = self._write_question(artifact, revision=0)
            artifact.question = str(question_response.get("question", ""))
            if artifact.question:
                self._record_question_version(
                    artifact,
                    source="question",
                    revision=0,
                    response=question_response,
                )
            if not artifact.question and self._question_needs_leaf_growth(question_response):
                for growth_round in range(self.config.max_revisions + 1):
                    growth_report = self._grow_question_leaf_paths(artifact, question_response)
                    artifact.iterations.append(
                        {
                            "stage": "question_leaf_growth",
                            "status": "completed" if growth_report.get("grown_paths") else "skipped",
                            "round": growth_round,
                            "report": growth_report,
                        }
                    )
                    if not growth_report.get("grown_paths"):
                        break
                    question_response = self._write_question(artifact, revision=growth_round + 1)
                    artifact.question = str(question_response.get("question", ""))
                    if artifact.question:
                        self._record_question_version(
                            artifact,
                            source="question_after_leaf_growth",
                            revision=growth_round + 1,
                            response=question_response,
                        )
                    artifact.iterations.append(
                        {
                            "stage": "question_after_leaf_growth",
                            "status": "completed" if artifact.question else "failed",
                            "round": growth_round,
                            "response": question_response,
                        }
                    )
                    if artifact.question or not self._question_needs_leaf_growth(question_response):
                        break
            self._log(
                f"question_done: target={seed.target.entity_id} has_question={bool(artifact.question)}"
            )
            artifact.iterations.append(
                {
                    "stage": "question",
                    "status": "completed" if artifact.question else "failed",
                    "response": question_response,
                }
            )
            self._checkpoint_artifact(artifact)
        if not artifact.question:
            artifact.status = "rejected:no_question"
            return artifact

        if not getattr(self.config, "question_structural_only", False):
            if not self._repair_question_until_unique(artifact):
                return artifact
        else:
            # The structural Question gate sends the first draft to Solver.
            # Public ambiguity is decided only from a wrong Solver answer that
            # passes clause-complete independent adjudication.
            artifact.uniqueness_report = {
                "enabled": bool(self.config.uniqueness_enabled),
                "unique": None,
                "key": "deferred_to_solver",
                "reason": (
                    "pre-Solver uniqueness is deferred by the structural Question "
                    "gate; Solver/trajectory adjudication owns ambiguity feedback"
                ),
                "alternatives": [],
            }
            artifact.uniqueness_key = "deferred_to_solver"
            artifact.iterations.append(
                {
                    "stage": "uniqueness",
                    "status": "deferred_to_solver",
                    "revision": 0,
                    "question": artifact.question,
                    "report": deepcopy(artifact.uniqueness_report),
                }
            )

        if dry_run:
            artifact.solver_reports = []
            artifact.solver_summary = SolverSummary(
                total=0,
                correct=0,
                incorrect=0,
                accepted=True,
                status="accepted:dry_run",
                reason="dry run skips solver",
            )
            artifact.status = "accepted:dry_run"
            self._checkpoint_artifact(artifact)
            return artifact

        if not self.config.solver_enabled:
            artifact.solver_reports = []
            artifact.solver_summary = SolverSummary(
                total=0,
                correct=0,
                incorrect=0,
                accepted=True,
                status="accepted:solver_disabled",
                reason="solver disabled by V2_SOLVER_ENABLED=0",
            )
            artifact.iterations.append(
                {
                    "stage": "solver",
                    "status": artifact.solver_summary.status,
                    "question_revision": 0,
                    "question": artifact.question,
                    "summary": asdict(artifact.solver_summary),
                }
            )
            artifact.status = artifact.solver_summary.status
            self._checkpoint_artifact(artifact)
            self._log(f"solver_skipped: target={seed.target.entity_id} reason=solver_disabled")
            self._log(f"seed_workflow_done: target={seed.target.entity_id} status={artifact.status}")
            return artifact

        previous_questions = [artifact.question]
        best_accepted_snapshot: Dict[str, Any] | None = None
        pruning_passes = 0
        for question_revision in range(self.config.question_revisions + 1):
            question_version = self._current_question_version(artifact)
            self._write_solver_question_snapshot(
                artifact, question_version=question_version
            )
            self._log(
                f"solver_start: target={seed.target.entity_id} rollouts={self.config.solver_rollouts} "
                f"concurrency={self.config.solver_concurrency} question_revision={question_revision}"
            )
            artifact.solver_reports = self._run_solvers(
                artifact,
                question_version=question_version,
            )
            artifact.solver_summary = self._summarize_solvers(artifact.solver_reports, artifact.target.answer)
            trajectory_feedback: Dict[str, Any] = {}
            if artifact.solver_summary.incorrect > 0:
                trajectory_feedback = self._summarize_solver_trajectories(
                    artifact,
                    artifact.solver_reports,
                    artifact.solver_summary,
                )
                if str(trajectory_feedback.get("diagnosis") or "") == "invalid_or_ambiguous":
                    artifact.solver_summary.accepted = False
                    artifact.solver_summary.status = "needs_repair:ambiguous"
                    artifact.solver_summary.reason = (
                        "A solver found a coherent alternative satisfying the public "
                        "wording; add tree-only disambiguation before acceptance. "
                        + str(trajectory_feedback.get("repair_guidance") or "")
                    ).strip()
            self._log(
                "solver_done: "
                f"target={seed.target.entity_id} status={artifact.solver_summary.status} "
                f"correct={artifact.solver_summary.correct}/{artifact.solver_summary.total} "
                f"question_revision={question_revision}"
            )
            artifact.iterations.append(
                {
                    "stage": "solver",
                    "status": artifact.solver_summary.status,
                    "question_revision": question_revision,
                    "question": artifact.question,
                    "summary": asdict(artifact.solver_summary),
                }
            )
            solver_attempt = {
                "attempt": len(artifact.solver_attempts),
                "question_revision": question_revision,
                "question_version": question_version,
                "question": artifact.question,
                "question_state": deepcopy(artifact.question_state),
                "uniqueness_key": artifact.uniqueness_key,
                "uniqueness_report": deepcopy(artifact.uniqueness_report),
                "solver_reports": deepcopy(artifact.solver_reports),
                "solver_summary": asdict(artifact.solver_summary),
                "trajectory_feedback": deepcopy(trajectory_feedback),
            }
            artifact.solver_attempts.append(solver_attempt)
            self._write_solver_attempt_snapshot(artifact, solver_attempt)
            self._checkpoint_artifact(artifact)
            if artifact.solver_summary.accepted:
                candidate_snapshot = {
                    "question": artifact.question,
                    "question_state": deepcopy(artifact.question_state),
                    "solver_reports": deepcopy(artifact.solver_reports),
                    "solver_summary": deepcopy(artifact.solver_summary),
                    "uniqueness_key": artifact.uniqueness_key,
                    "uniqueness_report": deepcopy(artifact.uniqueness_report),
                }
                if best_accepted_snapshot is None or _solver_summary_score(
                    artifact.solver_summary
                ) > _solver_summary_score(
                    best_accepted_snapshot["solver_summary"]
                ):
                    best_accepted_snapshot = candidate_snapshot
                if artifact.solver_summary.status == "review:all_wrong":
                    break
                if (
                    _solver_prune_targets_reached(
                        artifact.solver_summary
                    )
                    or not getattr(self.config, "solver_continue_pruning", False)
                    or pruning_passes
                    >= max(0, int(getattr(self.config, "solver_pruning_revisions", 0)))
                ):
                    break
                pruning_passes += 1
            elif (
                best_accepted_snapshot is not None
                and pruning_passes
                >= max(0, int(getattr(self.config, "solver_pruning_revisions", 0)))
            ):
                break
            repairable_statuses = {
                "needs_repair:too_easy",
                "needs_repair:ambiguous",
            }
            if (
                not artifact.solver_summary.accepted
                and artifact.solver_summary.status not in repairable_statuses
            ):
                break
            if not trajectory_feedback:
                trajectory_feedback = self._summarize_solver_trajectories(
                    artifact,
                    artifact.solver_reports,
                    artifact.solver_summary,
                )
            artifact.iterations.append(
                {
                    "stage": "solver_trajectory_summary",
                    "status": str(trajectory_feedback.get("diagnosis") or "unknown"),
                    "question_revision": question_revision,
                    "report": trajectory_feedback,
                }
            )
            solver_attempt["trajectory_feedback"] = deepcopy(trajectory_feedback)
            self._write_solver_attempt_snapshot(artifact, solver_attempt)
            self._checkpoint_artifact(artifact)
            actionable_prune = bool(
                trajectory_feedback.get("shortcut_root_path_ids")
                or trajectory_feedback.get("shortcut_child_path_ids")
                or trajectory_feedback.get("shortcut_queries")
                or trajectory_feedback.get("recoverable_intermediate_entities")
            )
            # An accepted question is retained unless the trajectory summary
            # names a concrete shortcut root, child, query, or recoverable
            # intermediate. A bare recommendation such as ``shortcut_prune``
            # is not enough to justify rewriting a verified question.
            if artifact.solver_summary.accepted and not actionable_prune:
                break
            if question_revision >= self.config.question_revisions:
                break
            failure_report = self._solver_repair_failure_report(
                artifact.solver_reports,
                artifact.solver_summary,
                previous_questions,
                trajectory_feedback,
            )
            self._log(
                "question_revision_needed: "
                f"target={seed.target.entity_id} revision={question_revision + 1} "
                f"reason={artifact.solver_summary.reason}"
            )
            write_kwargs: Dict[str, Any] = {}
            if question_revision > 0:
                write_kwargs["force_fallback"] = True
            question_response = self._write_question(
                artifact,
                revision=question_revision + 1,
                previous_questions=previous_questions,
                failure_report=failure_report,
                **write_kwargs,
            )
            new_question = str(question_response.get("question", ""))
            artifact.iterations.append(
                {
                    "stage": "question_revision",
                    "status": "completed" if new_question else "failed",
                    "revision": question_revision + 1,
                    "failure_report": failure_report,
                    "response": question_response,
                }
            )
            if not new_question or new_question in previous_questions:
                self._log(
                    "question_revision_stop: "
                    f"target={seed.target.entity_id} revision={question_revision + 1} "
                    "reason=empty_or_repeated_question"
                )
                break
            artifact.question = new_question
            previous_questions.append(new_question)
            self._record_question_version(
                artifact,
                source="solver_repair",
                revision=question_revision + 1,
                response=question_response,
            )
            # The legacy pre-Solver uniqueness recheck remains available when
            # the structural Question gate is disabled.
            if (
                not getattr(self.config, "question_structural_only", False)
                and not self._repair_question_until_unique(artifact)
            ):
                if best_accepted_snapshot is None:
                    return artifact
                break
            if artifact.question not in previous_questions:
                previous_questions.append(artifact.question)
        if (
            best_accepted_snapshot is not None
            and (
                not artifact.solver_summary.accepted
                or _solver_summary_score(artifact.solver_summary)
                < _solver_summary_score(best_accepted_snapshot["solver_summary"])
                or artifact.question != best_accepted_snapshot["question"]
                or artifact.uniqueness_key
                != best_accepted_snapshot["uniqueness_key"]
            )
        ):
            artifact.question = best_accepted_snapshot["question"]
            artifact.question_state = best_accepted_snapshot["question_state"]
            artifact.solver_reports = best_accepted_snapshot["solver_reports"]
            artifact.solver_summary = best_accepted_snapshot["solver_summary"]
            artifact.uniqueness_key = best_accepted_snapshot["uniqueness_key"]
            artifact.uniqueness_report = best_accepted_snapshot["uniqueness_report"]
        if artifact.solver_summary.status == "needs_repair:too_hard":
            artifact.solver_summary.status = "review:all_wrong"
            artifact.solver_summary.reason = (
                "all solver rollouts remained wrong after the repair budget"
            )
        elif artifact.solver_summary.status == "needs_repair:too_easy":
            artifact.solver_summary.status = "rejected:too_easy"
            artifact.solver_summary.reason = (
                "successful solver trajectories remained below the separate API/tool effort target "
                "after the repair budget"
            )
        elif artifact.solver_summary.status == "needs_repair:ambiguous":
            artifact.solver_summary.status = "rejected:ambiguous"
            artifact.solver_summary.reason = (
                "a coherent alternate solution remained after the repair budget"
            )
        artifact.status = artifact.solver_summary.status
        if artifact.solver_attempts:
            artifact.solver_attempts[-1]["final_artifact_status"] = artifact.status
        self._checkpoint_artifact(artifact)
        self._log(f"seed_workflow_done: target={seed.target.entity_id} status={artifact.status}")
        return artifact

    def _repair_question_until_unique(self, artifact: Artifact) -> bool:
        previous_questions = [artifact.question]
        for revision in range(self.config.uniqueness_revisions + 1):
            uniqueness_report = self._run_uniqueness_check(artifact)
            artifact.uniqueness_report = uniqueness_report
            artifact.uniqueness_key = str(uniqueness_report.get("key", ""))
            self._update_question_uniqueness(artifact, uniqueness_report)
            artifact.iterations.append(
                {
                    "stage": "uniqueness",
                    "status": artifact.uniqueness_key or "skipped",
                    "revision": revision,
                    "question": artifact.question,
                    "report": uniqueness_report,
                }
            )
            consensus_required = bool(
                getattr(self.config, "uniqueness_require_consensus", True)
            )
            if artifact.uniqueness_key in {"unique", "skipped"} or (
                artifact.uniqueness_key == "uncertain" and not consensus_required
            ):
                return True

            if artifact.uniqueness_key == "check_failed":
                artifact.status = "verification_failed"
                artifact.iterations.append(
                    {
                        "stage": "uniqueness_final_rejection",
                        "status": artifact.status,
                        "revision": revision,
                        "question": artifact.question,
                        "report": uniqueness_report,
                    }
                )
                return False

            if (
                artifact.uniqueness_key == "not_unique"
                and getattr(self.config, "uniqueness_defer_to_solver", False)
            ):
                artifact.iterations.append(
                    {
                        "stage": "uniqueness_deferred_to_solver",
                        "status": "deferred",
                        "revision": revision,
                        "question": artifact.question,
                        "report": uniqueness_report,
                        "reason": (
                            "pre-solver alternatives are advisory; only a wrong Solver "
                            "answer with clause-complete adjudication can establish ambiguity"
                        ),
                    }
                )
                return True

            if revision >= self.config.uniqueness_revisions:
                if (
                    artifact.uniqueness_key == "uncertain"
                    and bool(
                        getattr(
                            self.config,
                            "uniqueness_allow_uncertain_after_repair",
                            True,
                        )
                    )
                    and not uniqueness_report.get("alternatives")
                    and not uniqueness_report.get("unresolved_prior_alternatives")
                    and not uniqueness_report.get("unresolved_blind_candidates")
                ):
                    artifact.iterations.append(
                        {
                            "stage": "uniqueness_bounded_acceptance",
                            "status": "uncertain_no_verified_alternative",
                            "revision": revision,
                            "question": artifact.question,
                            "report": uniqueness_report,
                            "reason": (
                                "bounded tree-only repairs exhausted without a complete "
                                "verified alternative; solver wrong answers still require "
                                "strong ambiguity adjudication"
                            ),
                        }
                    )
                    return True
                artifact.status = (
                    "rejected:not_unique"
                    if artifact.uniqueness_key == "not_unique"
                    else "rejected:uniqueness_uncertain"
                )
                artifact.iterations.append(
                    {
                        "stage": "uniqueness_final_rejection",
                        "status": artifact.status,
                        "revision": revision,
                        "question": artifact.question,
                        "report": uniqueness_report,
                    }
                )
                return False
            if not self.config.question_repair_enabled:
                artifact.status = "rejected:not_unique"
                artifact.iterations.append(
                    {
                        "stage": "uniqueness_question_revision",
                        "status": "skipped",
                        "revision": revision + 1,
                        "reason": "question repair disabled",
                        "question": artifact.question,
                        "report": uniqueness_report,
                    }
                )
                return False

            failure_report = {
                "stage": "uniqueness",
                "status": artifact.uniqueness_key,
                "reason": str(uniqueness_report.get("reason", "")),
                "alternatives": uniqueness_report.get("alternatives", []),
                "missing_disambiguator": str(
                    uniqueness_report.get("missing_disambiguator", "")
                ),
                "tree_only": True,
            }
            self._log(
                "uniqueness_question_revision_needed: "
                f"target={artifact.target.entity_id} revision={revision + 1}"
            )
            try:
                write_kwargs: Dict[str, Any] = {}
                if revision > 0:
                    write_kwargs["force_fallback"] = True
                question_response = self._write_question(
                    artifact,
                    revision=revision + 1,
                    previous_questions=previous_questions,
                    failure_report=failure_report,
                    **write_kwargs,
                )
            except Exception as exc:
                artifact.status = "rejected:uniqueness_repair_failed"
                artifact.iterations.append(
                    {
                        "stage": "uniqueness_question_revision",
                        "status": "failed",
                        "revision": revision + 1,
                        "failure_report": failure_report,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "question": artifact.question,
                    }
                )
                return False

            new_question = str(question_response.get("question", "")).strip()
            artifact.iterations.append(
                {
                    "stage": "uniqueness_question_revision",
                    "status": "completed" if new_question else "failed",
                    "revision": revision + 1,
                    "failure_report": failure_report,
                    "response": question_response,
                }
            )
            if not new_question or new_question in previous_questions:
                artifact.status = "rejected:uniqueness_repair_failed"
                artifact.iterations.append(
                    {
                        "stage": "uniqueness_final_rejection",
                        "status": artifact.status,
                        "revision": revision + 1,
                        "question": artifact.question,
                        "candidate_question": new_question,
                        "reason": "empty or repeated repaired question",
                    }
                )
                return False
            artifact.question = new_question
            previous_questions.append(new_question)
            self._record_question_version(
                artifact,
                source="uniqueness_repair",
                revision=revision + 1,
                response=question_response,
            )

        artifact.status = "rejected:not_unique"
        artifact.iterations.append(
            {
                "stage": "uniqueness_final_rejection",
                "status": artifact.status,
                "revision": self.config.uniqueness_revisions,
                "question": artifact.question,
                "report": artifact.uniqueness_report,
            }
        )
        return False

    def _run_uniqueness_check(self, artifact: Artifact) -> Dict[str, Any]:
        if not self.config.uniqueness_enabled:
            return {
                "enabled": False,
                "unique": None,
                "key": "skipped",
                "reason": "uniqueness check disabled by V2_UNIQUENESS_ENABLED=0",
                "alternatives": [],
                "missing_disambiguator": "",
            }
        blind_resolution = self._run_blind_question_resolution(artifact)
        public_clauses = _public_question_clauses(artifact.question)
        required_clause_ids = [item["clause_id"] for item in public_clauses]
        prior_verified_alternatives: List[Dict[str, Any]] = []
        seen_prior: set[str] = set()
        previous_uniqueness = (
            artifact.uniqueness_report
            if isinstance(artifact.uniqueness_report, dict)
            else {}
        )
        for item in [
            *previous_uniqueness.get("prior_verified_alternatives", []),
            *previous_uniqueness.get("alternatives", []),
        ]:
            if not isinstance(item, dict):
                continue
            key = _entity_text_key(item.get("name"))
            if key and key not in seen_prior:
                seen_prior.add(key)
                prior_verified_alternatives.append(deepcopy(item))
        prior_unresolved_candidates: List[Dict[str, Any]] = []
        seen_unresolved: set[str] = set()
        for item in previous_uniqueness.get("unresolved_blind_candidates", []):
            if not isinstance(item, dict):
                continue
            key = _entity_text_key(item.get("name"))
            if key and key not in seen_unresolved:
                seen_unresolved.add(key)
                prior_unresolved_candidates.append(deepcopy(item))
        payload = {
            "target": asdict(artifact.target),
            "question": artifact.question,
            "expected_answer": artifact.target.answer,
            "answer_field": artifact.target.answer_field,
            "public_clauses": public_clauses,
            "blind_resolution": blind_resolution,
            "prior_verified_alternatives": prior_verified_alternatives,
            "prior_unresolved_candidates": prior_unresolved_candidates,
        }
        normalized_reports: List[Dict[str, Any]] = []
        rollout_count = max(1, int(getattr(self.config, "uniqueness_rollouts", 1)))
        for rollout_id in range(rollout_count):
            try:
                report = self.runner.run_json(
                    self.config.agents["uniqueness"],
                    system_prompt=UNIQUENESS_PROMPT,
                    user_payload=payload,
                    rate_limit_scope=artifact.target.entity_id,
                )
                normalized = _normalize_uniqueness_response(
                    report,
                    required_clause_ids=required_clause_ids,
                )
            except Exception as exc:
                self._log(
                    "uniqueness_check_failed: "
                    f"target={artifact.target.entity_id} rollout={rollout_id} "
                    f"error={str(exc)[:220]}"
                )
                normalized = {
                    "enabled": True,
                    "unique": None,
                    "key": "check_failed",
                    "reason": f"uniqueness agent failed: {exc}",
                    "alternatives": [],
                    "missing_disambiguator": "",
                }
            normalized["rollout_id"] = rollout_id
            normalized_reports.append(normalized)

        rejected = [item for item in normalized_reports if item["unique"] is False]
        if rejected:
            primary = rejected[0]
            alternatives: List[Any] = []
            seen_alternatives: set[str] = set()
            for item in rejected:
                for alternative in item.get("alternatives", []):
                    key = _entity_text_key(
                        alternative.get("name")
                        if isinstance(alternative, dict)
                        else alternative
                    )
                    if key and key not in seen_alternatives:
                        seen_alternatives.add(key)
                        alternatives.append(alternative)
            out = {
                **primary,
                "alternatives": alternatives,
                "reason": " | ".join(
                    str(item.get("reason") or "") for item in rejected
                ),
            }
        elif all(item["unique"] is True for item in normalized_reports):
            out = dict(normalized_reports[0])
            out["reason"] = " | ".join(
                str(item.get("reason") or "") for item in normalized_reports
            )
        else:
            failed = any(item.get("key") == "check_failed" for item in normalized_reports)
            out = {
                "enabled": True,
                "unique": None,
                "key": "check_failed" if failed else "uncertain",
                "reason": " | ".join(
                    str(item.get("reason") or "") for item in normalized_reports
                ),
                "alternatives": [],
                "missing_disambiguator": next(
                    (
                        str(item.get("missing_disambiguator") or "")
                        for item in normalized_reports
                        if str(item.get("missing_disambiguator") or "")
                    ),
                    "",
                ),
            }
        if (
            out.get("unique") is None
            and not rejected
            and _blind_resolution_matches_expected(artifact.target, blind_resolution)
            and any(item.get("unique") is True for item in normalized_reports)
        ):
            positive = next(
                item for item in normalized_reports if item.get("unique") is True
            )
            out = {
                **positive,
                "unique": True,
                "key": "unique",
                "reason": (
                    "target-blind resolution independently matched the intended "
                    "target and at least one hidden-target rollout verified uniqueness; "
                    "remaining rollouts found no complete alternative. "
                    + " | ".join(
                        str(item.get("reason") or "")
                        for item in normalized_reports
                    )
                ),
            }
        out["answer_cardinality"] = {
            "verdict": "verified_upstream",
            "reason": "answer-field cardinality is owned by the Seed Gate",
            "expected_answer_supported": True,
            "alternate_answers": [],
        }
        out["rollouts"] = normalized_reports
        out["blind_resolution"] = blind_resolution
        blind_concern = _blind_resolution_concern(
            artifact.target,
            blind_resolution,
        )
        if out.get("unique") is True and blind_concern:
            rejected_blind_names = {
                _entity_text_key(item.get("name"))
                for report in normalized_reports
                for item in report.get("unsupported_alternatives", [])
                if isinstance(item, dict) and item.get("failed_clues")
            }
            unresolved_names = {
                _entity_text_key(item.get("name"))
                for item in blind_resolution.get("candidate_targets", [])
                if isinstance(item, dict)
                and not _blind_candidate_matches_target(artifact.target, item)
                and _entity_text_key(item.get("name")) not in rejected_blind_names
            }
            unresolved_names.discard("")
            if unresolved_names:
                out.update(
                    {
                        "unique": None,
                        "key": "uncertain",
                        "reason": (
                            str(out.get("reason") or "")
                            + " | hidden-target rollouts did not explicitly exclude "
                            "the independently resolved candidate(s): "
                            + ", ".join(sorted(unresolved_names))
                        ).strip(" |"),
                    }
                )
        verified_current_keys = {
            _entity_text_key(item.get("name"))
            for report in normalized_reports
            for item in report.get("alternatives", [])
            if isinstance(item, dict) and _entity_text_key(item.get("name"))
        }
        excluded_current_keys = {
            _entity_text_key(item.get("name"))
            for report in normalized_reports
            for item in report.get("unsupported_alternatives", [])
            if isinstance(item, dict)
            and item.get("failed_clues")
            and _entity_text_key(item.get("name"))
        }
        unresolved_prior = [
            item
            for item in prior_verified_alternatives
            if _entity_text_key(item.get("name")) not in verified_current_keys
            and _entity_text_key(item.get("name")) not in excluded_current_keys
        ]
        if unresolved_prior and out.get("unique") is True:
            out.update(
                {
                    "unique": None,
                    "key": "uncertain",
                    "reason": (
                        str(out.get("reason") or "")
                        + " | revised-question verifier did not adjudicate prior "
                        "verified alternative(s): "
                        + ", ".join(
                            str(item.get("name") or "") for item in unresolved_prior
                        )
                    ).strip(" |"),
                }
            )
        unresolved_blind_pool: List[Dict[str, Any]] = []
        unresolved_blind_keys: set[str] = set()
        current_blind_candidates = blind_resolution.get("candidate_targets", [])
        current_blind_candidates = (
            current_blind_candidates
            if isinstance(current_blind_candidates, list)
            else []
        )
        for item in [*prior_unresolved_candidates, *current_blind_candidates]:
            if not isinstance(item, dict):
                continue
            key = _entity_text_key(item.get("name"))
            if (
                not key
                or _blind_candidate_matches_target(artifact.target, item)
                or key in verified_current_keys
                or key in excluded_current_keys
                or key in unresolved_blind_keys
            ):
                continue
            unresolved_blind_keys.add(key)
            unresolved_blind_pool.append(deepcopy(item))
        if unresolved_blind_pool and out.get("unique") is True:
            out.update(
                {
                    "unique": None,
                    "key": "uncertain",
                    "reason": (
                        str(out.get("reason") or "")
                        + " | revised-question verifier did not adjudicate sourced "
                        "blind candidate(s): "
                        + ", ".join(
                            str(item.get("name") or "")
                            for item in unresolved_blind_pool
                        )
                    ).strip(" |"),
                }
            )
        history = [*prior_verified_alternatives]
        history_keys = {
            _entity_text_key(item.get("name")) for item in history
        }
        for item in out.get("alternatives", []):
            if not isinstance(item, dict):
                continue
            key = _entity_text_key(item.get("name"))
            if key and key not in history_keys:
                history_keys.add(key)
                history.append(deepcopy(item))
        out["prior_verified_alternatives"] = history
        out["unresolved_prior_alternatives"] = unresolved_prior
        out["unresolved_blind_candidates"] = unresolved_blind_pool
        self._log(
            "uniqueness_done: "
            f"target={artifact.target.entity_id} key={out['key']} "
            f"alternatives={len(out.get('alternatives', []))} rollouts={rollout_count}"
        )
        return out

    def _run_blind_question_resolution(self, artifact: Artifact) -> Dict[str, Any]:
        if not bool(getattr(self.config, "blind_uniqueness_enabled", False)):
            return {
                "enabled": False,
                "resolution": "skipped",
                "candidate_targets": [],
                "unresolved_clues": [],
                "reason": "blind uniqueness audit disabled",
            }
        try:
            public_clauses = _public_question_clauses(artifact.question)
            required_clause_ids = [item["clause_id"] for item in public_clauses]
            response = self.runner.run_json(
                self.config.agents["uniqueness"],
                system_prompt=BLIND_QUESTION_RESOLUTION_PROMPT,
                user_payload={
                    "question": artifact.question,
                    "answer_field": artifact.target.answer_field,
                    "public_clauses": public_clauses,
                },
                rate_limit_scope=artifact.target.entity_id,
            )
        except Exception as exc:
            return {
                "enabled": True,
                "resolution": "uncertain",
                "candidate_targets": [],
                "unresolved_clues": [],
                "reason": f"blind resolver failed: {type(exc).__name__}: {exc}",
            }
        resolution = str(response.get("resolution") or "uncertain").strip().lower()
        if resolution not in {"resolved", "multiple", "uncertain"}:
            resolution = "uncertain"
        raw_candidates = response.get("candidate_targets")
        raw_candidates = raw_candidates if isinstance(raw_candidates, list) else []
        candidates = [
            {
                "name": str(item.get("name") or "").strip(),
                "answer": str(item.get("answer") or "").strip(),
                "clue_checks": [
                    str(check).strip()
                    for check in item.get("clue_checks", [])
                    if str(check).strip()
                ] if isinstance(item.get("clue_checks"), list) else [],
                "source_urls": [
                    str(url).strip()
                    for url in item.get("source_urls", [])
                    if str(url).strip().startswith(("http://", "https://"))
                ] if isinstance(item.get("source_urls"), list) else [],
            }
            for item in raw_candidates
            if isinstance(item, dict) and str(item.get("name") or "").strip()
        ]
        unresolved_clues = [
            str(item).strip()
            for item in response.get("unresolved_clues", [])
            if str(item).strip()
        ] if isinstance(response.get("unresolved_clues"), list) else []
        complete_candidates = [
            item
            for item in candidates
            if item["source_urls"]
            and _clue_checks_cover_clauses(
                item["clue_checks"], required_clause_ids
            )
        ]
        if unresolved_clues or (
            resolution == "resolved" and len(complete_candidates) != 1
        ) or (
            resolution == "multiple" and len(complete_candidates) < 2
        ):
            resolution = "uncertain"
        return {
            "enabled": True,
            "resolution": resolution,
            "candidate_targets": candidates,
            "unresolved_clues": unresolved_clues,
            "reason": str(response.get("reason") or ""),
        }

    def _run_answer_cardinality_check(self, artifact: Artifact) -> Dict[str, Any]:
        return self._run_answer_cardinality_request(
            target=artifact.target,
            question=artifact.question,
            mode="question",
        )

    def _run_answer_cardinality_request(
        self,
        *,
        target: Target,
        question: str,
        mode: str,
    ) -> Dict[str, Any]:
        enabled = bool(getattr(self.config, "answer_cardinality_enabled", True))
        agent = getattr(self.config, "agents", {}).get("answer_cardinality")
        if not enabled or agent is None or not getattr(agent, "configured", False):
            return {
                "verdict": "skipped",
                "reason": "answer-cardinality verifier disabled or unconfigured",
                "expected_answer_supported": None,
                "alternate_answers": [],
            }
        try:
            response = self.runner.run_json(
                agent,
                system_prompt=ANSWER_CARDINALITY_PROMPT,
                user_payload={
                    "mode": mode,
                    "question": question,
                    "target": {
                        "name": target.name,
                        "entity_id": target.entity_id,
                        "entity_type": target.entity_type,
                    },
                    "answer_field": target.answer_field,
                    "expected_answer": target.answer,
                    "source_urls": list(target.source_urls),
                },
                rate_limit_scope=target.entity_id,
            )
        except Exception as exc:
            return {
                "verdict": "check_failed",
                "reason": f"{type(exc).__name__}: {exc}",
                "expected_answer_supported": None,
                "alternate_answers": [],
            }
        verdict = str(response.get("verdict") or "uncertain").strip().lower()
        if verdict not in {"single", "plural", "uncertain"}:
            verdict = "uncertain"
        alternatives = response.get("alternate_answers")
        alternatives = alternatives if isinstance(alternatives, list) else []
        verified = [
            item
            for item in alternatives
            if isinstance(item, dict)
            and item.get("same_target") is True
            and item.get("matches_question_field") is True
            and str(item.get("answer") or "").strip()
            and str(item.get("role_check") or "").strip()
            and isinstance(item.get("source_urls"), list)
            and any(str(url).strip() for url in item.get("source_urls", []))
        ]
        if verdict == "plural" and not verified:
            verdict = "uncertain"
        return {
            "verdict": verdict,
            "reason": str(response.get("reason") or ""),
            "expected_answer_supported": response.get("expected_answer_supported"),
            "alternate_answers": verified,
            "missing_disambiguator": str(
                response.get("missing_disambiguator") or ""
            ),
        }

    def _question_needs_leaf_growth(self, response: Dict[str, Any]) -> bool:
        reason = str(response.get("failure_reason", ""))
        return bool(response.get("missing_leaf_core_path_ids")) and (
            reason.startswith("need more core paths")
            or reason.startswith("required unique root bundle")
        )

    def _verify_seed_fact(self, target: Target) -> Dict[str, Any]:
        reports: List[Dict[str, Any]] = []
        rollout_count = max(1, int(getattr(self.config, "seed_verifier_rollouts", 1)))
        for rollout_id in range(rollout_count):
            try:
                response = self.runner.run_json(
                    self.config.agents["seed_verifier"],
                    system_prompt=SEED_FACT_VERIFIER_PROMPT,
                    user_payload={
                        "evaluation_date": date.today().isoformat(),
                        "target": asdict(target),
                        "verification_rollout": rollout_id,
                    },
                    rate_limit_scope=target.entity_id,
                    response_validator=_validate_seed_fact_response,
                )
                response = dict(response)
                response["accepted"] = all(
                    response.get(key) is True
                    for key in (
                        "accepted",
                        "target_found",
                        "answer_field_supported",
                        "answer_matches",
                    )
                )
            except Exception as exc:
                response = {
                    "accepted": False,
                    "target_found": False,
                    "answer_field_supported": False,
                    "answer_matches": False,
                    "reason": f"seed verifier rollout failed: {type(exc).__name__}: {exc}",
                    "source_urls": [],
                }
            response["rollout_id"] = rollout_id
            reports.append(response)
        cardinality = self._run_seed_answer_cardinality_check(target)
        answer_field_problem = _seed_answer_field_problem(target)
        accepted = bool(reports) and all(item.get("accepted") is True for item in reports)
        if answer_field_problem:
            accepted = False
        cardinality_verdict = str(cardinality.get("verdict") or "skipped").strip().lower()
        if cardinality_verdict != "skipped":
            # Fail closed when the exact target's requested field is plural or
            # the cardinality check did not resolve. This keeps ambiguous seed
            # contracts out of the expensive tree/question stages.
            accepted = (
                accepted
                and cardinality_verdict == "single"
                and cardinality.get("expected_answer_supported") is True
            )
            if cardinality.get("expected_answer_supported") is not True:
                accepted = False
        chosen = reports[-1] if reports else {}
        reason_parts = [
            str(item.get("reason") or "")
            for item in reports
            if str(item.get("reason") or "")
        ]
        if answer_field_problem:
            reason_parts.append(answer_field_problem)
        if cardinality_verdict == "plural":
            reason_parts.append(
                "same-target answer field is plural: "
                + str(cardinality.get("reason") or "")
            )
        elif cardinality_verdict in {"uncertain", "check_failed"}:
            reason_parts.append(
                "same-target answer cardinality is not verified ("
                + cardinality_verdict
                + "): "
                + str(cardinality.get("reason") or "")
            )
        elif cardinality_verdict == "single" and cardinality.get("expected_answer_supported") is False:
            reason_parts.append(
                "answer-cardinality verifier did not support the supplied answer"
            )
        return {
            **chosen,
            "enabled": True,
            "accepted": accepted,
            "rollouts": reports,
            "answer_cardinality": cardinality,
            "answer_cardinality_ok": (
                cardinality_verdict in {"single", "skipped"}
                and (
                    cardinality_verdict == "skipped"
                    or cardinality.get("expected_answer_supported") is True
                )
            ),
            "answer_field_concise": not answer_field_problem,
            "reason": " | ".join(reason_parts),
        }

    def _run_seed_answer_cardinality_check(self, target: Target) -> Dict[str, Any]:
        """Fail closed across independent seed-cardinality research rollouts."""
        rollout_count = max(
            1,
            int(
                getattr(
                    self.config,
                    "seed_cardinality_rollouts",
                    getattr(self.config, "seed_verifier_rollouts", 1),
                )
            ),
        )
        rollouts: List[Dict[str, Any]] = []
        for rollout_id in range(rollout_count):
            report = self._run_answer_cardinality_request(
                target=target,
                question="",
                mode="seed_contract",
            )
            report = dict(report)
            report["rollout_id"] = rollout_id
            rollouts.append(report)

        active = [
            item for item in rollouts if item.get("verdict") != "skipped"
        ]
        if not active:
            return {**rollouts[-1], "rollouts": rollouts}

        plural = [item for item in active if item.get("verdict") == "plural"]
        if plural:
            alternatives: List[Dict[str, Any]] = []
            seen: set[tuple[str, str]] = set()
            for item in plural:
                for alt in item.get("alternate_answers", []):
                    key = (
                        _answer_key(str(alt.get("answer") or "")),
                        _answer_key(str(alt.get("role_check") or "")),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    alternatives.append(dict(alt))
            return {
                "verdict": "plural",
                "reason": " | ".join(
                    str(item.get("reason") or "") for item in plural
                ),
                "expected_answer_supported": all(
                    item.get("expected_answer_supported") is True for item in plural
                ),
                "alternate_answers": alternatives,
                "missing_disambiguator": next(
                    (
                        str(item.get("missing_disambiguator") or "")
                        for item in plural
                        if str(item.get("missing_disambiguator") or "")
                    ),
                    "",
                ),
                "rollouts": rollouts,
            }

        all_single = all(item.get("verdict") == "single" for item in active)
        all_supported = all(
            item.get("expected_answer_supported") is True for item in active
        )
        if all_single and all_supported and len(active) == rollout_count:
            return {
                "verdict": "single",
                "reason": " | ".join(
                    str(item.get("reason") or "") for item in active
                ),
                "expected_answer_supported": True,
                "alternate_answers": [],
                "missing_disambiguator": "",
                "rollouts": rollouts,
            }

        failed = any(item.get("verdict") == "check_failed" for item in active)
        return {
            "verdict": "check_failed" if failed else "uncertain",
            "reason": "independent cardinality rollouts did not unanimously verify "
            "a complete answer contract: "
            + " | ".join(str(item.get("reason") or "") for item in active),
            "expected_answer_supported": (
                False
                if any(item.get("expected_answer_supported") is False for item in active)
                else None
            ),
            "alternate_answers": [],
            "missing_disambiguator": next(
                (
                    str(item.get("missing_disambiguator") or "")
                    for item in active
                    if str(item.get("missing_disambiguator") or "")
                ),
                "",
            ),
            "rollouts": rollouts,
        }

    def _repair_seed_fact(
        self,
        target: Target,
        verification: Dict[str, Any],
        *,
        attempt: int,
    ) -> tuple[Target | None, Dict[str, Any]]:
        """Correct a rejected answer contract without changing the target entity."""
        response = self.runner.run_json(
            self.config.agents["seed_repair"],
            system_prompt=SEED_REPAIR_PROMPT,
            user_payload={
                "attempt": attempt,
                "original_seed": asdict(target),
                "verification_report": verification,
            },
            rate_limit_scope=target.entity_id,
        )
        if not isinstance(response, dict):
            return None, {"action": "reject", "reason": "seed repair returned non-object JSON"}
        if str(response.get("action") or "").strip().lower() != "correct":
            return None, response
        raw_target = response.get("target")
        if not isinstance(raw_target, dict):
            return None, {**response, "action": "reject", "reason": "seed repair omitted target"}
        immutable = ("entity_id", "name", "entity_type")
        for key in immutable:
            if str(raw_target.get(key) or "").strip() != str(getattr(target, key)).strip():
                return None, {
                    **response,
                    "action": "reject",
                    "reason": f"seed repair changed immutable target field {key}",
                }
        source_urls = raw_target.get("source_urls")
        if not isinstance(source_urls, list):
            source_urls = list(target.source_urls)
        candidate_data = asdict(target)
        candidate_data.update(
            {
                "answer_field": str(raw_target.get("answer_field") or "").strip(),
                "answer": str(raw_target.get("answer") or "").strip(),
                "description": str(raw_target.get("description") or target.description).strip(),
                "source_urls": [str(item).strip() for item in source_urls if str(item).strip()],
                "domain_family": str(raw_target.get("domain_family") or target.domain_family).strip(),
                "domain_subtype": str(raw_target.get("domain_subtype") or target.domain_subtype).strip(),
            }
        )
        if not candidate_data["answer_field"] or not candidate_data["answer"] or not candidate_data["source_urls"]:
            return None, {
                **response,
                "action": "reject",
                "reason": "seed repair must provide non-empty answer_field, answer, and source_urls",
            }
        corrected = Target(**candidate_data)
        answer_field_problem = _seed_answer_field_problem(corrected)
        if answer_field_problem:
            return None, {
                **response,
                "action": "reject",
                "reason": f"seed repair returned invalid answer_field: {answer_field_problem}",
            }
        return corrected, response

    def run_seeds_parallel(
        self,
        seeds: Sequence[SeedRecord],
        *,
        output_dir: Path,
        dry_run: bool = False,
        indexes: Sequence[int] | None = None,
    ) -> List[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        seed_indexes = list(indexes) if indexes is not None else list(
            range(1, len(seeds) + 1)
        )
        if len(seed_indexes) != len(seeds):
            raise ValueError("indexes length must match seeds length")
        max_workers = max(1, min(self.config.seed_concurrency, len(seeds)))
        self._log(f"seed_parallel_start: seeds={len(seeds)} concurrency={max_workers}")
        results: List[Path] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.run_seed_and_save,
                    seed,
                    output_dir=output_dir,
                    index=idx,
                    dry_run=dry_run,
                ): idx
                for idx, seed in zip(seed_indexes, seeds)
            }
            for future in as_completed(futures):
                results.append(future.result())
        self._log(f"seed_parallel_done: artifacts={len(results)}")
        return sorted(results)

    def run_seed_and_save(
        self,
        seed: SeedRecord,
        *,
        output_dir: Path,
        index: int,
        dry_run: bool = False,
    ) -> Path:
        """Run one seed and persist either its artifact or a captured error."""
        output_dir.mkdir(parents=True, exist_ok=True)
        seed_dir = output_dir / f"seed_{index:03d}_{_safe_slug(seed.target.entity_id)}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        out_path = seed_dir / "artifact.json"
        run_context = {
            "seed_index": index,
            "seed_id": seed.target.entity_id,
            "seed_dir": str(seed_dir.absolute()),
            "artifact_path": str(out_path.absolute()),
        }
        try:
            artifact = self.run_seed(
                seed,
                dry_run=dry_run,
                run_context=run_context,
            )
        except Exception as exc:
            error_type = type(exc).__name__
            error_message = str(exc)
            traceback_text = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
            workflow_error = {
                "stage": "workflow",
                "status": "failed",
                "error_type": error_type,
                "error": error_message,
                "traceback": traceback_text,
            }
            if out_path.is_file():
                try:
                    partial = json.loads(out_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    partial = {}
                if isinstance(partial, dict) and partial:
                    partial["status"] = "rejected:workflow_error"
                    partial.setdefault("iterations", []).append(workflow_error)
                    partial["run_context"] = run_context
                    out_path.write_text(
                        json.dumps(partial, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    artifact = None
                else:
                    artifact = Artifact(
                        target=seed.target,
                        status="rejected:workflow_error",
                        notes=[seed.seed_note] if seed.seed_note else [],
                        iterations=[
                            {"stage": "seed", "status": "loaded", "target": asdict(seed.target)},
                            workflow_error,
                        ],
                        run_context=run_context,
                    )
            else:
                artifact = Artifact(
                    target=seed.target,
                    status="rejected:workflow_error",
                    notes=[seed.seed_note] if seed.seed_note else [],
                    iterations=[
                        {"stage": "seed", "status": "loaded", "target": asdict(seed.target)},
                        workflow_error,
                    ],
                    run_context=run_context,
                )
            self._log(
                "seed_workflow_error: "
                f"target={seed.target.entity_id} error={error_type}: {error_message}"
            )
        if artifact is not None:
            save_artifact(artifact, out_path)
            final_status = artifact.status
        else:
            final_status = "rejected:workflow_error"
        if self.config.verbose_progress:
            print(
                f"seed_done: idx={index} status={final_status} out={out_path}",
                flush=True,
            )
        return out_path

    def _verify_program_constraints(self, artifact: Artifact):
        verifier = (
            verify_pairwise_core_paths
            if self.config.root_pairwise_gate_enabled
            else verify_core_paths
        )
        report = verifier(
            artifact.target,
            artifact.constraints,
            min_core_paths=self.config.min_core_paths,
            min_distractor_paths=self.config.min_distractor_paths,
            max_distractor_paths=self.config.max_distractor_paths,
            min_single_candidates=self.config.min_single_candidates,
            max_single_candidates=self.config.max_single_candidates,
            **(
                {}
                if self.config.root_pairwise_gate_enabled
                else {
                    "min_pairwise_core_intersection": self.config.min_pairwise_core_intersection,
                }
            ),
            min_distractor_core_overlap=self.config.min_distractor_core_overlap,
            min_distractors_per_core=self.config.min_distractors_per_core,
        )
        if not report.accepted:
            return report
        core_ids = [
            path.path_id for path in artifact.constraints if path.role == "core"
        ]
        quality_failure = _root_constraint_quality_failure(
            artifact.target,
            artifact.constraints,
            min_expandable_core_paths=max(0, len(core_ids) - 1),
            require_source_diversity=self.config.root_require_source_diversity,
        )
        if quality_failure:
            return replace(
                report,
                accepted=False,
                reason=(
                    "root_quality:"
                    + json.dumps(quality_failure, ensure_ascii=False, separators=(",", ":"))
                ),
            )
        if self.config.root_ambiguity_verifier_enabled:
            if self.config.root_pairwise_gate_enabled:
                ambiguity = self._run_root_pairwise_ambiguity_verifier(
                    artifact,
                    required_core_path_ids=core_ids,
                )
            else:
                ambiguity = self._run_root_ambiguity_verifier(
                    artifact,
                    required_core_path_ids=core_ids,
                )
            artifact.root_ambiguity_report = ambiguity
            if not ambiguity["accepted"]:
                return replace(
                    report,
                    accepted=False,
                    reason=(
                        "root_ambiguity:"
                        + json.dumps(
                            ambiguity,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    ),
                )
        return report

    def _run_root_ambiguity_verifier(
        self,
        artifact: Artifact,
        *,
        required_core_path_ids: Sequence[str] = (),
    ) -> Dict[str, Any]:
        """Batch-check that no Root clue is a real-world single-clue shortcut."""
        agent = replace(
            self.config.agents["constraint"],
            name="root_ambiguity",
            max_iterations=min(self.config.agents["constraint"].max_iterations, 12),
        )
        try:
            response = self.runner.run_json(
                agent,
                system_prompt=ROOT_AMBIGUITY_PROMPT,
                user_payload={
                    "evaluation_date": date.today().isoformat(),
                    "target": _root_target_view(artifact.target),
                    "root_clues": [
                        {
                            "path_id": path.path_id,
                            "role": path.role,
                            "clue": path.clue,
                            "candidates": list(path.candidates),
                            "evidence_urls": [
                                evidence.url for evidence in path.evidence if evidence.url
                            ],
                        }
                        for path in artifact.constraints
                    ],
                    "required_core_path_ids": list(required_core_path_ids),
                },
                rate_limit_scope=artifact.target.entity_id,
                response_validator=_validate_root_ambiguity_response,
            )
        except Exception as exc:
            return {
                "accepted": False,
                "failure_count": 1,
                "failures": [
                    {
                        "code": "root_ambiguity_check_failed",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                ],
                "path_results": [],
            }
        raw_results = response.get("path_results")
        raw_results = raw_results if isinstance(raw_results, list) else []
        by_id = {
            str(item.get("path_id") or ""): item
            for item in raw_results
            if isinstance(item, dict) and str(item.get("path_id") or "")
        }
        failures: List[Dict[str, Any]] = []
        single_clue_shortcuts: List[Dict[str, Any]] = []
        target_aliases = {
            _entity_text_key(artifact.target.entity_id),
            _entity_text_key(artifact.target.name),
        }
        target_aliases.discard("")
        for path in artifact.constraints:
            item = by_id.get(path.path_id)
            if item is None:
                failures.append(
                    {
                        "code": "root_ambiguity_result_missing",
                        "path_id": path.path_id,
                    }
                )
                continue
            raw_alternatives = item.get("alternatives")
            raw_alternatives = (
                raw_alternatives if isinstance(raw_alternatives, list) else []
            )
            alternatives: List[str] = []
            alternative_keys: set[str] = set()
            for alternative in raw_alternatives:
                name = str(
                    alternative.get("name")
                    if isinstance(alternative, dict)
                    else alternative
                ).strip()
                key = _entity_text_key(name)
                if not name or not key or key in target_aliases or key in alternative_keys:
                    continue
                alternative_keys.add(key)
                alternatives.append(name)
            if item.get("target_satisfies") is not True:
                failures.append(
                    {
                        "code": "root_target_semantic_failure",
                        "path_id": path.path_id,
                        "reason": str(item.get("reason") or ""),
                        "source_urls": item.get("source_urls", []),
                    }
                )
            if item.get("individually_identifying") is True or len(alternatives) < 2:
                single_clue_shortcuts.append(
                    {
                        "code": "root_single_clue_shortcut",
                        "path_id": path.path_id,
                        "alternatives": alternatives,
                        "reason": str(item.get("reason") or ""),
                        "source_urls": item.get("source_urls", []),
                    }
                )
                failures.append(
                    {
                        "code": "root_single_clue_shortcut",
                        "path_id": path.path_id,
                        "alternatives": alternatives,
                        "reason": str(item.get("reason") or ""),
                        "source_urls": item.get("source_urls", []),
                    }
                )
        joint = response.get("joint_result")
        joint = joint if isinstance(joint, dict) else {}
        joint_alternatives = joint.get("alternatives")
        joint_alternatives = (
            joint_alternatives if isinstance(joint_alternatives, list) else []
        )
        expected_joint_ids = list(required_core_path_ids)
        returned_joint_ids = (
            [str(item) for item in joint.get("path_ids", []) if str(item)]
            if isinstance(joint.get("path_ids"), list)
            else []
        )
        if (
            returned_joint_ids != expected_joint_ids
            or joint.get("target_satisfies") is not True
            or joint.get("unique") is not True
            or joint_alternatives
        ):
            failures.append(
                {
                    "code": "root_joint_not_unique",
                    "path_ids": expected_joint_ids,
                    "returned_path_ids": returned_joint_ids,
                    "target_satisfies": joint.get("target_satisfies"),
                    "alternatives": joint_alternatives,
                    "reason": str(
                        joint.get("reason")
                        or "required Root bundle was not verified unique in the real world"
                    ),
                }
            )
        return {
            "accepted": not failures,
            "failure_count": len(failures),
            "failures": failures,
            "single_clue_shortcuts": single_clue_shortcuts,
            "path_results": raw_results,
            "joint_result": joint,
        }

    def _run_root_pairwise_ambiguity_verifier(
        self,
        artifact: Artifact,
        *,
        required_core_path_ids: Sequence[str] = (),
    ) -> Dict[str, Any]:
        """Strict optional audit: every Core pair must retain a real alternative."""
        agent = replace(
            self.config.agents["constraint"],
            name="root_pairwise_ambiguity",
            max_iterations=min(self.config.agents["constraint"].max_iterations, 16),
        )
        required_pairs = list(combinations(required_core_path_ids, 2))
        try:
            response = self.runner.run_json(
                agent,
                system_prompt=ROOT_PAIRWISE_AMBIGUITY_PROMPT,
                user_payload={
                    "evaluation_date": date.today().isoformat(),
                    "target": _root_target_view(artifact.target),
                    "root_clues": [
                        {
                            "path_id": path.path_id,
                            "role": path.role,
                            "clue": path.clue,
                            "candidates": list(path.candidates),
                            "evidence_urls": [
                                evidence.url
                                for evidence in path.evidence
                                if evidence.url
                            ],
                        }
                        for path in artifact.constraints
                    ],
                    "required_core_path_ids": list(required_core_path_ids),
                    "required_core_pairs": [list(pair) for pair in required_pairs],
                },
                rate_limit_scope=artifact.target.entity_id,
                response_validator=_validate_root_pairwise_ambiguity_response,
            )
        except Exception as exc:
            return {
                "accepted": False,
                "pairwise_gate_enabled": True,
                "failure_count": 1,
                "failures": [
                    {
                        "code": "root_pairwise_ambiguity_check_failed",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                ],
                "path_results": [],
                "pair_results": [],
            }

        raw_results = response.get("path_results")
        raw_results = raw_results if isinstance(raw_results, list) else []
        by_id = {
            str(item.get("path_id") or ""): item
            for item in raw_results
            if isinstance(item, dict) and str(item.get("path_id") or "")
        }
        target_aliases = {
            _entity_text_key(artifact.target.entity_id),
            _entity_text_key(artifact.target.name),
        }
        target_aliases.discard("")

        def alternatives_for(item: Any) -> List[Any]:
            raw = item.get("alternatives", []) if isinstance(item, dict) else []
            raw = raw if isinstance(raw, list) else []
            values: List[Any] = []
            seen: set[str] = set()
            for alternative in raw:
                name = str(
                    alternative.get("name")
                    if isinstance(alternative, dict)
                    else alternative
                ).strip()
                key = _entity_text_key(name)
                if name and key and key not in target_aliases and key not in seen:
                    seen.add(key)
                    values.append(alternative)
            return values

        failures: List[Dict[str, Any]] = []
        single_clue_shortcuts: List[Dict[str, Any]] = []
        for path in artifact.constraints:
            item = by_id.get(path.path_id)
            if item is None:
                failures.append(
                    {"code": "root_ambiguity_result_missing", "path_id": path.path_id}
                )
                continue
            alternatives = alternatives_for(item)
            if item.get("target_satisfies") is not True:
                failures.append(
                    {
                        "code": "root_target_semantic_failure",
                        "path_id": path.path_id,
                        "reason": str(item.get("reason") or ""),
                        "source_urls": item.get("source_urls", []),
                    }
                )
            if path.role == "core" and (
                item.get("individually_identifying") is True
                or len(alternatives) < 2
            ):
                failure = {
                    "code": "root_single_clue_shortcut",
                    "path_id": path.path_id,
                    "alternatives": alternatives,
                    "reason": str(item.get("reason") or ""),
                    "source_urls": item.get("source_urls", []),
                }
                single_clue_shortcuts.append(failure)
                failures.append(failure)

        raw_pair_results = response.get("pair_results")
        raw_pair_results = (
            raw_pair_results if isinstance(raw_pair_results, list) else []
        )
        pair_by_ids = {
            frozenset(str(path_id) for path_id in item.get("path_ids", [])): item
            for item in raw_pair_results
            if isinstance(item, dict)
            and isinstance(item.get("path_ids"), list)
            and len(item.get("path_ids")) == 2
        }
        for expected_pair in required_pairs:
            item = pair_by_ids.get(frozenset(expected_pair))
            alternatives = alternatives_for(item)
            if (
                item is None
                or item.get("target_satisfies") is not True
                or item.get("unique") is not False
                or not alternatives
            ):
                failures.append(
                    {
                        "code": "root_pair_identifying",
                        "path_ids": list(expected_pair),
                        "alternatives": alternatives,
                        "reason": str(
                            (item or {}).get("reason")
                            or "required core pair lacks a verified non-target alternative"
                        ),
                    }
                )

        joint = response.get("joint_result")
        joint = joint if isinstance(joint, dict) else {}
        joint_alternatives = alternatives_for(joint)
        returned_joint_ids = (
            [str(item) for item in joint.get("path_ids", []) if str(item)]
            if isinstance(joint.get("path_ids"), list)
            else []
        )
        if (
            returned_joint_ids != list(required_core_path_ids)
            or joint.get("target_satisfies") is not True
            or joint.get("unique") is not True
            or joint_alternatives
        ):
            failures.append(
                {
                    "code": "root_joint_not_unique",
                    "path_ids": list(required_core_path_ids),
                    "returned_path_ids": returned_joint_ids,
                    "target_satisfies": joint.get("target_satisfies"),
                    "alternatives": joint_alternatives,
                    "reason": str(
                        joint.get("reason")
                        or "required Root bundle was not verified unique"
                    ),
                }
            )
        return {
            "accepted": not failures,
            "pairwise_gate_enabled": True,
            "failure_count": len(failures),
            "failures": failures,
            "single_clue_shortcuts": single_clue_shortcuts,
            "path_results": raw_results,
            "pair_results": raw_pair_results,
            "joint_result": joint,
        }

    def _make_constraints(
        self,
        target: Target,
        *,
        previous_constraints: Sequence[ConstraintPath] = (),
        failure_report: Dict[str, Any] | None = None,
        revision: int = 0,
    ) -> List[ConstraintPath]:
        requirements = {
            "min_core_paths": self.config.min_core_paths,
            "min_distractor_paths": self.config.min_distractor_paths,
            "max_distractor_paths": self.config.max_distractor_paths,
            "min_single_candidates": self.config.min_single_candidates,
            "max_single_candidates": self.config.max_single_candidates,
            "max_attribute_core_paths": 1,
            "min_relation_core_paths": max(0, self.config.min_core_paths - 1),
        }
        if self.config.root_pairwise_gate_enabled:
            requirements.update(
                {
                    "pairwise_non_unique_required": True,
                    "min_pairwise_shared_candidates": 2,
                    "full_core_unique_required": True,
                }
            )
        user_payload: Dict[str, Any] = {
            "target": _root_target_view(target),
            "requirements": requirements,
            "evaluation_date": date.today().isoformat(),
        }
        if previous_constraints:
            retry_feedback = _compact_root_failure_report(failure_report)
            retry_feedback["revision"] = revision
            retry_feedback["previous_core_dimensions"] = sorted(
                {
                    str(item.branch or "unspecified").split(":", 1)[0]
                    for item in previous_constraints
                    if item.role == "core"
                }
            )
            replacement_ids: set[str] = set()
            if retry_feedback.get("code") in {
                "root_pair_candidate_overlap",
                "root_candidate_count_out_of_bounds",
            }:
                replacement_ids.update(
                    str(item) for item in retry_feedback.get("editable_path_ids", [])
                )
            for item in retry_feedback.get("failures", []):
                if not isinstance(item, dict):
                    continue
                if str(item.get("code") or "") in {
                    "forbidden_reference_in_root_clue",
                    "root_target_membership_unsupported",
                    "root_clue_not_atomic",
                    "root_core_uses_seed_source",
                    "root_too_many_attribute_cores",
                    "root_core_relation_required",
                    "insufficient_expandable_core_paths",
                }:
                    path_id = str(item.get("path_id") or "")
                    if path_id:
                        replacement_ids.add(path_id)
                    for path_id in item.get("path_ids", []):
                        if str(path_id).strip():
                            replacement_ids.add(str(path_id))
            if replacement_ids:
                retry_feedback["replacement_path_ids"] = sorted(replacement_ids)
            user_payload["previous_constraints"] = [
                _constraint_revision_view(item)
                for item in previous_constraints
                if item.path_id not in replacement_ids
            ]
            user_payload["retry_feedback"] = retry_feedback
        response = self.runner.run_json(
            self.config.agents["constraint"],
            system_prompt=(
                ROOT_PAIRWISE_CONSTRAINT_PROMPT
                if self.config.root_pairwise_gate_enabled
                else CONSTRAINT_PROMPT
            ),
            user_payload=user_payload,
            rate_limit_scope=target.entity_id,
            response_validator=_validate_root_constraint_response,
        )
        paths = [
            ConstraintPath.from_dict(item)
            for item in response.get("constraints", [])
        ]
        _preserve_verified_root_candidates(
            paths,
            previous_constraints=previous_constraints,
            failure_report=failure_report,
        )
        for path in paths:
            path.local_target_id = ""
            path.local_target_name = ""
            path.local_target_canonical_name = ""
            path.local_target_type = ""
            path.local_constraints = []
        return paths

    def _local_fuzzify(self, artifact: Artifact) -> tuple[List[ConstraintPath], Dict[str, Any]]:
        reports: List[Dict[str, Any]] = []
        state = LocalExpansionState(
            root_roles={p.path_id: p.role for p in artifact.constraints},
        )
        updated = self._expand_local_siblings(
            artifact,
            artifact.constraints,
            depth=0,
            parent_node_path=[],
            reports=reports,
            state=state,
        )
        failures = [item for item in reports if not item.get("accepted", True)]
        # Global acceptance: require at least N root core paths to have
        # a descendant reaching the configured minimum leaf depth, rather
        # than requiring *every* core node to expand.
        min_leaf = self.config.question_min_leaf_core_paths
        min_depth = self.config.question_min_leaf_depth
        root_core_ids = {
            p.path_id for p in artifact.constraints if p.role == "core"
        }
        cores_at_leaf = set()
        for report in reports:
            if (
                report.get("accepted", True)
                and report.get("role") == "core"
                and report.get("depth", 0) >= min_depth
                and report.get("node_path")
                and report["node_path"][0] in root_core_ids
            ):
                cores_at_leaf.add(report["node_path"][0])
        global_accepted = len(cores_at_leaf) >= min_leaf
        return updated, {
            "accepted": global_accepted,
            "acceptance_reason": (
                f"{len(cores_at_leaf)} root core paths reached depth>={min_depth}; "
                f"require {min_leaf}"
            ),
            "leaf_core_root_ids": sorted(cores_at_leaf),
            "required_leaf_core_paths": min_leaf,
            "required_leaf_depth": min_depth,
            "max_depth": self.config.local_max_depth,
            "core_max_depth": self.config.local_core_max_depth,
            "distractor_max_depth": self.config.local_distractor_max_depth,
            "expand_min_paths": self.config.local_expand_min_paths,
            "expand_max_paths": self.config.local_expand_max_paths,
            "expand_core_min_paths": self.config.local_expand_core_min_paths,
            "expand_core_max_paths": self.config.local_expand_core_max_paths,
            "expand_distractor_min_paths": self.config.local_expand_distractor_min_paths,
            "expand_distractor_max_paths": self.config.local_expand_distractor_max_paths,
            "expand_distractor_prob": self.config.local_expand_distractor_prob,
            "expand_roles": self.config.local_expand_roles,
            "max_deep_paths": self.config.local_max_deep_paths,
            "deep_path_roles": self.config.local_deep_path_roles,
            "deep_paths": state.deep_paths,
            "deep_paths_per_root": dict(state.deep_paths_per_root),
            "checked_nodes": len(reports),
            "failures": failures,
            "reports": reports,
        }

    def _grow_question_leaf_paths(
        self,
        artifact: Artifact,
        question_response: Dict[str, Any],
    ) -> Dict[str, Any]:
        core_ids = set(artifact.verifier.core_path_ids if artifact.verifier else [])
        current_leaf_ids = set(question_response.get("leaf_core_path_ids") or [])
        missing_ids = list(question_response.get("missing_leaf_core_path_ids") or [])
        needed = len(missing_ids)
        if needed <= 0:
            return {
                "accepted": True,
                "reason": "every required root already has a deep leaf",
                "grown_paths": [],
                "leaf_core_path_ids": sorted(current_leaf_ids),
            }

        missing_set = set(missing_ids)
        candidate_indexes = [
            idx
            for idx, path in enumerate(artifact.constraints)
            if path.path_id in core_ids and path.path_id in missing_set
        ]
        if not candidate_indexes:
            candidate_indexes = [
                idx
                for idx, path in enumerate(artifact.constraints)
                if path.path_id in core_ids and path.path_id not in current_leaf_ids
            ]

        reports: List[Dict[str, Any]] = []
        grown_paths: List[str] = []
        state = LocalExpansionState(deep_paths=len(current_leaf_ids))
        for idx in candidate_indexes:
            if needed <= 0:
                break
            path = artifact.constraints[idx]
            before_view = self._question_constraint_view(path, require_leaf=True)
            item, item_reports = self._expand_local_path(
                artifact,
                path,
                depth=0,
                node_path=[path.path_id],
                state=state,
            )
            artifact.constraints[idx] = item
            reports.extend(item_reports)
            after_view = self._question_constraint_view(item, require_leaf=True)
            if (
                after_view.get("question_clue_source") == "leaf_node_clue"
                and after_view.get("question_clue_depth", 0) >= self.config.question_min_leaf_depth
                and before_view.get("question_clue") != after_view.get("question_clue")
            ):
                current_leaf_ids.add(item.path_id)
                grown_paths.append(item.path_id)
                needed -= 1

        return {
            "accepted": needed <= 0,
            "needed_after": needed,
            "leaf_core_path_ids": sorted(current_leaf_ids),
            "candidate_path_ids": [artifact.constraints[idx].path_id for idx in candidate_indexes],
            "grown_paths": grown_paths,
            "reports": reports,
        }

    def _expand_local_path(
        self,
        artifact: Artifact,
        path: ConstraintPath,
        *,
        depth: int,
        node_path: List[str],
        state: LocalExpansionState,
        ancestor_local_targets: Sequence[Dict[str, str]] = (),
    ) -> tuple[ConstraintPath, List[Dict[str, Any]]]:
        max_depth = self._local_max_depth_for(path)
        root_id = node_path[0] if node_path else ""
        if self._local_deep_limit_reached(state, root_id=root_id):
            return path, [
                {
                    "node_path": node_path,
                    "depth": depth,
                    "max_depth": max_depth,
                    "accepted": True,
                    "expanded": False,
                    "role": path.role,
                    "reason": "local deep path budget exhausted",
                    "deep_paths": state.deep_paths,
                }
            ]
        if depth >= max_depth:
            claimed = self._try_claim_deep_path(state, path, root_id=root_id)
            if not claimed:
                # Another sibling won the per-root terminal-path race while
                # this node was already in flight.  Keep this branch as a
                # valid shallow clue, but do not retain a second deep leaf.
                path.local_constraints = []
                return path, [
                    {
                        "node_path": node_path,
                        "depth": depth,
                        "max_depth": max_depth,
                        "accepted": True,
                        "expanded": False,
                        "role": path.role,
                        "reason": "local deep path budget exhausted by sibling",
                        "deep_paths": state.deep_paths,
                    }
                ]
            path.local_constraints = []
            deep_paths = state.deep_paths
            return path, [
                {
                    "node_path": node_path,
                    "depth": depth,
                    "max_depth": max_depth,
                    "accepted": True,
                    "expanded": False,
                    "role": path.role,
                    "reason": "local max depth reached",
                    "deep_paths": deep_paths,
                }
            ]
        self._log(
            "local_node_start: "
            f"target={artifact.target.entity_id} depth={depth}/{max_depth} "
            f"node={'/'.join(node_path)} role={path.role}"
        )
        base_user_payload = {
            "evaluation_date": date.today().isoformat(),
            "current_node": {
                "path_id": path.path_id,
                "clue": path.clue,
            },
            "requirements": {
                "min_core_paths": self.config.local_min_core_paths,
                "min_distractor_paths": self.config.local_min_distractor_paths,
                "min_expandable_core_paths": (
                    1 if depth + 1 < max_depth else 0
                ),
                "min_relation_core_paths": self.config.local_min_relation_core_paths,
                "min_attribute_core_paths": (
                    self.config.local_min_attribute_core_paths
                ),
            },
        }
        verification_failures: List[Dict[str, Any]] = []
        updated = path
        report: Dict[str, Any] = {}
        previous_response: Dict[str, Any] | None = None
        for verification_attempt in range(self.config.local_verification_retries + 1):
            user_payload = dict(base_user_payload)
            if verification_failures and previous_response is not None:
                user_payload["previous_attempt"] = previous_response
                user_payload["retry_feedback"] = verification_failures[-1]
            response: Dict[str, Any] = {}
            retries = max(0, int(os.environ.get("V2_LOCAL_AGENT_RETRIES", "2")))
            base_sleep = max(
                0.0,
                float(os.environ.get("V2_LOCAL_AGENT_RETRY_SLEEP_SECONDS", "20")),
            )
            for attempt in range(retries + 1):
                try:
                    local_agent = self.config.agents["local_constraint"]
                    if verification_attempt > 0:
                        retry_code = str(
                            verification_failures[-1].get("code") or ""
                        )
                        semantic_retry = retry_code in {
                            "child_core_branches_not_diverse",
                            "child_core_kind_imbalance",
                            "child_target_membership_unsupported",
                            "insufficient_expandable_local_core_paths",
                            "local_quality_failure",
                            "insufficient_non_shortcut_local_cores",
                            # A source-URL repair may require a fresh search;
                            # disabling tools here makes the model emit a
                            # tool-call-only response that cannot be parsed.
                            "child_source_url_reused",
                        }
                        local_agent = replace(
                            local_agent,
                            enabled_toolsets=(
                                local_agent.enabled_toolsets
                                if semantic_retry
                                else []
                            ),
                            max_iterations=min(
                                local_agent.max_iterations,
                                6 if semantic_retry else 3,
                            ),
                        )
                    else:
                        local_agent = replace(
                            local_agent,
                            max_iterations=min(local_agent.max_iterations, 6),
                        )
                    response = self.runner.run_json(
                        local_agent,
                        system_prompt=LOCAL_CONSTRAINT_PROMPT,
                        user_payload=user_payload,
                        rate_limit_scope=artifact.target.entity_id,
                        response_validator=_validate_local_constraint_response,
                    )
                    break
                except Exception as exc:
                    error_text = str(exc)
                    # AgentRunner already performs adaptive retries for 429s. Do not
                    # multiply that retry budget at the per-node workflow layer.
                    transient = (
                        not isinstance(exc, ProviderRateLimitError)
                        and _is_transient_agent_error(error_text)
                    )
                    if transient and attempt < retries:
                        sleep_seconds = base_sleep * (attempt + 1)
                        self._log(
                            "local_node_retry: "
                            f"target={artifact.target.entity_id} depth={depth} "
                            f"node={'/'.join(node_path)} attempt={attempt + 1}/{retries} "
                            f"sleep={sleep_seconds:.1f}s error={error_text[:180]}"
                        )
                        if sleep_seconds > 0:
                            time.sleep(sleep_seconds)
                        continue
                    report = {
                        "node_path": node_path,
                        "depth": depth,
                        "max_depth": max_depth,
                        "accepted": False,
                        "expanded": False,
                        "role": path.role,
                        "reason": "local agent failed",
                        "error": error_text,
                        "transient": transient,
                        "transport_attempts": attempt + 1,
                        "verification_attempts": verification_attempt + 1,
                        "failure_history": list(verification_failures),
                    }
                    self._log(
                        "local_node_error: "
                        f"target={artifact.target.entity_id} depth={depth} "
                        f"node={'/'.join(node_path)} transient={transient} "
                        f"error={error_text[:220]}"
                    )
                    return path, [report]

            if verification_failures and previous_response is not None:
                response = _merge_local_retry_response(
                    previous_response,
                    response,
                    verification_failures[-1],
                )
            previous_response = response
            updated = _apply_local_expansion_response(path, response)
            if not updated.local_constraints:
                # A relation clue is expected to contain a concrete entity anchor.
                # Give the Local model a bounded second chance when it stops before
                # the configured depth; attribute/meta clues may legitimately stop.
                if (
                    str(response.get("action") or "").strip().lower() == "stop"
                    and _local_branch_kind(path.branch) == "relation"
                    and depth < max_depth
                    and verification_attempt < self.config.local_verification_retries
                ):
                    report = {
                        "node_path": node_path,
                        "depth": depth,
                        "max_depth": max_depth,
                        "accepted": False,
                        "hard_accepted": False,
                        "expanded": False,
                        "role": updated.role,
                        "failure_kind": "local_stop_before_depth",
                        "reason": (
                            "relation clue stopped before the available depth; "
                            "retry and select an exact named entity from the clue"
                        ),
                        "verification_attempts": verification_attempt + 1,
                        "failure_history": list(verification_failures),
                    }
                    verification_failures.append(
                        _compact_local_retry_feedback(
                            report,
                            attempt=verification_attempt + 1,
                        )
                    )
                    previous_response = response
                    report["failure_history"] = list(verification_failures)
                    self._log(
                        "local_verification_retry: "
                        f"target={artifact.target.entity_id} depth={depth} "
                        f"node={'/'.join(node_path)} attempt={verification_attempt + 1}/"
                        f"{self.config.local_verification_retries} reason={report['reason']}"
                    )
                    continue
                # Per-node expansion is not an all-or-nothing gate. The global
                # local-fuzzify check counts roots that actually reached depth.
                report = {
                    "node_path": node_path,
                    "depth": depth,
                    "max_depth": max_depth,
                    "accepted": True,
                    "expanded": False,
                    "role": updated.role,
                    "reason": "no local constraints returned"
                    + (", core did not expand" if updated.role == "core" else ""),
                    "verification_attempts": verification_attempt + 1,
                    "failure_history": list(verification_failures),
                }
                self._log(
                    "local_node_done: "
                    f"target={artifact.target.entity_id} depth={depth} "
                    f"node={'/'.join(node_path)} expanded=False "
                    f"accepted={report['accepted']} reason={report['reason']}"
                )
                return updated, [report]

            report = self._verify_local_constraints(
                updated,
                node_path,
                depth,
                current_node_clue=path.clue,
                final_target=artifact.target,
                ancestor_local_targets=ancestor_local_targets,
                min_expandable_core_paths=(
                    1 if depth + 1 < max_depth else 0
                ),
            )
            report.update(
                {
                    "verification_attempts": verification_attempt + 1,
                    "failure_history": list(verification_failures),
                }
            )
            if (
                report.get("hard_accepted")
                and self.config.local_quality_verifier_enabled
            ):
                quality = self._run_local_quality_verifier(
                    updated,
                    rate_limit_scope=artifact.target.entity_id,
                )
                quality_accepted = _local_quality_is_accepted(quality)
                shortcut_failure = (
                    _local_shallow_shortcut_failure(
                        updated,
                        quality,
                        required=getattr(
                            self.config, "local_min_non_shortcut_root_cores", 0
                        ),
                    )
                    if depth == 0 and updated.role == "core"
                    else None
                )
                report["local_quality_verifier"] = quality
                if shortcut_failure:
                    report.update(
                        {
                            "accepted": not self.config.require_local_verification,
                            "hard_accepted": False,
                            "failure_kind": "insufficient_non_shortcut_local_cores",
                            **shortcut_failure,
                        }
                    )
                elif not quality_accepted:
                    report.update(
                        {
                            "accepted": not self.config.require_local_verification,
                            "hard_accepted": False,
                            "failure_kind": "local_quality_failure",
                            "reason": str(
                                quality.get("reason")
                                or "Local quality verifier found false or incoherent clues"
                            ),
                        }
                    )
            if report.get("hard_accepted", report["accepted"]):
                break
            failure = _compact_local_retry_feedback(
                report,
                attempt=verification_attempt + 1,
            )
            verification_failures.append(failure)
            report["failure_history"] = list(verification_failures)
            if verification_attempt >= self.config.local_verification_retries:
                break
            self._log(
                "local_verification_retry: "
                f"target={artifact.target.entity_id} depth={depth} "
                f"node={'/'.join(node_path)} attempt={verification_attempt + 1}/"
                f"{self.config.local_verification_retries} "
                f"reason={str(report.get('reason', ''))[:220]}"
            )

        reports = [report]
        self._log(
            "local_verify_done: "
            f"target={artifact.target.entity_id} depth={depth} node={'/'.join(node_path)} "
            f"accepted={report.get('accepted')} reason={report.get('reason')}"
        )
        if not report["accepted"] and self.config.require_local_verification:
            return path, reports

        if (
            depth < max_depth
            and self._local_deep_limit_reached(state, root_id=root_id)
        ):
            report = {
                **report,
                "expanded": True,
                "accepted": True,
                "reason": "another branch of this root reached the deep path budget",
            }
            return updated, reports

        updated.local_constraints = self._expand_local_siblings(
            artifact,
            updated.local_constraints,
            depth=depth + 1,
            parent_node_path=node_path,
            reports=reports,
            state=state,
            ancestor_local_targets=[
                *ancestor_local_targets,
                {
                    "entity_id": updated.local_target_id,
                    "name": updated.local_target_name,
                    "canonical_name": updated.local_target_canonical_name,
                    "entity_type": updated.local_target_type,
                },
            ] if updated.local_target_id or updated.local_target_name else list(ancestor_local_targets),
        )
        self._log(
            "local_node_done: "
            f"target={artifact.target.entity_id} depth={depth} node={'/'.join(node_path)} "
            f"children={len(updated.local_constraints)} reports={len(reports)}"
        )
        return updated, reports

    def _expand_local_siblings(
        self,
        artifact: Artifact,
        paths: Sequence[ConstraintPath],
        *,
        depth: int,
        parent_node_path: List[str],
        reports: List[Dict[str, Any]],
        state: LocalExpansionState,
        ancestor_local_targets: Sequence[Dict[str, str]] = (),
    ) -> List[ConstraintPath]:
        if not paths:
            return []
        selected_indexes = self._select_local_expand_indexes(
            artifact,
            paths,
            depth=depth,
            parent_node_path=parent_node_path,
            state=state,
        )
        selected = [(idx, paths[idx]) for idx in selected_indexes]
        if not selected:
            self._log(
                "local_siblings_skipped: "
                f"target={artifact.target.entity_id} depth={depth} count={len(paths)}"
            )
            return list(paths)
        self._log(
            "local_siblings_selected: "
            f"target={artifact.target.entity_id} depth={depth} selected={len(selected)}/{len(paths)} "
            f"indexes={selected_indexes}"
        )
        already_parallel = getattr(self._local_parallel_state, "active", False)
        max_workers = (
            1
            if already_parallel
            else max(1, min(self.config.local_concurrency, len(selected)))
        )
        if max_workers == 1:
            expanded = list(paths)
            for idx, path in selected:
                item, item_reports = self._expand_local_path(
                    artifact,
                    path,
                    depth=depth,
                    node_path=[*parent_node_path, path.path_id],
                    state=state,
                    ancestor_local_targets=ancestor_local_targets,
                )
                expanded[idx] = item
                reports.extend(item_reports)
            return expanded

        self._log(
            "local_siblings_parallel_start: "
            f"target={artifact.target.entity_id} depth={depth} selected={len(selected)} "
            f"count={len(paths)} concurrency={max_workers}"
        )
        expanded = list(paths)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._expand_local_path_in_parallel_worker,
                    artifact,
                    path,
                    depth=depth,
                    node_path=[*parent_node_path, path.path_id],
                    state=state,
                    ancestor_local_targets=ancestor_local_targets,
                ): idx
                for idx, path in selected
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    item, item_reports = future.result()
                except Exception as exc:
                    item = expanded[idx]
                    item_reports = [
                        {
                            "node_path": [*parent_node_path, item.path_id],
                            "depth": depth,
                            "max_depth": self._local_max_depth_for(item),
                            "accepted": False,
                            "expanded": False,
                            "role": item.role,
                            "reason": "local expansion worker failed",
                            "error": str(exc),
                            "transient": _is_transient_agent_error(str(exc)),
                        }
                    ]
                    self._log(
                        "local_worker_error: "
                        f"target={artifact.target.entity_id} depth={depth} "
                        f"node={'/'.join([*parent_node_path, item.path_id])} error={str(exc)[:220]}"
                    )
                expanded[idx] = item
                reports.extend(item_reports)
        self._log(
            "local_siblings_parallel_done: "
            f"target={artifact.target.entity_id} depth={depth} selected={len(selected)} count={len(paths)}"
        )
        return expanded

    def _run_local_quality_verifier(
        self,
        path: ConstraintPath,
        *,
        rate_limit_scope: str,
    ) -> Dict[str, Any]:
        core_paths = [
            child for child in path.local_constraints if child.role == "core"
        ]
        distractor_paths = [
            child for child in path.local_constraints if child.role == "distractor"
        ]
        agent = replace(
            self.config.agents["local_constraint"],
            name="local_quality",
            max_iterations=min(
                self.config.agents["local_constraint"].max_iterations,
                6,
            ),
        )
        try:
            response = self.runner.run_json(
                agent,
                system_prompt=LOCAL_QUALITY_PROMPT,
                user_payload={
                    "local_target": {
                        "entity_id": path.local_target_id,
                        "canonical_name": (
                            path.local_target_canonical_name
                            or path.local_target_name
                        ),
                        "entity_type": path.local_target_type,
                    },
                    "evaluation_date": date.today().isoformat(),
                    "core_clues": [
                        {
                            "path_id": child.path_id,
                            "branch": child.branch,
                            "clue": child.clue,
                        }
                        for child in core_paths
                    ],
                    "distractor_clues": [
                        {
                            "path_id": child.path_id,
                            "branch": child.branch,
                            "clue": child.clue,
                        }
                        for child in distractor_paths
                    ],
                },
                rate_limit_scope=rate_limit_scope,
            )
        except Exception as exc:
            # This verifier is a soft factual diagnostic.  A provider timeout,
            # moderation rejection, or transient channel error must not turn a
            # structurally valid Local expansion into a failed tree; retain the
            # error in the artifact for later inspection.
            return {
                "skipped": True,
                "reason": f"local quality verifier unavailable: {type(exc).__name__}: {exc}",
                "error_type": type(exc).__name__,
            }
        if isinstance(response, dict) and (
            response.get("error")
            or response.get("type") in {
                "data_inspection_failed",
                "bad_request",
                "provider_unavailable",
                "timeout",
            }
            or (
                "message" in response
                and not any(
                    key in response
                    for key in ("target_valid", "coherent", "jointly_identifying")
                )
            )
        ):
            return {
                "skipped": True,
                "reason": str(
                    response.get("message")
                    or response.get("error")
                    or "local quality verifier returned a provider error"
                ),
                "provider_error": response,
            }
        target_valid = response.get("target_valid")
        if isinstance(target_valid, str):
            target_valid = target_valid.strip().lower() in {"true", "yes", "valid"}
        coherent = response.get("coherent")
        if isinstance(coherent, str):
            coherent = coherent.strip().lower() in {"true", "yes", "coherent"}
        jointly_identifying = response.get("jointly_identifying")
        if isinstance(jointly_identifying, str):
            jointly_identifying = jointly_identifying.strip().lower() in {
                "true",
                "yes",
                "unique",
            }
        raw_suggestions = (
            response.get("repair_suggestions")
            if isinstance(response.get("repair_suggestions"), list)
            else []
        )
        repair_suggestions = [item for item in raw_suggestions if isinstance(item, dict)]
        return {
            "target_valid": target_valid if isinstance(target_valid, bool) else None,
            "coherent": coherent if isinstance(coherent, bool) else None,
            "jointly_identifying": (
                jointly_identifying
                if isinstance(jointly_identifying, bool)
                else None
            ),
            "joint_resolution_reason": str(
                response.get("joint_resolution_reason") or ""
            ),
            "reason": str(response.get("reason") or ""),
            "alternatives": (
                response.get("alternatives")
                if isinstance(response.get("alternatives"), list)
                else []
            ),
            "single_clue_shortcuts": (
                response.get("single_clue_shortcuts")
                if isinstance(response.get("single_clue_shortcuts"), list)
                else []
            ),
            "target_failures": (
                response.get("target_failures")
                if isinstance(response.get("target_failures"), list)
                else []
            ),
            "repair_suggestions": repair_suggestions,
        }

    def _expand_local_path_in_parallel_worker(
        self,
        artifact: Artifact,
        path: ConstraintPath,
        *,
        depth: int,
        node_path: List[str],
        state: LocalExpansionState,
        ancestor_local_targets: Sequence[Dict[str, str]] = (),
    ) -> tuple[ConstraintPath, List[Dict[str, Any]]]:
        previous = getattr(self._local_parallel_state, "active", False)
        self._local_parallel_state.active = True
        try:
            return self._expand_local_path(
                artifact,
                path,
                depth=depth,
                node_path=node_path,
                state=state,
                ancestor_local_targets=ancestor_local_targets,
            )
        finally:
            self._local_parallel_state.active = previous

    def _verify_local_constraints(
        self,
        path: ConstraintPath,
        node_path: List[str],
        depth: int,
        *,
        current_node_clue: str | None = None,
        final_target: Target | None = None,
        ancestor_local_targets: Sequence[Dict[str, str]] = (),
        min_expandable_core_paths: int = 0,
    ) -> Dict[str, Any]:
        if not path.local_target_id and not path.local_target_name:
            return {
                "node_path": node_path,
                "depth": depth,
                "accepted": not self.config.require_local_verification,
                "hard_accepted": False,
                "expanded": True,
                "failure_kind": "missing_local_target",
                "reason": "local_constraints returned without local_target_id/local_target_name",
            }
        local_target = Target(
            entity_id=path.local_target_id or path.local_target_name,
            name=path.local_target_name or path.local_target_id,
            entity_type=path.local_target_type or path.terminal_type or "intermediate entity",
            answer_field="",
            answer="",
            description=f"local target for {'/'.join(node_path)}",
            source_urls=[],
        )
        source_clue = path.clue if current_node_clue is None else current_node_clue
        surface_text = path.local_target_name.strip()
        if surface_text not in source_clue:
            return {
                "node_path": node_path,
                "depth": depth,
                "expanded": True,
                "accepted": not self.config.require_local_verification,
                "hard_accepted": False,
                "failure_kind": "local_target_not_in_current_clue",
                "local_target": asdict(local_target),
                "current_node_clue": source_clue,
                "reason": (
                    f"local target surface_text {surface_text!r} is not an exact "
                    "contiguous substring copied from current node clue"
                ),
            }
        surface_problem = _local_target_surface_problem(
            surface_text,
            path.local_target_type,
        )
        if surface_problem:
            return {
                "node_path": node_path,
                "depth": depth,
                "expanded": True,
                "accepted": not self.config.require_local_verification,
                "hard_accepted": False,
                "failure_kind": "local_target_not_useful_entity",
                "local_target": asdict(local_target),
                "current_node_clue": source_clue,
                "reason": surface_problem,
            }
        selected_target_leak = _selected_local_target_forbidden_reference(
            surface_text,
            final_target=final_target,
            ancestor_local_targets=ancestor_local_targets,
        )
        if selected_target_leak:
            return {
                "node_path": node_path,
                "depth": depth,
                "expanded": True,
                "accepted": not self.config.require_local_verification,
                "hard_accepted": False,
                "failure_kind": "forbidden_local_target_selection",
                "local_target": asdict(local_target),
                "current_node_clue": source_clue,
                "forbidden_reference": selected_target_leak,
                "reason": (
                    "selected local target matches a private forbidden entity: "
                    f"{selected_target_leak['kind']}"
                ),
            }
        textual_reference = surface_text
        candidate_verifier = _verify_local_tree_structure(
            local_target,
            path.local_constraints,
            min_core_paths=self.config.local_min_core_paths,
            min_distractor_paths=self.config.local_min_distractor_paths,
        )
        if final_target is not None:
            child_leaks = _find_local_child_clue_leaks(
                path.local_constraints,
                final_target=final_target,
                ancestor_local_targets=ancestor_local_targets,
                current_local_target=local_target,
                current_local_target_canonical_name=(
                    path.local_target_canonical_name
                ),
            )
            if child_leaks:
                first_leak = child_leaks[0]
                unsupported_children = [
                    child.path_id
                    for child in path.local_constraints
                    if not _local_evidence_mentions_target(
                        child,
                        local_target,
                        path.local_target_canonical_name,
                    )
                ]
                return {
                    "node_path": node_path,
                    "depth": depth,
                    "expanded": True,
                    "accepted": not self.config.require_local_verification,
                    "hard_accepted": False,
                    "failure_kind": "forbidden_reference_in_child_clue",
                    "local_target": asdict(local_target),
                    "child_clue_leak": first_leak,
                    "child_clue_leaks": child_leaks,
                    "offending_child_path_ids": unsupported_children,
                    "current_node_clue": source_clue,
                    "local_target_text_reference": textual_reference,
                    "verifier": asdict(candidate_verifier),
                    "reason": (
                        f"{len(child_leaks)} child clues contain forbidden references: "
                        + ", ".join(
                            f"{item['path_id']}={item['matched_value']!r}"
                            for item in child_leaks
                        )
                        + (
                            "; target-membership evidence is also missing for: "
                            + ", ".join(unsupported_children)
                            if unsupported_children
                            else ""
                        )
                    ),
                }
        for child in path.local_constraints:
            word_count = len(re.findall(r"\b\w+\b", child.clue or ""))
            if word_count == 0 or word_count > 40:
                return {
                    "node_path": node_path,
                    "depth": depth,
                    "expanded": True,
                    "accepted": not self.config.require_local_verification,
                    "hard_accepted": False,
                    "failure_kind": "child_clue_not_atomic",
                    "offending_child_path_id": child.path_id,
                    "word_count": word_count,
                    "reason": (
                        f"child clue {child.path_id} has {word_count} words; "
                        "expected 1-40"
                    ),
                    "verifier": asdict(candidate_verifier),
                }
            if not any(evidence.url.strip() for evidence in child.evidence):
                return {
                    "node_path": node_path,
                    "depth": depth,
                    "expanded": True,
                    "accepted": not self.config.require_local_verification,
                    "hard_accepted": False,
                    "failure_kind": "child_clue_missing_evidence",
                    "offending_child_path_id": child.path_id,
                    "reason": (
                        f"child clue {child.path_id} has no evidence URL"
                    ),
                    "verifier": asdict(candidate_verifier),
                }
        parent_source_urls = {
            _source_url_key(evidence.url)
            for evidence in path.evidence
            if _source_url_key(evidence.url)
        }
        reused_child_source_urls = {
            child.path_id: sorted(
                parent_source_urls
                & {
                    _source_url_key(evidence.url)
                    for evidence in child.evidence
                    if _source_url_key(evidence.url)
                }
            )
            for child in path.local_constraints
        }
        reused_child_source_urls = {
            child_id: urls
            for child_id, urls in reused_child_source_urls.items()
            if urls
        }
        if (
            reused_child_source_urls
            and getattr(self.config, "local_require_distinct_parent_child_sources", False)
        ):
            return {
                "node_path": node_path,
                "depth": depth,
                "expanded": True,
                "accepted": not self.config.require_local_verification,
                "hard_accepted": False,
                "failure_kind": "child_source_url_reused",
                "offending_child_path_ids": sorted(reused_child_source_urls),
                "reused_source_urls": reused_child_source_urls,
                "reason": (
                    "child evidence must use a source URL different from the "
                    "current node clue evidence"
                ),
                "verifier": asdict(candidate_verifier),
            }
        seen_clues: Dict[str, str] = {}
        duplicate_child_ids: List[str] = []
        for child in path.local_constraints:
            clue_key = _entity_text_key(child.clue)
            if clue_key in seen_clues:
                duplicate_child_ids.append(child.path_id)
            else:
                seen_clues[clue_key] = child.path_id
        if duplicate_child_ids:
            return {
                "node_path": node_path,
                "depth": depth,
                "expanded": True,
                "accepted": not self.config.require_local_verification,
                "hard_accepted": False,
                "failure_kind": "child_clues_not_distinct",
                "offending_child_path_ids": duplicate_child_ids,
                "reason": (
                    "local child clues must be textually distinct; replace: "
                    + ", ".join(duplicate_child_ids)
                ),
                "verifier": asdict(candidate_verifier),
            }
        cores = [child for child in path.local_constraints if child.role == "core"]
        branch_values = [child.branch.strip().casefold() for child in cores]
        if any(not branch for branch in branch_values) or len(set(branch_values)) != len(
            branch_values
        ):
            return {
                "node_path": node_path,
                "depth": depth,
                "expanded": True,
                "accepted": not self.config.require_local_verification,
                "hard_accepted": False,
                "failure_kind": "child_core_branches_not_diverse",
                "offending_child_path_ids": [child.path_id for child in cores],
                "branches": {
                    child.path_id: child.branch for child in cores
                },
                "reason": (
                    "local core branch labels must be non-empty and distinct"
                ),
                "verifier": asdict(candidate_verifier),
            }
        branch_kinds = {
            child.path_id: _local_branch_kind(child.branch) for child in cores
        }
        relation_ids = [
            path_id for path_id, kind in branch_kinds.items() if kind == "relation"
        ]
        attribute_ids = [
            path_id for path_id, kind in branch_kinds.items() if kind == "attribute"
        ]
        required_relations = min(
            len(cores), self.config.local_min_relation_core_paths
        )
        required_attributes = min(
            max(0, len(cores) - required_relations),
            self.config.local_min_attribute_core_paths,
        )
        if (
            len(relation_ids) < required_relations
            or len(attribute_ids) < required_attributes
            or len(relation_ids) + len(attribute_ids) != len(cores)
        ):
            return {
                "node_path": node_path,
                "depth": depth,
                "expanded": True,
                "accepted": not self.config.require_local_verification,
                "hard_accepted": False,
                "failure_kind": "child_core_kind_imbalance",
                "offending_child_path_ids": [
                    child.path_id
                    for child in cores
                    if branch_kinds[child.path_id] not in {"relation", "attribute"}
                ] or [child.path_id for child in cores],
                "required_relation_core_paths": required_relations,
                "required_attribute_core_paths": required_attributes,
                "relation_core_path_ids": relation_ids,
                "attribute_core_path_ids": attribute_ids,
                "branches": {child.path_id: child.branch for child in cores},
                "reason": (
                    "local core bundle lacks the required relation/attribute balance"
                ),
                "verifier": asdict(candidate_verifier),
            }
        expandable_core_ids = [
            child.path_id
            for child in cores
            if _local_branch_kind(child.branch) == "relation"
        ]
        if len(expandable_core_ids) < max(0, min_expandable_core_paths):
            return {
                "node_path": node_path,
                "depth": depth,
                "expanded": True,
                "accepted": not self.config.require_local_verification,
                "hard_accepted": False,
                "failure_kind": "insufficient_expandable_local_core_paths",
                "required": max(0, min_expandable_core_paths),
                "actual": len(expandable_core_ids),
                "expandable_core_path_ids": expandable_core_ids,
                "non_expandable_core_path_ids": [
                    child.path_id
                    for child in cores
                    if child.path_id not in expandable_core_ids
                ],
                "reason": "not enough core children can continue the Local chain",
                "verifier": asdict(candidate_verifier),
            }
        unsupported_children = [
            child.path_id
            for child in path.local_constraints
            if not _local_evidence_mentions_target(
                child,
                local_target,
                path.local_target_canonical_name,
            )
        ]
        if unsupported_children:
            return {
                "node_path": node_path,
                "depth": depth,
                "expanded": True,
                "accepted": not self.config.require_local_verification,
                "hard_accepted": False,
                "failure_kind": "child_target_membership_unsupported",
                "offending_child_path_ids": unsupported_children,
                "reason": (
                    "child evidence does not name the local target for: "
                    + ", ".join(unsupported_children)
                ),
                "verifier": asdict(candidate_verifier),
            }
        return {
            "node_path": node_path,
            "depth": depth,
            "expanded": True,
            "accepted": candidate_verifier.accepted or not self.config.require_local_verification,
            "hard_accepted": candidate_verifier.accepted,
            "local_target": asdict(local_target),
            "local_target_canonical_name": path.local_target_canonical_name,
            "current_node_clue": source_clue,
            "local_target_text_reference": textual_reference,
            "verifier": asdict(candidate_verifier),
            "reason": candidate_verifier.reason,
        }

    def _path_requirements(self) -> Dict[str, Any]:
        return {
            "min_core_paths": self.config.min_core_paths,
            "min_distractor_paths": self.config.min_distractor_paths,
            "max_distractor_paths": self.config.max_distractor_paths,
            "min_single_candidates": self.config.min_single_candidates,
            "max_single_candidates": self.config.max_single_candidates,
            "min_pairwise_core_intersection": self.config.min_pairwise_core_intersection,
            "min_distractor_core_overlap": self.config.min_distractor_core_overlap,
            "min_distractors_per_core": self.config.min_distractors_per_core,
            "local_verifier": {
                "min_core_paths": self.config.local_min_core_paths,
                "min_distractor_paths": self.config.local_min_distractor_paths,
                "candidate_matrix_required": False,
                "min_relation_core_paths": self.config.local_min_relation_core_paths,
                "min_attribute_core_paths": self.config.local_min_attribute_core_paths,
                "prefers_joint_semantic_uniqueness": True,
                "requires_joint_semantic_uniqueness": False,
                "requires_local_target_truth_and_coherence": True,
                "requires_local_target_in_current_clue": True,
                "requires_local_target_in_distractors": True,
                "forbid_seed_target_in_child_clues": True,
                "forbid_final_answer_in_child_clues": True,
                "forbid_ancestor_local_targets_in_child_clues": True,
            },
            "local_max_depth": self.config.local_max_depth,
            "local_core_max_depth": self.config.local_core_max_depth,
            "local_distractor_max_depth": self.config.local_distractor_max_depth,
            "local_expand_min_paths": self.config.local_expand_min_paths,
            "local_expand_max_paths": self.config.local_expand_max_paths,
            "local_expand_core_min_paths": self.config.local_expand_core_min_paths,
            "local_expand_core_max_paths": self.config.local_expand_core_max_paths,
            "local_expand_distractor_min_paths": self.config.local_expand_distractor_min_paths,
            "local_expand_distractor_max_paths": self.config.local_expand_distractor_max_paths,
            "local_expand_distractor_prob": self.config.local_expand_distractor_prob,
            "local_expand_roles": self.config.local_expand_roles,
            "local_max_deep_paths": self.config.local_max_deep_paths,
            "local_deep_path_roles": self.config.local_deep_path_roles,
            "require_local_verification": self.config.require_local_verification,
            "require_local_expansion": self.config.require_local_expansion,
        }

    def _local_max_depth_for(self, path: ConstraintPath) -> int:
        if path.role == "distractor":
            return max(0, self.config.local_distractor_max_depth)
        return max(0, self.config.local_core_max_depth)

    def _deep_path_limit_for_root(self, state: LocalExpansionState, root_id: str) -> int:
        """Return the deep-path budget for a given root, based on its role."""
        role = state.root_roles.get(root_id, "core")
        if role == "distractor":
            return max(0, self.config.local_max_deep_paths_distractor)
        return max(0, self.config.local_max_deep_paths)

    def _local_deep_limit_reached(self, state: LocalExpansionState, *, root_id: str = "") -> bool:
        if not root_id:
            limit = self.config.local_max_deep_paths
            if limit <= 0:
                return False
            with state.lock:
                return state.deep_paths >= limit
        limit = self._deep_path_limit_for_root(state, root_id)
        if limit <= 0:
            return False
        with state.lock:
            return state.deep_paths_per_root.get(root_id, 0) >= limit

    def _local_deep_remaining(self, state: LocalExpansionState, *, root_id: str = "") -> int | None:
        if not root_id:
            limit = self.config.local_max_deep_paths
            if limit <= 0:
                return None
            with state.lock:
                return max(0, limit - state.deep_paths)
        limit = self._deep_path_limit_for_root(state, root_id)
        if limit <= 0:
            return None
        with state.lock:
            return max(0, limit - state.deep_paths_per_root.get(root_id, 0))

    def _mark_deep_path(self, state: LocalExpansionState, path: ConstraintPath, *, root_id: str = "") -> int:
        if path.role not in self.config.local_deep_path_roles:
            return state.deep_paths
        with state.lock:
            limit = self._deep_path_limit_for_root(state, root_id) if root_id else self.config.local_max_deep_paths
            if limit > 0 and state.deep_paths_per_root.get(root_id, 0) >= limit:
                return state.deep_paths
            state.deep_paths += 1
            if root_id:
                state.deep_paths_per_root[root_id] = state.deep_paths_per_root.get(root_id, 0) + 1
            return state.deep_paths

    def _try_claim_deep_path(
        self,
        state: LocalExpansionState,
        path: ConstraintPath,
        *,
        root_id: str = "",
    ) -> bool:
        """Atomically reserve the single terminal deep path for a root."""
        if path.role not in self.config.local_deep_path_roles:
            return True
        with state.lock:
            limit = (
                self._deep_path_limit_for_root(state, root_id)
                if root_id
                else self.config.local_max_deep_paths
            )
            current = state.deep_paths_per_root.get(root_id, 0) if root_id else state.deep_paths
            if limit > 0 and current >= limit:
                return False
            state.deep_paths += 1
            if root_id:
                state.deep_paths_per_root[root_id] = current + 1
            return True

    def _select_local_expand_indexes(
        self,
        artifact: Artifact,
        paths: Sequence[ConstraintPath],
        *,
        depth: int,
        parent_node_path: List[str],
        state: LocalExpansionState,
    ) -> List[int]:
        root_id = parent_node_path[0] if parent_node_path else ""
        if self._local_deep_limit_reached(state, root_id=root_id):
            return []
        if depth == 0:
            if getattr(self.config, "local_expand_all_root_cores", False):
                # v5 expands every Root core at depth zero. Question may later
                # select an overcomplete subset, but it must have a deep chain
                # available for every selected root except one root-only fallback.
                required_core_ids = {
                    path.path_id for path in paths if path.role == "core"
                }
            else:
                required_core_ids = set(
                    _maximum_minimal_unique_root_bundle(artifact.verifier)
                )
                if not required_core_ids:
                    required_core_ids = {
                        path.path_id for path in paths if path.role == "core"
                    }
            selected_core = [
                idx
                for idx, path in enumerate(paths)
                if path.role == "core" and path.path_id in required_core_ids
            ]
            root_distractors = [
                idx for idx, path in enumerate(paths) if path.role == "distractor"
            ]
            eligible_distractors = [
                idx
                for idx in root_distractors
                if _stable_unit_float(
                    artifact.target.entity_id,
                    paths[idx].path_id,
                    "root_distractor",
                ) <= self.config.local_expand_distractor_prob
            ]
            required_distractors = min(
                len(root_distractors),
                max(0, self.config.local_expand_distractor_min_paths),
            )
            if len(eligible_distractors) < required_distractors:
                eligible_set = set(eligible_distractors)
                eligible_distractors.extend(
                    idx for idx in root_distractors if idx not in eligible_set
                )
            selected_distractors = self._stable_select_indexes(
                artifact,
                paths,
                eligible_distractors,
                depth=depth,
                parent_node_path=parent_node_path,
                min_paths=required_distractors,
                max_paths=self.config.local_expand_distractor_max_paths,
                salt="select_root_distractor",
            )
            return sorted(dict.fromkeys([*selected_core, *selected_distractors]))

        # Keep several core children as candidate branches at each layer. Once
        # any branch of this root reaches the deep-path budget, the cooperative
        # checks in _expand_local_path stop the other branches from going deeper.
        core_eligible: List[int] = []
        expand_roles = set(self.config.local_expand_roles)
        for idx, path in enumerate(paths):
            if path.role == "core" and path.role in expand_roles:
                core_eligible.append(idx)
        selected_core = self._stable_select_indexes(
            artifact,
            paths,
            core_eligible,
            depth=depth,
            parent_node_path=parent_node_path,
            min_paths=min(
                self.config.local_expand_core_min_paths,
                len(core_eligible),
            ),
            max_paths=self.config.local_expand_core_max_paths,
            salt="select_core",
        )
        return selected_core

    def _stable_select_indexes(
        self,
        artifact: Artifact,
        paths: Sequence[ConstraintPath],
        indexes: Sequence[int],
        *,
        depth: int,
        parent_node_path: List[str],
        min_paths: int,
        max_paths: int,
        salt: str,
    ) -> List[int]:
        unique_indexes = list(dict.fromkeys(indexes))
        if not unique_indexes:
            return []
        if max_paths <= 0:
            count = len(unique_indexes)
        else:
            upper = min(len(unique_indexes), max(1, max_paths))
            lower = min(max(0, min_paths), upper)
            if lower == upper:
                count = upper
            else:
                span = upper - lower + 1
                offset = _stable_int(
                    artifact.target.entity_id,
                    *parent_node_path,
                    str(depth),
                    salt,
                    "count",
                ) % span
                count = lower + offset
        if count <= 0:
            return []
        return sorted(
            unique_indexes,
            key=lambda idx: _stable_sort_key(
                artifact.target.entity_id,
                *parent_node_path,
                paths[idx].path_id,
                str(depth),
                salt,
            ),
        )[:count]

    def _checkpoint_artifact(self, artifact: Artifact) -> None:
        path = str((artifact.run_context or {}).get("artifact_path") or "").strip()
        if path:
            save_artifact(artifact, path)

    def _record_question_version(
        self,
        artifact: Artifact,
        *,
        source: str,
        revision: int,
        response: Dict[str, Any] | None = None,
    ) -> int:
        question = str(artifact.question or "").strip()
        if not question:
            return -1
        if artifact.question_history and artifact.question_history[-1].get("question") == question:
            version = int(artifact.question_history[-1].get("version") or 0)
            artifact.question_history[-1].update(
                {
                    "source": source,
                    "revision": revision,
                    "question_state": deepcopy(artifact.question_state),
                }
            )
        else:
            version = len(artifact.question_history)
            artifact.question_history.append(
                {
                    "version": version,
                    "source": source,
                    "revision": revision,
                    "question": question,
                    "question_state": deepcopy(artifact.question_state),
                    "generation": deepcopy(
                        (response or {}).get("_question_generation")
                    ),
                    "repair_model": deepcopy(
                        (response or {}).get("_question_repair_model")
                    ),
                    "validation": deepcopy((response or {}).get("validation")),
                    "uniqueness_key": "",
                    "uniqueness_report": {},
                }
            )
        seed_dir = str((artifact.run_context or {}).get("seed_dir") or "").strip()
        if seed_dir:
            question_dir = Path(seed_dir) / "questions"
            question_dir.mkdir(parents=True, exist_ok=True)
            (question_dir / f"question_{version:03d}.json").write_text(
                json.dumps(
                    artifact.question_history[version],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        self._checkpoint_artifact(artifact)
        return version

    def _current_question_version(self, artifact: Artifact) -> int:
        for item in reversed(artifact.question_history):
            if item.get("question") == artifact.question:
                return int(item.get("version") or 0)
        return self._record_question_version(
            artifact,
            source="implicit_before_solver",
            revision=len(artifact.question_history),
        )

    def _update_question_uniqueness(
        self, artifact: Artifact, report: Dict[str, Any]
    ) -> None:
        version = self._current_question_version(artifact)
        if version < 0 or version >= len(artifact.question_history):
            return
        artifact.question_history[version]["uniqueness_key"] = str(
            report.get("key") or ""
        )
        artifact.question_history[version]["uniqueness_report"] = deepcopy(report)
        seed_dir = str((artifact.run_context or {}).get("seed_dir") or "").strip()
        if seed_dir:
            question_dir = Path(seed_dir) / "questions"
            question_dir.mkdir(parents=True, exist_ok=True)
            (question_dir / f"question_{version:03d}.json").write_text(
                json.dumps(
                    artifact.question_history[version],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        self._checkpoint_artifact(artifact)

    def _solver_trace_context(
        self,
        artifact: Artifact,
        *,
        question_version: int,
        rollout_id: int,
    ) -> Dict[str, Any]:
        context = dict(artifact.run_context or {})
        context.update(
            {
                "seed_id": artifact.target.entity_id,
                "question_version": question_version,
                "rollout_id": rollout_id,
            }
        )
        return context

    def _write_solver_rollout_report(
        self,
        artifact: Artifact,
        *,
        question_version: int,
        rollout_id: int,
        report: Dict[str, Any],
    ) -> None:
        seed_dir = str((artifact.run_context or {}).get("seed_dir") or "").strip()
        if not seed_dir:
            return
        rollout_dir = (
            Path(seed_dir)
            / "solver"
            / f"question_{question_version:03d}"
            / f"rollout_{rollout_id:03d}"
        )
        rollout_dir.mkdir(parents=True, exist_ok=True)
        (rollout_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_solver_question_snapshot(
        self, artifact: Artifact, *, question_version: int
    ) -> None:
        seed_dir = str((artifact.run_context or {}).get("seed_dir") or "").strip()
        if not seed_dir:
            return
        question_dir = (
            Path(seed_dir) / "solver" / f"question_{question_version:03d}"
        )
        question_dir.mkdir(parents=True, exist_ok=True)
        history = next(
            (
                item
                for item in artifact.question_history
                if int(item.get("version") or 0) == question_version
            ),
            {},
        )
        (question_dir / "question.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_solver_attempt_snapshot(
        self, artifact: Artifact, attempt: Dict[str, Any]
    ) -> None:
        seed_dir = str((artifact.run_context or {}).get("seed_dir") or "").strip()
        if not seed_dir:
            return
        version = int(attempt.get("question_version") or 0)
        question_dir = Path(seed_dir) / "solver" / f"question_{version:03d}"
        question_dir.mkdir(parents=True, exist_ok=True)
        (question_dir / "attempt.json").write_text(
            json.dumps(attempt, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _log(self, message: str) -> None:
        if self.config.verbose_progress:
            print(message, flush=True)

    def _legacy_local_fuzzify(self, artifact: Artifact) -> List[ConstraintPath]:
        response = self.runner.run_json(
            self.config.agents["local_constraint"],
            system_prompt=LOCAL_CONSTRAINT_PROMPT,
            user_payload={
                "target": asdict(artifact.target),
                "verifier": asdict(artifact.verifier) if artifact.verifier else None,
                "constraints": [asdict(item) for item in artifact.constraints],
            },
            rate_limit_scope=artifact.target.entity_id,
            response_validator=_validate_local_constraint_response,
        )
        updated = [ConstraintPath.from_dict(item) for item in response.get("constraints", [])]
        return updated or artifact.constraints

    def _question_fallback_agent(self) -> Any | None:
        fallback = self.config.agents.get("question_fallback")
        primary = self.config.agents.get("question")
        if fallback is None or not fallback.configured:
            return None
        if primary is not None and fallback.model == primary.model:
            return None
        return fallback

    def _run_question_draft(
        self,
        artifact: Artifact,
        user_payload: Dict[str, Any],
        *,
        force_fallback: bool = False,
        fallback_reason: str = "",
    ) -> Dict[str, Any]:
        primary = self.config.agents["question"]
        fallback = self._question_fallback_agent()
        agent = fallback if force_fallback and fallback is not None else primary
        used_fallback = agent is fallback
        try:
            response = self.runner.run_json(
                agent,
                system_prompt=QUESTION_PROMPT,
                user_payload=user_payload,
                rate_limit_scope=artifact.target.entity_id,
            )
        except Exception as exc:
            if used_fallback or fallback is None:
                raise
            agent = fallback
            used_fallback = True
            fallback_reason = f"primary question call failed: {type(exc).__name__}: {exc}"
            response = self.runner.run_json(
                agent,
                system_prompt=QUESTION_PROMPT,
                user_payload=user_payload,
                rate_limit_scope=artifact.target.entity_id,
            )
        response["_question_generation"] = {
            "model": str(getattr(agent, "model", "")),
            "fallback_used": used_fallback,
            "fallback_reason": fallback_reason if used_fallback else "",
        }
        return response

    def _write_question(
        self,
        artifact: Artifact,
        *,
        revision: int = 0,
        previous_questions: Sequence[str] = (),
        failure_report: Dict[str, Any] | None = None,
        force_fallback: bool = False,
    ) -> Dict[str, Any]:
        core_ids = set(artifact.verifier.core_path_ids if artifact.verifier else [])
        core = [item for item in artifact.constraints if item.path_id in core_ids]
        distractors = [item for item in artifact.constraints if item.role == "distractor"]
        semantic_shortcut_ids = _local_quality_shortcut_ids(artifact)
        trajectory_shortcut_ids = _question_trajectory_shortcut_ids(
            artifact,
            failure_report,
        )
        minimal_bundle_ids = _maximum_minimal_unique_root_bundle(artifact.verifier)
        if not minimal_bundle_ids:
            return {
                "question": "",
                "answer": artifact.target.answer,
                "failure_reason": "no deletion-minimal unique root bundle is available",
            }
        all_core_payload = [
            self._question_constraint_view(
                item,
                require_leaf=False,
                semantic_shortcut_ids=semantic_shortcut_ids,
                avoid_path_ids=trajectory_shortcut_ids,
            )
            for item in core
        ]
        core_views_by_id = {item["path_id"]: item for item in all_core_payload}
        # Prefer every Root that has a Local continuation, then permit only one
        # root-only fallback. This allows an overcomplete Root set while keeping
        # the public question connected to deep chains wherever the tree permits.
        expanded_core_ids = [
            item["path_id"]
            for item in all_core_payload
            if _question_view_has_expanded_chain(item)
        ]
        unexpanded_core_ids = [
            item["path_id"]
            for item in all_core_payload
            if not _question_view_has_expanded_chain(item)
        ]
        if getattr(self.config, "question_allow_overcomplete_roots", False):
            planned_core_ids = list(dict.fromkeys(expanded_core_ids))
            shallow_required = [
                path_id
                for path_id in minimal_bundle_ids
                if path_id in unexpanded_core_ids
            ]
            if shallow_required:
                planned_core_ids.append(shallow_required[0])
            elif unexpanded_core_ids and not planned_core_ids:
                planned_core_ids.append(unexpanded_core_ids[0])
        else:
            planned_core_ids = list(minimal_bundle_ids)
        if not planned_core_ids:
            planned_core_ids = list(minimal_bundle_ids)
        deep_leaf_core_ids = [
            path_id
            for path_id in planned_core_ids
            if path_id in core_views_by_id
            and core_views_by_id[path_id].get("selected_path_chain")
            and core_views_by_id[path_id].get("question_clue_depth", 0)
            >= self.config.question_min_leaf_depth
        ]
        needed_deep_roots = max(
            0,
            self.config.question_min_leaf_core_paths - len(deep_leaf_core_ids),
        )
        missing_leaf_core_ids = [
            path_id
            for path_id in planned_core_ids
            if path_id not in deep_leaf_core_ids
        ][:needed_deep_roots]
        if needed_deep_roots:
            return {
                "question": "",
                "answer": artifact.target.answer,
                "failure_reason": (
                    "required unique root bundle needs a core leaf at "
                    f"depth>={self.config.question_min_leaf_depth}"
                ),
                "required_core_root_ids": planned_core_ids,
                "leaf_core_path_ids": deep_leaf_core_ids,
                "missing_leaf_core_path_ids": missing_leaf_core_ids,
                "core_path_depths": {
                    item["path_id"]: item.get("max_local_depth", 0) for item in all_core_payload
                },
            }
        selected_core_views = [core_views_by_id[path_id] for path_id in planned_core_ids]
        all_distractor_payload = [
            item
            for item in (
                self._question_constraint_view(
                    path,
                    require_leaf=False,
                    semantic_shortcut_ids=semantic_shortcut_ids,
                    avoid_path_ids=trajectory_shortcut_ids,
                )
                for path in distractors
            )
            if item.get("question_clue")
        ]
        if len(all_distractor_payload) < self.config.question_min_distractor_paths:
            return {
                "question": "",
                "answer": artifact.target.answer,
                "failure_reason": (
                    "need at least "
                    f"{self.config.question_min_distractor_paths} distractor chains"
                ),
                "available_distractor_path_ids": [
                    item["path_id"] for item in all_distractor_payload
                ],
            }
        initial_distractor_count = min(
            len(all_distractor_payload),
            max(self.config.question_min_distractor_paths, 1),
        )
        initial_distractor_views = sorted(
            all_distractor_payload,
            key=lambda item: (
                int(item.get("question_clue_depth") or 0),
                str(item.get("path_id") or ""),
            ),
            reverse=True,
        )[:initial_distractor_count]
        question_core_payload = [
            _question_model_constraint_view(item) for item in selected_core_views
        ]
        question_distractor_payload = [
            _question_model_constraint_view(item) for item in initial_distractor_views
        ]
        all_core_model_payload = [
            _question_model_constraint_view(item)
            for item in all_core_payload
            if item.get("question_clue")
        ]
        all_distractor_model_payload = [
            _question_model_constraint_view(item) for item in all_distractor_payload
        ]
        initial_question_payload = {
            "target": self._question_target_view(artifact.target),
            "selected_root_chains": question_core_payload,
            "optional_distractor_chains": question_distractor_payload,
            "required_core_root_ids": planned_core_ids,
            "max_words": self.config.question_max_words,
            "min_root_core_children": getattr(
                self.config, "question_min_root_core_children", 1
            ),
            "min_root_non_shortcut_children": getattr(
                self.config, "question_min_root_non_shortcut_children", 0
            ),
            "effort_target": _solver_effort_requirements(self.config),
        }
        repairing_existing = bool(failure_report and artifact.question.strip())
        if repairing_existing:
            response = {
                **artifact.question_state,
                "question": artifact.question,
                "answer": artifact.target.answer,
            }
            response = self._repair_question_response(
                response,
                artifact=artifact,
                available_core_payload=all_core_model_payload,
                available_distractor_payload=all_distractor_model_payload,
                revision=revision,
                previous_questions=previous_questions,
                failure_report=failure_report,
                required_core_path_ids=planned_core_ids,
                agent_override=(
                    self._question_fallback_agent() if force_fallback else None
                ),
                agent_override_reason=(
                    "GPT-5.5 did not resolve the previous semantic feedback"
                    if force_fallback
                    else ""
                ),
            )
        else:
            response = self._run_question_draft(
                artifact,
                initial_question_payload,
                force_fallback=force_fallback,
                fallback_reason=(
                    "GPT-5.5 did not resolve the previous semantic feedback"
                    if force_fallback
                    else ""
                ),
            )
        validation = self._validate_question_response(
            response,
            artifact.target,
            all_core_payload,
            all_distractor_payload,
            root_verifier=artifact.verifier,
            required_core_path_ids=planned_core_ids,
        )
        if (
            not validation["accepted"]
            and not repairing_existing
            and self.config.question_repair_enabled
            and str(response.get("question") or "").strip()
        ):
            response = self._repair_question_response(
                response,
                artifact=artifact,
                available_core_payload=all_core_model_payload,
                available_distractor_payload=all_distractor_model_payload,
                revision=revision,
                previous_questions=previous_questions,
                failure_report={
                    "stage": "question_validation",
                    "status": "invalid",
                    "reason": validation.get("reason", ""),
                    "validation": validation,
                },
                required_core_path_ids=planned_core_ids,
            )
            validation = self._validate_question_response(
                response,
                artifact.target,
                all_core_payload,
                all_distractor_payload,
                root_verifier=artifact.verifier,
                required_core_path_ids=planned_core_ids,
            )
        generation_meta = response.get("_question_generation")
        repair_model_meta = response.get("_question_repair_model")
        fallback_already_used = bool(
            (
                isinstance(generation_meta, dict)
                and generation_meta.get("fallback_used")
            )
            or (
                isinstance(repair_model_meta, dict)
                and repair_model_meta.get("fallback_used")
            )
        )
        if (
            not validation["accepted"]
            and self._question_fallback_agent() is not None
            and not fallback_already_used
        ):
            if repairing_existing:
                response = self._repair_question_response(
                    response,
                    artifact=artifact,
                    available_core_payload=all_core_model_payload,
                    available_distractor_payload=all_distractor_model_payload,
                    revision=revision,
                    previous_questions=previous_questions,
                    failure_report={
                        "stage": "question_validation",
                        "status": "invalid_after_primary_repair",
                        "reason": validation.get("reason", ""),
                    },
                    required_core_path_ids=planned_core_ids,
                    agent_override=self._question_fallback_agent(),
                )
            else:
                response = self._run_question_draft(
                    artifact,
                    initial_question_payload,
                    force_fallback=True,
                    fallback_reason=str(validation.get("reason") or "primary output invalid"),
                )
            validation = self._validate_question_response(
                response,
                artifact.target,
                all_core_payload,
                all_distractor_payload,
                root_verifier=artifact.verifier,
                required_core_path_ids=planned_core_ids,
            )
        fallback_agent = self._question_fallback_agent()
        response_generation = response.get("_question_generation")
        response_repair_model = response.get("_question_repair_model")
        response_used_fallback = bool(
            isinstance(response_generation, dict)
            and response_generation.get("fallback_used")
        ) or bool(
            isinstance(response_repair_model, dict)
            and response_repair_model.get("fallback_used")
        )
        if (
            not validation["accepted"]
            and response_used_fallback
            and fallback_agent is not None
            and self.config.question_repair_enabled
            and str(response.get("question") or "").strip()
        ):
            response = self._repair_question_response(
                response,
                artifact=artifact,
                available_core_payload=all_core_model_payload,
                available_distractor_payload=all_distractor_model_payload,
                revision=revision,
                previous_questions=previous_questions,
                failure_report={
                    "stage": "question_validation",
                    "status": "invalid_after_fallback",
                    "reason": validation.get("reason", ""),
                    "validation": validation,
                },
                required_core_path_ids=planned_core_ids,
                agent_override=fallback_agent,
                agent_override_reason="bounded fallback self-repair",
            )
            validation = self._validate_question_response(
                response,
                artifact.target,
                all_core_payload,
                all_distractor_payload,
                root_verifier=artifact.verifier,
                required_core_path_ids=planned_core_ids,
            )
        for semantic_retry in range(
            int(getattr(self.config, "question_validation_repair_attempts", 0))
        ):
            if validation["accepted"] or not str(response.get("question") or "").strip():
                break
            previous_candidate = str(response.get("question") or "").strip()
            response = self._repair_question_response(
                response,
                artifact=artifact,
                available_core_payload=all_core_model_payload,
                available_distractor_payload=all_distractor_model_payload,
                revision=revision,
                previous_questions=[*previous_questions, previous_candidate],
                failure_report={
                    "stage": "question_validation",
                    "status": "semantic_retry",
                    "semantic_retry": semantic_retry + 1,
                    "reason": validation.get("reason", ""),
                    "validation": validation,
                },
                required_core_path_ids=planned_core_ids,
                agent_override=fallback_agent,
                agent_override_reason=(
                    f"verifier-guided semantic repair {semantic_retry + 1}"
                ),
            )
            candidate = str(response.get("question") or "").strip()
            if not candidate or candidate == previous_candidate:
                break
            validation = self._validate_question_response(
                response,
                artifact.target,
                all_core_payload,
                all_distractor_payload,
                root_verifier=artifact.verifier,
                required_core_path_ids=planned_core_ids,
            )
        if not validation["accepted"]:
            artifact.question_attempts.append(
                {
                    "attempt": len(artifact.question_attempts),
                    "revision": revision,
                    "question": str(response.get("question") or ""),
                    "response": deepcopy(response),
                    "validation": deepcopy(validation),
                    "accepted": False,
                }
            )
            return {
                **response,
                "question": "",
                "failure_reason": "question_validation_failed",
                "validation": validation,
            }
        response["validation"] = validation
        response["required_core_root_ids"] = planned_core_ids
        artifact.question_attempts.append(
            {
                "attempt": len(artifact.question_attempts),
                "revision": revision,
                "question": str(response.get("question") or ""),
                "response": deepcopy(response),
                "validation": deepcopy(validation),
                "accepted": True,
            }
        )
        artifact.question_state = {
            key: response.get(key)
            for key in (
                "question",
                "answer",
                "used_core_path_ids",
                "used_distractor_path_ids",
                "chain_node_usage",
                "fuzzified_child_path_ids",
                "note",
                "required_core_root_ids",
                "generalized_required_root_ids",
            )
        }
        repair = response.get("repair")
        if isinstance(repair, dict) and repair.get("accepted") and repair.get("original_question"):
            artifact.question_repair_history.append(
                {
                    "revision": revision,
                    "original_question": str(repair.get("original_question", "")),
                    "repaired_question": str(response.get("question", "")),
                    "repair_note": str(repair.get("repair_note", "")),
                    "chain_node_usage": _as_list_of_dicts(response.get("chain_node_usage")),
                    "repair_actions": _as_list_of_dicts(response.get("repair_actions")),
                    "used_constraint_snapshots": _as_list_of_dicts(
                        repair.get("used_constraint_snapshots")
                    ),
                }
            )
        return response

    def _repair_question_response(
        self,
        response: Dict[str, Any],
        *,
        artifact: Artifact,
        available_core_payload: Sequence[Dict[str, Any]],
        available_distractor_payload: Sequence[Dict[str, Any]],
        revision: int,
        previous_questions: Sequence[str],
        failure_report: Dict[str, Any] | None,
        required_core_path_ids: Sequence[str] = (),
        agent_override: Any | None = None,
        agent_override_reason: str = "",
    ) -> Dict[str, Any]:
        original_question = str(response.get("question", ""))
        if not original_question.strip():
            return response
        available_root_constraints = [
            *[dict(item) for item in available_core_payload],
            *[dict(item) for item in available_distractor_payload],
        ]
        root_constraints_by_id = {
            str(item.get("path_id")): item
            for item in available_root_constraints
            if str(item.get("path_id") or "")
        }
        draft_used_core_ids = (
            response.get("used_core_path_ids", [])
            if isinstance(response.get("used_core_path_ids"), list)
            else []
        )
        draft_used_distractor_ids = (
            response.get("used_distractor_path_ids", [])
            if isinstance(response.get("used_distractor_path_ids"), list)
            else []
        )
        selected_root_ids = list(
            dict.fromkeys(
                str(item)
                for item in [
                    *draft_used_core_ids,
                    *draft_used_distractor_ids,
                ]
                if str(item) in root_constraints_by_id
            )
        )
        selected_root_constraints = [
            root_constraints_by_id[path_id] for path_id in selected_root_ids
        ]
        unexpanded_selected_root_ids = [
            str(item.get("path_id") or "")
            for item in selected_root_constraints
            if (
                not item.get("selected_path_chain")
                or (
                    len(item.get("selected_path_chain") or []) == 1
                    and not (item.get("selected_path_chain") or [{}])[0].get(
                        "child_clues"
                    )
                )
            )
            and str(item.get("path_id") or "")
        ]
        unselected_root_constraints = [
            item
            for item in available_root_constraints
            if str(item.get("path_id") or "") not in set(selected_root_ids)
        ]
        current_selection = {
            "used_core_path_ids": [
                str(item) for item in draft_used_core_ids
            ],
            "used_distractor_path_ids": [
                str(item) for item in draft_used_distractor_ids
            ],
            "chain_node_usage": _as_list_of_dicts(
                response.get("chain_node_usage")
            ),
        }
        participating_chain_nodes = _participating_chain_node_details(
            current_selection["chain_node_usage"],
            selected_root_constraints,
        )
        repair_mode = _question_repair_mode(failure_report)
        used_children = {
            str(child_id)
            for item in current_selection["chain_node_usage"]
            for child_id in item.get("used_child_path_ids", [])
            if str(child_id)
        }
        unused_selected_root_children: List[Dict[str, Any]] = []
        for root in selected_root_constraints:
            root_id = str(root.get("path_id") or "")
            for node in root.get("selected_path_chain", []):
                if not isinstance(node, dict) or int(node.get("depth") or 0) != 0:
                    continue
                for child in node.get("child_clues", []):
                    if (
                        not isinstance(child, dict)
                        or str(child.get("role") or "") != "core"
                        or str(child.get("path_id") or "") in used_children
                    ):
                        continue
                    unused_selected_root_children.append(
                        {
                            "root_path_id": root_id,
                            "node_path_id": str(node.get("path_id") or ""),
                            "node_depth": 0,
                            "child": dict(child),
                        }
                    )
        unused_selected_root_children.sort(
            key=lambda item: (
                bool((item.get("child") or {}).get("semantic_shortcut")),
                len((item.get("child") or {}).get("risk_flags") or []),
                str((item.get("child") or {}).get("path_id") or ""),
            )
        )
        trajectory_feedback = failure_report.get("trajectory_feedback")
        trajectory_feedback = (
            trajectory_feedback if isinstance(trajectory_feedback, dict) else {}
        )
        adjudication = trajectory_feedback.get("alternative_adjudication")
        if isinstance(adjudication, dict):
            adjudicated_items = (
                adjudication.get("verified_alternatives", [])
                if adjudication.get("verdict") == "verified_ambiguity"
                else []
            )
        else:
            # Kept for direct/unit callers that provide the already validated
            # adjudication result without its wrapper.
            adjudicated_items = trajectory_feedback.get("verified_alternatives", [])
        verified_alternatives = [
            item
            for item in adjudicated_items
            if isinstance(item, dict) and item.get("matches_all_clues") is True
        ]
        ambiguity_points = [
            item
            for item in trajectory_feedback.get("ambiguity_points", [])
            if isinstance(item, dict)
        ]
        # The repair model normally sees only the current public question. A
        # previous version is useful only after independent adjudication has
        # verified a different answer against every public clause and cited
        # sources.
        prior_question = ""
        if verified_alternatives and len(previous_questions) >= 2:
            prior_question = str(previous_questions[-2]).strip()
        repair_payload = {
            "target": {
                "entity_id": artifact.target.entity_id,
                "name": artifact.target.name,
                "entity_type": artifact.target.entity_type,
                "answer_field": artifact.target.answer_field,
                "answer": artifact.target.answer,
            },
            "repair_mode": repair_mode,
            "current_question": original_question,
            "current_question_note": str(response.get("note", "")),
            "current_selection": current_selection,
            "participating_chain_nodes": participating_chain_nodes,
            "selected_root_constraints": selected_root_constraints,
            "unselected_root_constraints": unselected_root_constraints,
            "unexpanded_selected_root_ids": unexpanded_selected_root_ids,
            "question_requirements": {
                "required_core_root_ids": list(required_core_path_ids),
                "min_distractor_paths": self.config.question_min_distractor_paths,
                "max_words": self.config.question_max_words,
                "min_root_core_children": getattr(
                    self.config, "question_min_root_core_children", 1
                ),
                "min_root_non_shortcut_children": getattr(
                    self.config, "question_min_root_non_shortcut_children", 0
                ),
                "effort_target": _solver_effort_requirements(self.config),
            },
            "revision": revision,
            "failure_report": failure_report,
            "repair_priorities": {
                "unused_selected_root_core_children": unused_selected_root_children,
                "verified_alternatives": verified_alternatives,
                "ambiguity_points": ambiguity_points,
                "must_exclude_verified_alternatives": bool(verified_alternatives),
                "must_try_selected_root_child_before_new_root": bool(
                    repair_mode == "uniqueness_supplement"
                    and unused_selected_root_children
                    and not verified_alternatives
                ),
            },
        }
        if prior_question:
            repair_payload["prior_question"] = prior_question
            repair_payload["prior_question_note"] = (
                "This is the immediately preceding, non-ambiguous baseline question "
                "for comparison. The current_question is the version just diagnosed "
                "as ambiguous. Do not copy unsupported facts from prior_question."
            )
        primary_agent = self.config.agents["question_repair"]
        fallback_agent = self._question_fallback_agent()
        agent = agent_override or primary_agent
        fallback_used = agent_override is not None
        fallback_reason = (
            agent_override_reason
            or "primary output failed program validation"
            if fallback_used
            else ""
        )
        try:
            repaired = self.runner.run_json(
                agent,
                system_prompt=QUESTION_REPAIR_PROMPT,
                user_payload=repair_payload,
                rate_limit_scope=artifact.target.entity_id,
            )
        except Exception as exc:
            if fallback_used or fallback_agent is None:
                return {
                    **response,
                    "repair": {
                        "attempted": True,
                        "accepted": False,
                        "reason": f"question repair failed: {exc}",
                    },
                }
            agent = fallback_agent
            fallback_used = True
            fallback_reason = f"primary repair call failed: {type(exc).__name__}: {exc}"
            repaired = self.runner.run_json(
                agent,
                system_prompt=QUESTION_REPAIR_PROMPT,
                user_payload=repair_payload,
                rate_limit_scope=artifact.target.entity_id,
            )
        repaired_question = str(repaired.get("question", "")).strip()
        if not repaired_question and not fallback_used and fallback_agent is not None:
            agent = fallback_agent
            fallback_used = True
            fallback_reason = "primary repair returned an empty question"
            repaired = self.runner.run_json(
                agent,
                system_prompt=QUESTION_REPAIR_PROMPT,
                user_payload=repair_payload,
                rate_limit_scope=artifact.target.entity_id,
            )
            repaired_question = str(repaired.get("question", "")).strip()
        if not repaired_question:
            return {
                **response,
                "repair": {
                    "attempted": True,
                    "accepted": False,
                    "reason": "question repair returned empty question",
                    "raw_response": repaired,
                },
            }
        repaired.setdefault("answer", artifact.target.answer)
        existing_generalized_root_ids = {
            str(path_id)
            for path_id in response.get("generalized_required_root_ids", [])
            if str(path_id) in set(required_core_path_ids)
        }
        trajectory_shortcut_root_ids = {
            str(path_id)
            for path_id in trajectory_feedback.get("shortcut_root_path_ids", [])
            if str(path_id) in set(required_core_path_ids)
        }
        repaired["generalized_required_root_ids"] = sorted(
            existing_generalized_root_ids
            | (
                trajectory_shortcut_root_ids
                if repair_mode == "shortcut_prune"
                else set()
            )
        )
        repaired["_question_repair_model"] = {
            "model": str(getattr(agent, "model", "")),
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason if fallback_used else "",
        }
        if not isinstance(repaired.get("used_core_path_ids"), list):
            repaired["used_core_path_ids"] = response.get("used_core_path_ids", [])
        if not isinstance(repaired.get("used_distractor_path_ids"), list):
            repaired["used_distractor_path_ids"] = response.get("used_distractor_path_ids", [])
        role_by_root_id = {
            str(item.get("path_id") or ""): str(item.get("role") or "")
            for item in available_root_constraints
            if str(item.get("path_id") or "")
        }
        declared_root_ids = list(
            dict.fromkeys(
                str(item)
                for item in [
                    *repaired.get("used_core_path_ids", []),
                    *repaired.get("used_distractor_path_ids", []),
                ]
                if str(item) in role_by_root_id
            )
        )
        repaired["used_core_path_ids"] = [
            path_id for path_id in declared_root_ids if role_by_root_id[path_id] == "core"
        ]
        repaired["used_distractor_path_ids"] = [
            path_id
            for path_id in declared_root_ids
            if role_by_root_id[path_id] == "distractor"
        ]
        repaired_selection = {
            "used_core_path_ids": [
                str(item) for item in repaired.get("used_core_path_ids", [])
            ],
            "used_distractor_path_ids": [
                str(item) for item in repaired.get("used_distractor_path_ids", [])
            ],
            "chain_node_usage": _as_list_of_dicts(
                repaired.get("chain_node_usage")
            ),
        }
        computed_selection_diff = _question_selection_diff(
            current_selection,
            repaired_selection,
            question_changed=original_question != repaired_question,
        )
        repaired["repair_actions"] = computed_selection_diff
        used_ids = {
            str(item)
            for item in [
                *repaired.get("used_core_path_ids", []),
                *repaired.get("used_distractor_path_ids", []),
            ]
        }
        used_constraint_snapshots = [
            item
            for item in available_root_constraints
            if str(item.get("path_id", "")) in used_ids
        ]
        repaired_participating_chain_nodes = _participating_chain_node_details(
            repaired_selection["chain_node_usage"],
            used_constraint_snapshots,
        )
        repaired["repair"] = {
            "attempted": True,
            "accepted": True,
            "original_question": original_question,
            "original_note": response.get("note", ""),
            "repair_note": repaired.get("note", ""),
            "repair_mode": repair_mode,
            "previous_selection": current_selection,
            "repaired_selection": repaired_selection,
            "computed_selection_diff": computed_selection_diff,
            "previous_participating_chain_nodes": participating_chain_nodes,
            "repaired_participating_chain_nodes": repaired_participating_chain_nodes,
            "used_constraint_snapshots": used_constraint_snapshots,
        }
        return repaired

    def _question_target_view(self, target: Target) -> Dict[str, Any]:
        return {
            "entity_type": target.entity_type,
            "answer_field": target.answer_field,
            "answer": target.answer,
        }

    def _question_constraint_view(
        self,
        path: ConstraintPath,
        *,
        require_leaf: bool,
        semantic_shortcut_ids: set[str] | None = None,
        avoid_path_ids: set[str] | None = None,
    ) -> Dict[str, Any]:
        deep_nodes = _deep_leaf_clue_nodes(path)
        eligible_leaf_clues = [
            item
            for item in deep_nodes
            if item.get("role") == "core"
        ]
        question_clue = ""
        question_clue_source = "missing_leaf_node_clue"
        question_clue_depth = 0
        question_node_path: List[str] = []
        selected_path_chain: List[Dict[str, Any]] = []
        if eligible_leaf_clues:
            max_depth = max(int(item["depth"]) for item in eligible_leaf_clues)
            deepest_candidates = [
                item
                for item in eligible_leaf_clues
                if int(item["depth"]) == max_depth
            ]
            shortcut_ids = semantic_shortcut_ids or set()
            avoided_ids = avoid_path_ids or set()
            deepest = min(
                deepest_candidates,
                key=lambda item: (
                    sum(
                        path_id in avoided_ids
                        for path_id in item.get("node_path", [])
                    ),
                    sum(
                        path_id in shortcut_ids
                        for path_id in item.get("node_path", [])
                    ),
                    len(str(item.get("clue") or "")),
                    tuple(str(value) for value in item.get("node_path", [])),
                ),
            )
            question_clue = deepest["clue"]
            question_clue_source = "leaf_node_clue"
            question_clue_depth = int(deepest["depth"])
            question_node_path = list(deepest.get("node_path", []))
            selected_path_chain = _selected_question_path_chain(
                path,
                question_node_path,
                semantic_shortcut_ids=semantic_shortcut_ids,
            )
        elif not require_leaf and path.clue:
            question_clue = path.clue
            question_clue_source = "root_clue"

        root_clue_forbidden = bool(path.local_constraints)
        return {
            "path_id": path.path_id,
            "role": path.role,
            "root_clue": path.clue,
            "question_clue": question_clue,
            "clue": question_clue,
            "question_clue_source": question_clue_source,
            "question_clue_depth": question_clue_depth,
            "question_node_path": question_node_path,
            "selected_path_chain": selected_path_chain,
            "root_clue_forbidden_for_question": False,
            "root_clue_verbatim_forbidden_for_question": root_clue_forbidden,
            "root_clue_omitted_for_question": False,
            "max_local_depth": _max_local_depth(path),
            "candidate_count_hint": len(path.candidates),
            "estimated_candidate_count": path.estimated_candidate_count,
        }

    def _validate_question_response(
        self,
        response: Dict[str, Any],
        target: Target,
        all_core_payload: Sequence[Dict[str, Any]],
        distractor_payload: Sequence[Dict[str, Any]],
        *,
        root_verifier: VerifierReport | None = None,
        required_core_path_ids: Sequence[str] = (),
    ) -> Dict[str, Any]:
        question = str(response.get("question", ""))
        # Structural-only mode keeps schema/tree checks in this method while
        # deferring semantic wording, shortcut, leakage, and candidate-set
        # decisions to Solver/Repair.
        structural_only = bool(
            getattr(self.config, "question_structural_only", False)
        )
        if not question.strip():
            return {
                "accepted": False,
                "reason": "empty question",
                "score": 0,
                "source": "basic_empty_guard",
            }
        declared_answer = response.get("answer")
        if declared_answer is not None and str(declared_answer).strip():
            if _entity_text_key(str(declared_answer)) != _entity_text_key(target.answer):
                return {
                    "accepted": False,
                    "reason": "question response answer does not match the seed answer",
                    "score": 0,
                    "source": "answer_contract_guard",
                }
        # The standard answer is program-owned metadata; do not retain a model
        # spelling variant in the artifact or training payload.
        response["answer"] = target.answer
        word_count = len(re.findall(r"\b\w+\b", question))
        if question.count("?") != 1:
            return {
                "accepted": False,
                "reason": (
                    "question must contain exactly one interrogative asking the "
                    "answer field"
                ),
                "score": 0,
                "source": "answer_type_structure_guard",
                "word_count": word_count,
            }
        meta_placeholder_patterns = (
            r"\bidentif(?:iable|ied)\s+without\s+(?:naming|being\s+named)\b",
            r"\b(?:name|identity)\s+(?:is|was)\s+(?:omitted|withheld|not\s+(?:given|stated|named))\b",
            r"\bwhose\s+(?:name|identity)\s+(?:is\s+)?not\s+(?:given|stated|named)\b",
        )
        if not structural_only and any(
            re.search(pattern, question, flags=re.IGNORECASE)
            for pattern in meta_placeholder_patterns
        ):
            return {
                "accepted": False,
                "reason": (
                    "question uses a meta redaction placeholder instead of a "
                    "public factual child clue"
                ),
                "score": 0,
                "source": "meta_placeholder_guard",
                "word_count": word_count,
            }
        used = response.get("used_core_path_ids", [])
        if not isinstance(used, list):
            used = []
        used = list(dict.fromkeys(str(item) for item in used if str(item)))
        valid_core_ids = {
            str(item.get("path_id") or "") for item in all_core_payload
        }
        unknown_core_ids = [path_id for path_id in used if path_id not in valid_core_ids]
        missing_required_ids = [
            str(path_id)
            for path_id in required_core_path_ids
            if str(path_id) not in set(used)
        ]
        if unknown_core_ids or missing_required_ids:
            return {
                "accepted": False,
                "reason": (
                    f"invalid root selection: unknown={unknown_core_ids}, "
                    f"missing_required={missing_required_ids}"
                ),
                "score": 0,
                "source": "root_selection_guard",
                "word_count": word_count,
            }
        leaf_core_ids = {
            item["path_id"]
            for item in all_core_payload
            if item.get("question_clue_source") == "leaf_node_clue"
            and item.get("question_clue_depth", 0) >= self.config.question_min_leaf_depth
        }
        used_leaf_ids = [item for item in used if item in leaf_core_ids]
        if word_count > self.config.question_max_words:
            return {
                "accepted": False,
                "reason": (
                    f"question has {word_count} words, above max "
                    f"{self.config.question_max_words}"
                ),
                "score": 0,
                "source": "question_length_guard",
                "word_count": word_count,
            }
        used_distractors = response.get("used_distractor_path_ids", [])
        if not isinstance(used_distractors, list):
            used_distractors = []
        valid_distractor_ids = {
            str(item.get("path_id", "")) for item in distractor_payload
        }
        used_distractor_ids = sorted(
            {
                str(item)
                for item in used_distractors
                if str(item) in valid_distractor_ids
            }
        )
        used_root_path_ids = {
            str(item) for item in used if str(item)
        } | set(used_distractor_ids)
        root_views_by_id = {
            str(item.get("path_id") or ""): item
            for item in [*all_core_payload, *distractor_payload]
            if str(item.get("path_id") or "")
        }
        unexpanded_used_roots = [
            root_id
            for root_id in sorted(set(used) & set(valid_core_ids))
            if not _question_view_has_expanded_chain(root_views_by_id.get(root_id, {}))
        ]
        if len(unexpanded_used_roots) > 1:
            return {
                "accepted": False,
                "reason": (
                    "question may use at most one unexpanded/root-only clue; "
                    f"found {len(unexpanded_used_roots)}"
                ),
                "unexpanded_root_path_ids": unexpanded_used_roots,
                "score": 0,
                "source": "root_chain_depth_guard",
                "word_count": word_count,
            }
        question_number_values = _number_values(question)
        generalized_required_root_ids = {
            str(path_id)
            for path_id in response.get("generalized_required_root_ids", [])
            if str(path_id) in set(required_core_path_ids)
        }
        response["generalized_required_root_ids"] = sorted(
            generalized_required_root_ids
        )
        root_fact_failures: List[Dict[str, Any]] = []
        if not structural_only:
            for constraint in [*all_core_payload, *distractor_payload]:
                root_id = str(constraint.get("path_id") or "")
                chain = constraint.get("selected_path_chain") or []
                root_only = (
                    not chain
                    or (
                        len(chain) == 1
                        and not (chain[0].get("child_clues") if isinstance(chain[0], dict) else [])
                    )
                )
                if (
                    root_id not in used_root_path_ids
                    or not root_only
                    or root_id in generalized_required_root_ids
                ):
                    continue
                required_values = _number_values(str(constraint.get("root_clue") or ""))
                missing_values = sorted(required_values - question_number_values)
                if missing_values:
                    root_fact_failures.append(
                        {
                            "path_id": root_id,
                            "missing_numeric_values": missing_values,
                            "root_clue": str(constraint.get("root_clue") or ""),
                        }
                    )
        if not structural_only and root_fact_failures:
            return {
                "accepted": False,
                "reason": "required root evidence lost an exact numeric/ordinal discriminator",
                "root_fact_failures": root_fact_failures,
                "score": 0,
                "source": "root_fact_fidelity_guard",
                "word_count": word_count,
            }
        private_references: List[Dict[str, str]] = [
            *(
                {"kind": "seed_target", "path_id": "", "text": str(value)}
                for value in _target_text_aliases(target)
                if str(value or "").strip()
            ),
            *(
                {"kind": "final_answer", "path_id": "", "text": alias}
                for alias in _answer_text_aliases(target)
            ),
        ]
        for constraint in [*all_core_payload, *distractor_payload]:
            root_id = str(constraint.get("path_id") or "")
            if root_id not in used_root_path_ids:
                continue
            for node in constraint.get("selected_path_chain", []):
                if not isinstance(node, dict):
                    continue
                local_target = node.get("local_target")
                if not isinstance(local_target, dict):
                    continue
                for key in ("name", "canonical_name", "entity_id"):
                    value = str(local_target.get(key) or "").strip()
                    if value:
                        private_references.append(
                            {
                                "kind": f"local_target_{key}",
                                "path_id": str(node.get("path_id") or ""),
                                "text": value,
                            }
                        )
        leaked_references: List[Dict[str, str]] = []
        seen_private: set[tuple[str, str]] = set()
        for item in private_references:
            text = str(item.get("text") or "").strip()
            key = (str(item.get("kind") or ""), _entity_text_key(text))
            if (
                text
                and key not in seen_private
                and _find_entity_text_reference(text, question)
            ):
                seen_private.add(key)
                leaked_references.append(item)
        if not structural_only and leaked_references:
            return {
                "accepted": False,
                "reason": "question exposes private construction labels",
                "leaked_references": leaked_references,
                "score": 0,
                "source": "private_reference_guard",
                "word_count": word_count,
            }
        chain_usage = _validate_chain_node_usage(
            response.get("chain_node_usage"),
            [*all_core_payload, *distractor_payload],
            used_root_path_ids=used_root_path_ids,
            min_root_core_children=getattr(
                self.config, "question_min_root_core_children", 1
            ),
            min_root_non_shortcut_children=getattr(
                self.config, "question_min_root_non_shortcut_children", 0
            ),
        )
        if not chain_usage["accepted"]:
            return {
                **chain_usage,
                "score": 0,
                "source": "chain_node_usage_guard",
                "word_count": word_count,
                "used_leaf_core_path_ids": sorted(set(used_leaf_ids)),
                "used_distractor_path_ids": used_distractor_ids,
            }
        response["chain_node_usage"] = chain_usage["normalized_usage"]
        used_child_ids = {
            child_id
            for item in chain_usage["normalized_usage"]
            for child_id in item.get("used_child_path_ids", [])
        }
        raw_fuzzified = response.get("fuzzified_child_path_ids", [])
        raw_fuzzified = raw_fuzzified if isinstance(raw_fuzzified, list) else []
        fuzzified_ids = list(
            dict.fromkeys(str(item) for item in raw_fuzzified if str(item))
        )
        invalid_fuzzified = [
            path_id for path_id in fuzzified_ids if path_id not in used_child_ids
        ]
        if invalid_fuzzified:
            return {
                "accepted": False,
                "reason": (
                    "fuzzified_child_path_ids are not used by chain_node_usage: "
                    + ", ".join(invalid_fuzzified)
                ),
                "score": 0,
                "source": "fuzzified_child_guard",
                "word_count": word_count,
            }
        response["fuzzified_child_path_ids"] = fuzzified_ids
        child_clues_by_id: Dict[str, Dict[str, Any]] = {}
        protected_root_numbers: set[int] = set()
        for constraint in [*all_core_payload, *distractor_payload]:
            if str(constraint.get("path_id") or "") not in used_root_path_ids:
                continue
            protected_root_numbers.update(
                _number_values(str(constraint.get("root_clue") or ""))
            )
            for node in constraint.get("selected_path_chain", []):
                if not isinstance(node, dict):
                    continue
                for child in node.get("child_clues", []):
                    if isinstance(child, dict) and str(child.get("path_id") or ""):
                        child_clues_by_id[str(child["path_id"])] = {
                            **child,
                            "_parent_depth": int(node.get("depth") or 0),
                        }
        shortcut_failures: List[Dict[str, Any]] = []
        for path_id in sorted(used_child_ids):
            child = child_clues_by_id.get(path_id) or {}
            if (
                not child.get("semantic_shortcut")
            ):
                continue
            overlap = _shortcut_text_overlap(str(child.get("clue") or ""), question)
            specificity_leaks = _semantic_shortcut_specificity_leaks(
                str(child.get("clue") or ""), question
            )
            specificity_leaks = [
                leak
                for leak in specificity_leaks
                if not (
                    _number_values(leak)
                    and _number_values(leak).issubset(protected_root_numbers)
                )
            ]
            if specificity_leaks:
                shortcut_failures.append(
                    {
                        "path_id": path_id,
                        "declared_fuzzified": path_id in fuzzified_ids,
                        "specificity_leaks": specificity_leaks,
                        "distinctive_token_overlap": round(overlap, 3),
                    }
                )
                continue
            if int(child.get("_parent_depth") or 0) == 0 and overlap >= 0.75:
                shortcut_failures.append(
                    {
                        "path_id": path_id,
                        "declared_fuzzified": path_id in fuzzified_ids,
                        "distinctive_token_overlap": round(overlap, 3),
                    }
                )
            elif path_id not in fuzzified_ids:
                fuzzified_ids.append(path_id)
        relation_failures: List[Dict[str, Any]] = []
        for path_id in sorted(used_child_ids):
            child = child_clues_by_id.get(path_id) or {}
            if not str(child.get("branch") or "").startswith("relation:"):
                continue
            leaked_names = _relation_named_entity_leaks(
                str(child.get("clue") or ""), question
            )
            if leaked_names:
                relation_failures.append(
                    {
                        "path_id": path_id,
                        "leaked_named_entities": leaked_names,
                        "distinctive_token_overlap": round(
                            _shortcut_text_overlap(str(child.get("clue") or ""), question),
                            3,
                        ),
                    }
                )
                continue
            overlap = _shortcut_text_overlap(str(child.get("clue") or ""), question)
            if overlap >= 0.85:
                relation_failures.append(
                    {
                        "path_id": path_id,
                        "distinctive_token_overlap": round(overlap, 3),
                    }
                )
            elif path_id not in fuzzified_ids:
                fuzzified_ids.append(path_id)
        response["fuzzified_child_path_ids"] = fuzzified_ids
        if not structural_only and shortcut_failures:
            return {
                "accepted": False,
                "reason": "semantic shortcut wording was not materially generalized",
                "shortcut_failures": shortcut_failures,
                "score": 0,
                "source": "semantic_shortcut_fuzzification_guard",
                "word_count": word_count,
            }
        if not structural_only and relation_failures:
            return {
                "accepted": False,
                "reason": "relation child exposes its named entity too directly",
                "relation_failures": relation_failures,
                "score": 0,
                "source": "relation_child_fuzzification_guard",
                "word_count": word_count,
            }
        if len(set(used_leaf_ids)) < self.config.question_min_leaf_core_paths:
            return {
                "accepted": False,
                "reason": (
                    "question uses too few core roots with a sufficiently deep leaf: "
                    f"{len(set(used_leaf_ids))}/{self.config.question_min_leaf_core_paths}"
                ),
                "score": 0,
                "source": "leaf_root_guard",
                "word_count": word_count,
                "used_leaf_core_path_ids": sorted(set(used_leaf_ids)),
            }
        if not structural_only and root_verifier is not None and not _root_ids_jointly_unique(
            root_verifier,
            used,
        ):
            return {
                "accepted": False,
                "reason": "used core root candidate sets do not jointly isolate the seed target",
                "score": 0,
                "source": "root_uniqueness_guard",
                "word_count": word_count,
                "used_leaf_core_path_ids": sorted(set(used_leaf_ids)),
            }
        if len(used_distractor_ids) < self.config.question_min_distractor_paths:
            return {
                "accepted": False,
                "reason": (
                    "question declared fewer than the required valid distractor paths: "
                    f"{len(used_distractor_ids)}/{self.config.question_min_distractor_paths}"
                ),
                "score": 0,
                "source": "distractor_path_guard",
                "word_count": word_count,
                "used_leaf_core_path_ids": sorted(set(used_leaf_ids)),
                "used_distractor_path_ids": used_distractor_ids,
                "chain_node_usage": chain_usage["normalized_usage"],
            }
        if structural_only or not self.config.question_verifier_enabled:
            return {
                "accepted": True,
                "reason": (
                    "LLM question verifier disabled; structural tree validation "
                    "passed and semantic checks are deferred to Solver/Repair"
                ),
                "score": 1,
                "source": "structural_question_verifier",
                "word_count": word_count,
                "used_leaf_core_path_ids": sorted(set(used_leaf_ids)),
                "used_distractor_path_ids": used_distractor_ids,
                "chain_node_usage": chain_usage["normalized_usage"],
            }
        intermediate_entity_audit: List[Dict[str, Any]] = []
        for constraint in [*all_core_payload, *distractor_payload]:
            root_id = str(constraint.get("path_id") or "")
            if root_id not in used_root_path_ids:
                continue
            for node in constraint.get("selected_path_chain", []):
                if not isinstance(node, dict):
                    continue
                local_target = node.get("local_target")
                if not isinstance(local_target, dict):
                    continue
                name = str(local_target.get("name") or "").strip()
                canonical = str(local_target.get("canonical_name") or "").strip()
                entity_id = str(local_target.get("entity_id") or "").strip()
                if not any((name, canonical, entity_id)):
                    continue
                intermediate_entity_audit.append(
                    {
                        "root_path_id": root_id,
                        "node_path_id": str(node.get("path_id") or ""),
                        "node_depth": int(node.get("depth") or 0),
                        "entity_type": str(local_target.get("entity_type") or ""),
                        "private_names": list(
                            dict.fromkeys(item for item in (name, canonical) if item)
                        ),
                        "private_entity_id": entity_id,
                        "used_child_path_ids": [
                            str(item.get("path_id") or "")
                            for item in node.get("child_clues", [])
                            if isinstance(item, dict)
                            and str(item.get("path_id") or "") in used_child_ids
                        ],
                    }
                )
        verifier_payload = {
            "target": {
                "entity_id": target.entity_id,
                "name": target.name,
                "entity_type": target.entity_type,
                "answer_field": target.answer_field,
                "answer": target.answer,
                "description": target.description,
                "source_urls": target.source_urls,
            },
            "question": question,
            "question_response": response,
            "leaf_core_path_ids": sorted(leaf_core_ids),
            "used_leaf_core_path_ids": sorted(set(used_leaf_ids)),
            "min_leaf_core_paths": self.config.question_min_leaf_core_paths,
            "min_distractor_paths": self.config.question_min_distractor_paths,
            "valid_distractor_path_ids": sorted(valid_distractor_ids),
            "used_distractor_path_ids": used_distractor_ids,
            "chain_node_usage": chain_usage["normalized_usage"],
            # This is verifier-only private context.  The Question/Repair model
            # receives neutral type labels instead of these names.
            "intermediate_entity_audit": intermediate_entity_audit,
            "distractor_constraint_chains": [
                _question_model_constraint_view(item)
                for item in distractor_payload
            ],
            "question_max_words": self.config.question_max_words,
            "word_count": word_count,
            "core_constraint_chains": [
                _question_model_constraint_view(item)
                for item in all_core_payload
                if item.get("question_clue")
            ],
        }
        try:
            verdict = self.runner.run_json(
                self.config.agents["question_verifier"],
                system_prompt=QUESTION_VERIFIER_PROMPT,
                user_payload=verifier_payload,
                rate_limit_scope=target.entity_id,
            )
        except Exception as exc:
            return {
                "accepted": False,
                "reason": f"question verifier failed: {exc}",
                "score": 0,
                "source": "llm_question_verifier",
                "word_count": word_count,
                "used_leaf_core_path_ids": sorted(set(used_leaf_ids)),
                "used_distractor_path_ids": used_distractor_ids,
                "chain_node_usage": chain_usage["normalized_usage"],
            }
        score = int(verdict.get("score", 0) or 0)
        verifier_reason = str(verdict.get("reason", ""))
        soft_warning = (
            score != 1
            and _question_verifier_soft_quality_warning(verifier_reason)
        )
        return {
            "accepted": score == 1 or soft_warning,
            "reason": (
                "soft_quality_warning: " + verifier_reason
                if soft_warning
                else verifier_reason
            ),
            "score": score,
            "source": "llm_question_verifier",
            "word_count": word_count,
            "used_leaf_core_path_ids": sorted(set(used_leaf_ids)),
            "used_distractor_path_ids": used_distractor_ids,
            "chain_node_usage": chain_usage["normalized_usage"],
        }

    def _adjudicate_solver_alternatives(
        self,
        artifact: Artifact,
        reports: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Use a stronger model to distinguish real ambiguity from Solver error."""
        candidates = []
        for item in reports:
            if item.get("verifier_is_correct") is not False:
                continue
            execution = item.get("_execution")
            execution = execution if isinstance(execution, dict) else {}
            model_response = str(item.get("model_response") or "").strip()
            if not model_response:
                continue
            candidates.append(
                {
                    "rollout_id": item.get("rollout_id"),
                    "model_answer": model_response[-8000:],
                    "verifier_reason": str(item.get("verifier_reason") or "")[:1200],
                    "search_queries": list(execution.get("search_queries") or [])[:40],
                    "api_call_count": int(execution.get("api_call_count") or 0),
                    "tool_call_count": int(execution.get("tool_call_count") or 0),
                }
            )
        agent = self.config.agents.get("solver_ambiguity")
        if not candidates or agent is None or not agent.configured:
            return {
                "verdict": "no_verified_alternative",
                "verified_alternatives": [],
                "unsupported_alternatives": [],
                "reason": "no wrong solver answer was available for independent adjudication",
                "source": "program_skip",
            }
        public_clauses = _public_question_clauses(artifact.question)
        required_clause_ids = [item["clause_id"] for item in public_clauses]
        payload = {
            "question": artifact.question,
            "public_clauses": public_clauses,
            "target": {
                "entity_type": artifact.target.entity_type,
                "answer_field": artifact.target.answer_field,
                "answer": artifact.target.answer,
                "name": artifact.target.name,
            },
            "wrong_solver_attempts": candidates,
            "selected_chain_usage": artifact.question_state.get(
                "chain_node_usage", []
            ),
        }
        try:
            response = self.runner.run_json(
                agent,
                system_prompt=SOLVER_ALTERNATIVE_ADJUDICATION_PROMPT,
                user_payload=payload,
                rate_limit_scope=artifact.target.entity_id,
                response_validator=lambda response: _validate_solver_ambiguity_response(
                    response,
                    required_clause_ids,
                ),
            )
        except Exception as exc:
            return {
                "verdict": "uncertain",
                "verified_alternatives": [],
                "unsupported_alternatives": [],
                "reason": f"ambiguity adjudicator failed: {type(exc).__name__}: {exc}",
                "source": "adjudicator_error",
            }
        verified = [
            item
            for item in response.get("verified_alternatives", [])
            if isinstance(item, dict)
            and item.get("matches_all_clues") is True
            and str(item.get("name") or "").strip()
            and _clue_checks_cover_clauses(
                item.get("clue_checks"), required_clause_ids
            )
            and isinstance(item.get("source_urls"), list)
            and any(str(url).strip() for url in item.get("source_urls", []))
        ]
        verdict = str(response.get("verdict") or "uncertain")
        if verified:
            verdict = "verified_ambiguity"
        elif verdict == "verified_ambiguity":
            verdict = "uncertain"
        return {
            "verdict": verdict,
            "verified_alternatives": verified,
            "unsupported_alternatives": response.get(
                "unsupported_alternatives", []
            ),
            "reason": str(response.get("reason") or ""),
            "source": "solver_ambiguity_agent",
        }

    def _summarize_solver_trajectories(
        self,
        artifact: Artifact,
        reports: Sequence[Dict[str, Any]],
        summary: SolverSummary,
    ) -> Dict[str, Any]:
        fallback_diagnosis = (
            "shortcut"
            if summary.status == "needs_repair:too_easy"
            else "too_hard"
            if summary.status == "needs_repair:too_hard"
            else "tool_failure"
        )
        compact_rollouts = []
        for item in reports:
            execution = item.get("_execution")
            execution = execution if isinstance(execution, dict) else {}
            compact_rollouts.append(
                {
                    "rollout_id": item.get("rollout_id"),
                    "correct": item.get("verifier_is_correct"),
                    "model_response": str(item.get("model_response") or "")[-500:],
                    "verifier_reason": str(item.get("verifier_reason") or "")[:600],
                    "api_call_count": int(execution.get("api_call_count") or 0),
                    "main_api_call_count": int(
                        execution.get("main_api_call_count") or 0
                    ),
                    "subagent_api_call_count": int(
                        execution.get("subagent_api_call_count") or 0
                    ),
                    "tool_call_count": int(execution.get("tool_call_count") or 0),
                    "successful_web_tool_count": int(
                        execution.get("successful_web_tool_count") or 0
                    ),
                    "search_queries": list(execution.get("search_queries") or [])[:10],
                    "reasoning_summaries": list(
                        execution.get("reasoning_summaries") or []
                    )[:4],
                }
            )
        fallback = {
            "diagnosis": fallback_diagnosis,
            "shared_success_path": [],
            "shortcut_queries": [],
            "shortcut_root_path_ids": [],
            "shortcut_child_path_ids": [],
            "shortcut_node_path_ids": [],
            "recoverable_intermediate_entities": [],
            "atomic_shortcut_facts": [],
            "useful_search_handles": [],
            "unresolved_steps": [],
            "recommended_mode": (
                "shortcut_prune"
                if fallback_diagnosis == "shortcut"
                else "solvability_restore"
            ),
            "repair_guidance": summary.reason,
            "source": "program_fallback",
            "rollouts": compact_rollouts,
        }
        if not getattr(self.config, "solver_trace_summary_enabled", True):
            return fallback

        shortcut_ids = _local_quality_shortcut_ids(artifact)
        root_views = [
            self._question_constraint_view(
                path,
                require_leaf=path.role == "core",
                semantic_shortcut_ids=shortcut_ids,
            )
            for path in artifact.constraints
        ]
        selected_ids = {
            str(item)
            for item in [
                *artifact.question_state.get("used_core_path_ids", []),
                *artifact.question_state.get("used_distractor_path_ids", []),
            ]
        }
        selected_views = [
            _question_model_constraint_view(item)
            for item in root_views
            if item.get("path_id") in selected_ids and item.get("selected_path_chain")
        ]
        participating_nodes = _participating_chain_node_details(
            _as_list_of_dicts(artifact.question_state.get("chain_node_usage")),
            selected_views,
        )
        selected_root_facts = [
            {
                "path_id": str(item.get("path_id") or ""),
                "role": str(item.get("role") or ""),
                "root_clue": str(item.get("root_clue") or ""),
                "required": str(item.get("path_id") or "")
                in set(artifact.question_state.get("required_core_root_ids") or []),
            }
            for item in root_views
            if str(item.get("path_id") or "") in selected_ids
        ]
        alternative_adjudication = self._adjudicate_solver_alternatives(
            artifact,
            reports,
        )
        try:
            response = self.runner.run_json(
                self.config.agents["trajectory_summary"],
                system_prompt=SOLVER_TRAJECTORY_SUMMARY_PROMPT,
                user_payload={
                    "question": artifact.question,
                    "difficulty_target": {
                        "api_calls": _solver_effort_requirements(self.config)["api_calls"],
                        "tool_calls": _solver_effort_requirements(self.config)["tool_calls"],
                        "relaxed_api_calls": _solver_effort_requirements(self.config)["relaxed_api_calls"],
                        "relaxed_tool_calls": _solver_effort_requirements(self.config)["relaxed_tool_calls"],
                        "correct": summary.correct,
                        "total": summary.total,
                        "prune_api_calls": _solver_prune_targets()[0],
                        "prune_tool_calls": _solver_prune_targets()[1],
                    },
                    "post_acceptance_pruning": bool(summary.accepted),
                    "alternative_adjudication": alternative_adjudication,
                    "participating_chain_nodes": participating_nodes,
                    "selected_root_facts": selected_root_facts,
                    "required_core_root_ids": list(
                        artifact.question_state.get("required_core_root_ids") or []
                    ),
                    "rollouts": compact_rollouts,
                },
                rate_limit_scope=artifact.target.entity_id,
                response_validator=_validate_trajectory_summary_response,
            )
        except Exception as exc:
            return {**fallback, "summary_error": f"{type(exc).__name__}: {exc}"}
        response["source"] = "trajectory_summary_agent"
        response["rollouts"] = compact_rollouts
        response["alternative_adjudication"] = alternative_adjudication
        early_memory_shortcuts = _detect_early_memory_shortcuts(
            compact_rollouts,
            selected_views,
            question=artifact.question,
        )
        if early_memory_shortcuts:
            response["early_memory_shortcuts"] = early_memory_shortcuts
            response["diagnosis"] = "shortcut"
            response["recommended_mode"] = "shortcut_prune"
            response["shortcut_node_path_ids"] = list(
                dict.fromkeys(
                    [
                        *response.get("shortcut_node_path_ids", []),
                        *[
                            item["node_path_id"]
                            for item in early_memory_shortcuts
                            if item.get("node_path_id")
                        ],
                    ]
                )
            )
            response["shortcut_child_path_ids"] = list(
                dict.fromkeys(
                    [
                        *response.get("shortcut_child_path_ids", []),
                        *[
                            child_id
                            for item in early_memory_shortcuts
                            for child_id in item.get("child_path_ids", [])
                        ],
                    ]
                )
            )
            recoverable_entities = [
                item
                for item in response.get("recoverable_intermediate_entities", [])
                if isinstance(item, dict)
            ]
            seen_recoverable_nodes = {
                str(item.get("node_path_id") or "")
                for item in recoverable_entities
            }
            for item in early_memory_shortcuts:
                node_id = str(item.get("node_path_id") or "")
                if node_id and node_id not in seen_recoverable_nodes:
                    seen_recoverable_nodes.add(node_id)
                    recoverable_entities.append(
                        {
                            "node_path_id": node_id,
                            "reason": item.get("reason", ""),
                        }
                    )
            response["recoverable_intermediate_entities"] = recoverable_entities
            response["repair_guidance"] = (
                "A correct rollout named a private intermediate before its first "
                "successful external result. Replace the listed node facts with "
                "a neutral type reference or a lower-information supplied sibling; "
                "do not merely reorder the same wording."
            )
        if alternative_adjudication.get("verdict") == "verified_ambiguity":
            response["diagnosis"] = "invalid_or_ambiguous"
            response["verified_alternatives"] = alternative_adjudication.get(
                "verified_alternatives", []
            )
            response["repair_guidance"] = (
                "A strong adjudicator verified a complete alternative. Add a tree-only "
                "complementary child that excludes it while preserving the requested field. "
                + str(alternative_adjudication.get("reason") or "")
            ).strip()
        elif (
            response.get("diagnosis") == "invalid_or_ambiguous"
            and not response.get("verified_alternatives")
        ):
            response["diagnosis"] = (
                "shortcut"
                if summary.correct > 0
                else "too_hard"
            )
            response["verified_alternatives"] = []
        valid_child_ids = {
            str(child.get("path_id") or "")
            for node in participating_nodes
            for child in node.get("used_child_clues", [])
            if isinstance(child, dict) and str(child.get("path_id") or "")
        }
        response["shortcut_child_path_ids"] = [
            str(path_id)
            for path_id in response.get("shortcut_child_path_ids", [])
            if str(path_id) in valid_child_ids
        ]
        response["shortcut_root_path_ids"] = [
            str(path_id)
            for path_id in response.get("shortcut_root_path_ids", [])
            if str(path_id) in selected_ids
        ]
        valid_node_ids = {
            str(node.get("node_path_id") or "")
            for node in participating_nodes
            if str(node.get("node_path_id") or "")
        }
        response["shortcut_node_path_ids"] = [
            str(path_id)
            for path_id in response.get("shortcut_node_path_ids", [])
            if str(path_id) in valid_node_ids
        ]
        response["recoverable_intermediate_entities"] = [
            item
            for item in response.get("recoverable_intermediate_entities", [])
            if isinstance(item, dict)
            and str(item.get("node_path_id") or "") in valid_node_ids
        ]
        response["atomic_shortcut_facts"] = [
            item
            for item in response.get("atomic_shortcut_facts", [])
            if isinstance(item, dict)
            and str(item.get("child_path_id") or "") in valid_child_ids
        ]
        available_child_ids = {
            str(child.get("path_id") or "")
            for node in participating_nodes
            for child in node.get("available_child_clues", [])
            if isinstance(child, dict) and str(child.get("path_id") or "")
        }
        normalized_ambiguity_points: List[Dict[str, Any]] = []
        for point in response.get("ambiguity_points", []):
            if not isinstance(point, dict):
                continue
            root_id = str(point.get("root_path_id") or "")
            node_id = str(point.get("node_path_id") or "")
            if root_id not in selected_ids or node_id not in valid_node_ids:
                continue
            candidates = [
                candidate
                for candidate in point.get("discriminating_child_candidates", [])
                if isinstance(candidate, dict)
                and str(candidate.get("child_path_id") or "") in available_child_ids
                and str(candidate.get("reason") or "").strip()
            ]
            normalized_ambiguity_points.append(
                {
                    **point,
                    "root_path_id": root_id,
                    "node_path_id": node_id,
                    "discriminating_child_candidates": candidates,
                }
            )
        response["ambiguity_points"] = normalized_ambiguity_points
        return response

    def _solver_repair_failure_report(
        self,
        reports: Sequence[Dict[str, Any]],
        summary: SolverSummary,
        previous_questions: Sequence[str],
        trajectory_feedback: Dict[str, Any],
    ) -> Dict[str, Any]:
        diagnosis = str(trajectory_feedback.get("diagnosis") or "")
        too_hard = summary.status == "needs_repair:too_hard"
        objective = (
            "repair_ambiguity"
            if diagnosis == "invalid_or_ambiguous"
            or summary.status == "needs_repair:ambiguous"
            else "restore_solvability"
            if too_hard
            else "prune_shortcut"
        )
        report = {
            "stage": "solver",
            "objective": objective,
            "reason": summary.reason,
            "status": summary.status,
            "correct": summary.correct,
            "total": summary.total,
            "effort_target": _solver_effort_requirements(self.config),
            "prune_target": {
                "api_calls": _solver_prune_targets()[0],
                "tool_calls": _solver_prune_targets()[1],
            },
            "successful_api_calls": list(summary.successful_api_calls),
            "successful_tool_calls": list(summary.successful_tool_calls),
            "trajectory_feedback": {
                key: value
                for key, value in trajectory_feedback.items()
                if key != "rollouts"
            },
        }
        adjudication = trajectory_feedback.get("alternative_adjudication")
        if isinstance(adjudication, dict):
            adjudicated_items = (
                adjudication.get("verified_alternatives", [])
                if adjudication.get("verdict") == "verified_ambiguity"
                else []
            )
        else:
            adjudicated_items = trajectory_feedback.get("verified_alternatives", [])
        verified_alternatives = [
            item
            for item in adjudicated_items
            if isinstance(item, dict) and item.get("matches_all_clues") is True
        ]
        if verified_alternatives and len(previous_questions) >= 2:
            report["prior_question"] = str(previous_questions[-2])
            report["verified_ambiguity"] = True
        return report

    def _run_solvers(
        self, artifact: Artifact, *, question_version: int = 0
    ) -> List[Dict[str, Any]]:
        jobs = list(range(self.config.solver_rollouts))
        max_workers = max(1, min(self.config.solver_concurrency, len(jobs)))
        reports_by_id: Dict[int, Dict[str, Any]] = {}
        self._log(f"solver_parallel_start: target={artifact.target.entity_id} jobs={len(jobs)} concurrency={max_workers}")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._run_one_solver,
                    artifact,
                    rollout_id,
                    question_version=question_version,
                ): rollout_id
                for rollout_id in jobs
            }
            for future in as_completed(futures):
                rollout_id = futures[future]
                reports_by_id[rollout_id] = future.result()
        self._log(f"solver_parallel_done: target={artifact.target.entity_id} reports={len(reports_by_id)}")
        return [reports_by_id[item] for item in jobs]

    def _run_one_solver(
        self,
        artifact: Artifact,
        rollout_id: int,
        *,
        question_version: int = 0,
    ) -> Dict[str, Any]:
        self._log(f"solver_rollout_start: target={artifact.target.entity_id} rollout={rollout_id}")
        try:
            trace_context = self._solver_trace_context(
                artifact,
                question_version=question_version,
                rollout_id=rollout_id,
            )
            report = self.runner.run_json(
                self.config.agents["solver"],
                system_prompt="",
                user_payload={"__raw_prompt": _solver_user_content(artifact)},
                rate_limit_scope=artifact.target.entity_id,
                trace_context=trace_context,
            )
        except Exception as exc:
            report = {
                "rollout_id": rollout_id,
                "final_answer": "",
                "confidence": "low",
                "evidence": [],
                "reasoning_summary": "",
                "solver_error": repr(exc),
                "verifier_is_correct": None,
                "verifier_status": "verification_failed",
                "verifier_reason": f"solver rollout failed after retries; correctness not evaluated: {exc}",
            }
            self._log(
                "solver_rollout_error: "
                f"target={artifact.target.entity_id} rollout={rollout_id} verification_failed=True "
                f"error={type(exc).__name__}"
            )
            self._write_solver_rollout_report(
                artifact,
                question_version=question_version,
                rollout_id=rollout_id,
                report=report,
            )
            return report
        report["rollout_id"] = rollout_id
        verifier = self._verify_solver_answer(
            report,
            artifact.target.answer,
            question=artifact.question,
            rate_limit_scope=artifact.target.entity_id,
            trace_context=trace_context,
        )
        report["verifier_is_correct"] = verifier["is_correct"]
        report["verifier_status"] = verifier["status"]
        report["verifier_reason"] = verifier["reason"]
        report["judge_trace_files"] = verifier.get("trace_files", {})
        self._log(
            "solver_rollout_done: "
            f"target={artifact.target.entity_id} rollout={rollout_id} correct={report['verifier_is_correct']}"
        )
        self._write_solver_rollout_report(
            artifact,
            question_version=question_version,
            rollout_id=rollout_id,
            report=report,
        )
        return report

    def _verify_solver_answer(
        self,
        solver_report: Any,
        expected_answer: str,
        *,
        question: str = "",
        rate_limit_scope: Optional[str] = None,
        trace_context: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        if isinstance(solver_report, dict):
            model_response = solver_report.get(
                "model_response", solver_report.get("final_answer", "")
            )
        else:
            model_response = solver_report
        try:
            response = self.runner.run_json(
                self.config.agents["solver_verifier"],
                system_prompt="",
                user_payload={
                    "__raw_prompt": _solver_judge_user_content(
                        question,
                        model_response,
                        expected_answer,
                    )
                },
                rate_limit_scope=rate_limit_scope,
                response_validator=_parse_solver_verifier_correct,
                trace_context=trace_context,
            )
            verdict = {
                "is_correct": _parse_solver_verifier_correct(response),
                "status": "verified",
                "reason": (
                    response.get("reason", "").strip()
                    if isinstance(response.get("reason"), str)
                    else ""
                ),
            }
            if response.get("_trace_files"):
                verdict["trace_files"] = response["_trace_files"]
            return verdict
        except Exception as exc:
            return {
                "is_correct": None,
                "status": "verification_failed",
                "reason": f"solver verifier failed after retries: {exc}",
            }

    def _summarize_solvers(self, reports: Sequence[Dict[str, Any]], expected_answer: str) -> SolverSummary:
        total = len(reports)
        correct = sum(1 for item in reports if item.get("verifier_is_correct") is True)
        incorrect = sum(1 for item in reports if item.get("verifier_is_correct") is False)
        verification_failed = total - correct - incorrect
        successful_api_calls = [
            int((item.get("_execution") or {}).get("api_call_count") or 0)
            for item in reports
            if item.get("verifier_is_correct") is True
        ]
        successful_tool_calls = [
            int((item.get("_execution") or {}).get("tool_call_count") or 0)
            for item in reports
            if item.get("verifier_is_correct") is True
        ]
        min_api = min(successful_api_calls, default=0)
        median_api = (
            float(median(successful_api_calls)) if successful_api_calls else 0.0
        )
        max_api = max(successful_api_calls, default=0)
        mean_api = (
            round(sum(successful_api_calls) / len(successful_api_calls), 2)
            if successful_api_calls
            else 0.0
        )
        min_tools = min(successful_tool_calls, default=0)
        median_tools = (
            float(median(successful_tool_calls)) if successful_tool_calls else 0.0
        )
        max_tools = max(successful_tool_calls, default=0)
        mean_tools = (
            round(sum(successful_tool_calls) / len(successful_tool_calls), 2)
            if successful_tool_calls
            else 0.0
        )
        effort = _solver_effort_gate(
            median_api,
            median_tools,
            getattr(self, "config", None),
        )

        def result(
            *,
            accepted: bool,
            status: str,
            reason: str,
        ) -> SolverSummary:
            return SolverSummary(
                total=total,
                correct=correct,
                incorrect=incorrect,
                accepted=accepted,
                status=status,
                reason=reason,
                verification_failed=verification_failed,
                successful_api_calls=successful_api_calls,
                successful_tool_calls=successful_tool_calls,
                min_success_api_calls=min_api,
                median_success_api_calls=median_api,
                max_success_api_calls=max_api,
                mean_success_api_calls=mean_api,
                min_success_tool_calls=min_tools,
                median_success_tool_calls=median_tools,
                max_success_tool_calls=max_tools,
                mean_success_tool_calls=mean_tools,
                below_target_api_successes=sum(
                    value < effort["requirements"]["api_calls"]
                    for value in successful_api_calls
                ),
                below_target_tool_successes=sum(
                    value < effort["requirements"]["tool_calls"]
                    for value in successful_tool_calls
                ),
                effort_relaxation_used=bool(effort["relaxation_used"]),
            )

        if total < 3:
            return result(
                accepted=False,
                status="rejected:too_few_solver_rollouts",
                reason="need n>=3",
            )
        if verification_failed:
            return result(
                accepted=False,
                status="verification_failed",
                reason=(
                    f"{verification_failed}/{total} solver rollouts could not be verified after retries; "
                    "not counted as incorrect"
                ),
            )
        if correct == 0:
            return result(
                accepted=True,
                status="review:all_wrong",
                reason=(
                    "all solver rollouts were wrong; retain the question unless strong "
                    "ambiguity adjudication verifies that a wrong answer also satisfies "
                    "the complete public wording"
                ),
            )
        if not effort["accepted"]:
            return result(
                accepted=False,
                status="needs_repair:too_easy",
                reason=str(effort["reason"]),
            )
        return result(
            accepted=True,
            status=(
                "accepted:hard"
                if correct * 3 < total * 2
                else "accepted:high_effort"
            ),
            reason=(
                f"{correct}/{total} rollouts were correct; median API calls="
                f"{median_api:g}, median tool calls={median_tools:g}; "
                f"effort mode={effort['mode']}"
            ),
        )


def _solver_effort_requirements(config: Any) -> Dict[str, int]:
    api_calls = max(1, int(getattr(config, "solver_min_success_api_calls", 50)))
    tool_calls = max(1, int(getattr(config, "solver_min_success_tool_calls", 120)))
    return {
        "api_calls": api_calls,
        "tool_calls": tool_calls,
        "relaxed_api_calls": min(
            api_calls,
            max(1, int(getattr(config, "solver_relaxed_min_success_api_calls", 45))),
        ),
        "relaxed_tool_calls": min(
            tool_calls,
            max(1, int(getattr(config, "solver_relaxed_min_success_tool_calls", 108))),
        ),
        "too_easy_api_calls": max(
            0, int(getattr(config, "solver_too_easy_api_calls", 0))
        ),
        "accept_api_calls": max(
            0, int(getattr(config, "solver_accept_api_calls", 0))
        ),
    }


def _solver_prune_targets() -> tuple[int, int]:
    return (
        max(1, int(os.environ.get("V2_SOLVER_PRUNE_TARGET_API_CALLS", "50"))),
        max(1, int(os.environ.get("V2_SOLVER_PRUNE_TARGET_TOOL_CALLS", "80"))),
    )


def _solver_prune_targets_reached(summary: Any) -> bool:
    api_target, tool_target = _solver_prune_targets()
    if isinstance(summary, dict):
        api_calls = float(summary.get("median_success_api_calls") or 0)
        tool_calls = float(summary.get("median_success_tool_calls") or 0)
    else:
        api_calls = float(getattr(summary, "median_success_api_calls", 0) or 0)
        tool_calls = float(getattr(summary, "median_success_tool_calls", 0) or 0)
    return api_calls >= api_target and tool_calls >= tool_target


def _solver_summary_score(summary: Any) -> tuple[float, float]:
    """Prefer accepted versions with more API turns, then more tool work."""
    if isinstance(summary, dict):
        return (
            float(summary.get("median_success_api_calls") or 0),
            float(summary.get("median_success_tool_calls") or 0),
        )
    return (
        float(getattr(summary, "median_success_api_calls", 0) or 0),
        float(getattr(summary, "median_success_tool_calls", 0) or 0),
    )


def _solver_effort_gate(
    median_api_calls: float,
    median_tool_calls: float,
    config: Any,
) -> Dict[str, Any]:
    requirements = _solver_effort_requirements(config)
    if requirements["accept_api_calls"] > 0:
        too_easy_limit = requirements["too_easy_api_calls"]
        accept_limit = requirements["accept_api_calls"]
        if median_api_calls < too_easy_limit:
            return {
                "accepted": False,
                "mode": "api_calls_too_easy",
                "relaxation_used": False,
                "requirements": requirements,
                "reason": (
                    f"API median {median_api_calls:g} is below too-easy limit "
                    f"{too_easy_limit}; tool median={median_tool_calls:g}"
                ),
            }
        if median_api_calls > accept_limit:
            return {
                "accepted": True,
                "mode": "api_calls_accepted",
                "relaxation_used": median_tool_calls < requirements["tool_calls"],
                "requirements": requirements,
                "reason": (
                    f"API median {median_api_calls:g} exceeds acceptance limit "
                    f"{accept_limit}; tool median={median_tool_calls:g} is recorded "
                    "separately"
                ),
            }
        return {
            "accepted": False,
            "mode": "api_calls_transition",
            "relaxation_used": False,
            "requirements": requirements,
            "reason": (
                f"API median {median_api_calls:g} is in the transition band "
                f"[{too_easy_limit}, {accept_limit}]; continue shortcut pruning"
            ),
        }
    api_full = median_api_calls >= requirements["api_calls"]
    tools_full = median_tool_calls >= requirements["tool_calls"]
    if api_full and tools_full:
        return {
            "accepted": True,
            "mode": "both_full",
            "relaxation_used": False,
            "requirements": requirements,
            "reason": "both API and tool-call medians meet their full thresholds",
        }
    if api_full and median_tool_calls >= requirements["relaxed_tool_calls"]:
        return {
            "accepted": True,
            "mode": "tool_calls_relaxed",
            "relaxation_used": True,
            "requirements": requirements,
            "reason": "API median is full and tool-call median meets its relaxed threshold",
        }
    if tools_full and median_api_calls >= requirements["relaxed_api_calls"]:
        return {
            "accepted": True,
            "mode": "api_calls_relaxed",
            "relaxation_used": True,
            "requirements": requirements,
            "reason": "tool-call median is full and API median meets its relaxed threshold",
        }
    if (
        median_api_calls >= requirements["relaxed_api_calls"]
        and median_tool_calls >= requirements["relaxed_tool_calls"]
    ):
        return {
            "accepted": True,
            "mode": "both_minimum",
            "relaxation_used": True,
            "requirements": requirements,
            "reason": (
                "both API and tool-call medians meet the minimum acceptable "
                f"thresholds ({requirements['relaxed_api_calls']}/"
                f"{requirements['relaxed_tool_calls']}); continue pruning toward "
                f"{_solver_prune_targets()[0]}/{_solver_prune_targets()[1]}"
            ),
        }
    return {
        "accepted": False,
        "mode": "below_target",
        "relaxation_used": False,
        "requirements": requirements,
        "reason": (
            f"separate effort medians are below target: API={median_api_calls:g} "
            f"(full {requirements['api_calls']}, relaxed {requirements['relaxed_api_calls']}), "
            f"tools={median_tool_calls:g} (full {requirements['tool_calls']}, "
            f"relaxed {requirements['relaxed_tool_calls']})"
        ),
    }


def _target_with_verified_seed_sources(
    target: Target,
    verification: Dict[str, Any],
) -> Target:
    """Carry exact-target sources discovered by Seed Verifier downstream."""
    source_urls: List[str] = list(target.source_urls)
    reports = [verification, *_as_list_of_dicts(verification.get("rollouts"))]
    for report in reports:
        raw_urls = report.get("source_urls")
        if not isinstance(raw_urls, list):
            continue
        source_urls.extend(
            str(url).strip()
            for url in raw_urls
            if str(url).strip().startswith(("http://", "https://"))
        )
    merged = list(dict.fromkeys(url for url in source_urls if str(url).strip()))
    return replace(target, source_urls=merged)


def load_seed(path: str | Path) -> SeedRecord:
    return SeedRecord.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def save_seed(seed: SeedRecord, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(seed), ensure_ascii=False, indent=2), encoding="utf-8")


def save_artifact(artifact: Artifact, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(artifact), ensure_ascii=False, indent=2), encoding="utf-8")


def _answer_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _entity_text_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(part for part in re.split(r"[\W_]+", normalized) if part)


def _source_url_key(value: str) -> str:
    """Normalize a source URL for parent/child and root-diversity checks."""
    text = str(value or "").strip()
    if not text:
        return ""
    parsed = urlparse(text)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme.casefold()}://{parsed.netloc.casefold()}{parsed.path.rstrip('/')}"
    return text.rstrip("/").casefold()


def _find_entity_text_reference(entity_name: str, source_text: str) -> str:
    """Match a local-target name while ignoring only case and punctuation."""
    entity_key = _entity_text_key(entity_name)
    source_key = _entity_text_key(source_text)
    if not entity_key or not source_key:
        return ""
    if f" {entity_key} " in f" {source_key} ":
        return entity_name
    if any(ord(char) > 127 for char in entity_key):
        if entity_key.replace(" ", "") in source_key.replace(" ", ""):
            return entity_name
    return ""


def _local_branch_kind(branch: str) -> str:
    prefix = str(branch or "").strip().casefold().split(":", 1)[0]
    if prefix == "relation":
        return "relation"
    if prefix in {"attribute", "property"}:
        return "attribute"
    return ""


def _verify_local_tree_structure(
    target: Target,
    constraints: Sequence[ConstraintPath],
    *,
    min_core_paths: int,
    min_distractor_paths: int,
) -> VerifierReport:
    """Check Local tree shape; real uniqueness belongs to the semantic verifier."""
    cores = [path for path in constraints if path.role == "core"]
    distractors = [path for path in constraints if path.role == "distractor"]
    target_key = _entity_text_key(target.entity_id or target.name)
    if len(cores) < max(2, min_core_paths):
        return VerifierReport(
            False,
            f"fewer than {max(2, min_core_paths)} local core paths",
            [path.path_id for path in cores],
            target_key,
            {},
            [],
            {"count": len(distractors)},
        )
    if len(distractors) < max(0, min_distractor_paths):
        return VerifierReport(
            False,
            f"fewer than {max(0, min_distractor_paths)} local distractor paths",
            [path.path_id for path in cores],
            target_key,
            {},
            [],
            {"count": len(distractors)},
        )
    path_ids = [path.path_id for path in constraints]
    if any(not path_id for path_id in path_ids) or len(path_ids) != len(set(path_ids)):
        return VerifierReport(
            False,
            "local child path ids must be non-empty and unique",
            [path.path_id for path in cores],
            target_key,
            {},
            [],
            {"count": len(distractors)},
        )
    return VerifierReport(
        True,
        "local tree structure is valid; factual quality is checked separately",
        [path.path_id for path in cores],
        target_key,
        {},
        [],
        {"count": len(distractors)},
    )


def _local_quality_is_accepted(quality: Dict[str, Any]) -> bool:
    if quality.get("skipped") is True:
        return True
    target_failures = quality.get("target_failures")
    target_failures = target_failures if isinstance(target_failures, list) else []
    return (
        quality.get("target_valid") is True
        and quality.get("coherent") is True
        and not target_failures
    )


def _local_shallow_shortcut_failure(
    path: ConstraintPath,
    quality: Dict[str, Any],
    *,
    required: int,
) -> Dict[str, Any] | None:
    core_ids = [
        child.path_id for child in path.local_constraints if child.role == "core"
    ]
    shortcut_items = quality.get("single_clue_shortcuts")
    shortcut_items = shortcut_items if isinstance(shortcut_items, list) else []
    shortcut_ids = {
        str(item.get("path_id") or "")
        for item in shortcut_items
        if isinstance(item, dict) and str(item.get("path_id") or "") in core_ids
    }
    required_safe = min(max(0, int(required or 0)), len(core_ids))
    if required_safe == 0:
        return None
    safe_ids = [path_id for path_id in core_ids if path_id not in shortcut_ids]
    if len(safe_ids) >= required_safe:
        return None
    return {
        "reason": (
            "depth-0 Local bundle has too few non-shortcut core clues: "
            f"{len(safe_ids)}/{required_safe}"
        ),
        "required_non_shortcut_core_paths": required_safe,
        "non_shortcut_core_path_ids": safe_ids,
        "shortcut_core_path_ids": sorted(shortcut_ids),
        "quality_repair_path_ids": sorted(shortcut_ids),
    }


def _local_target_surface_problem(surface_text: str, entity_type: str) -> str:
    """Reject only clearly weak model-extracted spans; semantic choice stays with LLM."""
    text = str(surface_text or "").strip()
    key = _entity_text_key(text)
    if not text or not key:
        return "local target surface_text is empty"
    if len(text) > 100 or len(key.split()) > 10:
        return "local target is too long to be one useful named entity"
    if not any(char.isalpha() for char in text):
        return "local target is numeric/temporal rather than a named entity"
    weak = {
        "the programme", "the program", "the episode", "the record", "the work",
        "the item", "the archive", "the series", "the event", "the project",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
        "sunday", "british", "american", "english", "european", "asian",
        "african", "australian", "canadian", "french", "german", "italian",
        "spanish", "china", "united states", "united kingdom", "england",
    }
    if key in weak:
        return f"local target {text!r} is a generic, temporal, or broad geographic reference"
    if re.fullmatch(r"(?:the )?(?:19|20)\d{2}s?", key):
        return f"local target {text!r} is a date or decade"
    type_key = _entity_text_key(entity_type)
    if type_key in {"date", "year", "decade", "country", "demonym", "adjective"}:
        return f"local target entity_type {entity_type!r} is not useful for recursive expansion"
    return ""


def _selected_local_target_forbidden_reference(
    surface_text: str,
    *,
    final_target: Target | None,
    ancestor_local_targets: Sequence[Dict[str, str]],
) -> Dict[str, str] | None:
    if final_target is None:
        return None
    for reference in _local_child_clue_forbidden_references(
        final_target,
        ancestor_local_targets,
    ):
        value = str(reference.get("value") or "").strip()
        if value and _entity_text_key(surface_text) == _entity_text_key(value):
            return reference
    return None


def _apply_local_expansion_response(
    path: ConstraintPath,
    response: Dict[str, Any],
) -> ConstraintPath:
    """Merge model-owned Local fields onto the program-owned parent node."""
    updated = replace(
        path,
        local_target_id="",
        local_target_name="",
        local_target_canonical_name="",
        local_target_type="",
        local_constraints=[],
    )
    if str(response.get("action") or "").strip().lower() != "expand":
        return updated
    raw_target = response.get("local_target")
    raw_target = raw_target if isinstance(raw_target, dict) else {}
    children = [
        ConstraintPath.from_dict(item)
        for item in response.get("local_constraints", [])
        if isinstance(item, dict)
    ]
    return replace(
        updated,
        local_target_id=str(raw_target.get("entity_id") or "").strip(),
        local_target_name=str(raw_target.get("surface_text") or "").strip(),
        local_target_canonical_name=str(
            raw_target.get("canonical_name") or ""
        ).strip(),
        local_target_type=str(raw_target.get("entity_type") or "").strip(),
        local_constraints=children,
    )


def _compact_local_retry_feedback(
    report: Dict[str, Any],
    *,
    attempt: int,
) -> Dict[str, Any]:
    """Keep only the actionable verifier failure for one no-browse retry."""
    kind = str(report.get("failure_kind") or "local_structure_rejected")
    feedback: Dict[str, Any] = {
        "attempt": attempt,
        "code": kind,
        "message": str(report.get("reason") or "local verifier rejected expansion"),
    }
    required_fixes: List[str] = []
    if kind in {
        "local_target_not_in_current_clue",
        "local_target_not_useful_entity",
        "forbidden_local_target_selection",
        "missing_local_target",
    }:
        required_fixes.append(
            "Extract a different useful named entity by copying an exact contiguous "
            "substring from current_node.clue, or return action='stop'."
        )
    if kind == "local_stop_before_depth":
        required_fixes.append(
            "The current clue is a relation branch. Do not stop if it contains an "
            "exact named entity; copy that contiguous span into local_target and "
            "return atomic child clues. Stop only when no such span exists."
        )
    raw_leaks = report.get("child_clue_leaks")
    leaks = (
        [item for item in raw_leaks if isinstance(item, dict)]
        if isinstance(raw_leaks, list)
        else []
    )
    if not leaks and isinstance(report.get("child_clue_leak"), dict):
        leaks = [report["child_clue_leak"]]
    if leaks:
        feedback["offending_children"] = [
            {
                "path_id": str(item.get("path_id") or ""),
                "forbidden_kind": str(item.get("kind") or ""),
                "forbidden_text": str(item.get("matched_value") or ""),
            }
            for item in leaks
        ]
        private_target_leaks = any(
            str(item.get("kind") or "") in {"seed_target", "final_answer"}
            for item in leaks
        )
        if private_target_leaks:
            required_fixes.append(
                "For every seed_target/final_answer leak, replace that child clue "
                "and evidence with a different truthful atomic fact or relation "
                "about the current local target. Do not paraphrase, describe, or "
                "indirectly hint at forbidden_text. Preserve non-offending fields."
            )
        else:
            required_fixes.append(
                "Rewrite every offending child clue in one response so its "
                "forbidden_text is absent; preserve all other fields."
            )
    if kind in {
        "child_clue_not_atomic",
        "child_clue_missing_evidence",
        "child_source_url_reused",
    }:
        feedback["offending_child_path_id"] = str(
            report.get("offending_child_path_id") or ""
        )
        required_fixes.append(
            "Repair only the offending child's clue/evidence and preserve all "
            "other fields exactly."
        )
    if kind == "child_source_url_reused":
        feedback["offending_child_path_ids"] = [
            str(item) for item in report.get("offending_child_path_ids", [])
        ]
        feedback["reused_source_urls"] = dict(
            report.get("reused_source_urls") or {}
        )
        required_fixes.append(
            "Replace the offending child evidence with a different source URL; "
            "the new source must still explicitly support the child clue and name "
            "the current local target."
        )
    if kind == "child_clues_not_distinct":
        feedback["offending_child_path_ids"] = [
            str(item) for item in report.get("offending_child_path_ids", [])
        ]
        required_fixes.append(
            "Replace each duplicate child with a different truthful atomic fact "
            "about the local target; core and distractor clues must not duplicate."
        )
    membership_unsupported = [
        str(item) for item in report.get("offending_child_path_ids", [])
    ]
    if kind == "child_target_membership_unsupported" and membership_unsupported:
        feedback["membership_unsupported_path_ids"] = membership_unsupported
        required_fixes.append(
            "Repair every offending child in one response. Remove all forbidden "
            "text. For membership_unsupported_path_ids, replace clue/evidence with "
            "a truthful atomic fact that the local target satisfies "
            "and evidence that explicitly names the canonical local target."
        )
    if kind == "child_core_branches_not_diverse":
        feedback["branch_repair_path_ids"] = membership_unsupported
        feedback["branches"] = report.get("branches", {})
        required_fixes.append(
            "Replace the listed cores with distinct property/relation branches; "
            "each replacement must remain evidence-backed and true of the target."
        )
    if kind == "child_core_kind_imbalance":
        feedback["branch_repair_path_ids"] = membership_unsupported
        feedback["required_relation_core_paths"] = int(
            report.get("required_relation_core_paths") or 0
        )
        feedback["required_attribute_core_paths"] = int(
            report.get("required_attribute_core_paths") or 0
        )
        feedback["branches"] = report.get("branches", {})
        required_fixes.append(
            "Replace only enough listed cores to satisfy relation:<dimension> and "
            "attribute:<dimension> counts; preserve target truth and evidence."
        )
    if kind == "insufficient_expandable_local_core_paths":
        feedback["expandable_repair_path_ids"] = [
            str(item)
            for item in report.get("non_expandable_core_path_ids", [])
        ]
        feedback["required_expandable_core_paths"] = int(
            report.get("required") or 0
        )
        required_fixes.append(
            "Replace only enough expandable_repair_path_ids with broad relations "
            "containing one new named entity, plus truthful target evidence."
        )
    if kind == "local_quality_failure":
        quality = report.get("local_quality_verifier")
        quality = quality if isinstance(quality, dict) else {}
        target_failures = [
            item for item in quality.get("target_failures", []) if isinstance(item, dict)
        ]
        suggestions = [
            item for item in quality.get("repair_suggestions", []) if isinstance(item, dict)
        ]
        feedback["quality_repair_path_ids"] = list(
            dict.fromkeys(
                str(item.get("path_id") or "")
                for item in [*target_failures, *suggestions]
                if str(item.get("path_id") or "")
            )
        )
        feedback["target_failures"] = target_failures
        feedback["quality_repair_suggestions"] = suggestions
        required_fixes.append(
            "Repair only quality_repair_path_ids so every child is true of the "
            "local target and entity scopes are coherent. Do not add uniqueness clues."
        )
    if kind == "insufficient_non_shortcut_local_cores":
        feedback["quality_repair_path_ids"] = [
            str(item) for item in report.get("shortcut_core_path_ids", [])
        ]
        feedback["required_non_shortcut_core_paths"] = int(
            report.get("required_non_shortcut_core_paths") or 0
        )
        feedback["non_shortcut_core_path_ids"] = [
            str(item) for item in report.get("non_shortcut_core_path_ids", [])
        ]
        required_fixes.append(
            "At depth 0, replace only enough quality_repair_path_ids with broad "
            "atomic facts that retain real same-type alternatives when searched "
            "alone. Avoid exact work titles, eponymous aphorisms, unique awards, "
            "identifiers, and rare verbatim phrases. Preserve verified safe children."
        )
    verifier = report.get("verifier")
    if isinstance(verifier, dict):
        feedback["structural_gate"] = {
            "reason": str(verifier.get("reason") or ""),
            "core_path_ids": [
                str(item) for item in verifier.get("core_path_ids", [])
            ],
        }
    if required_fixes:
        feedback["required_fix"] = " ".join(required_fixes)
    return feedback


def _merge_local_retry_response(
    previous: Dict[str, Any],
    repaired: Dict[str, Any],
    feedback: Dict[str, Any],
) -> Dict[str, Any]:
    """Enforce retry ownership so a narrow repair cannot redesign valid output."""
    previous_copy = json.loads(json.dumps(previous, ensure_ascii=False))
    repaired_copy = json.loads(json.dumps(repaired, ensure_ascii=False))
    code = str(feedback.get("code") or "")
    if code in {
        "forbidden_reference_in_child_clue",
        "child_source_url_reused",
        "child_clue_not_atomic",
        "child_clue_missing_evidence",
        "child_clues_not_distinct",
        "child_target_membership_unsupported",
        "child_core_branches_not_diverse",
        "child_core_kind_imbalance",
        "insufficient_expandable_local_core_paths",
        "local_quality_failure",
        "insufficient_non_shortcut_local_cores",
    }:
        repaired_children = {
            str(item.get("path_id") or ""): item
            for item in repaired_copy.get("local_constraints", [])
            if isinstance(item, dict)
        }
        editable_ids = {
            str(item.get("path_id") or "")
            for item in feedback.get("offending_children", [])
            if isinstance(item, dict)
        }
        if not editable_ids and feedback.get("offending_child_path_id"):
            editable_ids.add(str(feedback["offending_child_path_id"]))
        editable_ids.update(
            str(item) for item in feedback.get("offending_child_path_ids", [])
        )
        membership_ids = {
            str(item)
            for item in feedback.get("membership_unsupported_path_ids", [])
        }
        editable_ids.update(membership_ids)
        broad_repair_ids = {
            str(item)
            for item in [
                *feedback.get("branch_repair_path_ids", []),
                *feedback.get("expandable_repair_path_ids", []),
                *feedback.get("quality_repair_path_ids", []),
            ]
        }
        editable_ids.update(broad_repair_ids)
        if (
            code == "local_quality_failure"
            and "local_target.entity_id" in broad_repair_ids
        ):
            previous_target = previous_copy.get("local_target")
            repaired_target = repaired_copy.get("local_target")
            if isinstance(previous_target, dict) and isinstance(repaired_target, dict):
                repaired_entity_id = str(
                    repaired_target.get("entity_id") or ""
                ).strip()
                if repaired_entity_id:
                    # The verifier explicitly owns this correction. Preserve the
                    # exact surface selection and all other target fields.
                    previous_target["entity_id"] = repaired_entity_id
        previous_ids = {
            str(item.get("path_id") or "")
            for item in previous_copy.get("local_constraints", [])
            if isinstance(item, dict)
        }
        replacement_pool = [
            item
            for item in repaired_copy.get("local_constraints", [])
            if isinstance(item, dict)
            and str(item.get("path_id") or "") not in previous_ids
            and str(item.get("clue") or "").strip()
        ]
        used_replacements: set[str] = set()
        for child in previous_copy.get("local_constraints", []):
            if not isinstance(child, dict):
                continue
            path_id = str(child.get("path_id") or "")
            new_child = repaired_children.get(path_id)
            if path_id in editable_ids and not isinstance(new_child, dict):
                old_role = str(child.get("role") or "")
                old_kind = _local_branch_kind(str(child.get("branch") or ""))
                new_child = next(
                    (
                        candidate
                        for candidate in replacement_pool
                        if str(candidate.get("path_id") or "")
                        not in used_replacements
                        and str(candidate.get("role") or "") == old_role
                        and (
                            path_id in broad_repair_ids
                            or _local_branch_kind(
                                str(candidate.get("branch") or "")
                            ) == old_kind
                        )
                    ),
                    None,
                )
                if isinstance(new_child, dict):
                    used_replacements.add(str(new_child.get("path_id") or ""))
                    replacement = json.loads(
                        json.dumps(new_child, ensure_ascii=False)
                    )
                    replacement["path_id"] = path_id
                    child.clear()
                    child.update(replacement)
                    continue
            if (
                path_id in editable_ids
                and isinstance(new_child, dict)
                and str(new_child.get("clue") or "").strip()
            ):
                child["clue"] = str(new_child["clue"])
                if isinstance(new_child.get("evidence"), list):
                    child["evidence"] = new_child["evidence"]
                if path_id in broad_repair_ids:
                    child["branch"] = str(new_child.get("branch") or child.get("branch") or "")
        return previous_copy
    if code in {
        "local_target_not_in_current_clue",
        "local_target_not_useful_entity",
        "forbidden_local_target_selection",
        "missing_local_target",
    }:
        return repaired_copy
    return repaired_copy


def _local_child_clue_forbidden_references(
    final_target: Target,
    ancestor_local_targets: Sequence[Dict[str, str]],
    current_local_target: Target | None = None,
) -> List[Dict[str, str]]:
    references: List[Dict[str, str]] = []

    def add(kind: str, value: Any) -> None:
        text = str(value or "").strip()
        if text and not any(
            item["kind"] == kind and _entity_text_key(item["value"]) == _entity_text_key(text)
            for item in references
        ):
            references.append({"kind": kind, "value": text})

    for alias in _target_text_aliases(final_target):
        add("seed_target", alias)
    for alias in _answer_text_aliases(final_target):
        add("final_answer", alias)
    for ancestor in ancestor_local_targets:
        add("ancestor_local_target", ancestor.get("entity_id"))
        add("ancestor_local_target", ancestor.get("name"))
        add("ancestor_local_target", ancestor.get("canonical_name"))
    if current_local_target is not None:
        add("current_local_target", current_local_target.entity_id)
        add("current_local_target", current_local_target.name)
    return references


def _target_text_aliases(target: Target) -> List[str]:
    """Return explicit, distinctive title aliases already present in target metadata."""
    aliases: List[str] = []
    seen: set[str] = set()

    def add(value: Any, *, derived: bool = False) -> None:
        text = str(value or "").strip().strip(" \t\r\n\"'“”‘’")
        key = _entity_text_key(text)
        if not key or key in seen:
            return
        words = key.split()
        if derived and len(key) < 6:
            return
        if derived and len(words) < 2 and len(words[0]) < 10:
            return
        if derived and key in {
            "the episode",
            "the issue",
            "the film",
            "the programme",
            "the program",
        }:
            return
        seen.add(key)
        aliases.append(text)

    add(target.entity_id)
    add(target.name)
    name = str(target.name or "").strip()
    dash_parts = re.split(r"\s+[\u2013\u2014-]\s+", name, maxsplit=1)
    if len(dash_parts) == 2:
        main_title = dash_parts[0].strip()
        if len(_entity_text_key(main_title).split()) >= 2:
            add(main_title, derived=True)
            title_words = re.findall(
                r"[A-Za-z]+(?:['\u2019][A-Za-z]+)?",
                main_title,
            )
            initialism = "".join(word[0] for word in title_words).upper()
            if 4 <= len(initialism) <= 12:
                add(initialism)
    if ":" in name:
        add(name.split(":", 1)[1], derived=True)
    # Catalog records frequently shorten a slash-separated release title.  Use
    # only the existing non-trivial slash components as aliases; do not reduce
    # a title to a generic artist or keyword.
    if "/" in name:
        for part in name.split("/"):
            part = part.strip(" \t\r\n-|")
            if len(_entity_text_key(part).split()) >= 2:
                add(part, derived=True)
    # Catalogues often omit a medium qualifier or use a shorter year-only
    # parenthetical title (for example ``St. Elmo (1923)`` for a seed named
    # ``St. Elmo (1923 film)``).  Keep the aliases conservative by deriving
    # only the title before the final parenthetical and a descriptor-stripped
    # form of that parenthetical; never invent unrelated words.
    parenthetical = re.search(r"\s*\(([^()]*)\)\s*$", name)
    if parenthetical:
        base_title = name[: parenthetical.start()].strip()
        details = parenthetical.group(1).strip()
        if base_title:
            add(base_title, derived=True)
            if details:
                add(f"{base_title} ({details})", derived=True)
                stripped_details = re.sub(
                    r"\b(?:film|movie|picture|television|tv|series|novel|book)\b",
                    "",
                    details,
                    flags=re.IGNORECASE,
                )
                stripped_details = re.sub(r"\s+", " ", stripped_details).strip(" ,-")
                if stripped_details and stripped_details != details:
                    add(f"{base_title} ({stripped_details})", derived=True)
    without_year = re.sub(r"\s*\((?:18|19|20)\d{2}\)\s*$", "", name).strip()
    if without_year != name:
        add(without_year, derived=True)
    for pattern in (r'[“"]([^”"]+)[”"]', r"‘([^’]+)’"):
        for match in re.finditer(pattern, name):
            add(match.group(1), derived=True)
    return aliases


def _answer_text_aliases(target: Target) -> List[str]:
    """Return a composite answer and its explicit plural field members."""
    answer = str(target.answer or "").strip()
    if not answer:
        return []
    aliases = [answer]
    field = _entity_text_key(target.answer_field)
    plural_field = field.endswith("s") or any(
        marker in field
        for marker in (
            "co editor",
            "co author",
            "contributors",
            "presenters",
            "mentors",
            "members",
            "people",
            "persons",
            "pair",
        )
    )
    if not plural_field:
        return aliases
    normalized = re.sub(r"\s+(?:and|&)\s+", " | ", answer, flags=re.IGNORECASE)
    normalized = re.sub(r"\s*[;/]\s*", " | ", normalized)
    comma_parts = [part.strip() for part in normalized.split(",")]
    if len(comma_parts) > 1 and all(len(part.split()) >= 2 for part in comma_parts):
        normalized = " | ".join(comma_parts)
    parts = [part.strip(" \t\r\n,;|") for part in normalized.split("|")]
    parts = [part for part in parts if part]
    if len(parts) < 2 or not all(_looks_like_named_answer_member(part) for part in parts):
        return aliases
    aliases.extend(part for part in parts if _entity_text_key(part) != _entity_text_key(answer))
    return list(dict.fromkeys(aliases))


def _looks_like_named_answer_member(value: str) -> bool:
    words = re.findall(r"[A-Za-z][A-Za-z.'\u2019-]*", str(value or ""))
    if not 1 <= len(words) <= 6:
        return False
    particles = {"de", "del", "der", "di", "la", "le", "of", "the", "van", "von"}
    return all(word.casefold() in particles or word[0].isupper() for word in words)


def _find_local_child_clue_leaks(
    children: Sequence[ConstraintPath],
    *,
    final_target: Target,
    ancestor_local_targets: Sequence[Dict[str, str]],
    current_local_target: Target,
    current_local_target_canonical_name: str = "",
) -> List[Dict[str, str]]:
    forbidden = _local_child_clue_forbidden_references(
        final_target,
        ancestor_local_targets,
        current_local_target,
    )
    canonical_name = str(current_local_target_canonical_name or "").strip()
    if canonical_name and not any(
        _entity_text_key(item["value"]) == _entity_text_key(canonical_name)
        for item in forbidden
    ):
        forbidden.append(
            {"kind": "current_local_target", "value": canonical_name}
        )
    leaks: List[Dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for child in children:
        clue = child.clue or ""
        for item in forbidden:
            if _find_entity_text_reference(item["value"], clue):
                key = (child.path_id, _entity_text_key(item["value"]))
                if key not in seen:
                    seen.add(key)
                    leaks.append(
                        {
                            "path_id": child.path_id,
                            "kind": item["kind"],
                            "matched_value": item["value"],
                            "clue": clue,
                        }
                    )
    return leaks


def _local_evidence_mentions_target(
    child: ConstraintPath,
    local_target: Target,
    canonical_name: str = "",
) -> bool:
    """Require private evidence that the selected target satisfies each child clue."""
    aliases = [
        local_target.entity_id,
        local_target.name,
        canonical_name,
    ]
    for evidence in child.evidence:
        evidence_text = " ".join(
            [evidence.url, evidence.text, evidence.supports, evidence.source]
        )
        if any(
            _find_entity_text_reference(alias, evidence_text)
            for alias in aliases
            if str(alias or "").strip()
        ):
            return True
    return False


def _stable_digest(*parts: str) -> bytes:
    raw = "\x1f".join(parts).encode("utf-8", errors="replace")
    return hashlib.sha256(raw).digest()


def _stable_int(*parts: str) -> int:
    return int.from_bytes(_stable_digest(*parts)[:8], byteorder="big", signed=False)


def _stable_unit_float(*parts: str) -> float:
    return _stable_int(*parts) / float(2**64 - 1)


def _stable_sort_key(*parts: str) -> str:
    return _stable_digest(*parts).hex()


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "seed"


def _max_local_depth(path: ConstraintPath, depth: int = 0) -> int:
    if not path.local_constraints:
        return depth
    return max(_max_local_depth(child, depth + 1) for child in path.local_constraints)


def _maximum_minimal_unique_root_bundle(
    verifier: VerifierReport | None,
) -> List[str]:
    """Choose the largest deletion-minimal core bundle that isolates the target."""
    if verifier is None:
        return []
    ordered_ids = [
        path_id
        for path_id in verifier.core_path_ids
        if path_id in verifier.single_path_candidates
    ]
    target = str(verifier.target_key or "")
    if not ordered_ids or not target:
        return []

    candidate_sets = {
        path_id: {str(item) for item in verifier.single_path_candidates[path_id]}
        for path_id in ordered_ids
    }

    def uniquely_identifies(path_ids: Sequence[str]) -> bool:
        if not path_ids:
            return False
        intersection = set.intersection(
            *(candidate_sets[path_id] for path_id in path_ids)
        )
        return intersection == {target}

    minimal_bundles: List[tuple[str, ...]] = []
    for size in range(1, len(ordered_ids) + 1):
        for bundle in combinations(ordered_ids, size):
            if not uniquely_identifies(bundle):
                continue
            if any(
                uniquely_identifies(bundle[:index] + bundle[index + 1 :])
                for index in range(len(bundle))
            ):
                continue
            minimal_bundles.append(bundle)
    if minimal_bundles:
        max_size = max(len(bundle) for bundle in minimal_bundles)
        return list(next(bundle for bundle in minimal_bundles if len(bundle) == max_size))
    if set(verifier.all_core_candidates) == {target}:
        return ordered_ids
    return []


def _root_ids_jointly_unique(
    verifier: VerifierReport | None,
    root_ids: Sequence[str],
) -> bool:
    if verifier is None or not root_ids:
        return False
    target = str(verifier.target_key or "")
    sets = []
    for path_id in dict.fromkeys(str(item) for item in root_ids if str(item)):
        values = verifier.single_path_candidates.get(path_id)
        if not isinstance(values, list):
            return False
        sets.append({str(item) for item in values})
    return bool(sets) and set.intersection(*sets) == {target}


def _local_quality_shortcut_ids(artifact: Artifact) -> set[str]:
    shortcut_ids: set[str] = set()
    for iteration in artifact.iterations:
        if iteration.get("stage") != "local_constraint":
            continue
        report = iteration.get("report")
        if not isinstance(report, dict):
            continue
        for node_report in report.get("reports", []):
            if not isinstance(node_report, dict):
                continue
            quality = node_report.get("local_quality_verifier")
            if not isinstance(quality, dict):
                quality = node_report.get("semantic_verifier")
            if not isinstance(quality, dict):
                continue
            for shortcut in quality.get("single_clue_shortcuts", []):
                if isinstance(shortcut, dict) and str(shortcut.get("path_id") or ""):
                    shortcut_ids.add(str(shortcut["path_id"]))
    return shortcut_ids


def _question_trajectory_shortcut_ids(
    artifact: Artifact,
    failure_report: Dict[str, Any] | None = None,
) -> set[str]:
    """Accumulate child ids that successful solver traces used as shortcuts."""
    shortcut_ids: set[str] = set()
    feedback_items: List[Dict[str, Any]] = []
    for attempt in artifact.solver_attempts:
        feedback = attempt.get("trajectory_feedback")
        if isinstance(feedback, dict):
            feedback_items.append(feedback)
    current_feedback = (failure_report or {}).get("trajectory_feedback")
    if isinstance(current_feedback, dict):
        feedback_items.append(current_feedback)
    for feedback in feedback_items:
        if str(feedback.get("diagnosis") or "") != "shortcut":
            continue
        shortcut_ids.update(
            str(path_id)
            for path_id in feedback.get("shortcut_child_path_ids", [])
            if str(path_id)
        )
    return shortcut_ids


def _deep_clue_nodes(
    path: ConstraintPath,
    *,
    depth: int = 0,
    node_path: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    node = list(node_path or [path.path_id])
    items: List[Dict[str, Any]] = []
    clue = path.clue.strip()
    if clue:
        items.append(
            {
                "path_id": path.path_id,
                "node_path": node,
                "depth": depth,
                "role": path.role,
                "clue": clue,
            }
        )
    for child in path.local_constraints:
        items.extend(
            _deep_clue_nodes(
                child,
                depth=depth + 1,
                node_path=[*node, child.path_id],
            )
        )
    return items


def _deep_leaf_clue_nodes(
    path: ConstraintPath,
    *,
    depth: int = 0,
    node_path: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    node = list(node_path or [path.path_id])
    if not path.local_constraints:
        return [
            {
                "path_id": path.path_id,
                "node_path": node,
                "depth": depth,
                "role": path.role,
                "clue": path.clue,
            }
        ] if path.clue.strip() else []
    leaves: List[Dict[str, Any]] = []
    for child in path.local_constraints:
        leaves.extend(
            _deep_leaf_clue_nodes(
                child,
                depth=depth + 1,
                node_path=[*node, child.path_id],
            )
        )
    return leaves


def _selected_question_path_chain(
    root: ConstraintPath,
    node_path: Sequence[str],
    *,
    semantic_shortcut_ids: set[str] | None = None,
) -> List[Dict[str, Any]]:
    """Return the exact root-to-selected-node chain without sibling subtrees."""
    ids = [str(item) for item in node_path if str(item)]
    if not ids or ids[0] != root.path_id:
        return []

    chain: List[Dict[str, Any]] = []
    shortcut_ids = semantic_shortcut_ids or set()
    current = root
    for depth, path_id in enumerate(ids):
        if current.path_id != path_id:
            return []
        next_id = ids[depth + 1] if depth + 1 < len(ids) else ""
        selected_child = next(
            (
                child
                for child in current.local_constraints
                if child.path_id == next_id
            ),
            None,
        )
        if next_id and selected_child is None:
            return []
        chain.append(
            {
                "depth": depth,
                "path_id": current.path_id,
                "role": current.role,
                "clue": current.clue,
                "local_target": {
                    "entity_id": current.local_target_id,
                    "name": current.local_target_name,
                    "canonical_name": current.local_target_canonical_name,
                    "entity_type": current.local_target_type,
                },
                "selected_child_path_id": selected_child.path_id if selected_child else "",
                "selected_child_clue": selected_child.clue if selected_child else "",
                "child_clues": [
                    {
                        "path_id": child.path_id,
                        "role": child.role,
                        "branch": child.branch,
                        "clue": child.clue,
                        **(
                            {"semantic_shortcut": True}
                            if child.path_id in shortcut_ids
                            else {}
                        ),
                    }
                    for child in current.local_constraints
                ],
            }
        )
        if selected_child is not None:
            current = selected_child
    return chain


def _neutral_entity_label(entity_type: str) -> str:
    """Return a type-only label for a private construction-time entity."""
    value = str(entity_type or "").strip().casefold()
    if any(token in value for token in ("person", "human", "director", "actor")):
        return "a person"
    if any(token in value for token in ("place", "location", "city", "country", "venue")):
        return "a place"
    if any(token in value for token in ("organization", "organisation", "company", "institution", "studio")):
        return "an organization"
    if any(token in value for token in ("film", "book", "game", "record", "work", "publication", "song")):
        return "a work"
    if any(token in value for token in ("object", "artifact", "artefact", "building")):
        return "an object"
    return "an entity"


def _atomic_clue_options(clue: str) -> List[str]:
    """Split a possibly compound clue into short, faithful predicate options.

    This is deliberately syntactic.  The Question Agent still decides which
    option is factually useful, while the payload makes it harder to copy a
    compound numeric/relational fingerprint as one child fact.
    """
    text = re.sub(r"\s+", " ", str(clue or "").strip())
    if not text:
        return []
    parts = re.split(
        r"(?:\s*;\s*|(?<=[.!?])\s+|,\s+|\s+(?=(?:and|but|while)\s+[A-Za-z]))",
        text,
        flags=re.IGNORECASE,
    )
    options: List[str] = []
    for part in parts:
        value = re.sub(r"\s+", " ", part.strip(" ,.;:"))
        # Keep a conjunction fragment only when it still carries a predicate;
        # this prevents tiny conjunction remnants from becoming fake facts.
        value = re.sub(r"^(?:and|but|while)\s+", "", value, flags=re.IGNORECASE)
        if len(value) < 8:
            continue
        if value.casefold() not in {item.casefold() for item in options}:
            options.append(value)
    if not options:
        options = [text]
    return options[:4]


def _clue_risk_flags(clue: str) -> List[str]:
    """Mark clue features that commonly create semantic one-query shortcuts."""
    text = str(clue or "")
    lowered = text.casefold()
    flags: List[str] = []
    if re.search(r"\b\d+(?:\.\d+)?\b", text):
        flags.append("numeric_detail")
    if any(
        marker in lowered
        for marker in (
            "more than any", "than any other", "most ", "only ", "the first",
            "the sole", "largest", "longest", "highest", "best ", "exactly ",
        )
    ):
        flags.append("superlative_or_exclusive")
    if any(marker in lowered for marker in ("formerly", "landfill", "peninsula", "island", "donated", "million")):
        flags.append("definition_like")
    if len(re.findall(r"\b\w+\b", text)) >= 24:
        flags.append("compound")
    return flags


def _question_model_constraint_view(
    question_view: Dict[str, Any],
) -> Dict[str, Any]:
    """Expose only the selected tree chain needed by Question and Repair."""
    chain = question_view.get("selected_path_chain")
    if not isinstance(chain, list):
        chain = []
    sanitized_chain: List[Dict[str, Any]] = []
    for raw_node in chain:
        if not isinstance(raw_node, dict):
            continue
        node = dict(raw_node)
        local_target = node.get("local_target")
        if isinstance(local_target, dict):
            entity_type = str(local_target.get("entity_type") or "")
            # The model may use the private name to decide what to mask, but it
            # must never copy this field into public question prose.
            node["local_target"] = {
                "entity_id": str(local_target.get("entity_id") or ""),
                "name": str(local_target.get("name") or ""),
                "canonical_name": str(local_target.get("canonical_name") or ""),
                "entity_type": entity_type,
                "neutral_label": _neutral_entity_label(entity_type),
                "visibility": "private_planning_only",
                "replacement_strategies": [
                    {
                        "strategy": "neutral_type_reference",
                        "rule": "use only the neutral_label with pronouns",
                    },
                    {
                        "strategy": "child_fact_reference",
                        "rule": "use one low-risk atomic child fact; redact names and defining details",
                    },
                ],
            }
        children: List[Dict[str, Any]] = []
        for raw_child in node.get("child_clues", []):
            if not isinstance(raw_child, dict):
                continue
            child = dict(raw_child)
            child["atomic_fact_options"] = _atomic_clue_options(
                str(child.get("clue") or "")
            )
            child["risk_flags"] = _clue_risk_flags(str(child.get("clue") or ""))
            child["redact_named_spans"] = _relation_named_entity_spans(
                str(child.get("clue") or "")
            )
            flags = set(child["risk_flags"])
            if "definition_like" in flags:
                child["safe_abstraction_rule"] = (
                    "retain only the broad relation/attribute category; omit named objects, "
                    "mechanism, former/current state, dates, measurements, and geography"
                )
            elif "superlative_or_exclusive" in flags:
                child["safe_abstraction_rule"] = (
                    "retain only a non-ranked repeated/general relation; omit exact count and rank"
                )
            elif "numeric_detail" in flags:
                child["safe_abstraction_rule"] = (
                    "omit exact values or keep only one broad era/scale when needed"
                )
            else:
                child["safe_abstraction_rule"] = (
                    "use one atomic predicate and redact every named span"
                )
            children.append(child)
        node["child_clues"] = children
        sanitized_chain.append(node)
    result = {
        "path_id": str(question_view.get("path_id") or ""),
        "role": str(question_view.get("role") or ""),
        "selected_path_chain": sanitized_chain,
    }
    if not sanitized_chain or (
        len(sanitized_chain) == 1 and not sanitized_chain[0].get("child_clues")
    ):
        result["root_clue"] = str(question_view.get("root_clue") or "")
        result["root_clue_use"] = "required_root_evidence"
    return result


def _question_view_has_expanded_chain(question_view: Dict[str, Any]) -> bool:
    """Return true only when a root view contains at least one real child edge."""
    chain = question_view.get("selected_path_chain")
    if not isinstance(chain, list) or len(chain) < 2:
        return False
    return any(
        isinstance(node, dict) and bool(node.get("child_clues"))
        for node in chain
    )


def _detect_early_memory_shortcuts(
    rollouts: Sequence[Dict[str, Any]],
    selected_views: Sequence[Dict[str, Any]],
    *,
    question: str,
) -> List[Dict[str, Any]]:
    """Find correct rollouts naming a private intermediate before web evidence."""
    observations: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    public_question = str(question or "")
    for rollout in rollouts:
        if rollout.get("correct") is not True:
            continue
        summaries = rollout.get("reasoning_summaries")
        if not isinstance(summaries, list) or not summaries:
            continue
        first_reasoning = str(summaries[0] or "")
        if not first_reasoning:
            continue
        rollout_id = str(rollout.get("rollout_id") or "")
        for root_view in selected_views:
            root_id = str(root_view.get("path_id") or "")
            for node in root_view.get("selected_path_chain", []):
                if not isinstance(node, dict):
                    continue
                node_id = str(node.get("path_id") or "")
                local_target = node.get("local_target")
                if not isinstance(local_target, dict):
                    continue
                names = [
                    str(local_target.get(key) or "").strip()
                    for key in ("name", "canonical_name", "entity_id")
                ]
                names = [name for name in names if len(_entity_text_key(name)) >= 4]
                matched = next(
                    (
                        name
                        for name in names
                        if not _find_entity_text_reference(name, public_question)
                        and _find_entity_text_reference(name, first_reasoning)
                    ),
                    "",
                )
                if not matched:
                    continue
                child_ids = [
                    str(child.get("path_id") or "")
                    for child in node.get("child_clues", [])
                    if isinstance(child, dict) and str(child.get("path_id") or "")
                ]
                key = (rollout_id, root_id, node_id)
                if key in seen:
                    continue
                seen.add(key)
                observations.append(
                    {
                        "rollout_id": rollout_id,
                        "root_path_id": root_id,
                        "node_path_id": node_id,
                        "entity_type": str(local_target.get("entity_type") or ""),
                        "matched_entity": matched,
                        "child_path_ids": child_ids,
                        "reason": (
                            f"correct rollout {rollout_id or 'unknown'} named "
                            f"private intermediate {matched!r} before its first "
                            "successful external result"
                        ),
                    }
                )
    return observations


def _blind_candidate_matches_target(
    target: Target,
    candidate: Dict[str, Any],
) -> bool:
    candidate_key = _entity_text_key(candidate.get("name"))
    if not candidate_key:
        return False
    alias_keys = {
        _entity_text_key(alias)
        for alias in _target_text_aliases(target)
        if _entity_text_key(alias)
    }
    for alias in alias_keys:
        if candidate_key == alias or (
            min(len(candidate_key), len(alias)) >= 12
            and (candidate_key in alias or alias in candidate_key)
        ):
            return True
        alias_tokens = set(alias.split())
        candidate_tokens = set(candidate_key.split())
        common = alias_tokens & candidate_tokens
        alias_numbers = {token for token in alias_tokens if token.isdigit()}
        candidate_numbers = {
            token for token in candidate_tokens if token.isdigit()
        }
        numeric_identity_consistent = (
            not alias_numbers
            or not candidate_numbers
            or alias_numbers <= candidate_numbers
            or candidate_numbers <= alias_numbers
        )
        if (
            numeric_identity_consistent
            and len(common) >= 3
            and len(common) / max(1, len(alias_tokens)) >= 0.75
            and len(common) / max(1, len(candidate_tokens)) >= 0.4
        ):
            return True
    return False


def _blind_resolution_concern(
    target: Target,
    report: Dict[str, Any],
) -> bool:
    resolution = str(report.get("resolution") or "").strip().lower()
    if resolution not in {"resolved", "multiple"}:
        return False
    candidates = [
        item
        for item in report.get("candidate_targets", [])
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    ]
    if resolution == "multiple":
        return any(
            not _blind_candidate_matches_target(target, candidate)
            for candidate in candidates
        )
    return not any(
        _blind_candidate_matches_target(target, candidate)
        for candidate in candidates
    )


def _blind_resolution_matches_expected(
    target: Target,
    report: Dict[str, Any],
) -> bool:
    if str(report.get("resolution") or "").strip().lower() != "resolved":
        return False
    return any(
        isinstance(candidate, dict)
        and candidate.get("clue_checks")
        and candidate.get("source_urls")
        and _blind_candidate_matches_target(target, candidate)
        for candidate in report.get("candidate_targets", [])
    )


def _normalize_uniqueness_response(
    report: Any,
    *,
    required_clause_ids: Sequence[str] = (),
) -> Dict[str, Any]:
    report = report if isinstance(report, dict) else {}
    unique_value = report.get("unique")
    if isinstance(unique_value, str):
        lowered = unique_value.strip().lower()
        if lowered in {"true", "yes", "unique"}:
            unique_value = True
        elif lowered in {"false", "no", "not_unique"}:
            unique_value = False
        else:
            unique_value = None
    elif not isinstance(unique_value, bool):
        unique_value = None

    key = str(report.get("key", "")).strip().lower()
    if unique_value is True:
        key = "unique"
    elif unique_value is False:
        key = "not_unique"
    elif key not in {"uncertain", "check_failed"}:
        key = "uncertain"
    alternatives = report.get("alternatives", [])
    alternatives = alternatives if isinstance(alternatives, list) else []
    verified_alternatives = [
        item
        for item in alternatives
        if isinstance(item, dict)
        and item.get("matches_all_clues") is True
        and item.get("would_change_answer") is True
        and str(item.get("answer") or "").strip()
        and isinstance(item.get("clue_checks"), list)
        and item.get("clue_checks")
        and _clue_checks_cover_clauses(
            item.get("clue_checks"), required_clause_ids
        )
        and isinstance(item.get("source_urls"), list)
        and any(str(url).strip() for url in item.get("source_urls", []))
    ]
    explicit_unsupported = report.get("unsupported_alternatives", [])
    explicit_unsupported = (
        explicit_unsupported if isinstance(explicit_unsupported, list) else []
    )
    unsupported_alternatives = [
        item for item in alternatives if item not in verified_alternatives
    ] + [
        item
        for item in explicit_unsupported
        if isinstance(item, dict)
        and str(item.get("name") or "").strip()
        and isinstance(item.get("failed_clues"), list)
        and item.get("failed_clues")
        and isinstance(item.get("source_urls"), list)
        and any(str(url).strip() for url in item.get("source_urls", []))
    ]
    reason = str(report.get("reason", ""))
    if unique_value is False and not verified_alternatives:
        unique_value = None
        key = "uncertain"
        reason = (
            "No complete sourced alternative passed every public clue. " + reason
        ).strip()
    return {
        "enabled": True,
        "unique": unique_value,
        "key": key,
        "reason": reason,
        "alternatives": verified_alternatives,
        "unsupported_alternatives": unsupported_alternatives,
        "missing_disambiguator": str(report.get("missing_disambiguator", "")),
    }


def _shortcut_text_overlap(clue: str, question: str) -> float:
    stopwords = {
        "a", "an", "and", "as", "at", "by", "for", "from", "he", "her",
        "his", "in", "is", "it", "its", "of", "on", "she", "that", "the",
        "this", "to", "was", "were", "with", "film", "work", "person",
    }

    def tokens(value: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[\w]+", str(value or "").casefold())
            if len(token) > 1 and token not in stopwords
        }

    clue_tokens = tokens(clue)
    if not clue_tokens:
        return 0.0
    return len(clue_tokens & tokens(question)) / len(clue_tokens)


def _number_values(text: str) -> set[int]:
    """Extract digit and common English cardinal/ordinal values."""
    normalized = re.sub(r"[-\u2011]", " ", str(text or "").casefold())
    values = {int(item) for item in re.findall(r"\b\d+\b", normalized)}
    units = {
        "zero": 0, "one": 1, "first": 1, "two": 2, "second": 2,
        "three": 3, "third": 3, "four": 4, "fourth": 4, "five": 5,
        "fifth": 5, "six": 6, "sixth": 6, "seven": 7, "seventh": 7,
        "eight": 8, "eighth": 8, "nine": 9, "ninth": 9, "ten": 10,
        "tenth": 10, "eleven": 11, "eleventh": 11, "twelve": 12,
        "twelfth": 12, "thirteen": 13, "thirteenth": 13,
        "fourteen": 14, "fourteenth": 14, "fifteen": 15,
        "fifteenth": 15, "sixteen": 16, "sixteenth": 16,
        "seventeen": 17, "seventeenth": 17, "eighteen": 18,
        "eighteenth": 18, "nineteen": 19, "nineteenth": 19,
    }
    tens = {
        "twenty": 20, "twentieth": 20, "thirty": 30, "thirtieth": 30,
        "forty": 40, "fortieth": 40, "fifty": 50, "fiftieth": 50,
        "sixty": 60, "sixtieth": 60, "seventy": 70, "seventieth": 70,
        "eighty": 80, "eightieth": 80, "ninety": 90, "ninetieth": 90,
    }
    tokens = re.findall(r"\b[a-z]+\b", normalized)
    for index, token in enumerate(tokens):
        if token in units:
            values.add(units[token])
        if token in tens:
            value = tens[token]
            if index + 1 < len(tokens) and tokens[index + 1] in units:
                value += units[tokens[index + 1]]
            values.add(value)
    return values


def _relation_named_entity_spans(clue: str) -> List[str]:
    """Extract proper-name spans that a relation rewrite must redact.

    Relation children describe one named related entity.  A high token-overlap
    score can miss short names (for example ``Fred Niblo``), so check explicit
    capitalized spans independently.  This intentionally ignores sentence
    articles/pronouns and one-word generic labels; it does not attempt broad
    NER or reject ordinary descriptive adjectives.
    """
    token = r"[A-Z](?:[A-Za-z0-9.'\u2019\u2011-]*[A-Za-z0-9])?"
    spans = re.findall(rf"\b{token}(?:\s+{token})*", str(clue or ""))
    ignored = {
        "A", "An", "And", "As", "At", "By", "For", "From", "He", "Her",
        "His", "In", "It", "Its", "One", "She", "The", "Their", "This",
        "To", "Was", "Were", "With", "Film", "Films", "Series", "Project",
        "Database", "Company", "University", "American", "British",
        "English", "Australian", "German", "French", "Silent", "Broadway",
        "Academy", "National", "International", "European", "Modern",
        "Took", "Founded", "Directed", "Wrote", "Edited", "Served", "Became",
        "Made", "Had", "Appeared", "Played", "Died", "Born", "Held", "Gave",
        "Created", "Published", "Produced", "Launched", "Contains", "Includes",
        "Belongs", "Records", "Lists", "Carries", "Uses", "Developed", "Opened",
        "Closed", "Won", "Known", "Based", "Located", "Released", "Filmed",
        "Year", "Person", "Month", "Date", "Number", "Volume", "Page", "Issue",
        "Title", "Book", "Award", "Director", "Editor", "Writer", "Magazine",
        "Studio", "Studios", "Pictures", "Entertainment", "Media", "Group",
        "Holdings", "Corporation", "Organization", "Foundation", "Association",
        "Programme", "Program", "Arts", "Adapted", "Hollywood", "Best", "Actress", "First",
        "Atlantic", "Pacific", "Ocean", "Coast", "Ferris",
        "January", "February", "March", "April", "May", "June", "July", "August",
        "September", "October", "November", "December",
    }
    names: List[str] = []
    seen: set[str] = set()
    for raw in spans:
        phrase = raw.strip(" ,.;:()[]{}\"'\u201c\u201d")
        words = phrase.split()
        while words and words[0] in {"The", "A", "An"}:
            words = words[1:]
        if not words:
            continue
        if len(words) == 1:
            word = words[0]
            # Hyphenated demonyms/adjectives (e.g. German-language or
            # Australian-born) are descriptors, not named related entities.
            if "-" in word or word in ignored or len(word) < 3:
                continue
        phrase = " ".join(words)
        key = _entity_text_key(phrase)
        if not key or key in seen:
            continue
        # These are broad award labels, not named related entities.
        if key in {"academy award", "best actress"}:
            continue
        seen.add(key)
        names.append(phrase)
    return names


def _relation_named_entity_leaks(clue: str, question: str) -> List[str]:
    """Find relation proper-name spans copied into public question text."""
    return [
        phrase
        for phrase in _relation_named_entity_spans(clue)
        if _find_entity_text_reference(phrase, question)
    ]


def _semantic_shortcut_specificity_leaks(clue: str, question: str) -> List[str]:
    """Detect retained high-signal predicates after a shortcut child is fuzzified."""
    clue_text = str(clue or "")
    question_text = str(question or "")
    if not clue_text or not question_text:
        return []
    markers = [
        phrase
        for phrase in (
            "more than any",
            "more than all",
            "than any other",
            "most films",
            "most widely",
            "only one",
            "the only",
            "the first",
            "the sole",
            "uniquely",
            "unique",
            "exactly",
            "highest",
            "largest",
            "longest",
            "best director",
            "academy award",
        )
        if phrase in clue_text.casefold()
        and phrase in question_text.casefold()
    ]
    clue_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", clue_text))
    question_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", question_text))
    markers.extend(sorted(clue_numbers & question_numbers))
    number_words = {
        "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
        "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
        "thousand",
    }
    clue_number_words = set(re.findall(r"\b[a-z]+\b", clue_text.casefold())) & number_words
    question_number_words = set(re.findall(r"\b[a-z]+\b", question_text.casefold())) & number_words
    markers.extend(sorted(clue_number_words & question_number_words))
    return list(dict.fromkeys(markers))


def _question_repair_mode(failure_report: Dict[str, Any] | None) -> str:
    stage = str((failure_report or {}).get("stage") or "").strip().lower()
    if stage == "uniqueness":
        return "uniqueness_supplement"
    if stage == "solver":
        objective = str((failure_report or {}).get("objective") or "").strip().lower()
        if objective == "repair_ambiguity":
            return "uniqueness_supplement"
        if objective == "restore_solvability":
            return "solvability_restore"
        return "shortcut_prune"
    return "structural_repair"


def _question_verifier_soft_quality_warning(reason: str) -> bool:
    """Allow solver-driven repair to handle difficulty-only verifier feedback."""
    text = str(reason or "").casefold()
    quality_markers = (
        "over-specific",
        "over specific",
        "precise filter",
        "direct search",
        "one-query",
        "one query",
        "high-signal",
        "high signal",
        "shortcut",
    )
    hard_markers = (
        "leak",
        "intermediate",
        "semantic alias",
        "defining",
        "one obvious",
        "atomicity",
        "answer type",
        "different field",
        "target name",
        "final answer",
        "unknown child",
        "unknown root",
        "chain_node_usage",
        "not natural",
        "not answerable",
        "distractor",
    )
    return any(marker in text for marker in quality_markers) and not any(
        marker in text for marker in hard_markers
    )


def _participating_chain_node_details(
    raw_usage: Sequence[Dict[str, Any]],
    selected_root_constraints: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Resolve reported child ids to the exact node and child clue text."""
    node_index: Dict[tuple[str, str, int], Dict[str, Any]] = {}
    root_roles: Dict[str, str] = {}
    for root in selected_root_constraints:
        root_id = str(root.get("path_id") or "")
        if not root_id:
            continue
        root_roles[root_id] = str(root.get("role") or "")
        for node in root.get("selected_path_chain", []):
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("path_id") or "")
            try:
                depth = int(node.get("depth") or 0)
            except (TypeError, ValueError):
                continue
            if node_id:
                node_index[(root_id, node_id, depth)] = node

    details: List[Dict[str, Any]] = []
    for usage in raw_usage:
        if not isinstance(usage, dict):
            continue
        root_id = str(usage.get("root_path_id") or "")
        node_id = str(usage.get("node_path_id") or "")
        try:
            depth = int(usage.get("node_depth"))
        except (TypeError, ValueError):
            continue
        node = node_index.get((root_id, node_id, depth))
        if node is None:
            continue
        used_ids = {
            str(item)
            for item in usage.get("used_child_path_ids", [])
            if str(item)
        }
        used_children = [
            {
                "path_id": str(child.get("path_id") or ""),
                "role": str(child.get("role") or ""),
                "clue": str(child.get("clue") or ""),
                "semantic_shortcut": bool(child.get("semantic_shortcut")),
            }
            for child in node.get("child_clues", [])
            if isinstance(child, dict)
            and str(child.get("path_id") or "") in used_ids
        ]
        available_children = [
            {
                "path_id": str(child.get("path_id") or ""),
                "role": str(child.get("role") or ""),
                "clue": str(child.get("clue") or ""),
                "semantic_shortcut": bool(child.get("semantic_shortcut")),
                "used": str(child.get("path_id") or "") in used_ids,
            }
            for child in node.get("child_clues", [])
            if isinstance(child, dict) and str(child.get("path_id") or "")
        ]
        if not used_children:
            continue
        details.append(
            {
                "root_path_id": root_id,
                "root_role": root_roles.get(root_id, ""),
                "node_path_id": node_id,
                "node_depth": depth,
                "node_role": str(node.get("role") or ""),
                "node_clue": str(node.get("clue") or ""),
                "local_target": dict(node.get("local_target") or {}),
                "used_child_clues": used_children,
                "available_child_clues": available_children,
            }
        )
    return details


def _question_selection_diff(
    previous: Dict[str, Any],
    repaired: Dict[str, Any],
    *,
    question_changed: bool,
) -> List[Dict[str, Any]]:
    """Compute an auditable structural diff from old/new model selections."""

    def root_ids(selection: Dict[str, Any]) -> set[str]:
        return {
            str(item)
            for item in [
                *(
                    selection.get("used_core_path_ids", [])
                    if isinstance(selection.get("used_core_path_ids"), list)
                    else []
                ),
                *(
                    selection.get("used_distractor_path_ids", [])
                    if isinstance(selection.get("used_distractor_path_ids"), list)
                    else []
                ),
            ]
            if str(item)
        }

    def usage_map(selection: Dict[str, Any]) -> Dict[tuple[str, str, int], set[str]]:
        result: Dict[tuple[str, str, int], set[str]] = {}
        raw = selection.get("chain_node_usage")
        for item in raw if isinstance(raw, list) else []:
            if not isinstance(item, dict):
                continue
            root_id = str(item.get("root_path_id") or "")
            node_id = str(item.get("node_path_id") or "")
            try:
                depth = int(item.get("node_depth"))
            except (TypeError, ValueError):
                continue
            if not root_id or not node_id:
                continue
            children = item.get("used_child_path_ids")
            result[(root_id, node_id, depth)] = {
                str(child_id)
                for child_id in children if str(child_id)
            } if isinstance(children, list) else set()
        return result

    previous_roots = root_ids(previous)
    repaired_roots = root_ids(repaired)
    previous_usage = usage_map(previous)
    repaired_usage = usage_map(repaired)
    actions: List[Dict[str, Any]] = []

    for root_id in sorted(previous_roots - repaired_roots):
        actions.append(
            {
                "action": "remove_root",
                "root_path_id": root_id,
                "node_path_id": "",
                "node_depth": None,
                "child_path_ids": [],
                "source": "program_diff",
            }
        )
    for root_id in sorted(repaired_roots - previous_roots):
        actions.append(
            {
                "action": "add_root",
                "root_path_id": root_id,
                "node_path_id": "",
                "node_depth": None,
                "child_path_ids": [],
                "source": "program_diff",
            }
        )

    common_roots = previous_roots & repaired_roots
    node_keys = sorted(
        {
            key
            for key in [*previous_usage, *repaired_usage]
            if key[0] in common_roots
        },
        key=lambda item: (item[0], item[2], item[1]),
    )
    for root_id, node_id, depth in node_keys:
        removed = sorted(
            previous_usage.get((root_id, node_id, depth), set())
            - repaired_usage.get((root_id, node_id, depth), set())
        )
        added = sorted(
            repaired_usage.get((root_id, node_id, depth), set())
            - previous_usage.get((root_id, node_id, depth), set())
        )
        if removed:
            actions.append(
                {
                    "action": "remove_child_clue",
                    "root_path_id": root_id,
                    "node_path_id": node_id,
                    "node_depth": depth,
                    "child_path_ids": removed,
                    "source": "program_diff",
                }
            )
        if added:
            actions.append(
                {
                    "action": "add_child_clue",
                    "root_path_id": root_id,
                    "node_path_id": node_id,
                    "node_depth": depth,
                    "child_path_ids": added,
                    "source": "program_diff",
                }
            )

    if not actions:
        actions.append(
            {
                "action": (
                    "rewrite_same_selection" if question_changed else "no_change"
                ),
                "root_path_id": "",
                "node_path_id": "",
                "node_depth": None,
                "child_path_ids": [],
                "source": "program_diff",
            }
        )
    return actions


def _validate_chain_node_usage(
    raw_usage: Any,
    constraint_payloads: Sequence[Dict[str, Any]],
    *,
    used_root_path_ids: set[str],
    min_root_core_children: int = 1,
    min_root_non_shortcut_children: int = 0,
) -> Dict[str, Any]:
    expected: Dict[tuple[str, str, int], set[str]] = {}
    expected_core: Dict[tuple[str, str, int], set[str]] = {}
    expected_non_shortcut_core: Dict[tuple[str, str, int], set[str]] = {}
    expected_selected_child: Dict[tuple[str, str, int], str] = {}
    ordered_nodes: List[tuple[str, str, int]] = []
    for constraint in constraint_payloads:
        root_id = str(constraint.get("path_id") or "")
        chain = constraint.get("selected_path_chain")
        if (
            not root_id
            or root_id not in used_root_path_ids
            or not isinstance(chain, list)
            or not chain
        ):
            continue
        for node in chain:
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("path_id") or "")
            if not node_id:
                continue
            depth = int(node.get("depth") or 0)
            key = (root_id, node_id, depth)
            child_ids = {
                str(child.get("path_id") or "")
                for child in node.get("child_clues", [])
                if isinstance(child, dict) and str(child.get("path_id") or "")
            }
            core_child_ids = {
                str(child.get("path_id") or "")
                for child in node.get("child_clues", [])
                if isinstance(child, dict)
                and str(child.get("role") or "") == "core"
                and str(child.get("path_id") or "")
            }
            non_shortcut_core_child_ids = {
                str(child.get("path_id") or "")
                for child in node.get("child_clues", [])
                if isinstance(child, dict)
                and str(child.get("role") or "") == "core"
                and not child.get("semantic_shortcut")
                and str(child.get("path_id") or "")
            }
            expected[key] = child_ids
            expected_core[key] = core_child_ids
            expected_non_shortcut_core[key] = non_shortcut_core_child_ids
            expected_selected_child[key] = str(
                node.get("selected_child_path_id") or ""
            )
            ordered_nodes.append(key)

    if not expected:
        if isinstance(raw_usage, list) and raw_usage:
            return {
                "accepted": False,
                "reason": "chain_node_usage reports roots that were not used",
            }
        return {
            "accepted": True,
            "reason": "no used root has a selected path chain requiring usage reporting",
            "normalized_usage": [],
        }
    if not isinstance(raw_usage, list):
        return {
            "accepted": False,
            "reason": "chain_node_usage must be a list covering every selected chain node",
        }

    reported: Dict[tuple[str, str, int], List[str]] = {}
    for item in raw_usage:
        if not isinstance(item, dict):
            return {
                "accepted": False,
                "reason": "every chain_node_usage entry must be an object",
            }
        root_id = str(item.get("root_path_id") or "")
        node_id = str(item.get("node_path_id") or "")
        try:
            node_depth = int(item.get("node_depth"))
        except (TypeError, ValueError):
            return {
                "accepted": False,
                "reason": (
                    "node_depth must be an integer for "
                    f"root={root_id!r} node={node_id!r}"
                ),
            }
        key = (root_id, node_id, node_depth)
        if key not in expected:
            return {
                "accepted": False,
                "reason": (
                    "chain_node_usage references an unknown chain node: "
                    f"root={root_id!r} node={node_id!r} depth={node_depth}"
                ),
            }
        if key in reported:
            return {
                "accepted": False,
                "reason": (
                    "chain_node_usage contains a duplicate chain node: "
                    f"root={root_id!r} node={node_id!r} depth={node_depth}"
                ),
            }
        used_children = item.get("used_child_path_ids")
        if not isinstance(used_children, list):
            return {
                "accepted": False,
                "reason": (
                    "used_child_path_ids must be a list for "
                    f"root={root_id!r} node={node_id!r} depth={node_depth}"
                ),
            }
        normalized = list(
            dict.fromkeys(str(child_id) for child_id in used_children if str(child_id))
        )
        invalid = [child_id for child_id in normalized if child_id not in expected[key]]
        if invalid:
            return {
                "accepted": False,
                "reason": (
                    "chain_node_usage references child ids outside the node: "
                    f"root={root_id!r} node={node_id!r} depth={node_depth} "
                    f"invalid={invalid}"
                ),
            }
        reported[key] = normalized

    required_root_children = max(1, int(min_root_core_children or 1))
    for key, core_child_ids in expected_core.items():
        if key[2] != 0 or not core_child_ids:
            continue
        required = min(required_root_children, len(core_child_ids))
        actual = len(set(reported.get(key, [])) & core_child_ids)
        if actual < required:
            if actual == 0:
                reason = f"used core chain selected no child clues: root={key[0]!r}"
            else:
                reason = (
                    "expanded root replacement uses too few complementary core children: "
                    f"root={key[0]!r} required={required} actual={actual}"
                )
            return {
                "accepted": False,
                "reason": reason,
            }

    missing = [key for key in ordered_nodes if key not in reported]
    if missing:
        for key in missing:
            reported[key] = []
    for key, selected_child_id in expected_selected_child.items():
        if selected_child_id and selected_child_id not in reported.get(key, []):
            return {
                "accepted": False,
                "reason": (
                    "chain node omitted its selected continuation child: "
                    f"root={key[0]!r} node={key[1]!r} depth={key[2]} "
                    f"child={selected_child_id!r}"
                ),
            }
    required_non_shortcut = max(0, int(min_root_non_shortcut_children or 0))
    if required_non_shortcut:
        for key, safe_ids in expected_non_shortcut_core.items():
            if key[2] != 0 or not safe_ids or key[0] not in used_root_path_ids:
                continue
            actual = len(set(reported.get(key, [])) & safe_ids)
            if actual < min(required_non_shortcut, len(safe_ids)):
                return {
                    "accepted": False,
                    "reason": (
                        "expanded root selected too few non-shortcut depth-0 children: "
                        f"root={key[0]!r} required={min(required_non_shortcut, len(safe_ids))} actual={actual}"
                    ),
                }
    for root_id in used_root_path_ids:
        root_entries = [
            used_children
            for (entry_root, _, _), used_children in reported.items()
            if entry_root == root_id
        ]
        root_has_available_children = any(
            child_ids
            for (entry_root, _, _), child_ids in expected.items()
            if entry_root == root_id
        )
        if root_has_available_children and root_entries and not any(root_entries):
            return {
                "accepted": False,
                "reason": (
                    "used core chain selected no child clues: "
                    f"root={root_id!r}"
                ),
            }

    normalized_usage = [
        {
            "root_path_id": root_id,
            "node_path_id": node_id,
            "node_depth": node_depth,
            "used_child_path_ids": reported[(root_id, node_id, node_depth)],
        }
        for root_id, node_id, node_depth in ordered_nodes
    ]
    return {
        "accepted": True,
        "reason": (
            "chain-node child-clue usage references valid ids; omitted nodes were "
            "normalized to empty usage"
            if missing
            else "chain-node child-clue usage is complete and references valid ids"
        ),
        "normalized_usage": normalized_usage,
        "normalized_missing_nodes": [
            {"root_path_id": root, "node_path_id": node, "node_depth": depth}
            for root, node, depth in missing
        ],
    }


def _as_list_of_dicts(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _repair_extra_fields(response: Dict[str, Any]) -> Dict[str, Any]:
    standard_keys = {
        "question",
        "answer",
        "used_core_path_ids",
        "used_distractor_path_ids",
        "chain_node_usage",
        "repair_actions",
        "added_clues",
        "supporting_evidence",
        "note",
        "repair",
        "validation",
    }
    return {key: value for key, value in response.items() if key not in standard_keys}


def _is_transient_agent_error(error_text: str) -> bool:
    value = error_text.lower()
    return any(
        marker in value
        for marker in (
            "429",
            "rate limit",
            "tpm",
            "timeout",
            "timed out",
            "temporarily",
            "503",
            "502",
            "500",
            "504",
        )
    )


def _constraint_revision_view(path: ConstraintPath) -> Dict[str, Any]:
    """Return only model-owned root fields, never the recursively expanded tree."""
    return {
        "path_id": path.path_id,
        "role": path.role,
        "branch": path.branch,
        "hop_count": path.hop_count,
        "clue": path.clue,
        "candidates": list(path.candidates),
        "estimated_candidate_count": path.estimated_candidate_count,
        "terminal_type": path.terminal_type,
        "evidence": [asdict(item) for item in path.evidence],
        "notes": path.notes,
    }


def _root_target_view(target: Target) -> Dict[str, Any]:
    """Expose what Root generation needs while keeping the final field private."""
    return {
        "entity_id": target.entity_id,
        "name": target.name,
        "entity_type": target.entity_type,
        "forbidden_answer_field": target.answer_field,
        "source_urls": list(target.source_urls),
    }


def _root_evidence_mentions_target(
    path: ConstraintPath,
    target: Target,
) -> bool:
    """Recognize target membership without requiring one exact display string.

    Evidence pages commonly use a record number, a shortened title, ``v``/``vs``
    variants, or a different date format. These forms are accepted only when they
    are already grounded in the seed metadata; generic words alone never suffice.
    """
    source_urls = {
        _source_url_key(str(url))
        for url in target.source_urls
        if _source_url_key(str(url))
    }
    aliases = _root_target_membership_aliases(target)
    identifiers = _root_target_membership_identifiers(target)
    strong_tokens = _root_target_strong_tokens(target)
    for evidence in path.evidence:
        evidence_url = _source_url_key(evidence.url)
        if evidence_url and evidence_url in source_urls:
            return True
        evidence_text = " ".join(
            [evidence.url, evidence.text, evidence.supports, evidence.source]
        )
        if any(
            _find_entity_text_reference(alias, evidence_text)
            for alias in aliases
            if str(alias or "").strip()
        ):
            return True
        if any(
            _find_entity_text_reference(identifier, evidence_text)
            for identifier in identifiers
            if identifier
        ):
            return True
        normalized_evidence = _entity_text_key(evidence_text)
        strong_hits = {
            token
            for token in strong_tokens
            if f" {token} " in f" {normalized_evidence} "
        }
        if len(strong_tokens) >= 2 and len(strong_hits) >= 2:
            return True
    return False


def _root_target_membership_aliases(target: Target) -> List[str]:
    """Build conservative display-name variants from seed metadata."""
    aliases = list(_target_text_aliases(target))
    name = str(target.name or "").strip()
    normalized_name = _entity_text_key(name)
    if normalized_name:
        aliases.append(name.replace("The ", "", 1) if name.startswith("The ") else name)
    # Match common event separators and date orderings used by sports/catalogue
    # pages without treating a generic title fragment as a target identifier.
    date_match = re.search(
        r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+(\d{4})\b",
        name,
        flags=re.IGNORECASE,
    )
    if date_match:
        day, month_name, year = date_match.groups()
        month = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12,
        }[month_name.casefold()]
        prefix = name[: date_match.start()].strip(" ,;:-")
        date_forms = [
            f"{year}-{month:02d}-{int(day):02d}",
            f"{month_name} {int(day)} {year}",
            f"{month_name} {int(day):02d} {year}",
            f"{int(day):02d} {month_name} {year}",
        ]
        for separator in ("v", "vs"):
            for date_form in date_forms:
                aliases.append(f"{prefix.replace(' v ', f' {separator} ')} {date_form}")
    # Ignore only an initial article when matching title text. Do not add
    # arbitrary one-word fragments, which would make generic evidence pass.
    unique: List[str] = []
    seen: set[str] = set()
    for alias in aliases:
        value = str(alias or "").strip()
        key = _entity_text_key(value)
        if key and key not in seen and len(key.split()) >= 2:
            seen.add(key)
            unique.append(value)
    return unique


def _root_target_membership_identifiers(target: Target) -> List[str]:
    """Extract stable record/object identifiers already present in seed metadata."""
    values = [str(target.entity_id or ""), str(target.name or "")]
    values.extend(str(url) for url in target.source_urls)
    identifiers: List[str] = []
    seen: set[str] = set()
    for value in values:
        for match in re.findall(
            r"(?<![A-Za-z0-9])([A-Za-z]{1,4}[.-]?\d{3,}(?:[.-]\d+)*)\b",
            value,
        ):
            key = _entity_text_key(match)
            if key and not key.isdigit() and key not in seen:
                seen.add(key)
                identifiers.append(match)
        for match in re.findall(r"(?<![A-Za-z0-9])([Oo]\d{4,})\b", value):
            key = _entity_text_key(match)
            if key not in seen:
                seen.add(key)
                identifiers.append(match)
    return identifiers


def _root_target_strong_tokens(target: Target) -> set[str]:
    """Return non-generic name tokens for a conservative fallback match."""
    ignored = {
        "the", "a", "an", "v", "vs", "episode", "issue", "object", "poster",
        "record", "item", "film", "work", "publication", "volume", "number",
        "series", "programme", "program", "match", "game", "event", "radio",
        "january", "february", "march", "april", "may", "june", "july",
        "august", "september", "october", "november", "december",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", _entity_text_key(target.name))
        if token not in ignored and len(token) >= 4 and not token.isdigit()
    }


def _root_constraint_quality_failure(
    target: Target,
    constraints: Sequence[ConstraintPath],
    *,
    min_expandable_core_paths: int,
    require_source_diversity: bool = False,
) -> Dict[str, Any] | None:
    """Verify root clue truth/provenance and ensure Local has useful anchors."""
    failures: List[Dict[str, Any]] = []
    path_ids = [path.path_id for path in constraints]
    if len(path_ids) != len(set(path_ids)):
        failures.append(
            {"code": "duplicate_path_ids", "path_ids": path_ids}
        )
    target_aliases = {
        _entity_text_key(target.entity_id),
        _entity_text_key(target.name),
    }
    target_aliases.discard("")
    expandable_core_ids: List[str] = []
    attribute_core_ids: List[str] = []
    core_ids: List[str] = []
    core_source_urls: Dict[str, set[str]] = {}
    for path in constraints:
        word_count = len(re.findall(r"\b\w+\b", path.clue or ""))
        if word_count == 0 or word_count > 40:
            failures.append(
                {
                    "code": "root_clue_not_atomic",
                    "path_id": path.path_id,
                    "word_count": word_count,
                }
            )
        forbidden_values = [
            *(("seed_target", alias) for alias in _target_text_aliases(target)),
            *(("final_answer", alias) for alias in _answer_text_aliases(target)),
        ]
        for kind, value in forbidden_values:
            if _find_entity_text_reference(value, path.clue):
                failures.append(
                    {
                        "code": "forbidden_reference_in_root_clue",
                        "path_id": path.path_id,
                        "forbidden_kind": kind,
                        "forbidden_text": value,
                    }
                )
        if not any(evidence.url.strip() for evidence in path.evidence):
            failures.append(
                {"code": "root_clue_missing_evidence", "path_id": path.path_id}
            )
        # Exact target membership is judged by the Root Ambiguity model after it
        # opens the cited evidence.  This structural gate intentionally checks
        # only that evidence exists; heuristic text matching here caused valid
        # records with shortened titles or catalog wording to be rejected before
        # the evidence-aware verifier could assess them.
        if path.role == "core":
            core_ids.append(path.path_id)
            core_source_urls[path.path_id] = {
                _source_url_key(evidence.url)
                for evidence in path.evidence
                if _source_url_key(evidence.url)
            }
        candidate_keys = {
            _entity_text_key(candidate) for candidate in path.candidates
        }
        if path.role == "distractor" and not (candidate_keys & target_aliases):
            failures.append(
                {"code": "distractor_missing_target", "path_id": path.path_id}
            )
        if path.role == "core" and _local_branch_kind(path.branch) == "relation":
            expandable_core_ids.append(path.path_id)
        elif path.role == "core" and _local_branch_kind(path.branch) == "attribute":
            attribute_core_ids.append(path.path_id)
    seed_source_keys = {
        _source_url_key(url) for url in target.source_urls if _source_url_key(url)
    }
    core_seed_source_ids = sorted(
        path_id
        for path_id, urls in core_source_urls.items()
        if urls & seed_source_keys
    )
    if core_seed_source_ids and require_source_diversity:
        failures.append(
            {
                "code": "root_core_uses_seed_source",
                "path_ids": core_seed_source_ids,
                "source_urls": sorted(
                    {
                        url
                        for path_id in core_seed_source_ids
                        for url in core_source_urls.get(path_id, set())
                        if url in seed_source_keys
                    }
                ),
                "reason": (
                    "core evidence must use a non-seed URL so target/answer "
                    "source evidence is not reused"
                ),
            }
        )
    if len(attribute_core_ids) > 1:
        failures.append(
            {
                "code": "root_too_many_attribute_cores",
                "path_ids": attribute_core_ids,
                "actual": len(attribute_core_ids),
                "maximum": 1,
                "reason": "at most one Root core may be an attribute clue",
            }
        )
    required_relations = max(0, len(core_ids) - 1)
    if len(expandable_core_ids) < required_relations:
        failures.append(
            {
                "code": "root_core_relation_required",
                "path_ids": [
                    path.path_id
                    for path in constraints
                    if path.role == "core"
                    and path.path_id not in expandable_core_ids
                ],
                "required": required_relations,
                "actual": len(expandable_core_ids),
                "reason": (
                    "all Root cores except at most one attribute must be "
                    "relation branches with an expandable named entity"
                ),
            }
        )
    required = min(
        max(0, min_expandable_core_paths),
        sum(path.role == "core" for path in constraints),
    )
    if len(expandable_core_ids) < required:
        failures.append(
            {
                "code": "insufficient_expandable_core_paths",
                "required": required,
                "actual": len(expandable_core_ids),
                "expandable_core_path_ids": expandable_core_ids,
                "non_expandable_core_path_ids": [
                    path.path_id
                    for path in constraints
                    if path.role == "core" and path.path_id not in expandable_core_ids
                ],
            }
        )
    if failures:
        return {
            "code": "root_quality_failures",
            "failure_count": len(failures),
            "failures": failures,
        }
    return None


def _compact_root_failure_report(
    failure_report: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """Keep one actionable Root failure without recursive/local payloads."""
    if not isinstance(failure_report, dict):
        return {"code": "unknown", "message": "repair the previous root set"}
    reason = str(failure_report.get("reason") or "")
    compact: Dict[str, Any] = {
        "stage": str(failure_report.get("stage") or ""),
        "message": reason,
    }
    if reason.startswith(("root_quality:", "root_ambiguity:")):
        prefix = reason.split(":", 1)[0]
        try:
            quality = json.loads(reason.split(":", 1)[1])
        except (TypeError, ValueError, json.JSONDecodeError):
            quality = {"code": "root_verification", "message": reason}
        if isinstance(quality, dict):
            if prefix == "root_ambiguity":
                failures = [
                    {
                        key: item.get(key)
                        for key in (
                            "code",
                            "path_id",
                            "path_ids",
                            "required",
                            "actual",
                            "maximum",
                            "replacement_path_ids",
                            "expandable_core_path_ids",
                            "alternatives",
                            "reason",
                            "source_urls",
                        )
                        if key in item
                    }
                    for item in quality.get("failures", [])
                    if isinstance(item, dict)
                ]
                compact.update(
                    {
                        "code": "root_ambiguity_failures",
                        "failure_count": int(quality.get("failure_count") or 0),
                        "failures": failures,
                        "editable_path_ids": sorted(
                            {
                                str(path_id)
                                for item in failures
                                for path_id in (
                                    item.get("path_ids")
                                    if isinstance(item.get("path_ids"), list)
                                    else [item.get("path_id")]
                                )
                                if str(path_id or "").strip()
                            }
                        ),
                    }
                )
                verified_path_alternatives: Dict[str, List[str]] = {}
                for item in quality.get("path_results", []):
                    if not isinstance(item, dict):
                        continue
                    path_id = str(item.get("path_id") or "").strip()
                    names = _root_alternative_names(item.get("alternatives"))
                    if path_id and names:
                        verified_path_alternatives[path_id] = names[:4]
                if verified_path_alternatives:
                    compact["verified_path_alternatives"] = verified_path_alternatives
            else:
                quality_failures = quality.get("failures")
                safe_failures = [
                    {
                        key: item.get(key)
                        for key in (
                            "code",
                            "path_id",
                            "path_ids",
                            "forbidden_kind",
                            "required",
                            "actual",
                            "maximum",
                            "replacement_path_ids",
                            "expandable_core_path_ids",
                            "reason",
                            "source_urls",
                        )
                        if key in item
                    }
                    for item in quality_failures or []
                    if isinstance(item, dict)
                ] if isinstance(quality_failures, list) else []
                compact.update(
                    {
                        "message": "Root quality reported actionable private-safe failures",
                        "code": str(quality.get("code") or "root_quality_failures"),
                        "failure_count": int(
                            quality.get("failure_count") or len(safe_failures)
                        ),
                        "failures": safe_failures,
                    }
                )
                if isinstance(quality_failures, list):
                    compact["editable_path_ids"] = sorted(
                        {
                            str(path_id)
                            for item in safe_failures
                            if isinstance(item, dict)
                            for path_id in (
                                item.get("path_ids")
                                if isinstance(item.get("path_ids"), list)
                                else [item.get("path_id")]
                            )
                            if str(path_id or "").strip()
                        }
                    )
        return compact
    report = failure_report.get("report")
    if isinstance(report, dict):
        single = report.get("single_path_candidates")
        candidate_reason = str(report.get("reason") or reason)
        compact["candidate_gate"] = {
            "reason": candidate_reason,
            "target_key": str(report.get("target_key") or ""),
            "core_path_ids": [
                str(item) for item in report.get("core_path_ids", [])
            ],
            "all_core_candidates": [
                str(item) for item in report.get("all_core_candidates", [])
            ],
            "candidate_counts": {
                str(path_id): len(values) if isinstance(values, list) else 0
                for path_id, values in (single or {}).items()
            } if isinstance(single, dict) else {},
        }
        pair_match = re.search(
            r"core pair ([^/\s]+)/([^\s]+) has (\d+) shared candidates, below (\d+)",
            candidate_reason,
        )
        distractor_match = re.search(
            r"core path ([^\s]+) has only (\d+) distractors with >= (\d+) non-target shared candidates",
            candidate_reason,
        )
        core_count_match = re.search(
            r"core path ([^\s]+) has (\d+) candidates outside bounds",
            candidate_reason,
        )
        if pair_match:
            left, right, actual, required = pair_match.groups()
            compact.update(
                {
                    "code": "root_pair_candidate_overlap",
                    "path_ids": [left, right],
                    "editable_path_ids": [left, right],
                    "actual_shared_candidates": int(actual),
                    "required_shared_candidates": int(required),
                }
            )
        elif distractor_match:
            path_id, actual, required_overlap = distractor_match.groups()
            compact.update(
                {
                    "code": "distractor_core_coverage",
                    "path_id": path_id,
                    "editable_roles": ["distractor"],
                    "actual_supporting_distractors": int(actual),
                    "required_non_target_overlap": int(required_overlap),
                }
            )
        elif candidate_reason.startswith("all core paths do not uniquely identify target"):
            target_key = str(report.get("target_key") or "")
            survivors = [
                str(item)
                for item in report.get("all_core_candidates", [])
                if str(item) and str(item) != target_key
            ]
            compact.update(
                {
                    "code": "root_candidate_joint_not_unique",
                    "alternatives": survivors,
                    "candidate_survivors_unverified": True,
                    "editable_path_ids": [
                        str(item) for item in report.get("core_path_ids", [])
                    ],
                }
            )
        elif core_count_match:
            path_id, actual = core_count_match.groups()
            compact.update(
                {
                    "code": "root_candidate_count_out_of_bounds",
                    "path_id": path_id,
                    "editable_path_ids": [path_id],
                    "actual_candidates": int(actual),
                }
            )
    return compact


def _preserve_verified_root_candidates(
    paths: Sequence[ConstraintPath],
    *,
    previous_constraints: Sequence[ConstraintPath],
    failure_report: Dict[str, Any] | None,
) -> None:
    """Keep known candidate evidence when a Root clue itself was not rewritten."""
    previous_by_id = {path.path_id: path for path in previous_constraints}
    verified_by_path: Dict[str, List[str]] = {}
    compact = _compact_root_failure_report(failure_report)
    preserve_previous_samples = (
        str(compact.get("code") or "") != "root_candidate_joint_not_unique"
    )
    path_alternatives = compact.get("verified_path_alternatives")
    if isinstance(path_alternatives, dict):
        for path_id, alternatives in path_alternatives.items():
            if isinstance(alternatives, list):
                verified_by_path.setdefault(str(path_id), []).extend(
                    str(item) for item in alternatives if str(item).strip()
                )
    failures = compact.get("failures")
    failures = failures if isinstance(failures, list) else []
    for failure in failures:
        if not isinstance(failure, dict) or failure.get("code") not in {
            "root_joint_not_unique",
        }:
            continue
        path_ids = failure.get("path_ids")
        path_ids = path_ids if isinstance(path_ids, list) else []
        raw_alternatives = failure.get("alternatives")
        raw_alternatives = raw_alternatives if isinstance(raw_alternatives, list) else []
        alternatives = []
        for item in raw_alternatives:
            name = str(item.get("name") if isinstance(item, dict) else item).strip()
            if name:
                alternatives.append(name)
        for path_id in path_ids:
            verified_by_path.setdefault(str(path_id), []).extend(alternatives)

    for path in paths:
        previous = previous_by_id.get(path.path_id)
        if previous is None or _entity_text_key(previous.clue) != _entity_text_key(path.clue):
            continue
        merged: List[str] = []
        seen: set[str] = set()
        for candidate in [
            *path.candidates,
            *(previous.candidates if preserve_previous_samples else []),
            *verified_by_path.get(path.path_id, []),
        ]:
            text = str(candidate).strip()
            key = _entity_text_key(text)
            if text and key and key not in seen:
                seen.add(key)
                merged.append(text)
        path.candidates = merged


def _root_alternative_names(value: Any) -> List[str]:
    values = value if isinstance(value, list) else []
    names: List[str] = []
    seen: set[str] = set()
    for item in values:
        name = str(item.get("name") if isinstance(item, dict) else item).strip()
        key = _entity_text_key(name)
        if name and key and key not in seen:
            seen.add(key)
            names.append(name)
    return names


def _compact_local_failure_report(local_report: Dict[str, Any]) -> Dict[str, Any]:
    """Keep actionable local-verifier failures without embedding expanded trees."""
    compact: Dict[str, Any] = {
        key: local_report.get(key)
        for key in (
            "accepted",
            "acceptance_reason",
            "leaf_core_root_ids",
            "required_leaf_core_paths",
            "required_leaf_depth",
            "checked_nodes",
            "max_depth",
            "core_max_depth",
            "distractor_max_depth",
        )
        if key in local_report
    }
    failures = []
    for failure in (local_report.get("failures") or [])[:8]:
        if not isinstance(failure, dict):
            continue
        verifier = failure.get("verifier")
        entry = {
            key: failure.get(key)
            for key in ("path_id", "role", "depth", "reason")
            if key in failure
        }
        if isinstance(verifier, dict):
            entry["verifier"] = {
                key: verifier.get(key)
                for key in ("accepted", "reason", "score")
                if key in verifier
            }
        failures.append(entry)
    compact["failures"] = failures
    compact["failure_count"] = len(local_report.get("failures") or [])
    return compact


def _local_failure_reason(local_report: Dict[str, Any]) -> str:
    failures = local_report.get("failures") or []
    if not failures:
        return str(
            local_report.get("acceptance_reason")
            or "local verification failed"
        )
    first = failures[0]
    reason = first.get("reason")
    if reason:
        return str(reason)
    verifier = first.get("verifier")
    if isinstance(verifier, dict) and verifier.get("reason"):
        return str(verifier["reason"])
    return "local verification failed"


def filter_diverse_seeds(
    seeds: Sequence[SeedRecord],
    *,
    requested: int,
    existing_domain_counts: Dict[str, int] | None = None,
    preferred_domains: Sequence[str] = (),
    overused_source_domains: Sequence[str] = (),
) -> List[SeedRecord]:
    selected: List[SeedRecord] = []
    entity_type_counts: Dict[str, int] = {}
    answer_field_counts: Dict[str, int] = {}
    football_final_count = 0
    sports_match_count = 0
    domain_counts = dict(existing_domain_counts or {})
    preferred = [domain for domain in preferred_domains if domain != "model_choice"]
    indexed = list(enumerate(seeds))
    priority_indexes: list[int] = []
    for preferred_domain in preferred:
        match = next(
            (
                index
                for index, seed in indexed
                if index not in priority_indexes and target_domain(seed.target) == preferred_domain
            ),
            None,
        )
        if match is not None:
            priority_indexes.append(match)
    ordered = [indexed[index] for index in priority_indexes]
    ordered.extend(sorted(
        (item for item in indexed if item[0] not in priority_indexes),
        key=lambda item: (
            domain_counts.get(target_domain(item[1].target), 0),
            item[0],
        ),
    ))
    for _, seed in ordered:
        target = seed.target
        domain = target_domain(target)
        if normalize_domain(target.domain_family) == "unknown":
            continue
        entity_type_key = _seed_bucket(target.entity_type)
        answer_key = _answer_key(target.answer_field)
        name_key = _answer_key(target.name)
        is_match = entity_type_key in {"match", "sports match"} or "match" in entity_type_key
        is_football_final = (
            is_match
            and any(term in name_key for term in ["cup final", "world cup final", "super cup", "copa america final", "uefa"])
        )
        if is_match and sports_match_count >= max(2, requested // 4):
            continue
        if is_football_final and football_final_count >= 1:
            continue
        if entity_type_counts.get(entity_type_key, 0) >= max(1, requested // 4):
            continue
        if answer_field_counts.get(answer_key, 0) >= 2:
            continue
        current_min = min((domain_counts.get(item, 0) for item in DOMAIN_FAMILIES), default=0)
        if domain_counts.get(domain, 0) > current_min + 1 and domain not in preferred:
            continue
        selected.append(seed)
        entity_type_counts[entity_type_key] = entity_type_counts.get(entity_type_key, 0) + 1
        answer_field_counts[answer_key] = answer_field_counts.get(answer_key, 0) + 1
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        if is_match:
            sports_match_count += 1
        if is_football_final:
            football_final_count += 1
        if len(selected) >= requested:
            break
    return selected


def has_required_seed_sources(seed: SeedRecord) -> bool:
    urls = [url.strip() for url in seed.target.source_urls if url and url.strip()]
    unique_urls = list(dict.fromkeys(urls))
    if len(unique_urls) < 2:
        return False
    return any("wikipedia.org" not in url.lower() for url in unique_urls)


def _seed_bucket(value: str) -> str:
    key = _answer_key(value)
    if "football" in key or "soccer" in key or key == "match":
        return "match"
    if "publication" in key or "magazine" in key or "newspaper" in key:
        return "publication"
    if "episode" in key or "radio" in key or "tv" in key:
        return "episode"
    return key or "unknown"


def _dry_constraints(target: Target) -> List[ConstraintPath]:
    core_a = [target.entity_id, *[f"venue_period_{idx}" for idx in range(1, 20)]]
    core_b = [target.entity_id, *[f"competition_type_{idx}" for idx in range(1, 20)]]
    core_c = [target.entity_id, *[f"participant_pattern_{idx}" for idx in range(1, 20)]]
    return [
        ConstraintPath(
            path_id="core_venue_period",
            role="core",
            branch="venue-period",
            hop_count=2,
            estimated_candidate_count=60,
            clue="the event was staged at a national stadium in London during a compact late-2010s window",
            candidates=core_a,
        ),
        ConstraintPath(
            path_id="core_competition_type",
            role="core",
            branch="competition-type",
            hop_count=2,
            estimated_candidate_count=45,
            clue="the annual deciding match belonged to a national knockout cup for non-league clubs named after a vessel",
            candidates=core_b,
        ),
        ConstraintPath(
            path_id="core_finalist_pattern",
            role="core",
            branch="participant-pattern",
            hop_count=2,
            estimated_candidate_count=30,
            clue="its finalists combined a town-style club name with a local-industrial-style club name",
            candidates=core_c,
        ),
        ConstraintPath(
            path_id="distractor_month",
            role="distractor",
            branch="month",
            hop_count=1,
            estimated_candidate_count=120,
            clue="it was played in May, like several adjacent finals in the same competition",
            candidates=[target.entity_id, *core_a[1:8], *core_b[1:8], *core_c[1:6]],
        ),
        ConstraintPath(
            path_id="distractor_london_fixture",
            role="distractor",
            branch="location",
            hop_count=1,
            estimated_candidate_count=90,
            clue="it shares the broad London-stadium context with several other cup finals",
            candidates=[target.entity_id, *core_a[8:15], *core_b[8:15], *core_c[6:11]],
        ),
        ConstraintPath(
            path_id="distractor_non_league_context",
            role="distractor",
            branch="competition-level",
            hop_count=1,
            estimated_candidate_count=100,
            clue="it sits in a non-league national cup context that also describes adjacent nominal-year finals",
            candidates=[target.entity_id, *core_a[15:], *core_b[15:], *core_c[11:]],
        ),
    ]


def _dry_question(artifact: Artifact) -> str:
    clues = []
    for path in artifact.constraints:
        if path.role == "core":
            clues.append(path.clue)
    return "Identify the hidden event matching these dispersed clues, then give its " + artifact.target.answer_field + ": " + "; ".join(clues)
