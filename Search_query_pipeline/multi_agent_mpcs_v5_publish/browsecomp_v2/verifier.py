from __future__ import annotations

from itertools import combinations
import re
from typing import Dict, Iterable, List, Sequence, Set

from .schema import ConstraintPath, Target, VerifierReport


def verify_core_paths(
    target: Target,
    constraints: Sequence[ConstraintPath],
    *,
    min_core_paths: int = 2,
    min_distractor_paths: int = 0,
    max_distractor_paths: int = 3,
    min_single_candidates: int = 2,
    max_single_candidates: int = 500,
    min_pairwise_core_intersection: int = 0,
    min_distractor_core_overlap: int = 0,
    min_distractors_per_core: int = 0,
) -> VerifierReport:
    """Check Root candidate membership/counts and full-core uniqueness."""
    target_key = _canonical(target.entity_id or target.name)
    aliases = {_canonical(target.entity_id), _canonical(target.name)}
    cores = [path for path in constraints if path.role == "core"]
    distractors = [path for path in constraints if path.role == "distractor"]
    if len(cores) < min_core_paths:
        return _reject(
            f"fewer than {min_core_paths} core paths",
            cores,
            target_key,
            {},
            [],
            constraints,
        )
    if not (min_distractor_paths <= len(distractors) <= max_distractor_paths):
        return _reject(
            (
                f"distractor path count {len(distractors)} outside bounds "
                f"[{min_distractor_paths}, {max_distractor_paths}]"
            ),
            cores,
            target_key,
            {},
            [],
            constraints,
        )

    normalized_by_path: Dict[str, List[str]] = {}
    for path in cores:
        candidates = _drop_obvious_year_conflicts(path.clue, path.candidates)
        normalized = _normalize_candidates(candidates, target_key, aliases)
        normalized_by_path[path.path_id] = normalized
        unique = set(normalized)
        if target_key not in unique:
            return _reject(
                f"core path {path.path_id} does not include target",
                cores,
                target_key,
                normalized_by_path,
                [],
                constraints,
            )
        if not (min_single_candidates <= len(unique) <= max_single_candidates):
            return _reject(
                f"core path {path.path_id} has {len(unique)} candidates outside bounds",
                cores,
                target_key,
                normalized_by_path,
                [],
                constraints,
            )
        if unique == {target_key}:
            return _reject(
                f"core path {path.path_id} is uniquely identifying by itself",
                cores,
                target_key,
                normalized_by_path,
                [],
                constraints,
            )
    if min_pairwise_core_intersection > 0:
        for left, right in combinations(cores, 2):
            left_set = set(normalized_by_path[left.path_id])
            right_set = set(normalized_by_path[right.path_id])
            overlap = sorted(left_set & right_set)
            if len(overlap) < min_pairwise_core_intersection:
                return _reject(
                    (
                        f"core pair {left.path_id}/{right.path_id} has "
                        f"{len(overlap)} shared candidates, below {min_pairwise_core_intersection}"
                    ),
                    cores,
                    target_key,
                    normalized_by_path,
                    [],
                    constraints,
                )

    all_core = sorted(_intersect_all(normalized_by_path.values()))
    if all_core != [target_key]:
        return _reject(
            f"all core paths do not uniquely identify target: {all_core}",
            cores,
            target_key,
            normalized_by_path,
            all_core,
            constraints,
        )
    leaking = _leaking_distractor_ids(target_key, aliases, distractors)
    if leaking:
        return _reject(
            f"distractor paths uniquely leak target: {leaking}",
            cores,
            target_key,
            normalized_by_path,
            all_core,
            constraints,
        )

    distractor_sets = _normalized_distractor_sets(target_key, aliases, distractors)
    if min_distractors_per_core > 0 and min_distractor_core_overlap > 0:
        for path in cores:
            core_non_target = set(normalized_by_path[path.path_id]) - {target_key}
            supporting = [
                distractor_id
                for distractor_id, distractor_candidates in distractor_sets.items()
                if len(core_non_target & (distractor_candidates - {target_key})) >= min_distractor_core_overlap
            ]
            if len(supporting) < min_distractors_per_core:
                return _reject(
                    (
                        f"core path {path.path_id} has only {len(supporting)} distractors "
                        f"with >= {min_distractor_core_overlap} non-target shared candidates"
                    ),
                    cores,
                    target_key,
                    normalized_by_path,
                    all_core,
                    constraints,
                )

    return VerifierReport(
        accepted=True,
        reason="core paths contain the target and jointly identify it in the returned candidate matrix",
        core_path_ids=[path.path_id for path in cores],
        target_key=target_key,
        single_path_candidates={key: sorted(set(value)) for key, value in normalized_by_path.items()},
        all_core_candidates=all_core,
        distractor_report=_score_distractors(target_key, aliases, normalized_by_path, constraints),
    )


def verify_pairwise_core_paths(
    target: Target,
    constraints: Sequence[ConstraintPath],
    *,
    min_core_paths: int = 3,
    min_distractor_paths: int = 0,
    max_distractor_paths: int = 3,
    min_single_candidates: int = 2,
    max_single_candidates: int = 500,
    min_distractor_core_overlap: int = 0,
    min_distractors_per_core: int = 0,
) -> VerifierReport:
    """Strict optional Root gate: every core pair remains non-unique.

    The normal verifier remains unchanged.  Since every accepted core must
    contain the target, a pairwise intersection lower bound of two is exactly
    the required ``target + at least one real alternative`` condition.
    """
    report = verify_core_paths(
        target,
        constraints,
        min_core_paths=min_core_paths,
        min_distractor_paths=min_distractor_paths,
        max_distractor_paths=max_distractor_paths,
        min_single_candidates=min_single_candidates,
        max_single_candidates=max_single_candidates,
        min_pairwise_core_intersection=2,
        min_distractor_core_overlap=min_distractor_core_overlap,
        min_distractors_per_core=min_distractors_per_core,
    )
    if isinstance(report.distractor_report, dict):
        report.distractor_report["pairwise_gate_enabled"] = True
        report.distractor_report["pairwise_requirement"] = (
            "every core pair must contain target and at least one non-target candidate"
        )
    return report


def _drop_obvious_year_conflicts(
    clue: str,
    candidates: Sequence[str],
) -> List[str]:
    """Ignore candidate ids that explicitly contradict one exact clue year."""
    clue_years = set(re.findall(r"\b(?:18|19|20)\d{2}\b", str(clue or "")))
    if len(clue_years) != 1:
        return list(candidates)
    required_year = next(iter(clue_years))
    kept: List[str] = []
    for candidate in candidates:
        candidate_years = set(
            re.findall(r"\b(?:18|19|20)\d{2}\b", str(candidate or "").replace("_", " "))
        )
        if candidate_years and required_year not in candidate_years:
            continue
        kept.append(candidate)
    return kept


def verify_local_core_paths(
    target: Target,
    constraints: Sequence[ConstraintPath],
    *,
    min_core_paths: int = 3,
    min_distractor_paths: int = 1,
    min_single_candidates: int = 4,
    min_pairwise_core_intersection: int = 2,
) -> VerifierReport:
    """Legacy offline candidate diagnostic; production Local no longer calls this."""
    target_key = _canonical(target.entity_id or target.name)
    aliases = {_canonical(target.entity_id), _canonical(target.name)}
    cores = [path for path in constraints if path.role == "core"]
    distractors = [path for path in constraints if path.role == "distractor"]
    required_cores = max(2, min_core_paths)
    required_distractors = max(0, min_distractor_paths)
    required_candidates = max(2, min_single_candidates)
    required_pairwise_overlap = max(0, min_pairwise_core_intersection)
    if len(cores) < required_cores:
        return _reject(
            f"fewer than {required_cores} local core paths",
            cores,
            target_key,
            {},
            [],
            constraints,
        )
    if len(distractors) < required_distractors:
        return _reject(
            f"fewer than {required_distractors} local distractor paths",
            cores,
            target_key,
            {},
            [],
            constraints,
        )

    for path in distractors:
        normalized = set(_normalize_candidates(path.candidates, target_key, aliases))
        if target_key not in normalized:
            return _reject(
                f"local distractor path {path.path_id} does not include target",
                cores,
                target_key,
                {},
                [],
                constraints,
            )
        if len(normalized) < required_candidates:
            return _reject(
                (
                    f"local distractor path {path.path_id} has {len(normalized)} candidates; "
                    f"at least {required_candidates} are required"
                ),
                cores,
                target_key,
                {},
                [],
                constraints,
            )
    path_ids = [path.path_id for path in cores]
    if len(set(path_ids)) != len(path_ids):
        return _reject(
            "local core path ids are not unique",
            cores,
            target_key,
            {},
            [],
            constraints,
        )

    normalized_by_path: Dict[str, List[str]] = {}
    for path in cores:
        normalized = _normalize_candidates(path.candidates, target_key, aliases)
        normalized_by_path[path.path_id] = normalized
        unique = set(normalized)
        if target_key not in unique:
            return _reject(
                f"local core path {path.path_id} does not include target",
                cores,
                target_key,
                normalized_by_path,
                [],
                constraints,
            )
        if len(unique) < required_candidates:
            return _reject(
                (
                    f"local core path {path.path_id} has {len(unique)} candidates; "
                    f"at least {required_candidates} are required"
                ),
                cores,
                target_key,
                normalized_by_path,
                [],
                constraints,
            )

    if required_pairwise_overlap > 0:
        for left, right in combinations(cores, 2):
            left_set = set(normalized_by_path[left.path_id])
            right_set = set(normalized_by_path[right.path_id])
            overlap = sorted(left_set & right_set)
            if len(overlap) < required_pairwise_overlap:
                return _reject(
                    (
                        f"local core pair {left.path_id}/{right.path_id} has "
                        f"{len(overlap)} shared candidates, below "
                        f"{required_pairwise_overlap}"
                    ),
                    cores,
                    target_key,
                    normalized_by_path,
                    [],
                    constraints,
                )

    all_core = sorted(_intersect_all(normalized_by_path.values()))
    if all_core != [target_key]:
        return _reject(
            f"local core paths do not jointly identify target: {all_core}",
            cores,
            target_key,
            normalized_by_path,
            all_core,
            constraints,
        )

    distractor_report = _score_distractors(
        target_key,
        aliases,
        normalized_by_path,
        constraints,
    )
    distractor_report["note"] = (
        "local distractors include the local target but do not participate in the core uniqueness intersection"
    )
    return VerifierReport(
        accepted=True,
        reason="local core paths are individually ambiguous and jointly identify the local target",
        core_path_ids=path_ids,
        target_key=target_key,
        single_path_candidates={
            key: sorted(set(value)) for key, value in normalized_by_path.items()
        },
        all_core_candidates=all_core,
        distractor_report=distractor_report,
    )


def _reject(
    reason: str,
    cores: Sequence[ConstraintPath],
    target_key: str,
    normalized_by_path: Dict[str, List[str]],
    all_core: List[str],
    constraints: Sequence[ConstraintPath],
) -> VerifierReport:
    return VerifierReport(
        accepted=False,
        reason=reason,
        core_path_ids=[path.path_id for path in cores],
        target_key=target_key,
        single_path_candidates={key: sorted(set(value)) for key, value in normalized_by_path.items()},
        all_core_candidates=all_core,
        distractor_report=_score_distractors(target_key, {target_key}, normalized_by_path, constraints),
    )


def _score_distractors(
    target_key: str,
    aliases: Set[str],
    core_candidates: Dict[str, List[str]],
    constraints: Sequence[ConstraintPath],
) -> Dict[str, object]:
    core_union: Set[str] = set()
    for candidates in core_candidates.values():
        core_union.update(candidates)
    records = []
    for path in constraints:
        if path.role != "distractor":
            continue
        normalized = set(_normalize_candidates(path.candidates, target_key, aliases))
        overlap = sorted(normalized & core_union)
        per_core_overlap = {
            core_id: len((set(core_values) - {target_key}) & (normalized - {target_key}))
            for core_id, core_values in core_candidates.items()
        }
        records.append(
            {
                "path_id": path.path_id,
                "candidate_count": len(normalized),
                "overlap_with_core_union": len(overlap),
                "non_target_overlap_by_core": per_core_overlap,
                "overlap_ratio": (len(overlap) / max(1, len(normalized))),
                "contains_target": target_key in normalized,
                "leaks_target": normalized == {target_key},
            }
        )
    pairwise_core_intersections = {}
    for left, right in combinations(core_candidates.items(), 2):
        pairwise_core_intersections[f"{left[0]}__{right[0]}"] = len(set(left[1]) & set(right[1]))
    return {
        "count": len(records),
        "paths": records,
        "pairwise_core_intersections": pairwise_core_intersections,
        "core_estimated_candidate_counts": {
            path.path_id: path.estimated_candidate_count
            for path in constraints
            if path.role == "core" and path.estimated_candidate_count
        },
        "note": "distractors are scored but not required for uniqueness",
    }


def _leaking_distractor_ids(
    target_key: str,
    aliases: Set[str],
    distractors: Sequence[ConstraintPath],
) -> List[str]:
    leaking = []
    for path in distractors:
        normalized = set(_normalize_candidates(path.candidates, target_key, aliases))
        if normalized == {target_key}:
            leaking.append(path.path_id)
    return leaking


def _normalized_distractor_sets(
    target_key: str,
    aliases: Set[str],
    distractors: Sequence[ConstraintPath],
) -> Dict[str, Set[str]]:
    return {
        path.path_id: set(_normalize_candidates(path.candidates, target_key, aliases))
        for path in distractors
    }


def _intersect_all(candidate_sets: Iterable[Iterable[str]]) -> Set[str]:
    sets = [set(items) for items in candidate_sets]
    if not sets:
        return set()
    out = sets[0].copy()
    for items in sets[1:]:
        out &= items
    return out


def _normalize_candidates(candidates: Iterable[str], target_key: str, aliases: Set[str]) -> List[str]:
    out = []
    for candidate in candidates:
        key = _canonical(candidate)
        out.append(target_key if key in aliases else key)
    return out


def _canonical(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
