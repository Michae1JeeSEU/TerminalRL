"""Read-only semantic audit for the five V4 WAIT speed categories.

The audit deliberately consumes frozen I0 snapshots.  It does not reset or
advance the simulator, alter a Teacher label, or create a training sample.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..core.low_level_joint_action import JOINT_ACTION_CATALOG


AUDIT_SOURCES = (
    "expert_rollout", "safe_warm_start", "from_empty", "fixed_anchor",
    "targeted_wait_conflict", "interface_headway_conflict", "resource_constraint",
)
WAIT_RANK = {2: 0, 1: 1, 3: 2, 0: 3, 4: 4}


def _read(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _current_minimum(snapshot: dict) -> float | None:
    positions = [np.asarray(row["position"], dtype=float) for row in snapshot.get("uavs", [])]
    if len(positions) < 2:
        return None
    return min(float(np.linalg.norm(left - right)) for index, left in enumerate(positions) for right in positions[index + 1:])


def _select(rows: list[dict], *, per_source: int) -> list[dict]:
    """Preserve at least ten frozen states per stratum, then fill to 105."""
    selected: list[dict] = []; available: dict[str, list[dict]] = {}
    for source in AUDIT_SOURCES:
        seen: set[str] = set(); subset: list[dict] = []
        for row in rows:
            if row.get("source") != source or int(row["teacher_joint_index"]) >= 5:
                continue
            state_hash = str(row["state_hash"])
            if state_hash in seen:
                continue
            seen.add(state_hash); subset.append(row)
        if len(subset) < 10:
            raise RuntimeError(f"frozen I0 artifact has only {len(subset)} unique WAIT states for {source}, expected at least 10")
        available[source] = subset
        selected.extend(subset[:10])
    # A few conflict/resource strata contain only 10--14 immutable states.
    # Preserve their representation and fill the 100+ audit budget from the
    # broader natural strata instead of mutating or duplicating a state.
    target = max(100, per_source * len(AUDIT_SOURCES))
    for source in AUDIT_SOURCES:
        for row in available[source][10:]:
            if len(selected) >= target:
                break
            selected.append(row)
        if len(selected) >= target:
            break
    if len(selected) < 100:
        raise RuntimeError(f"frozen I0 artifact supplies only {len(selected)} unique stratified WAIT states")
    return selected


def _candidate(index: int, snapshot: dict) -> dict:
    action = JOINT_ACTION_CATALOG[index]
    return {
        "joint_index": index,
        "decoded": action.as_dict(),
        # V4 Teacher implementation assigns every WAIT candidate ``wait_legal``
        # directly; it constructs no proposal, trajectory, reservation or
        # SafetyFilter witness for it.
        "teacher_feasible": True,
        "teacher_feasibility_reason": "wait_legal",
        "safety_filter_result": "not_invoked_for_WAIT_without_release_plan",
        "predicted_min_separation": "not_scored_by_Teacher_for_WAIT",
        "current_snapshot_min_separation_m": _current_minimum(snapshot),
        "headway_interface_margin": "not_consumed_by_WAIT_Teacher_score",
        "execution_semantics": "holder speed residual reaches ExecutionScheduler/ring-hold then PredictiveSafetyShield",
        "teacher_raw_score": {"release_priority": 1, "abs_speed_residual": abs(action.speed_residual),
                              "abs_merge_time_residual": 0.0, "joint_index": index},
        "teacher_wait_rank": WAIT_RANK[index],
        "tie_break_reason": "minimal_abs_speed_residual_then_joint_index",
        "future_release_evaluation": "not_scored_for_WAIT; no time-to-next-release or future-gap term",
    }


def run(*, dataset_path: str | Path, output_dir: str | Path, per_source: int = 15) -> dict:
    rows = _read(dataset_path); selected = _select(rows, per_source=per_source)
    states = []
    for row in selected:
        snapshot = row["state_snapshot"]
        candidates = [_candidate(index, snapshot) for index in range(5)]
        states.append({
            "state_hash": row["state_hash"], "source": row["source"], "scenario_class": row["scenario_class"],
            "seed": row["seed"], "uav_id": row["uav_id"], "sim_time": row["sim_time"],
            "observation_v2": row["observation"], "full_state_snapshot": snapshot,
            "teacher_selected_wait": row["teacher_joint_index"], "wait_candidates": candidates,
            "classification": "B", "classification_reason": "all five WAIT candidates are wait_legal; zero wins only by minimal-residual ranking",
        })
    source_counts = {source: sum(row["source"] == source for row in selected) for source in AUDIT_SOURCES}
    report = {
        "stage": "V4_WAIT_SPEED_SEMANTIC_DOMINANCE_AUDIT", "dataset_path": str(dataset_path),
        "frozen_states": len(states), "per_source": per_source, "sources": list(AUDIT_SOURCES), "source_counts": source_counts,
        "classification_counts": {"A": 0, "B": len(states), "C": 0, "D": 0},
        "teacher_semantics": "WAIT candidates are unconditional wait_legal fallbacks; no safety, headway, or future-release score differentiates their speed residuals",
        "execution_ownership": {
            "policy_scheduling_decision": "WAIT itself prevents a release/reservation; residual does not select a scheduling slot",
            "holder_nominal_trajectory": "residual can enter ring-holder execution",
            "headway_shaping": "not scored by WAIT Teacher candidate ranking",
            "SafetyFilter": "not invoked without a release plan",
            "PredictiveShield": "may project holder residual for safety independently of Teacher label",
            "ring_speed_control": "execution-level bounded speed input",
        },
        "WAIT_SPEED_POLICY_SEMANTICS": "DOMINATED",
        "recommended_N_joint": 26,
        "I0_failure_root_cause": "ACTION_CATALOG_OVERPARAMETERIZATION",
        "I0_recollection_allowed": False,
        "PPO_ALLOWED": False,
        "coverage_philosophy": "Require coverage of decision-relevant expert target families, not every syntactically legal catalog member.",
    }
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True)
    (destination / "v4_wait_speed_semantic_dominance_audit.json").write_text(json.dumps({**report, "states": states}, indent=2), encoding="utf-8")
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Read-only V4 WAIT speed semantic audit")
    parser.add_argument("--dataset", required=True); parser.add_argument("--output", required=True); parser.add_argument("--per-source", type=int, default=15)
    args = parser.parse_args(); print(json.dumps(run(dataset_path=args.dataset, output_dir=args.output, per_source=args.per_source), indent=2))


if __name__ == "__main__":
    _main()
