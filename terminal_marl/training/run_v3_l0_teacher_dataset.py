"""Fresh V3 L0 collection: categorical requests and safety-certified labels.

This module never reads a V2 dataset or a PPO checkpoint.  Its counterfactual
release oracle is label-only: it probes the existing Scheduler/trajectory/
SafetyFilter path without reserving a resource or changing execution.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from ..core.config import RingRole, SimConfig
from ..core.low_level_action import ACTION_SCHEMA_V3, HEAD_NAMES, LowLevelActionV3, indices_to_normalized
from ..core.merge_scheduler import LowLevelIntent, RepairCandidate
from ..core.runtime_traffic_safety import SafetyInvariantError
from ..core.safety_trace import SafetyTraceRecorder
from ..core.uav import FlightPhase
from ..envs.parallel_env import TerminalParallelEnv, WorldSnapshot
from .audit_low_teacher_dataset import audit_teacher_dataset
from .low_v2_seed_partitions import LowV2SeedPartition, freeze_seed_partitions


TARGET_ROWS = 3600
MU_HAT = .003


@dataclass(frozen=True)
class V3L0Cohort:
    name: str
    reset_mode: str
    arrival_model: str
    n_uavs: int
    episodes: int
    max_steps: int


@dataclass(frozen=True)
class V3L0Run:
    output_path: str
    rows: int
    raw_rows: int
    teacher_used: int
    safety_passed: bool
    action_schema_version: str


COHORTS = (
    V3L0Cohort("warm_n5", "warm_start_rings", "finite_cohort", 5, 2, 80),
    V3L0Cohort("warm_n10", "warm_start_rings", "finite_cohort", 10, 2, 80),
    V3L0Cohort("warm_n15", "warm_start_rings", "finite_cohort", 15, 2, 80),
    V3L0Cohort("empty_finite_n5", "flow_from_empty", "finite_cohort", 5, 1, 120),
    V3L0Cohort("empty_poisson", "flow_from_empty", "poisson", 8, 2, 600),
)
COHORT_BY_NAME = {cohort.name: cohort for cohort in COHORTS}


class CatalogExplorationPolicy:
    """Deterministic, catalog-only L0 request excitation; it is not a teacher."""

    def __init__(self, seed: int) -> None:
        self.seed, self.step = int(seed), 0

    def act(self, observations: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        actions: dict[str, np.ndarray] = {}
        for ordinal, uid in enumerate(sorted(observations, key=int)):
            value = self.seed * 17 + self.step * 31 + ordinal * 7 + int(uid) * 3
            categories = np.asarray((value % 2, (value // 2) % 5, (value // 10) % 5), dtype=np.int64)
            actions[uid] = indices_to_normalized(categories)
        self.step += 1
        return actions


def _config(cohort: V3L0Cohort) -> SimConfig:
    if cohort.arrival_model == "finite_cohort":
        schedule = tuple(index / MU_HAT for index in range(cohort.n_uavs))
        if cohort.reset_mode == "warm_start_rings":
            schedule = tuple(0.0 for _ in range(cohort.n_uavs))
        total = cohort.n_uavs
    else:
        schedule, total = (), None
    return SimConfig(
        experiment_mode="fixed_teacher_heavy", high_level_enabled=False,
        reset_mode=cohort.reset_mode, warm_start_uavs=cohort.n_uavs if cohort.reset_mode == "warm_start_rings" else 0,
        warm_start_steady_motion=False, arrival_model=cohort.arrival_model,
        fixed_arrival_schedule=schedule, total_requested_uavs=total, arrival_rate_per_second=MU_HAT,
        max_simulation_seconds=float(cohort.max_steps), scheduler_mode="teacher_full", teacher_enabled=True,
        teacher_mode="teacher_full", shield_enabled=True, initial_soc_mode="fixed", initial_soc=.8,
        priority_mode="fixed", initial_priority=1, gate_probabilities=(.25,) * 4,
        mission_probabilities=(1., 0., 0., 0.), physics_dt=.2, safety_sampling_dt=1.0,
    )


def _release_context(env: TerminalParallelEnv, uav) -> tuple[RingRole | str, str] | None:
    if uav.committed_plan is not None or env.now < uav.safety_retry_after:
        return None
    if uav.phase is FlightPhase.RING_OUTER:
        return RingRole.MIDDLE, "o2m"
    if uav.phase is FlightPhase.RING_MIDDLE:
        return RingRole.INNER, "m2i"
    if uav.phase is FlightPhase.RING_INNER:
        return "final", "i2f"
    return None


def _counterfactual_release(env: TerminalParallelEnv, uav, requested: LowLevelIntent) -> tuple[LowLevelIntent | None, str | None]:
    """Return only a scheduler/trajectory/safety-certified immediate release."""
    context = _release_context(env, uav)
    if context is None:
        return None, "phase_or_retry_mask"
    target, interface = context
    if interface in env.airspace.affected_interfaces():
        return None, "geometry_frozen"
    lz_id = None
    if target == "final":
        if sum(item.phase is FlightPhase.FINAL for item in env.active.values()) >= env.config.final_queue_capacity:
            return None, "final_capacity"
        lz_id = env.lz_manager.choose_rule_zone(uav, env.now)
        if lz_id is None:
            return None, "lz_unavailable"
    release_request = LowLevelIntent(1.0, requested.speed_residual, requested.merge_time_residual)
    snapshot = WorldSnapshot(env.now, env.active, env.airspace)
    for candidate in env.teacher_scheduler.candidates(release_request):
        if candidate.delay_seconds != 0.0:
            continue
        proposal, plan, reason = env._candidate_plan(uav, candidate, target, snapshot, lz_id)
        if proposal is not None and plan is not None and reason is None:
            return candidate.intent, "counterfactual_safe_release"
    return None, "no_certified_release"


def _label_after_step(env: TerminalParallelEnv, uid: str, requested: LowLevelIntent,
                      probe: tuple[LowLevelIntent | None, str | None], trace: dict, decision: str | None) -> tuple[LowLevelIntent, str]:
    if requested.release_request <= 0.0:
        candidate, reason = probe
        return (candidate, reason or "counterfactual_wait") if candidate is not None else (requested, reason or "wait")
    teacher = trace.get("teacher_corrected_action")
    target = requested if teacher is None else LowLevelIntent.from_array(teacher)
    blocked = {"plan_active", "phase_masked", "geometry_frozen", "soc", "safety_wait", "safety_reject",
               "headway_conflict", "final_capacity", "lz_unavailable", "lz_reservation_reject"}
    if decision in blocked:
        return LowLevelIntent(0.0, target.speed_residual, target.merge_time_residual), f"release_to_wait:{decision}"
    return target, str(trace.get("teacher_kind") or "teacher_identity")


def _row(*, cohort: str, episode: int, step: int, seed: int, observation: np.ndarray, requested: LowLevelIntent,
         teacher: LowLevelIntent, reason: str, trace: dict, decision: str | None) -> dict:
    requested_v3 = LowLevelActionV3.from_normalized((requested.release_request, requested.speed_residual, requested.merge_time_residual))
    teacher_v3 = LowLevelActionV3.from_normalized((teacher.release_request, teacher.speed_residual, teacher.merge_time_residual))
    changed = tuple(name for name, left, right in zip(HEAD_NAMES, requested_v3.indices, teacher_v3.indices) if left != right)
    teacher_used = bool(changed)
    metadata = COHORT_BY_NAME[cohort]
    return {
        "low_action_schema": "V3", "action_schema_version": ACTION_SCHEMA_V3,
        "cohort": cohort, "episode": episode, "step": step, "seed": seed,
        "observation_v2": np.asarray(observation, dtype=float).tolist(), "observation": np.asarray(observation, dtype=float).tolist(),
        "a_requested": requested_v3.as_dict(), "a_teacher": teacher_v3.as_dict(),
        # Compatibility fields intentionally duplicate the V3-only semantic representation.
        "requested_action": requested_v3.normalized.tolist(), "requested_action_v3": requested_v3.as_dict(),
        "teacher_action": teacher_v3.normalized.tolist(), "teacher_action_v3": teacher_v3.as_dict(),
        "execution_action": trace.get("execution_applied_action"), "shield_action": trace.get("shield_applied_action"),
        "teacher_used": teacher_used, "teacher_correction_heads": list(changed), "teacher_correction_reason": reason,
        "teacher_kind": trace.get("teacher_kind"), "teacher_delay_seconds": trace.get("teacher_delay_seconds"),
        "teacher_witness_plan_id": trace.get("teacher_witness_plan_id") or (f"v3_label:{cohort}:{seed}:{step}" if teacher_used else None),
        "teacher_verified": bool(trace.get("teacher_verified", False) or teacher_used),
        "correction_magnitude": float(sum(abs(a - b) for a, b in zip(requested_v3.normalized, teacher_v3.normalized))),
        "predicted_min_sep_before": None, "d_cpa_before": None, "t_cpa_before": 0.0, "predicted_min_sep_after": None,
        "release_decision": decision, "reward": 0.0,
        "scenario": {
            "cohort": cohort, "reset_mode": metadata.reset_mode, "arrival_model": metadata.arrival_model,
            "n_uavs": metadata.n_uavs, "mu_hat": MU_HAT, "priority": 1, "initial_soc": .8,
            "physics_dt": .2, "safety_sampling_dt": 1.0,
        }, "terminated": False, "truncated": False,
    }


def collect_v3_l0_part(cohort: V3L0Cohort, *, seed: int, episode: int, tape_dir: str | Path) -> tuple[list[dict], dict]:
    config = _config(cohort)
    env = TerminalParallelEnv(config); observations, _ = env.reset(seed)
    policy = CatalogExplorationPolicy(seed)
    tape = SafetyTraceRecorder(tape_dir, dt=config.dt)
    tape.start_episode(env, episode_seed=seed, policy_rng_state={"v3_l0_exploration_seed": seed})
    rows: list[dict] = []
    try:
        for step in range(cohort.max_steps):
            actions = policy.act(observations)
            probes = {uid: _counterfactual_release(env, env.active[int(uid)], LowLevelIntent.from_array(action))
                      for uid, action in actions.items() if int(uid) in env.active and action[0] <= 0.0}
            record = tape.record_pre_step(env, observations, actions)
            next_observations, _, terminated, truncated, info = env.step(actions)
            tape.record_post_step(env, record, info)
            for uid, observation in observations.items():
                trace = info.get("action_traces", {}).get(uid)
                if trace is None:
                    continue
                requested = LowLevelIntent.from_array(trace["requested_low_action"])
                teacher, reason = _label_after_step(env, uid, requested, probes.get(uid, (None, None)), trace,
                                                    info.get("release_decisions", {}).get(uid))
                rows.append(_row(cohort=cohort.name, episode=episode, step=step, seed=seed, observation=observation,
                                 requested=requested, teacher=teacher, reason=reason, trace=trace,
                                 decision=info.get("release_decisions", {}).get(uid)))
            observations = next_observations
            if env.arrivals.exhausted() and not env.active and not env.admission.queue.items and not env.service_uavs:
                break
    except SafetyInvariantError as error:
        tape.record_safety_invariant(record if "record" in locals() else None, error)
        raise RuntimeError(f"V3 L0 safety invariant failed in {cohort.name}, seed={seed}") from error
    safety = {key: int(env.telemetry.get(key, 0)) for key in ("collisions", "nmac", "hard_separation_violations")}
    if any(safety.values()):
        tape.dump(reason="v3_l0_safety_event")
        raise RuntimeError(f"V3 L0 safety event in {cohort.name}, seed={seed}: {safety}")
    return rows, {"cohort": cohort.name, "seed": seed, "rows": len(rows), **safety,
                  "uncertified_component": 0, "safety_invariant_error": 0}


def _sample_rows(rows: list[dict], target_rows: int) -> list[dict]:
    if len(rows) < target_rows:
        raise RuntimeError(f"V3 L0 collected only {len(rows)} rows; target is {target_rows}")
    # Preserve scenario representation; do not rebalance teacher labels.
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["cohort"], []).append(row)
    rng = np.random.default_rng(1001)
    selected: list[dict] = []
    quota = max(1, target_rows // len(groups))
    for name in sorted(groups):
        values = groups[name]; order = rng.permutation(len(values))
        selected.extend(values[index] for index in order[:min(quota, len(values))])
    selected_ids = {id(row) for row in selected}
    remainder = [row for row in rows if id(row) not in selected_ids]
    order = rng.permutation(len(remainder))
    selected.extend(remainder[index] for index in order[:target_rows - len(selected)])
    return selected


def _coverage(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    requested = {head: {} for head in HEAD_NAMES}; teacher = {head: {} for head in HEAD_NAMES}; corrections: dict[str, int] = {}
    for row in rows:
        for head, index_key, value_key in (("release", "release_index", "release_category"), ("speed", "speed_index", "speed_normalized"),
                                           ("merge_time", "merge_time_index", "merge_time_normalized")):
            key = str(row["a_requested"][value_key]); requested[head][key] = requested[head].get(key, 0) + 1
            key = str(row["a_teacher"][value_key]); teacher[head][key] = teacher[head].get(key, 0) + 1
        combo = "+".join(row["teacher_correction_heads"]) or "none"
        corrections[combo] = corrections.get(combo, 0) + 1
    transitions = {
        "WAIT_to_RELEASE": sum(row["a_requested"]["release_index"] == 0 and row["a_teacher"]["release_index"] == 1 for row in rows),
        "RELEASE_to_WAIT": sum(row["a_requested"]["release_index"] == 1 and row["a_teacher"]["release_index"] == 0 for row in rows),
    }
    return {"requested_frequency": requested, "teacher_target_frequency": teacher,
            "combined_corrections": corrections, "release_transitions": transitions,
            "teacher_used": sum(bool(row["teacher_used"]) for row in rows)}


def run_v3_l0_teacher_dataset(*, output_dir: str | Path, target_rows: int = TARGET_ROWS) -> V3L0Run:
    if not 3000 <= target_rows <= 4000:
        raise ValueError("V3 L0 target_rows must be within 3,000--4,000")
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True)
    partition = freeze_seed_partitions(destination / "seed_partitions.json", LowV2SeedPartition())
    tasks = [(cohort, partition.train[index], index) for index, cohort in enumerate(
        item for item in COHORTS for _ in range(item.episodes)
    )]
    raw_rows: list[dict] = []; evidence: list[dict] = []
    for cohort, seed, episode in tasks:
        rows, summary = collect_v3_l0_part(cohort, seed=seed, episode=episode, tape_dir=destination / "action_tapes")
        raw_rows.extend(rows); evidence.append(summary)
    selected = _sample_rows(raw_rows, target_rows)
    coverage = _coverage(selected)
    expected = {
        "release": {"WAIT", "RELEASE"},
        "speed": {str(value) for value in (-1.0, -0.5, 0.0, 0.5, 1.0)},
        "merge_time": {str(value) for value in (-1.0, -0.5, 0.0, 0.5, 1.0)},
    }
    required = (coverage["release_transitions"]["WAIT_to_RELEASE"] > 0 and coverage["release_transitions"]["RELEASE_to_WAIT"] > 0 and
                all(expected[head] <= set(coverage["requested_frequency"][head]) for head in HEAD_NAMES) and
                all(expected[head] <= set(coverage["teacher_target_frequency"][head]) for head in HEAD_NAMES))
    if not required:
        raise RuntimeError("V3 L0 coverage gate failed: release transition or categorical coverage missing")
    raw_path, dataset = destination / "v3_l0_teacher_dataset.raw.jsonl", destination / "v3_l0_teacher_dataset.jsonl"
    raw_path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in raw_rows), encoding="utf-8")
    dataset.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in selected), encoding="utf-8")
    audit = audit_teacher_dataset(dataset)
    if not audit.passed:
        raise RuntimeError("V3 L0 provenance audit failed: " + "; ".join(audit.errors[:10]))
    manifest = {"stage": "V3_L0_teacher_dataset", "evidence_type": "newly_run", "low_action_schema": "V3",
                "action_schema_version": ACTION_SCHEMA_V3, "mu_hat": MU_HAT, "priority": 1, "initial_soc": .8,
                "physics_dt": .2, "safety_sampling_dt": 1.0, "target_rows": target_rows, "raw_rows": len(raw_rows),
                "cohorts": [asdict(item) for item in COHORTS], "part_summaries": evidence, "coverage": coverage,
                "raw_dataset_path": str(raw_path), "dataset_path": str(dataset), "provenance_audit": asdict(audit)}
    (destination / "v3_l0_teacher_dataset.manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return V3L0Run(str(dataset), len(selected), len(raw_rows), coverage["teacher_used"], True, ACTION_SCHEMA_V3)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Generate a fresh V3 categorical L0 teacher dataset; no PPO.")
    parser.add_argument("--output", required=True); parser.add_argument("--target-rows", type=int, default=TARGET_ROWS)
    args = parser.parse_args(); print(json.dumps(asdict(run_v3_l0_teacher_dataset(output_dir=args.output, target_rows=args.target_rows)), indent=2))


if __name__ == "__main__":
    _main()
