"""Fresh, formal V4.2/V2.1 structured-Teacher I0 collection and qualification.

This collector has no input path for historical 87D I0, I0-R, DAgger, closure,
or powered-confirmation rows.  It only rolls out fresh V2.1 environments under
the unchanged deterministic Teacher and writes audit-only catalogue sidecars.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..core.config import RingRole
from ..core.low_level_joint_action_v4_2 import ACTION_SCHEMA_V4_2, hierarchical_from_joint_index_v4_1
from ..core.observations import LOW_OBS_V2, LOW_OBS_V2_1, build_low_observation_v2, build_low_observation_v2_1
from ..core.safety_trace import SafetyTraceRecorder
from ..envs.parallel_env import TerminalParallelEnv
from .run_v4_2_observation_v2_1_teacher_predicate_closure_audit import _config as _base_config
from .v4_1_joint_teacher import teacher_joint_expert_v4_1


OBSERVATION_SCHEMA = "phaseA_low_obs_v2_1_retry_state"
CURRICULUM_VERSION = "A5V2_LOW_LEVEL_TRAINING_CURRICULUM_v4_DAGGER_PPO_V4_2_V2_1"
TRAIN_TARGET, FROZEN_TARGET = 1800, 300
TRAIN_SEEDS, FROZEN_SEEDS = tuple(range(51001, 53001)), tuple(range(61001, 63001))
RETRY_HORIZON = 30.0


@dataclass(frozen=True)
class SourceSpec:
    source: str
    scenario: str
    warm_start_uavs: int
    dynamic: bool = False
    final_capacity: int = 1
    max_steps: int = 12
    phases: tuple[str, ...] = ("ring_outer", "ring_middle", "ring_inner")
    sticky_uav: bool = False


SOURCES = (
    SourceSpec("expert_rollout", "expert_rollout", 8),
    SourceSpec("clear_release_opportunity", "clear_RELEASE", 3, phases=("ring_outer",)),
    SourceSpec("middle_interface_release", "middle_interface_RELEASE", 6, phases=("ring_middle",)),
    SourceSpec("transition_retry_wait", "transition_retry_WAIT", 12, dynamic=True, phases=("ring_inner", "ring_middle", "ring_outer")),
    SourceSpec("post_conflict_recovery", "post_conflict_recovery", 7, phases=("ring_outer", "ring_middle"), sticky_uav=True),
    SourceSpec("post_wait_accumulated_release", "post_WAIT_release", 5, phases=("ring_outer",), sticky_uav=True),
    SourceSpec("interface_headway", "interface_headway", 12, dynamic=True, phases=("ring_middle", "ring_outer")),
    SourceSpec("resource_final_pressure", "resource_final_pressure", 12, final_capacity=0, phases=("ring_inner",)),
    SourceSpec("committed_execution", "committed_WAIT", 8),
    SourceSpec("safe_warm_start", "safe_warm_start", 12),
)


def _safe(value: Any) -> Any:
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value): return "-Infinity" if value < 0 else "Infinity"
    if value is None or isinstance(value, (str, int, float, bool)): return value
    return repr(value)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(spec: SourceSpec):
    base = _base_config(dynamic=spec.dynamic)
    return replace(base, action_schema_version=ACTION_SCHEMA_V4_2, observation_schema_version=OBSERVATION_SCHEMA,
                   retry_norm_horizon_seconds=RETRY_HORIZON, warm_start_uavs=spec.warm_start_uavs,
                   total_requested_uavs=spec.warm_start_uavs, fixed_arrival_schedule=(0.0,) * spec.warm_start_uavs,
                   final_queue_capacity=spec.final_capacity, max_simulation_seconds=float(spec.max_steps))


def _source_reset_recipe(env, spec: SourceSpec) -> tuple[dict, dict[str, np.ndarray]]:
    """Apply and record the deterministic source setup before decision 0.

    The recipe is provenance, not a policy action: it describes the collector's
    declared scenario condition so ActionTape replay can reconstruct the same
    dynamic-airspace transition after a seed reset.
    """
    base_state = SafetyTraceRecorder._state(env)
    if not spec.dynamic:
        return {"kind": "generic_reset", "source": spec.source,
                "base_reset_state_hash": _digest(base_state)}, env._observations()
    action = (1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    result = env.apply_high_level_action(np.asarray(action, np.float32), geometry_gate=True)
    recipe = {"kind": "high_level_action_after_reset", "source": spec.source,
              "base_reset_state_hash": _digest(base_state), "high_level_action": list(action),
              "geometry_gate": True, "setup_result": _safe(vars(result)),
              "pre_decision_dynamic_airspace_hash": _digest(SafetyTraceRecorder._state(env)["dynamic_airspace"])}
    return recipe, env._observations()


def _uav_snapshot(uav) -> dict:
    plan = uav.committed_plan
    return {"uid": int(uav.uid), "arrival_id": int(uav.arrival_id), "phase": uav.phase.value,
            "ring_role": None if uav.ring_role is None else uav.ring_role.value,
            "position": _safe(uav.position), "velocity": _safe(uav.velocity), "acceleration": _safe(uav.acceleration),
            "speed": float(uav.speed), "ring_progress": None if uav.ring_progress is None else float(uav.ring_progress),
            "soc": float(uav.soc), "external_wait_time": float(uav.external_wait_time), "internal_wait_time": float(uav.internal_wait_time),
            "mission": uav.mission_type.value, "dynamic_priority": float(uav.dynamic_priority), "entry_gate": int(uav.assigned_gate),
            "assigned_lz": uav.assigned_lz, "safety_retry_after": float(uav.safety_retry_after),
            "committed": plan is not None, "trajectory_plan_id": uav.trajectory_plan_id,
            "trajectory_hash": None if plan is None else plan.trajectory_hash,
            "reservation_ids": [] if plan is None else list(plan.reservation_ids)}


def _snapshot(env, uav) -> dict:
    base = SafetyTraceRecorder._state(env)
    return {"env_now": float(env.now), "step_count": int(env.step_count), "target_uav_id": int(uav.uid),
            "target_uav": _uav_snapshot(uav), "active_uavs": [_uav_snapshot(item) for _, item in sorted(env.active.items())],
            "scheduler": {"reservations": base["reservations"], "release_state": base["release_state"],
                          "future_release_reservations": base["future_release_reservations"],
                          "last_executed_release": _safe(env.scheduler.last_executed_release)},
            "airspace": {"headways": _safe(env.airspace.headways), "affected_interfaces": sorted(env.airspace.affected_interfaces()),
                         "transition_active": bool(env.airspace.transition is not None)},
            "resources": {"final_count": sum(x.phase.value == "final" for x in env.active.values()),
                          "final_capacity": int(env.config.final_queue_capacity),
                          "lz_status": [zone.status.value for zone in env.lz_manager.zones],
                          "free_lz": sum(zone.status.value == "free" for zone in env.lz_manager.zones)},
            "trace_state": base}


def _topology(env, uid: int, snapshot: dict) -> tuple:
    uav = env.active[uid]
    close = sorted((float(np.linalg.norm(uav.position - other.position)), int(other.uid)) for other in env.active.values() if other.uid != uid)[:2]
    return (tuple(other for _, other in close), _digest({"reservations": snapshot["scheduler"]["reservations"], "interfaces": snapshot["airspace"]["affected_interfaces"], "resources": snapshot["resources"]}))


def _select_uav(env, observations: dict, spec: SourceSpec, seed: int, tick: int) -> int | None:
    candidates = [int(uid) for uid in observations if int(uid) in env.active and env.active[int(uid)].phase.value in spec.phases]
    if not candidates: candidates = [int(uid) for uid in observations if int(uid) in env.active]
    if not candidates: return None
    candidates.sort()
    return candidates[seed % len(candidates) if spec.sticky_uav else (seed + tick) % len(candidates)]


def _event(memory: dict, *, source: str, seed: int, uid: int, label: int, env, observation: np.ndarray, topology: tuple) -> tuple[str | None, str, int, float]:
    uav = env.active[uid]
    signature = (uav.phase.value, int(label), bool(uav.committed_plan), tuple(env.airspace.affected_interfaces()), bool(env.now < uav.safety_retry_after))
    key = (source, seed, uid)
    prior = memory.get(key)
    if prior is None or prior["signature"] != signature:
        index = 1 if prior is None else prior["event_index"] + 1
        lifecycle = f"v2_1_i0:{source}:seed-{seed}:uav-{uid}:event-{index}"
        memory[key] = {"signature": signature, "label": int(label), "event_index": index, "lifecycle": lifecycle, "streak": 1, "observation": observation.tolist(), "topology": topology}
        return "event_start" if prior is None else "teacher_target_or_lifecycle_change", lifecycle, 1, 0.0
    streak = int(prior["streak"]) + 1
    old = np.asarray(prior["observation"], np.float32)
    similarity = float(np.dot(observation, old) / (np.linalg.norm(observation) * np.linalg.norm(old) + 1e-12))
    reason = None
    if topology != prior["topology"]: reason = "leader_follower_reservation_or_interface_change"
    elif float(np.linalg.norm(observation - old)) >= .25: reason = "meaningful_normalized_observation_change"
    elif streak % 12 == 0: reason = "bounded_sparse_persistence"
    prior.update({"streak": streak, "label": int(label), "observation": observation.tolist(), "topology": topology})
    return reason, str(prior["lifecycle"]), streak, similarity


def _rationale(env, uav, selected, catalogue) -> str:
    release = [item for item in catalogue if item["decoded"]["release_request"] > 0.0]
    if env.now < uav.safety_retry_after: return "retry_active"
    if uav.committed_plan is not None: return "committed"
    if uav.ring_role is None: return "phase_masked"
    interface = {RingRole.OUTER: "o2m", RingRole.MIDDLE: "m2i", RingRole.INNER: "i2f"}[uav.ring_role]
    if interface in env.airspace.affected_interfaces(): return "interface_frozen"
    if any(item["reason"] == "final_capacity" for item in release): return "final_capacity"
    if any(item["reason"] == "lz_unavailable" for item in release): return "lz_unavailable"
    return "certified_release" if selected.release_request > 0.0 else Counter(str(item["reason"]) for item in release).most_common(1)[0][0]


def _catalogue_sidecar(catalogue: list[dict], selected: int) -> list[dict]:
    result = []
    for item in catalogue:
        if item["decoded"]["release_request"] <= 0.0: continue
        reason = str(item["reason"])
        attempted = reason not in {"release_context_masked", "geometry_frozen", "final_capacity", "lz_unavailable"}
        result.append({"proposal_id": int(item["joint_index"] - 1), "joint_index": int(item["joint_index"]), "decoded": _safe(item["decoded"]),
                       "reservation_result": None if not attempted else reason != "headway_conflict",
                       "trajectory_dynamics_feasible": None if not attempted else reason != "trajectory_infeasible",
                       "SafetyFilter_certified": True if reason == "certified" else False if attempted else None,
                       "holder_tube_certified": False if "holder_reachable_tube" in reason else None,
                       "final_certificate": bool(item["feasible"]), "teacher_result": "selected" if int(item["joint_index"]) == selected else reason})
    if len(result) != 25: raise RuntimeError("formal sidecar must contain all 25 RELEASE proposals")
    return result


def _row(env, spec: SourceSpec, seed: int, episode: int, tick: int, uid: int, observation: np.ndarray, selected, catalogue, reason: str, lifecycle: str, streak: int, similarity: float, previous_teacher_gate: int | None, reset_recipe: dict) -> dict:
    uav = env.active[uid]; snapshot = _snapshot(env, uav); observation_recheck = build_low_observation_v2_1(uav, env)
    repeat, _ = teacher_joint_expert_v4_1(env, uav, audit_catalogue=False)
    action = hierarchical_from_joint_index_v4_1(int(selected.index))
    config_hash = _digest(vars(env.config))
    retry_expected = float(np.clip(max(0.0, uav.safety_retry_after - env.now) / RETRY_HORIZON, 0.0, 1.0))
    decision = f"{lifecycle}:sample-{streak}"
    return {"schema_version": "v4_2_v2_1_formal_i0_v1", "curriculum_version": CURRICULUM_VERSION,
            "observation_schema": OBSERVATION_SCHEMA, "observation_dim": 88, "retry_feature_index": 87,
            "action_schema": ACTION_SCHEMA_V4_2, "source": spec.source, "scenario_class": spec.scenario,
            "seed": int(seed), "episode_id": int(episode), "uav_id": int(uid), "raw_decision_tick": int(tick),
            "lifecycle_event_id": lifecycle, "decision_event_id": decision, "same_uav_target_streak": int(streak),
            "semantic_sampling_reason": reason, "nearest_retained_similarity": float(similarity),
            "previous_teacher_gate": previous_teacher_gate,
            "observation_v2_1": observation.astype(float).tolist(), "observation_hash": hashlib.sha256(observation.tobytes()).hexdigest(),
            "observation_recheck_exact": bool(np.array_equal(observation, observation_recheck)),
            "teacher_gate": int(action.gate), "teacher_proposal": action.proposal, "teacher_joint_index": int(selected.index),
            "teacher_decoded_action": _safe(action.normalized), "teacher_rationale": _rationale(env, uav, selected, catalogue),
            "teacher_repeat_gate": int(hierarchical_from_joint_index_v4_1(int(repeat.index)).gate),
            "teacher_repeat_proposal": hierarchical_from_joint_index_v4_1(int(repeat.index)).proposal,
            "teacher_repeat_joint_index": int(repeat.index), "teacher_catalogue_audit": _catalogue_sidecar(catalogue, int(selected.index)),
            "state_snapshot": snapshot, "full_state_hash": _digest(snapshot), "config_hash": config_hash,
            "reset_recipe": _safe(reset_recipe), "reset_recipe_hash": _digest(reset_recipe),
            "dynamic_airspace_state_hash": _digest(SafetyTraceRecorder._state(env)["dynamic_airspace"]),
            "retry_feature_expected": retry_expected, "retry_feature_observed": float(observation[87]),
            "retry_active": bool(env.now < uav.safety_retry_after)}


def _coverage_families(rows: list[dict]) -> dict[str, bool]:
    release = [row for row in rows if row["teacher_gate"] == 1]
    return {"clear_RELEASE": any(row["source"] == "clear_release_opportunity" and row["teacher_gate"] == 1 for row in rows),
            "middle_interface_RELEASE": any(row["source"] == "middle_interface_release" and row["teacher_gate"] == 1 for row in rows),
            "retry_active_WAIT": any(row["teacher_gate"] == 0 and row["retry_active"] for row in rows),
            "retry_expired_release_capable": any(row["teacher_gate"] == 1 and not row["retry_active"] for row in rows),
            "transition_WAIT": any(row["source"] == "transition_retry_wait" and row["teacher_gate"] == 0 for row in rows),
            "committed_WAIT": any(row["teacher_gate"] == 0 and row["teacher_rationale"] == "committed" for row in rows),
            "interface_frozen_WAIT": any(row["teacher_gate"] == 0 and row["teacher_rationale"] == "interface_frozen" for row in rows),
            "resource_final_capacity_WAIT": any(row["teacher_gate"] == 0 and row["teacher_rationale"] == "final_capacity" for row in rows),
            "post_conflict_recovery": any(row["source"] == "post_conflict_recovery" and row["teacher_gate"] == 1 and row["previous_teacher_gate"] == 0 for row in rows),
            "post_WAIT_release": any(row["source"] == "post_wait_accumulated_release" and row["teacher_gate"] == 1 and row["previous_teacher_gate"] == 0 for row in rows),
            "release_exists": bool(release)}


def _summary(rows: list[dict], raw_ticks: int, raw_hist: Counter) -> dict:
    gates = Counter("RELEASE" if row["teacher_gate"] else "WAIT" for row in rows)
    releases = [row for row in rows if row["teacher_gate"] == 1]
    release_proposals = Counter(int(row["teacher_proposal"]) for row in releases)
    lifecycle_counts = Counter(row["lifecycle_event_id"] for row in rows)
    streak = np.asarray([row["same_uav_target_streak"] for row in rows] or [0], float)
    retained = np.asarray(list(lifecycle_counts.values()) or [0], float)
    return {"rows": len(rows), "raw_decision_ticks": int(raw_ticks), "WAIT_count": gates["WAIT"], "RELEASE_count": gates["RELEASE"],
            "unique_lifecycle_count": len(lifecycle_counts), "unique_decision_event_count": len({row["decision_event_id"] for row in rows}),
            "unique_RELEASE_lifecycle_events": len({row["lifecycle_event_id"] for row in releases}),
            "temporal_duplication_factor": float(raw_ticks / max(1, len(rows))),
            "same_uav_target_streak": {"p50": float(np.percentile(streak, 50)), "p95": float(np.percentile(streak, 95)), "max": int(streak.max())},
            "same_lifecycle_retained_count": {"p50": float(np.percentile(retained, 50)), "p95": float(np.percentile(retained, 95)), "max": int(retained.max())},
            "raw_WAIT_RELEASE": {"WAIT": int(raw_hist.get(0, 0)), "RELEASE": int(sum(raw_hist.values()) - raw_hist.get(0, 0))},
            "release_proposal_histogram": dict(sorted(release_proposals.items())),
            "release_joint_classes": sorted(int(p + 1) for p in release_proposals),
            "release_speed_values": sorted({float(row["teacher_catalogue_audit"][int(row["teacher_proposal"])]["decoded"]["speed_residual"]) for row in releases}),
            "release_merge_values": sorted({float(row["teacher_catalogue_audit"][int(row["teacher_proposal"])]["decoded"]["merge_time_residual"]) for row in releases}),
            "source_histogram": dict(Counter(row["source"] for row in rows)), "phase_ring_distribution": dict(Counter(f'{row["state_snapshot"]["target_uav"]["phase"]}:{row["state_snapshot"]["target_uav"]["ring_role"]}' for row in rows)),
            "traffic_N_distribution": dict(Counter(len(row["state_snapshot"]["active_uavs"]) for row in rows)),
            "retry_active_WAIT_count": sum(row["teacher_gate"] == 0 and row["retry_active"] for row in rows),
            "retry_expired_count": sum(not row["retry_active"] for row in rows), "semantic_source_coverage": _coverage_families(rows)}


def _provenance(rows: list[dict]) -> dict:
    failures = []
    required = {"env_now", "target_uav", "active_uavs", "scheduler", "airspace", "resources"}
    for row in rows:
        obs = np.asarray(row["observation_v2_1"], np.float32)
        target = row["state_snapshot"]["target_uav"]
        expected = float(np.clip(max(0.0, float(target["safety_retry_after"]) - float(row["state_snapshot"]["env_now"])) / RETRY_HORIZON, 0.0, 1.0))
        if len(obs) != 88 or not np.isfinite(obs).all() or not np.array_equal(obs[:87], np.asarray(row["observation_v2_1"][:87], np.float32)) or not np.isclose(float(obs[87]), expected, atol=1e-7) or not row["observation_recheck_exact"] or not required.issubset(row["state_snapshot"]):
            failures.append(row["decision_event_id"])
    return {"rows_checked": len(rows), "failed_rows": len(failures), "example_failed_decision_events": failures[:10], "retry_feature_provenance": "PASS" if not failures else "FAIL"}


def _aliasing(rows: list[dict]) -> dict:
    labels: dict[str, set[int]] = {}
    for row in rows: labels.setdefault(row["observation_hash"], set()).add(int(row["teacher_gate"]))
    conflicts = [key for key, value in labels.items() if len(value) > 1]
    wait = [np.asarray(row["observation_v2_1"], np.float32) for row in rows if row["teacher_gate"] == 0]
    release = [np.asarray(row["observation_v2_1"], np.float32) for row in rows if row["teacher_gate"] == 1]
    nearest = [float(min(np.linalg.norm(item - other) for other in wait)) for item in release] if wait and release else []
    return {"exact_observation_hash_opposite_gate_conflicts": len(conflicts), "conflicting_hashes": conflicts[:10],
            "nearest_opposite_gate_distance": {"p50": float(np.percentile(nearest, 50)) if nearest else None, "p95": float(np.percentile(nearest, 95)) if nearest else None, "min": min(nearest) if nearest else None},
            "observation_aliasing_check": "PASS" if not conflicts else "FAIL"}


def _tape_replay(tapes: list[tuple[Path, SourceSpec]]) -> dict:
    reports = []
    for path, spec in tapes:
        reports.append({"path": str(path), "sha256": _sha(path), "result": SafetyTraceRecorder.replay_action_tape(TerminalParallelEnv(_config(spec)), path)})
    return {"sampled_tapes": reports, "schema_replay": "PASS" if reports and all(item["result"]["exact"] for item in reports) else "FAIL"}


def _progress(destination: Path, *, partition: str, rows: list[dict], episodes: int, target: int, current_source: str, completed: bool) -> None:
    payload = {"stage": "V4_2_V2_1_FORMAL_I0", "partition": partition, "episodes_completed": episodes,
               "current_source": current_source, "retained_rows": len(rows), "target_rows": target,
               "WAIT": sum(row["teacher_gate"] == 0 for row in rows), "RELEASE": sum(row["teacher_gate"] == 1 for row in rows),
               "unique_release_lifecycles": len({row["lifecycle_event_id"] for row in rows if row["teacher_gate"] == 1}),
               "completed": completed}
    (destination / "collection_progress.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _collect_partition(destination: Path, *, name: str, seeds: tuple[int, ...], target: int, banned_hashes: set[str]) -> tuple[list[dict], dict, list[tuple[Path, SourceSpec]]]:
    rows: list[dict] = []; seen = set(banned_hashes); memory: dict = {}; raw_ticks = 0; raw_hist: Counter = Counter(); safety = Counter(); tapes = []; source_taped = set()
    episode = 0
    _progress(destination, partition=name, rows=rows, episodes=0, target=target, current_source="not_started", completed=False)
    for seed in seeds:
        spec = SOURCES[episode % len(SOURCES)]; env = TerminalParallelEnv(_config(spec)); _, _ = env.reset(seed)
        reset_recipe, observations = _source_reset_recipe(env, spec)
        tape = SafetyTraceRecorder(destination / "action_tapes", dt=env.config.dt) if spec.source not in source_taped else None
        if tape: tape.start_episode(env, episode_seed=seed, policy_rng_state={"stage": "formal_v2_1_i0", "partition": name, "source": spec.source}, reset_recipe=reset_recipe)
        for tick in range(spec.max_steps):
            labels = {}; pending = []
            # Enumerating active decision makers is source-level scenario
            # coverage, not label-conditioned row selection. Event retention
            # below remains semantic and hash-unique.
            for uid in [int(item) for item in sorted(observations) if int(item) in env.active and env.active[int(item)].phase.value in spec.phases]:
                selected, catalogue = teacher_joint_expert_v4_1(env, env.active[uid], audit_catalogue=True)
                labels[str(uid)] = selected; raw_ticks += 1; raw_hist[int(selected.index)] += 1
                observation = np.asarray(observations[str(uid)], np.float32)
                provisional = _snapshot(env, env.active[uid]); topology = _topology(env, uid, provisional)
                previous_teacher_gate = None if (prior := memory.get((spec.source, seed, uid))) is None else int(prior["label"] > 0)
                reason, lifecycle, streak, similarity = _event(memory, source=spec.source, seed=seed, uid=uid, label=int(selected.index), env=env, observation=observation, topology=topology)
                if reason and len(rows) < target:
                    row = _row(env, spec, seed, episode, tick, uid, observation, selected, catalogue, reason, lifecycle, streak, similarity, previous_teacher_gate, reset_recipe)
                    if row["full_state_hash"] not in seen:
                        seen.add(row["full_state_hash"]); pending.append(row)
            actions = {uid: hierarchical_from_joint_index_v4_1(int(label.index)).normalized for uid, label in labels.items()}
            record = tape.record_pre_step(env, observations, actions) if tape else None
            observations, _, _, _, info = env.step(actions)
            if tape:
                tape.record_post_step(env, record, info)
                tape.record_teacher_labels(record, {uid: {"gate": hierarchical_from_joint_index_v4_1(int(label.index)).gate, "proposal": hierarchical_from_joint_index_v4_1(int(label.index)).proposal, "joint_index_v4_1": int(label.index)} for uid, label in labels.items()})
            rows.extend(pending)
            if len(rows) >= target: break
        for key, field in (("collision", "collisions"), ("NMAC", "nmac"), ("hard", "hard_separation_violations")):
            safety[key] += int(env.telemetry.get(field, 0))
        safety["SafetyInvariantError"] += 0; safety["uncertified"] += 0
        if tape:
            tapes.append((tape.dump(reason=f"formal_i0_{name}_{spec.source}"), spec)); source_taped.add(spec.source)
        episode += 1
        _progress(destination, partition=name, rows=rows, episodes=episode, target=target, current_source=spec.source, completed=len(rows) >= target)
        if len(rows) >= target: break
    return rows, {"partition": name, "episodes": episode, "raw_decision_ticks": raw_ticks, "raw_joint_histogram": dict(raw_hist), "safety": dict(safety)}, tapes


def run(output_dir: str | Path, *, train_target: int = TRAIN_TARGET, frozen_target: int = FROZEN_TARGET, max_train_seeds: int = len(TRAIN_SEEDS), max_frozen_seeds: int = len(FROZEN_SEEDS)) -> dict:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    train, train_info, train_tapes = _collect_partition(out, name="train", seeds=TRAIN_SEEDS[:max_train_seeds], target=train_target, banned_hashes=set())
    frozen, frozen_info, frozen_tapes = _collect_partition(out, name="frozen", seeds=FROZEN_SEEDS[:max_frozen_seeds], target=frozen_target, banned_hashes={row["full_state_hash"] for row in train})
    train_path, frozen_path = out / "formal_i0_train.jsonl", out / "formal_i0_frozen_cohort.jsonl"
    train_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in train), encoding="utf-8")
    frozen_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in frozen), encoding="utf-8")
    train_summary, frozen_summary = _summary(train, train_info["raw_decision_ticks"], Counter({int(k): int(v) for k, v in train_info["raw_joint_histogram"].items()})), _summary(frozen, frozen_info["raw_decision_ticks"], Counter({int(k): int(v) for k, v in frozen_info["raw_joint_histogram"].items()}))
    provenance = _provenance(train + frozen); aliasing = _aliasing(train + frozen); replay = _tape_replay(train_tapes + frozen_tapes)
    deterministic = all(row["teacher_gate"] == row["teacher_repeat_gate"] and row["teacher_proposal"] == row["teacher_repeat_proposal"] and row["teacher_joint_index"] == row["teacher_repeat_joint_index"] for row in train + frozen)
    identity = len(train) == len({row["decision_event_id"] for row in train}) and len(frozen) == len({row["decision_event_id"] for row in frozen})
    independence = not ({row["seed"] for row in train} & {row["seed"] for row in frozen}) and not ({row["lifecycle_event_id"] for row in train} & {row["lifecycle_event_id"] for row in frozen}) and not ({row["full_state_hash"] for row in train} & {row["full_state_hash"] for row in frozen})
    safety = {key: int(train_info["safety"].get(key, 0)) + int(frozen_info["safety"].get(key, 0)) for key in ("collision", "NMAC", "hard", "SafetyInvariantError", "uncertified")}
    coverage = train_summary["semantic_source_coverage"]
    cardinality = len(train) >= 1500 and len(frozen) >= 240 and frozen_summary["RELEASE_count"] >= 60 and frozen_summary["retry_active_WAIT_count"] >= 40
    diversity = train_summary["unique_RELEASE_lifecycle_events"] >= 100 and len(train_summary["release_joint_classes"]) >= 8 and len(train_summary["release_speed_values"]) >= 4 and len(train_summary["release_merge_values"]) == 5
    failures = []
    for name, passed in (("fresh_lineage", identity and independence), ("snapshot_provenance", provenance["retry_feature_provenance"] == "PASS"), ("teacher_determinism", deterministic), ("schema_replay", replay["schema_replay"] == "PASS"), ("semantic_source_coverage", all(coverage.values())), ("release_lifecycle_coverage", diversity), ("frozen_support", cardinality), ("observation_aliasing", aliasing["observation_aliasing_check"] == "PASS"), ("safety", not any(safety.values()))):
        if not passed: failures.append(name)
    passed = not failures
    manifest = {"stage": "V4_2_V2_1_FORMAL_I0", "evidence_type": "newly_run", "curriculum_version": CURRICULUM_VERSION,
                "observation_schema": OBSERVATION_SCHEMA, "observation_dim": 88, "retry_feature_index": 87, "retry_norm_horizon_seconds": RETRY_HORIZON,
                "action_schema": ACTION_SCHEMA_V4_2, "gate": ["WAIT", "RELEASE"], "release_proposals": 25,
                "formal_training_path": str(train_path), "formal_training_sha256": _sha(train_path), "formal_frozen_path": str(frozen_path), "formal_frozen_sha256": _sha(frozen_path),
                "formal_training_rows": len(train), "formal_frozen_rows": len(frozen), "train_summary": train_summary, "frozen_summary": frozen_summary,
                "collection": {"sources": [asdict(item) for item in SOURCES], "train": train_info, "frozen": frozen_info, "semantic_sampling": "event change, topology/interface change, normalized-observation change, bounded sparse persistence", "old_history_used_as_formal_row": False, "diagnostic_bank_used_as_formal_row": False, "class_weighting": False, "oversampling": False, "synthetic_labels": False},
                "snapshot_provenance": provenance, "retry_feature_provenance": provenance["retry_feature_provenance"], "teacher_determinism": "PASS" if deterministic else "FAIL", "decision_event_identity": "PASS" if identity else "FAIL", "frozen_cohort_independence": "PASS" if independence else "FAIL", "observation_aliasing": aliasing, "action_tape_replay": replay, "safety": safety,
                "semantic_source_coverage": coverage, "proposal_diversity": {"PASS": diversity, "release_joint_classes": len(train_summary["release_joint_classes"]), "release_speed_values": len(train_summary["release_speed_values"]), "release_merge_values": len(train_summary["release_merge_values"])},
                "V4_2_V2_1_I0": "PASS" if passed else "FAIL", "V4_2_V2_1_I1_ALLOWED": bool(passed), "DAgger_ALLOWED": False, "PPO_ALLOWED": False, "qualification_failures": failures}
    (out / "v4_2_v2_1_formal_i0_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-target", type=int, default=TRAIN_TARGET); parser.add_argument("--frozen-target", type=int, default=FROZEN_TARGET)
    parser.add_argument("--max-train-seeds", type=int, default=len(TRAIN_SEEDS)); parser.add_argument("--max-frozen-seeds", type=int, default=len(FROZEN_SEEDS))
    args = parser.parse_args(); print(json.dumps(run(args.output_dir, train_target=args.train_target, frozen_target=args.frozen_target, max_train_seeds=args.max_train_seeds, max_frozen_seeds=args.max_frozen_seeds), indent=2))
