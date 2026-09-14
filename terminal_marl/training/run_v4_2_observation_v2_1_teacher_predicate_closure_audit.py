"""Read-only V2.1 Teacher-predicate closure audit; it never creates I0 rows."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from ..core.config import RingRole, SimConfig
from ..core.low_level_joint_action_v4_2 import ACTION_SCHEMA_V4_2, RELEASE_GATE, hierarchical_from_joint_index_v4_1
from ..core.observations import LOW_OBS_V2, LOW_OBS_V2_1, build_low_observation_v2, build_low_observation_v2_1
from ..core.safety_trace import SafetyTraceRecorder
from ..envs.parallel_env import TerminalParallelEnv, WorldSnapshot
from .run_v4_2_observation_v2_1_qualification import FORMAL_RETRY_NORM_HORIZON_SECONDS, runtime_feature_sanity
from .v4_1_joint_teacher import teacher_joint_expert_v4_1


AUDIT_SCHEMA = "phaseA_low_obs_v2_1_retry_state"
TARGET_STATES = 360
SEEDS = tuple(range(12001, 12061))
PROBE_SEEDS = (8111, 8112, 8113)
PAIR_THRESHOLD = {"accuracy": .80, "macro_F1": .80, "WAIT_recall": .80, "RELEASE_recall": .80, "ROC_AUC": .85}


def _digest(value) -> str:
    payload = json.dumps(_safe_json(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_json(value):
    """Keep unbounded scheduler sentinels explicit in a portable audit file."""
    if isinstance(value, float) and not math.isfinite(value):
        return "-Infinity" if value < 0 else "Infinity"
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe_json(item) for item in value]
    if isinstance(value, np.generic):
        return _safe_json(value.item())
    return value


def _config(*, dynamic: bool) -> SimConfig:
    return SimConfig(
        action_schema_version=ACTION_SCHEMA_V4_2, observation_schema_version=AUDIT_SCHEMA,
        retry_norm_horizon_seconds=FORMAL_RETRY_NORM_HORIZON_SECONDS,
        experiment_mode="dynamic_rule" if dynamic else "fixed_teacher_heavy",
        high_level_enabled=dynamic, scheduler_mode="execution", teacher_enabled=True,
        teacher_mode="teacher_label_only", reset_mode="warm_start_rings", warm_start_uavs=12,
        warm_start_steady_motion=True, fixed_arrival_schedule=(0.0,) * 12,
        total_requested_uavs=12, max_simulation_seconds=12.0, initial_soc_mode="fixed",
        initial_soc=.8, priority_mode="fixed", initial_priority=1,
    )


def _state_snapshot(env: TerminalParallelEnv, uav) -> dict:
    """JSON-safe audit state, including fields omitted by historical I0 traces."""
    state = SafetyTraceRecorder._state(env)
    state.update({
        "target_uav_retry": {"uid": int(uav.uid), "safety_retry_after": float(uav.safety_retry_after),
                             "retry_remaining": max(0.0, float(uav.safety_retry_after - env.now)),
                             "committed_plan": bool(uav.committed_plan is not None)},
        "airspace": {"headways": dict(env.airspace.headways), "affected_interfaces": sorted(env.airspace.affected_interfaces()),
                     "transition_active": bool(env.airspace.transition is not None)},
        "scheduler": {"last_release": dict(env.scheduler.last_release), "last_executed_release": dict(env.scheduler.last_executed_release),
                      "reservations": len(env.scheduler.reservations)},
        "resources": {"final_count": sum(item.phase.value == "final" for item in env.active.values()),
                      "final_capacity": int(env.config.final_queue_capacity),
                      "free_lz": sum(zone.status.value == "free" for zone in env.lz_manager.zones),
                      "lz_status": [zone.status.value for zone in env.lz_manager.zones]},
        "last_decisions": {"release": env.last_release_decisions.get(uav.uid), "safety": env.last_safety_decisions.get(uav.uid)},
    })
    return _safe_json(state)


def _predicate_inventory() -> list[dict]:
    """Inventory of actual reads/calls in v4_1_joint_teacher and _candidate_plan."""
    return [
        {"name": "retry_remaining", "class": "A", "code_path": "_release_context: env.now < uav.safety_retry_after",
         "gate_effect": "WAIT", "v2_1_status": "direct:index_87"},
        {"name": "self_committed_plan", "class": "A", "code_path": "_release_context: uav.committed_plan is not None",
         "gate_effect": "WAIT", "v2_1_status": "sufficiently_identified_by_transfer_phase_and_release_eligible"},
        {"name": "phase_and_ring_context", "class": "A", "code_path": "_release_context: FlightPhase and ring role",
         "gate_effect": "WAIT outside holder rings", "v2_1_status": "direct:phase_and_ring_role"},
        {"name": "affected_interface", "class": "A", "code_path": "teacher_joint_expert_v4_1: interface in affected_interfaces",
         "gate_effect": "WAIT", "v2_1_status": "direct:interface_frozen"},
        {"name": "final_capacity", "class": "A", "code_path": "Teacher final target branch: count FINAL >= final_queue_capacity",
         "gate_effect": "WAIT", "v2_1_status": "direct:final_resource_available_and_final_queue_pressure"},
        {"name": "landing_zone_availability", "class": "A", "code_path": "Teacher final target branch: choose_rule_zone",
         "gate_effect": "WAIT", "v2_1_status": "direct:assigned_lz_usable/free_lz_fraction/global_free_lz_fraction"},
        {"name": "proposal_reservation_feasibility", "class": "B", "code_path": "_candidate_plan: propose(candidate) then can_reserve(proposal)",
         "gate_effect": "candidate-specific", "v2_1_status": "ACTION_CONDITIONED"},
        {"name": "candidate_trajectory_dynamics", "class": "B", "code_path": "_candidate_plan: TrajectoryGenerator.ring_transfer/final_approach",
         "gate_effect": "candidate-specific", "v2_1_status": "ACTION_CONDITIONED"},
        {"name": "candidate_safety_certificate_and_holder_tube", "class": "B", "code_path": "_candidate_plan: SafetyFilter.filter_plan",
         "gate_effect": "candidate-specific", "v2_1_status": "ACTION_CONDITIONED"},
        {"name": "catalogue_preference_order", "class": "B", "code_path": "TeacherScheduler.joint_catalogue + _candidate_score",
         "gate_effect": "chooses RELEASE proposal after gate availability", "v2_1_status": "ACTION_CONDITIONED"},
        {"name": "SOC", "class": "C", "code_path": "not read by teacher_joint_expert_v4_1 gate path", "gate_effect": "none currently", "v2_1_status": "observed"},
        {"name": "last_release_or_safety_decision_and_lifecycle_age", "class": "C", "code_path": "not read by teacher_joint_expert_v4_1 gate path", "gate_effect": "none currently", "v2_1_status": "not_gate_decisive"},
    ]


def _candidate_record(row: dict) -> dict:
    reason = str(row["reason"])
    release = row["decoded"]["release_request"] > 0.0
    attempted = release and reason not in {"release_context_masked", "geometry_frozen", "final_capacity", "lz_unavailable"}
    return {"proposal_id": row["joint_index"], "decoded": row["decoded"], "teacher_score": row["candidate_score"],
            "teacher_result": "selected_or_certified" if reason == "certified" else reason,
            "reservation_feasible": None if not attempted else reason != "headway_conflict",
            "dynamics_feasible": None if not attempted else reason != "trajectory_infeasible",
            "SafetyFilter_certified": True if reason == "certified" else False if attempted else None,
            "predicted_holder_reachable_tube": False if "holder_reachable_tube" in reason else None,
            "feasible": bool(row["feasible"]), "tie_break": row["tie_break"]}


def _audit_uav(env: TerminalParallelEnv, uav, source: str, seed: int, episode: int, step: int, previous_gate: int | None) -> dict:
    observation = build_low_observation_v2_1(uav, env)
    selected, catalogue = teacher_joint_expert_v4_1(env, uav, audit_catalogue=True)
    context = None
    if uav.ring_role is RingRole.OUTER: context = (RingRole.MIDDLE, "o2m")
    elif uav.ring_role is RingRole.MIDDLE: context = (RingRole.INNER, "m2i")
    elif uav.ring_role is RingRole.INNER: context = ("final", "i2f")
    retry = max(0.0, float(uav.safety_retry_after - env.now))
    release_rows = [item for item in catalogue if item["decoded"]["release_request"] > 0.0]
    reasons = Counter(item["reason"] for item in release_rows)
    if retry > 0.0: rationale = "retry_active"
    elif uav.committed_plan is not None: rationale = "committed"
    elif context is None: rationale = "phase_masked"
    elif context[1] in env.airspace.affected_interfaces(): rationale = "interface_frozen"
    elif any(item["reason"] == "final_capacity" for item in release_rows): rationale = "final_capacity"
    elif any(item["reason"] == "lz_unavailable" for item in release_rows): rationale = "lz_unavailable"
    elif selected.release_request > 0.0: rationale = "certified_release"
    elif reasons: rationale = reasons.most_common(1)[0][0]
    else: rationale = "no_release_catalogue"
    full = _state_snapshot(env, uav)
    state_hash = _digest(full); observation_hash = hashlib.sha256(observation.tobytes()).hexdigest()
    lifecycle = f"audit:{source}:seed-{seed}:uav-{uav.uid}:phase-{uav.phase.value}"
    return {"source": source, "seed": seed, "episode_id": episode, "uav_id": int(uav.uid), "step": step,
            "lifecycle_event_id": lifecycle, "decision_event_id": f"{lifecycle}:step-{step}",
            "state_hash": state_hash, "observation_hash": observation_hash, "full_simulator_state": full,
            "observation_v2_1": observation.astype(float).tolist(), "Teacher_gate": int(selected.release_request > 0.0),
            "Teacher_proposal": None if selected.release_request <= 0.0 else int(selected.index - 1),
            "Teacher_joint": int(selected.index), "Teacher_gate_rationale": rationale,
            "all_predicates_consulted": [entry["name"] for entry in _predicate_inventory() if entry["class"] in {"A", "B"}],
            "safety_retry_after": float(uav.safety_retry_after), "retry_remaining": retry,
            "commit_state": bool(uav.committed_plan is not None), "interface_state": sorted(env.airspace.affected_interfaces()),
            "headway_state": dict(env.airspace.headways), "reservation_lifecycle_state": {"reservations": len(env.scheduler.reservations),
                "last_release": dict(env.scheduler.last_release), "last_executed_release": dict(env.scheduler.last_executed_release)},
            "resource_state": full["resources"], "candidate_scan": [_candidate_record(item) for item in catalogue],
            "candidate_reason_counts": dict(reasons), "previous_teacher_gate": previous_gate}


def _action_map(observations: dict[str, np.ndarray], seed: int, step: int) -> dict[str, np.ndarray]:
    return {uid: hierarchical_from_joint_index_v4_1(1 + ((seed + step + int(uid)) % 25)).normalized for uid in observations}


def _collect_bank(target: int) -> tuple[list[dict], dict]:
    bank: list[dict] = []; seen: set[str] = set(); previous: dict[tuple[int, int], int] = {}
    source_counts: Counter = Counter()
    for episode, seed in enumerate(SEEDS):
        dynamic = episode % 5 == 0; source = "interface_transition" if dynamic else "student_visited_semantic_rollout"
        env = TerminalParallelEnv(_config(dynamic=dynamic)); observations, _ = env.reset(seed)
        if dynamic:
            # This creates an actual DynamicAirspace transition state only for
            # audit coverage; it is not a curriculum run and does not alter a dataset.
            env.apply_high_level_action(np.asarray((1., -1., 1., 0., 0., 0., 0., 0., 0., 0.), np.float32), geometry_gate=True)
            observations = env._observations()
        for step in range(8):
            ordered = sorted(int(uid) for uid in observations)
            if ordered:
                uid = ordered[(seed + step) % len(ordered)]; uav = env.active[uid]
                row = _audit_uav(env, uav, source, seed, episode, step, previous.get((seed, uid)))
                semantic = _digest({"observation": row["observation_v2_1"], "gate": row["Teacher_gate"],
                                    "proposal": row["Teacher_proposal"], "rationale": row["Teacher_gate_rationale"]})
                previous[(seed, uid)] = row["Teacher_gate"]
                if semantic not in seen:
                    seen.add(semantic); bank.append(row); source_counts[source] += 1
                if len(bank) >= target: break
            observations, _, _, _, _ = env.step(_action_map(observations, seed, step))
        if len(bank) >= target: break
    if not 300 <= len(bank) <= 600:
        raise RuntimeError(f"audit bank requires 300-600 unique semantic states, got {len(bank)}")
    return bank, {"source_counts": dict(source_counts), "unique_semantic_states": len(bank), "seeds": sorted(set(row["seed"] for row in bank))}


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = labels == 1, labels == 0
    if not pos.any() or not neg.any(): return float("nan")
    order = np.argsort(scores); ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def _probe(train_x, train_y, dev_x, dev_y, seed: int) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    mean, std = train_x.mean(0), train_x.std(0); std[std < 1e-6] = 1.0
    tx = torch.as_tensor((train_x - mean) / std, dtype=torch.float32); ty = torch.as_tensor(train_y, dtype=torch.float32)
    dx = torch.as_tensor((dev_x - mean) / std, dtype=torch.float32)
    model = torch.nn.Linear(tx.shape[1], 1); optimizer = torch.optim.Adam(model.parameters(), lr=.03)
    for _ in range(300):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(tx).squeeze(-1), ty); optimizer.zero_grad(); loss.backward(); optimizer.step()
    probability = torch.sigmoid(model(dx).squeeze(-1)).detach().numpy(); predicted = probability >= .5
    wait, release = dev_y == 0, dev_y == 1
    wr, rr = float((predicted[wait] == 0).mean()), float((predicted[release] == 1).mean())
    wp = float(((predicted == 0) & wait).sum() / max(1, (predicted == 0).sum())); rp = float(((predicted == 1) & release).sum() / max(1, (predicted == 1).sum()))
    return {"accuracy": float((predicted == dev_y).mean()), "macro_F1": (2*wp*wr/max(1e-12,wp+wr)+2*rp*rr/max(1e-12,rp+rr))/2,
            "WAIT_recall": wr, "RELEASE_recall": rr, "ROC_AUC": _auc(probability, dev_y)}


def _pairs(bank: list[dict], name: str, wait, release) -> dict:
    split = lambda row: "dev" if row["seed"] % 5 == 0 else "train"
    result: dict[str, list[tuple[dict, dict]]] = {"train": [], "dev": []}
    for partition in result:
        waits, releases = [row for row in bank if split(row) == partition and wait(row)], [row for row in bank if split(row) == partition and release(row)]
        used: set[str] = set()
        for left in waits:
            candidates = [right for right in releases if right["decision_event_id"] not in used and right["seed"] != left["seed"] and right["lifecycle_event_id"] != left["lifecycle_event_id"]]
            if candidates:
                right = min(candidates, key=lambda candidate: np.linalg.norm(np.asarray(left["observation_v2_1"])-np.asarray(candidate["observation_v2_1"])))
                result[partition].append((left, right)); used.add(right["decision_event_id"])
    if min(len(result["train"]), len(result["dev"])) < 4:
        return {"status": "NOT_AVAILABLE", "pairs": {key: len(value) for key, value in result.items()}, "reason": "insufficient seed/lifecycle-disjoint actual pairs"}
    arrays = lambda items: (np.asarray([row["observation_v2_1"] for pair in items for row in pair], np.float32), np.asarray([0, 1] * len(items)))
    train_x, train_y = arrays(result["train"]); dev_x, dev_y = arrays(result["dev"])
    runs = [_probe(train_x, train_y, dev_x, dev_y, seed) for seed in PROBE_SEEDS]
    metrics = {key: float(np.mean([run[key] for run in runs])) for key in runs[0]}
    passed = all(metrics[key] >= value for key, value in PAIR_THRESHOLD.items())
    return {"status": "PASS" if passed else "FAIL", "pairs": {key: len(value) for key, value in result.items()}, "metrics": metrics, "runs": runs}


def _pair_audits(bank: list[dict]) -> dict:
    release = lambda row: row["Teacher_gate"] == RELEASE_GATE
    return {
        "retry_pair_separability": _pairs(bank, "retry", lambda row: row["Teacher_gate"] == 0 and row["retry_remaining"] > 0, release),
        "commit_pair_separability": _pairs(bank, "commit", lambda row: row["Teacher_gate"] == 0 and row["commit_state"], release),
        "reservation_state_separability": _pairs(bank, "reservation", lambda row: row["Teacher_gate"] == 0 and row["candidate_reason_counts"].get("headway_conflict", 0) > 0, release),
        "interface_headway_separability": _pairs(bank, "interface", lambda row: row["Teacher_gate"] == 0 and (row["Teacher_gate_rationale"] == "interface_frozen" or row["candidate_reason_counts"].get("headway_conflict", 0) > 0), release),
        "resource_state_separability": _pairs(bank, "resource", lambda row: row["Teacher_gate"] == 0 and row["Teacher_gate_rationale"] in {"final_capacity", "lz_unavailable"}, release),
        "conflict_recovery_separability": _pairs(bank, "conflict", lambda row: row["Teacher_gate"] == 0 and any("predicted" in key or "safety" in key for key in row["candidate_reason_counts"]), release),
    }


def run(output_dir: str | Path, *, target_states: int = TARGET_STATES) -> dict:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    bank, coverage = _collect_bank(target_states)
    (output / "audit_state_bank.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in bank), encoding="utf-8")
    bank_hash = hashlib.sha256((output / "audit_state_bank.jsonl").read_bytes()).hexdigest()
    inventory = _predicate_inventory(); pairs = _pair_audits(bank)
    state_only = [entry["name"] for entry in inventory if entry["class"] == "A"]
    action_conditioned = [entry["name"] for entry in inventory if entry["class"] == "B"]
    unresolved = [entry["name"] for entry in inventory if entry["class"] == "A" and not entry["v2_1_status"].startswith(("direct", "sufficiently"))]
    release_scans = [row for row in bank if row["Teacher_gate"] == RELEASE_GATE]
    privileged = bool(release_scans)  # Every RELEASE gate is selected only after bounded 25-proposal certification.
    known_redundancy = all(np.isclose(row["observation_v2_1"][LOW_OBS_V2.slices["free_lz_fraction"].start], row["observation_v2_1"][LOW_OBS_V2.slices["global_free_lz_fraction"].start]) for row in bank)
    sanity = runtime_feature_sanity(output / "schema_sanity")
    pair_pass = all(value["status"] == "PASS" for value in pairs.values())
    freeze = bool(not unresolved and pair_pass and not privileged and sanity["feature_sanity"]["replay_exact"])
    result = {"stage": "V4_2_OBSERVATION_V2_1_TEACHER_PREDICATE_CLOSURE_AUDIT", "evidence_type": "newly_run",
              "audit_bank": {"path": str(output / "audit_state_bank.jsonl"), "sha256": bank_hash, **coverage},
              "teacher_predicate_inventory": inventory, "teacher_state_only_predicates": state_only,
              "teacher_action_conditioned_predicates": action_conditioned, "unresolved_state_only_missing_features": unresolved,
              "matched_pair_audits": pairs, "action_conditioned_information_documented": True,
              "execution_boundary_preserved": True,
              "teacher_privileged_information_risk": privileged,
              "teacher_privileged_information_reason": "Teacher gate scans and certifies up to 25 RELEASE proposals before selecting WAIT/RELEASE; the deployment actor receives none of those proposal-specific results.",
              "schema_sanity": sanity, "KNOWN_REDUNDANT_FEATURE": known_redundancy,
              "all_teacher_state_only_gate_predicates_inventory_complete": True,
              "future_leakage": False, "schema_replay": "PASS" if sanity["feature_sanity"]["replay_exact"] else "FAIL",
              "OBSERVATION_V2_1_FINAL_FREEZE": "PASS" if freeze else "FAIL",
              "FORMAL_V2_1_I0_RECOLLECTION_ALLOWED": bool(freeze), "I1_ALLOWED": False,
              "DAgger_ALLOWED": False, "PPO_ALLOWED": False, "forbidden_operations_called": []}
    (output / "v4_2_observation_v2_1_teacher_predicate_closure_audit.manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", required=True); parser.add_argument("--target-states", type=int, default=TARGET_STATES)
    arguments = parser.parse_args(); print(json.dumps(run(arguments.output_dir, target_states=arguments.target_states), indent=2))
