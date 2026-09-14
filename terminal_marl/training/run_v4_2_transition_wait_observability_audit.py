"""Read-only observability audit for V4.2 transition_WAIT vs RELEASE states."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from ..core.low_level_joint_action_v4_2 import RELEASE_GATE
from ..core.observations import LOW_OBS_V2
from ..core.safety_trace import SafetyTraceRecorder
from ..envs.parallel_env import TerminalParallelEnv
from .run_v4_2_dagger_d1_conflict_source_attribution import DEV_SEEDS, NEW_SOURCE, _family, _phase
from .run_v4_2_dagger_r1_release_retention_diagnostic import _load_policy, _prediction_details, _sha256
from .run_v4_2_dagger_round1 import _config_for_steps
from .run_v4_2_i1_hierarchical_bc import _arrays, _rows
from .v4_1_joint_teacher import teacher_joint_expert_v4_1


PAIR_QUOTAS = {"clear_outer_RELEASE": {"train": 20, "dev": 10},
               "middle_or_interface_RELEASE": {"train": 21, "dev": 9},
               "frozen_RELEASE_anchor": {"train": 20, "dev": 10}}
CLASSIFIER_SEEDS = (7201, 7202, 7203)


def _digest_rows(rows: list[dict], path: Path) -> str:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return _sha256(path)


def _partition(row: dict, *, frozen: bool = False) -> str:
    # Frozen anchors use their own source seeds; no frozen seed appears in the
    # D1 train/dev seed sets.  Reserve the later anchor seeds for development.
    if frozen:
        return "dev" if int(row["seed"]) >= 2112 else "train"
    return "dev" if int(row["seed"]) in DEV_SEEDS else "train"


def _flips(i1_checkpoint: str | Path, r1_checkpoint: str | Path, frozen: list[dict]) -> list[dict]:
    x, gate, _, _ = _arrays(frozen)
    i1, r1 = _load_policy(i1_checkpoint, learning_rate=.0003), _load_policy(r1_checkpoint, learning_rate=.0003)
    before, after = _prediction_details(i1, x), _prediction_details(r1, x)
    selected = (gate.cpu().numpy() == RELEASE_GATE) & (before["gate"] == RELEASE_GATE) & (after["gate"] != RELEASE_GATE)
    return [frozen[int(index)] for index in np.where(selected)[0]]


def _pair_distance(left: dict, right: dict) -> float:
    a, b = np.asarray(left["observation"], np.float32), np.asarray(right["observation"], np.float32)
    return float(np.linalg.norm(a - b) / np.sqrt(a.size))


def _matched_pairs(i1_checkpoint: str | Path, r1_checkpoint: str | Path, new: list[dict], frozen: list[dict]) -> tuple[list[dict], list[dict]]:
    waits = [row for row in new if _family(row) == "transition_WAIT"]
    pools = {"clear_outer_RELEASE": [row for row in new if _family(row) == "clear_outer_RELEASE"],
             "middle_or_interface_RELEASE": [row for row in new if _family(row) == "middle_or_interface_RELEASE"],
             "frozen_RELEASE_anchor": [row for row in frozen if row["gate_target"] == RELEASE_GATE]}
    flips = _flips(i1_checkpoint, r1_checkpoint, frozen)
    priority = {row["decision_event_id"] for row in sorted(waits, key=lambda item: min(_pair_distance(item, flip) for flip in flips))[:12]}
    used_wait: set[str] = set(); used_release: set[str] = set(); pairs: list[dict] = []
    for kind, quotas in PAIR_QUOTAS.items():
        for partition, count in quotas.items():
            candidates = [row for row in waits if _partition(row) == partition and row["decision_event_id"] not in used_wait]
            releases = [row for row in pools[kind] if _partition(row, frozen=kind == "frozen_RELEASE_anchor") == partition and row["decision_event_id"] not in used_release]
            for _ in range(count):
                ranked = sorted(((0 if wait["decision_event_id"] in priority else 1, _pair_distance(wait, release), wait, release)
                                 for wait in candidates for release in releases), key=lambda value: (value[0], value[1]))
                if not ranked:
                    raise RuntimeError(f"insufficient {kind}/{partition} pair candidates")
                _, distance, wait, release = ranked[0]
                pairs.append({"pair_id": len(pairs), "split": partition, "release_kind": kind, "distance": distance,
                              "wait_id": wait["decision_event_id"], "release_id": release["decision_event_id"],
                              "wait": wait, "release": release, "priority_near_frozen_flip": wait["decision_event_id"] in priority})
                used_wait.add(wait["decision_event_id"]); used_release.add(release["decision_event_id"])
                candidates = [row for row in candidates if row["decision_event_id"] != wait["decision_event_id"]]
                releases = [row for row in releases if row["decision_event_id"] != release["decision_event_id"]]
    if not 50 <= len(pairs) <= 100:
        raise AssertionError(f"matched pair count outside protocol: {len(pairs)}")
    return pairs, flips


def _event_step(row: dict) -> int:
    return int(row["decision_event_id"].split(":step-")[1].split(":")[0])


def _teacher_rationale(env: TerminalParallelEnv, uid: int) -> dict:
    uav = env.active[uid]; selected, audit = teacher_joint_expert_v4_1(env, uav, audit_catalogue=True)
    releases = [row for row in audit if row["decoded"]["release_request"] > 0.0]
    reason_counts = Counter(row["reason"] for row in releases)
    retry_remaining = max(0.0, float(uav.safety_retry_after - env.now))
    if selected.release_request > 0.0:
        rationale = "certified_release_candidate"
    elif retry_remaining > 0.0:
        rationale = "retry_cooldown_mask"
    elif uav.committed_plan is not None:
        rationale = "committed_plan_mask"
    elif releases:
        rationale = reason_counts.most_common(1)[0][0]
    else:
        rationale = "phase_context_mask"
    full = {"now": env.now, "step": env.step_count, "phase": uav.phase.value, "ring_role": None if uav.ring_role is None else uav.ring_role.value,
            "committed_plan": uav.committed_plan is not None, "safety_retry_after": float(uav.safety_retry_after), "retry_remaining": retry_remaining,
            "soc": float(uav.soc), "phase_entry_time": float(uav.phase_entry_time), "time_since_lifecycle_transition": float(env.now - uav.phase_entry_time),
            "last_release_decision": env.last_release_decisions.get(uid), "last_safety_decision": env.last_safety_decisions.get(uid),
            "affected_interfaces": sorted(env.airspace.affected_interfaces()), "headways": dict(env.airspace.headways),
            "scheduler_last_executed_release": dict(env.scheduler.last_executed_release), "reservation_count": len(env.scheduler.reservations),
            "future_release_reservation_count": len(env.future_release_reservations), "final_queue_count": sum(item.phase.value == "final" for item in env.active.values()),
            "free_lz_count": sum(zone.status.value == "free" for zone in env.lz_manager.zones),
            "active_uav_count": len(env.active), "teacher_candidate_reason_counts": dict(reason_counts)}
    return {"teacher_joint": selected.index, "teacher_gate": int(selected.release_request > 0.0), "selected_action": selected.as_dict(),
            "gate_rationale": rationale, "catalogue_audit": audit, "full_state_teacher_access": full}


def _replay_teacher_audits(pairs: list[dict], tape_dir: str | Path) -> dict[str, dict]:
    wanted = {(int(item["seed"]), _event_step(item), int(item["uav_id"])) for pair in pairs for item in (pair["wait"], pair["release"])
              if item.get("source") == NEW_SOURCE}
    result: dict[str, dict] = {}
    for path in Path(tape_dir).glob("*.json"):
        tape = json.loads(path.read_text(encoding="utf-8")); seed = int(tape["episode_seed"])
        if not any(key[0] == seed for key in wanted):
            continue
        env = TerminalParallelEnv(_config_for_steps(10)); observations, _ = env.reset(seed)
        for record in tape.get("records", []):
            step = int(record["step"])
            for key in [key for key in wanted if key[0] == seed and key[1] == step]:
                uid = key[2]
                if uid in env.active:
                    result_key = f"{seed}:{step}:{uid}"; result[result_key] = _teacher_rationale(env, uid)
            actions = {str(uid): np.asarray(value, dtype=np.float32) for uid, value in record["a_requested"].items()}
            observations, _, _, _, _ = env.step(actions)
    if len(result) != len(wanted):
        raise RuntimeError(f"teacher replay shortfall: {len(result)} / {len(wanted)}")
    return result


def _frozen_rationale(row: dict) -> dict:
    snapshot = row.get("state_snapshot", {}); uid = int(row["uav_id"])
    uav = next((item for item in snapshot.get("uavs", []) if int(item.get("uid", -1)) == uid), {})
    return {"teacher_joint": row.get("teacher_semantic_joint_index_v4_1"), "teacher_gate": RELEASE_GATE,
            "gate_rationale": "stored_teacher_RELEASE_anchor_candidate_certificate_not_replayed_from_snapshot",
            "full_state_teacher_access": {"stored_snapshot": snapshot, "target_uav": uav},
            "catalogue_audit": None}


def _observation_mapping() -> dict:
    present = {feature.name: {"slice": [LOW_OBS_V2.slices[feature.name].start, LOW_OBS_V2.slices[feature.name].stop], "size": feature.size}
               for feature in LOW_OBS_V2.features}
    return {"present": present,
            "teacher_variable_status": {
                "phase_ring_role": "present", "committed_or_reserved_self": "indirectly_encoded", "soc": "present",
                "affected_interface": "present", "final_capacity": "present", "landing_zone_availability": "present",
                "headway_since_last_executed_release": "present", "target_predecessor_successor": "present",
                "nearest_predicted_conflict": "present", "safety_retry_after_or_retry_remaining": "absent",
                "reservation_lifecycle_exact_can_reserve": "indirectly_encoded", "candidate_trajectory_safety_certificate": "indirectly_encoded",
                "recent_rejection_reason": "absent", "previous_gate_or_action": "absent",
                "time_since_lifecycle_transition": "absent", "post_WAIT_accumulated_eligibility": "indirectly_encoded"}}


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positive, negative = labels == 1, labels == 0
    if not positive.any() or not negative.any(): return float("nan")
    order = np.argsort(scores); ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[positive].sum() - positive.sum() * (positive.sum() + 1) / 2) / (positive.sum() * negative.sum()))


def _binary_run(train_x: np.ndarray, train_y: np.ndarray, dev_x: np.ndarray, dev_y: np.ndarray, *, seed: int) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    mean, std = train_x.mean(0), train_x.std(0); std[std < 1e-6] = 1.0
    tx = torch.as_tensor((train_x - mean) / std, dtype=torch.float32); ty = torch.as_tensor(train_y, dtype=torch.float32)
    dx = torch.as_tensor((dev_x - mean) / std, dtype=torch.float32)
    model = torch.nn.Linear(tx.shape[1], 1); optimizer = torch.optim.Adam(model.parameters(), lr=.03)
    for _ in range(300):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(tx).squeeze(-1), ty)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    probability = torch.sigmoid(model(dx).squeeze(-1)).detach().numpy(); prediction = probability >= .5
    wait, release = dev_y == 0, dev_y == 1
    wait_recall = float((prediction[wait] == 0).mean()); release_recall = float((prediction[release] == 1).mean())
    p_wait = float(((prediction == 0) & wait).sum() / max(1, (prediction == 0).sum())); p_release = float(((prediction == 1) & release).sum() / max(1, (prediction == 1).sum()))
    f1_wait = 2 * p_wait * wait_recall / max(1e-12, p_wait + wait_recall); f1_release = 2 * p_release * release_recall / max(1e-12, p_release + release_recall)
    return {"accuracy": float((prediction == dev_y).mean()), "macro_F1": (f1_wait + f1_release) / 2, "WAIT_recall": wait_recall,
            "RELEASE_recall": release_recall, "ROC_AUC": _auc(probability, dev_y)}


def _classifier(pairs: list[dict]) -> dict:
    train_x: list[np.ndarray] = []; train_y: list[int] = []; dev_x: list[np.ndarray] = []; dev_y: list[int] = []
    for pair in pairs:
        target_x, target_y = (dev_x, dev_y) if pair["split"] == "dev" else (train_x, train_y)
        target_x.extend([np.asarray(pair["wait"]["observation"], np.float32), np.asarray(pair["release"]["observation"], np.float32)]); target_y.extend([0, 1])
    train, dev = np.asarray(train_x), np.asarray(dev_x); ty, dy = np.asarray(train_y), np.asarray(dev_y)
    full_runs = [_binary_run(train, ty, dev, dy, seed=seed) for seed in CLASSIFIER_SEEDS]
    full = {key: float(np.mean([run[key] for run in full_runs])) for key in full_runs[0]}
    groups = {}
    for feature in LOW_OBS_V2.features:
        section = slice(LOW_OBS_V2.slices[feature.name].start, LOW_OBS_V2.slices[feature.name].stop)
        runs = [_binary_run(train[:, section], ty, dev[:, section], dy, seed=seed) for seed in CLASSIFIER_SEEDS]
        groups[feature.name] = {key: float(np.mean([run[key] for run in runs])) for key in runs[0]}
    ranked = sorted(groups, key=lambda key: groups[key]["ROC_AUC"], reverse=True)
    return {"pairs": {"train": len(train_y) // 2, "dev": len(dev_y) // 2, "train_samples": len(train_y), "dev_samples": len(dev_y)},
            "full": full, "full_runs": full_runs, "single_feature_group_ablation": groups, "top_feature_groups_by_AUC": ranked[:8]}


def _root(audits: dict[str, dict], classifier: dict) -> tuple[str, list[str], list[str], bool, bool]:
    rationales = list(audits.values())
    retry = sum(item["gate_rationale"] == "retry_cooldown_mask" for item in rationales)
    reason_counts = Counter(item["gate_rationale"] for item in rationales)
    # Current Teacher code reads retry/cooldown directly.  Recent rejection,
    # previous action, and lifecycle elapsed time are absent from V2 but are
    # not direct predicates in this Teacher implementation, so they remain
    # documented mapping gaps rather than being claimed as decisive here.
    missing = ["safety_retry_after_or_retry_remaining"] if retry else []
    weak = ["reservation_lifecycle_exact_can_reserve", "candidate_trajectory_safety_certificate"]
    full = classifier["full"]; separable = bool(full["accuracy"] >= .80 and full["macro_F1"] >= .80 and full["WAIT_recall"] >= .80 and full["RELEASE_recall"] >= .80 and full["ROC_AUC"] >= .85)
    if missing:
        root = "MISSING_DECISION_STATE_IN_OBSERVATION"
    elif separable:
        root = "DAGGER_AGGREGATION_INTERFERENCE"
    else:
        root = "WEAK_OBSERVATION_REPRESENTATION"
    decisive = [f"teacher_gate_rationale:{name}={count}" for name, count in reason_counts.most_common()]
    return root, decisive, missing, root in {"MISSING_DECISION_STATE_IN_OBSERVATION", "WEAK_OBSERVATION_REPRESENTATION"}, root == "DAGGER_AGGREGATION_INTERFERENCE"


def run(i1_checkpoint: str | Path, r1_checkpoint: str | Path, d1_path: str | Path, frozen_cohort: str | Path, tape_dir: str | Path, output_dir: str | Path) -> dict:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    rows, frozen = _rows(d1_path), _rows(frozen_cohort); new = [row for row in rows if row.get("source") == NEW_SOURCE]
    pairs, flips = _matched_pairs(i1_checkpoint, r1_checkpoint, new, frozen)
    replayed = _replay_teacher_audits(pairs, tape_dir)
    audits: dict[str, dict] = {}
    for pair in pairs:
        for side in ("wait", "release"):
            row = pair[side]
            if row.get("source") == NEW_SOURCE:
                audits[row["decision_event_id"]] = replayed[f"{int(row['seed'])}:{_event_step(row)}:{int(row['uav_id'])}"]
            else:
                audits[row["decision_event_id"]] = _frozen_rationale(row)
    classifier = _classifier(pairs)
    root, decisive, missing, obs_revision, dagger_revision = _root(audits, classifier)
    pair_path = output / "transition_wait_matched_pairs.json"; pair_hash = _digest_rows(pairs, pair_path)
    result = {"stage": "V4_2_TRANSITION_WAIT_OBSERVABILITY_DECISION_BOUNDARY_AUDIT", "evidence_type": "newly_run",
              "inputs": {"I1_checkpoint_sha256": _sha256(i1_checkpoint), "R1_checkpoint_sha256": _sha256(r1_checkpoint), "D1_sha256": _sha256(d1_path), "frozen_cohort_sha256": _sha256(frozen_cohort)},
              "matched_pairs": {"path": str(pair_path), "sha256": pair_hash, "count": len(pairs), "priority_waits_near_frozen_flips": sum(pair["priority_near_frozen_flip"] for pair in pairs),
                                "release_kinds": dict(Counter(pair["release_kind"] for pair in pairs)), "split": dict(Counter(pair["split"] for pair in pairs)), "frozen_flip_count": len(flips)},
              "teacher_accessed_full_state_variables": ["phase/ring_role", "committed_plan", "safety_retry_after", "soc", "affected_interfaces", "final_queue_capacity", "LZ availability", "scheduler reservations/can_reserve", "candidate trajectory feasibility", "SafetyFilter certificate", "target predecessor/successor and committed plans", "lifecycle timing", "last release/safety decision"],
              "observation_v2_mapping": _observation_mapping(), "teacher_audits": audits, "pairwise_classifier": classifier,
              "ROOT_CAUSE": root, "transition_wait_release_observation_separable": bool(classifier["full"]["accuracy"] >= .80 and classifier["full"]["macro_F1"] >= .80 and classifier["full"]["WAIT_recall"] >= .80 and classifier["full"]["RELEASE_recall"] >= .80 and classifier["full"]["ROC_AUC"] >= .85),
              "decisive_teacher_variables": decisive, "missing_from_observation": missing, "weakly_encoded": ["reservation_lifecycle_exact_can_reserve", "candidate_trajectory_safety_certificate"],
              "observation_revision_required": obs_revision, "dagger_protocol_revision_required": dagger_revision,
              "V4_2_DAGGER_R1": "FAIL", "ROUND1_RERUN_ALLOWED": False, "ROUND2_ALLOWED": False, "PPO_ALLOWED": False, "forbidden_operations_called": []}
    (output / "v4_2_transition_wait_observability_audit.manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--i1-checkpoint", required=True); parser.add_argument("--r1-checkpoint", required=True)
    parser.add_argument("--d1", required=True); parser.add_argument("--frozen-cohort", required=True); parser.add_argument("--tape-dir", required=True); parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(); print(json.dumps(run(args.i1_checkpoint, args.r1_checkpoint, args.d1, args.frozen_cohort, args.tape_dir, args.output_dir), indent=2))
