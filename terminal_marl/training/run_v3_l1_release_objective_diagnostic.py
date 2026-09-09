"""Offline V3 L1 release-objective and decision-event diagnostic.

This module never calls :meth:`SharedPPO.update`.  The completed coverage
bridge did not persist per-tick transitions, so it reconstructs only rollout
slices whose *pre-update* actor is exactly available as a saved checkpoint:
update 1 from ``baseline.pt`` and update 4 from ``update3.pt``.  In
particular, update 1 is an exact replay of the reported 7,885 R->W ticks.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..algo.shared_ppo import SharedPPO
from ..core.checkpoints import load_checkpoint
from ..core.low_level_action import normalized_to_indices
from ..core.observations import LOW_OBS_V2
from ..envs.parallel_env import TerminalParallelEnv
from .run_v3_l0_teacher_dataset import _counterfactual_release, _label_after_step
from .run_v3_l1_coverage_bridge_pilot import LAMBDA, TARGETED_EPISODES, _targeted_warm_config
from .run_v3_l1_release_intervention_diagnostic import evaluate_release_cohort
from .run_v3_l1_scenario_mixture_calibration import TARGET_KINDS, _target_sources
from .run_v3_l1_short_pilot import _episode_design
from .train_low_v2_ppo import _read_seed_partition, l1_full_teacher_config


REPLAY_SLICES = ((1, "baseline"), (4, "update3"))
PROBE_STEPS = 96
PROBE_BATCH = 256


def _json(path: Path, value: dict) -> None:
    def _default(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        raise TypeError(f"unsupported JSON value: {type(item).__name__}")

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=_default), encoding="utf-8")
    temporary.replace(path)


def _initial_policy() -> SharedPPO:
    torch.manual_seed(2401); np.random.seed(2401)
    return SharedPPO(LOW_OBS_V2.dimension, teacher_bc_weight=.05)


def _load_policy(path: str) -> SharedPPO:
    policy = SharedPPO(LOW_OBS_V2.dimension, teacher_bc_weight=.05)
    root = l1_full_teacher_config(medium_rate=LAMBDA, reset_mode="warm_start_rings", demand_model="finite_cohort",
                                  warm_start_uavs=8, cohort_uavs=8, horizon_seconds=1200.0)
    load_checkpoint(path, policy, root)
    return policy


def _state_hash(observation: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(observation, dtype=np.float32).tobytes()).hexdigest()


def _interface_for(uav) -> str | None:
    from ..core.config import RingRole
    return {RingRole.OUTER: "o2m", RingRole.MIDDLE: "m2i", RingRole.INNER: "i2f"}.get(uav.ring_role)


def _release_full_state(env: TerminalParallelEnv, uav) -> dict:
    """Deployment-visible state used to attribute a V2 alias, never a label."""
    interface = _interface_for(uav)
    reservations = []
    for item in env.scheduler.reservations.values():
        target = item.target_stage
        item_interface = {"ring_middle": "o2m", "ring_inner": "m2i", "final": "i2f"}.get(target)
        if item_interface == interface:
            reservations.append({"owner": item.uav_id, "status": item.status, "merge_time": item.merge_time,
                                 "time_to_merge": item.merge_time - env.now, "expires_in": item.expires_at - env.now,
                                 "predecessor": item.predecessor_id, "successor": item.successor_id})
    active = []
    for item in sorted(env.active.values(), key=lambda row: int(row.uid)):
        plan = item.committed_plan
        active.append({"uid": item.uid, "phase": item.phase.value,
                       "ring_role": None if item.ring_role is None else item.ring_role.value,
                       "position": [float(value) for value in item.position],
                       "velocity": [float(value) for value in item.velocity],
                       "ring_progress": item.ring_progress, "merge_reservation_id": item.merge_reservation_id,
                       "retry_remaining": max(0.0, item.safety_retry_after - env.now),
                       "committed_plan_id": None if plan is None else plan.plan_id,
                       "committed_start": None if plan is None else plan.start_time,
                       "committed_end": None if plan is None else plan.end_time,
                       "committed_reservations": [] if plan is None else list(plan.reservation_ids)})
    all_reservations = sorted(({"owner": item.uav_id, "target": item.target_stage, "status": item.status,
                                "merge_time": item.merge_time, "time_to_merge": item.merge_time - env.now,
                                "predecessor": item.predecessor_id, "successor": item.successor_id}
                               for item in env.scheduler.reservations.values()),
                              key=lambda item: (item["merge_time"], item["owner"]))
    pending = env.future_release_reservations.get(uav.uid)
    committed = uav.committed_plan
    def _finite_or_none(value):
        value = float(value)
        return value if np.isfinite(value) else None

    return {
        "simulator": {"now": env.now, "active_count": len(env.active), "final_count": sum(x.phase.value == "final" for x in env.active.values()),
                      "interface": interface, "headway": None if interface is None else env.airspace.headways[interface],
                      "last_executed_release": None if interface is None else _finite_or_none(env.scheduler.last_executed_release[interface]),
                      "last_tentative_release": None if interface is None else _finite_or_none(env.scheduler.last_release[interface])},
        "lifecycle": {"phase": uav.phase.value, "ring_role": None if uav.ring_role is None else uav.ring_role.value,
                      "committed": committed is not None, "phase_entry_time": uav.phase_entry_time,
                      "release_retry_remaining": max(0.0, uav.safety_retry_after - env.now),
                      "merge_reservation_id": uav.merge_reservation_id},
        "scheduler": {"all_reservations": all_reservations,
                      "last_release": {key: _finite_or_none(value) for key, value in env.scheduler.last_release.items()},
                      "last_executed_release": {key: _finite_or_none(value) for key, value in env.scheduler.last_executed_release.items()}},
        "reservation": {"target_interface_reservations": sorted(reservations, key=lambda item: (item["merge_time"], item["owner"])),
                        "own_future_release_pending": pending is not None,
                        "own_future_release_not_before": None if pending is None else pending.get("not_before"),
                        "own_future_release_target": None if pending is None else str(pending.get("target"))},
        "active_traffic": active,
        "trajectory": {"committed_plan_id": None if committed is None else committed.plan_id,
                       "committed_start": None if committed is None else committed.start_time,
                       "committed_end": None if committed is None else committed.end_time,
                       "committed_reservations": [] if committed is None else list(committed.reservation_ids)},
    }


def _opposite_hash_pairs(records: list[dict], *, limit: int = 12) -> list[dict]:
    groups: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in records:
        if row["eligible"]: groups[row["observation_hash"]][row["all_event_teacher"]].append(row)
    result = []
    for state_hash, labels in groups.items():
        if set(labels) != {0, 1}: continue
        wait, release = labels[0][0], labels[1][0]
        result.append({"observation_hash": state_hash, "WAIT": {"update": wait["update"], "episode": wait["episode"], "uid": wait["uid"],
                        "tick": wait["tick"], "teacher_reason": wait["teacher_reason"], "native_decision": wait["native_decision"], "full_state": wait["full_state"]},
                       "RELEASE": {"update": release["update"], "episode": release["episode"], "uid": release["uid"],
                                   "tick": release["tick"], "teacher_reason": release["teacher_reason"], "native_decision": release["native_decision"], "full_state": release["full_state"]}})
    return result[:limit]


def _task_specs(pilot: dict, raw_dataset: str | Path, teacher_dataset: str | Path) -> list[dict]:
    """Build the exact update-1/update-4 task identities from frozen pilot rules."""
    sources, partition = _target_sources(raw_dataset), _read_seed_partition(teacher_dataset)
    specs: list[dict] = []
    for update, snapshot in REPLAY_SLICES:
        natural_before = sum(position < (update - 1) * 4 for position in range((update - 1) * 4) if position not in TARGETED_EPISODES)
        natural_cursor = natural_before
        for local_episode in range(4):
            global_episode = (update - 1) * 4 + local_episode
            if global_episode in TARGETED_EPISODES:
                kind = TARGET_KINDS[global_episode % len(TARGET_KINDS)]
                source = sources[kind][0]
                config = _targeted_warm_config(source, horizon_seconds=1200.0)
                seed, source_name = int(source["seed"]), f"targeted:{kind}"
            else:
                config, _ = _episode_design(global_episode, medium_rate=LAMBDA, horizon_seconds=1200.0)
                seed, source_name = partition.train[natural_cursor % len(partition.train)], "natural_l1"
                natural_cursor += 1
            specs.append({"update": update, "snapshot": snapshot, "checkpoint": pilot["snapshots"][snapshot],
                          "episode": local_episode, "seed": int(seed), "source": source_name, "config": config})
    return specs


def _replay_task(spec: dict) -> dict:
    """Replay one pre-update actor, while adding non-mutating teacher probes."""
    torch.set_num_threads(1)
    seed, config = spec["seed"], spec["config"]
    torch.manual_seed(seed); np.random.seed(seed)
    policy = _load_policy(spec["checkpoint"])
    env = TerminalParallelEnv(config); observations, _ = env.reset(seed)
    records: list[dict] = []
    for tick in range(1200):
        ids = list(observations)
        if not ids:
            observations, _, _, _, _ = env.step({})
            continue
        actor_obs = np.stack([observations[uid] for uid in ids]).astype(np.float32, copy=False)
        batch = policy.act_details(actor_obs, actor_obs, deterministic=False)
        actions = {uid: action for uid, action in zip(ids, batch.normalized)}
        # Counterfactual labels are required only for B/C's *legal* release
        # decision events.  Probing phase-masked rows is both semantically
        # irrelevant and unnecessarily expensive in dense traffic.
        eligibility = {uid: bool(actor_obs[index, LOW_OBS_V2.slices["release_eligible"].start] > .5)
                       for index, uid in enumerate(ids)}
        full_states = {uid: _release_full_state(env, env.active[int(uid)]) for uid in ids if int(uid) in env.active}
        probes = {
            uid: (_counterfactual_release(env, env.active[int(uid)], _intent(action))
                  if eligibility[uid] and action[0] <= config.release_threshold and int(uid) in env.active
                  else (None, "native_release_trace" if action[0] > config.release_threshold else "not_release_eligible"))
            for uid, action in actions.items()
        }
        next_observations, _, terminated, truncated, info = env.step(actions)
        for index, uid in enumerate(ids):
            full_state = full_states.get(uid)
            trace = info.get("action_traces", {}).get(uid, {})
            requested = np.asarray(trace.get("requested_low_action", actions[uid]), dtype=np.float32)
            requested_index = int(normalized_to_indices(requested)[0])
            native = trace.get("teacher_corrected_action")
            native_target = requested if native is None else np.asarray(native, dtype=np.float32)
            native_index = int(normalized_to_indices(native_target)[0])
            teacher_used = bool(trace.get("teacher_used", False))
            full_teacher, reason = _label_after_step(env, uid, _intent(requested), probes[uid], trace,
                                                      info.get("release_decisions", {}).get(uid))
            full_index = int(normalized_to_indices(np.asarray((full_teacher.release_request, full_teacher.speed_residual,
                                                                full_teacher.merge_time_residual), dtype=np.float32))[0])
            observation = actor_obs[index].copy()
            eligible = eligibility[uid]
            records.append({
                "update": spec["update"], "episode": spec["episode"], "seed": seed, "source": spec["source"],
                "tick": tick, "uid": str(uid), "phase": int(np.argmax(observation[LOW_OBS_V2.slices["phase"]])),
                "observation": observation.tolist(), "observation_hash": _state_hash(observation),
                "eligible": eligible, "requested": requested_index, "native_teacher": native_index,
                "all_event_teacher": full_index, "teacher_used": teacher_used,
                "teacher_action_present": native is not None,
                "native_release_corrected": bool(teacher_used and requested_index != native_index),
                "teacher_reason": reason, "native_decision": info.get("release_decisions", {}).get(uid), "full_state": full_state,
                "done": bool(terminated.get(uid, False) or truncated.get(uid, False)),
            })
        observations = next_observations
        if env.arrivals.exhausted() and not env.active and not env.admission.queue.items and not env.service_uavs:
            break
    return {"spec": {key: value for key, value in spec.items() if key != "config"}, "records": records,
            "safety": {"collision": int(env.telemetry.get("collisions", 0)), "NMAC": int(env.telemetry.get("nmac", 0)),
                       "hard": int(env.telemetry.get("hard_separation_violations", 0))}}


def _intent(action: np.ndarray):
    from ..core.merge_scheduler import LowLevelIntent
    return LowLevelIntent.from_array(action)


def _quantiles(values: list[float]) -> dict:
    if not values: return {"count": 0, "mean": None, "P50": None, "P95": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {"count": len(values), "mean": float(array.mean()), "P50": float(np.percentile(array, 50)),
            "P95": float(np.percentile(array, 95)), "max": int(array.max())}


def _streaks(records: list[dict], direction: tuple[int, int]) -> tuple[list[list[dict]], dict]:
    requested, target = direction
    rows = [row for row in records if row["native_release_corrected"] and row["requested"] == requested and row["native_teacher"] == target]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows: groups[(row["update"], row["episode"], row["uid"])].append(row)
    streaks: list[list[dict]] = []
    for value in groups.values():
        previous = None; active: list[dict] = []
        for row in sorted(value, key=lambda item: item["tick"]):
            if previous is None or row["tick"] != previous + 1:
                if active: streaks.append(active)
                active = [row]
            else: active.append(row)
            previous = row["tick"]
        if active: streaks.append(active)
    similarities = []
    exact_repeat = 0
    for streak in streaks:
        for left, right in zip(streak, streak[1:]):
            first, second = np.asarray(left["observation"], dtype=float), np.asarray(right["observation"], dtype=float)
            similarities.append(float(np.dot(first, second) / max(1e-12, np.linalg.norm(first) * np.linalg.norm(second))))
            exact_repeat += int(left["observation_hash"] == right["observation_hash"])
    repeated_by_phase = Counter((row["update"], row["episode"], row["uid"], row["phase"]) for row in rows)
    return streaks, {"raw_tick_corrections": len(rows), "unique_decision_events": len(streaks),
                     "duplication_factor": None if not streaks else len(rows) / len(streaks),
                     "streak_length": _quantiles([len(value) for value in streaks]),
                     "adjacent_observation_cosine": _quantiles(similarities), "adjacent_exact_observation_hash_matches": exact_repeat,
                     "same_UAV_same_phase_repeat_count": _quantiles([value for value in repeated_by_phase.values()])}


def _bc_distribution(records: list[dict]) -> dict:
    # This predicate is byte-for-byte the current SharedPPO.update teacher-row gate.
    selected = [row for row in records if row["teacher_used"] and row["teacher_action_present"]]
    counts = Counter()
    for row in selected:
        key = (row["requested"], row["native_teacher"])
        counts[{(1, 0): "RELEASE_to_WAIT", (0, 1): "WAIT_to_RELEASE", (1, 1): "correct_RELEASE", (0, 0): "correct_WAIT"}[key]] += 1
    return {"current_release_CE_gate": "teacher_used && teacher_action is not None", "sample_count": len(selected),
            "labels": {key: int(counts[key]) for key in ("RELEASE_to_WAIT", "WAIT_to_RELEASE", "correct_WAIT", "correct_RELEASE")},
            "CORRECTION_ONLY_BINARY_SUPERVISION": bool(counts["correct_WAIT"] == 0 and counts["correct_RELEASE"] == 0),
            "sample_selection_is_correction_only": True}


def _events(records: list[dict], *, deduplicate: bool) -> list[dict]:
    legal = [row for row in records if row["eligible"]]
    if not deduplicate: return legal
    out: list[dict] = []
    previous: dict[tuple, tuple[int, int]] = {}
    for row in sorted(legal, key=lambda item: (item["update"], item["episode"], item["uid"], item["tick"])):
        key = (row["update"], row["episode"], row["uid"])
        state = (row["tick"], row["all_event_teacher"])
        if key not in previous or previous[key][0] + 1 != state[0] or previous[key][1] != state[1]: out.append(row)
        previous[key] = state
    return out


def _ce(policy: SharedPPO, rows: list[dict]) -> float:
    obs = torch.as_tensor(np.asarray([row["observation"] for row in rows], dtype=np.float32))
    labels = torch.as_tensor([row.get("label", row["all_event_teacher"]) for row in rows], dtype=torch.long)
    with torch.no_grad(): return float(F.cross_entropy(policy.model.actor_distribution(obs).logits[0], labels).item())


def _probe(rows: list[dict], cohort: dict, *, balanced: bool) -> dict:
    if not rows: raise RuntimeError("offline release probe has no labeled rows")
    labels = np.asarray([row.get("label", row["all_event_teacher"]) for row in rows], dtype=np.int64)
    if len(np.unique(labels)) < 2: raise RuntimeError("offline release probe requires both WAIT and RELEASE labels")
    policy = _initial_policy(); optimizer = torch.optim.Adam(policy.model.parameters(), lr=3e-4)
    generator = np.random.default_rng(2401)
    by_label = {label: np.flatnonzero(labels == label) for label in (0, 1)}
    for _ in range(PROBE_STEPS):
        if balanced:
            choices = np.concatenate((generator.choice(by_label[0], PROBE_BATCH // 2, replace=True),
                                      generator.choice(by_label[1], PROBE_BATCH - PROBE_BATCH // 2, replace=True)))
            generator.shuffle(choices)
        else: choices = generator.choice(len(rows), PROBE_BATCH, replace=True)
        observations = torch.as_tensor(np.asarray([rows[index]["observation"] for index in choices], dtype=np.float32))
        targets = torch.as_tensor(labels[choices], dtype=torch.long)
        loss = F.cross_entropy(policy.model.actor_distribution(observations).logits[0], targets)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    result = evaluate_release_cohort(policy, cohort)
    observations = torch.as_tensor(np.asarray([row["observation_v2"] for row in cohort["rows"]], dtype=np.float32))
    targets = np.asarray([row["a_teacher"]["release_index"] for row in cohort["rows"]], dtype=np.int64)
    with torch.no_grad(): probabilities = torch.softmax(policy.model.actor_distribution(observations).logits[0], dim=-1)[:, 1].cpu().numpy()
    result.update({"train_samples": len(rows), "train_labels": {"WAIT": int((labels == 0).sum()), "RELEASE": int((labels == 1).sum())},
                   "optimizer_steps": PROBE_STEPS, "batch_size": PROBE_BATCH, "training_CE": _ce(policy, rows),
                   "cohort_CE": float(-(np.log(np.where(targets == 1, probabilities, 1.0 - probabilities) + 1e-12)).mean()),
                   "brier": float(np.mean((probabilities - targets) ** 2)),
                   "ECE_10": _ece(probabilities, targets),
                   "conditional_probability_gap": result["P_RELEASE_given_teacher_RELEASE"] - result["P_RELEASE_given_teacher_WAIT"]})
    return result


def _ece(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    values = 0.0
    for lower in np.linspace(0.0, 1.0, bins, endpoint=False):
        mask = (probabilities >= lower) & (probabilities < lower + 1.0 / bins)
        if np.any(mask): values += float(mask.mean() * abs(probabilities[mask].mean() - labels[mask].mean()))
    return values


def _aliasing_audit(rows: list[dict]) -> dict:
    """Inspect opposite Teacher labels at identical or nearest V2 event states.

    This is an audit only: the raw V2 vector and the Teacher labels remain
    unchanged.  Nearest cross-label comparisons make the requested release
    eligibility/topology/resource fields explicit instead of treating the
    87-D vector as an opaque classifier input.
    """
    observations = np.asarray([row["observation"] for row in rows], dtype=np.float32)
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    hashes: dict[str, set[int]] = defaultdict(set)
    for row in rows: hashes[row["observation_hash"]].add(int(row["label"]))
    conflicting_hashes = [key for key, value in hashes.items() if len(value) > 1]
    opposite_distance = []
    pairs: list[tuple[int, int]] = []
    for index, label in enumerate(labels):
        candidates = np.flatnonzero(labels != label)
        distances = np.linalg.norm(observations[candidates] - observations[index], axis=1)
        nearest = int(candidates[int(np.argmin(distances))])
        opposite_distance.append(float(distances.min())); pairs.append((index, nearest))
    group_slices = {
        "release_eligibility": LOW_OBS_V2.slices["release_eligible"],
        "phase_ring": slice(LOW_OBS_V2.slices["phase"].start, LOW_OBS_V2.slices["ring_role"].stop),
        "target_predecessor": LOW_OBS_V2.slices["target_ring_predecessor"],
        "target_successor": LOW_OBS_V2.slices["target_ring_successor"],
        "nearest_predicted_conflict": LOW_OBS_V2.slices["nearest_predicted_conflict"],
        "resource_state": slice(LOW_OBS_V2.slices["assigned_lz_present"].start, LOW_OBS_V2.slices["global_free_lz_fraction"].stop),
        "headway_and_interface": slice(LOW_OBS_V2.slices["headways"].start, LOW_OBS_V2.slices["reconfiguration_active_pending"].stop),
    }
    deltas = {}
    for name, field in group_slices.items():
        values = [np.mean(np.abs(observations[left, field] - observations[right, field])) for left, right in pairs]
        deltas[name] = float(np.mean(values))
    return {"deduplicated_events": len(rows), "exact_hashes": len(hashes), "opposite_label_exact_hashes": len(conflicting_hashes),
            "nearest_opposite_label_l2": _quantiles(opposite_distance), "nearest_opposite_label_feature_abs_delta": deltas,
            "interpretation": "Small nearest opposite-label distances with small topology/resource deltas indicate V2 aliasing; otherwise the failed supervised separation still establishes insufficient representation under this frozen probe."}


def _markdown(report: dict) -> str:
    audit, probes = report["event_audit"], report["probes"]
    lines = ["# V3-L1-D Release Objective + Decision-Event Diagnostic", "", "Evidence: deterministic replay from preexisting pilot checkpoints (newly_run) plus the fixed 224-state cohort (preexisting artifact). No PPO update was executed.", "",
             "## Evidence boundary", "", report["evidence_boundary"], "", "## Raw corrections vs. decision events", ""]
    for name in ("RELEASE_to_WAIT", "WAIT_to_RELEASE"):
        row = audit[name]; lines.append(f"- {name}: raw ticks={row['raw_tick_corrections']}; unique events={row['unique_decision_events']}; duplication={row['duplication_factor']:.3f}; streak mean/P50/P95/max={row['streak_length']['mean']:.3f}/{row['streak_length']['P50']:.3f}/{row['streak_length']['P95']:.3f}/{row['streak_length']['max']}; adjacent cosine mean={row['adjacent_observation_cosine']['mean']}; same-UAV/phase repeats mean={row['same_UAV_same_phase_repeat_count']['mean']:.3f}")
    lines += ["", "## Current release BC semantics", "", f"- Gate: `{report['bc_distribution']['current_release_CE_gate']}`.", f"- Labels: {report['bc_distribution']['labels']}.", f"- sample_selection_is_correction_only = {str(report['bc_distribution']['sample_selection_is_correction_only']).lower()}", f"- CORRECTION_ONLY_BINARY_SUPERVISION = {str(report['bc_distribution']['CORRECTION_ONLY_BINARY_SUPERVISION']).lower()}", "", "## Offline supervised probes", ""]
    for name in ("A_current_correction_only", "B_all_legal_ticks", "C_deduplicated_events", "D_deduplicated_balanced"):
        row = probes[name]
        lines.append(f"- {name}: n={row['train_samples']}; labels WAIT/RELEASE={row['train_labels']['WAIT']}/{row['train_labels']['RELEASE']}; accuracy={row['release_accuracy']:.4f}; false-release/wait={row['false_release_rate']:.4f}/{row['false_wait_rate']:.4f}; P(R|Teacher WAIT/RELEASE)={row['P_RELEASE_given_teacher_WAIT']:.4f}/{row['P_RELEASE_given_teacher_RELEASE']:.4f}; gap={row['conditional_probability_gap']:.4f}; CE={row['cohort_CE']:.4f}; brier={row['brier']:.4f}; ECE={row['ECE_10']:.4f}")
    aliasing = report["observation_aliasing_audit"]
    lines += ["", "## Observation V2 aliasing audit", "",
              f"- Deduplicated events / exact hashes / opposite-label exact hashes: {aliasing['deduplicated_events']} / {aliasing['exact_hashes']} / {aliasing['opposite_label_exact_hashes']}.",
              f"- Nearest opposite-label L2 mean/P50/P95/max: {aliasing['nearest_opposite_label_l2']['mean']:.4f}/{aliasing['nearest_opposite_label_l2']['P50']:.4f}/{aliasing['nearest_opposite_label_l2']['P95']:.4f}/{aliasing['nearest_opposite_label_l2']['max']}.",
              f"- Mean absolute feature deltas at nearest opposite-label pairs: {aliasing['nearest_opposite_label_feature_abs_delta']}."]
    lines += ["", "## Exact-hash full-state attribution", "",
              "- Teacher condition chain: release eligibility and topology slots are observed in V2; scheduler tentative/executed slot timing, owner/pending reservation, retry cooldown and committed-plan timing are hidden from V2."]
    for pair in report["exact_hash_opposite_label_pairs"]:
        lines.append(f"- hash={pair['observation_hash'][:12]}: WAIT reason={pair['WAIT']['teacher_reason']}, RELEASE reason={pair['RELEASE']['teacher_reason']}; full state is retained in the JSON artifact.")
    lines += ["", f"Observation V2 sufficient = {str(report['observation_v2_sufficient']).lower()}", f"final root cause = {report['root_cause']}", f"recommended minimum repair = {report['recommended_minimum_repair']}", f"fresh PPO rerun allowed = {str(report['fresh_PPO_rerun_allowed']).lower()}", ""]
    return "\n".join(lines)


def run_diagnostic(*, pilot_dir: str | Path, raw_dataset: str | Path, teacher_dataset: str | Path,
                   release_cohort: str | Path, output_dir: str | Path, workers: int = 4) -> dict:
    pilot_dir, destination = Path(pilot_dir), Path(output_dir); destination.mkdir(parents=True, exist_ok=True)
    pilot = json.loads((pilot_dir / "v3_l1_coverage_bridge_pilot.json").read_text(encoding="utf-8"))
    if pilot.get("status") != "completed": raise RuntimeError("diagnostic requires completed coverage-bridge pilot")
    cohort = json.loads(Path(release_cohort).read_text(encoding="utf-8"))
    if len(cohort.get("rows", [])) != 224: raise RuntimeError("requires the frozen 224-state cohort")
    specs = _task_specs(pilot, raw_dataset, teacher_dataset)
    with ProcessPoolExecutor(max_workers=workers) as pool: slices = list(pool.map(_replay_task, specs))
    records = [row for chunk in slices for row in chunk["records"]]
    if any(any(value for value in chunk["safety"].values()) for chunk in slices): raise RuntimeError("diagnostic replay safety failure")
    expected = pilot["updates"][0]["release_supervision"]
    update1 = [row for row in records if row["update"] == 1]
    observed_r2w = sum(row["native_release_corrected"] and row["requested"] == 1 and row["native_teacher"] == 0 for row in update1)
    if len(update1) != expected["release_decision_opportunities"] or observed_r2w != expected["RELEASE_to_WAIT"]:
        raise RuntimeError(f"exact update1 replay mismatch: rows={len(update1)} R2W={observed_r2w}; expected={expected}")
    r2w_streaks, r2w = _streaks(records, (1, 0)); _, w2r = _streaks(records, (0, 1))
    current = [dict(row, label=row["native_teacher"]) for row in records if row["teacher_used"] and row["teacher_action_present"]]
    all_ticks = [dict(row, label=row["all_event_teacher"]) for row in _events(records, deduplicate=False)]
    dedup = [dict(row, label=row["all_event_teacher"]) for row in _events(records, deduplicate=True)]
    probes = {"A_current_correction_only": _probe(current, cohort, balanced=False),
              "B_all_legal_ticks": _probe(all_ticks, cohort, balanced=False),
              "C_deduplicated_events": _probe(dedup, cohort, balanced=False),
              "D_deduplicated_balanced": _probe(dedup, cohort, balanced=True)}
    d = probes["D_deduplicated_balanced"]
    sufficient = bool(d["P_RELEASE_given_teacher_RELEASE"] >= d["P_RELEASE_given_teacher_WAIT"] + .20 and d["false_release_rate"] <= .25 and d["false_wait_rate"] <= .25)
    root_cause = "RELEASE_SUPERVISION_OBJECTIVE_SEMANTICS" if sufficient else "OBSERVATION_V2_ALIASING_OR_INSUFFICIENT_SEPARABILITY"
    repair = ("Use categorical release CE only on legal release decision events, deduplicate contiguous identical persistence events, and retain speed/merge correction BC unchanged."
              if sufficient else "Do not alter labels; first perform the requested Observation V2 aliasing audit before any objective change.")
    report = {"stage": "V3_L1_D_release_objective_decision_event_diagnostic", "evidence_type": "mixed",
              "evidence_boundary": "The pilot stored aggregate updates/checkpoints but no per-tick transitions. This run exactly replayed update1 from baseline and update4 from update3; update1 matched the stored 10,620 opportunities and 7,885 R→W corrections exactly. It does not claim raw-tick evidence for unreplayable pre-update states of updates 2/3/5/6.",
              "replay_slices": [{key: value for key, value in chunk["spec"].items() if key != "checkpoint"} for chunk in slices],
              "event_audit": {"RELEASE_to_WAIT": r2w, "WAIT_to_RELEASE": w2r, "all_legal_release_ticks": len(all_ticks), "deduplicated_legal_events": len(dedup)},
              "observation_aliasing_audit": _aliasing_audit(dedup),
              "exact_hash_opposite_label_pairs": _opposite_hash_pairs(records),
              "bc_distribution": _bc_distribution(records), "probes": probes, "observation_v2_sufficient": sufficient,
              "root_cause": root_cause, "recommended_minimum_repair": repair,
              "fresh_PPO_rerun_allowed": sufficient, "formal_L1_training_allowed": False,
              "prohibited_changes_confirmed": ["reward", "BC global weight", "PPO LR", "entropy", "action catalog", "Safety", "lambda", "targeted ratio", "L1b/L2"]}
    _json(destination / "v3_l1_release_objective_diagnostic.json", report)
    (destination / "v3_l1_release_objective_diagnostic.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Offline-only V3 L1 release objective diagnostic; never runs PPO.")
    parser.add_argument("--pilot-dir", required=True); parser.add_argument("--raw-dataset", required=True)
    parser.add_argument("--teacher-dataset", required=True); parser.add_argument("--release-cohort", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(); report = run_diagnostic(pilot_dir=args.pilot_dir, raw_dataset=args.raw_dataset,
        teacher_dataset=args.teacher_dataset, release_cohort=args.release_cohort, output_dir=args.output, workers=args.workers)
    print(json.dumps({key: report[key] for key in ("observation_v2_sufficient", "root_cause", "fresh_PPO_rerun_allowed")}, indent=2))


if __name__ == "__main__": _main()
