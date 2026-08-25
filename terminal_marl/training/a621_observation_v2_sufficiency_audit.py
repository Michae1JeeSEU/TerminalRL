"""A6.21: online high-level observation-v2 prototype sufficiency audit (PPO STOP)."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import csv
import json
from pathlib import Path

import numpy as np

from ..core.config import RingRole
from ..core.merge_scheduler import LowLevelIntent
from ..core.uav import FlightPhase
from ..envs.parallel_env import WorldSnapshot
from .a611_causal_audit import DemandProfile
from .a612_scheduler_repair_audit import _a611_mu0
from .a613_capacity_reaudit import repair_config
from .a614_reconfiguration_timing_audit import OUTER_PLUS_ONE, PPO_AUTHORIZATION, run_dynamic_at_time
from .a615_opportunity_sequencing_audit import _minimum_prediction
from .a620_window_identifiability_audit import PERMUTATIONS, WAITING_MATERIALITY, _label, _metrics, _predict


SEEDS = tuple(range(12))
SNAPSHOT_TIMES = (180.0, 210.0, 240.0, 270.0, 300.0, 330.0, 360.0)
INTERFACES = (("o2m", RingRole.OUTER, RingRole.MIDDLE, FlightPhase.RING_OUTER),
              ("m2i", RingRole.MIDDLE, RingRole.INNER, FlightPhase.RING_MIDDLE),
              ("i2f", RingRole.INNER, "final", FlightPhase.RING_INNER))
BASE_NAMES = ("actual_outer_radius", "actual_middle_radius", "actual_inner_radius", "actual_outer_height", "actual_middle_height", "actual_inner_height",
              "speed_profile_bias", "headway_o2m", "headway_m2i", "headway_i2f",
              "target_delta_outer_radius", "target_delta_middle_radius", "target_delta_inner_radius", "target_delta_outer_height", "target_delta_middle_height", "target_delta_inner_height",
              "external_queue", "final_pressure", "lz_utilization", "max_lz_service_remaining",
              "geometry_event_gate", "transition_active", "transition_progress", "episode_progress")
FLOW_SUFFIXES = ("eligible_ready_count", "pending_release_request_count", "time_to_next_eligibility")
RESERVATION_SUFFIXES = ("active_reservation_count", "time_to_nearest_reserved_merge", "minimum_prospective_separation_margin")
WINDOW_NAMES = ("transition_horizon_impacted_o2m_requests", "nearest_o2m_time_to_interface", "committed_reserved_flow_pressure", "current_prospective_safety_pressure")
V2_NAMES = BASE_NAMES + tuple(f"{interface}_{suffix}" for interface, *_ in INTERFACES for suffix in FLOW_SUFFIXES + RESERVATION_SUFFIXES) + WINDOW_NAMES
FLOW_INDEX = tuple(range(len(BASE_NAMES), len(BASE_NAMES) + 9))
RESERVATION_INDEX = tuple(range(len(BASE_NAMES) + 9, len(BASE_NAMES) + 18))
WINDOW_INDEX = tuple(range(len(BASE_NAMES) + 18, len(V2_NAMES)))


def _profile() -> DemandProfile:
    mu, _ = _a611_mu0(0)
    return DemandProfile("constant_high_repair_scheduler", ((0.0, 720.0, 1.55 * mu),), 720.0, 480.0)


def _role_from_target(value: str):
    return {"ring_outer": RingRole.OUTER, "ring_middle": RingRole.MIDDLE, "ring_inner": RingRole.INNER}.get(value)


def _source_ready(env, role: RingRole) -> list:
    return [uav for uav in env.active.values() if uav.ring_role is role and uav.committed_plan is None]


def _time_to_eligibility(env, role: RingRole) -> float:
    if _source_ready(env, role):
        return 0.0
    times = []
    target_phase = {RingRole.OUTER: "ring_outer", RingRole.MIDDLE: "ring_middle", RingRole.INNER: "ring_inner"}[role]
    for uav in env.active.values():
        plan = uav.committed_plan
        if plan is not None and plan.metadata.get("target_phase") == target_phase:
            times.append(max(0., float(plan.end_time - env.now)))
    return min(times, default=120.0) / 120.0


def _reservation_rows(env, interface: str) -> list:
    target = {"o2m": "middle", "m2i": "inner", "i2f": "final"}[interface]
    return [row for row in env.scheduler.reservations.values() if row.target_stage == target]


def _prospective_margin(env, uav, target, intent: LowLevelIntent) -> float:
    """Read a candidate on a clone; never reserve or mutate the formal branch."""
    clone = deepcopy(env)
    probe = clone.active.get(uav.uid)
    if probe is None or target == "final" and clone.lz_manager.choose_rule_zone(probe, clone.now) is None:
        return -1.0
    lz_id = None if target != "final" else clone.lz_manager.choose_rule_zone(probe, clone.now)
    snapshot = WorldSnapshot(clone.now, clone.active, clone.airspace)
    for candidate in clone.scheduler.repair_candidates(intent):
        proposal, plan, reason = clone._candidate_plan(probe, candidate, target, snapshot, lz_id)
        if reason is None and plan is not None:
            predicted, swept = _minimum_prediction(clone, plan, snapshot)
            value = min(item for item in (predicted, swept) if item is not None) if any(item is not None for item in (predicted, swept)) else clone.config.safe_distance
            return float(np.clip((value - clone.config.safe_distance) / clone.config.safe_distance, -1., 4.))
    return -1.0


def observation_v2(wrapper) -> dict:
    """Build only online, action-pre fields from the frozen live state."""
    env, airspace, context = wrapper.base, wrapper.base.airspace, wrapper._context()
    actual = airspace.normalized_state_vector().astype(float)
    target_delta = []
    for role in RingRole:
        target_delta.append((airspace.target_ring(role).radius - airspace.actual_ring(role).radius) / wrapper.config.max_radius_change_per_update)
    for role in RingRole:
        target_delta.append((airspace.target_ring(role).height - airspace.actual_ring(role).height) / wrapper.config.max_height_change_per_update)
    remain = [zone.service_remaining for zone in env.lz_manager.zones]
    base = list(actual) + list(np.clip(target_delta, -1., 1.)) + [
        min(1., context["external_queue"] / 20.), context["final_pressure"], context["lz_utilization"], min(1., max(remain, default=0.) / 120.),
        float(wrapper.geometry_event_gate()), float(airspace.transition is not None), airspace.transition_progress(env.now),
        min(1., env.now / max(1., wrapper.config.max_simulation_seconds)),
    ]
    actions = wrapper.low_level.act(env._observations())  # frozen current intent; no action is submitted.
    values, margins, o2m_pending, o2m_time = [], [], 0, 1.
    for interface, source, target, _phase in INTERFACES:
        ready = _source_ready(env, source)
        intents = {uav.uid: LowLevelIntent.from_array(actions.get(str(uav.uid), np.zeros(3, np.float32))) for uav in ready}
        pending = sum(intent.release_request > env.config.release_threshold for intent in intents.values())
        reservations = _reservation_rows(env, interface)
        nearest = min((max(0., row.merge_time - env.now) for row in reservations), default=120.) / 120.
        margin = _prospective_margin(env, ready[0], target, intents[ready[0].uid]) if ready else -1.0
        values.extend([min(1., len(ready) / 10.), min(1., pending / 10.), _time_to_eligibility(env, source),
                       min(1., len(reservations) / 10.), min(1., nearest), margin])
        margins.append(margin)
        if interface == "o2m":
            o2m_pending, o2m_time = pending, _time_to_eligibility(env, source)
    transition_seconds = wrapper.config.geometry_transition_seconds
    impacted = min(1., (o2m_pending + sum(1 for row in _reservation_rows(env, "o2m") if 0. <= row.merge_time - env.now <= transition_seconds)) / 10.)
    committed = sum(uav.committed_plan is not None for uav in env.active.values()) + len(env.scheduler.reservations)
    pressure = min(1., committed / max(1., sum(context["capacity"])))
    safety_pressure = min(1., max(0., -min(margins, default=0.)))
    vector = np.asarray(base + values + [impacted, o2m_time, pressure, safety_pressure], dtype=np.float32)
    if vector.shape != (len(V2_NAMES),):
        raise AssertionError(f"v2 shape {vector.shape} != ({len(V2_NAMES)},)")
    return {"schema": f"phaseA_high_obs_v2_{len(V2_NAMES)}d", "values": vector.astype(float).tolist(),
            "online_only": True, "source_time": float(env.now)}


def _task(profile, seed: int, action_time: float | None, capture_times: tuple[float, ...]):
    result = run_dynamic_at_time(repair_config(profile, dynamic=True), seed, float("inf") if action_time is None else action_time,
                                 force_initial_gate=False, observation_capture_times=capture_times,
                                 observation_capture_hook=observation_v2)
    if not result["safe"]:
        raise RuntimeError("A6.21 unsafe rollout")
    return seed, action_time, result


def collect_snapshots(*, seeds=SEEDS, times=SNAPSHOT_TIMES, workers: int = 6) -> list[dict]:
    profile = _profile(); output = {}; jobs = [(profile, seed, None, tuple(times)) for seed in seeds]
    jobs += [(profile, seed, time, (time,)) for seed in seeds for time in times]
    with ProcessPoolExecutor(max_workers=min(max(1, workers), len(jobs))) as pool:
        futures = [pool.submit(_task, *job) for job in jobs]
        for future in as_completed(futures):
            seed, time, result = future.result(); output[seed, time] = result
    rows = []
    for seed in seeds:
        nominal = output[seed, None]
        for time in times:
            dynamic = output[seed, time]
            base, intervention = nominal["pre_action_observations"][time], dynamic["pre_action_observations"][time]
            if base["observation_29d"] != intervention["observation_29d"] or base["hook"]["values"] != intervention["hook"]["values"]:
                raise RuntimeError("A6.21 CRN pre-action snapshot mismatch")
            delta = dynamic["metrics"]["all_request_waiting_area"] - nominal["metrics"]["all_request_waiting_area"]
            rows.append({"snapshot_id": f"seed{seed}_t{int(time)}", "seed": seed, "time": time,
                         "v1_29d": base["observation_29d"], "v2": base["hook"]["values"], "waiting_delta": float(delta), "outcome_label": _label(float(delta)),
                         "nominal_waiting_area": nominal["metrics"]["all_request_waiting_area"], "dynamic_waiting_area": dynamic["metrics"]["all_request_waiting_area"],
                         "geometry_applied_time": dynamic["timing"]["geometry_applied_time"], "geometry_effective_time": dynamic["timing"]["geometry_effective_time"],
                         "safe": nominal["safe"] and dynamic["safe"], "checked_executed": True,
                         "safety": {"collisions": dynamic["metrics"]["collisions"], "nmac": dynamic["metrics"]["nmac"], "hard_separation_violations": dynamic["metrics"]["hard_separation_violations"], "final_overcapacity_steps": dynamic["metrics"]["final_overcapacity_steps"]}})
    return rows


def _nearest(rows: list[dict], feature: str) -> tuple[list[dict], dict]:
    x = np.asarray([row[feature] for row in rows], float); z = (x - x.mean(0)) / np.maximum(x.std(0), 1e-6); labels = [row["outcome_label"] for row in rows]; result = []
    exact_conflicts = 0
    for index, row in enumerate(rows):
        choice = min((item for item in range(len(rows)) if item != index), key=lambda item: float(np.linalg.norm(z[index] - z[item])))
        distance = float(np.linalg.norm(z[index] - z[choice])); conflict = labels[index] != labels[choice]
        different = [item for item in range(len(rows)) if labels[item] != labels[index]]
        conflicting = min(different, key=lambda item: float(np.linalg.norm(z[index] - z[item]))) if different else None
        conflict_distance = None if conflicting is None else float(np.linalg.norm(z[index] - z[conflicting]))
        exact = conflict_distance is not None and conflict_distance < 1e-9
        result.append({"feature_set": feature, "snapshot_id": row["snapshot_id"], "outcome_label": labels[index], "nearest_snapshot_id": rows[choice]["snapshot_id"], "nearest_label": labels[choice], "distance_z": distance, "label_conflict": conflict,
                       "nearest_conflicting_snapshot_id": None if conflicting is None else rows[conflicting]["snapshot_id"], "nearest_conflicting_label": None if conflicting is None else labels[conflicting],
                       "nearest_conflicting_distance_z": conflict_distance, "exact_conflicting_label": exact})
        exact_conflicts += int(exact)
    return result, {"feature_set": feature, "nearest_conflict_rate": float(np.mean([row["label_conflict"] for row in result])), "exact_conflict_count": exact_conflicts}


def _feature_key(feature: str) -> str:
    """Keep a human-facing feature-set name separate from stored columns."""
    return "v2" if feature == "v2_full" else feature


def _evaluate(rows: list[dict], feature: str, indices: tuple[int, ...] | None = None) -> list[dict]:
    x = np.asarray([row[_feature_key(feature)] for row in rows], float); x = x if indices is None else x[:, indices]
    y = np.asarray([row["outcome_label"] for row in rows], object); seed = np.asarray([row["seed"] for row in rows]); rng = np.random.default_rng(62121 + len(x[0])); result = []
    for held in sorted(set(seed)):
        train, test = np.where(seed != held)[0], np.where(seed == held)[0]
        for model in ("logistic_linear", "shallow_tree"):
            prediction = _predict(model, x[train], y[train], x[test]); metric = _metrics(y[test], prediction); perm = []
            for _ in range(PERMUTATIONS):
                shuffled = rng.permutation(y); perm.append(_metrics(shuffled[test], _predict(model, x[train], shuffled[train], x[test]))["macro_f1_present"])
            result.append({"feature_set": feature, "model": model, "split": f"leave_seed_{held}_out", "held_out_seed": int(held), "feature_dimension": x.shape[1], **metric,
                           "permutation_macro_f1_mean": float(np.mean(perm)), "permutation_macro_f1_p95": float(np.percentile(perm, 95)), "predictions": prediction.tolist(), "targets": y[test].tolist()})
    for model in ("logistic_linear", "shallow_tree"):
        subset = [row for row in result if row["model"] == model]
        result.append({"feature_set": feature, "model": model, "split": "leave_one_seed_out_aggregate", "held_out_seed": "all", "feature_dimension": x.shape[1],
                       **{key: float(np.mean([row[key] for row in subset])) for key in ("accuracy", "balanced_accuracy", "macro_f1_present", "permutation_macro_f1_mean")},
                       "harmful_heldout_seed_count": sum(row["recall_harmful"] is not None for row in subset),
                       "harmful_recall_mean": float(np.mean([row["recall_harmful"] for row in subset if row["recall_harmful"] is not None])) if any(row["recall_harmful"] is not None for row in subset) else None})
    return result


def _best(rows: list[dict], feature: str) -> dict:
    return max((row for row in rows if row["feature_set"] == feature and row["split"] == "leave_one_seed_out_aggregate"), key=lambda row: row["macro_f1_present"])


def classify(rows, conflicts, classifiers) -> str:
    labels = Counter(row["outcome_label"] for row in rows); harmful_seeds = {row["seed"] for row in rows if row["outcome_label"] == "HARMFUL"}
    if labels["HARMFUL"] < 3 or len(harmful_seeds) < 3:
        return "INSUFFICIENT_HARMFUL_COVERAGE"
    v1, v2 = _best(classifiers, "v1_29d"), _best(classifiers, "v2_full")
    uplift = v2["macro_f1_present"] - v2["permutation_macro_f1_mean"]
    conflict = {row["feature_set"]: row for row in conflicts}; reduced = conflict["v2_full"]["nearest_conflict_rate"] < conflict["v1_29d"]["nearest_conflict_rate"] - .10
    if v2["macro_f1_present"] <= v1["macro_f1_present"] + .05 or uplift <= .05:
        return "NO_IMPROVEMENT_OVER_V1"
    if v2["macro_f1_present"] >= .70 and uplift >= .20 and reduced and v2["harmful_heldout_seed_count"] >= 3 and (v2["harmful_recall_mean"] or 0.) >= .60:
        return "V2_IDENTIFIABLE"
    if uplift >= .10 and v2["macro_f1_present"] >= .45:
        return "V2_PARTIALLY_IDENTIFIABLE"
    return "V2_NOT_IDENTIFIABLE"


def run_audit(*, seeds=SEEDS, times=SNAPSHOT_TIMES, workers: int = 6) -> dict:
    rows = collect_snapshots(seeds=seeds, times=times, workers=workers)
    if not all(row["safe"] and row["checked_executed"] and all(value == 0 for value in row["safety"].values()) for row in rows):
        raise RuntimeError("A6.21 hard safety failure")
    v1_conflict, v1_summary = _nearest(rows, "v1_29d"); v2_conflict, v2_summary = _nearest(rows, "v2")
    classifiers = _evaluate(rows, "v1_29d") + _evaluate(rows, "v2_full")
    ablations = []
    all_indices = tuple(range(len(V2_NAMES)))
    for name, dropped in (("v2_without_interface_flow", FLOW_INDEX), ("v2_without_reservation_safety_margin", RESERVATION_INDEX), ("v2_without_reconfiguration_window", WINDOW_INDEX)):
        keep = tuple(index for index in all_indices if index not in dropped)
        rows_for_ablation = _evaluate(rows, "v2", keep)
        for row in rows_for_ablation:
            row["feature_set"] = name
        ablations.extend(rows_for_ablation)
    return {"stage": "A6.21 High-Level Observation v2 Prototype & Sufficiency Audit", "evidence_type": "newly_run",
            "schema": {"name": f"phaseA_high_obs_v2_{len(V2_NAMES)}d", "dimension": len(V2_NAMES), "feature_names": V2_NAMES,
                       "blocks": {"base_configuration_workload": list(range(len(BASE_NAMES))), "interface_flow": list(FLOW_INDEX), "reservation_safety_margin": list(RESERVATION_INDEX), "reconfiguration_window": list(WINDOW_INDEX)},
                       "online_only": True, "excludes": ["future_arrivals", "post_action_state", "collision_rate", "reward", "outcome_label"]},
            "inputs": {"seeds": list(seeds), "snapshot_times": list(times), "intervention": "Outer R 600→612 m canonical full_dynamic", "materiality": WAITING_MATERIALITY, "ppo_authorization": PPO_AUTHORIZATION},
            "snapshot_labels": rows, "conflicts": v1_conflict + v2_conflict, "conflict_summary": [v1_summary, v2_summary], "classifier_results": classifiers, "ablation_results": ablations,
            "summary": {"classification": classify(rows, [v1_summary, v2_summary], classifiers), "label_counts": dict(Counter(row["outcome_label"] for row in rows)), "safe_rollouts": len(rows), "ppo_authorization": PPO_AUTHORIZATION,
                        "protocol_risks": ["LOSO is internal seed-held-out validation, not independent traffic-distribution generalization.", "Finite 12-seed diagnostic set; harmful coverage is gated explicitly.", "v2 fields are action-pre online state/committed-plan/current frozen-policy intent only; no PPO is trained."]}}


def _csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row}) or ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def write_report(report: dict, output: str | Path) -> None:
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    (output / "a621_observation_schema_v2.json").write_text(json.dumps(report["schema"], indent=2), encoding="utf-8")
    labels = [{key: value for key, value in row.items() if key not in {"v1_29d", "v2", "safety"}} for row in report["snapshot_labels"]]
    _csv(output / "a621_snapshot_labels.csv", labels); _csv(output / "a621_v1_v2_conflicts.csv", report["conflicts"])
    _csv(output / "a621_classifier_results.csv", report["classifier_results"]); _csv(output / "a621_ablation_results.csv", report["ablation_results"])
    (output / "a621_audit.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    best_v1, best_v2 = _best(report["classifier_results"], "v1_29d"), _best(report["classifier_results"], "v2_full")
    conflict = {row["feature_set"]: row for row in report["conflict_summary"]}
    (output / "a621_summary.md").write_text(
        "# A6.21 High-Level Observation v2 Prototype & Sufficiency Audit\n\n"
        f"Classification: **{report['summary']['classification']}**. Labels: `{report['summary']['label_counts']}`.\n\n"
        "| feature set | best LOSO macro-F1 | permutation macro-F1 | nearest-label conflict rate |\n|---|---:|---:|---:|\n"
        f"| v1_29d | {best_v1['macro_f1_present']:.3f} | {best_v1['permutation_macro_f1_mean']:.3f} | {conflict['v1_29d']['nearest_conflict_rate']:.3f} |\n"
        f"| v2_full | {best_v2['macro_f1_present']:.3f} | {best_v2['permutation_macro_f1_mean']:.3f} | {conflict['v2_full']['nearest_conflict_rate']:.3f} |\n\n"
        f"v2 harmful held-out seeds: {best_v2.get('harmful_heldout_seed_count')}; mean harmful recall: {best_v2.get('harmful_recall_mean')}.\n\n"
        f"All {report['summary']['safe_rollouts']} paired counterfactual snapshots are hard-safe and checked==executed. PPO authorization: **STOP**.\n\n"
        "v1 is frozen; v2 is a diagnostic schema only. No scheduler, safety, reward, action space, decision interval, dwell, or transition semantics changed.\n", encoding="utf-8")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--seeds", default=",".join(map(str, SEEDS))); parser.add_argument("--times", default=",".join(str(int(x)) for x in SNAPSHOT_TIMES)); parser.add_argument("--workers", type=int, default=6); parser.add_argument("--output", default=None)
    args = parser.parse_args(); output = Path(args.output) if args.output else Path.home() / "training_outputs" / "terminal" / "event_gated_stage15_diagnostics" / "a621"
    report = run_audit(seeds=tuple(map(int, args.seeds.split(","))), times=tuple(map(float, args.times.split(","))), workers=args.workers); write_report(report, output); print(json.dumps(report["summary"]))


if __name__ == "__main__": main()
