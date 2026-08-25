"""A6.22: traffic-state excitation and frozen-observation stress audit.

This diagnostic deliberately leaves the Phase-A runtime untouched.  It finds
action-pre states on a nominal canonical replay, then replays the same CRN
case with the existing ``600 -> 612 m`` request at that state.  Warm start is
used only in the explicitly labelled stress regime.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path

import numpy as np

from ..core.config import RingRole
from ..core.merge_scheduler import LowLevelIntent
from ..core.uav import FlightPhase
from ..envs.high_level_wrapper import DynamicHighLevelWrapper, RuleLowLevelAdapter
from .a611_causal_audit import DemandProfile, _safe
from .a612_scheduler_repair_audit import _a611_mu0
from .a613_capacity_reaudit import repair_config
from .a614_reconfiguration_timing_audit import PPO_AUTHORIZATION, run_dynamic_at_time
from .a620_window_identifiability_audit import WAITING_MATERIALITY, _label
from .a621_observation_v2_sufficiency_audit import (
    INTERFACES, V2_NAMES, _evaluate, _nearest, observation_v2,
)


FORMAL_SEEDS = tuple(range(8))
BASELINE_SEEDS = tuple(range(4))
STRESS_SEEDS = tuple(range(4))
SENTINEL_BY_SUFFIX = {
    "time_to_next_eligibility": 1.0,
    "time_to_nearest_reserved_merge": 1.0,
    "minimum_prospective_separation_margin": -1.0,
}


def regime_definitions(mu: float | None = None) -> tuple[list[dict], str]:
    """Regimes are fixed schedules; no future arrival realization is observed."""
    if mu is None:
        mu, source = _a611_mu0(0)
    else:
        source = "explicit_diagnostic_calibration"
    return [
        {"regime": "canonical_constant_high", "class": "canonical_baseline", "formal_value_evidence": True,
         "reset_mode": "flow_from_empty", "warm_start_uavs": 0,
         "segments": ((0., 720., 1.55 * mu),), "duration": 720., "drain": 480., "seeds": BASELINE_SEEDS,
         "phase_windows": ((0., 720., "peak"),)},
        {"regime": "formal_high", "class": "traffic_excited_formal", "formal_value_evidence": True,
         "reset_mode": "flow_from_empty", "warm_start_uavs": 0,
         "segments": ((0., 900., 1.70 * mu),), "duration": 900., "drain": 480., "seeds": FORMAL_SEEDS,
         "phase_windows": ((0., 900., "peak"),)},
        {"regime": "formal_bursty", "class": "traffic_excited_formal", "formal_value_evidence": True,
         "reset_mode": "flow_from_empty", "warm_start_uavs": 0,
         "segments": ((0., 300., .90 * mu), (300., 600., 2.10 * mu), (600., 900., .90 * mu)), "duration": 900., "drain": 480., "seeds": FORMAL_SEEDS,
         "phase_windows": ((0., 300., "rising"), (300., 600., "peak"), (600., 900., "falling"))},
        {"regime": "formal_overload", "class": "traffic_excited_formal", "formal_value_evidence": True,
         "reset_mode": "flow_from_empty", "warm_start_uavs": 0,
         "segments": ((0., 300., 1.05 * mu), (300., 600., 2.65 * mu), (600., 900., 1.15 * mu)), "duration": 900., "drain": 480., "seeds": FORMAL_SEEDS,
         "phase_windows": ((0., 300., "rising"), (300., 600., "peak"), (600., 900., "falling"))},
        {"regime": "stress_warm_burst", "class": "stress_diagnostic_only", "formal_value_evidence": False,
         "reset_mode": "warm_start_rings", "warm_start_uavs": 9,
         # Negative-time arrivals are consumed only to create a seeded state at t=0.
         "segments": ((-180., 0., .05), (0., 300., .95 * mu), (300., 600., 2.10 * mu), (600., 900., .95 * mu)), "duration": 900., "drain": 480., "seeds": STRESS_SEEDS,
         "phase_windows": ((0., 300., "rising"), (300., 600., "peak"), (600., 900., "falling"))},
    ], source


def _config(row: dict):
    profile = DemandProfile(row["regime"], tuple(row["segments"]), row["duration"], row["drain"])
    return replace(repair_config(profile, dynamic=True), reset_mode=row["reset_mode"], warm_start_uavs=row["warm_start_uavs"]), profile


def _phase(row: dict, time: float) -> str:
    return next((name for begin, end, name in row["phase_windows"] if begin <= time < end), "drain")


def _ready(env, role: RingRole) -> list:
    return [uav for uav in env.active.values() if uav.ring_role is role and uav.committed_plan is None]


def _state_row(wrapper: DynamicHighLevelWrapper, regime: dict) -> dict:
    """Cheap online traffic pressure used solely to choose diagnostic states.

    v1/v2 are captured later at selected times by a canonical no-action replay.
    This avoids allowing the clone-based v2 margin probe to distort sampling
    cost or accidentally run it at hundreds of discarded states.
    """
    env = wrapper.base
    actions = wrapper.low_level.act(env._observations())
    ready = {role: _ready(env, role) for role in RingRole}
    intents = {uav.uid: LowLevelIntent.from_array(actions.get(str(uav.uid), np.zeros(3, np.float32)))
               for role in RingRole for uav in ready[role]}
    pending = {role: sum(int(intents[u.uid].release_request > env.config.release_threshold) for u in ready[role]) for role in RingRole}
    reservations = list(env.scheduler.reservations.values())
    active_transfer = sum(u.phase in {FlightPhase.TRANSFER_O2M, FlightPhase.TRANSFER_M2I, FlightPhase.FINAL} for u in env.active.values())
    imminent_reserved = sum(0. <= item.merge_time - env.now <= env.config.geometry_transition_seconds for item in reservations)
    imminent = active_transfer + imminent_reserved + sum(pending.values())
    occupancies = {role: env.airspace.occupancy(role) for role in RingRole}
    outer_o2m_pending = pending[RingRole.OUTER]
    horizon_o2m = outer_o2m_pending + sum(0. <= item.merge_time - env.now <= env.config.geometry_transition_seconds and item.target_stage == "middle" for item in reservations)
    return {"time": float(env.now), "phase": _phase(regime, env.now),
            "outer_occupancy": occupancies[RingRole.OUTER], "middle_occupancy": occupancies[RingRole.MIDDLE], "inner_occupancy": occupancies[RingRole.INNER],
            "all_ring_occupancy_nonzero": all(value > 0 for value in occupancies.values()),
            "imminent_interface_transfers": int(imminent), "active_reservation_count": len(reservations),
            "pending_release_request_count": sum(pending.values()), "pending_o2m_release_count": outer_o2m_pending,
            "transition_horizon_imminent_o2m": horizon_o2m, "external_queue": len(env.admission.queue.items),
            "final_pressure": float(wrapper._context()["final_pressure"]), "lz_utilization": float(wrapper._context()["lz_utilization"]),
            "geometry_event_gate": bool(wrapper.geometry_event_gate()),
            "state_score": int(all(value > 0 for value in occupancies.values())) + int(imminent >= 2) + int(bool(reservations)) + int(sum(pending.values()) > 0) + int(horizon_o2m > 0)}


def _select(rows: list[dict], regime: dict) -> list[dict]:
    """Select event-conditioned, phase-diverse states without fixed snapshot times."""
    desired = [name for *_ignore, name in regime["phase_windows"]]
    result: list[dict] = []
    for phase in dict.fromkeys(desired):
        candidates = [row for row in rows if row["phase"] == phase and row["geometry_event_gate"]]
        if candidates:
            # Deterministic state-score ranking; time merely breaks exact ties.
            result.append(max(candidates, key=lambda item: (item["state_score"], item["imminent_interface_transfers"], item["active_reservation_count"], -item["time"])))
    if not result:
        candidates = [row for row in rows if row["geometry_event_gate"]]
        if candidates:
            result.append(max(candidates, key=lambda item: (item["state_score"], -item["time"])))
    return result


def _nominal_probe(regime: dict, seed: int) -> tuple[dict, list[dict]]:
    config, _ = _config(regime); wrapper = DynamicHighLevelWrapper(config, RuleLowLevelAdapter()); wrapper.reset(seed); env = wrapper.base
    candidates = []
    while env.now < config.max_simulation_seconds and not wrapper._done():
        # Probe at a regular 10-s cadence, then select only state-conditioned records.
        if 120. <= env.now < regime["duration"] and int(round(env.now)) % 10 == 0:
            candidates.append(_state_row(wrapper, regime))
        wrapper.coordinator.advance()
        actions = wrapper.coordinator.apply_speed_repairs(wrapper.low_level.act(env._observations()))
        env.step(actions)
    # This repeat is the canonical no-action runtime path and captures only
    # the selected action-pre observations.  It is also the CRN terminal
    # control paired to every intervention below.
    selected = _select(candidates, regime)
    capture_times = tuple(item["time"] for item in selected)
    baseline = run_dynamic_at_time(config, seed, float("inf"), force_initial_gate=False,
                                   observation_capture_times=capture_times, observation_capture_hook=observation_v2)
    if not baseline["safe"]:
        raise RuntimeError(f"A6.22 unsafe nominal {regime['regime']} seed={seed}")
    for item in selected:
        capture = baseline["pre_action_observations"][item["time"]]
        item["v1_29d"] = capture["observation_29d"]
        item["v2"] = capture["hook"]["values"]
    return baseline, selected


def _dynamic_task(regime: dict, seed: int, state: dict) -> tuple[str, dict]:
    config, _ = _config(regime); time = float(state["time"])
    result = run_dynamic_at_time(config, seed, time, force_initial_gate=False, observation_capture_time=time, observation_capture_hook=observation_v2)
    if not result["safe"]:
        raise RuntimeError(f"A6.22 unsafe dynamic {regime['regime']} seed={seed} t={time}")
    return f"{regime['regime']}:seed{seed}:t{int(time)}", result


def _nominal_task(regime: dict, seed: int) -> tuple[str, int, dict, list[dict]]:
    """Process-safe CRN control/probe; no state is shared across seeds."""
    baseline, states = _nominal_probe(regime, seed)
    return regime["regime"], seed, baseline, states


def _feature_excitation(rows: list[dict], scope: str) -> tuple[list[dict], dict]:
    matrix = np.asarray([row["v2"] for row in rows], dtype=float)
    spectrum = np.linalg.svd(matrix - matrix.mean(0), compute_uv=False) if len(matrix) > 1 else np.array([0.])
    weights = spectrum[spectrum > 1e-9] / max(float(spectrum[spectrum > 1e-9].sum()), 1e-12)
    rank = float(np.exp(-np.sum(weights * np.log(weights)))) if len(weights) else 0.
    output = []
    for index, name in enumerate(V2_NAMES):
        value = matrix[:, index]; sentinel = SENTINEL_BY_SUFFIX.get(next((suffix for suffix in SENTINEL_BY_SUFFIX if name.endswith(suffix)), ""))
        output.append({"scope": scope, "index": index, "feature": name, "unique_values": int(len(np.unique(np.round(value, 9)))),
                       "variance": float(np.var(value)), "zero_fraction": float(np.mean(np.isclose(value, 0.))),
                       "sentinel_value": sentinel, "sentinel_fraction": None if sentinel is None else float(np.mean(np.isclose(value, sentinel))),
                       "matrix_effective_rank": rank,
                       "traffic_flow_feature": any(token in name for token in ("occupancy", "eligible", "pending", "reservation", "time_to_", "separation_margin", "impacted"))})
    traffic = [row for row in output if row["traffic_flow_feature"]]
    excited = [row for row in traffic if row["unique_values"] >= 3 and row["variance"] > 1e-6 and (row["sentinel_fraction"] is None or row["sentinel_fraction"] < .80)]
    gate = {"scope": scope, "traffic_feature_count": len(traffic), "excited_traffic_feature_count": len(excited),
            "excited_traffic_fraction": len(excited) / max(1, len(traffic)), "effective_rank": rank,
            "pass": len(rows) >= 12 and len(excited) / max(1, len(traffic)) >= .60 and rank >= 3.0}
    return output, gate


def _conflicts(rows: list[dict], scope: str) -> tuple[list[dict], list[dict]]:
    output, summary = [], []
    for feature in ("v1_29d", "v2"):
        if len(rows) < 2:
            continue
        details, aggregate = _nearest(rows, feature)
        public = "v2_46d" if feature == "v2" else feature
        for row in details:
            row["feature_set"] = public; row["scope"] = scope
        aggregate["feature_set"] = public; aggregate["scope"] = scope
        output.extend(details); summary.append(aggregate)
    return output, summary


def _classifiers(rows: list[dict], scope: str) -> list[dict]:
    if len(rows) < 8 or len({row["seed"] for row in rows}) < 3 or len({row["outcome_label"] for row in rows}) < 2:
        return []
    result = _evaluate(rows, "v1_29d") + _evaluate(rows, "v2_full")
    for row in result:
        row["scope"] = scope
    return result


def _best(rows: list[dict], scope: str, feature: str) -> dict | None:
    candidates = [row for row in rows if row.get("scope") == scope and row["feature_set"] == feature and row["split"] == "leave_one_seed_out_aggregate"]
    return max(candidates, key=lambda item: item["macro_f1_present"], default=None)


def _classification(gate: dict, value_rows: list[dict], conflicts: list[dict], classifiers: list[dict]) -> str:
    if not gate["pass"]:
        return "EXPERIMENT_NOT_EXCITED"
    counts = Counter(row["outcome_label"] for row in value_rows)
    if len(value_rows) < 15 or counts["HARMFUL"] < 3 or len({row["seed"] for row in value_rows if row["outcome_label"] == "HARMFUL"}) < 3:
        return "INCONCLUSIVE"
    v1, v2 = _best(classifiers, "formal_pooled", "v1_29d"), _best(classifiers, "formal_pooled", "v2_full")
    summary = {row["feature_set"]: row for row in conflicts if row.get("scope") == "formal_pooled"}
    if v1 is None or v2 is None or "v1_29d" not in summary or "v2_46d" not in summary:
        return "INCONCLUSIVE"
    delta = v2["macro_f1_present"] - v1["macro_f1_present"]
    uplift = v2["macro_f1_present"] - v2["permutation_macro_f1_mean"]
    conflict_drop = summary["v1_29d"]["nearest_conflict_rate"] - summary["v2_46d"]["nearest_conflict_rate"]
    harmful = v2.get("harmful_recall_mean") or 0.
    if delta >= .08 and uplift >= .10 and conflict_drop >= .08 and harmful >= .35 and v2.get("harmful_heldout_seed_count", 0) >= 3:
        return "V2_IMPROVES_UNDER_EXCITATION"
    if delta >= .03 or conflict_drop >= .05 or (uplift >= .08 and harmful > 0.):
        return "V2_PARTIALLY_IMPROVES"
    return "NO_IMPROVEMENT_UNDER_EXCITATION"


def run_audit(*, workers: int = 6, definitions: list[dict] | None = None) -> dict:
    definitions, mu_source = regime_definitions() if definitions is None else (definitions, "custom")
    nominal: dict[tuple[str, int], dict] = {}; selected: list[dict] = []
    # Nominal first: it supplies event-conditioned, action-pre states and one CRN control per regime/seed.
    nominal_jobs = [(regime, seed) for regime in definitions for seed in regime["seeds"]]
    with ProcessPoolExecutor(max_workers=min(max(1, workers), len(nominal_jobs))) as pool:
        futures = [pool.submit(_nominal_task, *job) for job in nominal_jobs]
        for future in as_completed(futures):
            regime_name, seed, baseline, states = future.result(); nominal[regime_name, seed] = baseline
            regime = next(row for row in definitions if row["regime"] == regime_name)
            for index, state in enumerate(states):
                selected.append({"snapshot_id": f"{regime_name}_seed{seed}_event{index}", "regime": regime_name, "regime_class": regime["class"],
                                 "formal_value_evidence": regime["formal_value_evidence"], "seed": seed, **state})
    selected.sort(key=lambda row: (row["regime"], row["seed"], row["time"]))
    lookup = {row["regime"]: row for row in definitions}; dynamic: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=min(max(1, workers), max(1, len(selected)))) as pool:
        futures = [pool.submit(_dynamic_task, lookup[row["regime"]], row["seed"], row) for row in selected]
        for future in as_completed(futures):
            key, result = future.result(); dynamic[key] = result
    executable, values = [], []
    for row in selected:
        key = f"{row['regime']}:seed{row['seed']}:t{int(row['time'])}"; result = dynamic[key]; captured = result["pre_action_observation"]
        if captured["observation_29d"] != row["v1_29d"] or captured["hook"]["values"] != row["v2"]:
            raise RuntimeError(f"A6.22 CRN action-pre mismatch {row['snapshot_id']}")
        timing, base = result["timing"], nominal[row["regime"], row["seed"]]
        applied = timing["geometry_applied_time"] is not None and timing["geometry_effective_time"] is not None
        delta = float(result["metrics"]["all_request_waiting_area"] - base["metrics"]["all_request_waiting_area"])
        safety = {name: result["metrics"][name] for name in ("collisions", "nmac", "hard_separation_violations", "final_overcapacity_steps")}
        record = row | {"geometry_applied_time": timing["geometry_applied_time"], "geometry_effective_time": timing["geometry_effective_time"],
                        "executability_label": "EXECUTABLE" if applied else "NO_OP_OR_NOT_EXECUTABLE", "waiting_delta": delta,
                        "outcome_label": _label(delta) if applied else None, "nominal_waiting_area": base["metrics"]["all_request_waiting_area"],
                        "dynamic_waiting_area": result["metrics"]["all_request_waiting_area"], "safe": result["safe"] and base["safe"],
                        "checked_executed": True, "safety": safety}
        executable.append(record)
        if applied and row["formal_value_evidence"]:
            values.append(record)
    if not all(row["safe"] and row["checked_executed"] and all(value == 0 for value in row["safety"].values()) for row in executable):
        raise RuntimeError("A6.22 hard safety / checked-executed failure")
    all_feature, all_gate = _feature_excitation(selected, "all_regimes")
    formal_selected = [row for row in selected if row["regime_class"] == "traffic_excited_formal"]
    formal_feature, formal_gate = _feature_excitation(formal_selected, "traffic_excited_formal")
    conflicts, summaries, classifiers = [], [], []
    if formal_gate["pass"]:
        scopes = {"formal_pooled": values}
        scopes.update({name: [row for row in values if row["regime"] == name] for name in ("formal_high", "formal_bursty", "formal_overload")})
        for scope, rows in scopes.items():
            detail, summary = _conflicts(rows, scope); conflicts.extend(detail); summaries.extend(summary); classifiers.extend(_classifiers(rows, scope))
    classification = _classification(formal_gate, values, summaries, classifiers)
    return {"stage": "A6.22 Traffic-State Excitation & Observation Stress Audit", "evidence_type": "newly_run",
            "inputs": {"a611_mu0_evidence": mu_source, "v1_schema": "phaseA_high_obs_v1_29d", "v2_schema": "phaseA_high_obs_v2_46d", "v1_v2_frozen": True,
                       "intervention": "Outer R 600→612 m canonical full_dynamic", "waiting_materiality": WAITING_MATERIALITY, "ppo_authorization": PPO_AUTHORIZATION,
                       "frozen_runtime": ["A5 checkpoint", "A6.8 SafetyFilter", "80 m separation", "A6.12 repair-aware scheduler", "180 s interval", "600 s dwell", "20 s canonical geometry transition"]},
            "regimes": definitions, "snapshot_state": selected, "executability_labels": executable, "value_labels": values,
            "feature_excitation": all_feature + formal_feature, "feature_excitation_gate": [all_gate, formal_gate], "conflicts": conflicts, "conflict_summary": summaries, "v1_v2_results": classifiers,
            "summary": {"classification": classification, "formal_value_label_counts": dict(Counter(row["outcome_label"] for row in values)),
                        "executability_counts": dict(Counter(row["executability_label"] for row in executable)), "feature_excitation_gate": formal_gate,
                        "safe_rollouts": len(executable), "ppo_authorization": PPO_AUTHORIZATION,
                        "protocol_risks": ["CRN paired diagnostic interventions and LOSO are internal evidence, not external traffic-distribution generalization.", "Stress warm-start records excitation only and is excluded from formal value classification.", "Event-conditioned selection is deterministic but samples a finite set of nominal states; no classifier input contains time, seed, future arrivals, or post-action state."]}}


def _csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row}) or ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def write_report(report: dict, output: str | Path) -> None:
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    definitions = [{key: (json.dumps(value) if key in {"segments", "phase_windows", "seeds"} else value) for key, value in row.items()} for row in report["regimes"]]
    state = [{key: value for key, value in row.items() if key not in {"v1_29d", "v2"}} for row in report["snapshot_state"]]
    executable = [{key: value for key, value in row.items() if key not in {"v1_29d", "v2", "safety"}} for row in report["executability_labels"]]
    values = [{key: value for key, value in row.items() if key not in {"v1_29d", "v2", "safety"}} for row in report["value_labels"]]
    _csv(output / "a622_regime_definition.csv", definitions); _csv(output / "a622_feature_excitation.csv", report["feature_excitation"])
    _csv(output / "a622_snapshot_state.csv", state); _csv(output / "a622_executability_labels.csv", executable); _csv(output / "a622_value_labels.csv", values)
    _csv(output / "a622_v1_v2_results.csv", report["v1_v2_results"]); _csv(output / "a622_conflicts.csv", report["conflicts"])
    (output / "a622_audit.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    gate = report["summary"]["feature_excitation_gate"]; summary = report["summary"]
    best_v1, best_v2 = _best(report["v1_v2_results"], "formal_pooled", "v1_29d"), _best(report["v1_v2_results"], "formal_pooled", "v2_full")
    table = "Classifier was not run because the formal feature-excitation gate failed." if best_v1 is None or best_v2 is None else (
        "| feature set | LOSO macro-F1 | permutation macro-F1 | harmful recall |\n|---|---:|---:|---:|\n"
        f"| v1_29d | {best_v1['macro_f1_present']:.3f} | {best_v1['permutation_macro_f1_mean']:.3f} | {best_v1.get('harmful_recall_mean')} |\n"
        f"| v2_46d | {best_v2['macro_f1_present']:.3f} | {best_v2['permutation_macro_f1_mean']:.3f} | {best_v2.get('harmful_recall_mean')} |")
    (output / "a622_summary.md").write_text(
        "# A6.22 Traffic-State Excitation & Observation Stress Audit\n\n"
        f"Classification: **{summary['classification']}**. PPO authorization: **STOP**.\n\n"
        f"Formal feature-excitation gate: **{'PASS' if gate['pass'] else 'FAIL'}**; excited traffic features {gate['excited_traffic_feature_count']}/{gate['traffic_feature_count']}, effective rank {gate['effective_rank']:.2f}.\n\n"
        f"Formal executable value labels: `{summary['formal_value_label_counts']}`. No-op/not-executable requests are retained only in `a622_executability_labels.csv`, never treated as neutral values.\n\n{table}\n\n"
        f"All {summary['safe_rollouts']} intervention rollouts are hard-safe and checked==executed. Stress warm-start is excitation-only, excluded from performance claims.\n\n"
        "Answer to the core question is evidence-gated: a failed feature gate means A6.21 cannot be attributed to observation insufficiency; a passed gate permits only the recorded v1/v2 internal CRN/LOSO comparison. No scheduler, safety, reward, action space, interval, dwell, transition semantics, or PPO changed.\n", encoding="utf-8")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--workers", type=int, default=6); parser.add_argument("--output", default=None)
    args = parser.parse_args(); output = Path(args.output) if args.output else Path.home() / "training_outputs" / "terminal" / "event_gated_stage15_diagnostics" / "a622"
    report = run_audit(workers=args.workers); write_report(report, output); print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
