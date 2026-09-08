"""Pure-rollout V3 L1 scenario-mixture coverage calibration.

No PPO update is performed here.  The diagnostic deliberately reports both
the safety-qualified L0 counterfactual label and the environment's native
``teacher_corrected_action`` trace, so a scenario-coverage conclusion cannot
silently conflate the two teacher semantics.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from ..algo.shared_ppo import SharedPPO
from ..core.checkpoints import load_checkpoint
from ..core.low_level_action import HEAD_NAMES
from ..core.merge_scheduler import LowLevelIntent
from ..core.runtime_traffic_safety import SafetyInvariantError
from ..envs.parallel_env import TerminalParallelEnv
from .run_v3_l0_teacher_dataset import _counterfactual_release, _label_after_step
from .train_low_v2_ppo import _episode_design, l1_full_teacher_config


RATIOS = (.10, .20, .30)
# This is a coverage *preflight*, not a performance pilot.  The initial
# release decision is the qualified target state, so one tick is the smallest
# faithful SafetyFilter/Teacher witness.  Completion/throughput are reported
# as one-tick diagnostics only, never treated as performance evidence.
EPISODES_PER_RATIO = 12
MAX_STEPS = 1
TARGET_KINDS = ("RELEASE_to_WAIT", "WAIT_to_RELEASE", "correct_RELEASE", "correct_WAIT")


def _read_rows(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def _kind(row: dict) -> str | None:
    request, target = row["a_requested"]["release_index"], row["a_teacher"]["release_index"]
    if request == 1 and target == 0: return "RELEASE_to_WAIT"
    if request == 0 and target == 1: return "WAIT_to_RELEASE"
    if request == 1 and target == 1: return "correct_RELEASE"
    if request == 0 and target == 0: return "correct_WAIT"
    return None


def _target_sources(raw_dataset: str | Path) -> dict[str, list[dict]]:
    groups = {name: [] for name in TARGET_KINDS}
    for row in _read_rows(raw_dataset):
        # Step-zero warm states are direct, deterministic resets.  They are
        # not state injection and do not alter airspace, traffic rate, or
        # SafetyFilter behaviour.
        if row.get("low_action_schema") != "V3" or row.get("step") != 0 or row.get("cohort") not in {"warm_n10", "warm_n15"}:
            continue
        kind = _kind(row)
        if kind is not None: groups[kind].append(row)
    for kind, values in groups.items():
        values.sort(key=lambda row: (row["cohort"], row["seed"], row["episode"], json.dumps(row["observation_v2"])))
        if not values: raise RuntimeError(f"no step-zero safety-qualified targeted source for {kind}")
    return groups


def _load_policy(checkpoint: str | Path) -> SharedPPO:
    policy = SharedPPO(87, teacher_bc_weight=.05)
    config = l1_full_teacher_config(medium_rate=.001575, reset_mode="warm_start_rings", demand_model="finite_cohort",
                                    warm_start_uavs=8, cohort_uavs=8, horizon_seconds=1200.0)
    load_checkpoint(checkpoint, policy, config)
    return policy


def _target_config(row: dict):
    count = int(row["scenario"]["n_uavs"])
    return l1_full_teacher_config(medium_rate=.001575, reset_mode="warm_start_rings", demand_model="finite_cohort",
                                  warm_start_uavs=count, cohort_uavs=count, horizon_seconds=float(MAX_STEPS))


def _rollout(policy: SharedPPO, config, seed: int, *, source: str, source_kind: str | None) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    env = TerminalParallelEnv(config); observations, _ = env.reset(seed)
    counts = Counter(); release_opportunities = 0; requested_actions = 0; shield = 0
    completion_before = len(env.completed_uavs)
    try:
        for _ in range(MAX_STEPS):
            ids = list(observations)
            if not ids:
                observations, _, _, _, _ = env.step({}); continue
            actor_obs = np.stack([observations[uid] for uid in ids]).astype(np.float32, copy=False)
            batch = policy.act_details(actor_obs, actor_obs, deterministic=False)
            actions = {uid: action for uid, action in zip(ids, batch.normalized)}
            probes = {uid: _counterfactual_release(env, env.active[int(uid)], LowLevelIntent.from_array(action))
                      for uid, action in actions.items() if action[0] <= config.release_threshold and int(uid) in env.active}
            observations, _, _, _, info = env.step(actions)
            for uid in ids:
                trace = info.get("action_traces", {}).get(uid)
                if trace is None: continue
                requested = LowLevelIntent.from_array(trace["requested_low_action"])
                teacher, _ = _label_after_step(env, uid, requested, probes.get(uid, (None, None)), trace,
                                                info.get("release_decisions", {}).get(uid))
                request_index, teacher_index = int(requested.release_request > 0), int(teacher.release_request > 0)
                release_opportunities += 1; requested_actions += 1
                if request_index == 1 and teacher_index == 0: counts["RELEASE_to_WAIT"] += 1
                elif request_index == 0 and teacher_index == 1: counts["WAIT_to_RELEASE"] += 1
                elif request_index == 1: counts["correct_RELEASE"] += 1
                else: counts["correct_WAIT"] += 1
                corrected = trace.get("teacher_corrected_action")
                if corrected is not None:
                    corrected_intent = LowLevelIntent.from_array(corrected)
                    if int(corrected_intent.release_request > 0) != request_index:
                        counts["native_trace_release_correction"] += 1
                shield += int(bool(trace.get("shield_used", False)))
            if env.arrivals.exhausted() and not env.active and not env.admission.queue.items and not env.service_uavs: break
    except SafetyInvariantError as error:
        raise RuntimeError(f"SafetyInvariantError source={source} kind={source_kind} seed={seed}: {error}") from error
    safety = {"collision": int(env.telemetry.get("collisions", 0)), "NMAC": int(env.telemetry.get("nmac", 0)),
              "hard": int(env.telemetry.get("hard_separation_violations", 0)), "SafetyInvariantError": 0,
              "uncertified_component": 0}
    if any(safety.values()): raise RuntimeError(f"safety gate failed source={source} kind={source_kind} seed={seed}: {safety}")
    return {"source": source, "source_kind": source_kind, "seed": seed, "release_decision_opportunities": release_opportunities,
            "requested_actions": requested_actions, "release_counts": dict(counts), "completion": len(env.completed_uavs) - completion_before,
            "throughput": (len(env.completed_uavs) - completion_before) / max(1, env.step_count),
            "queue": len(env.admission.queue.items), "shield_burden": 100.0 * shield / max(1, release_opportunities), "safety": safety}


def _aggregate(rows: list[dict], ratio: float) -> dict:
    counts = Counter(); safety = Counter()
    for row in rows:
        counts.update(row["release_counts"]); safety.update(row["safety"])
    opp = sum(row["release_decision_opportunities"] for row in rows)
    target_rows = [row for row in rows if row["source"] == "targeted"]
    native = counts["native_trace_release_correction"]
    label_corrections = counts["RELEASE_to_WAIT"] + counts["WAIT_to_RELEASE"]
    return {"targeted_ratio_nominal": ratio, "episodes": len(rows), "natural_source_episodes": len(rows) - len(target_rows),
            "targeted_source_episodes": len(target_rows), "targeted_source_kind_counts": dict(Counter(row["source_kind"] for row in target_rows)),
            "total_release_decision_opportunities": opp, "release_teacher_corrections_label_semantics": label_corrections,
            "RELEASE_to_WAIT": counts["RELEASE_to_WAIT"], "WAIT_to_RELEASE": counts["WAIT_to_RELEASE"],
            "correct_RELEASE": counts["correct_RELEASE"], "correct_WAIT": counts["correct_WAIT"],
            "release_correction_per_100_release_intents": 100.0 * label_corrections / max(1, opp),
            "native_teacher_trace_release_corrections": native,
            "completion": sum(row["completion"] for row in rows), "throughput": float(np.mean([row["throughput"] for row in rows])),
            "queue": float(np.mean([row["queue"] for row in rows])), "shield_burden": float(np.mean([row["shield_burden"] for row in rows])),
            "safety": dict(safety)}


def run_scenario_mixture_calibration(*, raw_dataset: str | Path, policy_checkpoint: str | Path, output_dir: str | Path) -> dict:
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True)
    policy, sources = _load_policy(policy_checkpoint), _target_sources(raw_dataset)
    results = []
    for ratio in RATIOS:
        targeted_n = round(EPISODES_PER_RATIO * ratio)
        rows = []
        for episode in range(EPISODES_PER_RATIO):
            if episode < targeted_n:
                kind = TARGET_KINDS[episode % len(TARGET_KINDS)]; source_row = sources[kind][episode % len(sources[kind])]
                rows.append(_rollout(policy, _target_config(source_row), int(source_row["seed"]), source="targeted", source_kind=kind))
            else:
                config, _ = _episode_design(episode, medium_rate=.001575, horizon_seconds=float(MAX_STEPS))
                rows.append(_rollout(policy, config, 8101 + episode, source="natural", source_kind=None))
        summary = _aggregate(rows, ratio); (destination / f"coverage_ratio_{int(ratio * 100)}.json").write_text(json.dumps({"summary": summary, "episodes": rows}, indent=2), encoding="utf-8")
        results.append(summary)
    # Native PPO/BC observes only ``teacher_corrected_action``.  A zero here
    # means changing reset frequencies alone cannot supply release-head BC.
    eligible = [row for row in results if row["RELEASE_to_WAIT"] > 0 and row["WAIT_to_RELEASE"] > 0 and
                row["native_teacher_trace_release_corrections"] > 0 and row["natural_source_episodes"] > row["targeted_source_episodes"]]
    selected = min(eligible, key=lambda row: row["targeted_ratio_nominal"]) if eligible else None
    report = {"stage": "V3_L1_C_scenario_mixture_coverage_calibration", "evidence_type": "newly_run",
              "frozen": {"action_schema": "V3", "teacher": "FULL", "mu_hat": .003, "lambda": .001575,
                         "reward_changed": False, "bc_weight_changed": False, "ppo_hyperparameters_changed": False},
              "protocol": {"ratios": RATIOS, "episodes_per_ratio": EPISODES_PER_RATIO, "max_steps": MAX_STEPS,
                           "policy_checkpoint": str(policy_checkpoint), "raw_target_source": str(raw_dataset)},
              "ratios": results, "selected_targeted_ratio": None if selected is None else selected["targeted_ratio_nominal"],
              "coverage_calibration_passed": selected is not None,
              "blocker": None if selected is not None else "Native Teacher FULL action traces contain no release-head correction; L0 counterfactual labels do, so a scenario-only change cannot create release BC supervision.",
              "fresh_coverage_bridge_pilot_started": False, "formal_L1_training_allowed": False}
    (destination / "v3_l1_scenario_mixture_calibration.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# V3 L1 Scenario Mixture Coverage Calibration", "", "Evidence type: newly_run; no PPO update was executed.", ""]
    for row in results:
        lines += [f"## {int(row['targeted_ratio_nominal'] * 100)}% targeted", "", f"- Natural / targeted episodes: {row['natural_source_episodes']} / {row['targeted_source_episodes']}",
                  f"- Opportunities: {row['total_release_decision_opportunities']}; label R→W / W→R: {row['RELEASE_to_WAIT']} / {row['WAIT_to_RELEASE']}",
                  f"- Native Teacher trace release corrections: {row['native_teacher_trace_release_corrections']}",
                  f"- Completion / throughput / queue / shield: {row['completion']} / {row['throughput']:.6f} / {row['queue']:.3f} / {row['shield_burden']:.3f}", f"- Safety: {row['safety']}", ""]
    lines += [f"selected_targeted_ratio = {report['selected_targeted_ratio']}", f"V3_L1_COVERAGE_CALIBRATION = {'PASS' if report['coverage_calibration_passed'] else 'FAIL'}", f"blocker = {report['blocker']}", ""]
    (destination / "v3_l1_scenario_mixture_calibration.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Pure V3 L1 scenario-mixture coverage calibration; no PPO.")
    parser.add_argument("--raw-dataset", required=True); parser.add_argument("--policy-checkpoint", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); report = run_scenario_mixture_calibration(raw_dataset=args.raw_dataset, policy_checkpoint=args.policy_checkpoint, output_dir=args.output)
    print(json.dumps({key: report[key] for key in ("selected_targeted_ratio", "coverage_calibration_passed", "fresh_coverage_bridge_pilot_started")}, indent=2))


if __name__ == "__main__": _main()
