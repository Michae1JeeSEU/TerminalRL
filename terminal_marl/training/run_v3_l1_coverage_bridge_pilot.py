"""Fresh V3 L1 coverage-bridge PPO pilot.

This is intentionally a narrow follow-up to the scenario-mixture calibration:
it creates a new categorical actor/critic, retains the natural L1 mixture,
uses its already-selected 10% targeted warm-start source, and stops before a
second PPO update can be collected without release supervision.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from ..algo.shared_ppo import RolloutBuffer, SharedPPO, Transition
from ..core.checkpoints import save_checkpoint
from ..core.observations import LOW_OBS_V2
from ..core.runtime_traffic_safety import SafetyInvariantError
from .audit_low_teacher_dataset import audit_teacher_dataset
from .run_v3_l1_release_intervention_diagnostic import evaluate_release_cohort
from .run_v3_l1_scenario_mixture_calibration import TARGET_KINDS, _target_sources
from .run_v3_l1_short_pilot import (
    MU_HAT, LAMBDA, V3L1ShortPilotProtocol, _collect, _distribution_metrics,
    _model_state, _records_metrics, _serializable, _training_groups,
)
from .train_low_v2_ppo import _episode_design, _read_frozen_capacity, _read_seed_partition, _rollout_task, l1_full_teacher_config


TARGETED_RATIO = .10
TARGETED_EPISODES = (0, 12)  # round(12 * .10) once in each frozen 12-episode block
SNAPSHOTS = {0: "baseline", 3: "update3", 6: "update6"}
SAFETY_KEYS = ("collisions", "nmac", "hard_separation_violations")


def _json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_serializable(payload), indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(_serializable(row), allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def _release_coverage(metrics: dict) -> dict:
    head = metrics["distribution"]["heads"]["release"]
    transitions = metrics["distribution"]["transitions"]
    return {
        "release_decision_opportunities": transitions,
        "release_teacher_correction_count": int(round(transitions * head["teacher_correction_rate"])),
        "RELEASE_to_WAIT": int(head["RELEASE_to_WAIT"]),
        "WAIT_to_RELEASE": int(head["WAIT_to_RELEASE"]),
        "P_RELEASE": head["policy_probability"]["RELEASE"],
        "release_entropy": head["entropy"],
        "release_logits_or_probabilities": head["policy_probability"],
        "false_release": head["false_release_rate"],
        "false_wait": head["false_wait_rate"],
        "speed_teacher_correction_rate": metrics["distribution"]["heads"]["speed"]["teacher_correction_rate"],
        "merge_teacher_correction_rate": metrics["distribution"]["heads"]["merge_time"]["teacher_correction_rate"],
    }


def _safety_ok(metrics: dict) -> bool:
    return all(metrics["environment"][key] == 0 for key in SAFETY_KEYS)


def _checkpoint(destination: Path, label: str, policy: SharedPPO, root) -> str:
    path = destination / f"{label}.pt"
    save_checkpoint(path, policy, root, stage="V3_L1_coverage_bridge", snapshot=label,
                    fresh_initialization=True, resume_checkpoint=None)
    return str(path)


def _cohort_metrics(policy: SharedPPO, cohort: dict) -> dict:
    # ``evaluate_release_cohort`` owns the frozen teacher labels and reports
    # P(RELEASE | Teacher WAIT/RELEASE), false decisions, entropy and logits.
    return evaluate_release_cohort(policy, cohort)


def _targeted_warm_config(source: dict, *, horizon_seconds: float):
    """Replay the qualified warm-start identity for a full L1 episode.

    Calibration used a one-tick horizon solely to count labels.  The PPO
    bridge must keep the actual L1 episode length and only change its reset
    source, otherwise the targeted fraction would also alter returns.
    """
    count = int(source["scenario"]["n_uavs"])
    return l1_full_teacher_config(medium_rate=LAMBDA, reset_mode="warm_start_rings",
                                  demand_model="finite_cohort", warm_start_uavs=count,
                                  cohort_uavs=count, horizon_seconds=horizon_seconds)


def _healthy_ppo(row: dict) -> bool:
    return all(np.isfinite(row["ppo"][key]) for key in (
        "actor_loss", "value_loss", "entropy", "entropy_release", "entropy_speed",
        "entropy_merge_time", "approx_kl", "clip_fraction", "explained_variance", "grad_norm",
    ))


def _decision(validation: dict, updates: list[dict], *, safety: bool) -> tuple[str, str, bool]:
    base, final = validation["baseline"], validation["update6"]
    discrimination = (
        final["P_RELEASE_given_teacher_WAIT"] < base["P_RELEASE_given_teacher_WAIT"] and
        final["P_RELEASE_given_teacher_RELEASE"] >= .90 * base["P_RELEASE_given_teacher_RELEASE"] and
        final["false_release_rate"] < base["false_release_rate"] and
        final["false_wait_rate"] <= base["false_wait_rate"] + .05 and
        final["per_kind"]["RELEASE_to_WAIT"]["predicted_RELEASE_rate"] <
        base["per_kind"]["RELEASE_to_WAIT"]["predicted_RELEASE_rate"]
    )
    burden = [row["environment"]["teacher_interventions_per_100_release_intents"] for row in updates]
    burden_ok = all(value is not None for value in burden) and burden[-1] <= burden[0]
    shield = [row["environment"]["shield_interventions_per_100_release_intents"] for row in updates]
    shield_ok = all(value is not None for value in shield) and shield[-1] <= shield[0]
    ppo = all(_healthy_ppo(row) for row in updates)
    completion = updates[-1]["environment"]["completed"] >= .5 * updates[0]["environment"]["completed"]
    if discrimination and burden_ok and shield_ok and ppo and completion and safety:
        return "PASS", "PASS", True
    # Coverage is hard-gated inside the loop.  A completed, covered pilot that
    # still cannot discriminate belongs to the objective/representation path.
    return "FAIL", "FAIL", False


def _markdown(report: dict) -> str:
    lines = ["# V3 L1 Coverage-Bridge PPO Pilot", "", "## Frozen protocol", "",
             "- Fresh V3 factorized categorical actor/critic; Teacher = FULL; release labels enabled.",
             "- Targeted warm-start ratio: 10% nominal (one target reset per 12-episode block); remaining episodes use the natural L1 mixture.",
             "- μ_hat = 0.003; λ = 0.001575; nominal geometry; priority 1; standard mission; SOC 0.8.",
             "- Reward, BC weight, PPO hyperparameters, Observation V2 and safety stack unchanged.", "",
             "## Release supervision coverage", ""]
    for row in report["updates"]:
        coverage = row["release_supervision"]
        lines.append(f"- Update {row['update']}: opportunities={coverage['release_decision_opportunities']}; corrections={coverage['release_teacher_correction_count']}; R→W/W→R={coverage['RELEASE_to_WAIT']}/{coverage['WAIT_to_RELEASE']}")
    lines += ["", "## Fixed 224-state release cohort", ""]
    for label in ("baseline", "update3", "update6"):
        metric = report["release_cohort_validation"][label]
        lines.append(f"- {label}: P(R|Teacher WAIT/RELEASE)={metric['P_RELEASE_given_teacher_WAIT']:.4f}/{metric['P_RELEASE_given_teacher_RELEASE']:.4f}; false-release/wait={metric['false_release_rate']:.4f}/{metric['false_wait_rate']:.4f}; entropy={metric['release_entropy']:.4f}")
    lines += ["", "## Teacher burden and PPO health", ""]
    for row in report["updates"]:
        env, ppo = row["environment"], row["ppo"]
        lines.append(f"- Update {row['update']}: teacher/100={env['teacher_interventions_per_100_release_intents']}; shield/100={env['shield_interventions_per_100_release_intents']}; completion={env['completed']}; throughput={env['throughput_per_second']}; queue={env['external_queue_mean']}; KL={ppo['approx_kl']:.6f}; clip={ppo['clip_fraction']:.6f}; entropy={ppo['entropy']:.6f}; EV={ppo['explained_variance']:.6f}; grad={ppo['grad_norm']:.6f}")
    lines += ["", f"Safety = {'PASS' if report['safety_passed'] else 'FAIL'}",
              f"release_learning = {report['release_learning']}",
              f"V3_L1_COVERAGE_BRIDGE = {report['V3_L1_COVERAGE_BRIDGE']}",
              f"longer_L1_continuation_allowed = {str(report['longer_L1_continuation_allowed']).lower()}",
              "formal_L1_training_allowed = false", ""]
    return "\n".join(lines)


def run_coverage_bridge(*, c0_report: str | Path, teacher_dataset: str | Path, raw_dataset: str | Path,
                        release_cohort: str | Path, output_dir: str | Path) -> dict:
    protocol = V3L1ShortPilotProtocol()
    audit = audit_teacher_dataset(teacher_dataset)
    if not audit.passed:
        raise RuntimeError("teacher dataset audit failed: " + "; ".join(audit.errors[:10]))
    mu_hat, interval = _read_frozen_capacity(c0_report)
    if mu_hat != MU_HAT or abs(sum(interval) / 2.0 - LAMBDA) > 1e-12:
        raise RuntimeError("frozen capacity calibration drift")
    cohort = json.loads(Path(release_cohort).read_text(encoding="utf-8"))
    if len(cohort.get("rows", [])) != 224:
        raise RuntimeError("coverage bridge requires the exact frozen 224-state release cohort")
    sources = _target_sources(raw_dataset)
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True)
    root = l1_full_teacher_config(medium_rate=LAMBDA, reset_mode="warm_start_rings", demand_model="finite_cohort",
                                  warm_start_uavs=8, cohort_uavs=8, horizon_seconds=float(protocol.max_steps))
    torch.manual_seed(2401); np.random.seed(2401)
    policy = SharedPPO(LOW_OBS_V2.dimension, teacher_bc_weight=protocol.teacher_bc_weight)
    partition = _read_seed_partition(teacher_dataset)
    snapshots: dict[str, str] = {"baseline": _checkpoint(destination, "baseline", policy, root)}
    validation = {"baseline": _cohort_metrics(policy, cohort)}
    updates: list[dict] = []; episodes: list[dict] = []; seed_cursor = 0
    try:
        for update in range(protocol.updates):
            state = _model_state(policy); tasks = []; designs = []
            for episode in range(protocol.episodes_per_update):
                global_episode = update * protocol.episodes_per_update + episode
                if global_episode in TARGETED_EPISODES:
                    kind = TARGET_KINDS[global_episode % len(TARGET_KINDS)]
                    source = sources[kind][0]
                    config, design, seed, source_name = _targeted_warm_config(
                        source, horizon_seconds=float(protocol.max_steps)), {"targeted_kind": kind}, int(source["seed"]), "targeted_warm_start"
                else:
                    config, design = _episode_design(global_episode, medium_rate=LAMBDA, horizon_seconds=float(protocol.max_steps))
                    seed, source_name = partition.train[seed_cursor % len(partition.train)], "natural_l1"
                    seed_cursor += 1
                tasks.append((state, config, seed, update, episode, protocol.max_steps, False, str(destination / "safety_action_tapes")))
                designs.append({**design, "source": source_name})
            collected = _collect(tasks, protocol.rollout_workers)
            buffer = RolloutBuffer()
            for (rows, summary), design in zip(collected, designs):
                if any((summary.collisions, summary.nmac, summary.hard_separation_violations)):
                    raise RuntimeError("TRAINING_SAFETY_FAILURE: collision/NMAC/hard event; exact ActionTape retained")
                for row in rows: buffer.add(row)
                episodes.append({**asdict(summary), **design, "update": update + 1})
            metrics = _records_metrics(collected, state)
            coverage = _release_coverage(metrics)
            if coverage["release_teacher_correction_count"] <= 0 or coverage["RELEASE_to_WAIT"] <= 0:
                raise RuntimeError("TRAINING_COVERAGE_FAILURE: release teacher correction or RELEASE→WAIT absent")
            ppo = policy.update(buffer, epochs=protocol.ppo_epochs, minibatch_size=protocol.minibatch_size)
            record = {"update": update + 1, "environment": metrics["environment"], "release_supervision": coverage,
                      "speed_merge_corrections": {"speed": coverage["speed_teacher_correction_rate"], "merge_time": coverage["merge_teacher_correction_rate"]},
                      "by_scenario": _training_groups(collected, state), "ppo": ppo}
            if not _healthy_ppo(record):
                raise RuntimeError("PPO_HEALTH_FAILURE")
            updates.append(record); _jsonl(destination / "coverage_bridge_updates.jsonl", updates)
            if update + 1 in (3, 6):
                label = SNAPSHOTS[update + 1]
                snapshots[label] = _checkpoint(destination, label, policy, root)
                validation[label] = _cohort_metrics(policy, cohort)
    except (SafetyInvariantError, RuntimeError, FloatingPointError) as error:
        partial = {"stage": "V3_L1_coverage_bridge", "status": "stopped", "reason": str(error), "updates": updates,
                   "snapshots": snapshots, "formal_L1_training_allowed": False}
        _json(destination / "v3_l1_coverage_bridge_pilot.json", partial)
        (destination / "v3_l1_coverage_bridge_pilot_summary.md").write_text(_markdown({**partial, "release_cohort_validation": validation,
            "safety_passed": False, "release_learning": "INCONCLUSIVE", "V3_L1_COVERAGE_BRIDGE": "FAIL", "longer_L1_continuation_allowed": False}), encoding="utf-8")
        raise
    safety = all(_safety_ok(row) for row in updates)
    learning, bridge, continuation = _decision(validation, updates, safety=safety)
    report = {"stage": "V3_L1_coverage_bridge", "status": "completed", "evidence_type": "newly_run",
              "frozen": {"action_schema": "V3_factorized_categorical", "teacher": "FULL", "teacher_release_label_enabled": True,
                         "targeted_warm_start_ratio_nominal": TARGETED_RATIO, "mu_hat": MU_HAT, "lambda": LAMBDA,
                         "geometry": "nominal", "priority": 1, "mission": "standard", "SOC": .8,
                         "reward_changed": False, "BC_weight_changed": False, "PPO_hyperparameters_changed": False,
                         "Observation_V2_changed": False, "safety_stack_changed": False},
              "protocol": asdict(protocol), "targeted_episode_positions": list(TARGETED_EPISODES), "snapshots": snapshots,
              "updates": updates, "episodes": episodes, "release_cohort_validation": validation,
              "safety_passed": safety, "release_learning": learning, "V3_L1_COVERAGE_BRIDGE": bridge,
              "longer_L1_continuation_allowed": continuation, "formal_L1_training_allowed": False}
    _jsonl(destination / "coverage_bridge_episodes.jsonl", episodes)
    _json(destination / "v3_l1_coverage_bridge_pilot.json", report)
    (destination / "v3_l1_coverage_bridge_pilot_summary.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run the fresh V3 L1 coverage-bridge PPO pilot; no resume support.")
    parser.add_argument("--c0-report", required=True); parser.add_argument("--teacher-dataset", required=True)
    parser.add_argument("--raw-dataset", required=True); parser.add_argument("--release-cohort", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run_coverage_bridge(c0_report=args.c0_report, teacher_dataset=args.teacher_dataset, raw_dataset=args.raw_dataset,
                                 release_cohort=args.release_cohort, output_dir=args.output)
    print(json.dumps({key: report[key] for key in ("release_learning", "V3_L1_COVERAGE_BRIDGE", "longer_L1_continuation_allowed")}, indent=2))


if __name__ == "__main__":
    _main()
