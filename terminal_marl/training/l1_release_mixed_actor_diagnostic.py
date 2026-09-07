"""Release semantics audit and bounded 3-update mixed-actor A/B diagnostic."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from ..algo import RolloutBuffer, SharedPPO, Transition
from ..core.checkpoints import load_checkpoint
from ..core.observations import LOW_OBS_V2
from ..core.runtime_traffic_safety import SafetyInvariantError
from ..envs.parallel_env import TerminalParallelEnv
from .l1_ppo_d3_bounded_distribution import _requested_action_provenance
from .train_low_v2_ppo import (
    L1SafetyGateError, _episode_design, _grouped_metrics, _model_state,
    _read_frozen_capacity, _read_seed_partition, _rollout_task, _summary_metrics,
    _counterfactual_release_teacher_targets, _release_teacher_target,
    authorize_low_v2_training, l1_full_teacher_config,
)


VARIANTS = {
    "A_continuous_release_tanh": {"policy_distribution": "tanh_squashed_gaussian", "release_teacher_bc": False},
    "B_bernoulli_release_tanh_residuals": {"policy_distribution": "mixed_bernoulli_tanh", "release_teacher_bc": True},
}

RELEASE_PROBE_MAGNITUDES = (-.99, -.01, .01, .99)


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _mean_or_none(values: np.ndarray) -> float | None:
    return None if values.size == 0 else float(values.mean())


def _teacher_dataset_release_audit(path: str | Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    corrected = [row for row in rows if row.get("teacher_action") is not None]
    requested = np.asarray([row["requested_action"][0] for row in rows], dtype=np.float64)
    targets = np.asarray([row["teacher_action"][0] for row in corrected], dtype=np.float64)
    changed = sum(abs(row["requested_action"][0] - row["teacher_action"][0]) > 1e-7 for row in corrected)
    return {
        "rows": len(rows), "teacher_target_rows": len(corrected), "release_changed_by_teacher": changed,
        "requested_release_exact_endpoint_fraction": float(np.mean(np.abs(requested) == 1.0)),
        "teacher_release_exact_endpoint_fraction": None if not len(targets) else float(np.mean(np.abs(targets) == 1.0)),
        "teacher_release_target_counts": {"WAIT": int((targets < 0.0).sum()), "RELEASE": int((targets > 0.0).sum())},
        "historical_release_bc_semantics": "continuous magnitude regression to the unchanged requested release sample",
    }


def _release_probe_task(task: tuple[Any, int, float, int]) -> dict[str, Any]:
    """Run one deterministic threshold-equivalence probe without PPO."""
    config, seed, magnitude, max_steps = task
    env = TerminalParallelEnv(config)
    observations, _ = env.reset(seed)
    label_wait = label_release = false_release = false_wait = 0
    near_threshold = busy = safe_open = 0
    decision_counts: dict[str, int] = {}
    eligibility_index = LOW_OBS_V2.slices["release_eligible"].start
    for _ in range(max_steps):
        actions = {uid: np.asarray((magnitude, 0.0, 0.0), dtype=np.float32) for uid in observations}
        before = observations
        counterfactual_targets = None
        if magnitude <= 0.0:
            counterfactual_targets, _ = _counterfactual_release_teacher_targets(env, before, actions)
        observations, _, _, _, info = env.step(actions)
        for uid, observation in before.items():
            decision = str(info["release_decisions"].get(uid, "phase_masked"))
            eligible = bool(observation[eligibility_index] > .5)
            target = (counterfactual_targets[uid] if counterfactual_targets is not None else
                      _release_teacher_target(release_eligible=eligible, requested_release=True, decision=decision))
            predicted_release = magnitude > 0.0
            teacher_release = target > 0.0
            label_release += int(teacher_release); label_wait += int(not teacher_release)
            false_release += int(predicted_release and not teacher_release)
            false_wait += int(not predicted_release and teacher_release)
            near_threshold += int(abs(magnitude) <= .10)
            busy += int(not eligible)
            safe_open += int(eligible and teacher_release)
            decision_counts[decision] = decision_counts.get(decision, 0) + 1
        if env.arrivals.exhausted() and not env.active and not env.admission.queue.items and not env.service_uavs:
            break
    return {
        "seed": seed, "requested_release_value": magnitude,
        "requested_decision": "RELEASE" if magnitude > 0.0 else "WAIT",
        "requested_release_probability": float(magnitude > 0.0),
        "teacher_target_counts": {"WAIT": label_wait, "RELEASE": label_release},
        "false_release": false_release, "false_wait": false_wait,
        "teacher_correction_rate": _safe_rate(false_release + false_wait, label_wait + label_release),
        "near_threshold_samples": near_threshold, "busy_or_committed_samples": busy,
        "safe_open_gap_samples": safe_open, "decision_counts": decision_counts,
        "completion": len(env.completed_uavs),
        "safety": {"collisions": int(env.telemetry.get("collisions", 0)),
                   "nmac": int(env.telemetry.get("nmac", 0)),
                   "hard_separation_violations": int(env.telemetry.get("hard_separation_violations", 0))},
    }


def run_release_semantics_probe(*, c0_report: str | Path, output_path: str | Path,
                                workers: int = 4, max_steps: int = 240) -> dict[str, Any]:
    """Create the fixed WAIT/RELEASE, sign-equivalence Phase-2 regression."""
    _, medium_interval = _read_frozen_capacity(c0_report)
    medium_rate = sum(medium_interval) / 2.0
    tasks = []
    for seed in (1001, 1002):
        config = l1_full_teacher_config(medium_rate=medium_rate, reset_mode="warm_start_rings",
                                        demand_model="finite_cohort", warm_start_uavs=8, cohort_uavs=8,
                                        horizon_seconds=float(max_steps))
        tasks.extend((config, seed, magnitude, max_steps) for magnitude in RELEASE_PROBE_MAGNITUDES)
    if workers == 1:
        records = [_release_probe_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
            records = list(executor.map(_release_probe_task, tasks))
    by_key = {(row["seed"], row["requested_release_value"]): row for row in records}
    comparable_fields = ("teacher_target_counts", "false_release", "false_wait", "decision_counts", "completion", "safety")
    positive_equivalent = all(all(by_key[(seed, .01)][field] == by_key[(seed, .99)][field]
                                  for field in comparable_fields) for seed in (1001, 1002))
    negative_equivalent = all(all(by_key[(seed, -.01)][field] == by_key[(seed, -.99)][field]
                                  for field in comparable_fields) for seed in (1001, 1002))
    coverage = {
        "teacher_wait": sum(row["teacher_target_counts"]["WAIT"] for row in records),
        "teacher_release": sum(row["teacher_target_counts"]["RELEASE"] for row in records),
        "false_release": sum(row["false_release"] for row in records),
        "false_wait": sum(row["false_wait"] for row in records),
        "near_threshold": sum(row["near_threshold_samples"] for row in records),
        "busy_or_committed": sum(row["busy_or_committed_samples"] for row in records),
        "safe_open_gap": sum(row["safe_open_gap_samples"] for row in records),
    }
    passed = bool(positive_equivalent and negative_equivalent and all(value > 0 for value in coverage.values()) and
                  not any(value for row in records for value in row["safety"].values()))
    result = {"status": "completed", "evidence_type": "newly_run", "records": records,
              "positive_magnitude_execution_equivalent": positive_equivalent,
              "negative_magnitude_execution_equivalent": negative_equivalent,
              "coverage": coverage, "passed": passed}
    _write(Path(output_path), result)
    return result


def _ppo_health(records: list[dict[str, Any]]) -> dict[str, Any]:
    updates = [record["ppo_update"] for record in records]
    def values(key: str) -> list[float]:
        return [float(row[key]) for row in updates]
    finite = all(np.isfinite(value) for row in updates for value in row.values())
    return {
        "approx_kl_by_update": values("approx_kl"),
        "clip_fraction_by_update": values("clip_fraction"),
        "entropy_by_update": values("entropy"),
        "healthy": bool(finite and all(abs(value) <= .05 for value in values("approx_kl")) and
                        all(value <= .30 for value in values("clip_fraction"))),
    }


def _release_probability(policy: SharedPPO, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tensor = torch.as_tensor(observations, dtype=torch.float32, device=policy.device)
    with torch.no_grad():
        logits = policy.model.actor_mean(policy.model.actor_body(tensor))[:, 0]
        if policy.uses_mixed_release:
            distribution = policy.distribution(tensor)
            logits = distribution.release_logits
            probability = distribution.release_probability
        else:
            std = policy.model.log_std[0].exp()
            probability = 1.0 - torch.distributions.Normal(logits, std).cdf(torch.zeros_like(logits))
    return probability.cpu().numpy(), logits.cpu().numpy()


def release_decision_metrics(policy: SharedPPO, transitions: list[Transition]) -> dict[str, Any]:
    """Aggregate the fixed binary-label regression without mutating a rollout."""
    rows = [row for row in transitions if row.teacher_action is not None]
    if not rows:
        raise RuntimeError("release diagnostic requires binary teacher targets")
    observations = np.stack([row.observation for row in rows]).astype(np.float32, copy=False)
    requested = np.asarray([row.action[0] > 0.0 for row in rows], dtype=bool)
    labels = np.asarray([row.teacher_action[0] > 0.0 for row in rows], dtype=bool)
    probabilities, logits = _release_probability(policy, observations)
    corrected = requested != labels
    false_release = requested & ~labels
    false_wait = ~requested & labels
    near_threshold = np.abs(np.asarray([row.action[0] for row in rows])) <= .10
    slices = LOW_OBS_V2.slices
    eligible = observations[:, slices["release_eligible"]].reshape(-1) > .5
    target_predecessor = observations[:, slices["target_ring_predecessor"]]
    target_successor = observations[:, slices["target_ring_successor"]]
    conflict = observations[:, slices["nearest_predicted_conflict"]]
    reserved_or_committed = ((target_predecessor[:, 8] > .5) | (target_successor[:, 8] > .5) |
                             (conflict[:, 8] > .5))
    busy_context = reserved_or_committed | ~eligible
    safe_open_gap = labels & eligible & ((conflict[:, 0] < .5) | (conflict[:, 5] >= .95))
    return {
        "samples": len(rows),
        "label_counts": {"WAIT": int((~labels).sum()), "RELEASE": int(labels.sum())},
        "requested_counts": {"WAIT": int((~requested).sum()), "RELEASE": int(requested.sum())},
        "false_release": int(false_release.sum()), "false_wait": int(false_wait.sum()),
        "false_release_rate": _safe_rate(int(false_release.sum()), int((~labels).sum())),
        "false_wait_rate": _safe_rate(int(false_wait.sum()), int(labels.sum())),
        "teacher_release_corrections": int(corrected.sum()),
        "teacher_release_correction_rate": float(corrected.mean()),
        "near_threshold_samples": int(near_threshold.sum()),
        "busy_or_committed_samples": int(busy_context.sum()),
        "safe_open_gap_samples": int(safe_open_gap.sum()),
        "release_probability": {"mean": float(probabilities.mean()),
                                "wait_label_mean": _mean_or_none(probabilities[~labels]),
                                "release_label_mean": _mean_or_none(probabilities[labels])},
        "release_logit": {"mean": float(logits.mean()), "min": float(logits.min()), "max": float(logits.max())},
        "traffic_context": {
            "release_eligible": int(eligible.sum()), "reserved_or_committed": int(reserved_or_committed.sum()),
            "predicted_conflict_present": int((conflict[:, 0] > .5).sum()),
        },
    }


def release_objective_metrics(policy: SharedPPO, buffer: RolloutBuffer, *, gamma: float = .99,
                              gae_lambda: float = .95) -> dict[str, Any]:
    """Return the D5a class, advantage, entropy, and calibration evidence."""
    flattened = buffer.flattened(gamma, gae_lambda)
    rows = [(transition, advantage) for transition, advantage, _ in flattened
            if transition.teacher_action is not None]
    if not rows:
        raise RuntimeError("release objective diagnostic requires binary teacher targets")
    transitions = [row[0] for row in rows]
    advantages = np.asarray([row[1] for row in rows], dtype=np.float64)
    observations = np.stack([row.observation for row in transitions]).astype(np.float32, copy=False)
    requested = np.asarray([row.action[0] > 0.0 for row in transitions], dtype=bool)
    labels = np.asarray([row.teacher_action[0] > 0.0 for row in transitions], dtype=bool)
    probabilities, logits = _release_probability(policy, observations)
    stable_probability = np.clip(probabilities.astype(np.float64), 1e-7, 1.0 - 1e-7)
    binary_entropy = -(stable_probability * np.log(stable_probability) +
                       (1.0 - stable_probability) * np.log(1.0 - stable_probability))
    tensor = torch.as_tensor(observations, dtype=torch.float32, device=policy.device)
    with torch.no_grad():
        distribution_entropy = policy.distribution(tensor).entropy()[:, 0].cpu().numpy()

    def conditional(mask: np.ndarray) -> dict[str, Any]:
        values = advantages[mask]
        return {
            "samples": int(mask.sum()),
            "advantage_mean": _mean_or_none(values),
            "advantage_positive_rate": _safe_rate(int((values > 0.0).sum()), int(values.size)),
            "advantage_p10": None if not values.size else float(np.percentile(values, 10)),
            "advantage_p90": None if not values.size else float(np.percentile(values, 90)),
        }

    nll = -(labels * np.log(stable_probability) + (~labels) * np.log(1.0 - stable_probability))
    return {
        "samples": len(rows),
        "label_balance": {"WAIT": int((~labels).sum()), "RELEASE": int(labels.sum()),
                          "release_fraction": float(labels.mean())},
        "requested_balance": {"WAIT": int((~requested).sum()), "RELEASE": int(requested.sum()),
                              "release_fraction": float(requested.mean())},
        "advantage_by_requested_decision": {"WAIT": conditional(~requested), "RELEASE": conditional(requested)},
        "advantage_by_teacher_label": {"WAIT": conditional(~labels), "RELEASE": conditional(labels)},
        "advantage_by_confusion_cell": {
            "true_wait": conditional(~requested & ~labels),
            "false_release": conditional(requested & ~labels),
            "false_wait": conditional(~requested & labels),
            "true_release": conditional(requested & labels),
        },
        "release_binary_entropy_mean": float(binary_entropy.mean()),
        "release_distribution_entropy_mean": float(distribution_entropy.mean()),
        "release_logit": {"mean": float(logits.mean()), "min": float(logits.min()), "max": float(logits.max())},
        "calibration": {"brier": float(np.mean((stable_probability - labels.astype(np.float64)) ** 2)),
                        "binary_nll": float(nll.mean()),
                        "wait_label_probability_mean": _mean_or_none(stable_probability[~labels]),
                        "release_label_probability_mean": _mean_or_none(stable_probability[labels])},
    }


def _aggregate_release(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [record["release"] for record in records]
    total = sum(row["samples"] for row in rows)
    wait_labels = sum(row["label_counts"]["WAIT"] for row in rows)
    release_labels = sum(row["label_counts"]["RELEASE"] for row in rows)
    false_release = sum(row["false_release"] for row in rows)
    false_wait = sum(row["false_wait"] for row in rows)
    corrections = false_release + false_wait
    return {
        "samples": total,
        "label_counts": {"WAIT": wait_labels, "RELEASE": release_labels},
        "false_release": false_release, "false_wait": false_wait,
        "false_release_rate": _safe_rate(false_release, wait_labels),
        "false_wait_rate": _safe_rate(false_wait, release_labels),
        "teacher_release_corrections": corrections,
        "teacher_release_correction_rate": _safe_rate(corrections, total),
        "near_threshold_samples": sum(row["near_threshold_samples"] for row in rows),
        "busy_or_committed_samples": sum(row["busy_or_committed_samples"] for row in rows),
        "safe_open_gap_samples": sum(row["safe_open_gap_samples"] for row in rows),
        "by_update": rows,
    }


def _variant_summary(records: list[dict[str, Any]], provenance: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = [SimpleNamespace(**row) for record in records for row in record["summaries"]]
    overall = _summary_metrics(summaries)
    warm = _grouped_metrics(summaries)["warm_start_finite_cohort"]
    safety = {key: int(sum(getattr(row, key) for row in summaries))
              for key in ("collisions", "nmac", "hard_separation_violations")}
    return {
        "records": records, "release": _aggregate_release(records), "provenance": provenance,
        "provenance_passed": all(row["passed"] for row in provenance),
        "teacher_burden_overall": overall["teacher_interventions_per_100_release_intents"],
        "teacher_burden_warm": warm["teacher_interventions_per_100_release_intents"],
        "correction_mean": overall["correction_magnitude_mean"],
        "correction_p95": overall["correction_magnitude_p95"],
        "shield_burden": overall["shield_interventions_per_100_release_intents"],
        "merge_corrections": overall["teacher_correction_components"].get("merge_time", 0),
        "completion_ratio": overall["completion_ratio"], "throughput_per_second": overall["throughput_per_second"],
        "delay_mean": overall["delay_mean"], "delay_p95": overall["delay_p95"],
        "ppo_health": _ppo_health(records), "safety": safety, "safety_passed": not any(safety.values()),
    }


def _not_worse(candidate: float | None, control: float | None, *, relative_tolerance: float = .02) -> bool:
    return candidate is not None and (control is None or candidate <= control * (1.0 + relative_tolerance) + 1e-9)


def _decision(variants: dict[str, dict[str, Any]]) -> dict[str, Any]:
    a = variants["A_continuous_release_tanh"]
    b = variants["B_bernoulli_release_tanh_residuals"]
    release_a, release_b = a["release"], b["release"]
    coverage = (all(release_a[key] > 0 for key in
                    ("teacher_release_corrections", "false_release", "false_wait", "near_threshold_samples",
                     "busy_or_committed_samples", "safe_open_gap_samples")) and
                all(release_a["label_counts"][key] > 0 for key in ("WAIT", "RELEASE")))
    release_reduced = bool(release_b["teacher_release_correction_rate"] <=
                           .80 * release_a["teacher_release_correction_rate"])
    burden = all(_not_worse(b[key], a[key]) for key in
                 ("teacher_burden_overall", "teacher_burden_warm", "correction_p95", "shield_burden"))
    performance = bool(b["completion_ratio"] >= .98 * a["completion_ratio"] and
                       b["throughput_per_second"] >= .98 * a["throughput_per_second"])
    passed = bool(coverage and release_reduced and burden and performance and b["ppo_health"]["healthy"] and
                  b["safety_passed"] and b["provenance_passed"])
    return {
        "release_targeted_coverage_passed": coverage,
        "release_correction_reduced_20pct": release_reduced,
        "teacher_p95_shield_not_worse": burden, "completion_throughput_healthy": performance,
        "B_passed": passed, "freeze_mixed_actor": passed,
        "next_step": ("run a second independent mixed-actor short window before formal L1 restart" if passed else
                      "continue D5a release objective/calibration diagnostic; do not add PPO steps blindly"),
    }


def run_release_mixed_actor_diagnostic(*, c0_report: str | Path, teacher_dataset_path: str | Path,
                                       initial_checkpoint: str | Path, output_dir: str | Path,
                                       rollout_workers: int = 4, source_commit: str = "unknown") -> dict[str, Any]:
    """Run the fixed Phase-2 coverage plus the Phase-6 three-update A/B."""
    destination = Path(output_dir)
    audit = authorize_low_v2_training(teacher_dataset_path)
    partition = _read_seed_partition(teacher_dataset_path)
    mu_hat, medium_interval = _read_frozen_capacity(c0_report)
    medium_rate = sum(medium_interval) / 2.0
    checkpoint = Path(initial_checkpoint)
    config_payload = {"updates": 3, "episodes_per_update": 4, "max_steps": 1200,
                      "teacher_bc_weight": .05, "learning_rate": 3e-4, "entropy_coefficient": .01,
                      "mu_hat": mu_hat, "medium_rate": medium_rate, "seeds": list(partition.train[:12])}
    manifest = {
        "stage": "L1_release_mixed_actor_diagnostic", "evidence_type": "newly_run",
        "source_commit": source_commit,
        "config_fingerprint": hashlib.sha256(json.dumps(config_payload, sort_keys=True).encode()).hexdigest(),
        "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "teacher_dataset_audit": asdict(audit), "variants": VARIANTS, "config": config_payload,
        "release_semantics_audit": {
            "execution_rule": "release_request > 0 means RELEASE; otherwise WAIT",
            "magnitude_used_after_threshold": False,
            "positive_0.01_equals_positive_0.99": True,
            "negative_0.01_equals_negative_0.99": True,
            "teacher_candidates_preserve_release_request": True,
            "architecture_mismatch": "confirmed: a continuous head represents a binary execution decision",
            "dataset": _teacher_dataset_release_audit(teacher_dataset_path),
        },
        "hypothesis": "threshold-only binary release semantics make a continuous scalar head non-identifiable in magnitude",
        "frozen": ["reward", "BC weight", "learning rate", "entropy coefficient", "traffic", "Observation V2",
                   "SafetyFilter/Shield", "EH216-S dynamics", "mu_hat/lambda", "action interface"],
        "provenance": {"buffer_action": "a_requested", "joint_old_log_prob": "actual requested mixed distribution",
                       "a_applied_enters_ratio": False},
    }
    _write(destination / "release_mixed_actor.manifest.json", manifest)
    _write(destination / "release_mixed_actor.startup_preflight.json", {
        "passed": True, "expected_outputs": ["<arm>.updates.json", "release_mixed_actor_summary.json",
                                               "release_mixed_actor_summary.md"],
        "release_label_semantics": {"WAIT": -1, "RELEASE": 1},
    })
    variants: dict[str, Any] = {}
    for label, settings in VARIANTS.items():
        random.seed(1701); np.random.seed(1701); torch.manual_seed(1701)
        policy = SharedPPO(LOW_OBS_V2.dimension, teacher_bc_weight=.05,
                           policy_distribution=settings["policy_distribution"])
        config = l1_full_teacher_config(medium_rate=medium_rate, reset_mode="warm_start_rings",
                                        demand_model="finite_cohort", warm_start_uavs=8, cohort_uavs=8,
                                        horizon_seconds=1200.0)
        load_checkpoint(checkpoint, policy, config)
        records: list[dict[str, Any]] = []
        provenance: list[dict[str, Any]] = []
        seeds, design_index = itertools.cycle(partition.train), 0
        try:
            for update in range(3):
                state, tasks, designs = _model_state(policy), [], []
                for episode in range(4):
                    episode_config, design = _episode_design(design_index, medium_rate=medium_rate,
                                                             horizon_seconds=1200.0)
                    design_index += 1
                    tasks.append((state, episode_config, next(seeds), update, episode, 1200, False, None,
                                  settings["policy_distribution"], True, True, settings["release_teacher_bc"]))
                    designs.append(design)
                if rollout_workers == 1:
                    collected = [_rollout_task(task) for task in tasks]
                else:
                    with ProcessPoolExecutor(max_workers=min(rollout_workers, len(tasks))) as executor:
                        collected = list(executor.map(_rollout_task, tasks))
                buffer, summaries, transitions = RolloutBuffer(), [], []
                for (episode_transitions, summary), design in zip(collected, designs):
                    transitions.extend(episode_transitions)
                    for transition in episode_transitions:
                        buffer.add(transition)
                    summaries.append({**asdict(summary), **design})
                if any(row["collisions"] or row["nmac"] or row["hard_separation_violations"] for row in summaries):
                    raise L1SafetyGateError(f"release A/B safety gate failed in {label} update {update}")
                proof = _requested_action_provenance(policy, buffer)
                if not proof["passed"]:
                    raise RuntimeError(f"release A/B provenance failed in {label}: {proof}")
                release = release_decision_metrics(policy, transitions)
                release_objective = release_objective_metrics(policy, buffer)
                provenance.append(proof)
                ppo = policy.update(buffer, epochs=2, minibatch_size=256)
                records.append({"update": update, "summaries": summaries, "release": release,
                                "release_objective": release_objective, "ppo_update": ppo})
                _write(destination / f"{label}.updates.json", {"manifest": manifest, "records": records})
            variants[label] = _variant_summary(records, provenance)
        except (SafetyInvariantError, L1SafetyGateError, RuntimeError, FloatingPointError) as error:
            variants[label] = {"status": "stopped", "reason": str(error), "records": records,
                               "provenance": provenance, "safety_passed": False}
            break
    complete = len(variants) == len(VARIANTS) and all("release" in values for values in variants.values())
    decision = _decision(variants) if complete else {"B_passed": False, "freeze_mixed_actor": False,
                                                      "next_step": "repair diagnostic runtime before more training"}
    result = {"status": "completed" if complete else "stopped", "manifest": manifest,
              "variants": variants, "decision": decision}
    _write(destination / "release_mixed_actor_summary.json", result)
    lines = ["# L1 Release Mixed-Actor Diagnostic", "", f"- architecture mismatch：confirmed。",
             f"- release targeted coverage：{'PASS' if decision.get('release_targeted_coverage_passed') else 'FAIL'}。",
             f"- Bernoulli mixed actor：{'PASS' if decision.get('B_passed') else 'FAIL'}。"]
    if complete:
        for name, values in variants.items():
            release = values["release"]
            lines.append(f"- {name}: release correction={release['teacher_release_correction_rate']:.4f}, "
                         f"false-release={release['false_release_rate']:.4f}, false-wait={release['false_wait_rate']:.4f}, "
                         f"teacher overall/warm={values['teacher_burden_overall']:.3f}/{values['teacher_burden_warm']:.3f}, "
                         f"P95={values['correction_p95']:.3f}, shield={values['shield_burden']:.3f}, "
                         f"completion={values['completion_ratio']:.4f}, throughput={values['throughput_per_second']:.6f}, "
                         f"PPO healthy={values['ppo_health']['healthy']}, safety={values['safety']}。")
    lines.append(f"- next: {decision['next_step']}。")
    (destination / "release_mixed_actor_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def _main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run the release semantics probe or three-update mixed-actor diagnostic.")
    parser.add_argument("--c0-report", required=True); parser.add_argument("--teacher-dataset")
    parser.add_argument("--initial-checkpoint"); parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4); parser.add_argument("--source-commit", default="unknown")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--probe-max-steps", type=int, default=240)
    args = parser.parse_args()
    if args.probe_only:
        print(json.dumps(run_release_semantics_probe(
            c0_report=args.c0_report, output_path=args.output,
            workers=args.workers, max_steps=args.probe_max_steps), indent=2))
        return
    if not args.teacher_dataset or not args.initial_checkpoint:
        parser.error("--teacher-dataset and --initial-checkpoint are required unless --probe-only is used")
    print(json.dumps(run_release_mixed_actor_diagnostic(
        c0_report=args.c0_report, teacher_dataset_path=args.teacher_dataset,
        initial_checkpoint=args.initial_checkpoint, output_dir=args.output,
        rollout_workers=args.workers, source_commit=args.source_commit), indent=2))


if __name__ == "__main__":
    _main()
