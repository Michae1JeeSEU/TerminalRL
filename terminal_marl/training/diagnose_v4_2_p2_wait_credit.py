"""Fresh diagnostic for P2 V4 pre-attempt WAIT and team-credit behavior.

The runner never updates a policy.  A teacher-label-only sidecar audits the
student-requested gate while student actions remain authoritative.  Fresh
stochastic rollouts expose team GAE, gate calibration, confusion cells, and
WAIT streaks without reopening the consumed 755001--755006 heldout cohort.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from terminal_marl.algo.student_ppo_v4_2 import StudentPPOV42, StudentStepV42, team_gae
from terminal_marl.core.observations import LOW_OBS_V2_1
from terminal_marl.training.evaluate_v4_2_p2_v4 import (
    BASELINE_SHA,
    BASELINE_STAGE,
    CANDIDATE_SHA,
    CANDIDATE_STAGE,
    TRAINING_PROTOCOL_SHA,
    load_evaluation_policy,
)
from terminal_marl.training.run_v4_2_ppo_p2_resume_v4 import p2_config, p2_source_hashes
from terminal_marl.training.run_v4_2_v2_1_formal_i0_r3 import _write
from terminal_marl.training.run_v4_2_v2_1_formal_i0_r4 import historical_registry
from terminal_marl.training.run_v4_2_v2_1_i1_v2 import _sha, reserve_once
from terminal_marl.training.student_rollout_v4_2 import rollout


PROTOCOL = Path("terminal_marl/configs/low_v4_2/ppo_p2_v4_wait_diagnostic.yaml")
PROTOCOL_SHA = "f64b0fcea24ad72a57ded6c31c5b41bdc6247a33fbdf631ba9169fa3c921174a"
HELDOUT_MANIFEST_SHA = "167d22cb29d8ef6d08275cb0ff4cbe270c0d9005611a176b85d81fd64730f104"
FORMAL_SEEDS = (771001, 771002, 771003, 771004, 771005, 771006)
FORMAL_SCENARIOS = ("flow6", "flow8", "flow12")
CONTROLLERS = ("candidate", "baseline")
SMOKE_SEED = 771000
SMOKE_STEPS = 160


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def protocol() -> dict[str, Any]:
    _require(_sha(PROTOCOL) == PROTOCOL_SHA, "WAIT diagnostic protocol changed after preregistration")
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), "WAIT diagnostic protocol must be a mapping")
    return value


def diagnostic_config(total_requests: int):
    """Keep P2 physics/reward while enabling a non-authoritative label sidecar."""
    return replace(
        p2_config(total_requests),
        experiment_mode="fixed_teacher_heavy",
        scheduler_mode="execution",
        teacher_enabled=True,
        teacher_mode="teacher_label_only",
        teacher_release_label_enabled=False,
    )


def _interface(phase: str) -> str | None:
    return {"ring_outer": "o2m", "ring_middle": "m2i", "ring_inner": "i2f"}.get(phase)


def _stats(values: list[float] | np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"n": 0, "mean": None, "p10": None, "median": None, "p90": None, "max": None}
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
    }


def _conditional_advantage(rows: list[Mapping[str, Any]]) -> dict[str, float | int | None]:
    values = np.asarray([float(row["advantage"]) for row in rows], dtype=np.float64)
    result = _stats(values)
    result["positive_rate"] = None if not values.size else float((values > 0.0).mean())
    return result


def _streaks(rows: list[Mapping[str, Any]], *, false_wait_only: bool) -> list[int]:
    """Return consecutive WAIT streak lengths within each UAV/interface stream."""
    grouped: dict[tuple[int, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row.get("seed", -1)), int(row["uav_id"]), str(row["interface"]))].append(row)
    lengths: list[int] = []
    for stream in grouped.values():
        current = 0
        previous_step: int | None = None
        for row in sorted(stream, key=lambda item: int(item["step"])):
            step = int(row["step"])
            qualifies = int(row["requested_gate"]) == 0
            if false_wait_only:
                qualifies = qualifies and int(row["teacher_gate"]) == 1
            if previous_step is not None and step != previous_step + 1:
                if current:
                    lengths.append(current)
                current = 0
            if qualifies:
                current += 1
            elif current:
                lengths.append(current)
                current = 0
            previous_step = step
        if current:
            lengths.append(current)
    return lengths


def decision_diagnostics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Pure aggregation used by tests and the runtime diagnostic."""
    rows = [dict(row) for row in rows]
    requested_wait = [row for row in rows if int(row["requested_gate"]) == 0]
    requested_release = [row for row in rows if int(row["requested_gate"]) == 1]
    teacher_wait = [row for row in rows if int(row["teacher_gate"]) == 0]
    teacher_release = [row for row in rows if int(row["teacher_gate"]) == 1]
    cells = {
        "true_wait": [row for row in rows if int(row["requested_gate"]) == 0 and int(row["teacher_gate"]) == 0],
        "false_release": [row for row in rows if int(row["requested_gate"]) == 1 and int(row["teacher_gate"]) == 0],
        "false_wait": [row for row in rows if int(row["requested_gate"]) == 0 and int(row["teacher_gate"]) == 1],
        "true_release": [row for row in rows if int(row["requested_gate"]) == 1 and int(row["teacher_gate"]) == 1],
    }

    def context(group: list[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "samples": len(group),
            "wait_feature": _stats([float(row["wait_feature"]) for row in group]),
            "headway_slack": _stats([float(row["headway_slack"]) for row in group]),
            "retry_remaining": _stats([float(row["retry_remaining"]) for row in group]),
            "release_probability": _stats([float(row["release_probability"]) for row in group]),
            "release_eligible_rate": None if not group else float(np.mean([bool(row["release_eligible"]) for row in group])),
            "predicted_conflict_rate": None if not group else float(np.mean([bool(row["predicted_conflict"]) for row in group])),
        }

    probabilities = np.asarray([float(row["release_probability"]) for row in rows], dtype=np.float64)
    labels = np.asarray([int(row["teacher_gate"]) for row in rows], dtype=np.float64)
    stable = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    interfaces = {}
    for name in ("o2m", "m2i", "i2f"):
        group = [row for row in rows if row["interface"] == name]
        interfaces[name] = {
            "samples": len(group),
            "requested_WAIT": sum(int(row["requested_gate"]) == 0 for row in group),
            "requested_RELEASE": sum(int(row["requested_gate"]) == 1 for row in group),
            "teacher_WAIT": sum(int(row["teacher_gate"]) == 0 for row in group),
            "teacher_RELEASE": sum(int(row["teacher_gate"]) == 1 for row in group),
            "false_WAIT": sum(int(row["requested_gate"]) == 0 and int(row["teacher_gate"]) == 1 for row in group),
        }
    return {
        "samples": len(rows),
        "requested": {"WAIT": len(requested_wait), "RELEASE": len(requested_release)},
        "teacher": {"WAIT": len(teacher_wait), "RELEASE": len(teacher_release)},
        "false_wait_rate": None if not teacher_release else len(cells["false_wait"]) / len(teacher_release),
        "false_release_rate": None if not teacher_wait else len(cells["false_release"]) / len(teacher_wait),
        "calibration": {
            "brier": None if not rows else float(np.mean((probabilities - labels) ** 2)),
            "binary_nll": None if not rows else float(np.mean(-(labels * np.log(stable) + (1.0 - labels) * np.log(1.0 - stable)))),
            "release_probability": _stats(probabilities),
        },
        "advantage_by_requested_gate": {
            "WAIT": _conditional_advantage(requested_wait),
            "RELEASE": _conditional_advantage(requested_release),
        },
        "advantage_by_teacher_gate": {
            "WAIT": _conditional_advantage(teacher_wait),
            "RELEASE": _conditional_advantage(teacher_release),
        },
        "advantage_by_confusion_cell": {name: _conditional_advantage(group) for name, group in cells.items()},
        "context_by_confusion_cell": {name: context(group) for name, group in cells.items()},
        "wait_streaks": _stats(_streaks(rows, false_wait_only=False)),
        "false_wait_streaks": _stats(_streaks(rows, false_wait_only=True)),
        "interfaces": interfaces,
    }


def _decorate_rows(
    policy: StudentPPOV42, steps: list[StudentStepV42], label_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    advantages, _ = team_gae(steps, gamma=0.99, gae_lambda=0.95)
    actor = [row for row in label_rows if bool(row.get("actor_mask")) and _interface(str(row.get("phase"))) is not None]
    if not actor:
        return []
    observations = np.asarray([row["observation_v2_1"] for row in actor], dtype=np.float32)
    with torch.no_grad():
        gate_dist, _ = policy.model.distributions(torch.as_tensor(observations, device=policy.device))
        probabilities = gate_dist.probs[:, 1].cpu().numpy()
    slices = LOW_OBS_V2_1.slices
    result = []
    for row, probability, observation in zip(actor, probabilities, observations, strict=True):
        step = int(row["step"])
        conflict = slices["nearest_predicted_conflict"]
        result.append({
            "seed": int(row["seed"]),
            "step": step,
            "uav_id": int(row["uav_id"]),
            "phase": str(row["phase"]),
            "interface": _interface(str(row["phase"])),
            "requested_gate": int(row["requested_gate"]),
            "teacher_gate": int(row["teacher_gate"]),
            "release_probability": float(probability),
            "advantage": float(advantages[step]),
            "release_eligible": bool(observation[slices["release_eligible"]][0] > 0.5),
            "wait_feature": float(observation[slices["wait"]][0]),
            "headway_slack": float(observation[slices["interface_headway_slack"]][0]),
            "retry_remaining": float(observation[slices["safety_retry_remaining_norm"]][0]),
            "predicted_conflict": bool(observation[conflict.start] > 0.5),
        })
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def diagnose_episode(
    policy: StudentPPOV42, config: Any, seed: int, output: Path, *, max_steps: int
) -> dict[str, Any]:
    steps, summary = rollout(
        policy,
        config,
        seed=seed,
        max_steps=max_steps,
        output=output,
        deterministic=False,
        collect_labels=True,
    )
    labels = _read_jsonl(output / "labels.jsonl")
    rows = _decorate_rows(policy, steps, labels)
    _write(output / "decision_rows.json", rows)
    metrics = decision_diagnostics(rows)
    safety = dict(summary["safety"])
    requested = metrics["requested"]
    teacher = metrics["teacher"]
    gates = {
        "no_policy_update": True,
        "student_authoritative_execution": True,
        "teacher_label_only": bool(labels) and all(
            "teacher_gate" in row and "requested_gate" in row for row in labels
        ),
        "safety_all_zero": not any(safety.values()),
        "shield_zero": int(summary["counts"]["shield"]) == 0,
        "terminal_and_all_requests_complete": bool(
            summary["terminal"] and int(summary["completed"]) == int(config.total_requested_uavs)
        ),
        "requested_WAIT_and_RELEASE_observed": requested["WAIT"] > 0 and requested["RELEASE"] > 0,
        "teacher_WAIT_and_RELEASE_observed": teacher["WAIT"] > 0 and teacher["RELEASE"] > 0,
        "all_three_interfaces_observed": all(metrics["interfaces"][name]["samples"] > 0 for name in ("o2m", "m2i", "i2f")),
    }
    return {
        "seed": seed,
        "steps": len(steps),
        "seconds": float(summary["seconds"]),
        "completed": int(summary["completed"]),
        "requested": int(config.total_requested_uavs),
        "terminal": bool(summary["terminal"]),
        "safety": safety,
        "shield": int(summary["counts"]["shield"]),
        "actor_decisions": int(summary["counts"]["actor_decisions"]),
        "label_rows": len(labels),
        "diagnostic_rows": len(rows),
        "metrics": metrics,
        "gates": gates,
        "action_tape": str(summary["action_tape"]),
        "training_called": False,
        "teacher_enabled_for_labels_only": True,
        "teacher_replaced_behavior": False,
    }


def aggregate(records: list[Mapping[str, Any]], controller: str) -> dict[str, Any]:
    selected = [record for record in records if record.get("controller") == controller]
    rows = [row for record in selected for row in record["rows"]]
    metrics = decision_diagnostics(rows)
    return {
        "controller": controller,
        "episodes": len(selected),
        "completed": sum(int(record["result"]["completed"]) for record in selected),
        "requested_uavs": sum(int(record["result"]["requested"]) for record in selected),
        "safety_all_zero": all(not any(record["result"]["safety"].values()) for record in selected),
        "shield_zero": all(int(record["result"]["shield"]) == 0 for record in selected),
        "all_episode_gates": bool(selected) and all(all(record["result"]["gates"].values()) for record in selected),
        "metrics": metrics,
    }


def _formal_complete(records: list[Mapping[str, Any]]) -> bool:
    if len(records) != 12:
        return False
    for controller in CONTROLLERS:
        selected = [record for record in records if record.get("controller") == controller]
        if len(selected) != 6:
            return False
        ordered = sorted(selected, key=lambda item: int(item["result"]["seed"]))
        if [int(item["result"]["seed"]) for item in ordered] != list(FORMAL_SEEDS):
            return False
        if [str(item["scenario"]) for item in ordered] != list(FORMAL_SCENARIOS) * 2:
            return False
    return True


def run(
    output: Path, checkpoint: Path, baseline: Path, heldout_manifest: Path,
    history_root: Path, *, smoke: bool = False,
) -> dict[str, Any]:
    p = protocol()
    _require(p["bindings"]["candidate_checkpoint_sha256"] == CANDIDATE_SHA, "candidate binding changed")
    _require(p["bindings"]["baseline_checkpoint_sha256"] == BASELINE_SHA, "baseline binding changed")
    _require(p["bindings"]["heldout_manifest_sha256"] == HELDOUT_MANIFEST_SHA, "heldout binding changed")
    _require(p["formal"]["seeds"] == list(FORMAL_SEEDS), "formal seeds changed")
    _require(heldout_manifest.is_file() and _sha(heldout_manifest) == HELDOUT_MANIFEST_SHA,
             "heldout result manifest binding mismatch")
    candidate, _ = load_evaluation_policy(
        checkpoint,
        expected_sha=CANDIDATE_SHA,
        expected_stage=CANDIDATE_STAGE,
        expected_formal=True,
        expected_protocol_sha=TRAINING_PROTOCOL_SHA,
        expected_source_hashes=p2_source_hashes(),
    )
    baseline_policy, _ = load_evaluation_policy(
        baseline,
        expected_sha=BASELINE_SHA,
        expected_stage=BASELINE_STAGE,
        expected_formal=True,
    )
    output.mkdir(parents=True, exist_ok=False)
    seeds = [SMOKE_SEED] if smoke else list(FORMAL_SEEDS)
    scenarios = ["flow6"] if smoke else list(FORMAL_SCENARIOS)
    if not smoke:
        history = historical_registry(history_root, output)
        _require(not (set(seeds) & history["seeds"]), "WAIT diagnostic seeds already consumed")
        reserve_once(
            checkpoint.parent / "p2_v4_wait_diagnostic_consumed.json",
            {"protocol_sha256": PROTOCOL_SHA, "seeds": seeds, "output": str(output)},
        )
    _write(output / "pre_reset_registry.json", {
        "protocol_sha256": PROTOCOL_SHA,
        "seeds": seeds,
        "scenarios": scenarios,
        "controllers": list(CONTROLLERS),
        "smoke": smoke,
        "training_called": False,
    })
    records: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        scenario = scenarios[index % len(scenarios)]
        n = 6 if scenario == "flow6" else 8 if scenario == "flow8" else 12
        config = diagnostic_config(n)
        max_steps = SMOKE_STEPS if smoke else int(config.max_simulation_seconds / config.dt)
        for controller, policy in (("candidate", candidate), ("baseline", baseline_policy)):
            folder = output / f"{controller}_{seed}"
            result = diagnose_episode(policy, config, seed, folder, max_steps=max_steps)
            result["scenario"] = scenario
            compact_rows = json.loads((folder / "decision_rows.json").read_text(encoding="utf-8"))
            records.append({"controller": controller, "scenario": scenario, "result": result, "rows": compact_rows})
            _write(output / "progress.json", {"completed": len(records), "total": 2 * len(seeds), "records": [
                {"controller": item["controller"], "scenario": item["scenario"], "result": item["result"]}
                for item in records
            ]})
    aggregates = {controller: aggregate(records, controller) for controller in CONTROLLERS}
    complete = (not smoke) and _formal_complete(records)
    result = {
        "stage": "V4_2_P2_WAIT_CREDIT_DIAGNOSTIC",
        "smoke": smoke,
        "DIAGNOSTIC_COMPLETE": bool(complete and all(value["all_episode_gates"] for value in aggregates.values())),
        "PPO_ALLOWED": False,
        "curriculum_promotion_allowed": False,
        "records": [{"controller": item["controller"], "scenario": item["scenario"], "result": item["result"]} for item in records],
        "aggregates": aggregates,
        "protocol_sha256": PROTOCOL_SHA,
        "source_hashes": {
            **p2_source_hashes(),
            "diagnostic_runner": _sha(Path(__file__)),
            "diagnostic_protocol": _sha(PROTOCOL),
        },
        "training_called": False,
        "teacher_label_only": True,
        "teacher_replaced_behavior": False,
    }
    _write(output / "manifest.json", result)
    print(json.dumps({"DIAGNOSTIC_COMPLETE": result["DIAGNOSTIC_COMPLETE"], "smoke": smoke, "records": len(records)}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("output", "checkpoint", "baseline", "heldout-manifest", "history-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    torch.set_num_threads(1)
    run(**vars(parser.parse_args()))
