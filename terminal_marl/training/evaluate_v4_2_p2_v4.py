"""Once-only held-out evaluation for the formal P2 V4 checkpoint.

The evaluator is intentionally independent of the P2 training runner.  It
loads both learned policies without an optimizer, evaluates them deterministically
alongside the nominal rule controller, and writes only descriptive evidence.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from terminal_marl.algo.student_ppo_v4_2 import StudentPPOV42
from terminal_marl.core.checkpoints import load_checkpoint
from terminal_marl.core.safety_trace import SafetyTraceRecorder
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training.run_v4_2_curriculum_pilot import episode as rule_episode
from terminal_marl.training.run_v4_2_i2_dagger_preflight import _policy_tape_logprob_exact
from terminal_marl.training.run_v4_2_ppo_p2_resume_v4 import p2_config, p2_source_hashes
from terminal_marl.training.run_v4_2_v2_1_formal_i0_r3 import _write
from terminal_marl.training.run_v4_2_v2_1_formal_i0_r4 import historical_registry
from terminal_marl.training.run_v4_2_v2_1_i1_v2 import _sha, reserve_once
from terminal_marl.training.student_rollout_v4_2 import rollout


PROTOCOL = Path("terminal_marl/configs/low_v4_2/ppo_p2_v4_evaluation.yaml")
PROTOCOL_SHA = "b3666bfee8a6c21002cab9088437f4d061b81128a46bab3df02ab83ba235ac81"
TRAINING_PROTOCOL_SHA = "01c92c44aecc50035e76b89a4e8d330fac5cfe5124b79d593c92a69260154ebf"
TRAINING_MANIFEST_SHA = "73502ed0f956e517c47dd14ea6bfcc411cb927eba43c4a409fdee119fa7c82f0"
CANDIDATE_SHA = "5e6811b8349d875fb007943476aec62d93550a74ffaf4ae46614c5c188ab004f"
BASELINE_SHA = "6420df9a8562135756819459c7682ec897d47acd64e841b75a21c697298ea14f"
CANDIDATE_STAGE = "V4_2_P2_FORMAL_TRAIN"
BASELINE_STAGE = "V4_2_V2_1_PPO"
FORMAL_SEEDS = (755001, 755002, 755003, 755004, 755005, 755006)
FORMAL_SCENARIOS = ("flow6", "flow8", "flow12")
CONTROLLERS = ("candidate", "baseline", "rule")
SMOKE_SEED = 770000
SMOKE_SCENARIO = "flow6"
SMOKE_STEPS = 160
SHIELD_ACTOR_FRACTION_MAX = 0.05


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def protocol() -> dict[str, Any]:
    """Load the evaluation YAML only after its preregistered byte hash check."""
    if _sha(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("P2 V4 evaluation protocol changed after preregistration")
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("evaluation protocol must be a mapping")
    return value


def evaluation_source_hashes() -> dict[str, str]:
    """Bind the V4 training source snapshot and this evaluator protocol."""
    return {
        **p2_source_hashes(),
        "evaluation_runner": _sha(Path(__file__)),
        "evaluation_protocol": _sha(PROTOCOL),
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_training_manifest_payload(
    report: Mapping[str, Any], *, candidate_name: str = "checkpoint_8.pt"
) -> dict[str, Any]:
    """Validate the frozen semantic shape of the eight-episode V4 train report."""
    _require(report.get("stage") == "train", "training manifest stage mismatch")
    _require(report.get("PASS") is True, "training manifest did not pass")
    _require(report.get("protocol_sha256") == TRAINING_PROTOCOL_SHA, "training protocol mismatch")
    _require(report.get("formal") is True, "training manifest is not formal")
    records = report.get("records")
    _require(isinstance(records, list) and len(records) == 8, "training manifest must contain 8 records")
    _require(report.get("source_hashes") == p2_source_hashes(), "training source hashes mismatch")
    for index, record in enumerate(records):
        _require(isinstance(record, Mapping), f"training record {index} is not an object")
        gates = record.get("gates")
        _require(isinstance(gates, Mapping) and bool(gates) and all(gates.values()),
                 f"training record {index} has a failed gate")
    final_checkpoint = records[-1].get("checkpoint")
    _require(isinstance(final_checkpoint, str) and Path(final_checkpoint).name == candidate_name,
             "training manifest final checkpoint filename mismatch")
    return dict(report)


def validate_training_manifest(path: Path, *, candidate_name: str = "checkpoint_8.pt") -> dict[str, Any]:
    """Check the exact immutable manifest bytes and then its semantic payload."""
    _require(_sha(path) == TRAINING_MANIFEST_SHA, "training manifest SHA mismatch")
    return validate_training_manifest_payload(_read_json(path), candidate_name=candidate_name)


def validate_checkpoint_payload(
    payload: Mapping[str, Any], *, expected_stage: str, expected_formal: bool,
    expected_protocol_sha: str | None = None,
    expected_source_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate lineage metadata after ``load_checkpoint`` validates the schema."""
    _require(payload.get("stage") == expected_stage, "checkpoint stage mismatch")
    _require(payload.get("formal") is expected_formal, "checkpoint formal flag mismatch")
    if expected_protocol_sha is not None:
        _require(payload.get("protocol_sha256") == expected_protocol_sha, "checkpoint protocol mismatch")
    if expected_source_hashes is not None:
        _require(payload.get("source_hashes") == dict(expected_source_hashes), "checkpoint source hashes mismatch")
    _require(isinstance(payload.get("model"), Mapping) and bool(payload["model"]),
             "checkpoint model payload missing")
    return dict(payload)


def load_evaluation_policy(
    path: Path, *, expected_sha: str, expected_stage: str, expected_formal: bool,
    expected_protocol_sha: str | None = None,
    expected_source_hashes: Mapping[str, str] | None = None,
) -> tuple[StudentPPOV42, dict[str, Any]]:
    """Load one policy with a fresh StudentPPOV42 and no optimizer state."""
    _require(path.is_file(), f"checkpoint not found: {path}")
    _require(path.name == "checkpoint_8.pt" if expected_stage == CANDIDATE_STAGE else True,
             "candidate checkpoint filename mismatch")
    _require(_sha(path) == expected_sha, "checkpoint SHA mismatch")
    policy = StudentPPOV42(88, learning_rate=0.00003)
    payload = load_checkpoint(path, policy, p2_config(6), load_optimizer=False)
    validate_checkpoint_payload(
        payload,
        expected_stage=expected_stage,
        expected_formal=expected_formal,
        expected_protocol_sha=expected_protocol_sha,
        expected_source_hashes=expected_source_hashes,
    )
    policy.model.eval()
    return policy, payload


def _geometry_replay(
    config: Any, tape_path: str | Path
) -> tuple[dict[str, Any], dict[str, float], tuple[float | None, float | None]]:
    """Replay a tape while collecting geometry and lifecycle performance metrics."""
    env = TerminalParallelEnv(config)
    geometry: dict[str, float] = {
        "samples": 0,
        "reverse_samples": 0,
        "max_inner_excursion_m": 0.0,
    }
    peaks = {"active": 0, "queue": 0}

    def observe(world: TerminalParallelEnv) -> None:
        peaks["active"] = max(peaks["active"], len(world.active))
        peaks["queue"] = max(peaks["queue"], len(world.admission.queue.items))
        for uav in world.active.values():
            plan = uav.committed_plan
            if plan is None or plan.mode != "ring_transfer":
                continue
            xy, velocity = uav.position[:2], uav.velocity[:2]
            radius = float(np.linalg.norm(xy))
            geometry["samples"] += 1
            cross = (xy[0] * velocity[1] - xy[1] * velocity[0]) / max(radius, 1e-9)
            geometry["reverse_samples"] += int(cross < -0.001)
            endpoint_radius = float(np.linalg.norm(plan.points[-1].position[:2]))
            geometry["max_inner_excursion_m"] = max(
                geometry["max_inner_excursion_m"], endpoint_radius - radius
            )

    env.set_step_observer(observe)
    replay = SafetyTraceRecorder.replay_action_tape(env, tape_path)
    completed = list(env.completed_uavs)
    wait_seconds, request_to_touchdown = _wait_metrics(completed)
    return (
        replay,
        geometry | {"active_peak": peaks["active"], "queue_peak": peaks["queue"]},
        (wait_seconds, request_to_touchdown),
    )


def _wait_metrics(completed: list[Any]) -> tuple[float | None, float | None]:
    if not completed:
        return None, None
    waits = [float(uav.internal_wait_time + uav.external_wait_time) for uav in completed]
    request_to_touchdown = [float(uav.touchdown_time - uav.request_time) for uav in completed]
    return float(np.mean(waits)), float(np.mean(request_to_touchdown))


def _base_gates(result: Mapping[str, Any], *, require_likelihood: bool = False) -> dict[str, bool]:
    safety = result.get("safety", {})
    geometry = result.get("geometry", {})
    replay = bool(result.get("replay_exact"))
    if require_likelihood:
        replay = replay and bool(result.get("student_likelihood_exact"))
    return {
        "complete": bool(result.get("terminal") and result.get("completed") == result.get("requests")),
        "safety": not any(bool(value) for value in safety.values()),
        "replay": replay,
        "geometry": bool(
            geometry.get("samples", 0) > 0
            and geometry.get("reverse_samples", 0) == 0
            and geometry.get("max_inner_excursion_m", float("inf")) <= 0.05
        ),
    }


def evaluate(policy: StudentPPOV42, config: Any, seed: int, output: Path, *, smoke: bool) -> dict[str, Any]:
    """Run one deterministic student-only episode and collect evidence metrics."""
    _, summary = rollout(
        policy,
        config,
        seed=seed,
        max_steps=SMOKE_STEPS if smoke else int(config.max_simulation_seconds / config.dt),
        output=output,
        deterministic=True,
        collect_labels=False,
    )
    replay, geometry, lifecycle_metrics = _geometry_replay(config, summary["action_tape"])
    likelihood = _policy_tape_logprob_exact(policy, config, summary["action_tape"])
    wait_seconds, request_to_touchdown_seconds = lifecycle_metrics
    actor_decisions = int(summary["counts"]["actor_decisions"])
    shield_actor_fraction = summary["counts"]["shield_actor"] / max(1, actor_decisions)
    result = {
        "seed": seed,
        "requests": int(config.total_requested_uavs),
        "completed": int(summary["completed"]),
        "terminal": bool(summary["terminal"]),
        "seconds": float(summary["seconds"]),
        "safety": dict(summary["safety"]),
        "replay_exact": bool(replay["exact"]),
        "geometry": {
            "samples": int(geometry["samples"]),
            "reverse_samples": int(geometry["reverse_samples"]),
            "max_inner_excursion_m": float(geometry["max_inner_excursion_m"]),
        },
        "active_peak": int(geometry["active_peak"]),
        "queue_peak": int(geometry["queue_peak"]),
        "weighted_wait_seconds": None if wait_seconds is None else float(wait_seconds),
        "request_to_touchdown_seconds": (
            None if request_to_touchdown_seconds is None else float(request_to_touchdown_seconds)
        ),
        "mean_wait_seconds": None if wait_seconds is None else float(wait_seconds),
        "mean_request_to_touchdown_seconds": (
            None if request_to_touchdown_seconds is None else float(request_to_touchdown_seconds)
        ),
        "student_likelihood_exact": bool(likelihood),
        "requested_logprob_exact": bool(likelihood),
        "shield": int(summary["counts"]["shield"]),
        "shield_actor": int(summary["counts"]["shield_actor"]),
        "actor_decisions": actor_decisions,
        "shield_actor_fraction": float(shield_actor_fraction),
        "deterministic": True,
        "teacher_enabled": False,
        "label_rows": int(summary.get("label_rows", 0)),
        "training_called": False,
        "summary": summary,
    }
    result["gates"] = {
        **_base_gates(result, require_likelihood=True),
        "student_likelihood": bool(result["student_likelihood_exact"]),
        "shield_actor_fraction": result["shield_actor_fraction"] <= SHIELD_ACTOR_FRACTION_MAX,
        "student_only": result["label_rows"] == 0 and not result["teacher_enabled"],
        "no_updates": result["training_called"] is False,
    }
    _write(output / "evaluation.json", result)
    return result


def evaluate_rule(config: Any, seed: int, output: Path, *, smoke: bool) -> dict[str, Any]:
    """Run the nominal no-policy rule controller under the same flow config."""
    result = dict(rule_episode(config, seed, output, max_steps=SMOKE_STEPS if smoke else None))
    result["replay_exact"] = bool(result.get("replay_exact"))
    result["student_likelihood_exact"] = None
    result["requested_logprob_exact"] = None
    result["weighted_wait_seconds"] = result.get("mean_wait_seconds")
    result["request_to_touchdown_seconds"] = result.get("mean_request_to_touchdown_seconds")
    result["deterministic"] = True
    result["teacher_enabled"] = False
    result["training_called"] = False
    result["gates"] = {
        **_base_gates(result),
        "rule_shield": int(result.get("shield", 0)) == 0,
        "no_updates": True,
        "student_only": True,
    }
    _write(output / "evaluation.json", result)
    return result


def _weighted_mean(rows: list[Mapping[str, Any]], key: str) -> float | None:
    completed = sum(int(row.get("completed", 0)) for row in rows)
    values = [(float(row[key]), int(row.get("completed", 0))) for row in rows if row.get(key) is not None]
    if not values:
        return None
    weight = sum(count for _, count in values)
    return float(sum(value * count for value, count in values) / max(1, weight))


def aggregate_records(records: list[Mapping[str, Any]], controller: str) -> dict[str, Any]:
    """Aggregate a controller without allowing incomplete episodes to pass gates."""
    rows = [dict(row["result"]) for row in records if row.get("controller") == controller]
    requests = sum(int(row.get("requests", 0)) for row in rows)
    completed = sum(int(row.get("completed", 0)) for row in rows)
    mean_seconds = float(np.mean([float(row["seconds"]) for row in rows])) if rows else None
    all_gates = bool(rows) and all(
        isinstance(row.get("gates"), Mapping) and bool(row["gates"]) and all(row["gates"].values())
        for row in rows
    )
    weighted_wait = _weighted_mean(rows, "weighted_wait_seconds")
    request_to_touchdown = _weighted_mean(rows, "request_to_touchdown_seconds")
    safety_keys = sorted({
        key
        for row in rows
        for key in (row.get("safety") or {})
    })
    safety_counts = {
        key: sum(int(bool((row.get("safety") or {}).get(key, 0))) for row in rows)
        for key in safety_keys
    }
    shield = sum(int(row.get("shield", 0)) for row in rows)
    shield_actor = sum(int(row.get("shield_actor", 0)) for row in rows)
    actor_decisions = sum(int(row.get("actor_decisions", 0)) for row in rows)
    result = {
        "controller": controller,
        "records": len(rows),
        "requests": requests,
        "completed": completed,
        "completion_fraction": completed / max(1, requests),
        "mean_seconds": mean_seconds,
        "seconds": mean_seconds,
        "weighted_wait_seconds": weighted_wait,
        "mean_wait_seconds": weighted_wait,
        "request_to_touchdown_seconds": request_to_touchdown,
        "mean_request_to_touchdown_seconds": request_to_touchdown,
        "all_gates": all_gates,
        "safety": safety_counts,
        "safety_all_zero": not any(safety_counts.values()),
        "shield": shield,
        "shield_actor": shield_actor,
        "actor_decisions": actor_decisions,
        "shield_actor_fraction": shield_actor / max(1, actor_decisions),
    }
    return result


def _delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def pair_deltas(records: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return candidate-minus-baseline/rule deltas for each seed/scenario."""
    grouped: dict[tuple[int, str], dict[str, Mapping[str, Any]]] = {}
    for row in records:
        result = row.get("result", {})
        key = (int(result["seed"]), str(row.get("scenario", result.get("scenario", ""))))
        grouped.setdefault(key, {})[str(row["controller"])] = result
    deltas = []
    for (seed, scenario), rows in sorted(grouped.items()):
        candidate, baseline, rule = (rows.get(name) for name in CONTROLLERS)
        if candidate is None or baseline is None or rule is None:
            continue
        metrics = ("seconds", "weighted_wait_seconds", "request_to_touchdown_seconds")
        vs_baseline = {metric: _delta(candidate.get(metric), baseline.get(metric)) for metric in metrics}
        vs_rule = {metric: _delta(candidate.get(metric), rule.get(metric)) for metric in metrics}
        deltas.append({
            "seed": seed,
            "scenario": scenario,
            "candidate_minus_baseline": vs_baseline,
            "candidate_minus_rule": vs_rule,
            "seconds_delta": {"baseline": vs_baseline["seconds"], "rule": vs_rule["seconds"]},
            "weighted_wait_seconds_delta": {
                "baseline": vs_baseline["weighted_wait_seconds"],
                "rule": vs_rule["weighted_wait_seconds"],
            },
            "request_to_touchdown_seconds_delta": {
                "baseline": vs_baseline["request_to_touchdown_seconds"],
                "rule": vs_rule["request_to_touchdown_seconds"],
            },
            "wait_seconds_delta": {
                "baseline": vs_baseline["weighted_wait_seconds"],
                "rule": vs_rule["weighted_wait_seconds"],
            },
        })
    return deltas


def smoke_semantics(*, smoke: bool, seeds: list[int] | tuple[int, ...], scenarios: list[str] | tuple[str, ...], steps: int) -> dict[str, bool]:
    """Pure helper used by tests to freeze smoke's non-reservation semantics."""
    return {
        "single_seed": smoke and list(seeds) == [SMOKE_SEED],
        "flow6": smoke and list(scenarios) == [SMOKE_SCENARIO],
        "steps": smoke and int(steps) == SMOKE_STEPS,
        "no_reservation": bool(smoke),
    }


def _formal_shape(records: list[Mapping[str, Any]]) -> bool:
    if len(records) != 18:
        return False
    by_controller = {name: [r for r in records if r.get("controller") == name] for name in CONTROLLERS}
    return all(
        len(rows) == 6
        and {int(r.get("result", {}).get("seed", -1)) for r in rows} == set(FORMAL_SEEDS)
        and [str(r.get("scenario")) for r in sorted(rows, key=lambda item: int(item.get("result", {}).get("seed", -1)))]
        == list(FORMAL_SCENARIOS) * 2
        and all(isinstance(r.get("result", {}).get("gates"), Mapping) and all(r["result"]["gates"].values()) for r in rows)
        for rows in by_controller.values()
    )


def run(output: Path, checkpoint: Path, baseline: Path, history_root: Path, *, smoke: bool = False) -> dict[str, Any]:
    """Run the smoke probe or the once-only formal paired evaluation."""
    p = protocol()
    _require(p["candidate"]["checkpoint_sha256"] == CANDIDATE_SHA, "candidate binding changed")
    _require(p["baseline"]["checkpoint_sha256"] == BASELINE_SHA, "baseline binding changed")
    _require(p["formal"]["seeds"] == list(FORMAL_SEEDS), "formal seeds binding changed")
    _require(p["formal"]["scenarios"] == list(FORMAL_SCENARIOS), "formal scenarios binding changed")
    _require(p["smoke"]["seed"] == SMOKE_SEED and p["smoke"]["steps"] == SMOKE_STEPS, "smoke binding changed")
    if not smoke:
        validate_training_manifest(checkpoint.parent / "manifest.json")
    candidate, candidate_payload = load_evaluation_policy(
        checkpoint,
        expected_sha=CANDIDATE_SHA,
        expected_stage=CANDIDATE_STAGE,
        expected_formal=True,
        expected_protocol_sha=TRAINING_PROTOCOL_SHA,
        expected_source_hashes=p2_source_hashes(),
    )
    baseline_policy, baseline_payload = load_evaluation_policy(
        baseline,
        expected_sha=BASELINE_SHA,
        expected_stage=BASELINE_STAGE,
        expected_formal=True,
    )
    output.mkdir(parents=True, exist_ok=False)
    seeds = [SMOKE_SEED] if smoke else list(FORMAL_SEEDS)
    scenarios = [SMOKE_SCENARIO] if smoke else list(FORMAL_SCENARIOS)
    if not smoke:
        history = historical_registry(history_root, output)
        _require(not (set(seeds) & history["seeds"]), "held-out seeds already consumed")
        reserve_once(
            checkpoint.parent / "p2_v4_heldout_consumed.json",
            {
                "protocol_sha256": PROTOCOL_SHA,
                "training_manifest_sha256": TRAINING_MANIFEST_SHA,
                "candidate_checkpoint_sha256": CANDIDATE_SHA,
                "baseline_checkpoint_sha256": BASELINE_SHA,
                "seeds": seeds,
                "output": str(output),
            },
        )
    _write(
        output / "pre_reset_registry.json",
        {
            "protocol_sha256": PROTOCOL_SHA,
            "seeds": seeds,
            "scenarios": scenarios,
            "controllers": list(CONTROLLERS),
            "smoke": smoke,
            "reservation": not smoke,
        },
    )
    records: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        scenario = scenarios[index % len(scenarios)]
        config = p2_config(6 if scenario == "flow6" else 8 if scenario == "flow8" else 12)
        for name, policy in (("candidate", candidate), ("baseline", baseline_policy)):
            folder = output / f"{name}_{seed}"
            result = evaluate(policy, config, seed, folder, smoke=smoke)
            result["scenario"] = scenario
            records.append({"controller": name, "scenario": scenario, "result": result})
            _write(output / "progress.json", {"completed": len(records), "total": 3 * len(seeds), "records": records})
        folder = output / f"rule_{seed}"
        result = evaluate_rule(config, seed, folder, smoke=smoke)
        result["scenario"] = scenario
        records.append({"controller": "rule", "scenario": scenario, "result": result})
        _write(output / "progress.json", {"completed": len(records), "total": 3 * len(seeds), "records": records})
    aggregates = {name: aggregate_records(records, name) for name in CONTROLLERS}
    formal_pass = (not smoke) and _formal_shape(records) and all(
        aggregates[name]["all_gates"] for name in CONTROLLERS
    )
    result = {
        "stage": "V4_2_P2_HELDOUT_EVALUATION",
        "smoke": smoke,
        "PASS": bool(formal_pass),
        "records": records,
        "aggregates": aggregates,
        "paired_deltas": pair_deltas(records),
        "checkpoint_sha256": CANDIDATE_SHA,
        "baseline_sha256": BASELINE_SHA,
        "training_manifest_sha256": TRAINING_MANIFEST_SHA,
        "protocol_sha256": PROTOCOL_SHA,
        "source_hashes": evaluation_source_hashes(),
        "training_called": False,
        "teacher_enabled": False,
        "deterministic": True,
        "candidate_payload_stage": candidate_payload.get("stage"),
        "baseline_payload_stage": baseline_payload.get("stage"),
        "smoke_semantics": smoke_semantics(smoke=smoke, seeds=seeds, scenarios=scenarios, steps=SMOKE_STEPS if smoke else 0),
    }
    _write(output / "manifest.json", result)
    print(json.dumps({"PASS": result["PASS"], "smoke": smoke, "records": len(records)}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("output", "checkpoint", "baseline", "history-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    torch.set_num_threads(1)
    run(**vars(parser.parse_args()))
