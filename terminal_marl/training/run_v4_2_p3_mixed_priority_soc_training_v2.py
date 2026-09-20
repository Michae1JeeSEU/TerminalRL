"""Preregistered P3 mixed priority/SOC PPO training repair.

V2 preserves the V1 safety, replay, likelihood, geometry, and optimizer
contracts while treating rollout completion as a reported performance
variable.  Entry and formal training each start from the original qualified
K=3 checkpoint with a fresh optimizer; entry updates are discarded.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from terminal_marl.core.checkpoints import save_checkpoint
from terminal_marl.training import (
    diagnose_v4_2_p3_mixed_priority_soc_pilot_v2 as pilot_v2,
)
from terminal_marl.training import (
    evaluate_v4_2_p2_gate_effectiveness_v1 as effectiveness,
)
from terminal_marl.training.evaluate_v4_2_p2_gate_runtime_shortlist_v1 import (
    _tape_shortlist_checks,
)
from terminal_marl.training.run_v4_2_i2_dagger_preflight import (
    _policy_tape_logprob_exact,
)
from terminal_marl.training.run_v4_2_p2_runtime_shortlist_entry_v1 import (
    _fresh_policy,
    _frozen_state_equal,
    _policy_tape_shortlists_exact,
    _restored_policy,
)
from terminal_marl.training.run_v4_2_ppo_block2 import replay_geometry, state_equal
from terminal_marl.training.run_v4_2_ppo_p2_resume_v4 import make_update_gates
from terminal_marl.training.run_v4_2_v2_1_formal_i0_r4 import historical_registry
from terminal_marl.training.run_v4_2_v2_1_i1_v2 import reserve_once
from terminal_marl.training.run_v4_2_v2_1_ppo import requested_logprob_error
from terminal_marl.training.student_rollout_v4_2 import rollout


STAGE = "V4_2_P3_MIXED_PRIORITY_SOC_TRAINING_V2"
V1_STAGE = "V4_2_P3_MIXED_PRIORITY_SOC_TRAINING_V1"
PROTOCOL = Path(
    "terminal_marl/configs/low_v4_2/p3_mixed_priority_soc_training_v2.yaml"
)
# Filled once the protocol bytes are final and before any training reset.
PROTOCOL_SHA = "4a445d885f0e7662c9f1592f44464d970ece8322aa8a7cf81012d59e7fd916ed"
SCHEDULER_SOURCE = Path("terminal_marl/core/merge_scheduler.py")
SCHEDULER_SOURCE_SHA = (
    "58bf3f3a29b86f00855610000a819424e5c90b0c1f1433a780bcb42f86309ce1"
)
PILOT_STAGE = "V4_2_P3_MIXED_PRIORITY_SOC_PILOT_V4"
V1_FAILED_MANIFEST_SHA = (
    "7b6420139c53f269aa48c25e75e6f1bd9205ea91fd38e2fdf8d0088b59619c59"
)
TRAINABLE_PREFIXES = ("gate_adapter.", "critic_encoder.", "critic_head.")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def protocol() -> dict[str, Any]:
    """Read the byte-frozen V2 training protocol and its immutable bindings."""
    if PROTOCOL_SHA == "TO_BE_FROZEN" or _sha256(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("P3 mixed priority/SOC training V2 protocol changed")
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("protocol") != STAGE:
        raise ValueError("P3 mixed priority/SOC training V2 protocol id changed")
    if value.get("status") != "PREREGISTERED_TRAINING_REPAIR":
        raise ValueError("P3 V2 protocol status changed")
    bindings = value.get("bindings", {})
    if bindings.get("superseded_training_stage") != V1_STAGE:
        raise ValueError("superseded V1 stage binding changed")
    if bindings.get("superseded_training_manifest_sha256") != V1_FAILED_MANIFEST_SHA:
        raise ValueError("superseded V1 manifest SHA binding changed")
    failure = bindings.get("superseded_training_failure")
    expected_failure = {
        "kind": "completion_floor",
        "seed": 818003,
        "scenario": "dense24",
        "formal_episode_index": 3,
        "formal_episode_count": 8,
        "only_failed_gate": "completion_floor",
        "safety_and_integrity_gates_passed": True,
        "update_attempted": False,
        "observed_completion_fraction": 0.5,
        "required_completion_fraction": 0.75,
        "heldout_seeds_already_reserved": [819001, 819002, 819003, 819004, 819005, 819006],
    }
    if failure != expected_failure:
        raise ValueError("superseded V1 semantic failure binding changed")
    if bindings.get("pilot_stage") != PILOT_STAGE:
        raise ValueError("P3 pilot stage binding changed")
    if bindings.get("scheduler_source") != SCHEDULER_SOURCE.as_posix():
        raise ValueError("P3 scheduler source binding changed")
    if bindings.get("scheduler_source_sha256") != SCHEDULER_SOURCE_SHA:
        raise ValueError("P3 scheduler source SHA binding changed")
    hard_gates = value.get("hard_gates", {})
    if not isinstance(hard_gates, Mapping):
        raise ValueError("P3 V2 hard gates are not a mapping")
    if "completion_floor_by_scenario" in hard_gates:
        raise ValueError("P3 V2 must not preregister a completion gate")
    reporting = value.get("completion_reporting", {})
    if (
        reporting.get("mode") != "report_only"
        or reporting.get("training_rollout_completion_gate") is not False
    ):
        raise ValueError("P3 V2 completion semantics changed")
    return value


def source_hashes() -> dict[str, str]:
    """Hash sources whose behavior is part of the V2 training contract."""
    paths = (
        PROTOCOL,
        Path("terminal_marl/training/run_v4_2_p3_mixed_priority_soc_training_v2.py"),
        Path("terminal_marl/training/diagnose_v4_2_p3_mixed_priority_soc_pilot_v2.py"),
        Path("terminal_marl/training/run_v4_2_p2_runtime_shortlist_entry_v1.py"),
        Path("terminal_marl/training/run_v4_2_ppo_p2_resume_v4.py"),
        Path("terminal_marl/algo/joint_ppo_v4_2.py"),
        Path("terminal_marl/algo/student_ppo_v4_2.py"),
        Path("terminal_marl/algo/gate_calibrated_adapter_v4_2.py"),
        Path("terminal_marl/core/checkpoints.py"),
        Path("terminal_marl/core/low_level_joint_action_v4_2.py"),
        Path("terminal_marl/core/safety_trace.py"),
        Path("terminal_marl/core/arrival_process.py"),
        Path("terminal_marl/core/priority.py"),
        Path("terminal_marl/core/merge_scheduler.py"),
        Path("terminal_marl/envs/parallel_env.py"),
        Path("terminal_marl/training/student_rollout_v4_2.py"),
    )
    return {path.as_posix(): _sha256(path) for path in paths}


def validate_pilot_manifest(path: Path, p: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless the exact V4 pilot authorizes preregistration."""
    bindings = p["bindings"]
    if _sha256(path) != bindings["pilot_manifest_sha256"]:
        raise ValueError("P3 V4 pilot manifest SHA mismatch")
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("P3 V4 pilot manifest must be an object")
    gates = report.get("gates")
    expected = {
        "stage": PILOT_STAGE,
        "PASS": True,
        "training_called": False,
        "PPO_ALLOWED": False,
        "P3_MIXED_TRAINING_PROTOCOL_PREREGISTRATION_ALLOWED": True,
        "protocol_sha256": bindings["pilot_protocol_sha256"],
        "scheduler_source": bindings["scheduler_source"],
        "scheduler_source_sha256": bindings["scheduler_source_sha256"],
        "checkpoint_sha256": bindings["candidate_checkpoint_sha256"],
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise PermissionError("P3 V4 pilot has not authorized training preregistration")
    if report.get("P3_FORMAL_ALLOWED", False) is not False:
        raise PermissionError("P3 V4 pilot unexpectedly authorizes formal training")
    if report.get("checkpoint_promotable", False) is not False:
        raise PermissionError("P3 V4 pilot unexpectedly promotes a checkpoint")
    if not isinstance(gates, Mapping) or not gates or not all(bool(value) for value in gates.values()):
        raise PermissionError("P3 V4 pilot gates are not all true")
    return report


def validate_superseded_v1_manifest(path: Path, p: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the immutable V1 failure before reserving any V2 seed."""
    bindings = p["bindings"]
    if _sha256(path) != bindings["superseded_training_manifest_sha256"]:
        raise ValueError("superseded V1 manifest SHA mismatch")
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("superseded V1 manifest must be an object")
    if report.get("stage") != V1_STAGE or report.get("PASS") is not False:
        raise PermissionError("superseded V1 root stage or PASS changed")
    if report.get("HELDOUT_EVALUATION_ALLOWED") is not False:
        raise PermissionError("superseded V1 unexpectedly authorizes heldout evaluation")
    if report.get("heldout_untouched") is not True:
        raise PermissionError("superseded V1 heldout was not untouched")
    if report.get("P4_ALLOWED") is not False:
        raise PermissionError("superseded V1 unexpectedly authorizes P4")
    if report.get("checkpoint_promotable") is not False:
        raise PermissionError("superseded V1 unexpectedly promotes a checkpoint")
    entry = report.get("entry")
    formal = report.get("formal")
    if not isinstance(entry, Mapping) or entry.get("PASS") is not True:
        raise PermissionError("superseded V1 entry result is not the bound PASS result")
    if not isinstance(formal, Mapping) or formal.get("PASS") is not False:
        raise PermissionError("superseded V1 formal result is not the bound FAIL result")
    if formal.get("final_checkpoint") is not None or formal.get("final_checkpoint_sha256") is not None:
        raise PermissionError("superseded V1 unexpectedly has a final checkpoint")
    if report.get("formal_training_called") is not True:
        raise PermissionError("superseded V1 formal training result is missing")
    formal_gates = formal.get("gates")
    if not isinstance(formal_gates, Mapping):
        raise PermissionError("superseded V1 formal stage summary is missing gates")
    if formal_gates.get("all_episodes") is not False or formal_gates.get("all_episode_gates") is not False:
        raise PermissionError("superseded V1 formal stop summary changed")
    expected_failure = {
        "kind": "completion_floor",
        "seed": 818003,
        "scenario": "dense24",
        "formal_episode_index": 3,
        "formal_episode_count": 8,
        "only_failed_gate": "completion_floor",
        "safety_and_integrity_gates_passed": True,
        "update_attempted": False,
        "observed_completion_fraction": 0.5,
        "required_completion_fraction": 0.75,
        "heldout_seeds_already_reserved": [819001, 819002, 819003, 819004, 819005, 819006],
    }
    if bindings.get("superseded_training_failure") != expected_failure:
        raise PermissionError("superseded V1 semantic failure binding changed")
    return report


# Alias kept for callers that use the shorter name in validation harnesses.
validate_v1_failed_manifest = validate_superseded_v1_manifest


def scenario_config(name: str, p: Mapping[str, Any]):
    """Reuse the qualified V2 pilot builder for exact scenario semantics."""
    return pilot_v2.scenario_config(name, dict(p))


def stage_initialization_contract(stage: str, p: Mapping[str, Any]) -> dict[str, Any]:
    if stage not in {"entry", "formal"}:
        raise ValueError(f"unknown P3 training stage: {stage}")
    return {
        "checkpoint_sha256": p["bindings"]["candidate_checkpoint_sha256"],
        "fresh_optimizer": True,
        "entry_updates_discarded": stage == "entry",
        "formal_initialization_is_original": stage == "formal",
    }


def _scenario_horizon(
    config: Any, name: str, *, stage: str, index: int, p: Mapping[str, Any]
) -> tuple[Any, int]:
    if stage == "entry":
        horizon = int(p["entry"]["steps"][index])
        return replace(config, max_simulation_seconds=float(horizon) * config.dt), horizon
    if name == "dense24":
        seconds = float(p["formal"]["dense_horizon_seconds"])
        return replace(config, max_simulation_seconds=seconds), int(seconds / config.dt)
    return config, int(config.max_simulation_seconds / config.dt)


def _action_tape_path(result: Mapping[str, Any]) -> str:
    summary = result.get("summary")
    path = summary.get("action_tape") if isinstance(summary, Mapping) else None
    if not isinstance(path, str) or not path:
        path = result.get("action_tape")
    if not isinstance(path, str) or not path:
        raise KeyError("evaluation result has no action_tape")
    return path


def _no_failed_outcomes(summary: Mapping[str, Any]) -> bool:
    values = (summary.get("outcomes") or {}).values()
    return all(str(getattr(value, "value", value)) == "completed" for value in values)


def _rollout_gates(
    policy: Any,
    config: Any,
    steps: list[Any],
    summary: Mapping[str, Any],
    p: Mapping[str, Any],
    *,
    formal: bool,
    scenario: str,
) -> tuple[dict[str, bool], dict[str, Any], dict[str, Any], float]:
    del formal, scenario  # Completion is deliberately report-only in V2.
    tape_path = _action_tape_path({"summary": summary})
    replay_exact, geometry = replay_geometry(config, tape_path)
    likelihood = _policy_tape_logprob_exact(policy, config, tape_path)
    initial_error = requested_logprob_error(policy, steps)
    shortlist = _tape_shortlist_checks(tape_path, expected_k=3)
    counts = summary.get("counts") or {}
    actor_decisions = int(counts.get("actor_decisions", 0))
    shield_actor = int(counts.get("shield_actor", 0))
    shield_fraction = shield_actor / max(1, actor_decisions)
    hard = p["hard_gates"]
    gates: dict[str, bool] = {
        "safety": not any(bool(value) for value in (summary.get("safety") or {}).values()),
        "replay": bool(replay_exact),
        "requested_likelihood": bool(likelihood),
        "initial_ratio": initial_error <= float(hard["initial_ratio_error_max"]),
        "student_only": int(summary.get("label_rows", 0)) == 0
        and all(not bool(step.teacher_used) for step in steps),
        "stochastic": bool(steps) and all(bool(step.stochastic_behavior) for step in steps),
        "actor_decisions": actor_decisions > 0,
        "runtime_shortlist_contract": bool(shortlist.get("all")),
        "shield_burden": shield_fraction <= float(hard["shield_actor_fraction_max"]),
        "geometry": int(geometry["reverse_samples"]) <= int(hard["geometry_reverse_samples_max"])
        and float(geometry["max_inner_excursion_m"])
        <= float(hard["geometry_inner_excursion_max_m"]),
        "no_failed_outcomes": _no_failed_outcomes(summary),
    }
    completion = int(summary.get("completed", 0)) / max(
        1, int(summary.get("requests", config.total_requested_uavs))
    )
    diagnostics = {
        "replay_exact": bool(replay_exact),
        "requested_likelihood_exact": bool(likelihood),
        "initial_ratio_error": float(initial_error),
        "geometry": geometry,
        "runtime_shortlist_checks": shortlist,
        "actor_decisions": actor_decisions,
        "shield_actor_fraction": shield_fraction,
        "completion_fraction": completion,
        "no_failed_outcomes": _no_failed_outcomes(summary),
        "completion_gate": False,
    }
    return gates, diagnostics, shortlist, completion


def _coverage(attributes: list[dict[str, Any]], p: Mapping[str, Any]) -> dict[str, Any]:
    priorities = sorted({int(value) for row in attributes for value in row.get("priorities", [])})
    soc = [float(value) for row in attributes for value in row.get("initial_soc", [])]
    required = [int(value) for value in p["hard_gates"]["formal_priority_coverage_required"]]
    lower = float(p["attributes"]["initial_soc_min"])
    upper = float(p["attributes"]["initial_soc_max"])
    return {
        "priorities_observed": priorities,
        "priority_coverage": priorities == required,
        "soc_count": len(soc),
        "soc_min": None if not soc else min(soc),
        "soc_max": None if not soc else max(soc),
        "soc_in_bounds": bool(soc) and all(lower <= value <= upper for value in soc),
        "soc_range_observed": bool(soc)
        and min(soc) <= float(p["hard_gates"]["formal_observed_soc_min_at_most"])
        and max(soc) >= float(p["hard_gates"]["formal_observed_soc_max_at_least"]),
    }


def _transactional_update(
    policy: Any,
    steps: list[Any],
    config: Any,
    tape_path: str,
    p: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, bool]]:
    optimizer = p["optimizer"]
    model_snapshot = copy.deepcopy(policy.model.state_dict())
    optimizer_snapshot = copy.deepcopy(policy.optimizer.state_dict())
    keys = ("minibatch_size", "gamma", "gae_lambda", "clip_ratio", "value_coef", "entropy_coef")
    try:
        update = policy.update(
            steps,
            training_authorized=True,
            epochs=int(optimizer["update_epochs"]),
            target_kl=float(optimizer["target_kl"]),
            **{key: optimizer[key] for key in keys},
        )
        gates = make_update_gates(
            update,
            policy.model.state_dict(),
            policy.optimizer.state_dict(),
            optimizer,
        )
        gates["frozen_proposal_subgraph"] = _frozen_state_equal(
            model_snapshot, policy.model.state_dict(),
        )
        gates["runtime_shortlist_stability"] = _policy_tape_shortlists_exact(
            policy, config, tape_path,
        )
    except (KeyError, TypeError, ValueError, RuntimeError, FloatingPointError) as exc:
        policy.model.load_state_dict(model_snapshot)
        policy.optimizer.load_state_dict(optimizer_snapshot)
        return {"exception": type(exc).__name__, "message": str(exc)}, {"update_transaction": False}
    if not all(bool(value) for value in gates.values()):
        policy.model.load_state_dict(model_snapshot)
        policy.optimizer.load_state_dict(optimizer_snapshot)
    return update, {key: bool(value) for key, value in gates.items()}


def _training_stage(
    *, stage: str, output: Path, checkpoint: Path, p: Mapping[str, Any]
) -> dict[str, Any]:
    formal = stage == "formal"
    section = p["formal"] if formal else p["entry"]
    stage_initialization_contract(stage, p)
    policy, _, trainable_parameters = _fresh_policy(checkpoint, dict(p))
    output.mkdir(parents=True, exist_ok=False)
    np.random.seed(int(section["seeds"][0]))
    records: list[dict[str, Any]] = []
    attributes: list[dict[str, Any]] = []
    for index, seed_value in enumerate(section["seeds"]):
        seed = int(seed_value)
        scenario = str(section["scenarios"][index])
        config, horizon = _scenario_horizon(
            scenario_config(scenario, p), scenario, stage=stage, index=index, p=p,
        )
        steps, summary = rollout(
            policy,
            config,
            seed=seed,
            max_steps=horizon,
            output=output / f"seed{seed}",
            deterministic=False,
            collect_labels=False,
        )
        tape_path = _action_tape_path({"summary": summary})
        gates, diagnostics, shortlist, completion = _rollout_gates(
            policy, config, steps, summary, p, formal=formal, scenario=scenario,
        )
        row_attributes = None
        if formal:
            row_attributes = pilot_v2.lifecycle_metrics(
                config,
                tape_path,
                low_soc_threshold=float(p["attributes"]["low_soc_analysis_threshold"]),
            )
            attributes.append(row_attributes)
            gates["attribute_replay"] = bool(row_attributes["replay_exact"])
        update = None
        update_gates: dict[str, bool] | None = None
        saved: Path | None = None
        # Completion is diagnostic only.  Every gate here is integrity, data,
        # safety, or update validity; a deficient completion value still trains.
        if all(bool(value) for value in gates.values()):
            update, update_gates = _transactional_update(policy, steps, config, tape_path, p)
            gates.update(update_gates)
        if all(bool(value) for value in gates.values()) and formal:
            saved = output / f"checkpoint_{index + 1}.pt"
            try:
                save_checkpoint(
                    saved,
                    policy,
                    config,
                    stage=STAGE,
                    formal=True,
                    ppo_called=True,
                    protocol_sha256=PROTOCOL_SHA,
                    source_hashes=source_hashes(),
                    initial_checkpoint_sha256=p["bindings"]["candidate_checkpoint_sha256"],
                    checkpoint_promotable=False,
                )
                restored = _restored_policy(saved, config, dict(p))
                gates["checkpoint_restore"] = state_equal(
                    policy.model.state_dict(), restored.model.state_dict(),
                ) and state_equal(
                    policy.optimizer.state_dict(), restored.optimizer.state_dict(),
                )
                if gates["checkpoint_restore"]:
                    policy = restored
                else:
                    saved.unlink(missing_ok=True)
                    saved = None
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                gates["checkpoint_restore"] = False
                saved.unlink(missing_ok=True)
                saved = None
        record = {
            "seed": seed,
            "scenario": scenario,
            "horizon_steps": horizon,
            "summary": summary,
            "completion_fraction": completion,
            "completion_report_only": True,
            "diagnostics": diagnostics,
            "runtime_shortlist_checks": shortlist,
            "attributes": row_attributes,
            "update": update,
            "update_gates": update_gates,
            "checkpoint": None if saved is None else str(saved),
            "gates": gates,
        }
        records.append(record)
        _write(output / "progress.json", {
            "stage": stage,
            "completed": len(records),
            "total": len(section["seeds"]),
            "entry_updates_discarded": not formal,
            "formal_starts_original_checkpoint": formal,
            "completion_report_only": True,
            "records": records,
        })
        # Stop only on an integrity/update failure; performance is not a gate.
        if not all(bool(value) for value in gates.values()):
            break
    stage_gates = {
        "all_episodes": len(records) == len(section["seeds"]),
        "all_episode_gates": bool(records)
        and all(all(bool(value) for value in row["gates"].values()) for row in records),
    }
    coverage = None
    if formal:
        coverage = _coverage(attributes, p)
        stage_gates.update({
            "priority_coverage": bool(coverage["priority_coverage"]),
            "soc_in_bounds": bool(coverage["soc_in_bounds"]),
            "soc_range_observed": bool(coverage["soc_range_observed"]),
        })
    passed = all(bool(value) for value in stage_gates.values())
    final_checkpoint = (
        Path(records[-1]["checkpoint"])
        if formal and records and records[-1].get("checkpoint") else None
    )
    result = {
        "stage": "P3_FORMAL" if formal else "P3_ENTRY",
        "PASS": passed,
        "gates": stage_gates,
        "records": records,
        "coverage": coverage,
        "entry_updates_discarded": not formal,
        "formal_starts_original_checkpoint": formal,
        "formal_training_called": formal,
        "checkpoint_promotable": False,
        "P4_ALLOWED": False,
        "completion_report_only": True,
        "final_checkpoint": None if final_checkpoint is None else str(final_checkpoint),
        "final_checkpoint_sha256": None if final_checkpoint is None else _sha256(final_checkpoint),
        "trainable_parameters": list(trainable_parameters),
    }
    _write(output / "manifest.json", result)
    return result


def run(
    *,
    output: Path,
    pilot_manifest_path: Path,
    superseded_v1_manifest_path: Path,
    checkpoint: Path,
    history_root: Path,
) -> dict[str, Any]:
    """Validate immutable inputs, reserve fresh seeds, then run entry/formal."""
    p = protocol()
    bindings = p["bindings"]
    validate_superseded_v1_manifest(superseded_v1_manifest_path, p)
    validate_pilot_manifest(pilot_manifest_path, p)
    if _sha256(SCHEDULER_SOURCE) != SCHEDULER_SOURCE_SHA:
        raise ValueError("P3 scheduler source changed after pilot qualification")
    if _sha256(checkpoint) != bindings["candidate_checkpoint_sha256"]:
        raise ValueError("P3 candidate checkpoint SHA mismatch")

    output.mkdir(parents=True, exist_ok=False)
    entry_seeds = [int(seed) for seed in p["entry"]["seeds"]]
    formal_seeds = [int(seed) for seed in p["formal"]["seeds"]]
    heldout_seeds = [int(seed) for seed in p["heldout"]["seeds"]]
    all_reserved = set(entry_seeds + formal_seeds + heldout_seeds)
    history = historical_registry(history_root, output)
    overlap = sorted(all_reserved & set(history["seeds"]))
    if overlap:
        raise ValueError(f"P3 V2 training or heldout seeds were previously consumed: {overlap}")
    marker = checkpoint.parent / str(p["reservation"]["marker"])
    reserve_once(marker, {
        "stage": STAGE,
        "protocol_sha256": PROTOCOL_SHA,
        "pilot_manifest_sha256": bindings["pilot_manifest_sha256"],
        "superseded_training_manifest_sha256": bindings["superseded_training_manifest_sha256"],
        "candidate_checkpoint_sha256": bindings["candidate_checkpoint_sha256"],
        "entry_seeds": entry_seeds,
        "formal_seeds": formal_seeds,
        "heldout_seeds": heldout_seeds,
        "heldout_untouched": True,
        "output": str(output),
        "reserved_before_first_reset": True,
    })
    _write(output / "pre_reset_registry.json", {
        "stage": STAGE,
        "protocol_sha256": PROTOCOL_SHA,
        "pilot_manifest_sha256": bindings["pilot_manifest_sha256"],
        "superseded_training_manifest_sha256": bindings["superseded_training_manifest_sha256"],
        "candidate_checkpoint_sha256": bindings["candidate_checkpoint_sha256"],
        "entry_seeds": entry_seeds,
        "formal_seeds": formal_seeds,
        "heldout_seeds": heldout_seeds,
        "reservation_marker": str(marker),
        "heldout_untouched": True,
        "reserved_before_first_reset": True,
    })

    entry = _training_stage(
        stage="entry", output=output / "entry", checkpoint=checkpoint, p=p,
    )
    formal = None
    if entry["PASS"]:
        formal = _training_stage(
            stage="formal", output=output / "formal", checkpoint=checkpoint, p=p,
        )
    passed = formal is not None and bool(formal["PASS"])
    result = {
        "stage": STAGE,
        "PASS": bool(passed),
        "entry": {
            "PASS": bool(entry["PASS"]),
            "gates": entry["gates"],
            "entry_updates_discarded": True,
            "formal_initialization_checkpoint_sha256": bindings["candidate_checkpoint_sha256"],
        },
        "formal": None if formal is None else {
            "PASS": bool(formal["PASS"]),
            "gates": formal["gates"],
            "coverage": formal["coverage"],
            "final_checkpoint": formal["final_checkpoint"],
            "final_checkpoint_sha256": formal["final_checkpoint_sha256"],
            "formal_initialization_checkpoint_sha256": bindings["candidate_checkpoint_sha256"],
        },
        "entry_updates_discarded": True,
        "formal_starts_original_checkpoint": True,
        # Entry always performs PPO updates before formal admission is known.
        # Keep this root audit field truthful even if entry later blocks formal.
        "PPO_called": True,
        "formal_training_called": formal is not None,
        "heldout_untouched": True,
        "heldout_seeds_reserved": heldout_seeds,
        "HELDOUT_EVALUATION_ALLOWED": bool(passed),
        "formal_pass_authorizes_heldout_only": True,
        "P4_ALLOWED": False,
        "checkpoint_promotable": False,
        "completion_report_only": True,
        "protocol_sha256": PROTOCOL_SHA,
        "source_hashes": source_hashes(),
        "pilot_manifest_sha256": bindings["pilot_manifest_sha256"],
        "pilot_protocol_sha256": bindings["pilot_protocol_sha256"],
        "superseded_training_manifest_sha256": bindings["superseded_training_manifest_sha256"],
        "superseded_training_failure": bindings["superseded_training_failure"],
        "scheduler_source": bindings["scheduler_source"],
        "scheduler_source_sha256": SCHEDULER_SOURCE_SHA,
        "candidate_checkpoint_sha256": bindings["candidate_checkpoint_sha256"],
    }
    _write(output / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--superseded-v1-manifest", "--v1-manifest", dest="superseded_v1_manifest", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--history-root", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = run(
        output=args.output,
        pilot_manifest_path=args.pilot_manifest,
        superseded_v1_manifest_path=args.superseded_v1_manifest,
        checkpoint=args.candidate_checkpoint,
        history_root=args.history_root,
    )
    print(json.dumps({
        "stage": result["stage"],
        "PASS": result["PASS"],
        "HELDOUT_EVALUATION_ALLOWED": result["HELDOUT_EVALUATION_ALLOWED"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
