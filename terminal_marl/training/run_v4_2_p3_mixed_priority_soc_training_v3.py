"""P3 V3 PPO repair with optional priority/SOC urgency shaping.

V3 is an integrity-gated training protocol.  It starts both entry and formal
training from the original qualified K=3 checkpoint with a fresh optimizer;
the previous V2 trained checkpoint is retained only as immutable failure
evidence.  Completion and wait performance are diagnostics, never training
stop gates.
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

from terminal_marl.algo.gate_calibrated_adapter_v4_2 import (
    CALIBRATION_TEMPERATURE,
    TemperatureCalibratedGateResidualAdapterStudentPPOV42,
)
from terminal_marl.core.checkpoints import load_checkpoint, save_checkpoint
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
    _frozen_state_equal,
    _policy_tape_shortlists_exact,
)
from terminal_marl.training.run_v4_2_ppo_block2 import replay_geometry, state_equal
from terminal_marl.training.run_v4_2_ppo_p2_resume_v4 import (
    make_update_gates,
    p2_config,
)
from terminal_marl.training.run_v4_2_v2_1_formal_i0_r4 import historical_registry
from terminal_marl.training.run_v4_2_v2_1_i1_v2 import reserve_once
from terminal_marl.training.run_v4_2_v2_1_ppo import requested_logprob_error
from terminal_marl.training.student_rollout_v4_2 import rollout


STAGE = "V4_2_P3_MIXED_PRIORITY_SOC_TRAINING_V3"
PROTOCOL = Path(
    "terminal_marl/configs/low_v4_2/p3_mixed_priority_soc_training_v3.yaml"
)
PROTOCOL_SHA = "c69dc4b308e97652936c8da5bb6a47ef61a73a4f3cb599de8f39cb703f3c4d25"
FAILED_HELDOUT_STAGE = "V4_2_P3_MIXED_PRIORITY_SOC_HELDOUT_V1"
FAILED_HELDOUT_MANIFEST_SHA = (
    "fdfdaefb4ae6ae0e72eb016301b20990543ab63227e9348201308f907921bae6"
)
FAILED_HELDOUT_PROTOCOL_SHA = (
    "acbc92492edf0d65ced142a5069d0b1386f742105d8ad7a616aa3390949442d3"
)
INITIAL_CHECKPOINT_SHA = (
    "4b8be66c100aaaeb5b5699dcce68a96dbe1237964e4c308e7b23b95c62fb8c8c"
)
V2_TRAINED_CHECKPOINT_SHA = (
    "bdff62ac94fbbd42ebb0fc571c09b6b2cd7b182058b0bf1a2ef496ab863ca9c6"
)
SCHEDULER_SOURCE = Path("terminal_marl/core/merge_scheduler.py")
SCHEDULER_SOURCE_SHA = (
    "58bf3f3a29b86f00855610000a819424e5c90b0c1f1433a780bcb42f86309ce1"
)
REWARD_SOURCE_HASHES = {
    "terminal_marl/core/rewards.py": (
        "e39503bb2e6ed021d03268c0894fa123efb39ef30b85a225f721b4a1f9dfb28b"
    ),
    "terminal_marl/core/config.py": (
        "5fbadde0a690b02db05fcaafdca76c814502a33e382f57a5700aed6ce86f9dd7"
    ),
    "terminal_marl/envs/parallel_env.py": (
        "925ac9a03e56327b98ef29d0b5c408720795cfa61fde6e8ff3ef5036f8255316"
    ),
}
TRAINABLE_PREFIXES = ("gate_adapter.", "critic_encoder.", "critic_head.")
REWARD_CONFIG = {
    "low_level_priority_time_weight": 1.0,
    "low_level_soc_urgency_time_weight": 2.0,
    "low_level_soc_urgency_threshold": 0.65,
}


class V3InputError(ValueError):
    """Raised when a frozen V3 input violates its contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V3InputError(message)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(path: str | Path) -> str:
    return _sha256(path)


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"JSON input missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def protocol() -> dict[str, Any]:
    """Read and validate the byte-frozen V3 protocol."""
    _require(PROTOCOL.is_file(), f"V3 protocol missing: {PROTOCOL}")
    _require(PROTOCOL_SHA != "TO_BE_FROZEN" and _sha(PROTOCOL) == PROTOCOL_SHA,
             "P3 V3 protocol changed after freeze")
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), "P3 V3 protocol must be a mapping")
    _require(value.get("protocol") == STAGE, "P3 V3 protocol id changed")
    _require(value.get("status") == "PREREGISTERED_TRAINING_REPAIR",
             "P3 V3 protocol status changed")
    bindings = value.get("bindings")
    _require(isinstance(bindings, Mapping), "P3 V3 bindings are missing")
    expected_bindings = {
        "superseded_heldout_stage": FAILED_HELDOUT_STAGE,
        "superseded_heldout_manifest_sha256": FAILED_HELDOUT_MANIFEST_SHA,
        "superseded_heldout_protocol_sha256": FAILED_HELDOUT_PROTOCOL_SHA,
        "superseded_heldout_pass": False,
        "superseded_heldout_p3_qualified": False,
        "superseded_heldout_checkpoint_promotable": False,
        "superseded_heldout_P4_allowed": False,
        "superseded_heldout_records": 18,
        "superseded_heldout_episode_integrity": True,
        "initial_checkpoint_sha256": INITIAL_CHECKPOINT_SHA,
        "forbidden_v2_trained_checkpoint_sha256": V2_TRAINED_CHECKPOINT_SHA,
        "scheduler_source": SCHEDULER_SOURCE.as_posix(),
        "scheduler_source_sha256": SCHEDULER_SOURCE_SHA,
    }
    for key, expected in expected_bindings.items():
        _require(bindings.get(key) == expected, f"V3 binding changed: {key}")
    _require(bindings.get("reward_sources") == REWARD_SOURCE_HASHES,
             "V3 reward-source binding changed")
    expected_failure = {
        "all_records": True,
        "all_episode_integrity": True,
        "frozen_proposal_subgraph": True,
        "none_metrics_fail_closed": True,
        "trained_sparse_completion": True,
        "trained_dense_completion_aggregate_nonworse_than_initial": True,
        "trained_dense_strict_improvement": False,
        "trained_sparse_priority_weighted_wait_nonworse_than_initial": True,
        "trained_dense_priority_weighted_wait_nonworse_than_initial": False,
        "trained_overall_low_soc_mean_wait_nonworse_than_initial": False,
        "rule_diagnostic_only": True,
        "all": False,
    }
    _require(bindings.get("superseded_heldout_failed_gates") == expected_failure,
             "superseded held-out failure gates changed")
    reward = value.get("reward")
    _require(isinstance(reward, Mapping), "V3 reward section is missing")
    for key, expected in REWARD_CONFIG.items():
        _require(float(reward.get(key)) == expected, f"V3 reward config changed: {key}")
    _require(reward.get("urgency_reward_config_exact") is True,
             "urgency reward gate binding changed")
    _require(value.get("completion_reporting", {}).get("mode") == "report_only",
             "V3 completion mode changed")
    _require(value.get("completion_reporting", {}).get("training_rollout_completion_gate") is False,
             "V3 completion gate changed")
    _require("completion_floor_by_scenario" not in value.get("hard_gates", {}),
             "V3 must not use a completion training gate")
    hard = value.get("hard_gates", {})
    _require(hard.get("urgency_reward_config_exact") is True,
             "V3 urgency hard gate changed")
    _require(value.get("entry", {}).get("seeds") == [823001, 823002],
             "V3 entry seeds changed")
    _require(value.get("entry", {}).get("scenarios") == ["dense24", "flow6"],
             "V3 entry scenarios changed")
    _require(value.get("entry", {}).get("steps") == [320, 200],
             "V3 entry steps changed")
    _require(value.get("formal", {}).get("seeds") == list(range(824001, 824013)),
             "V3 formal seeds changed")
    _require(value.get("formal", {}).get("scenarios") == [
        "dense24", "flow6", "dense24", "flow12",
        "dense24", "flow6", "dense24", "flow12",
        "dense24", "flow6", "dense24", "flow12",
    ], "V3 formal scenarios changed")
    _require(value.get("formal", {}).get("full_scenario_horizon") is True,
             "V3 formal horizon contract changed")
    _require(value.get("heldout", {}).get("seeds") == list(range(825001, 825007)),
             "V3 held-out seeds changed")
    _require(value.get("heldout", {}).get("untouched") is True,
             "V3 held-out immutability changed")
    _require(value.get("reservation", {}).get("marker") ==
             "p3_mixed_priority_soc_training_v3_consumed.json",
             "V3 reservation marker changed")
    _require(value.get("authorization", {}).get("P3_QUALIFIED") is False,
             "V3 P3 qualification authorization changed")
    _require(value.get("authorization", {}).get("P4_ALLOWED") is False,
             "V3 P4 authorization changed")
    _require(value.get("authorization", {}).get("checkpoint_promotable") is False,
             "V3 checkpoint promotion authorization changed")
    return value


def source_hashes() -> dict[str, str]:
    paths = (
        PROTOCOL,
        Path("terminal_marl/training/run_v4_2_p3_mixed_priority_soc_training_v3.py"),
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
        SCHEDULER_SOURCE,
        Path("terminal_marl/core/rewards.py"),
        Path("terminal_marl/core/config.py"),
        Path("terminal_marl/envs/parallel_env.py"),
        Path("terminal_marl/training/student_rollout_v4_2.py"),
    )
    hashes = {path.as_posix(): _sha(path) for path in paths}
    for path, expected in REWARD_SOURCE_HASHES.items():
        _require(hashes.get(path) == expected, f"reward source changed: {path}")
    return hashes


def validate_failed_heldout_manifest(path: Path, p: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact V1 held-out failure before any V3 reset."""
    bindings = p["bindings"]
    _require(_sha(path) == bindings["superseded_heldout_manifest_sha256"],
             "superseded held-out manifest SHA mismatch")
    report = _read_json(path)
    for key, expected in {
        "stage": FAILED_HELDOUT_STAGE,
        "PASS": False,
        "P3_QUALIFIED": False,
        "checkpoint_promotable": False,
        "P4_ALLOWED": False,
        "protocol_sha256": FAILED_HELDOUT_PROTOCOL_SHA,
        "trained_checkpoint_sha256": V2_TRAINED_CHECKPOINT_SHA,
        "initial_checkpoint_sha256": INITIAL_CHECKPOINT_SHA,
    }.items():
        _require(report.get(key) == expected, f"superseded held-out field changed: {key}")
    records = report.get("records")
    _require(isinstance(records, list) and len(records) == 18,
             "superseded held-out record count changed")
    for index, record in enumerate(records):
        _require(isinstance(record, Mapping), f"superseded held-out record {index} invalid")
        _require(record.get("episode_integrity") is True,
                 f"superseded held-out record {index} integrity changed")
        result = record.get("result")
        _require(isinstance(result, Mapping),
                 f"superseded held-out record {index} result missing")
        gates = result.get("gates")
        _require(isinstance(gates, Mapping) and gates and all(bool(v) for v in gates.values()),
                 f"superseded held-out record {index} episode gate changed")
    _require(report.get("decision") == bindings["superseded_heldout_failed_gates"],
             "superseded held-out failed-gate evidence changed")
    return report


def scenario_config(name: str, p: Mapping[str, Any]):
    """Reuse only the frozen V2 scenario builder, then apply V3 reward values."""
    reward = p["reward"]
    return replace(
        pilot_v2.scenario_config(name, dict(p)),
        low_level_priority_time_weight=float(reward["low_level_priority_time_weight"]),
        low_level_soc_urgency_time_weight=float(reward["low_level_soc_urgency_time_weight"]),
        low_level_soc_urgency_threshold=float(reward["low_level_soc_urgency_threshold"]),
    )


def _urgency_reward_config_exact(config: Any, p: Mapping[str, Any]) -> bool:
    return all(
        float(getattr(config, key)) == float(p["reward"][key])
        for key in REWARD_CONFIG
        if key != "urgency_reward_config_exact"
    )


def _fresh_policy(checkpoint: Path, p: Mapping[str, Any]):
    bindings, optimizer = p["bindings"], p["optimizer"]
    _require(_sha(checkpoint) == bindings["initial_checkpoint_sha256"],
             "V3 initial checkpoint SHA mismatch")
    policy, payload = effectiveness.load_candidate_checkpoint(
        checkpoint, expected_sha256=bindings["initial_checkpoint_sha256"]
    )
    trainable: list[str] = []
    for name, parameter in policy.model.named_parameters():
        enabled = name.startswith(TRAINABLE_PREFIXES)
        parameter.requires_grad_(enabled)
        if enabled:
            trainable.append(name)
    _require(bool(trainable), "V3 trainable parameter scope is empty")
    policy.optimizer = torch.optim.Adam(
        [parameter for parameter in policy.model.parameters() if parameter.requires_grad],
        lr=float(optimizer["learning_rate"]),
    )
    _require(len(policy.optimizer.state) == 0, "V3 optimizer was not fresh")
    return policy, payload, tuple(trainable)


def _restored_policy(checkpoint: Path, config: Any, p: Mapping[str, Any]):
    policy = TemperatureCalibratedGateResidualAdapterStudentPPOV42(
        88,
        temperature=CALIBRATION_TEMPERATURE,
        learning_rate=float(p["optimizer"]["learning_rate"]),
    )
    for name, parameter in policy.model.named_parameters():
        parameter.requires_grad_(name.startswith(TRAINABLE_PREFIXES))
    policy.optimizer = torch.optim.Adam(
        [parameter for parameter in policy.model.parameters() if parameter.requires_grad],
        lr=float(p["optimizer"]["learning_rate"]),
    )
    load_checkpoint(checkpoint, policy, config, load_optimizer=True)
    return policy


def stage_initialization_contract(stage: str, p: Mapping[str, Any]) -> dict[str, Any]:
    _require(stage in {"entry", "formal"}, f"unknown V3 stage: {stage}")
    return {
        "checkpoint_sha256": p["bindings"]["initial_checkpoint_sha256"],
        "fresh_optimizer": True,
        "entry_updates_discarded": stage == "entry",
        "formal_initialization_is_original": stage == "formal",
    }


def _scenario_horizon(config: Any, name: str, *, stage: str, index: int,
                      p: Mapping[str, Any]) -> tuple[Any, int]:
    if stage == "entry":
        horizon = int(p["entry"]["steps"][index])
        return replace(config, max_simulation_seconds=float(horizon) * config.dt), horizon
    expected_seconds = float(p["scenarios"][name]["full_horizon_seconds"])
    _require(float(config.max_simulation_seconds) == expected_seconds,
             f"{name} did not use its full frozen horizon")
    return config, int(expected_seconds / config.dt)


def _action_tape_path(result: Mapping[str, Any]) -> str:
    summary = result.get("summary")
    path = summary.get("action_tape") if isinstance(summary, Mapping) else None
    if not isinstance(path, str) or not path:
        path = result.get("action_tape")
    if not isinstance(path, str) or not path:
        raise V3InputError("evaluation result has no action tape")
    return path


def _no_failed_outcomes(summary: Mapping[str, Any]) -> bool:
    values = (summary.get("outcomes") or {}).values()
    return all(str(getattr(value, "value", value)) == "completed" for value in values)


def _rollout_gates(policy: Any, config: Any, steps: list[Any],
                   summary: Mapping[str, Any], p: Mapping[str, Any], *,
                   formal: bool, scenario: str):
    del formal, scenario
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
    gates = {
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
        and float(geometry["max_inner_excursion_m"]) <= float(hard["geometry_inner_excursion_max_m"]),
        "no_failed_outcomes": _no_failed_outcomes(summary),
        "urgency_reward_config_exact": _urgency_reward_config_exact(config, p),
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
        "completion_gate": False,
        "urgency_reward_config_exact": gates["urgency_reward_config_exact"],
    }
    return gates, diagnostics, shortlist, completion


def _coverage(attributes: list[dict[str, Any]], p: Mapping[str, Any]) -> dict[str, Any]:
    if not attributes:
        return {
            "priorities_observed": [], "priority_coverage": False, "soc_count": 0,
            "soc_min": None, "soc_max": None, "soc_in_bounds": False,
            "soc_range_observed": False,
        }
    priorities: list[int] = []
    soc: list[float] = []
    for attribute in attributes:
        _require(isinstance(attribute, Mapping), "formal lifecycle attribute is invalid")
        rows = attribute.get("rows")
        _require(isinstance(rows, list) and rows, "formal lifecycle rows are missing")
        _require(attribute.get("replay_exact") is True, "formal lifecycle replay is not exact")
        for row in rows:
            _require(isinstance(row, Mapping), "formal lifecycle row is invalid")
            priority = row.get("base_priority")
            initial_soc = row.get("initial_soc")
            _require(isinstance(priority, (int, float)) and not isinstance(priority, bool),
                     "formal priority is invalid")
            _require(isinstance(initial_soc, (int, float)) and not isinstance(initial_soc, bool)
                     and np.isfinite(float(initial_soc)), "formal SOC is invalid")
            priorities.append(int(priority)); soc.append(float(initial_soc))
    required = [int(item) for item in p["hard_gates"]["formal_priority_coverage_required"]]
    lower = float(p["attributes"]["initial_soc_min"]); upper = float(p["attributes"]["initial_soc_max"])
    observed_min = min(soc); observed_max = max(soc)
    return {
        "priorities_observed": sorted(set(priorities)),
        "priority_coverage": sorted(set(priorities)) == required,
        "soc_count": len(soc), "soc_min": observed_min, "soc_max": observed_max,
        "soc_in_bounds": all(lower <= value <= upper for value in soc),
        "soc_range_observed": observed_min <= float(p["hard_gates"]["formal_observed_soc_min_at_most"])
        and observed_max >= float(p["hard_gates"]["formal_observed_soc_max_at_least"]),
    }


def _transactional_update(policy: Any, steps: list[Any], config: Any,
                          tape_path: str, p: Mapping[str, Any]):
    optimizer = p["optimizer"]
    model_snapshot = copy.deepcopy(policy.model.state_dict())
    optimizer_snapshot = copy.deepcopy(policy.optimizer.state_dict())
    keys = ("minibatch_size", "gamma", "gae_lambda", "clip_ratio", "value_coef", "entropy_coef")
    try:
        update = policy.update(
            steps, training_authorized=True, epochs=int(optimizer["update_epochs"]),
            target_kl=float(optimizer["target_kl"]),
            **{key: optimizer[key] for key in keys},
        )
        gates = make_update_gates(update, policy.model.state_dict(), policy.optimizer.state_dict(), optimizer)
        gates["frozen_proposal_subgraph"] = _frozen_state_equal(model_snapshot, policy.model.state_dict())
        gates["runtime_shortlist_stability"] = _policy_tape_shortlists_exact(policy, config, tape_path)
    except (KeyError, TypeError, ValueError, RuntimeError, FloatingPointError) as exc:
        policy.model.load_state_dict(model_snapshot); policy.optimizer.load_state_dict(optimizer_snapshot)
        return {"exception": type(exc).__name__, "message": str(exc)}, {"update_transaction": False}
    if not all(bool(value) for value in gates.values()):
        policy.model.load_state_dict(model_snapshot); policy.optimizer.load_state_dict(optimizer_snapshot)
    return update, {key: bool(value) for key, value in gates.items()}


def _training_stage(*, stage: str, output: Path, checkpoint: Path,
                    p: Mapping[str, Any]) -> dict[str, Any]:
    formal = stage == "formal"
    section = p["formal"] if formal else p["entry"]
    stage_initialization_contract(stage, p)
    policy, _, trainable_parameters = _fresh_policy(checkpoint, p)
    output.mkdir(parents=True, exist_ok=False)
    np.random.seed(int(section["seeds"][0]))
    records: list[dict[str, Any]] = []
    attributes: list[dict[str, Any]] = []
    for index, seed_value in enumerate(section["seeds"]):
        seed = int(seed_value); scenario = str(section["scenarios"][index])
        config, horizon = _scenario_horizon(
            scenario_config(scenario, p), scenario, stage=stage, index=index, p=p,
        )
        steps, summary = rollout(
            policy, config, seed=seed, max_steps=horizon,
            output=output / f"seed{seed}", deterministic=False, collect_labels=False,
        )
        tape_path = _action_tape_path({"summary": summary})
        gates, diagnostics, shortlist, completion = _rollout_gates(
            policy, config, steps, summary, p, formal=formal, scenario=scenario,
        )
        row_attributes = None
        if formal:
            row_attributes = pilot_v2.lifecycle_metrics(
                config, tape_path,
                low_soc_threshold=float(p["attributes"]["low_soc_analysis_threshold"]),
            )
            attributes.append(row_attributes)
            gates["attribute_replay"] = bool(row_attributes["replay_exact"])
        update = None; update_gates: dict[str, bool] | None = None; saved: Path | None = None
        # Completion is a report-only performance diagnostic.  It never gates
        # whether PPO is attempted or whether a formal episode proceeds.
        if all(bool(value) for value in gates.values()):
            update, update_gates = _transactional_update(policy, steps, config, tape_path, p)
            gates.update(update_gates)
        if all(bool(value) for value in gates.values()) and formal:
            saved = output / f"checkpoint_{index + 1}.pt"
            try:
                save_checkpoint(
                    saved, policy, config, stage=STAGE, formal=True, ppo_called=True,
                    protocol_sha256=PROTOCOL_SHA, source_hashes=source_hashes(),
                    initial_checkpoint_sha256=p["bindings"]["initial_checkpoint_sha256"],
                    checkpoint_promotable=False,
                    urgency_reward_config=dict(REWARD_CONFIG),
                )
                restored = _restored_policy(saved, config, p)
                gates["checkpoint_restore"] = state_equal(
                    policy.model.state_dict(), restored.model.state_dict(),
                ) and state_equal(policy.optimizer.state_dict(), restored.optimizer.state_dict())
                if gates["checkpoint_restore"]:
                    policy = restored
                else:
                    saved.unlink(missing_ok=True); saved = None
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                gates["checkpoint_restore"] = False
                saved.unlink(missing_ok=True); saved = None
        record = {
            "seed": seed, "scenario": scenario, "horizon_steps": horizon,
            "summary": summary, "completion_fraction": completion,
            "completion_report_only": True, "diagnostics": diagnostics,
            "runtime_shortlist_checks": shortlist, "attributes": row_attributes,
            "update": update, "update_gates": update_gates,
            "checkpoint": None if saved is None else str(saved), "gates": gates,
        }
        records.append(record)
        _write(output / "progress.json", {
            "stage": stage, "completed": len(records), "total": len(section["seeds"]),
            "entry_updates_discarded": not formal,
            "formal_starts_original_checkpoint": formal,
            "completion_report_only": True, "records": records,
        })
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
    final_checkpoint = Path(records[-1]["checkpoint"]) if formal and records and records[-1].get("checkpoint") else None
    result = {
        "stage": "P3_V3_FORMAL" if formal else "P3_V3_ENTRY",
        "PASS": passed, "gates": stage_gates, "records": records, "coverage": coverage,
        "entry_updates_discarded": not formal, "formal_starts_original_checkpoint": formal,
        "formal_training_called": formal, "checkpoint_promotable": False,
        "P3_QUALIFIED": False, "P4_ALLOWED": False, "completion_report_only": True,
        "final_checkpoint": None if final_checkpoint is None else str(final_checkpoint),
        "final_checkpoint_sha256": None if final_checkpoint is None else _sha(final_checkpoint),
        "trainable_parameters": list(trainable_parameters),
    }
    _write(output / "manifest.json", result)
    return result


def run(*, output: Path, superseded_heldout_manifest_path: Path,
        initial_checkpoint: Path, history_root: Path) -> dict[str, Any]:
    p = protocol(); bindings = p["bindings"]
    validate_failed_heldout_manifest(superseded_heldout_manifest_path, p)
    _require(_sha(SCHEDULER_SOURCE) == SCHEDULER_SOURCE_SHA, "scheduler source changed")
    source_hashes()
    _require(_sha(initial_checkpoint) == bindings["initial_checkpoint_sha256"],
             "V3 initialization checkpoint SHA mismatch")
    _require(_sha(initial_checkpoint) != bindings["forbidden_v2_trained_checkpoint_sha256"],
             "V3 must not initialize from the failed V2 trained checkpoint")
    output.mkdir(parents=True, exist_ok=False)
    entry_seeds = [int(seed) for seed in p["entry"]["seeds"]]
    formal_seeds = [int(seed) for seed in p["formal"]["seeds"]]
    heldout_seeds = [int(seed) for seed in p["heldout"]["seeds"]]
    all_reserved = entry_seeds + formal_seeds + heldout_seeds
    _require(len(set(all_reserved)) == len(all_reserved), "V3 fresh seed schedule overlaps itself")
    forbidden = set(int(seed) for seed in p["history"]["forbidden_seeds"])
    _require(not (set(all_reserved) & forbidden), "V3 seed schedule overlaps preregistered history")
    history = historical_registry(history_root, output)
    overlap = sorted(set(all_reserved) & set(history["seeds"]))
    _require(not overlap, f"V3 seeds were previously consumed: {overlap}")
    marker = initial_checkpoint.parent / str(p["reservation"]["marker"])
    reserve_once(marker, {
        "stage": STAGE, "protocol_sha256": PROTOCOL_SHA,
        "superseded_heldout_manifest_sha256": bindings["superseded_heldout_manifest_sha256"],
        "initial_checkpoint_sha256": bindings["initial_checkpoint_sha256"],
        "forbidden_v2_trained_checkpoint_sha256": bindings["forbidden_v2_trained_checkpoint_sha256"],
        "entry_seeds": entry_seeds, "formal_seeds": formal_seeds,
        "heldout_seeds": heldout_seeds, "heldout_untouched": True,
        "output": str(output), "reserved_before_first_reset": True,
        "urgency_reward_config": dict(REWARD_CONFIG),
    })
    _write(output / "pre_reset_registry.json", {
        "stage": STAGE, "protocol_sha256": PROTOCOL_SHA,
        "entry_seeds": entry_seeds, "formal_seeds": formal_seeds,
        "heldout_seeds": heldout_seeds, "reservation_marker": str(marker),
        "heldout_untouched": True, "reserved_before_first_reset": True,
        "initial_checkpoint_sha256": bindings["initial_checkpoint_sha256"],
        "urgency_reward_config": dict(REWARD_CONFIG),
    })
    entry = _training_stage(stage="entry", output=output / "entry", checkpoint=initial_checkpoint, p=p)
    formal = None
    if entry["PASS"]:
        formal = _training_stage(stage="formal", output=output / "formal", checkpoint=initial_checkpoint, p=p)
    passed = formal is not None and bool(formal["PASS"])
    result = {
        "stage": STAGE, "PASS": bool(passed),
        "entry": {"PASS": bool(entry["PASS"]), "gates": entry["gates"],
                   "entry_updates_discarded": True,
                   "formal_initialization_checkpoint_sha256": bindings["initial_checkpoint_sha256"]},
        "formal": None if formal is None else {
            "PASS": bool(formal["PASS"]), "gates": formal["gates"],
            "coverage": formal["coverage"], "final_checkpoint": formal["final_checkpoint"],
            "final_checkpoint_sha256": formal["final_checkpoint_sha256"],
            "formal_initialization_checkpoint_sha256": bindings["initial_checkpoint_sha256"],
        },
        "entry_updates_discarded": True, "formal_starts_original_checkpoint": True,
        "PPO_called": True, "formal_training_called": formal is not None,
        "heldout_untouched": True, "heldout_seeds_reserved": heldout_seeds,
        "HELDOUT_EVALUATION_ALLOWED": bool(passed),
        "formal_pass_authorizes_heldout_only": True,
        "P3_QUALIFIED": False, "P4_ALLOWED": False, "checkpoint_promotable": False,
        "completion_report_only": True, "protocol_sha256": PROTOCOL_SHA,
        "source_hashes": source_hashes(), "superseded_heldout_manifest_sha256": bindings["superseded_heldout_manifest_sha256"],
        "superseded_heldout_protocol_sha256": bindings["superseded_heldout_protocol_sha256"],
        "forbidden_v2_trained_checkpoint_sha256": bindings["forbidden_v2_trained_checkpoint_sha256"],
        "initial_checkpoint_sha256": bindings["initial_checkpoint_sha256"],
        "scheduler_source": bindings["scheduler_source"], "scheduler_source_sha256": SCHEDULER_SOURCE_SHA,
        "urgency_reward_config": dict(REWARD_CONFIG), "reservation_marker": str(marker),
    }
    _write(output / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--superseded-heldout-manifest", "--heldout-manifest",
                        dest="superseded_heldout_manifest_path", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--history-root", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = run(
        output=args.output,
        superseded_heldout_manifest_path=args.superseded_heldout_manifest_path,
        initial_checkpoint=args.initial_checkpoint,
        history_root=args.history_root,
    )
    print(json.dumps({
        "stage": result["stage"], "PASS": result["PASS"],
        "HELDOUT_EVALUATION_ALLOWED": result["HELDOUT_EVALUATION_ALLOWED"],
        "P3_QUALIFIED": result["P3_QUALIFIED"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
