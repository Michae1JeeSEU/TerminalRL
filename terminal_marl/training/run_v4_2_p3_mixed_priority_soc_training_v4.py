"""P3 V4 dense-completion-guarded PPO training protocol.

This runner is deliberately a fresh protocol.  It keeps the V3 integrity
gates, starts from the original K=3 checkpoint with a fresh optimizer, and
uses deterministic dense development episodes as the first-checkpoint
selection guard.  No held-out episode is evaluated by this module.
"""
from __future__ import annotations

import argparse
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
    run_v4_2_p3_mixed_priority_soc_training_v3 as v3,
)
from terminal_marl.training.evaluate_v4_2_p2_gate_runtime_shortlist_v1 import (
    _tape_shortlist_checks,
)
from terminal_marl.training.run_v4_2_ppo_block2 import replay_geometry, state_equal
from terminal_marl.training.run_v4_2_v2_1_formal_i0_r4 import historical_registry
from terminal_marl.training.run_v4_2_v2_1_i1_v2 import reserve_once
from terminal_marl.training.student_rollout_v4_2 import rollout


STAGE = "V4_2_P3_MIXED_PRIORITY_SOC_TRAINING_V4"
PROTOCOL = Path(
    "terminal_marl/configs/low_v4_2/p3_mixed_priority_soc_training_v4.yaml"
)
RUNNER = Path("terminal_marl/training/run_v4_2_p3_mixed_priority_soc_training_v4.py")
PROTOCOL_SHA = "50fac4e44171239e3e09458987808c4a6567cfc36178181b7906e8c18ad19a54"
STATUS = "PREREGISTERED_DENSE_COMPLETION_GUARDED_TRAINING"
INITIAL_CHECKPOINT_SHA = (
    "4b8be66c100aaaeb5b5699dcce68a96dbe1237964e4c308e7b23b95c62fb8c8c"
)
V3_ROOT_MANIFEST_SHA = (
    "38703fa2af89a56f2ac46551f3c6006bf750547a7768f707990c7873e8209b76"
)
V3_PROTOCOL_SHA = (
    "c69dc4b308e97652936c8da5bb6a47ef61a73a4f3cb599de8f39cb703f3c4d25"
)
V3_FINAL_CHECKPOINT_SHA = (
    "0fb4acf9b05ba2fa8f5700e64eca6c2b9fc6174bb091c4e12ee4c1ea3942fb6b"
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
REWARD_CONFIG = {
    "low_level_priority_time_weight": 1.0,
    "low_level_soc_urgency_time_weight": 2.0,
    "low_level_soc_urgency_threshold": 0.65,
}


class V4InputError(ValueError):
    """Raised when a frozen V4 input violates its contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V4InputError(message)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(path: str | Path) -> str:
    return _sha256(path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if hasattr(value, "value") and not isinstance(value, (str, bytes)):
        return _jsonable(value.value)
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"JSON input missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def protocol() -> dict[str, Any]:
    """Read and validate the byte-frozen V4 protocol."""
    _require(PROTOCOL.is_file(), f"V4 protocol missing: {PROTOCOL}")
    _require(PROTOCOL_SHA != "TO_BE_FROZEN" and _sha(PROTOCOL) == PROTOCOL_SHA,
             "P3 V4 protocol changed after freeze")
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), "P3 V4 protocol must be a mapping")
    _require(value.get("protocol") == STAGE, "P3 V4 protocol id changed")
    _require(value.get("status") == STATUS, "P3 V4 protocol status changed")
    bindings = value.get("bindings")
    _require(isinstance(bindings, Mapping), "P3 V4 bindings are missing")
    expected = {
        "superseded_training_stage": v3.STAGE,
        "superseded_training_manifest_sha256": V3_ROOT_MANIFEST_SHA,
        "superseded_training_protocol_sha256": V3_PROTOCOL_SHA,
        "superseded_training_pass": True,
        "superseded_training_heldout_untouched": True,
        "initial_checkpoint_sha256": INITIAL_CHECKPOINT_SHA,
        "selected_checkpoint_must_be_fresh_v4": True,
        "forbidden_v3_checkpoint_sha256": V3_FINAL_CHECKPOINT_SHA,
        "scheduler_source": SCHEDULER_SOURCE.as_posix(),
        "scheduler_source_sha256": SCHEDULER_SOURCE_SHA,
    }
    for key, expected_value in expected.items():
        _require(bindings.get(key) == expected_value, f"V4 binding changed: {key}")
    _require(bindings.get("reward_sources") == REWARD_SOURCE_HASHES,
             "V4 reward-source binding changed")
    reward = value.get("reward")
    _require(isinstance(reward, Mapping), "V4 reward section is missing")
    for key, expected_value in REWARD_CONFIG.items():
        _require(float(reward.get(key)) == expected_value,
                 f"V4 reward config changed: {key}")
    _require(reward.get("urgency_reward_config_exact") is True,
             "V4 urgency reward gate binding changed")
    completion = value.get("completion_reporting")
    _require(isinstance(completion, Mapping), "V4 completion section is missing")
    _require(completion.get("training_mode") == "report_only",
             "V4 training completion mode changed")
    _require(completion.get("formal_rollout_completion_gate") is False,
             "V4 formal completion gate changed")
    _require(completion.get("dev_mode") == "hard_gate",
             "V4 dev completion mode changed")
    entry = value.get("entry")
    formal = value.get("formal")
    dev = value.get("dev")
    heldout = value.get("heldout")
    _require(entry == {
        "seeds": [827001, 827002], "scenarios": ["dense24", "flow6"],
        "steps": [320, 200], "stochastic": True, "updates_discarded": True,
        "checkpoints_formal_initialization": "forbidden",
    }, "V4 entry schedule changed")
    _require(formal.get("seeds") == list(range(827101, 827107)),
             "V4 formal seeds changed")
    _require(formal.get("scenarios") == [
        "dense24", "flow6", "dense24", "flow12", "dense24", "dense24",
    ], "V4 formal scenarios changed")
    for key, expected_value in {
        "stochastic": True, "full_scenario_horizon": True,
        "dense_horizon_seconds": 1800, "sparse_uses_full_scenario_horizon": True,
        "sequential_updates": True, "max_updates": 6,
        "stop_on_hard_gate_failure": True, "completion_is_report_only": True,
        "completion_failure_continues_schedule": True,
    }.items():
        _require(formal.get(key) == expected_value, f"V4 formal field changed: {key}")
    _require(dev == {
        "seeds": [828001, 828002], "scenarios": ["dense24", "dense24"],
        "deterministic": True, "full_horizon_seconds": 1800,
        "controllers": ["initial", "candidate"],
        "initial_baseline_before_formal_update": True,
        "evaluate_after_every_formal_update": True,
        "required_completion_fraction": 1.0, "safety_all_zero": True,
        "exact_replay": True, "no_teacher": True, "no_labels": True,
        "no_updates": True, "first_passing_checkpoint_only": True,
        "priority_weighted_wait_strictly_better_than_initial": True,
        "low_soc_wait_nonworse_than_initial": True,
    }, "V4 development schedule changed")
    _require(heldout == {
        "seeds": [829001, 829002, 829003, 829004, 829005, 829006],
        "scenarios": ["flow6", "flow12", "dense24", "flow6", "flow12", "dense24"],
        "untouched": True, "reserve_before_first_reset": True,
    }, "V4 held-out schedule changed")
    _require(value.get("authorization", {}).get("P3_QUALIFIED") is False,
             "V4 P3 authorization changed")
    _require(value.get("authorization", {}).get("P4_ALLOWED") is False,
             "V4 P4 authorization changed")
    _require(value.get("authorization", {}).get("checkpoint_promotable") is False,
             "V4 promotion authorization changed")
    _require(value.get("reservation", {}).get("marker") ==
             "p3_mixed_priority_soc_training_v4_consumed.json",
             "V4 reservation marker changed")
    return value


def source_hashes() -> dict[str, str]:
    """Hash all frozen implementation inputs used by a V4 checkpoint."""
    inherited = v3.source_hashes()
    paths = tuple(Path(path) for path in inherited) + (PROTOCOL, RUNNER)
    hashes = {path.as_posix(): _sha(path) for path in paths}
    for path, expected in REWARD_SOURCE_HASHES.items():
        _require(hashes.get(path) == expected, f"reward source changed: {path}")
    _require(hashes[SCHEDULER_SOURCE.as_posix()] == SCHEDULER_SOURCE_SHA,
             "scheduler source changed")
    return hashes


def validate_superseded_v3_manifest(path: Path, p: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact successful V3 root before any V4 reset."""
    bindings = p["bindings"]
    _require(_sha(path) == bindings["superseded_training_manifest_sha256"],
             "superseded V3 root manifest SHA mismatch")
    report = _read_json(path)
    expected = {
        "stage": v3.STAGE, "PASS": True,
        "HELDOUT_EVALUATION_ALLOWED": True, "heldout_untouched": True,
        "P3_QUALIFIED": False, "P4_ALLOWED": False,
        "checkpoint_promotable": False,
        "protocol_sha256": V3_PROTOCOL_SHA,
        "initial_checkpoint_sha256": INITIAL_CHECKPOINT_SHA,
    }
    for key, expected_value in expected.items():
        _require(report.get(key) == expected_value,
                 f"superseded V3 root field changed: {key}")
    formal = report.get("formal")
    _require(isinstance(formal, Mapping) and formal.get("PASS") is True,
             "superseded V3 formal PASS missing")
    _require(formal.get("final_checkpoint_sha256") == V3_FINAL_CHECKPOINT_SHA,
             "superseded V3 final checkpoint changed")
    return report


def scenario_config(name: str, p: Mapping[str, Any]):
    """Build the V3 scenario and change only dense24's horizon to 1800 s."""
    config = v3.scenario_config(name, p)
    expected = float(p["scenarios"][name]["full_horizon_seconds"])
    if name == "dense24":
        config = replace(config, max_simulation_seconds=expected)
    _require(float(config.max_simulation_seconds) == expected,
             f"{name} scenario horizon changed")
    for key, expected_value in REWARD_CONFIG.items():
        _require(float(getattr(config, key)) == expected_value,
                 f"{name} reward config changed: {key}")
    return config


def _urgency_reward_config_exact(config: Any, p: Mapping[str, Any]) -> bool:
    return all(
        float(getattr(config, key)) == float(p["reward"][key])
        for key in REWARD_CONFIG
    )


def _fresh_policy(checkpoint: Path, p: Mapping[str, Any]):
    _require(_sha(checkpoint) == p["bindings"]["initial_checkpoint_sha256"],
             "V4 initialization checkpoint SHA mismatch")
    policy, payload, trainable = v3._fresh_policy(checkpoint, p)
    _require(_sha(checkpoint) != p["bindings"]["forbidden_v3_checkpoint_sha256"],
             "V4 must not initialize from the V3 trained checkpoint")
    return policy, payload, trainable


def _restored_policy(checkpoint: Path, config: Any, p: Mapping[str, Any]):
    return v3._restored_policy(checkpoint, config, p)


def stage_initialization_contract(stage: str, p: Mapping[str, Any]) -> dict[str, Any]:
    _require(stage in {"entry", "formal"}, f"unknown V4 training stage: {stage}")
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
    expected = float(p["scenarios"][name]["full_horizon_seconds"])
    _require(float(config.max_simulation_seconds) == expected,
             f"{name} did not use its full frozen horizon")
    return config, int(round(expected / config.dt))


def _action_tape_path(summary: Mapping[str, Any]) -> str:
    path = summary.get("action_tape")
    if not isinstance(path, str) or not path:
        raise V4InputError("evaluation result has no action tape")
    return path


def _no_failed_outcomes(summary: Mapping[str, Any]) -> bool:
    return all(
        str(getattr(value, "value", value)) == "completed"
        for value in (summary.get("outcomes") or {}).values()
    )


def _rollout_gates(policy: Any, config: Any, steps: list[Any],
                   summary: Mapping[str, Any], p: Mapping[str, Any], *,
                   deterministic: bool, scenario: str) -> tuple[dict[str, bool], dict[str, Any], dict[str, Any], float]:
    del scenario
    tape_path = _action_tape_path(summary)
    replay_exact, geometry = replay_geometry(config, tape_path)
    likelihood = v3._policy_tape_logprob_exact(policy, config, tape_path) if hasattr(v3, "_policy_tape_logprob_exact") else True
    if not hasattr(v3, "_policy_tape_logprob_exact"):
        from terminal_marl.training.run_v4_2_i2_dagger_preflight import _policy_tape_logprob_exact
        likelihood = _policy_tape_logprob_exact(policy, config, tape_path)
    initial_error = v3.requested_logprob_error(policy, steps) if hasattr(v3, "requested_logprob_error") else 0.0
    if not hasattr(v3, "requested_logprob_error"):
        from terminal_marl.training.run_v4_2_v2_1_ppo import requested_logprob_error
        initial_error = requested_logprob_error(policy, steps)
    shortlist = _tape_shortlist_checks(tape_path, expected_k=3)
    counts = summary.get("counts") or {}
    actor = int(counts.get("actor_decisions", 0))
    shield_actor = int(counts.get("shield_actor", 0))
    hard = p["hard_gates"]
    gates = {
        "safety": not any(bool(value) for value in (summary.get("safety") or {}).values()),
        "replay": bool(replay_exact),
        "requested_likelihood": bool(likelihood),
        "initial_ratio": initial_error <= float(hard["initial_ratio_error_max"]),
        "student_only": int(summary.get("label_rows", 0)) == 0
        and all(not bool(step.teacher_used) for step in steps),
        "stochastic": bool(steps) and all(
            bool(step.stochastic_behavior) is (not deterministic) for step in steps
        ),
        "actor_decisions": actor > 0,
        "runtime_shortlist_contract": bool(shortlist.get("all")),
        "shield_burden": shield_actor / max(1, actor) <= float(hard["shield_actor_fraction_max"]),
        "geometry": int(geometry["reverse_samples"]) <= int(hard["geometry_reverse_samples_max"])
        and float(geometry["max_inner_excursion_m"]) <= float(hard["geometry_inner_excursion_max_m"]),
        "no_failed_outcomes": _no_failed_outcomes(summary),
        "urgency_reward_config_exact": _urgency_reward_config_exact(config, p),
        "no_updates": True,
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
        "actor_decisions": actor,
        "shield_actor_fraction": shield_actor / max(1, actor),
        "completion_fraction": completion,
        "completion_gate": False,
        "urgency_reward_config_exact": gates["urgency_reward_config_exact"],
    }
    return gates, diagnostics, shortlist, completion


def _training_update(policy: Any, steps: list[Any], config: Any,
                     tape_path: str, p: Mapping[str, Any]):
    return v3._transactional_update(policy, steps, config, tape_path, p)


def _coverage(attributes: list[dict[str, Any]], p: Mapping[str, Any]) -> dict[str, Any]:
    return v3._coverage(attributes, p)


def _finite_metric(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool) and np.isfinite(float(value))


def _lifecycle_aggregate(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[Mapping[str, Any]] = []
    for record in records:
        lifecycle = record.get("lifecycle")
        _require(isinstance(lifecycle, Mapping), "development lifecycle metrics missing")
        _require(lifecycle.get("replay_exact") is True, "development lifecycle replay is not exact")
        record_rows = lifecycle.get("rows")
        _require(isinstance(record_rows, list), "development lifecycle rows missing")
        rows.extend(record_rows)
    _require(rows, "development lifecycle has no completed rows")
    waits: list[float] = []
    weights: list[float] = []
    low_waits: list[float] = []
    threshold = 0.65
    for row in rows:
        _require(isinstance(row, Mapping), "development lifecycle row invalid")
        priority = row.get("base_priority")
        soc = row.get("initial_soc")
        wait = row.get("wait_seconds")
        _require(_finite_metric(priority) and _finite_metric(soc) and _finite_metric(wait),
                 "development lifecycle metric is missing or nonfinite")
        waits.append(float(wait)); weights.append(float(priority))
        if float(soc) <= threshold:
            low_waits.append(float(wait))
    return {
        "completed": len(rows),
        "requests": sum(int(record.get("requests", 0)) for record in records),
        "completion_fraction": float(sum(int(record.get("completed", 0)) for record in records)) / max(
            1, sum(int(record.get("requests", 0)) for record in records)
        ),
        "priority_weighted_wait_seconds": float(np.average(waits, weights=weights)),
        "low_soc_mean_wait_seconds": None if not low_waits else float(np.mean(low_waits)),
        "low_soc_count": len(low_waits),
        "rows": rows,
    }


def _dev_selection_gates(initial: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, bool]:
    """Pure first-pass development decision; missing metrics fail closed."""
    fields = ("completion_fraction", "priority_weighted_wait_seconds")
    metrics_defined = (
        all(_finite_metric(initial.get(key)) for key in fields)
        and _finite_metric(initial.get("low_soc_mean_wait_seconds"))
        and all(_finite_metric(candidate.get(key)) for key in fields)
        and _finite_metric(candidate.get("low_soc_mean_wait_seconds"))
    )
    if not metrics_defined:
        return {
            "metrics_defined": False,
            "initial_complete": False,
            "candidate_complete": False,
            "priority_wait_strictly_better": False,
            "low_soc_wait_nonworse": False,
        }
    return {
        "metrics_defined": True,
        "initial_complete": float(initial["completion_fraction"]) >= 1.0,
        "candidate_complete": float(candidate["completion_fraction"]) >= 1.0,
        "priority_wait_strictly_better": float(candidate["priority_weighted_wait_seconds"]) < float(initial["priority_weighted_wait_seconds"]),
        "low_soc_wait_nonworse": float(candidate["low_soc_mean_wait_seconds"]) <= float(initial["low_soc_mean_wait_seconds"]),
    }


def _dev_pass(initial: Mapping[str, Any], candidate: Mapping[str, Any]) -> tuple[bool, dict[str, bool]]:
    gates = _dev_selection_gates(initial, candidate)
    return bool(all(gates.values())), gates


def _dev_episode(policy: Any, config: Any, seed: int, output: Path,
                 p: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    steps, summary = rollout(
        policy, config, seed=int(seed), max_steps=int(round(config.max_simulation_seconds / config.dt)),
        output=output, deterministic=True, collect_labels=False,
    )
    gates, diagnostics, shortlist, completion = _rollout_gates(
        policy, config, steps, summary, p, deterministic=True, scenario="dense24",
    )
    tape_path = _action_tape_path(summary)
    lifecycle = pilot_v2.lifecycle_metrics(
        config, tape_path,
        low_soc_threshold=float(p["attributes"]["low_soc_analysis_threshold"]),
    )
    gates.update({
        "completion": completion >= float(p["dev"]["required_completion_fraction"]),
        "lifecycle_replay": bool(lifecycle.get("replay_exact")),
        "metrics_defined": lifecycle.get("low_soc", {}).get("mean_wait_seconds") is not None,
    })
    hard_fail = any(not bool(value) for key, value in gates.items() if key != "completion")
    return {
        "label": label, "seed": int(seed), "scenario": "dense24",
        "horizon_steps": int(round(config.max_simulation_seconds / config.dt)),
        "requests": int(config.total_requested_uavs),
        "summary": summary, "completion_fraction": completion,
        "lifecycle": lifecycle, "gates": gates,
        "diagnostics": diagnostics, "runtime_shortlist_checks": shortlist,
        "fatal_hard_failure": bool(hard_fail),
        "PASS": bool(not hard_fail and gates["completion"]),
        "no_updates": True, "no_teacher": True, "no_labels": True,
    }


def _evaluate_dev(policy: Any, config: Any, p: Mapping[str, Any], output: Path,
                  *, label: str) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    for index, seed in enumerate(p["dev"]["seeds"]):
        record = _dev_episode(
            policy, config, int(seed), output / f"seed{int(seed)}", p, label=label,
        )
        records.append(record)
        if record["fatal_hard_failure"]:
            break
    try:
        aggregate = _lifecycle_aggregate(records) if records and all(
            isinstance(record.get("lifecycle"), Mapping) for record in records
        ) else None
    except (V4InputError, KeyError, TypeError, ValueError):
        aggregate = None
    expected_count = len(p["dev"]["seeds"])
    complete_records = len(records) == expected_count
    if aggregate is None:
        passed = False
        selection_metrics = None
    else:
        selection_metrics = aggregate
        passed = complete_records and all(bool(record["PASS"]) for record in records)
    return {
        "label": label, "records": records, "aggregate": aggregate,
        "complete_records": complete_records, "PASS": bool(passed),
        "fatal_hard_failure": any(bool(record["fatal_hard_failure"]) for record in records),
        "performance_failure_only": bool(complete_records and not passed and not any(
            bool(record["fatal_hard_failure"]) for record in records
        )),
        "selection_metrics": selection_metrics,
    }


def _entry_stage(*, output: Path, checkpoint: Path, p: Mapping[str, Any]) -> dict[str, Any]:
    stage_initialization_contract("entry", p)
    policy, _, trainable = _fresh_policy(checkpoint, p)
    output.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    for index, seed in enumerate(p["entry"]["seeds"]):
        scenario = str(p["entry"]["scenarios"][index])
        config, horizon = _scenario_horizon(
            scenario_config(scenario, p), scenario, stage="entry", index=index, p=p,
        )
        steps, summary = rollout(
            policy, config, seed=int(seed), max_steps=horizon,
            output=output / f"seed{int(seed)}", deterministic=False,
            collect_labels=False,
        )
        gates, diagnostics, shortlist, completion = _rollout_gates(
            policy, config, steps, summary, p, deterministic=False, scenario=scenario,
        )
        update = None
        update_gates = None
        if all(bool(value) for value in gates.values()):
            tape_path = _action_tape_path(summary)
            update, update_gates = _training_update(policy, steps, config, tape_path, p)
            gates.update(update_gates)
        record = {
            "seed": int(seed), "scenario": scenario, "horizon_steps": horizon,
            "summary": summary, "completion_fraction": completion,
            "completion_report_only": True, "diagnostics": diagnostics,
            "runtime_shortlist_checks": shortlist, "gates": gates,
            "update": update, "update_gates": update_gates,
        }
        records.append(record)
        _write(output / "progress.json", {
            "stage": "P3_V4_ENTRY", "completed": len(records),
            "total": len(p["entry"]["seeds"]), "records": records,
            "entry_updates_discarded": True, "completion_report_only": True,
        })
        if not all(bool(value) for value in gates.values()):
            break
    stage_gates = {
        "all_episodes": len(records) == len(p["entry"]["seeds"]),
        "all_episode_gates": bool(records) and all(
            all(bool(value) for value in row["gates"].values()) for row in records
        ),
    }
    result = {
        "stage": "P3_V4_ENTRY", "PASS": bool(all(stage_gates.values())),
        "gates": stage_gates, "records": records,
        "entry_updates_discarded": True,
        "formal_initialization_checkpoint_sha256": p["bindings"]["initial_checkpoint_sha256"],
        "trainable_parameters": list(trainable), "completion_report_only": True,
    }
    _write(output / "manifest.json", result)
    return result


def _formal_stage(*, output: Path, checkpoint: Path, p: Mapping[str, Any]) -> dict[str, Any]:
    stage_initialization_contract("formal", p)
    policy, _, trainable = _fresh_policy(checkpoint, p)
    output.mkdir(parents=True, exist_ok=False)
    dense = scenario_config("dense24", p)
    baseline_policy, _, _ = _fresh_policy(checkpoint, p)
    baseline = _evaluate_dev(baseline_policy, dense, p, output / "dev_initial", label="initial")
    _write(output / "dev_initial.json", baseline)
    if not baseline["PASS"]:
        result = {
            "stage": "P3_V4_FORMAL", "PASS": False,
            "gates": {"initial_baseline_complete_and_safe": False},
            "records": [], "dev_initial": baseline, "dev_records": [],
            "selected_checkpoint": None, "selected_checkpoint_sha256": None,
            "coverage": None, "trainable_parameters": list(trainable),
            "formal_initialization_checkpoint_sha256": p["bindings"]["initial_checkpoint_sha256"],
        }
        _write(output / "manifest.json", result)
        return result
    initial_metrics = baseline["aggregate"]
    attributes: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    dev_records: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for index, seed in enumerate(p["formal"]["seeds"]):
        scenario = str(p["formal"]["scenarios"][index])
        config, horizon = _scenario_horizon(
            scenario_config(scenario, p), scenario, stage="formal", index=index, p=p,
        )
        steps, summary = rollout(
            policy, config, seed=int(seed), max_steps=horizon,
            output=output / f"seed{int(seed)}", deterministic=False,
            collect_labels=False,
        )
        tape_path = _action_tape_path(summary)
        gates, diagnostics, shortlist, completion = _rollout_gates(
            policy, config, steps, summary, p, deterministic=False, scenario=scenario,
        )
        lifecycle = pilot_v2.lifecycle_metrics(
            config, tape_path,
            low_soc_threshold=float(p["attributes"]["low_soc_analysis_threshold"]),
        )
        attributes.append(lifecycle)
        gates["attribute_replay"] = bool(lifecycle.get("replay_exact"))
        update = None
        update_gates: dict[str, bool] | None = None
        saved: Path | None = None
        if all(bool(value) for value in gates.values()):
            update, update_gates = _training_update(policy, steps, config, tape_path, p)
            gates.update(update_gates)
        if all(bool(value) for value in gates.values()):
            saved = output / f"checkpoint_{index + 1}.pt"
            try:
                save_checkpoint(
                    saved, policy, config, stage=STAGE, formal=True, ppo_called=True,
                    protocol_sha256=PROTOCOL_SHA, source_hashes=source_hashes(),
                    initial_checkpoint_sha256=p["bindings"]["initial_checkpoint_sha256"],
                    checkpoint_promotable=False, urgency_reward_config=dict(REWARD_CONFIG),
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
            "seed": int(seed), "scenario": scenario, "horizon_steps": horizon,
            "summary": summary, "completion_fraction": completion,
            "completion_report_only": True, "diagnostics": diagnostics,
            "runtime_shortlist_checks": shortlist, "attributes": lifecycle,
            "update": update, "update_gates": update_gates,
            "checkpoint": None if saved is None else str(saved),
            "checkpoint_sha256": None if saved is None else _sha(saved),
            "gates": gates,
        }
        records.append(record)
        if not all(bool(value) for value in gates.values()):
            break
        if saved is None:
            break
        dev = _evaluate_dev(
            policy, dense, p, output / f"dev_checkpoint_{index + 1}",
            label=f"checkpoint_{index + 1}",
        )
        dev_records.append(dev)
        dev_metrics = dev.get("aggregate")
        if dev["fatal_hard_failure"]:
            record["dev"] = dev
            break
        if dev["PASS"] and dev_metrics is not None:
            try:
                selection = _dev_selection_gates(initial_metrics, dev_metrics)
            except V4InputError:
                selection = {"initial_complete": False, "candidate_complete": False,
                             "priority_wait_strictly_better": False,
                             "low_soc_wait_nonworse": False}
            dev["selection_gates"] = selection
            if selected is None and all(bool(value) for value in selection.values()):
                selected = {
                    "index": index + 1, "path": str(saved),
                    "sha256": _sha(saved), "dev": dev,
                }
        record["dev"] = dev
        _write(output / "progress.json", {
            "stage": "P3_V4_FORMAL", "completed": len(records),
            "total": len(p["formal"]["seeds"]), "records": records,
            "dev_records": dev_records, "selected_checkpoint": selected,
            "completion_report_only": True,
        })
        if selected is not None:
            break
    coverage = _coverage(attributes, p) if attributes else None
    all_episode_gates = bool(records) and all(
        all(bool(value) for value in record["gates"].values()) for record in records
    )
    selected_dev_pass = selected is not None and bool(selected["dev"]["PASS"])
    gates = {
        "all_episodes_before_selection_hard_gates": all_episode_gates,
        "initial_baseline_complete_and_safe": bool(baseline["PASS"]),
        "selected_checkpoint_exists": selected is not None,
        "selected_dev_pass": selected_dev_pass,
        "priority_coverage": bool(coverage and coverage["priority_coverage"]),
        "soc_in_bounds": bool(coverage and coverage["soc_in_bounds"]),
        "soc_range_observed": bool(coverage and coverage["soc_range_observed"]),
    }
    passed = bool(all(gates.values()))
    result = {
        "stage": "P3_V4_FORMAL", "PASS": passed, "gates": gates,
        "records": records, "dev_initial": baseline, "dev_records": dev_records,
        "selected_checkpoint": None if selected is None else selected["path"],
        "selected_checkpoint_sha256": None if selected is None else selected["sha256"],
        "selected_checkpoint_index": None if selected is None else selected["index"],
        "coverage": coverage, "trainable_parameters": list(trainable),
        "formal_initialization_checkpoint_sha256": p["bindings"]["initial_checkpoint_sha256"],
        "completion_report_only": True,
    }
    _write(output / "manifest.json", result)
    return result


def run(*, output: Path, superseded_v3_manifest_path: Path,
        initial_checkpoint: Path, history_root: Path) -> dict[str, Any]:
    p = protocol()
    _require(not output.exists(), f"V4 output already exists: {output}")
    root = validate_superseded_v3_manifest(superseded_v3_manifest_path, p)
    _require(_sha(initial_checkpoint) == p["bindings"]["initial_checkpoint_sha256"],
             "V4 initial checkpoint SHA mismatch")
    _require(_sha(initial_checkpoint) != p["bindings"]["forbidden_v3_checkpoint_sha256"],
             "V4 must not initialize from V3 checkpoint")
    _require(_sha(SCHEDULER_SOURCE) == SCHEDULER_SOURCE_SHA,
             "scheduler source changed")
    hashes = source_hashes()
    entry_seeds = [int(seed) for seed in p["entry"]["seeds"]]
    formal_seeds = [int(seed) for seed in p["formal"]["seeds"]]
    dev_seeds = [int(seed) for seed in p["dev"]["seeds"]]
    heldout_seeds = [int(seed) for seed in p["heldout"]["seeds"]]
    reserved = entry_seeds + formal_seeds + dev_seeds + heldout_seeds
    _require(len(reserved) == len(set(reserved)), "V4 schedule overlaps itself")
    forbidden = set(int(seed) for seed in p["history"]["forbidden_seeds"])
    _require(not (set(reserved) & forbidden), "V4 schedule overlaps preregistered history")
    output.mkdir(parents=True, exist_ok=False)
    history = historical_registry(history_root, output)
    overlap = sorted(set(reserved) & set(history["seeds"]))
    _require(not overlap, f"V4 seeds were previously consumed: {overlap}")
    marker = initial_checkpoint.parent / str(p["reservation"]["marker"])
    reserve_once(marker, {
        "stage": STAGE, "protocol_sha256": PROTOCOL_SHA,
        "superseded_training_manifest_sha256": p["bindings"]["superseded_training_manifest_sha256"],
        "initial_checkpoint_sha256": p["bindings"]["initial_checkpoint_sha256"],
        "entry_seeds": entry_seeds, "formal_seeds": formal_seeds,
        "dev_seeds": dev_seeds, "heldout_seeds": heldout_seeds,
        "heldout_untouched": True, "output": str(output),
        "reserved_before_first_reset": True,
        "original_checkpoint_initialization": True,
        "urgency_reward_config": dict(REWARD_CONFIG),
    })
    _write(output / "pre_reset_registry.json", {
        "stage": STAGE, "protocol_sha256": PROTOCOL_SHA,
        "entry_seeds": entry_seeds, "formal_seeds": formal_seeds,
        "dev_seeds": dev_seeds, "heldout_seeds": heldout_seeds,
        "history_registry": history, "reservation_marker": str(marker),
        "heldout_untouched": True, "reserved_before_first_reset": True,
        "initial_checkpoint_sha256": p["bindings"]["initial_checkpoint_sha256"],
        "urgency_reward_config": dict(REWARD_CONFIG),
    })
    entry = _entry_stage(output=output / "entry", checkpoint=initial_checkpoint, p=p)
    formal = None
    if entry["PASS"]:
        formal = _formal_stage(output=output / "formal", checkpoint=initial_checkpoint, p=p)
    passed = formal is not None and bool(formal["PASS"])
    result = {
        "stage": STAGE, "PASS": bool(passed),
        "entry": {"PASS": bool(entry["PASS"]), "gates": entry["gates"],
                   "entry_updates_discarded": True,
                   "formal_initialization_checkpoint_sha256": p["bindings"]["initial_checkpoint_sha256"]},
        "formal": None if formal is None else {
            "PASS": bool(formal["PASS"]), "gates": formal["gates"],
            "coverage": formal["coverage"],
            "selected_checkpoint": formal["selected_checkpoint"],
            "selected_checkpoint_sha256": formal["selected_checkpoint_sha256"],
            "selected_checkpoint_index": formal["selected_checkpoint_index"],
            "formal_initialization_checkpoint_sha256": p["bindings"]["initial_checkpoint_sha256"],
        },
        "entry_updates_discarded": True, "formal_starts_original_checkpoint": True,
        "PPO_called": True, "formal_training_called": formal is not None,
        "heldout_untouched": True, "heldout_seeds_reserved": heldout_seeds,
        "HELDOUT_EVALUATION_ALLOWED": bool(passed),
        "formal_pass_authorizes_heldout_only": True,
        "P3_QUALIFIED": False, "P4_ALLOWED": False, "checkpoint_promotable": False,
        "completion_report_only": True, "protocol_sha256": PROTOCOL_SHA,
        "source_hashes": hashes,
        "superseded_training_manifest_sha256": p["bindings"]["superseded_training_manifest_sha256"],
        "superseded_training_protocol_sha256": p["bindings"]["superseded_training_protocol_sha256"],
        "initial_checkpoint_sha256": p["bindings"]["initial_checkpoint_sha256"],
        "scheduler_source": p["bindings"]["scheduler_source"],
        "scheduler_source_sha256": SCHEDULER_SOURCE_SHA,
        "urgency_reward_config": dict(REWARD_CONFIG), "reservation_marker": str(marker),
        "superseded_v3_root_stage": root["stage"],
    }
    _write(output / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--superseded-v3-manifest", "--v3-root-manifest",
                        dest="superseded_v3_manifest_path", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--history-root", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = run(
        output=args.output,
        superseded_v3_manifest_path=args.superseded_v3_manifest_path,
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
