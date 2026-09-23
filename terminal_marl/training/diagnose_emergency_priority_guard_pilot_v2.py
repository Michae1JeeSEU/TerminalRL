"""Paired Guard V2 diagnostic with a fresh seed matrix.

Guard V2 freezes the current energy-guard source snapshot and reuses the
immutable V1 case runner.  It does not train, update, collect labels, or
promote a checkpoint.  Formal seeds are intentionally not consumed by local
tests or smoke serialization checks.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

from terminal_marl.training import diagnose_emergency_priority_guard_pilot_v1 as v1


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = PROJECT_ROOT / "terminal_marl" / "configs" / "low_v4_2" / "emergency_priority_guard_pilot_v2.yaml"
PROTOCOL_SHA256 = "151ae69248626d7f1afcfa8e0fb0c5e0193f86552fd620167ec054bddaa8943f"
CHECKPOINT_SHA256 = "4b8be66c100aaaeb5b5699dcce68a96dbe1237964e4c308e7b23b95c62fb8c8c"
FORMAL_SEEDS = (840501, 840502)
SMOKE_SEED = 840500
CONTROLLERS = ("rule", "original_k3")
GUARD_MODES = (False, True)
ACTION_SCHEMA = "low_action_schema_v4_2_hierarchical"
OBSERVATION_SCHEMA = "phaseA_low_obs_v2_1_retry_state"
RUNTIME_SHORTLIST_K = 3
SOURCE_BINDING_KEYS = (
    "terminal_marl/core/config.py",
    "terminal_marl/core/uav.py",
    "terminal_marl/envs/parallel_env.py",
    "terminal_marl/training/diagnose_emergency_priority_guard_pilot_v1.py",
)
SAFETY_COUNTER_KEYS = v1.SAFETY_COUNTER_KEYS
PREDECESSOR_MANIFEST_SHA256 = "af111076ef98870a1cee930b3396d7555f68be7111cd72c743a9054595da2510"


class ProtocolError(ValueError):
    """Raised when the frozen V2 binding is not satisfied."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def _sha256(path: str | Path) -> str:
    return v1._sha256(path)


_write_json = v1._write_json


def load_protocol(path: str | Path = PROTOCOL_PATH) -> dict[str, Any]:
    """Validate protocol, source and scenario bindings before any reset."""
    path = Path(path)
    _require(path.resolve() == PROTOCOL_PATH.resolve(), "only the frozen Guard V2 protocol is accepted")
    _require(_sha256(path) == PROTOCOL_SHA256, "Guard V2 protocol SHA256 mismatch")
    value = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), "Guard V2 protocol root must be a mapping")
    _require(value.get("protocol") == "EMERGENCY_PRIORITY_GUARD_PILOT_V2", "protocol id changed")
    _require(value.get("status") == "PREREGISTERED_EMERGENCY_PRIORITY_GUARD_DIAGNOSTIC", "protocol status changed")

    freeze = value.get("freeze")
    _require(isinstance(freeze, Mapping), "freeze section missing")
    for key in (
        "diagnostic_only", "no_training", "no_optimizer", "no_teacher", "no_labels",
        "no_updates", "deterministic", "no_output_overwrite", "no_rerun_no_reset_no_tuning",
    ):
        _require(freeze.get(key) is True, f"freeze.{key} must be true")
    _require(freeze.get("ppo_allowed") is False, "freeze.ppo_allowed must be false")

    checkpoint = value.get("checkpoint")
    _require(isinstance(checkpoint, Mapping), "checkpoint section missing")
    _require(checkpoint.get("controller") == "original_k3", "checkpoint controller changed")
    _require(checkpoint.get("role") == "frozen_original_k3", "checkpoint role changed")
    _require(checkpoint.get("sha256") == CHECKPOINT_SHA256, "checkpoint SHA binding changed")

    predecessor = value.get("predecessor")
    _require(isinstance(predecessor, Mapping), "predecessor section missing")
    _require(predecessor.get("protocol") == "EMERGENCY_PRIORITY_GUARD_PILOT_V1", "predecessor protocol changed")
    _require(predecessor.get("manifest_sha256") == PREDECESSOR_MANIFEST_SHA256, "predecessor manifest binding changed")

    bindings = value.get("source_bindings")
    _require(isinstance(bindings, Mapping), "source_bindings section missing")
    _require(tuple(bindings) == SOURCE_BINDING_KEYS, "source binding list changed")
    for relative in SOURCE_BINDING_KEYS:
        expected = str(bindings.get(relative, "")).lower()
        _require(len(expected) == 64, f"source binding is not SHA256: {relative}")
        observed = _sha256(PROJECT_ROOT / relative)
        _require(observed == expected, f"source binding mismatch: {relative}")

    scenario = value.get("scenario")
    _require(isinstance(scenario, Mapping), "scenario section missing")
    expected_scenario = {
        "reset_mode": "warm_start_rings",
        "warm_start_uavs": 6,
        "warm_start_steady_motion": True,
        "total_requested_uavs": 12,
        "initial_soc_mode": "uniform",
        "initial_soc_min": 0.35,
        "initial_soc_max": 0.55,
        "horizon_seconds": 1200,
        "lz_assignment_mode": "nearest_available",
        "release_arbitration_mode": "downstream_priority",
        "final_queue_capacity": 3,
        "energy_model_mode": "phase_motion_v2",
        "phase_motion_nominal_solo_energy": 240.9,
        "emergency_landing_enabled": True,
        "emergency_trigger_soc": 0.40,
        "emergency_energy_margin_seconds": 15,
    }
    for key, expected_value in expected_scenario.items():
        _require(scenario.get(key) == expected_value, f"scenario.{key} must equal {expected_value!r}")
    schedule = tuple(float(item) for item in scenario.get("fixed_arrival_schedule", ()))
    _require(
        schedule == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0),
        "scenario.fixed_arrival_schedule changed",
    )

    low_level = value.get("low_level")
    _require(isinstance(low_level, Mapping), "low_level section missing")
    for key, expected_value in {
        "action_schema": ACTION_SCHEMA,
        "observation_schema": OBSERVATION_SCHEMA,
        "observation_dimension": 88,
        "runtime_proposal_shortlist_k": RUNTIME_SHORTLIST_K,
    }.items():
        _require(low_level.get(key) == expected_value, f"low_level.{key} changed")

    safety = value.get("safety")
    _require(isinstance(safety, Mapping), "safety section missing")
    for key, expected_value in (("safe_distance_m", 80), ("protected_floor_m", 40), ("collision_distance_m", 10)):
        _require(safety.get(key) == expected_value, f"safety.{key} must equal {expected_value}")

    matrix = value.get("matrix")
    _require(isinstance(matrix, Mapping), "matrix section missing")
    _require(tuple(int(seed) for seed in matrix.get("seeds", ())) == FORMAL_SEEDS, "formal seed binding changed")
    _require(matrix.get("expected_records") == len(FORMAL_SEEDS) * len(CONTROLLERS) * len(GUARD_MODES), "formal record count changed")
    _require(matrix.get("controllers") == list(CONTROLLERS), "controller order changed")
    _require(matrix.get("guard_modes") == list(GUARD_MODES), "guard mode order changed")
    _require(matrix.get("paired_same_seed_controller_guard") is True, "paired_same_seed_controller_guard must be true")
    _require(matrix.get("deterministic") is True, "matrix.deterministic must be true")

    smoke = value.get("smoke")
    _require(isinstance(smoke, Mapping), "smoke section missing")
    _require(smoke.get("seed") == SMOKE_SEED, "smoke seed changed")
    _require(isinstance(smoke.get("steps"), int) and smoke["steps"] > 0, "smoke steps invalid")
    _require(smoke.get("formal_marker") is False, "smoke formal marker must be false")

    hard = value.get("hard_gates")
    _require(isinstance(hard, Mapping), "hard_gates section missing")
    for key in (
        "exact_request_accounting", "all_records_terminal", "zero_safety_events",
        "guard_on_zero_failed_energy", "zero_in_terminal_low_soc_diversion", "exact_action_replay",
        "no_teacher", "no_labels", "no_updates", "runtime_shortlist_k3",
    ):
        _require(hard.get(key) is True, f"hard_gates.{key} must be true")

    authorization = value.get("authorization")
    _require(isinstance(authorization, Mapping), "authorization section missing")
    for key in ("PPO_ALLOWED", "training_allowed", "checkpoint_promotion_allowed"):
        _require(authorization.get(key) is False, f"authorization.{key} must be false")
    return dict(value)


def build_config(protocol: Mapping[str, Any], *, guard_enabled: bool):
    """Build V1's static scenario with only the explicit guard flag changed."""
    _require(isinstance(guard_enabled, bool), "guard_enabled must be boolean")
    return v1.build_config(protocol, guard_enabled=guard_enabled)


def _run_case(
    policy: Any,
    config: Any,
    seed: int,
    controller: str,
    guard_enabled: bool,
    output: Path,
    *,
    max_steps: int,
) -> dict[str, Any]:
    """Delegate one deterministic case to the frozen V1 case runner."""
    return v1._run_case(policy, config, seed, controller, guard_enabled, output, max_steps=max_steps)


paired_deltas = v1.paired_deltas
_aggregate = v1._aggregate


def _smoke_aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    outcomes = Counter()
    for record in records:
        outcomes.update(record.get("outcome_counts", {}))
    return {
        "records": len(records),
        "outcome_counts": dict(sorted(outcomes.items())),
        "safety_totals": {
            key: sum(int(record.get("safety", {}).get(key, 0)) for record in records)
            for key in SAFETY_COUNTER_KEYS
        },
        "hard_gates": {key: False for key in (
            "expected_records", "exact_request_accounting", "all_records_terminal", "zero_safety_events",
            "guard_on_zero_failed_energy", "zero_in_terminal_low_soc_diversion", "exact_action_replay",
            "no_teacher", "no_labels", "no_updates", "runtime_shortlist_k3",
        )},
        "PASS": False,
    }


def run(*, checkpoint: str | Path, output: str | Path, smoke: bool = False) -> dict[str, Any]:
    """Run smoke or the fresh eight-record Guard V2 matrix."""
    protocol = load_protocol()
    output = Path(output)
    _require(not output.exists(), f"refusing to overwrite existing output: {output}")
    checkpoint = Path(checkpoint)
    _require(checkpoint.is_file(), f"checkpoint not found: {checkpoint}")
    _require(_sha256(checkpoint) == CHECKPOINT_SHA256, "checkpoint SHA256 mismatch")

    # Repeat the source check in the run path so all bindings are still valid
    # immediately before checkpoint loading and the first environment reset.
    source_hashes = {relative: _sha256(PROJECT_ROOT / relative) for relative in SOURCE_BINDING_KEYS}
    _require(
        source_hashes == {relative: str(protocol["source_bindings"][relative]).lower() for relative in SOURCE_BINDING_KEYS},
        "source bindings changed after protocol validation",
    )
    policy, payload = v1.paired.effectiveness.load_candidate_checkpoint(
        checkpoint, expected_sha256=CHECKPOINT_SHA256,
    )
    _require(payload.get("PPO_ALLOWED") is False, "checkpoint PPO_ALLOWED binding changed")
    _require(len(policy.optimizer.state) == 0, "checkpoint optimizer state was loaded")

    output.mkdir(parents=False, exist_ok=False)
    smoke_value = protocol["smoke"]
    seeds = (int(smoke_value["seed"]),) if smoke else FORMAL_SEEDS
    base_config = build_config(protocol, guard_enabled=False)
    max_steps = int(smoke_value["steps"]) if smoke else int(float(protocol["scenario"]["horizon_seconds"]) / base_config.dt)
    records: list[dict[str, Any]] = []
    for seed in seeds:
        for controller in CONTROLLERS:
            for guard_enabled in GUARD_MODES:
                mode = "on" if guard_enabled else "off"
                case_output = output / f"{controller}_guard_{mode}_seed_{seed}"
                config = replace(base_config, emergency_priority_guard_enabled=guard_enabled)
                records.append(_run_case(policy, config, seed, controller, guard_enabled, case_output, max_steps=max_steps))
                _write_json(output / "progress.json", {
                    "completed_records": len(records),
                    "expected_records": len(seeds) * len(CONTROLLERS) * len(GUARD_MODES),
                    "records": records,
                })

    aggregate = _smoke_aggregate(records) if smoke else _aggregate(records)
    manifest = {
        "protocol": protocol["protocol"],
        "protocol_path": str(PROTOCOL_PATH),
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "runner_sha256": _sha256(Path(__file__)),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_path": str(checkpoint),
        "checkpoint_payload": {key: payload.get(key) for key in ("stage", "architecture_id", "qualification_pass", "PPO_ALLOWED")},
        "source_sha256": source_hashes,
        "smoke": bool(smoke),
        "seeds": list(seeds),
        "controllers": list(CONTROLLERS),
        "guard_modes": ["off", "on"],
        "records": records,
        "paired_deltas": paired_deltas(records),
        "aggregate": aggregate,
        "diagnostic_only": True,
        "training_called": False,
        "teacher_used": False,
        "labels_collected": False,
        "optimizer_loaded": False,
        "PPO_ALLOWED": False,
        "guard_telemetry_available": False,
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    manifest = run(checkpoint=args.checkpoint, output=args.output, smoke=args.smoke)
    print(json.dumps({
        "PASS": manifest["aggregate"]["PASS"],
        "smoke": manifest["smoke"],
        "records": len(manifest["records"]),
    }, sort_keys=True), flush=True)
    return 0 if manifest["smoke"] or manifest["aggregate"]["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
