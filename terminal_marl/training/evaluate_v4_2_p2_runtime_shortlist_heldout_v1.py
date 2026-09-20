"""Untouched paired heldout evaluation for K=3 formal PPO V1."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from terminal_marl.algo.gate_calibrated_adapter_v4_2 import (
    CALIBRATION_TEMPERATURE,
    TemperatureCalibratedGateResidualAdapterStudentPPOV42,
)
from terminal_marl.core.checkpoints import load_checkpoint
from terminal_marl.training import evaluate_v4_2_p2_gate_effectiveness_v1 as effectiveness
from terminal_marl.training.evaluate_v4_2_p2_gate_runtime_shortlist_v1 import (
    _tape_shortlist_checks,
)
from terminal_marl.training.run_v4_2_p2_runtime_shortlist_entry_v1 import (
    _frozen_state_equal,
)
from terminal_marl.training.run_v4_2_ppo_p2_resume_v4 import p2_config
from terminal_marl.training.run_v4_2_v2_1_formal_i0_r4 import historical_registry
from terminal_marl.training.run_v4_2_v2_1_i1_v2 import reserve_once


STAGE = "V4_2_P2_RUNTIME_SHORTLIST_HELDOUT_V1"
PROTOCOL = Path("terminal_marl/configs/low_v4_2/ppo_p2_runtime_shortlist_heldout_v1.yaml")
PROTOCOL_SHA = "eee2543918bf80610e80ccf11efdac602369d3bb46dd2226ce53ed87d57950e6"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True), encoding="utf-8")


def _delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def protocol() -> dict[str, Any]:
    if _sha256(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("runtime-shortlist heldout protocol changed")
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    if value.get("protocol") != STAGE:
        raise ValueError("runtime-shortlist heldout protocol id changed")
    return value


def source_hashes() -> dict[str, str]:
    paths = (
        Path("terminal_marl/training/evaluate_v4_2_p2_runtime_shortlist_heldout_v1.py"),
        Path("terminal_marl/training/evaluate_v4_2_p2_gate_runtime_shortlist_v1.py"),
        Path("terminal_marl/training/evaluate_v4_2_p2_gate_effectiveness_v1.py"),
        Path("terminal_marl/training/evaluate_v4_2_p2_v4.py"),
        Path("terminal_marl/training/run_v4_2_i2_dagger_preflight.py"),
        Path("terminal_marl/training/student_rollout_v4_2.py"),
        Path("terminal_marl/envs/parallel_env.py"),
        Path("terminal_marl/core/safety_trace.py"),
    )
    return {path.as_posix(): _sha256(path) for path in paths}


def validate_formal(manifest: dict[str, Any], p: dict[str, Any]) -> None:
    bindings = p["bindings"]
    if (manifest.get("stage") != "V4_2_P2_RUNTIME_SHORTLIST_FORMAL_V1"
            or manifest.get("PASS") is not True
            or not manifest.get("gates")
            or not all(manifest["gates"].values())
            or manifest.get("HELDOUT_EVALUATION_ALLOWED") is not True
            or manifest.get("formal_training_called") is not True
            or manifest.get("checkpoint_promotable") is not False):
        raise PermissionError("runtime-shortlist formal training has not passed")
    if (manifest.get("protocol_sha256") != bindings["formal_protocol_sha256"]
            or manifest.get("final_checkpoint_sha256")
            != bindings["trained_checkpoint_sha256"]
            or manifest.get("candidate_checkpoint_sha256")
            != bindings["initial_checkpoint_sha256"]
            or manifest.get("heldout_evaluation_seeds_reserved")
            != p["heldout"]["seeds"]):
        raise ValueError("runtime-shortlist formal binding changed")


def _load_trained(checkpoint: Path, p: dict[str, Any]):
    bindings = p["bindings"]
    if _sha256(checkpoint) != bindings["trained_checkpoint_sha256"]:
        raise ValueError("trained checkpoint SHA mismatch")
    policy = TemperatureCalibratedGateResidualAdapterStudentPPOV42(
        88, temperature=CALIBRATION_TEMPERATURE,
    )
    payload = load_checkpoint(checkpoint, policy, p2_config(6), load_optimizer=False)
    expected = {
        "stage": "V4_2_P2_RUNTIME_SHORTLIST_FORMAL_V1",
        "formal": True,
        "ppo_called": True,
        "protocol_sha256": bindings["formal_protocol_sha256"],
        "initial_checkpoint_sha256": bindings["initial_checkpoint_sha256"],
        "checkpoint_promotable": False,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("trained checkpoint payload binding changed")
    policy.model.eval()
    return policy, payload


def run(*, output: Path, formal_manifest_path: Path, trained_checkpoint: Path,
        initial_checkpoint: Path, history_root: Path) -> dict[str, Any]:
    p = protocol()
    bindings, heldout = p["bindings"], p["heldout"]
    if _sha256(formal_manifest_path) != bindings["formal_manifest_sha256"]:
        raise ValueError("formal manifest SHA mismatch")
    formal_manifest = json.loads(formal_manifest_path.read_text(encoding="utf-8"))
    validate_formal(formal_manifest, p)
    if _sha256(initial_checkpoint) != bindings["initial_checkpoint_sha256"]:
        raise ValueError("initial checkpoint SHA mismatch")
    trained, _ = _load_trained(trained_checkpoint, p)
    initial, _ = effectiveness.load_candidate_checkpoint(initial_checkpoint)
    frozen_proposal_subgraph = _frozen_state_equal(
        initial.model.state_dict(), trained.model.state_dict(),
    )
    output.mkdir(parents=True, exist_ok=False)
    seeds = [int(seed) for seed in heldout["seeds"]]
    history = historical_registry(history_root, output)
    if set(seeds) & set(history["seeds"]):
        raise ValueError("heldout seeds were previously consumed")
    marker = trained_checkpoint.parent / str(heldout["reservation_marker"])
    reserve_once(marker, {
        "stage": STAGE,
        "protocol_sha256": PROTOCOL_SHA,
        "heldout_seeds": seeds,
        "controllers": list(heldout["controllers"]),
        "output": str(output),
        "reserved_before_first_reset": True,
    })
    _write(output / "pre_reset_registry.json", {
        "stage": STAGE,
        "heldout_seeds": seeds,
        "protocol_sha256": PROTOCOL_SHA,
        "reservation_marker": str(marker),
        "reserved_before_first_reset": True,
    })
    records: list[dict[str, Any]] = []
    controllers = ("trained_k3", "initial_k3", "rule")
    for index, seed in enumerate(seeds):
        scenario = str(heldout["scenarios"][index])
        config = replace(
            p2_config(int(scenario.removeprefix("flow"))),
            runtime_proposal_shortlist_k=3,
        )
        for controller, policy in (
            ("trained_k3", trained),
            ("initial_k3", initial),
            ("rule", None),
        ):
            folder = output / f"{controller}_{seed}"
            result = (effectiveness._rule_result(config, seed, folder, smoke=False)
                      if policy is None else effectiveness._student_result(
                          policy, config, seed, folder, smoke=False,
                      ))
            checks = ({
                "shortlists_supplied": 0,
                "fallback_selected": 0,
                "exhausted": 0,
                "max_candidates_checked": 0,
                "requested_is_first": True,
                "unique": True,
                "bounded": True,
                "teacher_free": True,
                "all": True,
            } if policy is None else _tape_shortlist_checks(
                result["summary"]["action_tape"], expected_k=3,
            ))
            result["runtime_shortlist_checks"] = checks
            result["gates"]["runtime_shortlist_contract"] = bool(checks["all"])
            result["gates"]["all"] = all(
                bool(value) for key, value in result["gates"].items() if key != "all"
            )
            _write(folder / "evaluation.json", result)
            records.append({
                "controller": controller,
                "seed": seed,
                "scenario": scenario,
                "result": result,
            })
            _write(output / "progress.json", {
                "completed": len(records),
                "planned": len(seeds) * len(controllers),
                "last_controller": controller,
                "last_seed": seed,
                "last_gates": result["gates"],
            })
            if not result["gates"]["all"]:
                raise RuntimeError(f"heldout episode gate failed: {controller}/{seed}")
    aggregates = {
        controller: effectiveness.aggregate_records(records, controller)
        for controller in controllers
    }
    trained_metrics = aggregates["trained_k3"]
    initial_metrics = aggregates["initial_k3"]
    rule_metrics = aggregates["rule"]
    versus_initial = {
        "weighted_wait_seconds": _delta(
            trained_metrics["weighted_wait_seconds"],
            initial_metrics["weighted_wait_seconds"],
        ),
        "request_to_touchdown_seconds": _delta(
            trained_metrics["request_to_touchdown_seconds"],
            initial_metrics["request_to_touchdown_seconds"],
        ),
    }
    versus_rule = {
        "weighted_wait_seconds": _delta(
            trained_metrics["weighted_wait_seconds"], rule_metrics["weighted_wait_seconds"],
        ),
        "request_to_touchdown_seconds": _delta(
            trained_metrics["request_to_touchdown_seconds"],
            rule_metrics["request_to_touchdown_seconds"],
        ),
    }
    trained_checks = [
        record["result"]["runtime_shortlist_checks"]
        for record in records if record["controller"] == "trained_k3"
    ]
    fallback_total = sum(int(check["fallback_selected"]) for check in trained_checks)
    decision = {
        "all_records_complete": len(records) == len(seeds) * len(controllers),
        "all_episode_gates": all(record["result"]["gates"]["all"] for record in records),
        "frozen_proposal_subgraph": bool(frozen_proposal_subgraph),
        "trained_runtime_fallback_observed": fallback_total > 0,
        "trained_wait_nonworse_than_initial": versus_initial["weighted_wait_seconds"] <= 0.0,
        "trained_request_to_touchdown_nonworse_than_initial":
            versus_initial["request_to_touchdown_seconds"] <= 0.0,
        "at_least_one_metric_strictly_better_than_initial": any(
            value < 0.0 for value in versus_initial.values()
        ),
        "trained_wait_nonworse_than_rule": versus_rule["weighted_wait_seconds"] <= 0.0,
        "trained_request_to_touchdown_nonworse_than_rule":
            versus_rule["request_to_touchdown_seconds"] <= 0.0,
    }
    decision["all"] = all(decision.values())
    passed = bool(decision["all"])
    result = {
        "stage": STAGE,
        "PASS": passed,
        "decision": decision,
        "records": records,
        "aggregates": aggregates,
        "trained_minus_initial": versus_initial,
        "trained_minus_rule": versus_rule,
        "trained_runtime_shortlist_fallback_selected": fallback_total,
        "checkpoint_promotable": passed,
        "STATIC_GEOMETRY_QUALIFICATION_ALLOWED": passed,
        "P3_ALLOWED": False,
        "protocol_sha256": PROTOCOL_SHA,
        "source_hashes": source_hashes(),
        "formal_manifest_sha256": bindings["formal_manifest_sha256"],
        "trained_checkpoint_sha256": bindings["trained_checkpoint_sha256"],
        "initial_checkpoint_sha256": bindings["initial_checkpoint_sha256"],
        "reservation_marker": str(marker),
    }
    _write(output / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--formal-manifest", type=Path, required=True)
    parser.add_argument("--trained-checkpoint", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--history-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        output=args.output,
        formal_manifest_path=args.formal_manifest,
        trained_checkpoint=args.trained_checkpoint,
        initial_checkpoint=args.initial_checkpoint,
        history_root=args.history_root,
    )
    print(json.dumps({
        "stage": result["stage"],
        "PASS": result["PASS"],
        "checkpoint_promotable": result["checkpoint_promotable"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
