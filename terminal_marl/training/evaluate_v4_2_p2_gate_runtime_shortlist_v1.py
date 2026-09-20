"""Fresh K=1/K=3/rule evaluation of bounded runtime proposal certification."""
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

from terminal_marl.training import evaluate_v4_2_p2_gate_effectiveness_v1 as effectiveness
from terminal_marl.training.run_v4_2_ppo_p2_resume_v4 import p2_config
from terminal_marl.training.run_v4_2_v2_1_formal_i0_r4 import historical_registry
from terminal_marl.training.run_v4_2_v2_1_i1_v2 import reserve_once


STAGE = "V4_2_P2_GATE_RUNTIME_SHORTLIST_EVALUATION_V1"
PROTOCOL = Path("terminal_marl/configs/low_v4_2/ppo_p2_gate_runtime_shortlist_evaluation_v1.yaml")
PROTOCOL_SHA = "82ec0721972b120c105d79ceb288ee11bfdba36d3c950977c090bb0f5a36e82f"
CANDIDATE_SHA = effectiveness.REPAIRED_CHECKPOINT_SHA
FORMAL_SEEDS = (805001, 805002, 805003, 805004, 805005, 805006)
SMOKE_SEED = 805000
CONTROLLERS = ("candidate_k1", "candidate_k3", "rule")
RESERVATION_MARKER = "p2_gate_runtime_shortlist_evaluation_v1_consumed.json"


def configure_successor_v2() -> None:
    """Select the fresh successor after correcting likelihood tape replay."""
    global STAGE, PROTOCOL, PROTOCOL_SHA, FORMAL_SEEDS, SMOKE_SEED, RESERVATION_MARKER
    STAGE = "V4_2_P2_GATE_RUNTIME_SHORTLIST_EVALUATION_V2"
    PROTOCOL = Path("terminal_marl/configs/low_v4_2/ppo_p2_gate_runtime_shortlist_evaluation_v2.yaml")
    PROTOCOL_SHA = "67e61165ef5c056d1b1ebd0a789cafd17f5e0554d07b29128617df3aa8bf6d5d"
    FORMAL_SEEDS = (806001, 806002, 806003, 806004, 806005, 806006)
    SMOKE_SEED = 806000
    RESERVATION_MARKER = "p2_gate_runtime_shortlist_evaluation_v2_consumed.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


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


def protocol() -> dict[str, Any]:
    _require(_sha256(PROTOCOL) == PROTOCOL_SHA, "runtime-shortlist protocol changed")
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    _require(value.get("protocol") == STAGE, "runtime-shortlist protocol id changed")
    bindings = value["bindings"]
    expected = {
        "observation_schema": effectiveness.OBSERVATION_SCHEMA,
        "observation_dimension": effectiveness.OBSERVATION_DIM,
        "action_schema": effectiveness.ACTION_SCHEMA,
        "checkpoint_sha256": CANDIDATE_SHA,
        "deterministic": True,
        "one_policy_forward": True,
        "requested_proposal_is_rank_1": True,
        "max_runtime_proposals": 3,
        "full_catalogue_search": False,
        "delayed_candidate_search": False,
        "no_teacher": True,
        "no_labels": True,
        "no_updates": True,
    }
    for key, expected_value in expected.items():
        _require(bindings.get(key) == expected_value, f"runtime-shortlist binding changed: {key}")
    formal = value["formal"]
    _require(formal["seeds"] == list(FORMAL_SEEDS), "formal seeds changed")
    _require(formal["controllers"] == list(CONTROLLERS), "formal controllers changed")
    _require(value["smoke"]["seed"] == SMOKE_SEED, "smoke seed changed")
    return value


def _tape_shortlist_checks(path: str | Path, *, expected_k: int) -> dict[str, Any]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))["action_tape"]
    supplied = 0
    fallback = 0
    exhausted = 0
    max_checked = 0
    first_matches = True
    unique = True
    bounded = True
    teacher_free = True
    for row in rows:
        shortlists = row.get("runtime_proposal_shortlists", {})
        results = row.get("runtime_shortlist_result", {})
        if expected_k == 1:
            bounded = bounded and not shortlists
        for uid, values in shortlists.items():
            supplied += 1
            bounded = bounded and 1 <= len(values) <= expected_k
            first_matches = first_matches and values[0] == row.get("a_requested", {}).get(uid)
            proposals = [tuple(float(component) for component in value) for value in values]
            unique = unique and len(set(proposals)) == len(proposals)
        for result in results.values():
            checked = int(result.get("candidates_checked", 0))
            max_checked = max(max_checked, checked)
            fallback += int(bool(result.get("used")))
            exhausted += int(checked > 0 and result.get("rank") is None)
            bounded = bounded and checked <= expected_k
        teacher_free = teacher_free and all(
            value is None for value in row.get("teacher_joint", {}).values()
        )
    return {
        "shortlists_supplied": supplied,
        "fallback_selected": fallback,
        "exhausted": exhausted,
        "max_candidates_checked": max_checked,
        "requested_is_first": first_matches,
        "unique": unique,
        "bounded": bounded,
        "teacher_free": teacher_free,
        "all": bool(first_matches and unique and bounded and teacher_free
                    and (supplied > 0 if expected_k > 1 else supplied == 0)),
    }


def _delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def run(*, candidate_checkpoint: Path, output: Path, history_root: Path,
        smoke: bool = False) -> dict[str, Any]:
    protocol()
    _require(not output.exists(), f"output already exists: {output}")
    _require(_sha256(candidate_checkpoint) == CANDIDATE_SHA, "candidate checkpoint SHA mismatch")
    candidate, _ = effectiveness.load_candidate_checkpoint(candidate_checkpoint)
    output.mkdir(parents=True, exist_ok=False)
    seeds = (SMOKE_SEED,) if smoke else FORMAL_SEEDS
    reservation = None
    if not smoke:
        registry = historical_registry(history_root, output)
        _require(not (set(FORMAL_SEEDS) & set(registry["seeds"])), "formal seeds already consumed")
        marker = candidate_checkpoint.parent / RESERVATION_MARKER
        _require(not marker.exists(), "runtime-shortlist evaluation already consumed")
        reservation = {
            "protocol": STAGE, "protocol_sha256": PROTOCOL_SHA,
            "seeds": list(FORMAL_SEEDS), "controllers": list(CONTROLLERS),
            "output": str(output), "reserved_before_first_reset": True,
        }
        reserve_once(marker, reservation)
    records: list[dict[str, Any]] = []
    for seed in seeds:
        k1 = replace(p2_config(12), runtime_proposal_shortlist_k=1)
        k3 = replace(p2_config(12), runtime_proposal_shortlist_k=3)
        for controller, policy, config in (
            ("candidate_k1", candidate, k1),
            ("candidate_k3", candidate, k3),
            ("rule", None, k1),
        ):
            folder = output / f"{controller}_{seed}"
            result = (effectiveness._rule_result(config, seed, folder, smoke=smoke)
                      if policy is None else effectiveness._student_result(
                          policy, config, seed, folder, smoke=smoke,
                      ))
            tape_checks = ({
                "shortlists_supplied": 0, "fallback_selected": 0, "exhausted": 0,
                "max_candidates_checked": 0, "requested_is_first": True,
                "unique": True, "bounded": True, "teacher_free": True, "all": True,
            } if policy is None else _tape_shortlist_checks(
                result["summary"]["action_tape"],
                expected_k=3 if controller == "candidate_k3" else 1,
            ))
            result["runtime_shortlist_checks"] = tape_checks
            result["gates"]["runtime_shortlist_contract"] = tape_checks["all"]
            result["gates"]["all"] = all(
                bool(value) for key, value in result["gates"].items() if key != "all"
            )
            _write(folder / "evaluation.json", result)
            records.append({"controller": controller, "seed": seed,
                            "scenario": "flow12", "result": result})
            _write(output / "progress.json", {
                "completed": len(records), "planned": len(seeds) * len(CONTROLLERS),
                "last_controller": controller, "last_seed": seed,
                "last_gates": result["gates"],
            })
            if not smoke:
                _require(result["gates"]["all"], f"episode gate failed: {controller}/{seed}")
    aggregates = {
        name: effectiveness.aggregate_records(records, name) for name in CONTROLLERS
    }
    k1, k3 = aggregates["candidate_k1"], aggregates["candidate_k3"]
    deltas = {
        "weighted_wait_seconds": _delta(k3["weighted_wait_seconds"], k1["weighted_wait_seconds"]),
        "request_to_touchdown_seconds": _delta(
            k3["request_to_touchdown_seconds"], k1["request_to_touchdown_seconds"],
        ),
        "shield_actor_fraction": _delta(k3["shield_actor_fraction"], k1["shield_actor_fraction"]),
    }
    k3_checks = [row["result"]["runtime_shortlist_checks"] for row in records
                 if row["controller"] == "candidate_k3"]
    fallback_total = sum(int(row["fallback_selected"]) for row in k3_checks)
    decision = {
        "all_episode_gates": all(row["result"]["gates"]["all"] for row in records),
        "k3_completion_fraction": float(k3["completion_fraction"]) >= 1.0,
        "k3_safety_zero": bool(k3["safety_all_zero"]),
        "k3_contract_all": bool(k3_checks) and all(row["all"] for row in k3_checks),
        "k3_fallback_selected_nonzero": fallback_total > 0,
        "k3_wait_nonworse": deltas["weighted_wait_seconds"] is not None
            and deltas["weighted_wait_seconds"] <= 0.0,
        "k3_request_to_touchdown_nonworse": deltas["request_to_touchdown_seconds"] is not None
            and deltas["request_to_touchdown_seconds"] <= 0.0,
        "at_least_one_effectiveness_metric_strictly_better": any(
            value is not None and value < 0.0
            for value in (deltas["weighted_wait_seconds"], deltas["request_to_touchdown_seconds"])
        ),
    }
    decision["all"] = all(decision.values())
    result = {
        "stage": STAGE, "status": "INCOMPLETE" if smoke else "COMPLETE",
        "smoke": smoke, "protocol": str(PROTOCOL), "protocol_sha256": PROTOCOL_SHA,
        "candidate_checkpoint": str(candidate_checkpoint),
        "candidate_checkpoint_sha256": CANDIDATE_SHA,
        "records": records, "aggregates": aggregates, "k3_minus_k1": deltas,
        "runtime_shortlist_fallback_selected": fallback_total,
        "decision": decision,
        "RUNTIME_SHORTLIST_FREEZE": bool(not smoke and decision["all"]),
        "PPO_ALLOWED": False, "checkpoint_promotable": False,
        "reservation": reservation,
    }
    _write(output / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--successor-v2", action="store_true")
    args = parser.parse_args()
    if args.successor_v2:
        configure_successor_v2()
    print(json.dumps(_jsonable(run(
        candidate_checkpoint=args.candidate_checkpoint, output=args.output,
        history_root=args.history_root, smoke=args.smoke,
    )), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
