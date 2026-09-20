"""P3 priority/SOC curriculum: qualification, entry, then formal PPO."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from terminal_marl.core.checkpoints import save_checkpoint
from terminal_marl.core.safety_trace import SafetyTraceRecorder
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training import evaluate_v4_2_p2_gate_effectiveness_v1 as effectiveness
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
from terminal_marl.training.run_v4_2_ppo_p2_resume_v4 import (
    make_update_gates,
    p2_config,
)
from terminal_marl.training.run_v4_2_v2_1_formal_i0_r4 import historical_registry
from terminal_marl.training.run_v4_2_v2_1_i1_v2 import reserve_once
from terminal_marl.training.run_v4_2_v2_1_ppo import requested_logprob_error
from terminal_marl.training.student_rollout_v4_2 import rollout


STAGE = "V4_2_P3_PRIORITY_SOC_CURRICULUM_V1"
PROTOCOL = Path("terminal_marl/configs/low_v4_2/p3_priority_soc_curriculum_v1.yaml")
PROTOCOL_SHA = "ea5efc064f763b39cd18f07f55e84f07754aebbe40d8ded588b9b61998559b25"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def protocol() -> dict[str, Any]:
    if _sha256(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("P3 priority/SOC curriculum protocol changed")
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    if value.get("protocol") != STAGE:
        raise ValueError("P3 priority/SOC curriculum protocol id changed")
    return value


def source_hashes() -> dict[str, str]:
    paths = (
        Path("terminal_marl/training/run_v4_2_p3_priority_soc_curriculum_v1.py"),
        Path("terminal_marl/training/run_v4_2_p2_runtime_shortlist_entry_v1.py"),
        Path("terminal_marl/algo/joint_ppo_v4_2.py"),
        Path("terminal_marl/algo/student_ppo_v4_2.py"),
        Path("terminal_marl/algo/gate_calibrated_adapter_v4_2.py"),
        Path("terminal_marl/training/student_rollout_v4_2.py"),
        Path("terminal_marl/envs/parallel_env.py"),
        Path("terminal_marl/core/safety_trace.py"),
        Path("terminal_marl/core/arrival_process.py"),
        Path("terminal_marl/core/priority.py"),
    )
    return {path.as_posix(): _sha256(path) for path in paths}


def validate_selection_review(review: dict[str, Any], p: dict[str, Any]) -> None:
    bindings = p["bindings"]
    if (review.get("stage") != "V4_2_P2_RUNTIME_SHORTLIST_SELECTION_REVIEW_V1"
            or review.get("PASS") is not True
            or not review.get("gates")
            or not all(review["gates"].values())
            or review.get("SELECTED_LOW_LEVEL_CANDIDATE") != "original_k3"
            or review.get("selected_checkpoint_sha256")
            != bindings["initial_checkpoint_sha256"]
            or review.get("P2_PPO_STOP") is not True
            or review.get("P3_PROTOCOL_PREREGISTRATION_ALLOWED") is not True
            or review.get("P3_ALLOWED") is not False):
        raise PermissionError("P3 protocol was not authorized by candidate selection")


def p3_config(count: int, p: dict[str, Any]):
    scenario = p["scenario"]
    return replace(
        p2_config(count),
        runtime_proposal_shortlist_k=3,
        priority_mode=str(scenario["priority_mode"]),
        initial_priority=None,
        initial_soc_mode=str(scenario["initial_soc_mode"]),
        initial_soc=None,
        initial_soc_min=float(scenario["initial_soc_min"]),
        initial_soc_max=float(scenario["initial_soc_max"]),
        mission_probabilities=tuple(float(value) for value in scenario["mission_probabilities"]),
    )


def _completed_attributes(config, tape_path: str | Path) -> dict[str, Any]:
    env = TerminalParallelEnv(config)
    replay = SafetyTraceRecorder.replay_action_tape(env, tape_path)
    priorities = [int(uav.base_priority) for uav in env.completed_uavs]
    initial_soc = [
        float((uav.energy_remaining + uav.energy_consumed) / uav.energy_capacity)
        for uav in env.completed_uavs
    ]
    missions = [str(uav.mission_type.value) for uav in env.completed_uavs]
    return {
        "replay_exact": bool(replay["exact"]),
        "completed": len(env.completed_uavs),
        "priorities": priorities,
        "initial_soc": initial_soc,
        "missions": missions,
    }


def _coverage(attribute_rows: list[dict[str, Any]], p: dict[str, Any], *, formal: bool) -> dict[str, Any]:
    priorities = sorted({value for row in attribute_rows for value in row["priorities"]})
    soc = [value for row in attribute_rows for value in row["initial_soc"]]
    section = p["hard_gates"] if formal else p["qualification"]
    required = list(section[
        "formal_priority_coverage_required" if formal else "priority_coverage_required"
    ])
    minimum_limit = float(section[
        "formal_observed_soc_min_at_most" if formal else "observed_soc_min_at_most"
    ])
    maximum_limit = float(section[
        "formal_observed_soc_max_at_least" if formal else "observed_soc_max_at_least"
    ])
    lower = float(p["scenario"]["initial_soc_min"])
    upper = float(p["scenario"]["initial_soc_max"])
    return {
        "priorities_observed": priorities,
        "priority_coverage": priorities == required,
        "soc_count": len(soc),
        "soc_min": None if not soc else min(soc),
        "soc_max": None if not soc else max(soc),
        "soc_in_bounds": bool(soc) and all(lower <= value <= upper for value in soc),
        "soc_range_observed": bool(soc) and min(soc) <= minimum_limit and max(soc) >= maximum_limit,
    }


def _qualification(output: Path, checkpoint: Path, p: dict[str, Any]) -> dict[str, Any]:
    policy, _ = effectiveness.load_candidate_checkpoint(checkpoint)
    section = p["qualification"]
    output.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    attributes: list[dict[str, Any]] = []
    for index, seed_value in enumerate(section["seeds"]):
        seed = int(seed_value)
        scenario = str(section["scenarios"][index])
        config = p3_config(int(scenario.removeprefix("flow")), p)
        for controller, active_policy in (("candidate_k3", policy), ("rule", None)):
            folder = output / f"{controller}_{seed}"
            result = (effectiveness._rule_result(config, seed, folder, smoke=False)
                      if active_policy is None else effectiveness._student_result(
                          active_policy, config, seed, folder, smoke=False,
                      ))
            checks = ({
                "shortlists_supplied": 0, "fallback_selected": 0, "exhausted": 0,
                "max_candidates_checked": 0, "requested_is_first": True,
                "unique": True, "bounded": True, "teacher_free": True, "all": True,
            } if active_policy is None else _tape_shortlist_checks(
                result["summary"]["action_tape"], expected_k=3,
            ))
            result["runtime_shortlist_checks"] = checks
            result["gates"]["runtime_shortlist_contract"] = bool(checks["all"])
            result["gates"]["all"] = all(
                bool(value) for key, value in result["gates"].items() if key != "all"
            )
            if active_policy is not None:
                row_attributes = _completed_attributes(config, result["summary"]["action_tape"])
                result["attributes"] = row_attributes
                attributes.append(row_attributes)
            records.append({
                "controller": controller, "seed": seed,
                "scenario": scenario, "result": result,
            })
            _write(output / "progress.json", {
                "completed": len(records), "total": len(section["seeds"]) * 2,
                "last_seed": seed, "last_controller": controller,
            })
            if not result["gates"]["all"]:
                break
        if not records[-1]["result"]["gates"]["all"]:
            break
    aggregates = {
        controller: effectiveness.aggregate_records(records, controller)
        for controller in ("candidate_k3", "rule")
    }
    candidate, rule = aggregates["candidate_k3"], aggregates["rule"]
    coverage = _coverage(attributes, p, formal=False)
    candidate_checks = [
        row["result"]["runtime_shortlist_checks"]
        for row in records if row["controller"] == "candidate_k3"
    ]
    gates = {
        "all_records": len(records) == len(section["seeds"]) * 2,
        "all_episode_gates": bool(records)
                             and all(row["result"]["gates"]["all"] for row in records),
        "candidate_complete": candidate["completion_fraction"] == 1.0,
        "candidate_safe": bool(candidate["safety_all_zero"]),
        "runtime_shortlist_contract": bool(candidate_checks)
                                      and all(row["all"] for row in candidate_checks),
        "runtime_fallback_observed": sum(
            int(row["fallback_selected"]) for row in candidate_checks
        ) > 0,
        "priority_coverage": bool(coverage["priority_coverage"]),
        "soc_in_bounds": bool(coverage["soc_in_bounds"]),
        "soc_range_observed": bool(coverage["soc_range_observed"]),
        "candidate_wait_nonworse_than_rule": (
            candidate["weighted_wait_seconds"] <= rule["weighted_wait_seconds"]
        ),
        "candidate_request_to_touchdown_nonworse_than_rule": (
            candidate["request_to_touchdown_seconds"]
            <= rule["request_to_touchdown_seconds"]
        ),
    }
    result = {
        "stage": "P3_QUALIFICATION", "PASS": all(gates.values()),
        "gates": gates, "records": records, "aggregates": aggregates,
        "coverage": coverage, "training_called": False,
        "P3_ENTRY_ALLOWED": all(gates.values()),
    }
    _write(output / "manifest.json", result)
    return result


def _training_stage(*, stage: str, output: Path, checkpoint: Path,
                    p: dict[str, Any]) -> dict[str, Any]:
    formal_stage = stage == "formal"
    section = p["formal" if formal_stage else "entry"]
    optimizer = p["optimizer"]
    policy, _, trainable_parameters = _fresh_policy(checkpoint, p)
    output.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    attributes: list[dict[str, Any]] = []
    np.random.seed(int(section["seeds"][0]))
    for index, seed_value in enumerate(section["seeds"]):
        seed = int(seed_value)
        scenario = str(section["scenarios"][index])
        config = p3_config(int(scenario.removeprefix("flow")), p)
        horizon = int(config.max_simulation_seconds / config.dt)
        if not formal_stage:
            horizon = int(section["steps"][index])
            config = replace(config, max_simulation_seconds=float(horizon) * config.dt)
        steps, summary = rollout(
            policy, config, seed=seed, max_steps=horizon,
            output=output / f"seed{seed}", deterministic=False,
            collect_labels=False,
        )
        initial_error = requested_logprob_error(policy, steps)
        replay_exact, geometry = replay_geometry(config, summary["action_tape"])
        likelihood = _policy_tape_logprob_exact(policy, config, summary["action_tape"])
        shortlist = _tape_shortlist_checks(summary["action_tape"], expected_k=3)
        actor_decisions = int(summary["counts"]["actor_decisions"])
        shield_fraction = summary["counts"]["shield_actor"] / max(1, actor_decisions)
        gates = {
            "safety": not any(summary["safety"].values()),
            "replay": bool(replay_exact),
            "requested_likelihood": bool(likelihood),
            "initial_ratio": initial_error <= float(
                p["hard_gates"]["initial_ratio_error_max"]),
            "student_only": int(summary.get("label_rows", 0)) == 0
                            and all(not step.teacher_used for step in steps),
            "stochastic": bool(steps) and all(step.stochastic_behavior for step in steps),
            "actor_decisions": actor_decisions > 0,
            "runtime_shortlist_contract": bool(shortlist["all"]),
            "shield_burden": shield_fraction <= float(
                p["hard_gates"]["shield_actor_fraction_max"]),
        }
        row_attributes = None
        if formal_stage:
            row_attributes = _completed_attributes(config, summary["action_tape"])
            attributes.append(row_attributes)
            gates.update({
                "all_requested_complete": summary["completed"] == config.total_requested_uavs,
                "terminal_and_queue_drained": bool(
                    summary["terminal"] and summary["queue"] and summary["queue"][-1] == 0
                ),
                "geometry": geometry["reverse_samples"] <= int(
                    p["hard_gates"]["geometry_reverse_samples_max"])
                    and geometry["max_inner_excursion_m"] <= float(
                        p["hard_gates"]["geometry_inner_excursion_max_m"]),
                "geometry_observed": geometry["samples"] > 0,
                "attribute_replay": bool(row_attributes["replay_exact"]),
                "attribute_count": row_attributes["completed"] == config.total_requested_uavs,
            })
        update = None
        saved = None
        if all(gates.values()):
            model_snapshot = copy.deepcopy(policy.model.state_dict())
            optimizer_snapshot = copy.deepcopy(policy.optimizer.state_dict())
            keys = ("minibatch_size", "gamma", "gae_lambda", "clip_ratio",
                    "value_coef", "entropy_coef")
            update = policy.update(
                steps, training_authorized=True,
                epochs=int(optimizer["update_epochs"]),
                target_kl=float(optimizer["target_kl"]),
                **{key: optimizer[key] for key in keys},
            )
            gates.update(make_update_gates(
                update, policy.model.state_dict(), policy.optimizer.state_dict(), optimizer,
            ))
            gates["frozen_proposal_subgraph"] = _frozen_state_equal(
                model_snapshot, policy.model.state_dict(),
            )
            gates["runtime_shortlist_stability"] = _policy_tape_shortlists_exact(
                policy, config, summary["action_tape"],
            )
            if not all(gates.values()):
                policy.model.load_state_dict(model_snapshot)
                policy.optimizer.load_state_dict(optimizer_snapshot)
        if all(gates.values()):
            saved = output / f"checkpoint_{index + 1}.pt"
            try:
                save_checkpoint(
                    saved, policy, config,
                    stage="V4_2_P3_FORMAL" if formal_stage else "V4_2_P3_ENTRY",
                    formal=formal_stage, ppo_called=True,
                    protocol_sha256=PROTOCOL_SHA, source_hashes=source_hashes(),
                    initial_checkpoint_sha256=p["bindings"]["initial_checkpoint_sha256"],
                    checkpoint_promotable=False,
                )
                restored = _restored_policy(saved, config, p)
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
        records.append({
            "seed": seed, "scenario": scenario, "horizon": horizon,
            "summary": summary, "geometry": geometry,
            "attributes": row_attributes,
            "initial_logprob_error": initial_error,
            "shield_actor_fraction": shield_fraction,
            "runtime_shortlist_checks": shortlist,
            "update": update,
            "checkpoint": None if saved is None else str(saved),
            "gates": gates,
        })
        _write(output / "progress.json", {
            "stage": stage, "completed": len(records),
            "total": len(section["seeds"]), "records": records,
        })
        if not all(gates.values()):
            break
    stage_gates = {
        "all_episodes": len(records) == len(section["seeds"]),
        "all_episode_gates": bool(records)
                             and all(all(row["gates"].values()) for row in records),
    }
    coverage = None
    if formal_stage:
        coverage = _coverage(attributes, p, formal=True)
        stage_gates.update({
            "priority_coverage": bool(coverage["priority_coverage"]),
            "soc_in_bounds": bool(coverage["soc_in_bounds"]),
            "soc_range_observed": bool(coverage["soc_range_observed"]),
        })
    passed = all(stage_gates.values())
    final_checkpoint = (
        Path(records[-1]["checkpoint"])
        if records and records[-1]["checkpoint"] else None
    )
    result = {
        "stage": "P3_FORMAL" if formal_stage else "P3_ENTRY",
        "PASS": passed, "gates": stage_gates, "records": records,
        "coverage": coverage,
        "P3_FORMAL_ALLOWED": bool(not formal_stage and passed),
        "HELDOUT_EVALUATION_ALLOWED": bool(formal_stage and passed),
        "formal_training_called": formal_stage,
        "entry_updates_discarded": not formal_stage,
        "checkpoint_promotable": False, "P4_ALLOWED": False,
        "final_checkpoint": None if final_checkpoint is None else str(final_checkpoint),
        "final_checkpoint_sha256": (
            None if final_checkpoint is None else _sha256(final_checkpoint)
        ),
        "trainable_parameters": list(trainable_parameters),
    }
    _write(output / "manifest.json", result)
    return result


def run(*, output: Path, selection_review_path: Path, checkpoint: Path,
        history_root: Path) -> dict[str, Any]:
    p = protocol()
    if _sha256(selection_review_path) != p["bindings"]["selection_review_sha256"]:
        raise ValueError("selection review SHA mismatch")
    review = json.loads(selection_review_path.read_text(encoding="utf-8"))
    validate_selection_review(review, p)
    if _sha256(checkpoint) != p["bindings"]["initial_checkpoint_sha256"]:
        raise ValueError("selected checkpoint SHA mismatch")
    output.mkdir(parents=True, exist_ok=False)
    all_seeds = {
        int(seed)
        for section in (p["qualification"], p["entry"], p["formal"])
        for seed in section["seeds"]
    }
    heldout = {int(seed) for seed in p["formal"]["heldout_seeds_reserved"]}
    history = historical_registry(history_root, output)
    if (all_seeds | heldout) & set(history["seeds"]):
        raise ValueError("P3 pipeline or heldout seeds were previously consumed")
    marker = checkpoint.parent / str(p["formal"]["reservation_marker"])
    reserve_once(marker, {
        "stage": STAGE, "protocol_sha256": PROTOCOL_SHA,
        "pipeline_seeds": sorted(all_seeds),
        "heldout_seeds_reserved": sorted(heldout),
        "output": str(output), "reserved_before_first_reset": True,
    })
    _write(output / "pre_reset_registry.json", {
        "stage": STAGE, "protocol_sha256": PROTOCOL_SHA,
        "pipeline_seeds": sorted(all_seeds),
        "heldout_seeds_reserved": sorted(heldout),
        "reservation_marker": str(marker), "reserved_before_first_reset": True,
    })
    qualification = _qualification(output / "qualification", checkpoint, p)
    entry = formal = None
    if qualification["PASS"]:
        entry = _training_stage(
            stage="entry", output=output / "entry", checkpoint=checkpoint, p=p,
        )
    if entry is not None and entry["PASS"]:
        formal = _training_stage(
            stage="formal", output=output / "formal", checkpoint=checkpoint, p=p,
        )
    passed = formal is not None and formal["PASS"]
    result = {
        "stage": STAGE, "PASS": bool(passed),
        "qualification": {
            "PASS": qualification["PASS"], "gates": qualification["gates"],
            "coverage": qualification["coverage"],
        },
        "entry": None if entry is None else {
            "PASS": entry["PASS"], "gates": entry["gates"],
        },
        "formal": None if formal is None else {
            "PASS": formal["PASS"], "gates": formal["gates"],
            "coverage": formal["coverage"],
            "final_checkpoint": formal["final_checkpoint"],
            "final_checkpoint_sha256": formal["final_checkpoint_sha256"],
        },
        "P3_FORMAL_TRAINING_COMPLETED": bool(passed),
        "HELDOUT_EVALUATION_ALLOWED": bool(passed),
        "P4_ALLOWED": False, "checkpoint_promotable": False,
        "heldout_seeds_reserved": sorted(heldout),
        "protocol_sha256": PROTOCOL_SHA,
        "source_hashes": source_hashes(),
        "selected_checkpoint_sha256": p["bindings"]["initial_checkpoint_sha256"],
        "selection_review_sha256": p["bindings"]["selection_review_sha256"],
    }
    _write(output / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-review", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--history-root", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = run(
        output=args.output, selection_review_path=args.selection_review,
        checkpoint=args.candidate_checkpoint, history_root=args.history_root,
    )
    print(json.dumps({
        "stage": result["stage"], "PASS": result["PASS"],
        "P3_FORMAL_TRAINING_COMPLETED": result["P3_FORMAL_TRAINING_COMPLETED"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
