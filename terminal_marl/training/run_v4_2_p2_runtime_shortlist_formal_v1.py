"""Formal PPO training for the qualified K=3 runtime shortlist boundary."""
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


STAGE = "V4_2_P2_RUNTIME_SHORTLIST_FORMAL_V1"
PROTOCOL = Path("terminal_marl/configs/low_v4_2/ppo_p2_runtime_shortlist_formal_v1.yaml")
PROTOCOL_SHA = "7e6cf765af93925ffe960ff2ab8a4f141e13c1101e8b2f933fd64a2296bf3048"


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
        raise ValueError("runtime-shortlist formal protocol changed")
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    if value.get("protocol") != STAGE:
        raise ValueError("runtime-shortlist formal protocol id changed")
    return value


def source_hashes() -> dict[str, str]:
    paths = (
        Path("terminal_marl/training/run_v4_2_p2_runtime_shortlist_formal_v1.py"),
        Path("terminal_marl/training/run_v4_2_p2_runtime_shortlist_entry_v1.py"),
        Path("terminal_marl/algo/joint_ppo_v4_2.py"),
        Path("terminal_marl/algo/student_ppo_v4_2.py"),
        Path("terminal_marl/algo/gate_calibrated_adapter_v4_2.py"),
        Path("terminal_marl/core/checkpoints.py"),
        Path("terminal_marl/core/low_level_joint_action_v4_2.py"),
        Path("terminal_marl/core/safety_trace.py"),
        Path("terminal_marl/envs/parallel_env.py"),
        Path("terminal_marl/training/student_rollout_v4_2.py"),
        Path("terminal_marl/training/run_v4_2_i2_dagger_preflight.py"),
    )
    return {path.as_posix(): _sha256(path) for path in paths}


def validate_entry(admission: dict[str, Any], p: dict[str, Any]) -> None:
    bindings = p["bindings"]
    if (admission.get("stage") != "V4_2_P2_RUNTIME_SHORTLIST_PPO_ENTRY_V1"
            or admission.get("PASS") is not True
            or not admission.get("gates")
            or not all(admission["gates"].values())
            or admission.get("PPO_ALLOWED") is not True
            or admission.get("formal_training_called") is not False
            or admission.get("entry_checkpoints_eligible_for_formal") is not False
            or admission.get("checkpoint_promotable") is not False):
        raise PermissionError("runtime-shortlist PPO entry has not passed")
    if (admission.get("protocol_sha256") != bindings["entry_protocol_sha256"]
            or admission.get("candidate_checkpoint_sha256")
            != bindings["candidate_checkpoint_sha256"]):
        raise ValueError("runtime-shortlist PPO entry binding changed")


def run(*, output: Path, admission_path: Path, checkpoint: Path,
        history_root: Path) -> dict[str, Any]:
    p = protocol()
    bindings, formal, optimizer = p["bindings"], p["formal"], p["optimizer"]
    if _sha256(admission_path) != bindings["entry_admission_sha256"]:
        raise ValueError("entry admission SHA mismatch")
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    validate_entry(admission, p)
    policy, _, trainable_parameters = _fresh_policy(checkpoint, p)
    seeds = [int(seed) for seed in formal["seeds"]]
    heldout = [int(seed) for seed in formal["heldout_evaluation_seeds_reserved"]]
    output.mkdir(parents=True, exist_ok=False)
    history = historical_registry(history_root, output)
    if (set(seeds) | set(heldout)) & set(history["seeds"]):
        raise ValueError("formal or reserved heldout seeds were previously consumed")
    marker = checkpoint.parent / str(formal["reservation_marker"])
    reserve_once(marker, {
        "stage": STAGE,
        "protocol_sha256": PROTOCOL_SHA,
        "formal_seeds": seeds,
        "heldout_evaluation_seeds_reserved": heldout,
        "output": str(output),
        "reserved_before_first_reset": True,
    })
    _write(output / "pre_reset_registry.json", {
        "stage": STAGE,
        "formal_seeds": seeds,
        "heldout_evaluation_seeds_reserved": heldout,
        "protocol_sha256": PROTOCOL_SHA,
        "candidate_checkpoint_sha256": bindings["candidate_checkpoint_sha256"],
        "reservation_marker": str(marker),
        "reserved_before_first_reset": True,
    })
    np.random.seed(seeds[0])
    records: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        scenario = str(formal["scenarios"][index])
        count = int(scenario.removeprefix("flow"))
        config = replace(p2_config(count), runtime_proposal_shortlist_k=3)
        horizon = int(config.max_simulation_seconds / config.dt)
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
            "all_requested_complete": summary["completed"] == config.total_requested_uavs,
            "deadline": 0 < summary["seconds"] <= config.max_simulation_seconds,
            "terminal_and_queue_drained": bool(summary["terminal"] and summary["queue"]
                                               and summary["queue"][-1] == 0),
            "shield_burden": shield_fraction <= float(
                p["hard_gates"]["shield_actor_fraction_max"]),
            "replay": bool(replay_exact),
            "requested_likelihood": bool(likelihood),
            "initial_ratio": initial_error <= float(
                p["hard_gates"]["initial_ratio_error_max"]),
            "student_only": int(summary.get("label_rows", 0)) == 0
                            and all(not step.teacher_used for step in steps),
            "stochastic": bool(steps) and all(step.stochastic_behavior for step in steps),
            "actor_decisions": actor_decisions > 0,
            "runtime_shortlist_contract": bool(shortlist["all"]),
            "geometry": geometry["reverse_samples"] <= int(
                p["hard_gates"]["geometry_reverse_samples_max"])
                and geometry["max_inner_excursion_m"] <= float(
                    p["hard_gates"]["geometry_inner_excursion_max_m"]),
            "geometry_observed": geometry["samples"] > 0,
        }
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
                    saved, policy, config, stage=STAGE, formal=True,
                    ppo_called=True, protocol_sha256=PROTOCOL_SHA,
                    source_hashes=source_hashes(),
                    initial_checkpoint_sha256=bindings["candidate_checkpoint_sha256"],
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
        record = {
            "seed": seed,
            "scenario": scenario,
            "horizon": horizon,
            "summary": summary,
            "geometry": geometry,
            "initial_logprob_error": initial_error,
            "shield_actor_fraction": shield_fraction,
            "runtime_shortlist_checks": shortlist,
            "update": update,
            "checkpoint": None if saved is None else str(saved),
            "gates": gates,
        }
        records.append(record)
        _write(output / "progress.json", {
            "stage": STAGE,
            "completed": len(records),
            "total": len(seeds),
            "records": records,
        })
        if not all(gates.values()):
            break
    final_checkpoint = Path(records[-1]["checkpoint"]) if records and records[-1]["checkpoint"] else None
    gates = {
        "all_episodes": len(records) == len(seeds),
        "all_episode_gates": bool(records)
                             and all(all(record["gates"].values()) for record in records),
    }
    passed = all(gates.values())
    result = {
        "stage": STAGE,
        "PASS": passed,
        "gates": gates,
        "records": records,
        "PPO_ALLOWED": passed,
        "formal_training_called": True,
        "HELDOUT_EVALUATION_ALLOWED": passed,
        "P3_ALLOWED": False,
        "checkpoint_promotable": False,
        "final_checkpoint": None if final_checkpoint is None else str(final_checkpoint),
        "final_checkpoint_sha256": None if final_checkpoint is None else _sha256(final_checkpoint),
        "protocol_sha256": PROTOCOL_SHA,
        "source_hashes": source_hashes(),
        "candidate_checkpoint_sha256": bindings["candidate_checkpoint_sha256"],
        "entry_admission_sha256": bindings["entry_admission_sha256"],
        "heldout_evaluation_seeds_reserved": heldout,
        "trainable_parameters": list(trainable_parameters),
        "scope": p["scope"],
    }
    _write(output / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--entry-admission", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--history-root", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = run(
        output=args.output,
        admission_path=args.entry_admission,
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
