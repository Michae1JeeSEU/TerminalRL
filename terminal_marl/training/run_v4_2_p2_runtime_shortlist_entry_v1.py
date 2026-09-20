"""Fresh PPO entry validation for the qualified K=3 runtime shortlist."""
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

from terminal_marl.algo.gate_calibrated_adapter_v4_2 import (
    CALIBRATION_TEMPERATURE,
    TemperatureCalibratedGateResidualAdapterStudentPPOV42,
)
from terminal_marl.core.checkpoints import load_checkpoint, save_checkpoint
from terminal_marl.core.low_level_joint_action_v4_2 import hierarchical_from_normalized_v4_2
from terminal_marl.core.safety_trace import SafetyTraceRecorder
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training import evaluate_v4_2_p2_gate_effectiveness_v1 as effectiveness
from terminal_marl.training.evaluate_v4_2_p2_gate_runtime_shortlist_v1 import _tape_shortlist_checks
from terminal_marl.training.run_v4_2_i2_dagger_preflight import _policy_tape_logprob_exact
from terminal_marl.training.run_v4_2_ppo_block2 import state_equal
from terminal_marl.training.run_v4_2_ppo_p2_resume_v4 import (
    make_update_gates, p2_config,
)
from terminal_marl.training.run_v4_2_v2_1_formal_i0_r4 import historical_registry
from terminal_marl.training.run_v4_2_v2_1_ppo import requested_logprob_error
from terminal_marl.training.student_rollout_v4_2 import rollout


STAGE = "V4_2_P2_RUNTIME_SHORTLIST_PPO_ENTRY_V1"
PROTOCOL = Path("terminal_marl/configs/low_v4_2/ppo_p2_runtime_shortlist_entry_v1.yaml")
PROTOCOL_SHA = "d3400bf47d0f7f0d86ce4ed16c5b547b6ab30e0220691eb1f33eef320af932be"
ENTRY_TRAINABLE_PREFIXES = ("gate_adapter.", "critic_encoder.", "critic_head.")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protocol() -> dict[str, Any]:
    if _sha256(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("runtime-shortlist PPO entry protocol changed")
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    if value.get("protocol") != STAGE:
        raise ValueError("runtime-shortlist PPO entry protocol id changed")
    return value


def source_hashes() -> dict[str, str]:
    paths = (
        Path(__file__),
        Path("terminal_marl/algo/joint_ppo_v4_2.py"),
        Path("terminal_marl/algo/student_ppo_v4_2.py"),
        Path("terminal_marl/algo/gate_calibrated_adapter_v4_2.py"),
        Path("terminal_marl/core/checkpoints.py"),
        Path("terminal_marl/core/low_level_joint_action_v4_2.py"),
        Path("terminal_marl/envs/parallel_env.py"),
        Path("terminal_marl/core/safety_trace.py"),
        Path("terminal_marl/training/student_rollout_v4_2.py"),
        Path("terminal_marl/training/run_v4_2_i2_dagger_preflight.py"),
    )
    return {path.as_posix(): _sha256(path) for path in paths}


def _policy_tape_shortlists_exact(policy, config, tape_path: str | Path) -> bool:
    """Require the post-update policy to emit the same deterministic tails.

    PPO accounts for the sampled rank-1 request.  The other K=3 candidates
    are deterministic functions of the proposal logits, so their recorded
    ordering must remain unchanged for the old action's likelihood ratio to
    keep representing the same executable shortlist.
    """
    payload = json.loads(Path(tape_path).read_text(encoding="utf-8"))
    env = TerminalParallelEnv(config)
    observations, _ = env.reset(int(payload["episode_seed"]))
    for row in payload["action_tape"]:
        shortlists = row.get("runtime_proposal_shortlists", {})
        for uid, values in shortlists.items():
            recorded = [
                hierarchical_from_normalized_v4_2(value).proposal
                for value in values
            ]
            selected = row.get("requested_proposal", {}).get(uid)
            if selected is None or not recorded or recorded[0] != int(selected):
                return False
            tensor = torch.as_tensor(
                np.asarray(observations[uid], dtype=np.float32)[None],
                dtype=torch.float32,
                device=policy.device,
            )
            with torch.no_grad():
                _, proposal_distribution = policy.model.distributions(tensor)
            ranked = torch.argsort(
                proposal_distribution.logits[0], descending=True, stable=True,
            ).tolist()
            expected = [int(selected)] + [
                int(value) for value in ranked if int(value) != int(selected)
            ]
            if recorded != expected[:len(recorded)]:
                return False
        actions = {
            uid: np.asarray(value, dtype=np.float32)
            for uid, value in row["a_requested"].items()
        }
        supplied = {
            str(uid): tuple(np.asarray(action, dtype=np.float32) for action in values)
            for uid, values in shortlists.items()
        }
        observations, *_ = env.step(actions, supplied)
    return True


def validate_review(report: dict[str, Any], p: dict[str, Any]) -> None:
    bindings = p["bindings"]
    if (report.get("stage") != "V4_2_P2_RUNTIME_SHORTLIST_ADMISSION_REVIEW_V1"
            or report.get("PASS") is not True
            or not report.get("gates") or not all(report["gates"].values())
            or report.get("ENTRY_PROTOCOL_PREREGISTRATION_ALLOWED") is not True
            or report.get("ENTRY_VALIDATION_ALLOWED") is not False
            or report.get("PPO_ALLOWED") is not False):
        raise PermissionError("runtime-shortlist admission review has not passed")
    if (report.get("protocol_sha256") != bindings["admission_review_protocol_sha256"]
            or report.get("candidate_checkpoint_sha256") != bindings["candidate_checkpoint_sha256"]):
        raise ValueError("runtime-shortlist admission review binding changed")


def _configure_entry_optimizer(policy, learning_rate: float) -> tuple[str, ...]:
    trainable = []
    for name, parameter in policy.model.named_parameters():
        enabled = name.startswith(ENTRY_TRAINABLE_PREFIXES)
        parameter.requires_grad_(enabled)
        if enabled:
            trainable.append(name)
    if not trainable:
        raise RuntimeError("entry trainable parameter scope is empty")
    policy.optimizer = torch.optim.Adam(
        [parameter for parameter in policy.model.parameters() if parameter.requires_grad],
        lr=float(learning_rate),
    )
    return tuple(trainable)


def _frozen_state_equal(before: dict[str, Any], after: dict[str, Any]) -> bool:
    frozen = {
        key for key in before
        if not key.startswith(ENTRY_TRAINABLE_PREFIXES)
    }
    return frozen == {
        key for key in after
        if not key.startswith(ENTRY_TRAINABLE_PREFIXES)
    } and all(state_equal(before[key], after[key]) for key in frozen)


def _fresh_policy(checkpoint: Path, p: dict[str, Any]):
    bindings, optimizer = p["bindings"], p["optimizer"]
    if _sha256(checkpoint) != bindings["candidate_checkpoint_sha256"]:
        raise ValueError("candidate checkpoint SHA mismatch")
    policy, payload = effectiveness.load_candidate_checkpoint(checkpoint)
    trainable = _configure_entry_optimizer(policy, float(optimizer["learning_rate"]))
    if policy.optimizer.state:
        raise RuntimeError("entry must start with a fresh optimizer")
    return policy, payload, trainable


def _restored_policy(checkpoint: Path, config, p: dict[str, Any]):
    optimizer = p["optimizer"]
    restored = TemperatureCalibratedGateResidualAdapterStudentPPOV42(
        88, temperature=CALIBRATION_TEMPERATURE,
        learning_rate=float(optimizer["learning_rate"]),
    )
    _configure_entry_optimizer(restored, float(optimizer["learning_rate"]))
    load_checkpoint(checkpoint, restored, config, load_optimizer=True)
    return restored


def run(*, output: Path, review_path: Path, checkpoint: Path, history_root: Path) -> dict[str, Any]:
    p = protocol()
    bindings, entry, optimizer = p["bindings"], p["entry"], p["optimizer"]
    if _sha256(review_path) != bindings["admission_review_sha256"]:
        raise ValueError("admission review SHA mismatch")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    validate_review(review, p)
    policy, _, trainable_parameters = _fresh_policy(checkpoint, p)
    seeds = [int(seed) for seed in entry["seeds"]]
    output.mkdir(parents=True, exist_ok=False)
    if set(seeds) & set(historical_registry(history_root, output)["seeds"]):
        raise ValueError("entry seeds were previously consumed")
    (output / "pre_reset_registry.json").write_text(json.dumps({
        "stage": STAGE, "seeds": seeds, "protocol_sha256": PROTOCOL_SHA,
        "candidate_checkpoint_sha256": bindings["candidate_checkpoint_sha256"],
        "reserved_before_first_reset": True,
    }, indent=2, sort_keys=True), encoding="utf-8")
    np.random.seed(seeds[0])
    records: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        scenario = str(entry["scenarios"][index])
        count = int(scenario.removeprefix("flow"))
        horizon = int(entry["steps"][index])
        config = replace(
            p2_config(count), runtime_proposal_shortlist_k=3,
            max_simulation_seconds=float(horizon),
        )
        steps, summary = rollout(
            policy, config, seed=seed, max_steps=horizon,
            output=output / f"seed{seed}", deterministic=False,
            collect_labels=False,
        )
        initial_error = requested_logprob_error(policy, steps)
        replay = SafetyTraceRecorder.replay_action_tape(
            TerminalParallelEnv(config), summary["action_tape"],
        )
        likelihood = _policy_tape_logprob_exact(policy, config, summary["action_tape"])
        shortlist = _tape_shortlist_checks(summary["action_tape"], expected_k=3)
        gates = {
            "safety": not any(summary["safety"].values()),
            "replay": bool(replay["exact"]),
            "requested_likelihood": bool(likelihood),
            "initial_ratio": initial_error <= float(p["hard_gates"]["initial_ratio_error_max"]),
            "stochastic_student": bool(steps) and all(step.stochastic_behavior and not step.teacher_used for step in steps),
            "actor_decisions": int(summary["counts"]["actor_decisions"]) > 0,
            "runtime_shortlist_contract": bool(shortlist["all"]),
            "teacher_free": int(summary.get("label_rows", 0)) == 0,
        }
        update = None
        saved = None
        if all(gates.values()):
            model_snapshot = copy.deepcopy(policy.model.state_dict())
            optimizer_snapshot = copy.deepcopy(policy.optimizer.state_dict())
            keys = ("minibatch_size", "gamma", "gae_lambda", "clip_ratio", "value_coef", "entropy_coef")
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
            saved = output / f"entry_checkpoint_{index + 1}.pt"
            save_checkpoint(
                saved, policy, config, stage=STAGE, formal=False,
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
        record = {
            "seed": seed, "scenario": scenario, "horizon": horizon,
            "summary": summary, "initial_logprob_error": initial_error,
            "runtime_shortlist_checks": shortlist, "update": update,
            "checkpoint": None if saved is None else str(saved), "gates": gates,
        }
        records.append(record)
        (output / "progress.json").write_text(json.dumps({
            "completed": len(records), "total": len(seeds), "records": records,
        }, indent=2, sort_keys=True), encoding="utf-8")
        if not all(gates.values()):
            break
    gates = {
        "all_entries": len(records) == len(seeds),
        "all_entry_gates": bool(records) and all(all(row["gates"].values()) for row in records),
    }
    passed = all(gates.values())
    result = {
        "stage": STAGE, "PASS": passed, "gates": gates, "records": records,
        "PPO_ALLOWED": passed, "formal_training_called": False,
        "entry_checkpoints_eligible_for_formal": False,
        "P3_ALLOWED": False, "checkpoint_promotable": False,
        "protocol_sha256": PROTOCOL_SHA, "source_hashes": source_hashes(),
        "candidate_checkpoint_sha256": bindings["candidate_checkpoint_sha256"],
        "admission_review_sha256": bindings["admission_review_sha256"],
        "trainable_parameters": list(trainable_parameters),
        "scope": p["scope"],
    }
    (output / "admission.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--history-root", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = run(
        output=args.output, review_path=args.review,
        checkpoint=args.candidate_checkpoint, history_root=args.history_root,
    )
    print(json.dumps({"stage": result["stage"], "PASS": result["PASS"],
                      "PPO_ALLOWED": result["PPO_ALLOWED"]}, indent=2))


if __name__ == "__main__":
    main()
