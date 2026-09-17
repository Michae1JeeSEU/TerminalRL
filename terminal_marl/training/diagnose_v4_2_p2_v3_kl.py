"""Diagnostic-only paired PPO update for the stopped P2 V3 entry.

This module intentionally does not alter the formal PPO kernel or protocol. It
rolls out once with the V3 starting checkpoint, then applies the real
StudentPPOV42.update to the same immutable steps with the restored Adam state
and with a fresh zero-state Adam optimizer. Outputs are analysis artifacts and
must never be used to continue formal training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from terminal_marl.algo.joint_ppo_v4_2 import hierarchical_joint_logprob
from terminal_marl.algo.student_ppo_v4_2 import StudentPPOV42, StudentStepV42, team_gae
from terminal_marl.core.checkpoints import load_checkpoint
from terminal_marl.training.run_v4_2_ppo_block2 import (
    replay_geometry,
    restore,
    state_equal,
)
from terminal_marl.training.run_v4_2_ppo_p2_resume_v3 import p2_config
from terminal_marl.training.run_v4_2_v2_1_i1_q_cohort import (
    _consume_object,
    _history_files,
    _identity_accumulator,
)
from terminal_marl.training.run_v4_2_v2_1_ppo import requested_logprob_error
from terminal_marl.training.run_v4_2_i2_dagger_preflight import _policy_tape_logprob_exact
from terminal_marl.training.student_rollout_v4_2 import rollout


PROTOCOL = Path("terminal_marl/configs/low_v4_2/ppo_p2_v3_kl_diagnostic_v1.yaml")
OBSERVATION_DIM = 88
ACTOR_PREFIXES = ("actor_encoder", "gate_head", "proposal_head")
CRITIC_PREFIXES = ("critic_encoder", "critic_head")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _finite(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    return bool(np.isfinite(value))


def _state_l2(policy: StudentPPOV42, prefixes: tuple[str, ...]) -> float:
    return float(sum(float(parameter.detach().float().square().sum())
                     for name, parameter in policy.model.named_parameters()
                     if name.startswith(prefixes)) ** 0.5)


def _state_snapshot(policy: StudentPPOV42) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().clone() for name, parameter in policy.model.named_parameters()}


def _state_delta(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor], prefixes: tuple[str, ...]) -> dict[str, float]:
    values = [after[name].float() - before[name].float() for name in before if name.startswith(prefixes)]
    return {
        "l2": float(sum(float(value.square().sum()) for value in values) ** 0.5),
        "max_abs": float(max((float(value.abs().max()) for value in values), default=0.0)),
    }


def _plan_batches(length: int, epochs: int, minibatch_size: int) -> list[np.ndarray]:
    batches: list[np.ndarray] = []
    for _ in range(epochs):
        permutation = np.random.permutation(length)
        batches.extend(permutation[start:start + minibatch_size]
                       for start in range(0, length, minibatch_size))
    return batches


def _plan_sha256(plan: list[np.ndarray]) -> str:
    payload = json.dumps([batch.tolist() for batch in plan], separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _actor_rows(steps: list[StudentStepV42], indices: np.ndarray) -> tuple[list[StudentStepV42], list[int]]:
    batch = [steps[int(index)] for index in indices]
    active = [row for row in batch if bool(row.actor_mask.any())]
    counts = [int(row.actor_mask.sum()) for row in active]
    return active, counts


def _requested_kl(policy: StudentPPOV42, steps: list[StudentStepV42], indices: np.ndarray | None = None) -> float:
    if indices is None:
        indices = np.arange(len(steps), dtype=int)
    active, counts = _actor_rows(steps, indices)
    if not active:
        return 0.0
    observations = np.concatenate([row.world[row.actor_mask] for row in active])
    gates = np.concatenate([row.requested_gate[row.actor_mask] for row in active])
    proposals = np.concatenate([row.requested_proposal[row.actor_mask] for row in active])
    old = np.concatenate([row.requested_logprob[row.actor_mask] for row in active])
    with torch.no_grad():
        tensor_obs = torch.as_tensor(observations, dtype=torch.float32, device=policy.device)
        gate_dist, proposal_dist = policy.model.distributions(tensor_obs)
        _, _, current = hierarchical_joint_logprob(
            gate_dist,
            proposal_dist,
            torch.as_tensor(gates, dtype=torch.long, device=policy.device),
            torch.as_tensor(proposals, dtype=torch.long, device=policy.device),
        )
        logratio = current - torch.as_tensor(old, dtype=torch.float32, device=policy.device)
        ratio = logratio.exp()
        chunks = torch.split((ratio - 1.) - logratio, counts)
        return float(torch.stack([chunk.mean() for chunk in chunks]).mean().cpu())


def _gae_stats(steps: list[StudentStepV42], gamma: float, gae_lambda: float) -> dict[str, float | int]:
    raw, _ = team_gae(steps, gamma, gae_lambda)
    actor_steps = np.asarray([bool(row.actor_mask.any()) for row in steps])
    actor = raw[actor_steps]
    if len(actor):
        normalized = (raw - actor.mean()) / (actor.std() + 1e-8)
        normalized_actor = normalized[actor_steps]
        return {
            "actor_time_steps": int(len(actor)),
            "actor_agent_decisions": int(sum(int(row.actor_mask.sum()) for row in steps)),
            "raw_actor_gae_mean": float(actor.mean()),
            "raw_actor_gae_std": float(actor.std()),
            "normalized_actor_gae_mean": float(normalized_actor.mean()),
            "normalized_actor_gae_std": float(normalized_actor.std()),
        }
    return {
        "actor_time_steps": 0,
        "actor_agent_decisions": 0,
        "raw_actor_gae_mean": 0.0,
        "raw_actor_gae_std": 0.0,
        "normalized_actor_gae_mean": 0.0,
        "normalized_actor_gae_std": 0.0,
    }


def _run_update(policy: StudentPPOV42, steps: list[StudentStepV42], protocol: dict, seed: int) -> dict:
    epochs = int(protocol["update_epochs"])
    minibatch_size = int(protocol["minibatch_size"])
    np.random.seed(seed)
    random_state = np.random.get_state()
    plan = _plan_batches(len(steps), epochs, minibatch_size)
    np.random.set_state(random_state)
    records: list[dict[str, object]] = []
    state = {"batch": 0, "current": None}
    original_clip = torch.nn.utils.clip_grad_norm_
    original_step = policy.optimizer.step

    def clip_and_trace(parameters, *args, **kwargs):
        index = state["batch"]
        indices = plan[index]
        active, counts = _actor_rows(steps, indices)
        actor_agents = int(sum(counts))
        record = {
            "batch": int(index),
            "epoch": int(index // ((len(steps) + minibatch_size - 1) // minibatch_size)),
            "worlds": int(len(indices)),
            "actor_worlds": int(len(active)),
            "actor_agent_decisions": actor_agents,
            "batch_requested_kl_before_step": _requested_kl(policy, steps, indices),
        }
        result = original_clip(parameters, *args, **kwargs)
        record["grad_norm_pre_clip"] = float(result)
        state["current"] = record
        return result

    def step_and_trace(*args, **kwargs):
        before = _state_snapshot(policy)
        result = original_step(*args, **kwargs)
        after = _state_snapshot(policy)
        record = state["current"]
        if record is None:
            raise RuntimeError("optimizer step did not follow a clipped update")
        record["actor_parameter_delta"] = _state_delta(before, after, ACTOR_PREFIXES)
        record["critic_parameter_delta"] = _state_delta(before, after, CRITIC_PREFIXES)
        records.append(record)
        state["batch"] += 1
        state["current"] = None
        return result

    torch.nn.utils.clip_grad_norm_ = clip_and_trace
    policy.optimizer.step = step_and_trace
    try:
        update = policy.update(
            steps,
            training_authorized=True,
            epochs=epochs,
            minibatch_size=minibatch_size,
            gamma=float(protocol["gamma"]),
            gae_lambda=float(protocol["gae_lambda"]),
            clip_ratio=float(protocol["clip_ratio"]),
            value_coef=float(protocol["value_coef"]),
            entropy_coef=float(protocol["entropy_coef"]),
        )
    finally:
        torch.nn.utils.clip_grad_norm_ = original_clip
        policy.optimizer.step = original_step
    if len(records) != int(update["updates"]):
        raise RuntimeError("optimizer trace count differs from PPO update count")
    update = {key: float(value) if _finite(value) else value for key, value in update.items()}
    return {
        "update": update,
        "trace": records,
        "batch_plan_sha256": _plan_sha256(plan),
        "trace_mean_kl": float(np.mean([row["batch_requested_kl_before_step"] for row in records])) if records else 0.0,
        "trace_actor_batches": int(sum(bool(row["actor_agent_decisions"]) for row in records)),
    }


def _fresh_zero_adam(checkpoint: Path, config, learning_rate: float, reference: StudentPPOV42) -> StudentPPOV42:
    policy = StudentPPOV42(OBSERVATION_DIM, learning_rate=learning_rate)
    load_checkpoint(checkpoint, policy, config, load_optimizer=False)
    if not state_equal(reference.model.state_dict(), policy.model.state_dict()):
        raise RuntimeError("fresh Adam policy does not match restored model")
    if policy.optimizer.state_dict()["state"]:
        raise RuntimeError("fresh Adam unexpectedly has optimizer state")
    return policy


def _run_case(protocol: dict, checkpoint: Path, output: Path, seed: int, scenario: str, steps_limit: int, history_root: Path) -> dict:
    config = p2_config(int(scenario[4:]))
    restored = restore(checkpoint, protocol)
    seed_dir = output / f"seed{seed}"
    seed_dir.mkdir(parents=False, exist_ok=False)
    rollout_dir = seed_dir / "rollout"
    steps, summary = rollout(restored, config, seed=seed, max_steps=steps_limit, output=rollout_dir,
                             deterministic=False, collect_labels=False)
    replay_exact, geometry = replay_geometry(config, summary["action_tape"])
    requested_exact = _policy_tape_logprob_exact(restored, config, summary["action_tape"])
    initial_error = requested_logprob_error(restored, steps)
    if not steps:
        raise RuntimeError("diagnostic rollout produced no steps")
    gae = _gae_stats(steps, float(protocol["gamma"]), float(protocol["gae_lambda"]))
    fresh = _fresh_zero_adam(checkpoint, config, float(protocol["learning_rate"]), restored)
    conditions = {}
    for name, policy in (("restored_adam", restored), ("fresh_zero_adam", fresh)):
        before = _state_snapshot(policy)
        before_kl = _requested_kl(policy, steps)
        result = _run_update(policy, steps, protocol, seed)
        after = _state_snapshot(policy)
        conditions[name] = {
            "initial_full_rollout_requested_kl": before_kl,
            "final_full_rollout_requested_kl": _requested_kl(policy, steps),
            "parameter_delta": {
                "actor": _state_delta(before, after, ACTOR_PREFIXES),
                "critic": _state_delta(before, after, CRITIC_PREFIXES),
            },
            **result,
        }
        torch.save({
            "diagnostic_only": True,
            "formal_training_allowed": False,
            "checkpoint_promotion_allowed": False,
            "condition": name,
            "seed": seed,
            "scenario": scenario,
            "model": policy.model.state_dict(),
            "optimizer": policy.optimizer.state_dict(),
        }, seed_dir / f"{name}_state_snapshot.pt")
    return {
        "seed": seed,
        "scenario": scenario,
        "steps_limit": steps_limit,
        "rollout": summary,
        "safety_pass": not any(summary["safety"].values()),
        "replay_exact": bool(replay_exact),
        "requested_logprob_exact": bool(requested_exact),
        "initial_logprob_error": float(initial_error),
        "geometry": geometry,
        "gae": gae,
        "conditions": conditions,
        "history_root": str(history_root),
    }


def _historical_seed_check(history_root: Path, output: Path, seeds: list[int]) -> dict[str, int | bool]:
    # Reuse the project's registry parser, but do not call its writer. This
    # keeps the diagnostic independent while applying the same seed semantics.
    identities = _identity_accumulator()
    files = _history_files(history_root, output)
    for path in files:
        try:
            if path.suffix.lower() == ".jsonl":
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        try:
                            _consume_object(json.loads(line), identities)
                        except json.JSONDecodeError:
                            pass
            else:
                content = path.read_text(encoding="utf-8")
                try:
                    _consume_object(json.loads(content), identities)
                except json.JSONDecodeError:
                    for line in content.splitlines():
                        try:
                            _consume_object(json.loads(line), identities)
                        except json.JSONDecodeError:
                            pass
        except (OSError, UnicodeDecodeError):
            continue
    consumed = sorted(set(seeds) & identities["seeds"])
    if consumed:
        raise ValueError(f"diagnostic seed(s) already consumed: {consumed}")
    return {"history_files": len(files), "historical_seed_count": len(identities["seeds"]), "fresh": True}


def run(args: argparse.Namespace) -> dict:
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8-sig"))
    if not protocol.get("diagnostic_only") or protocol.get("formal_training_allowed"):
        raise ValueError("diagnostic protocol must remain non-formal")
    if _sha(args.checkpoint) != protocol["initial_checkpoint_sha256"]:
        raise ValueError("diagnostic checkpoint hash mismatch")
    if args.output.exists():
        raise FileExistsError(f"diagnostic output already exists: {args.output}")
    if args.smoke:
        seeds = [int(protocol["smoke_seed"])]
        scenarios = [str(protocol["smoke_scenario"])]
        limits = [int(protocol["smoke_steps"])]
    else:
        seeds = [int(value) for value in protocol["seeds"]]
        scenarios = [str(value) for value in protocol["scenarios"]]
        limits = [int(value) for value in protocol["steps"]]
    if len(seeds) != len(scenarios) or len(seeds) != len(limits):
        raise ValueError("diagnostic seed/scenario/step lengths differ")
    registry = _historical_seed_check(args.history_root, args.output, seeds)
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for seed, scenario, limit in zip(seeds, scenarios, limits):
        results.append(_run_case(protocol, args.checkpoint, args.output, seed, scenario, limit, args.history_root))
    result = {
        "protocol": protocol["protocol"],
        "protocol_sha256": _sha(PROTOCOL),
        "diagnostic_only": True,
        "formal_training_allowed": False,
        "checkpoint_promotion_allowed": False,
        "initial_checkpoint": str(args.checkpoint),
        "initial_checkpoint_sha256": _sha(args.checkpoint),
        "registry": registry,
        "source_hashes": {
            "student_ppo_v4_2.py": _sha(Path("terminal_marl/algo/student_ppo_v4_2.py")),
            "joint_ppo_v4_2.py": _sha(Path("terminal_marl/algo/joint_ppo_v4_2.py")),
            "student_rollout_v4_2.py": _sha(Path("terminal_marl/training/student_rollout_v4_2.py")),
            "trajectory_generator.py": _sha(Path("terminal_marl/core/trajectory_generator.py")),
            "diagnostic_runner": _sha(Path(__file__)),
        },
        "results": results,
    }
    _json_write(args.output / "manifest.json", result)
    _json_write(args.output / "progress.json", {"stage": "diagnostic", "PASS": True, "completed": len(results), "total": len(results)})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--history-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    torch.set_num_threads(1)
    result = run(parser.parse_args())
    print(json.dumps({"protocol": result["protocol"], "PASS": True, "results": len(result["results"])}, sort_keys=True), flush=True)
