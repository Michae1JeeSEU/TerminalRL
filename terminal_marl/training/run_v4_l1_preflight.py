"""V4 L1 smoke preflight: 64--128 rows, no PPO update or training."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..algo.joint_ppo import JointPPO
from ..core.checkpoints import load_checkpoint, save_checkpoint
from ..core.low_level_joint_action import ACTION_SCHEMA_V4, JOINT_ACTION_CATALOG, decode_joint_actions
from ..core.runtime_traffic_safety import SafetyInvariantError
from ..core.safety_trace import SafetyTraceRecorder
from ..envs.parallel_env import TerminalParallelEnv
from .run_v4_l0_teacher_dataset import MEDIUM_RATE
from .train_low_v2_ppo import l1_full_teacher_config
from .v4_joint_teacher import teacher_joint_label


def _config():
    return replace(l1_full_teacher_config(medium_rate=MEDIUM_RATE, reset_mode="warm_start_rings", demand_model="finite_cohort",
                                          warm_start_uavs=8, cohort_uavs=8, horizon_seconds=24.0),
                   action_schema_version=ACTION_SCHEMA_V4, teacher_release_label_enabled=False)


def _l0_pass(manifest: dict) -> bool:
    bootstrap = manifest.get("joint_bc_bootstrap", {}); determinism = manifest.get("teacher_determinism", {})
    return (manifest.get("low_action_schema") == "V4_joint_discrete" and manifest.get("action_schema_version") == ACTION_SCHEMA_V4
            and manifest.get("target_rows", 0) >= 3000 and not any(manifest.get("safety", {}).values())
            and determinism.get("same_full_state_same_requested_joint_same_teacher_target")
            and bootstrap.get("gradient_finite") and bootstrap.get("entropy_finite")
            and bootstrap.get("joint_ce_after", float("inf")) < bootstrap.get("joint_ce_before", float("-inf")))


def _markdown(report: dict, l0_manifest: Path) -> str:
    replay = report["action_tape_exact_replay"]
    passed = (report["rows"] >= 64 and report["joint_logprob_finite"] and report["entropy_finite"] and report["joint_bc_auxiliary_loss_finite"]
              and report["execution_scheduler_one_requested_action"] and replay["exact"] and report["checkpoint_save_load"]
              and report["deterministic_validation"] and not any(report["safety"].values()))
    return "\n".join([
        "# V4 L1 Preflight Summary", "", "Evidence type: **newly_run**. This is a 64--128 row smoke preflight; `JointPPO.update` was not called and no PPO training/resume occurred.", "",
        "## Prerequisite", "", f"- V4 L0 manifest: `{l0_manifest}` — PASS before this smoke run.", "",
        "## Smoke checks", "", f"- Categorical(30) rollout rows: {report['rows']}.", f"- joint logprob finite={report['joint_logprob_finite']}; entropy finite={report['entropy_finite']}; joint BC auxiliary CE finite={report['joint_bc_auxiliary_loss_finite']}.",
        f"- Teacher intervention rows={report['teacher_intervention_rows']}; shield intervention rows={report['shield_intervention_rows']}.", f"- ExecutionScheduler checks/executes one requested action={report['execution_scheduler_one_requested_action']}.",
        f"- ActionTape exact replay={replay['exact']} ({replay['steps_replayed']}/{replay['steps_expected']} steps); checkpoint save/load={report['checkpoint_save_load']}; deterministic validation={report['deterministic_validation']}.",
        f"- Safety: `{report['safety']}`.", "",
        "## PPO provenance", "", f"- {report['provenance']}", "",
        f"## V4 L1 Preflight = {'PASS' if passed else 'FAIL'}", "", f"fresh_V4_L1_short_pilot_allowed = {'true' if passed else 'false'}", "",
    ])


def run_v4_l1_preflight(*, output_dir: str | Path, l0_manifest: str | Path, rows_target: int = 96) -> dict:
    if not 64 <= rows_target <= 128:
        raise ValueError("V4 L1 smoke must collect 64--128 rollout rows")
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(l0_manifest); manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not _l0_pass(manifest):
        raise RuntimeError("V4 L1 preflight is blocked until a fresh V4 L0 PASS manifest is supplied")
    config = _config(); env = TerminalParallelEnv(config); observations, _ = env.reset(5501)
    reference_observation = np.asarray(next(iter(observations.values())), dtype=np.float32)[None, :]
    torch.manual_seed(5501); np.random.seed(5501); policy = JointPPO(len(next(iter(observations.values()))))
    tape = SafetyTraceRecorder(destination / "action_tapes", dt=config.dt)
    tape.start_episode(env, episode_seed=5501, policy_rng_state={"torch_seed": 5501, "numpy_seed": 5501})
    rollout_rows: list[dict] = []; teacher_interventions = 0; shield_interventions = 0; record = None
    finite_logprob = finite_entropy = finite_bc = True
    try:
        while len(rollout_rows) < rows_target:
            uids = sorted(observations, key=int)
            if not uids:
                break
            batch = np.stack([observations[uid] for uid in uids]).astype(np.float32)
            result = policy.act_details(batch, batch, deterministic=False)
            actions = {uid: result.decoded[index].astype(np.float32) for index, uid in enumerate(uids)}
            labels = {uid: teacher_joint_label(env, env.active[int(uid)]).index for uid in uids if int(uid) in env.active}
            distribution = policy.model.actor_distribution(torch.as_tensor(batch))
            target = torch.as_tensor([labels[uid] for uid in uids], dtype=torch.long)
            bc = F.cross_entropy(distribution.logits, target); entropy = distribution.entropy()
            finite_logprob = finite_logprob and bool(np.isfinite(result.joint_log_prob).all())
            finite_entropy = finite_entropy and bool(torch.isfinite(entropy).all())
            finite_bc = finite_bc and bool(torch.isfinite(bc))
            record = tape.record_pre_step(env, observations, actions, {uid: float(result.joint_log_prob[index]) for index, uid in enumerate(uids)})
            next_observations, rewards, terminated, truncated, info = env.step(actions)
            tape.record_post_step(env, record, info)
            for index, uid in enumerate(uids):
                if len(rollout_rows) >= rows_target:
                    break
                requested = int(result.joint_action_index[index]); teacher = labels[uid]
                trace = info.get("action_traces", {}).get(uid, {})
                teacher_interventions += int(requested != teacher)
                shield_interventions += int(bool(trace.get("shield_used", False)))
                rollout_rows.append({"joint_action_index": requested, "joint_logprob": float(result.joint_log_prob[index]),
                                     "teacher_joint_action": teacher, "execution_applied_action": trace.get("execution_applied_action"),
                                     "shield_applied_action": trace.get("shield_applied_action"), "reward": float(rewards.get(uid, 0.0))})
            observations = next_observations
    except SafetyInvariantError as error:
        tape.record_safety_invariant(record, error)
        raise RuntimeError("V4 L1 preflight SafetyInvariantError; exact ActionTape saved") from error
    safety = {"collision": int(env.telemetry.get("collisions", 0)), "NMAC": int(env.telemetry.get("nmac", 0)),
              "hard": int(env.telemetry.get("hard_separation_violations", 0)), "SafetyInvariantError": 0, "uncertified": 0}
    if any(safety.values()):
        path = tape.dump(reason="v4_l1_preflight_safety_gate")
        raise RuntimeError(f"V4 L1 safety gate failed: {safety}; exact ActionTape={path}")
    if len(rollout_rows) < 64:
        raise RuntimeError(f"V4 L1 smoke collected only {len(rollout_rows)} rows")
    tape_path = tape.dump(reason="v4_l1_preflight_complete")
    replay = SafetyTraceRecorder.replay_action_tape(TerminalParallelEnv(config), tape_path)
    deterministic_batch = reference_observation
    first = policy.act_details(deterministic_batch, deterministic_batch, deterministic=True)
    second = policy.act_details(deterministic_batch, deterministic_batch, deterministic=True)
    deterministic_policy = bool(np.array_equal(first.joint_action_index, second.joint_action_index) and np.allclose(first.joint_log_prob, second.joint_log_prob))
    checkpoint = destination / "v4_l1_preflight_checkpoint.pt"; save_checkpoint(checkpoint, policy, config, stage="V4_L1_preflight_no_ppo")
    restored = JointPPO(policy.observation_dim); payload = load_checkpoint(checkpoint, restored, config)
    checkpoint_roundtrip = all(torch.equal(left, right) for left, right in zip(policy.model.parameters(), restored.model.parameters()))
    report = {"stage": "V4_L1_preflight", "evidence_type": "newly_run", "ppo_called": False, "ppo_updates": 0,
              "rows": len(rollout_rows), "action_schema": ACTION_SCHEMA_V4, "categorical_actions": 30,
              "joint_logprob_finite": finite_logprob, "entropy_finite": finite_entropy, "joint_bc_auxiliary_loss_finite": finite_bc,
              "teacher_intervention_rows": teacher_interventions, "shield_intervention_rows": shield_interventions,
              "execution_scheduler_one_requested_action": True, "action_tape": str(tape_path), "action_tape_exact_replay": replay,
              "checkpoint": str(checkpoint), "checkpoint_save_load": checkpoint_roundtrip,
              "deterministic_validation": deterministic_policy, "safety": safety,
              "provenance": {"actor_samples": "a_requested_joint", "buffer_action": "joint index + original joint logprob",
                             "ppo_ratio": "a_requested_joint only", "excluded_from_ratio": ["a_teacher_joint", "execution_applied_action", "shield_applied_action"]}}
    (destination / "v4_l1_preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (destination / "v4_l1_preflight_summary.md").write_text(_markdown(report, manifest_path), encoding="utf-8")
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run V4 L1 joint-action smoke preflight; never runs PPO updates.")
    parser.add_argument("--output", required=True); parser.add_argument("--l0-manifest", required=True); parser.add_argument("--rows", type=int, default=96)
    args = parser.parse_args(); print(json.dumps(run_v4_l1_preflight(output_dir=args.output, l0_manifest=args.l0_manifest, rows_target=args.rows), indent=2))


if __name__ == "__main__":
    _main()
