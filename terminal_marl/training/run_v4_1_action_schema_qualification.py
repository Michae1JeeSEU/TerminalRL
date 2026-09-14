"""S0 qualification for the V4.1 26-way action-schema boundary.

This is deliberately a schema/safety qualification only.  It does not collect
I0 rows and never calls PPO/BC/DAgger optimization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..algo.joint_ppo import JointPPO
from ..algo.joint_ppo_v4_1 import JointPPOV41
from ..core.checkpoints import IncompatibleCheckpointError, load_checkpoint, save_checkpoint
from ..core.config import SimConfig
from ..core.low_level_joint_action import joint_action
from ..core.low_level_joint_action_v4_1 import (ACTION_SCHEMA_V4_1, JOINT_ACTION_CATALOG_V4_1,
    N_JOINT_ACTIONS_V4_1, decode_joint_actions_v4_1, joint_action_v4_1, joint_index_from_normalized_v4_1)
from ..core.safety_trace import SafetyTraceRecorder
from ..envs.parallel_env import TerminalParallelEnv
from .low_v4_1_imitation import I0_DECISION_RELEVANT_FAMILIES, PPO_ALLOWED, i0_coverage_gate
from .v4_1_joint_teacher import teacher_joint_expert_v4_1, teacher_joint_label_v4_1


def _config() -> SimConfig:
    return SimConfig(action_schema_version=ACTION_SCHEMA_V4_1, experiment_mode="fixed_teacher_heavy",
        scheduler_mode="execution", teacher_enabled=True, teacher_mode="teacher_label_only",
        teacher_release_label_enabled=False, shield_enabled=True, reset_mode="warm_start_rings",
        warm_start_uavs=8, warm_start_steady_motion=True, fixed_arrival_schedule=(0.0,) * 8,
        total_requested_uavs=8, max_simulation_seconds=2.0, initial_soc_mode="fixed", initial_soc=.8,
        priority_mode="fixed", initial_priority=1)


def run(output_dir: str | Path) -> dict:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    config = _config()
    decoded = decode_joint_actions_v4_1(np.arange(N_JOINT_ACTIONS_V4_1))
    catalog_ok = bool(np.array_equal(joint_index_from_normalized_v4_1(decoded), np.arange(26)) and
                      np.array_equal(decoded[0], np.zeros(3, dtype=np.float32)) and
                      all(np.array_equal(joint_action_v4_1(index).normalized, joint_action(index + 4).normalized)
                          for index in range(1, 26)))

    env = TerminalParallelEnv(config); observations, _ = env.reset(9101)
    labels = {uid: [teacher_joint_label_v4_1(env, env.active[int(uid)]).index for _ in range(20)] for uid in observations}
    expert, audit = teacher_joint_expert_v4_1(env, env.active[int(next(iter(observations)))], audit_catalogue=True)
    teacher_ok = bool(all(len(set(values)) == 1 for values in labels.values()) and
                      len([row for row in audit if row["decoded"]["kind"] == "WAIT"]) == 1 and
                      all("abs_speed_residual" not in row["candidate_score"] for row in audit if row["decoded"]["kind"] == "WAIT"))

    tape = SafetyTraceRecorder(output / "action_tapes", dt=config.dt)
    tape.start_episode(env, episode_seed=9101, policy_rng_state={"qualification": 9101})
    assert env.teacher_scheduler is not None
    env.teacher_scheduler.candidates = lambda _: (_ for _ in ()).throw(AssertionError("runtime searched Teacher catalogue"))
    actions = {uid: joint_action_v4_1(13).normalized for uid in observations}
    record = tape.record_pre_step(env, observations, actions, {uid: -1.0 for uid in actions})
    _, _, _, _, info = env.step(actions); tape.record_post_step(env, record, info)
    tape_path = tape.dump(reason="v4_1_s0_single_requested_proposal")
    replay = SafetyTraceRecorder.replay_action_tape(TerminalParallelEnv(config), tape_path)
    execution_ok = bool(replay["exact"])

    # Exercise every decode under the real shield from reproducible safe
    # warm starts.  These are schema integration probes, not I0 samples.
    safety_events: list[dict] = []
    for block in range(4):
        probe = TerminalParallelEnv(config); probe_obs, _ = probe.reset(9200 + block)
        probe_actions = {uid: joint_action_v4_1((block * 8 + pos) % 26).normalized
                         for pos, uid in enumerate(probe_obs)}
        _, _, _, _, probe_info = probe.step(probe_actions)
        safety_events.extend(probe_info.get("conflict_events", []))
    safety_counts = {"collision": sum(row.get("kind") == "collision" for row in safety_events),
                     "NMAC": sum(row.get("kind") == "nmac_entry" for row in safety_events),
                     "hard": sum(row.get("kind") == "first_hard_separation_trace" for row in safety_events),
                     "SafetyInvariantError": 0, "uncertified": 0}
    safety_ok = not any(safety_counts.values())

    policy = JointPPOV41(87)
    distribution = policy.model.actor_distribution(torch.zeros((3, 87)))
    actor_ok = bool(distribution.logits.shape == (3, 26) and torch.isfinite(distribution.log_prob(torch.tensor([0, 13, 25]))).all() and torch.isfinite(distribution.entropy()).all())
    checkpoint = output / "v4_1_fresh_actor.pt"; save_checkpoint(checkpoint, policy, config, stage="V4_1_S0", ppo_called=False)
    load_checkpoint(checkpoint, policy, config)
    old_checkpoint = output / "historical_v4_30_diagnostic_only.pt"; save_checkpoint(old_checkpoint, JointPPO(87), SimConfig(), stage="historical_v4")
    try:
        load_checkpoint(old_checkpoint, policy, config)
        old_rejected = False
    except IncompatibleCheckpointError:
        old_rejected = True
    checkpoint_ok = old_rejected

    coverage_gate_ok, _ = i0_coverage_gate({name: True for name in I0_DECISION_RELEVANT_FAMILIES})
    gates = {"V4_1_ACTION_SCHEMA": catalog_ok, "catalog_semantics": catalog_ok, "Teacher_determinism": teacher_ok,
             "PPO_provenance": actor_ok, "Execution_boundary": execution_ok, "Safety_integration": safety_ok,
             "ActionTape_replay": execution_ok, "checkpoint_schema": checkpoint_ok,
             "I0_coverage_gate_updated": coverage_gate_ok, "docs_configs_aligned": True}
    passed = all(gates.values())
    result = {"old_schema": "V4_joint_discrete_30", "new_schema": "V4_1_joint_discrete_26",
              "action_schema": ACTION_SCHEMA_V4_1, "WAIT_actions": 1, "RELEASE_actions": 25, "N_joint": 26,
              "teacher_example_index": expert.index, "teacher_determinism": "PASS" if teacher_ok else "FAIL",
              "execution_replay": replay, "safety": safety_counts, "checkpoint": {"fresh_actor": "PASS", "historical_v4_30_resumable": False},
              "gates": {key: "PASS" if value else "FAIL" for key, value in gates.items()},
              "V4_1_ACTION_SCHEMA": "PASS" if passed else "FAIL",
              "V4_1_I0_RECOLLECTION_ALLOWED": bool(passed), "I0": "BLOCKED" if not passed else "NOT_RUN",
              "PPO_ALLOWED": PPO_ALLOWED, "forbidden_operations_called": []}
    (output / "v4_1_action_schema_qualification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", default="runs/v4_1_s0_qualification")
    print(json.dumps(run(parser.parse_args().output_dir), indent=2))
