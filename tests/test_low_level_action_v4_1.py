import json
import numpy as np
import pytest
import torch

from terminal_marl.algo.joint_ppo import JointPPO
from terminal_marl.algo.joint_ppo_v4_1 import JointPPOV41
from terminal_marl.core.checkpoints import IncompatibleCheckpointError, load_checkpoint, save_checkpoint
from terminal_marl.core.config import SimConfig
from terminal_marl.core.low_level_joint_action import joint_action
from terminal_marl.core.low_level_joint_action_v4_1 import (ACTION_SCHEMA_V4_1, JOINT_ACTION_CATALOG_V4_1,
    N_JOINT_ACTIONS_V4_1, decode_joint_actions_v4_1, joint_action_v4_1, joint_index_from_normalized_v4_1)
from terminal_marl.core.safety_trace import SafetyTraceRecorder
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training.low_v4_1_imitation import I0_DECISION_RELEVANT_FAMILIES, i0_coverage_gate
from terminal_marl.training.v4_1_joint_teacher import teacher_joint_expert_v4_1, teacher_joint_label_v4_1


def _config(**extra):
    return SimConfig(action_schema_version=ACTION_SCHEMA_V4_1, teacher_enabled=True,
        teacher_mode="teacher_label_only", scheduler_mode="execution", experiment_mode="fixed_teacher_heavy",
        reset_mode="warm_start_rings", warm_start_uavs=8, warm_start_steady_motion=True,
        fixed_arrival_schedule=(0.0,) * 8, total_requested_uavs=8, max_simulation_seconds=2.0, **extra)


def test_v4_1_catalogue_is_one_wait_plus_25_release_and_preserves_release_decode():
    assert N_JOINT_ACTIONS_V4_1 == 26
    assert JOINT_ACTION_CATALOG_V4_1[0].normalized.tolist() == [0.0, 0.0, 0.0]
    assert len([row for row in JOINT_ACTION_CATALOG_V4_1 if row.kind == "WAIT"]) == 1
    decoded = decode_joint_actions_v4_1(np.arange(26))
    assert np.array_equal(joint_index_from_normalized_v4_1(decoded), np.arange(26))
    for index in range(1, 26):
        assert np.array_equal(joint_action_v4_1(index).normalized, joint_action(index + 4).normalized)


def test_v4_1_teacher_is_deterministic_and_wait_has_no_residual_ranking():
    env = TerminalParallelEnv(_config()); observations, _ = env.reset(31)
    uav = env.active[int(next(iter(observations)))]
    labels = [teacher_joint_label_v4_1(env, uav).index for _ in range(20)]
    expert, audit = teacher_joint_expert_v4_1(env, uav, audit_catalogue=True)
    assert set(labels) == {expert.index}
    wait_row = next(row for row in audit if row["decoded"]["kind"] == "WAIT")
    assert "abs_speed_residual" not in wait_row["candidate_score"]


def test_v4_1_runtime_executes_one_requested_proposal_and_tape_replays(tmp_path):
    config = _config(); env = TerminalParallelEnv(config); observations, _ = env.reset(17)
    assert env.teacher_scheduler is not None
    env.teacher_scheduler.candidates = lambda _: (_ for _ in ()).throw(AssertionError("runtime searched teacher candidates"))
    tape = SafetyTraceRecorder(tmp_path, dt=config.dt); tape.start_episode(env, episode_seed=17, policy_rng_state={})
    actions = {uid: joint_action_v4_1(13).normalized for uid in observations}
    row = tape.record_pre_step(env, observations, actions, {uid: -1.0 for uid in actions})
    _, _, _, _, info = env.step(actions); tape.record_post_step(env, row, info)
    path = tape.dump(reason="v4_1_test")
    payload = json.loads(path.read_text())
    action = payload["action_tape"][0]
    assert action["action_schema_version"] == ACTION_SCHEMA_V4_1
    assert action["joint_action_index"] and action["decoded_action"] and "post_state_hash" in action
    assert {row["joint_action_index"] for row in action["decoded_action"].values()} == {13}
    assert SafetyTraceRecorder.replay_action_tape(TerminalParallelEnv(config), path)["exact"]


def test_v4_1_fresh_actor_and_checkpoint_boundary(tmp_path):
    policy = JointPPOV41(4)
    distribution = policy.model.actor_distribution(torch.zeros((3, 4)))
    assert distribution.logits.shape == (3, 26)
    assert torch.isfinite(distribution.log_prob(torch.tensor([0, 13, 25]))).all()
    assert torch.isfinite(distribution.entropy()).all()
    config = _config(); path = tmp_path / "v4_1.pt"; save_checkpoint(path, policy, config); load_checkpoint(path, policy, config)
    old_path = tmp_path / "v4.pt"; old = JointPPO(4); save_checkpoint(old_path, old, SimConfig())
    with pytest.raises(IncompatibleCheckpointError):
        load_checkpoint(old_path, policy, config)


def test_v4_1_i0_gate_is_target_family_not_catalogue_occupancy():
    passed, missing = i0_coverage_gate({name: True for name in I0_DECISION_RELEVANT_FAMILIES})
    assert passed and not missing
    passed, missing = i0_coverage_gate({"WAIT_expert": True})
    assert not passed and "RELEASE_expert" in missing
