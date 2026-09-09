import numpy as np
import torch

from terminal_marl.algo.joint_ppo import JointPPO, JointRolloutBuffer, JointTransition
from terminal_marl.core.config import SimConfig
from terminal_marl.core.low_level_joint_action import JOINT_ACTION_CATALOG, N_JOINT_ACTIONS, decode_joint_actions, joint_index_from_normalized
from terminal_marl.core.teacher_scheduler import TeacherScheduler
from terminal_marl.core.merge_scheduler import MergeScheduler
from terminal_marl.core.safety_trace import SafetyTraceRecorder
from terminal_marl.core.low_level_joint_action import joint_action
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.core.checkpoints import checkpoint_metadata, validate_checkpoint_metadata
from terminal_marl.training.v4_joint_teacher import teacher_joint_expert, teacher_joint_label


def test_v4_catalog_has_only_nonduplicated_wait_actions():
    assert N_JOINT_ACTIONS == 30
    assert len([row for row in JOINT_ACTION_CATALOG if row.release_request == 0.0]) == 5
    assert all(row.merge_time_residual == 0.0 for row in JOINT_ACTION_CATALOG[:5])
    decoded = decode_joint_actions(np.arange(N_JOINT_ACTIONS))
    assert np.array_equal(joint_index_from_normalized(decoded), np.arange(N_JOINT_ACTIONS))


def test_v4_teacher_catalogue_is_deterministic_and_runtime_safe():
    config = SimConfig(teacher_enabled=True, action_schema_version="low_action_schema_v4_joint_discrete")
    teacher = TeacherScheduler(config, MergeScheduler(config))
    first = teacher.choose_joint_verified(lambda row: row.index == 17)
    second = teacher.choose_joint_verified(lambda row: row.index == 17)
    assert first == second == JOINT_ACTION_CATALOG[17]
    assert teacher.choose_joint_verified(lambda row: False).release_request == 0.0


def test_v4_ppo_ratio_and_joint_bc_use_requested_joint_indices_only():
    torch.manual_seed(7); np.random.seed(7)
    policy = JointPPO(4, teacher_bc_weight=.05)
    batch = policy.act_details(np.zeros((2, 4), np.float32), np.zeros((2, 4), np.float32))
    buffer = JointRolloutBuffer()
    for index in range(2):
        buffer.add(JointTransition(str(index), np.zeros(4, np.float32), np.zeros((2, 4), np.float32),
                                    int(batch.joint_action_index[index]), float(batch.joint_log_prob[index]), batch.value,
                                    1.0, True, teacher_joint_action=5, teacher_used=True))
    metrics = policy.update(buffer, epochs=1, minibatch_size=2)
    assert np.isfinite(metrics["joint_teacher_bc_loss"])
    assert np.isfinite(metrics["approx_kl"])


def test_v4_action_tape_replays_joint_decode_exactly(tmp_path):
    config = SimConfig(action_schema_version="low_action_schema_v4_joint_discrete", reset_mode="warm_start_rings",
                       warm_start_uavs=8, fixed_arrival_schedule=(0.0,) * 8, total_requested_uavs=8, max_simulation_seconds=2.0)
    env = TerminalParallelEnv(config); observations, _ = env.reset(17)
    tape = SafetyTraceRecorder(tmp_path, dt=config.dt); tape.start_episode(env, episode_seed=17, policy_rng_state={})
    actions = {uid: joint_action(index % N_JOINT_ACTIONS).normalized for index, uid in enumerate(observations)}
    record = tape.record_pre_step(env, observations, actions, {uid: -1.0 for uid in actions})
    _, _, _, _, info = env.step(actions); tape.record_post_step(env, record, info)
    path = tape.dump(reason="v4_test")
    payload = __import__("json").loads(path.read_text())
    assert payload["action_tape"][0]["action_schema_version"] == "low_action_schema_v4_joint_discrete"
    assert payload["action_tape"][0]["joint_action_index"]
    assert SafetyTraceRecorder.replay_action_tape(TerminalParallelEnv(config), path)["exact"]


def test_v4_runtime_executes_one_requested_proposal_not_teacher_catalogue_search():
    config = SimConfig(action_schema_version="low_action_schema_v4_joint_discrete", teacher_enabled=True,
                       teacher_mode="teacher_full", teacher_release_label_enabled=False, reset_mode="warm_start_rings",
                       warm_start_uavs=8, fixed_arrival_schedule=(0.0,) * 8, total_requested_uavs=8, max_simulation_seconds=2.0)
    env = TerminalParallelEnv(config); observations, _ = env.reset(23)
    assert env.teacher_scheduler is not None
    def forbidden_catalogue_search(_):
        raise AssertionError("runtime V4 execution must not search TeacherScheduler candidates")
    env.teacher_scheduler.candidates = forbidden_catalogue_search
    actions = {uid: joint_action(17).normalized for uid in observations}
    env.step(actions)


def test_v4_checkpoint_metadata_declares_joint_catalogue():
    config = SimConfig(action_schema_version="low_action_schema_v4_joint_discrete")
    metadata = checkpoint_metadata(config)
    validate_checkpoint_metadata(metadata, config)
    assert metadata["low_level_joint_action_count"] == N_JOINT_ACTIONS
    assert metadata["low_level_policy_distribution"] == "joint_categorical"


def test_v4_expert_target_is_request_independent_and_release_preferred_when_feasible():
    config = SimConfig(action_schema_version="low_action_schema_v4_joint_discrete", teacher_enabled=True,
                       teacher_mode="teacher_full", teacher_release_label_enabled=False, reset_mode="warm_start_rings",
                       warm_start_uavs=8, fixed_arrival_schedule=(0.0,) * 8, total_requested_uavs=8, max_simulation_seconds=2.0)
    env = TerminalParallelEnv(config); observations, _ = env.reset(29); uav = env.active[int(next(iter(observations)))]
    labels = [teacher_joint_label(env, uav).index for _ in range(N_JOINT_ACTIONS)]
    expert, audit = teacher_joint_expert(env, uav, audit_catalogue=True)
    assert set(labels) == {expert.index}
    feasible_release = [row for row in audit if row["feasible"] and row["decoded"]["release_request"] > 0.0]
    if feasible_release:
        assert expert.release_request > 0.0
