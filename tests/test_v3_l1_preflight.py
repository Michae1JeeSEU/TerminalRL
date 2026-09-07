"""Regression coverage for V3 L1 preflight-only guarantees."""
from __future__ import annotations

from terminal_marl.core.low_level_action import LowLevelActionV3
from terminal_marl.core.safety_trace import SafetyTraceRecorder
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training.train_low_v2_ppo import l1_full_teacher_config


def test_v3_l1_config_keeps_qualified_physics_and_safety_sampling():
    config = l1_full_teacher_config(medium_rate=.003, reset_mode="warm_start_rings", demand_model="finite_cohort",
                                    warm_start_uavs=8, cohort_uavs=8, horizon_seconds=8.0)
    assert config.physics_dt == .2
    assert config.safety_sampling_dt == 1.0


def test_action_tape_replays_v3_release_despite_runtime_uuid_allocation(tmp_path):
    config = l1_full_teacher_config(medium_rate=.003, reset_mode="warm_start_rings", demand_model="finite_cohort",
                                    warm_start_uavs=8, cohort_uavs=8, horizon_seconds=8.0)
    env = TerminalParallelEnv(config)
    observations, _ = env.reset(3001)
    recorder = SafetyTraceRecorder(tmp_path, dt=config.dt)
    recorder.start_episode(env, episode_seed=3001, policy_rng_state={"unit": "v3"})
    action = LowLevelActionV3(1, 3, 4).normalized
    actions = {uid: action for uid in observations}
    record = recorder.record_pre_step(env, observations, actions)
    _, _, _, _, info = env.step(actions)
    recorder.record_post_step(env, record, info)
    assert env.scheduler.reservations  # Exercise UUID-bearing scheduler state.
    replay = SafetyTraceRecorder.replay_action_tape(TerminalParallelEnv(config), recorder.dump(reason="v3_uuid_regression"))
    assert replay["exact"]
