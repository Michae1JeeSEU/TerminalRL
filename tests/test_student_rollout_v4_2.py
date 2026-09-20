from dataclasses import replace

import numpy as np
import pytest
import torch

from terminal_marl.algo.student_ppo_v4_2 import StudentPPOV42, StudentStepV42
from terminal_marl.core.safety_trace import SafetyTraceRecorder
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training.run_v4_2_hierarchical_s0_qualification import _config
from terminal_marl.core.low_level_joint_action_v4_2 import hierarchical_from_normalized_v4_2
from terminal_marl.training.student_rollout_v4_2 import (_student_request_details, rollout,
    validate_student_config, world_value)


def config():
    return replace(_config(), observation_schema_version="phaseA_low_obs_v2_1_retry_state",
                   experiment_mode="fixed_low_student", teacher_enabled=False,
                   warm_start_uavs=1, total_requested_uavs=1, fixed_arrival_schedule=(0.,),
                   max_simulation_seconds=3.)


def test_rejects_historical_teacher_configuration():
    with pytest.raises(ValueError):
        validate_student_config(_config())


def test_empty_world_retains_a_valid_critic_update():
    policy = StudentPPOV42(88)
    world = np.zeros((0, 88), dtype=np.float32)
    value = world_value(policy, world)
    row = StudentStepV42(world, np.zeros(0, dtype=int), np.zeros(0, dtype=int),
        np.zeros(0), np.zeros(0), np.zeros(0, dtype=bool), value, value, 0., False, True)
    result = policy.update([row], training_authorized=True, epochs=1)
    assert result["actor_decisions"] == 0
    assert np.isfinite(result["loss"])


@pytest.mark.parametrize("release", [False, True])
def test_student_labels_are_sidecars_and_requested_tape_replays(tmp_path, release):
    policy = StudentPPOV42(88)
    # Exercise canonical WAIT and a nominal RELEASE, not random proposals.
    with torch.no_grad():
        policy.model.gate_head.weight.zero_()
        policy.model.gate_head.bias.copy_(torch.tensor([-10., 10.] if release else [10., -10.]))
        policy.model.proposal_head.weight.zero_()
        policy.model.proposal_head.bias.fill_(-10.)
        policy.model.proposal_head.bias[12] = 10.
    label_config = replace(config(), experiment_mode="fixed_teacher_heavy", teacher_enabled=True, teacher_mode="teacher_label_only")
    steps, result = rollout(policy, label_config, seed=701, max_steps=2,
                            output=tmp_path / "rollout", deterministic=True, collect_labels=True)
    assert len(steps) == 2
    assert result["label_rows"] == 2
    assert not any(result["safety"].values())
    assert not result["PPO_ALLOWED"]
    assert not steps[0].stochastic_behavior
    if release:
        assert result["counts"]["RELEASE"] >= 1
    replay = SafetyTraceRecorder.replay_action_tape(TerminalParallelEnv(label_config), result["action_tape"])
    assert replay["exact"]


def test_empty_initial_arrival_interval_does_not_end_episode(tmp_path):
    policy = StudentPPOV42(88)
    delayed = replace(config(), reset_mode="flow_from_empty", warm_start_uavs=0,
                      fixed_arrival_schedule=(10.,), max_simulation_seconds=3.)
    steps, result = rollout(policy, delayed, seed=702, max_steps=2, output=tmp_path / "delayed")
    assert len(steps) == 2
    assert all(len(row.world) == 0 for row in steps)
    assert not result["terminal"]
    assert steps[-1].episode_end


def test_single_forward_produces_requested_first_bounded_ranked_shortlist():
    policy = StudentPPOV42(88)
    with torch.no_grad():
        policy.model.gate_head.weight.zero_()
        policy.model.gate_head.bias.copy_(torch.tensor([-10., 10.]))
        policy.model.proposal_head.weight.zero_()
        policy.model.proposal_head.bias.fill_(-10.)
        policy.model.proposal_head.bias[4] = 3.
        policy.model.proposal_head.bias[7] = 2.
        policy.model.proposal_head.bias[2] = 1.
    runtime = replace(config(), runtime_proposal_shortlist_k=3)
    env = TerminalParallelEnv(runtime)
    observations, _ = env.reset(703)
    details = _student_request_details(
        policy, env, observations, deterministic=True, shortlist_k=3,
    )
    requested, shortlists = details[3], details[-1]
    uid = next(iter(shortlists))
    ranked = [hierarchical_from_normalized_v4_2(row).proposal for row in shortlists[uid]]
    assert ranked == [4, 7, 2]
    assert np.array_equal(shortlists[uid][0], requested[uid])


def test_k3_rollout_records_and_exactly_replays_runtime_shortlists(tmp_path):
    policy = StudentPPOV42(88)
    with torch.no_grad():
        policy.model.gate_head.weight.zero_()
        policy.model.gate_head.bias.copy_(torch.tensor([-10., 10.]))
        policy.model.proposal_head.weight.zero_()
        policy.model.proposal_head.bias.copy_(torch.arange(25, dtype=torch.float32))
    runtime = replace(config(), runtime_proposal_shortlist_k=3)
    _, result = rollout(
        policy, runtime, seed=704, max_steps=2,
        output=tmp_path / "k3", deterministic=True,
    )
    replay = SafetyTraceRecorder.replay_action_tape(
        TerminalParallelEnv(runtime), result["action_tape"],
    )
    assert replay["exact"]
