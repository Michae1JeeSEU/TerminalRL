"""Regression observations for the opt-in pre-action forced-WAIT protocol.

The forced-WAIT metadata assertion protects the probability-one tape contract.
The remaining tests exercise the existing environment and PPO masking behavior
without using a checkpoint or a held-out artifact.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from terminal_marl.algo.joint_ppo_v4_2 import JointPPOV42
from terminal_marl.algo.student_ppo_v4_2 import StudentPPOV42, StudentStepV42
from terminal_marl.core.config import SimConfig
from terminal_marl.core.low_level_joint_action_v4_2 import (
    ACTION_SCHEMA_V4_2,
    WAIT_GATE,
    hierarchical_from_joint_index_v4_1,
)
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training.student_rollout_v4_2 import _student_request_details


OBSERVATION_SCHEMA = "phaseA_low_obs_v2_1_retry_state"


def _config(*, decision_mask_mode: str = "legacy") -> SimConfig:
    return SimConfig(
        action_schema_version=ACTION_SCHEMA_V4_2,
        observation_schema_version=OBSERVATION_SCHEMA,
        experiment_mode="fixed_low_student",
        teacher_enabled=False,
        scheduler_mode="execution",
        reset_mode="warm_start_rings",
        warm_start_uavs=1,
        warm_start_steady_motion=True,
        fixed_arrival_schedule=(0.0,),
        total_requested_uavs=1,
        max_simulation_seconds=2.0,
        student_decision_mask_mode=decision_mask_mode,
    )


def _warm_env(*, decision_mask_mode: str = "legacy", seed: int = 9101):
    env = TerminalParallelEnv(_config(decision_mask_mode=decision_mask_mode))
    observations, _ = env.reset(seed)
    uid = next(iter(observations))
    return env, observations, uid, env.active[int(uid)]


def test_forced_wait_row_records_probability_one_metadata() -> None:
    """Record the forced-WAIT contract and expose the current raw values.

    ``actor_mask=False`` already suppresses actor training.  The protocol
    additionally requires a canonical probability-one WAIT record so a tape
    cannot retain the old policy WAIT likelihood.
    """

    env, observations, uid, uav = _warm_env(
        decision_mask_mode="pre_action_forced_wait", seed=9102,
    )
    uav.safety_retry_after = env.now + 5.0
    torch.manual_seed(9103)
    policy = JointPPOV42(88)
    details = _student_request_details(
        policy, env, observations, deterministic=False, shortlist_k=1,
    )
    actor_mask, gates, proposals = details[2], details[4], details[5]
    gate_lp, proposal_lp, likelihoods = details[6], details[7], details[8]
    observed = {
        "uid": uid,
        "actor_mask": actor_mask.tolist(),
        "gate": gates.tolist(),
        "proposal": proposals.tolist(),
        "gate_logprob": gate_lp.tolist(),
        "proposal_logprob": proposal_lp.tolist(),
        "joint_logprob": [likelihoods[uid]["joint"]],
        "proposal_metadata": likelihoods[uid]["proposal"],
    }
    contract_ok = (
        np.array_equal(actor_mask, np.asarray([False], dtype=bool))
        and np.array_equal(gates, np.asarray([WAIT_GATE], dtype=int))
        and np.array_equal(proposals, np.asarray([-1], dtype=int))
        and np.allclose(gate_lp, 0.0, atol=0.0, rtol=0.0)
        and np.isnan(proposal_lp).all()
        and np.allclose([likelihoods[uid]["joint"]], 0.0, atol=0.0, rtol=0.0)
        and likelihoods[uid]["proposal"] is None
    )
    assert contract_ok, f"forced WAIT raw observed={observed}"


def test_legacy_mode_preserves_frozen_policy_request_rng_and_action() -> None:
    """Legacy and opt-in requests are identical while no forced condition holds."""

    legacy_env, legacy_obs, _, legacy_uav = _warm_env(
        decision_mask_mode="legacy", seed=9104,
    )
    forced_env, forced_obs, _, forced_uav = _warm_env(
        decision_mask_mode="pre_action_forced_wait", seed=9104,
    )
    legacy_uav.safety_retry_after = legacy_env.now
    forced_uav.safety_retry_after = forced_env.now
    policy_legacy = JointPPOV42(88)
    policy_forced = JointPPOV42(88)
    policy_forced.model.load_state_dict(policy_legacy.model.state_dict())

    torch.manual_seed(9105)
    legacy = _student_request_details(
        policy_legacy, legacy_env, legacy_obs, deterministic=False, shortlist_k=1,
    )
    torch.manual_seed(9105)
    forced = _student_request_details(
        policy_forced, forced_env, forced_obs, deterministic=False, shortlist_k=1,
    )
    assert legacy[0] == forced[0]
    np.testing.assert_array_equal(legacy[1], forced[1])
    np.testing.assert_array_equal(legacy[2], forced[2])
    np.testing.assert_array_equal(legacy[4], forced[4])
    np.testing.assert_array_equal(legacy[5], forced[5])
    np.testing.assert_allclose(legacy[6], forced[6], equal_nan=True)
    np.testing.assert_allclose(legacy[7], forced[7], equal_nan=True)
    assert legacy[3].keys() == forced[3].keys()
    for key in legacy[3]:
        np.testing.assert_array_equal(legacy[3][key], forced[3][key])
        assert legacy[8][key] == forced[8][key]


def test_forced_wait_blocks_all_real_env_release_candidates_as_retry() -> None:
    """All 25 real-environment RELEASE proposals hit the retry guard."""

    observed_decisions = []
    for proposal in range(25):
        env, observations, uid, uav = _warm_env(
            decision_mask_mode="pre_action_forced_wait", seed=9200 + proposal,
        )
        uav.safety_retry_after = env.now + 100.0
        action = hierarchical_from_joint_index_v4_1(proposal + 1).normalized
        _, _, _, _, info = env.step({uid: action})
        decision = info["release_decisions"][uid]
        observed_decisions.append(decision)
    assert observed_decisions == ["safety_wait"] * 25


def test_forced_wait_blocks_real_env_interface_affected_actor(monkeypatch) -> None:
    """A real warm-start scene with a patched affected interface is masked."""

    env, observations, _, uav = _warm_env(
        decision_mask_mode="pre_action_forced_wait", seed=9230,
    )
    uav.safety_retry_after = env.now
    monkeypatch.setattr(env.airspace, "affected_interfaces", lambda: {"o2m"})
    details = _student_request_details(
        JointPPOV42(88), env, observations, deterministic=True, shortlist_k=1,
    )
    assert details[2].tolist() == [False]
    assert details[4].tolist() == [WAIT_GATE]
    assert details[5].tolist() == [-1]


def test_positive_soc_energy_urgent_actionable_actor_is_not_masked() -> None:
    """Positive, low SOC alone does not force the actor row out of the mask."""

    env, observations, _, uav = _warm_env(
        decision_mask_mode="pre_action_forced_wait", seed=9231,
    )
    uav.energy_remaining = 0.01 * uav.energy_capacity
    uav.safety_retry_after = env.now
    details = _student_request_details(
        JointPPOV42(88), env, observations, deterministic=True, shortlist_k=1,
    )
    assert uav.soc > 0.0
    assert details[2].tolist() == [True]


def _student_step(world: np.ndarray, *, actor: bool, reward: float, value: float) -> StudentStepV42:
    return StudentStepV42(
        world=np.asarray(world, dtype=np.float32),
        requested_gate=np.zeros(1, dtype=np.int64),
        requested_proposal=np.full(1, -1, dtype=np.int64),
        gate_logprob=np.zeros(1, dtype=np.float32),
        proposal_logprob=np.full(1, np.nan, dtype=np.float32),
        actor_mask=np.asarray([actor], dtype=bool),
        value=float(value),
        next_value=0.0,
        reward=float(reward),
        terminated=False,
        episode_end=True,
    )


def _run_masked_critic_case(masked_world: np.ndarray, masked_reward: float) -> dict:
    torch.manual_seed(9301)
    policy = StudentPPOV42(88)
    rows = [
        _student_step(np.zeros((1, 88), dtype=np.float32), actor=True, reward=0.5, value=0.1),
        _student_step(np.ones((1, 88), dtype=np.float32) * 0.25, actor=True, reward=-0.5, value=0.2),
        _student_step(masked_world, actor=False, reward=masked_reward, value=0.0),
    ]
    np.random.seed(9302)
    return policy.update(
        rows, training_authorized=True, epochs=1, minibatch_size=3,
    )


def test_masked_rows_retain_team_tick_and_critic_but_not_actor_metrics() -> None:
    """Masked rows remain in team critic/tick accounting and actor filters."""

    baseline = _run_masked_critic_case(np.zeros((1, 88), dtype=np.float32), 0.0)
    perturbed = _run_masked_critic_case(np.ones((1, 88), dtype=np.float32) * 10.0, 100.0)
    for result in (baseline, perturbed):
        assert result["team_steps"] == 3
        assert result["actor_decisions"] == 2
        assert np.isfinite(result["value_loss"])
    assert perturbed["value_loss"] != pytest.approx(baseline["value_loss"])
    for key in ("actor_loss", "entropy", "approx_kl", "clip_fraction"):
        assert perturbed[key] == pytest.approx(baseline[key], abs=1e-7)
