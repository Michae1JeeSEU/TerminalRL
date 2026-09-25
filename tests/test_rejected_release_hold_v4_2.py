from __future__ import annotations

import numpy as np
import pytest

from terminal_marl.core.config import SimConfig
from terminal_marl.core.low_level_joint_action_v4_1 import RESIDUAL_VALUES
from terminal_marl.core.low_level_joint_action_v4_2 import ACTION_SCHEMA_V4_2, hierarchical_from_joint_index_v4_1
from terminal_marl.core.merge_scheduler import LowLevelIntent, RepairCandidate
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.core.safety_trace import _digest, _environment_config_match


def _config(mode: str = "legacy_requested") -> SimConfig:
    return SimConfig(
        action_schema_version=ACTION_SCHEMA_V4_2,
        observation_schema_version="phaseA_low_obs_v2_1_retry_state",
        v4_2_rejected_release_hold_mode=mode,
        experiment_mode="fixed_low_student",
        teacher_enabled=False,
        teacher_mode="teacher_label_only",
        scheduler_mode="execution",
        runtime_proposal_shortlist_k=1,
        reset_mode="warm_start_rings",
        warm_start_uavs=1,
        warm_start_steady_motion=True,
        fixed_arrival_schedule=(0.0,),
        total_requested_uavs=1,
        max_simulation_seconds=2.0,
    )


def _release_action(*, speed: float = RESIDUAL_VALUES[0], merge: float = 0.0) -> np.ndarray:
    speed_index = RESIDUAL_VALUES.index(speed)
    merge_index = RESIDUAL_VALUES.index(merge)
    return hierarchical_from_joint_index_v4_1(1 + speed_index * len(RESIDUAL_VALUES) + merge_index).normalized


def _patch_rejection(env: TerminalParallelEnv, monkeypatch, *, projected_speed: float | None = None):
    observations, _ = env.reset(75)
    uid = next(iter(observations))
    requested = _release_action()
    assert env.active[int(uid)].ring_role is not None

    def reject(_uav, _intent, _shortlist):
        return "safety_reject"

    seen_by_shield: list[LowLevelIntent] = []
    held: list[LowLevelIntent] = []

    def protect(_env, intents, traces):
        seen_by_shield.append(intents[int(uid)])
        if projected_speed is None:
            return intents
        projected = dict(intents)
        projected_intent = LowLevelIntent(0.0, projected_speed, 0.0)
        projected[int(uid)] = projected_intent
        traces[int(uid)].shield_applied = projected_intent
        traces[int(uid)].shield_used = True
        traces[int(uid)].shield_reason = "test_projection"
        return projected

    def advance(_uav, intent):
        held.append(intent)
        return 0.0

    monkeypatch.setattr(env, "_release", reject)
    monkeypatch.setattr(env.runtime_traffic, "protect", protect)
    monkeypatch.setattr(env, "_advance_ring_hold", advance)
    _, _, _, _, info = env.step({uid: requested})
    return uid, requested, seen_by_shield, held, info


def test_canonical_wait_clears_rejected_release_hold_but_preserves_request(monkeypatch):
    env = TerminalParallelEnv(_config("canonical_wait"))
    uid, requested, shield_inputs, held, info = _patch_rejection(env, monkeypatch)

    assert info["release_decisions"][uid] == "safety_reject"
    assert shield_inputs[0].release_request == 0.0
    assert shield_inputs[0].speed_residual == 0.0
    assert held[0] == LowLevelIntent(0.0, 0.0, 0.0)
    trace = info["action_traces"][uid]
    assert trace["requested_low_action"] == pytest.approx(requested.tolist())
    assert trace["execution_applied_action"] == [0.0, 0.0, 0.0]
    assert trace["shield_used"] is False
    assert info["actions"][0][:3] == pytest.approx(tuple(float(value) for value in requested))


def test_legacy_requested_keeps_rejected_release_hold_behavior(monkeypatch):
    env = TerminalParallelEnv(_config())
    uid, requested, shield_inputs, held, info = _patch_rejection(env, monkeypatch)

    assert env.config.v4_2_rejected_release_hold_mode == "legacy_requested"
    assert info["release_decisions"][uid] == "safety_reject"
    assert shield_inputs[0].release_request == pytest.approx(float(requested[0]))
    assert shield_inputs[0].speed_residual == pytest.approx(float(requested[1]))
    assert held[0].speed_residual == pytest.approx(float(requested[1]))
    trace = info["action_traces"][uid]
    assert trace["requested_low_action"] == pytest.approx(requested.tolist())
    assert trace["execution_applied_action"] == pytest.approx(requested.tolist())


def test_approved_release_keeps_certified_requested_proposal(monkeypatch):
    env = TerminalParallelEnv(_config("canonical_wait"))
    observations, _ = env.reset(75)
    uid = next(iter(observations))
    requested = hierarchical_from_joint_index_v4_1(13).normalized
    shield_inputs: list[LowLevelIntent] = []

    def observe_protect(_env, intents, _traces):
        shield_inputs.append(intents[int(uid)])
        return intents

    monkeypatch.setattr(env.runtime_traffic, "protect", observe_protect)
    _, _, _, _, info = env.step({uid: requested})

    assert info["release_decisions"][uid] == "approved"
    assert shield_inputs[0].speed_residual == pytest.approx(float(requested[1]))
    assert info["action_traces"][uid]["execution_applied_action"] == pytest.approx(requested.tolist())


def test_shield_can_project_after_canonical_wait(monkeypatch):
    env = TerminalParallelEnv(_config("canonical_wait"))
    uid, _, shield_inputs, held, info = _patch_rejection(env, monkeypatch, projected_speed=0.5)

    assert shield_inputs[0] == LowLevelIntent(0.0, 0.0, 0.0)
    assert held[0] == LowLevelIntent(0.0, 0.5, 0.0)
    trace = info["action_traces"][uid]
    assert trace["execution_applied_action"] == [0.0, 0.0, 0.0]
    assert trace["shield_applied_action"] == [0.0, 0.5, 0.0]
    assert trace["shield_used"] is True


def test_original_wait_remains_nominal_under_canonical_mode(monkeypatch):
    env = TerminalParallelEnv(_config("canonical_wait"))
    observations, _ = env.reset(75)
    uid = next(iter(observations))
    seen: list[LowLevelIntent] = []

    def protect(_env, intents, _traces):
        seen.append(intents[int(uid)])
        return intents

    monkeypatch.setattr(env.runtime_traffic, "protect", protect)
    _, _, _, _, info = env.step({uid: np.asarray((0.0, 0.0, 0.0), dtype=np.float32)})

    assert info["release_decisions"][uid] == "not_requested"
    assert seen[0] == LowLevelIntent(0.0, 0.0, 0.0)
    assert info["action_traces"][uid]["requested_low_action"] == [0.0, 0.0, 0.0]
    assert info["action_traces"][uid]["execution_applied_action"] == [0.0, 0.0, 0.0]


def test_future_reservation_wait_keeps_internal_holder_residual(monkeypatch):
    env = TerminalParallelEnv(_config("canonical_wait"))
    observations, _ = env.reset(75)
    uid = next(iter(observations))
    candidate = RepairCandidate("future", LowLevelIntent(1.0, 0.5, 0.0), delay_seconds=5.0)
    env.future_release_reservations[int(uid)] = {"candidate": candidate}
    requested = _release_action()
    shield_inputs: list[LowLevelIntent] = []
    held: list[LowLevelIntent] = []

    def future_wait(_uav, _intent, _shortlist):
        return "future_reservation_wait"

    def protect(_env, intents, _traces):
        shield_inputs.append(intents[int(uid)])
        return intents

    def advance(_uav, intent):
        held.append(intent)
        return 0.0

    monkeypatch.setattr(env, "_release", future_wait)
    monkeypatch.setattr(env.runtime_traffic, "protect", protect)
    monkeypatch.setattr(env, "_advance_ring_hold", advance)
    _, _, _, _, info = env.step({uid: requested})

    assert info["release_decisions"][uid] == "future_reservation_wait"
    assert shield_inputs[0] == LowLevelIntent(0.0, 0.5, 0.0)
    assert held[0] == LowLevelIntent(0.0, 0.5, 0.0)


def test_rejected_release_hold_mode_rejects_unknown_value():
    with pytest.raises(ValueError, match="unsupported v4_2_rejected_release_hold_mode"):
        SimConfig(v4_2_rejected_release_hold_mode="unknown")


def test_legacy_default_matches_hash_written_before_new_field():
    config = _config("legacy_requested")
    old_values = dict(vars(config))
    old_values.pop("v4_2_rejected_release_hold_mode")

    matched, compatibility = _environment_config_match(config, _digest(old_values))

    assert matched is True
    assert compatibility == "legacy_requested_default_field"


def test_canonical_wait_does_not_accept_legacy_hash():
    config = _config("canonical_wait")
    old_values = dict(vars(config))
    old_values.pop("v4_2_rejected_release_hold_mode")

    matched, compatibility = _environment_config_match(config, _digest(old_values))

    assert matched is False
    assert compatibility is None


def test_legacy_default_rejects_hash_with_another_config_change():
    config = _config("legacy_requested")
    old_values = dict(vars(config))
    old_values.pop("v4_2_rejected_release_hold_mode")
    old_values["max_simulation_seconds"] = 99.0

    matched, compatibility = _environment_config_match(config, _digest(old_values))

    assert matched is False
    assert compatibility is None
