from __future__ import annotations

from dataclasses import replace
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from terminal_marl.algo.student_ppo_v4_2 import StudentPPOV42, StudentStepV42, team_gae
from terminal_marl.core.checkpoints import (
    IncompatibleCheckpointError,
    checkpoint_metadata,
    validate_checkpoint_metadata,
)
from terminal_marl.core.config import RingRole, SimConfig
from terminal_marl.core.low_level_joint_action_v4_2 import ACTION_SCHEMA_V4_2
from terminal_marl.core.observations import (
    LOW_OBS_V2_1,
    LOW_OBS_V2_4,
    CenterLandingZoneBroadcast,
    build_center_broadcast,
    build_low_observation_v2_1,
    build_low_observation_v2_4,
    observation_schema_for_version,
)
from terminal_marl.core.safety_trace import (
    _digest,
    _environment_config_match,
    SafetyTraceRecorder,
)
from terminal_marl.core.uav import FlightPhase, MissionType
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training.evaluate_p3_task_neutral import _config_snapshot
from terminal_marl.training.student_rollout_v4_2 import (
    OBS_SCHEMA_V2_1,
    OBS_SCHEMA_V2_4,
    _student_request_details,
    rollout,
    validate_student_config,
)


def _observation_config(
    *,
    schema: str = OBS_SCHEMA_V2_4,
    mission_probabilities: tuple[float, ...] = (0.45, 0.25, 0.10, 0.20),
    task_priority_enabled: bool = False,
) -> SimConfig:
    return SimConfig(
        action_schema_version=ACTION_SCHEMA_V4_2,
        observation_schema_version=schema,
        experiment_mode="fixed_low_student",
        teacher_enabled=False,
        scheduler_mode="execution",
        reset_mode="warm_start_rings",
        warm_start_uavs=1,
        warm_start_steady_motion=True,
        fixed_arrival_schedule=(0.0,),
        total_requested_uavs=1,
        max_active_uavs=2,
        initial_soc_mode="fixed",
        initial_soc=0.8,
        mission_probabilities=mission_probabilities,
        task_priority_enabled=task_priority_enabled,
        max_simulation_seconds=3.0,
    )


def _mask_env(*, mode: str, now: float = 0.0, retry_after: float = 0.0,
              soc: float = 0.8, affected: tuple[str, ...] = ()):
    uav = SimpleNamespace(
        committed_plan=None,
        phase=FlightPhase.RING_OUTER,
        ring_role=RingRole.OUTER,
        safety_retry_after=retry_after,
        soc=soc,
    )
    airspace = SimpleNamespace(affected_interfaces=lambda: affected)
    return SimpleNamespace(
        config=SimpleNamespace(student_decision_mask_mode=mode),
        now=now,
        active={1: uav},
        future_release_reservations={},
        airspace=airspace,
    )


def _mask_value(mode: str, *, now: float = 0.0, retry_after: float = 0.0,
                soc: float = 0.8, affected: tuple[str, ...] = ()) -> bool:
    policy = StudentPPOV42(88)
    env = _mask_env(
        mode=mode, now=now, retry_after=retry_after,
        soc=soc, affected=affected,
    )
    details = _student_request_details(
        policy, env, {"1": np.zeros(88, dtype=np.float32)},
        deterministic=True, shortlist_k=1,
    )
    return bool(details[2][0])


def test_pre_action_mask_is_opt_in_and_expired_retry_restores_actor() -> None:
    assert SimConfig().student_decision_mask_mode == "legacy"
    with pytest.raises(ValueError, match="student_decision_mask_mode"):
        SimConfig(student_decision_mask_mode=True)
    with pytest.raises(ValueError, match="student_decision_mask_mode"):
        SimConfig(student_decision_mask_mode="unknown")

    # Legacy keeps the historical actor mask even when pre-action state would
    # force WAIT.  The opt-in mode excludes each listed blocker before action.
    assert _mask_value("legacy", retry_after=5.0, affected=("o2m",), soc=0.0)
    assert not _mask_value("pre_action_forced_wait", retry_after=5.0)
    assert _mask_value("pre_action_forced_wait", now=5.0, retry_after=5.0)
    assert not _mask_value("pre_action_forced_wait", now=5.0, retry_after=5.0, affected=("o2m",))
    assert not _mask_value("pre_action_forced_wait", now=5.0, retry_after=5.0, soc=0.0)


def test_no_actor_steps_still_carry_tick_reward_through_team_gae() -> None:
    world = np.zeros((1, 90), dtype=np.float32)
    steps = [
        StudentStepV42(
            world.copy(), np.zeros(1, dtype=int), np.full(1, -1, dtype=int),
            np.zeros(1), np.full(1, np.nan), np.zeros(1, dtype=bool),
            0.0, 0.0, -0.01, False, False,
        ),
        StudentStepV42(
            world.copy(), np.zeros(1, dtype=int), np.full(1, -1, dtype=int),
            np.zeros(1), np.full(1, np.nan), np.zeros(1, dtype=bool),
            0.0, 0.0, -0.01, False, True,
        ),
    ]
    advantages, returns = team_gae(steps, gamma=0.99, gae_lambda=0.95)
    assert np.all(advantages < 0.0)
    assert np.all(returns < 0.0)
    assert float(advantages[0]) < float(advantages[1])


def test_v24_preserves_v21_prefix_and_task_neutral_mission_token() -> None:
    standard = TerminalParallelEnv(_observation_config(mission_probabilities=(1.0, 0.0, 0.0, 0.0)))
    cargo = TerminalParallelEnv(_observation_config(mission_probabilities=(0.0, 1.0, 0.0, 0.0)))
    standard_obs, _ = standard.reset(301)
    cargo_obs, _ = cargo.reset(301)
    standard_uav = next(iter(standard.active.values()))
    cargo_uav = next(iter(cargo.active.values()))
    standard_v24 = standard_obs[str(standard_uav.uid)]
    cargo_v24 = cargo_obs[str(cargo_uav.uid)]

    assert LOW_OBS_V2_4.dimension == 90
    assert standard_v24.shape == cargo_v24.shape == (90,)
    assert np.array_equal(
        standard_v24[: LOW_OBS_V2_1.dimension],
        build_low_observation_v2_1(standard_uav, standard),
    )
    assert np.array_equal(
        cargo_v24[: LOW_OBS_V2_1.dimension],
        build_low_observation_v2_1(cargo_uav, cargo),
    )
    assert np.array_equal(standard_v24[:88], cargo_v24[:88])
    assert standard_v24[-2] == pytest.approx(30.0 / 120.0)
    assert cargo_v24[-2] == pytest.approx(120.0 / 120.0)
    assert not np.array_equal(standard_v24[-2:], cargo_v24[-2:])
    assert np.array_equal(
        cargo_v24[LOW_OBS_V2_1.slices["mission"]],
        np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )


def test_v24_compatible_pad_eta_is_minimum_and_unknown_is_unavailable() -> None:
    env = TerminalParallelEnv(_observation_config(mission_probabilities=(0.0, 1.0, 0.0, 0.0)))
    env.reset(302)
    uav = next(iter(env.active.values()))
    center = build_center_broadcast(env)

    one_second = tuple(
        replace(zone, compatible_missions=(MissionType.CARGO.value,), eta_seconds=1.0)
        for zone in center.landing_zones
    )
    one_hundred_twenty = tuple(
        replace(zone, compatible_missions=(MissionType.CARGO.value,), eta_seconds=120.0)
        for zone in center.landing_zones
    )
    incompatible = tuple(
        replace(zone, compatible_missions=(MissionType.STANDARD.value,), eta_seconds=0.0)
        for zone in center.landing_zones
    )
    unknown = tuple(
        replace(zone, compatible_missions=(MissionType.CARGO.value,), eta_seconds=math.inf)
        for zone in center.landing_zones
    )
    vectors = [
        build_low_observation_v2_4(uav, env, center_broadcast=replace(center, landing_zones=one_second)),
        build_low_observation_v2_4(uav, env, center_broadcast=replace(center, landing_zones=one_hundred_twenty)),
        build_low_observation_v2_4(uav, env, center_broadcast=replace(center, landing_zones=incompatible)),
        build_low_observation_v2_4(uav, env, center_broadcast=replace(center, landing_zones=unknown)),
    ]
    assert vectors[0][-1] == pytest.approx(1.0 / 31.0)
    assert vectors[1][-1] == pytest.approx(120.0 / 150.0)
    assert vectors[0][-1] < vectors[1][-1]
    assert vectors[2][-1] == pytest.approx(1.0)
    assert vectors[3][-1] == pytest.approx(1.0)


def test_v24_schema_hash_and_checkpoint_validation_reject_old_88d() -> None:
    assert observation_schema_for_version(OBS_SCHEMA_V2_1) is LOW_OBS_V2_1
    assert observation_schema_for_version(OBS_SCHEMA_V2_4) is LOW_OBS_V2_4
    assert LOW_OBS_V2_1.dimension == 88
    assert LOW_OBS_V2_4.dimension == 90
    assert LOW_OBS_V2_1.feature_hash != LOW_OBS_V2_4.feature_hash

    old_config = _observation_config(schema=OBS_SCHEMA_V2_1)
    new_config = _observation_config(schema=OBS_SCHEMA_V2_4)
    metadata = checkpoint_metadata(old_config)
    with pytest.raises(IncompatibleCheckpointError):
        validate_checkpoint_metadata(metadata, new_config)


def test_legacy_tape_hash_compatibility_appends_mask_marker_only_for_legacy() -> None:
    config = SimConfig()
    old_values = dict(vars(config))
    old_values.pop("student_decision_mask_mode")
    matched, compatibility = _environment_config_match(config, _digest(old_values))
    assert matched is True
    assert compatibility == "student_decision_mask_legacy_default_field"

    old_values.pop("v4_2_rejected_release_hold_mode")
    matched, compatibility = _environment_config_match(config, _digest(old_values))
    assert matched is True
    assert compatibility == "legacy_requested_default_field+student_decision_mask_legacy_default_field"

    old_values.pop("warm_start_layout_randomized")
    matched, compatibility = _environment_config_match(config, _digest(old_values))
    assert matched is True
    assert compatibility == (
        "legacy_requested_default_field+warm_start_layout_randomized_default_field+"
        "student_decision_mask_legacy_default_field"
    )

    opt_in = SimConfig(student_decision_mask_mode="pre_action_forced_wait")
    historical = dict(vars(opt_in))
    historical.pop("student_decision_mask_mode")
    matched, compatibility = _environment_config_match(opt_in, _digest(historical))
    assert matched is False
    assert compatibility is None


def test_task_neutral_config_snapshot_keeps_legacy_shape_and_records_opt_in() -> None:
    legacy = _config_snapshot(SimConfig())
    opt_in = _config_snapshot(SimConfig(student_decision_mask_mode="pre_action_forced_wait"))
    assert "student_decision_mask_mode" not in legacy
    assert opt_in["student_decision_mask_mode"] == "pre_action_forced_wait"


def test_v24_two_tick_rollout_records_schema_mask_and_replays_exactly(tmp_path) -> None:
    config = _observation_config(schema=OBS_SCHEMA_V2_4)
    policy = StudentPPOV42(LOW_OBS_V2_4.dimension)
    validate_student_config(config)
    steps, result = rollout(
        policy, config, seed=303, max_steps=2,
        output=tmp_path / "v24_rollout", deterministic=True,
    )
    assert len(steps) == 2
    assert result["observation_schema"] == OBS_SCHEMA_V2_4
    assert result["observation_dimension"] == 90
    assert result["student_decision_mask_mode"] == "legacy"
    payload = json.loads((tmp_path / "v24_rollout" / "action_tapes").glob("*.json").__next__().read_text(encoding="utf-8"))
    policy_rng = SafetyTraceRecorder.decode_policy_rng_state(payload)
    assert policy_rng["observation_schema"] == OBS_SCHEMA_V2_4
    assert policy_rng["observation_dimension"] == 90
    assert policy_rng["student_decision_mask_mode"] == "legacy"
    replay = SafetyTraceRecorder.replay_action_tape(
        TerminalParallelEnv(config), result["action_tape"],
    )
    assert replay["exact"] is True
    assert replay["steps_replayed"] == replay["steps_expected"] == 2
