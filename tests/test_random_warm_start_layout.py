import copy
import json

import numpy as np

from terminal_marl.core.config import SimConfig
from terminal_marl.core.low_level_joint_action import N_JOINT_ACTIONS, joint_action
from terminal_marl.core.safety_trace import (
    _digest,
    _environment_config_match,
    SafetyTraceRecorder,
)
from terminal_marl.envs.parallel_env import TerminalParallelEnv


def _config(count: int, *, randomized: bool = False) -> SimConfig:
    return SimConfig(
        action_schema_version="low_action_schema_v4_joint_discrete",
        reset_mode="warm_start_rings",
        warm_start_uavs=count,
        warm_start_layout_randomized=randomized,
        fixed_arrival_schedule=(0.0,) * count,
        total_requested_uavs=count,
        max_simulation_seconds=2.0,
    )


def _layout(env: TerminalParallelEnv) -> tuple[tuple[int, str, float, tuple[float, ...]], ...]:
    return tuple(
        (
            int(uid),
            uav.ring_role.value,
            float(uav.ring_progress),
            tuple(float(value) for value in uav.position),
        )
        for uid, uav in sorted(env.active.items())
    )


def _reset_layout(config: SimConfig, seed: int) -> tuple[tuple[int, str, float, tuple[float, ...]], ...]:
    env = TerminalParallelEnv(config)
    env.reset(seed)
    return _layout(env)


def _assert_pairwise_margin(layout: tuple[tuple[int, str, float, tuple[float, ...]], ...], margin: float) -> None:
    points = [np.asarray(row[3], dtype=np.float64) for row in layout]
    distances = [
        float(np.linalg.norm(points[first] - points[second]))
        for first in range(len(points))
        for second in range(first + 1, len(points))
    ]
    assert distances
    assert min(distances) >= margin - 1e-9


def test_randomized_layout_repeats_by_seed_and_is_safe_for_12_and_30() -> None:
    for count in (12, 30):
        config = _config(count, randomized=True)
        first = _reset_layout(config, 17)
        second = _reset_layout(config, 17)
        assert first == second
        assert len(first) == count
        _assert_pairwise_margin(first, 1.2 * config.safe_distance)

    layouts = {_reset_layout(_config(12, randomized=True), seed) for seed in (1, 2, 3)}
    assert len(layouts) > 1


def test_randomized_layout_uses_independent_arrival_and_global_rng() -> None:
    seed = 23
    np.random.seed(101)
    before = copy.deepcopy(np.random.get_state())
    randomized = TerminalParallelEnv(_config(12, randomized=True))
    randomized.reset(seed)
    after = np.random.get_state()
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]

    deterministic = TerminalParallelEnv(_config(12, randomized=False))
    deterministic.reset(seed)
    assert json.dumps(randomized.arrivals.rng.bit_generator.state, sort_keys=True) == json.dumps(
        deterministic.arrivals.rng.bit_generator.state, sort_keys=True
    )


def test_default_warm_start_layout_stays_on_existing_deterministic_path() -> None:
    config = _config(12)
    env = TerminalParallelEnv(config)
    env.reset(29)
    expected = env._safe_warm_start_layout(config.warm_start_uavs)
    actual = [(uav.ring_role, uav.ring_progress) for _, uav in sorted(env.active.items())]
    assert actual == expected


def test_randomized_layout_trace_replays_exactly_and_hash_compatibility_is_narrow(tmp_path) -> None:
    config = _config(12, randomized=True)
    env = TerminalParallelEnv(config)
    observations, _ = env.reset(31)
    tape = SafetyTraceRecorder(tmp_path, dt=config.dt)
    tape.start_episode(env, episode_seed=31, policy_rng_state={})
    for step in range(2):
        actions = {
            uid: joint_action(step % N_JOINT_ACTIONS).normalized
            for uid in observations
        }
        record = tape.record_pre_step(env, observations, actions)
        observations, _, _, _, info = env.step(actions)
        tape.record_post_step(env, record, info)
    path = tape.dump(reason="randomized_layout_test")
    replay = SafetyTraceRecorder.replay_action_tape(TerminalParallelEnv(config), path)
    assert replay["exact"] is True
    assert replay["steps_replayed"] == replay["steps_expected"] == 2

    old_values = dict(vars(config))
    old_values.pop("warm_start_layout_randomized")
    legacy_match, compatibility = _environment_config_match(config, _digest(old_values))
    assert legacy_match is False
    assert compatibility is None

    default_config = _config(12)
    historical_values = dict(vars(default_config))
    historical_values.pop("warm_start_layout_randomized")
    historical_match, historical_compatibility = _environment_config_match(
        default_config,
        _digest(historical_values),
    )
    assert historical_match is True
    assert historical_compatibility == "warm_start_layout_randomized_default_field"


def test_canonical_wait_trace_missing_randomized_default_field_replays_exactly(tmp_path) -> None:
    config = _config(12)
    config.v4_2_rejected_release_hold_mode = "canonical_wait"
    env = TerminalParallelEnv(config)
    observations, _ = env.reset(37)
    tape = SafetyTraceRecorder(tmp_path, dt=config.dt)
    tape.start_episode(env, episode_seed=37, policy_rng_state={})
    actions = {uid: joint_action(0).normalized for uid in observations}
    record = tape.record_pre_step(env, observations, actions)
    _, _, _, _, info = env.step(actions)
    tape.record_post_step(env, record, info)
    path = tape.dump(reason="canonical_wait_legacy_hash_test")

    payload = json.loads(path.read_text(encoding="utf-8"))
    historical_values = dict(vars(config))
    historical_values.pop("warm_start_layout_randomized")
    payload["environment_config_hash"] = _digest(historical_values)
    path.write_text(json.dumps(payload), encoding="utf-8")

    replay = SafetyTraceRecorder.replay_action_tape(TerminalParallelEnv(config), path)
    assert replay["exact"] is True
    assert replay["config_compatibility"] == "warm_start_layout_randomized_default_field"
