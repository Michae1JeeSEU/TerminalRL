"""Opt-in external release ordering retains native execution authority."""
from dataclasses import replace
import pickle

import numpy as np
import pytest

from terminal_marl.core.config import SimConfig
from terminal_marl.envs.parallel_env import TerminalParallelEnv


MODES = ("legacy_insertion", "downstream_priority", "downstream_soc_priority")
FIELDS = {"dispatch_order", "dispatch_order_requested", "normal_approved_releases"}


def make_env(mode="external_dispatch_v1", count=3):
    return TerminalParallelEnv(replace(
        SimConfig(), reset_mode="warm_start_rings", warm_start_uavs=count,
        total_requested_uavs=count, fixed_arrival_schedule=(0.0,) * count,
        release_arbitration_mode=mode, rule_flow_coordination_enabled=False,
    ), seed=896091)


def waits(env):
    return {str(uid): np.array([0., 0., 0.]) for uid in env.active}


@pytest.mark.parametrize("mode", MODES)
def test_legacy_none_matches_omitted_order_and_has_no_dispatch_metadata(mode):
    env = make_env(mode)
    replay = pickle.loads(pickle.dumps(env))
    result = env.step(waits(env))
    replay_result = replay.step(waits(replay), release_order=None)
    np.testing.assert_equal(result, replay_result)
    np.testing.assert_equal(env.runtime_trace, replay.runtime_trace)
    assert not FIELDS.intersection(result[-1])
    assert not FIELDS.intersection(env.runtime_trace[-1])
    assert SimConfig().release_arbitration_mode == "legacy_insertion"


@pytest.mark.parametrize("mode", MODES)
def test_legacy_order_rejected_before_any_mutation(mode):
    env = make_env(mode)
    before = pickle.dumps(env)
    with pytest.raises(ValueError, match="only supported"):
        env.step(waits(env), release_order=tuple(env.active))
    assert pickle.dumps(env) == before


@pytest.mark.parametrize("order", [None, (), (0, 1), (0, 1, 1), (0, 1, 99),
                                   (0, 1, 2.0), (False, 1, 2), (0, 1, "2"), 7,
                                   (np.bool_(False), 1, 2)])
def test_invalid_external_order_leaves_complete_state_unchanged(order):
    env = make_env()
    before = pickle.dumps(env)
    with pytest.raises(ValueError):
        env.step(waits(env), release_order=order)
    assert pickle.dumps(env) == before


def test_reverse_order_reaches_native_release_and_preserves_decisions(monkeypatch):
    env = make_env()
    requested = tuple(reversed(env.active))
    actual = []
    native_release = env._release

    def observe(uav, intent, ranked_intents=None):
        actual.append(uav.uid)
        return native_release(uav, intent, ranked_intents)

    monkeypatch.setattr(env, "_release", observe)
    info = env.step(waits(env), release_order=requested)[-1]
    assert actual == list(requested)
    assert info["dispatch_order_requested"] == list(requested)
    assert info["dispatch_order"] == actual == env.last_dispatch_order
    assert info["normal_approved_releases"] == []
    assert all(value == "not_requested" for value in info["release_decisions"].values())
    for key in FIELDS:
        assert env.runtime_trace[-1][key] == info[key]


def test_same_tick_admissions_append_after_requested_prefix(monkeypatch):
    env = make_env(count=5)
    new_uavs = {uid: env.active.pop(uid) for uid in (4, 3)}
    requested = tuple(reversed(env.active))
    original = env.admission.step

    def admit(*args, **kwargs):
        result = original(*args, **kwargs)
        # Reverse insertion demonstrates that new UIDs sort independently.
        for uid in (4, 3):
            env.active[uid] = new_uavs[uid]
        return result

    monkeypatch.setattr(env.admission, "step", admit)
    actions = {str(uid): np.array([0., 0., 0.]) for uid in range(5)}
    info = env.step(actions, release_order=requested)[-1]
    assert info["dispatch_order_requested"] == list(requested)
    assert info["dispatch_order"] == [*requested, 3, 4]


def test_processing_omits_removed_ids_and_appends_new_ids():
    env = make_env()
    requested = tuple(reversed(env.active))
    uav = env.active.pop(1)
    env.active[9] = uav
    assert [uid for uid, _ in env._release_processing_order(requested)] == [2, 0, 9]


def test_incompatible_flow_authority_rejected():
    with pytest.raises(ValueError, match="rule_flow_coordination_enabled=False"):
        replace(SimConfig(), release_arbitration_mode="external_dispatch_v1",
                rule_flow_coordination_enabled=True)


def test_reset_and_pickle_replay_with_numpy_integer_order():
    env = make_env()
    order = tuple(np.int64(uid) for uid in reversed(env.active))
    env.step(waits(env), release_order=order)
    replay = pickle.loads(pickle.dumps(env))
    order = tuple(reversed(env.active))
    np.testing.assert_equal(env.step(waits(env), release_order=order),
                            replay.step(waits(replay), release_order=order))
    np.testing.assert_equal(env.runtime_trace, replay.runtime_trace)
    env.reset(seed=896091)
    assert env.last_dispatch_order == []
    assert env.last_dispatch_order_requested is None
    fresh = make_env()
    order = tuple(reversed(env.active))
    np.testing.assert_equal(env.step(waits(env), release_order=order),
                            fresh.step(waits(fresh), release_order=order))


def test_empty_order_is_valid_for_empty_active_set():
    env = make_env(count=0)
    info = env.step({}, release_order=())[-1]
    assert all(info[key] == [] for key in FIELDS)


def test_native_approved_releases_are_separate_from_requested_and_processing_order():
    env = make_env()
    requested = tuple(reversed(env.active))
    actions = {str(uid): np.array([float(uid != 1), 0., 0.]) for uid in env.active}
    info = env.step(actions, release_order=requested)[-1]
    assert info["dispatch_order_requested"] == info["dispatch_order"] == [2, 1, 0]
    assert info["normal_approved_releases"] == [2, 0]
    assert info["release_decisions"]["1"] == "not_requested"
    for uid in (2, 0):
        assert env.active[uid].committed_plan is not None
        assert info["release_decisions"][str(uid)] == "approved"
