from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest

from terminal_marl.configs.p3_integrated_repair import build_config
from terminal_marl.core import trajectory_context_observation as context
from terminal_marl.core.config import RingRole
from terminal_marl.core.observations import LOW_OBS_V2_4, build_low_observation_v2_4
from terminal_marl.core.uav import FlightPhase
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.scripts.diagnose_p3_direct_contract import canonical, state_record


@pytest.fixture
def env():
    config = replace(build_config(requests=6, warm_start=6, high_soc=True),
                     warm_start_energy_reachability_check_enabled=False)
    env = TerminalParallelEnv(config)
    env.reset(112233)
    return env


def test_new_api_preserves_old_observations_and_state_without_queries(env, monkeypatch):
    before = canonical(state_record(env))
    old = env._observations()
    def forbidden(*a, **kw):
        raise AssertionError('observation must not query an action or advance the world')
    monkeypatch.setattr(env, '_candidate_plan', forbidden)
    monkeypatch.setattr(env, 'step', forbidden)
    monkeypatch.setattr(env, 'reset', forbidden)
    monkeypatch.setattr(context.SafetyFilter, 'filter_plan', forbidden)
    result = env.trajectory_context_observations()
    assert canonical(state_record(env)) == before
    for uid, obs in result.items():
        assert obs.ego.shape == (105,) and obs.traffic.shape == (5, 50)
        assert set(obs.traffic_uids) == set(env.active)-{int(uid)}
        assert np.array_equal(old[uid], env._observations()[uid])
        assert np.isfinite(obs.ego).all() and np.isfinite(obs.traffic).all()


def test_relative_target_gap_rotation_and_cache_agreement(env):
    ego = env.active[4]
    assert ego.ring_role is RingRole.MIDDLE
    first = context.build_low_observation_v2_5(ego, env)
    s = context.EGO_SCHEMA.slices['target_reference_successor']
    assert first.ego[s][1] == pytest.approx(-1.)
    assert build_low_observation_v2_4(ego, env)[LOW_OBS_V2_4.slices['target_ring_successor']][1] == pytest.approx(-.925)
    angle = .4
    rotate = np.array([[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    for uav in env.active.values():
        uav.position = rotate@uav.position
        uav.velocity = rotate@uav.velocity
        uav.ring_progress += angle
    cache = context.TrajectoryContextCache(env)
    rotated = cache.build(ego)
    assert rotated.ego[s][1] == pytest.approx(first.ego[s][1])
    np.testing.assert_allclose(first.traffic, rotated.traffic, atol=1e-6, rtol=0)
    assert np.array_equal(rotated.ego, context.build_low_observation_v2_5(ego, env).ego)


class IncomingPlan:
    def __init__(self, uav, end=5., target='ring_inner'):
        self.plan_id = f'test_{uav.uid}'
        self.trajectory_hash = f'hash_{uav.uid}'
        self.end_time = end
        self.metadata = dict(target_phase=target)
        self.points = [NS(position=np.array([200., 0., 40.]), velocity=np.array([0., 16., 0.]))]
    def sample_positions(self, times):
        return np.repeat(self.points[-1].position[None, :], len(times), axis=0)


@pytest.mark.parametrize('end,selected', [(5., True), (18.5, True), (18.5001, False)])
def test_incoming_is_selected_at_reference_handoff_and_always_in_set(env, end, selected):
    ego, other = env.active[4], env.active[5]
    other.committed_plan = IncomingPlan(other, end=end)
    other.phase, other.ring_role = FlightPhase.TRANSFER_M2I, None
    result = context.build_low_observation_v2_5(ego, env)
    assert (5 in result.target_uids) is selected
    assert 5 in result.traffic_uids
    row = result.traffic[result.traffic_uids.index(5)]
    assert row[context.TRAFFIC_SCHEMA.slices['committed']][0] == 1
    assert row[context.TRAFFIC_SCHEMA.slices['committed_target']].tolist() == [0., 0., 1., 0.]
    assert row[context.TRAFFIC_SCHEMA.slices['plan_remaining']][0] == pytest.approx(end/(end+150))


def test_multiple_committed_neighbors_survive_container_order(env):
    for uid in (1, 2, 5):
        env.active[uid].committed_plan = IncomingPlan(env.active[uid])
        env.active[uid].ring_role = None
    result = context.build_low_observation_v2_5(env.active[4], env)
    env.active = dict(reversed(list(env.active.items())))
    reordered = context.build_low_observation_v2_5(env.active[4], env)
    assert result.traffic_uids == reordered.traffic_uids
    assert np.array_equal(result.traffic, reordered.traffic)
    assert {1, 2, 5} <= set(result.traffic_uids)


def test_forecast_reaches_post_commit_ring_and_no_distance_clipping(env):
    other = env.active[5]
    other.committed_plan = IncomingPlan(other)
    other.ring_role = None
    cache = context.TrajectoryContextCache(env)
    assert np.linalg.norm(cache.forecasts[5][-1][:2]) == pytest.approx(200.)
    assert not np.allclose(cache.forecasts[5][0], cache.forecasts[5][-1])
    other.position = np.array([2000., 0., 40.])
    result = context.build_low_observation_v2_5(env.active[4], env)
    row = result.traffic[result.traffic_uids.index(5)]
    assert np.max(np.abs(row[context.TRAFFIC_SCHEMA.slices['relative_position']])) > 1.


def test_padding_empty_traffic_and_metadata_excluded(env):
    full = context.build_low_observation_v2_5(env.active[4], env)
    env.active = {4: env.active[4]}
    empty = context.build_low_observation_v2_5(env.active[4], env)
    ego, neighbors, mask = context.pad_trajectory_context([full, empty])
    assert ego.shape == (2, 105) and neighbors.shape == (2, 5, 50)
    assert mask[0].all() and not mask[1].any() and not neighbors[1].any()
    assert empty.traffic.shape == (0, 50)
