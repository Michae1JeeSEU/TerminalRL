from copy import deepcopy
from dataclasses import replace

import numpy as np

from terminal_marl.configs.p3_v25_course import build_config
from terminal_marl.core.landing_zone import LZStatus
from terminal_marl.core.resource_context_observation import ResourceContextCache, PAD_FEATURES
from terminal_marl.core.trajectory_context_observation import TrajectoryContextCache
from terminal_marl.core.uav import FlightPhase
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.scripts.diagnose_p3_v25_lz_alias import swap_ground_service


def state():
    config=replace(build_config('dense24_fixed'),total_requested_uavs=3,
                   warm_start_uavs=3,fixed_arrival_schedule=(0.,)*3)
    env=TerminalParallelEnv(config,seed=981)
    ground=deepcopy(env.active[0]);ground.uid=999;ground.assigned_lz=4
    ground.phase=FlightPhase.SERVICE;ground.position=env.lz_manager.zones[4].position.copy()
    env.service_uavs[999]=ground
    pad=env.lz_manager.zones[4];pad.status=LZStatus.OCCUPIED
    pad.occupied_by=999;pad.service_remaining=17.
    return env


def test_pad_identity_resolves_aggregate_alias_and_preserves_v25_prefix():
    env=state();other=deepcopy(env);swap_ground_service(other,5,4)
    old=TrajectoryContextCache(env).build(env.active[0]);old_other=TrajectoryContextCache(other).build(other.active[0])
    np.testing.assert_array_equal(old.ego,old_other.ego)
    new=ResourceContextCache(env).build(env.active[0]);new_other=ResourceContextCache(other).build(other.active[0])
    assert new.ego.shape==(159,) and not np.array_equal(new.ego,new_other.ego)
    np.testing.assert_array_equal(new.ego[:105],old.ego)
    np.testing.assert_array_equal(new.traffic,old.traffic)
    np.testing.assert_array_equal(new.traffic,new_other.traffic)
    np.testing.assert_array_equal(new.traffic_mask,old.traffic_mask)


def test_known_eta_and_unknown_reservation_are_distinct_without_queries(monkeypatch):
    env=state();pad=env.lz_manager.zones[2]
    pad.status=LZStatus.RESERVED;pad.reserved_by=123;pad.reservation_start=env.now
    def forbidden(*args,**kwargs):raise AssertionError('candidate generation queried')
    monkeypatch.setattr(env.generator,'final_approach',forbidden)
    before=ResourceContextCache(env).build(env.active[0]).ego[105:].reshape(6,len(PAD_FEATURES))
    assert before[2,7]==0 and before[2,8]==0
    pad.expected_release_at=env.now+30.
    after=ResourceContextCache(env).build(env.active[0]).ego[105:].reshape(6,len(PAD_FEATURES))
    assert after[2,7]==1 and np.isclose(after[2,8],30/180)
    assert after[4,5]==1 and np.isclose(after[4,8],17/167)
    assert after[0,3]==1 and after[0,7]==1 and after[0,8]==0
def test_resource_actor_schema_and_old_weight_shape_are_explicit():
    import pytest
    import torch
    from terminal_marl.algo.trajectory_set_actor import TrajectorySetActor
    from terminal_marl.core.resource_context_observation import VERSION
    old=TrajectorySetActor();new=TrajectorySetActor(ego_dim=159)
    assert new.observation_schema==VERSION!=old.observation_schema
    gate,proposal=new(torch.zeros(2,159),torch.zeros(2,0,50),torch.zeros(2,0,dtype=torch.bool))
    assert gate.shape==(2,2) and proposal.shape==(2,25)
    with pytest.raises(ValueError):new(torch.zeros(2,105),torch.zeros(2,0,50),torch.zeros(2,0,dtype=torch.bool))
    with pytest.raises(RuntimeError):new.load_state_dict(old.state_dict())
