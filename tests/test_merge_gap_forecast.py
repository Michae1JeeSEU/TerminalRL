import math
from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest

from terminal_marl.configs.p3_integrated_repair import build_config
from terminal_marl.core.config import RingRole
from terminal_marl.core.merge_gap_forecast import target_ring_angles, largest_gap_center
from terminal_marl.core.safety_filter import SafetyFilter
from terminal_marl.core.uav import FlightPhase
from terminal_marl.envs.parallel_env import TerminalParallelEnv, WorldSnapshot
from terminal_marl.scripts.diagnose_p3_direct_contract import canonical, state_record


@pytest.fixture
def env():
    config=replace(build_config(requests=6,warm_start=6,high_soc=True),
                   warm_start_energy_reachability_check_enabled=False)
    return TerminalParallelEnv(config,seed=112233)


class Plan:
    def __init__(self,end,target='ring_inner'):
        self.end_time=end
        self.metadata={'target_phase':target}
        self.plan_id='test_plan';self.trajectory_hash='test_hash'
        self.points=[NS(position=np.array([200.,0.,40.]),velocity=np.array([0.,16.,0.]))]
    def sample_positions(self,times):
        return np.repeat(self.points[-1].position[None,:],len(times),axis=0)


@pytest.mark.parametrize('end,expected',[(10.,True),(20.,True),(20.0001,False)])
def test_incoming_membership_uses_requested_slot_not_current_role(env,end,expected):
    ego,other=env.active[4],env.active[5]
    env.active={4:ego,5:other}
    other.committed_plan=Plan(end);other.ring_role=None;other.phase=FlightPhase.TRANSFER_M2I
    snap=WorldSnapshot(env.now,env.active,env.airspace)
    actual=target_ring_angles(env.config,ego,RingRole.INNER,20.,snap)
    assert bool(actual) is expected
    if expected:
        position=SafetyFilter(env.config)._committed_positions(other.committed_plan,snap,np.array([20.]))[0]
        assert actual==pytest.approx([math.atan2(position[1],position[0])%(2*math.pi)])


def test_committed_departure_is_not_indefinite_target_holder(env):
    ego,other=env.active[4],env.active[5]
    env.active={4:ego,5:other};other.committed_plan=Plan(15.,'landing')
    other.ring_role=RingRole.INNER
    assert target_ring_angles(env.config,ego,RingRole.INNER,20.,WorldSnapshot(env.now,env.active,env.airspace))==[]


def test_future_holder_phase_uses_finite_response_and_speed_bias(env,monkeypatch):
    ego,other=env.active[4],env.active[5]
    env.active={4:ego,5:other};other.speed=7.;env.airspace.speed_profile_bias=-1.
    snap=WorldSnapshot(env.now,env.active,env.airspace)
    before=canonical(state_record(env))
    def forbidden(*a,**k):raise AssertionError('forecast must not query candidates or change the world')
    monkeypatch.setattr(env,'step',forbidden);monkeypatch.setattr(env,'_candidate_plan',forbidden)
    monkeypatch.setattr(SafetyFilter,'filter_plan',forbidden)
    angle=target_ring_angles(env.config,ego,RingRole.INNER,20.,snap)[0]
    position=SafetyFilter(env.config)._holding_prediction(other,snap,20.)
    assert angle==pytest.approx(math.atan2(position[1],position[0])%(2*math.pi))
    legacy=(other.ring_progress+env.airspace.actual_ring(RingRole.INNER).nominal_speed*20/200)%(2*math.pi)
    assert abs(angle-legacy)>.01
    assert before==canonical(state_record(env))


def test_gap_midpoint_wraps_and_is_independent_of_container_order(env):
    ego=env.active[0];snap=WorldSnapshot(env.now,env.active,env.airspace)
    angles=target_ring_angles(env.config,ego,RingRole.MIDDLE,70.,snap)
    snap.active=dict(reversed(list(env.active.items())))
    assert target_ring_angles(env.config,ego,RingRole.MIDDLE,70.,snap)==angles
    assert largest_gap_center([2*math.pi-.1])==pytest.approx(math.pi-.1)
    assert largest_gap_center([.2,1.,2.])==pytest.approx((2+(2*math.pi+.2-2)/2)%(2*math.pi))
    with pytest.raises(ValueError):largest_gap_center([])
