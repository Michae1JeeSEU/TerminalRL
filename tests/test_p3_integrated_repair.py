"""Regression contracts for the integrated repair; no training experiment."""
from dataclasses import replace
import math
from types import SimpleNamespace

import numpy as np
import pytest

from terminal_marl.configs.p3_integrated_repair import build_config, PPO_ALLOWED, PPO_UPDATE_OPTIONS
from terminal_marl.core.config import RingRole, SimConfig
from terminal_marl.core.merge_scheduler import LowLevelIntent
from terminal_marl.core.uav import ArrivalRequest, QueuedArrivalState, UAVState, MissionType, FlightPhase
from terminal_marl.core.vehicle_dynamics import VehicleDynamics
from terminal_marl.core.vehicle_profiles import RESEARCH_EVTOL
from terminal_marl.envs.parallel_env import TerminalParallelEnv, WorldSnapshot


def empty_env(**overrides):
    config = replace(build_config(requests=1, warm_start=0), total_requested_uavs=0,
                     fixed_arrival_schedule=(), **overrides)
    return TerminalParallelEnv(config)


def holder(env, *, uid=0, speed=16.0, soc=.8, angle=0.):
    request = ArrivalRequest(uid, 0., 0, MissionType.STANDARD, 1, 1500., soc)
    queued = QueuedArrivalState(request, soc*1500.)
    ring = env.airspace.actual_ring(RingRole.INNER)
    tangent = np.array([-math.sin(angle), math.cos(angle), 0.])
    radial = np.array([-math.cos(angle), -math.sin(angle), 0.])
    u = UAVState.from_request(uid, request, queued, -radial*ring.radius+np.array([0,0,ring.height]),0.)
    u.ring_role=RingRole.INNER; u.ring_progress=angle; u.phase=FlightPhase.RING_INNER
    u.speed=speed; u.velocity=tangent*speed; u.acceleration=radial*speed**2/ring.radius
    env.active[uid]=u
    return u


@pytest.mark.parametrize('start,target', [(math.sqrt(300),10.),(17.2,10.),(16.,math.sqrt(300)),(0.,math.sqrt(300)),(17.32,17.3)])
def test_boundary_ramp_has_continuous_acceleration_budget(start,target):
    dynamics=VehicleDynamics(RESEARCH_EVTOL,continuous_curve_limits=True)
    duration=dynamics.ring_transition_duration(start,target,200.)
    s=np.linspace(0,1,2001)
    speed=start+(target-start)*(3*s*s-2*s**3)
    tangent=(target-start)*6*s*(1-s)/duration
    total=np.hypot(tangent,speed**2/200.)
    assert total.max() <= 1.5+1e-9
    assert tangent[0]==tangent[-1]==0


def test_legal_acceleration_then_braking_recovers_from_inner_cap():
    dynamics=VehicleDynamics(RESEARCH_EVTOL,continuous_curve_limits=True)
    ring=SimpleNamespace(radius=200.,height=40.)
    theta,speed=0.,16.
    for _ in range(500):
        motion=dynamics.ring_motion_step(ring,theta,speed,18.,.2)
        theta,speed=motion.theta,motion.speed
        assert np.linalg.norm(motion.acceleration)<=1.5+1e-9
    assert speed==pytest.approx(math.sqrt(300),abs=1e-4)
    for _ in range(500):
        motion=dynamics.ring_motion_step(ring,theta,speed,16.,.2)
        theta,speed=motion.theta,motion.speed
    assert speed==pytest.approx(16.,abs=1e-4)


@pytest.mark.parametrize('speed',[17.1,17.2,math.sqrt(300)])
def test_generator_can_plan_deceleration_from_valid_boundary(speed):
    env=empty_env();u=holder(env,speed=speed);ring=env.airspace.actual_ring(u.ring_role)
    duration=env.generator.dynamics.ring_transition_duration(speed,10.,ring.radius)
    points=env.generator.ring_hold_points(ring,0.,speed,10.,0.,duration,initial_acceleration=u.acceleration)
    assert np.linalg.norm(points[-1].velocity)==pytest.approx(10.)
    np.testing.assert_allclose(points[0].acceleration,u.acceleration)
    from terminal_marl.core.polynomial_trajectory import QuinticSegment
    for left,right in zip(points,points[1:]):
        segment=QuinticSegment.from_boundary(left.t,right.t,left.position,left.velocity,left.acceleration,
                                            right.position,right.velocity,right.acceleration)
        assert segment.bounds()['max_horizontal_accel']<=1.5+1e-3


def test_nominal_contract_preserves_release_authority_and_matches_execution():
    env=empty_env();u=holder(env,speed=17.2)
    assert env.config.max_speed_residual_mps==2.
    times=np.arange(0,1.01,.2)
    low,high=env.safety.holder_reachable_tube(u,WorldSnapshot(0.,env.active,env.airspace),times)
    np.testing.assert_array_equal(low,high)
    actions={u.uid:LowLevelIntent(0.,1.,0.)}
    projected=env._next_state_prediction(actions)[u.uid]
    protected=env.runtime_traffic.protect(env,actions)
    assert protected[u.uid].speed_residual==0.
    env._advance_ring_hold(u,actions[u.uid])
    np.testing.assert_allclose(u.position,projected,atol=1e-8)
    assert u.ring_progress==pytest.approx(low[-1])


def test_nominal_contract_backup_never_uses_hidden_residual_authority(monkeypatch):
    env=empty_env();u=holder(env)
    observed=[]
    def assess(_env,residuals,**kwargs):
        observed.append(residuals)
        return SimpleNamespace(approved=False)
    monkeypatch.setattr(env.runtime_traffic,'_joint_assessment',assess)
    backup,_=env.runtime_traffic._backup_safe(env,{u.uid:2.},())
    assert backup is None and observed==[{u.uid:0.}]


def test_failed_urgent_route_does_not_virtually_claim_a_free_pad():
    env=empty_env();healthy=holder(env,uid=0);holder(env,uid=1,soc=.3,angle=math.pi)
    env.lz_manager.zones=env.lz_manager.zones[:1]
    assert not env._emergency_lz_priority_blocks(healthy)
    env.config.emergency_priority_mode='at_risk_count'
    assert env._emergency_lz_priority_blocks(healthy)


def test_route_deadline_can_trigger_before_fixed_soc_threshold(monkeypatch):
    env=empty_env();u=holder(env,soc=.6)
    monkeypatch.setattr(env.landing_budget,'holder_budget',lambda u,now:u.energy_remaining-10.)
    assert u.soc>env.config.emergency_trigger_soc
    assert env._route_deadline_due(u)
    assert env.energy_reachability_diagnostics[u.uid]['wait_route_slack_seconds']==pytest.approx(10.)


def test_admission_counts_requests_accepted_in_the_same_batch():
    env=empty_env(max_active_uavs=1)
    for i in range(3):
        env.admission.enqueue(ArrivalRequest(i,0.,i,MissionType.STANDARD,1,1500.,1.))
    accepted,records=env.admission.step(0.,{},env.airspace,env.energy,env.priority,
        lz_manager=env.lz_manager,executable_energy_budget=lambda q,pending:200.)
    assert len(accepted)==1
    assert [r.reason_code for r in records].count('active_cap')==2


def test_unavailable_route_waits_but_does_not_bypass_external_wait_limit():
    env=empty_env(max_external_wait_seconds=1.)
    env.admission.enqueue(ArrivalRequest(0,0.,0,MissionType.STANDARD,1,1500.,1.))
    kwargs=dict(lz_manager=env.lz_manager,executable_energy_budget=lambda q,pending:math.inf)
    accepted,records=env.admission.step(0.,{},env.airspace,env.energy,env.priority,**kwargs)
    assert not accepted and records[0].reason_code=='executable_route_unavailable'
    env.admission.queue.items[0].external_wait_time=1.
    accepted,records=env.admission.step(1.,{},env.airspace,env.energy,env.priority,**kwargs)
    assert not accepted and records[0].reason_code=='external_wait_limit'


def test_upstream_cost_is_team_only_and_charged_once(monkeypatch):
    env=empty_env()
    env.admission.enqueue(ArrivalRequest(99,0.,0,MissionType.STANDARD,1,1500.,1.))
    monkeypatch.setattr(env.admission,'step',lambda *a,**kw:([],[]))
    _,rewards,_,_,info=env.step({})
    assert rewards=={}
    assert info['team_reward_components']=={'external_queue':pytest.approx(-.01)}
    assert info['team_reward']==pytest.approx(-.01)
    assert env.external_queue_uav_seconds==1.


def test_new_profile_is_explicit_and_ppo_stays_locked():
    new=build_config();old=SimConfig()
    assert not PPO_ALLOWED
    assert new.observation_schema_version.endswith('90d')
    assert new.student_decision_mask_mode=='pre_action_forced_wait'
    assert new.low_level_release_rejected_penalty==0.
    assert old.holding_speed_contract=='bounded_residual'
    assert not old.low_level_queue_cost_enabled
    assert PPO_UPDATE_OPTIONS['entropy_mode']=='gate_only'



def test_actual_v3_solo_route_is_finite_and_does_not_use_old_calibration():
    env=empty_env()
    q=QueuedArrivalState(ArrivalRequest(0,0.,0,MissionType.STANDARD,1,1500.,1.),1500.)
    used,duration=env.landing_budget._solo_route(q,env.admission.gates[0],env.lz_manager.zones[0])
    assert math.isfinite(used) and used>0
    assert math.isfinite(duration) and duration>0
    assert not env.active and all(z.reserved_by is None for z in env.lz_manager.zones)
    env.config.phase_motion_nominal_solo_energy=1e9
    assert env.landing_budget._solo_route(q,env.admission.gates[0],env.lz_manager.zones[0])==(used,duration)


def test_pad_workload_counts_pending_admissions_without_mutating_resources(monkeypatch):
    env=empty_env();env.lz_manager.zones=env.lz_manager.zones[:1]
    q=QueuedArrivalState(ArrivalRequest(0,0.,0,MissionType.STANDARD,1,1500.,1.),1500.)
    gate=env.admission.gates[0]
    monkeypatch.setattr(env.landing_budget,'_solo_route',lambda *a:(100.,10.))
    before=env.landing_budget.admission_budget(q,gate,0.,{},[])
    after=env.landing_budget.admission_budget(q,gate,0.,{},[(q,gate)]*3)
    assert after>before
    assert env.lz_manager.zones[0].reserved_by is None



def test_gate_only_entropy_does_not_include_frozen_proposal_entropy():
    import torch
    from terminal_marl.algo.student_ppo_v4_2 import StudentPPOV42, StudentStepV42
    policy=StudentPPOV42(88)
    world=np.zeros((2,88),dtype=np.float32)
    batch=policy.act_details(world,world)
    step=StudentStepV42(world,batch.requested_gate,
        np.asarray([-1 if p is None else int(p) for p in batch.requested_proposal]),
        batch.gate_logprob,np.asarray([np.nan if p is None else p for p in batch.proposal_logprob]),
        np.ones(2,dtype=bool),batch.value,0.,0.,True,True)
    with torch.no_grad():
        gate,proposal=policy.model.distributions(torch.as_tensor(world))
        expected=float(gate.entropy().mean())
        joint=float((gate.entropy()+gate.probs[:,1]*proposal.entropy()).mean())
    metrics=policy.update([step],training_authorized=True,epochs=1,minibatch_size=1,
                          entropy_mode='gate_only',actor_reduction='joint_team')
    assert metrics['entropy']==pytest.approx(expected)
    assert metrics['entropy']<joint


def test_new_admission_spends_interval_in_flight_not_also_in_external_queue():
    env=TerminalParallelEnv(build_config(requests=1,warm_start=0,high_soc=True),seed=890301)
    _,rewards,_,_,info=env.step({})
    assert len(env.active)==1 and not env.admission.queue.items
    u=next(iter(env.active.values()))
    assert u.external_wait_time==0.
    expected=env.energy.estimate_plan_energy(u.committed_plan,0.,env.config.dt)
    assert u.energy_consumed==pytest.approx(expected)
    assert env.external_queue_uav_seconds==0.
    assert info['team_reward_components']['external_queue']==0.
    assert info['reward_components'][str(u.uid)]['time']==pytest.approx(-.01)
    assert info['team_reward']==pytest.approx(sum(rewards.values()))


def test_failed_inbound_commit_spends_interval_in_queue_once(monkeypatch):
    env=TerminalParallelEnv(build_config(requests=1,warm_start=0,high_soc=True),seed=890301)
    monkeypatch.setattr(env,'_commit',lambda *a,**kw:False)
    _,rewards,_,_,info=env.step({})
    assert not env.active and len(env.admission.queue.items)==1
    q=env.admission.queue.items[0]
    assert q.external_wait_time==env.config.dt
    assert q.energy_consumed==pytest.approx(env.energy.phase_power(FlightPhase.EXTERNAL_QUEUE)*env.config.dt)
    assert env.external_queue_uav_seconds==env.config.dt
    assert rewards=={}
    assert info['team_reward']==pytest.approx(-.01)


def test_legacy_low_soc_diversion_precedence_is_preserved():
    env=empty_env(executable_admission_energy_enabled=False,max_external_wait_seconds=1.)
    env.admission.enqueue(ArrivalRequest(0,0.,0,MissionType.STANDARD,1,1500.,.01))
    env.admission.queue.items[0].external_wait_time=1.
    accepted,records=env.admission.step(1.,{},env.airspace,env.energy,env.priority,
                                      lz_manager=env.lz_manager)
    assert not accepted and records[0].reason_code=='low_soc'
