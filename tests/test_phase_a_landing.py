import math
import numpy as np
import pytest

from terminal_marl.core.config import SimConfig, RingRole
from terminal_marl.core.dynamic_airspace import DynamicAirspace
from terminal_marl.core.ring_capacity import geometric_ring_capacity, minimum_radius_for_occupancy
from terminal_marl.core.uav import ArrivalRequest, MissionType, QueuedArrivalState, TerminalOutcome, FlightPhase, UAVState
from terminal_marl.core.arrival_process import ArrivalProcess
from terminal_marl.core.admission import AdmissionManager
from terminal_marl.core.energy import EnergyModel
from terminal_marl.core.priority import PriorityEngine
from terminal_marl.core.landing_zone import LandingZoneManager, LZStatus
from terminal_marl.core.lifecycle import transition
from terminal_marl.core.merge_scheduler import LowLevelIntent
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.core.checkpoints import validate_checkpoint_metadata, IncompatibleCheckpointError


def finite_config(**overrides):
    data = {"arrival_model": "finite_cohort", "total_requested_uavs": 1, "max_simulation_seconds": 2000.0}
    data.update(overrides)
    return SimConfig(**data)


def test_a0_schema_and_action_dimensions_are_locked():
    c = SimConfig()
    assert (c.schema_version, c.airspace_schema_version) == ("phaseA_landing_v1", "three_ring_v1")
    assert (c.action_schema_version, c.high_action_schema_version) == ("low_intent_v1_3d", "high_airspace_v1_10d")
    assert DynamicAirspace.ACTION_SIZE == 10
    assert LowLevelIntent.from_array([0, 0, 0])
    with pytest.raises(ValueError): LowLevelIntent.from_array([0, 0, 0, 0])
    with pytest.raises(IncompatibleCheckpointError): validate_checkpoint_metadata({"schema_version":"legacy"})


def test_a1_three_ring_order_capacity_and_occupancy_projection():
    c=SimConfig(); a=DynamicAirspace(c)
    assert [a.actual_ring(r).radius for r in RingRole] == [600,400,200]
    assert [a.actual_ring(r).height for r in RingRole] == [500,350,200]
    assert geometric_ring_capacity(600,80)>geometric_ring_capacity(400,80)>geometric_ring_capacity(200,80)
    n=4; assert geometric_ring_capacity(minimum_radius_for_occupancy(n,80),80)>=n
    result=a.request_update([-1,-1,-1,0,0,0,0,0,0,0],now=600,occupancies={RingRole.OUTER:0,RingRole.MIDDLE:0,RingRole.INNER:3},committed_incoming={RingRole.INNER:2})
    assert result.structural_applied and a.target_ring(RingRole.INNER).radius>=minimum_radius_for_occupancy(5,80)
    assert {"o2m","m2i","i2f"} == a.affected_interfaces()
    a.step_transition(620); assert not a.affected_interfaces()


def test_a1_priority_energy_lifecycle_and_lz_service():
    c=SimConfig(); e=EnergyModel(c); p=PriorityEngine(c)
    req=ArrivalRequest(1,0,0,MissionType.CARGO,3,100,1.0); q=QueuedArrivalState(req,100)
    before=p.compute(q); e.consume(q,FlightPhase.EXTERNAL_QUEUE,20); q.external_wait_time=3600; after=p.compute(q)
    assert q.energy_remaining<100 and after>before
    m=LandingZoneManager(); assert len(m.zones)==6 and all(MissionType.CARGO in z.compatible_missions for z in m.zones)
    assert m.reserve(0,7,0).approved and not m.reserve(0,8,0).approved
    m.touchdown(0,7,MissionType.CARGO); assert m.zones[0].status is LZStatus.OCCUPIED
    assert m.step(119)==[] and m.step(1)==[7] and m.zones[0].status is LZStatus.FREE


def test_a2_common_random_numbers_initial_soc_and_four_physical_gates():
    c=finite_config(total_requested_uavs=4,arrival_rate_per_second=1.0)
    p1,p2=ArrivalProcess(c,42),ArrivalProcess(c,42)
    stream1,stream2=[],[]
    for t in range(10): stream1.extend(p1.requests_until(t)); stream2.extend(p2.requests_until(t))
    assert stream1==stream2 and all(c.soc_reserve<=r.initial_soc<=1.0 for r in stream1)
    admission=AdmissionManager(c); points=[tuple(np.round(g.spawn_point,3)) for g in admission.gates]
    assert len(set(points))==4 and {round(math.degrees(g.angle))%360 for g in admission.gates}=={0,90,180,270}


def test_a2_external_queue_is_not_active_and_consumes_soc():
    c=finite_config(max_active_uavs=1); a=DynamicAirspace(c); manager=AdmissionManager(c); energy=EnergyModel(c); priority=PriorityEngine(c)
    r=ArrivalRequest(1,0,0,MissionType.STANDARD,1,10000,0.8); manager.enqueue(r); q=manager.queue.items[0]; manager.queue.step(10,energy,priority,10)
    assert q.energy_remaining<8000 and q.external_wait_time==10
    admitted,_=manager.step(10,{99:object()},a,energy,priority)
    assert not admitted and len(manager.queue.items)==1


def test_a3_canonical_plan_is_checked_and_executed_with_tangent_transfer():
    env=TerminalParallelEnv(finite_config())
    for _ in range(100):
        env.step()
        transfers=[u for u in env.active.values() if u.committed_plan and u.committed_plan.mode=="ring_transfer"]
        if transfers: break
    assert transfers
    plan=transfers[0].committed_plan
    assert plan.plan_id in env.safety.checked_plan_ids
    assert plan.metadata["source_tangent"].shape==(3,) and plan.metadata["target_tangent"].shape==(3,)
    assert not env.generator.validate_kinematics(plan)


def test_a4_single_uav_touchdown_exits_active_then_service_completes():
    env=TerminalParallelEnv(finite_config())
    ledger=env.run_rule_only()
    assert ledger.outcomes=={0:TerminalOutcome.COMPLETED}
    touchdown=[event for event in env.events if event[0]=="touchdown"][0]; completed=[event for event in env.events if event[0]=="completed"][0]
    assert completed[2]>touchdown[2] and not env.active and not env.service_uavs and ledger.closes()


def test_a4_multi_uav_rule_pipeline_closes_without_lz_double_booking():
    c=finite_config(total_requested_uavs=10,arrival_rate_per_second=.01,max_simulation_seconds=4000)
    env=TerminalParallelEnv(c); ledger=env.run_rule_only()
    assert len(ledger.outcomes)==10 and ledger.closes()
    assert all(outcome is TerminalOutcome.COMPLETED for outcome in ledger.outcomes.values())
    assert all(z.status is LZStatus.FREE for z in env.lz_manager.zones)
