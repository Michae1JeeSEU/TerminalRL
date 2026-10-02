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
    assert (c.action_schema_version, c.high_action_schema_version) == ("low_action_schema_v4_joint_discrete", "high_airspace_v1_10d")
    assert DynamicAirspace.ACTION_SIZE == 10
    assert LowLevelIntent.from_array([0, 0, 0])
    with pytest.raises(ValueError): LowLevelIntent.from_array([0, 0, 0, 0])
    with pytest.raises(IncompatibleCheckpointError): validate_checkpoint_metadata({"schema_version":"legacy"})


def test_a1_three_ring_order_capacity_and_occupancy_projection():
    c=SimConfig(); a=DynamicAirspace(c)
    assert [a.actual_ring(r).radius for r in RingRole] == [600,400,200]
    assert [a.actual_ring(r).height for r in RingRole] == [100,70,40]
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


def test_ehang_reference_endurance_and_profile_landing_floor():
    c = SimConfig()
    e = EnergyModel(c)
    assert c.energy_capacity == pytest.approx(c.vehicle_profile.reference_max_flight_time_s)
    assert e.min_landing_soc == pytest.approx(c.vehicle_profile.min_landing_soc)
    request = ArrivalRequest(1, 0, 0, MissionType.STANDARD, 1, c.energy_capacity, 1.0)
    queued = QueuedArrivalState(request, c.energy_capacity)
    for _ in range(int(c.energy_capacity)):
        e.consume(queued, FlightPhase.RING_INNER, 1.0)
    assert queued.energy_consumed == pytest.approx(c.energy_capacity)
    assert queued.energy_remaining == pytest.approx(0.0)
    queued.energy_remaining = c.energy_capacity * c.min_landing_soc
    assert e.has_landing_reserve(queued)
    queued.energy_remaining -= 1.0
    assert not e.has_landing_reserve(queued)


def test_active_low_soc_diversion_is_non_completion_and_not_double_resolved():
    c = finite_config(
        total_requested_uavs=0,
        arrival_rate_per_second=0.0,
    )
    env = TerminalParallelEnv(c)
    initial_soc = c.min_landing_soc + (
        0.5 * EnergyModel(c).phase_power(FlightPhase.RING_INNER) * c.dt / c.energy_capacity
    )
    request = ArrivalRequest(1, 0, 0, MissionType.STANDARD, 1, c.energy_capacity, initial_soc)
    queued = QueuedArrivalState(request, request.energy_capacity * initial_soc)
    uav = UAVState.from_request(0, request, queued, np.zeros(3), 0.0)
    uav.phase = FlightPhase.RING_INNER
    uav.ring_role = RingRole.INNER
    ring = env.airspace.actual_ring(RingRole.INNER)
    uav.ring_progress = 0.0
    uav.position = np.asarray((ring.radius, 0.0, ring.height), dtype=float)
    uav.speed = ring.nominal_speed
    uav.velocity = np.asarray((0.0, ring.nominal_speed, 0.0), dtype=float)
    env.ledger.request(request.arrival_id)
    env.active[uav.uid] = uav
    # Isolate the reserve branch while retaining the real environment step.
    env.runtime_traffic.protect = lambda _env, intents, _traces: intents
    _, rewards, terminated, _, info = env.step({"0": np.zeros(3, dtype=float)})
    assert env.ledger.outcomes[request.arrival_id] is TerminalOutcome.DIVERTED_LOW_SOC
    assert request.arrival_id not in env.ledger.unresolved
    assert uav.terminal_reason == "low_soc_before_touchdown"
    assert not env.active and not env.service_uavs
    assert terminated == {"0": True}
    assert info["reward_components"]["0"]["failure"] == pytest.approx(-c.low_level_energy_failure_penalty)
    assert rewards["0"] < 0.0
    with pytest.raises(ValueError, match="duplicate terminal outcome"):
        env._resolve(uav, TerminalOutcome.DIVERTED_LOW_SOC, "duplicate")


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
    env=TerminalParallelEnv(c); env.reset(seed=13); ledger=env.run_rule_only()
    assert len(ledger.outcomes)==10 and ledger.closes()
    allowed = {TerminalOutcome.COMPLETED, TerminalOutcome.DIVERTED_LOW_SOC}
    assert set(ledger.outcomes.values()) <= allowed
    assert not any(outcome in {
        TerminalOutcome.FAILED_ENERGY,
        TerminalOutcome.FAILED_COLLISION,
        TerminalOutcome.INVALID,
        TerminalOutcome.UNRESOLVED,
    } for outcome in ledger.outcomes.values())
    diverted_ids = {
        uid for uid, outcome in ledger.outcomes.items()
        if outcome is TerminalOutcome.DIVERTED_LOW_SOC
    }
    low_soc = [row for row in env.energy_outcomes
               if row["outcome"] == TerminalOutcome.DIVERTED_LOW_SOC.value]
    assert len(low_soc) == len(diverted_ids)
    assert {row["arrival_id"] for row in low_soc} == diverted_ids
    assert all(row["reason"] == "low_soc_before_touchdown" for row in low_soc)
    assert all(row["soc"] < c.min_landing_soc for row in low_soc)
    assert all(z.status is LZStatus.FREE for z in env.lz_manager.zones)
