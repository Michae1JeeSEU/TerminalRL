"""Runtime evidence for selected FT-RS-S1-1.0 contracts, not 94-case certification."""
from dataclasses import FrozenInstanceError, replace
import math
import pytest

from terminal_marl.core.rotating_slots.demand import ArrivalManifest, ArrivalRequest
from terminal_marl.core.rotating_slots.resources import (SlotCalendar, PadCalendar,
    ResourceConflict, ResourceDependencyError)
from terminal_marl.core.rotating_slots.trajectory import generate_transfer, generate_final, make_stable_plan, generate_inbound
from terminal_marl.core.rotating_slots.energy import integrate_energy
from terminal_marl.core.rotating_slots.world import RotatingSlotWorld, EmergencyDependencyConflict, SchedulerPort


def transfer(aircraft_id="a", t=0., version=0, target=0):
    c=generate_transfer(aircraft_id,"outer",0,target,t,source_version=version)
    assert c.status == "FEASIBLE", c
    return c.plan


def final(aircraft_id="a", pad=0, emergency=False, ring="inner", slot=0, t=0.):
    c=generate_final(aircraft_id,ring,slot,pad,t,emergency=emergency)
    assert c.status == "FEASIBLE", c
    return c.plan


def test_manifest_fixed_draws_truncation_and_future_isolation():
    m=ArrivalManifest.generate(11,180)
    assert m == ArrivalManifest.generate(11,180)
    assert all(.60 < r.initial_soc < .90 for r in m.requests)
    assert len({r.id for r in m.requests}) == len(m.requests)
    assert {r.gate_id for r in m.requests} == {0,1,2,3}
    w=RotatingSlotWorld(manifest=m)
    s=w.snapshot()
    assert not s.requests and not hasattr(s,"manifest")
    with pytest.raises(FrozenInstanceError):
        s.time=1
    with pytest.raises(TypeError):
        s.metrics["admitted"]=10


def test_half_open_calendar_concurrency_and_transaction():
    c=PadCalendar()
    c.reserve(0,"a",0,60)
    c.reserve(0,"b",60,120)
    c.reserve(1,"c",0,120)
    assert [r.aircraft_id for r in c.active(0,60)] == ["b"]
    before=c.reservations
    with pytest.raises(ResourceConflict):
        with c.transaction():
            c.reserve(2,"d",0,1)
            c.reserve(0,"e",59,61)
    assert c.reservations == before


def test_resource_dependency_is_explicit_and_cannot_disappear():
    c=SlotCalendar()
    c.reserve(("outer",0),"a",0,20,plan_hash="departure")
    c.reserve(("outer",0),"b",20,math.inf,committed=True,dependencies=("departure",))
    with pytest.raises(ResourceDependencyError):
        c.remove_plan("departure")
    assert len(c.reservations)==2


def test_exact_service_ground_soc_freeze_and_same_time_merge():
    requests=(ArrivalRequest("x",.35,0,.75,"short"), ArrivalRequest("y",.35,1,.75,"long"))
    w=RotatingSlotWorld(manifest=ArrivalManifest(requests))
    w.seed_service("service",pad_id=0,remaining=.35,soc=.48)
    w.advance(.6)
    assert w.aircraft["service"].status=="CLEARED"
    assert w.aircraft["service"].soc==.48
    assert w.decision_times == [0.,.35]
    kinds=[e["type"] for e in w.events if e["time"]==.35]
    assert kinds[:3] == ["pad_service_completed","request_released","request_released"]
    assert w.metrics()["cost_reroute"]==1200
    assert w.metrics()["conservation"]


def test_no_legacy_aircraft_capacity_and_physical_vs_reservation_counts():
    w=RotatingSlotWorld()
    for ring,n in (("outer",37),("middle",28),("inner",18)):
        for slot in range(n):
            w.seed_hold(f"{ring}_{slot}",ring,slot)
    assert w.metrics()["physical_airborne_count"]==83
    assert w.metrics()["slot_reservations_active"]==83
    assert w.metrics()["conservation"]


def test_transfer_executes_exact_checked_object_and_has_no_double_count():
    w=RotatingSlotWorld()
    w.seed_hold("a")
    p=transfer()
    assert w.submit((p,)).accepted
    assert w.active["a"] is p
    assert w.aircraft["a"].status=="TRANSFER_O_M"
    assert w.metrics()["physical_airborne_count"]==1
    assert w.cancel("a").reason=="COMMITTED_MODIFICATION"
    q=replace(p,target_slot=1,trajectory_hash="")
    assert w.submit((q,)).reason=="COMMITTED_MODIFICATION"


def test_current_plan_immutable_but_next_maneuver_can_be_soft():
    w=RotatingSlotWorld()
    w.seed_hold("a")
    p=transfer()
    assert w.submit((p,)).accepted
    c=generate_transfer("a","middle",0,0,p.end_time+1,source_version=1)
    assert c.status=="FEASIBLE",c
    assert w.submit((c.plan,)).accepted
    assert w.active["a"] is p and w.soft["a"] is c.plan
    assert w.cancel("a").accepted
    assert w.active["a"] is p


def test_stale_version_illegal_action_and_batch_rollback():
    w=RotatingSlotWorld()
    w.seed_hold("a")
    p=transfer()
    before=w.snapshot()
    assert w.submit((replace(p,source_version=1,trajectory_hash=""),)).reason=="STALE_STATE"
    stable=make_stable_plan("a","outer",0)
    assert w.submit((stable,)).reason=="INVALID_ACTION"
    assert w.submit((p,p)).reason=="DUPLICATE_AIRCRAFT"
    assert not w.active and not w.soft
    assert w.snapshot().aircraft==before.aircraft
    assert w.snapshot().slots==before.slots


@pytest.mark.parametrize("latency",[.2,1.2,5.2])
def test_latency_advances_energy_and_rejects_stale_start(latency):
    # Active atomic final avoids unrelated hold-reference searches in a latency test.
    w=RotatingSlotWorld()
    w.seed_hold("a","inner",0,.7)
    p=final()
    assert w.submit((p,)).accepted
    initial=w.aircraft["a"].soc
    result=w.submit((p,),simulated_latency=latency)
    assert result.reason=="EXPIRED_TIME"
    assert w.time==pytest.approx(latency)
    assert w.aircraft["a"].soc < initial
    assert w.active["a"] is p


def test_touchdown_reserve_complete_descent_and_service_clock():
    p=final()
    q=integrate_energy(p,p.start_time,p.end_time)
    w=RotatingSlotWorld()
    w.seed_hold("a","inner",0,.25+q.soc_used-.001)
    assert w.submit((p,)).reason=="LANDING_RESERVE"
    w=RotatingSlotWorld()
    w.seed_hold("a","inner",0,.25+q.soc_used+.00001,service_class="short")
    assert w.submit((p,)).accepted
    w.advance(p.end_time)
    a=w.aircraft["a"]
    assert a.status=="LANDED_SERVICE" and a.p[2]==pytest.approx(0.,abs=1e-8)
    assert a.soc==pytest.approx(.25001,abs=1e-8)
    assert a.touchdown_time==p.end_time and a.service_end==p.end_time+60
    w.advance(a.service_end)
    assert w.aircraft["a"].status=="CLEARED"
    assert w.aircraft["a"].soc==a.soc


def test_emergency_permission_and_actual_start_cost_once():
    w=RotatingSlotWorld()
    w.seed_hold("a","inner",0,.65)
    p=final(emergency=True)
    assert w.submit((p,)).reason=="NO_EMERGENCY_ELIGIBILITY"
    w=RotatingSlotWorld()
    w.seed_hold("a","inner",0,.35,emergency=True)
    assert w.submit((p,)).accepted
    assert w.metrics()["cost_emergency"]==1200
    w.advance(p.end_time)
    assert w.metrics()["cost_emergency"]==1200
    assert w.metrics()["emergency_landed"]==1
    assert w.aircraft["a"].emergency.declared


def test_busy_pad_emergency_cannot_preempt_service_or_delete_aircraft():
    w=RotatingSlotWorld()
    w.seed_service("service",0,remaining=200.)
    w.seed_hold("a","inner",0,.35,emergency=True)
    p=final(emergency=True)
    before=w.snapshot()
    assert w.submit((p,)).reason=="PAD_NOT_CURRENTLY_FREE"
    assert w.snapshot().aircraft==before.aircraft
    assert w.aircraft["service"].service_end==200.
    assert w.aircraft["a"].status=="HOLD_INNER"


def test_normal_currently_busy_but_future_released_pad_is_legal():
    w=RotatingSlotWorld()
    w.seed_service("service",0,remaining=30.)
    w.seed_hold("a","inner",0,.65)
    p=final()
    assert p.end_time-13.5 > 30
    assert w.submit((p,)).accepted
    assert len(w.pads.reservations)==2


def test_world_export_has_actual_plans_exact_events_and_no_future_snapshot():
    w=RotatingSlotWorld()
    w.seed_service("service",0,remaining=.125)
    w.advance(.2)
    record=w.to_dict()
    assert .125 in [s["time"] for s in record["states"]]
    assert record["trajectories"] and record["metrics"]["conservation"]
    assert record["sample_dt_s"]==.2


def test_admission_immediate_local_reference_without_global_certificate():
    request=ArrivalRequest("r",0.,0,.85,"short")
    w=RotatingSlotWorld(manifest=ArrivalManifest((request,)))
    candidate=generate_inbound("r",0,1,0)
    assert candidate.status=="FEASIBLE"
    outcomes=[]
    def action(snapshot,world):
        if "r" not in world.aircraft:
            outcomes.append(world.admit("r",candidate.plan))
    w.advance(0,action)
    assert outcomes[0].accepted
    assert w.aircraft["r"].status=="INBOUND"
    assert w.metrics()["admitted"]==1 and w.metrics()["rerouted"]==0
    assert w.metrics()["conservation"]


def test_admission_low_energy_does_not_resample_or_retry_gate():
    request=ArrivalRequest("r",0.,0,.10,"short")
    w=RotatingSlotWorld(manifest=ArrivalManifest((request,)))
    candidate=generate_inbound("r",0,1,0)
    outcomes=[]
    w.advance(0,lambda s,world: outcomes.append(world.admit("r",candidate.plan)))
    assert not outcomes[0].accepted
    assert w.aircraft["r"].status=="REROUTED" and w.aircraft["r"].soc==.10
    assert w.manifest.requests[0] == request
    assert w.metrics()["cost_reroute"]==600 and w.metrics()["conservation"]


def test_energy_failure_off_grid_stops_and_censors_without_deletion():
    future=ArrivalRequest("unreleased",10.,0,.75,"standard")
    w=RotatingSlotWorld(manifest=ArrivalManifest((future,)))
    for pad in range(6):
        w.seed_service(f"pad_{pad}",pad,remaining=200.)
    a=w.seed_hold("failing","outer",0,.00001)
    w.seed_hold("other","middle",0,.75)
    from terminal_marl.core.rotating_slots.energy import evaluate_power
    expected=.00001*85*252*3600/evaluate_power(a.v,a.a).battery_w
    w.advance(1.)
    assert w.failed and w.time==pytest.approx(expected,abs=.000001)
    assert 0 < w.time < .2
    assert w.aircraft["failing"].status=="ENERGY_FAILURE"
    assert w.aircraft["other"].status=="CENSORED_DUE_TO_FAILURE"
    assert all(f"pad_{pad}" in w.aircraft for pad in range(6))
    assert w.metrics()["unreleased"]==1 and w.metrics()["cost_failure"]==1000000
    assert w.metrics()["conservation"] and w.metrics()["truncated"]
    stopped=w.time
    w.advance(10.)
    assert w.time==stopped


def test_full_normal_lifecycle_through_both_rings_and_complete_service():
    from terminal_marl.core.rotating_slots.driver import DeterministicDriver
    w=RotatingSlotWorld(manifest=ArrivalManifest((ArrivalRequest("r",0,0,.85,"short"),)))
    w.advance(300,DeterministicDriver())
    assert w.aircraft["r"].status=="CLEARED"
    assert w.metrics()["normal_landed"]==1 and not w.failed
    assert [e["type"] for e in w.events if e["type"] in ("admitted","inbound_completed","transfer_completed","touchdown","pad_service_completed")] == [
        "admitted","inbound_completed","transfer_completed","transfer_completed","touchdown","pad_service_completed"]
    assert w.aircraft["r"].soc >= .25 and w.metrics()["conservation"]


def test_started_inbound_dependency_blocks_cancel_but_allows_equivalent_release():
    w=RotatingSlotWorld(manifest=ArrivalManifest((ArrivalRequest("r",0,0,.85,"short"),)))
    w.seed_hold("departing","outer",1)
    old=generate_transfer("departing","outer",1,1,.001).plan
    assert w.submit((old,)).accepted
    inbound=generate_inbound("r",0,1,0,durations=[55]).plan
    assert inbound is not None
    results=[]
    w.advance(0,lambda s,x:results.append(x.admit("r",inbound)))
    assert results[0].accepted
    assert w.cancel("departing").reason=="RESOURCE_DEPENDENCY"
    new=generate_transfer("departing","outer",1,1,0).plan
    assert w.submit((new,)).accepted
    assert old.trajectory_hash not in w.dependencies[inbound.trajectory_hash]
    assert new.trajectory_hash in w.dependencies[inbound.trajectory_hash]
    assert w.active["r"] is inbound and w.active["departing"] is new


def test_all_busy_then_exact_release_rescue_and_below_reserve_landing():
    w=RotatingSlotWorld()
    for pad in range(6):
        w.seed_service(f"pad_{pad}",pad,remaining=.35)
    w.seed_hold("rescue","inner",0,.30)
    w.advance(250)
    a=w.aircraft["rescue"]
    assert a.emergency.declared and a.emergency.start_time==pytest.approx(.35)
    assert a.touchdown_time is not None and 0 < a.soc < .25
    assert w.metrics()["emergency_landed"]==1 and w.metrics()["landing_below_reserve"]==1
    assert w.metrics()["cost_emergency"]==1200 and not w.failed


def test_deadline_proof_includes_complete_descent_energy_before_hold_exhaustion():
    w=RotatingSlotWorld()
    for pad in range(6):
        w.seed_service(f"pad_{pad}",pad,remaining=2.)
    w.seed_hold("rescue","inner",0,.023)
    w.advance(0)
    a=w.aircraft["rescue"]
    assert a.emergency.deadline_missed and a.status=="HOLD_INNER" and not w.failed
    event=next(e for e in w.events if e["type"]=="PAD_DEADLINE_MISSED")
    d=event["details"]
    assert d["wait_soc"] < a.soc
    assert d["wait_soc"]+d["complete_rescue_energy_lower_bound_soc"] >= a.soc


def test_simultaneous_emergencies_one_pad_and_irreversible_declaration():
    w=RotatingSlotWorld()
    for pad in range(5):
        w.seed_service(f"pad_{pad}",pad,remaining=100.)
    w.seed_hold("a","inner",0,.30)
    w.seed_hold("b","inner",9,.30)
    w.advance(0)
    assert sum(p.kind=="FINAL_EMERGENCY" for p in w.active.values())==1
    waiting=next(a for a in w.aircraft.values() if a.status=="HOLD_INNER")
    assert waiting.emergency.declared
    ordinary=generate_final(waiting.id,"inner",waiting.slot,5,0).plan
    assert ordinary is not None
    assert w.submit((ordinary,)).reason=="EMERGENCY_DECLARED"
    assert w.metrics()["cost_emergency"]==1200


def test_six_pads_can_jointly_accept_compatible_normal_finals():
    w=RotatingSlotWorld()
    plans=[]
    for pad in range(6):
        name=f"a{pad}"
        w.seed_hold(name,"inner",pad*3,.8)
        plans.append(final(name,pad=pad,slot=pad*3))
    result=w.submit(plans)
    assert result.accepted,result
    assert len(w.active)==6 and len(w.pads.reservations)==6
    assert {p.pad_id for p in w.active.values()} == set(range(6))
    assert w.metrics()["physical_airborne_count"]==6


def test_inner_alert_uses_immediate_legal_normal_final_before_declaration():
    w=RotatingSlotWorld()
    for pad in range(6):
        w.seed_service(f"pad_{pad}",pad,remaining=.35)
    w.seed_hold("normal_priority","inner",0,.35)
    w.advance(0)
    assert w.active["normal_priority"].kind=="FINAL_NORMAL"
    assert not w.aircraft["normal_priority"].emergency.declared
    assert not w.aircraft["normal_priority"].emergency.eligible
    assert not w.aircraft["normal_priority"].emergency.pending
    assert w.aircraft["normal_priority"].emergency.state=="NONE"
    assert w.metrics()["cost_emergency"]==0


def test_emergency_at_2p2_precedes_due_ordinary_transfer_and_callback():
    from terminal_marl.core.rotating_slots.driver import DeterministicDriver
    from terminal_marl.core.rotating_slots.energy import evaluate_power
    state=make_stable_plan("trigger","middle",0).state(0)
    rate=evaluate_power(state.v,state.a).battery_w/(85*252*3600)
    w=RotatingSlotWorld()
    w.seed_hold("trigger","middle",0,.35+2.1*rate)
    ordinary=generate_transfer("trigger","middle",0,0,2.2).plan
    assert ordinary is not None and w.submit((ordinary,)).accepted
    observations=[]
    driver=DeterministicDriver()
    def action(snapshot,world):
        observations.append((snapshot.time,snapshot.aircraft[0].emergency.declared,
                             snapshot.aircraft[0].status))
        driver(snapshot,world)
    w.advance(2.2,action)
    a=w.aircraft["trigger"]
    assert a.emergency.declared and a.emergency.start_time==pytest.approx(2.2)
    assert w.active["trigger"].kind=="FINAL_EMERGENCY"
    assert w.active["trigger"].source_ring=="middle"
    assert [t for t in w.decision_times if t==pytest.approx(2.2)]==[pytest.approx(2.2)]
    assert observations[-1][1:] == (True,"FINAL_EMERGENCY")
    assert not any(e["type"]=="maneuver_started" and e["details"].get("trajectory_hash")==ordinary.trajectory_hash for e in w.events)
    assert any(e["type"]=="soft_plan_cancelled" and e["details"].get("trajectory_hash")==ordinary.trajectory_hash for e in w.events)


def test_submit_does_not_start_another_due_soft_plan_during_callback():
    w=RotatingSlotWorld()
    w.seed_hold("a","outer",0,.85)
    w.seed_hold("b","outer",18,.85)
    pb=generate_transfer("b","outer",18,14,1.).plan
    pa=generate_transfer("a","outer",0,0,1.).plan
    assert pa is not None and pb is not None
    assert w.submit((pb,)).accepted
    observations=[]
    def action(snapshot,world):
        if abs(snapshot.time-1.)<1e-8:
            assert "b" not in world.active
            result=world.submit((pa,))
            observations.append((result.accepted,"b" in world.active,"b" in world.soft))
    w.advance(1.,action)
    assert observations==[(True,False,True)]
    assert w.active["a"] is pa and w.active["b"] is pb


def test_active_transfer_pending_is_unchanged_until_target_ring_rescue():
    from terminal_marl.core.rotating_slots.driver import DeterministicDriver
    w=RotatingSlotWorld()
    w.seed_hold("a","outer",0,.35)
    p=transfer()
    assert w.submit((p,)).accepted
    w.advance(0,DeterministicDriver())
    assert w.active["a"] is p and w.aircraft["a"].emergency.pending
    assert not w.aircraft["a"].emergency.declared
    next_normal=generate_transfer("a","middle",0,0,p.end_time+1,source_version=1).plan
    assert next_normal is not None
    assert w.submit((next_normal,)).reason=="EMERGENCY_DISPOSITION_REQUIRED"
    w.advance(p.end_time,DeterministicDriver())
    a=w.aircraft["a"]
    assert a.emergency.declared and not a.emergency.pending
    assert a.emergency.start_time==pytest.approx(p.end_time)
    assert a.emergency.start_ring=="middle" and w.active["a"].source_ring=="middle"
    assert w.metrics()["cost_emergency"]==2400


def test_emergency_protected_departure_conflict_stops_without_deleting_dependency():
    # Named injected emergency fixture: establish the already-started inbound
    # dependency first, then inject low SOC before the first dispatch cycle.
    w=RotatingSlotWorld(manifest=ArrivalManifest((ArrivalRequest("r",0,0,.85,"short"),)))
    for pad in range(6):
        w.seed_service(f"pad_{pad}",pad,remaining=200.)
    w.seed_hold("departing","outer",1,.85)
    old=generate_transfer("departing","outer",1,1,.001).plan
    assert w.submit((old,)).accepted
    w._settle()
    inbound=generate_inbound("r",0,1,0,durations=[55]).plan
    assert inbound is not None and w.admit("r",inbound).accepted
    w.aircraft["departing"]=replace(w.aircraft["departing"],soc=.30)
    calendar=w.slots.reservations
    with pytest.raises(EmergencyDependencyConflict) as caught:
        w.advance(1.)
    assert caught.value.reason=="RESOURCE_DEPENDENCY"
    assert w.time==0 and not w.failed
    assert w.soft["departing"] is old and w.active["r"] is inbound
    assert w.slots.reservations==calendar
    assert old.trajectory_hash in w.dependencies[inbound.trajectory_hash]
    assert w.aircraft["departing"].status=="HOLD_OUTER"
    assert any(e["type"]=="emergency_dependency_conflict" for e in w.events)
    assert w.metrics()["cost_failure"]==0 and w.metrics()["rerouted"]==0


def test_optional_state_sink_preserves_full_frames_with_only_latest_in_world():
    class CollectingSink:
        def __init__(self):
            self.frames=[]
            self.flush_count=0
        def __call__(self,frame):
            if self.frames and self.frames[-1]["time"]==frame["time"]:
                self.frames[-1]=frame
            else:
                self.frames.append(frame)
        def flush(self):
            self.flush_count+=1
    sink=CollectingSink()
    manifest=ArrivalManifest((ArrivalRequest("r",.35,0,.75,"short"),))
    reference=RotatingSlotWorld(manifest=manifest)
    streamed=RotatingSlotWorld(manifest=manifest,state_sink=sink)
    for w in (reference,streamed):
        w.seed_service("service",0,remaining=.35,soc=.48)
        w.advance(.2)
        w.to_dict()  # Midrun export must not freeze the timestamp or clock.
        w.advance(.6)
    record=streamed.to_dict()
    expected=reference.to_dict()
    assert sink.frames==expected["states"]
    assert len(streamed.records)==1
    assert streamed.records[0]==expected["states"][-1]
    assert record["states_streamed"] and record["states_count"]==len(sink.frames)
    assert not record["states_complete"]  # Generic callback has no readable archive path.
    assert sink.flush_count==2
    assert record["events"]==expected["events"] and record["metrics"]==expected["metrics"]


def test_state_sink_requires_callable_without_changing_default_recording():
    with pytest.raises(TypeError,match="state_sink must be callable"):
        RotatingSlotWorld(state_sink="states.jsonl")
    w=RotatingSlotWorld()
    w.advance(.6)
    assert len(w.records)==4
    assert "states_streamed" not in w.to_dict()


def test_reference_canonical_templates_ignore_first_phase_and_warm_order():
    first,second=RotatingSlotWorld(),RotatingSlotWorld()
    for t in (.1,6.):
        assert first._reference("middle",0,t)[0] is not None
    for t in (6.,.1):
        assert second._reference("middle",0,t)[0] is not None
    before=dict(first._reference_templates)
    q1,p1=first._reference("middle",0,3.)
    q2,p2=second._reference("middle",0,3.)
    assert q1==q2 and p1.to_dict()==p2.to_dict()
    assert first._reference_templates==second._reference_templates==before
    assert first.events==second.events==[]  # Reference lookup is numerically read-only.
    key=("middle",0,first.config.profile_hash)
    center_time=(.5*2*math.pi/24)/first.config.airspace.common_angular_speed_radps
    canonical=generate_final("canonical_reference","middle",0,0,center_time,emergency=True,config=first.config)
    assert canonical.status=="FEASIBLE"
    assert first._reference_templates[key]==canonical.plan.segments[0].duration


def test_cache_modes_preserve_moving_phase_trigger_events_and_execution():
    from terminal_marl.core.rotating_slots.energy import evaluate_power
    state=make_stable_plan("a","middle",0).state(0)
    rate=evaluate_power(state.v,state.a).battery_w/(85*252*3600)
    worlds=[]
    for enabled in (True,False):
        w=RotatingSlotWorld(cache_enabled=enabled)
        w.seed_hold("a","middle",0,.35+.3*rate)
        w.advance(.6)
        worlds.append(w)
    cached,uncached=worlds
    assert cached.events==uncached.events
    assert cached.records==uncached.records
    assert cached.trajectories==uncached.trajectories
    assert cached.aircraft["a"].emergency.start_time==pytest.approx(.4)
    assert cached.aircraft["a"].emergency.declared
    assert cached.metrics()==uncached.metrics()
    assert len(uncached._q_cache)==len(uncached._energy_cache)==0
    assert cached._reference_templates==uncached._reference_templates


def test_exact_result_caches_are_bounded_and_eviction_preserves_reference():
    w=RotatingSlotWorld()
    q,p=w._reference("middle",0,3.)
    stable=make_stable_plan("stable","middle",0)
    energy=w._energy(stable,0,.2)
    for i in range(w.Q_CACHE_CAPACITY+1):
        w._store_cache(w._q_cache,("eviction_fixture",i),(q,p),w.Q_CACHE_CAPACITY)
    for i in range(w.ENERGY_CACHE_CAPACITY+1):
        w._store_cache(w._energy_cache,("eviction_fixture",i),energy,w.ENERGY_CACHE_CAPACITY)
    assert len(w._q_cache)==w.Q_CACHE_CAPACITY
    assert len(w._energy_cache)==w.ENERGY_CACHE_CAPACITY
    q2,p2=w._reference("middle",0,3.)
    energy2=w._energy(stable,0,.2)
    assert q2==q and p2.to_dict()==p.to_dict() and energy2==energy
    metadata=w.to_dict()["cache"]
    assert metadata["q_size"]<=metadata["q_capacity"]==512
    assert metadata["energy_size"]<=metadata["energy_capacity"]==4096
    assert metadata["canonical_template_count"]<=metadata["canonical_template_capacity"]==72


def test_scheduler_callback_port_hides_future_audit_and_freezes_current_mappings():
    manifest=ArrivalManifest((ArrivalRequest("visible",0,0,.75,"short"),
                              ArrivalRequest("future",10,2,.8,"long")))
    w=RotatingSlotWorld(manifest=manifest)
    w.seed_service("service",0,remaining=60.)
    seen=[]
    def action(snapshot,port):
        assert isinstance(port,SchedulerPort) and port is not w
        assert port.time==snapshot.time==0
        assert set(port.requests)=={"visible"}
        assert [r.id for r in snapshot.requests]==["visible"]
        for name in ("manifest","events","records","to_dict","_record","_world","world"):
            assert not hasattr(port,name)
        for metrics in (snapshot.metrics,port.metrics(),port.snapshot().metrics):
            assert "planned_manifest" not in metrics and "unreleased" not in metrics
            with pytest.raises(TypeError):
                metrics["offered"]=123
        for mapping in (port.aircraft,port.active,port.soft,port.requests):
            with pytest.raises(TypeError):
                mapping["injected"]=None
        with pytest.raises(FrozenInstanceError):
            port.aircraft["service"].soc=.9
        with pytest.raises(AttributeError):
            port.time=1
        with pytest.raises(AttributeError):
            port.config.airspace.common_angular_speed_radps=.1
        assert port.reject("fixture_rejection","visible").reason=="fixture_rejection"
        assert port.reroute("visible").accepted
        seen.append(port.metrics()["rerouted"])
    w.advance(0,action)
    assert seen==[1]
    assert w.metrics()["planned_manifest"]==2 and w.metrics()["unreleased"]==1
    audit=w.to_dict()
    assert audit["states"][-1]["metrics"]["unreleased"]==1
    assert len(audit["manifest"]["requests"])==2


def test_scheduler_observation_does_not_disclose_future_manifest_cardinality():
    visible=ArrivalRequest("visible",0,0,.75,"short")
    future=(ArrivalRequest("f1",10,1,.8,"long"),ArrivalRequest("f2",20,2,.7,"standard"))
    observations=[]
    for requests in ((visible,), (visible,)+future):
        w=RotatingSlotWorld(manifest=ArrivalManifest(requests))
        current=[]
        def action(snapshot,port):
            current.append((snapshot.to_dict(),dict(port.metrics()),tuple(port.requests)))
            port.reroute("visible")
        w.advance(0,action)
        observations.append(current)
    assert observations[0]==observations[1]


def test_public_admission_does_not_bind_predictably_emergency_future_release():
    from terminal_marl.core.rotating_slots.trajectory import make_stable_plan
    w=RotatingSlotWorld(manifest=ArrivalManifest((ArrivalRequest("r",0,0,.85,"short"),)))
    q,_=w._reference("outer",1,0.)
    hold=make_stable_plan("reference_hold","outer",1,start_time=0.,end_time=10.)
    qhold=integrate_energy(hold,0.,10.).soc_used
    for pad in range(6):
        w.seed_service(f"busy_{pad}",pad,remaining=200.)
    w.seed_hold("departing","outer",1,max(.35,q+qhold+.25)+1e-8)
    departure=generate_transfer("departing","outer",1,1,.001).plan
    inbound=generate_inbound("r",0,1,0,durations=[55.]).plan
    assert departure is not None and inbound is not None
    observed=[]
    def action(snapshot,port):
        if snapshot.time != 0.: return
        assert not port.aircraft["departing"].emergency.eligible
        assert port.submit((departure,)).accepted
        slots=tuple(w.slots.reservations)
        result=port.admit("r",inbound)
        observed.append(result)
        assert "r" not in w.active and not w.dependencies
        assert w.slots.reservations==slots
        assert w.soft["departing"] is departure
    w.advance(.002,action)
    assert len(observed)==1 and observed[0].reason=="FUTURE_EMERGENCY_DISPOSITION"
    details=observed[0].details
    assert details["supporting_aircraft_id"]=="departing"
    assert details["witness_time"]==.001 and details["witness"]=="scheduled_start"
    assert details["predicted_soc"]-details["q_e"]-details["q_hold"] <= .25
    assert w.time==pytest.approx(.002) and not w.failed
    assert w.aircraft["r"].status=="REROUTED"
    assert w.aircraft["departing"].emergency.declared
    assert "departing" not in w.soft
    assert not any(e["type"]=="emergency_dependency_conflict" for e in w.events)
    assert w.metrics()["conservation"]


def test_starting_transfer_cannot_bind_future_alert_inner_normal_final_atomically():
    w=RotatingSlotWorld()
    w.seed_hold("landing","inner",0,.35)
    w.seed_hold("incoming","middle",1,.85)
    future=generate_final("landing","inner",0,5,.001,durations=[35.]).plan
    incoming=generate_transfer("incoming","middle",1,0,0.,durations=[40.]).plan
    assert future is not None and incoming is not None
    assert w.submit((future,)).accepted
    before=w.snapshot()
    slots,pads=w.slots.reservations,w.pads.reservations
    dependencies=dict(w.dependencies)
    result=w.submit((incoming,))
    assert result.reason=="FUTURE_EMERGENCY_DISPOSITION",result
    assert result.details["supporting_aircraft_id"]=="landing"
    assert w.snapshot().aircraft==before.aircraft
    assert w.slots.reservations==slots and w.pads.reservations==pads
    assert w.dependencies==dependencies and w.soft=={"landing":future} and not w.active


def test_binding_checks_downstream_future_release_in_chain():
    w=RotatingSlotWorld(manifest=ArrivalManifest((ArrivalRequest("r",0,0,.85,"short"),)))
    q,_=w._reference("inner",0,0.)
    hold=make_stable_plan("reference_hold","inner",0,start_time=0.,end_time=10.)
    qhold=integrate_energy(hold,0.,10.).soc_used
    w.seed_hold("inner","inner",0,max(.35,q+qhold+.25)+1e-8)
    w.seed_hold("middle","middle",1,.85)
    w.seed_hold("outer","outer",1,.85)
    inner=generate_final("inner","inner",0,5,.003,durations=[35.]).plan
    middle=generate_transfer("middle","middle",1,0,.001,durations=[40.]).plan
    outer=generate_transfer("outer","outer",1,1,.002,durations=[40.]).plan
    inbound=generate_inbound("r",0,1,0.,durations=[55.]).plan
    assert all(p is not None for p in (inner,middle,outer,inbound))
    assert w.submit((inner,middle,outer)).accepted
    results=[]
    def action(snapshot,port):
        if snapshot.time==0.:
            assert not port.aircraft["inner"].emergency.eligible
            results.append(port.admit("r",inbound))
    w.advance(0.,action)
    assert results[0].reason=="FUTURE_EMERGENCY_DISPOSITION"
    assert results[0].details["supporting_aircraft_id"]=="inner"
    assert "r" not in w.active and not w.dependencies


def test_protected_future_release_replacement_checks_predicted_emergency_atomically():
    w=RotatingSlotWorld(manifest=ArrivalManifest((ArrivalRequest("r",0,0,.85,"short"),)))
    q,_=w._reference("outer",1,0.)
    hold=make_stable_plan("reference_hold","outer",1,start_time=0.,end_time=10.)
    qhold=integrate_energy(hold,0.,10.).soc_used
    w.seed_hold("departing","outer",1,max(.35,q+qhold+.25)+1e-8)
    old=generate_transfer("departing","outer",1,1,.000001).plan
    candidate=generate_transfer("departing","outer",1,1,.001,durations=[39.9])
    if candidate.status != "FEASIBLE":
        candidate=generate_transfer("departing","outer",1,1,.001,durations=[39.5])
    replacement=candidate.plan
    inbound=generate_inbound("r",0,1,0.,durations=[55.]).plan
    assert old is not None and candidate.status=="FEASIBLE" and inbound is not None
    assert w._protect_times(replacement)[0] <= w._protect_times(old)[0]
    assert w.submit((old,)).accepted
    results=[]
    w.advance(0.,lambda snapshot,port:results.append(port.admit("r",inbound)))
    assert results[0].accepted
    slots,pads=w.slots.reservations,w.pads.reservations
    dependencies=dict(w.dependencies)
    result=w.submit((replacement,))
    assert result.reason=="FUTURE_EMERGENCY_DISPOSITION",result
    assert w.slots.reservations==slots and w.pads.reservations==pads
    assert w.dependencies==dependencies and w.soft["departing"] is old
    assert w.active["r"] is inbound


def test_future_support_checks_current_active_target_before_next_source():
    w=RotatingSlotWorld()
    current=transfer()
    q=integrate_energy(current,current.start_time,current.end_time).soc_used
    w.seed_hold("a","outer",0,.35+q-1e-6)
    assert w.submit((current,)).accepted
    next_plan=generate_transfer("a","middle",0,0,current.end_time+1,source_version=1).plan
    assert next_plan is not None and w.submit((next_plan,)).accepted
    witness=w._future_support_witness(next_plan)
    assert witness["witness"]=="current_target"
    assert witness["witness_time"]==current.end_time
    assert witness["predicted_soc"]==pytest.approx(.35-1e-6)
    assert w.active["a"] is current and w.soft["a"] is next_plan
