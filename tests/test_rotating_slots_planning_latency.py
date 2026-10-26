"""Finite real-world checks of future-time normal planning and atomic delivery."""
from dataclasses import replace
import json
import math
from types import SimpleNamespace

import pytest

from terminal_marl.core.rotating_slots.config import thaw
from terminal_marl.core.rotating_slots.demand import ArrivalManifest, ArrivalRequest
from terminal_marl.core.rotating_slots.energy import integrate_energy
from terminal_marl.core.rotating_slots.planning_latency import (
    EFFECTIVE_STATE_VERSION, LatencyAwareScheduler, PlanningTiming, ProjectionRejected,
    project_visible_snapshot, prepare_submission, publish_submission, guard_submission,
    mark_submission_ready)
from terminal_marl.core.rotating_slots.stage2_campaign import (
    checkpoint_identity, digest, restore_checkpoint, save_checkpoint)
from terminal_marl.core.rotating_slots.stage2_planning import EvaluationContext, NormalSchedulerCallback
from terminal_marl.core.rotating_slots.trajectory import generate_final, generate_transfer
from terminal_marl.core.rotating_slots.world import (
    RotatingSlotWorld, STAGE2_POLICY_ID, evaluate_planning_snapshot)


class Durations:
    """Inject work durations without replacing real certification or execution."""
    def __init__(self, *durations):
        self.durations = iter(durations)
        self.value = 0.
        self.open = False

    def __call__(self):
        if self.open: self.value += next(self.durations,0.)
        self.open = not self.open
        return self.value


class CertifiedFinals:
    def __init__(self, pairs=(("a",0),)):
        self.pairs, self.calls, self.plans, self.acks = pairs, [], (), []

    def __call__(self, port):
        s = port.build_planning_snapshot()
        self.calls.append(s)
        plans = []
        for aid, pad in self.pairs:
            a = s.aircraft[aid]
            candidate = generate_final(aid,a.ring,a.slot,pad,s.time,
                                       source_version=a.version,config=s.config)
            assert candidate.plan is not None, candidate
            plans.append(candidate.plan)
        self.plans = tuple(plans)
        evaluated = port.evaluate_submission(self.plans)
        if evaluated.accepted and not evaluated.refinement_required:
            deferred = port.commit_validated_prefix(evaluated,snapshot_version=s.version)
            assert not deferred.accepted and deferred.status=="DEFERRED"

    def record_latency_ack(self, timing, record, plans):
        self.acks.append(dict(record))


def world():
    return RotatingSlotWorld(experiment_policy_id=STAGE2_POLICY_ID)


def assert_same_state(a, b):
    for key in ("p","v","a"):
        assert getattr(a,key)==pytest.approx(getattr(b,key),abs=1e-6)


def test_ten_second_work_keeps_old_hold_until_atomic_future_start(tmp_path):
    w = world()
    initial = w.seed_hold("a","inner",0,.9)
    old = w._base(initial)
    callback = CertifiedFinals()
    wrapper = LatencyAwareScheduler(callback,clock=Durations(10.,2.))
    w.decision_callback = wrapper
    w.advance(29.5)
    assert callback.calls[0].time==30. and callback.calls[0].planning_timing==PlanningTiming(0.,30.,29.)
    assert callback.calls[0].effective_state_version==EFFECTIVE_STATE_VERSION
    assert len(callback.calls)==1 and wrapper.dirty
    assert wrapper.records[0]["status"]=="PREPARED"
    assert not w.active and not w.soft and not w.pads.reservations
    assert_same_state(w.aircraft["a"],old.state(w.time))
    assert w.aircraft["a"].soc < initial.soc
    w.advance(30.01)
    assert wrapper.records[0]["status"]=="COMMITTED",wrapper.records
    plan = callback.plans[0]
    assert w.active["a"] is plan
    assert_same_state(old.state(30.),plan.state(30.))
    assert_same_state(w.aircraft["a"],plan.state(w.time))
    assert [e["time"] for e in w.events if e["type"]=="maneuver_started"]==[30.]
    assert callback.acks[0]["submitted"]==1
    assert wrapper.records[0]["normal_work_s"]==10.
    assert wrapper.records[0]["prepare_work_s"]==2.
    assert wrapper.records[0]["preparation_started_time"]>=wrapper.records[0]["result_time"]
    assert wrapper.records[0]["online_service_s"]>0.
    assert any(r["kind"]=="monitor" for r in wrapper.online_services)
    path = tmp_path/"timeline.json"
    path.write_text(json.dumps(dict(records=wrapper.records,events=w.events,
        soc=dict(initial=initial.soc,final=w.aircraft["a"].soc)),allow_nan=False))
    assert json.loads(path.read_text())["records"][0]["submitted"]==1


def test_common_emergency_response_during_work_rejects_whole_normal_prefix():
    w = world()
    w.seed_hold("a","inner",0,.9)
    w.seed_hold("e","inner",9,.9)
    callback = CertifiedFinals()
    wrapper = LatencyAwareScheduler(callback,clock=Durations(10.,1.))
    w.decision_callback = wrapper
    injected = []
    def emergency_at_5(snapshot, port):
        if snapshot.time>=5. and not injected:
            a = w.aircraft["e"]
            # Fixture injects the eligibility event only. The unmodified common
            # emergency handler selects/certifies/commits the actual trajectory.
            w.aircraft[a.id] = replace(a,emergency=replace(a.emergency,eligible=True,
                declared=True,declaration_time=w.time,declaration_ring=a.ring))
            injected.append(w.time)
    w.admission_callback = emergency_at_5
    w.advance(30.01)
    assert injected==[5.] and w.active["e"].kind=="FINAL_EMERGENCY"
    assert w.active["e"].pad_id==0
    assert "a" not in w.active and "a" not in w.soft
    assert wrapper.records[0]["status"]=="REJECTED"
    assert wrapper.records[0]["reason"] in ("RESOURCE_CONFLICT","TRAFFIC_COMMITMENT_CHANGED","DEPENDENCY_CHANGED")
    assert wrapper.records[0]["submitted"]==0
    assert all(r.aircraft_id!="a" for r in w.pads.reservations)
    assert any(e["type"]=="maneuver_started" and e["aircraft_id"]=="e" for e in w.events)


def test_timeout_and_soc_shortfall_never_execute_expired_prefix():
    w = world(); w.seed_hold("a","inner",0,.9)
    callback = CertifiedFinals()
    wrapper = LatencyAwareScheduler(callback,clock=Durations(31.))
    w.decision_callback = wrapper
    w.advance(30.)
    assert wrapper.pending is not None and wrapper.records[0]["status"]=="COMPUTING"
    assert len(callback.calls)==1 and callback.calls[0].planning_timing.snapshot_time==0.
    assert not w.active and not w.soft and not w.pads.reservations
    w.advance(31.1)
    assert wrapper.records[0]["reason"]=="READY_DEADLINE_MISSED"
    assert wrapper.records[0]["finished_time"]>=31.
    assert wrapper.pending is None and len(callback.calls)==1
    assert not w.active and not w.soft and not w.pads.reservations
    assert not any(e["type"]=="maneuver_started" for e in w.events)
    w.advance(35.1)
    assert len(callback.calls)==2
    assert callback.calls[1].planning_timing.snapshot_time==35.
    assert callback.calls[1].planning_timing.snapshot_time>wrapper.records[0]["finished_time"]
    low = world(); low.seed_hold("a","inner",0,.251)
    timing = PlanningTiming(0.,30.,29.)
    plan = generate_final("a","inner",0,0,30.,config=low.config).plan
    result, token = prepare_submission(low,(plan,),timing)
    assert not result.accepted and token is None
    assert result.reason in ("LANDING_RESERVE","SOC_INSUFFICIENT","PREPARED_ENERGY_BOUND")
    assert not low.active and not low.soft and not low.pads.reservations


def test_projection_uses_only_revealed_inputs_and_known_inflight_endpoints():
    manifest = ArrivalManifest((ArrivalRequest("later",5.,0,.8,"standard"),))
    w = RotatingSlotWorld(manifest=manifest)
    w.seed_hold("a","outer",0,.9)
    p = generate_transfer("a","outer",0,0,0.,config=w.config).plan
    assert w.submit((p,)).accepted
    before = (w.time,dict(w.aircraft),w.slots.reservations,dict(w.trajectories),thaw(w._energy_state()))
    s = w.build_planning_snapshot()
    during = project_visible_snapshot(s,10.)
    assert during.active["a"] is p and during.aircraft["a"].version==0
    after = project_visible_snapshot(s,p.end_time+1.)
    assert not after.active and after.aircraft["a"].version==1
    assert after.aircraft["a"].status=="HOLD_MIDDLE"
    assert after.aircraft["a"].soc < s.aircraft["a"].soc
    assert not during.requests and not after.requests
    assert_same_state(during.aircraft["a"],p.state(10.))
    assert before==(w.time,dict(w.aircraft),w.slots.reservations,dict(w.trajectories),thaw(w._energy_state()))
    with pytest.raises(ProjectionRejected): project_visible_snapshot(s,-1.)


def test_multi_aircraft_one_energy_basis_changed_never_partially_applies():
    w = world()
    for aid,slot in (("a",0),("b",3)): w.seed_hold(aid,"inner",slot,.9)
    callback = CertifiedFinals((("a",0),("b",1)))
    wrapper = LatencyAwareScheduler(callback,clock=Durations(10.,1.))
    w.decision_callback = wrapper
    w.advance(29.)
    assert wrapper.records[0]["status"]=="PREPARED"
    w.aircraft["b"] = replace(w.aircraft["b"],soc=w.aircraft["b"].soc-.01)
    w.advance(30.01)
    assert wrapper.records[0]["reason"]=="ENERGY_BASIS_CHANGED"
    assert not w.active and not w.soft and not w.pads.reservations
    assert not any(e["type"]=="maneuver_started" for e in w.events)


def test_prepared_guard_allows_natural_completion_calendar_rebuild_and_estimate_refresh():
    w = world(); w.seed_hold("a","outer",0,.9)
    w.seed_hold("b","outer",18,.9)
    w.seed_service("historical",pad_id=5,remaining=.5)
    current = generate_transfer("a","outer",0,0,0.,config=w.config).plan
    assert w.submit((current,)).accepted
    effective = current.end_time+2.
    existing = generate_transfer("b","outer",18,14,effective,config=w.config).plan
    assert w.submit((existing,)).accepted
    w.advance(1.,False)
    timing = PlanningTiming(1.,effective,effective-1.)
    plan = generate_transfer("a","middle",0,0,effective,source_version=1,config=w.config).plan
    result, token = prepare_submission(w,(plan,),timing)
    assert result.accepted,result
    _, unready = prepare_submission(w,(plan,),timing)
    assert mark_submission_ready(w,token).reason=="PREPARATION_NOT_COMPLETE"
    w.advance(2.,False)
    assert mark_submission_ready(w,token).accepted
    # An unrelated already-cleared audit record is not a traffic dependency.
    old = w.aircraft["historical"]
    w.aircraft[old.id] = replace(old,energy_j=old.energy_j+1.)
    w.advance(effective-.5,False)
    # Rebuilding calendars after known completion changes target->occupied.
    slots,pads,clears = w._calendar_stage(w.soft,())
    w.slots,w.pads,w._source_clear = slots,pads,clears
    a = w.aircraft["a"]
    w.aircraft[a.id] = replace(a,emergency=replace(a.emergency,q_e=.01,normal_margin=.1))
    observed=[]
    class BoundaryDelivery:
        next_event_time=effective
        def __call__(self,*args): pass
        def online_service(self,*args): pass
        def planning_event(self,port):
            assert port.active["b"] is existing
            assert publish_submission(w,unready).reason=="NO_TIMELY_READY_RECEIPT"
            guarded=guard_submission(w,token)
            assert guarded.accepted,guarded
            observed.append(publish_submission(w,token))
            self.next_event_time=math.inf
    w.advance(effective+.01,BoundaryDelivery())
    assert observed[0].accepted,observed
    assert w.active["a"] is plan
    assert not publish_submission(w,token).accepted
    assert not publish_submission(w,"invented").accepted


def test_explicit_effective_prefix_and_legacy_window_are_distinct():
    w = world(); w.seed_hold("a","inner",0,.9)
    legacy = w.build_planning_snapshot()
    future = project_visible_snapshot(legacy,30.,timing=PlanningTiming(0.,30.,29.))
    def column(t):
        return SimpleNamespace(kind="MOVE",plans=(SimpleNamespace(start_time=t,aircraft_id="a"),))
    assert len(EvaluationContext(legacy).prefix((column(4.),)))==1
    assert len(EvaluationContext(future).prefix((column(30.),column(31.))))==1
    assert not EvaluationContext(future).prefix((column(31.),))


def test_priority_service_charges_once_and_delays_readiness_without_retiming_effective():
    wrapper = LatencyAwareScheduler(CertifiedFinals(),clock=Durations(10.,2.))
    w = world(); w.seed_hold("a","inner",0,.9)
    wrapper(w._scheduler_port)
    initial_due = wrapper.pending["due"]
    wrapper.online_service(1.,3.,2.,"emergency")
    assert wrapper.pending["due"]==initial_due+3.
    assert wrapper.pending["timing"].effective_time==30.
    assert wrapper.records[0]["submitted"]==0
    # Completion handoff is charged once to preparation; no self-rescheduling loop.
    wrapper.online_service(wrapper.pending["due"],1.,.5,"admission")
    assert wrapper.pending["due"]==initial_due+3.
    assert wrapper._handoff_service==1.
    assert len(wrapper.online_services)==2
    wrapper.pending.update(stage="READY_SERVICE",due=28.5)
    wrapper.online_service(20.,1.,.5,"emergency")
    assert wrapper.pending["due"]==29.5
    assert wrapper.next_event_time==29.5
    port=SimpleNamespace(time=29.,discard_latency_submission=lambda token:None,
        planning_latency_event=lambda *args,**kwargs:None)
    wrapper.planning_event(port)
    assert wrapper.pending is not None
    saved=dict(wrapper.pending)
    port.time=29.5
    wrapper.planning_event(port)
    assert wrapper.records[0]["reason"]=="READY_DEADLINE_MISSED"
    assert wrapper._handoff_service==0. and wrapper._service_until==30.5
    wrapper.pending=dict(saved,due=29.,record_index=len(wrapper.records))
    wrapper.records.append(dict(status="PREPARING",submitted=0))
    wrapper._service_until=29.
    prior_service=sum(r["wall_s"] for r in wrapper.online_services)
    wrapper.online_service(29.,2.,1.,"admission")
    port.time=29.
    port.mark_latency_submission_ready=lambda token,handoff:SimpleNamespace(accepted=False,reason="DEPENDENCY_CHANGED")
    wrapper.planning_event(port)
    registration=wrapper.records[-1]["ready_registration_bookkeeping_s"]
    assert wrapper.pending is None and wrapper._handoff_service==0.
    assert wrapper._service_until==pytest.approx(31.+registration)
    assert sum(r["wall_s"] for r in wrapper.online_services)==pytest.approx(prior_service+2.+registration)


def test_pending_packet_and_scheduler_roundtrip_with_existing_checkpoint_codec(tmp_path):
    # Save only a completed Python call: the pending work is executor event state.
    w = world(); w.seed_hold("a","inner",0,.9)
    timing = PlanningTiming(0.,30.,29.)
    plan = generate_final("a","inner",0,0,30.,config=w.config).plan
    result, token = prepare_submission(w,(plan,),timing)
    assert result.accepted,result
    w.advance(1.,False)
    assert mark_submission_ready(w,token).accepted
    scheduler = LatencyAwareScheduler(NormalSchedulerCallback())
    scheduler.pending = dict(timing=timing,stage="PREPARED",due=30.,proposal=(plan,),
        evidence=[],reason=None,token=token,online_service_s=0.,record_index=0)
    scheduler.records = [dict(snapshot_time=0.,effective_time=30.,ready_deadline=29.,submitted=0)]
    identity = checkpoint_identity(dict(source_hash="test",base_config_hash=w.config.config_hash,
        overlay_sha256="test"),digest(w.manifest.to_dict()))
    path = tmp_path/"pending.json"
    save_checkpoint(path,w,scheduler,identity)
    loaded_scheduler = LatencyAwareScheduler(NormalSchedulerCallback())
    loaded = restore_checkpoint(path,loaded_scheduler,identity)
    assert loaded_scheduler.pending["token"]==token
    assert loaded_scheduler.pending["timing"]==timing
    assert guard_submission(loaded,token).accepted
    assert loaded_scheduler.state_dict()["emergency_timing_mode"]=="existing_priority_response_cpu_charged"
