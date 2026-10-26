"""Affected-traffic certification with real approved trajectories and executor."""
from dataclasses import replace
import math
import time

import pytest

from tests.test_rotating_slots_planning_latency import Durations, CertifiedFinals, assert_same_state
from terminal_marl.core.rotating_slots import planning_latency as latency
from terminal_marl.core.rotating_slots.demand import ArrivalManifest, ArrivalRequest
from terminal_marl.core.rotating_slots.models import ValidationResult
from terminal_marl.core.rotating_slots.trajectory import generate_final, generate_transfer, TrajectoryPlan
from terminal_marl.core.rotating_slots.world import RotatingSlotWorld, STAGE2_POLICY_ID
from terminal_marl.core.rotating_slots.stage2_campaign import save_checkpoint, restore_checkpoint, checkpoint_identity, digest


class StatefulFinals(CertifiedFinals):
    def state_dict(self): return dict(plans=[p.to_dict() for p in self.plans])
    def load_state_dict(self,state): self.plans=tuple(TrajectoryPlan.from_dict(p) for p in state["plans"])


def setup(arrivals=(20.,),revalidation_seconds=2.,callback=None,ring="inner"):
    manifest=ArrivalManifest(tuple(ArrivalRequest("new"+str(i),t,(i+2)%4,.9,"standard") for i,t in enumerate(arrivals)))
    w=RotatingSlotWorld(manifest=manifest,experiment_policy_id=STAGE2_POLICY_ID)
    w.seed_hold("a",ring,0,.9)
    callback=callback or StatefulFinals()
    clock=Durations(10.,1.)
    wrapper=latency.LatencyAwareScheduler(callback,clock=clock); w.decision_callback=wrapper
    admitted=[]
    def admission(snapshot,port):
        for rid in sorted(port.requests):
            if rid in w.aircraft: continue
            result=port.admit(rid)
            assert result.accepted,result
            admitted.append(rid)
            # Next timed calls are inspection and affected certification.
            clock.durations=iter((0.,revalidation_seconds,0.))
    w.admission_callback=admission
    return w,wrapper,callback,clock,admitted


def test_unrelated_arrival_revalidates_and_joins_original_plan_without_early_resources(monkeypatch):
    w,wrapper,callback,_,admitted=setup()
    original_admission=w.admission_callback; checked_cap=[]
    def admission(snapshot,port):
        original_admission(snapshot,port)
        if admitted and not checked_cap:
            token=wrapper.pending["token"]
            packet=w._planning_latency_packets[token]
            old=dict(packet["pair_certificates"])
            with monkeypatch.context() as patch:
                def forbidden(*args,**kwargs): raise AssertionError("zero compute cap started a projection")
                patch.setattr(latency,"project_visible_snapshot",forbidden)
                result=port.revalidate_latency_submission(token,max_work_s=0.)
            assert result.status=="UNKNOWN" and result.reason=="REVALIDATION_WORK_DEADLINE"
            assert packet["revalidation"] is None and packet["ready_at"] is None
            assert packet["pair_certificates"]==old
            checked_cap.append(True)
    w.admission_callback=admission
    initial=w.aircraft["a"]; old=w._base(initial)
    w.advance(29.5)
    record=wrapper.records[0]
    assert admitted==["new0"] and record["revalidation_count"]==1
    assert record["revalidations"][0]["status"]=="CERTIFIED",record
    assert record["revalidations"][0]["affected_aircraft_ids"]==["new0"]
    assert record["revalidations"][0]["pairs_checked"]==1
    assert record["revalidation_work_s"]>=2. and record["prepare_work_s"]==1.
    assert checked_cap==[True]
    assert record["ready_time"]<=record["ready_deadline"]
    p=callback.plans[0]
    assert p.trajectory_hash not in w.trajectories and "a" not in w.active and "a" not in w.soft
    assert all(r.aircraft_id!="a" for r in w.pads.reservations)
    w.advance(30.01)
    assert wrapper.records[0]["status"]=="COMMITTED",wrapper.records
    assert w.active["a"].trajectory_hash==p.trajectory_hash and p.start_time==30.
    assert_same_state(old.state(30.),p.state(30.))
    assert w.aircraft["a"].soc<initial.soc


def test_unknown_affected_pair_and_changed_pad_reject_entire_group(monkeypatch):
    w,wrapper,_,_,_=setup()
    monkeypatch.setattr(latency,"validate_separation",lambda *a,**kw:ValidationResult(False,"UNKNOWN","PAIR_UNRESOLVED"))
    w.advance(30.01)
    assert wrapper.records[0]["status"]=="REJECTED" and wrapper.records[0]["reason"]=="PAIR_UNRESOLVED"
    assert "a" not in w.active and "a" not in w.soft
    monkeypatch.undo()
    w,wrapper,_,_,_=setup(arrivals=())
    w.seed_hold("e","inner",9,.9)
    sent=[]
    def occupy(snapshot,port):
        if snapshot.time>=20. and not sent:
            a=w.aircraft["e"]; w.aircraft[a.id]=replace(a,emergency=replace(a.emergency,eligible=True,declared=True))
            p=generate_final(a.id,a.ring,a.slot,0,w.time,emergency=True,config=w.config).plan
            result=port.submit((p,)); assert result.accepted,result
            sent.append(p)
    w.admission_callback=occupy
    w.advance(30.01)
    assert wrapper.records[0]["status"]=="REJECTED" and wrapper.records[0]["reason"]=="RESOURCE_DEPENDENCY_CHANGED"
    assert "a" not in w.active and w.active["e"] is sent[0]


def test_late_revalidation_occupies_cpu_and_late_arrival_never_executes_expired_action():
    w,wrapper,callback,_,_=setup(revalidation_seconds=15.)
    w.advance(30.01)
    assert wrapper.pending["stage"]=="REVALIDATING"
    record=wrapper.records[0]
    assert record["ready_time"] is None and "completed_time" not in record["revalidations"][0]
    assert len(callback.calls)==1 and "a" not in w.active
    w.advance(37.)
    assert record["reason"]=="READY_DEADLINE_MISSED" and record["finished_time"]>=35.
    assert record["revalidations"][0]["status"]=="LATE"
    w,wrapper,_,_,_=setup(arrivals=(29.5,))
    w.advance(30.01)
    assert wrapper.records[0]["reason"]=="TRAFFIC_CHANGED_AFTER_DEADLINE"
    assert wrapper.records[0]["revalidation_count"]==0 and "a" not in w.active


def test_change_during_work_supersedes_candidate_certificate_before_join():
    w,wrapper,_,clock,admitted=setup(arrivals=(20.,21.),revalidation_seconds=4.)
    old_admission=w.admission_callback
    def admission(snapshot,port):
        before=len(admitted); old_admission(snapshot,port)
        if len(admitted)>before and len(admitted)==2:
            # First completion guard, then replacement certification.
            clock.durations=iter((0.,1.,0.))
    w.admission_callback=admission
    w.advance(30.01)
    record=wrapper.records[0]
    assert len(admitted)==2 and record["status"]=="COMMITTED",record
    assert record["revalidation_count"]==2
    assert [r["status"] for r in record["revalidations"]]==["SUPERSEDED","CERTIFIED"]
    assert record["revalidations"][1]["affected_aircraft_ids"]==["new0","new1"]
    assert record["revalidations"][1]["pairs_checked"]==2


def test_two_successive_groups_with_arrivals_preserve_soc_c2_and_hashes():
    class Transfers(StatefulFinals):
        def __call__(self,port):
            clock.durations=iter((10.,1.,0.))
            s=port.build_planning_snapshot(); self.calls.append(s); a=s.aircraft["a"]
            if a.id in s.active or a.id in s.soft:
                evaluated=port.evaluate_submission(())
                assert evaluated.accepted
                assert port.commit_validated_prefix(evaluated).status=="DEFERRED"
                return
            p=generate_transfer(a.id,a.ring,a.slot,0,s.time,source_version=a.version,config=s.config).plan
            assert p is not None
            self.plans=(*self.plans,p)
            evaluated=port.evaluate_submission((p,)); assert evaluated.accepted,evaluated
            assert port.commit_validated_prefix(evaluated).status=="DEFERRED"
    callback=Transfers()
    w,wrapper,_,clock,admitted=setup(arrivals=(20.,65.),callback=callback,ring="outer")
    w.seed_hold("b","outer",18,.9)
    original_admission=w.admission_callback; changed=[]
    def admission(snapshot,port):
        original_admission(snapshot,port)
        if snapshot.time>=20. and not changed:
            p=generate_transfer("b","outer",18,14,100.,config=w.config).plan
            result=port.submit((p,)); assert result.accepted,result
            changed.append(p)
    w.admission_callback=admission
    start=w.aircraft["a"]; old=w._base(start)
    w.advance(29.5); assert "a" not in w.active and "a" not in w.soft
    w.advance(30.01); first_soc=w.aircraft["a"].soc
    first=callback.plans[0]; assert w.active["a"].trajectory_hash==first.trajectory_hash
    w.advance(79.5)
    assert "a" not in w.soft
    assert callback.plans[1].trajectory_hash not in w.trajectories
    w.advance(80.01)
    second=callback.plans[1]
    assert [r["status"] for r in wrapper.records[:3]]==["COMMITTED","NO_EXECUTABLE_PREFIX","COMMITTED"],wrapper.records
    assert any(r["horizon_extension_pairs"]>0 for r in wrapper.records[0]["revalidations"])
    assert wrapper.records[2]["revalidations"][0]["reused_pairs"]>0
    assert len(admitted)==2 and w.active["a"].trajectory_hash==second.trajectory_hash
    assert start.soc>first_soc>w.aircraft["a"].soc
    assert_same_state(old.state(30.),first.state(30.))
    assert_same_state(first.state(80.),second.state(80.))


def test_pending_revalidation_checkpoint_roundtrip_and_future_wall_clock_binding(tmp_path):
    w,wrapper,callback,_,_=setup(revalidation_seconds=5.)
    w.advance(21.)
    assert wrapper.pending["stage"]=="REVALIDATING"
    identity=checkpoint_identity(dict(source_hash="test",base_config_hash=w.config.config_hash,overlay_sha256="test"),digest(w.manifest.to_dict()))
    path=tmp_path/"revalidation.checkpoint.json"
    save_checkpoint(path,w,wrapper,identity)
    restored_scheduler=latency.LatencyAwareScheduler(StatefulFinals(),clock=Durations())
    restored=restore_checkpoint(path,restored_scheduler,identity)
    assert restored_scheduler.pending["stage"]=="REVALIDATING"
    assert restored_scheduler.records[0]["ready_time"] is None
    restored.advance(30.01,restored_scheduler)
    assert restored_scheduler.records[0]["status"]=="COMMITTED",restored_scheduler.records
    assert restored.active["a"].trajectory_hash==callback.plans[0].trajectory_hash
    captured=[]
    class Capture(StatefulFinals):
        def __call__(self,port):
            captured.append(port.planning_wall_deadline)
            super().__call__(port)
    w,_,_,_,_=setup(arrivals=())
    scheduler=latency.LatencyAwareScheduler(Capture(),clock=Durations(100.))
    scheduler._service_until=2.
    before=time.perf_counter(); scheduler(w._scheduler_port); after=time.perf_counter()
    assert before+27.<=captured[0]<=after+27.


def test_promote_guard_time_enters_ready_deadline_and_successful_cpu_queue(monkeypatch):
    class Perf:
        def __init__(self): self.value=100.
        def __call__(self): self.value+=.001; return self.value
    for now,handoff,should_accept in ((28.9,.05,False),(20.,.5,True)):
        w,wrapper,_,_,_=setup(arrivals=())
        w.advance(now)
        p=wrapper.pending; record=wrapper.records[0]
        assert p["stage"]=="PREPARED"
        record["revalidations"]=[dict(status="GUARDING")]
        p.update(stage="READY_SERVICE",due=now,reason=None,completion_revalidation_index=0,
                 completion_revalidation_status="CERTIFIED",revalidation_attempt=1)
        wrapper._handoff_service=handoff
        perf=Perf()
        def promote(*args,**kwargs):
            perf.value+=.06
            return ValidationResult(True,"REVALIDATED")
        port=type("Port",(),{})()
        port.time=w.time
        port.complete_latency_revalidation=promote
        port.mark_latency_submission_ready=lambda token,delay:latency.mark_submission_ready(w,token,handoff_delay=delay)
        port.discard_latency_submission=w._scheduler_port.discard_latency_submission
        port.planning_latency_event=w._scheduler_port.planning_latency_event
        previous=record["ready_registration_bookkeeping_s"]
        with monkeypatch.context() as patch:
            patch.setattr(latency.time,"perf_counter",perf)
            wrapper.planning_event(port)
        assert record["ready_registration_bookkeeping_s"]>previous+.06
        if should_accept:
            assert wrapper.pending["stage"]=="PREPARED" and record["ready_time"]>now+handoff+.06
            assert wrapper._service_until>=record["ready_time"]
        else:
            assert record["reason"]=="READY_DEADLINE_MISSED" and wrapper.pending is None
