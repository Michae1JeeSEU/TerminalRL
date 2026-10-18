"""New demand/pad policy: bounded real physics and queue accounting checks."""
from dataclasses import replace
from hashlib import sha256
import json

import pytest

from terminal_marl.core.rotating_slots import generate_final, generate_inbound
from terminal_marl.core.rotating_slots.audit import audit_replay
from terminal_marl.core.rotating_slots.demand import ArrivalManifest, ArrivalRequest
from terminal_marl.core.rotating_slots.driver import DeterministicDriver
from terminal_marl.core.rotating_slots.models import CandidateResult
from terminal_marl.core.rotating_slots.replay import ReplayBundle
from terminal_marl.core.rotating_slots.world import POLICY_ID, RotatingSlotWorld


def test_same_gate_real_wait_then_admit_keeps_request_soc_and_accounts_wait(tmp_path):
    requests=(ArrivalRequest("first",0,0,.9,"short"),ArrivalRequest("second",.1,0,.9,"short"))
    w=RotatingSlotWorld(manifest=ArrivalManifest(requests))
    driver=DeterministicDriver()
    w.advance(.1,driver)
    assert "first" in w.aircraft and "second" not in w.aircraft
    assert w.metrics()["upstream_waiting"]==1 and w.metrics()["physical_airborne_count"]==1
    assert w.snapshot().to_dict()["upstream"][0]["initial_soc"]==.9
    assert w.metrics()["rerouted"]==0 and w.metrics()["conservation"]
    w.advance(20,driver)
    second=w.aircraft["second"]
    assert second.admitted and second.release_time==.1 and second.admission_time>.1
    assert second.upstream_wait_seconds==pytest.approx(second.admission_time-.1)
    entry=next(frame for frame in w.records if frame["time"]==second.admission_time)
    assert next(a for a in entry["aircraft"] if a["id"]=="second")["soc"]==.9
    assert w.manifest.requests==requests
    m=w.metrics()
    assert m["cost_upstream_wait"]==pytest.approx(second.upstream_wait_seconds)
    assert m["total_cost"]==pytest.approx(m["airborne_time"]+m["cost_upstream_wait"])
    archive=ReplayBundle.write(tmp_path/"wait_admit",world=w,code_commit="test-new-policy")
    report=audit_replay(archive.path)
    assert report["status"]=="PASS",report["issues"]


def test_driver_same_gate_fifo_other_gate_independent():
    requests=(ArrivalRequest("z",0,0,.1,"short"),ArrivalRequest("a",.1,0,.9,"short"),
              ArrivalRequest("other",.1,1,.9,"short"))
    w=RotatingSlotWorld(manifest=ArrivalManifest(requests)); w.advance(.1,DeterministicDriver())
    assert "z" not in w.aircraft and "a" not in w.aircraft and w.aircraft["other"].admitted
    assert not any(e["type"]=="admission_waiting" and e["aircraft_id"]=="a" for e in w.events)
    assert {r["id"] for r in w.snapshot().to_dict()["upstream"]}=={"z","a"}
    assert w.metrics()["conservation"]


def test_delayed_entry_does_not_backdate_hold_calendar_to_request_release():
    r=ArrivalRequest("delayed",0,0,.9,"short")
    w=RotatingSlotWorld(manifest=ArrivalManifest((r,)))
    w.advance(5.,False)
    assert w.admit("delayed").accepted
    p=w.active["delayed"]
    # Narrow single-aircraft endpoint/settlement test, not a sampled replay.
    w._advance_energy(p.end_time);w._settle()
    assert w.aircraft["delayed"].status=="HOLD_OUTER"
    slots,_,_=w._calendar_stage(w.soft,())
    reservation=next(r for r in slots.reservations if r.aircraft_id=="delayed")
    assert w.aircraft["delayed"].release_time==0.
    assert reservation.start==w.aircraft["delayed"].admission_time==5.


def test_full_37_perpetual_capacity_only_and_finite_release_not_capacity(monkeypatch):
    request=ArrivalRequest("r",0,0,.9,"short")
    w=RotatingSlotWorld(manifest=ArrivalManifest((request,)))
    for slot in range(37): w.seed_hold(f"outer{slot}","outer",slot,.9)
    assert w.reroute("r").reason=="INVALID_REROUTE"
    w._settle()
    evidence=w._outer_capacity()
    assert len(evidence)==37
    w.slots._entries.append(replace(w.slots._entries[0],start=-2.,end=-1.))
    assert len(w._outer_capacity())==37
    outcome=w.admit("r")
    assert not outcome.accepted and outcome.status=="REROUTED" and outcome.reason=="OUTER_CAPACITY"
    assert w.aircraft["r"].status=="REROUTED"
    assert w.reroute("r").reason=="INVALID_REROUTE"
    assert next(e for e in w.events if e["type"]=="rerouted")["details"]["reason"]=="OUTER_CAPACITY"
    assert w.metrics()["conservation"]
    # A recorded finite release invalidates a capacity proof even while full now.
    w.aircraft.pop("r")
    w.slots._entries[0]=replace(w.slots._entries[0],end=10.)
    assert w._outer_capacity() is None
    calls=[]
    from terminal_marl.core.rotating_slots import world as module
    def unavailable(*args,**kwargs):
        calls.append(args[2]); return CandidateResult("NOT_FOUND",reason="bounded search exhausted")
    monkeypatch.setattr(module,"generate_inbound",unavailable)
    result=w.admit("r")
    assert result.status=="WAITING" and calls==list(range(37)) and "r" not in w.aircraft
    assert not w.reroute("r","NO_PLAN_FOUND").accepted


def test_energy_or_not_found_waits_and_horizon_is_lower_bound(tmp_path):
    r=ArrivalRequest("low",0,0,.01,"short")
    w=RotatingSlotWorld(manifest=ArrivalManifest((r,)))
    candidate=generate_inbound("low",0,1,0).plan
    w.advance(0,lambda s,p:p.admit("low",candidate))
    assert "low" not in w.aircraft and w.metrics()["upstream_waiting"]==1
    assert w.snapshot().to_dict()["upstream"][0]["reason_categories"]==["SOC_INSUFFICIENT"]
    w.advance(.4,False)
    m=w.metrics()
    assert m["upstream_wait_time"]==pytest.approx(.4) and m["energy_j"]==0 and m["holding_time"]==0
    assert m["cost_is_lower_bound"] and not m["truncated"] and m["physical_airborne_count"]==0
    archive=ReplayBundle.write(tmp_path/"waiting",world=w,code_commit="test-new-policy")
    report=audit_replay(archive.path)
    assert report["status"]=="PASS",report["issues"]
    assert report["physical_outcome"]=="IN_PROGRESS"


def test_external_queue_snapshot_is_deep_immutable_and_export_detached():
    r=ArrivalRequest("low",0,0,.01,"short")
    w=RotatingSlotWorld(manifest=ArrivalManifest((r,)))
    p=generate_inbound("low",0,1,0).plan
    w.advance(0,lambda s,port:port.admit("low",p))
    snapshot=w.snapshot()
    with pytest.raises(TypeError): snapshot.upstream[0]["reason"]="changed"
    with pytest.raises(AttributeError): snapshot.upstream[0]["reason_categories"].append("changed")
    exported=snapshot.to_dict()
    exported["upstream"][0]["reason_categories"].append("changed")
    assert w.snapshot().to_dict()["upstream"][0]["reason_categories"]==["SOC_INSUFFICIENT"]


def test_physical_failure_censors_external_queue_separately(tmp_path):
    requests=(ArrivalRequest("waiting",0,0,.1,"short"),ArrivalRequest("future",100,1,.8,"short"))
    w=RotatingSlotWorld(manifest=ArrivalManifest(requests)); w.auto_emergency=False
    w.seed_hold("failure","inner",0,.00001)
    w.advance(1,False)
    m=w.metrics()
    assert w.failed and m["upstream_waiting"]==m["censored_upstream"]==1 and m["unreleased"]==1
    assert m["physical_airborne_count"]==0 and m["conservation"]
    assert w.snapshot().to_dict()["upstream"][0]["status"]=="CENSORED_UPSTREAM"
    frozen=(w.time,dict(m),w.manifest,w.snapshot().to_dict()["upstream"])
    assert w.admit("waiting").reason=="EPISODE_STOPPED"
    assert w.reroute("waiting").reason=="EPISODE_STOPPED"
    assert (w.time,w.metrics(),w.manifest,w.snapshot().to_dict()["upstream"])==frozen
    archive=ReplayBundle.write(tmp_path/"failure",world=w,code_commit="test-new-policy")
    report=audit_replay(archive.path)
    assert report["status"]=="PASS",report["issues"]


def test_diagnostic_drain_does_not_call_external_waiting_clear(tmp_path,monkeypatch):
    from terminal_marl.core.rotating_slots.diagnostics import _diagnostic_job
    monkeypatch.setattr(ArrivalManifest,"generate",classmethod(lambda cls,*args:cls((ArrivalRequest("waiting",0,0,.01,"short"),))))
    row=_diagnostic_job(dict(id="named_queue_fixture",kind="traffic",seed=11,rate=60.,window=0.,drain=0.,
                             output=str(tmp_path),command="unit-test external queue horizon"))
    assert row["status"]=="COMPLETED" and row["metrics"]["upstream_waiting"]==1
    assert row["cleared_by_drain_end"] is False and row["clear_completed_time"] is None
    assert row["metrics"]["cost_is_lower_bound"] is True


def test_emergency_can_depart_before_service_release_but_overlap_rejects():
    def make(remaining):
        w=RotatingSlotWorld(); w.seed_service("service",0,remaining=remaining)
        w.seed_hold("emergency","inner",0,.8,emergency=True)
        p=generate_final("emergency","inner",0,0,0,emergency=True).plan
        return w,p
    w,p=make(10.)
    assert p.end_time-13.5>10 and w.submit((p,)).accepted
    assert next(r for r in w.pads.reservations if r.aircraft_id=="emergency").start==p.end_time-13.5
    w,p=make(300.)
    result=w.submit((p,))
    assert result.reason=="RESOURCE_CONFLICT" and result.details["blocking_reservations"][0]["aircraft_id"]=="service"
    assert "emergency" not in w.active and w.aircraft["service"].service_end==300.


def test_explicit_legacy_policy_fixture_remains_readable(tmp_path):
    w=RotatingSlotWorld();w.seed_service("ground",0,remaining=.4);w.advance(.2,False)
    record=w.to_dict()
    record.pop("policy_id")
    record["metrics"].pop("policy_id")
    for frame in record["states"]:
        frame.pop("policy_id");frame.pop("upstream")
        frame["metrics"].pop("policy_id")
    archive=ReplayBundle.write(tmp_path/"legacy",record=record,code_commit="test-legacy")
    report=audit_replay(archive.path)
    assert report["status"]=="PASS",report["issues"]
    assert report["policy_id"]=="legacy_immediate_admission_departure_pad"


def test_new_audit_recomputes_queue_cost_despite_valid_archive_hashes(tmp_path):
    r=ArrivalRequest("waiting",0,0,.01,"short")
    w=RotatingSlotWorld(manifest=ArrivalManifest((r,)))
    w.advance(.4,False)
    archive=ReplayBundle.write(tmp_path/"tampered_queue",world=w,code_commit="test-new-policy")
    metrics_path=archive.path/"metrics.json"
    metrics=json.loads(metrics_path.read_bytes())
    metrics["cost_upstream_wait"]+=1
    metrics["total_cost"]+=1
    payload=json.dumps(metrics,allow_nan=False).encode()
    metrics_path.write_bytes(payload)
    index_path=archive.path/"replay.json"; index=json.loads(index_path.read_bytes())
    index["files"]["metrics.json"]=sha256(payload).hexdigest()
    index_path.write_text(json.dumps(index),encoding="utf-8")
    report=audit_replay(archive.path)
    assert report["status"]=="FAIL"
    assert any(i["code"]=="METRIC_LEDGER" and i.get("metric")=="cost_upstream_wait" for i in report["issues"])


@pytest.mark.parametrize("ordinary_start,active,should_preempt",[(10.,False,True),(500.,False,False),(0.,True,False)])
def test_soft_preemption_only_for_overlap_and_never_active(monkeypatch,ordinary_start,active,should_preempt):
    """Isolate resource transactions; real continuous physics is tested separately."""
    from terminal_marl.core.rotating_slots import world as module
    from terminal_marl.core.rotating_slots.models import ValidationResult
    w=RotatingSlotWorld()
    w.seed_hold("ordinary","inner",0,.99)
    w.seed_hold("rescue","inner",9,.9,emergency=True)
    normal=generate_final("ordinary","inner",0,0,ordinary_start).plan
    emergency=generate_final("rescue","inner",9,0,0,emergency=True).plan
    assert normal and emergency
    monkeypatch.setattr(module,"validate_separation",lambda *a,**k:ValidationResult(True,"SAFE"))
    assert w.submit((normal,)).accepted
    assert ("ordinary" in w.active)==active
    result=w.submit((emergency,))
    if active:
        assert not result.accepted and w.active["ordinary"] is normal
    else:
        assert result.accepted
        assert ("ordinary" not in w.soft)==should_preempt
        cancelled=[e for e in w.events if e["type"]=="soft_plan_cancelled" and e["aircraft_id"]=="ordinary"]
        assert bool(cancelled)==should_preempt
