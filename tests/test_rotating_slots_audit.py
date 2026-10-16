"""Independent physical-ledger checks, including self-consistent hash tampering."""
import json
import math
from hashlib import sha256

import pytest

from terminal_marl.core.rotating_slots.audit import audit_replay
from terminal_marl.core.rotating_slots.demand import ArrivalManifest,ArrivalRequest
from terminal_marl.core.rotating_slots.driver import DeterministicDriver
from terminal_marl.core.rotating_slots.energy import integrate_energy
from terminal_marl.core.rotating_slots.replay import ReplayBundle,BufferedStateJSONLSink
from terminal_marl.core.rotating_slots.trajectory import generate_transfer
from terminal_marl.core.rotating_slots.world import RotatingSlotWorld


def _rewrite_member(root,name,payload):
    if name.endswith(".jsonl"):
        content=b"".join(json.dumps(row,sort_keys=True).encode()+b"\n" for row in payload)
    else:
        content=json.dumps(payload,sort_keys=True).encode()
    (root/name).write_bytes(content)
    index=json.loads((root/"replay.json").read_bytes())
    index["files"][name]=sha256(content).hexdigest()
    (root/"replay.json").write_text(json.dumps(index),encoding="utf-8")


@pytest.fixture
def normal_archive(tmp_path):
    with BufferedStateJSONLSink(tmp_path/"source_states.jsonl") as sink:
        w=RotatingSlotWorld(manifest=ArrivalManifest((ArrivalRequest("normal",0,0,.85,"short"),)),state_sink=sink)
        w.advance(250.,DeterministicDriver())
        assert w.aircraft["normal"].status=="CLEARED"
        archive=ReplayBundle.write(tmp_path/"normal",world=w,code_commit="audit-test")
    return archive.path


def test_audits_actual_normal_archive_with_bounded_streaming_and_no_mutation(normal_archive,tmp_path):
    before=(normal_archive/"replay.json").read_bytes()
    report=audit_replay(normal_archive,output=tmp_path/"audit.json")
    assert report["status"]=="PASS",report["issues"]
    assert report["physical_outcome"]=="COMPLETE"
    assert report["counts"]["normal_landed"]==1 and report["counts"]["admitted"]==1
    assert report["energy_intervals"]>100 and report["used_plans"]>=4
    assert report["conservation_equations"]==[True,True,True]
    assert (normal_archive/"replay.json").read_bytes()==before
    assert json.loads((tmp_path/"audit.json").read_text())["status"]=="PASS"


def test_audits_real_energy_failure_censoring_and_unreleased_demand(tmp_path):
    w=RotatingSlotWorld(manifest=ArrivalManifest((ArrivalRequest("future",10,0,.75,"short"),)))
    for pad in range(6): w.seed_service(f"pad{pad}",pad,remaining=200.)
    w.seed_hold("failure","outer",0,.00001)
    w.seed_hold("censored","middle",0,.75)
    w.advance(1.)
    assert w.failed
    archive=ReplayBundle.write(tmp_path/"failure",world=w,code_commit="audit-test")
    report=audit_replay(archive.path)
    assert report["status"]=="PASS",report["issues"]
    assert report["physical_outcome"]=="ENERGY_FAILURE"
    assert report["counts"]["unreleased"]==1 and report["counts"]["censored"]==1
    assert report["derived_costs"]["cost_failure"]==1000000
    assert report["continuous_pair_runs"]>0


def test_state_tamper_with_updated_content_hash_is_detected(normal_archive):
    states=[json.loads(row) for row in (normal_archive/"states.jsonl").read_text().splitlines()]
    for axis in range(3):
        states[3]["aircraft"][0]["p"][axis]+=.0009  # >1mm in 3D despite <1mm per coordinate.
    _rewrite_member(normal_archive,"states.jsonl",states)
    ReplayBundle.validate(normal_archive)  # Byte/hash consistency alone is insufficient.
    report=audit_replay(normal_archive)
    assert report["status"]=="FAIL"
    assert any(i["code"]=="POSE_MISMATCH" for i in report["issues"])


def test_metrics_tamper_with_updated_hash_is_not_hidden_by_conservation_flag(normal_archive):
    metrics=json.loads((normal_archive/"metrics.json").read_bytes())
    metrics["admitted"]+=1
    metrics["conservation"]=True
    _rewrite_member(normal_archive,"metrics.json",metrics)
    ReplayBundle.validate(normal_archive)
    report=audit_replay(normal_archive)
    assert report["status"]=="FAIL"
    assert any(i["code"]=="METRIC_COUNT" and i.get("metric")=="admitted" for i in report["issues"])


def test_airborne_energy_and_soc_tamper_is_detected_by_interval_integration(tmp_path):
    w=RotatingSlotWorld();w.seed_hold("transfer","outer",0,.75)
    c=generate_transfer("transfer","outer",0,0,0)
    assert c.status=="FEASIBLE" and w.submit((c.plan,)).accepted
    w.advance(.4)
    archive=ReplayBundle.write(tmp_path/"energy_ledger",world=w,code_commit="audit-test")
    baseline=audit_replay(archive.path)
    assert baseline["status"]=="PASS",baseline["issues"]
    assert baseline["physical_outcome"]=="IN_PROGRESS"
    states=[json.loads(row) for row in (archive.path/"states.jsonl").read_text().splitlines()]
    states[1]["aircraft"][0]["energy_j"]+=100.
    states[1]["aircraft"][0]["soc"]+=.001
    _rewrite_member(archive.path,"states.jsonl",states)
    ReplayBundle.validate(archive.path)
    report=audit_replay(archive.path)
    codes={i["code"] for i in report["issues"]}
    assert report["status"]=="FAIL" and {"ENERGY_DELTA","SOC_DELTA"}<=codes


def test_continuous_pair_run_finds_crossing_despite_safe_recorded_endpoints(tmp_path):
    # Negative archive deliberately claims two individually executable crossed
    # transfers. It was not accepted by a joint world executor.
    w=RotatingSlotWorld()
    w.seed_hold("a","outer",0,.9)
    w.seed_hold("b","outer",1,.9)
    source=w.to_dict()
    plans=[]
    for aid,slot,target in (("a",0,1),("b",1,0)):
        c=generate_transfer(aid,"outer",slot,target,0,durations=[60])
        assert c.status=="FEASIBLE",c
        plans.append(c.plan)
    start=json.loads(json.dumps(source["states"][0]));end=json.loads(json.dumps(start));end["time"]=60.
    source["trajectories"]={p.trajectory_hash:p.to_dict() for p in plans}
    source["events"]=[]
    energy=0.
    for i,p in enumerate(plans):
        a=start["aircraft"][i];b=end["aircraft"][i]
        a["trajectory_hash"]=b["trajectory_hash"]=p.trajectory_hash
        a["status"]="TRANSFER_O_M"
        k=p.state(60);q=integrate_energy(p,0,60)
        assert q.valid
        b.update(status="HOLD_MIDDLE",ring="middle",slot=p.target_slot,p=list(k.p),v=list(k.v),a=list(k.a),
                 soc=.9-q.soc_used,energy_j=q.joules,peak_seconds=q.peak_seconds,airborne_time=60.)
        energy+=q.joules
        source["events"].append({"time":0.,"type":"maneuver_started","aircraft_id":p.aircraft_id,"details":{"trajectory_hash":p.trajectory_hash}})
    assert math.dist(start["aircraft"][0]["p"],start["aircraft"][1]["p"])>=80
    assert math.dist(end["aircraft"][0]["p"],end["aircraft"][1]["p"])>=80
    source["states"]=[start,end]
    source["metrics"].update(energy_j=energy,energy_kwh=energy/3600000,airborne_time=120.,total_cost=120.)
    archive=ReplayBundle.write(tmp_path/"crossing",record=source,code_commit="negative-audit-test")
    report=audit_replay(archive.path)
    assert report["status"]=="FAIL" and report["continuous_pair_runs"]==1
    assert any(i["code"]=="UNREPORTED_HARD_SEPARATION" for i in report["issues"])


def test_explicit_analytic_visual_fixture_is_not_reported_as_audited_physics(tmp_path):
    w=RotatingSlotWorld();w.seed_hold("display","outer",0)
    source=w.to_dict();source["metrics"]["fixture_only"]=True
    archive=ReplayBundle.write(tmp_path/"visual",record=source,code_commit="visual-fixture")
    report=audit_replay(archive.path)
    assert report["status"]=="NOT_PHYSICAL_EPISODE"
    assert report["continuous_separation"]=="NOT_AUDITED"
