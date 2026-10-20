"""Independent Stage2 replay acceptance and self-consistent corruption checks."""
from datetime import datetime, timezone
from dataclasses import replace
from hashlib import sha256
import json
import time

import pytest

from terminal_marl.core.rotating_slots.audit import audit_replay
from terminal_marl.core.rotating_slots.demand import ArrivalManifest, ArrivalRequest
from terminal_marl.core.rotating_slots.replay import ReplayBundle
from terminal_marl.core.rotating_slots.trajectory import generate_final
from terminal_marl.core.rotating_slots.world import RotatingSlotWorld, STAGE2_POLICY_ID


@pytest.fixture(autouse=True)
def timing(request):
    utc=datetime.now(timezone.utc).isoformat();clock=time.perf_counter()
    yield
    print(json.dumps(dict(test=request.node.nodeid,start_utc=utc,
        end_utc=datetime.now(timezone.utc).isoformat(),wall_s=time.perf_counter()-clock)))


def write_member(root,name,data):
    content=(b"".join(json.dumps(row,sort_keys=True).encode()+b"\n" for row in data)
             if name.endswith(".jsonl") else json.dumps(data,sort_keys=True).encode())
    (root/name).write_bytes(content)
    index=json.loads((root/"replay.json").read_bytes())
    index["files"][name]=sha256(content).hexdigest()
    (root/"replay.json").write_text(json.dumps(index),encoding="utf-8")


def stage2_world(manifest=None):
    w=RotatingSlotWorld(experiment_policy_id=STAGE2_POLICY_ID,manifest=manifest)
    # These fixtures audit physical ledgers; common-dispatch policy is separate.
    w.auto_emergency=False
    w._update_emergency=lambda: False
    return w


@pytest.fixture
def loss_archive(tmp_path):
    w=stage2_world()
    w.seed_service("ground",2,remaining=.2,soc=.6)
    w.seed_hold("lost_a","outer",0,soc=1e-6)
    w.seed_hold("lost_b","outer",18,soc=1e-6)
    w.seed_hold("other","middle",0)
    w.advance(.4)
    return ReplayBundle.write(tmp_path/"loss",world=w,code_commit="stage2-audit-test").path


def test_stage2_no_loss_ground_exclusion_upstream_pending_and_conservation(tmp_path):
    w=stage2_world(ArrivalManifest((ArrivalRequest("waiting",.1,0,.8,"short"),)))
    w.seed_service("ground",2,remaining=.2,soc=.6)
    w.seed_hold("airborne","middle",0)
    w.advance(.4)
    path=ReplayBundle.write(tmp_path/"no_loss",world=w,code_commit="stage2-audit-test").path
    report=audit_replay(path)
    assert report["status"]=="PASS",report["issues"]
    assert report["counts"]["initial_ground_count"]==1
    assert report["counts"]["N_offered"]==2 and report["counts"]["N_admitted_total"]==1
    assert report["counts"]["L_success"]==0 and report["counts"]["N_upstream_pending"]==1
    assert report["conservation_equations"]==[True,True,True]


def test_stage2_simultaneous_isolation_continues_with_independent_energy_checks(loss_archive):
    original=(loss_archive/"replay.json").read_bytes()
    report=audit_replay(loss_archive)
    assert report["status"]=="PASS",report["issues"]
    assert report["physical_outcome"]=="ENERGY_LOSSES_CONTINUED"
    assert report["counts"]["F_battery"]==2 and report["counts"]["F_peak"]==0
    assert report["counts"]["N_airborne_pending"]==1 and report["counts"]["censored"]==0
    assert report["derived_costs"]["cost_failure"]==2000000.
    assert report["continuous_separation"]=="CERTIFIED"
    assert (loss_archive/"replay.json").read_bytes()==original


def test_stage2_success_after_loss_uses_touchdown_event_not_service_completion(tmp_path):
    w=stage2_world()
    w.seed_hold("lost","outer",18,soc=1e-6)
    w.seed_hold("success","inner",0,soc=.8)
    plan=generate_final("success","inner",0,0,0.).plan
    assert plan and w.submit((plan,)).accepted
    w.advance(plan.end_time)
    path=ReplayBundle.write(tmp_path/"postloss",world=w,code_commit="stage2-audit-test").path
    report=audit_replay(path)
    assert report["status"]=="PASS",report["issues"]
    assert report["counts"]["successful_touchdowns_before_first_failure"]==0
    assert report["counts"]["successful_touchdowns_after_first_failure"]==1
    assert report["primary_key"][:2]==[1,-1]


def test_genuine_peak_budget_loss_is_audited_separately(tmp_path):
    w=stage2_world()
    w.seed_hold("peak","inner",0,soc=.8)
    plan=generate_final("peak","inner",0,0,0.).plan
    assert plan and w.submit((plan,)).accepted
    # A named synthetic test baseline exercises cumulative peak exhaustion.
    w.aircraft["peak"]=replace(w.aircraft["peak"],peak_seconds=119.99)
    w._record(force=True)
    w.advance(plan.end_time)
    assert w.failure_archive["peak"]["reason"]=="PEAK_BUDGET_EXHAUSTED"
    path=ReplayBundle.write(tmp_path/"peak",world=w,code_commit="stage2-audit-test").path
    report=audit_replay(path)
    assert report["status"]=="PASS",report["issues"]
    assert report["counts"]["F_peak"]==1 and report["counts"]["F_battery"]==0


@pytest.mark.parametrize("corruption,code",[
    ("reason","UNEXPLAINED_PEAK_LOSS"),
    ("duplicate","DUPLICATE_ENERGY_LOSS"),
    ("penalty","METRIC_LEDGER"),
    ("energy","ISOLATED_STATE_CHANGED"),
    ("success","LOSS_DOUBLE_OUTCOME"),
    ("resource","LOSS_RESOURCE_RETAINED"),
    ("archive","LOSS_LAST_STATE"),
    ("ground","METRIC_COUNT"),
    ("fakeholding","FABRICATED_TOUCHDOWN"),
])
def test_self_consistent_stage2_corruption_is_rejected(loss_archive,corruption,code):
    root=loss_archive
    if corruption in ("reason","duplicate","success","fakeholding"):
        events=[json.loads(line) for line in (root/"events.jsonl").read_text().splitlines()]
        loss=next(e for e in events if e["type"]=="aircraft_energy_failure")
        if corruption=="reason":
            for e in events:
                if e.get("aircraft_id")==loss["aircraft_id"] and e["type"] in ("aircraft_energy_failure","failure_isolated"):
                    e["details"]["reason"]="PEAK_BUDGET_EXHAUSTED"
            records=json.loads((root/"failure_archive.json").read_bytes())
            records[loss["aircraft_id"]]["reason"]="PEAK_BUDGET_EXHAUSTED"
            write_member(root,"failure_archive.json",records)
        elif corruption=="duplicate":
            events.insert(events.index(loss)+1,json.loads(json.dumps(loss)))
        elif corruption=="success":
            events.append(dict(time=.4,type="touchdown",aircraft_id=loss["aircraft_id"],details={}))
        else:
            states=[json.loads(line) for line in (root/"states.jsonl").read_text().splitlines()]
            other=next(a for a in states[-1]["aircraft"] if a["id"]=="other")
            other.update(status="LANDED_SERVICE",touchdown_time=.4,service_end=60.4,pad_id=1)
            write_member(root,"states.jsonl",states)
            events.append(dict(time=.4,type="touchdown",aircraft_id="other",details={}))
        write_member(root,"events.jsonl",events)
    elif corruption in ("penalty","ground"):
        metrics=json.loads((root/"metrics.json").read_bytes())
        metrics["cost_failure" if corruption=="penalty" else "L_success"]+=1
        write_member(root,"metrics.json",metrics)
    elif corruption=="archive":
        records=json.loads((root/"failure_archive.json").read_bytes())
        records["lost_a"]["last_state"]["energy_j"]+=100.
        write_member(root,"failure_archive.json",records)
    else:
        states=[json.loads(line) for line in (root/"states.jsonl").read_text().splitlines()]
        if corruption=="energy":
            next(a for a in states[-1]["aircraft"] if a["id"]=="lost_a")["energy_j"]+=100.
        else:
            next(r for r in states[-1]["slots"] if r["aircraft_id"]=="lost_a")["end"]=None
        write_member(root,"states.jsonl",states)
    ReplayBundle.validate(root)
    report=audit_replay(root)
    assert report["status"]=="FAIL"
    assert code in {row["code"] for row in report["issues"]},report["issues"]
