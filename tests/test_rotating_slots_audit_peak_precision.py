"""Independent peak envelopes tolerate numerical refinement, not ledger drift."""
from hashlib import sha256
import json

import pytest

from terminal_marl.core.rotating_slots.audit import audit_replay
from terminal_marl.core.rotating_slots.energy import integrate_energy
from terminal_marl.core.rotating_slots.replay import ReplayBundle
from terminal_marl.core.rotating_slots.trajectory import generate_final
from terminal_marl.core.rotating_slots.world import RotatingSlotWorld


def rewrite_states(root,states):
    payload=b"".join(json.dumps(row,sort_keys=True).encode()+b"\n" for row in states)
    (root/"states.jsonl").write_bytes(payload)
    index=json.loads((root/"replay.json").read_bytes())
    index["files"]["states.jsonl"]=sha256(payload).hexdigest()
    (root/"replay.json").write_text(json.dumps(index),encoding="utf-8")


@pytest.fixture
def refined_archive(tmp_path):
    world=RotatingSlotWorld(practical_runtime=True)
    world.seed_hold("refined","inner",0,.9)
    plan=generate_final("refined","inner",0,0,0.).plan
    assert plan is not None and world.submit((plan,)).accepted
    world.advance(130.1)
    old_peak=world.aircraft["refined"].peak_seconds
    world._refine_executed_energy("refined")
    correction=world.aircraft["refined"].peak_seconds-old_peak
    assert correction < -1.e-5
    world._record(force=True)
    world.advance(131.)
    path=ReplayBundle.write(tmp_path/"refined",world=world,code_commit="peak-precision-test").path
    return path,plan,correction


def test_actual_coarse_history_refinement_is_inside_existing_peak_envelope(refined_archive):
    path,plan,correction=refined_archive
    frames=[json.loads(line) for line in (path/"states.jsonl").read_text().splitlines()]
    old=next(f for f in frames if abs(f["time"]-130.)<1.e-7)
    new=next(f for f in frames if abs(f["time"]-130.1)<1.e-7)
    q=integrate_energy(plan,old["time"],new["time"],max_dt=.2)
    delta=new["aircraft"][0]["peak_seconds"]-old["aircraft"][0]["peak_seconds"]
    assert abs(delta-q.peak_seconds)>1.e-5
    assert delta-q.peak_seconds==pytest.approx(correction,abs=1.e-10)
    report=audit_replay(path)
    assert report["status"]=="PASS",report["issues"]
    assert report["peak_envelope_checks"]>=len(frames)
    assert abs(correction)<=report["peak_envelope_max_width_s"]<.02
    assert report["peak_step_mismatch_within_envelope_count"]>=1
    assert "not reconstructed" in report["peak_check_scope"]


@pytest.mark.parametrize("tamper",(-.02,.02))
def test_peak_change_outside_existing_envelope_is_rejected(refined_archive,tamper):
    path,_,_=refined_archive
    frames=[json.loads(line) for line in (path/"states.jsonl").read_text().splitlines()]
    for frame in frames:
        if frame["time"]>=120.: frame["aircraft"][0]["peak_seconds"]+=tamper
    rewrite_states(path,frames)
    ReplayBundle.validate(path)
    report=audit_replay(path)
    assert report["status"]=="FAIL"
    assert any(issue["code"]=="PEAK_DELTA" for issue in report["issues"])


def test_sub_tolerance_step_drift_cannot_accumulate_free_peak_budget(tmp_path):
    world=RotatingSlotWorld();world.seed_hold("drift","outer",0,.9);world.advance(5.)
    path=ReplayBundle.write(tmp_path/"drift",world=world,code_commit="peak-drift-test").path
    frames=[json.loads(line) for line in (path/"states.jsonl").read_text().splitlines()]
    for index,frame in enumerate(frames): frame["aircraft"][0]["peak_seconds"]+=index*5.e-6
    assert all(abs(b["aircraft"][0]["peak_seconds"]-a["aircraft"][0]["peak_seconds"])<1.e-5
               for a,b in zip(frames,frames[1:]))
    rewrite_states(path,frames)
    ReplayBundle.validate(path)
    report=audit_replay(path)
    assert report["status"]=="FAIL"
    assert any(issue["code"]=="PEAK_DELTA" for issue in report["issues"])


def test_ground_peak_ledger_stays_frozen(tmp_path):
    world=RotatingSlotWorld();world.seed_service("ground",0,remaining=2.);world.advance(.4)
    path=ReplayBundle.write(tmp_path/"ground",world=world,code_commit="peak-ground-test").path
    frames=[json.loads(line) for line in (path/"states.jsonl").read_text().splitlines()]
    frames[-1]["aircraft"][0]["peak_seconds"]+=.00002
    rewrite_states(path,frames)
    report=audit_replay(path)
    assert report["status"]=="FAIL"
    assert any(issue["code"]=="PEAK_DELTA" for issue in report["issues"])
