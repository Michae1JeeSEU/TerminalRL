"""Bounded executor evidence for Stage2 T12/13 and T17--T21/T24/T26."""
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
import time

import pytest

from terminal_marl.core.rotating_slots.config import thaw
from terminal_marl.core.rotating_slots.demand import ArrivalManifest, ArrivalRequest
from terminal_marl.core.rotating_slots.energy import EnergyResult, integrate_energy
from terminal_marl.core.rotating_slots.trajectory import generate_final, generate_transfer
from terminal_marl.core.rotating_slots.world import (
    RotatingSlotWorld, STAGE2_POLICY_ID, evaluate_planning_snapshot,
)


@pytest.fixture(autouse=True)
def test_timing(request):
    start = datetime.now(timezone.utc).isoformat()
    clock = time.perf_counter()
    yield
    print(json.dumps(dict(test=request.node.nodeid, start_utc=start,
        end_utc=datetime.now(timezone.utc).isoformat(), wall_s=time.perf_counter()-clock)))


def world():
    w = RotatingSlotWorld(experiment_policy_id=STAGE2_POLICY_ID)
    w.auto_emergency = False
    w._update_emergency = lambda: False
    return w


def test_t17_isolated_battery_loss_other_aircraft_continues_once():
    w = world()
    w.seed_hold("lost", "outer", 0, soc=1e-6)
    w.seed_hold("other", "outer", 18)
    w.advance(1.)
    t = w.failure_archive["lost"]["time"]
    assert 0. < t < .2 and w.time == 1.
    assert w.aircraft["lost"].status == "ENERGY_MISSION_FAILED"
    assert w.aircraft["lost"].soc == 0.
    assert w.aircraft["other"].airborne_time == pytest.approx(1.)
    assert not w.failed and w.has_energy_losses and not w.terminated_invalid
    energy = w.aircraft["lost"].energy_j
    w.advance(2.)
    assert w.aircraft["lost"].energy_j == energy and w.costs["failure"] == 1000000.
    assert w.metrics()["F_battery"] == 1 and w.metrics()["conservation"]


def test_t18_simultaneous_losses_do_not_repeat_or_stall():
    w = world()
    for key, slot in (("a",0),("b",18)):
        w.seed_hold(key,"outer",slot,soc=1e-6)
    w.advance(1.)
    assert w.time == 1. and len(w.failure_archive) == 2
    assert w.failure_archive["a"]["time"] == w.failure_archive["b"]["time"]
    assert w.costs["failure"] == 2000000.
    assert len([e for e in w.events if e["type"] == "failure_isolated"]) == 2


def test_t19_releases_only_failed_resources_and_preserves_other_active():
    w = world()
    for key, slot in (("a",0),("b",18)):
        w.seed_hold(key,"outer",slot)
    pa = generate_transfer("a","outer",0,0,0.).plan
    pb = generate_transfer("b","outer",18,14,0.).plan
    assert pa and pb and w.submit((pa,pb)).accepted
    w.dependencies[pb.trajectory_hash] = (pa.trajectory_hash,)
    w.aircraft["a"] = replace(w.aircraft["a"],soc=1e-6)
    w.advance(.3)
    failure_time = w.failure_archive["a"]["time"]
    assert w.active["b"] is pb and "a" not in w.active and "a" not in w.soft
    assert w.aircraft["b"].trajectory_hash == pb.trajectory_hash
    assert w.dependencies[pb.trajectory_hash] == ()
    own = [r for r in w.slots.reservations if r.aircraft_id == "a"]
    assert own and all(r.end <= failure_time for r in own)
    assert any(r.aircraft_id == "b" and r.end > .3 for r in w.slots.reservations)
    assert w.failure_archive["a"]["active"]["trajectory_hash"] == pa.trajectory_hash


def test_t20_peak_loss_is_distinct_from_numerical_error(monkeypatch):
    w = world()
    w.seed_hold("peak")
    w.aircraft["peak"] = replace(w.aircraft["peak"],peak_seconds=119.95)
    energy = w._energy
    monkeypatch.setattr(w,"_energy",lambda p,a,b,**kw: replace(energy(p,a,b,**kw),peak_seconds=b-a))
    w.advance(.2)
    assert w.failure_archive["peak"]["reason"] == "PEAK_BUDGET_EXHAUSTED"
    assert w.metrics()["F_peak"] == 1 and w.metrics()["F_battery"] == 0
    bad = world()
    bad.seed_hold("bad")
    monkeypatch.setattr(bad,"_energy",lambda *a,**kw: EnergyResult(0.,0.,0.,False,reason="MODEL_BAD"))
    with pytest.raises(RuntimeError,match="ENERGY_MODEL_INVALID"):
        bad.advance(.2)
    assert bad.failed and bad.terminated_invalid and not bad.failure_archive
    assert bad.costs["failure"] == 0.


def test_t21_authoritative_trajectory_mismatch_is_invalid():
    w = world()
    w.seed_hold("a")
    p = generate_transfer("a","outer",0,0,0.).plan
    assert w.submit((p,)).accepted
    w.aircraft["a"] = replace(w.aircraft["a"],trajectory_hash="wrong")
    with pytest.raises(RuntimeError,match="AUTHORITATIVE_TRAJECTORY_MISMATCH"):
        w.advance(.2)
    assert w.terminated_invalid and not w.has_energy_losses and w.metrics()["F_energy_total"] == 0


def test_t21_executed_80m_violation_stops_invalid_without_energy_penalty():
    w = world()
    a = w.seed_hold("a")
    # Deliberately inject an unsafe executor fixture, bypassing seed validation.
    w.aircraft["b"] = replace(a,id="b")
    w.advance(.2)
    assert w.failed and w.terminated_invalid and not w.has_energy_losses
    assert not w.failure_archive and w.costs["failure"] == 0.
    assert any(e["type"] == "invalid_run_stopped" and
               e["details"]["reason"] == "EXECUTED_HARD_SEPARATION_VIOLATION" for e in w.events)


def test_t12_normal_final_exact_reserve_and_peak_budget_include_descent():
    w = world()
    p = generate_final("a","inner",0,0,0.).plan
    q = integrate_energy(p,0.,p.end_time,fine=True)
    initial_soc = math.fsum((.25,q.soc_used))
    w.seed_hold("a","inner",0,soc=initial_soc)
    w.aircraft["a"] = replace(w.aircraft["a"],peak_seconds=120.-q.peak_seconds)
    result = w.evaluate_submission((p,))
    assert result.accepted
    assert w.commit_validated_prefix(result).accepted
    assert w._targeted_energy["fine_plans"][p.trajectory_hash]
    w.advance(p.end_time)
    a = w.aircraft["a"]
    assert a.status == "LANDED_SERVICE" and a.soc == .25
    assert a.touchdown_time == p.end_time and a.descent_time == pytest.approx(13.5)
    assert a.peak_seconds == pytest.approx(120.,abs=1e-9)
    assert a.energy_j == pytest.approx(q.joules,abs=1e-5)
    assert not w.failure_archive and w.metrics()["L_normal"] == 1


def test_t13_detached_evaluation_order_leaves_real_ledger_and_resources_unchanged():
    w = world()
    w.seed_hold("a","inner",0)
    plans = [generate_final("a","inner",0,pad,0.).plan for pad in (0,1)]
    assert all(plans)
    w.aircraft["a"] = replace(w.aircraft["a"],soc=.25+integrate_energy(plans[0],0.,plans[0].end_time).soc_used)
    snapshot = w.build_planning_snapshot()
    before = (dict(w.aircraft),tuple(w.events),tuple(w.rejections),w.slots.reservations,
              w.pads.reservations,thaw(snapshot.energy_ledger),dict(w.trajectories),dict(w.cpu))
    forward = [evaluate_planning_snapshot(snapshot,(p,)) for p in plans]
    backward = [w.evaluate_submission((p,),snapshot=snapshot) for p in reversed(plans)]
    assert [(r.accepted,r.reason) for r in forward] == [(r.accepted,r.reason) for r in reversed(backward)]
    after = (dict(w.aircraft),tuple(w.events),tuple(w.rejections),w.slots.reservations,
             w.pads.reservations,thaw(w.build_planning_snapshot().energy_ledger),dict(w.trajectories),dict(w.cpu))
    assert before == after
    assert not hasattr(snapshot,"manifest") and not hasattr(snapshot,"world")
    with pytest.raises(TypeError):
        snapshot.aircraft["other"] = w.aircraft["a"]


def test_atomic_commit_rejects_stale_evaluation_and_keeps_other_plans():
    w = world()
    w.seed_hold("a")
    p = generate_transfer("a","outer",0,0,0.).plan
    result = w.evaluate_submission((p,))
    assert result.accepted and not w.active
    w.aircraft["a"] = replace(w.aircraft["a"],soc=.74)
    assert w.commit_validated_prefix(result).reason == "STALE_PLANNING_SNAPSHOT"
    result = w.evaluate_submission((p,))
    assert w.commit_validated_prefix(result).accepted and w.active["a"] is p
    assert not w.commit_validated_prefix(result).accepted


def test_t24_cutoff_settles_without_new_decisions_and_resume_dispatches_once():
    w = world()
    calls = []
    w.seed_service("ground",remaining=1.)
    w.advance(1.,lambda snapshot,port: calls.append(snapshot.time))
    assert calls == [0.] and w.aircraft["ground"].status == "CLEARED"
    assert w.metrics()["L_success"] == 0 and w.metrics()["N_offered"] == 0
    w.advance(1.1,lambda snapshot,port: calls.append(snapshot.time))
    assert calls == [0.,1.]
    assert w.metrics()["conservation"]


def test_t26_dispatch_measurement_includes_monitor_rescue_admission_and_scheduler(monkeypatch):
    w = world()
    w.auto_emergency = True
    order = []
    def phase(label):
        order.append(label)
        time.sleep(.002)
    monkeypatch.setattr(w,"_update_emergency",lambda: phase("monitor"))
    monkeypatch.setattr(w,"handle_emergencies",lambda: phase("rescue"))
    w.admission_callback = lambda snapshot,port: phase("admission")
    w.advance(.1,lambda snapshot,port: phase("normal"))
    row = w.dispatch_timings[0]
    assert order == ["monitor","rescue","admission","normal"]
    assert row["emergency_wall_s"] >= .004
    assert row["admission_wall_s"] >= .002 and row["normal_wall_s"] >= .002
    assert row["dispatch_wall_s"] >= sum(row[k] for k in ("emergency_wall_s","admission_wall_s","normal_wall_s"))


def test_t14_planning_snapshot_does_not_depend_on_hidden_manifest_suffix():
    a, b = world(), world()
    b.manifest = ArrivalManifest((ArrivalRequest("future",100.,0,.8,"short"),))
    assert a.build_planning_snapshot() == b.build_planning_snapshot()
    snapshot=a.build_planning_snapshot()
    assert snapshot.emergency_monitor["enabled"] is True
    with pytest.raises(TypeError):
        snapshot.emergency_monitor["imminent"]["x"] = {}
    a._practical_runtime["imminent"]["x"] = dict(time=5.,ring="outer",slot=0,version=0)
    assert "x" not in snapshot.emergency_monitor["imminent"]
    assert a.build_planning_snapshot().version != snapshot.version
