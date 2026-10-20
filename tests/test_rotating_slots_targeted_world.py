"""Targeted short-step screening and persistent whole-plan precision selection."""
from copy import deepcopy
from dataclasses import replace
import math
from types import SimpleNamespace

import pytest

from terminal_marl.core.rotating_slots import world as module
from terminal_marl.core.rotating_slots.energy import EnergyResult, integrate_energy
from terminal_marl.core.rotating_slots.models import ValidationResult
from terminal_marl.core.rotating_slots.resources import ResourceConflict
from terminal_marl.core.rotating_slots.trajectory import generate_final
from terminal_marl.core.rotating_slots.performance.checkpoint import (
    AdvanceCursor, CheckpointError, CheckpointIdentity, POLICY_ID, StateCodec,
    capture_checkpoint, restore_checkpoint,
)


@pytest.fixture(scope="module")
def final_plan():
    result = generate_final("a","inner",0,0,0.)
    assert result.status == "FEASIBLE"
    return result.plan


def world(soc=.8):
    w = module.RotatingSlotWorld()
    w.seed_hold("a","inner",0,soc)
    w.auto_emergency = False
    return w


def identity(w):
    c = w.config
    return CheckpointIdentity(POLICY_ID,c.config_hash,c.profile_hash,c.raw_hash,"rules","candidates","numeric","test")


def test_execution_screen_precedes_pair_bounds_and_preserves_minimum(monkeypatch):
    w = world()
    w.seed_hold("b","outer",5)
    seen = []
    safe = ValidationResult(True,"SAFE","",dict(lower_bound_m=100.,upper_bound_m=110.,witness_time=0.))
    def screen(plans,start,end,config):
        seen.append((set(plans),start,end))
        return {("a","b"):safe}
    monkeypatch.setattr(module,"_execution_distance_screen",screen)
    monkeypatch.setattr(module,"validate_separation",lambda *a,**k: pytest.fail("screened pair reached bounds checker"))
    w._check_execution(.2)
    assert seen == [({"a","b"},0.,.2)]
    assert w.minimum_separation == dict(lower_bound_m=100.,upper_bound_m=110.,time=0.,pair=["a","b"])


@pytest.mark.parametrize("end",[.2,1.])
def test_unscreened_near_or_future_pair_uses_original_checker(monkeypatch,end):
    w = world();w.seed_hold("b","outer",5)
    seen = []
    monkeypatch.setattr(module,"_execution_distance_screen",lambda *args: {})
    def validate(a,b,start,stop,**kwargs):
        seen.append((start,stop));return ValidationResult(True,"SAFE")
    monkeypatch.setattr(module,"validate_separation",validate)
    w._check_execution(end)
    assert seen == [(0.,end)]


def test_protection_search_keeps_conservative_endpoints_without_rounding_events(monkeypatch):
    w = world()
    plan = SimpleNamespace(start_time=.123456789,end_time=3.142857,source_ring="outer",source_slot=0,
                           target_ring="middle",target_slot=0)
    clear,capture = 1.0012345,2.0183746
    def validate(a,b,start,end,**kwargs):
        accepted = start >= clear if b.aircraft_id == "source_reference" else end <= capture
        return ValidationResult(accepted,"SAFE" if accepted else "UNSAFE")
    monkeypatch.setattr(module,"validate_separation",validate)
    monkeypatch.setattr(module,"_execution_distance_screen",lambda *args: pytest.fail("short screen used for future protection"))
    actual_clear,actual_capture = w._protect_times(plan)
    assert clear <= actual_clear <= clear+.01
    assert capture-.01 <= actual_capture <= capture
    assert plan.start_time == .123456789 and plan.end_time == 3.142857
    assert actual_clear != round(actual_clear,2)


def test_energy_cache_includes_locked_precision(monkeypatch,final_plan):
    w=world();seen=[]
    def energy(plan,start,end,max_dt,config,fine=False):
        seen.append((max_dt,fine));return EnergyResult(1.,1e-8,0.,True,error_j=.1)
    monkeypatch.setattr(module,"integrate_energy",energy)
    w._energy(final_plan,0.,.2);w._energy(final_plan,0.,.2)
    w._energy(final_plan,0.,.2,fine=True);w._energy(final_plan,0.,.2)
    assert seen == [(1.,False),(1.,True)]
    assert w._targeted_energy["fine_plans"][final_plan.trajectory_hash]


@pytest.mark.parametrize("soc",[.9,.05])
def test_remote_feasible_and_infeasible_budgets_do_not_refine(monkeypatch,final_plan,soc):
    w=world(soc);seen=[]
    def energy(plan,start,end,max_dt,config,fine=False):
        seen.append(fine);return EnergyResult(w._energy_capacity()*.1,.1,0.,True,error_j=1.)
    monkeypatch.setattr(module,"integrate_energy",energy)
    w._budget_energy("a",((final_plan,0.,final_plan.end_time),),reserve=True)
    assert seen == [False] and not w._targeted_energy["fine_plans"]


@pytest.mark.parametrize("boundary",["empty","reserve","peak"])
def test_whole_budget_near_threshold_selects_fine(monkeypatch,final_plan,boundary):
    w=world(.1 if boundary == "empty" else .35 if boundary == "reserve" else .8)
    if boundary == "peak":w.aircraft["a"]=replace(w.aircraft["a"],peak_seconds=119.8)
    seen=[]
    def energy(plan,start,end,max_dt,config,fine=False):
        seen.append((end-start,fine))
        peak=.19 if fine else .25
        return EnergyResult(w._energy_capacity()*.1,.1,peak if boundary == "peak" else 0.,True,
                            error_j=.001,peak_error_s=0. if fine else .1)
    monkeypatch.setattr(module,"integrate_energy",energy)
    q,=w._budget_energy("a",((final_plan,0.,final_plan.end_time),),reserve=boundary == "reserve")
    assert seen == [(final_plan.end_time,False),(final_plan.end_time,True)]
    assert w._targeted_energy["fine_plans"][final_plan.trajectory_hash]
    if boundary == "peak":assert w.aircraft["a"].peak_seconds+q.peak_seconds < 120


def test_executed_records_merge_and_refinement_corrects_history_before_next_step(monkeypatch,final_plan):
    w=world();w.active["a"]=final_plan
    w.trajectories[final_plan.trajectory_hash]=final_plan.to_dict()
    w.aircraft["a"]=replace(w.aircraft["a"],status="FINAL_NORMAL",trajectory_hash=final_plan.trajectory_hash,
                             normal_final_touchdown_soc=.5)
    def energy(plan,start,end,max_dt,config,fine=False):
        dt=end-start;j=(999. if fine else 1000.)*dt
        return EnergyResult(j,j/w._energy_capacity(),(.09 if fine else .1)*dt,True,
                            error_j=(.0001 if fine else 10.)*dt,peak_error_s=(0. if fine else .01)*dt)
    monkeypatch.setattr(module,"integrate_energy",energy)
    w._advance_energy(.2);w._advance_energy(.4)
    records=w._targeted_energy["executed"]["a"]
    assert len(records) == 1 and records[0]["charged_j"] == pytest.approx(400.)
    remaining=energy(final_plan,.4,final_plan.end_time,1.,w.config).soc_used
    w.aircraft["a"]=replace(w.aircraft["a"],soc=.25+remaining+1e-6)
    before=w.aircraft["a"]
    w._budget_energy("a",((final_plan,.4,final_plan.end_time),),reserve=True)
    a=w.aircraft["a"]
    assert w.time == .4
    assert a.energy_j == pytest.approx(before.energy_j-.4)
    assert a.soc == pytest.approx(before.soc+.4/w._energy_capacity())
    assert a.peak_seconds == pytest.approx(before.peak_seconds-.004)
    assert records[0]["fine"] and records[0]["error_j"] == pytest.approx(.00004)
    assert a.normal_final_touchdown_soc == pytest.approx(a.soc-energy(final_plan,.4,final_plan.end_time,1.,w.config,True).soc_used)
    w._advance_energy(.6)
    assert len(records) == 1
    assert records[0]["charged_j"] == pytest.approx(599.4)
    assert records[0]["error_j"] == pytest.approx(.00006)


def test_fine_selection_survives_later_resource_rejection(monkeypatch,final_plan):
    w=world(.35);before=(w.time,w.slots.reservations,w.pads.reservations)
    def energy(plan,start,end,max_dt,config,fine=False):
        soc=.099 if fine else .1
        return EnergyResult(soc*w._energy_capacity(),soc,0.,True,error_j=1.)
    monkeypatch.setattr(module,"integrate_energy",energy)
    monkeypatch.setattr(module,"validate_plan",lambda *a,**k: ValidationResult(True,"SAFE"))
    def conflict(*args):raise ResourceConflict("fixture")
    monkeypatch.setattr(w,"_calendar_stage",conflict)
    result=w.submit((final_plan,))
    assert result.reason == "RESOURCE_CONFLICT"
    assert w._targeted_energy["fine_plans"][final_plan.trajectory_hash]
    assert (w.time,w.slots.reservations,w.pads.reservations) == before
    assert not w.active and not w.soft


def test_energy_codec_accepts_only_known_appended_default():
    codec=StateCodec();encoded=codec.encode(EnergyResult(1.,.1,.2,True,error_j=.01))
    old=deepcopy(encoded);old["fields"].pop()
    assert codec.decode(old).peak_error_s == 0.
    corrupt=deepcopy(old);corrupt["fields"].pop()
    with pytest.raises(CheckpointError,match="fields"):codec.decode(corrupt)
    corrupt=deepcopy(encoded);corrupt["fields"].append(["unknown",0])
    with pytest.raises(CheckpointError,match="fields"):codec.decode(corrupt)


@pytest.mark.parametrize("mode",["cold","warm"])
def test_checkpoint_preserves_precision_charges_and_legacy_initial_metrics(final_plan,mode):
    w=world();w.time=.4
    h=final_plan.trajectory_hash;w.trajectories[h]=final_plan.to_dict()
    w._targeted_energy={"fine_plans":{h:True},"executed":{"a":[dict(trajectory_hash=h,start=0.,end=.4,
        charged_j=100.,charged_peak=.01,error_j=.001,peak_error_s=.0001,fine=True)]}}
    w._record()
    payload=capture_checkpoint(w,identity=identity(w),cursor=AdvanceCursor(1.,"loop_start","test"),driver=False)
    restored=restore_checkpoint(payload,expected_identity=identity(w),cache_mode=mode).world
    assert restored._targeted_energy == w._targeted_energy
    assert restored.metrics() == w.metrics()
    codec=StateCodec();optional=codec.decode(payload["optional_runtime"]);optional.pop("_targeted_energy")
    payload["optional_runtime"]=codec.encode(optional)
    legacy=restore_checkpoint(payload,expected_identity=identity(w),cache_mode=mode).world
    assert legacy._targeted_energy == {"fine_plans":{},"executed":{}}
    assert legacy.metrics() == w.metrics()


def test_actual_plan_coarse_execution_is_additive(final_plan):
    whole=integrate_energy(final_plan,0.,final_plan.end_time,max_dt=1.)
    mid=final_plan.end_time*.37
    parts=[integrate_energy(final_plan,lo,hi,max_dt=1.) for lo,hi in ((0.,mid),(mid,final_plan.end_time))]
    assert math.fsum(q.joules for q in parts) == pytest.approx(whole.joules,abs=1e-7)
    assert math.fsum(q.peak_seconds for q in parts) == pytest.approx(whole.peak_seconds,abs=1e-9)
    assert math.fsum(q.error_j for q in parts) == pytest.approx(whole.error_j,abs=1e-10)
    assert math.fsum(q.peak_error_s for q in parts) == pytest.approx(whole.peak_error_s,abs=1e-10)


def test_actual_coarse_reserve_boundary_is_decided_by_fine_budget(final_plan):
    coarse=integrate_energy(final_plan,0.,final_plan.end_time,max_dt=1.)
    fine=integrate_energy(final_plan,0.,final_plan.end_time,max_dt=1.,fine=True)
    soc=math.fsum((.25,coarse.soc_used))
    w=world(soc)
    result=w.submit((final_plan,))
    assert w._targeted_energy["fine_plans"][final_plan.trajectory_hash]
    assert result.accepted == (soc >= math.fsum((.25,fine.soc_used)))
    if not result.accepted:
        assert result.reason == "LANDING_RESERVE"
        assert result.details["required_soc"] == math.fsum((.25,fine.soc_used))
