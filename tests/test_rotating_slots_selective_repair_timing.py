"""Coalesced v2 lifecycle, compatibility, and executor boundary guards."""
import copy
import pytest
from terminal_marl.core.rotating_slots.world import RotatingSlotWorld
from terminal_marl.core.rotating_slots.models import ValidationResult
from terminal_marl.core.rotating_slots.planning_latency import LatencyAwareScheduler


class Callback:
    def __init__(self, empty=False): self.calls=[]; self.empty=empty
    def __call__(self,port):
        self.calls.append(port.build_planning_snapshot())
        if self.empty: port.proposal=()
    def state_dict(self): return {}
    def load_state_dict(self,state): pass


class Clock:
    def __init__(self,duration=.1): self.value=0.; self.open=False; self.duration=duration
    def __call__(self):
        if self.open: self.value+=self.duration
        self.open=not self.open
        return self.value


def setup(empty=False):
    w=RotatingSlotWorld(); w.seed_hold("a","outer",0,.9)
    cb=Callback(empty)
    s=LatencyAwareScheduler(cb,lead_seconds=20.,preparation_margin=1.,
        search_handoff_reserve_seconds=4.,clock=Clock(),restart_policy="coalesced_restart_v2")
    s(w._scheduler_port)
    return w,cb,s


@pytest.mark.parametrize("status",["COMMITTED","REJECTED","NO_EXECUTABLE_PREFIX"])
def test_finish_then_executor_restart_and_checkpoint_guard(status):
    w,cb,s=setup(); w.advance(1.,False)
    s(None,w._scheduler_port)  # A real subsequent trigger coalesces once.
    result=ValidationResult(True,"ACCEPTED","") if status=="COMMITTED" else None
    s.pending["proposal"]=()
    s._finish(w._scheduler_port,status,result=result)
    assert len(cb.calls)==1 and s.pending is None
    s.online_service(w.time,.2,.1,"boundary_public_work")
    state=copy.deepcopy(s.state_dict())
    restored=LatencyAwareScheduler(Callback(),lead_seconds=20.,search_handoff_reserve_seconds=4.,
        clock=Clock(),restart_policy="coalesced_restart_v2")
    restored.load_state_dict(state)
    assert restored.restart_coalesced(w._scheduler_port)
    assert restored.pending["timing"].snapshot_time==1.
    assert restored.pending["timing"].effective_time==21.
    assert restored.pending["due"]>=1.3-1.e-8
    assert restored.records[0]["finish_to_next_start_s"]==0.
    assert restored.records[1]["restart_reason"]=="COALESCED_AFTER_"+status
    assert not restored.restart_coalesced(w._scheduler_port)
    # Repeated zero-time finish/trigger cannot create a restart loop.
    restored.dirty=True; restored._finish(w._scheduler_port,"NO_EXECUTABLE_PREFIX")
    assert not restored.restart_coalesced(w._scheduler_port)
    roundtrip=restored.state_dict()
    again=LatencyAwareScheduler(Callback(),lead_seconds=20.,search_handoff_reserve_seconds=4.,restart_policy="coalesced_restart_v2")
    again.load_state_dict(roundtrip)
    assert not again.restart_coalesced(w._scheduler_port)


@pytest.mark.parametrize("empty",[False,True])
def test_world_boundary_continues_once_and_endpoint_is_truncated(empty):
    w,cb,s=setup(empty); s.dirty=True
    due=s.pending["due"]
    w.advance(due,s,settle_only_at_end=True)
    assert len(cb.calls)==1 and s.pending is not None
    w.advance(due+.15,s,settle_only_at_end=True)
    assert len(cb.calls)==2
    finished=s.records[0]["finished_time"]
    assert cb.calls[1].planning_timing.snapshot_time==finished
    assert s.records[0]["next_started_time"]==finished
    assert len([r for r in s.records if r["snapshot_time"]==finished])==1
    assert not any(e["type"]=="maneuver_started" for e in w.events)


def test_legacy_checkpoint_missing_restart_fields_stays_legacy():
    cb=Callback(); legacy=LatencyAwareScheduler(cb)
    state=legacy.state_dict()
    for key in ("restart_policy","restart_requested","last_normal_start_time","last_restart_time"): state.pop(key)
    loaded=LatencyAwareScheduler(Callback()); loaded.load_state_dict(state)
    assert loaded.restart_policy=="legacy" and loaded._last_restart_time is None
    with pytest.raises(ValueError,match="binding"):
        LatencyAwareScheduler(Callback(),restart_policy="coalesced_restart_v2").load_state_dict(state)


@pytest.mark.parametrize("origin",[0.,280.])
def test_real_final_publication_then_same_boundary_restart_sees_live_resources(origin):
    from terminal_marl.core.rotating_slots.trajectory import generate_final
    class FinalOnce(Callback):
        def __init__(self): super().__init__(); self.real_sources=[]; self.plan=None
        def __call__(self,port):
            self.calls.append(port.build_planning_snapshot()); self.real_sources.append(port._source)
            if self.plan is not None: return
            snapshot=port.build_planning_snapshot(); a=snapshot.aircraft["a"]
            self.plan=generate_final("a",a.ring,a.slot,0,snapshot.time,source_version=a.version,config=snapshot.config).plan
            assert self.plan is not None
            result=port.evaluate_submission((self.plan,))
            assert result.accepted and not result.refinement_required
            port.commit_validated_prefix(result,snapshot_version=snapshot.version)
    w=RotatingSlotWorld(); w.seed_hold("a","inner",0,.9)
    if origin: w.advance(origin,False)
    cb=FinalOnce(); s=LatencyAwareScheduler(cb,lead_seconds=20.,search_handoff_reserve_seconds=4.,
        clock=Clock(2.),restart_policy="coalesced_restart_v2")
    s(w._scheduler_port); effective=s.pending["timing"].effective_time; cutoff=origin+20.
    w.advance(cutoff,s,settle_only_at_end=True)
    assert cb.plan.start_time==effective and cb.plan.start_time==pytest.approx(cutoff) and "a" not in w.active
    assert not any(e["type"]=="maneuver_started" and e.get("aircraft_id")=="a" for e in w.events)
    assert s.pending is not None,[(r.get("snapshot_time"),r.get("reason")) for r in s.records]
    assert s.pending["stage"]=="PREPARED" and len(cb.calls)==1,s.records
    w.advance(cutoff+.05,s,settle_only_at_end=True)
    assert len(cb.calls)==2 and cb.real_sources[1].time==cutoff
    real=cb.real_sources[1]
    assert real.active["a"].trajectory_hash==cb.plan.trajectory_hash
    assert any(r.aircraft_id=="a" and r.committed for r in real.pads)
    starts=[e for e in w.events if e["type"]=="maneuver_started" and e.get("aircraft_id")=="a"]
    assert len(starts)==1 and starts[0]["time"]==cutoff
    assert s.records[0]["status"]=="COMMITTED" and s.records[0]["next_started_time"]==cutoff
