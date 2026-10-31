"""A4 executed common boundary, exact clone and cold scheduler evidence."""
from copy import deepcopy
from contextlib import contextmanager
import os
from types import SimpleNamespace

import pytest

from scripts.stage3 import run_selective_common_initial_a4 as runner
from terminal_marl.core.rotating_slots import stage2_campaign as campaign
from terminal_marl.core.rotating_slots.planning_latency import LatencyAwareScheduler, LOCAL_TRAFFIC_POLICY
from terminal_marl.core.rotating_slots.stage2_planning import NormalSchedulerCallback
from terminal_marl.core.rotating_slots.trajectory import generate_final
from terminal_marl.core.rotating_slots.world import RotatingSlotWorld, STAGE2_POLICY_ID


class Clock:
    def __init__(self): self.value=0.; self.open=False
    def __call__(self):
        if self.open: self.value+=2.
        self.open=not self.open
        return self.value


class FinalOnce:
    def __init__(self): self.calls=[]; self.plan=None
    def __call__(self,port):
        self.calls.append(port.build_planning_snapshot())
        if self.plan is not None: return
        snapshot=port.build_planning_snapshot(); a=snapshot.aircraft["a"]
        self.plan=generate_final("a",a.ring,a.slot,0,snapshot.time,source_version=a.version,
                                 config=snapshot.config).plan
        assert self.plan is not None
        result=port.evaluate_submission((self.plan,))
        assert result.accepted and not result.refinement_required
        port.commit_validated_prefix(result,snapshot_version=snapshot.version)


def live_world():
    world=RotatingSlotWorld(); world.seed_hold("a","inner",0,.9)
    cb=FinalOnce()
    wrapper=LatencyAwareScheduler(cb,lead_seconds=20.,preparation_margin=1.,
        search_handoff_reserve_seconds=4.,clock=Clock(),restart_policy=runner.RESTART)
    return world,cb,wrapper


def test_hook_stops_after_activation_bookkeeping_before_second_normal():
    world,callback,wrapper=live_world()
    cuts=[]
    def capture(w,s):
        cuts.append((w.time,s.pending,len(s.records),"a" in w.active))
        return runner.clean_boundary(w,s)
    reached=world.advance(30.,wrapper,decision_boundary_hook=capture)
    assert reached==world.time==20.
    assert cuts[-1]==(20.,None,1,True)
    assert len(callback.calls)==1 and wrapper.pending is None
    assert wrapper._restart_requested is not None
    assert any(e["type"]=="maneuver_started" and e["time"]==20. for e in world.events)
    assert world.records[-1]["time"]==20.
    assert world._initial_settled


def test_default_advance_still_restarts_and_reaches_target():
    world,callback,wrapper=live_world()
    result=world.advance(20.05,wrapper)
    assert world.time==20.05 and result is not None
    assert len(callback.calls)==2 and wrapper.pending is not None
    assert wrapper.records[0]["next_started_time"]==20.


def test_hook_is_not_called_while_normal_task_pending():
    world,callback,wrapper=live_world()
    calls=[]
    world.advance(10.,wrapper,decision_boundary_hook=lambda *args:calls.append(args) or True)
    assert not calls and wrapper.pending is not None and len(callback.calls)==1


def test_stage2_hook_finishes_dispatch_accounting():
    world=RotatingSlotWorld(experiment_policy_id=STAGE2_POLICY_ID)
    world.auto_emergency=False; world.seed_hold("a","inner",0,.9)
    cb=FinalOnce(); wrapper=LatencyAwareScheduler(cb,lead_seconds=20.,preparation_margin=1.,
        search_handoff_reserve_seconds=4.,clock=Clock(),restart_policy=runner.RESTART)
    world.advance(30.,wrapper,decision_boundary_hook=runner.clean_boundary)
    assert world.time==20. and not world._pending_dispatch
    assert world._monitor_timing_pending==(0.,0.)
    assert world.dispatch_timings[-1]["time"]==20.
    assert world.cpu["decision"]>0. and len(cb.calls)==1


def dependencies():
    return SimpleNamespace(NormalSchedulerCallback=NormalSchedulerCallback,
        LatencyAwareScheduler=LatencyAwareScheduler,restore_checkpoint=campaign.restore_checkpoint,
        save_checkpoint=campaign.save_checkpoint,atomic_json=campaign.atomic_json,
        shared_fifo_admission=campaign.shared_fifo_admission)


def common_checkpoint(tmp_path):
    world=RotatingSlotWorld(experiment_policy_id=STAGE2_POLICY_ID)
    world.auto_emergency=False; world.seed_hold("a","inner",0,.9)
    a=world.aircraft["a"]
    plan=generate_final("a",a.ring,a.slot,0,0.,source_version=a.version,config=world.config).plan
    assert plan is not None and world.submit((plan,)).accepted
    world.advance(3.,False,settle_only_at_end=True)
    callback=runner.common_callback(dependencies(),"RULE_LOCAL_K8")
    callback.initialize_imported_history(world.build_planning_snapshot(),events=world.events)
    callback.records.append({"initializer_only":True})
    callback.selective_blockers=[{"initializer_only":"blocker"}]
    wrapper=LatencyAwareScheduler(callback,lead_seconds=20.,preparation_margin=1.,
        search_handoff_reserve_seconds=4.,restart_policy=runner.RESTART)
    wrapper.records.append(dict(snapshot_time=0.,finished_time=3.,status="COMMITTED",submitted=1))
    wrapper.dirty=True; wrapper._service_until=3.7; wrapper._idle_service=(3.,.2)
    wrapper._last_normal_start_time=0.; wrapper._last_restart_time=None
    wrapper._restart_requested=dict(record_index=0,finished_time=3.,reason="COALESCED_AFTER_COMMITTED")
    summary=dict(status="COMMON_INITIAL_READY",t0=3.,baseline=runner.baseline_observation(world),
                 inherited_finals=runner.inherited_finals(world))
    identity=campaign.checkpoint_identity(dict(source_hash="test",base_config_hash=world.config.config_hash,
        overlay_sha256="test",planning_profile="downstream_service_v8"),campaign.digest(world.manifest.to_dict()))
    path=tmp_path/"common.json"
    campaign.save_checkpoint(path,world,wrapper,identity,boundary_summary=summary)
    return path,world,wrapper,summary,plan


def serialized_world(world):
    fields=set(vars(world))-campaign.RUNTIME_FIELDS-campaign.RUNTIME_EXTRA-campaign.DERIVED_RESCUE_FIELDS
    return campaign.Stage2Codec().encode({key:vars(world)[key] for key in sorted(fields)})


def rewrite_common_state(path,*,pending=None,packets=None):
    """Build historical/invalid fixtures without advancing or publishing physics."""
    envelope=runner.base.load(path); payload=envelope["payload"]; codec=campaign.Stage2Codec()
    saved=codec.decode(payload["scheduler"])
    saved.pop("traffic_policy",None)
    saved.pop("delivery_policy",None)
    saved["pending"]=pending
    payload["scheduler"]=codec.encode(saved)
    if packets is not None:
        world=codec.decode(payload["world"]); world["_planning_latency_packets"]=packets
        payload["world"]=codec.encode(world)
    envelope["sha256"]=campaign.digest(payload)
    campaign.atomic_json(path,envelope)
    return saved


def test_legacy_packet_free_common_handoff_binds_local_treatment_without_clock_or_physics_change(tmp_path):
    path,original,initializer,_,_=common_checkpoint(tmp_path)
    saved=rewrite_common_state(path)
    before=path.read_bytes()
    world,callback,wrapper,metadata=runner.restore_common(path,"FULL_H",evaluation_profile="service_opportunity_v1",d=dependencies())
    assert path.read_bytes()==before and serialized_world(world)==serialized_world(original)
    assert wrapper.traffic_policy==LOCAL_TRAFFIC_POLICY
    assert metadata["initializer_traffic_policy"]=="legacy" and metadata["method_traffic_policy"]==LOCAL_TRAFFIC_POLICY
    assert metadata["initializer_delivery_policy"] is None and metadata["initializer_delivery_policy_field_present"] is False
    if hasattr(wrapper,"delivery_policy"):
        assert metadata["method_delivery_policy"]==wrapper.delivery_policy=="absolute_first_action_reservation_v1"
        assert wrapper.state_dict()["delivery_policy"]==wrapper.delivery_policy
    assert callback.selective_history.state_dict()==initializer.callback.selective_history.state_dict()
    resumed=wrapper.state_dict()
    for key in ("idle_service","handoff_service","service_until","dirty","last_normal_start_time","last_restart_time"):
        assert resumed[key]==saved[key]
    assert world.time==3. and wrapper.pending is None and not getattr(world,"_planning_latency_packets",{})
    assert wrapper.records==wrapper.online_services==callback.records==[]


@pytest.mark.parametrize("pending,packets,message",[
    (dict(stage="RESULT"),None,"pending normal work"),
    (None,{"issued_packet":{}},"issued latency packets")])
def test_common_policy_handoff_rejects_pending_work_or_any_issued_packet(tmp_path,pending,packets,message):
    path,_,_,_,_=common_checkpoint(tmp_path)
    rewrite_common_state(path,pending=pending,packets=packets)
    before=path.read_bytes()
    with pytest.raises(ValueError,match=message):
        runner.restore_common(path,"FULL_H",evaluation_profile="service_opportunity_v1",d=dependencies())
    assert path.read_bytes()==before


@pytest.mark.parametrize("pending",[None,dict(stage="RESULT")])
def test_generic_latency_restore_keeps_strict_legacy_to_local_policy_binding(tmp_path,pending):
    path,_,_,_,_=common_checkpoint(tmp_path)
    saved=rewrite_common_state(path,pending=pending)
    callback=runner.common_callback(dependencies(),"RULE_LOCAL_K8",evaluation_profile="service_opportunity_v1")
    wrapper=LatencyAwareScheduler(callback,lead_seconds=20.,preparation_margin=1.,
        search_handoff_reserve_seconds=4.,restart_policy=runner.RESTART)
    with pytest.raises(ValueError,match="traffic policy binding mismatch"):
        wrapper.load_state_dict(saved)
    assert wrapper.pending is None and wrapper.records==[]


def test_two_clones_exact_physics_history_and_independent_cold_caches(tmp_path):
    path,original,old_wrapper,summary,plan=common_checkpoint(tmp_path)
    a,ca,wa,ma=runner.restore_common(path,"FULL_H",d=dependencies())
    b,cb,wb,mb=runner.restore_common(path,"RULE_LOCAL_K8",d=dependencies())
    assert serialized_world(a)==serialized_world(b)==serialized_world(original)
    assert a.aircraft["a"].soc<.9 and a.aircraft["a"].peak_seconds==original.aircraft["a"].peak_seconds
    assert a.active==b.active==original.active and a.soft==b.soft==original.soft
    assert a.pads.reservations==b.pads.reservations==original.pads.reservations
    assert a.active["a"].trajectory_hash==plan.trajectory_hash
    assert ca.selective_history.state_dict()==cb.selective_history.state_dict()==old_wrapper.callback.selective_history.state_dict()
    assert ca.selective_history is not cb.selective_history
    for callback,wrapper,metadata in ((ca,wa,ma),(cb,wb,mb)):
        assert callback.records==wrapper.records==wrapper.online_services==[]
        assert callback.selective_blockers==[] and metadata["initializer_blockers"]==[{"initializer_only":"blocker"}]
        assert callback.global_frame is None and callback.provider.previous=={}
        assert callback.provider.references.cache=={} and callback.provider.references.templates=={}
        assert wrapper._service_until==3.7 and wrapper._idle_service==(3.,.2)
        assert wrapper.dirty and wrapper._last_normal_start_time==0.
        assert wrapper._restart_requested["record_index"] is None
        assert wrapper._restart_requested["external_predecessor"]["initializer_record_index"]==0
        assert metadata["horizon"]==[3.,303.]
        assert metadata["initializer_tasks"]==old_wrapper.records
        assert metadata["initializer_callback_records"]==old_wrapper.callback.records
        assert all(row.get("currsize",row.get("entries"))==0 for row in metadata["cache_reset"]["after"].values())
    assert ca.provider is not cb.provider and ca.provider.references is not cb.provider.references
    ca.provider.previous["sentinel"]=()
    assert "sentinel" not in cb.provider.previous and not old_wrapper.callback.provider.previous
    assert wa._restart_requested is not wb._restart_requested
    ma["initializer_callback_records"][0]["initializer_only"]="changed_copy"
    assert mb["initializer_callback_records"]==old_wrapper.callback.records==[{"initializer_only":True}]


def test_increment_metrics_preserve_ledger_and_attribute_inherited_touchdown(tmp_path):
    path,original,_,summary,plan=common_checkpoint(tmp_path)
    world,callback,wrapper,metadata=runner.restore_common(path,"FULL_H",d=dependencies())
    initial_energy=world.aircraft["a"].energy_j
    assert initial_energy>0.
    zero=runner.increment_observation(world,summary)
    assert zero["elapsed_s"]==0. and zero["cumulative_delta"]["energy_j"]==0.
    world.advance(plan.end_time,False,settle_only_at_end=True)
    delta=runner.increment_observation(world,summary)
    assert delta["inherited_final_touchdown_ids"]==["a"]
    assert delta["treatment_final_touchdown_ids"]==[] and delta["cumulative_delta"]["L_success"]==1
    assert delta["energy_by_aircraft"]["a"]==pytest.approx(world.aircraft["a"].energy_j-initial_energy)
    assert world.aircraft["a"].energy_j>initial_energy


def test_increment_landing_failure_counts_and_pending_are_distinct(monkeypatch):
    initial=dict(L_success=2,L_normal=1,L_emergency=1,failed_before_landing=4,
                 N_airborne_pending=7,N_upstream_pending=5,airborne_or_censored=7)
    current=dict(L_success=5,L_normal=2,L_emergency=3,failed_before_landing=6,
                 N_airborne_pending=3,N_upstream_pending=2,airborne_or_censored=3)
    monkeypatch.setattr(campaign,"episode_metrics",lambda world:dict(current))
    summary=dict(t0=3.,inherited_finals=[],baseline=dict(metrics=initial,
        touchdown_ids=[],event_count=2,frame_count=4,dispatch_count=1,energy_by_aircraft={}))
    world=SimpleNamespace(time=13.,aircraft={},events=[{}]*5,_record_count=8,dispatch_timings=[{}]*3)
    observed=runner.increment_observation(world,summary)
    assert observed["cumulative_delta"]==dict(L_success=3,L_normal=1,L_emergency=2,failed_before_landing=2)
    assert observed["ending_pending"]==dict(N_airborne_pending=3,N_upstream_pending=2,airborne_or_censored=3)
    assert not set(observed["ending_pending"]) & set(observed["cumulative_delta"])


def test_initialization_freeze_uses_independent_shared_ledger_key(tmp_path,monkeypatch):
    from scripts.stage3 import run_selective_quality_gap_a4 as a4
    from scripts.stage3 import run_selective_reoptimization_repair as old
    checkpoint,_,_,_,_=common_checkpoint(tmp_path)
    source=dict(source_hash="new_common_freeze",commit="new_common_commit",
        files={"scripts/stage3/run_selective_common_initial_a4.py":"test"})
    freeze=tmp_path/"source.json"; campaign.atomic_json(freeze,source)
    origin=tmp_path/"origin"; origin.mkdir()
    binding=dict(inputs={"s3r23":dict(run_checkpoint=str(checkpoint),
        initial_checkpoint_sha256=runner.base.file_hash(checkpoint),
        row=dict(seed=7324,scenario_profile="medium24_rate150_v1"))})
    binding["binding_id"]=runner.base.digest(binding)
    campaign.atomic_json(origin/"resolved_run_binding.json",binding)
    ledger=dict(a4_source_hash="old_diagnostic_freeze",jobs={"common_initial:s3r23":dict(
        status="running",parent_pid=os.getppid())})
    @contextmanager
    def transaction(*args,**kwargs): yield ledger
    monkeypatch.setattr(a4,"ledger_transaction",transaction)
    monkeypatch.setattr(a4,"validated_output",lambda path:path)
    monkeypatch.setattr(runner.base,"verify_sources",lambda source:None)
    monkeypatch.setattr(runner,"_dependencies",lambda d:dependencies())
    monkeypatch.setattr(old,"set_cpu",lambda args,d:None)
    monkeypatch.setattr(runner,"initialize_common",lambda *args,**kwargs:dict(status="TEST_BOUND_ONLY"))
    result=runner.main(["--stage","initialize-child","--output",str(tmp_path/"output"),
        "--origin-output",str(origin),"--source-freeze",str(freeze),"--root","s3r23","--cpu","12"])
    assert result==dict(status="TEST_BOUND_ONLY")
    assert ledger["a4_source_hash"]=="old_diagnostic_freeze"
    assert ledger["common_initial_source_hash"]=="new_common_freeze"


@pytest.mark.parametrize("external",[True,False])
def test_external_restart_index_requires_provenance_and_never_rewrites_new_record(external):
    world=RotatingSlotWorld()
    class NoProposal:
        def __call__(self,port): pass
    wrapper=LatencyAwareScheduler(NoProposal(),lead_seconds=20.,clock=Clock(),restart_policy=runner.RESTART)
    wrapper.dirty=True
    wrapper._restart_requested=dict(record_index=None,finished_time=0.,reason="COMMON")
    if external:
        wrapper._restart_requested["external_predecessor"]={"common_checkpoint_sha256":"test"}
        assert wrapper.restart_coalesced(world._scheduler_port)
        assert len(wrapper.records)==1
        record=wrapper.records[0]
        assert record["previous_record_index"] is None and record["external_predecessor"]["common_checkpoint_sha256"]=="test"
        assert "next_started_time" not in record
    else:
        with pytest.raises(ValueError,match="provenance"):
            wrapper.restart_coalesced(world._scheduler_port)
        assert wrapper.records==[] and wrapper._restart_requested is not None


def test_clean_boundary_rejects_missed_commit_and_due_soft():
    world,callback,wrapper=live_world()
    wrapper.records=[dict(status="COMMITTED",submitted=1,snapshot_time=0.,finished_time=0.)]*2
    with pytest.raises(ValueError,match="first nonempty"):
        runner.clean_boundary(world,wrapper)


def test_evaluation_profile_placeholder_fails_closed():
    with pytest.raises(ValueError,match="not bound"):
        runner.common_callback(dependencies(),"FULL_H",evaluation_profile="unimplemented")


def test_resume_common_starts_once_at_t0_and_keeps_queued_service(tmp_path):
    path,_,_,_,_=common_checkpoint(tmp_path)
    world,callback,wrapper,metadata=runner.restore_common(path,"FULL_H",d=dependencies())
    class NoProposal:
        def __call__(self,port): pass
    wrapper.callback=NoProposal(); wrapper.clock=Clock()
    record=runner.resume_common(world,wrapper)
    assert world.time==record["snapshot_time"]==3.
    assert record["effective_time"]==23. and record["online_service_s"]==pytest.approx(.7)
    assert wrapper.pending["due"]==pytest.approx(5.7)
    assert record["previous_record_index"] is None
    with pytest.raises(ValueError,match="exactly once"):
        runner.resume_common(world,wrapper)


@pytest.mark.parametrize("deliver",[True,False])
def test_initializer_saves_full_clean_checkpoint_or_explicit_not_reached(tmp_path,deliver):
    d=dependencies()
    class ManagedFinal(NormalSchedulerCallback):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs); self.calls=[]; self.plan=None
        def __call__(self,port):
            if deliver: return FinalOnce.__call__(self,port)
            self.calls.append(port.build_planning_snapshot())
    world=RotatingSlotWorld(experiment_policy_id=STAGE2_POLICY_ID)
    world.auto_emergency=False; world.seed_hold("a","inner",0,.9)
    callback=runner.common_callback(d,"RULE_LOCAL_K8")
    initial_wrapper=LatencyAwareScheduler(callback,lead_seconds=20.,preparation_margin=1.,
        search_handoff_reserve_seconds=4.,restart_policy=runner.RESTART)
    identity=campaign.checkpoint_identity(dict(source_hash="test",base_config_hash=world.config.config_hash,
        overlay_sha256="test",planning_profile="downstream_service_v8"),campaign.digest(world.manifest.to_dict()))
    path=tmp_path/"original.json"
    campaign.save_checkpoint(path,world,initial_wrapper,identity)
    original_bytes=path.read_bytes()
    d.NormalSchedulerCallback=ManagedFinal
    d.LatencyAwareScheduler=lambda cb,**kwargs:LatencyAwareScheduler(cb,clock=Clock(),**kwargs)
    directory=tmp_path/"initializer"
    summary=runner.initialize_common(path,directory,d=d)
    assert path.read_bytes()==original_bytes
    assert summary["baseline"]["metrics"]["energy_j"]>0.
    assert (directory/"boundary_summary.json").is_file()
    if deliver:
        assert summary["pending"] is False
        assert summary["status"]=="COMMON_INITIAL_READY" and summary["t0"]==20.
        assert summary["normal_task_count"]==1 and len(summary["inherited_finals"])==1
        _,clone_callback,clone_wrapper,metadata=runner.restore_common(directory/"common.checkpoint.json","FULL_H",d=dependencies())
        assert clone_callback.records==clone_wrapper.records==[] and metadata["horizon"]==[20.,320.]
    else:
        assert summary["status"]=="INIT_NOT_REACHED" and summary["t0"]==60.
        assert not (directory/"common.checkpoint.json").exists()
        assert (directory/"not_reached.checkpoint.json").is_file()
