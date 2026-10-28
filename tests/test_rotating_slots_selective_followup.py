"""Bounded follow-up intervention, binding and output checks; no learned selector."""
from copy import deepcopy
from inspect import signature
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.stage3 import run_selective_reoptimization_a as base
from scripts.stage3 import run_selective_reoptimization_followup as followup
from scripts.stage3 import run_latency_interface as latency
from terminal_marl.core.rotating_slots.world import RotatingSlotWorld,STAGE2_POLICY_ID
from terminal_marl.core.rotating_slots.stage2_planning import CandidateProvider,EvaluationContext,NormalSchedulerCallback,PlanningResult,RollingHorizonCPSATScheduler
from terminal_marl.core.rotating_slots.selective_replanning import RuleSelector,PlanningHistory,build_global_frame
from terminal_marl.core.rotating_slots.planning_latency import PlanningTiming,LatencyAwareScheduler


def mixed_fixture():
    w=RotatingSlotWorld()
    for i in range(10):w.seed_hold(f"a_middle_{i:02d}","middle",i,.8)
    for i in range(8):w.seed_hold(f"z_outer_{i:02d}","outer",i,.99)
    snapshot=w.build_planning_snapshot();provider=CandidateProvider(planning_profile="downstream_service_v8")
    frame=build_global_frame(snapshot,provider,EvaluationContext(snapshot))
    history=PlanningHistory();history.observe(snapshot)
    for aid,entry in history.entries.items():entry.progressed=-10. if aid.startswith("a_") else -1.
    return w,snapshot,frame,history


def test_baseline_default_unchanged_and_outer_queue_increases_selected_outer():
    _,s,f,h=mixed_fixture()
    default,td=RuleSelector().select(s,f,h)
    explicit,tb=RuleSelector(variant="baseline").select(s,f,h)
    outer,to=RuleSelector(variant="outer_priority_v1").select(s,f,h)
    assert default==explicit and td==tb
    assert not default.mandatory_ids and not outer.mandatory_ids
    count=lambda scope:sum(s.aircraft[aid].ring=="outer" for aid in scope.replan_ids)
    assert count(outer)>count(default)
    assert outer.actual_k==default.actual_k==8
    assert any("OUTER_RING_QUEUE" in reasons for aid,reasons in outer.selection_reasons)
    assert to["energy_masks"]==tb["energy_masks"]
    assert all(to["eligible_ring_by_id"][aid]=="outer" for aid,reasons in outer.selection_reasons if "OUTER_RING_QUEUE" in reasons)


def test_outer_queue_mandatory_first_and_overflow_unchanged():
    _,s,f,h=mixed_fixture()
    h.entries["a_middle_00"].origin=-65.
    for variant in ("baseline","outer_priority_v1"):
        scope,_=RuleSelector(variant=variant).select(s,f,h)
        assert "a_middle_00" in scope.mandatory_ids and set(scope.mandatory_ids)<=set(scope.replan_ids)
    # More than twelve mandatory objects retain the existing explicit full fallback.
    for entry in h.entries.values():entry.origin=-65.
    left,tl=RuleSelector().select(s,f,h);right,tr=RuleSelector(variant="outer_priority_v1").select(s,f,h)
    assert left==right and len(right.mandatory_ids)==18
    assert tr["status"]==tl["status"]=="MANDATORY_OVERFLOW"
    assert right.status=="MANDATORY_OVERFLOW_FULL" and set(right.replan_ids)==set(right.eligible_ids)


def test_outer_queue_known_progress_age_before_unknown():
    _,s,f,h=mixed_fixture()
    h.entries["z_outer_07"].progressed=-100.
    unknown=h.entries["z_outer_00"];unknown.origin=None;unknown.progressed=None;unknown.review_origin_known=False
    scope,telemetry=RuleSelector(variant="outer_priority_v1").select(s,f,h)
    seeds=[aid for aid,reasons in scope.selection_reasons if "OUTER_RING_QUEUE" in reasons]
    assert seeds[0]=="z_outer_07" and "z_outer_00" not in seeds
    assert telemetry["ages"]["z_outer_00"]["progress_unknown"]


def test_variant_state_roundtrip_old_baseline_compatibility_and_mismatch():
    make=lambda variant="baseline":NormalSchedulerCallback("rh_cp_sat",planning_profile="downstream_service_v8",selective_method="RULE_LOCAL_K8",selective_rule_variant=variant)
    baseline=make();old=baseline.state_dict();old["selective_replanning"].pop("rule_variant")
    restored=make();restored.load_state_dict(old)
    assert restored.state_dict()==baseline.state_dict()
    outer=make("outer_priority_v1");outer.selective_history.entries=deepcopy(baseline.selective_history.entries)
    same=make("outer_priority_v1");same.load_state_dict(outer.state_dict())
    assert same.state_dict()==outer.state_dict()
    with pytest.raises(ValueError,match="rule variant binding mismatch"):make().load_state_dict(outer.state_dict())
    with pytest.raises(ValueError,match="requires selective method"):
        NormalSchedulerCallback("rh_cp_sat",selective_rule_variant="outer_priority_v1")
    with pytest.raises(ValueError,match="unknown selective rule variant"):RuleSelector(variant="unknown")


def test_exact_four_configurations_isolate_rule_and_lead_factors():
    rows={variant:followup.treatment(variant,15.) for variant in followup.VARIANTS}
    assert {(r["selector_variant"],r["lead_seconds"]) for r in rows.values()}=={
        ("baseline",20.),("outer_priority_v1",20.),("baseline",15.),("outer_priority_v1",15.)}
    shared={"selective_method","heartbeat_seconds","preparation_margin_seconds","search_handoff_reserve_seconds","nominal_k","max_k","learning","training","stage_B","stage_C"}
    assert all({k:r[k] for k in shared}=={k:rows["baseline20"][k] for k in shared} for r in rows.values())
    assert rows["baseline15"]["search_budget_seconds"]==rows["outer15"]["search_budget_seconds"]==10.
    with pytest.raises(ValueError,match="exactly 15"):followup.treatment("baseline15",10.)


@pytest.mark.parametrize("lead",[15.,20.])
def test_timing_construction_changes_future_time_not_physical_world(lead):
    w=RotatingSlotWorld();w.seed_hold("a","outer",0,.95)
    before=w.build_planning_snapshot()
    timing=PlanningTiming.from_lead(before.time,lead,1.)
    wrapper=LatencyAwareScheduler(NormalSchedulerCallback("rh_cp_sat",planning_profile="downstream_service_v8",selective_method="RULE_LOCAL_K8"),lead_seconds=lead,search_handoff_reserve_seconds=4.)
    assert timing.effective_time==lead and timing.ready_deadline==lead-1.
    assert wrapper.search_handoff_reserve_seconds==4. and w.build_planning_snapshot()==before


def test_followup_binding_uses_input_lineage_not_old_results(tmp_path):
    origin=dict(binding_id="old_A",inputs=dict(s3r23=dict(row=dict(seed=7324),run_checkpoint="original_t0",initial_checkpoint_sha256="original")),research_status="TRADEOFF_OBSERVED",model="MUST_NOT_IMPORT")
    source=dict(commit="new_followup_source",source_hash="new_hash",files={})
    args=SimpleNamespace(root="s3r23",variant="outer15",short_lead=15.,seconds=10,cpu=12,baseline_output=tmp_path/"readonly_A")
    saved=deepcopy(origin);bound=followup.build_binding(origin,source,args,{})
    assert origin==saved and bound["source_commit"]=="new_followup_source" and bound["origin_input_binding"]=="old_A"
    assert bound["schema"]=="rs_selective_followup_binding_v1" and bound["run_role"]=="SMOKE_NOT_FORMAL"
    assert "model" not in bound and "research_status" not in bound
    assert bound["treatment"]["selector_variant"]=="outer_priority_v1" and bound["treatment"]["lead_seconds"]==15.


def test_actual_parameterized_closed_job_ten_second_follow_output(tmp_path,monkeypatch):
    """Real restore/holding/checkpoint/replay; fixed FOLLOW selection, no CP."""
    from terminal_marl.core.rotating_slots.stage2_campaign import atomic_json,digest,save_checkpoint,restore_checkpoint,episode_metrics,shared_fifo_admission
    from terminal_marl.core.rotating_slots.stage3_learning import evaluation
    from terminal_marl.core.rotating_slots.replay import ReplayBundle
    class Provider(CandidateProvider):
        def iter_progressive_columns(self,snapshot,budget,context,*,scope=None):
            columns={aid:[self._column(snapshot,snapshot.aircraft[aid],(),"KEEP",context)] for aid in scope.replan_ids}
            yield self._progressive_catalog(snapshot,budget,context,columns,{aid:rows[0].reference for aid,rows in columns.items()},[snapshot.aircraft[aid] for aid in columns])
    def fixed(self,snapshot,catalog,context,budget,*,incumbent=None):
        rows=tuple(values[0] for values in catalog.columns.values());check,_=context.check_batch(rows)
        assert check.accepted
        return PlanningResult(rows,(),context.check_commit_prefix(rows),"FOLLOW_OUTPUT_FIXTURE",catalog.downstream.key(rows),certified=True)
    monkeypatch.setattr(RollingHorizonCPSATScheduler,"plan",fixed)
    seen=[]
    def callback(*args,**kwargs):
        seen.append(dict(kwargs))
        return NormalSchedulerCallback(*args,candidate_provider=Provider(planning_profile="downstream_service_v8"),**kwargs)
    w=RotatingSlotWorld(experiment_policy_id=STAGE2_POLICY_ID)
    w.seed_hold("a","outer",0,.95);w.seed_hold("b","middle",4,.95)
    initial=tmp_path/"input.json"
    identity=dict(base_config_hash=w.config.config_hash,manifest_sha256=digest(w.manifest.to_dict()),planning_profile="downstream_service_v8")
    save_checkpoint(initial,w,NormalSchedulerCallback("rh_cp_sat",planning_profile="downstream_service_v8"),identity)
    original=base.file_hash(initial)
    d=SimpleNamespace(NormalSchedulerCallback=callback,LatencyAwareScheduler=LatencyAwareScheduler,restore_checkpoint=restore_checkpoint,
        evaluation=evaluation,atomic_json=atomic_json,save_checkpoint=save_checkpoint,episode_metrics=episode_metrics,
        shared_fifo_admission=shared_fifo_admission,ReplayBundle=ReplayBundle,latency=latency)
    binding=dict(binding_id="FOLLOWUP_SMOKE",source_hash="new_source",source_commit="SMOKE_ONLY",inputs=dict(s3r23=dict(run_checkpoint=str(initial),initial_checkpoint_sha256=original,row=dict(identity=identity))))
    output=tmp_path/"followup"
    result=base.closed_job(output,"s3r23","RULE_LOCAL_K8",binding,d,lead_seconds=15.,selector_variant="outer_priority_v1",episode_seconds=10)
    directory=output/"evaluation/s3r23/RULE_LOCAL_K8"
    assert result["status"]=="COMPLETE" and result["completed_boundary_s"]==10
    assert seen[0]["selective_rule_variant"]=="outer_priority_v1"
    assert {p.name for p in directory.glob('boundary_*.json')}=={"boundary_0005.json","boundary_0010.json"}
    assert base.load(directory/"checkpoint_terminal.json")["payload"]["time"]==10.
    assert base.file_hash(initial)==original
    bundle=ReplayBundle.validate(directory/"replay");assert bundle.state_count>0
    observed=followup.saved_observables(directory,d,10)
    assert observed["selections"] and all(row["rule_variant"]=="outer_priority_v1" for row in observed["selections"])
    assert all(not row["unknown_ring_ids"] and row["eligible_by_ring"]=={"outer":1,"middle":1,"inner":0}
               for row in observed["selections"])
    assert signature(base.closed_job).parameters["episode_seconds"].default==300


def test_saved_clock_metrics_conditional_commits_and_scope_work_split(tmp_path):
    from terminal_marl.core.rotating_slots.stage2_campaign import atomic_json
    d=SimpleNamespace(latency=latency,atomic_json=atomic_json)
    scope=dict(replan_ids=["a"],eligible_ids=["a"],effective_time=15.,selection_reasons=[["a",["OUTER_RING_QUEUE"]]],status="LOCAL")
    callbacks=[dict(planning_timing=dict(snapshot_time=0.),selective_replanning=dict(scope=scope,rule_variant="outer_priority_v1",eligible_ring_by_id=dict(a="outer"))),
        dict(planning_timing=dict(snapshot_time=5.),selective_replanning=dict(scope=dict(scope,status="MANDATORY_OVERFLOW_FULL"),eligible_ring_by_id=dict(a="outer")))]
    tasks=[dict(snapshot_time=0.,status="COMMITTED",finished_time=15.,ready_time=10.,effective_time=15.,normal_work_s=2.),dict(snapshot_time=5.,status="COMPUTING",normal_work_s=3.)]
    for name,value in (("scheduler_records.json",callbacks),("normal_tasks.json",tasks),("online_services.json",[]),("replay/trajectories.json",dict(h=dict(kind="TRANSFER",source_ring="outer",target_ring="middle")))):atomic_json(tmp_path/name,value)
    (tmp_path/"replay/events.jsonl").write_text(json.dumps(dict(type="maneuver_started",time=3.,aircraft_id="a",details=dict(trajectory_hash="h")))+'\n',encoding="utf8")
    (tmp_path/"replay/states.jsonl").write_text(''.join(json.dumps(dict(time=t,aircraft=[dict(id="a",ring="outer",status="HOLD_OUTER")]))+'\n' for t in (0.,5.,10.)),encoding="utf8")
    result=followup.saved_observables(tmp_path,d,10)
    assert result["final_ready_wait_committed"]["count"]==1 and result["final_ready_wait_committed"]["mean_s"]==5.
    assert result["snapshot_start_intervals"]["total_s"]==5.
    assert result["complete_normal_work_by_scope_kind"]["LOCAL"]["total_s"]==2.
    assert result["complete_normal_work_by_scope_kind"]["OVERFLOW_FULL"]["total_s"]==3.
    assert result["boundaries"][1]["transition_starts"]["outer_to_middle"]==1
    assert result["selections"][0]["selected_by_ring"]["outer"]==1
