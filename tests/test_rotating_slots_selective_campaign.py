"""Stage A input, accounting and saved-artifact gates (not whole-contract acceptance)."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts.stage3 import run_selective_reoptimization_a as runner
from scripts.stage3 import report_selective_reoptimization_a as report
from scripts.stage3 import run_latency_interface as latency


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding="utf8")


def test_binding_digest_survives_real_atomic_json_cpu_keys_and_unicode(tmp_path):
    from terminal_marl.core.rotating_slots.stage2_campaign import atomic_json
    binding=dict(environment=dict(cpu_load_percent={2:1.5,10:2.5,12:3.5},description="服务器 CPU 测量"),
                 source=dict(files={"source.py":"源码哈希"}))
    binding["binding_id"]=runner.digest(binding)
    atomic_json(tmp_path/"resolved_run_binding.json",binding)
    restored=runner.load(tmp_path/"resolved_run_binding.json")
    assert set(restored["environment"]["cpu_load_percent"])=={"2","10","12"}
    assert restored["binding_id"]==runner.digest({k:v for k,v in restored.items() if k!="binding_id"})
    restored["environment"]["cpu_load_percent"]["10"]+=1.
    atomic_json(tmp_path/"resolved_run_binding.json",restored)
    with pytest.raises(ValueError,match="run binding changed"):
        runner.verify(tmp_path,SimpleNamespace())


def test_digest_preserves_existing_all_string_source_hash():
    source={"terminal_marl/source.py":"abc123","scripts/工具.py":"源码"}
    old=sha256(json.dumps(source,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    assert runner.digest(source)==old


def test_terminal_result_prints_nested_real_sections_and_rejects_nonfinite(capsys):
    from terminal_marl.core.rotating_slots.config import Section
    value=dict(stage="job",status="COMPLETE",rows=[dict(configuration=Section(dict(physical=Section(dict(separation_m=80.)),label="实际配置")))])
    runner.print_result(value)
    observed=json.loads(capsys.readouterr().out)
    assert observed==dict(stage="job",status="COMPLETE",rows=[dict(configuration=dict(physical=dict(separation_m=80.),label="实际配置"))])
    with pytest.raises(ValueError):
        runner.print_result(dict(status="COMPLETE",configuration=Section(dict(invalid=float("nan")))))
    assert capsys.readouterr().out==""


def test_A_contract_actually_parsed_and_B_C_rejected(tmp_path):
    parsed=runner.parse_contract()
    assert parsed["phase"]["settings"]["methods"]==list(runner.METHODS)
    for phase in ("B","C"):
        with pytest.raises(ValueError,match="only Stage A"):
            runner.parse_contract(runner.PACKAGE/f"phase_{phase}_v1_0.yaml")


@pytest.mark.parametrize("section,key,value",[("timing","lead_s",10.),("preserve","minimum_separation_m",79.),("search","cp_total_cap_s",9.),("execution","cp_workers",2),("phase_A","learned_selector",True)])
def test_changed_mandatory_controls_are_not_ignored(tmp_path,section,key,value):
    phase=yaml.safe_load((runner.PACKAGE/"phase_A_v1_0.yaml").read_text(encoding="utf8"))
    parent=yaml.safe_load((runner.PACKAGE/phase["extends"]).read_text(encoding="utf8"))
    parent[section][key]=value
    (tmp_path/phase["extends"]).write_text(yaml.safe_dump(parent),encoding="utf8")
    path=tmp_path/"phase_A.yaml";path.write_text(yaml.safe_dump(phase),encoding="utf8")
    with pytest.raises(ValueError,match="changed specification"):
        runner.parse_contract(path)


def test_unknown_config_key_rejected(tmp_path):
    phase=yaml.safe_load((runner.PACKAGE/"phase_A_v1_0.yaml").read_text(encoding="utf8"))
    phase["settings"]["unused_override"]=42
    (tmp_path/phase["extends"]).write_bytes((runner.PACKAGE/phase["extends"]).read_bytes())
    path=tmp_path/"phase_A.yaml";path.write_text(yaml.safe_dump(phase),encoding="utf8")
    with pytest.raises(ValueError,match="changed specification"):
        runner.parse_contract(path)


def test_server_source_verifier_never_uses_git(tmp_path,monkeypatch):
    target=tmp_path/"scripts/stage3/run_selective_reoptimization_a.py"
    target.parent.mkdir(parents=True);target.write_text("actual local source",encoding="utf8")
    names={target.relative_to(tmp_path).as_posix():runner.file_hash(target)}
    source=dict(files=names,source_hash=runner.digest(names),commit="new_local_commit")
    monkeypatch.setattr(runner,"PROJECT",tmp_path);monkeypatch.setattr(runner,"DRIVER",target)
    monkeypatch.setattr(runner,"git",lambda *args:pytest.fail("server invoked Git"))
    runner.verify_sources(source)
    target.write_text("unsynchronized",encoding="utf8")
    with pytest.raises(ValueError,match="not synchronized"):
        runner.verify_sources(source)


def test_passing_xml_is_evidence_not_all_acceptance(tmp_path):
    good=tmp_path/"good.xml";good.write_text('<testsuite><testcase name="bounded_gate"/></testsuite>',encoding="utf8")
    assert runner.checks_evidence([good])[0]["tests"]==1
    bad=tmp_path/"bad.xml";bad.write_text('<testsuite><testcase><skipped/></testcase></testsuite>',encoding="utf8")
    with pytest.raises(ValueError,match="skipped"):
        runner.checks_evidence([bad])


def test_selection_record_public_schema_and_disjointness():
    scope=dict(eligible_ids=["a","b"],replan_ids=["a"],fixed_background_ids=["b"],mandatory_ids=["a"],nominal_k=8,actual_k=1,objective_frame_id="frame")
    rows=[dict(time=20.,planning_timing=dict(snapshot_time=0.,effective_time=20.),selective_replanning=dict(scope=scope,status="LOCAL"))]
    original=deepcopy(rows)
    exported=runner.selection_envelopes(rows,"run","binding","s3r23")[0]
    assert exported["scope"]["selected_ids"]==["a"] and exported["scope"]["background_ids"]==["b"]
    assert exported["snapshot_time"]==0. and exported["effective_time"]==20.
    assert rows==original
    schema=json.loads((runner.PACKAGE/"records.schema.json").read_text(encoding="utf8"))
    assert set(schema["required"])<=set(exported)
    assert set(schema["properties"]["scope"]["required"])<=set(exported["scope"])
    assert exported["schema_version"]==schema["properties"]["schema_version"]["const"]
    assert exported["record_type"] in schema["properties"]["record_type"]["enum"]
    for key in ("eligible_ids","selected_ids","background_ids","mandatory_ids"):
        assert all(isinstance(aid,str) for aid in exported["scope"][key])
        assert len(exported["scope"][key])==len(set(exported["scope"][key]))
    scope["fixed_background_ids"]=["a","b"]
    with pytest.raises(ValueError,match="overlap"):
        runner.selection_envelopes(rows,"run","binding","s3r23")


def test_incomplete_frame_keeps_raw_unknown_without_inventing_selection():
    rows=[dict(time=1.,selective_replanning=dict(status="LOCAL_SEARCH_INCOMPLETE",reason="FRAME_UNKNOWN",scope=None,objective_frame_id=None,solve_calls=0,scope_expansions=0))]
    original=deepcopy(rows)
    assert runner.selection_envelopes(rows,"run","binding","s3r23")==[]
    summary=report.selection_summary(rows)
    assert summary["status_counts"]=={"LOCAL_SEARCH_INCOMPLETE":1}
    assert summary["scope_rows"][0]["scope"] is None
    assert rows==original


def test_X01_conservation_pending_not_failed():
    quality=dict(N_offered=15,N_admitted_total=12,N_rerouted=1,N_upstream_pending=2,L_success=3,F_energy_total=1,N_airborne_pending=8)
    assert report.conservation(quality)["status"]=="PASS"
    quality["F_energy_total"]+=8
    assert report.conservation(quality)["status"]=="FAIL"


def test_rejected_proposal_same_hash_other_legitimate_task_not_false_failure():
    callbacks=[dict(planning_timing=dict(snapshot_time=0.,effective_time=20.),actual_delivery_status="REJECTED"),
               dict(planning_timing=dict(snapshot_time=1.,effective_time=20.),actual_delivery_status="COMMITTED",actual_submitted_trajectory_hashes=["h"])]
    tasks=[dict(snapshot_time=0.,proposed_trajectory_hashes=["h"])]
    events=[dict(type="maneuver_started",time=20.,details=dict(trajectory_hash="h"))]
    assert report.proposal_nonexecution(callbacks,events,tasks)["status"]=="PASS"
    assert report.proposal_nonexecution(callbacks[:1],events,tasks)["status"]=="FAIL"
    assert report.proposal_nonexecution(callbacks[:1],events)["status"]=="INCOMPLETE"


def test_complete_normal_work_includes_history_exactly_once_and_pending_partial():
    tasks=[dict(snapshot_time=0.,status="COMMITTED",finished_time=20.,normal_work_s=2.,selective_history_work_s=.3),
           dict(snapshot_time=25.,status="COMPUTING",normal_work_s=1.,selective_history_work_s=.2)]
    services=[dict(kind="selective_history",wall_s=.3,cpu_s=.3),dict(kind="selective_history",wall_s=.2,cpu_s=.2)]
    result=runner.lifecycle_summary(tasks,[],services,latency)
    assert result["full_normal_planning_work"]["total_s"]==pytest.approx(3.5)
    assert result["pending_tasks"]==1
    assert result["availability_delay_s"]["total_s"]==20.
    assert result["late_rate"]==0.


def test_existing_ledger_credit_survives_restart_and_cannot_reset(tmp_path):
    from terminal_marl.core.rotating_slots.stage3_learning.campaign import CampaignLedger
    now=[100.]
    path=tmp_path/"ledger.json"
    ledger=CampaignLedger(path,clock=lambda:now[0]);ledger.data.update(total_s=7200.,compute_stop_s=7200.,phase_caps={"A":7200.})
    ledger.pause();ledger.credit("actual_tests",12.5,phase="A")
    again=CampaignLedger(path,clock=lambda:now[0]);again.credit("actual_tests",12.5,phase="A")
    assert again.elapsed==pytest.approx(12.5)
    with pytest.raises(ValueError,match="already differs"):
        again.credit("actual_tests",0.,phase="A")
    assert again.data["total_s"]==7200.


def test_small_actual_checkpoint_and_replay_output_roundtrip(tmp_path):
    """Actual shared serialization path; no CP solver and no formal experiment."""
    from terminal_marl.core.rotating_slots.stage2_campaign import busy_fixture, frozen_manifest, save_checkpoint, restore_checkpoint, digest
    from terminal_marl.core.rotating_slots.stage2_planning import NormalSchedulerCallback
    from terminal_marl.core.rotating_slots.replay import ReplayBundle
    world=busy_fixture(frozen_manifest(),scenario_profile="medium24_rate150_v1",rescue_rule_profile="rescue_need_v7")
    callback=NormalSchedulerCallback("rh_cp_sat",planning_profile="downstream_service_v8",selective_method="RULE_LOCAL_K8")
    identity=dict(base_config_hash=world.config.config_hash,manifest_sha256=digest(world.manifest.to_dict()),
                  service_profile=world.config.service_profile,rescue_rule_profile="rescue_need_v7",planning_profile="downstream_service_v8")
    checkpoint=tmp_path/"smoke.json";save_checkpoint(checkpoint,world,callback,identity)
    restored=restore_checkpoint(checkpoint,NormalSchedulerCallback("rh_cp_sat",planning_profile="downstream_service_v8",selective_method="RULE_LOCAL_K8"),identity)
    assert restored.time==world.time and restored.metrics()==world.metrics()
    record=restored.to_dict();record.update(decisions=[],stage2_provenance=dict(source="SMOKE_ONLY"),tentative_predictions=[])
    ReplayBundle.write(tmp_path/"replay",record=record,code_commit="SMOKE_ONLY",eager=False)
    archive=ReplayBundle.validate(tmp_path/"replay")
    assert archive.state_count>=1


def test_report_uses_authoritative_common_endpoint_costs_and_pending(tmp_path):
    write(tmp_path/"resolved_run_binding.json",dict(binding_id="binding",source_commit="actual_new_commit"))
    quality=dict(N_offered=3,N_admitted_total=3,N_rerouted=0,N_upstream_pending=0,L_success=1,F_energy_total=0,N_airborne_pending=2,L_normal=1,L_emergency=0,E_terminal_J=100.,holding_time=20.,initial_airborne_count=3,initial_ground_count=0,manifest_offered=0,unresolved_low_soc_count=1)
    life=runner.lifecycle_summary([dict(snapshot_time=0.,status="COMPUTING",normal_work_s=2.)],[],[],latency)
    selections=dict(max_review_age_s=15.,max_progress_age_s=10.,overflow_full_fallback_tasks=0)
    for root in report.ROOTS:
        for method in report.METHODS:
            base=tmp_path/"evaluation"/root/method
            write(base/"boundary_0005.json",dict(time=5.,quality=quality,normal_lifecycle=life,costs=dict(dispatch_wall_s=3.),selection_summary=selections))
            write(base/"scheduler_records.json",[]);write(base/"normal_tasks.json",[]);write(base/"online_services.json",[])
        write(tmp_path/"evaluation"/root/"FULL_H"/"boundary_0010.json",dict(time=10.,quality=dict(quality,L_normal=9),normal_lifecycle=dict(life,pending_tasks=99),costs=dict(dispatch_wall_s=999.)))
    result=report.report(tmp_path,runner.PROJECT)
    summary=runner.load(tmp_path/"analysis_summary.json")
    assert summary["pairs"]["s3r23"]["common_completed_boundary_s"]==5
    assert summary["pairs"]["s3r23"]["methods"]["FULL_H"]["pending_computation_callbacks"]==1
    assert summary["acceptance"]["A20"]["status"]=="INCOMPLETE_OR_FAILED"
    assert result["human_replay_status"]=="HUMAN_PENDING"
    text=(tmp_path/"system_counts.csv").read_text(encoding="utf8")
    assert "999" not in text and "99" not in text


def test_snapshot_pair_same_cold_frame_and_shared_outer_deadline(tmp_path,monkeypatch):
    from terminal_marl.core.rotating_slots import stage2_planning as planning
    from terminal_marl.core.rotating_slots import selective_replanning as selective
    from terminal_marl.core.rotating_slots.planning_latency import PlanningTiming
    checkpoint=tmp_path/"registered.json";write(checkpoint,dict(physical="original"))
    item=dict(path=str(checkpoint),time=15,sha256=runner.file_hash(checkpoint))
    source=SimpleNamespace(time=15.,version="actual_source")
    projected=SimpleNamespace(time=35.,version="future_same")
    preparation_calls=[];discarded=[];tokens=set()
    def prepare(proposal,timing):
        preparation_calls.append((proposal,timing));token="token"+str(len(preparation_calls));tokens.add(token)
        return SimpleNamespace(accepted=True,status="PREPARED",reason=""),token
    def discard(token):
        discarded.append(token);tokens.remove(token)
    world=SimpleNamespace(build_planning_snapshot=lambda:source,events=[dict(type="maneuver_started")],
        _scheduler_port=SimpleNamespace(prepare_latency_submission=prepare,discard_latency_submission=discard))
    monkeypatch.setattr(runner,"restore",lambda *args:world)
    monkeypatch.setattr(planning,"CandidateProvider",lambda **kwargs:SimpleNamespace(cold=True))
    monkeypatch.setattr(planning,"EvaluationContext",lambda *args,**kwargs:object())
    frame=SimpleNamespace(identity="same_frame")
    monkeypatch.setattr(selective,"build_global_frame",lambda *args:frame)
    clocks=[0.]
    def clock():
        clocks[0]+=.1;return clocks[0]
    monkeypatch.setattr(runner.time,"perf_counter",clock)
    ports=[];callbacks=[]
    class Port:
        def __init__(self,snapshot,original,deadline,**kwargs):
            assert snapshot is projected and original is source
            self.proposal=[SimpleNamespace(trajectory_hash="proposal")];self.deadline=deadline;self.origin=kwargs["planning_wall_origin"]
            assert deadline-self.origin==pytest.approx(19.)
            assert kwargs["search_wall_deadline"]-self.origin==pytest.approx(15.)
            ports.append(self)
    class Callback:
        def __init__(self):self.records=[];self.initialized=False
        def initialize_imported_history(self,actual,events):
            assert actual is source and events is world.events;self.initialized=True
        def __call__(self,port):
            assert self.initialized
            from terminal_marl.core.rotating_slots import safety
            assert not safety._PLAN_CERTIFICATES and not safety._NUMERIC_CERTIFICATES
            # This arm warms process-global certificates; the next arm must
            # observe emptiness before doing any of its own numerical work.
            safety._PLAN_CERTIFICATES["previous_arm"]=True
            safety._NUMERIC_CERTIFICATES["previous_arm"]=True
            self.records.append(dict(time=35.,planning_timing=dict(snapshot_time=15.,effective_time=35.),selective_replanning=dict(scope=dict(eligible_ids=["a"],replan_ids=["a"],fixed_background_ids=[],mandatory_ids=[],actual_k=1,nominal_k=8,objective_frame_id="same_frame"))))
    def scheduler(*args):
        assert args[-1] is frame
        value=Callback();callbacks.append(value);return value
    monkeypatch.setattr(runner,"scheduler",scheduler)
    d=SimpleNamespace(PlanningTiming=PlanningTiming,project_visible_snapshot=lambda *args,**kwargs:projected,FuturePort=Port,atomic_json=write)
    binding=dict(inputs=dict(s3r23=dict(snapshots=[item])),binding_id="binding")
    result=runner.snapshot_job(tmp_path/"output","s3r23",15,binding,d)
    assert result["same_frame"] and result["cold_normal_cache"]
    assert len(callbacks)==2 and len(ports)==2
    assert len(preparation_calls)==2 and len(discarded)==2 and not tokens
    assert result["rows"][1]["cache_reset"]["before"]["safety._PLAN_CERTIFICATES"]["entries"]==1
    assert all(item.get("currsize",item.get("entries"))==0 for row in result["rows"] for item in row["cache_reset"]["after"].values())
    assert all(row["preparation"]["calls"]==1 and row["preparation"]["ready_eligible"] for row in result["rows"])
    assert all(row["full_online_pipeline_cost_s"] is None for row in result["rows"])
    assert "complete_counterfactual_cost_s" not in result["rows"][0]
    from terminal_marl.core.rotating_slots.performance.parallel import _clear_worker_accelerators
    _clear_worker_accelerators()


@pytest.mark.parametrize("accepted,status,token,finish,expected",[(True,"PREPARED","issued",20.,"LATE"),(False,"UNKNOWN",None,2.,"UNKNOWN"),(False,"REJECTED",None,2.,"REJECTED"),(True,"PREPARED",None,2.,"UNKNOWN")])
def test_counterfactual_preparation_once_no_publish_and_late_not_usable(monkeypatch,accepted,status,token,finish,expected):
    calls=[];discarded=[];clock=iter([1.,finish])
    monkeypatch.setattr(runner.time,"perf_counter",lambda:next(clock))
    def prepare(plans,timing):
        calls.append((plans,timing));return SimpleNamespace(accepted=accepted,status=status,reason="observed"),token
    port=SimpleNamespace(prepare_latency_submission=prepare,discard_latency_submission=discarded.append,
        publish_latency_submission=lambda *args:pytest.fail("counterfactual published"))
    value=runner.counterfactual_preparation(SimpleNamespace(_scheduler_port=port),["unchanged_plan"],"unchanged_timing",ready_wall_deadline=19.)
    assert len(calls)==1 and calls[0]==(("unchanged_plan",),"unchanged_timing")
    assert value["status"]==expected and not value["ready_eligible"] and not value["published"]
    assert discarded==([token] if token else [])
