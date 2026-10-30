"""Small output/measurement preflight; no solver, server, or formal rollout."""
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.stage3 import run_service_opportunity_comparison as runner
from scripts.stage3 import report_service_opportunity_comparison as report


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding="utf8")


def frames(demand=True):
    aircraft=[dict(id="arbitrary_name",status="HOLD_MIDDLE",service_class="fast") ] if demand else []
    return [dict(time=t,aircraft=aircraft,pads=[]) for t in (20.,320.)]


def test_interval_union_counts_repeated_overlap_once():
    assert report.merged([(20,30),(25,35),(20,30)],20,320)==[(20,35)]
    assert report.measure(report.complement([(20,30),(25,35)],20,320))==285


def test_actual_middle_inner_starts_and_completed_supplies_ignore_names():
    plans={"x":dict(trajectory_hash="x",source_ring="middle",target_ring="inner",end_time=50),
           "y":dict(trajectory_hash="y",source_ring="outer",target_ring="middle",end_time=60),
           "z":dict(trajectory_hash="z",source_ring="middle",target_ring="inner",end_time=400)}
    events=[dict(type="maneuver_started",time=t,aircraft_id=aid,details=dict(trajectory_hash=h))
            for t,aid,h in ((25,"initial_outer", "x"),(30,"initial_middle","y"),(310,"other","z"))]
    events.append(dict(type="transfer_completed",time=50,aircraft_id="initial_outer"))
    result=report.middle_inner(events,plans,20,320)
    assert result["actual_maneuver_start_count"]==2
    assert result["completed_effective_supply_count"]==1
    assert result["completed"][0]["aircraft_id"]=="initial_outer"


def test_physical_pad_approach_plus_service_and_no_demand_not_avoidable():
    f=frames(False)
    for row in f: row["aircraft"]=[dict(id="a",status="LANDED_SERVICE",pad_id=0,service_end=40,service_class="fast")]
    events=[dict(type="maneuver_started",time=25,aircraft_id="a",details=dict(trajectory_hash="p"))]
    plan=dict(aircraft_id="a",pad_id=0,end_time=60.)
    idle=report.physical_idle(events,{"p":plan},f,1,{"fast":20})
    assert idle["occupied_by_pad"][0]==[(20.,40),(46.5,80.)]
    assert idle["pad_seconds_idle"]==246.5
    assert idle["no_demand_pad_seconds_idle"]==246.5
    witness=report.witnessed_idle([],events,idle,[],event_log_complete=True)
    assert witness["witnessed_avoidable_idle_pad_seconds"] is None


def certificate():
    return dict(joint_prefix_certified=True,source_snapshot_id="source",projection_protocol="fixed_committed_hold_v1",
        effective_snapshot_id="effective",effective_time=40.,remaining_ready_margin_s=1.,
        finals=[dict(pad_id=0,flight_start=40.,descent_start=50.,service_end=80.)])


def test_witness_lower_bound_union_and_state_change_cutoff():
    cert=certificate(); cb=dict(planning_timing=dict(snapshot_time=20.),service_opportunity_certificates=[cert,cert])
    idle=report.physical_idle([],{},frames(),1,{"fast":20})
    got=report.witnessed_idle([cb],[],idle,[(20,320)],event_log_complete=True)
    assert got["witnessed_avoidable_idle_pad_seconds"]==30.
    assert got["unclassified_idle_pad_seconds"]==270.
    changed=report.witnessed_idle([cb],[dict(time=60,type="admitted")],idle,[(20,320)],event_log_complete=True)
    assert changed["witnessed_avoidable_idle_pad_seconds"]==10.
    assert changed["unclassified_idle_pad_seconds"]==290.
    early=report.witnessed_idle([cb],[dict(time=30,type="admitted")],idle,[(20,320)],event_log_complete=True)
    assert early["witnessed_avoidable_idle_pad_seconds"] is None
    assert early["unclassified_idle_pad_seconds"]==300.
    busy=dict(idle,idle_by_pad={0:[]},pad_seconds_idle=0.)
    assert report.witnessed_idle([cb],[],busy,[(20,320)],event_log_complete=True)["witnessed_avoidable_idle_pad_seconds"]==0.
    assert report.witnessed_idle([cb],[],idle,[],event_log_complete=True)["witnessed_avoidable_idle_pad_seconds"]==0.


def test_missing_bindings_and_evidence_are_null_not_zero():
    idle=report.physical_idle([],{},frames(),1,{"fast":20})
    cb=dict(planning_timing=dict(snapshot_time=20.),service_opportunity_certificates=[dict(certificate(),source_snapshot_id=None)])
    assert report.witnessed_idle([cb],[],idle,[(20,320)],event_log_complete=True)["witnessed_avoidable_idle_pad_seconds"] is None
    assert report.witnessed_idle([],[],idle,[(20,320)])["witnessed_avoidable_idle_pad_seconds"] is None
    assert report.physical_idle([],{},[],1,{})["pad_seconds_idle"] is None


def test_supply_gap_demand_only_and_window_edges_censored():
    got=report.supply_gap([dict(end=100.)],[(20.,320.)])
    assert got["longest_no_inner_supply_s"]==220.
    assert got["intervals"][0]["left_censored"] is True
    assert got["intervals"][-1]["right_censored"] is True
    assert report.supply_gap([],[])["longest_no_inner_supply_s"]==0.
    assert report.supply_gap([],None)["longest_no_inner_supply_s"] is None


def test_inner_stock_empty_is_distinct_from_no_transfer_completion():
    f=frames(); f[0]["aircraft"].append(dict(id="stock",status="HOLD_INNER",ring="inner"))
    assert report.inner_stock_empty(f)["longest_inner_stock_empty_with_upstream_demand_s"]==0.
    empty=frames()
    for row in empty: row["aircraft"][0]["ring"]="middle"
    assert report.inner_stock_empty(empty)["longest_inner_stock_empty_with_upstream_demand_s"]==300.
    assert report.inner_stock_empty([])["longest_inner_stock_empty_with_upstream_demand_s"] is None


def test_common_copy_preserves_bytes_and_existing_output(tmp_path):
    origin=tmp_path/"old"; source=origin/"common_initial/s3r23"
    checkpoint=source/"common.checkpoint.json"
    payload=dict(boundary_summary=dict(status="COMMON_INITIAL_READY",t0=20.,source_binding=dict(root="s3r23")),sink=None)
    save(checkpoint,dict(payload=payload))
    for name in runner.COMMON_FILES[1:]: save(source/name,[])
    before={p.name:p.read_bytes() for p in source.iterdir()}
    output=tmp_path/"new"
    got=runner.copy_common(output,"s3r23",origin=origin,expected_hash=runner.base.file_hash(checkpoint))
    assert got["initializer_regenerated"] is False
    assert {p.name:p.read_bytes() for p in source.iterdir()}==before
    assert {p.name:p.read_bytes() for p in (output/"common_initial/s3r23").iterdir()}==before
    with pytest.raises(ValueError,match="no overwrite"): runner.copy_common(output,"s3r23",origin=origin,expected_hash=runner.base.file_hash(checkpoint))


def test_cli_and_literal_thread_settings(monkeypatch):
    args=runner.parser().parse_args(["--stage","smoke","--output","x","--root","s3r23","--method","FULL_H","--cpu","16","--source-freeze","y"])
    assert args.evaluation_profile=="service_opportunity_v1"
    assert runner.CAPS==dict(prepare=120.,arm=1200.,audit=300.,smoke=120.)
    for key in runner.base.THREADS: monkeypatch.setenv(key,"1")
    assert len(runner.launch_threads())==5
    monkeypatch.setenv(runner.base.THREADS[0],"01")
    with pytest.raises(ValueError,match="literal"): runner.launch_threads()


@pytest.mark.parametrize("output_name,stage,seconds",[
    (runner.OUTPUT_NAME,"smoke",20),(runner.SUPPLY_R2_OUTPUT_NAME,"smoke",25),
    (runner.LOCAL_RECOVERY_OUTPUT_NAME,"smoke",25),
    (runner.OUTPUT_NAME,"arm",300),(runner.SUPPLY_R2_OUTPUT_NAME,"arm",300),
    (runner.LOCAL_RECOVERY_OUTPUT_NAME,"arm",300)])
def test_smoke_is_twenty_seconds_separate_output(monkeypatch,tmp_path,output_name,stage,seconds):
    for key in runner.base.THREADS: monkeypatch.setenv(key,"1")
    source=dict(commit="test",source_hash="hash",files={p: "x" for p in (
        "scripts/stage3/run_service_opportunity_comparison.py","scripts/stage3/report_service_opportunity_comparison.py",
        "tests/test_service_opportunity_campaign.py")})
    freeze=tmp_path/"freeze.json"; save(freeze,source)
    monkeypatch.setattr(runner,"validated_output",lambda path:Path(path))
    monkeypatch.setattr(runner.base,"verify_sources",lambda source:None)
    monkeypatch.setattr(runner.base,"dependencies",lambda project:SimpleNamespace())
    monkeypatch.setattr(runner.base,"file_hash",lambda path:runner.COMMON_HASHES["s3r23"])
    monkeypatch.setattr(runner.base,"print_result",lambda result:None)
    monkeypatch.setattr(runner.old,"set_cpu",lambda args,d:None)
    @contextmanager
    def ledger(*args,**kwargs):
        assert kwargs["policy"]["campaign"]==output_name
        yield dict(jobs={"service_"+stage+"_s3r23_FULL_H":dict(status="running",parent_pid=runner.os.getppid())})
    monkeypatch.setattr(runner.old,"ledger_transaction",ledger)
    captured={}
    def arm(output,root,method,checkpoint,binding,**kwargs):
        captured.update(output=output,binding=binding,**kwargs)
        return dict(status="COMPLETE",target_relative_s=kwargs["episode_seconds"])
    monkeypatch.setattr(runner.comparison,"run_arm",arm)
    output=tmp_path/output_name
    result=runner.main(["--stage",stage,"--output",str(output),"--root","s3r23","--method","FULL_H",
        "--cpu","16","--source-freeze",str(freeze),"--_child-job-id","service_"+stage+"_s3r23_FULL_H"])
    assert result["target_relative_s"]==seconds
    assert captured["output"]==(output/"smoke_only" if stage=="smoke" else output)
    assert captured["binding"]["run_role"]==("SMOKE_NOT_FORMAL" if stage=="smoke" else "FORMAL_SINGLE_ARM")
    assert captured["binding"]["campaign"]==output_name


def test_only_authorized_output_names_and_fixed_parent(monkeypatch,tmp_path):
    monkeypatch.setattr(Path,"home",classmethod(lambda cls:tmp_path))
    parent=tmp_path/"training_outputs/terminal"
    for name in runner.OUTPUT_NAMES:
        assert runner.validated_output(parent/name)==(parent/name).resolve()
    with pytest.raises(ValueError): runner.validated_output(parent/"unapproved_campaign")
    with pytest.raises(ValueError): runner.validated_output(tmp_path/runner.SUPPLY_R2_OUTPUT_NAME)


def test_campaign_policies_and_ledgers_are_independent(monkeypatch,tmp_path):
    monkeypatch.setattr(Path,"home",classmethod(lambda cls:tmp_path))
    parent=tmp_path/"training_outputs/terminal"
    first=parent/runner.OUTPUT_NAME; second=parent/runner.SUPPLY_R2_OUTPUT_NAME
    original=dict(runner.POLICY)
    assert runner.policy_for_output(first)==original
    assert runner.policy_for_output(second)["previous_ledger_readonly"]==runner.OUTPUT_NAME+"/repair_budget_ledger.json"
    assert runner.policy_for_output(second)["run_role"]=="AUTHORIZED_SUPPLY_RESERVATION_REPAIR_FOLLOWUP"
    third=parent/runner.LOCAL_RECOVERY_OUTPUT_NAME
    third_policy=runner.policy_for_output(third)
    assert third_policy["previous_ledger_readonly"]==runner.SUPPLY_R2_OUTPUT_NAME+"/repair_budget_ledger.json"
    assert third_policy["run_role"]=="AUTHORIZED_LOCAL_RECOVERY_FOLLOWUP"
    assert third_policy["cap_child_wall_s"]==7200.
    with runner.old.ledger_transaction(first,policy=runner.policy_for_output(first)) as ledger:
        ledger["jobs"]["previous_attempt"]=dict(status="completed",child_wall_s=17.)
        ledger["settled_child_wall_s"]=17.
    old_bytes=(first/"repair_budget_ledger.json").read_bytes()
    with runner.old.ledger_transaction(second,policy=runner.policy_for_output(second)) as ledger:
        assert ledger["campaign"]==runner.SUPPLY_R2_OUTPUT_NAME
        assert ledger["jobs"]=={} and ledger["settled_child_wall_s"]==0.
        assert ledger["cap_child_wall_s"]==7200.
        assert ledger["historical_campaign_budget_reset"] is False
    assert (first/"repair_budget_ledger.json").read_bytes()==old_bytes
    assert runner.POLICY==original


def test_output_json_csv_grammar_and_missing_metrics(tmp_path,monkeypatch):
    monkeypatch.setattr(report,"PROJECT",tmp_path)
    metric=report.extract_metrics(tmp_path/"absent","s3r23","FULL_H",dict(energy_kwh=99.,increment=dict(
        cumulative_delta=dict(L_normal=3,energy_kwh=2.5),inherited_final_touchdown_ids=["inherited"],treatment_final_touchdown_ids=["a","b"])))
    assert metric["normal_touchdown_delta"]==3 and metric["inherited_final_touchdown_count"]==1
    assert metric["increment_energy_kwh"]==2.5 and metric["energy_j"] is None
    assert metric["treatment_final_touchdown_count"]==2 and metric["physical_idle"] is None
    value=dict(schema="service_opportunity_comparison_v1",campaigns=dict(new=dict(input_root=str(tmp_path/"inputs"),metrics=[metric])))
    output=report.write_outputs(tmp_path/"artifacts/report",value)
    assert json.loads((output/"summary.json").read_text())["campaigns"]["new"]["metrics"][0]["physical_idle"] is None
    assert "null" in (output/"comparison.csv").read_text()
    with pytest.raises(FileExistsError): report.write_outputs(output,value)


def lifecycle_row(hash_,**changes):
    return dict(dict(candidate_hash=hash_,aircraft_id="a",action_hashes=[hash_],critical=True,
        source_ring="middle",single_accepted=True,retained_in_catalog=True,catalog_identities=["cat1"],
        actual_solve_events=[],joint_checked_events=[],certified_events=[],predicted_service_window=70.),**changes)


def test_critical_actual_cp_denominator_and_repeated_catalog_deduplication():
    solved=lifecycle_row("move",actual_solve_events=[dict(catalog_identity="cat1",model_index=2)],
        joint_checked_events=[dict(elapsed_s=2.)],certified_events=[dict(elapsed_s=3.)])
    repeat=dict(solved,catalog_identities=["cat1","cat2"])
    rejected=lifecycle_row("unsafe",single_accepted=False,retained_in_catalog=False,catalog_identities=[],
        single_rejection_reason="SEPARATION_CONFLICT")
    pending=lifecycle_row("unsubmitted")
    starts=[dict(aircraft_id="a",trajectory_hash="move",start=40.,end=65.)]
    callbacks=[dict(service_candidate_lifecycle=[solved,rejected,pending],solver_stop_reason="BUDGET_EXHAUSTED",
        service_search_events=[dict(event="JOINT_CHECKED",certificate_reused=False,batch_status="CONFLICT",
            conflicts=[[dict(candidate_hash="move"),dict(candidate_hash="other")]])]),
        dict(service_candidate_lifecycle=[repeat])]
    got=report.candidate_lifecycle_metrics(callbacks,dict(starts=starts,completed=starts))
    assert got["critical_generated_count"]==3
    assert got["critical_single_accepted_count"]==2
    assert got["critical_catalog_eligible_count"]==2
    assert got["critical_actual_cp_participated_count"]==1
    assert got["critical_actual_cp_participation_rate"]==.5
    assert got["critical_generated_actual_cp_participation_rate"]==pytest.approx(1/3)
    assert got["critical_unsolved_count"]==2
    assert got["unsolved_reason_counts"]["SEPARATION_CONFLICT"]==1
    assert got["real_joint_check_count"]==1 and got["joint_conflict_count"]==1
    assert len(got["middle_inner_actual_timing"])==1
    assert got["middle_inner_actual_timing"][0]["completion_minus_predicted_window_s"]==[-5.]
    assert report.candidate_lifecycle_metrics([{}],dict(starts=[],completed=[])) is None
    assert report.candidate_lifecycle_metrics([dict(service_candidate_lifecycle=[])],dict(starts=[],completed=[]))["critical_generated_count"]==0


def test_local_recovery_real_checks_deliveries_withdrawals_and_timeout_categories():
    delta=dict(affected_action_hashes=["bad"],retained_action_hashes=["safe"],
        withdrawn_action_reasons={"bad":"RELEASE_SUPPORT_WITHDRAWN"},recursive_release_edges=[["support","bad"]],
        pairs_checked=2,reused_pairs=4,envelope_proofs=1,conflicts=[dict(action_hash="bad",reason="CONFLICT")])
    tasks=[dict(traffic_policy="action_local_recovery_v1",status="COMMITTED",submitted=1,
        recovery_delivery="standby",revalidations=[dict(local_recovery=delta,status="CERTIFIED")]),
        dict(traffic_policy="action_local_recovery_v1",status="COMMITTED",submitted=2,recovery_delivery="repaired"),
        dict(traffic_policy="action_local_recovery_v1",status="REJECTED",submitted=0,finished_time=45.,
            reason="REVALIDATION_WORK_DEADLINE",revalidations=[dict(local_recovery=delta)],repair=dict(timed_out=True))]
    got=report.local_recovery_metrics(tasks)
    assert got["affected_action_occurrences"]==2 and got["retained_action_occurrences"]==2
    assert got["recursive_release_edge_occurrences"]==2
    assert got["real_joint_pair_checks"]==4 and got["reused_pair_certificates"]==8
    assert got["real_joint_conflict_count"]==2
    assert got["local_change_whole_batch_rejection_count"]==1
    assert got["local_change_no_certified_standby_cancellation_count"]==0
    assert got["certified_revalidation_join_count"]==1
    assert got["delivered_task_counts"]==dict(original=0,standby=1,repaired=1)
    assert got["delivered_plan_counts"]["repaired"]==2
    assert got["timeout_reason_counts"]=={"REVALIDATION_WORK_DEADLINE":1}
    assert got["repair_timed_out_count"]==1
    tasks[-1]["reason"]="NO_CERTIFIED_STANDBY"
    assert report.local_recovery_metrics(tasks)["local_change_no_certified_standby_cancellation_count"]==1
    assert report.local_recovery_metrics([dict(status="COMMITTED",submitted=2)]) is None


def test_late_guard_only_rejection_counts_task_once_and_reads_recovery_conflicts():
    observed=dict(affected_action_hashes=["withdrawn"],retained_action_hashes=["safe"],
        pairs_checked=0,conflicts=[],observed_time=46.,reason="TRAFFIC_REVALIDATION_REQUIRED")
    conflict=dict(action_hash="withdrawn",other_aircraft_id="traffic",interval=[46.,60.],
        status="CONFLICT",reason="SEPARATION_CONFLICT")
    attempt=dict(hashes=["safe","withdrawn"],accepted=False,status="CONFLICT",reason="SEPARATION_CONFLICT")
    repair=dict(repair_attempts=[attempt],timed_out=False)
    task=dict(traffic_policy="action_local_recovery_v1",status="REJECTED",submitted=0,finished_time=46.,
        ready_deadline=45.,reason="TRAFFIC_CHANGED_AFTER_DEADLINE",revalidations=[],
        guard_local_recovery=[observed,observed],repair=repair,
        local_recovery_events=[dict(conflicts=[conflict]),dict(conflicts=[conflict]),repair])
    got=report.local_recovery_metrics([task])
    assert got["local_change_whole_batch_rejection_count"]==1
    assert got["local_change_no_certified_standby_cancellation_count"]==0
    assert got["real_joint_pair_checks"]==0
    assert got["real_joint_conflict_count"]==1 and got["real_joint_conflicts"]==[conflict]
    assert got["failed_repair_attempt_count"]==1
    assert got["failed_repair_attempt_status_counts"]=={"CONFLICT":1}
    assert got["failed_repair_attempt_reason_counts"]=={"SEPARATION_CONFLICT":1}


def test_real_report_fixture_reads_replay_and_keeps_missing_old_instrumentation_null(tmp_path):
    directory=tmp_path/"arms/s3r23/FULL_H/evaluation/s3r23/FULL_H"
    replay=directory/"replay"; replay.mkdir(parents=True)
    events=[dict(type="maneuver_started",time=30.,aircraft_id="a",details=dict(trajectory_hash="move")),
        dict(type="transfer_completed",time=60.,aircraft_id="a"),
        dict(type="touchdown",time=80.,aircraft_id="a",details=dict(emergency=False))]
    for name,rows in (("events",events),("states",frames())):
        (replay/(name+".jsonl")).write_text("".join(json.dumps(r)+"\n" for r in rows),encoding="utf8")
    save(replay/"trajectories.json",dict(move=dict(trajectory_hash="move",aircraft_id="a",
        source_ring="middle",target_ring="inner",end_time=60.,pad_id=None)))
    (replay/"config.yaml").write_text("pads:\n  count: 1\n  service_classes:\n    fast:\n      duration_s: 20\n",encoding="utf8")
    save(directory/"method_scheduler_records.json",[dict(service_candidate_lifecycle=[lifecycle_row("move")])])
    save(directory/"method_normal_tasks.json",[dict(traffic_policy="action_local_recovery_v1",status="COMMITTED",submitted=1)])
    save(replay/"replay.json",dict(files={p.name:report.sha256(p.read_bytes()).hexdigest() for p in replay.iterdir()}))
    ordinary=dict(checks=dict(replay=True),energy_kwh=999.,increment=dict(cumulative_delta=dict(L_normal=1,energy_j=4.,energy_kwh=.25,L_emergency=0,
        failed_before_landing=0),inherited_final_touchdown_ids=[],treatment_final_touchdown_ids=["a"]))
    got=report.extract_metrics(tmp_path,"s3r23","FULL_H",ordinary)
    assert got["event_log_complete"] is True
    assert got["middle_inner_supply"]["completed_effective_supply_count"]==1
    assert got["candidate_lifecycle"]["middle_inner_actual_timing"][0]["actual_completion"]==60.
    assert got["local_recovery"]["delivered_task_counts"]["original"]==1
    assert got["treatment_normal_touchdown_count"]==1 and got["physical_idle"]["pad_seconds_idle"]==300.
    assert got["increment_energy_kwh"]==.25 and got["energy_j"]==4.
    save(directory/"method_scheduler_records.json",[dict()]); save(directory/"method_normal_tasks.json",[dict()])
    before={p:p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    old=report.extract_metrics(tmp_path,"s3r23","FULL_H",ordinary)
    assert old["candidate_lifecycle"] is None and old["local_recovery"] is None
    assert {p:p.read_bytes() for p in before}==before
