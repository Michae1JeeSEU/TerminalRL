"""Small output/measurement preflight; no solver, server, or formal rollout."""
from contextlib import contextmanager
import csv
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.stage3 import run_service_opportunity_comparison as runner
from scripts.stage3 import report_service_opportunity_comparison as report


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding="utf8")


def save_serialization_continuation_chain(parent):
    original=parent/runner.LOCAL_RECOVERY_OUTPUT_NAME/"repair_budget_ledger.json"
    save(original,dict(campaign=runner.LOCAL_RECOVERY_OUTPUT_NAME,cap_child_wall_s=7200.,
        settled_child_wall_s=8.607537163188681,jobs=dict(preflight=dict(status="failed"))))
    previous_output=parent/runner.LOCAL_RECOVERY_RESTOREFIX_OUTPUT_NAME
    previous=previous_output/"repair_budget_ledger.json"
    saved=dict(runner.policy_for_output(previous_output),settled_child_wall_s=966.7981451372616,
        jobs=dict(mappingproxy=dict(status="failed",exit_code=1),stopped=dict(status="failed",exit_code=-15)))
    save(previous,saved)
    return original,previous,saved


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
    (runner.LOCAL_RECOVERY_RESTOREFIX_OUTPUT_NAME,"smoke",25),
    (runner.LOCAL_RECOVERY_SERIALIZATIONFIX_OUTPUT_NAME,"smoke",25),
    (runner.FUTURE_COMMITMENT_OUTPUT_NAME,"smoke",25),
    (runner.OUTPUT_NAME,"arm",300),(runner.SUPPLY_R2_OUTPUT_NAME,"arm",300),
    (runner.LOCAL_RECOVERY_OUTPUT_NAME,"arm",300),
    (runner.LOCAL_RECOVERY_RESTOREFIX_OUTPUT_NAME,"arm",300),
    (runner.LOCAL_RECOVERY_SERIALIZATIONFIX_OUTPUT_NAME,"arm",300),
    (runner.FUTURE_COMMITMENT_OUTPUT_NAME,"arm",300)])
def test_smoke_is_twenty_seconds_separate_output(monkeypatch,tmp_path,output_name,stage,seconds):
    for key in runner.base.THREADS: monkeypatch.setenv(key,"1")
    source=dict(commit="test",source_hash="hash",files={p: "x" for p in (
        "scripts/stage3/run_service_opportunity_comparison.py","scripts/stage3/report_service_opportunity_comparison.py",
        "tests/test_service_opportunity_campaign.py","scripts/stage3/run_selective_common_initial_a4.py",
        "tests/test_selective_common_initial_a4.py")})
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
    if output_name==runner.LOCAL_RECOVERY_RESTOREFIX_OUTPUT_NAME:
        save(tmp_path/runner.LOCAL_RECOVERY_OUTPUT_NAME/"repair_budget_ledger.json",dict(
            campaign=runner.LOCAL_RECOVERY_OUTPUT_NAME,cap_child_wall_s=7200.,settled_child_wall_s=17.,jobs={}))
    elif output_name==runner.LOCAL_RECOVERY_SERIALIZATIONFIX_OUTPUT_NAME:
        save_serialization_continuation_chain(tmp_path)
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


def test_r4_independent_budget_and_io_only_watchdog(monkeypatch,tmp_path):
    monkeypatch.setattr(Path,"home",classmethod(lambda cls:tmp_path))
    output=tmp_path/"training_outputs/terminal"/runner.FUTURE_COMMITMENT_OUTPUT_NAME
    policy=runner.policy_for_output(output)
    assert policy["cap_child_wall_s"]==7200. and policy["prior_audits_child_wall_s"]==0.
    assert "historical_charge_chain" not in policy
    assert policy["previous_ledger_readonly"]==runner.LOCAL_RECOVERY_SERIALIZATIONFIX_OUTPUT_NAME+"/repair_budget_ledger.json"
    assert runner.stage_cap("arm",output)==1500.
    assert runner.stage_cap("arm",output.parent/runner.LOCAL_RECOVERY_SERIALIZATIONFIX_OUTPUT_NAME)==1200.
    assert [runner.stage_cap(s,output) for s in ("prepare","smoke","audit")]==[120.,120.,300.]
    for key in runner.base.THREADS: monkeypatch.setenv(key,"1")
    captured={}
    def bounded(*args,**kwargs): captured.update(kwargs); return dict(status="completed")
    monkeypatch.setattr(runner.old,"run_bounded_job",bounded)
    runner.main(["--stage","arm","--output",str(output),"--root","s3r23","--method","FULL_H","--cpu","16","--source-freeze","future_freeze.json"])
    assert captured["cap_s"]==1500. and captured["policy"]["cap_child_wall_s"]==7200.


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


def test_restorefix_budget_carries_prior_charge_without_writing_previous_ledger(tmp_path,monkeypatch):
    monkeypatch.setattr(Path,"home",classmethod(lambda cls:tmp_path))
    parent=tmp_path/"training_outputs/terminal"
    previous=parent/runner.LOCAL_RECOVERY_OUTPUT_NAME/"repair_budget_ledger.json"
    old=dict(campaign=runner.LOCAL_RECOVERY_OUTPUT_NAME,cap_child_wall_s=7200.,settled_child_wall_s=17.5,
        jobs=dict(prepare=dict(status="completed",child_wall_s=2.5),smoke=dict(status="failed",child_wall_s=15.)))
    save(previous,old); before=previous.read_bytes()
    output=parent/runner.LOCAL_RECOVERY_RESTOREFIX_OUTPUT_NAME
    policy=runner.policy_for_output(output)
    assert policy["cap_child_wall_s"]==7182.5
    assert policy["previous_campaign_settled_child_wall_s"]==17.5
    assert policy["shared_original_cap_child_wall_s"]==7200.
    assert policy["previous_ledger_readonly"]==runner.LOCAL_RECOVERY_OUTPUT_NAME+"/repair_budget_ledger.json"
    assert policy["previous_ledger_sha256"]==runner.base.file_hash(previous)
    assert "PRIOR_PREFLIGHTS_REMAIN_CHARGED" in policy["continuation_reason"]
    with runner.old.ledger_transaction(output,policy=policy) as ledger:
        assert ledger["settled_child_wall_s"]==0. and ledger["jobs"]=={}
        assert ledger["cap_child_wall_s"]+ledger["previous_campaign_settled_child_wall_s"]==7200.
        assert ledger["historical_campaign_budget_reset"] is False and ledger["old_ledgers_reset"] is False
    assert previous.read_bytes()==before
    old["jobs"]["smoke"]["status"]="running"; save(previous,old)
    with pytest.raises(ValueError,match="terminal"):
        runner.policy_for_output(output)
    old["jobs"]["smoke"]["status"]="failed"; old["settled_child_wall_s"]=7200.; save(previous,old)
    with pytest.raises(ValueError,match="exhausted"):
        runner.policy_for_output(output)


def test_restorefix_requires_readable_bound_previous_ledger(tmp_path,monkeypatch):
    monkeypatch.setattr(Path,"home",classmethod(lambda cls:tmp_path))
    parent=tmp_path/"training_outputs/terminal"; output=parent/runner.LOCAL_RECOVERY_RESTOREFIX_OUTPUT_NAME
    with pytest.raises(ValueError,match="ledger required"):
        runner.policy_for_output(output)
    previous=parent/runner.LOCAL_RECOVERY_OUTPUT_NAME/"repair_budget_ledger.json"
    save(previous,dict(campaign="wrong",cap_child_wall_s=7200.,settled_child_wall_s=0.,jobs={}))
    with pytest.raises(ValueError,match="binding differs"):
        runner.policy_for_output(output)


def test_serializationfix_subtracts_previous_actual_cap_and_preserves_readonly_charge_chain(tmp_path,monkeypatch):
    monkeypatch.setattr(Path,"home",classmethod(lambda cls:tmp_path))
    parent=tmp_path/"training_outputs/terminal"
    original,previous,saved=save_serialization_continuation_chain(parent)
    before={p:p.read_bytes() for p in (original,previous)}
    output=parent/runner.LOCAL_RECOVERY_SERIALIZATIONFIX_OUTPUT_NAME
    policy=runner.policy_for_output(output)
    assert policy["cap_child_wall_s"]==7191.392462836811-966.7981451372616
    assert policy["cap_child_wall_s"]==6224.59431769955
    assert policy["cap_child_wall_s"]!=7200.-966.7981451372616
    assert policy["previous_ledger_sha256"]==runner.base.file_hash(previous)
    assert policy["previous_campaign_settled_child_wall_s"]==966.7981451372616
    assert policy["shared_original_cap_child_wall_s"]==7200.
    assert policy["historical_charged_child_wall_s"]==pytest.approx(8.607537163188681+966.7981451372616)
    chain=policy["historical_charge_chain"]
    assert [r["campaign"] for r in chain]==[runner.LOCAL_RECOVERY_OUTPUT_NAME,runner.LOCAL_RECOVERY_RESTOREFIX_OUTPUT_NAME]
    assert [r["ledger_sha256"] for r in chain]==[runner.base.file_hash(original),runner.base.file_hash(previous)]
    with runner.old.ledger_transaction(output,policy=policy) as ledger:
        assert ledger["settled_child_wall_s"]==0. and ledger["jobs"]=={}
        assert ledger["cap_child_wall_s"]+ledger["historical_charged_child_wall_s"]==pytest.approx(7200.)
        assert ledger["historical_campaign_budget_reset"] is False and ledger["old_ledgers_reset"] is False
        assert ledger["historical_charge_chain"]==chain
    assert {p:p.read_bytes() for p in before}==before


@pytest.mark.parametrize("status",["running","unknown"])
def test_serializationfix_rejects_nonterminal_previous_jobs(tmp_path,monkeypatch,status):
    monkeypatch.setattr(Path,"home",classmethod(lambda cls:tmp_path))
    parent=tmp_path/"training_outputs/terminal"
    _,previous,saved=save_serialization_continuation_chain(parent)
    saved["jobs"]["stopped"]["status"]=status; save(previous,saved)
    with pytest.raises(ValueError,match="terminal"):
        runner.policy_for_output(parent/runner.LOCAL_RECOVERY_SERIALIZATIONFIX_OUTPUT_NAME)


def test_serializationfix_rejects_previous_cap_reset_or_exhaustion(tmp_path,monkeypatch):
    monkeypatch.setattr(Path,"home",classmethod(lambda cls:tmp_path))
    parent=tmp_path/"training_outputs/terminal"
    _,previous,saved=save_serialization_continuation_chain(parent)
    output=parent/runner.LOCAL_RECOVERY_SERIALIZATIONFIX_OUTPUT_NAME
    changed=dict(saved,cap_child_wall_s=7200.); save(previous,changed)
    with pytest.raises(ValueError,match="binding differs"):
        runner.policy_for_output(output)
    changed=dict(saved,settled_child_wall_s=saved["cap_child_wall_s"]); save(previous,changed)
    with pytest.raises(ValueError,match="exhausted"):
        runner.policy_for_output(output)


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


def future_action(hash_,aid,source,target,start,end,kind="TRANSFER"):
    return dict(trajectory_hash=hash_,aircraft_id=aid,kind=kind,source_ring=source,target_ring=target,start_time=start,end_time=end)


def future_callback(actions,*,snapshot=20.,effective=40.,reason=None):
    rows=[]; selected=[]
    for index,chain in enumerate(actions):
        first=chain[0]; key="candidate_"+first["trajectory_hash"]
        rows.append(dict(candidate_hash=key,aircraft_id=first["aircraft_id"],action_hashes=[a["trajectory_hash"] for a in chain],actions=chain,
            actual_solve_events=[dict(model_index=index)],selected_events=[dict(elapsed_s=1.)],certified_events=[dict(elapsed_s=2.)]))
        selected.append(dict(aircraft_id=first["aircraft_id"],candidate_hash=key,actions=chain,first_action=first,selected_kind="MOVE"))
    return dict(planning_timing=dict(snapshot_time=snapshot,effective_time=effective),service_candidate_lifecycle=rows,
        final_selected_first_actions=selected,delivery_certification_scope="effective_hold_then_absolute_first_action",
        actual_delivery_status="REJECTED" if reason else "COMMITTED",actual_delivery_reason=reason)


def reservation_event(type_,time,id_,action,*,effective=40.,**details):
    return dict(type="reservation_"+type_,time=time,aircraft_id=action["aircraft_id"],details=dict(
        reservation_id=id_,trajectory_hash=action["trajectory_hash"],effective_time=effective,
        start_time=action["start_time"],end_time=action["end_time"],**details))


def test_mixed_immediate_final_future_middle_inner_and_suffix_have_individual_event_chains():
    final=future_action("final","f","inner",None,40.,70.,"FINAL_NORMAL")
    move=future_action("move","m","middle","inner",80.,120.)
    suffix=future_action("forecast_only","m","inner",None,160.,190.,"FINAL_NORMAL")
    callbacks=[future_callback([[final],[move,suffix]])]
    events=[]
    for id_,action in (("rf",final),("rm",move)):
        events.extend([reservation_event("prepared",30.,id_,action),reservation_event("published",39.,id_,action),
            reservation_event("started",action["start_time"],id_,action),reservation_event("completed",action["end_time"],id_,action)])
    got=report.future_reservation_metrics(callbacks,events,{})
    assert got["all_first_actions"]["counts"]["published"]==2
    assert got["middle_inner"]["counts"]==dict(generated=1,actual_cp=1,selected_final=1,delivery_certified=1,published=1,started=1,completed=1,cancelled=0,completed_or_cancelled=1,pending_cancelled=0,superseded_pending=0)
    assert len(got["predicted_suffix_rows"])==1 and got["predicted_suffix_rows"][0]["trajectory_hash"]=="forecast_only"
    assert {r["aircraft_id"] for r in got["first_action_rows"]}=={"m","f"}
    assert next(r for r in got["reservation_rows"] if r["reservation_id"]=="rm")["published_times"]==[39.]
    assert got["replacement_count"]==got["absolute_start_postponement_count"]==0


def test_final_selected_unpublished_reports_occurrence_loss_without_attributing_callback_wait_to_action():
    move=future_action("move","m","middle","inner",80.,120.)
    callback=future_callback([[move]],reason="READY_DEADLINE_MISSED"); callback["empty_reason"]="REASONABLE_WAIT"
    callback["service_candidate_lifecycle"][0]["selected_events"]=[]
    got=report.future_reservation_metrics([callback],[],{})
    assert got["middle_inner"]["selected_any_intermediate_count"]==0
    assert got["middle_inner"]["counts"]["selected_final"]==1
    assert got["middle_inner"]["selected_final_not_published_count"]==1
    occurrence=got["final_selected_occurrence_rows"][0]
    assert occurrence["state"]=="not_published_by_window_end"
    assert occurrence["callback_delivery_observation"]["reason"]=="READY_DEADLINE_MISSED"
    assert got["first_action_rows"][0]["reasons"]==[]
    loss=next(r for r in got["middle_inner"]["transitions"] if r["from_stage"]=="selected_final")
    assert loss["exported_reason_counts"]=={} and loss["associated_callback_reason_counts"]["REASONABLE_WAIT"]==1


def test_published_outside_window_is_pending_not_missed():
    action=future_action("later","m","middle","inner",400.,440.)
    cb=future_callback([[action]],snapshot=300.,effective=320.)
    events=[reservation_event("prepared",314.,"r",action,effective=320.),reservation_event("published",319.,"r",action,effective=320.)]
    got=report.future_reservation_metrics([cb],events,{})
    assert got["pending_beyond_window_count"]==1
    assert got["due_without_observed_start_count"]==got["missed_start_count"]==0
    assert got["middle_inner"]["counts"]["started"]==0


def test_replacement_cancel_dedup_and_explicit_postponement():
    old=future_action("old","m","middle","inner",100.,140.)
    new=future_action("new","m","middle","inner",140.,180.)
    callbacks=[future_callback([[old]]),future_callback([[new]],snapshot=60.,effective=80.)]
    cancellation=reservation_event("cancelled",79.,"r_old",old,reason="atomic_replacement",replacement_reservation_id="r_new")
    events=[reservation_event("prepared",30.,"r_old",old),reservation_event("published",39.,"r_old",old),cancellation,cancellation,
        reservation_event("prepared",70.,"r_new",new,effective=80.),
        reservation_event("published",79.,"r_new",new,effective=80.,replaced_reservation_ids=["r_old"]),
        reservation_event("started",140.,"r_new",new,effective=80.),reservation_event("completed",180.,"r_new",new,effective=80.)]
    got=report.future_reservation_metrics(callbacks,events,{})
    assert got["replacement_count"]==got["absolute_start_postponement_count"]==got["cancelled_reservation_count"]==1
    assert got["due_without_observed_start_count"]==got["missed_start_count"]==0
    assert got["first_action_rows"][1]["reservation_reason_observations"] or got["first_action_rows"][0]["reservation_reason_observations"]


def test_repeated_selected_hash_keeps_occurrences_and_live_existing_reservation():
    action=future_action("same","m","middle","inner",100.,140.)
    callbacks=[future_callback([[action]]),future_callback([[action]],snapshot=60.,effective=80.),future_callback([[action]],snapshot=80.,effective=100.)]
    events=[reservation_event("prepared",30.,"r",action),reservation_event("published",39.,"r",action),
        reservation_event("cancelled",70.,"r",action,reason="RESERVATION_SOURCE_CHANGED")]
    got=report.future_reservation_metrics(callbacks,events,{})
    assert got["middle_inner"]["counts"]["generated"]==got["middle_inner"]["counts"]["published"]==1
    assert got["final_selected_occurrence_state_counts"]==dict(published_for_occurrence=1,already_reserved=1,not_published_by_window_end=1)
    assert got["final_selected_occurrence_rows"][-1]["callback_index"]==2
    assert got["missed_start_count"]==0


def test_late_start_guard_cancel_has_specific_reason_and_counts_miss_once():
    action=future_action("late","m","middle","inner",100.,140.)
    guard=reservation_event("start_guard",101.,"r",action,accepted=False,reason="RESERVATION_START_TIME_MISMATCH")
    cancel=reservation_event("cancelled",101.,"r",action,reason="RESERVATION_START_TIME_MISMATCH")
    events=[reservation_event("prepared",30.,"r",action),reservation_event("published",39.,"r",action),guard,cancel,cancel]
    got=report.future_reservation_metrics([future_callback([[action]])],events,{})
    assert got["missed_start_count"]==got["cancelled_reservation_count"]==1
    assert got["due_without_observed_start_count"]==0
    reasons=got["first_action_rows"][0]["reservation_reason_observations"]
    assert {r["event_type"] for r in reasons}=={"reservation_start_guard","reservation_cancelled"}
    assert all(r["scope"]=="reservation" for r in reasons)


def test_unclassified_metadata_visible_and_r3_future_fields_unknown():
    cb=dict(service_candidate_lifecycle=[dict(candidate_hash="c",aircraft_id="m",action_hashes=["unknown"],actual_solve_events=[])],final_selected_first_actions=[])
    got=report.future_reservation_metrics([cb],[],{})
    assert got["unclassified_first_action_count"]==1
    assert got["middle_inner"]["denominator_excludes_unclassified_count"]==1
    assert got["replacement_count"] is None
    assert report.future_reservation_metrics([dict(service_candidate_lifecycle=[])],[],{}) is None


def test_pending_discard_superseded_and_later_successful_publish_are_not_published_cancellations():
    pending=future_action("pending","p","middle","inner",80.,120.)
    superseded=future_action("superseded","s","middle","inner",400.,440.)
    live=future_action("live","l","middle","inner",100.,140.)
    first=future_callback([[pending]],reason="READY_DEADLINE_MISSED")
    trial=future_callback([[superseded]],snapshot=30.,effective=50.); trial["final_selected_first_actions"]=[]
    delivered=future_callback([[live]],snapshot=40.,effective=60.)
    binding=dict(objective=[0,-1,0,0,1,1],first_action_hashes=["live"],prefix_hashes=["live"],certified=True)
    delivered["delivery_selection_binding"]=deepcopy(binding); delivered["actual_delivery_selection_binding"]=deepcopy(binding)
    repeated=future_callback([[live]],snapshot=70.,effective=90.)
    discard=reservation_event("cancelled",38.,"rp",pending,previous_state="PENDING_DELIVERY",published=False,reason="READY_DEADLINE_MISSED")
    events=[reservation_event("prepared",30.,"rp",pending),discard,discard,
        reservation_event("prepared",35.,"rs",superseded,effective=50.),
        reservation_event("pending_superseded",36.,"rs",superseded,effective=50.,state="SUPERSEDED_PENDING",previous_state="PENDING_DELIVERY",published=False,reason="UNSELECTED_REPAIR"),
        reservation_event("prepared",45.,"rl",live,effective=60.),
        reservation_event("cancelled",46.,"rl",live,effective=60.,previous_state="PENDING_DELIVERY",published=False,reason="DISCARDED_CANDIDATE"),
        reservation_event("prepared",55.,"rl",live,effective=60.),reservation_event("published",59.,"rl",live,effective=60.),
        reservation_event("started",100.,"rl",live,effective=60.),reservation_event("completed",140.,"rl",live,effective=60.)]
    got=report.future_reservation_metrics([first,trial,delivered,repeated],events,{})
    assert got["cancelled_reservation_count"]==got["middle_inner"]["counts"]["cancelled"]==0
    assert got["pending_cancelled_reservation_count"]==2 and got["superseded_pending_reservation_count"]==1
    assert got["pending_beyond_window_count"]==got["due_without_observed_start_count"]==got["missed_start_count"]==0
    assert got["middle_inner"]["counts"]["published"]==got["middle_inner"]["counts"]["started"]==got["middle_inner"]["counts"]["completed"]==1
    live_occurrence=next(o for o in got["final_selected_occurrence_rows"] if o["callback_index"]==2)
    assert live_occurrence["delivery_selection_binding"]==live_occurrence["actual_delivery_selection_binding"]==binding
    assert next(o for o in got["final_selected_occurrence_rows"] if o["callback_index"]==3)["state"]=="already_reserved"


def test_future_metric_json_csv_agree(tmp_path,monkeypatch):
    monkeypatch.setattr(report,"PROJECT",tmp_path)
    action=future_action("m","a","middle","inner",80.,120.)
    metrics=report.future_reservation_metrics([future_callback([[action]])],[],{})
    value=dict(campaigns=dict(new=dict(input_root=str(tmp_path/"inputs"),metrics=[dict(root="s3r23",method="FULL_H",future_reservation_metrics=metrics)])))
    target=report.write_outputs(tmp_path/"artifacts/future",value)
    row=next(csv.DictReader((target/"comparison.csv").open(encoding="utf8")))
    saved=json.loads((target/"summary.json").read_text())["campaigns"]["new"]["metrics"][0]["future_reservation_metrics"]
    assert json.loads(row["future_reservation_metrics"])==saved==metrics


def test_saved_r3_baseline_selects_new_four_arms_and_marks_preexisting(tmp_path,monkeypatch):
    arms=[dict(root=r,method=m,status="COMPLETE",common_checkpoint_sha256=r) for r in report.ROOTS for m in report.METHODS]
    def campaign(name,value): return dict(input_root=str(tmp_path/name),ordinary=dict(arms=deepcopy(arms)),metrics=[dict(root=a["root"],method=a["method"],treatment_normal_touchdown_count=value) for a in arms])
    old=campaign(runner.SUPPLY_R2_OUTPUT_NAME,7); new=campaign(runner.LOCAL_RECOVERY_SERIALIZATIONFIX_OUTPUT_NAME,3)
    summary=tmp_path/"summary.json"; save(summary,dict(window=[20.,320.],campaigns=dict(old=old,new=new)))
    before=summary.read_bytes()
    monkeypatch.setattr(report,"existing_summary",lambda:SimpleNamespace(collect=lambda path:(dict(arms=deepcopy(arms)),[],[])))
    got=report.collect(None,tmp_path/runner.FUTURE_COMMITMENT_OUTPUT_NAME,old_summary=summary)
    assert {r["treatment_normal_touchdown_count"] for r in got["campaigns"]["old"]["metrics"]}=={3}
    assert all(r["evidence_type"]=="preexisting_artifact" and r["future_reservation_metrics"] is None for r in got["campaigns"]["old"]["metrics"])
    assert summary.read_bytes()==before
    duplicate=deepcopy(new); duplicate["metrics"][1]=deepcopy(duplicate["metrics"][0])
    save(summary,dict(window=[20.,320.],campaigns=dict(new=duplicate)))
    with pytest.raises(ValueError,match="four unique"): report.saved_baseline(summary)
    save(summary,dict(window=[20.,320.],campaigns=dict(new=old)))
    with pytest.raises(ValueError,match="not R2"): report.saved_baseline(summary)
