"""Small synthetic file fixtures; no world construction or advance."""
from copy import deepcopy
import csv
import importlib.util
import json
from pathlib import Path

import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("a4_numeric_summary",HERE/"summarize_common_comparison.py")
summary=importlib.util.module_from_spec(spec); spec.loader.exec_module(summary)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,allow_nan=False),encoding="utf8")


def checkpoint(path):
    from terminal_marl.core.rotating_slots.stage2_campaign import Stage2Codec,digest
    from terminal_marl.core.rotating_slots.selective_replanning.history import PlanningHistory,AgeEntry
    from terminal_marl.core.rotating_slots.world import AircraftState
    codec=Stage2Codec(); history=PlanningHistory(progress_policy="maneuver_started_v2")
    history.entries["a"]=AgeEntry(origin=0.,reviewed=100.,progressed=50.)
    history.last_observed_time=300.
    aircraft={aid:AircraftState(aid,status,.8,(0.,0.,0.),(0.,0.,0.),(0.,0.,0.),(0.,0.,0.))
              for aid,status in (("a","HOLD_INNER"),("b","HOLD_OUTER"),("c","LANDED_SERVICE"))}
    payload=dict(time=320.,world=codec.encode(dict(aircraft=aircraft)),
        scheduler=codec.encode(dict(callback=dict(selective_replanning=dict(history=history.state_dict())))))
    write(path,dict(payload=payload,sha256=digest(payload)))


@pytest.fixture
def files(tmp_path):
    root=tmp_path/"synthetic_inputs"
    for ancestry in summary.ROOTS:
        common=root/"common_initial"/ancestry/"common.checkpoint.json"
        write(common,dict(synthetic_common=ancestry))
        for method in summary.METHODS:
            home=root/"arms"/ancestry/method; d=home/"evaluation"/ancestry/method
            tasks=[dict(snapshot_time=20.,effective_time=40.,status="COMMITTED",submitted=2,finished_time=40.,
                        ready_time=35.7,next_started_time=40.,finish_to_next_start_s=0.),
                   dict(snapshot_time=60.,effective_time=80.,status="NO_EXECUTABLE_PREFIX",submitted=0,
                        finished_time=80.,reason="NO_EXECUTABLE_PREFIX"),
                   dict(snapshot_time=100.,effective_time=120.,status="COMPUTING",submitted=0)]
            cbs=[dict(time=40.,planning_timing=dict(snapshot_time=20.),deferred=True,noop=False,
                      actual_submitted_trajectory_hashes=["h-final"],fallback=False,
                      timings=dict(candidate_generation_and_single_vehicle=[3.,2.],trajectory_validation=[4.,3.]),
                      search_handoff=dict(first_nonempty_certified=dict(wall_elapsed_s=2.,prefix_hashes=["h-first"])),
                      selective_replanning=dict(scope=dict(status="LOCAL",actual_k=8)),
                      suffix_completion=dict(attempts=[dict(suffix_status="KNOWN",wall_s=.3,
                          complete_candidate_in_catalog=True,priority_slice_attempted=True,skip_reason=None)])),
                 dict(time=80.,planning_timing=dict(snapshot_time=60.),deferred=False,noop=True,
                      actual_submitted_trajectory_hashes=[],fallback=True,
                      timings=dict(candidate_generation_and_single_vehicle=[1.,.5],prefix_validation=[0.,0.]),
                      selective_replanning=dict(scope=dict(status="MANDATORY_OVERFLOW_FULL",actual_k=14),
                          overflow_cause="HARD_MANDATORY_GT_MAX12")),
                 dict(time=120.,planning_timing=dict(snapshot_time=100.),deferred=True,noop=False,
                      timings=dict(other_phase=[5.,6.]),
                      selective_replanning=dict(scope=dict(status="LOCAL",actual_k=9),ages=dict(a=dict(review_age=5.))))]
            work=[dict(snapshot_time=t,full_normal_planning_work_s=w,
                       observed_components=dict(normal_work_s=w-3.,selective_history_work_s=3.))
                  for t,w in ((20.,10.),(60.,20.),(100.,5.))]
            increment=dict(cumulative_delta=dict(L_normal=2,L_emergency=0,F_energy_total=0,failed_before_landing=0,
                energy_j=100.,energy_kwh=100./3600000.,holding_time=10.),
                ending_pending=dict(N_airborne_pending=2,N_upstream_pending=1,airborne_or_censored=2),
                inherited_final_touchdown_ids=["old"],treatment_final_touchdown_ids=["new"])
            life=dict(per_task_work=work,selective_history_work=dict(unattributed_service_s=3.),
                full_normal_planning_work=dict(total_s=38.),join_rate=1.,
                join_rate_denominator="finished nonempty proposed groups; pending and no-prefix excluded",
                finished_nonempty_groups=1,committed_nonempty_groups=1,actual_submitted_plans=2)
            write(d/"result.json",dict(status="COMPLETE",t0=20.,time=320.,completed_relative_s=300,target_relative_s=300,
                evaluation_profile="stitched_continuation_v1",increment=increment))
            write(d/"boundary_0300.json",dict(t0=20.,time=320.,evaluation_profile="stitched_continuation_v1",
                increment=increment,quality=dict(conservation=True),normal_lifecycle=life,
                holding_time_by_ring=dict(outer=10.),holding_energy_by_ring=dict(outer=30.),
                total_executed_energy_by_source_ring=dict(outer=100.),inherited_finals=[dict(aircraft_id="old")]))
            write(home/"resolved_run_binding.json",dict(evaluation_profile="stitched_continuation_v1",
                common_checkpoint_sha256=summary.digest_file(common)))
            write(d/"method_scheduler_records.json",cbs); write(d/"method_normal_tasks.json",tasks)
            write(d/"method_online_services.json",[dict(kind="selective_history",wall_s=12.)])
            checkpoint(d/"checkpoint_terminal.json")
            events=[dict(type="maneuver_started",time=t,aircraft_id="new",details=dict(trajectory_hash="h-final",maneuver_kind="TRANSFER"))
                    for t in (10.,40.)]
            (d/"replay").mkdir()
            (d/"replay/events.jsonl").write_text("\n".join(json.dumps(e) for e in events)+"\n",encoding="utf8")
    return root


def test_missing_formal_arm_stays_incomplete_and_null(files,tmp_path):
    missing=files/"arms/s3r24/RULE_LOCAL_K8/evaluation/s3r24/RULE_LOCAL_K8/result.json"
    missing.unlink()
    report,timeline,timings=summary.collect(files)
    assert report["status"]=="INCOMPLETE"
    arm=report["arms"][-1]
    assert arm["missing_inputs"]==[str(missing)] and "increment" not in arm
    output=tmp_path/"analysis_missing"
    summary.write_outputs(output,report,timeline,timings)
    with (output/"four_arm_summary.csv").open(encoding="utf8") as stream: rows=list(csv.DictReader(stream))
    assert rows[-1]["energy_j"]==rows[-1]["task_count"]=="null"
    with pytest.raises(FileExistsError): summary.write_outputs(output,report,timeline,timings)
    with pytest.raises(ValueError,match="separate"):
        summary.write_outputs(files/"analysis_bad",report,timeline,timings)


def test_rate_history_cost_timing_and_hash_fields(files,tmp_path):
    report,timeline,timings=summary.collect(files)
    assert report["status"]=="COMPLETE"
    arm=report["arms"][0]; n=arm["normal_planning"]
    assert n["committed_nonempty_count"]==1 and n["finished_task_count"]==2
    assert n["committed_nonempty_over_all_finished"]==.5
    assert n["existing_lifecycle"]["join_rate"]==1. and n["submitted_plan_total"]==2
    assert n["per_task_work_observed_sum"]["sum"]==35. and n["unattributed_history_work_s"]==3.
    assert n["existing_lifecycle"]["full_normal_planning_work"]["total_s"]==38.
    assert arm["timing_summary"]["candidate_generation_and_single_vehicle"]["wall"]==dict(n=2,sum=4.,mean=2.,max=3.)
    assert arm["timing_summary"]["prefix_validation"]["wall"]==dict(n=1,sum=0.,mean=0.,max=0.)
    assert arm["timing_summary"]["trajectory_validation"]["missing_callbacks"]==2
    first=timeline[0]
    assert first["first_nonempty_certified_wall_elapsed_s"]==2. and first["ready_time"]==35.7
    assert first["first_prefix_start_matches"]==[dict(trajectory_hash="h-first",starts=[])]
    matches=first["actual_starts"][0]["starts"]
    assert [x["within_method_window"] for x in matches]==[False,True]
    assert [x["matches_task_effective"] for x in matches]==[False,True]
    assert arm["completion"][0]["report"]["attempts"][0]["priority_slice_attempted"] is True
    assert any(row["phase"]=="other_phase" for row in timings)
    assert any(row["phase"]=="prefix_validation" and row["wall_s"] is None for row in timings)
    summary.write_outputs(tmp_path/"analysis_complete",report,timeline,timings)


def test_history_at_terminal_time_unmodified_and_profile_mismatch(files):
    path=files/"arms/s3r23/FULL_H/evaluation/s3r23/FULL_H/checkpoint_terminal.json"
    before=path.read_bytes(); ending=summary.ending_history(path)
    assert path.read_bytes()==before and ending["history_unmodified"]
    assert ending["absolute_time"]==320. and ending["history_last_observed_time"]==300.
    assert ending["airborne_count"]==2 and ending["known_review_ge60_count"]==1
    assert ending["unknown_review_count"]==1 and ending["max_known_review_age_s"]==220.
    assert ending["max_known_progress_age_s"]==270.
    p=files/"arms/s3r24/FULL_H/resolved_run_binding.json"
    row=summary.load(p); row["evaluation_profile"]="other"; write(p,row)
    report,_,_=summary.collect(files)
    assert report["status"]=="INCOMPLETE" and not report["same_evaluation_profile"]
    latest=report["arms"][0]["latest_callback_ages"][0]
    assert latest["age_basis_snapshot_time"]==100. and latest["ages"]["a"]["review_age"]==5.
    missing=files/"arms/s3r23/FULL_H/evaluation/s3r23/FULL_H/boundary_0300.json"
    row=summary.load(missing); del row["increment"]["cumulative_delta"]["energy_j"]; write(missing,row)
    report,_,_=summary.collect(files)
    assert report["arms"][0]["status"]=="INCOMPLETE"
    assert report["arms"][0]["missing_metric_fields"]==["increment.cumulative_delta.energy_j"]


def test_duplicate_match_and_missing_work_do_not_guess(files):
    d=files/"arms/s3r23/FULL_H/evaluation/s3r23/FULL_H"
    cbs=summary.load(d/"method_scheduler_records.json")
    cbs[1]["planning_timing"]["snapshot_time"]=20.; write(d/"method_scheduler_records.json",cbs)
    boundary=summary.load(d/"boundary_0300.json")
    boundary["normal_lifecycle"]["per_task_work"].pop(1); write(d/"boundary_0300.json",boundary)
    report,timeline,_=summary.collect(files)
    assert report["status"]=="INCOMPLETE"
    assert timeline[0]["callback_match"]=="DUPLICATE" and timeline[0]["callback_indices"]==[0,1]
    assert timeline[0]["actual_k"] is None and timeline[0]["final_submitted_hashes"] is None
    assert timeline[1]["work_match"]=="MISSING" and timeline[1]["full_normal_planning_work_s"] is None
    assert report["arms"][0]["callback_matching"]["issues"]


def test_restart_gap_preserves_incoming_and_computes_only_finite_outgoing():
    tasks=[dict(snapshot_time=20.,status="COMMITTED",finished_time=40.,next_started_time=43.,
                finish_to_next_start_s=0.,previous_finished_time=20.,previous_record_index=4),
           dict(snapshot_time=43.,status="COMPUTING",finish_to_next_start_s=0.,
                previous_finished_time=40.,previous_record_index=0),
           dict(snapshot_time=60.,status="REJECTED",finished_time=80.,next_started_time=float("inf"),
                finish_to_next_start_s=9.)]
    callbacks=[dict(planning_timing=dict(snapshot_time=t["snapshot_time"])) for t in tasks]
    work=[dict(snapshot_time=t["snapshot_time"]) for t in tasks]
    rows,checks=summary.task_rows(tasks,callbacks,work,[],"s3r23","FULL_H",20.,320.)
    assert not any(checks.values())
    assert rows[0]["raw_restart_gap_s"]==0. and rows[0]["finish_to_next_start_s"]==3.
    assert rows[0]["previous_finished_time"]==20. and rows[0]["previous_record_index"]==4
    assert rows[1]["raw_restart_gap_s"]==0. and rows[1]["finish_to_next_start_s"] is None
    assert rows[1]["previous_finished_time"]==40. and rows[1]["previous_record_index"]==0
    assert rows[2]["raw_restart_gap_s"]==9. and rows[2]["finish_to_next_start_s"] is None
