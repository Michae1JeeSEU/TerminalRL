"""Saved-artifact report fixtures; no world advance, solve, fitting or inference."""
from copy import deepcopy
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from scripts.stage3 import report_latency_interface as report


def save(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,allow_nan=False),encoding="utf8")


def fixture(output):
    scope=dict(source_commit="fixture-commit",source_hash="fixture-source",shared_manifest_hash="manifest",initial_state_hash="initial",
               model_fingerprint={"original":"unchanged"},protocol=dict(target_sim_seconds=10))
    save(output/"execution_scope.json",scope)
    save(output/"summary.json",dict(common_completed_boundary_s=10))
    save(output/"roots/s3r23/root.json",dict(manifest_hash="manifest",initial_state_hash="initial"))
    for arm in report.ARMS:
        tasks=[dict(snapshot_time=0.,effective_time=5.,ready_deadline=4.,result_time=2.,ready_time=3.,finished_time=5.,
                    status="COMMITTED",submitted=1,normal_work_s=6.,prepare_work_s=2.,ready_guard_work_s=.5,
                    ready_registration_bookkeeping_s=.1,publication_bookkeeping_wall_s=.1,normal_evidence_wall_s=.1),
               dict(snapshot_time=6.,effective_time=10.,ready_deadline=9.,status="COMPUTING" if arm==report.ARMS[0] else "PREPARED",
                    normal_work_s=20.,submitted=0)]
        if arm==report.ARMS[1]: tasks[1]["ready_time"]=8.
        callbacks=[dict(planning_timing={"snapshot_time":t,"effective_time":5. if t==0. else 10.},deferred=True,noop=False,
                        actual_delivery_status="COMMITTED" if t==0. else None,actual_submitted_trajectory_hashes=["saved-plan"] if t==0. else []) for t in (0.,6.)]
        q=dict(L_normal=1,L_emergency=0,F_energy_total=0,N_airborne_pending=2,N_admitted_total=3,E_terminal_J=100.,holding_time=5.)
        terminal=dict(status="OBSERVED",observed_time_s=10.,quality=q)
        base=output/"evaluation"/arm
        save(base/"normal_tasks.json",tasks); save(base/"scheduler_records.json",callbacks)
        save(base/"dispatch_timings.json",[dict(dispatch_wall_s=2.,dispatch_cpu_s=1.,normal_wall_s=1.)])
        save(base/"online_services.json",[dict(kind="monitor",wall_s=.5,cpu_s=.3)])
        for t in (5,10): save(base/f"boundary_{t:04d}.json",dict(time=t,quality=q))
        save(base/"terminal_observation.json",terminal)
        save(base/"result.json",dict(status="EVALUATION_PARTIAL",completed_boundary_s=10,quality=q,
            terminal_observation=terminal,shared_manifest_hash="manifest",initial_state_hash="initial",
            model_fingerprint=scope["model_fingerprint"],execution_scope=scope,actual_cpu=[12],full_job_wall_s=23.,
            costs=dict(full_online_wall_s=20.),terminal_costs=dict(full_online_wall_s=21.,world_advance_wall_s=18.),
            all_callbacks=dict(catalogs=[dict(large_payload="do not copy")],coverage_by_callback=["do not copy"],rank_learned=3)))
        save(output/"receipts"/(arm+".json"),dict(status="completed",exit_code=0,wall_s=24.))


def test_saved_report_preserves_inputs_and_separates_pending_work_and_readiness(tmp_path):
    pytest.importorskip("matplotlib")
    fixture(tmp_path)
    before={p:p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    report.main(["--output",str(tmp_path)])
    assert all(p.read_bytes()==value for p,value in before.items())
    assert all((tmp_path/name).is_file() for name in report.OUTPUTS)
    value=report.load(tmp_path/"analysis_summary.json")
    h,l=(value["cells"][arm] for arm in report.ARMS)
    assert h["normal_lifecycle"]["all_task_observed_work"]["total_s"]==pytest.approx(28.3)
    assert h["normal_lifecycle"]["all_task_observed_work"]["p95_s"]==20.
    assert h["normal_lifecycle"]["finished_task_complete_work"]["p95_s"]==pytest.approx(8.3)
    assert h["normal_lifecycle"]["pending_past_ready_deadline"]==1
    assert l["normal_lifecycle"]["pending_past_ready_deadline"]==0
    assert l["normal_lifecycle"]["pending_nonempty"]==1 and l["normal_lifecycle"]["join_rate"]==1.
    assert h["replay"]["status"]=="UNAVAILABLE_REPLAY" and value["identity_checks"]["same_actual_normal_start_hashes"] is None
    assert h["normal_lifecycle"]["revalidation"]["availability"]=="UNAVAILABLE_LEGACY_FIELDS"
    assert h["normal_lifecycle"]["revalidation"]["certified_rounds"] is None
    assert "catalogs" not in h["all_callbacks"]
    with (tmp_path/"boundary_quality.csv").open() as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==4 and {float(r["time_s"]) for r in rows}=={5.,10.}
    text=ET.parse(tmp_path/"planning_timeline.svg").getroot().findall(".//{http://www.w3.org/2000/svg}text")
    assert len(text)>10 and value["figure_qa"]["export_dpi"]==300
    assert any("lead unavailable s" in (element.text or "") for element in text)
    assert value["human_replay_acceptance"]=="HUMAN_PENDING"
    text=(tmp_path/"FINAL_REPORT.md").read_text(encoding="utf8")
    assert "PENDING_PRIMARY_REVIEW" in text
    assert "Search handoff trace: unavailable" in text
    assert "| Observed metric | H | L |\n| --- | ---: | ---: |" in text
    with pytest.raises(ValueError,match="already exist"): report.main(["--output",str(tmp_path)])


def test_saved_replay_group_start_and_pva_checks_do_not_invent_missing_proposals(tmp_path):
    root=tmp_path/"replay"; root.mkdir()
    plans={h:dict(kind="STABLE" if h=="old" else "TRANSFER",start_time=0. if h=="old" else 5.,
                  saved_p=[0.,0.,0.],saved_v=[1.,0.,0.],saved_a=[0.,0.,0.]) for h in ("old","p1","p2")}
    save(root/"trajectories.json",plans)
    starts=[dict(type="maneuver_started",time=5.,aircraft_id=aid,details=dict(trajectory_hash=h,maneuver_kind="TRANSFER"))
            for aid,h in (("a","p1"),("b","p2"))]
    (root/"events.jsonl").write_text("".join(json.dumps(r)+"\n" for r in starts))
    frames=[dict(time=t,aircraft=[dict(id=aid,status="HOLD_OUTER" if t==0. else "TRANSFER_O_M",trajectory_hash="old" if t==0. else h,
                p=[0.,0.,0.],v=[1.,0.,0.],a=[0.,0.,0.]) for aid,h in (("a","p1"),("b","p2"))]) for t in (0.,5.,6.)]
    (root/"states.jsonl").write_text("".join(json.dumps(r)+"\n" for r in frames))
    def index():
        save(root/"replay.json",dict(code_commit="fixture",trajectory_hashes=sorted(plans),
            files={name:report.file_hash(root/name) for name in ("trajectories.json","events.jsonl","states.jsonl")}))
    index()
    # A test-only decoder reads vectors from this tiny serialized fixture. The
    # production decoder is frozen TrajectoryPlan.from_dict / ReplayBundle.validate.
    core=dict(validate=lambda p:object(),config=lambda p:None,
        decode=lambda raw,c:SimpleNamespace(state=lambda t:SimpleNamespace(p=raw["saved_p"],v=raw["saved_v"],a=raw["saved_a"])))
    callbacks=[dict(planning_timing=dict(snapshot_time=0.,effective_time=5.),actual_delivery_status="COMMITTED",
                    actual_submitted_trajectory_hashes=["p1","p2"]),
               dict(planning_timing=dict(snapshot_time=6.,effective_time=10.),actual_delivery_status="REJECTED",actual_submitted_trajectory_hashes=[],submitted=0)]
    value,errors,_=report.audit_replay(tmp_path,callbacks,core)
    assert value["file_hashes_match"] and value["checked_pva_samples"]==6
    assert value["max_pva_errors"]==dict(p=0.,v=0.,a=0.) and len(errors)==6
    assert value["committed_group_checks"][0]["complete_same_time"] is True
    assert value["committed_group_checks"][1]["status"]=="UNAVAILABLE_PROPOSED_HASHES"
    assert value["normal_start_coverage"]["unattributed_hashes"]==[]
    starts[1]["time"]=6.
    (root/"events.jsonl").write_text("".join(json.dumps(r)+"\n" for r in starts)); index()
    value,_,_=report.audit_replay(tmp_path,callbacks,core)
    assert value["start_time_mismatch_count"]==1 and not value["committed_group_checks"][0]["complete_same_time"]


def test_saved_revalidation_multiple_joins_and_pending_are_observed_not_assumed(tmp_path):
    fixture(tmp_path)
    for arm in report.ARMS:
        base=tmp_path/"evaluation"/arm
        tasks=report.load(base/"normal_tasks.json")
        event=dict(started_time=2.,due=3.,completed_time=3.,status="CERTIFIED",reason="NEW_TRAFFIC",
                   affected_aircraft_ids=["new-a"],pairs_checked=4,reused_pairs=12,full_fallback=False,wall_s=.6,cpu_s=.5)
        tasks[0].update(submitted=13,revalidation_work_s=.6,revalidation_work_cpu_s=.5,dependency_guard_work_s=.3,
                       dependency_guard_work_cpu_s=.25,revalidation_count=1,revalidations=[event])
        tasks.insert(1,dict(snapshot_time=3.,effective_time=8.,ready_deadline=7.,ready_time=6.,finished_time=8.,
                            status="COMMITTED",submitted=2,revalidation_count=0,revalidations=[]))
        tasks[-1].update(status="REVALIDATING",revalidation_work_s=.2,revalidation_count=1,
            revalidations=[dict(started_time=9.,due=10.,status="GUARDING",geometry_completed_time=9.5,guard_delivery_due=10.,reason="NEW_TRAFFIC",
                affected_aircraft_ids=["new-b"],pairs_checked=1,reused_pairs=3,full_fallback=True,wall_s=.2,cpu_s=.1)])
        callbacks=report.load(base/"scheduler_records.json")
        callbacks[0].update(cp_continuation=[dict(state_reused=True,state_invalidated=False,incumbent_source="CALLER",
            caller_incumbent_accepted=True,pair_cuts_reused=7,single_cuts_reused=2,prefix_nogoods_reused=1,
            single_results_reused=4,solver_internal_state_reused=False)],
            rank_batch_cache=dict(hits=3,miss_reasons={"MEMBERSHIP_CHANGED":2},max_budget_fraction_drift=.08),
            ready_budget=dict(planning_wall_deadline=29.,remaining_at_start_s=28.,remaining_at_return_s=17.))
        callbacks.insert(1,dict(planning_timing=dict(snapshot_time=3.),deferred=True,noop=False))
        save(base/"normal_tasks.json",tasks);save(base/"scheduler_records.json",callbacks)
        save(base/"online_services.json",[dict(kind="dependency_guard",wall_s=.3,cpu_s=.25)])
    value,timeline,*_=report.collect(tmp_path)
    life=value["cells"][report.ARMS[0]]["normal_lifecycle"]
    assert life["actual_joined_groups"]==2 and life["actual_submitted_plans"]==15
    assert life["submitted_plans_by_joined_group"]==[13,2]
    assert life["finished_nonempty"]==2 and life["pending_nonempty"]==1 and life["join_rate"]==1.
    assert life["all_task_observed_work"]["total_s"]==pytest.approx(29.4)
    assert life["finished_task_complete_work"]["total_s"]==pytest.approx(9.2)
    assert life["components"]["dependency_guard_work_s"]["total_s"]==pytest.approx(.3)
    reval=life["revalidation"]
    assert reval["certified_rounds"]==1 and reval["pending_rounds"]==1
    assert reval["pairs_checked"]==5 and reval["reused_pairs"]==15 and reval["full_fallback_rounds"]==1
    assert reval["affected_aircraft_ids"]==["new-a","new-b"]
    pending=next(r for r in timeline if r["status"]=="REVALIDATING")
    assert pending["revalidations"][0].get("completed_time") is None
    text=report.report_text(value,timeline)
    assert "| Actual joined groups / submitted plans | 2 / 15 | 2 / 15 |" in text
    assert "| Checked pairs | 5 | 5 |" in text and "| Reused pairs | 15 | 15 |" in text
    assert "| Prepared-packet dependency inspection (s) | 0.3 | 0.3 |" in text
    assert "| Reused pair cuts | 7 | 7 |" in text and "| Comparable group cache hits | 3 | 3 |" in text
    assert "| Maximum cached budget-fraction drift | 0.08 | 0.08 |" in text
    assert value["cells"][report.ARMS[0]]["search_reuse"]["cp"]["solver_internal_state_reused"]==0
    assert "completed unavailable" in text and "observed joining bottleneck" not in text
    assert report.search_reuse_summary([{}])["cp"]["pair_cuts_reused"] is None
    assert report.search_reuse_summary([{}])["rank_cache_hits"] is None
    _,rejected=report.task_facts([dict(snapshot_time=0.,status="REJECTED",finished_time=2.,reason="REVALIDATION_WORK_DEADLINE")],[],2.)
    assert rejected["revalidation_work_deadline_rejections"]==1 and rejected["ready_deadline_rejections"]==0


@pytest.mark.parametrize("lead",[20.,30.,None])
def test_report_timing_scope_uses_recorded_lead_and_missing_is_unavailable(tmp_path,lead):
    fixture(tmp_path)
    scope=report.load(tmp_path/"execution_scope.json")
    if lead is not None:
        scope["protocol"].update(lead_seconds=lead,preparation_margin=1.,scheduler_period=5.)
    save(tmp_path/"execution_scope.json",scope)
    value,timeline,*_=report.collect(tmp_path)
    text=report.report_text(value,timeline)
    if lead is None:
        assert "lead_seconds=unavailable s; preparation_margin=unavailable s; scheduler_period=unavailable s" in text
        assert "recorded lead - margin = unavailable s" in text
        assert "lead_seconds=30" not in text
    else:
        assert f"lead_seconds={lead:g} s; preparation_margin=1 s; scheduler_period=5 s" in text
        assert f"recorded lead - margin = {lead-1:g} s" in text
    assert "Ready deadline = snapshot time + lead - margin" in text


def test_saved_search_certificate_return_and_final_authorization_are_distinct(tmp_path):
    fixture(tmp_path)
    scope=report.load(tmp_path/"execution_scope.json")
    scope["protocol"].update(lead_seconds=20.,preparation_margin=1.,scheduler_period=5.,
        search_handoff_reserve_seconds=4.,search_handoff_protocol="search_handoff_reserve_v1")
    save(tmp_path/"execution_scope.json",scope)
    trace=dict(protocol="search_handoff_reserve_v1",reserve_seconds=4.,planning_wall_origin=100.,
        planning_wall_deadline=119.,search_wall_deadline=115.,first_nonempty_certified=dict(wall_elapsed_s=2.),
        events=[dict(event="STRICT_CERTIFIED_IMPROVEMENT",wall_elapsed_s=9.)],
        callback_return=dict(wall_elapsed_s=14.5,certified=True),heavy_checks={"validation":dict(started=1,skipped=2)})
    for arm in report.ARMS:
        base=tmp_path/"evaluation"/arm
        tasks=report.load(base/"normal_tasks.json");tasks[0].update(search_handoff=trace,preparation_started_time=3.)
        save(base/"normal_tasks.json",tasks)
    value,timeline,*_=report.collect(tmp_path)
    text=report.report_text(value,timeline)
    assert "Search handoff reserve: 4 s; protocol: search_handoff_reserve_v1" in text
    assert "actual search wall deadline 115 (perf-clock s), offset from planning origin 15 s" in text
    assert "First nonempty search certificate at wall elapsed 2 s" in text
    assert "strict certified improvements at wall elapsed [9.0] s" in text
    assert "callback return at wall elapsed 14.5 s, certified True" in text
    assert "Actual preparation start 3" in text and "search certificate is not final authorization" in text
    assert timeline[0]["search_handoff"]==trace and timeline[0]["preparation_started_time"]==3.
