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
    assert "catalogs" not in h["all_callbacks"]
    with (tmp_path/"boundary_quality.csv").open() as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==4 and {float(r["time_s"]) for r in rows}=={5.,10.}
    text=ET.parse(tmp_path/"planning_timeline.svg").getroot().findall(".//{http://www.w3.org/2000/svg}text")
    assert len(text)>10 and value["figure_qa"]["export_dpi"]==300
    assert value["human_replay_acceptance"]=="HUMAN_PENDING"
    text=(tmp_path/"FINAL_REPORT.md").read_text(encoding="utf8")
    assert "PENDING_PRIMARY_REVIEW" in text
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
