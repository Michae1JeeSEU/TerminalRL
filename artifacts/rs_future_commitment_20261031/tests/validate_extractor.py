"""Synthetic schema checks only; no experiment or world execution."""
from copy import deepcopy
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("r4_extract",ROOT/"extract_compact_results.py")
extractor=importlib.util.module_from_spec(spec); spec.loader.exec_module(extractor)
fixture=ROOT/"tests"/(sys.argv[1] if len(sys.argv)>1 else "extractor_validation_20261031")
summary=fixture/"inputs/final_report/summary.json"; ledger_path=fixture/"inputs/ledger.json"
summary.parent.mkdir(parents=True,exist_ok=False)


def unit(id_,start,*,cancelled=False,starts=None,effective=40.,published=True):
    return dict(reservation_id=id_,trajectory_hash="hash_"+id_,aircraft_id="aircraft_"+id_,
        effective_time=effective,start_time=start,end_time=(start+40 if start is not None else None),published=published,
        published_times=[39.] if published else [],started_times=starts or [],completed_times=([t+40 for t in starts or []]),
        cancelled=cancelled,cancelled_times=[70.] if cancelled else [],cancellation_reasons=["cancel"] if cancelled else [])


timely=unit("timely",80.,starts=[80.])
future=dict(schema="future_reservation_metrics_v1",reservation_rows=[timely,deepcopy(timely),unit("cancel",100.,cancelled=True),
    unit("outside",400.,effective=300.),unit("missing",200.),unit("late",90.,starts=[90.1]),
    unit("immediate",40.,starts=[40.]),unit("unknown",None)],first_action_rows=[dict(action=None)],predicted_suffix_rows=[],
    final_selected_occurrence_rows=[dict(callback_index=1,trajectory_hash="lost",first_action_middle_inner=True,
        state="not_published_by_window_end",callback_delivery_observation=dict(reason="READY_DEADLINE_MISSED")),
        dict(callback_index=2,trajectory_hash="unfinished",state="not_published_by_window_end",callback_delivery_observation=dict(reason=None)),
        dict(callback_index=3,trajectory_hash="future_pending",state="pending_effective_outside_window")],
    final_selected_occurrence_state_counts=dict(not_published_by_window_end=2,pending_effective_outside_window=1),
    unclassified_first_action_count=1,malformed_event_rows=[],replacement_count=1,absolute_start_postponement_count=1,
    all_first_actions=dict(counts=dict(published=7)),middle_inner=dict(counts=dict(published=5)))
arms=[dict(root=r,method=m,status="COMPLETE",t0=20.,time=320.,common_checkpoint_sha256="common_"+r,
    normal_planning=dict(committed_nonempty_count=1,finished_task_count=2,existing_lifecycle=dict(finished_nonempty_groups=1,join_rate=1.)))
    for r,m in sorted(extractor.EXPECTED)]
value=dict(schema="service_opportunity_comparison_v1",window=[20.,320.],campaigns={})
jobs={}
for label in ("old","new"):
    metrics=[]; timeline=[]
    for arm in arms:
        r,m=arm["root"],arm["method"]
        metrics.append(dict(root=r,method=m,treatment_normal_touchdown_count=1,inherited_final_touchdown_count=2,
            increment_energy_kwh=3.,future_reservation_metrics=deepcopy(future) if label=="new" else None,
            saved_audit_result=dict(status="PASS",wall_s=.2)))
        timeline.append(dict(root=r,method=m,callback_indices=[2],status="COMPUTING",finished_time=None))
        for stage,wall in (("arm",1.5),("audit",.5)): jobs["service_"+stage+"_"+r+"_"+m]=dict(status="completed",child_wall_s=wall)
    value["campaigns"][label]=dict(input_root="synthetic_"+label,ordinary=dict(status="COMPLETE",arms=deepcopy(arms),task_timeline=timeline),metrics=metrics)
summary.write_text(json.dumps(value),encoding="utf8")
ledger_path.write_text(json.dumps(dict(cap_child_wall_s=10.,settled_child_wall_s=8.,jobs=jobs)),encoding="utf8")
before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in (summary,ledger_path)}
output=fixture/"fresh_results"; compact=extractor.run(summary,output,ledger_path)
assert len(compact["rows"])==8
new=next(r for r in compact["rows"] if r["campaign"]=="new"); old=next(r for r in compact["rows"] if r["campaign"]=="old")
r=new["future_reservation_review"]
assert (r["published_future_count"],r["published_future_started_count"],r["published_future_timely_started_count"],r["published_future_mistimed_started_count"])==(5,2,1,1)
assert (r["published_future_cancelled_count"],r["published_future_pending_outside_count"],r["published_future_unexplained_due_absent_count"])==(1,1,1)
assert r["published_missing_future_time_metadata_count"]==1 and r["selected_unpublished_callback_group_reason_unknown_count"]==1
assert r["selected_unpublished_occurrences"][1]["task_observation"]=="task unfinished at observation end"
assert new["audit_internal_wall_s"]==.2 and new["audit_charged_child_wall_s"]==.5
assert old["future_reservation_metrics"] is None and old["evidence_type"]=="preexisting_artifact" and new["evidence_type"]=="newly_run"
details=json.loads((output/"future_reservation_details.json").read_text())
assert details["arms"]["new/"+new["root"]+"/"+new["method"]]==future
csv_rows=list(csv.DictReader((output/"compact_metrics.csv").open(encoding="utf8")))
for row,saved in zip(csv_rows,compact["rows"]):
    assert all((row[k]==v if isinstance(v,str) else json.loads(row[k])==v) for k,v in saved.items())
assert before=={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in before}
try: extractor.run(summary,output,ledger_path)
except ValueError: pass
else: raise AssertionError("overwrite was accepted")
bad=deepcopy(value); bad["campaigns"]["new"]["ordinary"]["status"]="INCOMPLETE"
bad_path=summary.with_name("incomplete.json"); bad_path.write_text(json.dumps(bad),encoding="utf8")
try: extractor.run(bad_path,fixture/"rejected_output",ledger_path)
except ValueError: pass
else: raise AssertionError("incomplete report was accepted")
assert not (fixture/"rejected_output").exists()
print("Synthetic checks passed: 8 unique arms, lifecycle counts, missing metadata, unknown reasons, task scope, audit wall distinction, full telemetry, JSON/CSV, read-only inputs, exclusive output, incomplete rejection.")
