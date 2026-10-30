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


def test_smoke_is_twenty_seconds_separate_output(monkeypatch,tmp_path):
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
        yield dict(jobs={"service_smoke_s3r23_FULL_H":dict(status="running",parent_pid=runner.os.getppid())})
    monkeypatch.setattr(runner.old,"ledger_transaction",ledger)
    captured={}
    def arm(output,root,method,checkpoint,binding,**kwargs):
        captured.update(output=output,binding=binding,**kwargs)
        return dict(status="COMPLETE",target_relative_s=kwargs["episode_seconds"])
    monkeypatch.setattr(runner.comparison,"run_arm",arm)
    result=runner.main(["--stage","smoke","--output",str(tmp_path/"new"),"--root","s3r23","--method","FULL_H",
        "--cpu","16","--source-freeze",str(freeze),"--_child-job-id","service_smoke_s3r23_FULL_H"])
    assert result["target_relative_s"]==20
    assert captured["output"]==tmp_path/"new/smoke_only"
    assert captured["binding"]["run_role"]=="SMOKE_NOT_FORMAL"


def test_output_json_csv_grammar_and_missing_metrics(tmp_path,monkeypatch):
    monkeypatch.setattr(report,"PROJECT",tmp_path)
    metric=report.extract_metrics(tmp_path/"absent","s3r23","FULL_H",dict(increment=dict(
        cumulative_delta=dict(L_normal=3),inherited_final_touchdown_ids=["inherited"],treatment_final_touchdown_ids=["a","b"])))
    assert metric["normal_touchdown_delta"]==3 and metric["inherited_final_touchdown_count"]==1
    assert metric["treatment_final_touchdown_count"]==2 and metric["physical_idle"] is None
    value=dict(schema="service_opportunity_comparison_v1",campaigns=dict(new=dict(input_root=str(tmp_path/"inputs"),metrics=[metric])))
    output=report.write_outputs(tmp_path/"artifacts/report",value)
    assert json.loads((output/"summary.json").read_text())["campaigns"]["new"]["metrics"][0]["physical_idle"] is None
    assert "null" in (output/"comparison.csv").read_text()
    with pytest.raises(FileExistsError): report.write_outputs(output,value)
