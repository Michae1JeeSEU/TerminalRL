"""Saved report semantics without a simulation, solver, or temporary directory."""
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.stage3 import report_selective_reoptimization_repair_v3 as report


def test_conflict_retains_hash_interval_and_missing_witness_without_distance_invention():
    raw=dict(reason="PAIR_UNKNOWN",details=dict(pair_evidence=dict(check_interval=[20.,40.],
        participants=[dict(aircraft_id="a",candidate_hash="ca",trajectory_hashes=["ta"])])))
    result=report.conflict_facts({"calls":[{"pair_conflict_evidence":[raw]}]}, {"a":7})
    row=result["rows"][0]
    assert row["raw"]==raw and row["participants"][0]["uid"]==7
    assert row["participants"][0]["candidate_hash"]=="ca"
    assert row["check_interval"]==[20.,40.]
    assert row["witness"]["distance_m"] is None
    assert "minimum_distance_m" not in row
    assert report.conflict_facts({}, {})["status"]=="UNAVAILABLE_NOT_RECORDED"


def test_p2_approach_occupancy_touchdown_are_distinct_and_future_completion_censored():
    events=[dict(type="pad_service_completed",aircraft_id="ground",time=30.)]
    states=[dict(time=30.,aircraft=[dict(id="ground",pad_id=2)],pads=[]),
        dict(time=46.,aircraft=[dict(id="ground",pad_id=2),dict(id="a",pad_id=2,status="FINAL_NORMAL")],
            pads=[dict(resource=2,aircraft_id="a",committed=True,start=45.,end=80.,kind="final",plan_hash="h")])]
    actions=[dict(pad_id=2,start_event_index=2,start_event_time=35.,aircraft_id="a",trajectory_hash="h",
        completion_type="touchdown",completion_time=60.)]
    row=report.p2_facts(events,states,actions,50.)["rows"][0]
    assert row["actual_start_gap_s"]==5. and row["occupancy_gap_s"]==15.
    assert row["touchdown"] is None and row["touchdown_censored"]
    assert row["touchdown_censor_time"]==50.
    row=report.p2_facts(events,states,actions,70.)["rows"][0]
    assert row["touchdown_gap_s"]==30. and not row["touchdown_censored"]


def test_p2_already_started_approach_is_separate_from_next_new_start():
    events=[dict(type="pad_service_completed",aircraft_id="ground",time=30.)]
    states=[dict(time=46.,aircraft=[dict(id="ground",pad_id=2),dict(id="a",pad_id=2)],
        pads=[dict(resource=2,aircraft_id="a",committed=True,start=45.,kind="final",plan_hash="h")])]
    actions=[dict(pad_id=2,start_event_index=2,start_event_time=25.,aircraft_id="a",trajectory_hash="h",
        completion_type="touchdown",completion_time=60.)]
    row=report.p2_facts(events,states,actions,70.)["rows"][0]
    assert row["actual_start"] is None and row["actual_start_censored"]
    assert row["occupancy_gap_s"]==15. and row["touchdown_gap_s"]==30.
    assert row["approaches_already_underway_at_service_end"][0]["trajectory_hash"]=="h"
    assert row["approaches_already_underway_at_service_end"][0]["start_event_time"]==25.


def test_diagnostic_pairing_requires_sha_and_complete_equal_timing_not_version_hash():
    timing=dict(snapshot_time=40.,effective_time=60.,ready_deadline=59.)
    old=dict(root="r",path="old",checkpoint_sha256="sha",timing=timing,physical_source_version="oldv")
    new=dict(root="r",path="new",checkpoint_sha256="sha",timing=timing,physical_source_version="newv")
    pair=report.pair_diagnostics([old],[new])[0]
    assert pair["status"]=="MATCHED"
    assert pair["version_observations"][0]["content_changed"] is None
    for changed in (dict(new,checkpoint_sha256="different"),dict(new,timing=None),
                    dict(new,timing=dict(timing,effective_time=61.))):
        assert report.pair_diagnostics([old],[changed])[0]["status"]=="UNMATCHED"


def test_actual_and_probe_payloads_remain_cold_boundary_diagnostics():
    directory=Path("new/snapshots/r/case")
    actual=dict(checkpoint_observation=dict(checkpoint_sha256="sha",boundary_minus_historical_source_s=2.,
        historical_input_recovered=False),cold_normal_cache=True,actual_catalog={"candidates":[{"candidate_hash":"h"}]},
        treatment=dict(selective_rule_variant="opportunity_executable_v3"),
        scheduler_records=[dict(planning_timing={"snapshot_time":42.,"effective_time":62.,"ready_deadline":61.})])
    probes=dict(results=[dict(label="a",result={"accepted":False})])
    def load(path, default=None):
        return {"actual_diagnostic.json":actual,"probe_results.json":probes}.get(Path(path).name,default)
    with patch.object(Path,"glob",return_value=[directory]),patch.object(Path,"is_dir",return_value=True), \
         patch.object(report.saved,"load",side_effect=load):
        row=report.diagnostic_inputs("new")[0]
    assert row["actual_diagnostic"]==actual and row["probes"]==probes
    assert row["boundary_delta_s"]==2. and row["cold_normal_cache"]
    assert not row["historical_input_recovered"]
    assert row["catalog_basis"]=="COLD_DIAGNOSTIC_CATALOG_NOT_HISTORICAL_CATALOG"
    assert row["variant"]=="opportunity_executable_v3"


def test_current_root_variants_pair_but_preexisting_reference_never_participates():
    timing=dict(snapshot_time=42.,effective_time=62.,ready_deadline=61.)
    def diagnostic(path,variant,sha="sha",stamp=timing):
        return dict(root="r",path=path,variant=variant,checkpoint_sha256=sha,timing=stamp)
    v2=diagnostic("new/v2","opportunity_cooperative_v2")
    v3=diagnostic("new/v3","opportunity_executable_v3")
    reference=diagnostic("old/v2","opportunity_cooperative_v2")
    def inputs(path): return [v2,v3] if str(path)=="new" else [reference]
    with patch.object(report,"diagnostic_inputs",side_effect=inputs):
        result=report.build_summary("new","old",roots=())["diagnostics"]
    assert result["pairs"][0]["status"]=="MATCHED"
    assert result["pairs"][0]["old_paths"]==["new/v2"]
    assert result["preexisting_reference"]==[reference]
    assert result["current_v2"]==[v2] and result["current_v3"]==[v3]
    for unmatched in (dict(v3,checkpoint_sha256="other"),dict(v3,timing=dict(timing,effective_time=63.))):
        def changed_inputs(path): return [v2,unmatched] if str(path)=="new" else [reference]
        with patch.object(report,"diagnostic_inputs",side_effect=changed_inputs):
            pairs=report.build_summary("new","old",roots=())["diagnostics"]["pairs"]
        assert pairs[0]["status"]=="UNMATCHED" and pairs[0]["old_paths"]==[]


def test_cell_reuses_saved_statistics_and_decisions_instead_of_reimplementing_them():
    statistics=dict(normal_lifecycle={"per_task_work":[]},task_timeline=[],endpoint_s=300.)
    extracted=dict(rounds=[],checks={"tasks":0},issues=[],uid_by_id={},all_started_actions=[])
    with patch.object(Path,"is_file",return_value=True), \
         patch.object(report.saved,"saved_cell",return_value=statistics) as saved_cell, \
         patch.object(report.decisions,"extract",return_value=extracted) as extract, \
         patch.object(report.saved,"load",return_value=[]),patch.object(report.saved,"lines",return_value=[]):
        result=report.cell("evidence",300)
    assert result["statistics"] is statistics
    saved_cell.assert_called_once_with(Path("evidence"),300)
    extract.assert_called_once()


def test_missing_formal_results_are_not_zero_and_old_root_cannot_be_output():
    with patch.object(Path,"is_file",return_value=False):
        assert report.cell("missing")==dict(status="NOT_RUN",directory="missing",statistics=None,rounds=[])
    with pytest.raises(ValueError,match="differ"):
        report.write_report(Path("same"),Path("same"))
    with pytest.raises(ValueError,match="outside"):
        report.write_report(Path("old/new"),Path("old"))


def test_writer_emits_only_requested_new_outputs_and_explicit_not_run_status():
    empty=dict(status="NOT_RUN",statistics=None,rounds=[])
    summary=dict(cells={"r":dict(old_v2=empty,new_v3=empty)})
    writes=[]
    def write(path,text,**kwargs): writes.append((path,text))
    with patch.object(report,"build_summary",return_value=summary), \
         patch.object(Path,"mkdir"),patch.object(Path,"write_text",autospec=True,side_effect=write), \
         patch.object(report.decisions,"write_csv") as write_csv:
        assert report.write_report(Path("new"),Path("old")) is summary
    assert [path for path,_ in writes]==[Path("new/v3_analysis_summary.json"),Path("new/REPORT.md")]
    assert "NOT_RUN" in writes[1][1]
    write_csv.assert_called_once_with(Path("new/rounds.csv"),[])
