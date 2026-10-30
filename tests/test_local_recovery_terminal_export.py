"""Hand-built terminal checkpoints; no simulation or planning is executed."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.stage3 import export_local_recovery_terminal as export
from terminal_marl.core.rotating_slots import stage2_campaign as campaign
from terminal_marl.core.rotating_slots.demand import ArrivalManifest
from terminal_marl.core.rotating_slots.planning_latency import LatencyAwareScheduler
from terminal_marl.core.rotating_slots.replay import BufferedStateJSONLSink, ReplayBundle
from terminal_marl.core.rotating_slots.stage2_planning import NormalSchedulerCallback
from terminal_marl.core.rotating_slots.stage3_learning import evaluation
from terminal_marl.core.rotating_slots.world import RotatingSlotWorld, STAGE2_POLICY_ID


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    campaign.atomic_json(path,value)


@pytest.fixture
def terminal_fixture(tmp_path,monkeypatch):
    output=tmp_path/export.runner.LOCAL_RECOVERY_SERIALIZATIONFIX_OUTPUT_NAME
    directory=export.comparison.evaluation_directory(output,"s3r24","FULL_H")
    directory.mkdir(parents=True)
    source=dict(source_hash="original_frozen_source",commit="ead8b8e")
    common=output/"common_initial/s3r24/common.checkpoint.json"; save(common,{"immutable":"common"})
    monkeypatch.setitem(export.runner.COMMON_HASHES,"s3r24",export.base.file_hash(common))
    binding=dict(source_hash=source["source_hash"],source_commit=source["commit"],root="s3r24",method="FULL_H",
        horizon=[20.,320.],evaluation_profile=export.runner.PROFILE,common_checkpoint_sha256=export.base.file_hash(common))
    binding["binding_id"]=export.base.digest(binding)
    save(output/"arms/s3r24/FULL_H/resolved_run_binding.json",binding)
    boundary=dict(root_id="s3r24",method="FULL_H",time=320.,t0=20.,relative_time_s=300.,
        source_binding=binding["binding_id"],evaluation_profile=export.runner.PROFILE,
        costs=dict(full_job_wall_s=1111.),holding_energy_by_ring=dict(inner=12.),holding_time_by_ring=dict(inner=3.),
        total_executed_energy_by_source_ring=dict(inner=14.),increment=dict(cumulative_delta=dict(energy_kwh=2.)),
        normal_lifecycle=dict(pending_task_count=1),quality=dict(L_normal=0))
    sink=BufferedStateJSONLSink(directory/"original_states.jsonl")
    world=RotatingSlotWorld(manifest=ArrivalManifest(seed=7325),state_sink=sink,experiment_policy_id=STAGE2_POLICY_ID)
    # This is a hand-crafted terminal serialization fixture, not a rollout.
    world.time=320.
    callback=NormalSchedulerCallback("rh_cp_sat",planning_profile="downstream_service_v8",online_policy="progressive_v1",
        selective_method="FULL_H",selective_rule_variant=export.common.RULE,evaluation_profile=export.runner.PROFILE,
        search_policy=dict(export.common.SWITCHES))
    callback.initialize_imported_history(world.build_planning_snapshot(),events=world.events)
    callback.records=[dict(time=300.,planning_timing=dict(snapshot_time=300.),status="FIXTURE")]
    wrapper=LatencyAwareScheduler(callback,lead_seconds=20.,preparation_margin=1.,search_handoff_reserve_seconds=4.,restart_policy=export.common.RESTART)
    wrapper.records=[dict(snapshot_time=300.,status="PREPARED",submitted=0)]
    wrapper.pending=dict(stage="PREPARED",token="issued_fixture",record_index=0,due=321.)
    world._planning_latency_packets={"issued_fixture":dict(traffic_policy=wrapper.traffic_policy,
        certificate=dict(proof=("fixture",320.)))}
    identity=campaign.checkpoint_identity(dict(source_hash=source["source_hash"],base_config_hash=world.config.config_hash,
        overlay_sha256="fixture",planning_profile="downstream_service_v8"),campaign.digest(world.manifest.to_dict()))
    checkpoint=directory/"checkpoint_0300.json"
    campaign.save_checkpoint(checkpoint,world,wrapper,identity,boundary_summary=boundary)
    sink.close()
    metadata=dict(t0=20.,checkpoint_sha256=export.base.file_hash(common),initializer_callback_records=[],
        initializer_tasks=[],initializer_services=[],common_summary=dict(t0=20.,baseline=dict(event_count=0,dispatch_count=0)))
    common_restores=[]; restored=[]
    def restore_common(*args,**kwargs):
        common_restores.append((args,kwargs))
        return SimpleNamespace(state_sink=None),None,None,deepcopy(metadata)
    monkeypatch.setattr(export.common,"restore_common",restore_common)
    def restore(*args,**kwargs):
        value=campaign.restore_checkpoint(*args,**kwargs)
        restored.append((value,args[1]))
        return value
    def forbidden(*args,**kwargs): pytest.fail("terminal export invoked simulation/planning")
    monkeypatch.setattr(RotatingSlotWorld,"advance",forbidden)
    monkeypatch.setattr(NormalSchedulerCallback,"__call__",forbidden)
    monkeypatch.setattr(export.common,"resume_common",forbidden)
    d=SimpleNamespace(NormalSchedulerCallback=NormalSchedulerCallback,LatencyAwareScheduler=LatencyAwareScheduler,
        restore_checkpoint=restore,evaluation=evaluation,ReplayBundle=ReplayBundle)
    ledger=dict(service_source_hash=source["source_hash"],service_evaluation_profile=export.runner.PROFILE,
        jobs=dict(service_arm_s3r24_FULL_H=dict(status="BUDGET_LIMITED",child_wall_s=1200.,exit_code=-15)))
    return SimpleNamespace(output=output,source=source,ledger=ledger,d=d,checkpoint=checkpoint,
        directory=directory,boundary=boundary,restored=restored,common_restores=common_restores)


def test_real_pending_checkpoint_export_preserves_archive_records_and_certificates_without_advancing(terminal_fixture):
    f=terminal_fixture
    before={p:p.read_bytes() for p in f.output.rglob("*") if p.is_file()}
    result=export.export_terminal(f.output,"s3r24","FULL_H",f.source,f.ledger,exporter_commit="exporter_commit",d=f.d)
    assert {p:p.read_bytes() for p in before}==before
    assert f.ledger["jobs"]["service_arm_s3r24_FULL_H"]["status"]=="BUDGET_LIMITED"
    world,wrapper=f.restored[0]
    assert world.time==320. and wrapper.pending["token"]=="issued_fixture"
    assert world._planning_latency_packets["issued_fixture"]["certificate"]["proof"]==("fixture",320.)
    assert wrapper.callback.records[0]["status"]=="FIXTURE" and wrapper.records[0]["status"]=="PREPARED"
    assert f.common_restores[0][1]["directory"].is_relative_to(f.output/"terminal_export")
    home=f.output/"terminal_export/arms/s3r24/FULL_H"
    directory=export.comparison.evaluation_directory(f.output/"terminal_export","s3r24","FULL_H")
    assert (directory/"checkpoint_terminal.json").read_bytes()==f.checkpoint.read_bytes()
    assert export.base.load(directory/"boundary_0300.json")==f.boundary
    assert result["status"]=="COMPLETE" and result["completed_state_export_only"] is True
    assert result["source_original_job_status"]=="BUDGET_LIMITED" and result["source_original_job_wall_s"]==1200.
    assert result["simulation_advance_calls"]==result["callback_calls"]==result["algorithm_resume_calls"]==0
    assert result["costs"]==f.boundary["costs"]
    assert export.base.load(home/"export_receipt.json")["exporter_script_sha256"]==export.base.file_hash(export.DRIVER)
    assert (directory/"replay/replay.json").is_file() and (home/"arm.finished.json").is_file()
    with pytest.raises(ValueError,match="no overwrite"):
        export.export_terminal(f.output,"s3r24","FULL_H",f.source,f.ledger,exporter_commit="exporter_commit",d=f.d)


@pytest.mark.parametrize("change,message",[("hash","envelope hash"),("time","exact completed"),("job","BUDGET_LIMITED"),("source","source freeze")])
def test_input_rejections_leave_original_archive_unchanged(terminal_fixture,change,message):
    f=terminal_fixture
    if change in ("hash","time"):
        value=export.base.load(f.checkpoint)
        if change=="hash": value["sha256"]="wrong"
        else:
            value["payload"]["boundary_summary"]["relative_time_s"]=295.
            value["sha256"]=campaign.digest(value["payload"])
        save(f.checkpoint,value)
    elif change=="job": f.ledger["jobs"]["service_arm_s3r24_FULL_H"]["status"]="completed"
    else: f.source["source_hash"]="other"
    before={p:p.read_bytes() for p in f.output.rglob("*") if p.is_file()}
    with pytest.raises(ValueError,match=message):
        export.export_terminal(f.output,"s3r24","FULL_H",f.source,f.ledger,exporter_commit="new",d=f.d)
    assert {p:p.read_bytes() for p in before}==before
    assert not (f.output/"terminal_export").exists()


def test_failed_terminal_state_is_rejected_without_source_writes(terminal_fixture):
    f=terminal_fixture; value=export.base.load(f.checkpoint); codec=campaign.Stage2Codec()
    state=codec.decode(value["payload"]["world"]); state["failed"]=True
    value["payload"]["world"]=codec.encode(state); value["sha256"]=campaign.digest(value["payload"])
    save(f.checkpoint,value); before=f.checkpoint.read_bytes()
    with pytest.raises(ValueError,match="failed/invalid"):
        export.export_terminal(f.output,"s3r24","FULL_H",f.source,f.ledger,exporter_commit="new",d=f.d)
    assert f.checkpoint.read_bytes()==before


def test_audit_targets_only_export_tree_and_validates_lineage(terminal_fixture,monkeypatch):
    f=terminal_fixture
    export.export_terminal(f.output,"s3r24","FULL_H",f.source,f.ledger,exporter_commit="new",d=f.d)
    observed=[]
    monkeypatch.setattr(export.comparison,"audit_arm",lambda output,root,method,**kwargs:observed.append((output,root,method)) or dict(status="PASS"))
    assert export.audit_export(f.output,"s3r24","FULL_H",f.source,f.ledger)["status"]=="PASS"
    assert observed==[(f.output/"terminal_export","s3r24","FULL_H")]
    receipt=f.output/"terminal_export/arms/s3r24/FULL_H/export_receipt.json"
    value=export.base.load(receipt); value["source_checkpoint_sha256"]="other"; save(receipt,value)
    with pytest.raises(ValueError,match="receipt lineage"):
        export.audit_export(f.output,"s3r24","FULL_H",f.source,f.ledger)
    assert len(observed)==1


@pytest.mark.parametrize("stage,cap",[("export",180.),("audit",300.)])
def test_cli_charges_export_and_audit_to_same_original_ledger(monkeypatch,tmp_path,stage,cap):
    for key in export.base.THREADS: monkeypatch.setenv(key,"1")
    output=tmp_path/export.runner.LOCAL_RECOVERY_SERIALIZATIONFIX_OUTPUT_NAME
    monkeypatch.setattr(export.runner,"validated_output",lambda path:path)
    monkeypatch.setattr(export.runner,"policy_for_output",lambda output:dict(cap_child_wall_s=1234.))
    observed={}
    def bounded(output,job,command,**kwargs):
        observed.update(output=output,job=job,command=command,**kwargs); return dict(status="completed")
    monkeypatch.setattr(export.old,"run_bounded_job",bounded)
    export.main([stage,"--output",str(output),"--root","s3r24","--method","FULL_H","--cpu","18",
        "--source-freeze","original_freeze.json","--exporter-commit","new_commit"])
    assert observed["output"]==output and observed["cap_s"]==cap
    assert observed["job"]=="terminal_"+stage+"_s3r24_FULL_H"
    assert observed["policy"]["cap_child_wall_s"]==1234.


def test_combined_report_splices_only_high_export_arms_and_recomputes_checks(tmp_path,monkeypatch):
    output=tmp_path/export.runner.LOCAL_RECOVERY_SERIALIZATIONFIX_OUTPUT_NAME
    def collected(old,new):
        is_export=Path(new).name=="terminal_export"
        arms=[dict(root=root,method=method,status="COMPLETE",evaluation_profile=export.runner.PROFILE,
            common_checkpoint_sha256=root,value=(100 if is_export else 10)+(0 if root=="s3r23" else 1))
            for root in export.report.ROOTS for method in export.report.METHODS]
        ordinary=dict(arms=arms,task_timeline=[dict(root=a["root"],method=a["method"]) for a in arms],callback_timings=[])
        return dict(campaigns=dict(old=dict(input_root=str(old),ordinary=deepcopy(ordinary),metrics=deepcopy(arms)),
            new=dict(input_root=str(new),ordinary=ordinary,metrics=deepcopy(arms))))
    monkeypatch.setattr(export.report,"collect",collected)
    for method in export.report.METHODS:
        save(output/"terminal_export/arms/s3r24"/method/"export_receipt.json",dict(completed_state_export_only=True))
    value=export.combined_report(tmp_path/"old",output)
    rows=value["campaigns"]["new"]["metrics"]
    assert [r["value"] for r in rows]==[10,10,101,101]
    assert all(r["export_lineage"] is None for r in rows if r["root"]=="s3r23")
    assert all(r["export_lineage"]["completed_state_export_only"] for r in rows if r["root"]=="s3r24")
    assert all(value["same_common_checkpoint_across_campaigns"].values())
    monkeypatch.setattr(export,"PROJECT",tmp_path)
    target=export.write_report(tmp_path/"artifacts/fresh_report",value,output)
    assert json.loads((target/"summary.json").read_text())["campaigns"]["new"]["ordinary"]["status"]=="COMPLETE"
    assert (target/"comparison.csv").is_file()
    with pytest.raises(FileExistsError): export.write_report(target,value,output)
    with pytest.raises(ValueError): export.write_report(output/"arms/overwrite",value,output)
    def duplicate_collect(old,new):
        result=collected(old,new)
        result["campaigns"]["old"]["metrics"].append(deepcopy(result["campaigns"]["old"]["metrics"][0]))
        return result
    monkeypatch.setattr(export.report,"collect",duplicate_collect)
    with pytest.raises(ValueError,match="four unique"):
        export.combined_report(tmp_path/"old",output)


def test_server_report_whitelist_and_original_arm_protection(tmp_path,monkeypatch):
    monkeypatch.setattr(Path,"home",classmethod(lambda cls:tmp_path))
    monkeypatch.setattr(export,"PROJECT",tmp_path/"server_project")
    output=tmp_path/"training_outputs/terminal"/export.runner.LOCAL_RECOVERY_SERIALIZATIONFIX_OUTPUT_NAME
    value=dict(campaigns=dict(old=dict(input_root=str(tmp_path/"old"),metrics=[]),new=dict(input_root=str(output),metrics=[])))
    assert export.write_report(output/"fresh_report",value,output).is_dir()
    with pytest.raises(ValueError): export.write_report(output/"terminal_export/overwrite",value,output)
    with pytest.raises(ValueError): export.write_report(tmp_path/"other",value,output)
