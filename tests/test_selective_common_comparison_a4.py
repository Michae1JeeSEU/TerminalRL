"""Tiny real checkpoint/replay closure with an explicit temporary score API shim."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace, MappingProxyType

import pytest

from scripts.stage3 import run_selective_common_comparison_a4 as runner
from scripts.stage3 import run_selective_common_initial_a4 as common
from scripts.stage3 import run_selective_reoptimization_a as base
from scripts.stage3 import run_latency_interface as latency
from terminal_marl.core.rotating_slots import stage2_campaign as campaign
from terminal_marl.core.rotating_slots.demand import ArrivalManifest
from terminal_marl.core.rotating_slots.planning_latency import LatencyAwareScheduler
from terminal_marl.core.rotating_slots.replay import BufferedStateJSONLSink, ReplayBundle
from terminal_marl.core.rotating_slots.stage2_planning import NormalSchedulerCallback
from terminal_marl.core.rotating_slots.stage3_learning import evaluation
from terminal_marl.core.rotating_slots.trajectory import generate_final
from terminal_marl.core.rotating_slots.world import RotatingSlotWorld, STAGE2_POLICY_ID

PROFILE="test_explicit_evaluation_api_shim"


class Clock:
    def __init__(self): self.value=0.; self.open=False
    def __call__(self):
        if self.open: self.value+=2.
        self.open=not self.open
        return self.value


def callback_record(port,plans=()):
    return dict(time=port.time,planning_timing=asdict(port.build_planning_snapshot().planning_timing),
        deferred=bool(plans),noop=not plans,proposed_trajectory_hashes=[p.trajectory_hash for p in plans],
        timings=dict(candidate_generation_and_single_vehicle=0.,selection=0.,geometry=0.,prefix=0.),
        wall_s=0.,submitted=0,status="TINY_TEST_CALLBACK")


class InitialFinal(NormalSchedulerCallback):
    def __call__(self,port):
        snapshot=port.build_planning_snapshot(); a=snapshot.aircraft["a"]
        plan=generate_final("a",a.ring,a.slot,0,snapshot.time,source_version=a.version,config=snapshot.config).plan
        assert plan is not None
        result=port.evaluate_submission((plan,))
        assert result.accepted and not result.refinement_required
        port.commit_validated_prefix(result,snapshot_version=snapshot.version)
        self.records.append(callback_record(port,(plan,)))


class MethodNoop(NormalSchedulerCallback):
    def __call__(self,port): self.records.append(callback_record(port))


def dependencies():
    return SimpleNamespace(NormalSchedulerCallback=InitialFinal,
        LatencyAwareScheduler=lambda cb,**kwargs:LatencyAwareScheduler(cb,clock=Clock(),**kwargs),
        restore_checkpoint=campaign.restore_checkpoint,save_checkpoint=campaign.save_checkpoint,
        atomic_json=campaign.atomic_json,shared_fifo_admission=campaign.shared_fifo_admission,
        episode_metrics=campaign.episode_metrics,ReplayBundle=ReplayBundle,evaluation=evaluation,latency=latency)


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf8").splitlines() if line.strip()]


@pytest.fixture
def tiny_common(tmp_path,monkeypatch):
    d=dependencies(); output=tmp_path/"output"
    sink=BufferedStateJSONLSink(tmp_path/"original_states.jsonl")
    world=RotatingSlotWorld(manifest=ArrivalManifest(seed=7324),state_sink=sink,experiment_policy_id=STAGE2_POLICY_ID)
    world.auto_emergency=False; world.seed_hold("a","inner",0,.9); world.seed_hold("b","outer",0,.9)
    cb=common.common_callback(d,"RULE_LOCAL_K8")
    wrapper=d.LatencyAwareScheduler(cb,lead_seconds=20.,preparation_margin=1.,search_handoff_reserve_seconds=4.,
        restart_policy=common.RESTART)
    identity=campaign.checkpoint_identity(dict(source_hash="tiny_initializer",base_config_hash=world.config.config_hash,
        overlay_sha256="test",planning_profile="downstream_service_v8"),campaign.digest(world.manifest.to_dict()))
    initial=tmp_path/"original.checkpoint.json"
    campaign.save_checkpoint(initial,world,wrapper,identity)
    directory=output/"common_initial"/"s3r23"
    summary=common.initialize_common(initial,directory,d=d,source_binding=dict(root="s3r23",seed=7324))
    assert summary["status"]=="COMMON_INITIAL_READY" and summary["t0"]==20.
    checkpoint=directory/"common.checkpoint.json"
    source_bytes=checkpoint.read_bytes(); stream=directory/"restored_states.jsonl"; stream_bytes=stream.read_bytes()
    original_callback=common.common_callback
    requested=[]
    def score_api_shim(actual_d,method,*,evaluation_profile=None):
        # The new score implementation is intentionally outside these tests.
        # This shim only verifies the required nonempty argument is forwarded.
        requested.append((method,evaluation_profile))
        assert evaluation_profile==PROFILE
        return original_callback(actual_d,method,evaluation_profile=None)
    monkeypatch.setattr(common,"common_callback",score_api_shim)
    d.NormalSchedulerCallback=MethodNoop
    yield dict(output=output,checkpoint=checkpoint,d=d,summary=summary,source_bytes=source_bytes,
        stream=stream,stream_bytes=stream_bytes,requested=requested)
    sink.close()


@pytest.mark.parametrize("method",base.METHODS)
def test_tiny_arm_full_checkpoint_replay_and_increment_closure(tiny_common,method):
    case=tiny_common; previous_advance=RotatingSlotWorld._advance_energy
    result=runner.run_arm(case["output"],"s3r23",method,case["checkpoint"],
        dict(source_commit="tiny_comparison",source_hash="tiny_comparison_hash"),
        evaluation_profile=PROFILE,d=case["d"],episode_seconds=130)
    assert RotatingSlotWorld._advance_energy is previous_advance
    directory=runner.evaluation_directory(case["output"],"s3r23",method)
    assert result["status"]=="COMPLETE" and result["completed_relative_s"]==130
    assert result["t0"]==20. and result["time"]==150.
    assert case["requested"]==[(method,PROFILE)]
    assert case["checkpoint"].read_bytes()==case["source_bytes"]
    assert case["stream"].read_bytes()==case["stream_bytes"]
    assert sorted(p.name for p in directory.glob("boundary_*.json"))==[
        f"boundary_{relative:04d}.json" for relative in range(0,131,5)]
    assert base.load(directory/"checkpoint_0000.json")["payload"]["time"]==20.
    assert base.load(directory/"checkpoint_0130.json")["payload"]["time"]==150.
    quality=base.load(directory/"boundary_0130.json")
    assert quality["quality"]["conservation"]
    assert quality["increment"]["cumulative_delta"]["L_normal"]==1
    assert quality["increment"]["inherited_final_touchdown_ids"]==["a"]
    assert quality["increment"]["treatment_final_touchdown_ids"]==[]
    assert quality["holding_time_by_ring"]["outer"]==pytest.approx(130.)
    assert quality["holding_energy_by_ring"]["outer"]>0.
    assert sum(quality["total_executed_energy_by_source_ring"].values())==pytest.approx(
        quality["increment"]["cumulative_delta"]["energy_j"])
    method_tasks=base.load(directory/"method_normal_tasks.json")
    all_tasks=base.load(directory/"normal_tasks.json")
    callbacks=base.load(directory/"scheduler_records.json")
    method_callbacks=base.load(directory/"method_scheduler_records.json")
    assert method_tasks[0]["snapshot_time"]==20. and method_tasks[0]["effective_time"]==40.
    assert method_tasks[0]["previous_record_index"] is None
    assert all_tasks[1]["previous_record_index"]==0
    assert len(all_tasks)==len(method_tasks)+1 and all_tasks[0]["a4_record_role"]=="INITIALIZER_PROVENANCE"
    assert len(callbacks)==len(method_callbacks)+1 and callbacks[0]["actual_delivery_status"]=="COMMITTED"
    assert all("audit_record_index" not in row for row in method_tasks+method_callbacks)
    assert quality["normal_lifecycle"]["task_count"]==len(method_tasks)
    assert quality["normal_lifecycle"]["task_count"]!=len(all_tasks)
    assert quality["costs"]["initial_resume_wall_s"]>0.
    assert quality["costs"]["online_advance_wall_s"]>=quality["costs"]["initial_resume_wall_s"]
    assert quality["costs"]["setup_wall_s"]>0. and quality["costs"]["output_io_wall_s"]>0.
    archive=ReplayBundle.validate(directory/"replay")
    assert archive.time_bounds==(0.,150.)
    frames=read_lines(directory/"replay/states.jsonl")
    original_frames=[json.loads(line) for line in case["stream_bytes"].splitlines()]
    assert [row for row in frames if row["time"]<20.]==[row for row in original_frames if row["time"]<20.]
    assert next(row for row in frames if row["time"]==20.)["aircraft"]==original_frames[-1]["aircraft"]
    replay_events=read_lines(directory/"replay/events.jsonl")
    assert any(row["type"]=="maneuver_started" and row["time"]==20. for row in replay_events)
    assert any(row["type"]=="touchdown" and row["aircraft_id"]=="a" for row in replay_events)
    mapping=base.load(directory/"record_index_mapping.json")
    first=mapping["calculation_event_rows"][0]
    assert first["method_record_index"] is None and first["audit_record_index"]==0
    assert replay_events[first["event_index"]]["details"]["record_index"] is None
    with pytest.raises(FileExistsError):
        runner.run_arm(case["output"],"s3r23",method,case["checkpoint"],
            dict(source_commit="tiny",source_hash="tiny"),evaluation_profile=PROFILE,d=case["d"],episode_seconds=5)


def test_real_tiny_saved_replay_audit_entrypoint(tiny_common):
    case=tiny_common
    runner.run_arm(case["output"],"s3r23","FULL_H",case["checkpoint"],
        dict(source_commit="tiny_comparison",source_hash="tiny"),evaluation_profile=PROFILE,d=case["d"],episode_seconds=5)
    result=runner.audit_arm(case["output"],"s3r23","FULL_H")
    directory=runner.evaluation_directory(case["output"],"s3r23","FULL_H")
    assert result["status"]=="PASS" and result["full_a20_covered"] is True
    assert (directory/"independent_replay_audit.json").is_file()
    assert (directory/"saved_resource_c2_audit.json").is_file()
    assert (directory/"saved_pva_audit.json").is_file()
    assert (directory/"proposal_nonexecution_audit.json").is_file()
    assert base.load(directory/"saved_pva_audit.json")["normal_start_coverage"]["unattributed_hashes"]==[]
    with pytest.raises(FileExistsError): runner.audit_arm(case["output"],"s3r23","FULL_H")


def test_exported_indices_are_offset_without_mutating_method_records():
    metadata=dict(initializer_callback_records=[{"time":0.}],initializer_tasks=[{"snapshot_time":0.}],
        initializer_services=[dict(sequence=0)])
    tasks=[dict(previous_record_index=None,external_predecessor=dict(initializer_record_index=0)),
           dict(previous_record_index=0)]
    wrapper=SimpleNamespace(records=tasks,online_services=[dict(sequence=0)],pending=dict(stage="RESULT",due=25.,record_index=1))
    callback=SimpleNamespace(records=[{"time":20.}])
    original=deepcopy((metadata,wrapper.records,wrapper.online_services,callback.records))
    cb,joined,services,mapping=runner.merged_records(metadata,callback,wrapper)
    assert [row.get("previous_record_index") for row in joined]==[None,0,1]
    assert services[-1]["sequence"]==1 and mapping["pending_task"]["audit_record_index"]==2
    assert (metadata,wrapper.records,wrapper.online_services,callback.records)==original


def test_record_export_normalizes_real_config_sections_without_copying_pending_packet():
    from terminal_marl.core.rotating_slots.config import Section, load_config
    from terminal_marl.core.rotating_slots.replay import _plain, _json
    class UncopyablePacket:
        def __deepcopy__(self,memo): raise AssertionError("pending packet must not be copied")
        def to_dict(self): raise AssertionError("pending packet must not be exported")
    config=load_config()
    section=Section(dict(diagnostic=dict(timing=[20.,5.],energy=3.)))
    history=MappingProxyType(dict(age=7.,seen=("a","b")))
    plan=generate_final("a","inner",0,0,20.,config=config).plan
    record=dict(section=section,immutable=MappingProxyType(dict(config=config,history=history)),
                trajectory=plan.to_dict())
    metadata=dict(initializer_callback_records=[record],initializer_tasks=[dict(record)],
                  initializer_services=[dict(record,sequence=0)])
    callback=SimpleNamespace(records=[record],history=history)
    wrapper=SimpleNamespace(records=[dict(record,previous_record_index=0)],
        online_services=[dict(record,sequence=0)],
        pending=dict(stage="RESULT",due=45.,record_index=0,packet=UncopyablePacket()))
    originals=_json((metadata,callback.records,wrapper.records,wrapper.online_services))
    trajectory_bytes=_json(plan.to_dict())
    with pytest.raises(RecursionError): deepcopy(record)
    exported=runner.merged_records(metadata,callback,wrapper)
    callbacks,tasks,services,mapping=exported
    expected=json.loads(_json(record))
    for rows in (callbacks,tasks,services):
        for row in rows:
            assert row["section"]==expected["section"] and row["immutable"]==expected["immutable"]
            assert row["trajectory"]==expected["trajectory"]
    assert mapping["pending_task"]==dict(stage="RESULT",due=45.,method_record_index=0,audit_record_index=1)
    assert tasks[1]["previous_record_index"]==1 and services[1]["sequence"]==1
    assert campaign.canonical(exported)==_json(exported)
    assert _json((metadata,callback.records,wrapper.records,wrapper.online_services))==originals
    assert callback.history is history and callback.records[0] is record
    assert record["section"] is section and record["immutable"]["config"] is config
    assert _json(plan.to_dict())==trajectory_bytes
    callbacks[0]["section"]["diagnostic"]["energy"]=99.
    assert section.diagnostic.energy==3.
    assert isinstance(_plain(exported),list)


def test_live_frozen_follow_downstream_window_wait_exports_through_existing_plain_view():
    from terminal_marl.core.rotating_slots.config import Section
    from terminal_marl.core.rotating_slots.replay import _json
    from terminal_marl.core.rotating_slots.stage2_planning import CandidateProvider, EvaluationContext
    from terminal_marl.core.rotating_slots.selective_replanning import build_global_frame
    world=RotatingSlotWorld(); world.seed_hold("a","middle",0,.9)
    snapshot=world.build_planning_snapshot(); before=_json(world.to_dict())
    provider=CandidateProvider(planning_profile="downstream_service_v8",evaluation_profile="stitched_continuation_v1")
    context=EvaluationContext(snapshot)
    frame=build_global_frame(snapshot,provider,context)
    follow=provider._column(snapshot,snapshot.aircraft["a"],(),"WAIT",context)
    joint=frame.downstream({follow.signature:frame.baseline_by_id["a"]},{"a"})
    _,rows=joint.evaluate((follow,))
    assert isinstance(rows["a"]["window_wait_seconds_by_ring"],Section)
    live=dict(downstream=rows)
    with pytest.raises(RecursionError): deepcopy(live)
    callback=SimpleNamespace(records=[live])
    wrapper=SimpleNamespace(records=[],online_services=[],pending=None)
    metadata=dict(initializer_callback_records=[],initializer_tasks=[],initializer_services=[])
    exported=runner.merged_records(metadata,callback,wrapper)
    assert exported[0][0]["downstream"]==json.loads(_json(rows))
    assert campaign.canonical(exported)==_json(exported)
    assert callback.records[0] is live and isinstance(live["downstream"]["a"]["window_wait_seconds_by_ring"],Section)
    assert _json(world.to_dict())==before


@pytest.mark.parametrize("value",[None,"", "   ", False])
def test_explicit_profile_required_before_world_restore(value):
    with pytest.raises(ValueError,match="nonempty"):
        runner.run_arm("unused","s3r23","FULL_H","absent",{},evaluation_profile=value)


def cli_values(tmp_path,stage="arm"):
    return ["--stage",stage,"--output",str(tmp_path),"--root","s3r23","--method","FULL_H",
        "--evaluation-profile",PROFILE,"--source-freeze",str(tmp_path/"freeze.json"),"--cpu","12"]


@pytest.mark.parametrize("stage,cap",[("arm",1200.),("audit",300.)])
def test_parent_dispatches_only_one_bounded_job(tmp_path,monkeypatch,stage,cap):
    from scripts.stage3 import run_selective_quality_gap_a4 as a4
    ledger=dict(jobs={}); observed=[]
    @contextmanager
    def transaction(*args,**kwargs): yield ledger
    def bounded(output,job_id,command,*,cap_s):
        observed.append((output,job_id,command,cap_s)); ledger["jobs"][job_id]={}
        return dict(status="TEST_NO_CHILD_LAUNCHED")
    monkeypatch.setattr(a4,"validated_output",lambda path:path)
    monkeypatch.setattr(a4,"ledger_transaction",transaction)
    monkeypatch.setattr(a4,"run_bounded_job",bounded)
    assert runner.main(cli_values(tmp_path,stage))["status"]=="TEST_NO_CHILD_LAUNCHED"
    assert len(observed)==1 and observed[0][3]==cap
    assert observed[0][2][-2:]==["--_child-job-id","comparison_"+stage+"_s3r23_FULL_H"]
    assert observed[0][2].count("--method")==1


def test_child_guard_blocks_before_dependencies_or_world(tmp_path,monkeypatch):
    from scripts.stage3 import run_selective_quality_gap_a4 as a4
    @contextmanager
    def transaction(*args,**kwargs): yield dict(jobs={})
    monkeypatch.setattr(a4,"validated_output",lambda path:path)
    monkeypatch.setattr(a4,"ledger_transaction",transaction)
    monkeypatch.setattr(base,"dependencies",lambda path:pytest.fail("dependencies must not load without parent watchdog"))
    with pytest.raises(ValueError,match="active comparison parent"):
        runner.main(cli_values(tmp_path)+["--_child-job-id","comparison_arm_s3r23_FULL_H"])


def test_shared_cap_and_no_retry_are_enforced_before_start(tmp_path):
    from scripts.stage3 import run_selective_quality_gap_a4 as a4
    with a4.ledger_transaction(tmp_path) as ledger:
        assert ledger["cap_child_wall_s"]==7200.
        ledger["settled_child_wall_s"]=7200.
    with pytest.raises(ValueError,match="budget exhausted"):
        a4.run_bounded_job(tmp_path,"must_not_start",[sys.executable,"-c","raise SystemExit(91)"],cap_s=1200.)
    with a4.ledger_transaction(tmp_path) as ledger:
        assert "must_not_start" not in ledger["jobs"]
        ledger["settled_child_wall_s"]=0.
        ledger["jobs"]["already_attempted"]=dict(status="failed")
    with pytest.raises(ValueError,match="already attempted"):
        a4.run_bounded_job(tmp_path,"already_attempted",[sys.executable,"-c","raise SystemExit(91)"],cap_s=1200.)


def test_child_freeze_key_and_launch_thread_capture(tiny_common,tmp_path,monkeypatch):
    from scripts.stage3 import run_selective_quality_gap_a4 as a4
    from scripts.stage3 import run_selective_reoptimization_repair as old
    case=tiny_common
    names=[runner.DRIVER.relative_to(runner.PROJECT).as_posix(),
           "tests/test_selective_common_comparison_a4.py",base.DRIVER.relative_to(runner.PROJECT).as_posix()]
    files={name:base.file_hash(runner.PROJECT/name) for name in names}
    source=dict(files=files,source_hash=base.digest(files),commit="test_freeze")
    freeze=tmp_path/"freeze.json"; campaign.atomic_json(freeze,source)
    ledger=dict(a4_source_hash="diagnostic",common_initial_source_hash="initializer",jobs={
        "comparison_arm_s3r23_FULL_H":dict(status="running",parent_pid=os.getppid())})
    @contextmanager
    def transaction(*args,**kwargs): yield ledger
    monkeypatch.setattr(a4,"validated_output",lambda path:path)
    monkeypatch.setattr(a4,"ledger_transaction",transaction)
    for name in base.THREADS: monkeypatch.setenv(name,"1")
    def load_dependencies(path):
        for name in base.THREADS: os.environ[name]="99"
        return case["d"]
    monkeypatch.setattr(base,"dependencies",load_dependencies)
    monkeypatch.setattr(old,"set_cpu",lambda args,d: base.require(
        d.launch_threads=={key:"1" for key in base.THREADS},"launch thread capture changed"))
    calls=[]
    monkeypatch.setattr(runner,"run_arm",lambda *args,**kwargs:calls.append(kwargs) or dict(status="BOUND_ONLY"))
    values=cli_values(case["output"])
    values[values.index("--source-freeze")+1]=str(freeze)
    result=runner.main(values+["--_child-job-id","comparison_arm_s3r23_FULL_H"])
    assert result["status"]=="BOUND_ONLY" and calls[0]["evaluation_profile"]==PROFILE
    assert ledger["a4_source_hash"]=="diagnostic" and ledger["common_initial_source_hash"]=="initializer"
    assert ledger["comparison_source_hash"]==source["source_hash"]
