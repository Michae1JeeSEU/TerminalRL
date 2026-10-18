"""Pure admission checks and opt-in fresh-process real cold-window fixture.

The synthetic empty/service fixture tests continuation, never performance
qualification or frozen traffic availability. Set RUN_WINDOW_INTEGRATION=1
only after collaborating implementation files have stopped changing.
"""
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "terminal_marl/core/rotating_slots"))
from performance import corpus, provenance, windows
from performance.checkpoint import CheckpointIdentity

REFERENCE = ROOT / "artifacts/rotating_slots/performance_reference_r0"
SCRIPT = ROOT / "scripts/server/run_rotating_slots_performance_windows.py"
SEMANTICS = {k: "a" * 64 for k in windows.IDENTITY_KEYS}
SEMANTICS["policy_id"] = provenance.R0_POLICY_ID


def seal(document):
    document["manifest_sha256"] = provenance.digest({k: v for k, v in document.items() if k != "manifest_sha256"})
    return document


def manifest_fixture():
    sources = {}
    for case, (rate, seed) in corpus.CASES.items():
        cuts = []
        for _, case_id, requested in corpus.WINDOWS:
            if case_id != case:
                continue
            cuts.append({"actual_sim_time": requested, "requested_sim_times": [requested],
                "pinned": True, "corpus": True, "checkpoint_sha256": "b" * 64,
                "sidecar_sha256": "c" * 64,
                "cursor": {"original_target": 3600., "phase": "loop_start", "outer_phase": "arrival_window"}})
        metadata = {"kernel": "R0", "kernel_commit": provenance.R0_COMMIT,
            "reference_commit": provenance.R0_COMMIT, "reference_source_sha256": provenance.R0_SOURCE_SHA256,
            "runner_execution_sha256": "d" * 64, "semantic_fingerprint": SEMANTICS,
            "observation_original_source_sha256": "e" * 64,
            "observation_generated_source_sha256": "f" * 64}
        sources[case] = {"status": "SUSPENDED_CAPTURE", "stop_time": max(c["actual_sim_time"] for c in cuts) + 60.,
            "cuts": cuts, "workload": {"seed": seed, "rate_per_hour": rate}, "source_identity": metadata}
    rows = corpus.select_windows(sources)
    identity = asdict(CheckpointIdentity(**SEMANTICS, implementation_sha256="d" * 64))
    for row in rows:
        row["validation"] = {"identity": identity, "world_time": row["cut"]["actual_sim_time"],
            "cache_mode": "cold", "restored_without_advance": True,
            "cursor": dict(row["cut"]["cursor"], shim_source_sha256="f" * 64)}
    return seal({"schema": "ft_rs_performance_corpus_v1_0", "status": "FROZEN",
        "reference_commit": provenance.R0_COMMIT, "reference_source_sha256": provenance.R0_SOURCE_SHA256,
        "semantic_fingerprint": SEMANTICS, "sources": sources, "windows": rows})


def test_parser_only_real_single_window_command_and_required_expected_hash():
    parser = windows.parser()
    args = parser.parse_args(["window", "--corpus-manifest", "corpus.json", "--corpus-sha256", "a" * 64,
        "--window", "W71-A", "--duration", "10", "--kernel", "R0", "--reference", "ref",
        "--deployment-manifest", "deployment.json", "--implementation-commit", "b" * 40, "--output", "new"])
    assert args.mode == "benchmark" and args.migration_approval is None
    with pytest.raises(SystemExit):
        parser.parse_args(["paired"])
    with pytest.raises(SystemExit):
        parser.parse_args(["window", "--window", "W71-A", "--duration", "1"])


def test_canonical_frozen_six_selection_and_duration():
    document = manifest_fixture()
    for window, _, _ in corpus.WINDOWS:
        assert windows.validate_manifest(document, document["manifest_sha256"], window, 10)["window"] == window
    assert windows.validate_manifest(document, document["manifest_sha256"], "W71-A", 60)
    with pytest.raises(windows.WindowError, match="digest"):
        windows.validate_manifest(document, "0" * 64, "W71-A", 10)
    with pytest.raises(windows.WindowError, match="duration"):
        windows.validate_manifest(document, document["manifest_sha256"], "W71-A", 20)


@pytest.mark.parametrize("mutation", ["incomplete", "five", "duplicate", "source", "target", "cursor",
                                      "cold", "time", "checkpoint_hash", "short", "selection"])
def test_manifest_rejects_changed_frozen_inputs(mutation):
    document = manifest_fixture()
    row = document["windows"][0]
    if mutation == "incomplete": document["status"] = "INCOMPLETE"
    if mutation == "five": document["windows"].pop()
    if mutation == "duplicate": document["windows"][-1]["window"] = "W71-A"
    if mutation == "source": document["sources"][row["case"]]["source_identity"]["reference_commit"] = "0" * 40
    if mutation == "target": row["validation"]["cursor"]["original_target"] = 310.
    if mutation == "cursor": row["validation"]["cursor"]["phase"] = "returned"
    if mutation == "cold": row["validation"]["restored_without_advance"] = False
    if mutation == "time": row["validation"]["world_time"] += .2
    if mutation == "checkpoint_hash": row["cut"]["checkpoint_sha256"] = ""
    if mutation == "short": document["sources"][row["case"]]["stop_time"] -= .1
    if mutation == "selection": row["original_requested_sim_time"] += .1
    seal(document)
    with pytest.raises((windows.WindowError, corpus.CorpusError)):
        windows.validate_manifest(document, document["manifest_sha256"], "W71-A", 10)


def approval_fixture(tmp_path):
    source = CheckpointIdentity(**SEMANTICS, implementation_sha256="b" * 64)
    target = CheckpointIdentity(**SEMANTICS, implementation_sha256="c" * 64)
    metadata = {"observation_original_source_sha256": "d" * 64,
                "observation_generated_source_sha256": "e" * 64}
    evidence = tmp_path / "oracle.json"
    evidence.write_text('{"fixture":true}', encoding="utf8")
    approval = {"schema": "window_migration_approval_v1", "qualified": True,
        "corpus_sha256": "a" * 64, "target_execution_sha256": "c" * 64,
        "target_observation_original_sha256": "d" * 64, "target_observation_generated_sha256": "e" * 64,
        "checkpoint_helper_sha256": "f" * 64, "supporting_evidence": [
            {"path": str(evidence.resolve()), "sha256": sha256(evidence.read_bytes()).hexdigest()}],
        "migrations": [{"migration_id": "fixture-only-approval", "source_identity": asdict(source),
            "target_identity": asdict(target), "source_observation_original_sha256": "d" * 64,
            "source_observation_generated_sha256": "e" * 64}]}
    kwargs = dict(source=source, target=target, corpus_digest="a" * 64,
                  source_metadata=metadata, target_metadata=dict(metadata), checkpoint_helper_sha256="f" * 64)
    return approval, kwargs


def test_approval_consumes_exact_pair_and_existing_evidence(tmp_path):
    approval, kwargs = approval_fixture(tmp_path)
    assert windows.validate_approval(approval, **kwargs) == "fixture-only-approval"


def test_window_entrypoint_bytes_participate_in_execution_identity():
    entry = "scripts/server/run_rotating_slots_performance_windows.py"
    runtime = {"source": {"helper_execution_files_sha256": {"old.py": "b" * 64},
        "actual_core_files_sha256": {"core.py": "c" * 64}},
        "deployment": {"actual_source_files_sha256": {entry: "a" * 64}},
        "observed": type("Shim", (), {"generated_source_sha256": "d" * 64})()}
    before = windows.bind_window_entrypoint(runtime, "R0")
    runtime["deployment"]["actual_source_files_sha256"][entry] = "e" * 64
    after = windows.bind_window_entrypoint(runtime, "R0")
    assert before != after
    assert runtime["source"]["helper_execution_files_sha256"][entry] == "e" * 64


@pytest.mark.parametrize("mutation", ["unqualified", "corpus", "execution", "source", "target", "source_shim",
    "target_shim", "helper", "none", "missing", "tampered", "duplicate", "semantic", "changed_shim"])
def test_approval_rejects_stale_pair_shim_helper_and_missing_evidence(tmp_path, mutation):
    approval, kwargs = approval_fixture(tmp_path)
    pair = approval["migrations"][0]
    if mutation == "unqualified": approval["qualified"] = False
    if mutation == "corpus": approval["corpus_sha256"] = "0" * 64
    if mutation == "execution": approval["target_execution_sha256"] = "0" * 64
    if mutation == "source": pair["source_identity"]["implementation_sha256"] = "0" * 64
    if mutation == "target": pair["target_identity"]["implementation_sha256"] = "0" * 64
    if mutation == "source_shim": pair["source_observation_generated_sha256"] = "0" * 64
    if mutation == "target_shim": approval["target_observation_generated_sha256"] = "0" * 64
    if mutation == "helper": approval["checkpoint_helper_sha256"] = "0" * 64
    if mutation == "none": approval["supporting_evidence"] = []
    if mutation == "missing": Path(approval["supporting_evidence"][0]["path"]).unlink()
    if mutation == "tampered": Path(approval["supporting_evidence"][0]["path"]).write_bytes(b"changed")
    if mutation == "duplicate": approval["migrations"].append(deepcopy(pair))
    if mutation == "semantic": kwargs["target"] = CheckpointIdentity(**dict(SEMANTICS, config_hash="0" * 64), implementation_sha256="c" * 64)
    if mutation == "changed_shim": kwargs["target_metadata"]["observation_generated_source_sha256"] = "0" * 64
    with pytest.raises(windows.WindowError):
        windows.validate_approval(approval, **kwargs)


# Test-only admission seam: this fixture is explicitly an empty/service world,
# not one of the traffic sources. It exercises the production cold restore,
# natural hook, checkpoint, recorder, replay and timing implementation.
FIXTURE_CHILD = r'''
import json, pathlib, sys, argparse
sys.dont_write_bytecode = True
root, reference, deployment, output = map(pathlib.Path, sys.argv[1:5])
sys.path.insert(0, str(root / "terminal_marl/core/rotating_slots"))
from performance import windows, runner, provenance
from performance.checkpoint import capture_checkpoint, write_checkpoint, atomic_write_json, StateCodec, load_checkpoint, restore_checkpoint
from performance.observation import SuspendAtCut
args = argparse.Namespace(kernel="R0", reference=reference, deployment_manifest=deployment,
 implementation_commit="1c5afd37c891a0ed4776f64367fc7264db3c1c54", output=output,
 window="W71-A", duration=10, mode=sys.argv[5], migration_approval=None,
 corpus_sha256="a"*64, corpus_manifest=output.parent / (output.name+"-fixture-corpus.json"))
runtime = runner.bootstrap(args, root)
source_identity = runner._checkpoint_identity(runtime)
source_metadata = json.loads(json.dumps(runtime["source"]))
windows.bind_window_entrypoint(runtime,"R0")
identity = runner._checkpoint_identity(runtime)
migration_fixture=len(sys.argv)>6 and sys.argv[6] in ("migration","missing_approval")
checkpoint_identity=source_identity if migration_fixture else identity
source = output.parent / (output.name+"-source")
source.mkdir()
sink = runtime["replay_module"].BufferedStateJSONLSink(source / "states.jsonl")
w = runtime["world_module"].RotatingSlotWorld(config=runtime["config"], state_sink=sink)
w.seed_service("fixture-service", pad_id=0, remaining=.35)
driver=runtime["driver_module"].DeterministicDriver()
saved=[]
def take(world,cursor,driver):
 if world.time >= .31:
  saved.append(capture_checkpoint(world, identity=checkpoint_identity, cursor=cursor, driver=driver))
  raise SuspendAtCut()
try: runtime["observed"].run(w,3600.,driver,take,outer_phase="arrival_window")
except SuspendAtCut: pass
payload=saved[0]
checkpoint=source/"checkpoint.json"
write_checkpoint(checkpoint,payload)
sidecar=source/"checkpoint.runner.json"
atomic_write_json(sidecar,{"fixture":True})
before=(source/"states.jsonl").read_bytes()
cut={"actual_sim_time":w.time,"checkpoint":str(checkpoint),"sidecar":str(sidecar),
 "checkpoint_sha256":windows._sha(checkpoint),"sidecar_sha256":windows._sha(sidecar)}
row={"cut":cut,"frozen_checkpoint":str(checkpoint),"frozen_sidecar":str(sidecar),
 "validation":{"cursor":StateCodec().decode(payload["cursor"])}}
document={"fixture":"service-events-not-performance-corpus"}
atomic_write_json(args.corpus_manifest,document)
source_data={"source_attempt":str(source),"source_files_sha256":{},"source_identity":source_metadata,
 "workload":{"seed":11,"fixture":"service-events"}}
windows.validate_inputs=lambda *_: (document,row,source_data,checkpoint_identity,payload)
if migration_fixture:
 assert checkpoint_identity!=identity
 if sys.argv[6]=="missing_approval":
  try: windows.execute_window(args,runtime)
  except windows.WindowError as exc: assert "explicit qualified migration approval required" in str(exc)
  else: raise AssertionError("unapproved migration accepted")
  assert not output.exists()
  w.state_sink.close()
  print(json.dumps({"status":"MISSING_APPROVAL_REJECTED","output_not_created":True}))
  raise SystemExit(0)
 from dataclasses import asdict
 evidence=source/"test-only-migration-evidence.json"
 atomic_write_json(evidence,{"scope":"test-only service-events identity registration; not production equivalence approval",
  "source_identity":asdict(checkpoint_identity),"target_identity":asdict(identity),
  "source_checkpoint_sha256":windows._sha(checkpoint)})
 approval={"schema":"window_migration_approval_v1","qualified":True,"corpus_sha256":args.corpus_sha256,
  "target_execution_sha256":identity.implementation_sha256,
  "target_observation_original_sha256":runtime["source"]["observation_original_source_sha256"],
  "target_observation_generated_sha256":runtime["source"]["observation_generated_source_sha256"],
  "checkpoint_helper_sha256":windows._sha(root/"terminal_marl/core/rotating_slots/performance/checkpoint.py"),
  "supporting_evidence":[{"path":str(evidence),"sha256":windows._sha(evidence)}],
  "migrations":[{"migration_id":"test-only-entrypoint-identity-extension",
   "source_identity":asdict(checkpoint_identity),"target_identity":asdict(identity),
   "source_observation_original_sha256":source_metadata["observation_original_source_sha256"],
   "source_observation_generated_sha256":source_metadata["observation_generated_source_sha256"]}]}
 args.migration_approval=source/"test-only-migration-approval.json"
 atomic_write_json(args.migration_approval,approval)
injected_failure=len(sys.argv)>6 and sys.argv[6]=="inject_failure"
if injected_failure:
 from dataclasses import replace
 def partial_step(world,*a,**kw):
  world.time+=.123
  raise RuntimeError("test-only partial-step failure")
 runtime["observed"]=replace(runtime["observed"],function=partial_step)
result=windows.execute_window(args,runtime)
if injected_failure:
 assert result["status"]=="IMPLEMENTATION_FAILURE",result
 heartbeat=json.loads((output/"heartbeat.json").read_bytes())
 assert heartbeat["last_complete_sim_time"]==.35
 assert result["last_complete_sim_time"]==.35 and result["actual_stop_sim_time"]==.473
 assert result["last_complete_state_checkpoint"]==str(checkpoint)
 assert (source/"states.jsonl").read_bytes()==before
 w.state_sink.close()
 print(json.dumps({"status":result["status"],"last_complete":heartbeat["last_complete_sim_time"]}))
 raise SystemExit(0)
assert result["status"]=="COMPLETED_WINDOW",result
assert bool(result["migration_id"]) is migration_fixture
assert result["original_advance_target"]==3600. and result["outer_phase"]=="arrival_window"
assert result["start_sim_time"]==.35 and result["actual_sim_delta_s"]>=10.
assert result["simulation_deferral_s"]>0
assert not result["physical_outcome"]["episode_complete"] and not result["hard_timeout_enforced"]
assert (source/"states.jsonl").read_bytes()==before
w.state_sink.close()
# Uninterrupted run with the SAME original 3600 target and same natural hook
# threshold, so neither comparison inserts a new physical interval boundary.
direct=output.parent/(output.name+"-direct")
direct.mkdir()
baseline=runtime["world_module"].RotatingSlotWorld(config=runtime["config"],
 state_sink=runtime["replay_module"].BufferedStateJSONLSink(direct/"states.jsonl"))
baseline.seed_service("fixture-service",pad_id=0,remaining=.35)
baseline_driver=runtime["driver_module"].DeterministicDriver()
def finish(world,cursor,driver):
 if world.time>=result["nominal_stop_sim_time"]: raise SuspendAtCut()
try: runtime["observed"].run(baseline,3600.,baseline_driver,finish,outer_phase="arrival_window")
except SuspendAtCut: pass
baseline.flush_records()
assert (direct/"states.jsonl").read_bytes()==(output/"states.jsonl").read_bytes()
assert baseline.events==[json.loads(line) for line in (output/"replay/events.jsonl").read_bytes().splitlines()]
assert baseline_driver.calls==result["end_counts"]["driver_calls"]
archive=runtime["replay_module"].ReplayBundle.validate(output/"replay")
assert archive.time_bounds[1]==baseline.time
baseline.state_sink.close()
last=load_checkpoint(result["terminal_checkpoint"],expected_identity=identity)
last_restored=restore_checkpoint(last,expected_identity=identity,cache_mode="cold",
 new_attempt_dir=output.parent/(output.name+"-terminal-restore"),world_class=runtime["world_module"].RotatingSlotWorld)
assert last_restored.world.time==result["actual_stop_sim_time"]
assert last_restored.cursor.original_target==3600.
last_restored.world.state_sink.close()
# The public executor must reject an existing output before copying state.
try: windows.execute_window(args,runtime)
except windows.WindowError as exc: assert "output already exists" in str(exc)
else: raise AssertionError("existing output was overwritten")
print(json.dumps({"status":result["status"],"output":str(output),"sim_delta":result["actual_sim_delta_s"],
 "deferral":result["simulation_deferral_s"],"source_unchanged":True,"same_target_records_equal":True}))
'''


@pytest.mark.skipif(os.environ.get("RUN_WINDOW_INTEGRATION") != "1", reason="requires stable implementation fingerprint")
def test_real_cold_window_fixture_profile_and_benchmark_same_records(tmp_path):
    assert (REFERENCE / ".reference_manifest.json").is_file()
    # Local fingerprint generation in a separate stdlib/helper-only process.
    deployment = tmp_path / "deployment.json"
    code = "import sys,pathlib,json;sys.dont_write_bytecode=True;root=pathlib.Path(sys.argv[1]);sys.path.insert(0,str(root/'terminal_marl/core/rotating_slots'));from performance.provenance import implementation_fingerprint;m=implementation_fingerprint(root);m['implementation_commit']='1c5afd37c891a0ed4776f64367fc7264db3c1c54';pathlib.Path(sys.argv[2]).write_text(json.dumps(m))"
    process = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(ROOT), str(deployment)],
                             cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, (process.stdout, process.stderr)
    evidence = []
    for name, mode, extra in (("benchmark", "benchmark", []), ("profile", "profile", []),
                               ("migration", "benchmark", ["migration"])):
        output = tmp_path / name
        child = subprocess.run([sys.executable, "-I", "-B", "-c", FIXTURE_CHILD,
            str(ROOT), str(REFERENCE), str(deployment), str(output), mode, *extra], cwd=ROOT,
            capture_output=True, text=True, timeout=120)
        (tmp_path / (name + ".stdout.txt")).write_text(child.stdout, encoding="utf8")
        (tmp_path / (name + ".stderr.txt")).write_text(child.stderr, encoding="utf8")
        assert child.returncode == 0, (child.stdout, child.stderr)
        evidence.append(json.loads(child.stdout))
        result = json.loads((output / "result.json").read_bytes())
        assert result["performance_status"] == "NOT_QUALIFIED" and result["human_status"] == "HUMAN_PENDING"
        assert (output / "window.prof").exists() is (mode == "profile")
        assert result["start_counts"]["kernel_work_counts"] == "MISSING"
        assert result["timing"]["rollout_wall_s"] > 0 and result["timing"]["rollout_parent_process_cpu_s"] > 0
        assert result["terminal_checkpoint"] and Path(result["terminal_checkpoint"]).exists()
    assert (tmp_path / "benchmark/states.jsonl").read_bytes() == (tmp_path / "profile/states.jsonl").read_bytes()
    assert (tmp_path / "benchmark/states.jsonl").read_bytes() == (tmp_path / "migration/states.jsonl").read_bytes()
    missing = subprocess.run([sys.executable, "-I", "-B", "-c", FIXTURE_CHILD,
        str(ROOT), str(REFERENCE), str(deployment), str(tmp_path / "missing-approval"), "benchmark", "missing_approval"],
        cwd=ROOT, capture_output=True, text=True, timeout=120)
    (tmp_path / "missing-approval.stdout.txt").write_text(missing.stdout, encoding="utf8")
    (tmp_path / "missing-approval.stderr.txt").write_text(missing.stderr, encoding="utf8")
    assert missing.returncode == 0, (missing.stdout, missing.stderr)
    failure_output = tmp_path / "partial-failure"
    failed = subprocess.run([sys.executable, "-I", "-B", "-c", FIXTURE_CHILD,
        str(ROOT), str(REFERENCE), str(deployment), str(failure_output), "benchmark", "inject_failure"],
        cwd=ROOT, capture_output=True, text=True, timeout=120)
    (tmp_path / "partial-failure.stdout.txt").write_text(failed.stdout, encoding="utf8")
    (tmp_path / "partial-failure.stderr.txt").write_text(failed.stderr, encoding="utf8")
    assert failed.returncode == 0, (failed.stdout, failed.stderr)
    (tmp_path / "fixture_evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf8")
    # Model a stale launcher digest without mutating shared repository bytes.
    stale = json.loads(deployment.read_bytes())
    stale["actual_source_files_sha256"]["scripts/server/run_rotating_slots_performance_windows.py"] = "0" * 64
    stale["implementation_sha256"] = provenance.digest(stale["actual_source_files_sha256"])
    stale_path = tmp_path / "stale-deployment.json"
    stale_path.write_text(json.dumps(stale), encoding="utf8")
    rejected = subprocess.run([sys.executable, "-I", "-B", str(SCRIPT), "window",
        "--corpus-manifest", str(tmp_path / "unused-corpus.json"), "--corpus-sha256", "a" * 64,
        "--window", "W71-A", "--duration", "10", "--kernel", "R0", "--reference", str(REFERENCE),
        "--deployment-manifest", str(stale_path), "--implementation-commit", provenance.R0_COMMIT,
        "--output", str(tmp_path / "must-not-create")], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert rejected.returncode == 2, (rejected.stdout, rejected.stderr)
    assert "DEPLOYMENT_BYTES_MISMATCH" in json.loads(rejected.stdout)["error"]
    assert not (tmp_path / "must-not-create").exists()
