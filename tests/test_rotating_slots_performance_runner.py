"""Fresh-process P0/P1 runner integration against the byte-frozen R0 export."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/server/run_rotating_slots_performance.py"
REFERENCE = ROOT / "artifacts/rotating_slots/performance_reference_r0"
COMMIT = "1c5afd37c891a0ed4776f64367fc7264db3c1c54"


def invoke(command, manifest, *extra, expected=0):
    args = [sys.executable, "-B", str(SCRIPT), command, "--reference", str(REFERENCE),
            "--deployment-manifest", str(manifest), "--implementation-commit", COMMIT, *map(str, extra)]
    process = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert process.returncode == expected, (process.stdout, process.stderr)
    return json.loads(process.stdout)


@pytest.fixture
def manifest(tmp_path):
    if not (REFERENCE / ".reference_manifest.json").is_file():
        pytest.fail("fixed R0 export is required; create via P0 export_reference locally")
    path = tmp_path / "deployment.json"
    result = invoke("preflight", path, "--write-deployment-manifest")
    assert result["status"] == "PREFLIGHT_OK"
    return path


def budget(workers=0):
    return ["--inner-workers", workers, "--core-budget", max(1, workers),
            "--process-budget", workers + 1, "--wall-budget", 90]


def document(path):
    return json.loads(Path(path).read_bytes())


def semantic_bundle(path):
    path = Path(path)
    rows = [json.loads(line) for line in (path / "states.jsonl").read_bytes().splitlines()]
    # Frame metrics are physical accounting; R0 timers/cache work live outside
    # the replay metrics and do not require filtering here.
    return {"states": rows, "events": (path / "events.jsonl").read_bytes(),
            "rejections": (path / "rejections.jsonl").read_bytes(),
            "trajectories": document(path / "trajectories.json"),
            "manifest": document(path / "manifest.json"), "metrics": document(path / "metrics.json")}


def test_cli_help_only_implemented_commands():
    process = subprocess.run([sys.executable, "-B", str(SCRIPT), "--help"], cwd=ROOT,
                             capture_output=True, text=True, timeout=30)
    assert process.returncode == 0
    assert "{preflight,capture,micro}" in process.stdout
    for command in ("preflight", "capture", "micro"):
        sub = subprocess.run([sys.executable, "-B", str(SCRIPT), command, "--help"], cwd=ROOT,
                             capture_output=True, text=True, timeout=30)
        assert sub.returncode == 0


def test_preflight_exact_reference_and_external_helper_imports(manifest):
    result = invoke("preflight", manifest)
    source = result["source_identity"]
    assert source["kernel_commit"] == source["reference_commit"] == COMMIT
    assert source["reference_source_sha256"] == "cd76b4c14b0a48d6d2a71ec7c63ce77010e6ea019b545807635b926eb2984891"
    assert all(Path(path).is_relative_to(REFERENCE) for path in source["imported_module_paths"]["core"].values())
    assert all(Path(path).is_relative_to(ROOT / "terminal_marl/core/rotating_slots/performance")
               for path in source["imported_module_paths"]["helpers"].values())
    assert "scripts/server/run_rotating_slots_performance.py" in source["helper_execution_files_sha256"]
    assert "terminal_marl/core/rotating_slots/performance/runner.py" in source["helper_execution_files_sha256"]


def test_real_R0_micro_serial_pool_full_records_equal(manifest, tmp_path):
    serial, parallel = tmp_path / "serial", tmp_path / "parallel"
    for path, workers in ((serial, 0), (parallel, 1)):
        output = invoke("micro", manifest, "--output", path, *budget(workers), "--fixture", "inner-hold", "--until", .4)
        assert output["status"] == "COMPLETED_MICRO", output
    a, b = document(serial / "result.json"), document(parallel / "result.json")
    assert semantic_bundle(serial / "replay") == semantic_bundle(parallel / "replay")
    assert a["recorded_state_count"] == b["recorded_state_count"] == 3
    assert a["physical_outcome"] == b["physical_outcome"]
    assert a["last_reference_fallback"] == b["last_reference_fallback"]
    assert b["resources"]["closed_and_joined"]
    assert b["resources"]["resources"]["cpu_complete"]
    assert b["resources"]["pad_tasks_completed"] > 0
    assert b["source_identity"]["prefix_execution"] == "R0_PARALLEL"
    assert (parallel / "rss_samples.jsonl").stat().st_size > 0
    assert a["execution_backend"] == "R0_SERIAL" and b["execution_backend"] == "R0_PARALLEL"
    assert not a["hard_timeout_enforced"] and a["performance_status"] == "NOT_QUALIFIED"
    for path in (serial, parallel):
        result = document(path / "result.json")
        for filename in ("observation.patch", "observation_generated.py", "observation_original.py"):
            assert (path / filename).stat().st_size > 0
        assert _file_sha(path / "observation_generated.py") == result["source_identity"]["observation_generated_source_sha256"]
        assert _file_sha(path / "observation_original.py") == result["source_identity"]["observation_original_source_sha256"]
        assert all(Path(p).is_relative_to(REFERENCE) for p in result["final_imported_module_paths"]["core"].values())


def test_isolated_R0_real_hold_nohook_hook_and_cold_warm_restore(manifest):
    # Fresh isolated interpreter, external helpers, one core namespace. No test
    # replacements of reference, energy, emergency monitoring or propagation.
    code = r'''
import json, pathlib, sys, copy
reference, helper = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
sys.path.insert(0, str(helper / "terminal_marl/core/rotating_slots"))
sys.path.insert(0, str(reference))
from terminal_marl.core.rotating_slots.world import RotatingSlotWorld
from terminal_marl.core.rotating_slots.config import load_config
from performance.checkpoint import CheckpointIdentity, capture_checkpoint, restore_checkpoint
from performance.observation import SuspendAtCut, make_observed_advance
c = load_config()
identity = CheckpointIdentity("upstream_capacity_only_pad_arrival_v1_20261018", c.config_hash,
 c.profile_hash, c.raw_hash, "test-rules", "test-candidate", "test-numeric", "test-execution")
def scene():
 w = RotatingSlotWorld(); w.seed_hold("real", "inner", 0, soc=.9); return w
def semantic(w):
 value = copy.deepcopy(w.to_dict())
 for key in ("cpu_seconds", "cache", "cache_hits", "reference_counts"): value.pop(key, None)
 return value
plain = scene(); plain.advance(.4, False)
obs = make_observed_advance(RotatingSlotWorld)
hooked = scene(); obs.run(hooked, .4, False, lambda *a: None)
interrupted = scene(); saved = []
def hook(w, cursor, driver):
 if w.time == .2:
  saved.append(capture_checkpoint(w, identity=identity, cursor=cursor, driver=driver)); raise SuspendAtCut()
try: obs.run(interrupted, .4, False, hook)
except SuspendAtCut: pass
parity = []
for mode in ("cold", "warm"):
 restored = restore_checkpoint(saved[0], expected_identity=identity, world_class=RotatingSlotWorld, cache_mode=mode)
 obs.resume(restored)
 parity.append(semantic(restored.world) == semantic(plain))
assert semantic(plain) == semantic(hooked) and all(parity)
assert all(pathlib.Path(m.__file__).resolve().is_relative_to(reference) for n,m in sys.modules.items()
 if n == "terminal_marl" or n.startswith("terminal_marl."))
a = plain.aircraft["real"]
print(json.dumps({"parity": parity, "soc_hex": a.soc.hex(), "qe_hex": a.emergency.q_e.hex(),
 "hash": a.trajectory_hash, "decision_times": plain.decision_times,
 "frame_times": [f["time"] for f in plain.records]}))
'''
    process = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(REFERENCE), str(ROOT)],
                             cwd=REFERENCE, capture_output=True, text=True, timeout=120)
    assert process.returncode == 0, (process.stdout, process.stderr)
    result = json.loads(process.stdout)
    assert result["parity"] == [True, True]
    assert result["frame_times"] == [0., .2, .4] and result["decision_times"] == [0.]


def test_capture_natural_prefix_resume_new_attempt_and_ledger(manifest, tmp_path):
    first, resumed, direct = (tmp_path / name for name in ("first", "resumed", "direct"))
    common = ["--rate", 60, "--seed", 23]
    invoke("capture", manifest, "--output", first, *budget(), *common, "--capture-until", .4)
    source_result = document(first / "result.json")
    assert source_result["status"] == "SUSPENDED_CAPTURE"
    assert source_result["original_advance_target"] == 3600.
    assert source_result["workload"]["arrival_window_s"] == 3600.
    assert source_result["replay_scope"] == "complete_observed_prefix"
    assert not source_result["physical_outcome"]["episode_complete"]
    before = (first / "states.jsonl").read_bytes()
    checkpoint = source_result["checkpoints"][-1]
    invoke("capture", manifest, "--output", resumed, *budget(), *common, "--capture-until", .8,
           "--checkpoint", checkpoint)
    invoke("capture", manifest, "--output", direct, *budget(), *common, "--capture-until", .8)
    assert (first / "states.jsonl").read_bytes() == before
    assert semantic_bundle(resumed / "replay") == semantic_bundle(direct / "replay")
    result = document(resumed / "result.json")
    previous_ledger = document(first / "attempt_ledger.json")
    assert result["timing"]["prior_consumed_wall_s"] == previous_ledger["cumulative_effective_wall_s"]
    assert result["timing"]["cumulative_effective_wall_s"] >= previous_ledger["cumulative_effective_wall_s"]
    assert result["driver_calls"] == [0.]
    assert result["wall_accounting"] == "COMPLETE"


def test_micro_profile_is_bounded_and_benchmark_has_no_profiler(manifest, tmp_path):
    out = tmp_path / "profile"
    invoke("micro", manifest, "--output", out, *budget(), "--fixture", "service-events", "--mode", "profile", "--until", .4)
    result = document(out / "result.json")
    assert result["timing"]["mode"] == "profile"
    assert result["timing"]["profile_timer"] == "time.process_time"
    assert (out / "micro.prof").stat().st_size > 0
    rows = semantic_bundle(out / "replay")
    assert b"pad_service_completed" in rows["events"]


def _file_sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def test_OPT_and_R0_bind_identical_frozen_config_bytes(manifest, tmp_path):
    r0, opt = tmp_path / "r0", tmp_path / "opt"
    for output, kernel in ((r0, "R0"), (opt, "OPT")):
        invoke("micro", manifest, "--output", output, *budget(), "--kernel", kernel,
               "--fixture", "service-events", "--until", .4)
    reference, current = document(r0 / "result.json"), document(opt / "result.json")
    a, b = reference["source_identity"], current["source_identity"]
    assert a["physical_config_path"] == b["physical_config_path"] == str(REFERENCE / "new10.16/fixed_terminal_rotating_slots_stage1_config_v1_0.yaml")
    assert a["execution_physical_config_bytes_sha256"] == b["execution_physical_config_bytes_sha256"]
    assert a["semantic_fingerprint"] == b["semantic_fingerprint"]
    assert semantic_bundle(r0 / "replay") == semantic_bundle(opt / "replay")
    assert all(Path(path).is_relative_to(ROOT / "terminal_marl") for path in current["final_imported_module_paths"]["core"].values())


def test_source_mislabel_and_forged_byte_manifest_rejected_before_output(manifest, tmp_path):
    wrong = deepcopy(document(manifest)); wrong["implementation_commit"] = "f67b5c8462433ba701191c493a37e4a8e472334e"
    path = tmp_path / "wrong_commit.json"; path.write_text(json.dumps(wrong))
    output = tmp_path / "must-not-exist"
    result = invoke("micro", path, "--output", output, *budget(), expected=2)
    assert "DEPLOYMENT_COMMIT_MISMATCH" in result["error"] and not output.exists()
    wrong = deepcopy(document(manifest))
    filename = "terminal_marl/core/rotating_slots/performance/runner.py"
    wrong["actual_source_files_sha256"][filename] = "0" * 64
    wrong["implementation_sha256"] = sha256(json.dumps(wrong["actual_source_files_sha256"], sort_keys=True,
        separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    path = tmp_path / "wrong_bytes.json"; path.write_text(json.dumps(wrong))
    result = invoke("micro", path, "--output", output, *budget(), expected=2)
    assert "DEPLOYMENT_BYTES_MISMATCH" in result["error"] and not output.exists()


def test_invalid_capture_requests_and_budgets_rejected(manifest, tmp_path):
    out = tmp_path / "invalid"
    result = invoke("capture", manifest, "--output", out, *budget(), "--rate", 60, "--seed", 23,
                    "--capture-until", .4, "--milestones", .2, expected=2)
    assert "latest milestone + 60" in result["error"] and not out.exists()
    result = invoke("micro", manifest, "--output", out, "--inner-workers", 1, "--core-budget", 1,
                    "--process-budget", 1, "--wall-budget", 90, expected=2)
    assert "budget" in result["error"] and not out.exists()


def test_soft_budget_saved_at_original_safe_cut_not_physical_failure(manifest, tmp_path):
    out = tmp_path / "soft-timeout"
    result = invoke("capture", manifest, "--output", out, "--inner-workers", 0, "--core-budget", 1,
                    "--process-budget", 1, "--wall-budget", .00001, "--rate", 60, "--seed", 23, "--capture-until", .4)
    assert result["status"] == "PERFORMANCE_TIMEOUT"
    report = document(out / "result.json")
    assert report["physical_outcome"]["failed"] is False
    assert report["original_advance_target"] == 3600.
    assert report["physical_outcome"]["sim_time"] == .2
    assert report["checkpoints"] and not report["hard_timeout_enforced"]
