"""Deterministic corpus selection and bounded actual frozen-R0 validation."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "terminal_marl/core/rotating_slots"))
from performance import corpus, provenance
from performance.checkpoint import write_checkpoint

REFERENCE = ROOT / "artifacts/rotating_slots/performance_reference_r0"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf8")


def cut(t, requests=(), pinned=False):
    return {"actual_sim_time": t, "requested_sim_times": list(requests), "pinned": pinned, "corpus": True}


def selection_sources():
    return {case: {"status": "SUSPENDED_CAPTURE", "stop_time": max(t for _, c, t in corpus.WINDOWS if c == case) + 60.,
                   "workload": {"seed": corpus.CASES[case][1]},
                   "cuts": [cut(t, [t], True) for _, c, t in corpus.WINDOWS if c == case]}
            for case in corpus.CASES}


def test_fixed_order_preferred_and_deferred_actual_cut():
    sources = selection_sources()
    sources["rate180_seed71"]["cuts"][0]["actual_sim_time"] = 300.2
    rows = corpus.select_windows(sources)
    assert [r["window"] for r in rows] == [w for w, _, _ in corpus.WINDOWS]
    assert rows[0]["cut"]["actual_sim_time"] == 300.2
    assert all(r["status"] == "AVAILABLE" for r in rows)
    assert rows[0]["original_requested_sim_time"] == 300.


def test_physical_early_stop_latest_cut_60_and_same_seed_30_boundaries():
    sources = selection_sources()
    sources["rate180_seed71"] = {"status": "PHYSICAL_FAILURE", "stop_time": 250., "workload": {"seed": 71},
        "cuts": [cut(t) for t in (100., 129.999, 130., 159.999, 160., 189.999, 190., 190.001)]}
    rows = corpus.select_windows(sources)
    assert [r["cut"]["actual_sim_time"] for r in rows[:3]] == [190., 160., 130.]
    assert all("availability_only" in r["selection_reason"] for r in rows[:3])


def test_unavailable_no_short_or_arbitrary_suspended_substitution():
    sources = selection_sources()
    sources["rate180_seed71"] = {"status": "PHYSICAL_FAILURE", "stop_time": 59.999, "cuts": [cut(0.)], "workload": {"seed": 71}}
    sources["rate60_seed23"]["cuts"] = [cut(500.)]
    rows = corpus.select_windows(sources)
    assert all(r["status"] == "UNAVAILABLE" for r in rows[:4])
    sources["rate180_seed71"] = {"status": "PHYSICAL_FAILURE", "stop_time": 100., "cuts": [cut(40.), cut(10.)], "workload": {"seed": 71}}
    assert [r["status"] for r in corpus.select_windows(sources)[:3]] == ["AVAILABLE", "AVAILABLE", "UNAVAILABLE"]


def test_missing_preferred_after_observed_suffix_does_not_allow_replacement():
    sources = selection_sources()
    sources["rate60_seed23"] = {"status": "PHYSICAL_FAILURE", "stop_time": 700., "cuts": [cut(640.)], "workload": {"seed": 23}}
    row = corpus.select_windows(sources)[3]
    assert row["status"] == "UNAVAILABLE"
    assert row["selection_reason"] == "missing_preferred_pin_after_observed_60s_suffix"


def test_same_seed_spacing_applies_across_rates_with_no_arbitrary_replacement():
    sources = selection_sources()
    sources["rate120_seed11"] = {"status": "PHYSICAL_FAILURE", "stop_time": 590.,
        "cuts": [cut(530.)], "workload": {"seed": 11}}
    rows = corpus.select_windows(sources)
    assert rows[4]["cut"]["actual_sim_time"] == 530.
    assert rows[5]["status"] == "UNAVAILABLE"
    assert rows[5]["selection_reason"] == "preferred_same_seed_spacing_conflict"
    sources["rate180_seed11"] = {"status": "PHYSICAL_FAILURE", "stop_time": 580.,
        "cuts": [cut(520.), cut(500.)], "workload": {"seed": 11}}
    assert corpus.select_windows(sources)[5]["cut"]["actual_sim_time"] == 500.


@pytest.fixture
def reference_data():
    return provenance._verify_export(REFERENCE), provenance.semantic_fingerprint(REFERENCE)


@pytest.fixture
def campaign(tmp_path, reference_data):
    manifest, semantics = reference_data
    campaign = tmp_path / "campaign"
    helpers = {"terminal_marl/core/rotating_slots/performance/" + n: corpus._sha(ROOT / "terminal_marl/core/rotating_slots/performance" / n)
               for n in ("__init__.py", "checkpoint.py")}
    for case, (rate, seed) in corpus.CASES.items():
        root = (campaign / case).resolve()
        root.mkdir(parents=True)
        for n in ("observation_original.py", "observation_generated.py"):
            (root / n).write_bytes(b"fixture\n")
        shim = corpus._sha(root / "observation_generated.py")
        source = {"kernel": "R0", "kernel_commit": provenance.R0_COMMIT, "reference_commit": provenance.R0_COMMIT,
            "reference_source_sha256": provenance.R0_SOURCE_SHA256, "actual_core_files_sha256": manifest["actual_source_files_sha256"],
            "semantic_fingerprint": semantics, "source_semantic_projection": semantics,
            "execution_physical_config_bytes_sha256": semantics["physical_config_bytes_sha256"],
            "source_physical_config_bytes_sha256": semantics["physical_config_bytes_sha256"],
            "prefix_execution": "R0_SERIAL", "helper_execution_files_sha256": helpers,
            "observation_generated_source_sha256": shim, "observation_original_source_sha256": shim}
        source["runner_execution_sha256"] = provenance.digest({"kernel": "R0", "core": source["actual_core_files_sha256"],
                "helpers": helpers, "observation_shim": shim})
        workload = {"kind": "full_traffic_prefix", "rate_per_hour": rate, "seed": seed,
                    "arrival_window_s": 3600., "external_phase_targets": [3600., 5400.]}
        rows, cuts = [], []
        for sequence, requested in enumerate(t for _, c, t in corpus.WINDOWS if c == case):
            path = root / "checkpoints" / f"checkpoint-{sequence:08d}.json"
            write(path, {"synthetic": True})
            evidence = {"actual_sim_time": requested, "requested_sim_times": [requested],
                        "simulation_deferral_s": [0.], "reasons": ["requested_milestone"]}
            sidecar = {"schema": "ft_rs_runner_checkpoint_v1_0", "checkpoint_sha256": corpus._sha(path),
                "runner_execution_sha256": source["runner_execution_sha256"], "source_attempt": str(root),
                "workload": workload, "cumulative_effective_wall_at_checkpoint_s": 1., "cut": evidence,
                "cursor": {"original_target": 3600., "phase": "loop_start", "outer_phase": "arrival_window"}}
            write(path.with_suffix(".runner.json"), sidecar)
            rows.append({"sequence": sequence, "file": path.name, "sha256": corpus._sha(path),
                         "pinned": True, "corpus": True, "milestone": [requested]})
            cuts.append(evidence)
        stop = max(t for _, c, t in corpus.WINDOWS if c == case) + 60.
        result = {"schema": "ft_rs_performance_attempt_v1_0", "status": "SUSPENDED_CAPTURE", "stop_reason": "SUSPENDED_CAPTURE",
            "finished_utc": "finish", "started_utc": "start", "pid": 123, "failure": None, "wall_accounting": "COMPLETE",
            "source_identity": source, "execution_backend": "R0_SERIAL", "inner_workers": 0,
            "subcommand": "capture", "workload": workload, "current_phase": "arrival_window", "original_advance_target": 3600.,
            "physical_outcome": {"sim_time": stop, "failed": False, "episode_complete": False, "failure_time": None},
            "target_sim_time": stop, "cuts": cuts, "recorded_state_count": 1,
            "checkpoints": [str(root / "checkpoints" / r["file"]) for r in rows],
            "replay_path": str(root / "replay")}
        ledger = {"schema": "ft_rs_attempt_ledger_v1_0", "status": "SUSPENDED_CAPTURE", "pid": 123, "started_utc": "start",
            "wall_accounting": "COMPLETE", "runner_execution_sha256": source["runner_execution_sha256"], "cumulative_effective_wall_s": 2.}
        for n, value in (("result.json", result), ("attempt_ledger.json", ledger), ("source_identity.json", source),
                          ("checkpoints/checkpoint_index.json", {"schema": "ft_rs_checkpoint_v1_0", "entries": rows})):
            write(root / n, value)
        (campaign / (case + ".exitcode.txt")).write_text("0\n")
    return campaign


@pytest.mark.parametrize("mutation,match", [("running", "terminal"), ("timeout", "terminal"), ("exit", "exit code"),
    ("hash", "hash mismatch"), ("policy", "configuration"), ("execution", "execution hash"),
    ("short", "target\\+60"), ("sourcepath", "absolute paths"), ("missing", "No such file|cannot find")])
def test_source_negatives(campaign, reference_data, mutation, match):
    case = "rate60_seed23"
    root = campaign / case
    result, ledger, source = (corpus._read(root / n) for n in ("result.json", "attempt_ledger.json", "source_identity.json"))
    if mutation in ("running", "timeout"):
        result["status"] = result["stop_reason"] = ledger["status"] = "RUNNING" if mutation == "running" else "PERFORMANCE_TIMEOUT"
    elif mutation == "exit":
        (campaign / (case + ".exitcode.txt")).write_text("2\n")
    elif mutation == "hash":
        (root / "checkpoints/checkpoint-00000000.json").write_bytes(b"changed")
    elif mutation == "policy":
        source["semantic_fingerprint"]["policy_id"] = "H0"
        result["source_identity"] = source
    elif mutation == "execution":
        source["runner_execution_sha256"] = "0" * 64
        result["source_identity"] = source
    elif mutation == "short":
        result["target_sim_time"] = 659.9
    elif mutation == "sourcepath":
        sidecar = root / "checkpoints/checkpoint-00000000.runner.json"
        value = corpus._read(sidecar); value["source_attempt"] = "/nonlocal/source"
        write(sidecar, value)
    elif mutation == "missing":
        (root / "checkpoints/checkpoint-00000000.runner.json").unlink()
    for n, value in (("result.json", result), ("attempt_ledger.json", ledger), ("source_identity.json", source)):
        write(root / n, value)
    with pytest.raises((corpus.CorpusError, OSError), match=match):
        corpus.read_source(campaign, case, *reference_data)


def fake_validation(reference, sources, rows, validation_root):
    return {"replays": {c: {"test_double": True} for c in sources},
            "checkpoints": {r["window"]: {"test_double": True} for r in rows if r["status"] == "AVAILABLE"}}


def test_atomic_freeze_copies_and_digest_no_overwrite(campaign, tmp_path, monkeypatch):
    monkeypatch.setattr(corpus, "validate_reference_inputs", fake_validation)
    output = tmp_path / "frozen"
    result = corpus.freeze_corpus(campaign, REFERENCE, output)
    assert result["status"] == "FROZEN" and result["performance_status"] == "NOT_QUALIFIED"
    assert len(result["windows"]) == 6
    digest = result.pop("manifest_sha256")
    assert provenance.digest(result) == digest
    for row in result["windows"]:
        assert Path(row["frozen_checkpoint"]).read_bytes() == Path(row["cut"]["checkpoint"]).read_bytes()
        assert Path(row["frozen_sidecar"]).read_bytes() == Path(row["cut"]["sidecar"]).read_bytes()
    with pytest.raises(corpus.CorpusError, match="overwrite"):
        corpus.freeze_corpus(campaign, REFERENCE, output)


def test_failed_validation_has_no_final_manifest(campaign, tmp_path, monkeypatch):
    def fail(*args):
        raise ValueError("validation fixture failure")
    monkeypatch.setattr(corpus, "validate_reference_inputs", fail)
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="fixture failure"):
        corpus.freeze_corpus(campaign, REFERENCE, output)
    assert not (output / "corpus_manifest.json").exists()
    assert corpus._read(output / "freeze_report.json")["status"] == "INCOMPLETE"


def test_unavailable_window_publishes_only_incomplete_manifest(campaign, tmp_path, monkeypatch):
    root = campaign / "rate180_seed71"
    result = corpus._read(root / "result.json")
    result["status"] = result["stop_reason"] = "PHYSICAL_FAILURE"
    result["physical_outcome"].update(sim_time=59., failed=True, failure_time=59.)
    result["cuts"], result["checkpoints"] = [], []
    ledger = corpus._read(root / "attempt_ledger.json"); ledger["status"] = "PHYSICAL_FAILURE"
    write(root / "result.json", result); write(root / "attempt_ledger.json", ledger)
    write(root / "checkpoints/checkpoint_index.json", {"schema": "ft_rs_checkpoint_v1_0", "entries": []})
    monkeypatch.setattr(corpus, "validate_reference_inputs", fake_validation)
    observed = corpus.freeze_corpus(campaign, REFERENCE, tmp_path / "incomplete")
    assert observed["status"] == "INCOMPLETE"
    assert [r["status"] for r in observed["windows"][:3]] == ["UNAVAILABLE"] * 3


def test_copy_failure_cannot_publish_frozen_manifest(campaign, tmp_path, monkeypatch):
    monkeypatch.setattr(corpus, "validate_reference_inputs", fake_validation)
    original = corpus.shutil.copyfile
    calls = []
    def fail_second(src, dst):
        calls.append(str(dst))
        if len(calls) == 2:
            raise OSError("copy fixture failure")
        return original(src, dst)
    monkeypatch.setattr(corpus.shutil, "copyfile", fail_second)
    output = tmp_path / "copy_failed"
    with pytest.raises(OSError, match="copy fixture"):
        corpus.freeze_corpus(campaign, REFERENCE, output)
    assert len(calls) == 2 and Path(calls[0]).is_file()
    assert not (output / "corpus_manifest.json").exists()
    assert corpus._read(output / "freeze_report.json")["status"] == "INCOMPLETE"


@pytest.fixture
def real_fixture(tmp_path, reference_data):
    root = (tmp_path / "actual_R0").resolve()
    root.mkdir()
    _, semantics = reference_data
    helper = ROOT / "terminal_marl/core/rotating_slots/performance"
    helpers = {"terminal_marl/core/rotating_slots/performance/" + n: corpus._sha(helper / n) for n in ("__init__.py", "checkpoint.py")}
    source = {"helper_execution_files_sha256": helpers, "semantic_fingerprint": semantics,
              "runner_execution_sha256": "fixture-exact-execution", "observation_generated_source_sha256": "fixture-shim"}
    identity = {k: semantics[k] for k in ("policy_id", "config_hash", "profile_hash", "physical_config_bytes_sha256",
                 "effective_rules_hash", "candidate_policy_hash", "numerical_policy_hash")}
    identity["implementation_sha256"] = source["runner_execution_sha256"]
    code = "import pathlib, sys\nroot = pathlib.Path(" + repr(str(root)) + ")\nhelper = " + repr(str(helper.parent)) + "\nidentity = " + repr(identity) + "\n" + r'''
sys.path.insert(0, helper)
from performance.checkpoint import CheckpointIdentity, capture_checkpoint, write_checkpoint
from performance.observation import make_observed_advance, SuspendAtCut
from terminal_marl.core.rotating_slots.world import RotatingSlotWorld
from terminal_marl.core.rotating_slots.demand import ArrivalManifest
from terminal_marl.core.rotating_slots.replay import BufferedStateJSONLSink, ReplayBundle
world = RotatingSlotWorld(state_sink=BufferedStateJSONLSink(root / "states.jsonl"))
world.manifest = ArrivalManifest.generate(23, 60, 3600.)
observed = make_observed_advance(RotatingSlotWorld)
def hook(w, cur, driver):
    if w.time == .2:
        payload = capture_checkpoint(w, identity=CheckpointIdentity(**identity), cursor=cur, driver=driver)
        write_checkpoint(root / "checkpoint.json", payload)
    if w.time >= .4:
        raise SuspendAtCut()
try:
    observed.run(world, 3600., False, hook, outer_phase="arrival_window")
except SuspendAtCut:
    pass
world.flush_records()
ReplayBundle.write(root / "replay", world=world, config_path=world.config.path,
    code_commit="1c5afd37c891a0ed4776f64367fc7264db3c1c54", seed=23, eager=False)
world.state_sink.close()
query_result = {"time": world.time, "shim": observed.generated_source_sha256, "state_count": world._record_count}
'''
    evidence = provenance.run_reference(REFERENCE, code=code)["query_result"]
    source["observation_generated_source_sha256"] = evidence["shim"]
    sources = {"rate60_seed23": {"source_attempt": str(root), "stop_time": .4, "source_identity": source,
                "recorded_state_count": evidence["state_count"], "workload": {"seed": 23, "rate_per_hour": 60}}}
    selected = [{"status": "AVAILABLE", "window": "fixture", "case": "rate60_seed23", "cut": {
        "checkpoint": str(root / "checkpoint.json"), "actual_sim_time": .2,
        "cursor": {"original_target": 3600., "phase": "loop_start", "outer_phase": "arrival_window"}}}]
    return root, sources, selected


def test_real_frozen_R0_replay_and_checkpoint_restore(real_fixture, tmp_path):
    root, sources, selected = real_fixture
    before = (root / "states.jsonl").read_bytes()
    evidence = corpus.validate_reference_inputs(REFERENCE, sources, selected, tmp_path / "validated")
    assert evidence["replays"]["rate60_seed23"]["time_bounds"] == [0., .4]
    assert evidence["checkpoints"]["fixture"]["world_time"] == .2
    assert (root / "states.jsonl").read_bytes() == before
    assert (tmp_path / "validated/fixture/states.jsonl").is_file()


@pytest.mark.parametrize("mutation,match", [("cursor", "cursor/world/workload"), ("policy", "identity mismatch"),
    ("prefix", "record prefix"), ("missingprefix", "No such file|cannot find"), ("pending", "pending frame hash"),
    ("replay", "replay content hash")])
def test_real_validation_negatives(real_fixture, tmp_path, mutation, match):
    root, sources, selected = real_fixture
    path = root / "checkpoint.json"
    payload = corpus._read(path)["payload"]
    if mutation == "cursor":
        selected[0]["cut"]["cursor"]["original_target"] = 5400.
    elif mutation == "policy":
        payload["identity"]["effective_rules_hash"] = "wrong"
    elif mutation == "prefix":
        payload["state_stream"]["prefix_sha256"] = "0" * 64
    elif mutation == "missingprefix":
        (root / "states.jsonl").unlink()
    elif mutation == "pending":
        payload["state_stream"]["pending_sha256"] = "0" * 64
    elif mutation == "replay":
        (root / "replay/events.jsonl").write_bytes(b"changed\n")
    write_checkpoint(path, payload)
    with pytest.raises(provenance.ProvenanceError, match=match):
        corpus.validate_reference_inputs(REFERENCE, sources, selected, tmp_path / "validation")


def test_cli_is_real_standalone_and_missing_campaign_fails(tmp_path):
    script = ROOT / "scripts/server/freeze_rotating_slots_performance_inputs.py"
    process = subprocess.run([sys.executable, "-B", str(script), "--campaign", str(tmp_path / "missing"),
        "--reference", str(REFERENCE), "--output", str(tmp_path / "output")], capture_output=True, text=True, timeout=60)
    assert process.returncode == 2
    assert json.loads(process.stdout)["status"] == "INCOMPLETE"
    assert not (tmp_path / "output/corpus_manifest.json").exists()


def test_real_style_unpinned_periodic_cut_is_accepted(campaign, reference_data):
    root = campaign / "rate60_seed23"
    index = corpus._read(root / "checkpoints/checkpoint_index.json")
    source = root / "checkpoints/checkpoint-00000000.json"
    target = root / "checkpoints/checkpoint-00000001.json"
    target.write_bytes(source.read_bytes())
    sidecar = corpus._read(source.with_suffix(".runner.json"))
    sidecar["cut"] = {"actual_sim_time": 630., "requested_sim_times": [630.],
        "simulation_deferral_s": [0.], "reasons": ["simulation_interval"]}
    write(target.with_suffix(".runner.json"), sidecar)
    index["entries"].append({"sequence": 1, "file": target.name, "sha256": corpus._sha(target),
                             "pinned": False, "corpus": True, "milestone": None})
    write(root / "checkpoints/checkpoint_index.json", index)
    result = corpus._read(root / "result.json")
    result["cuts"].append(sidecar["cut"]); result["checkpoints"].append(str(target))
    write(root / "result.json", result)
    observed = corpus.read_source(campaign, "rate60_seed23", *reference_data)
    assert len(observed["cuts"]) == 2 and observed["cuts"][-1]["pinned"] is False


def test_large_request_is_written_to_evidence_file_not_command(real_fixture, tmp_path, monkeypatch):
    _, sources, selected = real_fixture
    sources["rate60_seed23"]["large_evidence"] = "x" * 500_000
    observed = {}
    def fake_reference(reference, **kwargs):
        observed.update(kwargs)
        return {"query_result": {"replays": {}, "checkpoints": {}}, "imported_module_paths": {"fixture": "path"}}
    monkeypatch.setattr(provenance, "run_reference", fake_reference)
    result = corpus.validate_reference_inputs(REFERENCE, sources, selected, tmp_path / "validation")
    assert len(observed["code"]) < 10_000
    assert Path(result["request_path"]).stat().st_size > 500_000
    assert result["reference_process"]["imported_module_paths"] == {"fixture": "path"}
