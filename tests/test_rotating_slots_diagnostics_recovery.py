"""Recovery mechanics with tiny synthetic archives; no formal seed is run here."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from scripts.server import recover_rotating_slots_diagnostics as recovery
from terminal_marl.core.rotating_slots import diagnostics
from terminal_marl.core.rotating_slots.qualification import write_json
from terminal_marl.core.rotating_slots.replay import ReplayBundle


ORIGINAL_COMMAND = "synthetic frozen-matrix fixture; not a research result"
RECOVERY_COMMAND = "synthetic single-job recovery test"


def tiny_result(job, current):
    """Real hash-validated archive, synthetic measurements, unchanged job specs."""
    root = Path(job["output"]) / job["id"]
    root.mkdir(parents=True, exist_ok=True)
    metrics = {"conservation": True, "fixture_count": 0, "synthetic_test_fixture": True}
    manifest = {"seed": job.get("seed", 11), "requests": []}
    record = {"states": [{"time": 0., "failed": False}, {"time": .2, "failed": False}],
              "metrics": metrics, "manifest": manifest, "events": [], "trajectories": {}}
    archive = ReplayBundle.write(root / "replay", record=record, eager=False,
        code_commit=recovery.FROZEN_COMMIT, command=job["command"], seed=job.get("seed", 11))
    info = deepcopy(current)
    info["command"] = job["command"]
    row = {"id": job["id"], "job": deepcopy(job), "status": "COMPLETED", "reason": "",
        "provenance": info, "metrics": metrics, "manifest": manifest,
        "replay_path": str(archive.path), "events_path": str(archive.path / "events.jsonl"),
        "manifest_path": str(archive.path / "manifest.json"), "state_count": 2,
        "wall_seconds": 2., "cpu_seconds": 1., "peak_rss_bytes": 1024,
        "simulated_seconds": 1., "sim_seconds_per_wall_second": .5,
        "decision_times": [0., .2], "inflight_admitted": job.get("mode") == "inflight_attempt",
        "researcher_visual_acceptance": "PENDING"}
    if job["kind"] == "dt":
        row.update(events=[], energy_j=100., minimum={"lower_bound_m": 100., "upper_bound_m": 100.0001},
                   committed_script={"plans": [], "synthetic_test_fixture": True})
    write_json(root / "result.json", row)
    return row


@pytest.fixture
def original(tmp_path):
    source = tmp_path / "original"
    source.mkdir()
    info = recovery._current_provenance(ORIGINAL_COMMAND, recovery.FROZEN_COMMIT)
    jobs = [dict(spec, output=str(source), command=ORIGINAL_COMMAND) for spec in recovery.frozen_jobs()]
    write_json(source / "jobs.json", jobs)
    for job in jobs:
        if job["id"] != recovery.INTERRUPTED_ID:
            tiny_result(job, info)
    interrupted = source / recovery.INTERRUPTED_ID
    interrupted.mkdir()
    write_json(interrupted / "progress.json", {"status": "RUNNING", "sim_time": 1265., "wall_seconds": 71462.})
    (interrupted / "states.jsonl").write_bytes(b'{"time":1274.8,"failed":false}\n{"time":1275,"aircraft":[')
    return source, info


def tree_hashes(path):
    return {p.relative_to(path).as_posix(): sha256(p.read_bytes()).hexdigest()
            for p in path.rglob("*") if p.is_file()}


def change_json(path, update):
    value = json.loads(path.read_bytes())
    update(value)
    write_json(path, value)


def test_reuses_60_dispatches_only_one_and_matches_frozen_aggregation(original, tmp_path, monkeypatch):
    source, info = original
    output = tmp_path / "recovered"
    before = tree_hashes(source)
    observed = []
    old_rows = {j["id"]: json.loads((source / j["id"] / "result.json").read_bytes())
                for j in recovery.frozen_jobs() if j["id"] != recovery.INTERRUPTED_ID}

    def runner(job):
        observed.append(deepcopy(job))
        return tiny_result(job, info)

    report = recovery.recover(source, output, code_commit=recovery.FROZEN_COMMIT,
        command=RECOVERY_COMMAND, recovery_commit="a" * 40, runner=runner)
    assert len(observed) == 1 and observed[0]["id"] == recovery.INTERRUPTED_ID
    assert {k: observed[0][k] for k in recovery.frozen_jobs()[0]} == next(
        s for s in recovery.frozen_jobs() if s["id"] == recovery.INTERRUPTED_ID)
    assert observed[0]["output"] == str(output) and observed[0]["command"] == RECOVERY_COMMAND
    assert [r["id"] for r in report["jobs"]] == [s["id"] for s in recovery.frozen_jobs()]
    for row in report["jobs"]:
        if row["id"] != recovery.INTERRUPTED_ID:
            assert row == old_rows[row["id"]]
    assert tree_hashes(source) == before
    assert report["coverage"]["actual_count"] == 61
    assert all(not report["coverage"][k] for k in ("missing", "duplicates", "unexpected", "failures", "pending"))
    assert report["stage1_status"] == "NOT_ACCEPTED" and report["researcher_visual_acceptance"] == "PENDING"
    metadata = report["recovery"]
    assert metadata["checkpoint_resume"] is False and metadata["restart_from_simulated_time"] == 0
    assert metadata["wrapper"]["provided_recovery_commit"] == "a" * 40
    assert metadata["source"]["frozen_provenance"]["code_commit"] == recovery.FROZEN_COMMIT
    assert metadata["source"]["interrupted_attempt"]["states"]["bounded_tail"]["ends_with_newline"] is False
    assert metadata["source"]["interrupted_attempt"]["states"]["bounded_tail"]["last_complete_projection"] == {
        "time": 1274.8, "failed": False}
    assert metadata["timing"]["interrupted_last_progress_wall_seconds"] == 71462.
    assert metadata["timing"]["restarted_job_wall_seconds"] == 2.
    assert all(value == "1" for value in metadata["threads"].values())
    #Use the actual frozen aggregator as the reference, without executing a job.
    monkeypatch.setattr(diagnostics, "execute_jobs", lambda jobs, root, workers: (
        report["jobs"], report["coverage"]))
    reference = diagnostics.run_diagnostics(tmp_path / "aggregate_reference", command=RECOVERY_COMMAND)
    assert {k: v for k, v in report.items() if k != "recovery"} == reference
    assert json.loads((output / "diagnostics.json").read_bytes()) == report


def test_preflight_cli_is_read_only_and_does_not_dispatch(original, tmp_path, monkeypatch, capsys):
    source, _ = original
    output = tmp_path / "preflight_output"
    before = tree_hashes(source)
    monkeypatch.setattr(diagnostics, "_diagnostic_job", lambda job: pytest.fail("preflight dispatched a job"))
    code = recovery.main(["--source", str(source), "--output", str(output),
                          "--code-commit", recovery.FROZEN_COMMIT, "--workers", "1", "--preflight-only"])
    record = json.loads(capsys.readouterr().out)
    assert code == 0 and record["status"] == "PREFLIGHT_PASSED"
    assert record["evidence"]["validated_completed_count"] == 60
    assert record["wrapper"]["sha256"] == recovery._hash(Path(recovery.__file__))
    assert not output.exists() and tree_hashes(source) == before


@pytest.mark.parametrize("job_id,field,value", [
    ("traffic_rate60_seed11", "rate", 61),
    ("traffic_rate60_seed11", "seed", 12),
    ("traffic_rate60_seed11", "window", 3599.),
    ("traffic_rate60_seed11", "drain", 1799.),
    ("benchmark_n10_stable_r0", "count", 9),
    ("benchmark_n10_stable_r0", "mode", "unknown"),
    ("benchmark_n10_stable_r0", "repeat", 7),
    ("benchmark_n10_stable_r0", "duration", .2),
    ("dt_0.2", "dt", .3),
    ("scheduler_1.0", "period", 2.),
    ("traffic_rate60_seed11", "pilot", True),
])
def test_frozen_parameter_changes_fail_before_output(original, tmp_path, job_id, field, value):
    source, _ = original
    change_json(source / "jobs.json", lambda jobs: next(j for j in jobs if j["id"] == job_id).__setitem__(field, value))
    output = tmp_path / "refused"
    with pytest.raises(recovery.RecoveryError, match="frozen job parameter"):
        recovery.preflight(source, output, code_commit=recovery.FROZEN_COMMIT)
    assert not output.exists()


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unexpected"])
def test_matrix_ids_must_be_exact(original, tmp_path, mutation):
    source, _ = original
    def update(jobs):
        if mutation == "missing": jobs.pop()
        elif mutation == "duplicate": jobs[-1] = deepcopy(jobs[0])
        else: jobs[-1]["id"] = "not_in_frozen_matrix"
    change_json(source / "jobs.json", update)
    with pytest.raises(recovery.RecoveryError, match="coverage mismatch"):
        recovery.preflight(source, tmp_path / "refused", code_commit=recovery.FROZEN_COMMIT)


@pytest.mark.parametrize("field", ["code_commit", "code_sha256", "config_hash", "source_files_sha256"])
def test_result_provenance_must_match_actual_frozen_files(original, tmp_path, field):
    source, _ = original
    change_json(source / "traffic_rate60_seed11/result.json", lambda row: row["provenance"].__setitem__(field, "wrong"))
    with pytest.raises(recovery.RecoveryError, match="provenance mismatch"):
        recovery.preflight(source, tmp_path / "refused", code_commit=recovery.FROZEN_COMMIT)


@pytest.mark.parametrize("status", ["FAILED", "BLOCKED"])
def test_noncompleted_results_cannot_be_reused(original, tmp_path, status):
    source, _ = original
    change_json(source / "traffic_rate60_seed11/result.json", lambda row: row.__setitem__("status", status))
    with pytest.raises(recovery.RecoveryError, match="only real COMPLETED"):
        recovery.preflight(source, tmp_path / "refused", code_commit=recovery.FROZEN_COMMIT)


def test_missing_other_result_cannot_be_rerun(original, tmp_path):
    source, _ = original
    (source / "traffic_rate60_seed11/result.json").unlink()
    with pytest.raises(FileNotFoundError):
        recovery.preflight(source, tmp_path / "refused", code_commit=recovery.FROZEN_COMMIT)


def test_completed_last_job_and_unexpected_result_are_refused(original, tmp_path):
    source, info = original
    job = next(j for j in json.loads((source / "jobs.json").read_bytes()) if j["id"] == recovery.INTERRUPTED_ID)
    tiny_result(job, info)
    with pytest.raises(recovery.RecoveryError, match="already has a result"):
        recovery.preflight(source, tmp_path / "refused", code_commit=recovery.FROZEN_COMMIT)


@pytest.mark.parametrize("kind", ["corrupt_archive", "mismatched_metrics", "mismatched_bundle_commit"])
def test_replay_integrity_and_result_correspondence(original, tmp_path, kind):
    source, _ = original
    jobroot = source / "traffic_rate60_seed11"
    if kind == "corrupt_archive":
        (jobroot / "replay/states.jsonl").write_bytes(b'{"time":999}\n')
    elif kind == "mismatched_metrics":
        change_json(jobroot / "result.json", lambda row: row["metrics"].__setitem__("extra", 1))
    else:
        change_json(jobroot / "replay/replay.json", lambda index: index.__setitem__("code_commit", "wrong"))
    with pytest.raises(ValueError, match="hash mismatch|metrics mismatch|provenance mismatch"):
        recovery.preflight(source, tmp_path / "refused", code_commit=recovery.FROZEN_COMMIT)


def test_existing_or_nested_output_and_wrong_workers_are_refused(original, tmp_path):
    source, _ = original
    existing = tmp_path / "existing"
    existing.mkdir()
    for output, workers, error in ((existing, 1, FileExistsError),
                                    (source / "nested", 1, recovery.RecoveryError),
                                    (tmp_path / "new", 2, recovery.RecoveryError)):
        with pytest.raises(error):
            recovery.preflight(source, output, code_commit=recovery.FROZEN_COMMIT, workers=workers)


def test_current_frozen_fingerprint_mismatch_is_refused(original, tmp_path, monkeypatch):
    source, info = original
    wrong = deepcopy(info)
    wrong["code_sha256"] = "wrong"
    monkeypatch.setattr(recovery, "provenance", lambda command: wrong)
    with pytest.raises(recovery.RecoveryError, match="current frozen"):
        recovery.preflight(source, tmp_path / "refused", code_commit=recovery.FROZEN_COMMIT)


@pytest.mark.parametrize("failure", ["software", "corrupt_new_archive", "source_changed"])
def test_failed_new_job_or_validation_never_writes_formal_report(original, tmp_path, failure):
    source, info = original
    output = tmp_path / "failed_recovery"
    before = tree_hashes(source)
    def runner(job):
        row = tiny_result(job, info)
        if failure == "software":
            row.update(status="FAILED", reason="synthetic software failure")
            write_json(Path(job["output"]) / job["id"] / "result.json", row)
        elif failure == "corrupt_new_archive":
            (Path(row["replay_path"]) / "states.jsonl").write_bytes(b"corrupted\n")
        else:
            (source / recovery.INTERRUPTED_ID / "states.jsonl").write_bytes(b"external change during test\n")
        return row
    with pytest.raises(ValueError):
        recovery.recover(source, output, code_commit=recovery.FROZEN_COMMIT,
                         command=RECOVERY_COMMAND, runner=runner)
    assert not (output / "diagnostics.json").exists()
    assert (output / recovery.INTERRUPTED_ID / "result.json").is_file()
    assert json.loads((output / "recovery_manifest.json").read_bytes())["status"] == "FAILED"
    if failure != "source_changed":
        assert tree_hashes(source) == before


def test_real_kernel_small_independent_output_path_smoke(tmp_path):
    """Separate pilot fixture; never substitutes for a frozen formal job."""
    recovery._current_provenance("independent path smoke", recovery.FROZEN_COMMIT)
    job = {"id": "independent_recovery_path_smoke", "kind": "traffic", "rate": 60,
           "seed": 11, "window": .2, "drain": 0., "pilot": True,
           "output": str(tmp_path / "smoke"), "command": "pytest independent small pilot path smoke"}
    row = diagnostics._diagnostic_job(job)
    assert row["status"] == "COMPLETED" and row["simulated_seconds"] == .2
    assert Path(row["replay_path"]) == tmp_path / "smoke" / job["id"] / "replay"
    assert ReplayBundle.validate(row["replay_path"]).state_count == row["state_count"]
    assert row["provenance"]["code_sha256"] == recovery.FROZEN_CODE_SHA
    assert all(v == "1" for v in row["provenance"]["threads"].values())
