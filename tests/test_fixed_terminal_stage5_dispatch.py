"""Only tiny subprocesses: no traffic simulation or model training."""
import json
from pathlib import Path
import sys
import pytest

from terminal_marl.scripts import run_fixed_terminal_stage5 as dispatch


def tiny_jobs(phase, models, stage4_root, output, codes=(0, 0)):
    jobs = []
    code = ("import os,sys,json,time;from pathlib import Path;"
            "p=Path(sys.argv[1]);p.mkdir(parents=True);"
            "print(json.dumps({k:os.environ.get(k) for k in "
            "['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','TERMINAL_SOURCE_COMMIT']}),flush=True);"
            "time.sleep(.05);(p/'result.json').write_text(json.dumps({'done':True}));"
            "print('fixture stderr',file=sys.stderr,flush=True);sys.exit(int(sys.argv[2]))")
    for i, exit_code in enumerate(codes):
        target = str(Path(output)/"episodes"/f"tiny-{i}")
        jobs.append({"name": f"tiny-{i}", "mode": "episode", "output": target,
                     "command": [sys.executable, "-c", code, target, str(exit_code)]})
    return jobs


def test_exact_jobs_and_default_dry_run_writes_nothing(tmp_path):
    output = tmp_path/"untouched"
    result = dispatch.run_phase("test", None, None, output)
    assert result["status"] == "dry_run" and result["tasks"] == 360
    assert not output.exists()
    jobs = result["jobs"]
    assert sum(j["mode"] == "snapshot" for j in jobs) == 30
    assert sum(j["mode"] == "episode" for j in jobs) == 330
    assert len({j["output"] for j in jobs}) == 360
    assert jobs == dispatch.build_jobs("test", None, None, output)
    continuation = dispatch.build_jobs("continuation", None, "old-stage4", output)
    assert len(continuation) == 9
    assert all("--models" not in j["command"] and "--stage4-root" in j["command"] for j in continuation)
    assert all(j["seed"] == 4303 for j in continuation)


def test_real_subprocess_counts_exit_serialization_and_no_repeat(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatch, "build_jobs", tiny_jobs)
    calls = []
    def freeze(path, smoke=False):
        calls.append((path, smoke))
        return {"manifest_sha256": "fixture-frozen", "files": {"model": "same"}}
    monkeypatch.setattr(dispatch, "frozen_models", freeze)
    monkeypatch.setenv("TERMINAL_SOURCE_COMMIT", "a"*40)
    output = tmp_path/"run"
    result = dispatch.run_phase("test", "models", None, output, workers=2, execute=True)
    assert result["exit_code"] == 0 and result["status"] == "completed"
    assert result["counts"] == {"total": 2, "pending": 0, "running": 0, "completed": 2, "failed": 0}
    assert result["peak_running"] == 2
    assert len(calls) == 2 and result["model_hashes_unchanged"]
    assert len({j["pid"] for j in result["jobs"]}) == 2
    stored = json.loads((output/"execution_test.json").read_text())
    assert stored == result
    progress = [json.loads(s) for s in (output/"progress_test.jsonl").read_text().splitlines()]
    assert len(progress) == 2 and all(p["returncode"] == 0 for p in progress)
    for job in result["jobs"]:
        assert job["started_utc"] <= job["finished_utc"] and job["duration_seconds"] > 0
        env = json.loads(Path(job["stdout"]).read_text())
        assert env["TERMINAL_SOURCE_COMMIT"] == "a"*40
        assert all(env[name] == "1" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
        assert Path(job["stderr"]).read_text().strip() == "fixture stderr"
    before = (output/"execution_test.json").read_bytes()
    with pytest.raises(FileExistsError, match="no automatic resume"):
        dispatch.run_phase("test", "models", None, output, workers=2, execute=True)
    assert (output/"execution_test.json").read_bytes() == before


def test_failure_returncode_and_cli_exit1(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatch, "build_jobs", lambda *a: tiny_jobs(*a, codes=(0, 7)))
    monkeypatch.setattr(dispatch, "frozen_models", lambda *a, **kw: {"same": True})
    output = tmp_path/"failure"
    code = dispatch.main(["--phase", "test", "--models", "fixture", "--output", str(output), "--workers", "2", "--execute"])
    assert code == 1
    result = json.loads((output/"execution_test.json").read_text())
    assert result["counts"]["failed"] == 1 and result["counts"]["completed"] == 1
    assert sorted(j["returncode"] for j in result["jobs"]) == [0, 7]
    with pytest.raises(FileExistsError):
        dispatch.run_phase("test", "fixture", None, output, execute=True)


def test_freeze_failure_prevents_test_and_postcheck_marks_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatch, "build_jobs", lambda *a: tiny_jobs(*a, codes=(0,)))
    def reject(*a, **kw):
        raise ValueError("not a frozen formal artifact")
    monkeypatch.setattr(dispatch, "frozen_models", reject)
    with pytest.raises(ValueError, match="frozen formal"):
        dispatch.run_phase("test", "fixture", None, tmp_path/"blocked", execute=True)
    assert not (tmp_path/"blocked").exists()
    versions = iter(({"hash": "before"}, {"hash": "after"}))
    monkeypatch.setattr(dispatch, "frozen_models", lambda *a, **kw: next(versions))
    result = dispatch.run_phase("test", "fixture", None, tmp_path/"changed", execute=True)
    assert result["exit_code"] == 1 and not result["model_hashes_unchanged"]
    assert result["counts"]["completed"] == 1


def test_continuation_no_new_model_dependency_and_existing_job_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatch, "build_jobs", lambda *a: tiny_jobs(*a, codes=(0,)))
    def forbidden(*a, **kw):
        raise AssertionError("continuation must not load stage5 models")
    monkeypatch.setattr(dispatch, "frozen_models", forbidden)
    result = dispatch.run_phase("continuation", None, "old-stage4", tmp_path/"continuation", execute=True)
    assert result["exit_code"] == 0 and result["frozen_models_before"] is None
    existing = tmp_path/"exists"
    (existing/"episodes/tiny-0").mkdir(parents=True)
    with pytest.raises(FileExistsError, match="job output/log"):
        dispatch.run_phase("continuation", None, "old-stage4", existing, execute=True)
    assert not (existing/"execution_continuation.json").exists()
