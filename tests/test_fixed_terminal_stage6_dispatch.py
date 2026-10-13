"""Dispatcher contract with tiny subprocesses, never formal experiments."""
import json
from pathlib import Path
import sys
import pytest

from terminal_marl.scripts import run_fixed_terminal_stage6 as dispatch


def tiny_jobs(phase, models, output, codes=(0, 0)):
    code = ("import os,sys,json,time;from pathlib import Path;"
            "p=Path(sys.argv[1]);p.mkdir(parents=True);"
            "print(json.dumps({k:os.environ.get(k) for k in "
            "['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','TERMINAL_SOURCE_COMMIT']}),flush=True);"
            "time.sleep(.05);(p/'result.json').write_text(json.dumps({'done':True}));"
            "print('fixture stderr',file=sys.stderr,flush=True);sys.exit(int(sys.argv[2]))")
    return [{"name": f"tiny-{i}", "mode": "episode", "output": str(Path(output) / "episodes" / f"tiny-{i}"),
             "command": [sys.executable, "-c", code, str(Path(output) / "episodes" / f"tiny-{i}"), str(exit_code)]}
            for i, exit_code in enumerate(codes)]


def test_exact_264_15_jobs_30_states_240_rows_and_dryrun_zero_write(tmp_path):
    target = tmp_path / "untouched"
    episodes = dispatch.run_phase("episodes", None, target)
    panels = dispatch.run_phase("panels", None, target)
    assert episodes["status"] == panels["status"] == "dry_run" and not target.exists()
    assert episodes["tasks"] == 264 and panels["tasks"] == 15
    assert panels["tasks"] * 2 == 30 and panels["tasks"] * 2 * 8 == 240
    assert episodes["workers"] == 20 and panels["workers"] == 1
    assert len({j["name"] for j in episodes["jobs"]}) == len({j["output"] for j in episodes["jobs"]}) == 264
    assert len({j["output"] for j in panels["jobs"]}) == 15
    assert len({(j["load"], j["soc"], j["seed"], j["variant"]) for j in episodes["jobs"]}) == 33
    assert all(j["seed"] in (6301, 6302, 6303) and "--smoke" not in j["command"] for j in episodes["jobs"] + panels["jobs"])
    assert sum(j["variant"] != "standard" for j in panels["jobs"]) == 6
    assert episodes["jobs"] == dispatch.build_jobs("episodes", None, target)
    with pytest.raises(ValueError, match="one worker"):
        dispatch.run_phase("panels", None, target, workers=2)
    assert not target.exists()


def test_real_subprocess_counts_threads_hashes_and_no_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatch, "build_jobs", tiny_jobs)
    calls = []
    def freeze(path, smoke=False):
        calls.append((path, smoke))
        return {"manifest_sha256": "fixture", "files": {"all_four": "same"}}
    monkeypatch.setattr(dispatch, "frozen_models", freeze)
    monkeypatch.setenv("TERMINAL_SOURCE_COMMIT", "a" * 40)
    output = tmp_path / "run"
    result = dispatch.run_phase("episodes", "models", output, workers=2, execute=True)
    assert result["status"] == "completed" and result["exit_code"] == 0
    assert result["counts"] == {"total": 2, "pending": 0, "running": 0, "completed": 2, "failed": 0}
    assert result["peak_running"] == 2 and len(calls) == 2
    assert result["frozen_models_before"] == result["frozen_models_after"] and result["model_hashes_unchanged"]
    assert json.loads((output / "execution_episodes.json").read_text()) == result
    assert len((output / "progress_episodes.jsonl").read_text().splitlines()) == 2
    for job in result["jobs"]:
        observed = json.loads(Path(job["stdout"]).read_text())
        assert observed["TERMINAL_SOURCE_COMMIT"] == "a" * 40
        assert all(observed[k] == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
        assert Path(job["stderr"]).read_text().strip() == "fixture stderr"
        assert job["pid"] and job["returncode"] == 0 and job["duration_seconds"] > 0
    before = (output / "execution_episodes.json").read_bytes()
    with pytest.raises(FileExistsError, match="no automatic resume"):
        dispatch.run_phase("episodes", "models", output, execute=True)
    assert (output / "execution_episodes.json").read_bytes() == before


def test_failure_returncode_cli_and_frozen_postcheck(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatch, "build_jobs", lambda *args: tiny_jobs(*args, codes=(0, 7)))
    monkeypatch.setattr(dispatch, "frozen_models", lambda *a, **kw: {"hash": "same"})
    output = tmp_path / "failure"
    assert dispatch.main(["--phase", "episodes", "--models", "fixture", "--output", str(output), "--workers", "2", "--execute"]) == 1
    state = json.loads((output / "execution_episodes.json").read_text())
    assert state["counts"]["failed"] == 1 and sorted(j["returncode"] for j in state["jobs"]) == [0, 7]
    monkeypatch.setattr(dispatch, "build_jobs", lambda *args: tiny_jobs(*args, codes=(0,)))
    versions = iter(({"hash": "before"}, {"hash": "after"}))
    monkeypatch.setattr(dispatch, "frozen_models", lambda *a, **kw: next(versions))
    state = dispatch.run_phase("episodes", "models", tmp_path / "changed", execute=True)
    assert state["exit_code"] == 1 and not state["model_hashes_unchanged"]
    assert state["counts"]["completed"] == 1 and state["frozen_models_after"] == {"hash": "after"}


def test_missing_models_existing_job_and_launch_error(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatch, "build_jobs", lambda *args: tiny_jobs(*args, codes=(0,)))
    with pytest.raises(ValueError, match="required"):
        dispatch.run_phase("episodes", None, tmp_path / "missing", execute=True)
    assert not (tmp_path / "missing").exists()
    monkeypatch.setattr(dispatch, "frozen_models", lambda *a, **kw: {"hash": "same"})
    existing = tmp_path / "existing"
    (existing / "episodes/tiny-0").mkdir(parents=True)
    with pytest.raises(FileExistsError, match="job output/log"):
        dispatch.run_phase("episodes", "models", existing, execute=True)
    assert not (existing / "execution_episodes.json").exists()
    def reject(*a, **kw):
        raise OSError("fixture launch failure")
    monkeypatch.setattr(dispatch.subprocess, "Popen", reject)
    result = dispatch.run_phase("panels", "models", tmp_path / "launch", execute=True)
    assert result["counts"] == {"total": 1, "pending": 0, "running": 0, "completed": 0, "failed": 1}
    assert result["exit_code"] == 1 and "fixture launch failure" in result["jobs"][0]["error"]
