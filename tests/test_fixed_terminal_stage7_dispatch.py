"""Bounded subprocess dispatcher contracts, no formal experiments."""
import json
from pathlib import Path
import sys
import pytest
from terminal_marl.scripts import run_fixed_terminal_stage7 as dispatch


@pytest.fixture(autouse=True)
def source_sha(monkeypatch):
    monkeypatch.setenv("TERMINAL_SOURCE_COMMIT", "a" * 40)


def tiny_jobs(phase, models, output, source_stage4=None, codes=(0, 0), produce=True):
    code = ("import os,sys,json;from pathlib import Path;"
            "p=Path(sys.argv[1]);p.mkdir(parents=True);"
            "print(json.dumps({'cwd':os.getcwd(),'threads':os.environ['OMP_NUM_THREADS']}),flush=True);"
            "(p/'result.json').write_text('{}') if sys.argv[3]=='yes' else None;sys.exit(int(sys.argv[2]))")
    return [{"name": f"tiny-{i}", "mode": "episode", "output": str(Path(output) / "episodes" / f"tiny-{i}"),
             "expected_artifacts": [str(Path(output) / "episodes" / f"tiny-{i}" / "result.json")],
             "command": [sys.executable, "-c", code, str(Path(output) / "episodes" / f"tiny-{i}"), str(exit_code), "yes" if produce else "no"]}
            for i, exit_code in enumerate(codes)]


def test_exact_job_matrices_and_dryrun(tmp_path):
    data = dispatch.run_phase("data", None, tmp_path / "none", source_stage4="old")
    episodes = dispatch.run_phase("episodes", None, tmp_path / "none")
    panels = dispatch.run_phase("panels", None, tmp_path / "none")
    assert (data["tasks"], episodes["tasks"], panels["tasks"]) == (36, 264, 15)
    assert sum(j["split"] == "train" for j in data["jobs"]) * 4 == 108
    assert sum(j["split"] == "val" for j in data["jobs"]) * 4 == 36
    assert all("--source-stage4" in j["command"] for j in data["jobs"])
    assert all(j["seed"] in (7301, 7302, 7303) for j in episodes["jobs"] + panels["jobs"])
    assert len({j["name"] for j in episodes["jobs"]}) == 264
    assert not (tmp_path / "none").exists()
    with pytest.raises(ValueError, match="one worker"):
        dispatch.run_phase("panels", None, tmp_path / "none", workers=2)


def test_real_subprocess_cwd_threads_exit_artifacts_no_repeat(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatch, "build_jobs", tiny_jobs)
    monkeypatch.setattr(dispatch, "frozen_models", lambda *a, **kw: {"hash": "same"})
    output = tmp_path / "run"
    result = dispatch.run_phase("episodes", "models", output, workers=2, execute=True)
    assert result["status"] == "completed" and result["exit_code"] == 0
    assert result["counts"]["completed"] == 2 and result["model_hashes_unchanged"]
    for job in result["jobs"]:
        observed = json.loads(Path(job["stdout"]).read_text())
        assert Path(observed["cwd"]) == Path(dispatch.__file__).resolve().parents[2]
        assert observed["threads"] == "1" and job["returncode"] == 0 and job["pid"]
    assert not (output / ".phase.lock").exists()
    with pytest.raises(FileExistsError, match="no automatic resume"):
        dispatch.run_phase("episodes", "models", output, execute=True)


def test_missing_expected_artifact_exit_failure_and_phase_overlap(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatch, "build_jobs", lambda *a: tiny_jobs(*a, codes=(0,), produce=False))
    monkeypatch.setattr(dispatch, "frozen_models", lambda *a, **kw: {"hash": "same"})
    result = dispatch.run_phase("episodes", "models", tmp_path / "missing", execute=True)
    assert result["exit_code"] == 1 and result["jobs"][0]["returncode"] == 0
    assert "expected artifacts missing" in result["jobs"][0]["error"]
    output = tmp_path / "overlap"
    output.mkdir()
    (output / ".phase.lock").write_text('{"phase":"data"}')
    with pytest.raises(FileExistsError):
        dispatch.run_phase("panels", "models", output, execute=True)
    assert not (output / "execution_panels.json").exists()
    (output / ".phase.lock").unlink()
    (output / "execution_data.json").write_text('{"status":"running"}')
    with pytest.raises(ValueError, match="another phase"):
        dispatch.run_phase("panels", "models", output, execute=True)


def test_required_data_source_and_nonzero_exit(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="source"):
        dispatch.run_phase("data", None, tmp_path / "none", execute=True)
    monkeypatch.setattr(dispatch, "build_jobs", lambda *a: tiny_jobs(*a, codes=(7,)))
    monkeypatch.setattr(dispatch, "frozen_models", lambda *a, **kw: {"hash": "same"})
    result = dispatch.run_phase("episodes", "models", tmp_path / "failure", execute=True)
    assert result["exit_code"] == 1 and result["jobs"][0]["returncode"] == 7
