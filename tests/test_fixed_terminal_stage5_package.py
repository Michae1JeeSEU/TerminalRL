from hashlib import sha256
import json
from pathlib import Path
import zipfile

import pytest

from terminal_marl.scripts import package_fixed_terminal_stage5 as delivery


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    from terminal_marl.scripts import evaluate_fixed_terminal_stage5 as evaluation
    from terminal_marl.core.fixed_terminal.learning_features import FEATURE_NAMES
    from terminal_marl.configs.fixed_terminal_stage5 import MODEL_FILES
    repository, results, stage4, output = [tmp_path / n for n in ("repo", "results", "stage4", "delivery")]
    commit = "a" * 40
    monkeypatch.setattr(delivery, "local_head", lambda _: commit)
    for name in delivery.DOC_FILES + delivery.SOURCE_FILES:
        write(repository / name, {"tiny": True})
    write(repository / "docs/evidence/fixed_terminal_stage5/results.md", {})
    write(repository / "tmp_stage5_preflight/stage4_diagnostics.json", {})
    write(repository / "tmp_stage5_preflight/validation.json", {})
    write(repository / "tmp_stage5_preflight/models/large_smoke.json", {})
    for split in ("train", "val"):
        write(stage4 / "data" / split / "tiny/context.json", {})
    write(stage4 / "models/protocol.json", {})
    write(stage4 / "summary/offline_test.json", {})
    for load in ("low", "medium", "high"):
        for soc in ("high", "medium", "low"):
            for name in ("frozen_input.json", "generation.json", "t15/context.json"):
                write(stage4 / "data/test" / f"{load}-{soc}-4303" / name, {})
    for name in MODEL_FILES.values():
        write(results / "models" / name, {"tiny": "hash-only checkpoint fixture; never loaded"})
    manifest = dict(schema="fixed-terminal-stage5-model-manifest-v1", status="completed", frozen=True,
                    roundtrip_predictions_equal=True, test_read=False, smoke_only=False,
                    producer="terminal_marl.core.fixed_terminal.validation_models",
                    feature_names=list(FEATURE_NAMES),
                    files={n: sha256((results / "models" / n).read_bytes()).hexdigest() for n in MODEL_FILES.values()})
    write(results / "models/manifest.json", manifest)
    frozen = evaluation.frozen_models(results / "models")
    for phase, count in (("test", 360), ("continuation", 9)):
        write(results / f"execution_{phase}.json", dict(status="completed", exit_code=0,
            counts=dict(total=count, pending=0, running=0, completed=count, failed=0),
            jobs=[dict(name=f"job-{i}", status="completed", returncode=0) for i in range(count)],
            model_hashes_unchanged=True, frozen_models_before=frozen, frozen_models_after=frozen))
    write(results / "summary/summary.json", dict(schema="fixed-terminal-stage5-summary-v1",
          status="completed", allow_incomplete=False, counts=delivery.SUMMARY_COUNTS))
    write(results / "pilot/validation.json", {})
    write(results / "__pycache__/skip.pyc", {})
    return repository, results, stage4, output, commit


def test_tiny_build_crc_hashes_no_overwrite(tiny):
    dry = delivery.package(*tiny)
    assert dry["status"] == "ready" and not tiny[3].exists()
    result = delivery.package(*tiny, build=True)
    assert result["status"] == "completed" and result["crc_valid"] and result["entry_hashes_valid"]
    target = Path(result["zip"])
    assert result["sha256"] == sha256(target.read_bytes()).hexdigest()
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read("MANIFEST.json"))
        assert len(archive.namelist()) == len(manifest["files"]) + 1 == result["entries"]
        for name, row in manifest["files"].items():
            content = archive.read(name)
            assert sha256(content).hexdigest() == row["sha256"] and len(content) == row["bytes"]
        assert not any("__pycache__" in n or "large_smoke" in n for n in archive.namelist())
        assert str(tiny[0]).encode() not in archive.read("BUNDLE_README.md")
    original = target.read_bytes()
    with pytest.raises(FileExistsError):
        delivery.package(*tiny, build=True)
    assert target.read_bytes() == original


@pytest.mark.parametrize("kind", ["incomplete", "hash", "summary", "head"])
def test_refuses_missing_completion_and_mutation(tiny, monkeypatch, kind):
    if kind == "incomplete":
        path = tiny[1] / "execution_test.json"
        state = json.loads(path.read_text())
        state["jobs"][0]["returncode"] = 7
        write(path, state)
    elif kind == "hash":
        (tiny[1] / "models/linear_rank.pt").write_bytes(b"changed")
    elif kind == "summary":
        path = tiny[1] / "summary/summary.json"
        summary = json.loads(path.read_text())
        summary["status"] = "fixture_or_pilot_incomplete"
        write(path, summary)
    else:
        monkeypatch.setattr(delivery, "local_head", lambda _: "b" * 40)
    assert delivery.package(*tiny)["status"] == "not_ready"
    with pytest.raises(ValueError):
        delivery.package(*tiny, build=True)
    assert not tiny[3].exists()


def test_symlink_escape_refused(tiny, monkeypatch):
    # Portable Windows check without requiring symlink-creation privileges.
    suspect = tiny[1] / "pilot/validation.json"
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda p: p == suspect or original(p))
    assert delivery.package(*tiny)["status"] == "not_ready"
    with pytest.raises(ValueError, match="symlink"):
        delivery.package(*tiny, build=True)
    assert not tiny[3].exists()
