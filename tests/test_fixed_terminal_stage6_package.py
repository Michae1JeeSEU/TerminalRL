from hashlib import sha256
import json
from pathlib import Path
import zipfile
import pytest

from terminal_marl.scripts import package_fixed_terminal_stage6 as delivery
from terminal_marl.scripts import summarize_fixed_terminal_stage6 as summary


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    from terminal_marl.scripts.evaluate_fixed_terminal_stage5 import frozen_models
    from terminal_marl.core.fixed_terminal.learning_features import FEATURE_NAMES
    from terminal_marl.configs.fixed_terminal_stage5 import MODEL_FILES
    repo, results, stage5, output = [tmp_path/n for n in ("repo", "results", "stage5", "delivery")]
    commit = "a"*40
    monkeypatch.setattr(delivery, "local_head", lambda _: commit)
    for name in delivery.DOC_FILES+delivery.SOURCE_FILES:
        write(repo/name, {})
    write(repo/"docs/evidence/fixed_terminal_stage6/run_status.md", {})
    for name in MODEL_FILES.values():
        write(stage5/"models"/name, {"fixture": "hash-only; never loaded"})
    write(stage5/"models/manifest.json", dict(schema="fixed-terminal-stage5-model-manifest-v1", status="completed", frozen=True,
        roundtrip_predictions_equal=True, test_read=False, smoke_only=False,
        producer="terminal_marl.core.fixed_terminal.validation_models", feature_names=list(FEATURE_NAMES),
        files={n:sha256((stage5/"models"/n).read_bytes()).hexdigest() for n in MODEL_FILES.values()}))
    frozen = frozen_models(stage5/"models")
    state = dict(frozen_models_before=frozen, frozen_models_after=frozen)
    record = dict(schema="fixed-terminal-stage6-summary-v1", status="completed", allow_incomplete=False,
                  counts=summary.SUMMARY_COUNTS, execution={"episodes": state, "panels": state})
    write(results/"summary/summary.json", record)
    write(results/"raw/example.json", {})
    write(stage5/"old_raw/must_not_bundle.json", {})
    # Isolate ZIP mechanics; strict raw matrix rejection is tested separately below.
    monkeypatch.setattr(summary, "summarize", lambda _: record)
    return repo, results, stage5, output, commit


def test_tiny_build_entry_hashes_crc_scope_and_no_overwrite(tiny):
    assert delivery.package(*tiny)["status"] == "ready" and not tiny[3].exists()
    result = delivery.package(*tiny, build=True)
    target = Path(result["zip"])
    assert result["crc_valid"] and result["entry_hashes_valid"]
    assert result["sha256"] == sha256(target.read_bytes()).hexdigest()
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read("MANIFEST.json"))
        for name, item in manifest["files"].items():
            content = archive.read(name)
            assert sha256(content).hexdigest() == item["sha256"] and len(content) == item["bytes"]
        assert not any("old_raw" in name for name in archive.namelist())
        assert "inputs/stage5/models/manifest.json" in archive.namelist()
        assert "results/stage6/raw/example.json" in archive.namelist()
    original = target.read_bytes()
    with pytest.raises(FileExistsError):
        delivery.package(*tiny, build=True)
    assert target.read_bytes() == original


@pytest.mark.parametrize("kind", ["summary", "head", "hash"])
def test_rejects_summary_head_or_model_mutation(tiny, monkeypatch, kind):
    if kind == "summary":
        write(tiny[1]/"summary/summary.json", {"status": "fixture_or_pilot_incomplete"})
    elif kind == "head":
        monkeypatch.setattr(delivery, "local_head", lambda _: "b"*40)
    else:
        (tiny[2]/"models/linear_rank.pt").write_bytes(b"changed")
    assert delivery.package(*tiny)["status"] == "not_ready"
    with pytest.raises(ValueError):
        delivery.package(*tiny, build=True)
    assert not tiny[3].exists()


def test_real_strict_validator_refuses_incomplete_raw(tmp_path, monkeypatch):
    commit = "a"*40
    monkeypatch.setattr(delivery, "local_head", lambda _: commit)
    args = (tmp_path/"repo", tmp_path/"results", tmp_path/"stage5", tmp_path/"output", commit)
    result = delivery.package(*args)
    assert result["status"] == "not_ready" and "incomplete stage6 evidence" in result["reason"]
    with pytest.raises(summary.IncompleteEvidence):
        delivery.package(*args, build=True)
    assert not args[3].exists()


def test_manifest_auxiliary_files_survive_bundle_and_frozen_validation(tiny, monkeypatch):
    from terminal_marl.scripts.evaluate_fixed_terminal_stage5 import frozen_models
    model_root = tiny[2]/"models"
    auxiliaries = ("protocol.json", "training.json", "rank_curve.csv")
    manifest = json.loads((model_root/"manifest.json").read_text())
    for name in auxiliaries:
        write(model_root/name, {"auxiliary": name})
        manifest["files"][name] = sha256((model_root/name).read_bytes()).hexdigest()
    write(model_root/"manifest.json", manifest)
    write(model_root/"unreferenced.json", {"exclude": True})
    frozen = frozen_models(model_root)
    record = json.loads((tiny[1]/"summary/summary.json").read_text())
    record["execution"] = {phase: dict(frozen_models_before=frozen, frozen_models_after=frozen)
                           for phase in ("episodes", "panels")}
    write(tiny[1]/"summary/summary.json", record)
    monkeypatch.setattr(summary, "summarize", lambda _: record)
    result = delivery.package(*tiny, build=True)
    extracted = tiny[3]/"extracted_models"
    extracted.mkdir()
    with zipfile.ZipFile(result["zip"]) as archive:
        prefix = "inputs/stage5/models/"
        assert all(prefix+name in archive.namelist() for name in auxiliaries)
        assert prefix+"unreferenced.json" not in archive.namelist()
        for name in ("manifest.json", *manifest["files"]):
            (extracted/name).write_bytes(archive.read(prefix+name))
    restored = frozen_models(extracted)
    assert restored["manifest"] == frozen["manifest"]
    assert restored["manifest_sha256"] == frozen["manifest_sha256"]
