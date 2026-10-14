"""Delivery mechanics and manifest dependencies; strict raw tested separately."""
from hashlib import sha256
import json
from pathlib import Path
import zipfile
import pytest
from terminal_marl.scripts import package_fixed_terminal_stage7 as delivery
from terminal_marl.scripts import summarize_fixed_terminal_stage7 as summary
from terminal_marl.scripts.evaluate_fixed_terminal_stage7 import frozen_models
from terminal_marl.core.fixed_terminal.search_features import FEATURE_NAMES, FEATURE_DIM, SCHEMA
from terminal_marl.configs.fixed_terminal_stage7 import TRAINING, TRAIN_SEEDS, VAL_SEEDS, TRACE_WIDTHS


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    repo, results, models, output = [tmp_path / n for n in ("repo", "results", "models", "delivery")]
    commit = "a" * 40
    monkeypatch.setattr(delivery, "local_head", lambda _: commit)
    for name in delivery.DOC_FILES + delivery.SOURCE_FILES:
        write(repo / name, {})
    write(repo / "docs/evidence/fixed_terminal_stage7/run_status.md", {})
    write(models / "gradient_boost_expand.joblib", {"fixture": "hash-only; never loaded"})
    write(models / "protocol.json", {"threshold": .5})
    write(models / "training.json", {"fit_count": 1})
    files = {name: sha256((models / name).read_bytes()).hexdigest() for name in ("gradient_boost_expand.joblib", "protocol.json", "training.json")}
    write(models / "manifest.json", dict(schema="fixed-terminal-stage7-model-manifest-v1", status="completed", frozen=True,
          roundtrip_predictions_equal=True, test_read=False, smoke_only=False, kind="expand", name="gradient_boost_expand", files=files,
          feature_schema=SCHEMA, feature_names=list(FEATURE_NAMES), feature_dim=FEATURE_DIM, training=TRAINING,
          threshold=.5, fit_count=1, trace_widths=list(TRACE_WIDTHS), metrics={"train": {"nodes": 2, "positive": 1, "negative": 1}}, model_class="GradientBoostingClassifier", splits={"train": list(TRAIN_SEEDS), "val": list(VAL_SEEDS)}))
    frozen = frozen_models(models)
    state = dict(frozen_models_before=frozen, frozen_models_after=frozen)
    record = dict(schema="fixed-terminal-stage7-summary-v1", status="completed", allow_incomplete=False,
                  counts=summary.SUMMARY_COUNTS, execution={"episodes": state, "panels": state})
    write(results / "summary/summary.json", record)
    write(results / "data/raw.json", {})
    write(models / "unreferenced.json", {"exclude": True})
    monkeypatch.setattr(summary, "summarize", lambda _: record)
    return repo, results, models, output, commit


def test_crc_entry_hashes_plan_models_aux_and_no_overwrite(tiny):
    assert delivery.package(*tiny)["status"] == "ready" and not tiny[3].exists()
    result = delivery.package(*tiny, build=True)
    target = Path(result["zip"])
    assert result["crc_valid"] and result["entry_hashes_valid"]
    assert result["sha256"] == sha256(target.read_bytes()).hexdigest()
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read("MANIFEST.json"))
        assert manifest["final_commit"] == tiny[4]
        for name, item in manifest["files"].items():
            content = archive.read(name)
            assert sha256(content).hexdigest() == item["sha256"] and len(content) == item["bytes"]
        assert "docs/fixed_terminal_stage7_learning_to_search_plan.md" in archive.namelist()
        assert "inputs/stage7/models/protocol.json" in archive.namelist()
        assert "inputs/stage7/models/unreferenced.json" not in archive.namelist()
        assert "results/stage7/data/raw.json" in archive.namelist()
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        delivery.package(*tiny, build=True)
    assert target.read_bytes() == before


@pytest.mark.parametrize("kind", ("summary", "head", "hash", "aux"))
def test_summary_head_and_any_manifest_dependency_mutation_rejected(tiny, monkeypatch, kind):
    if kind == "summary":
        write(tiny[1] / "summary/summary.json", {"status": "fixture_or_pilot_incomplete"})
    elif kind == "head":
        monkeypatch.setattr(delivery, "local_head", lambda _: "b" * 40)
    else:
        (tiny[2] / ("protocol.json" if kind == "aux" else "gradient_boost_expand.joblib")).write_bytes(b"changed")
    assert delivery.package(*tiny)["status"] == "not_ready"
    with pytest.raises(ValueError):
        delivery.package(*tiny, build=True)
    assert not tiny[3].exists()


def test_strict_validator_refuses_missing_formal_raw(tmp_path, monkeypatch):
    commit = "a" * 40
    monkeypatch.setattr(delivery, "local_head", lambda _: commit)
    args = (tmp_path / "repo", tmp_path / "results", tmp_path / "models", tmp_path / "output", commit)
    result = delivery.package(*args)
    assert result["status"] == "not_ready" and "incomplete stage7 evidence" in result["reason"]
    with pytest.raises(summary.IncompleteEvidence):
        delivery.package(*args, build=True)
    assert not args[3].exists()


def test_manifest_unsafe_path_rejected(tiny):
    path = tiny[2] / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"]["../escape.json"] = "a" * 64
    write(path, manifest)
    with pytest.raises(ValueError, match="unsafe"):
        frozen_models(tiny[2])


def test_single_class_diagnostic_classifier_requires_class_counts(tiny):
    path = tiny[2] / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["model_class"] = "DummyClassifier"
    manifest["metrics"]["train"] = {"nodes": 2, "positive": 0, "negative": 2}
    write(path, manifest)
    assert frozen_models(tiny[2])["manifest"]["model_class"] == "DummyClassifier"
    manifest["metrics"]["train"] = {"nodes": 2, "positive": 1, "negative": 1}
    write(path, manifest)
    with pytest.raises(ValueError, match="classifier differs"):
        frozen_models(tiny[2])
