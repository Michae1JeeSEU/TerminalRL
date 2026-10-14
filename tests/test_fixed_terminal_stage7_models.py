import json
from pathlib import Path
import pytest
from terminal_marl.core.fixed_terminal.search_models import train_model, FrozenSearchPolicy, MODEL_FILE
from terminal_marl.core.fixed_terminal.search_features import FEATURE_DIM


def context(labels=(0, 1, 0, 1)):
    rows = [[float(y)] + [0.]*(FEATURE_DIM-1) for y in labels]
    return {"metadata": {"split": "pilot", "seed": 7001, "load": "low", "soc": "high", "scenario_group": [7001, "low", "high"], "root_time": 0.},
            "path": "synthetic-pilot", "context_hash": "fixture", "features": rows, "targets": list(labels)}


def test_single_fit_frozen_boost_roundtrip_and_hash_guard(tmp_path):
    c = context()
    manifest = train_model([c], [c], tmp_path/"model", smoke=True)
    policy = FrozenSearchPolicy.from_directory(tmp_path/"model")
    assert manifest["fit_count"] == 1 and manifest["roundtrip_predictions_equal"]
    assert manifest["model_class"] == "GradientBoostingClassifier"
    assert policy.predict_expand(c["features"])[1] > .5
    assert policy.predict_expand([]) == []
    assert manifest["metrics"]["train"]["positive"] == 2
    with pytest.raises(FileExistsError):
        train_model([c], [c], tmp_path/"model", smoke=True)
    with pytest.raises(ValueError, match="feature"):
        policy.predict_expand([[0.]])
    with (tmp_path/"model"/MODEL_FILE).open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        FrozenSearchPolicy.from_directory(tmp_path/"model")


def test_one_class_reports_dummy_and_test_never_accepted(tmp_path):
    c = context((0, 0))
    manifest = train_model([c], [c], tmp_path/"dummy", smoke=True)
    assert manifest["model_class"] == "DummyClassifier"
    assert FrozenSearchPolicy.from_directory(tmp_path/"dummy").predict_expand(c["features"]) == [0., 0.]
    c["metadata"]["split"] = "test"
    with pytest.raises(ValueError, match="split"):
        train_model([c], [c], tmp_path/"bad", smoke=True)


def test_manifest_schema_and_formal_coverage_guard(tmp_path):
    c = context()
    c["metadata"]["split"] = "train"
    with pytest.raises(ValueError, match="inventory"):
        train_model([c], [context()], tmp_path/"bad")
    c = context()
    train_model([c], [c], tmp_path/"model", smoke=True)
    path = tmp_path/"model"/"manifest.json"
    manifest = json.loads(path.read_text())
    manifest["feature_names"][0] = "wrong"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="schema"):
        FrozenSearchPolicy.from_directory(path.parent)
