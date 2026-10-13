"""Small deterministic fixtures exercise training, grouping and TOTAL budgets."""
from copy import deepcopy
import json
import numpy as np
import pytest
import torch

from terminal_marl.configs.fixed_terminal_stage5 import (
    scenario_jobs, ONLINE_ARMS, MODEL_FILES, MODEL_NAMES, SNAPSHOT_TIMES, TRAINING)
from terminal_marl.core.fixed_terminal.learning_features import FEATURE_NAMES
from terminal_marl.core.fixed_terminal.validation_models import (
    total_budget_indices, ordinal_targets, validation_metrics, selection_key,
    train_validation_models, train_models, validate_groups, ValidationScorer, file_hash)


def context(offset=0):
    levels = [0, 0, 1, 2, 3, 4, 5, 6]
    roots = [{"root_id": f"r{i}", "plans": [] if i == 0 else
              [{"aircraft_id": f"uav-{i}", "wait_cycles": [i % 3]}]} for i in range(8)]
    values = [{"J": [0, level, 0, 0, -float(i), -2.*i, -.01*i]} for i, level in enumerate(levels)]
    # Two genuinely tied keys, despite distinct roots/features.
    values[1] = deepcopy(values[0])
    return {"metadata": {"split": "pilot", "seed": 5001, "load": "low", "soc": "medium",
                         "root_time": offset, "status": "completed"},
            "path": f"fixture-{offset}", "roots": roots,
            "pool_metadata": {"anchors": ["r0", "r1"], "beam_order": [r["root_id"] for r in roots]},
            "features": [[float((i+j) % 11)+offset for j in range(93)] for i in range(8)],
            "values": values, "targets": [v["J"] for v in values]}


def write_context(base, c):
    base.mkdir(parents=True)
    payloads = {"context.json": {"schema": "fixed-terminal-stage5-context-v1", **c["metadata"]},
                "features.json": {"feature_names": list(FEATURE_NAMES), "rows": [
                    {"root_id": r["root_id"], "values": x} for r, x in zip(c["roots"], c["features"])]},
                "labels.json": {"rows": [{"root_id": r["root_id"], "value": v, "delta_vs_HOLD": {"delta_J": target}}
                                          for r, v, target in zip(c["roots"], c["values"], c["targets"])]},
                "roots.json": {"roots": c["roots"], "pool_metadata": c["pool_metadata"]}}
    for name, payload in payloads.items():
        (base/name).write_text(json.dumps(payload), encoding="utf-8")


def test_frozen_domain_and_total_budget_anchors():
    assert len(scenario_jobs("test")) == 30
    assert len(scenario_jobs("pilot")) == 10
    assert len(ONLINE_ARMS) == 11 and len({r["name"] for r in ONLINE_ARMS}) == 11
    assert SNAPSHOT_TIMES == (0., 15., 90.) and TRAINING["max_epochs"] == 300
    with pytest.raises(ValueError):
        scenario_jobs("train")
    c = context()
    assert total_budget_indices(c, list(reversed(range(8))), 5) == [0, 1, 7, 6, 5]
    c["pool_metadata"]["anchors"] = ["r0", "r0"]
    assert total_budget_indices(c, list(reversed(range(8))), 5) == [0, 7, 6, 5, 4]
    assert len(total_budget_indices(c, list(range(8)), 8)) == 8
    with pytest.raises(ValueError, match="permutation"):
        total_budget_indices(c, [0, 1], 5)
    c["pool_metadata"]["anchors"] = ["r0", "r1"]
    with pytest.raises(ValueError, match="mandatory anchors"):
        total_budget_indices(c, list(range(8)), 1)


def test_ordinal_ties_and_validation_total_retention():
    c = context()
    labels = ordinal_targets(c)
    assert labels[0] == labels[1] == 0 and labels[-1] == 1
    assert labels[2] == pytest.approx(1/6)
    equal = deepcopy(c); equal["values"] = [c["values"][0]]*8
    assert ordinal_targets(equal) == [0.]*8
    class Scores:
        kind = "rank"
        def predict(self, rows):
            return [[float(i)] for i in range(len(rows))]
    measured = validation_metrics([c, equal], Scores())
    assert measured == {"informative_contexts": 1, "best_key_retained": 1,
                        "retention_at_total_budget": 1., "total_budget": 5}
    # Both anchors count toward B, rather than topB plus anchors.
    c["pool_metadata"]["anchors"] = ["r0", "r7"]
    class Reverse(Scores):
        def predict(self, rows):
            return [[-float(i)] for i in range(len(rows))]
    assert validation_metrics([c], Reverse())["best_key_retained"] == 1
    assert selection_key(.9, 100) > selection_key(.8, 0)
    assert selection_key(.9, .2) > selection_key(.9, .3)
    assert selection_key(.9, .2) == selection_key(.9, .2)


@pytest.mark.parametrize("split,seed", [("test", 5301), ("train", 4101), ("pilot", 4001)])
def test_smoke_rejects_other_splits_and_seeds(split, seed):
    c = context(); c["metadata"].update(split=split, seed=seed)
    with pytest.raises(ValueError, match="pilot5001"):
        validate_groups([c], [context()], smoke=True)


def test_formal_rejects_test_and_cli_hyperparameter_override():
    a, b = context(), context()
    a["metadata"].update(split="train", seed=4101)
    b["metadata"].update(split="val", seed=4201)
    validate_groups([a], [b])
    b["metadata"].update(split="test", seed=5301)
    with pytest.raises(ValueError, match="test never"):
        validate_groups([a], [b])
    from terminal_marl.scripts.train_fixed_terminal_stage5 import main
    with pytest.raises(SystemExit):
        main(["--stage4-root", "unused", "--output", "unused", "--epochs", "2"])


def test_four_real_smoke_models_train_save_load_without_test(tmp_path):
    train, val = context(), context(2)
    for split, c in (("train", train), ("val", val)):
        write_context(tmp_path/"data"/split/"tiny", c)
    # A malformed held-out directory must not even be parsed by training.
    (tmp_path/"data/test").mkdir()
    (tmp_path/"data/test/context.json").write_text("not-json")
    result = train_validation_models(tmp_path, tmp_path/"models", smoke=True)
    assert result["status"] == "completed" and set(result["models"]) == set(MODEL_NAMES)
    output = tmp_path/"models"
    protocol = json.loads((output/"protocol.json").read_text())
    manifest = json.loads((output/"manifest.json").read_text())
    assert protocol["epochs"] == protocol["trees"] == 3
    assert protocol["test_read"] is False and protocol["smoke_only"] is True
    assert np.array_equal(protocol["inputs"]["mean"], np.mean(train["features"], axis=0))
    assert manifest["frozen"] and manifest["roundtrip_predictions_equal"] and manifest["smoke_only"]
    for name in MODEL_NAMES:
        model = ValidationScorer.load(output/MODEL_FILES[name])
        prediction = model.predict(val["features"])
        assert prediction == ValidationScorer.load(output/MODEL_FILES[name]).predict(val["features"])
        assert len(prediction) == 8 and len(prediction[0]) == (7 if name == "value" else 1)
        assert np.isfinite(prediction).all()
        assert manifest["files"][MODEL_FILES[name]] == file_hash(output/MODEL_FILES[name])
        selected = result["models"][name]
        assert 1 <= selected["iteration"] <= 3
        assert validation_metrics([val], model)["retention_at_total_budget"] == selected["retention_at_total_budget"]
        curve = [json.loads(line) for line in (output/f"{name}_curve.jsonl").read_text().splitlines()]
        assert len(curve) == 3
        expected = max(curve, key=lambda r: selection_key(r["retention_at_total_budget"], r["val_loss"]))
        assert selected["iteration"] == expected["iteration"]
        if name == "gradient_boost":
            assert model.model.n_estimators_ == selected["iteration"]
        else:
            assert next(model.model.parameters()).device.type == "cpu"
    assert torch.get_num_threads() == 1
    with pytest.raises(FileExistsError):
        train_validation_models(tmp_path, output, smoke=True)
    # joblib is accepted only with its recorded local manifest and matching bytes.
    manifest["files"][MODEL_FILES["gradient_boost"]] = "0"*64
    (output/"manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest/hash"):
        ValidationScorer.load(output/MODEL_FILES["gradient_boost"])
    manifest["files"][MODEL_FILES["gradient_boost"]] = file_hash(output/MODEL_FILES["gradient_boost"])
    (output/"manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_missing_formal_input_does_not_create_models(tmp_path):
    with pytest.raises(ValueError, match="no completed contexts"):
        train_validation_models(tmp_path, tmp_path/"models")
    assert not (tmp_path/"models").exists()
