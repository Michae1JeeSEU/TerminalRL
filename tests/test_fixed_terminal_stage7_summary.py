"""Raw recomputation from real tiny stage7 episodes/panels."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import json
import shutil
import pytest
from terminal_marl.configs.fixed_terminal_stage1 import DEFAULT_CONFIG
from terminal_marl.core.fixed_terminal.scenarios import Request, Scenario
from terminal_marl.scripts import evaluate_fixed_terminal_stage7 as evaluation
from terminal_marl.scripts import summarize_fixed_terminal_stage7 as summary


def tiny_outputs(root, monkeypatch):
    scenario = Scenario(7001, "low", "high", (Request("a", 0., .8, 0, .1),), 1, 2)
    config = replace(DEFAULT_CONFIG, horizon=2.)
    policies = {"gradient_boost_expand": SimpleNamespace(name="gradient_boost_expand", kind="expand", predict_expand=lambda rows: [.75 for _ in rows])}
    monkeypatch.setattr(evaluation, "load_scorers", lambda directory, names, smoke: (policies, {}, None))
    for arm in summary.ARMS:
        evaluation.run_episode(root / "episodes" / (arm + "-tiny"), arm, smoke=True, config=config, scenario=scenario, soc="high")
    evaluation.run_panels(root / "panels" / "tiny", smoke=True, config=config, scenario=scenario, times=(0.,))


def write(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def test_real_raw_recompute_quality_and_panel_schema(tmp_path, monkeypatch):
    tiny_outputs(tmp_path, monkeypatch)
    raw_paths = list((tmp_path / "episodes").glob("*/decisions.jsonl")) + list((tmp_path / "panels").glob("*/t0/*.json"))
    raw_before = {path: path.read_bytes() for path in raw_paths}
    loaded_episodes = summary.load_episodes(tmp_path, True)[0]
    for record in loaded_episodes:
        raw = list(summary.rows_jsonl(tmp_path / "episodes" / (record["arm"] + "-tiny") / "decisions.jsonl"))
        projected = record["_decisions"]
        assert summary.adaptive_totals(raw) == summary.adaptive_totals(projected)
        assert summary.sum_numeric([r["work"] for r in raw]) == summary.sum_numeric([r["work"] for r in projected])
        for before, after in zip(raw, projected):
            assert after["search"]["counters"] == before["search"]["counters"]
            assert after["search"]["stages"] == before["search"]["stages"]
            assert after["search"]["node_count"] == len(before["search"]["nodes"])
            assert "nodes" not in after["search"] and "leaf_links" not in after["search"]
            assert not {"pool", "evaluations"} & after.keys()
    loaded_panels = summary.load_panels(tmp_path, True)[0]
    for projected in loaded_panels:
        raw = summary.read_json(tmp_path / "panels/tiny/t0" / (projected["arm"] + ".json"))
        assert projected["search"]["counters"] == raw["search"]["counters"]
        assert projected["search"]["node_count"] == len(raw["search"]["nodes"])
        assert "nodes" not in projected["search"] and "leaf_links" not in projected["search"]
    result = summary.summarize(tmp_path, True)
    def no_detailed_nodes(value):
        if isinstance(value, dict):
            assert not isinstance(value.get("nodes"), list)
            assert "leaf_links" not in value
            for child in value.values():
                no_detailed_nodes(child)
        elif isinstance(value, list):
            for child in value:
                no_detailed_nodes(child)
    no_detailed_nodes(result)
    assert all(path.read_bytes() == content for path, content in raw_before.items())
    assert result["counts"] == dict(episodes=8, generations=1, timing_panels=1, timing_rows=8)
    assert result["status"] == "fixture_or_pilot_incomplete"
    assert result["allow_incomplete"] and result["missing_formal_paths"]
    assert all(r["all_final_execution_checks_passed"] for r in result["episode_groups"])
    assert len(result["paired_online"]["rows"]) == 15
    for row in result["same_state"]["pairs"]:
        assert row["degenerate"] is False
        assert "arm_search_counters" in row and "method_pool_single_root" in row
    output = tmp_path / "summary"
    summary.write_outputs(output, result)
    assert summary.read_json(output / "summary.json")["counts"] == result["counts"]
    with pytest.raises(FileExistsError):
        summary.write_outputs(output, result)
    with pytest.raises(summary.IncompleteEvidence):
        summary.summarize(tmp_path)


def test_duplicate_and_raw_work_mutation_rejected(tmp_path, monkeypatch):
    tiny_outputs(tmp_path, monkeypatch)
    path = next((tmp_path / "episodes").glob("*/result.json"))
    saved = summary.read_json(path)
    saved["work"]["expanded_nodes"] += 1
    write(path, saved)
    with pytest.raises(ValueError, match="work aggregate"):
        summary.load_episodes(tmp_path, True)
    saved["work"]["expanded_nodes"] -= 1
    write(path, saved)
    shutil.copytree(path.parent, tmp_path / "episodes" / "duplicate")
    with pytest.raises(ValueError, match="duplicate episode"):
        summary.load_episodes(tmp_path, True)


def test_partial_links_exact_cache_and_final_validation_rejected(tmp_path, monkeypatch):
    tiny_outputs(tmp_path, monkeypatch)
    row = summary.read_json(tmp_path / "panels/tiny/t0/full_reference.json")
    summary.validate_stages(row)
    bad = deepcopy(row)
    bad["adaptive"]["stages"][0]["new_root_ids"].append(bad["adaptive"]["stages"][0]["new_root_ids"][0])
    with pytest.raises(ValueError, match="duplicate cached"):
        summary.validate_stages(bad)
    bad = deepcopy(row)
    bad["final_check"]["status"] = "unknown"
    with pytest.raises(ValueError, match="final independent"):
        summary.validate_stages(bad)
    rows = summary.load_panels(tmp_path, True)[0]
    # A narrow method singleton remains nondegenerate when the common full pool is larger.
    for r in rows:
        if r["arm"] == "learned_pruning":
            r["pool_count"] = 1
            r["degenerate"] = True
            r["work"]["root_rollout_count"] = 1
            r["timing"]["total"] = .1
        elif r["arm"] in summary.REFERENCES:
            r["work"]["root_rollout_count"] = 10
            r["timing"]["total"] = 1.
    pairs = summary.same_state_comparisons(rows)
    subset = [r for r in pairs["pairs"] if r["arm"] == "learned_pruning"]
    assert subset and all(not r["degenerate"] and r["method_pool_single_root"] for r in subset)
    groups = [r for r in pairs["groups"] if r["arm"] == "learned_pruning" and r["stratum"] == "nondegenerate" and r["scope"] == r["axis"] == "all"]
    assert groups and all(r["compute_condition_passed"] for r in groups)


def test_inventory_nonfinite_and_missing_data(tmp_path):
    paths, expected = summary.expected_paths()
    assert {k: len(v) for k, v in expected.items()} == dict(episodes=264, panels=30, generations=15)
    assert "execution_data.json" in paths and len([p for p in paths if p.startswith("data/") and p.endswith("context.json")]) == 144
    with pytest.raises(summary.IncompleteEvidence) as error:
        summary.summarize(tmp_path)
    assert "execution_data.json" in error.value.missing
    bad = tmp_path / "bad.json"
    bad.write_text('{"n":NaN}')
    with pytest.raises(ValueError, match="nonfinite"):
        summary.read_json(bad)


def test_model_single_fit_binds_exact_context_hashes(tmp_path):
    from hashlib import sha256
    from terminal_marl.core.fixed_terminal.search_features import FEATURE_NAMES, FEATURE_DIM, SCHEMA
    from terminal_marl.configs.fixed_terminal_stage7 import TRAINING, TRAIN_SEEDS, VAL_SEEDS, TRACE_WIDTHS
    models = tmp_path / "models"
    models.mkdir()
    protocol = {"fit_count": 1, "test_read": False, "smoke_only": False, "training": TRAINING, "threshold": .5,
                "trace_widths": list(TRACE_WIDTHS), "train": [{"scenario_group": [4101, "low", "high"], "time": 0., "context_sha256": "train-hash"}],
                "val": [{"scenario_group": [4201, "low", "high"], "time": 0., "context_sha256": "val-hash"}]}
    data = {"splits": {split: [{"seed": seed, "load": "low", "soc": "high", "time": 0., "context_hash": split + "-hash"}]
                       for split, seed in (("train", 4101), ("val", 4201))}}
    metrics = {"train": {"nodes": 2, "positive": 1, "negative": 1}, "val": {"nodes": 1}}
    write(models / "protocol.json", protocol)
    write(models / "training.json", {"status": "completed", "protocol": protocol, "model_class": "GradientBoostingClassifier",
          "one_class_dummy": False, "roundtrip_predictions_equal": True, "metrics": metrics, "fit_seconds": .1})
    write(models / "gradient_boost_expand.joblib", {"fixture": "hash-only"})
    files = {name: sha256((models / name).read_bytes()).hexdigest() for name in ("protocol.json", "training.json", "gradient_boost_expand.joblib")}
    write(models / "manifest.json", {"schema": "fixed-terminal-stage7-model-manifest-v1", "status": "completed", "frozen": True,
          "roundtrip_predictions_equal": True, "test_read": False, "smoke_only": False, "kind": "expand", "name": "gradient_boost_expand",
          "feature_schema": SCHEMA, "feature_names": list(FEATURE_NAMES), "feature_dim": FEATURE_DIM,
          "training": TRAINING, "threshold": .5, "trace_widths": list(TRACE_WIDTHS), "fit_count": 1, "model_class": "GradientBoostingClassifier",
          "splits": {"train": list(TRAIN_SEEDS), "val": list(VAL_SEEDS)}, "metrics": metrics, "files": files})
    frozen = evaluation.frozen_models(models)
    for phase in ("episodes", "panels"):
        write(tmp_path / f"execution_{phase}.json", {"frozen_models_before": frozen, "frozen_models_after": frozen})
    assert summary.model_training_evidence(tmp_path, data)["context_hashes_match"]
    data["splits"]["val"][0]["context_hash"] = "changed-raw-bytes"
    with pytest.raises(ValueError, match="context identity/hash"):
        summary.model_training_evidence(tmp_path, data)
