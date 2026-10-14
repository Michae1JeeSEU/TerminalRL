import json
from copy import deepcopy
from terminal_marl.core.fixed_terminal import search_data as data
from terminal_marl.core.fixed_terminal.learning_data import write_json
from terminal_marl.core.fixed_terminal.value_analysis import visible_snapshot, canonical_json
from terminal_marl.core.fixed_terminal.scenarios import Request
from dataclasses import replace
from terminal_marl.configs.fixed_terminal_stage1 import DEFAULT_CONFIG
from terminal_marl.envs.fixed_terminal_env import FixedTerminalEnv
from terminal_marl.core.fixed_terminal.scenarios import Scenario
import pytest
from pathlib import Path
from terminal_marl.core.fixed_terminal.baselines import apply_scheduler
from terminal_marl.core.fixed_terminal.scenarios import generate_scenario


def tiny(*requests):
    return FixedTerminalEnv(replace(DEFAULT_CONFIG, horizon=12.), Scenario(7001, "low", "high", tuple(requests), 1, 2))


def metadata():
    return {"split": "pilot", "seed": 7001, "load": "low", "soc": "high", "smoke_only": True}


def test_trace_labels_have_real_leaf_linkage_and_no_internal_rollouts(tmp_path):
    env = tiny(Request("a", 0, .95, 0, .1))
    record = data.generate_context(env, tmp_path/"context", metadata())
    trace = json.loads((tmp_path/"context"/"trace.json").read_text())
    labels = json.loads((tmp_path/"context"/"labels.json").read_text())
    leaves = json.loads((tmp_path/"context"/"leaves.json").read_text())
    assert all(not r["own_rollout"] for r in labels["rows"])
    assert all(r["evaluation"]["status"] == "feasible" and "certificate" in r for r in leaves["rows"])
    assert all(len(n["lineage"]) == n["depth"] for n in trace["nodes"])
    assert record["trace_source"].startswith("new_instrumented")
    assert record["work"]["rollouts"] == len(leaves["rows"])
    assert record["trace_widths"] == [3, 5, 8, 128]
    assert len(trace["stages"]) == 4
    assert trace["counters"]["partial_check_cache_hits"] > 0
    assert all(trace["nodes"][int(key.split("-n")[1])]["stage"] == 3 for row in leaves["rows"] for key in row["tree_leaf_node_ids"])
    contexts = data.load_contexts(tmp_path, "pilot", training=True)
    from terminal_marl.core.fixed_terminal.search_features import FEATURE_NAMES
    width_index = FEATURE_NAMES.index("search_configured_width")
    assert {row[width_index] for row in contexts[0]["features"]} == {3., 5., 8., 128.}
    assert len(contexts) == 1 and len(contexts[0]["features"]) == record["training_node_count"]
    report = data.analyze_contexts(tmp_path)
    assert report["totals"]["contexts"] == 1
    assert report["depths"]["0"]["positive"] == 4
    with pytest.raises(ValueError, match="training cannot read test"):
        data.load_contexts(tmp_path, "test", training=True)
    (tmp_path/"context"/"features.json").write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        data.load_contexts(tmp_path, "pilot", training=True)


def test_old_artifact_roundoff_is_recorded_and_discrete_mismatch_rejected(tmp_path):
    env = tiny(Request("a", 0, .95, 0, .1))
    snapshot = canonical_json(visible_snapshot(env))
    snapshot["aircraft"][0]["theta"] += 1e-15
    write_json(tmp_path/"stored.json", snapshot)
    record = data.generate_context(env, tmp_path/"roundoff", metadata(), source_snapshot=tmp_path/"stored.json")
    audit = record["source_snapshot_comparison"]
    assert not audit["exact_match"] and audit["float_difference_count"] == 1
    assert audit["max_abs_difference"] < 1e-12
    wrong = deepcopy(snapshot)
    wrong["aircraft"][0]["aircraft_id"] = "changed"
    write_json(tmp_path/"wrong.json", wrong)
    with pytest.raises(ValueError, match="discrete"):
        data.generate_context(env, tmp_path/"failed", metadata(), source_snapshot=tmp_path/"wrong.json")
    assert not (tmp_path/"failed").exists()


def test_scenario_generator_smoke_output_and_formal_guard(tmp_path):
    env = tiny(Request("a", 0, .95, 0, .1))
    records = data.generate_scenario_data(tmp_path/"pilot", "pilot", 7001, "low", "high", smoke=True,
                                         config=env.config, scenario=env.scenario, times=(0., 5.))
    assert [r["time"] for r in records] == [0., 5.]
    assert json.loads((tmp_path/"pilot"/"generation.json").read_text())["status"] == "completed"
    with pytest.raises(ValueError, match="frozen"):
        data.generate_scenario_data(tmp_path/"bad", "train", 4101, "low", "high", config=env.config)


def test_existing_stage4_medium_high_prefix_snapshot():
    base = Path(__file__).resolve().parents[1]/".worktrees"/"arrival-restructure"/"tmp_stage4_results"/"data"/"train"/"medium-high-4101"
    if not (base/"t15"/"snapshot.json").exists():
        pytest.skip("local historical stage4 fixture absent")
    frozen = json.loads((base/"frozen_input.json").read_text())
    scenario = generate_scenario(DEFAULT_CONFIG, 4101, "medium", "high")
    assert canonical_json(data.asdict(scenario)) == frozen["scenario"]
    env = FixedTerminalEnv(DEFAULT_CONFIG, scenario)
    next_decision = 0.
    while env.time < 15.-1e-9:
        if env.time >= next_decision-1e-9:
            apply_scheduler(env, "joint_window")
            next_decision += 5.
        env.step()
    audit = data.compare_source_snapshot(env, base/"t15"/"snapshot.json")
    assert audit["max_abs_difference"] <= 1e-12
