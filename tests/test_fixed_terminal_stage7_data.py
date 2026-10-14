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
    source_audit = data.compare_source_scenario(canonical_json(data.asdict(env.scenario)), canonical_json(data.asdict(env.scenario)))
    context_metadata = {**metadata(), "source_input": {"source_scenario_comparison": source_audit}}
    record = data.generate_context(env, tmp_path/"context", context_metadata)
    assert record["source_scenario_comparison"] == source_audit
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


def source_scenario_fixture():
    return canonical_json(data.asdict(Scenario(4201, "high", "low", tuple(
        Request(f"a-{index}", float(index), .5, None, .1) for index in range(42)), 1, 2)))


@pytest.mark.parametrize("profile,index,stored_soc,rebuilt_soc", [
    ("medium-low", 23, .17577007356836272, .17577007356836266),
    ("high-low", 23, .17577007356836272, .17577007356836266),
    ("high-low", 41, .4411787410317879, .44117874103178784),
    ("high-medium", 41, .7194144892381455, .7194144892381454),
])
def test_source_scenario_four_observed_soc_roundoff_patterns(profile, index, stored_soc, rebuilt_soc):
    stored = source_scenario_fixture()
    stored["load"], stored["soc_profile"] = profile.split("-")
    stored["requests"][index]["soc"] = stored_soc
    rebuilt = deepcopy(stored)
    rebuilt["requests"][index]["soc"] = rebuilt_soc
    before_stored, before_rebuilt = deepcopy(stored), deepcopy(rebuilt)
    audit = data.compare_source_scenario(stored, rebuilt)
    assert not audit["exact_match"] and audit["float_difference_count"] == 1
    assert audit["max_abs_difference"] == abs(stored_soc-rebuilt_soc)
    assert audit["absolute_tolerance"] == 1e-12 and audit["relative_tolerance"] == 0.
    assert audit["differing_paths"] == [f"scenario.requests[{index}].soc"]
    assert audit["differences"] == [{"path": f"scenario.requests[{index}].soc", "stored": stored_soc,
                                      "rebuilt": rebuilt_soc, "abs_difference": abs(stored_soc-rebuilt_soc)}]
    assert "soc" in audit["scope"] and "all_other_fields" in audit["scope"]
    assert stored == before_stored and rebuilt == before_rebuilt


def test_source_scenario_exact_and_multi_difference_audit():
    stored = source_scenario_fixture()
    exact = data.compare_source_scenario(stored, deepcopy(stored))
    assert exact["exact_match"] and exact["float_difference_count"] == 0
    assert exact["max_abs_difference"] == 0. and exact["differing_paths"] == []
    stored["requests"][23]["soc"] = .17577007356836272
    stored["requests"][41]["soc"] = .4411787410317879
    rebuilt = deepcopy(stored)
    rebuilt["requests"][23]["soc"] = .17577007356836266
    rebuilt["requests"][41]["soc"] = .44117874103178784
    audit = data.compare_source_scenario(stored, rebuilt)
    assert audit["float_difference_count"] == 2
    assert audit["differing_paths"] == ["scenario.requests[23].soc", "scenario.requests[41].soc"]


@pytest.mark.parametrize("changed_soc", [.5+2e-12, 0, True, "0.5", float("nan"), float("inf"), -float("inf")])
def test_source_scenario_soc_large_difference_type_and_nonfinite_rejected(changed_soc):
    stored = source_scenario_fixture()
    rebuilt = deepcopy(stored)
    rebuilt["requests"][0]["soc"] = changed_soc
    with pytest.raises(ValueError):
        data.compare_source_scenario(stored, rebuilt)
    with pytest.raises(ValueError):
        data.compare_source_scenario(rebuilt, stored)


@pytest.mark.parametrize("mutation", ["arrival_time", "theta", "seed", "bool_type", "soc_profile", "request_id",
                                     "request_length", "request_order", "request_key_order", "scenario_key_order", "missing_soc"])
def test_source_scenario_other_float_discrete_and_structure_changes_rejected(mutation):
    stored = source_scenario_fixture()
    rebuilt = deepcopy(stored)
    if mutation in ("arrival_time", "theta"):
        rebuilt["requests"][1][mutation] += 1e-15
    elif mutation == "seed":
        rebuilt["seed"] += 1
    elif mutation == "bool_type":
        rebuilt["demand_stream_seed"] = True
    elif mutation == "soc_profile":
        rebuilt["soc_profile"] = "medium"
    elif mutation == "request_id":
        rebuilt["requests"][0]["aircraft_id"] = "changed"
    elif mutation == "request_length":
        rebuilt["requests"].pop()
    elif mutation == "request_order":
        rebuilt["requests"][0], rebuilt["requests"][1] = rebuilt["requests"][1], rebuilt["requests"][0]
    elif mutation == "request_key_order":
        rebuilt["requests"][0] = dict(reversed(list(rebuilt["requests"][0].items())))
    elif mutation == "scenario_key_order":
        rebuilt = dict(reversed(list(rebuilt.items())))
    elif mutation == "missing_soc":
        del rebuilt["requests"][0]["soc"]
    with pytest.raises(ValueError):
        data.compare_source_scenario(stored, rebuilt)


def test_source_scenario_generation_uses_stored_soc_and_saves_audit(tmp_path, monkeypatch):
    stored_scenario = Scenario(4201, "medium", "low", (Request("a", 0., .17577007356836272, 0, .1),), 1, 2)
    rebuilt_scenario = replace(stored_scenario, requests=(replace(stored_scenario.requests[0], soc=.17577007356836266),))
    source = tmp_path/"source"/"frozen_input.json"
    frozen_row = {"split": "val", "seed": 4201, "load": "medium", "soc": "low",
                  "scenario": data.asdict(stored_scenario), "config": data.asdict(DEFAULT_CONFIG), "times": data.SNAPSHOT_TIMES}
    write_json(source, frozen_row)
    source_hash = data.file_hash(source)
    monkeypatch.setattr(data, "generate_scenario", lambda *args: rebuilt_scenario)
    monkeypatch.setattr(data, "apply_scheduler", lambda *args: None)
    captured = []
    def context(env, output, metadata, source_snapshot=None):
        assert env.scenario.requests[0].soc == stored_scenario.requests[0].soc
        assert env.scenario.requests[0].soc != rebuilt_scenario.requests[0].soc
        captured.append(metadata)
        return {**metadata, "time": env.time, "status": "completed"}
    monkeypatch.setattr(data, "generate_context", context)
    output = tmp_path/"generated"
    records = data.generate_scenario_data(output, "val", 4201, "medium", "low", source_stage4=source.parent)
    assert len(records) == len(captured) == 4
    generated_input = json.loads((output/"frozen_input.json").read_text())
    generated = json.loads((output/"generation.json").read_text())
    audit = generated["source_scenario_comparison"]
    assert audit["float_difference_count"] == 1 and not audit["exact_match"]
    assert generated_input["scenario"] == canonical_json(data.asdict(stored_scenario)) == generated["scenario"]
    assert generated_input["source_scenario_comparison"] == audit
    assert all(c["source_input"]["source_scenario_comparison"] == audit for c in captured)
    assert generated["source"]["sha256"] == source_hash == data.file_hash(source)


@pytest.mark.parametrize("field", ["config", "times"])
def test_source_input_config_and_times_types_remain_exact(tmp_path, field):
    config = canonical_json(data.asdict(DEFAULT_CONFIG))
    times = list(data.SNAPSHOT_TIMES)
    if field == "config":
        config["horizon"] = int(config["horizon"])
    else:
        times[0] = 0
    write_json(tmp_path/"frozen_input.json", {"split": "val", "seed": 4201, "load": "medium", "soc": "low",
               "config": config, "times": times})
    with pytest.raises(ValueError, match="type"):
        data._source_input(tmp_path, "val", 4201, "medium", "low", DEFAULT_CONFIG, data.SNAPSHOT_TIMES)
