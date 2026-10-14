"""Real bounded stage7 decisions and output accounting; no formal training."""
from dataclasses import replace
from types import SimpleNamespace
import json
import pytest
from terminal_marl.configs.fixed_terminal_stage1 import DEFAULT_CONFIG
from terminal_marl.core.fixed_terminal.scenarios import Request, Scenario
from terminal_marl.envs.fixed_terminal_env import FixedTerminalEnv
from terminal_marl.scripts import evaluate_fixed_terminal_stage7 as evaluation


def fixture(empty=False):
    requests = () if empty else (Request("a", 0., .8, 0, .1), Request("b", 0., .7, 1, .4))
    return Scenario(7001, "low", "high", requests, 1, 2)


def policies():
    return {"gradient_boost_expand": SimpleNamespace(name="gradient_boost_expand", kind="expand", predict_expand=lambda rows: [.75 for _ in rows])}


def test_seed_freeze_and_no_future_occupancy_change(tmp_path):
    from terminal_marl.scripts.evaluate_fixed_terminal_stage6 import scenario_for
    for seed in (7301, 7302, 7303):
        for variant in ("standard", "initial_sparse", "initial_dense"):
            assert evaluation.scenario_for(seed, "medium", "medium", variant) == scenario_for(seed, "medium", "medium", variant)
    with pytest.raises(ValueError, match="seed"):
        evaluation.validate_run(7001, False, DEFAULT_CONFIG)
    with pytest.raises(ValueError, match="unchanged"):
        evaluation.validate_run(7301, False, replace(DEFAULT_CONFIG, horizon=2.))
    with pytest.raises(ValueError, match="frozen|require"):
        evaluation.run_episode(tmp_path / "missing", "learned_pruning", smoke=True, config=replace(DEFAULT_CONFIG, horizon=2.), scenario=fixture(), soc="high")
    assert not (tmp_path / "missing").exists()


def test_real_episode_json_conservation_counters_and_no_overwrite(tmp_path):
    target = tmp_path / "episode"
    result = evaluation.run_episode(target, "beam_b3", smoke=True, config=replace(DEFAULT_CONFIG, horizon=6.), scenario=fixture(), soc="high")
    rows = [json.loads(line) for line in (target / "decisions.jsonl").read_text().splitlines()]
    assert result["request_conservation"] and result["summary"]["conservation_ok"]
    assert result["work"] == evaluation.aggregate_work(rows)
    assert result["adaptive_summary"] == evaluation.adaptive_summary(rows)
    assert len(rows) == 2 and all(r["search"]["nodes"] for r in rows)
    assert all(r["final_check"]["status"] == "feasible" for r in rows)
    assert result["model_hashes_unchanged"]
    with pytest.raises(FileExistsError):
        evaluation.run_episode(target, "beam_b3", smoke=True, config=replace(DEFAULT_CONFIG, horizon=6.), scenario=fixture(), soc="high")


@pytest.mark.parametrize("empty", (False, True))
def test_real_shared_root_panel_and_independent_search_cost(tmp_path, empty):
    env = FixedTerminalEnv(replace(DEFAULT_CONFIG, horizon=2.), fixture(empty))
    before = evaluation.canonical_json(evaluation.visible_snapshot(env))
    result = evaluation.timing_panel(env, tmp_path / "panel", policies(), 2)
    assert len(result["rows"]) == 8 and result["arm_order"][0] == "beam_b8"
    assert result["identical_snapshot_and_catalogue_domain"] and result["original_environment_unchanged"]
    assert result["cross_method_cache"] is False
    assert evaluation.canonical_json(evaluation.visible_snapshot(env)) == before
    labels = json.loads((tmp_path / "panel/full_values.json").read_text())
    full_best = max(evaluation.value_comparison_key(labels["values"][rid]) for rid in labels["full_pool_root_ids"])
    common = None
    for row in result["rows"]:
        raw = json.loads((tmp_path / "panel" / (row["arm"] + ".json")).read_text())
        ids = set(raw["evaluations"])
        common = ids if common is None else common & ids
        assert raw["budget"] == raw["work"]["root_rollout_count"] == len(ids)
        assert evaluation.canonical_json(row["chosen_value"]) == evaluation.canonical_json(labels["values"][row["chosen_root_id"]])
        assert evaluation.value_comparison_key(row["full_best_value"]) == full_best
        assert row["new_value_above_finite_reference"] == (evaluation.value_comparison_key(row["chosen_value"]) > full_best)
    assert common
    # Actual repeated closed evaluations have different runtime but identical scientific values.
    rid = sorted(common)[0]
    values = [json.loads((tmp_path / "panel" / (r["arm"] + ".json")).read_text())["evaluations"][rid]["value"] for r in result["rows"]]
    assert all(evaluation.scientific_value(v) == evaluation.scientific_value(values[0]) for v in values)
    assert len({v["elapsed_through_stop_seconds"] for v in values}) > 1
    if empty:
        assert all(r["degenerate"] and r["budget"] == 1 for r in result["rows"])


def test_panel_generation_prefix_and_model_posthash(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "load_scorers", lambda *a: (policies(), {"gradient_boost_expand": .01}, None))
    records = evaluation.run_panels(tmp_path / "panels", smoke=True, config=replace(DEFAULT_CONFIG, horizon=6.), scenario=fixture(), times=(0., 5.))
    assert [r["time"] for r in records] == [0., 5.]
    generation = json.loads((tmp_path / "panels/generation.json").read_text())
    assert generation["prefix_decisions_before_snapshot"] and generation["model_hashes_unchanged"]
    with pytest.raises(ValueError, match="changed"):
        evaluation._postcheck(None, True, {"changed": True})
