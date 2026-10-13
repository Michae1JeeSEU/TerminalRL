"""Bounded real simulations and frozen-input contract checks; no formal runs."""
from dataclasses import asdict, replace
import json
from random import Random
from types import SimpleNamespace

import pytest

from terminal_marl.configs.fixed_terminal_stage1 import DEFAULT_CONFIG, SOC_PROFILES
from terminal_marl.core.fixed_terminal.scenarios import Request, Scenario, stream_seed, truncated_normal
from terminal_marl.envs.fixed_terminal_env import FixedTerminalEnv
from terminal_marl.scripts import evaluate_fixed_terminal_stage6 as evaluation


def fixture(soc="high", empty=False):
    requests = () if empty else (Request("a", 0., .8, 0, .1), Request("b", 2., .7, None, 0.))
    return Scenario(6001, "low", soc, requests, 1, 2)


def scorers():
    return {name: SimpleNamespace(name=name, kind="rank", predict=lambda rows: [[0.] for _ in rows])
            for name in ("mlp_rank", "gradient_boost")}


def read(path):
    return json.loads(path.read_text(), parse_constant=lambda value: pytest.fail("nonfinite JSON: " + value))


@pytest.mark.parametrize("seed", (6301, 6302, 6303))
def test_occupancy_future_pairing_and_independent_extra_soc(seed):
    base = evaluation.scenario_for(seed, "medium", "medium")
    initial = [r for r in base.requests if r.ring is not None]
    future = [replace(r, aircraft_id="ignored") for r in base.requests if r.ring is None]
    for variant, source, counts in (("initial_sparse", "low", [2, 2, 3]), ("initial_dense", "high", [6, 8, 9])):
        sample = evaluation.scenario_for(seed, "medium", "medium", variant)
        assert sample == evaluation.scenario_for(seed, "medium", "medium", variant)
        assert (sample.load, sample.soc_profile, sample.demand_stream_seed, sample.soc_stream_seed) == (
            base.load, base.soc_profile, base.demand_stream_seed, base.soc_stream_seed)
        assert [replace(r, aircraft_id="ignored") for r in sample.requests if r.ring is None] == future
        assert [r.aircraft_id for r in sample.requests] == [f"uav-{i:04d}" for i in range(len(sample.requests))]
        actual_initial = [r for r in sample.requests if r.ring is not None]
        source_initial = [r for r in evaluation.generate_scenario(DEFAULT_CONFIG, seed, source, "medium").requests if r.ring is not None]
        assert [(r.ring, r.theta) for r in actual_initial] == [(r.ring, r.theta) for r in source_initial]
        assert [r.soc for r in actual_initial[:len(initial)]] == [r.soc for r in initial[:len(actual_initial)]]
        metadata = evaluation.scenario_metadata(sample, variant, DEFAULT_CONFIG)
        assert metadata["initial_counts_by_ring"] == counts
        assert metadata["initial_source_load"] == source
        assert metadata["initial_capacity_and_spacing_validated"]
        if variant == "initial_dense":
            extra_seed = stream_seed(seed, "stage6-extra-initial-soc")
            assert metadata["extra_initial_soc_stream_seed"] == extra_seed
            rng, spec = Random(extra_seed), SOC_PROFILES["medium"]
            expected = [truncated_normal(rng, spec.mean, spec.std, spec.lower, spec.upper) for _ in range(8)]
            assert [r.soc for r in actual_initial[15:]] == expected
            assert expected != [r.soc for r in base.requests if r.ring is None][:8]
        else:
            assert metadata["extra_initial_soc_stream_seed"] is None
        FixedTerminalEnv(DEFAULT_CONFIG, sample)


def test_formal_seed_config_fixture_and_model_isolation(tmp_path):
    for seed, smoke in ((6001, False), (6301, True), (True, True), (5001, True)):
        with pytest.raises(ValueError, match="seed"):
            evaluation.validate_run(seed, smoke, DEFAULT_CONFIG)
    with pytest.raises(ValueError, match="unchanged"):
        evaluation.validate_run(6301, False, replace(DEFAULT_CONFIG, horizon=6.))
    with pytest.raises(ValueError, match="unchanged"):
        evaluation.validate_run(6301, False, DEFAULT_CONFIG, fixture())
    with pytest.raises(ValueError, match="matching|require"):
        evaluation.run_episode(tmp_path / "no-model", "rank_adaptive", smoke=True,
                               config=replace(DEFAULT_CONFIG, horizon=6.), scenario=fixture(), soc="high")
    assert not (tmp_path / "no-model").exists()
    with pytest.raises(ValueError, match="manifest"):
        evaluation.run_episode(tmp_path / "formal", "beam_b3", seed=6301)
    assert not (tmp_path / "formal").exists()
    with pytest.raises(ValueError, match="frozen"):
        evaluation.run_panels(tmp_path / "time", seed=6301, times=(0.,))
    with pytest.raises(ValueError, match="occupancy"):
        evaluation.scenario_for(6001, "low", "medium", "initial_dense")


def test_arm_mapping_selective_load_and_identity(monkeypatch):
    assert {name: evaluation.scorer_name(name) for name in evaluation._config().ARMS} == {
        "beam_b3": None, "beam_b5": None, "beam_b8": None, "full_reference": None,
        "beam_adaptive": None, "rank_adaptive": "mlp_rank", "gb_adaptive": "gradient_boost",
        "hybrid_adaptive": "mlp_rank"}
    from terminal_marl.core.fixed_terminal.validation_models import ValidationScorer
    loads, checks = [], []
    def freeze(path, smoke=False):
        checks.append((path, smoke))
        return {"four_hashes": "unchanged"}
    def load(path):
        loads.append(path.name)
        return scorers()[path.stem]
    monkeypatch.setattr(evaluation, "frozen_models", freeze)
    monkeypatch.setattr(ValidationScorer, "load", load)
    assert evaluation.load_scorers("models", ())[0] == {}
    assert loads == [] and len(checks) == 1
    loaded, timings, _ = evaluation.load_scorers("models", ("gradient_boost",))
    assert set(loaded) == set(timings) == {"gradient_boost"} and loads == ["gradient_boost.joblib"]
    monkeypatch.setattr(ValidationScorer, "load", lambda _: SimpleNamespace(name="wrong", kind="rank"))
    with pytest.raises(ValueError, match="identity"):
        evaluation.load_scorers("models", ("mlp_rank",))
    with pytest.raises(ValueError, match="matching"):
        evaluation._apply(FixedTerminalEnv(replace(DEFAULT_CONFIG, horizon=2.), fixture()), "rank_adaptive", {})


def test_small_episode_real_json_conservation_and_no_overwrite(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINAL_SOURCE_COMMIT", "fixture-source")
    config = replace(DEFAULT_CONFIG, horizon=6.)
    target = tmp_path / "episode"
    result = evaluation.run_episode(target, "beam_b3", smoke=True, config=config, scenario=fixture(), soc="high")
    assert result["schema"] == "fixed-terminal-stage6-episode-v1"
    assert result["source_commit"] == "fixture-source" and result["smoke_only"]
    assert result["decision_timings"]["all"]["decisions"] == 2
    assert result["request_conservation"] and result["summary"]["conservation_ok"]
    assert sum(result["request_outcome_counts"].values()) == 2
    assert result["model_hashes_unchanged"] and result["models"] == result["models_after"]
    assert result["model_load_seconds"] == {}
    assert result["final_value"]["time"] == 6. and result["final_value"]["episode_cumulative"]
    assert len(list(target.iterdir())) == 5
    rows = [json.loads(line) for line in (target / "decisions.jsonl").read_text().splitlines()]
    assert result["work"] == evaluation.aggregate_work(rows)
    assert result["adaptive_summary"] == evaluation.adaptive_summary(rows)
    for path in target.iterdir():
        if path.suffix == ".json":
            read(path)
        else:
            for line in path.read_text().splitlines():
                json.loads(line, parse_constant=lambda v: pytest.fail(v))
    with pytest.raises(FileExistsError):
        evaluation.run_episode(target, "beam_b3", smoke=True, config=config, scenario=fixture(), soc="high")


@pytest.mark.parametrize("empty", (False, True))
def test_panel_same_state_pool_rotation_and_external_single_root_label(tmp_path, empty):
    env = FixedTerminalEnv(replace(DEFAULT_CONFIG, horizon=2.), fixture(empty=empty))
    before = evaluation.canonical_json(evaluation.visible_snapshot(env))
    result = evaluation.timing_panel(env, tmp_path / "panel", scorers(), 2)
    assert len(result["rows"]) == 8 and result["arm_order"][0] == "beam_b8"
    assert result["identical_snapshot_and_pool"] and result["original_environment_unchanged"]
    assert result["cross_method_cache"] is False and result["model_loading_excluded"]
    assert evaluation.canonical_json(evaluation.visible_snapshot(env)) == before
    values = read(tmp_path / "panel/full_values.json")["values"]
    pools = []
    for row in result["rows"]:
        raw = read(tmp_path / "panel" / (row["arm"] + ".json"))
        pools.append(evaluation._pool_keys(raw))
        assert evaluation.canonical_json(row["chosen_value"]) == evaluation.canonical_json(values[row["chosen_root_id"]])
        assert row["timing"]["total"] >= sum(v for k, v in row["timing"].items() if k != "total") - 1e-8
        assert len(raw["evaluations"]) == row["work"]["rollouts"]
    assert all(pool == pools[0] for pool in pools)
    assert len(values) == result["pool_count"]
    if empty:
        assert result["degenerate"] and len(values) == 1
        assert all(r["budget"] == r["work"]["rollouts"] == 0 for r in result["rows"])
        assert result["analysis_label_cost"]["work"]["rollouts"] == 1
    else:
        assert result["analysis_label_cost"]["seconds"] == 0.


def test_panel_job_prefix_metadata_and_posthash(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "load_scorers", lambda *a: (scorers(), {"mlp_rank": .01, "gradient_boost": .02}, None))
    records = evaluation.run_panels(tmp_path / "panels", smoke=True, config=replace(DEFAULT_CONFIG, horizon=6.),
                                    scenario=fixture(), times=(0., 5.))
    assert [r["time"] for r in records] == [0., 5.]
    assert all(r["arm_rows"] == 8 and r["smoke_only"] for r in records)
    generation = read(tmp_path / "panels/generation.json")
    assert generation["models"] == generation["models_after"] and generation["model_hashes_unchanged"]
    assert generation["scenario"] == asdict(fixture()) | {"requests": [asdict(r) for r in fixture().requests]}
    assert generation["config"]["horizon"] == 6. and generation["model_load_seconds"]["mlp_rank"] == .01
    assert read(tmp_path / "panels/t0/summary.json")["rotation_index"] + 1 == read(tmp_path / "panels/t5/summary.json")["rotation_index"]
    with pytest.raises(ValueError, match="changed"):
        evaluation._postcheck(None, True, {"changed": True})
