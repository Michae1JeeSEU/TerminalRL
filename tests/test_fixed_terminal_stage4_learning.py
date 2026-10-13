from copy import deepcopy
from dataclasses import replace
import json
import numpy as np
import pytest
import torch
from terminal_marl.configs.fixed_terminal_stage4 import SNAPSHOT_TIMES, jobs, validate_split
from terminal_marl.core.fixed_terminal.learning_data import generate_scenario_data, generate_context, load_contexts
from terminal_marl.core.fixed_terminal.learning_models import Standardizer, MLP, Scorer, true_pairs, weighted_training_pairs, train_models, validate_training_groups, offline_metrics
from terminal_marl.core.fixed_terminal.scenarios import Request, Scenario
from terminal_marl.configs.fixed_terminal_stage1 import DEFAULT_CONFIG
from terminal_marl.envs.fixed_terminal_env import FixedTerminalEnv
from terminal_marl.scripts.evaluate_fixed_terminal_stage4 import run_episode, evaluate_offline, aggregate, timing_summary


def tiny(*requests, horizon=20):
    return FixedTerminalEnv(replace(DEFAULT_CONFIG, horizon=horizon), Scenario(4001, "test", "test", tuple(requests), 1, 2))


@pytest.fixture
def dataset(tmp_path):
    env = tiny(Request("a", 0, .7, 0, 0))
    path = tmp_path/"data"
    generate_context(env, path/"t0", {"split": "pilot", "seed": 4001, "load": "low", "soc": "medium"})
    return path, load_contexts(path, "pilot", training=True)


def test_group_split_trainonly_scaling_and_frozen_snapshot_times(dataset):
    path, contexts = dataset
    assert SNAPSHOT_TIMES == (0., 5., 15., 90.)
    assert len(jobs("train"))*len(SNAPSHOT_TIMES) == 108
    assert len(jobs("val"))*len(SNAPSHOT_TIMES) == 36
    assert len(jobs("test"))*len(SNAPSHOT_TIMES) == 108
    with pytest.raises(ValueError):
        validate_split("train", 4301)
    with pytest.raises(ValueError, match="never read test"):
        load_contexts(path, "test", training=True)
    train, val = deepcopy(contexts), deepcopy(contexts)
    train[0]["metadata"].update(split="train", seed=4101)
    val[0]["metadata"].update(split="val", seed=4201)
    validate_training_groups(train, val)
    scaler = Standardizer.fit(train[0]["features"])
    old = scaler.serializable()
    val[0]["features"] = np.full_like(np.array(val[0]["features"]), 9999).tolist()
    assert scaler.serializable() == old
    val[0]["metadata"].update(split="test", seed=4301)
    with pytest.raises(ValueError, match="test"):
        validate_training_groups(train, val)
    assert np.all(Standardizer.fit([[1, 2], [1, 4]]).scale == np.array([1, 1]))


def test_rank_ties_direction_and_persistence(tmp_path, dataset):
    _, contexts = dataset
    context = deepcopy(contexts[0])
    first = context["values"][0]
    context["values"] = [deepcopy(first) for _ in range(3)]
    context["values"][1]["J"] = list(context["values"][1]["J"])
    context["values"][1]["J"][1] += 1
    assert true_pairs(context) == [(1, 0), (1, 2)]
    context["features"] = [[0]*93 for _ in range(3)]
    other = deepcopy(context)
    other["values"][2]["J"] = list(other["values"][2]["J"])
    other["values"][2]["J"][1] -= 1
    pairs, weights = weighted_training_pairs([context, other])
    assert len(pairs) == 5 and sum(weights[:2]) == pytest.approx(.5) and sum(weights[2:]) == pytest.approx(.5)
    assert torch.nn.functional.softplus(torch.tensor(-2.)).item() < torch.nn.functional.softplus(torch.tensor(2.)).item()
    inputs = Standardizer.fit(contexts[0]["features"])
    torch.manual_seed(4401)
    scorer = Scorer("rank", MLP(1), inputs)
    prediction = scorer.predict(contexts[0]["features"])
    scorer.save(tmp_path/"rank.pt", {"smoke_only": True})
    loaded = Scorer.load(tmp_path/"rank.pt")
    assert prediction == loaded.predict(contexts[0]["features"])
    assert torch.get_num_threads() == 1 and next(loaded.model.parameters()).device.type == "cpu"


def test_tiny_training_offline_and_online_serialization(tmp_path, dataset):
    path, contexts = dataset
    models = tmp_path/"models"
    result = train_models(contexts, contexts, models, epochs=2, smoke=True)
    assert set(result) == {"value", "rank"}
    protocol = json.loads((models/"protocol.json").read_text())
    assert protocol["test_read"] is False and protocol["smoke_only"]
    offline = evaluate_offline(path, models, tmp_path/"offline.json", "pilot")
    assert set(offline["methods"]) == {"beam_topk", "value_topk", "rank_topk"}
    for method in offline["methods"].values():
        assert method["groups"]["all"]["filtered_vs_full"]["win"] == 0
    env = tiny(Request("a", 0, .7, 0, 0), horizon=12)
    episode = run_episode(tmp_path/"episode", "value_topk", seed=4001, models=models, config=env.config, scenario=env.scenario)
    assert episode["scheduler"] == "value_topk" and episode["summary"]["conservation_ok"]
    assert episode["summary"]["separation_violation_pair_ticks"] == 0
    assert episode["decision_timings"]["all"]["decisions"] == 3
    assert "p95" in episode["decision_timings"]["all"]["seconds"]["total"]
    assert aggregate(tmp_path/"episode", tmp_path/"aggregate")[0]["method"] == "value_topk"
    assert json.loads((tmp_path/"aggregate.json").read_text())[0]["requests"] == 1
    for file in tmp_path.rglob("*.json"):
        json.loads(file.read_text(encoding="utf-8"), parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
    for file in tmp_path.rglob("*.jsonl"):
        for line in file.read_text(encoding="utf-8").splitlines():
            json.loads(line)


def test_prefix_snapshots_are_predecision_and_globally_aligned(tmp_path):
    env = tiny(Request("a", 0, .7, 0, 0), horizon=20)
    generate_scenario_data(tmp_path/"data", "pilot", 4001, "low", "medium", config=env.config, scenario=env.scenario, times=(0, 5, 15))
    first = json.loads((tmp_path/"data/t0/snapshot.json").read_text())
    second = json.loads((tmp_path/"data/t5/snapshot.json").read_text())
    assert first["aircraft"][0]["state"] == "holding"
    assert second["aircraft"][0]["plan"]["segments"][0]["start"] == 0
