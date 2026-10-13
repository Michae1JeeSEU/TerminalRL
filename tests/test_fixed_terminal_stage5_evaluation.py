from dataclasses import asdict, replace
import json
from math import pi
from types import SimpleNamespace
import pytest

from terminal_marl.configs.fixed_terminal_stage1 import DEFAULT_CONFIG
from terminal_marl.core.fixed_terminal.scenarios import Request, Scenario
from terminal_marl.envs.fixed_terminal_env import FixedTerminalEnv
from terminal_marl.scripts import evaluate_fixed_terminal_stage5 as evaluation


def scenario():
    return Scenario(5001, "low", "medium", (Request("a", 0, .8, 0, .1),), 1, 2)


def test_initial_rotation_only_theta_and_paired_draws():
    standard = evaluation.scenario_for(5001, "medium", "medium")
    rotated = evaluation.scenario_for(5001, "medium", "medium", "initial_rotated")
    assert standard.demand_stream_seed == rotated.demand_stream_seed and standard.soc_stream_seed == rotated.soc_stream_seed
    for before, after in zip(standard.requests, rotated.requests):
        if before.ring is not None:
            assert after == replace(before, theta=(before.theta+(before.ring+1)*pi/7) % (2*pi))
        else:
            assert after == before
    with pytest.raises(ValueError):
        evaluation.scenario_for(5001, "low", "low", "initial_rotated")


def test_cli_jobs_and_formal_lock(capsys, tmp_path):
    for mode, count in (("episode", 330), ("snapshot", 30), ("continuation", 9)):
        assert evaluation.main(["--mode", mode, "--list-jobs"]) == 0
        assert len(json.loads(capsys.readouterr().out)) == count
    with pytest.raises(ValueError, match="manifest"):
        evaluation.run_episode(tmp_path/"blocked", "beam_b5", seed=5301)
    assert not (tmp_path/"blocked").exists()


def test_small_pilot_episode_and_snapshot_json(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINAL_SOURCE_COMMIT", "fixture-source")
    config = replace(DEFAULT_CONFIG, horizon=6.)
    result = evaluation.run_episode(tmp_path/"episode", "beam_b3", seed=5001, smoke=True, config=config, scenario=scenario())
    assert result["schema"] == "fixed-terminal-stage5-episode-v1" and result["source_commit"] == "fixture-source"
    assert result["smoke_only"] and result["decision_timings"]["all"]["decisions"] == 2
    assert result["work"]["final_submit_count"] == 2
    assert result["final_value"]["episode_cumulative"] and result["final_value"]["time"] == 6.
    records = evaluation.run_snapshots(tmp_path/"snapshots", seed=5001, smoke=True, config=config, scenario=scenario(), times=(0., 5.), enable_panels=False)
    assert len(records) == 2 and all(r["status"] == "completed" and r["split"] == "pilot" for r in records)
    for path in tmp_path.rglob("*.json"):
        json.loads(path.read_text())
    for path in tmp_path.rglob("*.jsonl"):
        for line in path.read_text().splitlines():
            json.loads(line)
    curves = json.loads((tmp_path/"snapshots/t0/budget_curves.json").read_text())
    assert len(curves["rows"]) == 3 and all(r["would_evaluate_root_count"] <= r["budget"] for r in curves["rows"])
    with pytest.raises(FileExistsError):
        evaluation.run_episode(tmp_path/"episode", "beam_b3", seed=5001, smoke=True, config=config, scenario=scenario())


def test_real_same_state_panel_rotates_order_and_keeps_source(tmp_path):
    env = FixedTerminalEnv(replace(DEFAULT_CONFIG, horizon=2.), scenario())
    scorers = {name: SimpleNamespace(kind="value" if name == "value" else "rank",
                                    predict=lambda rows, name=name: [[0.]*(7 if name == "value" else 1) for _ in rows])
               for name in ("linear_rank", "mlp_rank", "gradient_boost", "value")}
    before = evaluation.canonical_json(evaluation.visible_snapshot(env))
    result = evaluation.timing_panel(env, tmp_path/"panel", scorers, 2)
    assert len(result["rows"]) == 11 and result["arm_order"][0] == "beam_b8"
    assert result["identical_snapshot_and_pool"] and result["cross_method_cache"] is False
    assert evaluation.canonical_json(evaluation.visible_snapshot(env)) == before


def test_saved_windows_context_path_on_any_host():
    for raw in (r"D:\7_codespace\terminal\tmp_stage4_results\data\test\medium-low-4303\t15",
                "/home/user/tmp_stage4_results/data/test/medium-low-4303/t15"):
        path = evaluation.saved_context_path(raw)
        assert path.parent.name == "medium-low-4303" and path.name == "t15"


def test_continuation_saved_control_union_and_distinct_future_denominator(tmp_path, monkeypatch):
    config = replace(DEFAULT_CONFIG, horizon=16.)
    sample = Scenario(4303, "low", "medium", (Request("a", 0, .8, 0, .1), Request("future", 15.5, .8, None, 0)), 1, 2)
    monkeypatch.setattr(evaluation, "generate_scenario", lambda *args: sample)
    base = tmp_path/"old/data/test/low-medium-4303"
    evaluation.write_json(base/"frozen_input.json", {"config": asdict(config), "scenario": asdict(sample)})
    env = FixedTerminalEnv(config, sample)
    while env.time < 15:
        if env.time % 5 == 0:
            evaluation.apply_scheduler(env, "joint_window")
        env.step()
    evaluation.write_json(base/"t15/snapshot.json", evaluation.visible_snapshot(env))
    roots, meta = evaluation.make_pool(env)
    assert len(roots) == 1
    evaluation.write_json(base/"t15/roots.json", {"roots": [r.serializable() for r in roots], "pool_metadata": meta})
    closed = evaluation.closed_evaluation(env, roots[0])["value"]
    evaluation.write_json(base/"t15/labels.json", {"rows": [{"root_id": roots[0].root_id, "value": closed}]})
    row = {"context": r"D:\old\data\test\low-medium-4303\t15", "filtered_chosen": roots[0].root_id, "full_best_root": roots[0].root_id}
    evaluation.write_json(tmp_path/"old/summary/offline_test.json", {"methods": {m: {"contexts": [row]} for m in ("beam_topk", "rank_topk", "value_topk")}})
    result = evaluation.continuation_diagnostic(tmp_path/"old", tmp_path/"diagnostic", "low", "medium")
    assert len(result["rows"]) == 1 and len(set(result["selected_by"].values())) == 1
    assert result["rows"][0]["closed_value"]["request_denominator"] == 1
    assert result["rows"][0]["realized_value"]["request_denominator"] == 2
    assert json.loads((tmp_path/"diagnostic/result.json").read_text())["scope"].startswith("ex_post")
