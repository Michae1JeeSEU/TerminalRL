"""Hand-computable stage6 fixtures; no model training or simulation."""
from copy import deepcopy
import json
import shutil
import pytest

from terminal_marl.scripts import summarize_fixed_terminal_stage6 as summary
from terminal_marl.scripts.evaluate_fixed_terminal_stage4 import timing_summary


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def value(n=1, f=0, c=6, delay=10.):
    components = {"normal_landed": n, "emergency_landed": 0, "failed_energy": f, "arrived_unresolved": c,
                  "normal_delay_sum_seconds": delay, "observed_request_seconds": 20., "energy_consumed_soc_all_requests": .1}
    return {"J": [-f, n, 0, -c, -20., -delay, -.1], "components": components,
            "request_denominator": n+f+c, "time": 240., "episode_cumulative": True}


def decision(seconds=1., rollouts=3, degenerate=False):
    ids = [f"root-{i}" for i in range(rollouts)]
    cert = dict(A=7, F0=0, E0=0, F=0, E=0, N=1, C=6, passed=False)
    stages = [dict(target_budget=rollouts, new_root_ids=ids, cumulative_count=rollouts,
                   chosen_root_id=ids[0], certificate=cert, elapsed_seconds=seconds)] if ids else []
    return {"time": 0., "timing": {"generation": seconds/2, "exact_rollout": seconds/2, "total": seconds},
            "degenerate": degenerate, "pool_count": 1 if degenerate else 8,
            "budget": rollouts, "final_check": {"status": "feasible"},
            "work": dict(zip(summary.WORK_FIELDS, (rollouts, 10*rollouts, rollouts, rollouts, 1))),
            "statistics": {"generated": 3, "examined": 5, "joint_assignments": 7, "lookahead_states": 11,
                           "truncated": False, "rejection_reasons": {}},
            "adaptive": {"enabled": True, "stages": stages, "fallback_count": 0,
                         "stop_reason": "single_root" if degenerate else "fixed_budget",
                         "quality_certified": None if degenerate else False, "visible_counts": {"A": 7, "F0": 0, "E0": 0}}}


def episode(root, arm="beam_b5", load="low", soc="medium", variant="standard", n=1, f=0, delay=10., seconds=1.):
    path = root/"episodes"/f"{arm}-{load}-{soc}-6001-{variant}"
    d = decision(seconds)
    final = value(n, f, 7-n-f, delay)
    counts = {"normal_landed": n, "emergency_landed": 0, "failed_energy": f, "censored": 7-n-f, "not_arrived": 0}
    requests = [{"aircraft_id": str(i), "outcome": outcome, "observed_request_seconds": 2., "energy_consumed_soc": .01} for i, outcome in enumerate(
        ["normal_landed"]*n+["failed_energy"]*f+["censored"]*(7-n-f))]
    scenario = {"seed": 6001, "load": load, "soc_profile": soc, "requests": ["paired scenario"]}
    result = {"schema": "fixed-terminal-stage6-episode-v1", "scheduler": arm, "arm": {"name": arm}, "variant": variant,
              "config": {}, "scenario": scenario, "smoke_only": True, "models": None, "models_after": None,
              "model_hashes_unchanged": True, "requests": requests, "request_conservation": True,
              "source_commit": "fixture", "model_load_seconds": {}, "environment_step_seconds": .01,
              "decision_event_io_seconds": .02, "final_value": final, "decision_timings": timing_summary([d]), "work": d["work"],
              "summary": {"request_count": 7, "outcomes": counts, "conservation_ok": True,
                          "normal_delay_mean_seconds": delay/n if n else None,
                          "separation_violation_pair_ticks": 0, "spacing_uncertified_pair_ticks": 0, "minimum_landed_soc": .1}}
    write(path/"result.json", result)
    write(path/"frozen_input.json", {k: result[k] for k in ("scenario", "config", "models", "arm", "variant")})
    (path/"decisions.jsonl").write_text(json.dumps(d)+"\n")
    (path/"requests.jsonl").write_text("".join(json.dumps(r)+"\n" for r in requests))
    return path


def test_quality_failure_cannot_be_offset_by_more_normal(tmp_path):
    for arm in summary.REFERENCES:
        episode(tmp_path, arm, n=2)
    episode(tmp_path, "rank_adaptive", n=5, f=1)
    rows, _ = summary.load_episodes(tmp_path, True)
    paired = summary.paired_episodes(rows)
    assert len(paired["rows"]) == 5
    assert all(r["comparison"] == "loss" for r in paired["rows"])
    assert all(r["normal_condition"] and r["failed_scenario_increases"] == 1 and not r["quality_condition_passed"] for r in paired["conditions"])


def test_conditional_delay_full_denominator_grouping_and_zero(tmp_path):
    episode(tmp_path, n=2, delay=10.)
    episode(tmp_path, load="medium", variant="initial_sparse", n=1, delay=20.)
    episode(tmp_path, load="high", variant="initial_dense", n=0, delay=0.)
    rows, _ = summary.load_episodes(tmp_path, True)
    groups = summary.aggregate_episodes(rows)
    all_row = next(r for r in groups if r["scope"] == r["axis"] == "all")
    assert all_row["normal_delay_weighted_mean_seconds"] == 10.
    assert all_row["normal_delay_denominator"] == 3 and all_row["request_count"] == 21
    assert all_row["normal_proportion_all_requests"] == 3/21
    assert {r["scope"] for r in groups} == {"all", "standard", "initial_sparse", "initial_dense"}
    assert next(r for r in groups if r["scope"] == "initial_dense")["normal_delay_weighted_mean_seconds"] is None


def test_nested_unique_budget_and_certificate_validation():
    row = decision()
    stage = deepcopy(row["adaptive"]["stages"][0])
    stage.update(target_budget=5, new_root_ids=["root-3", "root-4"], cumulative_count=5)
    row["adaptive"]["stages"].append(stage)
    row["adaptive"]["fallback_count"] = 1
    row["budget"] = row["work"]["root_rollout_count"] = 5
    summary.validate_stages(row)
    totals = summary.adaptive_totals([row])
    assert totals["unique_exact_root_evaluations"] == 5 and totals["expansion_rate"] == 1
    row["adaptive"]["stages"][-1]["new_root_ids"][0] = "root-0"
    with pytest.raises(ValueError, match="duplicate cached"):
        summary.validate_stages(row)
    single = decision(rollouts=0, degenerate=True)
    summary.validate_stages(single)
    assert summary.adaptive_totals([single])["quality_certificate_rate"] is None


def test_compute_condition_uses_nondegenerate_and_zero_denominator():
    rows = []
    for time, degenerate, budgets, seconds in ((0., False, (3, 5), (8., 10.)), (60., True, (0, 0), (100., 100.))):
        for arm, budget, wall in zip(("rank_adaptive", "beam_b5"), budgets, seconds):
            rows.append({"arm": arm, "seed": 6001, "load": "low", "soc": "high", "variant": "standard", "time": time,
                         **decision(wall, budget, degenerate), "chosen_value": value()})
            rows[-1]["time"] = time
    result = summary.same_state_comparisons(rows)
    groups = {r["stratum"]:r for r in result["groups"] if r["scope"] == r["axis"] == "all"}
    assert groups["nondegenerate"]["compute_condition_passed"] is True
    assert groups["nondegenerate"]["wall_ratio"] == .8 and groups["nondegenerate"]["rollout_ratio"] == .6
    assert groups["all"]["compute_condition_passed"] is False
    assert groups["degenerate"]["rollout_ratio"] is None and groups["degenerate"]["compute_condition_passed"] is False


def test_strict_matrix_partial_status_serialization_no_overwrite(tmp_path):
    _, expected = summary.expected_paths()
    assert {k:len(v) for k,v in expected.items()} == dict(episodes=264, panels=30, generations=15)
    with pytest.raises(summary.IncompleteEvidence):
        summary.summarize(tmp_path)
    episode(tmp_path)
    result = summary.summarize(tmp_path, True)
    assert result["status"] == "fixture_or_pilot_incomplete"
    output = tmp_path/"summary"
    summary.write_outputs(output, result)
    assert len(list(output.glob("*.csv"))) == 5
    assert summary.read_json(output/"summary.json")["counts"]["episodes"] == 1
    with pytest.raises(FileExistsError):
        summary.write_outputs(output, result)
    shutil.copytree(next((tmp_path/"episodes").iterdir()), tmp_path/"episodes/duplicate")
    with pytest.raises(ValueError, match="duplicate episode"):
        summary.summarize(tmp_path, True)


def test_nonfinite_model_and_request_mismatch_rejected(tmp_path):
    bad = tmp_path/"bad.json"
    bad.write_text('{"n":NaN}')
    with pytest.raises(ValueError, match="nonfinite"):
        summary.read_json(bad)
    path = episode(tmp_path)
    result = summary.read_json(path/"result.json")
    result["models_after"] = {"changed": True}
    write(path/"result.json", result)
    with pytest.raises(ValueError, match="model hashes"):
        summary.load_episodes(tmp_path, True)

def test_certificate_at_full_budget_is_not_early_stop():
    row = decision(rollouts=8)
    cert = row["adaptive"]["stages"][0]["certificate"]
    cert.update(N=7, C=0, passed=True)
    row["adaptive"].update(quality_certified=True, stop_reason="visible_normal_upper_bound_reached")
    summary.validate_stages(row)
    totals = summary.adaptive_totals([row])
    assert totals["quality_certificate_rate"] == 1
    assert totals["certified_early_stop_rate"] == 0 and totals["full_pool_evaluated_rate"] == 1
    assert totals["full_pool_exhausted_without_certificate_rate"] == 0
    row["adaptive"]["enabled"] = False
    totals = summary.adaptive_totals([row])
    assert totals["full_pool_evaluated_rate"] is None and totals["expansion_rate"] is None


def test_panel_raw_summary_consistency(tmp_path):
    base = tmp_path/"panels/fixture"
    scenario = {"seed": 6001, "load": "low", "soc_profile": "high"}
    shared = dict(config={}, scenario=scenario, variant="standard", models=None, times=[0.])
    write(base/"frozen_input.json", shared)
    write(base/"generation.json", {**shared, "schema": "fixed-terminal-stage6-generation-v1", "status": "completed",
          "models_after": None, "model_hashes_unchanged": True, "smoke_only": True,
          "contexts": [{"time": 0., "status": "completed"}]})
    rows = []
    for arm in summary.ARMS:
        row = {**decision(), "arm": arm, "chosen_root_id": "root-0", "chosen_value": value(),
               "full_best_value": value(), "best_key_retained": True}
        rows.append(row)
        write(base/"t0"/(arm+".json"), row)
    panel = dict(schema="fixed-terminal-stage6-panel-v1", identical_snapshot_and_pool=True,
                 cross_method_cache=False, original_environment_unchanged=True, analysis_label_cost={"seconds": 0}, rows=rows)
    write(base/"t0/summary.json", panel)
    write(base/"t0/full_values.json", {"values": {"root-0": value()}})
    loaded, keys, _, _ = summary.load_panels(tmp_path, True)
    assert len(loaded) == 8 and len(keys) == 1
    panel["rows"][0]["timing"]["total"] = 100.
    write(base/"t0/summary.json", panel)
    with pytest.raises(ValueError, match="panel summary/raw mismatch"):
        summary.load_panels(tmp_path, True)
