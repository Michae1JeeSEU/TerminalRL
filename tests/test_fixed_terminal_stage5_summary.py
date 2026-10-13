"""Hand-written artifact fixtures only; no model or simulator execution."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import pytest

from terminal_marl.scripts import summarize_fixed_terminal_stage5 as summary
from terminal_marl.scripts.evaluate_fixed_terminal_stage4 import timing_summary


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def value(n=1, f=0, c=6, delay=10., exposure=20., energy=.1):
    components = {"normal_landed": n, "emergency_landed": 0, "failed_energy": f, "arrived_unresolved": c,
                  "normal_delay_sum_seconds": delay, "observed_request_seconds": exposure, "energy_consumed_soc_all_requests": energy}
    return {"J": [-f, n, 0, -c, -exposure, -delay, -energy], "components": components,
            "request_denominator": n+f+c, "time": 240., "episode_cumulative": True}


def work(rollouts=2):
    return dict(zip(summary.WORK_FIELDS, (rollouts, 10*rollouts, 2*rollouts, rollouts, 1)))


def decision(seconds=1.):
    return {"time": 0., "timing": {"generation": seconds/2, "exact_rollout": seconds/2, "total": seconds},
            "degenerate": False, "pool_count": 4, "work": work(),
            "statistics": {"generated": 3, "examined": 5, "joint_assignments": 7, "lookahead_states": 11,
                           "truncated": True, "rejection_reasons": {"unknown:test": 2}}}


def episode(root, arm="beam_b5", load="low", soc="medium", variant="standard", n=1, f=0, delay=10., seconds=1.):
    path = root/"episodes"/f"{arm}-{load}-{soc}-5001-{variant}"
    d = decision(seconds)
    final = value(n, f, 7-n-f, delay)
    counts = {"normal_landed": n, "emergency_landed": 0, "failed_energy": f, "censored": 7-n-f, "not_arrived": 0}
    scenario = {"seed": 5001, "load": load, "soc_profile": soc, "requests": ["same-frozen-scenario"]}
    result = {"schema": "fixed-terminal-stage5-episode-v1", "scheduler": arm, "arm": {"name": arm}, "variant": variant,
              "config": {}, "scenario": scenario, "smoke_only": True, "models": None,
              "source_commit": "fixture-source", "model_load_seconds": {}, "environment_step_seconds": .01,
              "final_value": final, "decision_timings": timing_summary([d]), "work": work(),
              "summary": {"request_count": 7, "outcomes": counts, "normal_delay_mean_seconds": delay/n if n else None,
                          "separation_violation_pair_ticks": 0, "spacing_uncertified_pair_ticks": 0, "minimum_landed_soc": .1}}
    write(path/"result.json", result)
    write(path/"frozen_input.json", {k: result[k] for k in ("scenario", "config", "models", "arm", "variant")})
    (path/"decisions.jsonl").write_text(json.dumps(d)+"\n")
    return path


def snapshot(root):
    base = root/"snapshots/low-medium-5001-standard"
    path = base/"t0"
    fields = {"seed": 5001, "load": "low", "soc": "medium", "variant": "standard", "root_time": 0., "pool_limit": 14}
    context = {"schema": "fixed-terminal-stage5-context-v1", "status": "completed", "smoke_only": True,
               **fields, "root_count": 2, "source_commit": "fixture-source"}
    full, hold = value(), value(0, c=7)
    write(path/"context.json", context)
    write(path/"labels.json", {"rows": [{"root_id": "full", "value": full}, {"root_id": "hold", "value": hold}]})
    write(path/"budget_curves.json", {"full_best_root_id": "full", "full_best_value": full, "informative": True,
              "degenerate": False, "rows": [{"method": "beam", "budget": 3, "selected_root_ids": ["hold", "full"],
              "chosen_root_id": "full", "chosen_value": full, "raw_J_regret_full_minus_chosen": [0]*7,
              "best_key_retained": True, "would_evaluate_root_count": 2}]})
    write(path/"timing_panel/summary.json", {"identical_snapshot_and_pool": True, "cross_method_cache": False,
              "rows": [{"arm": arm, "timing": {"total": seconds}, "work": work(), "degenerate": False, "chosen_root_id": "full"}
                       for arm, seconds in (("beam_b5", 1.), ("full_reference", 2.))]})
    write(base/"generation.json", {"status": "completed", "contexts": [context], "models": None})


def test_weighted_mean_groups_raw_timing_and_search_sums(tmp_path):
    episode(tmp_path, n=2, delay=10., seconds=1.)
    episode(tmp_path, load="medium", soc="medium", variant="initial_rotated", n=1, delay=20., seconds=3.)
    rows, _ = summary.load_episodes(tmp_path, True)
    groups = summary.aggregate_episodes(rows)
    all_row = next(r for r in groups if (r["scope"], r["axis"]) == ("all", "all"))
    assert all_row["normal_delay_weighted_mean_seconds"] == 10.  # (10+20)/(2+1), not (5+20)/2
    assert all_row["normal_delay_denominator"] == 3 and all_row["request_count"] == 14
    assert all_row["timings"]["all"]["seconds"]["total"]["total"] == 4.
    assert all_row["search"]["examined"] == 10 and all_row["search"]["joint_assignments"] == 14
    assert all_row["search"]["rejection_reasons"] == {"unknown:test": 4}
    assert {r["scope"] for r in groups} == {"all", "standard", "initial_rotated"}


def test_paired_lexicographic_failure_precedes_normal_count(tmp_path):
    episode(tmp_path, "beam_b5", n=2, f=0)
    episode(tmp_path, "full_reference", n=2, f=0)
    episode(tmp_path, "mlp_rank_b5", n=5, f=1)
    rows, _ = summary.load_episodes(tmp_path, True)
    paired = summary.paired_episodes(rows)
    selected = [r for r in paired["rows"] if r["arm"] == "mlp_rank_b5"]
    assert len(selected) == 2 and all(r["comparison"] == "loss" for r in selected)
    assert selected[0]["raw_J_delta_arm_minus_reference"][:2] == [-1, 3]
    assert selected[0]["raw_component_delta_arm_minus_reference"]["failed_energy"] == 1


def test_formal_missing_returns_list_and_creates_no_output(tmp_path, capsys):
    output = tmp_path/"summary"
    assert summary.main(["--root", str(tmp_path), "--output", str(output)]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "incomplete" and "execution_test.json" in result["missing"]
    assert not output.exists()
    _, expected = summary.expected_paths()
    assert {k: len(v) for k, v in expected.items()} == {"episodes": 330, "contexts": 99, "panels": 24, "continuations": 9}


def test_duplicate_episode_rejected_even_with_incomplete(tmp_path):
    path = episode(tmp_path)
    shutil.copytree(path, tmp_path/"episodes/duplicate")
    with pytest.raises(ValueError, match="duplicate episode"):
        summary.summarize(tmp_path, True)


def test_fixture_complete_serialization_curves_panel_and_no_overwrite(tmp_path, capsys):
    root, output = tmp_path/"raw", tmp_path/"summary"
    episode(root)
    snapshot(root)
    assert summary.main(["--root", str(root), "--output", str(output), "--allow-incomplete"]) == 0
    result = summary.read_json(output/"summary.json")
    assert result["status"] == "fixture_or_pilot_incomplete" and result["counts"]["timing_rows"] == 2
    assert result["offline"]["groups"][0]["contexts"] == 1
    assert all(r["retention"] == 1 for r in result["offline"]["groups"])
    panel = next(r for r in result["same_state"]["rows"] if r["arm"] == "full_reference")
    assert panel["ratio_total_to_beam_b5"] == 2. and panel["chosen_closed_value"]["J"][1] == 1
    assert len(list(output.glob("*.csv"))) == 9
    with pytest.raises(FileExistsError):
        summary.main(["--root", str(root), "--output", str(output), "--allow-incomplete"])


def test_nonfinite_and_mislabeled_formal_partial_rejected(tmp_path):
    bad = tmp_path/"nonfinite.json"
    bad.write_text('{"x":NaN}')
    with pytest.raises(ValueError, match="nonfinite"):
        summary.read_json(bad)
    path = episode(tmp_path)
    result = json.loads((path/"result.json").read_text())
    result["smoke_only"] = False
    write(path/"result.json", result)
    with pytest.raises(ValueError, match="fixture/pilot"):
        summary.summarize(tmp_path, True)


def test_real_decision_extra_work_fields_and_single_root_fastpath(tmp_path):
    """Real pilot decisions carry work aliases and counters beyond result.work."""
    path = episode(tmp_path, n=0, delay=0., seconds=.2)
    row = decision(.2)
    row.update(degenerate=True, pool_count=1)
    row["timing"] = {"generation": .15, "features": 0., "inference": 0., "exact_rollout": 0.,
                     "final_validation_and_emergency": .05, "total": .2}
    row["work"] = {**work(0), "rollouts": 0, "pool_validation_calls": 1, "final_validation_calls": 1,
                   "counter_scope": "accumulated_search_operations; counters_may_revisit_candidates",
                   "closed_step_count": "ceil((horizon-root_time)/dt); original_evaluator_fixed_step_loop"}
    row["statistics"].update(generated=0, examined=0, joint_assignments=1, lookahead_states=0,
                             truncated=False, rejection_reasons={}, priority_windows={})
    (path/"decisions.jsonl").write_text(json.dumps(row)+"\n")
    result = json.loads((path/"result.json").read_text())
    result.update(work=work(0), decision_timings=timing_summary([row]),
                  model_load_seconds={"linear_rank": .1, "mlp_rank": .2, "gradient_boost": .3, "value": .4})
    write(path/"result.json", result)
    records, _ = summary.load_episodes(tmp_path, True)
    assert records[0]["work"]["root_rollout_count"] == 0
    assert records[0]["work"]["final_submit_count"] == 1
    assert records[0]["timings"]["nondegenerate"]["decisions"] == 0
    group = next(r for r in summary.aggregate_episodes(records) if r["scope"] == r["axis"] == "all")
    assert group["degenerate_decisions"] == 1 and group["model_load_seconds_sum"] == 1.
    assert group["normal_delay_weighted_mean_seconds"] is None
