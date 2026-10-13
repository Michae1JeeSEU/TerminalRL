import json
from dataclasses import asdict, replace
from terminal_marl.configs.fixed_terminal_stage1 import DEFAULT_CONFIG
from terminal_marl.configs.fixed_terminal_stage2 import METHODS, DEFAULT_SCHEDULER_CONFIG
from terminal_marl.configs.fixed_terminal_stage3 import Context, FORMAL_CONTEXTS, REFERENCE_SETTINGS
from terminal_marl.core.fixed_terminal.reference import make_root_pool, Root, select_observed_roots, value_comparison_key
from terminal_marl.core.fixed_terminal.value_analysis import replay_episode
from terminal_marl.scripts.evaluate_fixed_terminal_stage2 import run
from terminal_marl.scripts.analyze_fixed_terminal_stage3 import run_context


def test_root_pool_controls_expanded_budget_and_all_output_parse(tmp_path):
    sources = tmp_path/"stage2"
    config = replace(DEFAULT_CONFIG, horizon=2)
    for method in METHODS:
        run(sources/f"{method}-low-medium-101", method, "low", "medium", 101, config=config)
    env = replay_episode(sources/"fcfs-low-medium-101", 0)
    context = Context("low", "medium", 101)
    roots, metadata = make_root_pool(env, context, sources)
    labels = {label for root in roots for label in root.labels}
    assert set(METHODS) | {"HOLD"} <= labels
    assert len(roots) <= 19
    assert metadata["reference_settings"]["joint_aircraft_limit"] == 8
    assert metadata["reference_settings"]["beam_width"] == 128
    assert all(p["source"] != "preexisting_stage2_root" or "original_decision_seconds" in p for root in roots for p in root.provenance)
    result = run_context(tmp_path/"analysis", context, sources)
    assert result["observed_selections"]["until240"]["no_worse_than_any_evaluated_root_in_this_pool"]
    for path in (tmp_path/"analysis").rglob("*.json"):
        assert json.loads(path.read_text()) is not None
    for line in (tmp_path/"analysis"/"decision_database.jsonl").read_text().splitlines():
        row = json.loads(line)
        assert row["realized_values"] and row["observed_values"]
        assert row["delta_observed_vs_HOLD"] and row["delta_observed_vs_FCFS"]
    assert result["learning_or_teacher_training"] is False
    assert "serialization_and_other_seconds" in result["timings"]


def test_contexts_and_budget_changes_are_exactly_scoped():
    assert len(FORMAL_CONTEXTS) == len({c.key for c in FORMAL_CONTEXTS}) == 15
    assert sum(c.time == 60 for c in FORMAL_CONTEXTS) == 3
    changes = {key for key, value in asdict(REFERENCE_SETTINGS).items() if value != asdict(DEFAULT_SCHEDULER_CONFIG)[key]}
    assert changes == {"joint_aircraft_limit", "catalogue_scan_limit", "candidates_per_aircraft", "beam_width"}


def test_near_zero_float_residue_does_not_change_root_selection():
    from terminal_marl.core.fixed_terminal.scenarios import Scenario, Request
    from terminal_marl.core.fixed_terminal.routes import build_plan
    from terminal_marl.envs.fixed_terminal_env import FixedTerminalEnv
    env = FixedTerminalEnv(replace(DEFAULT_CONFIG, horizon=2), Scenario(1, "test", "test", (Request("a", 0, .8, 0, 0),), 1, 2))
    hold = Root("hold", (), ["HOLD"], [], {"status": "feasible"})
    plan = Root("plan", (build_plan(env.aircraft["a"], 0, env.config),), [], [], {"status": "feasible"})
    baseline = {"J": (0, 0, 0, -1, -1., 0., -.1)}
    residue = {"J": (0, 0, 0, -1, -1.+1e-12, 0., -.1+1e-12)}
    assert value_comparison_key(baseline) == value_comparison_key(residue)
    evaluations = {"hold": {"status": "feasible", "values": {"2": baseline}},
                   "plan": {"status": "feasible", "values": {"2": residue}}}
    chosen = select_observed_roots(env, [hold, plan], evaluations)
    assert all(row["root_id"] == "hold" and row["no_worse_than_any_evaluated_root_in_this_pool"] for row in chosen.values())
