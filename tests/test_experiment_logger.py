"""Tests for standard JSON/CSV rollout artifacts."""

import csv
import json

from terminal_marl import ExperimentMode, SimConfig, TerminalParallelEnv
from terminal_marl.experiments.logger import ExperimentLogger


def test_logger_writes_complete_artifact_set(tmp_path) -> None:
    env = TerminalParallelEnv(SimConfig(), experiment_mode=ExperimentMode.DYNAMIC_RULE)
    env.reset(seed=1)
    logger = ExperimentLogger(
        tmp_path,
        config_name="dynamic_rule",
        experiment_mode="dynamic_rule",
        policy="rule-only",
        seed=1,
        traffic_demand_level="medium",
    )
    logger.record_step(env)
    summary = logger.finalize({"throughput": 0.0})

    expected = {
        "episode_summary.json",
        "step_metrics.csv",
        "airspace_trace.csv",
        "safety_events.csv",
        "agent_events.csv",
    }
    assert expected == {path.name for path in tmp_path.iterdir()}
    loaded = json.loads((tmp_path / "episode_summary.json").read_text(encoding="utf-8"))
    assert loaded["config_name"] == "dynamic_rule"
    assert loaded["seed"] == 1
    assert loaded["metrics"]["throughput"] == 0.0
    assert summary["traffic_demand_level"] == "medium"


def test_step_csv_contains_required_operational_fields(tmp_path) -> None:
    env = TerminalParallelEnv(SimConfig(), experiment_mode=ExperimentMode.DYNAMIC_RULE)
    env.reset(seed=2)
    logger = ExperimentLogger(
        tmp_path,
        config_name="dynamic_rule",
        experiment_mode="dynamic_rule",
        policy="rule-only",
        seed=2,
        traffic_demand_level="low",
    )
    logger.record_step(env)
    logger.finalize({})
    with (tmp_path / "step_metrics.csv").open(encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert {
        "step",
        "landed_count",
        "active_count",
        "outer_occupancy",
        "inner_occupancy",
        "final_occupancy",
        "conflict_count",
        "safety_override_count",
    } <= set(row)
