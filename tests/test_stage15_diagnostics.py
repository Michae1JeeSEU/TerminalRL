import numpy as np

from terminal_marl.experiments.evaluation_protocol import (
    WindowedEvaluationProtocol,
    run_windowed_evaluation,
)
from terminal_marl.experiments.stage15_diagnostics import (
    generate_common_schedules,
    summarize_action_saturation,
    summarize_normalized_metric_saturation,
)


def test_common_fixed_schedule_makes_request_count_policy_independent(tmp_path) -> None:
    protocol = WindowedEvaluationProtocol(
        warm_up_seconds=10.0,
        measurement_seconds=20.0,
        max_drain_seconds=10.0,
        metrics_sampling_interval_seconds=10.0,
        decision_interval_seconds=10.0,
        reconfiguration_dwell_seconds=20.0,
    )
    schedules = generate_common_schedules(
        config_name="event_gated_hrl",
        output_dir=tmp_path,
        protocol=protocol,
        seeds=(20,),
        demands=("medium",),
    )
    schedule = schedules[("medium", 20)]
    rule = run_windowed_evaluation(
        "fixed_rule",
        protocol=protocol,
        policy="rule-only",
        demand_level="medium",
        seed=20,
        fixed_arrival_schedule=schedule,
    )["summary"]
    event_rule = run_windowed_evaluation(
        "event_rule",
        protocol=protocol,
        policy="event-rule",
        demand_level="medium",
        seed=20,
        fixed_arrival_schedule=schedule,
    )["summary"]
    assert rule["arrival_request_count"] == event_rule["arrival_request_count"]
    assert rule["arrival_request_count"] == float(len(schedule))


def test_stage15_summary_helpers_report_saturation_fields() -> None:
    samples = []
    for value in np.linspace(-1.0, 1.0, 5):
        row = {
            "policy_label": "event-gated PPO20k",
            "traffic_demand_level": "high",
            "window": "measurement",
            "normalized_congestion": 0.99,
            "normalized_waiting": 0.96,
            "normalized_backlog": 0.5,
        }
        for dim in range(8):
            row[f"requested_action_dim_{dim}"] = float(value)
            row[f"applied_action_dim_{dim}"] = float(value if dim >= 4 else 0.0)
        samples.append(row)
    action_rows = summarize_action_saturation(samples)
    metric_rows = summarize_normalized_metric_saturation(samples)
    assert any(row["dim"] == 0 and row["action_kind"] == "applied" for row in action_rows)
    congestion = next(row for row in metric_rows if row["metric"] == "normalized_congestion")
    assert congestion["reward_state_saturation"] == 1.0
