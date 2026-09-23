"""Focused checks for the bounded dense-cohort energy diagnostic CLI."""

from __future__ import annotations

import json

import pytest

from terminal_marl.training.diagnose_energy_reachability_guard_smoke import (
    ARRIVAL_SCHEDULE,
    _config_summary,
    build_config,
    main,
    run_smoke,
)


def test_default_config_preserves_the_frozen_6_plus_6_scenario() -> None:
    config = build_config(
        initial_soc_min=0.41,
        initial_soc_max=0.55,
        route_guard_enabled=False,
        max_simulation_seconds=1200.0,
    )

    assert config.warm_start_uavs == 6
    assert config.total_requested_uavs == 12
    assert config.fixed_arrival_schedule == ARRIVAL_SCHEDULE
    summary = _config_summary(config, seed=840620, guard_on=False)
    assert summary["warm_start_uavs"] == 6
    assert summary["total_requested_uavs"] == 12
    assert summary["fixed_arrival_schedule"] == list(ARRIVAL_SCHEDULE)
    assert config.final_approach_geometry_mode == "brake_bezier_v1"
    assert summary["final_approach_geometry_mode"] == "brake_bezier_v1"


def test_continuous_final_approach_geometry_is_opt_in() -> None:
    config = build_config(
        initial_soc_min=0.55,
        initial_soc_max=0.90,
        route_guard_enabled=False,
        max_simulation_seconds=1.0,
        dense_warm_start_uavs=6,
        final_approach_geometry_mode="continuous_inward_v1",
    )

    assert config.final_approach_geometry_mode == "continuous_inward_v1"
    summary = _config_summary(config, seed=840620, guard_on=False)
    assert summary["final_approach_geometry_mode"] == "continuous_inward_v1"


def test_soc_priority_arbitration_cli_option_is_parsed_and_serialized(tmp_path) -> None:
    config = build_config(
        initial_soc_min=0.55,
        initial_soc_max=0.90,
        route_guard_enabled=False,
        max_simulation_seconds=1.0,
        dense_warm_start_uavs=6,
        release_arbitration_mode="downstream_soc_priority",
    )

    assert config.release_arbitration_mode == "downstream_soc_priority"
    summary = _config_summary(config, seed=840620, guard_on=False)
    assert summary["release_arbitration_mode"] == "downstream_soc_priority"

    output = tmp_path / "soc_priority.json"
    assert main(
        [
            "--seed", "840620",
            "--initial-soc-min", "0.55",
            "--initial-soc-max", "0.90",
            "--max-steps", "1",
            "--progress-every-steps", "1",
            "--dense-warm-start-uavs", "6",
            "--release-arbitration-mode", "downstream_soc_priority",
            "--output", str(output),
        ]
    ) == 0
    serialized = json.loads(output.read_text(encoding="utf-8"))
    assert serialized["config"]["release_arbitration_mode"] == "downstream_soc_priority"

    default = build_config(
        initial_soc_min=0.55,
        initial_soc_max=0.90,
        route_guard_enabled=False,
        max_simulation_seconds=1.0,
        dense_warm_start_uavs=6,
    )
    assert default.release_arbitration_mode == "downstream_priority"
    assert _config_summary(default, seed=840620, guard_on=False)["release_arbitration_mode"] == "downstream_priority"


def test_dense_config_uses_one_simultaneous_schedule_for_all_uavs() -> None:
    config = build_config(
        initial_soc_min=0.55,
        initial_soc_max=0.90,
        route_guard_enabled=False,
        max_simulation_seconds=1.0,
        dense_warm_start_uavs=30,
    )

    assert config.warm_start_uavs == 30
    assert config.total_requested_uavs == 30
    assert config.fixed_arrival_schedule == (0.0,) * 30
    summary = _config_summary(config, seed=840620, guard_on=False)
    assert summary["warm_start_uavs"] == 30
    assert summary["total_requested_uavs"] == 30
    assert summary["fixed_arrival_schedule"] == [0.0] * 30


def test_mixed_config_uses_warm_start_then_fixed_interval_arrivals() -> None:
    config = build_config(
        initial_soc_min=0.55,
        initial_soc_max=0.90,
        route_guard_enabled=False,
        max_simulation_seconds=2.0,
        mixed_total_uavs=30,
        mixed_warm_start_uavs=12,
        mixed_arrival_interval_seconds=8.0,
    )

    assert config.warm_start_uavs == 12
    assert config.total_requested_uavs == 30
    assert config.fixed_arrival_schedule == (
        (0.0,) * 12 + tuple(float(8 * index) for index in range(1, 19))
    )
    assert len(config.fixed_arrival_schedule) == 30


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"mixed_total_uavs": 0}, "mixed_total_uavs must be positive"),
        (
            {"mixed_total_uavs": 30, "mixed_warm_start_uavs": 31},
            "mixed_warm_start_uavs cannot exceed mixed_total_uavs",
        ),
        (
            {"mixed_total_uavs": 30, "mixed_arrival_interval_seconds": 0.0},
            "mixed_arrival_interval_seconds must be finite and positive",
        ),
    ),
)
def test_mixed_config_validates_counts_and_interval(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        build_config(
            initial_soc_min=0.55,
            initial_soc_max=0.90,
            route_guard_enabled=False,
            max_simulation_seconds=2.0,
            **kwargs,
        )


def test_mixed_options_are_inactive_without_mixed_total() -> None:
    config = build_config(
        initial_soc_min=0.41,
        initial_soc_max=0.55,
        route_guard_enabled=False,
        max_simulation_seconds=2.0,
        mixed_warm_start_uavs=3,
        mixed_arrival_interval_seconds=1.0,
    )

    assert config.warm_start_uavs == 6
    assert config.total_requested_uavs == 12
    assert config.fixed_arrival_schedule == ARRIVAL_SCHEDULE


def test_dense_one_step_serializes_accounting_and_progress(tmp_path) -> None:
    output = tmp_path / "dense.json"
    result = run_smoke(
        seed=840620,
        guard_on=False,
        initial_soc_min=0.55,
        initial_soc_max=0.90,
        max_steps=1,
        progress_every_steps=1,
        output=output,
        dense_warm_start_uavs=30,
    )

    progress = output.with_name(f"{output.name}.progress.json")
    assert output.exists()
    assert progress.exists()
    serialized = json.loads(output.read_text(encoding="utf-8"))
    progress_record = json.loads(progress.read_text(encoding="utf-8"))
    assert serialized["config"]["warm_start_uavs"] == 30
    assert serialized["config"]["total_requested_uavs"] == 30
    assert serialized["config"]["final_approach_geometry_mode"] == "brake_bezier_v1"
    assert serialized["config"]["fixed_arrival_schedule"] == [0.0] * 30
    assert serialized["request_accounting"]["requested"] == 30
    assert serialized["request_accounting"]["accounting_identity"] is True
    assert serialized["training_called"] is False
    assert serialized["teacher_used"] is False
    assert serialized["PPO_ALLOWED"] is False
    assert serialized["steps_executed"] == 1
    assert sum(serialized["release_decision_counts"].values()) == 30
    assert sum(serialized["release_primary_blocker_counts"].values()) <= 30
    assert sum(serialized["release_safety_reason_counts"].values()) <= 30
    assert "release_secondary_constraint_counts" in serialized
    assert isinstance(serialized["release_secondary_constraint_counts"], dict)
    assert sum(serialized["release_terminal_decision_counts"].values()) <= 30
    json.dumps(serialized, allow_nan=False)
    assert progress_record["step"] == 1
    assert progress_record["horizon_reached"] is True
    assert result["config"] == serialized["config"]
    before = output.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_smoke(
            seed=840620,
            guard_on=False,
            initial_soc_min=0.55,
            initial_soc_max=0.90,
            max_steps=1,
            progress_every_steps=1,
            output=output,
            dense_warm_start_uavs=30,
        )
    assert output.read_text(encoding="utf-8") == before


def test_dense_one_step_serializes_continuous_final_approach_mode(tmp_path) -> None:
    output = tmp_path / "dense_continuous.json"
    exit_code = main(
        [
            "--seed", "840620",
            "--initial-soc-min", "0.55",
            "--initial-soc-max", "0.90",
            "--max-steps", "1",
            "--progress-every-steps", "1",
            "--dense-warm-start-uavs", "6",
            "--final-approach-geometry-mode", "continuous_inward_v1",
            "--output", str(output),
        ]
    )

    assert exit_code == 0
    serialized = json.loads(output.read_text(encoding="utf-8"))
    assert serialized["steps_executed"] == 1
    assert serialized["config"]["warm_start_uavs"] == 6
    assert serialized["config"]["final_approach_geometry_mode"] == "continuous_inward_v1"


def test_mixed_one_step_serializes_schedule_and_summary(tmp_path) -> None:
    output = tmp_path / "mixed.json"
    exit_code = main(
        [
            "--seed", "840620",
            "--initial-soc-min", "0.55",
            "--initial-soc-max", "0.90",
            "--max-steps", "1",
            "--progress-every-steps", "1",
            "--mixed-total-uavs", "30",
            "--mixed-warm-start-uavs", "12",
            "--mixed-arrival-interval-seconds", "8",
            "--output", str(output),
        ]
    )

    assert exit_code == 0
    serialized = json.loads(output.read_text(encoding="utf-8"))
    assert serialized["steps_executed"] == 1
    # At t=1 s only the 12 warm-start entries have entered the ledger; the
    # remaining 18 requests are scheduled from t=8 s onward.
    assert serialized["request_accounting"]["requested"] == 12
    assert serialized["request_accounting"]["accounting_identity"] is True
    assert serialized["config"]["scenario_mode"] == "mixed_arrival"
    assert serialized["config"]["mixed_total_uavs"] == 30
    assert serialized["config"]["mixed_warm_start_uavs"] == 12
    assert serialized["config"]["mixed_arrival_interval_seconds"] == 8.0
    assert serialized["config"]["warm_start_uavs"] == 12
    assert serialized["config"]["total_requested_uavs"] == 30
    assert serialized["config"]["fixed_arrival_schedule"] == (
        [0.0] * 12 + [float(8 * index) for index in range(1, 19)]
    )
    assert serialized["PPO_ALLOWED"] is False
    json.dumps(serialized, allow_nan=False)
