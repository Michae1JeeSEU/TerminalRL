"""Configuration and CLI contracts for the explicit dense30 viewer scenes."""

from __future__ import annotations

import pytest

from terminal_marl.visualization.rollout_viewer import _config_for_mode, _parser


def test_dense30_viewer_scenes_reuse_frozen_config_and_only_toggle_coordinator() -> None:
    baseline = _config_for_mode(
        "fixed_rule", 1800.0, "fixed_rule_dense30_baseline"
    )
    coordinated = _config_for_mode(
        "fixed_rule", 1800.0, "fixed_rule_dense30_coordinated"
    )

    for config in (baseline, coordinated):
        assert config.experiment_mode == "fixed_rule"
        assert config.high_level_enabled is False
        assert config.reset_mode == "warm_start_rings"
        assert config.warm_start_uavs == 30
        assert config.warm_start_steady_motion is True
        assert config.total_requested_uavs == 30
        assert config.fixed_arrival_schedule == (0.0,) * 30
        assert config.initial_soc_min == pytest.approx(0.55)
        assert config.initial_soc_max == pytest.approx(0.90)
        assert config.final_approach_geometry_mode == "continuous_inward_v1"
        assert config.release_arbitration_mode == "downstream_priority"
        assert config.reconcile_late_merge_slots_enabled is False

    assert baseline.rule_flow_coordination_enabled is False
    assert coordinated.rule_flow_coordination_enabled is True


@pytest.mark.parametrize(
    "scenario",
    ("fixed_rule_dense30_baseline", "fixed_rule_dense30_coordinated"),
)
def test_dense30_scenes_are_fixed_rule_only(scenario: str) -> None:
    with pytest.raises(ValueError, match="only with fixed_rule"):
        _config_for_mode("dynamic_rule", 60.0, scenario)


def test_existing_default_and_warm_flow_scenes_remain_unchanged() -> None:
    default = _config_for_mode("fixed_rule", 60.0, "default")
    warm = _config_for_mode("fixed_rule", 60.0, "fixed_rule_warm_flow")

    assert default.rule_flow_coordination_enabled is False
    assert warm.rule_flow_coordination_enabled is False
    assert warm.warm_start_uavs == 12
    assert warm.total_requested_uavs == len(warm.fixed_arrival_schedule)


def test_parser_exposes_dense30_scenario_choices() -> None:
    parser = _parser()
    args = parser.parse_args(
        [
            "--mode",
            "fixed_rule",
            "--scenario",
            "fixed_rule_dense30_coordinated",
        ]
    )
    assert args.scenario == "fixed_rule_dense30_coordinated"
