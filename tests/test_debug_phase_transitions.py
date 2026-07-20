"""Smoke test for the phase-transition debug tool."""

from terminal_marl.experiments.debug_phase_transitions import run_debug_phase_transitions


def test_debug_phase_transitions_writes_expected_files(tmp_path) -> None:
    summary = run_debug_phase_transitions(
        config_name="dynamic_hrl",
        demand_level="medium",
        policy="rule-only",
        initial_uavs=1,
        disable_runtime_arrivals=True,
        high_level_steps=3,
        output_dir=tmp_path,
        seed=0,
    )

    assert summary["metrics"]["total_spawned_count"] >= 1.0
    assert (tmp_path / "phase_trace.csv").is_file()
    assert (tmp_path / "release_attempts.csv").is_file()
    assert (tmp_path / "trajectory_diagnostics.csv").is_file()
    assert (tmp_path / "dynamic_rule_trace.csv").is_file()
    assert (tmp_path / "phase_release_summary.csv").is_file()
    assert (tmp_path / "summary.json").is_file()
