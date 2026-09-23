"""Contracts for the UID-specific dense30 energy forensic diagnostic."""

from __future__ import annotations

import json

from terminal_marl.training.diagnose_dense30_failed_energy_trace import run_trace
from terminal_marl.training.diagnose_energy_reachability_guard_smoke import run_smoke


def test_one_step_trace_is_serializable_and_matches_uninstrumented_outcome(tmp_path) -> None:
    traced_path = tmp_path / "traced.json"
    plain_path = tmp_path / "plain.json"
    traced = run_trace(
        seed=840626,
        max_steps=1,
        progress_every_steps=1,
        output=traced_path,
        compare_expected=False,
    )
    plain = run_smoke(
        seed=840626,
        guard_on=True,
        initial_soc_min=0.55,
        initial_soc_max=0.90,
        max_steps=1,
        progress_every_steps=1,
        output=plain_path,
        dense_warm_start_uavs=30,
        final_approach_geometry_mode="continuous_inward_v1",
    )

    assert traced["steps_executed"] == plain["steps_executed"] == 1
    assert traced["outcome_counts"] == plain["outcome_counts"]
    assert traced["safety_error"] == plain["safety_error"]
    assert traced["comparison"]["matches"] is None
    assert set(traced["target_records"]) == {"13", "14"}
    assert all(traced["target_records"][uid] for uid in ("13", "14"))
    assert len(traced["emergency_attempts_by_target"]) == 2
    assert {row["uid"] for row in traced["emergency_attempts_by_target"]} == {13, 14}
    assert all(row["result"] is False for row in traced["emergency_attempts_by_target"])
    assert traced["emergency_plan_commits_by_target"] == []
    json.dumps(traced, allow_nan=False)
    json.dumps(json.loads(traced_path.read_text(encoding="utf-8")), allow_nan=False)
    assert (tmp_path / "traced.json.progress.json").exists()


def test_two_step_mixed_trace_attaches_late_target_and_serializes(tmp_path) -> None:
    traced_path = tmp_path / "mixed_traced.json"
    traced = run_trace(
        seed=840627,
        max_steps=2,
        progress_every_steps=1,
        output=traced_path,
        compare_expected=False,
        mixed_total_uavs=30,
        mixed_warm_start_uavs=12,
        mixed_arrival_interval_seconds=8.0,
        target_uids=(44,),
    )

    assert traced["steps_executed"] == 2
    assert traced["comparison"]["matches"] is None
    assert traced["config"]["mixed_total_uavs"] == 30
    assert traced["config"]["mixed_warm_start_uavs"] == 12
    assert traced["config"]["mixed_arrival_interval_seconds"] == 8.0
    assert traced["config"]["target_uids"] == [44]
    assert set(traced["target_records"]) == {"44"}
    assert traced["target_records"]["44"] == []
    assert all(value <= 6 for value in traced["scan_counts"].values())
    json.dumps(traced, allow_nan=False)
    json.dumps(json.loads(traced_path.read_text(encoding="utf-8")), allow_nan=False)
    assert (tmp_path / "mixed_traced.json.progress.json").exists()
