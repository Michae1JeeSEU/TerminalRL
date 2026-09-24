"""Small contract checks for the landing-flow comparison diagnostic."""

from __future__ import annotations

import json

import pytest

from terminal_marl.core.runtime_traffic_safety import SafetyInvariantError
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training.diagnose_landing_flow import (
    _terminal,
    run_diagnostic,
)


def test_one_step_output_is_serialized_and_bounded(tmp_path):
    output = tmp_path / "landing_flow_one_step.json"
    result = run_diagnostic(seed=841301, coordinated=False, max_steps=1, output=output)

    assert result["steps"] == 1
    assert result["stop_reason"] == "max_steps"
    assert result["config"]["rule_flow_coordination_enabled"] is False
    assert "telemetry" in result
    assert "safety_summary" in result
    assert "energy_outcomes" in result
    assert "uid3_uid4" in result
    assert "normal_commits" in result
    assert "emergency_commits" in result
    assert "touchdowns" in result
    assert "candidate_rejection_counts" in result
    assert "release_row_reason_counts" in result
    for key in (
        "minimum_actual_separation",
        "minimum_swept_separation",
        "shield_fail_closed",
        "joint_safety_fail_closed",
        "checked_executed_plan_hash_mismatches",
        "requested",
        "resolved",
        "unresolved",
    ):
        assert key in result["safety_summary"]
    assert output.exists()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["steps"] == 1
    assert output.stat().st_size < 500 * 1024
    progress = tmp_path / "landing_flow_one_step.json.progress.json"
    assert progress.exists()
    progress_payload = json.loads(progress.read_text(encoding="utf-8"))
    assert progress_payload["step"] == 1
    assert isinstance(progress_payload["outcome_counts"], dict)
    assert isinstance(progress_payload["phases"], dict)
    assert isinstance(progress_payload["normal_commits"], int)
    assert isinstance(progress_payload["emergency_commits"], int)
    assert "minimum_active_soc" in progress_payload
    assert progress_payload["minimum_active_soc"] is None or isinstance(
        progress_payload["minimum_active_soc"], float
    )
    for record in result["normal_commits"] + result["emergency_commits"]:
        assert "tangential_speed_bias_mps" in record
        assert record["tangential_speed_bias_mps"] is None or isinstance(
            record["tangential_speed_bias_mps"], float
        )


def test_output_and_progress_paths_are_never_overwritten(tmp_path):
    output = tmp_path / "landing_flow_once.json"
    run_diagnostic(seed=841301, coordinated=True, max_steps=1, output=output)
    with pytest.raises(FileExistsError, match="overwrite"):
        run_diagnostic(seed=841301, coordinated=True, max_steps=1, output=output)


def test_safety_error_diagnostic_and_runtime_fail_closed_are_serialized(monkeypatch, tmp_path):
    diagnostic = {
        "unsafe_pairs": ((3, 4),),
        "initial_minimum_separation": 79.5,
        "recursive_gate": "backup_existence",
    }

    def raise_safety_error(self, actions):
        self.runtime_traffic.telemetry["shield_fail_closed"] = 1
        self.runtime_traffic.telemetry["joint_safety_fail_closed"] = 1
        self.last_release_telemetry = {3: {"decision": "safety_wait"}, 4: {"decision": "approved"}}
        raise SafetyInvariantError("synthetic safety barrier", diagnostic=diagnostic)

    monkeypatch.setattr(TerminalParallelEnv, "step", raise_safety_error)
    output = tmp_path / "landing_flow_safety_error.json"
    result = run_diagnostic(seed=841301, coordinated=True, max_steps=1, output=output)

    assert result["stop_reason"] == "safety_error"
    assert result["safety_error_diagnostic"]["unsafe_pairs"] == [[3, 4]]
    assert result["safety_error_diagnostic"]["recursive_gate"] == "backup_existence"
    assert result["safety_error_pair_context"]["unsafe_pairs"] == [[3, 4]]
    assert result["safety_error_pair_context"]["minimum_predicted_separation"] == 79.5
    assert result["safety_error_pair_context"]["recursive_gate"] == "backup_existence"
    assert result["safety_error_pair_context"]["last_release_telemetry"] == {
        "3": {"decision": "safety_wait"},
        "4": {"decision": "approved"},
    }
    assert [row["uid"] for row in result["safety_error_pair_context"]["pairs"][0]["uavs"]] == [3, 4]
    assert all("phase" in row and "position" in row and "speed" in row for row in result["safety_error_pair_context"]["pairs"][0]["uavs"])
    assert result["safety_summary"]["shield_fail_closed"] == 1
    assert result["safety_summary"]["joint_safety_fail_closed"] == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["safety_error_diagnostic"]["initial_minimum_separation"] == 79.5


def test_terminal_requires_queue_and_service_to_be_empty():
    class Arrivals:
        def __init__(self, exhausted):
            self._exhausted = exhausted

        def exhausted(self):
            return self._exhausted

    class Admission:
        class Queue:
            items = []

        queue = Queue()

    class Env:
        arrivals = Arrivals(True)
        active = {}
        admission = Admission()
        service_uavs = {}

    env = Env()
    assert _terminal(env)
    env.service_uavs = {7: object()}
    assert not _terminal(env)
