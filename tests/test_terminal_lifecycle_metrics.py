from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from terminal_marl.core.uav import MissionType
from terminal_marl.training import evaluate_p3_adaptive_kl_v7 as evaluator
from terminal_marl.training import terminal_lifecycle_metrics as metrics


class _Value:
    def __init__(self, value: str):
        self.value = value


class _UAV:
    def __init__(
        self,
        arrival_id: int,
        *,
        priority: int,
        initial_soc: float,
        outcome: str,
        touchdown: float | None,
        completion: float | None,
        phase: str,
        emergency: bool = False,
        mission_type: MissionType | None = MissionType.STANDARD,
    ):
        self.arrival_id = arrival_id
        self.uid = 100 + arrival_id
        self.base_priority = priority
        self.request_time = float(arrival_id)
        self.energy_capacity = 1500.0
        self.energy_remaining = 1500.0 * initial_soc
        self.energy_consumed = 0.0
        self.phase = _Value(phase)
        self.external_wait_time = 2.0 + arrival_id
        self.internal_wait_time = 3.0 + arrival_id
        self.touchdown_time = touchdown
        self.completion_time = completion
        self.terminal_outcome = _Value(outcome)
        self.terminal_reason = "ground_service_complete"
        self.emergency_landing = emergency
        self.mission_type = mission_type

    @property
    def soc(self) -> float:
        return max(0.0, self.energy_remaining / self.energy_capacity)


class _Ledger:
    def __init__(self, requested: set[int]):
        self.requested = set(requested)
        self.outcomes: dict[int, _Value] = {}


class _Env:
    def __init__(self, config):
        self.config = config
        self.now = 0.0
        self.active: dict[int, _UAV] = {}
        self.service_uavs: dict[int, _UAV] = {}
        self.completed_uavs: list[_UAV] = []
        self.ledger = _Ledger(set(range(int(config.total_requested_uavs))))

    def step(self, *_args, **_kwargs):
        return None

    def _resolve(self, uav, outcome, reason):
        self.ledger.outcomes[int(uav.arrival_id)] = outcome
        uav.terminal_outcome = outcome
        uav.terminal_reason = reason
        self.active.pop(int(uav.uid), None)


def _config(total: int = 3) -> SimpleNamespace:
    return SimpleNamespace(total_requested_uavs=total)


def _fake_replay(env: _Env, _path: str | Path):
    normal = _UAV(
        0, priority=1, initial_soc=.8, outcome="completed", touchdown=8.0,
        completion=10.0, phase="completed",
    )
    emergency = _UAV(
        1, priority=3, initial_soc=.6, outcome="emergency_landed", touchdown=9.0,
        completion=11.0, phase="completed", emergency=True,
    )
    failed = _UAV(
        2, priority=5, initial_soc=.55, outcome="failed_energy", touchdown=None,
        completion=None, phase="ring_outer",
    )
    failed.energy_remaining = 0.0
    failed.energy_consumed = 825.0
    env.active = {uav.uid: uav for uav in (normal, emergency, failed)}
    env.step({})
    env.completed_uavs = [normal, emergency]
    env.now = 30.0
    env._resolve(failed, _Value("failed_energy"), "energy_depleted")
    env.ledger.outcomes[0] = _Value("completed")
    env.ledger.outcomes[1] = _Value("emergency_landed")
    return {"exact": True, "steps_expected": 1, "steps_replayed": 1}


def test_exact_replay_collects_normal_emergency_and_energy_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metrics, "TerminalParallelEnv", _Env)
    monkeypatch.setattr(metrics.SafetyTraceRecorder, "replay_action_tape", _fake_replay)
    result = metrics.terminal_lifecycle_metrics(_config(), "tape.json", .65)

    assert result["replay_exact"] is True
    assert result["steps_expected"] == result["steps_replayed"] == 1
    assert result["row_count"] == 3
    assert result["observed_terminal_count"] == 3
    assert result["requested_denominator"] == 3
    assert result["missing_arrival_ids"] == []
    assert result["outcomes"] == {"completed": 1, "emergency_landed": 1, "failed_energy": 1}
    assert result["priority_weighted_wait_seconds"] == pytest.approx(71.0 / 9.0)
    assert result["low_soc"]["count"] == 2
    assert result["low_soc"]["landing_soc_observed_count"] == 1
    assert result["low_soc"]["mean_landing_soc"] == pytest.approx(.6)
    assert result["low_soc"]["terminal_soc_observed_count"] == 2
    assert result["low_soc"]["mean_terminal_soc"] == pytest.approx(.3)
    assert result["success_only"]["row_count"] == 2
    failed = next(row for row in result["rows"] if row["arrival_id"] == 2)
    assert failed["initial_soc"] == pytest.approx(.55)
    assert failed["touchdown_time"] is None
    assert failed["terminal_time"] == pytest.approx(30.0)
    assert result["failure_terminal_time_not_touchdown"] is True
    assert result["ledger_alignment"] is True


def test_unobserved_request_has_explicit_missing_denominator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metrics, "TerminalParallelEnv", _Env)
    monkeypatch.setattr(metrics.SafetyTraceRecorder, "replay_action_tape", _fake_replay)
    result = metrics.terminal_lifecycle_metrics(_config(total=4), "tape.json", .65)

    assert result["requested_denominator"] == 4
    assert result["observed_terminal_count"] == 3
    assert result["missing_arrival_ids"] == [3]
    assert result["missing_request_count"] == 1
    assert result["all_requests_observed"] is False
    assert result["observed_metrics_denominator"] == 3
    assert result["ledger_alignment"] is False
    assert all(row["arrival_id"] != 3 for row in result["rows"])


def test_mission_type_groups_are_separate_from_priority_and_outcome() -> None:
    result = metrics.summarize_terminal_rows(
        [
            {
                "arrival_id": 0,
                "base_priority": 5,
                "mission_type": MissionType.STANDARD,
                "wait_seconds": 2.0,
                "outcome": "completed",
            },
            {
                "arrival_id": 1,
                "base_priority": 5,
                "mission_type": MissionType.EMERGENCY,
                "wait_seconds": 4.0,
                "outcome": "completed",
            },
            {
                "arrival_id": 2,
                "base_priority": 1,
                "mission_type": MissionType.STANDARD,
                "wait_seconds": 6.0,
                "outcome": "emergency_landed",
                "emergency_landing": True,
            },
        ],
        low_soc_threshold=.65,
    )

    assert result["by_priority"]["5"]["count"] == 2
    assert result["by_mission_type"]["standard"] == {
        "terminal_count": 2,
        "count": 2,
        "mean_wait_seconds": 4.0,
        "outcomes": {"completed": 1, "emergency_landed": 1},
    }
    assert result["by_mission_type"]["emergency"]["outcomes"] == {"completed": 1}
    assert result["by_mission_type"]["emergency"]["mean_wait_seconds"] == 4.0


def test_missing_or_unknown_mission_type_is_reported_separately() -> None:
    result = metrics.summarize_terminal_rows(
        [
            {
                "arrival_id": 0,
                "base_priority": 5,
                "wait_seconds": 2.0,
                "outcome": "completed",
            },
            {
                "arrival_id": 1,
                "base_priority": 1,
                "mission_type": "unknown_mission",
                "wait_seconds": 4.0,
                "outcome": "completed",
            },
            {
                "arrival_id": 2,
                "base_priority": 3,
                "mission_type": MissionType.STANDARD,
                "wait_seconds": 6.0,
                "outcome": "completed",
            },
        ],
        low_soc_threshold=.65,
    )

    assert result["missing_mission_type_count"] == 2
    assert set(result["by_mission_type"]) == {"standard"}
    assert result["by_mission_type"]["standard"]["count"] == 1


def test_row_serializes_mission_type_value_and_missing_as_none() -> None:
    uav = _UAV(
        0,
        priority=5,
        initial_soc=.8,
        outcome="completed",
        touchdown=8.0,
        completion=10.0,
        phase="completed",
        mission_type=MissionType.EMERGENCY,
    )
    row = metrics._row_from_uav(
        uav,
        first_seen={},
        terminal_time=10.0,
        outcome="completed",
        reason="ground_service_complete",
        source="test",
    )
    assert row["mission_type"] == "emergency"

    del uav.mission_type
    missing_row = metrics._row_from_uav(
        uav,
        first_seen={},
        terminal_time=10.0,
        outcome="completed",
        reason="ground_service_complete",
        source="test",
    )
    assert missing_row["mission_type"] is None


def test_duplicate_rows_and_nonexact_replay_are_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    row = {
        "arrival_id": 0,
        "base_priority": 1,
        "initial_soc": .8,
        "wait_seconds": 1.0,
        "outcome": "completed",
    }
    with pytest.raises(metrics.LifecycleMetricsError, match="unique"):
        metrics.summarize_terminal_rows([row, dict(row)], low_soc_threshold=.65)

    monkeypatch.setattr(metrics, "TerminalParallelEnv", _Env)
    monkeypatch.setattr(
        metrics.SafetyTraceRecorder,
        "replay_action_tape",
        lambda *_args, **_kwargs: {"exact": False, "steps_expected": 1, "steps_replayed": 1},
    )
    with pytest.raises(metrics.LifecycleMetricsError, match="not exact"):
        metrics.terminal_lifecycle_metrics(_config(), "tape.json", .65)


def test_ledger_outcome_mismatch_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metrics, "TerminalParallelEnv", _Env)

    def mismatched(env, path):
        result = _fake_replay(env, path)
        env.ledger.outcomes[0] = _Value("failed_energy")
        return result

    monkeypatch.setattr(metrics.SafetyTraceRecorder, "replay_action_tape", mismatched)
    with pytest.raises(metrics.LifecycleMetricsError, match="ledger outcome mismatch"):
        metrics.terminal_lifecycle_metrics(_config(), "tape.json", .65)


def test_evaluator_preserves_success_only_and_uses_full_lifecycle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tape = tmp_path / "tape.json"
    tape.write_text("{}", encoding="utf-8")
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = SimpleNamespace(
        total_requested_uavs=1,
        max_simulation_seconds=2.0,
        dt=1.0,
        teacher_enabled=False,
    )
    old_lifecycle = {
        "replay_exact": True,
        "rows": [{"uid": 1, "base_priority": 3, "initial_soc": .7,
                  "landing_soc": .6, "wait_seconds": 4.0,
                  "request_to_touchdown_seconds": 5.0}],
        "priority_weighted_wait_seconds": 4.0,
    }
    raw = {
        "label": "source_03c0", "scenario": "flow6", "seed": 1,
        "status": "completed", "requests": 1, "normal": 1, "emergency": 0,
        "failed_energy": 0, "unresolved": 0, "other_failures": {},
        "summary": {"terminal": True}, "action_tape": str(tape),
        "lifecycle": old_lifecycle, "gates": {"safety": True},
        "weights_unchanged": True, "teacher_enabled": False,
        "PPO_ALLOWED": False, "label_rows": 0, "_steps": [],
    }
    full = {
        "scope": "all_requested_terminal_rows",
        "rows": [{"arrival_id": 0, "base_priority": 1, "initial_soc": .7,
                  "wait_seconds": 4.0, "outcome": "completed",
                  "final_soc": .6, "request_time": 0.0,
                  "terminal_time": 5.0, "touchdown_time": 5.0}],
        "row_count": 1,
        "priority_weighted_wait_seconds": 4.0,
    }
    monkeypatch.setattr(evaluator.admission, "_record_rollout", lambda **_kwargs: raw)
    monkeypatch.setattr(evaluator.terminal_metrics, "terminal_lifecycle_metrics", lambda *args: full)
    monkeypatch.setattr(evaluator.v6, "_dev_pass", lambda _record: True)
    result = evaluator._episode_record(
        policy=SimpleNamespace(), label="source_03c0", checkpoint=checkpoint,
        config=config, protocol={}, runtime_protocol={"attributes": {"low_soc_analysis_threshold": .65}},
        scenario="flow6", seed=1, output=tmp_path / "rollout", max_steps=2,
        require_full_resolution=False,
    )
    assert result["normal"] == 1
    assert result["gates"] == {"safety": True}
    assert result["success_lifecycle"] == old_lifecycle
    assert result["lifecycle"] == full
    assert result["lifecycle_scope"] == "all_requested_terminal_rows"


def test_evaluator_aggregate_does_not_merge_arrival_ids_across_episodes() -> None:
    row = {
        "arrival_id": 0, "base_priority": 1, "initial_soc": .7,
        "wait_seconds": 2.0, "outcome": "completed", "final_soc": .6,
        "request_time": 0.0, "terminal_time": 3.0, "touchdown_time": 3.0,
    }
    records = [
        {"requests": 1, "lifecycle": {"scope": "all_requested_terminal_rows", "rows": [row], "missing_request_count": 0}},
        {"requests": 1, "lifecycle": {"scope": "all_requested_terminal_rows", "rows": [dict(row, wait_seconds=4.0)], "missing_request_count": 0}},
    ]
    summary = evaluator._lifecycle_summary(records, threshold=.65)
    assert summary["scope"] == "all_requested_terminal_rows_aggregate"
    assert summary["requested_denominator"] == 2
    assert summary["row_count"] == 2
    assert summary["observed_wait_count"] == 2


def test_evaluator_aggregate_counts_legacy_and_missing_episodes_as_missing() -> None:
    full_row = {
        "arrival_id": 0, "base_priority": 1, "initial_soc": .7,
        "wait_seconds": 2.0, "outcome": "completed", "final_soc": .6,
        "request_time": 0.0, "terminal_time": 3.0, "touchdown_time": 3.0,
    }
    legacy_row = {
        "uid": 7, "base_priority": 3, "initial_soc": .7,
        "landing_soc": .6, "wait_seconds": 4.0,
        "request_to_touchdown_seconds": 5.0,
    }
    summary = evaluator._lifecycle_summary([
        {
            "requests": 2,
            "lifecycle": {
                "scope": "all_requested_terminal_rows",
                "rows": [full_row],
                "missing_request_count": 0,
            },
        },
        {"requests": 3, "lifecycle": {"rows": [legacy_row]}},
        {"requests": 4, "lifecycle": None},
    ], threshold=.65)
    assert summary["scope"] == "mixed_incomplete"
    assert summary["requested_denominator"] == 9
    assert summary["missing_request_count"] == 8
    assert summary["incomplete_metrics_episodes"] == 2
    assert summary["legacy_rows_excluded_from_observed_metrics"] == 1
    assert summary["all_requests_observed"] is False
    assert summary["row_count"] == 1


def test_evaluator_aggregate_zero_full_rows_keeps_missing_denominator() -> None:
    summary = evaluator._lifecycle_summary([
        {
            "requests": 3,
            "lifecycle": {
                "scope": "all_requested_terminal_rows",
                "rows": [],
                "missing_request_count": 0,
            },
        },
    ], threshold=.65)
    assert summary["available"] is True
    assert summary["scope"] == "all_requested_terminal_rows_aggregate"
    assert summary["row_count"] == 0
    assert summary["requested_denominator"] == 3
    assert summary["missing_request_count"] == 3
    assert summary["all_requests_observed"] is False
