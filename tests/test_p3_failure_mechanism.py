from __future__ import annotations

import json
from pathlib import Path

import pytest

from terminal_marl.training import run_p3_failure_mechanism_20260930 as mechanism


def _candidate(step: int) -> dict[str, object]:
    return {
        "episode": 24,
        "step": step,
        "uid": mechanism.TARGET_UID,
        "arrival_id": mechanism.TARGET_ARRIVAL_ID,
        "first_actions": {"57": [1.0, 0.0, -0.5]},
        "runtime_proposal_shortlists": {"57": [[1.0, 0.0, -0.5], [1.0, 0.0, 0.0], [1.0, 1.0, -1.0]]},
    }


def test_quantile_selection_is_sorted_floor_and_unique() -> None:
    result = mechanism._select_points([_candidate(step) for step in (50, 10, 40, 20, 30)])
    assert [item["step"] for item in result["selected"]] == [20, 30, 40]
    assert [item["eligible_index"] for item in result["selected"]] == [1, 2, 3]
    assert len({item["step"] for item in result["selected"]}) == 3


def test_quantile_selection_rejects_nonunique_registered_indices() -> None:
    with pytest.raises(mechanism.FailureMechanismError, match="not unique"):
        mechanism._select_points([_candidate(step) for step in (1, 2, 3)])


def test_eligible_points_require_target_mask_and_arrival() -> None:
    rows = [
        {"step": 0, "a_requested": {"57": [1.0, 0.0, -0.5]}, "runtime_proposal_shortlists": {"57": []}},
        {"step": 1, "a_requested": {"57": [1.0, 0.0, -0.5]}, "runtime_proposal_shortlists": {"57": []}},
        {"step": 2, "a_requested": {"57": [1.0, 0.0, -0.5]}, "runtime_proposal_shortlists": {"57": []}},
    ]
    captures = [
        {"step": 0, "state_context": {"failure_target": {"uid": 57, "arrival_id": 14, "active": True, "actor_mask": False, "world_index": 0}}},
        {"step": 1, "state_context": {"failure_target": {"uid": 57, "arrival_id": 13, "active": True, "actor_mask": True, "world_index": 0}}},
        {"step": 2, "state_context": {"failure_target": {"uid": 57, "arrival_id": 14, "active": True, "actor_mask": True, "phase": "ring_inner", "soc": 0.42, "world_index": 0}}},
    ]
    result = mechanism._eligible_points(captures, rows)
    assert [item["step"] for item in result] == [2]
    assert result[0]["phase"] == "ring_inner"
    assert result[0]["soc"] == 0.42


def test_lifecycle_keeps_failed_energy_row() -> None:
    request_rows = [{"arrival_id": index} for index in range(24)]
    terminal_rows = [{"arrival_id": index, "outcome": "completed", "terminal_time": 10, "uid": index} for index in range(24)]
    terminal_rows[mechanism.TARGET_ARRIVAL_ID] = {
        "arrival_id": mechanism.TARGET_ARRIVAL_ID,
        "uid": mechanism.TARGET_UID,
        "outcome": "failed_energy",
        "terminal_time": mechanism.TARGET_FAILURE_STEP,
    }
    result = mechanism._validate_lifecycle({
        "terminal": True,
        "all_requests_observed": True,
        "ledger_alignment": True,
        "request_rows": request_rows,
        "rows": terminal_rows,
    })
    assert result["request_count"] == mechanism.EXPECTED_REQUESTS
    assert len(result["failed_energy_rows"]) == 1
    assert result["terminal_rows"][mechanism.TARGET_ARRIVAL_ID]["outcome"] == "failed_energy"
    assert result["all_rows_retained"] is True


def test_preflight_dispatch_does_not_simulate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fail_replay(*args: object, **kwargs: object) -> None:
        raise AssertionError("preflight invoked replay")

    monkeypatch.setattr(mechanism.credit, "_capture_replay", fail_replay)
    monkeypatch.setattr(mechanism, "_build_preflight", lambda: {
        "stage": mechanism.STAGE,
        "phase": "preflight",
        "status": "PREFLIGHT_OK",
    })
    result = mechanism.run_phase("preflight", tmp_path)
    assert result["status"] == "PREFLIGHT_OK"
    assert json.loads((tmp_path / "preflight.json").read_text(encoding="utf-8"))["phase"] == "preflight"


def test_progress_uses_failure_stage(tmp_path: Path) -> None:
    mechanism._progress(tmp_path, "replay", "RUNNING", 3)
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["stage"] == mechanism.STAGE


def test_failed_phase_writes_atomic_error_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fail_preflight() -> dict[str, object]:
        raise mechanism.FailureMechanismError("raw test failure")

    monkeypatch.setattr(mechanism, "_build_preflight", fail_preflight)
    with pytest.raises(mechanism.FailureMechanismError, match="raw test failure"):
        mechanism.run_phase("preflight", tmp_path)
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "FAILED"
    assert report["error"]["message"] == "raw test failure"


def test_branch_callback_writes_strict_json_artifact(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    writes: list[Path] = []

    def fake_write(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")
        writes.append(path)

    callback_factory = mechanism.runner_utils._paired_branch_callback
    callback = callback_factory(
        tmp_path,
        selection_order=1,
        total_branches=1,
        completed_states=0,
        expected_states=1,
        started=0.0,
        registered_selection_size=3,
    )
    monkeypatch.setattr(mechanism.runner_utils, "_atomic_write_json", fake_write)
    callback({"branch": "wait", "steps": 1, "terminal": False, "censored": True}, 1, 1)
    assert any(path.name == "branch_0001.json" for path in writes)
    artifact = next(path for path in writes if path.name == "branch_0001.json")
    assert json.loads(artifact.read_text(encoding="utf-8"))["censored"] is True
