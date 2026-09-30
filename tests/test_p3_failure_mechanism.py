from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

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


def _fake_blocker_env() -> tuple[SimpleNamespace, SimpleNamespace]:
    actor = SimpleNamespace(
        uid=mechanism.TARGET_UID,
        arrival_id=mechanism.TARGET_ARRIVAL_ID,
        mission_type="normal",
        phase="ring_outer",
        ring_role="outer",
        soc=0.42,
        energy_remaining=12.0,
        energy_consumed=3.0,
        safety_retry_after=4.0,
        emergency_retry_after=5.0,
        committed_plan=None,
        emergency_landing=False,
    )
    zone_manager = SimpleNamespace(
        zones=[],
        rank_available_zones=lambda _actor, _now: [],
    )
    env = SimpleNamespace(
        active={mechanism.TARGET_UID: actor},
        config=SimpleNamespace(emergency_priority_guard_enabled=True),
        lz_manager=zone_manager,
        now=10.0,
        telemetry={"emergency_attempts": 0.0, "emergency_blocked_energy": 0.0},
        last_safety_decisions={mechanism.TARGET_UID: "blocked"},
        energy_reachability_diagnostics={mechanism.TARGET_UID: {"slack_energy": -1.0}},
    )
    return env, actor


def test_passive_emergency_trace_preserves_result_and_calls_native_once() -> None:
    env, actor = _fake_blocker_env()
    native_calls: list[tuple[object, tuple[object, ...], dict[str, object]]] = []

    def native(uav: object, *args: object, **kwargs: object) -> bool:
        native_calls.append((uav, args, kwargs))
        return True

    env._try_emergency_landing = native
    calls: list[dict[str, object]] = []
    mechanism._install_emergency_trace(env, calls)

    assert env._try_emergency_landing(actor, force=True) is True
    assert len(native_calls) == 1
    assert native_calls[0][1] == ()
    assert native_calls[0][2] == {"force": True}
    assert len(calls) == 1
    assert calls[0]["uid"] == mechanism.TARGET_UID
    assert calls[0]["returned"] is True


def test_first_step_payload_contains_telemetry_and_context_schema() -> None:
    env, _actor = _fake_blocker_env()
    payload = mechanism._first_step_payload(
        proposal=7,
        proposal_probability=0.04,
        env=env,
        before_target={"uid": mechanism.TARGET_UID, "energy_remaining": 12.0},
        after_target={"uid": mechanism.TARGET_UID, "energy_remaining": 11.5},
        before_telemetry={"emergency_attempts": 0.0, "emergency_blocked_energy": 0.0},
        after_telemetry={"emergency_attempts": 1.0, "emergency_blocked_energy": 1.0},
        info={
            "release_decisions": {"57": "approved"},
            "release_intent_telemetry": {"57": {"reason": "test", "runtime_shortlist_rank": 0}},
        },
        reward=-0.25,
        terminated=False,
        truncated=False,
    )
    assert payload["proposal"] == 7
    assert payload["proposal_probability"] == 0.04
    assert payload["decision"] == "approved"
    assert payload["release_intent_telemetry"]["reason"] == "test"
    assert payload["emergency_telemetry_delta"] == {
        "emergency_attempts": 1.0,
        "emergency_blocked_energy": 1.0,
    }
    assert payload["before_target_context"]["energy_remaining"] == 12.0
    assert payload["post_target_context"]["energy_remaining"] == 11.5
    assert "D" not in payload
    assert "delta_Q" not in payload


def test_first_step_error_is_retained_without_full_cycle_diagnostic() -> None:
    env, _actor = _fake_blocker_env()
    payload = mechanism._first_step_payload(
        proposal=None,
        env=env,
        before_target={"uid": mechanism.TARGET_UID},
        after_target={"uid": mechanism.TARGET_UID},
        before_telemetry={},
        after_telemetry={},
        info=None,
        reward=None,
        error=RuntimeError("synthetic branch error"),
    )
    assert payload["error"] == {"type": "RuntimeError", "message": "synthetic branch error"}
    assert "full_cycle_D" not in payload
    assert "D" not in payload


def test_blockers_fixture_writes_and_reloads_phase_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    registered = tmp_path / "registered_failure"
    registered.mkdir()
    config_snapshot = {"config": "fixture", "schema": "test"}
    (registered / "preflight.json").write_text(
        json.dumps({"status": "PREFLIGHT_OK", "config_snapshot": config_snapshot}) + "\n",
        encoding="utf-8",
    )
    selected = [
        {"step": step, "episode": 24, "uid": mechanism.TARGET_UID,
         "arrival_id": mechanism.TARGET_ARRIVAL_ID}
        for step in mechanism.REGISTERED_SELECTED_STEPS
    ]
    (registered / "selected.json").write_text(
        json.dumps({"selection": {"selected": selected}}) + "\n", encoding="utf-8"
    )
    record_path = tmp_path / "record.json"
    tape_path = tmp_path / "tape.json"
    record_path.write_text("{}\n", encoding="utf-8")
    tape_path.write_text("{}\n", encoding="utf-8")
    inputs = {
        "record_path": record_path,
        "tape_path": tape_path,
        "tape_rows": [],
        "source_path": tmp_path / "source.pt",
    }
    monkeypatch.setattr(mechanism, "FAILURE_OUTPUT_DIR", registered)
    monkeypatch.setattr(mechanism, "_resolve", lambda path: Path(path))
    monkeypatch.setattr(mechanism, "_strict_config", lambda: (
        SimpleNamespace(), config_snapshot,
    ))
    monkeypatch.setattr(mechanism, "_load_fixed_inputs", lambda: inputs)
    monkeypatch.setattr(
        mechanism, "_hash_artifact",
        lambda path, expected=None: {"path": str(path), "sha256": "fixture"},
    )
    monkeypatch.setattr(
        mechanism, "_source_policy",
        lambda config, source_path: (object(), "fixture-source-digest", {}),
    )
    monkeypatch.setattr(
        mechanism.rank_eval, "_model_digest", lambda policy: "fixture-source-digest"
    )
    monkeypatch.setattr(
        mechanism.runner_utils, "_loaded_code_manifest", lambda: {"fixture.py": "hash"}
    )

    states: dict[int, dict[str, object]] = {}
    for step in mechanism.REGISTERED_SELECTED_STEPS:
        env, _actor = _fake_blocker_env()
        states[step] = {
            "snapshot": env,
            "first_actions": {"57": [1.0, 0.0, -0.5]},
            "first_shortlists": {},
        }
    monkeypatch.setattr(
        mechanism, "_capture_selected_states",
        lambda policy, config, fixed_inputs, selected_rows, output: states,
    )
    monkeypatch.setattr(
        mechanism.paired_gate_diagnostic, "_details_context",
        lambda policy, env: {
            "uids": ["57"],
            "actor_mask": [True],
            "proposal_logits": [[float(index) for index in range(25)]],
            "proposal_probabilities": [[1.0 / 25.0 for _ in range(25)]],
        },
    )
    def fake_first_step_branch(
        policy: object, state: dict[str, object], proposal: int | None,
        logits: object, proposal_probability: float | None = None,
    ) -> dict[str, object]:
        return {
            "proposal": proposal,
            "proposal_probability": proposal_probability,
            "error": None,
        }

    monkeypatch.setattr(mechanism, "_first_step_branch", fake_first_step_branch)
    request_rows = [{"arrival_id": index} for index in range(mechanism.EXPECTED_REQUESTS)]
    terminal_rows = [
        {"arrival_id": index, "uid": index, "outcome": "completed", "terminal_time": 10}
        for index in range(mechanism.EXPECTED_REQUESTS)
    ]
    terminal_rows[mechanism.TARGET_ARRIVAL_ID] = {
        "arrival_id": mechanism.TARGET_ARRIVAL_ID,
        "uid": mechanism.TARGET_UID,
        "outcome": "failed_energy",
        "terminal_time": mechanism.TARGET_FAILURE_STEP,
    }
    replay_env, _actor = _fake_blocker_env()
    output = tmp_path / "failure_blockers"
    failed_output = tmp_path / "failure_blockers_failed"
    replay_calls = 0

    def fake_capture_replay(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal replay_calls
        replay_calls += 1
        expected_output = output if replay_calls == 1 else failed_output
        assert (expected_output / "firststep_blockers.json").is_file()
        observer = kwargs["state_observer"]
        assert callable(observer)
        observer(replay_env)
        if replay_calls == 2:
            raise RuntimeError("synthetic original replay failure")
        return {
            "replay": {"exact": True, "steps_replayed": mechanism.EXPECTED_STEPS},
            "lifecycle": {
                "terminal": True,
                "all_requests_observed": True,
                "ledger_alignment": True,
                "request_rows": request_rows,
                "rows": terminal_rows,
            },
        }

    monkeypatch.setattr(mechanism.credit, "_capture_replay", fake_capture_replay)
    replay_env._try_emergency_landing = lambda uav, *args, **kwargs: False
    manifest = mechanism._run_blockers(output)

    output_names = (
        "firststep_blockers.json", "original_auto_emergency_trace.json",
        "manifest.json", "progress.json",
    )
    loaded = {
        name: json.loads((output / name).read_text(encoding="utf-8"))
        for name in output_names
    }
    first_step = loaded["firststep_blockers.json"]
    assert [item["state"]["step"] for item in first_step["selected_states"]] == [280, 445, 726]
    assert first_step["branch_counts"] == {"expected": 78, "successful": 78, "errors": 0}
    assert sum(len(item["branches"]) for item in first_step["selected_states"]) == 78
    for item in first_step["selected_states"]:
        assert [branch["proposal_probability"] for branch in item["branches"]] == [None] + [1.0 / 25.0] * 25
    trace = loaded["original_auto_emergency_trace.json"]
    assert trace["replay"] == {"exact": True, "steps": 984, "requests": 24}
    assert trace["lifecycle"]["all_rows_retained"] is True
    assert trace["lifecycle"]["failed_energy_rows"][0]["uid"] == mechanism.TARGET_UID
    assert manifest["selected_steps"] == [280, 445, 726]
    assert loaded["manifest.json"]["branch_counts"] == first_step["branch_counts"]
    assert loaded["progress.json"]["status"] == "COMPLETED"

    with pytest.raises(RuntimeError, match="synthetic original replay failure"):
        mechanism._run_blockers(failed_output)
    failed_trace = json.loads(
        (failed_output / "original_auto_emergency_trace.json").read_text(encoding="utf-8")
    )
    failed_first_step = json.loads(
        (failed_output / "firststep_blockers.json").read_text(encoding="utf-8")
    )
    assert failed_trace["status"] == "FAILED"
    assert failed_trace["error"]["message"] == "synthetic original replay failure"
    assert failed_first_step["branch_counts"] == {"expected": 78, "successful": 78, "errors": 0}
    assert failed_first_step["original_replay_status"] == "PENDING"
