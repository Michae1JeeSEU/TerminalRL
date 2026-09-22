"""Focused checks for the frozen current-MDP P3 baseline evaluator."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from terminal_marl.training import evaluate_v4_2_p3_current_mdp_baseline_v1 as evaluator


def _record(
    controller: str,
    *,
    cohort: str = "high_soc_qualification",
    seed: int = 999001,
    completed: int = 24,
    diversions: int = 0,
    unresolved: int = 0,
    terminal: bool = True,
    shortlist: bool = True,
) -> dict:
    outcomes = {"completed": completed}
    if diversions:
        outcomes["diverted_low_soc"] = diversions
    if unresolved:
        outcomes["unresolved"] = unresolved
    safety = {key: 0 for key in evaluator.SAFETY_COUNTER_KEYS}
    return {
        "controller": controller,
        "cohort": cohort,
        "seed": seed,
        "status": "completed",
        "requests": 24,
        "completed": completed,
        "completion_fraction": completed / 24,
        "diversions": diversions,
        "unresolved": unresolved,
        "terminal": terminal,
        "seconds": 740.0,
        "safety": safety,
        "replay_safety": {
            "hard_separation_violations": 0,
            "nmac": 0,
            "collisions": 0,
        },
        "replay_error": None,
        "outcomes": outcomes,
        "replay_exact": True,
        "teacher_free": True,
        "labels_zero": True,
        "updates_zero": True,
        "runtime_shortlist_contract": shortlist,
        "runtime_shortlist_checks": {"all": shortlist},
        "mean_wait_seconds_completed_only": 100.0,
        "mean_request_to_touchdown_seconds_completed_only": 300.0,
        "valid_for_pair_metrics": (
            completed == 24 and diversions == 0 and unresolved == 0 and terminal and shortlist
        ),
        "hard_gates": {"all": shortlist},
    }


def test_protocol_bindings_and_cohort_case_matrix_are_frozen() -> None:
    protocol = evaluator.load_protocol()
    config = evaluator.scenario_config("high_soc_qualification", raw_protocol=protocol)

    assert config.total_requested_uavs == 24
    assert config.safe_distance == pytest.approx(80.0)
    assert config.nmac_distance == pytest.approx(40.0)
    assert config.collision_distance == pytest.approx(10.0)
    assert config.max_simulation_seconds == pytest.approx(1800.0)
    assert config.initial_soc_min == pytest.approx(0.90)
    assert config.initial_soc_max == pytest.approx(1.00)
    assert config.runtime_proposal_shortlist_k == 3
    assert config.fixed_arrival_schedule == (0.0,) * 12 + tuple(15.0 * i for i in range(1, 13))

    formal = evaluator.build_case_matrix("high_soc_qualification", protocol_value=protocol)
    smoke = evaluator.build_case_matrix(
        "high_soc_qualification", protocol_value=protocol, smoke=True
    )
    assert len(formal) == 6
    assert len(smoke) == 2
    assert {row["seed"] for row in smoke}.isdisjoint({835001, 835002, 835003})
    assert [row["controller"] for row in formal[:2]] == ["rule", "original_k3"]


def test_final_gates_distinguish_high_completion_from_low_report_only() -> None:
    high_records = [_record("rule"), _record("original_k3")]
    high = evaluator._final_gates(
        high_records, cohort="high_soc_qualification", expected_records=2
    )
    assert high["all"] is True

    low_records = [
        _record("rule", cohort="low_soc_stress", completed=20, diversions=4),
        _record("original_k3", cohort="low_soc_stress", completed=19, diversions=5),
    ]
    low = evaluator._final_gates(
        low_records, cohort="low_soc_stress", expected_records=2
    )
    assert low["exact_request_accounting"] is True
    assert low["all"] is True


def test_paired_metrics_are_null_when_either_side_is_incomplete() -> None:
    records = [
        _record("rule"),
        _record("original_k3", completed=23, unresolved=1, terminal=False),
    ]
    result = evaluator.paired_metrics(records)
    assert result == [{
        "seed": 999001,
        "eligible": False,
        "reason": "both_controller_records_not_complete_and_valid",
        "rule_minus_original_k3": None,
    }]


def test_raw_summary_fails_closed_for_missing_safety_and_overcount(tmp_path: Path) -> None:
    missing_safety = tmp_path / "missing_safety.json"
    missing_safety.write_text(json.dumps({"outcomes": {"0": "completed"}}), encoding="utf-8")
    with pytest.raises(evaluator.BaselineEpisodeFailure, match="safety counters"):
        evaluator._validate_raw_summary(missing_safety)

    overcount = tmp_path / "overcount.json"
    overcount.write_text(json.dumps({
        "safety": {key: 0 for key in evaluator.SAFETY_COUNTER_KEYS},
        "safety_error": None,
        "outcomes": {"0": "completed", "1": "completed", "2": "completed"},
        "completed": 3,
        "seconds": 3.0,
        "terminal": True,
    }), encoding="utf-8")
    with pytest.raises(evaluator.BaselineEpisodeFailure, match="outcome accounting"):
        evaluator._validate_raw_summary(overcount, expected_requests=2)


def test_raw_summary_fails_closed_for_unknown_nonzero_safety_counter(tmp_path: Path) -> None:
    summary = tmp_path / "unknown_safety.json"
    safety = {key: 0 for key in evaluator.SAFETY_COUNTER_KEYS}
    safety["new_future_safety_counter"] = 1
    summary.write_text(json.dumps({
        "safety": safety,
        "safety_error": None,
        "outcomes": {"completed": 2},
        "completed": 2,
        "seconds": 2.0,
        "terminal": True,
    }), encoding="utf-8")
    with pytest.raises(evaluator.BaselineEpisodeFailure, match="safety counters"):
        evaluator._validate_raw_summary(summary, expected_requests=2)


def test_outcome_accounting_preserves_other_statuses_as_unresolved() -> None:
    accounting = evaluator._outcome_accounting(
        {"completed": 1, "energy_exhausted": 1}, expected_requests=2
    )
    assert accounting["accounting_complete"] is True
    assert accounting["other_outcomes"] == {"energy_exhausted": 1}
    assert accounting["diversions"] == 0
    assert accounting["unresolved"] == 1


def test_replay_telemetry_is_a_separate_safety_gate_and_record_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replay_state = {"hard_separation_violations": 0, "nmac": 0, "collisions": 0}

    def fake_rule_result(config, seed, output, *, smoke):
        del config, seed, smoke
        output.mkdir(parents=True, exist_ok=False)
        tape = output / "action_tape.json"
        tape.write_text("{}", encoding="utf-8")
        (output / "summary.json").write_text(json.dumps({
            "safety": {key: 0 for key in evaluator.SAFETY_COUNTER_KEYS},
            "safety_error": None,
            "outcomes": {"completed": 24},
            "completed": 24,
            "seconds": 120.0,
            "terminal": True,
            "action_tape": str(tape),
        }), encoding="utf-8")
        return {
            "replay_exact": True,
            "teacher_enabled": False,
            "label_rows": 0,
            "training_called": False,
            "mean_wait_seconds": 20.0,
            "mean_request_to_touchdown_seconds": 40.0,
            "summary": {"action_tape": str(tape)},
        }

    monkeypatch.setattr(evaluator.effectiveness, "_rule_result", fake_rule_result)
    monkeypatch.setattr(
        evaluator.safety_energy,
        "_replay_telemetry",
        lambda config, tape: (
            dict(replay_state),
            {"exact": True, "steps_replayed": 1, "steps_expected": 1},
            None,
            None,
        ),
    )
    config = object()
    valid = evaluator._execute_episode(
        {"case_id": "valid", "cohort": "high_soc_qualification", "seed": 999001, "controller": "rule"},
        config=config,
        policy=None,
        output=tmp_path,
        smoke=False,
    )
    assert valid["replay_safety"] == replay_state

    replay_state["hard_separation_violations"] = 1
    with pytest.raises(evaluator.BaselineEpisodeFailure, match="replay telemetry safety"):
        evaluator._execute_episode(
            {"case_id": "replay_bad", "cohort": "high_soc_qualification", "seed": 999002, "controller": "rule"},
            config=config,
            policy=None,
            output=tmp_path,
            smoke=False,
        )


def test_failure_manifest_preserves_attempted_and_consumed_smoke_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "mock_checkpoint.pt"
    checkpoint.write_bytes(b"mock")
    real_sha = evaluator._sha256

    def fake_sha(path: str | Path) -> str:
        resolved = Path(path).resolve()
        if resolved == evaluator.PROTOCOL.resolve():
            return evaluator.PROTOCOL_SHA
        if resolved == checkpoint.resolve():
            return evaluator.ORIGINAL_K3_SHA256
        return real_sha(path)

    monkeypatch.setattr(evaluator, "_sha256", fake_sha)
    monkeypatch.setattr(
        evaluator.effectiveness,
        "load_candidate_checkpoint",
        lambda path, expected_sha256: (
            SimpleNamespace(observation_dim=88),
            {"PPO_ALLOWED": False, "stage": "mock"},
        ),
    )

    def fail_episode(*args, **kwargs):
        raise evaluator.BaselineEpisodeFailure("synthetic replay failure")

    monkeypatch.setattr(evaluator, "_execute_episode", fail_episode)
    output = tmp_path / "baseline_failure"
    manifest = evaluator.run(
        cohort="high_soc_qualification",
        output=output,
        checkpoint=checkpoint,
        history_root=tmp_path / "history",
        smoke=True,
    )

    assert manifest["status"] == "FAIL"
    assert manifest["PASS"] is False
    assert manifest["consumed_seeds"] == [835000]
    assert len(manifest["attempted_cases"]) == 1
    assert len(manifest["records"]) == 1
    assert manifest["records"][0]["status"] == "failed"
    progress = json.loads((output / "progress.json").read_text(encoding="utf-8"))
    records = json.loads((output / "records.json").read_text(encoding="utf-8"))
    assert progress["phase"] == "failed_stop"
    assert records["records"][0]["status"] == "failed"


def test_cli_smoke_flag_does_not_select_formal_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {
            "stage": evaluator.STAGE,
            "status": "SMOKE",
            "PASS": False,
            "cohort": kwargs["cohort"],
            "records": [],
            "PPO_ALLOWED": False,
        }

    monkeypatch.setattr(evaluator, "run", fake_run)
    evaluator.main([
        "--cohort", "low_soc_stress",
        "--output", "smoke-output",
        "--checkpoint", "checkpoint.pt",
        "--history-root", "history",
        "--smoke",
    ])
    assert captured["smoke"] is True
    assert evaluator.SMOKE_SEEDS["low_soc_stress"][0] not in {835101, 835102, 835103}


def test_cli_terminal_fail_returns_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        evaluator,
        "run",
        lambda **kwargs: {
            "stage": evaluator.STAGE,
            "status": "FAIL",
            "PASS": False,
            "cohort": kwargs["cohort"],
            "records": [],
            "PPO_ALLOWED": False,
        },
    )
    with pytest.raises(SystemExit) as exc_info:
        evaluator.main([
            "--cohort", "high_soc_qualification",
            "--output", "failure-output",
            "--checkpoint", "checkpoint.pt",
            "--history-root", "history",
        ])
    assert exc_info.value.code == 1
