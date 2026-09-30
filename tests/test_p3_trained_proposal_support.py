"""Contract tests for the native trained-checkpoint proposal-support runner."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.distributions import Categorical

from terminal_marl.core.config import SimConfig
from terminal_marl.core.low_level_joint_action_v4_2 import (
    HierarchicalActionV42,
    RELEASE_GATE,
    hierarchical_from_normalized_v4_2,
)
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training import student_rollout_v4_2 as student
from terminal_marl.training import run_p3_trained_proposal_support_20261001 as runner


def _config(*, max_seconds: float = 2.0) -> SimConfig:
    return SimConfig(
        action_schema_version=runner.ACTION_SCHEMA,
        observation_schema_version=runner.OBSERVATION_SCHEMA,
        experiment_mode="fixed_low_student",
        teacher_enabled=False,
        scheduler_mode="execution",
        reset_mode="warm_start_rings",
        warm_start_uavs=1,
        warm_start_steady_motion=True,
        fixed_arrival_schedule=(0.0,),
        total_requested_uavs=1,
        max_simulation_seconds=max_seconds,
        runtime_proposal_shortlist_k=3,
    )


def _native_env(seed: int = 9101):
    env = TerminalParallelEnv(_config())
    observations, _ = env.reset(seed)
    return env, observations


def _fake_policy(logits: np.ndarray, *, gate_logits=(0.0, 10.0)):
    source_logits = torch.as_tensor(np.asarray(logits, dtype=np.float32)).reshape(1, -1)
    gate = torch.as_tensor(np.asarray(gate_logits, dtype=np.float32)).reshape(1, 2)

    class Model:
        def distributions(self, _observations):
            return Categorical(logits=gate), Categorical(logits=source_logits)

    return SimpleNamespace(observation_dim=2, device=torch.device("cpu"), model=Model())


def _fake_env():
    actor = SimpleNamespace(
        committed_plan=None,
        phase=SimpleNamespace(value="ring_outer"),
        ring_role=SimpleNamespace(value="outer"),
    )
    return SimpleNamespace(
        active={1: actor},
        future_release_reservations={},
        now=0.0,
        step_count=4,
        config=SimpleNamespace(student_decision_mask_mode="legacy"),
        airspace=SimpleNamespace(affected_interfaces=lambda: set()),
    )


def _fake_record(cell: dict, arm: str) -> dict:
    return {
        "status": "completed",
        "cell": dict(cell),
        "arm": arm,
        "outcome_counts": {"requests": 1, "normal": 1, "emergency": 0, "failed_energy": 0, "other": 0, "unresolved": 0},
        "requests": 1,
        "weights_unchanged": True,
        "safety_nonzero": False,
        "safety": {
            "collisions": 0,
            "nmac": 0,
            "hard_separation_violations": 0,
            "uncertified_execution": 0,
            "SafetyInvariantError": 0,
        },
        "lifecycle_valid": True,
        "summary": {"reward_total": 1.0, "reward_component_totals": {}},
        "lifecycle": {
            "scope": "all_requested_terminal_rows",
            "replay_exact": True,
            "all_requests_observed": True,
            "ledger_alignment": True,
            "missing_request_count": 0,
            "row_count": 1,
            "rows": [{"arrival_id": 0, "outcome": "completed", "wait_seconds": 1.0}],
        },
        "teacher_called": False,
        "training_or_update_called": False,
        "model_unchanged": True,
        "caller_rng_unchanged": True,
    }


def _full_fake_record(
    cell: dict,
    arm: str,
    *,
    normal: int = 24,
    wait: float = 24.0,
    energy: float = 12.0,
    status: str = "completed",
    terminal: bool = True,
) -> dict:
    rows = []
    for arrival_id in range(24):
        outcome = "completed" if arrival_id < normal else "failed_energy"
        rows.append({
            "arrival_id": arrival_id,
            "request_time": float(arrival_id),
            "terminal_time": float(arrival_id) + wait / 24.0,
            "mission_type": "passenger",
            "initial_soc": 0.9,
            "energy_capacity": 100.0,
            "energy_consumed": energy / 24.0,
            "wait_seconds": wait / 24.0,
            "outcome": outcome,
        })
    return {
        "status": status,
        "cell": dict(cell),
        "arm": arm,
        "requests": 24,
        "admitted_count": 24,
        "weights_unchanged": True,
        "safety_nonzero": False,
        "safety": {
            "collisions": 0,
            "nmac": 0,
            "hard_separation_violations": 0,
            "uncertified_execution": 0,
            "SafetyInvariantError": 0,
        },
        "lifecycle_valid": True,
        "summary": {
            "terminal": terminal,
            "seconds": 24.0,
            "reward_total": 1.0,
            "reward_component_totals": {},
        },
        "lifecycle": {
            "scope": "all_requested_terminal_rows",
            "replay_exact": True,
            "all_requests_observed": True,
            "ledger_alignment": True,
            "missing_request_count": 0,
            "requested_arrival_ids": list(range(24)),
            "observed_arrival_ids": list(range(24)),
            "row_count": 24,
            "rows": rows,
        },
        "teacher_called": False,
        "training_or_update_called": False,
        "model_unchanged": True,
        "caller_rng_unchanged": True,
    }


def test_mixture_floor_normalization_and_source_immutability() -> None:
    source = np.zeros(25, dtype=np.float64)
    source[24] = 7.0
    before = source.copy()
    mixed = runner._proposal_mixture(source)
    assert np.array_equal(source, before)
    assert mixed.shape == (25,)
    assert np.isfinite(mixed).all()
    assert mixed.min() >= runner.Q_FLOOR_MIN
    assert mixed.sum() == pytest.approx(1.0, abs=1e-12)
    assert mixed[24] > mixed[0]


@pytest.mark.parametrize(
    "bad",
    [np.zeros(24), np.zeros(25), np.asarray([np.nan] + [1.0] * 24),
     np.asarray([-1.0] + [1.0] * 24)],
)
def test_mixture_rejects_invalid_q(bad: np.ndarray) -> None:
    with pytest.raises(runner.TrainedProposalSupportError):
        runner._proposal_mixture(bad)


def test_epsilon_zero_delegates_native_function_and_preserves_rng() -> None:
    env, observations = _native_env()
    policy = __import__("terminal_marl.algo.joint_ppo_v4_2", fromlist=["JointPPOV42"]).JointPPOV42(88)
    torch.manual_seed(9102)
    before = runner._rng_snapshot()
    native = student._student_request_details(
        policy, env, observations, deterministic=False, shortlist_k=3,
    )
    native_after = runner._rng_snapshot()
    runner._restore_rng(before)
    env2, observations2 = _native_env()
    parity = runner._native_support_request_details(
        policy, env2, observations2, deterministic=False, shortlist_k=3,
        epsilon=0.0, original=student._student_request_details,
    )
    parity_after = runner._rng_snapshot()
    assert native[0] == parity[0]
    np.testing.assert_array_equal(native[1], parity[1])
    np.testing.assert_array_equal(native[2], parity[2])
    np.testing.assert_array_equal(native[4], parity[4])
    np.testing.assert_array_equal(native[5], parity[5])
    np.testing.assert_allclose(native[6], parity[6], equal_nan=True)
    np.testing.assert_allclose(native[7], parity[7], equal_nan=True)
    assert native[3].keys() == parity[3].keys()
    for uid in native[3]:
        np.testing.assert_array_equal(native[3][uid], parity[3][uid])
    assert native[8] == parity[8]
    assert native[9].keys() == parity[9].keys()
    for uid in native[9]:
        assert len(native[9][uid]) == len(parity[9][uid])
        for left, right in zip(native[9][uid], parity[9][uid]):
            np.testing.assert_array_equal(left, right)
    assert runner._rng_equal(native_after, parity_after)
    runner._restore_rng(before)
    assert runner._rng_equal(before, runner._rng_snapshot())


def test_support_uses_mixed_likelihood_and_source_logits_for_k3() -> None:
    policy = _fake_policy(np.zeros(25, dtype=np.float32))
    env = _fake_env()
    audit: dict[str, object] = {}
    details = runner._native_support_request_details(
        policy, env, {"1": np.zeros(2, dtype=np.float32)}, deterministic=True,
        shortlist_k=3, epsilon=runner.EPSILON, audit=audit,
    )
    proposal = int(details[5][0])
    mixed = runner._proposal_mixture(np.full(25, 1.0 / 25.0))
    assert details[4].tolist() == [RELEASE_GATE]
    assert details[8]["1"]["proposal"] == pytest.approx(float(np.log(mixed[proposal])))
    assert len(details[9]["1"]) == 3
    ranked_tail = [index for index in (0, 1, 2) if index != proposal]
    expected = tuple(HierarchicalActionV42(RELEASE_GATE, index).normalized for index in (proposal, *ranked_tail[:2]))
    for actual, wanted in zip(details[9]["1"], expected):
        np.testing.assert_array_equal(actual, wanted)
    assert audit["source_q_immutable"] is True
    assert audit["source_logits_immutable"] is True
    assert audit["k3_source_logits_ranked"] is True


def test_support_k3_uses_source_logits_when_mixture_tail_rounds_together() -> None:
    logits = np.full(25, -40.0, dtype=np.float32)
    logits[0] = 8.0
    logits[1] = 1.0
    logits[2] = 0.999999
    policy = _fake_policy(logits)
    env = _fake_env()
    details = runner._native_support_request_details(
        policy, env, {"1": np.zeros(2, dtype=np.float32)}, deterministic=True,
        shortlist_k=3, epsilon=runner.EPSILON,
    )
    selected = int(details[5][0])
    tail = [int(hierarchical_from_normalized_v4_2(action).proposal) for action in details[9]["1"]]
    assert tail[0] == selected
    assert tail[1:] == [index for index in (0, 1, 2) if index != selected][:2]


def test_support_sampler_restores_native_hook_on_success_and_error() -> None:
    original = student._student_request_details
    with runner._support_sampler({}):
        assert student._student_request_details is not original
    assert student._student_request_details is original
    with pytest.raises(RuntimeError, match="fixture"):
        with runner._support_sampler({}):
            raise RuntimeError("fixture")
    assert student._student_request_details is original


def test_runtime_factory_capture_returns_same_env_once_and_restores(monkeypatch: pytest.MonkeyPatch) -> None:
    original = student.TerminalParallelEnv
    capture: dict[str, object] = {}
    with runner._capture_runtime_env(capture):
        env = student.TerminalParallelEnv(_config())
    assert capture["factory_calls"] == 1
    assert capture["env"] is env
    assert env.__class__.__name__ == "_CapturedTerminalEnv"
    assert [row["arrival_id"] for row in env._captured_request_rows] == [0]
    assert student.TerminalParallelEnv is original

    base_env = TerminalParallelEnv(_config())
    base_obs, _ = base_env.reset(9103)
    captured: dict[str, object] = {}
    with runner._capture_runtime_env(captured):
        captured_env = student.TerminalParallelEnv(_config())
        captured_obs, _ = captured_env.reset(9103)
    assert set(base_obs) == set(captured_obs)
    for uid in base_obs:
        np.testing.assert_array_equal(base_obs[uid], captured_obs[uid])
    assert [row["arrival_id"] for row in captured_env._captured_request_rows] == [0]
    assert [row["arrival_id"] for row in captured_env._captured_request_rows].count(0) == 1
    with pytest.raises(RuntimeError, match="fixture"):
        with runner._capture_runtime_env({}):
            raise RuntimeError("fixture")
    assert student.TerminalParallelEnv is original


def test_censored_record_keeps_observed_outcomes_but_hides_full_metrics() -> None:
    record = _full_fake_record(dict(runner.CELL_SPECS[0]), "after_legacy", normal=23, terminal=False)
    metrics = runner._record_metrics(record)
    assert metrics["censored"] is True
    assert metrics["complete_scene_metrics_known"] is False
    assert metrics["normal"] == 23
    assert metrics["failed"] == 1
    assert metrics["successful"] is None
    assert metrics["wait_seconds"] is None
    assert metrics["wait_seconds_observed"] is not None


def _partial_rank_prefix_record(*, terminal: bool = False) -> dict:
    cell = dict(runner.CELL_SPECS[0])
    record = _full_fake_record(cell, "after_legacy", terminal=terminal)
    safety = record.pop("safety")
    record["summary"]["safety"] = safety
    record["status"] = "completed"
    record["error"] = {
        "type": "IncompleteLifecycle",
        "message": "full-request lifecycle metrics were not available",
    }
    record["weights_unchanged"] = True
    captured = [
        {
            "arrival_id": arrival_id,
            "request_time": 0.0,
            "mission": "passenger",
            "initial_soc": 0.9,
            "energy_capacity": 100.0,
        }
        for arrival_id in range(12)
    ]
    record["lifecycle"] = {
        "scope": "all_requested_terminal_rows",
        "replay_exact": True,
        "all_requests_observed": False,
        "ledger_alignment": True,
        "missing_request_count": 24,
        "requested_arrival_ids": list(range(12)),
        "observed_arrival_ids": [],
        "row_count": 0,
        "rows": [],
    }
    record["runtime_capture"] = {
        "available": True,
        "factory_calls": 1,
        "generated_requests": 12,
        "captured_request_ids": list(range(12)),
        "captured_request_rows": captured,
        "requested_arrival_ids": list(range(12)),
        "outcome_arrival_ids": [],
        "energy_outcome_arrival_ids": [],
        "energy_duplicate_arrival_ids": [],
        "energy_total_consumed": None,
        "ledger_alignment": True,
        "outcomes": {},
        "admitted_count": 12,
        "telemetry_arrivals_admitted": 0,
    }
    return record


def test_nonterminal_incomplete_lifecycle_is_censored_with_raw_error_preserved() -> None:
    record = runner._reclassify_incomplete_lifecycle(_partial_rank_prefix_record())
    metrics = runner._record_metrics(record)
    assert record["error"] is None
    assert record["lifecycle_qualification_error"]["type"] == "IncompleteLifecycle"
    assert metrics["technical_ok"] is True
    assert metrics["hard_safety_clear"] is True
    assert metrics["censored"] is True
    assert metrics["complete_scene_metrics_known"] is False
    assert metrics["generated"] == 12
    assert metrics["unresolved"] == 12
    assert metrics["successful"] is None
    assert metrics["wait_seconds"] is None
    assert metrics["energy_total_consumed"] is None


def test_terminal_incomplete_lifecycle_keeps_error_as_technical_failure() -> None:
    record = runner._reclassify_incomplete_lifecycle(_partial_rank_prefix_record(terminal=True))
    metrics = runner._record_metrics(record)
    assert record["error"]["type"] == "IncompleteLifecycle"
    assert "lifecycle_qualification_error" not in record
    assert metrics["technical_ok"] is False
    assert metrics["censored"] is False
    assert metrics["complete_scene_metrics_known"] is False


def test_early_censor_with_two_generated_requests_remains_technical_and_unknown() -> None:
    cell = dict(runner.CELL_SPECS[0])
    rows = []
    captured = []
    for arrival_id in range(2):
        row = {
            "arrival_id": arrival_id,
            "request_time": float(arrival_id),
            "terminal_time": None,
            "mission_type": "passenger",
            "initial_soc": 0.9,
            "energy_capacity": 100.0,
            "energy_consumed": None,
            "wait_seconds": None,
        }
        rows.append(row)
        captured.append({
            "arrival_id": arrival_id,
            "request_time": float(arrival_id),
            "mission": "passenger",
            "initial_soc": 0.9,
            "energy_capacity": 100.0,
        })
    record = _full_fake_record(cell, "after_legacy", normal=0, terminal=False)
    record["lifecycle"] = {
        "scope": "partial",
        "replay_exact": False,
        "all_requests_observed": False,
        "ledger_alignment": True,
        "missing_request_count": 22,
        "requested_arrival_ids": [0, 1],
        "observed_arrival_ids": [0, 1],
        "row_count": 2,
        "rows": rows,
    }
    record["runtime_capture"] = {
        "available": True,
        "factory_calls": 1,
        "generated_requests": 2,
        "captured_request_ids": [0, 1],
        "captured_request_rows": captured,
        "requested_arrival_ids": [0, 1],
        "outcome_arrival_ids": [],
        "energy_outcome_arrival_ids": [],
        "energy_duplicate_arrival_ids": [],
        "ledger_alignment": True,
    }
    metrics = runner._record_metrics(record)
    assert metrics["technical_ok"] is True
    assert metrics["censored"] is True
    assert metrics["generated"] == 2
    assert metrics["complete_scene_metrics_known"] is False
    assert metrics["wait_seconds"] is None


def test_duplicate_identity_and_new_failure_are_unknown_or_retained() -> None:
    cell = dict(runner.CELL_SPECS[0])
    records = {arm: _full_fake_record(cell, arm) for arm in runner.ARMS}
    rows = records["after_support"]["lifecycle"]["rows"]
    rows[-1]["arrival_id"] = rows[-2]["arrival_id"]
    summary = runner._pair_summary(cell, records)
    assert summary["request_identity_match"] is False
    assert summary["support_to_after_legacy"]["known"] is False
    records = {arm: _full_fake_record(cell, arm) for arm in runner.ARMS}
    records["after_support"]["lifecycle"]["rows"][-1]["outcome"] = "failed_energy"
    summary = runner._pair_summary(cell, records)
    assert summary["new_failed_arrival_ids"] == [23]


@pytest.mark.parametrize("bad_energy", [None, float("inf")])
def test_natural_terminal_missing_or_nonfinite_energy_is_technical(bad_energy: float | None) -> None:
    record = _full_fake_record(dict(runner.CELL_SPECS[0]), "after_legacy")
    record["lifecycle"]["rows"][-1]["energy_consumed"] = bad_energy
    metrics = runner._record_metrics(record)
    assert metrics["complete_scene_metrics_known"] is False
    assert metrics["censored"] is False
    assert metrics["technical_ok"] is False
    assert runner._technical_failure_arm({**record, "metrics": metrics}) is not None


def test_natural_terminal_missing_wait_is_technical() -> None:
    record = _full_fake_record(dict(runner.CELL_SPECS[0]), "after_legacy")
    record["lifecycle"]["rows"][-1]["wait_seconds"] = None
    metrics = runner._record_metrics(record)
    assert metrics["complete_scene_metrics_known"] is False
    assert metrics["censored"] is False
    assert metrics["technical_ok"] is False


def test_missing_hard_safety_fields_cannot_be_treated_as_clear() -> None:
    record = _full_fake_record(dict(runner.CELL_SPECS[0]), "after_legacy")
    del record["safety"]["nmac"]
    metrics = runner._record_metrics(record)
    assert metrics["hard_safety_clear"] is False
    assert metrics["technical_ok"] is False


def test_complete_lifecycle_energy_sum_must_match_live_ledger() -> None:
    record = _full_fake_record(dict(runner.CELL_SPECS[0]), "after_legacy")
    record["runtime_capture"] = {
        "available": True,
        "factory_calls": 1,
        "generated_requests": 24,
        "captured_request_ids": list(range(24)),
        "captured_request_rows": [
            {"arrival_id": i, "request_time": float(i), "mission": "passenger",
             "initial_soc": 0.9, "energy_capacity": 100.0}
            for i in range(24)
        ],
        "requested_arrival_ids": list(range(24)),
        "outcome_arrival_ids": list(range(24)),
        "energy_outcome_arrival_ids": list(range(24)),
        "energy_duplicate_arrival_ids": [],
        "energy_total_consumed": 13.0,
        "ledger_alignment": True,
        "outcomes": {str(i): "completed" for i in range(24)},
        "release_rejection_penalty": 0.0,
    }
    metrics = runner._record_metrics(record)
    assert metrics["complete_scene_metrics_known"] is False
    assert metrics["technical_ok"] is False


def test_runtime_capture_live_ledger_fields_are_serializable() -> None:
    config = _config()
    env, _ = _native_env()
    snapshot = runner._live_runtime_snapshot({"factory_calls": 1, "env": env}, config)
    assert snapshot["available"] is True
    assert snapshot["generated_requests"] == 1
    assert snapshot["requested_arrival_ids"] == [0]
    assert snapshot["admitted_count"] == 1
    encoded = json.dumps(runner._jsonable(snapshot), allow_nan=False)
    assert "NaN" not in encoded


def test_technical_failure_stops_before_next_arm_and_retains_partial_pair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def failed_arm(*, config, cell, arm, output):
        calls.append(arm)
        record = _fake_record(dict(cell), arm)
        record["status"] = "failed"
        record["error"] = {"type": "FixtureError", "message": "stop"}
        return record

    monkeypatch.setattr(runner, "_arm_record", failed_arm)
    result = runner._run_cell(tmp_path, SimpleNamespace(), dict(runner.CELL_SPECS[0]))
    assert calls == ["before_legacy"]
    assert result["status"] == "PAIR_TECHNICAL_ERROR"
    assert (tmp_path / "arms" / "cell_01_before_legacy.json").is_file()
    assert (tmp_path / "pairs" / "cell_01.json").is_file()


def test_pilot_failure_is_written_and_not_reusable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    preflight = {
        "stage": runner.STAGE, "status": "PREFLIGHT_OK", "PPO_ALLOWED": False,
        "heldout_used": False, "teacher_used": False, "optimizer_used": False,
        "off_policy": True, "no_training": True, "runner_sha256": runner._sha256(runner.__file__),
        "protocol_sha256": runner._sha256(runner.PROTOCOL), "cells": [dict(item) for item in runner.CELL_SPECS],
        "runtime": {"fixture": True}, "epsilon": runner.EPSILON,
        "q_floor_min": runner.Q_FLOOR_MIN, "proposal_count": runner.PROPOSAL_COUNT,
        "checkpoints": {"fixture": True},
    }
    monkeypatch.setattr(runner, "_phase_inputs", lambda _output: (preflight, SimpleNamespace()))
    def failed_arm(*, config, cell, arm, output):
        record = _full_fake_record(dict(cell), arm)
        record["status"] = "failed"
        record["error"] = {"type": "FixtureError", "message": "technical"}
        return record
    monkeypatch.setattr(runner, "_arm_record", failed_arm)
    output = tmp_path / "pilot_failure"
    output.mkdir()
    (output / "preflight.json").write_text(json.dumps(preflight), encoding="utf-8")
    with pytest.raises(runner.TrainedProposalSupportError, match="pilot stopped"):
        runner.run_pilot(output)
    payload = json.loads((output / "pilot.json").read_text(encoding="utf-8"))
    assert payload["status"] == "PILOT_TECHNICAL_ERROR"
    with pytest.raises(runner.TrainedProposalSupportError, match="pilot is not reusable"):
        runner._validate_pilot_reuse(output, preflight)


def test_preflight_rejects_changed_runtime_max_steps() -> None:
    config, _ = runner._build_inputs()
    payload = {
        "stage": runner.STAGE,
        "status": "PREFLIGHT_OK",
        "PPO_ALLOWED": False,
        "heldout_used": False,
        "teacher_used": False,
        "optimizer_used": False,
        "off_policy": True,
        "no_training": True,
        "runner_sha256": runner._sha256(runner.__file__),
        "epsilon": runner.EPSILON,
        "q_floor_min": runner.Q_FLOOR_MIN,
        "proposal_count": runner.PROPOSAL_COUNT,
        "cells": [dict(item) for item in runner.CELL_SPECS],
        "protocol_sha256": runner._sha256(runner.PROTOCOL),
        "runtime": {
            "max_steps": runner.MAX_STEPS - 1,
            "config_snapshot": runner._config_snapshot(config),
        },
    }
    with pytest.raises(runner.TrainedProposalSupportError, match="max_steps"):
        runner._validate_preflight(payload)


def test_business_review_rejects_lower_success_even_when_wait_improves() -> None:
    pairs = []
    for cell in runner.CELL_SPECS:
        cell = dict(cell)
        records = {arm: _full_fake_record(cell, arm) for arm in runner.ARMS}
        records["after_support"]["lifecycle"]["rows"][-1]["outcome"] = "failed_energy"
        records["after_support"]["lifecycle"]["rows"][-1]["wait_seconds"] = 0.0
        pairs.append(runner._pair_summary(cell, records))
    review = runner._business_review(pairs)
    assert review["support_to_after_legacy"]["values"]["wait_seconds"] < 0
    assert review["support_to_after_legacy"]["values"]["successful"] < 0
    assert review["development_candidate"] is False


def test_business_review_aggregate_can_be_known_with_one_pair_normal_lower() -> None:
    pairs = []
    for index, cell in enumerate(runner.CELL_SPECS):
        cell = dict(cell)
        records = {arm: _full_fake_record(cell, arm, normal=23 if index < 3 else 24) for arm in runner.ARMS}
        records["after_support"] = _full_fake_record(cell, "after_support", normal=24 if index < 3 else 23)
        if index == 3:
            records["after_support"]["lifecycle"]["rows"][-1]["outcome"] = "emergency_landed"
        pairs.append(runner._pair_summary(cell, records))
    review = runner._business_review(pairs)
    assert review["support_to_after_legacy"]["known"] is True
    assert review["support_to_after_legacy"]["values"]["normal"] > 0
    assert review["development_candidate"] is True


def test_pair_summary_keeps_issue_identity_and_review_requires_known_strict_gain() -> None:
    cell = dict(runner.CELL_SPECS[0])
    records = {arm: _fake_record(cell, arm) for arm in runner.ARMS}
    records["after_support"]["lifecycle"]["rows"] = [{"arrival_id": 1, "outcome": "completed", "wait_seconds": 0.5}]
    summary = runner._pair_summary(cell, records)
    assert summary["new_failed_arrival_ids"] is None
    assert summary["support_to_after_legacy"]["known"] is False
    all_pairs = [copy.deepcopy(summary) for _ in runner.CELL_SPECS]
    review = runner._business_review(all_pairs)
    assert review["all_four_natural"] is False
    assert review["development_candidate"] is False
    summary["support_to_after_legacy"]["known"] = False
    assert runner._business_review([summary] + all_pairs[1:])["development_candidate"] is False


def test_mocked_pilot_full_writes_outputs_and_does_not_rerun_pilot_cell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    preflight = {
        "stage": runner.STAGE, "status": "PREFLIGHT_OK", "PPO_ALLOWED": False,
        "heldout_used": False, "teacher_used": False, "optimizer_used": False,
        "off_policy": True, "no_training": True, "runner_sha256": runner._sha256(runner.__file__),
        "protocol_sha256": runner._sha256(runner.PROTOCOL), "cells": [dict(item) for item in runner.CELL_SPECS],
        "runtime": {"fixture": True}, "epsilon": runner.EPSILON,
        "q_floor_min": runner.Q_FLOOR_MIN, "proposal_count": runner.PROPOSAL_COUNT,
        "checkpoints": {"fixture": True},
    }
    config = SimpleNamespace()
    calls: list[int] = []
    monkeypatch.setattr(runner, "_phase_inputs", lambda _output: (preflight, config))
    monkeypatch.setattr(runner, "_arm_record", lambda *, config, cell, arm, output: calls.append(int(cell["cell_id"])) or _full_fake_record(dict(cell), arm))
    output = tmp_path / "run"
    output.mkdir()
    (output / "preflight.json").write_text(json.dumps(preflight), encoding="utf-8")
    pilot = runner.run_pilot(output)
    assert pilot["status"] == "PILOT_OK"
    before = list(calls)
    full = runner.run_full(output)
    assert full["pair_count"] == 4
    assert calls == before + [2, 2, 2, 3, 3, 3, 4, 4, 4]
    assert (output / "pilot.json").is_file()
    assert (output / "full.json").is_file()
    json.loads((output / "full.json").read_text(encoding="utf-8"))
    arm_path = output / "arms" / "cell_01_before_legacy.json"
    original_bytes = arm_path.read_bytes()
    arm_path.write_bytes(original_bytes + b"\n")
    with pytest.raises(runner.TrainedProposalSupportError, match="pilot before_legacy artifact changed"):
        runner._validate_pilot_reuse(output, preflight)
