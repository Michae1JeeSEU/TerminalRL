"""Contract tests for the bounded P3 foundation repair evaluator."""
from __future__ import annotations

from dataclasses import dataclass
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from terminal_marl.core.low_level_joint_action_v4_2 import WAIT_GATE
from terminal_marl.scripts import evaluate_p3_foundation_repair_20261002 as evaluator
from terminal_marl.training import paired_gate_diagnostic as paired


@dataclass
class _Config:
    observation_schema_version: str = evaluator.OBSERVATION_SCHEMA_88
    final_approach_execution_mode: str = "legacy"
    trajectory_schema_version: str = evaluator.TRAJECTORY_SCHEMA_V3
    vehicle_execution_mode: str = "performance_constrained"


class _Model:
    def distributions(self, observations):
        rows = int(observations.shape[0])
        return (
            torch.distributions.Categorical(logits=torch.zeros((rows, 2))),
            torch.distributions.Categorical(logits=torch.zeros((rows, 25))),
        )


class _Policy:
    def __init__(self, dimension: int):
        self.model = _Model()
        self.observation_dim = int(dimension)
        self.device = torch.device("cpu")


def _source_metadata(*, trajectory_schema: str) -> dict[str, object]:
    return {
        "observation_schema_version": evaluator.OBSERVATION_SCHEMA_88,
        "observation_dimension": evaluator.OBSERVATION_DIM_88,
        "action_schema_version": evaluator.ACTION_SCHEMA_V4_2,
        "low_level_joint_action_count": 26,
        "low_level_policy_distribution": "hierarchical_gate_2_conditional_proposal_25",
        "trajectory_schema_version": trajectory_schema,
    }


class _Actor:
    def __init__(self, *, phase: str, arrival_id: int):
        self.phase = SimpleNamespace(value=phase)
        self.committed_plan = None
        self.arrival_id = int(arrival_id)


class _ObservationEnv:
    def __init__(self, dimension: int):
        self.step_count = 7
        self.future_release_reservations = {}
        self.active = {
            1: _Actor(phase="ring_outer", arrival_id=101),
            2: _Actor(phase="landed", arrival_id=102),
        }
        self._data = {
            1: np.zeros(dimension, dtype=np.float32),
            2: np.ones(dimension, dtype=np.float32),
        }

    def _observations(self):
        return self._data


def test_config_variants_keep_modes_explicit():
    base = _Config()
    legacy = evaluator._variant_config(base, schema="88", execution="legacy")
    repaired88 = evaluator._variant_config(base, schema="88", execution="speed_timing_v1")
    repaired90 = evaluator._variant_config(base, schema="90", execution="speed_timing_v1")
    assert legacy.observation_schema_version == evaluator.OBSERVATION_SCHEMA_88
    assert legacy.final_approach_execution_mode == "legacy"
    assert repaired88.observation_schema_version == evaluator.OBSERVATION_SCHEMA_88
    assert repaired88.final_approach_execution_mode == "speed_timing_v1"
    assert repaired90.observation_schema_version == evaluator.OBSERVATION_SCHEMA_90
    assert repaired90.final_approach_execution_mode == "speed_timing_v1"
    assert {legacy.trajectory_schema_version, repaired88.trajectory_schema_version,
            repaired90.trajectory_schema_version} == {evaluator.TRAJECTORY_SCHEMA_V3}


def test_source_checkpoint_load_is_v2_while_arm_runtime_stays_v3(monkeypatch, tmp_path):
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"frozen-source")
    seen: dict[str, object] = {}

    monkeypatch.setattr(evaluator, "_local_checkpoint_path", lambda: checkpoint)
    monkeypatch.setattr(evaluator, "_sha256", lambda _path: evaluator.SOURCE_CHECKPOINT_SHA256)

    def fake_load(_checkpoint, config):
        seen["config"] = config
        return _Policy(evaluator.OBSERVATION_DIM_88), {
            "metadata": _source_metadata(trajectory_schema=evaluator.TRAJECTORY_SCHEMA_V2),
        }

    monkeypatch.setattr(evaluator.probe, "_load_policy", fake_load)
    runtime_config = _Config()

    _policy, _payload, _path, binding = evaluator._load_policy_for_arm(runtime_config, "88")

    load_config = seen["config"]
    assert load_config.trajectory_schema_version == evaluator.TRAJECTORY_SCHEMA_V2
    assert runtime_config.trajectory_schema_version == evaluator.TRAJECTORY_SCHEMA_V3
    assert binding["source_checkpoint_trajectory_schema_version"] == evaluator.TRAJECTORY_SCHEMA_V2
    assert binding["source_load_trajectory_schema_version"] == evaluator.TRAJECTORY_SCHEMA_V2
    assert binding["runtime_environment_trajectory_schema_version"] == evaluator.TRAJECTORY_SCHEMA_V3
    assert binding["inference_transfer"] == evaluator.FROZEN_INFERENCE_TRANSFER


def test_source_checkpoint_v3_metadata_is_rejected_after_strict_v2_load(monkeypatch, tmp_path):
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"frozen-source")
    monkeypatch.setattr(evaluator, "_local_checkpoint_path", lambda: checkpoint)
    monkeypatch.setattr(evaluator, "_sha256", lambda _path: evaluator.SOURCE_CHECKPOINT_SHA256)
    monkeypatch.setattr(
        evaluator.probe,
        "_load_policy",
        lambda _checkpoint, _config: (
            _Policy(evaluator.OBSERVATION_DIM_88),
            {"metadata": _source_metadata(trajectory_schema=evaluator.TRAJECTORY_SCHEMA_V3)},
        ),
    )

    with pytest.raises(
        evaluator.FoundationRepairProtocolError,
        match="source checkpoint trajectory schema must be executable_trajectory_v2",
    ):
        evaluator._load_source_policy(_Config())


def test_continuation_stop_waits_for_service_to_empty():
    env = SimpleNamespace(
        arrivals=SimpleNamespace(exhausted=lambda: True),
        active={},
        admission=SimpleNamespace(queue=SimpleNamespace(items=[])),
        service_uavs={"uav": object()},
        config=SimpleNamespace(max_simulation_seconds=1800.0),
        now=10.0,
    )
    reason, natural = paired._continuation_stop_reason(env, False, 1, 1800)
    assert (reason, natural) == (None, False)
    env.service_uavs = {}
    reason, natural = paired._continuation_stop_reason(env, False, 2, 1800)
    assert (reason, natural) == ("terminal", True)


def test_missing_duplicate_and_unknown_metrics_stay_out_of_failed_count():
    counts = evaluator._classify_outcomes(
        {"1": "failed_energy", "2": "diverted_low_soc"},
        unresolved_ids=[3],
        requested_ids=[1, 2, 3, 4],
    )
    assert counts == {
        "normal": 0,
        "emergency": 0,
        "failed": 1,
        "diverted": 1,
        "unresolved": 2,
    }
    integrity = evaluator._metrics_integrity(
        requested_ids=[1, 2],
        outcomes={"1": "completed", "2": "completed"},
        unresolved_ids=[],
        terminal_rows=[{"arrival_id": 1}, {"arrival_id": 1}, {"arrival_id": 2}],
        energy={
            "coverage": {"valid": True},
            "rows": {
                "1": {"energy_consumed": 1.0},
                "2": {"energy_consumed": 2.0},
            },
        },
        wait={
            "all_request_cumulative_wait_seconds": 1.0,
            "external_queue_integral_seconds": 0.0,
            "internal_holding_sum_seconds": 1.0,
            "request_elapsed_seconds_by_arrival": {"1": 1.0, "2": 2.0},
        },
        status="completed",
        lifecycle_alignment={"valid": False},
    )
    assert integrity["status"] == "invalid"
    assert "duplicate terminal rows" in integrity["technical_errors"]
    assert "terminal rows do not align with ledger outcomes" in integrity["technical_errors"]
    assert integrity["business_metrics_independent_from_validity"] is True


def test_missing_elapsed_or_energy_is_null_and_invalid():
    elapsed, elapsed_error = evaluator._total_elapsed_seconds(
        {"request_elapsed_seconds_by_arrival": {"1": None}}, [1]
    )
    energy, energy_error = evaluator._total_energy_consumed(
        {"rows": {"1": {"energy_consumed": float("nan")}}}, [1]
    )
    assert elapsed is None and elapsed_error
    assert energy is None and energy_error


def _comparison_arm(*, schema: str, hash_value: str):
    dimension = evaluator.OBSERVATION_DIM_88 if schema == "88" else evaluator.OBSERVATION_DIM_90
    return {
        "arm_id": f"repaired{schema}",
        "status": "completed",
        "censored": False,
        "validity": {"status": "valid"},
        "safety": {
            "safety_violation_status": "none",
            "SafetyInvariantError": 0,
            "uncertified": 0,
            "collision_events": [],
            "nmac_events": [],
            "protected_floor_events": [],
            "hard_separation_events": [],
        },
        "config": {
            "observation_schema_version": evaluator.OBSERVATION_SCHEMA_88 if schema == "88" else evaluator.OBSERVATION_SCHEMA_90,
            "final_approach_execution_mode": "speed_timing_v1",
            "other": 1,
        },
        "request_identity": [{"arrival_id": 1}],
        "requested_action_groups": [[{"action": [1.0, 0.0, 0.0]}]],
        "ledger": {"outcomes": {"1": "completed"}},
        "summary": {
            "normal": 1, "emergency": 0, "failed": 0, "diverted": 0, "unresolved": 0,
            "total_wait_seconds": 1.0,
            "total_elapsed_seconds": 2.0,
            "total_energy_consumed": 3.0,
        },
        "wait_metrics": {
            "all_request_cumulative_wait_seconds": 1.0,
            "external_queue_integral_seconds": 0.0,
            "internal_holding_sum_seconds": 1.0,
            "request_elapsed_seconds_by_arrival": {"1": 2.0},
        },
        "energy_ledger": {"rows": {"1": {"energy_consumed": 3.0}}},
        "final_requested_accepted_actual_trace": [{"trajectory_hash": hash_value, "plan_id": "uuid"}],
    }


def _smoke_arm(*, arm_id: str, validity_status: str = "valid", hard_safety: bool = False):
    safety = {
        "safety_violation_status": "none",
        "SafetyInvariantError": 0,
        "uncertified": 0,
        "collision_events": [],
        "nmac_events": [],
        "protected_floor_events": [],
        "hard_separation_events": [{"kind": "test"}] if hard_safety else [],
    }
    return {
        "arm_id": arm_id,
        "status": "censored",
        "censored": True,
        "error": None,
        "lifecycle_error": None,
        "validity": {"status": validity_status, "technical_errors": []},
        "safety": safety,
        "observation_schema": evaluator.OBSERVATION_SCHEMA_88,
        "observation_dimension": evaluator.OBSERVATION_DIM_88,
        "trajectory_schema_version": evaluator.TRAJECTORY_SCHEMA_V3,
        "source_checkpoint_trajectory_schema_version": evaluator.TRAJECTORY_SCHEMA_V2,
        "source_load_trajectory_schema_version": evaluator.TRAJECTORY_SCHEMA_V2,
        "runtime_environment_trajectory_schema_version": evaluator.TRAJECTORY_SCHEMA_V3,
        "inference_transfer": evaluator.FROZEN_INFERENCE_TRANSFER,
        "checkpoint_sha256": evaluator.SOURCE_CHECKPOINT_SHA256,
        "checkpoint_metadata": {
            "trajectory_schema_version": evaluator.TRAJECTORY_SCHEMA_V2,
        },
        "progress": [{}, {}],
        "elapsed_wall_seconds": 0.1,
        "energy_ledger": {
            "coverage": {
                "requested_ids": [0],
                "missing_requested_ids": [],
                "unexpected_energy_ids": [],
                "valid": True,
            }
        },
        "summary": {
            "total_wait_seconds": 0.0,
            "total_elapsed_seconds": 0.0,
            "total_energy_consumed": 0.0,
        },
        "final_requested_accepted_actual_trace": [{
            "plan_id": "plan",
            "mode": "continuous_inward_v1",
            "trajectory_hash": "hash",
            "requested_accepted_actual_metadata": {},
            "release_intent_telemetry": {},
        }],
    }


def _patch_smoke_inputs(monkeypatch, arms):
    monkeypatch.setattr(evaluator, "_base_config", lambda: _Config())
    specs = [{"cell_id": "cell_1", "arm_id": arm["arm_id"]} for arm in arms]
    monkeypatch.setattr(evaluator, "_arm_specs", lambda: specs)
    by_id = {arm["arm_id"]: arm for arm in arms}

    def fake_run(spec, _base_config, *, max_steps):
        assert max_steps == 2
        return by_id[str(spec["arm_id"])]

    monkeypatch.setattr(evaluator, "_run_arm", fake_run)
    monkeypatch.setattr(
        evaluator,
        "_compare_zero_initialized_arms",
        lambda _left, _right: {"requested_action_tape": {"equal": True}},
    )


def test_smoke_rejects_censored_arm_with_invalid_ledger(monkeypatch):
    arms = [
        _smoke_arm(arm_id="legacy88"),
        _smoke_arm(arm_id="repaired88", validity_status="invalid"),
        _smoke_arm(arm_id="repaired90_zero_initialized"),
    ]
    _patch_smoke_inputs(monkeypatch, arms)

    with pytest.raises(
        evaluator.FoundationRepairProtocolError,
        match="two-tick smoke arm validity is not valid",
    ):
        evaluator._smoke()


def test_smoke_rejects_censored_arm_with_hard_safety_issue(monkeypatch):
    arms = [
        _smoke_arm(arm_id="legacy88"),
        _smoke_arm(arm_id="repaired88", hard_safety=True),
        _smoke_arm(arm_id="repaired90_zero_initialized"),
    ]
    _patch_smoke_inputs(monkeypatch, arms)

    with pytest.raises(
        evaluator.FoundationRepairProtocolError,
        match="two-tick smoke arm has a hard safety issue",
    ) as exc_info:
        evaluator._smoke()
    assert "repaired88" in str(exc_info.value)
    assert "hard_separation_events" in str(exc_info.value)


def test_zero_init_equivalence_tolerance_and_hash_independence():
    left = _comparison_arm(schema="88", hash_value="hash-left")
    right = _comparison_arm(schema="90", hash_value="hash-right")
    comparison = evaluator._compare_zero_initialized_arms(left, right)
    assert comparison["config_differences_only_allowed_fields"] is True
    assert comparison["wait_elapsed_energy"]["equal_with_tolerance"] is True
    assert comparison["requested_action_tape"]["equal"] is True
    assert comparison["final_trajectory_hashes"]["equal"] is False
    assert comparison["equivalent_excluding_hash"] is True


def test_matrix_issue_classification_keeps_business_failures_separate():
    arm = _comparison_arm(schema="88", hash_value="hash")
    arm["summary"]["failed"] = 1
    assert evaluator._arm_has_issue(arm) is False
    arm["censored"] = True
    assert evaluator._arm_has_issue(arm) is True


def test_hard_safety_counter_triggers_immediate_violation_signal():
    arm = {
        "safety": {
            "safety_violation_status": "none",
            "SafetyInvariantError": 0,
            "uncertified": 0,
            "collision_events": [],
            "nmac_events": [],
            "protected_floor_events": [],
            "hard_separation_events": [{"kind": "first_hard_separation_trace"}],
        }
    }
    assert evaluator._hard_safety_violation(arm) is True


def test_safety_report_uses_runtime_fail_closed_counter_zero_as_certified():
    env = SimpleNamespace(
        telemetry={evaluator.UNCERTIFIED_COUNTER_SOURCE: 0},
        conflict_events=[],
    )
    report = evaluator._safety_report(
        env,
        {evaluator.UNCERTIFIED_COUNTER_SOURCE: 0},
    )
    assert report["uncertified"] == 0
    assert report["uncertified_counter_source"] == evaluator.UNCERTIFIED_COUNTER_SOURCE
    assert evaluator._hard_safety_issue({"safety": report}) is False


def test_safety_report_positive_or_missing_runtime_counter_is_fail_closed():
    positive = evaluator._safety_report(
        SimpleNamespace(
            telemetry={evaluator.UNCERTIFIED_COUNTER_SOURCE: 2},
            conflict_events=[],
        ),
        {evaluator.UNCERTIFIED_COUNTER_SOURCE: 0},
    )
    assert positive["uncertified"] == 2
    assert evaluator._hard_safety_issue({"safety": positive}) is True

    missing = evaluator._safety_report(
        SimpleNamespace(
            telemetry={"uncertified_joint_safety_action": 4},
            conflict_events=[],
        ),
        {"uncertified_joint_safety_action": 0},
    )
    assert missing["uncertified"] is None
    assert missing["uncertified_counter_source"] == evaluator.UNCERTIFIED_COUNTER_SOURCE
    assert evaluator._hard_safety_issue({"safety": missing}) is True


@pytest.mark.parametrize("dimension", [evaluator.OBSERVATION_DIM_88, evaluator.OBSERVATION_DIM_90])
def test_tiny_context_and_keyed_request_support_both_schemas(dimension):
    env = _ObservationEnv(dimension)
    policy = _Policy(dimension)
    context = evaluator._tiny_context(policy, env)
    assert context["world"].shape == (2, dimension)
    assert context["actor_mask"].tolist() == [True, False]
    request = evaluator._source_request(policy, env, policy_seed=878201)
    assert request["observation_dimension"] == dimension
    assert request["actions"]["2"][0] == 0.0
    assert request["rows"][1]["gate"] == WAIT_GATE
    encoded = json.dumps(evaluator._jsonable(request), allow_nan=False)
    decoded = json.loads(encoded)
    assert decoded["observation_dimension"] == dimension
    assert decoded["rows"][1]["action"][0] == 0.0


def test_nominal_rule_uses_fixed_shortlist_and_wait_for_non_actor():
    env = _ObservationEnv(evaluator.OBSERVATION_DIM_88)
    request = evaluator._rule_request(env)
    assert request["actions"]["2"][0] == 0.0
    assert request["rows"][1]["gate"] == WAIT_GATE
    assert request["shortlists"]["1"]
    assert [item.tolist() for item in request["shortlists"]["1"]] == [
        [1.0, 0.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
    ]
