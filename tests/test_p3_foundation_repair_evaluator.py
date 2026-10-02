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
