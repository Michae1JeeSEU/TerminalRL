from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import numpy as np

from terminal_marl.core.low_level_joint_action_v4_2 import (
    hierarchical_from_joint_index_v4_1,
)
from terminal_marl.training import paired_gate_diagnostic as paired
from terminal_marl.training import run_p3_forced_wait_frozen_20260930 as frozen
from terminal_marl.training import run_p3_gate_failure_replay_20261001 as replay


def test_action_trace_decodes_the_full_v42_catalogue() -> None:
    for joint_index in range(26):
        action = hierarchical_from_joint_index_v4_1(joint_index).normalized
        fields = replay._action_fields(action)
        assert fields["joint_index_v4_1"] == joint_index
        assert fields["gate"] == (0 if joint_index == 0 else 1)
        assert fields["proposal_index"] == (None if joint_index == 0 else joint_index - 1)

    assert replay._action_fields([0.25, 0.0, 0.0]) == {
        "gate": None,
        "proposal_index": None,
        "joint_index_v4_1": None,
    }


def test_emergency_observer_records_target_call_and_restores_step_hook(monkeypatch) -> None:
    class BaseEnv:
        def __init__(self, config):
            self.config = config
            self.telemetry = {"emergency_attempts": 0}
            self.now = 10.0
            self.active = {}

        def _try_emergency_landing(self, uav, *, force=False):
            self.telemetry["emergency_attempts"] += 1
            return bool(force)

    actor = SimpleNamespace(
        uid=7,
        arrival_id=21,
        phase="ring_outer",
        ring_role="outer",
        position=np.zeros(3),
        energy_remaining=10.0,
        energy_consumed=1.0,
        soc=0.5,
        safety_retry_after=None,
        emergency_retry_after=None,
        emergency_landing=False,
        committed_plan=None,
        assigned_lz=None,
    )
    monkeypatch.setattr(frozen, "_CapturedTerminalEnv", BaseEnv)
    trace = replay._TraceCollector([21])
    observed = replay._make_observed_env_class(trace)(SimpleNamespace())
    observed.active[actor.uid] = actor

    assert observed._try_emergency_landing(actor, force=True) is True
    assert len(trace.emergency_calls) == 1
    call = trace.emergency_calls[0]
    assert call["arrival_id"] == 21
    assert call["uid"] == 7
    assert call["force"] is True
    assert call["result"] is True
    assert call["telemetry_delta"] == {"emergency_attempts": 1.0}
    assert call["lz_rejection_detail"] is None

    original_step = paired._call_step

    def fake_step(env, actions, shortlists):
        env.step_count += 1
        env.now += 1.0
        return {}, {}, False, False, {}

    monkeypatch.setattr(paired, "_call_step", fake_step)
    try:
        trace2 = replay._TraceCollector([21])
        env = observed
        env.step_count = 0
        with replay._instrument_runtime(trace2):
            paired._call_step(env, {7: np.zeros(3)}, {7: [0, 1, 2]})
        assert paired._call_step is fake_step
        assert frozen._CapturedTerminalEnv is BaseEnv
        assert len(trace2.steps) == 1
        assert trace2.steps[0]["targets"]["21"]["before"]["action_fields"] == {
            "gate": 0,
            "proposal_index": None,
            "joint_index_v4_1": 0,
        }
    finally:
        monkeypatch.setattr(paired, "_call_step", original_step)


def test_replay_validation_requires_exact_identity_and_terminal_rows() -> None:
    expected = {
        "mode": "legacy",
        "env_seed": 877104,
        "keyed_policy_seed": 9303103,
        "status": "completed",
        "censored": False,
        "ticks": 694,
        "request_identity": [{"arrival_id": 21, "mission": "normal"}],
        "request_groups": [[{"arrival_id": 21}]],
        "outcome_counts": {"normal": 1},
        "model_unchanged": True,
        "caller_rng_unchanged": True,
        "safety": {"safety_violation": False},
        "ledger": {"terminal_rows": [{
            "arrival_id": 21,
            "outcome": "failed_energy",
            "energy_consumed": 3.0,
        }]},
    }
    actual = copy.deepcopy(expected)
    actual["ledger"]["terminal_rows"][0]["energy_consumed"] = 3.0 + 5e-10
    assert replay._validate_arm_replay(actual, expected, mode="legacy")["match"] is True

    actual["request_identity"][0]["mission"] = "emergency"
    validation = replay._validate_arm_replay(actual, expected, mode="legacy")
    assert validation["match"] is False
    assert any(item["path"] == "request_identity" for item in validation["mismatches"])


def test_trace_payload_is_json_safe_and_keeps_missing_fields_null() -> None:
    trace = replay._TraceCollector([21, 13])
    trace.steps.append({
        "step_before": 0,
        "step_after": 1,
        "targets": {"21": {"before": {"state": None}, "after": {"state": None}}},
    })
    payload = trace.payload(arm_mode="legacy", env_seed=877104, keyed_seed=9303103)
    assert payload["schema"] == "p3_gate_failure_replay_trace_v1"
    assert payload["target_arrival_ids"] == [21, 13]
    assert payload["summary"]["native_lz_rejection_detail"] is None
    assert replay._jsonable(payload)["summary"]["step_records"] == 1

    assert replay._validate_trace(trace, {"ticks": 1})["match"] is False


def test_telemetry_delta_skips_unchanged_inf_and_marks_nonfinite_changes_unknown() -> None:
    delta = replay._telemetry_delta(
        {"emergency_attempts": 4.0, "unchanged_inf": math.inf, "changed_nonfinite": math.inf},
        {"emergency_attempts": 5.0, "unchanged_inf": math.inf, "changed_nonfinite": -math.inf},
    )
    assert delta == {"changed_nonfinite": None, "emergency_attempts": 1.0}
