"""Bounded contract tests for the frozen pre-action WAIT protocol."""
from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from terminal_marl.core.config import SimConfig
from terminal_marl.core.low_level_joint_action_v4_2 import HierarchicalActionV42, WAIT_GATE
from terminal_marl.training import run_p3_forced_wait_frozen_20260930 as runner


def _context(uids: list[str], actor_mask: list[bool]) -> dict:
    count = len(uids)
    return {
        "uids": uids,
        "world": np.zeros((count, 2), dtype=np.float32),
        "actor_mask": np.asarray(actor_mask, dtype=bool),
        "gate_logits": np.zeros((count, 2), dtype=np.float64),
        "proposal_logits": np.zeros((count, 25), dtype=np.float64),
        "gate_probabilities": np.tile(np.asarray([[0.25, 0.75]]), (count, 1)),
        "proposal_probabilities": np.tile(np.full((1, 25), 1.0 / 25.0), (count, 1)),
        "step": 7,
    }


def _fake_env(uid_to_arrival: dict[str, int], *, now: float = 0.0, affected=()):
    actors = {
        int(uid): SimpleNamespace(
            arrival_id=int(arrival),
            committed_plan=None,
            phase=SimpleNamespace(value="ring_outer"),
            ring_role=SimpleNamespace(value="outer"),
            safety_retry_after=0.0,
            emergency_retry_after=0.0,
            soc=0.8,
        )
        for uid, arrival in uid_to_arrival.items()
    }
    return SimpleNamespace(
        active=actors,
        now=float(now),
        step_count=7,
        future_release_reservations={},
        airspace=SimpleNamespace(affected_interfaces=lambda: set(affected)),
    )


def _request_by_arrival(result: dict) -> dict[int, tuple]:
    return {
        int(row["arrival_id"]): runner._action_signature(row)
        for row in result["rows"]
    }


def test_keyed_draws_use_arrival_id_and_preserve_process_rng(monkeypatch: pytest.MonkeyPatch) -> None:
    env_one = _fake_env({"10": 5, "20": 6})
    env_two = _fake_env({"101": 5, "202": 6})
    monkeypatch.setattr(
        runner.paired,
        "_details_context",
        lambda _policy, env: _context(sorted(str(uid) for uid in env.active), [True, True]),
    )
    random.seed(81)
    np.random.seed(82)
    torch.manual_seed(83)
    before = runner.paired._rng_snapshot()
    first = runner._keyed_request(object(), env_one, 9303101)
    after = runner.paired._rng_snapshot()
    second = runner._keyed_request(object(), env_two, 9303101)

    assert runner.paired._rng_equal(before, after)
    assert _request_by_arrival(first) == _request_by_arrival(second)
    assert first["keying"] == "arrival_id|absolute_tick|stream"


def test_forced_row_does_not_shift_other_arrival_draw(monkeypatch: pytest.MonkeyPatch) -> None:
    full_env = _fake_env({"10": 5, "20": 6})
    single_env = _fake_env({"10": 5})

    def details(_policy, env):
        uids = sorted(str(uid) for uid in env.active)
        return _context(uids, [True] if len(uids) == 1 else [True, False])

    monkeypatch.setattr(runner.paired, "_details_context", details)
    full = runner._keyed_request(object(), full_env, 9303102)
    single = runner._keyed_request(object(), single_env, 9303102)
    assert _request_by_arrival(full)[5] == _request_by_arrival(single)[5]
    assert full["rows"][1]["actor_mask"] is False


def test_arrival_identity_cannot_fallback_to_uid() -> None:
    env = SimpleNamespace(active={10: SimpleNamespace(arrival_id=None)})
    with pytest.raises(runner.ForcedWaitProtocolError, match="arrival_id missing"):
        runner._arrival_id(env, 10)


def test_mask_counts_keep_retry_boundary_and_emergency_retry_outside_mask() -> None:
    env = _fake_env({"10": 5}, now=4.0)
    env.active[10].safety_retry_after = 5.0
    counts = runner._mask_counts(env, ["10"], [False])
    assert counts["legacy_eligible"] == 1
    assert counts["forced_wait"] == 1
    assert counts["retry_blocked"] == 1

    env.now = 5.0
    env.active[10].emergency_retry_after = 100.0
    counts = runner._mask_counts(env, ["10"], [True])
    assert counts["current_actor"] == 1
    assert counts["forced_wait"] == 0
    assert counts["retry_blocked"] == 0


def _arm(status: str, requests: list[dict], *, unresolved: list[int] = ()) -> dict:
    ids = [int(row["arrival_id"]) for row in requests]
    outcomes = {str(value): "completed" for value in ids if value not in unresolved}
    return {
        "status": status,
        "request_identity": requests,
        "ledger": {
            "requested_ids": ids,
            "outcomes": outcomes,
            "unresolved_ids": unresolved,
            "conservation": {
                "known": True,
                "requested_equals_resolved_union_unresolved": True,
                "resolved_unresolved_disjoint": True,
            },
        },
        "model_unchanged": True,
        "reward_total": 2.0,
        "raw_reward_excluding_rejection_penalty": 2.1,
        "release_rejection_penalty": -0.1,
        "simulation_seconds": 10.0,
        "ticks": 10,
        "generated_requests": len(requests),
        "admitted_count": len(requests),
        "completed_count": len(ids) - len(unresolved),
        "_action_events": [],
    }


def test_pair_difference_keeps_known_and_unknown_deltas_separate() -> None:
    requests = [{"arrival_id": 0, "request_time": 0.0}]
    known = runner._paired_difference(_arm("completed", requests), _arm("completed", requests))
    assert known["both_natural_terminal_and_complete"] is True
    assert known["delta_forced_minus_legacy"]["reward_total"] == 0.0

    censored = runner._paired_difference(_arm("completed", requests), _arm("censored", requests))
    assert censored["both_natural_terminal_and_complete"] is False
    assert all(value is None for value in censored["delta_forced_minus_legacy"].values())


def test_source_binding_rejects_unregistered_digest() -> None:
    with pytest.raises(runner.ForcedWaitProtocolError, match=r"source 3fe7\+a82 model digest"):
        runner._source_binding({"source_model_digest": "wrong"})


def test_mocked_full_phase_writes_strict_outputs_and_reuses_pilot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(
        observation_schema_version=runner.OBSERVATION_SCHEMA,
        action_schema_version=runner.ACTION_SCHEMA,
        runtime_proposal_shortlist_k=3,
        max_simulation_seconds=3600.0,
        dt=1.0,
        teacher_enabled=False,
        scheduler_mode="execution",
        teacher_release_label_enabled=False,
        high_level_enabled=False,
        rule_flow_coordination_enabled=False,
    )
    specs = (
        {"pair_id": 1, "env_seed": 877101, "keyed_policy_seed": 9303101},
        {"pair_id": 2, "env_seed": 877101, "keyed_policy_seed": 9303102},
    )
    monkeypatch.setattr(runner, "PAIR_SPECS", specs)
    monkeypatch.setattr(
        runner.neutral_eval,
        "_build_neutral_inputs",
        lambda: ({}, {"dense24": config}, {}),
    )
    monkeypatch.setattr(runner, "_runtime_source", lambda _config: {"policy": object(), "source_model_digest": "digest"})
    calls: list[int] = []

    def fake_pair(_source, _config, pair, **_kwargs):
        calls.append(int(pair["pair_id"]))
        requests = [{"arrival_id": 0, "request_time": 0.0}]
        left = _arm("censored" if pair["pair_id"] == 2 else "completed", requests)
        right = _arm("censored" if pair["pair_id"] == 2 else "completed", requests)
        return {
            "stage": runner.STAGE,
            "pair": dict(pair),
            "config_snapshot": runner._config_snapshot(config),
            "runtime_manifest": runner.followup._loaded_code_manifest(),
            "arms": {"legacy": left, "pre_action_forced_wait": right},
            "paired_difference": {
                "request_identity_match": True,
                "lifecycle_conservation_match": True,
                "both_natural_terminal_and_complete": pair["pair_id"] == 1,
                "delta_forced_minus_legacy": {"reward_total": None},
            },
            "pair_status": "PAIR_OK" if pair["pair_id"] == 1 else "PAIR_INCOMPLETE",
            "source_model_digest": "digest",
        }

    monkeypatch.setattr(runner, "_run_pair", fake_pair)
    output = tmp_path / "forced_wait_frozen"
    output.mkdir()
    preflight = {
        "stage": runner.STAGE,
        "status": "PREFLIGHT_OK",
        "PPO_ALLOWED": False,
        "source_binding": {},
        "runtime": {"config_snapshot": {}},
    }
    monkeypatch.setattr(runner, "_ensure_preflight", lambda _output: preflight)
    monkeypatch.setattr(runner, "_validate_live_contract", lambda *_args: None)
    pilot_pair = fake_pair({}, config, specs[0])
    runner._write_json(output / "pilot.json", {
        "stage": runner.STAGE, "phase": "pilot", "status": "PILOT_OK", "pair": pilot_pair,
    })
    runner._write_json(output / "pairs" / "pair_0001.json", pilot_pair)
    calls.clear()

    result = runner.run_full(output)
    assert result["pilot_reused"] is True
    assert calls == [2]
    assert result["pair_ids"] == [1, 2]
    assert result["status"] == "FULL_INCOMPLETE"
    assert (output / "pairs" / "pair_0002.json").is_file()
    loaded = json.loads((output / "full.json").read_text(encoding="utf-8"))
    assert loaded["pair_count"] == 2
    assert loaded["paired_differences"][1]["delta_forced_minus_legacy"]["reward_total"] is None
    assert json.loads((output / "progress.json").read_text(encoding="utf-8"))["status"] == "FULL_INCOMPLETE"


def test_small_real_env_arm_captures_warm_start_and_failure_accounting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimConfig(
        action_schema_version=runner.ACTION_SCHEMA,
        observation_schema_version=runner.OBSERVATION_SCHEMA,
        runtime_proposal_shortlist_k=3,
        arrival_model="finite_cohort",
        total_requested_uavs=1,
        fixed_arrival_schedule=(0.0,),
        reset_mode="warm_start_rings",
        warm_start_uavs=1,
        initial_soc_mode="fixed",
        initial_soc=0.1,
        max_simulation_seconds=3.0,
    )
    policy = SimpleNamespace(model=nn.Module())
    monkeypatch.setattr(runner, "MAX_STEPS", 3)

    def wait_request(_policy, env, keyed_seed):
        rows = [
            {
                "uid": str(uid),
                "arrival_id": int(uav.arrival_id),
                "actor_mask": True,
                "gate": WAIT_GATE,
                "proposal": None,
                "shortlist": [],
            }
            for uid, uav in sorted(env.active.items())
        ]
        return {
            "actions": {
                row["uid"]: HierarchicalActionV42(WAIT_GATE, None).normalized.copy()
                for row in rows
            },
            "shortlists": {},
            "rows": rows,
            "step": int(env.step_count),
            "keyed_seed": int(keyed_seed),
            "keying": "arrival_id|absolute_tick|stream",
        }

    monkeypatch.setattr(runner, "_keyed_request", wait_request)
    result = runner._run_arm(policy, config, env_seed=877101, keyed_seed=9303101, mode="legacy")

    assert result["request_identity_count"] == 1
    assert result["request_identity"][0]["energy_capacity"] == pytest.approx(config.energy_capacity)
    assert result["request_identity"][0]["service_seconds_source"] == "MISSION_SERVICE_SECONDS[mission_type]"
    assert result["request_capture_integrity"]["captured_ids_equal_ledger"] is True
    assert result["request_capture_integrity"]["arrivals_generated_equals_captured"] is True
    assert result["ledger"]["requested_ids"] == [0]
    assert result["energy_ledger"]["coverage"]["valid"] is True
    assert result["energy_ledger"]["rows"]["0"]["energy_consumed"] >= 0.0
    assert result["admitted_count"] == 1
    assert result["telemetry_arrivals_admitted"] == 0
    assert result["wait_metrics"]["time_scope"].startswith("all_requested_ids")
    assert result["wait_integrity"]["all_request_wait_observed_finite"] is True
    assert result["wait_integrity"]["elapsed_by_requested_id_observed_finite"] is True
    assert result["ledger"]["terminal_alignment"]["valid"] is True
    assert result["caller_rng_unchanged"] is True
    assert result["outcome_counts"]["diverted"] + result["outcome_counts"]["failed"] >= 1
    assert result["resolved_count"] == 1
    assert result["completed_count"] == 0


def test_pilot_rejects_technical_pair_after_retaining_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(
        observation_schema_version=runner.OBSERVATION_SCHEMA,
        action_schema_version=runner.ACTION_SCHEMA,
        runtime_proposal_shortlist_k=3,
        max_simulation_seconds=10.0,
        dt=1.0,
        teacher_enabled=False,
        scheduler_mode="execution",
        teacher_release_label_enabled=False,
        high_level_enabled=False,
        rule_flow_coordination_enabled=False,
        student_decision_mask_mode="legacy",
    )
    specs = ({"pair_id": 1, "env_seed": 877101, "keyed_policy_seed": 9303101},)
    monkeypatch.setattr(runner, "PAIR_SPECS", specs)
    monkeypatch.setattr(
        runner.neutral_eval,
        "_build_neutral_inputs",
        lambda: ({}, {"dense24": config}, {}),
    )
    monkeypatch.setattr(
        runner, "_runtime_source",
        lambda _config: {"policy": object(), "source_model_digest": "digest"},
    )
    monkeypatch.setattr(runner, "_validate_live_contract", lambda *_args: None)
    def gate_error_arm(_policy, _config, *, env_seed, keyed_seed, mode, **_kwargs):
        if mode == "legacy":
            return {
                "mode": mode,
                "env_seed": int(env_seed),
                "keyed_policy_seed": int(keyed_seed),
                "status": "technical_error",
                "censored": None,
                "error": {"type": "GateError", "message": "invalid release gate"},
                "model_unchanged": True,
                "caller_rng_unchanged": True,
                "request_identity": [],
                "ledger": {"conservation": {"known": False}},
            }
        raise AssertionError("the paired runner should stop after the first technical arm")

    monkeypatch.setattr(runner, "_run_arm", gate_error_arm)
    preflight = {
        "stage": runner.STAGE,
        "source_binding": {},
        "runtime": {"config_snapshot": runner._config_snapshot(config)},
    }
    monkeypatch.setattr(runner, "_ensure_preflight", lambda _output: preflight)

    output = tmp_path / "pilot"
    with pytest.raises(runner.ForcedWaitProtocolError, match="pilot stopped"):
        runner.run_pilot(output)

    pair = json.loads((output / "pairs" / "pair_0001.json").read_text(encoding="utf-8"))
    assert pair["pair_status"] == "PAIR_TECHNICAL_ERROR"
    assert json.loads((output / "arms" / "pair_0001_legacy.json").read_text(encoding="utf-8"))["error"]["type"] == "GateError"
    pilot = json.loads((output / "pilot.json").read_text(encoding="utf-8"))
    assert pilot["status"] == "PILOT_TECHNICAL_ERROR"
    assert json.loads((output / "progress.json").read_text(encoding="utf-8"))["status"] == "PILOT_TECHNICAL_ERROR"


def test_live_contract_rejects_stored_config_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    config = SimpleNamespace(
        observation_schema_version=runner.OBSERVATION_SCHEMA,
        action_schema_version=runner.ACTION_SCHEMA,
        runtime_proposal_shortlist_k=3,
        max_simulation_seconds=10.0,
        dt=1.0,
        teacher_enabled=False,
        scheduler_mode="execution",
        teacher_release_label_enabled=False,
        high_level_enabled=False,
        rule_flow_coordination_enabled=False,
        student_decision_mask_mode="legacy",
    )
    binding = {
        "source_model_digest": "digest",
        "candidate_digest_before_warmup": "candidate",
        "resolved_checkpoint": "source.pt",
        "warmup_checkpoint": "warmup.pt",
    }
    monkeypatch.setattr(runner, "_source_binding", lambda _source: dict(binding))
    monkeypatch.setattr(runner.followup, "_loaded_code_manifest", lambda: {"runner.py": "sha"})
    preflight = {
        "runtime": {
            "config_snapshot": {**runner._config_snapshot(config), "dt": 2.0},
            "code_manifest": {"runner.py": "sha"},
        },
        "source_binding": binding,
        "fixed_env_seeds": list(runner.ENV_SEEDS),
        "fixed_keyed_policy_seeds": list(runner.POLICY_SEEDS),
        "pair_grid": [dict(item) for item in runner.PAIR_SPECS],
    }
    with pytest.raises(runner.ForcedWaitProtocolError, match="config snapshot differs"):
        runner._validate_live_contract(preflight, config, {"source_model_digest": "digest"})
