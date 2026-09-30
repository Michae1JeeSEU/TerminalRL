"""Contract tests for the frozen always-RELEASE gate control."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from torch import nn

from terminal_marl.core.config import SimConfig
from terminal_marl.core.low_level_joint_action_v4_2 import RELEASE_GATE, WAIT_GATE
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training import paired_gate_diagnostic as paired
from terminal_marl.training import run_p3_gate_release_control_20260930 as runner


def _context(uids, actor_mask):
    count = len(uids)
    proposal_logits = np.tile(np.arange(25, dtype=np.float64), (count, 1))
    proposal_probabilities = np.zeros((count, 25), dtype=np.float64)
    proposal_probabilities[:, 3] = 1.0
    return {
        "uids": [str(uid) for uid in uids],
        "actor_mask": np.asarray(actor_mask, dtype=bool),
        "world": np.zeros((count, 88), dtype=np.float32),
        "gate_logits": np.asarray([[0.0, 1.0]] * count, dtype=np.float64),
        "proposal_logits": proposal_logits,
        "gate_probabilities": np.asarray([[1.0, 0.0]] * count, dtype=np.float64),
        "proposal_probabilities": proposal_probabilities,
        "step": 17,
    }


def _actor(arrival_id: int, *, committed=None):
    return SimpleNamespace(arrival_id=int(arrival_id), committed_plan=committed)


def _natural_arm(request_identity):
    outcomes = {str(row["arrival_id"]): "completed" for row in request_identity}
    return {
        "status": "completed",
        "generated_requests": 24,
        "request_identity_count": 24,
        "request_identity": request_identity,
        "model_digest_before": "paired-digest",
        "model_digest_after": "paired-digest",
        "model_unchanged": True,
        "caller_rng_unchanged": True,
        "safety": {"safety_violation": False},
        "ledger": {
            "outcomes": outcomes,
            "requested_ids": [int(row["arrival_id"]) for row in request_identity],
            "unresolved_ids": [],
            "conservation": {"requested_equals_resolved_union_unresolved": True},
            "terminal_alignment": {"valid": True},
        },
        "energy_ledger": {
            "rows": {
                str(row["arrival_id"]): {
                    "arrival_id": int(row["arrival_id"]), "energy_consumed": 1.0,
                }
                for row in request_identity
            },
            "unique_arrival_ids": [int(row["arrival_id"]) for row in request_identity],
            "coverage": {
                "requested_ids": [int(row["arrival_id"]) for row in request_identity],
                "missing_requested_ids": [], "unexpected_energy_ids": [],
                "unique_arrival_ids": 24, "valid": True,
            },
        },
        "wait_integrity": {
            "all_request_wait_observed_finite": True,
            "external_queue_integral_observed_finite": True,
            "internal_holding_observed_finite": True,
            "elapsed_by_requested_id_observed_finite": True,
        },
        "outcome_counts": {"normal": 24, "emergency": 0, "failed": 0, "diverted": 0, "unresolved": 0},
        "admitted_count": 24,
        "completed_count": 24,
        "resolved_count": 24,
        "simulation_seconds": 20.0,
        "wait_metrics": {
            "all_request_cumulative_wait_seconds": 10.0,
            "request_elapsed_seconds_by_arrival": {str(row["arrival_id"]): 1.0 for row in request_identity},
        },
        "reward_total": 2.0,
        "raw_reward_excluding_rejection_penalty": 2.0,
        "release_rejection_penalty": 0.0,
    }


def test_control_forces_gate_preserves_q_uniform_k3_logits_and_rng(monkeypatch: pytest.MonkeyPatch):
    policy = SimpleNamespace(model=nn.Linear(2, 2))
    env = SimpleNamespace(active={1: _actor(101), 2: _actor(202)})
    context = _context([1, 2], [True, False])
    monkeypatch.setattr(paired, "_details_context", lambda _policy, _env: context)
    before_state = {key: value.detach().clone() for key, value in policy.model.state_dict().items()}
    before_rng = paired._rng_snapshot()

    request = runner._control_keyed_request(policy, env, 9303101)

    assert paired._rng_equal(before_rng, paired._rng_snapshot())
    assert all(np.array_equal(before_state[key], value) for key, value in policy.model.state_dict().items())
    assert request["actions"]["1"][0] == RELEASE_GATE
    assert request["actions"]["2"][0] == WAIT_GATE
    assert "1" in request["shortlists"]
    assert "2" not in request["shortlists"]
    row1, row2 = request["rows"]
    assert row1["legacy_gate_sample"] == WAIT_GATE
    assert row1["gate"] == RELEASE_GATE
    assert row1["proposal"] == 3
    assert row1["proposal_uniform"] == paired.keyed_uniform(9303101, 17, 101, "proposal")
    expected_shortlist = paired._shortlist_from_logits(3, context["proposal_logits"][0])
    assert all(np.array_equal(left, right) for left, right in zip(request["shortlists"]["1"], expected_shortlist))
    assert row2["gate_uniform"] is None and row2["proposal_uniform"] is None
    assert np.array_equal(context["proposal_logits"], np.tile(np.arange(25), (2, 1)))
    assert np.array_equal(context["proposal_probabilities"][:, 3], np.ones(2))


def test_reference_integrity_and_issue_deltas_are_explicit():
    request_identity = [{"arrival_id": index} for index in range(1, 25)]
    reference = _natural_arm(request_identity)
    control = copy.deepcopy(reference)
    checks = runner._integrity_checks(control, {"arm": reference})
    assert checks["technical_ok"] is True
    control["request_identity"] = [{"arrival_id": 999}] + request_identity[1:]
    assert runner._integrity_checks(control, {"arm": reference})["technical_ok"] is False

    control = copy.deepcopy(reference)
    control["ledger"]["outcomes"]["1"] = "failed_energy"
    control_metrics = runner._business_metrics(control)
    reference_metrics = runner._business_metrics(reference)
    assert runner._paired_business_delta(control_metrics, reference_metrics)["complete_scene_delta_known"] is True
    assert runner._issue_rows(control, "failed") == [{"arrival_id": 1, "outcome": "failed_energy"}]
    assert control_metrics["energy_total_consumed"] == pytest.approx(24.0)
    assert control_metrics["energy_total_consumed_field"] == "arm.energy_ledger.rows[*].energy_consumed"

    missing = copy.deepcopy(reference)
    missing["energy_ledger"]["rows"].pop("1")
    missing_metrics = runner._business_metrics(missing)
    assert missing_metrics["energy_total_consumed"] is None
    assert missing_metrics["energy_total_consumed_complete"] is False

    duplicate = copy.deepcopy(reference)
    duplicate["energy_ledger"]["rows"] = list(duplicate["energy_ledger"]["rows"].values())
    duplicate["energy_ledger"]["rows"].append(copy.deepcopy(duplicate["energy_ledger"]["rows"][0]))
    duplicate_metrics = runner._business_metrics(duplicate)
    assert duplicate_metrics["energy_total_consumed"] is None
    assert duplicate_metrics["energy_total_consumed_complete"] is False


def _summary_for_review(control, reference):
    control_metrics = runner._business_metrics(control)
    reference_metrics = runner._business_metrics(reference)
    summary = {
        "control_metrics": control_metrics,
        "reference_metrics": reference_metrics,
        "complete_scene_delta": runner._paired_business_delta(control_metrics, reference_metrics),
    }
    for category, field in (("failed", "new_failed_arrival_ids"),
                            ("unresolved", "new_unresolved_arrival_ids"),
                            ("diverted", "new_diverted_arrival_ids")):
        control_rows = runner._issue_rows(control, category)
        reference_ids = {int(row["arrival_id"]) for row in runner._issue_rows(reference, category)}
        summary[field] = [int(row["arrival_id"]) for row in control_rows
                          if int(row["arrival_id"]) not in reference_ids]
    return summary


def test_business_review_requires_all_nine_complete_pairs_and_arrival_level_deltas():
    reference = _natural_arm([{"arrival_id": index} for index in range(1, 25)])
    summaries = [_summary_for_review(copy.deepcopy(reference), reference) for _ in runner.PAIR_SPECS]
    baseline = runner._business_review(summaries)
    assert baseline["pair_count"] == 9
    assert baseline["all_nine_natural"] is True
    assert baseline["candidate_review_pass"] is False
    assert baseline["control_aggregate_metrics"]["complete_scene_aggregate_known"] is True

    improved_control = copy.deepcopy(reference)
    improved_control["wait_metrics"]["all_request_cumulative_wait_seconds"] = 9.0
    improved = [_summary_for_review(copy.deepcopy(reference), reference) for _ in runner.PAIR_SPECS]
    improved[0] = _summary_for_review(improved_control, reference)
    assert runner._business_review(improved)["candidate_review_pass"] is True

    penalty_only = copy.deepcopy(reference)
    penalty_only["release_rejection_penalty"] = -1.0
    penalty_summary = [_summary_for_review(penalty_only, reference) for _ in runner.PAIR_SPECS]
    assert runner._business_review(penalty_summary)["candidate_review_pass"] is False

    reference_failed = copy.deepcopy(reference)
    reference_failed["ledger"]["outcomes"]["1"] = "failed_energy"
    reference_failed["outcome_counts"] = {"normal": 23, "emergency": 0, "failed": 1, "diverted": 0, "unresolved": 0}
    moved_failure = copy.deepcopy(reference_failed)
    moved_failure["ledger"]["outcomes"].pop("1")
    moved_failure["ledger"]["outcomes"]["2"] = "failed_energy"
    moved_failure["outcome_counts"] = {"normal": 23, "emergency": 0, "failed": 1, "diverted": 0, "unresolved": 0}
    moved = [_summary_for_review(copy.deepcopy(reference), reference) for _ in runner.PAIR_SPECS]
    moved[0] = _summary_for_review(moved_failure, reference_failed)
    assert moved[0]["new_failed_arrival_ids"] == [2]
    assert runner._business_review(moved)["no_new_failure_or_unresolved"] is False

    censored = copy.deepcopy(reference)
    censored["status"] = "censored"
    censored["wait_metrics"]["all_request_cumulative_wait_seconds"] = 8.0
    censored_summaries = [_summary_for_review(copy.deepcopy(reference), reference) for _ in runner.PAIR_SPECS]
    censored_summaries[0] = _summary_for_review(censored, reference)
    censored_review = runner._business_review(censored_summaries)
    assert censored_review["all_nine_natural"] is False
    assert censored_review["candidate_review_pass"] is False
    assert censored_review["control_aggregate_metrics"]["complete_scene_aggregate_known"] is False
    assert censored_summaries[0]["complete_scene_delta"]["complete_scene_delta_known"] is False


def test_censored_result_is_retained_unknown_and_pilot_reuse_does_not_overwrite(tmp_path: Path):
    reference = _natural_arm([{"arrival_id": index} for index in range(1, 25)])
    censored = {"status": "censored", "control": {"off_policy": True},
                "reference_control_checks": {"technical_ok": None}}
    config = SimpleNamespace()
    preflight = {"runner_sha256": "runner"}
    expected = runner.PAIR_SPECS[0]
    pair = {
        "pair": expected, "status": "PAIR_CENSORED", "control_metrics": {"status": "censored"},
        "reference_metrics": runner._business_metrics(reference),
        "reference_control_checks": {"technical_ok": None},
    }
    (tmp_path / "pairs").mkdir()
    (tmp_path / "arms").mkdir()
    (tmp_path / "pairs" / "pair_0001.json").write_text(json.dumps(pair), encoding="utf-8")
    (tmp_path / "arms" / "pair_0001_control.json").write_text(json.dumps(censored), encoding="utf-8")
    pilot = {"status": "PILOT_CENSORED", "pair": pair, "preflight_runner_sha256": "runner"}
    runner._validate_pilot_reuse(pilot, preflight, tmp_path)
    with pytest.raises(runner.GateReleaseControlError):
        runner._validate_pilot_reuse(
            {"status": "PILOT_TECHNICAL_ERROR", "pair": pair, "preflight_runner_sha256": "runner"},
            preflight, tmp_path,
        )
    (tmp_path / "pilot.json").write_text(json.dumps(pilot), encoding="utf-8")
    with pytest.raises(ValueError):
        runner._write_json(tmp_path / "pilot.json", pilot, refuse_existing=True)
    assert censored["reference_control_checks"]["technical_ok"] is None


def test_real_tiny_env_control_serializes_censored_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = SimConfig(
        action_schema_version="low_action_schema_v4_2_hierarchical",
        observation_schema_version="phaseA_low_obs_v2_1_retry_state",
        runtime_proposal_shortlist_k=3,
        arrival_model="finite_cohort",
        total_requested_uavs=1,
        fixed_arrival_schedule=(0.0,),
        reset_mode="warm_start_rings",
        warm_start_uavs=1,
        initial_soc_mode="fixed",
        initial_soc=0.8,
        max_simulation_seconds=3.0,
    )
    policy = SimpleNamespace(model=nn.Linear(2, 2))
    reference = {"arm": {"model_digest_before": paired._model_digest(policy)}, "metadata": {"sha256": "fixture"}}
    original_details = paired._details_context

    def details(_policy, env):
        uid = str(next(iter(env.active)))
        return _context([uid], [True]) | {"step": int(env.step_count)}

    def fake_run_arm(_policy, _config, *, env_seed, keyed_seed, mode, progress, pair_id):
        env = TerminalParallelEnv(_config)
        env.reset(int(env_seed))
        request = runner._control_keyed_request(policy, env, keyed_seed)
        paired._call_step(env, request["actions"], request["shortlists"])
        return {
            "status": "censored", "censored": True, "model_unchanged": True,
            "caller_rng_unchanged": True, "model_digest_before": paired._model_digest(policy),
            "model_digest_after": paired._model_digest(policy), "generated_requests": 1,
            "request_identity_count": 1, "request_identity": [], "outcome_counts": {},
            "energy_ledger": {}, "wait_metrics": {}, "control": {},
        }

    monkeypatch.setattr(paired, "_details_context", details)
    monkeypatch.setattr(runner.frozen, "_run_arm", fake_run_arm)
    result = runner._run_pair(
        {"policy": policy}, config, runner.PAIR_SPECS[0], reference, tmp_path,
    )
    pair_path = tmp_path / "pairs" / "pair_0001.json"
    arm_path = tmp_path / "arms" / "pair_0001_control.json"
    assert result["summary"]["status"] == "PAIR_CENSORED"
    assert json.loads(pair_path.read_text(encoding="utf-8"))["complete_scene_delta"]["complete_scene_delta_known"] is False
    assert json.loads(arm_path.read_text(encoding="utf-8"))["control"]["off_policy"] is True
    monkeypatch.setattr(paired, "_details_context", original_details)


def test_real_frozen_run_arm_natural_fixture_serializes_energy_and_elapsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Exercise the production frozen arm helper through a one-actor natural terminal."""
    config = SimConfig(
        action_schema_version=runner.frozen.ACTION_SCHEMA,
        observation_schema_version=runner.frozen.OBSERVATION_SCHEMA,
        runtime_proposal_shortlist_k=3,
        arrival_model="finite_cohort",
        total_requested_uavs=1,
        fixed_arrival_schedule=(0.0,),
        reset_mode="warm_start_rings",
        warm_start_uavs=1,
        initial_soc_mode="fixed",
        initial_soc=0.001,
        max_simulation_seconds=300.0,
    )
    policy = SimpleNamespace(model=nn.Linear(1, 1))

    def details(_policy, env):
        uids = [str(uid) for uid in sorted(env.active)]
        count = len(uids)
        actor_mask = np.asarray(
            [bool(runner.frozen._legacy_actor(env, uid)) for uid in uids], dtype=bool,
        )
        proposal_probabilities = np.zeros((count, 25), dtype=np.float64)
        proposal_logits = np.zeros((count, 25), dtype=np.float64)
        if count:
            proposal_probabilities[:, 12] = 1.0
            proposal_logits[:, 12] = 1.0
        return {
            "uids": uids,
            "world": np.zeros((count, runner.frozen.OBSERVATION_DIM), dtype=np.float32),
            "actor_mask": actor_mask,
            "gate_logits": np.tile(np.asarray([[0.0, -1.0]]), (count, 1)),
            "proposal_logits": proposal_logits,
            "gate_probabilities": np.tile(np.asarray([[1.0, 0.0]]), (count, 1)),
            "proposal_probabilities": proposal_probabilities,
            "step": int(env.step_count),
        }

    original_keyed = runner.frozen._keyed_request
    monkeypatch.setattr(paired, "_details_context", details)
    monkeypatch.setattr(runner.frozen, "_keyed_request", runner._control_keyed_request)
    try:
        arm = runner.frozen._run_arm(
            policy, config, env_seed=877101, keyed_seed=9303101, mode="legacy",
        )
    finally:
        runner.frozen._keyed_request = original_keyed

    assert arm["status"] == "completed"
    assert arm["censored"] is False
    assert arm["ticks"] == 1
    assert arm["model_unchanged"] is True
    assert arm["caller_rng_unchanged"] is True
    assert arm["ledger"]["requested_ids"] == [0]
    assert arm["ledger"]["outcomes"] == {"0": "diverted_low_soc"}
    assert arm["energy_ledger"]["coverage"]["valid"] is True
    assert arm["energy_ledger"]["unique_arrival_ids"] == [0]
    assert arm["energy_ledger"]["rows"]["0"]["energy_consumed"] == pytest.approx(1.1)
    metrics = runner._business_metrics(arm)
    assert metrics["energy_total_consumed"] == pytest.approx(1.1)
    assert metrics["energy_total_consumed_complete"] is True
    assert metrics["request_elapsed_total_seconds"] == pytest.approx(0.0)
    assert metrics["request_elapsed_total_complete"] is True
    assert metrics["makespan_seconds"] == pytest.approx(1.0)
    output = tmp_path / "natural.json"
    runner._write_json(output, arm)
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "completed"
