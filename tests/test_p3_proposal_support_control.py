"""Contract tests for the fixed proposal-support control."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from terminal_marl.core.low_level_joint_action_v4_2 import RELEASE_GATE, WAIT_GATE
from terminal_marl.training import paired_gate_diagnostic as paired
from terminal_marl.training import run_p3_forced_wait_frozen_20260930 as frozen
from terminal_marl.training import run_p3_proposal_support_control_20261001 as runner


def _context(uids, masks):
    count = len(uids)
    q = np.zeros((count, 25), dtype=np.float64)
    q[:, 3] = 1.0
    logits = np.tile(np.arange(25, dtype=np.float64), (count, 1))
    return {
        "uids": [str(uid) for uid in uids],
        "actor_mask": np.asarray(masks, dtype=bool),
        "world": np.zeros((count, 88), dtype=np.float32),
        "gate_logits": np.tile(np.asarray([[0.0, 1.0]]), (count, 1)),
        "proposal_logits": logits,
        "gate_probabilities": np.tile(np.asarray([[0.0, 1.0]]), (count, 1)),
        "proposal_probabilities": q,
        "step": 17,
    }


def _env(*arrival_ids):
    return SimpleNamespace(
        active={int(uid): SimpleNamespace(arrival_id=int(uid)) for uid in arrival_ids},
        step_count=17,
    )


def test_mixture_floor_mass_and_source_immutability():
    source = np.zeros(25, dtype=np.float64)
    source[7] = 1.0
    before = source.copy()
    mixed = runner._proposal_mixture(source)
    assert np.array_equal(source, before)
    assert mixed.shape == (25,)
    assert np.isfinite(mixed).all()
    assert mixed.min() >= runner.Q_FLOOR_MIN
    assert mixed.sum() == pytest.approx(1.0, abs=1e-12)
    assert mixed[7] > mixed[0]


def test_epsilon_zero_matches_frozen_keying_and_preserves_caller_rng(monkeypatch):
    policy = SimpleNamespace()
    env = _env(101, 202)
    context = _context([101, 202], [True, False])
    context["gate_probabilities"][:] = np.asarray([[0.5, 0.5], [0.5, 0.5]])
    context["proposal_probabilities"][0] = 0.0
    context["proposal_probabilities"][0, 3] = 2.0
    context["proposal_probabilities"][0, 12] = 8.0
    monkeypatch.setattr(paired, "_details_context", lambda _policy, _env: context)
    before = paired._rng_snapshot()
    frozen_request = frozen._keyed_request(policy, env, 9303101)
    support_request = runner._support_keyed_request(policy, env, 9303101, epsilon=0.0)
    assert paired._rng_equal(before, paired._rng_snapshot())
    assert frozen_request["keying"] == support_request["keying"]
    assert frozen_request["actions"].keys() == support_request["actions"].keys()
    for uid in frozen_request["actions"]:
        assert np.array_equal(frozen_request["actions"][uid], support_request["actions"][uid])
    assert frozen_request["shortlists"].keys() == support_request["shortlists"].keys()
    for uid in frozen_request["shortlists"]:
        for left, right in zip(frozen_request["shortlists"][uid], support_request["shortlists"][uid]):
            assert np.array_equal(left, right)
    for frozen_row, support_row in zip(frozen_request["rows"], support_request["rows"]):
        for key in ("arrival_id", "actor_mask", "gate", "proposal", "gate_uniform", "proposal_uniform"):
            assert frozen_row[key] == support_row[key]
    assert np.array_equal(np.asarray(support_request["rows"][0]["source_q"]), context["proposal_probabilities"][0])


def test_support_changes_only_release_proposal_distribution_and_keeps_gate_mask(monkeypatch):
    policy = SimpleNamespace()
    env = _env(101, 202)
    context = _context([101, 202], [True, False])
    context["proposal_probabilities"][0] = np.linspace(1.0, 25.0, 25)
    original_q = context["proposal_probabilities"].copy()
    original_logits = context["proposal_logits"].copy()
    monkeypatch.setattr(paired, "_details_context", lambda _policy, _env: context)
    baseline = frozen._keyed_request(policy, env, 9303101)
    support = runner._support_keyed_request(policy, env, 9303101)
    assert [row["actor_mask"] for row in support["rows"]] == [row["actor_mask"] for row in baseline["rows"]]
    assert [row["gate"] for row in support["rows"]] == [row["gate"] for row in baseline["rows"]]
    assert support["rows"][1]["gate"] == WAIT_GATE
    assert support["rows"][1]["proposal"] is None
    assert support["rows"][0]["mixed_q"] is not None
    assert np.asarray(support["rows"][0]["mixed_q"]).min() >= runner.Q_FLOOR_MIN
    assert np.array_equal(context["proposal_probabilities"], original_q)
    assert np.array_equal(context["proposal_logits"], original_logits)


def test_sampled_first_k3_shortlist_and_hook_restore(monkeypatch):
    policy = SimpleNamespace()
    env = _env(101)
    context = _context([101], [True])
    context["proposal_probabilities"][0] = np.full(25, 1.0 / 25.0)
    monkeypatch.setattr(paired, "_details_context", lambda _policy, _env: context)
    request = runner._support_keyed_request(policy, env, 9303101)
    row = request["rows"][0]
    shortlist = request["shortlists"]["101"]
    assert row["proposal"] == next(
        index for index in range(25)
        if np.array_equal(np.asarray(shortlist[0]),
                          np.asarray(runner.HierarchicalActionV42(RELEASE_GATE, index).normalized))
    )
    assert len(shortlist) == 3
    assert all(np.asarray(item).shape == (3,) for item in shortlist)
    original = frozen._keyed_request
    def fail_run(*_args, **_kwargs):
        raise RuntimeError("fixture stop")
    monkeypatch.setattr(frozen, "_run_arm", fail_run)
    with pytest.raises(RuntimeError, match="fixture stop"):
        runner._run_support_arm(
            {"policy": policy}, SimpleNamespace(), runner.PAIR_SPECS[0], {"arm": {}}, lambda _row: None,
        )
    assert frozen._keyed_request is original


def _natural_reference_arm():
    ids = list(range(1, 25))
    return {
        "status": "completed", "generated_requests": 24, "admitted_count": 24,
        "completed_count": 24, "resolved_count": 24,
        "model_digest_before": "reference", "model_unchanged": True,
        "caller_rng_unchanged": True, "safety": {"safety_violation": False},
        "outcome_counts": {"normal": 24, "emergency": 0, "failed": 0, "diverted": 0, "unresolved": 0},
        "ledger": {
            "outcomes": {str(item): "completed" for item in ids}, "requested_ids": ids,
            "unresolved_ids": [],
            "conservation": {"requested_equals_resolved_union_unresolved": True},
            "terminal_alignment": {"valid": True},
        },
        "energy_ledger": {
            "rows": {str(item): {"arrival_id": item, "energy_consumed": 1.0} for item in ids},
            "unique_arrival_ids": ids,
            "coverage": {"requested_ids": ids, "valid": True},
        },
        "wait_integrity": {
            "all_request_wait_observed_finite": True,
            "external_queue_integral_observed_finite": True,
            "internal_holding_observed_finite": True,
            "elapsed_by_requested_id_observed_finite": True,
        },
        "wait_metrics": {
            "all_request_cumulative_wait_seconds": 10.0,
            "request_elapsed_seconds_by_arrival": {str(item): 1.0 for item in ids},
        },
        "simulation_seconds": 10.0,
    }


def test_run_pair_retains_censored_unknown_metrics_and_failure_ids(tmp_path: Path, monkeypatch):
    censored = {
        "status": "censored", "censored": True, "generated_requests": 24, "admitted_count": 24,
        "completed_count": None, "resolved_count": None,
        "outcome_counts": {"normal": 0, "emergency": 0, "failed": 1, "diverted": 0, "unresolved": 23},
        "ledger": {"outcomes": {"7": "failed_energy"}, "requested_ids": list(range(1, 25)),
                   "unresolved_ids": list(range(8, 25))},
        "wait_metrics": {}, "wait_integrity": {}, "energy_ledger": {},
        "proposal_support_control": {"off_policy": True},
    }
    reference = {"arm": _natural_reference_arm(), "metadata": {"sha256": "reference"}}
    monkeypatch.setattr(runner, "_run_support_arm", lambda *_args, **_kwargs: censored)
    result = runner._run_pair(
        {"policy": object()}, SimpleNamespace(), runner.PAIR_SPECS[0], reference, tmp_path,
    )
    summary = result["summary"]
    assert summary["status"] == "PAIR_CENSORED"
    assert summary["control_metrics"]["complete_scene_metrics_known"] is False
    assert summary["control_metrics"]["wait_seconds"] is None
    assert summary["control_metrics"]["request_elapsed_total_seconds"] is None
    assert summary["control_metrics"]["energy_total_consumed"] is None
    assert summary["control_failed_rows"] == [{"arrival_id": 7, "outcome": "failed_energy"}]
    assert summary["new_failed_arrival_ids"] is None
    pair_path = tmp_path / "pairs" / "pair_0001.json"
    arm_path = tmp_path / "arms" / "pair_0001_support.json"
    assert json.loads(pair_path.read_text(encoding="utf-8"))["control_metrics"]["wait_seconds"] is None
    assert json.loads(arm_path.read_text(encoding="utf-8"))["status"] == "censored"


@pytest.mark.parametrize(
    "bad_q",
    [np.zeros(24), np.zeros(25), np.asarray([np.nan] + [1.0] * 24),
     np.asarray([-1.0] + [1.0] * 24)],
)
def test_mixture_rejects_invalid_q_shapes_values_and_zero_mass(bad_q):
    with pytest.raises(runner.ProposalSupportControlError):
        runner._proposal_mixture(bad_q)


def test_preflight_contract_rejects_changed_epsilon():
    payload = {
        "stage": runner.STAGE, "status": "PREFLIGHT_OK", "PPO_ALLOWED": False,
        "heldout_used": False, "teacher_used": False, "optimizer_used": False,
        "off_policy": True, "no_logprob": True, "no_training": True,
        "runner_sha256": runner._sha256(runner.__file__),
        "proposal_support": {
            "pair_ids": list(runner._PAIR_ORDER), "epsilon": 0.11,
            "q_floor_min": runner.Q_FLOOR_MIN, "proposal_count": 25,
        },
        "base_gate_preflight": {},
    }
    with pytest.raises(runner.ProposalSupportControlError, match="parameters changed"):
        runner._validate_preflight(payload, Path("unused"))


def test_pilot_reuse_requires_immutable_artifacts_and_retains_censored_unknown(tmp_path: Path):
    output = tmp_path
    (output / "arms").mkdir()
    (output / "pairs").mkdir()
    preflight = {
        "runner_sha256": "self", "proposal_support": {"epsilon": runner.EPSILON},
        "runtime": {"config": "fixed"}, "reference": {"full_sha256": "full"},
        "reference_arms": {"1": {"sha256": "ref"}},
        "base_gate_preflight": {"runner_sha256": "base"},
    }
    (output / "preflight.json").write_text("{}", encoding="utf-8")
    metrics = None
    arm = {
        "status": "censored", "proposal_support_control": {"off_policy": True},
        "reference_control_checks": {"technical_ok": None},
    }
    arm["proposal_support_control"] = {
        "name": runner.CONTROL_NAME, "epsilon": runner.EPSILON,
        "q_floor_min": runner.Q_FLOOR_MIN, "proposal_count": runner.PROPOSAL_COUNT,
        "sampled_first": True, "off_policy": True, "logprob_recorded": False,
        "training": False,
    }
    metrics = runner.gate._business_metrics(arm)
    summary = {
        "pair": runner.PAIR_SPECS[0], "status": "PAIR_CENSORED",
        "control_metrics": metrics, "support_metrics": metrics,
        "reference_metrics": {"status": "completed"},
        "reference_control_checks": {"technical_ok": None},
        "complete_scene_delta": {"complete_scene_delta_known": False},
        "new_failed_arrival_ids": None, "new_unresolved_arrival_ids": None,
        "new_diverted_arrival_ids": None,
    }
    pair_path = output / "pairs" / "pair_0001.json"
    arm_path = output / "arms" / "pair_0001_support.json"
    pair_path.write_text(json.dumps(summary), encoding="utf-8")
    arm_path.write_text(json.dumps(arm), encoding="utf-8")
    binding = {
        "proposal_support": preflight["proposal_support"], "runtime": preflight["runtime"],
        "reference": preflight["reference"], "reference_arms": preflight["reference_arms"],
        "base_gate_runner_sha256": "base", "self_runner_sha256": "self",
    }
    pilot = {
        "status": "PILOT_CENSORED", "pair": summary,
        "preflight_runner_sha256": "self",
        "preflight_sha256": runner._sha256(output / "preflight.json"),
        "preflight_binding": binding,
        "pilot_artifacts": {
            "arm_path": str(arm_path), "arm_sha256": runner._sha256(arm_path),
            "pair_path": str(pair_path), "pair_sha256": runner._sha256(pair_path),
        },
    }
    runner._validate_pilot_reuse(pilot, preflight, output)
    tampered = dict(arm)
    tampered["proposal_support_control"] = {**arm["proposal_support_control"], "epsilon": 0.2}
    arm_path.write_text(json.dumps(tampered), encoding="utf-8")
    pilot["pilot_artifacts"]["arm_sha256"] = runner._sha256(arm_path)
    with pytest.raises(runner.ProposalSupportControlError, match="proposal-support parameters changed"):
        runner._validate_pilot_reuse(pilot, preflight, output)
