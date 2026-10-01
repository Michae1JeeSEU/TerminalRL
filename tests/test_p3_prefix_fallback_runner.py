from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from terminal_marl.core.low_level_joint_action_v4_2 import (
    HierarchicalActionV42,
    RELEASE_GATE,
    hierarchical_from_normalized_v4_2,
)
from terminal_marl.core.merge_scheduler import LowLevelIntent
from terminal_marl.training import student_rollout_v4_2 as student
from terminal_marl.training import terminal_lifecycle_metrics
from terminal_marl.training import run_p3_prefix_fallback_20261001 as runner
from terminal_marl.training.p3_prefix_fallback_control import PrefixFallbackContractError


def _proposal_ids(actions):
    values = []
    for item in actions:
        decoded = hierarchical_from_normalized_v4_2(np.asarray(item, dtype=np.float32))
        values.append(int(decoded.proposal))
    return values


class _Model:
    def __init__(self) -> None:
        self.calls = 0

    def distributions(self, world):
        self.calls += 1
        gate = torch.distributions.Categorical(logits=torch.tensor([[0.0, 3.0]], dtype=torch.float32).repeat(len(world), 1))
        logits = torch.arange(25, dtype=torch.float32).repeat(len(world), 1)
        proposal = torch.distributions.Categorical(logits=logits)
        return gate, proposal


class _Policy:
    def __init__(self) -> None:
        self.model = _Model()
        self.observation_dim = 4


class _Env:
    def __init__(self) -> None:
        self.step_count = 5
        self.now = 5.0


def _native_details(policy, env, observations, *, deterministic, shortlist_k):
    uids = sorted(observations)
    if not uids:
        return ([], np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=bool), {},
                np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                np.zeros(0), np.zeros(0), {}, {})
    world = np.asarray([observations[uid] for uid in uids], dtype=np.float32)
    gate_dist, proposal_dist = policy.model.distributions(torch.as_tensor(world))
    actor_mask = np.ones(len(uids), dtype=bool)
    gates = np.ones(len(uids), dtype=np.int64)
    proposals = np.full(len(uids), 7, dtype=np.int64)
    actions = {
        uid: HierarchicalActionV42(RELEASE_GATE, 7).normalized.copy()
        for uid in uids
    }
    ranked = torch.argsort(proposal_dist.logits[0], descending=True, stable=True).tolist()
    order = [7] + [int(value) for value in ranked if int(value) != 7]
    shortlists = {
        uid: tuple(HierarchicalActionV42(RELEASE_GATE, item).normalized.copy() for item in order[:3])
        for uid in uids
    }
    zeros = np.zeros(len(uids), dtype=np.float32)
    likelihoods = {uid: {"gate": 0.0, "proposal": 0.0, "joint": 0.0} for uid in uids}
    return (uids, world, actor_mask, actions, gates, proposals, zeros, zeros, likelihoods, shortlists)


def test_native_hook_preserves_return_and_captures_one_forward(monkeypatch):
    policy = _Policy()
    env = _Env()
    observations = {"3": np.zeros(4, dtype=np.float32)}
    original = student._student_request_details
    monkeypatch.setattr(student, "_student_request_details", _native_details)
    baseline = student._student_request_details(policy, env, observations, deterministic=False, shortlist_k=3)
    audit = {"arm_id": "fixture", "registrations": {}}
    with runner._prefix_sampler(policy, audit):
        hooked = student._student_request_details(policy, env, observations, deterministic=False, shortlist_k=3)
    assert student._student_request_details is _native_details
    assert policy.model.calls == 2
    assert len(audit["forwards"]) == 1
    assert len(audit["registrations"]) == 1
    for left, right in zip(baseline, hooked):
        if isinstance(left, dict):
            assert left.keys() == right.keys()
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
    assert _proposal_ids(hooked[9]["3"]) == [7, 24, 23]
    assert original is not None


def test_native_hook_empty_observations_has_no_forward(monkeypatch):
    policy = _Policy()
    audit = {"arm_id": "empty", "registrations": {}}
    monkeypatch.setattr(student, "_student_request_details", _native_details)
    with runner._prefix_sampler(policy, audit):
        result = student._student_request_details(policy, _Env(), {}, deterministic=False, shortlist_k=3)
    assert result[0] == []
    assert policy.model.calls == 0
    assert audit["forwards"] == []
    assert audit["registrations"] == {}


def test_native_hook_restores_alias_after_exception(monkeypatch):
    policy = _Policy()

    def failing(*_args, **_kwargs):
        raise RuntimeError("fixture")

    monkeypatch.setattr(student, "_student_request_details", failing)
    with pytest.raises(RuntimeError, match="fixture"):
        with runner._prefix_sampler(policy, {"arm_id": "error", "registrations": {}}):
            student._student_request_details(policy, _Env(), {"3": np.zeros(4)}, deterministic=False, shortlist_k=3)
    assert student._student_request_details is failing
    assert "distributions" not in policy.model.__dict__


def test_missing_current_tick_registration_fails_closed():
    audit = {
        "registrations": {"4|3": {"step": 4, "env_now": 4.0, "uid": 3}},
    }
    with pytest.raises(PrefixFallbackContractError, match="missing unique"):
        runner._registration_lookup(audit, _Env(), 3)


def test_replay_factory_uses_saved_registration_and_restores(monkeypatch):
    class FakeConfig:
        release_threshold = 0.0

    class FakeEnv:
        def __init__(self, *_args, **_kwargs):
            self.config = FakeConfig()
            self.step_count = 5
            self.now = 5.0
            self.last_release_telemetry = {}

        def _runtime_shortlist_release(self, _uav, _intent, shortlist):
            self.last_release_telemetry[3] = {
                "runtime_shortlist_candidates_checked": len(shortlist),
                "runtime_shortlist_rank": None,
            }
            return "safety_reject"

    old_student = student.TerminalParallelEnv
    old_lifecycle = terminal_lifecycle_metrics.TerminalParallelEnv
    monkeypatch.setattr(runner, "_CapturedTerminalEnv", FakeEnv)
    order = [7] + list(range(24))
    order = order[:25]
    metadata = {"sampled_proposal": 7, "source_logits": [float(25 - i) for i in range(25)], "requested_k": 3}
    # Stable logits rank is 0..24, with sampled proposal prepended.
    order = [7] + [i for i in range(25) if i != 7]
    audit = {
        "registrations": {"5|3": {"step": 5, "env_now": 5.0, "uid": 3, "order": order, "metadata": metadata}},
        "controllers": [],
    }
    capture = {}
    with runner._prefix_runtime_scope(audit, capture):
        replay_env = terminal_lifecycle_metrics.TerminalParallelEnv()
        prefix = tuple(LowLevelIntent.from_array(HierarchicalActionV42(RELEASE_GATE, item).normalized) for item in order[:3])
        decision = replay_env._runtime_shortlist_release(
            SimpleNamespace(uid=3), prefix[0], prefix,
        )
        assert decision == "safety_reject"
        assert capture["replay_factory_calls"] == 1
    assert student.TerminalParallelEnv is old_student
    assert terminal_lifecycle_metrics.TerminalParallelEnv is old_lifecycle
    assert audit["controllers"][0].records[3].full_proposals == tuple(order)


def test_sidecar_roundtrip_and_invalid_prefix_payload(tmp_path: Path):
    audit = {
        "arm_id": "cell_01",
        "registrations": {
            "5|3": {
                "step": 5, "env_now": 5.0, "uid": 3,
                "order": [7] + [i for i in range(25) if i != 7],
                "prefix": [7, 0, 1],
                "metadata": {"sampled_proposal": 7, "source_logits": [float(25 - i) for i in range(25)], "requested_k": 3},
            },
        },
    }
    path = tmp_path / "orders.json"
    runner._write_json(path, runner._sidecar_payload(audit, runner.CELL_SPECS[0]))
    loaded = runner._read_json(path)
    assert loaded["registration_count"] == 1
    assert loaded["registrations"]["5|3"]["metadata"]["requested_k"] == 3
    assert json.loads(path.read_text(encoding="utf-8"))["arm"] == runner.CANDIDATE_ARM


def _metric_row(*, complete: bool, normal: int = 24, failed: int = 0, wait: float | None = 5.0):
    return {
        "censored": not complete,
        "complete_scene_metrics_known": complete,
        "normal": normal if complete else 24,
        "successful": normal if complete else None,
        "failed": failed if complete else 1,
        "wait_seconds": wait if complete else None,
        "request_elapsed_total_seconds": 10.0 if complete else None,
        "energy_total_consumed": 2.0 if complete else None,
    }


def _summary(cell_id: int, *, complete: bool = True, normal: int = 24, failed: int = 0, wait: float | None = 5.0):
    metrics = _metric_row(complete=complete, normal=normal, failed=failed, wait=wait)
    return {
        "status": "PAIR_OK" if complete else "PAIR_CENSORED",
        "cell": dict(runner.CELL_SPECS[cell_id - 1]),
        "candidate_metrics": metrics,
        "reference_metrics": _metric_row(complete=True),
        "request_identity_match": True,
        "request_identity_exact_24": True,
        "technical_ok": complete,
        "hard_safety_clear": complete,
        "new_failed_arrival_ids": [] if complete else None,
        "new_diverted_arrival_ids": [] if complete else None,
        "new_unresolved_arrival_ids": [] if complete else None,
    }


def test_business_screen_keeps_unknown_censor_costs_unknown():
    review = runner._business_review([_summary(i, complete=False) for i in range(1, 5)])
    assert review["criterion_pass"] is False
    assert review["candidate_aggregate"]["wait_seconds"] is None
    assert review["candidate_aggregate"]["energy_total_consumed"] is None


def test_business_screen_requires_strict_gain_and_cost_bounds():
    rows = [_summary(i, normal=25 if i == 1 else 24) for i in range(1, 5)]
    review = runner._business_review(rows)
    assert review["criterion_pass"] is True
    rows[0]["candidate_metrics"]["wait_seconds"] = 6.0
    assert runner._business_review(rows)["criterion_pass"] is False


def test_baseline_arm_identity_and_metrics_are_pinned(monkeypatch):
    cell = dict(runner.CELL_SPECS[0])
    metrics = {"censored": False, "complete_scene_metrics_known": True, "normal": 18}
    arm = {
        "status": "completed", "cell": cell,
        "seed": cell["env_seed"], "policy_seed": cell["policy_seed"],
        "model_unchanged": True, "caller_rng_unchanged": True, "safety_nonzero": False,
        "checkpoint_sha256": runner.AFTER_CHECKPOINT_SHA256, "metrics": metrics,
    }
    expected = {"arms": {runner.REFERENCE_ARM_NAME: {"metrics": metrics}}}
    monkeypatch.setattr(runner.support, "_record_metrics", lambda _arm: metrics)
    runner._validate_baseline_arm(cell, arm, expected)
    wrong_seed = dict(arm, seed=int(cell["env_seed"]) + 1)
    with pytest.raises(runner.PrefixFallbackRunnerError, match="environment seed"):
        runner._validate_baseline_arm(cell, wrong_seed, expected)
    wrong_metrics = {"arms": {runner.REFERENCE_ARM_NAME: {"metrics": {"normal": 99}}}}
    with pytest.raises(runner.PrefixFallbackRunnerError, match="metrics differ"):
        runner._validate_baseline_arm(cell, arm, wrong_metrics)


def test_arm_binding_checks_reject_mutated_config_or_dependency_snapshot():
    record = {
        "checkpoint_sha256": runner.AFTER_CHECKPOINT_SHA256,
        "model_unchanged": True, "caller_rng_unchanged": True, "observer_rng_unchanged": True,
        "runtime_capture": {"factory_calls": 1, "replay_factory_calls": 1},
        "prefix_fallback_control": {
            "source_distributions_immutable": True,
            "sampler_aliases_restored": True,
            "runtime_aliases_restored": True,
        },
    }
    checks = runner._binding_checks(
        record, config_before={"dt": 1.0}, config_after={"dt": 1.0},
        dependencies_before={"student": "a"}, dependencies_after={"student": "a"},
    )
    assert all(checks.values())
    checks = runner._binding_checks(
        record, config_before={"dt": 1.0}, config_after={"dt": 2.0},
        dependencies_before={"student": "a"}, dependencies_after={"student": "a"},
    )
    assert checks["config_snapshot_unchanged"] is False
    checks = runner._binding_checks(
        record, config_before={"dt": 1.0}, config_after={"dt": 1.0},
        dependencies_before={"student": "a"}, dependencies_after={"student": "b"},
    )
    assert checks["runtime_dependency_shas_unchanged"] is False


def test_pilot_validation_roundtrips_path_strings(tmp_path: Path, monkeypatch):
    preflight = {"runner_sha256": "runner"}
    output = tmp_path
    (output / "preflight.json").write_text("{}", encoding="utf-8")
    pair = output / "pairs" / "cell_01.json"
    arm = output / "arms" / "cell_01_after_prefix_fallback.json"
    pair.parent.mkdir()
    arm.parent.mkdir()
    pair.write_text('{"status":"PAIR_OK"}', encoding="utf-8")
    sidecar = tmp_path / "sidecars" / "cell_01_orders.json"
    sidecar.parent.mkdir()
    runner._write_json(sidecar, {"cell": {"cell_id": 1}, "arm": runner.CANDIDATE_ARM, "registration_count": 0})
    arm.write_text(json.dumps({
        "status": "completed",
        "prefix_fallback_control": {
            "orders_sidecar": str(sidecar), "orders_sidecar_sha256": "digest", "registration_count": 0,
        },
    }), encoding="utf-8")
    monkeypatch.setattr(runner, "_sha256", lambda path: "digest")
    pilot = {
        "status": "PILOT_OK", "preflight_sha256": "digest", "preflight_runner_sha256": "runner",
        "pair": {"status": "PAIR_OK"},
        "pilot_artifacts": {"pair_path": str(pair), "pair_sha256": "digest", "arm_path": str(arm), "arm_sha256": "digest"},
    }
    (output / "pilot.json").write_text(json.dumps(pilot), encoding="utf-8")
    assert runner._validate_pilot_reuse(output, preflight)["status"] == "PILOT_OK"
