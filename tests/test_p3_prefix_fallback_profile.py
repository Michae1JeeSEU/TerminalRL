from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from terminal_marl.training import profile_p3_prefix_fallback_20261001 as profile
from terminal_marl.training import run_p3_prefix_fallback_20261001 as prefix


class _Model:
    def distributions(self, *_args, **_kwargs):
        return "distribution"


class _Policy:
    def __init__(self) -> None:
        self.model = _Model()


class _ProbeEnv:
    def __init__(self) -> None:
        self.step_count = 0
        self.now = 0.0

    def _observations(self):
        return {7: np.zeros(4, dtype=np.float32)}


def test_jsonable_and_timing_summary_are_finite_and_empty_safe(tmp_path: Path) -> None:
    assert profile._jsonable(float("inf")) is None
    assert profile._jsonable(float("nan")) is None
    assert profile._timing_summary([])["count"] == 0
    summary = profile._timing_summary([1_000_000, 2_000_000])
    assert summary["count"] == 2
    assert summary["mean_ms"] == 1.5
    path = tmp_path / "finite.json"
    profile._write_json(path, {"value": float("inf")})
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": None}


def test_probe_uses_one_forward_and_restores_rng_without_executing_action(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"details": 0, "distributions": 0}

    def details(policy, _env, observations, *, deterministic, shortlist_k):
        calls["details"] += 1
        policy.model.distributions(observations)
        return ((),)

    original = profile.student._student_request_details
    monkeypatch.setattr(profile.student, "_student_request_details", details)
    try:
        observer = profile._PassiveObserver(_ProbeEnv(), _Policy(), "before_legacy")
        observer.policy.model.distributions = lambda *_args, **_kwargs: calls.__setitem__("distributions", calls["distributions"] + 1)
        observer._probe_actor({7: np.zeros(4, dtype=np.float32)})
        assert calls == {"details": 1, "distributions": 1}
        assert observer.actor_forward_counts == [1]
        assert observer.rng_restored is True
        observer._probe_actor({})
        assert observer.actor_forward_counts[-1] == 0
    finally:
        monkeypatch.setattr(profile.student, "_student_request_details", original)


def test_step_observer_records_team_reward_and_components(monkeypatch: pytest.MonkeyPatch) -> None:
    def details(*_args, **_kwargs):
        _args[0].model.distributions(None)
        return ((),)

    monkeypatch.setattr(profile.student, "_student_request_details", details)
    env = _ProbeEnv()
    observer = profile._PassiveObserver(env, _Policy(), "before_legacy")

    def original_step(*_args, **_kwargs):
        return ({7: np.zeros(4, dtype=np.float32)}, {7: -2.5}, False, False, {
            "reward_components": {7: {"time": -2.5, "total": -2.5}},
        })

    wrapped = observer.wrap_step(original_step)
    result = wrapped(env)
    assert result[1] == {7: -2.5}
    assert observer.step_reward_totals == [-2.5]
    assert observer.step_reward_components == {"time": -2.5}
    assert observer.step_reward_component_rows == 1


def test_private_candidate_counters_use_actual_plan_and_approved_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    env = SimpleNamespace(
        telemetry={"runtime_shortlist_candidates_checked": 0},
        last_release_telemetry={},
        step_count=3,
        now=3.0,
    )
    observer = profile._PassiveObserver(env, _Policy(), "before_prefix_fallback")
    candidate = SimpleNamespace(kind="runtime_shortlist_rank_4")

    def native_plan(*_args, **_kwargs):
        return (True, None, "accepted")

    plan = observer.wrap_candidate_plan(native_plan)

    def original_release(uav, intent, shortlist):
        plan(env, "uav", candidate, "final", object(), "LZ-1")
        env.telemetry["runtime_shortlist_candidates_checked"] = 4
        env.last_release_telemetry[uav.uid] = {
            "runtime_shortlist_candidates_checked": 4,
            "runtime_shortlist_rank": 4,
            "decision_history": ["approved"],
        }
        return "approved"

    class Uav:
        uid = 7

    release = observer.wrap_release(original_release)
    assert release(env, Uav(), object(), (object(),)) == "approved"
    assert observer.candidate_plan_calls == 1
    assert observer.landing_zone_pair_checks == 1
    assert observer.suffix_plan_pairs == 1
    assert observer.suffix_proposal_checks == 1
    assert observer.approvals_rank_gt3 == 1
    assert observer.release_records[0]["runtime_candidates_checked_delta"] == 4


def test_reward_comparison_requires_known_fields_to_be_observed() -> None:
    observer = SimpleNamespace(
        step_reward_totals=[-3.0],
        step_reward_components={"time": -2.5, "energy": -0.5},
        step_reward_component_rows=1,
    )
    record = {
        "summary": {"reward_total": -3.0, "reward_component_totals": {"time": -2.5, "energy": -0.5}},
        "metrics": {"release_rejection_penalty_observed": -0.25},
    }
    compared = profile._compare_rewards(record, observer, {"release_rejection_penalty": -0.25})
    assert compared["available"] is True
    assert compared["coverage_known"] is True
    assert compared["all_match"] is True
    missing = profile._compare_rewards(record, observer, {})
    assert missing["requires_match"] is True
    assert missing["coverage_known"] is False
    assert missing["all_match"] is None


def test_sidecar_registration_lookup_is_bound_to_step_and_env_now() -> None:
    audit = {"registrations": {"4|7": {"step": 4, "env_now": 8.0, "uid": 7, "order": list(range(25))}}}
    env = SimpleNamespace(step_count=4, now=8.0)
    assert prefix._registration_lookup(audit, env, 7)["order"] == list(range(25))
    env.now = 9.0
    with pytest.raises(prefix.PrefixFallbackContractError):
        prefix._registration_lookup(audit, env, 7)


def test_instance_alias_restores_after_exception() -> None:
    obj = SimpleNamespace(value=3)
    had, previous = profile._set_instance_method(obj, "method", lambda _self: 1)
    try:
        assert obj.method() == 1
        raise RuntimeError("fixture")
    except RuntimeError:
        profile._restore_instance_method(obj, "method", had, previous)
    assert not hasattr(obj, "method")


def test_validate_pilot_reuse_requires_completed_exact_reward_and_metrics(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    profile._write_json(output / "preflight.json", {"stage": profile.STAGE, "status": "PREFLIGHT_OK"})
    arm_root = output / "arms"
    arm_root.mkdir()
    binding = {
        "model_unchanged": True,
        "caller_rng_unchanged": True,
        "replay_exact": True,
        "replay_steps_exact": True,
        "lifecycle_valid": True,
        "input_action_tape_sha256_match": True,
        "checkpoint_sha256_match": True,
        "controller_sidecar_order_lookup": True,
        "reward_binding_match": True,
    }
    for arm in ("before_legacy", "before_prefix_fallback"):
        profile._write_json(arm_root / f"cell_01_{arm}.json", {
            "status": "completed", "binding_checks": binding,
            "passive_counters": {}, "timing": {}, "error": None,
            "comparison": {"all_match": True}, "reward_comparison": {"all_match": True},
            "cell": {"cell_id": 1}, "arm": arm,
        })
    preflight_sha = profile._sha256(output / "preflight.json")
    profile._write_json(output / "pilot.json", {"status": "PILOT_OK", "preflight_sha256": preflight_sha})
    rows = profile._validate_pilot_reuse(output, {})
    assert [row["arm"] for row in rows] == ["before_legacy", "before_prefix_fallback"]


def test_sync_mapping_is_applied_to_concrete_arm_and_sidecar_files(monkeypatch: pytest.MonkeyPatch) -> None:
    def map_leaf(path):
        path = Path(path)
        return Path("mirror") / path.name

    monkeypatch.setattr(profile, "_sync_path", map_leaf)
    spec = {"root": "/home/user/training_outputs/terminal/frozen"}
    assert profile._arm_path(spec, 1, "before_prefix_fallback") == Path("mirror/cell_01_before_prefix_fallback.json")
    assert profile._sidecar_path(spec, 1, "before_prefix_fallback") == Path("mirror/cell_01_before_prefix_fallback_orders.json")


def test_arm_sequence_stops_after_first_failed_arm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = []
    failed = {
        "status": "failed",
        "binding_checks": {"model_unchanged": False},
        "comparison": {"all_match": False},
        "reward_comparison": {"requires_match": True, "all_match": False},
    }
    completed = {
        "status": "completed",
        "binding_checks": {"model_unchanged": True},
        "comparison": {"all_match": True},
        "reward_comparison": {"requires_match": False, "all_match": None},
    }

    def fake_profile_one(_output, _config, item, phase):
        calls.append((item, phase))
        return failed if len(calls) == 1 else completed

    monkeypatch.setattr(profile, "_profile_one", fake_profile_one)
    rows = profile._run_arm_sequence(tmp_path, object(), [{"id": 1}, {"id": 2}], "pilot")
    assert len(rows) == 1
    assert len(calls) == 1
