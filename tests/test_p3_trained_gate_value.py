"""Contract tests for the registered P3 trained gate-value diagnostic.

The tests use small synthetic branch reports and NPZ metadata.  They do not
construct a terminal environment, replay a tape, load a checkpoint, or run a
physical/simulation episode.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from terminal_marl.training import diagnose_p3_trained_gate_value_20261001 as runner


def _branch(value: float, *, terminal: bool = True, censored: bool = False,
            error: str | None = None, approved: bool = True,
            outcome: str = "completed") -> dict:
    return {
        "branch": "release_0" if approved else "wait",
        "proposal": 0 if approved else None,
        "terminal": terminal,
        "censored": censored,
        "error": error,
        "safety_violation": False,
        "safety_invariant_error": False,
        "termination_classification": "natural" if terminal else "censored",
        "undiscounted_return": float(value),
        "first_step": {"actual_immediate_approval": bool(approved)},
        "outcome_counts": {outcome: 1},
        "energy_outcomes": [{"arrival_id": 1, "energy_consumed": 2.0}],
        "request_elapsed_integral_seconds": 4.0,
    }


def _paired_report(*, complete: bool = True, bad_branch: str | None = None) -> dict:
    q = np.arange(1, 26, dtype=np.float64)
    q /= q.sum()
    waits = []
    releases = []
    for seed_index in range(4):
        wait = _branch(10.0 + seed_index, approved=False)
        if bad_branch == "wait":
            wait["terminal"] = False
            wait["censored"] = True
        waits.append(wait)
        for proposal in range(25):
            row = _branch(20.0 + seed_index + proposal, approved=True)
            row["proposal"] = proposal
            row["branch"] = f"release_{proposal}"
            if bad_branch == "release" and seed_index == 0 and proposal == 3:
                row["error"] = "synthetic branch error"
                row["terminal"] = False
            releases.append(row)
    return {
        "proposal_q": q.tolist(),
        "continuation_seeds": list(runner.CONTINUATION_SEEDS),
        "wait": waits,
        "releases": releases,
        "terminal_complete": bool(complete),
        "external_common_prefix": {"valid": bool(complete)},
    }


def test_summary_uses_normalized_q_and_includes_first_reward_in_d() -> None:
    summary = runner._summarize_paired(_paired_report())
    q = np.arange(1, 26, dtype=np.float64)
    q /= q.sum()
    assert np.allclose(summary["proposal_q"], q)
    # Per seed: weighted release (20 + seed + proposal) minus WAIT (10 + seed).
    expected = float(np.dot(q, 20.0 + np.arange(25)) - 10.0)
    assert summary["D"] == pytest.approx(expected)
    assert summary["C"] == pytest.approx(1.0)
    assert all(item["complete"] for item in summary["per_seed"])
    assert summary["qualified"] is True
    assert summary["outcome_counts_all_branches"]["completed"] == 104


@pytest.mark.parametrize("bad_branch", ["wait", "release"])
def test_summary_preserves_incomplete_branches_and_withholds_d(bad_branch: str) -> None:
    summary = runner._summarize_paired(_paired_report(complete=False, bad_branch=bad_branch))
    assert summary["D"] is None
    assert summary["qualified"] is False
    assert any(item["terminal"] is False for item in summary["branch_metrics"])


def test_summary_keeps_business_failed_outcomes_in_branch_metrics() -> None:
    report = _paired_report()
    report["releases"][0]["outcome_counts"] = {"failed_energy": 1}
    summary = runner._summarize_paired(report)
    assert summary["outcome_counts_all_branches"]["failed_energy"] == 1
    assert summary["branch_metrics"][4]["outcome_counts"] == {"failed_energy": 1}


def test_first_actions_and_shortlists_require_non_target_rows_to_match() -> None:
    base_actions = {"1": [0.0, 0.1, 0.2], "2": [1.0, 0.2, 0.3]}
    base_shortlists = {"1": ([1.0] * 88,), "2": ([2.0] * 88,)}
    report = {"branches": [{"first_step": {
        "actions": {"1": [0.0, 0.1, 0.2], "2": [9.0, 0.2, 0.3]},
        "proposal_shortlists": {"1": ([1.0] * 88,), "2": ([2.0] * 88,)},
    }}]}
    assert runner._first_actions_unchanged(report, base_actions, base_shortlists, 2)
    assert not runner._first_actions_unchanged(report, base_actions, base_shortlists, 1)


def _write_npz(path: Path, *, step: int = 168) -> None:
    metadata = {
        "format": "matched_gate_shift_v1",
        "reward_scale_applied": False,
        "selection_rule": "first chronological actor-eligible inner state",
    }
    rows = 2
    probs = np.full((rows, 2), 0.5, dtype=np.float64)
    q = np.full((rows, 25), 1 / 25, dtype=np.float64)
    np.savez(
        path,
        world=np.zeros((rows, 88), dtype=np.float32),
        step=np.asarray([step, step + 1], dtype=np.int64),
        uid=np.asarray([3, 4], dtype=np.int64),
        arrival_id=np.asarray([3, 4], dtype=np.int64),
        phase=np.asarray(["ring_inner", "ring_outer"]),
        soc=np.asarray([0.45, 0.8], dtype=np.float64),
        actor_mask=np.asarray([True, False]),
        retry_blocked=np.asarray([False, False]),
        before_gate_probs=probs,
        after_gate_probs=probs,
        proposal_q_before=q,
        proposal_q_after=q,
        metadata_json=np.asarray(json.dumps(metadata)),
    )


def test_npz_selection_enforces_registered_state_and_schema(tmp_path: Path) -> None:
    path = tmp_path / "state.npz"
    _write_npz(path)
    selection = runner._load_npz_state(path, 1)
    assert (selection["step"], selection["uid"], selection["arrival_id"]) == (168, 3, 3)
    assert selection["phase"] == "ring_inner"


def test_npz_selection_rejects_changed_locked_step(tmp_path: Path) -> None:
    path = tmp_path / "state_changed.npz"
    _write_npz(path, step=169)
    with pytest.raises(runner.TrainedGateValueError, match="registered state changed"):
        runner._load_npz_state(path, 1)


def test_pilot_reuse_checks_artifact_hash_and_identity(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    preflight = {"status": "PREFLIGHT_OK"}
    (output / "preflight.json").write_text(json.dumps(preflight), encoding="utf-8")
    state = {"cell": {"cell_id": 1}}
    state_path = output / "state_01.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    pilot = {
        "status": "PILOT_OK",
        "preflight_sha256": runner._sha256(output / "preflight.json"),
        "state_sha256": runner._sha256(state_path),
        "state": state,
    }
    (output / "pilot.json").write_text(json.dumps(pilot), encoding="utf-8")
    assert runner._validate_pilot(output, preflight)["status"] == "PILOT_OK"
    state_path.write_text(json.dumps({"cell": {"cell_id": 2}}), encoding="utf-8")
    with pytest.raises(runner.TrainedGateValueError, match="state changed"):
        runner._validate_pilot(output, preflight)


def test_preflight_writer_is_no_environment_and_strict_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"stage": runner.STAGE, "status": "PREFLIGHT_OK", "cells": [], "scope": {"env_created": False}}
    monkeypatch.setattr(runner, "_build_preflight", lambda: payload)
    output = tmp_path / "preflight"
    result = runner.run_preflight(output)
    assert result["scope"]["env_created"] is False
    assert json.loads((output / "preflight.json").read_text(encoding="utf-8"))["status"] == "PREFLIGHT_OK"
    assert (output / "progress.json").is_file()


def test_preflight_path_binding_roundtrip_and_phase_validation(tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    config = SimpleNamespace()
    binding = {
        "matched_full_path": Path("C:/registered/matched/full.json"),
        "matched_full_sha256": "matched-sha",
        "native_full_path": Path("C:/registered/native/full.json"),
        "native_full_sha256": "native-sha",
    }
    payload = {
        "stage": runner.STAGE, "phase": "preflight", "status": "PREFLIGHT_OK",
        "matched_binding": binding, "cells": [], "source": {}, "scope": {"env_created": False},
        "config_snapshot": {"config": "fixed"},
        "runner_sha256": runner._sha256(Path(runner.__file__)),
        "protocol_sha256": runner._sha256(runner.PROTOCOL),
    }
    monkeypatch.setattr(runner, "_build_preflight", lambda: payload)
    monkeypatch.setattr(runner, "_load_input_bundle", lambda: (binding, []))
    monkeypatch.setattr(runner.trained_support, "_build_inputs", lambda: (config, {}))
    monkeypatch.setattr(runner.trained_support, "_config_snapshot", lambda value: {"config": "fixed"})
    monkeypatch.setattr(runner, "_load_policies", lambda value: (SimpleNamespace(), SimpleNamespace(), {}))
    output = tmp_path / "roundtrip"
    runner.run_preflight(output)
    loaded = runner._read_json(output / "preflight.json")
    runner._validate_preflight(loaded)
    runner._phase_inputs(output)
    for key, changed in (("matched_full_path", "C:/changed/full.json"),
                         ("matched_full_sha256", "changed-sha")):
        tampered = json.loads(json.dumps(loaded))
        tampered["matched_binding"][key] = changed
        with pytest.raises(runner.TrainedGateValueError, match="matched/native full binding changed"):
            runner._validate_preflight(tampered)


def test_pilot_preflight_failure_writes_error_without_overwriting_existing_error(tmp_path: Path,
                                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "pilot_failure"
    output.mkdir()
    (output / "preflight.json").write_text("{}", encoding="utf-8")
    (output / "progress.json").write_text(json.dumps({"status": "PREFLIGHT_OK"}), encoding="utf-8")
    existing = "{\"preserved\":true}\n"
    (output / "error.json").write_text(existing, encoding="utf-8")
    monkeypatch.setattr(runner, "_phase_inputs", lambda value: (_ for _ in ()).throw(
        runner.TrainedGateValueError("binding changed")))
    with pytest.raises(runner.TrainedGateValueError, match="binding changed"):
        runner.run_pilot(output)
    assert (output / "error.json").read_text(encoding="utf-8") == existing
    assert json.loads((output / "progress.json").read_text(encoding="utf-8"))["status"] == "FAILED"


def test_jsonable_nonfinite_values_are_missing() -> None:
    assert runner._jsonable({"x": float("inf"), "y": [float("nan")]}) == {"x": None, "y": [None]}


def test_locked_constants_are_fixed_and_pilot_has_no_population_claim() -> None:
    assert runner.TARGET_STEPS == (168, 202, 198, 187)
    assert runner.TARGET_UIDS == (3, 2, 2, 3)
    assert runner.CONTINUATION_SEEDS == (9303201, 9303202, 9303203, 9303204)
    assert runner.PPO_ALLOWED is False


def test_arrival_keyed_scope_adapts_seed_and_restores_hook(monkeypatch: pytest.MonkeyPatch) -> None:
    original = runner.paired.keyed_request
    calls: list[int] = []

    def fake_keyed(policy, env, keyed_seed):
        calls.append(int(keyed_seed))
        return {"actions": {"7": np.zeros(3, dtype=np.float32)},
                "shortlists": {}, "step": 12, "keyed_seed": int(keyed_seed)}

    monkeypatch.setattr(runner.frozen_wait, "_keyed_request", fake_keyed)
    env = SimpleNamespace(_paired_seed=9303201)
    with runner._arrival_keyed_continuation_scope():
        request = runner.paired.keyed_request(SimpleNamespace(), env)
        assert request.actions["7"].tolist() == [0.0, 0.0, 0.0]
        assert request.details["keyed_seed"] == 9303201
    assert calls == [9303201]
    assert runner.paired.keyed_request is original


def test_arrival_keyed_scope_restores_hook_on_error() -> None:
    original = runner.paired.keyed_request
    try:
        with pytest.raises(runner.TrainedGateValueError, match="seed is missing"):
            with runner._arrival_keyed_continuation_scope():
                runner.paired.keyed_request(SimpleNamespace(), SimpleNamespace())
    finally:
        assert runner.paired.keyed_request is original


def test_registered_arrival_keying_survives_uid_renumbering(monkeypatch: pytest.MonkeyPatch) -> None:
    class Actor:
        def __init__(self, arrival_id: int):
            self.arrival_id = arrival_id

    def context(_policy, env):
        uids = [str(uid) for uid in sorted(env.active)]
        return {
            "uids": uids,
            "world": np.zeros((2, 88), dtype=np.float32),
            "actor_mask": np.asarray([True, True]),
            "step": 17,
            "gate_probabilities": np.asarray([[0.8, 0.2], [0.7, 0.3]], dtype=np.float64),
            "proposal_probabilities": np.full((2, 25), 1 / 25, dtype=np.float64),
            "proposal_logits": np.tile(np.arange(25, dtype=np.float64), (2, 1)),
        }

    monkeypatch.setattr(runner.paired, "_details_context", context)
    env_a = SimpleNamespace(_paired_seed=9303201, active={1: Actor(101), 2: Actor(102)})
    env_b = SimpleNamespace(_paired_seed=9303201, active={11: Actor(101), 22: Actor(102)})
    before = runner._rng_snapshot()
    result_a = runner.frozen_wait._keyed_request(SimpleNamespace(), env_a, 9303201)
    result_b = runner.frozen_wait._keyed_request(SimpleNamespace(), env_b, 9303201)
    assert runner._rng_equal(before, runner._rng_snapshot())
    rows_a = {row["arrival_id"]: row for row in result_a["rows"]}
    rows_b = {row["arrival_id"]: row for row in result_b["rows"]}
    assert rows_a.keys() == rows_b.keys()
    for arrival_id in rows_a:
        assert rows_a[arrival_id]["gate"] == rows_b[arrival_id]["gate"]
        assert rows_a[arrival_id]["proposal"] == rows_b[arrival_id]["proposal"]
        assert rows_a[arrival_id]["action"] == rows_b[arrival_id]["action"]


def _synthetic_run_state_inputs(tmp_path: Path) -> tuple[dict, SimpleNamespace, SimpleNamespace]:
    q = np.full((1, 25), 1 / 25, dtype=np.float64)
    context = {"uids": ["7"], "proposal_probabilities": q, "proposal_logits": np.zeros((1, 25))}
    captured = {
        "snapshot": SimpleNamespace(), "selection": {"step": 4, "uid": 7, "arrival_id": 7},
        "details": context, "world": np.zeros((1, 88), dtype=np.float32),
        "first_actions": {"7": np.zeros(3, dtype=np.float32)}, "first_shortlists": {},
        "npz_step_rows": {"before_gate_probs": np.asarray([[0.4, 0.6]]),
                           "after_gate_probs": np.asarray([[0.3, 0.7]]),
                           "proposal_q_before": q},
        "replay": {"exact": True}, "selected_replay_check": {"post_state_match": True},
        "arrival_prefix": {"generated": 24, "exhausted": True, "future_arrivals_pending": False},
        "caller_rng_unchanged": True,
    }
    item = {"cell": {"cell_id": 1}, "group_path": "group.json", "group_sha256": "g",
            "npz_path": "state.npz", "npz_sha256": "n", "tape_path": "tape.json", "tape_sha256": "t"}
    return {"item": item, "captured": captured, "context": context}, SimpleNamespace(marker="before"), SimpleNamespace(marker="after")


def test_run_state_serializes_callback_and_keeps_observed_branch(tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    bundle, before, after = _synthetic_run_state_inputs(tmp_path)
    monkeypatch.setattr(runner, "_capture_state", lambda *args, **kwargs: bundle["captured"])
    monkeypatch.setattr(runner.paired, "_context_target", lambda context, uid: (0, str(uid)))
    monkeypatch.setattr(runner.paired, "gate_update_alignment", lambda *args, **kwargs: {
        "proposal_probability_unchanged": True, "model_unchanged": True, "rng_unchanged": True,
        "before_gate_release_probability": [0.6], "after_gate_release_probability": [0.7],
        "linear_approximation": [0.05], "actual_probability_delta": [0.1],
        "exact_linear_residual": [0.05],
    })
    monkeypatch.setattr(runner.rank_eval, "_model_digest", lambda policy: policy.marker)
    monkeypatch.setattr(runner, "_first_actions_unchanged", lambda *args, **kwargs: True)
    monkeypatch.setattr(runner, "_summarize_paired", lambda report: {
        "C": 0.5, "D": 2.0, "qualified": True, "per_seed": [], "branch_metrics": [],
        "outcome_counts_all_branches": {}, "all_branches_natural": True,
    })

    def fake_paired(*args, **kwargs):
        kwargs["branch_callback"]({"branch": "wait", "terminal": True}, 1, 1)
        return {"branches": [{"error": None, "safety_violation": False,
                               "safety_invariant_error": False}],
                "model_unchanged": True, "env_snapshot_unchanged": True, "rng_unchanged": True,
                "external_common_prefix": {"valid": True}}

    monkeypatch.setattr(runner.paired, "paired_gate_value", fake_paired)
    result = runner._run_state(bundle["item"], SimpleNamespace(), before, after, tmp_path, "pilot")
    assert result["local_m"] == pytest.approx(0.2)
    assert result["alignment"]["exact_linear_residual"] == pytest.approx(0.05)
    branch = tmp_path / "branches" / "cell_01_001.json"
    assert branch.is_file()
    assert json.loads(branch.read_text(encoding="utf-8"))["continuation_seed"] == runner.CONTINUATION_SEEDS[0]


def test_run_state_stops_on_hard_branch_and_preserves_artifact(tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    bundle, before, after = _synthetic_run_state_inputs(tmp_path)
    original_hook = runner.paired.keyed_request
    monkeypatch.setattr(runner, "_capture_state", lambda *args, **kwargs: bundle["captured"])
    monkeypatch.setattr(runner.paired, "_context_target", lambda context, uid: (0, str(uid)))
    monkeypatch.setattr(runner.paired, "gate_update_alignment", lambda *args, **kwargs: {
        "proposal_probability_unchanged": True, "model_unchanged": True, "rng_unchanged": True,
        "before_gate_release_probability": [0.6], "after_gate_release_probability": [0.7],
        "linear_approximation": [0.05], "actual_probability_delta": [0.1],
        "exact_linear_residual": [0.05],
    })
    monkeypatch.setattr(runner.rank_eval, "_model_digest", lambda policy: policy.marker)

    def fake_paired(*args, **kwargs):
        kwargs["branch_callback"]({"branch": "release_0", "terminal": False,
                                    "error": {"type": "hard"}}, 1, 104)
        raise AssertionError("callback must stop before the next branch")

    monkeypatch.setattr(runner.paired, "paired_gate_value", fake_paired)
    with pytest.raises(runner.TrainedGateValueError, match="technical or hard-safety"):
        runner._run_state(bundle["item"], SimpleNamespace(), before, after, tmp_path, "pilot")
    assert (tmp_path / "branches" / "cell_01_001.json").is_file()
    assert runner.paired.keyed_request is original_hook
