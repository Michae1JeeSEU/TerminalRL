"""Contract tests for the matched-state P3 gate-shift diagnostic.

These tests exercise analytic forwards and output contracts only.  They do not
create a terminal environment, replay a tape, or run a checkpoint evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.distributions import Categorical

from terminal_marl.training import diagnose_p3_matched_gate_shift_20261001 as runner
from terminal_marl.training import run_p3_trained_proposal_support_20261001 as support


class _DistributionModel:
    def __init__(self, gate_bias: float, proposal_logits: np.ndarray):
        self.gate_bias = float(gate_bias)
        self.proposal_logits = torch.as_tensor(proposal_logits, dtype=torch.float32)

    def distributions(self, observations: torch.Tensor):
        count = int(observations.shape[0])
        gate_logits = torch.zeros((count, 2), dtype=torch.float32)
        gate_logits[:, 1] = self.gate_bias
        proposal_logits = self.proposal_logits.reshape(1, -1).repeat(count, 1)
        return Categorical(logits=gate_logits), Categorical(logits=proposal_logits)


def _policy(gate_bias: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(
        device=torch.device("cpu"),
        observation_dim=runner.OBSERVATION_DIM,
        model=_DistributionModel(gate_bias, np.arange(25, dtype=np.float32)),
    )


def _row(step: int, uid: int, *, group: str = "outer", actor_mask: bool = True) -> dict:
    return {
        "step": step,
        "uid": uid,
        "arrival_id": uid,
        "phase": f"ring_{group}",
        "group": group,
        "soc": 0.8,
        "retry_blocked": False,
        "actor_mask": actor_mask,
        "world": np.zeros(runner.OBSERVATION_DIM, dtype=np.float32).tolist(),
        "gate_probs_before": [0.4, 0.6],
        "gate_probs_after": [0.2, 0.8],
        "gate_logits_before": np.log([0.4, 0.6]).tolist(),
        "gate_logits_after": np.log([0.2, 0.8]).tolist(),
        "gate_release_before": 0.6,
        "gate_release_after": 0.8,
        "gate_release_delta": 0.2,
        "proposal_q_before": (np.ones(25, dtype=np.float32) / 25).tolist(),
        "proposal_q_after": (np.ones(25, dtype=np.float32) / 25).tolist(),
        "proposal_qmax_before": 1 / 25,
        "proposal_qmax_after": 1 / 25,
        "proposal_entropy_before": 3.2,
        "proposal_entropy_after": 3.2,
        "proposal_max_abs_diff": 0.0,
        "gate_argmax_diff": False,
        "proposal_argmax_diff": False,
        "gate_logprob_tape": -0.5 if actor_mask else None,
        "proposal_logprob_tape": -3.2 if actor_mask else None,
        "outcome": {"outcome": "completed"},
    }


def test_analytic_gate_shift_keeps_proposal_distribution_and_rng() -> None:
    world = np.zeros((2, runner.OBSERVATION_DIM), dtype=np.float32)
    before = _policy(0.0)
    after = _policy(1.5)
    rng_before = support._rng_snapshot()
    before_snapshot = runner._distribution_snapshot(before, world)
    after_snapshot = runner._distribution_snapshot(after, world)
    assert support._rng_equal(rng_before, support._rng_snapshot())
    assert np.array_equal(before_snapshot["proposal_q"], after_snapshot["proposal_q"])
    assert np.array_equal(before_snapshot["proposal_logits"], after_snapshot["proposal_logits"])
    assert not np.array_equal(before_snapshot["gate_probs"], after_snapshot["gate_probs"])
    assert np.array_equal(before_snapshot["proposal_entropy"], after_snapshot["proposal_entropy"])
    forward_kl = runner._categorical_kl_from_log_probs(np.log([0.4, 0.6]), np.log([0.2, 0.8]))
    reverse_kl = runner._categorical_kl_from_log_probs(np.log([0.2, 0.8]), np.log([0.4, 0.6]))
    assert forward_kl > 0.0 and reverse_kl > 0.0


def test_world_dimension_and_actor_context_are_explicit() -> None:
    observations = {"2": np.zeros(88, dtype=np.float32), "1": np.ones(88, dtype=np.float32)}
    uids, world = runner._world_from_observations(observations)
    assert uids == ["1", "2"]
    assert world.shape == (2, 88)
    empty_uids, empty_world = runner._world_from_observations({})
    assert empty_uids == []
    assert empty_world.shape == (0, 88)
    with pytest.raises(runner.MatchedGateShiftError):
        runner._world_from_observations({"1": np.zeros(87, dtype=np.float32)})

    actor = SimpleNamespace(
        arrival_id=101,
        phase=SimpleNamespace(value="ring_outer"),
        ring_role=SimpleNamespace(value="outer"),
        soc=0.75,
        safety_retry_after=5.0,
    )
    env = SimpleNamespace(active={7: actor}, now=2.0)
    context = runner._actor_context(env, 7, {7: {"outcome": "completed"}})
    assert context["arrival_id"] == 101
    assert context["group"] == "outer"
    assert context["retry_blocked"] is True


def test_lp_and_replay_contract_reject_mismatch() -> None:
    runner._validate_tape_logprob(-1.0, -1.0 + 1e-7, label="gate")
    with pytest.raises(runner.MatchedGateShiftError, match="mismatch"):
        runner._validate_tape_logprob(-1.0, -0.5, label="gate")
    runner._require_replay_exact({"exact": True, "steps_replayed": 3}, 3)
    with pytest.raises(runner.MatchedGateShiftError, match="not exact"):
        runner._require_replay_exact({"exact": False, "steps_replayed": 3}, 3)
    with pytest.raises(runner.MatchedGateShiftError, match="step count"):
        runner._require_replay_exact({"exact": True, "steps_replayed": 2}, 3)


def test_changed_input_hash_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "input.json"
    path.write_text("{}\n", encoding="utf-8")
    actual = runner._sha256(path)
    assert actual != "registered-sha"
    with pytest.raises(runner.MatchedGateShiftError, match="SHA mismatch"):
        runner._require_sha(path, "registered-sha", "input")


def test_summary_groups_outcomes_and_nonindependent_rows() -> None:
    rows = [_row(0, 1), _row(1, 2, group="middle"), _row(2, 3, group="inner", actor_mask=False)]
    summary = runner._summary_rows(rows)
    assert summary["rows"] == 3
    assert summary["eligible_rows"] == 2
    assert summary["groups"]["middle"]["rows"] == 1
    assert summary["groups"]["inner"]["eligible_rows"] == 0
    assert summary["groups"]["outer"]["actor_eligible_gate_kl_before_to_after_mean"] > 0.0
    assert summary["groups"]["inner"]["actor_eligible_gate_kl_before_to_after_mean"] is None
    assert summary["actual_outcomes"] == {"completed": 3}
    assert summary["repeated_actor_rows_are_not_independent_trials"] is True


def test_npz_serialization_has_step_offsets_and_nan_sentinels(tmp_path: Path) -> None:
    rows = [_row(0, 1), _row(0, 2, actor_mask=False), _row(1, 1)]
    rows[1]["gate_logprob_tape"] = None
    rows[1]["proposal_logprob_tape"] = None
    path = tmp_path / "matched_states.npz"
    runner._save_npz(path, rows, expected_steps=2, metadata={"format": "test"}, team_rewards=[5.0, 6.0])
    with np.load(path, allow_pickle=False) as data:
        assert data["world"].shape == (3, 88)
        assert data["step_offsets"].tolist() == [0, 2, 3]
        assert np.isnan(data["gate_logprob"][1])
        assert data["actor_mask"].tolist() == [True, False, True]
        assert data["team_reward_raw"].tolist() == [5.0, 6.0]
        assert json.loads(str(data["metadata_json"].item()))["format"] == "test"


def test_npz_preserves_native_string_uid_order_within_step(tmp_path: Path) -> None:
    first = _row(0, 10)
    second = _row(0, 2)
    first["world"][0] = 10.0
    second["world"][0] = 2.0
    path = tmp_path / "uid_order.npz"
    runner._save_npz(path, [first, second], expected_steps=1,
                     metadata={"format": "uid-order"}, team_rewards=[0.0])
    with np.load(path, allow_pickle=False) as data:
        assert data["uid"].tolist() == [10, 2]
        assert data["world"][:, 0].tolist() == [10.0, 2.0]


def test_preflight_output_is_no_environment_and_strict_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cell = {"cell_id": 1, "env_seed": 11, "policy_seed": 12}
    binding = {"runner_sha256": "runner", "protocol_sha256": "protocol"}
    arms = [{"cell": cell}]
    monkeypatch.setattr(runner, "_preflight_payload", lambda: (binding, arms, None, None, None))
    output = tmp_path / "preflight"
    result = runner.run_preflight(output)
    assert result["status"] == "PREFLIGHT_OK"
    assert result["scope"]["env_created"] is False
    assert json.loads((output / "preflight.json").read_text(encoding="utf-8"))["cells"] == [cell]
    assert json.loads((output / "progress.json").read_text(encoding="utf-8"))["status"] == "PREFLIGHT_OK"


def test_jsonable_nonfinite_values_are_missing_not_fabricated() -> None:
    value = runner._jsonable({"x": float("inf"), "nested": [float("nan")]})
    assert value == {"x": None, "nested": [None]}


class _ReplayActor:
    arrival_id = 17
    phase = SimpleNamespace(value="ring_outer")
    ring_role = SimpleNamespace(value="outer")
    soc = 0.8
    safety_retry_after = 0.0


class _ReplayEnv:
    instances: list["_ReplayEnv"] = []

    def __init__(self, config):
        self.config = config
        self.step_count = 0
        self.now = 0.0
        self.active = {7: _ReplayActor()}
        self.future_release_reservations = {}
        self.executed: list[dict] = []
        _ReplayEnv.instances.append(self)

    def _observations(self):
        if self.step_count == 0:
            return {"7": np.zeros(88, dtype=np.float32)}
        return {}

    def step(self, actions=None, proposal_shortlists=None):
        self.executed.append({"actions": actions, "shortlists": proposal_shortlists})
        rewards = {"7": 1.25} if self.step_count == 0 else {"7": 0.75}
        self.step_count += 1
        self.now += 1.0
        self.active = {} if self.step_count >= 1 else self.active
        return {}, rewards, {}, {}, {}


class _ReplayModel:
    def __init__(self, gate_bias: float):
        self.gate_bias = float(gate_bias)

    def distributions(self, observations: torch.Tensor):
        count = int(observations.shape[0])
        gate_logits = torch.tensor([[0.0, self.gate_bias]], dtype=torch.float32).repeat(count, 1)
        proposal_logits = torch.arange(25, dtype=torch.float32).reshape(1, -1).repeat(count, 1)
        return Categorical(logits=gate_logits), Categorical(logits=proposal_logits)


def _replay_policy(gate_bias: float) -> SimpleNamespace:
    return SimpleNamespace(device=torch.device("cpu"), observation_dim=88, model=_ReplayModel(gate_bias))


def _replay_item(tmp_path: Path, *, proposal: int | None = 3) -> tuple[dict, list[dict]]:
    observation_hash = runner._digest({"7": np.zeros(88, dtype=np.float32)})
    after = _replay_policy(1.0)
    world = np.zeros((1, 88), dtype=np.float32)
    after_dist = runner._distribution_snapshot(after, world)
    row0 = {
        "step": 0,
        "observation_hash": observation_hash,
        "requested_gate": {"7": 1},
        "requested_proposal": {"7": proposal},
        "gate_logprob": {"7": runner._categorical_logprob(after_dist["gate_logits"], 0, 1)},
        "proposal_logprob": {
            "7": (runner._categorical_logprob(after_dist["proposal_logits"], 0, proposal)
                   if proposal is not None else None)
        },
        "post_state_hash": "synthetic-post-0",
    }
    row1 = {
        "step": 1,
        "observation_hash": runner._digest({}),
        "requested_gate": {},
        "requested_proposal": {},
        "gate_logprob": {},
        "proposal_logprob": {},
        "post_state_hash": "synthetic-post-1",
    }
    rows = [row0, row1]
    item = {
        "cell": {"cell_id": 1, "env_seed": 1, "policy_seed": 2, "scenario": "synthetic"},
        "arm_path": str(tmp_path / "arm.json"),
        "tape_path": str(tmp_path / "tape.json"),
        "tape_sha256": "synthetic-tape-sha",
        "tape_rows": rows,
        "outcome_by_uid": {7: {"outcome": "completed"}},
        "arm": {"summary": {"reward_total": 2.0}},
    }
    return item, rows


def _install_replay_dependencies(monkeypatch: pytest.MonkeyPatch, rows: list[dict], *, exact: bool = True,
                                 consume_rng: bool = False) -> None:
    def details(_policy, _env, observations, *, deterministic, shortlist_k):
        if consume_rng:
            torch.rand(1)
        if not observations:
            return ([], np.zeros((0, 88), dtype=np.float32), np.zeros(0, dtype=bool), {},
                    np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0), np.zeros(0), {}, {})
        return (
            ["7"], np.zeros((1, 88), dtype=np.float32), np.asarray([True]),
            {}, np.asarray([1]), np.asarray([3]), np.asarray([0.0]), np.asarray([0.0]),
            {"7": {"gate": 0.0, "proposal": 0.0}}, {},
        )

    def replay(environment, _path):
        environment.step({"7": np.asarray([1.0, 0.0, 0.0], dtype=np.float32)}, {})
        environment.step({}, {})
        return {"exact": exact, "steps_replayed": 2, "steps_expected": 2,
                "reset_state_match": True, "config_match": True, "steps": []}

    monkeypatch.setattr(runner, "TerminalParallelEnv", _ReplayEnv)
    monkeypatch.setattr(runner.student, "_student_request_details", details)
    monkeypatch.setattr(runner.SafetyTraceRecorder, "replay_action_tape", replay)
    monkeypatch.setattr(runner.rank_eval, "_model_digest", lambda policy: "before" if policy is _BEFORE else "after")


_BEFORE = _replay_policy(0.0)
_AFTER = _replay_policy(1.0)


def test_replay_cell_synthetic_two_steps_and_npz_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    item, rows = _replay_item(tmp_path)
    _install_replay_dependencies(monkeypatch, rows)
    before_rng = support._rng_snapshot()
    result = runner._replay_cell(item, _BEFORE, _AFTER, SimpleNamespace(), tmp_path / "out")
    assert result["status"] == "COMPLETED"
    assert result["counts"] == {"replayed_steps": 2, "matched_rows": 1, "eligible_rows": 1,
                                  "team_reward_steps": 2, "team_reward_raw_total": 2.0}
    assert result["invariants"]["instance_step_hook_restored"] is True
    assert result["invariants"]["observed_raw_team_reward_captured"] is True
    assert result["invariants"]["synthetic_advantage_or_counterfactual_reward_computed"] is False
    assert support._rng_equal(before_rng, support._rng_snapshot())
    env = _ReplayEnv.instances[-1]
    assert len(env.executed) == 2
    assert set(env.executed[0]["actions"]) == {"7"}
    assert env.executed[1]["actions"] == {}
    npz_path = Path(result["artifacts"]["matched_states_npz"])
    assert result["artifacts"]["matched_states_npz_sha256"] == runner._sha256(npz_path)
    with np.load(npz_path, allow_pickle=False) as data:
        assert data["world"].shape == (1, 88)
        assert data["proposal_q_before"].shape == (1, 25)
        assert data["before_gate_logits"].shape == (1, 2)
        assert data["step_offsets"].tolist() == [0, 1, 1]
        assert data["team_reward_raw"].tolist() == [1.25, 0.75]


def test_replay_cell_missing_release_proposal_restores_hook_and_rng(tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    item, rows = _replay_item(tmp_path, proposal=None)
    _install_replay_dependencies(monkeypatch, rows)
    before_rng = support._rng_snapshot()
    with pytest.raises(runner.MatchedGateShiftError, match="RELEASE proposal"):
        runner._replay_cell(item, _BEFORE, _AFTER, SimpleNamespace(), tmp_path / "out")
    assert support._rng_equal(before_rng, support._rng_snapshot())
    env = _ReplayEnv.instances[-1]
    assert getattr(env.step, "__func__", None) is _ReplayEnv.step
    assert not (tmp_path / "out" / "cell_01_matched_states.npz").exists()


def test_replay_cell_detects_observer_rng_use_and_nonexact_replay(tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    item, rows = _replay_item(tmp_path)
    _install_replay_dependencies(monkeypatch, rows, consume_rng=True)
    before_rng = support._rng_snapshot()
    with pytest.raises(runner.MatchedGateShiftError, match="caller RNG"):
        runner._replay_cell(item, _BEFORE, _AFTER, SimpleNamespace(), tmp_path / "rng")
    assert support._rng_equal(before_rng, support._rng_snapshot())

    item2, rows2 = _replay_item(tmp_path / "nonexact")
    _install_replay_dependencies(monkeypatch, rows2, exact=False)
    with pytest.raises(runner.MatchedGateShiftError, match="not exact"):
        runner._replay_cell(item2, _BEFORE, _AFTER, SimpleNamespace(), tmp_path / "nonexact-out")
    assert not (tmp_path / "nonexact-out" / "cell_01_matched_states.npz").exists()


def test_replay_cell_rejects_reward_total_mismatch_after_exact_steps(tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    item, rows = _replay_item(tmp_path)
    item["arm"]["summary"]["reward_total"] = 3.0
    _install_replay_dependencies(monkeypatch, rows)
    before_rng = support._rng_snapshot()
    with pytest.raises(runner.MatchedGateShiftError, match="team reward total"):
        runner._replay_cell(item, _BEFORE, _AFTER, SimpleNamespace(), tmp_path / "mismatch")
    assert support._rng_equal(before_rng, support._rng_snapshot())
    assert not (tmp_path / "mismatch" / "cell_01_matched_states.npz").exists()
