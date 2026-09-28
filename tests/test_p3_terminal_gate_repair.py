from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.distributions import Categorical

from terminal_marl.training import train_p3_terminal_gate_repair as runner


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor_encoder = nn.Linear(3, 3, bias=False)
        self.base_gate = nn.Linear(3, 2, bias=False)
        self.gate_adapter = nn.Linear(3, 2, bias=False)
        self.proposal_head = nn.Linear(3, 3, bias=False)

    def distributions(self, observations: torch.Tensor) -> tuple[Categorical, Categorical]:
        encoded = self.actor_encoder(observations)
        gate = Categorical(logits=self.base_gate(encoded) + self.gate_adapter(observations))
        proposal = Categorical(logits=self.proposal_head(encoded))
        return gate, proposal


class _IndependentAdvantageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_adapter = nn.Linear(2, 2, bias=False)
        nn.init.zeros_(self.gate_adapter.weight)

    def distributions(self, observations: torch.Tensor) -> tuple[Categorical, Categorical]:
        gate = Categorical(logits=self.gate_adapter(observations))
        proposal = Categorical(logits=torch.zeros((len(observations), 2), device=observations.device))
        return gate, proposal


def _toy_policy() -> SimpleNamespace:
    torch.manual_seed(17)
    return SimpleNamespace(model=_ToyModel())


def _toy_batches() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    observations = torch.tensor(
        [[0.2, -0.4, 0.8], [-0.7, 0.1, 0.3], [0.5, 0.6, -0.2]], dtype=torch.float32
    )
    labels = torch.tensor([0, 1, 1], dtype=torch.long)
    weights = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float32)
    anchors = torch.tensor(
        [[0.1, 0.2, 0.3], [-0.3, 0.4, 0.5], [0.6, -0.2, 0.7]], dtype=torch.float32
    )
    return observations, labels, weights, anchors


def test_training_rows_uses_real_terminal_credit_schema_and_excludes_uncertain() -> None:
    root = Path(runner.TERMINAL_CREDIT_ROOT)
    if not (root / "selection.json").is_file():
        root = Path(r"D:\7_codespace\training_outputs\terminal\p3_terminal_credit_20260928\terminal_credit")
    if not (root / "selection.json").is_file():
        pytest.skip("completed terminal-credit artifacts are unavailable")
    selection, summary = runner._load_terminal_credit(root)
    rows = runner._training_rows(selection, summary)

    assert [row["state_index"] for row in rows] == [1, 2, 3, 6, 11]
    assert [row["label"] for row in rows] == ["wait", "wait", "release", "release", "release"]
    assert 0 not in {row["state_index"] for row in rows}
    assert sum(row["weight"] for row in rows) == pytest.approx(1.0)
    assert all(row["observation"].shape == (88,) for row in rows)
    assert all(np.isfinite(row["observation"]).all() for row in rows)
    assert all(row["repeat_improvements_seconds"][0] >= row["raw_weight"] for row in rows)
    for row in rows:
        assert (row["seed"], row["step"], row["uid"], row["arrival_id"]) in {
            (
                int(selected["seed"]),
                int(selected["step"]),
                int(selected["uid"]),
                int(selected["arrival_id"]),
            )
            for selected in selection["selected_rows"]
        }


def test_kl_violation_restores_model_and_optimizer_state() -> None:
    policy = _toy_policy()
    observations, labels, weights, anchors = _toy_batches()
    before = runner.coverage_fit._state_cpu(policy.model)

    result = runner._run_updates(
        policy,
        observations,
        labels,
        weights,
        anchors,
        max_updates=1,
        kl_mean_limit=-1.0,
        kl_max_limit=-1.0,
    )

    after = runner.coverage_fit._state_cpu(policy.model)
    assert result["attempted_updates"] == 1
    assert result["accepted_updates"] == 0
    assert result["rollback"] is True
    assert result["stop_reason"] == "kl_cap"
    assert runner.coverage_fit._state_equal(before, after)
    assert policy.optimizer.state_dict()["state"] == {}


def test_only_gate_adapter_changes_and_checkpoint_has_no_optimizer(tmp_path: Path) -> None:
    policy = _toy_policy()
    observations, labels, weights, anchors = _toy_batches()
    before = runner.coverage_fit._state_cpu(policy.model)
    proposal_before = runner._proposal_logits(policy, anchors).detach().clone()

    result = runner._run_updates(
        policy,
        observations,
        labels,
        weights,
        anchors,
        max_updates=1,
        kl_mean_limit=1e9,
        kl_max_limit=1e9,
    )
    after = runner.coverage_fit._state_cpu(policy.model)

    assert result["accepted_updates"] == 1
    assert result["trainable_parameter_names"] == ["gate_adapter.weight"]
    assert not policy.model.actor_encoder.weight.requires_grad
    assert not policy.model.base_gate.weight.requires_grad
    assert not policy.model.proposal_head.weight.requires_grad
    assert torch.equal(before["actor_encoder.weight"], after["actor_encoder.weight"])
    assert torch.equal(before["base_gate.weight"], after["base_gate.weight"])
    assert torch.equal(proposal_before, runner._proposal_logits(policy, anchors).detach())
    assert not torch.equal(before["gate_adapter.weight"], after["gate_adapter.weight"])

    checkpoint = tmp_path / "terminal_gate_repair.pt"
    runner._save_checkpoint(
        checkpoint,
        policy,
        {"stage": runner.STAGE, "source_checkpoint_sha256": "source"},
        {"status": "COMPLETED", "accepted_updates": 1},
    )
    payload = torch.load(checkpoint, map_location="cpu")
    assert set(payload) == {"metadata", "model", "report"}
    assert "optimizer" not in payload
    assert torch.equal(payload["model"]["gate_adapter.weight"], after["gate_adapter.weight"])


def _optimizer_state_without_learning_rate(policy: SimpleNamespace) -> dict[str, object]:
    state = policy.optimizer.state_dict()
    return {
        "state": state["state"],
        "param_groups": [
            {key: value for key, value in group.items() if key != "lr"}
            for group in state["param_groups"]
        ],
    }


def _assert_nested_optimizer_state_equal(left: dict[str, object], right: dict[str, object]) -> None:
    assert left["param_groups"] == right["param_groups"]
    left_state = left["state"]
    right_state = right["state"]
    assert set(left_state) == set(right_state)
    for key in left_state:
        assert set(left_state[key]) == set(right_state[key])
        for field in left_state[key]:
            left_value = left_state[key][field]
            right_value = right_state[key][field]
            if isinstance(left_value, torch.Tensor):
                assert torch.equal(left_value, right_value)
            else:
                assert left_value == right_value


def test_bounded_backtracking_accepts_first_kl_safe_smaller_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = _toy_policy()
    observations, labels, weights, anchors = _toy_batches()
    kl_values = iter(((0.002, 0.02), (0.0001, 0.005)))
    monkeypatch.setattr(runner, "_anchor_kl", lambda *_args: next(kl_values))

    result = runner._run_updates(
        policy,
        observations,
        labels,
        weights,
        anchors,
        max_updates=1,
        bounded_backtracking=True,
        kl_mean_limit=runner.KL_MEAN_LIMIT,
        kl_max_limit=runner.KL_MAX_LIMIT,
    )

    entry = result["history"][0]
    assert result["attempted_updates"] == 1
    assert result["optimizer_attempts"] == 2
    assert result["accepted_updates"] == 1
    assert entry["accepted_scale"] == pytest.approx(0.5)
    assert entry["rollback"] is False
    assert [trial["scale"] for trial in entry["trials"]] == [1.0, 0.5]
    assert [trial["accepted"] for trial in entry["trials"]] == [False, True]
    assert result["bounded_backtracking"] is True

    reference = _toy_policy()
    monkeypatch.setattr(runner, "_anchor_kl", lambda *_args: (0.0, 0.0))
    runner._run_updates(
        reference,
        observations,
        labels,
        weights,
        anchors,
        max_updates=1,
        learning_rate=runner.LEARNING_RATE * 0.5,
        kl_mean_limit=runner.KL_MEAN_LIMIT,
        kl_max_limit=runner.KL_MAX_LIMIT,
    )
    assert runner.coverage_fit._state_equal(
        runner.coverage_fit._state_cpu(policy.model),
        runner.coverage_fit._state_cpu(reference.model),
    )
    _assert_nested_optimizer_state_equal(
        _optimizer_state_without_learning_rate(policy),
        _optimizer_state_without_learning_rate(reference),
    )


def test_bounded_backtracking_restores_full_model_and_optimizer_after_all_rejections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _toy_policy()
    observations, labels, weights, anchors = _toy_batches()
    kl_values = iter(((0.0, 0.0),) + ((0.002, 0.02),) * 5)
    monkeypatch.setattr(runner, "_anchor_kl", lambda *_args: next(kl_values))

    result = runner._run_updates(
        policy,
        observations,
        labels,
        weights,
        anchors,
        max_updates=2,
        bounded_backtracking=True,
        kl_mean_limit=runner.KL_MEAN_LIMIT,
        kl_max_limit=runner.KL_MAX_LIMIT,
    )

    entry = result["history"][1]
    reference = _toy_policy()
    monkeypatch.setattr(runner, "_anchor_kl", lambda *_args: (0.0, 0.0))
    reference_result = runner._run_updates(
        reference,
        observations,
        labels,
        weights,
        anchors,
        max_updates=1,
        bounded_backtracking=True,
        kl_mean_limit=runner.KL_MEAN_LIMIT,
        kl_max_limit=runner.KL_MAX_LIMIT,
    )
    assert result["attempted_updates"] == 2
    assert result["optimizer_attempts"] == 6
    assert result["accepted_updates"] == 1
    assert result["stop_reason"] == "kl_cap"
    assert result["rollback"] is True
    assert result["history"][0]["accepted_scale"] == pytest.approx(1.0)
    assert entry["accepted_scale"] is None
    assert entry["rollback"] is True
    assert len(entry["trials"]) == 5
    assert all(trial["accepted"] is False for trial in entry["trials"])
    assert reference_result["accepted_updates"] == 1
    assert policy.optimizer.state_dict()["state"]
    assert runner.coverage_fit._state_equal(
        runner.coverage_fit._state_cpu(policy.model),
        runner.coverage_fit._state_cpu(reference.model),
    )
    _assert_nested_optimizer_state_equal(
        _optimizer_state_without_learning_rate(policy),
        _optimizer_state_without_learning_rate(reference),
    )


@pytest.mark.parametrize(
    ("deltas", "expected"),
    [
        ([1.0, 2.0, 3.0], 2.0),
        ([-1.0, -2.0, -3.0], -2.0),
        ([1.0, -1.0, 0.0], 0.0),
        ([0.0, 0.0, 0.0], 0.0),
    ],
)
def test_normal_advantage_sign_rules(deltas: list[float], expected: float) -> None:
    assert runner._normal_advantage_from_deltas(deltas) == pytest.approx(expected)


def _normal_advantage_fixture(*, missing_field: str | None = None) -> tuple[dict, dict]:
    selected = [
        {
            "seed": 873103,
            "step": step,
            "uid": 42,
            "arrival_id": 12,
            "observation": [float(step)] + [0.0] * 87,
        }
        for step in range(102, 105)
    ]
    deltas = ([1.0, 2.0, 3.0], [-1.0, -2.0, -3.0], [1.0, -1.0, 0.0])
    states = []
    for state_index, (row, values) in enumerate(zip(selected, deltas)):
        comparisons = [
            {
                "normal_delta": value,
                "failed_delta": 0,
                "baseline_failed": 0,
                "candidate_failed": 0,
                "technical_ok": True,
            }
            for value in values
        ]
        release = {
            "comparisons": comparisons,
            "normal_opportunity": {
                "normal_opportunity": True,
                "normal_deltas": list(values),
                "technical_ok": True,
            },
        }
        if missing_field is not None:
            release["comparisons"][0].pop(missing_field, None)
        states.append({
            "state_index": state_index,
            "seed": row["seed"],
            "step": row["step"],
            "uid": row["uid"],
            "arrival_id": row["arrival_id"],
            "repeat_count": 3,
            "gate": {"release": release},
        })
    return {"selected_rows": selected}, {"states": states}


def test_normal_advantage_rows_use_fixed_window_and_report_only_teacher_gate() -> None:
    selection, summary = _normal_advantage_fixture()
    rows = runner._normal_advantage_rows(selection, summary)
    assert [row["step"] for row in rows] == [102, 103, 104]
    assert [row["advantage"] for row in rows] == pytest.approx([2.0, -2.0, 0.0])
    assert [row["teacher_gate"] for row in rows] == [1, 0, None]
    assert all(row["weight"] == pytest.approx(1.0 / 3.0) for row in rows)


def test_normal_advantage_rows_reject_missing_comparison_field() -> None:
    selection, summary = _normal_advantage_fixture(missing_field="technical_ok")
    with pytest.raises(runner.TerminalGateRepairError, match="technical_ok"):
        runner._normal_advantage_rows(selection, summary)


def test_normal_advantage_rows_refuse_without_positive_advantage() -> None:
    selection, summary = _normal_advantage_fixture()
    for state in summary["states"]:
        comparisons = state["gate"]["release"]["comparisons"]
        for comparison in comparisons:
            comparison["normal_delta"] = -1.0
        state["gate"]["release"]["normal_opportunity"]["normal_deltas"] = [-1.0, -1.0, -1.0]
    with pytest.raises(runner.TerminalGateRepairError, match="no positive normal advantage"):
        runner._normal_advantage_rows(selection, summary)


def test_normal_advantage_requires_bounded_backtracking(tmp_path: Path) -> None:
    with pytest.raises(runner.TerminalGateRepairError, match="bounded backtracking"):
        runner._fit(
            output_dir=tmp_path / "fit",
            terminal_credit_root=tmp_path / "credit",
            formal_root=tmp_path / "formal",
            anchor_path=tmp_path / "anchors.npz",
            smoke=True,
            bounded_backtracking=False,
            normal_advantage=True,
        )


def test_normal_advantage_objective_is_linear_probability_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = _toy_policy()
    observations, _labels, _weights, anchors = _toy_batches()
    source_probabilities = runner._gate_release_probabilities(policy, observations).detach().clone()
    advantages = torch.tensor([0.001, -0.001, 0.0])
    monkeypatch.setattr(runner, "_anchor_kl_tensor", lambda _source, candidate: candidate.sum(dim=-1) * 0.0)
    objective, metrics = runner._training_objective(
        policy,
        observations,
        None,
        None,
        anchors,
        runner._gate_logits(policy, anchors).detach().clone(),
        advantages=advantages,
        source_release_probabilities=source_probabilities,
    )
    expected = -torch.mean(advantages * (torch.tensor(metrics["release_probabilities"]) - source_probabilities))
    assert metrics["weighted_ce"] is None
    assert metrics["normal_advantage_term"] == pytest.approx(float(expected))
    assert float(objective.detach()) == pytest.approx(float(expected))


def test_normal_advantage_small_probability_gap_is_not_inverse_probability_scaled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _toy_policy()
    observations = torch.zeros((1, 3))
    source_probabilities = torch.tensor([0.990])
    advantages = torch.tensor([0.001])
    fixed_logits = torch.log(torch.tensor([[0.009, 0.991]]))
    monkeypatch.setattr(runner, "_gate_logits", lambda _policy, batch: fixed_logits.repeat(len(batch), 1))
    monkeypatch.setattr(runner, "_anchor_kl_tensor", lambda _source, candidate: candidate.sum(dim=-1) * 0.0)
    objective, metrics = runner._training_objective(
        policy,
        observations,
        None,
        None,
        observations,
        fixed_logits,
        advantages=advantages,
        source_release_probabilities=source_probabilities,
    )
    expected = -0.001 * (0.991 - 0.990)
    assert metrics["normal_advantage_term"] == pytest.approx(expected, abs=1e-9)
    assert float(objective.detach()) == pytest.approx(expected, abs=1e-9)
    assert abs(float(objective.detach())) < 1e-4


def test_normal_advantage_gradient_moves_release_probability_by_sign() -> None:
    torch.manual_seed(5)
    policy = SimpleNamespace(model=_IndependentAdvantageModel())
    observations = torch.eye(2)
    source_probabilities = torch.tensor([0.5, 0.5])
    advantages = torch.tensor([1.0, -1.0])
    result = runner._run_updates(
        policy,
        observations,
        None,
        None,
        observations.clone(),
        max_updates=1,
        kl_mean_limit=1e9,
        kl_max_limit=1e9,
        advantages=advantages,
        source_release_probabilities=source_probabilities,
    )
    assert result["accepted_updates"] == 1
    probabilities = runner._gate_release_probabilities(policy, observations).detach()
    assert probabilities[0] > source_probabilities[0]
    assert probabilities[1] < source_probabilities[1]
