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
