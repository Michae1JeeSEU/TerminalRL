from __future__ import annotations

import pytest
import torch

from terminal_marl.training.k3_coverage_objective import (
    K3CoverageObjectiveInputError,
    k3_coverage_objective,
)
from terminal_marl.training.k3_gate_feasibility import k3_certified_mass


def test_missed_row_has_finite_gradient_only_on_best_certified_logit() -> None:
    logits = torch.full((1, 25), -20.0, dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        logits[0, 0] = 80.0
        logits[0, 1] = 79.0
        logits[0, 2] = 1.0
        logits[0, 5] = 0.0

    result = k3_coverage_objective(logits, [[5]])
    result["loss"].backward()

    assert result["miss_count"] == 1
    assert torch.isfinite(result["loss"])
    assert torch.softmax(logits.detach(), dim=-1)[0, 5].item() < 1e-30
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 5].item() == pytest.approx(-1.0)
    assert torch.count_nonzero(logits.grad).item() == 1


def test_covered_and_empty_rows_are_zero_and_have_no_gradient() -> None:
    logits = torch.zeros((2, 25), dtype=torch.float32, requires_grad=True)
    result = k3_coverage_objective(logits, [[0], []])

    assert result["miss_count"] == 0
    assert torch.equal(result["row_loss"], torch.zeros(2))
    result["loss"].backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_constant_logit_shift_does_not_change_objective() -> None:
    base = torch.linspace(-3.0, 3.0, 25, dtype=torch.float32).reshape(1, 25)
    shifted = base + 17.25

    first = k3_coverage_objective(base, [[0]])
    second = k3_coverage_objective(shifted, [[0]])

    assert first["miss_count"] == second["miss_count"]
    assert torch.allclose(first["row_loss"], second["row_loss"])
    assert torch.allclose(first["loss"], second["loss"])


def test_top3_hit_does_not_imply_qk3_one_and_counterfactual_can_improve_qk3() -> None:
    covered_logits = torch.tensor(
        [[3.0, 2.0, 1.0, 0.5] + [-2.0] * 21], dtype=torch.float32
    )
    covered = k3_coverage_objective(covered_logits, [[2]])
    covered_qk3 = k3_certified_mass(covered_logits[0], {2})["certified_mass"]

    assert covered["miss_count"] == 0
    assert covered_qk3 < 1.0

    missed_logits = covered_logits.clone()
    missed_logits[0, 2] = 1.0
    missed_logits[0, 3] = 0.5
    before = k3_certified_mass(missed_logits[0], {3})["certified_mass"]
    after_logits = missed_logits.clone()
    after_logits[0, 3] = missed_logits[0, 2] + 0.1
    after = k3_certified_mass(after_logits[0], {3})["certified_mass"]

    assert after > before
    assert after < 1.0


def test_batch_mean_and_stable_tie_selection() -> None:
    logits = torch.zeros((3, 25), dtype=torch.float32, requires_grad=True)
    result = k3_coverage_objective(logits, [[2], [], [3, 4]], margin=0.1)

    assert result["miss_count"] == 1
    assert result["row_loss"].tolist() == pytest.approx([0.0, 0.0, 0.1])
    assert result["loss"].item() == pytest.approx(0.1 / 3.0)
    result["loss"].backward()
    assert logits.grad[2, 2].item() == pytest.approx(0.0)
    assert logits.grad[2, 3].item() == pytest.approx(-1.0 / 3.0)
    assert logits.grad[2, 4].item() == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("logits", "certified", "shortlist_k", "margin"),
    [
        (torch.zeros((1, 24)), [[]], 3, 0.1),
        (torch.zeros((1, 25)), [[25]], 3, 0.1),
        (torch.zeros((1, 25)), [[1, 1]], 3, 0.1),
        (torch.full((1, 25), float("nan")), [[]], 3, 0.1),
        (torch.zeros((1, 25)), [[]], 1, 0.1),
        (torch.zeros((1, 25)), [[]], 3, -0.1),
    ],
)
def test_invalid_inputs_are_rejected(logits, certified, shortlist_k, margin) -> None:
    with pytest.raises(K3CoverageObjectiveInputError):
        k3_coverage_objective(
            logits,
            certified,
            shortlist_k=shortlist_k,
            margin=margin,
        )
