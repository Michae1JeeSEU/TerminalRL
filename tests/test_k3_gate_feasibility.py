from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from terminal_marl.training.k3_gate_feasibility import (
    K3GateFeasibilityInputError,
    k3_certified_mass,
)


def _expected_probability(logits: torch.Tensor, index: int) -> float:
    return float(torch.distributions.Categorical(logits=logits).probs[index])


def test_top_two_certified_cover_all_k3_requests() -> None:
    logits = torch.arange(25, dtype=torch.float32)

    result = k3_certified_mass(logits, {24, 23}, shortlist_k=3)

    assert result["certified_mass"] == 1.0
    assert 0.0 <= result["certified_mass"] <= 1.0
    assert result["catalogue_has_certified"] is True
    assert result["ranked_indices"][:3] == [24, 23, 22]


def test_only_low_rank_certified_proposal_contributes_its_request_probability() -> None:
    logits = torch.arange(25, dtype=torch.float32)
    certified = {0}

    result = k3_certified_mass(logits, certified, shortlist_k=3)

    assert result["certified_mass"] == pytest.approx(_expected_probability(logits, 0))
    assert result["shortlist_by_requested"][0] == [0, 24, 23]
    assert 0 not in result["shortlist_by_requested"][3]


def test_empty_certified_catalogue_has_zero_mass() -> None:
    result = k3_certified_mass(np.zeros(25, dtype=np.float32), set(), shortlist_k=3)

    assert result["certified_mass"] == 0.0
    assert result["catalogue_has_certified"] is False


def test_k1_is_exact_requested_index_mass() -> None:
    logits = torch.linspace(-2.0, 2.0, 25)
    certified = {2, 17}

    result = k3_certified_mass(logits, certified, shortlist_k=1)
    expected = sum(_expected_probability(logits, index) for index in certified)

    assert result["certified_mass"] == pytest.approx(expected)
    assert all(len(shortlist) == 1 for shortlist in result["shortlist_by_requested"])
    assert result["shortlist_by_requested"][17] == [17]


def test_stable_ties_and_extreme_logits_preserve_runtime_rank_and_probability() -> None:
    tied = torch.zeros(25, dtype=torch.float32)
    tied_result = k3_certified_mass(tied, {2}, shortlist_k=3)

    assert tied_result["ranked_indices"] == list(range(25))
    assert tied_result["shortlist_by_requested"][0] == [0, 1, 2]
    assert tied_result["shortlist_by_requested"][1] == [1, 0, 2]
    assert tied_result["certified_mass"] == pytest.approx(3.0 / 25.0)

    extreme = torch.full((25,), -1000.0, dtype=torch.float32)
    extreme[7] = 0.0
    extreme_result = k3_certified_mass(extreme, {7}, shortlist_k=1)
    assert math.isfinite(extreme_result["certified_mass"])
    assert extreme_result["certified_mass"] == pytest.approx(1.0)


def test_inputs_and_rng_are_unchanged() -> None:
    logits = torch.linspace(-1.0, 1.0, 25)
    logits_before = logits.clone()
    rng_before = torch.random.get_rng_state().clone()

    result = k3_certified_mass(logits, {3, 4}, shortlist_k=3)

    assert torch.equal(logits, logits_before)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert result["ranked_indices"]


@pytest.mark.parametrize(
    ("logits", "certified", "shortlist_k"),
    [
        (torch.zeros((1, 25)), set(), 3),
        (torch.zeros(24), set(), 3),
        (torch.tensor([0.0] * 24 + [float("nan")]), set(), 3),
        (torch.zeros(25), {25}, 3),
        (torch.zeros(25), {1.5}, 3),
        (torch.zeros(25), [1, 1], 3),
        (torch.zeros(25), set(), 2),
    ],
)
def test_invalid_inputs_are_rejected(logits, certified, shortlist_k) -> None:
    with pytest.raises(K3GateFeasibilityInputError):
        k3_certified_mass(logits, certified, shortlist_k=shortlist_k)
