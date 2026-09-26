"""Certified probability mass of the current proposal shortlist contract.

The caller supplies the frozen proposal logits and a set of proposal indices
already certified executable for the same pre-action state.  This module only
reproduces the runtime K=1/K=3 ranking rule; it does not call the environment,
SafetyFilter, model sampling, or any training code.
"""
from __future__ import annotations

from collections.abc import Iterable
from numbers import Integral
from typing import Any

import torch


PROPOSAL_COUNT = 25
SUPPORTED_SHORTLIST_K = (1, 3)


class K3GateFeasibilityInputError(ValueError):
    """Raised when a proposal coverage input violates the fixed contract."""


def _validated_logits(proposal_logits: Any) -> torch.Tensor:
    try:
        values = torch.as_tensor(proposal_logits)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise K3GateFeasibilityInputError("proposal_logits must be a tensor-like value") from exc
    if values.ndim != 1 or int(values.numel()) != PROPOSAL_COUNT:
        raise K3GateFeasibilityInputError(
            f"proposal_logits must be a one-dimensional vector of length {PROPOSAL_COUNT}"
        )
    try:
        values = values.detach().to(device="cpu", dtype=torch.float32).clone()
    except (RuntimeError, TypeError) as exc:
        raise K3GateFeasibilityInputError("proposal_logits must be numeric") from exc
    if not bool(torch.isfinite(values).all().item()):
        raise K3GateFeasibilityInputError("proposal_logits must be finite")
    return values


def _validated_certified_indices(certified_proposal_indices: Iterable[int]) -> frozenset[int]:
    if isinstance(certified_proposal_indices, (str, bytes)):
        raise K3GateFeasibilityInputError("certified proposal indices must be integers")
    try:
        raw_indices = tuple(certified_proposal_indices)
    except TypeError as exc:
        raise K3GateFeasibilityInputError(
            "certified proposal indices must be an iterable"
        ) from exc

    result: set[int] = set()
    for raw_index in raw_indices:
        if isinstance(raw_index, bool) or not isinstance(raw_index, Integral):
            raise K3GateFeasibilityInputError(
                "certified proposal indices must be integer values"
            )
        index = int(raw_index)
        if not 0 <= index < PROPOSAL_COUNT:
            raise K3GateFeasibilityInputError(
                f"certified proposal index out of range: {index}"
            )
        if index in result:
            raise K3GateFeasibilityInputError(
                f"certified proposal index is duplicated: {index}"
            )
        result.add(index)
    return frozenset(result)


def _validated_shortlist_k(shortlist_k: int) -> int:
    if isinstance(shortlist_k, bool) or not isinstance(shortlist_k, Integral):
        raise K3GateFeasibilityInputError("shortlist_k must be 1 or 3")
    value = int(shortlist_k)
    if value not in SUPPORTED_SHORTLIST_K:
        raise K3GateFeasibilityInputError("shortlist_k must be 1 or 3")
    return value


def k3_certified_mass(
    proposal_logits: Any,
    certified_proposal_indices: Iterable[int],
    *,
    shortlist_k: int = 3,
) -> dict[str, Any]:
    """Return the proposal probability mass covered by certified K-shortlists.

    ``proposal_logits`` is the 25-vector from the runtime proposal
    distribution (for example, ``policy.model.distributions(...)[1].logits``),
    rather than an arbitrary pre-softmax model-head tensor.  For every
    possible sampled proposal ``j``, the shortlist is constructed as
    ``[j]`` followed by the remaining proposal indices in stable descending
    logit order, truncated to ``shortlist_k`` entries.  The returned mass is
    ``sum_j p(j)`` over shortlists intersecting the caller-provided certified
    set.  It is a state-local shortlist coverage measure, not a closed-loop
    success probability.
    """

    values = _validated_logits(proposal_logits)
    k = _validated_shortlist_k(shortlist_k)
    certified = _validated_certified_indices(certified_proposal_indices)

    # This is the same float32 Categorical construction used by the runtime;
    # constructing the distribution does not sample or change RNG state.
    distribution = torch.distributions.Categorical(logits=values)
    probabilities = distribution.probs.detach().cpu().tolist()
    ranked_indices = torch.argsort(values, descending=True, stable=True).tolist()

    shortlist_by_requested: list[list[int]] = []
    covered: list[bool] = []
    for requested in range(PROPOSAL_COUNT):
        remaining = [index for index in ranked_indices if int(index) != requested]
        shortlist = [requested, *remaining[: k - 1]]
        shortlist_by_requested.append([int(index) for index in shortlist])
        covered.append(bool(certified.intersection(shortlist)))

    certified_mass = float(
        sum(float(probability) for probability, is_covered in zip(probabilities, covered) if is_covered)
    )
    # JSON/checkpoint probabilities can sum to 1 + a few float32 ulps.  This
    # is only a reporting boundary; it does not alter actor probabilities or
    # introduce a release floor.
    certified_mass = min(1.0, max(0.0, certified_mass))
    return {
        "certified_mass": certified_mass,
        "catalogue_has_certified": bool(certified),
        "ranked_indices": [int(index) for index in ranked_indices],
        "shortlist_by_requested": shortlist_by_requested,
    }


__all__ = [
    "K3GateFeasibilityInputError",
    "PROPOSAL_COUNT",
    "SUPPORTED_SHORTLIST_K",
    "k3_certified_mass",
]
