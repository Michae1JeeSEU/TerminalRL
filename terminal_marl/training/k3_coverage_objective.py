"""Differentiable ranking surrogate for the runtime K=3 shortlist.

The runtime samples a proposal and then builds a deterministic K=3 shortlist
from the proposal logits.  This module supplies a narrow, offline ranking
surrogate for rows whose deterministic top-three list contains no certified
proposal.  It does not create gate labels, change the runtime shortlist, or
represent a utility-optimal proposal target.  The stochastic ``qK3`` coverage
of any resulting policy must still be measured separately with
``k3_gate_feasibility.k3_certified_mass``.

Training callers must pass the proposal head's raw, unnormalized logits.
Passing ``Categorical.logits`` or ``log_softmax`` output inserts a normalization
operation into the computation graph; its chain rule can propagate gradient to
other original logits, so the selected-certified-only gradient statement does
not apply to that transformed input.  A shared proposal head can also change
other states when its parameters are updated.  The restriction here is about
the logit-layer gradient, not a guarantee about model parameters or other
states.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from numbers import Integral, Real
from typing import Any

import torch
from torch.nn import functional as F


PROPOSAL_COUNT = 25
SHORTLIST_K = 3
DEFAULT_MARGIN = 0.1


class K3CoverageObjectiveInputError(ValueError):
    """Raised when a K=3 coverage objective input violates the contract."""


def _validate_logits(logits: Any) -> torch.Tensor:
    try:
        values = torch.as_tensor(logits)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise K3CoverageObjectiveInputError(
            "logits must be a tensor-like floating-point value"
        ) from exc
    if values.ndim != 2 or int(values.shape[1]) != PROPOSAL_COUNT:
        raise K3CoverageObjectiveInputError(
            f"logits must have shape [B, {PROPOSAL_COUNT}]"
        )
    if int(values.shape[0]) == 0:
        raise K3CoverageObjectiveInputError("logits batch must not be empty")
    if not torch.is_floating_point(values):
        raise K3CoverageObjectiveInputError("logits must use a floating-point dtype")
    if not bool(torch.isfinite(values).all().item()):
        raise K3CoverageObjectiveInputError("logits must be finite")
    return values


def _row_indices(value: Any) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise K3CoverageObjectiveInputError(
            "certified proposal indices must be an iterable of integers"
        )
    if isinstance(value, torch.Tensor):
        if value.ndim != 1:
            raise K3CoverageObjectiveInputError(
                "each certified proposal row must be one-dimensional"
            )
        raw_indices = tuple(value.detach().cpu().tolist())
    else:
        try:
            raw_indices = tuple(value)
        except TypeError as exc:
            raise K3CoverageObjectiveInputError(
                "each certified proposal row must be an iterable"
            ) from exc

    result: list[int] = []
    for raw_index in raw_indices:
        if isinstance(raw_index, bool) or not isinstance(raw_index, Integral):
            raise K3CoverageObjectiveInputError(
                "certified proposal indices must be integer values"
            )
        index = int(raw_index)
        if not 0 <= index < PROPOSAL_COUNT:
            raise K3CoverageObjectiveInputError(
                f"certified proposal index out of range: {index}"
            )
        if index in result:
            raise K3CoverageObjectiveInputError(
                f"certified proposal index is duplicated: {index}"
            )
        result.append(index)
    return tuple(sorted(result))


def _validate_certified_batch(
    certified_indices: Iterable[Iterable[int]],
    batch_size: int,
) -> tuple[tuple[int, ...], ...]:
    if isinstance(certified_indices, (str, bytes)):
        raise K3CoverageObjectiveInputError(
            "certified_indices must contain one iterable per batch row"
        )
    try:
        rows = tuple(certified_indices)
    except TypeError as exc:
        raise K3CoverageObjectiveInputError(
            "certified_indices must contain one iterable per batch row"
        ) from exc
    if len(rows) != batch_size:
        raise K3CoverageObjectiveInputError(
            f"certified_indices must contain {batch_size} rows"
        )
    return tuple(_row_indices(row) for row in rows)


def _validate_shortlist_k(shortlist_k: int) -> None:
    if isinstance(shortlist_k, bool) or not isinstance(shortlist_k, Integral):
        raise K3CoverageObjectiveInputError("shortlist_k must be 3")
    if int(shortlist_k) != SHORTLIST_K:
        raise K3CoverageObjectiveInputError("shortlist_k must be 3")


def _validate_margin(margin: Real | torch.Tensor) -> float:
    if isinstance(margin, torch.Tensor):
        if margin.ndim != 0:
            raise K3CoverageObjectiveInputError("margin must be a scalar")
        try:
            value = float(margin.detach().cpu().item())
        except (RuntimeError, TypeError) as exc:
            raise K3CoverageObjectiveInputError("margin must be numeric") from exc
    else:
        if isinstance(margin, bool) or not isinstance(margin, Real):
            raise K3CoverageObjectiveInputError("margin must be numeric")
        value = float(margin)
    if not math.isfinite(value) or value < 0.0:
        raise K3CoverageObjectiveInputError("margin must be finite and non-negative")
    return value


def k3_coverage_objective(
    logits: torch.Tensor,
    certified_indices: Sequence[Iterable[int]],
    *,
    shortlist_k: int = SHORTLIST_K,
    margin: float = DEFAULT_MARGIN,
) -> dict[str, Any]:
    """Return a differentiable top-three ranking repair surrogate.

    For each row with a non-empty certified set whose stable deterministic
    top-three logits contain no certified index, the objective selects the
    highest-logit certified proposal (stable by proposal index) and compares
    it with the third-highest non-certified boundary.  The row contribution is

    ``relu(detached_boundary_logit + margin - best_certified_logit)``.

    The boundary is detached deliberately: this objective supplies gradient
    only to the selected best certified logit and cannot lower an existing
    top-three logit.  Empty and already-covered rows contribute an attached
    zero, and ``loss`` is the mean over the complete batch.  The returned
    ``miss_count`` counts rows receiving the surrogate.  This is only a rank
    surrogate for deterministic K=3 gaps; it is not a gate label or an
    optimal utility target, and stochastic ``qK3`` remains a separate metric.
    The ``logits`` argument must be raw, unnormalized proposal-head logits;
    feeding normalized ``Categorical.logits``/``log_softmax`` output can send
    chain-rule gradient to other original logits.  Even with raw logits, a
    shared head update may affect other states; this objective gives no
    parameter-level isolation guarantee.
    """

    values = _validate_logits(logits)
    _validate_shortlist_k(shortlist_k)
    margin_value = _validate_margin(margin)
    certified_rows = _validate_certified_batch(certified_indices, int(values.shape[0]))

    # Ranking is a runtime-matched discrete operation.  Detaching it keeps the
    # rank decision out of autograd while preserving the selected logit's
    # gradient below.
    detached = values.detach()
    ranked = torch.argsort(detached, dim=1, descending=True, stable=True).tolist()
    row_losses: list[torch.Tensor] = []
    miss_count = 0

    for row_index, certified in enumerate(certified_rows):
        certified_set = set(certified)
        top3 = ranked[row_index][:SHORTLIST_K]
        zero = values[row_index].sum() * 0.0
        if not certified_set or certified_set.intersection(top3):
            row_losses.append(zero)
            continue

        # The input is sorted, so stable descending argsort resolves equal
        # certified logits by the lower proposal index.
        certified_order = sorted(
            certified,
            key=lambda index: (-float(detached[row_index, index].item()), index),
        )
        best_certified_index = certified_order[0]
        noncertified_ranked = [
            index for index in ranked[row_index] if index not in certified_set
        ]
        boundary_index = noncertified_ranked[SHORTLIST_K - 1]
        boundary = detached[row_index, boundary_index]
        best_certified = values[row_index, best_certified_index]
        row_losses.append(F.relu(boundary + margin_value - best_certified))
        miss_count += 1

    row_loss = torch.stack(row_losses)
    return {
        "loss": row_loss.mean(),
        "row_loss": row_loss,
        "miss_count": miss_count,
    }


__all__ = [
    "DEFAULT_MARGIN",
    "K3CoverageObjectiveInputError",
    "PROPOSAL_COUNT",
    "SHORTLIST_K",
    "k3_coverage_objective",
]
