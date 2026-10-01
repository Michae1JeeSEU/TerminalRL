"""Pure NumPy last-layer gate-logit projection.

This module is deliberately a small mathematical helper.  It does not read
artifacts, construct a model, run an environment, or update parameters.  The
caller supplies the hidden features and the calibrated source gate logits.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


HIDDEN_DIM = 97
TARGET_COUNT = 4
DEFAULT_LAMBDA = 1.0e-6
DEFAULT_KL_CAP = 9.5e-4
DEFAULT_RESIDUAL_TOL = 1.0e-8
DEFAULT_BISECTION_STEPS = 60


def _finite_array(value: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    """Convert an input to float64 and reject shape-independent bad values."""

    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric array") from exc
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got {array.ndim}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def stable_sigmoid(logits: Any) -> np.ndarray:
    """Evaluate sigmoid without overflow for finite float inputs."""

    values = _finite_array(logits, name="logits")
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0.0
    with np.errstate(over="ignore", under="ignore", invalid="raise"):
        result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
        exp_values = np.exp(values[~positive])
        result[~positive] = exp_values / (1.0 + exp_values)
    return result


def bernoulli_kl_from_logits(candidate_logits: Any, reference_logits: Any) -> np.ndarray:
    """Return ``KL(Bern(sigmoid(candidate)) || Bern(sigmoid(reference)))``.

    Log probabilities use ``logaddexp`` instead of clipping probabilities.  A
    finite input therefore remains a finite calculation whenever the actual
    KL is representable in float64.
    """

    candidate = _finite_array(candidate_logits, name="candidate_logits")
    reference = _finite_array(reference_logits, name="reference_logits")
    if candidate.shape != reference.shape:
        raise ValueError(
            "candidate_logits and reference_logits must have the same shape"
        )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        candidate_probability = stable_sigmoid(candidate)
        log_candidate_probability = -np.logaddexp(0.0, -candidate)
        log_candidate_complement = -np.logaddexp(0.0, candidate)
        log_reference_probability = -np.logaddexp(0.0, -reference)
        log_reference_complement = -np.logaddexp(0.0, reference)
        divergence = (
            candidate_probability
            * (log_candidate_probability - log_reference_probability)
            + (1.0 - candidate_probability)
            * (log_candidate_complement - log_reference_complement)
        )
    if not np.all(np.isfinite(divergence)):
        raise ValueError("Bernoulli KL became non-finite")
    # Round-off can make the identity calculation a signed zero or a tiny
    # negative number.  It is still the same finite KL value for this helper.
    return np.maximum(divergence, 0.0)


def _vector(value: Any, *, name: str, length: int) -> np.ndarray:
    array = _finite_array(value, name=name, ndim=1)
    if array.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {array.shape}")
    return array


def project_gate_last_layer(
    H_anchor: Any,
    H_target: Any,
    source_anchor_logits: Any,
    source_target_logits: Any,
    pstar: Any,
    *,
    lambda_reg: float = DEFAULT_LAMBDA,
    kl_cap: float = DEFAULT_KL_CAP,
    residual_tol: float = DEFAULT_RESIDUAL_TOL,
    bisection_steps: int = DEFAULT_BISECTION_STEPS,
) -> Dict[str, Any]:
    """Project four target gate logits while controlling all-row Bernoulli KL.

    The minimum-norm direction is computed from

    ``Q = H_anchor.T @ H_anchor / N + lambda_reg * I``

    followed by the four exact linear target constraints.  The returned
    ``scale_alpha`` is the largest point on that direction in ``[0, 1]``
    whose maximum anchor/target KL does not exceed ``kl_cap``.  When the full
    direction is feasible, no bisection is performed.
    """

    anchor_features = _finite_array(H_anchor, name="H_anchor", ndim=2)
    target_features = _finite_array(H_target, name="H_target", ndim=2)
    if anchor_features.shape[1:] != (HIDDEN_DIM,):
        raise ValueError(
            f"H_anchor must have shape (N, {HIDDEN_DIM}), got {anchor_features.shape}"
        )
    if anchor_features.shape[0] < 1:
        raise ValueError("H_anchor must contain at least one row")
    if target_features.shape != (TARGET_COUNT, HIDDEN_DIM):
        raise ValueError(
            f"H_target must have shape ({TARGET_COUNT}, {HIDDEN_DIM}), "
            f"got {target_features.shape}"
        )

    n_anchor = anchor_features.shape[0]
    anchor_logits = _vector(
        source_anchor_logits, name="source_anchor_logits", length=n_anchor
    )
    target_logits = _vector(
        source_target_logits, name="source_target_logits", length=TARGET_COUNT
    )
    target_probability = _vector(pstar, name="pstar", length=TARGET_COUNT)
    if np.any((target_probability <= 0.0) | (target_probability >= 1.0)):
        raise ValueError("pstar must be strictly inside (0, 1)")

    try:
        lambda_value = float(lambda_reg)
        cap_value = float(kl_cap)
        tolerance = float(residual_tol)
    except (TypeError, ValueError) as exc:
        raise ValueError("lambda_reg, kl_cap, and residual_tol must be scalars") from exc
    if not np.isfinite(lambda_value) or lambda_value <= 0.0:
        raise ValueError("lambda_reg must be finite and positive")
    if not np.isfinite(cap_value) or cap_value < 0.0:
        raise ValueError("kl_cap must be finite and non-negative")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("residual_tol must be finite and positive")
    if isinstance(bisection_steps, bool) or int(bisection_steps) != bisection_steps:
        raise ValueError("bisection_steps must be an integer")
    steps_limit = int(bisection_steps)
    if steps_limit <= 0:
        raise ValueError("bisection_steps must be positive")

    rank = int(np.linalg.matrix_rank(target_features))
    if rank != TARGET_COUNT:
        raise ValueError(f"H_target must have rank {TARGET_COUNT}, got {rank}")

    identity = np.eye(HIDDEN_DIM, dtype=np.float64)
    Q = (anchor_features.T @ anchor_features) / float(n_anchor)
    Q = Q + lambda_value * identity
    try:
        X = np.linalg.solve(Q, target_features.T)
        target_gram = target_features @ X
        requested_delta = np.log(target_probability) - np.log1p(-target_probability)
        requested_delta = requested_delta - target_logits
        direction = X @ np.linalg.solve(target_gram, requested_delta)
    except np.linalg.LinAlgError as exc:
        raise ValueError("projection linear system is singular") from exc
    if not np.all(np.isfinite(direction)):
        raise ValueError("projection direction is non-finite")

    target_residual_vector = target_features @ direction - requested_delta
    target_residual_max = float(np.max(np.abs(target_residual_vector)))
    if not np.isfinite(target_residual_max) or target_residual_max > tolerance:
        raise ValueError(
            f"target constraint residual {target_residual_max} exceeds {tolerance}"
        )

    all_features = np.concatenate((anchor_features, target_features), axis=0)
    all_source_logits = np.concatenate((anchor_logits, target_logits), axis=0)
    projected_direction_logits = all_features @ direction
    if not np.all(np.isfinite(projected_direction_logits)):
        raise ValueError("projected logits are non-finite")

    def _kl_at(alpha: float) -> tuple[float, np.ndarray]:
        candidate_logits = all_source_logits + float(alpha) * projected_direction_logits
        if not np.all(np.isfinite(candidate_logits)):
            raise ValueError("candidate logits are non-finite")
        values = bernoulli_kl_from_logits(candidate_logits, all_source_logits)
        return float(np.max(values)), values

    max_kl_at_one, _ = _kl_at(1.0)
    if max_kl_at_one <= cap_value:
        alpha = 1.0
        used_bisection_steps = 0
    else:
        low = 0.0
        high = 1.0
        for _ in range(steps_limit):
            midpoint = (low + high) / 2.0
            midpoint_kl, _ = _kl_at(midpoint)
            if midpoint_kl <= cap_value:
                low = midpoint
            else:
                high = midpoint
        alpha = low
        used_bisection_steps = steps_limit

    max_kl, all_kl = _kl_at(alpha)
    anchor_kl = all_kl[:n_anchor]
    target_kl = all_kl[n_anchor:]
    applied_delta = alpha * (target_features @ direction)
    candidate_anchor_logits = anchor_logits + alpha * (anchor_features @ direction)
    candidate_target_logits = target_logits + applied_delta
    result: Dict[str, Any] = {
        "direction": direction,
        "scale_alpha": float(alpha),
        "requested_target_logit_delta": requested_delta,
        "applied_target_logit_delta": applied_delta,
        "target_constraint_residual": target_residual_vector,
        "target_constraint_residual_max": target_residual_max,
        "rank": rank,
        "lambda_reg": lambda_value,
        "kl_cap": cap_value,
        "bisection_steps": used_bisection_steps,
        "max_kl": max_kl,
        "anchor_kl_max": float(np.max(anchor_kl)),
        "target_kl_max": float(np.max(target_kl)),
        "max_kl_at_alpha_one": max_kl_at_one,
        "source_anchor_probability": stable_sigmoid(anchor_logits),
        "source_target_probability": stable_sigmoid(target_logits),
        "candidate_anchor_probability": stable_sigmoid(candidate_anchor_logits),
        "candidate_target_probability": stable_sigmoid(candidate_target_logits),
    }
    if result["max_kl"] > cap_value + max(1.0e-12, tolerance):
        raise ValueError("bisection result exceeds KL cap")
    return result


# Short aliases keep the mathematical entry point discoverable without adding
# another implementation or any stateful wrapper.
solve_projection = project_gate_last_layer
last_layer_projection = project_gate_last_layer


__all__ = [
    "DEFAULT_BISECTION_STEPS",
    "DEFAULT_KL_CAP",
    "DEFAULT_LAMBDA",
    "DEFAULT_RESIDUAL_TOL",
    "HIDDEN_DIM",
    "TARGET_COUNT",
    "bernoulli_kl_from_logits",
    "last_layer_projection",
    "project_gate_last_layer",
    "solve_projection",
    "stable_sigmoid",
]
