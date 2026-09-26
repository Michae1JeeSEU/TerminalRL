"""Frozen reference distributions for the V4.2 gate PPO update."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


def _finite_scalar(value: Any, *, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not np.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{name} must be finite and >= {minimum}" if minimum is not None
                         else f"{name} must be finite")
    return result


@dataclass(frozen=True)
class GateReference:
    """A detached, weighted categorical gate reference distribution.

    ``reference_logits`` stores the distribution's normalized log-probabilities
    (``Categorical.logits``), rather than taking ``log(reference_probs)``.  This
    keeps the reference finite when a float32 softmax underflows a probability
    to zero.
    """

    observations: torch.Tensor
    reference_logits: torch.Tensor
    reference_probs: torch.Tensor
    weights: torch.Tensor
    coefficient: float = 0.1
    max_kl: float = 0.05
    soft_kl_target: float | None = None
    soft_kl_gain: float = 0.0

    def __post_init__(self) -> None:
        observations = torch.as_tensor(self.observations, dtype=torch.float32).detach().clone()
        reference_logits = torch.as_tensor(
            self.reference_logits, dtype=torch.float32,
        ).detach().clone()
        reference_probs = torch.as_tensor(
            self.reference_probs, dtype=torch.float32,
        ).detach().clone()
        weights = torch.as_tensor(self.weights, dtype=torch.float32).detach().clone()
        if observations.ndim != 2 or observations.shape[1] != 88 or observations.shape[0] < 1:
            raise ValueError("reference observations must have shape (N, 88) with N > 0")
        if reference_logits.ndim != 2 or tuple(reference_logits.shape) != (len(observations), 2):
            raise ValueError("reference logits must have shape (N, 2)")
        if reference_probs.ndim != 2 or tuple(reference_probs.shape) != (len(observations), 2):
            raise ValueError("reference probabilities must have shape (N, 2)")
        if weights.ndim != 1 or len(weights) != len(observations):
            raise ValueError("reference weights must have shape (N,)")
        if not bool(torch.isfinite(observations).all()):
            raise ValueError("reference observations must be finite")
        if not bool(torch.isfinite(reference_logits).all()):
            raise ValueError("reference logits must be finite")
        if not bool(torch.isfinite(reference_probs).all()) or bool((reference_probs < 0).any()):
            raise ValueError("reference probabilities must be finite and nonnegative")
        probability_sum = reference_probs.sum(dim=-1, keepdim=True)
        if bool((probability_sum <= 0).any()):
            raise ValueError("reference probabilities must have positive row sums")
        reference_probs = reference_probs / probability_sum
        if not bool(torch.isclose(
            torch.logsumexp(reference_logits, dim=-1),
            torch.zeros_like(torch.logsumexp(reference_logits, dim=-1)),
            atol=2e-5, rtol=2e-5,
        ).all()):
            raise ValueError("reference logits must be normalized log-probabilities")
        if not bool(torch.allclose(
            torch.softmax(reference_logits, dim=-1), reference_probs,
            atol=2e-5, rtol=2e-5,
        )):
            raise ValueError("reference logits and probabilities disagree")
        if not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
            raise ValueError("reference weights must be finite and positive")
        weight_sum = weights.sum()
        if not bool(torch.isfinite(weight_sum)) or float(weight_sum) <= 0.0:
            raise ValueError("reference weights must have a positive finite sum")
        weights = weights / weight_sum
        coefficient = _finite_scalar(self.coefficient, name="reference coefficient", minimum=0.0)
        max_kl = _finite_scalar(self.max_kl, name="reference max_kl", minimum=0.0)
        soft_kl_gain = _finite_scalar(self.soft_kl_gain, name="soft KL gain", minimum=0.0)
        soft_kl_target = None
        if self.soft_kl_target is not None:
            soft_kl_target = _finite_scalar(self.soft_kl_target, name="soft KL target", minimum=0.0)
        if soft_kl_gain > 0.0 and (
            soft_kl_target is None or soft_kl_target >= max_kl
        ):
            raise ValueError("soft KL target must be finite and satisfy 0 <= target < max_kl when gain > 0")
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "reference_logits", reference_logits)
        object.__setattr__(self, "reference_probs", reference_probs)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "max_kl", max_kl)
        object.__setattr__(self, "soft_kl_target", soft_kl_target)
        object.__setattr__(self, "soft_kl_gain", soft_kl_gain)

    @classmethod
    def from_policy(
        cls,
        policy: Any,
        observations: Any,
        weights: Any,
        *,
        coefficient: float = 0.1,
        max_kl: float = 0.05,
        soft_kl_target: float | None = None,
        soft_kl_gain: float = 0.0,
    ) -> "GateReference":
        """Snapshot the current policy's gate distribution without gradients."""

        model = getattr(policy, "model", policy)
        if not hasattr(model, "distributions"):
            raise TypeError("policy must expose model.distributions")
        device = getattr(policy, "device", None)
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration as exc:
                raise ValueError("policy model has no parameters") from exc
        tensor_observations = torch.as_tensor(
            observations, dtype=torch.float32, device=device,
        )
        if tensor_observations.ndim != 2 or tensor_observations.shape[1] != 88:
            raise ValueError("reference observations must have shape (N, 88)")
        with torch.no_grad():
            gate_distribution, _ = model.distributions(tensor_observations)
            reference_logits = gate_distribution.logits.detach().clone()
            reference_probs = gate_distribution.probs.detach().clone()
        return cls(
            tensor_observations.detach().clone(), reference_logits, reference_probs,
            weights, coefficient=coefficient, max_kl=max_kl,
            soft_kl_target=soft_kl_target, soft_kl_gain=soft_kl_gain,
        )

    @property
    def q_logits(self) -> torch.Tensor:
        return self.reference_logits

    @property
    def q_probs(self) -> torch.Tensor:
        return self.reference_probs

    @property
    def reference_log_probs(self) -> torch.Tensor:
        return self.reference_logits

    @property
    def reference_logprobs(self) -> torch.Tensor:
        return self.reference_logits

    @property
    def q_log_probs(self) -> torch.Tensor:
        return self.reference_logits

    @property
    def q_logprobs(self) -> torch.Tensor:
        return self.reference_logits

    def _gate_distribution(self, model: Any):
        candidate = getattr(model, "model", model)
        if not hasattr(candidate, "distributions"):
            raise TypeError("model must expose distributions")
        try:
            device = next(candidate.parameters()).device
        except StopIteration as exc:
            raise ValueError("model has no parameters") from exc
        observations = self.observations.to(device=device)
        gate_distribution, _ = candidate.distributions(observations)
        if tuple(gate_distribution.logits.shape) != tuple(self.reference_logits.shape):
            raise ValueError("model gate distribution shape does not match reference")
        return gate_distribution

    def _per_sample_kl(self, model: Any) -> tuple[torch.Tensor, torch.Tensor]:
        gate_distribution = self._gate_distribution(model)
        current_log_probs = gate_distribution.logits
        reference_probs = self.reference_probs.to(
            device=current_log_probs.device, dtype=current_log_probs.dtype,
        )
        reference_log_probs = self.reference_logits.to(
            device=current_log_probs.device, dtype=current_log_probs.dtype,
        )
        per_sample = torch.sum(
            reference_probs * (reference_log_probs - current_log_probs), dim=-1,
        )
        return per_sample, gate_distribution.probs

    def weighted_kl(self, model: Any) -> torch.Tensor:
        """Return differentiable weighted ``KL(reference || current)``."""

        per_sample, _ = self._per_sample_kl(model)
        weights = self.weights.to(device=per_sample.device, dtype=per_sample.dtype)
        return torch.sum(weights * per_sample)

    def regularization_loss(self, weighted_kl_tensor: torch.Tensor) -> torch.Tensor:
        """Return the reference penalty while preserving the KL gradient."""

        weighted_kl = torch.as_tensor(weighted_kl_tensor)
        loss = float(self.coefficient) * weighted_kl
        if self.soft_kl_gain > 0.0:
            target = torch.as_tensor(
                float(self.soft_kl_target), dtype=weighted_kl.dtype, device=weighted_kl.device,
            )
            loss = loss + float(self.soft_kl_gain) * torch.relu(weighted_kl - target).square()
        return loss

    def metrics(self, model: Any) -> dict[str, Any]:
        """Return detached reference-KL and current RELEASE-probability metrics."""

        with torch.no_grad():
            per_sample, current_probs = self._per_sample_kl(model)
            weights = self.weights.to(device=per_sample.device, dtype=per_sample.dtype)
            weighted = torch.sum(weights * per_sample)
            release = current_probs[:, 1]
            quantile_values = (0.05, 0.50, 0.95)
            quantiles = {
                str(value): float(torch.quantile(release, value).detach().cpu())
                for value in quantile_values
            }
            max_sample = torch.max(per_sample)
            effective_coefficient = float(self.coefficient)
            if self.soft_kl_gain > 0.0:
                effective_coefficient += 2.0 * float(self.soft_kl_gain) * max(
                    float(weighted.detach().cpu()) - float(self.soft_kl_target), 0.0,
                )
            result = {
                "reference_kl": float(weighted.detach().cpu()),
                "kl": float(weighted.detach().cpu()),
                "mean_p_release": float(release.mean().detach().cpu()),
                "p_release_mean": float(release.mean().detach().cpu()),
                "p_release_quantiles": quantiles,
                "quantiles": quantiles,
                "max_kl": float(max_sample.detach().cpu()),
                "max_reference_kl": float(max_sample.detach().cpu()),
                "effective_kl_coefficient": effective_coefficient,
                "soft_kl_target": self.soft_kl_target,
                "soft_kl_gain": float(self.soft_kl_gain),
            }
        return result
