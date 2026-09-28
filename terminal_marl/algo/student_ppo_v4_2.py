"""Student-only V4.2 PPO kernel. Formal entry points must pass admission first.

The historical BC container remains unchanged. Team reward and team critic
are recorded once per simulator step; committed/takeover agents contribute
to critic context but are masked out of the actor objective.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch

from .joint_ppo_v4_2 import JointPPOV42, hierarchical_joint_logprob
from ..core.observations import pad_agent_sets


@dataclass(frozen=True)
class StudentStepV42:
    world: np.ndarray
    requested_gate: np.ndarray
    requested_proposal: np.ndarray  # -1 for WAIT
    gate_logprob: np.ndarray
    proposal_logprob: np.ndarray  # NaN for WAIT
    actor_mask: np.ndarray
    value: float
    next_value: float
    reward: float  # single team reward, matching the centralized critic
    terminated: bool
    episode_end: bool  # includes time limits; those bootstrap next_value
    teacher_used: bool = False
    stochastic_behavior: bool = True

    @property
    def requested_logprob(self) -> np.ndarray:
        return self.gate_logprob + np.where(self.requested_gate == 1, self.proposal_logprob, 0.)

    def validate(self, observation_dim: int) -> None:
        if self.teacher_used:
            raise ValueError("Teacher must not replace student behavior in PPO")
        if self.world.ndim != 2 or self.world.shape[1] != observation_dim:
            raise ValueError("world must match observation schema")
        n = len(self.world)
        fields = (self.requested_gate, self.requested_proposal, self.gate_logprob, self.proposal_logprob, self.actor_mask)
        if any(np.shape(v) != (n,) for v in fields):
            raise ValueError("requested actions, likelihoods and actor mask must align with world")
        if self.actor_mask.dtype != np.bool_:
            raise ValueError("actor mask must be boolean")
        release = self.requested_gate == 1
        if not np.isin(self.requested_gate, (0, 1)).all():
            raise ValueError("invalid requested gate")
        if not np.issubdtype(self.requested_proposal.dtype, np.integer):
            raise ValueError("proposal indices must be integers")
        if np.any(self.requested_proposal[~release] != -1) or np.any((self.requested_proposal[release] < 0) | (self.requested_proposal[release] >= 25)):
            raise ValueError("invalid WAIT sentinel or RELEASE proposal")
        if not np.isnan(self.proposal_logprob[~release]).all():
            raise ValueError("WAIT has no conditional proposal likelihood")
        if not (np.isfinite(self.world).all() and np.isfinite(self.gate_logprob).all()
                and np.isfinite(self.proposal_logprob[release]).all()
                and np.isfinite([self.value, self.next_value, self.reward]).all()):
            raise ValueError("non-finite PPO rollout")
        if self.terminated and (not self.episode_end or self.next_value != 0.):
            raise ValueError("terminal transition must end episode without bootstrap")


def team_gae(steps: list[StudentStepV42], gamma: float, gae_lambda: float) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros(len(steps), dtype=np.float32)
    following = 0.
    for index in range(len(steps) - 1, -1, -1):
        row = steps[index]
        delta = row.reward + gamma * (0. if row.terminated else row.next_value) - row.value
        following = delta + gamma * gae_lambda * (0. if row.episode_end else following)
        advantages[index] = following
    return advantages, advantages + np.asarray([s.value for s in steps], dtype=np.float32)


def _full_rollout_sampled_approx_kl(policy: "StudentPPOV42", steps: list[StudentStepV42]) -> float:
    """Compute the requested-action KL proxy over all actor worlds in a rollout.

    The sampled requested gate and conditional proposal are the same actions and
    old log-probabilities used by PPO.  Agent likelihoods are averaged within a
    world and then across actor worlds.  The network remains float32; the
    expm1 subtraction is promoted to float64 to avoid cancellation near zero.
    """
    active = [row for row in steps if bool(row.actor_mask.any())]
    if not active:
        return 0.0
    observations = np.concatenate([row.world[row.actor_mask] for row in active])
    gates = np.concatenate([row.requested_gate[row.actor_mask] for row in active])
    proposals = np.concatenate([row.requested_proposal[row.actor_mask] for row in active])
    old = np.concatenate([row.requested_logprob[row.actor_mask] for row in active])
    counts = [int(row.actor_mask.sum()) for row in active]
    with torch.no_grad():
        tensor_obs = torch.as_tensor(observations, dtype=torch.float32, device=policy.device)
        gate_dist, proposal_dist = policy.model.distributions(tensor_obs)
        _, _, current = hierarchical_joint_logprob(
            gate_dist,
            proposal_dist,
            torch.as_tensor(gates, dtype=torch.long, device=policy.device),
            torch.as_tensor(proposals, dtype=torch.long, device=policy.device),
        )
        logratio = current - torch.as_tensor(old, dtype=torch.float32, device=policy.device)
        logratio64 = logratio.to(dtype=torch.float64)
        sampled_kl = torch.expm1(logratio64) - logratio64
        world_means = torch.stack([chunk.mean() for chunk in torch.split(sampled_kl, counts)])
        return float(world_means.mean().cpu())


def _state_finite(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all())
    if isinstance(value, dict):
        return all(_state_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_state_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True


class StudentPPOV42(JointPPOV42):
    """Opt-in optimizer; constructing a BC policy never unlocks formal PPO."""

    def update(self, steps: list[StudentStepV42], *, training_authorized: bool = False,
               epochs: int = 4, minibatch_size: int = 32, gamma: float = .99,
               gae_lambda: float = .95, clip_ratio: float = .2,
               value_coef: float = .5, entropy_coef: float = .01,
               target_kl: float | None = None,
               actor_reduction: str = "world_mean") -> dict:
        """Run PPO with either historical world means or joint-team scaling.

        ``world_mean`` averages the clipped per-agent objective inside each
        world and then across worlds, preserving the historical update.  The
        opt-in ``joint_team`` mode sums the per-agent objective in each
        minibatch and divides by one rollout-wide actor-count normalizer.  It
        removes state-dependent 1/N weighting of the team score gradient and
        applies a fixed rollout scaling; the per-agent PPO ratios remain
        individually clipped and are not replaced by a product of ratios.
        """
        gate_reference = getattr(self, "gate_reference", None)
        if not training_authorized:
            raise PermissionError("PPO requires student-only qualification and admission")
        if not steps or epochs < 1 or minibatch_size < 1:
            raise ValueError("nonempty rollout and positive update budget required")
        if not (0 <= gamma <= 1 and 0 <= gae_lambda <= 1 and 0 < clip_ratio < 1):
            raise ValueError("invalid PPO discount or clipping settings")
        if actor_reduction not in ("world_mean", "joint_team"):
            raise ValueError("actor_reduction must be 'world_mean' or 'joint_team'")
        if target_kl is not None:
            if not np.isfinite(target_kl) or target_kl <= 0:
                raise ValueError("target_kl must be finite and positive")
            target_kl = float(target_kl)
        if gate_reference is not None and target_kl is None:
            raise ValueError("target_kl is required when gate_reference is set")
        for row in steps:
            row.validate(self.observation_dim)
            if not row.stochastic_behavior:
                raise ValueError("deterministic evaluation is not on-policy PPO behavior")
        advantages, returns = team_gae(steps, gamma, gae_lambda)
        actor_steps = np.asarray([bool(row.actor_mask.any()) for row in steps])
        actor_decisions = sum(int(row.actor_mask.sum()) for row in steps)
        actor_count_normalizer = actor_decisions / len(steps)
        if actor_steps.any():
            values = advantages[actor_steps]
            advantages = (advantages - values.mean()) / (values.std() + 1e-8)
        totals = {key: 0. for key in ("loss", "actor_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "grad_norm")}
        updates = 0
        attempted_updates = 0
        rejected_updates = 0
        stopped = False
        final_rollout_kl = 0.0
        max_accepted_rollout_kl = 0.0
        reference_kl_before = 0.0
        reference_kl_final = 0.0
        reference_kl_max_accepted = 0.0
        reference_rejected_updates = 0
        rejection_reason = None
        if gate_reference is not None:
            initial_reference_metrics = gate_reference.metrics(self)
            reference_kl_before = float(initial_reference_metrics["reference_kl"])
            reference_kl_final = reference_kl_before
            reference_limit = float(gate_reference.max_kl)
            if not np.isfinite(reference_kl_before) or reference_kl_before > reference_limit:
                attempted_updates = 1
                rejected_updates = 1
                reference_rejected_updates = 1
                stopped = True
                rejection_reason = (
                    "reference_kl_nonfinite"
                    if not np.isfinite(reference_kl_before)
                    else "reference_kl_before_exceeds_max"
                )
        for _ in range(epochs):
            if stopped:
                break
            permutation = np.random.permutation(len(steps))
            for start in range(0, len(steps), minibatch_size):
                indices = permutation[start:start + minibatch_size]
                batch = [steps[i] for i in indices]
                padded, valid = pad_agent_sets([row.world for row in batch])
                if padded.shape[1] == 0:
                    padded = np.zeros((len(batch), 1, self.observation_dim), dtype=np.float32)
                    valid = np.zeros((len(batch), 1), dtype=bool)
                world = torch.as_tensor(padded, device=self.device)
                mask = torch.as_tensor(valid, device=self.device)
                predicted_value = self.model.critic(world, mask)
                target = torch.as_tensor(returns[indices], device=self.device)
                value_loss = (predicted_value - target).square().mean()
                # Legacy world_mean averages agents inside each world, then
                # worlds. joint_team sums the shared team score and applies
                # one fixed rollout-wide actor-count normalizer below.
                active = [i for i, row in enumerate(batch) if row.actor_mask.any()]
                actor = entropy = kl = clipped = value_loss.new_zeros(())
                if active:
                    observations = np.concatenate([batch[i].world[batch[i].actor_mask] for i in active])
                    gates = np.concatenate([batch[i].requested_gate[batch[i].actor_mask] for i in active])
                    proposals = np.concatenate([batch[i].requested_proposal[batch[i].actor_mask] for i in active])
                    old = np.concatenate([batch[i].requested_logprob[batch[i].actor_mask] for i in active])
                    counts = [int(batch[i].actor_mask.sum()) for i in active]
                    tensor_obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
                    gate_dist, proposal_dist = self.model.distributions(tensor_obs)
                    _, _, current = hierarchical_joint_logprob(gate_dist, proposal_dist,
                        torch.as_tensor(gates, dtype=torch.long, device=self.device),
                        torch.as_tensor(proposals, dtype=torch.long, device=self.device))
                    logratio = current - torch.as_tensor(old, dtype=torch.float32, device=self.device)
                    ratio = logratio.exp()
                    advantage = torch.as_tensor(np.repeat(advantages[indices[active]], counts), device=self.device)
                    objective = -torch.minimum(ratio * advantage, ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantage)
                    def team_mean(values):
                        return torch.stack([chunk.mean() for chunk in torch.split(values, counts)]).mean()
                    if actor_reduction == "joint_team":
                        actor = (objective.sum() / (len(batch) * actor_count_normalizer)
                                 if actor_count_normalizer else value_loss.new_zeros(()))
                    else:
                        actor = team_mean(objective)
                    entropy = team_mean(gate_dist.entropy() + gate_dist.probs[:, 1] * proposal_dist.entropy())
                    kl = team_mean((ratio - 1) - logratio)
                    clipped = team_mean((torch.abs(ratio - 1) > clip_ratio).float())
                reference_kl = None
                if gate_reference is not None:
                    reference_kl = gate_reference.weighted_kl(self.model)
                    if not torch.isfinite(reference_kl):
                        raise FloatingPointError("non-finite reference gate KL")
                loss = actor + value_coef * value_loss - entropy_coef * entropy
                if reference_kl is not None:
                    regularization_loss = getattr(gate_reference, "regularization_loss", None)
                    if callable(regularization_loss):
                        loss = loss + regularization_loss(reference_kl)
                    else:
                        loss = loss + float(gate_reference.coefficient) * reference_kl
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite PPO objective")
                model_snapshot = copy.deepcopy(self.model.state_dict()) if target_kl is not None else None
                optimizer_snapshot = copy.deepcopy(self.optimizer.state_dict()) if target_kl is not None else None
                if target_kl is not None:
                    attempted_updates += 1
                self.optimizer.zero_grad()
                try:
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
                    self.optimizer.step()
                except (FloatingPointError, RuntimeError):
                    if target_kl is None:
                        raise
                    self.model.load_state_dict(model_snapshot)
                    self.optimizer.load_state_dict(optimizer_snapshot)
                    self.optimizer.zero_grad(set_to_none=True)
                    rejected_updates += 1
                    stopped = True
                    if gate_reference is not None:
                        rejection_reason = "optimizer_step_error"
                    break
                if target_kl is not None:
                    model_finite = _state_finite(self.model.state_dict())
                    optimizer_finite = _state_finite(self.optimizer.state_dict())
                    if model_finite and optimizer_finite:
                        try:
                            rollout_kl = _full_rollout_sampled_approx_kl(self, steps)
                        except (FloatingPointError, RuntimeError, ValueError):
                            rollout_kl = float("nan")
                    else:
                        rollout_kl = float("nan")
                    finite_update = model_finite and optimizer_finite and np.isfinite(rollout_kl)
                    reference_kl_after = None
                    reference_ok = True
                    if gate_reference is not None:
                        try:
                            reference_metrics = gate_reference.metrics(self)
                            reference_kl_after = float(reference_metrics["reference_kl"])
                        except (FloatingPointError, RuntimeError, ValueError, KeyError, TypeError):
                            reference_kl_after = float("nan")
                        reference_ok = (
                            np.isfinite(reference_kl_after)
                            and reference_kl_after <= float(gate_reference.max_kl)
                        )
                    if not finite_update or rollout_kl > target_kl or not reference_ok:
                        self.model.load_state_dict(model_snapshot)
                        self.optimizer.load_state_dict(optimizer_snapshot)
                        self.optimizer.zero_grad(set_to_none=True)
                        rejected_updates += 1
                        stopped = True
                        if gate_reference is not None:
                            reference_rejected_updates += int(not reference_ok)
                            if not reference_ok:
                                rejection_reason = (
                                    "reference_kl_nonfinite"
                                    if not np.isfinite(reference_kl_after)
                                    else "reference_kl_exceeds_max"
                                )
                            elif not finite_update:
                                rejection_reason = "rollout_or_state_nonfinite"
                            else:
                                rejection_reason = "rollout_kl_exceeds_target"
                        break
                for key, value in (("loss", loss), ("actor_loss", actor), ("value_loss", value_loss),
                                   ("entropy", entropy), ("approx_kl", kl), ("clip_fraction", clipped), ("grad_norm", norm)):
                    totals[key] += float(value.detach())
                updates += 1
                if target_kl is not None:
                    final_rollout_kl = float(rollout_kl)
                    max_accepted_rollout_kl = max(max_accepted_rollout_kl, final_rollout_kl)
                    if gate_reference is not None:
                        reference_kl_final = float(reference_kl_after)
                        reference_kl_max_accepted = max(reference_kl_max_accepted, reference_kl_final)
            if stopped:
                break
        result = {**{key: value / updates if updates else 0.0 for key, value in totals.items()}, "updates": updates,
                  "team_steps": len(steps), "actor_decisions": actor_decisions}
        if actor_reduction == "joint_team":
            result.update({
                "joint_team_actor": True,
                "actor_count_normalizer": float(actor_count_normalizer),
            })
        if target_kl is not None:
            result.update({
                "accepted_updates": updates,
                "attempted_updates": attempted_updates,
                "rejected_updates": rejected_updates,
                "stopped": stopped,
                "target_kl": target_kl,
                "final_rollout_kl": final_rollout_kl,
                "max_accepted_rollout_kl": max_accepted_rollout_kl,
            })
        if gate_reference is not None:
            result.update({
                "reference_kl_before": reference_kl_before,
                "reference_kl_final": reference_kl_final,
                "reference_kl_max_accepted": reference_kl_max_accepted,
                "reference_rejected_updates": reference_rejected_updates,
                "rejection_reason": rejection_reason,
            })
        return result
