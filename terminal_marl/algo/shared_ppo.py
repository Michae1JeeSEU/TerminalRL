"""Shared 3-D low-level PPO with a masked mean/max set critic.

The actor consumes one UAV observation at a time; the critic receives the
variable-size active-UAV set.  No attention, GNN, or fixed max-N action exists.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Any
import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli, Normal


class TanhSquashedNormal:
    """A Normal policy transformed into the open interval ``(-1, 1)``.

    The action stored in the rollout is the *bounded* action.  Its density is
    consequently the change-of-variables density, rather than the density of
    a pre-squash Gaussian evaluated at a post-hoc-clamped value.
    """

    _endpoint_epsilon = 1e-6

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor) -> None:
        self.loc, self.scale = loc, scale
        self.base_dist = Normal(loc, scale)

    @property
    def mean(self) -> torch.Tensor:
        return self._squash(self.loc)

    @property
    def mode(self) -> torch.Tensor:
        return self.mean

    @classmethod
    def _squash(cls, latent: torch.Tensor) -> torch.Tensor:
        action = torch.tanh(latent)
        # Finite precision can round tanh(large finite z) to exactly +/-1.
        # Move only those numerical round-offs one representable value inward;
        # this is not a behaviour clamp and preserves the tanh policy support.
        inward = torch.zeros_like(action)
        return torch.where(action == 1, torch.nextafter(action, inward),
                           torch.where(action == -1, torch.nextafter(action, inward), action))

    def sample(self) -> torch.Tensor:
        return self._squash(self.base_dist.sample())

    def rsample(self) -> torch.Tensor:
        return self._squash(self.base_dist.rsample())

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        # atanh is needed only to recover z from a stored bounded action.
        # Exact teacher endpoints never enter PPO, but clamping internally
        # makes diagnostics/auxiliary endpoint probes finite without changing
        # their external +/-1 semantics.
        stable_action = action.clamp(-1 + self._endpoint_epsilon, 1 - self._endpoint_epsilon)
        latent = torch.atanh(stable_action)
        log_jacobian = torch.log1p(-stable_action.square())
        return self.base_dist.log_prob(latent) - log_jacobian

    def entropy(self) -> torch.Tensor:
        """Deterministic Gauss-Hermite estimate of the transformed entropy."""
        nodes = torch.as_tensor(
            (-5.38748089001123, -4.60368244955074, -3.94476404011562,
             -3.34785456738322, -2.78880605842813, -2.25497400208928,
             -1.73853771211659, -1.23407621539532, -.737473728545394,
             -.245340708300901, .245340708300901, .737473728545394,
             1.23407621539532, 1.73853771211659, 2.25497400208928,
             2.78880605842813, 3.34785456738322, 3.94476404011562,
             4.60368244955074, 5.38748089001123), device=self.loc.device, dtype=self.loc.dtype)
        weights = torch.as_tensor(
            (2.22939364553415e-13, 4.39934099227318e-10, 1.08606937076928e-7,
             7.80255647853206e-6, 0.000228338636016354, 0.00324377334223786,
             0.0248105208874636, 0.109017206020023, 0.286675505362834,
             0.46224366960061, 0.46224366960061, 0.286675505362834,
             0.109017206020023, 0.0248105208874636, 0.00324377334223786,
             0.000228338636016354, 7.80255647853206e-6, 1.08606937076928e-7,
             4.39934099227318e-10, 2.22939364553415e-13), device=self.loc.device, dtype=self.loc.dtype)
        latent = self.loc.unsqueeze(-1) + (2.0 ** .5) * self.scale.unsqueeze(-1) * nodes
        # Clamp before logarithm.  Clamping the *result* would leave an
        # infinite derivative at tanh(z)==+/-1 in float precision, which can
        # turn the entropy bonus into NaN for a boundary-seeking actor.
        jacobian = (1.0 - torch.tanh(latent).square()).clamp_min(self._endpoint_epsilon)
        log_jacobian = torch.log(jacobian)
        expected_log_jacobian = (log_jacobian * (weights / np.sqrt(np.pi))).sum(dim=-1)
        return self.base_dist.entropy() + expected_log_jacobian


class MixedReleaseTanhNormal:
    """Bernoulli WAIT/RELEASE plus two tanh-squashed residual heads.

    The external action remains the canonical three-vector.  Release samples
    are mapped to exact ``-1`` (WAIT) or ``+1`` (RELEASE); speed and merge keep
    the bounded stochastic distribution established by D3.
    """

    def __init__(self, logits: torch.Tensor, scale: torch.Tensor) -> None:
        self.logits = logits
        self.release = Bernoulli(logits=logits[..., 0])
        self.residual = TanhSquashedNormal(logits[..., 1:], scale[..., 1:])

    @property
    def release_probability(self) -> torch.Tensor:
        return self.release.probs

    @property
    def mean(self) -> torch.Tensor:
        expected_release = 2.0 * self.release_probability - 1.0
        return torch.cat((expected_release.unsqueeze(-1), self.residual.mean), dim=-1)

    @property
    def mode(self) -> torch.Tensor:
        release = torch.where(self.release_probability >= 0.5, 1.0, -1.0)
        return torch.cat((release.unsqueeze(-1), self.residual.mode), dim=-1)

    def sample(self) -> torch.Tensor:
        release = 2.0 * self.release.sample() - 1.0
        return torch.cat((release.unsqueeze(-1), self.residual.sample()), dim=-1)

    def rsample(self) -> torch.Tensor:
        # Bernoulli has no pathwise reparameterisation; PPO uses ``sample``.
        return self.sample()

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        release_label = action[..., 0].gt(0.0).to(action.dtype)
        release_log_prob = self.release.log_prob(release_label).unsqueeze(-1)
        return torch.cat((release_log_prob, self.residual.log_prob(action[..., 1:])), dim=-1)

    def entropy(self) -> torch.Tensor:
        return torch.cat((self.release.entropy().unsqueeze(-1), self.residual.entropy()), dim=-1)


def mlp(input_dim: int, output_dim: int, hidden: int = 96) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, output_dim))


class SharedActorCentralCritic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int = 3, hidden: int = 96) -> None:
        super().__init__()
        self.actor_body = mlp(observation_dim, hidden, hidden)
        self.actor_mean = nn.Linear(hidden, action_dim)
        # Start from a safe executable intent (request release, nominal speed,
        # modestly earlier merge) rather than a random no-release controller.
        # PPO remains free to move every component within [-1, 1].
        nn.init.zeros_(self.actor_mean.weight)
        with torch.no_grad(): self.actor_mean.bias.copy_(torch.tensor((1.2, 0.0, -0.50)))
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.7))
        self.critic_encoder = mlp(observation_dim, hidden, hidden)
        self.critic_head = mlp(2 * hidden, 1, hidden)

    def actor_distribution(self, observations: torch.Tensor, *, bounded: bool = True,
                           mixed_release: bool = False) -> TanhSquashedNormal | MixedReleaseTanhNormal | Normal:
        logits = self.actor_mean(self.actor_body(observations))
        scale = self.log_std.exp().expand_as(logits)
        if mixed_release:
            return MixedReleaseTanhNormal(logits, scale)
        # ``bounded=False`` exists solely for the frozen D3 control arm.  New
        # training and all normal call sites use the tanh-squashed policy.
        return TanhSquashedNormal(logits, scale) if bounded else Normal(torch.tanh(logits), scale)

    def critic(self, agent_sets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded = self.critic_encoder(agent_sets)
        valid = mask.unsqueeze(-1)
        count = valid.sum(dim=1).clamp_min(1)
        mean = (encoded * valid).sum(dim=1) / count
        maximum = encoded.masked_fill(~valid, float("-inf")).max(dim=1).values
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        return self.critic_head(torch.cat((mean, maximum), dim=-1)).squeeze(-1)


@dataclass
class Transition:
    """A rollout row whose ``action`` is always the policy-requested action."""
    agent_id: str
    observation: np.ndarray
    world: np.ndarray
    action: np.ndarray
    log_prob: float
    value: float
    reward: float
    done: bool
    teacher_action: np.ndarray | None = None
    applied_action: np.ndarray | None = None
    teacher_used: bool = False
    shield_used: bool = False
    log_prob_by_head: np.ndarray | None = None
    teacher_head_mask: np.ndarray | None = None


class RolloutBuffer:
    def __init__(self) -> None: self.by_agent: dict[str, list[Transition]] = defaultdict(list)
    def add(self, transition: Transition) -> None: self.by_agent[transition.agent_id].append(transition)
    def clear(self) -> None: self.by_agent.clear()
    def flattened(self, gamma: float, gae_lambda: float):
        rows = []
        for trace in self.by_agent.values():
            advantage = 0.0
            for index in range(len(trace) - 1, -1, -1):
                current = trace[index]
                next_value = 0.0 if current.done or index == len(trace) - 1 else trace[index + 1].value
                delta = current.reward + gamma * next_value - current.value
                advantage = delta + gamma * gae_lambda * (0.0 if current.done else advantage)
                rows.append((current, advantage, advantage + current.value))
        return list(reversed(rows))


class SharedPPO:
    def __init__(self, observation_dim: int, *, device: str = "cpu", learning_rate: float = 3e-4, clip_ratio: float = .2, value_coef: float = .5, entropy_coef: float = .01, teacher_bc_weight: float = 0.0,
                 policy_distribution: str = "tanh_squashed_gaussian", teacher_bc_loss: str = "action_smooth_l1",
                 intervention_ppo_mask: bool = False) -> None:
        if policy_distribution not in {"tanh_squashed_gaussian", "mixed_bernoulli_tanh", "gaussian_hard_clamp"}:
            raise ValueError("unsupported low-level policy distribution")
        if teacher_bc_loss not in {"action_smooth_l1", "squashed_nll"}:
            raise ValueError("unsupported teacher BC loss")
        if teacher_bc_loss == "squashed_nll" and policy_distribution != "tanh_squashed_gaussian":
            raise ValueError("squashed NLL BC requires the tanh-squashed policy")
        self.device = torch.device(device)
        self.model = SharedActorCentralCritic(observation_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.clip_ratio, self.value_coef, self.entropy_coef, self.teacher_bc_weight = clip_ratio, value_coef, entropy_coef, teacher_bc_weight
        self.observation_dim = observation_dim
        self.policy_distribution = policy_distribution
        self.teacher_bc_loss = teacher_bc_loss
        self.intervention_ppo_mask = intervention_ppo_mask

    @property
    def uses_bounded_distribution(self) -> bool:
        return self.policy_distribution in {"tanh_squashed_gaussian", "mixed_bernoulli_tanh"}

    @property
    def uses_mixed_release(self) -> bool:
        return self.policy_distribution == "mixed_bernoulli_tanh"

    def distribution(self, observations: torch.Tensor) -> TanhSquashedNormal | MixedReleaseTanhNormal | Normal:
        return self.model.actor_distribution(observations, bounded=self.uses_bounded_distribution,
                                             mixed_release=self.uses_mixed_release)

    def act(self, observations: np.ndarray, world: np.ndarray, *, deterministic: bool = False,
            return_head_log_probs: bool = False):
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        world_tensor = torch.as_tensor(world[None], dtype=torch.float32, device=self.device)
        mask = torch.ones((1, world.shape[0]), dtype=torch.bool, device=self.device)
        with torch.no_grad():
            distribution = self.distribution(obs)
            actions = distribution.mode if deterministic else distribution.sample()
            if not self.uses_bounded_distribution:
                # D3's deliberately inconsistent historical control only.
                actions = actions.clamp(-1, 1)
            log_prob_by_head = distribution.log_prob(actions)
            log_prob = log_prob_by_head.sum(dim=-1)
            value = self.model.critic(world_tensor, mask)[0]
        result = (actions.cpu().numpy().astype(np.float32), log_prob.cpu().numpy(), float(value.cpu()))
        return (*result, log_prob_by_head.cpu().numpy().astype(np.float32)) if return_head_log_probs else result

    def _teacher_loss_per_head(self, distribution: TanhSquashedNormal | MixedReleaseTanhNormal | Normal, means: torch.Tensor,
                               targets: torch.Tensor) -> torch.Tensor:
        """Return a [batch, head] BC loss without changing target semantics."""
        if self.uses_mixed_release:
            release_labels = targets[:, 0].gt(0.0).to(targets.dtype)
            release_loss = nn.functional.binary_cross_entropy_with_logits(
                distribution.logits[:, 0], release_labels, reduction="none")
            residual_loss = nn.functional.smooth_l1_loss(means[:, 1:], targets[:, 1:], reduction="none")
            return torch.cat((release_loss.unsqueeze(-1), residual_loss), dim=-1)
        if self.teacher_bc_loss == "squashed_nll":
            # TanhSquashedNormal.log_prob internally stabilizes exact +/-1
            # targets only while evaluating their density.
            return -distribution.log_prob(targets)
        return nn.functional.smooth_l1_loss(means, targets, reduction="none")

    @staticmethod
    def _teacher_correction_mask(rows: list[Transition], actions: torch.Tensor) -> torch.Tensor:
        targets = torch.as_tensor(np.stack([
            row.teacher_action if row.teacher_used and row.teacher_action is not None else row.action for row in rows
        ]), dtype=actions.dtype, device=actions.device)
        used = torch.as_tensor([row.teacher_used and row.teacher_action is not None for row in rows],
                               dtype=torch.bool, device=actions.device).unsqueeze(-1)
        return used & (targets - actions).abs().gt(1e-7)

    @staticmethod
    def _quantiles(values: list[float] | np.ndarray) -> dict[str, float | None]:
        array = np.asarray(values, dtype=np.float64)
        if array.size == 0:
            return {"mean": None, "median": None, "p10": None, "p90": None}
        return {"mean": float(array.mean()), "median": float(np.median(array)),
                "p10": float(np.percentile(array, 10)), "p90": float(np.percentile(array, 90))}

    @staticmethod
    def _grad_norm(loss: torch.Tensor, parameters: list[torch.Tensor], *, retain_graph: bool = True) -> float:
        gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
        squared = sum(float(gradient.detach().pow(2).sum()) for gradient in gradients if gradient is not None)
        return float(np.sqrt(squared))

    def diagnostic_snapshot(self, buffer: RolloutBuffer, *, teacher_bc_weight: float,
                            gamma: float = .99, gae_lambda: float = .95) -> dict[str, Any]:
        """Return non-mutating per-head and gradient evidence for one PPO update.

        ``Transition.action`` and ``Transition.log_prob`` remain the requested
        behaviour action and its log-probability.  This method never calls an
        optimizer, clips gradients, or changes a model parameter; it exists
        solely to make a frozen diagnostic update explainable.
        """
        rows = buffer.flattened(gamma, gae_lambda)
        heads = ("release_request", "speed_residual", "merge_time_residual")
        if not rows:
            return {"rows": 0, "heads": {name: {} for name in heads}, "gradient": {},
                    "advantages": {"raw": self._quantiles([]), "normalized": self._quantiles([])}}
        raw_advantages = np.asarray([item[1] for item in rows], dtype=np.float32)
        normalized = (raw_advantages - raw_advantages.mean()) / (raw_advantages.std() + 1e-8)
        transitions = [item[0] for item in rows]
        observations = torch.as_tensor(np.stack([row.observation for row in transitions]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.stack([row.action for row in transitions]), dtype=torch.float32, device=self.device)
        old_log_prob = torch.as_tensor([row.log_prob for row in transitions], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(normalized, dtype=torch.float32, device=self.device)
        logits = self.model.actor_mean(self.model.actor_body(observations))
        means = torch.tanh(logits)
        distribution = self.distribution(observations)
        new_log_prob = distribution.log_prob(actions).sum(-1)
        ratio = (new_log_prob - old_log_prob).exp()
        actor_loss_rows = -torch.minimum(ratio * advantages,
                                         ratio.clamp(1 - self.clip_ratio, 1 + self.clip_ratio) * advantages)
        actor_loss = actor_loss_rows.mean()
        worlds = [row.world for row in transitions]
        max_n = max(len(world) for world in worlds)
        padded = np.zeros((len(worlds), max_n, self.observation_dim), np.float32)
        mask = np.zeros((len(worlds), max_n), bool)
        for index, world in enumerate(worlds):
            padded[index, :len(world)] = world
            mask[index, :len(world)] = True
        returns = torch.as_tensor([item[2] for item in rows], dtype=torch.float32, device=self.device)
        values = self.model.critic(torch.as_tensor(padded, device=self.device), torch.as_tensor(mask, device=self.device))
        value_loss = (values - returns).pow(2).mean()
        actor_parameters = list(self.model.actor_body.parameters()) + list(self.model.actor_mean.parameters()) + [self.model.log_std]
        critic_parameters = list(self.model.critic_encoder.parameters()) + list(self.model.critic_head.parameters())
        actor_grad_norm = self._grad_norm(actor_loss, actor_parameters)
        critic_grad_norm = self._grad_norm(self.value_coef * value_loss, critic_parameters)

        action_array = actions.detach().cpu().numpy()
        mean_array = means.detach().cpu().numpy()
        logit_array = logits.detach().cpu().numpy()
        std_array = self.model.log_std.detach().exp().cpu().numpy()
        teacher_indices = [index for index, row in enumerate(transitions) if row.teacher_used and row.teacher_action is not None]
        teacher_targets = np.asarray([transitions[index].teacher_action for index in teacher_indices], dtype=np.float32) \
            if teacher_indices else np.empty((0, 3), dtype=np.float32)
        head_rows: dict[str, dict[str, Any]] = {}
        head_gradient_rows: dict[str, dict[str, Any]] = {
            name: {"ppo": [], "bc": [], "combined": [], "cosine": [], "pre_mask_cosine": [], "credit_samples": []}
            for name in heads
        }
        parameter_blocks = [self.model.actor_mean.weight, self.model.actor_mean.bias, self.model.log_std]

        def head_gradient(loss: torch.Tensor, head_index: int) -> np.ndarray:
            gradients = torch.autograd.grad(loss, parameter_blocks, retain_graph=True, allow_unused=True)
            weight = np.zeros_like(self.model.actor_mean.weight[head_index].detach().cpu().numpy()) if gradients[0] is None \
                else gradients[0][head_index].detach().cpu().numpy()
            bias = np.zeros(1, dtype=np.float32) if gradients[1] is None else gradients[1][head_index:head_index + 1].detach().cpu().numpy()
            log_std = np.zeros(1, dtype=np.float32) if gradients[2] is None else gradients[2][head_index:head_index + 1].detach().cpu().numpy()
            return np.concatenate((weight.reshape(-1), bias.reshape(-1), log_std.reshape(-1))).astype(np.float64, copy=False)

        for head_index, head in enumerate(heads):
            bounded = np.abs(action_array[:, head_index]) >= .98
            exact_bound = np.abs(action_array[:, head_index]) >= 1.0 - 1e-7
            mean_boundary = np.abs(mean_array[:, head_index]) >= .98
            head_rows[head] = {
                "pre_squash_logit": self._quantiles(logit_array[:, head_index]),
                "post_squash_actor_mean": self._quantiles(mean_array[:, head_index]),
                "std": float(std_array[head_index]), "log_std": float(self.model.log_std.detach().cpu()[head_index]),
                "post_bound_requested_action": self._quantiles(action_array[:, head_index]),
                "saturation_rate": float(bounded.mean()), "exact_bound_pile_rate": float(exact_bound.mean()),
                "actor_mean_boundary_rate": float(mean_boundary.mean()),
                "sample_bound_hit_away_from_mean_rate": float((bounded & ~mean_boundary).mean()),
                "entropy": float(distribution.entropy()[:, head_index].mean().detach()),
                "teacher_correction_rate": float(len(teacher_indices) / len(transitions)),
                "teacher_target": self._quantiles(teacher_targets[:, head_index]) if len(teacher_targets) else self._quantiles([]),
                "teacher_target_saturation_rate": None if not len(teacher_targets) else float((np.abs(teacher_targets[:, head_index]) >= .98).mean()),
                "absolute_requested_teacher_difference": self._quantiles(
                    np.abs(action_array[teacher_indices, head_index] - teacher_targets[:, head_index]) if len(teacher_targets) else []),
            }
        for row_index in teacher_indices:
            one_observation = observations[row_index:row_index + 1]
            one_action = actions[row_index:row_index + 1]
            one_logits = self.model.actor_mean(self.model.actor_body(one_observation))
            one_mean = torch.tanh(one_logits)
            one_distribution = self.distribution(one_observation)
            one_ratio = (one_distribution.log_prob(one_action).sum(-1) - old_log_prob[row_index:row_index + 1]).exp()
            one_advantage = advantages[row_index:row_index + 1]
            one_ppo = -torch.minimum(one_ratio * one_advantage,
                                     one_ratio.clamp(1 - self.clip_ratio, 1 + self.clip_ratio) * one_advantage).mean()
            target = torch.as_tensor(transitions[row_index].teacher_action, dtype=torch.float32, device=self.device)
            for head_index, head in enumerate(heads):
                one_bc = self._teacher_loss_per_head(one_distribution, one_mean, target.unsqueeze(0))[0, head_index]
                pre_mask_ppo_gradient = head_gradient(one_ppo, head_index)
                corrected = bool(abs(float(transitions[row_index].teacher_action[head_index] -
                                     transitions[row_index].action[head_index])) > 1e-7)
                masked = bool(self.intervention_ppo_mask and corrected)
                ppo_gradient = np.zeros_like(pre_mask_ppo_gradient) if masked else pre_mask_ppo_gradient
                bc_gradient = head_gradient(one_bc, head_index)
                combined_gradient = ppo_gradient + teacher_bc_weight * bc_gradient
                ppo_norm, bc_norm = np.linalg.norm(ppo_gradient), np.linalg.norm(bc_gradient)
                pre_mask_norm = np.linalg.norm(pre_mask_ppo_gradient)
                cosine = None if ppo_norm <= 1e-12 or bc_norm <= 1e-12 else float(np.dot(ppo_gradient, bc_gradient) / (ppo_norm * bc_norm))
                pre_mask_cosine = None if pre_mask_norm <= 1e-12 or bc_norm <= 1e-12 else float(
                    np.dot(pre_mask_ppo_gradient, bc_gradient) / (pre_mask_norm * bc_norm))
                head_gradient_rows[head]["ppo"].append(float(ppo_norm))
                head_gradient_rows[head]["bc"].append(float(bc_norm))
                head_gradient_rows[head]["combined"].append(float(np.linalg.norm(combined_gradient)))
                if cosine is not None:
                    head_gradient_rows[head]["cosine"].append(cosine)
                if pre_mask_cosine is not None:
                    head_gradient_rows[head]["pre_mask_cosine"].append(pre_mask_cosine)
                if corrected:
                    head_gradient_rows[head]["credit_samples"].append({
                        "requested_action": float(transitions[row_index].action[head_index]),
                        "teacher_action": float(transitions[row_index].teacher_action[head_index]),
                        "advantage_raw": float(raw_advantages[row_index]),
                        "advantage_normalized": float(normalized[row_index]),
                        "correction_direction": float(np.sign(transitions[row_index].teacher_action[head_index] - transitions[row_index].action[head_index])),
                        "ppo_gradient_norm": float(pre_mask_norm), "bc_gradient_norm": float(bc_norm),
                        "ppo_bc_cosine_pre_mask": pre_mask_cosine,
                        "ppo_gradient_masked": masked,
                        "teacher_rescue_credit_leakage": bool(raw_advantages[row_index] > 0.0 and pre_mask_cosine is not None and pre_mask_cosine < 0.0),
                    })
        for head, values_by_kind in head_gradient_rows.items():
            head_rows[head]["teacher_used_head_output_gradients"] = {
                kind: self._quantiles(values) for kind, values in values_by_kind.items()
                if kind in {"ppo", "bc", "combined"}
            }
            cosines = values_by_kind["cosine"]
            head_rows[head]["teacher_used_head_output_gradient_cosine"] = {
                **self._quantiles(cosines), "negative_fraction": None if not cosines else float(np.mean(np.asarray(cosines) < 0.0)),
                "samples": len(cosines),
            }
            pre_mask = values_by_kind["pre_mask_cosine"]
            samples = values_by_kind["credit_samples"]
            leakage = [sample for sample in samples if sample["teacher_rescue_credit_leakage"]]
            active_leakage = [sample for sample in leakage if not sample["ppo_gradient_masked"]]
            head_rows[head]["teacher_rescue_credit_assignment"] = {
                "corrected_samples": len(samples),
                "corrected_positive_advantage_samples": int(sum(sample["advantage_raw"] > 0.0 for sample in samples)),
                "corrected_positive_advantage_negative_cosine_samples": len(leakage),
                "corrected_positive_advantage_negative_cosine_fraction": None if not samples else float(len(leakage) / len(samples)),
                "active_credit_leakage_samples": len(active_leakage),
                "active_credit_leakage_fraction": None if not samples else float(len(active_leakage) / len(samples)),
                "pre_mask_cosine": {**self._quantiles(pre_mask), "negative_fraction": None if not pre_mask else float(np.mean(np.asarray(pre_mask) < 0.0))},
                "ppo_masked_corrected_samples": int(sum(sample["ppo_gradient_masked"] for sample in samples)),
                "samples": samples,
            }
        return {
            "rows": len(rows), "teacher_used_rows": len(teacher_indices), "heads": head_rows,
            "advantages": {"raw": self._quantiles(raw_advantages), "normalized": self._quantiles(normalized)},
            "optimizer": {"learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                          "entropy_coefficient": float(self.entropy_coef), "clip_ratio": float(self.clip_ratio)},
            "ppo": {"approx_kl": float((old_log_prob - new_log_prob).mean().detach()),
                    "clip_fraction": float(((ratio - 1.0).abs() > self.clip_ratio).float().mean().detach()),
                    "actor_loss": float(actor_loss.detach()), "value_loss": float(value_loss.detach()),
                    "value_to_abs_actor_loss": float(value_loss.detach() / actor_loss.detach().abs().clamp_min(1e-8)),
                    "explained_variance": float((1.0 - (returns - values).var(unbiased=False) / returns.var(unbiased=False).clamp_min(1e-8)).detach())},
            "gradient_blocks": {"actor": actor_grad_norm, "critic": critic_grad_norm,
                                "shared_backbone": 0.0, "shared_backbone_note": "not_applicable: actor_body and critic_encoder are disjoint"},
        }

    def update(self, buffer: RolloutBuffer, *, epochs: int = 4, minibatch_size: int = 128, gamma: float = .99, gae_lambda: float = .95) -> dict[str, float]:
        rows = buffer.flattened(gamma, gae_lambda)
        metric_names = ("loss", "actor_loss", "value_loss", "entropy", "teacher_bc_loss", "approx_kl",
                        "clip_fraction", "explained_variance", "grad_norm", "intervention_masked_head_fraction")
        if not rows:
            return {name: 0.0 for name in metric_names}
        advantages = np.asarray([x[1] for x in rows], np.float32); advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        totals = defaultdict(float); updates = 0
        for _ in range(epochs):
            permutation = np.random.permutation(len(rows))
            for begin in range(0, len(rows), minibatch_size):
                choices = permutation[begin:begin + minibatch_size]
                selected = [rows[i] for i in choices]
                observations = torch.as_tensor(np.stack([x[0].observation for x in selected]), dtype=torch.float32, device=self.device)
                actions = torch.as_tensor(np.stack([x[0].action for x in selected]), dtype=torch.float32, device=self.device)
                old_log_prob = torch.as_tensor([x[0].log_prob for x in selected], dtype=torch.float32, device=self.device)
                returns = torch.as_tensor([x[2] for x in selected], dtype=torch.float32, device=self.device)
                adv = torch.as_tensor(advantages[choices], dtype=torch.float32, device=self.device)
                worlds = [x[0].world for x in selected]; max_n = max(len(x) for x in worlds)
                padded = np.zeros((len(worlds), max_n, self.observation_dim), np.float32); mask = np.zeros((len(worlds), max_n), bool)
                for i, world in enumerate(worlds): padded[i, :len(world)] = world; mask[i, :len(world)] = True
                distribution = self.distribution(observations)
                new_log_prob_by_head = distribution.log_prob(actions)
                new_log_prob = new_log_prob_by_head.sum(-1)
                masked_head_fraction = torch.zeros((), device=self.device)
                if self.intervention_ppo_mask:
                    if any(row[0].log_prob_by_head is None for row in selected):
                        raise RuntimeError("head-wise intervention PPO mask requires stored requested log-probabilities by head")
                    correction_mask = self._teacher_correction_mask([row[0] for row in selected], actions)
                    # Retain the joint PPO ratio value, but detach any corrected
                    # head from its actor gradient.  Thus every uncorrected head
                    # retains its ordinary PPO surrogate while a teacher-rescued
                    # head has exactly zero PPO actor gradient.
                    effective_log_prob = torch.where(correction_mask, new_log_prob_by_head.detach(), new_log_prob_by_head).sum(-1)
                    new_log_prob_for_actor = effective_log_prob
                    masked_head_fraction = correction_mask.float().mean()
                else:
                    new_log_prob_for_actor = new_log_prob
                log_ratio = new_log_prob_for_actor - old_log_prob
                ratio = log_ratio.exp()
                actor_loss = -torch.minimum(ratio * adv, ratio.clamp(1-self.clip_ratio, 1+self.clip_ratio) * adv).mean()
                values = self.model.critic(torch.as_tensor(padded, device=self.device), torch.as_tensor(mask, device=self.device))
                value_loss = (values - returns).pow(2).mean()
                entropy = distribution.entropy().sum(-1).mean()
                teacher_rows = [row for row in selected if row[0].teacher_action is not None and (
                    row[0].teacher_used or (row[0].teacher_head_mask is not None and np.asarray(row[0].teacher_head_mask).any()))]
                if teacher_rows:
                    teacher_obs = torch.as_tensor(np.stack([row[0].observation for row in teacher_rows]), dtype=torch.float32, device=self.device)
                    teacher_targets = torch.as_tensor(np.stack([row[0].teacher_action for row in teacher_rows]), dtype=torch.float32, device=self.device)
                    teacher_logits = self.model.actor_mean(self.model.actor_body(teacher_obs))
                    teacher_means = torch.tanh(teacher_logits)
                    teacher_distribution = self.distribution(teacher_obs)
                    teacher_loss_by_head = self._teacher_loss_per_head(teacher_distribution, teacher_means, teacher_targets)
                    teacher_masks = torch.as_tensor(np.stack([
                        row[0].teacher_head_mask if row[0].teacher_head_mask is not None else np.ones(3, dtype=np.float32)
                        for row in teacher_rows
                    ]), dtype=torch.float32, device=self.device)
                    teacher_loss = (teacher_loss_by_head * teacher_masks).sum() / teacher_masks.sum().clamp_min(1.0)
                else:
                    teacher_loss = torch.zeros((), device=self.device)
                loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy + self.teacher_bc_weight * teacher_loss
                if not torch.isfinite(loss): raise FloatingPointError("non-finite PPO loss")
                approx_kl = (old_log_prob - new_log_prob).mean()
                clip_fraction = ((ratio - 1.0).abs() > self.clip_ratio).float().mean()
                variance = returns.var(unbiased=False)
                explained_variance = 1.0 - (returns - values).var(unbiased=False) / variance.clamp_min(1e-8)
                self.optimizer.zero_grad(); loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                totals["loss"] += float(loss.detach()); totals["actor_loss"] += float(actor_loss.detach()); totals["value_loss"] += float(value_loss.detach()); totals["entropy"] += float(entropy.detach()); totals["teacher_bc_loss"] += float(teacher_loss.detach())
                totals["approx_kl"] += float(approx_kl.detach()); totals["clip_fraction"] += float(clip_fraction.detach())
                totals["explained_variance"] += float(explained_variance.detach()); totals["grad_norm"] += float(grad_norm.detach()); updates += 1
                totals["intervention_masked_head_fraction"] += float(masked_head_fraction.detach())
        buffer.clear()
        return {key: value / max(1, updates) for key, value in totals.items()}
