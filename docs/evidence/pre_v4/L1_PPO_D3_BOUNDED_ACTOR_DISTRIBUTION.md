# L1-PPO-D3 Bounded Actor Distribution Repair

## Frozen scope

This diagnostic changes only the low-level actor distribution. It retains the
normalized three-dimensional request action, teacher target semantics,
`merge_time_residual * 6 s` execution conversion, speed scaling, reward,
teacher/safety stack, Observation V2, checkpoint, seeds, PPO hyperparameters,
and traffic configuration. It is not an authorization for L1b, L2, or a
formal long PPO run.

## Audit result

The historical chain was:

```text
z ~ Normal(tanh(logit), std)
→ a_requested = clamp(z, -1, 1)
→ store a_requested and Normal.log_prob(a_requested)
→ PPO ratio
```

This is not the density of the sampled-and-clamped random variable. Its
probability mass outside either bound is collapsed into point masses at `-1`
and `+1`, whereas `Normal.log_prob(±1)` is a continuous pre-clamp density.
Thus a hard-clamped endpoint is both an artificial pile-up and not a
mathematically consistent PPO behaviour distribution.

The repaired chain is:

```text
z ~ Normal(logit, std)
→ a_requested = tanh(z)
→ store a_requested and [log p(z) - log(1 - a_requested²)]
→ PPO ratio
```

The Jacobian correction is used whenever a stored bounded action is evaluated.
Evaluation uses `tanh(logit)` deterministically. Internal `atanh` evaluation
uses epsilon only for numerical stability at an externally supplied exact
endpoint; teacher labels themselves remain exact `±1` and the BC loss remains
in action space.

## D3 outputs and acceptance rule

`terminal_marl.training.l1_ppo_d3_bounded_distribution` runs exactly three
PPO updates for the historical `gaussian_hard_clamp` control and the repaired
`tanh_squashed_gaussian` arm. It records requested-action provenance,
exact/near-bound rates, actor mean/std, teacher/shield and correction burden,
head-wise PPO/BC gradient cosine, PPO health, completion/throughput, and all
hard-safety counters.

The bounded arm passes only if requested-action provenance, safety and PPO
health pass; release and merge exact-bound pile rates fall below 0.1% and below
the control; and teacher burden, correction P95, and shield burden do not
worsen. A later log-std/entropy ablation is allowed only when the exact pile
is repaired but material near-boundary saturation remains.
