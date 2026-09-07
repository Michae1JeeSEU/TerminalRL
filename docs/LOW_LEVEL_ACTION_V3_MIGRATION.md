# Low-Level Action V3 migration

Low-Level Action V3 supersedes the continuous-action learning experiments for
policy-learning conclusions. It does not change terminal-airspace scheduling
semantics, vehicle profiles, reward, observation V2, or SafetyFilter authority.

## Canonical policy action

```text
release: Categorical(2) = WAIT / RELEASE
speed:   Categorical(5) = {-1, -.5, 0, .5, 1} normalized residual
merge:   Categorical(5) = {-1, -.5, 0, .5, 1} normalized residual
```

PPO uses the sum of the three head log-probabilities and the sum of their
entropies. The rollout buffer stores the sampled categories and their original
joint log-probability. The environment receives only their normalized adapter
values. `a_teacher` and `a_applied` are recorded separately and never replace
the sampled action in a PPO ratio.

## Profile-independent execution

The policy retains normalized levels. `SimConfig.max_speed_residual_mps` and
`SimConfig.max_merge_time_residual_seconds` remain the only mappings to the
active vehicle profile (currently approximately ±2 m/s and ±6 s). Thus a
Crazyflie profile can use the identical policy catalogue with another adapter
authority.

## Teacher, safety and replay

Teacher candidates are quantized once to the same V3 catalogue before safety
verification. BC is head-wise categorical cross-entropy. SafetyFilter,
PredictiveSafetyShield, joint candidate search, backup behaviour, and fixed
`physics_dt=0.2` / `safety_sampling_dt=1.0` are unchanged in authority.

V3 ActionTape records requested category/index, normalized values, per-head
and joint log-probability, executed mapped residual, observation hash, and
post-state hash. Legacy continuous tapes remain readable only as legacy traces;
they are not V3 exact-replay evidence.

## Checkpoints and datasets

Continuous V2 checkpoints are intentionally incompatible with V3 heads and
must not be resumed as V3 policies. L0 continuous teacher labels are not BC
targets. A V3 L0 dataset must be regenerated unless a separately recorded,
deterministic migration proves every requested/teacher/execution label maps
exactly to the V3 catalogue.
