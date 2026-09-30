# P3 proposal-support control protocol (2026-10-01)

## Status

This document pre-registers a bounded, frozen off-policy proposal intervention.
It is a coverage test for the proposal distribution and is not a P3
qualification or a training run. `PPO_ALLOWED`, teacher use, optimizer use, held-out evaluation,
and log-probability recording are all `false`.

## Fixed comparison

The source is the registered `3fe7+a82` model and the existing 88D V2.1,
`K=3`, `max_steps=1800` legacy reference arms. The fixed pair order is
`1,5,9,13,17,21,14,15,16`; each pair keeps its registered environment seed,
keyed policy seed, arrival IDs, absolute ticks, and stream names. Existing
reference artifacts are reused. The control does not rerun a baseline arm.

For each legacy actor-mask row whose source gate sample is `RELEASE`, proposal
sampling uses the single fixed mixture

```text
q_mix = 0.90 * (q / sum(q)) + 0.10 / 25
```

with `q_floor_min=0.004`. All 25 proposal values remain in the categorical
support. The same keyed proposal uniform is used for the source and mixture
comparisons. `WAIT` rows have no proposal. At the same observed state, source
and intervention use the same gate probabilities, keyed gate draw, and legacy
actor mask. After a proposal changes the executed trajectory, later states and
later gate actions may diverge; the protocol does not claim per-tick gate
identity across the two trajectories. Gate probabilities, actor masks,
source proposal logits, sampled-proposal-first K3 shortlist construction,
physics, safety, energy, emergency, capacity, and execution code remain
unchanged. The source `q` and logits are checked for bitwise preservation in
the request helper; no argmax or eligibility oracle is allowed.

The keyed contract is `arrival_id|absolute_tick|stream`. Caller RNG is saved
and restored around each request. The intervention is frozen off-policy and records no
training likelihood. Branch failures, censoring, unresolved requests, and
technical errors are retained; a technical or hard-safety error stops the
phase without a retry or resampling.

## Phases and acceptance fields

`preflight` creates no environment and binds the runner, source/config,
runtime, and all nine reference artifacts. `pilot` runs pair 1 once. `full`
reuses that pilot and runs only the other eight registered pairs. Output root:

`/home/user/training_outputs/terminal/p3_followup_20260930/proposal_support_control_20261001`

The business review reports normal/emergency/failed/diverted/unresolved
outcomes, generated/admitted/completed/resolved denominators, all-request wait,
request elapsed time, energy, raw reward excluding rejection penalty, and the
penalty separately. A development candidate requires all nine natural and
hard-safe scenes, no new failed/diverted/unresolved arrival IDs, 24 admitted
requests per scene, successful completion no lower than the paired reference,
total wait no higher, and a strict business improvement. Unknown metrics remain
unknown when a scene is censored or otherwise incomplete. Passing this review
does not promote the method or resolve formal P3 qualification.

## Verification artifacts

The implementation is
`terminal_marl/training/run_p3_proposal_support_control_20261001.py`, with
contract tests in `tests/test_p3_proposal_support_control.py`. Tests cover
epsilon-zero equivalence to the frozen keyed request, keyed RNG neutrality,
floor and mass validation, gate/mask and WAIT preservation, source q/logit
immutability, K3 shortlist ordering, hook restoration, and retained unknown
metrics. No server phase is authorized by this protocol alone.
