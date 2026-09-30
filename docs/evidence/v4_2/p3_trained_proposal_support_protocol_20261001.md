# P3 Trained Checkpoint Proposal-Support Protocol

Date: 2026-10-01. This is a bounded development comparison for the rejected
`d449` checkpoint. It does not train, update, promote, consume held-out data,
or authorize PPO.

## Fixed inputs

- Before checkpoint: `3fe7dd1e01d032227521f6fc5c8464fe8d00d98fb0e9f0dbd13411123a27d4ab`.
- After checkpoint: `d44990dc7d129573d9145f169df1cbc4f91fa32bc5127b1c58d937260c02a24f`.
- Warmup checkpoint binding: `a82ed7a287035fb1fc868fc955e056c0481de75bcb99880d5c79f35588dde710`.
- Configuration: the original task-neutral `dense24` configuration, V2.1/88D,
  K=3, `fixed_low_student`, execution scheduler, safety and physical paths
  unchanged, with an 1800-second simulation cap.
- Four original cells are paired by environment and policy seeds:
  `(875101,875201)`, `(875102,875202)`, `(875103,875203)`, and
  `(875104,875204)`. Each arm must generate and admit 24 requests.

## Arms and sampler boundary

Each cell contains `before_legacy`, `after_legacy`, and `after_support`.
The first two call the native `student_rollout_v4_2` request sampler unchanged.
The support arm temporarily replaces only
`student_rollout_v4_2._student_request_details` and restores it in `finally`.
It keeps the source gate distribution, actor mask, physics, safety, energy,
emergency, and capacity paths unchanged. On sampled RELEASE rows only, the
proposal distribution is

`q_mix = 0.9 * (q / sum(q)) + 0.1 / 25`.

The sampled proposal is rank one in the K=3 shortlist; the remaining two
positions use the original source proposal logits with stable ordering. The
recorded proposal likelihood is conditional on `q_mix`. Source q, logits, and
gate outputs are checked for immutability. An epsilon-zero helper delegates
directly to the original request-details function for parity tests and is not
a production arm.

Each arm saves and restores the caller RNG state. The native rollout keeps its
own policy seed behavior. Because the intervention can change trajectory and
batch shape, this protocol does not claim a common random-number trajectory or
an identified causal effect.

For accounting, the runner uses the already-qualified passive
`_CapturedTerminalEnv` request observer through the scoped student environment
factory. It returns the native environment subclass without extra steps or
resampling, records the actual request rows created across reset/warm-start,
and restores the factory in `finally`. Generated requests, ledger requested
and resolved IDs, active/service/completed admitted IDs, `energy_outcomes`,
`arrivals_admitted`, and `release_rejected` are retained separately. The
rejection-only reward component is computed from the configured rejected
release count; the native `release` reward component is not relabeled as a
penalty. A natural terminal arm must have the complete 24-request lifecycle,
request identity, ledger, energy coverage, finite nonnegative wait, elapsed,
and energy rows, and all five safety fields (`collisions`, `nmac`,
`hard_separation_violations`, `uncertified_execution`, and
`SafetyInvariantError`). Censored arms retain observed outcome counts and IDs
while complete-scene wait, elapsed, energy, and paired deltas remain unknown;
their partial generated/request ledger is checked against the actual captured
request count. The evaluator's `IncompleteLifecycle` marker is retained as
`lifecycle_qualification_error` for a nonterminal, replay-exact, hard-safe
prefix and is treated as censor qualification metadata; an independent replay,
source, runtime, or safety error remains technical. A natural terminal record
with missing or nonfinite required fields is a technical failure rather than a
qualified censor.

## Execution phases and gates

`preflight` creates no environment. It binds the three checkpoint hashes,
model dimensions/state keys, empty optimizer state, configuration snapshot,
cell grid, sampler constants, runner hash, protocol hash, and the bitwise frozen
actor encoder, proposal head, gate head, and gate temperature groups. Only the
registered gate-adapter and critic groups may differ between the before and
after checkpoints.

`pilot` runs all three arms for cell 1 and writes immutable arm and pair
artifacts. `full` verifies the pilot artifact hashes and binding, reuses cell 1
without rerunning it, and runs only cells 2–4. A technical, source, RNG,
ledger, finite-metric, or hard-safety failure is retained and stops that phase.
Business failure or censoring is retained with unknown complete-scene deltas;
it is not removed or retried.

The report keeps generated/admitted/successful/resolved counts, normal and
emergency outcomes, failed/diverted/unresolved IDs, cumulative wait, request
elapsed time, energy, raw reward, and rejection penalty separate. It reports
before-to-after legacy change and support-to-after-legacy change separately.
The development candidate screen requires all four cells natural and
technical-safe, no new failed/diverted/unresolved IDs, and, across the four
cells, after-support normal and successful counts at least as high as
after-legacy, total wait no higher, and a strict business gain. Passing this
screen would remain a development result and would not qualify P3 or enable
PPO.

The previously observed 80-to-74 normal-completion change is historical
context only until this exact native-sampler comparison reproduces it under the
current runtime binding.
