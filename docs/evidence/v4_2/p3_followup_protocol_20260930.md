# P3 follow-up diagnostic protocol (2026-09-30)

This protocol defines a bounded, source-bound diagnostic for the current P3
task-neutral policy. It is an evidence collection procedure rather than a
performance evaluation protocol.

## Fixed inputs

- Scenario: `dense24`.
- Legacy low-level observation schema: V2.1, 88 dimensions.
- Action schema: hierarchical low-level action V4.2.
- Frozen source policy: checkpoint `3fe7dd1e...a27d4ab` followed by the
  registered critic-only warmup `a82ed7a2...8dde710`; the warmup metadata must
  report `gamma=1.0` and `reward_scale=0.01`.
- Six development tapes are used, with environment seeds `877101` through
  `877106`, policy seeds `878201` through `878206`, and recorded step counts
  `(725, 864, 699, 729, 724, 756)`.
- Every tape has 24 recorded request rows. The total replay budget is 10,800
  steps, with a per-tape cap of 1,000 steps.
- `PPO_ALLOWED=false`; no optimizer, teacher, held-out input, or new reward,
  mask, observation, or proposal definition is permitted.

## Phases

`preflight` reads and hashes the six records, action tapes, source and warmup
checkpoints, and the registered after-checkpoint summary. It does not create an
environment or replay a tape.

`replay` exactly replays all six tapes with the frozen source policy. It records
lightweight pre-step state context and eligible actor occurrences, then sorts
occurrences by `(episode, step, uid)` and samples 24 without replacement using
`numpy.random.default_rng(9303001)`. It writes a selection artifact and an
exact replay ledger.

`pilot` replays the source tapes again and deep-copies only the first selected
pre-step environment. It runs the paired WAIT plus 25 RELEASE branches for
continuation seed `9303101`, with a maximum of 1,800 continuation steps.

`diagnose` replays the source tapes again and deep-copies only the 24 selected
pre-step environments. Each selected state is evaluated with continuation
seeds `9303101` through `9303104`, preserving the paired 26-branch structure
and writing one state JSON atomically at a time.

For the registered compute schedule, `diagnose --shard-count 4` assigns a
state with global `selection_order` (s) to shard
`(s - 1) % 4`. Shards therefore process global orders 1,5,9,…; 2,6,10,…;
3,7,11,…; and 4,8,12,… respectively. Each shard receives six of the same
registered 24 states, preserves global state filenames and ordering, and writes
to its own output directory. `--shard-index` is zero based. Non-default shard
arguments are accepted only by `diagnose`; preflight, replay, and pilot retain
the single default schedule. The four runs use `OMP_NUM_THREADS=1` and
`MKL_NUM_THREADS=1`; this is scheduling only and does not add an experimental
factor or change the estimator.

## Recorded four-shard completion

The final named group reports under `causal/shard_0` through `causal/shard_3`
all report `COMPLETED`. Each shard contains six selected states and 624 branch
rows; together they contain 24 states and 2,496 WAIT/RELEASE branch rows. The
four `diagnose.json` modification times were `19:12:51`, `18:23:26`,
`19:19:08`, and `18:14:25` for shards 0 through 3 respectively. All 24 state
reports have complete `D` and `m`; all 2,496 branches are
`natural_terminal`, with zero errors and zero censored branches. The state
reports mark model, environment, and caller RNG unchanged as `true`.

The reports bind source checkpoint SHA256
`3fe7dd1e01d032227521f6fc5c8464fe8d00d98fb0e9f0dbd13411123a27d4ab`, warmup
SHA256 `a82ed7a287035fb1fc868fc955e056c0481de75bcb99880d5c79f35588dde710`,
and the after batch-01 checkpoint SHA256
`a7bcf3d1e6f4bda82d0d051ec25966a18f267309f1d6030689b52d90e63aa366`.
`PPO_ALLOWED=false`, held-out input is false, optimizer calls are zero, and
teacher use is false in every shard. The maximum absolute difference between a
state `D` and the mean of its four seed differences is
`7.1054273576010019e-14`; the exact per-state table is recorded in the results
document.

This completion record is a bounded conditional diagnostic. Fourteen states are
recorded as pure `safety_wait`; root selected the separate frozen mask
comparison described in [the forced-WAIT protocol](p3_forced_wait_frozen_protocol_20260930.md).
This diagnostic changes no mask or PPO. The `3fe7+a82` representative and the
independent source16 / batch-02 episode-24 failure-mechanism record remain
separate from this 24-state group.

For every paired diagnostic, the target actor's first action is varied while
the other actors' first actions and the recorded
`runtime_proposal_shortlists` remain fixed to that tape row. The continuation
uses the source policy without updating its parameters. The 24-by-4 result is
therefore a development conditional diagnostic, not a population performance
claim.

## Alignment endpoint

The alignment report compares the source policy with the actual after-checkpoint
produced by episodic-learning batch 01. The after binding is accepted only when
its summary reports the registered source digest, checkpoint SHA, and after
model digest. Each selected state receives an alignment record on the target
actor's one-row world input. Its `delta_q` is the paired diagnostic's qualified
value when the branch group is complete; for a censored group it remains
unknown (`None`), so `local_m` remains unknown as well. No target value is
copied to other actors. The rejected `d449` endpoint is not used.

## State and failure contract

The runner writes `preflight.json`, `selection.json`, phase reports,
`summary.json`, and `progress.json` under the requested output directory.
Selected deep copies restore native bound environment methods before use and
retain business state. Any exact-replay, lifecycle, ledger, binding, branch
count, or serialization failure writes a `FAILED` report and stops the phase.
