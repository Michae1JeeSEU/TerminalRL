# P3 gate-anchor state collection protocol — 2026-10-01

## Scope and status

This protocol defines a replay-only preparation artifact for future shared-policy
gate fitting.  It is not a training, qualification, performance, or root-cause
experiment.  `PPO_ALLOWED=false`; no optimizer, teacher, held-out split, policy
forward, action sampler, or new rollout is allowed.  Pilot and full phases
require a later primary-agent review after the no-environment preflight.

## Frozen inputs

The collector uses the four immutable BEFORE-prefix tapes already bound by
`diagnose_p3_prefix_gate_events_20261001` and its `_load_cell_inputs` helper.
The source full artifact is SHA256
`30c28c03d26466eabeab2a35c9fc117424648f1a3ae17be6f8d69b6b92cb2195`; the
prefix-event full artifact is SHA256
`fb89f62339755f7ebcd2bb0f9672438000fe8b754067cd2cdb28bf8407feb3e6`.
The registered replay lengths are cell 1: 708, cell 2: 674, cell 3: 644,
and cell 4: 723, for 2,749 source ticks.  Source checkpoint bindings and the
existing config snapshot are read from the mature helpers rather than copied
into this protocol.

## Recorded state contract

Immediately before each original tape step, the observer calls the native
environment observation method and validates the source observation hash.  It
then calls `matched._world_from_observations`; the returned UID order is saved
explicitly with one float32 88D row per active actor.  Rows include WAIT,
RELEASE, masked, and deterministic actors without outcome or probability
filtering.  A step with no active observations remains in step metadata with
an empty UID list; no synthetic actor row is created.

Each row is keyed by `cell|step|numeric_uid` and stores the source observation
hash, UID, arrival ID, phase, SOC, retry state, committed-plan state, actor
mask, source `requested_gate`, and source `requested_proposal`.  Missing source
values remain JSON `null`.  The companion NPZ stores `world`, `step`, `uid`,
`arrival_id`, `row_index`, and integer-coded gate/mask/proposal arrays with
`world.shape[1] == 88` and `world.dtype == float32`.  JSON metadata points to
the NPZ field and row index; the NPZ SHA is recorded after atomic writing.

Counts for `requested_gate` and `actor_mask` are reported separately as
`true`, `false`, and `unknown`; they are descriptive denominators, not labels
for a future fit.  Existing safety provenance fields are copied when present
and otherwise remain null.  No new safety certificate is generated.

## Replay and phase rules

The collector reuses `_CapturedTerminalEnv`, `_controller_scope`,
`matched._world_from_observations`, `base._actor_context`, and
`SafetyTraceRecorder.replay_action_tape`.  It does not alter the action tape,
physics, controller, or source action.  The temporary `env.step` observer and
controller scope are restored in `finally` blocks, and the caller RNG is
restored on success and failure.  Exact replay requires the native replay
result to report `exact=true` and the registered step count; technical errors
write partial/error artifacts and stop without retry.

`preflight` creates no environment and binds the source/event full SHAs,
cell/tape/sidecar SHAs, config, source bindings, runner, and protocol.  It
refuses an existing output directory.  `pilot` replays cell 1 once after
preflight.  `full` reuses the immutable pilot cell and replays only cells 2–4;
it refuses existing cell artifacts.  The output is anchor coverage, not a new
performance population, and contains no `D` or global `m` value.

## Later use boundary

These anchors may be considered for a separately registered shared-policy
soft-target/anchor procedure only after coverage, source binding, and protocol
review.  This file does not authorize parameter updates, PPO, promotion, or
qualification, and it does not claim that the saved rows are independent
samples.
