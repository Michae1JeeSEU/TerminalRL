# P3 prefix normal-inner gate-value protocol (2026-10-01)

This document records a frozen, off-policy diagnostic of the first registered
normal-inner event with `arrival_id >= 13` in each of four immutable
BEFORE-prefix tapes.  Selection is mechanical: retain rows whose status is
`NORMAL_ELIGIBLE`, whose phase is `ring_inner`, and whose context arrival ID
is at least 13; select the first `(absolute step, numeric UID)` row.  Neither
`p`, `delta_p`, reward, terminal outcome, nor any value estimate participates
in selection.  The earlier step-0 warm-start events remain recorded in the
source event cohort and are not replaced by this four-state cohort.

The fixed states are:

| cell | key | arrival | SOC |
|---:|---|---:|---:|
| 1 | `307|15` | 15 | 0.7262173226131179 |
| 2 | `321|25` | 15 | 0.6731115455491892 |
| 3 | `303|20` | 15 | 0.6404768921709051 |
| 4 | `365|46` | 15 | 0.47479269296572224 |

The input event artifact is `prefix_gate_events_20261001/full.json` with SHA
`fb89f62339755f7ebcd2bb0f9672438000fe8b754067cd2cdb28bf8407feb3e6`.  The
completed six-scene fresh artifact is
`prefix_fallback_fresh_validation_20261001_v4` with full SHA
`30c28c03d26466eabeab2a35c9fc117424648f1a3ae17be6f8d69b6b92cb2195`; only
its existing cells 1--4 BEFORE-prefix tapes are selected for this mechanism
cohort.
The frozen checkpoints are BEFORE `3fe7dd1e01d032227521f6fc5c8464fe8d00d98fb0e9f0dbd13411123a27d4ab`
and AFTER `d44990dc7d129573d9145f169df1cbc4f91fa32bc5127b1c58d937260c02a24f`;
the warmup/a82 artifact is not injected.

At each state, the original BEFORE-prefix tape is replayed exactly once with
the saved 25-way order provider and public K=3 prefix.  The saved pre-state is
then cloned for four fixed continuation seeds `9303401..9303404`, with
`WAIT + 25 RELEASE` branches (26 branches per seed, 104 per state, 416 across
the four states).  The pilot therefore has 104 branches and the subsequent
full phase adds 312 new branches.  Other first-step actions remain the tape
actions.  The target WAIT
branch uses `proposal=None`; a target RELEASE branch uses proposal `j` and a
K=3 public prefix beginning with `j`, while its private controller order is
`(j, stable-logit-ranked remaining proposals)`.  Subsequent BEFORE-policy
requests use one details forward per step, arrival-ID/absolute-tick/stream
keyed draws, and a K=3 public prefix with the corresponding 25-way provider.
The native prefix controller and safety, energy, service, emergency, retry,
and capacity rules remain unchanged.

The immediate coverage `C` is recorded separately from full-cycle value.  For
each seed, `D` is the q-weighted complete RELEASE return over all 25 proposals
minus the complete WAIT return, including the intervened first reward.  The
four-seed mean is known only when every WAIT and RELEASE branch is natural and
technically safe.  `local_m = delta_p * D` is state-local and is never summed
as a global gain.  Censored, failed, error, and hard-safety branches are
retained; incomplete `D` and `local_m` remain unknown.  No survivor weighting,
resampling, held-out data, teacher, optimizer, PPO, or training is used.

Preflight is static and must create no environment and perform no model
forward.  It binds the protocol, runner, source files, checkpoint hashes,
event/full input hashes, four selected keys, config snapshot, and output
schema.  Pilot runs only cell 1; full reuses the immutable pilot artifact and
runs cells 2--4 once.  Model, caller-RNG, environment, exact replay, source
q/logit, controller-alias, and hard-safety mismatches stop the phase while
preserving partial artifacts.  This diagnostic does not establish a global
policy value, a training cause, or a qualification result.
