# P3 forced-WAIT frozen comparison protocol (2026-09-30)

This document records the bounded comparison that follows the completed
24-state paired diagnostic. The new experiment has 24 complete-scene seed
pairs. It is a frozen-policy technical experiment
protocol; it is not a training run or a qualification protocol.

## Registered scope and observed starting facts

The completed paired diagnostic classified its 24 prior diagnostic states as
follows:

| first-step classification | states |
|---|---:|
| pure `safety_wait` | 14 |
| pure `safety_reject` | 8 |
| `emergency_lz_priority` | 1 |
| two approved proposals with very low q | 1 |

The last state is the state with `C=2.0938614857594729e-14` (selection 21).
All 24 first-step weighted reward differences were approximately `-0.05`.
The completed full-cycle `D` values include mixed signs and positive values;
the first-step result is not used as a full-cycle validity label.

## Frozen comparison

The comparison uses the existing opt-in `pre_action_forced_wait` strategy
against the legacy strategy. The source is the fixed `3fe7` checkpoint followed
by `a82` warmup, with the 88D observation schema, K=3 proposal shortlist, and
`dense24` scenario. The only strategy mode factor is `legacy` versus
`pre_action_forced_wait`.

The comparison does not change learning, PPO, teacher use, held-out inputs,
observations, proposals, rewards, or safety rules. `PPO_ALLOWED=false` remains
in force and formal qualification status is unchanged.

There are 24 complete-scene seed pairs from environment seeds `877101` through
`877106` and keyed policy continuation seeds `9303101` through `9303104`,
giving 24 pairs and 48 arms. Pair order is environment seed ascending, then
keyed policy seed ascending. Both arms use new common-random-number sampling
keyed by `arrival_id + tick + gate/proposal stream`; the key does not use UID.
This is a new paired-sampling protocol and is not a numerical comparison to
the historical exact-tape values.

The fixed pilot pair is environment seed `877101` with keyed policy seed
`9303101`. After the pilot technical checks, the full run reuses that first
pair and evaluates only the remaining 23 complete-scene pairs. A censored
pilot pair does not admit the full run; censored pairs in the full run remain
retained with unknown business metrics. The protocol does not select seeds by
outcome and does not rerun the completed failure cases.

Each scene is allowed to reach a natural terminal state or the 1,800-tick cap;
unknown cap outcomes are retained and reported. Pair and arm records are
written atomically, including errors and censored outcomes.

## Forced-row contract

The forced-WAIT row may use only the existing reasons `safety_retry`,
`interface`, or `SOC<=0`. An urgent positive-SOC actor is not masked, and
`emergency_retry` is not treated as a normal forced-WAIT reason. Time, energy,
service, and automatic-emergency behavior continue to use their existing
paths.

## Readiness and execution order

Before any environment is created, the preflight checks the frozen model/source
hashes, runtime source manifest, neutral configuration, and strategy mode
contract. A small meaningful fixture/test must verify the paired sampling,
forced-row metadata, and serialization fields. The root readiness check is
required before the pilot; the pilot must pass its technical checks and be
uncensored before the remaining 23 complete-scene pairs are started.

The output root is
`/home/user/training_outputs/terminal/p3_followup_20260930/forced_wait_frozen`.
It retains preflight and summary reports, a runtime manifest, atomic per-pair
records, progress, and all anomalies. The registered scope is exactly 24
pairs; no additional matrix is added.

Pilot and full-run admission requires the input, source, neutral configuration,
external identity, ledger, serialization, and hard-safety checks to pass. A
natural `failed_energy` outcome is a retained business terminal outcome; it is
not removed or used to select a new seed. A technical, binding, or hard-safety
failure stops the corresponding phase and leaves its artifacts for review;
Natural business failures and censored outcomes remain recorded. Censored
full-run arms retain unknown full-cycle paired differences; they are not treated
as technical success with complete business evidence. This protocol does not
tune rules or safety floors.

## Recorded metrics and claim boundary

Metrics are reported in this order: safety counters and violations; then
`normal`, `emergency`, `failed`, `diverted`, and `unresolved` outcomes with
their generated/admitted/completed denominators stated; then wait, elapsed
request time, and energy. Raw reward and rejection penalty are separate
fields. Removing a penalty is not reported as a business improvement.

The paired results are development evidence rather than an independent
population sample. They do not by themselves establish a training-degeneration
fix or a performance improvement. No PPO, held-out evaluation, or formal
qualification is performed in this protocol.
