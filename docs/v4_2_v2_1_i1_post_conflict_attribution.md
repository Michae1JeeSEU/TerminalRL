# V4.2 / V2.1 I1 Post-Conflict RELEASE Error Attribution

## Decision

This was a non-mutating evaluation of the existing formal I0-R training set,
the already-consumed 643-row frozen cohort, and the diagnostic-only seed-4201
checkpoint. No row was changed and no training occurred.

```text
POST_CONFLICT_ROOT_CAUSE = TRAINING_SUPPORT_SPARSE
POST_CONFLICT_WEAKNESS = EXPECTED_I1_BOOTSTRAP_LIMITATION
observation_revision_required = false
recommended_role = I2_SAFE_DAGGER_PRIORITY_DIAGNOSTIC_FAMILY
```

The result does not retrospectively qualify the checkpoint.

## Experiment evidence

- Evidence type: `newly_run`, 2026-09-14.
- Aggregation level: decision state, with lifecycle/episode/seed support also
  reported.
- Train input: 2,020 immutable formal I0-R rows, SHA256
  `6bcbc1782a171f909282126c4f6d27dd93355b09513e2827093228af4fe05f05`.
- Frozen input: 643 already-consumed diagnostic-reference rows, SHA256
  `229e9d0c087eac1e238c90d857e74dc71525d4eac1b343ed88f33fa7e0695aa2`.
- Checkpoint: diagnostic-only seed 4201, SHA256
  `3b4ddd305dd873e9f892774d4d86fa321230fd7d6c3103bb0583ce5c6c1fb725`.
- Output artifact:
  `~/training_outputs/terminal/v4_2_v2_1_i1_post_conflict_attribution_20260914/v4_2_v2_1_i1_post_conflict_attribution.json`.
- Artifact SHA256:
  `84552e09b8773b5b0436d1d86ca111804bb2689645dfc84be2b8751c6c330be4`.
- Mutations: retraining=false, relabel=false, threshold tuning=false,
  Observation/Teacher/Safety/architecture changes=false.

## Frozen-state results

Exactly 12 Teacher-RELEASE states satisfy the predicate-defined
`post_conflict` family. Four were predicted RELEASE and eight were predicted
WAIT.

| Group | N | mean P(RELEASE) | median P(RELEASE) | mean logit margin | nearest train RELEASE L2 | nearest train WAIT L2 | nearest train post-conflict L2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| correct RELEASE | 4 | 0.6527 | 0.6666 | 0.6675 | 1.0349 | 1.1911 | 1.4071 |
| false WAIT | 8 | 0.1610 | 0.1302 | -2.0496 | 1.2753 | 1.0271 | 1.6366 |

Only the weakest correct state was at the decision boundary
(`P(RELEASE)=0.5035`, margin `0.0138`). None of the eight false-WAIT states was
within a logit margin of 0.5; their least negative margin was `-0.6760`.
Consequently, `BOUNDARY_MARGIN_ERROR` is not the dominant attribution.

## Per-state audit

Proposal IDs are canonical joint IDs. The full JSON artifact records the
complete requested context for every row: retry, commit/plan/reservations,
interface and headway state, final/LZ resources, nearest-conflict topology
slot (`path gap`, distance, relative speed, closing rate, dCPA and tCPA),
previous WAIT rationale, elapsed conflict-clearance time, same-UAV lifecycle
history, and identifiers for each nearest training neighbor.

| Frozen decision event | Result | P(R) | Margin | Teacher / predicted proposal | Phase | Cleared (s) | nearest R / W / post-conflict L2 |
|---|---|---:|---:|---|---|---:|---|
| seed-111005 uid-0 decision-5 | WAIT | .0522 | -2.8983 | 11 / 14 | outer | 7 | 1.4142 / 1.2096 / 1.4142 |
| seed-111005 uid-10 decision-4 | RELEASE | .7407 | 1.0495 | 13 / 13 | inner | 9 | .8000 / .7484 / .8000 |
| seed-111005 uid-11 decision-4 | RELEASE | .5926 | .3748 | 13 / 13 | inner | 9 | 1.4094 / 2.0062 / 2.0000 |
| seed-111005 uid-3 decision-4 | RELEASE | .5035 | .0138 | 12 / 12 | outer | 9 | .5159 / .4096 / 1.4142 |
| seed-111005 uid-6 decision-5 | WAIT | .0442 | -3.0743 | 14 / 13 | middle | 9 | .9691 / 1.2103 / 2.0000 |
| seed-111005 uid-8 decision-5 | WAIT | .2961 | -.8661 | 13 / 12 | middle | 9 | .9691 / .8259 / 2.0000 |
| seed-111006 uid-0 decision-5 | WAIT | .0417 | -3.1345 | 11 / 14 | outer | 7 | 1.4542 / .6168 / 1.8111 |
| seed-111006 uid-10 decision-4 | RELEASE | .7742 | 1.2319 | 13 / 13 | inner | 9 | 1.4142 / 1.6000 / 1.4142 |
| seed-111006 uid-11 decision-4 | WAIT | .3372 | -.6760 | 13 / 13 | inner | 9 | 1.4094 / 1.4230 / 1.4142 |
| seed-111006 uid-3 decision-4 | WAIT | .2780 | -.9545 | 12 / 12 | outer | 9 | 1.3729 / .8988 / 1.6248 |
| seed-111006 uid-6 decision-5 | WAIT | .0306 | -3.4568 | 14 / 13 | middle | 9 | 1.4142 / 1.2066 / 1.4142 |
| seed-111006 uid-8 decision-5 | WAIT | .2081 | -1.3362 | 13 / 12 | middle | 9 | 1.1997 / .8259 / 1.4142 |

All 12 states had retry remaining `0`, no committed plan, no affected/frozen
interface, no active airspace transition, free final capacity and six free
landing zones. Their previous WAIT rationale was `geometry_conflict`; clearance
elapsed time was 7 or 9 seconds. Nearest-conflict tCPA and closing-rate fields
were zero for both result groups. Normalized dCPA distributions overlapped:
correct mean `0.7428`, false-WAIT mean `0.7685`. These fields do not reveal a
new systematic missing state variable in this bank.

Proposal-head correctness was 4/4 for correct-gate states and 2/8 for
false-WAIT states. This association is compatible with weak shared support,
but it is not causal evidence that `PROPOSAL_COUPLING` is the primary defect.

## Formal training support

```text
post_conflict rows = 12
post_conflict independent lifecycles = 12
post_conflict RELEASE rows/lifecycles = 12 / 12
post_conflict WAIT rows = 0
independent seeds = 2
independent episodes = 2
proposal distribution = {11: 2, 12: 2, 13: 6, 14: 2}
phase/ring distribution = {outer: 4, middle: 4, inner: 4}
```

The nominal proposal and ring categories appear, but they are supplied by only
two episode-level traffic configurations and contain no within-family WAIT
context. The false-WAIT group is closer on average to general training WAIT
than to training RELEASE, and is farther from the tiny post-conflict support
than the correct group. This is sufficient to classify support as sparse; it
is not evidence for row duplication, weighting, or a Teacher-label change.

## Aliasing and protocol risks

Exact Observation-V2.1 hashes with opposite Teacher gate labels across formal
train plus frozen data: `0`. No new Teacher-decisive state-only missing variable
was found. Observation closure therefore remains closed.

Risks:

- The 12 frozen states arise from only two episodes, so family recall is an
  unstable boundary diagnostic rather than a standalone hard qualification
  statistic.
- Nearest-neighbor distances describe empirical support; they do not prove a
  causal mechanism inside the network.
- The frozen cohort and seed-4201 checkpoint were already consumed. This audit
  is diagnostic reference only and cannot support retrospective promotion.

Post-conflict must be recorded as a priority disagreement family in a future
Safe DAgger stage, if and only if a future fresh I1 passes the independently
preregistered qualification protocol.
