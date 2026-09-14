# V4.2 / Observation V2.1 Fresh Formal I0-R Qualification

## Outcome

The fresh formal recollection passed every preregistered gate. The resulting
training and frozen rows were generated from new simulator rollouts only; no
historical V2/V2.1 I0, collector-preflight, closure, reachability, support-bank,
or DAgger row was used.

```text
V4_2_V2_1_I0_R = PASS
V4_2_V2_1_I1_ALLOWED = true
DAgger_ALLOWED = false
PPO_ALLOWED = false
```
Evidence type: `newly_run`.

Formal server artifact:

```text
~/training_outputs/terminal/v4_2_v2_1_formal_i0_r_20260913
```

## Frozen schemas and protocol

```text
observation_schema = phaseA_low_obs_v2_1_retry_state
observation_dim = 88
retry_feature_index = 87
action_schema = low_action_schema_v4_2_hierarchical
required_selected_support = {11,12,13,14}
```

Teacher, Teacher ranking, ExecutionScheduler, TrajectoryGenerator,
SafetyFilter, PredictiveShield, the action catalogue, Observation V2.1,
release-family predicates, and transition-ledger semantics were unchanged.
Rows were retained at semantic events. One complete all-strata cycle preserved
WAIT/conflict/resource coverage; subsequent whole release-rich episodes closed
the preregistered support gates. There was no row- or label-conditioned
admission, duplication, class weighting, oversampling, or synthetic labeling.

## Dataset lineage and counts

| Item | Training | Frozen cohort |
|---|---:|---:|
| Retained rows | 2020 | 643 |
| WAIT rows | 1551 | 533 |
| RELEASE rows | 469 | 110 |
| Unique seeds / episodes | 74 / 74 | 17 / 17 |
| Unique UAV lifecycles | 532 | 133 |
| Unique decision events | 2020 | 643 |
| Independent RELEASE lifecycles | 467 | 108 |
| Raw decision ticks | 7104 | 2316 |
| Temporal duplication factor | 3.5168 | 3.6019 |

Training and frozen seeds, episode IDs, lifecycle IDs, state hashes, and
decision-event IDs are disjoint. Frozen cohort independence is `PASS`.

Dataset hashes:

```text
training SHA256 = 6bcbc1782a171f909282126c4f6d27dd93355b09513e2827093228af4fe05f05
frozen SHA256   = 229e9d0c087eac1e238c90d857e74dc71525d4eac1b343ed88f33fa7e0695aa2
manifest SHA256 = 51868f7892395d7d5a369d956f6c374ba536d3c8023ec5d0f42a2d9f208f8a31
```

## Stable proposal support

Observed Teacher-selected support was exactly `[11, 12, 13, 14]` in both
partitions.

| Proposal | Training independent lifecycles | Frozen independent lifecycles | Minimum |
|---:|---:|---:|---:|
| 11 | 51 | 13 | 10 |
| 12 | 97 | 23 | 10 |
| 13 | 280 | 62 | 10 |
| 14 | 39 | 10 | 10 |

The prospective support gate therefore passed for both partitions. Speed and
merge marginals remain diagnostics only: selected speed support was `[0.0]`;
selected merge support was `[-1.0, -0.5, 0.0, 0.5]`.

## Predicate-defined RELEASE families

Family membership was computed only from current state plus the same-UAV
transition ledger. Collection source names were not used as labels. Families
may overlap.

| Family | Training independent lifecycles | Frozen independent lifecycles | Training minimum |
|---|---:|---:|---:|
| clear_outer | 195 | 45 | 15 |
| middle_interface | 185 | 43 | 15 |
| post_WAIT | 222 | 56 | 10 |
| post_conflict | 12 | 12 | 10 |

Both training and frozen family gates passed.

## WAIT coverage

Training retained 307 retry-active WAIT, 435 committed WAIT, 104
interface-frozen WAIT, 34 resource WAIT, 104 conflict WAIT, 671 ordinary
scheduling WAIT, and 264 transition WAIT rows. The separately collected frozen
cohort retained 82 retry-active WAIT, 110 committed WAIT, 100
interface-frozen WAIT, 14 resource WAIT, 100 conflict WAIT, 227 ordinary
scheduling WAIT, and 259 transition WAIT rows. No explicit headway-conflict row
was selected in this fresh lineage (`0` in both partitions); interface,
transition, geometry/conflict, resource, committed, retry, and ordinary WAIT
contexts remained represented.

## Provenance, determinism, aliasing, replay, and safety

All 2663 rows passed the 88D schema and retry-state provenance audit:

```text
retry_mismatch_count = 0
schema_mismatch_count = 0
retry_feature_provenance = PASS
schema_provenance = PASS
```

For every retained row, the repeated Teacher gate, RELEASE proposal when
applicable, and decoded joint action were identical. Teacher determinism is
`PASS`.

No observation hash had opposite Teacher gate labels. The nearest
opposite-gate L2 distance was min `0.0386429`, P50 `0.870665`, and P95
`2.001329`; observation aliasing is `PASS`.

ActionTape covered all nine formal strata in both training and frozen
partitions. All 18 tapes exactly reproduced reset/config, reset recipe,
observation, requested category, Teacher label, executed action, and post-state:

```text
exact_tapes = 18
total_tapes = 18
exact_replay_rate = 1.0
ActionTape_exact_replay = PASS
```

Safety counts across both partitions were:

```text
collision = 0
NMAC = 0
hard separation violation = 0
SafetyInvariantError = 0
uncertified execution = 0
safety = PASS
```

## Qualification decision

Every formal gate passed: fresh lineage, schema/retry provenance, Teacher
determinism, observation aliasing, exact replay, safety, training RELEASE
lifecycle/family/stable-proposal coverage, and frozen independence plus RELEASE,
family, and stable-proposal coverage.

This authorizes the next stage to run a **fresh V4.2 hierarchical I1 BC** using
this I0-R lineage. It does not authorize DAgger or PPO, and no I1 training was
run in this qualification.
