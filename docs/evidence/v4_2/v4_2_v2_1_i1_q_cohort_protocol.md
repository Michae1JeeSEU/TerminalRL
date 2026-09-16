# V4.2 / V2.1 Untouched I1 Qualification Cohort Protocol

## Purpose and authorization boundary

`V2_1_I1_Q` is a new, never-trained qualification cohort for the already
frozen I1 Qualification Protocol V1. Collection is allowed only after the
protocol file and its hashes are recorded in the reproducibility manifest.

```text
I1_Q_COHORT_COLLECTION_ALLOWED = true
NEVER_TRAIN = true
NEVER_MODEL_SELECT = true
NEVER_THRESHOLD_TUNE = true
```

The cohort is not collected or evaluated in this preregistration phase.

## Independence and seed registry

The reserved candidate seed range is `210001..210999`. Before collection, the
collector must build a seed registry from formal I0-R, the old frozen cohort,
all closure/powered/support/reachability banks, failed formal datasets, and
DAgger diagnostic data. Any collision rejects the seed before reset.

Every retained state must be disjoint from I0-R training, the old frozen
cohort, and every diagnostic bank by:

```text
seed
episode_id
lifecycle_event_id
full_state_hash
```

Duplicate observation hashes are also audited. An exact observation hash with
opposite Teacher gates is a stop condition, not an admissible cohort row.

## Natural scenario strata

The collector reuses the frozen formal V2.1 Teacher, Observation, action schema,
execution and Safety semantics. It launches complete, independently seeded
episodes from preregistered natural scenario strata:

```text
clear_outer_low_traffic
middle_interface
post_WAIT_release
conflict_recovery
transition_retry
nominal_mixed
resource_pressure
proposal_12_context
proposal_14_context
```

Rows are admitted only by the existing semantic-event sampler. Model
predictions are never computed during collection and cannot admit a row.
After a complete episode, coverage counters may select the next stratum from a
fixed round-robin deficit schedule; the entire episode remains eligible and no
Teacher label is selectively discarded. This is scenario-level stratification,
not row duplication, oversampling, or class weighting.

## Minimum frozen support

Collection succeeds only if all minima are met with independent lifecycles:

| Support | Minimum |
|---|---:|
| Teacher WAIT rows/lifecycles | 300 / 250 |
| Teacher RELEASE rows/lifecycles | 160 / 150 |
| Proposal 11 RELEASE lifecycles | 15 |
| Proposal 12 RELEASE lifecycles | 15 |
| Proposal 13 RELEASE lifecycles | 30 |
| Proposal 14 RELEASE lifecycles | 15 |
| clear_outer RELEASE lifecycles | 30 |
| middle_interface RELEASE lifecycles | 30 |
| post_WAIT RELEASE lifecycles | 30 |
| post_conflict RELEASE lifecycles | 30 |
| retry-active WAIT lifecycles | 60 |
| retry-expired RELEASE lifecycles | 60 |

Family predicates may overlap; counts must report both rows and unique
lifecycles. The expected size is roughly 650–900 retained semantic events. A
hard ceiling of 1,200 rows prevents indefinite collection. Reaching the ceiling
without every minimum gives `I1_Q_COHORT_FROZEN=false`; minima may not be met by
copying rows or relaxing the protocol.

## Row provenance and freeze gates

Each row stores the same reconstructable V2.1 provenance as formal I0-R:
reset recipe and RNG hashes, environment time, complete relevant UAV/airspace/
scheduler/resource state, `safety_retry_after`, Teacher gate/proposal/rationale,
25-proposal audit sidecar, source/seed/episode/lifecycle/decision IDs, config,
state, observation and post-state hashes.

Required pre-freeze gates:

```text
Observation length = 88
retry feature provenance = 100% PASS
Teacher determinism = PASS
ActionTape exact replay = 100%
collision/NMAC/hard/SafetyInvariantError/uncertified = 0
semantic-event identity uniqueness = PASS
seed/episode/lifecycle/state-hash disjointness = PASS
exact observation opposite-gate conflict = 0
all support minima = PASS
```

On success, the JSONL bytes, collection manifest, seed registry, reset recipes,
and collection code/config are hashed. Those hashes are entered in the mainline
reproducibility manifest before any actor training begins. The cohort then
becomes immutable and is hidden from training code until all three fresh
final-epoch checkpoints are complete.

## Evaluation discipline

All three checkpoints are evaluated in one invocation after verifying their
training lineage and fixed seeds `[4202, 4203, 4204]`. Evaluation produces a
single manifest containing per-seed results, mean/std/min/max, the conjunctive
decision, and the predetermined canonical seed-4202 checkpoint hash.

No model prediction may change cohort membership, and no result may change a
threshold, select a seed, or select an epoch.
