# V4.2 / Observation V2.1 Untouched I1 Qualification Cohort Freeze

## Decision

The preregistered `V2_1_I1_Q` cohort was collected from fresh, historically
disjoint seeds and passed every freeze gate. No actor checkpoint was loaded or
trained during collection.

```text
I1_Q_COHORT_FROZEN = true
I1_FRESH_MULTI_SEED_TRAINING_ALLOWED = true
V4_2_V2_1_I2_DAGGER_PREFLIGHT_ALLOWED = false
PPO_ALLOWED = false
```

## Experiment evidence

- Evidence type: `newly_run`, 2026-09-14.
- Workdir: `~/projects/terminal` on `lab-server`.
- Output directory:
  `~/training_outputs/terminal/v4_2_v2_1_i1_q_cohort_20260914`.
- Candidate seed range: `210001..210999`; the pre-reset historical registry
  rejected already-consumed smoke/diagnostic seeds before reset. The formal
  collection used 27 seeds, `210012..210038`.
- Scenario scheduling: one complete nine-stratum cycle followed by the fixed
  round-robin unmet-deficit schedule, between whole episodes only.
- Row admission: the existing semantic-event sampler with bounded sparse
  persistence. Teacher labels, future model predictions, class counts and
  model outputs never admitted individual rows.
- Training/DAgger/PPO: not run.

## Frozen cohort support

```text
Q_rows = 1124

Q_WAIT_rows = 947
Q_WAIT_lifecycles = 257

Q_RELEASE_rows = 177
Q_RELEASE_lifecycles = 167

proposal_11_lifecycles = 26
proposal_12_lifecycles = 30
proposal_13_lifecycles = 89
proposal_14_lifecycles = 22

clear_outer = 62
middle_interface = 67
post_WAIT = 88
post_conflict = 60

retry_active_WAIT = 99
retry_expired_RELEASE = 167
```

All counts satisfy the frozen protocol without changing a minimum, duplicating
a row, oversampling a class, or using a model prediction. Family memberships
may overlap as preregistered.

The collector processed 6,684 raw UAV decision ticks and retained 1,124
semantic events. Raw Teacher decisions were 6,507 WAIT and 177 RELEASE. Stable
lifecycle flooding was bounded only for secondary topology/continuous-change/
persistence snapshots; Teacher target, retry/eligibility, phase/commit and
interface/resource changes remained mandatory retention events.

## Provenance, replay and safety

```text
Observation dim = 88
schema provenance = PASS (1124/1124)
retry provenance = PASS (1124/1124; mismatch count 0)
Teacher determinism = PASS

ActionTape exact replay = PASS
exact tapes = 9/9
exact replay rate = 1.0

collision = 0
NMAC = 0
hard violation = 0
SafetyInvariantError = 0
uncertified execution = 0

seed independence = PASS
episode independence = PASS
lifecycle independence = PASS
full-state-hash independence = PASS
decision-event uniqueness = PASS
full-state-hash uniqueness = PASS

exact opposite-gate observation-hash conflicts = 0
```

The minimum nearest opposite-gate Observation-V2.1 L2 distance was `0.038643`;
this is a boundary diagnostic, not an exact contradiction. Exact observation
hash conflicts were zero.

## Frozen artifact hashes

```text
Q JSONL SHA256 =
8d1e80bbf7efbf8bdada1814efc3c435daa80dd00564935c2e9dd3ba466dffe5

Q manifest SHA256 =
a24ef05fb78093a81945f28d91bb16907ebf443c127f2158651613ce401c40f2

pre-reset seed registry SHA256 =
83fe604d85eb46a457b0a883b8a4e072687ff3b96426c3b0b4e57cd4b82d0b8e

final seed registry SHA256 =
59891c7dbfff04069ea229196098fb365c6e840c65e3adca48bde409072a815e

collection config SHA256 =
47de473288d2c05b3e86b19143cf22140be61c93428c50003a7144d9b0fb5d9c

qualification protocol SHA256 =
6da3ff8f3c790428c6e16d093a63c4e09c1db44e5ff2676e4abfb95fa3d47d11

collection code SHA256 =
05baad51a65be2d838e0f6d8ce58eb74dc4fbe8f118e4971643b0f115a28a173

reset recipe hash count = 27
```

The final manifest contains all 27 reset-recipe hashes and all nine replay tape
hashes. From this point the Q dataset is:

```text
IMMUTABLE
NEVER_TRAIN
NEVER_MODEL_SELECT
NEVER_THRESHOLD_TUNE
HIDDEN_FROM_TRAINING_CODE
```

## Protocol risks

- This is an internal engineering qualification cohort, not an external
  generalization test.
- Scenario-level coverage targets intentionally shape episode selection, but
  no individual row was selected by Teacher label or model prediction.
- Family flags overlap; family totals must not be summed as disjoint classes.
- The Q cohort may be evaluated exactly once after all three preregistered
  final-epoch actors are trained and hashed.
