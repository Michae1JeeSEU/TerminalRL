# V4.2 DAgger D1 Conflict-Source Attribution

## Decision

```text
ROOT_CAUSE = SPECIFIC_D1_STATE_FAMILY_CONFLICT
dominant_conflicting_families = [transition_WAIT]
conflict_is_localized = true
development_split_ready = true

ROUND1_RERUN_ALLOWED = false
ROUND2_ALLOWED = false
PPO_ALLOWED = false
```

The identified family is not a license to delete, reweight, or otherwise
alter those events.  It identifies the state family that a future,
independently approved DAgger protocol must diagnose before any rerun.

## Experiment evidence

- Evidence type: `newly_run`.
- Read-only inputs: formal I1 checkpoint, formal failed R1 checkpoint, formal
  D1 (SHA-256 `d94c13f1699dfddda2624537e3ac83edcd5b4878518bfb286efd5a7bad680213`),
  frozen cohort, and Round-1 ActionTapes.
- Final output manifest:
  `~/training_outputs/terminal/v4_2_dagger_d1_conflict_source_attribution_20260911_rerun/v4_2_dagger_d1_conflict_source_attribution.manifest.json`,
  SHA-256 `d76f0a128a16d45404bbdbe3eca6c24cb959b8b59251521af28f83fee1896637`.
- Each diagnostic condition starts at the same I1 checkpoint and uses exactly
  10 CE-only epochs, batch size 64, learning rate `3e-4`, and three retraining
  seeds (5101–5103).  No checkpoint is promoted.
- D1, Teacher targets, Observation, loss weighting, reward, Safety, and all
  qualification thresholds were unchanged.  The only new files are derived
  views, metrics, and manifests.

## Independent DAgger development split

The immutable development view holds out the independently collected Round-1
seeds 9812–9814 and their lifecycle events.  The corresponding diagnostic
training view uses the original 1,500 parent rows plus 408 new events from
the remaining seeds; the original 2,002-row D1 is untouched.

| View | Rows | Seeds | SHA-256 |
|---|---:|---|---|
| Development split | 94 | 9812–9814 | `7e3a31a4a0cc2c0fbe15a28f9eaf0fb5ecdf47b99d1fa2ec5137a1b60177a4d0` |
| Attribution training view | 1,908 | parent + remaining R1 seeds | `4ff33717ca3e8e460e65d74b10813838d4089e290828c766b7dbe0100176bdb5` |

The split is seed/lifecycle disjoint.  It contains 66 WAIT and 28 RELEASE
states, including clear release, conflict/recovery, post-WAIT release, and
interface/headway contexts.  Its family counts are 19 clear outer RELEASE, 9
middle/interface RELEASE, 8 persistent middle WAIT, 22 other persistent WAIT,
and 36 transition WAIT.  It is the only development view to use for future
aggregation or epoch protocol decisions; the frozen cohort remains a
post-hoc regression qualification gate.

## R1 event taxonomy

All 502 new events have raw source tag `student_closed_loop_dagger_round1`.
Their attribution family is derived from immutable Teacher target, phase, and
semantic event reason; reservation/interface/recovery columns describe the
recorded global pre-state, not a new label.

| Family | Count | Teacher / phase | Persistence and state context |
|---|---:|---|---|
| `clear_outer_RELEASE` | 101 | RELEASE; outer 101 | 64 target-change, 23 bounded-persistence; 85 with reservations, 87 with recorded interface/headway context |
| `middle_or_interface_RELEASE` | 51 | RELEASE; middle 37, inner 14 | 33 target-change; 37 with reservations/interface context; 37 recovery/conflict records |
| `persistent_middle_WAIT` | 50 | WAIT; middle 50 | all bounded-persistence; all reservation/interface/recovery context |
| `persistent_other_WAIT` | 136 | WAIT; outer 84, final 26, m2i 24, o2m 2 | all bounded-persistence; all reservation/interface/recovery context |
| `transition_WAIT` | 164 | WAIT; middle 70, outer 65, final 14, m2i 14, o2m 1 | 113 event-start, 37 Teacher-target-change, 14 meaningful-observation-change; 80 reservation/interface/recovery and 84 clear pre-states |

## Six frozen RELEASE flips and D1 neighborhoods

For the six states where I1 predicted correct RELEASE and R1 predicted WAIT,
the top-10 nearest new D1 neighbors distribute as follows.  All neighbors are
from a different seed/lifecycle from the frozen state.

| Neighbor family | Top-10 neighbor count | Mean / min distance | Teacher target | Mean gate margin I1 → R1 | Same phase fraction |
|---|---:|---:|---|---:|---:|
| `transition_WAIT` | 38 | .1349 / .0057 | WAIT | .196 → -.479 | 1.00 |
| `persistent_middle_WAIT` | 14 | .1256 / .0421 | WAIT | -.626 → -1.585 | 1.00 |
| `middle_or_interface_RELEASE` | 8 | .1221 / .1074 | RELEASE | .358 → -.739 | 1.00 |
| `clear_outer_RELEASE` | 0 | — | — | — | — |
| `persistent_other_WAIT` | 0 | — | — | — | — |

The closest transition-WAIT neighbor is only `.0057` normalized observation
distance from a flipped clear-release state.  Its positive I1 gate margin
becomes negative after R1 training, matching the observed RELEASE-to-WAIT
boundary drift.

## Add/remove diagnostic effects

Values are means over three seeds.  “Add” compares I1 plus the named family
against parent-only training.  “Remove” compares full attribution training
minus that family against full attribution training.  Frozen values are
reported only as post-hoc regression evidence, never used to select a model.

| Family | Add: Δ dev RELEASE / WAIT / F1 / D1 disagreement | Remove: Δ dev RELEASE / WAIT / F1 / D1 disagreement | Remove: frozen RELEASE Δ | Attribution result |
|---|---|---|---:|---|
| `clear_outer_RELEASE` | +.262 / -.152 / +.060 / +.045 | -.024 / +.040 / +.005 / -.044 | +.079 | not harmful on inclusion |
| `middle_or_interface_RELEASE` | +.190 / -.298 / -.050 / +.009 | -.119 / +.061 / -.059 / -.012 | +.009 | no removal recovery |
| `persistent_middle_WAIT` | -.095 / -.025 / -.084 / +.006 | +.012 / -.020 / -.002 / -.002 | +.065 | inclusion harms, but recovery is not consistent |
| `persistent_other_WAIT` | -.095 / -.061 / -.099 / -.004 | -.036 / +.025 / -.011 / +.003 | +.014 | inclusion harms, but no removal recovery |
| `transition_WAIT` | -.095 / +.045 / -.054 / +.002 | +.155 / -.258 / -.049 / -.003 | +.171 | **localized conflict: 3/3 harm; 3/3 recovery** |

`transition_WAIT` is the only predeclared localized attribution match:

```text
include transition_WAIT
  -> dev RELEASE recall falls for all 3 seeds

remove transition_WAIT
  -> frozen RELEASE recall recovers for all 3 seeds
  -> mean recovery is +0.1713 versus full attribution training
```

The aggregate full view has competing effects—RELEASE-state families improve
some development RELEASE metrics while several WAIT families hurt them—so
the correct conclusion is localized conflict, not a generic “remove WAIT”
claim.  The substantial dev WAIT/macro-F1 trade-off when transition WAIT is
removed is further evidence against any direct filtering action.

## Recommended protocol change

Freeze the development split and pre-register its use for future DAgger
aggregation/epoch decisions.  Keep the frozen cohort exclusively for the
regression qualification gate.  Before a future authorized Round-1 attempt,
investigate why transition events at event starts, Teacher target changes, and
meaningful observation changes occupy clear-release decision neighborhoods;
do not apply class weighting, oversampling, undersampling, relabeling, or a
threshold change.

## Protocol risks

- This is an internal 94-row development split and three-seed short diagnostic,
  not an independent generalization study.
- The frozen cohort was inspected post-hoc for attribution verification; it
  must not be used to choose a future checkpoint.
- Nearest-neighbor distance gives a local observational association; it is not
  a causal feature attribution proof.
