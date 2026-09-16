# V4.2 / Observation V2.1 Retry-Closure Protocol Diagnosis

## Decision

```text
RETRY_FEATURE_SEMANTIC_CLOSURE = PASS
RETRY_MATCHED_PAIR_CONSTRUCTION = PASS
RETRY_LEARNED_PAIR_PROBE = FAIL
gate_A_false_negative_count = 2
gate_A_false_negative_common_missing_state = false
ROOT_CAUSE = DIRECT_PREDICATE_AND_LEARNABILITY_GATE_CONFLATION
observation_revision_required = false
closure_protocol_revision_required = true
independent_release_coverage_required = true
powered_confirmation_allowed = true
FORMAL_V2_1_I0_RECOLLECTION_ALLOWED = false
I1_ALLOWED = false
DAgger_ALLOWED = false
PPO_ALLOWED = false
```

## Experiment evidence

- Status / evidence type: `newly_run`.
- Input bank SHA-256: `0af25921a413434529fb284cf3a40b6050d5ca22ea4834bc0ca4ac954b34baa6`.
- Output manifest: `~/training_outputs/terminal/v4_2_v2_1_retry_closure_diagnostic_20260911/v4_2_v2_1_retry_closure_diagnostic.manifest.json`.
- Output manifest SHA-256: `d6a25d7ebd438af8f00cdad26374d56926be653d9f7c33c9504d976a67d23ca2`.

No formal collection, policy training, Teacher/Safety/observation/action change, class weighting, oversampling, row duplication, certificate exposure, or threshold change occurred.

## A. Direct retry semantics

For all 600 full-state records, the audit recomputed `state_time < safety_retry_after`, `obs[87] > 0`, and `clip(max(0, safety_retry_after - state_time) / 30, 0, 1)`.

| Check | Result |
|---|---:|
| runtime retry-active count | 265 |
| observation retry-active count | 265 |
| predicate disagreements | 0 |
| disagreement rate | 0.0 |
| normalized-value mismatches at `1e-6` | 0 |

The direct rule `obs[87] > 0` has accuracy, blocked recall, and nonblocked recall all equal to `1.0000`. There is no retry provenance or runtime bug.

## B. Retry-pair construction

The formal construction was replayed exactly: 25 train and 6 dev pairs. Every pair has WAIT-side Teacher WAIT / retry active / `obs[87] > 0`, and RELEASE-side Teacher RELEASE / retry inactive / `obs[87] = 0`. Each stored pair includes state IDs/hashes, gates, retry predicates, feature values, phase/ring, eligibility, commit, interface, and resource context. `RETRY_MATCHED_PAIR_CONSTRUCTION = PASS`.

The original complete-88D learned pair probe remains FAIL: accuracy `0.6667`, macro-F1 `0.6571`, WAIT recall `0.5000`, RELEASE recall `0.8333`, and AUC `0.6944` on 50/12 train/dev samples. It predicts a Teacher WAIT/RELEASE distinction from a tiny matched subset, not the direct retry predicate. It cannot invalidate exact feature semantic closure.

## C. Gate-A false negatives

There are two dev Teacher-RELEASE false negatives among six dev RELEASE states. The middle-ring case at seed 14030 / step 0 has probability `0.4761`, selected proposal `12`, and nearest train RELEASE distance `2.5512`. The outer-ring case at seed 14060 / step 9 has probability `0.0124`, selected proposal `21`, and distance `11.1563`.

Both are retry inactive, uncommitted, interface-open, resource-valid, and have a certified selected proposal. They do not share a missing state-only Teacher predicate. The second is comparatively distant from train RELEASE support; with only six dev RELEASE states this is finite positive-support/boundary evidence, not authorization to modify the observation or frozen threshold.

## Required interpretation

The direct semantic PASS plus correct pair definition means retry semantics are closed. The learned pair probe remains a model-capacity/finite-sample diagnostic and must not be conflated with state representation. The historic closure FAIL remains read-only and is not retroactively changed.
