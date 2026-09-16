# V4.2 / Observation V2.1 Fresh Formal I0 Recollection

## Final qualification result

```text
formal_training_rows = 1800
formal_frozen_rows = 300
unique_release_lifecycles = 93 (training) / 13 (frozen)
release_joint_classes = 5
release_speed_values = 2
release_merge_values = 3

retry_feature_provenance = PASS
teacher_determinism = PASS
observation_aliasing_check = PASS
safety = PASS

V4_2_V2_1_I0 = FAIL
V4_2_V2_1_I1_ALLOWED = false
DAgger_ALLOWED = false
PPO_ALLOWED = false
```

This is a newly run formal collection, not a conversion or augmentation of a
historic 87D dataset. The formal artifacts are in
`~/training_outputs/terminal/v4_2_v2_1_formal_i0_20260912`.

## Evidence and positive gates

- Fresh train and frozen seed/episode/lifecycle/state-hash lineages were
  disjoint. The collection retained 1,800 training and 300 frozen rows.
- All 2,100 retained snapshots regenerated the 88D observation and retry
  feature exactly; no missing retry values were filled or inferred.
- Repeated Teacher evaluation was deterministic for every retained state.
- Exact observation-hash / opposite-gate conflicts were zero. The nearest
  opposite-gate distance was 0.7218 (P50), 1.6022 (P95), and 0.02128 (min).
- Safety totals were collision = 0, NMAC = 0, hard = 0,
  SafetyInvariantError = 0, uncertified = 0.

## Failing gates

### RELEASE coverage and frozen support

Training retained 93 RELEASE rows / lifecycles, below the required 100
independent RELEASE lifecycle events. The independent frozen cohort retained
only 13 RELEASE rows/lifecycles, below its preregistered minimum of 60.

The 93 training RELEASE events use five joint proposals, two speed values, and
three merge values; the required diversity was at least eight joint proposals,
four speed values, and all five merge values. Source tags exist, but the
actual Teacher-target coverage did not establish clear RELEASE,
middle/interface RELEASE, post-conflict recovery, or post-WAIT RELEASE.
Consequently semantic-source coverage and proposal-diversity gates fail.

This is a coverage deficiency, not permission to duplicate, rebalance, or
alter labels. The failed formal bank is read-only evidence and must not be
used for I1.

### ActionTape exact replay

Only 16 of 20 sampled tapes replayed exactly. Four dynamic-source tapes reset
to the same initial state and matched requested/executed actions, but then
diverged in observation and post-state hashes on every step. The current
generic replay starts directly from reset while those collection sources apply
their deterministic dynamic-airspace setup between reset and the first logged
low-level step. The replay artifact does not recreate that pre-step setup.

This is a schema/replay qualification failure. It does not demonstrate a
collision, Teacher-label, or low-level requested-action mismatch; nevertheless
it blocks formal lineage promotion until replay restores the recorded
precondition.

## Collection statistics

| Partition | WAIT | RELEASE | Raw decision ticks | Retained duplication factor | Retry-active WAIT |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training | 1,707 | 93 | 2,654 | 1.4744 | 785 |
| Frozen | 287 | 13 | 455 | 1.5167 | 123 |

Artifact hashes:

```text
training_jsonl_sha256 = 776734b9194846465368ae25125d858ac17b75430d32b4bce930d6289d17378b
frozen_jsonl_sha256   = 87578340c00d25c2876038b5f039e3849b655d629108512460918d1de06003e5
formal_manifest_sha256 = 8894444f82e1b42364fa934a46d157ef379ae47b36c35832a748088b9420230b
```

## Protocol risk and permitted next work

The formal conclusion is FAIL, not a partial pass. A future repair needs two
separate, predeclared steps: (1) fix and requalify ActionTape replay on the
existing failed artifact by recording/reconstructing the deterministic
pre-step dynamic-source condition; (2) only with authorization, collect a
new fresh V2.1 I0 lineage using naturally release-rich scenarios to meet all
RELEASE coverage and frozen-support gates. Neither step authorizes I1,
DAgger, PPO, class weighting, oversampling, label changes, or synthetic rows.
