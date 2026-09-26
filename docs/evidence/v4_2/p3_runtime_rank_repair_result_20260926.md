# P3 runtime rank repair result (2026-09-26)

This file records the completed runtime-label collection and bounded fit. It
does not replace the frozen protocol file
`docs/evidence/v4_2/p3_runtime_rank_repair_20260926.md`.

## Runtime collection

- Source checkpoint SHA256:
  `03c0acb0d561219cbe1dc8dac88cd4df696ade4e4691ff0d4db1d4a72ed5e3ba`.
- Collection status: `COMPLETED`; `runtime_labels=true`; `training=false`;
  `partial=false`.
- Collection size: 164 rows, with 82 train rows and 82 validation rows.
- Runtime branch evaluations: 51 per row, 8,364 total.
- Six records completed with `replay_exact=true`, `outcomes_match=true`,
  `rng_unchanged=true`, `model_weights_unchanged=true`, empty error lists, and
  zero nonzero safety-counter deltas in every record.
- Runtime q-error summary: count 164, maximum
  `9.371259834800583e-08`, mean `6.1514805471674525e-09`, p50 `0.0`, and p95
  `4.155851651699294e-08`.

The four rows whose runtime `certified_proposals` differ from the catalogue
value are identified by `(seed, step)` as follows:

| seed | step | uid | arrival_id |
|---:|---:|---:|---:|
| 873103 | 0 | 0 | 0 |
| 873103 | 587 | 309 | 21 |
| 873104 | 255 | 2 | 2 |
| 873106 | 0 | 0 | 0 |

The corresponding full row values are in the synced collection
`rows.json`.

## Bounded fit

- Source checkpoint SHA256:
  `03c0acb0d561219cbe1dc8dac88cd4df696ade4e4691ff0d4db1d4a72ed5e3ba`.
- Result checkpoint SHA256:
  `9045e913ec5ac35e7a76829c7790e5ab313f8406e60e61ff4a23df560eb6e8dc`.
- Fit status: `COMPLETED`; 86 accepted steps, 0 rejected steps; stopped on
  `train_miss_count_zero`.
- Train metrics, before to after: top-3 misses `2 -> 0`; qK3 mean
  `0.4999999952115447 -> 0.5243902387446742`; KL mean
  `0.0 -> 1.7181198982027893e-05`.
- Validation metrics, before to after: top-3 misses `0 -> 0`; qK3 mean
  `0.5121951170550024 -> 0.5121951174942141`; KL mean
  `0.0 -> 0.0025956418079999756`; KL maximum after fit
  `0.07967605193476601`.
- The fit report records `gate_logits_bitwise_unchanged=true`,
  `all_nonproposal_parameters_frozen=true`,
  `nonproposal_state_bitwise_unchanged=true`, and exact checkpoint round trips
  for all state, gate, model, and optimizer state.

The fit reused development seed `890401`; this development-sample reuse is
not an independent qualification. The fit report flags `P3_QUALIFIED=false`,
`P4_ALLOWED=false`, and `PPO_ALLOWED=false`.

## Sparse-target fit

- Fit report: `training_outputs/terminal/p3_runtime_rank_repair_20260926/fit_sparse_targets/fit_report.json`.
- Fit mode: `sparse_target_rows`, target output rows `[9, 20]`; source
  checkpoint SHA256 remains
  `03c0acb0d561219cbe1dc8dac88cd4df696ade4e4691ff0d4db1d4a72ed5e3ba`.
- Sparse result checkpoint SHA256:
  `a6da1bbbb170cbe6e437ce24d68399f70104e1ba6ea0077b421824aff91de0df`.
- Fit status: `COMPLETED`; 86 accepted steps, 0 rejected steps; stopped on
  `train_miss_count_zero`.
- The model reports 2,425 parameters with `requires_grad`, with 194 effective
  update parameters from the sparse mask. All non-proposal parameters were
  frozen; gate logits, non-target proposal rows, and non-proposal state were
  bitwise unchanged. All-state, gate, model, and optimizer checkpoint
  round-trips were exact.
- Train metrics, before to after: top-3 misses `2 -> 0`; qK3 mean
  `0.4999999952115447 -> 0.5243902384585634`; KL mean
  `0.0 -> 9.967305567987998e-05`; KL maximum after fit
  `0.0015661225664284676`.
- Validation metrics, before to after: top-3 misses `0 -> 0`; qK3 mean
  `0.5121951170550024 -> 0.5121951173888689`; KL mean
  `0.0 -> 0.0007074511187254912`; KL maximum after fit
  `0.023059984122966257`.
- This fit also reuses development seed `890401` and is not an independent
  qualification. The full sparse-target evaluation is recorded below.

## V1 evaluation

- Evaluation manifest: `training_outputs/terminal/p3_runtime_rank_repair_20260926/evaluation/manifest.json`.
- Rank-delta artifact: `training_outputs/terminal/p3_runtime_rank_repair_20260926/rank_delta.json`.
- Evaluation manifest status: `COMPLETED`; records `6/6`; requests `84/84`;
  all 84 requests completed; failed-energy `0`; other failures `0`; unresolved
  `0`; safety-nonzero `false` for every record.
- The evaluation was development-only, consumed no held-out data, called no
  training/update or teacher path, and retained exact fit restore and frozen
  source-state checks.

| split | requests | source normal | source emergency | V1 normal | V1 emergency | source low-SOC wait | V1 low-SOC wait | source priority wait | V1 priority wait |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 84 | 76 | 8 | 73 | 11 | 89.77926924393867 | 85.30758030026561 | 86.66905761611285 | 94.51305448265538 |
| train | 42 | 39 | 3 | 39 | 3 | 62.44447310923468 | 55.815372985243094 | 70.69670994786392 | 74.03961724345051 |
| validation | 42 | 37 | 5 | 34 | 8 | 115.01138875289621 | 112.53115628336333 | 102.95458857197451 | 115.38793166772703 |

The old evaluation manifest records `coverage_improved=false`; this is a
nested-schema reporting bug. The code has since been corrected, while the old
manifest is retained unchanged. The train qK3 metric did improve as recorded
above, but the V1 closed-loop performance rejection is unchanged:
`no_regression_strict.passed=false`, `checkpoint_promotable=false`,
`P3_QUALIFIED=false`, and `P4_ALLOWED=false`. V1 is not accepted, the
coverage repair did not remove the closed-loop degradation, and the source
checkpoint remains the reference.

## Sparse-target evaluation

- Evaluation manifest:
  `training_outputs/terminal/p3_runtime_rank_repair_20260926/evaluation_sparse_targets/manifest.json`.
- Manifest status: `COMPLETED`; records `6/6`; requests `84/84`; failed-energy
  `0`; other failures `0`; unresolved `0`; every record had
  `replay_exact=true`, `all_requests_observed=true`, and
  `safety_nonzero=false`.
- Source checkpoint SHA256:
  `03c0acb0d561219cbe1dc8dac88cd4df696ade4e4691ff0d4db1d4a72ed5e3ba`.
- Sparse candidate checkpoint SHA256:
  `a6da1bbbb170cbe6e437ce24d68399f70104e1ba6ea0077b421824aff91de0df`.
- Fit binding retained `P3_RUNTIME_RANK_REPAIR_V1`, proposal-head-only scope,
  exact restore, empty source/candidate optimizer state, and unchanged
  candidate weights. Aggregate safety, failed-energy, other-failure, and
  unresolved counters were all zero.

| split | requests | source normal/emergency | sparse normal/emergency | source priority wait | sparse priority wait | source low-SOC wait | sparse low-SOC wait |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 84 | 76 / 8 | 74 / 10 | 86.66905761611285 | 83.43568426066302 | 89.77926924393867 | 90.951938078612 |
| train | 42 | 39 / 3 | 39 / 3 | 70.69670994786392 | 70.02610829700959 | 62.44447310923468 | 58.69853206100069 |
| validation | 42 | 37 / 5 | 35 / 7 | 102.95458857197451 | 97.10819308634886 | 115.01138875289621 | 120.72431286409936 |

The sparse report records `coverage_improved=true` and
`closed_loop_improvement=true`, while `development_screen=false`. Strict
no-regression also remained false for both validation and overall: normal
counts fell from `37 -> 35` and `76 -> 74`, respectively. The safety and
wait gates were fully checked, but the normal-count gate failed in both
screened splits. The primary decision is to reject adoption of this sparse
candidate; it is not promoted and does not change the V1 rejection. A single
trajectory-anchor-constrained evaluation is recorded below.

## Anchored fit and evaluation

- Fit report:
  `training_outputs/terminal/p3_runtime_rank_repair_20260926/fit_anchored/fit_report.json`.
- Fit mode: `sparse_target_rows_anchored`, target output rows `[9, 20]`;
  checkpoint SHA256
  `f6bdf401920c42991b59bba63713c691727445107fcac1e7c7059b8870f8590a`.
- Fit status: `COMPLETED`; 106 accepted steps, 0 rejected steps; stopped on
  `train_miss_count_zero`. Effective update parameters: `194`.
- Train metrics, before to after: top-3 misses `2 -> 0`; qK3 mean
  `0.4999999952115447 -> 0.5243902407608294`; KL mean
  `0.0 -> 0.0002602320385417421`; KL maximum after fit
  `0.005187117164266531`.
- Validation metrics, before to after: top-3 misses `0 -> 0`; qK3 mean
  `0.5121951170550024 -> 0.5121951164955306`; KL mean
  `0.0 -> 0.00037035959045462`; KL maximum after fit
  `0.014602860253320318`.
- Gate logits, non-target proposal rows, and non-proposal state were bitwise
  unchanged; all-state, gate, model, and optimizer checkpoint round-trips were
  exact. Anchor MSE after fit was `2.8555426597595215`, with `331` top-3 order
  changes and `244` top-3 set changes across the `11589` anchor rows.
- Evaluation manifest:
  `training_outputs/terminal/p3_runtime_rank_repair_20260926/evaluation_anchored/manifest.json`.
  Status was `COMPLETED`; records `6/6`; requests `84/84`; failed-energy,
  other-failure, unresolved, and safety-nonzero counts were all zero.

The fit reused development seed `890401`; it is not an independent
qualification. The anchored evaluation did not consume heldout data.

## Three-candidate decision

| candidate | split | normal / emergency | priority-weighted wait | low-SOC wait |
|---|---|---:|---:|---:|
| source `03c0` | overall | 76 / 8 | 86.6690576161 | 89.7792692439 |
| source `03c0` | train | 39 / 3 | 70.6967099479 | 62.4444731092 |
| source `03c0` | validation | 37 / 5 | 102.9545885720 | 115.0113887529 |
| V1 runtime rank | overall | 73 / 11 | 94.5130544827 | 85.3075803003 |
| V1 runtime rank | train | 39 / 3 | 74.0396172435 | 55.8153729852 |
| V1 runtime rank | validation | 34 / 8 | 115.3879316677 | 112.5311562834 |
| sparse rows `[9,20]` | overall | 74 / 10 | 83.4356842607 | 90.9519380786 |
| sparse rows `[9,20]` | train | 39 / 3 | 70.0261082970 | 58.6985320610 |
| sparse rows `[9,20]` | validation | 35 / 7 | 97.1081930863 | 120.7243128641 |
| anchored rows `[9,20]` | overall | 75 / 9 | 89.7123814381 | 103.6657082349 |
| anchored rows `[9,20]` | train | 40 / 2 | 70.2204691126 | 58.6151987277 |
| anchored rows `[9,20]` | validation | 35 / 7 | 109.5864881228 | 145.2507939339 |

Primary decision: all three candidates are rejected for adoption; source
checkpoint `03c0` remains the reference and training degradation is not
resolved. Runtime-label mismatch, non-target proposal-head drift, and the
reporting interface have been corrected. Coverage improvement is not
end-to-end utility evidence. The next bounded diagnosis targets action credit
for terminal `normal`/`emergency` outcomes and long-range waiting; continued
coverage/MSE tuning or a larger formal run is not treated as a validated
solution. The frozen protocol remains unchanged, with
`P3_QUALIFIED=false`, `P4_ALLOWED=false`, and `PPO_ALLOWED=false`.

## Rank delta

`rank_delta.json` records 23 rows with a changed top-3 order and one row with
an argmax change. The aggregate counts are:

| split | rows | top-3 order changes | top-3 set changes | top-1/argmax changes | top-1 set changes |
|---|---:|---:|---:|---:|---:|
| all | 164 | 23 | 19 | 1 | 1 |
| train | 82 | 9 | 8 | 0 | 0 |
| validation | 82 | 14 | 11 | 1 | 1 |

The V1, sparse-target, and anchored evaluations are recorded above; all three
candidates are rejected for adoption.

## Implementation provenance

- Implementation commit: `ed945ffb44257645cd292afba0e24c6969fffc64`.
- Local validation: 51 tests passed; no remote push.
- Sparse-target restriction and nested coverage-screen fix commit:
  `66a0ec6ff9c48f795005e85f55fdea2e0cf8b2d6`; the six-file integration suite
  passed `57` tests; no remote push.
- Anchored collector/trainer/test and evidence integration commit:
  `2097990f3efad3d4184d802b05861a97ea7bcc45`; the specified local suite
  passed `61` tests; Git has no remote and nothing was pushed.
- Collection and fit artifacts are under
  `training_outputs/terminal/p3_runtime_rank_repair_20260926/`.
