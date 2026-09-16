# I1 V2 / untouched Q2 — PASS

Evidence type: newly_run. Output:
`~/training_outputs/terminal/v4_2_i1_v2_formal_20260917`.
Q2 completed all 45 preregistered episodes, 1,787 rows, 9/9 exact replays,
all support/schema/independence/aliasing/safety gates PASS. All safety counts
were zero. Its manifest SHA256 is
`5e45f452092083aed786ab33097d2b5ad311874756f88557cbf28a59ec7b0ced`.

Three fresh actors (6201/6202/6203) trained on 7,576 R4 training rows only.
All final checkpoints froze before the single Q2 evaluation. All three
seeds passed every original hard gate. Across-seed minimum / mean:

| Metric | Minimum | Mean |
|---|---:|---:|
| WAIT recall | .994656 | .995547 |
| RELEASE recall | .948276 | .977012 |
| Gate macro-F1 | .979127 | .986229 |
| Proposal top-1 | 1.000000 | 1.000000 |
| Proposal top-3 | 1.000000 | 1.000000 |
| Decoded accuracy | .988808 | .992539 |

Qualification SHA256:
`0e3817ad4fc928f20ba8f6cff7cf7b61801507c557e62daa53aa845039cf87cc`.
Canonical seed6201 checkpoint SHA256:
`f1a5dce05657173d97c402da6d28fa47b59bf016bd61304b3d5fb84f23c7b172`.

`I2_PREFLIGHT_ALLOWED=true`; `PPO_ALLOWED=false`. This demonstrates fixed
cohort decision generalization, not closed-loop operational performance.
The earlier registry-stall notification was a monitor path error: collection
progress is `q2/collection_progress_qualification.json`. The run was never
stopped or restarted and Q2 was not evaluated again.
