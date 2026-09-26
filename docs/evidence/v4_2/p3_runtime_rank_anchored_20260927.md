# P3 runtime-label anchored sparse-target rank repair (2026-09-27)

## Frozen configuration

This protocol records one optional anchored fit in the
`P3_RUNTIME_RANK_REPAIR_V1` family.  The source is the calibrated 03c0
checkpoint
`03c0acb0d561219cbe1dc8dac88cd4df696ade4e4691ff0d4db1d4a72ed5e3ba`.
The existing development seeds `873101`--`873106` and development seed
`890401` are reused; the anchor file may contain only the train seeds
`873101`--`873103` and no validation rows.  This is one `fit_anchored` run,
followed by the same six-development-seed frozen evaluation.  It does not
create new seed qualification or a new collector protocol.

This mode requires `--sparse-target-rows --anchor-dir`.  The sparse mask is
derived from the source train misses and remains restricted to proposal output
rows `9` and `20`.  The source policy's frozen encoder computes one fixed
anchor feature batch from `anchors.npz`; source and candidate raw logits are
compared only on those two output rows.  The loss adds
`0.01 * mean((candidate_anchor_raw[9,20] - source_anchor_raw[9,20]) ** 2)`
to the existing raw-logit rank loss and full 82-row train KL.  All other
proposal rows and model components retain the existing frozen/bitwise rules.

Adam, learning rate `0.001`, gradient clipping `1.0`, 120-step maximum,
train KL cap `0.05` with model and optimizer rollback, and train
`miss_count == 0` early stop remain fixed.  Validation is report-only and is
excluded from anchor construction, anchor loss, and checkpoint selection.

## Reporting boundary

The report records the exact anchor manifest and NPZ hashes, train seed set,
anchor MSE at each accepted or rejected step, and final source/candidate
anchor top-3 order-change count.  No MSE term, seed, row, or validation input
is added outside this frozen configuration.  This mode does not automatically
pass P3 or P4.

## Results

Reserved for the authorized anchored fit and the same six-seed development
evaluation.
