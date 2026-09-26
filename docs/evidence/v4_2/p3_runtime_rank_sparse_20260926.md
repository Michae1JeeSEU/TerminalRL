# P3 runtime-label sparse-target rank repair (2026-09-26)

## Frozen configuration

This short protocol records one optional sparse-target run in the
`P3_RUNTIME_RANK_REPAIR_V1` family.  The source remains the calibrated 03c0
checkpoint
`03c0acb0d561219cbe1dc8dac88cd4df696ade4e4691ff0d4db1d4a72ed5e3ba`.
The already consumed development seeds `873101`--`873106` are reused, with
the existing 82-row train and 82-row validation split; development seed
`890401` is also reused.  No new seed or collection is introduced.

The optional `--sparse-target-rows` mode derives proposal output rows only
from source initial training K=3 misses: for each miss it takes the
highest-logit certified proposal, then uses the union of those rows.  The
current source collection derives rows `9` and `20`.  The proposal head keeps
all 2,425 parameters under `requires_grad`, while the optimizer gradient mask
allows updates only to the derived output rows (194 effective parameters for
rows `9` and `20`).  All other proposal output rows must remain bitwise equal
to source.

The loss and update limits remain unchanged: raw proposal logits,
`k3_coverage_objective(raw_logits, C)['loss'] + 1 * KL(source || candidate)`
over all 82 training anchors, Adam at `0.001`, gradient clipping at `1.0`,
maximum 120 steps, post-step train KL cap `0.05` with model and optimizer
rollback and stop, and train `miss_count == 0` early stop.  Validation is
reported only and is not used for loss or selection.  No MSE, floor, forced
gate, `lateon`, or reward change is added.

## Planned execution and reporting boundary

Run one `fit_sparsetargets` using the completed collection, then evaluate the
same six development seeds with the frozen development screen.  Record the
derived mask rows, the 2,425 `requires_grad` parameter count, the effective
update count, and the non-target proposal-row bitwise invariant.  This mode
does not automatically pass P3 or P4.

## Results

Reserved for the authorized sparse-target fit and the same six-seed
development evaluation.
