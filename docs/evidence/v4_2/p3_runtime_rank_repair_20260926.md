# P3 runtime-label rank repair V1 (2026-09-26)

## Frozen configuration

This document records the configuration for one bounded development fit.  The
source is the calibrated 03c0 checkpoint
`03c0acb0d561219cbe1dc8dac88cd4df696ade4e4691ff0d4db1d4a72ed5e3ba`.
The input is the completed runtime-label collection derived from the already
consumed six development seeds `873101`--`873106`: the first three seeds are
the 82-row training split and the last three seeds are the 82-row validation
split.  Reusing development seed `890401` is recorded as a reuse; no new seed
is reserved.

The fit updates only `proposal_head.weight` and `proposal_head.bias` (2,425
parameters).  The candidate uses raw
`model.proposal_head(model.actor_encoder(x))` logits and the objective is
`k3_coverage_objective(raw_logits, C)['loss'] + 1 * KL(source || candidate)`
averaged over all training anchors.  The batch is the complete 82-row
training split, Adam uses learning rate `0.001`, gradient norm clipping is
`1.0`, and at most 120 optimizer steps are attempted.  A post-step training KL
above `0.05` restores both model and optimizer state and stops with
`STOPPED_KL_CAP`; no automatic step-size reduction or retry is performed.  A
training `miss_count == 0` stops the fit.  Validation is reported only and is
not used in the loss or checkpoint selection.

The collection must be `COMPLETED` with `runtime_labels=true`.  Each row's
fixed-other-tape-action runtime singleton scope marker, source checkpoint
binding, and exact source-record flags are checked.  The old proposal-probe
protocol is retained only as collection lineage; it is not the fit protocol.

No floor, forced gate, `lateon`, or reward change is part of this stage.

## Planned execution and reporting boundary

The permitted sequence is a runtime-collection smoke check, the completed
164-row runtime collection, one fit, and a six-development-seed frozen
evaluation.  The post-fit screen follows the existing development screen and
reports strict non-degradation separately.  This stage does not consume
untouched held-out data and does not automatically pass P3 or P4.

## Results

Reserved for the execution report after the bounded fit is authorized and
run.  Code and tests establish the input, update-scope, rollback, and
checkpoint-roundtrip contracts; they do not claim a training result.
