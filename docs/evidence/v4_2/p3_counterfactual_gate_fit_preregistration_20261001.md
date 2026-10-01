# P3 counterfactual gate adapter fit pre-registration

This document registers a readiness-only input contract for a possible future
gate adapter fit. It does not authorize fitting and does not create an
optimizer, run a model forward pass, calculate gradients, write a candidate
checkpoint, run PPO, or access held-out data. The accompanying preflight
module only validates metadata and immutable replay-anchor files.

The fixed target artifact is the prepared four-cell soft-target JSON output
with SHA256
`4085299501758c55c476fac9676092ace9d9c69f4ea5d3a68b74f786eaaf1473`; this
SHA is for the target JSON, not for the raw gate-value full output. The
future anchor input must bind to the registered source full artifact
`30c28c03d26466eabeab2a35c9fc117424648f1a3ae17be6f8d69b6b92cb2195`, event
full artifact
`fb89f62339755f7ebcd2bb0f9672438000fe8b754067cd2cdb28bf8407feb3e6`, BEFORE
checkpoint `3fe7dd1e01d032227521f6fc5c8464fe8d00d98fb0e9f0dbd13411123a27d4ab`,
and AFTER checkpoint
`d44990dc7d129573d9145f169df1cbc4f91fa32bc5127b1c58d937260c02a24f`. It
contains cells 1--4 with registration keys `307|15`, `321|25`, `303|20`, and
`365|46`, respectively. The expected replay input is 2,749 source ticks in
total. Each tick has one `step_metadata` record, including an empty UID list
when there are no active actors; actor rows are variable and are counted
separately from ticks. The companion NPZ has finite float32 88-dimensional
world observations and an immutable row identifier for every actor tick/UID.

Each cell must report `steps_expected` and `steps_replayed` equal to its
registered tick count, `replay_exact=true`, controller restoration, and both
caller-RNG restoration checks. For these tapes, its ordered `step_metadata`
must cover exactly `step=0..N-1`, with contiguous `row_start`/`row_end` ranges
and UID order matching the JSON actor rows. The manifest checks the cell JSON SHA and
NPZ SHA. The NPZ `step`, `uid`, `arrival_id`, `row_index`, `actor_mask`,
`requested_gate`, and `requested_proposal` arrays must match those rows; a
duplicate `step|UID`, missing empty tick, range gap, or malformed mapping
leaves the readiness result `NOT_READY`.
Source-provenance and safety fields are carried as recorded; missing or null
values remain unknown and are never converted to a safety-clear or zero value.

The registered future fit specification is metadata only at this stage:
load the BEFORE checkpoint, leave the critic, proposal head, and temperature
frozen, and allow only `gate_adapter.*` parameters to be trainable. If ROOT
later authorizes the fit, its fixed seed is `9303501`, learning rate is
`1e-4`, and the update cap is 80 deterministic full-batch updates. The future
optimizer metadata is fresh Adam with zero weight decay; no optimizer is
created by this preflight. The objective is the mean BCE over the four target
rows plus coefficient 1.0 times the mean Bernoulli KL from the candidate gate
probability to the BEFORE probability over every anchor row. Anchor rows may
be processed in chunks of 512, but the final anchor term remains one equal
overall mean across all rows. The cap is KL `0.001`. A violation would restore
the previous valid adapter parameters and stop. No alternative seed, learning
rate, epoch count, sampling rule, or survivor reweighting is registered here.

The preflight result is `NOT_READY` when the target or anchor bundle is
missing, has an unknown status, has a binding mismatch, omits a row, contains
duplicate row identifiers, or has a non-finite or non-88D world array. It
never fabricates anchor observations or target values and never drops failed,
censored, or unknown rows to make a bundle complete. A complete metadata
check produces `READY_FOR_ROOT_REVIEW`; this remains a review artifact with
`fit_authorized=false`, `optimizer_used=false`, `backprop_used=false`,
`model_forward_used=false`, `candidate_checkpoint_writer=false`, and
`PPO_ALLOWED=false`.

The four target states carry mixed four-seed uncertainty. Their local D and
local-m values remain evidence fields; they are not anchor labels, are not
summed into a global gain, and the four states are not a qualification
population or a system-wide guarantee. The preflight tool does not fit a
shared policy and does not make a performance claim.

If a later ROOT-admitted intrinsic fit is run, its mechanical screen is fixed
in advance: final four-row target BCE must be lower than its initial value by
more than `1e-10`; each target must satisfy
`abs(p_after-pstar) <= abs(p_before-pstar)+1e-6`; the maximum anchor KL and
target KL must be at most `0.001+1e-10`; and critic, all non-adapter
parameters, proposal head, gate head, temperature, and source q values must
remain frozen and bitwise equal. A save/reload check must also pass. These are future
intrinsic checks only; this preflight does not execute them or authorize an
optimizer.

The bounded command shape is:

```text
python -m terminal_marl.training.p3_counterfactual_gate_fit_preflight_20261001 \
  --targets <targets_v1.json> --anchors <anchor_bundle> --output <readiness.json>
```

The output path is write-once. Any later fit, checkpoint writer, or business
qualification requires a separate explicit ROOT admission and a new audited
phase.
