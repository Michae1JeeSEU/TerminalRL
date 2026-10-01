# P3 counterfactual gate adapter fit protocol — 2026-10-01

This protocol records the finite fit explicitly authorized by the user with
“继续”. It prepares one intrinsic, adapter-only candidate from the frozen
BEFORE checkpoint. It is not a business qualification, performance claim, or
PPO admission.

## Frozen inputs and model

The target JSON is `targets_v1.json` (SHA-256
`4085299501758c55c476fac9676092ace9d9c69f4ea5d3a68b74f786eaaf1473`). The
anchor manifest is the reviewed 4-cell bundle (SHA-256
`1129676804003dd5a569a7723015ef9ceddd8c1cc016cb1ffca02c3cd4318d35`) with
2,749 replay ticks and 28,570 actor rows. The target artifact retains all four
fixed states and all four continuation seeds; no rows or states are selected by
their measured D value.

The source is the BEFORE `terminal_gate_repair.pt` checkpoint (SHA-256
`3fe7dd1e01d032227521f6fc5c8464fe8d00d98fb0e9f0dbd13411123a27d4ab`, model
digest `3434378719efdb02bf3d41febb52e66797a0d3928756fc0cbbfde056ddd337b0`).
The fit constructs `TemperatureCalibratedGateResidualAdapterActorCentralCriticV42`
directly with the calibrated temperature 2.5 and loads the source `model`
state strictly. It does not instantiate the StudentPPO wrapper and does not
load the source optimizer or training history as fit state.

Only parameters named `gate_adapter.*` are trainable. The actor encoder, gate
head, proposal head, temperature buffer, centralized critic, and all other
parameters and buffers remain frozen. The fresh optimizer is Adam with learning
rate `1e-4`, weight decay zero, fit seed `9303501`, CPU execution, and at most
80 updates. The output metadata keeps inference schema/configuration fields,
records the parent source hash, and explicitly keeps `PPO_ALLOWED=false`,
`teacher_used=false`, `heldout_used=false`, and `candidate_promoted=false`.

## Objective and hard checks

For every update, the loss is the mean of the four target BCE terms plus one
times the equal overall mean Bernoulli KL
`KL(candidate || BEFORE)` over all 28,570 anchor rows. Anchor rows are processed
in chunks of 512, with each chunk weighted by its row count. The gate release
probability is the calibrated two-class release probability. The proposal
distribution is not optimized and must remain bitwise equal to the source.
The KL calculation uses stable logits in double precision and introduces no
probability floor or argmax label.

The recorded target `p0` values must match the direct BEFORE model within
`1e-6`. After every optimizer step, all anchor and four target KL values must
remain at most `0.001 + 1e-10`, and loss, gradients, and state must be finite.
If this cap is exceeded, the previous valid adapter parameters are restored,
the rejected update is recorded, and the fit stops with `ROLLBACK_STOP`.
There is no alternate learning rate, retry, or best-of selection.

The intrinsic screen requires a target BCE decrease greater than `1e-10`,
non-worsening target distance at every fixed target within `1e-6`, the anchor
and target KL caps, unchanged frozen state, and unchanged proposal
probabilities. A candidate checkpoint is written once only when all checks pass.
The checkpoint is strict-reload checked and contains no optimizer state. A
failed intrinsic screen writes its measured fit report and no candidate
checkpoint. Technical, nonfinite, source-binding, or frozen-state errors stop
and preserve a partial/error artifact.

Caller Python, NumPy, and Torch RNG states are snapshotted before model
construction and restored after the fit. The fit creates no environment, does
not sample a trajectory, and does not use PPO, teacher data, held-out data, or
additional populations.

## Interpretation boundary

The result is an offline intrinsic screen for a soft gate target. It does not
establish a business improvement, a causal explanation of P3 degradation, a
qualification result, or a default promotion. Any later shared-anchor fitting
or physical evaluation requires a separate root review and preregistration;
the four fixed states and 416 counterfactual branches are not an independent
performance population.
