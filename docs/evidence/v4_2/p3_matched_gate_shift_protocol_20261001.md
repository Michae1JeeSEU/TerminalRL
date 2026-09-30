# P3 matched gate-shift diagnostic protocol (2026-10-01)

## Scope and registered inputs

This is a passive, matched-state observation of the four completed
`after_legacy` tapes from the trained proposal-support evaluation.  It does
not create new performance episodes, sample a policy action, compute a
return, or update a model.  Repeated rows from one replayed trajectory are
reported as state observations and are not independent trials.

The registered cells are `cell_01` through `cell_04`, with the existing
development `dense24` configuration, environment seeds `875101`–`875104`,
and policy seeds `875201`–`875204`.  The input bindings are recorded by the
runner and must match the registered full artifact SHA
`3f65446f49474452d31c3e1498c8019f04f4250cad3e8594c75107a123525919`, pilot
artifact SHA `b9133fdbe1c94b7e2a2d5225c052c103339bb55ca2d59d6df3e885a871a55cda`,
and preflight artifact SHA
`0911e9a9edafc17b2108ea9c56f770f7c54cbd1dfbc6b3c72b17974fc2a2fe38`.
The before, after, and warmup checkpoint bindings are respectively
`3fe7dd1e01d032227521f6fc5c8464fe8d00d98fb0e9f0dbd13411123a27d4ab`,
`d44990dc7d129573d9145f169df1cbc4f91fa32bc5127b1c58d937260c02a24f`, and
`a82ed7a287035fb1fc868fc955e056c0481de75bcb99880d5c79f35588dde710`.

## Replay and observation procedure

`preflight` checks the artifact, tape, configuration, checkpoint, model-freeze,
and protocol bindings without creating an environment.  `pilot` replays only
cell 1.  `full` reuses the immutable pilot result and replays cells 2–4.  A
replay mismatch, source mismatch, model mutation, caller-RNG mutation, safety
invariant failure, or serialization error stops the phase before the next
tape.  Existing natural outcomes are retained; no failed or censored episode
is removed or rerun.

For each recorded tape step, the observer computes the native V4.2 observation
and actor eligibility mask.  The recorded action tape remains the only action
executed.  Deterministic request details are used only to recover UID order,
world rows, eligibility, and runtime shortlist metadata; their actions are
never passed to `env.step`.  The before and after policies are evaluated with
`torch.no_grad()` on the same 88D world rows and no categorical draw.

The comparison freezes the actor encoder, proposal head, gate head, and gate
temperature groups.  Proposal logits and probabilities must remain equal
within `1e-7`.  Gate release probabilities, entropy, proposal maxima,
analytic argmax differences, and bidirectional gate KL are recorded by
outer/middle/inner phase.  Aggregate gate statistics retain an all-state
scope and provide a separate actor-eligible scope so service-only rows are
not interpreted as executed actor decisions.  On eligible rows, the after-policy gate and
proposal log probabilities must match the requested tape likelihoods within
`1e-5`; inactive rows are excluded from this likelihood check.  This is an
analytic actor comparison and does not establish a counterfactual action
value, credit assignment result, or PPO/entropy/clipping cause.

## Artifacts and limits

Each completed cell writes a JSON summary and a compressed
`*_matched_states.npz` containing the exact 88D world rows, step offsets,
UID/arrival/phase/SOC/mask metadata, before/after normalized gate logits and
probabilities, common proposal probabilities, and requested action likelihood
fields.  Eligible-row gate KL is descriptive and computed in both directions.
The runner writes progress every 50 replayed steps.  The NPZ also stores the
raw team reward returned by each exact `env.step` replay, one value per tape
step, and checks its sum against the registered arm total; no reward scale is
applied.  These are replayed observations, not counterfactual returns.  No
advantage, return, or critic comparison is written.  Any unavailable optional
trajectory field remains missing.  `PPO_ALLOWED=false`, optimizer use is false, and
Teacher/heldout data are not used.

The output is descriptive evidence for the registered tapes only.  It does
not qualify a new policy, claim that the P3 performance degradation is solved,
or establish a causal explanation for any observed gate shift.
