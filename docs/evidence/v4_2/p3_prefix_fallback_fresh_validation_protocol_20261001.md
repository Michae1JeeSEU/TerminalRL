# P3 prefix-fallback fresh validation protocol

## Scope and question

This protocol preregisters a new independent development evaluation after the passive profiling pass. It evaluates whether the policy and executor interaction observed on the repaired executor is reproduced on six new fixed scene conditions. It is not the old formal run, held-out evaluation, qualification, default promotion, or an optimizer/root-cause experiment; `PPO_ALLOWED=false` remains in force.

The six fixed scenes are `env886101..886106` with native initial policy seeds `886201..886206`. These scene and seed identifiers are new fixed inputs and were not previously found in the searched repository protocols before this preregistration. Each scene is run with dense24, 24 requests, a 1800-step cap, fixed airspace, `phase_motion_v2`, task-neutral settings, 88D observations, public K=3, frozen proposal temperature 2.5, and `rule_flow_coordination=false`.

Each scene has four frozen arms:

- `BEFORE` source `3fe7`, native legacy K=3;
- `AFTER` source `d449`, native legacy K=3;
- `BEFORE` source `3fe7`, repaired prefix fallback with max25 controller;
- `AFTER` source `d449`, repaired prefix fallback with max25 controller.

The exact source bindings are read from the existing frozen runtime metadata and are not remapped. The checkpoint critic is loaded normally for each arm; warmup `a82` is historical metadata only and is not injected. No policy trajectory is resampled, no intermediate checkpoint is selected, and the native Torch initial sampler is used. It is not an exact keyed CRN after histories diverge.

This is a fair 2×2 panel of six new scene conditions and 24 fixed arms, with at most 43,200 simulation steps. The primary question is whether the repaired executor interaction is independently visible and whether the repaired executor avoids the observed regression. It does not claim a general optimizer explanation or a universal training cause.

## Preflight and execution phases

Preflight must run without environment mutation and validate the six scene/seed bindings, source and checkpoint bindings, selected configuration, output schema, request identity, likelihood/proposal fields, safety/replay invariants, and the no-training/no-held-out flags. Outputs use the existing artifact style under:

`/home/user/training_outputs/terminal/p3_followup_20260930/prefix_fallback_fresh_validation_20261001`

The pilot runs all four arms of cell 1 once. Root review is required after the pilot's technical checks; a business-negative pilot is retained and does not trigger seed replacement. FULL reuses the immutable pilot tapes and runs the remaining 20 arms once. Technical, model, RNG, alias, environment, replay, or safety errors stop the phase while retaining partial artifacts. Natural business or energy failures are retained and the fixed budget continues. Censored, error, and unknown values remain explicit unknowns; there is no resampling, cap expansion, cherry-picking, or automatic retraining.

## Metrics and denominators

Each arm reports generated, admitted, successful, and resolved counts; exact request identities; `N/E/F/D/U` outcomes; normal/emergency counts; wait, request elapsed, and energy; raw reward excluding rejection penalty; rejection penalty; safety and replay checks; and per-cell technical status. All per-cell deltas and regressions remain visible. Aggregate values are reported across the six scenes and are not described as 576 independent samples.

The passive observer's call and timing instrumentation may be reused when available. Any timing output keeps native actor/request construction, private executor/controller invocation, and environment-step wall time separate, reports explicit denominators and distributions, and makes no hardware real-time claim. Instrumentation must not alter actions, state, outcomes, or RNG; otherwise the phase stops.

## Preregistered development screen

Common technical gates require all arms to be natural, exact 24-request identity, safety/RNG/replay checks clear, and no unclassified technical error. For each scene, the opt-in `AFTER-prefix` arm must introduce no new failed, diverted, or unresolved arrival IDs relative to `AFTER` legacy and relative to `BEFORE-prefix`.

For each named reference, aggregate normal and successful counts must be at least the reference and aggregate wait, request elapsed, and energy must be no higher than the reference plus `1e-8`. A strict normal increase or failed decrease versus `AFTER` legacy is required for executor benefit. Stability of `AFTER-prefix` versus `BEFORE-prefix` is sufficient for the nondegradation comparison; no unregistered per-cell cost gate is invented. Original native `AFTER` versus `BEFORE` and repaired `AFTER-prefix` versus `BEFORE-prefix` deltas are reported even when a gate fails, with every local regression retained.

A passing result authorizes only a separate root-reviewed bounded development training-admission review. It does not authorize PPO, automatic launch, formal held-out evaluation, default promotion, retraining, or budget expansion. A negative result is retained for root diagnosis and does not trigger automatic tuning or retraining.

## Provenance and claim boundary

The protocol reuses existing artifact and passive-observer conventions without adding a broad manifest framework. It does not change old helpers, environment rules, student rollout behavior, safety filters, or formal qualification boundaries. The result is a fixed development validation on six new scene conditions; it cannot establish universal causality, exact-CRN difference-in-differences, or a unique P3 training root cause.
