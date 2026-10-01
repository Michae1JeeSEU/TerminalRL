# P3 prefix-preserving fallback controller protocol

**Status: pre-registered implementation/test design; not launched.** This document records root-approved scope for a frozen comparison; execution phases remain subject to stated preflight/pilot review. It does not authorize training, PPO, checkpoint promotion, or a default runtime change.

## Objective and scope

The experiment tests whether a frozen `AFTER=d449` policy can use additional raw-source-logit ranked proposals after the native sampled-first K=3 prefix is rejected. The controller is instance-scoped with `policy_requested_k3` and `controller_max25`. It applies only to the four existing development cells:

| cell | environment seed | policy seed |
|---:|---:|---:|
| 1 | 875101 | 875201 |
| 2 | 875102 | 875202 |
| 3 | 875103 | 875203 |
| 4 | 875104 | 875204 |

There are no new profiles, seeds, populations, held-out data, or replacement baseline arms. The immutable after-legacy baseline is imported from `trained_proposal_support_20261001/full.json`, SHA256 `3f65446f49474452d31c3e1498c8019f04f4250cad3e8594c75107a123525919`.

The source and checkpoint bindings are imported from the existing protocol: `AFTER=d449`, `BEFORE=3fe7`, warmup `a82`, together with the frozen configuration, helper, runner, protocol, baseline-arm, model, and caller-RNG hashes. A no-environment preflight must verify every binding before a pilot or full phase.

## Fixed controller behavior

The controller uses the same native gate, proposal distribution, requested-action probability `q`, logprob, and native sampler. It preserves the native sampled-first K=3 prefix and the original private release call. The source-logit ranking is captured from that same native forward pass; there is no second forward, resampling, teacher/oracle choice, or model change.

The original private release loop is called once with at most 25 candidates. If one of the first three sampled-first candidates (one sampled proposal and two source-ranked entries) succeeds, the native early-stop, retry, emergency, safety, capacity, and reservation behavior applies. If all three are rejected, the controller continues with the remaining candidates in stable raw-source-logit rank order, up to `controller_max25`. The native core functions are unchanged. The explicit capacity or reservation break remains a native early stop.

Expanding the all-checked `emergency_energy_guard` exhaustion criterion from three checked candidates to as many as 25 can change induced emergency behavior when the first three all have that reason. The protocol therefore measures emergency outcomes, failures, energy, and per-cell regressions; it does not claim that emergency behavior is identical to the native K=3 path. The same applies to any trajectory divergence after a fallback succeeds.

The controller records the per absolute tick and target UID candidate order in a sidecar. The sidecar is used only to replay the same action tape under the same controller. UID is an action-map key; it is not the continuation key. UID/tick identifies saved runtime candidate orders for exact tape replay; policy sampling retains the native Torch RNG sequence seeded by `policy_seed`. No continuation branch seeds or keyed CRN sampler are introduced. Requested-action `q` is not the probability of an executed fallback action.

## Comparison and execution phases

The candidate is an off-policy frozen-execution comparison and produces no PPO data. The pilot runs cell 1 only after preflight and root review. The full phase reuses the immutable pilot artifact and runs cells 2–4. The cap is 1800 steps per arm. No rerun, resampling, seed selection, budget expansion, or population expansion is permitted.

All failure, censor, error, invalid, diverted, unresolved, and safety records are retained. A hard technical, model, environment, RNG, source-binding, or safety failure stops the phase. A censored or failed business arm remains observed with its complete-cycle metrics unknown where appropriate; it is not replaced or rerun.

The native seeded pairs share the configured initial inputs, but they are not exact CRN after action histories diverge. This comparison must not be presented as an exact-CRN causal estimate or as a generalization result.

## Required preflight and evidence

The no-environment preflight must bind and report:

- source `d449`, before `3fe7`, warmup `a82`, frozen configuration, helper, runner, protocol, baseline-arm, model, and caller-RNG hashes;
- `policy_requested_k3`, `controller_max25`, cap `1800`, native sampler/gate/proposal/q/logprob mode, and the unchanged safety/retry/capacity settings;
- the four fixed cell identities and the imported after-legacy baseline records;
- `PPO_ALLOWED=false`, `heldout=false`, `teacher=false`, `optimizer_used=false`, and `training=false`.

Each arm must retain source and model digests, caller and observer RNG checks, action-order sidecar, request identities, ledger, energy, wait and elapsed metrics, raw reward, rejection penalty, terminal rows, replay checks, wall time, and all failure classifications. Raw reward and rejection penalty must be reported separately.

## Development candidate screen for this development comparison

The screen is applied only after all four candidate cells have terminal evidence:

1. Every cell has 24 request identities and passes natural-terminal, technical, safety, ledger, energy, wait, and action-tape replay checks.
2. No cell introduces a new failed, diverted, or unresolved arrival ID relative to its immutable after-legacy baseline.
3. Aggregate normal and successful-completed counts are at least the after-legacy baseline.
4. Strict business gain is present through an aggregate normal increase or failed-count decrease.
5. Aggregate wait, request elapsed, and energy do not exceed the baseline by more than `1e-8`.
6. The report exposes every per-cell regression, the check and wall-time denominators, raw reward versus rejection penalty, and any emergency or energy change induced by the expanded candidate search.

A censored or technically incomplete full set cannot pass this screen. Passing would remain a development result requiring separate review; it would not authorize PPO, qualification, default promotion, or a claim that P3 has been solved.

## Evidence boundary and rationale

The design is motivated by the existing source-16 step-445 witness: proposals 22/23 had positive complete-cycle `G(RELEASE)-G(WAIT)` across four continuation seeds while their requested probabilities were extremely small. That is a local q-starvation witness, not a population estimate or a proven P3 cause. The single failure witness is not treated as a global root cause. The previously tested uniform-q floor was rejected after the trained comparison showed wait increasing by `9.163%` and energy by `4.493%`; this protocol does not claim a universal cure.

The experiment must not be used to claim a causal training explanation for entropy, team credit, gradient clipping, or any other optimizer mechanism. It does not authorize Teacher, optimizer, PPO, held-out, training, default promotion, or source replacement. `PPO_ALLOWED=false` remains in force.
