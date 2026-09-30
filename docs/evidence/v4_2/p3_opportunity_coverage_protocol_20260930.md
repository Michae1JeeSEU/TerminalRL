# P3 executable-opportunity coverage diagnostic (2026-09-30)

This protocol registers a passive, frozen-policy diagnostic for the next P3
cause check. It is a development diagnostic, not PPO training, a checkpoint
selection procedure, or a qualification run. `PPO_ALLOWED=false` remains in
force; the run does not use a Teacher, an optimizer, held-out inputs, or a
changed environment rule.

## Fixed inputs and scene layout

The source is the existing `3fe7` checkpoint followed by the existing `a82`
warmup model. The runtime is the neutral `dense24` configuration with the
88D/V2.1 observation schema, V4.2 hierarchical action schema, K=3 runtime
shortlists, legacy action mask, and the existing safety, energy, service, and
automatic-emergency paths. The frozen source model and all existing runtime
files are bound to the prior forced-WAIT preflight. This diagnostic adds one
new runner file and does not modify those runtime files.

The reference directory is
`/home/user/training_outputs/terminal/p3_followup_20260930/forced_wait_frozen`.
The reference is read-only. Its `full.json` must describe the completed 24
pair comparison. The nine legacy arms are read from pair IDs
`1, 5, 9, 13, 17, 21, 14, 15, 16`, corresponding respectively to:

| pair IDs | environment seed | keyed policy seeds |
|---|---:|---:|
| 1, 5, 9, 13, 17, 21 | 877101, 877102, 877103, 877104, 877105, 877106 | 9303101 |
| 14, 15, 16 | 877104 | 9303102, 9303103, 9303104 |

These are six fixed demand profiles with three additional keyed-policy
continuations for the `877104` profile, not 216 independent requests. The
diagnostic replays each selected scene with the same environment/keyed policy
inputs and must exactly match its legacy reference arm before its opportunity
records are considered technically usable.

## Registered opportunity windows and branches

The absolute tick windows are `[100,149]`, `[300,349]`, and `[500,549]`.
Only the first state in each window that satisfies all registered predicates
is selected. A selected actor must be in an outer, middle, or inner ring, have
no committed plan, have positive SOC, have the legacy actor mask set to true,
be at or past `safety_retry_after`, and have its native interface outside the
current affected-interface geometry. Positive-SOC urgent actors remain
eligible; `emergency_retry_after` is not added to the normal mask. In
`877104` the target must have arrival ID 13. In the other scenes the eligible
actor with the greatest internal wait is selected, with the smallest arrival
ID breaking ties. If a window has no eligible state, the result is explicitly
unknown; the runner does not search another window or seed.

At a selected state, the real keyed request and all other actors' actions and
shortlists are frozen. The runner evaluates exactly 26 one-step shadow
branches from one restored snapshot: WAIT and all 25 proposal indices, each
with its ordinary K=3 shortlist. The source proposal probabilities and logits
remain frozen. A branch records only its immediate reward difference from the
WAIT branch and its native decision. `C` is the sum of the original proposal
probabilities for branches whose native decision is exactly `approved`;
`emergency_redirected` is not counted as approval. `pC` is the gate RELEASE
probability times `C`. Any branch error, technical mismatch, or hard-safety
violation makes `C` and `pC` unknown. No survivor renormalization, best-of-25
selection, full-cycle return, `D`, or local `m` is produced here.

For the arrival-13 trace, every real tick records phase, SOC, gate RELEASE
probability, maximum proposal probability, retry time, and resource context,
plus the actual native decision after the real step. Each selected state also
stores the target 88D world observation, the full 25 proposal probabilities
and logits, and the real joint action/shortlists for later before/after
analysis.

## Integrity and stopping rules

Every shadow branch is made from a deep copy restored with the existing paired
diagnostic hook repair. The live environment, frozen model, and caller RNG
must remain unchanged. The source replay is compared against the selected
legacy arm on request identity, ticks, ledger outcomes, terminal rows, wait,
elapsed, energy, model/RNG flags, and hard-safety fields. Reward comparison
uses an absolute tolerance of `1e-9`. Errors, censored states, and unknown
values are retained. A source, environment, RNG, replay, model, or safety
mismatch stops the phase and preserves its artifacts. The diagnostic never
retries a failed branch or removes a failed/censored result.

The pilot is pair `1` and contains all three windows. The full phase reuses
that pilot artifact and evaluates only the other eight fixed pairs. Outputs
are written atomically and existing artifacts are never overwritten.

## Digest-binding amendment (2026-09-30)

The first pilot completed as `PILOT_OK`, but its pair summary contains two
different model-digest conventions that must remain distinct. The existing
`source_model_digest` summary field is the execution digest produced by
`paired_gate_diagnostic._model_digest(policy)`; the canonical source binding
is the digest produced by
`run_p3_task_neutral_pilot._model_digest(policy)` and is stored as
`canonical_source_model_digest` in newly written pair summaries. The summary
also records both algorithm names explicitly. The full-phase reuse guard
compares the pilot execution digest with the live policy using the paired
diagnostic algorithm, and compares the live canonical source digest with the
stored preflight source binding using the canonical algorithm. This preserves
the existing V1 `source_model_digest` field and accepts a V1 pilot artifact
that predates the additional algorithm-label fields when its computed values
pass these checks.

The repaired full run uses the new output root
`/home/user/training_outputs/terminal/p3_followup_20260930/opportunity_coverage_v2`.
Its preflight binds the repaired runner while retaining the same source,
configuration, and existing 115 runtime-file hashes. After read-only checks
of the V1 pilot, arm, pair, and three state artifacts, those artifacts may be
copied into the V2 output root for reuse; the copy is an output-artifact
operation and does not update source code. The repair does not change the
physical environment, action, window, proposal, sampling, or score rules,
and does not rerun failed or censored branches.
