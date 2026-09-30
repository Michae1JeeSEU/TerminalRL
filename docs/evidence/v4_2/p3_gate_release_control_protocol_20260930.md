# P3 frozen always-RELEASE gate control (2026-09-30)

This protocol registers a frozen off-policy control for the P3 gate factor.
It reuses the frozen source, the existing `dense24` runtime, and the
completed forced-WAIT legacy arms. It does not train, update, qualify, or
promote a policy. `PPO_ALLOWED=false`; Teacher, optimizer, held-out inputs,
and checkpoint selection remain disabled.

## Fixed design

The control uses pair IDs `1, 5, 9, 13, 17, 21, 14, 15, 16`, the same nine
fixed scenes registered for the opportunity diagnostic, with environment seeds
`877101..877106` and keyed policy seeds `9303101..9303104` as registered in
that diagnostic. The simulation cap remains the frozen 1800-step cap. The
existing V2 opportunity preflight is read as the source/config/runtime
binding; the control writes a separate output root:

`/home/user/training_outputs/terminal/p3_followup_20260930/gate_release_control`

The only action intervention is the gate factor. For every actor with the
frozen legacy `actor_mask=true`, the control sends `RELEASE`; for every actor
with `actor_mask=false`, it sends canonical `WAIT`, including committed,
non-ring, and other actors excluded by the existing legacy mask. The
proposal distribution, proposal logits, keyed uniforms, K=3 shortlist rule,
physics, native safety, energy, emergency, and protected-floor rules are
unchanged. The proposal is sampled from the frozen proposal probabilities with
the same `arrival_id|absolute_tick|stream` proposal uniform. The sampled
proposal occupies shortlist rank 1; the remaining two entries are filled by
the highest source-logit candidates, using the existing shortlist helper.
The gate uniform is recorded for eligible actors to preserve the registered
common-random-number key; its value is not used to select the forced gate.

All actors receive this same rule. UID is used only as the action-map key and
the keyed draws remain arrival-ID based. The control is off-policy and does
not produce log probabilities, teacher labels, optimizer state, or PPO data.
It is a gate-factor control, not an oracle and not a claim of global
feasibility.

## Measurements and integrity

The report presents outcomes in this order: primary safety, then
normal/emergency/failed/diverted/unresolved categories with generated,
admitted, completed, and resolved denominators, followed by wait, elapsed,
and energy metrics. `reward_total`, raw reward excluding rejection penalty,
and rejection penalty are retained as separate fields. A gain caused only by
reducing a rejection penalty is recorded as penalty movement and is not
treated as a business improvement.

Each control arm is checked against its existing legacy reference for the
fixed request identity and count (24), ledger conservation and terminal
alignment, energy coverage, wait/elapsed integrity, source model/RNG
integrity, and hard-safety fields. A hard technical or safety error stops the
phase after preserving the arm and progress artifacts. Business failures and
censored arms are retained with their original status and are not rerun.

The pilot is pair 1. Full reuses its completed pilot artifacts and evaluates
only the other eight pairs. Existing reference and prior diagnostic outputs
are never overwritten. No conclusion is registered from a single scene or a
local gate result. A later control acceptance review may consider the result
only when all nine arms are natural and complete, each scene admits 24
requests, no new failed, unresolved, or diverted arrival ID appears relative
to its paired legacy arm, the aggregate normal and successful-completion
counts do not decrease, total wait does not increase, and at least one of
normal count increase, failed count decrease, wait decrease, request-elapsed
decrease, or energy decrease is strictly observed. Resolved count remains a
separate denominator and is not treated as successful completion. Each pair
retains its normal/wait deltas so scene-level worsening cannot be hidden by an
aggregate. Censored or unknown values make the acceptance result unknown and
ineligible. Passing this review would still require the separate
qualification process.
