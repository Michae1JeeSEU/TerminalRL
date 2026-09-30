# P3 trained gate-value diagnostic protocol (2026-10-01)

This protocol registers a passive, matched-state mechanism diagnostic. It does
not create a new performance episode and does not qualify P3, PPO, a teacher,
or a held-out result.

## Registered inputs

The input is the completed matched gate-shift run with full SHA
`237bc7393e0dc831204b59aaf29aa99ea55dc142ce7ab3cc9fbc4effe8c4ec3e`, its four
cell JSON summaries, action tapes, and matched-state NPZ files. The native
after-legacy full binding is SHA
`3f65446f49474452d31c3e1498c8019f04f4250cad3e8594c75107a123525919`. The
policy bindings are the registered before checkpoint `3fe7dd1e01d032227521f6fc5c8464fe8d00d98fb0e9f0dbd13411123a27d4ab`, the
after checkpoint `d44990dc7d129573d9145f169df1cbc4f91fa32bc5127b1c58d937260c02a24f`,
and warmup input `a82ed7a287035fb1fc868fc955e056c0481de75bcb99880d5c79f35588dde710`.
The warmup hash is an input binding; no critic value is compared in this
diagnostic.

Four states are fixed before replay: `(step, UID, arrival)` equals
`(168, 3, 3)`, `(202, 2, 2)`, `(198, 2, 2)`, and `(187, 3, 3)` for cells 1–4.
The selection rule is the first chronological native actor-eligible
`ring_inner` row with `0.4 < SOC <= 0.5` and `retry_blocked == false`, ordered by
the native UID order. A missing or changed lock is an input failure; no state
is substituted.

## Replay and branches

Each pre-action state is restored by exact replay of the registered after-legacy
action tape. The replay must contain the registered observation/action/post-state
hashes and the selected UID/arrival identity. The artifact records the actual
arrival prefix (`generated`, `exhausted`, and `future_arrivals_pending`) instead
of assuming that every future request has already been generated. Future and
queued admissions are copied by the exact tape and checked through the common
external-prefix invariant; no new request is sampled by this diagnostic.
The other actors' first actions and proposal shortlists are copied from that
tape. Only the target's first action is replaced by WAIT or one of 25 RELEASE
proposals. Proposal weights are the before-policy distribution over all 25
proposals; there is no proposal filtering or survivor renormalization.

Every branch then continues from the copied state with the frozen before
policy, keyed by UID and absolute tick, for continuation seeds
`9303201..9303204` and a 1800-step budget. There are 26 branches per seed and
104 branches per registered state. Keyed continuation sampling is retained in
the artifact metadata; the diagnostic performs no optimizer, teacher, PPO, or
held-out operation. Technical, model, RNG, replay, and hard-safety errors stop
the phase after preserving the completed branch artifact. Business failures
and censoring remain in the fixed branch grid and do not become zeros.

## Recorded quantities

For a state, `p_before` and `p_after` are the target RELEASE probabilities of
the before and after policies in the same replayed world. For each continuation
seed, `C` is the before-policy q-weighted immediate approval fraction and
`D` is

`D = sum_j q_j [G(RELEASE_j) - G(WAIT)]`,

where `G` is the complete undiscounted branch return including the first
action's reward. `D` is qualified only when all 104 branches are terminal,
replay-valid, and free of technical or hard-safety errors. The state-local
quantity `local_m = (p_after - p_before) * D` is retained per state; it is not
aggregated as a population performance estimate. Incomplete branches retain
observed outcome rows while complete-cycle `D`, energy, and other full metrics
remain unknown.

The artifacts also record per-seed values, normal/emergency/failed/diverted/
unresolved outcome counts where present, request elapsed integrals, and unique
arrival energy rows. Resolved-only energy is labeled separately; incomplete
branches do not receive a fabricated full energy total. Raw branch rewards are
replay observations, with no reward scaling or advantage construction.

## Phase discipline

`preflight` creates no environment and binds the runner, protocol, four cell
groups, NPZs, tapes, checkpoint hashes, configuration, and continuation grid.
`pilot` runs cell 1 only after a fresh output check. `full` reuses the immutable
pilot state and evaluates only cells 2–4. A terminal artifact is a diagnostic
record, not evidence that the P3 performance problem is solved or that a
particular mechanism is causal.
