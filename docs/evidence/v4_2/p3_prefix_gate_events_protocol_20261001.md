# P3 prefix gate events replay protocol

**Status: pre-registered preparation scope; no server phase is authorized by
this file alone.** This protocol records a frozen, off-policy replay that
enumerates all registered native requested-RELEASE events, including native
K=3 and later-rank registrations, in the four existing BEFORE-prefix tapes. It does not authorize training, PPO
admission, checkpoint promotion, or a claim that P3 is solved. `PPO_ALLOWED`
remains false.

## Fixed input and budget

The input is the existing four-cell BEFORE-prefix full artifact with SHA
`30c28c03d26466eabeab2a35c9fc117424648f1a3ae17be6f8d69b6b92cb2195`, source
checkpoints BEFORE `3fe7dd1e01d032227521f6fc5c8464fe8d00d98fb0e9f0dbd13411123a27d4ab`
and AFTER `d44990dc7d129573d9145f169df1cbc4f91fa32bc5127b1c58d937260c02a24f`.
The output root is
`/home/user/training_outputs/terminal/p3_followup_20260930/prefix_gate_events_20261001`.

The runner uses only the four existing action tapes and their declared
registration sidecars, at most 1,800 recorded ticks per cell and at most 7,200
stored ticks in total. It creates no trajectory, policy population, seed,
environment condition, or held-out sample. The native initial policy sampler
and all source physics remain unchanged.

## Potential-event enumeration and one replay

Every registered sidecar entry matching a tape request with `requested_gate=1`
is a potential event. Runtime `used=false`, a missing runtime rank, or runtime
rank zero is retained as an unknown or not-attempted descriptor; it is not
silently filtered. Unregistered canonical/system requests remain outside this
catalogue. Rank 1--3, rank>3, and unknown-rank registrations are retained; the
rank is descriptive and is validated against current replay telemetry when
available. Enumeration is ordered by `(absolute step, numeric UID)` and does
not consult probability changes, rewards, terminal outcomes, or labelled
`pre_state.release_state`/`decision_history`. There is no first-state choice,
upper SOC filter, or substitution when a candidate is missing.

Each cell is replayed exactly once over its full recorded cycle with the
existing prefix controller and sidecar order/provider. At every potential tick
the runner records the full 88D pre-world and predicate context, executes the
original tape step once, then checks the current `info.release_decisions` and
`env.last_release_decisions` for approval and the current runtime rank. A stale
approved history is insufficient. Every potential receives predicate values,
unknown fields, and exclusion reasons; excluded events never receive policy
probabilities.

An event is a normal comparison event only if current approval is confirmed,
the actual actor mask is true, the phase is one of
`ring_outer/ring_middle/ring_inner`, SOC is finite and greater than 0.4,
`now >= safety_retry_after`, and there is no committed plan. SOC at or below
0.4 is retained as an auto-emergency exclusion with no p/q. Missing context is
unknown and is not converted to false evidence. No upper SOC bound is applied.

For every normal event, the frozen BEFORE and AFTER policies evaluate the same
saved 88D world under `torch.no_grad`; no second environment step or tape replay
is allowed. The full 25-way proposal q must be bitwise identical. The source
proposal logits and source RELEASE gate log probability are checked against the
BEFORE evaluation only. The record contains `p_before`, `p_after`, `delta_p`,
the full q and source logits, UID/arrival/phase/SOC/time/retry/commit context,
and model/RNG/config/replay bindings. `D` and `local_m` are always `None`; the
descriptive event counts and phase breakdown are not global gains or causal
evidence.

## Phases and integrity rules

Preflight creates no environment and performs no model forward, gradient, or
optimizer operation. It binds the input tapes, sidecars, source files,
configuration, protocol, and runner. Pilot may process cell 1 only. After
review, FULL may reuse the immutable pilot cell and process cells 2–4; it may
not rewrite cell 1. The serialized pilot/full summaries must bind the stored
preflight and pilot hashes.

Model construction RNG is snapshotted and restored separately. The complete
replay scope restores the controller, temporary step hook, caller RNG, and
policy digests in `finally` blocks. A q mismatch, source-LP mismatch, model,
RNG, environment, safety, alias, or exact-replay failure writes retained
partial/error output and stops without retry. Censored, failed, excluded, and
unknown records remain in the output. No new latency or business acceptance
gate is introduced.

No Teacher, optimizer, PPO, held-out data, retry-rule change, safety-rule
change, physics change, or automatic promotion is permitted. This measurement
does not select a repair and cannot by itself establish a P3 root cause.
