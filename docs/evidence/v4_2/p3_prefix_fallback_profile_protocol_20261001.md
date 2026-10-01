# P3 prefix-fallback passive profiling protocol

## Scope and status

This document pre-registers a passive measurement pass over existing immutable tapes. It profiles the native execution path and the already completed prefix-fallback comparisons without changing policy outputs, actions, physics, safety rules, or training state. It is not a new performance population, a counterfactual experiment, a qualification, or a PPO admission; `PPO_ALLOWED=false`.

The fixed panel is the four existing cells (`env875101..875104` with their existing native initial policy seeds `875201..875204`) crossed with four existing arms: native `BEFORE` legacy, native `AFTER` legacy, `BEFORE-prefix`, and `AFTER-prefix`. This is exactly 16 existing tapes. No policy trajectory is regenerated, no seed or population is resampled, and no held-out data, training, optimizer, or student rollout is used.

The immutable inputs are the original native full artifact SHA `3f65446f49474452d31c3e1498c8019f04f4250cad3e8594c75107a123525919`, the completed `BEFORE-prefix` full SHA `b4ee3a191c26911edf94fd24e1da4077bd2a0962e10cd5507174b52a8af73028`, and the completed `AFTER-prefix` full SHA `020312813fc2c90a445bd3fe30e94e6739797151fe46efb65f760809a9361c50`. Existing arm metadata binds the frozen source set `3fe7`/`d449`; the exact stored bindings are read from those artifacts and are not remapped. Warmup `a82` is metadata only. Existing helper, runner, environment, and student-rollout files remain unchanged.

## Replay and phase scope

The pilot replays cell 1's native `BEFORE` legacy and `BEFORE-prefix` tapes exactly once. Root review is required before FULL; FULL reuses those pilot profiles and processes the remaining 14 immutable tapes once. The original tape and sidecars provide the replay observations and action records. The observer may request policy outputs for measurement, but actor-probe outputs are never executed and no policy or hyperparameter behavior is changed.

Every arm must preserve the original full-request ledger, request identities, outcomes, energy, wait, reward, hard-safety status, and model/caller/observer RNG restoration. Existing failed, censored, unknown, and error rows remain retained. A hard technical, safety, state, or RNG mismatch stops the phase; there is no automatic rerun, seed selection, or failure removal. If profiling changes actions, outcomes, state, or RNG, validation stops and the affected result remains invalid/unknown.

## Passive counters

The observer records actual runtime calls and keeps separate denominators:

- all actual private release calls;
- environment telemetry proposal checks and their deltas;
- actual `candidate_plan` calls, with proposal candidates and landing-zone (LZ) pair checks counted separately;
- approvals with rank greater than 3;
- actual suffix checks;
- observed cases where the first three candidates are all energy-blocked versus cases with a mixed full candidate set.

Early-return calls are counted as calls and are not treated as stale telemetry. The report does not infer emergency outcomes for candidates that were not executed, does not turn landing-zone pair checks into additional executed actions, and creates no duplicate request or proposal rows.

## Timing measurements

Using `perf_counter_ns`, the observer records separate distributions for native actor forward/request construction, the actual private executor/controller invocation, and environment-step wall time. Timing runs use the fixed CPU environment with `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`. RNG snapshots are restored around measurement-only observation so the replay sequence remains unchanged.

For each timing category and arm, the output reports count, mean, p50, p95, and maximum, with the denominator named explicitly. Observer and audit overhead are reported separately. Candidate wall time including replay is not relabeled as inference cost. The measurements make no hardware real-time claim and do not create a latency hard gate; measured period fraction is informative only.

## Output and validation

Outputs are written outside the project tree under `/home/user/training_outputs/terminal/p3_followup_20260930/prefix_fallback_profile_20261001`, using the existing artifact style: preflight/progress, one arm JSON per completed tape, and a final full JSON. The schema records source/artifact bindings, cell/arm identity, exact request counts, lifecycle and safety checks, passive counter denominators, timing summaries, RNG restoration, and terminal status. Unknown, censored, failed, or stopped metrics remain unknown and are not filled with zero.

Before the pilot, unit checks validate the input manifest bindings, output schema, counter denominator fields, timing fields, JSON serialization, and the no-action/no-RNG-advance observer contract. The pilot and FULL outputs must identify all 16 tapes and separately report any missing or invalid arm. The previously qualified K=3 remains the formal boundary; this measurement and any existing max25 observation do not supersede it. The result is a measurement pass only; it does not claim that prefix fallback is universally causal, formally qualified, hardware-real-time safe, or sufficient to resolve P3.
