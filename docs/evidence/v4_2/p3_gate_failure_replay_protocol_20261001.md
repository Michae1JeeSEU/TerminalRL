# P3 gate failure passive replay protocol (2026-10-01)

Status: preregistered, passive replay only. This protocol does not train, update a
checkpoint, consume held-out data, or admit PPO (`PPO_ALLOWED=false`).

## Registered inputs

- Pair 15, environment seed `877104`, keyed policy seed `9303103`.
- Target request: arrival `21`; trace context also includes arrival `13`.
  `--include-arrival13` is a fixed protocol flag; it adds no episode or replay.
- Source candidate and warmup bindings are the registered `3fe7` and `a82`
  artifacts used by the completed gate-control run. The live source, config, and
  runtime bindings must match the saved preflight before a replay starts.
- Dense24 neutral config, Observation V2.1 `phaseA_low_obs_v2_1_retry_state`
  (88D), action schema `low_action_schema_v4_2_hierarchical`, K=3, and the
  registered 1800-step simulation cap.
- Reference arms are the pair-15 legacy and always-RELEASE control JSON files
  under `p3_followup_20260930/forced_wait_frozen` and
  `p3_followup_20260930/gate_release_control`.

## Procedure and scope

`preflight` creates no environment. It binds the current dense24 config, source
digests, code/runtime metadata, registered arm hashes, and the replay runner
hash. `check` repeats those bindings and refuses changed artifacts.

`run` executes each registered arm once through the existing frozen runner. The
legacy arm uses the existing keyed request path; the control arm uses the
existing gate-control wrapper. The only added behavior is a temporary observer
around `paired._call_step` and a temporary `_try_emergency_landing` subclass
wrapper. All hooks are restored in `finally` blocks.

The observer records pre/post tick state for arrivals 21 and 13, including phase,
position, SOC and energy, retry times, committed/landing flags, action and
decoded V4.2 gate/proposal/joint index, proposal shortlist, release decisions,
release-intent telemetry, last safety decision, energy reachability diagnostics,
LZ/resource state, airspace state, and emergency telemetry. The emergency hook
records each native target call, force flag, return value, before/after target
state, telemetry delta, and LZ snapshots. Native blocker details are retained as
`null` when the executor does not expose them; no extra feasibility probe is run.

## Exact replay checks

The replay must retain all generated requests and terminal rows, including the
registered target `failed_energy` row. Request identity, request groups, tick
count, outcome counts, model-unchanged and caller-RNG-unchanged flags, and hard
safety status are exact checks. Terminal rows and scalar accounting use
`abs_tol=1e-8`, `rel_tol=1e-10`; a mismatch stops the phase and preserves the
trace and failure report. The trace must contain exactly one record per runner
tick and observe target arrival 21 at least once. The output is `REPLAY_OK` only
after both arms pass these checks.

Telemetry deltas skip unchanged nonfinite counters, retain finite emergency
counter differences, and encode changed nonfinite values as unknown (`null`). A
bounded native one-tick fixture v3 passed native-arm equivalence, model/RNG and
hard-safety checks, and JSON serialization using the production sanitizer.
The recorded fixture is `.tmp/p3_gate_failure_replay_native_fixture_v3.json`.

## Output and stopping rules

The default output root is
`/home/user/training_outputs/terminal/p3_followup_20260930/gate_failure_replay_20261001`.
It contains `preflight.json`, `progress.json`, one replay arm and trace per mode,
and `failure_replay.json`. Existing files are never overwritten.

Any source/config/runtime binding change, missing identity or target terminal row,
model/RNG change, hard-safety violation, hook exception, serialization failure,
or replay mismatch produces a failed status with the raw error and retained
partial trace. No resampling, seed change, cap change, intervention selection,
optimizer, teacher, PPO, or held-out evaluation is permitted.
