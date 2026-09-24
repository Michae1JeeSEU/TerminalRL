# Trained Dense30 V2 Diagnostic Result

Development-validation record for the preregistered
[Trained Dense30 V2 Diagnostic Protocol](trained_dense30_v2_diagnostic_protocol_20260924.md).
This file records development checks and current execution state only. It is
not a qualification result, does not mark the protocol PASS, and does not
authorize PPO.

## Status at record time

- Development validation: completed for the observations listed below.
- Server preflight: confirmed; 12 local/server file hashes matched and the
  server test set reported `7 passed` in `1.34 s`.
- Registered smoke `842000`: completed once with exit status `0`.
- Registered diagnostic seed `842101`: completed at terminal state; this is the
  first bounded diagnostic result, not a PASS or qualification result.
- Registered diagnostic seed `842102`: untouched and paused for energy-failure
  review.
- `PPO_ALLOWED=false`.
- No trained weights were changed and no optimizer update was performed.

## Directly observed development checks

The latest local focused check set completed with `7 passed`. The actual checkpoint
load matched the preregistered SHA-256
`714d141f7b472a17321e95e9c4c5574fdc3bcaa0c2911494f67b5b1a9f27592e`; the
loaded weights were unchanged and no optimizer was invoked.

An unplanned local development API/CLI smoke used development seed `1` with a
maximum of `5` steps. It was executed three times, with outputs preserved at:

- `.tmp_trained_dense30_v2_smoke`
- `.tmp_trained_dense30_v2_smoke2`
- `.tmp_trained_dense30_v2_smoke3`

All three observed no safety error. The final development record contained
`150` trace rows and `146` actor decisions: `133 WAIT` and `13 RELEASE`. The
loaded weights were unchanged. A stub/test-only seed `3` was separate from the
registered seeds; no aggregate reset count is asserted here.

These development calls are validation evidence only. They are outside the
registered cohort and must not be substituted for smoke `842000` or diagnostic
seeds `842101/842102`, selected for tuning, or treated as registered
diagnostic evidence.
The output directories are preserved; no deletion or overwrite was performed.

## Confirmed server smoke

The server base-9 source hashes matched. The configured interpreter is:

```text
/home/user/envs/terminal_env/bin/python
```

Source identifiers recorded for this preflight are:

- implementation commit: `9c1d468`; no remote configured;
- diagnostic module SHA-256:
  `6f2a96826b27688d66ef7018a87dfdab0439d35ac849f85bad482d439a4e266c`;
- protocol SHA-256:
  `7754517ad579c0de7530687a9e6411e4ad3d67daf80ce3bbe8b950d6de679aba`.

Smoke `842000` ran once under
`/home/user/training_outputs/terminal/trained_dense30_v2_diagnostic_20260924/smoke`
and exited `0`; its log was
`/home/user/training_outputs/terminal/trained_dense30_v2_diagnostic_20260924/smoke.log`.
The smoke log recorded `PILOT_EXIT_STATUS=0`; the run produced `5` steps and
`150` trace rows. Actor decisions were `146` total: `134 WAIT` and `12 RELEASE`;
there were `11` vetoes/rejections and `1` approval. `30` UAVs remained unresolved, as expected for the
prefix smoke. Minimum actual and swept separations were
`202.3429416475199 m` and `202.2374841615668 m`. Safety counters were `0` and
the safety error was `null`. Weights were unchanged, with no Teacher and no
optimizer.

A read-only validator initially compared the initialization digest against the
loaded digest and failed. It was corrected to compare rollout before/after
digests, which passed. The smoke pilot was not rerun after this correction.

The development and smoke records do not qualify the actor or authorize PPO.

## Confirmed bounded diagnostic-1: seed 842101

The run reached terminal state after `1173` simulated steps, with wall time
`64.53536491142586 s`. Outcomes were `8 normal`, `18 emergency`, `4
failed_energy`, and `0 unresolved`. Actor decisions were `14877 WAIT` and
`1055 RELEASE`; `24` RELEASEs were approved and `1031` were rejected.

Minimum actual and swept separations were
`80.00345665736643 m` and `80.0 m`. `safety_error=null`; no geometric safety
event was observed: collision, NMAC, hard-separation, shield fail-closed,
joint-safety fail-closed, and executed-plan hash-mismatch counters were `0`.
The run recorded `9` shield interventions,
which are corrections rather than failures, and `1315` total safety
interventions. The named incident counters are therefore distinct from the
nonzero intervention/veto counters; this record does not claim that all safety
counters were zero. Weights were unchanged, with no Teacher and no optimizer.

The runtime heartbeat was deleted after the terminal state. Read-only trace
aggregation is complete; the source artifact is
[trace_aggregation.json](../../../../training_outputs/terminal/trained_dense30_v2_diagnostic_20260924/seed842101/trace_aggregation.json).
For the four failed-energy identities:

| UID | phase | actor RELEASE rows | approved | RELEASE rows before first SOC ≤ .40 | WAIT rows before threshold |
|---:|---|---:|---:|---:|---:|
| 7 | outer | 56 | 0 | 3 (all rejected) | 411 |
| 10 | outer | 39 | 0 | 3 (all rejected) | 305 |
| 12 | outer | 54 | 0 | 3 (all rejected) | 493 |
| 22 | middle | 40 | 0 | 9 (all rejected) | 238 |

The trace therefore shows both sparse early RELEASE requests and execution
vetoes. It contains no independent feasible RELEASE witness for the WAIT
states, so it cannot label all WAIT decisions wrong or approve an
imitation-only fix. The main observed inner-ring RELEASE `primary_blocker` category was
`LZ_RESOURCE` in `593/615` rows: `510` `emergency_lz_priority` and `83`
`lz_unavailable`, with `8` approvals. `PREDICTED_SEPARATION` is a
runtime-reported category, not proof of every exact underlying cause.

The authorized next step is to resolve feasible-action and energy-fallback
evidence for the neural execution path before deciding targeted imitation or
Safe DAgger. Targeted imitation repair is deferred until feasible labels and
evidence exist; this is a decision boundary, not a safety-solver diagnosis.
There is no PPO, safety-threshold lowering, second-seed replay/reset, or rule
rerun. No `842101` runner remains. Seed `842102` remains untouched and paused
for energy-failure review; its output directory is absent. Existing outputs are
immutable, and the global loader/core remain unchanged.
