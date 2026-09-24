# Random-SOC Arbitration Repair V1 Development Result

Protocol: [random-SOC arbitration repair V1](random_soc_arbitration_repair_v1_20260925.md)

This page records a development comparison of two arms using the same frozen
repair checkpoint. It is not a P3 qualification, model-promotion, or PPO
admission result. `P3_QUALIFIED=false`, `PPO_ALLOWED=false`, and
`P4_ALLOWED=false` remain unchanged.

## Binding and smoke

Both arms used repair checkpoint SHA256
`e04e5b4a8f8365b52a41f66c96ad1f66a09a554ea64118fcab8040d64175d353`.
The baseline used `downstream_priority`; the variant used
`downstream_soc_priority`. The smoke `847000` completed with exit `0`, with
both arms binding the complete configuration for `2` steps. The local/server
targeted test report contained `19` passed tests. Runner commit: `464d435`;
protocol commit: `a683edb`.

## Paired dense24 development runs

The table reports direct `Counter(summary["outcomes"].values())` values in
normal / emergency / failed_energy / unresolved order. Lifecycle values were
computed with `p3._lifecycle_aggregate` from the raw arm records.

| pair / arm | steps | outcomes | priority-weighted wait | low-SOC mean wait (count) |
|---|---:|---:|---:|---:|
| 847101 baseline | 867 | 19 / 5 / 0 / 0 | 156.3371109062 | 145.1880824575 (6) |
| 847101 soc_priority | 834 | 19 / 5 / 0 / 0 | 191.2342229423 | 115.7573166695 (6) |
| 847102 baseline | 821 | 22 / 2 / 0 / 0 | 150.7183596642 | 128.6053749990 (6) |
| 847102 soc_priority | 772 | 22 / 2 / 0 / 0 | 154.7539523277 | 94.4285529652 (6) |

Each pair therefore contains `24` requests. Across both pairs, each arm has
`48` requests and `41 normal + 7 emergency + 0 failed_energy + 0 unresolved`.
The derived aggregate is
`/home/user/training_outputs/terminal/random_soc_arbitration_repair_v1_20260925/development_pair_aggregate.json`.
Its overall lifecycle values are baseline priority-weighted wait
`153.6317862341`, low-SOC mean wait `136.8967287283` (count `12`), and
`soc_priority` priority-weighted wait `173.6696482019`, low-SOC mean wait
`105.0929348174` (count `12`).

In both pairs, `soc_priority` reduced low-SOC mean wait while increasing
priority-weighted wait. The recomputed selection gates are
`metrics_defined=true`, `initial_complete=true`, `candidate_complete=true`,
`low_soc_wait_nonworse=true`, and `priority_wait_strictly_better=false`.

Both pairs completed with no software or `SafetyInvariantError`; terminal,
replay, geometry, requested-likelihood, K3 shortlist, safety, no-Teacher,
no-labels, no-updates, and weights-unchanged checks passed. The only false arm
gate was `no_failed_outcomes=false`, caused by the recorded emergency outcomes.
All raw outcomes are retained.

## Decision boundary and next reference

The arbitration result does not support adopting `downstream_soc_priority`;
the default remains `downstream_priority`. Both arms were repair candidates,
not the historical canonical initial. The repair source is the `714d...`
checkpoint lineage; the historical P3 canonical initial is the separate
`4b8be66c...` checkpoint. The current next step is the registered
[canonical reference protocol](random_soc_canonical_reference_v1_20260925.md),
with implementation and preflight pending before any `848xxx` reset. Historical
held-out seeds `829001–829006` remain untouched.
