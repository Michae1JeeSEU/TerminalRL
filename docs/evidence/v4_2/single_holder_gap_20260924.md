# Single-holder merge-gap diagnostic protocol (2026-09-24)

This document preregisters a bounded paired diagnostic for the scheduler case
where exactly one UAV occupies the target ring. The current implementation
places the merge capture at the predicted holder antipode. The legacy variant
temporarily reproduces the historical same-angle result through
`terminal_marl.training.diagnose_single_holder_gap --legacy-single-holder-gap`.
The wrapper restores the authoritative `_gap_center` method after every run.

The diagnostic does not change SafetyFilter thresholds, route geometry,
vehicle limits, action K1, or the late-slot reconciliation setting. Every
candidate still passes the complete energy guard and SafetyFilter chain.

## Fresh paired cells

The fresh seed check found no existing repository references for seeds 841301
and 841302 before this protocol was written. The planned cells are the
Cartesian product of those seeds and the fixed antipode implementation versus
the legacy same-angle reproduction. Late-slot reconciliation is held off in
every cell:

| seed | single-holder gap variant | late-slot reconciliation | output name |
|---:|---|---|---|
| 841301 | antipode fix | off | `seed841301_fixed_off_dense30_full.json` |
| 841301 | legacy same-angle | off | `seed841301_legacy_off_dense30_full.json` |
| 841302 | antipode fix | off | `seed841302_fixed_off_dense30_full.json` |
| 841302 | legacy same-angle | off | `seed841302_legacy_off_dense30_full.json` |

Fixed conditions for all cells:

- `dense_warm_start_uavs=30`, all arrivals at time zero;
- initial SOC range `0.55–0.90`;
- route-aware energy guard enabled;
- `final_approach_geometry_mode=continuous_inward_v1`;
- rule-only K1 action `[1, 0, 0]` for every active UAV;
- `max_steps=1800`, with progress every 100 steps;
- no learned policy, Teacher, or training call.

The only paired implementation factor is the single-holder gap branch. The
late-slot setting is recorded in every output name and remains explicitly off
for all cells.

## Preflight and launch template

Before the full cells, run the targeted merge-gap tests and a one-step pilot
for each variant after the local files are visible on the server through the
configured Syncthing path. Pilot outputs must use unique names and are not
full-cell evidence.

```bash
python3 -m terminal_marl.training.diagnose_single_holder_gap \
  --seed 841301 --initial-soc-min 0.55 --initial-soc-max 0.90 \
  --max-steps 1 --progress-every-steps 1 --dense-warm-start-uavs 30 \
  --final-approach-geometry-mode continuous_inward_v1 \
  --output "$HOME/training_outputs/terminal/single_holder_gap_20260924/seed841301_fixed_pilot.json"

python3 -m terminal_marl.training.diagnose_single_holder_gap \
  --seed 841301 --initial-soc-min 0.55 --initial-soc-max 0.90 \
  --max-steps 1 --progress-every-steps 1 --dense-warm-start-uavs 30 \
  --final-approach-geometry-mode continuous_inward_v1 \
  --legacy-single-holder-gap \
  --output "$HOME/training_outputs/terminal/single_holder_gap_20260924/seed841301_legacy_pilot.json"
```

Full cells use the same command with the corresponding seed, variant flag, and
unique output name from the table, with late-slot reconciliation left off.
Each JSON output and its adjacent progress JSON are retained. No result is
interpreted as a formal qualification or PPO authorization by this protocol.

## Recorded preflight and full-cell outputs

The targeted local preflight passed: `35 passed in 3.72s`. The two server
one-step pilots completed with one step, `failed_energy=0`,
`emergency_committed=0`, `safety_error=null`, and `release_approved=1`.
Their serialized labels were `antipode_fix` with
`legacy_single_holder_gap=false` and `legacy_same_angle` with
`legacy_single_holder_gap=true`; both recorded
`reconcile_late_merge_slots_enabled=false`. Each pilot produced its JSON and
adjacent progress JSON.

The four full cells produced the following raw counts:

| seed | variant | steps | completed | emergency_landed | failed_energy | release_approved | safety_error |
|---:|---|---:|---:|---:|---:|---:|---|
| 841301 | antipode fix | 1364 | 18 | 8 | 4 | 51 | `null` |
| 841301 | legacy same-angle | 1201 | 12 | 13 | 5 | 37 | `null` |
| 841302 | antipode fix | 1090 | 14 | 9 | 7 | 40 | `null` |
| 841302 | legacy same-angle | 1023 | 14 | 10 | 6 | 41 | `null` |

All four full outputs recorded
`reconcile_late_merge_slots_enabled=false`, and their serialized variant
labels matched the output names. The JSON and adjacent progress artifacts are
under
`$HOME/training_outputs/terminal/single_holder_gap_20260924/` on the server,
with names from the preregistered table.

Across all four full outputs, `safety_error` was `null`,
`hard_separation_violations`, `nmac`, `collisions`, `shield_fail_closed`,
`joint_safety_fail_closed`, and `operational_buffer_entries` were all `0`,
`unresolved` was `0`, and `accounting_identity` was `true`. The fixed variant
also used more terminal steps than legacy in both paired cells: `1364` versus
`1201` for seed `841301`, and `1090` versus `1023` for seed `841302`.
Therefore the higher fixed normal count is descriptive paired evidence and
does not by itself establish higher overall throughput.

## Required recorded fields

Record terminal status and step count, requested/active/resolved/unresolved
counts, normal/emergency/failed-energy outcomes, release approvals, primary
and secondary rejection reasons, minimum actual and swept separation, route
checks and negative route checks, safety error, and the complete raw telemetry.
Report late-slot reconciliation and the selected single-holder variant from
the output name and serialized variant fields separately.
