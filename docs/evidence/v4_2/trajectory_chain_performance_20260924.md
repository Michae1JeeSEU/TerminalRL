# Trajectory-chain performance evidence (2026-09-24)

This record covers the fixed-rule, dense30, 100-step performance-chain artifacts. It
is a runtime/performance diagnostic only; no training or PPO update was called.

## Frozen run conditions

All four JSON records report seed `841201`, `30` warm-start UAVs, SOC `.75`,
`continuous_inward_v1`, `reconcile_late_merge_slots_enabled=false`, route guard on,
`100` simulation steps, and `training_called=false`. Each record has
`failed_energy=0`, `safety_error=null`, zero hard-separation violations, zero NMAC,
zero collisions, zero shield fail-closed events, and zero joint safety fail-closed
events. Each has `release_approved=12`, one completed UAV at the 100-second horizon,
and 29 unresolved UAVs. These are 100-step prefix measurements with
`terminal=false` and `horizon_reached=true`; they are incomplete runtime diagnostics,
not full-flow qualification or capacity/zero-failure evidence.

The four measurements use the same scheduler version and the late-slot flag is false.
They were collected before the singleton target-gap behavior fix, so the comparison
isolates the equivalent performance changes and does not measure that later fix.

## Wall-time measurements

| artifact | simulation wall time (s) | semantic outcome |
|---|---:|---|
| baseline | `28.60701879998669` | unchanged |
| optimized | `23.053352100308985` | unchanged |
| baseline r2 | `28.87467119982466` | unchanged |
| optimized r2 | `23.500971000175923` | unchanged |

The optimized run reduced wall time by `19.41%` against baseline; the r2 pair reduced
wall time by `18.61%`. These percentages use the raw
`simulation_wall_time_seconds` values above.

## Normalized semantic comparison

The four JSON objects were compared recursively after normalizing only:

- wall-time fields;
- `progress_path` values;
- UUID-valued fields and UUID-shaped values.

The baseline versus optimized comparison had `0` semantic differences. The baseline r2
versus optimized r2 comparison had `0` semantic differences. The baseline versus
baseline r2 comparison had `0` semantic differences, and optimized versus optimized r2
also had `0` semantic differences. No other fields were normalized.

The exact raw records are:

- [baseline](../../../results/performance_chain_20260924_baseline.json)
- [optimized](../../../results/performance_chain_20260924_optimized.json)
- [baseline r2](../../../results/performance_chain_20260924_baseline_r2.json)
- [optimized r2](../../../results/performance_chain_20260924_optimized_r2.json)

The optimized profiling artifacts are [profile data](../../../results/performance_chain_20260924_optimized_profile.prof)
and [top-20 profile report](../../../results/performance_chain_20260924_optimized_profile_top20.txt).

## Additional reported microbenchmarks

Separate microbenchmarks report a `17.41x` sampling speedup and `30` ring-transfer cases
with no endpoint mismatch. These measurements are recorded separately and are not
substituted for the JSON semantic comparison above.
