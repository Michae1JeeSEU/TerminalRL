# Density efficiency exploratory protocol (2026-09-24)

This document freezes a bounded exploratory density sweep for the current fixed-rule
V4.2 runtime. It is diagnostic evidence only: no training, PPO update, policy tuning,
SafetyFilter threshold change, or source-code change is authorized by this protocol.

## Predetermined cells

The six cells are the Cartesian product of seeds `841001`, `841002` and dense warm-start
cohorts of `12`, `20`, and `30` UAVs:

| cell | seed | dense warm-start UAVs | status |
|---|---:|---:|---|
| A | 841001 | 12 | planned |
| B | 841001 | 20 | planned |
| C | 841001 | 30 | planned |
| D | 841002 | 12 | planned |
| E | 841002 | 20 | planned |
| F | 841002 | 30 | planned |

The fixed conditions for every cell are:

- rule action K1: `[1, 0, 0]` for every active UAV;
- `initial_soc_min = initial_soc_max = 0.75`;
- `route_guard_enabled = true`;
- `final_approach_geometry_mode = continuous_inward_v1`;
- `release_arbitration_mode = downstream_priority`;
- `max_steps = 1800`, `progress_every_steps = 100`;
- `max_simulation_seconds = 1800`, with no learned policy or training call.

The runner is `terminal_marl/training/diagnose_energy_reachability_guard_smoke.py`,
using `build_config(...)` and `run_smoke(...)`. Results and progress files must be
written through `ssh lab-server` under:

`$HOME/training_outputs/terminal/density_efficiency_20260924/`

with one unique JSON output path per cell. The one-step pilot is a preflight and is not
one of the six cells.

## Required recorded fields

Each cell records terminal status, terminal steps and wall time; requested, active,
resolved, normal, emergency, and failed-energy counts; release approvals and rejection
primary/secondary reason counters; route checks and negative route checks; minimum actual
and swept separation; hard-separation, NMAC, collision, and safety-event counts; and the
raw progress JSON path.

## Hard gates and interpretation

The hard gates for a complete cell are `failed_energy == 0`, `safety_error=null`,
`hard_separation_violations=0`, `nmac=0`, `collisions=0`, `shield_fail_closed=0`,
and `joint_safety_fail_closed=0`. Rejected-candidate `safety_interventions` and
operational-buffer counters are recorded separately and are not substituted for these
hard-gate fields. A run that stops before all requested UAVs resolve is recorded as
incomplete and cannot be marked pass, regardless of its partial counters. The sweep is
not a qualification result and cannot authorize PPO.

## Launch command template

```bash
python -m terminal_marl.training.diagnose_energy_reachability_guard_smoke \
  --seed <SEED> --guard-on --initial-soc-min 0.75 --initial-soc-max 0.75 \
  --max-steps 1800 --progress-every-steps 100 \
  --dense-warm-start-uavs <N> \
  --final-approach-geometry-mode continuous_inward_v1 \
  --output "$HOME/training_outputs/terminal/density_efficiency_20260924/seed<SEED>_dense<N>_full.json"
```

## Baseline results

The six baseline cells were run with `reconcile_late_merge_slots_enabled=false`
(the default baseline; no late-slot reconciliation was enabled). All six reached a
terminal state before the 1800-step horizon. The outcome triplet is
`normal / emergency / failed-energy`.

| cell | seed × dense UAVs | terminal steps / sim s / wall s | outcome triplet | release approvals | route checks / negative | emergency attempts / commits | min actual / swept separation (m) | hard sep / NMAC / collision / shield fail-closed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | 841001 × 12 | 364 / 364 / 4.6848 | 12 / 0 / 0 | 28 | 112 / 0 | 0 / 0 | 81.5020 / 81.4196 | 0 / 0 / 0 / 0 |
| B | 841001 × 20 | 726 / 726 / 17.8745 | 18 / 2 / 0 | 44 | 420 / 0 | 3 / 2 | 81.9246 / 81.9148 | 0 / 0 / 0 / 0 |
| C | 841001 × 30 | 817 / 817 / 39.5079 | 24 / 6 / 0 | 60 | 869 / 12 | 72 / 6 | 80.2622 / 80.1862 | 0 / 0 / 0 / 0 |
| D | 841002 × 12 | 389 / 389 / 5.1513 | 12 / 0 / 0 | 28 | 135 / 0 | 0 / 0 | 88.0353 / 87.9935 | 0 / 0 / 0 / 0 |
| E | 841002 × 20 | 628 / 628 / 17.6651 | 19 / 1 / 0 | 44 | 424 / 7 | 7 / 1 | 85.6505 / 85.6488 | 0 / 0 / 0 / 0 |
| F | 841002 × 30 | 819 / 819 / 38.6089 | 22 / 8 / 0 | 56 | 836 / 11 | 51 / 8 | 80.0280 / 80.0280 | 0 / 0 / 0 / 0 |

All six records have `failed_energy=0`, `safety_error=null`, `nmac=0`,
`collisions=0`, `hard_separation_violations=0`, `shield_fail_closed=0`, and
`joint_safety_fail_closed=0`. Operational-buffer `entries/takeovers` are `0/0` in
every cell. The raw `lz_idle_with_inner_demand_steps` values were A `91`, B `151`,
C `215`, D `165`, E `214`, and F `270`; these are occupancy observations and are
not individually certified missed opportunities.
The raw `safety_interventions` counters were A `0`, B `4`, C `16`, D `0`, E `5`,
and F `46`; these are rejected-candidate operational counters recorded without
interpretation. `horizon_reached=false` for all six records. The raw primary and
secondary release counters are:

| cell | primary blocker counters | secondary constraint counters |
|---|---|---|
| A | `FINAL_RESOURCE=21, HEADWAY=12, LZ_RESOURCE=5, NO_MERGE_GAP=13, PREDICTED_SEPARATION=551` | `dynamics_unreachable_merge_time=20, merge_slot_retime_conflict=19, predicted_holder_reachable_tube=10, predicted_holder_separation=3, predicted_separation=32` |
| B | `FINAL_RESOURCE=59, HEADWAY=15, NO_MERGE_GAP=93, PREDICTED_SEPARATION=3057` | `dynamics_unreachable_merge_time=151, merge_slot_retime_conflict=24, predicted_holder_reachable_tube=83, predicted_holder_separation=9, predicted_post_plan_holder_reachable_tube=3, predicted_post_plan_holder_separation=1, predicted_separation=95, predicted_swept_separation=2` |
| C | `FINAL_RESOURCE=231, HEADWAY=30, LZ_RESOURCE=269, NO_MERGE_GAP=183, PREDICTED_SEPARATION=6596, SOC=12` | `dynamics_unreachable_merge_time=394, emergency_energy_guard=12, merge_slot_retime_conflict=33, predicted_holder_reachable_tube=151, predicted_holder_separation=16, predicted_post_plan_holder_reachable_tube=8, predicted_post_plan_holder_separation=6, predicted_separation=178` |
| D | `FINAL_RESOURCE=28, HEADWAY=14, NO_MERGE_GAP=8, PREDICTED_SEPARATION=728` | `dynamics_unreachable_merge_time=34, merge_slot_retime_conflict=22, predicted_holder_reachable_tube=8, predicted_holder_separation=1, predicted_separation=42` |
| E | `FINAL_RESOURCE=54, HEADWAY=21, LZ_RESOURCE=8, NO_MERGE_GAP=51, PREDICTED_SEPARATION=3030, SOC=6` | `dynamics_unreachable_merge_time=203, emergency_energy_guard=6, merge_slot_retime_conflict=32, predicted_holder_reachable_tube=46, predicted_holder_separation=6, predicted_post_plan_holder_reachable_tube=3, predicted_post_plan_holder_separation=1, predicted_separation=77` |
| F | `FINAL_RESOURCE=142, HEADWAY=46, LZ_RESOURCE=263, NO_MERGE_GAP=207, PREDICTED_SEPARATION=6383, SOC=8` | `dynamics_unreachable_merge_time=411, emergency_energy_guard=8, merge_slot_retime_conflict=52, predicted_holder_reachable_tube=144, predicted_holder_separation=12, predicted_post_plan_holder_reachable_tube=10, predicted_post_plan_holder_separation=6, predicted_separation=126, predicted_swept_separation=2` |

The six raw records and their progress files are:

- [seed841001 dense12 JSON](../../../training_outputs/terminal/density_efficiency_20260924/seed841001_dense12_full.json) and [progress](../../../training_outputs/terminal/density_efficiency_20260924/seed841001_dense12_full.json.progress.json)
- [seed841001 dense20 JSON](../../../training_outputs/terminal/density_efficiency_20260924/seed841001_dense20_full.json) and [progress](../../../training_outputs/terminal/density_efficiency_20260924/seed841001_dense20_full.json.progress.json)
- [seed841001 dense30 JSON](../../../training_outputs/terminal/density_efficiency_20260924/seed841001_dense30_full.json) and [progress](../../../training_outputs/terminal/density_efficiency_20260924/seed841001_dense30_full.json.progress.json)
- [seed841002 dense12 JSON](../../../training_outputs/terminal/density_efficiency_20260924/seed841002_dense12_full.json) and [progress](../../../training_outputs/terminal/density_efficiency_20260924/seed841002_dense12_full.json.progress.json)
- [seed841002 dense20 JSON](../../../training_outputs/terminal/density_efficiency_20260924/seed841002_dense20_full.json) and [progress](../../../training_outputs/terminal/density_efficiency_20260924/seed841002_dense20_full.json.progress.json)
- [seed841002 dense30 JSON](../../../training_outputs/terminal/density_efficiency_20260924/seed841002_dense30_full.json) and [progress](../../../training_outputs/terminal/density_efficiency_20260924/seed841002_dense30_full.json.progress.json)

These are exploratory fixed-rule records only. They contain no training call and do
not authorize PPO or change the existing safety thresholds.
