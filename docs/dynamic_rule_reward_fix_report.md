# Dynamic-rule landing pipeline and reward sanity fix report

This round addressed the server pilot failure mode where traffic was present but
`dynamic_rule` / `dynamic_hrl` produced `landed_count=0`.  The work remained
simulation-only: no ROS2, Crazyswarm2, Crazyflie hardware, or PPO pilot training
was started.

## Root cause

Baseline diagnostics showed that the fixed rule pipeline could land aircraft,
while the dynamic-rule high-level policy repeatedly expanded/shifted airspace,
reduced inner-ring speed, and increased headways.  This combination starved
`inner -> final` and `final -> land` transitions even though the low-level rule
baseline and VTAC/SafetyFilter path were functional.

Metrics also had two reward-scaling issues:

- no-pair scenes reported `min_pairwise_distance=0`, which looked like a severe
  conflict;
- raw conflict/waiting counts were used without enough normalization, making
  episode length and backlog size dominate reward diagnostics.

## Code changes

1. `DynamicRuleHighLevelPolicy`
   - uses normalized pairwise conflict rate instead of a boolean conflict flag;
   - prioritizes throughput-friendly speed/headway changes over continuous ring
     expansion;
   - applies small nominal recovery toward the configured airspace;
   - avoids increasing final headway when there is backlog and final is empty;
   - records `last_info` diagnostics for trace files and tests.

2. `TerminalParallelEnv.metrics()`
   - reports `has_pair`, `current_pair_count`, `pair_opportunities`,
     `raw_pair_conflict_samples`, `unique_conflict_events`,
     `conflict_vehicle_seconds`, and `normalized_conflict_rate`;
   - keeps `min_pairwise_distance=inf` for no-pair scenes instead of converting
     it to zero;
   - tracks approach/ring/final waiting and normalized congestion pressure.

3. `HighLevelReward`
   - adds phase-progress dense reward;
   - normalizes waiting by interval duration and `max_active_uavs`;
   - normalizes conflict by pair opportunities;
   - keeps raw and normalized diagnostics separate.

4. Diagnostics and calibration
   - `debug_phase_transitions.py` now records low-level-step approvals,
     `dynamic_rule_trace.csv`, and `phase_release_summary.csv`;
   - action isolation modes include radius-only, height-only, speed-only,
     headway-only, geometry-only, no-headway, no-speed, and nominal-recovery-only;
   - `reward_diagnostics.py` outputs mean/std/mean_abs/p50/p90/p99/max_abs,
     component share, and a 20x scaling warning;
   - `capacity_calibration.py` performs rule-only arrival-interval calibration
     without training.

## Server verification summary

Output root:

```text
/home/user/training_outputs/terminal/dynamic_rule_reward_fix/
```

Key server dry-run results under `iteration_01`:

| case | spawned | landed | active_end | normalized_conflict_rate | min_pairwise_distance | safety_override_count |
|---|---:|---:|---:|---:|---:|---:|
| single_rule | 1 | 1 | 0 | 0.0000 | inf | 0 |
| six_rule | 6 | 6 | 0 | 0.0676 | 41.506 | 0 |
| six_dynamic | 6 | 6 | 0 | 0.0674 | 41.506 | 0 |
| medium_rule_50hl | 52 | 12 | 40 | 0.0520 | 0.286 | n/a |
| medium_dynamic_50hl | 51 | 11 | 40 | 0.0514 | 0.286 | n/a |

Local full test suite:

```text
237 passed, 30 warnings
```

Server targeted tests:

```text
15 passed
```

## Readiness conclusion

The dynamic-rule baseline is no longer stuck at zero landings in the no-arrival
landing-pipeline check and recovers to 100% of fixed-rule landed count in the
six-UAV scenario.  Medium-demand dry-runs also produce non-zero landed counts
and non-zero waiting/congestion diagnostics.

This is sufficient for a small PPO pilot readiness check, but the next run
should still start with short smoke/pilot stages and inspect reward diagnostics
before launching any long sweep.
