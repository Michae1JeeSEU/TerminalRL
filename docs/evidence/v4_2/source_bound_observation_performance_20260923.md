# V4.2 source-bound observation performance evidence

Date: 2026-09-23
Status: observation-only runtime comparison
Environment: local Windows 11, `D:\7_codespace\terminal\venv\Scripts\python.exe`, Python 3.12.4, NumPy 2.4.4
Benchmark utility: `terminal_marl/scripts/benchmark_v4_2_throughput.py`

## Scope and method

This evidence compares the existing V2.1 retry-state observation with the
source-bound V2.3 observation. The simulation configuration was held constant
except for the observation schema and the V2.3 communication range:

```text
action schema: low_action_schema_v4_2_hierarchical
action: [1, 0, 0] -> RELEASE proposal 12
energy model: phase_motion_v2
guard: emergency_priority_guard_enabled=false
reset: warm_start_rings, warm_start_steady_motion=true
timing: 5 untimed warmup steps, then timed environment steps
safety: 80/40/10 m thresholds, shield enabled, 30 s horizon
old observation: phaseA_low_obs_v2_1_retry_state, neighbor_comm_range_m=null
new observation: phaseA_low_obs_v2_3_source_bound_88d, neighbor_comm_range_m=320
```

The runs used the existing local benchmark utility, with no output files,
training, PPO updates, server jobs, or formal qualification markers.

Commands:

```powershell
.\venv\Scripts\python.exe .\terminal_marl\scripts\benchmark_v4_2_throughput.py --label paired12_old_v21 --seeds 840600 840601 --observation-schema phaseA_low_obs_v2_1_retry_state --repeats 3 --warmup-steps 5 --timed-steps 20
.\venv\Scripts\python.exe .\terminal_marl\scripts\benchmark_v4_2_throughput.py --label paired12_new_v23 --seeds 840600 840601 --observation-schema phaseA_low_obs_v2_3_source_bound_88d --neighbor-comm-range-m 320 --repeats 3 --warmup-steps 5 --timed-steps 20
.\venv\Scripts\python.exe .\terminal_marl\scripts\benchmark_v4_2_throughput.py --label paired30_old_v21 --seeds 840602 --warm-start-uavs 30 --observation-schema phaseA_low_obs_v2_1_retry_state --repeats 2 --warmup-steps 5 --timed-steps 15
.\venv\Scripts\python.exe .\terminal_marl\scripts\benchmark_v4_2_throughput.py --label paired30_new_v23 --seeds 840602 --warm-start-uavs 30 --observation-schema phaseA_low_obs_v2_3_source_bound_88d --neighbor-comm-range-m 320 --repeats 2 --warmup-steps 5 --timed-steps 15
```

All four commands exited with code `0`.

## Paired simulation timing

The signed delta is `(V2.3 - V2.1) / V2.1`.

| UAVs / seed | Mean step time V2.1 → V2.3 (s) | Delta | Median (s) | Delta | P95 (s) | Delta | Steps/s V2.1 → V2.3 | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 / 840600 | 0.077278 → 0.080714 | +4.446% | 0.052317 → 0.053981 | +3.180% | 0.133410 → 0.136911 | +2.624% | 12.9402 → 12.3894 | −4.257% |
| 12 / 840601 | 0.073427 → 0.076299 | +3.911% | 0.052479 → 0.055050 | +4.899% | 0.115438 → 0.111973 | −3.001% | 13.6189 → 13.1064 | −3.763% |
| 30 / 840602 | 0.899733 → 0.886568 | −1.463% | 0.602093 → 0.592816 | −1.541% | 1.898781 → 1.857913 | −2.152% | 1.11144 → 1.12795 | +1.485% |

The 12-UAV V2.1 aggregates were 60 timed steps per seed; the 30-UAV
aggregates were 30 timed steps. The paired runs reported zero hard-separation
violations and zero emergency touchdowns.

## Off-screen viewer timing

The benchmark used direct PyVista off-screen frame updates. All frame updates
returned successfully.

| UAVs / seed | Mean update V2.1 → V2.3 (s/frame) | Delta | P95 update V2.1 → V2.3 (s/frame) | Delta |
|---|---:|---:|---:|---:|
| 12 / 840600 | 0.089562 → 0.090552 | +1.106% | 0.090289 → 0.090615 | +0.361% |
| 12 / 840601 | 0.093706 → 0.095149 | +1.541% | 0.094948 → 0.097737 | +2.937% |
| 30 / 840602 | 0.094278 → 0.098071 | +4.023% | 0.096165 → 0.100943 | +4.969% |

## Dense30 cProfile evidence

A separate in-memory cProfile used seed `840602`, the V2.1 dense30
configuration, 5 warmup steps, and 5 profiled steps with the same explicit
RELEASE action. Profiled elapsed wall time was `10.1616726 s`.

Top cumulative entries included:

```text
TerminalParallelEnv.step                 10.1612081 s cumulative
PredictiveSafetyShield.protect            5.0833082 s cumulative
PredictiveSafetyShield._recursive_safe     5.0726529 s cumulative
PredictiveSafetyShield._joint_assessment   4.9736340 s cumulative
PredictiveSafetyShield._backup_safe        4.6964373 s cumulative
SafetyFilter.filter_plan                   3.1438833 s cumulative
_joint_position_tracks                     3.0680418 s cumulative
parallel_env._candidate_plan               2.3867370 s cumulative
```

These entries overlap through the call tree. They are reported as profile
measurements and are not additive component totals.

## Evidence limits

- The viewer measurement is off-screen PyVista only; native on-screen GUI FPS
  and GPU-specific timing were not measured.
- The samples are short and subject to CPU state, interpreter, renderer, and
  run-order noise.
- V2.1 and V2.3 change observation information semantics, so this is a runtime
  comparison and is not a policy-learning comparison.
- The benchmark does not establish performance acceptance, PPO readiness,
  safety qualification, or end-to-end training throughput.
