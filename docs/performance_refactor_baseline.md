# P0 Simulation Performance Refactor — Baseline Audit

Evidence type: newly run, non-training local `cProfile` baseline on the
current D7-qualified dynamics implementation.  C0 and PPO are not started.

## Audit status

The required P-A through P-H matrix is being collected before behavioural
optimisations are applied.  Results are appended per completed scenario so
that each subsequent P0 phase has an auditable before/after reference.

## P-A: holder-only, N=60

Command: `python -m cProfile -o pa_holder_n60.prof -m
terminal_marl.training.dynamics_qualification --uavs 60 --physics-dts 0.2
--simulated-seconds 60`.

| Metric | Baseline |
|---|---:|
| Simulated duration | 60 s |
| Wall time | 15.07 s |
| Environment steps/s | 3.98 |
| Peak RSS | not sampled locally |
| Hard separation / NMAC / collision | 0 / 0 / 0 |

Top cumulative costs (15.28 s profiled):

| Function group | Cumulative time (s) |
|---|---:|
| Observation V2 total | 11.15 |
| `topology_neighbors._slot` | 8.92 |
| `nearest_predicted_conflict` | 8.30 |
| Runtime separation monitor | 2.41 |
| Ring-holder execution | 1.75 |
| `VehicleDynamics.ring_motion_step` | 1.57 |

Interpretation: the dominant holder-only cost is repeated topology/CPA work
per observed UAV, not a relaxation of the EH216-S finite-acceleration model.
P0-J/K is therefore a required performance gate; its cached implementation
must be numerically compared with the present observation definition.
