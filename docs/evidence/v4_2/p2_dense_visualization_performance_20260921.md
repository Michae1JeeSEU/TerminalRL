# P2 dense visualization and simulator performance evidence (2026-09-21)

## Scope

This pass separates renderer cost from canonical simulator cost for the frozen
P2 K3 checkpoint in `p2_dense24`.  It does not change safety thresholds,
prediction horizons, sampling intervals, action semantics, or checkpoint
behavior.

Frozen replay inputs:

```text
checkpoint: p2_gate_repair_v3_anchored_20260919/gate_repair_v3_qualification_candidate.pt
profile: p2_initial_k3
scenario: p2_dense24
seed: 809006
window: first 20 simulation steps for the paired cProfile comparison
```

## Root cause and change

The no-GUI baseline showed that lag was not primarily a PyVista problem.  The
recursive runtime shield deep-copied every active UAV, including immutable
trajectory arrays, and collision review repeatedly constructed full
`TrajectoryPoint` objects when only position was required.

The repair keeps the same authoritative calculations and instead:

- shallow-copies the top-level `UAVState` objects used only by the recursive
  successor projection;
- caches identical committed-position tracks for one prediction step;
- uses exact position-only piecewise-linear interpolation in position-only
  safety checks.

## Paired profile result

| cProfile cumulative item | before | after |
|---|---:|---:|
| total profiled runtime | 19.560 s | 12.977 s |
| 20 calls to `TerminalParallelEnv.step` | 15.726 s | 8.052 s |
| runtime shield `protect` | 9.099 s | 2.507 s |
| recursive feasibility | 7.684 s | 1.636 s |
| successor-state projection | 5.140 s | 0.012 s |
| Python `deepcopy` | 5.107 s | absent from the hot path |
| full `TrajectoryPlan.sample` | 169,397 calls / 2.363 s | 1,458 calls / 0.023 s |
| position-only interpolation | unavailable | 158,849 calls / 0.560 s |

This is a 48.8% reduction in cumulative simulator-step time for the frozen
20-step dense replay.  The result is evidence for dense rollout throughput,
not a claimed end-to-end PPO epoch speedup: policy inference, optimization,
I/O, and other curricula have different cost shares.

The before/after replay produced the same control and safety telemetry,
including `release_approved=12`, `release_rejected=4`,
`safety_interventions=5`, `max_concurrent_final=2`,
`recursive_feasibility_checks=19`, and zero hard-separation violations, NMACs,
collisions, or fail-closed events.  Minimum actual and swept separations also
remained exactly `202.23680909084456` and `202.1616029218398` metres.

An independent 30-step wall-clock sample measured 5.376 s without a viewer and
6.424 s with off-screen PyVista.  Thus rendering adds measurable cost, while
the canonical simulation remains the larger component.  Native on-screen FPS
was not measured.

## Viewer trajectory semantics

The viewer now renders only the still-executable part of each committed plan:
current authoritative position followed by future canonical plan points.
Already executed points are omitted.  Trajectory actors are rebuilt when plan
identity or line topology changes and removed when no trajectory remains.

Each UAV and its trajectory share a stable identity colour across the
lifecycle.  Selecting a UAV makes its trajectory wider and more opaque.  The
P2 checkpoint viewer defaults to one rendered frame per two simulation ticks
and at most 12 points per visible trajectory; CLI flags can tune both without
changing simulation semantics.

An independent 180-step actor-lifecycle check observed at most 16 visible
trajectory actors, observed actor removal at multiple steps, and found all
trajectory actor/mesh/signature containers aligned.  No unbounded historical
trajectory accumulation was observed.

## Verification

```text
56 passed: performance, vehicle dynamics, trajectory parameterization,
           visualization snapshot/integration, P2 viewer
4 passed:  focused A6.8 safety prediction and swept verification
4 passed:  recursive successor/cache/partial-handoff focused checks
```

The long historical A6.8 scenario was not rerun to completion in this pass.

