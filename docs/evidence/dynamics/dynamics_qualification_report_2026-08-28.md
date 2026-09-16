# D7 Dynamics Qualification Report — 2026-08-28

Evidence type: newly run, non-training deterministic qualification.  No PPO
optimisation or policy checkpoint update was invoked.

## Core gates

Server command group: dynamics qualification, vehicle dynamics, trajectory
time parameterisation, Phase-A landing, scheduler repair, controllability and
reconfiguration safety.  Result: **35 passed, 1 deselected in 12.72 s**.

The deselected test is the historical A6.8 t=720 reconfiguration replay.  It
is a separate long-horizon reconfiguration-compatibility audit and does not
belong in the base D7 dynamics unit gate.  It remains a required follow-up
before treating dynamic-airspace research results as qualified.

`tests/test_dynamics_qualification.py` verifies:

- 0.2 s versus 0.1 s finite-acceleration ring-holder convergence;
- checked trajectory hash equals every executed physics sample, with zero
  content-hash mismatches;
- holder prediction integration scales linearly in ordered safety samples.

## 3600-second performance benchmark

Output directory: `/home/user/training_outputs/terminal/dynamics_d7_20260828`.
The benchmark fixes warm-start traffic in holders to isolate execution and
separation-monitor cost; it is not a policy evaluation.

| UAVs | physics dt (s) | wall time (s) | env steps/s | collisions / NMAC / hard separation |
|---:|---:|---:|---:|---:|
| 10 | 0.2 | 11.41 | 315.53 | 0 / 0 / 0 |
| 10 | 0.1 | 12.86 | 279.90 | 0 / 0 / 0 |
| 30 | 0.2 | 68.21 | 52.78 | 0 / 0 / 0 |
| 30 | 0.1 | 72.39 | 49.73 | 0 / 0 / 0 |
| 60 | 0.2 | 216.62 | 16.62 | 0 / 0 / 0 |
| 60 | 0.1 | 224.81 | 16.01 | 0 / 0 / 0 |

Decision: retain and freeze `physics_dt=0.2 s` for the current
`performance_constrained_v2` simulation schema.  It passes the convergence
gate and is faster at every measured scale.  The benchmark confirms that
finite-acceleration execution is feasible for the planned static low-level
curriculum; dynamic-airspace long-horizon compatibility is tracked separately.

## Dynamic-airspace compatibility closure

The separate historical A6.8 t=720 reconfiguration replay completed with
observable, line-buffered output after the D7 performance fixes:

| Check | Result |
|---|---|
| Fixture | `test_a68_historical_t720_486_735_geometry_fixture_is_hard_safe` |
| Result | **1 passed** |
| Wall time | 660.91 s (11 min) |
| Peak RSS | 638,700 KB |
| Exit code | 0 |

The replay covers a 300-second branch from the deterministic t=720 high-demand
snapshot, including the former TRANSFER-holder conflict window.  The branch
recorded zero new hard-separation violations.  The audit now scopes
`first_conflict` to events generated after the snapshot; prefix diagnostics
remain separately exposed as `preexisting_first_conflict` and cannot be
misreported as a branch failure.

## D7 decision

**PASS.** The D7 dynamics qualification is closed: the base gates, dt
convergence, fixed-rule performance sweep and long-horizon dynamic-airspace
compatibility regression all passed.  C0 capacity calibration is unblocked.
This decision does **not** authorize low-level PPO; PPO remains gated by the
subsequent C0 calibration and curriculum requirements.
