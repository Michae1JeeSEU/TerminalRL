# Performance-Constrained Vehicle Dynamics Specification

Implemented facts as of D6:

- Vehicle: `research_evtol` / EHang EH216-S / `eh216s_research_v1`.
- State semantics: position, 3-D velocity, horizontal `speed`, acceleration, yaw and yaw-rate diagnostics.
- Operational horizontal speed is 25.0 m/s; 29.17 m/s is V_NE and 36.11 m/s is metadata only.
- Translational constraints: 1.5 m/s² horizontal accel/decel/lateral, 4/6 m/s climb/descent, and 1.0 m/s² vertical acceleration.
- Ring feasibility uses the combined tangential/centripetal constraint `sqrt(a_t^2 + (v^2/r)^2) <= 1.5`.
- A `GeometricPath` is converted by `TrajectoryTimeParameterizer` to a single authoritative executable `TrajectoryPlan` with position, velocity, acceleration, timestamps, profile identity and trajectory hash.
- The parameterizer applies curvature, horizontal acceleration, climb/descent and end-speed constraints through forward/backward speed passes.
- The environment executes an approved plan at `physics_dt=0.2 s`, propagating position, velocity, acceleration and yaw-rate diagnostics.  Ring holders use the same finite-acceleration integrator.
- `SafetyFilter`, runtime shielding and dynamic-reconfiguration prediction all use `VehicleDynamics`; safety sampling is `0.2 s` and execution is rejected unless the checked content hash still matches.
- Checkpoints now bind the vehicle-dynamics and trajectory schema versions plus profile identifier/version.

D7 core dynamics qualification passed on 2026-08-28; see
`docs/dynamics_qualification_report_2026-08-28.md`.  `physics_dt=0.2 s` is
frozen for `performance_constrained_v2`.  The separate long-horizon A6.8
dynamic-airspace replay remains required before C0 and low-level PPO.

P0 performance refactor (2026-08-28) preserves these dynamics limits.  It
adds explicit geometric-sampling configuration (`trajectory_bezier_samples`,
`trajectory_final_samples`) for convergence qualification; neither value is a
vehicle-performance parameter.  Safety-sampling convergence remains an open
P0 gate, so the frozen production value is still `safety_sampling_dt=0.2 s`.
