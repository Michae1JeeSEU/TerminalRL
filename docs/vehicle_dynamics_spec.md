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
- `SafetyFilter`, runtime shielding and dynamic-reconfiguration prediction all use `VehicleDynamics`; execution is rejected unless the checked content hash still matches.
- Checkpoints now bind the vehicle-dynamics and trajectory schema versions plus profile identifier/version.

D7 core dynamics qualification passed on 2026-08-28; see
`docs/dynamics_qualification_report_2026-08-28.md`.  `physics_dt=0.2 s` is
frozen for `performance_constrained_v2`.  The separate long-horizon A6.8
dynamic-airspace replay remains required before C0 and low-level PPO.

P0 performance qualification passed on 2026-08-29 without changing these
dynamics limits. `physics_dt=0.2 s` remains frozen. Deterministic fixed,
mixed, shield-risk and 400-s saturated witnesses produced equivalent
SafetyFilter/shield decisions and identical safety outcomes at
`safety_sampling_dt=0.2/0.5/1.0 s`; the saturated minimum actual/swept
separations were 87.508/87.063 m with zero hard violation, NMAC or collision
at all three values. Production therefore uses the coarsest converged value,
`safety_sampling_dt=1.0 s`, while retaining continuous swept-segment checks.
Production geometric sampling is 41 Bezier and 51 final points, selected
against 81/101-point references with a 0.05 m dense-reference tolerance and
unchanged feasibility/safety decisions. None of these values changes the
EH216-S vehicle envelope.

## Fixed-rule populated-ring transfer repair (2026-08-31)

The `fixed_rule_warm_flow` visualization represents aircraft already
established in flight.  Its warm-start states therefore initialize the
complete steady ring-motion state (speed, tangential velocity and centripetal
acceleration); legacy curriculum warm starts keep their zero-speed semantics
unless `warm_start_steady_motion` is explicitly enabled.

Ring transfers preserve tangent-continuous, dynamically admissible entry
speed instead of forcing every source-ring holder to stop before entering the
Bezier.  The entry limit is derived from the same EH216-S curvature,
horizontal/vertical speed and acceleration constraints used by the canonical
time parameterizer.  Empty-target merge reservations propagate the
requester's source-ring phase to the merge time, avoiding a backward/looping
curve caused by an unrelated fixed or source-gap angle.

The SafetyFilter also verifies a ring-handoff candidate for the complete
30-second post-plan shield horizon against every committed trajectory.  This
closes the case where an earlier committed arrival has already become a fast
holder behind a later, slower candidate.  Safe distance, swept verification,
the EH216-S envelope, `physics_dt=0.2 s`, and `safety_sampling_dt=1.0 s` are
unchanged.

Deterministic `fixed_rule_warm_flow`, seed 17, 240-second evidence changed
from no outer-to-middle/middle-to-inner progress (only 2 releases approved;
3026 `safety_wait`) to 3 completed outer-to-middle and 3 completed
middle-to-inner transfers, with ongoing gate admissions.  Minimum actual and
swept separation was 81.189 m and hard-separation/NMAC/collision counts were
all zero.  Full regression after the repair: 202 passed.
