# Performance-Constrained Vehicle Dynamics Specification

Current frozen implementation facts:

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

## Current EH216-S safety and energy contract

The main simulation uses `research_evtol` / EHang EH216-S /
`eh216s_research_v1`. Its current safety-distance semantics are:

- `80 m` desired/planning operating separation, certified first by executor-equivalent
  swept-segment checking plus recursive backup proof;
- `40 m` protected floor and NMAC threshold;
- `<10 m` collision-event research proxy, not an EH216-S certification distance.

If the nominal candidate cannot prove `80 m` but a bounded `PredictiveSafetyShield`
takeover can prove the actual executed trajectory remains `>=40 m` under the same swept
and recursive checks, the episode may continue and other certified UAVs are not stopped.
The `40--80 m` buffer incursion is logged separately; takeover and intervention are
charged to the shield reward, including a takeover whose numeric correction is zero. A missing proof, a
predicted/actual value below `40 m`, or an uncertified execution remains fail-closed
before physics execution.

The normalized EH216-S energy reference is `power_reference = 1.0` unit/s with
`energy_capacity = 1500` s-equivalent. Phase multipliers are engineering values until
an external power calibration is available. The profile's `min_landing_soc = 0.25` is a
frozen engineering research assumption; it is not a verified EH216-S AFM or type-
certification limit. Under the normalized reference, this floor corresponds to 375
normalized energy units, not necessarily 375 seconds of real flight. The conditional
literature basis and its limits are recorded in
[`landing_reserve_basis_20260924.md`](../evidence/v4_2/landing_reserve_basis_20260924.md).
A successful landing requires the vehicle profile's floor; early low-SOC termination
is not completion and is recorded as diversion or energy failure according to lifecycle
semantics.

The older `20 m` collision threshold, `10000` normalized energy capacity, and historical
qualification rule that treated any actual `80 m` separation breach as a hard failure
remain historical-only semantics. Archived evidence is not rewritten.

D7 core dynamics qualification passed on 2026-08-28; see
[`dynamics_qualification_report_2026-08-28.md`](../evidence/dynamics/dynamics_qualification_report_2026-08-28.md). `physics_dt=0.2 s` is
frozen for `performance_constrained_v2`. Later phase authorization and current blockers are recorded only in
[CURRENT_STATUS.md](../CURRENT_STATUS.md).

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
