# V4.1 Joint Action 30→26 Migration

V4's `V4_joint_discrete_30` catalog is superseded because the frozen WAIT
speed semantic audit classified nonzero WAIT residuals as policy-dominated.

| Item | V4.1 result |
|---|---|
| Schema | `low_action_schema_v4_1_joint_discrete_26` |
| WAIT actions | 1: index `0` → `[WAIT, 0, 0]` |
| RELEASE actions | 25: indices `1..25`, unchanged V4 speed×merge physical decode |
| Runtime boundary | ExecutionScheduler certifies one requested proposal; no catalog search |
| WAIT ownership | no policy speed residual; ring-holder execution and PredictiveSafetyShield own correction |
| Checkpoints | V4 `Categorical(30)` is not resumable; V4.1 uses a fresh 26-way actor |
| ActionTape | schema-versioned V4.1 exact replay; V4 tapes are historical/read-only and rejected by V4.1 replay |

Teacher enumeration now contains the 25 RELEASE candidates and one canonical
WAIT, without a WAIT `abs(speed_residual)` tie-break. PPO provenance remains
the requested joint index/log-probability only. The I0 gate now checks
decision-relevant expert target families rather than all catalog classes.

No I0 collection, I1 BC, DAgger, PPO update, reward change, Observation
change, or safety-authority change was performed in this migration.
