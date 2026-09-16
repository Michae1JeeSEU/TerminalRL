# V4 I2 Safe DAgger Preflight Summary

Implementation: `terminal_marl.training.low_v4_imitation:run_i2_dagger_preflight`.

The short preflight is gated on I1 PASS. The student chooses the requested
joint action and remains authoritative through `ExecutionScheduler`, trajectory
generation, SafetyFilter and PredictiveSafetyShield. The Teacher queries the
same pre-step state only to attach its expert label. Any trace reporting a
Teacher execution correction aborts the run.

The preflight records student requested joint action, Teacher joint label,
execution-applied action, shield-applied action, decision metadata, state
source and ActionTape. It appends records to a new aggregate dataset rather
than overwriting I0, with parent/checkpoint/round/dedup/Teacher-version/label
distribution lineage, then verifies exact ActionTape replay.

Current evidence state:

```text
I2_DAGGER_PREFLIGHT = FAIL (not run; blocked until I1 PASS)
PPO_ALLOWED = false
```
# V4.1 amendment

I2 remains blocked until a separately collected V4.1 I0 dataset passes and a
subsequent V4.1 I1 joint-BC qualification passes. No DAgger round was run.
