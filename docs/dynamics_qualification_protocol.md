# D7 Dynamics Qualification Protocol

This protocol is a non-training gate between the EH216-S dynamics upgrade and
C0 capacity calibration.  It must not invoke PPO optimisation.

## Gates

1. Run the D7 unit gates: acceleration, ring feasibility, time
   parameterisation, checked-equals-executed hash telemetry, dt convergence,
   and cached holder prediction complexity.
2. Run the fixed-rule landing, scheduler, controllability and
   reconfiguration regressions.  Exclude tests marked or named as PPO
   training; those are not a D7 prerequisite.
3. Run `terminal_marl.training.dynamics_qualification` for N=10/30/60 and
   `physics_dt=0.2/0.1`.  Store the JSON output in the server training-output
   directory, never in the source tree.
4. Accept only if deterministic scenarios have zero collision, NMAC and hard
   separation violations, all tests pass, and the measured throughput is
   recorded.  The result fixes the formal `physics_dt`.

## Server commands

After Syncthing has propagated the local code to `~/projects/terminal`:

```bash
cd ~/projects/terminal
/home/user/envs/terminal_env/bin/python -m pytest \
  tests/test_dynamics_qualification.py \
  tests/test_vehicle_dynamics.py \
  tests/test_trajectory_time_parameterization.py \
  tests/test_phase_a_landing.py \
  tests/test_a612_scheduler_repair.py \
  tests/test_a67_controllability_audit.py \
  tests/test_a68_reconfiguration_safety.py -q

/home/user/envs/terminal_env/bin/python -m terminal_marl.training.dynamics_qualification \
  --uavs 10 30 60 --physics-dts 0.2 0.1 --simulated-seconds 3600 \
  --output /home/user/training_outputs/terminal/dynamics_d7_YYYYMMDD/benchmark.json
```

The benchmark deliberately holds the warm-start traffic to isolate execution
and separation-monitor cost; it does not train or evaluate a learned policy.
Long-horizon rule-only traffic regressions are run separately with line-buffered
logs and a timeout, so a failure is observable and cannot leave an orphaned
server process.

## Performance implementation note

`SafetyFilter` samples holder predictions once for an ordered safety horizon.
It must not reintegrate a holder from its origin for every query time.  This
keeps the finite-acceleration semantics intact while changing prediction cost
from quadratic to linear in the number of safety samples.
