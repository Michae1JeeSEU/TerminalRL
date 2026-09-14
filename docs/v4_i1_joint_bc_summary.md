# V4 I1 Joint BC Summary

Implementation: `terminal_marl.training.low_v4_imitation:train_i1_joint_bc`.

I1 starts only from a passing new I0 manifest and initializes a fresh V4 joint
actor. Its sole update is `CE(joint_logits, teacher_joint_index)`; it does not
call PPO update logic, use reward, alter observations, alter the catalog, or
alter safety authority.

The runner records CE, joint top-1/top-3, WAIT and RELEASE recall, RELEASE
precision, release macro-F1, speed/merge disagreement, entropy, and gradient
norm. Its frozen threshold block is written before the outcome and checks CE
reduction, both class recalls, macro-F1, non-collapsed fixed-cohort outputs,
finite gradients, and safety/provenance status.

Current evidence state:

```text
I1_JOINT_BC = NOT_RUN (blocked by failed formal I0)
I2_DAGGER_PREFLIGHT_ALLOWED = false
PPO_ALLOWED = false
```
# V4.1 amendment

No V4.1 I1 joint-BC run occurred. Any future V4.1 I1 must use a fresh
26-way actor and a frozen V4.1 I0 validation cohort; V4 30-way checkpoints
and labels are not valid initialization or formal data sources.
