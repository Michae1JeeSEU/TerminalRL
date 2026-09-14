# V4 New Mainline Migration Summary

Curriculum: `A5V2_LOW_LEVEL_TRAINING_CURRICULUM_v4_DAGGER_PPO`.

**V4.1 amendment:** the active action schema is
`low_action_schema_v4_1_joint_discrete_26` (`1 WAIT + 25 RELEASE`). V4's
30-class catalog is superseded due to WAIT action-space overparameterization.
V4.1 I0 is permitted only after its S0 gate (now PASS) and remains a separate,
not-yet-run collection.

The active low-level training path is now:

```text
I0 Structured Expert Dataset -> I1 Joint BC -> I2 Safe DAgger preflight
-> I3 student-only imitation qualification -> PPO admission review
```

`terminal_marl/training/low_v4_imitation.py` is isolated from historical PPO
runners and refuses its `ppo` command while `PPO_ALLOWED = false`.

## Migration audit

| Classification | Location | Handling |
|---|---|---|
| Historical / ablation | `train_low_v2_ppo.py`, `run_v4_l1_short_pilot.py`, `l1_*` diagnostics and tests | Kept intact as prior FULL-Teacher PPO evidence; never referenced by `configs/low_v4/`. |
| Required infrastructure | `v4_joint_teacher.py`, joint action catalog, Observation V2, `ExecutionScheduler`, trajectory generator, SafetyFilter, PredictiveSafetyShield, `SafetyTraceRecorder` | Reused unchanged by I0/I1/I2. |
| Disabled from new mainline | Full Teacher execution correction, `IMMEDIATE_ONLY`, L1/L1b/L2 stage progression, teacher-intervention reward / teacher-corrected PPO rollout | Not present in the new runner or configs. Teacher is an I0/I2 labeler only. |

## Recorded current status

```text
C0 = PASS                         # user-provided frozen capacity evidence; mu_hat=0.003
S0_V4_SYSTEM_QUALIFICATION = PASS # existing V4 system qualification
I0_STRUCTURED_EXPERT_DATASET = NOT QUALIFIED
I1_JOINT_BC = NOT QUALIFIED
I2_DAGGER_PREFLIGHT = NOT STARTED
I3_STUDENT_IMITATION_QUAL = NOT STARTED
PPO_ALLOWED = false
```

No old V4 L0 dataset (including its 3511 WAIT / 89 RELEASE distribution) is
accepted as I0 evidence. A new collector writes a fresh dataset and a separate
frozen decision-state validation cohort before I1 can start.

```text
core_V4_architecture_kept = true
old_FULL_Teacher_PPO_mainline_disabled = true
I0_STRUCTURED_EXPERT_DATASET = FAIL (formal collection: WAIT speed coverage)
I1_JOINT_BC = NOT_RUN (blocked by I0)
I2_DAGGER_PREFLIGHT = NOT_RUN (blocked by I1)
PPO_ALLOWED = false
```
