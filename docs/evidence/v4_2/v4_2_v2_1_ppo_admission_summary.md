# V4.2 / 88D PPO admission — PASS

Evidence type: newly_run, 2026-09-17. Code commit `1470229`.
The low policy is admitted for the initial fixed-geometry N3–N6 curriculum.
Full static-geometry robustness qualification and A6 dynamic airspace remain
downstream. The formal PPO block itself has not been started.

| Stage | Result | Evidence |
|---|---|---|
| R4 | PASS | 7,576 training rows, 1,886 audit rows; all12 gates; 52/52 replay |
| I1 V2 / Q2 | PASS | 1,787 untouched Q2 rows; one evaluation; all3 actor seeds pass |
| 88D I2 preflight | PASS | two6-step episodes, requested likelihood and replay exact |
| DAgger R1 | PASS | 976 new events, D1=8,552, 35epochs, 16/16 replay |
| I3 student-only | PASS | all24 UAVs completed across6 episodes; queues drained; zero safety/shield |
| Admission review V2 | PASS | exact lineage, schema, student-only behavior and scope |
| PPO entry V2 | PASS | two fresh rollouts, finite updates, exact initial ratio, save/load/continuation |

Final admission:
`~/training_outputs/terminal/v4_2_ppo_entry_verify_v2_20260917/admission.json`

SHA256: `959bf9300fb44f4eec02783136dcfda11826dbf125e36a85e60550e241511c62`.
`PASS=true`, `PPO_ALLOWED=true`, `formal_training_called=false`.
The formal-mode admission validator accepted this file against current source
hashes (`FORMAL_ENTRY_ADMISSION_ACCEPTED`).

Entry seed671001: 32steps, 6 actor decisions, 2 minibatch updates,
approximate KL9.93e-9. Seed671002: 128steps, 1 actor decision, 8 minibatch
updates, KL0.0. Initial requested logprob error was0.0 for both. All gradients,
losses and parameters finite; all safety counts zero; replay and checkpoint
reload gates pass. This validates software execution, not PPO convergence.
V1's insufficient32-step natural-arrival window remains a recorded FAIL.

## Formal training command

Run on `lab-server`, after entering `~/projects/terminal` and checking code
has synced. Outputs stay outside the source tree. The first registered block
is bounded to at most12 episodes; this command starts that block and must be
followed by Luna high monitoring per the user's execution preference.

```bash
~/envs/terminal_env/bin/python -u -m terminal_marl.training.run_v4_2_v2_1_ppo \
  --mode train \
  --output-dir ~/training_outputs/terminal/v4_2_ppo_initial_block_20260917 \
  --dagger-manifest ~/training_outputs/terminal/v4_2_dagger_v2_1_r1_formal_20260917/manifest.json \
  --admission ~/training_outputs/terminal/v4_2_ppo_entry_verify_v2_20260917/admission.json \
  --history-root ~/training_outputs/terminal \
  --episodes 12
```

Initialization is always the qualified DAgger actor, SHA256
`a97e0284b07b18404f60cd666c72806cbf5e261bf807c7f9f3a1d897c69b1584`.
Entry-validation checkpoints are ineligible. Protocol and implementation
changes invalidate admission. PPO ratios never use Teacher or shield-applied
actions. Current stage has no live server job or child monitor remaining.

## Curriculum assessment and remaining limits

BC → Safe DAgger → student-only qualification → PPO is supported by the new
evidence. The earlier blockers were support-budget mismatch, obsolete87D
lineage, incomplete PPO/rollout plumbing and the natural-arrival approach
interval. None required bypassing Teacher/safety semantics or reusing Q2.

The remaining bottleneck is operational headroom: deterministic flow6 takes
867 of900s. Continue bounded PPO at qualified loads before larger traffic,
shorter headways or geometry changes. Current evidence does not establish
saturated throughput, broad generalization, high-level dynamic control or
hardware readiness. Six I3 episodes and two entry rollouts justify only the
specified initial training scope.

Validation:16 targeted admission/PPO/rollout/I3 tests pass, plus the preceding
stage-specific tests and actual small-sample/qualification runs. Source and
Git changes were local; Syncthing hashes matched before execution. No Git
remote is configured, so commits have not been pushed.
