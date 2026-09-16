# V4.2 / Observation V2.1 Multi-Seed I1 Training Summary

## Evidence and isolation

- Evidence type: `newly_run`, 2026-09-14.
- Workdir: `~/projects/terminal` on `lab-server`.
- Output: `~/training_outputs/terminal/v4_2_v2_1_i1_multiseed_formal_20260914`.
- Python `3.12.3`, PyTorch `2.13.0+cu130`, CPU execution (`cuda_available=false`).
- Training input: immutable formal I0-R, 2,020 rows, SHA256
  `6bcbc1782a171f909282126c4f6d27dd93355b09513e2827093228af4fe05f05`.
- Protocol SHA256:
  `6da3ff8f3c790428c6e16d093a63c4e09c1db44e5ff2676e4abfb95fa3d47d11`.

All three actors used fresh initialization, no checkpoint resume, Adam at
`3e-4`, batch size `64`, `100` epochs, and final-epoch selection. The loss was
the unweighted sum of gate CE over all rows and proposal CE over
Teacher-RELEASE rows. No class weighting, focal loss, over/undersampling,
balanced sampling, row duplication, label smoothing, proposal weighting, or
hyperparameter search was used.

An interpreter audit hook fail-closed on any attempt to open either Q artifact
during the training phase. All three checkpoints and logs were complete and
hashed before the process entered its single Q-evaluation phase.

```text
Q_DATASET_ACCESS_DURING_TRAINING = false

TRAINING_4202_FROZEN = true
seed4202_checkpoint_sha256 = ce0702e69454fe8893189be80a6e2c68ab186e9de46f1dec540cce8027b9a284
seed4202_training_log_sha256 = cd7f99c885041d56edf247601f974e9e0bbb057f48e98c0dc285519f81f3286a

TRAINING_4203_FROZEN = true
seed4203_checkpoint_sha256 = 64397412cf02904915a0210349c7054e547badf31f3c85546787947e555f7957
seed4203_training_log_sha256 = 573550405183b6e1ca34200bf243342da95d8d50bbaf756915da31bcd41a1d58

TRAINING_4204_FROZEN = true
seed4204_checkpoint_sha256 = da6c7f7eda6f64407414bb7888355a1bf62c42b8d3657ffa6225111ff69314e5
seed4204_training_log_sha256 = 655f949168d8b977f73609811a14b6c545a6c9d31d76aeafd47b9226bc541462

training_freeze_manifest_sha256 = 2225ce22ae397e632e559e95ca1a366ba8105a64f242cbab4a2940bb1c571a0c
```

Every recorded gradient norm was finite. Training-only metrics were logged at
every epoch, but did not change the fixed epoch count, optimizer, checkpoint,
architecture, seed, or any qualification threshold.
