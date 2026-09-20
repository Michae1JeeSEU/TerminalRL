# K=3 Runtime Shortlist Formal PPO V1

## Frozen protocol

- Protocol SHA256: `7e6cf765af93925ffe960ff2ab8a4f141e13c1101e8b2f933fd64a2296bf3048`
- Entry admission SHA256: `8d425da16345dd0546851d83bfe706877dd4e6298437b8ef9bd91da324175db3`
- Initial qualified candidate SHA256: `4b8be66c100aaaeb5b5699dcce68a96dbe1237964e4c308e7b23b95c62fb8c8c`
- Initialization: original qualified candidate with a fresh Adam optimizer; entry checkpoints were not used.
- Trainable scope: gate adapter and centralized critic only. The proposal-generating subgraph and
  calibration temperature remained frozen.
- Formal seeds: `808001–808006`; paired heldout seeds `809001–809006` were reserved and untouched.

## Result

Server output: `/home/user/training_outputs/terminal/p2_runtime_shortlist_formal_v1_808001_808006`

Manifest SHA256: `3c1bc69a518f7e5d99bb606d98b0356397e86f97b493947a0f14b0c8b49b010b`

| Seed | Scenario | Completed | Seconds | Accepted / rejected updates | Max accepted rollout KL | Fallbacks | Exhausted |
|---|---|---:|---:|---:|---:|---:|---:|
| 808001 | flow6 | 6/6 | 749 | 48 / 0 | 0.000336858 | 1 | 0 |
| 808002 | flow8 | 8/8 | 940 | 60 / 0 | 0.000514575 | 1 | 1 |
| 808003 | flow12 | 12/12 | 1174 | 74 / 0 | 0.004244336 | 3 | 7 |
| 808004 | flow6 | 6/6 | 760 | 48 / 0 | 0.000076847 | 0 | 3 |
| 808005 | flow8 | 8/8 | 926 | 58 / 0 | 0.004458141 | 0 | 1 |
| 808006 | flow12 | 12/12 | 1025 | 66 / 0 | 0.002737323 | 5 | 3 |

All 52 requests completed. All six episodes passed safety, geometry, exact replay, exact requested
likelihood, K=3 contract, frozen proposal-subgraph equality, post-update shortlist stability,
target-KL accounting, finite update, and exact checkpoint restore. In total, 354/354 optimizer
steps were accepted and none rejected; all safety counters and shield fractions were zero.

The final checkpoint is `checkpoint_6.pt`, SHA256
`714d141f7b472a17321e95e9c4c5574fdc3bcaa0c2911494f67b5b1a9f27592e`.

## Local 3D replay

The existing PyVista viewer can replay the learned P2 controller with the same fixed geometry,
natural arrival flow, deterministic action selection, and K=3 runtime proposal shortlist used by
the P2 evaluation path:

```powershell
python -m terminal_marl.visualization.ppo_checkpoint_viewer `
  --checkpoint D:\7_codespace\training_outputs\terminal\p2_runtime_shortlist_formal_v1_808001_808006\checkpoint_6.pt `
  --profile p2_trained_k3 --scenario flow12 --seed 809006 --playback-speed 2
```

For a visual comparison, use `--profile p2_initial_k3` with the original qualified checkpoint.
The replay is diagnostic only and does not change the heldout promotion decision below.

## Decision

`PASS=true` and `HELDOUT_EVALUATION_ALLOWED=true`. The checkpoint remains non-promotable and
`P3_ALLOWED=false` until the separately preregistered untouched paired heldout evaluation passes.
