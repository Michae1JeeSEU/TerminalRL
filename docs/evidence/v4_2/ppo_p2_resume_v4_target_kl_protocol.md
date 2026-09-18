# P2 V4 target-KL guarded resume protocol

Status: preregistered implementation; no formal P2 V4 result is claimed here.
This document records the protocol boundary before any V4 qualification, entry,
or formal training run.

## Question

Can a fresh student-only qualification and guarded PPO continuation from the
last accepted P2 checkpoint complete the remaining P2 curriculum while keeping
each accepted optimizer update within a sampled requested-action target KL of
0.01, without relaxing the historical approximate-KL limit of 0.05?

The answer is intentionally unrun in this document. A code/test PASS is not a
training or policy-quality PASS.

## Frozen inputs and artifact hashes

The protocol is
`terminal_marl/configs/low_v4_2/ppo_p2_resume_v4.yaml`, whose SHA256 is
`01c92c44aecc50035e76b89a4e8d330fac5cfe5124b79d593c92a69260154ebf`.

The only permitted resume artifact is `checkpoint_5.pt` with SHA256
`8a8466f61119f7f5b5f780249d26f218ee9b8ae3e4aededc5c7f5fe2e51b201b`.
It must contain the V2.1 PPO model and a non-empty restored Adam state; a
different filename or byte hash is rejected.

The immutable failed V3 entry manifest is
`p2_resume_v3_20260918/entry/manifest.json` with SHA256
`2098af1c3109c17fc489b7c9f20a9db11eb67aa62b77b5a03beb2de66909cd7d`.
Before any reset, the runner also checks that it is semantically an entry
failure (`PASS: false`) with exactly one record (`seed: 763001`), with
`bounded_KL` as the only false gate and a present update object.

The source snapshot is bound into each V4 manifest/checkpoint. The current
pre-run hashes are:

| source | SHA256 |
|---|---|
| `terminal_marl/training/run_v4_2_ppo_p2_resume_v4.py` | `bb9dfcf15d6c74385e8b1d6ea3e47e527f9c83c4d67840f275335e5ac6b337b6` |
| `terminal_marl/configs/low_v4_2/ppo_p2_resume_v4.yaml` | `01c92c44aecc50035e76b89a4e8d330fac5cfe5124b79d593c92a69260154ebf` |
| `terminal_marl/algo/student_ppo_v4_2.py` | `bd590f907960de26d5290121f816e1cd428b20e2981705a2a6a5327b5b179301` |
| `terminal_marl/algo/joint_ppo_v4_2.py` | `abc2f3678b68de7808328ffdeccc5fa7f18d6c21527a38627fdcfc38fd173490` |
| `terminal_marl/core/checkpoints.py` | `2fc41879c256e86b04a2702aacb5a9cd0a02b9ae549bd10faf508dc82acec686` |
| `terminal_marl/core/trajectory_generator.py` | `8cb448b41893cb788cf013cc941d70b04f82102680b87384ce19dadb21bba0aa` |
| `terminal_marl/core/trajectory_time_parameterization.py` | `03517565d617a466d34b50ab9ad4879cc5e512ec2826f7ddac87ba55486a2850` |
| `terminal_marl/envs/parallel_env.py` | `8ea8b6fcf8f111c7c07610440da090d7c208bedccb734a5c6cfbaa647429e6cf` |
| `terminal_marl/training/student_rollout_v4_2.py` | `d7a91e38f2af95fe75e81f5e0686ceb08f109176210a19c24ca7c7fa2a537573` |
| `terminal_marl/training/run_v4_2_v2_1_student_qualification.py` | `a11cc3f3a3f773a00348864d52852ce16d27520bc1aca9abe76c8a768c59b59b` |
| `terminal_marl/training/run_v4_2_v2_1_ppo.py` | `fa6476faaccaeafd683316890a27fbee553dfefe41b3f5e44695e7c087b86d61` |
| `terminal_marl/training/run_v4_2_ppo_block2.py` | `b59cba2228b54a4fc3fa9e21aa2a75679033506563b23838827a90e4eb02c4ed` |

These hashes are the pre-run source snapshot emitted by the runner's
`p2_source_hashes()` helper; any source change causes a new hash-bound protocol
instance to be required before formal execution.

## Fresh seeds and budget

The held-out evaluation cohort `755001–755006` is reserved and untouched.
Fresh V4 seeds are disjoint from V3 and from the held-out cohort:

| phase | seeds | budget / scenario |
|---|---|---|
| qualification | `766001–766006` | six episodes; no PPO updates |
| entry | `767001`, `767002` | `flow6`, `flow8`; 160, 192 steps; admission-only |
| smoke | `768000` | `flow6`; 160 steps; diagnostic only |
| formal train | `769001–769008` | eight episodes; `flow6/flow8/flow12` cycle; full configured horizon |

Training uses eight episodes total, with the three flow scenarios cycled in
order. The P2 distance-loop reward remains disabled (`0.0`); other frozen
environment and PPO settings come from the YAML protocol.

## Update gates and rollback

For each guarded entry or formal-train update, `policy.update(...,
target_kl=0.01)` returns update accounting and KL observations. The pure gate
helper requires all of the following:

1. every numeric update value, model value, and optimizer value is finite;
2. `accepted_updates == updates` and `updates >= 1`;
3. `attempted_updates == accepted_updates + rejected_updates`;
4. `0 <= rejected_updates <= 1`;
5. `stopped` is true exactly when `rejected_updates > 0`;
6. both `final_rollout_kl` and `max_accepted_rollout_kl` are at most `0.01`;
7. legacy `approx_kl` is at most `0.05`.

An optimizer-step rejection restores both the model state and the complete
Adam optimizer state captured immediately before that step, then stops the
current PPO update. If at least one earlier step was accepted and every gate
still passes, the phase may continue with the next episode. A non-finite update
is rejected. Zero accepted updates are a failure and cannot create a
checkpoint. A checkpoint is saved and reloaded only after all other update
gates pass; exact model and Adam equality after reload is the `restore` gate.
A failed restore does not leave a promotable checkpoint.

## Phase ordering and evidence boundary

The pipeline is:

```text
checkpoint_5 + restored Adam
        → fresh qualification (no update)
        → entry (guarded, nonformal, discarded)
        → fresh formal train (guarded, formal checkpoints only)
```

Entry weights are never used as the formal training starting point: formal
training restores the original checkpoint-5 artifact again. The qualification
stage must pass all episode gates and observe both WAIT and RELEASE. Entry and
formal training stop at the first failed episode/update gate. Smoke is
diagnostic only and never grants `PPO_ALLOWED`; only a fully passed entry
manifest may set that field, and that field is not a claim of final policy
quality.

Historical V1/V2/V3 manifests and the failed V3 manifest are read-only. The
runner scans history for consumed seeds but does not write to the history
root. The held-out cohort is not evaluated, selected, or otherwise consumed by
this protocol. No unrun qualification, entry, formal-train, checkpoint, or
policy-quality PASS is claimed until a separately recorded execution produces
the corresponding manifest and artifacts.
