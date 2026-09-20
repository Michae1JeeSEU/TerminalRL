# K=3 Runtime Shortlist PPO Entry V1

## Frozen inputs

- Protocol: `V4_2_P2_RUNTIME_SHORTLIST_PPO_ENTRY_V1`
- Protocol SHA256: `d3400bf47d0f7f0d86ce4ed16c5b547b6ab30e0220691eb1f33eef320af932be`
- Admission review SHA256: `a549ffc60c35cb5ac1403ba0c4b59412096b2936d4a209ddb1a25219e09e8783`
- Initial candidate SHA256: `4b8be66c100aaaeb5b5699dcce68a96dbe1237964e4c308e7b23b95c62fb8c8c`
- Runtime boundary: one policy forward, requested proposal at rank 1, at most three ordered candidates.
- Trainable scope: `gate_adapter`, `critic_encoder`, and `critic_head` only. The actor encoder,
  base gate head, proposal head, and calibration temperature were frozen.
- Entry updates and entry checkpoints were admission-only and are forbidden as formal-training
  initialization.

## Execution

Server output:
`/home/user/training_outputs/terminal/p2_runtime_shortlist_ppo_entry_v1_807001_807002`

`admission.json` SHA256:
`8d425da16345dd0546851d83bfe706877dd4e6298437b8ef9bd91da324175db3`

| Seed | Scenario | Steps | Actor decisions | Accepted / rejected updates | Final rollout KL | Shortlists | Fallbacks | Max checked |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 807001 | flow6 | 160 | 3 | 10 / 0 | `1.82848e-10` | 3 | 0 | 1 |
| 807002 | flow12 | 256 | 7 | 16 / 0 | `2.74221e-7` | 7 | 1 | 2 |

Both entries passed safety, exact environment replay, exact requested-action likelihood, initial
ratio, stochastic student-only behavior, bounded K=3 contract, finite model/optimizer/update,
target-KL accounting, frozen proposal-subgraph equality, post-update shortlist-order stability,
and exact model/optimizer checkpoint restore. All safety counters were zero. The targeted server
suite passed `26` tests.

## Decision

`PASS=true` and `PPO_ALLOWED=true` for a separately preregistered formal run using the same
K=3 boundary and frozen proposal-generating subgraph. `P3_ALLOWED=false` and
`checkpoint_promotable=false`. Formal training must restart from the original qualified candidate
with a fresh optimizer; neither entry checkpoint is eligible as its initialization.
