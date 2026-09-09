# V4 Low-Action Curriculum Impact

| Curriculum asset | V4 disposition |
|---|---|
| C0 μ_hat calibration | KEEP |
| Performance qualification | KEEP |
| Safety implementation | KEEP; revalidate V4 integration |
| SafetyTraceRecorder | KEEP; joint-action fields added |
| Observation V2 | KEEP |
| V3 L0 dataset | REGENERATE for V4 joint labels |
| V3 L1 checkpoints | NOT reusable |
| V4 L1 | fresh training only, after offline PASS |
| L1b/L2+ | not started |

V3 factorized actor, independent per-head Teacher targets, and per-head BC
conclusions are **SUPERSEDED**. No reward, BC weight, PPO hyperparameter,
lambda, Observation V2 field, or Safety authority changes are authorized by
this refactor. This report does not authorize PPO.
