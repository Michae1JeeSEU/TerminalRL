# P3 refresh energy-failure mechanism protocol (2026-09-30)

This bounded diagnostic uses one independent failure-mechanism sample. It is
not merged into the 24-random-actor cohort and does not estimate population
performance.

## Fixed inputs

- Root: `/home/user/training_outputs/terminal/p3_task_neutral_critic_refresh_comparison_20260929/full/refresh/training`.
- Record: `batch_02/training/episode_24/record.json`.
- Tape: `batch_02/training/episode_24/rollout/action_tapes/safety_trace_seed_881148_step_983.json`.
- Environment seed `881148`, policy seed `881248`, `984` steps, `24` requests.
- The retained natural target row is arrival `14`, UID `57`, outcome
  `failed_energy`, terminal step `892`.
- The source producer is `batch_01/policy.pt`, SHA256
  `1637a15d73184617a1e386cd29cbcdbe29c0679b1414f5fb413ec7a9fbef3805`,
  model digest `861cdc5f86e5e782a92b9a7cff9874836d3c3069397b7bcf8dddfbe0b8283e5c`.
- The alignment checkpoint is `batch_02/policy.pt`, SHA256
  `370344b6523f250bd6816bbfd6ef51bd065c4aea1352b21d8910adc1275e942c`.
  Its model digest is read from the actual `batch_02/summary.json` during
  preflight; it is used for target-row alignment only.

## Binding and phases

The configuration is exactly `neutral_eval._build_neutral_inputs()[1]["dense24"]`.
The source is loaded with `rank_eval._load_policy` as a full checkpoint.
Warmup checkpoints and after-checkpoint continuation are excluded. `PPO_ALLOWED`
is false; no optimizer, Teacher, or held-out input is used.

`preflight` reads and hashes the record, tape, source, after checkpoint and
summary, records runtime versions/config/loaded source-file hashes, and does
not create an environment or simulate a step.

`replay` performs one exact `984`-step tape replay with
`credit._capture_replay(state_observer=...)`. The ledger retains all `24`
requests, including the natural `failed_energy` row. The observer records each
step at which UID `57` is active, arrival `14`, and the canonical actor mask
is true. Each eligible row records the observed phase, SOC, energy remaining,
and mask status. Eligible rows are sorted by `(episode, step, UID)`.

The three registered states use, before seeing any diagnostic return, the
indices `floor((n-1)*q)` for `q=(0.25, 0.50, 0.75)`. Duplicate indices are
rejected. The selected rows retain the actual tape first actions and K=3
runtime proposal shortlists.

`diagnose` replays the tape once more and deep-copies only the three selected
pre-step states. Each state uses the existing passive-hook restoration helper,
the frozen source policy, continuation seeds
`9303201, 9303202, 9303203, 9303204`, and a `1800` branch-tick cap subject to
the original model/environment horizon. All `26` branches per seed are saved,
including failed, censored, and error branches. A qualified `D` is absent when
the paired result is incomplete; `m` is then absent as well. Each state saves
paired `D`, `C`, `p`, and `m` plus the source-to-after target-row alignment.

## Execution observations

- Actual preflight v3 reported `PREFLIGHT_OK` before replay.
- The replay completed exactly `984` steps and observed `24` requests. All
  terminal rows were retained, including the natural `failed_energy` row for
  arrival `14` and UID `57`; the source model digest was unchanged.
- The registered target states were step `280`, SOC `0.3986557690`; step
  `445`, SOC `0.2886557690`; and step `726`, SOC `0.1109871990`.
- Historical startup observation: the diagnose process was `RUNNING` with PID
  `2013086`; the first observed natural-terminal branch ended at `474` ticks,
  with `312` branch outcomes planned across the three states and four
  continuation seeds.
- The focused test file reported `8 passed`.

## Completed counterfactual observations (2026-09-30 17:49 +08:00)

The completed artifact `diagnose.json` reports `COMPLETED` for three states and
`312` branches. All branches were `natural_terminal` and uncensored; the model
snapshot, environment snapshot, and caller RNG were unchanged.

| state | step | SOC | C | p | D | m | full D by continuation seed |
|---|---:|---:|---:|---:|---:|---:|---|
| F01 | 280 | 0.398656 | 0 | 8.28291e-5 | -0.05 | +2.7650e-8 | `[-0.05, -0.05, -0.05, -0.05]` |
| F02 | 445 | 0.288656 | 0 | 0.000650799135 | +2.30523854878933 | -0.0002574839567 | `[0.593587408892546, -0.05, -0.05, 8.72736678626487]` |
| F03 | 726 | 0.110987 | 0 | 2.21449e-5 | -0.05 | +1.58285e-7 | `[-0.05, -0.05, -0.05, -0.05]` |

The full-return values use
`paired.expected_release_undiscounted_return_by_repeat[i] - wait[i].undiscounted_return`;
the mean check was within `1e-8`. For target arrival `14`, the comparisons
included WAIT and true q-weighted RELEASE. The RELEASE outcomes by seed were:
F01: seeds `9303201`, `9303202`, and `9303204` `emergency_landed`, seed
`9303203` `failed_energy`; F02: all four `emergency_landed`; F03: all four
`failed_energy`.

The bounded blocker telemetry remains in the `implementation` phase and has
not launched; no `D` is assigned from a one-step check. `PPO_ALLOWED=false`
and held-out inputs were unused. 第726步的一次gate干预不能在冻结后续策略下挽救该机；第445步续跑仍可应急落地，需要核对等待期资源/安全/能量拒绝原因，尚未选定训练修复。

The after update is reported as an alignment comparison only; this protocol
does not claim that it caused the already observed batch-02 failure. The
source model/environment snapshot and caller RNG remain unchanged; branch
copies evolve during simulation.
