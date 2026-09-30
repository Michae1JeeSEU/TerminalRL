# P3 follow-up 修复与复核事实（2026-09-30）

## 范围与集成检查

本页记录本轮 review scope 的实现、只读诊断 artifact 和测试事实；不把回归测试写成 qualification，也不声称 P3 性能改善。未运行训练、optimizer、held-out 或 pilot 性能评估。

本地唯一集成命令为：

```text
D:\7_codespace\venv\Scripts\python.exe -m pytest -p no:cacheprovider -q \
  tests/test_paired_gate_diagnostic.py tests/test_service_context_migration_v4_2.py \
  tests/test_high_cost_conservation.py tests/test_research_high_level_repairs.py \
  tests/test_a6_effectiveness.py tests/test_a6_dynamic_high.py \
  tests/test_forced_wait_protocol.py tests/test_research_low_level_repairs.py \
  tests/test_student_rollout_v4_2.py tests/test_p3_followup_diagnostics.py
```

结果为 `82 passed in 45.59s`、exit `0`。`--collect-only -q` 同样 exit `0`，收集 `82` 个唯一 nodeids。原始日志位于 `.tmp/p3_followup_review/integration.log` 和 `.tmp/p3_followup_review/collect_only.log`。

| 模块 | nodeids |
|---|---:|
| `test_paired_gate_diagnostic.py` | 16 |
| `test_service_context_migration_v4_2.py` | 11 |
| `test_high_cost_conservation.py` | 5 |
| `test_research_high_level_repairs.py` | 7 |
| `test_a6_effectiveness.py` | 2 |
| `test_a6_dynamic_high.py` | 10 |
| `test_forced_wait_protocol.py` | 6 |
| `test_research_low_level_repairs.py` | 8 |
| `test_student_rollout_v4_2.py` | 9 |
| `test_p3_followup_diagnostics.py` | 10 |

## 已确认的实现修复

- [`paired_gate_diagnostic.py`](../../../terminal_marl/training/paired_gate_diagnostic.py) 的 `restore_diagnostic_snapshot` 在 `TerminalParallelEnv` 深拷贝后重新绑定该 copy 的 `_resolve` 与 `arrivals.requests_until`。真实隔离测试 [`test_paired_gate_diagnostic.py`](../../../tests/test_paired_gate_diagnostic.py) 检查 branch hook 的 `__self__`、未来请求读取和 source ledger/RNG 不变。
- [`student_rollout_v4_2.py`](../../../terminal_marl/training/student_rollout_v4_2.py) 在 opt-in `pre_action_forced_wait` 下将 forced rows 的 gate/joint logprob 记录为 `0`；proposal 继续为 `-1/NaN`，metadata 继续为 `None`。legacy 模式和 actor 可行动 rows 保持原请求/RNG/动作结果。相关 6 个测试包含 25 个真实环境 proposal、affected-interface、正 SOC 低电量和 PPO masked-row 检查。
- high-level cost metadata、600 秒 sustained-low shrink regression、queue history/unresolved overlap 和 V4.2 adapter fail-fast 均在集成套件中通过；这些是代码回归事实，不是 A6 qualification。

## 服务上下文迁移

[`service_context_migration_v4_2.py`](../../../terminal_marl/algo/service_context_migration_v4_2.py) 与 [`test_service_context_migration_v4_2.py`](../../../tests/test_service_context_migration_v4_2.py) 的 11 个测试通过。迁移绑定的 source 是 `phaseA_low_obs_v2_1_retry_state` 88D，target 是 V2.4 90D；新增输入权重列初始化为 0，实际 V2.4 观测列按 UAV/LZ 状态计算。V2.3 只有在调用方显式传入该 schema 时被拒绝，不能据 88D 维度推断所有 schema 语义等价。迁移保留已有状态、mode、gradient 和 CPU RNG；fresh optimizer required，qualification status 仍为 experimental only。

## Paired gate helper 与新 runner

[`paired_gate_diagnostic.py`](../../../terminal_marl/training/paired_gate_diagnostic.py) 当前集成测试为 16 passed。每个 repeat 的 25 个 proposal 分支按实际请求和权重记录；branch error/censor 时不产生 qualified `delta_Q`，raw branch report 和 progress 仍保留。continuation 使用同一个 frozen policy；diagnostic 不调用 optimizer，也不改变 model、gradient、state 或 RNG。

[`run_p3_followup_diagnostics.py`](../../../terminal_marl/training/run_p3_followup_diagnostics.py) 是新的独立诊断入口，配合 credit observer；其 runner tests 为 10 passed。代码登记的输入契约是 6 episodes、每 episode 24 requests、总计 4,497 steps；selection test 固定 24 个 occurrence、无放回且顺序可复现。sharding 后的 focused runner suite 为 `10 passed`，不计入上述 `82` 个已记录集成 cases。本轮只验证代码和契约，没有把 runner phase 输出写成性能结果。

追加事实：此前 Python `-m` 运行时的 `__main__` entry file 被 name filter 遗漏；当前仅在 package root 内纳入该 entry file。该修复的 focused suite 为 `11 passed in 2.35s`，重复测试不计入新的总数。当前运行中的 preflight 未重跑或覆盖；修复适用于后续 preflight，既有 hash 和 startup observation 保持原边界。

## 既有 episode-gradient replay artifact

实际服务器运行使用的旧模块 [`diagnose_p3_task_neutral_episode_gradients.py`](../../../terminal_marl/training/diagnose_p3_task_neutral_episode_gradients.py) 是本次 replay 的 artifact source；它不是本提交新增的自包含入口。`runtime_sources.json` 记录其 SHA256：`c5e35752f78e33698510c59aa36ed9dd714606c07a80d785aa76c49d42aa19bf`。

- preflight 为 `CHECK_ONLY_OK`：6 episodes、4,497 steps、144 requests，`runtime_optimizer_calls=0`、`runtime_replay_calls=0`、`runtime_env_reset=false`。
- legacy replay 为 `COMPLETED`：6 episodes、4,497 steps、144 requests、6 replay calls；`replay_exact`、`mc_suffix_match_all`、`aggregate_gradient_matches_sum`、`model_unchanged_after_replay_and_gradient` 均为 true，`training_called=false`、`optimizer_used=false`、`PPO_ALLOWED=false`。
- artifact paths：`/home/user/training_outputs/terminal/p3_followup_20260930/runtime_sources.json`、`legacy_preflight/report.json`、`legacy_replay/report.json`、`legacy_replay/summary.json`。

## pilot artifact

给定 pilot artifact `D:/7_codespace/training_outputs/terminal/research_review_repairs_20260930/p3_followup/pilot.json`（服务器路径：`/home/user/training_outputs/terminal/research_review_repairs_20260930/p3_followup/pilot.json`）状态为 `COMPLETED`。它只包含 episode 1、step 278、UID 5、continuation seed `9303101`：26 个 branch 均 `natural_terminal`，branch steps 范围 `410..427`，总计 `10677` ticks，无 censor 和 safety event；prefix 410 valid，model/source snapshot、RNG 和 `q_unchanged` 均为 true。

该状态的 paired 值为 `C=0`、`p=0.7176065445`、`p*C=0`、`D=-1.4162728544`。对实际 batch-01 after model 的 gate 对齐为 `0.7176070213→0.7267845273`，`delta=0.0091775060`，linear approximation `0.0093069952`，residual `-0.0001294892`，`local_m=-0.0129978526`。

该条件状态的一次续跑显示更新方向不利；一个状态一个seed不支持认定credit、entropy或clip为退化根因，也不构成性能修复证据。

历史计算记录：`C` 固定为 `24×4`，由 4 个进程按 modulo 分片；这只是计算调度，不增加实验 factor、不重抽样、不删除失败 branch。此前 4 个分片进程 PID 为 `2007505`、`2007601`、`2007551`、`2007650`；timestamp snapshot 曾显示 `14/24`，该快照不是终态，最终结果见下节。`PPO_ALLOWED=false`，未消费 held-out。此前 `82` 个集成测试只作为已记录 raw count，不在本节重复累计。

## C 初步结果

raw artifact `causal/shard_0/diagnose/state_0001.json` 当前包含同一首状态的 `104` 个分支，来自 `4` 个 continuation seeds，均为 `natural_terminal`；model、environment 和 caller RNG 均保持不变。该状态的 `C=0`、`p=0.7176065`、平均 `D=6.56333546`、平均 `m=0.06023505`；四个 seed 的 `D` 为 `[-1.41627285, -0.05, 11.62504392, 16.09457077]`。

同一首状态四种续跑的符号混合，单seed pilot负值不能用于根因判定；C=0仅表示即时覆盖为零，不能据此认定全周期动作无效或直接强制WAIT。该段是早期 pilot/timestamp snapshot；完整 24 状态的最终 named-group 结果见下节。失败机制三状态、312 个 branch 已完成，不构成性能改善证据。

## C 全部 24 状态最终结果

本节读取四个 causal shard 的 named group JSON（`summary.json`、`report.json`、`diagnose.json` 及其列出的 state report JSON），不把 raw branch 文件作为汇总输入。四个 shard 的状态均为 `COMPLETED`，各含 6 个 state、624 个 branch；全局共 24 个 state、2,496 个 branch（每个 state 为 4 个 continuation seed、每个 seed 26 个 WAIT/RELEASE 分支）。四个 shard 的 `diagnose.json` 修改时间分别为：shard 0 `19:12:51`、shard 1 `18:23:26`、shard 2 `19:19:08`、shard 3 `18:14:25`。

24/24 个 state 的 D 与 m 均有值；2,496/2,496 个 branch 为 `natural_terminal`，error 与 censor 均为 0。所有 state 的 model、environment 和 caller RNG 不变标记均为 `true`，四个 shard 的 `PPO_ALLOWED=false`、held-out=false、optimizer calls=0、teacher=false。source checkpoint `3fe7dd1e01d032227521f6fc5c8464fe8d00d98fb0e9f0dbd13411123a27d4ab` 与 warmup `a82ed7a287035fb1fc868fc955e056c0481de75bcb99880d5c79f35588dde710` 固定；after batch-01 checkpoint 的 SHA256 为 `a7bcf3d1e6f4bda82d0d051ec25966a18f267309f1d6030689b52d90e63aa366`。D 与四个 seed 差分均值的最大绝对误差为 `7.1054273576010019e-14`。

表中 `C` 是 `C_i`，`p` 是 gate 的 `p_release`，`D` 是 qualified full-return difference，`m` 是 alignment 的 per-row `local_m`。`Dseed` 顺序为 continuation seeds `9303101`、`9303102`、`9303103`、`9303104`；表中标注“浮点近零”的 D 保留原始科学计数值，不改写成 0。

| selection | step / UID | C | p | D | m | Dseed |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 278 / 5 | 0.0000000000000000e+00 | 7.1760654449462891e-01 | 6.5633354576678471e+00 | 6.0235050345867361e-02 | `-1.4162728544092147e+00, -4.9999999999954525e-02, 1.1625043919265863e+01, 1.6094570765814922e+01` |
| 2 | 319 / 97 | 0.0000000000000000e+00 | 6.8654189817607403e-06 | -5.0000000000011369e-02 | 1.8383116184854403e-08 | `-5.0000000000011369e-02, -5.0000000000125056e-02, -5.0000000000011369e-02, -4.9999999999954525e-02` |
| 3 | 354 / 237 | 0.0000000000000000e+00 | 1.9145017722621560e-03 | -2.4999999999948841e-02 | -4.5512570068147030e-08 | `2.8421709430404007e-14 (浮点近零), 5.6843418860808015e-14 (浮点近零), -4.9999999999926104e-02, -5.0000000000011369e-02` |
| 4 | 361 / 298 | 0.0000000000000000e+00 | 5.5721051289543363e-16 | -5.0000000000011369e-02 | 1.5885547060386379e-18 | `-4.9999999999954525e-02, -4.9999999999954525e-02, -5.0000000000011369e-02, -5.0000000000011369e-02` |
| 5 | 96 / 2 | 0.0000000000000000e+00 | 2.2396884858608246e-02 | -9.7227664391866142e+00 | -1.5937326957307082e-02 | `-5.0000000000068212e-02, -4.6285767845538089e+01, 4.7968038411779617e+00, 2.6478982476137958e+00` |
| 6 | 131 / 2 | 0.0000000000000000e+00 | 3.4216021682631492e-10 | -4.9999999999897682e-02 | 2.6690594179202613e-13 | `-4.9999999999954525e-02, -4.9999999999897682e-02, -5.0000000000011369e-02, -4.9999999999840838e-02` |
| 7 | 155 / 18 | 0.0000000000000000e+00 | 6.3217245042324066e-02 | -3.7499999999965894e-02 | -3.2444112002820074e-05 | `-5.6843418860808015e-14 (浮点近零), -5.0000000000011369e-02, -5.0000000000068212e-02, -5.0000000000011369e-02` |
| 8 | 187 / 0 | 0.0000000000000000e+00 | 7.6557704620228151e-11 | -5.0000000000068212e-02 | 2.2415402867212491e-13 | `-5.0000000000011369e-02, -5.0000000000011369e-02, -5.0000000000068212e-02, -5.0000000000011369e-02` |
| 9 | 312 / 5 | 0.0000000000000000e+00 | 1.6058335217117303e-10 | -5.0000000000011369e-02 | 1.1853712456047414e-13 | `-4.9999999999954525e-02, -5.0000000000011369e-02, -4.9999999999954525e-02, -5.0000000000011369e-02` |
| 10 | 335 / 53 | 0.0000000000000000e+00 | 9.3492984888143837e-05 | -5.0000000000068212e-02 | 6.0744823713404112e-07 | `-5.0000000000068212e-02, -5.0000000000068212e-02, -5.0000000000068212e-02, -5.0000000000068212e-02` |
| 11 | 373 / 0 | 0.0000000000000000e+00 | 5.8644739910960197e-03 | -5.0000000000039790e-02 | 4.6765222214199566e-05 | `-4.9999999999982947e-02, -5.0000000000039790e-02, -5.0000000000068212e-02, -5.0000000000039790e-02` |
| 12 | 417 / 74 | 0.0000000000000000e+00 | 1.6107469524850870e-11 | -5.0000000000011369e-02 | 8.5018016932078455e-14 | `-5.0000000000011369e-02, -5.0000000000039790e-02, -4.9999999999982947e-02, -5.0000000000068212e-02` |
| 13 | 440 / 74 | 0.0000000000000000e+00 | 8.1910187699991218e-16 | -4.9999999999954525e-02 | -1.1249656330710945e-20 | `-4.9999999999982947e-02, -4.9999999999926104e-02, -4.9999999999982947e-02, -4.9999999999982947e-02` |
| 14 | 497 / 32 | 0.0000000000000000e+00 | 3.7413606150948908e-06 | 4.1246121364927717e+00 | -1.5179809506855633e-06 | `4.9905124224302710e+00, 5.3624389315122869e+00, 4.1015253646625638e-01, 5.7353446555621872e+00` |
| 15 | 2 / 11 | 0.0000000000000000e+00 | 1.5535855008859903e-18 | -4.9999999999954525e-02 | 4.4498181052249653e-22 | `-5.0000000000068212e-02, -4.9999999999897682e-02, -4.9999999999897682e-02, -5.0000000000011369e-02` |
| 16 | 297 / 211 | 0.0000000000000000e+00 | 1.9081822710473317e-14 | -4.9999999999954525e-02 | 8.8530528394223347e-17 | `-4.9999999999954525e-02, -4.9999999999954525e-02, -4.9999999999954525e-02, -5.0000000000011369e-02` |
| 17 | 41 / 5 | 0.0000000000000000e+00 | 3.7640047328979520e-11 | -5.0000000000011369e-02 | 1.0240315539980058e-13 | `-5.0000000000068212e-02, -5.0000000000011369e-02, -4.9999999999954525e-02, -5.0000000000068212e-02` |
| 18 | 328 / 0 | 0.0000000000000000e+00 | 1.4511273801326752e-01 | -5.0000000000068212e-02 | -1.0516569018378300e-04 | `-5.0000000000125056e-02, -5.0000000000011369e-02, -5.0000000000125056e-02, -5.0000000000125056e-02` |
| 19 | 42 / 8 | 0.0000000000000000e+00 | 6.0440950619522482e-05 | -5.0000000000011369e-02 | -2.7586356736725883e-07 | `-5.0000000000011369e-02, -5.0000000000011369e-02, -4.9999999999897682e-02, -5.0000000000068212e-02` |
| 20 | 238 / 69 | 0.0000000000000000e+00 | 1.8143804726572961e-17 | -5.0000000000011369e-02 | 1.1942999120165839e-20 | `-5.0000000000011369e-02, -4.9999999999954525e-02, -5.0000000000011369e-02, -5.0000000000011369e-02` |
| 21 | 274 / 265 | 2.0938614857594729e-14 | 2.0441766083240509e-01 | 5.6843418860808015e-14 (浮点近零) | -1.2761017218507024e-15 | `3.9790393202565610e-13 (浮点近零), -5.6843418860808015e-14 (浮点近零), 0.0000000000000000e+00 (浮点近零), -1.1368683772161603e-13 (浮点近零)` |
| 22 | 328 / 307 | 0.0000000000000000e+00 | 7.5787720543729620e-10 | -5.0000000000011369e-02 | -2.0763946118056711e-13 | `-5.0000000000011369e-02, -5.0000000000039790e-02, -4.9999999999982947e-02, -5.0000000000068212e-02` |
| 23 | 418 / 264 | 0.0000000000000000e+00 | 2.8082777280360460e-03 | -5.0000000000011369e-02 | -3.2498734071858168e-06 | `-5.0000000000011369e-02, -5.0000000000011369e-02, -4.9999999999982947e-02, -4.9999999999982947e-02` |
| 24 | 616 / 69 | 0.0000000000000000e+00 | 3.8778381422162056e-03 | -4.9999999999982947e-02 | 7.0226495154178037e-06 | `-4.9999999999954525e-02, -4.9999999999982947e-02, -4.9999999999997158e-02, -4.9999999999982947e-02` |

`m` 是每个 state 的 local alignment 项，不将 24 个 m 相加解释为 global gain。named group JSON 中若报告 baseline branch outcome aggregate，则它是重复 counterfactual 场景的累加计数，不是请求总体样本量，也不作为总体比例。failure-mechanism 仍保持独立的 source16 / batch-02 episode-24 记录，不并入本表。

## P3 energy-failure mechanism completed

服务器 artifact `/home/user/training_outputs/terminal/p3_followup_20260930/failure_mechanism/diagnose.json` 已报告 `COMPLETED`：3 个状态共 `312` 个 branch，全部为 `natural_terminal` 且 uncensored；model、environment 和 caller RNG 均保持不变。

| state | step | SOC | C | p | D | m | full D by continuation seed |
|---|---:|---:|---:|---:|---:|---:|---|
| F01 | 280 | 0.398656 | 0 | 8.28291e-5 | -0.05 | +2.7650e-8 | `[-0.05, -0.05, -0.05, -0.05]` |
| F02 | 445 | 0.288656 | 1.1869090464568813e-14 | 0.000650799135 | +2.30523854878933 | -0.0002574839567 | `[0.593587408892546, -0.05, -0.05, 8.72736678626487]` |
| F03 | 726 | 0.110987 | 0 | 2.21449e-5 | -0.05 | +1.58285e-7 | `[-0.05, -0.05, -0.05, -0.05]` |

Full-return `D` is `paired.expected_release_undiscounted_return_by_repeat[i] - wait[i].undiscounted_return`; the mean check was within `1e-8`. Target arrival `14` was evaluated under WAIT and true q-weighted RELEASE. RELEASE outcomes by seed were F01: `9303201`, `9303202`, `9303204` `emergency_landed`, `9303203` `failed_energy`; F02: all four `emergency_landed`; F03: all four `failed_energy`. The F02 raw `C` is `1.1869090464568813e-14`; the earlier `C=0` display was rounded. Proposals `22`, `23`, and `24` were approved, each with K=3 runtime rank `1`; proposal `11` carried approximately `99.999999986%` of the true q mass. The new blocker first-step record agrees with the old paired first-step record; F02 is not described as 25 proposals all rejected.

第726步的一次gate干预不能在冻结后续策略下挽救该机；第445步续跑仍可应急落地，需要核对等待期资源/安全/能量拒绝原因，尚未选定训练修复。one-step check 不产生 `D`。该诊断未使用 PPO 或 held-out。

## Bounded blocker telemetry completed

历史启动记录：初次 PID `2025086` 因工作目录缺少项目包产生
`ModuleNotFoundError: No module named 'terminal_marl'`；该次真实 phase 未开始，
没有产生 phase outputs。`blockers_v1.log` 保留该旧错误；纠正 project cwd 后，
同一 blocker phase 产生了以下完成产物，旧错误不指向当前完成结果。

`firststep_blockers.json` 报告 `78/78` 首步分支成功、`0` 个错误，三个固定
状态各 `26` 个分支。`original_auto_emergency_trace.json` 报告 exact `984`
步、`24` requests 和 terminal rows，保留 failed UID `57`、time `892`，
source unchanged，target calls `878`、trace errors `0`。`manifest.json` 和
`progress.json` 均为 `COMPLETED`。

原始 trace 的 `878` 个 target calls 中有 `102` 次实际 attempt。pad-attempt
reason counts 为 `blocked_lz=5`、`blocked_safety=96`、`blocked_energy=226`、
`blocked_dynamics=0`、`committed=0`；这些是 pad 尝试的原因次数，不能求和当作
调用次数或全局收益。首次 attempt 在 step `278`、SOC `0.399989`、energy
`599.984`，compatible-pad availability 为 `4/0`；其中
`blocked_safety=2` 的 reason 包含 `predicted_holder_reachable_tube`。首次
energy rejection 在 step `656`、SOC `0.157654`、energy `236.481`，
compatible-pad availability 为 `5/3`，reason counts 为 `energy=1`、
`safety=1`。所有 native calls 均未成功；UID `57` 在 `892` 仍为
`failed_energy`。`attempt=false` 不计作 reject。

源码和 tests 的本地/server SHA 一致，focused suite 为 `12 passed`（commit
`e3e13d7`）。全局 `PPO_ALLOWED=false`，未使用 held-out；该 phase 不改变
physics、guard、model 或 optimizer，也不宣称修复或新增因果归因。

原始失败回放显示安全拒绝先发生，后期出现能量拒绝；step `445` 仍存在极低
q 的可执行 K=3 分支，因此不将该时刻描述为不可行。这只是单次 failure 加
frozen continuation，尚未选择训练修复，P3 仍未解决。

## 边界

pilot 测量已完成，尚无性能改善结论。P3 退化是否解决仍未由本轮回归测试和 replay 证明；strong baselines、sensitivity、joint recursive feasibility、完整 V4.2 high adapter 和 qualification 仍是未关闭事实。`C` 的完整 `24×4` 诊断已 `COMPLETED`：24/24 state、2,496/2,496 branch，自然终止且无 error/censor；`PPO_ALLOWED=false` 保持。已观测到 14 个纯 `safety_wait` state，root 已选择独立的 frozen `legacy`/`pre_action_forced_wait` 对照，范围为 24 个 complete-scene seed pairs；readiness review corrections 进行中，尚未启动 server，PPO 保持 false。
