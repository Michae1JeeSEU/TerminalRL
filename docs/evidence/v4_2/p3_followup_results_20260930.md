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

后续 `C` 固定为 `24×4`，由 4 个进程按 modulo 分片；这只是计算调度，不增加实验 factor、不重抽样、不删除失败 branch。`C` 已启动并处于 `RUNNING`，4 个分片进程 PID 为 `2007505`、`2007601`、`2007551`、`2007650`，当前尚未完成；`PPO_ALLOWED=false`，未消费 held-out。此前 `82` 个集成测试只作为已记录 raw count，不在本节重复累计。

## C 初步结果

raw artifact `causal/shard_0/diagnose/state_0001.json` 当前包含同一首状态的 `104` 个分支，来自 `4` 个 continuation seeds，均为 `natural_terminal`；model、environment 和 caller RNG 均保持不变。该状态的 `C=0`、`p=0.7176065`、平均 `D=6.56333546`、平均 `m=0.06023505`；四个 seed 的 `D` 为 `[-1.41627285, -0.05, 11.62504392, 16.09457077]`。

同一首状态四种续跑的符号混合，单seed pilot负值不能用于根因判定；C=0仅表示即时覆盖为零，不能据此认定全周期动作无效或直接强制WAIT。完整24状态及失败机制尚未完成，不构成性能改善证据。

## 边界

pilot 测量已完成，尚无性能改善结论。P3 退化是否解决仍未由本轮回归测试和 replay 证明；完整 causal paired outcomes、strong baselines、sensitivity、joint recursive feasibility、完整 V4.2 high adapter 和 qualification 仍是未关闭事实。`PPO_ALLOWED=false` 保持。
