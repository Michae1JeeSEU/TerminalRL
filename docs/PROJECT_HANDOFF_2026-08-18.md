# Terminal MARL 项目交接（2026-08-18）

> 用途：为下一次 Codex 对话提供项目状态、已验证证据、冻结约束与下一步。
> 这是一份交接记录，**不是**项目的 canonical specification；设计/实现判断仍只以本文末尾列出的六份 canonical docs 为准。

## 1. 项目与工作约束

- 主开发线：`terminal_marl/`。`src/` 是 legacy 原型，不要在那里新增主功能。
- 本地代码目录：`D:\7_codespace\terminal`。
- 服务器：`ssh lab-server`；代码目录：`~/projects/terminal`；环境：`source ~/envs/terminal_env/bin/activate`。
- 源码通过 Syncthing 同步。本地编辑/提交；服务器只运行测试、评估和读输出。不要 scp/rsync 源码。
- 服务器输出根目录：`$HOME/training_outputs/terminal`，同步到本地 `D:\7_codespace\training_outputs\terminal`。
- Git 只在本地操作；显式暂存本轮文件；当前未配置 Git remote，因此不能/不应声称已 push。
- 任何耗时实验优先 staged/fail-fast；独立 rollout 可在服务器多进程并行。不要因为性能改动而降低安全阈值。

## 2. Canonical source of truth

以下六份文件是用户明确指定的唯一 canonical source of truth：

1. `docs/research_positioning.md`
2. `docs/requirements.md`
3. `docs/design_spec.md`
4. `docs/technical_spec.md`
5. `docs/execution_plan.md`
6. `docs/revision.md`

仓库有大量历史/用户未提交改动，尤其是 `src/`、旧 Phase 3 脚本及六份 docs 本身；不要丢弃、覆盖或把它们混入新的提交。开始任何编辑前先查看 `git status --short`。

## 3. 已完成的核心架构（A0–A6.13）

### A0–A4：landing-only 基础环境

- 只实现 Phase A landing-only；没有旧 two-loop、旧 4D low-level action、旧 checkpoint 兼容分支。
- `ExternalQueue` 用 mutable `QueuedArrivalState` 包装 immutable `ArrivalRequest`；排队期间更新 energy/SOC、wait、dynamic priority；admission 后才创建 `UAVState(INBOUND)`。
- `initial_SOC` 可配置，默认 `Uniform[SOC_reserve, 1.0]`，在相同 seed/CRN 下可复现。
- 三环结构：Outer/Middle/Inner；四个真实 entry gates；六个 LZ 默认全部兼容四种 mission type。
- 任务 service time：standard/cargo/emergency/surveillance = 30/120/10/60 s。
- touchdown 后 UAV 退出 low-level active-agent 集合；`LandingZoneManager` 独立推进 SERVICE；ledger 的正式 COMPLETED 以 service completion 为准。低层 terminal reward 是 touchdown，不给已经终止的 actor service reward。
- canonical trajectory generator、merge scheduler、SafetyFilter、runtime safety 都在 `terminal_marl/core/` 和 `terminal_marl/envs/parallel_env.py`。

### A5：shared low-level PPO（已冻结）

- 低层唯一 action：`[release_request, speed_residual, merge_time_residual]`（3D），不允许增加维度。
- shared actor、variable-N、centralized critic（masked mean/max pooling）；不使用 attention/GNN。
- A5 checkpoint 已通过 gate，必须冻结；不要训练、替换或修改其算法，除非发现明确实现 bug。
- 低层 terminal success 一律 touchdown；service completion 仅用于 global/high-level/outcome metrics。

### A6.1–A6.2：动态高层、安全闭环（已冻结）

- canonical high-level action：10D；180 s decision interval。
- 结构 6D（R/H）受 event gate 和 reconfiguration coordinator 管理；speed-profile + 三个 headway 可独立 prospective apply，且对既有 committed reservation 不追溯失效。
- `ReconfigurationSafetyCoordinator` 语义已冻结：traffic-repair-first → predictive verification → delay → minimal projection → reject last。
- 高层 pending action state machine 已修复，避免每秒 retry storm。
- `RuntimeSeparationMonitor` 是运行时唯一权威实际/扫掠 separation 指标来源：minimum actual/swept separation、unique hard separation、unique NMAC、unique collision 与 exposure steps。
- 不得改动 A6.8/A6.2 冻结的 safety threshold/semantics；特别是 `safe_distance=80 m` 不得降低。

### A6.11：初始因果审计（已有证据）

- 结论：原始 scheduler 下 Outer/Middle/Inner capacity 基本不 binding；大量 O2M/M2I/I2F release 被 safety_wait / safety_reject 阻塞。
- 冻结 8 个 A6.11 rejected states 中：2 个 immediate safe repair、6 个 SAFE_AFTER_DELAY（5–20 s），均在现有 ±2 m/s、±6 s authority 内找到 witness。
- 结论为 `SAFETY_SCHEDULER_CONSERVATISM`；PPO STOP。
- 产物：`training_outputs/terminal/event_gated_stage15_diagnostics/a611/`。

### A6.12：Prospective Safe Merge Scheduling & Repair Integration（本轮完成）

实现提交：

- `6ef38b0 feat: add prospective safe merge repair scheduler`
- `961b395 perf: stage prospective repair screening`
- `53201ac docs: summarize scheduler repair audit evidence`

主要实现：

- `MergeScheduler.repair_candidates()` 按最小干预次序搜索已有合法 authority：requested → immediate merge-time → immediate speed → immediate combination → bounded future delay + repair。
- residual 不超过原有边界：merge time ±6 s、speed ±2 m/s；没有新的 RL action 或自由飞行 heading/pitch 控制。
- 对每个 candidate 使用 canonical `TrajectoryGenerator` 和 authoritative swept `SafetyFilter`。
- `SAFE_AFTER_DELAY` 会变成一次 future reservation；source-ring hold 被写入 canonical plan（包括 delayed I2F final plan），在实际开始时再原子复查并执行同一 canonical plan。
- `checked_plan_id == executed plan_id` 被显式核验。
- release telemetry：唯一 `release_intent_id`、first attempt、approval、retry、scheduled delay、repair flag、requested/applied residual、plan id、decision history；primary blocker 互斥分类。
- 每个 release intent 最多一次 expensive forecast，避免以 forecast 形式重建 retry storm；按 `(delay, speed residual)` 共享 source-ring canonical forecast，再选择全局最小干预成功 candidate。

新增/重点文件：

- `terminal_marl/core/config.py`
- `terminal_marl/core/merge_scheduler.py`
- `terminal_marl/core/trajectory_generator.py`
- `terminal_marl/envs/parallel_env.py`
- `terminal_marl/training/a612_scheduler_repair_audit.py`
- `tests/test_a612_scheduler_repair.py`

证据（`newly_run`）：

- server pytest：A6.12/A5/A6.11 组合 `16 passed`。
- frozen-state regression：**8/8** A6.11 witnesses 在线找到 repair；所有 replay hard-safe。
- 5 个 CRN、RuleLowLevelPolicy、fixed nominal geometry（Outer/Middle/Inner capacity 47/31/15）、900 s demand window + 240 s drain 的 mechanism screen：
  - protected metrics 均为 0：collision、unique NMAC、hard separation violation、final overcapacity。
  - `constant_high`：O2M approval +14.3pp，retry 显著降低；completed 12→12，all-request waiting area 1,955,389.9→2,002,898.8（变差）。
  - `low_high_low_ramp`：completed 10→10，waiting 无变化；多数拒绝在有限 legal grid 内不可行。
  - `medium_burst`：completed 13→12，waiting 2,537,685.4→2,505,549.8（改善）。
  - `long_burst`：同 medium-burst（900 s window 只覆盖相同前缀）。
  - `double_burst`：completed 12→12，waiting 2,436,083.0→2,533,135.2（变差）。
- A6.12 最终分类：`SCHEDULER_CONSERVATISM_CONFIRMED`；这是“安全 approvals/部分 waiting 改善”的机制证据，**不是** PPO 或终端吞吐提升证明。
- PPO authorization：**STOP**。

正式 A6.12 产物：

- `training_outputs/terminal/event_gated_stage15_diagnostics/a612/a612_scheduler_repair_audit.json`
- `.../a612_frozen_rejection_regression.csv`
- `.../a612_interface_flow_comparison.csv`
- `.../a612_terminal_performance_comparison.csv`
- `.../a612_summary.md`

### A6.13：repair-aware capacity-value re-audit（本轮完成的 pilot）

实现提交：

- `e312529 feat: add repair-aware capacity re-audit`

文件：

- `terminal_marl/training/a613_capacity_reaudit.py`
- `tests/test_a613_capacity_reaudit.py`

方法：

- 固定 RuleLowLevelPolicy + A6.12 repair-aware scheduler；无 PPO。
- 使用 A6.11 的 μ0 preexisting calibration artifact，constant-high pilot、seed 0。
- 先 run repair-aware nominal；之后用 6 worker 并行跑 Outer/Middle/Inner 的 static capacity +1 与 nominal→dynamic capacity expansion。
- capacity 仍仅由合法 geometry 导出；不新增 capacity action。
- 所有新 branch fail-closed：任一 safety 失败即 audit 失败。

证据（`newly_run`）：

- pytest：A6.13/A6.12/A6.11 组合 `10 passed`。
- 7 个 branch 全 hard-safe，所有 collision/NMAC/hard separation = 0。
- nominal：completed 12，waiting area 1,955,897.8。
- Outer static +1：completed 12，waiting 1,930,420.1（约 1.3% 改善）。
- Outer dynamic +1：completed 12，waiting 1,955,897.8（没有兑现 static gain）。
- Middle/Inner static/dynamic 没有价值，部分更差。
- per-ring conclusion：Outer = `RECONFIGURATION_COST_OFFSETS_GAIN`；Middle/Inner = `CAPACITY_NOT_BINDING`。
- overall A6.13 result：`RECONFIGURATION_TIMESCALE_MISMATCH`；PPO authorization **STOP**。

正式 A6.13 产物：

- `training_outputs/terminal/event_gated_stage15_diagnostics/a613/a613_capacity_reaudit.json`
- `.../a613_static_dynamic_comparison.csv`
- `.../a613_summary.md`

## 4. 当前明确结论与禁止事项

### 当前 bottleneck 结论

1. A6.11 证明原始系统主要被 scheduler/safety search conservatism 阻塞，而不是环容量持续饱和。
2. A6.12 证明在不放宽 safety 的前提下，合法 timing/speed repair 能提高部分 safe approvals、减少 retries；但无法稳定提升 completed，且不同 demand profile 有 trade-off。
3. A6.13 证明修复 scheduler 后，只有 Outer static +1 在 pilot 中表现出小的 waiting benefit；dynamic Outer expansion 没有兑现该 benefit。
4. 因此下一根因是 **reconfiguration transition cost / timing / event-gate mismatch**，不是“加 PPO steps”或“降低 80 m separation”。

### 仍然禁止

- 不训练任何 PPO；不启动 A7/A8。
- 不做 simultaneous high/low joint training 或 alternating training。
- 不修改 A5 checkpoint、当前 A6 checkpoint、A6.2/A6.8 safety semantics/thresholds。
- 不降低 80 m separation；不得以吞吐换取 collision/NMAC/hard-separation failure。
- 不增加低层 action dimension；仍严格为 3D intention action。
- 不以 full long 4-policy matrix 开始下一步；保持 fail-fast mechanism funnel。

## 5. 下一步建议：A6.14（尚未实现，需先做机制诊断）

在没有新的用户要求覆盖时，建议下一对话先定义并执行 **A6.14 — Outer reconfiguration timing/cost causal audit**，仍不训练 PPO。重点不是重新调 policy，而是确定 Outer static +1 的收益为什么被 dynamic transition 抵消：

1. 用相同 CRN、repair-aware scheduler，比较 `nominal`、`outer static +1 from t=0`、`outer dynamic +1`。
2. 扫描 dynamic request 时点（例如 demand start、pre-peak、peak、post-peak）和 legal transition/dwell windows；只做少数 representative points，先 seed 0 fail-fast。
3. 分解损失：interface frozen duration、traffic repair magnitude、future reservation waits、defer/project/reject、transition delay、outer holder time、queue area。
4. 对 dynamic request 做 counterfactual：idealized “already at target” static 对照与真实 transition 对照；二者差就是 transition cost 上界。
5. 不得把 committed traffic 改轨；使用当前 coordinator 的 canonical repair authority。
6. 仅当存在跨时点、hard-safe、可复现的 dynamic benefit，再考虑重新授权 high-level PPO。否则继续 `STOP_BEFORE_PPO` 并把 high-level control 结论降级为 timescale mismatch。

这一步的具体协议未由用户单独 canonical 化；新对话应先核对六份 canonical docs 与本交接，再确认是否按该建议推进。

## 6. 常用命令

```bash
# 服务器环境
ssh lab-server
source ~/envs/terminal_env/bin/activate
cd ~/projects/terminal

# A6.12 tests
python -m pytest tests/test_a612_scheduler_repair.py tests/test_a5_shared_ppo.py tests/test_a611_causal_audit.py -q

# A6.13 tests
python -m pytest tests/test_a613_capacity_reaudit.py tests/test_a612_scheduler_repair.py tests/test_a611_causal_audit.py -q

# 读取结果
python -m terminal_marl.training.a612_scheduler_repair_audit --help
python -m terminal_marl.training.a613_capacity_reaudit --help
```

## 7. Git 状态与最近提交

最近相关提交（均在本地 `master`）：

```text
e312529 feat: add repair-aware capacity re-audit
53201ac docs: summarize scheduler repair audit evidence
961b395 perf: stage prospective repair screening
6ef38b0 feat: add prospective safe merge repair scheduler
7d3a8bf feat: add demand-capacity-safety causal audit
2391544 feat: audit objective hierarchy and pareto robustness
```

本轮结束时没有配置 remote。工作树中存在大量不相关的用户历史修改和 untracked files；任何新提交只暂存本轮明确文件。
