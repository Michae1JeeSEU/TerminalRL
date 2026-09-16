# 项目当前状态

> **状态日期**：2026-09-17
>
> **代码快照**：分支 `codex/v4-2-v2-1-formal-i1`，R4 预注册提交 `1fb0105`
>
> **当前阶段**：Phase A / A5 固定空域低层策略，expert dataset qualification
>
> 本文是滚动状态页。规范语义以 [README.md](README.md) 中列出的主规范为准。

## 一句话状态

I0-R4 全部 12 个 gate PASS：训练 7,576 rows / 208 episodes，审计 1,886 rows / 52 episodes，52/52 replay 一致。I1 V2 三 seed 与 fresh Q2 已全部 PASS，按 BC → Safe DAgger → student-only qualification → PPO admission 推进；PPO_ALLOWED 仍为 false。旧 R3 和 I1 的 FAIL 不变。

## 当前冻结的主线语义

| 项目 | 当前值 |
|---|---|
| 研究范围 | Phase A，landing-only；Phase B/C 仅为未来路线 |
| 代码主线 | `terminal_marl/`；`src/` 只读参考 |
| 空域 | `three_ring_v1`：OUTER / MIDDLE / INNER |
| 研究机型 | `eh216s_research_v1` |
| 动力学 | `performance_constrained_v2`；`physics_dt = 0.2 s` |
| 低层观测 | `phaseA_low_obs_v2_1_retry_state`，88D |
| 低层动作 | `low_action_schema_v4_2_hierarchical` |
| 低层分布 | `WAIT/RELEASE` 二分类 gate + RELEASE 条件下 25-way proposal |
| 执行解码 | 1 个 canonical WAIT + 25 个 `speed × merge-time` RELEASE 意图 |
| 高层动作 | `high_airspace_v1_10d`，当前训练关闭 |
| 安全权威 | TrajectoryGenerator → SafetyFilter → PredictiveSafetyShield |
| Teacher 角色 | 训练期 expert labeler / auditor；不是部署控制器或 PPO behavior action |

低层完整语义见 [reference/low_level_policy.md](reference/low_level_policy.md)。

## 里程碑与证据

| 里程碑 | 状态 | 说明 / 证据 |
|---|---|---|
| Phase A 核心环境与 rule pipeline | 已进入后续训练使用 | 具体接口见主规范；历史迁移过程不再作为状态源 |
| D0–D7 动力学 qualification | PASS | [正式报告](evidence/dynamics/dynamics_qualification_report_2026-08-28.md) |
| P0 仿真性能 qualification | PASS | 结果摘要保留在 [历史升级计划](history/plans/CRAZYFLIE_DYNAMICS_LAYER_UPGRADE_PLAN.md) 末节 |
| C0 capacity calibration | PASS | 作为 user-provided frozen evidence 记录于 [V4 migration summary](evidence/v4/v4_new_mainline_migration_summary.md) |
| V4.2 hierarchical S0 | PASS | [S0 qualification](evidence/v4_2/v4_2_s0_qualification_summary.md) |
| Observation V2.1 final freeze | PASS | [final freeze v2](evidence/v4_2/v4_2_v2_1_final_freeze_decision_v2.md) |
| Fresh V2.1 I0-R | PASS | [I0-R qualification](evidence/v4_2/v4_2_v2_1_formal_i0_r_summary.md) |
| Fresh multi-seed I1 | FAIL | RELEASE manifold/context generalization gap；[qualification](evidence/v4_2/v4_2_v2_1_i1_formal_qualification_summary.md) 与 [attribution](evidence/v4_2/v4_2_v2_1_i1_failure_attribution.md) |
| I0-R2 repair collection | FAIL | 未闭合所需 support；协议和结果保存在 `evidence/v4_2/` |
| I0-R3 repair collection | FAIL（历史） | [R3 result](evidence/v4_2/v4_2_v2_1_i0_r3_collection_summary.md) |
| I0-R4 repair collection | PASS | [正式结果](evidence/v4_2/v4_2_v2_1_i0_r4_collection_summary.md) |
| I1 Qualification Protocol V2 / Q2 | PASS | [正式结果](evidence/v4_2/v4_2_v2_1_i1_v2_summary.md)，Q2 1,787 rows，三 seed 全通过 |
| I2 Safe DAgger | R1 PASS / I3 PREREGISTERED | [I3 协议与 R1 结果](evidence/v4_2/v4_2_v2_1_student_qualification_protocol.md)，976 新事件、35 epochs、全部 gate PASS |
| PPO | `PPO_ALLOWED = false` | 需要 student-only imitation qualification 和 PPO admission checklist 全部通过 |
| A6 高层动态空域训练 | BLOCKED | 需要低层 static-geometry qualification 后冻结低层策略 |
| Crazyflie 实物验证 | FUTURE | 仅在主仿真策略冻结后进入 small-N 缩尺验证 |

## 最新失败：I0-R3

I0-R3 的 provenance、Teacher determinism、exact replay、Safety、partition independence、
aliasing 与 nominal anchors 均通过，但正式结果仍为 FAIL：

- complete-cycle 停止产生 4,541 个训练 rows 和 1,511 个内部审计 rows，超过冻结上限
  4,200 / 1,400；
- proposal-14 的 N=5 支持只有 12 个独立 lifecycle，低于最低 15；
- shared gate false-negative、shared proposal error，以及 post-WAIT、transition/retry、
  post-conflict、proposal-11、proposal-14 的 radius coverage gate 全部未达预注册阈值。

因此不能把 R3 标记为“基本通过”，也不能直接进入新 I1。完整数值见
[I0-R3 collection summary](evidence/v4_2/v4_2_v2_1_i0_r3_collection_summary.md)。

## 当前允许的下一步

- R4 正式结果与哈希见 [R4 summary](evidence/v4_2/v4_2_v2_1_i0_r4_collection_summary.md)。
- I1 V2 保持原 BC 超参数与 hard gates；全新 actor seeds 为 6201/6202/6203。
- Q2 固定 45 episodes，seeds 430001–430045，先完成历史独立性检查并冻结；全部 actor 训练冻结后只评估一次。
- 新入口及采集器 11 项测试通过；2 episodes / 9 rows smoke 的 schema、Teacher determinism、replay、Safety 和 aliasing 均通过。smoke 不产生正式准入。
- I1 V2 已完成输出：`~/training_outputs/terminal/v4_2_i1_v2_formal_20260917`，同名 `.log`。
- DAgger R1 已通过：D1 共 8,552 rows、12 个采集和 4 个验证 episodes，Safety 与 replay 全通过。
- I3 预注册六个 fresh seed，warm3 / flow3 / flow6，确定性与随机 student-only，各最多 900 秒；仅面向初始 N3–N6 课程准入。
- 已有独立 StudentPPOV42 和 88D rollout；尚需完成 DAgger、student-only qualification、PPO admission 与正式训练入口验证。
- 按用户要求，长任务由 Luna high 子 agent 监控，主窗口等待；终态返回后继续下一 gate。
- 本地 Git 无 remote；仅通过 Syncthing 同步服务器，不能声称 Git 推送成功。

```text
I1 V2 / Q2 PASS → 88D I2 preflight
→ I2 Safe DAgger → student-only qualification → PPO admission → training entry verification
```
## 不应恢复的旧路线

- V3 factorized 三头策略；
- V4 30-way 与 V4.1 flat-26 actor；
- FULL-Teacher PPO 或用 Teacher corrected action 计算 requested-action PPO ratio；
- 87D V2 checkpoint 直接续训 88D V2.1；
- 两环、8D high-level、自由 `ψ/γ/V` 或姿态/电机 action；
- 把日期型 handoff 或单次 diagnostic 当作新的 canonical specification。
