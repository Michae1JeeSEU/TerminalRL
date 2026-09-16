# 项目当前状态

> **状态日期**：2026-09-16
>
> **代码快照**：分支 `codex/v4-2-v2-1-formal-i1`，整理前 HEAD `d46daab`
>
> **当前阶段**：Phase A / A5 固定空域低层策略，expert dataset qualification
>
> 本文是滚动状态页。规范语义以 [README.md](README.md) 中列出的主规范为准。

## 一句话状态

三环 landing-only 环境、EH216-S 性能约束动力学、轨迹/安全链和 V4.2/Observation
V2.1 schema 已建立并通过各自的基础 qualification；当前瓶颈是低层模仿学习所需的
RELEASE 决策流形覆盖。I0-R3 数据集未通过预注册 gate；2026-09-16 已完成停止规则、
场景配置和距离指标诊断，预注册 I0-R4 修复。新的 I1、DAgger、PPO 以及高层动态空域
训练仍须等待对应前置 gate，不能以 R4 已启动代替 PASS。

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
| I0-R3 repair collection | **FAIL（最新）** | [R3 result](evidence/v4_2/v4_2_v2_1_i0_r3_collection_summary.md) |
| I1 Qualification Protocol V2 / Q2 | BLOCKED | 只有 fresh I0 repair PASS 后才能预注册并采集 |
| I2 Safe DAgger | BLOCKED | 需要新的 I1 qualification PASS |
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

### 2026-09-16 修复执行

- [R4 协议与课程判断](evidence/v4_2/v4_2_v2_1_i0_r4_repair_protocol.md)：保留
  BC → Safe DAgger → student-only → PPO 主线；修正不兼容的整轮停止预算。
- R3 每轮约 756 rows，C05 仅 2 lifecycles/轮；原 4,200 rows 无法容纳所需约 8 轮。
- R4 固定 208 个训练 episodes、52 个内部审计 episodes；真实场景支持、安全、replay、
  schema 与独立性仍为 hard gates。88D radius 仅作为新协议的诊断项，R3 FAIL 不变。
- R2/R3 文档搬迁路径已修复；14 个定向测试通过。R4 小样本采集、序列化、
  88D/retry provenance、Teacher determinism、4/4 exact replay 和 Safety 已验证。
  smoke 不满足正式样本量且不产生准入授权。
- I2 仍需接入 88D 新 lineage；V4.2 actor 尚无 PPO update 实现，不能直接运行旧 V2 PPO。
- 当前允许启动 R4；仅 R4 全 gate PASS 后登记 I1 V2 和 untouched Q2。

1. 先诊断 R3 的“场景可达性 / 完整 cycle 停止规则 / 88D 距离 gate”为什么同时失配；
2. 若需要新一轮数据修复，必须在采集前单独预注册新协议、seeds、上限和 qualification gate；
3. 只有新的 fresh I0 repair 全部 PASS，才能依次执行：

```text
preregister I1 Qualification Protocol V2
→ collect and freeze untouched Q2
→ train fresh multi-seed V4.2 actors
→ evaluate Q2 once
→ I2 Safe DAgger preflight
→ student-only imitation qualification
→ PPO admission review
```

4. 当前不得通过改 Teacher、Safety threshold、observation/action schema、class weighting、
   row duplication 或消费旧 Q cohort 来绕过失败；若研究判断确实要求改 schema，必须新开明确版本。

## 不应恢复的旧路线

- V3 factorized 三头策略；
- V4 30-way 与 V4.1 flat-26 actor；
- FULL-Teacher PPO 或用 Teacher corrected action 计算 requested-action PPO ratio；
- 87D V2 checkpoint 直接续训 88D V2.1；
- 两环、8D high-level、自由 `ψ/γ/V` 或姿态/电机 action；
- 把日期型 handoff 或单次 diagnostic 当作新的 canonical specification。
