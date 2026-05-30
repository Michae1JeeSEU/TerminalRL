# 执行计划

> **2026-05-18 状态**：Phase 1-4（初版）→ Phase 2(R)/3a(R)/3b(R)/3b.5(R)（修订版）全部完成。
> 详见修订版计划：`C:\Users\user\.claude\plans\bubbly-percolating-graham.md`。
>
> **已完成**：
> - Phase 2(R)：轨道盘旋 + (ψ,γ,V) 连续下降 + per-UAV 局部观测 + 4进入点 + 流入模型（39 测试通过）
> - Phase 3a(R)：单 UAV + 干扰机 PPO（100% 着陆率 vs 10 背景干扰）
> - Phase 3b(R)：全 UAV 受控清空空域 PPO + LZ 占用机制
> - Phase 3b.5(R)：SafetyController 反射式碰撞避免（5 规则）+ 5 项安全修复
>   - 碰撞移除（不终止 episode）+ 轨道 Snap 安全检查 + 环容量限制 + 距离接近惩罚 + SubprocVecEnv
>   - N=30 着陆率：PPO 79% / Heuristic 81%（从 17%/20% 提升）
>
> **2026-05-28 安全修订**：针对无持续流入阶段仍发生碰撞的问题，低层动作空间由 `Box(max_n,5)` 收缩为 `Box(max_n,2)`，RL 只请求 DESCEND/CIRCLE 与盘旋速度；环境规则层负责下降航迹、放行计时、预测安全过滤、同环纵向间隔和目标环捕获。N=30 无流入诊断中 Heuristic 与旧 PPO 兼容各 5 个 seed 均达到 30/30 着陆、0 failed、0 conflicts。
>
> **2026-05-29 安全学习修订**：针对多规模诊断中 PPO 近距预警显著高于 Heuristic 的问题，引入 SafetyController 介入惩罚、显式间距观测和 100m 连续安全裕度塑形。观测维度由 79/旧 checkpoint 输入升级为 85，Phase 3b PPO 需要重新训练后再评估。
>
> **2026-05-30 多规模训练修订**：随机 N=10-40 直接训练出现 timeout 和近距预警爆炸。原因包括固定 N eval wrapper 未写回 `config.num_uavs`、多环境共享 config 的潜在污染、大 N 下近距/安全惩罚按 pair 数放大而着陆奖励被 N 稀释，以及 max_steps 被误报为 terminated 而非 truncated。已改为课程式多规模采样、固定 N reset 修复、config 深拷贝、近距惩罚按 N 归一化、跨 N 稳定着陆奖励，并取消普通 DESCEND 放行阻塞惩罚。

## 总体原则

- **增量推进**：一个 Phase 完成并验证后，再启动下一个
- **每 Phase 结束**：更新本文档状态，记录实际完成情况与偏差
- **遇问题即停**：不确定的点先讨论再动手

---

## Phase 1: 项目基础设施 ✅ 进行中

| # | 任务 | 输入 | 输出 | 完成标准 | 状态 |
|---|------|------|------|----------|------|
| 1.1 | 创建目录结构 | - | docs/, dev_logs/ | 目录存在 | ✅ |
| 1.2 | 编写需求文档 | 用户确认的需求 | docs/requirements.md | 用户审阅通过 | ✅ |
| 1.3 | 编写技术规范 | 需求文档 | docs/technical_spec.md | 用户审阅通过 | ✅ |
| 1.4 | 编写设计规范 | 需求+技术文档 | docs/design_spec.md | 用户审阅通过 | ✅ |
| 1.5 | 编写执行计划 | 全部前置文档 | docs/execution_plan.md | 本文档完成 | ✅ |
| 1.6 | 创建首日日志 | - | dev_logs/2026-05-13.md | 日志就位 | ⬜ |
| 1.7 | 创建 CLAUDE.md | 全部文档路径 | CLAUDE.md | 用户审阅通过 | ⬜ |
| 1.8 | 最终确认 | 全部产出 | - | 用户确认进入 Phase 2 | ⬜ |

---

## Phase 2: Gym 环境搭建 ✅ 已完成

| # | 任务 | 说明 | 依赖 | 状态 |
|---|------|------|------|------|
| 2.1 | 实现 RingAirspace | 三层环模型 + 硬约束 + 体积内随机位置 | Phase 1 | ✅ |
| 2.2 | 实现 UAV 运动学 | 质点模型 + 能耗计算 + 混合优先级 | Phase 1 | ✅ |
| 2.3 | 实现 UAVManager | 多 UAV 批量管理 + 初始环分布 + 随机状态 | 2.2 | ✅ |
| 2.4 | 实现 ConflictDetector | 安全间隔检测 | 2.2 | ✅ |
| 2.5 | 实现 PriorityEngine | 混合优先级计算 | 2.2 | ✅ |
| 2.6 | 实现 MetricsCollector | 指标收集汇总 | - | ✅ |
| 2.7 | 实现 TerminalAreaEnv | Gym 环境主循环 + 整合全部模块 | 2.1-2.6 | ✅ |
| 2.8 | 单元测试 | 各模块测试 | 2.1-2.7 | ✅ |
| 2.9 | 集成测试 | 环境 reset/step 测试 | 2.8 | ✅ |
| 2.10 | 编写环境说明文档 | 使用方法与配置说明 | 2.7 | ✅ |

**实际偏离**：
- 动作空间从设计阶段的 per-UAV 独立动作改为 shared-action（所有 UAV 共享同一控制），作为简化基线。per-UAV 独立动作留待后续实现。
- 初始分布已实现：UAV 分布在三个环中（50%/30%/20%），初始电量、等待时间、状态随机化。

---

## Phase 3: 单 UAV 基线 ✅ 已完成

| # | 任务 | 说明 | 依赖 | 状态 |
|---|------|------|------|------|
| 3.1 | 简化环境（单 UAV） | 屏蔽冲突检测，1架 UAV | Phase 2 | ✅ |
| 3.2 | 实现随机策略基线 | 随机动作性能下界 | 3.1 | ✅ |
| 3.3 | 实现启发式规则基线 | 最短路径优先 / 先到先服务 | 3.1 | ✅ |
| 3.4 | 单层 PPO 训练 | 仅低层决策，空域固定 | 3.1 | ✅ |
| 3.5 | 基线对比分析 | 各策略指标对比 | 3.2-3.4 | ✅ |

**实际偏离**：
- PPO 单 UAV 训练初期遇到 0 landing 问题，经多轮修复（waypoints 偏移、下降方向、进度奖励设计）后达到 100% 着陆率。
- 关键发现：进度奖励必须与 DESCENDING 状态绑定，否则 agent 学会"下降一段后 hover"。

---

## Phase 4 (初版): 多 UAV Shared-Action — 已弃用，由 Phase 2(R)+3b(R) 替代

> 此阶段采用 shared-action（所有 UAV 同一控制），已被 per-UAV 共享策略替代。
> PPO 79% 着陆率，冲突率偏高（shared-action 导致同时向内收敛）。

---

## Phase 2(R): 核心环境重构 ✅ 已完成

> 修订版计划，替代初版 Phase 2-4。

| # | 任务 | 说明 | 状态 |
|---|------|------|------|
| 2.1 | config.py 更新 | 删 HOVER/waypoints，增 inflow/k_nearest/max_uavs/mission_types | ✅ |
| 2.2 | UAV 模型重写 | 删 HOVER+waypoints，轨道锁定盘旋 + (ψ,γ,V) 连续下降 | ✅ |
| 2.3 | UAVManager 重写 | 轨道环分布 + 流入调度 + mission_type 随机分配 | ✅ |
| 2.4 | TerminalAreaEnv 重写 | per-UAV 局部观测矩阵 (max_n,75) + per-UAV 动作空间 | ✅ |
| 2.5 | PerUAVDiscreteWrapper | MultiDiscrete → 连续参数自动计算 | ✅ |
| 2.6 | 测试更新 | 39 测试全部通过 | ✅ |

**核心设计变更**：
- **轨道模型**：UAV 严格锁定在环轨道 (R_i, H_i) 上，角速度 ω=V_i/R_i
- **连续下降**：DESCEND 用 (ψ, γ, V) 三个连续量定义 3D 速度矢量
- **Per-UAV 策略**：共享 MLP 逐 UAV 独立决策，天然支持变长 UAV

---

## Phase 3a(R): 单 UAV + 干扰机 ✅ 已完成

| # | 任务 | 说明 | 状态 |
|---|------|------|------|
| 3a.1 | 干扰机环境 | N_bg 架背景 UAV 只盘旋，1 架受控降落 | ✅ |
| 3a.2 | Heuristic 基线 | 单机在干扰中降落 | ✅ |
| 3a.3 | PPO 训练 | DiscreteActionWrapper, 2 actions | ✅ |
| 3a.4 | 不同密度对比 | 5/10/20 背景 UAV | ✅ |

**结果**：单 UAV PPO 在有 10+ 背景干扰下达到 100% 着陆率。

---

## Phase 3b(R): 全 UAV 受控清空空域 ✅ 已完成

| # | 任务 | 说明 | 状态 |
|---|------|------|------|
| 3b.1 | Phase3bWrapper | 全部 UAV is_controlled=True，全部着陆/失败终止 | ✅ |
| 3b.2 | PerUAVDiscreteWrapper | Box(max_n,5)：混合离散/连续，CIRCLE调速 + DESCEND全飞行参数 RL 控制 | ✅ |
| 3b.3 | PerUAVFeatureExtractor | 共享 UAV encoder (75→128→64) + mean-pooling 全局聚合 | ✅ |
| 3b.4 | SubprocVecEnv 并行训练 | N=10 训练, 500k steps, 4 envs, 125 FPS | ✅ |
| 3b.5 | Heuristic 基线 | 多规模评估基准 | ✅ |
| 3b.6 | 评估对比 | 多规模 PPO vs Heuristic @ N=5/10/15/20/25/30 | ✅ |

**2026-05-18 最新多规模结果**（5 项安全修复后重训练）：

| N | PPO 着陆率 | Heuristic 着陆率 |
|---|-----------|-----------------|
| 5 | 100% | 100% |
| 10 | 98% | 100% |
| 15 | 97% | 92% |
| 20 | 94% | 92% |
| 25 | 85% | 88% |
| 30 | 79% | 81% |

**模型文件**：
- `logs/best_model_phase3b/best_model.zip`（EvalCallback 最优模型）
- `logs/ppo_phase3b.zip`（500k 最终模型）

---

## Phase 3b.5(R): SafetyController + 安全修复 ✅ 已完成

| # | 任务 | 说明 | 状态 |
|---|------|------|------|
| 3b5.1 | SafetyController 5 规则 | 同环角间距/盘旋让下降/双下降发散/通用刹车/紧急覆盖 | ✅ |
| 3b5.2 | Plan B 观测集成 | Safety 介入信号（speed_mult, psi_offset, gamma_offset, blocked）作为 +4 观测维 | ✅ |
| 3b5.3 | 轨道 Snap 安全检查 | 下降中止回环时搜索 ±30° 安全槽位 | ✅ |
| 3b5.4 | 环容量限制 | 环上 UAV ≥ 90% 2πR/min_spacing 时阻止非紧急下降 | ✅ |
| 3b5.5 | 距离接近惩罚 | 间距 < 30m 触发梯度惩罚 -5/(dist+0.1) | ✅ |
| 3b5.6 | 碰撞移除机制 | 碰撞 UAV 标记死亡并移除，不终止 episode | ✅ |
| 3b5.7 | SafetyController 测试 | 36 专项测试 + 环境回归测试持续维护；当前 83 测试全部通过 | ✅ |
| 3b5.8 | 初始间距保证 | `_find_safe_angle()` 拒绝采样确保初始 UAV 间距 ≥ safe_separation | ✅ |
| 3b5.9 | CIRCLE 调速动作 | 动作空间改为 `Box(max_n,2)`：dim0=control_raw, dim1=speed_factor [0.3,1.3] | ✅ |
| 3b5.10 | DESCEND RL 控制（历史方案） | 曾扩展到 `Box(max_n,5)` 让 RL 直接输出 ψ/γ/V；2026-05-28 已回退为规则下降 + `Box(max_n,2)` | 已回退 |
| 3b5.11 | Safety 介入惩罚 | SafetyController 规则触发、减速、阻塞进入 reward 扣分；DESCEND 放行阻塞仅进入观测 | ✅ |
| 3b5.12 | 显式间距观测 | 增加最近 3D 间距、同环前后弧距、短时预测最近距离、上步介入状态等 +6 维 | ✅ |
| 3b5.13 | 安全裕度塑形 | 100m 内二次间距惩罚，提前于 50m 近距预警提供分离梯度 | ✅ |
| 3b5.14 | 多规模训练修复 | 课程式随机 N、固定 N wrapper 修复、config 深拷贝、reward 跨规模归一化 | ✅ |
| 3b5.15 | Time-limit 语义修复 | max_steps 改为 truncated，避免诊断中 timeout 被统计为普通 terminated | ✅ |

**核心设计**：
- SafetyController 是 RL 动作与 UAV 运动学之间的**反射层**，做减速/微调航向，不做路径规划
- Safety 学习：显式间距 + safety 调整量编码为观测维度，并通过介入惩罚让 PPO 学会**减少**安全干预触发
- 碰撞不再终止 episode，而是移除碰撞 UAV 并施加重惩罚

---

## Phase 4(R): 持续流入场景（计划）

| # | 任务 | 说明 |
|---|------|------|
| 4.1 | 流入调度集成 | uav_inflow_rate > 0，从 4 进入点持续流入 |
| 4.2 | 不同流量负载 | 10/30/60 UAVs/hour 测试 |
| 4.3 | PPO 训练 | 持续流入 + per-UAV 共享策略 |
| 4.4 | 稳态指标 | 吞吐量、平均等待时间、冲突率 vs 流量曲线 |

**验收标准**：30 UAVs/hour 下稳态吞吐量 >25 landings/hour，冲突率 <5%

---

## Phase 5: HRL Options 框架（计划）

| # | 任务 | 说明 | 依赖 |
|---|------|------|------|
| 5.1 | 实现 Options 框架 | Option 定义 + 终止条件 + 切换逻辑 | Phase 4 |
| 5.2 | 实现高层策略 | 空域构型决策（SAC/PPO） | 5.1 |
| 5.3 | 实现低层策略 | UAV 控制（PPO + 离散/连续混合） | 5.1 |
| 5.4 | 高低层协同训练 | Options 联合训练 | 5.2-5.3 |
| 5.5 | HRL vs 单层基线对比 | 消融实验 | 5.4 |
| 5.6 | 训练稳定性调优 | 学习率/奖励权重/网络结构 | 5.4 |

---

## Phase 6: 中等规模实验（计划）

| # | 任务 | 说明 | 依赖 |
|---|------|------|------|
| 6.1 | 50 架 UAV 训练 | 中等规模端到端训练 | Phase 5 |
| 6.2 | 100 架 UAV 训练 | 扩展测试 | 6.1 |
| 6.3 | 200 架 UAV 训练 | 压力测试 | 6.2 |
| 6.4 | 消融实验 | 各模块贡献度分析 | 6.1-6.3 |
| 6.5 | 敏感性分析 | 超参数/环境参数敏感性 | 6.1-6.3 |
| 6.6 | 结果可视化 | 论文级图表 | 6.1-6.5 |

---

## Phase 7: 大规模扩展（计划）

| # | 任务 | 说明 | 依赖 |
|---|------|------|------|
| 7.1 | Ray RLlib 迁移 | 分布式训练框架 | Phase 6 |
| 7.2 | 分布式低层策略 | 多 UAV 独立决策 | 7.1 |
| 7.3 | >200 架训练 | 大规模端到端 | 7.2 |
| 7.4 | 可扩展性分析 | 不同规模对比报告 | 7.3 |
| 7.5 | 最终论文实验数据 | 完整实验结果汇总 | 7.4 |

---

## 里程碑

| 里程碑 | Phase | 验收标准 | 状态 |
|--------|-------|----------|------|
| M1: 环境可用 | Phase 2 | 环境可 reset/step，单 UAV 可完成降落 | ✅ |
| M2: 基线建立 | Phase 3 | 单 UAV PPO 100% 着陆率，Heuristic 基线就绪 | ✅ |
| M3: 环境重构 | Phase 2(R) | 轨道盘旋 + 连续下降 + per-UAV 观测，39 测试通过 | ✅ |
| M4: 多机协同 | Phase 3b(R) | 全 UAV PPO 98%@N=10, 79%@N=30，LZ 占用机制 | ✅ |
| M4.5: 安全层 | Phase 3b.5(R) | SafetyController 5 规则 + 5 安全修复，76 测试通过 | ✅ |
| M5: 持续流入 | Phase 4(R) | 稳态吞吐量 >25/hr，冲突率 <5% | ⬜ |
| M6: HRL 收敛 | Phase 5 | HRL 优于单层基线 | ⬜ |
| M7: 规模验证 | Phase 6 | 200 架稳定运行，指标合理 | ⬜ |
| M8: 论文就绪 | Phase 7 | 完整实验数据与图表 | ⬜ |
