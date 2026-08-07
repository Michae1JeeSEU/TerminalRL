# 研究定位与创新主线（Q1 投稿导向版）

> 本文档用于约束后续代码开发、论文写作和实验设计。核心判断：本项目不能被写成“环形空域 + 强化学习 + 无人机降落 + Crazyflie 验证”的普通组合，而应被定位为“动态终端区空域构型调节与多 UAV 降落流协同管理”。

## 1. 推荐论文主线

推荐主线：

> 面向城市低空起降场终端区，研究如何根据动态入场交通流实时调节环形终端区空域构型，并协同决策多架 UAV 的等待、切环、最终进近和降落意图，从而在安全间隔和轨迹可执行性约束下提升降落吞吐量、降低等待延误和冲突风险。

因此，本项目不是底层飞控论文，也不是普通多 UAV 路径规划论文，而是一个低空交通终端区运行管理问题：

```text
Dynamic terminal airspace reconfiguration
+ Multi-UAV landing-flow coordination
+ Safety-filtered executable trajectory generation
+ Simulation-to-Crazyflie scaled validation
```

## 2. 需要避免的低创新表述

以下表述不应作为论文核心创新：

- 使用环形空域组织 UAV 降落；
- 使用强化学习控制无人机进近/降落；
- 使用多智能体强化学习处理多 UAV；
- 使用 Crazyflie 做实机验证；
- 让 RL 直接输出航向角、下降角和速度。

原因：已有 UAM/vertiport arrival 研究已经覆盖了 circular airspace、DRL arrival policy、small UAV demonstration 等相近组合。如果只强调这些元素，论文容易被认为是已有思想的工程复现或场景替换。

## 3. 应当突出的创新差异

后续代码和论文必须围绕以下差异展开：

1. **从固定终端区结构到动态终端区构型**
   不再假设 ring 半径、高度、速度和放行间隔固定，而是把它们作为运行期可调节决策变量。

2. **从单机 arrival action 到跨层空域—降落流协同**
   高层策略调节空域容量和流组织方式，低层策略决策 UAV 的等待、切环、进近和降落意图。

3. **从自由飞行动作到轨迹意图与安全可执行轨迹**
   RL 不直接输出任意 `heading/altitude/speed` 动作，而是输出 release、speed residual、timing residual、slot residual 等交通意图；轨迹生成器负责转换为可执行轨迹。

4. **从纯仿真到仿真—实物缩尺一致性验证**
   大尺度仿真证明交通运行性能，Crazyflie 缩尺实验验证轨迹执行、安全过滤和动态空域策略的可落地性。

## 4. 推荐 research gaps

### Gap 1：固定终端区空域结构难以适应动态入场交通流

现有 vertiport / UAV terminal-area 研究通常在预先给定的 arrival routes、metering fixes、holding rings、approach points 或 funnel-shaped layers 内优化排序、速度、等待或降落许可。它们主要回答“UAV 如何通过既定结构”，而较少回答“终端区结构是否应随交通流状态在线调整”。

对应本文切入点：

> 将环半径、高度、速度、容量阈值和放行间隔建模为高层动态决策变量。

### Gap 2：空域构型设计与单机降落动作缺少跨层协同

空域结构决定终端区容量、等待空间和流线组织方式；单机等待、切环、进近和降落决策决定实际运行效率。但现有研究往往把空域结构设计与 aircraft-level arrival/landing control 分开处理。

对应本文切入点：

> 构建高层动态空域调节与低层多 UAV 降落意图协同的 HRL 框架。

### Gap 3：学习型终端区控制缺少面向真实无人机平台的轨迹可执行约束

已有学习方法常在仿真中直接输出 abstract action 或 free-flight action，但真实 Crazyflie/Crazyswarm2 平台更适合执行 waypoint、trajectory 或 setpoint。直接输出自由航向角、下降角和速度会带来轨迹不连续、安全边界难控制和实物平台部署困难。

对应本文切入点：

> 使用规则基线、residual policy、safety filter 和 trajectory generator，把学习策略限制在安全可执行的轨迹意图空间内。

## 5. 推荐贡献点

论文贡献建议凝练为四点：

1. **Dynamic ring-based terminal airspace model**
   提出动态环形终端区模型，将 ring radius、height、nominal speed、capacity threshold 和 release headway 作为在线调节变量。

2. **Hierarchical reinforcement learning for airspace-flow coordination**
   提出跨时间尺度 HRL 框架，高层低频调整终端区结构与容量，低层高频协同 UAV 等待、切环、进近和降落意图。

3. **Safety-filtered residual policy with executable trajectory generation**
   提出规则基线 + residual RL + safety filter + trajectory generator 机制，避免 RL 直接输出不可部署的自由飞行动作。

4. **Simulation-to-Crazyflie scaled validation**
   建立大尺度仿真与 Crazyflie/Lighthouse/Crazyswarm2 缩尺实验验证流程，证明所提方法在交通运行性能和实物可执行性上的有效性。

## 6. 推荐标题方向

优先标题：

```text
Hierarchical Reinforcement Learning for Dynamic Vertiport Terminal Airspace Management and Multi-UAV Landing Coordination
```

备选标题：

```text
Dynamic Terminal Airspace Reconfiguration and Multi-UAV Landing Coordination Using Hierarchical Reinforcement Learning
```

强调实物验证时可用：

```text
Simulation-to-Real Hierarchical Reinforcement Learning for Dynamic Vertiport Terminal Airspace Management with Multi-UAV Flight Experiments
```

## 7. 实验体系要求

为了支撑 Q1 论文，实验不能只展示一组训练曲线或一段 Crazyflie 飞行动画。至少需要以下实验矩阵。

### 7.1 主要基线

| Baseline | 作用 |
|---|---|
| FCFS / rule-only | 传统排序和规则放行基准 |
| Fixed airspace + rule | 证明固定结构下规则方法的上限 |
| Fixed airspace + low-level RL | 证明只有低层学习不够 |
| Dynamic airspace + rule | 证明只有动态结构、无学习协同也不够 |
| Full HRL | 完整方法 |
| Full HRL without safety filter | 证明安全过滤必要性 |
| Direct action RL | 证明直接自由动作不适合部署 |

第四轮工程化后的正式实验矩阵：

| 配置 / Policy | 空域 | 低层 | 安全 | 实验作用 |
|---|---|---|---|---|
| `fixed_rule` | 固定 | rule-only | on | 固定结构规则基准 |
| `fixed_residual` | 固定 | intention residual | on | 仅低层学习基准 |
| `dynamic_rule` | 动态 | rule-only | on | 仅动态结构基准 |
| `dynamic_hrl` | 动态 | intention residual | on | 完整跨层方法 |
| `full_hrl_no_safety` | 动态 | intention residual | off, simulation only | 安全层消融 |
| `high-level-random` | 随机动态 | rule-only | on | 随机高层基准 |
| `high-level-static` | 不更新 | rule-only | on | 高层静态基准 |
| `dynamic-rule` | 启发式动态 | rule-only | on | 规则动态构型基准 |
| `high-level-ppo` | 学习动态 | rule/intention placeholder | on | 学习型高层策略 |

所有模式仍研究动态终端区构型与多 UAV 降落流协同；random policy 也不获得自由
航向、高度或速度向量控制权。

### 7.2 场景维度

- 不同入场流率：low / medium / high / burst demand；
- 不同降落点服务率或占用时间；
- 不同环容量约束；
- 定位噪声、通信延迟、setpoint dropout；
- Crazyflie lab 缩尺场景：2–3 架起步，逐步扩展。

### 7.3 指标体系

| 指标 | 含义 |
|---|---|
| Average landing delay | 平均降落延误 |
| Throughput | 单位时间成功降落数量 |
| Makespan | 完成全部降落所需时间 |
| Holding time | 环上等待时间 |
| Final queue length | 最终进近排队长度 |
| Conflict / near miss count | 冲突或近距事件数量 |
| Separation violation | 最小间隔违规次数 |
| Trajectory smoothness | 轨迹平滑性 |
| Speed/acceleration violation | 可执行性约束违规 |
| Crazyflie tracking error | 实物轨迹跟踪误差 |

## 8. 对代码开发的直接约束

1. `terminal_marl/` 是唯一主线；`src/` 为 legacy。
2. 必须实现 `DynamicAirspace`，否则论文主线无法体现动态空域创新。
3. 低层动作必须转为交通意图或 residual，不再扩展自由 `heading/altitude/speed` 控制。
4. 必须实现 `TrajectoryGenerator`，作为从算法到 Crazyflie 的桥梁。
5. 必须实现 safety filter，并在实验中做消融。
6. 必须新增 `crazyflie_lab` 缩尺配置，不能用大尺度仿真参数直接宣称实物可验证。
7. 论文实验脚本必须支持固定空域、动态空域、规则、低层 RL、完整 HRL 等模式切换。

## 9. Server training preparation and paper experiment support

第六轮新增服务器训练准备后，论文实验矩阵可以按如下路径形成证据链：

1. 使用 `terminal_marl/configs/training/ppo_smoke.yaml` 和 `ppo_pilot_medium.yaml` 验证 high-level PPO 训练入口、日志、checkpoint 与评估流水线；
2. 使用 `ppo_main_medium.yaml`、`ppo_main_high.yaml`、`ppo_main_bursty.yaml` 对 medium/high/bursty traffic demand 做 multi-seed 训练；
3. 使用 `ppo_reward_sweep.yaml` 与 `reward_diagnostics.py` 校准 reward component，避免某个惩罚项支配学习；
4. 使用 `experiment_matrix.py` 固化论文对照组：
   - fixed airspace + rule-only；
   - fixed airspace + residual placeholder；
   - dynamic airspace + rule heuristic；
   - dynamic HRL + high-level PPO；
   - high-level random/static；
   - full HRL without safety filter。

这些配置服务于 dynamic terminal airspace reconfiguration + multi-UAV landing-flow coordination 的论文主线。PPO checkpoint 表示高层空域/流量管理策略，不表示真实 Crazyflie 底层飞控策略。
## 10. Event-gated HRL experiment matrix

The next formal study uses periodic high-level decisions with event-gated
structural airspace reconfiguration. This keeps the paper focused on dynamic
terminal airspace reconfiguration and multi-UAV landing-flow coordination,
rather than low-level free-flight control.

| Condition | High-level interval | Geometry permission | Low-level behavior | Purpose |
|---|---:|---|---|---|
| `fixed_rule` | n/a | fixed nominal | rule-only | fixed-airspace baseline |
| `event_rule` | 180 s | event gate + dwell | rule-only | rule-based gated geometry baseline |
| `periodic_hrl` | 180 s | dwell only | rule-only/placeholder | isolates periodic reconfiguration without event trigger |
| `event_gated_hrl` | 180 s | event gate + dwell | rule-only/placeholder | proposed learning condition |

All PPO runs in this matrix use a 20 s metrics sampler, 600 s geometry dwell,
3600 s episodes, and a medium/high/bursty training demand mixture
(0.40/0.30/0.30). Because the MDP interval changed from the old 20 s pilot, new
PPO evidence must be trained from scratch.

## 11. Formal evaluation semantics after PRE-RETRAIN revision

Formal experiments should now report capacity and outcomes using the revised
terminal-management semantics:

- Ring capacity is not a hand-tuned constant. It is derived from the current
  actual ring radius and the true safety separation using the adjacent-chord
  formula. Dynamic airspace reconfiguration therefore changes usable holding
  capacity through geometry, not through an arbitrary reward bonus.
- During geometry transitions, the experiment uses the intermediate actual
  radius and height. Reported capacity and occupancy pressure must not jump to
  the target geometry before the transition has executed.
- Finite-cohort evaluation is preferred for paper evidence. A scenario requests
  exactly N UAVs; each request must end as `LANDED`, `CRASHED_COLLISION`,
  `CRASHED_BATTERY`, or a diagnostic `UNRESOLVED/INVALID` outcome.
- A valid formal episode has unresolved count equal to zero and satisfies:

```text
total_requested = landed + crashed_collision + crashed_battery + unresolved
```

- Landing success rate, collision crash rate, battery crash rate, mission
  failure rate, total resolution rate, request-to-terminal time, external
  waiting, and internal system time should be computed from the outcome ledger.
- Old PPO checkpoints trained before this revision are diagnostic only and must
  not be resumed as evidence for the revised model.

These semantics keep the paper focused on dynamic terminal airspace
reconfiguration and multi-UAV landing-flow coordination, rather than low-level
free-flight control.
