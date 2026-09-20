# 研究定位与创新主线

> **文档地位**：本文档是当前论文项目的研究定位总纲。  
> **当前唯一研究范围**：**Phase A — Landing-only terminal-area operations**。  
> Phase B（departure-only）与 Phase C（arrival + departure integrated operations）仅作为后续研究路线，不进入当前代码实现、训练或正式实验。  
> 本文档与 `requirements.md`、`design_spec.md`、`technical_spec.md`、`execution_plan.md` 共同构成当前唯一有效的主线说明；阅读入口与文档权威关系见 `README.md`，当前进度见 `CURRENT_STATUS.md`。历史“两环版本”“直接输出自由飞行动作版本”、日期型 handoff 和旧迁移指南不再作为实现依据。

---

## 1. 论文要解决的核心问题

本文研究面向城市低空垂直起降场/低空终端区的**多无人机到达与降落管理**。核心问题不是底层姿态控制，也不是任意三维路径规划，而是：

> 在随机、突发且可能超出瞬时服务能力的到达需求下，如何通过**动态终端区空域构型调节**与**多 UAV 降落流协同决策**，在安全间隔、有限着陆服务资源、电量约束、任务优先级和轨迹可执行性约束下，提高终端区吞吐量、降低外部排队和内部等待延误，并保持公平、稳定和可解释的运行秩序。

当前方法由四个层次组成：

```text
External arrival demand and admission control
                  ↓
Dynamic three-ring terminal airspace
                  ↓
Shared per-UAV landing-intention policy
                  ↓
Merge/resource scheduler → trajectory generator → safety filter
                  ↓
Executable landing trajectories and LZ service
```

本文的“强化学习控制”限定在**交通管理与飞行意图层**。无人机的具体连续轨迹由确定性的轨迹生成器产生，并由 safety filter 进行安全与可执行性检查。

---

## 2. 当前唯一场景：Phase A Landing-only

### 2.1 运行链路

正式场景采用以下状态链路：

```text
ARRIVAL_REQUESTED
      ↓
EXTERNAL_QUEUE
      ↓ admission
INBOUND_CORRIDOR
      ↓
OUTER_RING      — buffer / arrival absorption
      ↓
MIDDLE_RING     — metering / sequencing
      ↓
INNER_RING      — landing staging
      ↓
FINAL_APPROACH
      ↓
LANDING
      ↓
LZ_OCCUPIED / SERVICE
      ↓
COMPLETED
```

异常终态包括：

```text
DIVERTED
FAILED_COLLISION
FAILED_ENERGY
INVALID / UNRESOLVED   # 仅诊断，正式实验不得出现
```

### 2.2 三环功能分工

| 环 | 角色 | 主要作用 |
|---|---|---|
| Outer ring | Buffer | 吸收突发到达、避免需求直接压到内层 |
| Middle ring | Metering | 排序、速度微调、控制向内层的放行节奏 |
| Inner ring | Landing staging | 等待最终进近和 LZ 资源，形成最后一级安全放行队列 |

数学记号统一为：

\[
R_O > R_M > R_I,\qquad H_O > H_M > H_I
\]

其中 `O/M/I` 分别表示 outer / middle / inner。代码、日志和论文均禁止继续使用含义不明确的“Ring 1/2/3”来推断内外顺序；如需整数索引，必须同时携带明确的 `RingRole`。

### 2.3 后续扩展但当前不实现

- **Phase B — departure-only**：`LZ → departure corridor → terminal exit`。
- **Phase C — integrated arrival/departure**：到达与离场共享 pad/LZ/time resource，并研究 final 与 climb-out corridor 的冲突协调。

当前 Phase A 不为 departure 预留额外动作维度，也不让起飞交通进入 arrival holding rings。

---

## 3. 研究定位：什么是贡献，什么不是

### 3.1 不作为主要创新的内容

以下元素可以作为系统组成部分，但不能单独写成核心创新：

- 使用环形等待空域；
- 使用 PPO/MARL；
- 使用多架 UAV；
- 使用规则避碰；
- 使用 Crazyflie 或其他小型无人机验证；
- 通过增加状态变量获得更高性能。

### 3.2 当前主创新主线

#### 创新 1：动态三环终端区构型与流量调节

高层策略根据当前与近期交通压力，调整：

- 三个环的半径；
- 三个环的高度；
- 整体 nominal-speed profile bias；
- `Outer→Middle`、`Middle→Inner`、`Inner→Final` 三个放行 headway。

环容量不是独立动作，而是由**当前实际半径与安全间隔**派生，避免“扩大半径”和“人为增加容量阈值”重复表达同一个物理效应。

#### 创新 2：空域构型与多 UAV 降落流的层次协同

高层回答：当前终端区应该提供怎样的空间结构和流量节奏？

低层共享策略回答：对于某一架 UAV，现在是否应该请求进入下一阶段、应该如何微调速度、希望把 merge 时间向前或向后调整多少？

两层之间通过 occupancy、queue、LZ availability、SOC、priority、headway、merge reservation 等信息耦合，而不是让两个策略分别解决互不相关的问题。

#### 创新 3：从学习意图到安全可执行轨迹

低层策略不直接生成任意航向、航迹角、高度或姿态命令，而输出：

\[
a_i^L = [r_i,\;\Delta v_i,\;\Delta t_i]
\]

分别表示：

1. `release_request`：二元 WAIT/RELEASE 请求；
2. `speed_residual`：五级 normalized speed residual；
3. `merge_time_residual`：五级 normalized merge-time residual。

这是 factorized discrete MARL scheduling/request layer；飞行器执行仍是连续、
受性能约束的 3D point-mass trajectory execution，而不是离散飞控。

随后由：

```text
intent
→ resource/merge scheduler
→ canonical trajectory generator
→ safety filter
→ execution
```

统一生成和检查轨迹。

#### 创新 4：终端空域—地面服务资源耦合

论文不把“成功进入 final”当成服务完成，而显式建模：

- 多个离散 Landing Zone；
- reservation；
- touchdown 后的 occupied/service；
- 任务类型相关的服务时间；
- 服务能力对 inner-ring 放行、等待、电量和动态空域构型的反作用。

这使研究对象真正成为 terminal operation，而不是单纯的空中队列。

---

## 4. 关键研究假设

### H1：动态构型在非平稳需求下优于固定构型

在 low demand 下，动态构型不应明显恶化效率；在 high/bursty demand 下，它应通过扩大有效 holding capacity、改变层间流量节奏和释放瓶颈，提高成功服务率并降低高分位延误。

### H2：三级功能分层优于简单两环队列

三环结构通过 `buffer → metering → staging` 将到达吸收、排序和最终资源等待分离。其价值必须通过 `2-ring` ablation 验证，而不是仅凭设计直觉主张。

### H3：低层意图学习优于纯规则，但不需要自由飞行动作

在固定空域下，低层共享策略应能通过 release、speed residual 和 merge-time residual 改善排序与等待，同时保持轨迹连续和安全过滤可解释。

### H4：跨层组合优于单独优化任一层

完整 `dynamic_hrl` 应优于 `fixed_rule`、`fixed_low_rl`、`dynamic_rule`。否则不能声称高低层协同本身带来了额外收益。

### H5：资源、电量和优先级机制会改变最优流量管理策略

任务服务时间、剩余 SOC、等待时间与优先级会改变“谁应该先进入 final”和“什么时候应该扩/缩环”，因此不能从主模型中删去。

---

## 5. 研究问题（Research Questions）

- **RQ1**：动态三环构型是否能在随机和突发 arrival demand 下，提高 landing throughput 并降低 delay？
- **RQ2**：三环功能分层是否比两环结构更能缓冲突发流量并降低 inner/final 拥堵传播？
- **RQ3**：共享的 per-UAV intention policy 是否能够在未知数量 UAV 下实现可扩展的 sequencing 和 merge decision？
- **RQ4**：任务类型、LZ service time、动态 priority 和 SOC 约束是否会显著改变终端区运行结果与 learned policy？
- **RQ5**：event-gated structural reconfiguration 是否比固定周期无条件重构更稳定、更少产生无效结构变化？
- **RQ6**：完整 HRL 是否优于“仅低层学习”和“仅高层动态”的方法？

---

## 6. 研究对象中的核心状态与约束

每架 UAV 至少具有：

```text
uid
phase
position / ring progress
current ring role
speed
SOC
energy consumed
request/admission/entry times
wait time
base priority
dynamic priority
mission type
assigned entry gate
assigned LZ
LZ reservation state
merge reservation
transition state
terminal outcome
```

系统至少具有：

```text
three-ring geometry
current/target geometry during reconfiguration
ring occupancies and derived capacities
three release headways
external queue
four entry-corridor queues
final queue
M discrete LZ states
arrival demand state
safety/trajectory reservation state
```

硬约束包括：三环半径/高度严格有序、geometry change rate limit、geometry minimum dwell、occupancy-aware shrink、最小 separation、headway/merge reservation、飞行可执行性、LZ mutual exclusion、SOC reserve/diversion，以及 structural reconfiguration 时受影响接口暂停新 transfer。

---

## 7. 高低层策略与训练原则

### 7.1 两个策略网络

\[
\pi_H(a_H|s_H),\qquad \pi_L(a_i|o_i)
\]

- `π_H`：全局、高层、低频；
- `π_L`：per-UAV、共享参数、高频。

低层不是“一架 UAV 一个网络”，而是所有 active UAV 共享一个 actor；每一步把 active UAV observation 组成 batch 并行前向。

### 7.2 低层 CTDE

低层推荐采用 centralized training, decentralized execution：actor 只看本机和有限局部/系统摘要；critic 通过 permutation-invariant set/attention encoder 聚合可变数量 UAV。

不得用固定 `max_n × full_state` 作为最终理论依赖；实现层可以保留 padding/mask 作为 batching 手段，但策略语义必须与实际 active UAV 数解耦。

### 7.3 训练不能从两个随机策略联合开始

```text
Stage 0: rule-only environment validation
Stage 1: fixed airspace + train low-level policy
Stage 2: freeze low-level + train high-level policy
Stage 3: full hierarchical evaluation
Stage 4: optional alternating fine-tuning
```

不接受“两个策略从随机初始化同时 PPO、效果不好再调 reward”的主开发路线。

---

## 8. 动态空域控制的研究语义

### 8.1 高层动作

当前目标高层动作统一为 10 维：

\[
a^H = [
\Delta R_O,\Delta R_M,\Delta R_I,
\Delta H_O,\Delta H_M,\Delta H_I,
\Delta V_{profile},
\Delta h_{OM},\Delta h_{MI},\Delta h_{IF}
]
\]

其中前 6 维为 structural action，第 7 维整体偏置三环 nominal-speed profile，后 3 维为 operational flow-control action。三环物理 capacity 由 geometry 自动计算，不额外输出 capacity action。

### 8.2 Event-gated geometry

默认：

```text
metrics sampling interval = 20 s
high-level decision interval = 180 s
geometry minimum dwell = 600 s
geometry transition duration = 20 s
```

当 structural action 被 gate mask 时，speed-profile bias 和 headway action 仍可执行，radius/height 保持当前 target。

### 8.3 在途 UAV 对动态环的响应

- 正在 ring hold 的 UAV 跟随连续变化的 `R(t), H(t)` moving reference；
- 已经进入 ring-transfer trajectory 的 UAV 完成原已承诺轨迹；
- reconfiguration 涉及的接口暂停接受新的 transfer；
- geometry settled 后恢复该接口；
- 不允许 teleport 到新的 ring 几何。

---

## 9. 轨迹与 merge 的研究语义

### 9.1 Ring hold

\[
x=R(t)\cos\theta,\quad y=R(t)\sin\theta,\quad z=H(t)
\]

### 9.2 Ring transfer

采用切向连续的平滑曲线，不采用径向直线直接切入下一环。推荐：水平面 cubic Bézier / 等价 C1 连续样条；垂直方向 quintic smoothstep；起点切线与 source ring 一致，终点切线与 target ring 一致。

### 9.3 Merge point

目标环进入点不是固定角度 offset，而由 merge scheduler 根据预测交通 gap 决定：

```text
candidate target time
→ predict target-ring aircraft phases
→ find feasible predecessor/successor gap
→ reserve (merge_time, merge_angle)
→ generate exact transition trajectory
```

同一条 trajectory 必须同时用于 safety checking 和实际执行。

### 9.4 Final 与 Landing

Inner→Final 的放行必须同时满足 assigned LZ reservation、final corridor 可用、headway、candidate trajectory safety、SOC/priority/queue rules。touchdown 后 UAV 从空中交通集合退出，但对应 LZ 保持 `OCCUPIED`，直到 mission-specific service time 到期。

---

## 10. 任务、服务资源、电量和优先级

这些不是“额外 realism option”，而是主场景必要机制。

### 10.1 Mission type

| Mission | Probability | LZ service time |
|---|---:|---:|
| standard | 0.45 | 30 s |
| cargo | 0.25 | 120 s |
| emergency | 0.10 | 10 s |
| surveillance | 0.20 | 60 s |

任务类型主要决定服务时间与可能的资源兼容性，不直接代替 dynamic priority。

### 10.2 Priority

保留 `base_priority ∈ {1,3,5}`，默认概率 `0.50/0.35/0.15`；`emergency` mission 的 base priority 至少为 5。动态优先级由 base priority、等待时间和 SOC urgency 共同决定。

### 10.3 Energy / SOC

使用 SOC 与 phase-dependent energy consumption，而不是所有状态统一按 elapsed time 减少 endurance。低 SOC 可提升动态优先级、触发最早可行 landing，或在尚未 admission 时触发 diversion。

---

## 11. 正式实验矩阵

### 11.1 核心 baseline

| Mode | Airspace | Low level | Purpose |
|---|---|---|---|
| `fixed_rule` | 固定三环 | rule | 基础规则基准 |
| `fixed_low_rl` | 固定三环 | learned intention | 只验证低层学习 |
| `dynamic_rule` | 动态三环 | rule | 只验证动态构型 |
| `dynamic_hrl` | 动态三环 | learned intention | proposed method |

### 11.2 关键 ablation

| Ablation | 目的 |
|---|---|
| `two_ring_dynamic_hrl` | 验证三环功能分层价值 |
| `periodic_geometry_hrl` | 对比 event-gated geometry |
| `homogeneous_service` | 去除任务服务时间异质性 |
| `no_dynamic_priority` | 验证 wait/SOC priority |
| `no_safety_filter` | simulation-only，验证安全层作用 |
| `fixed_headway` | 验证 flow-control action |

### 11.3 Demand scenarios

至少包含 low、medium、high、bursty、overload stress test。正式比较采用 common random numbers：同 seed 下，不同 policy 接收相同 arrival request times、mission types、priority classes、gate assignments 和 initial energy draws。

低层课程必须同时覆盖两类互补状态：`flow_from_empty` 验证真实到达与完整生命周期，
`warm_start_rings` 主动覆盖 O/M/I 三环同时已有 UAV 的瞬时竞争。warm-start 可用于 curriculum、
压力诊断和独立资格，但正式主结论不能只依赖 warm-start；dense bridge 应把两类 cohort 分开报告。

### 11.4 两类正式评估

**Protocol A — Finite-cohort resolution**：固定请求数量 N，所有 request 必须进入终态。  
**Protocol B — Continuous-flow capacity**：固定仿真时长，在 warm-up 后统计稳定区间 throughput、queue、utilization 与 delay。

---

## 12. 论文指标

必须至少报告：

- **Safety**：collision、NMAC/separation violation、minimum separation、trajectory reject/modify；
- **Efficiency**：throughput、request-to-completion delay、external waiting、internal terminal time、holding、makespan、final queue；
- **Resource**：LZ utilization、reservation wait、service bottleneck、ring occupancy/capacity；
- **Dense landing flow**：nearest-LZ assignment regret、concurrent final、free-LZ-with-inner-demand、
  individually-certified-but-not-requested RELEASE opportunity；
- **Energy**：energy consumed、SOC at landing、low-SOC diversion、reserve violation；
- **Fairness**：priority/mission 分组 delay、starvation、分位差；
- **Dynamic airspace**：geometry change count、transition time、gate mask ratio、headway/capacity changes。

---

## 13. 论文叙事边界

当前论文可以主张：Landing-only dynamic terminal management、three-ring functional architecture、HRL airspace-flow coordination、variable-size shared low-level policy、LZ service coupling、safety-filtered executable trajectory generation。

当前论文不应主张：已解决 arrival+departure integrated operations、城市级 network routing、真实动力学级飞控、任意机型直接部署，或把仿真参数当作真实法规 capacity。

---

## 14. 后续研究路线

### Phase B — Departure-only

```text
LZ release
→ departure scheduling
→ departure corridor
→ departure fix
→ terminal exit
```

arrival holding ring 不直接复用于 departure。

### Phase C — Integrated arrival + departure

研究 pad time-slot sharing、final/climb-out corridor conflict、arrival/departure priority、bidirectional terminal capacity、integrated high-level airspace allocation。

Phase B/C 的动作、状态和 reward 均不得提前混入当前 Phase A 模型。

---

## 15. 一句话研究定位

> 本项目研究的是：**在有限着陆服务能力、能量与安全约束下，利用层次强化学习在线调节三环终端区的空间构型和放行节奏，并通过共享的多 UAV 降落意图策略、安全调度与可执行轨迹生成，实现动态到达流的高效、安全和可扩展管理。**
