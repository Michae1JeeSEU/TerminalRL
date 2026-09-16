# A5-v2 低层策略训练纲领与场景课程计划

> **文档定位**  
> 本文件是 A5-v2 低层飞行意图策略训练阶段的纲领性文件，用于统一后续代码配置、训练运行、实验记录、3D 回放、阶段晋级与最终 qualification。它不是某一次训练 run 的参数清单，而是整个低层训练阶段的**方法学与场景协议**。
>
> **适用架构**  
> - Shared low-level policy `π_L`
> - Topology-aware Observation V2
> - Low-level action `[release_request, Δv, Δt]`
> - TeacherScheduler / ExecutionScheduler 分离
> - Canonical Trajectory Generator
> - SafetyFilter
> - Predictive Safety Shield
> - High-level policy 在低层完成 geometry qualification 前保持关闭
>
> **建议版本号**：`A5V2-TRAINING-CURRICULUM-v3`

> **v3 文档边界说明**
> - 本文件只规定 **low-level policy 的训练、验证、静态 geometry 泛化和最终 qualification**。
> - Dynamic reconfiguration 的状态机、Affected-set Freeze、重构持续时间、cooldown、switching cost、高层 reward / observation 以及 H0/CH0 economics calibration **不在本文件定义**。
> - 这些内容统一放入 `RECONFIGURATION_AND_HIGH_LEVEL_INTEGRATION_SPEC.md`。
> - 若动态重构期间受影响 UAV 的 low-level action 被系统屏蔽，则这些时间步不作为 low-level actor 的训练目标；低层只需保证在重构结束、重新获得控制权后的状态分布上能够恢复稳定协调。

---

# 1. 总体目标

A5-v2 低层训练的目标不是简单得到一个高 reward PPO，而是得到一个能够在复杂终端交通中自主完成正常协调、仅在少数极端情况下依赖 predictive shield 的共享低层策略。

最终策略应满足：

1. 根据本机状态、局部拓扑邻机、资源状态和几何状态，自主决定：
   - 是否 release；
   - `Δv`；
   - `Δt`。
2. 正常 sequencing / conflict avoidance 主要由 learned policy 完成，而不是 scheduler 长期搜索替代动作。
3. TeacherScheduler 只在训练早期充当老师，随后逐步退出。
4. 最终部署仍保留：
   - ExecutionScheduler；
   - reservation / mutual exclusion / hard feasibility；
   - canonical trajectory；
   - SafetyFilter；
   - Predictive Safety Shield。
5. Student-only 条件下应保持：
   - collision = 0；
   - NMAC = 0；
   - hard separation violation = 0；
   - shield intervention burden 足够低；
   - completion / throughput / delay 不因 teacher 退出而明显崩溃。
6. 在高层训练前，低层还必须适应高层未来可能产生的 terminal geometry，而不能只在 nominal geometry 上有效。

完整训练主线：

```text
Capacity calibration
    ↓
Teacher-guided coordination learning
    ↓
Teacher authority annealing
    ↓
Student-only nominal coordination
    ↓
Static geometry generalization
    ↓
Post-reconfiguration recovery compatibility
    ↓
Low-level qualification
    ↓
Freeze π_L
    ↓
Hand off frozen π_L to reconfiguration / high-level workflow
```

---

# 2. 核心训练原则

## 2.1 使用二维 Curriculum

训练有两条独立难度轴。

### 轴 A：Teacher Authority

```text
FULL
  ↓
IMMEDIATE_ONLY
  ↓
LABEL_ONLY
  ↓
OFF
```

### 轴 B：Environment / Traffic Complexity

```text
simple
  ↓
medium stationary
  ↓
variable density
  ↓
dense / asymmetric
  ↓
mission / SOC / priority
  ↓
bursty / resource-limited
  ↓
variable-N
  ↓
geometry variation
  ↓
OOD compound stress
```

**一次阶段切换原则上只主要改变一个难度来源。**

不要直接进行：

```text
medium + FULL teacher
        ↓
bursty + low SOC + asymmetric + LABEL_ONLY
```

否则失败后无法判断原因来自 teacher 权限、traffic density、SOC、mission、resource scarcity 还是 observation distribution shift。

---

## 2.2 使用 Structured Domain Randomization

训练场景不是“越随机越真实越好”。

不应大量出现这种完全独立随机得到的极端组合：

```text
极端 overload
+ 极高 simultaneously-active N
+ 强单入口偏置
+ 大量 emergency
+ 大量 reserve-SOC
+ 多个 LZ unavailable
+ geometry 边界
```

这类场景可能本身不可服务，只会污染 policy gradient。

训练应通过明确的 Scenario Classes 逐步扩大场景集合。

---

## 2.3 Safety、Execution 与 Teacher 严格分工

### TeacherScheduler
训练期老师。可提供：
- action 是否合理；
- bounded correction；
- teacher label；
- correction magnitude；
- disagreement 信息。

Teacher 不属于最终运行策略。

### ExecutionScheduler
始终存在。职责：
- 将 policy request 映射为单一执行候选；
- reservation/resource consistency；
- headway / LZ mutual exclusion 等硬执行语义；
- 单候选 feasibility。

**不允许在 deployment mode 中主动枚举大量 `Δv/Δt` 候选替 RL 找“更好动作”。**

### SafetyFilter
始终存在。职责：
- 对 canonical trajectory 做硬安全验证；
- 不充当日常策略优化器。

### Predictive Safety Shield
始终存在。职责：
- 在预测未来将发生 hard-safety violation 时进行最小安全干预；
- 触发基于 predicted minimum separation / CPA / TTC / planned trajectory conflict；
- 不应等到已经低于 hard safe distance 才介入。

---

# 3. PPO 与 Teacher Correction 的动作语义

每一步必须区分：

```text
a_requested
a_teacher
a_applied
```

- `a_requested`：policy 实际采样动作；
- `a_teacher`：teacher 建议或修正动作；
- `a_applied`：环境真正执行动作。

关键约束：

> **PPO rollout buffer、old log-prob 和 policy ratio 必须对应 `a_requested`，不能把 `a_teacher` 伪装成 PPO 自己采样出的行为动作。**

Teacher correction 可以进入 auxiliary loss / reward penalty，例如：


a_total = policy request;  environment transition may be shielded/teacher-corrected.

训练目标可包含：

- PPO objective；
- teacher BC auxiliary loss；
- teacher intervention penalty；
- correction magnitude penalty；
- shield intervention penalty。

但行为动作语义不能混淆。

---

# 4. C0–L4 阶段固定不变的环境语义

在 nominal low-level autonomy 形成之前，下列核心定义原则上保持固定。

| 项目 | 设定 |
|---|---|
| terminal geometry | nominal fixed three-ring |
| high-level policy | OFF |
| dynamic reconfiguration | OFF |
| ring direction / topology | 固定 |
| simulation `dt` | 1 s（若代码 canonical config 不同，以代码为准） |
| O→M headway | 9 s |
| M→I headway | 9 s |
| I→F headway | 12 s |
| LZ 数量 | 6 |
| hard safe distance | 80 m |
| NMAC threshold | 40 m |
| collision threshold | 20 m |
| action | `[release_request, Δv, Δt]` |
| Observation | A5-v2 topology-aware V2 |
| ExecutionScheduler | ON |
| Trajectory Generator | ON |
| SafetyFilter | ON |
| Predictive Safety Shield | ON |
| TeacherScheduler | 随阶段退火 |
| energy model | 从第一阶段始终 ON |
| reward 主体 | L1 后尽量冻结 |

> 若代码 canonical 数值与本表不同，应在正式训练前用代码事实更新本表。训练开始后不要无版本地修改这些常量。

---

# 5. Low-level Observation 的训练前要求

Observation V2 至少应覆盖以下信息。

## 5.1 本机
- phase；
- current ring / target ring；
- progress；
- speed；
- SOC；
- waiting time；
- mission；
- priority；
- gate；
- target LZ；
- release eligibility；
- reservation / lifecycle state。

## 5.2 Topological neighbors
至少有：

```text
same-ring leader
same-ring follower
target-ring predecessor
target-ring successor
nearest predicted conflict
```

邻机 feature 可包含：
- relative distance / arc gap；
- relative speed；
- relative ETA；
- CPA distance；
- time-to-CPA；
- phase/ring；
- presence mask；
- reservation/intent（如实现支持）。

## 5.3 Resource summary
- ring occupancy；
- headway availability；
- assigned-LZ usability；
- free-LZ fraction；
- final resource pressure；
- external queue / relevant congestion summary。

## 5.4 Geometry-aware information
在进入 L5 前必须确认 low-level 能感知 geometry。推荐包含：
- current ring radius；
- target ring radius；
- radius difference；
- distance-to-next-merge；
- nominal time-to-merge；
- current circumference 或等价归一化量；
- 可选全局 `[r_outer, r_middle, r_inner]`。

优先使用具有物理意义、对尺度更稳健的量：

```text
distance-to-merge
nominal-time-to-merge
relative ETA
relative gap
CPA / TTC
```

---

# 6. C0：容量与场景标定（不训练 PPO）

## 6.1 目标

得到 nominal geometry 下的经验可持续容量：

`μ_hat = sustainable arrival rate of the fixed terminal`

之后统一用：

`ρ = λ / μ_hat`

定义 low / medium / high / near-capacity / overload。

未经容量标定，不应永久把某个绝对 `λ` 称为 medium 或 high。

---

## 6.2 控制架构

```text
TeacherScheduler = FULL
ExecutionScheduler = ON
SafetyFilter = ON
PredictiveShield = ON
High-level = OFF
Geometry = nominal fixed
π_L = 不训练 / teacher-safe baseline
```

---

## 6.3 初始容量扫描建议

| λ (UAV/s) | UAV/h |
|---:|---:|
| 0.015 | 54 |
| 0.020 | 72 |
| 0.030 | 108 |
| 0.040 | 144 |
| 0.050 | 180 |
| 0.060 | 216 |
| 0.075 | 270 |
| 0.090 | 324 |

根据结果在转折区间加密采样。

每个点建议：
- 多个固定 seeds；
- 3600 s 级别 episode；
- warm-up 与 measurement 分离；
- 对 controller 比较使用相同 arrival schedule / CRN。

---

## 6.4 容量指标

至少记录：
- requested arrival rate；
- admitted rate；
- completed throughput；
- completion/request ratio；
- external queue slope；
- active UAV count；
- peak active UAV；
- ring occupancy；
- LZ utilization；
- final utilization；
- mean/P95 delay；
- teacher intervention；
- shield intervention。

重点观察 external queue 的长期斜率。

可以把 `μ_hat` 定义为满足下列条件的最大 `λ`：
- completion/request ≥ 预先冻结阈值，例如 0.95 或 0.98；
- measurement window 内 queue slope 近似 0；
- safety 全满足。

**0.95/0.98 只是工程初值，正式实验前必须冻结，不允许看完 qualification 再改。**

---

# 7. Traffic Levels

得到 `μ_hat` 后统一使用：

| Level | Offered load |
|---|---:|
| Low | `0.30–0.40 μ_hat` |
| Medium | `0.45–0.60 μ_hat` |
| High | `0.70–0.85 μ_hat` |
| Near-capacity | `0.85–0.95 μ_hat` |
| Overload | `1.05–1.20 μ_hat` |

Overload 用于 robustness / graceful degradation，不应主导早期或主体 PPO gradient。

---

# 8. Traffic / Arrival Models

## 8.1 Deterministic interval
用途：
- L0；
- debug；
- anchor scenario；
- deterministic witness。

优点：可重复、易定位因果、适合 3D 回放。

## 8.2 Stationary Poisson
作为主体训练流量。

均匀四入口时可采用 aggregate Poisson + gate thinning：

```text
p_gate = [0.25, 0.25, 0.25, 0.25]
```

## 8.3 Asymmetric gate demand

Mild：

```text
[0.40, 0.20, 0.20, 0.20]
```

Strong：

```text
[0.55, 0.15, 0.15, 0.15]
```

Dominant gate 应轮换，避免空间方位 shortcut。

## 8.4 Dynamic gate imbalance
后期可在 episode 内改变 dominant gate，例如：

```text
first half:  [0.55, 0.15, 0.15, 0.15]
second half: [0.15, 0.15, 0.55, 0.15]
```

## 8.5 Bursty flow：Piecewise NHPP
示例：

```text
normal:   180–300 s @ 0.5 μ_hat
burst:     60–120 s @ 1.1–1.3 μ_hat
recovery: 180–300 s @ 0.3–0.4 μ_hat
```

持续时间和强度随机。

## 8.6 MMPP 作为 OOD 候选

```text
NORMAL ↔ BURST
```

训练如果使用 stationary Poisson + NHPP，可把 MMPP 留给 OOD；也可反过来。

## 8.7 Fixed schedule + CRN
建议支持预生成 arrival schedule，并在不同 policy 对比时复用，实现 Common Random Numbers，降低比较方差。

---

# 9. Reset：From-empty 与 Safe Warm-start

## 9.1 From-empty

```text
t = 0
active UAV = 0
```

优点：完整 lifecycle；缺点：早期有效交通交互样本少。

## 9.2 Safe Warm-start
必须通过统一 `safe_warm_start_generator` 生成，不能随机撒飞机。

要求：
- initial separation 安全；
- ring order 合法；
- leader/follower consistent；
- reservation consistent；
- committed trajectory consistent；
- lifecycle consistent；
- reset 后不应立刻因构造错误触发 shield。

建议：

`d_min(0) >= 1.2 * d_safe`

若 `d_safe = 80m`，初始可从约 100m 以上作为工程值起步。

## 9.3 Reset curriculum

| 阶段 | Warm-start | From-empty |
|---|---:|---:|
| L0 | 60–70% | 30–40% |
| L1 | 60% | 40% |
| L2 | 40% | 60% |
| L3 | 20–30% | 70–80% |
| L4 | 10–20% | 80–90% |
| Qualification | 单独报告 | 主体使用 |

Warm-start 是训练加速器，不是最终问题定义。

---

# 10. Scenario Classes

| ID | 场景 | 主要目的 |
|---|---|---|
| S0 | Single / very low | lifecycle sanity |
| S1 | Nominal medium symmetric | basic sequencing |
| S2 | Dense stationary | topology coordination |
| S3 | Asymmetric gate | spatial imbalance |
| S4 | Bursty / recovery | transient congestion |
| S5 | Resource-limited LZ | resource effects |
| S6 | Energy / priority | urgency sequencing |
| S7 | Variable-N | scalability |
| S8 | Near-capacity / overload | graceful degradation |
| S9 | Static geometry variation | geometry robustness |
| S10 | Dynamic geometry transition | coordinator qualification + post-transition recovery |
| S11 | OOD compound | final stress |

后续所有 stage config 尽量引用这些场景类，而不是重新定义不同 env。

---

# 11. L0：Teacher Dataset / 可选 BC

## 11.1 目标
构造：

```text
state
→ requested action
→ teacher judgment/correction
```

建立安全与运行意图先验。

## 11.2 模式

```text
Teacher = FULL
ExecutionScheduler = ON
SafetyFilter = ON
PredictiveShield = ON
High-level = OFF
Geometry = nominal
```

## 11.3 场景

| 维度 | L0 |
|---|---|
| traffic / cohort | 5–15 为主 |
| reset | 60–70% warm-start |
| demand | finite cohort + deterministic interval |
| load | low / medium |
| gate | uniform |
| mission | standard 100% |
| SOC | 0.8–0.9 fixed/narrow |
| priority | nominal |
| LZ | all free |
| burst | OFF |
| overload | OFF |
| noise/delay | OFF |
| geometry | nominal |

## 11.4 Dataset balance
必须包含：

```text
request already safe
minor correction
major correction
release reject / hold
safe release
unsafe release
speed correction
timing correction
```

避免 BC 退化成“看到 traffic 就 hold”。

## 11.5 BC 非必需
建议 short pilot 比较 PPO cold-start 与 BC warm-start。若 BC 没显著改善早期学习速度或 teacher dependence，则可跳过长期 BC。

---

# 12. L1：FULL Teacher PPO — 基本协调

## 12.1 目标
学习：
- release timing；
- speed residual；
- merge timing；
- leader/follower；
- predecessor/successor gap；
- basic conflict anticipation。

## 12.2 Teacher

```text
FULL
```

允许 bounded correction 和 future-delay repair/search。

## 12.3 场景

| 维度 | L1 |
|---|---|
| traffic | Medium |
| ρ | 0.45–0.60 |
| demand | finite cohort + stationary Poisson |
| warm-start | ~60% |
| N_init | 8–16 主体 |
| cohort N | 8–20 |
| gate | uniform |
| mission | standard |
| SOC | ~0.8 |
| priority | nominal |
| LZ | all free |
| burst | OFF |
| geometry | nominal |

## 12.4 Reward
需要显式体现：
- teacher intervention penalty；
- teacher correction magnitude penalty；
- safety；
- progress/completion；
- delay；
- energy；
- starvation / fairness（如已有）。

## 12.5 晋级逻辑
不要只看 reward。

核心指标：
- teacher interventions / 100 release intents；
- correction magnitude；
- completion；
- release request rate；
- shield intervention；
- anchor scenario stability。

目标趋势：teacher dependence 显著下降。

“下降 40–60%”可作为 pilot 后待冻结的工程候选门槛，而非理论常数。

如出现：

```text
reward ↑
teacher intervention ≈ constant
```

应先诊断 teacher leakage / reward loophole，而不是继续堆 steps。

---

# 13. L1b：FULL Teacher + 环境复杂度提升

目的：teacher 权限不变，只提升场景难度。

加入：

`ρ ~ U(0.4, 0.7)`

`N_init ~ 5...20`

Gate 混合：

```text
25/25/25/25
40/20/20/20
20/40/20/20
20/20/40/20
20/20/20/40
```

仍保持：
- mission 简单；
- SOC 高；
- LZ mostly free；
- burst OFF；
- geometry nominal。

---

# 14. L2：IMMEDIATE_ONLY Teacher

## 14.1 目标
禁止 teacher 做大范围未来 candidate search，让 policy 真正承担 timing decision。

## 14.2 模式

```text
Teacher = IMMEDIATE_ONLY
future-delay search = OFF
```

只允许 requested action 附近的小范围立即修正。

## 14.3 场景

| 维度 | L2 |
|---|---|
| ρ | 0.50–0.75 |
| arrival | stationary Poisson 主体 |
| active N | ~5–25 |
| warm-start | 40% |
| from-empty | 60% |
| gate | uniform + mild asymmetric |
| SOC | U(0.5,1.0) |
| mission | standard + cargo + surveillance |
| emergency | OFF 或极低 |
| priority | 开始随机 |
| LZ busy | 0–1 |
| burst | OFF |
| geometry | nominal |

---

# 15. SOC / Energy Curriculum

Energy model 从第一阶段就 ON，只改变初始 SOC distribution。

```text
L0/L1: 0.8–0.9 fixed/narrow
L2:    U(0.5,1.0)
L3a:   U(0.3,1.0)
L3b+:  U(0.15,1.0)
```

不要从 L1 就大量采样 reserve-SOC，否则 basic coordination、energy urgency 和 dynamic priority 同时进入 credit assignment。

---

# 16. Mission Curriculum

若正式分布为：

```text
standard      45%
cargo         25%
emergency     10%
surveillance  20%
```

建议：

```text
L0/L1:
standard 100%

L2:
standard      60%
cargo         20%
surveillance  20%
emergency      0%

L3+:
45 / 25 / 10 / 20
```

---

# 17. Priority Curriculum

```text
L0/L1:
fixed nominal priority

L2:
开始随机 normal/elevated/high

L3+:
正式 priority distribution
+ 完整 waiting/SOC urgency semantics
```

---

# 18. L2b：IMMEDIATE_ONLY + High/Dense Traffic

Teacher 权限保持不变，只提升 density。

建议：

`ρ = 0.70–0.85`

`N_init = 10–30` 作为 stress-rich 训练区间。

重点增加：
- leader/follower 高频变化；
- predecessor/successor 高频变化；
- merge interface congestion；
- ring occupancy 高。

暂不同时加入最强 burst + 最极端 resource stress。

---

# 19. L3a：LABEL_ONLY Teacher

## 19.1 关键切换

Teacher 可以计算“它会怎么修”，但**不能修改环境执行动作**。

执行链：

```text
π_L
→ a_requested
→ ExecutionScheduler
→ TrajectoryGenerator
→ SafetyFilter
→ PredictiveShield if required
```

Teacher 只用于：
- disagreement；
- hypothetical correction；
- optional BC auxiliary target；
- diagnostics。

## 19.2 场景

| 维度 | L3a |
|---|---|
| demand | Medium / High Poisson |
| ρ | 0.55–0.80 |
| mission | 正式 mix |
| priority | 正式 |
| SOC | U(0.30,1.0) |
| LZ busy | 0–2 |
| gate | uniform + asymmetric |
| warm-start | ~30% |
| geometry | nominal |
| burst | 少量或暂不开 |

## 19.3 新主指标
主要看：

```text
teacher disagreement rate
shield intervention rate
```

理想趋势：两者都下降，同时 completion/throughput 不下降。

---

# 20. L3b：LABEL_ONLY + Full Domain Randomization

逐步打开：
- full SOC `U(0.15,1.0)`；
- emergency mission；
- full priority；
- LZ initial busy 0–3；
- random remaining service time；
- strong gate imbalance；
- external queue build-up；
- bursty demand；
- variable-N；
- 长 episode recovery dynamics。

仍保持 geometry nominal。

---

# 21. LZ / Resource Curriculum

```text
L0/L1:
6/6 free

L2:
5–6 free

L3/L4:
3–6 free
```

若 reset 时 LZ busy，必须同时构造合法 service lifecycle：
- 对应 UAV；
- remaining service time；
- reservation；
- final/landing occupancy；
- LZ assignment consistency。

不能只设置 `busy=True`。

---

# 22. L4：Teacher OFF — Nominal Student Consolidation

## 22.1 模式

```text
Teacher = OFF
High-level = OFF
Geometry = nominal
ExecutionScheduler = ON
TrajectoryGenerator = ON
SafetyFilter = ON
PredictiveShield = ON
```

## 22.2 Distribution

| 维度 | L4 |
|---|---|
| from-empty | 80–90% |
| warm-start | 10–20% |
| ρ | 0.45–0.90 |
| Poisson | ON |
| burst | ON |
| gate imbalance | ON |
| mission | full |
| SOC | U(0.15,1.0) |
| priority | full |
| LZ | randomized |
| variable-N | ON |
| teacher | OFF |
| shield | ON |
| geometry | nominal |

推荐初始场景比例：

```text
normal/medium    35%
high             35%
bursty           20%
near-capacity     5%
overload          5%
```

Overload 不应主导 gradient。

---

# 23. Variable-N：必须记录多个 N

至少区分：

```text
N_requested
N_initial
N_active(t)
N_active_peak
```

`finite cohort N=60` 不等价于 `60 simultaneously active UAV`。

L4 warm-start 可覆盖：

`N_init ∈ [5,40]`

但主体集中在 5–25，较少采样 25–40。

可把 50/60 或明显高于训练 envelope 的 active density 留作 OOD。

---

# 24. Episode Length Curriculum

| 阶段 | 建议 episode |
|---|---:|
| L0 | finite cohort until cleared |
| L1 | 600–1200 s |
| L2 | 900–1800 s |
| L3 | 1200–2400 s |
| L4 | 1800–3600 s |
| Qualification | 3600 s 或正式协议 |

早期不需要全部 3600 s，否则 reset diversity 太低。

后期需要覆盖：
- queue formation；
- congestion recovery；
- burst aftermath；
- SOC accumulation；
- starvation；
- LZ resource recycling。

正式 continuous-flow evaluation 建议分：

```text
warm-up
measurement
resolution/cooldown
```

---

# 25. Anchor Scenarios

固定场景用于训练过程回归测试，不随随机化变化。

| Anchor | 描述 |
|---|---|
| A1 | single UAV full lifecycle |
| A2 | 5 UAV simple O→M / M→I |
| A3 | 12 UAV four-gate symmetric |
| A4 | 20 UAV asymmetric medium |
| A5 | deterministic merge-conflict witness |
| A6 | dense predecessor/successor exchange |
| A7 | low-SOC priority witness |
| A8 | LZ busy resource witness |

每若干 PPO updates 固定评估一次，用于检测：
- catastrophic forgetting；
- reward hacking；
- teacher 退火回归；
- shield dependency。

---

# 26. 3D 可视化的训练用途

每个 milestone 固定回放至少 5–10 个代表 episode。

重点检查：
- 长期无意义绕圈；
- release collapse；
- 全体同时 aggressive release；
- merge interface 反复加减速；
- shield ping-pong；
- 某 gate starvation；
- emergency 粗暴插队；
- low-SOC UAV 被压制；
- LZ busy 后 inner ring 堆积；
- intervention 是否集中在固定空间位置。

推荐显示：
- requested/applied action；
- teacher label；
- shield intervention；
- predicted min separation；
- conflict pair；
- leader/follower；
- predecessor/successor；
- reservation；
- phase/ring；
- SOC / mission / priority。

---

# 27. Teacher / Shield 指标归一化

主指标不要只用 interventions per simulation step。

优先使用：

```text
interventions / 100 release intents
```

同时报告：
- interventions / UAV；
- correction magnitude；
- P95 correction magnitude；
- repeatedly rescued UAV count；
- max interventions for one UAV；
- intervention phase/location distribution。

---

# 28. L3 → L4 晋级门槛

不要只按训练步数晋级。

## Safety
Protected validation：

```text
collision = 0
NMAC = 0
hard separation violation = 0
```

## Autonomy
Shield intervention 已明显下降。

`≤15 interventions / 100 release intents` 可作为初始工程 gate，pilot 后冻结。

## Teacher disagreement
不能仍有大量 release intent 被 label-only teacher 判定需要大幅修正。

## Progress
防止：

```text
release_request_rate → 0
```

## Efficiency
相对 Full Teacher baseline，completion / throughput / delay 不得发生不可接受退化。

---

# 29. L4 Nominal Qualification

L4 结束后得到：

> `low_nominal_qualified`

不是最终交给 high-level 的 checkpoint。

工程初始 gate：

### ID

```text
shield <= 10 / 100 release intents
```

### OOD stress

```text
shield <= 20 / 100 release intents
```

并保持：

```text
collision = 0
NMAC = 0
hard violation = 0
```

这些阈值必须在正式 qualification 结果揭晓前冻结。

---

# 30. 为什么 L4 后还需要 Geometry Curriculum

若低层只在 nominal geometry `g0` 训练，而高层未来输出 `g1 != g0`，低层将面对 OOD 状态。

若低层只熟悉 nominal geometry，则任何后续系统级 geometry 变化都会使低层进入 OOD，因而在进入系统级重构/高层阶段前必须先完成 static geometry generalization。

因此高层训练前必须完成：

> **L5 Geometry Generalization**

---

# 31. Geometry 的职责边界

低层：

```text
given current legal geometry
→ how should each UAV coordinate?
```

高层：

```text
given terminal/system state
→ which geometry should be selected?
```

ReconfigurationCoordinator：
- geometry transition feasibility；
- committed trajectory protection；
- reservation consistency；
- transition legality。

低层不负责几何本身的执行变形。

---

# 32. Geometry Domain

不要穷举所有半径。

先定义：

`G_physically_feasible`

例如满足：
- 每个 ring radius 在允许范围内；
- O–M / M–I 最小间距；
- kinematics / curvature；
- airspace footprint；
- transfer geometry；
- final path；
- safety。

之后通过运行能力得到：

`G_operational ⊆ G_physically_feasible`

即不仅物理合法，而且低层能够稳定运行的 geometry envelope。

---

# 33. L5a：Near-nominal Static Geometry

## 33.1 目标
验证 nominal student 对轻微 geometry shift 的适应能力。

## 33.2 Geometry
每个 episode 采样一个 geometry，episode 内固定。

初始可用：

`r_k = r_k0 * (1 + epsilon_k)`

`epsilon_k ~ U(-0.05, 0.05)`

±5% 只是工程起点；最终范围由独立 geometry/reconfiguration 规格文件定义的 operational geometry envelope 决定。

## 33.3 模式

```text
Teacher = LABEL_ONLY
Shield = ON
High-level = OFF
Geometry = static per episode
```

不要重新回 FULL teacher。

## 33.4 Traffic
先使用 medium / medium-high、moderate N、降低 compound stress，以隔离 geometry shift。

---

# 34. L5b：Full Feasible Static Geometry

## 34.1 目标
覆盖正式系统允许的完整 low-level operational geometry envelope。

## 34.2 Structured geometry sampling
推荐初始比例：

```text
20% nominal / near-nominal
20% boundary cases
50% random feasible interior
10% hard geometry
```

可使用：
- stratified sampling；
- Latin Hypercube；
- Sobol；
- 手工 geometry classes。

## 34.3 Geometry classes
- nominal；
- compact terminal；
- expanded terminal；
- narrow O–M gap；
- narrow M–I gap；
- large outer；
- small inner；
- feasible boundary；
- random interior；
- hard-but-feasible。

## 34.4 模式

```text
Teacher = OFF
Shield = ON
```

若 L5a 出现严重 generalization gap，可以短暂使用 label-only teacher 做诊断或 auxiliary supervision，但不建议恢复 teacher correction 到执行链。

---

# 35. Geometry Feasibility / Capacity Map

建议在 L5 前或并行用 rule-safe / Full Teacher baseline 对 geometry domain 做采样。

记录：

```text
mu(g)
delay(g)
energy(g)
teacher burden(g)
shield burden(g)
throughput(g)
feasibility(g)
```

作用：
- 识别 low-level 可稳定运行的 geometry envelope；
- 排除运行质量极差 geometry；
- 为后续系统级 geometry/reconfiguration 设计提供数据依据。

---

# 36. Geometry 与 Traffic Load 的归一化

不同 geometry 可能拥有不同容量 `μ_hat(g)`。

因此同一 absolute `λ` 在不同 geometry 下具有不同相对难度。

L5 应同时保留两类实验：

## Fixed absolute demand
回答：同一真实需求下，不同 geometry 的运行表现如何？

## Fixed relative load
使用：

`ρ(g) = λ / μ_hat(g)`

回答：相似交通难度下，low-level 是否具备 geometry robustness？

---

# 37. L5c：Post-Reconfiguration Recovery Compatibility

## 37.1 文档边界

本阶段只讨论 **low-level policy 自身需要承担的能力**。

Dynamic reconfiguration 的以下内容不在本文件定义：

- 哪些 UAV 属于 affected set；
- Global Freeze / Affected-set Freeze 的选择；
- ReconfigurationCoordinator 的状态机；
- committed trajectory protection 的系统实现；
- radial / vertical transition rate；
- cooldown / dwell time；
- switching cost；
- geometry-state cost；
- high-level observation / reward；
- high-level reconfiguration economics calibration。

这些内容统一见：

> `RECONFIGURATION_AND_HIGH_LEVEL_INTEGRATION_SPEC.md`

---

## 37.2 Low-level 在 reconfiguration 期间的训练语义

如果当前系统设定为：

```text
reconfiguration active
+ UAV is controlled by reconfiguration mechanism
→ low-level intention is masked / not authoritative
```

则该 UAV 在这些 step 上 **不应被要求学习“如何执行 geometry transition”**。

必须保证：

```text
policy_action_valid = 0
```

或等价：

```text
actor_loss_mask = 0
```

原则是：

> coordinator / system controller 产生的运动不能被伪装成 `π_L` 自己采样的 PPO behavior action。

这与全文第 3 节的 `a_requested / a_teacher / a_applied` 动作语义保持一致。

---

## 37.3 L5c 真正要训练/验证的能力

L5c 的低层目标仅是：

> **当一次合法 geometry transition 已经由外部 reconfiguration mechanism 完成，low-level 重新获得 action authority 后，能够在新的 traffic state distribution 上恢复稳定协调。**

重构后的状态可能与“直接 reset 到同一静态 geometry”不同，例如：

- external queue 在 transition 期间累积；
- 多架 UAV 同时重新获得 release eligibility；
- leader/follower 与 predecessor/successor 关系发生变化；
- ring occupancy 与 headway distribution 改变；
- SOC 继续消耗；
- LZ / final resource 状态继续演化；
- arrival burst 可能尚未结束。

因此 L5c 主要处理：

```text
post-transition distribution shift
```

而不是：

```text
geometry morphing control
```

---

## 37.4 推荐训练方式

使用外部 scripted transition 或预生成 transition trace 构造训练 episode：

```text
stable traffic
→ external reconfiguration interval
→ low-level authority restored
→ continue episode and train π_L
```

在 transition interval：

- 被系统接管 UAV 的 actor loss mask = 0；
- low-level 不对 coordinator 动作负责。

在 transition complete 以后：

- `policy_action_valid = 1`；
- 正常 PPO rollout 恢复；
- 重点采集 transition 后最初一段 traffic recovery 样本。

L5c 不需要 high-level PPO。

---

## 37.5 L5c 场景建议

首轮建议：

- geometry pair 来自已经通过 L5b qualification 的 `G_operational`；
- medium / high traffic；
- moderate variable-N；
- realistic mission / SOC / priority / LZ randomization；
- 每个 episode 只包含 1 次 transition；
- transition 后保留足够长 recovery window；
- 不同时加入最强 overload 与最极端 compound stress。

稳定后可扩展到：

- 2 次 transition / episode；
- burst during transition；
- asymmetric gate demand；
- higher active-N；
- reduced LZ availability。

---

## 37.6 主要低层指标

重点记录：

```text
post-transition shield intervention / 100 release intents
post-transition release-request rate
post-transition release acceptance rate
queue clearance time
throughput recovery time
completion rate
gate-specific delay
priority / low-SOC starvation
```

可定义：

`T_recover`

表示从 low-level authority 恢复，到系统关键指标重新进入预定义稳定区间所需时间。

---

## 37.7 典型失败模式

### Simultaneous aggressive release

```text
transition complete
→ many UAV become eligible
→ policy releases too aggressively
→ shield spike
```

### Conservative recovery

```text
transition complete
→ policy remains passive
→ backlog cannot clear
```

### Merge-interface shock

transition 后 predecessor / successor 高频变化，导致 merge 行为抖动。

### Persistent shield dependency

transition 后很长时间 shield intervention 仍显著高于 static-geometry baseline。

---

## 37.8 L5c 晋级标准

L5c 通过的核心不是 reconfiguration mechanism 本身是否正确，而是：

1. low-level 恢复 action authority 后 safety 继续满足；
2. post-transition shield intervention 能回落；
3. queue / throughput 能在有限时间内恢复；
4. 不出现 gate starvation / low-SOC starvation；
5. recovery performance 在多个 unseen geometry pair 上保持稳定。

Reconfiguration mechanism 自身的 safety、affected-set、状态连续性和成本，应在独立 reconfiguration 文档规定的 qualification 中验证，不在本文件重复定义。

---

# 38. Geometry Train / Validation / Test 分离

定义：

```text
G_train
G_validation
G_test
```

Test 至少包含：
- unseen interior；
- unseen boundary；
- interpolation；
- mild extrapolation；
- difficult witness geometry。

不要让训练看过所有 geometry，否则无法证明 generalization。

---

# 39. Geometry-aware Observation Ablation

可比较：

### A
`Topology features only`

### B
`Topology + explicit geometry features`

比较：
- ID；
- unseen geometry；
- shield burden；
- throughput；
- delay；
- sample efficiency。

如果 A 已表现良好，可能说明 topology representation 本身具有较强 geometry invariance。

---

# 40. Sensor Noise / Communication Delay

不要在主 L0–L5 早期加入。

先让 deterministic-observation student 通过 geometry qualification，再单独测试：
- position noise；
- velocity noise；
- SOC noise；
- observation delay；
- communication delay；
- dropped neighbor observation。

如果显著退化，再决定：
- domain-randomization fine-tuning；
- GRU/LSTM；
- temporal encoder。

---

# 41. Checkpoint 管理

每阶段至少保存：

```text
stage_entry
best_reward
best_low_intervention
best_validation
latest
stage_exit
```

每个 checkpoint 必须绑定：

```text
architecture_version
low_obs_version
reward_version
teacher_mode
scheduler_mode
shield_version
geometry_mode
traffic_profile
mission_profile
soc_profile
seed_partition
git_commit
config_hash
```

---

# 42. Seed 分区

正式训练前冻结：

```text
train seeds
validation seeds
qualification seeds
OOD seeds
```

Qualification seeds 不应在训练中反复查看并据此调参，避免 test leakage。

---

# 43. Reward Freeze 原则

L1 正式训练后 reward 主结构尽量冻结。

若必须修改：
1. 更新 reward version；
2. 旧/new run 不混为同一条实验线；
3. 重跑 anchors；
4. 重建 stage baseline；
5. 文档记录修改原因。

---

# 44. 标准日志

## Safety
- collision；
- NMAC；
- hard separation violation；
- minimum actual separation；
- minimum predicted separation；
- CPA / TTC distribution。

## Autonomy
- teacher intervention / 100 intents；
- teacher disagreement / 100 intents；
- correction magnitude；
- shield intervention / 100 intents；
- repeated shield rescue count。

## Progress
- release request rate；
- release acceptance rate；
- completion；
- throughput；
- queue growth；
- active N；
- peak active N。

## Efficiency
- total delay；
- ring wait；
- final wait；
- energy；
- mission completion；
- service time。

## Fairness / Priority
- gate-specific delay；
- mission-specific delay；
- priority-specific delay；
- starvation；
- low-SOC delay。

## Geometry（L5 起）
- `r_O/r_M/r_I`；
- geometry class；
- transition；
- capacity-normalized load；
- shield vs geometry。

---

# 45. Stop / Diagnose Conditions

出现以下情况，不应只继续增加 PPO steps。

## 45.1 Teacher dependency plateau

```text
reward ↑
teacher intervention flat
```

检查：teacher leakage、correction penalty、observation insufficiency。

## 45.2 Shield dependency plateau

```text
teacher OFF
shield intervention high and flat
```

检查：policy safety learning、predictive features、reward、scheduler/shield boundary。

## 45.3 Conservative collapse

```text
release_request_rate → 0
safety excellent
throughput collapses
```

## 45.4 Aggressive collapse

```text
release_request very high
shield rescues everything
```

## 45.5 Gate starvation
整体正常但某 gate 长期排队。

## 45.6 Curriculum regression
新阶段使旧 anchor 明显退化。

## 45.7 Geometry boundary collapse
只在 geometry boundary 频繁 shield。需判断：
- geometry 是否应移出 operational domain；
- observation 是否缺 geometry 信息；
- 当前 absolute demand 是否已超过 `μ(g)`。

---

# 46. 阶段总表

| 阶段 | Teacher | Geometry | Traffic | Randomization | 核心目标 |
|---|---|---|---|---|---|
| C0 | Full | nominal | capacity sweep | nominal | 标定 μ_hat |
| L0 | Full | nominal | low/medium deterministic | 极低 | teacher dataset / BC |
| L1 | Full | nominal | medium | homogeneous | 基本 coordination |
| L1b | Full | nominal | variable medium | N/gate | 场景扩展 |
| L2 | Immediate | nominal | medium-variable | light mission/SOC | 去 future search 依赖 |
| L2b | Immediate | nominal | high/dense | moderate | dense coordination |
| L3a | Label-only | nominal | medium/high | mission/priority/SOC | student 自主执行 |
| L3b | Label-only | nominal | high/bursty | full | 降低 shield dependence |
| L4 | OFF | nominal | mixed ID | full | nominal student consolidation |
| L5a | Label-only | near-nominal static | medium/high | moderate | geometry adaptation |
| L5b | OFF | full static feasible | mixed | structured full | geometry generalization |
| L5c | OFF | post-transition states | mixed | realistic | post-reconfiguration recovery compatibility |
| Q_L | OFF | ID + OOD | ID + OOD | stress matrix | 最终 qualification |

---

# 47. 推荐实际执行顺序

```text
1. C0 capacity calibration
2. Freeze μ_hat and traffic level definitions
3. Freeze train/val/OOD/qualification seed partitions
4. Validate safe warm-start generator
5. Collect L0 teacher dataset
6. Optional BC pilot
7. L1 FULL-teacher PPO
8. L1b traffic/N/gate expansion
9. FULL → IMMEDIATE_ONLY（保持场景基本不变）
10. L2 stabilize
11. L2b dense traffic
12. IMMEDIATE_ONLY → LABEL_ONLY（保持场景基本不变）
13. L3a autonomous execution
14. L3b full mission/SOC/LZ/burst randomization
15. LABEL_ONLY → OFF
16. L4 nominal student consolidation
17. Nominal qualification
18. Geometry feasibility/capacity sampling
19. L5a near-nominal static geometry
20. L5b full feasible static geometry
21. L5c post-reconfiguration recovery compatibility
22. Final low-level qualification Q_L
23. Freeze L5_geo_qualified checkpoint
24. Hand off the frozen low-level checkpoint to the separate reconfiguration / high-level workflow
```

---

# 48. Final Qualification Matrix

## 48.1 Traffic quantity
Finite cohort：

```text
N_requested = 1, 5, 10, 20, 40, 60
```

Warm-start exact active：

```text
N_init = 5, 10, 20, 40, 50/60
```

分别研究 total demand 与 simultaneous density。

## 48.2 Demand

```text
medium Poisson
high Poisson
bursty NHPP
near-capacity
overload
OOD arrival process
```

## 48.3 Gate

```text
uniform
mild imbalance
strong imbalance
dominant-gate switch
```

## 48.4 Resource / mission

```text
nominal mission mix
cargo-heavy
emergency spike
low-SOC stress
reduced LZ availability
long service-time stress
```

## 48.5 Geometry

```text
nominal
near-nominal unseen
unseen interior
boundary
hard feasible
post-transition recovery traces
unseen transition end-states
mild static extrapolation
```

## 48.6 OOD arrival mechanism
至少保留一个训练中从未使用的 arrival-generation mechanism。

---

# 49. 最终成功应展示的学习轨迹

```text
FULL Teacher
    ↓
policy learns basic intention decisions

IMMEDIATE_ONLY
    ↓
policy no longer relies on long rule-based repair search

LABEL_ONLY
    ↓
policy executes autonomously; teacher only observes

OFF
    ↓
student + minimal predictive shield

STATIC GEOMETRY GENERALIZATION
    ↓
student works across operational geometry envelope

POST-RECONFIGURATION RECOVERY COMPATIBILITY
    ↓
student restores traffic after external geometry transition

FINAL QUALIFICATION
    ↓
freeze π_L
```

希望趋势：

```text
teacher correction rate       ↓
teacher correction magnitude  ↓
teacher disagreement          ↓
shield intervention           ↓
```

同时：

```text
completion / throughput maintained or improved
collision = 0
NMAC = 0
hard violation = 0
```

---

# 50. 必须避免的典型错误

1. 未经 C0 就定义 medium/high；
2. L1 一次性打开所有随机性；
3. Teacher 权限与环境难度同时大幅变化；
4. 只 from-empty，导致样本效率差；
5. 只 warm-start，失去完整 lifecycle；
6. warm-start 随机撒 UAV；
7. overload 主导训练；
8. teacher-corrected action 替换 PPO behavior action；
9. shield 等 breach 后才介入；
10. L4 nominal checkpoint 直接作为 geometry-generalized low-level；
11. 穷举所有 ring radius；
12. 只在 nominal geometry 训练却宣称 geometry generalization；
13. 把 reconfiguration 期间系统接管动作当成 PPO actor action；
14. qualification 结果出来后再修改 gate threshold；
15. reward 版本变化却仍把曲线当成同一 run；
16. train/validation/test seeds 泄漏。

---
# 51. 推荐配置结构

```text
configs/
  low_v2/
    common.yaml
    c0_capacity.yaml
    l0_teacher_dataset.yaml
    l1_full_teacher.yaml
    l1b_full_teacher_variable.yaml
    l2_immediate.yaml
    l2b_immediate_dense.yaml
    l3a_label_only.yaml
    l3b_label_only_full_domain.yaml
    l4_student_nominal.yaml
    l5a_geometry_near.yaml
    l5b_geometry_full.yaml
    l5c_post_reconfiguration_recovery.yaml
    qualification_id.yaml
    qualification_ood.yaml
```

所有 stage config 继承 `common.yaml`，防止无意修改 hard-safety 和 topology 核心定义。

---

# 52. 实验命名规范

例如：

```text
A5V2_L1_fullT_med_seed001
A5V2_L2_immediate_dense_seed003
A5V2_L3b_label_bursty_seed007
A5V2_L4_student_nominal_seed012
A5V2_L5b_geo_static_seed020
A5V2_L5c_postreconf_seed025
```

每个 run 至少记录：

```text
stage
run_id
git_commit
config_hash
parent_checkpoint
teacher_mode
scheduler_mode
shield_version
geometry_mode
traffic_mode
seed
```

---

# 53. 低层训练结束时应形成的产物

1. nominal `μ_hat` capacity calibration；
2. traffic-level definitions；
3. Scenario Class library；
4. safe warm-start generator；
5. train/validation/OOD/qualification seed partitions；
6. L0 teacher dataset；
7. L1/L2/L3/L4 training curves；
8. teacher dependence curves；
9. shield intervention curves；
10. nominal-qualified checkpoint；
11. geometry feasibility/capacity map；
12. L5 static geometry-generalized checkpoint；
13. static geometry qualification；
14. post-reconfiguration recovery compatibility report；
15. OOD-N qualification；
16. fixed 3D anchor replay collection；
17. final low-level qualification report；
18. frozen `L5_geo_qualified` checkpoint。

低层文档到此完成职责闭环。之后所有 dynamic reconfiguration mechanism 与 high-level integration 工作转入：

> `RECONFIGURATION_AND_HIGH_LEVEL_INTEGRATION_SPEC.md`

---

# 54. 最终决策规则

后续如果训练出现问题，优先排查顺序：

1. 场景是否物理/运行上有意义；
2. policy 是否有足够 observation 信息；
3. teacher / scheduler / shield 是否越权；
4. reward 是否存在 loophole 或冲突；
5. curriculum 是否一次改变了太多因素；
6. PPO 实现与 rollout semantics 是否正确；
7. 最后才考虑继续增加训练步数、改变网络规模或换算法。

低层研究的核心不是“把 reward 训练得尽可能高”，而是证明：

> 在复杂、可变、可泛化的 terminal traffic 中，低层共享策略已经承担正常交通协调责任；规则系统只负责明确的执行约束与最低限度安全保证。

---

# 55. 一句话主线

> **先标定容量，再在 nominal geometry 上用 teacher 教会低层自主协调并逐步退掉 teacher；在 full domain randomization 下形成 student-only 能力；随后完成 static geometry generalization，并仅针对“外部重构完成后 low-level 重新接管”的状态分布做 recovery compatibility；最终通过 ID/OOD qualification 后冻结低层。Dynamic reconfiguration 的机制、成本与 high-level integration 全部由独立规格文件管理。**
