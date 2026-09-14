# A5-v2 低层策略训练纲领与场景课程计划

> **V4.2 active-schema amendment (2026-09-10).** V4.2 preserves V4.1's 26
> execution intents but replaces the flat categorical actor with a two-way
> WAIT/RELEASE gate and a conditional 25-way RELEASE proposal head. The schema
> is `low_action_schema_v4_2_hierarchical`; no independent V3 speed/merge
> heads are reintroduced. V4.2 uses a fresh actor and a read-only V4.1 I0
> hierarchical-target view. Its requested log-probability is `log P(WAIT)` or
> `log P(RELEASE)+log P(proposal|RELEASE)`. PPO remains disabled pending its
> separate authorization.
>
> **V4.1 historical amendment (2026-09-10).** V4's 30-class catalogue is
> superseded for all new work due to WAIT action-space overparameterization.
> V4.1 is `1 WAIT + 25 RELEASE(speed×merge) = 26`, with
> `low_action_schema_v4_1_joint_discrete_26`. WAIT is one canonical
> no-release/no-reservation policy decision; holder speed remains execution/
> shield owned. V4 30-class checkpoints and tapes are historical/read-only,
> and V4.1 learning must use a fresh actor. I0 gates decision-relevant expert
> target families rather than requiring every catalogue action.

> **文档定位**
> 本文件是 A5-v2 低层飞行意图策略训练阶段的纲领性文件，用于统一后续代码配置、数据采集、模仿学习、DAgger、PPO、实验记录、3D 回放、阶段晋级、geometry generalization 与最终 qualification。
> 它不是某一次 run 的参数清单，而是整个低层训练阶段的**方法学、数据分布与场景协议**。
>
> **适用架构**
> - Shared low-level policy `π_L`
> - Topology-aware Observation V2（后续若升版必须单独 version）
> - Low-Level Action V4.1：joint discrete scheduling intention
> - `N_joint = 26`（1 WAIT + 25 RELEASE）
> - TeacherScheduler：训练期 **expert labeler / auditor**
> - ExecutionScheduler：始终存在，仅检查/执行 policy 请求的一个候选
> - Canonical Trajectory Generator
> - SafetyFilter
> - Predictive Safety Shield
> - High-level policy 在低层完成 geometry qualification 前保持关闭
>
> **建议版本号**：`A5V2-TRAINING-CURRICULUM-v4`
>
> **v4 核心变化**
> 1. 废弃主线中的 `FULL Teacher PPO → IMMEDIATE_ONLY → LABEL_ONLY → OFF` 执行权限退火。
> 2. Teacher 不再作为 PPO 阶段长期修改动作的控制器；其主职责改为 **state-conditioned expert labeling、DAgger supervision 与 diagnostics**。
> 3. 主训练范式改为：
>
> ```text
> Expert dataset
> → joint BC bootstrap
> → Safe DAgger
> → student-only imitation qualification
> → student-controlled PPO
> → environment/domain curriculum
> → geometry curriculum
> → final qualification
> ```
>
> 4. SafetyFilter / Predictive Safety Shield 仍属于训练与部署的硬安全架构；Teacher 不承担最终 safety authority。
> 5. PPO 只有在 student 已能在 Teacher 不接管执行的条件下稳定闭环运行后才允许启动。
> 6. V3 factorized action / per-head Teacher supervision 视为历史方案，不再属于本纲领。
>
> **文档边界**
> - 本文件只规定 low-level policy 的训练、验证、静态 geometry 泛化、post-reconfiguration recovery compatibility 与最终 qualification。
> - Dynamic reconfiguration 的状态机、Affected-set Freeze、重构持续时间、cooldown、switching cost、高层 reward / observation 以及 H0/CH0 economics calibration 不在本文件定义。
> - 上述内容统一放入 `RECONFIGURATION_AND_HIGH_LEVEL_INTEGRATION_SPEC.md`。
> - 动态重构期间若 UAV 被外部机制接管，则该时间步不作为 low-level actor 的 PPO behavior action。

---

# 1. 总体目标

A5-v2 低层训练的目标不是得到一个“高 reward PPO”，而是得到一个能够在复杂 terminal traffic 中自主完成正常 sequencing / coordination，并仅在少数极端情况下依赖 predictive shield 的共享低层策略。

最终策略应满足：

1. 根据本机、局部拓扑邻机、资源和 geometry 状态，自主输出一个 V4.1 joint scheduling intention：
   - 一个 canonical WAIT（无 release/reservation proposal）；
   - 或 RELEASE speed×merge category。
2. 正常 sequencing / conflict anticipation 主要由 learned policy 完成，而不是 Teacher 或 ExecutionScheduler 长期搜索替代动作。
3. TeacherScheduler 不属于最终运行策略。
4. 最终部署保留：
   - ExecutionScheduler；
   - reservation / mutual exclusion / hard feasibility；
   - canonical trajectory；
   - SafetyFilter；
   - Predictive Safety Shield。
5. Student-only 条件下保持：
   - collision = 0；
   - NMAC = 0；
   - hard separation violation = 0；
   - completion / throughput 不崩溃；
   - queue 不长期发散；
   - shield burden 足够低。
6. 在 high-level training 前，low-level 必须完成 static geometry generalization。
7. policy 的最终目标不是复制 Teacher，而是：
   - 先通过 imitation 获得稳定可行的 coordination prior；
   - 再通过 PPO 优化 throughput / delay / energy / fairness，并允许在 safety 约束下超过 Teacher。

完整主线：

```text
C0  Capacity calibration
 ↓
S0  System / action / safety / expert qualification
 ↓
I0  Structured expert dataset
 ↓
I1  Joint BC bootstrap
 ↓
I2  Safe DAgger state-coverage aggregation
 ↓
I3  Student-only nominal imitation qualification
 ↓
P1  Student-controlled nominal PPO
 ↓
P2  Traffic / N / gate expansion
 ↓
P3  Mission / SOC / priority / resource expansion
 ↓
P4  Dense / bursty / variable-N full-domain PPO
 ↓
P5  Teacher-OFF nominal consolidation
 ↓
G1  Near-nominal static geometry
 ↓
G2  Full feasible static geometry
 ↓
G3  Post-reconfiguration recovery compatibility
 ↓
Q_L Final low-level qualification
 ↓
Freeze π_L
 ↓
Hand off frozen π_L to reconfiguration / high-level workflow
```

---

# 2. 核心方法学原则

## 2.1 两条独立难度轴

### 轴 A：Learning Authority / Supervision

主线不再按 Teacher 执行权限退火，而按 supervision 角色变化：

```text
EXPERT DATA
    ↓
BC
    ↓
LABEL-ONLY DAgger
    ↓
LABEL-ONLY diagnostics during PPO
    ↓
OFF
```

关键原则：

> Teacher supervision 可以逐步减少，但 Teacher 不应在主要 PPO 阶段长期成为环境实际动作的控制者。

### 轴 B：Environment / Traffic Complexity

```text
simple
  ↓
medium stationary
  ↓
variable density / N / gate
  ↓
mission / SOC / priority / resource
  ↓
dense / bursty / variable-N
  ↓
geometry variation
  ↓
OOD compound stress
```

**一次阶段切换原则上只主要改变一个难度来源。**

避免：

```text
nominal imitation
   ↓
bursty + low SOC + asymmetric + reduced LZ + geometry shift + new PPO objective
```

否则失败后无法归因。

---

## 2.2 为什么采用 Expert BC → DAgger → PPO

三类学习机制承担不同职责：

### BC：解决 cold start

```text
expert states
→ expert joint action
→ supervised policy bootstrap
```

### DAgger：解决 covariate shift

Student 自己控制闭环并产生自己会访问的状态；Teacher 只对这些状态给 expert label：

```text
student policy
→ student-visited states
→ Teacher label
→ aggregate dataset
→ retrain student
```

这避免只学习 Teacher trajectory distribution。

### PPO：解决 expert suboptimality

当 student 已经能稳定闭环后，PPO 才负责优化：

- throughput；
- delay；
- energy；
- fairness；
- starvation；
- recovery efficiency。

Teacher 是 safe / feasible / explainable expert，而不是最优策略真值。

---

## 2.3 Safety、Execution、Teacher 严格分工

### TeacherScheduler

主线职责：

- 对当前 state 评估 30-action V4 catalog；
- 输出确定性的 `a_teacher_expert`；
- 为 I0/I1 提供 expert dataset；
- 为 I2 Safe DAgger 标注 student-visited states；
- 在 I3/P1–P4 可作为 label-only diagnostic；
- 提供 disagreement / candidate-score / label provenance。

Teacher **不得**在主要 I2–P4 learning rollout 中改写环境实际执行动作。

允许 Teacher 直接控制的情况仅限：

- C0 capacity / rule-safe expert baseline；
- offline expert rollout / dataset generation；
- 单独声明的 ablation；
- debug / witness replay。

这些 transition 不得被伪装成 student PPO behavior。

### ExecutionScheduler

始终存在：

```text
policy samples ONE joint action
→ decode
→ ExecutionScheduler checks ONE proposal
```

职责：

- reservation/resource consistency；
- headway / LZ mutual exclusion；
- single-candidate feasibility；
- runtime lifecycle consistency。

禁止在 deployment/student learning 中枚举 30-action catalog 替 policy 找最佳动作。

### SafetyFilter

始终存在：

- 对 canonical trajectory 做 hard verification；
- fail closed；
- 不充当日常 policy optimizer。

### Predictive Safety Shield

始终存在：

- 在 predicted hard-safety breach 前进行最小干预；
- 可 hold / block uncommitted release；
- 可做允许范围内的最小速度/时序安全修正；
- committed trajectory 保护保持不变。

---

# 3. Low-Level Action V4 语义

## 3.1 Joint catalog

V4 policy 输出一个 joint categorical action：

```text
a_requested_joint ∈ {0, ..., 29}
```

catalog：

| family | count | semantic |
|---|---:|---|
| WAIT | 5 | `[WAIT, speed∈{-1,-0.5,0,+0.5,+1}, merge=0]` |
| RELEASE | 25 | `[RELEASE, speed×merge]` |

WAIT 保留 speed residual，因为 holder/runtime shield 仍可使用；WAIT 的 merge residual canonicalize 为 0。

## 3.2 Actor

```text
Observation V2
→ shared encoder
→ 30 joint logits
→ Categorical(30)
```

不再使用 V3 的三个独立 Teacher learning targets。

## 3.3 PPO provenance

必须区分：

```text
a_requested_joint
a_teacher_expert
a_applied
```

其中：

- `a_requested_joint`：policy 实际采样；
- `a_teacher_expert`：Teacher 对当前 state 的 expert label；
- `a_applied`：Execution/Safety 后真正执行的动作。

关键约束：

> PPO rollout buffer、old log-prob、entropy 和 ratio 只能对应 `a_requested_joint`。

Teacher label、ExecutionScheduler result、SafetyFilter/Shield action 不得替代 behavior action。

---

# 4. Imitation / PPO 的因果边界

## 4.1 主线禁止 FULL-Teacher PPO

禁止主线使用：

```text
student samples action
→ Teacher replaces action
→ environment transition
→ PPO credits student request
```

因为这会形成 Teacher-wrapped transition distribution，并削弱 action → reward / next-state 的因果对应。

## 4.2 PPO 阶段

PPO 阶段执行链必须是：

```text
π_L
→ a_requested_joint
→ ExecutionScheduler
→ TrajectoryGenerator
→ SafetyFilter
→ PredictiveShield
→ environment
```

Teacher：

```text
state
→ a_teacher_expert
→ diagnostics only
```

默认：

```text
teacher BC auxiliary weight = 0
```

若需要 small BC auxiliary，必须作为单独 ablation / version，不得无版本地混入主线。

## 4.3 Shield 仍属于环境

Shield 是最终部署组成部分，因此 PPO 可以在固定 shield 语义下学习。

但若 shield burden 高，则表示 actor 仍未形成正常 coordination，不能用“shield 保证 safety”掩盖策略失败。

---

# 5. Observation V2 要求

Observation V2 仍以 topology-aware、scale-stable 表示为主。

至少覆盖：

## 5.1 Self / flight

- phase；
- current / target ring；
- normalized progress；
- speed；
- SOC；
- waiting time；
- mission；
- priority；
- gate；
- target LZ；
- release eligibility；
- reservation / lifecycle state。

## 5.2 Semantic topology slots

至少：

```text
same-ring leader
same-ring follower
target-ring predecessor
target-ring successor
nearest predicted conflict
```

推荐每 slot 包含：

- presence；
- gap / path distance；
- Euclidean distance；
- relative speed；
- closing rate；
- dCPA；
- tCPA；
- priority delta；
- reservation / committed status。

## 5.3 Resource / flow

- ring occupancy；
- interface/headway availability；
- assigned-LZ usability；
- free-LZ fraction；
- final resource pressure；
- external queue / congestion summary。

## 5.4 Geometry awareness

进入 G1 前必须确认 policy 能感知对 decision 有意义的 geometry 变化。

优先使用 dimensionless / physically meaningful features：

```text
distance-to-merge / reference distance
nominal-time-to-merge / reference time
relative ETA
relative gap
CPA / safe distance
TTC / reference time
```

## 5.5 Observation 升版原则

若发现 observation aliasing：

1. 先做 same-state / label / transition attribution；
2. 确认缺失状态确实部署可观测；
3. 新增最小 feature；
4. 提升 `low_obs_version`；
5. 重跑 I0/I1/I2 preflight；
6. 不允许把 Teacher 最终 label 直接作为 feature。

---

# 6. C0：Capacity Calibration

## 6.1 目标

在 nominal geometry 下得到经验可持续容量：

```text
μ_hat = sustainable arrival rate
ρ = λ / μ_hat
```

定义：

| level | offered load |
|---|---:|
| Low | `0.30–0.40 μ_hat` |
| Medium | `0.45–0.60 μ_hat` |
| High | `0.70–0.85 μ_hat` |
| Near-capacity | `0.85–0.95 μ_hat` |
| Overload | `1.05–1.20 μ_hat` |

## 6.2 控制架构

C0 不训练 policy：

```text
expert / rule-safe baseline = ON
ExecutionScheduler = ON
SafetyFilter = ON
PredictiveShield = ON
High-level = OFF
Geometry = nominal
PPO = OFF
```

Teacher 在此作为 capacity baseline controller 是允许的，因为没有 student PPO credit assignment。

## 6.3 Capacity gate

`μ_hat` 应满足预冻结的：

- completion/request threshold；
- measurement window queue slope near zero；
- safety 全满足。

Overload 只用于 robustness，不作为主体训练 gradient。

---

# 7. Traffic / Arrival Models

## 7.1 Deterministic interval

用途：

- C0 debug；
- I0 expert dataset；
- anchor；
- deterministic witness。

## 7.2 Stationary Poisson

主体训练流量。

均匀四入口：

```text
p_gate = [0.25, 0.25, 0.25, 0.25]
```

## 7.3 Asymmetric demand

Mild：

```text
[0.40, 0.20, 0.20, 0.20]
```

Strong：

```text
[0.55, 0.15, 0.15, 0.15]
```

dominant gate 必须轮换。

## 7.4 Bursty NHPP

示例：

```text
normal:   180–300 s @ 0.5 μ_hat
burst:     60–120 s @ 1.1–1.3 μ_hat
recovery: 180–300 s @ 0.3–0.4 μ_hat
```

## 7.5 OOD arrival process

至少保留一种训练中未使用的 arrival-generation mechanism，例如 MMPP。

## 7.6 CRN

policy / baseline 对比优先复用相同 arrival schedule。

---

# 8. Reset Curriculum

## 8.1 From-empty

```text
t = 0
active UAV = 0
```

优点：完整 lifecycle。
缺点：早期 interaction sample efficiency 低。

## 8.2 Safe Warm-start

必须保证：

- initial separation 安全；
- ring order 合法；
- topology consistent；
- reservation consistent；
- committed trajectory consistent；
- lifecycle consistent；
- reset 后不应因构造错误立即触发 shield。

## 8.3 推荐比例

| Stage | Warm-start | From-empty |
|---|---:|---:|
| I0 expert data | 60–70% | 30–40% |
| I1 BC | offline dataset | offline dataset |
| I2 Safe DAgger | 50–60% | 40–50% |
| I3 imitation qualification | 30–40% | 60–70% |
| P1 nominal PPO | 20–30% | 70–80% |
| P2/P3 | 15–25% | 75–85% |
| P4/P5 | 10–20% | 80–90% |
| Qualification | separately reported | dominant |

Warm-start 是 state-coverage 工具，不是最终问题定义。

---

# 9. Scenario Classes

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
| S10 | Dynamic geometry transition | post-transition recovery |
| S11 | OOD compound | final stress |

后续 stage 尽量引用 Scenario Class，不重新定义互不兼容的 env。

---

# 10. S0：System Qualification

在任何新学习主线开始前冻结：

- V4 joint catalog；
- Observation version；
- ExecutionScheduler semantics；
- trajectory semantics；
- SafetyFilter / Shield；
- Teacher expert determinism；
- ActionTape exact replay；
- checkpoint/action schema；
- train/validation/qualification/OOD seed partition。

必须验证：

```text
same full state
→ same deterministic a_teacher_expert
```

以及：

```text
same requested joint action
→ deterministic decode
→ execution semantics stable
```

S0 PASS 只表示系统可训练，不表示 policy 已学会。

---

# 11. I0：Structured Expert Dataset

## 11.1 目标

构建：

```text
state
→ a_teacher_expert
```

而不是：

```text
requested action
→ teacher repair action
```

dataset 必须使用 V4 joint target。

## 11.2 Teacher

```text
Teacher = EXPERT LABEL
PPO = OFF
```

可使用 Teacher 自己的 rollout 产生一部分 expert states，但不得只依赖 Teacher trajectory。

## 11.3 Decision-event sampling

禁止简单把每个 raw tick 作为等权 imitation sample。

必须记录：

```text
raw ticks
unique/meaningful decision events
per-UAV temporal streak
expert target persistence
state similarity
```

对连续高度相似的 WAIT-expert states：

- 去重；
- temporal subsample；
- 或 bounded persistence weight。

不得让“等待持续时间”自动变成无限大的 WAIT class 权重。

## 11.4 Coverage

I0 必须主动包含：

- WAIT-expert；
- RELEASE-expert；
- WAIT 5 speed variants 所在状态；
- RELEASE 多种 speed×merge target；
- simple release opportunity；
- release rejection；
- predecessor/successor conflict；
- holder conflict；
- interface/resource constraint。

不要求人工 50/50，但必须保证 RELEASE-expert state 有足够统计支撑。

## 11.5 Source tagging

每条样本必须标记：

```text
source = expert_rollout / warm_start / from_empty / anchor / targeted_state
scenario_class
seed
decision_event_id
state_hash
teacher_joint_target
```

---

# 12. I1：Joint BC Bootstrap

## 12.1 目标

训练初始 student：

```text
π_0(a_joint | s)
```

loss：

```text
L_BC = CE(joint_logits, a_teacher_expert)
```

## 12.2 禁止

- per-head独立 Teacher target；
- release-only Teacher label；
- 用 joint index 差作为物理 correction magnitude；
- 人工篡改 Teacher label；
- 只看 global accuracy。

## 12.3 固定评估 cohort

建立不随 policy 改变的 frozen decision-state cohort：

```text
WAIT-expert states
RELEASE-expert states
different speed/merge targets
simple / conflict / resource cases
```

至少报告：

- joint top-1；
- joint top-k；
- WAIT recall；
- RELEASE recall；
- release macro-F1；
- speed disagreement；
- merge disagreement；
- joint CE；
- entropy。

## 12.4 I1 Gate

只有：

- joint CE 明显下降；
- WAIT 与 RELEASE 都有可用 discrimination；
- 不 all-WAIT；
- 不 all-RELEASE；
- fixed cohort 无严重 state-only aliasing；

才允许 I2。

---

# 13. I2：Safe DAgger

## 13.1 核心思想

DAgger 的状态由 student 自己产生：

```text
student policy
→ ExecutionScheduler
→ Safety stack
→ next state
```

Teacher 并行做：

```text
same visited state
→ a_teacher_expert
→ dataset label
```

Teacher 不替换 student action。

## 13.2 Round protocol

每轮：

```text
DAgger-k student checkpoint
→ run closed-loop student episodes
→ collect student-visited decision events
→ query Teacher expert
→ aggregate D_k
→ retrain/fine-tune joint BC
→ fixed-state + closed-loop validation
```

建议从 2–4 轮 pilot 开始，不按固定轮数硬晋级。

## 13.3 State-source mixture

aggregated dataset 至少包含：

```text
expert bootstrap states
current student states
previous student failure states
warm-start states
from-empty states
anchor states
```

如果出现 all-WAIT failure，必须把其产生的大量 RELEASE-expert recovery states 加入 dataset。

如果出现 aggressive RELEASE failure，也必须收集对应 WAIT-expert states。

## 13.4 Safety

DAgger rollout 仍使用：

- ExecutionScheduler；
- SafetyFilter；
- Predictive Shield。

Teacher不是安全执行器。

## 13.5 DAgger 主指标

每轮记录：

```text
expert disagreement on student states
WAIT/RELEASE expert label distribution
student WAIT/RELEASE distribution
source-wise label distribution
raw tick vs decision-event ratio
release macro-F1 on fixed cohort
joint agreement on fixed cohort
closed-loop completion
throughput
queue slope
delay
shield burden
```

## 13.6 Covariate-shift audit

必须比较：

```text
expert-label distribution on:
training aggregated states
student rollout states
fixed natural validation states
```

若出现巨大差距，例如：

```text
training almost all WAIT
natural validation majority RELEASE
```

不得启动 PPO，应继续做 state-source attribution / DAgger coverage 修复。

---

# 14. I3：Student-only Nominal Imitation Qualification

## 14.1 执行

```text
Student = authoritative
Teacher = LABEL_ONLY diagnostics
PPO = OFF
Geometry = nominal
Traffic = medium
```

Teacher不得修改动作。

## 14.2 两种必须分开的验证

### A. Fixed-state capability

回答：

> policy 在相同的一组状态上是否真的学会 expert decision？

### B. Closed-loop capability

回答：

> policy 自己产生的 state distribution 是否健康？

不得用一个指标代替另一个。

## 14.3 PPO Admission Gate

只有 I3 PASS 后才允许 PPO。

至少要求：

### Safety

```text
collision = 0
NMAC = 0
hard violation = 0
```

### No collapse

```text
not all-WAIT
not all-RELEASE
completion > 0
throughput healthy
queue does not diverge
```

### Fixed-state discrimination

WAIT 与 RELEASE 均有稳定 recall，不允许只因 class imbalance 获得高 overall accuracy。

### Closed-loop performance

建议以 expert/rule-safe baseline 的相对性能定义工程 gate，例如：

```text
completion >= 0.90 × baseline
throughput >= 0.85–0.90 × baseline
```

具体阈值必须在 formal run 前冻结。

### Shield

shield burden 必须下降到“正常 policy 已承担主要 coordination”的范围，不能依赖 shield 才完成任务。

---

# 15. Pure PPO Baseline（必须保留为研究对照）

为了客观评估 Teacher/DAgger 是否真正有价值，应保留：

```text
B0 = fresh PPO + ExecutionScheduler + Safety
```

从随机或标准初始化直接训练，不使用 Teacher supervision。

其用途不是主线 fallback，而是论文 baseline。

比较：

- sample efficiency；
- collapse frequency；
- completion；
- throughput；
- safety/shield burden；
- final reward；
- seed variance。

主线：

```text
B3 = BC + DAgger + PPO
```

至少应与：

```text
B0 = Pure PPO
B1 = BC only
B2 = BC + DAgger
B3 = BC + DAgger + PPO
```

做消融。

---

# 16. P1：Student-Controlled Nominal PPO

## 16.1 启动条件

必须已有：

```text
I3 = PASS
PPO admission = PASS
```

## 16.2 执行链

```text
π_L
→ a_requested_joint
→ ExecutionScheduler
→ TrajectoryGenerator
→ SafetyFilter
→ PredictiveShield
→ environment
```

Teacher 默认：

```text
LABEL_ONLY diagnostics
BC auxiliary weight = 0
```

## 16.3 PPO 目标

主体优化：

- completion/progress；
- throughput；
- delay；
- energy；
- fairness/starvation；
- shield intervention penalty（如 reward 中已有）。

Teacher intervention penalty 从主线 reward 中移除，因为 Teacher不再接管执行。

## 16.4 Teacher 角色

Teacher 只用于：

- disagreement；
- fixed-state expert comparison；
- 判断 policy 是否明显偏离合理 operating region。

不把 Teacher agreement 当作最终优化目标。

## 16.5 P1 Gate

需要同时看到：

```text
student closed-loop healthy
PPO numerically healthy
reward/performance improves or stable
shield burden not worsening
no WAIT/RELEASE collapse
```

只有这样才允许增加环境难度。

---

# 17. P2：Traffic / N / Gate Expansion

保持 Teacher 不执行。

环境加入：

```text
ρ ~ U(0.4, 0.7)
N_init ~ 5...20
```

Gate mixture：

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

目的：

- variable density；
- variable-N；
- spatial imbalance；
- topology switching。

---

# 18. P3：Mission / SOC / Priority / Resource Expansion

在 traffic/N/gate 已稳定后加入：

## SOC

```text
U(0.5, 1.0)
```

## Mission

建议：

```text
standard      60%
cargo         20%
surveillance  20%
emergency      0%
```

## Priority

开始随机：

```text
normal / elevated / high
```

## LZ

```text
5–6 free
```

暂不同时加入最强 burst + extreme low-SOC + strongest gate imbalance。

---

# 19. P4：Full-Domain Nominal Geometry PPO

逐步打开：

- formal mission mix；
- full priority semantics；
- SOC 扩展至正式范围；
- LZ initial busy 0–3；
- random remaining service time；
- strong gate imbalance；
- bursty demand；
- variable-N；
- external queue build-up；
- long episode recovery。

仍保持：

```text
geometry = nominal
```

Overload 不应主导 gradient。

推荐场景比例初值：

```text
normal/medium    35%
high             35%
bursty           20%
near-capacity     5%
overload          5%
```

---

# 20. P5：Teacher OFF Nominal Consolidation

## 20.1 模式

```text
Teacher = OFF
High-level = OFF
Geometry = nominal
ExecutionScheduler = ON
TrajectoryGenerator = ON
SafetyFilter = ON
PredictiveShield = ON
```

## 20.2 目标

证明 policy 已不依赖 expert label：

- safety；
- completion；
- throughput；
- queue recovery；
- low shield burden；
- fairness；
- mission / SOC / resource robustness。

P5 结束后得到：

```text
low_nominal_qualified
```

但不是最终 high-level handoff checkpoint。

---

# 21. SOC / Energy Curriculum

Energy model 从第一阶段始终 ON，只改变 distribution：

```text
I0/I1/I2/I3/P1: 0.8–0.9 fixed/narrow
P2:             0.7–0.9
P3:             U(0.5,1.0)
P4/P5:          U(0.15,1.0) or formal distribution
```

不要在 imitation 早期同时引入 reserve-SOC 和复杂 priority。

---

# 22. Mission Curriculum

```text
I0–I3/P1:
standard 100%

P2:
standard dominant

P3:
standard      60%
cargo         20%
surveillance  20%
emergency      0%

P4/P5:
formal mission distribution
```

若正式分布另有版本，以 frozen experiment spec 为准。

---

# 23. Priority Curriculum

```text
I0–I3/P1:
fixed nominal priority

P2:
mostly nominal

P3:
normal / elevated / high randomized

P4/P5:
formal priority distribution
+ waiting/SOC urgency semantics
```

---

# 24. LZ / Resource Curriculum

```text
I0–I3/P1:
6/6 free

P2:
mostly free

P3:
5–6 free

P4/P5:
3–6 free
```

若 reset 时 LZ busy，必须构造合法 service lifecycle：

- corresponding UAV/service object；
- remaining service time；
- reservation；
- final/landing occupancy；
- LZ assignment consistency。

不能只设置 `busy=True`。

---

# 25. Variable-N

必须区分：

```text
N_requested
N_initial
N_active(t)
N_active_peak
```

finite cohort `N=60` 不等于 60 架 simultaneously active。

P4/P5 warm-start 可覆盖较宽 `N_init`，但主体仍应集中在 operational range。

明显高于训练 envelope 的 active-N 留给 OOD。

---

# 26. Episode Length Curriculum

| Stage | 建议 episode |
|---|---:|
| I0 | finite cohort / event collection |
| I1 | offline |
| I2 | 600–1200 s |
| I3 | 900–1200 s |
| P1 | 900–1800 s |
| P2 | 1200–1800 s |
| P3 | 1200–2400 s |
| P4/P5 | 1800–3600 s |
| G1/G2 | 1800–3600 s |
| G3 | transition + sufficient recovery |
| Qualification | 3600 s 或 formal protocol |

连续流评估分：

```text
warm-up
measurement
resolution/cooldown
```

---

# 27. Anchor / Fixed-State Cohorts

必须同时维护两类固定评估资产。

## 27.1 Closed-loop anchors

例如：

| Anchor | 描述 |
|---|---|
| A1 | single UAV lifecycle |
| A2 | 5 UAV simple transfer |
| A3 | 12 UAV symmetric |
| A4 | 20 UAV asymmetric medium |
| A5 | deterministic merge conflict |
| A6 | dense predecessor/successor |
| A7 | low-SOC priority |
| A8 | LZ busy |

## 27.2 Fixed decision-state cohort

必须显式覆盖：

```text
Teacher WAIT states
Teacher RELEASE states
multiple speed targets
multiple merge targets
resource/headway cases
student-failure recovery states
```

用途：

- 将“policy能力变化”与“policy访问状态分布变化”分离；
- 防止不同 checkpoint 的 closed-loop state distribution 使 agreement 指标不可直接比较。

---

# 28. 3D 可视化

每个 milestone 固定回放代表 episode。

重点检查：

- 长期绕圈；
- all-WAIT；
- aggressive RELEASE；
- interface oscillation；
- shield ping-pong；
- gate starvation；
- emergency starvation / abuse；
- low-SOC starvation；
- LZ resource congestion；
- post-transition release burst。

显示：

- requested joint action；
- decoded WAIT/RELEASE/speed/merge；
- applied action；
- Teacher expert label（若开启 diagnostics）；
- shield intervention；
- predicted min separation；
- leader/follower；
- predecessor/successor；
- reservation；
- phase/ring；
- SOC / mission / priority。

---

# 29. 标准指标

## 29.1 Safety

- collision；
- NMAC；
- hard separation violation；
- minimum actual separation；
- minimum predicted separation；
- CPA / TTC。

## 29.2 Imitation / DAgger

- joint expert agreement；
- WAIT recall；
- RELEASE recall；
- release macro-F1；
- per-class joint count；
- teacher expert WAIT/RELEASE distribution；
- student WAIT/RELEASE distribution；
- source-wise distribution；
- DAgger dataset growth；
- decision-event/raw-tick ratio；
- state-distribution gap。

## 29.3 PPO

- actor loss；
- value loss；
- approx KL；
- clip fraction；
- entropy；
- explained variance；
- grad norm。

## 29.4 Autonomy

- shield intervention / 100 requested decisions；
- repeated shield rescue；
- release acceptance；
- policy vs expert disagreement（diagnostic only）。

## 29.5 Progress

- completion；
- throughput；
- queue growth；
- release request rate；
- active N；
- peak active N。

## 29.6 Efficiency

- total delay；
- ring wait；
- final wait；
- energy；
- service time。

## 29.7 Fairness

- gate-specific delay；
- mission-specific delay；
- priority-specific delay；
- starvation；
- low-SOC delay。

---

# 30. Stop / Diagnose Conditions

出现以下情况，不得只增加 steps。

## 30.1 DAgger coverage failure

```text
training expert distribution
≠
student/fixed-natural distribution
```

尤其：

```text
training almost all WAIT
validation many RELEASE
```

优先检查：

- state source；
- decision-event weighting；
- warm-start / from-empty mixture；
- student failure states 是否进入 aggregate dataset。

## 30.2 Imitation class collapse

```text
policy → all WAIT
```

或：

```text
policy → all RELEASE
```

检查：

- fixed-state WAIT/RELEASE recall；
- dataset source；
- temporal oversampling；
- label distribution；
- observation sufficiency。

不要先改 reward。

## 30.3 PPO collapse

```text
release rate → 0
throughput → 0
```

或：

```text
release excessive
shield rescues most proposals
```

检查：

- PPO admission 是否过早；
- reward loophole；
- entropy / optimization；
- observation；
- shield exploitation。

## 30.4 Shield dependency plateau

policy closed-loop 指标正常但 shield 高且长期不降。

## 30.5 Curriculum regression

新阶段导致旧 fixed cohort / anchor 明显退化。

## 30.6 Geometry boundary collapse

只在 geometry boundary 高频 shield：

- geometry 是否超出 operational domain；
- observation 是否缺 geometry 信息；
- absolute demand 是否超过 `μ(g)`。

---

# 31. Reward Freeze 原则

## 31.1 Imitation 阶段

I0–I3 不依赖 PPO reward 作为主学习信号。

可记录 reward，但不得用其掩盖 imitation failure。

## 31.2 PPO 阶段

P1 正式开始前冻结 reward 主结构。

若必须修改：

1. bump `reward_version`；
2. 旧/new run 分开；
3. 重跑 anchors；
4. 重建 P1 baseline；
5. 记录原因。

Teacher intervention penalty 默认不属于新主线，因为 Teacher 不接管 PPO execution。

---

# 32. Checkpoint / Dataset 管理

## 32.1 Checkpoint

每阶段至少：

```text
stage_entry
best_validation
best_closed_loop
latest
stage_exit
```

PPO 阶段可增加：

```text
best_reward
best_low_shield
```

## 32.2 Metadata

绑定：

```text
architecture_version
low_action_schema
low_obs_version
reward_version
teacher_mode
dagger_round
dataset_version
scheduler_mode
shield_version
geometry_mode
traffic_profile
mission_profile
soc_profile
seed_partition
git_commit
working_tree_diff_hash (if needed)
config_hash
```

## 32.3 Dataset lineage

每个 DAgger dataset 必须记录：

```text
parent_dataset
student_checkpoint_source
state_sources
decision_event_filter
teacher_version
label_distribution
deduplication/persistence policy
```

---

# 33. Seed Partition

正式实验前冻结：

```text
train seeds
DAgger collection seeds
validation seeds
qualification seeds
OOD seeds
```

Qualification seeds 不得在训练中反复查看调参。

Fixed-state cohort 必须有独立来源记录。

---

# 34. G1：Near-Nominal Static Geometry

## 34.1 目标

验证 nominal student 对轻微 geometry shift 的适应。

每 episode 固定一个 geometry：

```text
r_k = r_k0 * (1 + ε_k)
ε_k ~ U(-0.05, 0.05)
```

±5% 只是工程初值，最终由 operational geometry envelope 决定。

## 34.2 模式

推荐：

```text
Teacher = LABEL_ONLY diagnostics or OFF
Shield = ON
High-level = OFF
Geometry = static per episode
```

不要恢复 Teacher execution correction。

Traffic 先用 medium / medium-high，降低 compound stress。

---

# 35. G2：Full Feasible Static Geometry

## 35.1 目标

覆盖正式允许的 low-level operational geometry envelope。

## 35.2 Sampling

推荐：

```text
20% nominal / near-nominal
20% boundary
50% random feasible interior
10% hard-but-feasible
```

可使用：

- stratified sampling；
- Latin Hypercube；
- Sobol；
- geometry classes。

## 35.3 Geometry classes

- nominal；
- compact；
- expanded；
- narrow O–M；
- narrow M–I；
- large outer；
- small inner；
- feasible boundary；
- random interior；
- hard feasible。

## 35.4 模式

```text
Teacher = OFF
Shield = ON
```

若出现 generalization gap，Teacher label-only 只用于诊断，不恢复 active correction。

---

# 36. Geometry Feasibility / Capacity Map

建议在 G1/G2 前或并行用 expert/rule-safe baseline 采样 geometry domain。

记录：

```text
μ(g)
delay(g)
energy(g)
shield burden(g)
throughput(g)
feasibility(g)
```

定义：

```text
G_operational ⊆ G_physically_feasible
```

同时报告：

### Fixed absolute demand

同一真实需求下不同 geometry 的运行表现。

### Fixed relative load

```text
ρ(g) = λ / μ_hat(g)
```

比较相似交通难度下的 robustness。

---

# 37. G3：Post-Reconfiguration Recovery Compatibility

## 37.1 边界

本阶段不训练 low-level 去“执行 geometry morphing”。

目标是：

> 外部合法 transition 完成、low-level 重新获得 action authority 后，能够在新的 traffic state distribution 上恢复稳定 coordination。

## 37.2 Transition interval

若 UAV 被 coordinator 接管：

```text
policy_action_valid = 0
actor_loss_mask = 0
```

Coordinator 动作不得成为 PPO behavior action。

## 37.3 Recovery

transition complete 后：

```text
policy_action_valid = 1
student-controlled execution resumes
```

重点收集：

- simultaneous release eligibility；
- topology reorder；
- queue accumulation；
- changed headway；
- SOC evolution；
- LZ/resource state；
- burst aftermath。

## 37.4 Metrics

```text
post-transition shield /100 decisions
release request / acceptance
queue clearance time
throughput recovery time
completion
gate delay
priority/low-SOC starvation
```

定义：

```text
T_recover
```

## 37.5 Gate

- safety；
- shield 能回落；
- queue / throughput 有限时间恢复；
- 无 starvation；
- unseen geometry pair 稳定。

---

# 38. Geometry Train / Validation / Test 分离

定义：

```text
G_train
G_validation
G_test
```

G_test 至少含：

- unseen interior；
- unseen boundary；
- interpolation；
- mild extrapolation；
- difficult witness geometry。

不得让训练看完全部 geometry 后再宣称 generalization。

---

# 39. Sensor Noise / Communication Delay

不要在主 I0–G2 早期加入。

先完成 deterministic-observation geometry qualification，再单独测试：

- position noise；
- velocity noise；
- SOC noise；
- observation delay；
- communication delay；
- dropped neighbor observation。

若显著退化，再决定：

- domain-randomization fine-tuning；
- temporal encoder；
- GRU/LSTM。

---

# 40. P5 / Nominal Qualification

P5 结束后形成：

```text
low_nominal_qualified
```

工程初始 gate 可包括：

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

阈值必须在正式 qualification 揭晓前冻结。

---

# 41. Final Qualification Matrix

## 41.1 Traffic quantity

Finite cohort：

```text
N_requested = 1, 5, 10, 20, 40, 60
```

Warm-start active：

```text
N_init = 5, 10, 20, 40, 50/60
```

## 41.2 Demand

```text
medium Poisson
high Poisson
bursty NHPP
near-capacity
overload
OOD arrival process
```

## 41.3 Gate

```text
uniform
mild imbalance
strong imbalance
dominant-gate switch
```

## 41.4 Resource / mission

```text
nominal mix
cargo-heavy
emergency spike
low-SOC stress
reduced LZ
long service-time
```

## 41.5 Geometry

```text
nominal
near-nominal unseen
unseen interior
boundary
hard feasible
post-transition recovery
unseen transition end-state
mild static extrapolation
```

---

# 42. 阶段总表

| Stage | Teacher role | Policy execution | Learning | Geometry | Core goal |
|---|---|---|---|---|---|
| C0 | expert controller baseline | expert | none | nominal | capacity |
| S0 | expert auditor | none/test | none | nominal | system qualification |
| I0 | expert labeler | expert/mixed data generation only | dataset | nominal | expert coverage |
| I1 | expert labeler | offline | joint BC | nominal | bootstrap |
| I2 | label-only | **student authoritative** | DAgger + BC | nominal | covariate-shift closure |
| I3 | label-only diagnostic | student | no PPO | nominal | imitation qualification |
| P1 | label-only diagnostic | student | PPO | nominal | nominal optimization |
| P2 | label-only/off | student | PPO | nominal | traffic/N/gate |
| P3 | label-only/off | student | PPO | nominal | mission/SOC/resource |
| P4 | off preferred | student | PPO | nominal | full-domain robustness |
| P5 | OFF | student | PPO/finetune | nominal | nominal consolidation |
| G1 | label-only/off | student | finetune if needed | near nominal | geometry adaptation |
| G2 | OFF | student | finetune if needed | full feasible static | geometry generalization |
| G3 | OFF | student after transition | PPO/finetune recovery | post-transition | recovery compatibility |
| Q_L | OFF | student | none | ID+OOD | final qualification |

---

# 43. 推荐实际执行顺序

```text
1.  C0 capacity calibration
2.  Freeze μ_hat and traffic definitions
3.  Freeze seed partitions
4.  Validate safe warm-start generator
5.  S0 V4 action / Observation / Safety / Teacher expert qualification

6.  I0 collect structured expert dataset
7.  I1 joint BC bootstrap
8.  Build fixed WAIT/RELEASE/joint decision-state cohort

9.  I2 Safe DAgger round 1
10. I2 Safe DAgger round 2+
11. Repeat until state-distribution / fixed-state gates pass

12. I3 student-only nominal imitation qualification
13. PPO admission decision

14. Run required Pure PPO baseline separately
15. P1 student-controlled nominal PPO
16. P2 traffic/N/gate expansion
17. P3 mission/SOC/priority/resource expansion
18. P4 dense/bursty/full-domain nominal geometry
19. P5 Teacher-OFF nominal consolidation
20. Nominal qualification

21. Geometry feasibility/capacity sampling
22. G1 near-nominal static geometry
23. G2 full feasible static geometry
24. G3 post-reconfiguration recovery compatibility

25. Final low-level qualification Q_L
26. Freeze final low-level checkpoint
27. Hand off frozen π_L to high-level/reconfiguration workflow
```

---

# 44. Algorithmic Baselines / Ablations

正式研究至少建议比较：

| ID | Method |
|---|---|
| B0 | Pure PPO + Execution + Safety |
| B1 | Expert BC only |
| B2 | Expert BC + Safe DAgger |
| B3 | Expert BC + Safe DAgger + PPO（主线） |
| B4 | B3 + small auxiliary BC during early PPO（可选） |

这样可以回答：

- Teacher 是否真的提高 sample efficiency？
- DAgger 是否真的解决 closed-loop distribution shift？
- PPO 是否真的超过 expert imitation？
- auxiliary BC 是否限制 PPO 超越 Teacher？

---

# 45. Success Trajectory

最终希望展示的不是 Teacher correction rate 下降，而是：

```text
expert bootstrap
    ↓
fixed-state imitation ability ↑

Safe DAgger
    ↓
student-state expert disagreement ↓
state-distribution mismatch ↓

Student-only qualification
    ↓
completion / throughput healthy

PPO
    ↓
delay / energy / throughput / fairness improve

Teacher OFF
    ↓
policy retains autonomy

Geometry curriculum
    ↓
operational geometry robustness

Final qualification
    ↓
freeze π_L
```

核心趋势：

```text
fixed-state WAIT/RELEASE discrimination ↑
DAgger student-state disagreement ↓
shield burden ↓
completion / throughput maintained or improved
queue stability ↑
```

始终：

```text
collision = 0
NMAC = 0
hard violation = 0
```

---

# 46. 必须避免的典型错误

1. 未经 C0 就定义 medium/high；
2. 只收 Teacher trajectory，忽略 student covariate shift；
3. raw WAIT ticks 按等权样本无限重复；
4. fixed-state discrimination 未通过就启动 PPO；
5. 主线使用 FULL Teacher correction + PPO；
6. 把 Teacher action 伪装成 PPO behavior action；
7. 把高 Teacher agreement 当最终性能目标；
8. all-WAIT 策略因 safety 好就判 PASS；
9. aggressive RELEASE 依赖 shield 仍判 PASS；
10. 一次打开多种 domain 难度；
11. warm-start 随机撒 UAV；
12. overload 主导 gradient；
13. ExecutionScheduler 搜 catalog 替 policy 优化；
14. shield 等 breach 后才介入；
15. reward 改版却沿用同一实验曲线；
16. train/validation/test seed 泄漏；
17. closed-loop validation 与 fixed-state validation 混为一谈；
18. Teacher label distribution 与 student state distribution 严重错位仍继续 PPO；
19. LZ busy 但 lifecycle 不完整；
20. nominal checkpoint 直接宣称 geometry generalization；
21. reconfiguration controller 动作当成 PPO actor action；
22. qualification 出结果后再改 gate threshold。

---

# 47. 推荐配置结构

```text
configs/
  low_v4/
    common.yaml

    c0_capacity.yaml
    s0_system_qualification.yaml

    i0_expert_dataset.yaml
    i1_joint_bc.yaml
    i2_dagger.yaml
    i3_student_imitation_qual.yaml

    baseline_pure_ppo.yaml

    p1_ppo_nominal.yaml
    p2_ppo_traffic_ngate.yaml
    p3_ppo_mission_soc_resource.yaml
    p4_ppo_full_domain.yaml
    p5_student_nominal.yaml

    g1_geometry_near.yaml
    g2_geometry_full.yaml
    g3_post_reconfiguration_recovery.yaml

    qualification_id.yaml
    qualification_ood.yaml
```

所有配置继承 `common.yaml`，避免无意修改 hard-safety / action semantics。

---

# 48. 实验命名规范

例如：

```text
A5V4_I0_expert_med_seed001
A5V4_I1_bc_datasetv003
A5V4_I2_dagger_r02_seed006
A5V4_I3_studentqual_seed010

A5V4_B0_pureppo_med_seed020
A5V4_P1_ppo_nominal_seed030
A5V4_P2_ppo_ngate_seed040
A5V4_P4_ppo_bursty_seed050

A5V4_G2_geo_static_seed060
A5V4_G3_postreconf_seed070
```

每个 run 至少记录：

```text
stage
run_id
git_commit
working_tree_diff_hash
config_hash
parent_checkpoint
dataset_version
dagger_round
teacher_mode
scheduler_mode
shield_version
geometry_mode
traffic_mode
seed
```

---

# 49. 低层训练结束时应形成的产物

1. nominal `μ_hat` capacity calibration；
2. traffic-level definitions；
3. Scenario Class library；
4. safe warm-start generator；
5. seed partitions；
6. V4 action / Safety / Teacher expert qualification；
7. structured expert dataset；
8. decision-event sampling specification；
9. joint BC bootstrap checkpoint；
10. fixed WAIT/RELEASE/joint decision-state cohort；
11. DAgger aggregated dataset lineage；
12. DAgger learning curves；
13. student-only imitation qualification；
14. Pure PPO baseline；
15. PPO stage curves；
16. shield dependency curves；
17. nominal-qualified checkpoint；
18. geometry feasibility/capacity map；
19. static geometry-generalized checkpoint；
20. post-reconfiguration recovery compatibility report；
21. OOD-N qualification；
22. fixed 3D anchor replay collection；
23. final low-level qualification report；
24. frozen final low-level checkpoint；
25. BC / DAgger / PPO ablation report。

---

# 50. 最终决策规则

训练出现问题时优先排查：

1. state distribution / dataset source 是否合理；
2. decision-event weighting 是否造成 WAIT temporal oversampling；
3. policy observation 是否充分；
4. Teacher expert semantics 是否 deterministic / state-conditioned；
5. Execution / Safety / Teacher 是否越权；
6. student 是否真的控制 closed-loop；
7. reward 是否有 loophole；
8. curriculum 是否一次改变太多因素；
9. PPO implementation / rollout semantics；
10. 最后才考虑更多 steps、网络规模或换算法。

低层研究的核心不是“模仿 Teacher 越像越好”，也不是“PPO reward 越高越好”，而是证明：

> 在复杂、可变、可泛化的 terminal traffic 中，共享 low-level policy 已经承担正常协调责任；Teacher 只提供训练期 expert supervision，Execution/Safety 只负责明确执行约束与最低限度安全保证。

---

# 51. 一句话主线

> **先标定容量与冻结系统语义；用结构化 expert data 做 joint BC bootstrap，再让 student 自己控制闭环、通过 Safe DAgger 让 Teacher 标注 student 真正访问的状态；只有 student-only nominal coordination 通过资格门槛后才启动 PPO，用 PPO 优化运营性能并逐步扩大 traffic / mission / SOC / resource 难度；随后 Teacher 完全退出，完成 static geometry generalization、post-reconfiguration recovery 与 ID/OOD qualification，最终冻结 low-level policy。**

---

# Appendix A. v3 → v4 Curriculum Migration

## 保留

```text
Capacity calibration
Structured traffic curriculum
Safe warm-start
Scenario classes
Observation topology philosophy
ExecutionScheduler boundary
Trajectory Generator
SafetyFilter
Predictive Safety Shield
Student-only qualification
Geometry generalization
Post-reconfiguration recovery boundary
ID/OOD qualification
```

## 被替代

```text
V3 factorized action
→ V4 joint Categorical(30)

FULL Teacher PPO
→ Expert BC + Safe DAgger

FULL → IMMEDIATE_ONLY → LABEL_ONLY → OFF
→ EXPERT DATA → LABEL-ONLY DAgger → LABEL-ONLY diagnostics → OFF

teacher intervention reduction as main learning target
→ student closed-loop capability + state-distribution closure
```

## 不再作为主线阶段

```text
L1 FULL Teacher PPO
L1b FULL Teacher variable
L2 IMMEDIATE_ONLY
L2b IMMEDIATE_ONLY dense
```

这些可保留为历史实验或 ablation，但不再定义正式训练主线。

---

# Appendix B. Mainline PPO Admission Checklist

启动 P1 前必须逐项回答：

```text
[ ] V4 action semantics qualified
[ ] Teacher expert deterministic and state-conditioned
[ ] fixed WAIT/RELEASE cohort built
[ ] joint BC does not collapse
[ ] at least one Safe DAgger cycle completed
[ ] student-visited states included in aggregate dataset
[ ] training vs validation expert distribution mismatch explained/controlled
[ ] student-only closed-loop completion healthy
[ ] throughput healthy
[ ] queue non-divergent
[ ] WAIT/RELEASE both discriminated
[ ] shield burden acceptable
[ ] Safety PASS
[ ] reproducible checkpoint/config/data lineage frozen
```

任何一项核心 gate 为 FAIL：

```text
PPO_ALLOWED = false
```

---

# Appendix C. Research Interpretation

本纲领将 Teacher 定义为：

> **safe, deterministic, explainable expert supervisor**

而不是：

> optimal ground-truth controller

因此最终论文应区分：

```text
expert imitation quality
student closed-loop autonomy
PPO operational improvement
hard-safety guarantee
```

Teacher 的存在是否必要，应由 B0–B4 baseline / ablation 实证回答，而不是由架构预设。
