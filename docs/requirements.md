# 项目需求文档

> **版本范围**：Phase A — Landing-only  
> **文档职责**：定义系统必须具备的能力与正式场景不可变语义。  
> 实现细节见 `technical_spec.md`，模块组织见 `design_spec.md`，开发顺序见 `execution_plan.md`。

---

## 1. 总体需求

系统必须能够模拟并训练一个 landing-only 低空终端区，使随机到达的多架 UAV 经由：

```text
external queue
→ inbound corridor
→ outer ring
→ middle ring
→ inner ring
→ final
→ discrete landing zone
→ service completion
```

完成任务。

环境必须显式建模：

1. 三层环形空域；
2. 4 个真实入场方向；
3. 动态 arrival demand 和 external queue；
4. 多架 UAV；
5. phase-dependent energy / SOC；
6. base priority 与 dynamic priority；
7. mission type；
8. M 个离散 LZ；
9. reservation 与 task-dependent service time；
10. ring/headway/capacity constraints；
11. canonical trajectory generation；
12. safety filtering；
13. 高层动态空域与低层共享意图策略；
14. 可审计的 outcome ledger 和实验日志。

---

## 2. 统一术语和编号

### 2.1 Ring naming

统一使用：

```text
OUTER
MIDDLE
INNER
```

数学符号：

\[
R_O>R_M>R_I,\qquad H_O>H_M>H_I
\]

代码建议：

```python
class RingRole(Enum):
    OUTER = "outer"
    MIDDLE = "middle"
    INNER = "inner"
```

可以存在内部整数 `index`，但任何接口不得依赖“index 越大越外/越内”的隐含约定。

### 2.2 Phase naming

统一生命周期：

```text
EXTERNAL_QUEUE
INBOUND
RING_OUTER
TRANSFER_O2M
RING_MIDDLE
TRANSFER_M2I
RING_INNER
FINAL
LANDING
SERVICE
COMPLETED
DIVERTED
FAILED
```

---

## 3. 空域结构需求

### 3.1 默认三环 nominal geometry

| Parameter | Outer | Middle | Inner |
|---|---:|---:|---:|
| Radius | 600 m | 400 m | 200 m |
| Height | 500 m | 350 m | 200 m |
| Nominal speed | 28 m/s | 24 m/s | 20 m/s |

这些数值是可配置的**工程仿真默认值**，不是对真实 vertiport 的法规性声明。

### 3.2 Geometry bounds

初始统一边界：

```text
radius_min = 120 m
radius_max = 900 m
height_min = 120 m
height_max = 650 m

min_ring_gap = 120 m
min_height_gap = 80 m

max_radius_change_per_high_level_update = 20 m
max_height_change_per_high_level_update = 20 m
```

目标几何必须满足：

\[
R_O-R_M\ge G_R,\quad R_M-R_I\ge G_R
\]

\[
H_O-H_M\ge G_H,\quad H_M-H_I\ge G_H
\]

### 3.3 Derived capacity

ring capacity 不允许作为独立 RL action。对半径 `R` 与安全弦长 `d_sep`：

\[
\Delta\theta_{min}=2\arcsin\left(\frac{d_{sep}}{2R}\right)
\]

\[
C(R)=\left\lfloor\frac{2\pi}{\Delta\theta_{min}}\right\rfloor
\]

实际实现需处理 `d_sep >= 2R` 等边界情况。

环缩小时还必须满足：

```text
target_capacity
>= current_ring_occupancy
 + committed_incoming_transfers
 + reservation_margin
```

若不满足，structural action 必须被 projection/clipping 拒绝或修正。

### 3.4 High-level timing

默认：

```text
metrics_sampling_interval_seconds = 20
high_level_decision_interval_seconds = 180
geometry_min_dwell_seconds = 600
geometry_transition_seconds = 20
```

高层每 180 s 产生一次 action；geometry event gate 决定 structural dimensions 是否执行。

---

## 4. 入口和 arrival demand

### 4.1 四入口

系统必须有 4 个独立入场方向：

```text
0°
90°
180°
270°
```

每个 gate 都有自己的 upstream corridor。

默认：

```text
approach_threshold_radius = 1000 m
spawn_radius = 1300 m
gate_height = 600 m
gate_count = 4
```

新 UAV 不允许都从同一个 bearing 生成后再分配 gate。

### 4.2 Arrival request 与 admission 分离

`ArrivalProcess` 产生的是**外部需求 request**，不能因为 terminal 当前满载而丢弃 request。

每个 request 先进入 `EXTERNAL_QUEUE`，只有满足 admission 条件后才生成/激活物理 inbound UAV。

### 4.3 Demand model

必须支持：

```text
deterministic_interval
poisson
bursty_wave
fixed_schedule
finite_cohort
```

`low/medium/high/bursty/overload` 的实际 rate 由 YAML 明确配置，不允许把 demand level 名称变成不可追踪的隐藏常数。

### 4.4 Admission constraints

至少检查：

- `max_active_uavs`；
- assigned gate corridor 是否可用；
- outer-ring admission capacity；
- SOC 是否高于进入终端区所需 reserve；
- safety spacing。

无法 admission 的 request 留在 external queue，并累计：

```text
external_wait_time
energy consumption
dynamic priority
```

### 4.5 Overflow / diversion

不得通过“超过 max_active_uavs 就停止生成 request”掩盖过载。

出现长时间拥堵时：

- request 保留在 external queue；
- 若预测剩余能量不足以安全完成 terminal operation，则标记 `DIVERTED_LOW_SOC`；
- 可设置 `max_external_wait` 作为 stress-test 规则，超限标记 `DIVERTED_CAPACITY`。

---

## 5. UAV 任务属性

每个 request 在产生时固定以下属性，之后不得因 policy 改变：

```text
arrival_id
request_time
entry_gate
mission_type
base_priority
energy_capacity
initial_SOC
random-seed-derived identity
```

### 5.1 Mission type

| ID | Name | Probability | LZ service time |
|---:|---|---:|---:|
| 1 | standard | 0.45 | 30 s |
| 2 | cargo | 0.25 | 120 s |
| 3 | emergency | 0.10 | 10 s |
| 4 | surveillance | 0.20 | 60 s |

mission type 的主要作用是 LZ service time、可选的 LZ compatibility，以及 emergency mission 对 base priority 的最低要求。

### 5.2 Base priority

| Priority | Probability |
|---:|---:|
| 1 | 0.50 |
| 3 | 0.35 |
| 5 | 0.15 |

若 `mission_type == emergency`：

```text
base_priority = max(sampled_base_priority, 5)
```

mission type 与 priority 不完全等价。

---

## 6. 电量与能耗

### 6.1 状态量

统一使用：

```text
energy_capacity
energy_remaining
SOC = energy_remaining / energy_capacity
energy_consumed
```

`battery_endurance_seconds` 不再作为主模型状态。

EH216-S 主仿真 profile 使用 `energy_capacity = 1500` s-equivalent，
`power_reference = 1.0` unit/s。它是将公开的参考最大飞行时间映射到归一化模型的
工程基准，不是未经验证的实际功耗曲线；phase multiplier 只表示相对功耗。

### 6.2 Phase-dependent energy

通用模型：

\[
E_{t+\Delta t}=E_t-P_{phase}(v,\dot z)\Delta t
\]

没有机型标定数据前，使用 nominal power model：

\[
P_{phase}=k_{phase}P_{ref}
\]

建议初始倍率：

| Phase | k_phase |
|---|---:|
| external airborne wait | 1.00 |
| inbound | 1.00 |
| ring hold | 1.00 |
| ring transfer | 1.10 |
| final approach | 0.90 |
| landing | 1.05 |

这些倍率必须可配置，并在论文中称为 normalized/nominal energy model，除非后续得到实际功耗标定。

### 6.3 SOC thresholds

当前 `research_evtol/eh216s_research_v1` 默认：

```text
SOC_warning = 0.35
SOC_reserve = 0.25
SOC_emergency = 0.20
min_landing_soc = 0.25
```

语义：

- `< warning`：进入 observation 与指标；
- `< reserve`：若尚未 admission，优先 diversion，不再正常放入 terminal；
- 接近 reserve 且已在 terminal：通过 SOC urgency 提高调度优先性；
- 已在 terminal 且成功 touchdown 前低于 `min_landing_soc`：作为低电量未完成退出；
- `< emergency`：仅作为历史或扩展配置的告警区间；当前主仿真在达到该值前已按
  `min_landing_soc` 退出，不能据此声称仍可正常申请 final clearance；
- touchdown 时 `SOC >= min_landing_soc` 才计为成功完成；
- 在达到成功 touchdown 前低于 profile 的 `min_landing_soc`：不计为完成，按
  `DIVERTED_LOW_SOC` 或相应能量失败语义记录；
- `SOC <= 0` 且未完成：`FAILED_ENERGY`。

历史实验中出现的 `energy_capacity = 10000` 或旧 SOC 门槛只属于历史协议，
不得回写为当前 EH216-S 规范，也不得改写其已归档结果。

---

## 7. Dynamic priority

动态优先级统一为：

\[
P_i(t)=P_i^{base}
\left[1+\alpha\left(\frac{W_i(t)}{T_W}\right)^\beta\right]
\left[1+\gamma U_i(t)^\delta\right]
\]

其中：

\[
U_i(t)=\max\left(0,\frac{SOC_{reserve}-SOC_i(t)}{SOC_{reserve}}\right)
\]

默认：

```text
alpha = 0.5
beta = 1.5
gamma = 1.0
delta = 2.0
T_W = 3600 s
```

Priority 用于 rule scheduler tie-breaking、LZ/merge conflict resolution、low-level observation、priority-weighted delay metric 与 urgency handling。高优先级永远不能绕过 hard separation。

---

## 8. Landing Zone 资源

### 8.1 数量与布局

默认：

```text
num_landing_zones = 6
```

固定地面坐标：

```text
LZ0 = (-80, -40, 0)
LZ1 = (  0, -40, 0)
LZ2 = ( 80, -40, 0)
LZ3 = (-80,  40, 0)
LZ4 = (  0,  40, 0)
LZ5 = ( 80,  40, 0)
```

LZ 是 ground infrastructure，**不能随着 inner-ring radius 改变而移动**。

### 8.2 LZ state

每个 LZ 至少包含：

```text
FREE
RESERVED
OCCUPIED
```

字段：

```text
lz_id
position
status
reserved_by
reservation_start_time
occupied_by / last_uav
service_remaining
compatible_missions
```

### 8.3 Reservation semantics

`Inner→Final` 前必须先完成 LZ reservation。

LZ 由 resource scheduler 分配，低层策略不直接输出 `lz_id`。历史协议保留
`lowest_id` 以支持精确回放；密集降落流使用 `nearest_available`，在所有 FREE 且任务兼容的
LZ 中按 UAV 当前三维位置的欧氏距离排序，距离相同时以 `lz_id` 稳定打破平局。无论采用哪种
排序，reservation 与最终航迹仍必须通过同一 SafetyFilter，距离近不构成安全豁免。

```text
FREE → RESERVED → OCCUPIED → FREE
```

以下情况必须释放 reservation：final clearance 永久取消、UAV diverted/failed、reservation timeout、trajectory planning 失败且不再 retry。

### 8.4 Service completion

touchdown 时：

```text
UAV leaves airborne traffic
LZ: RESERVED → OCCUPIED
service_remaining = service_time[mission_type]
```

`service_remaining == 0` 时：

```text
LZ → FREE
mission outcome → COMPLETED
```

论文中的“任务完成时间”默认取 service completion；同时单独记录 touchdown time。

---

## 9. Final resource 与 corridor

Final 不是无限容量区域。

历史默认（旧 P0/P1/P2 证据回放）：

```text
final_queue_capacity = 1
inner_to_final_min_headway = 12 s
```

P2R 密集降落流配置显式采用：

```text
final_queue_capacity = 3
lz_assignment_mode = nearest_available
release_arbitration_mode = downstream_priority
```

`final_queue_capacity` 是并发上限，不是并发目标；每条 final trajectory 仍逐条通过 headway、
reservation 和 SafetyFilter。环容量同样是安全上限而非利用率目标：当下游资源可用时，应及时
排空 INNER/MIDDLE，而不是为了“填满环”让 UAV 无效等待。

必须显式记录：

```text
final_reserved_by
last_final_release_time
final_trajectory_occupancy
```

Inner→Final clearance 至少需要：

1. LZ reservation 成功；
2. final resource 可用；
3. headway 满足；
4. candidate trajectory 通过 safety filter；
5. aircraft state 可执行；
6. geometry interface 未被冻结。

---

## 10. 低层动作需求

### 10.1 Action

每个 active UAV 使用 V4.2 hierarchical scheduling intention：

```text
gate: Categorical(2) = WAIT / RELEASE
proposal: Categorical(25), only when gate = RELEASE
```

`WAIT` 是唯一 canonical wait，不携带 proposal，执行解码为 `[0, 0, 0]`。
`RELEASE` proposal 是五档 speed residual 与五档 merge-time residual 的 25 个组合：

```text
speed_residual ∈ {-1.0, -0.5, 0.0, +0.5, +1.0}
merge_time_residual ∈ {-1.0, -0.5, 0.0, +0.5, +1.0}
```

V4.2 actor 的 factorization 是 `WAIT/RELEASE gate + conditional proposal`。K=1 作为消融时执行层
只接收一个解码后的三元意图；当前冻结的 K=3 模式允许同一次前向附带最多三个按 proposal logit
排序且不重复的三元意图，顺序均固定为
`[release_request, speed_residual, merge_time_residual]`。是否批准 RELEASE 仍由
ExecutionScheduler、trajectory 与 safety layer 决定。执行层只能按序认证 policy 提供的 bounded
shortlist 并执行首个通过项，不得自行扩展到 25 项 catalogue 或生成 delayed repair。

#### `speed_residual`

\[
normalized level is selected from the five-level catalog and is mapped by the
vehicle/profile adapter to \(\Delta v_i\in[-\Delta v_{max},+\Delta v_{max}]\).
\]

默认：

```text
max_speed_residual = 2.0 m/s
```

#### `merge_time_residual`

\[
normalized level is selected from the same five-level catalog and mapped by
the scheduler/profile adapter to \(\Delta t_i\in[-\Delta t_{max},+\Delta t_{max}]\).
\]

默认：

```text
max_merge_time_residual = 6.0 s
```

### 10.2 禁止项

最终主线低层 action 不得包含：

```text
free heading command
free flight-path-angle command
free altitude command
motor command
attitude command
independent slot_residual
```

merge slot/angle 由 scheduler 根据 merge time 和目标环 gap 派生。
完整 action、概率和 provenance 规范见 `reference/low_level_policy.md`。

---

## 11. 高层动作需求

### 11.1 Action shape

统一为 10 维：

```text
[
 dR_outer,
 dR_middle,
 dR_inner,
 dH_outer,
 dH_middle,
 dH_inner,
 dV_profile,
 dHeadway_O2M,
 dHeadway_M2I,
 dHeadway_I2F
]
```

所有输入先 clip 到 `[-1,1]`，再映射到 bounded increment。

### 11.2 Speed profile bias

基础：

```text
V0 = [28, 24, 20] m/s
```

高层 `dV_profile` 对三环 nominal speed 同方向施加有限偏置：

```text
max_speed_profile_change_per_update = 2 m/s
speed_min = 8 m/s
speed_max = 36 m/s
```

低层仍可在其上叠加 per-UAV speed residual。

### 11.3 Headways

默认：

```text
headway_O2M = 9 s
headway_M2I = 9 s
headway_I2F = 12 s

headway_min = 3 s
headway_max = 30 s
max_headway_change_per_update = 1 s
```

---

## 12. Geometry reconfiguration 需求

### 12.1 Moving reference

geometry update 必须经过连续 transition：

```text
actual_geometry(t)
→ interpolate
→ target_geometry
```

环上 UAV 跟踪 actual geometry，而不是 target geometry。

### 12.2 Interface freezing

若 reconfiguration 改变某一 ring pair：

```text
OUTER/MIDDLE affected → freeze new O2M transfers
MIDDLE/INNER affected → freeze new M2I transfers
INNER affected         → freeze new I2F if final-entry geometry is affected
```

但 ring hold、speed residual 和已 committed transfer 继续。

---

## 13. Merge scheduler 需求

每次阶段转移需要建立 reservation：

```text
source_uav
target_stage
candidate_merge_time
candidate_merge_angle
predecessor
successor
reservation_expiry
```

算法至少：

1. 根据 nominal schedule 产生 `t_base`；
2. 应用 `merge_time_residual`；
3. 预测目标环 UAV 到 `t_target` 的相位；
4. 寻找满足前后 separation 的 gap；
5. 如不可行，在允许窗口内向后搜索；
6. 建立 reservation；
7. 用 reservation 生成 canonical trajectory；
8. safety filter 检查；
9. approve / modify / reject。

不得继续用固定 `release_offset_deg` 作为正式 merge point。

---

## 14. Trajectory requirements

### 14.1 Canonical generator

系统只能有一个 authoritative trajectory generator。同一份 plan 用于：

```text
safety check
simulation execution
logging
visualization
future hardware adapter
```

### 14.2 Ring hold

支持 dynamic `R(t),H(t)`。

### 14.3 Ring transfer

要求：起终点 C0 continuous、水平切向至少 C1 continuous、垂直 profile 平滑、不超过速度/加速度/爬升下降/转弯约束，并到达 target merge reservation。

### 14.4 Final/landing

Final trajectory 至少包含：

```text
inner release state
→ final entry
→ stabilized approach
→ assigned LZ
```

不得在 touchdown 时把位置瞬移到 LZ。

---

## 15. Safety requirements

### 15.1 两级 safety

**Intent/resource-level** 拒绝：phase 非法、headway 不满足、capacity 不足、LZ 不可用、interface frozen、SOC/admission 不可行、reservation 冲突。

**Trajectory-level** 检查：pairwise predicted separation、NMAC/collision threshold、speed、acceleration、climb/descent rate、turn rate、world/airspace bounds、LZ/final occupancy。

### 15.2 Safety override

当前 EH216-S 仿真的安全距离契约为：`80 m` desired/planning operating separation、
`40 m` protected floor/NMAC 阈值，以及 `<10 m` collision event 研究代理阈值。
它们是工程仿真和评估语义，不是机型适航认证间隔。

SafetyFilter 对提交的 canonical trajectory 做 `80 m` 轨迹审查；运行时
PredictiveSafetyShield 对执行轨迹做与 executor 等价的 swept-segment 检查和递归
后备证明。若原候选不能满足 `80 m`，但 bounded SafetyShield intervention 对实际执行
轨迹仍可证明全程 `>=40 m`，则允许 shield 接管本步并继续整个 episode；其他已经
认证的 UAV 不因该次 buffer incursion 停止。`40--80 m` incursion 单独记录；
takeover/shield intervention 进入 reward safety cost（包括零数值修正的 takeover）。

安全层可以：

```text
delay release
replace merge time
reduce residual
reject trajectory
issue hold instruction
bounded shield takeover
```

若无法取得 `>=40 m` 的 swept + recursive 证明，预测/实际低于 `40 m`，或执行
轨迹未认证，则必须在物理执行前 fail-closed；不得通过 reward 扣分替代安全阻断，
也不得通过任意 heading/pitch 偏转把 UAV 临时“推开”后继续执行未验证路径。

---

## 16. Observation requirements

### 16.1 Per-UAV actor observation

当前 actor observation 固定为
`phaseA_low_obs_v2_1_retry_state`（88D）。前 87 维为 topology-aware V2，最后一维为
当前 `safety_retry_remaining_norm`。分组必须是：

**Self / flight（18D）**：phase、ring role、ring progress (`sinθ,cosθ`)、speed、SOC、wait、dynamic priority、release eligibility。

**Mission / resource identity（12D）**：mission type、assigned gate、assigned-LZ presence/usability、final resource availability、free-LZ fraction。

**Semantic topology（45D）**：same-ring leader/follower、target-ring predecessor/successor、nearest predicted conflict；每槽固定为 presence、path gap、Euclidean distance、relative speed、closing rate、`d_CPA`、`t_CPA`、priority delta、reserved/committed flag。

**System / flow（12D）**：ring occupancy ratios、三个 interface headway、external queue ratio、final queue pressure、global free-LZ fraction、interface frozen、interface headway slack、reconfiguration active/pending。

**Retry state（1D）**：`safety_retry_remaining_norm`。

Observation 不得包含未来动作、25-proposal 扫描结果、trajectory certificate 或 Teacher 私有信息。
V2.1 必须保持 V2 87D prefix 的顺序和语义，并通过 schema version 与 feature hash 防止漂移。

### 16.2 Neighbor selection

不得以“欧氏距离最近 k 架”作为唯一交通邻居定义。优先：

```text
same-ring leader/follower
target-ring predecessor/successor
conflict-nearest fallback
```

### 16.3 Variable number of UAVs

Actor：

```text
shared encoder: R^88
→ WAIT/RELEASE gate logits (2)
→ conditional RELEASE proposal logits (25)
```

对 N 架 active UAV batch 执行。

Critic：

```text
per-agent encoder
→ permutation-invariant pooling / attention
→ centralized value
```

Padding/mask 仅作为 batch implementation，不改变模型定义。

---

## 17. Rule baseline requirements

任何 RL 训练之前，必须有完整 rule controller 能够完成：

- 4 gate admission；
- outer entry；
- O2M；
- M2I；
- I2F；
- LZ reservation；
- final；
- touchdown；
- service complete；
- nominal non-overload scene 下无 collision、无 deadlock。

Rule baseline 排序原则：

```text
hard safety
> SOC emergency
> dynamic priority
> FCFS/request time
```

同一 control tick 存在跨环竞争时，P2R 执行仲裁先按 `INNER → MIDDLE → OUTER` 排空下游，
环内再按 dynamic priority、累计等待时间、UID 稳定排序。该顺序只决定候选被认证的先后，
不得降低 hard safety、headway、LZ reservation 或轨迹可执行性门槛。

---

## 18. 训练需求

### Stage 0 — Environment validation
无 RL。

### Stage 1 — Low-level learning
固定三环、高层禁用，训练 shared low-level policy。

### Stage 2 — High-level learning
低层冻结为稳定 rule policy 或已训练 low-level policy；训练 high-level。

### Stage 3 — Full HRL
组合已训练策略做正式评估。

### Stage 4 — Optional alternating fine-tuning
只在 Stage 3 已稳定后执行。

禁止两个随机网络从零同时训练作为主路线。

---

## 19. Episode 与正式评估

### 19.1 Formal finite cohort

必须满足：

```text
total_requested
=
completed
+ diverted
+ failed_collision
+ failed_energy
+ invalid_or_unresolved
```

正式有效 run：

```text
invalid_or_unresolved = 0
```

`40--80 m` buffer incursion、shield takeover 和 intervention 不属于
`invalid_or_unresolved`，但必须作为独立安全负担报告；任一 `<40 m` protected-floor
违规、collision event 或 uncertified execution 都使该 run fail-closed。

### 19.2 Continuous flow

记录：

```text
requested
admitted
completed
queue_remaining
diverted
failures
```

不能只报告 active UAV 中 landed 的比例。

---

## 20. 非功能需求

- 所有随机过程必须 seedable；
- 所有重要配置进入 YAML；
- core 不依赖 ROS2；
- pytest 覆盖核心状态机和资源守恒；
- checkpoint 必须携带 schema/version hash；
- observation/action schema 变化后旧 checkpoint 自动拒绝加载；
- 实验日志记录 config snapshot 和 git commit；
- 正式训练不得静默 fallback 到 rule-only；
- 文档、代码、YAML、tests 对 action shape 和 ring count 必须一致。

---

## 21. 当前范围明确排除

Phase A 当前不实现：departure traffic、shared arrival/departure corridor、network-level route assignment、weather avoidance、wind field、communications outage、arbitrary free-flight RL control、full rotor/aerodynamic dynamics。

风、噪声、通信延迟等只在核心 Phase A 稳定后作为 robustness 扩展。
