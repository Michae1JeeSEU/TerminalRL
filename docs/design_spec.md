# 设计规范

> **当前架构**：Phase A — Landing-only  
> **核心原则**：三环终端区 + 连续到达流 + 地面服务资源 + 两层策略 + 确定性调度/轨迹/安全层。  
> 本文档描述当前目标系统结构，不承担历史兼容说明；旧迁移过程保存在 `history/`，当前完成度见 `CURRENT_STATUS.md`。
>
> **当前车辆基准**：主仿真使用 `research_evtol` / EHang EH216-S /
> `eh216s_research_v1`。安全距离和能量值是可审计的工程仿真语义，不是
> EH216-S 的适航认证间隔或功耗结论。

---

## 1. 总体架构

```text
┌───────────────────────────────────────────────────────────────────────┐
│ Experiment / Training                                                │
│                                                                       │
│  High-Level Policy πH                 Shared Low-Level Policy πL       │
│  global / 180 s                       per-UAV / 1 s                    │
│         │                                     │                       │
└─────────┼─────────────────────────────────────┼───────────────────────┘
          │                                     │
          ▼                                     ▼
┌───────────────────────────────────────────────────────────────────────┐
│ TerminalParallelEnv                                                  │
│                                                                       │
│  ArrivalProcess → AdmissionManager                                   │
│                     │                                                 │
│                     ▼                                                 │
│  DynamicAirspace ─ UAVLifecycle/TrafficState                         │
│          │          │                                                 │
│          │          ├─ PriorityEngine                                 │
│          │          ├─ EnergyModel                                    │
│          │          └─ LandingZoneManager                             │
│          │                                                            │
│          └──────────────┐                                             │
│                         ▼                                             │
│             Intent & Resource Scheduler                               │
│                         │                                             │
│                         ▼                                             │
│                  MergeScheduler                                       │
│                         │                                             │
│                         ▼                                             │
│               CanonicalTrajectoryGenerator                           │
│                         │                                             │
│                         ▼                                             │
│                    SafetyFilter                                       │
│                         │                                             │
│                         ▼                                             │
│                   TrajectoryExecutor                                  │
│                         │                                             │
│                         ▼                                             │
│          ConflictMonitor / Metrics / OutcomeLedger                    │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 2. 模块边界

建议目标目录：

```text
terminal_marl/
├── core/
│   ├── config.py
│   ├── enums.py
│   ├── dynamic_airspace.py
│   ├── ring_capacity.py
│   ├── uav.py
│   ├── lifecycle.py
│   ├── energy.py
│   ├── priority.py
│   ├── landing_zone.py
│   ├── arrival_process.py
│   ├── admission.py
│   ├── merge_scheduler.py
│   ├── trajectory_generator.py
│   ├── safety_filter.py
│   ├── observations.py
│   └── rewards.py
├── envs/
│   ├── parallel_env.py
│   └── high_level_wrapper.py
├── algo/
│   ├── shared_low_level_ppo.py
│   └── high_level_ppo.py
├── policies/
│   ├── rule_low_level.py
│   └── dynamic_rule.py
├── experiments/
├── configs/
└── tests/
```

允许迁移期复用已有文件名，但最终职责必须符合本规范。

---

## 3. Domain model

### 3.1 RingRole

```python
class RingRole(str, Enum):
    OUTER = "outer"
    MIDDLE = "middle"
    INNER = "inner"
```

### 3.2 FlightPhase

```python
class FlightPhase(str, Enum):
    EXTERNAL_QUEUE = "external_queue"
    INBOUND = "inbound"
    RING_OUTER = "ring_outer"
    TRANSFER_O2M = "transfer_o2m"
    RING_MIDDLE = "ring_middle"
    TRANSFER_M2I = "transfer_m2i"
    RING_INNER = "ring_inner"
    FINAL = "final"
    LANDING = "landing"
    SERVICE = "service"
    COMPLETED = "completed"
    DIVERTED = "diverted"
    FAILED = "failed"
```

Phase 是唯一生命周期真值。辅助字段可以存在，但必须由 phase 状态机约束，不能通过多个 boolean 组合推断真实阶段。

### 3.3 UAVState

目标结构至少包含：

```python
@dataclass
class UAVState:
    uid: int
    arrival_id: int

    # immutable request attributes
    request_time: float
    mission_type: MissionType
    base_priority: int
    assigned_gate: int
    energy_capacity: float

    # dynamic flight state
    phase: FlightPhase
    position: np.ndarray
    velocity: np.ndarray
    speed: float
    heading: float
    ring_role: RingRole | None
    ring_progress: float | None
    phase_entry_time: float

    # energy / urgency
    energy_remaining: float
    energy_consumed: float
    dynamic_priority: float

    # resources
    assigned_lz: int | None
    merge_reservation_id: str | None
    trajectory_plan_id: str | None

    # timestamps
    admission_time: float | None
    outer_entry_time: float | None
    final_entry_time: float | None
    touchdown_time: float | None
    completion_time: float | None

    # accounting
    external_wait_time: float
    internal_wait_time: float
    holding_time_by_ring: dict[RingRole, float]
    path_length: float
    terminal_outcome: TerminalOutcome | None
    terminal_reason: str | None
```

`SOC` 建议作为派生 property：`energy_remaining / energy_capacity`。

---

## 4. DynamicAirspace

### 4.1 Airspace state

```python
@dataclass
class RingState:
    role: RingRole
    radius: float
    height: float
    nominal_speed: float
    derived_capacity: int

@dataclass
class DynamicAirspaceState:
    outer: RingState
    middle: RingState
    inner: RingState
    headway_o2m: float
    headway_m2i: float
    headway_i2f: float
    transition_active: bool
```

必须同时维护 `actual_geometry` 与 `target_geometry`。

### 4.2 High-level action

`ACTION_SIZE = 10`：

```text
0 dR_outer
1 dR_middle
2 dR_inner
3 dH_outer
4 dH_middle
5 dH_inner
6 dV_profile
7 dHeadway_O2M
8 dHeadway_M2I
9 dHeadway_I2F
```

### 4.3 Projection order

`apply/request_high_level_action()` 必须按以下顺序处理：

1. finite check；
2. clip raw action 到 `[-1,1]`；
3. action scale；
4. per-update bounds；
5. absolute bounds；
6. ordered radius projection；
7. ordered height projection；
8. occupancy-aware minimum radius；
9. event-gate structural mask；
10. set target；
11. start/interpolate geometry transition；
12. recompute capacity from **actual** geometry。

### 4.4 Speed profile

基础 profile：

```text
outer  = 28 m/s
middle = 24 m/s
inner  = 20 m/s
```

高层只调整 shared bias：

```text
V_role = clip(V_role_base + speed_profile_bias, speed_min, speed_max)
```

避免把高层机械扩成 3 个独立 speed dimension。

---

## 5. ArrivalProcess 与 AdmissionManager

### 5.1 ArrivalRequest

```python
@dataclass(frozen=True)
class ArrivalRequest:
    arrival_id: int
    request_time: float
    gate_id: int
    mission_type: MissionType
    base_priority: int
    energy_capacity: float
    initial_soc: float
```

随机属性在 request 创建时一次性生成。不同 policy 在同 seed 下使用同一 request stream。

### 5.2 ExternalQueue

队列保存 `ArrivalRequest`，不是已经进入物理终端区的 UAV。

排序 baseline：

```text
SOC emergency
→ dynamic priority
→ request_time
```

必须记录 FCFS 被 priority 跳过的情况，供 fairness 分析。

### 5.3 EntryGate

```python
@dataclass
class EntryGate:
    gate_id: int
    angle: float
    spawn_point: np.ndarray
    gate_point: np.ndarray
    active_inbound: list[int]
    last_admission_time: float
```

四个 gate 使用不同 radial corridor。

---

## 6. LandingZoneManager

### 6.1 LandingZone

```python
class LZStatus(str, Enum):
    FREE = "free"
    RESERVED = "reserved"
    OCCUPIED = "occupied"

@dataclass
class LandingZone:
    lz_id: int
    position: np.ndarray
    status: LZStatus
    reserved_by: int | None
    reservation_start: float | None
    service_remaining: float
    compatible_missions: frozenset[MissionType]
```

### 6.2 Core operations

```python
reserve(lz_id, uav_id, now) -> ReservationResult
release_reservation(lz_id, uav_id, reason) -> None
touchdown(lz_id, uav_id, service_time) -> None
step(dt) -> list[ServiceCompletion]
available_for(uav) -> list[int]
earliest_release_time(lz_id) -> float
```

### 6.3 Assignment strategy

当前实现提供两个显式模式：

```text
lowest_id          # 历史协议默认；只用于兼容与精确回放
nearest_available  # P2R；FREE + mission-compatible 后按三维距离、lz_id 排序
```

低层 RL **不直接输出 `lz_id`**。LZ assignment 属于 resource scheduler，避免 action space 随 LZ 数量变化。
`nearest_available` 只是资源排序器；选中的 LZ 仍需原子 reservation，并由最终进近轨迹与
SafetyFilter 决定是否可执行。后续如加入预测服务时间或轨迹时长成本，应新增模式，不得静默改变
历史 `lowest_id` 或当前 `nearest_available` 的语义。

---

## 7. EnergyModel

```python
class EnergyModel:
    def power(self, uav, phase, trajectory_state=None) -> float: ...
    def consume(self, uav, dt, phase) -> None: ...
    def estimate_energy_to_complete(self, uav, env_state) -> float: ...
```

初期使用 phase nominal coefficients；以后可替换为 calibrated model，但接口保持不变。

当前 EH216-S 仿真的归一化基准为 `power_reference = 1.0` unit/s，
`energy_capacity = 1500` s-equivalent；phase multiplier 只表达工程化的相对功耗，
不能被解释为已完成的机型功耗标定。成功 touchdown 必须满足 vehicle profile 的
`min_landing_soc = 0.25`；低于该门槛的提前降落/终止不计为成功完成，应按
`DIVERTED_LOW_SOC` 或相应能量失败语义记录。

AdmissionManager 应通过：

```text
energy_remaining >= estimated_energy_to_complete + reserve_margin
```

决定低 SOC request 是否可以进入 terminal。

---

## 8. PriorityEngine

```python
class PriorityEngine:
    def compute(self, uav, now) -> float: ...
    def rank_key(self, uav, now) -> tuple: ...
```

`compute()` 必须与 `requirements.md` 公式一致。PriorityEngine 不拥有动作，也不绕过 safety constraints。

P2R 的同一步执行顺序为：先按环角色 `INNER → MIDDLE → OUTER`，再按更高 dynamic
priority、更长累计等待、更小 UID。该顺序在每步动作认证前刷新 priority；只改变竞争候选的处理
顺序，不改变策略请求、proposal shortlist、reservation 或 SafetyFilter 的所有权。

---

## 9. Low-level intention interface

### 9.1 Policy action 与执行意图

```python
@dataclass(frozen=True)
class HierarchicalActionV42:
    gate: int                 # WAIT=0, RELEASE=1
    proposal: int | None      # WAIT=None, RELEASE in [0, 25)
```

V4.2 policy 使用二分类 gate 与 RELEASE 条件下的 25-way proposal head。proposal 解码为
五档 speed residual × 五档 merge-time residual；唯一 WAIT 解码为 `[0, 0, 0]`。
ExecutionScheduler 的原子输入仍是一个规范化 `LowLevelIntent`：

```python
@dataclass(frozen=True)
class LowLevelIntent:
    release_request: float
    speed_residual: float
    merge_time_residual: float
```

实际物理单位由配置 profile mapper 给出，不得硬编码进 actor。V4/V4.1 的 actor 表示只用于历史诊断。

部署资格边界为 `runtime_proposal_shortlist_k=3`；K=1 仅保留为消融。K=3 由一次 policy forward 产生
`requested + two ranked alternatives`，逐一构造 canonical trajectory 并认证，首个通过项原子提交；
最多检查三个 proposal，不调用 Teacher、不扫描完整 catalogue、不生成 future-delay candidate。
fallback 的 applied action 与 rank 必须独立记录，不能重写 rank-1 requested action provenance。

### 9.2 Phase-specific interpretation

| Current phase | release target |
|---|---|
| INBOUND | OUTER |
| RING_OUTER | MIDDLE |
| RING_MIDDLE | INNER |
| RING_INNER | FINAL |
| FINAL/LANDING | 无新的 stage-release |

在 FINAL/LANDING 时，动作应根据 committed trajectory 做 mask，不产生新的自由飞行动作。

### 9.3 ActionMask

```python
ActionMask(
    can_request_release: bool,
    reason_codes: list[str],
)
```

WAIT 不产生 proposal；RELEASE gate 不可用时，环境必须 mask/reject 并记录原因。mask 不得包含
候选 trajectory certificate，也不得替 actor 在 25 个 proposal 中搜索。

---

## 10. MergeScheduler

### 10.1 MergeReservation

```python
@dataclass
class MergeReservation:
    reservation_id: str
    uav_id: int
    source_stage: str
    target_stage: str
    merge_time: float
    merge_angle: float | None
    predecessor_id: int | None
    successor_id: int | None
    status: str
    expires_at: float
```

### 10.2 Scheduling flow

```text
baseline target time
      ↓
apply Δt residual
      ↓
headway check
      ↓
predict target traffic
      ↓
find target gap
      ↓
resource reservation
      ↓
trajectory generation
      ↓
trajectory safety
      ↓
commit
```

Rule baseline 的 `t_base` 根据 earliest headway-safe time、queue order、priority adjustment 得到。RL 的 `Δt` 只能在有限窗口内修正 baseline，不能绕过 headway。

---

## 11. CanonicalTrajectoryGenerator

### 11.1 TrajectoryPlan

```python
@dataclass
class TrajectoryPoint:
    t: float
    position: np.ndarray
    velocity: np.ndarray | None
    yaw: float | None

@dataclass
class TrajectoryPlan:
    plan_id: str
    uav_id: int
    mode: str
    points: list[TrajectoryPoint]
    start_time: float
    end_time: float
    reservation_ids: list[str]
    metadata: dict
```

### 11.2 ring_hold

根据 current `actual_geometry(t)` 生成 moving-reference orbit。

### 11.3 ring_transfer

水平 cubic Bézier：

\[
B(s)=(1-s)^3P_0+3(1-s)^2sP_1+3(1-s)s^2P_2+s^3P_3
\]

其中：

```text
P0 = current position
P1 = P0 + L0 * source tangent
P3 = target merge point
P2 = P3 - L1 * target tangent
```

`L0/L1` 由几何距离和 curvature/turn-rate limit 限定。

垂直方向使用 quintic smoothstep：

\[
q(s)=10s^3-15s^4+6s^5
\]

\[
z(s)=z_0+[z_1-z_0]q(s)
\]

### 11.4 final_approach

推荐语义：

```text
inner release
→ final entry waypoint
→ stabilized approach waypoint
→ LZ touchdown
```

可以用 composite spline 实现，但日志中保留分段语义。

### 11.5 Execution

环境每步从当前 committed plan 采样下一状态。不得让 VTAC 再独立计算另一套 transfer curve。

---

## 12. SafetyFilter

### 12.1 API

```python
class SafetyFilter:
    def check_intent(...) -> IntentSafetyResult: ...
    def check_trajectory(...) -> TrajectorySafetyResult: ...
    def check_pairwise_plans(...) -> list[ConflictPrediction]: ...
```

### 12.2 Safety hierarchy

```text
protected floor / collision
> desired operating separation
> trajectory executability
> resource reservation
> SOC urgency / priority
> efficiency
```

当前安全距离契约为：

- `80 m` 是首选的 desired/planning operating separation。SafetyFilter 对提交的
  canonical trajectory 做轨迹审查；运行时 PredictiveSafetyShield 对实际执行轨迹
  做 executor-equivalent 的 swept-segment 检查与递归后备证明；
- `40 m` 是运行时不得突破的 protected floor，同时作为 NMAC 保护阈值；
- `<10 m` 才记为 collision event 的研究代理阈值，不代表 EH216-S 的机体碰撞认证距离。

`80 m`、`40 m` 和 `<10 m` 都是当前仿真的工程/评估语义。高 priority 只影响
“两个均满足安全契约的候选谁先走”，不能直接降低 `40 m` protected floor。

### 12.3 Safety responses

当候选无法证明满足 `80 m`，但 bounded intervention 的实际执行轨迹仍能通过
同一套 swept + recursive 检查并证明全程不低于 `40 m` 时，
`PredictiveSafetyShield` 可以接管本步的速度/时序意图；本机和其他已认证 UAV
继续运行，episode 不因一次 `40--80 m` buffer incursion 自动终止。buffer
incursion 单独记录；takeover/intervention 计入 shield reward cost，即使接管后
数值修正为零也收取一次干预代价。

允许：approve、delay release、replace merge time、reduce residual、hold、bounded takeover、reject。
若无法证明执行轨迹全程 `>=40 m`，预测/实际进入 `<40 m`，或缺少递归后备证明，
则在物理执行前 fail-closed；不能仅通过最终 reward 扣分来替代安全阻断。
不允许把任意 heading/pitch offset 当作主安全机制。

---

## 13. Dynamic geometry execution protocol

### 13.1 GeometryTransition

```python
@dataclass
class GeometryTransition:
    start_time: float
    end_time: float
    start_state: GeometryState
    target_state: GeometryState
    affected_interfaces: set[str]
```

### 13.2 Ring holder

每步跟踪 interpolated actual radius/height，低层 `speed_residual` 继续生效。

### 13.3 Already committed transfer

已产生 committed `TrajectoryPlan` 的 UAV 继续执行。DynamicAirspace 在 shrink feasibility 中把 committed arrival 纳入 occupancy。

### 13.4 New transfer

对应接口在 `affected_interfaces` 时：

```text
release_request → masked/rejected: GEOMETRY_RECONFIGURING
```

---

## 14. ObservationBuilder

### 14.1 Schema 化

```python
ObservationSchema(
    names=[...],
    slices={...},
    dimension=...,
)
```

禁止依赖匿名“第 37 维”约定。

当前 schema 固定为 `phaseA_low_obs_v2_1_retry_state`（88D）：V2 的 87D topology-aware
prefix 保持不变，索引 87 追加 `safety_retry_remaining_norm`。任何字段顺序变化都必须升级
schema version，并拒绝不兼容 checkpoint。

2026-09-20 已冻结待验证候选 `phaseA_low_obs_v2_2_scheduler_context_94d`：它保持前 88 维
不变，在索引 88–93 追加当前接口 active/future reservation、committed transfer、reservation
block remaining，以及同一步 contender/predecessor 数量。候选仍禁止 proposal catalogue scan、
trajectory certificate 和 Teacher 私有字段；在 fresh closure 与 learnability 通过前，已资格的
正式 schema 仍是 V2.1。

### 14.2 Topological context

优先提取：

```text
same_ring_leader
same_ring_follower
target_ring_predecessor
target_ring_successor
nearest_conflict_aircraft
```

不足时用 zero + presence mask。

### 14.3 LZ summary

当前 88D actor 使用紧凑、可在线获取的 LZ/resource summary：

```text
assigned_lz_present
assigned_lz_usable
final_resource_available
free_lz_fraction
```

单个 LZ 的 service remaining、earliest release 和完整 candidate 列表保留在 scheduler/resource
层，不进入当前 actor schema。未来若需要加入，必须升级 observation schema 并重新 qualification；
核心网络不得假设 M 永远等于 6。

---

## 15. Low-level network

Actor：

```text
per-UAV 88D feature vector
→ shared encoder
→ WAIT/RELEASE gate head
→ conditional 25-way RELEASE proposal head
→ one decoded 3-value execution intent
```

Centralized critic：

```text
all active UAV embeddings
→ attention / DeepSets-style pooling
→ global traffic embedding
→ concat airspace/resource state
→ value head
```

支持 N 动态变化。

---

## 16. High-level observation

建议包含：

- **Geometry**：actual/target R/H、speed-profile bias、headways、transition progress；
- **Traffic**：ring occupancy/capacity/ratio、queue age、mean/max holding、transfer queues；
- **Demand**：recent request/admission rate、burst indicator、external queue；
- **Resource**：free/reserved/occupied LZ、service remaining、final utilization；
- **Energy/Priority**：SOC quantiles、warning/reserve counts、priority quantiles、oldest wait；
- **Event gate**：pressure score、gate state、dwell remaining。

统计应基于 rolling window，而不只看单步瞬时值。

---

## 17. Reward design

### 17.1 High-level reward

\[
R_H = w_c\Delta N_{completed}-w_dD-w_qQ-w_sS-w_eE-w_fF-w_rC_{reconfig}
\]

其中 `D` 是 priority-weighted delay，`Q` 是 queue/occupancy pressure，`S` 是 safety cost，`E` 是 energy，`F` 是 diversion/failure，`C_reconfig` 是无效或过度 reconfiguration cost。

reward 不得直接奖励“半径更大”或“capacity 数字更大”。

### 17.2 Low-level reward

语义：

```text
+ stage progression
+ safe merge completion
+ touchdown/service-completion credit
- elapsed time
- energy use
- excessive residual magnitude
- safety rejection
- 40--80 m buffer incursion
- shield takeover/intervention (including zero-delta takeover)
- separation risk
- starvation/fairness cost
- failure
```

权重全部由 YAML 管理并通过 reward diagnostics 校准。

---

## 18. RuleLowLevelPolicy

Rule policy 必须与 RL 使用完全相同的 3D intention interface 和后端 pipeline。

### 18.1 Inbound→Outer
检查 gate corridor、outer capacity、headway、merge gap。

### 18.2 Outer→Middle / Middle→Inner
检查最低驻留、priority、headway、target gap、interface availability。

### 18.3 Inner→Final
额外检查 LZ assignment/reservation、final resource、final headway。

### 18.4 Speed guidance
采用 leader/follower gap-based bounded residual，不直接改任意 heading。

---

## 19. Landing service state

touchdown 后：

```text
phase = SERVICE
lz = OCCUPIED
service_remaining -= dt
```

服务结束：

```text
phase = COMPLETED
outcome = COMPLETED
completion_time = now
LZ = FREE
```

---

## 20. MetricsCollector / OutcomeLedger

### 20.1 OutcomeLedger

每个 `arrival_id` 唯一一行：

```text
request_time
admission_time
outer_entry_time
final_entry_time
touchdown_time
completion_time
outcome
reason
mission_type
base_priority
initial_soc
landing_soc
assigned_lz
external_wait
internal_time
service_time
```

### 20.2 Event logs

至少记录：high-level requested/applied action、geometry gate/transition、low-level action/mask、release approve/reject、merge reservation、LZ reservation、trajectory approve/reject、safety event、touchdown、service complete、diversion/failure。

---

## 21. Formal reset modes

### `flow_from_empty` — 论文主模式

`t=0` terminal 空，arrival requests 随时间进入。

### `warm_start_rings` — 调试/curriculum

可在三个环预置 UAV，默认：

```text
outer  = 0.60
middle = 0.25
inner  = 0.15
```

仅用于 transition unit tests、reward sanity、low-level curriculum、stress diagnostics。正式主结果不得只使用 warm-start。

---

## 22. Experiment modes

正式模式统一为：

```text
fixed_rule
fixed_low_rl
dynamic_rule
dynamic_hrl
```

主要变体：

```text
two_ring_dynamic_hrl
periodic_geometry_hrl
homogeneous_service
no_dynamic_priority
no_safety_filter
fixed_headway
```

旧的同义 mode 名称不得继续作为正式实验接口并存。

---

## 23. Phase B/C 接口边界

当前代码可预留 `OperationDirection.ARRIVAL`，但不实现 departure state/action。Phase B 后再增加 `DEPARTURE`，不得为了未来 departure 让当前 observation/action 膨胀。
