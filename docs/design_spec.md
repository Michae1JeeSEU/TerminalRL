# 设计规范

> **当前架构**：Phase A — Landing-only  
> **核心原则**：三环终端区 + 连续到达流 + 地面服务资源 + 两层策略 + 确定性调度/轨迹/安全层。  
> 本文档描述目标系统结构，不承担历史兼容说明；迁移工作见 `revision.md`。

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

Rule baseline：

```text
feasible LZ
→ earliest service availability
→ shortest final path
→ lz_id tie-break
```

低层 RL **不直接输出 `lz_id`**。LZ assignment 属于 resource scheduler，避免 action space 随 LZ 数量变化。

---

## 7. EnergyModel

```python
class EnergyModel:
    def power(self, uav, phase, trajectory_state=None) -> float: ...
    def consume(self, uav, dt, phase) -> None: ...
    def estimate_energy_to_complete(self, uav, env_state) -> float: ...
```

初期使用 phase nominal coefficients；以后可替换为 calibrated model，但接口保持不变。

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

---

## 9. Low-level intention interface

### 9.1 Action dataclass

```python
@dataclass(frozen=True)
class LowLevelIntent:
    release_request: float
    speed_residual: float
    merge_time_residual: float
```

三个 raw 分量均在 `[-1,1]`，实际物理量由 mapper 转换。

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
    speed_residual_enabled: bool,
    timing_residual_enabled: bool,
    reason_codes: list[str],
)
```

policy 输出不可用维度时，环境必须 mask 并记录统计。

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
hard collision/separation
> trajectory executability
> resource reservation
> SOC urgency / priority
> efficiency
```

高 priority 只影响“两个均安全候选谁先走”，不能降低 hard safety threshold。

### 12.3 Safety responses

允许：approve、delay release、replace merge time、reduce residual、hold、reject。  
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

actor 至少使用：

```text
assigned_lz_state
assigned_lz_service_remaining
free_lz_fraction
earliest_lz_release
top-K candidate LZ availability
```

默认 M=6 时可附加 6 个 LZ occupancy feature，但核心网络不得假设 M 永远等于 6。

---

## 15. Low-level network

Actor：

```text
per-UAV feature vector
→ shared encoder
→ actor head
→ 3D intent
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
