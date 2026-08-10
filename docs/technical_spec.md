# 技术规范

> **目标版本**：Phase A — Landing-only canonical implementation  
> **用途**：用于 Codex 直接修改 `terminal_marl/`。不要求兼容旧训练 checkpoint、旧两环 schema 或旧 4 维 intention schema。

---

## 1. 开发栈与依赖

```text
Python >= 3.10
numpy
scipy
PyTorch
Gymnasium / PettingZoo-style parallel API
PyYAML
pytest
matplotlib
```

core 仿真不得强依赖 ROS2、Crazyswarm2、crazyflie_py。硬件适配以后放在独立 adapter。

---

## 2. 配置分层

建议：

```python
SimConfig
AirspaceConfig
ArrivalConfig
VehicleConfig
EnergyConfig
PriorityConfig
LandingZoneConfig
SafetyConfig
LowLevelConfig
HighLevelConfig
TrainingConfig
EvaluationConfig
```

配置必须可序列化为 YAML snapshot，并带 `schema_version`。

---

## 3. Canonical YAML 示例

```yaml
schema_version: phaseA_landing_v1

simulation:
  dt: 1.0
  max_simulation_seconds: 3600
  max_active_uavs: 60
  reset_mode: flow_from_empty

airspace:
  rings:
    outer:  {radius: 600.0, height: 500.0, speed: 28.0}
    middle: {radius: 400.0, height: 350.0, speed: 24.0}
    inner:  {radius: 200.0, height: 200.0, speed: 20.0}
  radius_min: 120.0
  radius_max: 900.0
  height_min: 120.0
  height_max: 650.0
  min_ring_gap: 120.0
  min_height_gap: 80.0
  speed_min: 8.0
  speed_max: 36.0
  max_radius_change_per_update: 20.0
  max_height_change_per_update: 20.0
  max_speed_profile_change_per_update: 2.0
  headway_o2m: 9.0
  headway_m2i: 9.0
  headway_i2f: 12.0
  headway_min: 3.0
  headway_max: 30.0
  max_headway_change_per_update: 1.0
  metrics_sampling_interval_seconds: 20.0
  high_level_decision_interval_seconds: 180.0
  geometry_min_dwell_seconds: 600.0
  geometry_transition_seconds: 20.0
  geometry_gate_mode: event

arrival:
  gate_count: 4
  gate_angles_deg: [0.0, 90.0, 180.0, 270.0]
  approach_threshold_radius: 1000.0
  spawn_radius: 1300.0
  gate_height: 600.0
  model: poisson
  demand_level: medium
  arrival_rate_per_second: 0.03
  gate_min_time_gap_seconds: 20.0

vehicle:
  safe_distance: 80.0
  nmac_distance: 40.0
  collision_distance: 20.0
  max_speed: 36.0
  max_acceleration: 4.0
  max_turn_rate_deg_s: 12.0
  max_climb_rate: 5.0
  max_descent_rate: 7.0

energy:
  model: phase_nominal
  power_reference: 1.0
  phase_multipliers:
    external_wait: 1.00
    inbound: 1.00
    ring_hold: 1.00
    ring_transfer: 1.10
    final: 0.90
    landing: 1.05
  soc_warning: 0.30
  soc_reserve: 0.15
  soc_emergency: 0.08

priority:
  base_values: [1, 3, 5]
  base_probabilities: [0.50, 0.35, 0.15]
  alpha: 0.5
  beta: 1.5
  gamma: 1.0
  delta: 2.0
  wait_reference_seconds: 3600.0

landing_zones:
  count: 6
  positions:
    - [-80.0, -40.0, 0.0]
    - [  0.0, -40.0, 0.0]
    - [ 80.0, -40.0, 0.0]
    - [-80.0,  40.0, 0.0]
    - [  0.0,  40.0, 0.0]
    - [ 80.0,  40.0, 0.0]
  mission_service_seconds:
    standard: 30.0
    cargo: 120.0
    emergency: 10.0
    surveillance: 60.0
  mission_probabilities:
    standard: 0.45
    cargo: 0.25
    emergency: 0.10
    surveillance: 0.20

low_level:
  action_dim: 3
  max_speed_residual_mps: 2.0
  max_merge_time_residual_seconds: 6.0
  release_threshold: 0.0

high_level:
  action_dim: 10

safety:
  trajectory_horizon_seconds: 30.0
  reservation_timeout_seconds: 60.0
```

training/evaluation preset 必须引用相同 schema，不得复制出不同模型语义。

---

## 4. `DynamicAirspace` API

```python
class DynamicAirspace:
    ACTION_SIZE = 10

    def state_vector(self) -> np.ndarray: ...
    def normalized_state_vector(self) -> np.ndarray: ...

    def request_update(
        self,
        action: np.ndarray,
        *,
        now: float,
        occupancies: dict[RingRole, int],
        committed_incoming: dict[RingRole, int],
        geometry_gate: GeometryGateState,
    ) -> AirspaceUpdateResult: ...

    def step_transition(self, now: float) -> None: ...
    def actual_ring(self, role: RingRole) -> RingState: ...
    def target_ring(self, role: RingRole) -> RingState: ...
    def capacity(self, role: RingRole) -> int: ...
    def affected_interfaces(self) -> set[str]: ...
```

`state_vector()` canonical ordering：

```text
R_O,R_M,R_I,
H_O,H_M,H_I,
V_profile_bias,
headway_O2M,headway_M2I,headway_I2F
```

实际三环速度由 base profile + bias 派生。

---

## 5. Capacity implementation

复用独立函数：

```python
def geometric_ring_capacity(radius: float, required_separation: float) -> int: ...

def minimum_radius_for_occupancy(
    occupancy: int,
    required_separation: float,
) -> float: ...
```

必须测试：capacity 随 radius 单调；inverse consistency；transition 中使用 actual geometry；shrink 不违反 occupancy/committed incoming。

---

## 6. UAV lifecycle implementation

`lifecycle.py` 统一管理合法 transition：

```python
ALLOWED_TRANSITIONS = {
    EXTERNAL_QUEUE: {INBOUND, DIVERTED},
    INBOUND: {RING_OUTER, DIVERTED, FAILED},
    RING_OUTER: {TRANSFER_O2M, DIVERTED, FAILED},
    TRANSFER_O2M: {RING_MIDDLE, FAILED},
    RING_MIDDLE: {TRANSFER_M2I, DIVERTED, FAILED},
    TRANSFER_M2I: {RING_INNER, FAILED},
    RING_INNER: {FINAL, DIVERTED, FAILED},
    FINAL: {LANDING, FAILED},
    LANDING: {SERVICE, FAILED},
    SERVICE: {COMPLETED},
}
```

第一版不实现 go-around；如果未来加入，必须作为单独 revision，不留半实现 transition。

---

## 7. ArrivalProcess API

```python
class ArrivalProcess:
    def reset(self, seed: int) -> None: ...
    def requests_until(self, now: float) -> list[ArrivalRequest]: ...
    def exhausted(self) -> bool: ...
```

禁止 ArrivalProcess 根据 `max_active_uavs` censor demand。active capacity 属于 AdmissionManager。

---

## 8. AdmissionManager API

```python
class AdmissionManager:
    def enqueue(self, request: ArrivalRequest) -> None: ...

    def step(
        self,
        *,
        now: float,
        airspace: DynamicAirspace,
        gates: list[EntryGate],
        active_uavs: Mapping[int, UAVState],
        energy_model: EnergyModel,
        priority_engine: PriorityEngine,
    ) -> AdmissionDecisionBatch: ...
```

每个 request 返回：

```text
ADMIT
WAIT
DIVERT
```

并带 `reason_code`。

---

## 9. Mission / priority sampling

request 创建时执行：

```python
mission = rng.choice(
    ["standard", "cargo", "emergency", "surveillance"],
    p=[0.45, 0.25, 0.10, 0.20],
)

base_priority = rng.choice([1, 3, 5], p=[0.50, 0.35, 0.15])
if mission == "emergency":
    base_priority = max(base_priority, 5)
```

不得在 admission 时重新抽样。

---

## 10. Energy API

```python
@dataclass
class EnergyState:
    capacity: float
    remaining: float
    consumed: float

    @property
    def soc(self) -> float: ...

class EnergyModel:
    def phase_power(self, phase, speed, vertical_rate) -> float: ...
    def consume(self, state, phase, dt, ...) -> None: ...
    def estimate_to_complete(self, ...) -> float: ...
```

当前 canonical Phase A 规定 external queue 表示 UAV 已在上游空中等待，因此按 `external_wait` power 消耗能量。未来如需地面 dispatch queue，另设状态，不复用 external queue。

---

## 11. LandingZoneManager API

```python
class LandingZoneManager:
    def available_zones(self, uav: UAVState, now: float) -> list[int]: ...
    def choose_rule_zone(self, uav: UAVState, now: float) -> int | None: ...
    def reserve(self, lz_id: int, uav_id: int, now: float) -> ReservationResult: ...
    def release(self, lz_id: int, uav_id: int, reason: str) -> None: ...
    def touchdown(self, lz_id: int, uav_id: int, mission: MissionType) -> None: ...
    def step(self, dt: float) -> list[int]: ...
```

`step()` 返回 service completed 的 UAV ids。

---

## 12. Low-level action mapping

输入 shape：`(3,)`。canonical order：

```text
[release_request, speed_residual, merge_time_residual]
```

mapper：

```python
release_requested = raw[0] > release_threshold
delta_v = raw[1] * max_speed_residual_mps
delta_t = raw[2] * max_merge_time_residual_seconds
```

任何 code/doc/test 都不得出现不同顺序。

---

## 13. Per-agent action spec

```python
Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
```

环境按 agent 返回 action spec：

```python
actions: dict[str, np.ndarray]
```

训练时可 stack：

```text
batch_obs    = (N_active, d_obs)
batch_action = (N_active, 3)
```

不是一个 `Box(max_n,3)` 的全局策略动作。

---

## 14. Observation schema API

```python
@dataclass(frozen=True)
class FeatureSpec:
    name: str
    size: int

class ObservationSchema:
    features: tuple[FeatureSpec, ...]
    slices: dict[str, slice]
    dimension: int

class ObservationBuilder:
    schema: ObservationSchema
    def build_agent(self, uav, world) -> np.ndarray: ...
```

每个 feature normalized 到稳定范围；presence mask 与 zero padding 成对出现。最终 `d_obs` 由 schema 计算，变更时 bump `observation_schema_version`。

---

## 15. Topological neighbor utilities

实现：

```python
same_ring_leader(uav, ring_uavs)
same_ring_follower(uav, ring_uavs)
predict_target_ring_neighbors(uav, target_ring, target_time)
nearest_predicted_conflict(...)
```

环上距离使用 arc/chord 语义，不用 3D Euclidean nearest 代替交通拓扑。

---

## 16. MergeScheduler API

```python
class MergeScheduler:
    def baseline_time(self, uav, target_stage, snapshot) -> float: ...
    def propose(self, uav, intent, snapshot) -> MergeProposal: ...
    def reserve(self, proposal) -> MergeReservation: ...
    def cancel(self, reservation_id: str, reason: str) -> None: ...
```

对 ring target：

```text
merge_angle = predicted feasible gap center at merge_time
```

对 final target：

```text
merge geometry = final entry + assigned LZ reservation
```

---

## 17. TrajectoryGenerator API

```python
class TrajectoryGenerator:
    def ring_hold(
        self, uav, airspace, start_time, duration, dt
    ) -> TrajectoryPlan: ...

    def ring_transfer(
        self, uav, source, target, reservation, airspace, dt
    ) -> TrajectoryPlan: ...

    def final_approach(
        self, uav, lz, final_reservation, dt
    ) -> TrajectoryPlan: ...

    def validate_kinematics(self, plan) -> list[str]: ...
```

删除/废弃任何第二套独立 transfer path generator。

---

## 18. SafetyFilter API

```python
class SafetyFilter:
    def filter_intent(self, uav, intent, snapshot) -> IntentSafetyResult: ...

    def filter_plan(
        self,
        plan: TrajectoryPlan,
        committed_plans: Mapping[int, TrajectoryPlan],
        snapshot: WorldSnapshot,
    ) -> TrajectorySafetyResult: ...
```

`filter_plan` 至少在未来 horizon 内离散采样 pairwise distance。reject 后保持 holding 或继续当前 committed plan，不临时注入未计划的 heading offset。

---

## 19. Geometry reconfiguration 与 plan consistency

Geometry transition 中：

- ring hold 使用实时 actual ring reference；
- affected interface 的 new transfer action mask = false；
- committed transfer 继续原 plan；
- target planning 使用相应时间点预测 geometry；
- 若 high-level target 会导致 committed plan endpoint 不再可安全 capture，则 action 在 commit 前被拒绝或推迟。

因此 `DynamicAirspace.request_update()` 必须接收 committed reservations/plans 的约束摘要。

---

## 20. `TerminalParallelEnv.step()` 推荐顺序

```text
1. advance arrival requests
2. update external-queue energy / priority
3. admission decisions
4. update geometry transition actual state
5. update LZ service timers
6. build pre-action world snapshot
7. map per-agent raw actions → intents
8. apply action masks / intent safety
9. schedule resources and merge reservations
10. generate candidate trajectories
11. trajectory safety check
12. commit approved plans
13. execute one dt along committed plans / ring hold
14. consume energy
15. detect actual separation/collision
16. lifecycle transitions
17. touchdown → LZ occupied
18. service completions
19. reward computation
20. metrics / ledger / observations
```

顺序必须有 integration test，防止 reservation 与 movement 同步错误。

---

## 21. Rule policy

```python
class RuleLowLevelPolicy:
    def act(self, observations, snapshot) -> dict[str, np.ndarray]: ...
```

输出仍是 3D intention，与 RL 使用完全相同的 scheduler/trajectory/safety pipeline。规则策略不能绕过 pipeline 直接改 UAV state。

---

## 22. Shared low-level PPO

推荐起始网络：

```text
AgentEncoder(d_obs → 128 → 128)
ActorHead(128 → 64 → 3)
```

Centralized critic：

```text
all agent embeddings
→ masked mean/max pooling or attention
→ concat(global state)
→ MLP
→ V
```

第一版可先 mean/max pooling，接口允许以后换 attention。必须用 mask 排除 inactive padding。

---

## 23. High-level wrapper

每 `180 s` 形成一个高层 transition：

```text
s_H(t)
a_H(t)
run low-level for interval
aggregate interval metrics
r_H
s_H(t+1)
```

low-level policy 必须是显式对象：`rule` 或 trained checkpoint。禁止字符串 placeholder 静默映射为 zero actions。

训练日志记录：

```text
low_level_policy_type
low_level_checkpoint_hash
```

---

## 24. Event gate

事件 gate 只控制前 6 个 structural dimensions。

```text
structural mask: radius/height dimensions
action always enabled: speed-profile + 3 headways
```

gate 输入来自 rolling traffic pressure：occupancy ratios、external queue、inner/final pressure、LZ availability、recent arrival rate。

---

## 25. Reward API

```python
class LowLevelReward:
    def compute_agent(...) -> RewardBreakdown: ...

class HighLevelReward:
    def compute_interval(...) -> RewardBreakdown: ...
```

`RewardBreakdown` 至少有：

```text
total
components: dict[str, float]
```

训练日志保留 component mean/std/max。

### 25.1 Low-level 初始尺度

仅作为训练初始化，后续通过 diagnostics 调整：

| Component | Initial scale |
|---|---:|
| elapsed second | -0.01 |
| approved stage transfer | +0.25 |
| completed ring transfer | +0.50 |
| final entry | +1.00 |
| touchdown | +2.00 |
| service completion | +3.00 |
| safety rejection | -0.10 |
| NMAC event | -5.00 |
| collision failure | -50.00 |
| energy failure | -20.00 |
| normalized speed residual magnitude | -0.01 |
| normalized timing residual magnitude | -0.01 |

重复持续事件必须去重/cooldown。

### 25.2 High-level reward

按 interval 归一化：

```text
+ completed missions
- priority-weighted request delay increment
- external queue area
- final/ring overpressure
- NMAC/collision/failure
- normalized energy consumed
- unnecessary reconfiguration magnitude
```

不要为“动作非零”或“半径增大”给正奖励。

---

## 26. Metrics and outcome schema

每 run 生成：

```text
config.yaml
run_metadata.json
outcomes.csv
interval_metrics.csv
events.jsonl
training_metrics.*   # if training
```

`run_metadata.json`：

```text
schema_version
git_commit
seed
experiment_mode
low_level_checkpoint
high_level_checkpoint
observation_schema_version
action_schema_version
```

---

## 27. Training / experiment modes

正式统一：

```text
fixed_rule
fixed_low_rl
dynamic_rule
dynamic_hrl
```

变体：

```text
two_ring_dynamic_hrl
periodic_geometry_hrl
homogeneous_service
no_dynamic_priority
no_safety_filter
fixed_headway
```

配置 loader 遇到旧 mode 名应报错并指向 migration 文档，不静默兼容。

---

## 28. Formal evaluation protocols

### Finite cohort

建议起始：

```text
N = [20, 40, 60]
multi-seed >= 10
```

正式论文可根据算力扩大。

### Continuous flow

建议：

```text
episode = 3600 s
warmup = 600 s
measurement = 2400 s
resolution/cooldown = explicit
```

终止时仍未完成的 request 必须在 ledger 明确标记，不得丢弃。

---

## 29. Test requirements

至少新增/重写：

```text
test_three_ring_airspace.py
test_ring_capacity_three_ring.py
test_geometry_transition_three_ring.py
test_geometry_shrink_with_occupancy.py

test_arrival_request_not_censored.py
test_four_real_entry_corridors.py
test_external_queue_energy.py

test_mission_sampling.py
test_dynamic_priority.py
test_energy_model.py

test_lz_reservation.py
test_lz_service_time.py
test_lz_release_on_failure.py

test_lifecycle_state_machine.py
test_rule_pipeline_full_landing.py

test_low_level_action_schema.py
test_variable_agent_batching.py
test_topological_observation.py

test_merge_gap_selection.py
test_merge_reservation_conflict.py
test_transfer_bezier_tangency.py
test_same_plan_checked_and_executed.py

test_reconfiguration_interface_freeze.py
test_committed_transfer_survives_reconfiguration.py

test_outcome_accounting.py
test_checkpoint_schema_rejection.py
```

---

## 30. Mandatory integration gates

### Gate A
1 UAV 从 request 到 service completion。

### Gate B
多 UAV、固定三环、rule-only：

```text
zero collision
zero unresolved
all feasible requests resolve
LZ reservation correct
```

### Gate C
高/突发需求：

```text
external queue grows
no demand censoring
no resource double booking
no deadlock
```

### Gate D
固定三环 + low-level PPO 可训练，N 可变化。

### Gate E
三环动态 geometry 可运行且 reconfiguration 不破坏 committed traffic。

### Gate F
通过以上 gate 后才允许正式 high-level PPO。

---

## 31. Checkpoint compatibility

定义：

```text
action_schema_version = low_intent_v1_3d
high_action_schema_version = high_airspace_v1_10d
observation_schema_version = phaseA_obs_v1
airspace_schema_version = three_ring_v1
```

checkpoint metadata 不匹配：

```text
raise IncompatibleCheckpointError
```

旧 checkpoint 只可历史诊断，不得 resume。

---

## 32. 主路径必须删除/替换的逻辑

```text
exactly two rings assertion
8D high-level action
4D [release,speed,slot,timing] intention
slot_residual as second speed tweak
timing_residual as release-threshold bias
uniform endurance-seconds battery
single-bearing fake four-gate inflow
LZ = "FINAL 里是否有飞机"
fixed-angle ring transfer release point
two different transfer trajectory implementations
safety heading/pitch nudge as primary protection
```

迁移清单见 `revision.md`。
