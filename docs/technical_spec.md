# 技术规范

> **目标版本**：Phase A — Landing-only canonical implementation  
> **用途**：用于 Codex 直接修改 `terminal_marl/`。V4.2/88D/mission/base 字段的接口兼容语义
> 继续保留；旧 checkpoint、旧两环 schema 和旧 4 维 intention 不能直接作为新训练入口。

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
  task_priority_enabled: false  # new main course; omitted legacy configs default to true

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
  profile: research_evtol/eh216s_research_v2
  safe_distance: 80.0
  nmac_distance: 40.0
  collision_distance: 10.0
  max_speed: 25.0
  max_acceleration: 1.5
  max_turn_rate_deg_s: 12.0
  max_climb_rate: 4.0
  max_descent_rate: 6.0

energy:
  model: legacy_phase  # phase_motion_v2 is an explicit V2 energy-course opt-in
  capacity_seconds_equivalent: 1500.0
  power_reference: 1.0
  phase_multipliers:
    external_wait: 1.00
    inbound: 1.00
    ring_hold: 1.00
    ring_transfer: 1.10
    final: 0.90
    landing: 1.05
  soc_warning: 0.35
  soc_reserve: 0.25
  soc_emergency: 0.20
  min_landing_soc: 0.10
  emergency_landing_enabled: false
  emergency_trigger_soc: 0.35
  emergency_energy_margin_seconds: 15.0
  phase_motion_nominal_solo_energy: 240.9
  emergency_landing_penalty: 8.0

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
  desired_separation_m: 80.0
  protected_floor_m: 40.0
  collision_event_distance_m: 10.0
  intervention_mode: bounded_swept_recursive
```

training/evaluation preset 必须引用相同 schema，不得复制出不同模型语义。新主课程使用
`task_priority_enabled=false`；历史配置省略该开关时按兼容默认 `true` 解释。

`80 m` 是 desired/planning operating separation；SafetyFilter 审查提交的
canonical trajectory，运行时 PredictiveSafetyShield 再按与 executor 等价的
swept + recursive 证明认证执行轨迹。若原候选不能满足 `80 m`，但 bounded
SafetyShield takeover 对实际执行轨迹仍可证明全程 `>=40 m`，则允许继续执行并让
其他已认证 UAV 继续运行。`40--80 m` buffer incursion 单独记录；takeover 和
intervention（包括零数值修正）进入 safety reward cost。低于 `40 m` 或无法取得
递归证明时，物理执行前 fail-closed，不能只靠 reward 惩罚。

`10 m` 是 collision event 的研究代理阈值，不是 EH216-S 适航认证距离。历史协议
中使用的 `20 m` collision threshold、`10000` normalized energy capacity 和旧的
“任一安全异常立即终止”qualification 语义仅用于解释归档结果，不得作为当前规范。

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

sampled_base_priority = rng.choice([1, 3, 5], p=[0.50, 0.35, 0.15])
if task_priority_enabled:
    base_priority = max(sampled_base_priority, 5) if mission == "emergency" else sampled_base_priority
else:
    base_priority = 1
```

`sampled_base_priority` 在中性课程只用于保持相同 RNG 消耗，不作为 request 字段保存；request
保留 raw `mission` 和有效 `base_priority`。不得在 admission 时重新抽样。中性课程的 policy
mission one-hot 固定为 `STANDARD`，而 mission service/compatibility metadata 与
`30/120/10/60 s` service time 仍保留。

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

`emergency_landing` 是始终保留的能量安全后备开关，与 mission-level emergency preference
和 `task_priority_enabled` 独立。启用时，环上低 SOC UAV 的应急计划必须从真实当前位置和速度连续切向
脱环：用 cubic Bézier 进入内部 staging point，OUTER/MIDDLE staging 高度为 `120 m`，INNER
保持当前高度；parameterizer 绑定真实 initial velocity，并将脱环段的 horizontal end speed
设为 `0`，再连接现有垂直下降 leg 到已预约 LZ。该计划仍需经过原 SafetyFilter，`80/40/10 m`
安全契约不因 SOC priority 改变。
V2 的 `min_landing_soc=.10` 是用户选定研究 floor；`soc_reserve=.25` 仍是
queue/admission compatibility threshold，不能互换。`phase_motion_v2` 的 uniform 默认采样仅在
未显式设置 `initial_soc_min/max` 时使用 `max(.35, .25, .10 + 240.9/1500) = .35`
下限；显式边界是场景协议覆盖值，不能解释为实测续航保证。

---

## 11. LandingZoneManager API

```python
class LandingZoneManager:
    def __init__(self, assignment_mode: str = "lowest_id") -> None: ...
    def available_zones(self, uav: UAVState, now: float) -> list[int]: ...
    def rank_available_zones(self, uav: UAVState, now: float) -> list[int]: ...
    def choose_rule_zone(self, uav: UAVState, now: float) -> int | None: ...
    def reserve(self, lz_id: int, uav_id: int, now: float) -> ReservationResult: ...
    def release(self, lz_id: int, uav_id: int, reason: str) -> None: ...
    def touchdown(self, lz_id: int, uav_id: int, mission: MissionType) -> None: ...
    def step(self, dt: float) -> list[int]: ...
```

`step()` 返回 service completed 的 UAV ids。

`assignment_mode` 只允许 `lowest_id` 与 `nearest_available`。后者以当前三维几何距离排序并以
`lz_id` 打破平局；reserved、occupied 或 mission-incompatible LZ 必须在排序前剔除。

---

## 12. Low-level action mapping

Execution adapter input shape remains `(3,)`; active V4.2 policy output is a
two-way WAIT/RELEASE gate and, only for RELEASE, a conditional 25-way proposal.
The decoder preserves 26 execution intentions（one WAIT + 25 RELEASE）and uses
the canonical order:

```text
[release_request, speed_residual, merge_time_residual]
```

mapper：

```python
release_requested = release_category == RELEASE
delta_v = raw[1] * max_speed_residual_mps
delta_t = raw[2] * max_merge_time_residual_seconds
```

任何 code/doc/test 都不得出现不同顺序。

最终进近由 `SimConfig.final_approach_execution_mode` 选择：`legacy` 是默认值并保留既有
geometry/execution；`speed_timing_v1` 为 opt-in，只能用于
`low_action_schema_v4_2_hierarchical` + `vehicle_execution_mode=performance_constrained`
以及 `final_approach_geometry_mode=continuous_inward_v1`。其速度意图是
`10 m/s + physical_speed_residual` 的目标速度，由 vehicle profile 与 parameterizer
施加速度、加速度和减速度边界；该目标不表示整段 final approach 恒速。

`SimConfig.runtime_proposal_shortlist_k` 只允许 `1` 或 `3`。`1` 保留历史单请求消融；冻结部署值 `3` 仅适用于
V4.2、`scheduler_mode=execution`、Teacher disabled。K=3 rollout 必须在一次 distribution forward
中形成 request-first unique Top-3，调用
`env.step(actions, proposal_shortlists)`；环境逐项认证并在第一个通过项停止。ActionTape 字段
`runtime_proposal_shortlists` 与 `runtime_shortlist_result` 是精确回放的一部分。

---

## 13. Per-agent action spec

```python
{
    "gate": Discrete(2),
    "release_proposal": Discrete(25),  # only valid for RELEASE
}
```

环境按 agent 返回 action spec：

```python
actions: dict[str, np.ndarray]
```

训练时可 stack：

```text
batch_obs    = (N_active, d_obs)
batch_gate = (N_active,)                 # WAIT / RELEASE
batch_release_proposal = (N_active,)     # null/sentinel for WAIT
batch_action_normalized = (N_active, 3)  # execution adapter values
```

WAIT has no proposal and decodes to zero residuals. RELEASE has five speed by
five merge-time variants. BC uses `CE(gate) + CE(proposal | Teacher RELEASE)`；
未来 PPO 的 requested log-probability 为 `log P(WAIT)`，或
`log P(RELEASE) + log P(proposal | RELEASE)`。execution、Teacher 与 shield
actions 不得进入 PPO ratio。

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

当前正式 schema 是 `phaseA_low_obs_v2_1_retry_state`（88D）。它保留 V2 的 87D prefix，
仅追加 `safety_retry_remaining_norm`；详细分组见 `reference/low_level_policy.md`。

实验性 90D schema 的准确值为 `phaseA_low_obs_v2_4_service_context_90d`：前 88D 与 V2.1
保持一致，追加 `self_service_duration_norm` 和
`center_earliest_compatible_lz_release_wait_norm`。`initialize_service_context_policy`
只对已通过 V2.1 schema/hash 与 V4.2 action metadata 校验的 loaded source policy 做
in-memory zero-column initialization，返回只含 `model`、`observation_dim=90`、`device`
的 inference wrapper；source weights 不变且不创建/迁移 optimizer。该 helper 不产生
qualified 90D checkpoint。

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

若 target ring 只有一个 holder，target gap 取该 holder 预测位置的对径点
（antipode）；target ring 无 holder 或存在多个 holder 时保持既有 gap 语义不变。
该 gap 选择仍必须经过完整 trajectory generation、energy/dynamics 与
SafetyFilter certification。

对 final target：

```text
merge geometry = final entry + assigned LZ reservation
```

在 opt-in `speed_timing_v1` final-approach contract 中，`MergeReservation.merge_time`
是 scheduler 的 requested reservation departure。候选 delay 与有限减速 prefix 的
earliest-feasible lower bound 必须在接受前检查；可行边界内可 clamp，越界或无法形成可执行
prefix 时 reject。该 contract 的 plan/action trace 可收集
`requested_reservation_departure`（scheduler 请求）、`accepted_reservation_departure`
（solver 接受的 slot）、`scheduled_departure`（canonical plan 的计划 departure）、
`timing_clamped`、`timing_clamp_reason`、`requested_target_speed_mps`、
`accepted_target_speed_mps`、`actual_entry_speed_mps`。`scheduled_departure` 与
`actual_entry_speed_mps` 是仿真/计划执行记录，不宣称硬件实测 departure；legacy 路径和普通 ring-transfer 不要求
这些新增字段。

`reconcile_late_merge_slots_enabled` 默认为 `false`。启用时，它只是一个 opt-in
operational scheduling variant：对已经生成的 ring-transfer，在 scheduler 完成
headway/reservation 检查后重新分配实际结束时隙；原始 requested merge time 与 lateness
必须保留在 plan metadata 中。重分配后仍须执行完整的 energy、dynamics 和
SafetyFilter certification。该开关不改变 trajectory geometry、proposal shortlist 或
策略接口，也不授权 PPO。该 exploratory variant 已在 fresh `.55–.90` late-slot
pairs 中评估，aggregate `zero_energy_failure` 未通过，未晋级；默认仍为 `false`。

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

    def emergency_landing(
        self, uav, lz, airspace, now
    ) -> TrajectoryPlan: ...

    def validate_kinematics(self, plan) -> list[str]: ...
```

删除/废弃任何第二套独立 transfer path generator。

`final_approach_execution_mode="legacy"` 与固定 `emergency_landing` route 是默认兼容
语义。`speed_timing_v1` 只改变 opt-in final approach 的目标速度/时序参数化；不得改变
emergency route、SafetyFilter certification 顺序或 execution takeover 语义。

### 17.1 Executable trajectory V3 contract

`trajectory_schema_version="executable_trajectory_v3"` 必须显式 opt-in，并使用
`vehicle_execution_mode="performance_constrained"`。V3 `TrajectoryPlan` 的 position、
velocity、acceleration 来自同一条 quintic curve；parameterizer 必须提供整段 analytic
bounds，覆盖 horizontal/vertical speed、acceleration 与 altitude。V3 commit 前检查
`start_time≈env.now`，并以 `1e-6` 比较起点 position/velocity/acceleration 与 live state；
ring handoff endpoint 以当前 ring 的 radius/height 几何容差和 `1e-6` 的切向 velocity、
centripetal acceleration 检查。失败必须 fail-closed 并记录 intervention，executor 不得
在 startup teleport 或依靠 endpoint rewrite 修正计划。

V3 当前仅支持静态 geometry；`tangent_yaw=True` 必须返回明确的 unsupported error。
V2 是默认和 historical replay 路径，旧 schema 不因 V3 opt-in 而改变。物理资格检查与
业务收益评估分开：V3 先检查解析导数/积分、动力学、执行边界和 SafetyFilter，再单独
报告 waiting、throughput、energy 与 outcome；历史 V2 的不安全 duration 不作为性能基线。

注册的 zero-training common-V3 comparison 固定使用 env seeds `877101/877102`、policy
seeds `878201/878202`、8 arms、每臂最多 24 requests 和 1800 ticks，输出根目录为
`/home/user/training_outputs/terminal/p3_foundation_repair_v3_20261002`，并保持
`PPO_ALLOWED=false`。该登记矩阵用于共同物理/schema 对照，不等同于 qualification 或
business-benefit PASS。

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

对 holder 的 bounded proxy 预测必须复用 V3 continuous ring limits，并把 curve-error margin
显式加入距离/可达性检查；proxy 不能替代真实 committed curve 或放宽 protected-floor。

`emergency_landing` 计划与普通计划使用相同的 `filter_plan` 和运行时 swept + recursive 证明；
安全拒绝继续保持/重试，不能通过降低 `80 m` desired separation 或 `40 m` protected floor
来覆盖，`<10 m` 仍只作为 collision event 研究代理阈值。

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
2. update external-queue energy / waiting urgency
3. admission decisions
4. update geometry transition actual state
5. update LZ service timers
6. legacy path: for eligible low-SOC ring holders, build/reserve/certify fixed emergency trajectory
7. build pre-action world snapshot
8. map per-agent raw actions → intents
9. apply action masks / intent safety
10. schedule resources and merge reservations
11. generate candidate trajectories
12. trajectory safety check
13. commit approved plans
14. execute one dt along committed plans / ring hold
15. consume energy
16. detect actual separation/collision
17. lifecycle transitions
18. touchdown → LZ occupied
19. service completions
20. reward computation
21. metrics / ledger / observations
```

当 `rule_flow_coordination_enabled=true` 时，步骤 6 的 legacy energy-emergency pass 被跳过；
normal release 先按 coordinated flow 顺序完成，之后才对仍为 uncommitted ring holder 的
UAV 运行 route-energy/slack emergency pass。该分支不以 `SOC <= emergency_trigger_soc`
单独接管；实际 emergency route、`.10` landing floor、15 s margin、route-slack window、
完整 SafetyFilter 和 runtime fail-closed 约束仍必须通过。legacy 路径的硬 SOC 资格门保持兼容。
该 energy-emergency pass 与 raw `mission_type=emergency` 及 `task_priority_enabled` 独立，
不得把物理服务类别当作主课程放行偏好。

顺序必须有 integration test，防止 reservation 与 movement 同步错误。

---

## 21. Rule policy

```python
class RuleLowLevelPolicy:
    def act(self, observations, snapshot) -> dict[str, np.ndarray]: ...
```

输出仍是 3D intention，与 RL 使用完全相同的 scheduler/trajectory/safety pipeline。规则策略不能绕过 pipeline 直接改 UAV state。

### Opt-in finite landing-flow coordination

`SimConfig.rule_flow_coordination_enabled` 默认关闭，只用于明确标记的
fixed-rule flow 实验；关闭时保留 legacy 默认动作域
`max_speed_residual_mps=2.0`。开启时，RELEASE 意图由固定规则产生有限的
canonical candidates，不冒充 learned policy，并派生持速动作域
`max_speed_residual_mps=0`；任何非零 speed residual 都拒绝执行。

当前候选集合为：

```text
O -> M: 48, 60, 72, 84 s
M -> I: 60, 72, 84, 96 s × tangential speed bias {0, -4, -8} m/s
I -> final: 20, 28, 36, 48, 60, 90 s
Energy-emergency final: 28, 36, 48, 60, 90 s
```

候选使用 scheduler 实际 arrival/merge slot，最终进近先按实际可用 LZ
reserve，再绑定生成的 executable route。能量 urgency 用于 release 排序和
deadline fallback 判定，
不能作为 energy、dynamics 或 safety certificate。每个候选必须通过原有
dynamics 检查。在 coordinated flow 分支，normal release 顺序固定为
`INNER -> MIDDLE -> OUTER`；同一环内按 remaining energy slack、dynamic priority、UID 排序
（slack 较小者优先，再按 dynamic priority 和 UID）。`task_priority_enabled=false` 时
dynamic priority 只包含等待与能量紧迫度；ring/path feasibility、resource reservation 和完整
SafetyFilter 仍是独立认证约束。全部 normal
release 尝试完成后才进行 emergency takeover。

M -> I admission 使用 inner queue capacity `6` 和额外的 inner-departure
headway 作为 normal preference。先执行廉价的 array preview headway 检查，
按 `12 × 16 / 200 = 0.96 rad` 的角间隔计算，并计入当前 holders 与
committed incoming。只有第一次 normal 候选因 `inner_departure_headway`
拒绝后，才允许第二次 normal M -> I 检查；第二次只检查第一次被拒绝的
那些 duration/bias 候选，且必须有 fresh、finite、non-negative 的
route-energy slack，且不超过一个 M 环周期乘当前 phase power。两次检查都
必须满足 capacity、scheduler reservation、dynamics validation、完整的
`80 m` SafetyFilter certification 和 `80/40/10 m` runtime monitoring；第二次成功提交时记录
`flow_deadline_fallback` telemetry。

holder 预测使用 nominal zero-residual holding domain；安全认证同时检查
matching point distance 与 `1 s` swept distance。coordinated emergency 路径
没有硬性的 `SOC <= 0.40` ceiling；它使用实际 emergency route 的 energy
检查，包含 `.10` landing floor 与 15 s emergency margin，并要求 route slack
不超过 `max(29 s, 当前环一周时间) × current ring-phase power`。若排除
geometry-infeasible 后所有 normal 候选均为 `emergency_energy_guard`，可使用显式
normal-energy-guard override 绕过这一条 slack urgency limit；实际 emergency route
energy、reserve、margin 与完整 SafetyFilter 仍始终必需。legacy 路径的 SOC 行为保持不变。
normal 与 emergency 路由都保留完整的 SafetyFilter、
dynamics validation、trajectory hash 检查和运行时 fail-closed 约束；flow
规则仍为 opt-in。

等价性能路径中，normal M -> I rejection 先使用 array preview，再决定是否
materialize trajectory points。权威 `validate_plan` 可使用 vectorized
`samples_are_feasible` pass predicate；predicate 失败或输入为 malformed/
non-standard 时回退到 scalar transition diagnostics，并保留原有约束、
sampling grids 和 error ordering。

UI 必须覆盖两个明确场景：低 SOC UAV 使用橙色 `LOW SOC`，进入 emergency
landing 后使用红色 `EMERGENCY`；`failed_energy` 终态保留真实位置的红色
`X` 和可分页终态事件记录（含时间、SOC、剩余能量、阶段和原因），不得从
可视化中无记录消失。

---

## 22. Shared low-level PPO

V4.2 当前网络（与 fresh BC actor 一致）：

```text
AgentEncoder(88 → 96 → 96)
GateHead(96 → 2)
ConditionalProposalHead(96 → 25)
```

现有 trained checkpoint 尚未在 dense30 或 V2 `eh216s_research_v2` vehicle/energy
profile 上完成 qualification。`fixed_rule` dense-flow 结果只属于规则协调和可视化诊断，
不得作为 learned checkpoint performance 报告。

Centralized critic：

```text
all agent embeddings
→ masked mean/max pooling or attention
→ concat(global state)
→ MLP
→ V
```

第一版可先 mean/max pooling，接口允许以后换 attention。必须用 mask 排除 inactive padding。

`algo/student_ppo_v4_2.py` 提供独立 PPO 内核；历史 `JointPPOV42` 仍为 BC/采样容器。
`training/student_rollout_v4_2.py` 提供 88D rollout。上述接口完成不代表正式准入，
正式入口仍须绑定通过 student-only qualification 的 checkpoint 和 admission 结果。

- 每个 simulator step 存一份 world、team reward（当步各 UAV reward 之和）、
  centralized value 与 next value；critic 每个 world 只计算一次损失。
- actor 对每个 world 内有效 UAV 求均值，再对 world 求均值。committed trajectory、
  非 holder 阶段和 future reservation 使用 canonical WAIT 并从 actor loss 排除。
- WAIT proposal 为 `-1`，条件 log-prob 不存在；RELEASE ratio 使用 gate 与 proposal
  log-prob 之和。Teacher、执行、安全层动作仅作 sidecar，不能替换 buffer 请求。
- 时间截断保留 next value，但截断处停止 GAE 跨 episode 传播；真正 terminal 不 bootstrap。
  空入场/服务区间继续推进 simulator，不将空观测误判为 terminal。
- PPO 仅接收随机采样 rollout；确定性评估轨迹禁止更新。DAgger 可启用显式
  `teacher_label_only`，正式 student/PPO rollout 关闭 Teacher。

### 22.1 当前 task-neutral 开发修复路径

task-neutral source/candidate 配对 baseline 已完成，证据见
[critic repair result](evidence/v4_2/p3_task_neutral_critic_repair_result_20260928.md)。当前开发修复
在进入本轮有界 PPO block 前做一次 critic-only warmup，使用既有完整终态真实 reward 构造
`gamma=.99` 的 terminal MC target；预热阶段仅训练 critic，所有非-critic 参数保持不变。
warmup 拟合数据不称独立 validation，也不凭 loss 通过。
本次 `fa0e` candidate 未在新物理场景验证中获保留；warmup 只是已评估的开发修复尝试，不是已证实解决方案，当前下一步以 [CURRENT_STATUS](CURRENT_STATUS.md) 为准，不将此流程设为所有未来 PPO 的强制规范。

若后续通过 [CURRENT_STATUS](CURRENT_STATUS.md) 的 admission 进入有界 PPO，沿用原 PPO/GAE
（`gamma=.99`、`gae_lambda=.95`），更新 `gate_adapter` 与 critic，`actor_encoder`/`gate_head`、
`proposal_head`、`gate_temperature` 和 proposal 配置保持冻结。该路径不增加普通 mission priority
保护；能量应急接管和全部安全约束保持。后续评估依据最终 normal、`failed_energy`、safety、wait
与新物理场景结果；formal、heldout 和全局 flags 仍由当前准入规定，不因本节自动放行。

### 22.2 研究复核后的训练信号与运行计量边界（2026-09-30）

qualification 先核对 `failed`、`unresolved` 与 safety hard counts，再做 `normal/emergency` 业务 screen，
最后读取 wait/throughput。raw low-level reward 是软权衡，不是词典序，也不是全系统成本；discount、batch 或
critic 的不足都不足以单独修复或解释 pre-existing degradation。team advantage 不是天然 bug。对 factorized entropy，
$H=H(\mathrm{gate})+p_{\mathrm{release}}H(\mathrm{proposal})$ 的驻点为
$p^*_{\mathrm{release}}=\sigma(H(\mathrm{proposal}))$；只有均匀 25-proposal 时
$H(\mathrm{proposal})=\log 25$，才有 $p^*=25/26$。当前 frozen proposal 通常非均匀，不能把 25/26 写成当前实际最优值。
global joint clipping 的 actor/critic 耦合是实现事实，对退化的贡献留待 paired experiment，不在此处改算法。

proposal、gate、feasibility 与长期收益必须用同一 pre-state 的 paired expected outcome 检查，不得用 25 个 proposal
中的 best-of-25 代替实际 K=3 stochastic execution。现有 terminal counterfactual 的两次 repeat 只有 `5` 个稳定
标签和 `7` 个 uncertain 状态；paired gate diagnostic helper 已实现并通过 9 个 focused checks。real-artifact
smoke 在注册的 2-tick horizon 内完成 26 个分支，所有 invariance 检查通过；由于 `censored=true`，该有限 smoke
不产生 `qualified_delta_Q`，full-cycle causal diagnosis 仍待完成。诊断的随机配对键为
`(continuation_seed, absolute_step, uid, channel)`；snapshot selection 另按 `(step, uid, arrival)` 记录，使用
weighted all-25 K=3，并在旧/新 checkpoint 的真实参数概率下复核。

本轮运行计量边界登记如下：wrapper 的 sustained-low gate 为 `low600 + dwell` 且继续执行 safety
checks；每 tick 累计 queue UAV-seconds、unresolved request-seconds，并在 arrival ledger 永久保留 energy；
`high_reward_v2_time_integrated` 是新 reward/schema，旧 high-level 结果不等价。SharedPPO V4.2
adapter 已实施并通过 early-fail-fast 回归；这只证明不兼容时提前失败，不等于完整等价支持。

运行计量中，`footprint_seconds` 按每 tick 累计归一化环面积和/成本代理
`sum(actual R²) / sum(initial R²)`；它不是实测物理占地，也不是三维体积。新输出 metadata 必须包含
`footprint_definition`，明确记录该定义及版本。

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
| 40--80 m buffer incursion | 单独统计；接管代价计入 shield reward |
| shield takeover/intervention, including zero-delta takeover | safety-cost term |
| NMAC event | -5.00 |
| collision failure | -50.00 |
| energy failure | -20.00 |
| normalized speed residual magnitude | -0.01 |
| normalized timing residual magnitude | -0.01 |
| task-priority shaping | 0 in new course; legacy-only when explicitly enabled |

重复持续事件必须去重/cooldown。

### 25.2 High-level reward

按 interval 归一化：

```text
+ completed missions
- full-request delay / waiting increment
- external queue area
- final/ring overpressure
- NMAC/collision/failure
- normalized energy consumed
- unnecessary reconfiguration magnitude
```

不要为“动作非零”或“半径增大”给正奖励。新主课程 `task_priority_enabled=false` 时
priority weight 必须为 `0`；SOC/energy urgency、energy use 和 safety terms 保留。历史
priority-weighted reward 只在兼容模式记录，不能与新课程 reward 直接比较。

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

密集降落流诊断还必须记录：

```text
max_concurrent_final
final_concurrency_step_sum
free_lz_step_sum
lz_idle_with_inner_demand_steps
lz_assignments
lz_assignment_distance_sum
lz_nearest_distance_sum
lz_assignment_regret_sum
```

`lz_idle_with_inner_demand_steps` 只表示“有内环需求且名义资源空闲”，不能直接解释为策略漏放行；
正式归因需要再报告 individually-certified counterfactual RELEASE opportunity。

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
test_landing_zone.py
test_landing_flow_efficiency.py

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
nearest-LZ assignment regret = 0 in nearest_available mode
concurrent final never exceeds configured capacity
every concurrent final trajectory remains safety-certified
```

### Gate D
固定三环 + V4.2 low-level student-only qualification 通过，N 可变化；新主课程还必须完成
full-cycle outcome ledger、全请求等待均值/p95/max 与 starvation tail 监测。这里不增加每个
mission/子组无退化门槛，也不把旧课程报表跨任务合并。PPO 只有在独立 admission checklist
通过后才允许启动。

### Gate E
三环动态 geometry 可运行且 reconfiguration 不破坏 committed traffic。

### Gate F
通过以上 gate 后才允许正式 high-level PPO。

---

## 31. Checkpoint compatibility

定义：

```text
action_schema_version = low_action_schema_v4_2_hierarchical
high_action_schema_version = high_airspace_v1_10d
observation_schema_version = phaseA_low_obs_v2_1_retry_state
airspace_schema_version = three_ring_v1
low_level_joint_action_count = 26
low_level_policy_distribution = hierarchical_gate_2_conditional_proposal_25
```

V4.2 keeps 26 execution decodes but records a two-way WAIT/RELEASE gate and
a conditional 25-way RELEASE proposal.  A future requested-action likelihood
is `log P(WAIT)` or `log P(RELEASE) + log P(proposal | RELEASE)`; V4.1 flat-26
checkpoints are historical and must be rejected rather than resumed.

checkpoint metadata 不匹配：

```text
raise IncompatibleCheckpointError
```

旧 checkpoint 只可历史诊断，不得 resume。

本轮新增实验性 Observation V2.4 已实现接口，为 V2.1/88D 加 `self_service_duration_norm` 与 `center_earliest_compatible_lz_release_wait_norm` 两字段，形成 90D，
并使用新的 schema/feature hash；旧 checkpoint 必须拒绝加载，代码回归已通过但尚未 qualification。正式 P3 在 V2.4 qualification 完成前继续使用
V4.2/88D，不得把 90D 实现写成当前 qualification 结果。

本次 88D source 到 90D 的 in-memory inference initialization 采用
`initialize_service_context_policy(loaded_source_policy,
source_checkpoint_metadata=payload_metadata)`。该 helper 必须严格验证 source 的 V2.1
schema、88D dimension、feature hash、V4.2 action schema 与 action distribution，新增两列
为 zero，source weights 不变且不创建/迁移 optimizer；返回 wrapper 不是 qualified 90D
policy。原生且 metadata/schema/dimension/feature hash 相符的 90D checkpoint 仍通过常规
strict loader 加载；`core.load_checkpoint` 不允许把 88D checkpoint 直接加载到 90D policy。
本接口规范不改变训练流程、safety/energy thresholds 或 reward 定义。

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

历史迁移清单见 `history/plans/phase_a_alignment_revision.md`；它仅用于追溯，不定义当前实现。

---

## 33. 仿真性能与可视化约束

安全审查的性能优化不得减少预测时域、采样密度或安全对象。允许的等价优化包括：

- 递归 successor projection 只复制会被投影修改的 `UAVState` 顶层对象；不可变
  `TrajectoryPlan` 可共享，但任何新增的嵌套可变字段在采用该路径前必须单独审计；
- 同一个 authoritative prediction step 内，完全相同的 plan、target phase 与时间网格可复用
  只读位置轨迹；`begin_prediction_step()` 必须清空该缓存；
- 只需要位置的碰撞计算使用 `TrajectoryPlan.sample_position()`；需要速度、加速度或姿态的执行
  路径仍使用完整 `sample()`。

批量 position interpolation 与 deferred waiting-prefix materialization 只用于减少重复
计算，保持原有 sampling grids、trajectory checks 和 certification 顺序不变；它们不得
改变轨迹几何、仿真步语义或安全对象集合。

viewer是只读observer，不得反向修改环境或重新定义航迹。实时航迹显示规则为：

```text
displayed path = authoritative current position + future committed-plan points
executed history = not displayed
normal UAV color = stable uid identity color
trajectory color = stable uid identity color
low-SOC/emergency UAV color = alert color
plan id or point topology changes = rebuild line actor
plan completion/disappearance = remove line actor
```

当前 PyVista/VTK viewer 以约 `30 Hz` 的展示节拍消费只读 snapshot，并缓存场景 actor。
UAV 使用一个持久的 point-cell cloud actor；严格拾取把 `(actor, point_index)` 映射到 UID，
未命中不回退到最近对象。只有 trajectory actor 在 plan ID 或点拓扑改变时重建。累计
`terminal_events` 在 source frame、进程缓冲和最终窗口中保留，并继续用于
`failed_energy` 的终态标记和面板记录。

能量与终态也属于只读快照契约的一部分。正常 UAV 与其未来轨迹继续使用稳定的
UID 身份色；`energy_alert="low_soc"` 的 UAV 使用橙色并显示 `LOW SOC`，
`emergency_landing=True` 优先使用红色并显示 `EMERGENCY`。快照中的
`terminal_events` 是累计的不可变事件序列，并可随进程快照 pickle；每条事件保留
`outcome`、`reason`、时间、SOC、剩余能量和终态阶段。若终态没有可用位置则
`position=None`，不会为外部队列或其他无位置事件伪造坐标；由 `_resolve` 记录的
失败事件使用 UAV 从 active 移除前的真实位置。

viewer 对累计事件去重后持续保留。`failed_energy` 事件若有真实位置，会在场景中
留下直到窗口关闭的红色 X 静态标记，并在可分页的终态事件面板中记录，不会把失败
UAV 当作仍在 active 集合中的飞机重新绘制。面板输出也会打印终态事件。3D 场景中的
降落点使用短标 `P0`–`P5`；固定的 `PAD STATUS` 面板逐行显示
`P<id>: FREE`、`P<id>: RESERVED UID=<uid>` 或 `P<id>: SERVICE UID=<uid>`，避免长
状态文字挤在降落点附近。

`BufferedProcessTerminalViewerObserver` 使用有界的四帧 source buffer。已经进入队列
的 source frame 按时间顺序完整消费；生产者在队列满时施加短暂 back-pressure，不用
替换旧帧，因此累计终态事件不会因漏帧隐藏。即使播放速度设为 `max`，仍逐个消费
source frame。只有相同 UID、相同 phase 和相同 plan ID、且相邻 source 时间
间隔不超过 `1.01 s` 时才允许位置线性插值；phase/plan 变化或更大时间间隔直接使用
目标 source 位置。

GUI 正常结束时会先提交并消费带终态事件的最终 source frame，然后保留最终画面直到
用户关闭窗口；headless 或 off-screen 运行不等待用户关闭。用户提前关闭窗口或 viewer
发生错误时按停止信号结束。

`ppo_checkpoint_viewer` 的密集场景默认 `frame_stride=2`、
`trajectory_max_points=12`。两者只控制渲染采样，不改变仿真步、策略输入或安全认证；命令行可用
`--frame-stride`、`--trajectory-max-points`、`--[no-]trajectories` 与
`--[no-]labels` 调整。


## 2026-10-03 P3 repair interface

新profile入口 `terminal_marl.configs.p3_integrated_repair.build_config`；新选项holding_speed_contract、emergency_priority_mode、route_aware_emergency_enabled、executable_admission_energy_enabled、low_level_queue_cost_enabled均保留历史默认。启用队列成本时env.step info追加team_reward/team_reward_components，student_rollout使用该团队总量而不添加虚构agent。StudentPPOV42.update新增entropy_mode=joint/gate_only；默认joint。新检查入口 `python -m terminal_marl.scripts.diagnose_p3_integrated_repair --check-only --output <new-dir>`，效果验证前必须先完成定向回归和smoke。

新profile显式选择 `external_queue_step_mode=post_admission`：在now安全提交入场后，只对剩余队列推进dt并计算队列成本；成功入场者推进dt飞行、获得首tick time/urgency reward。失败commit退回队列者只计算一次队列区间。旧默认pre_admission仅用于历史重放。

### 33.1 Local visualization console and offline replay

`terminal_marl.visualization.console_server` 提供本地标准库 HTTP 工作台。Windows 入口为
`.\venv\Scripts\python.exe -m terminal_marl.visualization.console_server`，跨平台入口为
`python -m terminal_marl.visualization.console_server`；默认只绑定 `127.0.0.1:8787`。启动器
`scripts/start_terminal_console.ps1` 只选择仓库 venv 或 PATH 中的 Python，不安装依赖、不修改
系统配置，也不隐藏用户主动启动的终端。

服务的静态文件白名单为 `index.html`、`console.css`、`console.js` 和 `scene.js`。仿真在独立
进程运行，父进程收集 plain `WorldRenderSnapshot` 并向 HTTP 客户端提供 session、frames、control
和 export 接口；实时缓存最多 6000 帧。初始帧默认暂停，暂停时单步只放行一个环境 tick；停止、重建、
子进程异常和退出都产生可见终态。HTTP 只接受 loopback Host/Origin 与有效 token，非法输入返回
普通 JSON 错误并保留当前运行。

P3 checkpoint adapter 沿用原 formal environment config；用户提交的 `max_steps` 和秒数只限制本次
执行的有效 tick 数，不改写 checkpoint 的环境配置。`checkpoint12`、P2 与 P3 profile 的场景、SHA
和 schema 约束继续由各自 profile 校验。

`write_playback(path, frames, title)` 保持原 API，但新文件从当前 web 工作台内联 CSS、Scene 和
console 脚本，并写入 `window.__TERMINAL_REPLAY__`。导出文件是 snapshot-only 单文件 HTML，可以
设计为在 `file://` 下运行，不依赖服务、Node、CDN 或外部资源；当前只完成离线文件生成、读回、无外部资源和 inline 脚本语法检查，未在本环境实测 `file://` 浏览器打开。空帧以及几何、时间和计数等关键字段的非有限数值必须拒绝，reset telemetry 的不可用非有限值转为 JSON `null`，旧
`snapshot_playback.html` 继续用于兼容历史记录。JSON/旧 HTML/新单文件 HTML 的导入验证 `time`、
`step_count`、三维点和有限值；`time` 使用绝对 simulation time，`arrival_id` 用于参考回放对齐。

工作台提供实时/历史时间轴、paused/current metrics、UAV 详情、终态事件、遥测、参考比较、图层和
相机控制。键盘 `Space` 播放/暂停，`←`/`→` 历史步进，`F` 跟随选中 UAV。缓存发生截断时必须在
状态和 export metadata 中保留 truncation 信息。

浏览器本地导入上限为 `50 MiB`、`100000` 帧；命令行 `--replay`/`load_replay` 上限为 `256 MiB`、
`6000` 帧，实时父进程的有界缓存也是 `6000` 帧。参考回放只按相同 absolute simulation time 的
sample 查找，并在该 sample 内按 `arrival_id` 匹配；采样时间不一致时保持未对齐，不自动重采样或外推。

可视化最低项目依赖为 `numpy`、`gymnasium`、`torch` 和 `PyYAML`；静态 HTTP 与回放读取不要求
torch。PyVista/VTK 只作为旧 live 3D diagnostic viewer 的可选依赖，见
`requirements-visualization.txt`。console 不提供 ROS/Crazyflie 实机控制，不训练模型，不改变
既有 checkpoint、action/observation 或 snapshot metadata schema，也不宣称实现 SUMO 的全部功能
或未经测量的浏览器性能。
