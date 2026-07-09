# 技术规范（Q1 创新主线重构版）

> 本文档定义下一阶段实现规范。目标是在不破坏当前 `terminal_marl/` 可运行性的基础上，逐步加入动态空域、高低层策略、轨迹生成器、安全过滤、实验模式和 Crazyflie 缩尺验证接口。

## 1. 开发环境

| 项目 | 选型 | 说明 |
|---|---|---|
| 语言 | Python 3.10+ | 兼容当前项目和 ROS2 常见环境 |
| 强化学习 | PyTorch + 自定义 PPO / SB3 可选 | 当前 `terminal_marl` 已有 shared PPO |
| 多智能体接口 | PettingZoo-style parallel env | 保留 `TerminalParallelEnv` 风格 |
| 配置 | dataclass + YAML preset | 逐步增加 `simulation_large` 和 `crazyflie_lab` |
| 测试 | pytest | 核心模块必须配套测试 |
| 实物平台 | ROS2 + Crazyswarm2 | 仅在 hardware adapter 中懒加载 |

## 2. 依赖规范

基础依赖：

```text
numpy
scipy
matplotlib
pytest
pyyaml
```

强化学习依赖：

```text
torch
stable-baselines3    # optional
gymnasium            # optional
```

硬件依赖不得进入 core 强依赖：

```text
ros2
crazyswarm2
crazyflie_py
```

要求：没有 ROS2/Crazyswarm2 时，仿真、测试和 dry-run 仍能正常运行。

## 3. 代码规范

- 新增公开类和函数必须有类型注解；
- 新模块顶部必须有 docstring；
- `terminal_marl/core/` 不得 import ROS2、Crazyswarm2 或 crazyflie_py；
- `terminal_marl/hardware/` 负责硬件适配；
- safety 相关逻辑必须返回可解释的 violation info；
- 每个核心模块必须有 pytest 测试；
- 新增 action mode 不应破坏旧脚本。

## 4. 配置规范

### 4.1 配置分层

建议拆分：

```text
SimConfig：通用仿真和环境参数
AirspaceConfig：ring/headway/capacity/bounds 参数
TrainingConfig：PPO/HRL 训练参数
HardwareConfig：Crazyflie/Lighthouse/Crazyswarm2 参数
```

### 4.2 simulation_large preset

用于论文主仿真：

```yaml
name: simulation_large
center: [0.0, 0.0, 0.0]
dt: 1.0
outer_ring_radius: 400.0
inner_ring_radius: 200.0
outer_ring_height: 400.0
inner_ring_height: 200.0
nominal_speed_outer: 28.0
nominal_speed_inner: 20.0
speed_min: 8.0
speed_max: 35.0
safe_distance: 50.0
nmac_distance: 25.0
arrival_rate: variable
```

### 4.3 crazyflie_lab preset

用于室内缩尺验证：

```yaml
name: crazyflie_lab
center: [0.0, 0.0, 0.0]
dt: 0.2
outer_ring_radius: 1.2
inner_ring_radius: 0.7
outer_ring_height: 1.2
inner_ring_height: 0.8
landing_height: 0.15
nominal_speed_outer: 0.35
nominal_speed_inner: 0.25
speed_min: 0.15
speed_max: 0.6
safe_distance: 0.5
nmac_distance: 0.35
collision_distance: 0.2
max_acceleration: 0.8
max_climb_rate: 0.3
max_descent_rate: 0.3
lighthouse_boundary: [-1.8, 1.8, -1.8, 1.8, 0.1, 1.6]
```

## 5. DynamicAirspace 接口规范

建议文件：

```text
terminal_marl/core/dynamic_airspace.py
```

核心接口：

```python
class DynamicAirspace:
    def state_vector(self) -> np.ndarray: ...
    def apply_high_level_action(self, action: np.ndarray) -> dict: ...
    def validate(self) -> bool: ...
    def to_ring_specs(self) -> list: ...
    def get_headways(self) -> dict: ...
```

要求：

- action shape 初期为 `(8,)`；
- action 从 `[-1, 1]` 映射到小幅增量；
- 所有更新必须 clip 到边界；
- 违反约束必须返回 `violations`；
- 支持固定空域模式，即不启用高层时行为不变。

## 6. 动作空间规范

### 6.1 高层动作

```text
a_high = [dR_outer, dR_inner, dH_outer, dH_inner, dV_outer, dV_inner, dHeadway_ring, dHeadway_final]
```

### 6.2 低层动作

推荐：

```text
a_low = [release_request, speed_residual, timing_residual, slot_residual]
```

可先实现前三维，保留 `slot_residual` 为扩展项。

动作解释：

- `release_request > threshold`：请求阶段转移；
- `speed_residual`：映射到 `[-Δv_max, Δv_max]`；
- `timing_residual`：映射到 `[-Δt_max, Δt_max]`；
- `slot_residual`：映射到目标相位偏移。

禁止把最终路线继续扩展为自由：

```text
ψ heading angle
γ flight path angle
arbitrary altitude command
motor / attitude command
```

除非作为对照实验中的 `direct_action_rl_baseline`。

## 7. SafetyFilter 接口规范

建议接口：

```python
class SafetyFilter:
    def filter_intentions(self, states, intentions, airspace) -> tuple[dict, dict]: ...
    def filter_trajectories(self, trajectories, states, airspace) -> tuple[list, dict]: ...
    def check_pairwise_separation(self, trajectories) -> list[str]: ...
    def check_executability(self, trajectory) -> list[str]: ...
```

返回 info：

```python
{
    "approved": bool,
    "modified": bool,
    "rejected_uavs": [...],
    "violations": [...],
    "min_distance": float,
}
```

## 8. TrajectoryGenerator 接口规范

建议文件：

```text
terminal_marl/core/trajectory_generator.py
```

数据结构：

```python
@dataclass
class TrajectoryPoint:
    x: float
    y: float
    z: float
    yaw: float
    t: float

@dataclass
class TrajectoryPlan:
    uav_id: str
    mode: str
    points: list[TrajectoryPoint]
    metadata: dict
```

核心函数：

```python
class TrajectoryGenerator:
    def ring_hold(self, uav_state, ring, duration, dt) -> TrajectoryPlan: ...
    def ring_transfer(self, uav_state, from_ring, to_ring, duration, dt) -> TrajectoryPlan: ...
    def final_approach(self, uav_state, landing_pad, duration, dt) -> TrajectoryPlan: ...
    def landing(self, uav_state, landing_pad, duration, dt) -> TrajectoryPlan: ...
    def emergency_hover(self, uav_state, duration, dt) -> TrajectoryPlan: ...
```

轨迹必须检查：

- waypoint 数量；
- 时间单调；
- 速度上界；
- 加速度上界；
- 高度边界；
- ring 几何一致性。

## 9. 实验模式枚举

建议统一使用：

```python
ExperimentMode = Literal[
    "fixed_airspace_rule",
    "fixed_airspace_low_rl",
    "dynamic_airspace_rule",
    "full_hrl",
    "full_hrl_no_safety",
    "direct_action_rl_baseline",
]
```

每个模式都应有明确开关：

```text
dynamic_airspace: on/off
low_level_policy: rule/rl/direct_action
safety_filter: on/off
trajectory_generator: on/off
hardware_backend: simulation/dry_run/crazyswarm2
```

## 10. 日志与指标规范

必须记录：

```text
airspace radius/height/speed/headway over time
UAV stage transitions
release decisions
safety filter modifications/rejections
trajectory plans
pairwise minimum distances
landing delay
throughput
final queue length
holding time
speed/acceleration violations
tracking error in lab mode
```

建议输出：

```text
results/<run_id>/metrics.json
results/<run_id>/airspace_history.csv
results/<run_id>/uav_history.csv
results/<run_id>/trajectory_history.csv
results/<run_id>/safety_events.csv
results/<run_id>/config.yaml
```

## 11. 测试规范

第一轮必须新增：

```text
tests/test_dynamic_airspace.py
tests/test_trajectory_generator.py
```

后续新增：

```text
tests/test_safety_filter.py
tests/test_action_modes.py
tests/test_experiment_modes.py
tests/test_lab_config_bounds.py
```

每次开发后至少运行：

```bash
python -m pytest tests/test_dynamic_airspace.py tests/test_trajectory_generator.py -v
```

## 12. 硬件适配规范

`Crazyswarm2Adapter` 只接收 `TrajectoryPlan`，不接收裸 RL action。

硬件执行流程：

```text
TrajectoryPlan
  -> preflight dry-run check
  -> boundary and speed check
  -> Crazyswarm2 command conversion
  -> live state monitoring
  -> emergency hover/land if violation
```

真实飞行前必须先完成：

- dry-run trajectory export；
- visualization；
- single UAV rule-only；
- 2 UAV rule-only；
- 2–3 UAV residual policy；
- low-frequency dynamic airspace update。

## 13. YAML 配置加载规范

正式入口：

```python
from terminal_marl.config import load_experiment_config, load_sim_config

config = load_experiment_config("dynamic_hrl")
```

preset 支持以下分层：

```text
experiment_mode
rings
geometry
vehicle_limits
safety
timing
arrival_flow
reward_weights
```

加载器在 Python 默认 `SimConfig` 上覆盖 YAML 字段，因此旧脚本直接使用
`SimConfig()` 的行为保持兼容。当前所有 preset 必须设置
`simulation_only: true`；配置加载不得触发任何硬件逻辑。

## 14. 高层 HRL 时间尺度接口

`TerminalParallelEnv` 提供：

```python
should_update_high_level() -> bool
apply_high_level_action(action) -> dict
get_high_level_observation() -> np.ndarray
```

高层动作只在 `step_count % high_level_interval_steps == 0` 时应用。其他
低层 step 收到的高层动作只缓存最新值，并在下一个高层边界应用。高层观测
包括动态空域状态、ring/final 占用、active traffic、conflict risk 和时间相位。

## 15. 安全放行与轨迹级检查

所有 outer→inner、inner→final 和 final→land 请求必须先经过
`SafetyFilter.approve_release()`。审批至少检查：

```text
current pairwise separation
final approach occupancy
landing pad occupancy
candidate trajectory separation
boundary and height range
monotonic waypoint time
minimum segment duration
speed and acceleration limits
```

拒绝结果为可解释的 `HOLD`，不删除 UAV。不可执行的候选轨迹替换为边界内
stationary hover plan。当前 `FlightPhase.FINAL` 内的下降由 final→land
审批门控，尚未引入独立底层飞控或硬件急停。

## 16. 消融实验 runner

入口：

```bash
python -m terminal_marl.experiments.run_ablation \
  --configs fixed_rule fixed_residual dynamic_rule dynamic_hrl \
  --policy rule-only
```

当前支持 `rule-only`、`random-policy` 和 `dry-run`，统一输出：

```text
landed_count
average_delay
total_completion_time
conflict_count
min_pairwise_distance
safety_override_count
final_queue_time
throughput
```

该 runner 用于 dynamic terminal airspace reconfiguration 与 multi-UAV
landing-flow coordination 的实验工程化，不用于姿态、电机或自由飞行动作控制。

## 17. High-level HRL wrapper

`HighLevelHRLWrapper` 提供 Gym/Gymnasium 风格的 `reset()`、`step()`、
`observation_space` 和 `action_space`。每个 8 维高层动作先更新动态空域，然后
环境内部执行 `high_level_interval_steps` 个低层 step。低层当前支持
rule-only、intention-residual 占位和 random-policy，不包含 PPO 训练。

## 18. High-level interval reward

高层奖励只在完整高层区间结束后计算，包含：

```text
+ landed count / throughput
- final queue waiting
- conflict / safety override
- excessive airspace change
- ring imbalance / congestion
- non-smooth consecutive airspace changes
```

返回值包含每个加权 component、原始增量和 total reward diagnostics。

## 19. Delay 与日志规范

每架 UAV 记录：

```text
spawn_time
entry_time
planned_nominal_landing_time
actual_landing_time
landing_delay = actual - planned
```

每次 rollout 输出 `episode_summary.json`、`step_metrics.csv`、
`airspace_trace.csv`、`safety_events.csv` 和 `agent_events.csv`。summary
记录本地 git commit（若可用），不依赖远程仓库。

## 20. Multi-seed demand sweep

消融 CLI 支持 `--seeds`、`--demand-levels low medium high bursty`、
`--episodes` 和 `--output-dir`。demand profile 通过复制并覆盖
`SimConfig.arrival_flow` 字段实现，不在环境动力学中硬编码。每个组合拥有独立
日志目录，并生成 `aggregate_results.csv`。

## 21. No-safety 消融与 baseline

`full_hrl_no_safety` 只能在 `simulation_only` 下运行，Crazyflie lab 禁止关闭
安全。该模式绕过 release/transition 审批，但仍保留 NaN、动作范围、动态空域
边界和轨迹数值保护。

当前 runner 支持：

```text
rule-only
random-policy
dry-run
high-level-random
high-level-static
```

这些 policy 都是交通意图/空域构型 baseline，不输出自由 heading、altitude
或 velocity vector。
