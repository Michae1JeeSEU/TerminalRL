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

## 22. High-level PPO

`terminal_marl.training.train_high_level_ppo` 使用 SB3 PPO 和
`HighLevelHRLWrapper`。SB3 采用延迟导入；缺失时训练/加载入口返回明确错误，
不会破坏普通仿真 import。训练保存 model、配置、progress CSV 和 eval summary。

## 23. Dynamic-rule baseline

启发式高层策略根据 outer/inner occupancy、final queue、active ratio 和 conflict
risk 生成 8 维空域增量。拥堵时扩环/调速，final 压力时增大 headway，冲突时
降速并增加间隔，空闲时恢复 nominal airspace。动作仍经过 SafetyFilter 和
DynamicAirspace 限幅。

## 24. Runtime arrival process

`ArrivalProcess` 支持 deterministic interval、Poisson 和 bursty wave。新 UAV
受最大活动数和入口安全间隔约束，并记录 spawn、entry 和 nominal landing time。

## 25. Nominal schedule

`NominalScheduleEstimator` 支持 analytic 与 rule-only-reference。所有 delay
结果必须同时记录 `nominal_delay_mode`，避免将 completion time 当作 delay。

## 26. 聚合统计

按 config、policy、demand 分组输出 mean、sample std、min、max、median 和
normal-approximation 95% CI（`mean ± 1.96*std/sqrt(n)`），并记录 seed 数。

## 27. 图表流水线

matplotlib 单图输出 throughput/delay/conflict/safety vs demand，以及 airspace
trace 和 landed count time series；不使用 seaborn 或复杂 subplot。

## 28. Server-side high-level PPO training

服务器训练入口为：

```bash
python -m terminal_marl.training.train_high_level_ppo \
  --training-config terminal_marl/configs/training/ppo_main_medium.yaml
```

新增训练参数包括：

```text
--n-envs
--vec-env dummy/subproc
--device cpu/cuda/auto
--num-threads
--checkpoint-freq
--eval-freq
--save-replay-traces
--resume-from
--training-config
```

默认 `device=cpu`，因为当前 high-level PPO 与仿真环境主要是 CPU 密集型。Stable-Baselines3 保持延迟导入；缺少 SB3 时 PPO 入口给出清晰错误，普通仿真和非 PPO 测试不受影响。

训练只作用于 high-level dynamic airspace policy，不输出自由 heading、altitude 或 velocity-vector 控制命令。

## 29. Training YAML presets

`terminal_marl/configs/training/` 包含：

```text
ppo_smoke.yaml
ppo_pilot_medium.yaml
ppo_main_medium.yaml
ppo_main_high.yaml
ppo_main_bursty.yaml
ppo_reward_sweep.yaml
```

每个 preset 包含 simulation-only 的 config、demand、timesteps、parallel env、PPO hyperparameters、seed list、low-level placeholder policy、nominal delay mode 和 output directory。命令行显式传入参数覆盖 YAML。

## 30. Server scripts and upload manifest

`scripts/server/` 提供：

```text
00_check_env.sh
01_train_smoke_cpu.sh
02_train_pilot_medium_cpu.sh
03_train_main_cpu.slurm
04_train_sweep_array.slurm
05_eval_ablation_cpu.slurm
06_aggregate_and_plot.sh
pack_for_server.py
README_SERVER_TRAINING.md
```

脚本默认 CPU、无 sudo、可选 module 环境。SLURM 分区通过 `PARTITION` 变量设置，默认建议为 `6240-36C-192G`，可改为 `normal` 或其他 CPU 队列。打包脚本生成 `terminal_server_training_package.zip` 与 `SERVER_UPLOAD_MANIFEST.txt`，排除 `.git/`、`runs/`、缓存、bytecode、本地模型和大型日志。

## 31. Reward diagnostics and experiment matrix

`terminal_marl/experiments/reward_diagnostics.py` 对 `rule-only`、`dynamic-rule`、`high-level-random` 和 `untrained-ppo` 做短 dry-run，输出 reward component 的 mean、variance 和 magnitude，帮助正式训练前判断 reward 是否失衡。

`terminal_marl/experiments/experiment_matrix.py` 定义论文推荐组合并可导出 `matrix.csv`：

```text
fixed_rule + rule-only
fixed_residual + random-policy
dynamic_rule + dynamic-rule
dynamic_hrl + high-level-ppo
dynamic_hrl + high-level-random
dynamic_hrl + high-level-static
full_hrl_no_safety + high-level-ppo
```

## 32. Landing pipeline / phase-transition safety regression rules

Server pilot debugging showed that a healthy arrival process can still produce
`landed_count=0` if the landing pipeline is blocked at `outer_to_inner`.
The environment therefore follows these engineering rules:

1. VTAC eligibility is checked before trajectory SafetyFilter approval.  A UAV
   that is not queue head, not ring-established, outside release horizon, or
   blocked by merge/headway constraints receives `vtac_not_eligible`.  This is
   a normal HOLD reason and is not counted as a safety override.
2. Full-scale configs use `max_acceleration: 4.0`, giving enough margin over
   inner-ring centripetal acceleration and ring-transfer transients.
   `crazyflie_lab.yaml` remains scaled and simulation-only with
   `max_acceleration: 0.8`.
3. `SafetyFilter.filter_trajectory_plan()` returns both symbolic violations
   and numeric `kinematic_detail`: max observed speed, max observed
   acceleration, allowed limits, minimum segment duration, and violating segment
   indices.
4. `approve_release()` diagnostics include current and trajectory minimum
   distances.  For `outer_to_inner`, source-ring moving traffic is governed by
   current pairwise checks and VTAC merge/spacing rules; trajectory separation
   focuses on non-source-ring/target-relevant states to avoid treating all
   source-ring aircraft as static obstacles.
5. `safety_override_count` is de-duplicated per high-level interval, UAV,
   target stage, and reason.  `raw_safety_override_count` preserves the raw
   low-level rejection count for debugging.
6. `arrival_rate=0.0` disables runtime arrivals.  Demand sweeps should set
   non-zero rates through `apply_demand_profile()`.

Debug commands:

```bash
python -m terminal_marl.experiments.debug_phase_transitions --config dynamic_hrl --demand-level medium --policy rule-only --initial-uavs 1 --disable-runtime-arrivals --high-level-steps 15 --output-dir runs/debug_phase_transitions_local
python -m terminal_marl.experiments.debug_arrivals --config dynamic_hrl --demand-level medium --policy rule-only --high-level-steps 50 --output-dir runs/debug_arrivals_local
```

Expected local sanity signals:

- single-UAV rule-only reaches `FINISHED`;
- six-UAV rule-only with runtime arrivals disabled reaches inner ring, final,
  and landing completion;
- medium demand debug shows non-zero `inner_occupancy`/`final_occupancy` and
  positive `total_landed_count` over a sufficiently long run.

## 33. Dynamic-rule reward/pipeline repair and readiness metrics

The server pilot exposed a failure mode where `dynamic_rule` could have active
traffic but zero landings.  The repaired baseline keeps the paper scope focused
on dynamic terminal airspace reconfiguration and landing-flow coordination:
high-level actions remain ring radius/height/speed/headway adjustments and
low-level actions remain rule/intention residuals, not free flight control.

Technical updates:

1. `DynamicRuleHighLevelPolicy`
   - consumes normalized pairwise conflict rate from `env.metrics()`;
   - favors bounded speed/headway changes under backlog;
   - uses small nominal recovery toward the configured airspace;
   - avoids default continuous geometry expansion and avoids increasing final
     headway under backlog when final is empty;
   - exposes `last_info` for diagnostics.
2. Pairwise metrics now distinguish raw samples from normalized reward-facing
   risk:
   - `pair_opportunities`
   - `raw_pair_conflict_samples`
   - `unique_conflict_events`
   - `conflict_vehicle_seconds`
   - `normalized_conflict_rate = raw_pair_conflict_samples / pair_opportunities`
   - `has_pair=false` and `min_pairwise_distance=inf` when fewer than two active
     aircraft exist.
3. Waiting/congestion metrics include approach backlog, outer/inner/final
   queues, active/max pressure, and normalized congestion pressure.  If
   approach backlog exists or active count reaches capacity, waiting and
   congestion diagnostics should be non-zero.
4. High-level reward uses interval deltas and normalized rates:
   - landed/throughput positive terms;
   - phase-progress dense reward;
   - normalized waiting, conflict, safety override, airspace change, imbalance,
     congestion, and smoothness penalties.
5. `reward_diagnostics.py` reports mean, std, mean_abs, p50, p90, p99, max_abs,
   component share, and a scaling warning if the largest component mean_abs is
   more than 20x the second largest.
6. `debug_phase_transitions.py` records low-level release/landing approvals and
   writes `dynamic_rule_trace.csv`, `phase_release_summary.csv`,
   `phase_trace.csv`, `release_attempts.csv`, `trajectory_diagnostics.csv`, and
   `summary.json`.
7. `capacity_calibration.py` runs rule-only deterministic-arrival interval
   sweeps and writes `capacity_calibration.csv/json`; it is simulation-only and
   does not start training.

Regression expectations before PPO pilot:

```text
single rule-only no arrivals: landed_count = 1/1
six fixed-rule no arrivals: landed_count >= 5/6
six dynamic-rule no arrivals: landed_count > 0 and >= 80% fixed-rule
medium 50 high-level intervals: dynamic-rule landed_count > 0
no-pair metric: has_pair=false, pair_opportunities=0, normalized_conflict_rate=0
```

## 34. High-level timing study and evaluation protocol

The project now separates four physical-time concepts that were previously
coupled through `high_level_interval_steps`:

```yaml
high_level:
  metrics_sampling_interval_seconds: 20
  decision_interval_seconds: 60
  reconfiguration_dwell_seconds: 60
  geometry_transition_seconds: 20
```

Backward compatibility is preserved: legacy YAML files may still define
`high_level_interval`, and `load_sim_config()` converts it to
`SimConfig.high_level_interval_steps`. If a `high_level:` mapping is present,
`decision_interval_seconds` defines one complete high-level MDP step; the
high-level PPO wrapper therefore receives one observation, emits one action,
and the environment internally advances for the full decision interval. Metrics
sampling can remain at 20 s and does not change the MDP action frequency.

The windowed evaluation protocol uses three phases:

1. Warm-up: arrivals are enabled, but metrics are not counted in the measurement
   window.
2. Measurement: arrivals remain enabled and throughput, delay, waiting,
   backlog, conflict, safety override, and action diagnostics are counted.
3. Drain: arrivals are disabled by setting the runtime arrival process to zero;
   the simulation continues until all active UAVs land or the drain limit is
   reached.

This makes interval ablations comparable by physical duration rather than by
high-level step count. Arrival rates remain in physical seconds and are not
rescaled by the decision interval.

New evaluation-only tools:

- `terminal_marl.experiments.evaluation_protocol`: reusable
  warm-up/measurement/drain runner and timing helpers.
- `terminal_marl.experiments.airspace_response_diagnostics`: one-shot response
  probes for radius, speed, ring headway, final headway, and safe combination
  changes.
- `terminal_marl.experiments.high_level_interval_study`: complete dry-run study
  that writes timing audit, response diagnostics, interval sweep, horizon
  censoring diagnosis, demand separability, action effectiveness, and final
  retraining recommendation artifacts.

These tools are simulation-only. They do not connect to ROS2, Crazyswarm2, or
Crazyflie hardware, and they do not start PPO training.

## 35. Event-gated 180s high-level protocol

The active training protocol is now a fixed 180 s high-level MDP with 20 s
metric sampling and a 600 s minimum dwell for structural geometry changes.
`decision_interval_seconds` must be divisible by
`metrics_sampling_interval_seconds`; the YAML loader fails fast if the timing
cannot be represented by `dt`.

`GeometryEventGate` is implemented in
`terminal_marl/control/geometry_event_gate.py`. The gate owns persistence,
hysteresis, dwell, nominal-recovery eligibility, action masking, and diagnostic
state. `TerminalParallelEnv` samples pressure metrics every 20 s and exposes
gate state in high-level observation/info/logs.

Action semantics remain the stable eight-dimensional incremental
`DynamicAirspace` action:

- indices 0--3: structural geometry (`outer_radius`, `inner_radius`,
  `outer_height`, `inner_height`);
- indices 4--7: operational flow parameters (`outer_speed`, `inner_speed`,
  `ring_headway`, `final_headway`).

When the gate is closed, only indices 0--3 become explicit no-ops. Operational
indices continue through the existing SafetyFilter/DynamicAirspace limits.
Reward diagnostics separate requested/applied action and include an applied
geometry-change cost, so masked geometry requests do not create geometry-change
penalty.

New simulation-only modes:

- `event_rule`: rule-only low-level flow with event-gated structural geometry;
- `periodic_hrl`: fixed 180 s high-level decisions, geometry allowed by dwell;
- `event_gated_hrl`: fixed 180 s high-level decisions, geometry allowed only by
  event gate and dwell.

Training presets for the new PPO pilots use an episode-level medium/high/bursty
mixture (0.40/0.30/0.30). These runs must start from scratch; never resume an
old 20 s PPO checkpoint after changing the high-level MDP interval.

## 36. Stage-1.5 evaluation integrity diagnostics

Stage-1.5 is a diagnostics-only phase after the event-gated/periodic 20k pilot.
It may load existing 20k checkpoints for evaluation, but it must not start PPO
learning or resume training.

### Arrival request and admission semantics

`ArrivalProcess` generates policy-independent external request demand.  It no
longer censors requests when the terminal reaches `max_active_uavs`.
`TerminalParallelEnv` interprets `max_active_uavs` as an internal active-aircraft
cap.  When the cap is full, requests enter a FIFO external/pre-entry queue.

Metrics exposed by `env.metrics()` include:

- `arrival_request_count`;
- `admitted_arrival_count`;
- `blocked_arrival_count`;
- `deferred_arrival_count`;
- `dropped_arrival_count`;
- `external_queue_count`;
- `external_queue_waiting_time`;
- `active_cap_hit`;
- `time_at_active_cap_seconds`;
- `fraction_time_at_active_cap`;
- `longest_continuous_cap_seconds`;
- `time_to_first_cap`;
- `internal_active_count`;
- `internal_capacity_ratio`;
- `external_backlog`;
- `total_system_backlog`.

`total_spawned_count` remains the number of actual UAV instances in the
terminal, including the initial queue and admitted runtime arrivals.  It is not
the same as external request demand.

### Common random numbers

`terminal_marl.core.arrival_process.generate_arrival_schedule()` creates a
deterministic `scenario + seed + physical duration` request schedule.
`run_windowed_evaluation(..., fixed_arrival_schedule=...)` replays that exact
schedule so rule-only, event-rule, periodic PPO, event-gated PPO, and random
baselines see identical request counts.  Policies can affect admission time,
external backlog, and completion, but not external request count.

### Windowed throughput

The formal steady-state throughput metric is
`measurement_throughput_per_second` / `measurement_throughput_per_hour`, computed
only over the measurement window.  Warm-up landings and drain landings are
reported separately:

- `warmup_landed`;
- `measurement_landed`;
- `measurement_duration_seconds`;
- `measurement_throughput_per_second`;
- `measurement_throughput_per_hour`;
- `drain_landed`;
- `drain_duration_seconds`;
- `total_landed`;
- `overall_throughput_per_hour`.

### Action, gate, and airspace diagnostics

Every sampled evaluation row contains requested/applied 8D action fields,
geometry gate active/allowed/dwell/mask diagnostics, configured/effective dwell,
airspace parameters, raw/normalized congestion/waiting/backlog, and cap/external
queue metrics.  With 180 s high-level decisions and 600 s configured dwell, the
effective geometry dwell on the decision grid is 720 s.

### Stage-1.5 runner

`terminal_marl.experiments.stage15_diagnostics` writes the final diagnostic
bundle under:

```text
$HOME/training_outputs/terminal/event_gated_stage15_diagnostics/final
```

The only valid final decision values are:

```text
READY_FOR_50K_WITH_CURRENT_MDP
RETRAIN_REQUIRED_AFTER_MDP_FIX
ENVIRONMENT_NOT_READY
```

Because the external-queue fix changes MDP/reward semantics, existing pilots are
diagnostic evidence only unless a future run explicitly retrains under the fixed
semantics.

## PRE-RETRAIN core model revision: dynamic capacity and finite cohorts

Before retraining high-level PPO, the terminal model uses geometry-derived ring
capacity rather than fixed ring occupancy constants. For each ring with actual
radius `R` and required separation `d`, capacity is:

```text
d_adj(N) = 2 R sin(pi / N)
capacity(R, d) = 1, if d > 2R
capacity(R, d) = floor(pi / asin(d / (2R))), otherwise
```

The implementation lives in `terminal_marl/core/ring_capacity.py` and is used by
`DynamicAirspace`. Capacity is recomputed from the actual current radius, so a
geometry transition exposes intermediate capacity rather than the final target
capacity. Radius contraction is projected against current VTAC ring occupancy;
unsafe shrink requests are clipped and reported with
`radius_contraction_limited` diagnostics.

Finite-cohort evaluation is configured through `SimConfig.total_requested_uavs`.
When it is set, runtime arrival generation creates exactly `N` external
requests. Every requested UAV must end in one primary terminal outcome:

```text
LANDED | CRASHED_COLLISION | CRASHED_BATTERY | UNRESOLVED | INVALID
```

`UNRESOLVED` and `INVALID` are diagnostics, not successful formal outcomes. The
environment exposes `simulation_outcome_summary()` and `uav_outcome_rows()` for
conservation checks and `uav_outcomes.csv` logging.

Battery endurance is currently a transparent time-based model:
`battery_endurance_seconds` is decremented from `request_time`, including time
in the external queue. A depleted UAV is removed from queues/control/conflict
accounting and marked `CRASHED_BATTERY`. A collision-distance breach marks both
partners `CRASHED_COLLISION` and counts the unordered pair once.

This revision remains simulation-only: no ROS2, Crazyswarm2, or real Crazyflie
hardware is connected, and low-level policy is not expanded into free
heading/altitude/velocity control.
