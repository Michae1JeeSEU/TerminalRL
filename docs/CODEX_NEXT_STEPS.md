# CODEX_NEXT_STEPS（Q1 创新主线版）

> 本文件用于交给 Codex 继续开发。请先阅读 `AGENTS.md`、`docs/research_positioning.md`、`docs/requirements.md`、`docs/design_spec.md`、`docs/technical_spec.md`、`docs/execution_plan.md`，再执行本文件任务。

## 当前开发状态

前四轮基础工程已经完成：

1. **第一轮**：实现 `DynamicAirspace`、`TrajectoryGenerator` 及其测试；
2. **第二轮**：实现 `ExperimentMode`、fixed/dynamic 环境接入、`crazyflie_lab` preset、`SafetyFilter` 骨架和 `intention_residual` 最小闭环；
3. **第三轮**：
   - 实现 YAML → `SimConfig` 加载器，并增加 fixed/dynamic/lab presets；
   - 建立显式 high-level interval、动作缓存和高层观测接口；
   - 将 SafetyFilter 接入 outer→inner、inner→final、final→land 审批；
   - 完善 trajectory boundary/time/speed/acceleration/height/min-segment 检查；
   - 建立 rule-only/random-policy/dry-run 消融 runner 和统一指标。
4. **第四轮**：
   - 新增 Gym-style `HighLevelHRLWrapper`，一个高层动作执行完整低层时间区间；
   - 新增 interval-level high-level reward 与可解释 diagnostics；
   - 增加 nominal landing schedule、严格 landing delay 及分位数指标；
   - 建立 episode summary、step、airspace、safety、agent 五类日志；
   - 支持 multi-seed、low/medium/high/bursty demand 和多 episode sweep；
   - 增加 simulation-only `full_hrl_no_safety` 消融；
   - 增加 high-level-random 与 high-level-static baseline。

以上实现仍然全部属于 simulation/dry-run，不包含 ROS2、Crazyswarm2 或真实 Crazyflie 控制。

## 一、核心原则

后续开发必须服务于论文主线：

```text
Dynamic terminal airspace reconfiguration
+ Multi-UAV landing-flow coordination
+ Safety-filtered executable trajectory generation
+ Crazyflie scaled validation
```

代码主线：

- `terminal_marl/`：唯一主开发线；
- `src/`：legacy prototype，仅参考，不新增主功能。

禁止事项：

- 不要把最终路线继续扩展为自由 `heading/altitude/speed` 飞行动作控制；
- 不要把 ROS2/Crazyswarm2 import 到 `terminal_marl/core/`；
- 不要删除旧代码；
- 不要跳过测试直接接入真实 Crazyflie；
- 不要把 Crazyflie 写成研究对象本身，它只是缩尺验证平台。

## 二、第一轮开发目标

第一轮只做核心基础模块，不训练模型，不接入 ROS2，不真实飞行。

目标：

1. 新增动态空域模块；
2. 新增轨迹生成器初始结构；
3. 新增测试；
4. 保持现有主流程兼容。

## 三、Task 1：新增 DynamicAirspace

创建：

```text
terminal_marl/core/dynamic_airspace.py
```

实现：

```python
@dataclass
class DynamicRingSpec:
    index: int
    radius: float
    height: float
    speed: float
    capacity: int

@dataclass
class AirspaceBounds:
    radius_min: float
    radius_max: float
    height_min: float
    height_max: float
    speed_min: float
    speed_max: float
    min_ring_gap: float
    min_height_gap: float
    max_radius_change_per_update: float
    max_height_change_per_update: float
    max_speed_change_per_update: float
    headway_min: float
    headway_max: float

class DynamicAirspace:
    def __init__(...): ...
    def state_vector(self) -> np.ndarray: ...
    def apply_high_level_action(self, action: np.ndarray) -> dict: ...
    def validate(self) -> bool: ...
    def to_ring_specs(self) -> list: ...
    def get_headways(self) -> dict: ...
```

要求：

- 高层 action shape 初期为 `(8,)`；
- action 从 `[-1, 1]` 映射为小幅增量；
- 更新对象包括 `R_outer, R_inner, H_outer, H_inner, V_outer, V_inner, headway_ring, headway_final`；
- 必须保证 `R_outer > R_inner + min_ring_gap`；
- 必须保证 `H_outer > H_inner + min_height_gap`；
- 所有参数必须 clip 到 bounds；
- 返回 info，例如：

```python
{"clipped": bool, "violations": list[str], "old_state": ..., "new_state": ...}
```

## 四、Task 2：新增 DynamicAirspace 测试

创建：

```text
tests/test_dynamic_airspace.py
```

至少测试：

1. 初始化状态合法；
2. 随机 action 后状态仍合法；
3. 极端 action 会被 clip；
4. ring gap 和 height gap 永远满足；
5. `state_vector()` shape 稳定；
6. `to_ring_specs()` 能兼容现有 ring spec 或返回可被现有代码读取的结构。

## 五、Task 3：新增 TrajectoryGenerator

创建：

```text
terminal_marl/core/trajectory_generator.py
```

实现：

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

class TrajectoryGenerator:
    def ring_hold(self, uav_state, ring, duration: float, dt: float) -> TrajectoryPlan: ...
    def ring_transfer(self, uav_state, from_ring, to_ring, duration: float, dt: float) -> TrajectoryPlan: ...
    def final_approach(self, uav_state, landing_pad, duration: float, dt: float) -> TrajectoryPlan: ...
    def landing(self, uav_state, landing_pad, duration: float, dt: float) -> TrajectoryPlan: ...
    def emergency_hover(self, uav_state, duration: float, dt: float) -> TrajectoryPlan: ...
```

本轮重点：

- 先实现 `ring_hold`；
- 其他函数可以给出可运行的简单版本或清晰 TODO；
- 不依赖 ROS2；
- 输出 trajectory points 可用于 dry-run 和绘图。

## 六、Task 4：新增 TrajectoryGenerator 测试

创建：

```text
tests/test_trajectory_generator.py
```

至少测试：

1. `ring_hold` 输出点数正确；
2. 时间严格递增；
3. 所有点近似位于目标 ring 半径；
4. 所有点高度近似等于 ring height；
5. yaw 为有限值；
6. `TrajectoryPlan.mode == "ring_hold"`。

## 七、兼容性要求

本轮开发后必须满足：

- 不破坏现有 `terminal_marl` imports；
- 不破坏当前训练/评估脚本；
- 不强制安装 ROS2 或 Crazyswarm2；
- 不删除 `src/`；
- 不修改大规模训练逻辑，除非只是为了 import 兼容。

## 八、运行测试

完成后运行：

```bash
python -m pytest tests/test_dynamic_airspace.py tests/test_trajectory_generator.py -v
```

如果项目现有测试存在与本轮无关的问题，请单独说明，不要为了通过测试删除功能。

## 九、完成后汇报格式

请按以下格式汇报：

```text
1. 修改/新增文件
2. 新增核心类和函数
3. 与论文主线的对应关系
4. 测试命令和结果
5. 兼容性说明
6. 下一轮建议
```

## 十、第五轮建议

下一轮继续围绕 HRL 训练证据链，而不是底层飞控：

1. 为 `HighLevelHRLWrapper` 接入独立 high-level PPO trainer；
2. 固化 low-level rule/intention policy checkpoint 接口，避免高低层同时从零训练；
3. 增加 dynamic-rule 的可解释高层启发式 baseline；
4. 执行多 seed × demand × mode 小规模实验，验证日志和 reward 尺度；
5. 增加 aggregate mean/std/CI 与论文表格导出；
6. 校准 nominal landing schedule，使 delay 与 FCFS/uncongested baseline 对齐；
7. 增加 `crazyflie_lab` 轨迹 CSV/3D dry-run 导出，仍不连接真实硬件。
