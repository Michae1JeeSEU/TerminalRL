# Terminal MARL 低空终端区实时 3D 可视化平台开发总领

> **文档用途**：作为 Codex 开发 `terminal_marl/` 主线实时可视化模块的总领性任务说明。  
> **任务性质**：附加型诊断/观察模块，不改变现有仿真、RL、轨迹生成与安全逻辑。  
> **核心目标**：训练时默认完全关闭可视化；需要诊断策略时，以一个开关启动轻量实时 3D GUI，直观看到低空终端区空域、飞行器与降落过程。  
> **优先级**：低侵入、语义零影响、可关闭、易诊断 > 画面精细度。  
> **主开发线**：仅针对 `terminal_marl/`；`src/` 是 legacy 原型，不在其中新增正式功能。

---

## 0. 先给结论：这个可视化模块是必要的

本项目已经进入“策略是否真正学会正确的交通组织行为，而不仅仅是指标变好”的阶段。仅依赖 reward、throughput、delay、collision/NMAC、release approval 等聚合指标，已经不足以高效诊断以下问题：

- UAV 是否在某一环上产生异常拥堵或长时间绕飞；
- 学到的 release 行为是否符合预期的流量组织逻辑；
- UAV 是否在错误的时机切环；
- merge reservation 与真实运动是否在空间上直观一致；
- final approach 是否形成合理的进近序列；
- 某些 safety intervention 是合理保护还是策略持续制造不良状态；
- dynamic airspace 改变半径/高度时，交通是否出现不合理的挤压、等待或冻结；
- 某个 reward 改动后，策略究竟改变了什么物理行为；
- collision/NMAC/长时间等待出现前，场景中到底发生了什么；
- 高层空域调节与低层放行之间是否形成了可解释的协同。

因此建议把实时 3D 可视化定义为：

> **研究诊断工具（diagnostic observer），而不是仿真核心组成部分。**

它不参与决策，不改变状态，不修正动作，也不承担仿真真值。它只是观察 authoritative simulation state。

这一点也与当前 canonical `docs/requirements.md` 一致：项目已经明确要求同一份 authoritative trajectory plan 用于：

```text
safety check
simulation execution
logging
visualization
future hardware adapter
```

所以本任务不是增加与论文无关的“漂亮界面”，而是补齐当前架构已经要求、但尚未正式实现的 visualization consumer。

---

# 1. 当前代码状态与本任务相关的事实

Codex 开发前必须先阅读：

```text
docs/research_positioning.md
docs/requirements.md
docs/design_spec.md
docs/technical_spec.md
docs/execution_plan.md
docs/revision.md
docs/PROJECT_HANDOFF_2026-08-18.md
AGENTS.md
```

并再次检查：

```bash
git status --short
```

当前仓库有大量历史未提交修改。**禁止 reset、checkout、clean、覆盖或顺手整理用户已有改动。**

## 1.1 正式主线

当前正式主线是：

```text
terminal_marl/
```

`src/` 已被明确标记为 legacy，不允许为了复用旧可视化脚本而把新功能重新接回 `src/`。

仓库根目录已有：

```text
_visualize_phase3a.py
_visualize_phase3b.py
_visualize_static.py
_visualize_n30_diagnostic.py
```

这些脚本可以作为“过去确实需要可视化诊断”的参考，但它们：

- 基于 legacy `src/`；
- 主要生成 matplotlib 2D/GIF；
- 与旧状态结构强耦合；
- 不应成为新模块的正式依赖；
- 不应直接修改后继续作为 canonical visualization。

## 1.2 当前主环境已经拥有可视化所需的大部分真值

`terminal_marl/envs/parallel_env.py::TerminalParallelEnv` 已维护：

```text
env.now
env.step_count
env.active
env.service_uavs
env.completed_uavs
env.admission
env.airspace
env.lz_manager
env.scheduler
env.telemetry
env.conflict_events
env.runtime_trace
env.last_release_decisions
env.last_safety_decisions
```

每个 `UAVState` 已包含：

```text
uid
arrival_id
phase
position
velocity
speed
ring_role
ring_progress
assigned_gate
assigned_lz
trajectory_plan_id
merge_reservation_id
committed_plan
energy_remaining
energy_capacity
dynamic_priority
```

因此**不要为可视化再创建一套物理状态或重复维护 UAV 坐标。**

## 1.3 三环真实几何已经可直接读取

`DynamicAirspace` 同时维护 actual 与 target geometry：

```python
airspace.actual_ring(role)
airspace.target_ring(role)
airspace.transition
airspace.transition_progress(now)
airspace.capacity(role)
airspace.occupancy(role)
```

三环默认结构约为：

```text
Outer:  R=600 m, H=500 m
Middle: R=400 m, H=350 m
Inner:  R=200 m, H=200 m
```

可视化必须读取**当前 actual geometry**，不能重新根据配置自行推导动态环位置。

当 geometry transition 正在发生时，可以额外显示 target geometry，但 target 只能作为辅助参考，不能代替 actual geometry。

## 1.4 Gate 与 LZ 已有真实位置

4 个真实 entry gate 位于：

```text
0°
90°
180°
270°
```

并由：

```text
env.admission.gates[*].spawn_point
env.admission.gates[*].gate_point
```

提供坐标。

6 个 Landing Zone 已由：

```text
env.lz_manager.zones[*].position
```

提供真实位置与资源状态：

```text
FREE
RESERVED
OCCUPIED
```

不要在渲染器里重新复制一组硬编码 LZ 坐标。

## 1.5 Canonical trajectory 已经可以直接用于轨迹显示

`UAVState.committed_plan` 是当前执行中的 canonical `TrajectoryPlan`。

其包含：

```text
plan_id
uav_id
mode
points
start_time
end_time
reservation_ids
metadata
```

可视化若显示 trajectory，必须直接使用：

```text
uav.committed_plan.points
```

禁止为了画图重新计算一条“看起来差不多”的 Bezier/final path。

否则会违反：

> same plan checked == same plan executed == same plan visualized

这一 canonical 约束。

---

# 2. 任务边界

## 2.1 本任务必须实现

开发一个**可选、实时、交互式 3D terminal-area viewer**，至少支持：

1. 三个环形空域；
2. 当前 actual 半径与高度；
3. 4 个 entry corridor / gate；
4. 6 个起降场 Landing Zone；
5. 当前 airborne UAV；
6. UAV 的实时 3D 位置；
7. 基本 phase/ring 区分；
8. 鼠标拖拽旋转视角；
9. 平移；
10. 缩放；
11. 仿真时间、active 数、external queue、completed 等基本 HUD；
12. `gui=False` 时完全不启动可视化；
13. `gui=True` 时用于策略 rollout 的实时观察；
14. fixed airspace 和 dynamic airspace 都能工作；
15. fixed-rule、fixed-low-RL、dynamic-rule、learned-high/frozen-low 等现有模式可复用同一 viewer；
16. 能直接观察已训练 checkpoint 的 deterministic rollout。

## 2.2 推荐实现但不是第一版硬门槛

在基础版本稳定后增加：

- committed trajectory path；
- selected UAV 信息；
- target ring；
- geometry transition progress；
- ring occupancy/capacity；
- LZ free/reserved/occupied 状态；
- release/safety decision；
- hard separation/NMAC/collision 事件高亮；
- pause/resume；
- single-step；
- 1x / 2x / 5x / 10x / max playback；
- UAV ID 标签开关；
- trajectory 显示开关；
- top / isometric 快捷视角；
- screenshot。

## 2.3 明确不做

本任务**不要**：

- 制作高精度城市模型；
- 加建筑物、道路、地形等无关资产；
- 制作真实无人机 3D 模型；
- 把 UAV 做成复杂 rigid-body；
- 在 GUI 中直接控制飞行器；
- 从 GUI 修改 action；
- 从 GUI 修改 safety threshold；
- 在 GUI 中重新实现 collision detector；
- 在 GUI 中重新生成 trajectory；
- 重写 `TrajectoryGenerator`；
- 重写 `SafetyFilter`；
- 重写 `MergeScheduler`；
- 改 PPO action/observation/reward；
- 改 checkpoint schema；
- 改 A5 checkpoint；
- 改现有 A6 checkpoint；
- 降低 80 m safe distance；
- 为了 viewer 修改仿真时间推进语义；
- 在本任务中训练或重新训练 PPO；
- 修改 legacy `src/` 作为正式实现。

---

# 3. 第一设计原则：可视化必须是“旁路观察者”

目标数据流：

```text
                 ┌──────────────────────┐
actions ───────► │ TerminalParallelEnv  │
                 │ authoritative world  │
                 └──────────┬───────────┘
                            │ read-only
                            ▼
                 ┌──────────────────────┐
                 │ RenderSnapshotBuilder│
                 │ copy only needed data│
                 └──────────┬───────────┘
                            │ immutable snapshot
                            ▼
                 ┌──────────────────────┐
                 │ Terminal3DViewer     │
                 │ visualization only   │
                 └──────────────────────┘
```

绝对禁止：

```text
viewer
  ↓
modify env.active
modify UAV.position
modify airspace
modify committed_plan
modify action
modify safety result
```

即：

> **Simulation owns truth. Viewer only consumes snapshots.**

---

# 4. 第二设计原则：不要把 GUI 配置塞入 canonical `SimConfig`

不建议新增：

```python
SimConfig(render=True)
SimConfig(gui=True)
```

原因：

`SimConfig` 当前不仅是运行参数，也参与 canonical experiment semantics、checkpoint metadata 和大量测试。

可视化属于**运行/诊断层配置**，不是 physics / traffic / RL semantics。

推荐新增独立：

```python
@dataclass
class VisualizationConfig:
    enabled: bool = False
    frame_stride: int = 1
    show_trajectories: bool = True
    show_labels: bool = False
    show_target_geometry: bool = True
    playback_speed: float = 0.0
```

其中：

```text
enabled=False
```

必须是默认值。

CLI 对用户暴露：

```text
--gui
--no-gui
```

或：

```text
--gui true
--gui false
```

但内部不要污染 `SimConfig`。

---

# 5. 第三设计原则：默认关闭时，训练路径不能加载任何 GUI 依赖

训练服务器与 headless 环境不应因为新增 viewer 被迫安装或 import VTK/Qt。

必须满足：

```python
from terminal_marl.envs.parallel_env import TerminalParallelEnv
```

在没有安装 PyVista 的环境中仍然正常。

禁止在这些模块顶层 import：

```text
terminal_marl/core/*
terminal_marl/envs/parallel_env.py
terminal_marl/envs/high_level_wrapper.py
terminal_marl/algo/*
terminal_marl/training/*
```

禁止：

```python
import pyvista
```

或 GUI toolkit。

GUI 依赖只允许出现在：

```text
terminal_marl/visualization/*
```

并采用 lazy import。

若用户启动：

```text
--gui
```

但未安装 visualization dependency，应给清晰错误，例如：

```text
Visualization dependencies are not installed.
Install optional visualization requirements and retry.
```

而不是影响普通训练/评估。

---

# 6. 推荐技术方案

## 6.1 第一版推荐：PyVista

第一版建议使用：

```text
PyVista
```

原因：

- 原生 3D；
- 能直接显示点、线和简单 mesh；
- 默认支持鼠标旋转、平移、缩放；
- 可以持续修改 points 后 redraw；
- 支持 non-blocking interactive update；
- 当前需求只需要几十架 UAV + 3 个 ring + 少量 trajectory，性能足够；
- 比完整游戏引擎或自建 OpenGL GUI 轻得多；
- 可以不引入复杂 Qt 主窗口。

MVP 不建议一开始就做完整 PySide6/PyQt 主界面。

优先尝试：

```python
pyvista.Plotter()
plotter.show(interactive_update=True, ...)
plotter.update()
```

做到单窗口实时更新。

只有在后续确实需要复杂 dock、按钮、timeline 控件时，再考虑：

```text
pyvistaqt + PySide6
```

## 6.2 依赖隔离

推荐新增：

```text
requirements-visualization.txt
```

例如只放：

```text
pyvista
```

若后续使用 `pyvistaqt` 再加入：

```text
pyvistaqt
PySide6
```

不要为了 viewer 直接把训练服务器主 requirements 变成强 GUI 依赖。

## 6.3 交互与可读性补充需求（2026-08-26）

第一版 GUI 必须采用**白色或近白色背景**，HUD、Gate 标签和 selected-UAV 文本采用高对比深色文字。三个 actual ring 必须保持稳定且彼此可区分的颜色；target ring 只能使用低饱和、低透明度的辅助样式。canonical trajectory 使用中性细线，避免与 ring 或 UAV 混淆。

每架 UAV 必须按 `uid` 分配稳定的身份颜色，且在 INBOUND、各 ring、transfer、FINAL 等 phase 之间不改变颜色。phase、ring 和其他运行状态应通过选中信息面板表达；唯一允许覆盖身份颜色的情况是 authoritative conflict/NMAC/collision 事件的短时诊断高亮。

GUI 必须提供运行中可用的控制：

- Pause / resume 开关：暂停只能阻塞后续 wall-clock tick，不能修改 `config.dt`、动作、随机序列或 authoritative state；
- playback speed 控件：支持至少 `0.25x / 0.5x / 1x / 2x / 5x / 10x / max`，仅改变 renderer observer 的 wall-clock delay；
- 鼠标选中 UAV：选中后在本地面板显示 `uid`、arrival ID、phase、ring、speed、剩余能量/SOC、mission、base/dynamic priority、gate、LZ、plan/reservation ID、最后的 release/safety decision。该面板只消费 immutable snapshot，不能直接读取或修改 env。

---

# 7. 最小侵入式集成方案

## 7.1 允许修改现有核心文件的范围

理想情况下，现有正式代码只需要修改一个文件：

```text
terminal_marl/envs/parallel_env.py
```

修改内容仅用于增加**通用 step observer hook**。

不要 import visualization package。

推荐形态：

```python
class TerminalParallelEnv:
    def __init__(...):
        self._step_observer = None
        ...

    def set_step_observer(self, observer):
        self._step_observer = observer

    def clear_step_observer(self):
        self._step_observer = None
```

在每个完整 simulator `dt` 推进结束以后：

```python
if self._step_observer is not None:
    self._step_observer(self)
```

注意：

- 不改变 `step(actions)` 入参；
- 不改变 `step()` 返回五元组；
- 不改变 `reset()` 返回；
- 不改变 action schema；
- 不改变 observation；
- 不改变 reward；
- 不改变 termination；
- 不改变时间推进；
- 不改变任何 safety/trajectory 调用顺序。

observer 调用点必须位于一个完整仿真 tick 已经执行完、`env.now` 与 UAV 坐标一致之后。

## 7.2 为什么需要这个 hook

对于普通 low-level rollout：

```python
env.step(action)
viewer.update(...)
```

外层 runner 可以直接更新 GUI。

但 `DynamicHighLevelWrapper.step()` 当前内部会进行：

```text
约 180 秒 / dt
```

的连续 low-level simulation：

```python
for ...:
    self.base.step(low_actions)
```

如果只在高层 `wrapper.step()` 返回后刷新 viewer，画面会每 180 秒跳一次，失去实时观察价值。

因此让 `TerminalParallelEnv` 每一个 canonical dt 完成后发出 observer callback，是侵入最小、又能同时兼容 low-level 与 high-level wrapper 的方案。

## 7.3 observer 不应做什么

observer 本身不能：

```text
sleep as part of simulation semantics
change env
return modified state
change action
raise policy decisions
```

若需要降低显示帧率，应在 visualization adapter 内部做：

```text
frame_stride
render throttle
drop frame
```

不要修改 simulator dt。

---

# 8. 新模块建议目录

建议新增：

```text
terminal_marl/
└── visualization/
    ├── __init__.py
    ├── config.py
    ├── schema.py
    ├── snapshot.py
    ├── pyvista_viewer.py
    └── rollout_viewer.py
```

测试：

```text
tests/
├── test_visualization_snapshot.py
└── test_visualization_integration.py
```

可选依赖：

```text
requirements-visualization.txt
```

不要新增到：

```text
src/
```

不要把 viewer 放进：

```text
terminal_marl/core/
```

---

# 9. Visualization snapshot 数据契约

不要让 renderer 到处直接访问几十个 env 内部字段。

应建立一个**轻量、只读、与渲染后端无关**的 snapshot schema。

示意：

```python
@dataclass(frozen=True)
class RingRenderState:
    role: str
    radius: float
    height: float
    target_radius: float
    target_height: float
    capacity: int
    occupancy: int


@dataclass(frozen=True)
class GateRenderState:
    gate_id: int
    spawn_position: tuple[float, float, float]
    gate_position: tuple[float, float, float]


@dataclass(frozen=True)
class LandingZoneRenderState:
    lz_id: int
    position: tuple[float, float, float]
    status: str
    reserved_by: int | None
    occupied_by: int | None
    service_remaining: float


@dataclass(frozen=True)
class UAVRenderState:
    uid: int
    arrival_id: int
    phase: str
    ring_role: str | None
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    speed: float
    soc: float
    assigned_gate: int
    assigned_lz: int | None
    trajectory_plan_id: str | None
    merge_reservation_id: str | None
    committed: bool
    trajectory_points: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class WorldRenderSnapshot:
    time: float
    step_count: int
    rings: tuple[RingRenderState, ...]
    gates: tuple[GateRenderState, ...]
    landing_zones: tuple[LandingZoneRenderState, ...]
    uavs: tuple[UAVRenderState, ...]
    external_queue_size: int
    service_count: int
    completed_count: int
    transition_active: bool
    transition_progress: float
    telemetry: dict
```

实际字段可小幅调整，但必须遵守：

> renderer 只依赖 snapshot，不直接依赖 policy、PPO、scheduler 或 SafetyFilter。

---

# 10. SnapshotBuilder 的数据来源要求

## 10.1 Ring

读取：

```python
for role in RingRole:
    actual = env.airspace.actual_ring(role)
    target = env.airspace.target_ring(role)
```

读取：

```text
actual.radius
actual.height
target.radius
target.height
env.airspace.capacity(role)
env.airspace.occupancy(role)
```

禁止从 `SimConfig.rings` 获取 current ring，因为 dynamic geometry 可能已改变。

## 10.2 UAV

读取：

```text
env.active.values()
```

作为 airborne traffic。

需要显示 ground service 时，可额外读取：

```text
env.service_uavs
```

但 SERVICE UAV 不应重新被当成 airborne UAV。

可将 service 状态显示在对应 LZ 上，而不是把它重新放回空中。

## 10.3 External queue

External queue 当前没有真实 terminal 内物理位置。

因此第一版：

> **不要伪造 external queue UAV 的 3D 坐标。**

只在 HUD 中显示：

```text
external queue = N
```

以后如果研究需要展示 upstream corridor queue，再单独定义真实 upstream visualization semantics。

## 10.4 Trajectory

若：

```python
uav.committed_plan is not None
```

直接复制：

```python
tuple(tuple(point.position) for point in uav.committed_plan.points)
```

禁止调用 `TrajectoryGenerator` 再生成。

## 10.5 Telemetry

第一版 HUD 可读取：

```text
minimum_actual_separation
minimum_swept_separation
hard_separation_violations
nmac
collisions
safety_interventions
release_approved
release_rejected
geometry_updates
```

如果某字段在当前 run 不存在，viewer 应 graceful fallback，不要改变 core telemetry schema 只为画面服务。

---

# 11. 3D 场景最低要求

## 11.1 世界坐标

直接使用模拟单位：

```text
x: meter
y: meter
z: meter
```

不做额外 physics scaling。

为视觉效果可以设置 camera clipping / axis scale，但 UAV 数据本身不得被改写。

## 11.2 三个环

每个 ring 显示为：

```text
center = (0, 0, H_actual)
radius = R_actual
```

推荐使用 circle/polyline。

三个 ring 必须容易区分。

如果 geometry transition 发生：

- 实线/主线：actual ring；
- 半透明/虚线：target ring（可选）。

target ring 不能替代 actual。

## 11.3 Gate

每个 gate 至少画：

```text
spawn_point
gate_point
```

并可用一条细线连接，表达 inbound corridor。

## 11.4 Landing Zone

6 个 LZ 在 z=0 附近显示为：

```text
small disk / square / point
```

根据：

```text
free
reserved
occupied
```

改变视觉状态。

不要建造复杂 vertiport 模型。

## 11.5 UAV

第一版 UAV 使用：

```text
point / small sphere
```

即可。

不做真实机身 mesh。

建议区分：

```text
INBOUND
RING_OUTER
RING_MIDDLE
RING_INNER
TRANSFER
FINAL
```

可通过颜色、marker 或大小表达。

必须确保：

> 视觉编码只是显示，不能反向影响仿真状态。

## 11.6 UAV label

默认：

```text
show_labels=False
```

因为 20–60 架 UAV 全显示 label 会严重遮挡。

允许按键切换：

```text
UID
phase
SOC
```

作为第二阶段功能。

---

# 12. Camera / interaction 要求

MVP 必须拥有：

- 左键拖拽旋转；
- 平移；
- 鼠标滚轮缩放；
- reset camera；
- isometric view；
- top-down view。

建议快捷键：

```text
R  reset camera
I  isometric
T  top view
L  labels on/off
P  trajectories on/off
Space pause/resume   # 如果本阶段实现 playback control
N  single step       # 可选
```

不要为这些功能自研复杂 GUI 框架；优先使用 renderer 已有 camera / key event。

---

# 13. HUD 最低要求

画面角落显示纯文本即可。

至少：

```text
Simulation time
Step
Active airborne UAVs
External queue
Service
Completed

Outer occupancy / capacity
Middle occupancy / capacity
Inner occupancy / capacity

Geometry transition: yes/no
Minimum actual separation
Minimum swept separation
Hard separation events
NMAC
Collisions
```

若是 dynamic high-level rollout，可选再显示：

```text
last high action time
geometry gate
transition progress
```

第一版不需要完整 dashboard。

---

# 14. 实时刷新与性能要求

训练时：

```text
gui=False
```

不得建立 snapshot，不得 render。

诊断时：

```text
gui=True
```

才启动。

## 14.1 frame stride

支持：

```python
frame_stride = 1
```

表示每 simulator tick 都可发送一帧。

也支持：

```text
2
5
10
```

用于高负载快速观察。

注意：

> frame stride 只能跳过“画面刷新”，不能跳过 simulator step。

## 14.2 不允许 GUI backpressure 改变仿真逻辑

如果 renderer 来不及：

```text
prefer dropping stale render frames
```

而不是让一堆历史帧不断堆积。

对于第一版单线程 PyVista 实现，可以只保留“最新状态”并立即 redraw，不需要创建无限 frame queue。

若未来改为 thread/process：

```text
queue maxsize = 1~3
```

并采用 drop-oldest/latest-state 策略。

## 14.3 playback speed 与 simulator time 分离

如果需要看慢动作：

```text
wall clock delay
```

只放在 visualization runner。

禁止修改：

```text
config.dt
```

来实现“播放速度”。

仿真 1 秒仍然必须是仿真 1 秒。

---

# 15. 建议的启动接口

## 15.1 fixed rule

```bash
python -m terminal_marl.visualization.rollout_viewer \
  --mode fixed_rule \
  --seed 0 \
  --gui
```

## 15.2 frozen low-level RL

```bash
python -m terminal_marl.visualization.rollout_viewer \
  --mode fixed_low_rl \
  --low-checkpoint <path/to/fixed_low_rl.pt> \
  --seed 0 \
  --gui
```

## 15.3 dynamic rule

```bash
python -m terminal_marl.visualization.rollout_viewer \
  --mode dynamic_rule \
  --seed 0 \
  --gui
```

## 15.4 learned high + frozen low

```bash
python -m terminal_marl.visualization.rollout_viewer \
  --mode learned_high_frozen_low \
  --high-checkpoint <path/to/high_level.pt> \
  --low-checkpoint <path/to/fixed_low_rl.pt> \
  --seed 0 \
  --gui
```

具体 CLI 名称可调整，但必须满足：

- 不复制 checkpoint loading logic；
- 复用现有 `load_checkpoint`；
- 复用 `FrozenLowLevelPolicy`；
- 复用 `load_high_level`；
- deterministic evaluation；
- 不在 visualization runner 中重新实现 PPO forward semantics。

---

# 16. `gui=False` 的严格语义

这是本任务最重要的验收条件之一。

关闭 GUI 时应等价于现在的程序：

```text
no PyVista import
no plotter
no render snapshot
no render callback
no wall-clock sleep
no extra trajectory generation
no state mutation
```

下面两次运行：

```text
same config
same seed
same policy/actions
```

一份：

```text
visualization disabled
```

一份：

```text
原始环境 / 没挂 observer
```

必须得到相同：

```text
UAV state sequence
outcomes
telemetry
reward
termination time
collision/NMAC
trajectory plan IDs under same execution
```

允许的唯一区别：

```text
wall-clock runtime
```

而且 disabled 模式的额外 wall-clock overhead 应接近零。

---

# 17. `gui=True` 也不得改变研究语义

开启 viewer 后：

```text
same seed
same deterministic policy
```

必须仍得到与无 viewer 相同的最终 simulation outcome。

可接受：

```text
运行变慢
```

不可接受：

```text
UAV 路径不同
release decision 不同
arrival stream 不同
geometry action 不同
random sequence 不同
collision 结果不同
```

如果发生，说明 viewer 侵入了仿真，需要修复后才能交付。

---

# 18. DynamicAirspace 的特殊要求

这是新 viewer 与旧 `_visualize_phase3*.py` 最大差异之一。

当前 geometry 不是永远固定的。

因此 viewer 必须每帧更新：

```text
Outer R/H
Middle R/H
Inner R/H
```

不能只在初始化时画三条固定圆环。

对于 ring holder：

当前 `TerminalParallelEnv._advance_ring_hold()` 会让 UAV 跟随 moving actual ring。

因此 viewer 只需要显示 `UAVState.position`。

不要在 viewer 内重新计算 holder 的圆周位置。

---

# 19. Canonical trajectory 显示要求

trajectory 可视化是非常有价值的诊断功能，建议列入 P1。

对于 committed UAV：

```text
current position
+
remaining/current full canonical plan
```

可以显示为 3D polyline。

建议：

- 只显示 committed plan；
- 默认显示 active UAV 的 committed path；
- 不显示无穷历史轨迹；
- 可选只显示 selected UAV path。

尤其用于诊断：

```text
INBOUND
O2M
M2I
I2F / FINAL
future-reserved delayed release
```

### 强制要求

viewer 不能：

```python
generator.ring_transfer(...)
generator.final_approach(...)
```

来画线。

它必须消费原计划。

---

# 20. Safety / conflict 诊断建议

第一版无需制作复杂 collision volume。

建议仅增加简单视觉提示：

当：

```text
hard separation
NMAC
collision
```

新事件发生时：

- HUD counter 更新；
- 相关 UAV 暂时高亮；
- 可选显示两机连线；
- 可选在 terminal 输出事件信息。

事件数据以现有：

```text
env.conflict_events
env.telemetry
```

为准。

不要在 viewer 里再次计算 pairwise safety 并把结果当 authoritative。

---

# 21. Release / scheduler 行为诊断建议

后续可选：点击/选中 UAV 时显示：

```text
uid
arrival_id
phase
ring
speed
SOC
dynamic priority
assigned LZ
merge reservation ID
trajectory plan ID
last release decision
last safety decision
```

这是比增加复杂机体模型更有研究价值的功能。

如果时间有限，优先做这些状态信息，而不是美术。

---

# 22. 开发阶段划分

## V0 — 基线冻结与接口确认

### 任务

1. 阅读 canonical docs；
2. `git status --short`；
3. 确认 `TerminalParallelEnv.step()` 的结束状态时点；
4. 确认 `DynamicHighLevelWrapper` 内部 low-level loop；
5. 确认 checkpoint load 路径；
6. 建立一个 deterministic baseline state signature。

### 不修改功能。

### 完成标准

能证明：

```text
same seed + same policy => reproducible rollout
```

并有后续对照方式。

---

## V1 — Snapshot schema，无 GUI

新增：

```text
visualization/config.py
visualization/schema.py
visualization/snapshot.py
```

实现：

```python
build_world_render_snapshot(env)
```

### 完成标准

snapshot 能正确读取：

- time；
- ring actual/target；
- gate；
- LZ；
- active UAV；
- service 状态；
- queue size；
- telemetry；
- canonical committed path。

并且构造 snapshot 前后 env state signature 完全一致。

---

## V2 — 最小 3D viewer

新增：

```text
visualization/pyvista_viewer.py
```

实现：

```text
3 rings
4 gates
6 LZ
UAV points
camera interaction
basic HUD
```

先使用 synthetic snapshot 或固定 rule rollout 测试。

### 完成标准

可以旋转、缩放、平移，并实时看到 UAV 移动。

---

## V3 — Step observer 接入

在：

```text
terminal_marl/envs/parallel_env.py
```

增加最小 observer hook。

viewer adapter 挂到：

```text
env.set_step_observer(...)
```

### 完成标准

普通 `TerminalParallelEnv.step()` 每个 dt 可刷新。

更重要：

```text
DynamicHighLevelWrapper.step()
```

内部每一个 low-level dt 也能刷新，而不是 180 s 跳一次。

---

## V4 — Policy rollout runner

新增：

```text
visualization/rollout_viewer.py
```

支持：

```text
fixed_rule
fixed_low_rl
dynamic_rule
learned_high_rule_low
learned_high_frozen_low
```

至少优先完成：

```text
fixed_rule
fixed_low_rl
learned_high_frozen_low
```

### 完成标准

可以给 checkpoint + seed，直接观察 deterministic rollout。

---

## V5 — Dynamic geometry + trajectory

增加：

- actual ring 动态更新；
- target ring；
- transition progress；
- committed canonical trajectory；
- LZ status；
- occupancy/capacity。

### 完成标准

geometry transition 时画面连续变化，且 UAV 坐标与 ring actual geometry 一致。

---

## V6 — Diagnostic polish

可选：

```text
pause/resume
single step
playback speed
labels
conflict highlight
selected UAV panel
trajectory toggle
```

不要求复杂 Qt 主窗口。

---

# 23. 测试要求

至少新增以下测试。

## 23.1 `test_visualization_snapshot.py`

### `test_snapshot_is_read_only`

对 env 生成 snapshot 前后比较 state signature：

```text
active positions
phase
energy
airspace state
LZ state
telemetry
```

必须完全一致。

### `test_snapshot_uses_actual_dynamic_geometry`

触发/构造 geometry transition 后：

```text
snapshot.rings[*].radius
snapshot.rings[*].height
```

必须来自 actual geometry，不是 initial config。

### `test_snapshot_uses_canonical_plan_points`

有 committed plan 时：

```text
snapshot trajectory_plan_id == uav.trajectory_plan_id
snapshot trajectory points == uav.committed_plan.points
```

### `test_external_queue_is_not_fake_airborne_uav`

external queue 只进入 queue count，不伪造 3D 飞行状态。

---

## 23.2 `test_visualization_integration.py`

### `test_env_runs_without_visualization_dependencies`

在不 import viewer 的路径下：

```python
TerminalParallelEnv(...)
```

正常。

### `test_step_observer_disabled_by_default`

默认没有 observer。

### `test_step_observer_called_once_per_sim_tick`

一次：

```python
env.step(...)
```

observer 只收到一个完成 tick 的状态。

### `test_observer_does_not_change_step_api`

`step()` 返回结构与原实现一致。

### `test_disabled_visualization_is_deterministic_equivalent`

same seed/actions：

```text
no observer
Null/disabled viewer
```

rollout state signature 和结果一致。

### `test_high_level_wrapper_emits_low_step_frames`

对 dynamic wrapper 做一个高层 macro step。

若推进了 `N` 个 base dt，observer 应收到约 `N` 个 low-level frame，而不是 1 个 high-level frame。

这是本任务的关键回归测试。

---

# 24. GUI 测试策略

自动化测试不要依赖真实显示器。

CI / server tests 只测试：

```text
snapshot
observer
disabled behavior
renderer data update functions where possible
```

真实 PyVista window 做：

```text
manual smoke test
```

即可。

不要因为 server 没有 DISPLAY 导致核心 pytest 失败。

---

# 25. 性能验收

## Disabled

目标：

```text
visualization disabled overhead ≈ 0
```

严格来说会多一个：

```python
if self._step_observer is not None:
```

这样的常数级检查可以接受。

但不能：

```text
每步复制所有轨迹
每步构建 snapshot
每步 import renderer
```

## Enabled

当前最大 active UAV 默认 60，需求只包含简单 point/line。

目标不是高 FPS 游戏，而是：

> **足够流畅地诊断交通流。**

建议：

```text
10–30 visual FPS 已足够
```

仿真可以比 realtime 更快或更慢。

若画面落后，优先：

```text
frame_stride
drop stale frame
disable labels
selected trajectory only
```

不要改变 simulation semantics。

---

# 26. Error handling

## 用户关闭窗口

visualization runner 应 graceful stop：

- 关闭 viewer；
- 结束本次 diagnostic rollout；
- 不写坏 checkpoint；
- 不修改训练状态。

## GUI dependency 缺失

只在 `--gui` 时提示缺依赖。

## Headless server

若检测不到合适 display：

```text
gui=False
```

仍正常。

如果用户显式要求 GUI，则清晰报错，不 fallback 到偷偷改变核心运行方式。

---

# 27. 与现有 `runtime_trace` 的关系

当前：

```python
self.runtime_trace = deque(maxlen=60)
```

已经保存少量最近 active state，用于 conflict trace。

不要为了 viewer 把它改成：

```text
完整 episode unlimited history
```

否则长 episode 会增加内存。

实时 viewer 应直接消费 step snapshot。

如果未来需要 replay：

> 单独实现 record/replay 文件格式，不要让 runtime_trace 无限增长。

本任务第一阶段不必实现完整 replay。

---

# 28. 与现有 `terminal_marl/plot/` 的关系

`terminal_marl/plot/` 当前更接近训练 history/统计图绘制。

新 3D viewer 不建议继续塞入：

```text
terminal_marl/plot/
```

推荐独立：

```text
terminal_marl/visualization/
```

原因：

```text
plot = offline analysis
visualization = live simulation observer
```

职责不同。

---

# 29. 不允许出现的反模式

Codex 如果准备采用以下方案，应停止并改设计。

## 29.1 在 `TerminalParallelEnv.step()` 中直接写 PyVista

禁止：

```python
def step(...):
    import pyvista
    plotter...
```

## 29.2 为可视化复制第二个 UAV manager

禁止。

## 29.3 为画 trajectory 再调用 generator

禁止。

## 29.4 用 visualization update 改 simulation dt

禁止。

## 29.5 `gui=False` 仍然每步创建 frame

禁止。

## 29.6 修改 observation 增加 render 字段

禁止。

## 29.7 修改 PPO model 输入

禁止。

## 29.8 修改 reward 让轨迹“更好看”

禁止。

## 29.9 为 viewer 修改 safety threshold

禁止。

## 29.10 在 legacy `src/` 中实现正式版本

禁止。

---

# 30. 推荐交付文件清单

第一轮最终期望：

```text
terminal_marl/visualization/__init__.py
terminal_marl/visualization/config.py
terminal_marl/visualization/schema.py
terminal_marl/visualization/snapshot.py
terminal_marl/visualization/pyvista_viewer.py
terminal_marl/visualization/rollout_viewer.py

tests/test_visualization_snapshot.py
tests/test_visualization_integration.py

requirements-visualization.txt
```

现有文件最多必要修改：

```text
terminal_marl/envs/parallel_env.py
```

如果 Codex 发现必须修改更多正式核心文件，应先重新评估是否是设计侵入过大。

尤其原则上**不应修改**：

```text
terminal_marl/core/trajectory_generator.py
terminal_marl/core/safety_filter.py
terminal_marl/core/merge_scheduler.py
terminal_marl/core/runtime_separation.py
terminal_marl/algo/*
```

---

# 31. 回归测试范围

新增可视化后，除 visualization tests 外，至少运行与当前环境主链路相关的现有 tests。

最低建议：

```bash
python -m pytest \
  tests/test_phase_a_landing.py \
  tests/test_a5_shared_ppo.py \
  tests/test_visualization_snapshot.py \
  tests/test_visualization_integration.py \
  -q
```

如果当前阶段 canonical regression 已有更新后的 A6 tests，继续按 `PROJECT_HANDOFF` 与本地状态运行匹配的安全回归，例如：

```text
A6.12
A6.13
当前最新 safety / scheduler regression
```

不要为了节省时间跳过“same plan checked/executed”相关回归。

---

# 32. 最关键的等价性验证

开发完成后必须写一个 deterministic regression：

```text
seed = S
policy = fixed deterministic policy / frozen checkpoint
episode horizon = H
```

运行 A：

```text
GUI disabled
```

运行 B：

```text
observer attached
GUI rendering关闭或 NullRenderer
```

比较：

```text
ledger outcomes
resolution times
completed uavs
final positions
energy
airspace state
telemetry
conflict events
```

必须一致。

然后人工运行 C：

```text
GUI enabled
```

确认最终结果仍一致。

这比“GUI 能打开”更重要。

---

# 33. 视觉效果的优先级

开发资源优先级：

```text
P0 位置正确
P0 动态 ring 正确
P0 UAV phase 正确
P0 trajectory 与 canonical plan 一致
P0 GUI on/off 不影响仿真

P1 occupancy/capacity
P1 LZ resource state
P1 conflict highlight
P1 pause / speed

P2 更漂亮的 mesh
P2 阴影
P2 城市场景
P2 动画特效
```

任何时候：

> 正确性 > 可解释性 > 性能 > 美观。

---

# 34. 第一版画面建议

一个合格的 MVP 画面可以非常简单：

```text
                         Outer ring @ H_O
                    o  o  o  UAV points

                 Middle ring @ H_M

                  Inner ring @ H_I
                       \
                        \ canonical final path
                         \
                    [ six LZ on ground ]

4 radial entry corridors extending outward
```

右上角文字：

```text
t = 438 s
active = 19
external queue = 7
service = 3
completed = 11

O  8/47
M  6/31
I  3/15

min sep = ...
hard sep = 0
NMAC = 0
collision = 0
```

这已经足以显著提升研究诊断效率。

---

# 35. Codex 实施时的工作纪律

1. 开始前：
   ```bash
   git status --short
   ```

2. 不得丢弃任何用户已有未提交修改。

3. 不得：
   ```bash
   git add .
   ```

4. 只显式暂存本任务文件。

5. 本任务不训练 PPO。

6. 不修改 checkpoint。

7. 不修改 safety threshold。

8. 不修改 canonical action schema。

9. 不修改 observation schema。

10. 不修改 reward。

11. 不修改主实验默认 behavior。

12. GUI 默认关闭。

13. PyVista 为 optional dependency。

14. 所有 core regression 通过后再提交。

15. 最终报告必须列出：
    - 新增文件；
    - 修改文件；
    - 为什么需要修改；
    - tests；
    - GUI smoke test；
    - disabled-equivalence test；
    - commit hash；
    - 是否 push（若无 remote 不得声称 push）。

---

# 36. 本任务的最终验收定义

只有同时满足以下条件，才算完成。

## 功能

- [ ] 可以打开实时 3D terminal viewer；
- [ ] 可以鼠标旋转；
- [ ] 可以平移；
- [ ] 可以缩放；
- [ ] 可以看到三个 actual ring；
- [ ] 可以看到 4 gate/corridor；
- [ ] 可以看到 6 LZ；
- [ ] 可以看到 active UAV；
- [ ] UAV 位置每个渲染 tick 更新；
- [ ] dynamic geometry 时 ring 会更新；
- [ ] 可以观察 deterministic RL checkpoint rollout；
- [ ] high-level macro step 内仍能看到 low-level 连续运动；
- [ ] committed trajectory 使用 canonical plan。

## 解耦

- [ ] `gui=False` 默认；
- [ ] core 不 import PyVista；
- [ ] 无 GUI dependency 时训练仍正常；
- [ ] viewer 不修改 env；
- [ ] viewer 不重新计算 physics/trajectory/safety；
- [ ] legacy `src/` 未成为新依赖。

## 回归

- [ ] existing core tests 通过；
- [ ] visualization snapshot tests 通过；
- [ ] observer integration tests 通过；
- [ ] GUI disabled 与原始 deterministic rollout 结果一致；
- [ ] GUI enabled 的最终 deterministic result 不发生语义变化。

## 性能

- [ ] disabled 几乎无额外开销；
- [ ] enabled 能流畅观察几十架 UAV；
- [ ] 不出现无限 frame/history 内存增长。

---

# 37. Codex 的推荐执行顺序

严格按下面顺序推进，不要一开始直接写 GUI：

```text
1. Read canonical docs + git status
2. Freeze deterministic baseline
3. Add immutable RenderSnapshot schema
4. Add SnapshotBuilder
5. Test snapshot has zero mutation
6. Add optional step observer to TerminalParallelEnv
7. Test observer does not alter step semantics
8. Build PyVista static scene
9. Add real-time UAV point updates
10. Add dynamic ring updates
11. Add gate + LZ state
12. Add canonical trajectory lines
13. Add HUD
14. Add rollout CLI
15. Connect existing checkpoint loaders
16. Verify DynamicHighLevelWrapper inner dt visualization
17. Run deterministic disabled-equivalence regression
18. Run existing Phase-A/A5/A6 safety regressions
19. Manual GUI smoke test
20. Explicit git diff / commit only task files
```

---

# 38. 最后原则

这个 viewer 的价值不是“做一个看起来像游戏的仿真界面”。

它的价值是：

> 当策略指标异常时，可以马上看到物理交通过程发生了什么；  
> 当 reward 或 policy 改动后，可以判断策略究竟学到了什么行为；  
> 当 safety/scheduler 阻塞时，可以看到堵塞是在 Outer、Middle、Inner、merge 还是 final；  
> 当 dynamic airspace 调节时，可以看到空域变化与交通响应是否符合研究假设。

所以请始终把它实现成：

```text
small
optional
read-only
diagnostic
canonical-state-driven
```

而不是新的核心子系统。

**如果某个设计需要大规模修改现有仿真核心才能完成可视化，优先认为该设计方向错了，而不是继续扩大改动。**
