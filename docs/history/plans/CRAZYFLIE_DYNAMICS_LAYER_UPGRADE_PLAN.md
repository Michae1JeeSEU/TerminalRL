# EH216-S Vehicle Dynamics Layer Upgrade & Crazyflie Zero-Shot Hardware Transfer Plan

> **文档定位**  
> 本文件是当前低空终端区 MARL 项目在启动 A5-v2 正式低层训练之前必须完成的 **Vehicle Dynamics / Performance Layer 升级纲领**。  
> 主研究对象是**真实尺度 vertiport 场景中的指定 eVTOL 型号**；当前阶段要把较理想化的几何质点执行升级为该 eVTOL 的 **performance-constrained 3D point-mass / kinematic vehicle model**。Crazyflie 仅作为后续受实验条件限制的缩比物理验证平台，不参与主训练动力学参数的确定。研究重点仍然是 terminal traffic management / flight-intention decision，而不是姿态控制、桨叶气动或完整六自由度飞控。
>
> **建议文档名**：`docs/CRAZYFLIE_DYNAMICS_LAYER_UPGRADE_PLAN.md`  
> **建议 schema version**：`vehicle_dynamics_v4_eh216s`  
> **执行优先级**：必须在 `C0 capacity calibration` 与 A5-v2 正式低层 PPO 训练之前完成并通过资格测试。
>
> **Codex 重要要求**  
> Codex 开始执行前必须首先审计当前仓库的最新代码和 `docs/`，不得机械假定本文件中提到的历史文件内容仍与当前实现完全一致。每完成一个 implementation phase，必须**立即同步更新相关 `docs/*.md`**，不能等所有代码完成后一次性补文档。

---

# 1. 为什么现在必须升级动力学层

当前正式低层训练之前必须先补齐 vehicle dynamics，原因不是为了“模拟 Crazyflie”，而是为了让**指定 eVTOL 型号**的主研究仿真满足基本可执行性：

1. policy 输出的飞行意图不能导致物理上瞬时不可实现的速度变化；
2. ring / transfer / final trajectory 必须满足该 eVTOL 的有限速度、加速度、横向加速度、爬升/下降等性能边界；
3. SafetyFilter 检查的轨迹必须是真正能够被 vehicle execution layer 执行的轨迹；
4. scheduler 的 merge-time feasibility 必须建立在动力学可达性上，而不是几何可达性上；
5. simulation 与未来 hardware validation 共享同一套 `TrajectoryPlan` / normalized policy interface / execution abstraction；
6. Crazyflie 只在后续 hardware branch 中承担缩比 position/velocity/trajectory tracking，不定义主仿真的车辆参数；
7. 低层策略仍然是 hierarchy 中的 traffic-intention layer，而不是 aircraft attitude controller。

当前代码历史版本已经包含类似：

```text
max_speed
max_acceleration
max_turn_rate_deg_s
max_climb_rate
max_descent_rate
```

但历史实现主要只对 speed / vertical rate 做部分检查，ring holder、transfer 和 final 仍存在直接按几何位置推进、瞬时速度变化或折线路径的问题。

因此，本次升级的重点不是“增加几个配置参数”，而是让**固定的研究 eVTOL 性能约束**真正进入：

```text
trajectory planning
→ time parameterization
→ safety prediction
→ execution
→ telemetry
```

完整链路。

> **关键边界**：当前阶段不进行 eVTOL 型号/性能参数 domain randomization。研究使用一个明确指定、参数冻结的 eVTOL 型号；车辆内部泛化不是本研究当前目标。

# 2. 研究建模边界

## 2.1 正式模型名称

论文和代码建议统一称为：

> **Performance-Constrained 3D Point-Mass Vehicle Model**

或者：

> **Performance-Constrained Kinematic Vehicle Execution Layer**

不要把它描述成完整 eVTOL / quadrotor 6-DOF dynamics。

---

## 2.2 本阶段明确实现

必须实现：

- 3D position；
- 3D velocity；
- horizontal speed limit；
- climb/descent speed limit；
- finite horizontal acceleration；
- finite vertical acceleration（推荐）；
- path curvature / lateral-acceleration feasibility；
- independent yaw / yaw-rate limit（如果 yaw 被用于轨迹输出）；
- smooth trajectory time parameterization；
- ring motion 的 centripetal acceleration 检查；
- checked trajectory 与 executed trajectory 一致；
- future Crazyflie-compatible trajectory/export abstraction；
- simulation/hardware scaling abstraction；
- normalized policy observation/action interface needed for later zero-shot scaling。

---

## 2.3 本阶段明确不实现

暂不实现：

- rotor RPM；
- individual motor thrust；
- propeller aerodynamics；
- full rigid-body attitude dynamics；
- roll/pitch body-rate inner-loop simulation；
- motor electrical model；
- blade flapping；
- detailed battery voltage sag；
- full CFD / wind field；
- PX4-level flight-control simulation；
- eVTOL vehicle-family parameter randomization；
- eVTOL performance-domain randomization；
- Crazyflie-specific dynamics calibration；
- hardware-specific RL fine-tuning。

这些要么属于 flight-control / high-fidelity validation，要么属于后续 hardware validation，不是当前 MARL 训练前置。

# 3. 主研究机型正式冻结：EHang EH216-S

## 3.1 机型选择结论

本研究从本版本起正式将主研究 eVTOL 型号冻结为：

> **EHang EH216-S passenger-carrying pilotless multirotor eVTOL**

代码与论文统一使用：

```text
profile_id = research_evtol
vehicle_model_name = EHang EH216-S
```

选择 EH216-S 的原因：

1. 属于面向城市空中交通（UAM）的载人 eVTOL；
2. 为纯多旋翼 / lift-only 构型，与当前 terminal ring / hover-capable trajectory abstraction 一致；
3. 已取得中国民航局型号合格证，机型与部分运行限制有公开适航依据；
4. 公开最大设计速度与当前代码历史 `36 m/s` 上限量级接近，但正式运行速度限制能够帮助纠正当前过于激进的 nominal speed；
5. 与后续 Crazyflie 缩比验证在“多旋翼、可悬停、通过平动加速度跟踪三维轨迹”的抽象层上更一致；
6. 不需要引入 tilt-wing / tilt-rotor transition mode，避免把研究问题扩展为不同推进构型与转换飞行控制。

本研究**不声称 EH216-S 代表全部 eVTOL 构型**。论文应表述为：

> EH216-S is selected as a representative short-range urban multirotor eVTOL for the terminal-area arrival and vertiport landing study.

本研究当前也不做跨 eVTOL 型号泛化；正式训练期间 EH216-S profile 保持固定。

---

## 3.2 参数来源分级

所有参数必须标记来源等级，禁止把工程建模值写成“厂商/适航认证值”。

### A. 官方 / 适航公开值

来自 EHang 官方产品页面、CAAC EH216-S 适航/型号资料：

| 参数 | 冻结值 | 用途 |
|---|---:|---|
| 最大起飞重量 MTOW | `620 kg` | metadata / energy sanity |
| 最大设计速度 | `130 km/h = 36.11 m/s` | metadata，**不是正常 executor 上限** |
| 不可超越速度 V_NE | `105 km/h = 29.17 m/s` | hard invalid / emergency envelope |
| 最大平飞速度 | `90 km/h = 25.00 m/s` | **主仿真 operational horizontal speed upper bound** |
| 最大航程 | `30 km` | energy/range sanity check |
| 最大航时 | `25 min` | individual-flight energy sanity check |
| 最大运行高度 | `120 m AGL` | terminal vertical geometry hard bound |
| 最大运行高度 | `1000 m MSL` | metadata，需结合场址海拔 |
| 落地最低电量 | `25%` | normal-operation landing reserve |
| 乘员座位数 | `2` | metadata |
| 机体宽度 | 约 `5.7 m` | metadata / physical-body sanity |
| 动力构型 | 8 对升力/推力部件（16 个旋翼） | model justification |

### B. 工程建模值

下列字段并非 CAAC TCDS 明确给出的 EH216-S certified limit，而是为了 performance-constrained point-mass model 冻结的研究建模参数：

| 参数 | 冻结值 | 说明 |
|---|---:|---|
| horizontal operational acceleration | `1.5 m/s²` | 正式 executor 使用 |
| horizontal operational deceleration | `1.5 m/s²` | 正式 executor 使用 |
| hard lateral acceleration | `1.73 m/s²` | 约等于 `g*tan(10°)`，只作为工程 hard envelope |
| operational lateral acceleration | `1.5 m/s²` | curvature / ring feasibility 使用 |
| vertical acceleration up | `1.0 m/s²` | 平滑 terminal trajectory |
| vertical acceleration down | `1.0 m/s²` | 平滑 terminal trajectory |
| independent yaw-rate limit | `12 deg/s` | 保留历史值，但只限制 yaw，不作为主要平移转弯能力 |
| jerk hard state | 暂不启用 | 通过 trajectory smoothness/time parameterization 控制 |

### C. 暂定运行值（需由本项目统一采用，不代表厂家限制）

| 参数 | 冻结值 | 说明 |
|---|---:|---|
| max climb rate | `4.0 m/s` | terminal operational model |
| max descent rate | `6.0 m/s` | terminal operational model |
| touchdown vertical speed | 由 landing profile 单独定义，必须显著低于 `6 m/s` | 不得用最大下降率直接触地 |

> `4/6 m/s` 当前按项目运行性能值使用，必须在 `docs/vehicle_dynamics_spec.md` 中明确标记为 **research operational values**，不能写成“CAAC certified climb/descent limit”，除非后续找到可直接核验的 EH216-S 飞行手册/批准数据。

---

## 3.3 正式 `research_evtol.yaml`

Codex 应将唯一正式训练 profile 建议实现为：

```yaml
profile_id: research_evtol
vehicle_model_name: EHang EH216-S
profile_version: eh216s_research_v1

metadata:
  configuration: multirotor_lift_only
  mtow_kg: 620.0
  seats: 2
  reference_range_m: 30000.0
  reference_max_flight_time_s: 1500.0
  max_operating_altitude_agl_m: 120.0
  max_operating_altitude_msl_m: 1000.0

speed:
  max_level_mps: 25.0
  never_exceed_mps: 29.17
  design_mps: 36.11

acceleration:
  max_horizontal_accel_mps2: 1.5
  max_horizontal_decel_mps2: 1.5
  max_lateral_operational_mps2: 1.5
  max_lateral_hard_mps2: 1.73
  max_vertical_accel_up_mps2: 1.0
  max_vertical_accel_down_mps2: 1.0

vertical:
  max_climb_mps: 4.0
  max_descent_mps: 6.0

orientation:
  max_yaw_rate_deg_s: 12.0

battery:
  min_landing_soc: 0.25

trajectory:
  jerk_xy_limit: null
  jerk_z_limit: null
```

注意：

- `36.11 m/s` 只作为 **maximum design speed metadata**；
- `29.17 m/s` 是不可超越边界；
- 正常 trajectory/executor 绝不能把 `36.11 m/s` 当成 operational max；
- 正常 horizontal speed 上限使用 `25.0 m/s`；
- ring/transfer/final 的实际 phase speed 还必须同时受 curvature、vertical coupling、reservation timing 等限制。

---

## 3.4 现有代码参数必须同步修改

根据当前项目历史 canonical config，本次动力学升级必须同时迁移以下参数。

### 3.4.1 Vehicle performance

| 当前历史值 | 新值 | 动作 |
|---|---:|---|
| `max_speed = 36 m/s` | `max_level_speed = 25.0 m/s` | **修改**；另保留 `design_speed=36.11`、`never_exceed=29.17` metadata/hard limit |
| `max_acceleration = 4.0 m/s²` | `max_horizontal_accel = 1.5 m/s²` | **修改** |
| 历史未明确 decel | `max_horizontal_decel = 1.5 m/s²` | **新增** |
| `max_turn_rate_deg_s = 12` | `max_yaw_rate_deg_s = 12` | 数值暂保留，但**语义修改**：不再等价于 path-turn feasibility |
| `max_climb_rate = 5 m/s` | `4.0 m/s` | **修改** |
| `max_descent_rate = 7 m/s` | `6.0 m/s` | **修改** |
| 无 lateral accel hard limit | operational `1.5`, hard `1.73 m/s²` | **新增** |
| 无 vertical accel limit | up/down `1.0 m/s²` | **新增** |

路径主要使用：

```text
v^2 * kappa <= max_lateral_operational_mps2
```

而不是单纯：

```text
heading_rate <= 12 deg/s
```

判断多旋翼能否跟随轨迹。

---

### 3.4.2 三环 nominal speed

当前历史值：

```text
Outer  = 28 m/s
Middle = 24 m/s
Inner  = 20 m/s
```

正式修改为：

```text
Outer  = 22 m/s
Middle = 20 m/s
Inner  = 16 m/s
```

理由：

1. `28 m/s` 已高于 EH216-S 最大平飞速度 `25 m/s`；
2. 需要给 low-level `speed_residual` 留出动态调整余量；
3. 原 Inner `20 m/s @ r=200 m` 的向心加速度为 `2.0 m/s²`，高于本研究 operational lateral envelope；
4. 新 nominal 值对应：

```text
Outer:  22² / 600 = 0.807 m/s²
Middle: 20² / 400 = 1.000 m/s²
Inner:  16² / 200 = 1.280 m/s²
```

均低于 `1.5 m/s²` operational lateral limit。

---

### 3.4.3 三环与入口高度

当前历史 geometry：

```text
entry_gate_height = 600 m
Outer  = 500 m
Middle = 350 m
Inner  = 200 m
```

这与 EH216-S：

```text
max operating altitude = 120 m AGL
```

不兼容，必须修改。

本研究从本版本起建议 nominal vertical geometry 冻结为：

```text
entry_gate_height = 110 m AGL
Outer  = 100 m AGL
Middle = 70 m AGL
Inner  = 40 m AGL
LZ     = 0 m AGL reference
```

原则：

- 所有正常 terminal trajectory `z_AGL <= 120 m`；
- 预留至少约 10 m 高度 margin，不把 nominal traffic 放在认证高度边界上；
- 保留 Outer→Middle→Inner 的单调下降结构；
- final 从 Inner 40 m AGL 进入受控进近和着陆；
- terrain/elevation 若以后引入，必须明确区分 AGL 与 MSL，不能把仿真 z 直接当 MSL。

Horizontal geometry 暂保留：

```text
Outer radius  = 600 m
Middle radius = 400 m
Inner radius  = 200 m
approach_threshold_radius = 1000 m
spawn_radius = 1300 m
```

这些值本次不因 EH216-S 选型自动改变，但动力学升级后必须重新做 C0 capacity calibration。

---

### 3.4.4 SOC 阈值

当前历史：

```text
soc_warning   = 0.30
soc_reserve   = 0.15
soc_emergency = 0.08
```

EH216-S TCDS 给出的落地最低电量为：

```text
min_landing_soc = 0.25
```

因此正式修改为：

```text
soc_warning   = 0.35
soc_reserve   = 0.25
soc_emergency = 0.20
```

语义：

- `>=0.35`：正常；
- `[0.25, 0.35)`：warning / urgency 增强；
- `[0.20, 0.25)`：已经低于正常 landing reserve，属于异常/应急处置区；
- `<0.20`：emergency critical。

特别注意：

> `soc_emergency = 0.20` 是研究建模阈值，不代表 EH216-S 允许正常运行到 20%。

正常 arrival generation 不应简单采样到 `SOC=0.25` 后还要求完成完整终端流程。正常初始 SOC 下界应满足：

```text
initial_SOC >= min_landing_soc + predicted_energy_to_land + margin
```

在 energy model 尚未重新标定前，建议普通训练场景的 provisional 初始下界不得低于 `0.35`。低于 `0.25` 只用于明确标记的 emergency/OOD witness，不进入普通训练主体。

Codex 必须同步检查并更新 `A5V2_LOW_LEVEL_TRAINING_CURRICULUM*.md` 中所有旧的：

```text
U(0.15, 1.0)
soc_reserve = 0.15
```

等设定，避免动力学 profile 与训练 curriculum 冲突。

---

### 3.4.5 暂时保持不变、但 C0 后重新验证的参数

以下参数与 EH216-S 选型没有直接证据要求立即修改，因此当前保持：

```text
safe_distance = 80 m
NMAC = 40 m
collision = 20 m

O→M headway = 9 s
M→I headway = 9 s
I→F headway = 12 s

LZ count = 6
dt_policy = 1 s
max_active = 60
```

但是：

- nominal speed 降低；
- vertical geometry 大幅降低；
- acceleration envelope 收紧；
- transfer/final duration 变为 dynamics-aware；

都会改变 capacity，因此这些 headway 与 traffic-rate 定义必须在 DYN qualification 后通过 C0 重新确认。

---

## 3.5 代码迁移原则：旧字段不得继续成为第二套事实源

Codex 必须搜索所有：

```text
36.0
28.0
24.0
20.0
500.0
350.0
200.0
600.0
4.0   # old acceleration context
5.0   # old climb context
7.0   # old descent context
0.15  # SOC reserve context
0.08  # SOC emergency context
```

但**不能机械全局替换数字**。必须按字段语义定位：

- config；
- defaults；
- tests；
- warm-start；
- trajectory generator；
- runtime holder；
- observation normalization；
- scenario YAML；
- visualization；
- docs；
- regression expected values。

`VehiclePerformanceProfile` 与 canonical terminal geometry config 必须成为唯一事实源。

---

## 3.6 主仿真与 Crazyflie 缩比验证的关系

Crazyflie 实机验证是后续缩比实验，因此禁止：

```text
Crazyflie measured max speed / acceleration
→ overwrite EH216-S research profile
→ retrain main policy
```

也禁止直接组合：

```text
full-scale EH216-S terminal geometry
+ Crazyflie physical performance
```

正确关系：

```text
fixed EH216-S research profile
→ full-scale simulation + policy training
→ freeze policy weights
→ later Crazyflie calibration
→ choose scale / execution / safety mapping
→ zero-shot small-N hardware validation
```

---

## 3.7 Profile 目录建议

当前阶段：

```text
vehicle_profiles/
    research_evtol.yaml       # EHang EH216-S，唯一正式训练 profile
```

后续 hardware branch 再新增：

```text
hardware_profiles/
    crazyflie_calibrated.yaml
```

不要在当前训练阶段启用 Crazyflie profile。

---

## 3.8 必须预留 Scale Configuration，但当前不需要 Crazyflie 数值

建议定义：

```python
@dataclass(frozen=True)
class ScaleConfig:
    length_scale: float
    time_scale: float
```

派生：

```text
velocity_scale     = length_scale / time_scale
acceleration_scale = length_scale / time_scale^2
jerk_scale         = length_scale / time_scale^3
```

注意：

> 这只是 terminal traffic / trajectory 的 kinematic scaling，不应声称实现严格 aerodynamic dynamic similarity。

当前 dynamics upgrade 只要求 API / schema 能支持该映射；**不要求现在确定 Crazyflie 的 `length_scale` 或 `time_scale`**。

---

## 3.9 Hardware Safety Distance 属于后续硬件阶段

实机最小安全距离未来应考虑：

```text
physical body size
+ tracking error
+ localization error
+ communication/control latency
+ relative closing-speed buffer
+ engineering margin
```

这些参数：

- 不用于当前 EH216-S 主仿真安全距离定义；
- 不用于当前低层训练；
- 不反向修改 EH216-S profile。

因此禁止用：

```text
80 m * length_scale
```

直接声称得到实机安全距离。

---

## 3.10 本节参数来源

优先事实源：

1. EHang EH216-S official product page  
   `https://www.ehang.com/ehang216s/`
2. CAAC EH216-S Special Conditions, SC-21-002  
   `https://www.caac.gov.cn/XXGK/XXGK/BZGF/ZYTJHHM/202202/t20220222_211914.html`
3. EH216-S Type Certificate Data Sheet, Revision 0  
   代码/docs 中引用时应优先保存并核验正式 TCDS；当前可核验公开文本包含：
   - 252 Ah battery；
   - 66.0–95.7 V；
   - landing SOC ≥ 25%；
   - design speed 130 km/h；
   - never-exceed 105 km/h；
   - max level speed 90 km/h；
   - MTOW 620 kg；
   - max operating altitude 120 m AGL / 1000 m MSL。
4. 本文件中 `1.5 m/s²` horizontal/lateral acceleration、`1.0 m/s²` vertical acceleration、`12 deg/s` yaw rate、三环 nominal speed 与三环高度均属于**研究工程建模值**，必须在论文中与官方/适航值分开标注。

# 4. Crazyflie 接口依据

本项目不需要让 RL 直接输出电机或姿态指令。

Crazyflie 官方软件栈已经支持：

- position setpoint；
- velocity setpoint；
- full-state setpoint（position / velocity / acceleration / orientation / angular rates）；
- high-level commander；
- smooth polynomial trajectories；
- onboard position/velocity controllers。

因此本项目最自然的 hardware boundary 是：

```text
MARL traffic intention
↓
TrajectoryPlan
↓
performance-constrained executable trajectory
↓
hardware adapter
↓
Crazyflie position/full-state/high-level trajectory setpoints
↓
onboard controller
```

而不是：

```text
MARL
↓
roll/pitch/thrust
```

---

# 5. 关键建模决策：四旋翼平动与 yaw 分离

不要采用固定翼式：

```text
velocity direction == yaw
```

作为硬假设。

Crazyflie 可以通过 roll/pitch 倾斜产生任意水平横向加速度，平移方向和 yaw 可以独立。

因此核心动态约束应该放在：

```text
horizontal velocity
horizontal acceleration vector
vertical velocity
vertical acceleration
trajectory curvature / lateral acceleration
```

而 yaw 作为独立状态：

```text
yaw
max_yaw_rate
```

如果硬件执行选择“机头始终沿轨迹切线”，再额外把 tangent-yaw 作为 trajectory mode，而不是底层物理硬约束。

---

# 6. 建议动力学状态

## 6.1 UAVState 扩展

建议至少保证 `UAVState` 中存在：

```python
position: np.ndarray      # [x, y, z]
velocity: np.ndarray      # [vx, vy, vz]
speed: float              # horizontal or total speed，必须定义清楚
heading/yaw: float
```

建议新增：

```python
acceleration: np.ndarray  # [ax, ay, az]
yaw_rate: float
```

可选诊断状态：

```python
reference_velocity
reference_acceleration
tracking_error
vehicle_profile_id
```

---

## 6.2 speed 语义必须统一

禁止代码中有时把 `speed` 表示 3D norm、有时表示 horizontal speed。

建议：

```text
horizontal_speed = sqrt(vx^2 + vy^2)
vertical_speed   = vz
total_speed      = ||v||
```

`UAVState.speed` 建议明确为：

> **horizontal air/ground speed used by terminal trajectory logic**

如果使用总速度，必须统一修改所有 scheduler / observation / reward 使用点。

---

# 7. VehiclePerformanceProfile

建议新增：

```python
@dataclass(frozen=True)
class VehiclePerformanceProfile:
    profile_id: str
    vehicle_model_name: str
    profile_version: str
    parameter_source: str

    # Metadata / certified-reference fields
    mtow_kg: float | None
    max_operating_altitude_agl_m: float | None
    max_operating_altitude_msl_m: float | None
    reference_range_m: float | None
    reference_max_flight_time_s: float | None
    min_landing_soc: float | None

    # Speed envelopes
    max_horizontal_speed: float          # EH216-S operational max level speed = 25.0
    never_exceed_speed: float | None     # 29.17
    design_speed: float | None           # 36.11, metadata only
    min_horizontal_speed: float

    # Translational performance
    max_horizontal_accel: float          # 1.5
    max_horizontal_decel: float          # 1.5
    max_climb_rate: float                # 4.0
    max_descent_rate: float              # 6.0
    max_vertical_accel_up: float         # 1.0
    max_vertical_accel_down: float       # 1.0
    max_lateral_accel: float             # operational = 1.5
    max_lateral_accel_hard: float | None # 1.73

    # Orientation only
    max_yaw_rate_deg_s: float            # 12.0

    # Optional trajectory smoothness constraints
    max_jerk_xy: float | None = None
    max_jerk_z: float | None = None
    max_yaw_accel_deg_s2: float | None = None
```

所有旧的：

```text
max_speed
max_acceleration
max_turn_rate_deg_s
max_climb_rate
max_descent_rate
```

应迁移或映射到这个 profile，不再散落在不同模块解释。

正式训练固定加载：

```text
profile_id = research_evtol
vehicle_model_name = EHang EH216-S
profile_version = eh216s_research_v1
```

并禁止 training config 随机修改车辆性能字段。

关键语义：

```text
max_horizontal_speed = 25.0 m/s
never_exceed_speed    = 29.17 m/s
design_speed           = 36.11 m/s
```

其中 `design_speed` 不得被 executor 当作正常速度上限。

`tracking_error_bound`、`command_latency_seconds` 等 hardware-only 字段不属于 EH216-S 主训练 profile，由未来 Crazyflie hardware adapter/profile 单独管理。

# 8. 基础平动模型

研究层模型使用：

```text
p_dot = v
v_dot = a
```

其中：

```text
||v_xy|| <= v_xy_max
-v_descent_max <= vz <= v_climb_max
||a_xy|| <= a_xy_max
```

垂向：

```text
-a_z_down_max <= az <= a_z_up_max
```

如区分加速与制动：

```text
longitudinal acceleration <= a_accel_max
longitudinal deceleration <= a_decel_max
```

---

# 9. 为什么不建议现在加入完整 jerk-state MDP

Crazyflie 实机轨迹需要平滑 acceleration / jerk，但如果把 jerk 作为真正动态状态，会扩大低层 Markov state 和 observation。

第一版推荐：

1. **vehicle executor 以有限 acceleration 为核心状态转移；**
2. **committed trajectory 在 time parameterization 阶段保证 C2，推荐进一步保证 jerk 不超过可配置上限；**
3. jerk 作为 trajectory feasibility constraint，而不是必须成为 RL observation state。

这样既提高实机可执行性，又不必立即再次扩大 A5-v2 observation。

---

# 10. 轨迹数据结构升级

历史 `TrajectoryPoint` 只有：

```text
t
position
velocity(optional)
yaw(optional)
```

建议升级：

```python
@dataclass
class TrajectoryPoint:
    t: float
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    yaw: float | None = None
    yaw_rate: float | None = None
```

可选：

```python
jerk: np.ndarray | None = None
```

---

# 11. 明确区分三种 trajectory

必须建立：

## 11.1 Geometric Path

只有空间形状：

```text
p(s), s in [0,1]
```

例如：
- circle arc；
- cubic Bezier；
- final smooth spline。

它**不能直接交给 SafetyFilter 执行安全确认**。

---

## 11.2 Executable Time-Parameterized Trajectory

定义：

```text
p(t)
v(t)
a(t)
yaw(t)
```

并满足 vehicle profile。

SafetyFilter 检查的必须是这一层。

---

## 11.3 Actual Executed State

主训练 deterministic mode 中，允许假设：

> 只要 executable trajectory 已通过性能约束，onboard controller 能理想跟踪该 reference trajectory。

因此：

```text
actual state = sampled executable trajectory state
```

这仍然是 performance-constrained point-mass，而不是无约束 teleport。

以后 robustness / hardware validation 再加入：

```text
tracking lag
tracking noise
state-estimation error
latency
```

---

# 12. 新增模块建议

建议文件：

```text
terminal_marl/core/vehicle_dynamics.py
terminal_marl/core/trajectory_time_parameterization.py
terminal_marl/core/vehicle_profiles.py
```

如果当前仓库已有同类模块，Codex 应复用/重构，禁止重复建立第二套 backend。

---

# 13. VehicleDynamics API

建议：

```python
class VehicleDynamics:
    def validate_state(self, state, profile) -> list[str]: ...

    def validate_transition(
        self,
        previous: TrajectoryPoint,
        current: TrajectoryPoint,
        profile,
    ) -> list[str]: ...

    def required_acceleration(...): ...

    def max_feasible_speed_for_curvature(
        self,
        curvature: float,
        profile,
        tangent_yaw: bool = False,
    ) -> float: ...
```

不要让 SafetyFilter、TrajectoryGenerator、ParallelEnv 各自重新实现同一套公式。

---

# 14. TimeParameterizer API

建议：

```python
class TrajectoryTimeParameterizer:
    def parameterize(
        self,
        path: GeometricPath,
        initial_state: UAVState,
        target_speed: float,
        profile: VehiclePerformanceProfile,
        constraints: TrajectoryConstraints,
    ) -> TrajectoryPlan: ...
```

核心职责：

- 依据曲率生成 local speed limit；
- forward acceleration pass；
- backward deceleration pass；
- climb/descent rate constraint；
- vertical acceleration constraint；
- optional jerk smoothing；
- 输出 `position/velocity/acceleration` 都完整的 TrajectoryPlan。

---

# 15. Path Curvature 与横向加速度

对平面路径 `p(s)` 计算曲率：

```text
kappa(s)
```

横向加速度约束：

```text
a_lat = v^2 * kappa
```

必须满足：

```text
a_lat <= a_lat_max
```

因此曲线局部速度上限：

```text
v_curve_max = sqrt(a_lat_max / max(kappa, eps))
```

如果采用 tangent-yaw：

```text
yaw_rate = v * kappa
```

额外：

```text
v_yaw_max = yaw_rate_max / max(kappa, eps)
```

最终：

```text
v_max(s) = min(
    profile.max_horizontal_speed,
    v_curve_max,
    v_yaw_max if tangent_yaw else inf,
    vertical_constraint_speed
)
```

---

# 16. Forward / Backward Speed Profile

沿离散路径弧长 `s_i`：

forward：

```text
v[i+1]^2 <= v[i]^2 + 2 * a_accel * ds
```

backward：

```text
v[i]^2 <= v[i+1]^2 + 2 * a_decel * ds
```

然后根据：

```text
dt_i = 2 * ds / (v_i + v_{i+1})
```

形成时间戳。

对于很小速度必须使用数值保护。

---

# 17. 垂向约束

必须分别检查：

```text
vz <= max_climb_rate
-vz <= max_descent_rate
```

推荐同时约束：

```text
az <= max_vertical_accel_up
-az <= max_vertical_accel_down
```

transfer 中原有 height smoothstep 可以保留思想，但必须与统一 time-parameterization 联动，而不是只用：

```text
height_difference / max_climb_rate
```

估算 duration。

---

# 18. Ring Hold 动力学升级

历史 ring holder 的核心问题是：

```text
speed = nominal + residual
```

后下一步直接按照该 speed 改变 angular position。

新逻辑必须是：

## 18.1 Target tangential speed

```text
v_ref = nominal_speed + speed_profile_bias + speed_residual
```

先 clip 到 profile speed envelope。

---

## 18.2 Finite tangential acceleration

```text
dv = clip(
    v_ref - v_current,
    -a_decel * dt,
    +a_accel * dt,
)

v_next = v_current + dv
```

---

## 18.3 Centripetal acceleration

对于半径 `r`：

```text
a_c = v^2 / r
```

即使 tangential speed 不变，飞圆环也需要横向加速度。

必须满足：

```text
sqrt(a_tangent^2 + a_c^2) <= a_xy_max
```

如果不满足：

```text
reduce feasible ring speed
```

而不是继续执行物理不可行速度。

---

## 18.4 Ring Speed Feasibility

对每个 ring/profile 计算：

```text
v_ring_max_dynamic = sqrt(a_lat_max * r)
```

最终 nominal speed 必须：

```text
nominal_speed <= min(v_xy_max, v_ring_max_dynamic)
```

这项检查在 config load / geometry update 时都必须执行。

---

# 19. Ring Geometry 变化

Reconfiguration 的系统级 freeze/gating 规则由单独 reconfiguration spec 管理，本文件不重新定义高层策略。

但 vehicle dynamics layer 必须提供统一能力，使 ReconfigurationCoordinator 能检查：

- radial velocity；
- vertical velocity；
- radial / vertical acceleration；
- combined horizontal acceleration；
- safe transition duration。

禁止 reconfiguration 继续使用“直接把 UAV position 投影到新 ring”而完全绕过 vehicle performance。

---

# 20. Inbound Path 升级

历史 inbound 是：

```text
current position
→ outer entry point
```

线性插值，并按 distance / fixed speed 设 duration。

新方案：

1. 建立 geometric inbound segment；
2. 起始速度取 UAV 当前真实 velocity；
3. 目标 terminal capture velocity 与 Outer ring tangent 兼容；
4. 使用 acceleration-limited time parameterization；
5. 垂向 descent/climb 同时满足限制；
6. 在 capture 前避免瞬时 heading / velocity vector 跳变。

---

# 21. Ring Transfer 升级

当前 cubic Bezier 路径思想可以保留，但必须升级为：

```text
Bezier geometry
↓
analytic / numeric derivatives
↓
curvature profile
↓
performance-constrained time parameterization
↓
Executable TrajectoryPlan
```

---

## 21.1 Bezier 必须计算

至少：

```text
p(s)
p'(s)
p''(s)
curvature(s)
```

---

## 21.2 Duration 不再只由 average speed 决定

禁止继续只用：

```text
curve_length / commanded_speed
```

作为主要 duration。

必须由实际 speed profile 积分得到：

```text
T = sum(dt_i)
```

---

## 21.3 Merge reservation 与 dynamics 耦合

Scheduler 提出的 merge time 必须满足：

```text
T_min_dynamic <= available_time
```

如果 reservation 太早，飞行器物理上无法达到：

```text
reject / choose later slot
```

不能生成一条超加速度轨迹硬赶上 merge time。

---

# 22. Final Approach 必须取消瞬时折线转向

历史 final：

```text
current
→ entry
→ stabilized point
→ LZ
```

如果只是分段直线线性插值，会在 waypoint 形成瞬时 velocity direction change。

正式方案必须改为 smooth path。

可选实现优先级：

### 方案 A（推荐）

piecewise quintic / 7th-order polynomial，至少满足：

```text
C2 continuity
(position, velocity, acceleration continuous)
```

### 方案 B

chained cubic Bezier，并显式保证 control-point tangent / acceleration continuity。

不允许 waypoint 处瞬时转弯。

---

# 23. Landing Phase

landing trajectory 至少满足：

- finite horizontal speed at stabilization；
- controlled vertical descent；
- touchdown 前水平速度限制；
- touchdown 垂向速度限制；
- no instantaneous stop in air；
- touchdown 后才进入 SERVICE。

建议新增：

```text
max_touchdown_horizontal_speed
max_touchdown_vertical_speed
```

主仿真中的 touchdown limits 必须来自固定的 `research_evtol` profile / 该 eVTOL 型号的性能设定；后续 Crazyflie 硬件阶段使用独立的 hardware touchdown limits，不反向修改主仿真。

---

# 24. SafetyFilter 升级

SafetyFilter 必须按如下顺序：

```text
1. trajectory structural validity
2. vehicle dynamic feasibility
3. resource/reservation consistency
4. trajectory separation prediction
5. approval
```

---

## 24.1 新增拒绝原因

建议标准化：

```text
speed_limit
climb_rate_limit
descent_rate_limit
horizontal_acceleration_limit
vertical_acceleration_limit
lateral_acceleration_limit
curvature_limit
yaw_rate_limit
jerk_limit          # optional
dynamic_unreachable_merge_time
invalid_time_parameterization
```

---

## 24.2 Safety 检查对象必须是 executable plan

禁止：

```text
geometric path checked safe
→ another dynamics/executor modifies it
→ execute
```

必须：

```text
geometric path
→ time parameterizer
→ exact executable plan
→ SafetyFilter
→ same plan executed
```

---

# 25. Checked == Executed 不变量

必须建立强 invariant：

> **SafetyFilter 所批准的 position-time trajectory 与 environment 实际执行的 trajectory 是同一份 authoritative plan。**

建议 TrajectoryPlan 增加：

```text
plan_schema_version
vehicle_profile_id
trajectory_hash
```

commit 后执行器只允许执行该 plan，不再临时重新计算另一条轨迹。

---

# 26. Runtime Safety 的预测也必须使用同一个 vehicle model

历史 holder prediction 如果采用：

```text
constant nominal speed around ring
```

在加入 finite acceleration 后可能与实际执行不一致。

必须改为共享 helper：

```python
predict_holder_state(...)
```

由 VehicleDynamics / ring executor 提供。

SafetyFilter、runtime traffic safety、reconfiguration safety 不允许各自复制不同运动预测公式。

---

# 27. Scheduler 升级

Scheduler 的时间计算必须从几何估计升级为 dynamics-aware estimate。

建议新增：

```python
TrajectoryGenerator.estimate_minimum_duration(...)
```

或者：

```python
TrajectoryTimeParameterizer.estimate_duration(...)
```

Scheduler 使用它来判断：

```text
candidate merge time reachable?
```

---

## 27.1 Teacher 也不能绕过动力学

TeacherScheduler 提供 correction 时，所有 teacher candidate 也必须走：

```text
time parameterization
→ dynamic feasibility
→ safety
```

不能 teacher 生成物理不可执行动作再让 PPO 模仿。

---

# 28. Low-level Action 语义不变

不要因为增加 vehicle dynamics 就改变 A5-v2 action schema。

仍保持：

```text
[release_request, speed_residual, merge_time_residual]
```

其中：

- `speed_residual` 表示 desired / target traffic speed bias；
- 不是“下一秒立即变成这个速度”；
- VehicleDynamics 负责有限加速度跟踪；
- `merge_time_residual` 最终必须通过 dynamics-aware reachability。

---

# 29. Observation 与 Action：为后续 Zero-Shot 缩比迁移做尺度解耦

增加动力学后，Observation 仍遵循“只为保持 Markov 性与尺度一致性而修改”的原则，不做无边界扩维。

但为了让**同一个冻结 policy**未来能够用于 Crazyflie 缩比验证，以下要求从“建议”升级为正式接口要求。

## 29.1 动态状态

至少确认 low-level observation 中存在：

- actual horizontal speed；
- vertical speed 或等价；
- distance / time-to-merge；
- geometry-aware features；
- topology-neighbor relative state。

如果最终实现 jerk-limited acceleration 并使 `a_t` 依赖上一时刻 acceleration，则必须把 relevant acceleration state 加入 observation 并 bump observation schema version。

---

## 29.2 尽量避免网络直接依赖绝对物理尺度

优先使用类似：

```text
distance / reference_distance_or_safe_distance
relative_distance / reference_distance
speed / nominal_or_reference_speed
relative_speed / reference_speed
CPA_distance / safe_distance
TTC / reference_time_or_headway
time_to_merge / reference_time
ring_radius / reference_radius
vertical_distance / reference_height
```

这些 normalized semantics 在真实尺度 eVTOL 仿真和后续 Crazyflie 缩比验证中保持一致。

这不是为了让网络适应不同 eVTOL 型号，而是为了**同一交通策略跨物理尺度迁移**。

---

## 29.3 Action 保持 normalized intention

A5-v2 action 语义仍然：

```text
[release_request, speed_residual, merge_time_residual]
```

其中 residual 应保持 normalized，例如 `[-1,1]`，由运行 profile / scale adapter 映射到：

```text
Δv physical units
Δt physical units
```

禁止让 actor 权重直接绑定“+2 m/s”“+6 s”这样的单一绝对尺度语义。

Codex 不得静默改变 observation/action schema 并继续加载旧 checkpoint。

# 30. Policy Time Step 与 Dynamics Time Step 分离

当前 RL decision step 可以继续：

```text
policy_dt = 1.0 s
```

但对于 dynamics / safety，1 s 可能太粗。

建议配置：

```text
policy_dt
physics_dt
safety_sampling_dt
```

初始候选：

```text
policy_dt = 1.0 s
physics_dt = 0.1–0.2 s
safety_sampling_dt <= physics_dt or equal
```

具体值不能直接冻结，必须通过 computational benchmark + dt-convergence test 决定。

---

# 31. 内部 Substep 机制

一个 RL step：

```text
1. policy samples one intention
2. scheduler / plan state updated
3. environment advances N physics substeps
4. runtime safety checked across the substeps
5. aggregated reward / state returned to PPO
```

要求：

```text
policy_dt / physics_dt
```

为稳定可控整数或通过 robust time accumulation 实现。

---

# 32. 不允许重新引入“每个 substep RL 都决策”

低层研究意图仍是 traffic-level decision。

因此 physics substep 只用于：

- finite acceleration；
- trajectory execution；
- swept safety；
- better dynamics fidelity。

不是让 PPO 变成 10–20 Hz flight controller。

---

# 33. Energy Model 的处理

本阶段不要顺带彻底重写 energy model。

第一版允许保留：

```text
phase-based nominal power model
```

但建议接口允许未来：

```text
power = f(v, vz, acceleration)
```

动态升级完成后，可单独做 energy-model revision。

禁止一次修改 dynamics + reward + energy + observation 多个大语义，导致训练因果不可辨识。

---

# 34. Crazyflie Calibration：明确后移到 Hardware Validation Phase

Crazyflie 单机/双机实测 calibration **不是 A5-v2 低层训练的前置条件**。

当前阶段只要求：

- eVTOL dynamics API 与 TrajectoryPlan 对未来 hardware adapter 兼容；
- ScaleConfig 接口存在；
- policy observation/action 具备尺度解耦能力。

不要求：

- 现在测 Crazyflie 最大速度；
- 现在测 Crazyflie acceleration；
- 现在确定 Crazyflie safe distance；
- 现在用 Crazyflie profile 重新跑 C0；
- 现在因为 Crazyflie 参数修改 low-level training distribution。

Crazyflie calibration 的正确时间点：

```text
finish simulation research
→ freeze low-level / hierarchical policy weights
→ enter hardware branch
→ Crazyflie calibration
→ scale + execution + safety mapping
→ zero-shot hardware validation
```

# 35. 后续 Crazyflie Calibration 的目标（仅作为未来接口需求）

Hardware phase 中再测：

```text
v_xy_operational
accel_xy_operational
decel_xy_operational
v_climb_operational
v_descent_operational
az_operational
yaw_rate_operational
tracking_error distribution
command / communication latency
```

其目的不是修改主研究 eVTOL 模型，而是确定：

```text
HardwareVehicleProfile
ScaleConfig
hardware safety margin
trajectory timing / execution adapter
```

并验证 Crazyflie 是否能够在一个缩比、较慢、较宽松的 envelope 中复现 policy 的交通决策语义。

# 36. Crazyflie Calibration 数据与主训练严格隔离

未来 hardware calibration 建议记录：

```text
stateEstimate.x/y/z
stateEstimate.vx/vy/vz
```

如果可用再记录：

- acceleration；
- attitude；
- setpoint；
- controller output；
- battery voltage；
- timestamps。

所有 calibration script 与 raw log 纳入实验版本管理。

但是：

> **这些数据不得用于回头修改 `research_evtol.yaml`。**

它们只用于 hardware mapping / validation。

# 37. Zero-Shot Transfer 的硬边界

后续 Crazyflie validation 的目标是：

```text
same policy architecture
same observation semantics
same action semantics
same policy weights
no BC
no PPO fine-tuning
no hardware-specific RL retraining
```

允许变化的只有：

```text
ScaleConfig
HardwareVehicleProfile
trajectory timing / execution mapping
hardware safety margins
localization / communication adapter
```

如果 hardware calibration 后发现 Crazyflie 的物理能力与主仿真绝对数值差异很大，优先调整：

1. 缩比空间尺度；
2. 时间尺度；
3. hardware nominal speed；
4. trajectory duration；
5. hardware safety margin。

**不得默认通过重新训练 policy 来解决。**

如果在合理 scale/adapter 下仍无法执行，应记录为 sim-to-real / scale-transfer limitation，而不是悄悄进行 hardware-specific fine-tune 后继续称为 zero-shot transfer。

# 38. Hardware Adapter（当前只要求接口预留）

建议未来新增：

```text
terminal_marl/hardware/crazyflie_adapter.py
```

接口例如：

```python
class CrazyflieTrajectoryAdapter:
    def upload_plan(self, uav_id, plan: TrajectoryPlan): ...
    def start_plan(self, uav_id, plan_id): ...
    def stop(self, uav_id): ...
    def read_state(self, uav_id) -> HardwareState: ...
```

当前 dynamics upgrade 阶段：

- 不要求完成 multi-Crazyflie 控制；
- 不要求连接真实 Crazyflie；
- 不要求填入实测硬件参数；
- 只要求 `TrajectoryPlan` / normalized policy interface 不阻断未来 adapter。

# 39. Hardware Trajectory Output 模式（未来）

未来优先支持两个 backend：

## A. Full-state streaming

发送：

```text
position
velocity
acceleration
yaw/yaw-rate
```

适合动态生成轨迹。

## B. High-level / polynomial trajectory

将 `TrajectoryPlan` 转成 Crazyflie high-level commander / polynomial trajectory。

适合预先规划的 committed segments。

两种模式必须共享相同 upstream trajectory semantics，且不能要求重新训练 policy。

# 40. 主仿真到 Crazyflie 的 Policy Transfer 原则

目标是同一个 frozen policy 直接用于缩比 hardware：


a) 主仿真阶段：

```text
fixed research_evtol dynamics
full-scale terminal
policy training
```

b) 训练结束：

```text
freeze θ
```

c) hardware branch：

```text
Crazyflie calibration
→ ScaleConfig
→ HardwareVehicleProfile
→ scaled simulation
→ physical validation
```

整个 c) 阶段：


a) 不改变 actor weights；

b) 不改变 actor architecture；

c) 不改变 observation/action semantics。

需要尺度变化的绝对物理量由 normalized observation/action 与 scale adapter 处理。

# 41. Observation Scaling Audit 是当前 Dynamics Task 的硬要求

Codex 必须在 D0/D6 阶段生成 audit：

```text
which features are absolute?
which are normalized?
which would change under hardware scaling?
which action components are already normalized?
```

目标不是做 eVTOL performance generalization，而是保证：

> **同一 policy 在 full-scale eVTOL simulation 与 scaled Crazyflie validation 中看到语义等价的交通状态。**

如果当前 observation 大量使用绝对米/秒并会造成缩比严重 OOD，则必须在正式低层训练前完成必要的 normalization，并：

1. bump `low_obs_version`；
2. 更新 low-level training curriculum；
3. 所有旧 low-level checkpoints 标记 incompatible / legacy；
4. 重跑 observation tests。

不要在本 dynamics task 中无边界扩展 observation 内容；只处理 dynamics Markov 性和 scale-transfer 所必需的变化。

# 42. 实现 Phase D0：Repository Audit & Schema Freeze

Codex 首先完成：

1. 搜索所有 vehicle limits 使用点；
2. 搜索所有直接修改 `uav.position/velocity/speed` 的代码；
3. 搜索所有 trajectory interpolation；
4. 搜索 SafetyFilter 的 kinematic validation；
5. 搜索 scheduler duration estimate；
6. 搜索 runtime holder prediction；
7. 搜索 reconfiguration motion；
8. 搜索 telemetry / renderer 使用的 state；
9. 搜索 tests 中假设 instant speed change 的地方；
10. 审计当前 docs/config 是否已经明确“研究采用的 eVTOL 型号、性能参数、来源”；
11. 审计 low-level observation/action 的 absolute-vs-normalized semantics。

输出一个短 audit 表：

```text
file
current behavior
required change
risk
```

另外输出：

```text
Research eVTOL profile status:
- selected model name
- parameter source
- current values
- unresolved fields before C0
```

当前研究机型已经冻结为 **EHang EH216-S**。D0 audit 必须核对仓库中所有旧参数是否与本文件第 3 节冲突，并输出：

```text
EH216-S migration report:
- old value
- new canonical value
- file/location
- semantic impact
- test impact
- docs impact
```

然后再开始修改。

# 43. 实现 Phase D1：Fixed Research eVTOL Profile + Core Dynamics

主要修改：

```text
terminal_marl/core/config.py
terminal_marl/core/uav.py
terminal_marl/core/vehicle_profiles.py   # new if absent
terminal_marl/core/vehicle_dynamics.py   # new if absent
```

完成：

- `VehiclePerformanceProfile` dataclass；
- 唯一正式训练 profile：`research_evtol = EHang EH216-S`；
- 固定 `eh216s_research_v1` 参数集；
- official/certified-reference 与 engineering-model 字段分级 metadata；
- config validation；
- state helpers；
- horizontal/vertical acceleration validation；
- curvature → speed limit；
- ring feasibility helper；
- no duplicated formulas。

明确禁止：

```text
sample a different vehicle profile per episode
randomize acceleration limits during PPO
randomize max speed / climb / lateral acceleration for generalization
```

本研究当前只训练一个指定 eVTOL 模型。

# 44. 实现 Phase D2：Trajectory Representation + Time Parameterization

主要修改：

```text
terminal_marl/core/trajectory_generator.py
terminal_marl/core/trajectory_time_parameterization.py
```

完成：

- richer TrajectoryPoint；
- geometric path abstraction；
- arc-length sampling；
- speed profile；
- acceleration constraints；
- vertical constraints；
- executable trajectory generation；
- trajectory hash / schema metadata。

---

# 45. 实现 Phase D3：Phase-specific Trajectories

依次升级：

1. ring hold；
2. inbound；
3. O2M；
4. M2I；
5. I2F/final；
6. landing。

每完成一种 trajectory，立即加对应 unit tests。

不要一次改完所有 phase 后再测试。

---

# 46. 实现 Phase D4：Environment Execution Integration

主要修改：

```text
terminal_marl/envs/parallel_env.py
```

要求：

- committed plan 按 executable trajectory sample 执行；
- ring holder 使用 finite acceleration；
- `uav.velocity` 真实反映执行值；
- acceleration/yaw diagnostics 更新；
- path_length 使用实际 executed samples；
- energy 先保持既有模型；
- policy action 与 vehicle execution 严格分层。

---

# 47. 实现 Phase D5：Safety + Scheduler Integration

主要修改：

```text
terminal_marl/core/safety_filter.py
terminal_marl/core/runtime_traffic_safety.py
terminal_marl/core/merge_scheduler.py
terminal_marl/core/reconfiguration_safety.py
```

要求：

- dynamics feasibility 在 separation 前检查；
- pairwise prediction 用 executable plans；
- holder prediction 使用统一 dynamics；
- merge timing dynamics-aware；
- teacher candidates 也必须 dynamics-safe；
- reconfiguration 不得绕过 vehicle profile。

---

# 48. 实现 Phase D6：Observation / Telemetry / 3D Visualization

主要审计：

```text
terminal_marl/core/observations.py
terminal_marl/envs/parallel_env.py
3D visualizer telemetry adapter
```

新增 telemetry 建议：

```text
actual_velocity
actual_acceleration
reference_velocity
reference_acceleration
horizontal_accel
lateral_accel
vertical_rate
dynamic_limit_margin
vehicle_profile_id
trajectory_hash
dynamic_rejection_reason
```

3D 平台建议支持：

- actual trajectory；
- reference trajectory；
- acceleration vector；
- dynamically infeasible plan rejection；
- velocity profile plot。

---

# 49. 实现 Phase D7：Regression & Dynamics Qualification

在任何低层 PPO 训练前必须通过。

---

# 50. Unit Test Matrix

建议新增：

```text
tests/test_vehicle_dynamics.py
tests/test_trajectory_time_parameterization.py
tests/test_ring_hold_dynamics.py
tests/test_transfer_dynamic_feasibility.py
tests/test_final_trajectory_smoothness.py
tests/test_checked_equals_executed_dynamics.py
tests/test_dynamic_merge_reachability.py
tests/test_dynamics_dt_convergence.py
```

---

# 51. Test：Acceleration Limit

构造：

```text
v0 = low speed
v_ref = much higher
```

断言每一步：

```text
|Δv| / Δt <= a_limit
```

不能下一秒直接跳到 target speed。

---

# 52. Test：Ring Centripetal Feasibility

对多个：

```text
radius × speed
```

验证：

```text
v^2 / r <= a_lat_max
```

超限时：

- config fail；或
- speed cap；

具体策略必须全局统一。

---

# 53. Test：Transfer Curvature

对 O2M/M2I Bezier：

验证每个 sampled point：

```text
v^2 * kappa <= a_lat_max
```

并验证 acceleration continuity。

---

# 54. Test：Final Smoothness

至少检查：

```text
position continuity
velocity continuity
acceleration continuity
```

禁止 waypoint 处速度方向瞬间跳变。

---

# 55. Test：Merge Reachability

给出一个过早 merge time：

```text
reservation time < dynamic minimum arrival time
```

必须：

```text
reject / move to later feasible slot
```

不能产生超性能 trajectory。

---

# 56. Test：Checked == Executed

对 approved plan：

1. SafetyFilter 保存 trajectory hash；
2. environment 执行；
3. 每个 execution sample 与 approved plan 对应；
4. 结束后 hash / plan id 保持一致。

这是最高优先级 regression gate。

---

# 57. Test：Physics dt Convergence

至少比较：

```text
physics_dt = 0.2 s
physics_dt = 0.1 s
```

在固定 deterministic scenarios 中比较：

- trajectory endpoint；
- duration；
- minimum separation；
- energy/path length；
- merge timing。

若差异过大，继续缩小 dt。

选择正式 dt 后冻结到 config version。

---

# 58. Test：Performance Benchmark

因为 internal substeps 会增加 MARL 训练成本，必须测：

```text
env steps / second
wall-clock per 3600-s episode
N=10 / 30 / 60 scaling
```

如果性能不可接受：

优先优化：
- vectorized plan sampling；
- analytic holder motion；
- adaptive safety sampling；
- cached curvature/profile；

不要退回无动力学瞬时执行。

---

# 59. Legacy Model Mode

为了 regression，可以保留：

```text
vehicle_execution_mode = legacy_kinematic
```

但：

- 只能用于历史对比；
- 正式 A5-v2 training 默认必须使用 `performance_constrained`；
- results 必须记录 mode；
- 不允许 official checkpoint 混用两种 dynamics。

---

# 60. Schema / Checkpoint Compatibility

动力学升级会改变 MDP transition。

因此：

> **在 dynamics-v2 之前训练的 low-level checkpoint 不得作为新的正式 A5-v2 resume 起点。**

可以保留为：

```text
legacy baseline / diagnostic replay
```

如果 observation dimension 也改变，更必须禁止加载。

建议 bump：

```text
vehicle_dynamics_schema_version
trajectory_schema_version
```

如 observation 修改，再 bump：

```text
observation_schema_version
```

---

# 61. C0 必须重新做

动力学升级后：

- inbound duration 会变；
- ring acceleration response 会变；
- transfer duration 会变；
- final duration 会变；
- scheduler reachable merge slots 会变；
- capacity 会变。

因此任何旧：

```text
μ_hat
medium/high traffic definitions
```

都必须重新标定。

正确顺序：

```text
freeze EH216-S research profile (`eh216s_research_v1`)
↓
Dynamics qualification
↓
freeze observation/action normalization schema
↓
C0 capacity calibration
↓
L0/L1...
```

这里冻结的是**指定 eVTOL 主研究 profile**，与 Crazyflie calibration 无关。

# 62. 低层训练纲领中的前置 Gate

必须在 `A5V2_LOW_LEVEL_TRAINING_CURRICULUM` 最前面增加：

> **DYN0 — Research eVTOL Vehicle Dynamics Qualification**

只有满足本文件 Definition of Done，且：

```text
selected eVTOL model = EHang EH216-S
research_evtol profile = eh216s_research_v1
EH216-S parameter migration complete
terminal altitude <= 120 m AGL
observation/action scale semantics fixed
```

才允许进入 C0。

Crazyflie calibration **不是**该 gate 的组成部分。

# 63. Crazyflie Sim-to-Real Qualification（所有主仿真训练结束后）

Crazyflie branch 必须后移到 simulation policy 已冻结之后。

建议：

```text
Q_CF0: single-Crazyflie calibration
Q_CF1: derive ScaleConfig + HardwareVehicleProfile
Q_CF2: scaled single-UAV trajectory replay
Q_CF3: 2-UAV deterministic conflict-free replay
Q_CF4: small-N frozen-policy simulation replay
Q_CF5: small-N frozen-policy physical flight test
```

实机验证不能承担高密度 capacity 证明，只验证：

- trajectory executability；
- policy intention → trajectory compatibility；
- small-N coordination；
- safety interface；
- scale-transfer / sim-to-real behavior consistency。

关键：

```text
policy weights remain frozen throughout Q_CF0–Q_CF5
```

# 64. Crazyflie Hardware Qualification 必须分层

不要第一天直接运行 learned MARL。

顺序：

```text
manual/single trajectory
↓
scripted trajectory
↓
scripted multi-UAV
↓
rule policy
↓
frozen learned low-level policy
↓
(optional) frozen hierarchical policy
```

任何阶段出现执行问题，优先修：

```text
scale
trajectory timing
hardware adapter
safety margin
localization / communication
```

而不是重新训练 policy。

# 65. Safety Hardware Override

实机必须有独立 emergency safety layer，例如：

- external kill；
- geofence；
- altitude bound；
- minimum separation override；
- communication-loss behavior。

这不属于 RL policy。

实机安全距离使用 hardware-calibrated value，而不是论文仿真 80 m 的简单同比缩放。

这类 hardware-only safety calibration 不改变主 eVTOL 仿真参数，也不触发 policy retraining。

# 66. Codex 文件级修改清单

Codex 必须先按当前 repo 实际结构确认，预计涉及：

## Core

```text
terminal_marl/core/config.py
terminal_marl/core/uav.py
terminal_marl/core/trajectory_generator.py
terminal_marl/core/safety_filter.py
terminal_marl/core/runtime_traffic_safety.py
terminal_marl/core/merge_scheduler.py
terminal_marl/core/reconfiguration_safety.py
```

## New Core

```text
terminal_marl/core/vehicle_profiles.py
terminal_marl/core/vehicle_dynamics.py
terminal_marl/core/trajectory_time_parameterization.py
```

## Environment

```text
terminal_marl/envs/parallel_env.py
```

## Observation / telemetry

```text
terminal_marl/core/observations.py
visualization / telemetry modules
```

## Configs

```text
terminal_marl/configs/*
```

## Tests

```text
tests/test_vehicle_dynamics.py
tests/test_trajectory_time_parameterization.py
...
```

如果最新代码已经新增 teacher/student modules，Codex 必须保证它们全部使用新的 authoritative trajectory backend。

---

# 67. 文档必须“即时更新”，不是最后补文档

这是本任务的硬要求。

Codex 每完成一个 Phase 都必须：

```text
code
→ tests
→ docs update
→ commit / checkpoint
→ next phase
```

禁止：

```text
先改完所有代码
→ 最后凭记忆统一改 docs
```

---

# 68. 必须更新的 docs 文件

Codex 应搜索当前 `docs/`，至少更新以下已有文件（如果仍存在）：

```text
docs/requirements.md
docs/design_spec.md
docs/technical_spec.md
docs/execution_plan.md
docs/research_positioning.md
docs/revision.md
docs/PROJECT_HANDOFF_*.md
```

此外如果项目中已有：

```text
A5V2_LOW_LEVEL_TRAINING_CURRICULUM*.md
RECONFIGURATION_AND_HIGH_LEVEL_INTEGRATION_SPEC.md
```

也要检查是否需要同步。

---

## 68.1 本次 EH216-S 选型后必须同步更新的 docs 参数

Codex 在完成 D0/D1 参数迁移时，必须即时搜索 `docs/` 并修改所有与下列旧值冲突的文本：

```text
max_speed = 36
max_acceleration = 4
max_climb_rate = 5
max_descent_rate = 7

outer/middle/inner speed = 28/24/20
entry/outer/middle/inner height = 600/500/350/200

soc_reserve = 0.15
soc_emergency = 0.08
normal SOC lower bound = 0.15
```

统一改为本文件第 3 节的 EH216-S research profile 和新的 nominal terminal geometry。

至少检查：

```text
requirements.md
design_spec.md
technical_spec.md
execution_plan.md
research_positioning.md
revision.md
PROJECT_HANDOFF_*.md
A5V2_LOW_LEVEL_TRAINING_CURRICULUM*.md
RECONFIGURATION_AND_HIGH_LEVEL_INTEGRATION_SPEC.md
vehicle_dynamics_spec.md
```

其中低层训练纲领必须特别更新 SOC curriculum 和所有依赖旧 nominal speed/height 的说明。

---

# 69. requirements.md 必须更新的内容

加入硬需求：

- official execution uses performance-constrained vehicle model；
- finite acceleration；
- curvature/lateral acceleration feasibility；
- smooth final trajectory；
- dynamics-aware scheduler reachability；
- checked == executed trajectory；
- fixed research eVTOL profile/versioning；
- no vehicle-performance randomization in main training；
- hardware scaling interface；
- Crazyflie zero-shot validation boundary。

原来类似：

```text
full rotor/aerodynamic dynamics not implemented
```

可以保留，但必须改清楚为：

> 不实现 full 6-DOF / rotor aerodynamics；已实现 performance-constrained point-mass dynamics。

---

# 70. design_spec.md 必须更新

加入正式 architecture：

```text
LowLevelIntent
→ Scheduler
→ GeometricPathGenerator
→ TrajectoryTimeParameterizer
→ VehicleDynamicsValidator
→ SafetyFilter
→ VehicleExecutor
```

加入：

- data classes；
- module ownership；
- authoritative plan rule；
- hardware adapter boundary。

---

# 71. technical_spec.md 必须更新

这是最详细的技术事实源。

必须加入：

- equations；
- units；
- VehiclePerformanceProfile fields；
- `policy_dt / physics_dt`；
- ring centripetal constraint；
- curvature formula；
- acceleration-limited speed profile；
- dynamic rejection reasons；
- API signatures；
- schema version；
- config examples；
- fixed research eVTOL profile semantics；
- Crazyflie calibration is hardware-phase only；
- zero-shot policy transfer boundary。

---

# 72. execution_plan.md 必须更新

在 C0/A5-v2 训练之前插入：

```text
DYN0–DYN7 Vehicle Dynamics Upgrade
```

并明确：

```text
DYN qualification
→ C0 capacity calibration
→ low-level curriculum
```

旧的“hardware optional after simulation”表述应澄清为：

> hardware flight 与 Crazyflie calibration 均在主仿真训练之后；当前低层训练前置只包括 fixed research-eVTOL dynamics 与 future hardware-compatible normalized trajectory/policy abstraction。

---

# 73. research_positioning.md 必须更新

明确论文建模边界：

> 本研究针对一个明确指定的 eVTOL 型号，采用其固定性能参数的 performance-constrained 3D point-mass model；低层 policy 位于 terminal traffic-intention layer，不学习姿态控制；最终使用 Crazyflie 在不重新训练 policy 的条件下做缩比 small-N physical validation。

避免审稿人误解“low-level policy = low-level flight controller”。

---

# 74. revision.md 必须更新

记录：

- 为什么引入 dynamics-v2；
- 哪些旧 semantics 被废弃；
- 哪些 checkpoint / capacity result 失效；
- 新 schema version；
- 新 tests；
- C0 必须重跑。

---

# 75. PROJECT_HANDOFF 必须更新

每次本任务完成阶段后更新：

```text
current dynamics phase
passed tests
remaining blockers
current fixed research eVTOL profile
current schema versions
next executable command
```

如果旧 handoff 日期文件已经不再作为主文件，应新建最新日期版本，而不是继续把所有新事实塞进过期 handoff。

---

# 76. Low-level Training Curriculum 必须更新

至少增加：

```text
Prerequisite:
DYN0 vehicle dynamics qualification must pass before C0.
```

并修改：

- C0 使用新的 research-eVTOL dynamics；
- old `μ_hat` invalid；
- training logs 必须记录 selected eVTOL model / fixed profile version / dynamics version / physics dt；
- training 不随机化 vehicle performance parameters；
- geometry L5 使用同一 fixed eVTOL performance layer。

不要把 Crazyflie hardware control 细节混入低层训练纲领，只写训练真正需要的前置事实。

---

# 77. Reconfiguration Spec 的更新边界

如果 reconfiguration 文档存在，只更新与 vehicle performance 直接有关的接口：

- geometry transition 必须通过同一 VehicleDynamics；
- radial/vertical transition rate；
- affected UAV movement 不能 teleport；
- committed trajectory protection。

不要把本动力学文档重新塞入高层 reward 讨论。

---

# 78. 建议新增 docs/vehicle_dynamics_spec.md

Codex 完成实现后建议再生成：

```text
docs/vehicle_dynamics_spec.md
```

作为最终“实现事实”文档。

区别：

- 本文件 = upgrade plan；
- `vehicle_dynamics_spec.md` = 完成后实际 implemented model / parameters / APIs / tests。

---

# 79. Commit / Implementation 顺序

推荐：

```text
Commit D0
repo audit + docs plan alignment

Commit D1
vehicle profiles + dynamics core + unit tests + docs

Commit D2
time parameterizer + trajectory schema + tests + docs

Commit D3
ring/inbound/transfer/final dynamics + tests + docs

Commit D4
environment execution integration + tests + docs

Commit D5
scheduler/safety/runtime prediction integration + tests + docs

Commit D6
telemetry/visualizer/observation compatibility + tests + docs

Commit D7
full regression + qualification + docs/spec/handoff
```

不要把所有改变放进一个超大 commit。

---

# 80. DYN Qualification Gate

正式低层训练前至少必须满足：

## Dynamics

```text
no acceleration limit violations
no vertical-rate violations
no lateral-acceleration violations
no dynamically unreachable approved merge
```

## Smoothness

```text
no instantaneous transfer/final direction jump
C2 continuity for committed curved trajectories
```

## Safety

```text
SafetyFilter checks executable trajectory
checked plan == executed plan
collision = 0 in deterministic qualification
NMAC = 0
hard violation = 0
```

## Runtime consistency

```text
holder prediction == holder executor semantics
scheduler duration == trajectory duration semantics
```

## Numerical

```text
physics dt convergence acceptable
no NaN / zero-duration segment
```

## Performance

```text
env throughput computationally acceptable for training
```

---

# 81. Definition of Done

本 dynamics upgrade 只有满足以下条件才算完成：

- [ ] `VehiclePerformanceProfile` 成为唯一车辆性能事实源；
- [ ] 主仿真只使用一个固定 `research_evtol` profile；
- [ ] 主研究机型已固定为 `EHang EH216-S`；
- [ ] `research_evtol = eh216s_research_v1` 参数与来源已冻结；
- [ ] `25.0 / 29.17 / 36.11 m/s` 三种速度语义在代码中严格区分；
- [ ] `1.5 m/s²` horizontal/lateral operational acceleration 已成为实际执行约束；
- [ ] climb/descent 已迁移为 `4/6 m/s`；
- [ ] nominal ring speed 已迁移为 `22/20/16 m/s`；
- [ ] nominal vertical geometry 已迁移为 `gate/outer/middle/inner = 110/100/70/40 m AGL`；
- [ ] normal landing reserve 已迁移为 `SOC=0.25`，普通初始 SOC 不再使用 `0.15` 下界；
- [ ] training config 不存在 vehicle-performance domain randomization；
- [ ] Crazyflie calibration 不属于 DYN/C0/low-level training gate；
- [ ] future hardware `ScaleConfig` / adapter interface 已预留，但无需当前实测数值；
- [ ] observation/action 中会破坏缩比迁移的关键绝对尺度已完成必要 normalization；
- [ ] `TrajectoryPoint` 包含可执行的 velocity / acceleration；
- [ ] ring speed residual 不再导致瞬时速度跳变；
- [ ] ring centripetal acceleration 被检查；
- [ ] inbound 是 acceleration-feasible；
- [ ] O2M/M2I Bezier 有 curvature/time parameterization；
- [ ] final 不再是带瞬时折角的线性 waypoint trajectory；
- [ ] landing 有有限 touchdown speed；
- [ ] scheduler merge time 检查 dynamic reachability；
- [ ] SafetyFilter 先验证 dynamic feasibility；
- [ ] runtime safety 预测使用相同 dynamics；
- [ ] approved trajectory 与 executed trajectory 完全一致；
- [ ] teacher candidates 不能绕过 dynamics；
- [ ] reconfiguration movement 不能绕过 dynamics；
- [ ] policy dt 与 physics dt 可分离；
- [ ] dynamics unit tests 全部通过；
- [ ] existing regression tests 通过或有明确、合理、文档化更新；
- [ ] 3D visualizer 能显示 actual/reference/dynamic state；
- [ ] schema version 已 bump；
- [ ] pre-dynamics checkpoint 标记 legacy；
- [ ] old capacity calibration 标记 invalid；
- [ ] `docs/` 所有相关文件已即时同步；
- [ ] 最新 handoff 明确下一步是重新执行 C0；
- [ ] 不再有散落在代码中的第二套 dynamics formula。

# 82. Codex Stop Conditions

Codex 遇到以下情况必须停止当前 phase 并先修复，不允许继续向后堆代码：

### S1

同一条 trajectory：

```text
SafetyFilter checked != executor executed
```

### S2

新增 dynamics 后旧 scheduler 大量产生 unreachable merge time。

应修 scheduler，不应放宽 dynamics limit。

### S3

为了通过 tests 临时把 acceleration / curvature limit 设得极大。

禁止。

### S4

内部 substep 导致训练速度不可接受。

先 profile / optimize，再决定 physics dt；不能恢复瞬时运动。

### S5

Observation schema 因 dynamics/normalization 改变，但 trainer 仍能静默加载旧 checkpoint。

必须加 compatibility fail-fast。

### S6

Codex 发现仓库中的代码或 docs 仍把研究机型写成“未定 / generic eVTOL”，或使用与本文件第 3 节冲突的旧 canonical 参数。

停止进入 C0，先完成 EH216-S 参数迁移与 docs 同步；不得自行换成其他机型，也不得用 Crazyflie 参数代替。

### S7

任何实现试图在主训练阶段随机化 `research_evtol` 的 max speed / acceleration / climb / lateral limits 以增强 vehicle generalization。

停止并移除；这不是当前研究范围。

### S8

任何实现把 Crazyflie calibration 结果用于修改主 eVTOL profile，或提出 hardware-specific PPO/BC fine-tuning 作为默认路线。

停止；hardware branch 的默认目标是 frozen-policy zero-shot transfer。

# 83. 推荐 Codex 执行入口提示

可以将下面这段直接作为 Codex 任务说明：

```text
Read docs/CRAZYFLIE_DYNAMICS_LAYER_UPGRADE_PLAN.md first.

The main research vehicle is FIXED as the EHang EH216-S.
Use profile_id=research_evtol and profile_version=eh216s_research_v1.
Do NOT substitute another eVTOL, do NOT use Crazyflie parameters to define the
main simulation, and do NOT introduce eVTOL vehicle-performance domain randomization.

Before C0, migrate all historical canonical parameters to the values in Section 3,
including:
- max level speed 25.0 m/s (29.17 m/s never-exceed, 36.11 m/s design metadata only);
- horizontal/lateral operational acceleration 1.5 m/s^2;
- climb/descent 4/6 m/s;
- ring speeds 22/20/16 m/s;
- gate/outer/middle/inner heights 110/100/70/40 m AGL;
- normal landing SOC reserve 0.25.
Do not mechanically global-replace numeric literals; update them by semantic field.

Before editing code, audit the current repository and all docs for vehicle dynamics,
trajectory execution, scheduler duration estimates, SafetyFilter kinematics,
runtime holder prediction, reconfiguration movement, direct UAV state mutation,
and observation/action scaling semantics.

Implement the upgrade phase-by-phase (D0-D7). Maintain one authoritative
performance-constrained trajectory backend. The low-level RL action schema must
remain flight-intention [release, speed residual, merge-time residual]; do not
turn PPO into an attitude or motor controller.

The official low-level training uses only the frozen research_evtol profile.
Crazyflie calibration belongs to the later hardware-validation branch and is NOT
a prerequisite for C0 or low-level PPO training. Hardware validation must default
to the same frozen policy weights; solve mapping differences with ScaleConfig,
trajectory timing, hardware profile, adapter, and safety margins rather than
hardware-specific RL retraining.

After EACH phase:
1. run the relevant unit/regression tests;
2. update every affected markdown file under docs/ immediately;
3. record schema/config/checkpoint compatibility changes in revision/handoff;
4. only then continue to the next phase.

Do not start C0 capacity calibration or any new official low-level PPO training
until the DYN qualification gate in this document passes and the EH216-S
research_evtol profile (`eh216s_research_v1`) is frozen.
```

# 84. EH216-S 研究机型参考资料

本节用于主研究车辆参数追溯。

- EHang EH216-S official product page  
  `https://www.ehang.com/ehang216s/`  
  支持：620 kg MTOW、30 km range、130 km/h maximum design speed、UAM/multirotor product定位。

- CAAC EH216-S Special Conditions SC-21-002  
  `https://www.caac.gov.cn/XXGK/XXGK/BZGF/ZYTJHHM/202202/t20220222_211914.html`  
  支持：EH216-S 作为全电动、可载人、垂直起降、低空 UAM 运行机型的审定背景；附件同时给出 30 km 最大航程、25 min 最大航时等基础信息。

- EH216-S Type Certificate Data Sheet, Revision 0  
  正式项目归档时应保存可核验版本。当前参数迁移采用的数据包括：
  - battery 252 Ah；
  - working voltage 66.0–95.7 V；
  - landing SOC ≥ 25%；
  - maximum design speed 130 km/h；
  - never-exceed speed 105 km/h；
  - maximum level speed 90 km/h；
  - MTOW 620 kg；
  - maximum operating altitude 120 m AGL / 1000 m MSL。

> 注意：`1.5 m/s²` horizontal/lateral acceleration、`1.0 m/s²` vertical acceleration、`12 deg/s` yaw rate、`4/6 m/s` climb/descent、三环 nominal speed 与三环高度是本研究冻结的 operational/engineering values，不得伪装成上述 TCDS 的 certified numeric limits。

---

# 85. 官方 Crazyflie 参考资料（仅供后续 Hardware Validation 阶段核对）

这些资料用于未来：

- hardware adapter；
- trajectory upload / streaming；
- logging；
- Crazyflie calibration；
- physical safety layer。

**它们不用于确定主研究 eVTOL 的训练动力学参数。**

Codex 不应从博客或论坛随意抄一个“Crazyflie 最大速度”作为 operational limit；硬件阶段优先参考官方接口/硬件资料，并通过实测得到 Crazyflie 的 conservative hardware envelope。

- Bitcraze Crazyflie 2.1+ product/specifications:  
  `https://www.bitcraze.io/products/crazyflie-2-1-plus/`

- Crazyflie High Level Commander API:  
  `https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/api/cflib/crazyflie/high_level_commander/`

- Crazyflie Commander full-state/velocity setpoints:  
  `https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/api/cflib/crazyflie/commander/`

- Crazyflie controllers:  
  `https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/functional-areas/sensor-to-control/controllers/`

- Crazyflie trajectory formats / polynomial trajectory:  
  `https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/functional-areas/trajectory_formats/`

- Crazyflie logging variables / stateEstimate:  
  `https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/api/logs/`

# 86. 最终推荐主线

正式项目流程更新为：

```text
D0–D7
Performance-constrained dynamics upgrade
for EHang EH216-S
        ↓
freeze EH216-S research_evtol profile (`eh216s_research_v1`)
+ freeze normalized observation/action schema
        ↓
DYN qualification
        ↓
重新执行 C0 capacity calibration
        ↓
A5-v2 low-level curriculum
        ↓
static geometry qualification
        ↓
reconfiguration compatibility
        ↓
freeze low-level
        ↓
high-level / hierarchical training and simulation qualification
        ↓
freeze final policy weights
========================================
Hardware validation branch starts here
========================================
        ↓
Crazyflie single-vehicle calibration
        ↓
ScaleConfig + HardwareVehicleProfile
        ↓
Crazyflie-compatible scaled simulation
        ↓
small-N zero-shot physical validation
```

核心原则：

> **主研究使用一个明确指定且参数固定的 eVTOL 模型，不做车辆性能内部泛化；真实动力学约束进入 trajectory execution，但 RL 仍学习 terminal traffic intention。Crazyflie 仅在所有主仿真策略训练完成并冻结之后，用于缩比 small-N 物理验证。Crazyflie calibration 只决定 scale / execution / hardware safety mapping，不反向修改主 eVTOL 训练参数，也不默认触发 policy retraining。**

# 87. P0 Simulation Performance Refactor 最终状态（2026-08-29）

P0-A--N 已完成并通过 Performance Qualification。EH216-S
`eh216s_research_v1` envelope 未改变；`physics_dt=0.2 s` 保持冻结。

- Full Teacher future repair 改为只包含预测所需状态的 `ForecastWorld`，
  不再 `deepcopy(TerminalParallelEnv)` 或执行完整 env step。
- safety sampling 在 0.2/0.5/1.0 s 的 fixed、mixed、shield-risk 与
  400-s saturated witnesses 上收敛；选择最粗等价值 1.0 s，并保留 swept
  hard-safety verification。
- trajectory sampling 选择 Bezier 41 / Final 51；33/41 的下一级密度超过
  0.05 m dense-reference tolerance，因此未采用。
- Viewer 默认关闭；debug/live 默认 stride 5、隐藏全部轨迹、轨迹最多 20
  点。渲染线程使用 latest-only queue、丢弃旧帧并复用 UAV/LZ/ring actors。
- N=30 sustained mixed/committed/shield/Full Teacher 分别达到
  11.43/27.26/20.93/12.04 steps/s；完整回归 198 passed。

完整 before/after、top cumulative profile、收敛表和真实 PyVista 验证见：
`D:\7_codespace\training_outputs\terminal\p0_performance_refactor\final_qualified_v4\PERFORMANCE_QUALIFICATION.md`。

Performance Qualification = **PASS**。下一步是 C0 capacity calibration，
本阶段未启动 C0 或 PPO。
