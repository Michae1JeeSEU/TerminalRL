# 设计规范

## 1. 系统架构

```
┌──────────────────────────────────────────────────────────┐
│                      Trainer                             │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐ │
│  │ High-Level  │  │ Low-Level   │  │  Experience      │ │
│  │ Policy π_H  │  │ Policy π_L  │  │  Replay Buffer   │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────────────┘ │
│         │                │                                │
│         ▼                ▼                                │
│  ┌──────────────────────────────────────────────────┐    │
│  │              TerminalAreaEnv (Gym)               │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │    │
│  │  │RingAir-  │ │UAV       │ │ConflictDetector  │ │    │
│  │  │space     │ │Manager   │ │                  │ │    │
│  │  └──────────┘ └────┬─────┘ └──────────────────┘ │    │
│  │  ┌──────────┐ ┌────┴─────┐ ┌──────────────────┐ │    │
│  │  │Priority  │ │Safety    │ │Metrics           │ │    │
│  │  │Engine    │ │Controller│ │Collector         │ │    │
│  │  └──────────┘ └──────────┘ └──────────────────┘ │    │
│  └──────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

## 2. 模块职责

### 2.1 RingAirspace（空域模型）

**职责**：管理三层环结构的状态与合法性。

```
RingAirspace
├── 属性
│   ├── radii: tuple[float, float, float]    # R1, R2, R3
│   ├── heights: tuple[float, float, float]   # H1, H2, H3
│   └── speeds: tuple[float, float, float]    # V1, V2, V3
├── 方法
│   ├── update(r, h, v) -> bool              # 更新参数，返回是否合法
│   ├── get_ring(position) -> int            # 根据位置判断所在环 (1/2/3, 0=外)
│   ├── get_ring_orbit(ring_id) -> (r, h)    # 返回环的轨道参数
│   ├── get_entry_points() -> np.ndarray     # 4 个固定进入点 (4,3)
│   ├── random_position_on_ring(ring_id)     # 环轨道上随机角度位置
│   ├── is_valid() -> bool                   # 检查硬约束 R1>R2>R3, H1>H2>H3
│   ├── get_boundary(ring_id) -> tuple       # 获取环的几何边界
│   └── state_vector -> np.ndarray (9,)      # [R1,R2,R3,H1,H2,H3,V1,V2,V3]
```

**硬约束校验**：
```python
def _validate(self) -> bool:
    return (self.r1 > self.r2 > self.r3 and
            self.h1 > self.h2 > self.h3 and
            all(r > 0 for r in self.radii) and
            all(h > 0 for h in self.heights))
```

### 2.2 UAV（无人机模型）

**职责**：单个 UAV 的状态、动力学与能耗模型。

```
UAV
├── 属性
│   ├── id: int
│   ├── position: np.ndarray (3,)      # x, y, z
│   ├── velocity: np.ndarray (3,)      # vx, vy, vz
│   ├── ring_id: int                   # 当前所在环 (1/2/3)
│   ├── battery: float                 # 剩余电量
│   ├── base_priority: int            # 基础优先级 (1/3/5)
│   ├── mission_type: int             # 任务类型 (1-4)，影响 LZ 占用
│   ├── wait_time: float              # 累计等待时间
│   ├── ring_entry_time: float        # 进入当前环时的 wait_time
│   ├── status: Enum[CRUISING, DESCENDING, LANDED]  # 无 HOVER
│   ├── landing_zone: int             # 选定的着陆区 (-1=无)
│   ├── is_controlled: bool           # 是否受策略控制
│   └── _landing_target: np.ndarray | None  # 降落点坐标 (x,y,0)
├── 方法
│   ├── step(action, dt) -> None       # 按动作推进一步
│   ├── can_descend() -> bool         # 环驻留 + 电量检查
│   ├── compute_priority() -> float   # 计算当前动态优先级
│   └── reset() -> None
```

**运动学模型**：
- **盘旋 (CIRCLE)**：锁定在环轨道 (R_i, H_i) 上做圆周运动，角速度 ω = V_i / R_i。每步强制吸附到轨道（用 ring_radius/ring_height 覆盖位置），消除漂移
- **下降 (DESCEND)**：速度向量 v = [V·cosγ·cosψ, V·cosγ·sinψ, V·sinγ]。γ 为负 = 下降。到达地面 (z ≤ 1.0) 自动着陆

**能耗模型**：
```
E_cruise = P_cruise × dt      # 巡航功率 × 时间
E_descend = P_descend × dt    # 下降功率 × 时间
```
背景 UAV（is_controlled=False）不消耗电量。

### 2.3 UAVManager（无人机管理器）

**职责**：管理全部 UAV 实例的创建、销毁与批量操作。

**初始化分布（无流入，Phase 3b）**：
- UAV 分布在三个环的轨道上：环1 60%、环2 25%、环3 15%
- 每架 UAV 随机角度、随机电量（critical~max）、随机等待时间（0~600s）、随机优先级（1/3/5）和随机任务类型（1-4，按权重）
- **初始间距保证**：`_find_safe_angle()` 通过拒绝采样（200 次尝试）+ 最大间隙回退策略，确保新生成 UAV 与同环已有 UAV 的弧间距 ≥ `safe_separation`

**流入调度（Phase 4+，uav_inflow_rate > 0）**：
- 按时钟累积 dt，到达 inflow_interval = 3600/uav_inflow_rate 时生成一架
- 新 UAV 从 4 个 entry_point（环1 0°/90°/180°/270°）随机选一进入
- 超过 max_uavs 时暂停生成

**碰撞处理**：碰撞 UAV 被标记为 `battery=0`（`is_dead=True`），释放 LZ 预留，下一 `remove_landed()` 移除。**不再终止 episode**。

**环间分离控制**：原 `speed_adjustments` 逻辑已提取到 SafetyController Rule 1。

```
UAVManager
├── uavs: list[UAV]
├── landed_count: int              # 累计着陆数
├── failed_count: int              # 累计失败数（含碰撞死亡）
├── _INIT_RING_PROBS: (0.60, 0.25, 0.15)
├── _find_safe_angle(ring_id) -> float  # 拒绝采样确保初始间距 ≥ safe_separation
├── _spawn_one(ring_id, mission_type, ensure_separation, ...) -> UAV
├── remove_landed() -> None        # 移除 LANDED + is_dead (battery=0)
├── try_spawn_inflow(dt) -> None   # 按时钟调度流入
├── step_all(actions, dt, safety_controller) -> None
├── get_state_matrix() -> np.ndarray  # (N, 6) [x,y,z,battery,priority,angle]
├── ring_counts -> (c1, c2, c3)
└── avg_wait_time -> float
```

### 2.4 SafetyController（安全控制器）★ 新增 Phase 3b.5

**职责**：反射式碰撞避免层，在 RL 动作与 UAV 运动学之间介入。只做减速/微调航向，不做路径规划。

**设计原则**：
- 反射层介入在 RL 动作执行前：`RL action → SafetyController → UAV.step()`
- 紧急 UAV（priority≥5 或 battery<critical×max_battery）**永不减速**
- 低优先级给高优先级让路；同等优先级双方各让半步

**Plan B 架构**：Safety 调整量编码为 +4 维观测输入（speed_mult, psi_offset, gamma_offset, blocked），RL 策略学会**预见**安全干预，而非被动接受。

```
SafetyController
├── config: SafetyConfig
├── evaluate(uavs, actions) → dict[uav_id, SafetyAdjustment]
│   ├── _classify_priority(u1, u2) → (higher, lower)
│   ├── _rule1_angular_separation(uavs)         # 同环角间距减速
│   ├── _rule2_circling_yields_descending(uavs)  # 盘旋让下降
│   ├── _rule3_dual_descent_divergence(uavs)     # 双下降航向偏转
│   ├── _rule4_universal_brake(uavs)            # 通用近距刹车
│   ├── _rule5_emergency_override(uavs)          # 紧急 UAV 豁免
│   └── _compose_adjustments(per_rule_results)   # 取最保守值
├── apply(actions, adjustments) → dict[uav_id, LowLevelAction]
└── compose(adjustments) → SafetyAdjustment      # 组合多规则
```

**5 条安全规则**：

| 规则 | 触发条件 | 动作 | 优先权 |
|------|---------|------|--------|
| R1: 同环角间距 | 同环两架 UAV 角间距 < 50m | 跟随者减速至 0.85-0.98x | 后方减速 |
| R2: 盘旋让下降 | 下降 UAV 穿越环的高度层 | 盘旋者减速至 0.85x | 下降者优先 |
| R3: 双下降发散 | 两架下降中，3D 距离 < 50m 且接近中 | 各偏转 psi ±15° | 双方各半 |
| R4: 通用刹车 | 任意两架 3D 距离 < 30m | closing speed > 0 方减速至 0.7x | 低优先更激进 |
| R5: 紧急覆盖 | priority ≥ 5 或 battery < 15% | 紧急 UAV 不受影响，非紧急让其路 | 紧急绝对优先 |

**SafetyAdjustment 数据结构**：
```python
@dataclass
class SafetyAdjustment:
    speed_mult: float = 1.0      # 速度系数 [0.6, 1.3]
    psi_offset: float = 0.0      # 航向偏转 (rad)
    gamma_offset: float = 0.0    # 航迹角偏转 (rad)
    blocked: bool = False        # 是否强制 CIRCLE
    active_rules: list[str] = [] # 触发的规则名列表
```

**组合逻辑**：多个规则同时触发时，取最小 `speed_mult`、最大 `|psi_offset|`、OR 的 `blocked`。

**集成方式**（[terminal_area_env.py](src/env/terminal_area_env.py) `step()`）：
```python
sc = self.safety_controller if self.config.safety_enabled else None
self.uav_manager.step_all(uav_actions, dt, sc)
```

```python
# UAVManager.step_all 内部：
safety_adjustments = safety_controller.evaluate(uavs, actions)
adjusted_actions = safety_controller.apply(actions, safety_adjustments)
# 再用 adjusted_actions 调用 uav.step()
```

### 2.5 SafetyConfig（安全配置）★ 新增

```python
@dataclass
class SafetyConfig:
    # Rule 1: Same-ring angular separation
    angular_separation_threshold: float = 50.0     # 触发距离 (m)
    # Rule 2: Circling yields to descending
    descend_crossing_lookahead: float = 10.0       # 预测窗口 (s)
    descend_crossing_slowdown: float = 0.85        # 盘旋者减速系数
    # Rule 3: Dual descent divergence
    dual_descend_threshold: float = 50.0           # 触发距离 (m)
    dual_descend_psi_offset: float = 15.0          # 偏转角 (deg)
    # Rule 4: Universal brake
    universal_brake_threshold: float = 30.0        # 触发距离 (m)
    universal_brake_low: float = 0.70              # 低优先减速系数
    universal_brake_high: float = 0.85             # 高优先减速系数
    # Rule 5: Emergency override
    emergency_yield_slowdown: float = 0.60         # 对紧急 UAV 让路减速
    emergency_yield_psi: float = 25.0              # 对紧急 UAV 让路偏转 (deg)
```

### 2.6 ConflictDetector（冲突检测器）

**职责**：检测 UAV 间安全冲突，提供惩罚信号。

```
ConflictDetector
├── safe_distance: float
├── check_pairwise(uav1, uav2) -> bool  # 检测两架 UAV 冲突
├── check_all(uavs) -> list[tuple]      # 返回所有冲突对
└── compute_penalty(conflicts) -> float # 冲突惩罚项
```

### 2.7 PriorityEngine（优先级引擎）

**职责**：计算每架 UAV 的动态优先级。

```
PriorityEngine
├── compute(uav, current_time) -> float
│   # priority = base × f(wait) × g(battery)
│   # f(wait) = 1 + α × (wait / max_wait)^β
│   # g(battery) = 1 + γ × (1 - battery / critical)^δ
└── get_sorting_key(uavs) -> list[UAV]  # 按优先级排序
```

### 2.8 MetricsCollector（指标收集器）

**职责**：收集并汇总每一步的性能指标。

```
MetricsCollector
├── episode_energy: float
├── episode_wait_times: list[float]
├── episode_throughput: float
├── step(env) -> None
├── get_episode_summary() -> dict
└── reset() -> None
```

### 2.9 TerminalAreaEnv（Gym 环境）

**职责**：环境主循环，组合上述模块。

```
TerminalAreaEnv
├── airspace: RingAirspace
├── uav_manager: UAVManager
├── conflict_detector: ConflictDetector
├── priority_engine: PriorityEngine
├── safety_controller: SafetyController  ★ 新增
├── metrics: MetricsCollector
├── landing_zones: np.ndarray (M, 3)      # LZ 坐标
├── lz_occupied_until: np.ndarray (M,)    # 各 LZ 剩余占用时间
├── _lz_reserved_by: dict[lz_idx→uav_id] # LZ 预留状态
├── _local_obs_dim: int = 75 (+4 safety)  # per-UAV 观测维度 (79 with safety)
│
├── reset() -> obs, info
│   └── 重置所有模块 + LZ 状态清零
├── step(action) -> obs, reward, terminated, truncated, info
│   ├── 每 high_level_interval 步：应用高层 action → airspace.update()
│   ├── _parse_low_level_actions()：约束强制执行（LZ 占用/环驻留/轨道吸附）
│   ├── SafetyController.evaluate() → 安全调整量  ★ 新增
│   ├── SafetyController.apply() → 修正后的动作       ★ 新增
│   ├── uav_manager.step_all()：推进所有 UAV
│   ├── 更新 LZ 占用计时器 + 清理失效预留
│   ├── remove_landed() → 移除 LANDED + is_dead
│   ├── try_spawn_inflow()（Phase 4+）
│   ├── _compute_reward() → 碰撞惩罚（不终止）        ★ 修改
│   └── _get_obs()：per-UAV 局部观测矩阵（含 safety 信号）
└── _get_info() -> dict
```

**关键机制**：

1. **SafetyController 集成**（★ 新增）：`safety_controller.evaluate()` 在 RL 动作后、UAV 执行前介入，根据 5 条规则调整速度/航向/航迹角。调整后的观测包含 4 维 safety 信号（speed_mult / psi_offset / gamma_offset / blocked）

2. **安全轨道吸附**（★ 修改）：`_safe_orbit_snap()` 在将 UAV 吸附回环轨道前，搜索 ±30°（步长 5°）找安全槽位。找到空位后仅修正 z 高度，避免 snap 与其他 UAV 碰撞

3. **环容量限制**（★ 新增）：`_ring_has_capacity()` 计算环容量 = 2πR / min_ring_spacing（默认 100m）。当环上 UAV 数 ≥ 90% 容量时，阻止非紧急 UAV 下降进入该环。紧急（priority≥5）或临界电量（<15%）豁免

4. **LZ 预留与占用**：
   - UAV 环3+DESCEND+LZ 选择 → 检查 LZ 是否被占用/预留 → 阻塞则强制 CIRCLE
   - 通过检查 → 预留 LZ（_lz_reserved_by）+ 设置 _landing_target
   - 着陆后 → 预留转占用计时器（lz_occupied_until = mission_type 对应时长）
   - UAV 碰撞死亡/移除/中止下降 → 自动释放预留

5. **碰撞处理**（★ 修改）：碰撞 UAV 被标记 `battery=0`（`is_dead=True`），LZ 预留自动释放，`failed_count` 计入，下一 `remove_landed()` 移除。Episode **不终止**，其他 UAV 继续飞行

6. **环驻留约束**：`can_descend()` 检查 ring_residency ≥ min_ring_wait × priority_factor。base_priority=5（紧急）或电量 < 15% 时完全豁免

## 3. 奖励函数设计（实际实现）

```python
def _compute_reward(self) -> float:
    # 能耗惩罚（每步增量功率，非累计能量）
    step_power = sum(power_map[u.status] * dt for u in uavs)
    R_energy = -w_energy * step_power / n  # w_energy=0.3

    # 等待时间惩罚（归一化到小时）
    R_wait = -w_wait * avg_wait_time / 3600.0  # w_wait=1.0

    # 吞吐量奖励（每新增着陆 +20/n 的立即奖励）
    new_landings = landed_count - prev_landed
    R_throughput = w_throughput * 20.0 * new_landings / n  # w_throughput=2.0

    # 安全冲突惩罚（两层级：碰撞 + 接近）
    conflicts = conflict_detector.check_all(uavs)  # 碰撞 (<10m)
    near_misses = conflict_detector.check_near_misses(uavs)  # 近距预警 (<50m)
    R_safety = -w_safety * len(conflicts) * penalty_conflict  # w_safety=5.0
    R_safety -= w_near * len(near_misses) * penalty_near_miss  # w_near=0.5

    # ★ 距离接近惩罚（连续梯度，替代二元碰撞惩罚）
    R_proximity = 0.0
    active_uavs = [u for u in uavs if u.status != UAVStatus.LANDED]
    for i in range(len(active_uavs)):
        for j in range(i + 1, len(active_uavs)):
            dist = float(np.linalg.norm(active_uavs[i].position - active_uavs[j].position))
            if dist < proximity_penalty_threshold:  # 30m
                R_proximity -= proximity_penalty_scale / max(dist, 0.1)  # 5.0/dist

    # ★ 碰撞惩罚（碰撞后移除 UAV，不终止 episode）
    if len(collisions) > 0:
        collision_penalty = -(penalty_conflict + penalty_conflict * 0.5 * len(collision_ids))
        # penalty_conflict = 500, 一对碰撞 = -750, 两对 = -1000

    # 优先级违反惩罚
    R_priority = -w_priority * violations  # w_priority=2.0

    # 电量耗尽惩罚（仅未着陆的死 UAV）
    dead = sum(1 for u in uavs if u.is_dead)
    R_battery = -w_battery * dead  # w_battery=10.0

    # 密集进度奖励：仅对 DESCENDING 状态的 UAV 给予位置进度奖励
    R_progress = 0.0
    for u in uavs:
        if u.status == UAVStatus.DESCENDING:
            radial_progress = 1.0 - min(r / R1, 1.0)
            vertical_progress = 1.0 - min(z / H1, 1.0)
            R_progress += 2.0 * (radial_progress + vertical_progress)
    R_progress /= n

    return R_energy + R_wait + R_throughput + R_safety + R_proximity
           + R_priority + R_battery + R_progress
```

**设计要点**：
- 能耗使用每步增量功率而非累计值，避免奖励随 step 线性增长
- 着陆奖励为 spike 奖励（仅新增着陆时给），避免重复计数
- 进度奖励仅 DESCENDING 时生效：强制 agent 持续下降而非找到好位置后悬停
- ★ **接近惩罚**：连续梯度 `-scale/distance`，在 30m 阈值内提供平滑的分离信号，RL 可以从密度中直接学习避让
- ★ **碰撞惩罚**：碰撞 UAV 被移除并施加 `-(500 + 250×N_colliding)` 惩罚，episode 继续

## 4. Options 框架实现设计

```python
class Option:
    """高层 option = 一组空域构型参数"""
    def __init__(self, radii, heights, speeds):
        self.radii = radii
        self.heights = heights
        self.speeds = speeds
        self.duration = 1800  # 30 min

class HighLevelPolicy:
    """选 option → 应用到空域"""
    def select_option(self, obs) -> Option: ...

class LowLevelPolicy:
    """在 option 约束下控制各 UAV"""
    def select_action(self, obs, option) -> dict[int, LowLevelAction]: ...
```

## 5. 接口约定

### 5.1 高低层通信

```
高层 → 低层: Option(radii, heights, speeds)
低层 → 高层: 累计统计数据 (avg_wait, throughput, conflicts, landings)
```

### 5.2 环境 → 外部

```
每个 episode 结束时产出:
{
    "episode": int,
    "total_landed": int,
    "total_failed": int,
    "avg_wait_time": float,
    "avg_energy": float,
    "throughput": float,
    "safety_violations": int,
    "avg_priority_score": float,
}
```
