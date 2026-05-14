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
│  │  └──────────┘ └──────────┘ └──────────────────┘ │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │    │
│  │  │Priority  │ │Metrics   │ │Renderer          │ │    │
│  │  │Engine    │ │Collector │ │(optional)        │ │    │
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
│   ├── get_ring(position) -> int            # 根据位置判断所在环
│   ├── is_valid() -> bool                   # 检查硬约束
│   └── get_boundary(ring_id) -> tuple       # 获取环的几何边界
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
│   ├── battery: float                 # 剩余电量 [0, 1]
│   ├── base_priority: int            # 基础优先级
│   ├── wait_time: float              # 累计等待时间
│   ├── status: Enum[CRUISING, HOLDING, DESCENDING, LANDED]
│   └── target: np.ndarray | None     # 降落目标区域
├── 方法
│   ├── step(action, dt) -> None       # 按动作推进一步
│   ├── compute_energy(dt) -> float   # 计算该步能耗
│   ├── get_priority() -> float       # 计算当前优先级
│   └── reset() -> None
```

**运动学模型（简化）**：
- 盘旋：沿当前环做圆周运动，v = ring_speed
- 悬停：位置不变，v = 0
- 下降：沿路径点序列插值，速度 = descent_speed

**能耗模型（简化）**：
```
E_hover = P_hover × dt              # 悬停功率 × 时间
E_cruise = P_cruise × dt            # 巡航功率 × 时间
E_descend = P_descend × dt          # 下降功率 × 时间
```

### 2.3 UAVManager（无人机管理器）

**职责**：管理全部 UAV 实例的创建、销毁与批量操作。初始 UAV 按比例分布到三个环中（~50%/30%/20%），每架 UAV 的初始位置（环内体积均匀随机）、电量（critical~max 均匀随机）、等待时间（0~600s 均匀随机）、状态（按环的概率分布分配）均为随机。替换生成在环1边界，状态同样随机化。

```
UAVManager
├── uavs: list[UAV]
├── landed_count: int
├── failed_count: int
├── _INIT_RING_PROBS: (0.5, 0.3, 0.2)   # 初始环分布比例
├── _INIT_STATUS_MAP: dict               # 各环状态概率分布
├── _spawn_one(ring_id, battery, wait_time, status) -> UAV
├── remove_landed() -> None              # 移除已着陆/失败 UAV + 替换生成
├── get_uavs_by_ring(ring_id) -> list[UAV]
├── get_state_matrix() -> np.ndarray     # 所有 UAV 状态矩阵 (N,5)
└── step_all(actions, dt) -> None
```

### 2.4 ConflictDetector（冲突检测器）

**职责**：检测 UAV 间安全冲突，提供惩罚信号。

```
ConflictDetector
├── safe_distance: float
├── check_pairwise(uav1, uav2) -> bool  # 检测两架 UAV 冲突
├── check_all(uavs) -> list[tuple]      # 返回所有冲突对
└── compute_penalty(conflicts) -> float # 冲突惩罚项
```

### 2.5 PriorityEngine（优先级引擎）

**职责**：计算每架 UAV 的动态优先级。

```
PriorityEngine
├── compute(uav, current_time) -> float
│   # priority = base × f(wait) × g(battery)
│   # f(wait) = 1 + α × (wait / max_wait)^β
│   # g(battery) = 1 + γ × (1 - battery / critical)^δ
└── get_sorting_key(uavs) -> list[UAV]  # 按优先级排序
```

### 2.6 MetricsCollector（指标收集器）

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

### 2.7 TerminalAreaEnv（Gym 环境）

**职责**：环境主循环，组合上述模块。

```
TerminalAreaEnv
├── ring_airspace: RingAirspace
├── uav_manager: UAVManager
├── conflict_detector: ConflictDetector
├── priority_engine: PriorityEngine
├── metrics: MetricsCollector
├── step_count: int
├── high_level_interval: int           # 高层调用间隔（以 step 计）
│
├── reset() -> obs, info
├── step(action) -> obs, reward, terminated, truncated, info
│   ├── 每 high_level_interval 步：应用高层 action → 更新空域
│   └── 每步：应用低层 action → 推进 UAV 状态
├── _compute_reward() -> float
│   # R = w1×R_energy + w2×R_wait + w3×R_throughput
│   #   + w4×R_safety + w5×R_priority + w6×R_battery
└── render() -> None
```

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

    # 安全冲突惩罚
    conflicts = conflict_detector.check_all(uavs)
    R_safety = -w_safety * len(conflicts) * penalty_conflict  # w_safety=5.0

    # 优先级违反惩罚
    R_priority = -w_priority * violations  # w_priority=2.0

    # 电量耗尽惩罚（仅未着陆的死 UAV）
    dead = sum(1 for u in uavs if u.is_dead)
    R_battery = -w_battery * dead  # w_battery=10.0

    # 密集进度奖励：仅对 DESCENDING 状态的 UAV 给予位置进度奖励
    # 关键设计：进度奖励与动作状态绑定，防止 agent 学会"下降一段后悬停"
    R_progress = 0.0
    for u in uavs:
        if u.status == UAVStatus.DESCENDING:
            radial_progress = 1.0 - min(r / R1, 1.0)
            vertical_progress = 1.0 - min(z / H1, 1.0)
            R_progress += 2.0 * (radial_progress + vertical_progress)
    R_progress /= n

    return R_energy + R_wait + R_throughput + R_safety + R_priority + R_battery + R_progress
```

**设计要点**：
- 能耗使用每步增量功率而非累计值，避免奖励随 step 线性增长
- 着陆奖励为 spike 奖励（仅新增着陆时给），避免重复计数
- 进度奖励仅 DESCENDING 时生效：强制 agent 持续下降而非找到好位置后悬停

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
